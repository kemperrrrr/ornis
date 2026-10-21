//! `syn` → IR lowering: every supported AST shape becomes structural
//! nodes; nothing here renders WGSL text (the only strings produced are
//! lexical token mappings — literal suffixes, identifier spellings — and
//! `syn::Error::to_compile_error` output for truly unsupported nodes,
//! which carries its message, not WGSL structure).
//!
//! Loud-failure contract (mirrors the historical translator one-to-one):
//! anything untranslatable lowers to [`IrExpr::Unsupported`] or a
//! [`IrExpr::Verbatim`] marker call, which naga rejects — code never
//! vanishes silently.

use ornis_shader_lang::ShaderBuiltin;
use ornis_shader_lang::ShaderType;
use ornis_shader_lang::ir::{
    IrBinOp, IrBlock, IrCallee, IrElse, IrExpr, IrStmt, IrType, IrUnOp, UNSUPPORTED_MARKER,
};
use ornis_shader_lang::writer::{option_struct_name, print_ty};
use syn::spanned::Spanned;
use syn::{Block, ExprForLoop, Local, Pat, Stmt};

/// The subset of Rust patterns a `#[kernel]` `match` / `if let` can
/// spell in WGSL.
pub(crate) enum ArmPattern {
    /// `_` — matches anything, binds nothing.
    Wild,
    /// A bare identifier — binds the whole remaining value (must be the
    /// last arm / the only `if let` branch).
    Bind(String),
    /// A literal — compared with `==`.
    Lit,
    /// `Some(x)` — discriminant test plus a payload binding.
    Some(String),
    /// `None` — discriminant test only.
    None,
}

/// Fall-through message for a pattern shape the lowerer does not model:
/// every unsupported pattern reports exactly this text.
const UNSUPPORTED_PATTERN: &str = "this pattern is not supported in a `#[kernel]` `match`";

/// Classify one `match`/`if let` pattern into the supported subset.
/// `Err` carries a message suitable for a spanned macro-time error.
///
/// Every arm delegates to a helper, so this reads as routing rather
/// than branching: the shape decisions live in the `classify_*` functions
/// below, each of which is either pure logic or pure delegation.
pub(crate) fn classify_pattern(pat: &Pat) -> Result<ArmPattern, String> {
    match pat {
        Pat::Wild(_) => Ok(ArmPattern::Wild),
        Pat::Ident(pi) => classify_ident_pattern(pi),
        Pat::Lit(_) => Ok(ArmPattern::Lit),
        Pat::Path(p) => classify_path_pattern(&p.path),
        Pat::TupleStruct(ts) => classify_some_pattern(ts),
        Pat::Paren(pp) => classify_pattern(&pp.pat),
        Pat::Guard(pg) => Err(format!(
            "match guards (`pat if cond`) have no WGSL spelling in `#[kernel]` (on `{}`); \
             restructure with an explicit `if` inside the arm body, or split into an extra arm",
            quote::ToTokens::to_token_stream(&pg.guard)
        )),
        Pat::Range(_) => Err(
            "range patterns have no WGSL spelling in `#[kernel]`; compare with explicit \
             literal arms instead"
                .to_string(),
        ),
        Pat::Or(_) => Err(
            "or-patterns (`a | b`) have no WGSL spelling in `#[kernel]`; split them into \
             separate arms"
                .to_string(),
        ),
        Pat::Struct(_) => Err(
            "struct patterns have no WGSL spelling in `#[kernel]`; match on scalars or \
             `Option` payloads instead"
                .to_string(),
        ),
        Pat::Tuple(_) => Err(
            "tuple patterns have no WGSL spelling in `#[kernel]`; WGSL has no tuples".to_string(),
        ),
        _ => Err(UNSUPPORTED_PATTERN.to_string()),
    }
}

/// `Pat::Ident`: the bare `None`/`Some` discriminants (syn 3 parses both
/// as identifier patterns) and the plain binding case. `x @ subpattern`
/// and `ref` bindings have no WGSL spelling and fall through loudly.
fn classify_ident_pattern(pi: &syn::PatIdent) -> Result<ArmPattern, String> {
    if pi.subpat.is_some() || pi.by_ref.is_some() {
        return Err(UNSUPPORTED_PATTERN.to_string());
    }
    // syn 3 parses the bare `None` path pattern as an identifier
    // pattern; it is a discriminant test, not a binding.
    if pi.ident == "None" {
        return Ok(ArmPattern::None);
    }
    if pi.ident == "Some" {
        return Err("`Some` in a pattern must bind a payload: `Some(x)`".to_string());
    }
    Ok(ArmPattern::Bind(pi.ident.to_string()))
}

/// `Pat::Path`: only the `None` discriminant (bare, or through
/// `Option`/`core::option`). Every other path names an enum variant,
/// which has no WGSL spelling.
fn classify_path_pattern(path: &syn::Path) -> Result<ArmPattern, String> {
    if path.segments.last().is_some_and(|s| s.ident == "None") {
        return Ok(ArmPattern::None);
    }
    Err(
        "enum-variant paths have no WGSL spelling in `#[kernel]`; fieldless enums lower to \
         a `u32` tag — match on the integer tag with literal arms instead"
            .to_string(),
    )
}

/// `Pat::TupleStruct`: only `Some(binding)`, binding exactly one plain
/// identifier. Any other tuple-struct pattern is unsupported.
fn classify_some_pattern(ts: &syn::PatTupleStruct) -> Result<ArmPattern, String> {
    if !ts.path.segments.last().is_some_and(|s| s.ident == "Some") {
        return Err(UNSUPPORTED_PATTERN.to_string());
    }
    if ts.elems.len() != 1 {
        return Err("`Some(...)` in a pattern binds exactly one payload".to_string());
    }
    let Pat::Ident(pi) = &ts.elems[0] else {
        return Err(
            "`Some(...)` in a pattern must bind a plain identifier, not a nested pattern"
                .to_string(),
        );
    };
    if pi.subpat.is_some() || pi.by_ref.is_some() {
        return Err(
            "`Some(...)` in a pattern must bind a plain identifier, not a nested pattern"
                .to_string(),
        );
    }
    Ok(ArmPattern::Some(pi.ident.to_string()))
}

thread_local! {
    /// The enclosing kernel's return type, in scope for the whole of one
    /// `lower_body_typed` call. `return None` inside an `if` block has
    /// no local type context; this is the only place it can come from.
    static RETURN_TY: std::cell::RefCell<Option<IrType>> =
        const { std::cell::RefCell::new(None) };
}

/// Set the return type for the duration of `f`, restoring the previous
/// value afterwards so nested bodies do not leak into each other.
fn with_return_ty<T>(ty: Option<&IrType>, f: impl FnOnce() -> T) -> T {
    // A nested body with no context of its own (an `if` block, a
    // statement-position `match`) keeps the enclosing kernel's return
    // type: `None` here means "not given", not "returns nothing".
    let Some(ty) = ty else {
        return f();
    };
    let prev = RETURN_TY.with(|cell| cell.borrow_mut().take());
    RETURN_TY.with(|cell| *cell.borrow_mut() = Some(ty.clone()));
    let out = f();
    RETURN_TY.with(|cell| *cell.borrow_mut() = prev);
    out
}

fn current_return_ty() -> Option<IrType> {
    RETURN_TY.with(|c| c.borrow().clone())
}

/// The type context a `return` expression and the `Some`/`None`
/// constructors should use: local propagation first, then the
/// function's return type.
fn expected_or_return(expected: Option<&IrType>) -> Option<IrType> {
    expected.cloned().or_else(current_return_ty)
}

/// A loud placeholder: any message that must survive to a naga parse
/// error rather than being dropped at macro time.
fn loud(msg: &str, span: proc_macro2::Span) -> IrExpr {
    IrExpr::Verbatim(
        syn::Error::new(span, format!("{UNSUPPORTED_MARKER}: {msg}"))
            .to_compile_error()
            .to_string(),
    )
}

/// Lower one expression with no type context.
pub fn lower_expr(e: &syn::Expr) -> IrExpr {
    lower_expr_expected(e, None)
}

/// Lower one expression, propagating the expected type so `Some`/`None`
/// know which generated `Option_<T>` struct to construct.
///
/// Every arm is a delegation call, so this match routes rather than
/// branches (IOSP Integration); each shape builds in its own helper.
fn lower_expr_expected(e: &syn::Expr, expected: Option<&IrType>) -> IrExpr {
    use syn::Expr::*;
    match e {
        Lit(l) => lower_lit(l),
        Path(p) => lower_path(p, expected),
        Field(f) => lower_field(f),
        Binary(b) => lower_binary(b),
        Unary(u) => lower_unary(u),
        Paren(p) => IrExpr::Paren(Box::new(lower_expr_expected(&p.expr, expected))),
        Call(c) => lower_call(c, expected),
        MethodCall(m) => lower_method_call(m),
        If(i) => lower_if(i, expected),
        Let(_) => loud("a bare `let` expression has no WGSL spelling", e.span()),
        Match(m) => lower_match_expr(m),
        Block(b) => IrExpr::Block(lower_block(&b.block)),
        Assign(a) => lower_assign(a),
        Index(ix) => lower_index(ix),
        Struct(s) => lower_struct_lit(s),
        Continue(c) => lower_continue(c),
        Break(b) => lower_break(b),
        // Rust casts lower to WGSL conversion constructors (`x as u32`
        // → `u32(x)`); anything without a WGSL scalar spelling is a
        // compile error, never a silently dropped coercion.
        Cast(c) => lower_cast(c),
        Return(r) => IrExpr::Return(
            r.expr
                .as_ref()
                .map(|e| Box::new(lower_expr_expected(e, expected))),
        ),
        Array(a) => lower_array_literal(a, None),
        Repeat(r) => lower_array_repeat(r, None),
        _ => IrExpr::Verbatim(
            syn::Error::new_spanned(e, "expression not supported in WGSL kernel")
                .to_compile_error()
                .to_string(),
        ),
    }
}

/// `base.member` — the member is a lexical name, never a WGSL decision.
fn lower_field(f: &syn::ExprField) -> IrExpr {
    IrExpr::Field {
        base: Box::new(lower_expr(&f.base)),
        member: member_name(&f.member),
    }
}

/// Spelling of a struct field reference: the identifier, or the tuple
/// field's ordinal.
fn member_name(member: &syn::Member) -> String {
    match member {
        syn::Member::Named(ident) => ident.to_string(),
        syn::Member::Unnamed(index) => index.index.to_string(),
    }
}

/// `-x` / `!x` / `*x` — the operator maps one-to-one, except that every
/// non-unary spelling shares the historical `*` fallback.
fn lower_unary(u: &syn::ExprUnary) -> IrExpr {
    IrExpr::Unary {
        op: un_op_of(&u.op),
        expr: Box::new(lower_expr(&u.expr)),
    }
}

/// WGSL spelling of a Rust unary operator.
fn un_op_of(op: &syn::UnOp) -> IrUnOp {
    match op {
        syn::UnOp::Neg(_) => IrUnOp::Neg,
        syn::UnOp::Not(_) => IrUnOp::Not,
        // Deref and anything else both spelled `*` (historical).
        _ => IrUnOp::DerefOrUnknown,
    }
}

/// `lhs = rhs` — both sides lower independently.
fn lower_assign(a: &syn::ExprAssign) -> IrExpr {
    IrExpr::Assign {
        left: Box::new(lower_expr(&a.left)),
        right: Box::new(lower_expr(&a.right)),
    }
}

/// `base[index]` — both sides lower independently.
fn lower_index(ix: &syn::ExprIndex) -> IrExpr {
    IrExpr::Index {
        base: Box::new(lower_expr(&ix.expr)),
        index: Box::new(lower_expr(&ix.index)),
    }
}

/// `'label: continue` has no WGSL spelling (WGSL labels only bind
/// loops); an unlabelled one lowers straight through.
fn lower_continue(c: &syn::ExprContinue) -> IrExpr {
    if c.label.is_some() {
        IrExpr::Unsupported
    } else {
        IrExpr::Continue
    }
}

/// `'label: break` and `break value` both have no WGSL spelling; only
/// the bare loop break lowers through.
fn lower_break(b: &syn::ExprBreak) -> IrExpr {
    if b.label.is_some() || b.expr.is_some() {
        IrExpr::Unsupported
    } else {
        IrExpr::Break
    }
}

/// Lower a whole body: statements in order, last one flagged as tail.
pub fn lower_body(stmts: &[Stmt]) -> IrBlock {
    lower_body_typed(stmts, None)
}

/// Lower a whole body with a return-type context. The context names the
/// generated `Option_<T>` struct for a tail `None`/`Some` and for a
/// tail or `return match …`.
pub fn lower_body_typed(stmts: &[Stmt], return_ty: Option<&IrType>) -> IrBlock {
    with_return_ty(return_ty, || {
        let count = stmts.len();
        stmts
            .iter()
            .enumerate()
            .flat_map(|(i, s)| {
                let is_tail = i == count - 1;
                lower_stmt_expected(s, is_tail, if is_tail { return_ty } else { None })
            })
            .collect()
    })
}

/// Lower one statement. Every arm delegates, so this match routes
/// rather than branches (IOSP Integration).
fn lower_stmt_expected(s: &Stmt, is_tail: bool, expected: Option<&IrType>) -> Vec<IrStmt> {
    match s {
        Stmt::Local(local) => lower_local(local),
        Stmt::Expr(syn::Expr::ForLoop(f), _) => lower_for_range(f),
        Stmt::Expr(expr, semi) => lower_stmt_expr(expr, semi.is_some(), is_tail, expected),
        // Loud by construction: anything untranslatable emits an unknown
        // call that naga rejects, instead of vanishing silently and
        // changing shader semantics. Pixel probes backstop drift.
        _ => unsupported_stmts(),
    }
}

/// One expression statement: `return`, `match` and the `if`/`block`
/// shapes each decide their own expansion.
fn lower_stmt_expr(
    expr: &syn::Expr,
    semi: bool,
    is_tail: bool,
    expected: Option<&IrType>,
) -> Vec<IrStmt> {
    match expr {
        syn::Expr::Return(_) => lower_return_stmt(expr, semi, expected),
        syn::Expr::Match(m) => lower_match_stmt_expr(m, semi, is_tail, expected),
        syn::Expr::If(_) | syn::Expr::Block(_) => flow_stmts(lower_expr(expr), semi),
        _ => expr_stmts(lower_expr_expected(expr, expected), semi, is_tail),
    }
}

/// `return …` — a `return match scrut { .. };` statement folds into the
/// value expansion with the `return` baked in; every other `return`
/// lowers as an ordinary expression statement.
fn lower_return_stmt(ret: &syn::Expr, semi: bool, expected: Option<&IrType>) -> Vec<IrStmt> {
    let syn::Expr::Return(r) = ret else {
        unreachable!("only reached for a `return` expression");
    };
    let expected = expected_or_return(expected);
    match (r.expr.as_deref(), expected.as_ref()) {
        (Some(syn::Expr::Match(m)), Some(expected)) => lower_match_value(m, Some(expected)),
        _ => expr_stmts(lower_expr_expected(ret, expected.as_ref()), semi, false),
    }
}

/// A `match` in statement position: the tail form assigns through the
/// value temporary and returns it, everything else folds into a nested
/// if/else chain over whole statement blocks.
fn lower_match_stmt_expr(
    m: &syn::ExprMatch,
    semi: bool,
    is_tail: bool,
    expected: Option<&IrType>,
) -> Vec<IrStmt> {
    match (is_tail, expected) {
        (true, Some(expected)) => lower_match_value(m, Some(expected)),
        _ => flow_stmts(lower_match_stmt(m), semi),
    }
}

/// One statement-position `if`/`block`, as the single-element list the
/// statement dispatchers return.
fn flow_stmts(expr: IrExpr, semi: bool) -> Vec<IrStmt> {
    vec![IrStmt::Flow { expr, semi }]
}

/// One general expression statement, as the single-element list the
/// statement dispatchers return.
fn expr_stmts(expr: IrExpr, semi: bool, is_tail: bool) -> Vec<IrStmt> {
    vec![IrStmt::Expr {
        expr,
        semi,
        is_tail,
    }]
}

/// The loud placeholder statement, as the single-element list the
/// statement dispatchers return.
fn unsupported_stmts() -> Vec<IrStmt> {
    vec![unsupported_stmt()]
}

/// Name of the temporary a value-position `match` assigns into. Leading
/// underscores keep it out of the way of user bindings.
const MATCH_RESULT: &str = "ornis_match_result";

/// A value-position `match` expanded to
/// `var __match_result: T; if (c) { … __match_result = …; } return __match_result;`.
fn lower_match_value(m: &syn::ExprMatch, expected: Option<&IrType>) -> Vec<IrStmt> {
    let decl_ty = expected.map(print_ty);
    let mut stmts = lower_match_assign(MATCH_RESULT, decl_ty, m);
    stmts.push(IrStmt::Tail(IrExpr::Path(vec![MATCH_RESULT.to_string()])));
    stmts
}

fn unsupported_stmt() -> IrStmt {
    IrStmt::Expr {
        expr: IrExpr::Verbatim(UNSUPPORTED_MARKER.to_string()),
        semi: true,
        is_tail: false,
    }
}

fn lower_lit(l: &syn::ExprLit) -> IrExpr {
    use syn::Lit::*;
    match &l.lit {
        Int(i) => IrExpr::Int(int_lit(i)),
        Float(f) => IrExpr::Float(float_lit(f)),
        Bool(b) => IrExpr::Bool(b.value),
        other => IrExpr::Verbatim(
            syn::Error::new_spanned(other, "literal type not supported in WGSL kernel")
                .to_compile_error()
                .to_string(),
        ),
    }
}

fn int_lit(i: &syn::LitInt) -> String {
    // Hex/octal/binary literals: pass through unchanged.
    let repr = i.to_string();
    if repr.starts_with("0x") || repr.starts_with("0o") || repr.starts_with("0b") {
        return repr;
    }
    // Rust integer suffixes are not valid WGSL; map the common
    // ones to the WGSL `u`/`i` suffixes and drop the rest.
    let digits = i.base10_digits();
    match i.suffix() {
        "u32" | "u64" | "usize" | "u16" | "u8" => format!("{digits}u"),
        "i32" | "i64" | "isize" | "i16" | "i8" => format!("{digits}i"),
        _ => digits.to_string(),
    }
}

fn float_lit(f: &syn::LitFloat) -> String {
    let base = f.base10_digits();
    if base.contains('.') || base.contains('e') || base.contains('E') {
        base.to_string()
    } else {
        format!("{base}.0")
    }
}

fn path_segments(p: &syn::ExprPath) -> Vec<String> {
    p.path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect()
}

/// Lower `e as T` to a WGSL conversion constructor (`f32(x)`,
/// `u32(x)`, `i32(x)`). Only the exact 32-bit scalar targets are
/// accepted: narrow targets (`u8`/`i16`/…) would need masking or
/// sign-extension to match Rust's wrapping semantics, and wide targets
/// (`u64`/`f64`/…) would silently narrow — both are rejected loudly
/// instead. Anything else (bool, vectors, structs) has no conversion
/// spelling and fails loudly instead of vanishing from the shader.
///
/// Value-semantics note: for in-range values the constructors truncate
/// like Rust `as`. Out-of-range float→int (where Rust saturates and maps
/// NaN to zero) is indeterminate in WGSL — keep kernel inputs in range.
fn lower_cast(c: &syn::ExprCast) -> IrExpr {
    let inner = lower_expr(&c.expr);
    let Some(ty) = cast_target_ty(&c.ty) else {
        return IrExpr::Verbatim(
            syn::Error::new_spanned(
                &c.ty,
                "cast target not supported in WGSL kernel; supported: f32, u32, i32",
            )
            .to_compile_error()
            .to_string(),
        );
    };
    IrExpr::Call {
        target: IrCallee::Constructor { ty },
        args: vec![inner],
    }
}

/// WGSL scalar for a Rust cast target: exactly `f32`/`u32`/`i32` as a
/// bare single-segment path, `None` for everything else. Qualified
/// spellings (`foo::u32`) are rejected too: the name could shadow a
/// different type, and kernel bodies have no use for them. Shared with
/// the `#[kernel]` validator so rejected targets fail at macro time,
/// not at naga.
pub(crate) fn cast_target_ty(ty: &syn::Type) -> Option<ShaderType> {
    if let syn::Type::Path(tp) = ty
        && tp.qself.is_none()
        && tp.path.segments.len() == 1
        && let Some(seg) = tp.path.segments.first()
    {
        return match seg.ident.to_string().as_str() {
            "f32" => Some(ShaderType::F32),
            "u32" => Some(ShaderType::U32),
            "i32" => Some(ShaderType::I32),
            _ => None,
        };
    }
    None
}

fn lower_binary(b: &syn::ExprBinary) -> IrExpr {
    use syn::BinOp::*;
    let op = match &b.op {
        Add(_) => IrBinOp::Add,
        Sub(_) => IrBinOp::Sub,
        Mul(_) => IrBinOp::Mul,
        Div(_) => IrBinOp::Div,
        Rem(_) => IrBinOp::Rem,
        AddAssign(_) => IrBinOp::AddAssign,
        SubAssign(_) => IrBinOp::SubAssign,
        MulAssign(_) => IrBinOp::MulAssign,
        DivAssign(_) => IrBinOp::DivAssign,
        RemAssign(_) => IrBinOp::RemAssign,
        ShlAssign(_) => IrBinOp::ShlAssign,
        ShrAssign(_) => IrBinOp::ShrAssign,
        BitAndAssign(_) => IrBinOp::BitAndAssign,
        BitOrAssign(_) => IrBinOp::BitOrAssign,
        BitXorAssign(_) => IrBinOp::BitXorAssign,
        // Rust `&&`/`||` are WGSL `&&`/`||` (short-circuit, not bitwise).
        And(_) => IrBinOp::And,
        Or(_) => IrBinOp::Or,
        Eq(_) => IrBinOp::Eq,
        Ne(_) => IrBinOp::Ne,
        Lt(_) => IrBinOp::Lt,
        Le(_) => IrBinOp::Le,
        Gt(_) => IrBinOp::Gt,
        Ge(_) => IrBinOp::Ge,
        Shl(_) => IrBinOp::Shl,
        Shr(_) => IrBinOp::Shr,
        BitXor(_) => IrBinOp::BitXor,
        BitAnd(_) => IrBinOp::BitAnd,
        BitOr(_) => IrBinOp::BitOr,
        _ => {
            return IrExpr::Verbatim(
                syn::Error::new_spanned(b.op, "operator not supported in WGSL kernel")
                    .to_compile_error()
                    .to_string(),
            );
        }
    };
    IrExpr::Binary {
        op,
        left: Box::new(lower_expr(&b.left)),
        right: Box::new(lower_expr(&b.right)),
    }
}

/// A path expression, with the expected type carried into `Some`/`None`.
fn lower_path(p: &syn::ExprPath, expected: Option<&IrType>) -> IrExpr {
    if p.qself.is_none() && p.path.segments.len() == 1 {
        let ident = &p.path.segments[0].ident;
        if ident == "None" {
            return lower_none(expected, p.span());
        }
        if ident == "Some" {
            return loud("`Some` needs one payload: `Some(x)`", p.span());
        }
    }
    IrExpr::Path(path_segments(p))
}

/// The generated struct name for `Some`/`None` at `expected`. Returns
/// the message to surface when the expected type is not an `Option`.
fn option_name_of(expected: Option<&IrType>) -> Result<String, String> {
    match expected {
        Some(IrType::Optional(inner)) => Ok(option_struct_name(&print_ty(inner))),
        _ => Err(
            "`Some`/`None` need a static `Option<T>` type context: annotate the binding \
             (`let x: Option<T> = …`) or the function's return type"
                .to_string(),
        ),
    }
}

/// The generated struct name and payload type for `Some`/`None` at a
/// known `Option<T>` context; the message to surface when the expected
/// type is not an `Option` comes back as `Err`.
fn option_target(
    expected: Option<&IrType>,
    span: proc_macro2::Span,
) -> Result<(String, Box<IrType>), IrExpr> {
    let expected = expected_or_return(expected);
    let expected = expected.as_ref();
    let name = match option_name_of(expected) {
        Ok(name) => name,
        Err(msg) => return Err(loud(&msg, span)),
    };
    let Some(IrType::Optional(inner)) = expected else {
        unreachable!("option_name_of only succeeds for IrType::Optional");
    };
    Ok((name, inner.clone()))
}

/// `None` at a known `Option<T>` context: the generated struct with a
/// zero payload (WGSL has no union, so the payload always exists).
fn lower_none(expected: Option<&IrType>, span: proc_macro2::Span) -> IrExpr {
    let (name, inner) = match option_target(expected, span) {
        Ok(target) => target,
        Err(msg) => return msg,
    };
    IrExpr::Call {
        target: IrCallee::Expr(Box::new(IrExpr::Path(vec![name]))),
        args: vec![IrExpr::Int("0u".to_string()), inner.zero_expr()],
    }
}

/// `Some(x)` at a known `Option<T>` context.
fn lower_some(c: &syn::ExprCall, expected: Option<&IrType>) -> IrExpr {
    let span = c.span();
    let (name, _) = match option_target(expected, span) {
        Ok(target) => target,
        Err(msg) => return msg,
    };
    if c.args.len() != 1 {
        return loud("`Some` takes exactly one payload", span);
    }
    IrExpr::Call {
        target: IrCallee::Expr(Box::new(IrExpr::Path(vec![name]))),
        args: vec![IrExpr::Int("1u".to_string()), lower_expr(&c.args[0])],
    }
}

/// A bare `None(...)`-shaped call is not valid Rust; kept for symmetry
/// with the value form so an accidental `None` constructor fails loudly.
fn lower_none_call(
    _c: &syn::ExprCall,
    expected: Option<&IrType>,
    span: proc_macro2::Span,
) -> IrExpr {
    lower_none(expected, span)
}

fn lower_call(c: &syn::ExprCall, expected: Option<&IrType>) -> IrExpr {
    if let syn::Expr::Path(p) = c.func.as_ref()
        && p.qself.is_none()
        && p.path.segments.len() == 1
    {
        let ident = &p.path.segments[0].ident;
        if ident == "Some" {
            return lower_some(c, expected);
        }
        if ident == "None" {
            return lower_none_call(c, expected, c.span());
        }
    }

    let args: Vec<IrExpr> = c.args.iter().map(lower_expr).collect();
    if let syn::Expr::Path(p) = c.func.as_ref() {
        let segs: Vec<_> = p.path.segments.iter().collect();
        let ident = segs.last().map(|s| s.ident.to_string()).unwrap_or_default();
        // Constructors: glam::Vec3::new(a,b,c) / Vec3::splat(1.0) /
        // Mat3::from_cols(c0,c1,c2) (column order is the WGSL
        // mat3x3 constructor order; from_rows has no direct spelling
        // and stays loud below).
        if (ident == "new" || ident == "splat" || ident == "from_cols")
            && let Some(type_seg) = segs.iter().rev().nth(1)
            && let Some(ty) = ShaderType::from_rust(&type_seg.ident.to_string())
        {
            return IrExpr::Call {
                target: IrCallee::Constructor { ty },
                args,
            };
        }
        if let Some(b) = ShaderBuiltin::from_rust(&ident) {
            return IrExpr::Call {
                target: IrCallee::Builtin(b),
                args,
            };
        }
    }
    IrExpr::Call {
        target: IrCallee::Expr(Box::new(lower_expr(&c.func))),
        args,
    }
}

fn lower_method_call(m: &syn::ExprMethodCall) -> IrExpr {
    let receiver = lower_expr(&m.receiver);
    let method = m.method.to_string();
    let rest: Vec<IrExpr> = m.args.iter().map(lower_expr).collect();

    if method == "powi" {
        return lower_powi(m, receiver, rest);
    }
    // Swizzles become field access; registry built-ins lower;
    // anything else (kernels/helpers) stays a structural method node
    // for the writer to print verbatim.
    if ornis_shader_lang::is_swizzle_method(&method) {
        return IrExpr::Field {
            base: Box::new(receiver),
            member: method,
        };
    }
    let mut all_args = vec![receiver];
    all_args.extend(rest);
    if let Some(b) = ShaderBuiltin::from_rust(&method) {
        return IrExpr::Call {
            target: IrCallee::Builtin(b),
            args: all_args,
        };
    }
    let mut rest = all_args;
    let receiver = rest.remove(0);
    IrExpr::Method {
        base: Box::new(receiver),
        method,
        args: rest,
    }
}

/// Expand `x.powi(n)` for small positive integer literals into repeated
/// multiplication; everything else falls back to WGSL `pow(...)`.
fn lower_powi(m: &syn::ExprMethodCall, receiver: IrExpr, rest: Vec<IrExpr>) -> IrExpr {
    if let Some(exponent) = powi_exponent(m) {
        let mut iter = std::iter::repeat_n(receiver, exponent as usize);
        if let Some(first) = iter.next() {
            return iter.fold(first, |acc, part| IrExpr::Binary {
                op: IrBinOp::Mul,
                left: Box::new(acc),
                right: Box::new(part),
            });
        }
        return IrExpr::Verbatim(String::new());
    }
    let mut args = vec![receiver];
    args.extend(rest);
    IrExpr::Call {
        target: IrCallee::Builtin(ShaderBuiltin::Pow),
        args,
    }
}

/// The literal exponent of a `powi` call, if it is a small positive int.
fn powi_exponent(m: &syn::ExprMethodCall) -> Option<u32> {
    let arg = m.args.first()?;
    let syn::Expr::Lit(lit) = arg else {
        return None;
    };
    let syn::Lit::Int(int_lit) = &lit.lit else {
        return None;
    };
    let p = int_lit.base10_parse::<i32>().unwrap_or(0);
    u32::try_from(p).ok()
}

/// Struct literal → positional constructor: field values in literal order.
/// Callers must list fields in declaration order — the macro cannot see
/// the struct definition. The writer's `/* names */` comment lets the
/// `interface_field_order` shader test verify the order against the
/// mirror's `WGSL_FIELDS` instead of trusting it blindly.
fn lower_struct_lit(s: &syn::ExprStruct) -> IrExpr {
    let name = s
        .path
        .segments
        .last()
        .map(|seg| seg.ident.to_string())
        .unwrap_or_default();
    if s.rest.is_some() {
        return IrExpr::Verbatim(
            syn::Error::new_spanned(
                &s.rest,
                "struct update syntax (`..base`) is not supported in WGSL",
            )
            .to_compile_error()
            .to_string(),
        );
    }
    let fields = s
        .fields
        .iter()
        .map(|f| {
            let member = match &f.member {
                syn::Member::Named(ident) => ident.to_string(),
                syn::Member::Unnamed(index) => index.index.to_string(),
            };
            (member, lower_expr(&f.expr))
        })
        .collect();
    IrExpr::Struct { name, fields }
}

fn lower_if(i: &syn::ExprIf, expected: Option<&IrType>) -> IrExpr {
    // `if let Some(v) = e` / `if let None = e` / `if let _ = e` lower to
    // the same discriminant test `match` uses. The scrutinee is named
    // twice (`e.has`, `e.value`), so it must be side-effect free.
    if let syn::Expr::Let(lt) = i.cond.as_ref() {
        return lower_if_let(lt, i, expected);
    }
    let cond = Box::new(lower_expr(&i.cond));
    let then = lower_block_expected(&i.then_branch, expected);
    let els = i
        .else_branch
        .as_ref()
        .map(|(_, else_expr)| match else_expr.as_ref() {
            syn::Expr::Block(blk) => IrElse::Block(lower_block_expected(&blk.block, expected)),
            syn::Expr::If(inner_if) => IrElse::If(Box::new(lower_if(inner_if, expected))),
            _ => IrElse::Block(vec![]),
        });
    IrExpr::If { cond, then, els }
}

/// `if let PAT = scrutinee { .. } else { .. }`.
fn lower_if_let(lt: &syn::ExprLet, i: &syn::ExprIf, expected: Option<&IrType>) -> IrExpr {
    let scrutinee = lower_expr(&lt.expr);
    let pattern = match classify_pattern(&lt.pat) {
        Ok(p) => p,
        Err(msg) => return loud(&msg, lt.pat.span()),
    };
    let cond = match &pattern {
        ArmPattern::Wild | ArmPattern::Bind(_) => IrExpr::Bool(true),
        ArmPattern::Lit => {
            let Pat::Lit(l) = lt.pat.as_ref() else {
                unreachable!("classify_pattern only reports Lit for Pat::Lit");
            };
            IrExpr::Binary {
                op: IrBinOp::Eq,
                left: Box::new(scrutinee.clone()),
                right: Box::new(lower_expr(&syn::Expr::Lit(l.clone()))),
            }
        }
        ArmPattern::Some(_) => has_test(&scrutinee, IrBinOp::Ne),
        ArmPattern::None => has_test(&scrutinee, IrBinOp::Eq),
    };
    let mut then = lower_block_expected(&i.then_branch, expected);
    if let Some(name) = pattern_binding(&pattern) {
        then.insert(
            0,
            IrStmt::Let {
                name,
                mutable: false,
                init: Some(IrExpr::Field {
                    base: Box::new(scrutinee),
                    member: "value".to_string(),
                }),
                decl_ty: None,
            },
        );
    }
    let els = i
        .else_branch
        .as_ref()
        .map(|(_, else_expr)| match else_expr.as_ref() {
            syn::Expr::Block(blk) => IrElse::Block(lower_block_expected(&blk.block, expected)),
            syn::Expr::If(inner_if) => IrElse::If(Box::new(lower_if(inner_if, expected))),
            _ => IrElse::Block(vec![]),
        });
    IrExpr::If {
        cond: Box::new(cond),
        then,
        els,
    }
}

/// The binding name an `Option` pattern introduces, if any.
fn pattern_binding(pattern: &ArmPattern) -> Option<String> {
    match pattern {
        ArmPattern::Bind(name) | ArmPattern::Some(name) => Some(name.clone()),
        ArmPattern::Wild | ArmPattern::Lit | ArmPattern::None => None,
    }
}

fn lower_block(block: &Block) -> IrBlock {
    lower_body(&block.stmts)
}

fn lower_block_expected(block: &Block, expected: Option<&IrType>) -> IrBlock {
    lower_body_typed(&block.stmts, expected)
}

/// Expansion cap for `[e; N]` repeats (solver scratch needs ≤ 81; 1024
/// leaves an order of magnitude of headroom while a typo'd length fails
/// loudly instead of blowing macro memory).
const MAX_REPEAT: usize = 1024;

/// Lowered element type for a Rust array element type: registry scalars /
/// vectors / matrices, or a nested fixed array — structurally, so the
/// writer owns every spelling. `None` for anything else (heap types,
/// structs — loud at the call site).
fn array_ir_ty(ty: &syn::Type) -> Option<IrType> {
    match ty {
        syn::Type::Path(tp) => {
            let last = tp.path.segments.last()?.ident.to_string();
            if last == "bool" {
                // Registry-excluded for call classification, printable
                // as an explicit array element type.
                return Some(IrType::Bool);
            }
            ShaderType::from_rust(&last).map(IrType::Scalar)
        }
        syn::Type::Array(arr) => {
            let len = array_len(&arr.len)?;
            let inner = array_ir_ty(&arr.elem)?;
            Some(IrType::Array {
                elem: Box::new(inner),
                len,
            })
        }
        _ => None,
    }
}

/// Integer length of `[T; N]`: plain integer literals only (a Rust
/// `const` length has no WGSL spelling at macro time — loud instead).
fn array_len(len: &syn::Expr) -> Option<usize> {
    if let syn::Expr::Lit(lit) = len
        && let syn::Lit::Int(i) = &lit.lit
    {
        return i.base10_parse::<usize>().ok();
    }
    None
}

/// Infer an element type for an UNANNOTATED array value from its first
/// item (literals only). Anything else is loud — annotate the `let`.
fn infer_lit_ty(e: &syn::Expr) -> Option<IrType> {
    match e {
        syn::Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Float(_) => Some(IrType::Scalar(ShaderType::F32)),
            syn::Lit::Int(i) => match i.suffix() {
                "u32" | "u64" | "usize" | "u16" | "u8" => Some(IrType::Scalar(ShaderType::U32)),
                _ => Some(IrType::Scalar(ShaderType::I32)),
            },
            syn::Lit::Bool(_) => Some(IrType::Bool),
            _ => None,
        },
        syn::Expr::Repeat(r) => infer_lit_ty(&r.expr),
        syn::Expr::Array(a) => a.elems.first().and_then(infer_lit_ty),
        _ => None,
    }
}

fn loud_array(msg: &str, span: proc_macro2::Span) -> IrExpr {
    IrExpr::Verbatim(syn::Error::new(span, msg).to_compile_error().to_string())
}

/// Lower `[e; N]` with a known element type (`Some`, the annotation's own
/// level) or an inferred one (`None`). The element must be a literal, a
/// path (constant), or a nested array shape: anything effectful
/// (`[foo(); 3]`) would evaluate N times after expansion instead of
/// once — loud, never silently reordered. Nested repeats descend one
/// annotation level per layer, so `[[f32; 3]; 3]` keeps the inner
/// `array<f32, 3>` instead of smearing the outer type inward.
fn lower_array_repeat(r: &syn::ExprRepeat, elem_ty: Option<IrType>) -> IrExpr {
    use syn::spanned::Spanned;
    let Some(len) = array_len(&r.len) else {
        return loud_array(
            "array repeat length must be an integer literal in WGSL kernels",
            r.len.span(),
        );
    };
    if len > MAX_REPEAT {
        return loud_array(
            "array repeat length exceeds the 1024-element macro cap",
            r.len.span(),
        );
    }
    let elem: &syn::Expr = &r.expr;
    let one: Option<IrExpr> = match elem {
        syn::Expr::Lit(_) | syn::Expr::Path(_) => Some(lower_expr(elem)),
        syn::Expr::Array(a) => {
            // Nested literal: lower items directly (each appears once).
            let items: Vec<IrExpr> = a.elems.iter().map(lower_expr).collect();
            let ty = elem_ty.clone().or_else(|| infer_lit_ty(elem));
            ty.map(|ty| IrExpr::Array { elem: ty, items })
        }
        syn::Expr::Repeat(inner) => {
            let inner: &syn::ExprRepeat = inner;
            let sub = match &elem_ty {
                Some(IrType::Array { elem, .. }) => Some(elem.as_ref().clone()),
                _ => None,
            };
            Some(lower_array_repeat(inner, sub))
        }
        _ => None,
    };
    let Some(one) = one else {
        return loud_array(
            "array repeat elements must be literals, constants or nested array shapes in WGSL kernels",
            r.expr.span(),
        );
    };
    let Some(ty) = elem_ty.or_else(|| infer_lit_ty(elem)) else {
        return loud_array(
            "cannot infer the array element type; annotate the `let` binding",
            r.span(),
        );
    };
    IrExpr::Array {
        elem: ty,
        items: std::iter::repeat_n(one, len).collect(),
    }
}

/// Lower `[a, b, ..]` with a known element type (`Some`, from the `let`
/// annotation) or an inferred one (`None`). Literal items evaluate once
/// each, in order — no effect restriction needed.
fn lower_array_literal(a: &syn::ExprArray, elem_ty: Option<IrType>) -> IrExpr {
    use syn::spanned::Spanned;
    let items: Vec<IrExpr> = a.elems.iter().map(lower_expr).collect();
    let Some(ty) = elem_ty.or_else(|| a.elems.first().and_then(infer_lit_ty)) else {
        return loud_array(
            "cannot infer the array element type; annotate the `let` binding",
            a.span(),
        );
    };
    IrExpr::Array { elem: ty, items }
}

/// Lower an initializer KNOWN to sit under a `[T; N]` annotation with the
/// annotation's element type. Returns `None` for non-array shapes (the
/// caller keeps the historical path); unexpandable array shapes come
/// back loud via the callees, never silently dropped.
fn lower_typed_array(elem_ty: IrType, init: &syn::Expr) -> Option<IrExpr> {
    match init {
        syn::Expr::Array(a) => Some(lower_array_literal(a, Some(elem_ty))),
        syn::Expr::Repeat(r) => Some(lower_array_repeat(r, Some(elem_ty))),
        _ => None,
    }
}

/// `for i in start..end` / `start..=end` → range loop over `u32`.
/// Anything else (iterators, steps, patterns) is loud via [`IrExpr::Unsupported`].
fn lower_for_range(f: &ExprForLoop) -> Vec<IrStmt> {
    let syn::Pat::Ident(pi) = &*f.pat else {
        return vec![unsupported_stmt()];
    };
    let (start, end, closed) = match &*f.expr {
        syn::Expr::Range(r) => {
            let start = r
                .start
                .as_ref()
                .map(|e| lower_expr(e))
                .unwrap_or_else(|| IrExpr::Int("0u".to_string()));
            let end = r.end.as_ref().map(|e| lower_expr(e));
            let closed = matches!(r.limits, syn::RangeLimits::Closed(_));
            (start, end, closed)
        }
        _ => return vec![unsupported_stmt()],
    };
    let Some(end) = end else {
        return vec![unsupported_stmt()];
    };
    vec![IrStmt::For {
        var: pi.ident.to_string(),
        start,
        end,
        closed,
        body: lower_block(&f.body),
    }]
}

/// Translate a `let` / `let mut` binding. `let mut` maps to WGSL `var`
/// (the only reassignable binding kind); plain `let` maps to `let`.
fn lower_local(local: &Local) -> Vec<IrStmt> {
    // Handle `let x = if/else { .. } else { .. };` — wrap if/else into var+if
    if let Some(init) = &local.init
        && let syn::Expr::If(if_expr) = init.expr.as_ref()
    {
        let name = pat_name(&local.pat);
        let cond = Box::new(lower_expr(&if_expr.cond));
        let then_val = last_expr_in_block(&if_expr.then_branch);
        let else_val = if_expr
            .else_branch
            .as_ref()
            .and_then(|(_, else_expr)| {
                if let syn::Expr::Block(block) = else_expr.as_ref() {
                    Some(last_expr_in_block(&block.block))
                } else {
                    None
                }
            })
            .unwrap_or(IrExpr::Verbatim(String::new()));
        // Convert: let x = if c { a } else { b };
        // To WGSL: var x = b; if (c) { x = a; }
        let assign = IrStmt::Expr {
            expr: IrExpr::Assign {
                left: Box::new(IrExpr::Path(vec![name.clone()])),
                right: Box::new(then_val),
            },
            semi: true,
            is_tail: false,
        };
        return vec![
            IrStmt::Let {
                name,
                mutable: true,
                init: Some(else_val),
                decl_ty: None,
            },
            IrStmt::Flow {
                expr: IrExpr::If {
                    cond,
                    then: vec![assign],
                    els: None,
                },
                semi: false,
            },
        ];
    }
    // `let x = match scrut { .. };`: the value form expands to
    // `var x = <last arm>; if (c) { x = <arm>; } …`.
    if let Some(init) = &local.init
        && let syn::Expr::Match(m) = init.expr.as_ref()
    {
        let decl_ty = match &local.pat {
            syn::Pat::Type(pt) => Some(crate::wgsl::var_type(&pt.ty)),
            _ => None,
        };
        let name = pat_name(&local.pat);
        return lower_match_assign(&name, decl_ty, m);
    }
    let (name, is_mut) = match &local.pat {
        syn::Pat::Ident(pi) => (pi.ident.to_string(), pi.mutability.is_some()),
        syn::Pat::Type(pt) => match pt.pat.as_ref() {
            syn::Pat::Ident(pi) => (pi.ident.to_string(), pi.mutability.is_some()),
            _ => (pat_name(&local.pat), false),
        },
        _ => (pat_name(&local.pat), false),
    };
    // Annotated fixed-size scratch (`let mut d: [f32; 6] = [0.0; 6];`):
    // the annotation drives the element type, the value its length.
    // Anything else keeps the historical annotation-blind path below.
    if let (Some(init), syn::Pat::Type(pt)) = (&local.init, &local.pat)
        && let syn::Type::Array(arr) = pt.ty.as_ref()
        && let Some(elem_ty) = array_ir_ty(&arr.elem)
        && let Some(array) = lower_typed_array(elem_ty, &init.expr)
    {
        let decl_ty = match &array {
            IrExpr::Array { elem, items } => Some(print_ty(&IrType::Array {
                elem: Box::new(elem.clone()),
                len: items.len(),
            })),
            _ => None,
        };
        return vec![IrStmt::Let {
            name,
            mutable: is_mut,
            init: Some(array),
            decl_ty,
        }];
    }
    // `let x: Option<T> = Some(v) | None;` — the annotation is the
    // type context that names the generated struct.
    if let syn::Pat::Type(pt) = &local.pat
        && let Some(opt_ty) = crate::wgsl_options::option_struct_ty(&pt.ty)
    {
        let init = local
            .init
            .as_ref()
            .map(|init| lower_expr_expected(&init.expr, Some(&opt_ty)));
        return vec![IrStmt::Let {
            name,
            mutable: is_mut,
            init,
            decl_ty: Some(print_ty(&opt_ty)),
        }];
    }
    let init = local.init.as_ref().map(|init| lower_expr(&init.expr));
    // Rust `let mut` becomes WGSL `var` — the only WGSL binding kind that
    // can be reassigned.
    let decl_ty = if init.is_none() {
        if let syn::Pat::Type(pt) = &local.pat {
            // Uninitialized declaration (`let mut out: T;`): WGSL needs the
            // explicit type. Only occurs in stripped stage bodies — compiled
            // Rust always initializes through the first branch.
            Some(crate::wgsl::var_type(&pt.ty))
        } else {
            None
        }
    } else {
        None
    };
    vec![IrStmt::Let {
        name,
        mutable: is_mut,
        init,
        decl_ty,
    }]
}

fn last_expr_in_block(block: &Block) -> IrExpr {
    block
        .stmts
        .last()
        .and_then(|s| {
            if let Stmt::Expr(e, _) = s {
                Some(lower_expr(e))
            } else {
                None
            }
        })
        .unwrap_or(IrExpr::Verbatim(String::new()))
}

/// `scrutinee.has != 0u` (`op_ne`) or `scrutinee.has == 0u` (`op_eq`).
fn has_test(scrutinee: &IrExpr, op: IrBinOp) -> IrExpr {
    IrExpr::Binary {
        op,
        left: Box::new(IrExpr::Field {
            base: Box::new(scrutinee.clone()),
            member: "has".to_string(),
        }),
        right: Box::new(IrExpr::Int("0u".to_string())),
    }
}

/// One `match` arm: the discriminant test (absent for `_` / a bare
/// binding) plus the statements its body contributes.
struct MatchArm {
    cond: Option<IrExpr>,
    then: Vec<IrStmt>,
}

/// Classify every arm of a `match` over an already-lowered scrutinee.
fn match_arm_conds(m: &syn::ExprMatch, scrutinee: &IrExpr) -> Vec<MatchArm> {
    let mut out = Vec::new();
    for arm in &m.arms {
        let pattern = match classify_pattern(&arm.pat) {
            Ok(p) => p,
            Err(msg) => {
                out.push(MatchArm {
                    cond: Some(loud(&msg, arm.pat.span())),
                    then: arm_body_block(&arm.body),
                });
                continue;
            }
        };
        let cond = arm_cond(&pattern, &arm.pat, scrutinee);
        let mut then = arm_body_block(&arm.body);
        if let Some(name) = pattern_binding(&pattern) {
            then.insert(0, payload_binding(&name, scrutinee));
        }
        out.push(MatchArm { cond, then });
    }
    out
}

/// The discriminant test for one pattern, or `None` when the arm is the
/// unconditional remainder.
fn arm_cond(pattern: &ArmPattern, pat: &Pat, scrutinee: &IrExpr) -> Option<IrExpr> {
    match pattern {
        ArmPattern::Wild | ArmPattern::Bind(_) => None,
        ArmPattern::Lit => {
            let Pat::Lit(l) = pat else {
                unreachable!("classify_pattern only reports Lit for Pat::Lit");
            };
            Some(IrExpr::Binary {
                op: IrBinOp::Eq,
                left: Box::new(scrutinee.clone()),
                right: Box::new(lower_expr(&syn::Expr::Lit(l.clone()))),
            })
        }
        ArmPattern::Some(_) => Some(has_test(scrutinee, IrBinOp::Ne)),
        ArmPattern::None => Some(has_test(scrutinee, IrBinOp::Eq)),
    }
}

/// `let <name> = <scrutinee>.value;`
fn payload_binding(name: &str, scrutinee: &IrExpr) -> IrStmt {
    IrStmt::Let {
        name: name.to_string(),
        mutable: false,
        init: Some(IrExpr::Field {
            base: Box::new(scrutinee.clone()),
            member: "value".to_string(),
        }),
        decl_ty: None,
    }
}

/// Statement-position `match`: every arm contributes a full statement
/// block, so the arms fold right-to-left into a nested if/else chain
/// with no artificial "last arm is unconditional" rule.
fn lower_match_stmt(m: &syn::ExprMatch) -> IrExpr {
    let scrutinee = lower_expr(&m.expr);
    fold_match_arms(match_arm_conds(m, &scrutinee))
}

/// Right-fold the classified arms into `IrExpr::If` / `IrElse` nodes.
/// An arm with no condition is the unconditional remainder.
fn fold_match_arms(arms: Vec<MatchArm>) -> IrExpr {
    // Folded right-to-left: the accumulator is the else-branch of the
    // next `if`, so an unconditional arm simply becomes that else.
    let mut acc: Option<IrElse> = None;
    for arm in arms {
        match arm.cond {
            None => acc = Some(IrElse::Block(arm.then)),
            Some(cond) => {
                let els = acc.take();
                acc = Some(IrElse::If(Box::new(IrExpr::If {
                    cond: Box::new(cond),
                    then: arm.then,
                    els,
                })));
            }
        }
    }
    match acc {
        Some(IrElse::Block(stmts)) => IrExpr::Block(stmts),
        Some(IrElse::If(e)) => *e,
        None => IrExpr::Unsupported,
    }
}

/// A value-position `match` is not a statement chain: it assigns into
/// one temporary. Each arm therefore contributes a value, not a block.
struct MatchValueArm {
    cond: Option<IrExpr>,
    /// `let <payload> = <scrutinee>.value;` before the arm body runs.
    prelude: Vec<IrStmt>,
    value: IrExpr,
}

fn match_arm_values(m: &syn::ExprMatch, scrutinee: &IrExpr) -> Vec<MatchValueArm> {
    m.arms
        .iter()
        .map(|arm| {
            let pattern = match classify_pattern(&arm.pat) {
                Ok(p) => p,
                Err(msg) => {
                    return MatchValueArm {
                        cond: Some(loud(&msg, arm.pat.span())),
                        prelude: Vec::new(),
                        value: IrExpr::Unsupported,
                    };
                }
            };
            let mut prelude = Vec::new();
            if let Some(name) = pattern_binding(&pattern) {
                prelude.push(payload_binding(&name, scrutinee));
            }
            MatchValueArm {
                cond: arm_cond(&pattern, &arm.pat, scrutinee),
                prelude,
                value: arm_value_expr(&arm.body),
            }
        })
        .collect()
}

/// The value of one value-position arm: a bare expression, or a block
/// holding exactly one trailing expression.
fn arm_value_expr(body: &syn::Expr) -> IrExpr {
    if let syn::Expr::Block(b) = body {
        return match b.block.stmts.as_slice() {
            [syn::Stmt::Expr(e, None)] => lower_expr(e),
            _ => loud(
                "arms of a value `match` must be a single expression; bind intermediate \
                 results with `let` and keep the arm to one trailing expression",
                body.span(),
            ),
        };
    }
    lower_expr(body)
}

/// The statements one statement-position arm contributes. A block body
/// is spliced verbatim; a bare expression becomes a lone statement.
fn arm_body_block(body: &syn::Expr) -> Vec<IrStmt> {
    if let syn::Expr::Block(b) = body {
        return b
            .block
            .stmts
            .iter()
            .flat_map(|st| lower_stmt_expected(st, false, None))
            .collect();
    }
    vec![IrStmt::Expr {
        expr: lower_expr(body),
        semi: false,
        is_tail: false,
    }]
}

/// `let x: Option<T> = match …` / `let x = match …` (and the tail and
/// `return` hosts) lower to
/// `var x = <last arm>; if (c) { x = <arm>; } else …`.
fn lower_match_assign(name: &str, decl_ty: Option<String>, m: &syn::ExprMatch) -> Vec<IrStmt> {
    let scrutinee = lower_expr(&m.expr);
    let arms = match_arm_values(m, &scrutinee);
    if arms.is_empty() {
        return vec![unsupported_stmt()];
    }
    if let Some(decl_ty) = decl_ty {
        return assigned_chain(name, decl_ty, arms);
    }
    seeded_chain(name, arms)
}

/// One arm of the chain: its prelude, then the assignment into `name`.
fn assign_arm(name: &str, arm: &MatchValueArm) -> Vec<IrStmt> {
    let mut then = arm.prelude.clone();
    then.push(IrStmt::Expr {
        expr: IrExpr::Assign {
            left: Box::new(IrExpr::Path(vec![name.to_string()])),
            right: Box::new(arm.value.clone()),
        },
        semi: true,
        is_tail: false,
    });
    then
}

/// A known type: declare the temporary uninitialized and let the full
/// if/else chain — including the last arm — fill it in. This is the
/// only shape that can express an arm which binds a payload.
fn assigned_chain(name: &str, decl_ty: String, arms: Vec<MatchValueArm>) -> Vec<IrStmt> {
    let mut stmts = vec![IrStmt::Let {
        name: name.to_string(),
        mutable: true,
        init: None,
        decl_ty: Some(decl_ty),
    }];
    let mut acc: Option<IrElse> = None;
    for arm in arms.into_iter().rev() {
        match arm.cond.clone() {
            None => acc = Some(IrElse::Block(assign_arm(name, &arm))),
            Some(cond) => {
                let els = acc.take();
                acc = Some(IrElse::If(Box::new(IrExpr::If {
                    cond: Box::new(cond),
                    then: assign_arm(name, &arm),
                    els,
                })));
            }
        }
    }
    match acc {
        Some(IrElse::Block(stmts_tail)) => stmts.extend(stmts_tail),
        Some(IrElse::If(e)) => stmts.push(IrStmt::Flow {
            expr: *e,
            semi: false,
        }),
        None => {}
    }
    stmts
}

/// No annotation: seed the temporary from the last arm, then let the
/// remaining conditional arms overwrite it.
fn seeded_chain(name: &str, mut arms: Vec<MatchValueArm>) -> Vec<IrStmt> {
    let Some(last) = arms.pop() else {
        return vec![unsupported_stmt()];
    };
    if !last.prelude.is_empty() {
        return vec![unsupported_stmt()];
    }
    let mut stmts = vec![IrStmt::Let {
        name: name.to_string(),
        mutable: true,
        init: Some(last.value.clone()),
        decl_ty: None,
    }];
    let mut chain: Option<Box<IrExpr>> = None;
    for arm in arms.into_iter().rev() {
        let Some(cond) = arm.cond.clone() else {
            // An unconditional arm before the last one is dead code;
            // keep it out of the chain so the writer stays honest.
            continue;
        };
        let node = IrExpr::If {
            cond: Box::new(cond),
            then: assign_arm(name, &arm),
            els: chain.take().map(IrElse::If),
        };
        chain = Some(Box::new(node));
    }
    if let Some(node) = chain {
        stmts.push(IrStmt::Flow {
            expr: *node,
            semi: false,
        });
    }
    stmts
}

/// Value-position `match` in an expression position with no known type:
/// loud, rather than a silent `Unsupported` that naga would only catch
/// much later.
fn lower_match_expr(m: &syn::ExprMatch) -> IrExpr {
    loud(
        "`match` only lowers as a statement or as a value with a known type \
         (a `let` initializer, a tail expression, or `return match …`)",
        m.span(),
    )
}

fn pat_name(p: &Pat) -> String {
    match p {
        syn::Pat::Ident(pi) => pi.ident.to_string(),
        _ => "_".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn lerp_lowers_to_mix_node_not_string() {
        let expr: syn::Expr = parse_quote!(a.lerp(b, t));
        assert_eq!(
            lower_expr(&expr),
            IrExpr::Call {
                target: IrCallee::Builtin(ShaderBuiltin::Mix),
                args: vec![
                    IrExpr::Path(vec!["a".to_string()]),
                    IrExpr::Path(vec!["b".to_string()]),
                    IrExpr::Path(vec!["t".to_string()]),
                ],
            }
        );
    }

    #[test]
    fn powi_two_lowers_to_multiply_chain() {
        let expr: syn::Expr = parse_quote!(x.powi(2));
        let x = || IrExpr::Path(vec!["x".to_string()]);
        assert_eq!(
            lower_expr(&expr),
            IrExpr::Binary {
                op: IrBinOp::Mul,
                left: Box::new(x()),
                right: Box::new(x()),
            }
        );
    }

    #[test]
    fn swizzle_lowers_to_field_node() {
        let expr: syn::Expr = parse_quote!(pos.xy);
        assert_eq!(
            lower_expr(&expr),
            IrExpr::Field {
                base: Box::new(IrExpr::Path(vec!["pos".to_string()])),
                member: "xy".to_string(),
            }
        );
    }

    #[test]
    fn let_if_expands_to_var_plus_if() {
        let stmt: Stmt = parse_quote!(let x = if c { a } else { b };);
        let Stmt::Local(local) = stmt else {
            panic!("expected local");
        };
        let stmts = lower_local(&local);
        assert_eq!(stmts.len(), 2);
        assert!(matches!(
            &stmts[0],
            IrStmt::Let {
                name,
                mutable: true,
                init: Some(_),
                decl_ty: None,
            } if name == "x"
        ));
        assert!(matches!(&stmts[1], IrStmt::Flow { semi: false, .. }));
    }

    #[test]
    fn casts_lower_to_constructor_nodes() {
        for (rust, wgsl) in [
            ("f32", ShaderType::F32),
            ("u32", ShaderType::U32),
            ("i32", ShaderType::I32),
        ] {
            let ty: syn::Type = syn::parse_str(rust).unwrap();
            assert_eq!(cast_target_ty(&ty), Some(wgsl), "{rust}");
        }
        // Narrow, wide and non-scalar targets have no conversion
        // spelling: the validator rejects them before lowering.
        for rust in ["u8", "i16", "u64", "usize", "f64", "bool", "Vec3"] {
            let ty: syn::Type = syn::parse_str(rust).unwrap();
            assert_eq!(cast_target_ty(&ty), None, "{rust}");
        }
        // Qualified spellings are rejected even when the last segment
        // names a supported scalar: the path could shadow another type.
        let qualified: syn::Type = syn::parse_str("foo::u32").unwrap();
        assert_eq!(cast_target_ty(&qualified), None);
        let expr: syn::Expr = parse_quote!(x as u32);
        assert_eq!(
            lower_expr(&expr),
            IrExpr::Call {
                target: IrCallee::Constructor {
                    ty: ShaderType::U32
                },
                args: vec![IrExpr::Path(vec!["x".to_string()])],
            }
        );
    }

    #[test]
    fn unsupported_cast_target_is_loud() {
        let bool_ty: syn::Type = syn::parse_str("bool").unwrap();
        assert_eq!(cast_target_ty(&bool_ty), None);
        let expr: syn::Expr = parse_quote!(x as bool);
        assert!(matches!(lower_expr(&expr), IrExpr::Verbatim(_)));
    }

    #[test]
    fn constructor_classifies_at_lowering() {
        let expr: syn::Expr = parse_quote!(glam::Vec3::new(1.0, 2.0, 3.0));
        assert!(matches!(
            lower_expr(&expr),
            IrExpr::Call {
                target: IrCallee::Constructor { .. },
                ..
            }
        ));
    }

    #[test]
    fn unknown_call_stays_structural() {
        let expr: syn::Expr = parse_quote!(my_kernel(x));
        assert_eq!(
            lower_expr(&expr),
            IrExpr::Call {
                target: IrCallee::Expr(Box::new(IrExpr::Path(vec!["my_kernel".to_string()]))),
                args: vec![IrExpr::Path(vec!["x".to_string()])],
            }
        );
    }
}
