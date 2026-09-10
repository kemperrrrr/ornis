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
    IrBinOp, IrBlock, IrCallee, IrElse, IrExpr, IrStmt, IrUnOp, UNSUPPORTED_MARKER,
};
use syn::{Block, ExprForLoop, Local, Pat, Stmt};

/// Lower one expression.
pub fn lower_expr(e: &syn::Expr) -> IrExpr {
    use syn::Expr::*;
    match e {
        Lit(l) => lower_lit(l),
        Path(p) => IrExpr::Path(path_segments(p)),
        Field(f) => {
            let member = match &f.member {
                syn::Member::Named(ident) => ident.to_string(),
                syn::Member::Unnamed(index) => index.index.to_string(),
            };
            IrExpr::Field {
                base: Box::new(lower_expr(&f.base)),
                member,
            }
        }
        Binary(b) => lower_binary(b),
        Unary(u) => {
            use syn::UnOp::*;
            let op = match &u.op {
                Neg(_) => IrUnOp::Neg,
                Not(_) => IrUnOp::Not,
                // Deref and anything else both spelled `*` (historical).
                _ => IrUnOp::DerefOrUnknown,
            };
            IrExpr::Unary {
                op,
                expr: Box::new(lower_expr(&u.expr)),
            }
        }
        Paren(p) => IrExpr::Paren(Box::new(lower_expr(&p.expr))),
        Call(c) => lower_call(c),
        MethodCall(m) => lower_method_call(m),
        If(i) => lower_if(i),
        Block(b) => IrExpr::Block(lower_block(&b.block)),
        Assign(a) => IrExpr::Assign {
            left: Box::new(lower_expr(&a.left)),
            right: Box::new(lower_expr(&a.right)),
        },
        Index(ix) => IrExpr::Index {
            base: Box::new(lower_expr(&ix.expr)),
            index: Box::new(lower_expr(&ix.index)),
        },
        Struct(s) => lower_struct_lit(s),
        Continue(c) => {
            if c.label.is_some() {
                IrExpr::Unsupported
            } else {
                IrExpr::Continue
            }
        }
        Break(b) => {
            if b.label.is_some() || b.expr.is_some() {
                IrExpr::Unsupported
            } else {
                IrExpr::Break
            }
        }
        // Rust casts are type-coercion hints for the DSL; WGSL infers the
        // type from context, so the cast itself is dropped.
        Cast(c) => lower_expr(&c.expr),
        Return(r) => IrExpr::Return(r.expr.as_ref().map(|e| Box::new(lower_expr(e)))),
        _ => IrExpr::Verbatim(
            syn::Error::new_spanned(e, "expression not supported in WGSL kernel")
                .to_compile_error()
                .to_string(),
        ),
    }
}

/// Lower a whole body: statements in order, last one flagged as tail.
pub fn lower_body(stmts: &[Stmt]) -> IrBlock {
    let count = stmts.len();
    stmts
        .iter()
        .enumerate()
        .flat_map(|(i, s)| lower_stmt(s, i == count - 1))
        .collect()
}

/// Lower one statement (a `let = if` expands to two).
pub fn lower_stmt(s: &Stmt, is_tail: bool) -> Vec<IrStmt> {
    match s {
        Stmt::Local(local) => lower_local(local),
        Stmt::Expr(syn::Expr::ForLoop(f), _) => lower_for_range(f),
        Stmt::Expr(expr, semi) => match expr {
            syn::Expr::If(_) | syn::Expr::Block(_) => vec![IrStmt::Flow {
                expr: lower_expr(expr),
                semi: semi.is_some(),
            }],
            _ => vec![IrStmt::Expr {
                expr: lower_expr(expr),
                semi: semi.is_some(),
                is_tail,
            }],
        },
        // Loud by construction: anything untranslatable emits an unknown
        // call that naga rejects, instead of vanishing silently and
        // changing shader semantics. Pixel probes backstop drift.
        _ => vec![unsupported_stmt()],
    }
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

fn lower_call(c: &syn::ExprCall) -> IrExpr {
    let args: Vec<IrExpr> = c.args.iter().map(lower_expr).collect();
    if let syn::Expr::Path(p) = c.func.as_ref() {
        let segs: Vec<_> = p.path.segments.iter().collect();
        let ident = segs.last().map(|s| s.ident.to_string()).unwrap_or_default();
        // Constructors: glam::Vec3::new(a,b,c) / Vec3::splat(1.0).
        if (ident == "new" || ident == "splat")
            && let Some(type_seg) = segs.iter().rev().nth(1)
            && let Some(ty) =
                ShaderType::from_rust(&type_seg.ident.to_string()).map(|t| t.wgsl().to_string())
        {
            return IrExpr::Call {
                target: IrCallee::Constructor { wgsl_ty: ty },
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

fn lower_if(i: &syn::ExprIf) -> IrExpr {
    let cond = Box::new(lower_expr(&i.cond));
    let then = lower_block(&i.then_branch);
    let els = i
        .else_branch
        .as_ref()
        .map(|(_, else_expr)| match else_expr.as_ref() {
            syn::Expr::Block(blk) => IrElse::Block(lower_block(&blk.block)),
            syn::Expr::If(inner_if) => IrElse::If(Box::new(lower_if(inner_if))),
            _ => IrElse::Block(vec![]),
        });
    IrExpr::If { cond, then, els }
}

fn lower_block(block: &Block) -> IrBlock {
    lower_body(&block.stmts)
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
    let (name, is_mut) = match &local.pat {
        syn::Pat::Ident(pi) => (pi.ident.to_string(), pi.mutability.is_some()),
        syn::Pat::Type(pt) => match pt.pat.as_ref() {
            syn::Pat::Ident(pi) => (pi.ident.to_string(), pi.mutability.is_some()),
            _ => (pat_name(&local.pat), false),
        },
        _ => (pat_name(&local.pat), false),
    };
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
