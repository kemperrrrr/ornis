//! Shader IR: structural nodes between the Rust AST and WGSL text.
//!
//! The translator lowers `syn` expressions/statements to these nodes
//! ([`IrExpr`]/[`IrStmts`]) and a separate writer renders them to WGSL.
//! The split exists so translation is testable structurally (match on
//! nodes, not on strings) and future passes (validation, folding) have
//! something to consume besides text. Node shapes mirror the current
//! language subset one-to-one — including the loud-failure contract:
//! untranslatable syntax lowers to [`IrExpr::Unsupported`], which the
//! writer prints as a call naga rejects (`no definition in scope`),
//! never as silently dropped code.

use crate::{ShaderBuiltin, ShaderType};

/// Marker call for untranslatable syntax: guaranteed to fail naga
/// (`no definition in scope`), never to pass silently.
pub const UNSUPPORTED_MARKER: &str = "__wgsl_dsl_unsupported_statement__()";

/// A lowered expression: what the value IS, not how it prints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrExpr {
    /// Integer literal, pre-rendered (`42`, `1u`, `0xFF`): suffix mapping
    /// is lexical token work done at lowering, not WGSL structure.
    Int(String),
    /// Float literal, pre-rendered (`1.0`, `2e-3`): the `.0` fix is lexical.
    Float(String),
    /// `true` / `false`.
    Bool(bool),
    /// Dotted path as segments (`["quad"]`, `["glam", "Vec3", "ZERO"]`):
    /// the writer resolves vector constants and type aliases.
    Path(Vec<String>),
    /// `base.member` (swizzles, struct fields, `.x`).
    Field {
        base: Box<IrExpr>,
        member: String,
    },
    /// `base[index]`.
    Index {
        base: Box<IrExpr>,
        index: Box<IrExpr>,
    },
    /// Source-level parentheses (preserved verbatim: the writer adds no
    /// precedence parens of its own).
    Paren(Box<IrExpr>),
    /// `left op right`, including compound assignment (`+=` etc. — syn
    /// parses those as binary expressions with an assign-op).
    Binary {
        op: IrBinOp,
        left: Box<IrExpr>,
        right: Box<IrExpr>,
    },
    /// `-x` / `!c` / `*p` (deref and unknown prefix ops both spell `*`,
    /// matching the translator's historical behavior).
    Unary {
        op: IrUnOp,
        expr: Box<IrExpr>,
    },
    /// `left = right` as an expression (WGSL has no assign-expressions;
    /// the translator historically emits it inline — preserved).
    Assign {
        left: Box<IrExpr>,
        right: Box<IrExpr>,
    },
    /// Any call: the callee is classified at lowering (constructors and
    /// built-ins resolve through the registry; kernels/helpers stay
    /// verbatim; exotic callee shapes stay structural).
    Call {
        target: IrCallee,
        args: Vec<IrExpr>,
    },
    /// Unrecognized `recv.method(args)`: WGSL has no methods, so this only
    /// survives for shapes naga will reject loudly (same as before).
    Method {
        base: Box<IrExpr>,
        method: String,
        args: Vec<IrExpr>,
    },
    /// Struct literal → positional constructor; the writer appends the
    /// `/* field, names */` comment so the field-order test can verify
    /// declaration order instead of trusting it blindly.
    Struct {
        name: String,
        fields: Vec<(String, IrExpr)>,
    },
    /// `if (c) { .. } [else ..]` as an expression.
    If {
        cond: Box<IrExpr>,
        then: Vec<IrStmt>,
        els: Option<IrElse>,
    },
    /// `{ stmts }` as an expression.
    Block(Vec<IrStmt>),
    /// `return [expr]` in expression position.
    Return(Option<Box<IrExpr>>),
    /// `continue` / `break` (labels/values lower to [`IrExpr::Unsupported`]).
    Continue,
    Break,
    /// `e as T`: a type-coercion hint for the DSL; WGSL infers the type
    /// from context, so the cast itself is dropped (transparent).
    Cast(Box<IrExpr>),
    /// Pre-rendered escape hatch: `syn::Error::to_compile_error` output
    /// for truly unsupported nodes (carries the message into the WGSL
    /// text, where naga reports it). No new uses — prefer [`IrExpr::Unsupported`].
    Verbatim(String),
    /// Untranslatable syntax with no message to carry: the writer prints
    /// [`UNSUPPORTED_MARKER`].
    Unsupported,
}

/// Binary operators of the supported subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Shl,
    Shr,
    BitXor,
    BitAnd,
    BitOr,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    ShlAssign,
    ShrAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
}

/// Prefix operators of the supported subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrUnOp {
    Neg,
    Not,
    /// `*p` — and any other unrecognized prefix op (historical behavior).
    DerefOrUnknown,
}

/// A classified call target: resolved once at lowering, printed dumbly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrCallee {
    /// `glam::Vec3::new(..)` / `Vec3::splat(..)` → `vec3<f32>(..)`.
    /// The WGSL spelling resolves at lowering (name mapping); the writer
    /// only applies it.
    Constructor { wgsl_ty: String },
    /// A registry built-in: applied by the writer via [`ShaderBuiltin::lower`].
    Builtin(ShaderBuiltin),
    /// Kernels/helpers: WGSL shares the spelling by design.
    Verbatim(String),
    /// Non-`Path` callee shape (`(f)(x)`): printed structurally.
    Expr(Box<IrExpr>),
}

/// The `else` arm of an [`IrExpr::If`]: a block or a chained `if`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrElse {
    Block(Vec<IrStmt>),
    If(Box<IrExpr>),
}

/// A lowered statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrStmt {
    /// `let`/`var` binding. `decl_ty` covers uninitialized declarations
    /// (`let mut out: T;` → `var out: T;`); `init` covers the rest.
    /// (`let x = if ..` expands to several statements at lowering.)
    Let {
        name: String,
        mutable: bool,
        init: Option<IrExpr>,
        decl_ty: Option<String>,
    },
    /// `for i in start..end` over `u32` (half-open or closed).
    For {
        var: String,
        start: IrExpr,
        end: IrExpr,
        closed: bool,
        body: Vec<IrStmt>,
    },
    /// `if` / block in statement position (with the source semicolon flag).
    Flow { expr: IrExpr, semi: bool },
    /// Any other expression as a statement (with the source semicolon flag
    /// and the tail flag that promotes a trailing value to `return`).
    Expr {
        expr: IrExpr,
        semi: bool,
        is_tail: bool,
    },
    /// A trailing block value promoted to `return ..;` at lowering.
    Tail(IrExpr),
}

/// A lowered function body: statements in order.
pub type IrBlock = Vec<IrStmt>;

/// Reference the registries so rustdoc links resolve.
#[allow(dead_code)]
fn _registry_links() {
    let _ = ShaderType::F32;
    let _ = ShaderBuiltin::Mix;
}
