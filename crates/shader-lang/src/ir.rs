//! Shader IR: structural nodes between the Rust AST and WGSL text.
//!
//! The translator lowers `syn` expressions/statements to these nodes
//! ([`IrExpr`]/[`IrStmt`]) and a separate writer renders them to WGSL.
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

/// Component counts of the registry's vector and matrix spellings.
/// Named rather than written inline so constructor arithmetic reads as
/// dimensions — the magic-number allow-list only covers `0`/`1`/`2`/`-1`.
const VEC3_DIM: usize = 3;
/// Component count of a glam / WGSL `Vec4`.
const VEC4_DIM: usize = 4;
/// Component count of a glam / WGSL `Mat3` (square, row-major).
const MAT3_DIM: usize = 9;
/// Component count of a glam / WGSL `Mat4` (square, row-major).
const MAT4_DIM: usize = 16;

/// Zero component spellings per scalar kind, shared by every constructor
/// [`IrType::zero_expr`] expands to.
const F32_ZERO: &str = "0.0";
/// Zero component spelling for `i32`.
const I32_ZERO: &str = "0i";
/// Zero component spelling for `u32`.
const U32_ZERO: &str = "0u";

/// Opaque user-defined shader type (struct mirrors): a lexical name that is
/// neither a [`ShaderType`] primitive nor `bool`.
///
/// Newtype enum (not a bare `String`) so custom-type names never mix with
/// WGSL spellings, field names or call targets at the type level. The WGSL
/// spelling is the name itself (see `writer::print_ty`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderCustomTy {
    /// Named struct mirror (e.g. `CameraUniform`, `VertexOutput`).
    Named(Box<str>),
}

impl ShaderCustomTy {
    /// Wraps a custom type name.
    pub fn new(name: impl Into<Box<str>>) -> Self {
        Self::Named(name.into())
    }

    /// Custom type name as a string slice.
    pub fn as_str(&self) -> &str {
        match self {
            ShaderCustomTy::Named(name) => name,
        }
    }
}

impl std::fmt::Display for ShaderCustomTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for ShaderCustomTy {
    fn from(name: &str) -> Self {
        Self::Named(name.into())
    }
}

impl From<String> for ShaderCustomTy {
    fn from(name: String) -> Self {
        Self::Named(name.into_boxed_str())
    }
}

impl From<&String> for ShaderCustomTy {
    fn from(name: &String) -> Self {
        Self::Named(name.as_str().into())
    }
}

/// A lowered type: what the value IS, not how it prints. Scalar
/// spellings come from the [`ShaderType`] registry; `Bool` is separate
/// because the registry deliberately excludes it (call classification,
/// not printing); arrays compose structurally so the writer owns the
/// single `array<T, N>` spelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrType {
    Scalar(ShaderType),
    Bool,
    /// Opaque named type (struct mirrors): lexical, like [`IrExpr::Path`]
    /// segments — never a spelling.
    Custom(ShaderCustomTy),
    Array {
        elem: Box<IrType>,
        len: usize,
    },
    /// Runtime-sized `array<T>` (storage buffers document capacity in
    /// Rust, WGSL sizes at bind time).
    RuntimeArray(Box<IrType>),
    /// `Option<T>` for a registry scalar/vector/bool: lowered to a
    /// generated two-field struct (see `writer::option_struct_decl`).
    /// Nested `Option` is rejected upstream — WGSL has no recursive
    /// struct layout.
    Optional(Box<IrType>),
    /// The DSL entity handle (`index: u32, generation: u32`): a
    /// generated struct (see `writer::entity_struct_decl`), spelled
    /// `Entity` in WGSL and mirrored by the per-kernel CPU struct the
    /// macro emits.
    Entity,
}

impl IrType {
    /// Zero-valued literal for this type: the payload placeholder for
    /// `None`. Vectors and matrices expand to positional constructors
    /// filled with component zeros; [`IrType::Entity`] expands to the
    /// null handle `Entity(0u, 0u)`; anything without a literal spelling
    /// (`Custom`, arrays, nested `Optional`) is [`IrExpr::Unsupported`].
    ///
    /// Scalar dispatch and the component table live in the same body, so
    /// the function stays pure logic — no project call next to the
    /// branching, which is what lets the `None` placeholder be reasoned
    /// about on its own.
    pub fn zero_expr(&self) -> IrExpr {
        match self {
            IrType::Scalar(ShaderType::F32) => IrExpr::Float(F32_ZERO.to_string()),
            IrType::Scalar(ShaderType::I32) => IrExpr::Int(I32_ZERO.to_string()),
            IrType::Scalar(ShaderType::U32) => IrExpr::Int(U32_ZERO.to_string()),
            IrType::Bool => IrExpr::Bool(false),
            // The null handle: positional constructor over the two `u32`
            // fields, the same spelling a struct literal lowers to.
            IrType::Entity => IrExpr::Call {
                target: IrCallee::Expr(Box::new(IrExpr::Path(vec!["Entity".to_string()]))),
                args: vec![
                    IrExpr::Int(U32_ZERO.to_string()),
                    IrExpr::Int(U32_ZERO.to_string()),
                ],
            },
            IrType::Scalar(t) => {
                // Component count and that component's zero literal, in one
                // table: `Mat3`/`Mat4` are square float matrices, scalars
                // never reach here (the arms above take them).
                let (count, zero) = match t {
                    ShaderType::Vec2 => (2, IrExpr::Float(F32_ZERO.to_string())),
                    ShaderType::Vec3 => (VEC3_DIM, IrExpr::Float(F32_ZERO.to_string())),
                    ShaderType::Vec4 => (VEC4_DIM, IrExpr::Float(F32_ZERO.to_string())),
                    ShaderType::UVec2 => (2, IrExpr::Int(U32_ZERO.to_string())),
                    ShaderType::UVec3 => (VEC3_DIM, IrExpr::Int(U32_ZERO.to_string())),
                    ShaderType::UVec4 => (VEC4_DIM, IrExpr::Int(U32_ZERO.to_string())),
                    ShaderType::IVec2 => (2, IrExpr::Int(I32_ZERO.to_string())),
                    ShaderType::IVec3 => (VEC3_DIM, IrExpr::Int(I32_ZERO.to_string())),
                    ShaderType::IVec4 => (VEC4_DIM, IrExpr::Int(I32_ZERO.to_string())),
                    ShaderType::BVec2 => (2, IrExpr::Bool(false)),
                    ShaderType::BVec3 => (VEC3_DIM, IrExpr::Bool(false)),
                    ShaderType::BVec4 => (VEC4_DIM, IrExpr::Bool(false)),
                    ShaderType::Mat3 => (MAT3_DIM, IrExpr::Float(F32_ZERO.to_string())),
                    ShaderType::Mat4 => (MAT4_DIM, IrExpr::Float(F32_ZERO.to_string())),
                    // A lone scalar reads as a one-element constructor,
                    // which naga still accepts.
                    _ => (1, IrExpr::Float(F32_ZERO.to_string())),
                };
                IrExpr::Call {
                    target: IrCallee::Constructor { ty: *t },
                    args: std::iter::repeat_n(zero, count).collect(),
                }
            }
            _ => IrExpr::Unsupported,
        }
    }
}

/// Module-scope declaration: what the item IS, not how it prints.
/// Expression-level [`IrExpr`] never carries these spellings; the
/// writer prints items and bodies from the same nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrItem {
    Global(IrGlobal),
    Struct {
        name: String,
        fields: Vec<IrStructField>,
    },
}

/// A module-scope `var`: address space and type travel structurally,
/// `@group`/`@binding` numbers stay data (today always group 0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrGlobal {
    Storage {
        group: u32,
        binding: u32,
        name: String,
        elem: IrType,
        read_only: bool,
    },
    Uniform {
        group: u32,
        binding: u32,
        name: String,
        ty: IrType,
    },
    Texture {
        group: u32,
        binding: u32,
        name: String,
        kind: IrTexture,
    },
    Sampler {
        group: u32,
        binding: u32,
        name: String,
        comparison: bool,
    },
    /// Module-scope `var<private>` with initializer (the only mutable
    /// module scope; also the address space for lookup tables: `const`
    /// arrays reject dynamic indices, `private` allows them).
    Private {
        name: String,
        ty: IrType,
        init: IrExpr,
    },
}

/// Closed texture world (mirrors the render `ResourceKind` set): a
/// free-form token paste cannot produce a spelling the validator never
/// sees — unknown shapes are loud at parse, not at naga.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrTexture {
    Tex2dF32,
    Tex2dU32,
    Tex2dI32,
    Depth2d,
    Depth2dArray,
    DepthCubeArray,
}

/// One `struct` field with its WGSL attribute, if any.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrStructField {
    pub attr: Option<IrFieldAttr>,
    pub name: String,
    pub ty: IrType,
}

/// Field attributes: a builtin semantic (lexical ident) or a location
/// number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrFieldAttr {
    Builtin(String),
    Location(u32),
}

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
    /// `e as T`: lowered to a WGSL conversion-constructor call at
    /// lowering time (`f32(x)`/`u32(x)`/`i32(x)` — exactly the 32-bit
    /// scalars; narrower/wider targets are rejected at macro time), so
    /// this node no longer occurs in macro output; the writer keeps the
    /// historical transparent spelling as a fallback. In-range values
    /// truncate like Rust `as`; out-of-range float→int (Rust saturates,
    /// NaN→0) is indeterminate in WGSL — kernel inputs must stay in range.
    Cast(Box<IrExpr>),
    /// Fixed-size array value with a structural element type:
    /// `array<f32, 6>(0.0, …)`. Built from Rust `[T; N]` literals and
    /// repeats (see lowering); the writer derives `N` from the items.
    Array {
        elem: IrType,
        items: Vec<IrExpr>,
    },
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
    /// A type constructor (`Vec3::new` / `Mat3::from_cols` classified at
    /// lowering): the type travels as a registry variant, spelled by the
    /// writer — never a pre-rendered string.
    Constructor { ty: ShaderType },
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
