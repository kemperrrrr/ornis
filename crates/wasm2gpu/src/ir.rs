//! Straight-line wasm subset the lowerer accepts.

/// How many wasm parameters a toy signature may name.
///
/// `texture_sample_2d` has six operands and `storage_write` has seven.
/// Defined shader entries stay smaller (vertex: four `f32`, fragment: at most four).
pub const MAX_PARAMS: usize = 7;

/// A wasm value the toy subset accepts.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Wasm `i32`. Lowered as `u32` when it is an element index.
    I32,
    /// Wasm `f32`.
    F32,
}

/// One imported or defined function's signature, counted in values.
#[derive(Clone, Copy)]
pub struct Sig {
    /// Parameter count.
    pub params: u32,
    /// Result count.
    pub results: u32,
    /// `params` entries of [`Kind`], in wasm order. The tail is unused.
    pub param_kind: [Kind; MAX_PARAMS],
}

/// An `ornis_gpu` import.
pub struct Import {
    /// Import name (`raw_read`, `texture_sample_2d`, …).
    pub name: String,
    /// Type signature.
    pub sig: Sig,
}

/// A defined function, possibly exported.
pub struct Func {
    /// Export name. Empty when the function is not exported.
    pub export: String,
    /// Type signature. Vertex entries are `f32`. A fragment may take one `i32` index.
    pub sig: Sig,
    /// Body, not including a final `end`.
    pub ops: Vec<Op>,
}

/// Operators the lowerer knows.
#[derive(Clone, Copy)]
pub enum Op {
    /// Immediate `i32` (group, binding, or byte offset).
    I32(u32),
    /// `f32` bit pattern.
    F32(u32),
    /// `f32` parameter, or a local that is not an `i32` parameter.
    Local(u32),
    /// `i32` parameter. Used as an element index, never as a binding.
    LocalI32(u32),
    /// Call import `index`.
    Call(u32),
    /// Multiply the top two values.
    Mul,
    /// Add the top two values.
    Add,
}

/// Imports plus defined functions.
pub struct Module {
    /// Function imports, in index order.
    pub imports: Vec<Import>,
    /// Defined functions. Function index is `imports.len() + position`.
    pub funcs: Vec<Func>,
}
