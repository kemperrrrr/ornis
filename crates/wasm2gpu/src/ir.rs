//! Straight-line wasm subset the lowerer accepts.

/// One imported or defined function's signature, counted in values.
#[derive(Clone, Copy)]
pub struct Sig {
    /// Parameter count.
    pub params: u32,
    /// Result count.
    pub results: u32,
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
    /// Type signature. Toy shaders use only `f32` values.
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
    /// Parameter index.
    Local(u32),
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
