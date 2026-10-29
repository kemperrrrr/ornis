//! Wasm subset the lowerer accepts: straight-line ops plus structured control.

/// How many wasm parameters a signature may name.
///
/// Wide enough for a struct store: four provenance immediates plus the
/// flattened lanes of a small struct. Defined entries stay smaller.
pub const MAX_PARAMS: usize = {
    const WIDTH: usize = 12;
    WIDTH
};

/// A wasm value the subset accepts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
    /// `results` entries of [`Kind`], in wasm order. The tail is unused.
    pub result_kind: [Kind; MAX_PARAMS],
}

/// Block parameters and results for `block` / `loop` / `if`.
#[derive(Clone, Debug)]
pub struct BlockTy {
    /// Values the block pops from the outer stack.
    pub params: Vec<Kind>,
    /// Values the block pushes back.
    pub results: Vec<Kind>,
}

/// An `ornis_gpu` import.
pub struct Import {
    /// Import name (`raw_read`, `storage_read_f32`, …).
    pub name: String,
    /// Type signature.
    pub sig: Sig,
}

/// A defined function, possibly exported.
pub struct Func {
    /// Export name. Empty when the function is not exported.
    pub export: String,
    /// Type signature.
    pub sig: Sig,
    /// Locals, including parameters, in wasm index order.
    pub locals: Vec<Kind>,
    /// Structured body. The function's final `end` is not included.
    pub body: Vec<Stmt>,
}

/// Operators the lowerer knows. Control flow is a [`Stmt`], not an [`Op`].
#[derive(Clone, Copy, Debug)]
pub enum Op {
    /// Immediate `i32` (group, binding, element index, or struct id).
    I32(u32),
    /// `f32` bit pattern.
    F32(u32),
    /// `f32` local.
    Local(u32),
    /// `i32` local. An element index, or a condition. Not a binding by itself.
    LocalI32(u32),
    /// Write the top value into a local.
    LocalSet(u32),
    /// Write the top value and leave it on the stack.
    LocalTee(u32),
    /// Pop one value.
    Drop,
    /// Call function `index`. Imports occupy the low indices; a defined
    /// function is `imports.len() + position`.
    Call(u32),
    /// Multiply the top two `f32` values.
    Mul,
    /// Add the top two `f32` values.
    Add,
    /// Wrapping `i32` add.
    I32Add,
    /// Wrapping `i32` multiply.
    I32Mul,
    /// Wrapping `i32` subtract.
    I32Sub,
    /// `i32.eqz`.
    I32Eqz,
    /// `i32.eq`.
    I32Eq,
    /// `i32.ne`.
    I32Ne,
    /// Unsigned `i32.lt_u`.
    I32LtU,
    /// Unsigned `i32.gt_u`.
    I32GtU,
    /// Wasm `select`: `v1` if the condition is non-zero, otherwise `v2`.
    Select,
}

/// Structured statement. `br` targets are relative wasm label depths.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// One stack operator.
    Op(Op),
    /// `block` … `end`. A branch to this label exits the block.
    Block {
        /// Block type.
        ty: BlockTy,
        /// Body.
        body: Vec<Stmt>,
    },
    /// `loop` … `end`. A branch to this label restarts the loop.
    Loop {
        /// Block type. Parameters are the values carried by `br` to the header.
        ty: BlockTy,
        /// Body.
        body: Vec<Stmt>,
    },
    /// `if` … `else` … `end`. The condition is the top `i32`.
    If {
        /// Block type.
        ty: BlockTy,
        /// Then branch.
        then_body: Vec<Stmt>,
        /// Else branch. Empty when the wasm `if` has no `else`.
        else_body: Vec<Stmt>,
    },
    /// Unconditional branch to relative label `depth`.
    Br(u32),
    /// Branch to relative label `depth` when the top `i32` is non-zero.
    BrIf(u32),
    /// `return`. Void entries become a naga `return`. A value-returning
    /// `return` is rejected: it would have to compose shader outputs.
    Return,
    /// `br_table`. `targets[i]` is the relative depth for index `i`.
    ///
    /// The index is the top `i32`. An index past `targets` uses `default`.
    /// Both are relative wasm label depths, same as [`Stmt::Br`].
    BrTable {
        /// Arm depths, in index order. May be empty.
        targets: Vec<u32>,
        /// Depth used when the index is outside `targets`.
        default: u32,
    },
}

/// Imports, struct layouts, and defined functions.
pub struct Module {
    /// Function imports, in index order.
    pub imports: Vec<Import>,
    /// Structs from the `ornis_gpu.types` custom section, in id order.
    pub structs: Vec<crate::ty::StructDef>,
    /// Defined functions. Function index is `imports.len() + position`.
    pub funcs: Vec<Func>,
}
