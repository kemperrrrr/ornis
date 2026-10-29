//! Scratch state while a module is lowered.

use naga::Expression;
use naga::Type;

use crate::ir::Kind;
use crate::ir::MAX_PARAMS;

pub(super) const VS_PARAMS: u32 = 4;
pub(super) const VS_RESULTS: u32 = 6;
pub(super) const FS_PARAMS: u32 = 2;
pub(super) const FS_RESULTS: u32 = 4;
pub(super) const LANES: u32 = 4;
/// Largest uniform binding this translator will emit. WebGPU's limit is 64 KiB.
pub(super) const UNIFORM_BYTES: u32 = 1 << 16;
pub(super) const F32_WIDTH: u8 = 4;
pub(super) const VEC3_LANES: u32 = 3;

pub(super) struct Slot {
    pub(super) group: u32,
    pub(super) binding: u32,
    pub(super) var: naga::Handle<naga::GlobalVariable>,
}

/// Shape of one uniform binding. Two shapes cannot share a group and binding.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum UniformShape {
    /// The binding is the element itself (a `vec4`, or one struct).
    Bare(crate::ty::Elem),
    /// Fixed-size `array<elem, len>` in the uniform address space.
    Array {
        /// Element type.
        elem: crate::ty::Elem,
        /// Constant length. Never zero.
        len: u32,
    },
}

pub(super) struct UniformSlot {
    pub(super) group: u32,
    pub(super) binding: u32,
    pub(super) shape: UniformShape,
    pub(super) var: naga::Handle<naga::GlobalVariable>,
}

pub(super) struct StorageSlot {
    pub(super) group: u32,
    pub(super) binding: u32,
    pub(super) elem: crate::ty::Elem,
    pub(super) var: naga::Handle<naga::GlobalVariable>,
}

pub(super) struct Cx<'a> {
    pub(super) module: &'a mut naga::Module,
    pub(super) scalar: naga::Handle<Type>,
    pub(super) uint: naga::Handle<Type>,
    pub(super) sint: naga::Handle<Type>,
    pub(super) bool_ty: naga::Handle<Type>,
    pub(super) vec2: naga::Handle<Type>,
    pub(super) vec3: naga::Handle<Type>,
    pub(super) vec4: naga::Handle<Type>,
    pub(super) uniforms: Vec<UniformSlot>,
    pub(super) storages: Vec<StorageSlot>,
    pub(super) textures: Vec<Slot>,
    pub(super) samplers: Vec<Slot>,
    pub(super) struct_tys: Vec<(u16, naga::Handle<Type>)>,
    /// Specialized naga functions already emitted for defined wasm calls.
    pub(super) specs: Vec<Spec>,
    /// Defined-function indices currently being lowered. A repeat is recursion.
    pub(super) active: Vec<u32>,
}

/// One specialization of a defined wasm function.
///
/// `baked[i]` is `Some` when parameter `i` was an `i32` immediate at the
/// call. Those immediates are not naga parameters: the callee sees them as
/// constants, so a group or binding stays an immediate. A `None` slot is a
/// real argument (a dynamic index, or any `f32`).
#[derive(Clone, Copy)]
pub(super) struct Spec {
    /// Index into [`crate::ir::Module::funcs`].
    pub func: u32,
    /// Immediate parameters. The tail past `nparams` is unused.
    pub baked: [Option<u32>; MAX_PARAMS],
    /// Wasm parameter count.
    pub nparams: u8,
    /// Lowered naga function.
    pub handle: naga::Handle<naga::Function>,
    /// `Some` when result `i` folded to an immediate. `None` is a runtime value.
    pub result_imm: [Option<u32>; MAX_PARAMS],
    /// Result count.
    pub nresults: u8,
    /// Result kinds, so the caller can push `f32` and `i32` back.
    pub result_kind: [Kind; MAX_PARAMS],
}

#[derive(Clone, Copy)]
pub(super) enum Val {
    Imm(u32),
    Scalar(naga::Handle<Expression>),
    /// Runtime `u32` bits. An element index, or any other wasm `i32`.
    Index(naga::Handle<Expression>),
    /// One `vec4<f32>` from `raw_read`, `texture_sample_2d`, or `storage_read`.
    Vec(naga::Handle<Expression>),
}

pub(super) struct Stage {
    pub(super) name: &'static str,
    pub(super) params: u32,
    pub(super) results: u32,
    pub(super) vertex: bool,
}

pub(super) fn location(index: u32) -> naga::Binding {
    naga::Binding::Location {
        location: index,
        interpolation: Some(naga::Interpolation::Perspective),
        sampling: Some(naga::Sampling::Center),
        blend_src: None,
        per_primitive: false,
    }
}

pub(super) fn expr(body: &mut naga::Function, value: Expression) -> naga::Handle<Expression> {
    let mut block = std::mem::take(&mut body.body);
    let handle = expr_in(body, &mut block, value);
    body.body = block;
    handle
}

pub(super) fn expr_in(
    func: &mut naga::Function,
    block: &mut naga::Block,
    value: Expression,
) -> naga::Handle<Expression> {
    let handle = func.expressions.append(value, naga::Span::default());
    if !func.expressions[handle].needs_pre_emit() {
        let range = func.expressions.range_from(handle.index());
        block.push(naga::Statement::Emit(range), naga::Span::default());
    }
    handle
}

pub(super) fn sp() -> naga::Span {
    naga::Span::default()
}
