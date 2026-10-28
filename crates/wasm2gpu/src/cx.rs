//! Scratch state while a module is lowered.

use naga::Expression;
use naga::Type;

use crate::ir::Import;

pub(super) const VS_PARAMS: u32 = 4;
pub(super) const VS_RESULTS: u32 = 6;
pub(super) const FS_PARAMS: u32 = 2;
pub(super) const FS_RESULTS: u32 = 4;
pub(super) const LANES: u32 = 4;
pub(super) const READ_PARAMS: u32 = 3;
pub(super) const SAMPLE_PARAMS: u32 = 6;
pub(super) const STORAGE_WRITE_PARAMS: u32 = 7;
pub(super) const F32_WIDTH: u8 = 4;
/// Byte stride of `array<vec4<f32>>` elements. One access-chain index, not a packed pointer.
pub(super) const VEC4_STRIDE: u32 = 16;

pub(super) struct Slot {
    pub(super) group: u32,
    pub(super) binding: u32,
    pub(super) var: naga::Handle<naga::GlobalVariable>,
}

pub(super) struct Cx<'a> {
    pub(super) module: &'a mut naga::Module,
    pub(super) scalar: naga::Handle<Type>,
    pub(super) uint: naga::Handle<Type>,
    pub(super) vec2: naga::Handle<Type>,
    pub(super) vec4: naga::Handle<Type>,
    pub(super) vec4_array: naga::Handle<Type>,
    pub(super) uniforms: Vec<Slot>,
    pub(super) storages: Vec<Slot>,
    pub(super) textures: Vec<Slot>,
    pub(super) samplers: Vec<Slot>,
}

pub(super) enum Val {
    Imm(u32),
    Scalar(naga::Handle<Expression>),
    /// Runtime element index (`u32`). Not a group or a binding.
    Index(naga::Handle<Expression>),
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
    let handle = body.expressions.append(value, naga::Span::default());
    if !body.expressions[handle].needs_pre_emit() {
        let range = body.expressions.range_from(handle.index());
        body.body
            .push(naga::Statement::Emit(range), naga::Span::default());
    }
    handle
}

pub(super) fn import(
    imports: &[Import],
    index: u32,
) -> Result<&Import, crate::error::TranslateError> {
    imports
        .get(index as usize)
        .ok_or(crate::error::TranslateError::Unsupported)
}
