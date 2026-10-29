//! Uniform globals. One variable per group and binding.
//!
//! A bare binding is the value itself. An array is wrapped in a struct
//! because a WGSL uniform variable must be a structure. An element whose
//! natural stride is tighter than the uniform stride sits in a lane struct,
//! so byte 0 is still the value the host wrote.

use naga::ArraySize;
use naga::Span;
use naga::StructMember;
use naga::Type;
use naga::TypeInner;

use crate::cx::Cx;
use crate::cx::UniformShape;
use crate::cx::UniformSlot;
use crate::error::TranslateError;
use crate::globals;
use crate::ty::Elem;
use crate::ty::StructDef;

const MAX_LANE_FIELDS: u32 = 8;

pub(super) fn uniform(
    cx: &mut Cx<'_>,
    group: u32,
    binding: u32,
) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
    uniform_global(cx, group, binding, UniformShape::Bare(Elem::Vec4), &[])
}

pub(super) fn uniform_global(
    cx: &mut Cx<'_>,
    group: u32,
    binding: u32,
    shape: UniformShape,
    structs: &[StructDef],
) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
    if globals::storage_taken(&cx.storages, group, binding) {
        return Err(type_msg("binding is used as both uniform and storage"));
    }
    if let Some(slot) = find_uniform(&cx.uniforms, group, binding) {
        if slot.shape != shape {
            return Err(type_msg("binding is used as two uniform types"));
        }
        return Ok(slot.var);
    }
    let ty = uniform_ty(cx, shape, structs)?;
    let var = globals::append(
        cx,
        format!("uniform_{group}_{binding}"),
        naga::AddressSpace::Uniform,
        ty,
        globals::point(group, binding),
    );
    cx.uniforms.push(UniformSlot {
        group,
        binding,
        shape,
        var,
    });
    Ok(var)
}

pub(super) fn uniform_taken(slots: &[UniformSlot], group: u32, binding: u32) -> bool {
    find_uniform(slots, group, binding).is_some()
}

fn uniform_ty(
    cx: &mut Cx<'_>,
    shape: UniformShape,
    structs: &[StructDef],
) -> Result<naga::Handle<Type>, TranslateError> {
    match shape {
        UniformShape::Bare(elem) => globals::base_ty(cx, elem, structs),
        UniformShape::Array { elem, len } => wrapped_array(cx, elem, structs, len),
    }
}

fn wrapped_array(
    cx: &mut Cx<'_>,
    elem: Elem,
    structs: &[StructDef],
    len: u32,
) -> Result<naga::Handle<Type>, TranslateError> {
    let stride = crate::ty::uniform_elem_stride(elem, structs)?;
    let bytes = stride
        .checked_mul(len)
        .ok_or_else(|| type_msg("uniform binding exceeds 64KiB"))?;
    if bytes > crate::cx::UNIFORM_BYTES {
        return Err(type_msg("uniform binding exceeds 64KiB"));
    }
    let base = array_elem_ty(cx, elem, structs, stride)?;
    let array = fixed_array(cx, base, len, stride)?;
    // The array is the only member, at offset 0, so host strides still
    // address element 0 at byte 0.
    Ok(cx.module.types.insert(
        Type {
            name: Some(format!("U{len}")),
            inner: TypeInner::Struct {
                members: vec![StructMember {
                    name: Some("a".to_string()),
                    ty: array,
                    binding: None,
                    offset: 0,
                }],
                span: bytes,
            },
        },
        Span::default(),
    ))
}

fn array_elem_ty(
    cx: &mut Cx<'_>,
    elem: Elem,
    structs: &[StructDef],
    stride: u32,
) -> Result<naga::Handle<Type>, TranslateError> {
    if !crate::ty::uniform_needs_lane(elem, structs)? {
        return globals::base_ty(cx, elem, structs);
    }
    let layout = crate::ty::elem_layout(elem, structs)?;
    let field = globals::base_ty(cx, elem, structs)?;
    let mut members = Vec::new();
    let mut offset = 0u32;
    let mut index = 0u32;
    while offset < stride {
        members.push(StructMember {
            name: Some(lane_name(index)),
            ty: field,
            binding: None,
            offset,
        });
        let next = offset
            .checked_add(layout.size)
            .ok_or_else(|| type_msg("uniform binding exceeds 64KiB"))?;
        offset = ornis_gpu_abi::round_up(next, layout.align);
        index += 1;
        if index > MAX_LANE_FIELDS {
            return Err(type_msg("uniform array element cannot be padded"));
        }
    }
    if offset != stride {
        return Err(type_msg("uniform array element cannot be padded"));
    }
    Ok(cx.module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Struct {
                members,
                span: stride,
            },
        },
        Span::default(),
    ))
}

fn lane_name(index: u32) -> String {
    if index == 0 {
        "v".to_string()
    } else {
        format!("p{index}")
    }
}

fn fixed_array(
    cx: &mut Cx<'_>,
    base: naga::Handle<Type>,
    len: u32,
    stride: u32,
) -> Result<naga::Handle<Type>, TranslateError> {
    let size = std::num::NonZeroU32::new(len)
        .ok_or_else(|| type_msg("uniform array length must be positive"))?;
    Ok(cx.module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Array {
                base,
                size: ArraySize::Constant(size),
                stride,
            },
        },
        Span::default(),
    ))
}

fn find_uniform(slots: &[UniformSlot], group: u32, binding: u32) -> Option<&UniformSlot> {
    let mut index = 0usize;
    while index < slots.len() {
        if slots[index].group == group && slots[index].binding == binding {
            return Some(&slots[index]);
        }
        index += 1;
    }
    None
}

fn type_msg(msg: &str) -> TranslateError {
    TranslateError::UnsupportedType(msg.to_string())
}
