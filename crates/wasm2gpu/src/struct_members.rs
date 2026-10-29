//! Naga struct types for `gpu_struct!` fields, including the host tail.
//!
//! The WGSL writer drops a struct span. Trailing `u32` members keep that
//! span, so a uniform array of `struct { f32 }` has stride 16.

use naga::Span;
use naga::StructMember;
use naga::Type;
use naga::TypeInner;

use crate::cx::Cx;
use crate::error::TranslateError;
use crate::ty::FieldKind;

pub(super) fn struct_ty(
    cx: &mut Cx<'_>,
    id: u16,
    fields: &[FieldKind],
) -> Result<naga::Handle<Type>, TranslateError> {
    if let Some(handle) = cached_struct(cx, id) {
        return Ok(handle);
    }
    let layout = crate::ty::struct_layout(fields)?;
    let members = padded_members(cx, fields, &layout)?;
    let handle = cx.module.types.insert(
        Type {
            name: Some(format!("S{id}")),
            inner: TypeInner::Struct {
                members,
                span: layout.size,
            },
        },
        Span::default(),
    );
    cx.struct_tys.push((id, handle));
    Ok(handle)
}

fn padded_members(
    cx: &Cx<'_>,
    fields: &[FieldKind],
    layout: &crate::ty::Layout,
) -> Result<Vec<StructMember>, TranslateError> {
    let mut out = members(cx, fields, &layout.offsets);
    let mut end = tail_start(fields, &layout.offsets)?;
    let mut pad = 0u32;
    while end < layout.size {
        let next = end
            .checked_add(ornis_gpu_abi::F32_SIZE)
            .ok_or_else(|| type_msg("struct span overflow"))?;
        if next > layout.size {
            return Err(type_msg("struct span is not a multiple of 4"));
        }
        out.push(StructMember {
            name: Some(format!("p{pad}")),
            ty: cx.uint,
            binding: None,
            offset: end,
        });
        end = next;
        pad += 1;
    }
    Ok(out)
}

fn members(cx: &Cx<'_>, fields: &[FieldKind], offsets: &[u32]) -> Vec<StructMember> {
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < fields.len() {
        out.push(StructMember {
            name: Some(format!("f{index}")),
            ty: field_ty(cx, fields[index]),
            binding: None,
            offset: offsets[index],
        });
        index += 1;
    }
    out
}

fn tail_start(fields: &[FieldKind], offsets: &[u32]) -> Result<u32, TranslateError> {
    let last = fields
        .len()
        .checked_sub(1)
        .ok_or_else(|| type_msg("unsupported empty struct"))?;
    let (size, _) = crate::ty::field_size_align(fields[last]);
    offsets
        .get(last)
        .copied()
        .ok_or_else(|| type_msg("struct field offset is missing"))?
        .checked_add(size)
        .ok_or_else(|| type_msg("struct span overflow"))
}

fn type_msg(msg: &str) -> TranslateError {
    TranslateError::UnsupportedType(msg.to_string())
}

fn field_ty(cx: &Cx<'_>, kind: FieldKind) -> naga::Handle<Type> {
    match kind {
        FieldKind::F32 => cx.scalar,
        FieldKind::U32 => cx.uint,
        FieldKind::I32 => cx.sint,
        FieldKind::Vec2 => cx.vec2,
        FieldKind::Vec3 => cx.vec3,
        FieldKind::Vec4 => cx.vec4,
    }
}

fn cached_struct(cx: &Cx<'_>, id: u16) -> Option<naga::Handle<Type>> {
    let mut index = 0usize;
    while index < cx.struct_tys.len() {
        if cx.struct_tys[index].0 == id {
            return Some(cx.struct_tys[index].1);
        }
        index += 1;
    }
    None
}
