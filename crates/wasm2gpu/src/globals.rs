//! Uniform, texture, sampler, and storage globals. One variable per binding.
//!
//! A storage binding is also keyed by its element type. Using the same
//! group and binding as two different elements is rejected.

use naga::AddressSpace;
use naga::ArraySize;
use naga::ScalarKind;
use naga::Span;
use naga::StorageAccess;
use naga::Type;
use naga::TypeInner;

use crate::cx::Cx;
use crate::cx::Slot;
use crate::cx::StorageSlot;
use crate::error::TranslateError;
use crate::ty::Elem;
use crate::ty::StructDef;

pub(super) use crate::struct_members::struct_ty;
pub(super) use crate::uniform_global::uniform;
pub(super) use crate::uniform_global::uniform_global;

pub(super) fn storage(
    cx: &mut Cx<'_>,
    group: u32,
    binding: u32,
    elem: Elem,
    structs: &[StructDef],
) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
    if crate::uniform_global::uniform_taken(&cx.uniforms, group, binding) {
        return Err(TranslateError::UnsupportedType(
            "binding is used as both uniform and storage".to_string(),
        ));
    }
    if let Some((found, var)) = find_storage(&cx.storages, group, binding) {
        if found != elem {
            return Err(TranslateError::UnsupportedType(
                "binding is used as two storage types".to_string(),
            ));
        }
        return Ok(var);
    }
    let layout = crate::ty::elem_layout(elem, structs)?;
    let base = base_ty(cx, elem, structs)?;
    let ty = array_ty(cx, base, layout.stride);
    let var = append(
        cx,
        format!("storage_{group}_{binding}_"),
        AddressSpace::Storage {
            access: StorageAccess::LOAD | StorageAccess::STORE,
        },
        ty,
        point(group, binding),
    );
    cx.storages.push(StorageSlot {
        group,
        binding,
        elem,
        var,
    });
    Ok(var)
}

pub(super) fn texture(
    cx: &mut Cx<'_>,
    group: u32,
    binding: u32,
) -> naga::Handle<naga::GlobalVariable> {
    if let Some(var) = cached(&cx.textures, group, binding) {
        return var;
    }
    let ty = image_ty(cx);
    let var = append(
        cx,
        format!("tex_{group}_{binding}"),
        AddressSpace::Handle,
        ty,
        point(group, binding),
    );
    cx.textures.push(Slot {
        group,
        binding,
        var,
    });
    var
}

pub(super) fn sampler(
    cx: &mut Cx<'_>,
    group: u32,
    binding: u32,
) -> naga::Handle<naga::GlobalVariable> {
    if let Some(var) = cached(&cx.samplers, group, binding) {
        return var;
    }
    let ty = sampler_ty(cx);
    let var = append(
        cx,
        format!("samp_{group}_{binding}"),
        AddressSpace::Handle,
        ty,
        point(group, binding),
    );
    cx.samplers.push(Slot {
        group,
        binding,
        var,
    });
    var
}

fn find_storage(
    slots: &[StorageSlot],
    group: u32,
    binding: u32,
) -> Option<(Elem, naga::Handle<naga::GlobalVariable>)> {
    let mut index = 0usize;
    while index < slots.len() {
        if slots[index].group == group && slots[index].binding == binding {
            return Some((slots[index].elem, slots[index].var));
        }
        index += 1;
    }
    None
}

pub(super) fn storage_taken(slots: &[StorageSlot], group: u32, binding: u32) -> bool {
    find_storage(slots, group, binding).is_some()
}

fn array_ty(cx: &mut Cx<'_>, base: naga::Handle<Type>, stride: u32) -> naga::Handle<Type> {
    cx.module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Array {
                base,
                size: ArraySize::Dynamic,
                stride,
            },
        },
        Span::default(),
    )
}

pub(super) fn base_ty(
    cx: &mut Cx<'_>,
    elem: Elem,
    structs: &[StructDef],
) -> Result<naga::Handle<Type>, TranslateError> {
    match elem {
        Elem::F32 => Ok(cx.scalar),
        Elem::U32 => Ok(cx.uint),
        Elem::I32 => Ok(cx.sint),
        Elem::Vec2 => Ok(cx.vec2),
        Elem::Vec3 => Ok(cx.vec3),
        Elem::Vec4 => Ok(cx.vec4),
        Elem::Struct(id) => {
            let def = structs
                .get(id as usize)
                .ok_or(TranslateError::Unsupported)?;
            let fields = def.fields.clone();
            struct_ty(cx, id, &fields)
        }
    }
}

fn cached(slots: &[Slot], group: u32, binding: u32) -> Option<naga::Handle<naga::GlobalVariable>> {
    let mut index = 0usize;
    while index < slots.len() {
        if slots[index].group == group && slots[index].binding == binding {
            return Some(slots[index].var);
        }
        index += 1;
    }
    None
}

pub(super) fn point(group: u32, binding: u32) -> naga::ResourceBinding {
    naga::ResourceBinding { group, binding }
}

pub(super) fn append(
    cx: &mut Cx<'_>,
    name: String,
    space: AddressSpace,
    ty: naga::Handle<Type>,
    binding: naga::ResourceBinding,
) -> naga::Handle<naga::GlobalVariable> {
    cx.module.global_variables.append(
        naga::GlobalVariable {
            name: Some(name),
            space,
            binding: Some(binding),
            ty,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::default(),
    )
}

fn image_ty(cx: &mut Cx<'_>) -> naga::Handle<Type> {
    cx.module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Image {
                dim: naga::ImageDimension::D2,
                arrayed: false,
                class: naga::ImageClass::Sampled {
                    kind: ScalarKind::Float,
                    multi: false,
                },
            },
        },
        Span::default(),
    )
}

fn sampler_ty(cx: &mut Cx<'_>) -> naga::Handle<Type> {
    cx.module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Sampler { comparison: false },
        },
        Span::default(),
    )
}
