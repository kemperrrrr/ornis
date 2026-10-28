//! Uniform, texture, and sampler globals. One variable per (group, binding).

use naga::AddressSpace;
use naga::ScalarKind;
use naga::Span;
use naga::StorageAccess;
use naga::Type;
use naga::TypeInner;

use crate::cx::Cx;
use crate::cx::Slot;

pub(super) fn uniform(
    cx: &mut Cx<'_>,
    group: u32,
    binding: u32,
) -> naga::Handle<naga::GlobalVariable> {
    if let Some(var) = cached(&cx.uniforms, group, binding) {
        return var;
    }
    let ty = cx.vec4;
    let var = append(
        cx,
        format!("uniform_{group}_{binding}"),
        AddressSpace::Uniform,
        ty,
        point(group, binding),
    );
    cx.uniforms.push(Slot {
        group,
        binding,
        var,
    });
    var
}

pub(super) fn storage(
    cx: &mut Cx<'_>,
    group: u32,
    binding: u32,
) -> naga::Handle<naga::GlobalVariable> {
    if let Some(var) = cached(&cx.storages, group, binding) {
        return var;
    }
    let ty = cx.vec4_array;
    let var = append(
        cx,
        format!("storage_{group}_{binding}"),
        AddressSpace::Storage {
            access: StorageAccess::LOAD | StorageAccess::STORE,
        },
        ty,
        point(group, binding),
    );
    cx.storages.push(Slot {
        group,
        binding,
        var,
    });
    var
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

fn point(group: u32, binding: u32) -> naga::ResourceBinding {
    naga::ResourceBinding { group, binding }
}

fn append(
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
