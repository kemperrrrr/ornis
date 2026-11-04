//! Atomic storage and per-workgroup scratch globals.

use super::{append, array_ty, point, storage_taken};
use crate::cx::AtomicSlot;
use crate::cx::Cx;
use crate::error::TranslateError;
use naga::AddressSpace;
use naga::ArraySize;
use naga::ScalarKind;
use naga::Span;
use naga::StorageAccess;
use naga::Type;
use naga::TypeInner;

pub(crate) fn atomic_storage(
    cx: &mut Cx<'_>,
    group: u32,
    binding: u32,
) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
    if crate::uniform_global::uniform_taken(&cx.uniforms, group, binding) {
        return Err(TranslateError::UnsupportedType(
            "binding is used as both uniform and atomic storage".to_string(),
        ));
    }
    if storage_taken(&cx.storages, group, binding) {
        return Err(TranslateError::UnsupportedType(
            "binding is used as both atomic and ordinary storage".to_string(),
        ));
    }
    if let Some(slot) = super::find_binding(&cx.atomic_storages, group, binding, |slot| {
        (slot.group, slot.binding)
    }) {
        return Ok(slot.var);
    }
    let atomic = cx.module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Atomic(naga::Scalar {
                kind: ScalarKind::Uint,
                width: crate::cx::F32_WIDTH,
            }),
        },
        Span::default(),
    );
    let ty = array_ty(cx, atomic, ornis_gpu_abi::F32_SIZE);
    let var = append(
        cx,
        format!("atomic_storage_{group}_{binding}"),
        AddressSpace::Storage {
            access: StorageAccess::LOAD | StorageAccess::STORE,
        },
        ty,
        point(group, binding),
    );
    cx.atomic_storages.push(AtomicSlot {
        group,
        binding,
        var,
    });
    Ok(var)
}

pub(crate) fn workgroup(cx: &mut Cx<'_>) -> naga::Handle<naga::GlobalVariable> {
    if let Some(var) = cx.workgroup {
        return var;
    }
    let uint = cx.uint;
    let ty = workgroup_array(cx, uint);
    let var = cx.module.global_variables.append(
        naga::GlobalVariable {
            name: Some("workgroup_scratch".to_string()),
            space: AddressSpace::WorkGroup,
            binding: None,
            ty,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::default(),
    );
    cx.workgroup = Some(var);
    var
}

pub(crate) fn workgroup_atomics(cx: &mut Cx<'_>) -> naga::Handle<naga::GlobalVariable> {
    if let Some(var) = cx.workgroup_atomics {
        return var;
    }
    let atomic = cx.module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Atomic(naga::Scalar {
                kind: ScalarKind::Uint,
                width: crate::cx::F32_WIDTH,
            }),
        },
        Span::default(),
    );
    let ty = workgroup_array(cx, atomic);
    let var = cx.module.global_variables.append(
        naga::GlobalVariable {
            name: Some("workgroup_atomics".to_string()),
            space: AddressSpace::WorkGroup,
            binding: None,
            ty,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::default(),
    );
    cx.workgroup_atomics = Some(var);
    var
}

const _: () = assert!(ornis_gpu_abi::symbols::WORKGROUP_WORDS > 0);
const WORKGROUP_ARRAY_SIZE: std::num::NonZeroU32 = std::num::NonZeroU32::new(
    ornis_gpu_abi::symbols::WORKGROUP_WORDS,
)
.expect("WORKGROUP_WORDS must be nonzero");

fn workgroup_array(cx: &mut Cx<'_>, base: naga::Handle<Type>) -> naga::Handle<Type> {
    cx.module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Array {
                base,
                size: ArraySize::Constant(WORKGROUP_ARRAY_SIZE),
                stride: ornis_gpu_abi::F32_SIZE,
            },
        },
        Span::default(),
    )
}

pub(crate) fn atomic_taken(slots: &[AtomicSlot], group: u32, binding: u32) -> bool {
    super::find_binding(slots, group, binding, |slot| (slot.group, slot.binding)).is_some()
}
