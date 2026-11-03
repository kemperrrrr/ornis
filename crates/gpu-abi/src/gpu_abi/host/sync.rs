//! Host-shadow texture, atomic, and workgroup synchronization operations.

use super::{HOST, HostMemory, Kind, contract_breach, image_at};
use crate::layout::GpuVec4;

pub(in crate::gpu_abi) fn host_storage_atomic_add_u32(group: u32, binding: u32, index: u32, value: u32) -> u32 {
    HOST.with(|slot| {
        let mut borrowed = slot.borrow_mut();
        let Some(mem) = borrowed.as_mut() else {
            contract_breach("storage atomic outside with_host");
        };
        let offset = index
            .checked_mul(crate::layout::F32_SIZE)
            .unwrap_or_else(|| contract_breach("storage atomic offset overflow"));
        let old = mem.load_kind::<u32>(group, binding, Kind::Storage, offset);
        mem.store_kind(
            group,
            binding,
            Kind::Storage,
            offset,
            &old.wrapping_add(value),
        );
        old
    })
}

pub(in crate::gpu_abi) fn host_workgroup_load_u32(index: u32) -> u32 {
    HOST.with(|slot| {
        let mut borrowed = slot.borrow_mut();
        let Some(mem) = borrowed.as_mut() else {
            contract_breach("workgroup access outside with_host");
        };
        mem.ensure_workgroup();
        *mem.workgroup
            .get(index as usize)
            .unwrap_or_else(|| contract_breach("workgroup index out of bounds"))
    })
}

pub(in crate::gpu_abi) fn host_workgroup_store_u32(index: u32, value: u32) {
    HOST.with(|slot| {
        let mut borrowed = slot.borrow_mut();
        let Some(mem) = borrowed.as_mut() else {
            contract_breach("workgroup access outside with_host");
        };
        mem.ensure_workgroup();
        let Some(slot) = mem.workgroup.get_mut(index as usize) else {
            contract_breach("workgroup index out of bounds");
        };
        *slot = value;
    })
}

pub(in crate::gpu_abi) fn host_workgroup_atomic_add_u32(index: u32, value: u32) -> u32 {
    HOST.with(|slot| {
        let mut borrowed = slot.borrow_mut();
        let Some(mem) = borrowed.as_mut() else {
            contract_breach("workgroup atomic outside with_host");
        };
        mem.ensure_workgroup();
        let Some(slot) = mem.workgroup.get_mut(index as usize) else {
            contract_breach("workgroup index out of bounds");
        };
        let old = *slot;
        *slot = old.wrapping_add(value);
        old
    })
}

pub(in crate::gpu_abi) fn host_workgroup_barrier() {
    HOST.with(|slot| {
        if slot.borrow().is_none() {
            contract_breach("workgroup barrier outside with_host");
        }
    })
}

pub(in crate::gpu_abi) fn host_storage_barrier() {
    HOST.with(|slot| {
        if slot.borrow().is_none() {
            contract_breach("storage barrier outside with_host");
        }
    })
}

pub(in crate::gpu_abi) fn host_texture_load(group: u32, binding: u32, x: i32, y: i32) -> GpuVec4 {
    HOST.with(|slot| {
        let borrowed = slot.borrow();
        let Some(mem) = borrowed.as_ref() else {
            contract_breach("texture load outside with_host");
        };
        let image = image_at(mem, group, binding);
        assert!(x >= 0 && y >= 0, "texture coordinate out of bounds");
        let (x, y) = (x as u32, y as u32);
        assert!(
            x < image.width && y < image.height,
            "texture coordinate out of bounds"
        );
        image.texels[(y as usize * image.width as usize) + x as usize]
    })
}

pub(in crate::gpu_abi) fn host_texture_width(group: u32, binding: u32) -> u32 {
    HOST.with(|slot| {
        let borrowed = slot.borrow();
        let Some(mem) = borrowed.as_ref() else {
            contract_breach("texture dimensions outside with_host");
        };
        image_at(mem, group, binding).width
    })
}

pub(in crate::gpu_abi) fn host_texture_height(group: u32, binding: u32) -> u32 {
    HOST.with(|slot| {
        let borrowed = slot.borrow();
        let Some(mem) = borrowed.as_ref() else {
            contract_breach("texture dimensions outside with_host");
        };
        image_at(mem, group, binding).height
    })
}
