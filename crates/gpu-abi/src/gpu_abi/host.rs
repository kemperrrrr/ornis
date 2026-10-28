//! Host shadow of GPU bindings.
//!
//! Bytes live in an allocation this module owns. [`with_host`] moves that
//! allocation into a thread-local for the callback, then puts it back — including
//! when the callback panics. Reads and writes copy bytes out; they do not hand
//! the caller a pointer into the slot.

use std::cell::RefCell;

use super::bytes::AlignedBytes;
use super::bytes::load_window;
use super::bytes::store_window;
use super::contract_breach;
use crate::layout::GpuSafeLayout;
use crate::layout::GpuVec2;
use crate::layout::GpuVec4;

std::thread_local! {
    static HOST: RefCell<Option<HostMemory>> = const { RefCell::new(None) };
}

/// CPU mirror of the bindings a shader pass would have.
#[derive(Default)]
pub struct HostMemory {
    slots: Vec<Slot>,
}

impl HostMemory {
    /// Empty binding set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a uniform slot large enough for `T`.
    // qual:api — host upload entry; in-tree callers are the unit tests.
    pub fn bind_uniform<const G: u32, const B: u32, T: GpuSafeLayout>(&mut self) {
        self.bind_bytes(G, B, Kind::Uniform, T::SIZE, T::ALIGN);
    }

    /// Register a storage slot large enough for `T`.
    // qual:api — host upload entry; in-tree callers are the unit tests.
    pub fn bind_storage<const G: u32, const B: u32, T: GpuSafeLayout>(&mut self) {
        self.bind_bytes(G, B, Kind::Storage, T::SIZE, T::ALIGN);
    }

    /// Register a 2D texture. `texels` is row-major and must cover `width * height`.
    // qual:api — host upload entry; in-tree callers are the unit tests.
    pub fn bind_texture<const G: u32, const B: u32>(
        &mut self,
        width: u32,
        height: u32,
        texels: &[GpuVec4],
    ) {
        assert!(width > 0 && height > 0, "empty texture");
        let expected = (width as usize).saturating_mul(height as usize);
        assert!(expected == texels.len(), "texel count");
        self.insert(Slot {
            group: G,
            binding: B,
            kind: Kind::Texture,
            bytes: None,
            texture: Some(HostImage {
                width,
                height,
                texels: texels.to_vec(),
            }),
        });
    }

    /// Register a sampler. The host shadow checks that it is bound; filtering
    /// state is not modeled (samples are nearest).
    // qual:api — host upload entry; in-tree callers are the unit tests.
    pub fn bind_sampler<const G: u32, const B: u32>(&mut self) {
        self.insert(Slot {
            group: G,
            binding: B,
            kind: Kind::Sampler,
            bytes: None,
            texture: None,
        });
    }

    /// CPU upload into a uniform slot at offset 0.
    // qual:api — host upload entry; in-tree callers are the unit tests.
    pub fn write_uniform<const G: u32, const B: u32, T: GpuSafeLayout>(&mut self, value: &T) {
        self.store_kind::<T>(G, B, Kind::Uniform, 0, value);
    }

    /// CPU upload into a storage slot at offset 0.
    // qual:api — host upload entry; in-tree callers are the unit tests.
    pub fn write_storage<const G: u32, const B: u32, T: GpuSafeLayout>(&mut self, value: &T) {
        self.store_kind::<T>(G, B, Kind::Storage, 0, value);
    }

    /// CPU read of a uniform slot at offset 0.
    pub fn read_uniform<const G: u32, const B: u32, T: GpuSafeLayout>(&self) -> T {
        self.load_kind::<T>(G, B, Kind::Uniform, 0)
    }

    /// CPU read of a storage slot at offset 0.
    // qual:api — host upload entry; in-tree callers are the unit tests.
    pub fn read_storage<const G: u32, const B: u32, T: GpuSafeLayout>(&self) -> T {
        self.load_kind::<T>(G, B, Kind::Storage, 0)
    }

    fn bind_bytes(&mut self, group: u32, binding: u32, kind: Kind, size: u32, align: u32) {
        assert!(size > 0, "empty gpu binding");
        self.insert(Slot {
            group,
            binding,
            kind,
            bytes: Some(AlignedBytes::new(size as usize, align as usize)),
            texture: None,
        });
    }

    fn insert(&mut self, slot: Slot) {
        let mut index = 0usize;
        while index < self.slots.len() {
            let existing = &self.slots[index];
            assert!(
                existing.group != slot.group || existing.binding != slot.binding,
                "duplicate gpu binding"
            );
            index += 1;
        }
        self.slots.push(slot);
    }

    fn index_of(&self, group: u32, binding: u32) -> usize {
        let mut index = 0usize;
        while index < self.slots.len() {
            let slot = &self.slots[index];
            if slot.group == group && slot.binding == binding {
                return index;
            }
            index += 1;
        }
        contract_breach("missing gpu binding")
    }

    fn load_kind<T: GpuSafeLayout>(&self, group: u32, binding: u32, kind: Kind, offset: u32) -> T {
        let index = self.index_of(group, binding);
        let slot = &self.slots[index];
        assert!(slot.kind == kind, "gpu binding kind mismatch");
        let Some(bytes) = slot.bytes.as_ref() else {
            contract_breach("binding has no bytes");
        };
        load_window::<T>(bytes.window(), offset)
    }

    fn store_kind<T: GpuSafeLayout>(
        &mut self,
        group: u32,
        binding: u32,
        kind: Kind,
        offset: u32,
        value: &T,
    ) {
        let index = self.index_of(group, binding);
        let slot = &mut self.slots[index];
        assert!(slot.kind == kind, "gpu binding kind mismatch");
        let Some(bytes) = slot.bytes.as_mut() else {
            contract_breach("binding has no bytes");
        };
        store_window::<T>(bytes.window_mut(), offset, value);
    }
}

/// Run `body` with `mem` installed as the shadow the safe SDK reads.
///
/// Nested calls swap the shadow and restore it afterwards. A panic inside
/// `body` still restores `mem`.
pub fn with_host<R>(mem: &mut HostMemory, body: impl FnOnce() -> R) -> R {
    HOST.with(|slot| {
        let prev = slot.borrow_mut().replace(core::mem::take(mem));
        let _guard = Restore {
            slot,
            dest: mem,
            prev,
        };
        body()
    })
}

pub(super) fn host_read<T: GpuSafeLayout>(group: u32, binding: u32, offset: u32) -> T {
    HOST.with(|slot| {
        let borrowed = slot.borrow();
        let Some(mem) = borrowed.as_ref() else {
            contract_breach("GpuRef::read outside with_host");
        };
        let index = mem.index_of(group, binding);
        let binding_slot = &mem.slots[index];
        let Some(bytes) = binding_slot.bytes.as_ref() else {
            contract_breach("binding has no bytes");
        };
        load_window::<T>(bytes.window(), offset)
    })
}

pub(super) fn host_write<T: GpuSafeLayout>(group: u32, binding: u32, offset: u32, value: &T) {
    HOST.with(|slot| {
        let mut borrowed = slot.borrow_mut();
        let Some(mem) = borrowed.as_mut() else {
            contract_breach("GpuRef::write outside with_host");
        };
        let index = mem.index_of(group, binding);
        let binding_slot = &mut mem.slots[index];
        assert!(
            binding_slot.kind == Kind::Storage,
            "shader write requires a storage binding"
        );
        let Some(bytes) = binding_slot.bytes.as_mut() else {
            contract_breach("binding has no bytes");
        };
        store_window::<T>(bytes.window_mut(), offset, value);
    })
}

pub(super) fn host_sample<const TG: u32, const TB: u32, const SG: u32, const SB: u32>(
    uv: GpuVec2,
) -> GpuVec4 {
    HOST.with(|slot| {
        let borrowed = slot.borrow();
        let Some(mem) = borrowed.as_ref() else {
            contract_breach("texture sample outside with_host");
        };
        sampler_at(mem, SG, SB);
        nearest(image_at(mem, TG, TB), uv)
    })
}

fn image_at(mem: &HostMemory, group: u32, binding: u32) -> &HostImage {
    let index = mem.index_of(group, binding);
    let slot = &mem.slots[index];
    assert!(slot.kind == Kind::Texture, "binding is not a texture");
    let Some(image) = slot.texture.as_ref() else {
        contract_breach("texture binding has no image");
    };
    image
}

fn sampler_at(mem: &HostMemory, group: u32, binding: u32) {
    let index = mem.index_of(group, binding);
    let slot = &mem.slots[index];
    assert!(slot.kind == Kind::Sampler, "binding is not a sampler");
}

fn nearest(image: &HostImage, uv: GpuVec2) -> GpuVec4 {
    let x = axis_index(uv.x, image.width);
    let y = axis_index(uv.y, image.height);
    let index = (y * image.width + x) as usize;
    image.texels[index]
}

fn axis_index(t: f32, size: u32) -> u32 {
    assert!(size > 0, "empty texture axis");
    let clamped = t.clamp(0.0, 1.0);
    let mut index = (clamped * (size as f32)) as u32;
    if index >= size {
        index = size - 1;
    }
    index
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Uniform,
    Storage,
    Texture,
    Sampler,
}

struct Slot {
    group: u32,
    binding: u32,
    kind: Kind,
    bytes: Option<AlignedBytes>,
    texture: Option<HostImage>,
}

struct HostImage {
    width: u32,
    height: u32,
    texels: Vec<GpuVec4>,
}

struct Restore<'a> {
    slot: &'a RefCell<Option<HostMemory>>,
    dest: &'a mut HostMemory,
    prev: Option<HostMemory>,
}

impl Drop for Restore<'_> {
    fn drop(&mut self) {
        let updated = self.slot.replace(self.prev.take());
        if let Some(updated) = updated {
            *self.dest = updated;
        }
    }
}
