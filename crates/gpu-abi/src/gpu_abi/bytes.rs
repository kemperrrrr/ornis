//! Aligned slot storage and the checked window around a typed copy.

use super::contract_breach;
use super::copy::pod_load;
use super::copy::pod_store;
use crate::layout::GpuSafeLayout;

pub(super) fn load_window<T: GpuSafeLayout>(bytes: &[u8], offset: u32) -> T {
    assert!(T::SIZE as usize == core::mem::size_of::<T>(), "SIZE");
    assert!(
        (T::ALIGN as usize).is_multiple_of(core::mem::align_of::<T>()),
        "ALIGN"
    );
    let region = region(bytes, offset, T::SIZE as usize, T::ALIGN as usize);
    pod_load!(T, region)
}

pub(super) fn store_window<T: GpuSafeLayout>(bytes: &mut [u8], offset: u32, value: &T) {
    assert!(T::SIZE as usize == core::mem::size_of::<T>(), "SIZE");
    let region = region_mut(bytes, offset, T::SIZE as usize, T::ALIGN as usize);
    pod_store!(T, region, value);
}

fn region(bytes: &[u8], offset: u32, size: usize, align: usize) -> &[u8] {
    let start = checked_start(bytes, offset, size, align);
    &bytes[start..start + size]
}

fn region_mut(bytes: &mut [u8], offset: u32, size: usize, align: usize) -> &mut [u8] {
    let start = checked_start(bytes, offset, size, align);
    &mut bytes[start..start + size]
}

fn checked_start(bytes: &[u8], offset: u32, size: usize, align: usize) -> usize {
    assert!(size > 0, "zero-sized gpu access");
    assert!(align > 0, "zero alignment");
    assert!(
        bytes.as_ptr().addr().is_multiple_of(align),
        "gpu slot base is misaligned"
    );
    let start = offset as usize;
    assert!(start.is_multiple_of(align), "misaligned gpu access");
    let Some(end) = start.checked_add(size) else {
        contract_breach("gpu access size overflow");
    };
    assert!(end <= bytes.len(), "gpu access out of bounds");
    start
}

pub(super) struct AlignedBytes {
    buf: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedBytes {
    pub(super) fn new(len: usize, align: usize) -> Self {
        assert!(align.is_power_of_two(), "slot align");
        assert!(len > 0, "empty slot");
        let buf = vec![0; len + align];
        let mis = buf.as_ptr().addr() & (align - 1);
        let start = if mis == 0 { 0 } else { align - mis };
        assert!(start + len <= buf.len(), "aligned window fits");
        let slot = Self { buf, start, len };
        assert!(
            slot.window().as_ptr().addr().is_multiple_of(align),
            "window align"
        );
        slot
    }

    pub(super) fn window(&self) -> &[u8] {
        &self.buf[self.start..self.start + self.len]
    }

    pub(super) fn window_mut(&mut self) -> &mut [u8] {
        &mut self.buf[self.start..self.start + self.len]
    }
}
