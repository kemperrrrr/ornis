//! Byte copies and wasm32 import calls.
//!
//! This file is the only place the SDK spells `unsafe`. Callers have already
//! checked that the byte region is in-bounds and aligned for `T`, and that
//! `T: GpuSafeLayout` (every bit pattern is a valid value).

#[cfg(target_arch = "wasm32")]
use crate::layout::F32_SIZE;
#[cfg(target_arch = "wasm32")]
use crate::layout::GpuSafeLayout;
#[cfg(target_arch = "wasm32")]
use crate::layout::GpuVec2;
#[cfg(target_arch = "wasm32")]
use crate::layout::GpuVec4;
#[cfg(target_arch = "wasm32")]
use crate::layout::VEC4_SIZE;

/// Lane count of a `vec4<f32>`.
#[cfg(target_arch = "wasm32")]
const LANE_COUNT: usize = (VEC4_SIZE / F32_SIZE) as usize;

macro_rules! pod_load {
    ($ty:ty, $region:expr) => {{
        let region: &[u8] = $region;
        assert!(region.len() == ::core::mem::size_of::<$ty>());
        assert!(
            region
                .as_ptr()
                .addr()
                .is_multiple_of(::core::mem::align_of::<$ty>())
        );
        // SAFETY: `region` is exactly `size_of::<T>()` initialized bytes and
        // is aligned for `T`. `T: GpuSafeLayout`, so those bytes are a valid
        // `T`. The pointer is not retained.
        unsafe { region.as_ptr().cast::<$ty>().read() }
    }};
}

macro_rules! pod_store {
    ($ty:ty, $region:expr, $value:expr) => {{
        let region: &mut [u8] = $region;
        assert!(region.len() == ::core::mem::size_of::<$ty>());
        assert!(
            region
                .as_mut_ptr()
                .addr()
                .is_multiple_of(::core::mem::align_of::<$ty>())
        );
        let value: &$ty = $value;
        // SAFETY: `region` is in-bounds and aligned for `T`, and it does not
        // alias `value` (the slot allocation is distinct from the caller's
        // local). `T: GpuSafeLayout`. The write covers every byte of `region`.
        unsafe { region.as_mut_ptr().cast::<$ty>().write(*value) }
    }};
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) use pod_load;
#[cfg(not(target_arch = "wasm32"))]
pub(super) use pod_store;

/// Four `f32` lanes returned by the vec4 wasm import.
#[cfg(target_arch = "wasm32")]
#[repr(C)]
struct Word4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

#[cfg(target_arch = "wasm32")]
macro_rules! import_read_f32 {
    ($group:expr, $binding:expr, $offset:expr) => {{
        // SAFETY: `raw_read_f32` is the ornis_gpu import. It returns one
        // initialized f32 and does not expose linear memory.
        unsafe { raw_read_f32($group, $binding, $offset) }
    }};
}

#[cfg(target_arch = "wasm32")]
macro_rules! import_read_vec4 {
    ($group:expr, $binding:expr, $offset:expr) => {{
        // SAFETY: `raw_read` is the ornis_gpu vec4 import. It returns four
        // initialized lanes. The caller copies them only into a 16-byte
        // `GpuSafeLayout` value.
        unsafe { raw_read($group, $binding, $offset) }
    }};
}

#[cfg(target_arch = "wasm32")]
macro_rules! import_write {
    ($group:expr, $binding:expr, $offset:expr, $x:expr, $y:expr, $z:expr, $w:expr) => {{
        // SAFETY: `raw_write` is the ornis_gpu import. The four lanes are the
        // bytes of a valid `GpuSafeLayout` value. The import does not expose
        // linear memory.
        unsafe { raw_write($group, $binding, $offset, $x, $y, $z, $w) }
    }};
}

#[cfg(target_arch = "wasm32")]
macro_rules! import_sample {
    ($tg:expr, $tb:expr, $sg:expr, $sb:expr, $u:expr, $v:expr) => {{
        // SAFETY: `texture_sample_2d` is the ornis_gpu import. It returns four
        // initialized lanes and does not load from linear memory.
        unsafe { texture_sample_2d($tg, $tb, $sg, $sb, $u, $v) }
    }};
}

// SAFETY: these symbols are the ornis_gpu import ABI. Calling them is sound
// only through the wrappers in this file; they do not dereference a guest
// pointer. `storage_base` only returns the translator's base tag.
#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "ornis_gpu")]
unsafe extern "C" {
    fn storage_base(group: u32, binding: u32) -> u32;
    fn raw_read_f32(group: u32, binding: u32, offset: u32) -> f32;
    fn raw_read(group: u32, binding: u32, offset: u32) -> Word4;
    fn raw_write(group: u32, binding: u32, offset: u32, x: f32, y: f32, z: f32, w: f32);
    fn texture_sample_2d(
        tex_group: u32,
        tex_binding: u32,
        samp_group: u32,
        samp_binding: u32,
        u: f32,
        v: f32,
    ) -> Word4;
    fn uniform_field_f32(group: u32, binding: u32, sid: u32, field: u32) -> f32;
    fn uniform_field_u32(group: u32, binding: u32, sid: u32, field: u32) -> u32;
    fn uniform_read_f32(group: u32, binding: u32, len: u32, index: u32) -> f32;
    fn storage_read_f32(group: u32, binding: u32, index: u32) -> f32;
    fn storage_write_f32(group: u32, binding: u32, index: u32, value: f32);
    fn storage_read_u32(group: u32, binding: u32, index: u32) -> u32;
    fn storage_write_u32(group: u32, binding: u32, index: u32, value: u32);
}

#[cfg(target_arch = "wasm32")]
macro_rules! import_storage_base {
    ($group:expr, $binding:expr) => {{
        // SAFETY: `storage_base` is the ornis_gpu import. It returns the
        // translator's base tag for the provenance pair and touches no memory.
        unsafe { storage_base($group, $binding) }
    }};
}

#[cfg(target_arch = "wasm32")]
macro_rules! ptr_read {
    ($ty:ty, $addr:expr) => {{
        // SAFETY: this macro is only reachable through `GpuRef::load`, which
        // builds `addr` as `storage_base(G, B)` plus a `GpuRef` offset. Token
        // construction checks alignment: `index` rejects a stride that is not
        // a multiple of `T::ALIGN`, `field` rejects a misaligned field offset,
        // and the layout gate keeps `T::ALIGN` a multiple of `align_of::<T>()`,
        // so the address names a `$ty`-aligned element of that binding. (The
        // tests in `abi_ptr_tests.rs` pin the two token checks.) The four
        // bytes hold a valid `$ty` — `T: GpuSafeLayout` — and the pointer is
        // not retained.
        unsafe { ($addr as *const $ty).read_volatile() }
    }};
}

#[cfg(target_arch = "wasm32")]
macro_rules! ptr_write {
    ($ty:ty, $addr:expr, $value:expr) => {{
        // SAFETY: same address contract as `ptr_read`: only `GpuRef::store`
        // reaches this macro, and its offset half is alignment-checked when
        // the token is built. The write covers exactly the four bytes of
        // `addr`, which belongs to this binding's storage.
        unsafe { ($addr as *mut $ty).write_volatile($value) }
    }};
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn wasm_storage_base<const G: u32, const B: u32>() -> u32 {
    import_storage_base!(G, B)
}

/// Pointer-path read at a `storage_base` address.
///
/// The offset half of `addr` was alignment-checked when the `GpuRef` token
/// was built (`index` / `field` in `gpu_ref.rs`); this stays `inline(always)`
/// so the pointer never crosses a call as a raw value.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn ptr_load<T: GpuSafeLayout>(addr: u32) -> T {
    if core::mem::size_of::<T>() != F32_SIZE as usize {
        return crate::gpu_abi::contract_breach("pointer load supports 4-byte values");
    }
    ptr_read!(T, addr)
}

/// Pointer-path write at a `storage_base` address.
///
/// Same alignment contract and inlining as `ptr_load`.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn ptr_store<T: GpuSafeLayout>(addr: u32, value: T) {
    if core::mem::size_of::<T>() != F32_SIZE as usize {
        crate::gpu_abi::contract_breach("pointer store supports 4-byte values");
    }
    ptr_write!(T, addr, value)
}

#[cfg(target_arch = "wasm32")]
pub(super) fn wasm_read<T: GpuSafeLayout>(group: u32, binding: u32, offset: u32) -> T {
    let size = core::mem::size_of::<T>();
    if size == F32_SIZE as usize {
        wasm_read_f32::<T>(group, binding, offset)
    } else if size == VEC4_SIZE as usize {
        wasm_read_vec4::<T>(group, binding, offset)
    } else {
        crate::gpu_abi::contract_breach("wasm32 raw_read supports 4-byte and 16-byte values")
    }
}

#[cfg(target_arch = "wasm32")]
fn wasm_read_f32<T: GpuSafeLayout>(group: u32, binding: u32, offset: u32) -> T {
    let word = import_read_f32!(group, binding, offset);
    pod_load!(T, &word.to_le_bytes())
}

#[cfg(target_arch = "wasm32")]
fn wasm_read_vec4<T: GpuSafeLayout>(group: u32, binding: u32, offset: u32) -> T {
    let word = import_read_vec4!(group, binding, offset);
    pod_load!(T, &pack_word(word))
}

#[cfg(target_arch = "wasm32")]
pub(super) fn wasm_write<T: GpuSafeLayout>(group: u32, binding: u32, offset: u32, value: &T) {
    let size = core::mem::size_of::<T>();
    assert!(
        size == VEC4_SIZE as usize,
        "wasm32 raw_write supports 16-byte values"
    );
    let mut bytes = [0u8; VEC4_SIZE as usize];
    pod_store!(T, &mut bytes[..], value);
    let lanes = lanes_from_bytes(&bytes);
    wasm_raw_write(group, binding, offset, lanes);
}

#[cfg(target_arch = "wasm32")]
pub(super) fn wasm_sample<const TG: u32, const TB: u32, const SG: u32, const SB: u32>(
    uv: GpuVec2,
) -> GpuVec4 {
    let word = import_sample!(TG, TB, SG, SB, uv.x, uv.y);
    GpuVec4 {
        x: word.x,
        y: word.y,
        z: word.z,
        w: word.w,
    }
}

#[cfg(target_arch = "wasm32")]
fn pack_word(word: Word4) -> [u8; VEC4_SIZE as usize] {
    let mut bytes = [0u8; VEC4_SIZE as usize];
    let lanes = [word.x, word.y, word.z, word.w];
    let mut index = 0usize;
    while index < lanes.len() {
        let chunk = lanes[index].to_le_bytes();
        let start = index * core::mem::size_of::<f32>();
        bytes[start..start + chunk.len()].copy_from_slice(&chunk);
        index += 1;
    }
    bytes
}

#[cfg(target_arch = "wasm32")]
fn wasm_raw_write(group: u32, binding: u32, offset: u32, lanes: [f32; LANE_COUNT]) {
    import_write!(
        group, binding, offset, lanes[0], lanes[1], lanes[2], lanes[3]
    );
}

#[cfg(target_arch = "wasm32")]
macro_rules! import_uniform_field_f32 {
    ($group:expr, $binding:expr, $sid:expr, $field:expr) => {{
        // SAFETY: `uniform_field_f32` is the ornis_gpu import. It returns one
        // initialized f32 and does not expose linear memory.
        unsafe { uniform_field_f32($group, $binding, $sid, $field) }
    }};
}

#[cfg(target_arch = "wasm32")]
macro_rules! import_uniform_field_u32 {
    ($group:expr, $binding:expr, $sid:expr, $field:expr) => {{
        // SAFETY: `uniform_field_u32` is the ornis_gpu import. It returns one
        // initialized u32 and does not expose linear memory.
        unsafe { uniform_field_u32($group, $binding, $sid, $field) }
    }};
}

#[cfg(target_arch = "wasm32")]
macro_rules! import_uniform_read_f32 {
    ($group:expr, $binding:expr, $len:expr, $index:expr) => {{
        // SAFETY: `uniform_read_f32` is the ornis_gpu import. It returns one
        // initialized f32 and does not expose linear memory.
        unsafe { uniform_read_f32($group, $binding, $len, $index) }
    }};
}

#[cfg(target_arch = "wasm32")]
macro_rules! import_storage_read_f32 {
    ($group:expr, $binding:expr, $index:expr) => {{
        // SAFETY: `storage_read_f32` is the ornis_gpu import. It returns one
        // initialized f32 and does not expose linear memory.
        unsafe { storage_read_f32($group, $binding, $index) }
    }};
}

#[cfg(target_arch = "wasm32")]
macro_rules! import_storage_write_f32 {
    ($group:expr, $binding:expr, $index:expr, $value:expr) => {{
        // SAFETY: `storage_write_f32` is the ornis_gpu import. The lane is one
        // f32. The import does not expose linear memory.
        unsafe { storage_write_f32($group, $binding, $index, $value) }
    }};
}

#[cfg(target_arch = "wasm32")]
macro_rules! import_storage_read_u32 {
    ($group:expr, $binding:expr, $index:expr) => {{
        // SAFETY: `storage_read_u32` is the ornis_gpu import. It returns one
        // initialized u32 and does not expose linear memory.
        unsafe { storage_read_u32($group, $binding, $index) }
    }};
}

#[cfg(target_arch = "wasm32")]
macro_rules! import_storage_write_u32 {
    ($group:expr, $binding:expr, $index:expr, $value:expr) => {{
        // SAFETY: `storage_write_u32` is the ornis_gpu import. The lane is one
        // u32. The import does not expose linear memory.
        unsafe { storage_write_u32($group, $binding, $index, $value) }
    }};
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn call_uniform_field_f32(group: u32, binding: u32, sid: u32, field: u32) -> f32 {
    import_uniform_field_f32!(group, binding, sid, field)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn call_uniform_field_u32(group: u32, binding: u32, sid: u32, field: u32) -> u32 {
    import_uniform_field_u32!(group, binding, sid, field)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn call_uniform_read_f32(group: u32, binding: u32, len: u32, index: u32) -> f32 {
    import_uniform_read_f32!(group, binding, len, index)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn call_storage_read_f32(group: u32, binding: u32, index: u32) -> f32 {
    import_storage_read_f32!(group, binding, index)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn call_storage_write_f32(group: u32, binding: u32, index: u32, value: f32) {
    import_storage_write_f32!(group, binding, index, value)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn call_storage_read_u32(group: u32, binding: u32, index: u32) -> u32 {
    import_storage_read_u32!(group, binding, index)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn call_storage_write_u32(group: u32, binding: u32, index: u32, value: u32) {
    import_storage_write_u32!(group, binding, index, value)
}

#[cfg(target_arch = "wasm32")]
fn lanes_from_bytes(bytes: &[u8; VEC4_SIZE as usize]) -> [f32; LANE_COUNT] {
    let mut lanes = [0.0f32; LANE_COUNT];
    let mut index = 0usize;
    while index < lanes.len() {
        let start = index * core::mem::size_of::<f32>();
        let mut chunk = [0u8; F32_SIZE as usize];
        let len = chunk.len();
        chunk.copy_from_slice(&bytes[start..start + len]);
        lanes[index] = f32::from_le_bytes(chunk);
        index += 1;
    }
    lanes
}
