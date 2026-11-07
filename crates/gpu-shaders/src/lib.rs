//! no_std GPU kernels for the stable Rust → LLVM IR → WGSL path.
//!
//! The crate is a normal `no_std` library: cargo checks it on the host, while
//! `cargo xtask build-shaders` compiles the same source for
//! `wasm32-unknown-unknown` with `--emit=llvm-ir` and feeds the IR to
//! `ornis-llvm2gpu`. `cs_main` is the stage-3 cull/compaction candidate:
//! one invocation per source record, sphere-vs-frustum culling, one dense
//! slot from the indirect-args atomic, and a bit-preserving 13×`vec4` copy.

#![no_std]
#![warn(missing_docs)]

/// Four-word chunks in one `PerObject` record.
const RECORD_CHUNKS: u32 = 13;
/// Group-0 binding of the packed `CullParams` uniform.
const PARAMS_BINDING: i32 = 0;
/// `ornis_gpu.types` struct id of `CullParams`.
const PARAMS_SID: i32 = 0;
/// Field index of the `u32` count in `CullParams` (after six vec4 planes).
const COUNT_FIELD: i32 = 6;
/// Group-0 read-only source record binding.
const SRC_BINDING: i32 = 1;
/// Group-0 read-write destination record binding.
const DST_BINDING: i32 = 2;
/// Group-0 five-word indirect-args binding.
const ARGS_BINDING: i32 = 3;
/// Group-0 read-only scalar `f32` view of the same source buffer.
const MATH_BINDING: i32 = 4;
/// `f32` words in one source record (the 52-word `PerObject` layout).
const RECORD_WORDS: u32 = 52;

unsafe extern "C" {
    fn storage_len_u32x4(group: i32, binding: i32) -> i32;
    fn storage_read_f32(group: i32, binding: i32, index: i32) -> f32;
    fn storage_copy_u32x4(
        src_group: i32,
        src_binding: i32,
        src_index: i32,
        dst_group: i32,
        dst_binding: i32,
        dst_index: i32,
    );
    fn storage_atomic_add_u32(group: i32, binding: i32, index: i32, value: i32) -> i32;
    fn uniform_field_vec4_component_f32(
        group: i32,
        binding: i32,
        sid: i32,
        field: i32,
        component: i32,
    ) -> f32;
    fn uniform_field_u32(group: i32, binding: i32, sid: i32, field: i32) -> u32;
    fn gpu_length3_xyz(x: f32, y: f32, z: f32) -> f32;
}

/// One `f32` word of a source record through the read-only scalar view.
macro_rules! src_f32 {
    ($record:expr, $word:literal) => {{
        let index = $record * RECORD_WORDS + $word;
        // SAFETY: the caller guards `$record` against the source length; the
        // literal word stays inside the 52-word record.
        unsafe { storage_read_f32(0, MATH_BINDING, index as i32) }
    }};
}

/// One component of a `vec4` plane field in `CullParams`.
macro_rules! plane_component {
    ($plane:expr, $component:expr) => {
        // SAFETY: both indices are compile-time constants inside the
        // declared `CullParams` metadata.
        unsafe {
            uniform_field_vec4_component_f32(0, PARAMS_BINDING, PARAMS_SID, $plane, $component)
        }
    };
}

/// One plane test; `$plane` is a literal so the uniform indices are constants.
macro_rules! plane_test {
    ($plane:literal, $cx:expr, $cy:expr, $cz:expr, $radius:expr) => {{
        let px = plane_component!($plane, 0);
        let py = plane_component!($plane, 1);
        let pz = plane_component!($plane, 2);
        let pw = plane_component!($plane, 3);
        if px * $cx + py * $cy + pz * $cz + pw < -$radius {
            return false;
        }
    }};
}

/// Six unrolled plane tests. A NaN radius skips them and stays visible.
macro_rules! plane_tests {
    ($cx:expr, $cy:expr, $cz:expr, $radius:expr) => {{
        plane_test!(0, $cx, $cy, $cz, $radius);
        plane_test!(1, $cx, $cy, $cz, $radius);
        plane_test!(2, $cx, $cy, $cz, $radius);
        plane_test!(3, $cx, $cy, $cz, $radius);
        plane_test!(4, $cx, $cy, $cz, $radius);
        plane_test!(5, $cx, $cy, $cz, $radius);
    }};
}

fn length3(x: f32, y: f32, z: f32) -> f32 {
    // SAFETY: the import is a pure `vec3` length and takes plain scalars.
    unsafe { gpu_length3_xyz(x, y, z) }
}

/// Sphere against six planes; NaN/negative radii stay visible.
fn sphere_visible(cx: f32, cy: f32, cz: f32, radius: f32) -> bool {
    if radius >= 0.0 {
        plane_tests!(cx, cy, cz, radius);
    }
    true
}

/// One 16-byte record chunk from source `$src` to destination `$dst`.
macro_rules! copy_chunk {
    ($src:expr, $dst:expr, $chunk:literal) => {
        // SAFETY: both bases and the literal chunk stay inside the guarded
        // source/destination record arrays.
        unsafe {
            storage_copy_u32x4(
                0,
                SRC_BINDING,
                ($src + $chunk) as i32,
                0,
                DST_BINDING,
                ($dst + $chunk) as i32,
            );
        }
    };
}

/// Cull one instance and compact visible records into the dense destination.
///
/// Bindings: `0` padded `CullParams` uniform, `1` read-only source
/// `array<vec4<u32>>`, `2` read-write destination of the same shape, `3`
/// five-word indirect args with `instance_count` at word 1. The indirect
/// count is read back by the host after the dispatch.
#[unsafe(no_mangle)]
pub extern "C" fn cs_main(id: u32) {
    // SAFETY: both imports read the vec4-array lengths of their bindings.
    let src_len = unsafe { storage_len_u32x4(0, SRC_BINDING) } as u32 / RECORD_CHUNKS;
    // SAFETY: same as above for the destination binding.
    let dst_len = unsafe { storage_len_u32x4(0, DST_BINDING) } as u32 / RECORD_CHUNKS;
    // SAFETY: field 6 is the u32 count in the declared CullParams metadata.
    let count = unsafe { uniform_field_u32(0, PARAMS_BINDING, PARAMS_SID, COUNT_FIELD) };
    if id >= count {
        return;
    }
    if id >= src_len {
        return;
    }
    let cx = src_f32!(id, 12);
    let cy = src_f32!(id, 13);
    let cz = src_f32!(id, 14);
    let rx = length3(src_f32!(id, 0), src_f32!(id, 1), src_f32!(id, 2));
    let ry = length3(src_f32!(id, 4), src_f32!(id, 5), src_f32!(id, 6));
    let rz = length3(src_f32!(id, 8), src_f32!(id, 9), src_f32!(id, 10));
    let rxy = if rx > ry { rx } else { ry };
    let radius = if rxy > rz { rxy } else { rz };
    if sphere_visible(cx, cy, cz, radius) {
        // SAFETY: word 1 is `instance_count` in the five-word args array.
        let slot = unsafe { storage_atomic_add_u32(0, ARGS_BINDING, 1, 1) } as u32;
        if slot >= dst_len {
            return;
        }
        let src = id * RECORD_CHUNKS;
        let dst = slot * RECORD_CHUNKS;
        copy_chunk!(src, dst, 0);
        copy_chunk!(src, dst, 1);
        copy_chunk!(src, dst, 2);
        copy_chunk!(src, dst, 3);
        copy_chunk!(src, dst, 4);
        copy_chunk!(src, dst, 5);
        copy_chunk!(src, dst, 6);
        copy_chunk!(src, dst, 7);
        copy_chunk!(src, dst, 8);
        copy_chunk!(src, dst, 9);
        copy_chunk!(src, dst, 10);
        copy_chunk!(src, dst, 11);
        copy_chunk!(src, dst, 12);
    }
}
