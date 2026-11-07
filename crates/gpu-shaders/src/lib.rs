//! Index-compaction cull kernel for the stable Rust → LLVM IR path.
//!
//! The crate is a normal `no_std` library: cargo checks it on the host,
//! while the renderer build compiles the same source for
//! `wasm32-unknown-unknown` with `--emit=llvm-ir` and feeds the IR to
//! `ornis-llvm2gpu`. One invocation per source record runs the
//! sphere-vs-six-planes test, reserves a dense slot with the indirect-args
//! atomic, and stores the source index (`u32`). The vertex stage resolves
//! `per_objects[visible_indices[instance_index]]`.

#![no_std]
#![warn(missing_docs)]

use ornis_gpu_shader::gpu_entry;

/// Group-0 binding of the packed `CullParams` uniform.
const PARAMS_BINDING: i32 = 0;
/// `ornis_gpu.types` struct id of `CullParams`.
const PARAMS_SID: i32 = 0;
/// Field index of the `u32` count in `CullParams` (after six vec4 planes).
const COUNT_FIELD: i32 = 6;
/// Group-0 read-only scalar `f32` view of the source record binding.
const SRC_BINDING: i32 = 1;
/// Group-0 read-write compacted `u32` index binding.
const DST_BINDING: i32 = 2;
/// Group-0 five-word indirect-args binding.
const ARGS_BINDING: i32 = 3;
/// `f32` words in one source record (the 52-word `PerObject` layout).
const RECORD_WORDS: u32 = 52;

unsafe extern "C" {
    fn storage_len_f32(group: i32, binding: i32) -> i32;
    fn storage_len_u32(group: i32, binding: i32) -> i32;
    fn storage_read_f32(group: i32, binding: i32, index: i32) -> f32;
    fn storage_write_u32(group: i32, binding: i32, index: i32, value: u32);
    fn storage_atomic_add_u32(group: i32, binding: i32, index: i32, value: i32) -> i32;
    fn uniform_field_vec4_plane_dot(
        group: i32,
        binding: i32,
        sid: i32,
        field: i32,
        x: f32,
        y: f32,
        z: f32,
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
        unsafe { storage_read_f32(0, SRC_BINDING, index as i32) }
    }};
}

/// Signed distance from one point to a `vec4` plane field in `CullParams`.
macro_rules! plane_distance {
    ($plane:expr, $x:expr, $y:expr, $z:expr) => {
        // SAFETY: the plane index is a compile-time constant inside the
        // declared `CullParams` metadata.
        unsafe { uniform_field_vec4_plane_dot(0, PARAMS_BINDING, PARAMS_SID, $plane, $x, $y, $z) }
    };
}

/// One plane test; `$plane` is a literal so the uniform indices are constants.
macro_rules! plane_test {
    ($plane:literal, $cx:expr, $cy:expr, $cz:expr, $radius:expr) => {{
        if $radius >= 0.0 {
            if plane_distance!($plane, $cx, $cy, $cz) < -$radius {
                return;
            }
        }
    }};
}

/// Six unrolled plane tests; each culled plane returns from the entry.
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

/// Cull one instance and append its source index to the dense output.
///
/// Bindings: `0` padded `CullParams` uniform, `1` read-only `f32` source
/// view, `2` read-write compacted `u32` indices, `3` five-word indirect
/// args with `instance_count` at word 1.
#[gpu_entry(compute(threads(64)), uniform_tags = [6, 6, 6, 6, 6, 6, 2])]
#[unsafe(no_mangle)]
pub extern "C" fn cs_main(id: u32) {
    // SAFETY: both imports only read array lengths of their bindings.
    let src_records = unsafe { storage_len_f32(0, SRC_BINDING) } as u32 / RECORD_WORDS;
    // SAFETY: same as above for the index destination.
    let dst_slots = unsafe { storage_len_u32(0, DST_BINDING) } as u32;
    // SAFETY: field 6 is the u32 count in the declared CullParams metadata.
    let count = unsafe { uniform_field_u32(0, PARAMS_BINDING, PARAMS_SID, COUNT_FIELD) };
    if id >= count {
        return;
    }
    if id >= src_records {
        return;
    }
    let m0 = src_f32!(id, 0);
    let m1 = src_f32!(id, 1);
    let m2 = src_f32!(id, 2);
    let m4 = src_f32!(id, 4);
    let m5 = src_f32!(id, 5);
    let m6 = src_f32!(id, 6);
    let m8 = src_f32!(id, 8);
    let m9 = src_f32!(id, 9);
    let m10 = src_f32!(id, 10);
    let m12 = src_f32!(id, 12);
    let m13 = src_f32!(id, 13);
    let m14 = src_f32!(id, 14);
    let rx = length3(m0, m1, m2);
    let ry = length3(m4, m5, m6);
    let rz = length3(m8, m9, m10);
    let rxy = if rx > ry { rx } else { ry };
    let radius = if rxy > rz { rxy } else { rz };
    plane_tests!(m12, m13, m14, radius);
    // SAFETY: word 1 is `instance_count` in the five-word args array.
    let slot = unsafe { storage_atomic_add_u32(0, ARGS_BINDING, 1, 1) } as u32;
    if slot >= dst_slots {
        return;
    }
    // SAFETY: `slot` is below the guarded index capacity.
    unsafe { storage_write_u32(0, DST_BINDING, slot as i32, id) };
}
