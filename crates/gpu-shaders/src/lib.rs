//! Index-compaction cull kernel for the stable Rust → LLVM IR path.
//!
//! The crate is a normal `no_std` library: cargo checks it on the host, while
//! the renderer build compiles the same source for `wasm32-unknown-unknown`
//! with `--emit=llvm-ir` and feeds the IR to `ornis-llvm2gpu`. One invocation
//! per source record runs the sphere-vs-six-planes test, reserves a dense slot
//! with the indirect-args atomic, and stores the source index (`u32`). The
//! vertex stage resolves `per_objects[visible_indices[instance_index]]`.
//!
//! The body touches no raw `ornis_gpu` import: `#[gpu_entry]` binds the
//! `#[gpu(…)]` parameters to typed [`Storage`]/[`Uniform`] values that carry
//! their binding in the type, [`CullParams`] publishes its own field tags
//! through [`GpuStruct`], and the workgroup size comes from the attribute. The
//! translator reads the entry, the workgroup, and the uniform tags back out of
//! the emitted LLVM IR. The translated WGSL is unchanged from the version that
//! called the imports by hand — the typed methods inline to the same calls.

#![no_std]
#![warn(missing_docs)]

use ornis_gpu_shader::AtomicU32;
use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::ReadWrite;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::gpu_entry;

/// `f32` words in one source record (the 52-word `PerObject` layout).
pub const RECORD_WORDS: u32 = 52;

/// Word offset of the x/y/z scale of the model matrix's second row.
const ROW_Y: u32 = 4;
/// Word offset of the x/y/z scale of the model matrix's third row.
const ROW_Z: u32 = 8;
/// Word offset of the record's translation column.
const TRANSLATION: u32 = 12;
/// Word offset of the source index word inside the indirect-args array.
const ARGS_COUNT: u32 = 1;

/// Field tag of a `vec4<f32>` in `ornis_gpu.types`.
const TAG_VEC4: u8 = ornis_gpu_shader::TYPE_TAG_VEC4;
/// Field tag of a `u32` in `ornis_gpu.types`.
const TAG_U32: u8 = ornis_gpu_shader::TYPE_TAG_U32;

/// Field index of the `u32` instance count in [`CullParams`].
pub const COUNT_FIELD: usize = 6;

/// The packed `CullParams` uniform: six normalized frustum planes and the
/// source instance count, laid out exactly like the host `CullParams`
/// (`112` bytes, the count padded to the 16-byte uniform stride).
pub struct CullParams;

impl GpuStruct for CullParams {
    const ID: u32 = 0;
    const FIELD_TAGS: &'static [u8] = &[
        TAG_VEC4, TAG_VEC4, TAG_VEC4, TAG_VEC4, TAG_VEC4, TAG_VEC4, TAG_U32,
    ];
}

/// Length of the `(x, y, z)` vector: the row scale of a model matrix.
///
/// One `length` instruction instead of three multiplies and a software square
/// root: `no_std` has no `f32::sqrt`, and the import lowers to WGSL `length`.
#[inline(always)]
fn length3(x: f32, y: f32, z: f32) -> f32 {
    ornis_gpu_shader::Vec3::new(x, y, z).length()
}

/// One plane test; `$plane` is a literal field index, so the uniform access is
/// a constant and the tag check happens at compile time.
macro_rules! plane_test {
    ($params:expr, $plane:literal, $cx:expr, $cy:expr, $cz:expr, $radius:expr) => {{
        if $radius >= 0.0 {
            if $params.plane_distance::<$plane>($cx, $cy, $cz) < -$radius {
                return;
            }
        }
    }};
}

/// Six unrolled plane tests; each culled plane returns from the entry.
macro_rules! plane_tests {
    ($params:expr, $cx:expr, $cy:expr, $cz:expr, $radius:expr) => {{
        plane_test!($params, 0, $cx, $cy, $cz, $radius);
        plane_test!($params, 1, $cx, $cy, $cz, $radius);
        plane_test!($params, 2, $cx, $cy, $cz, $radius);
        plane_test!($params, 3, $cx, $cy, $cz, $radius);
        plane_test!($params, 4, $cx, $cy, $cz, $radius);
        plane_test!($params, 5, $cx, $cy, $cz, $radius);
    }};
}

/// Cull one instance and append its source index to the dense output.
///
/// Bindings: `0` padded `CullParams` uniform, `1` read-only `f32` source
/// view of `PerObject` records, `2` read-write compacted `u32` indices, `3`
/// five-word indirect args with `instance_count` at word 1.
///
/// A plain safe `fn`: `#[gpu_entry]` attaches the export name, and the resource
/// parameters are zero-sized, so the compiled entry is still `cs_main(i32)`.
#[gpu_entry(compute(threads(64)))]
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "cs_main")]`.
pub fn cs_main(
    id: u32,
    #[gpu(uniform, group = 0, binding = 0)] params: Uniform<CullParams>,
    #[gpu(storage_buffer, group = 0, binding = 1, read_only)] src: Storage<f32, ReadOnly>,
    #[gpu(storage_buffer, group = 0, binding = 2)] dst: Storage<u32, ReadWrite>,
    #[gpu(storage_buffer, group = 0, binding = 3)] args: Storage<AtomicU32, ReadWrite>,
) {
    let src_records = src.len() / RECORD_WORDS;
    let dst_slots = dst.len();
    let count = params.field_u32::<COUNT_FIELD>();
    if id >= count {
        return;
    }
    if id >= src_records {
        return;
    }
    let base = id * RECORD_WORDS;
    let m0 = src.load(base);
    let m1 = src.load(base + 1);
    let m2 = src.load(base + 2);
    let m4 = src.load(base + ROW_Y);
    let m5 = src.load(base + ROW_Y + 1);
    let m6 = src.load(base + ROW_Y + 2);
    let m8 = src.load(base + ROW_Z);
    let m9 = src.load(base + ROW_Z + 1);
    let m10 = src.load(base + ROW_Z + 2);
    let m12 = src.load(base + TRANSLATION);
    let m13 = src.load(base + TRANSLATION + 1);
    let m14 = src.load(base + TRANSLATION + 2);
    let rx = length3(m0, m1, m2);
    let ry = length3(m4, m5, m6);
    let rz = length3(m8, m9, m10);
    let rxy = if rx > ry { rx } else { ry };
    let radius = if rxy > rz { rxy } else { rz };
    plane_tests!(params, m12, m13, m14, radius);
    let slot = args.atomic_add(ARGS_COUNT, 1);
    if slot >= dst_slots {
        return;
    }
    dst.store(slot, id);
}
