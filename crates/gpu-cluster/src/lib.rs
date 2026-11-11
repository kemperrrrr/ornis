//! Cluster-assignment compute kernel for the stable Rust → LLVM IR path.
//!
//! One invocation per cluster builds the six half-space planes of that cell
//! from the camera `view_proj`, tests every clustered light's bounding sphere,
//! and claims a span of the shared index list with one atomic. The numbers
//! match `ornis-render`'s `CLUSTER_*` constants; a render test pins them so the
//! two copies cannot drift. Planes are six scalar values, not an array the
//! translator would reject as memory, and `exp` goes through an import because
//! `no_std` has no `f32::exp`.

#![no_std]
#![warn(missing_docs)]

mod light;
mod planes;

use light::bound;
use ornis_gpu_shader::AtomicU32;
use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::ReadWrite;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::gpu_entry;
use planes::Planes;
use planes::cluster_planes;
use planes::outside;

/// Horizontal tiles. Matches `CLUSTER_TILE_COUNT_X`.
pub const TILES_X: u32 = 16;
/// Vertical tiles. Matches `CLUSTER_TILE_COUNT_Y`.
pub const TILES_Y: u32 = 9;
/// Depth slices. Matches `CLUSTER_SLICE_COUNT`.
pub const SLICES: u32 = 24;
/// `TILES_X * TILES_Y * SLICES`.
pub const CLUSTER_COUNT: u32 = TILES_X * TILES_Y * SLICES;
/// Per-cluster index cap. Matches `MAX_LIGHTS_PER_CLUSTER`.
pub const MAX_LIGHTS: u32 = 128;
/// Shared index-list capacity.
pub const INDEX_CAP: u32 = CLUSTER_COUNT * MAX_LIGHTS;
/// First slice boundary, in view depth.
pub const DEPTH_NEAR: f32 = 0.5;
/// Last slice's far plane.
pub const FAR_LIMIT: f32 = 1.0e9;
/// `ln(400 / 0.5)` as `f32`, the same bits `cluster_log_ratio` computes.
pub const LOG_RATIO: f32 = f32::from_bits(1_087_760_471);
/// Spot kind scalar (`LIGHT_KIND_SPOT`).
pub const SPOT: f32 = 2.0;
/// Cone cosine below which a spot uses the point-light sphere.
pub const COS_FLOOR: f32 = 0.1;
/// `f32` words in one `GpuLight`.
pub(crate) const LIGHT_WORDS: u32 = 36;
/// Words in one `ClusterSpan`.
const SPAN_WORDS: u32 = 2;
/// Word of the span cursor in the counters binding.
const CURSOR: u32 = 0;
/// Word of the overflow counter.
const OVERFLOW: u32 = 1;

/// `view_proj` as four column `vec4`s — the prefix of the host `Camera` buffer.
pub struct CameraView;

impl GpuStruct for CameraView {
    /// First uniform struct in the entry metadata.
    const ID: u32 = 0;
    /// Four columns of `view_proj`.
    const FIELD_TAGS: &'static [u8] = &[6, 6, 6, 6];
}

/// `ClusterParams`: light count, head count, the clustered flag, padding.
pub struct ClusterCounts;

impl GpuStruct for ClusterCounts {
    /// Second uniform struct.
    const ID: u32 = 1;
    /// Four `u32` fields, in the host struct's declaration order.
    const FIELD_TAGS: &'static [u8] = &[2, 2, 2, 2];
}

/// Field index of `light_count`.
const LIGHT_COUNT: usize = 0;
/// Field index of `head_count`.
const HEAD_COUNT: usize = 1;

/// How many clustered lights touch `planes`, capped at [`MAX_LIGHTS`].
// qual:allow(test_quality, untested) reason: "shader helper; the host parity test runs the generated WGSL, not this Rust function"
#[inline(always)]
fn count_touching(
    lights: &Storage<f32, 0, 2, ReadOnly>,
    head: u32,
    total: u32,
    faces: &Planes,
) -> u32 {
    let mut count = 0u32;
    let mut index = head;
    while index < total {
        let (center, radius) = bound(lights, index);
        if !outside(center, radius, faces) {
            count += 1;
        }
        index += 1;
    }
    if count < MAX_LIGHTS {
        count
    } else {
        MAX_LIGHTS
    }
}

/// Append the touching lights at `base`, stopping after `span` of them.
// qual:allow(test_quality, untested) reason: "shader helper; the host parity test runs the generated WGSL, not this Rust function"
// qual:allow(srp, max_parameters=7) reason: "scalar loop bounds plus the two storage bindings; a struct argument becomes an alloca the translator rejects"
#[inline(always)]
fn write_indices(
    lights: &Storage<f32, 0, 2, ReadOnly>,
    indices: &Storage<u32, 0, 4, ReadWrite>,
    head: u32,
    total: u32,
    faces: &Planes,
    base: u32,
    span: u32,
) {
    let mut cursor = 0u32;
    let mut index = head;
    while index < total {
        if cursor >= span {
            break;
        }
        let (center, radius) = bound(lights, index);
        if !outside(center, radius, faces) {
            indices.store(base + cursor, index);
            cursor += 1;
        }
        index += 1;
    }
}

/// Assign one cluster: planes, count, atomic span, then the index list.
///
/// Bindings: `0` camera `view_proj`, `1` `ClusterParams`, `2` read-only light
/// words, `3` `ClusterSpan` pairs, `4` the index list, `5` the two atomics.
#[gpu_entry(compute(threads(64)))]
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `export_name`.
// qual:allow(srp, max_parameters=7) reason: "invocation id plus the six bindings of the assignment pass; resource parameters stay in the signature so the attribute can fill them"
pub fn cs_main(
    id: u32,
    #[gpu(uniform, group = 0, binding = 0)] camera: Uniform<CameraView>,
    #[gpu(uniform, group = 0, binding = 1)] params: Uniform<ClusterCounts>,
    #[gpu(storage_buffer, group = 0, binding = 2, read_only)] lights: Storage<f32, ReadOnly>,
    #[gpu(storage_buffer, group = 0, binding = 3)] grid: Storage<u32, ReadWrite>,
    #[gpu(storage_buffer, group = 0, binding = 4)] indices: Storage<u32, ReadWrite>,
    #[gpu(storage_buffer, group = 0, binding = 5)] counters: Storage<AtomicU32, ReadWrite>,
) {
    if id >= CLUSTER_COUNT {
        return;
    }
    let total = params.field_u32::<LIGHT_COUNT>();
    let head = params.field_u32::<HEAD_COUNT>();
    let tx = id % TILES_X;
    let ty = (id / TILES_X) % TILES_Y;
    let tz = id / (TILES_X * TILES_Y);
    let faces = cluster_planes(&camera, tx, ty, tz);
    let count = count_touching(&lights, head, total, &faces);
    let base = counters.atomic_add(CURSOR, count);
    let mut span = count;
    if base + count > INDEX_CAP {
        // `saturating_sub` lowers to `llvm.usub.sat`, which the translator
        // rejects; this branch becomes a `select` of the same value.
        #[allow(clippy::implicit_saturating_sub)]
        {
            span = if base >= INDEX_CAP {
                0
            } else {
                INDEX_CAP - base
            };
        }
        counters.atomic_add(OVERFLOW, count - span);
    }
    grid.store(id * SPAN_WORDS, base);
    grid.store(id * SPAN_WORDS + 1, span);
    write_indices(&lights, &indices, head, total, &faces, base, span);
}
