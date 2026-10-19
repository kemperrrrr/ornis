//! Groundwork for GPU frustum culling with `draw_indexed_indirect`
//! submits ([RAT-35](/RAT/issues/RAT-35)).
//!
//! CPU frustum culling is opt-in and off by default
//! ([`crate::extraction::cull_frame_upload`]); this module stages its GPU
//! replacement without relying on `multi_draw_indirect` (emulated by a loop
//! on Metal, wgpu #2148, no `MDI_COUNT`): one compute pass compacts the
//! visible instances into a dense buffer and writes one
//! [`DrawIndexedIndirectArgs`], and the gbuffer pass submits it as a
//! single
//! [`draw_indexed_indirect`](wgpu::RenderPass::draw_indexed_indirect)
//! per mesh batch through
//! [`Renderer3D::render_gbuffer_indirect`](crate::renderer::Renderer3D::render_gbuffer_indirect)
//! (pixel-identical to the direct `draw_indexed` fallback, covered by
//! `indirect_draw_gpu`). Shadows stay on the direct pre-pass:
//! off-frame casters can shadow the frame, so the camera-compacted batch
//! is never reused for the shadow maps (a per-light-frustum compaction
//! is future work). The frame plan submits the indirect batch when the
//! opt-in flag armed it for the frame, and stays direct otherwise
//! (including the [`CullerError::CapacityExceeded`](CullerError)
//! fallback).
//!
//! Slice 1: the CPU reference compaction, the indirect-args layout, the
//! compute-shader source and its uniform packing, plus parity tests
//! against the CPU culler. Slice 2: [`Culler`] (owner of the dispatch
//! buffers: staging → compacted → indirect args) and the indirect
//! gbuffer entry point above. Slice 2c: frame-plan selection — the
//! [`RenderBackendConfig::indirect_draw`](crate::render_backend::RenderBackendConfig::indirect_draw)
//! opt-in flag, `Renderer3D::prepare_indirect_batch`
//! (compact-or-direct-fallback) and the gbuffer node routing; shadows
//! stay direct. Slice 3: the GPU benchmark column
//! (`benches/culling_bench.rs` — CPU vs GPU culling at 10k/100k with the
//! camera seeing a small part; run manually, the table goes into the PR).
//!
//! Order contract: the CPU compaction preserves submission order
//! (strictly increasing indices — deterministic, the §24.1 reference).
//! The GPU pass compacts through one atomic counter, so compacted order is
//! dispatch order and **not** deterministic. That is safe for the opaque
//! gbuffer and shadow-depth passes (depth-tested, order-independent
//! output); the sorted transparent forward path keeps the CPU order until
//! a deterministic GPU prefix-sum pass exists.

use glam::Mat4;
use glam::Vec4;

use crate::camera::Frustum;
use crate::extraction::instance_sphere;
use crate::renderer::InstanceData;
use crate::renderer::PerObjectGpu;
use crate::renderer::StagedCustomMesh;
use crate::renderer::staging_capacity_for_instances;

/// Threads per compute workgroup in [`CULL_COMPUTE_WGSL`].
pub const CULL_WORKGROUP_SIZE: u32 = 64;

/// Frustum planes per frame (matches [`Frustum`]).
pub const FRUSTUM_PLANE_COUNT: usize = 6;

/// `u32` words per [`DrawIndexedIndirectArgs`].
pub const INDIRECT_ARGS_WORD_COUNT: usize = 5;

/// Bytes per indirect-args word.
pub const INDIRECT_ARGS_WORD_BYTES: u64 = 4;

/// Word index of `instance_count` inside the indirect-args buffer: the
/// only word the compute pass writes (via `atomicAdd`); the CPU
/// pre-writes the rest and zeroes this word before every dispatch with
/// [`reset_indirect_instance_count`].
pub const INSTANCE_COUNT_WORD: usize = 1;

/// Byte size of the [`CullParams`] uniform block: six `vec4` planes plus
/// the instance count padded to a 16-byte stride.
pub const CULL_PARAMS_BYTE_SIZE: u64 = 112;

/// Pad words after [`CullParams::count`]: the count is one `u32` inside a
/// 16-byte uniform stride.
const CULL_PARAM_PAD_WORDS: usize = 3;

/// `DrawIndexedIndirectArgs` for one
/// [`draw_indexed_indirect`](wgpu::RenderPass::draw_indexed_indirect)
/// submit: five 32-bit words (`index_count`, `instance_count`,
/// `first_index`, `base_vertex`, `first_instance`) — byte-identical to
/// [`wgpu::util::DrawIndexedIndirectArgs`], so `base_vertex` is the
/// signed `i32` of the wgpu/Vulkan/D3D12/Metal layouts.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectArgs {
    /// Indices per draw (mesh `num_indices`, written by the CPU).
    pub index_count: u32,
    /// Visible instances (CPU count, or the compute-pass atomic).
    pub instance_count: u32,
    /// First index into the index buffer (always `0` here).
    pub first_index: u32,
    /// Vertex-buffer base offset (always `0` here). Signed, as in the
    /// wgpu arg layout (`base_vertex` may be negative in the spec).
    pub base_vertex: i32,
    /// First instance slot (always `0`: the compacted buffer is dense).
    pub first_instance: u32,
}

impl DrawIndexedIndirectArgs {
    /// Args for a compacted draw: the full index range over
    /// `visible_instances` dense slots.
    pub fn compacted(index_count: u32, visible_instances: u32) -> Self {
        Self {
            index_count,
            instance_count: visible_instances,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        }
    }
}

/// Uniform block of [`CULL_COMPUTE_WGSL`]: the six normalized frustum
/// planes plus the source instance count, padded to `112` bytes (see
/// [`CULL_PARAMS_BYTE_SIZE`]).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullParams {
    /// Normalized frustum planes (`vec4`: xyz normal, w distance).
    pub planes: [[f32; 4]; FRUSTUM_PLANE_COUNT],
    /// Source instances in the `src` storage buffer.
    pub count: u32,
    /// Padding to the 16-byte uniform stride (never read).
    _pad: [u32; CULL_PARAM_PAD_WORDS],
}

/// Packs the frame culling uniform from `view_proj`: the same six
/// normalized planes [`Frustum`] tests on the CPU, so the compute pass
/// and [`Frustum::sphere_visible`] agree by construction.
///
/// Non-finite planes (degenerate `view_proj`, unusable normals) are
/// replaced with the always-passing `(0, 0, 0, f32::MAX)` so the GPU
/// fails open exactly like [`Frustum::sphere_visible`], which skips
/// them on the CPU.
pub fn cull_params(view_proj: &Mat4, count: u32) -> CullParams {
    let planes: [Vec4; FRUSTUM_PLANE_COUNT] = Frustum::from_view_proj(view_proj).planes();
    CullParams {
        planes: planes.map(|plane| {
            if plane.is_finite() {
                plane.to_array()
            } else {
                [0.0, 0.0, 0.0, f32::MAX]
            }
        }),
        count,
        _pad: [0; CULL_PARAM_PAD_WORDS],
    }
}

/// Deterministic CPU compaction (the GPU pass reference and fallback):
/// indices of `instances` visible in `view_proj`, in increasing order.
///
/// Uses the same sphere test as
/// [`crate::extraction::cull_frame_upload`] (see
/// [`instance_sphere`]) without
/// mutating the input, so the kept set always matches the destructive
/// CPU culler. Cost is `O(n)` sphere tests.
pub fn compact_visible_indices(instances: &[InstanceData], view_proj: &Mat4) -> Vec<u32> {
    let frustum = Frustum::from_view_proj(view_proj);
    instances
        .iter()
        .enumerate()
        .filter_map(|(index, instance)| {
            let (center, radius) = instance_sphere(instance);
            frustum
                .sphere_visible(center, radius)
                .then_some(index as u32)
        })
        .collect()
}

/// Indirect args for a CPU-compacted draw: the full index range over the
/// `visible` dense slots (an empty set is a valid no-op draw).
pub fn indirect_args_for_visible(index_count: u32, visible: &[u32]) -> DrawIndexedIndirectArgs {
    DrawIndexedIndirectArgs::compacted(index_count, visible.len() as u32)
}

/// One dirty run of the resident staging buffer: `count` records
/// starting at `start` (record indices, not bytes) whose content differs
/// from the last uploaded frame and must be re-written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirtyRange {
    /// First differing record.
    pub(crate) start: u32,
    /// Records in the run.
    pub(crate) count: u32,
}

/// Contiguous record runs where `next` (already converted) differs
/// from `prev`, compared byte-wise: the tail past the shared prefix
/// counts as dirty on growth, while shrinkage writes nothing (the
/// stale tail is never read — `params.count` bounds the threads).
/// Empty when the frames are identical. This is the one dirty-range
/// implementation in the crate: [`stage_next_instances`] composes it
/// over a fresh conversion, and [`Culler::compact_records`] feeds it
/// caller-converted frames, so both callers share one code path.
pub(crate) fn diff_staged_records(prev: &[PerObjectGpu], next: &[PerObjectGpu]) -> Vec<DirtyRange> {
    let shared = prev.len().min(next.len());
    let mut ranges = Vec::new();
    let mut run: Option<DirtyRange> = None;
    let flush = |run: &mut Option<DirtyRange>, ranges: &mut Vec<DirtyRange>| {
        if let Some(range) = run.take() {
            ranges.push(range);
        }
    };
    for index in 0..shared {
        if bytemuck::bytes_of(&prev[index]) == bytemuck::bytes_of(&next[index]) {
            flush(&mut run, &mut ranges);
        } else if let Some(range) = run.as_mut() {
            range.count += 1;
        } else {
            run = Some(DirtyRange {
                start: index as u32,
                count: 1,
            });
        }
    }
    flush(&mut run, &mut ranges);
    if next.len() > shared {
        let tail = (next.len() - shared) as u32;
        match ranges.last_mut() {
            Some(last) if last.start + last.count == shared as u32 => {
                last.count += tail;
            }
            _ => ranges.push(DirtyRange {
                start: shared as u32,
                count: tail,
            }),
        }
    }
    ranges
}

/// Rebuilds `next` from `instances` and returns the dirty runs against
/// `prev` (empty when the frames are identical — a static scene uploads
/// nothing). Indices past `prev` count as dirty (growth uploads the
/// extension); a shorter `next` leaves a stale tail the dispatch never
/// reads (`params.count` bounds the threads).
///
/// This is a thin composition of [`convert_instances`] and
/// [`diff_staged_records`] — the same two steps [`Culler::compact_records`]
/// runs on caller-converted frames — so the staging tests cover the
/// production dirty-range path rather than a parallel implementation.
pub(crate) fn stage_next_instances(
    prev: &[PerObjectGpu],
    next: &mut Vec<PerObjectGpu>,
    instances: &[InstanceData],
) -> Vec<DirtyRange> {
    next.clear();
    convert_instances(next, instances);
    diff_staged_records(prev, next)
}

/// Convert CPU instances into GPU records appended to `out`: one
/// straight pass, no branches (pure delegation for the gate — the
/// caller owns reuse and allocation). Shared by the direct upload and
/// the submit path so a frame converts once.
pub(crate) fn convert_instances(out: &mut Vec<PerObjectGpu>, instances: &[InstanceData]) {
    out.extend(instances.iter().map(|inst| {
        let mut record = PerObjectGpu::default();
        record.model = inst.model_matrix.to_cols_array_2d();
        record.normal_matrix = inst.normal_matrix.to_cols_array_2d();
        record.material_index = inst.material_index;
        record
    }));
}

/// Merge sphere and custom instances and convert them once: the merged
/// records feed the direct upload, and the sphere prefix (customs
/// follow in dense lane order) stages the shared batch without
/// converting twice. One allocation, no branches.
pub(crate) fn merge_and_convert_instances(
    spheres: &[InstanceData],
    customs: &[StagedCustomMesh],
) -> Vec<PerObjectGpu> {
    let mut instances = Vec::with_capacity(spheres.len().saturating_add(customs.len()));
    instances.extend_from_slice(spheres);
    instances.extend(customs.iter().map(|entry| entry.instance));
    let mut converted = Vec::with_capacity(staging_capacity_for_instances(instances.len()));
    convert_instances(&mut converted, &instances);
    converted
}

/// Zeroes the `instance_count` word ([`INSTANCE_COUNT_WORD`]) of an
/// indirect-args buffer before a culling dispatch. The compute pass only
/// ever `atomicAdd`s to that word, so every frame must restart it at
/// zero — without this reset the counter accumulates across dispatches.
pub fn reset_indirect_instance_count(queue: &wgpu::Queue, args: &wgpu::Buffer) {
    queue.write_buffer(
        args,
        INSTANCE_COUNT_WORD as u64 * INDIRECT_ARGS_WORD_BYTES,
        &0_u32.to_le_bytes(),
    );
}

/// Workgroups per culling dispatch for `instance_count` source instances
/// at [`CULL_WORKGROUP_SIZE`] threads (one thread per instance; the
/// shader guards the ragged tail against `params.count`).
pub fn cull_workgroup_count(instance_count: u32) -> u32 {
    instance_count.div_ceil(CULL_WORKGROUP_SIZE)
}

/// WGSL compute pass compacting visible instances: one thread per source
/// instance runs the same sphere-vs-six-planes test as
/// [`compact_visible_indices`] (IEEE comparisons fail open on NaN, like
/// the CPU path), claims a dense slot with `atomicAdd` on indirect-args
/// word [`INSTANCE_COUNT_WORD`], and copies the full `PerObject` record.
///
/// Bindings: `0` [`CullParams`] uniform, `1` source `PerObject` array
/// (read), `2` dense destination (read-write), `3` five-word indirect
/// args as `array<atomic<u32>, 5>` (read-write; the CPU pre-writes every
/// word but `instance_count` and zeroes it). `PerObject` repeats
/// the [`crate::renderer::PerObjectGpu`] layout (three `mat4x4`, two `u32`
/// fields and two pad words = 208 bytes) so the existing vertex stages index
/// the compacted buffer unchanged.
pub const CULL_COMPUTE_WGSL: &str = r#"
struct CullParams {
    planes: array<vec4<f32>, 6>,
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct PerObject {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    previous_model: mat4x4<f32>,
    material_index: u32,
    motion_valid: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: CullParams;
@group(0) @binding(1) var<storage, read> src: array<PerObject>;
@group(0) @binding(2) var<storage, read_write> dst: array<PerObject>;
@group(0) @binding(3) var<storage, read_write> args: array<atomic<u32>, 5>;

fn instance_sphere(model: mat4x4<f32>) -> vec4<f32> {
    let center = model[3].xyz;
    let rx = length(model[0].xyz);
    let ry = length(model[1].xyz);
    let rz = length(model[2].xyz);
    return vec4<f32>(center, max(rx, max(ry, rz)));
}

fn sphere_visible(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane = params.planes[i];
        if (dot(plane.xyz, center) + plane.w < -radius) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn cull_instances(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.count || i >= arrayLength(&src)) {
        return;
    }
    let sphere = instance_sphere(src[i].model);
    if (!sphere_visible(sphere.xyz, sphere.w)) {
        return;
    }
    let slot = atomicAdd(&args[1u], 1u);
    if (slot >= arrayLength(&dst)) {
        return;
    }
    dst[slot] = src[i];
}
"#;

mod culler;

pub(crate) use culler::PreviousInstanceData;
pub use culler::{Culler, CullerError};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::CameraView;
    use crate::extraction::cull_frame_upload;
    use crate::extraction::extract_render_data;
    use crate::renderer::MaterialIdx;
    use crate::renderer::PerObjectGpu;
    use ornis_assets::scene::MaterialDesc;
    use ornis_assets::scene::MeshDesc;
    use ornis_assets::scene::TransformDesc;
    use ornis_core::Engine;
    use ornis_core::units::Clamped01;
    use ornis_core::units::Degrees;
    use ornis_core::units::Meters;
    use ornis_core::units::PositiveF32;

    /// Test sphere mesh (mirrors the extraction test helper).
    fn test_sphere() -> MeshDesc {
        MeshDesc::Sphere {
            radius: PositiveF32::expect_valid(1.0),
            segments: 16,
            rings: 12,
        }
    }

    /// Test dielectric material (mirrors the extraction test helper).
    fn test_material() -> MaterialDesc {
        MaterialDesc::Dielectric {
            base_color: [0.8, 0.2, 0.2],
            roughness: Clamped01::new(0.4),
            emission: [0.0, 0.0, 0.0],
            metallic: ornis_core::Metallic::new(0.0),
        }
    }

    /// Perspective look-at frame matching the extraction cull tests.
    fn perspective_view() -> CameraView {
        CameraView {
            eye: glam::Vec3::new(0.0, 2.5, 9.0),
            target: glam::Vec3::ZERO,
            up: glam::Vec3::Y,
            projection: ornis_assets::scene::CameraProjection::Perspective,
            fov: Degrees::new(60.0),
            near: Meters::new(0.1),
            far: Meters::new(100.0),
        }
    }

    /// 100-sphere scene: 50 near the origin (visible), 50 at `x = 100`
    /// (offscreen), mirroring the extraction cull test.
    fn hundred_sphere_upload() -> crate::extraction::FrameUpload {
        let mut engine = Engine::new();
        for i in 0..100 {
            let x = if i < 50 { i as f32 * 0.05 } else { 100.0 };
            let store = engine.world_mut().store_mut().expect("store");
            let handle = store.create_entity();
            store.insert(
                handle,
                TransformDesc::from_translation(glam::Vec3::new(x, 0.0, 0.0)),
            );
            store.insert(handle, test_sphere());
            store.insert(handle, test_material());
        }
        extract_render_data(engine.world().store().expect("store"))
    }

    #[test]
    fn cpu_compaction_matches_cull_frame_upload() {
        // The index-based compaction must keep exactly the set the
        // destructive CPU culler keeps (the GPU pass reference gate).
        let upload = hundred_sphere_upload();
        assert_eq!(upload.instances.len(), 100);
        let view = perspective_view();
        let (view_proj, _) = crate::camera::camera_view_projection(&view, (160, 90));

        let visible = compact_visible_indices(&upload.instances, &view_proj);
        let mut culled = upload.clone();
        let stats = cull_frame_upload(&mut culled, &view_proj);

        assert_eq!(visible.len(), stats.kept);
        assert_eq!(stats.culled, 50, "{stats:?}");
        assert_eq!(stats.kept, 50, "{stats:?}");
        let compacted: Vec<glam::Vec4> = visible
            .iter()
            .map(|index| upload.instances[*index as usize].model_matrix.w_axis)
            .collect();
        let expected: Vec<glam::Vec4> = culled
            .instances
            .iter()
            .map(|instance| instance.model_matrix.w_axis)
            .collect();
        assert_eq!(compacted, expected);
    }

    #[test]
    fn compaction_is_deterministic_and_ordered() {
        // Reference order for §24.1: strictly increasing indices, stable
        // across runs.
        let upload = hundred_sphere_upload();
        let view = perspective_view();
        let (view_proj, _) = crate::camera::camera_view_projection(&view, (160, 90));
        let first = compact_visible_indices(&upload.instances, &view_proj);
        let second = compact_visible_indices(&upload.instances, &view_proj);
        assert_eq!(first, second);
        let mut sorted = first.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(first, sorted, "indices strictly increasing, no repeats");
    }

    #[test]
    fn empty_scene_yields_empty_indirect_draw() {
        let view = perspective_view();
        let (view_proj, _) = crate::camera::camera_view_projection(&view, (160, 90));
        let visible = compact_visible_indices(&[], &view_proj);
        assert!(visible.is_empty());
        let args = indirect_args_for_visible(42, &visible);
        assert_eq!(
            args,
            DrawIndexedIndirectArgs {
                index_count: 42,
                instance_count: 0,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            }
        );
    }

    #[test]
    fn indirect_args_layout_matches_wgpu() {
        // Five 32-bit words, 20 bytes, field order = wgpu's
        // `DrawIndexedIndirectArgs` (`index_count`, `instance_count`,
        // `first_index`, `base_vertex`, `first_instance`); `base_vertex`
        // is the signed `i32` of that layout.
        assert_eq!(std::mem::size_of::<DrawIndexedIndirectArgs>(), 20);
        assert_eq!(
            std::mem::size_of::<DrawIndexedIndirectArgs>(),
            INDIRECT_ARGS_WORD_COUNT * 4
        );
        assert_eq!(
            std::mem::offset_of!(DrawIndexedIndirectArgs, index_count),
            0
        );
        assert_eq!(
            std::mem::offset_of!(DrawIndexedIndirectArgs, instance_count),
            4
        );
        assert_eq!(
            std::mem::offset_of!(DrawIndexedIndirectArgs, first_index),
            8
        );
        assert_eq!(
            std::mem::offset_of!(DrawIndexedIndirectArgs, base_vertex),
            12
        );
        assert_eq!(
            std::mem::offset_of!(DrawIndexedIndirectArgs, first_instance),
            16
        );
        let args = DrawIndexedIndirectArgs::compacted(7, 0);
        assert_eq!(args.index_count, 7);
        assert_eq!(args.instance_count, 0);
        assert_eq!(bytemuck::bytes_of(&args).len(), 20);
    }

    #[test]
    fn indirect_args_bytes_match_wgpu() {
        // Byte-for-byte identity with `wgpu::util::DrawIndexedIndirectArgs`
        // (including the signed `base_vertex`), so a buffer written from
        // this type is a valid `draw_indexed_indirect` args buffer.
        let ours = DrawIndexedIndirectArgs {
            index_count: 3,
            instance_count: 5,
            first_index: 7,
            base_vertex: -11,
            first_instance: 13,
        };
        let theirs = wgpu::util::DrawIndexedIndirectArgs {
            index_count: 3,
            instance_count: 5,
            first_index: 7,
            base_vertex: -11,
            first_instance: 13,
        };
        assert_eq!(bytemuck::bytes_of(&ours), theirs.as_bytes());
    }

    #[test]
    fn cull_params_carry_frustum_planes() {
        // The uniform packs exactly the planes the CPU tests evaluate.
        let view = perspective_view();
        let (view_proj, _) = crate::camera::camera_view_projection(&view, (160, 90));
        let params = cull_params(&view_proj, 100);
        assert_eq!(params.count, 100);
        assert_eq!(
            std::mem::size_of::<CullParams>(),
            CULL_PARAMS_BYTE_SIZE as usize
        );
        assert_eq!(std::mem::offset_of!(CullParams, count), 96);
        let expected = Frustum::from_view_proj(&view_proj).planes();
        for (packed, plane) in params.planes.iter().zip(expected.iter()) {
            assert_eq!(*packed, plane.to_array());
        }
    }

    #[test]
    fn cull_params_replaces_non_finite_planes_with_pass_through() {
        // A degenerate view-proj leaves unusable plane normals (`Frustum`
        // marks them NaN and fails open); the uniform must carry
        // always-passing planes so the GPU test matches that rule
        // instead of culling on garbage comparisons.
        let params = cull_params(&Mat4::ZERO, 3);
        for plane in &params.planes {
            assert_eq!(*plane, [0.0, 0.0, 0.0, f32::MAX]);
        }
        let upload = hundred_sphere_upload();
        let visible = compact_visible_indices(&upload.instances, &Mat4::ZERO);
        assert_eq!(visible.len(), 100, "CPU fails open on a degenerate frustum");
    }

    #[test]
    fn cull_workgroup_count_covers_ragged_tail() {
        assert_eq!(cull_workgroup_count(0), 0);
        assert_eq!(cull_workgroup_count(1), 1);
        assert_eq!(cull_workgroup_count(CULL_WORKGROUP_SIZE), 1);
        assert_eq!(cull_workgroup_count(CULL_WORKGROUP_SIZE + 1), 2);
    }

    #[test]
    fn compute_shader_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(CULL_COMPUTE_WGSL)
            .expect("cull compute shader must parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("cull compute shader must validate");
    }

    /// Adapter handle, or `None` on headless hosts without a GPU. The
    /// skip reason is logged so a silent no-op pass is visible in the
    /// test output instead of looking like a green GPU run. When
    /// `ORNIS_REQUIRE_GPU` is set (see [`crate::gpu_required`]) a
    /// missing adapter panics instead of skipping.
    fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::empty(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            });
            let adapter = match instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
            {
                Ok(adapter) => adapter,
                Err(err) => {
                    assert!(
                        !crate::gpu_required(),
                        "ORNIS_REQUIRE_GPU is set but no wgpu adapter: {err}"
                    );
                    eprintln!("SKIPPED (gpu_compaction): no wgpu adapter: {err}");
                    return None;
                }
            };
            match adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
            {
                Ok(pair) => Some(pair),
                Err(err) => {
                    assert!(
                        !crate::gpu_required(),
                        "ORNIS_REQUIRE_GPU is set but no wgpu device: {err}"
                    );
                    eprintln!("SKIPPED (gpu_compaction): no wgpu device: {err}");
                    None
                }
            }
        })
    }

    /// Reference scene instances, the frame matrix and the CPU
    /// reference translations of the visible set for the culler tests.
    fn cull_test_scene() -> (Vec<InstanceData>, Mat4, Vec<[f32; 4]>) {
        let upload = hundred_sphere_upload();
        let view = perspective_view();
        let (view_proj, _) = crate::camera::camera_view_projection(&view, (160, 90));
        let expected = compact_visible_indices(&upload.instances, &view_proj)
            .iter()
            .map(|index| {
                upload.instances[*index as usize]
                    .model_matrix
                    .w_axis
                    .to_array()
            })
            .collect::<Vec<[f32; 4]>>();
        (upload.instances, view_proj, expected)
    }

    /// Indirect-args record read back from `args` (`None` when the
    /// headless readback fails — the caller skips instead of failing).
    fn cull_test_read_args(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        args: &wgpu::Buffer,
    ) -> Option<DrawIndexedIndirectArgs> {
        let raw = cull_test_read_bytes(
            device,
            queue,
            args,
            INDIRECT_ARGS_WORD_COUNT as u64 * INDIRECT_ARGS_WORD_BYTES,
        )?;
        Some(bytemuck::pod_read_unaligned(
            &raw[..std::mem::size_of::<DrawIndexedIndirectArgs>()],
        ))
    }

    /// Copies `bytes` from `src` into a mapping buffer and reads them
    /// back (`None` when the headless readback fails — the caller skips
    /// instead of failing).
    fn cull_test_read_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &wgpu::Buffer,
        bytes: u64,
    ) -> Option<Vec<u8>> {
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull test readback"),
            size: bytes.max(INDIRECT_ARGS_WORD_BYTES),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cull test readback encoder"),
        });
        encoder.copy_buffer_to_buffer(src, 0, &readback, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));
        let slice = readback.slice(0..bytes.max(INDIRECT_ARGS_WORD_BYTES));
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
            return None;
        }
        let Ok(Ok(())) = rx.recv() else {
            return None;
        };
        let Ok(view) = slice.get_mapped_range() else {
            return None;
        };
        let out = view[..bytes as usize].to_vec();
        drop(view);
        readback.unmap();
        Some(out)
    }

    /// First `count` compacted records read back from the destination
    /// buffer (`None` when the headless readback fails — the caller
    /// skips instead of failing).
    fn cull_test_read_records(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dst: &wgpu::Buffer,
        count: usize,
    ) -> Option<Vec<PerObjectGpu>> {
        let bytes = (std::mem::size_of::<PerObjectGpu>() * count) as u64;
        let raw = cull_test_read_bytes(device, queue, dst, bytes)?;
        Some(
            raw.chunks_exact(std::mem::size_of::<PerObjectGpu>())
                .map(bytemuck::pod_read_unaligned::<PerObjectGpu>)
                .collect(),
        )
    }

    /// GPU output set comparison: sort the model translations and line
    /// them up against the CPU reference (GPU compaction order is
    /// intentionally unchecked — atomics are nondeterministic, see the
    /// module docs).
    fn assert_same_visible_set(got: &[[f32; 4]], want: &[[f32; 4]]) {
        let mut got = got.to_vec();
        let mut want = want.to_vec();
        got.sort_by(|a, b| a[0].total_cmp(&b[0]));
        want.sort_by(|a, b| a[0].total_cmp(&b[0]));
        assert_eq!(got, want);
    }

    /// Visible-set assertion: reads back `count` compacted records from
    /// `dst` and checks the leading `expected.len()` against the CPU
    /// reference (order unchecked — atomics are nondeterministic, see
    /// module docs).
    fn assert_visible_records_match(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dst: &wgpu::Buffer,
        count: usize,
        expected: &[[f32; 4]],
    ) {
        let records =
            cull_test_read_records(device, queue, dst, count).expect("dst readback succeeds");
        let got = records[..expected.len()]
            .iter()
            .map(|record| record.model[3])
            .collect::<Vec<[f32; 4]>>();
        assert_same_visible_set(&got, expected);
    }

    /// Mesh index-count placeholder for the culler tests: carried
    /// through into the indirect args untouched (mirrors the bench
    /// `INDEX_COUNT`).
    const TEST_INDEX_COUNT: u32 = 42;

    /// Full frame through a `Culler`: stage with [`Culler::compact`],
    /// then dispatch with [`Culler::record_compact`] into a submitted
    /// encoder (the production frame records into its own encoder
    /// instead of submitting here).
    fn cull_test_compact(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        culler: &Culler,
        view_proj: &Mat4,
        instances: &[InstanceData],
    ) {
        culler
            .compact(queue, view_proj, TEST_INDEX_COUNT, instances)
            .expect("instances fit the capacity");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cull test compact"),
        });
        culler.record_compact(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));
    }

    #[test]
    fn gpu_compaction_matches_cpu_across_frames() {
        // End-to-end through the production `Culler`, two frames back
        // to back: the args counter must equal the CPU reference count,
        // the compacted `visible_buffer` content must match the CPU
        // visible set (order is intentionally unchecked — atomics are
        // nondeterministic, see module docs), and frame two must not
        // accumulate over frame one.
        //
        // Visible spheres are also appended past the workgroup boundary:
        // the dispatch width comes from
        // `cull_workgroup_count(staged_count)`, and a scene whose visible
        // instances all fit in workgroup 0 cannot notice a width that is
        // too small. With at least one visible index at or above
        // [`CULL_WORKGROUP_SIZE`], dropping a workgroup drops those
        // instances from the compacted buffer, and both the counter and
        // the visible set below go red.
        let Some((device, queue)) = try_device() else {
            return;
        };
        // The shader `PerObject` is the 144-byte `PerObjectGpu` layout
        // (two `mat4x4` + material `u32` + 3 pad words).
        assert_eq!(std::mem::size_of::<PerObjectGpu>(), 144);
        assert_eq!(std::mem::offset_of!(PerObjectGpu, material_index), 128);
        let (mut instances, view_proj, expected) = cull_test_scene();
        // Workgroup 0 covers indices `0..CULL_WORKGROUP_SIZE`, so three
        // spheres straddling that boundary put the visible set across two
        // workgroups.
        for index in [
            CULL_WORKGROUP_SIZE - 1,
            CULL_WORKGROUP_SIZE,
            CULL_WORKGROUP_SIZE + 1,
        ] {
            let x = index as f32 * 0.05;
            while instances.len() <= index as usize {
                instances.push(test_instance(100.0));
            }
            instances[index as usize] = test_instance(x);
        }
        let visible = compact_visible_indices(&instances, &view_proj);
        assert!(
            visible.iter().any(|index| *index >= CULL_WORKGROUP_SIZE),
            "premise: the compacted set must span a second workgroup"
        );
        assert_eq!(
            visible.len(),
            expected.len() + 3,
            "the appended spheres must be visible, not culled"
        );
        let expected = visible
            .iter()
            .map(|index| instances[*index as usize].model_matrix.w_axis.to_array())
            .collect::<Vec<[f32; 4]>>();
        let culler = Culler::try_new(&device, instances.len() as u32).expect("capacity above zero");
        for frame in 0..2 {
            cull_test_compact(&device, &queue, &culler, &view_proj, &instances);
            let args = cull_test_read_args(&device, &queue, &culler.args_buffer)
                .expect("args readback succeeds");
            assert_eq!(args.index_count, 42, "frame {frame}");
            assert_eq!(
                args.instance_count,
                expected.len() as u32,
                "frame {frame}: the counter must restart at zero each frame"
            );
            assert_visible_records_match(
                &device,
                &queue,
                &culler.visible_buffer,
                instances.len(),
                &expected,
            );
        }
    }

    #[test]
    fn culler_rejects_empty_capacity_and_overflow() {
        let Some((device, queue)) = try_device() else {
            return;
        };
        assert!(matches!(
            Culler::try_new(&device, 0).map(|_| ()),
            Err(CullerError::EmptyCapacity)
        ));
        let (instances, view_proj, _) = cull_test_scene();
        let culler = Culler::try_new(&device, 3).expect("capacity above zero");
        assert_eq!(culler.capacity, 3);
        assert_eq!(
            culler.compact(&queue, &view_proj, 0, &instances),
            Err(CullerError::CapacityExceeded {
                count: instances.len() as u32,
                capacity: 3,
            })
        );
    }

    /// Test instance with translation `x` (staging input builder for
    /// the resident-upload tests below).
    fn test_instance(x: f32) -> InstanceData {
        InstanceData {
            model_matrix: Mat4::from_translation(glam::Vec3::new(x, 0.0, 0.0)),
            normal_matrix: Mat4::IDENTITY,
            material_index: MaterialIdx::from_raw(0),
        }
    }

    /// One fused staging frame: converts `instances` into a fresh
    /// buffer against `prev`, returning the buffer and its dirty runs.
    fn stage_frame(
        prev: &[PerObjectGpu],
        instances: &[InstanceData],
    ) -> (Vec<PerObjectGpu>, Vec<super::DirtyRange>) {
        let mut next = Vec::new();
        let ranges = super::stage_next_instances(prev, &mut next, instances);
        (next, ranges)
    }

    /// Base scene of `count` translated instances plus its staged
    /// previous frame (shared arrange-step of the staging tests).
    fn stage_base(count: usize) -> (Vec<InstanceData>, Vec<PerObjectGpu>) {
        let base: Vec<InstanceData> = (0..count).map(|x| test_instance(x as f32)).collect();
        let (prev, _) = stage_frame(&[], &base);
        (base, prev)
    }

    #[test]
    fn staging_covers_first_frame_and_steady_state() {
        use super::DirtyRange;
        let (instances, _) = stage_base(4);
        let (next, ranges) = stage_frame(&[], &instances);
        assert_eq!(
            ranges,
            vec![DirtyRange { start: 0, count: 4 }],
            "first frame uploads everything"
        );
        let (_, steady) = stage_frame(&next, &instances);
        assert!(steady.is_empty(), "identical frames upload nothing");
        let (_, empty) = stage_frame(&[], &[]);
        assert!(empty.is_empty());
    }

    #[test]
    fn staging_isolates_changed_runs() {
        use super::DirtyRange;
        let (base, prev) = stage_base(6);
        let mut one = base.clone();
        one[2] = test_instance(100.0);
        assert_eq!(
            stage_frame(&prev, &one).1,
            vec![DirtyRange { start: 2, count: 1 }]
        );
        let mut two = base.clone();
        two[1] = test_instance(100.0);
        two[4] = test_instance(200.0);
        assert_eq!(
            stage_frame(&prev, &two).1,
            vec![
                DirtyRange { start: 1, count: 1 },
                DirtyRange { start: 4, count: 1 },
            ]
        );
        let mut adjacent = base.clone();
        adjacent[3] = test_instance(100.0);
        adjacent[4] = test_instance(200.0);
        assert_eq!(
            stage_frame(&prev, &adjacent).1,
            vec![DirtyRange { start: 3, count: 2 }],
            "adjacent changes merge into one write"
        );
    }

    #[test]
    fn staging_handles_growth_and_shrinkage() {
        use super::DirtyRange;
        let (base, prev) = stage_base(3);
        let mut grown = base.clone();
        grown.push(test_instance(3.0));
        grown.push(test_instance(4.0));
        assert_eq!(
            stage_frame(&prev, &grown).1,
            vec![DirtyRange { start: 3, count: 2 }],
            "growth uploads the extension"
        );
        let mut grown_changed = base.clone();
        grown_changed[2] = test_instance(100.0);
        grown_changed.push(test_instance(3.0));
        assert_eq!(
            stage_frame(&prev, &grown_changed).1,
            vec![DirtyRange { start: 2, count: 2 }],
            "a trailing change merges with the extension"
        );
        let (grown_prev, _) = stage_frame(&prev, &grown);
        assert!(
            stage_frame(&grown_prev, &base).1.is_empty(),
            "shrinkage writes nothing (the tail is never read)"
        );
    }

    #[test]
    fn diff_staged_records_covers_first_steady_growth() {
        use super::{DirtyRange, diff_staged_records};
        let base: Vec<InstanceData> = (0..3).map(|x| test_instance(x as f32)).collect();
        let mut first = Vec::new();
        super::convert_instances(&mut first, &base);
        assert_eq!(
            diff_staged_records(&[], &first),
            vec![DirtyRange { start: 0, count: 3 }],
            "first frame uploads everything"
        );
        assert!(
            diff_staged_records(&first, &first).is_empty(),
            "identical frames upload nothing"
        );
        let mut grown = base.clone();
        grown.push(test_instance(3.0));
        let mut grown_records = Vec::new();
        super::convert_instances(&mut grown_records, &grown);
        assert_eq!(
            diff_staged_records(&first, &grown_records),
            vec![DirtyRange { start: 3, count: 1 }],
            "growth uploads the extension"
        );
    }

    #[test]
    fn compact_records_matches_compact() {
        // Caller-converted staging must compact exactly like `compact`:
        // same args record and same visible set (order unchecked —
        // atomics are nondeterministic, see module docs).
        let Some((device, queue)) = try_device() else {
            return;
        };
        let (instances, view_proj, expected) = cull_test_scene();
        let from_instances =
            Culler::try_new(&device, instances.len() as u32).expect("capacity above zero");
        let from_records =
            Culler::try_new(&device, instances.len() as u32).expect("capacity above zero");
        cull_test_compact(&device, &queue, &from_instances, &view_proj, &instances);
        let mut converted = Vec::with_capacity(instances.len());
        super::convert_instances(&mut converted, &instances);
        from_records
            .compact_records(&queue, &view_proj, TEST_INDEX_COUNT, &converted)
            .expect("records fit the capacity");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compact records dispatch"),
        });
        from_records.record_compact(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));
        let args_records = cull_test_read_args(&device, &queue, &from_records.args_buffer)
            .expect("args readback succeeds");
        let args_compact = cull_test_read_args(&device, &queue, &from_instances.args_buffer)
            .expect("args readback succeeds");
        assert_eq!(args_records, args_compact);
        assert_eq!(args_records.instance_count, expected.len() as u32);
        assert_visible_records_match(
            &device,
            &queue,
            &from_records.visible_buffer,
            instances.len(),
            &expected,
        );
    }

    #[test]
    fn gpu_compaction_tracks_moved_instances() {
        // The resident upload must track changes: move visible instances
        // offscreen and the next compact must match the CPU reference of
        // the moved scene, not the first frame.
        //
        // The moved set deliberately covers what the review flagged as
        // uncovered on the production staging path:
        //   - index 0 (the only case before; a `index > 0` guard that
        //     skips every later record passes it),
        //   - an index above zero, and
        //   - an adjacent pair, which a run-merging bug that under-counts
        //     dirty records would mis-upload.
        // A disjoint change plus growth would additionally catch a
        // trailing-extension bug, but that needs a larger-capacity culler
        // than this scene; the unit cases above cover the pure diff.
        let Some((device, queue)) = try_device() else {
            return;
        };
        let (mut instances, view_proj, _) = cull_test_scene();
        let culler = Culler::try_new(&device, instances.len() as u32).expect("capacity above zero");
        cull_test_compact(&device, &queue, &culler, &view_proj, &instances);
        let offscreen = Mat4::from_translation(glam::Vec3::new(100.0, 0.0, 0.0));
        instances[0].model_matrix = offscreen;
        instances[3].model_matrix = offscreen;
        instances[4].model_matrix = offscreen;
        let expected = compact_visible_indices(&instances, &view_proj)
            .iter()
            .map(|index| instances[*index as usize].model_matrix.w_axis.to_array())
            .collect::<Vec<[f32; 4]>>();
        cull_test_compact(&device, &queue, &culler, &view_proj, &instances);
        let args = cull_test_read_args(&device, &queue, &culler.args_buffer)
            .expect("args readback succeeds");
        assert_eq!(args.instance_count, expected.len() as u32);
        assert_visible_records_match(
            &device,
            &queue,
            &culler.visible_buffer,
            instances.len(),
            &expected,
        );
    }

    /// Drives [`Culler::compact_records`] — the entry
    /// [`prepare_indirect_batch`](crate::renderer::Renderer3D::prepare_indirect_batch)
    /// takes on the production submit path — with the same moved-visible
    /// scene as [`gpu_compaction_tracks_moved_instances`], so the dirty
    /// runs that path re-writes are covered by a dispatch and a readback,
    /// not only by the shared unit cases of [`diff_staged_records`].
    ///
    /// [`gpu_compaction_tracks_moved_instances`] exercises [`Culler::compact`],
    /// which converts then diffs; this exercises the caller-converted
    /// half the submit path takes, where the review found the gap.
    #[test]
    fn compact_records_tracks_moved_instances() {
        let Some((device, queue)) = try_device() else {
            return;
        };
        let (mut instances, view_proj, _) = cull_test_scene();
        let culler = Culler::try_new(&device, instances.len() as u32).expect("capacity above zero");

        fn stage_records(
            queue: &wgpu::Queue,
            culler: &Culler,
            view_proj: &Mat4,
            instances: &[InstanceData],
        ) {
            let mut converted = Vec::with_capacity(instances.len());
            super::convert_instances(&mut converted, instances);
            culler
                .compact_records(queue, view_proj, TEST_INDEX_COUNT, &converted)
                .expect("records fit the capacity");
        }

        fn dispatch(queue: &wgpu::Queue, device: &wgpu::Device, culler: &Culler) {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compact records moved dispatch"),
            });
            culler.record_compact(&mut encoder);
            queue.submit(std::iter::once(encoder.finish()));
        }

        stage_records(&queue, &culler, &view_proj, &instances);
        dispatch(&queue, &device, &culler);

        // Index 0, an index above zero and an adjacent pair, mirroring
        // the coverage [`gpu_compaction_tracks_moved_instances`] gives
        // [`Culler::compact`].
        let offscreen = Mat4::from_translation(glam::Vec3::new(100.0, 0.0, 0.0));
        instances[0].model_matrix = offscreen;
        instances[3].model_matrix = offscreen;
        instances[4].model_matrix = offscreen;
        let expected = compact_visible_indices(&instances, &view_proj)
            .iter()
            .map(|index| instances[*index as usize].model_matrix.w_axis.to_array())
            .collect::<Vec<[f32; 4]>>();

        stage_records(&queue, &culler, &view_proj, &instances);
        dispatch(&queue, &device, &culler);
        let args = cull_test_read_args(&device, &queue, &culler.args_buffer)
            .expect("args readback succeeds");
        assert_eq!(args.instance_count, expected.len() as u32);
        assert_visible_records_match(
            &device,
            &queue,
            &culler.visible_buffer,
            instances.len(),
            &expected,
        );
    }
}
