//! GPU frustum culling with `draw_indexed_indirect` submits ([RAT-35](/RAT/issues/RAT-35)).
//!
//! CPU frustum culling is opt-in and off by default
//! ([`crate::extraction::cull_frame_upload`]); this module stages its GPU
//! replacement without relying on `multi_draw_indirect` (emulated by a loop
//! on Metal, wgpu #2148, no `MDI_COUNT`): one compute pass compacts the
//! visible instances into a dense buffer and writes one
//! [`DrawIndexedIndirectArgs`], then gbuffer and shadow passes submit a
//! single [`draw_indexed_indirect`](wgpu::RenderPass::draw_indexed_indirect)
//! per mesh/material.
//!
//! Slice 1 (this module): the CPU reference compaction, the indirect-args
//! layout, the compute-shader source and its uniform packing, plus parity
//! tests against the CPU culler. Pass wiring (gbuffer/shadows indirect
//! submits, owned culler buffers on [`crate::renderer::Renderer3D`]) and
//! the 10k/100k benchmark table land in follow-up slices.
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

/// Threads per compute workgroup in [`CULL_COMPUTE_WGSL`].
pub const CULL_WORKGROUP_SIZE: u32 = 64;

/// Frustum planes per frame (matches [`Frustum`]).
pub const FRUSTUM_PLANE_COUNT: usize = 6;

/// `u32` words per [`DrawIndexedIndirectArgs`].
pub const INDIRECT_ARGS_WORD_COUNT: usize = 5;

/// Word index of `instance_count` inside the indirect-args buffer: the
/// only word the compute pass writes (via `atomicAdd`); the CPU
/// pre-writes the rest and zeroes this word before dispatch.
pub const INSTANCE_COUNT_WORD: usize = 1;

/// Byte size of the [`CullParams`] uniform block: six `vec4` planes plus
/// the instance count padded to a 16-byte stride.
pub const CULL_PARAMS_BYTE_SIZE: u64 = 112;

/// Pad words after [`CullParams::count`]: the count is one `u32` inside a
/// 16-byte uniform stride.
const CULL_PARAM_PAD_WORDS: usize = 3;

/// `DrawIndexedIndirectArgs` for one
/// [`draw_indexed_indirect`](wgpu::RenderPass::draw_indexed_indirect)
/// submit: five `u32` words (`index_count`, `instance_count`,
/// `first_index`, `base_vertex`, `first_instance`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectArgs {
    /// Indices per draw (mesh `num_indices`, written by the CPU).
    pub index_count: u32,
    /// Visible instances (CPU count, or the compute-pass atomic).
    pub instance_count: u32,
    /// First index into the index buffer (always `0` here).
    pub first_index: u32,
    /// Vertex-buffer base offset (always `0` here).
    pub base_vertex: u32,
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
pub fn cull_params(view_proj: &Mat4, count: u32) -> CullParams {
    let planes: [Vec4; FRUSTUM_PLANE_COUNT] = Frustum::from_view_proj(view_proj).planes();
    CullParams {
        planes: planes.map(|plane| plane.to_array()),
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
/// the [`crate::renderer::PerObjectGpu`] layout (two `mat4x4` + `u32`
/// material index + pad = 144 bytes) so the existing vertex stages index
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
    material_index: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
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
    if (i >= params.count) {
        return;
    }
    let sphere = instance_sphere(src[i].model);
    if (!sphere_visible(sphere.xyz, sphere.w)) {
        return;
    }
    let slot = atomicAdd(&args[1u], 1u);
    dst[slot] = src[i];
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::CameraView;
    use crate::extraction::cull_frame_upload;
    use crate::extraction::extract_render_data;
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

    /// `PerObjectGpu`-layout staging record for tests: the renderer field
    /// is private to its module, so the test mirrors the 144-byte layout
    /// (two `mat4x4` + material `u32` + pad) instead of naming it.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestPerObject {
        model: [[f32; 4]; 4],
        normal_matrix: [[f32; 4]; 4],
        material_index: u32,
        _pad: [u32; 3],
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
        // Five `u32` words, 20 bytes, field order = wgpu's
        // `DrawIndexedIndirectArgs` (`index_count`, `instance_count`,
        // `first_index`, `base_vertex`, `first_instance`).
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

    /// Adapter handle, or `None` on headless CI without a GPU (skip).
    fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::empty(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok()?;
            adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .ok()
        })
    }

    /// Staged source records plus the frame matrix and CPU reference
    /// count for the dispatch test.
    fn cull_test_scene() -> (Vec<TestPerObject>, Mat4, u32) {
        let upload = hundred_sphere_upload();
        let view = perspective_view();
        let (view_proj, _) = crate::camera::camera_view_projection(&view, (160, 90));
        let expected = compact_visible_indices(&upload.instances, &view_proj).len() as u32;
        let staged = upload
            .instances
            .iter()
            .map(|instance| TestPerObject {
                model: instance.model_matrix.to_cols_array_2d(),
                normal_matrix: instance.normal_matrix.to_cols_array_2d(),
                material_index: instance.material_index.as_u32(),
                _pad: [0; CULL_PARAM_PAD_WORDS],
            })
            .collect::<Vec<_>>();
        (staged, view_proj, expected)
    }

    /// Source, destination, uniform and zeroed-args buffers for the
    /// dispatch test.
    fn cull_test_buffers(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        staged: &[TestPerObject],
        params: &CullParams,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let bytes = std::mem::size_of_val(staged) as u64;
        let src = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull test src"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&src, 0, bytemuck::cast_slice(staged));
        let dst = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull test dst"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let param_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull test params"),
            size: CULL_PARAMS_BYTE_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&param_buffer, 0, bytemuck::bytes_of(params));
        let args_init = DrawIndexedIndirectArgs::compacted(0, 0);
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull test args"),
            size: (INDIRECT_ARGS_WORD_COUNT * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&args, 0, bytemuck::bytes_of(&args_init));
        (src, dst, param_buffer, args)
    }

    /// One buffer binding of the culling pass.
    fn cull_test_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    /// Compute pipeline plus its bind-group layout for the dispatch test.
    fn cull_test_pipeline(device: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cull test shader"),
            source: wgpu::ShaderSource::Wgsl(CULL_COMPUTE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cull test bgl"),
            entries: &[
                cull_test_entry(0, wgpu::BufferBindingType::Uniform),
                cull_test_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                cull_test_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                cull_test_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cull test pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cull test pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cull_instances"),
            compilation_options: Default::default(),
            cache: None,
        });
        (pipeline, layout)
    }

    /// One dispatch plus the `instance_count` word read back from the
    /// indirect-args buffer (`None` when the headless readback fails —
    /// the caller skips instead of failing).
    fn cull_test_dispatch(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: (&wgpu::ComputePipeline, &wgpu::BindGroupLayout),
        buffers: (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer),
        instance_count: u32,
    ) -> Option<u32> {
        let (pipeline, layout) = pass;
        let (src, dst, param_buffer, args) = buffers;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cull test bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: args.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cull test encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cull test pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(cull_workgroup_count(instance_count), 1, 1);
        }
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull test readback"),
            size: (INDIRECT_ARGS_WORD_COUNT * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(args, 0, &readback, 0, (INDIRECT_ARGS_WORD_COUNT * 4) as u64);
        queue.submit(std::iter::once(encoder.finish()));
        let slice = readback.slice(..);
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
        let words: [u32; INDIRECT_ARGS_WORD_COUNT] = bytemuck::cast_slice(&view)
            [..INDIRECT_ARGS_WORD_COUNT]
            .try_into()
            .expect("readback holds five words");
        drop(view);
        readback.unmap();
        Some(words[INSTANCE_COUNT_WORD])
    }

    #[test]
    fn gpu_compaction_count_matches_cpu() {
        // End-to-end dispatch on a real adapter: the atomic counter must
        // equal the CPU reference count (order is intentionally
        // unchecked — atomics are nondeterministic, see module docs).
        let Some((device, queue)) = try_device() else {
            return;
        };
        let per_object_size = std::mem::size_of::<TestPerObject>() as u64;
        assert_eq!(per_object_size, 144);
        assert_eq!(std::mem::offset_of!(TestPerObject, material_index), 128);
        let (staged, view_proj, expected) = cull_test_scene();
        let instance_count = staged.len();
        let params = cull_params(&view_proj, instance_count as u32);
        let buffers = cull_test_buffers(&device, &queue, &staged, &params);
        let (pipeline, layout) = cull_test_pipeline(&device);
        let Some(counter) = cull_test_dispatch(
            &device,
            &queue,
            (&pipeline, &layout),
            (&buffers.0, &buffers.1, &buffers.2, &buffers.3),
            instance_count as u32,
        ) else {
            return;
        };
        assert_eq!(counter, expected);
    }
}
