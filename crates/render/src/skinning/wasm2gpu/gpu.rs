//! Persistent guest pipeline and storage buffers for one mesh shape.
//!
//! Buffers are allocated once per (vertex count, joint count) pair and
//! rewritten with `queue.write_buffer` per dispatch — the steady-state
//! frame shape issue #224 asks for: no per-frame buffer creation, only
//! the palette and pose inputs change.

use bytemuck::cast_slice;
use ornis_wasm2gpu_testkit::gpu_buffer;
#[cfg(test)]
use ornis_wasm2gpu_testkit::map_bytes;

use super::SkinInputs;

/// Storage binding of the joint palette lanes.
const PALETTE_BINDING: u32 = 0;
/// Storage binding of the bind-pose vertex lanes.
const BIND_BINDING: u32 = 1;
/// Storage binding of the joint-index lanes.
const JOINTS_BINDING: u32 = 2;
/// Storage binding of the weight lanes.
const WEIGHTS_BINDING: u32 = 3;
/// Storage binding of the skinned output lanes.
const OUT_BINDING: u32 = 4;
/// Storage binding of the vertex/joint count words.
const PARAMS_BINDING: u32 = 5;
/// Workgroup width of the guest entry (its `#[shader::compute]` record).
const SKIN_WORKGROUP: u32 = 64;
/// Lanes per packed palette joint / vertex (mirrors the guest ABI).
const JOINT_LANES: usize = 32;
/// Lanes per packed vertex record.
const VERTEX_LANES: usize = 6;
/// Influence lanes per vertex.
const MAX_INFLUENCES: usize = 4;
/// Count words in the params binding (vertex count, joint count, padding).
const PARAM_WORDS: usize = 4;
/// First normal lane inside one output vertex record.
const NORMAL_LANE: usize = 3;

/// Persistent compute pipeline and buffers of the guest skinning pass.
pub(crate) struct SkinGpu {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    _layout: wgpu::BindGroupLayout,
    palette: wgpu::Buffer,
    bind: wgpu::Buffer,
    joints: wgpu::Buffer,
    weights: wgpu::Buffer,
    out: wgpu::Buffer,
    params: wgpu::Buffer,
    #[cfg(test)]
    staging: wgpu::Buffer,
    vertex_count: u32,
    joint_count: u32,
}

impl SkinGpu {
    /// Builds the pipeline and exact-sized buffers for one mesh shape.
    pub(crate) fn new(device: &wgpu::Device, wgsl: &str, vertex_count: u32, joint_count: u32) -> Self {
        assert!(vertex_count > 0, "guest skin needs a vertex");
        assert!(joint_count > 0, "guest skin needs a joint");
        let storage = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wasm2gpu skinning layout"),
            entries: &[
                binding(PALETTE_BINDING, storage),
                binding(BIND_BINDING, storage),
                binding(JOINTS_BINDING, storage),
                binding(WEIGHTS_BINDING, storage),
                binding(OUT_BINDING, storage),
                binding(PARAMS_BINDING, storage),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wasm2gpu skin"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wasm2gpu skin"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wasm2gpu skin"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let vertices = vertex_count as usize;
        let palette = storage_buffer(device, "skin palette", joint_count as usize * JOINT_LANES);
        let bind = storage_buffer(device, "skin bind vertices", vertices * VERTEX_LANES);
        let joints = storage_buffer(device, "skin joints", vertices * MAX_INFLUENCES);
        let weights = storage_buffer(device, "skin weights", vertices * MAX_INFLUENCES);
        let out = storage_buffer(device, "skin output", vertices * VERTEX_LANES);
        let params = storage_buffer(device, "skin params", PARAM_WORDS);
        #[cfg(test)]
        let staging = gpu_buffer(
            device,
            "skin readback",
            (vertices * VERTEX_LANES * std::mem::size_of::<f32>()) as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wasm2gpu skinning bind group"),
            layout: &layout,
            entries: &[
                bind_entry(PALETTE_BINDING, &palette),
                bind_entry(BIND_BINDING, &bind),
                bind_entry(JOINTS_BINDING, &joints),
                bind_entry(WEIGHTS_BINDING, &weights),
                bind_entry(OUT_BINDING, &out),
                bind_entry(PARAMS_BINDING, &params),
            ],
        });
        Self {
            pipeline,
            bind_group,
            _layout: layout,
            palette,
            bind,
            joints,
            weights,
            out,
            params,
            #[cfg(test)]
            staging,
            vertex_count,
            joint_count,
        }
    }

    /// Uploads one packed mesh into the persistent buffers.
    ///
    /// The inputs must match the shape this instance was built for (the
    /// harness asserts it, like the auto-exposure guest harness).
    pub(crate) fn prepare(&self, queue: &wgpu::Queue, inputs: &SkinInputs) {
        assert_eq!(inputs.vertex_count, self.vertex_count);
        assert_eq!(inputs.joint_count, self.joint_count);
        queue.write_buffer(&self.palette, 0, cast_slice(&inputs.palette));
        queue.write_buffer(&self.bind, 0, cast_slice(&inputs.bind));
        queue.write_buffer(&self.joints, 0, cast_slice(&inputs.joints));
        queue.write_buffer(&self.weights, 0, cast_slice(&inputs.weights));
        let params = [self.vertex_count, self.joint_count, 0, 0];
        queue.write_buffer(&self.params, 0, cast_slice(&params));
    }

    /// Records the skinning dispatch (one lane per vertex).
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wasm2gpu skin"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.vertex_count.div_ceil(SKIN_WORKGROUP), 1, 1);
    }

    /// Runs one dispatch and reads the skinned positions and normals back.
    #[cfg(test)]
    pub(crate) fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &SkinInputs,
    ) -> Option<(Vec<[f32; 3]>, Vec<[f32; 3]>)> {
        self.prepare(queue, inputs);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wasm2gpu skinning encoder"),
        });
        self.encode(&mut encoder);
        let bytes = (self.vertex_count as usize * VERTEX_LANES * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&self.out, 0, &self.staging, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));
        let raw = map_bytes(device, &self.staging);
        let lanes: &[f32] = bytemuck::try_cast_slice(&raw).ok()?;
        if lanes.len() != self.vertex_count as usize * VERTEX_LANES {
            return None;
        }
        let mut positions = Vec::with_capacity(self.vertex_count as usize);
        let mut normals = Vec::with_capacity(self.vertex_count as usize);
        for vertex in lanes.chunks_exact(VERTEX_LANES) {
            positions.push([vertex[0], vertex[1], vertex[2]]);
            normals.push([
                vertex[NORMAL_LANE],
                vertex[NORMAL_LANE + 1],
                vertex[NORMAL_LANE + 2],
            ]);
        }
        Some((positions, normals))
    }
}

fn binding(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
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

fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn storage_buffer(device: &wgpu::Device, label: &str, lanes: usize) -> wgpu::Buffer {
    gpu_buffer(
        device,
        label,
        (lanes * std::mem::size_of::<u32>()) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    )
}
