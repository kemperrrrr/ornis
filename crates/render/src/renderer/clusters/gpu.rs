//! GPU owner of the clustered-light buffers and the assignment compute
//! pass (#130). The shader source lives in [`super::wgsl`], the CPU
//! reference in [`super::cpu`].
use super::wgsl::{
    BINDING_CAMERA, BINDING_COUNTERS, BINDING_GRID, BINDING_INDICES, BINDING_LIGHTS, BINDING_PARAMS,
};

use super::{
    CLUSTER_COUNT, CLUSTER_INDEX_LIST_LEN, CLUSTER_WORKGROUPS, ClusterParams, ClusterSpan,
    MAX_CLUSTERED_LIGHTS,
};
use crate::renderer::GpuLight;
use std::borrow::Cow;

/// Which cluster buffer a test readback pulls (#130 parity gate).
// qual:api — test-visible selector; in-tree callers are the parity gates.
#[cfg(test)]
pub(crate) enum ClusterBuffer {
    /// The per-cluster `array<ClusterSpan>` grid.
    Grid,
    /// The shared `array<u32>` index list.
    Indices,
    /// The `[span cursor, overflow]` counters.
    Counters,
}

/// The clustered-light GPU state: storage buffers, the assignment
/// pipeline and its bind group. Created once in
/// [`Renderer3D::new`](super::Renderer3D::new); every lit pass binds
/// the same buffers, the assignment pass rewrites the grid and index
/// list in place each frame.
pub(crate) struct ClusteredLights {
    params_buffer: wgpu::Buffer,
    lights_buffer: wgpu::Buffer,
    grid_buffer: wgpu::Buffer,
    indices_buffer: wgpu::Buffer,
    counters_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

/// One storage/uniform buffer of the assignment pass (test-visible
/// labels keep the debuggers useful).
fn cluster_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

/// One uniform-binding layout entry of the assignment pass.
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// One storage-binding layout entry of the assignment pass.
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl ClusteredLights {
    /// Create the buffers, the assignment pipeline and its bind group.
    /// `camera` is the shared camera buffer: the assignment reads the
    /// current `view_proj` from it, so it never depends on
    /// `set_lights`/`set_camera` call order.
    pub(crate) fn new(device: &wgpu::Device, camera: &wgpu::Buffer) -> Self {
        let [
            params_buffer,
            lights_buffer,
            grid_buffer,
            indices_buffer,
            counters_buffer,
        ] = Self::assignment_buffers(device);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cluster assignment shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(super::wgsl::cluster_compute_wgsl())),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cluster assignment bind group layout"),
            entries: &[
                uniform_entry(BINDING_CAMERA),
                uniform_entry(BINDING_PARAMS),
                storage_entry(BINDING_LIGHTS, true),
                storage_entry(BINDING_GRID, false),
                storage_entry(BINDING_INDICES, false),
                storage_entry(BINDING_COUNTERS, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cluster assignment pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cluster assignment pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("assign_lights_to_clusters"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = Self::assignment_bind_group(
            device,
            &bind_group_layout,
            camera,
            [
                &params_buffer,
                &lights_buffer,
                &grid_buffer,
                &indices_buffer,
                &counters_buffer,
            ],
        );
        Self {
            params_buffer,
            lights_buffer,
            grid_buffer,
            indices_buffer,
            counters_buffer,
            pipeline,
            bind_group,
        }
    }

    /// The five assignment-pass buffers, in binding order
    /// `[params, lights, grid, indices, counters]`.
    fn assignment_buffers(device: &wgpu::Device) -> [wgpu::Buffer; 5] {
        [
            cluster_buffer(
                device,
                "cluster params",
                std::mem::size_of::<ClusterParams>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            cluster_buffer(
                device,
                "cluster light storage",
                MAX_CLUSTERED_LIGHTS as u64 * std::mem::size_of::<GpuLight>() as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            cluster_buffer(
                device,
                "cluster grid",
                CLUSTER_COUNT as u64 * std::mem::size_of::<ClusterSpan>() as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            cluster_buffer(
                device,
                "cluster light index list",
                CLUSTER_INDEX_LIST_LEN as u64 * std::mem::size_of::<u32>() as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            cluster_buffer(
                device,
                "cluster counters",
                std::mem::size_of::<[u32; 2]>() as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            ),
        ]
    }

    /// The assignment-pass bind group: the shared camera buffer plus
    /// the five cluster buffers, in binding order.
    fn assignment_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        camera: &wgpu::Buffer,
        buffers: [&wgpu::Buffer; 5],
    ) -> wgpu::BindGroup {
        let [params, lights, grid, indices, counters] = buffers;
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cluster assignment bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: BINDING_CAMERA,
                    resource: camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BINDING_PARAMS,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BINDING_LIGHTS,
                    resource: lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BINDING_GRID,
                    resource: grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BINDING_INDICES,
                    resource: indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BINDING_COUNTERS,
                    resource: counters.as_entire_binding(),
                },
            ],
        })
    }

    /// Resolve a shading-table resource name to its buffer binding;
    /// `None` for non-cluster names (the shared table resolver falls
    /// through to it).
    pub(crate) fn entry(&self, name: &str) -> Option<wgpu::BindingResource<'_>> {
        match name {
            "cluster_params" => Some(self.params_buffer.as_entire_binding()),
            "clustered_lights" => Some(self.lights_buffer.as_entire_binding()),
            "cluster_light_indices" => Some(self.indices_buffer.as_entire_binding()),
            "cluster_grid" => Some(self.grid_buffer.as_entire_binding()),
            _ => None,
        }
    }

    /// Upload the assignment params and the light records: `head` (the
    /// uniform-window records, byte-identical to `lighting.lights`) at
    /// storage offset 0, the clustered point/spot records after it.
    pub(crate) fn write_frame(
        &self,
        queue: &wgpu::Queue,
        params: &ClusterParams,
        head: &[GpuLight],
        clustered: &[GpuLight],
    ) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
        queue.write_buffer(&self.lights_buffer, 0, bytemuck::cast_slice(head));
        if !clustered.is_empty() {
            let offset = std::mem::size_of_val(head) as u64;
            queue.write_buffer(&self.lights_buffer, offset, bytemuck::cast_slice(clustered));
        }
    }

    /// Record the assignment pass on `encoder`: reset the span cursor
    /// and overflow counter, then one thread per cluster rebuilds its
    /// list in place (grid entries are written only by their owning
    /// thread, index-list spans only after their atomic claim).
    pub(crate) fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.counters_buffer, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cluster light assignment"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(CLUSTER_WORKGROUPS, 1, 1);
    }

    /// Read `bytes` of one cluster buffer back (`None` when the map
    /// fails). Test-only: the parity gate compares the assignment
    /// output against the CPU reference.
    #[cfg(test)]
    pub(crate) fn read_back(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        which: ClusterBuffer,
        bytes: u64,
    ) -> Option<Vec<u8>> {
        let src = match which {
            ClusterBuffer::Grid => &self.grid_buffer,
            ClusterBuffer::Indices => &self.indices_buffer,
            ClusterBuffer::Counters => &self.counters_buffer,
        };
        // wgpu maps need a non-empty range: floor the readback at one
        // u32 word.
        let size = bytes.max(std::mem::size_of::<u32>() as u64);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster test readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cluster test readback encoder"),
        });
        encoder.copy_buffer_to_buffer(src, 0, &readback, 0, bytes);
        queue.submit([encoder.finish()]);
        pollster::block_on(async {
            let slice = readback.slice(0..size);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
            let view = slice.get_mapped_range().ok()?;
            let out = view[..bytes as usize].to_vec();
            drop(view);
            readback.unmap();
            Some(out)
        })
    }
}
