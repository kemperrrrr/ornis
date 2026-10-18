//! GPU owner of the clustered-light buffers and the assignment compute
//! pass (#130). The shader source lives in [`super::wgsl`], the CPU
//! reference in [`super::cpu`].
use super::*;
use super::wgsl::{
    BINDING_CAMERA, BINDING_COUNTERS, BINDING_GRID, BINDING_INDICES, BINDING_LIGHTS, BINDING_PARAMS,
};

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

impl ClusteredLights {
    /// Create the buffers, the assignment pipeline and its bind group.
    /// `camera` is the shared camera buffer: the assignment reads the
    /// current `view_proj` from it, so it never depends on
    /// `set_lights`/`set_camera` call order.
    pub(crate) fn new(device: &wgpu::Device, camera: &wgpu::Buffer) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster params"),
            size: std::mem::size_of::<ClusterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster light storage"),
            size: MAX_CLUSTERED_LIGHTS as u64 * std::mem::size_of::<GpuLight>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster grid"),
            size: CLUSTER_COUNT as u64 * std::mem::size_of::<ClusterSpan>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster light index list"),
            size: CLUSTER_INDEX_LIST_LEN as u64 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let counters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster counters"),
            size: std::mem::size_of::<[u32; 2]>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cluster assignment shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(super::wgsl::cluster_compute_wgsl())),
        });
        let uniform_compute = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_compute = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cluster assignment bind group layout"),
            entries: &[
                uniform_compute(BINDING_CAMERA),
                uniform_compute(BINDING_PARAMS),
                storage_compute(BINDING_LIGHTS, true),
                storage_compute(BINDING_GRID, false),
                storage_compute(BINDING_INDICES, false),
                storage_compute(BINDING_COUNTERS, false),
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cluster assignment bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: BINDING_CAMERA,
                    resource: camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BINDING_PARAMS,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BINDING_LIGHTS,
                    resource: lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BINDING_GRID,
                    resource: grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BINDING_INDICES,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BINDING_COUNTERS,
                    resource: counters_buffer.as_entire_binding(),
                },
            ],
        });
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
            let offset = (head.len() * std::mem::size_of::<GpuLight>()) as u64;
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

    /// The grid buffer (assignment readback in tests).
    #[cfg(test)]
    pub(crate) fn grid_buffer(&self) -> &wgpu::Buffer {
        &self.grid_buffer
    }

    /// The index-list buffer (assignment readback in tests).
    #[cfg(test)]
    pub(crate) fn indices_buffer(&self) -> &wgpu::Buffer {
        &self.indices_buffer
    }

    /// The `[span cursor, overflow]` counter buffer (readback in tests).
    #[cfg(test)]
    pub(crate) fn counters_buffer(&self) -> &wgpu::Buffer {
        &self.counters_buffer
    }
}
