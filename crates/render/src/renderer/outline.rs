//! Editor outline pipeline: inverted-hull silhouettes for selected/hovered
//! instances.
//!
//! The pipeline is created lazily on the first frame that supplies outline
//! flags and lives next to the renderer's camera/per-object buffers, so the
//! typed frame pass ([`crate::frame_passes::OutlinePass`]) stays free of
//! GPU handles — required because `wgpu` resources are `!Send` on wasm and
//! [`crate::system::FramePass`] is `Send`.

use super::*;

/// Outline pipeline plus the per-instance flag buffer.
pub(crate) struct OutlineGpu {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    flags: wgpu::Buffer,
    /// Capacity of `flags` in elements.
    capacity: usize,
}

impl OutlineGpu {
    /// Builds the pipeline, bind-group layout and the initial flag buffer.
    fn create(device: &wgpu::Device, format: wgpu::TextureFormat, sample_count: u32) -> Self {
        let entries: Vec<wgpu::BindGroupLayoutEntry> =
            shaders::outline_generated::OUTLINE_RESOURCES
                .iter()
                .map(|resource| shaders::bgl_entry(resource, false))
                .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("outline bind group layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("outline pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outline vertex"),
            source: wgpu::ShaderSource::Wgsl(shaders::outline_vertex().into()),
        });
        let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outline fragment"),
            source: wgpu::ShaderSource::Wgsl(shaders::outline_fragment().into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("outline pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex,
                entry_point: Some(shaders::outline_generated::vs_main::entry_point()),
                buffers: &[Some(Vertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment,
                entry_point: Some(shaders::outline_generated::fs_main::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Inverted hull: keep the back faces of the expanded shell.
                cull_mode: Some(wgpu::Face::Front),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                // Equal keeps the ring where the object's own depth is the
                // same surface; faces further back fail.
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        let flags = Self::flags_buffer(device, 1);
        Self {
            pipeline,
            layout,
            flags,
            capacity: 1,
        }
    }

    /// Storage buffer for `capacity` instance flags.
    fn flags_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outline flags"),
            size: (capacity.max(1) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

/// Inputs of one [`Renderer3D::render_outline`] call.
pub(crate) struct OutlineInputs<'a> {
    /// Presented layer to overlay (`ldr_layer` before AA, `target` without).
    pub output: &'a wgpu::TextureView,
    /// G-buffer depth the hull is tested against.
    pub depth: &'a wgpu::TextureView,
    /// Shared mesh batch drawn for every instance.
    pub mesh: &'a Mesh,
    /// Number of uploaded instances.
    pub instance_count: u32,
    /// Format of `output`.
    pub format: wgpu::TextureFormat,
    /// Sample count of `output` and `depth`.
    pub sample_count: u32,
    /// One flag per instance (`0` none, `1` hover, `2` selected).
    pub flags: &'a [u32],
}

impl Renderer3D {
    /// Records the editor outline pass over `inputs.output`.
    ///
    /// No-op for an empty flag set (including the instance-less case), so
    /// plans wired with the pass but no selection stay pixel-identical.
    /// Resources are built on the first non-empty frame and the flag buffer
    /// grows with the scene.
    pub(crate) fn render_outline(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        inputs: OutlineInputs<'_>,
    ) {
        let OutlineInputs {
            output,
            depth,
            mesh,
            instance_count,
            format,
            sample_count,
            flags,
        } = inputs;
        if flags.is_empty() || instance_count == 0 {
            return;
        }
        let mut outline = self
            .outline
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if outline.is_none() {
            *outline = Some(OutlineGpu::create(device, format, sample_count));
        }
        let Some(gpu) = outline.as_mut() else {
            return;
        };
        if flags.len() > gpu.capacity {
            gpu.capacity = flags.len().next_power_of_two();
            gpu.flags = OutlineGpu::flags_buffer(device, gpu.capacity);
        }
        queue.write_buffer(&gpu.flags, 0, bytemuck::cast_slice(flags));
        let per_objects = read_lock(&self.per_object_buffer);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outline bind group"),
            layout: &gpu.layout,
            entries: &shaders::bind_group_entries(
                &shaders::outline_generated::OUTLINE_RESOURCES,
                |resource| match resource.name {
                    "camera" => Some(self.camera_buffer.as_entire_binding()),
                    "per_objects" => Some(per_objects.as_entire_binding()),
                    "outline_flags" => Some(gpu.flags.as_entire_binding()),
                    _ => None,
                },
            )
            .unwrap_or_default(),
        });
        drop(per_objects);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("outline pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.num_indices, 0, 0..instance_count);
    }
}
