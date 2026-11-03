//! Editor outline pipeline: inverted-hull silhouettes for selected/hovered
//! instances.
//!
//! The pipeline is created lazily on the first frame that supplies outline
//! flags and lives next to the renderer's camera/per-object buffers, so the
//! typed frame pass ([`crate::frame_passes::OutlinePass`]) stays free of
//! GPU handles — required because `wgpu` resources are `!Send` on wasm and
//! [`crate::system::FramePass`] is `Send`.

use super::{Renderer3D, read_lock};
use crate::mesh::{Mesh, Vertex};
use crate::shaders;

/// Outline pipeline plus the per-instance flag buffer.
pub(crate) struct OutlineGpu {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    flags: wgpu::Buffer,
    /// Capacity of `flags` in elements.
    capacity: usize,
}

impl OutlineGpu {
    /// Builds the bind-group layout, pipeline and the initial flag buffer.
    fn create(device: &wgpu::Device, format: wgpu::TextureFormat, sample_count: u32) -> Self {
        let layout = Self::layout(device);
        let pipeline = Self::pipeline(device, &layout, format, sample_count);
        let flags = Self::flags_buffer(device, 1);
        Self {
            pipeline,
            layout,
            flags,
            capacity: 1,
        }
    }

    /// Bind-group layout from the outline resource table.
    fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let entries: Vec<wgpu::BindGroupLayoutEntry> =
            shaders::outline_generated::OUTLINE_RESOURCES
                .iter()
                .map(|resource| shaders::bgl_entry(resource, false))
                .collect();
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("outline bind group layout"),
            entries: &entries,
        })
    }

    /// Inverted-hull pipeline: expanded back faces, no depth writes.
    fn pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("outline pipeline layout"),
            bind_group_layouts: &[Some(bind_group_layout)],
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
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::test_util::{offscreen_target, readback_rgba8, try_device};
    use crate::renderer::{InstanceData, MaterialIdx};
    use glam::Mat4;

    /// Depth attachment matching the outline pipeline (Depth32Float, 1x).
    fn depth_target(device: &wgpu::Device, size: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("outline depth"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// A flagged instance draws silhouette pixels; an empty flag set records
    /// nothing (no panic, no GPU work). Skips without an adapter.
    #[test]
    fn outline_pass_draws_flagged_instances_and_ignores_empty_flags() {
        let Some((device, queue)) = try_device() else {
            eprintln!("SKIPPED (outline pass): no wgpu adapter");
            return;
        };
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 8,
            height: 8,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        let renderer = Renderer3D::new(&device, &surface_config, 1);
        renderer.upload_instances(
            &device,
            &queue,
            &[InstanceData {
                model_matrix: Mat4::IDENTITY,
                normal_matrix: Mat4::IDENTITY,
                material_index: MaterialIdx::from_raw(0),
            }],
        );
        let (color, color_view) = offscreen_target(&device, 8, 8);
        let (_depth, depth_view) = depth_target(&device, 8);
        let mesh = crate::create_sphere(&device, 1.0, 8, 6);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("outline test clear"),
        });
        {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("outline test clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        renderer.render_outline(
            &device,
            &queue,
            &mut encoder,
            OutlineInputs {
                output: &color_view,
                depth: &depth_view,
                mesh: &mesh,
                instance_count: 1,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                sample_count: 1,
                flags: &[2],
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        let pixels = readback_rgba8(&device, &queue, &color, 8, 8);
        assert!(
            pixels.iter().any(|&byte| byte != 0),
            "a selected instance must draw outline pixels"
        );

        // Empty flags: the pass returns before touching the GPU.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("outline test empty"),
        });
        renderer.render_outline(
            &device,
            &queue,
            &mut encoder,
            OutlineInputs {
                output: &color_view,
                depth: &depth_view,
                mesh: &mesh,
                instance_count: 1,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                sample_count: 1,
                flags: &[],
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
    }
}
