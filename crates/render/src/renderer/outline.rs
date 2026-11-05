//! Editor outline pipeline: inverted-hull silhouettes for selected/hovered
//! instances.
//!
//! The host supplies one flag per scene instance (`0` none, `1` hover,
//! `2` selected). The renderer compacts the flagged entries into an index
//! list and draws only those instances, so an idle viewport records no
//! outline commands. The pipeline is created lazily on the first outline
//! frame and lives next to the renderer's camera/per-object buffers, so the
//! composite pass ([`crate::frame_passes::Composite::with_outline`]) stays
//! free of GPU handles — required because `wgpu` resources are `!Send` on
//! wasm and [`crate::system::FramePass`] is `Send`.

use super::{Renderer3D, read_lock};
use crate::mesh::{Mesh, Vertex};
use crate::shaders;
use std::borrow::Cow;

/// Outline pipeline plus the compact per-frame index/flag buffers.
pub(crate) struct OutlineGpu {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// Attachment format the pipeline was built for.
    format: wgpu::TextureFormat,
    /// Sample count the pipeline was built for.
    sample_count: u32,
    /// Compact scene-instance indices (capacity in elements).
    indices: wgpu::Buffer,
    /// Flags parallel to `indices`.
    flags: wgpu::Buffer,
    /// Capacity of both buffers in elements.
    capacity: usize,
    /// Reused CPU staging for the compact lists (no per-frame allocation).
    indices_cpu: Vec<u32>,
    flags_cpu: Vec<u32>,
    /// Cached bind group and the buffer sizes it was built against.
    bind_group: Option<wgpu::BindGroup>,
    bind_sizes: (u64, u64),
}

impl OutlineGpu {
    /// Builds the bind-group layout, pipeline and the initial buffers.
    fn create(device: &wgpu::Device, format: wgpu::TextureFormat, sample_count: u32) -> Self {
        let layout = Self::layout(device);
        let pipeline = Self::pipeline(device, &layout, format, sample_count);
        Self {
            pipeline,
            layout,
            format,
            sample_count,
            indices: Self::storage_buffer(device, 1),
            flags: Self::storage_buffer(device, 1),
            capacity: 1,
            indices_cpu: Vec::new(),
            flags_cpu: Vec::new(),
            bind_group: None,
            bind_sizes: (0, 0),
        }
    }

    /// Whether this pipeline matches the current frame's attachment key.
    fn matches(&self, format: wgpu::TextureFormat, sample_count: u32) -> bool {
        self.format == format && self.sample_count == sample_count
    }

    /// Grows the index/flag buffers when the compact list outgrows them.
    // qual:allow(iosp) reason: "buffer growth keeps the two parallel lists and their cached bind group in lockstep"
    fn ensure_capacity(&mut self, device: &wgpu::Device, needed: usize) {
        if needed <= self.capacity {
            return;
        }
        self.capacity = needed.next_power_of_two();
        self.indices = Self::storage_buffer(device, self.capacity);
        self.flags = Self::storage_buffer(device, self.capacity);
        // The cached group referenced the old buffers.
        self.bind_group = None;
    }

    /// Bind group over the camera, per-object records and both compact lists.
    fn create_bind_group(
        &self,
        device: &wgpu::Device,
        camera: &wgpu::Buffer,
        per_objects: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("outline bind group"),
            layout: &self.layout,
            entries: &shaders::bind_group_entries(
                &shaders::outline_generated::OUTLINE_RESOURCES,
                |resource| match resource.name {
                    "camera" => Some(camera.as_entire_binding()),
                    "per_objects" => Some(per_objects.as_entire_binding()),
                    "outline_indices" => Some(self.indices.as_entire_binding()),
                    "outline_flags" => Some(self.flags.as_entire_binding()),
                    _ => None,
                },
            )
            .unwrap_or_default(),
        })
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
                // The hull's back faces lose to the object's own depth and
                // survive outside the silhouette (against the far clear).
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

    /// Storage buffer for `capacity` u32 entries.
    fn storage_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outline list"),
            size: (capacity.max(1) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

/// Flags adjusted to exactly `instance_count` entries: short input is
/// zero-padded (no stale reads past the written range) and extra input is
/// ignored. Returns the input unchanged when the lengths already match.
fn outline_flags_for(flags: &[u32], instance_count: u32) -> Cow<'_, [u32]> {
    let count = instance_count as usize;
    if flags.len() == count {
        return Cow::Borrowed(flags);
    }
    let mut padded = flags[..flags.len().min(count)].to_vec();
    padded.resize(count, 0);
    Cow::Owned(padded)
}

/// Inputs the composite pass hands to the outline overlay.
pub(crate) struct OutlineCompositeInputs<'a> {
    /// G-buffer depth the hull is tested against.
    pub depth: &'a wgpu::TextureView,
    /// Shared mesh batch drawn for every instance.
    pub mesh: &'a Mesh,
    /// Number of uploaded instances (flags are indexed by this order).
    pub instance_count: u32,
    /// Format of the composited layer.
    pub format: wgpu::TextureFormat,
    /// Sample count of the composited layer and depth.
    pub sample_count: u32,
    /// One flag per instance (`0` none, `1` hover, `2` selected).
    pub flags: &'a [u32],
}

/// A ready outline draw inside the caller's (composite) render pass.
pub(crate) struct PreparedOutline<'a> {
    pipeline: &'a wgpu::RenderPipeline,
    bind_group: &'a wgpu::BindGroup,
    instances: u32,
}

impl PreparedOutline<'_> {
    /// Binds and draws the compact outlined instances.
    pub(crate) fn record(&self, pass: &mut wgpu::RenderPass<'_>, mesh: &Mesh) {
        pass.set_pipeline(self.pipeline);
        pass.set_bind_group(0, self.bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.num_indices, 0, 0..self.instances);
    }
}

/// Depth attachment operations of the outline overlay.
///
/// `Store`, never `Discard`: the plan declares `Read<Depth>`, and later
/// passes (TAA) still read the buffer after the composite/outline pass.
pub(crate) fn outline_depth_ops() -> wgpu::Operations<f32> {
    wgpu::Operations {
        load: wgpu::LoadOp::Load,
        store: wgpu::StoreOp::Store,
    }
}

impl Renderer3D {
    /// Prepares the compact outline state and hands it to `f`, which records
    /// it into its own render pass (the composite pass). `f(None)` means no
    /// instance is flagged, so the caller records no outline commands.
    ///
    /// The pipeline and bind group are cached until the attachment key or a
    /// buffer changes; the compact index list is rebuilt per call (editor
    /// selection is small).
    // qual:allow(iosp) reason: "one frame submission: compact the flags, cache pipeline/bind state and hand the draw to the composite pass"
    // qual:allow(srp, max_parameters=7) reason: "one prepared frame: attachment key, flags and the drawing callback"
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_outline<R>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        flags: &[u32],
        instance_count: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
        f: impl FnOnce(Option<PreparedOutline<'_>>) -> R,
    ) -> R {
        if flags.is_empty() || instance_count == 0 {
            return f(None);
        }
        let effective = outline_flags_for(flags, instance_count);
        let mut outline = self.outline.lock().unwrap_or_else(|poisoned| {
            eprintln!("ornis-render: outline lock poisoned — continuing");
            poisoned.into_inner()
        });
        if outline
            .as_ref()
            .is_none_or(|gpu| !gpu.matches(format, sample_count))
        {
            *outline = Some(OutlineGpu::create(device, format, sample_count));
        }
        let Some(gpu) = outline.as_mut() else {
            return f(None);
        };
        gpu.indices_cpu.clear();
        gpu.flags_cpu.clear();
        for (index, &flag) in effective.iter().enumerate() {
            if flag != 0 {
                gpu.indices_cpu.push(index as u32);
                gpu.flags_cpu.push(flag);
            }
        }
        if gpu.indices_cpu.is_empty() {
            return f(None);
        }
        gpu.ensure_capacity(device, gpu.indices_cpu.len());
        queue.write_buffer(&gpu.indices, 0, bytemuck::cast_slice(&gpu.indices_cpu));
        queue.write_buffer(&gpu.flags, 0, bytemuck::cast_slice(&gpu.flags_cpu));
        let per_objects = read_lock(&self.per_object_buffer);
        let per_object_size = per_objects.size();
        let outline_size = gpu.indices.size();
        if gpu.bind_group.is_none() || gpu.bind_sizes != (per_object_size, outline_size) {
            let group = gpu.create_bind_group(device, &self.camera_buffer, &per_objects);
            gpu.bind_group = Some(group);
            gpu.bind_sizes = (per_object_size, outline_size);
        }
        drop(per_objects);
        let Some(bind_group) = gpu.bind_group.as_ref() else {
            return f(None);
        };
        f(Some(PreparedOutline {
            pipeline: &gpu.pipeline,
            bind_group,
            instances: gpu.indices_cpu.len() as u32,
        }))
    }
}

/// Standalone outline pass used by the device tests. Production draws the
/// overlay inside the composite pass.
#[cfg(test)]
impl Renderer3D {
    #[allow(clippy::too_many_arguments)]
    // qual:allow(srp, max_parameters=10) reason: "device-test helper mirrors the old standalone pass signature"
    pub(crate) fn render_outline_for_test(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        mesh: &Mesh,
        instance_count: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
        flags: &[u32],
    ) {
        self.with_outline(
            device,
            queue,
            flags,
            instance_count,
            format,
            sample_count,
            |prepared| {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("outline test pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth,
                        // Same ops as production: the test would catch a
                        // Discard regression here too.
                        depth_ops: Some(outline_depth_ops()),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if let Some(prepared) = prepared {
                    prepared.record(&mut pass, mesh);
                }
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::test_util::{offscreen_target, readback_rgba8, try_device};
    use crate::renderer::{InstanceData, MaterialIdx};
    use glam::{Mat4, Vec3};

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
    const CAMERA: [f32; 3] = [0.0, 0.0, 3.0];
    const FOV: ornis_core::units::Degrees = ornis_core::units::Degrees::new(60.0);
    const NEAR: ornis_core::units::Meters = ornis_core::units::Meters::new(0.1);
    const FAR: ornis_core::units::Meters = ornis_core::units::Meters::new(100.0);

    fn surface_config_sized(size: u32) -> wgpu::SurfaceConfiguration {
        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: FORMAT,
            width: size,
            height: size,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        }
    }

    fn surface_config() -> wgpu::SurfaceConfiguration {
        surface_config_sized(8)
    }

    fn target_with_format(
        device: &wgpu::Device,
        size: u32,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("outline test target"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

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

    fn instances(count: u32) -> Vec<InstanceData> {
        (0..count)
            .map(|index| InstanceData {
                model_matrix: Mat4::from_translation(Vec3::new(index as f32, 0.0, 0.0)),
                normal_matrix: Mat4::IDENTITY,
                material_index: MaterialIdx::from_raw(0),
            })
            .collect()
    }

    fn clear_targets(
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) {
        let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("outline test clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Transparent, not opaque black: then a non-zero byte
                    // means the outline actually drew.
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
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

    /// Short flag slices are padded with zeros (no stale reads past the
    /// written range); extra entries are ignored.
    #[test]
    fn outline_flags_pad_and_truncate_to_instance_count() {
        assert!(matches!(
            outline_flags_for(&[1, 2], 2),
            std::borrow::Cow::Borrowed(_)
        ));
        assert_eq!(&*outline_flags_for(&[2], 2), &[2, 0]);
        assert_eq!(&*outline_flags_for(&[1, 2, 2], 1), &[1]);
        assert_eq!(&*outline_flags_for(&[], 2), &[0, 0]);
    }

    /// Flagged instances draw; all-zero/empty flags record no draw at all.
    /// Skips without an adapter.
    #[test]
    fn outline_pass_draws_selected_and_skips_zero_flags() {
        let Some((device, queue)) = try_device() else {
            eprintln!("SKIPPED (outline pass): no wgpu adapter");
            return;
        };
        let renderer = Renderer3D::new(&device, &surface_config(), 1);
        let view = crate::camera::CameraView {
            eye: Vec3::from_array(CAMERA),
            target: Vec3::ZERO,
            up: Vec3::Y,
            projection: ornis_assets::scene::CameraProjection::Perspective,
            fov: FOV,
            near: NEAR,
            far: FAR,
        };
        let (view_proj, eye) = crate::camera::camera_view_projection(&view, (8, 8));
        renderer.set_camera(&queue, &view_proj.to_cols_array_2d(), eye.to_array());
        renderer.upload_instances(&device, &queue, &instances(2));
        let (color, color_view) = offscreen_target(&device, 8, 8);
        assert_eq!(
            color.format(),
            FORMAT,
            "the test target must match the pipeline format"
        );
        let (_depth, depth_view) = depth_target(&device, 8);
        let mesh = crate::create_sphere(&device, 1.0, 8, 6);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("outline test selected"),
        });
        clear_targets(&mut encoder, &color_view, &depth_view);
        renderer.render_outline_for_test(
            &device,
            &queue,
            &mut encoder,
            &color_view,
            &depth_view,
            &mesh,
            2,
            FORMAT,
            1,
            // Two compact entries grow the buffers (capacity 1 → 2).
            &[1, 2],
        );
        queue.submit(std::iter::once(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let validation = pollster::block_on(scope.pop());
        assert!(
            validation.is_none(),
            "outline pass must not raise validation errors: {validation:?}"
        );
        let selected = readback_rgba8(&device, &queue, &color, 8, 8);
        let drawn = selected.iter().filter(|&&byte| byte != 0).count();
        assert!(
            drawn > 0,
            "a selected instance must draw outline pixels (got {drawn} non-zero bytes)"
        );
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        // Zero flags: the pass compacts to an empty list and records no draw.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("outline test zero flags"),
        });
        clear_targets(&mut encoder, &color_view, &depth_view);
        renderer.render_outline_for_test(
            &device,
            &queue,
            &mut encoder,
            &color_view,
            &depth_view,
            &mesh,
            2,
            FORMAT,
            1,
            &[0, 0],
        );
        queue.submit(std::iter::once(encoder.finish()));
        let empty = readback_rgba8(&device, &queue, &color, 8, 8);
        assert!(
            empty.iter().all(|&byte| byte == 0),
            "zero flags must leave the frame untouched"
        );

        // A format change must rebuild the cached pipeline, not reuse the
        // sRGB one against an Unorm attachment.
        let (unorm, unorm_view) = target_with_format(&device, 8, wgpu::TextureFormat::Rgba8Unorm);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("outline test format change"),
        });
        renderer.render_outline_for_test(
            &device,
            &queue,
            &mut encoder,
            &unorm_view,
            &depth_view,
            &mesh,
            2,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            &[2, 0],
        );
        queue.submit(std::iter::once(encoder.finish()));
        let changed = readback_rgba8(&device, &queue, &unorm, 8, 8);
        assert!(
            changed.iter().any(|&byte| byte != 0),
            "the format-change frame must still draw the outline"
        );

        let validation = pollster::block_on(scope.pop());
        assert!(
            validation.is_none(),
            "outline format change must not raise validation errors: {validation:?}"
        );
    }

    /// The depth attachment must survive the overlay: the plan declares
    /// `Read<Depth>` and TAA (when wired) reads it afterwards.
    #[test]
    fn outline_depth_is_stored_for_later_passes() {
        assert!(matches!(outline_depth_ops().store, wgpu::StoreOp::Store));
    }

    /// RGB of one texel in an RGBA8 readback.
    fn rgb_at(pixels: &[u8], size: u32, x: u32, y: u32) -> [u8; 3] {
        let index = ((y * size + x) * 4) as usize;
        [pixels[index], pixels[index + 1], pixels[index + 2]]
    }

    /// The ring exists only outside the object's silhouette: with the
    /// object's own depth present, the centre stays untouched while a texel
    /// between the object and the expanded hull is painted. Skips without
    /// an adapter.
    #[test]
    fn outline_draws_only_outside_the_object_silhouette() {
        let Some((device, queue)) = try_device() else {
            eprintln!("SKIPPED (outline silhouette): no wgpu adapter");
            return;
        };
        const SIZE: u32 = 128;
        const CENTER: u32 = SIZE / 2;
        // Camera at z=3 with a unit sphere: object radius ≈39.2 px, the 5%
        // hull ≈41.5 px, so x = center + 40 sits in the ring.
        const RING_OFFSET: u32 = 40;
        let renderer = Renderer3D::new(&device, &surface_config(), 1);
        let view = crate::camera::CameraView {
            eye: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            projection: ornis_assets::scene::CameraProjection::Perspective,
            fov: FOV,
            near: NEAR,
            far: FAR,
        };
        let (view_proj, eye) = crate::camera::camera_view_projection(&view, (SIZE, SIZE));
        renderer.set_camera(&queue, &view_proj.to_cols_array_2d(), eye.to_array());
        renderer.upload_instances(
            &device,
            &queue,
            &[InstanceData {
                model_matrix: Mat4::IDENTITY,
                normal_matrix: Mat4::IDENTITY,
                material_index: MaterialIdx::from_raw(0),
            }],
        );
        let mesh = crate::create_sphere(&device, 1.0, 16, 12);
        let (_albedo, albedo_view) =
            target_with_format(&device, SIZE, wgpu::TextureFormat::Rgba8Unorm);
        let (_normal, normal_view) =
            target_with_format(&device, SIZE, wgpu::TextureFormat::Rg16Float);
        let (_material_id, material_id_view) =
            target_with_format(&device, SIZE, wgpu::TextureFormat::R16Uint);
        let (_world_pos, world_pos_view) =
            target_with_format(&device, SIZE, wgpu::TextureFormat::Rg16Float);
        let (_material_params, material_params_view) =
            target_with_format(&device, SIZE, wgpu::TextureFormat::Rgba16Float);
        let (_depth, depth_view) = depth_target(&device, SIZE);
        let targets = crate::renderer::GbufferTargets {
            albedo: &albedo_view,
            normal: &normal_view,
            material_id: &material_id_view,
            world_position: &world_pos_view,
            material_params: &material_params_view,
            depth: &depth_view,
            motion: None,
            previous_depth: None,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("outline silhouette gbuffer"),
        });
        renderer.render_gbuffer(&device, &mut encoder, &targets, &mesh, 1);
        queue.submit(std::iter::once(encoder.finish()));
        let (color, color_view) = offscreen_target(&device, SIZE, SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("outline silhouette overlay"),
        });
        renderer.render_outline_for_test(
            &device,
            &queue,
            &mut encoder,
            &color_view,
            &depth_view,
            &mesh,
            1,
            FORMAT,
            1,
            &[2],
        );
        queue.submit(std::iter::once(encoder.finish()));
        let pixels = readback_rgba8(&device, &queue, &color, SIZE, SIZE);
        assert_eq!(
            rgb_at(&pixels, SIZE, CENTER, CENTER),
            [0, 0, 0],
            "the object's own depth must occlude the hull centre"
        );
        assert_ne!(
            rgb_at(&pixels, SIZE, CENTER + RING_OFFSET, CENTER),
            [0, 0, 0],
            "the hull ring outside the silhouette must be painted"
        );
    }

    /// Production path: `new_with_aa_outline` with non-empty flags must run
    /// `render_composite_with_outline` (depth-compatible composite pipeline,
    /// one pass) without validation errors and paint the ring over the
    /// composited frame. The centre stays identical to the plain frame.
    #[test]
    fn composite_outline_production_path_draws_over_the_frame() {
        let Some((device, queue)) = try_device() else {
            // `try_device` panics instead when ORNIS_REQUIRE_GPU is set.
            eprintln!("SKIPPED (composite outline): no wgpu adapter");
            return;
        };
        const SIZE: u32 = 128;
        const CENTER: u32 = SIZE / 2;
        const RING_OFFSET: u32 = 40;
        let renderer = Renderer3D::new(&device, &surface_config_sized(SIZE), 1);
        renderer.set_anti_alias(crate::AntiAliasMode::Fxaa);
        let mut plan = crate::RenderFrame3D::new_with_aa_outline(
            FORMAT,
            (SIZE, SIZE),
            crate::Technique::Hybrid,
            crate::Bloom::Off,
            crate::AaWiring::new(crate::AntiAliasMode::Fxaa),
        );
        let view = crate::camera::CameraView {
            eye: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            projection: ornis_assets::scene::CameraProjection::Perspective,
            fov: FOV,
            near: NEAR,
            far: FAR,
        };
        let (view_proj, eye) = crate::camera::camera_view_projection(&view, (SIZE, SIZE));
        renderer.set_camera(&queue, &view_proj.to_cols_array_2d(), eye.to_array());
        renderer.upload_instances(&device, &queue, &instances(1));
        let mesh = crate::create_sphere(&device, 1.0, 16, 12);
        let (color, color_view) = offscreen_target(&device, SIZE, SIZE);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut render = |flags: &[u32]| {
            plan.set_outline_flags(flags);
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            plan.render(
                crate::RenderContext {
                    device: &device,
                    queue: &queue,
                    encoder: &mut encoder,
                    target: &color_view,
                },
                &renderer,
                &mesh,
                1,
            );
            queue.submit(std::iter::once(encoder.finish()));
            readback_rgba8(&device, &queue, &color, SIZE, SIZE)
        };
        let plain = render(&[]);
        let selected = render(&[2]);
        let validation = pollster::block_on(scope.pop());
        assert!(
            validation.is_none(),
            "composite+outline must not raise validation errors: {validation:?}"
        );
        assert_eq!(
            rgb_at(&plain, SIZE, CENTER, CENTER),
            rgb_at(&selected, SIZE, CENTER, CENTER),
            "the object's depth must keep the hull centre identical"
        );
        assert_ne!(
            rgb_at(&plain, SIZE, CENTER + RING_OFFSET, CENTER),
            rgb_at(&selected, SIZE, CENTER + RING_OFFSET, CENTER),
            "the outline ring must paint over the composited frame"
        );
    }
}
