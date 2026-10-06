//! Deferred frame assembly: the fullscreen lighting pass plus `render_deferred_frame`/`render_scene`, the all-in-one frame drivers.
use super::*;
/// Full-screen deferred lighting pass: reads the five g-buffer targets +
/// depth, evaluates the PBR BRDF, writes the HDR color image.
pub struct LightingPass {
    /// Full-screen triangle-strip pipeline writing Rgba16Float HDR.
    pipeline: wgpu::RenderPipeline,
    /// Bindings: camera/lighting/material buffers, 5 gbuffer views, depth, sampler.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Linear sampler for gbuffer fetches (MSAA resolve handled upstream).
    sampler: wgpu::Sampler,
}

impl Renderer3D {
    pub(super) fn create_lighting_pass(
        device: &wgpu::Device,
        output_view: &wgpu::TextureView,
        sample_count: u32,
    ) -> LightingPass {
        // Layout entries come from the pass resource table — the same table
        // that generates the WGSL declarations, so shader and layout agree.
        // Depth, material-id and the normal stay multisampled at 4x (the
        // normal must not box-filter the octahedral clear into +Z). Other
        // float layers bind the single-sample resolve.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> =
            shaders::lighting_generated::LIGHTING_RESOURCES
                .iter()
                .map(|r| {
                    shaders::bgl_entry(
                        r,
                        shaders::lighting_generated::per_sample_flag(
                            sample_count,
                            shaders::lighting_generated::keeps_per_sample(r.name, &r.kind),
                        ),
                    )
                })
                .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lighting bind group layout"),
            entries: &bgl_entries,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lighting sampler"),
            ..crate::flags::SamplerKind::LinearClamp.descriptor()
        });

        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lighting vertex (generated)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(
                shaders::lighting_generated::wgsl_vertex_source(),
            )),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lighting fragment (generated)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(
                shaders::lighting_generated::wgsl_source_for_samples(sample_count),
            )),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lighting pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lighting pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(shaders::lighting_generated::vs_main::entry_point()),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some(shaders::lighting_generated::fs_main::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_view.texture().format(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            // Fullscreen pass over resolved layers: single-sample in all
            // modes (the output target is single-sample; see `new`).
            multisample: wgpu::MultisampleState {
                count: SINGLE_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        LightingPass {
            pipeline,
            bind_group_layout,
            sampler,
        }
    }

    /// Record the deferred lighting pass: reconstructs surface data from the
    /// g-buffer in `g`, evaluates the OpenPBR BRDF and writes HDR color into `output`.
    ///
    /// In MSAA mode albedo, world position and material params are sampled
    /// from the renderer's stored resolve textures. Depth, material-id and
    /// the normal bind the multisampled `g` views: interior pixels shade
    /// sample 0, and edge pixels average only covered samples so a cleared
    /// octahedral `(0, 0)` (which decodes to +Z) cannot fringe the
    /// silhouette. `g` must be the renderer's own MSAA views in that mode,
    /// as [`render_scene`](Self::render_scene) passes. At 1x every binding
    /// is `g` itself.
    pub fn render_lighting(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        g: &GbufferTargets<'_>,
        output: &wgpu::TextureView,
    ) {
        // The bind group is rebuilt per frame: gbuffer views come from the
        // render-plan pool (transient) or from persistent textures, and the
        // material buffer may have grown since the last frame.
        // Binding numbers come from the table; only the name → live
        // resource mapping is written out here.
        let material = read_lock(&self.material_buffer);
        // Resolve views in MSAA mode, pass-through views at 1x (the helper
        // returns `g`'s own view there, so the 1x bind group is unchanged).
        // The normal stays on the multisampled view: lighting loads each
        // sample instead of the hardware box filter.
        let resolves = self.gbuffer.resolves.as_ref();
        let albedo_view = resolves.map_or(g.albedo, |r| &r.albedo_view);
        let normal_view = g.normal;
        let world_position_view = resolves.map_or(g.world_position, |r| &r.world_position_view);
        let material_params_view = resolves.map_or(g.material_params, |r| &r.material_params_view);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting bind group (frame)"),
            layout: &self.lighting_pass.bind_group_layout,
            entries: &shaders::bind_group_entries(
                &shaders::lighting_generated::LIGHTING_RESOURCES,
                |r| match r.name {
                    "camera" => Some(self.camera_buffer.as_entire_binding()),
                    "lighting" => Some(self.lighting_buffer.as_entire_binding()),
                    "materials" => Some(material.as_entire_binding()),
                    "albedo_tex" => Some(wgpu::BindingResource::TextureView(albedo_view)),
                    "normal_tex" => Some(wgpu::BindingResource::TextureView(normal_view)),
                    "material_id_tex" => Some(wgpu::BindingResource::TextureView(g.material_id)),
                    "world_pos_tex" => {
                        Some(wgpu::BindingResource::TextureView(world_position_view))
                    }
                    "mat_params_tex" => {
                        Some(wgpu::BindingResource::TextureView(material_params_view))
                    }
                    "depth_tex" => Some(wgpu::BindingResource::TextureView(g.depth)),
                    "lighting_sampler" => {
                        Some(wgpu::BindingResource::Sampler(&self.lighting_pass.sampler))
                    }
                    "shadow_tex" => {
                        Some(wgpu::BindingResource::TextureView(&self.shadow_array_view))
                    }
                    "shadow_sampler" => Some(wgpu::BindingResource::Sampler(&self.shadow_sampler)),
                    "shadow_cube_tex" => Some(wgpu::BindingResource::TextureView(
                        &self.shadow_cube_array_view,
                    )),
                    "prefilter_cube" | "irradiance_cube" | "brdf_lut" | "ibl_sampler" => {
                        self.ibl.binding(r.name)
                    }
                    _ => None,
                },
            )
            .unwrap_or_default(),
        });

        self.encode_ibl_if_needed(encoder);
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lighting pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&self.lighting_pass.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..FULLSCREEN_QUAD_VERTS, 0..1);
    }

    /// Deferred g-buffer plus lighting into `target`, without the forward
    /// pass. Test-only: the edge probe needs the deferred term alone.
    #[cfg(test)]
    fn render_deferred_frame(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        mesh: &Mesh,
    ) {
        let g = GbufferTargets {
            albedo: &self.gbuffer.albedo_view,
            normal: &self.gbuffer.normal_view,
            material_id: &self.gbuffer.material_id_view,
            world_position: &self.gbuffer.world_position_view,
            material_params: &self.gbuffer.material_params_view,
            depth: &self.gbuffer.depth_view,
        };
        self.render_gbuffer(encoder, &g, mesh, 1);
        self.render_lighting(device, encoder, &g, &self.pbr_texture_view);
        self.render_composite(
            device,
            queue,
            encoder,
            CompositeInputs {
                target,
                hdr: &self.pbr_texture_view,
                hdr_fwd: &self.pbr_texture_view,
                bloom: &self.pbr_texture_view,
                bloom_intensity: 0.0,
                mode: 0,
            },
        );
    }

    /// All-in-one legacy frame on the renderer's persistent targets:
    /// gbuffer -> lighting -> forward -> composite straight into `target`.
    /// The render-graph path (`frame_exec`) supersedes this for plan-driven
    /// execution, but it remains the reference hybrid pipeline.
    pub fn render_scene(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        mesh: &Mesh,
        instance_count: u32,
    ) {
        let g = GbufferTargets {
            albedo: &self.gbuffer.albedo_view,
            normal: &self.gbuffer.normal_view,
            material_id: &self.gbuffer.material_id_view,
            world_position: &self.gbuffer.world_position_view,
            material_params: &self.gbuffer.material_params_view,
            depth: &self.gbuffer.depth_view,
        };
        self.render_gbuffer(encoder, &g, mesh, instance_count);
        // Depth pre-passes for shadowed lights (no-op when none).
        self.render_shadows(device, encoder, mesh, instance_count);
        self.render_lighting(device, encoder, &g, &self.pbr_texture_view);
        self.render_forward(
            encoder,
            &self.gbuffer.depth_view,
            &self.forward_pass.color_view,
            mesh,
            instance_count,
            false,
        );
        self.render_composite(
            device,
            queue,
            encoder,
            CompositeInputs {
                target,
                hdr: &self.pbr_texture_view,
                // In MSAA mode the forward pass resolved into the stored
                // single-sample view; at 1x that view does not exist and the
                // color view is sampled directly (unchanged).
                hdr_fwd: self
                    .forward_pass
                    .resolve_view
                    .as_ref()
                    .unwrap_or(&self.forward_pass.color_view),
                bloom: &self.pbr_texture_view,
                bloom_intensity: 0.0,
                // Legacy path always runs the hybrid mix.
                mode: 2,
            },
        );
    }
}
