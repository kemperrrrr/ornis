//! G-buffer targets, resolve views, the geometry pipeline, and the geometry pre-pass (`render_gbuffer`) behind `Renderer3D`.
use super::*;
/// G-buffer texture views, fed either from persistent textures (legacy
/// path) or from render-plan pool slots (plan path).
pub struct GbufferTargets<'a> {
    /// Base albedo (sRGB) view.
    pub albedo: &'a wgpu::TextureView,
    /// View-space/world normals view.
    pub normal: &'a wgpu::TextureView,
    /// Material identifier view.
    pub material_id: &'a wgpu::TextureView,
    /// World-space positions view.
    pub world_position: &'a wgpu::TextureView,
    /// Material parameter table view.
    pub material_params: &'a wgpu::TextureView,
    /// Depth buffer view.
    pub depth: &'a wgpu::TextureView,
}

/// Persistent g-buffer textures of the legacy path (plan path draws into
/// pooled slots instead) plus their views.
pub struct GBufferTextures {
    /// Albedo/base color target (Rgba8Unorm).
    pub albedo: wgpu::Texture,
    /// View of [`GBufferTextures::albedo`].
    pub albedo_view: wgpu::TextureView,
    /// World-space normal target (Rg16Float).
    pub normal: wgpu::Texture,
    /// View of [`GBufferTextures::normal`].
    pub normal_view: wgpu::TextureView,
    /// Material id target (`R16Uint`: the 16-bit integer format is the
    /// widest integer format the WebGPU spec guarantees multisampling for —
    /// `R32Uint` has no `MULTISAMPLE_X*` flag without adapter-specific
    /// features, so 4x g-buffer creation would fail on native Metal. Ids
    /// above `u16::MAX` truncate (loudly debug-asserted at upload; the
    /// table holds dozens of entries in practice).
    pub material_id: wgpu::Texture,
    /// View of [`GBufferTextures::material_id`].
    pub material_id_view: wgpu::TextureView,
    /// World-space position target (Rg16Float xy + z from depth).
    pub world_position: wgpu::Texture,
    /// View of [`GBufferTextures::world_position`].
    pub world_position_view: wgpu::TextureView,
    /// Material parameter target (Rgba16Float).
    pub material_params: wgpu::Texture,
    /// View of [`GBufferTextures::material_params`].
    pub material_params_view: wgpu::TextureView,
    /// Depth buffer (Depth32Float), reused by the forward pass.
    pub depth: wgpu::Texture,
    /// View of [`GBufferTextures::depth`].
    pub depth_view: wgpu::TextureView,
    /// Single-sample resolve targets for the float color layers, `Some` only
    /// in MSAA mode ([`MSAA_SAMPLE_COUNT`]): the g-buffer pass resolves into
    /// them and downstream passes sample them. `None` at 1x (no extra
    /// textures, no resolves — the 1x command stream is unchanged).
    pub(super) resolves: Option<GbufferResolves>,
}

/// Single-sample MSAA resolve targets for the g-buffer float color layers
/// (see [`MSAA_SAMPLE_COUNT`]). Depth and the integer material-id layer have
/// no resolve target in `wgpu` and stay multisampled (loaded as sample 0).
pub(super) struct GbufferResolves {
    /// Resolve texture pairing [`GBufferTextures::albedo`].
    _albedo_texture: wgpu::Texture,
    /// View of the albedo resolve texture, sampled downstream.
    pub(super) albedo_view: wgpu::TextureView,
    /// Resolve texture pairing [`GBufferTextures::normal`].
    _normal_texture: wgpu::Texture,
    /// View of the normal resolve texture, sampled downstream.
    pub(super) normal_view: wgpu::TextureView,
    /// Resolve texture pairing [`GBufferTextures::world_position`].
    _world_position_texture: wgpu::Texture,
    /// View of the world-position resolve texture, sampled downstream.
    pub(super) world_position_view: wgpu::TextureView,
    /// Resolve texture pairing [`GBufferTextures::material_params`].
    _material_params_texture: wgpu::Texture,
    /// View of the material-params resolve texture, sampled downstream.
    pub(super) material_params_view: wgpu::TextureView,
}

/// Selector for one MSAA-resolvable g-buffer float color layer (see
/// [`Renderer3D::gbuffer_resolve_target`]). Depth and material-id are not
/// selectable: they have no resolve target in `wgpu`.
#[derive(Debug, Clone, Copy)]
pub(super) enum GbufferResolveSlot {
    /// Albedo/base color layer.
    Albedo,
    /// World-space normal layer.
    Normal,
    /// World-space position layer.
    WorldPosition,
    /// Material parameter layer.
    MaterialParams,
}

impl Renderer3D {
    pub(super) fn create_gbuffer(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> GBufferTextures {
        let albedo = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer albedo"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let albedo_view = albedo.create_view(&wgpu::TextureViewDescriptor::default());

        let normal = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer normal"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let normal_view = normal.create_view(&wgpu::TextureViewDescriptor::default());

        let material_id = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer material_id"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            // `R16Uint`, not `R32Uint`: the spec-guaranteed multisampleable
            // integer format (see the `material_id` field docs). The WGSL
            // type stays `u32`, so shaders and layouts are unchanged.
            format: wgpu::TextureFormat::R16Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let material_id_view = material_id.create_view(&wgpu::TextureViewDescriptor::default());

        let world_position = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer world_position"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let world_position_view =
            world_position.create_view(&wgpu::TextureViewDescriptor::default());

        let material_params = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer material_params"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let material_params_view =
            material_params.create_view(&wgpu::TextureViewDescriptor::default());

        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

        // Single-sample resolve targets for the float color layers, MSAA
        // mode only: the pass resolves into them, downstream passes sample
        // them. Depth and the integer id stay multisampled (no resolve
        // target exists for them; they load sample 0).
        let resolves = (sample_count > 1).then(|| {
            let (albedo_texture, albedo_view) = Self::create_resolve_target(
                device,
                width,
                height,
                wgpu::TextureFormat::Rgba8Unorm,
                "gbuffer albedo resolve",
            );
            let (normal_texture, normal_view) = Self::create_resolve_target(
                device,
                width,
                height,
                wgpu::TextureFormat::Rg16Float,
                "gbuffer normal resolve",
            );
            let (world_position_texture, world_position_view) = Self::create_resolve_target(
                device,
                width,
                height,
                wgpu::TextureFormat::Rg16Float,
                "gbuffer world_position resolve",
            );
            let (material_params_texture, material_params_view) = Self::create_resolve_target(
                device,
                width,
                height,
                wgpu::TextureFormat::Rgba16Float,
                "gbuffer material_params resolve",
            );
            GbufferResolves {
                _albedo_texture: albedo_texture,
                albedo_view,
                _normal_texture: normal_texture,
                normal_view,
                _world_position_texture: world_position_texture,
                world_position_view,
                _material_params_texture: material_params_texture,
                material_params_view,
            }
        });

        GBufferTextures {
            albedo,
            albedo_view,
            normal,
            normal_view,
            material_id,
            material_id_view,
            world_position,
            world_position_view,
            material_params,
            material_params_view,
            depth,
            depth_view,
            resolves,
        }
    }

    /// Single-sample resolve texture pairing an MSAA color target: same
    /// extent and format, `RENDER_ATTACHMENT` (resolve destination) plus
    /// `TEXTURE_BINDING` (sampled downstream).
    pub(super) fn create_resolve_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: SINGLE_SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    pub(super) fn create_gbuffer_pipeline(
        device: &wgpu::Device,
        _gbuffer: &GBufferTextures,
        camera_buffer: &wgpu::Buffer,
        per_object_buffer: &wgpu::Buffer,
        material_buffer: &wgpu::Buffer,
        sample_count: u32,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::BindGroup) {
        // Layout entries come from the pass resource table.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> =
            shaders::gbuffer_generated::GBUFFER_RESOURCES
                .iter()
                .map(|r| shaders::bgl_entry(r, sample_count > 1))
                .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gbuffer bind group layout"),
            entries: &bgl_entries,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gbuffer bind group"),
            layout: &bind_group_layout,
            entries: &shaders::bind_group_entries(
                &shaders::gbuffer_generated::GBUFFER_RESOURCES,
                |r| match r.name {
                    "camera" => Some(camera_buffer.as_entire_binding()),
                    "per_objects" => Some(per_object_buffer.as_entire_binding()),
                    "materials" => Some(material_buffer.as_entire_binding()),
                    _ => None,
                },
            )
            .unwrap_or_default(),
        });

        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gbuffer vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::gbuffer_vertex())),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gbuffer fragment"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::gbuffer_fragment())),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gbuffer pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gbuffer pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(shaders::gbuffer_generated::vs_main::entry_point()),
                buffers: &[Some(Vertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some(shaders::gbuffer_generated::fs_main::entry_point()),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        // `R16Uint`: the spec-guaranteed multisampleable
                        // integer format (see the `material_id` field docs).
                        format: wgpu::TextureFormat::R16Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
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

        (pipeline, bind_group_layout, bind_group)
    }

    /// Resolve target for one MSAA g-buffer float color layer: the stored
    /// single-sample view in MSAA mode, `None` at 1x (no resolve recorded —
    /// the 1x command stream is unchanged).
    ///
    /// In MSAA mode the `view` side must be a multisampled view of matching
    /// extent and format: the renderer's own g-buffer views (as
    /// [`render_scene`](Self::render_scene) passes) or same-extent custom 4x
    /// views. Depth and material-id never resolve (see [`MSAA_SAMPLE_COUNT`]).
    pub(super) fn gbuffer_resolve_target(
        &self,
        slot: GbufferResolveSlot,
    ) -> Option<&wgpu::TextureView> {
        let resolves = self.gbuffer.resolves.as_ref()?;
        Some(match slot {
            GbufferResolveSlot::Albedo => &resolves.albedo_view,
            GbufferResolveSlot::Normal => &resolves.normal_view,
            GbufferResolveSlot::WorldPosition => &resolves.world_position_view,
            GbufferResolveSlot::MaterialParams => &resolves.material_params_view,
        })
    }

    /// Record the gbuffer pass: fills the five MRT targets + depth for
    /// `instance_count` uploaded instances of `mesh`.
    ///
    /// In MSAA mode the float layers resolve into the stored single-sample
    /// views; `g` must then be the renderer's own multisampled views (as
    /// [`render_scene`](Self::render_scene) passes). At 1x no resolve is
    /// recorded.
    pub fn render_gbuffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        g: &GbufferTargets<'_>,
        mesh: &Mesh,
        instance_count: u32,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gbuffer pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: g.albedo,
                    depth_slice: None,
                    resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::Albedo),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.normal,
                    depth_slice: None,
                    resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::Normal),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.material_id,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.world_position,
                    depth_slice: None,
                    resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::WorldPosition),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.material_params,
                    depth_slice: None,
                    resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::MaterialParams),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: g.depth,
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

        rpass.set_pipeline(&self.gbuffer_pipeline);
        {
            let bind_group = read_lock(&self.gbuffer_bind_group);
            rpass.set_bind_group(0, &*bind_group, &[]);
        }
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.num_indices, 0, 0..instance_count);
    }

    /// Record the gbuffer pass with per-entity custom draws: the shared
    /// `mesh` batch first, then each entry of `customs` in order — one
    /// render pass, cleared once.
    ///
    /// [`SkinnedDraw::Cpu`] entries draw through the classic pipeline from
    /// their per-object slot; [`SkinnedDraw::Gpu`] entries draw through
    /// the skinned pipeline with their palette slot bound. A stale handle
    /// (at or past the last staged count) records no commands for that
    /// entry. With empty `customs` the command stream matches
    /// [`render_gbuffer`](Self::render_gbuffer) exactly (spheres-unchanged
    /// by construction).
    ///
    /// In MSAA mode `g` must be the renderer's own multisampled views (see
    /// [`render_gbuffer`](Self::render_gbuffer)).
    pub fn render_gbuffer_with_custom(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        g: &GbufferTargets<'_>,
        mesh: &Mesh,
        instance_count: u32,
        customs: &[CustomGbufferDraw<'_>],
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gbuffer pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: g.albedo,
                    depth_slice: None,
                    resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::Albedo),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.normal,
                    depth_slice: None,
                    resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::Normal),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.material_id,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.world_position,
                    depth_slice: None,
                    resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::WorldPosition),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.material_params,
                    depth_slice: None,
                    resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::MaterialParams),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: g.depth,
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

        rpass.set_pipeline(&self.gbuffer_pipeline);
        {
            let bind_group = read_lock(&self.gbuffer_bind_group);
            rpass.set_bind_group(0, &*bind_group, &[]);
        }
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.num_indices, 0, 0..instance_count);
        for custom in customs {
            match custom.draw {
                SkinnedDraw::Cpu => {
                    rpass.set_pipeline(&self.gbuffer_pipeline);
                    {
                        let bind_group = read_lock(&self.gbuffer_bind_group);
                        rpass.set_bind_group(0, &*bind_group, &[]);
                    }
                    rpass.set_vertex_buffer(0, custom.mesh.vertex_buffer.slice(..));
                    rpass.set_index_buffer(
                        custom.mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    rpass.draw_indexed(
                        0..custom.mesh.num_indices,
                        0,
                        custom.instance_slot..custom.instance_slot + 1,
                    );
                }
                SkinnedDraw::Gpu(handle) => {
                    if handle.index()
                        >= self
                            .palette_count
                            .load(std::sync::atomic::Ordering::Relaxed)
                            as usize
                    {
                        continue;
                    }
                    let palette = read_lock(&self.palette_buffer);
                    let per_object = read_lock(&self.per_object_buffer);
                    let material = read_lock(&self.material_buffer);
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("skinned gbuffer bind group"),
                        layout: &self.skinned_bind_group_layout,
                        entries: &shaders::bind_group_entries(&GBUFFER_SKINNED_RESOURCES, |r| {
                            match r.name {
                                "camera" => Some(self.camera_buffer.as_entire_binding()),
                                "per_objects" => Some(per_object.as_entire_binding()),
                                "materials" => Some(material.as_entire_binding()),
                                "palette" => {
                                    Some(wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &palette,
                                        offset: handle.byte_offset(),
                                        size: std::num::NonZeroU64::new(PALETTE_BYTE_SIZE as u64),
                                    }))
                                }
                                _ => None,
                            }
                        })
                        .unwrap_or_default(),
                    });
                    rpass.set_pipeline(&self.skinned_pipeline);
                    rpass.set_bind_group(0, &bind_group, &[]);
                    rpass.set_vertex_buffer(0, custom.mesh.vertex_buffer.slice(..));
                    rpass.set_index_buffer(
                        custom.mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    rpass.draw_indexed(
                        0..custom.mesh.num_indices,
                        0,
                        custom.instance_slot..custom.instance_slot + 1,
                    );
                }
            }
        }
    }
}
