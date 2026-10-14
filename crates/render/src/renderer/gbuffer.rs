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
    /// Optional motion-vector target (current UV − previous UV), present
    /// only when the temporal plan runs.
    pub motion: Option<&'a wgpu::TextureView>,
    /// Optional R16Uint previous-depth/validity target paired with `motion`.
    pub previous_depth: Option<&'a wgpu::TextureView>,
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

/// Build a WGSL shader module for either g-buffer vertex layout.
pub(super) fn gbuffer_shader_module(
    device: &wgpu::Device,
    label: &'static str,
    source: String,
) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
    })
}

/// Float g-buffer output target with replacement blending.
pub(super) fn gbuffer_float_target(format: wgpu::TextureFormat) -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState {
        format,
        blend: Some(wgpu::BlendState::REPLACE),
        write_mask: wgpu::ColorWrites::ALL,
    })
}

/// Integer g-buffer output target, which must not use blending.
pub(super) fn gbuffer_integer_target(
    format: wgpu::TextureFormat,
) -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState {
        format,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })
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
    ) -> (
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::BindGroupLayout,
        wgpu::BindGroup,
    ) {
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

        let vs_module = gbuffer_shader_module(device, "gbuffer vertex", shaders::gbuffer_vertex());
        let fs_module =
            gbuffer_shader_module(device, "gbuffer fragment", shaders::gbuffer_fragment());

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gbuffer pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let mut targets = vec![
            gbuffer_float_target(wgpu::TextureFormat::Rgba8Unorm),
            gbuffer_float_target(wgpu::TextureFormat::Rg16Float),
            gbuffer_integer_target(wgpu::TextureFormat::R16Uint),
            gbuffer_float_target(wgpu::TextureFormat::Rg16Float),
            gbuffer_float_target(wgpu::TextureFormat::Rgba16Float),
            None,
            None,
        ];
        let create_pipeline = |label: &'static str, targets: &[Option<wgpu::ColorTargetState>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
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
                    targets,
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
            })
        };
        let pipeline = create_pipeline("gbuffer pipeline", &targets);
        targets[5] = gbuffer_float_target(wgpu::TextureFormat::Rg16Float);
        targets[6] = gbuffer_integer_target(wgpu::TextureFormat::R16Uint);
        let motion_pipeline = create_pipeline("gbuffer motion pipeline", &targets);

        (pipeline, motion_pipeline, bind_group_layout, bind_group)
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
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        g: &GbufferTargets<'_>,
        mesh: &Mesh,
        instance_count: u32,
    ) {
        self.render_gbuffer_batch_with_custom(
            device,
            encoder,
            g,
            mesh,
            BatchDraw::Direct { instance_count },
            &[],
        );
    }

    // qual:api — production entry for the frame wiring slice; in-tree
    // callers are tests until that slice lands.
    /// Record the gbuffer pass drawing the culler's compacted batch with
    /// one
    /// [`draw_indexed_indirect`](wgpu::RenderPass::draw_indexed_indirect):
    /// `culler.visible_buffer` binds as `per_objects` and
    /// `culler.args_buffer` carries the draw (prepared by
    /// [`Culler::compact`](crate::gpu_culling::Culler::compact)). The
    /// CPU fallback is [`render_gbuffer`](Self::render_gbuffer) (direct
    /// `draw_indexed` over the staged range).
    pub fn render_gbuffer_indirect(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        g: &GbufferTargets<'_>,
        mesh: &Mesh,
        culler: &Culler,
    ) {
        self.render_gbuffer_batch_with_custom(
            device,
            encoder,
            g,
            mesh,
            BatchDraw::Indirect { culler },
            &[],
        );
    }

    /// Record the gbuffer pass with per-entity custom draws: the shared
    /// `mesh` batch first, then each entry of `customs` in order — one
    /// render pass, cleared once.
    ///
    /// Slice 2c selection: when an indirect batch was armed for this
    /// frame (`prepare_indirect_batch`), the shared batch submits with
    /// one `draw_indexed_indirect` over the compacted records; otherwise — and for every custom entry in
    /// both modes — the classic direct `draw_indexed` records. Customs
    /// stay direct: single-instance draws gain nothing from indirect.
    ///
    /// [`SkinnedDraw::Cpu`] entries draw through the classic pipeline from
    /// their per-object slot; [`SkinnedDraw::Gpu`] entries draw through
    /// the skinned pipeline with their palette slot bound. A stale handle
    /// (at or past the last staged count) records no commands for that
    /// entry. With empty `customs` and no armed batch the command stream
    /// matches [`render_gbuffer`](Self::render_gbuffer) exactly
    /// (spheres-unchanged by construction).
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
        let guard = read_lock(&self.culler);
        let batch =
            select_shared_batch(self.indirect_batch_ready(), guard.as_ref(), instance_count);
        self.render_gbuffer_batch_with_custom(device, encoder, g, mesh, batch, customs);
    }

    /// Shared core of the gbuffer entry points: one cleared pass, the
    /// shared mesh batch submitted per [`BatchDraw`], then the custom
    /// entries in order.
    fn render_gbuffer_batch_with_custom(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        g: &GbufferTargets<'_>,
        mesh: &Mesh,
        batch: BatchDraw<'_>,
        customs: &[CustomGbufferDraw<'_>],
    ) {
        let mut color_attachments = vec![
            Some(wgpu::RenderPassColorAttachment {
                view: g.albedo,
                depth_slice: None,
                resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::Albedo),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: g.normal,
                depth_slice: None,
                resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::Normal),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: g.material_id,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: g.world_position,
                depth_slice: None,
                resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::WorldPosition),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: g.material_params,
                depth_slice: None,
                resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::MaterialParams),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ];
        let motion = g.motion.is_some() && g.previous_depth.is_some();
        if let (Some(motion), Some(previous_depth)) = (g.motion, g.previous_depth) {
            color_attachments.push(Some(wgpu::RenderPassColorAttachment {
                view: motion,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }));
            color_attachments.push(Some(wgpu::RenderPassColorAttachment {
                view: previous_depth,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }));
        } else {
            // The non-temporal pipeline has trailing `None` targets at
            // locations 5 and 6; both matching attachment slots stay absent.
            color_attachments.push(None);
            color_attachments.push(None);
        }
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gbuffer pass"),
            color_attachments: &color_attachments,
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

        self.draw_shared_batch(device, &mut rpass, mesh, batch, motion);
        for custom in customs {
            match custom.draw {
                SkinnedDraw::Cpu => {
                    rpass.set_pipeline(if motion {
                        &self.gbuffer_motion_pipeline
                    } else {
                        &self.gbuffer_pipeline
                    });
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
                    let previous_palette = read_lock(&self.previous_palette_buffer);
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
                                "previous_palette" => {
                                    Some(wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &previous_palette,
                                        offset: handle.byte_offset(),
                                        size: std::num::NonZeroU64::new(PALETTE_BYTE_SIZE as u64),
                                    }))
                                }
                                _ => None,
                            }
                        })
                        .unwrap_or_default(),
                    });
                    rpass.set_pipeline(if motion {
                        &self.skinned_motion_pipeline
                    } else {
                        &self.skinned_pipeline
                    });
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

    /// Cached indirect bind group for `culler`: reused across frames
    /// instead of allocating a bind group per draw. The cache is filled
    /// on miss here and invalidated when the culler is (re)created or
    /// the material table grows, so a live group always binds the
    /// current `visible_buffer` and material table.
    fn cached_indirect_bind_group<'a>(
        &'a self,
        device: &wgpu::Device,
        culler: &Culler,
    ) -> std::sync::RwLockReadGuard<'a, Option<wgpu::BindGroup>> {
        if read_lock(&self.indirect_bind_group).is_none() {
            let group =
                self.gbuffer_bind_group_with(device, &self.camera_buffer, &culler.visible_buffer);
            *write_lock(&self.indirect_bind_group) = Some(group);
        }
        read_lock(&self.indirect_bind_group)
    }

    /// Records the shared mesh batch: classic `draw_indexed` over the
    /// staged per-object buffer ([`BatchDraw::Direct`]) or one
    /// [`draw_indexed_indirect`](wgpu::RenderPass::draw_indexed_indirect)
    /// over the culler's compacted batch ([`BatchDraw::Indirect`]).
    fn draw_shared_batch(
        &self,
        device: &wgpu::Device,
        rpass: &mut wgpu::RenderPass<'_>,
        mesh: &Mesh,
        batch: BatchDraw<'_>,
        motion: bool,
    ) {
        rpass.set_pipeline(if motion {
            &self.gbuffer_motion_pipeline
        } else {
            &self.gbuffer_pipeline
        });
        let cached = read_lock(&self.gbuffer_bind_group);
        let cached_indirect;
        let bind_group = match batch {
            BatchDraw::Direct { .. } => &*cached,
            BatchDraw::Indirect { culler } => {
                cached_indirect = self.cached_indirect_bind_group(device, culler);
                cached_indirect
                    .as_ref()
                    .expect("indirect bind group is cached on miss above")
            }
        };
        rpass.set_bind_group(0, bind_group, &[]);
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        match batch {
            BatchDraw::Direct { instance_count } => {
                rpass.draw_indexed(0..mesh.num_indices, 0, 0..instance_count);
            }
            BatchDraw::Indirect { culler } => {
                rpass.draw_indexed_indirect(&culler.args_buffer, 0);
            }
        }
    }
}
