//! Forward HDR layer: transparency modes, the lit-forward and textured-forward passes, and their `render_forward` entry points.
use super::*;
/// The two shader-visible IBL scalars, packed for an offset write that
/// does not touch the light array.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct IblTail {
    pub(super) weight: f32,
    pub(super) max_mip: f32,
}

/// Bind-group entries shared by the forward and legacy PBR layouts.
#[allow(clippy::too_many_arguments)]
pub(super) fn forward_ibl_entries<'a>(
    camera: &'a wgpu::Buffer,
    per_object: &'a wgpu::Buffer,
    material: &'a wgpu::Buffer,
    lighting: &'a wgpu::Buffer,
    shadow_array_view: &'a wgpu::TextureView,
    shadow_sampler: &'a wgpu::Sampler,
    shadow_cube_array_view: &'a wgpu::TextureView,
    ibl: &'a crate::ibl::IblTargets,
) -> Vec<wgpu::BindGroupEntry<'a>> {
    shaders::bind_group_entries(&shaders::pbr_generated::PBR_RESOURCES, |r| match r.name {
        "camera" => Some(camera.as_entire_binding()),
        "per_objects" => Some(per_object.as_entire_binding()),
        "materials" => Some(material.as_entire_binding()),
        "lighting" => Some(lighting.as_entire_binding()),
        "shadow_tex" => Some(wgpu::BindingResource::TextureView(shadow_array_view)),
        "shadow_sampler" => Some(wgpu::BindingResource::Sampler(shadow_sampler)),
        "shadow_cube_tex" => Some(wgpu::BindingResource::TextureView(shadow_cube_array_view)),
        "prefilter_cube" | "irradiance_cube" | "brdf_lut" | "ibl_sampler" => ibl.binding(r.name),
        _ => None,
    })
    .unwrap_or_default()
}

/// Blend mode of the forward HDR layer (typed replacement for the
/// `sorted_alpha: bool` flag).
///
/// [`BlendMode::Opaque`] (default) writes with `REPLACE`: the default frame
/// (all opacities at 1.0, where `REPLACE` and `ALPHA_BLENDING` coincide)
/// stays golden-pinned. [`BlendMode::Transparent`] blends with
/// `ALPHA_BLENDING` and requires back-to-front submission order — see
/// [`TransparencyOptions`]: sort with [`sort_by_depth`] (per-instance
/// depths) or [`crate::extraction::sort_by_depth`] (whole [`crate::extraction::FrameUpload`])
/// before uploading, farthest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlendMode {
    /// Opaque forward layer (`REPLACE`, default).
    #[default]
    Opaque,
    /// Sorted alpha blend (`ALPHA_BLENDING`, opt-in; back-to-front
    /// submission order — see [`TransparencyOptions`], including the
    /// single-draw order contract).
    Transparent,
}

impl BlendMode {
    /// `true` for [`BlendMode::Transparent`].
    pub fn is_transparent(self) -> bool {
        matches!(self, Self::Transparent)
    }

    /// Blend state of the forward pipeline for this mode.
    pub fn blend_state(self) -> wgpu::BlendState {
        match self {
            Self::Transparent => wgpu::BlendState::ALPHA_BLENDING,
            Self::Opaque => wgpu::BlendState::REPLACE,
        }
    }

    /// Maps a material opacity to a blend mode: finite opacities `>= 1.0`
    /// are opaque, finite opacities below that are transparent.
    ///
    /// # Errors
    ///
    /// Returns [`TransparencyError::NonFiniteOpacity`] for non-finite input.
    pub fn from_opacity(opacity: f32) -> Result<Self, TransparencyError> {
        if !opacity.is_finite() {
            return Err(TransparencyError::NonFiniteOpacity(opacity));
        }
        if opacity >= 1.0 {
            Ok(Self::Opaque)
        } else {
            Ok(Self::Transparent)
        }
    }
}

impl From<bool> for BlendMode {
    /// Legacy `sorted_alpha: bool` polarity.
    fn from(sorted_alpha: bool) -> Self {
        if sorted_alpha {
            Self::Transparent
        } else {
            Self::Opaque
        }
    }
}

impl From<BlendMode> for bool {
    /// Legacy `sorted_alpha: bool` polarity.
    fn from(mode: BlendMode) -> Self {
        mode.is_transparent()
    }
}

/// Rejected transparency input (fallible opacity classification).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum TransparencyError {
    /// Opacity must be finite to classify into a [`BlendMode`].
    #[error("opacity {0} is not finite")]
    NonFiniteOpacity(f32),
}

/// Opt-in transparency for the forward HDR layer.
///
/// The forward layer is always cleared to transparent black
/// (`ClearTransparent`); [`mode`](Self::mode) only selects the blend state
/// of the forward pipeline (see [`forward_blend_state`]) and whether
/// callers must submit instances back-to-front. With
/// [`BlendMode::Transparent`], sort before uploading: per-instance depths
/// via [`sort_by_depth`], or a whole extracted frame via
/// [`crate::extraction::sort_by_depth`] (both order farthest first,
/// stable). Off by default so the default frame stays golden-pinned.
///
/// Order contract: the current forward submit is a single instanced
/// `draw_indexed` over `0..instance_count` (see
/// [`render_forward`](Renderer3D::render_forward)), which ignores instance
/// order — sorting is a no-op today and only takes effect with future
/// multi-draw submits (one draw per instance, back-to-front).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TransparencyOptions {
    /// Blend mode of the forward pipeline (default [`BlendMode::Opaque`]).
    pub mode: BlendMode,
}

impl TransparencyOptions {
    /// Options for the given [`BlendMode`].
    pub fn new(mode: BlendMode) -> Self {
        Self { mode }
    }

    /// Legacy `sorted_alpha` polarity: `true` when transparent.
    pub fn sorted_alpha(self) -> bool {
        self.mode.is_transparent()
    }
}

impl From<bool> for TransparencyOptions {
    /// Legacy `sorted_alpha: bool` polarity.
    fn from(sorted_alpha: bool) -> Self {
        Self {
            mode: BlendMode::from(sorted_alpha),
        }
    }
}

impl From<BlendMode> for TransparencyOptions {
    /// Wraps the mode into options.
    fn from(mode: BlendMode) -> Self {
        Self { mode }
    }
}

/// Blend state of the forward pipeline for the given transparency
/// options: `REPLACE` by default, `ALPHA_BLENDING` when transparent.
/// Pure (no GPU access) so the default can be pinned without an adapter.
pub fn forward_blend_state(options: TransparencyOptions) -> wgpu::BlendState {
    options.mode.blend_state()
}

/// CPU-side back-to-front draw order for `depths` (view-space depth per
/// instance): indices sorted by descending depth, far first, stable
/// (equal depths keep submission order). `NaN` sorts as farthest
/// (`total_cmp`). Pure helper for the future sorted forward submit: the
/// current single-draw forward pass (one instanced `draw_indexed` in
/// [`render_forward`](Renderer3D::render_forward)) ignores instance order,
/// so sorting is a no-op until multi-draw submits land.
pub fn sort_by_depth(depths: &[f32]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..depths.len() as u32).collect();
    order.sort_by(|&a, &b| depths[b as usize].total_cmp(&depths[a as usize]));
    order
}

/// Forward pass for transparency-friendly objects: draws geometry with full
/// lighting into an HDR layer, testing against the gbuffer's depth.
pub struct ForwardPass {
    /// Lit-forward pipeline (same shading as the lighting pass).
    pipeline: wgpu::RenderPipeline,
    /// Bindings for the forward pipeline (layout is stable).
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    /// Current bind group, rebuilt whenever a storage buffer grows.
    pub(super) bind_group: std::sync::RwLock<wgpu::BindGroup>,
    /// Owned HDR color attachment.
    _color_texture: wgpu::Texture,
    /// View of `_color_texture`.
    pub(super) color_view: wgpu::TextureView,
    /// Owned single-sample resolve texture for the MSAA forward color
    /// layer, `Some` only in MSAA mode (see [`MSAA_SAMPLE_COUNT`]).
    _resolve_texture: Option<wgpu::Texture>,
    /// View of `_resolve_texture`: what the composite pass samples in MSAA
    /// mode. `None` at 1x (the composite samples `color_view` directly).
    pub(super) resolve_view: Option<wgpu::TextureView>,
}

/// Textured-forward pass: the legacy [`ForwardPass`] evaluation plus one
/// bound [`MaterialTextureSet`] per draw (see
/// [`crate::shaders::material_textures`]).
///
/// Built on demand by
/// [`Renderer3D::ensure_textured_forward`](Renderer3D::ensure_textured_forward)
/// (the constructor takes no queue, so upload happens there, not in
/// [`new`](Renderer3D::new)); the legacy passes never reference it, so
/// untextured frames stay pixel-identical by construction. Unbound role
/// slots resolve to neutral 1x1 white fallbacks (`x * 1.0 == x`,
/// IEEE-exact): color roles share one `Rgba8UnormSrgb` fallback (hardware
/// sRGB-decodes to linear white), the data role owns one `Rgba8Unorm`
/// fallback (linear white, so roughness/metallic factors stand alone).
pub struct TexturedForwardPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    fallback_color: GpuTexture,
    fallback_data: GpuTexture,
    sampler: wgpu::Sampler,
}

impl Renderer3D {
    // Internal pass constructor: 4 buffers + surface parameters —
    // grouping them into a struct would not improve call readability.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_forward_pass(
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        per_object_buffer: &wgpu::Buffer,
        material_buffer: &wgpu::Buffer,
        lighting_buffer: &wgpu::Buffer,
        shadow_array_view: &wgpu::TextureView,
        shadow_sampler: &wgpu::Sampler,
        shadow_cube_array_view: &wgpu::TextureView,
        ibl: &crate::ibl::IblTargets,
        width: u32,
        height: u32,
        sample_count: u32,
        transparency: TransparencyOptions,
    ) -> ForwardPass {
        // Same layout as the PBR bind group: entries from the shared table.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = shaders::pbr_generated::PBR_RESOURCES
            .iter()
            .map(|r| shaders::bgl_entry(r, false))
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forward bind group layout"),
            entries: &bgl_entries,
        });

        let bind_group = std::sync::RwLock::new(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("forward bind group"),
                layout: &bind_group_layout,
                entries: &shaders::bind_group_entries(
                    &shaders::pbr_generated::PBR_RESOURCES,
                    |r| match r.name {
                        "camera" => Some(camera_buffer.as_entire_binding()),
                        "per_objects" => Some(per_object_buffer.as_entire_binding()),
                        "materials" => Some(material_buffer.as_entire_binding()),
                        "lighting" => Some(lighting_buffer.as_entire_binding()),
                        "shadow_tex" => Some(wgpu::BindingResource::TextureView(shadow_array_view)),
                        "shadow_sampler" => Some(wgpu::BindingResource::Sampler(shadow_sampler)),
                        "shadow_cube_tex" => {
                            Some(wgpu::BindingResource::TextureView(shadow_cube_array_view))
                        }
                        "prefilter_cube" | "irradiance_cube" | "brdf_lut" | "ibl_sampler" => {
                            ibl.binding(r.name)
                        }
                        _ => None,
                    },
                )
                .unwrap_or_default(),
            }),
        );

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("forward color target"),
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
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Single-sample resolve target for the forward HDR color, MSAA mode
        // only: the pass resolves into it and the composite pass samples it.
        let (resolve_texture, resolve_view) = if sample_count > 1 {
            let (texture, view) = Self::create_resolve_target(
                device,
                width,
                height,
                wgpu::TextureFormat::Rgba16Float,
                "forward color resolve",
            );
            (Some(texture), Some(view))
        } else {
            (None, None)
        };

        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::pbr_vertex())),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward fragment"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::pbr_fragment())),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forward pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("forward pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(shaders::gbuffer_generated::vs_main::entry_point()),
                buffers: &[Some(Vertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some(shaders::pbr_generated::fs_main::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(forward_blend_state(transparency)),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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

        ForwardPass {
            pipeline,
            bind_group_layout,
            bind_group,
            _color_texture: color_texture,
            color_view,
            _resolve_texture: resolve_texture,
            resolve_view,
        }
    }

    /// Resolves the view sampled for `role`: the cache entry bound in
    /// `textures`, or the neutral fallback (color roles share
    /// `fallback_color`, the data role uses `fallback_data`) when the slot
    /// is unbound or the handle is stale.
    fn material_view<'a>(
        textures: &MaterialTextureSet,
        cache: &'a TextureCache,
        fallback_color: &'a wgpu::TextureView,
        fallback_data: &'a wgpu::TextureView,
        role: TextureRole,
    ) -> &'a wgpu::TextureView {
        textures
            .binding(role)
            .and_then(|handle| cache.get(handle))
            .map_or(
                match role {
                    TextureRole::MetallicRoughness => fallback_data,
                    TextureRole::BaseColor | TextureRole::Emissive => fallback_color,
                },
                |texture| &texture.view,
            )
    }

    /// Build the textured-forward pipeline and its neutral fallback
    /// textures; idempotent (later calls are no-ops).
    ///
    /// The fallbacks are 1x1 white images uploaded through
    /// [`upload_texture`](crate::textures::upload_texture) under their
    /// roles (sRGB for color, linear for data), and the sampler carries
    /// [`sampler_descriptor_for_role`](crate::textures::sampler_descriptor_for_role)
    /// defaults. The fragment stage is
    /// [`wgsl_source_textured`](crate::shaders::material_textures::wgsl_source_textured);
    /// everything else (vertex entry, targets, depth, MSAA) mirrors
    /// [`TransparencyOptions`]-aware [`ForwardPass`] construction, so only
    /// textured draws change appearance.
    pub fn ensure_textured_forward(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.textured_forward.is_some() {
            return;
        }
        let Ok(white) = CpuImage::from_rgba8(1, 1, vec![255; 4]) else {
            return;
        };
        let Ok(fallback_color) = upload_texture(device, queue, &white, TextureRole::BaseColor)
        else {
            return;
        };
        let Ok(fallback_data) =
            upload_texture(device, queue, &white, TextureRole::MetallicRoughness)
        else {
            return;
        };
        let sampler = device.create_sampler(&sampler_descriptor_for_role(TextureRole::BaseColor));

        // Same table-driven layout as every other pass: entries from the
        // textured resource table, so WGSL declarations and layout agree.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> =
            shaders::material_textures::TEXTURED_PBR_RESOURCES
                .iter()
                .map(|r| shaders::bgl_entry(r, false))
                .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("textured forward bind group layout"),
            entries: &bgl_entries,
        });

        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("textured forward vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::pbr_vertex())),
        });
        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("textured forward fragment"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(
                shaders::material_textures::wgsl_source_textured(),
            )),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("textured forward pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("textured forward pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(shaders::gbuffer_generated::vs_main::entry_point()),
                buffers: &[Some(Vertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some(shaders::material_textures::fs_main_textured::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(forward_blend_state(self.transparency)),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
                count: self.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        self.textured_forward = Some(TexturedForwardPass {
            pipeline,
            bind_group_layout,
            fallback_color,
            fallback_data,
            sampler,
        });
    }

    /// Record the forward pass: draws lit geometry into the HDR `output`
    /// layer, depth-testing against (and optionally clearing) `depth`.
    /// `clear_depth = true` when the forward pass runs standalone; `false`
    /// when it follows the gbuffer pass and must share its depth.
    ///
    /// In MSAA mode `output` must be a multisampled view of matching extent
    /// (the renderer's own forward color view, as
    /// [`render_scene`](Self::render_scene) passes): the pass resolves into
    /// the stored single-sample view. At 1x no resolve is recorded.
    pub fn render_forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        mesh: &Mesh,
        instance_count: u32,
        clear_depth: bool,
    ) {
        self.encode_ibl_if_needed(encoder);
        let depth_ops = wgpu::Operations {
            load: if clear_depth {
                wgpu::LoadOp::Clear(1.0)
            } else {
                wgpu::LoadOp::Load
            },
            store: wgpu::StoreOp::Store,
        };
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("forward pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: self.forward_pass.resolve_view.as_ref(),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(depth_ops),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&self.forward_pass.pipeline);
        {
            let bind_group = read_lock(&self.forward_pass.bind_group);
            rpass.set_bind_group(0, &*bind_group, &[]);
        }
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.num_indices, 0, 0..instance_count);
    }

    /// Record the forward pass with per-entity custom draws: the shared
    /// `mesh` batch first, then each [`SkinnedDraw::Cpu`] entry of
    /// `customs` in order — one render pass, cleared once.
    ///
    /// There is no skinned forward pipeline (skinned entries are carried
    /// by the deferred layer), so [`SkinnedDraw::Gpu`] entries are skipped
    /// here — a deliberate gap, not a silent drop: follow-up work is a
    /// skinned forward pipeline or routing transparent skinned materials
    /// through it. With empty `customs` the command stream matches
    /// [`render_forward`](Self::render_forward) exactly.
    ///
    /// In MSAA mode `output` must be a multisampled view of matching extent
    /// (see [`render_forward`](Self::render_forward)).
    #[allow(clippy::too_many_arguments)]
    pub fn render_forward_with_custom(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        mesh: &Mesh,
        instance_count: u32,
        clear_depth: bool,
        customs: &[CustomGbufferDraw<'_>],
    ) {
        self.encode_ibl_if_needed(encoder);
        let depth_ops = wgpu::Operations {
            load: if clear_depth {
                wgpu::LoadOp::Clear(1.0)
            } else {
                wgpu::LoadOp::Load
            },
            store: wgpu::StoreOp::Store,
        };
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("forward pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: self.forward_pass.resolve_view.as_ref(),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(depth_ops),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&self.forward_pass.pipeline);
        {
            let bind_group = read_lock(&self.forward_pass.bind_group);
            rpass.set_bind_group(0, &*bind_group, &[]);
        }
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.num_indices, 0, 0..instance_count);
        for custom in customs {
            let SkinnedDraw::Cpu = custom.draw else {
                continue;
            };
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

    /// Record the textured-forward pass: draws lit geometry with one bound
    /// [`MaterialTextureSet`] (resolved against `cache`, unbound slots fall
    /// back to neutral white) into the HDR `output` layer, depth-testing
    /// against (and optionally clearing) `depth`. `clear_depth = true` when
    /// the pass runs standalone; `false` when it follows the gbuffer pass
    /// and must share its depth. Mirrors [`render_forward`](Self::render_forward);
    /// the scalar `is_metallic` switch is untouched by bound textures.
    ///
    /// In MSAA mode `output` must be a multisampled view of matching extent
    /// (see [`render_forward`](Self::render_forward)).
    ///
    /// # Panics
    ///
    /// Panics when [`ensure_textured_forward`](Self::ensure_textured_forward)
    /// was not called yet — the pipeline and fallbacks do not exist until
    /// then (the constructor takes no queue, so they cannot be built there).
    #[allow(clippy::too_many_arguments)]
    pub fn render_forward_textured(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        mesh: &Mesh,
        instance_count: u32,
        clear_depth: bool,
        textures: &MaterialTextureSet,
        cache: &TextureCache,
    ) {
        let Some(pass) = self.textured_forward.as_ref() else {
            return;
        };
        self.encode_ibl_if_needed(encoder);
        // The bind group is rebuilt per frame: the material buffer may have
        // grown and the bound set may have changed. Binding numbers come
        // from the table; only the name → live resource mapping is here.
        let material = read_lock(&self.material_buffer);
        let per_object = read_lock(&self.per_object_buffer);
        let base_color_view = Self::material_view(
            textures,
            cache,
            &pass.fallback_color.view,
            &pass.fallback_data.view,
            TextureRole::BaseColor,
        );
        let data_view = Self::material_view(
            textures,
            cache,
            &pass.fallback_color.view,
            &pass.fallback_data.view,
            TextureRole::MetallicRoughness,
        );
        let emissive_view = Self::material_view(
            textures,
            cache,
            &pass.fallback_color.view,
            &pass.fallback_data.view,
            TextureRole::Emissive,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("textured forward bind group (frame)"),
            layout: &pass.bind_group_layout,
            entries: &shaders::bind_group_entries(
                &shaders::material_textures::TEXTURED_PBR_RESOURCES,
                |r| match r.name {
                    "camera" => Some(self.camera_buffer.as_entire_binding()),
                    "per_objects" => Some(per_object.as_entire_binding()),
                    "materials" => Some(material.as_entire_binding()),
                    "lighting" => Some(self.lighting_buffer.as_entire_binding()),
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
                    "base_color_tex" => Some(wgpu::BindingResource::TextureView(base_color_view)),
                    "metallic_roughness_tex" => Some(wgpu::BindingResource::TextureView(data_view)),
                    "emissive_tex" => Some(wgpu::BindingResource::TextureView(emissive_view)),
                    "material_sampler" => Some(wgpu::BindingResource::Sampler(&pass.sampler)),
                    _ => None,
                },
            )
            .unwrap_or_default(),
        });

        let depth_ops = wgpu::Operations {
            load: if clear_depth {
                wgpu::LoadOp::Clear(1.0)
            } else {
                wgpu::LoadOp::Load
            },
            store: wgpu::StoreOp::Store,
        };
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("textured forward pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: self.forward_pass.resolve_view.as_ref(),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(depth_ops),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&pass.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.num_indices, 0, 0..instance_count);
    }
}
