//! Post-processing: composite blend, bloom down/upsample chain, and the opt-in distance-fog pass.
use super::*;
/// Final blend pass mixing deferred HDR, forward HDR and bloom into the output.
pub struct CompositePass {
    /// Full-screen triangle-strip pipeline targeting the surface format.
    pipeline: wgpu::RenderPipeline,
    /// Bindings: two HDR layers, sampler, bloom view, params buffer.
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Inputs of the composite pass: the two HDR layers plus the bloom
/// contribution (view + blend intensity). Grouped so the pass signature
/// stays small as the mix gains terms. `mode` selects the blend in the
/// shader: 0 = deferred-only, 1 = forward-only, 2 = hybrid.
pub struct CompositeInputs<'a> {
    /// Output view written by the pass (usually the surface).
    pub target: &'a wgpu::TextureView,
    /// Deferred-lit HDR layer.
    pub hdr: &'a wgpu::TextureView,
    /// Forward-lit HDR layer.
    pub hdr_fwd: &'a wgpu::TextureView,
    /// Bloom contribution texture (may be black when culled).
    pub bloom: &'a wgpu::TextureView,
    /// Multiplier on the bloom contribution (0 disables it).
    pub bloom_intensity: f32,
    /// Layer mix selector in the shader: 0 = deferred-only, 1 = forward-only, 2 = hybrid.
    pub mode: u32,
}

/// Inputs of the opt-in distance-fog pass: the deferred HDR layer plus the
/// g-buffer depth it linearizes, the fog color/density, and the output view.
/// Grouped so the pass signature stays small (like [`CompositeInputs`]).
/// `density` must be finite and `> 0` — anything else records no commands
/// (exact no-op; see [`Renderer3D::render_fog`]).
pub struct FogInputs<'a> {
    /// Deferred-lit HDR layer (fogged in place conceptually; the pass
    /// writes the mix into [`target`](Self::target)).
    pub hdr: &'a wgpu::TextureView,
    /// G-buffer hardware depth buffer (linearized on the GPU).
    pub depth: &'a wgpu::TextureView,
    /// Output view written by the pass (usually the swapchain target).
    pub target: &'a wgpu::TextureView,
    /// Linear-space fog color.
    pub color: [f32; 3],
    /// Exponential density in 1/m.
    pub density: f32,
}

/// Per-frame bloom parameters shared by the bloom passes and the composite
/// pass. `threshold` gates the bright-pass (first downsample level only);
/// `intensity` scales the bloom contribution in the composite pass.
/// `mode` (composite only) picks the layer mix: 0 = deferred-only,
/// 1 = forward-only, 2 = hybrid.
///
/// The WGSL `BloomParams` declaration is generated from this layout
/// ([`BloomUniform::WGSL_SOURCE`]).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "BloomParams")]
pub(crate) struct BloomUniform {
    threshold: f32,
    intensity: f32,
    mode: u32,
    /// Trailing pad (padding: not shader-visible).
    #[wgsl(skip)]
    _pad: f32,
}

impl Default for BloomUniform {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            intensity: 1.0,
            mode: 0,
            _pad: 0.0,
        }
    }
}

/// Per-frame fog parameters for the opt-in distance-fog pass.
///
/// `color` is the linear-space fog color mixed toward with depth;
/// `density` is the exponential falloff rate (1/m, always positive —
/// [`crate::frame_passes::FogState::Disabled`] carries no density at all).
/// Depth is the view-space distance reconstructed from the g-buffer depth
/// buffer (see [`crate::shaders::fog_generated`]): hardware depth is
/// non-linear, so it is linearized through `Camera::inv_view_proj` (the
/// same [`reconstruct_world_pos`](crate::shaders::helpers) path lighting
/// uses) and the Euclidean distance to `Camera::camera_pos` feeds
/// `1 - exp(-density * depth)`.
///
/// The WGSL `FogParams` declaration is generated from this layout
/// ([`FogUniform::WGSL_SOURCE`]); the field list here is the single source
/// of truth for the buffer layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "FogParams")]
pub(crate) struct FogUniform {
    /// Linear-space fog color.
    color: [f32; 3],
    /// Exponential density (1/m, `> 0`).
    density: f32,
}

impl FogUniform {
    /// Packs raw fog color + density for upload.
    pub(crate) fn pack(color: [f32; 3], density: f32) -> Self {
        Self { color, density }
    }
}

/// Bloom pass pipelines: a downsample (replace-blend, clear) and an upsample
/// (additive blend over a loaded target) sharing one fragment shader.
pub struct BloomPass {
    down_pipeline: wgpu::RenderPipeline,
    up_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
}

/// Opt-in distance-fog fullscreen pass (see [`crate::shaders::fog_generated`]).
///
/// Reads the deferred HDR layer plus the g-buffer depth buffer, mixes toward
/// the fog color by `1 - exp(-density * depth)` (depth = view-space distance
/// reconstructed from hardware depth), and writes the swapchain target.
/// Disabled state records no commands (exact no-op); only the enabled mix
/// draws.
pub(crate) struct FogPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
}

/// Shared resources for composite/bloom sampling.
pub struct CompositeResources {
    /// Linear-filtering, clamp-to-edge sampler used by all full-screen passes.
    pub sampler: wgpu::Sampler,
}

impl Renderer3D {
    pub(super) fn create_composite_pass(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> CompositePass {
        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::composite_vertex())),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite fragment"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::composite_fragment())),
        });

        // The composite pass runs the HDR shaders: entries from that table.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> =
            shaders::hdr_composite_generated::HDR_RESOURCES
                .iter()
                .map(|r| shaders::bgl_entry(r, false))
                .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite bind group layout"),
            entries: &bgl_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(shaders::hdr_composite_generated::vs_main::entry_point()),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some(shaders::hdr_composite_generated::fs_main::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
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
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        CompositePass {
            pipeline,
            bind_group_layout,
        }
    }

    pub(super) fn create_bloom_pass(device: &wgpu::Device) -> BloomPass {
        // Bloom WGSL is now generated from Rust (path 2) — single
        // source of truth `shaders::bloom_generated::wgsl_source()`.
        let bloom_source = shaders::bloom_generated::wgsl_source();
        let bloom_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom shader (generated)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(bloom_source)),
        });
        // Vertex and fragment are one module with two entry points `vs_main`/`fs_main`.
        // Two variables point to the same module to preserve the signature
        // `bloom_pipeline(vertex, fragment, ...)`.
        let fs_module = &bloom_module;
        let vs_module = &bloom_module;

        // Layout entries come from the pass resource table.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> =
            shaders::bloom_generated::BLOOM_RESOURCES
                .iter()
                .map(|r| shaders::bgl_entry(r, false))
                .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom bind group layout"),
            entries: &bgl_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bloom params buffer"),
            contents: bytemuck::bytes_of(&BloomUniform::default()),
            usage: CPU_UNIFORM_USAGE,
        });

        // Downsample: replace-blend, target is cleared first.
        let down_pipeline =
            Self::bloom_pipeline(device, &pipeline_layout, fs_module, vs_module, None);
        // Upsample: additive blend over the loaded previous level.
        let up_pipeline = Self::bloom_pipeline(
            device,
            &pipeline_layout,
            fs_module,
            vs_module,
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
        );

        BloomPass {
            down_pipeline,
            up_pipeline,
            bind_group_layout,
            params_buffer,
        }
    }

    fn bloom_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        fs_module: &wgpu::ShaderModule,
        vs_module: &wgpu::ShaderModule,
        blend: Option<wgpu::BlendState>,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: vs_module,
                entry_point: Some(shaders::bloom_generated::vs_main::entry_point()),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: fs_module,
                entry_point: Some(shaders::bloom_generated::fs_main::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend,
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
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }

    /// Builds the opt-in distance-fog fullscreen pass for `surface_format`.
    ///
    /// The shader is generated from Rust
    /// ([`crate::shaders::fog_generated`], `#[stage]` entries — no
    /// handwritten WGSL); the layout comes from its `FOG_RESOURCES` table.
    /// The params buffer starts at a benign enabled-neutral value (black,
    /// zero density is never drawn — [`render_fog`](Self::render_fog)
    /// returns early on non-positive densities, so the disabled pass is an
    /// exact no-op).
    pub(super) fn create_fog_pass(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> FogPipeline {
        let fog_source = shaders::fog_generated::wgsl_source_for_samples(sample_count);
        let fog_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fog shader (generated)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(fog_source)),
        });
        let vertex_source = shaders::fog_generated::wgsl_vertex_source();
        let vertex_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fog vertex (generated)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(vertex_source)),
        });

        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = shaders::fog_generated::FOG_RESOURCES
            .iter()
            .map(|r| shaders::bgl_entry_for_samples(r, sample_count))
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fog bind group layout"),
            entries: &bgl_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fog pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fog params buffer"),
            contents: bytemuck::bytes_of(&FogUniform::pack([0.0; VEC3_COMPONENTS], 1.0)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fog pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_module,
                entry_point: Some(shaders::fog_generated::vs_main::entry_point()),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fog_module,
                entry_point: Some(shaders::fog_generated::fs_main::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
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
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        FogPipeline {
            pipeline,
            bind_group_layout,
            params_buffer,
        }
    }

    /// Record the final blend into `inputs.target`: mixes deferred + forward
    /// HDR layers per `inputs.mode` and adds bloom scaled by
    /// `inputs.bloom_intensity` (0 keeps the legacy path pixel-identical).
    pub fn render_composite(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        inputs: CompositeInputs<'_>,
    ) {
        // The bloom view is bound unconditionally; a zero intensity makes the
        // contribution null, so the legacy path (`render_scene`) stays
        // pixel-identical to the plan path with bloom culled.
        queue.write_buffer(
            &self.bloom_pass.params_buffer,
            0,
            bytemuck::bytes_of(&BloomUniform {
                threshold: 0.0,
                intensity: inputs.bloom_intensity,
                mode: inputs.mode,
                _pad: 0.0,
            }),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite bind group"),
            layout: &self.composite_pass.bind_group_layout,
            // Binding numbers come from the table; only the name → live
            // resource mapping is written out here.
            entries: &shaders::bind_group_entries(
                &shaders::hdr_composite_generated::HDR_RESOURCES,
                |r| match r.name {
                    "deferred_tex" => Some(wgpu::BindingResource::TextureView(inputs.hdr)),
                    "forward_tex" => Some(wgpu::BindingResource::TextureView(inputs.hdr_fwd)),
                    "composite_sampler" => {
                        Some(wgpu::BindingResource::Sampler(&self.composite_sampler))
                    }
                    "bloom_tex" => Some(wgpu::BindingResource::TextureView(inputs.bloom)),
                    "bloom_params" => Some(wgpu::BindingResource::Buffer(
                        self.bloom_pass.params_buffer.as_entire_buffer_binding(),
                    )),
                    _ => None,
                },
            )
            .unwrap_or_default(),
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: inputs.target,
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

        rpass.set_pipeline(&self.composite_pass.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..FULLSCREEN_QUAD_VERTS, 0..1);
    }

    /// Record the opt-in distance-fog mix: `hdr` through the fog blend into
    /// `target`, with depth from the g-buffer `depth` buffer.
    ///
    /// Depth is the hardware (non-linear) depth: it is linearized through
    /// `Camera::inv_view_proj` and the Euclidean distance to the eye feeds
    /// `color + (fog.color - color) * (1 - exp(-density * depth))` — the
    /// same math as [`crate::frame_passes::apply_fog`]. Non-positive or
    /// non-finite `density` records no commands (exact no-op, so the
    /// disabled pass leaves the frame pixel-identical).
    pub fn render_fog(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        inputs: FogInputs<'_>,
    ) {
        let density = inputs.density;
        if !density.is_finite() || density <= 0.0 {
            return;
        }
        queue.write_buffer(
            &self.fog.params_buffer,
            0,
            bytemuck::bytes_of(&FogUniform::pack(inputs.color, density)),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fog bind group"),
            layout: &self.fog.bind_group_layout,
            entries: &shaders::bind_group_entries(&shaders::fog_generated::FOG_RESOURCES, |r| {
                match r.name {
                    "hdr_tex" => Some(wgpu::BindingResource::TextureView(inputs.hdr)),
                    "fog_sampler" => Some(wgpu::BindingResource::Sampler(&self.composite_sampler)),
                    "depth_tex" => Some(wgpu::BindingResource::TextureView(inputs.depth)),
                    "camera" => Some(self.camera_buffer.as_entire_binding()),
                    "fog_params" => Some(wgpu::BindingResource::Buffer(
                        self.fog.params_buffer.as_entire_buffer_binding(),
                    )),
                    _ => None,
                }
            })
            .unwrap_or_default(),
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fog pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: inputs.target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&self.fog.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..FULLSCREEN_QUAD_VERTS, 0..1);
    }

    /// Downsample pass of the bloom chain: thresholded for the first level,
    /// plain downsample for deeper levels (`threshold` = 0 passes everything
    /// except pure black). Writes into `dst` with a replace blend.
    pub fn render_bloom_down(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        threshold: f32,
    ) {
        queue.write_buffer(
            &self.bloom_pass.params_buffer,
            0,
            bytemuck::bytes_of(&BloomUniform {
                threshold,
                intensity: 0.0,
                mode: 0,
                _pad: 0.0,
            }),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom down bind group"),
            layout: &self.bloom_pass.bind_group_layout,
            entries: &shaders::bind_group_entries(
                &shaders::bloom_generated::BLOOM_RESOURCES,
                |r| match r.name {
                    "src_tex" => Some(wgpu::BindingResource::TextureView(src)),
                    "src_sampler" => Some(wgpu::BindingResource::Sampler(&self.composite_sampler)),
                    "bloom_params" => Some(wgpu::BindingResource::Buffer(
                        self.bloom_pass.params_buffer.as_entire_buffer_binding(),
                    )),
                    _ => None,
                },
            )
            .unwrap_or_default(),
        });
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bloom down pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&self.bloom_pass.down_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..FULLSCREEN_QUAD_VERTS, 0..1);
    }

    /// Upsample pass of the bloom chain: samples `src`, adds the result over
    /// the *loaded* contents of `dst` (additive blend) — the classic
    /// "upsample with add" cascade that recombines the levels.
    pub fn render_bloom_up(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom up bind group"),
            layout: &self.bloom_pass.bind_group_layout,
            entries: &shaders::bind_group_entries(
                &shaders::bloom_generated::BLOOM_RESOURCES,
                |r| match r.name {
                    "src_tex" => Some(wgpu::BindingResource::TextureView(src)),
                    "src_sampler" => Some(wgpu::BindingResource::Sampler(&self.composite_sampler)),
                    "bloom_params" => Some(wgpu::BindingResource::Buffer(
                        self.bloom_pass.params_buffer.as_entire_buffer_binding(),
                    )),
                    _ => None,
                },
            )
            .unwrap_or_default(),
        });
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bloom up pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&self.bloom_pass.up_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..FULLSCREEN_QUAD_VERTS, 0..1);
    }
}
