//! Screen-space AO fullscreen passes: the raw SSAO/GTAO kernel pass plus
//! the bilateral blur (see [`crate::shaders::ao_generated`]).
//!
//! Both stages render at half resolution into the pooled `ao_raw`/`ao`
//! targets; the blurred factor modulates only the ambient and IBL terms of
//! the deferred lighting pass. The pipelines are size-independent (one
//! build per sample count), so [`Renderer3D::resize`] leaves them alone —
//! only the lighting bind group re-selects the AO view per frame.
use super::*;
use crate::ao::{AoSettings, DEFAULT_AO_BLUR_SIGMA};

/// Per-frame AO parameters for the raw and blur passes.
///
/// `radius`/`intensity`/`bias` feed the kernels, `blur_sigma` the bilateral
/// depth weight — the same math as the [`crate::ao`] CPU reference.
///
/// The WGSL `AoParams` declaration is generated from this layout
/// ([`AoUniform::WGSL_SOURCE`]); the field list here is the single source
/// of truth for the buffer layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "AoParams")]
pub(crate) struct AoUniform {
    /// Sample radius in meters.
    radius: f32,
    /// Occlusion strength in `[0, 1]`.
    intensity: f32,
    /// Depth bias in meters (SSAO only; GTAO ignores it).
    bias: f32,
    /// Bilateral depth sigma in meters (blur only).
    blur_sigma: f32,
}

impl AoUniform {
    /// Packs validated [`AoSettings`] for upload.
    pub(crate) fn pack(settings: AoSettings) -> Self {
        Self {
            radius: settings.radius(),
            intensity: settings.intensity(),
            bias: settings.bias(),
            blur_sigma: DEFAULT_AO_BLUR_SIGMA,
        }
    }
}

/// AO fullscreen pipelines: one raw pipeline per kernel sharing a layout,
/// plus the blur pipeline. Created per sample count (the raw layout binds
/// the multisampled depth/normal at 4x).
pub(crate) struct AoPipelines {
    raw_ssao: wgpu::RenderPipeline,
    raw_gtao: wgpu::RenderPipeline,
    blur: wgpu::RenderPipeline,
    raw_layout: wgpu::BindGroupLayout,
    blur_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params_buffer: wgpu::Buffer,
    _white: wgpu::Texture,
    white_view: wgpu::TextureView,
}

impl AoPipelines {
    /// White fallback texture (kept alive by [`AoPipelines::_white`]).
    pub(crate) fn white_texture(&self) -> &wgpu::Texture {
        &self._white
    }
}

/// Inputs of the AO raw pass: the kernel settings plus the g-buffer depth
/// and normals it reconstructs from, and the half-res target. Grouped so
/// the pass signature stays small (like [`FogInputs`](super::FogInputs)).
pub struct AoRawInputs<'a> {
    /// Kernel settings (SSAO/GTAO plus radius/intensity/bias).
    pub settings: AoSettings,
    /// G-buffer hardware depth buffer.
    pub depth: &'a wgpu::TextureView,
    /// G-buffer octahedral normal layer.
    pub normal: &'a wgpu::TextureView,
    /// Half-res raw factor target.
    pub output: &'a wgpu::TextureView,
}

/// Inputs of the AO blur pass: the kernel settings (blur sigma) plus the
/// half-res raw factor and the depth it stays edge-aware against.
/// Grouped like [`AoRawInputs`].
pub struct AoBlurInputs<'a> {
    /// Kernel settings (only the blur sigma is read here).
    pub settings: AoSettings,
    /// G-buffer hardware depth buffer (bilateral weight source).
    pub depth: &'a wgpu::TextureView,
    /// Half-res raw AO factor.
    pub ao_raw: &'a wgpu::TextureView,
    /// Half-res blurred factor target (same extent as `ao_raw`).
    pub output: &'a wgpu::TextureView,
}

impl Renderer3D {
    /// Copies the white staging texel into the AO fallback texture once
    /// (`new` has no queue — the IBL staging precedent).
    pub(super) fn encode_ao_white_if_needed(&self, encoder: &mut wgpu::CommandEncoder) {
        if self
            .ao_white_uploaded
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &self.ao_white_staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(AO_WHITE_ROW_STRIDE),
                    rows_per_image: Some(1),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: self.ao_passes.white_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.ao_white_uploaded
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// One half-res fullscreen pipeline over `layout`: shared by the raw
    /// kernels and the blur (same topology, target format and sample count).
    fn ao_fullscreen_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        vertex_module: &wgpu::ShaderModule,
        fragment_module: &wgpu::ShaderModule,
        label: &str,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: vertex_module,
                entry_point: Some(shaders::ao_generated::vs_main::entry_point()),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: fragment_module,
                entry_point: Some(shaders::ao_generated::fs_main::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::ao::AO_FORMAT,
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
                count: SINGLE_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }

    pub(super) fn create_ao_passes(device: &wgpu::Device, sample_count: u32) -> AoPipelines {
        let vertex_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ao vertex (generated)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::ao_vertex())),
        });
        let (raw_layout, blur_layout) = Self::ao_bind_layouts(device, sample_count);
        let raw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ao raw pipeline layout"),
            bind_group_layouts: &[Some(&raw_layout)],
            immediate_size: 0,
        });
        let raw_module_for = |device: &wgpu::Device, method: crate::ao::AoMethod, label: &str| {
            let source = shaders::ao_generated::wgsl_source_raw_for_samples(method, sample_count);
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
            })
        };
        let raw_ssao_module = raw_module_for(device, crate::ao::AoMethod::Ssao, "ao raw ssao");
        let raw_gtao_module = raw_module_for(device, crate::ao::AoMethod::Gtao, "ao raw gtao");
        let raw_ssao = Self::ao_fullscreen_pipeline(
            device,
            &raw_pipeline_layout,
            &vertex_module,
            &raw_ssao_module,
            "ao raw ssao pipeline",
        );
        let raw_gtao = Self::ao_fullscreen_pipeline(
            device,
            &raw_pipeline_layout,
            &vertex_module,
            &raw_gtao_module,
            "ao raw gtao pipeline",
        );
        let blur_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ao blur shader (generated)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::ao_blur_fragment())),
        });
        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ao blur pipeline layout"),
            bind_group_layouts: &[Some(&blur_layout)],
            immediate_size: 0,
        });
        let blur = Self::ao_fullscreen_pipeline(
            device,
            &blur_pipeline_layout,
            &vertex_module,
            &blur_module,
            "ao blur pipeline",
        );
        let (params_buffer, sampler, white, white_view) = Self::ao_frame_objects(device);
        AoPipelines {
            raw_ssao,
            raw_gtao,
            blur,
            raw_layout,
            blur_layout,
            sampler,
            params_buffer,
            _white: white,
            white_view,
        }
    }

    /// Bind-group layouts of the raw and blur stages. The raw layout binds
    /// the multisampled depth/normal at 4x (see
    /// [`ao_keeps_per_sample`](crate::shaders::ao_generated::ao_keeps_per_sample));
    /// the blur layout stays single-sample.
    fn ao_bind_layouts(
        device: &wgpu::Device,
        sample_count: u32,
    ) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
        let raw_entries: Vec<wgpu::BindGroupLayoutEntry> = shaders::ao_generated::AO_RAW_RESOURCES
            .iter()
            .map(|r| {
                shaders::bgl_entry(
                    r,
                    shaders::ao_generated::ao_per_sample_flag(
                        sample_count,
                        shaders::ao_generated::ao_keeps_per_sample(r.name, &r.kind),
                    ),
                )
            })
            .collect();
        let raw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ao raw bind group layout"),
            entries: &raw_entries,
        });
        let blur_entries: Vec<wgpu::BindGroupLayoutEntry> =
            shaders::ao_generated::AO_BLUR_RESOURCES
                .iter()
                .map(|r| shaders::bgl_entry(r, false))
                .collect();
        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ao blur bind group layout"),
            entries: &blur_entries,
        });
        (raw_layout, blur_layout)
    }

    /// Per-frame objects: the params buffer (default settings until the
    /// first pass uploads), the linear sampler, and the white fallback
    /// texture (filled from staging on the first lighting pass).
    fn ao_frame_objects(
        device: &wgpu::Device,
    ) -> (
        wgpu::Buffer,
        wgpu::Sampler,
        wgpu::Texture,
        wgpu::TextureView,
    ) {
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ao params buffer"),
            contents: bytemuck::bytes_of(&AoUniform::pack(AoSettings::default())),
            usage: CPU_UNIFORM_USAGE,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ao sampler"),
            ..crate::flags::SamplerKind::LinearClamp.descriptor()
        });
        let white = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ao white fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: SINGLE_SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: crate::ao::AO_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let white_view = white.create_view(&wgpu::TextureViewDescriptor::default());
        (params_buffer, sampler, white, white_view)
    }

    /// White 1×1 AO view: the lighting fallback when no AO pass ran, so the
    /// `× ao` modulate stays the exact identity there.
    pub fn ao_white_view(&self) -> &wgpu::TextureView {
        &self.ao_passes.white_view
    }

    /// Record the raw AO kernel pass: reconstructs the occluded factor from
    /// `inputs.depth`/`inputs.normal` into half-res `inputs.output`. The
    /// kernel comes from the settings; every pixel is overwritten (white
    /// clear is neutral).
    pub fn render_ao_raw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        inputs: AoRawInputs<'_>,
    ) {
        let AoRawInputs {
            settings,
            depth,
            normal,
            output,
        } = inputs;
        queue.write_buffer(
            &self.ao_passes.params_buffer,
            0,
            bytemuck::bytes_of(&AoUniform::pack(settings)),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ao raw bind group"),
            layout: &self.ao_passes.raw_layout,
            entries: &shaders::bind_group_entries(&shaders::ao_generated::AO_RAW_RESOURCES, |r| {
                match r.name {
                    "depth_tex" => Some(wgpu::BindingResource::TextureView(depth)),
                    "normal_tex" => Some(wgpu::BindingResource::TextureView(normal)),
                    "camera" => Some(self.camera_buffer.as_entire_binding()),
                    "ao_params" => Some(wgpu::BindingResource::Buffer(
                        self.ao_passes.params_buffer.as_entire_buffer_binding(),
                    )),
                    _ => None,
                }
            })
            .unwrap_or_default(),
        });
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ao raw pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if settings.method().is_ssao() {
            rpass.set_pipeline(&self.ao_passes.raw_ssao);
        } else {
            rpass.set_pipeline(&self.ao_passes.raw_gtao);
        }
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..FULLSCREEN_QUAD_VERTS, 0..1);
    }

    /// Record the bilateral blur: depth-weighted smooth of half-res
    /// `inputs.ao_raw` into `inputs.output` (same extent). Every pixel is
    /// overwritten.
    pub fn render_ao_blur(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        inputs: AoBlurInputs<'_>,
    ) {
        let AoBlurInputs {
            settings,
            depth,
            ao_raw,
            output,
        } = inputs;
        queue.write_buffer(
            &self.ao_passes.params_buffer,
            0,
            bytemuck::bytes_of(&AoUniform::pack(settings)),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ao blur bind group"),
            layout: &self.ao_passes.blur_layout,
            entries: &shaders::bind_group_entries(&shaders::ao_generated::AO_BLUR_RESOURCES, |r| {
                match r.name {
                    "ao_raw_tex" => Some(wgpu::BindingResource::TextureView(ao_raw)),
                    "blur_sampler" => Some(wgpu::BindingResource::Sampler(&self.ao_passes.sampler)),
                    "depth_tex" => Some(wgpu::BindingResource::TextureView(depth)),
                    "ao_params" => Some(wgpu::BindingResource::Buffer(
                        self.ao_passes.params_buffer.as_entire_buffer_binding(),
                    )),
                    _ => None,
                }
            })
            .unwrap_or_default(),
        });
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ao blur pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&self.ao_passes.blur);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..FULLSCREEN_QUAD_VERTS, 0..1);
    }
}

/// Staging bytes for the 1×1 white AO fallback: one white texel, the row
/// padded to the copy alignment (the `new`-has-no-queue precedent from
/// IBL — the first lighting pass copies this into the fallback texture).
pub(super) fn ao_white_staging_bytes() -> Vec<u8> {
    const TIGHT: usize = 4;
    const STRIDE: usize = 256;
    let mut bytes = vec![0u8; STRIDE];
    bytes[..TIGHT].copy_from_slice(&[255u8; TIGHT]);
    bytes
}

/// Byte stride of the white staging row.
pub(super) const AO_WHITE_ROW_STRIDE: u32 = 256;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ao_uniform_layout_matches_wgsl() {
        // `radius` at 0, `intensity` at 4, `bias` at 8, `blur_sigma` at 12:
        // 16 bytes total.
        assert_eq!(std::mem::size_of::<AoUniform>(), 16);
        assert_eq!(std::mem::offset_of!(AoUniform, radius), 0);
        assert_eq!(std::mem::offset_of!(AoUniform, intensity), 4);
        assert_eq!(std::mem::offset_of!(AoUniform, bias), 8);
        assert_eq!(std::mem::offset_of!(AoUniform, blur_sigma), 12);
        assert_eq!(
            AoUniform::FIELD_NAMES,
            &["radius", "intensity", "bias", "blur_sigma"]
        );
        let packed = AoUniform::pack(AoSettings::default());
        assert_eq!(packed.radius, crate::ao::DEFAULT_AO_RADIUS);
        assert_eq!(packed.intensity, crate::ao::DEFAULT_AO_INTENSITY);
        assert_eq!(packed.bias, crate::ao::DEFAULT_AO_BIAS);
        assert_eq!(packed.blur_sigma, crate::ao::DEFAULT_AO_BLUR_SIGMA);
    }
}
