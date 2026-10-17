//! AO pipeline construction: fullscreen pipelines, bind-group layouts, and
//! the per-frame params buffer, sampler, and white fallback texture.
//!
//! Split from the pass module so each file stays within the production-line
//! gate. Recording (`render_ao_raw` / `render_ao_blur`) stays next to the
//! pass inputs.

use super::super::{CPU_UNIFORM_USAGE, Renderer3D, SINGLE_SAMPLE_COUNT};
use super::{AoPipelines, AoUniform};
use crate::ao::AoSettings;
use crate::shaders;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

impl Renderer3D {
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

    /// AO fullscreen pipelines for `sample_count` (raw SSAO and GTAO share a
    /// layout; the blur stage stays single-sample).
    pub(in crate::renderer) fn create_ao_passes(
        device: &wgpu::Device,
        sample_count: u32,
    ) -> AoPipelines {
        let vertex_module = ao_shader_module(device, "ao vertex (generated)", shaders::ao_vertex());
        let (raw_layout, blur_layout) = Self::ao_bind_layouts(device, sample_count);
        let raw_pipeline_layout = ao_pipeline_layout(device, "ao raw pipeline layout", &raw_layout);
        let raw_module_for = |device: &wgpu::Device, method: crate::ao::AoMethod, label: &str| {
            let source = if sample_count == SINGLE_SAMPLE_COUNT {
                shaders::ao_raw_fragment(method)
            } else {
                shaders::ao_generated::wgsl_source_raw_for_samples(method, sample_count)
            };
            ao_shader_module(device, label, source)
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
        let blur_module = ao_shader_module(
            device,
            "ao blur shader (generated)",
            shaders::ao_blur_fragment(),
        );
        let blur_pipeline_layout =
            ao_pipeline_layout(device, "ao blur pipeline layout", &blur_layout);
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
}

/// One generated AO shader module.
fn ao_shader_module(device: &wgpu::Device, label: &str, source: String) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
    })
}

/// Pipeline layout over a single AO bind group.
fn ao_pipeline_layout(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    })
}
