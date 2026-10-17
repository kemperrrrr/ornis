//! Anti-aliasing pipeline construction: the temporal resolve (two colour
//! targets) and the post-process edge filter, plus the history ping-pong
//! textures.
//!
//! Split from the pass module so each file stays within the production-line
//! gate.

use super::super::{CPU_UNIFORM_USAGE, Renderer3D, SINGLE_SAMPLE_COUNT};
use super::{AaUniform, HISTORY_FORMAT};
use crate::antialias::{AntiAliasMode, EdgeFilterSettings};
use crate::shaders;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

/// One slot of the temporal history ping-pong.
pub(crate) struct AaHistory {
    /// The history texture the resolve writes.
    pub(crate) _texture: wgpu::Texture,
    /// The view of the history texture.
    pub(crate) view: wgpu::TextureView,
}

/// Anti-aliasing pipelines: the temporal resolve (two colour attachments)
/// and the shared edge filter, plus the history ping-pong and the
/// per-frame params buffer.
pub(crate) struct AaPipelines {
    /// The temporal resolve pipeline.
    pub(crate) taa: wgpu::RenderPipeline,
    /// The edge filter pipeline (FXAA and SMAA share it).
    pub(crate) edge: wgpu::RenderPipeline,
    /// Bind-group layout of the temporal resolve.
    pub(crate) taa_layout: wgpu::BindGroupLayout,
    /// Bind-group layout of the edge filter.
    pub(crate) edge_layout: wgpu::BindGroupLayout,
    /// Per-frame parameters of both passes.
    pub(crate) params_buffer: wgpu::Buffer,
    /// Linear clamp sampler shared by both passes.
    pub(crate) sampler: wgpu::Sampler,
    /// The history ping-pong: the resolve reads one slot and writes the
    /// other.
    pub(crate) history: std::sync::RwLock<Option<[AaHistory; 2]>>,
    /// The slot the next resolve reads from.
    current_history: std::sync::atomic::AtomicU32,
}

impl AaPipelines {
    /// The slots of the ping-pong: the one to read from and the one to
    /// write to.
    pub(crate) fn history_slots(&self) -> (usize, usize) {
        let read = self
            .current_history
            .load(std::sync::atomic::Ordering::Relaxed) as usize;
        (read, 1 - read)
    }

    /// The resolve wrote the slot `index`: the next frame reads it.
    pub(crate) fn advance_history(&self, index: usize) {
        self.current_history
            .store(index as u32, std::sync::atomic::Ordering::Relaxed);
    }

    /// Lazily creates the history textures on the first TAA frame.
    pub(crate) fn ensure_history(&self, device: &wgpu::Device, width: u32, height: u32) {
        let mut history = super::super::write_lock(&self.history);
        if history.is_none() {
            *history = Some([
                create_history(device, width, height),
                create_history(device, width, height),
            ]);
            self.current_history
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Resizes the history only if TAA has allocated it.
    pub(crate) fn resize_history(&self, device: &wgpu::Device, width: u32, height: u32) {
        let mut history = super::super::write_lock(&self.history);
        if history.is_some() {
            *history = Some([
                create_history(device, width, height),
                create_history(device, width, height),
            ]);
            self.current_history
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Whether the persistent history textures currently occupy GPU memory.
    pub(crate) fn has_history(&self) -> bool {
        super::super::read_lock(&self.history).is_some()
    }
}

impl Renderer3D {
    /// Anti-aliasing pipelines for a surface format and sample count
    /// (the temporal resolve binds the multisampled depth at 4x). The TAA
    /// history is allocated lazily, only when the temporal pass first runs.
    pub(in crate::renderer) fn create_aa_passes(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> AaPipelines {
        let taa_module = aa_shader_module(
            device,
            "taa shader (generated)",
            shaders::taa_fragment(sample_count),
        );
        let edge_module = aa_shader_module(
            device,
            "aa edge shader (generated)",
            shaders::aa_edge_fragment(),
        );
        let taa_layout = aa_bind_layout(
            device,
            "taa bind group layout",
            &shaders::aa_taa_generated::TAA_RESOURCES,
            sample_count,
        );
        let edge_layout = aa_bind_layout(
            device,
            "aa edge bind group layout",
            &shaders::aa_edge_generated::EDGE_RESOURCES,
            sample_count,
        );
        let taa = aa_pipeline(
            device,
            "taa pipeline",
            &taa_module,
            &taa_layout,
            &[surface_format, HISTORY_FORMAT],
            shaders::aa_taa_generated::vs_main::entry_point(),
            shaders::aa_taa_generated::fs_main::entry_point(),
        );
        let edge = aa_pipeline(
            device,
            "aa edge pipeline",
            &edge_module,
            &edge_layout,
            &[surface_format],
            shaders::aa_edge_generated::vs_main::entry_point(),
            shaders::aa_edge_generated::fs_main::entry_point(),
        );
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aa params buffer"),
            contents: bytemuck::bytes_of(&AaUniform::pack_edge(
                EdgeFilterSettings::default(),
                AntiAliasMode::Fxaa,
            )),
            usage: CPU_UNIFORM_USAGE,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("aa sampler"),
            ..crate::flags::SamplerKind::LinearClamp.descriptor()
        });
        AaPipelines {
            taa,
            edge,
            taa_layout,
            edge_layout,
            params_buffer,
            sampler,
            history: std::sync::RwLock::new(None),
            current_history: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

/// One history texture of the ping-pong.
fn create_history(device: &wgpu::Device, width: u32, height: u32) -> AaHistory {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("aa history"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: SINGLE_SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format: HISTORY_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    AaHistory {
        _texture: texture,
        view,
    }
}

/// One generated anti-aliasing shader module.
fn aa_shader_module(device: &wgpu::Device, label: &str, source: String) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
    })
}

/// Bind-group layout of one anti-aliasing pass: the depth binding of the
/// temporal resolve is multisampled at 4x.
fn aa_bind_layout(
    device: &wgpu::Device,
    label: &str,
    table: &[shaders::Resource],
    sample_count: u32,
) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = table
        .iter()
        .map(|r| {
            let multisampled = sample_count > 1 && shaders::resource_stays_multisampled(&r.kind);
            shaders::bgl_entry(r, multisampled)
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

/// One fullscreen anti-aliasing pipeline over `layout`: a triangle-strip
/// quad writing the given colour targets.
fn aa_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    targets: &[wgpu::TextureFormat],
    vertex_entry: &str,
    fragment_entry: &str,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("aa pipeline layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let color_targets: Vec<Option<wgpu::ColorTargetState>> = targets
        .iter()
        .map(|format| {
            Some(wgpu::ColorTargetState {
                format: *format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })
        })
        .collect();
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vertex_entry),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fragment_entry),
            targets: &color_targets,
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
