//! The deferred `Renderer3D`: imperative hybrid pipeline of
//! gbuffer (5 MRT) -> lighting -> forward -> composite passes, plus the bloom
//! chain. This is the production implementation behind
//! [`crate::render_backend::RenderBackend`]; see also [`crate::frame_exec`]
//! for the render-graph-driven equivalent.

use crate::mesh::{Mesh, Vertex};
use crate::shaders;
use glam::Mat4;
use ornis_core::material::{OPENPBR_MATERIAL_SIZE, OpenPBRMaterial};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

/// Frame-global camera uniform (binding shared by every pass).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// View-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Its inverse: reconstructs world position from depth in the lighting pass.
    pub inv_view_proj: [[f32; 4]; 4],
    /// World-space eye position (`w` = 1) for specular falloff.
    pub camera_pos: [f32; 4],
}

/// GPU per-instance record mirroring CPU [`InstanceData`] with padding to 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PerObjectGpu {
    /// Local-to-world matrix.
    pub model: [[f32; 4]; 4],
    /// Inverse-transpose model matrix (normal transform).
    pub normal_matrix: [[f32; 4]; 4],
    /// Index into the material buffer uploaded by `upload_materials`.
    pub material_index: u32,
    /// Aligns the record to 16-byte stride.
    _padding: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLight {
    direction: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LightingUniform {
    ambient_color: [f32; 4],
    lights: [GpuLight; 4],
    light_count: u32,
    _pad: [u32; 3],
}

/// CPU-side description of one drawn instance.
#[derive(Debug, Clone, Copy)]
pub struct InstanceData {
    /// Local-to-world matrix.
    pub model_matrix: Mat4,
    /// Normal matrix (inverse transpose of the linear part).
    pub normal_matrix: Mat4,
    /// Index into the material table.
    pub material_index: u32,
}

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
    /// Material id target (R32Uint).
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
}

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

/// Forward pass for transparency-friendly objects: draws geometry with full
/// lighting into an HDR layer, testing against the gbuffer's depth.
pub struct ForwardPass {
    /// Lit-forward pipeline (same shading as the lighting pass).
    pipeline: wgpu::RenderPipeline,
    /// Kept alive for the bind group.
    _bind_group_layout: wgpu::BindGroupLayout,
    /// Buffers bound once at construction.
    bind_group: wgpu::BindGroup,
    /// Owned HDR color attachment.
    _color_texture: wgpu::Texture,
    /// View of `_color_texture`.
    color_view: wgpu::TextureView,
}

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

/// Per-frame bloom parameters shared by the bloom passes and the composite
/// pass. `threshold` gates the bright-pass (first downsample level only);
/// `intensity` scales the bloom contribution in the composite pass.
/// `mode` (composite only) picks the layer mix: 0 = deferred-only,
/// 1 = forward-only, 2 = hybrid.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomUniform {
    threshold: f32,
    intensity: f32,
    mode: u32,
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

/// Bloom pass pipelines: a downsample (replace-blend, clear) and an upsample
/// (additive blend over a loaded target) sharing one fragment shader.
pub struct BloomPass {
    down_pipeline: wgpu::RenderPipeline,
    up_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
}

/// Shared resources for composite/bloom sampling.
pub struct CompositeResources {
    /// Linear-filtering, clamp-to-edge sampler used by all full-screen passes.
    pub sampler: wgpu::Sampler,
}

/// The four core GPU buffers (camera / per-object / material / lighting).
struct CoreBuffers {
    camera: wgpu::Buffer,
    per_object: wgpu::Buffer,
    material: wgpu::Buffer,
    lighting: wgpu::Buffer,
}

/// The deferred 3D renderer: owns GPU buffers, pipelines and persistent
/// targets; drives the hybrid deferred+forward+bloom frame via its
/// `render_*` methods or the all-in-one [`Renderer3D::render_scene`].
pub struct Renderer3D {
    camera_buffer: wgpu::Buffer,
    per_object_buffer: wgpu::Buffer,
    material_buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    _bind_group_layout: wgpu::BindGroupLayout,
    _bind_group: wgpu::BindGroup,
    _pipeline: wgpu::RenderPipeline,
    pbr_texture: wgpu::Texture,
    pbr_texture_view: wgpu::TextureView,
    sample_count: u32,
    max_objects: u32,
    max_materials: u32,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    gbuffer: GBufferTextures,
    gbuffer_pipeline: wgpu::RenderPipeline,
    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    gbuffer_bind_group: wgpu::BindGroup,
    lighting_pass: LightingPass,
    forward_pass: ForwardPass,
    composite_pass: CompositePass,
    /// Linear sampler shared by composite/bloom full-screen passes.
    composite_sampler: wgpu::Sampler,
    /// Bloom chain pipelines and params buffer.
    bloom_pass: BloomPass,
}

impl Renderer3D {
    /// Build every pipeline/target for `surface_config`'s format and extent.
    ///
    /// Capacity is fixed at 256 instances / 64 materials; zero-sized extents
    /// are clamped to 1 pixel.
    pub fn new(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
    ) -> Self {
        let max_objects = 256u32;
        let max_materials = 64u32;
        let format = surface_config.format;
        let width = surface_config.width.max(1);
        let height = surface_config.height.max(1);

        let buffers = Self::create_core_buffers(device, max_objects, max_materials);
        let (bind_group_layout, bind_group) = Self::create_pbr_bind_group(device, &buffers);
        let pipeline =
            Self::create_pbr_pipeline(device, surface_config, sample_count, &bind_group_layout);
        let (pbr_texture, pbr_texture_view) =
            Self::create_render_target(device, width, height, format, sample_count);

        let gbuffer = Self::create_gbuffer(device, width, height, sample_count);
        let (gbuffer_pipeline, gbuffer_bind_group_layout, gbuffer_bind_group) =
            Self::create_gbuffer_pipeline(
                device,
                &gbuffer,
                &buffers.camera,
                &buffers.per_object,
                &buffers.material,
                sample_count,
            );
        let lighting_pass = Self::create_lighting_pass(device, &pbr_texture_view, sample_count);
        let forward_pass = Self::create_forward_pass(
            device,
            &buffers.camera,
            &buffers.per_object,
            &buffers.material,
            &buffers.lighting,
            width,
            height,
            sample_count,
        );
        let composite_pass = Self::create_composite_pass(device, format);
        let bloom_pass = Self::create_bloom_pass(device);
        let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            camera_buffer: buffers.camera,
            per_object_buffer: buffers.per_object,
            material_buffer: buffers.material,
            lighting_buffer: buffers.lighting,
            _bind_group_layout: bind_group_layout,
            _bind_group: bind_group,
            _pipeline: pipeline,
            pbr_texture,
            pbr_texture_view,
            sample_count,
            max_objects,
            max_materials,
            format,
            width,
            height,
            gbuffer,
            gbuffer_pipeline,
            gbuffer_bind_group_layout,
            gbuffer_bind_group,
            lighting_pass,
            forward_pass,
            composite_pass,
            composite_sampler,
            bloom_pass,
        }
    }

    fn create_core_buffers(
        device: &wgpu::Device,
        max_objects: u32,
        max_materials: u32,
    ) -> CoreBuffers {
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera buffer"),
            contents: bytemuck::bytes_of(&CameraUniform {
                view_proj: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                inv_view_proj: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                camera_pos: [0.0, 0.0, 0.0, 1.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let per_object_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("per-object buffer"),
            size: (std::mem::size_of::<PerObjectGpu>() * max_objects as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let material_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material buffer"),
            size: (OPENPBR_MATERIAL_SIZE * max_materials as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let default_lighting = LightingUniform {
            ambient_color: [0.03, 0.03, 0.05, 1.0],
            lights: [GpuLight {
                direction: [0.0; 4],
                color: [0.0; 4],
            }; 4],
            light_count: 0,
            _pad: [0; 3],
        };
        let lighting_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lighting buffer"),
            contents: bytemuck::bytes_of(&default_lighting),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        CoreBuffers {
            camera: camera_buffer,
            per_object: per_object_buffer,
            material: material_buffer,
            lighting: lighting_buffer,
        }
    }

    fn create_pbr_bind_group(
        device: &wgpu::Device,
        buffers: &CoreBuffers,
    ) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbr bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pbr bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.per_object.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.material.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.lighting.as_entire_binding(),
                },
            ],
        });

        (bind_group_layout, bind_group)
    }

    fn create_pbr_pipeline(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbr vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::pbr_vertex())),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbr fragment"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::pbr_fragment())),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbr pipeline layout"),
            bind_group_layouts: &[Some(bind_group_layout)],
            immediate_size: 0,
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pbr render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
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
        })
    }

    fn create_render_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let pbr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pbr render target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = pbr_texture.create_view(&wgpu::TextureViewDescriptor::default());
        (pbr_texture, view)
    }

    fn create_gbuffer(
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
            format: wgpu::TextureFormat::R32Uint,
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
        }
    }

    fn create_gbuffer_pipeline(
        device: &wgpu::Device,
        _gbuffer: &GBufferTextures,
        camera_buffer: &wgpu::Buffer,
        per_object_buffer: &wgpu::Buffer,
        material_buffer: &wgpu::Buffer,
        sample_count: u32,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::BindGroup) {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gbuffer bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gbuffer bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: per_object_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: material_buffer.as_entire_binding(),
                },
            ],
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
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some("fs_main"),
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
                        format: wgpu::TextureFormat::R32Uint,
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

    fn create_lighting_pass(
        device: &wgpu::Device,
        output_view: &wgpu::TextureView,
        sample_count: u32,
    ) -> LightingPass {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lighting bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: sample_count > 1,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: sample_count > 1,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: sample_count > 1,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: sample_count > 1,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: sample_count > 1,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: sample_count > 1,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lighting sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lighting vertex (generated)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(
                shaders::lighting_generated::wgsl_vertex_source(),
            )),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lighting fragment (generated)"),
            source: wgpu::ShaderSource::Wgsl(
                Cow::Owned(shaders::lighting_generated::wgsl_source()),
            ),
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
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some("fs_main"),
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
            multisample: wgpu::MultisampleState {
                count: sample_count,
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

    // Internal pass constructor: 4 buffers + surface parameters —
    // grouping them into a struct would not improve call readability.
    #[allow(clippy::too_many_arguments)]
    fn create_forward_pass(
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        per_object_buffer: &wgpu::Buffer,
        material_buffer: &wgpu::Buffer,
        lighting_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> ForwardPass {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forward bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forward bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: per_object_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: material_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: lighting_buffer.as_entire_binding(),
                },
            ],
        });

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
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            _bind_group_layout: bind_group_layout,
            bind_group,
            _color_texture: color_texture,
            color_view,
        }
    }

    fn create_composite_pass(
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

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<BloomUniform>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
            ],
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
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some("fs_main"),
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

    fn create_bloom_pass(device: &wgpu::Device) -> BloomPass {
        // Bloom WGSL теперь генерируется из Rust (путь 2) — единственный
        // источник истины `shaders::bloom_generated::wgsl_source()`.
        // Легаси `shaders/wgsl/bloom_fragment.wgsl` остаётся как reference.
        let bloom_source = shaders::bloom_generated::wgsl_source();
        let bloom_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom shader (generated)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(bloom_source)),
        });
        // Vertex и fragment — один модуль с двумя entry points `vs_main`/`fs_main`.
        // Две переменные указывают на тот же модуль, чтобы сохранить сигнатуру
        // `bloom_pipeline(vertex, fragment, ...)`.
        let fs_module = &bloom_module;
        let vs_module = &bloom_module;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<BloomUniform>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bloom params buffer"),
            contents: bytemuck::bytes_of(&BloomUniform::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: fs_module,
                entry_point: Some("fs_main"),
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

    /// Reallocate all size-dependent textures and re-record dependent
    /// pipelines after the output extent changed. Extents are clamped to >= 1.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        self.width = width;
        self.height = height;

        self.pbr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pbr render target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: self.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.pbr_texture_view = self
            .pbr_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.gbuffer = Self::create_gbuffer(device, width, height, self.sample_count);
        let (gbuffer_pipeline, gbuffer_bind_group_layout, gbuffer_bind_group) =
            Self::create_gbuffer_pipeline(
                device,
                &self.gbuffer,
                &self.camera_buffer,
                &self.per_object_buffer,
                &self.material_buffer,
                self.sample_count,
            );
        self.gbuffer_pipeline = gbuffer_pipeline;
        self.gbuffer_bind_group_layout = gbuffer_bind_group_layout;
        self.gbuffer_bind_group = gbuffer_bind_group;

        self.lighting_pass =
            Self::create_lighting_pass(device, &self.pbr_texture_view, self.sample_count);

        self.forward_pass = Self::create_forward_pass(
            device,
            &self.camera_buffer,
            &self.per_object_buffer,
            &self.material_buffer,
            &self.lighting_buffer,
            width,
            height,
            self.sample_count,
        );

        self.composite_pass = Self::create_composite_pass(device, self.format);
    }

    /// View of the final lit HDR image produced by the legacy-path lighting pass.
    pub fn pbr_view(&self) -> &wgpu::TextureView {
        &self.pbr_texture_view
    }

    /// Bytes allocated by the persistent textures of the legacy path
    /// (5 g-buffer MRTs + g-buffer depth + lighting target + forward color).
    pub fn texture_budget(&self) -> u64 {
        let bpp = crate::frame_plan::format_bytes_per_pixel;
        let w = self.width as u64;
        let h = self.height as u64;
        let s = self.sample_count as u64;
        let gbuffer = (bpp(wgpu::TextureFormat::Rgba8Unorm)
            + bpp(wgpu::TextureFormat::Rg16Float)
            + bpp(wgpu::TextureFormat::R32Uint)
            + bpp(wgpu::TextureFormat::Rg16Float)
            + bpp(wgpu::TextureFormat::Rgba16Float)
            + bpp(wgpu::TextureFormat::Depth32Float)) as u64
            * w
            * h
            * s;
        let pbr = bpp(self.format) as u64 * w * h * s;
        let forward = bpp(wgpu::TextureFormat::Rgba16Float) as u64 * w * h * s;
        gbuffer + pbr + forward
    }

    /// Upload the camera uniform: view-projection, its inverse (computed here)
    /// and eye position. Call once per frame before rendering.
    pub fn set_camera(&self, queue: &wgpu::Queue, view_proj: &[[f32; 4]; 4], camera_pos: [f32; 3]) {
        let inv_view_proj = glam::Mat4::from_cols_array_2d(view_proj)
            .inverse()
            .to_cols_array_2d();
        let uniform = CameraUniform {
            view_proj: *view_proj,
            inv_view_proj,
            camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Upload ambient RGB plus up to four directional lights given as
    /// `(direction, intensity, color)`; directions are normalized here and
    /// excess lights beyond four are dropped (shader-side limit).
    pub fn set_lights(
        &self,
        queue: &wgpu::Queue,
        ambient: [f32; 3],
        lights: &[([f32; 3], f32, [f32; 3])],
    ) {
        let count = lights.len().min(4);
        let mut gpu_lights = [GpuLight {
            direction: [0.0; 4],
            color: [0.0; 4],
        }; 4];
        for i in 0..count {
            let (dir, intensity, col) = lights[i];
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            let nd = if len > 0.0 {
                [dir[0] / len, dir[1] / len, dir[2] / len, 0.0]
            } else {
                [0.0, 0.0, 1.0, 0.0]
            };
            gpu_lights[i] = GpuLight {
                direction: nd,
                color: [col[0], col[1], col[2], intensity],
            };
        }
        let lighting = LightingUniform {
            ambient_color: [ambient[0], ambient[1], ambient[2], 1.0],
            lights: gpu_lights,
            light_count: count as u32,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&lighting));
    }

    /// Replace the GPU material table (truncated to the 64-entry capacity);
    /// instances reference entries by index.
    pub fn upload_materials(&self, queue: &wgpu::Queue, materials: &[OpenPBRMaterial]) {
        let count = materials.len().min(self.max_materials as usize);
        queue.write_buffer(
            &self.material_buffer,
            0,
            bytemuck::cast_slice(&materials[..count]),
        );
    }

    /// Convert and upload up to 256 instances (excess dropped) into the
    /// per-object buffer used by both gbuffer and forward passes.
    pub fn upload_instances(&self, queue: &wgpu::Queue, instances: &[InstanceData]) {
        let count = instances.len().min(self.max_objects as usize);
        let mut gpu_objects: Vec<PerObjectGpu> = Vec::with_capacity(count);
        for inst in instances.iter().take(count) {
            let model_arr: [[f32; 4]; 4] = inst.model_matrix.to_cols_array_2d();
            let normal_arr: [[f32; 4]; 4] = inst.normal_matrix.to_cols_array_2d();
            gpu_objects.push(PerObjectGpu {
                model: model_arr,
                normal_matrix: normal_arr,
                material_index: inst.material_index,
                _padding: [0; 3],
            });
        }
        queue.write_buffer(
            &self.per_object_buffer,
            0,
            bytemuck::cast_slice(&gpu_objects),
        );
    }

    /// Record the gbuffer pass: fills the five MRT targets + depth for
    /// `instance_count` uploaded instances of `mesh`.
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
                    view: g.normal,
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
                    view: g.material_params,
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
        rpass.set_bind_group(0, &self.gbuffer_bind_group, &[]);
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.num_indices, 0, 0..instance_count);
    }

    /// Record the deferred lighting pass: reconstructs surface data from the
    /// g-buffer in `g`, evaluates the OpenPBR BRDF and writes HDR color into `output`.
    pub fn render_lighting(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        g: &GbufferTargets<'_>,
        output: &wgpu::TextureView,
    ) {
        // The bind group is rebuilt per frame: gbuffer views come from the
        // render-plan pool (transient) or from persistent textures.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting bind group (frame)"),
            layout: &self.lighting_pass.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.lighting_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.material_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(g.albedo),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(g.normal),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(g.material_id),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(g.world_position),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(g.material_params),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(g.depth),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(&self.lighting_pass.sampler),
                },
            ],
        });

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
        rpass.draw(0..4, 0..1);
    }

    /// Record the forward pass: draws lit geometry into the HDR `output`
    /// layer, depth-testing against (and optionally clearing) `depth`.
    /// `clear_depth = true` when the forward pass runs standalone; `false`
    /// when it follows the gbuffer pass and must share its depth.
    pub fn render_forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        mesh: &Mesh,
        instance_count: u32,
        clear_depth: bool,
    ) {
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
        rpass.set_bind_group(0, &self.forward_pass.bind_group, &[]);
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.num_indices, 0, 0..instance_count);
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
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(inputs.hdr),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(inputs.hdr_fwd),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(inputs.bloom),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(
                        self.bloom_pass.params_buffer.slice(..).into(),
                    ),
                },
            ],
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
        rpass.draw(0..4, 0..1);
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
                hdr_fwd: &self.forward_pass.color_view,
                bloom: &self.pbr_texture_view,
                bloom_intensity: 0.0,
                // Legacy path always runs the hybrid mix.
                mode: 2,
            },
        );
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
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        self.bloom_pass.params_buffer.slice(..).into(),
                    ),
                },
            ],
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
        rpass.draw(0..4, 0..1);
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
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        self.bloom_pass.params_buffer.slice(..).into(),
                    ),
                },
            ],
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
        rpass.draw(0..4, 0..1);
    }
}
