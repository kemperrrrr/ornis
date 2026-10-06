//! Renderer construction: `new` variants plus the legacy core resources (uniform buffers, PBR pipeline, palette store, render target) owned by `Renderer3D`.
use super::*;
impl Renderer3D {
    /// Build every pipeline/target for `surface_config`'s format and extent.
    ///
    /// `sample_count` is [`normalize_sample_count`]ed to `{1, 4}` on entry:
    /// pass [`MSAA_SAMPLE_COUNT`] for the native 4x path (after gating it
    /// through [`negotiate_sample_count` against the adapter — `new` only
    /// sees the device, so it cannot capability-gate itself), `1` (or
    /// anything else) for single-sample. The fullscreen lighting output
    /// stays single-sample in all modes (a fullscreen triangle has no edges
    /// for MSAA to smooth; resolving it would be a 4x-cost no-op), while the
    /// g-buffer/forward geometry layers go multisampled with resolve at 4x.
    ///
    /// Capacity starts at 256 instances / 64 materials and grows on
    /// demand: [`upload_instances`](Self::upload_instances) and
    /// [`upload_materials`](Self::upload_materials) reallocate (and rebind)
    /// their buffers when a frame needs more, so scenes are never silently
    /// truncated. Zero-sized extents are clamped to 1 pixel.
    pub fn new(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
    ) -> Self {
        let sample_count = normalize_sample_count(sample_count);
        let max_objects = INITIAL_MAX_OBJECTS;
        let max_materials = INITIAL_MAX_MATERIALS;
        let format = surface_config.format;
        let width = surface_config.width.max(1);
        let height = surface_config.height.max(1);

        let buffers = Self::create_core_buffers(device, max_objects, max_materials);
        let (shadow_maps, shadow_views, shadow_array_view, shadow_vp_buffers, shadow_sampler) =
            Self::create_shadow_targets(device);
        let (shadow_cube_maps, shadow_cube_views, shadow_cube_array_view, shadow_cube_vp_buffers) =
            Self::create_shadow_cube_targets(device);
        let ibl = crate::ibl::black_targets(device);
        let ibl_staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ibl staging"),
            contents: &crate::ibl::initial_staging_bytes(),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let (bind_group_layout, bind_group) = Self::create_pbr_bind_group(
            device,
            &buffers,
            &shadow_array_view,
            &shadow_sampler,
            &shadow_cube_array_view,
            &ibl,
        );
        let pipeline =
            Self::create_pbr_pipeline(device, surface_config, sample_count, &bind_group_layout);
        // The deferred lighting output is a fullscreen triangle: MSAA would
        // resolve four identical samples per pixel at 4x memory cost, so it
        // stays single-sample in all modes (the lighting pipeline's
        // `multisample.count` is 1 to match).
        let (pbr_texture, pbr_texture_view) = Self::create_render_target(
            device,
            width,
            height,
            DEFERRED_HDR_FORMAT,
            SINGLE_SAMPLE_COUNT,
        );

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
        let shadow_pipeline =
            Self::create_shadow_pipeline(device, &gbuffer_bind_group_layout, false);
        let shadow_cube_pipeline =
            Self::create_shadow_pipeline(device, &gbuffer_bind_group_layout, true);
        let palette_buffer = Self::create_palette_buffer(device, 1);
        let (skinned_pipeline, skinned_bind_group_layout) = Self::create_skinned_pipeline(
            device,
            &buffers.camera,
            &buffers.per_object,
            &buffers.material,
            &palette_buffer,
            sample_count,
        );
        let skinned_shadow_pipeline =
            Self::create_skinned_shadow_pipeline(device, &skinned_bind_group_layout, false);
        let skinned_shadow_cube_pipeline =
            Self::create_skinned_shadow_pipeline(device, &skinned_bind_group_layout, true);
        let lighting_pass = Self::create_lighting_pass(device, &pbr_texture_view, sample_count);
        let forward_pass = Self::create_forward_pass(
            device,
            &buffers.camera,
            &buffers.per_object,
            &buffers.material,
            &buffers.lighting,
            &shadow_array_view,
            &shadow_sampler,
            &shadow_cube_array_view,
            &ibl,
            width,
            height,
            sample_count,
            TransparencyOptions::default(),
        );
        let composite_pass = Self::create_composite_pass(device, format);
        let bloom_pass = Self::create_bloom_pass(device);
        let fog = Self::create_fog_pass(device, format, sample_count);
        let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("composite sampler"),
            ..crate::flags::SamplerKind::LinearClamp.descriptor()
        });

        Self {
            camera_buffer: buffers.camera,
            per_object_buffer: std::sync::RwLock::new(buffers.per_object),
            material_buffer: std::sync::RwLock::new(buffers.material),
            lighting_buffer: buffers.lighting,
            _bind_group_layout: bind_group_layout,
            _bind_group: bind_group,
            _pipeline: pipeline,
            pbr_texture,
            pbr_texture_view,
            sample_count,
            exposure: 1.0,
            transparency: TransparencyOptions::default(),
            max_objects: std::sync::atomic::AtomicU32::new(max_objects),
            max_materials: std::sync::atomic::AtomicU32::new(max_materials),
            format,
            width,
            height,
            gbuffer,
            gbuffer_pipeline,
            gbuffer_bind_group_layout,
            gbuffer_bind_group: std::sync::RwLock::new(gbuffer_bind_group),
            skinned_pipeline,
            skinned_bind_group_layout,
            skinned_shadow_pipeline,
            skinned_shadow_cube_pipeline,
            palette_buffer: std::sync::RwLock::new(palette_buffer),
            max_palettes: std::sync::atomic::AtomicU32::new(1),
            palette_count: std::sync::atomic::AtomicU32::new(0),
            lighting_pass,
            forward_pass,
            textured_forward: None,
            composite_pass,
            composite_sampler,
            bloom_pass,
            fog,
            shadow_maps,
            shadow_views,
            shadow_array_view,
            shadow_vp_buffers,
            shadow_pipeline,
            shadow_sampler,
            shadow_count: std::sync::atomic::AtomicU32::new(0),
            shadow_fit: std::sync::RwLock::new(None),
            last_light_stats: std::sync::RwLock::new(LightUploadStats {
                uploaded: 0,
                dropped_lights: 0,
                dropped_shadows: 0,
            }),
            shadow_cube_maps,
            shadow_cube_views,
            shadow_cube_array_view,
            shadow_cube_vp_buffers,
            shadow_cube_pipeline,
            point_shadow_count: std::sync::atomic::AtomicU32::new(0),
            ibl,
            ibl_staging,
            ibl_uploaded: std::sync::atomic::AtomicBool::new(false),
            ibl_weight_bits: std::sync::atomic::AtomicU32::new(0),
            ibl_weight_explicit: std::sync::atomic::AtomicBool::new(false),
            ibl_max_mip_bits: std::sync::atomic::AtomicU32::new(0),
            shading_debug: std::sync::atomic::AtomicU32::new(ShadingDebug::Beauty as u32),
        }
    }

    /// Like [`new`](Self::new) with an explicit forward-layer
    /// transparency mode: rebuilds only the forward pipeline with
    /// [`forward_blend_state`] for `transparency`. The default
    /// ([`BlendMode::Opaque`]) builds the same `REPLACE` pipeline as
    /// [`new`](Self::new), so the default frame is unchanged. With
    /// [`BlendMode::Transparent`], submit instances back-to-front
    /// ([`sort_by_depth`] or [`crate::extraction::sort_by_depth`]).
    pub fn new_with_transparency(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
        transparency: TransparencyOptions,
    ) -> Self {
        let mut this = Self::new(device, surface_config, sample_count);
        this.transparency = transparency;
        {
            let per_object = read_lock(&this.per_object_buffer);
            let material = read_lock(&this.material_buffer);
            this.forward_pass = Self::create_forward_pass(
                device,
                &this.camera_buffer,
                &per_object,
                &material,
                &this.lighting_buffer,
                &this.shadow_array_view,
                &this.shadow_sampler,
                &this.shadow_cube_array_view,
                &this.ibl,
                this.width,
                this.height,
                this.sample_count,
                transparency,
            );
        }
        this
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
            ambient_color: [
                DEFAULT_AMBIENT_RGB[0],
                DEFAULT_AMBIENT_RGB[1],
                DEFAULT_AMBIENT_RGB[2],
                1.0,
            ],
            lights: [GpuLight {
                kind: [LIGHT_KIND_DIRECTIONAL, 0.0, 0.0, 0.0],
                direction: [0.0; VEC4_COMPONENTS],
                position: [0.0; VEC4_COMPONENTS],
                color: [0.0; VEC4_COMPONENTS],
                params: [0.0, 0.0, 0.0, -1.0],
                shadow_vp: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }; MAX_LIGHTS],
            light_count: 0,
            ibl_weight: 0.0,
            ibl_max_mip: 0.0,
            debug_view: 0,
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
        shadow_array_view: &wgpu::TextureView,
        shadow_sampler: &wgpu::Sampler,
        shadow_cube_array_view: &wgpu::TextureView,
        ibl: &crate::ibl::IblTargets,
    ) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        // Layout entries come from the pass resource table.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = shaders::pbr_generated::PBR_RESOURCES
            .iter()
            .map(|r| shaders::bgl_entry(r, false))
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbr bind group layout"),
            entries: &bgl_entries,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pbr bind group"),
            layout: &bind_group_layout,
            // Binding numbers come from the table; only the name →
            // buffer mapping lives here.
            entries: &shaders::bind_group_entries(&shaders::pbr_generated::PBR_RESOURCES, |r| {
                match r.name {
                    "camera" => Some(buffers.camera.as_entire_binding()),
                    "per_objects" => Some(buffers.per_object.as_entire_binding()),
                    "materials" => Some(buffers.material.as_entire_binding()),
                    "lighting" => Some(buffers.lighting.as_entire_binding()),
                    "shadow_tex" => Some(wgpu::BindingResource::TextureView(shadow_array_view)),
                    "shadow_sampler" => Some(wgpu::BindingResource::Sampler(shadow_sampler)),
                    "shadow_cube_tex" => {
                        Some(wgpu::BindingResource::TextureView(shadow_cube_array_view))
                    }
                    "prefilter_cube" | "irradiance_cube" | "brdf_lut" | "ibl_sampler" => {
                        ibl.binding(r.name)
                    }
                    _ => None,
                }
            })
            .unwrap_or_default(),
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
                entry_point: Some(shaders::gbuffer_generated::vs_main::entry_point()),
                buffers: &[Some(Vertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some(shaders::pbr_generated::fs_main::entry_point()),
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

    /// Allocate the joint-palette storage buffer: `slots` packed
    /// [`PALETTE_BYTE_SIZE`] slots, grown by
    /// [`upload_skin_palettes`](Self::upload_skin_palettes). Starts
    /// zeroed; unwritten slots are never indexed (joint indices are
    /// validated `< joint count` before staging).
    fn create_palette_buffer(device: &wgpu::Device, slots: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skin palette buffer"),
            size: (PALETTE_BYTE_SIZE as u64) * (slots.max(1) as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
}
