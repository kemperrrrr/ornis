//! Shadow mapping: 2D-layer and cube-face view matrices, shadow target and pipeline creation, the shadow pre-pass, and the directional fit bounds.
use super::*;
/// Shadow-map layers (one per light slot).
pub const SHADOW_LAYERS: usize = 4;

/// Shadow-map resolution in pixels (square).
///
/// Depth is `Depth32Float`; 1024² × 4 layers ≈ 16 MiB, allocated once
/// at construction. Only lights with `shadow: ShadowCast::Enabled` render into a
/// layer; the rest of the array stays cleared and is never sampled
/// (`params.w = -1.0` skips the lookup branchlessly).
pub const SHADOW_SIZE: u32 = 1024;

/// Directional shadow ortho box: ±half-extent around the origin, light
/// eye at `SHADOW_DIR_DIST` along the to-light direction.
pub const SHADOW_ORTHO_HALF: f32 = 12.0;

const SHADOW_DIR_DIST: f32 = 30.0;

/// Near/far depth margin beyond the directional shadow box (m).
const SHADOW_DIR_DEPTH_MARGIN: f32 = 20.0;

/// Extra far-plane padding for fitted directional shadows (m).
const SHADOW_FIT_FAR_PAD: f32 = 10.0;

/// Absolute axis·Y above which the shadow look-at picks +X as up.
const SHADOW_UP_AXIS_DOT: f32 = 0.98;

/// Depth-bias pair for the shadow pre-pass (2D layers and cube faces
/// share it). The constant term is negligible on `Depth32Float`; the
/// slope term dominates on curved surfaces — large values erode small
/// occluder blobs in the map, small values risk acne (guarded by
/// `shadowed_directional_without_occluder_has_no_acne`).
const SHADOW_DEPTH_BIAS_CONSTANT: i32 = 2;

const SHADOW_DEPTH_BIAS_SLOPE: f32 = 1.0;

/// Point-light shadow cubes (one depth cube per slot) and resolution.
///
/// 512² × 6 faces × 2 cubes ≈ 12 MiB. Sampling is analytic: the
/// hardware picks the face from the fragment→light vector's major
/// axis, and the reference depth uses the same 90° perspective
/// formula as the face renders (near plane [`SHADOW_CUBE_NEAR`], far
/// plane = light range, so no new uniforms are needed).
pub const POINT_SHADOW_CUBES: usize = 2;

/// Face resolution of one point-shadow cube.
pub const SHADOW_CUBE_SIZE: u32 = 512;

/// Near plane shared by the cube-face renders and the analytic
/// sampling formula — change both together.
pub const SHADOW_CUBE_NEAR: f32 = 0.1;

/// Faces on a cube-map shadow (must match [`CUBE_FACES`]).
pub(super) const CUBE_FACE_COUNT: usize = 6;

/// Cube-face axes (direction from the light) with the ups that reproduce
/// the hardware cube-sampling frame (OpenGL/Metal convention: +X: (−z,−y),
/// −X: (+z,−y), +Y: (+x,+z), −Y: (+x,−z), +Z: (+x,−y), −Z: (−x,−y)).
/// The U direction matches by construction; the V match needs the
/// projection Y-mirror in [`point_cube_face_vp`] (rasterization puts
/// NDC y+1 at texture row 0, the sampler reads v=0 from the top).
/// Keep both together: correct depths at mirrored texels are what the
/// sampler then misses (all-lit point shadows).
const CUBE_FACES: [([f32; 3], [f32; 3]); CUBE_FACE_COUNT] = [
    ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
    ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
    ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
    ([0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
    ([0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
];

/// Pick an up vector non-parallel to the given shadow axis.
pub(super) fn shadow_up(axis: glam::Vec3) -> glam::Vec3 {
    if axis.y.abs() > SHADOW_UP_AXIS_DOT {
        glam::Vec3::X
    } else {
        glam::Vec3::Y
    }
}

/// Light-space clip matrix for a shadowed directional light: ortho box
/// ±[`SHADOW_ORTHO_HALF`] around the origin, eye on the light side.
/// Same `directx` depth convention as the main camera, so stored
/// depths compare directly in the evaluator.
pub(super) fn dir_shadow_vp(to_light: glam::Vec3) -> [[f32; 4]; 4] {
    let view = glam::camera::rh::view::look_at_mat4(
        to_light * SHADOW_DIR_DIST,
        glam::Vec3::ZERO,
        shadow_up(to_light),
    );
    let proj = glam::camera::rh::proj::directx::orthographic(
        -SHADOW_ORTHO_HALF,
        SHADOW_ORTHO_HALF,
        -SHADOW_ORTHO_HALF,
        SHADOW_ORTHO_HALF,
        SHADOW_DIR_DIST - SHADOW_DIR_DEPTH_MARGIN,
        SHADOW_DIR_DIST + SHADOW_DIR_DEPTH_MARGIN,
    );
    (proj * view).to_cols_array_2d()
}

/// Scene fit for one directional shadow map: `(center, half-extent)` from
/// a scene AABB (`min`/`max` corners, e.g. over instance translations and
/// custom-geometry points). The half-extent never shrinks below
/// [`SHADOW_ORTHO_HALF`], so small scenes keep the legacy box exactly;
/// non-finite inputs fall back to the origin default instead of poisoning
/// the projection.
pub fn shadow_fit_for_bounds(min: [f32; 3], max: [f32; 3]) -> ([f32; 3], f32) {
    let center = [
        (min[0] + max[0]) * HALF,
        (min[1] + max[1]) * HALF,
        (min[2] + max[2]) * HALF,
    ];
    let half = ((max[0] - min[0]) * HALF)
        .max((max[1] - min[1]) * HALF)
        .max((max[2] - min[2]) * HALF)
        .max(SHADOW_ORTHO_HALF);
    if center.iter().all(|v| v.is_finite()) && half.is_finite() {
        (center, half)
    } else {
        ([0.0, 0.0, 0.0], SHADOW_ORTHO_HALF)
    }
}

/// Light-space clip matrix for a shadowed directional light fitted to the
/// scene: ortho box ±`half` around `center`, eye at `half + 20` along the
/// to-light direction, depth range covering the box diameter plus margin.
/// Same `directx` depth convention as [`dir_shadow_vp`]; only used when a
/// scene fit was set via [`Renderer3D::set_shadow_bounds`].
pub(super) fn dir_shadow_vp_fitted(
    to_light: glam::Vec3,
    center: [f32; 3],
    half: f32,
) -> [[f32; 4]; 4] {
    let dist = half + SHADOW_DIR_DEPTH_MARGIN;
    let c = glam::Vec3::from_array(center);
    let view = glam::camera::rh::view::look_at_mat4(c + to_light * dist, c, shadow_up(to_light));
    let proj = glam::camera::rh::proj::directx::orthographic(
        -half,
        half,
        -half,
        half,
        1.0,
        dist + half + SHADOW_FIT_FAR_PAD,
    );
    (proj * view).to_cols_array_2d()
}

/// Light-space clip matrix for a shadowed spotlight: perspective cone
/// (`2 × outer_angle`, aspect 1) from the light position along the
/// emission axis, far plane at the light range.
pub(super) fn spot_shadow_vp(
    position: [f32; 3],
    axis: glam::Vec3,
    outer_angle_deg: f32,
    range: f32,
) -> [[f32; 4]; 4] {
    let eye = glam::Vec3::from_array(position);
    let view = glam::camera::rh::view::look_at_mat4(eye, eye + axis, shadow_up(axis));
    let proj = glam::camera::rh::proj::directx::perspective(
        outer_angle_deg.to_radians() * 2.0,
        1.0,
        HALF,
        range.max(1.0),
    );
    (proj * view).to_cols_array_2d()
}

/// Light-space clip matrix for one cube face of a shadowed point
/// light: 90° perspective (aspect 1) from the light position along
/// the face axis, far plane at the light range. Matches the analytic
/// sampling formula (`SHADOW_CUBE_NEAR`, `range.max(1.0)`).
///
/// The Y row of the projection is negated: rasterization maps NDC y+1
/// to texture row 0 (top), while the hardware cube-sampling frame
/// reads v=0 from the top with the GL axis convention (`CUBE_FACES`),
/// so an unmirrored render lands V-flipped versus the sampler (small
/// centered occluders vanish, large blobs only partly overlap). The
/// negation is a reflection — winding flips, so cube faces render
/// through the mirrored shadow pipeline (`front_face: Cw`).
pub(super) fn point_cube_face_vp(position: [f32; 3], range: f32, face: usize) -> [[f32; 4]; 4] {
    let (dir, up) = CUBE_FACES[face % CUBE_FACE_COUNT];
    let eye = glam::Vec3::from_array(position);
    let view = glam::camera::rh::view::look_at_mat4(
        eye,
        eye + glam::Vec3::from_array(dir),
        glam::Vec3::from_array(up),
    );
    let proj = glam::camera::rh::proj::directx::perspective(
        std::f32::consts::FRAC_PI_2,
        1.0,
        SHADOW_CUBE_NEAR,
        range.max(1.0),
    );
    let mut vp = (proj * view).to_cols_array_2d();
    // Mirror NDC y (whole row 1 — a single element would distort,
    // not reflect).
    for col in vp.iter_mut() {
        col[1] = -col[1];
    }
    vp
}

impl Renderer3D {
    /// Depth-only skinned pipeline for the shadow pre-pass: the same
    /// palette-blend vertex stage over the skinned bind-group layout, no
    /// fragment stage (varyings are discarded, like the classic shadow
    /// pipeline).
    ///
    /// `mirror_y` selects the cube-face variant (`front_face: Cw`), same
    /// convention as [`create_shadow_pipeline`](Self::create_shadow_pipeline).
    pub(super) fn create_skinned_shadow_pipeline(
        device: &wgpu::Device,
        skinned_bind_group_layout: &wgpu::BindGroupLayout,
        mirror_y: bool,
    ) -> wgpu::RenderPipeline {
        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinned shadow vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(wgsl_vertex_source_skinned())),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinned shadow pipeline layout"),
            bind_group_layouts: &[Some(skinned_bind_group_layout)],
            immediate_size: 0,
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skinned shadow pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(skinned_entry_point()),
                buffers: &[Some(SkinnedVertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            // Depth-only: no color targets, varyings are discarded.
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: if mirror_y {
                    wgpu::FrontFace::Cw
                } else {
                    wgpu::FrontFace::Ccw
                },
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
                bias: wgpu::DepthBiasState {
                    constant: SHADOW_DEPTH_BIAS_CONSTANT,
                    slope_scale: SHADOW_DEPTH_BIAS_SLOPE,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }

    /// Allocate the shadow-map array, per-layer views, the sampling
    /// array view, VP uniform buffers and the comparison sampler.
    #[allow(clippy::type_complexity)]
    pub(super) fn create_shadow_targets(
        device: &wgpu::Device,
    ) -> (
        wgpu::Texture,
        [wgpu::TextureView; SHADOW_LAYERS],
        wgpu::TextureView,
        [wgpu::Buffer; SHADOW_LAYERS],
        wgpu::Sampler,
    ) {
        let shadow_maps = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow maps"),
            size: wgpu::Extent3d {
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
                depth_or_array_layers: SHADOW_LAYERS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let shadow_views = std::array::from_fn(|layer| {
            shadow_maps.create_view(&wgpu::TextureViewDescriptor {
                label: Some("shadow map layer"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });
        let shadow_array_view = shadow_maps.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow map array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let shadow_vp_buffers = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shadow VP buffer"),
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow comparison sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..crate::flags::SamplerKind::LinearClamp.descriptor()
        });
        (
            shadow_maps,
            shadow_views,
            shadow_array_view,
            shadow_vp_buffers,
            shadow_sampler,
        )
    }

    /// Depth-only pipeline for the shadow pre-pass. It reuses the
    /// gbuffer vertex shader (and its bind group layout): each layer
    /// binds its light-space VP buffer into the `camera` slot, so no
    /// new shader or layout is needed. A depth bias (constant + slope)
    /// fights acne; the evaluator adds a small reference bias on top.
    ///
    /// `mirror_y` selects the cube-face variant (`front_face: Cw`):
    /// cube VPs mirror NDC y (see [`point_cube_face_vp`]), which flips
    /// winding, so faces must cull the mirrored side.
    pub(super) fn create_shadow_pipeline(
        device: &wgpu::Device,
        gbuffer_bind_group_layout: &wgpu::BindGroupLayout,
        mirror_y: bool,
    ) -> wgpu::RenderPipeline {
        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::gbuffer_vertex())),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow pipeline layout"),
            bind_group_layouts: &[Some(gbuffer_bind_group_layout)],
            immediate_size: 0,
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(shaders::gbuffer_generated::vs_main::entry_point()),
                buffers: &[Some(Vertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            // Depth-only: no color targets, varyings are discarded.
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: if mirror_y {
                    wgpu::FrontFace::Cw
                } else {
                    wgpu::FrontFace::Ccw
                },
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
                bias: wgpu::DepthBiasState {
                    constant: SHADOW_DEPTH_BIAS_CONSTANT,
                    slope_scale: SHADOW_DEPTH_BIAS_SLOPE,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }

    /// Allocate the point-shadow cube array, per-face views, the
    /// sampling cube-array view and per-face VP uniform buffers.
    /// Rendering reuses the depth-only shadow pipeline (same gbuffer
    /// vertex layout); only the bound VP buffer and target view change
    /// per face.
    #[allow(clippy::type_complexity)]
    pub(super) fn create_shadow_cube_targets(
        device: &wgpu::Device,
    ) -> (
        wgpu::Texture,
        [wgpu::TextureView; POINT_SHADOW_CUBES * CUBE_FACE_COUNT],
        wgpu::TextureView,
        [wgpu::Buffer; POINT_SHADOW_CUBES * CUBE_FACE_COUNT],
    ) {
        let shadow_cube_maps = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("point shadow cubes"),
            size: wgpu::Extent3d {
                width: SHADOW_CUBE_SIZE,
                height: SHADOW_CUBE_SIZE,
                depth_or_array_layers: (POINT_SHADOW_CUBES * CUBE_FACE_COUNT) as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let shadow_cube_views = std::array::from_fn(|layer| {
            shadow_cube_maps.create_view(&wgpu::TextureViewDescriptor {
                label: Some("point shadow cube face"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });
        let shadow_cube_array_view = shadow_cube_maps.create_view(&wgpu::TextureViewDescriptor {
            label: Some("point shadow cube array"),
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        });
        let shadow_cube_vp_buffers = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("point shadow face VP buffer"),
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        (
            shadow_cube_maps,
            shadow_cube_views,
            shadow_cube_array_view,
            shadow_cube_vp_buffers,
        )
    }

    /// Render depth pre-passes for the `0..shadow_count` layers assigned
    /// by [`set_lights`](Self::set_lights); a no-op without shadowed
    /// lights. Runs before lighting/forward in every frame path
    /// (legacy `render_scene` and the `LightingPass` plan pass call it
    /// explicitly).
    pub fn render_shadows(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &Mesh,
        instance_count: u32,
    ) {
        self.render_shadows_with_custom(device, encoder, mesh, instance_count, &[]);
    }

    /// Depth pre-pass with per-entity custom draws: the same layers and
    /// cube faces [`render_shadows`](Self::render_shadows) covers, one
    /// render pass per layer/face cleared once.
    ///
    /// The shared `mesh` batch draws first, then each entry of `customs`
    /// in order: [`SkinnedDraw::Cpu`] entries (from
    /// [`CustomGbufferDraw::shadow`]) draw through the classic depth
    /// pipeline from their per-object slot, [`SkinnedDraw::Gpu`] entries
    /// through the skinned depth pipeline with their palette slot — never
    /// bind-pose depth for a GPU claim. A stale handle records no commands
    /// for that entry. With empty `customs` the command stream matches
    /// [`render_shadows`](Self::render_shadows) exactly. A no-op without
    /// shadowed lights, or when both the batch and `customs` are empty.
    pub fn render_shadows_with_custom(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &Mesh,
        instance_count: u32,
        customs: &[CustomGbufferDraw<'_>],
    ) {
        if instance_count == 0 && customs.is_empty() {
            return;
        }
        let count = self
            .shadow_count
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(SHADOW_LAYERS as u32);
        let cubes = self
            .point_shadow_count
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(POINT_SHADOW_CUBES as u32);
        if count == 0 && cubes == 0 {
            return;
        }
        for layer in 0..count as usize {
            self.render_shadow_view_with_custom(
                device,
                encoder,
                mesh,
                instance_count,
                customs,
                &self.shadow_pipeline,
                &self.skinned_shadow_pipeline,
                &self.shadow_vp_buffers[layer],
                &self.shadow_views[layer],
            );
        }
        for cube in 0..cubes as usize {
            for face in 0..CUBE_FACE_COUNT {
                let idx = cube * CUBE_FACE_COUNT + face;
                self.render_shadow_view_with_custom(
                    device,
                    encoder,
                    mesh,
                    instance_count,
                    customs,
                    &self.shadow_cube_pipeline,
                    &self.skinned_shadow_cube_pipeline,
                    &self.shadow_cube_vp_buffers[idx],
                    &self.shadow_cube_views[idx],
                );
            }
        }
    }

    /// One depth-only draw set into a shadow view: the shared batch plus
    /// the custom entries through the classic pipeline, then the
    /// GPU-skinned entries through `skinned_pipeline` — a single pass with
    /// one clear. Shared by 2D layers and cube faces (same layouts,
    /// mirrored pipeline for faces).
    #[allow(clippy::too_many_arguments)]
    fn render_shadow_view_with_custom(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &Mesh,
        instance_count: u32,
        customs: &[CustomGbufferDraw<'_>],
        classic_pipeline: &wgpu::RenderPipeline,
        skinned_pipeline: &wgpu::RenderPipeline,
        vp_buffer: &wgpu::Buffer,
        view: &wgpu::TextureView,
    ) {
        let per_object = read_lock(&self.per_object_buffer);
        let material = read_lock(&self.material_buffer);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow bind group"),
            layout: &self.gbuffer_bind_group_layout,
            entries: &shaders::bind_group_entries(
                &shaders::gbuffer_generated::GBUFFER_RESOURCES,
                |r| match r.name {
                    "camera" => Some(vp_buffer.as_entire_binding()),
                    "per_objects" => Some(per_object.as_entire_binding()),
                    "materials" => Some(material.as_entire_binding()),
                    _ => None,
                },
            )
            .unwrap_or_default(),
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view,
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
        pass.set_pipeline(classic_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        if instance_count > 0 {
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.num_indices, 0, 0..instance_count);
        }
        for custom in customs {
            match custom.shadow {
                SkinnedDraw::Cpu => {
                    pass.set_vertex_buffer(0, custom.mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(
                        custom.mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    pass.draw_indexed(
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
                    let skinned_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("skinned shadow bind group"),
                        layout: &self.skinned_bind_group_layout,
                        entries: &shaders::bind_group_entries(&GBUFFER_SKINNED_RESOURCES, |r| {
                            match r.name {
                                "camera" => Some(vp_buffer.as_entire_binding()),
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
                    pass.set_pipeline(skinned_pipeline);
                    pass.set_bind_group(0, &skinned_bind_group, &[]);
                    pass.set_vertex_buffer(0, custom.mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(
                        custom.mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    pass.draw_indexed(
                        0..custom.mesh.num_indices,
                        0,
                        custom.instance_slot..custom.instance_slot + 1,
                    );
                    pass.set_pipeline(classic_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                }
            }
        }
    }

    /// Render skinned depth pre-passes for one skinned entry: the same
    /// layers and cube faces [`render_shadows`](Self::render_shadows)
    /// covers, drawn with the skinned depth pipelines and the entry's
    /// palette slot — skinned depth, never bind-pose depth.
    ///
    /// The entry instance must already sit in per-object slot 0 (via
    /// [`upload_instances`](Self::upload_instances) or
    /// [`render_skinned_entry`](Self::render_skinned_entry)); a stale
    /// handle records no commands. A no-op without shadowed lights.
    pub fn render_skinned_shadows(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &Mesh,
        handle: PaletteHandle,
    ) {
        if handle.index()
            >= self
                .palette_count
                .load(std::sync::atomic::Ordering::Relaxed) as usize
        {
            return;
        }
        let count = self
            .shadow_count
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(SHADOW_LAYERS as u32);
        let cubes = self
            .point_shadow_count
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(POINT_SHADOW_CUBES as u32);
        if count == 0 && cubes == 0 {
            return;
        }
        let per_object = read_lock(&self.per_object_buffer);
        let material = read_lock(&self.material_buffer);
        let palette = read_lock(&self.palette_buffer);
        for layer in 0..count as usize {
            self.render_skinned_shadow_layer(
                device,
                encoder,
                mesh,
                handle,
                &self.skinned_shadow_pipeline,
                &self.shadow_vp_buffers[layer],
                &self.shadow_views[layer],
                &per_object,
                &material,
                &palette,
            );
        }
        for cube in 0..cubes as usize {
            for face in 0..CUBE_FACE_COUNT {
                let idx = cube * CUBE_FACE_COUNT + face;
                self.render_skinned_shadow_layer(
                    device,
                    encoder,
                    mesh,
                    handle,
                    &self.skinned_shadow_cube_pipeline,
                    &self.shadow_cube_vp_buffers[idx],
                    &self.shadow_cube_views[idx],
                    &per_object,
                    &material,
                    &palette,
                );
            }
        }
    }

    /// One skinned depth-only draw into a shadow view: the light-space VP
    /// goes into the `camera` slot and the entry palette into binding 3.
    /// Shared by 2D layers and cube faces (same layout, mirrored pipeline
    /// for faces).
    #[allow(clippy::too_many_arguments)]
    fn render_skinned_shadow_layer(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &Mesh,
        handle: PaletteHandle,
        pipeline: &wgpu::RenderPipeline,
        vp_buffer: &wgpu::Buffer,
        view: &wgpu::TextureView,
        per_object: &wgpu::Buffer,
        material: &wgpu::Buffer,
        palette: &wgpu::Buffer,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinned shadow bind group"),
            layout: &self.skinned_bind_group_layout,
            entries: &shaders::bind_group_entries(&GBUFFER_SKINNED_RESOURCES, |r| match r.name {
                "camera" => Some(vp_buffer.as_entire_binding()),
                "per_objects" => Some(per_object.as_entire_binding()),
                "materials" => Some(material.as_entire_binding()),
                "palette" => Some(wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: palette,
                    offset: handle.byte_offset(),
                    size: std::num::NonZeroU64::new(PALETTE_BYTE_SIZE as u64),
                })),
                _ => None,
            })
            .unwrap_or_default(),
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("skinned shadow pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view,
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
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
    }

    /// Upload the camera uniform: view-projection, its inverse (computed here)
    /// and eye position. Call once per frame before rendering.
    /// Test-only readback of one 2D shadow layer as row-major depths
    /// (row 0 first): regression pin for the raster↔sampler V convention.
    #[cfg(test)]
    pub(crate) fn read_shadow_layer_for_tests(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layer: u32,
    ) -> Vec<f32> {
        /// Bytes per Depth32Float texel.
        const DEPTH32_BYTES: u32 = 4;
        let size = SHADOW_SIZE;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test shadow layer readback"),
            size: (size * size * DEPTH32_BYTES) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test shadow layer readback"),
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.shadow_maps,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size * DEPTH32_BYTES),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(enc.finish()));
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
            return Vec::new();
        }
        let Ok(Ok(())) = rx.recv() else {
            return Vec::new();
        };
        let Ok(data) = slice.get_mapped_range() else {
            return Vec::new();
        };
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buf.unmap();
        out
    }

    /// Fit the directional shadow frustum to a scene AABB (`min`/`max`
    /// corners over instance translations and custom-geometry points).
    /// The ortho half-extent grows from the ±[`SHADOW_ORTHO_HALF`]
    /// default to cover the box (see [`shadow_fit_for_bounds`]); pass the
    /// scene bounds once per scene, not per frame.
    pub fn set_shadow_bounds(&self, min: [f32; 3], max: [f32; 3]) {
        *write_lock(&self.shadow_fit) = Some(shadow_fit_for_bounds(min, max));
    }

    /// Drop the scene fit and return to the legacy ±[`SHADOW_ORTHO_HALF`]
    /// box around the origin.
    pub fn clear_shadow_bounds(&self) {
        *write_lock(&self.shadow_fit) = None;
    }

    /// Current directional-shadow ortho half-extent: the fitted value
    /// after [`set_shadow_bounds`](Self::set_shadow_bounds), else the
    /// ±[`SHADOW_ORTHO_HALF`] default.
    pub fn shadow_half_extent(&self) -> f32 {
        read_lock(&self.shadow_fit).map_or(SHADOW_ORTHO_HALF, |(_, half)| half)
    }
}
