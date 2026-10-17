//! Shadow mapping: 2D-layer and cube-face view matrices, shadow target and pipeline creation, the shadow pre-pass, and the directional fit bounds.
use super::*;
/// Shadow-map layers (one per light slot).
pub const SHADOW_LAYERS: usize = 4;

/// Cascades per directional light for CSM (#121).
pub const CSM_CASCADES: usize = 4;

/// Default practical-split blend: 0 = uniform, 1 = logarithmic.
pub const CSM_SPLIT_LAMBDA: f32 = 0.5;

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

/// Quantization of a cascade sphere diameter before it is rounded up (m).
///
/// `2⁻⁶`. Unprojection noise from a camera yaw stays under half a quantum,
/// so the ceiled side — and the texel size — stays bit-identical under a
/// pure rotation. Bevy measures the same length in camera space for the
/// same reason (`calculate_cascade`).
const CSM_DIAMETER_QUANTUM: f32 = 0.015625;

/// Texels of slack added around the ceiled diameter (two on each side).
///
/// Flooring the sphere center onto the texel grid moves the box by less
/// than one texel; two texels of slack on each side still contain the slice.
const CSM_SNAP_PAD_TEXELS: f32 = 4.0;

/// Pull the cascade near plane this far toward the light past the closest
/// caster or frustum corner (m).
const CSM_NEAR_PAD: f32 = 1.0;

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

/// Uniform split at fraction `t` of the camera range.
fn uniform_split(near: f32, far: f32, t: f32) -> f32 {
    near + (far - near) * t
}

/// Logarithmic split at fraction `t`: geometric progression from `near` to `far`.
fn log_split(near: f32, far: f32, t: f32) -> f32 {
    near * (far / near).powf(t)
}

/// Cascade boundaries for a directional light (practical split blend, #121).
///
/// Returns `CSM_CASCADES + 1` distances from `near` to `far`; entries beyond
/// `num_cascades` repeat `far` so the layout stays DSL-friendly (fixed
/// `array<f32>`). `lambda` blends logarithmic (`1`) and uniform (`0`);
/// degenerate ranges fall back to `near`/`far` without NaN.
pub(super) fn cascade_splits(
    near: f32,
    far: f32,
    num_cascades: usize,
    lambda: f32,
) -> [f32; CSM_CASCADES + 1] {
    // facade: validate inputs, blend the per-fraction helpers below.
    let count = num_cascades.clamp(1, CSM_CASCADES);
    let clamped = lambda.clamp(0.0, 1.0);
    if !(near.is_finite() && far.is_finite() && near > 0.0 && far > near) {
        return fallback_splits(near, far);
    }
    let mut splits = [far; CSM_CASCADES + 1];
    splits[0] = near;
    for (i, slot) in splits.iter_mut().enumerate().take(count).skip(1) {
        let t = i as f32 / count as f32;
        *slot = clamped * log_split(near, far, t) + (1.0 - clamped) * uniform_split(near, far, t);
    }
    splits
}

/// Degenerate-range fallback: first entry `near`, the rest `far`.
fn fallback_splits(near: f32, far: f32) -> [f32; CSM_CASCADES + 1] {
    let mut splits = [far; CSM_CASCADES + 1];
    splits[0] = near;
    splits
}

/// Bounding sphere of the slice corners, or `None` for non-finite input.
fn slice_bounding_sphere(corners: &[[f32; 3]; 8]) -> Option<(glam::Vec3, f32)> {
    let mut center = glam::Vec3::ZERO;
    for corner in corners {
        let p = glam::Vec3::from_array(*corner);
        if !p.is_finite() {
            return None;
        }
        center += p;
    }
    let center = center / corners.len() as f32;
    let radius = bounding_radius(center, corners);
    (radius > 0.0 && radius.is_finite()).then_some((center, radius))
}

/// Max distance from `center` to the corners.
fn bounding_radius(center: glam::Vec3, corners: &[[f32; 3]; 8]) -> f32 {
    corners
        .iter()
        .map(|c| center.distance(glam::Vec3::from_array(*c)))
        .fold(0.0, f32::max)
}

/// Light-space AABB of world-space frustum-slice corners under `view`.
fn light_space_aabb(view: glam::Mat4, corners: &[[f32; 3]; 8]) -> (glam::Vec3, glam::Vec3) {
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for corner in corners {
        let p = view.transform_point3(glam::Vec3::from_array(*corner));
        min = min.min(p);
        max = max.max(p);
    }
    (min, max)
}

/// Snap one coordinate down onto the texel grid.
fn snap_down_to_texel(value: f32, texel: f32) -> f32 {
    (value / texel).floor() * texel
}

/// Rotation-only light view anchored at the world origin.
///
/// Directional lights have no position. A view built from the moving
/// slice center drags the texel grid with the camera, and snapping in
/// that frame does not stabilize anything (Bevy keeps `world_from_light`
/// orthonormal with zero translation for the same reason).
fn csm_light_view(to_light: glam::Vec3) -> glam::Mat4 {
    glam::camera::rh::view::look_at_mat4(glam::Vec3::ZERO, -to_light, shadow_up(to_light))
}

/// Map side in world units: ceiled diameter of the slice sphere, plus
/// [`CSM_SNAP_PAD_TEXELS`] of slack. `None` when the slice or the
/// resolution is degenerate.
fn cascade_map_side(corners: &[[f32; 3]; 8], resolution: u32) -> Option<f32> {
    let (_, radius) = slice_bounding_sphere(corners)?;
    if resolution == 0 || !radius.is_finite() {
        return None;
    }
    let diameter = (radius * 2.0 / CSM_DIAMETER_QUANTUM).round() * CSM_DIAMETER_QUANTUM;
    let raw = diameter.ceil();
    if !(raw > 0.0 && raw.is_finite()) {
        return None;
    }
    let side = (raw + CSM_SNAP_PAD_TEXELS * raw / resolution as f32).ceil();
    (side > 0.0 && side.is_finite()).then_some(side)
}

/// Which corner of an AABB each slot picks: `0` is min, `1` is max, per axis.
const AABB_CORNER_SIGN: [[u8; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

/// Eight corners of an axis-aligned box.
fn aabb_corners(min: [f32; 3], max: [f32; 3]) -> [[f32; 3]; 8] {
    AABB_CORNER_SIGN.map(|sign| {
        [
            if sign[0] == 0 { min[0] } else { max[0] },
            if sign[1] == 0 { min[1] } else { max[1] },
            if sign[2] == 0 { min[2] } else { max[2] },
        ]
    })
}

/// Widen a light-space z range so it also covers `corners`.
fn include_z(view: glam::Mat4, corners: &[[f32; 3]; 8], z: (f32, f32)) -> (f32, f32) {
    let (min, max) = light_space_aabb(view, corners);
    (z.0.min(min.z), z.1.max(max.z))
}

/// Cascade view-projection for one camera-frustum slice (CSM, #121).
///
/// The light view is rotation-only ([`csm_light_view`]). The map side is
/// the ceiled diameter of the slice's bounding sphere, and the sphere
/// center is floored onto that texel grid — the stable-CSM fit from
/// Bevy's `calculate_cascade`. Depth covers `depth_world` (the whole
/// camera frustum) and, when `casters` is set, the scene AABB as well,
/// so a caster outside the slice or behind the camera still reaches the
/// map. Degenerate input falls back to [`dir_shadow_vp`].
pub(super) fn cascade_shadow_vp(
    to_light: glam::Vec3,
    corners_world: [[f32; 3]; 8],
    depth_world: [[f32; 3]; 8],
    casters: Option<([f32; 3], [f32; 3])>,
    resolution: u32,
) -> [[f32; 4]; 4] {
    // facade: sphere side, origin-anchored view, snap, depth union.
    let Some((center, _)) = slice_bounding_sphere(&corners_world) else {
        return dir_shadow_vp(to_light);
    };
    let Some(side) = cascade_map_side(&corners_world, resolution) else {
        return dir_shadow_vp(to_light);
    };
    let view = csm_light_view(to_light);
    let texel = side / resolution as f32;
    let center_ls = view.transform_point3(center);
    let snapped_x = snap_down_to_texel(center_ls.x, texel);
    let snapped_y = snap_down_to_texel(center_ls.y, texel);
    let half = side * HALF;
    let mut z = {
        let (min, max) = light_space_aabb(view, &depth_world);
        (min.z, max.z)
    };
    if let Some((min, max)) = casters {
        z = include_z(view, &aabb_corners(min, max), z);
    }
    let proj = glam::camera::rh::proj::directx::orthographic(
        snapped_x - half,
        snapped_x + half,
        snapped_y - half,
        snapped_y + half,
        -z.1 - CSM_NEAR_PAD,
        -z.0 + SHADOW_FIT_FAR_PAD,
    );
    (proj * view).to_cols_array_2d()
}

/// Unproject one NDC point to world space, or `None` at/behind the camera.
fn unproject_ndc(inv_view_proj: glam::Mat4, ndc: [f32; 3]) -> Option<[f32; 3]> {
    let world = inv_view_proj * glam::Vec4::new(ndc[0], ndc[1], ndc[2], 1.0);
    if world.w <= 0.0 || !world.is_finite() {
        return None;
    }
    Some((world.truncate() / world.w).to_array())
}

/// Components of one frustum-slice corner (`x`, `y`, `z`).
const CORNER_LANES: usize = 3;

/// Corners of a frustum slice: the near square and the far square.
const FRUSTUM_CORNERS: usize = 8;

/// World-space corners of a camera-frustum slice (CSM, #121).
///
/// Unprojects the NDC square at the view depths `split_near`/`split_far`
/// through `inv_view_proj` (directx depth convention, like the main
/// camera). Split bounds clamp into `[near, far]`; degenerate ranges
/// return `None` so callers fall back to the legacy shadow box.
pub(super) fn frustum_slice_corners(
    inv_view_proj: [[f32; 4]; 4],
    near: f32,
    far: f32,
    split_near: f32,
    split_far: f32,
) -> Option<[[f32; CORNER_LANES]; FRUSTUM_CORNERS]> {
    // facade: validate the range, map depths to NDC, unproject the slice.
    if !(near > 0.0 && far > near && near.is_finite() && far.is_finite()) {
        return None;
    }
    let inv = glam::Mat4::from_cols_array_2d(&inv_view_proj);
    if !inv.is_finite() {
        return None;
    }
    let lo = split_near.clamp(near, far);
    let hi = split_far.clamp(near, far);
    if lo >= hi {
        return None;
    }
    // DirectX perspective: NDC z = A + B / view_depth.
    let range = far - near;
    let mut corners = [[0.0; CORNER_LANES]; FRUSTUM_CORNERS];
    let mut idx = 0;
    for depth in [lo, hi] {
        let z = far / range - near * far / range / depth;
        for xy in [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]] {
            corners[idx] = unproject_ndc(inv, [xy[0], xy[1], z])?;
            idx += 1;
        }
    }
    Some(corners)
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

/// Labels for one depth-array pool (texture, layer view, array view, VP buffer).
struct DepthPoolLabels {
    texture: &'static str,
    layer: &'static str,
    array: &'static str,
    buffer: &'static str,
}

/// Layers of the point-shadow cube array (cubes × faces).
const POINT_SHADOW_FACE_LAYERS: usize = POINT_SHADOW_CUBES * CUBE_FACE_COUNT;

/// Depth-array pool shared by the legacy shadow maps, the CSM cascades,
/// and the point-light cube faces: one square depth texture, a 2D view
/// per layer, one sampling view, and a VP uniform per layer.
#[allow(clippy::type_complexity)]
fn create_depth_array_pool<const LAYERS: usize>(
    device: &wgpu::Device,
    labels: DepthPoolLabels,
    size: u32,
    array_dim: wgpu::TextureViewDimension,
) -> (
    wgpu::Texture,
    [wgpu::TextureView; LAYERS],
    wgpu::TextureView,
    [wgpu::Buffer; LAYERS],
) {
    let maps = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(labels.texture),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: LAYERS as u32,
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
    let views = std::array::from_fn(|layer| {
        maps.create_view(&wgpu::TextureViewDescriptor {
            label: Some(labels.layer),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: layer as u32,
            array_layer_count: Some(1),
            ..Default::default()
        })
    });
    let array_view = maps.create_view(&wgpu::TextureViewDescriptor {
        label: Some(labels.array),
        dimension: Some(array_dim),
        ..Default::default()
    });
    let vp_buffers = std::array::from_fn(|_| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(labels.buffer),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });
    (maps, views, array_view, vp_buffers)
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
        let (shadow_maps, shadow_views, shadow_array_view, shadow_vp_buffers) =
            create_depth_array_pool::<SHADOW_LAYERS>(
                device,
                DepthPoolLabels {
                    texture: "shadow maps",
                    layer: "shadow map layer",
                    array: "shadow map array",
                    buffer: "shadow VP buffer",
                },
                SHADOW_SIZE,
                wgpu::TextureViewDimension::D2Array,
            );
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

    /// Allocate the CSM depth pool: one layer per cascade with per-layer
    /// views, the sampling array view, and VP uniform buffers (#121).
    /// Shares the depth-only pipelines and the comparison sampler with
    /// the legacy pool (same format, same bias).
    pub(super) fn create_csm_targets(
        device: &wgpu::Device,
    ) -> (
        wgpu::Texture,
        [wgpu::TextureView; CSM_CASCADES],
        wgpu::TextureView,
        [wgpu::Buffer; CSM_CASCADES],
    ) {
        create_depth_array_pool::<CSM_CASCADES>(
            device,
            DepthPoolLabels {
                texture: "csm maps",
                layer: "csm map layer",
                array: "csm map array",
                buffer: "csm VP buffer",
            },
            SHADOW_SIZE,
            wgpu::TextureViewDimension::D2Array,
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
        create_depth_array_pool::<POINT_SHADOW_FACE_LAYERS>(
            device,
            DepthPoolLabels {
                texture: "point shadow cubes",
                layer: "point shadow cube face",
                array: "point shadow cube array",
                buffer: "point shadow face VP buffer",
            },
            SHADOW_CUBE_SIZE,
            wgpu::TextureViewDimension::CubeArray,
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
        self.render_shadows_batch(device, encoder, mesh, instance_count, customs);
    }

    /// Layer/cube loop of the depth pre-pass over the staged range.
    ///
    /// Shadows always draw directly: the pre-pass must cover every
    /// caster whose shadow can reach the frame, including casters
    /// outside the camera frustum, so the camera-compacted batch is
    /// never reused here (a per-light-frustum compaction is future
    /// work, not this entry point).
    fn render_shadows_batch(
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
        let cascades = self
            .csm_count
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(CSM_CASCADES as u32);
        if count == 0 && cubes == 0 && cascades == 0 {
            return;
        }
        for layer in 0..count as usize {
            self.render_shadow_view_batch(
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
        for layer in 0..cascades as usize {
            self.render_shadow_view_batch(
                device,
                encoder,
                mesh,
                instance_count,
                customs,
                &self.shadow_pipeline,
                &self.skinned_shadow_pipeline,
                &self.csm_vp_buffers[layer],
                &self.csm_views[layer],
            );
        }
        for cube in 0..cubes as usize {
            for face in 0..CUBE_FACE_COUNT {
                let idx = cube * CUBE_FACE_COUNT + face;
                self.render_shadow_view_batch(
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
    fn render_shadow_view_batch(
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
        read_depth_layer(device, queue, &self.shadow_maps, layer)
    }

    /// Test-only readback of one CSM cascade layer as row-major depths.
    #[cfg(test)]
    pub(crate) fn read_csm_layer_for_tests(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layer: u32,
    ) -> Vec<f32> {
        read_depth_layer(device, queue, &self.csm_maps, layer)
    }

    /// Test-only: replace one cascade layer with a constant depth.
    ///
    /// `0` compares as shadowed and `1` as lit for a receiver whose
    /// light-space depth sits inside the map, so a probe can mark which
    /// layer the shader sampled.
    #[cfg(test)]
    pub(crate) fn clear_csm_layer_for_tests(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: u32,
        depth: f32,
    ) {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("csm cascade marker"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.csm_views[layer as usize],
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(depth),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
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

/// Row-major Depth32Float readback of one array layer. Empty when the map fails.
#[cfg(test)]
fn read_depth_layer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    layer: u32,
) -> Vec<f32> {
    /// Bytes per Depth32Float texel.
    const DEPTH32_BYTES: u32 = 4;
    let size = SHADOW_SIZE;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test depth layer readback"),
        size: (size * size * DEPTH32_BYTES) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test depth layer readback"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
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

#[cfg(test)]
mod tests {
    use super::super::lights::build_lighting_uniform;
    use super::super::test_util::*;
    use super::super::*;
    use super::*;
    use crate::render_backend::tests::yawed_eye;
    use ornis_assets::scene::ShadowCast;
    #[test]
    fn dir_shadow_vp_centers_origin_with_light_depth_order() {
        // Light above: to-light = +Y, eye at +Y·30 looking at origin.
        let vp = dir_shadow_vp(glam::Vec3::Y);
        let center = ndc_of(vp, [0.0, 0.0, 0.0]);
        assert!(center[0].abs() < 1e-5 && center[1].abs() < 1e-5);
        assert!((0.0..=1.0).contains(&center[2]));
        // Nearer the light (higher Y) = smaller depth.
        let hi = ndc_of(vp, [0.0, 5.0, 0.0])[2];
        let lo = ndc_of(vp, [0.0, -5.0, 0.0])[2];
        assert!(hi < center[2] && center[2] < lo, "{hi} {center:?} {lo}");
        // Box rim stays inside the frustum.
        for p in [
            [12.0, 0.0, 0.0],
            [-12.0, 0.0, 0.0],
            [0.0, 0.0, 12.0],
            [0.0, 0.0, -12.0],
        ] {
            let n = ndc_of(vp, p);
            assert!(
                n[0].abs() <= 1.0 + 1e-4 && n[1].abs() <= 1.0 + 1e-4,
                "{p:?} -> {n:?}"
            );
        }
    }

    #[test]
    fn spot_shadow_vp_points_down_its_axis() {
        let vp = spot_shadow_vp([0.0, 6.0, 0.0], glam::Vec3::NEG_Y, 35.0, 40.0);
        let hit = ndc_of(vp, [0.0, 0.0, 0.0]);
        assert!(hit[0].abs() < 1e-4 && hit[1].abs() < 1e-4);
        assert!((0.0..=1.0).contains(&hit[2]));
        // Behind the light has negative clip w (mirrored projection).
        assert!(clip_of(vp, [0.0, 8.0, 0.0]).w < 0.0);
        // Off-axis outside the 35° cone (half-width ≈ 4.2 at depth 6;
        // the shadow frame maps world X onto NDC Y here).
        let side = ndc_of(vp, [5.0, 0.0, 0.0]);
        assert!(side[0].abs().max(side[1].abs()) > 1.0, "{side:?}");
    }

    #[test]
    fn point_cube_face_vp_centers_its_axis() {
        // Light at origin-ish, range 20: each face centers its axis
        // with directx depth order (nearer = smaller).
        let pos = [2.0, 3.0, 4.0];
        let axes = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for (face, axis) in axes.iter().enumerate() {
            let vp = point_cube_face_vp(pos, 20.0, face);
            let target = [
                pos[0] + axis[0] * 5.0,
                pos[1] + axis[1] * 5.0,
                pos[2] + axis[2] * 5.0,
            ];
            let hit = ndc_of(vp, target);
            assert!(
                hit[0].abs() < 1e-4 && hit[1].abs() < 1e-4,
                "face {face}: {hit:?}"
            );
            assert!((0.0..=1.0).contains(&hit[2]), "face {face}: {hit:?}");
            // Nearer along the axis = smaller depth (matches the
            // analytic sampling formula's monotonicity).
            let near = ndc_of(
                vp,
                [
                    pos[0] + axis[0] * 2.0,
                    pos[1] + axis[1] * 2.0,
                    pos[2] + axis[2] * 2.0,
                ],
            )[2];
            let far = ndc_of(
                vp,
                [
                    pos[0] + axis[0] * 10.0,
                    pos[1] + axis[1] * 10.0,
                    pos[2] + axis[2] * 10.0,
                ],
            )[2];
            assert!(
                near < hit[2] && hit[2] < far,
                "face {face}: {near} {hit:?} {far}"
            );
        }
    }

    #[test]
    fn point_cube_face_vp_matches_sampler_frame() {
        // Texel placement must match the hardware cube-sampling frame
        // (GL convention: +X: (−z,−y), −X: (+z,−y), +Y: (+x,+z),
        // −Y: (+x,−z), +Z: (+x,−y), −Z: (−x,−y)). Rasterization maps
        // NDC y+1 onto texture row 0 while the sampler reads v=0 from
        // the top, so a +u-source offset lands at positive NDC x and a
        // +v-source offset at NEGATIVE NDC y (the V-mirror in
        // `point_cube_face_vp`). Regression pin: without the mirror
        // the sampler reads mirrored texels (all-lit point shadows).
        let pos = [2.0, 3.0, 4.0];
        let axes: [[[f32; 3]; 2]; 6] = [
            [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        ];
        let face_axis = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for (face, ([u, v], axis)) in axes.iter().zip(face_axis.iter()).enumerate() {
            let vp = point_cube_face_vp(pos, 20.0, face);
            let base = [
                pos[0] + axis[0] * 5.0,
                pos[1] + axis[1] * 5.0,
                pos[2] + axis[2] * 5.0,
            ];
            let pu = [base[0] + u[0], base[1] + u[1], base[2] + u[2]];
            let pv = [base[0] + v[0], base[1] + v[1], base[2] + v[2]];
            let nu = ndc_of(vp, pu);
            let nv = ndc_of(vp, pv);
            assert!(nu[0] > 0.05, "face {face}: +u offset -> {nu:?}");
            assert!(nv[1] < -0.05, "face {face}: +v offset -> {nv:?}");
        }
    }

    #[test]
    fn shadow_overflow_counts_dropped_shadows() {
        // Five shadowed directionals over four 2D layers.
        let dirs: Vec<LightDesc> = (0..5)
            .map(|_| dir_probe([0.0, 1.0, 0.0], ornis_assets::scene::ShadowCast::Enabled))
            .collect();
        let stats = count_light_drops(&dirs);
        assert_eq!(stats.uploaded, 5);
        assert_eq!(stats.dropped_lights, 0);
        assert_eq!(stats.dropped_shadows, 1, "{stats:?}");
        // Three shadowed points over two cubes.
        let points: Vec<LightDesc> = (0..3)
            .map(|i| LightDesc::Point {
                position: glam::Vec3::new(i as f32, 4.0, 6.0),
                intensity: 100.0,
                color: [1.0, 1.0, 1.0],
                range: ornis_core::units::Meters::new(30.0),
                shadow: ShadowCast::Enabled,
            })
            .collect();
        let stats = count_light_drops(&points);
        assert_eq!(stats.dropped_shadows, 1, "{stats:?}");
        // A shadow request on a dropped excess light counts too.
        let mut lights: Vec<LightDesc> = (0..8)
            .map(|_| dir_probe([1.0, 1.0, 1.0], ornis_assets::scene::ShadowCast::Disabled))
            .collect();
        lights.push(dir_probe(
            [0.0, 1.0, 0.0],
            ornis_assets::scene::ShadowCast::Enabled,
        ));
        lights.push(dir_probe(
            [0.0, 1.0, 0.0],
            ornis_assets::scene::ShadowCast::Enabled,
        ));
        let stats = count_light_drops(&lights);
        assert_eq!(stats.dropped_lights, 2, "{stats:?}");
        assert_eq!(stats.dropped_shadows, 2, "{stats:?}");
    }

    #[test]
    fn legacy_shadow_path_matches_dir_shadow_vp() {
        // `fit: None` must reproduce the legacy matrix bit-for-bit:

        // directional-only scenes stay pixel-identical.
        let light = dir_probe([1.0, 1.0, 1.0], ornis_assets::scene::ShadowCast::Enabled);
        let built = build_lighting_uniform(
            [0.1, 0.1, 0.15],
            1.0,
            1.0,
            &[light],
            None,
            None,
            0.0,
            0.0,
            0,
        );
        let v = glam::Vec3::new(1.0, 1.0, 1.0).normalize();
        assert_eq!(built.uniform.lights[0].shadow_vp, dir_shadow_vp(v));
        assert_eq!(built.uniform.lights[0].params[3], 0.0);
    }

    #[test]
    fn fitted_shadow_covers_radius_50_scene() {
        let (center, half) = shadow_fit_for_bounds([-50.0; 3], [50.0; 3]);
        assert_eq!(center, [0.0, 0.0, 0.0]);
        assert!((half - 50.0).abs() < 1e-6, "{half}");
        let vp = dir_shadow_vp_fitted(glam::Vec3::Y, center, half);
        for x in [-50.0, 50.0] {
            for y in [-50.0, 50.0] {
                for z in [-50.0, 50.0] {
                    let n = ndc_of(vp, [x, y, z]);
                    assert!(
                        n[0].abs() <= 1.0 + 1e-3 && n[1].abs() <= 1.0 + 1e-3,
                        "{x},{y},{z} -> {n:?}"
                    );
                    assert!((0.0..=1.0).contains(&n[2]), "{x},{y},{z} -> {n:?}");
                }
            }
        }
    }

    #[test]
    fn default_fit_keeps_legacy_box() {
        assert_eq!(
            shadow_fit_for_bounds([-1.0; 3], [1.0; 3]),
            ([0.0; 3], SHADOW_ORTHO_HALF)
        );
        assert_eq!(SHADOW_ORTHO_HALF, 12.0);
        assert_eq!(
            shadow_fit_for_bounds([f32::NAN; 3], [0.0; 3]),
            ([0.0; 3], SHADOW_ORTHO_HALF)
        );
    }

    #[test]
    fn cascade_splits_span_near_to_far_monotonic() {
        let splits = cascade_splits(1.0, 101.0, CSM_CASCADES, CSM_SPLIT_LAMBDA);
        assert!((splits[0] - 1.0).abs() < 1e-6, "{splits:?}");
        assert!((splits[CSM_CASCADES] - 101.0).abs() < 1e-6, "{splits:?}");
        for pair in splits.windows(2) {
            assert!(pair[0] < pair[1], "{splits:?}");
        }
    }

    #[test]
    fn cascade_splits_lambda_extremes_match_uniform_and_log() {
        let uniform = cascade_splits(1.0, 101.0, CSM_CASCADES, 0.0);
        assert!((uniform[2] - 51.0).abs() < 1e-4, "{uniform:?}");
        let log = cascade_splits(1.0, 101.0, CSM_CASCADES, 1.0);
        let expected = 101.0f32.powf(0.5);
        assert!((log[2] - expected).abs() < 1e-3, "{log:?}");
    }

    #[test]
    fn cascade_splits_unused_tail_repeats_far() {
        let splits = cascade_splits(0.5, 60.0, 2, CSM_SPLIT_LAMBDA);
        assert!((splits[0] - 0.5).abs() < 1e-6, "{splits:?}");
        assert!(splits[1] > splits[0] && splits[1] < 60.0, "{splits:?}");
        for tail in splits.iter().skip(2) {
            assert!((*tail - 60.0).abs() < 1e-6, "{splits:?}");
        }
    }

    #[test]
    fn cascade_splits_degenerate_range_falls_back() {
        let splits = cascade_splits(10.0, 5.0, CSM_CASCADES, CSM_SPLIT_LAMBDA);
        assert!((splits[0] - 10.0).abs() < 1e-6, "{splits:?}");
        for tail in splits.iter().skip(1) {
            assert!((*tail - 5.0).abs() < 1e-6, "{splits:?}");
        }
    }

    fn box_corners(center: [f32; 3], half: f32) -> [[f32; 3]; 8] {
        let mut corners = [[0.0; 3]; 8];
        for (i, corner) in corners.iter_mut().enumerate() {
            corner[0] = center[0] + if i & 1 == 0 { -half } else { half };
            corner[1] = center[1] + if i & 2 == 0 { -half } else { half };
            corner[2] = center[2] + if i & 4 == 0 { -half } else { half };
        }
        corners
    }

    #[test]
    fn cascade_shadow_vp_contains_its_slice() {
        let to_light = glam::Vec3::new(0.42, 0.84, 0.3).normalize();
        let corners = box_corners([0.0, 0.0, 25.0], 5.0);
        let vp = cascade_shadow_vp(to_light, corners, corners, None, SHADOW_SIZE);
        for c in corners {
            let n = ndc_of(vp, c);
            assert!(
                n[0].abs() <= 1.0 + 1e-3 && n[1].abs() <= 1.0 + 1e-3,
                "{c:?} -> {n:?}"
            );
            assert!((0.0..=1.0).contains(&n[2]), "{c:?} -> {n:?}");
        }
    }

    #[test]
    fn cascade_texel_snap_stays_on_grid_and_contains() {
        let min = glam::Vec2::new(2.5, -1.75);
        let max = glam::Vec2::new(9.5, 4.25);
        let texel = (max.x - min.x).max(max.y - min.y) / SHADOW_SIZE as f32;
        let snapped = glam::Vec2::new(
            snap_down_to_texel(min.x, texel),
            snap_down_to_texel(min.y, texel),
        );
        assert!(snapped.x <= min.x && snapped.y <= min.y);
        for v in [snapped.x / texel, snapped.y / texel] {
            assert!((v.round() - v).abs() < 1e-3, "{v}");
        }
    }

    #[test]
    fn cascade_shadow_vp_moves_with_its_slice() {
        let to_light = glam::Vec3::new(0.42, 0.84, 0.3).normalize();
        let base = box_corners([0.0, 0.0, 25.0], 5.0);
        let vp = cascade_shadow_vp(to_light, base, base, None, SHADOW_SIZE);
        assert_eq!(
            vp,
            cascade_shadow_vp(to_light, base, base, None, SHADOW_SIZE)
        );
        let shifted = box_corners([40.0, 0.0, 25.0], 5.0);
        assert_ne!(
            vp,
            cascade_shadow_vp(to_light, shifted, shifted, None, SHADOW_SIZE)
        );
    }

    #[test]
    fn cascade_shadow_vp_degenerate_falls_back_to_legacy() {
        let to_light = glam::Vec3::Y;
        assert_eq!(
            cascade_shadow_vp(
                to_light,
                [[1.0, 2.0, 3.0]; 8],
                [[1.0, 2.0, 3.0]; 8],
                None,
                SHADOW_SIZE
            ),
            dir_shadow_vp(to_light)
        );
        assert_eq!(
            cascade_shadow_vp(
                to_light,
                box_corners([0.0, 0.0, 0.0], 1.0),
                box_corners([0.0, 0.0, 0.0], 1.0),
                None,
                0
            ),
            dir_shadow_vp(to_light)
        );
    }

    fn probe_camera() -> ([[f32; 4]; 4], glam::Vec3, glam::Vec3) {
        let eye = glam::Vec3::new(0.0, 3.0, 10.0);
        let fwd = (glam::Vec3::new(0.0, 1.0, 0.0) - eye).normalize();
        let view = glam::camera::rh::view::look_at_mat4(
            eye,
            glam::Vec3::new(0.0, 1.0, 0.0),
            glam::Vec3::Y,
        );
        let proj = glam::camera::rh::proj::directx::perspective(
            55.0f32.to_radians(),
            16.0 / 9.0,
            0.1,
            100.0,
        );
        let inv = (proj * view).inverse().to_cols_array_2d();
        (inv, eye, fwd)
    }

    #[test]
    fn frustum_slice_corners_match_ndc_faces() {
        let (inv, _, _) = probe_camera();
        let full = frustum_slice_corners(inv, 0.1, 100.0, 0.1, 100.0).expect("valid range");
        let vp = glam::Mat4::from_cols_array_2d(&inv)
            .inverse()
            .to_cols_array_2d();
        for (i, c) in full.iter().enumerate() {
            let n = ndc_of(vp, *c);
            assert!(n[0].abs() > 0.99 && n[0].abs() < 1.01, "{c:?} -> {n:?}");
            assert!(n[1].abs() > 0.99 && n[1].abs() < 1.01, "{c:?} -> {n:?}");
            let face = if i < 4 { 0.0 } else { 1.0 };
            assert!((n[2] - face).abs() < 1e-3, "{c:?} -> {n:?}");
        }
    }

    #[test]
    fn frustum_slice_corners_split_stays_inside() {
        let (inv, eye, fwd) = probe_camera();
        let splits = cascade_splits(0.1, 100.0, CSM_CASCADES, CSM_SPLIT_LAMBDA);
        for pair in splits.windows(2) {
            let corners =
                frustum_slice_corners(inv, 0.1, 100.0, pair[0], pair[1]).expect("valid split");
            for c in corners {
                let p = glam::Vec3::from_array(c);
                let depth = (p - eye).dot(fwd);
                assert!(
                    depth >= pair[0] - 0.05 && depth <= pair[1] + 0.05,
                    "{c:?} depth {depth} outside {:?}",
                    pair
                );
            }
        }
    }

    #[test]
    fn frustum_slice_corners_clamps_and_rejects() {
        let (inv, _, _) = probe_camera();
        let full = frustum_slice_corners(inv, 0.1, 100.0, 0.1, 100.0).expect("valid");
        assert_eq!(
            frustum_slice_corners(inv, 0.1, 100.0, -5.0, 500.0),
            Some(full)
        );
        assert!(frustum_slice_corners(inv, 0.1, 100.0, 50.0, 50.0).is_none());
        assert!(frustum_slice_corners(inv, 100.0, 0.1, 0.1, 100.0).is_none());
        assert!(frustum_slice_corners(inv, -1.0, 100.0, 0.1, 50.0).is_none());
    }

    #[test]
    fn build_csm_covers_golden_probe_frustum() {
        use super::super::lights::{CsmCamera, build_csm};
        // Exact replica of the golden CSM probe camera (200 m far plane).
        let eye = glam::Vec3::new(0.0, 10.0, 20.0);
        let target = glam::Vec3::new(0.0, 2.0, -60.0);
        let fwd = (target - eye).normalize();
        let view = glam::camera::rh::view::look_at_mat4(eye, target, glam::Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(
            55.0f32.to_radians(),
            320.0 / 180.0,
            0.1,
            200.0,
        );
        let inv = (proj * view).inverse().to_cols_array_2d();
        let to_light = glam::Vec3::new(0.42, 0.84, 0.3).normalize();
        let block = build_csm(
            inv,
            CsmCamera {
                near: 0.1,
                far: 200.0,
                forward: fwd.to_array(),
            },
            to_light,
            None,
        )
        .expect("golden probe fit must succeed");
        assert_eq!(block.cascade_count, CSM_CASCADES as u32);
        let splits = [
            block.splits_lo[0],
            block.splits_lo[1],
            block.splits_lo[2],
            block.splits_lo[3],
            block.split_far,
        ];
        assert!((splits[0] - 0.1).abs() < 1e-6, "{splits:?}");
        assert!((splits[CSM_CASCADES] - 200.0).abs() < 1e-3, "{splits:?}");
        for (i, pair) in splits.windows(2).enumerate() {
            let corners =
                frustum_slice_corners(inv, 0.1, 200.0, pair[0], pair[1]).expect("valid split");
            let vp = block.vps[i].cols;
            for c in corners {
                let n = ndc_of(vp, c);
                assert!(
                    n[0].abs() <= 1.0 + 1e-3 && n[1].abs() <= 1.0 + 1e-3,
                    "cascade {i} {c:?} -> {n:?}"
                );
                assert!((0.0..=1.0).contains(&n[2]), "cascade {i} {c:?} -> {n:?}");
            }
        }
        // The golden probe's occluder sits in front of the last slice:
        // its depth must still fall inside that cascade's range (#121).
        let receiver = glam::Vec3::new(0.0, 0.0, -70.0);
        let occluder = receiver + to_light * 10.0;
        for p in [receiver, occluder] {
            let n = ndc_of(block.vps[3].cols, p.to_array());
            assert!((0.0..=1.0).contains(&n[2]), "{p:?} -> {n:?}");
        }
    }

    /// Shadow-map texel of a world point (same UV the evaluator builds).
    fn shadow_texel(vp: [[f32; 4]; 4], world: [f32; 3]) -> [f32; 2] {
        let n = ndc_of(vp, world);
        [
            (n[0] * HALF + HALF) * SHADOW_SIZE as f32,
            (HALF - n[1] * HALF) * SHADOW_SIZE as f32,
        ]
    }

    /// Absolute distance of `b - a` from the nearest integer.
    fn fractional_shift(a: f32, b: f32) -> f32 {
        let delta = b - a;
        (delta - delta.round()).abs()
    }

    /// World size of one shadow texel, from the ortho scale of `vp`.
    fn texel_world(vp: [[f32; 4]; 4]) -> f32 {
        let row = glam::Vec3::new(vp[0][0], vp[1][0], vp[2][0]);
        (2.0 / row.length()) / SHADOW_SIZE as f32
    }

    /// A sub-texel slice translation moves a fixed world point by a whole
    /// number of texels. Dropping the snap (or anchoring the light view on
    /// the moving slice center) leaves a fractional drift.
    #[test]
    fn cascade_subtexel_shift_moves_texels_by_integers() {
        let to_light = glam::Vec3::new(0.42, 0.84, 0.3).normalize();
        let base = box_corners([0.0, 0.0, 25.0], 5.0);
        let vp = cascade_shadow_vp(to_light, base, base, None, SHADOW_SIZE);
        let probe = [1.0, 0.0, 25.0];
        let before = shadow_texel(vp, probe);
        let shift = texel_world(vp) * 0.3;
        let mut moved = base;
        for corner in &mut moved {
            corner[0] += shift;
        }
        let after = shadow_texel(
            cascade_shadow_vp(to_light, moved, moved, None, SHADOW_SIZE),
            probe,
        );
        assert!(
            fractional_shift(before[0], after[0]) < 1e-3,
            "{before:?} -> {after:?}"
        );
        assert!(
            fractional_shift(before[1], after[1]) < 1e-3,
            "{before:?} -> {after:?}"
        );
    }

    fn cascade_texel_bits(eye: glam::Vec3, target: glam::Vec3) -> [u32; CSM_CASCADES] {
        use super::super::lights::{CsmCamera, build_csm};
        let fwd = (target - eye).normalize();
        let view = glam::camera::rh::view::look_at_mat4(eye, target, glam::Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(
            55.0f32.to_radians(),
            320.0 / 180.0,
            0.1,
            200.0,
        );
        let block = build_csm(
            (proj * view).inverse().to_cols_array_2d(),
            CsmCamera {
                near: 0.1,
                far: 200.0,
                forward: fwd.to_array(),
            },
            glam::Vec3::new(0.42, 0.84, 0.3).normalize(),
            None,
        )
        .expect("yaw fit");
        std::array::from_fn(|i| texel_world(block.vps[i].cols).to_bits())
    }

    /// Yaw does not change the texel size: the map side is the sphere
    /// diameter, not the light-space AABB. An AABB fit changes it.
    #[test]
    fn cascade_texel_size_is_bit_identical_under_camera_yaw() {
        let eye = glam::Vec3::new(0.0, 10.0, 20.0);
        let target = glam::Vec3::new(0.0, 2.0, -60.0);
        let base = cascade_texel_bits(eye, target);
        for degrees in [1.0_f32, 5.0] {
            let yawed = cascade_texel_bits(yawed_eye(eye, target, degrees.to_radians()), target);
            assert_eq!(base, yawed, "yaw {degrees}° changed the texel size");
        }
    }

    fn cascade_zero_vp(eye: glam::Vec3, target: glam::Vec3) -> [[f32; 4]; 4] {
        use super::super::lights::{CsmCamera, build_csm};
        let fwd = (target - eye).normalize();
        let view = glam::camera::rh::view::look_at_mat4(eye, target, glam::Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(
            55.0f32.to_radians(),
            320.0 / 180.0,
            0.1,
            200.0,
        );
        build_csm(
            (proj * view).inverse().to_cols_array_2d(),
            CsmCamera {
                near: 0.1,
                far: 200.0,
                forward: fwd.to_array(),
            },
            glam::Vec3::new(0.42, 0.84, 0.3).normalize(),
            None,
        )
        .expect("yaw fit")
        .vps[0]
            .cols
    }

    /// A camera yaw moves a fixed world point by a whole number of texels.
    /// A light view anchored on the slice center leaves a fractional drift.
    #[test]
    fn cascade_yaw_shifts_texels_by_integers() {
        let eye = glam::Vec3::new(0.0, 10.0, 20.0);
        let target = glam::Vec3::new(0.0, 2.0, -60.0);
        let probe = target.to_array();
        let before = shadow_texel(cascade_zero_vp(eye, target), probe);
        for degrees in [1.0_f32, 5.0] {
            let yawed = yawed_eye(eye, target, degrees.to_radians());
            let after = shadow_texel(cascade_zero_vp(yawed, target), probe);
            assert!(
                fractional_shift(before[0], after[0]) < 1e-3
                    && fractional_shift(before[1], after[1]) < 1e-3,
                "yaw {degrees}° fractional drift {before:?} -> {after:?}"
            );
        }
    }

    /// A caster outside the camera box is inside the depth range only when
    /// its AABB is passed. Depth taken from the slice alone clips it.
    #[test]
    fn cascade_depth_includes_casters_outside_the_slice() {
        let to_light = glam::Vec3::Y;
        let slice = box_corners([0.0, 0.0, 0.0], 5.0);
        let caster = [0.0, 40.0, 0.0];
        let casters = Some(([-1.0, 39.0, -1.0], [1.0, 41.0, 1.0]));
        let covered = ndc_of(
            cascade_shadow_vp(to_light, slice, slice, casters, SHADOW_SIZE),
            caster,
        );
        assert!(
            (0.0..=1.0).contains(&covered[2]),
            "caster AABB must extend the near plane, got {covered:?}"
        );
        let clipped = ndc_of(
            cascade_shadow_vp(to_light, slice, slice, None, SHADOW_SIZE),
            caster,
        );
        assert!(
            !(0.0..=1.0).contains(&clipped[2]),
            "slice-only depth must clip a caster 40 m toward the light, got {clipped:?}"
        );
    }

    /// A point that sits in the full frustum but outside this slice stays
    /// in range. Passing the slice as the depth bounds clips it.
    #[test]
    fn cascade_depth_covers_a_neighboring_slice() {
        let to_light = glam::Vec3::Y;
        let slice = box_corners([0.0, 0.0, 0.0], 4.0);
        // The neighbor sits 30 m toward the light: inside the full frustum
        // box, far outside this slice's depth plus the 1 m near pad.
        let frustum = box_corners([0.0, 15.0, 0.0], 20.0);
        let neighbor = [0.0, 30.0, 0.0];
        let covered = ndc_of(
            cascade_shadow_vp(to_light, slice, frustum, None, SHADOW_SIZE),
            neighbor,
        );
        assert!(
            (0.0..=1.0).contains(&covered[2]),
            "full-frustum depth must include the neighbor, got {covered:?}"
        );
        let clipped = ndc_of(
            cascade_shadow_vp(to_light, slice, slice, None, SHADOW_SIZE),
            neighbor,
        );
        assert!(
            !(0.0..=1.0).contains(&clipped[2]),
            "slice-only depth must clip the neighbor, got {clipped:?}"
        );
    }

    /// `splits_lo` for λ = 0.5 matches the practical-split formula, and it
    /// is not the uniform split. `CSM_SPLIT_LAMBDA = 0` fails this.
    #[test]
    fn build_csm_splits_match_practical_lambda() {
        use super::super::lights::{CsmCamera, build_csm};
        let near = 0.1;
        let far = 200.0;
        let eye = glam::Vec3::new(0.0, 8.0, 12.0);
        let target = glam::Vec3::new(0.0, 0.0, -40.0);
        let fwd = (target - eye).normalize();
        let view = glam::camera::rh::view::look_at_mat4(eye, target, glam::Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(
            55.0f32.to_radians(),
            16.0 / 9.0,
            near,
            far,
        );
        let block = build_csm(
            (proj * view).inverse().to_cols_array_2d(),
            CsmCamera {
                near,
                far,
                forward: fwd.to_array(),
            },
            glam::Vec3::Y,
            None,
        )
        .expect("fit");
        let mut expected = [far; CSM_CASCADES + 1];
        expected[0] = near;
        for (i, slot) in expected[1..CSM_CASCADES].iter_mut().enumerate() {
            let t = (i + 1) as f32 / CSM_CASCADES as f32;
            let uniform = near + (far - near) * t;
            let log = near * (far / near).powf(t);
            *slot = HALF * log + (1.0 - HALF) * uniform;
        }
        let got = [
            block.splits_lo[0],
            block.splits_lo[1],
            block.splits_lo[2],
            block.splits_lo[3],
            block.split_far,
        ];
        for (i, (got, want)) in got.iter().zip(expected).enumerate() {
            assert!((got - want).abs() < 1e-3, "split {i}: {got} != {want}");
        }
        let uniform_mid = near + (far - near) * HALF;
        assert!(
            (got[2] - uniform_mid).abs() > 10.0,
            "λ = 0.5 must not collapse to the uniform split ({got:?} vs {uniform_mid})"
        );
    }
}
