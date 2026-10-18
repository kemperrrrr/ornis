//! Scene lighting upload: the `GpuLight` pack, the `LightingUniform` block built by `build_lighting_uniform`, and the `set_lights` entry points with their drop accounting.
use super::clusters::MAX_CLUSTERED_LIGHTS;
use super::shadows::{
    CUBE_FACE_COUNT, cascade_shadow_vp, cascade_splits, dir_shadow_vp, dir_shadow_vp_fitted,
    frustum_slice_corners, point_cube_face_vp, spot_shadow_vp,
};
use super::*;
/// One GPU light: direction + color packed as `vec4`s.
///
/// Evaluation kind of a [`GpuLight`] entry: directional, point, or spot.
/// Float (not enum): the stage DSL compares kinds with `>`.
pub(crate) const LIGHT_KIND_DIRECTIONAL: f32 = 0.0;

/// Evaluation kind of a [`GpuLight`] entry: point light.
pub(crate) const LIGHT_KIND_POINT: f32 = 1.0;

/// Evaluation kind of a [`GpuLight`] entry: spotlight.
pub(crate) const LIGHT_KIND_SPOT: f32 = 2.0;

/// Maximum lights uploaded per frame: the `LightingUniform` WGSL block
/// spells `array<Light, 8>`, and [`Renderer3D::set_lights`] uploads the
/// first eight entries of any kind. Excess lights are dropped and reported
/// in [`LightUploadStats::dropped_lights`] (never silently).
pub const MAX_LIGHTS: usize = 8;

/// Compile-time pin: the WGSL derive only accepts integer literals for
/// array lengths, so [`LightingUniform::lights`] spells `8` literally —
/// this assert keeps the spell and the limit in sync.
const _: [(); MAX_LIGHTS] = [(); 8];

/// Per-`set_lights` upload report: explicit counts for what the old
/// silent-truncate path dropped (excess lights, shadow requests without a
/// free layer/cube slot). Returned by
/// [`Renderer3D::set_lights_full`] and [`count_light_drops`], and published
/// for the last upload via [`Renderer3D::light_upload_stats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightUploadStats {
    /// Lights written into the uniform (`min(len, MAX_LIGHTS)`).
    pub uploaded: u32,
    /// Scene lights beyond [`MAX_LIGHTS`] that were not uploaded into
    /// the uniform window. With clustered lighting on, the point/spot
    /// lights among them are uploaded into the cluster storage instead
    /// (see [`Renderer3D::cluster_light_stats`]); only the directionals
    /// stay unrendered.
    pub dropped_lights: u32,
    /// Shadow requests (`shadow: ShadowCast::Enabled`) that received no map slot: 2D
    /// layers ([`SHADOW_LAYERS`]) for directional/spot lights and cube
    /// slots ([`POINT_SHADOW_CUBES`]) for point lights, including requests
    /// on dropped excess lights. These lights render unshadowed.
    pub dropped_shadows: u32,
}

/// Floor for light range / attenuation denominators (m).
const LIGHT_RANGE_EPS: f32 = 1e-3;

/// GPU light entry: kind-selected evaluation in both fragment entries
/// (deferred lighting and forward PBR share the layer evaluators).
///
/// The WGSL `Light` declaration is generated from this layout
/// ([`GpuLight::WGSL_SOURCE`]). `align(16)` matches the WGSL struct alignment
/// so the derive's nested-layout check holds (cf. physics `GpuBodyState`).
///
/// Layout note: `kind` is a full vec4 (kind in `x`) rather than a scalar —
/// a bare `f32` ahead of the vec4s would insert implicit padding that
/// `bytemuck::Pod` rejects and the derive cannot spell.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "Light")]
pub(crate) struct GpuLight {
    /// Evaluation kind in `x`: 0.0 directional, 1.0 point, 2.0 spot.
    pub(super) kind: [f32; 4],
    /// Direction toward the light (directional) or spot axis from the
    /// light into the scene (spot); `[0, 0, 1, 0]` for points.
    pub(super) direction: [f32; 4],
    /// World-space position (point/spot); unused by directionals.
    pub(super) position: [f32; 4],
    /// Emission color RGB + radiometric intensity in alpha.
    pub(super) color: [f32; 4],
    /// `(range, cos_inner, cos_outer, shadow_layer)`: range cutoff for
    /// point/spot, spot cone cosines, shadow-map layer or -1.0.
    pub params: [f32; 4],
    /// Light-space clip matrix for the shadow map (`params.w` layer);
    /// identity when the light casts no shadow.
    pub shadow_vp: [[f32; 4]; 4],
}

/// Typed evaluation kind of a [`GpuLight`] entry (mirrors `LIGHT_KIND_*`
/// scalars carried in `kind.x`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum GpuLightKind {
    /// Directional light (`0.0`).
    Directional,
    /// Point light (`1.0`).
    Point,
    /// Spotlight (`2.0`).
    Spot,
}

impl GpuLightKind {
    /// Raw kind scalar (`kind.x`).
    #[allow(dead_code)]
    pub(crate) fn as_scalar(self) -> f32 {
        match self {
            Self::Directional => LIGHT_KIND_DIRECTIONAL,
            Self::Point => LIGHT_KIND_POINT,
            Self::Spot => LIGHT_KIND_SPOT,
        }
    }

    /// Classifies a raw kind scalar; `None` for unknown values.
    #[allow(dead_code)]
    pub(crate) fn from_scalar(v: f32) -> Option<Self> {
        if v == LIGHT_KIND_DIRECTIONAL {
            Some(Self::Directional)
        } else if v == LIGHT_KIND_POINT {
            Some(Self::Point)
        } else if v == LIGHT_KIND_SPOT {
            Some(Self::Spot)
        } else {
            None
        }
    }
}

impl GpuLight {
    /// Evaluation kind scalar carried in `kind.x` (0/1/2 =
    /// directional/point/spot, see `LIGHT_KIND_*`).
    #[allow(dead_code)]
    pub(crate) fn kind_scalar(&self) -> f32 {
        self.kind[0]
    }

    /// Typed evaluation kind from `kind.x`; `None` for unknown scalars.
    /// Prefer [`GpuLight::kind_scalar`] only for raw shader debugging.
    #[allow(dead_code)]
    pub(crate) fn kind_units(&self) -> Option<GpuLightKind> {
        GpuLightKind::from_scalar(self.kind[0])
    }

    /// Emission color (RGB, exposure-baked) plus radiometric intensity in
    /// alpha, as linear RGBA.
    #[allow(dead_code)]
    pub(crate) fn color_units(&self) -> ornis_core::units::LinearRgba {
        ornis_core::units::LinearRgba::new(self.color)
    }

    /// Range cutoff carried in `params.x` (point/spot).
    #[allow(dead_code)]
    pub(crate) fn range_cutoff(&self) -> f32 {
        self.params[0]
    }

    /// Range cutoff in meters (point/spot).
    #[allow(dead_code)]
    pub(crate) fn range_units(&self) -> ornis_core::units::Meters {
        ornis_core::units::Meters::new(self.params[0])
    }
}

/// Which term the deferred lighting pass writes.
///
/// [`ShadingDebug::Beauty`] is the default and leaves the image unchanged.
/// Other variants replace the lit color so a capture can show one lobe.
/// Forward draws ignore the selector. The id is the `debug_view` field of
/// the lighting uniform; [`Renderer3D::set_shading_debug`] stores it across
/// [`Renderer3D::set_lights`] calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum ShadingDebug {
    /// Full lighting: ambient, direct, emission, IBL.
    #[default]
    Beauty = 0,
    /// Direct base specular lobe only (no diffuse, no IBL, no ambient).
    DirectSpecular = 1,
    /// Split-sum IBL specular only. Black when the IBL weight is 0.
    IblSpecular = 2,
    /// Direct diffuse plus ambient plus IBL diffuse.
    Diffuse = 3,
    /// Perceptual specular roughness in every channel (after geometric AA).
    Roughness = 4,
    /// Metalness in every channel.
    Metallic = 5,
    /// Shadow visibility in every channel. `1` when no light casts a map.
    Shadow = 6,
    /// Pre-tonemap heat: magenta if non-finite, red if luma exceeds 1,
    /// otherwise the beauty color.
    Heat = 7,
}

/// One cascade light-space VP: mat4 wrapper so [`CsmBlock`] can spell an
/// array of VPs (the derive accepts structs with a WGSL layout, not
/// nested raw arrays, as array elements). `align(16)` satisfies the
/// derive's nested-struct precondition (cf. [`GpuLight`]).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "CsmVp")]
pub(crate) struct CsmVp {
    /// Column-major clip matrix.
    pub(super) cols: [[f32; 4]; 4],
}

/// Cascaded-shadow data for a single directional light (#121).
///
/// The WGSL `Csm` declaration is generated from this layout
/// ([`CsmBlock::WGSL_SOURCE`]); the field order keeps every `vec4` on a
/// 16-byte boundary (cf. [`GpuLight`]) and the trailing pad rounds the
/// block size up to a multiple of 16.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "Csm")]
pub(crate) struct CsmBlock {
    /// One light-space VP per cascade (index = CSM layer).
    #[wgsl(to = "CsmVp")]
    pub(super) vps: [CsmVp; 4],
    /// Camera view direction (`w` = 0): cascade select uses view depth.
    pub(super) view_fwd: [f32; 4],
    /// Slice boundaries as view depths: camera near, then the first three
    /// splits (spells `vec4f`; the far edge rides in [`split_far`](Self::split_far)).
    pub(super) splits_lo: [f32; 4],
    /// Far edge of the last cascade (camera far).
    pub(super) split_far: f32,
    /// Active cascade count (`0` = CSM off, legacy path).
    pub(super) cascade_count: u32,
    /// Trailing pad: keeps the block size (304 bytes) a multiple of 16.
    pub(super) _pad: [u32; 2],
}

/// Compile-time pin: the WGSL derive only accepts integer literals for
/// array lengths, so [`CsmBlock::vps`] spells `4` literally — this assert
/// keeps the spell and the limit in sync (cf. [`MAX_LIGHTS`]).
const _: [(); CSM_CASCADES] = [(); 4];

/// Camera description for CSM cascade fitting (#121), stored by
/// [`Renderer3D::set_csm_camera`](super::Renderer3D::set_csm_camera).
/// CSM stays off until the backend provides it, so scenes that never opt
/// in render pixel-identical to the legacy rig.
#[derive(Debug, Clone, Copy)]
pub struct CsmCamera {
    /// Near view depth in meters (> 0).
    pub near: f32,
    /// Far view depth in meters (> `near`).
    pub far: f32,
    /// Camera view direction in world space (normalized at use).
    pub forward: [f32; 3],
}

/// Lighting uniform block: ambient + fixed light array + count.
///
/// The WGSL `Lighting` declaration is generated from this layout
/// ([`LightingUniform::WGSL_SOURCE`]); the field list here is the single
/// source of truth for the buffer layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "Lighting")]
pub(crate) struct LightingUniform {
    pub(super) ambient_color: [f32; 4],
    /// Eight lights; spelled `array<Light, 8>` in WGSL (the inner struct's
    /// `name` override is not visible here, hence the explicit `to`).
    /// The evaluators iterate `0..light_count`, so directional-only scenes
    /// with ≤4 lights render pixel-identical to the old 4-wide block.
    #[wgsl(to = "Light")]
    pub(super) lights: [GpuLight; 8],
    pub(super) light_count: u32,
    /// Split-sum IBL weight. `0` (default) keeps the direct-light result.
    pub(super) ibl_weight: f32,
    /// Highest mip of the prefiltered specular cube (`0` when IBL is off).
    pub(super) ibl_max_mip: f32,
    /// [`ShadingDebug`] discriminant. `0` is beauty.
    pub(super) debug_view: u32,
    /// Cascaded shadow data for one directional light (#121); zeroed
    /// (inactive) unless [`Renderer3D::set_lights_full`] fits cascades.
    /// The evaluators take the CSM path only for the directional entry
    /// whose `params.x` carries the cascade count — everything else stays
    /// pixel-identical to the pre-CSM block.
    #[wgsl(to = "Csm")]
    pub(super) csm: CsmBlock,
}

/// Identity clip matrix for lights that cast no shadow.
const NO_SHADOW_VP: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Packs a linear RGB scene color plus radiometric intensity into the GPU
/// `vec4` (RGB exposure-scaled, intensity in alpha).
fn pack_light_color(color: [f32; 3], intensity: f32, exposure: f32) -> [f32; 4] {
    let rgb = ornis_core::units::LinearRgb::new(color).as_array();
    [
        rgb[0] * exposure,
        rgb[1] * exposure,
        rgb[2] * exposure,
        intensity,
    ]
}

/// Pure result of [`build_lighting_uniform`]: the uploadable uniform, the
/// shadow VPs the depth pre-pass publishes (layers in assignment order,
/// cube faces in slot-major order), and the upload report.
pub(super) struct BuiltLighting {
    pub(super) uniform: LightingUniform,
    pub(super) stats: LightUploadStats,
    shadow_layer_vps: Vec<[[f32; 4]; 4]>,
    cube_face_vps: Vec<[[f32; 4]; 4]>,
}

/// Pure [`LightingUniform`] construction shared by
/// [`Renderer3D::set_lights_full`] and [`count_light_drops`]: ambient RGB
/// scaled by `ambient_intensity`, light colors scaled by `exposure`
/// (CPU-baked IBL-minimum multipliers — no shader-layout change; 1.0 is
/// the exact no-op), the first [`MAX_LIGHTS`] entries uploaded, the rest
/// reported as dropped. Directional VPs use the legacy ±
/// [`SHADOW_ORTHO_HALF`] box when `fit` is `None` and the fitted box
/// otherwise. Shadow layer VPs are collected in assignment order (layer
/// `i` ↔ entry `i`). `csm_owner` (an index into `lights`, decided by
/// [`Renderer3D::set_lights_full` once cascades fit) marks the single
/// directional entry whose cascades ride in [`CsmBlock`]: it consumes no
/// generic layer and carries the cascade count in `params.x`.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_lighting_uniform(
    ambient: [f32; 3],
    ambient_intensity: f32,
    exposure: f32,
    lights: &[LightDesc],
    fit: Option<([f32; 3], f32)>,
    csm_owner: Option<usize>,
    ibl_weight: f32,
    ibl_max_mip: f32,
    debug_view: u32,
) -> BuiltLighting {
    /// Normalize a direction, falling back to +Z on degenerate input.
    fn norm_dir(d: [f32; VEC3_COMPONENTS]) -> [f32; VEC4_COMPONENTS] {
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len > 0.0 {
            [d[0] / len, d[1] / len, d[2] / len, 0.0]
        } else {
            [0.0, 0.0, 1.0, 0.0]
        }
    }
    let count = lights.len().min(MAX_LIGHTS);
    let mut gpu_lights = [GpuLight {
        kind: [LIGHT_KIND_DIRECTIONAL, 0.0, 0.0, 0.0],
        direction: [0.0, 0.0, 1.0, 0.0],
        position: [0.0, 0.0, 0.0, 1.0],
        color: [0.0; VEC4_COMPONENTS],
        params: [0.0, 0.0, 0.0, -1.0],
        shadow_vp: NO_SHADOW_VP,
    }; MAX_LIGHTS];
    let mut shadow_count = 0u32;
    let mut cube_count = 0u32;
    let mut dropped_shadows = 0u32;
    let mut shadow_layer_vps: Vec<[[f32; VEC4_COMPONENTS]; VEC4_COMPONENTS]> = Vec::new();
    // (position, range, cube slot) for shadowed point lights, in
    // assignment order; face VPs are derived below.
    let mut cube_lights: Vec<([f32; VEC3_COMPONENTS], f32, usize)> = Vec::new();
    /// Assign the next shadow layer, or -1.0 when `wants` is false
    /// or the array is full. Returns `(layer, clip_matrix)`.
    macro_rules! shadow_layer {
        ($wants:expr, $vp:expr) => {{
            let wants: bool = $wants;
            if wants && (shadow_count as usize) < SHADOW_LAYERS {
                let layer = shadow_count;
                shadow_count += 1;
                (layer as f32, $vp)
            } else {
                (-1.0, NO_SHADOW_VP)
            }
        }};
    }
    for (i, light) in lights.iter().take(count).enumerate() {
        gpu_lights[i] = match light {
            LightDesc::Directional {
                direction,
                intensity,
                color,
                shadow,
            } => {
                // Typed descriptor fields → raw wire values for the GPU pack.
                // Already unit (normalized once on load/construction):
                // used as-is so results match the legacy `norm3(raw)`.
                let to_light = direction.get();
                // CSM owner (see `set_lights_full`): cascades ride in
                // `CsmBlock`, no generic layer is consumed.
                let csm = csm_owner == Some(i) && shadow.is_enabled();
                let vp = match fit {
                    None => dir_shadow_vp(to_light),
                    Some((center, half)) => dir_shadow_vp_fitted(to_light, center, half),
                };
                let (layer, vp) = shadow_layer!(shadow.is_enabled() && !csm, vp);
                if shadow.is_enabled() && !csm && layer < 0.0 {
                    dropped_shadows += 1;
                }
                if layer >= 0.0 {
                    shadow_layer_vps.push(vp);
                }
                GpuLight {
                    direction: norm_dir(direction.as_array()),
                    color: pack_light_color(*color, *intensity, exposure),
                    params: [if csm { CSM_CASCADES as f32 } else { 0.0 }, 0.0, 0.0, layer],
                    shadow_vp: vp,
                    ..gpu_lights[i]
                }
            }
            LightDesc::Point {
                position,
                intensity,
                color,
                range,
                shadow,
            } => {
                let (position, range) = (&position.to_array(), &range.get());
                // Cube slots live in a separate index space from the
                // 2D layers (the evaluator picks the pool by kind).
                let slot = if shadow.is_enabled() && (cube_count as usize) < POINT_SHADOW_CUBES {
                    let s = cube_count as usize;
                    cube_count += 1;
                    cube_lights.push((*position, *range, s));
                    s as f32
                } else {
                    -1.0
                };
                if shadow.is_enabled() && slot < 0.0 {
                    dropped_shadows += 1;
                }
                GpuLight {
                    kind: [LIGHT_KIND_POINT, 0.0, 0.0, 0.0],
                    position: [position[0], position[1], position[2], 1.0],
                    color: pack_light_color(*color, *intensity, exposure),
                    params: [range.max(LIGHT_RANGE_EPS), 0.0, 0.0, slot],
                    ..gpu_lights[i]
                }
            }
            LightDesc::Spot {
                position,
                direction,
                intensity,
                color,
                range,
                inner_angle,
                outer_angle,
                shadow,
            } => {
                let axis = direction.get();
                let (position, direction, range) =
                    (&position.to_array(), &direction.as_array(), &range.get());
                let (inner_angle, outer_angle) = (&inner_angle.get(), &outer_angle.get());
                // Cosine-ordered: inner must be the tighter cone.
                let ci = ornis_core::units::Degrees::new(*inner_angle)
                    .to_radians()
                    .get()
                    .cos();
                let co = ornis_core::units::Degrees::new(*outer_angle)
                    .to_radians()
                    .get()
                    .cos();
                let (layer, vp) = shadow_layer!(
                    shadow.is_enabled(),
                    spot_shadow_vp(*position, axis, *outer_angle, *range)
                );
                if shadow.is_enabled() && layer < 0.0 {
                    dropped_shadows += 1;
                }
                if layer >= 0.0 {
                    shadow_layer_vps.push(vp);
                }
                GpuLight {
                    kind: [LIGHT_KIND_SPOT, 0.0, 0.0, 0.0],
                    direction: norm_dir(*direction),
                    position: [position[0], position[1], position[2], 1.0],
                    color: pack_light_color(*color, *intensity, exposure),
                    params: [range.max(LIGHT_RANGE_EPS), ci.max(co), co.min(ci), layer],
                    shadow_vp: vp,
                }
            }
        };
    }
    // Excess lights beyond the upload window never reach the GPU; a
    // shadow request on one is a dropped shadow too.
    let mut dropped_lights = 0u32;
    for light in lights.iter().skip(count) {
        dropped_lights += 1;
        if light.shadow_cast().is_enabled() {
            dropped_shadows += 1;
        }
    }
    let mut cube_face_vps = Vec::with_capacity(cube_lights.len() * CUBE_FACE_COUNT);
    for (position, range, _) in &cube_lights {
        for face in 0..CUBE_FACE_COUNT {
            cube_face_vps.push(point_cube_face_vp(*position, *range, face));
        }
    }
    BuiltLighting {
        uniform: LightingUniform {
            ambient_color: [
                ambient[0] * ambient_intensity,
                ambient[1] * ambient_intensity,
                ambient[2] * ambient_intensity,
                1.0,
            ],
            lights: gpu_lights,
            light_count: count as u32,
            ibl_weight,
            ibl_max_mip,
            debug_view,
            // Cascades patch in later (see `set_lights_full`); the pure
            // preview path keeps the zeroed (inactive) block.
            csm: bytemuck::Zeroable::zeroed(),
        },
        stats: LightUploadStats {
            uploaded: count as u32,
            dropped_lights,
            dropped_shadows,
        },
        shadow_layer_vps,
        cube_face_vps,
    }
}

/// Pack the point/spot lights beyond the uniform window into cluster
/// storage records (#130). Head lights (`lights[..head_count]`) already
/// live in the uniform window and are written to storage offset 0 by
/// [`Renderer3D::set_lights_full`] byte-identical; these records ride
/// after them and reach pixels only through the cluster lists.
/// Directionals beyond the window stay dropped (only positioned lights
/// cluster), and clustered lights carry no shadow slot — the tiny
/// point/spot shadow pools stay with the uniform window; a shadow atlas
/// is follow-up work. Returns the records plus how many point/spot
/// lights did not fit the storage capacity.
pub(super) fn pack_cluster_light_records(
    lights: &[LightDesc],
    head_count: usize,
    exposure: f32,
    capacity: usize,
) -> (Vec<GpuLight>, u32) {
    let mut records = Vec::new();
    let mut dropped = 0u32;
    for light in lights.iter().skip(head_count) {
        let record = match light {
            LightDesc::Directional { .. } => None,
            LightDesc::Point {
                position,
                intensity,
                color,
                range,
                ..
            } => Some(GpuLight {
                kind: [LIGHT_KIND_POINT, 0.0, 0.0, 0.0],
                direction: [0.0, 0.0, 1.0, 0.0],
                position: [position.x, position.y, position.z, 1.0],
                color: pack_light_color(*color, *intensity, exposure),
                params: [range.get().max(LIGHT_RANGE_EPS), 0.0, 0.0, -1.0],
                shadow_vp: NO_SHADOW_VP,
            }),
            LightDesc::Spot {
                position,
                direction,
                intensity,
                color,
                range,
                inner_angle,
                outer_angle,
                ..
            } => {
                // Cosine-ordered like the uniform window: inner is the
                // tighter cone (bigger cosine).
                let axis = direction.get();
                let ci = inner_angle.get().to_radians().cos();
                let co = outer_angle.get().to_radians().cos();
                Some(GpuLight {
                    kind: [LIGHT_KIND_SPOT, 0.0, 0.0, 0.0],
                    direction: [axis.x, axis.y, axis.z, 0.0],
                    position: [position.x, position.y, position.z, 1.0],
                    color: pack_light_color(*color, *intensity, exposure),
                    params: [
                        range.get().max(LIGHT_RANGE_EPS),
                        ci.max(co),
                        co.min(ci),
                        -1.0,
                    ],
                    shadow_vp: NO_SHADOW_VP,
                })
            }
        };
        match record {
            Some(record) if records.len() < capacity => records.push(record),
            Some(_) => dropped += 1,
            None => {}
        }
    }
    (records, dropped)
}

/// Index of the first shadowed directional light in upload order, the
/// single entry eligible to own the CSM cascades (`None` when absent).
fn first_shadowed_directional_index(lights: &[LightDesc]) -> Option<usize> {
    lights.iter().take(MAX_LIGHTS).position(|light| {
        matches!(
            light,
            LightDesc::Directional { shadow, .. } if shadow.is_enabled()
        )
    })
}

/// CSM upload for one directional light, or `None` when the camera
/// description is degenerate (CSM stays off, legacy path applies).
///
/// Splits the `[near, far]` range (practical split), fits one snapped
/// ortho box per slice, and packs the VPs with the slice boundaries.
/// `casters` widens every cascade's depth to the scene AABB (same box
/// [`shadow_fit_for_bounds`](super::shadow_fit_for_bounds) describes).
pub(super) fn build_csm(
    inv_view_proj: [[f32; 4]; 4],
    camera: CsmCamera,
    to_light: glam::Vec3,
    casters: Option<([f32; 3], [f32; 3])>,
) -> Option<CsmBlock> {
    // facade: validate the camera, split the range, fit every slice.
    if !(camera.near > 0.0 && camera.far > camera.near) {
        return None;
    }
    if !camera.near.is_finite() || !camera.far.is_finite() {
        return None;
    }
    let fwd = glam::Vec3::from_array(camera.forward);
    if !fwd.is_finite() || fwd.length_squared() <= f32::EPSILON {
        return None;
    }
    let splits = cascade_splits(camera.near, camera.far, CSM_CASCADES, CSM_SPLIT_LAMBDA);
    // Depth spans the whole frustum so casters outside a slice still
    // darken it (see `cascade_shadow_vp`).
    let frustum = frustum_slice_corners(
        inv_view_proj,
        camera.near,
        camera.far,
        camera.near,
        camera.far,
    )?;
    let mut vps = [CsmVp { cols: NO_SHADOW_VP }; CSM_CASCADES];
    for (i, pair) in splits.windows(2).enumerate() {
        let corners =
            frustum_slice_corners(inv_view_proj, camera.near, camera.far, pair[0], pair[1])?;
        vps[i] = CsmVp {
            cols: cascade_shadow_vp(to_light, corners, frustum, casters, SHADOW_SIZE),
        };
    }
    let fwd = fwd.normalize();
    Some(CsmBlock {
        vps,
        view_fwd: [fwd.x, fwd.y, fwd.z, 0.0],
        splits_lo: [splits[0], splits[1], splits[2], splits[3]],
        split_far: splits[CSM_CASCADES],
        cascade_count: CSM_CASCADES as u32,
        _pad: [0; 2],
    })
}

/// Pure preview of what [`Renderer3D::set_lights`] would drop for
/// `lights`: excess beyond [`MAX_LIGHTS`] plus shadow requests without a
/// free layer/cube slot. No GPU access — safe to call per frame; log on
/// scene change, not per frame.
pub fn count_light_drops(lights: &[LightDesc]) -> LightUploadStats {
    build_lighting_uniform(
        [0.0; VEC3_COMPONENTS],
        1.0,
        1.0,
        lights,
        None,
        None,
        0.0,
        0.0,
        0,
    )
    .stats
}

impl Renderer3D {
    /// Upload ambient RGB plus up to eight scene lights of any kind
    /// ([`MAX_LIGHTS`]); excess lights are dropped and the drop is
    /// published via [`light_upload_stats`](Self::light_upload_stats),
    /// never silently. Directionals map exactly as before, so
    /// directional-only scenes render pixel-identical to the legacy rig.
    /// Identical to [`set_lights_full`](Self::set_lights_full) with
    /// `ambient_intensity = 1.0` and `exposure = [`exposure`](Self::exposure)
    /// (`1.0` by default — the exact no-op).
    ///
    /// Shadowed lights (directional/spot with `shadow: ShadowCast::Enabled`) are
    /// assigned map layers `0..shadow_count` (`params.w`); their
    /// light-space clip matrices go both into `GpuLight::shadow_vp`
    /// (sampled by the evaluators) and into the per-layer VP uniform
    /// buffers the depth pre-pass reuses through the gbuffer vertex
    /// shader's `camera` slot. Directional VPs use the legacy
    /// ±[`SHADOW_ORTHO_HALF`] box around the origin unless a scene fit
    /// was set via [`set_shadow_bounds`](Self::set_shadow_bounds).
    pub fn set_lights(&self, queue: &wgpu::Queue, ambient: [f32; 3], lights: &[LightDesc]) {
        self.set_lights_full(queue, ambient, 1.0, self.exposure, lights);
    }

    /// Publish CSM cascade VPs for the depth pre-pass (as camera
    /// uniforms, like the 2D layers) and the active cascade count.
    fn publish_csm_buffers(&self, queue: &wgpu::Queue, block: CsmBlock) {
        for (layer, vp) in block.vps.iter().enumerate() {
            let uniform = CameraUniform {
                view_proj: vp.cols,
                inv_view_proj: NO_SHADOW_VP,
                camera_pos: [0.0, 0.0, 0.0, 1.0],
            };
            queue.write_buffer(&self.csm_vp_buffers[layer], 0, bytemuck::bytes_of(&uniform));
        }
        self.csm_count
            .store(CSM_CASCADES as u32, std::sync::atomic::Ordering::Relaxed);
    }

    /// CSM upload for the first shadowed directional light, or `None`
    /// when the backend never called [`set_csm_camera`](Self::set_csm_camera)
    /// or the fit degenerates (CSM stays off, legacy path applies).
    fn csm_upload_for(&self, lights: &[LightDesc]) -> Option<(usize, CsmBlock)> {
        let camera = (*read_lock(&self.csm_camera))?;
        let inv_view_proj = read_lock(&self.camera_cpu).inv_view_proj;
        let index = first_shadowed_directional_index(lights)?;
        let LightDesc::Directional { direction, .. } = &lights[index] else {
            return None;
        };
        let casters = self.csm_caster_aabb();
        build_csm(inv_view_proj, camera, direction.get(), casters).map(|block| (index, block))
    }

    /// [`set_lights`](Self::set_lights) with IBL-minimum multipliers and
    /// an explicit upload report. `ambient_intensity` scales the ambient
    /// RGB and `exposure` scales every light color; both are baked on the
    /// CPU (no shader-layout change) and both are the exact no-op at
    /// `1.0`. Returns the [`LightUploadStats`] for this upload and
    /// publishes it for [`light_upload_stats`](Self::light_upload_stats).
    pub fn set_lights_full(
        &self,
        queue: &wgpu::Queue,
        ambient: [f32; 3],
        ambient_intensity: f32,
        exposure: f32,
        lights: &[LightDesc],
    ) -> LightUploadStats {
        let fit = *read_lock(&self.shadow_fit);
        let ibl_weight = f32::from_bits(
            self.ibl_weight_bits
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        let ibl_max_mip = f32::from_bits(
            self.ibl_max_mip_bits
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        let debug_view = self
            .shading_debug
            .load(std::sync::atomic::Ordering::Relaxed);
        // CSM first: the owner must be known before the pack assigns
        // generic layers (the owner consumes none). Without a fitted
        // block the pack keeps the legacy path exactly.
        let csm = self.csm_upload_for(lights);
        let csm_owner = csm.as_ref().map(|(index, _)| *index);
        let mut built = build_lighting_uniform(
            ambient,
            ambient_intensity,
            exposure,
            lights,
            fit,
            csm_owner,
            ibl_weight,
            ibl_max_mip,
            debug_view,
        );
        if let Some((_, block)) = csm {
            built.uniform.csm = block;
            self.publish_csm_buffers(queue, block);
        } else {
            self.csm_count
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&built.uniform));
        // Cluster-side upload (#130): the uniform-window records go to
        // cluster storage offset 0 byte-identical to `lighting.lights`
        // (the unified shading window reads them while clustering is
        // off, so the off path is value-identical to the legacy loop);
        // clustered point/spot records ride after them.
        let clustered_on = self
            .clustered_enabled
            .load(std::sync::atomic::Ordering::Relaxed);
        let head_count = built.uniform.light_count as usize;
        let head = &built.uniform.lights[..head_count];
        let capacity = (MAX_CLUSTERED_LIGHTS as usize).saturating_sub(head_count);
        let (cluster_records, dropped) = if clustered_on {
            pack_cluster_light_records(lights, head_count, exposure, capacity)
        } else {
            (Vec::new(), 0)
        };
        let params = ClusterParams {
            light_count: head_count as u32 + cluster_records.len() as u32,
            head_count: head_count as u32,
            clustered: u32::from(clustered_on),
            _pad: 0,
        };
        self.clusters
            .write_frame(queue, &params, head, &cluster_records);
        *write_lock(&self.last_cluster_stats) = ClusterLightStats {
            clustered: cluster_records.len() as u32,
            dropped,
        };
        // Publish the light-space VPs for the depth pre-pass (as camera
        // uniforms: the shadow pipeline reuses the gbuffer vertex shader,
        // which only reads `view_proj`) and the active layer count.
        // Layers publish in assignment order (layer `i` ↔ entry `i`).
        for (layer, vp) in built.shadow_layer_vps.iter().enumerate() {
            let uniform = CameraUniform {
                view_proj: *vp,
                inv_view_proj: NO_SHADOW_VP,
                camera_pos: [0.0, 0.0, 0.0, 1.0],
            };
            queue.write_buffer(
                &self.shadow_vp_buffers[layer],
                0,
                bytemuck::bytes_of(&uniform),
            );
        }
        self.shadow_count.store(
            built.shadow_layer_vps.len() as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        // Publish the cube-face VPs (as camera uniforms, like the 2D
        // layers) and the active cube count.
        for (index, vp) in built.cube_face_vps.iter().enumerate() {
            let uniform = CameraUniform {
                view_proj: *vp,
                inv_view_proj: NO_SHADOW_VP,
                camera_pos: [0.0, 0.0, 0.0, 1.0],
            };
            queue.write_buffer(
                &self.shadow_cube_vp_buffers[index],
                0,
                bytemuck::bytes_of(&uniform),
            );
        }
        self.point_shadow_count.store(
            (built.cube_face_vps.len() / CUBE_FACE_COUNT) as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        *write_lock(&self.last_light_stats) = built.stats;
        built.stats
    }

    /// Upload report of the last [`set_lights`](Self::set_lights) /
    /// [`set_lights_full`](Self::set_lights_full) call: what was uploaded
    /// and what was dropped (excess lights, shadow requests without a
    /// slot). Starts at zero before the first upload.
    pub fn light_upload_stats(&self) -> LightUploadStats {
        *read_lock(&self.last_light_stats)
    }

    /// Opt into (or out of) clustered point/spot lighting (#130).
    /// Takes effect at the next [`set_lights`](Self::set_lights) call,
    /// which moves point/spot lights beyond the uniform window
    /// ([`MAX_LIGHTS`]) into the cluster storage; every lit pass then
    /// dispatches the per-cluster assignment compute pass first. Off
    /// (the default) keeps every frame bit-identical to the legacy
    /// uniform window: the shading window multiplies the cluster span
    /// by this flag, so not even a stale grid can leak a light.
    pub fn set_clustered_lights(&self, enabled: bool) {
        self.clustered_enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether clustered point/spot lighting is enabled (see
    /// [`set_clustered_lights`](Self::set_clustered_lights)).
    pub fn clustered_lights_enabled(&self) -> bool {
        self.clustered_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Cluster-side report of the last [`set_lights`](Self::set_lights)
    /// call: point/spot lights uploaded into the cluster storage beyond
    /// the uniform window, and how many were dropped for storage
    /// capacity. Zeroed while clustering is off.
    pub fn cluster_light_stats(&self) -> ClusterLightStats {
        *read_lock(&self.last_cluster_stats)
    }

    /// Record the cluster assignment pass on `encoder` while clustering
    /// is enabled; a no-op otherwise. Called by every lit pass — a frame
    /// that records both the deferred lighting and a forward pass
    /// dispatches twice; the second run rewrites identical results (the
    /// span cursor resets per dispatch), so no frame-state guard is
    /// needed.
    pub(super) fn dispatch_light_clusters(&self, encoder: &mut wgpu::CommandEncoder) {
        if self
            .clustered_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            self.clusters.dispatch(encoder);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_util::*;
    use super::super::*;
    use super::*;
    #[test]
    fn lighting_uniform_spells_eight_lights() {
        // Multipliers stay CPU-baked. `debug_view` is the only added
        // shader-visible field; it occupies the former trailing pad.
        assert!(
            LightingUniform::WGSL_SOURCE.contains("array<Light, 8>"),
            "{}",
            LightingUniform::WGSL_SOURCE
        );
        assert!(LightingUniform::WGSL_SOURCE.contains("debug_view: u32"));
        assert!(LightingUniform::WGSL_SOURCE.contains("csm: Csm"));
        assert_eq!(
            LightingUniform::FIELD_NAMES,
            &[
                "ambient_color",
                "lights",
                "light_count",
                "ibl_weight",
                "ibl_max_mip",
                "debug_view",
                "csm"
            ]
        );
        assert_eq!(
            std::mem::offset_of!(LightingUniform, debug_view)
                - std::mem::offset_of!(LightingUniform, ibl_max_mip),
            4
        );
        assert_eq!(std::mem::size_of::<CsmBlock>(), 304);
        assert_eq!(std::mem::size_of::<CsmBlock>() % 16, 0);
        assert_eq!(std::mem::offset_of!(CsmBlock, view_fwd) % 16, 0);
        let built = build_lighting_uniform(
            [0.0; VEC3_COMPONENTS],
            1.0,
            1.0,
            &[],
            None,
            None,
            0.0,
            0.0,
            ShadingDebug::Shadow as u32,
        );
        assert_eq!(built.uniform.debug_view, ShadingDebug::Shadow as u32);
        assert_eq!(ShadingDebug::Beauty as u32, 0);
        assert_eq!(built.uniform.csm.cascade_count, 0);
    }

    #[test]
    fn ten_lights_upload_eight_and_drop_two() {
        let lights: Vec<LightDesc> = (0..10)
            .map(|_| dir_probe([1.0, 1.0, 1.0], ornis_assets::scene::ShadowCast::Disabled))
            .collect();
        let built =
            build_lighting_uniform([0.1, 0.1, 0.15], 1.0, 1.0, &lights, None, None, 0.0, 0.0, 0);
        assert_eq!(
            built.stats,
            LightUploadStats {
                uploaded: 8,
                dropped_lights: 2,
                dropped_shadows: 0,
            }
        );
        assert_eq!(built.uniform.light_count, 8);
        // The public preview agrees with the upload path (single logic).
        assert_eq!(count_light_drops(&lights), built.stats);
    }

    #[test]
    fn csm_owner_skips_generic_layer_and_flags_cascades() {
        let lights = [dir_probe(
            [1.0, 1.0, 1.0],
            ornis_assets::scene::ShadowCast::Enabled,
        )];
        let legacy =
            build_lighting_uniform([0.1, 0.1, 0.15], 1.0, 1.0, &lights, None, None, 0.0, 0.0, 0);
        assert_eq!(legacy.uniform.lights[0].params[3], 0.0);
        assert_eq!(legacy.stats.dropped_shadows, 0);
        let csm = build_lighting_uniform(
            [0.1, 0.1, 0.15],
            1.0,
            1.0,
            &lights,
            None,
            Some(0),
            0.0,
            0.0,
            0,
        );
        assert_eq!(csm.uniform.lights[0].params[3], -1.0);
        assert_eq!(csm.uniform.lights[0].params[0], CSM_CASCADES as f32);
        assert_eq!(csm.stats.dropped_shadows, 0);
    }

    #[test]
    fn build_csm_rejects_degenerate_camera() {
        let inv = glam::Mat4::IDENTITY.to_cols_array_2d();
        let to_light = glam::Vec3::Y;
        let ok = CsmCamera {
            near: 0.1,
            far: 100.0,
            forward: [0.0, 0.0, -1.0],
        };
        // Identity inverse VP still fits (all corners unproject validly).
        assert!(build_csm(inv, ok, to_light, None).is_some());
        assert!(build_csm(inv, CsmCamera { far: 0.1, ..ok }, to_light, None).is_none());
        assert!(
            build_csm(
                inv,
                CsmCamera {
                    forward: [0.0, 0.0, 0.0],
                    ..ok
                },
                to_light,
                None,
            )
            .is_none()
        );
    }
}
