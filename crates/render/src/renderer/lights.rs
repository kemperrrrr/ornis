//! Scene lighting upload: the `GpuLight` pack, the `LightingUniform` block built by `build_lighting_uniform`, and the `set_lights` entry points with their drop accounting.
use super::shadows::{
    CUBE_FACE_COUNT, dir_shadow_vp, dir_shadow_vp_fitted, point_cube_face_vp, spot_shadow_vp,
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

/// Maximum lights uploaded per frame: the [`LightingUniform`] WGSL block
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
    /// Scene lights beyond [`MAX_LIGHTS`] that were not uploaded.
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
/// `i` ↔ entry `i`).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_lighting_uniform(
    ambient: [f32; 3],
    ambient_intensity: f32,
    exposure: f32,
    lights: &[LightDesc],
    fit: Option<([f32; 3], f32)>,
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
                let vp = match fit {
                    None => dir_shadow_vp(to_light),
                    Some((center, half)) => dir_shadow_vp_fitted(to_light, center, half),
                };
                let (layer, vp) = shadow_layer!(shadow.is_enabled(), vp);
                if shadow.is_enabled() && layer < 0.0 {
                    dropped_shadows += 1;
                }
                if layer >= 0.0 {
                    shadow_layer_vps.push(vp);
                }
                GpuLight {
                    direction: norm_dir(direction.as_array()),
                    color: pack_light_color(*color, *intensity, exposure),
                    params: [0.0, 0.0, 0.0, layer],
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
                // Cosineordered: inner must be the tighter cone.
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

/// Pure preview of what [`Renderer3D::set_lights`] would drop for
/// `lights`: excess beyond [`MAX_LIGHTS`] plus shadow requests without a
/// free layer/cube slot. No GPU access — safe to call per frame; log on
/// scene change, not per frame.
pub fn count_light_drops(lights: &[LightDesc]) -> LightUploadStats {
    build_lighting_uniform([0.0; VEC3_COMPONENTS], 1.0, 1.0, lights, None, 0.0, 0.0, 0).stats
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
    /// light-space clip matrices go both into [`GpuLight::shadow_vp`]
    /// (sampled by the evaluators) and into the per-layer VP uniform
    /// buffers the depth pre-pass reuses through the gbuffer vertex
    /// shader's `camera` slot. Directional VPs use the legacy
    /// ±[`SHADOW_ORTHO_HALF`] box around the origin unless a scene fit
    /// was set via [`set_shadow_bounds`](Self::set_shadow_bounds).
    pub fn set_lights(&self, queue: &wgpu::Queue, ambient: [f32; 3], lights: &[LightDesc]) {
        self.set_lights_full(queue, ambient, 1.0, self.exposure, lights);
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
        let built = build_lighting_uniform(
            ambient,
            ambient_intensity,
            exposure,
            lights,
            fit,
            ibl_weight,
            ibl_max_mip,
            debug_view,
        );
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&built.uniform));
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
        assert_eq!(
            LightingUniform::FIELD_NAMES,
            &[
                "ambient_color",
                "lights",
                "light_count",
                "ibl_weight",
                "ibl_max_mip",
                "debug_view"
            ]
        );
        assert_eq!(
            std::mem::offset_of!(LightingUniform, debug_view)
                - std::mem::offset_of!(LightingUniform, ibl_max_mip),
            4
        );
        let built = build_lighting_uniform(
            [0.0; VEC3_COMPONENTS],
            1.0,
            1.0,
            &[],
            None,
            0.0,
            0.0,
            ShadingDebug::Shadow as u32,
        );
        assert_eq!(built.uniform.debug_view, ShadingDebug::Shadow as u32);
        assert_eq!(ShadingDebug::Beauty as u32, 0);
    }

    #[test]
    fn ten_lights_upload_eight_and_drop_two() {
        let lights: Vec<LightDesc> = (0..10)
            .map(|_| dir_probe([1.0, 1.0, 1.0], ornis_assets::scene::ShadowCast::Disabled))
            .collect();
        let built = build_lighting_uniform([0.1, 0.1, 0.15], 1.0, 1.0, &lights, None, 0.0, 0.0, 0);
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
}
