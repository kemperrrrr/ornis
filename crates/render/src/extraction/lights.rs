//! Frame lighting resource shared by native and WASM runtimes.

use glam::Vec3;
use serde::Deserialize;
use serde::Serialize;

use crate::renderer::LightUploadStats;
use crate::renderer::count_light_drops;
use ornis_assets::scene::LightDesc;
use ornis_assets::scene::Scene;
use ornis_assets::scene::ShadowCast;
use ornis_core::units::Color;
use ornis_core::units::EnvironmentWeight;
use ornis_core::units::Lux;
use ornis_core::units::UnitVec3;

/// Ambient plus directional lights of the frame as a world resource (X3,
/// Extract-free).
///
/// Written by the scene loader between frames (world `replace_scene`
/// or the platform's equivalent) and only read inside the schedule, so no
/// `Mutex` is needed (same contract as `GpuSurfaceState`). `RenderSubmit`
/// uploads it via [`Self::set_lights_args`] instead of a hardcoded rig.
///
/// The IBL-minimum multipliers ([`Self::ambient_intensity`],
/// [`Self::exposure`]) default to `Lux(1.0)` (exact no-op) and are
/// accepted by old serialized payloads through `serde` defaults. GPU
/// `[f32; …]` / raw `f32` values are produced only at the buffer upload
/// (`Renderer3D::set_lights` / [`Self::set_lights_args`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderLights {
    /// Ambient color. Linear RGB is taken at the GPU upload; alpha is not
    /// part of the lighting uniform.
    pub ambient: Color,
    /// Scene lights of any kind; the renderer uploads the first eight
    /// (see `renderer::MAX_LIGHTS`) and reports the rest via
    /// [`Self::light_upload_stats`].
    ///
    /// Directional entries stay [`LightDesc`] (direction is [`UnitVec3`]).
    /// Asset payloads may still carry legacy color arrays; those arrays
    /// become GPU floats inside `Renderer3D::set_lights`, not here.
    pub lights: Vec<LightDesc>,
    /// IBL-minimum ambient multiplier, baked into the ambient upload by
    /// `Renderer3D::set_lights_full`. Absent in older payloads — defaults
    /// to `Lux(1.0)` (no-op). Not [`Lux::default`], which is zero.
    #[serde(default = "default_ibl_factor")]
    pub ambient_intensity: Lux,
    /// IBL-minimum exposure multiplier applied to every light color, baked
    /// into the light upload by `Renderer3D::set_lights_full`. Absent in
    /// older payloads — defaults to `Lux(1.0)` (no-op).
    #[serde(default = "default_ibl_factor")]
    pub exposure: Lux,
    /// Explicit split-sum weight. `None` (the default, including payloads
    /// that predate the field) leaves the automatic weight: `0` when no
    /// environment cube is bound and `1` when one is. `Some` is copied
    /// into `LightingUniform.ibl_weight` on submit and is not replaced
    /// when a cube is bound or cleared. The setter does not upload a cube.
    #[serde(default)]
    pub environment_weight: Option<EnvironmentWeight>,
}

/// Default IBL-minimum multiplier (`Lux(1.0)`): the exact no-op for the
/// ambient and exposure uploads, and the `serde` default for older payloads.
fn default_ibl_factor() -> Lux {
    Lux::new(1.0)
}

/// The lighting rig `RenderSubmit` hardcoded before X3 — the resource
/// default, so a runtime that never loads a scene renders exactly as it
/// did (gate: zero pixel differences).
const LEGACY_AMBIENT: Color = Color::linear_rgb(0.10, 0.10, 0.15);
/// Legacy key light (`(1, 1, 1)` direction, stored normalized).
fn legacy_key_light() -> LightDesc {
    LightDesc::Directional {
        direction: UnitVec3::normalize(Vec3::ONE).unwrap_or(UnitVec3::Y),
        intensity: 0.6,
        color: [1.0, 1.0, 1.0],
        shadow: ShadowCast::Disabled,
    }
}
/// Legacy fill light (`(-0.5, 0.5, -0.5)` direction, stored normalized).
fn legacy_fill_light() -> LightDesc {
    LightDesc::Directional {
        direction: UnitVec3::normalize(Vec3::new(-0.5, 0.5, -0.5)).unwrap_or(UnitVec3::Y),
        intensity: 0.3,
        color: [0.8, 0.8, 1.0],
        shadow: ShadowCast::Disabled,
    }
}

impl Default for RenderLights {
    fn default() -> Self {
        Self {
            ambient: LEGACY_AMBIENT,
            lights: vec![legacy_key_light(), legacy_fill_light()],
            ambient_intensity: default_ibl_factor(),
            exposure: default_ibl_factor(),
            environment_weight: None,
        }
    }
}

impl RenderLights {
    /// The lighting of a serialized scene as the resource (X3).
    ///
    /// Logs a one-line drop report to stderr when the scene exceeds the
    /// renderer limits (lights or shadow slots) — once per scene load,
    /// never per frame. Per-frame callers stay silent and read
    /// [`Self::light_upload_stats`] instead.
    pub fn from_scene(scene: &Scene) -> Self {
        let stats = count_light_drops(&scene.lights);
        if stats.dropped_lights > 0 || stats.dropped_shadows > 0 {
            eprintln!(
                "RenderLights: scene '{}' drops {} light(s) and {} shadow(s) \
                 (renderer limits: 8 lights, 4 shadow layers, 2 point cubes)",
                scene.name, stats.dropped_lights, stats.dropped_shadows
            );
        }
        Self {
            ambient: Color::from(scene.ambient_units()),
            lights: scene.lights.clone(),
            ambient_intensity: default_ibl_factor(),
            exposure: default_ibl_factor(),
            environment_weight: None,
        }
    }

    /// Converts the lights into `Renderer3D::set_lights` arguments —
    /// the scene's [`LightDesc`] list as-is (up to eight entries; the
    /// renderer uploads the rest through the drop report, same as before).
    /// Deliberately silent: the per-frame upload path must not log — drops
    /// are reported once per scene load ([`Self::from_scene`]) and via
    /// [`Self::light_upload_stats`].
    pub fn set_lights_args(&self) -> Vec<LightDesc> {
        self.lights.clone()
    }

    /// Pure preview of what the renderer would drop for this resource:
    /// excess lights beyond the upload window plus shadow requests without
    /// a free layer/cube slot. No GPU access — safe to call per frame.
    pub fn light_upload_stats(&self) -> LightUploadStats {
        count_light_drops(&self.lights)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use ornis_assets::scene::LightDesc;
    use ornis_assets::scene::ShadowCast;
    use ornis_core::units::Color;
    use ornis_core::units::Lux;
    use ornis_core::units::UnitVec3;

    #[test]
    fn default_lights_reproduce_the_legacy_hardcoded_rig() {
        // X3: `RenderSubmit` no longer inlines a lighting rig — the
        // resource default must be exactly the old hardcoded arguments
        // (gate: zero pixel differences).
        let rig = RenderLights::default();
        assert_eq!(rig.ambient, Color::linear_rgb(0.10, 0.10, 0.15));
        assert_eq!(
            rig.ambient.to_linear_rgb().as_array(),
            [0.10, 0.10, 0.15],
            "legacy ambient stays the same linear RGB at the upload"
        );
        assert_eq!(rig.ambient_intensity, Lux::new(1.0));
        assert_eq!(rig.exposure, Lux::new(1.0));
        // Directions are stored normalized (`UnitVec3`); the evaluators
        // normalize anyway, so the lit result is unchanged.
        let key_dir = UnitVec3::normalize(Vec3::ONE).expect("non-zero");
        let fill_dir = UnitVec3::normalize(Vec3::new(-0.5, 0.5, -0.5)).expect("non-zero");
        assert!(matches!(
            rig.set_lights_args().as_slice(),
            [
                LightDesc::Directional {
                    direction: key_d,
                    intensity: key,
                    color: [1.0, 1.0, 1.0],
                    shadow: ShadowCast::Disabled,
                },
                LightDesc::Directional {
                    direction: fill_d,
                    intensity: fill,
                    color: [0.8, 0.8, 1.0],
                    shadow: ShadowCast::Disabled,
                },
            ] if *key == 0.6 && *fill == 0.3 && *key_d == key_dir && *fill_d == fill_dir
        ));
    }

    #[test]
    fn old_render_lights_payload_defaults_ibl_multipliers() {
        // Older resource payloads omit the IBL multipliers. `Lux::default`
        // is zero, so the serde default must stay the no-op `Lux(1.0)`.
        // Ambient keeps the linear RGB tuple scene files already use.
        let parsed: RenderLights =
            ron::de::from_str("(ambient: (0.10, 0.10, 0.15), lights: [])").expect("old payload");
        assert_eq!(parsed.ambient, Color::linear_rgb(0.10, 0.10, 0.15));
        assert_eq!(parsed.ambient_intensity, Lux::new(1.0));
        assert_eq!(parsed.exposure, Lux::new(1.0));
        assert!(parsed.lights.is_empty());

        let round = ron::ser::to_string(&parsed).expect("serialize");
        let again: RenderLights = ron::de::from_str(&round).expect("round trip");
        assert_eq!(again.ambient, parsed.ambient);
        assert_eq!(again.ambient_intensity, parsed.ambient_intensity);
        assert_eq!(again.exposure, parsed.exposure);
    }
}
