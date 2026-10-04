//! Typed directional light placed into a [`RenderLights`](crate::extraction::RenderLights) rig.
//!
//! The channels match a scene `Directional` light: [`UnitVec3`] points
//! toward the light, [`Lux`] is the existing intensity `f32`, and [`Color`]
//! is linear emission. [`DirectionalLight::to_light_desc`] is what the
//! renderer uploads.

use glam::Vec3;
use ornis_assets::scene::{LightDesc, ShadowCast};
use ornis_core::{Color, Lux, UnitVec3};

/// One infinitely distant light.
///
/// `..Default::default()` fills [`Self::shadow`] (off) and, when a field is
/// omitted, the legacy key direction, [`Self::DEFAULT_ILLUMINANCE`], and
/// [`Color::WHITE`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalLight {
    /// Direction toward the light, from the scene.
    pub direction: UnitVec3,
    /// Radiometric strength. Same number as scene `intensity`.
    pub illuminance: Lux,
    /// Linear emission color.
    pub color: Color,
    /// Shadow-map request ([`ShadowCast`]). Default is off, matching older scene files.
    pub shadow: ShadowCast,
}

impl DirectionalLight {
    /// Legacy key-light strength (`0.6`) from the pre-X3 rig.
    pub const DEFAULT_ILLUMINANCE: Lux = Lux(0.6);

    /// Scene light the renderer already knows how to upload.
    pub fn to_light_desc(self) -> LightDesc {
        LightDesc::Directional {
            direction: self.direction,
            intensity: self.illuminance.get(),
            color: self.color.to_linear_rgb().as_array(),
            shadow: self.shadow,
        }
    }
}

impl Default for DirectionalLight {
    /// Legacy key light: direction `(1, 1, 1)` normalized, illuminance
    /// [`Self::DEFAULT_ILLUMINANCE`], white, shadows off.
    fn default() -> Self {
        Self {
            direction: UnitVec3::normalize(Vec3::new(1.0, 1.0, 1.0)).unwrap_or(UnitVec3::Y),
            illuminance: Self::DEFAULT_ILLUMINANCE,
            color: Color::WHITE,
            shadow: ShadowCast::Disabled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_the_legacy_key_light() {
        let light = DirectionalLight::default();
        let desc = light.to_light_desc();
        match desc {
            LightDesc::Directional {
                direction,
                intensity,
                color,
                shadow,
            } => {
                let xyz = direction.get();
                assert!((xyz.length() - 1.0).abs() < 1e-5);
                assert!(xyz.x > 0.0 && xyz.y > 0.0 && xyz.z > 0.0);
                assert_eq!(intensity, DirectionalLight::DEFAULT_ILLUMINANCE.get());
                assert_eq!(color, [1.0, 1.0, 1.0]);
                assert_eq!(shadow, ShadowCast::Disabled);
            }
            other => panic!("directional light became {other:?}"),
        }
    }
}
