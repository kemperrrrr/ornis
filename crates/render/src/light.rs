//! Typed directional light and the studio key/fill pair placed into a
//! [`RenderLights`](crate::extraction::RenderLights) rig.
//!
//! The channels match a scene `Directional` light: [`UnitVec3`] points
//! toward the light, [`Lux`] is the existing intensity `f32`, and [`Color`]
//! is linear emission. [`DirectionalLight::to_light_desc`] is what the
//! renderer uploads. [`StudioLights`] is that pair on purpose; nothing else
//! publishes it.

use glam::Vec3;
use ornis_assets::scene::{LightDesc, ShadowCast};
use ornis_core::{Color, Lux, UnitVec3};

/// One infinitely distant light.
///
/// `..Default::default()` fills [`Self::shadow`] ([`ShadowCast::Enabled`])
/// and, when a field is omitted, the legacy key direction,
/// [`Self::DEFAULT_ILLUMINANCE`], and [`Color::WHITE`]. Scene files stay
/// unshadowed unless they set `shadow`: [`ShadowCast`]'s serde default is
/// still off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalLight {
    /// Direction toward the light, from the scene.
    pub direction: UnitVec3,
    /// Radiometric strength. Same number as scene `intensity`.
    pub illuminance: Lux,
    /// Linear emission color.
    pub color: Color,
    /// Shadow-map request ([`ShadowCast`]). [`Default`] casts a shadow.
    /// Older scene files stay off: [`ShadowCast`]'s serde default is
    /// [`ShadowCast::Disabled`].
    pub shadow: ShadowCast,
}

impl DirectionalLight {
    /// Legacy key-light strength (`0.6`) from the pre-X3 rig.
    pub const DEFAULT_ILLUMINANCE: Lux = Lux(0.6);

    /// Directional scene light, or [`None`] for a point or spot.
    fn from_light_desc(desc: &LightDesc) -> Option<Self> {
        let LightDesc::Directional {
            direction,
            intensity,
            color,
            shadow,
        } = desc
        else {
            return None;
        };
        Some(Self {
            direction: *direction,
            illuminance: Lux::new(*intensity),
            color: Color::linear_rgb(color[0], color[1], color[2]),
            shadow: *shadow,
        })
    }

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
    /// [`Self::DEFAULT_ILLUMINANCE`], white, shadow map on.
    fn default() -> Self {
        Self {
            direction: UnitVec3::normalize(Vec3::new(1.0, 1.0, 1.0)).unwrap_or(UnitVec3::Y),
            illuminance: Self::DEFAULT_ILLUMINANCE,
            color: Color::WHITE,
            shadow: ShadowCast::Enabled,
        }
    }
}

/// Studio key and fill.
///
/// [`Default`] keeps the directions, colors, and strengths of
/// [`RenderLights::default`](crate::extraction::RenderLights::default):
/// a white key toward `(1, 1, 1)` and a cooler fill toward `(-0.5, 0.5, -0.5)`.
/// The key casts a shadow; the fill does not. Spawning appends both lights.
/// Ambient and lights already in the world stay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudioLights {
    /// Brighter light of the pair.
    pub key: DirectionalLight,
    /// Dimmer light from the opposite octant.
    pub fill: DirectionalLight,
}

impl StudioLights {
    /// Fill strength (`0.3`) of [`Default`].
    pub const DEFAULT_FILL_ILLUMINANCE: Lux = Lux(0.3);

    /// Key then fill, in the order
    /// [`RenderLights::default`](crate::extraction::RenderLights::default) stores them.
    pub fn lights(self) -> [DirectionalLight; 2] {
        [self.key, self.fill]
    }
}

impl Default for StudioLights {
    /// Key and fill copied from
    /// [`RenderLights::default`](crate::extraction::RenderLights::default),
    /// then the key shadow is turned on and the fill shadow is turned off.
    ///
    /// Directions, colors, and strengths stay on the legacy rig. A missing
    /// entry falls back to [`DirectionalLight::default`] (key) or that same
    /// light with the shadow cleared (fill).
    fn default() -> Self {
        let lights = crate::extraction::RenderLights::default().lights;
        let mut key = lights
            .first()
            .and_then(DirectionalLight::from_light_desc)
            .unwrap_or_default();
        let mut fill = lights
            .get(1)
            .and_then(DirectionalLight::from_light_desc)
            .unwrap_or(DirectionalLight {
                shadow: ShadowCast::Disabled,
                ..DirectionalLight::default()
            });
        key.shadow = ShadowCast::Enabled;
        fill.shadow = ShadowCast::Disabled;
        Self { key, fill }
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
                assert_eq!(shadow, ShadowCast::Enabled);
            }
            other => panic!("directional light became {other:?}"),
        }
    }

    #[test]
    fn default_studio_lights_match_the_legacy_key_and_fill() {
        let pair = StudioLights::default();
        assert_eq!(pair.key, DirectionalLight::default());
        assert_eq!(pair.key.shadow, ShadowCast::Enabled);
        assert_eq!(pair.fill.shadow, ShadowCast::Disabled);
        let [key, fill] = pair.lights();
        let expected = crate::extraction::RenderLights::default().lights;
        // The resource default stays unshadowed. The studio key is that
        // light with the shadow map turned on; the fill matches it.
        let LightDesc::Directional {
            direction: key_dir,
            intensity: key_intensity,
            color: key_color,
            shadow: ShadowCast::Disabled,
        } = &expected[0]
        else {
            panic!("legacy key");
        };
        assert_eq!(key.direction, *key_dir);
        assert_eq!(key.illuminance.get(), *key_intensity);
        assert_eq!(key.color.to_linear_rgb().as_array(), *key_color);
        assert_eq!(
            format!("{:?}", fill.to_light_desc()),
            format!("{:?}", expected[1])
        );
        assert_eq!(fill.illuminance, StudioLights::DEFAULT_FILL_ILLUMINANCE);
    }
}
