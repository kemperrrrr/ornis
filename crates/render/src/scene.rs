//! Declarative scene descriptions (de)serialized as RON.
//!
//! The `*Desc` types are the serde-canonical contract shared by the demo
//! asset (`assets/scene.ron`), the editor protocol and the WASM viewport:
//! component payloads travel over the wire in exactly this shape.

use serde::{Deserialize, Serialize};

/// Full scene description in RON format (see `assets/scene.ron`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    /// Human-readable scene label.
    pub name: String,
    /// Renderable entities.
    pub entities: Vec<EntityDesc>,
    /// Scene lights.
    pub lights: Vec<LightDesc>,
    /// The single viewing camera.
    pub camera: CameraDesc,
    /// Ambient light RGB multiplier.
    pub ambient: [f32; 3],
}

/// One renderable object: identity plus its three components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDesc {
    /// Display name; the editor uses it as the default `Name` component.
    pub name: String,
    /// Placement in world space.
    pub transform: TransformDesc,
    /// Geometry.
    pub mesh: MeshDesc,
    /// OpenPBR surface description.
    pub material: MaterialDesc,
}

/// Placement of an entity in world space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformDesc {
    /// Translation in world units.
    pub translation: [f32; 3],
    /// Orientation as a quaternion in `(x, y, z, w)` order.
    pub rotation: [f32; 4],
    /// Non-uniform scale per axis.
    pub scale: [f32; 3],
}

/// Geometry description (procedurally generated at load time).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshDesc {
    /// UV sphere centered at the transform origin.
    Sphere {
        /// Radius in world units.
        radius: f32,
        /// Longitude divisions (minimum 3 at generation time).
        segments: u32,
        /// Latitude divisions (minimum 2 at generation time).
        rings: u32,
    },
    /// Axis-aligned box centered at the transform origin.
    Box {
        /// Full extents per axis in world units.
        size: [f32; 3],
    },
    /// Flat quad in the local XZ plane (`+Y` face normal), centered at
    /// the transform origin.
    Plane {
        /// Full extents in world units: `[width_x, depth_z]`.
        size: [f32; 2],
    },
    /// Right circular cylinder around local `+Y`, centered at the
    /// transform origin.
    Cylinder {
        /// Radius in world units.
        radius: f32,
        /// Height along `+Y` in world units.
        height: f32,
        /// Radial divisions (minimum 3 at generation time).
        radial_segments: u32,
    },
    /// Inline vertex soup: positions plus a triangle index list.
    /// Shading normals are NOT stored — recompute them at load (see
    /// `MeshData::from_positions` + `with_computed_normals` in the mesh
    /// editor); shape checks also live there, not in the transport.
    Custom {
        /// Vertex positions in engine units.
        positions: Vec<[f32; 3]>,
        /// Triangle index list (`u32`, triples, CCW from outside).
        indices: Vec<u32>,
    },
}

impl MeshDesc {
    /// Borrows the inline soup of a [`MeshDesc::Custom`].
    ///
    /// Returns `None` for procedural variants (`Sphere`, `Box`, `Plane`,
    /// `Cylinder`). The render extraction routes `Some` into the per-entity
    /// upload path (`mesh_upload::custom_vertices`); the transport itself
    /// never validates shapes — see `MeshData::validate` in the mesh editor.
    pub fn as_custom(&self) -> Option<(&[[f32; 3]], &[u32])> {
        match self {
            Self::Custom { positions, indices } => Some((positions, indices)),
            Self::Sphere { .. } | Self::Box { .. } | Self::Plane { .. } | Self::Cylinder { .. } => {
                None
            }
        }
    }
}

/// Material preset mapped onto the engine's OpenPBR surface model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MaterialDesc {
    /// Non-metal with specular reflection.
    Dielectric {
        /// Albedo color in linear space.
        base_color: [f32; 3],
        /// Microfacet roughness in [0, 1].
        roughness: f32,
        /// Emissive RGB in linear space (`[0, 0, 0]` = no emission).
        /// Absent in older files — defaults to off.
        #[serde(default)]
        emission: [f32; 3],
    },
    /// Conductor with tinted specular reflection.
    Metal {
        /// Reflectance color in linear space.
        base_color: [f32; 3],
        /// Microfacet roughness in [0, 1].
        roughness: f32,
        /// Emissive RGB in linear space (`[0, 0, 0]` = no emission).
        /// Absent in older files — defaults to off.
        #[serde(default)]
        emission: [f32; 3],
    },
    /// Base layer with a clearcoat on top.
    Coat {
        /// Albedo color of the base layer.
        base_color: [f32; 3],
        /// Clearcoat strength in [0, 1].
        coat_weight: f32,
        /// Clearcoat roughness in [0, 1].
        coat_roughness: f32,
        /// Emissive RGB in linear space (`[0, 0, 0]` = no emission).
        /// Absent in older files — defaults to off.
        #[serde(default)]
        emission: [f32; 3],
    },
    /// Rough diffuse-only surface (no specular lobe).
    Matte {
        /// Albedo color in linear space.
        base_color: [f32; 3],
        /// Diffuse roughness in [0, 1].
        roughness: f32,
    },
    /// Transparent refractive surface (thin-walled glass).
    Glass {
        /// Transmitted tint in linear space.
        base_color: [f32; 3],
        /// Microfacet roughness in [0, 1].
        roughness: f32,
        /// Index of refraction (>= 1.0). Absent in older files —
        /// defaults to 1.5.
        #[serde(default = "default_glass_ior")]
        ior: f32,
    },
}

/// Default [`MaterialDesc::Glass`] index of refraction (crown glass).
fn default_glass_ior() -> f32 {
    1.5
}

/// Light source description.
///
/// Convention (shared by the shader evaluators): `direction` fields point
/// *toward* the light, except spot axes, which point from the light into
/// the scene (spotlight aim).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LightDesc {
    /// Infinitely distant light shining from a fixed direction.
    Directional {
        /// Direction toward the light (from the scene).
        direction: [f32; 3],
        /// Radiometric strength multiplier.
        intensity: f32,
        /// Emission color in linear space.
        color: [f32; 3],
        /// Cast a shadow map (depth pre-pass + PCF in the evaluators).
        /// Absent in older files — defaults to off.
        #[serde(default)]
        shadow: bool,
    },
    /// Local light with inverse-square falloff and a finite range.
    Point {
        /// World-space position.
        position: [f32; 3],
        /// Radiometric strength multiplier.
        intensity: f32,
        /// Emission color in linear space.
        color: [f32; 3],
        /// Cutoff distance in world units (must be > 0).
        range: f32,
        /// Cast a shadow cube (6 depth faces + analytic major-axis
        /// sample in the evaluators).
        /// Absent in older files — defaults to off.
        #[serde(default)]
        shadow: bool,
    },
    /// Local light inside a cone aimed into the scene.
    Spot {
        /// World-space position.
        position: [f32; 3],
        /// Spotlight axis, from the light into the scene.
        direction: [f32; 3],
        /// Radiometric strength multiplier.
        intensity: f32,
        /// Emission color in linear space.
        color: [f32; 3],
        /// Cutoff distance in world units (must be > 0).
        range: f32,
        /// Inner cone angle in degrees (full brightness inside).
        inner_angle: f32,
        /// Outer cone angle in degrees (zero outside, soft edge between).
        outer_angle: f32,
        /// Cast a shadow map (depth pre-pass + PCF in the evaluators).
        /// Absent in older files — defaults to off.
        #[serde(default)]
        shadow: bool,
    },
}

/// Viewing camera described look-at style.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraDesc {
    /// Eye position in world units.
    pub position: [f32; 3],
    /// Point the camera looks at.
    pub target: [f32; 3],
    /// Up vector (should not be parallel to the view direction).
    pub up: [f32; 3],
    /// Vertical field of view in degrees.
    pub fov: f32,
    /// Near clip distance in world units.
    pub near: f32,
    /// Far clip distance in world units.
    pub far: f32,
}

impl Scene {
    /// Load a scene from a RON file.
    pub fn from_ron(ron_str: &str) -> Result<Self, ron::error::SpannedError> {
        ron::de::from_str(ron_str)
    }

    /// Save scene to RON string.
    pub fn to_ron(&self) -> Result<String, ron::error::Error> {
        ron::ser::to_string_pretty(self, Default::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One entity per material variant, one directional light.
    const FULL_SCENE_RON: &str = r#"
Scene(
    name: "test",
    entities: [
        (
            name: "dielectric",
            transform: (
                translation: (1.0, 2.0, 3.0),
                rotation: (0.0, 0.0, 0.0, 1.0),
                scale: (1.0, 1.0, 1.0),
            ),
            mesh: Sphere(radius: 2.0, segments: 16, rings: 8),
            material: Dielectric(base_color: (0.5, 0.5, 0.5), roughness: 0.9),
        ),
        (
            name: "metal",
            transform: (
                translation: (0.0, 0.0, 0.0),
                rotation: (0.0, 0.7071, 0.0, 0.7071),
                scale: (2.0, 2.0, 2.0),
            ),
            mesh: Sphere(radius: 1.0, segments: 32, rings: 24),
            material: Metal(base_color: (0.9, 0.7, 0.1), roughness: 0.2),
        ),
        (
            name: "coat",
            transform: (
                translation: (-1.0, 0.0, 0.0),
                rotation: (0.0, 0.0, 0.0, 1.0),
                scale: (1.0, 1.0, 1.0),
            ),
            mesh: Sphere(radius: 0.5, segments: 8, rings: 4),
            material: Coat(base_color: (1.0, 1.0, 1.0), coat_weight: 1.0, coat_roughness: 0.1),
        ),
    ],
    lights: [
        Directional(direction: (1.0, 1.0, 1.0), intensity: 0.6, color: (1.0, 1.0, 1.0)),
    ],
    camera: (
        position: (0.0, 2.5, 9.0),
        target: (0.0, 0.0, 0.0),
        up: (0.0, 1.0, 0.0),
        fov: 60.0,
        near: 0.1,
        far: 100.0,
    ),
    ambient: (0.1, 0.1, 0.15),
)
"#;

    #[test]
    fn parses_all_material_variants() {
        let scene = Scene::from_ron(FULL_SCENE_RON).expect("valid scene");
        assert_eq!(scene.name, "test");
        assert_eq!(scene.entities.len(), 3);
        assert_eq!(scene.lights.len(), 1);

        match &scene.entities[0].material {
            MaterialDesc::Dielectric {
                base_color,
                roughness,
                emission,
            } => {
                assert_eq!(*base_color, [0.5, 0.5, 0.5]);
                assert_eq!(*roughness, 0.9);
                // Old RON without `emission` defaults to no emission.
                assert_eq!(*emission, [0.0, 0.0, 0.0]);
            }
            other => panic!("expected Dielectric, got {other:?}"),
        }
        assert!(matches!(
            scene.entities[1].material,
            MaterialDesc::Metal { .. }
        ));
        match &scene.entities[2].material {
            MaterialDesc::Coat {
                coat_weight,
                coat_roughness,
                ..
            } => {
                assert_eq!(*coat_weight, 1.0);
                assert_eq!(*coat_roughness, 0.1);
            }
            other => panic!("expected Coat, got {other:?}"),
        }

        match &scene.entities[0].mesh {
            MeshDesc::Sphere {
                radius,
                segments,
                rings,
            } => {
                assert_eq!(*radius, 2.0);
                assert_eq!(*segments, 16);
                assert_eq!(*rings, 8);
            }
            // New variants must extend this assertion, not break it: old
            // files stay Sphere-only.
            other => panic!("expected Sphere, got {other:?}"),
        }
        assert_eq!(scene.entities[0].transform.translation, [1.0, 2.0, 3.0]);
        assert_eq!(scene.entities[0].transform.rotation, [0.0, 0.0, 0.0, 1.0]);

        match &scene.lights[0] {
            LightDesc::Directional {
                direction,
                intensity,
                color,
                ..
            } => {
                assert_eq!(*direction, [1.0, 1.0, 1.0]);
                assert_eq!(*intensity, 0.6);
                assert_eq!(*color, [1.0, 1.0, 1.0]);
            }
            _ => panic!("expected the directional test light"),
        }
        assert_eq!(scene.camera.fov, 60.0);
        assert_eq!(scene.camera.near, 0.1);
        assert_eq!(scene.camera.far, 100.0);
        assert_eq!(scene.ambient, [0.1, 0.1, 0.15]);
    }

    #[test]
    fn ron_round_trip_is_stable() {
        let scene = Scene::from_ron(FULL_SCENE_RON).expect("valid scene");
        let serialized = scene.to_ron().expect("serialize");
        let reparsed = Scene::from_ron(&serialized).expect("re-parse");
        let reserialized = reparsed.to_ron().expect("re-serialize");
        assert_eq!(serialized, reserialized);
    }

    #[test]
    fn rejects_malformed_ron() {
        assert!(Scene::from_ron("Scene(name: 42)").is_err());
        assert!(Scene::from_ron("not a scene at all").is_err());
        // Unknown material variant.
        assert!(
            Scene::from_ron(&FULL_SCENE_RON.replace("Dielectric(base_color", "Crystal(base_color"))
                .is_err()
        );
    }

    #[test]
    fn rejects_missing_fields() {
        // `ambient` is missing.
        let broken = FULL_SCENE_RON.replace("    ambient: (0.1, 0.1, 0.15),\n", "");
        assert!(Scene::from_ron(&broken).is_err());
    }

    #[test]
    fn emission_defaults_to_off_for_old_ron() {
        // `FULL_SCENE_RON` predates `emission`: all three variants must
        // deserialize with zero emission (additive schema change).
        let scene = Scene::from_ron(FULL_SCENE_RON).expect("valid scene");
        for entity in &scene.entities {
            let emission = match &entity.material {
                MaterialDesc::Dielectric { emission, .. }
                | MaterialDesc::Metal { emission, .. }
                | MaterialDesc::Coat { emission, .. } => *emission,
                MaterialDesc::Matte { .. } | MaterialDesc::Glass { .. } => [0.0, 0.0, 0.0],
            };
            assert_eq!(emission, [0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn emission_and_matte_round_trip() {
        let scene = Scene::from_ron(FULL_SCENE_RON).expect("valid scene");
        let with_new = scene.to_ron().expect("serialize").replacen(
            "emission: (0.0, 0.0, 0.0)",
            "emission: (2.0, 1.0, 0.5)",
            1,
        );
        let parsed = Scene::from_ron(&with_new).expect("emission parses");
        assert!(matches!(
            parsed.entities[0].material,
            MaterialDesc::Dielectric {
                emission: [2.0, 1.0, 0.5],
                ..
            }
        ));

        let matte_ron = FULL_SCENE_RON.replace(
            "Dielectric(base_color: (0.5, 0.5, 0.5), roughness: 0.9)",
            "Matte(base_color: (0.2, 0.4, 0.6), roughness: 0.7)",
        );
        let matte = Scene::from_ron(&matte_ron).expect("matte parses");
        assert!(matches!(
            matte.entities[0].material,
            MaterialDesc::Matte {
                base_color: [0.2, 0.4, 0.6],
                roughness,
            } if (roughness - 0.7).abs() < f32::EPSILON
        ));
        let reserialized = matte.to_ron().expect("serialize");
        let reparsed = Scene::from_ron(&reserialized).expect("re-parse");
        assert_eq!(reserialized, reparsed.to_ron().expect("re-serialize"));
    }

    #[test]
    fn glass_defaults_ior_and_round_trips() {
        // `ior` absent → 1.5 (additive schema change, same as `emission`).
        let without_ior = FULL_SCENE_RON.replace(
            "Dielectric(base_color: (0.5, 0.5, 0.5), roughness: 0.9)",
            "Glass(base_color: (0.9, 0.95, 1.0), roughness: 0.05)",
        );
        let scene = Scene::from_ron(&without_ior).expect("glass parses");
        assert!(matches!(
            scene.entities[0].material,
            MaterialDesc::Glass {
                base_color: [0.9, 0.95, 1.0],
                roughness,
                ior,
            } if (roughness - 0.05).abs() < f32::EPSILON && (ior - 1.5).abs() < f32::EPSILON
        ));

        let with_ior = FULL_SCENE_RON.replace(
            "Dielectric(base_color: (0.5, 0.5, 0.5), roughness: 0.9)",
            "Glass(base_color: (0.9, 0.95, 1.0), roughness: 0.05, ior: 1.33)",
        );
        let explicit = Scene::from_ron(&with_ior).expect("glass ior parses");
        assert!(matches!(
            explicit.entities[0].material,
            MaterialDesc::Glass { ior, .. } if (ior - 1.33).abs() < f32::EPSILON
        ));
        let serialized = explicit.to_ron().expect("serialize");
        let reparsed = Scene::from_ron(&serialized).expect("re-parse");
        assert_eq!(serialized, reparsed.to_ron().expect("re-serialize"));
    }

    #[test]
    fn demo_asset_parses() {
        // Keeps the shipped demo scene in sync with the schema.
        let scene = Scene::from_ron(include_str!("../../../assets/scene.ron")).expect("demo scene");
        assert_eq!(scene.name, "demo");
        assert_eq!(scene.entities.len(), 5);
        assert_eq!(scene.lights.len(), 2);
        assert_eq!(scene.camera.fov, 60.0);
    }

    #[test]
    fn procedural_variants_parse_and_round_trip() {
        let ron = FULL_SCENE_RON
            .replace(
                "mesh: Sphere(radius: 2.0, segments: 16, rings: 8),",
                "mesh: Box(size: (2.0, 4.0, 6.0)),",
            )
            .replace(
                "mesh: Sphere(radius: 1.0, segments: 32, rings: 24),",
                "mesh: Plane(size: (3.0, 5.0)),",
            )
            .replace(
                "mesh: Sphere(radius: 0.5, segments: 8, rings: 4),",
                "mesh: Cylinder(radius: 1.5, height: 7.0, radial_segments: 12),",
            );
        let scene = Scene::from_ron(&ron).expect("procedural variants parse");
        assert!(matches!(
            scene.entities[0].mesh,
            MeshDesc::Box { size } if size == [2.0, 4.0, 6.0]
        ));
        assert!(matches!(
            scene.entities[1].mesh,
            MeshDesc::Plane { size } if size == [3.0, 5.0]
        ));
        assert!(matches!(
            scene.entities[2].mesh,
            MeshDesc::Cylinder {
                radius,
                height,
                radial_segments,
            } if radius == 1.5 && height == 7.0 && radial_segments == 12
        ));
        let serialized = scene.to_ron().expect("serialize");
        let reparsed = Scene::from_ron(&serialized).expect("re-parse");
        let reserialized = reparsed.to_ron().expect("re-serialize");
        assert_eq!(serialized, reserialized);
    }

    #[test]
    fn as_custom_returns_none_for_procedurals() {
        let variants = [
            MeshDesc::Sphere {
                radius: 1.0,
                segments: 16,
                rings: 8,
            },
            MeshDesc::Box {
                size: [1.0, 2.0, 3.0],
            },
            MeshDesc::Plane { size: [1.0, 2.0] },
            MeshDesc::Cylinder {
                radius: 1.0,
                height: 2.0,
                radial_segments: 8,
            },
        ];
        for mesh in &variants {
            assert!(mesh.as_custom().is_none(), "procedural: {mesh:?}");
        }
        let custom = MeshDesc::Custom {
            positions: vec![[0.0, 0.0, 0.0]],
            indices: vec![0, 0, 0],
        };
        let (positions, indices) = custom.as_custom().expect("custom borrows soup");
        assert_eq!(positions.len(), 1);
        assert_eq!(indices.len(), 3);
    }
}
