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
    // Future: Box, Plane, Cylinder, Custom { path: String }
}

/// Material preset mapped onto the engine's OpenPBR surface model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialDesc {
    /// Non-metal with specular reflection.
    Dielectric {
        /// Albedo color in linear space.
        base_color: [f32; 3],
        /// Microfacet roughness in [0, 1].
        roughness: f32,
    },
    /// Conductor with tinted specular reflection.
    Metal {
        /// Reflectance color in linear space.
        base_color: [f32; 3],
        /// Microfacet roughness in [0, 1].
        roughness: f32,
    },
    /// Base layer with a clearcoat on top.
    Coat {
        /// Albedo color of the base layer.
        base_color: [f32; 3],
        /// Clearcoat strength in [0, 1].
        coat_weight: f32,
        /// Clearcoat roughness in [0, 1].
        coat_roughness: f32,
    },
}

/// Light source description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LightDesc {
    /// Infinitely distant light shining along a fixed direction.
    Directional {
        /// Direction the light travels (from light toward the scene).
        direction: [f32; 3],
        /// Radiometric strength multiplier.
        intensity: f32,
        /// Emission color in linear space.
        color: [f32; 3],
    },
    // Future: Point, Spot
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
            } => {
                assert_eq!(*base_color, [0.5, 0.5, 0.5]);
                assert_eq!(*roughness, 0.9);
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
        }
        assert_eq!(scene.entities[0].transform.translation, [1.0, 2.0, 3.0]);
        assert_eq!(scene.entities[0].transform.rotation, [0.0, 0.0, 0.0, 1.0]);

        match &scene.lights[0] {
            LightDesc::Directional {
                direction,
                intensity,
                color,
            } => {
                assert_eq!(*direction, [1.0, 1.0, 1.0]);
                assert_eq!(*intensity, 0.6);
                assert_eq!(*color, [1.0, 1.0, 1.0]);
            }
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
            Scene::from_ron(&FULL_SCENE_RON.replace("Dielectric(base_color", "Glass(base_color"))
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
    fn demo_asset_parses() {
        // Keeps the shipped demo scene in sync with the schema.
        let scene = Scene::from_ron(include_str!("../../../assets/scene.ron")).expect("demo scene");
        assert_eq!(scene.name, "demo");
        assert_eq!(scene.entities.len(), 5);
        assert_eq!(scene.lights.len(), 2);
        assert_eq!(scene.camera.fov, 60.0);
    }
}
