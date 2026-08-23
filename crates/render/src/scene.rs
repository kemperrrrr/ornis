use serde::{Deserialize, Serialize};

/// Scene description in RON format.
/// Example: `assets/scene.ron`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub name: String,
    pub entities: Vec<EntityDesc>,
    pub lights: Vec<LightDesc>,
    pub camera: CameraDesc,
    pub ambient: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDesc {
    pub name: String,
    pub transform: TransformDesc,
    pub mesh: MeshDesc,
    pub material: MaterialDesc,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformDesc {
    pub translation: [f32; 3],
    pub rotation: [f32; 4], // quaternion xyzw
    pub scale: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshDesc {
    Sphere {
        radius: f32,
        segments: u32,
        rings: u32,
    },
    // Future: Box, Plane, Cylinder, Custom { path: String }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialDesc {
    Dielectric {
        base_color: [f32; 3],
        roughness: f32,
    },
    Metal {
        base_color: [f32; 3],
        roughness: f32,
    },
    Coat {
        base_color: [f32; 3],
        coat_weight: f32,
        coat_roughness: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LightDesc {
    Directional {
        direction: [f32; 3],
        intensity: f32,
        color: [f32; 3],
    },
    // Future: Point, Spot
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraDesc {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32, // degrees
    pub near: f32,
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
