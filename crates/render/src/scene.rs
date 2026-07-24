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
    Sphere { radius: f32, segments: u32, rings: u32 },
    // Future: Box, Plane, Cylinder, Custom { path: String }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialDesc {
    Dielectric { base_color: [f32; 3], roughness: f32 },
    Metal { base_color: [f32; 3], roughness: f32 },
    Coat { base_color: [f32; 3], coat_weight: f32, coat_roughness: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LightDesc {
    Directional { direction: [f32; 3], intensity: f32, color: [f32; 3] },
    // Future: Point, Spot
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraDesc {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,  // degrees
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
