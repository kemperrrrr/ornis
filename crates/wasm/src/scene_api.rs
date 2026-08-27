//! Client-side mirror of the `/api/scene` JSON contract served by the
//! remote editor (see `src/remote.rs`). Kept free of web-sys/wgpu types so
//! it compiles — and is unit-tested — on native targets.
//!
//! Since the D2 rewrite (audit 2026-08-22 §6.2) the contract is generic:
//! every entity carries a `components` map keyed by the registry name,
//! and payloads are serde-canonical forms of the component types from
//! `ornis_render::scene` (externally-tagged enums) — both sides use the
//! same types, no per-variant mirror code. The server may answer with a
//! reduced variant whose entities lack `components` (no live world yet) —
//! parsing such a payload fails here and the caller falls back to the
//! static `scene.ron` path.

use ornis_render::scene::{
    CameraDesc, EntityDesc, LightDesc, MaterialDesc, MeshDesc, Scene, TransformDesc,
};
use serde::Deserialize;

/// Root object of the `/api/scene` response. Unknown fields
/// (`entity_count`, entity `id`/`generation`, …) are ignored by serde.
#[derive(Debug, Clone, Deserialize)]
pub struct ApiScene {
    /// Server-side scene version; the client re-uploads GPU data only when
    /// this changes between polls.
    pub version: u64,
    #[serde(default)]
    pub entities: Vec<ApiEntity>,
    #[serde(default)]
    pub lights: Vec<LightDesc>,
    pub camera: CameraDesc,
    #[serde(default)]
    pub ambient: [f32; 3],
}

/// Entity entry: `id`/`generation` (ignored here) plus the components map.
#[derive(Debug, Clone, Deserialize)]
pub struct ApiEntity {
    pub components: RenderComponents,
}

/// The components the renderer needs. They are mandatory: an entity
/// without them means the server has no live world, and the parse error
/// is the fallback signal. Other registry entries (`Name`, future types)
/// are ignored by serde.
#[derive(Debug, Clone, Deserialize)]
pub struct RenderComponents {
    #[serde(rename = "Transform")]
    pub transform: TransformDesc,
    #[serde(rename = "Mesh")]
    pub mesh: MeshDesc,
    #[serde(rename = "Material")]
    pub material: MaterialDesc,
}

/// A successfully parsed `/api/scene` payload converted into the render
/// crate's scene description. The WASM runtime inserts it into the shared
/// [`ornis_render::RenderWorld`] before ECS extraction and GPU upload.
pub struct LiveScene {
    pub version: u64,
    pub scene: Scene,
}

impl ApiScene {
    fn into_live(self) -> LiveScene {
        let entities = self
            .entities
            .into_iter()
            .map(|e| {
                let components = e.components;
                EntityDesc {
                    name: String::new(),
                    transform: components.transform,
                    mesh: components.mesh,
                    material: components.material,
                }
            })
            .collect();
        LiveScene {
            version: self.version,
            scene: Scene {
                name: "live".to_string(),
                entities,
                lights: self.lights,
                camera: self.camera,
                ambient: self.ambient,
            },
        }
    }
}

/// Parse a `/api/scene` JSON body. Returns `Err` for malformed JSON and —
/// intentionally — for the reduced server variant without per-entity
/// `components`, so the caller can fall back to `scene.ron`.
pub fn parse_scene_json(json: &str) -> Result<LiveScene, serde_json::Error> {
    Ok(serde_json::from_str::<ApiScene>(json)?.into_live())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full contract payload, in the canonical generic form.
    const FULL_CONTRACT: &str = r#"{
        "version": 5, "entity_count": 2,
        "entities": [{
            "id": 0, "generation": 0,
            "components": {
                "Name": "Red Sphere",
                "Transform": {"translation":[-5.6,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},
                "Mesh": {"Sphere": {"radius":1.0,"segments":32,"rings":24}},
                "Material": {"Dielectric": {"base_color":[0.8,0.2,0.2],"roughness":0.5}}
            }
        }],
        "lights": [{"Directional": {"direction":[1,1,1],"intensity":0.6,"color":[1,1,1]}}],
        "camera": {"position":[0,2.5,9],"target":[0,0,0],"up":[0,1,0],"fov":60.0,"near":0.1,"far":100.0},
        "ambient": [0.10,0.10,0.15]
    }"#;

    #[test]
    fn parses_full_contract() {
        let live = parse_scene_json(FULL_CONTRACT).expect("full contract must parse");
        assert_eq!(live.version, 5);
        assert_eq!(live.scene.entities.len(), 1);
        assert_eq!(live.scene.lights.len(), 1);
        assert_eq!(live.scene.ambient, [0.10, 0.10, 0.15]);
        assert_eq!(live.scene.camera.position, [0.0, 2.5, 9.0]);
        assert!((live.scene.camera.fov - 60.0).abs() < f32::EPSILON);
        let e = &live.scene.entities[0];
        assert_eq!(e.transform.translation, [-5.6, 0.0, 0.0]);
        assert!(matches!(
            e.mesh,
            MeshDesc::Sphere {
                radius,
                segments: 32,
                rings: 24
            } if (radius - 1.0).abs() < f32::EPSILON
        ));
        assert!(matches!(
            e.material,
            MaterialDesc::Dielectric { roughness, .. } if (roughness - 0.5).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn rejects_reduced_variant_without_entity_fields() {
        // Server without a live world: entities lack `components`.
        // Parse must fail so the caller falls back to the scene.ron path.
        let reduced = r#"{
            "version": 3, "entity_count": 1,
            "entities": [{"id": 0, "generation": 0}],
            "lights": [{"Directional": {"direction":[1,1,1],"intensity":0.6,"color":[1,1,1]}}],
            "camera": {"position":[0,2.5,9],"target":[0,0,0],"up":[0,1,0],"fov":60.0,"near":0.1,"far":100.0},
            "ambient": [0.10,0.10,0.15]
        }"#;
        assert!(parse_scene_json(reduced).is_err());
    }

    #[test]
    fn parses_coat_and_metal_materials_and_multiple_meshes() {
        let json = r#"{
            "version": 7,
            "entities": [
                {
                    "components": {
                        "Transform": {"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},
                        "Mesh": {"Sphere": {"radius":2.0,"segments":48,"rings":32}},
                        "Material": {"Coat": {"base_color":[0.1,0.2,0.9],"coat_weight":0.8,"coat_roughness":0.1}}
                    }
                },
                {
                    "components": {
                        "Transform": {"translation":[3,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},
                        "Mesh": {"Sphere": {"radius":0.5,"segments":16,"rings":12}},
                        "Material": {"Metal": {"base_color":[0.9,0.8,0.3],"roughness":0.2}}
                    }
                }
            ],
            "lights": [],
            "camera": {"position":[0,2,8],"target":[0,0,0],"up":[0,1,0],"fov":55.0,"near":0.1,"far":100.0},
            "ambient": [0.1,0.1,0.1]
        }"#;
        let live = parse_scene_json(json).expect("coat/metal payload must parse");
        assert_eq!(live.version, 7);
        assert_eq!(live.scene.entities.len(), 2);
        assert!(matches!(
            live.scene.entities[0].material,
            MaterialDesc::Coat {
                coat_weight,
                coat_roughness,
                ..
            } if (coat_weight - 0.8).abs() < f32::EPSILON
                && (coat_roughness - 0.1).abs() < f32::EPSILON
        ));
        assert!(matches!(
            live.scene.entities[1].material,
            MaterialDesc::Metal { .. }
        ));
        // Radii differ between entities — the GPU builder must not assume a
        // single shared radius.
        let MeshDesc::Sphere { radius: r0, .. } = live.scene.entities[0].mesh;
        let MeshDesc::Sphere { radius: r1, .. } = live.scene.entities[1].mesh;
        assert!((r0 - 2.0).abs() < f32::EPSILON);
        assert!((r1 - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn rejects_unknown_enum_variants() {
        let unknown_mesh = r#"{
            "version": 1,
            "entities": [{
                "components": {
                    "Transform": {"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},
                    "Mesh": {"Torus": {"radius":1.0}},
                    "Material": {"Dielectric": {"base_color":[1,1,1],"roughness":0.5}}
                }
            }],
            "lights": [],
            "camera": {"position":[0,0,5],"target":[0,0,0],"up":[0,1,0],"fov":60.0,"near":0.1,"far":100.0}
        }"#;
        assert!(parse_scene_json(unknown_mesh).is_err());

        let unknown_material = r#"{
            "version": 1,
            "entities": [{
                "components": {
                    "Transform": {"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},
                    "Mesh": {"Sphere": {"radius":1.0,"segments":32,"rings":24}},
                    "Material": {"Hair": {"base_color":[1,1,1]}}
                }
            }],
            "lights": [],
            "camera": {"position":[0,0,5],"target":[0,0,0],"up":[0,1,0],"fov":60.0,"near":0.1,"far":100.0}
        }"#;
        assert!(parse_scene_json(unknown_material).is_err());
    }

    #[test]
    fn rejects_missing_version_and_malformed_json() {
        let no_version = r#"{
            "entities": [],
            "lights": [],
            "camera": {"position":[0,0,5],"target":[0,0,0],"up":[0,1,0],"fov":60.0,"near":0.1,"far":100.0}
        }"#;
        assert!(parse_scene_json(no_version).is_err());
        assert!(parse_scene_json("not json").is_err());
        assert!(parse_scene_json("").is_err());
    }
}
