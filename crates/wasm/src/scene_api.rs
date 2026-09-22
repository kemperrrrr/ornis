//! Client-side mirror of the `/api/scene` JSON contract served by the
//! remote editor (see `src/remote.rs`). Kept free of web-sys/wgpu types so
//! it compiles — and is unit-tested — on native targets.
//!
//! Since the D2 rewrite (audit 2026-08-22 §6.2) the contract is generic:
//! every entity carries a `components` map keyed by the registry name,
//! the root also carries a transport `sequence`, and payloads are
//! serde-canonical forms of the component types from
//! `ornis_render::scene` (externally-tagged enums) — both sides use the
//! same types, no per-variant mirror code. The server may answer with a
//! reduced variant whose entities lack `components` (no live world yet) —
//! parsing such a payload fails here and the caller reports the error:
//! there is no static `scene.ron` fallback (unified runtime: `/api/scene`
//! is the sole source of truth).

use ornis_assets::scene::{
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
    /// Transport sequence assigned by the editor backend. It is independent
    /// of the scene version and lets consumers reject out-of-order replies.
    #[serde(default)]
    pub sequence: u64,
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
/// [`GameWorld`](ornis_app::GameWorld) before ECS extraction and GPU upload.
pub struct LiveScene {
    /// Authoritative server-side scene version.
    pub version: u64,
    /// Transport sequence assigned by the editor backend.
    pub sequence: u64,
    /// Renderable scene description reconstructed from the snapshot.
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
            sequence: self.sequence,
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
/// `components`, so the caller reports the error instead of rendering.
pub fn parse_scene_json(json: &str) -> Result<LiveScene, serde_json::Error> {
    Ok(serde_json::from_str::<ApiScene>(json)?.into_live())
}

#[cfg(test)]
/// The full contract payload, in the canonical generic form.
pub(crate) const FULL_CONTRACT: &str = r#"{
    "version": 5, "sequence": 12, "entity_count": 2,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_contract() {
        let live = parse_scene_json(FULL_CONTRACT).expect("full contract must parse");
        assert_eq!(live.version, 5);
        assert_eq!(live.sequence, 12);
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
        // Parse must fail so the caller reports the error (no fallback).
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
        let (r0, r1) = match (&live.scene.entities[0].mesh, &live.scene.entities[1].mesh) {
            (MeshDesc::Sphere { radius: r0, .. }, MeshDesc::Sphere { radius: r1, .. }) => {
                (*r0, *r1)
            }
            (a, b) => panic!("expected spheres, got {a:?} and {b:?}"),
        };
        assert!((r0 - 2.0).abs() < f32::EPSILON);
        assert!((r1 - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn parses_custom_mesh_variant() {
        let json = r#"{
            "version": 2,
            "entities": [{
                "components": {
                    "Transform": {"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},
                    "Mesh": {"Custom": {"positions": [[0,0,0],[1,0,0],[0,1,0]], "indices": [0,1,2]}},
                    "Material": {"Dielectric": {"base_color":[1,1,1],"roughness":0.5}}
                }
            }],
            "lights": [],
            "camera": {"position":[0,0,5],"target":[0,0,0],"up":[0,1,0],"fov":60.0,"near":0.1,"far":100.0}
        }"#;
        let live = parse_scene_json(json).expect("Custom mesh must parse");
        let MeshDesc::Custom { positions, indices } = &live.scene.entities[0].mesh else {
            panic!(
                "expected Custom mesh, got {:?}",
                live.scene.entities[0].mesh
            );
        };
        assert_eq!(positions.len(), 3);
        assert_eq!(*indices, vec![0, 1, 2]);
    }

    #[test]
    fn custom_mesh_with_bad_indices_still_parses() {
        // Shape checks live in mesh-editor validation, not in the scene
        // transport: the scene must parse even when `indices` is not a
        // multiple of 3 (the exact condition `MeshData::validate`
        // rejects with `IndexCountNotMultipleOfThree`).
        let json = r#"{
            "version": 2,
            "entities": [{
                "components": {
                    "Transform": {"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},
                    "Mesh": {"Custom": {"positions": [[0,0,0],[1,0,0],[0,1,0]], "indices": [0,1]}},
                    "Material": {"Dielectric": {"base_color":[1,1,1],"roughness":0.5}}
                }
            }],
            "lights": [],
            "camera": {"position":[0,0,5],"target":[0,0,0],"up":[0,1,0],"fov":60.0,"near":0.1,"far":100.0}
        }"#;
        let live = parse_scene_json(json).expect("transport must not shape-check");
        let MeshDesc::Custom { positions, indices } = &live.scene.entities[0].mesh else {
            panic!(
                "expected Custom mesh, got {:?}",
                live.scene.entities[0].mesh
            );
        };
        assert_eq!(positions.len(), 3);
        assert!(!indices.len().is_multiple_of(3));
    }

    #[test]
    fn parses_box_plane_cylinder_and_emission_parity() {
        // WASM transport parity for the procedural variants in `MeshDesc`
        // plus `MaterialDesc::emission`: new payloads parse with values
        // intact, while old payloads without `emission` keep loading with
        // emission off (serde default).
        let json = r#"{
            "version": 9,
            "entities": [
                {
                    "components": {
                        "Transform": {"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},
                        "Mesh": {"Box": {"size": [2.0, 4.0, 6.0]}},
                        "Material": {"Dielectric": {"base_color":[0.8,0.2,0.2],"roughness":0.4,"emission":[2.0,1.0,0.5]}}
                    }
                },
                {
                    "components": {
                        "Transform": {"translation":[3,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},
                        "Mesh": {"Plane": {"size": [3.0, 5.0]}},
                        "Material": {"Metal": {"base_color":[0.9,0.8,0.3],"roughness":0.2}}
                    }
                },
                {
                    "components": {
                        "Transform": {"translation":[-3,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},
                        "Mesh": {"Cylinder": {"radius": 1.5, "height": 7.0, "radial_segments": 12}},
                        "Material": {"Coat": {"base_color":[0.1,0.2,0.9],"coat_weight":0.8,"coat_roughness":0.1,"emission":[0.0,0.0,1.0]}}
                    }
                }
            ],
            "lights": [],
            "camera": {"position":[0,2,8],"target":[0,0,0],"up":[0,1,0],"fov":55.0,"near":0.1,"far":100.0},
            "ambient": [0.1,0.1,0.1]
        }"#;
        let live =
            parse_scene_json(json).expect("box/plane/cylinder + emission payload must parse");
        assert_eq!(live.scene.entities.len(), 3);
        assert!(matches!(
            live.scene.entities[0].mesh,
            MeshDesc::Box { size } if size == [2.0, 4.0, 6.0]
        ));
        assert!(matches!(
            live.scene.entities[0].material,
            MaterialDesc::Dielectric {
                emission: [2.0, 1.0, 0.5],
                ..
            }
        ));
        assert!(matches!(
            live.scene.entities[1].mesh,
            MeshDesc::Plane { size } if size == [3.0, 5.0]
        ));
        // Old payload without `emission`: defaults to off, still loads.
        assert!(matches!(
            live.scene.entities[1].material,
            MaterialDesc::Metal {
                emission: [0.0, 0.0, 0.0],
                ..
            }
        ));
        assert!(matches!(
            live.scene.entities[2].mesh,
            MeshDesc::Cylinder {
                radius,
                height,
                radial_segments,
            } if radius == 1.5 && height == 7.0 && radial_segments == 12
        ));
        assert!(matches!(
            live.scene.entities[2].material,
            MaterialDesc::Coat {
                emission: [0.0, 0.0, 1.0],
                ..
            }
        ));
    }

    #[test]
    fn box_and_emission_survive_transport_round_trip() {
        // Serialize components exactly as the server would, embed them in
        // an `/api/scene` payload, and parse back: the JSON transport
        // preserves Box geometry and emission bit-exactly.
        let mesh_json = serde_json::to_string(&MeshDesc::Box {
            size: [2.0, 4.0, 6.0],
        })
        .expect("mesh serializes");
        let material_json = serde_json::to_string(&MaterialDesc::Dielectric {
            base_color: [0.8, 0.2, 0.2],
            roughness: 0.4,
            emission: [2.0, 1.0, 0.5],
        })
        .expect("material serializes");
        let payload = format!(
            r#"{{"version": 1, "entities": [{{"components": {{
                "Transform": {{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]}},
                "Mesh": {mesh_json},
                "Material": {material_json}
            }} }}], "lights": [],
            "camera": {{"position":[0,0,5],"target":[0,0,0],"up":[0,1,0],"fov":60.0,"near":0.1,"far":100.0}} }}"#
        );
        let live = parse_scene_json(&payload).expect("round-tripped payload must parse");
        assert!(matches!(
            live.scene.entities[0].mesh,
            MeshDesc::Box { size } if size == [2.0, 4.0, 6.0]
        ));
        let MaterialDesc::Dielectric { emission, .. } = &live.scene.entities[0].material else {
            panic!(
                "expected Dielectric, got {:?}",
                live.scene.entities[0].material
            );
        };
        assert_eq!(*emission, [2.0, 1.0, 0.5]);
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
