//! glTF→[`Scene`](crate::scene::Scene) wiring.
//!
//! [`scene_from_gltf`] is the first real consumer of `ornis-gltf`: loaded
//! geometry and scalar materials become scene descriptions; textured
//! materials keep their scalar fallback until the GPU upload step learns
//! images (see `ornis-gltf` docs). Cameras, lights and ambient come from
//! host defaults — a glTF file carries no editor lighting rig.

use ornis_gltf::LoadedScene;

use crate::scene::{CameraDesc, EntityDesc, MaterialDesc, MeshDesc, Scene, TransformDesc};

/// Default viewing camera for imported scenes (matches the editor default).
fn default_camera() -> CameraDesc {
    CameraDesc {
        position: [0.0, 2.5, 9.0],
        target: [0.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        fov: 60.0,
        near: 0.1,
        far: 100.0,
    }
}

/// Converts a loaded glTF scene into an editor [`Scene`].
///
/// Every mesh primitive becomes one entity with a `Custom` soup mesh;
/// scalar PBR parameters become `Dielectric`/`Metal` by the loader's
/// metallic threshold. Textured slots keep the scalar fallback (see the
/// module docs). The result feeds the same replace path as `.ron` scenes.
pub fn scene_from_gltf(loaded: &LoadedScene) -> Scene {
    Scene {
        name: loaded.name.clone(),
        entities: loaded.entities.iter().map(entity_from_gltf).collect(),
        lights: Vec::new(),
        camera: default_camera(),
        ambient: [0.10, 0.10, 0.15],
    }
}

fn entity_from_gltf(entity: &ornis_gltf::LoadedEntity) -> EntityDesc {
    // `into_custom` already round-trips the flat list through
    // `Triangle::from_raw`/`as_u32`, so the soup stays triple-aligned by
    // construction; the typed view below is the loud counterpart.
    let (positions, indices) = entity.mesh.clone().into_custom();
    let _typed: Vec<ornis_gltf::Triangle> = indices
        .chunks_exact(3)
        .map(|c| ornis_gltf::Triangle::from_raw([c[0], c[1], c[2]]))
        .collect();
    EntityDesc {
        name: entity.name.clone(),
        transform: TransformDesc {
            translation: entity.translation,
            rotation: entity.rotation,
            scale: entity.scale,
        },
        mesh: MeshDesc::Custom { positions, indices },
        material: material_from_gltf(&entity.material),
    }
}

fn material_from_gltf(material: &ornis_gltf::LoadedMaterial) -> MaterialDesc {
    // glTF roughness is already in [0, 1]; the clamp keeps malformed
    // payloads loadable (same leniency as the `serde` impl).
    let roughness = ornis_core::units::Clamped01::new(material.roughness);
    if material.is_metallic() {
        MaterialDesc::Metal {
            base_color: material.base_color,
            roughness,
            emission: material.emission,
        }
    } else {
        MaterialDesc::Dielectric {
            base_color: material.base_color,
            roughness,
            emission: material.emission,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wiring_maps_geometry_and_scalar_materials() {
        let scene = ornis_gltf::LoadedScene {
            name: "wired".into(),
            entities: vec![ornis_gltf::LoadedEntity {
                name: "part".into(),
                translation: [1.0, 2.0, 3.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [2.0, 2.0, 2.0],
                mesh: ornis_gltf::LoadedMesh {
                    positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    indices: vec![0, 1, 2],
                    normals: None,
                    uvs: None,
                    joints: None,
                    weights: None,
                },
                skin: None,
                material: ornis_gltf::LoadedMaterial {
                    base_color: [0.9, 0.8, 0.2],
                    metallic: 1.0,
                    roughness: 0.2,
                    emission: [0.0, 0.0, 0.0],
                    base_color_texture: None,
                    metallic_roughness_texture: None,
                    emissive_texture: None,
                },
            }],
            skins: Vec::new(),
            skel_clips: Vec::new(),
            anim_clips: Vec::new(),
            stats: ornis_gltf::ImportStats::default(),
        };
        let scene = scene_from_gltf(&scene);
        assert_eq!(scene.name, "wired");
        assert_eq!(scene.entities.len(), 1);
        let entity = &scene.entities[0];
        assert_eq!(entity.transform.translation, [1.0, 2.0, 3.0]);
        assert!(matches!(entity.mesh, MeshDesc::Custom { .. }));
        assert!(matches!(entity.material, MaterialDesc::Metal { .. }));
        assert!(scene.lights.is_empty());
    }
}
