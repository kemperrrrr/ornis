//! glTF [`Model`](ornis_gltf::Model) → editor [`Scene`](crate::scene::Scene)
//! flattening.
//!
//! [`scene_from_model`] is the flat view the editor instantiates: every
//! imported primitive becomes one entity at the node's world TRS. The
//! [`Model`](ornis_gltf::Model) itself keeps the node tree. Cameras, lights
//! and ambient come from host defaults — a glTF file carries no editor
//! lighting rig. Textured materials keep their scalar fallback until the
//! GPU upload step learns images (see `ornis-gltf` docs).

use ornis_gltf::{Model, ModelPrimitive};

use crate::scene::{
    CameraDesc, CameraProjection, EntityDesc, MaterialDesc, MeshDesc, Scene, TransformDesc,
};

/// Indices per triangle (flat soup alignment).
const TRIANGLE_VERTS: usize = 3;
/// Default ambient for glTF imports (matches the editor scene default).
const DEFAULT_AMBIENT_RGB: [f32; 3] = [0.10, 0.10, 0.15];
/// Default camera eye height (m).
const DEFAULT_CAMERA_EYE_Y: f32 = 2.5;
/// Default camera eye distance along +Z (m).
const DEFAULT_CAMERA_EYE_Z: f32 = 9.0;
/// Default vertical FOV (degrees).
const DEFAULT_CAMERA_FOV_DEG: f32 = 60.0;
/// Default near clip (m).
const DEFAULT_CAMERA_NEAR: f32 = 0.1;
/// Default far clip (m).
const DEFAULT_CAMERA_FAR: f32 = 100.0;

/// Default viewing camera for imported scenes (matches the editor default).
fn default_camera() -> CameraDesc {
    CameraDesc {
        position: glam::Vec3::new(0.0, DEFAULT_CAMERA_EYE_Y, DEFAULT_CAMERA_EYE_Z),
        target: glam::Vec3::ZERO,
        up: ornis_core::units::UnitVec3::Y,
        fov: ornis_core::units::Degrees::new(DEFAULT_CAMERA_FOV_DEG),
        near: ornis_core::units::Meters::new(DEFAULT_CAMERA_NEAR),
        far: ornis_core::units::Meters::new(DEFAULT_CAMERA_FAR),
        projection: CameraProjection::Perspective,
    }
}

/// Flattens a glTF [`Model`] into an editor [`Scene`].
///
/// Every mesh primitive becomes one entity whose transform is
/// [`Model::world_transform`](ornis_gltf::Model::world_transform) of its
/// node (the old baked world TRS). The entity name is the node name, or
/// `mesh_{node}_{ordinal}` when the node is unnamed. Scalar metalness is
/// the glTF factor: a clamped `1` keeps [`MaterialDesc::Metal`], every
/// other value is [`MaterialDesc::Dielectric`] carrying that number.
/// The result feeds the same replace path as `.ron` scenes.
pub fn scene_from_model(model: &Model) -> Scene {
    Scene {
        name: model.name.clone(),
        entities: model
            .primitives
            .iter()
            .enumerate()
            .map(|(index, primitive)| entity_from_primitive(model, index, primitive))
            .collect(),
        lights: Vec::new(),
        camera: default_camera(),
        ambient: DEFAULT_AMBIENT_RGB,
    }
}

fn entity_from_primitive(model: &Model, index: usize, primitive: &ModelPrimitive) -> EntityDesc {
    let (positions, indices) = primitive.mesh.clone().into_custom();
    let _typed: Vec<ornis_gltf::Triangle> = indices
        .chunks_exact(TRIANGLE_VERTS)
        .map(|corner| ornis_gltf::Triangle::from_raw([corner[0], corner[1], corner[2]]))
        .collect();
    let world = model.world_transform(primitive.node);
    EntityDesc {
        name: primitive_name(model, index, primitive),
        transform: TransformDesc::from_arrays(
            world.translation.to_array(),
            world.rotation.get().to_array(),
            world.scale.to_array(),
        ),
        mesh: MeshDesc::Custom { positions, indices },
        material: material_from_gltf(&primitive.material),
    }
}

/// Node name, else `mesh_{node}_{ordinal}` among that node's primitives.
fn primitive_name(model: &Model, index: usize, primitive: &ModelPrimitive) -> String {
    if let Some(name) = model
        .nodes
        .get(primitive.node.index())
        .and_then(|node| node.name.clone())
    {
        return name;
    }
    let ordinal = model.primitives[..index]
        .iter()
        .filter(|other| other.node == primitive.node)
        .count();
    format!("mesh_{}_{ordinal}", primitive.node.0)
}

fn material_from_gltf(material: &ornis_gltf::LoadedMaterial) -> MaterialDesc {
    // glTF roughness is already in [0, 1]; the clamp keeps malformed
    // payloads loadable (same leniency as the `serde` impl). Metalness
    // uses the same clamp (`Metallic::new`: NaN → 0, outside [0, 1] folds
    // in). A full metal keeps the authored `Metal` preset — that is the
    // old edge and glTF's default factor of 1. Every other factor,
    // including values the old `>= 0.5` switch called metal, stays on
    // `Dielectric` and keeps the number.
    let roughness = ornis_core::units::Clamped01::new(material.roughness);
    let metallic = ornis_core::Metallic::new(material.metallic);
    if metallic.get() == 1.0 {
        MaterialDesc::Metal {
            base_color: material.base_color,
            roughness,
            emission: material.emission,
            metallic,
        }
    } else {
        MaterialDesc::Dielectric {
            base_color: material.base_color,
            roughness,
            emission: material.emission,
            metallic,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_core::Transform;
    use ornis_gltf::{ImportStats, LoadedMaterial, LoadedMesh, ModelNode, ModelPrimitive, NodeIdx};

    fn mesh() -> LoadedMesh {
        LoadedMesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            indices: vec![0, 1, 2],
            normals: None,
            uvs: None,
            joints: None,
            weights: None,
        }
    }

    fn material() -> LoadedMaterial {
        LoadedMaterial {
            base_color: [0.9, 0.8, 0.2],
            metallic: 1.0,
            roughness: 0.2,
            emission: [0.0, 0.0, 0.0],
            base_color_texture: None,
            metallic_roughness_texture: None,
            emissive_texture: None,
        }
    }

    #[test]
    fn wiring_maps_geometry_and_scalar_materials() {
        let model = Model {
            name: "wired".into(),
            nodes: vec![ModelNode {
                name: Some("part".into()),
                parent: None,
                local: Transform {
                    translation: glam::Vec3::new(1.0, 2.0, 3.0),
                    rotation: ornis_core::UnitQuat::IDENTITY,
                    scale: glam::Vec3::splat(2.0),
                },
                primitives: vec![0],
                skin: None,
            }],
            roots: vec![NodeIdx(0)],
            primitives: vec![ModelPrimitive {
                node: NodeIdx(0),
                mesh: mesh(),
                material: material(),
            }],
            skins: Vec::new(),
            skel_clips: Vec::new(),
            anim_clips: Vec::new(),
            stats: ImportStats::default(),
        };
        let scene = scene_from_model(&model);
        assert_eq!(scene.name, "wired");
        assert_eq!(scene.entities.len(), 1);
        let entity = &scene.entities[0];
        assert_eq!(entity.name, "part");
        assert_eq!(entity.transform.translation.to_array(), [1.0, 2.0, 3.0]);
        assert_eq!(entity.transform.scale.to_array(), [2.0, 2.0, 2.0]);
        assert!(matches!(entity.mesh, MeshDesc::Custom { .. }));
        assert!(matches!(entity.material, MaterialDesc::Metal { .. }));
        assert_eq!(entity.material.metallic_units().get(), 1.0);
        assert!(scene.lights.is_empty());
    }

    #[test]
    fn partial_metallic_factor_is_not_snapped() {
        // 0.3 used to fall under the 0.5 threshold and render as a pure
        // dielectric. The factor, albedo, roughness and emission all stay.
        let mut source = material();
        source.base_color = [0.1, 0.2, 0.3];
        source.metallic = 0.3;
        source.roughness = 0.55;
        source.emission = [0.2, 0.0, 0.1];
        let desc = material_from_gltf(&source);
        assert!(matches!(
            desc,
            MaterialDesc::Dielectric {
                emission: [0.2, 0.0, 0.1],
                ..
            }
        ));
        assert!((desc.metallic_units().get() - 0.3).abs() < f32::EPSILON);
        assert_eq!(desc.base_color_units().as_array(), [0.1, 0.2, 0.3]);
        assert!((desc.roughness_units().get() - 0.55).abs() < f32::EPSILON);

        // Above the old threshold, still the factor — not a full metal.
        source.metallic = 0.9;
        let above = material_from_gltf(&source);
        assert!(matches!(above, MaterialDesc::Dielectric { .. }));
        assert!((above.metallic_units().get() - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn metallic_edges_keep_the_old_presets() {
        let mut source = material();
        source.metallic = 0.0;
        let dielectric = material_from_gltf(&source);
        assert!(matches!(dielectric, MaterialDesc::Dielectric { .. }));
        assert_eq!(dielectric.metallic_units().get(), 0.0);

        source.metallic = 1.0;
        let metal = material_from_gltf(&source);
        assert!(matches!(metal, MaterialDesc::Metal { .. }));
        assert_eq!(metal.metallic_units().get(), 1.0);
    }

    #[test]
    fn flatten_composes_parent_and_names_unnamed_nodes() {
        let model = Model {
            name: "nested".into(),
            nodes: vec![
                ModelNode {
                    name: None,
                    parent: None,
                    local: Transform::from_translation(glam::Vec3::new(10.0, 0.0, 0.0)),
                    primitives: vec![0],
                    skin: None,
                },
                ModelNode {
                    name: Some("child".into()),
                    parent: Some(NodeIdx(0)),
                    local: Transform::from_translation(glam::Vec3::new(0.0, 5.0, 0.0)),
                    primitives: vec![1],
                    skin: None,
                },
            ],
            roots: vec![NodeIdx(0)],
            primitives: vec![
                ModelPrimitive {
                    node: NodeIdx(0),
                    mesh: mesh(),
                    material: material(),
                },
                ModelPrimitive {
                    node: NodeIdx(1),
                    mesh: mesh(),
                    material: material(),
                },
            ],
            skins: Vec::new(),
            skel_clips: Vec::new(),
            anim_clips: Vec::new(),
            stats: ImportStats::default(),
        };
        let scene = scene_from_model(&model);
        assert_eq!(scene.entities[0].name, "mesh_0_0");
        assert_eq!(
            scene.entities[0].transform.translation.to_array(),
            [10.0, 0.0, 0.0]
        );
        assert_eq!(scene.entities[1].name, "child");
        assert_eq!(
            scene.entities[1].transform.translation.to_array(),
            [10.0, 5.0, 0.0]
        );
    }
}
