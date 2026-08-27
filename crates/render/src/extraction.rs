//! ECS-backed render extraction shared by native and WASM runtimes.
//!
//! A [`RenderWorld`] is a small render-domain view over the logical
//! [`ornis_core::Engine`]. Scene descriptions are deserialized at the
//! serialization boundary and inserted as `TransformDesc`, `MeshDesc` and
//! `MaterialDesc` component lanes. The scheduled [`RenderExtract`] system
//! then produces one backend-neutral [`RenderExtracted`] snapshot.
//!
//! GPU resources, cameras and lights remain owned by the platform renderer;
//! this module deliberately stops at CPU-side instance/material data. That
//! keeps the server/editor world authoritative while allowing a native or
//! browser client to build its own physical GPU representation.

use std::sync::Mutex;

use glam::{Mat4, Quat, Vec3};
use ornis_core::{
    Engine, Entity, OpenPBRMaterial, Resources, SmartStore, System, SystemAccess,
};

use crate::renderer::InstanceData;
use crate::scene::{MaterialDesc, MeshDesc, Scene, TransformDesc};

/// CPU-side render data extracted from ECS for one frame.
#[derive(Clone, Debug)]
pub struct RenderExtracted {
    /// Maximum sphere tessellation required by the extracted entities.
    pub mesh_params: (u32, u32),
    /// GPU-ready materials in the same order as [`Self::instances`].
    pub materials: Vec<OpenPBRMaterial>,
    /// Per-entity model/normal matrices and material indices.
    pub instances: Vec<InstanceData>,
}

impl Default for RenderExtracted {
    fn default() -> Self {
        Self {
            mesh_params: (32, 24),
            materials: Vec::new(),
            instances: Vec::new(),
        }
    }
}

/// A logical render world with the common [`Engine`] frame boundary.
///
/// `RenderWorld` is intentionally not a second authoritative game world. It
/// is the client-side ECS representation populated from a serialized
/// [`Scene`], which is then extracted before the platform-specific renderer
/// uploads data. The native showcase and the WASM viewport can therefore use
/// the same scene-to-ECS and ECS-to-extraction code.
pub struct RenderWorld {
    engine: Engine,
    entities: Vec<Entity>,
}

impl Default for RenderWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderWorld {
    /// Creates an empty render world and installs the shared extraction pass.
    pub fn new() -> Self {
        let mut engine = Engine::new();
        install_render_extract(&mut engine);
        Self {
            engine,
            entities: Vec::new(),
        }
    }

    /// Creates a render world populated from a serialized scene description.
    pub fn from_scene(scene: &Scene) -> Self {
        let mut world = Self::new();
        world.replace_scene(scene);
        world
    }

    /// Returns the logical engine used by this render world.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Returns the logical engine for controlled setup or custom systems.
    ///
    /// Scene replacement and frame execution should normally use
    /// [`Self::replace_scene`] and [`Self::run_frame`] so the entity list and
    /// extraction resource remain consistent.
    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// Number of render entities currently represented in the ECS.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Replaces the renderable ECS entities with `scene.entities`.
    ///
    /// Camera, lights and ambient values are intentionally not copied here:
    /// they are frame/view state owned by the caller, while this world owns
    /// only renderable component lanes. The next [`Self::run_frame`] refreshes
    /// the extracted snapshot.
    pub fn replace_scene(&mut self, scene: &Scene) {
        let previous = std::mem::take(&mut self.entities);
        if let Some(store) = self.engine.world().store() {
            for entity in previous {
                if store.is_alive(entity) {
                    store.destroy_entity(entity);
                }
            }
        }
        self.entities = insert_scene_entities(&mut self.engine, &scene.entities);
    }

    /// Publishes time and runs the shared extraction schedule for one frame.
    pub fn run_frame(&mut self, delta_seconds: f32) {
        self.engine.run_frame(delta_seconds);
    }

    /// Returns the latest scheduled extraction snapshot.
    ///
    /// Call [`Self::run_frame`] after mutating the ECS or replacing the scene
    /// to publish a fresh value. A newly created world returns the default
    /// empty snapshot until its first frame.
    pub fn extracted(&self) -> RenderExtracted {
        self.engine
            .world()
            .resources()
            .get::<Mutex<RenderExtracted>>()
            .expect("RenderWorld always installs RenderExtracted")
            .lock()
            .expect("render extraction lock")
            .clone()
    }
}

/// Installs the extraction resource and system in `engine`.
///
/// The stage is backend-neutral: it converts ECS scene components into
/// CPU-side [`InstanceData`] and [`OpenPBRMaterial`] tables. A native or WASM
/// renderer can upload the snapshot to its own GPU resources afterwards.
pub fn install_render_extract(engine: &mut Engine) {
    let _ = engine
        .world_mut()
        .insert(Mutex::new(RenderExtracted::default()));
    engine.schedule_mut().add_system(RenderExtract);
}

/// Extracts complete renderable entities from the ECS store.
///
/// Entities missing any of the three render components are skipped. Dense
/// lane order is used as the deterministic extraction order; each instance's
/// material index points at the material emitted in the same iteration.
pub fn extract_render_data(store: &SmartStore) -> RenderExtracted {
    let mut extracted = RenderExtracted::default();
    let Some(transforms) = store.read_lane::<TransformDesc>() else {
        return extracted;
    };
    let Some(meshes) = store.read_lane::<MeshDesc>() else {
        return extracted;
    };
    let Some(materials) = store.read_lane::<MaterialDesc>() else {
        return extracted;
    };

    for (&entity, transform) in transforms.entities.iter().zip(&transforms.data) {
        let Some(mesh) = meshes.get(entity) else {
            continue;
        };
        let Some(material) = materials.get(entity) else {
            continue;
        };
        let MeshDesc::Sphere {
            radius,
            segments,
            rings,
        } = mesh;
        extracted.mesh_params.0 = extracted.mesh_params.0.max(*segments);
        extracted.mesh_params.1 = extracted.mesh_params.1.max(*rings);
        let model = Mat4::from_scale_rotation_translation(
            Vec3::from_array(transform.scale) * *radius,
            normalized_rotation(transform.rotation),
            Vec3::from_array(transform.translation),
        );
        extracted.materials.push(material_to_gpu(material));
        extracted.instances.push(InstanceData {
            model_matrix: model,
            normal_matrix: model.inverse().transpose(),
            material_index: extracted.materials.len() as u32 - 1,
        });
    }
    extracted
}

/// The schedule system that turns the three ECS render lanes into a snapshot.
struct RenderExtract;

impl System for RenderExtract {
    fn name(&self) -> &'static str {
        "render_extract"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<TransformDesc>()
            .reads_lane::<MeshDesc>()
            .reads_lane::<MaterialDesc>()
            .writes::<Mutex<RenderExtracted>>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(output) = resources.get::<Mutex<RenderExtracted>>() else {
            return;
        };
        *output.lock().expect("render extraction lock") = extract_render_data(store);
    }
}

fn insert_scene_entities(engine: &mut Engine, entities: &[crate::scene::EntityDesc]) -> Vec<Entity> {
    let store = engine.world_mut().store_mut().expect("render world store");
    let mut handles = Vec::with_capacity(entities.len());
    for entity in entities {
        let handle = store.create_entity();
        store.insert(handle, entity.transform.clone());
        store.insert(handle, entity.mesh.clone());
        store.insert(handle, entity.material.clone());
        handles.push(handle);
    }
    handles
}

fn material_to_gpu(material: &MaterialDesc) -> OpenPBRMaterial {
    match material {
        MaterialDesc::Dielectric {
            base_color,
            roughness,
        } => {
            let mut output = OpenPBRMaterial::dielectric();
            output.base.color_rgb(*base_color);
            output.specular.roughness(*roughness);
            output
        }
        MaterialDesc::Metal {
            base_color,
            roughness,
        } => {
            let mut output = OpenPBRMaterial::metal();
            output.base.color_rgb(*base_color);
            output.specular.roughness(*roughness);
            output
        }
        MaterialDesc::Coat {
            base_color,
            coat_weight,
            coat_roughness,
        } => {
            let mut output = OpenPBRMaterial::coat();
            output.base.color_rgb(*base_color);
            output.coat.weight(*coat_weight);
            output.coat.roughness(*coat_roughness);
            output
        }
    }
}

fn normalized_rotation(rotation: [f32; 4]) -> Quat {
    let orientation = Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
    let length_squared = orientation.length_squared();
    if length_squared.is_finite() && length_squared > 1e-12 {
        orientation.normalize()
    } else {
        Quat::IDENTITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> Scene {
        Scene {
            name: "test".into(),
            entities: vec![crate::scene::EntityDesc {
                name: "sphere".into(),
                transform: TransformDesc {
                    translation: [1.0, 2.0, 3.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
                mesh: MeshDesc::Sphere {
                    radius: 2.0,
                    segments: 48,
                    rings: 32,
                },
                material: MaterialDesc::Metal {
                    base_color: [0.9, 0.8, 0.2],
                    roughness: 0.2,
                },
            }],
            lights: Vec::new(),
            camera: crate::scene::CameraDesc {
                position: [0.0, 2.5, 9.0],
                target: [0.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
                fov: 60.0,
                near: 0.1,
                far: 100.0,
            },
            ambient: [0.1, 0.1, 0.1],
        }
    }

    #[test]
    fn render_world_runs_shared_engine_extraction() {
        let mut world = RenderWorld::from_scene(&scene());
        assert_eq!(world.entity_count(), 1);
        world.run_frame(0.0);

        let extracted = world.extracted();
        assert_eq!(extracted.mesh_params, (48, 32));
        assert_eq!(extracted.materials.len(), 1);
        assert_eq!(extracted.instances.len(), 1);
        assert_eq!(extracted.instances[0].material_index, 0);
        assert_eq!(
            extracted.instances[0].model_matrix.w_axis.truncate(),
            Vec3::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn replacing_scene_destroys_previous_entities_before_extraction() {
        let mut world = RenderWorld::from_scene(&scene());
        let empty = Scene {
            entities: Vec::new(),
            ..scene()
        };
        world.replace_scene(&empty);
        world.run_frame(0.0);

        assert_eq!(world.entity_count(), 0);
        assert!(world.extracted().instances.is_empty());
    }

    #[test]
    fn incomplete_entities_are_skipped() {
        let mut engine = Engine::new();
        let entity = engine.world_mut().store_mut().expect("store").create_entity();
        engine
            .world_mut()
            .store_mut()
            .expect("store")
            .insert(entity, TransformDesc {
                translation: Vec3::ZERO.to_array(),
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: Vec3::ONE.to_array(),
            });
        install_render_extract(&mut engine);
        engine.run_frame(0.0);
        assert!(extract_render_data(engine.world().store().expect("store")).instances.is_empty());
    }
}
