//! ECS-backed render extraction shared by native and WASM runtimes.
//!
//! A [`RenderWorld`] is a small render-domain view over the logical
//! [`ornis_core::Engine`]. Scene descriptions are deserialized at the
//! serialization boundary and inserted as `TransformDesc`, `MeshDesc` and
//! `MaterialDesc` component lanes. The scheduled [`install_render_extract`]
//! system then produces one backend-neutral [`RenderExtracted`] snapshot.
//!
//! GPU resources, cameras and lights remain owned by the platform renderer;
//! this module deliberately stops at CPU-side instance/material data. That
//! keeps the server/editor world authoritative while allowing a native or
//! browser client to build its own physical GPU representation.

use std::sync::Mutex;

use glam::{Mat4, Quat, Vec3};
use ornis_core::{Engine, Entity, OpenPBRMaterial, Resources, SmartStore, System, SystemAccess};

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

/// Tessellation floor when no complete renderable entity asks for more
/// (the `RenderExtracted::default` `mesh_params`).
const DEFAULT_MESH_PARAMS: (u32, u32) = (32, 24);

impl Default for RenderExtracted {
    fn default() -> Self {
        Self {
            mesh_params: DEFAULT_MESH_PARAMS,
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

    /// Returns the handles of entities populated from the current scene.
    ///
    /// The slice excludes auxiliary entities inserted through
    /// [`Self::engine_mut`], which lets a platform attach hidden runtime
    /// components such as physics bodies without changing the render count.
    pub fn entities(&self) -> &[Entity] {
        &self.entities
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

/// Maximum sphere tessellation over complete renderable entities — the
/// GPU mesh re-create criterion (X2, Extract-free).
///
/// The same canon as [`extract_render_data`]: entities missing any of
/// the three render components are skipped (even for the maximum), and
/// the result never falls below the `RenderExtracted::default` floor
/// (32, 24). Iterator form: the lane walk lives in closures (the
/// sanctioned lenient form, `rustqual.toml`).
pub fn max_mesh_params(store: &SmartStore) -> (u32, u32) {
    let Some(transforms) = store.read_lane::<TransformDesc>() else {
        return DEFAULT_MESH_PARAMS;
    };
    let Some(meshes) = store.read_lane::<MeshDesc>() else {
        return DEFAULT_MESH_PARAMS;
    };
    let Some(materials) = store.read_lane::<MaterialDesc>() else {
        return DEFAULT_MESH_PARAMS;
    };
    transforms
        .entities
        .iter()
        // Complete entities only: all three render components present.
        .filter(|&&entity| {
            meshes.get(entity).is_some() && materials.get(entity).is_some()
        })
        .filter_map(|&entity| meshes.get(entity))
        .fold(DEFAULT_MESH_PARAMS, |params, mesh| {
            let MeshDesc::Sphere { segments, rings, .. } = mesh;
            (params.0.max(*segments), params.1.max(*rings))
        })
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

fn insert_scene_entities(
    engine: &mut Engine,
    entities: &[crate::scene::EntityDesc],
) -> Vec<Entity> {
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
    fn direct_lane_read_is_byte_equal_to_scheduled_snapshot() {
        // X1 (Extract-free) oracle gate: `extract_render_data` stays the
        // single canon — a direct lane read at submit time must be
        // byte-equal to the snapshot the scheduled `RenderExtract` system
        // published for the same frame (materials compared as Pod bytes,
        // instances field-wise over the glam matrices).
        let varied = Scene {
            name: "oracle".into(),
            entities: vec![
                crate::scene::EntityDesc {
                    name: "dielectric".into(),
                    transform: TransformDesc {
                        translation: [1.0, 2.0, 3.0],
                        rotation: [0.0, 0.0, 0.0, 1.0],
                        scale: [1.0, 1.0, 1.0],
                    },
                    mesh: MeshDesc::Sphere {
                        radius: 2.0,
                        segments: 24,
                        rings: 16,
                    },
                    material: MaterialDesc::Dielectric {
                        base_color: [0.8, 0.2, 0.2],
                        roughness: 0.4,
                    },
                },
                crate::scene::EntityDesc {
                    name: "metal".into(),
                    transform: TransformDesc {
                        translation: [-1.0, 0.0, 2.0],
                        rotation: [0.0, 0.0, 0.0, 1.0],
                        scale: [2.0, 2.0, 2.0],
                    },
                    mesh: MeshDesc::Sphere {
                        radius: 0.5,
                        segments: 48,
                        rings: 32,
                    },
                    material: MaterialDesc::Metal {
                        base_color: [0.9, 0.8, 0.2],
                        roughness: 0.2,
                    },
                },
                crate::scene::EntityDesc {
                    name: "coat".into(),
                    transform: TransformDesc {
                        translation: [0.0, 5.0, -3.0],
                        rotation: [0.3, 0.2, 0.1, 0.9],
                        scale: [1.0, 1.0, 1.0],
                    },
                    mesh: MeshDesc::Sphere {
                        radius: 1.0,
                        segments: 32,
                        rings: 24,
                    },
                    material: MaterialDesc::Coat {
                        base_color: [0.2, 0.4, 0.9],
                        coat_weight: 0.7,
                        coat_roughness: 0.1,
                    },
                },
            ],
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
        };
        let mut world = RenderWorld::from_scene(&varied);
        world.run_frame(0.0);

        let snapshot = world.extracted();
        let direct = extract_render_data(world.engine().world().store().expect("store"));
        assert_eq!(snapshot.mesh_params, (48, 32));
        assert_eq!(snapshot.instances.len(), 3);
        assert_eq!(direct.mesh_params, snapshot.mesh_params);
        assert_eq!(direct.instances.len(), snapshot.instances.len());
        for (direct, snapshot) in direct.instances.iter().zip(&snapshot.instances) {
            assert_eq!(direct.model_matrix, snapshot.model_matrix);
            assert_eq!(direct.normal_matrix, snapshot.normal_matrix);
            assert_eq!(direct.material_index, snapshot.material_index);
        }
        assert_eq!(direct.materials.len(), snapshot.materials.len());
        for (direct, snapshot) in direct.materials.iter().zip(&snapshot.materials) {
            assert_eq!(bytemuck::bytes_of(direct), bytemuck::bytes_of(snapshot));
        }
    }

    #[test]
    fn incomplete_entities_are_skipped() {
        let mut engine = Engine::new();
        let entity = engine
            .world_mut()
            .store_mut()
            .expect("store")
            .create_entity();
        engine.world_mut().store_mut().expect("store").insert(
            entity,
            TransformDesc {
                translation: Vec3::ZERO.to_array(),
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: Vec3::ONE.to_array(),
            },
        );
        install_render_extract(&mut engine);
        engine.run_frame(0.0);
        assert!(
            extract_render_data(engine.world().store().expect("store"))
                .instances
                .is_empty()
        );
    }

    #[test]
    fn max_mesh_params_is_the_extraction_mesh_canon() {
        // X2 canon tie: the mesh re-create criterion must be exactly the
        // oracle's `mesh_params` — max tessellation over COMPLETE entities
        // only, floor (32, 24). The incomplete entity (mesh + material,
        // no transform, 96/64) must not push the maximum.
        let complete = Scene {
            name: "canon".into(),
            entities: vec![
                crate::scene::EntityDesc {
                    name: "fine".into(),
                    transform: TransformDesc {
                        translation: [0.0, 0.0, 0.0],
                        rotation: [0.0, 0.0, 0.0, 1.0],
                        scale: [1.0, 1.0, 1.0],
                    },
                    mesh: MeshDesc::Sphere {
                        radius: 1.0,
                        segments: 48,
                        rings: 32,
                    },
                    material: MaterialDesc::Metal {
                        base_color: [0.9, 0.8, 0.2],
                        roughness: 0.2,
                    },
                },
                crate::scene::EntityDesc {
                    name: "coarse".into(),
                    transform: TransformDesc {
                        translation: [2.0, 0.0, 0.0],
                        rotation: [0.0, 0.0, 0.0, 1.0],
                        scale: [1.0, 1.0, 1.0],
                    },
                    mesh: MeshDesc::Sphere {
                        radius: 1.0,
                        segments: 16,
                        rings: 12,
                    },
                    material: MaterialDesc::Dielectric {
                        base_color: [0.2, 0.8, 0.2],
                        roughness: 0.5,
                    },
                },
            ],
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
        };
        let mut world = RenderWorld::from_scene(&complete);
        // Incomplete entity: mesh + material without a transform lane
        // entry — a high tessellation that must NOT move the maximum.
        let store = world.engine_mut().world_mut().store_mut().expect("store");
        let incomplete = store.create_entity();
        store.insert(
            incomplete,
            MeshDesc::Sphere {
                radius: 1.0,
                segments: 96,
                rings: 64,
            },
        );
        store.insert(
            incomplete,
            MaterialDesc::Coat {
                base_color: [0.2, 0.4, 0.9],
                coat_weight: 0.7,
                coat_roughness: 0.1,
            },
        );
        world.run_frame(0.0);

        let store = world.engine().world().store().expect("store");
        let extracted = extract_render_data(store);
        assert_eq!(extracted.mesh_params, (48, 32));
        assert_eq!(max_mesh_params(store), extracted.mesh_params);
    }
}
