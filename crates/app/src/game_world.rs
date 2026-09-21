//! Single game world: one type, two instances.
//!
//! [`GameWorld`] is the only scene-backed world type: it owns one [`Engine`] plus the scene entity list and a monotonic mutation counter, and serves both the authoritative native/editor host and the browser replica — the two instances differ only in how scenes cross the serialization boundary (IDEAS §28), never in the world type itself.
//!
//! System registration uses the staged [`Engine`](ornis_core::Engine) plan
//! directly ([`Stage`](ornis_core::Stage) via
//! [`Engine::add_stage_system`](ornis_core::Engine::add_stage_system) /
//! [`Engine::stage_schedule_mut`](ornis_core::Engine::stage_schedule_mut)):
//! `Input` systems consume the per-frame input, `Gameplay` systems run
//! intent and physics at the fixed step, `PostFrame` runs the variable
//! script tick, propagates poses and extracts audio/render views.

use glam::Vec3;
use ornis_core::{Engine, Entity};
use ornis_physics::RigidBody;
use ornis_render::FrameUpload;
use ornis_render::extraction::{RenderLights, extract_render_data};
use ornis_render::scene::{EntityDesc, Scene};

/// Centre of the hidden showcase static floor in world units.
const FLOOR_CENTER: [f32; 3] = [0.0, -2.0, 0.0];
/// Half-extents of the hidden showcase static floor in world units.
const FLOOR_HALF_EXTENTS: [f32; 3] = [20.0, 1.0, 20.0];

/// Single scene-backed game world: one [`Engine`] plus its scene entities.
///
/// The native showcase and the editor server run the authoritative instance;
/// the browser viewport runs a replica instance populated from serialized
/// snapshots across the boundary (IDEAS §28). Physics, audio, scripting and
/// GPU state stay specialized resources installed by the platform through
/// [`GameWorld::engine_mut`], never duplicated here.
pub struct GameWorld {
    engine: Engine,
    entities: Vec<Entity>,
    version: u64,
}

impl Default for GameWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl GameWorld {
    /// Creates an empty world with no scene entities.
    pub fn new() -> Self {
        Self {
            engine: Engine::new(),
            entities: Vec::new(),
            version: 0,
        }
    }

    /// Creates a world populated from a serialized scene description.
    pub fn from_scene(scene: &Scene) -> Self {
        let mut world = Self::new();
        world.replace_scene(scene);
        world
    }

    /// Monotonic scene-mutation counter: bumped by every
    /// [`Self::replace_scene`] (and therefore once by [`Self::from_scene`]).
    /// Frame execution alone never bumps it.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Returns the shared engine for read-only inspection.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Returns the shared engine for platform setup (orbit camera, physics,
    /// gameplay, audio, GPU resources) and custom systems.
    ///
    /// Scene replacement and frame execution should normally use
    /// [`Self::replace_scene`] and [`Self::frame`] so the entity list and
    /// extraction resource remain consistent.
    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// Number of scene entities currently represented in the ECS.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns the handles of entities populated from the current scene.
    ///
    /// The slice excludes auxiliary entities inserted through
    /// [`Self::engine_mut`], which lets a platform attach hidden runtime
    /// components such as physics bodies without changing the scene count.
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Replaces the scene entities with `scene.entities` and publishes the
    /// scene lighting as the [`RenderLights`] resource.
    ///
    /// The camera stays frame/view state owned by the caller; lights and
    /// ambient are world state now. Bumps [`Self::version`] so snapshot
    /// clients can cheaply detect the replacement.
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
        let _ = self
            .engine
            .world_mut()
            .insert(RenderLights::from_scene(scene));
        self.version = self.version.saturating_add(1);
    }

    /// Reads the frame payload directly from the component lanes.
    ///
    /// Equivalent to calling [`extract_render_data`] on this world's
    /// store; provided for callers that own the [`GameWorld`].
    pub fn frame_upload(&self) -> FrameUpload {
        extract_render_data(self.engine.world().store().expect("game world store"))
    }

    /// Runs one frame and returns its CPU-side render payload.
    ///
    /// The engine advances bounded fixed steps first, then the
    /// once-per-frame schedule; extraction observes the final poses.
    /// Platform GPU presentation (`RenderFrame3D` / WASM adapter) consumes
    /// the returned [`FrameUpload`] through its own existing path.
    pub fn frame(&mut self, delta_seconds: f32) -> FrameUpload {
        self.engine.run_frame(delta_seconds);
        self.frame_upload()
    }
}

fn insert_scene_entities(engine: &mut Engine, entities: &[EntityDesc]) -> Vec<Entity> {
    let store = engine.world_mut().store_mut().expect("game world store");
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

/// Spawns the hidden showcase static floor into `engine`.
///
/// The floor carries only a physics component, so it never enters the frame
/// upload; it exists so dynamic showcase bodies have ground to rest on.
pub fn spawn_static_floor(engine: &mut Engine) -> Entity {
    let store = engine.world_mut().store_mut().expect("game runtime store");
    let floor = store.create_entity();
    store.insert(
        floor,
        RigidBody::new_box(
            Vec3::from_array(FLOOR_CENTER),
            Vec3::from_array(FLOOR_HALF_EXTENTS),
            0.0,
        ),
    );
    floor
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_core::{Stage as CoreStage, Time};
    use ornis_render::scene::{CameraDesc, MaterialDesc, MeshDesc, TransformDesc};

    /// Minimal probe system for staged-plan lookups.
    struct StageProbe(&'static str);

    impl ornis_core::System for StageProbe {
        fn name(&self) -> &'static str {
            self.0
        }
        fn access(&self) -> ornis_core::SystemAccess {
            ornis_core::SystemAccess::new()
        }
        fn run(&self, _: &ornis_core::Resources) {}
    }

    fn two_sphere_scene() -> Scene {
        let entity = |name: &str, x: f32| EntityDesc {
            name: name.into(),
            transform: TransformDesc {
                translation: [x, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
            mesh: MeshDesc::Sphere {
                radius: 1.0,
                segments: 16,
                rings: 8,
            },
            material: MaterialDesc::Dielectric {
                base_color: [0.8, 0.2, 0.2],
                roughness: 0.5,
                emission: [0.0, 0.0, 0.0],
            },
        };
        Scene {
            name: "runtime-test".into(),
            entities: vec![entity("a", -1.0), entity("b", 1.0)],
            lights: Vec::new(),
            camera: CameraDesc {
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

    fn scene() -> Scene {
        Scene {
            name: "test".into(),
            entities: vec![EntityDesc {
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
                    emission: [0.0, 0.0, 0.0],
                },
            }],
            lights: Vec::new(),
            camera: CameraDesc {
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
    fn frame_runs_shared_host_and_extracts_scene() {
        let scene = two_sphere_scene();
        let mut world = GameWorld::from_scene(&scene);
        assert_eq!(world.entity_count(), 2);
        assert_eq!(world.entities().len(), 2);

        let upload = world.frame(1.0 / 60.0);
        assert_eq!(upload.instances.len(), 2);
        // Identical materials dedup to one table entry (render extraction canon).
        assert_eq!(upload.materials.len(), 1);
        assert_eq!(upload.instances[0].material_index, 0);
        assert_eq!(upload.instances[1].material_index, 0);
        let time = world
            .engine()
            .world()
            .resources()
            .get::<Time>()
            .expect("engine publishes Time");
        assert_eq!(time.frame(), 1);
    }

    #[test]
    fn empty_frame_advances_clock_without_content() {
        let mut world = GameWorld::new();
        let upload = world.frame(1.0 / 60.0);
        assert!(upload.instances.is_empty());
        assert!(upload.materials.is_empty());
        let time = world
            .engine()
            .world()
            .resources()
            .get::<Time>()
            .expect("engine publishes Time");
        assert_eq!(time.frame(), 1);
    }

    #[test]
    fn hidden_floor_never_enters_frame_upload() {
        let scene = two_sphere_scene();
        let mut world = GameWorld::from_scene(&scene);
        let floor = spawn_static_floor(world.engine_mut());
        assert!(
            world
                .engine()
                .world()
                .store()
                .expect("store")
                .is_alive(floor)
        );
        assert_eq!(world.entity_count(), 2);

        let upload = world.frame(1.0 / 60.0);
        assert_eq!(upload.instances.len(), 2);
        assert!(upload.custom_meshes.is_empty());
    }

    #[test]
    fn replace_scene_swaps_entities() {
        let mut world = GameWorld::from_scene(&two_sphere_scene());
        let empty = Scene {
            entities: Vec::new(),
            ..two_sphere_scene()
        };
        world.replace_scene(&empty);
        assert_eq!(world.entity_count(), 0);
        assert!(world.frame(0.0).instances.is_empty());
    }

    #[test]
    fn version_counts_scene_replacements_only() {
        let mut world = GameWorld::new();
        assert_eq!(world.version(), 0);
        world.frame(1.0 / 60.0);
        assert_eq!(world.version(), 0);
        world.replace_scene(&two_sphere_scene());
        assert_eq!(world.version(), 1);
        world.replace_scene(&two_sphere_scene());
        assert_eq!(world.version(), 2);
        let from_scene = GameWorld::from_scene(&two_sphere_scene());
        assert_eq!(from_scene.version(), 1);
    }

    #[test]
    fn core_stage_names_are_stable() {
        assert_eq!(CoreStage::PreUpdate.name(), "pre_update");
        assert_eq!(CoreStage::Input.name(), "input");
        assert_eq!(CoreStage::Gameplay.name(), "gameplay");
        assert_eq!(CoreStage::PostFrame.name(), "post_frame");
    }

    #[test]
    fn each_registered_system_lives_in_its_staged_plan() {
        use crate::install_unified_runtime;
        let mut engine = Engine::new();
        install_unified_runtime(&mut engine);
        // Every system the unified runtime registers must sit in its
        // staged plan (`Gameplay` delegates to the fixed schedule,
        // `PostFrame` to the once-per-frame schedule).
        let gameplay = engine.stage_schedule(CoreStage::Gameplay).mermaid();
        let post_frame = engine.stage_schedule(CoreStage::PostFrame).mermaid();
        for name in ["physics_push", "velocity_to_body"] {
            assert!(
                gameplay.contains(name),
                "{name} should be registered in the Gameplay stage"
            );
        }
        for name in ["player_input", "transform_update", "body_to_transform"] {
            assert!(
                post_frame.contains(name),
                "{name} should be registered in the PostFrame stage"
            );
        }
    }

    #[test]
    fn stage_registration_channels_route_to_real_plans() {
        use ornis_core::Engine;
        let mut engine = Engine::new();
        // Each registrable system goes through the staged channels:
        // `add_stage_system` and `stage_schedule_mut`.
        engine.add_stage_system(CoreStage::Input, StageProbe("player_input"));
        engine
            .stage_schedule_mut(CoreStage::Gameplay)
            .add_system(StageProbe("physics_step"));

        assert_eq!(engine.stage_schedule(CoreStage::Input).len(), 1);
        assert!(
            engine
                .stage_schedule(CoreStage::Input)
                .mermaid()
                .contains("player_input")
        );
        let gameplay = engine.stage_schedule(CoreStage::Gameplay);
        assert_eq!(gameplay.len(), engine.fixed_schedule().len());
        assert!(gameplay.mermaid().contains("physics_step"));
        assert!(engine.stage_schedule(CoreStage::PreUpdate).is_empty());
    }

    #[test]
    fn game_world_extracts_scene_entities_from_lanes() {
        let mut world = GameWorld::from_scene(&scene());
        assert_eq!(world.entity_count(), 1);
        world.frame(0.0);

        let extracted = world.frame_upload();
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
        let mut world = GameWorld::from_scene(&scene());
        let empty = Scene {
            entities: Vec::new(),
            ..scene()
        };
        world.replace_scene(&empty);
        world.frame(0.0);

        assert_eq!(world.entity_count(), 0);
        assert!(world.frame_upload().instances.is_empty());
    }

    #[test]
    fn frame_upload_matches_the_direct_lane_canon() {
        // X1/X4 (Extract-free) data gate: `extract_render_data` is the
        // single canon — `GameWorld::frame_upload` and a plain canon
        // call on the same store must agree for a scene mixing all three
        // material kinds and varied tessellation (instances field-wise
        // over the glam matrices, materials by `Debug` bytes).
        let varied = Scene {
            name: "oracle".into(),
            entities: vec![
                EntityDesc {
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
                        emission: [0.0, 0.0, 0.0],
                    },
                },
                EntityDesc {
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
                        emission: [0.0, 0.0, 0.0],
                    },
                },
                EntityDesc {
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
                        emission: [0.0, 0.0, 0.0],
                    },
                },
            ],
            lights: Vec::new(),
            camera: CameraDesc {
                position: [0.0, 2.5, 9.0],
                target: [0.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
                fov: 60.0,
                near: 0.1,
                far: 100.0,
            },
            ambient: [0.1, 0.1, 0.1],
        };
        let mut world = GameWorld::from_scene(&varied);
        world.frame(0.0);

        // X1/X4 canon: one direct lane read covers all material kinds;
        // `GameWorld::frame_upload` must agree with the plain canon
        // call on the same store.
        let snapshot = world.frame_upload();
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
            assert_eq!(format!("{direct:?}"), format!("{snapshot:?}"));
        }
    }

    #[test]
    fn max_mesh_params_is_the_extraction_mesh_canon() {
        // X2 canon tie: the mesh re-create criterion must be exactly the
        // oracle's `mesh_params` — max tessellation over COMPLETE entities
        // only, floor (32, 24). The incomplete entity (mesh + material,
        // no transform, 96/64) must not push the maximum.
        use ornis_render::{extract_render_data as canon, max_mesh_params};
        let complete = Scene {
            name: "canon".into(),
            entities: vec![
                EntityDesc {
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
                        emission: [0.0, 0.0, 0.0],
                    },
                },
                EntityDesc {
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
                        emission: [0.0, 0.0, 0.0],
                    },
                },
            ],
            lights: Vec::new(),
            camera: CameraDesc {
                position: [0.0, 2.5, 9.0],
                target: [0.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
                fov: 60.0,
                near: 0.1,
                far: 100.0,
            },
            ambient: [0.1, 0.1, 0.1],
        };
        let mut world = GameWorld::from_scene(&complete);
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
                emission: [0.0, 0.0, 0.0],
            },
        );
        world.frame(0.0);

        let store = world.engine().world().store().expect("store");
        let extracted = canon(store);
        assert_eq!(extracted.mesh_params, (48, 32));
        assert_eq!(max_mesh_params(store), extracted.mesh_params);
    }

    #[test]
    fn replace_scene_publishes_scene_lighting_as_resource() {
        // X3: the scene loader owns lights/ambient — replacing a scene
        // must publish them as the `RenderLights` resource.
        use ornis_render::scene::LightDesc;
        let world = GameWorld::from_scene(&Scene {
            lights: vec![LightDesc::Directional {
                direction: [0.0, -1.0, 0.0],
                intensity: 2.0,
                color: [1.0, 0.9, 0.8],
                shadow: false,
            }],
            ambient: [0.2, 0.2, 0.2],
            ..scene()
        });
        let lights = world
            .engine()
            .world()
            .resources()
            .get::<RenderLights>()
            .expect("scene loader publishes RenderLights");
        assert_eq!(lights.ambient, [0.2, 0.2, 0.2]);
        assert_eq!(lights.lights.len(), 1);
        assert!(matches!(
            lights.set_lights_args().as_slice(),
            [LightDesc::Directional {
                direction: [0.0, -1.0, 0.0],
                intensity: v,
                color: [1.0, 0.9, 0.8],
                shadow: false,
            }] if *v == 2.0
        ));
    }
}
