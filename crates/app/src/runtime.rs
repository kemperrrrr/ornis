//! Headless game frame host over the shared engine world.
//!
//! [`GameRuntime`] is the minimal native/editor frame facade: it owns one
//! scene-backed [`RenderWorld`](ornis_render::RenderWorld) (and through it the single
//! [`ornis_core::Engine`]/[`World`](ornis_core::World)), while physics,
//! audio, scripting and GPU state stay specialized resources installed by
//! the platform. One frame is [`GameRuntime::frame`] = `run_frame` (bounded
//! fixed steps plus the once-per-frame schedule) + CPU-side render
//! extraction ([`FrameUpload`](ornis_render::FrameUpload)). GPU presentation stays platform-owned
//! (`RenderFrame3D` natively, the WASM adapter in the browser); the browser
//! keeps its snapshot-client loop behind the serialization boundary
//! (IDEAS §28) and does not use this host.
//!
//! [`GameStage`] fixes the gameplay-stage vocabulary (`PreUpdate` / `Input` /
//! `Gameplay` / `PostFrame`) over the existing fixed/frame schedule split
//! without changing it: see [`GameStage::stage_for_system`]. A fully staged
//! schedule is the documented next step, not part of this facade.

use glam::Vec3;
use ornis_core::{Engine, Entity};
use ornis_physics::RigidBody;
use ornis_render::{FrameUpload, RenderWorld, scene::Scene};

/// Centre of the hidden showcase static floor in world units.
const FLOOR_CENTER: [f32; 3] = [0.0, -2.0, 0.0];
/// Half-extents of the hidden showcase static floor in world units.
const FLOOR_HALF_EXTENTS: [f32; 3] = [20.0, 1.0, 20.0];

/// Minimal game frame host: scene lanes plus one shared [`Engine`].
///
/// The render world holds the `Transform`/`Mesh`/`Material` lanes and the
/// engine; gameplay (`Player`/`Position`/`Velocity`), physics (`RigidBody`)
/// and GPU resources are attached through [`GameRuntime::engine_mut`] by the
/// platform, never duplicated here.
pub struct GameRuntime {
    render: RenderWorld,
}

impl Default for GameRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl GameRuntime {
    /// Creates an empty runtime with no scene entities.
    pub fn new() -> Self {
        Self {
            render: RenderWorld::new(),
        }
    }

    /// Creates a runtime populated from a serialized scene description.
    pub fn from_scene(scene: &Scene) -> Self {
        Self {
            render: RenderWorld::from_scene(scene),
        }
    }

    /// Replaces the scene entities and publishes the scene lighting.
    pub fn replace_scene(&mut self, scene: &Scene) {
        self.render.replace_scene(scene);
    }

    /// Returns the shared engine for read-only inspection.
    pub fn engine(&self) -> &Engine {
        self.render.engine()
    }

    /// Returns the shared engine for platform setup (orbit camera, physics,
    /// gameplay, audio, GPU resources) and custom systems.
    pub fn engine_mut(&mut self) -> &mut Engine {
        self.render.engine_mut()
    }

    /// Number of scene entities currently represented in the ECS.
    pub fn entity_count(&self) -> usize {
        self.render.entity_count()
    }

    /// Handles of the entities populated from the current scene.
    pub fn entities(&self) -> &[Entity] {
        self.render.entities()
    }

    /// Reads the frame payload directly from the component lanes.
    pub fn frame_upload(&self) -> FrameUpload {
        self.render.frame_upload()
    }

    /// Runs one frame and returns its CPU-side render payload.
    ///
    /// The engine advances bounded fixed steps first, then the
    /// once-per-frame schedule; extraction observes the final poses.
    /// Platform GPU presentation (`RenderFrame3D` / WASM adapter) consumes
    /// the returned [`FrameUpload`] through its own existing path.
    pub fn frame(&mut self, delta_seconds: f32) -> FrameUpload {
        self.render.run_frame(delta_seconds);
        self.render.frame_upload()
    }
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

/// Named gameplay stages over the existing fixed/frame schedule split.
///
/// Mapping (see [`GameStage::stage_for_system`]): `Input` systems consume the
/// per-frame [`InputState`](ornis_core::InputState); `Gameplay` systems run
/// intent and physics at the fixed step (plus the variable script tick);
/// `PostFrame` propagates poses and extracts audio/render views.
/// `PreUpdate` is reserved for between-frame input ingest
/// (`apply_snapshot` / `apply_browser_input`), which is not a scheduled
/// system today. The fixed/frame schedules are unchanged; splitting them
/// into staged schedules is the next step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameStage {
    /// Between-frame input ingest (reserved, no scheduled system yet).
    PreUpdate,
    /// Once-per-frame input consumers.
    Input,
    /// Fixed-step intent and physics plus the variable script tick.
    Gameplay,
    /// Pose propagation and audio/render extraction views.
    PostFrame,
}

impl GameStage {
    /// Short stable label of the stage.
    pub fn name(self) -> &'static str {
        match self {
            GameStage::PreUpdate => "pre_update",
            GameStage::Input => "input",
            GameStage::Gameplay => "gameplay",
            GameStage::PostFrame => "post_frame",
        }
    }

    /// Classifies a scheduled system by its registration name.
    ///
    /// Returns `None` for unknown systems; classification never affects
    /// execution order.
    pub fn stage_for_system(system_name: &str) -> Option<GameStage> {
        match system_name {
            "player_input" | "orbit_camera_input" => Some(GameStage::Input),
            "physics_push" | "velocity_to_body" | "physics_sync_in" | "physics_step"
            | "physics_sync_out" | "script_tick" => Some(GameStage::Gameplay),
            "transform_update"
            | "body_to_transform"
            | "unified_render_extract"
            | "render_snapshot"
            | "audio_step"
            | "audio_listener_sync"
            | "render_mesh"
            | "render_submit"
            | "render_present"
            | "render_flush" => Some(GameStage::PostFrame),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_core::Time;
    use ornis_render::scene::{CameraDesc, EntityDesc, MaterialDesc, MeshDesc};

    fn two_sphere_scene() -> Scene {
        let entity = |name: &str, x: f32| EntityDesc {
            name: name.into(),
            transform: ornis_render::scene::TransformDesc {
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

    #[test]
    fn frame_runs_shared_host_and_extracts_scene() {
        let scene = two_sphere_scene();
        let mut runtime = GameRuntime::from_scene(&scene);
        assert_eq!(runtime.entity_count(), 2);
        assert_eq!(runtime.entities().len(), 2);

        let upload = runtime.frame(1.0 / 60.0);
        assert_eq!(upload.instances.len(), 2);
        assert_eq!(upload.materials.len(), 2);
        let time = runtime
            .engine()
            .world()
            .resources()
            .get::<Time>()
            .expect("engine publishes Time");
        assert_eq!(time.frame(), 1);
    }

    #[test]
    fn empty_frame_advances_clock_without_content() {
        let mut runtime = GameRuntime::new();
        let upload = runtime.frame(1.0 / 60.0);
        assert!(upload.instances.is_empty());
        assert!(upload.materials.is_empty());
        let time = runtime
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
        let mut runtime = GameRuntime::from_scene(&scene);
        let floor = spawn_static_floor(runtime.engine_mut());
        assert!(
            runtime
                .engine()
                .world()
                .store()
                .expect("store")
                .is_alive(floor)
        );
        assert_eq!(runtime.entity_count(), 2);

        let upload = runtime.frame(1.0 / 60.0);
        assert_eq!(upload.instances.len(), 2);
        assert!(upload.custom_meshes.is_empty());
    }

    #[test]
    fn replace_scene_swaps_entities() {
        let mut runtime = GameRuntime::from_scene(&two_sphere_scene());
        let empty = Scene {
            entities: Vec::new(),
            ..two_sphere_scene()
        };
        runtime.replace_scene(&empty);
        assert_eq!(runtime.entity_count(), 0);
        assert!(runtime.frame(0.0).instances.is_empty());
    }

    #[test]
    fn stages_classify_known_systems() {
        assert_eq!(
            GameStage::stage_for_system("player_input"),
            Some(GameStage::Input)
        );
        assert_eq!(
            GameStage::stage_for_system("orbit_camera_input"),
            Some(GameStage::Input)
        );
        assert_eq!(
            GameStage::stage_for_system("physics_step"),
            Some(GameStage::Gameplay)
        );
        assert_eq!(
            GameStage::stage_for_system("script_tick"),
            Some(GameStage::Gameplay)
        );
        assert_eq!(
            GameStage::stage_for_system("body_to_transform"),
            Some(GameStage::PostFrame)
        );
        assert_eq!(
            GameStage::stage_for_system("render_present"),
            Some(GameStage::PostFrame)
        );
        assert_eq!(GameStage::stage_for_system("no_such_system"), None);
        assert_eq!(GameStage::PreUpdate.name(), "pre_update");
        assert_eq!(GameStage::Input.name(), "input");
        assert_eq!(GameStage::Gameplay.name(), "gameplay");
        assert_eq!(GameStage::PostFrame.name(), "post_frame");
    }
}
