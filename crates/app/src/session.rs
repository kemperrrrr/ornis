//! Server-side ECS world for `editor-only` mode.
//!
//! In `editor-only` there is no native winit loop to consume `UiCommand`s,
//! so [`run`] spawns an `editor-world` thread that owns an [`EditorSession`]
//! (an `ornis-core::Engine` with its `World`, `SmartStore`, physics systems
//! and component registry), executes commands from `POST /api/command` and
//! publishes `GameEvent`s back to the HTTP server (`status`/`scene`
//! snapshots are cached by `remote.rs` for
//! `GET /api/status` and `GET /api/scene`; the rest reach `GET /api/events`).
//!
//! At startup the world loads `editor/scene.ron` (via
//! `ornis_assets::scene::Scene::from_ron`), so the initial live world matches
//! the scene used by the WASM viewport; subsequent changes arrive through
//! live snapshots. Component payloads reuse the
//! `ornis_render::scene` description types — **serde-canonical** JSON
//! (externally-tagged enums), served generically through the component
//! registry (F0, audit §10 D2). The JSON contract of
//! [`EditorSession::scene_json`] is:
//!
//! ```json
//! {
//!   "version": 5, "entity_count": 2,
//!   "entities": [{
//!     "id": 0, "generation": 0,
//!     "components": {
//!       "Name": "Red Sphere",
//!       "Transform": {"translation": [-5.6, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
//!       "Mesh": {"Sphere": {"radius": 1.0, "segments": 32, "rings": 24}},
//!       "Material": {"Dielectric": {"base_color": [0.8, 0.2, 0.2], "roughness": 0.5}}
//!     },
//!     "editor": {"editor_only": false, "selected": false, "hovered": false}
//!   }],
//!   "lights": [{"Directional": {"direction": [1, 1, 1], "intensity": 0.6, "color": [1, 1, 1]}}],
//!   "camera": {"position": [0, 2.5, 9], "target": [0, 0, 0], "up": [0, 1, 0], "fov": 60.0, "near": 0.1, "far": 100.0},
//!   "ambient": [0.10, 0.10, 0.15]
//! }
//! ```
//!
//! Commands (`POST /api/command`, body `{"type": …, "request_id"?: u64,
//! "data": …}`) receive a queue-level ACK; the engine emits a correlated
//! `CommandCompleted` event after execution:
//!
//! * `create_entity` — `{"name"?: string, "components"?: {"Transform": {…}, …}}`;
//!   overrides are validated **before** the spawn, an invalid payload
//!   leaves the world untouched;
//! * `destroy_entity` — `{"id": u32, "generation": u32}`;
//! * `set_component` — `{"id": u32, "generation"?: u32, "component": "Transform", "value": {…}}`;
//!   generic upsert through the registry, full replace of the component;
//! * `list_entities` — no payload;
//! * `save_scene` — `{"path"?: string}`; serializes the world to RON and
//!   writes it **atomically** (sibling `*.tmp` file + rename) to `path`
//!   (default `editor/scene.ron`, the file the WASM viewport renders),
//!   emitting `scene_saved {path, version}`. The `path` is sandboxed to
//!   `<workspace>/assets` or `<workspace>/editor` (see [`ScenePath`]);
//!   the world is not mutated;
//! * `load_scene` — `{"path"?: string}`; replaces the world with the scene
//!   read back from `path` (same sandbox), emitting `scene_loaded {path,
//!   version, entity_count}` plus fresh `status`/`scene` snapshots. A
//!   missing or malformed file emits `error` and leaves the world untouched.
//!
//! Path validation lives in two places, deliberately: the HTTP backend drops
//! escaping paths fail-fast in `build_command` (never queued), and this
//! session re-resolves every path on execution (defense in depth — direct
//! `UiCommand` senders bypass the HTTP edge). Both sides share the
//! validated [`ScenePath`] newtype, so a raw filesystem path can never
//! reach [`EditorSession::save_scene_file`]/[`load_scene_file`].
//!
//! `version` is incremented on every mutation so clients can cheaply detect
//! changes. Invalid commands never panic: they produce an `error` event and
//! leave the world (and its version) untouched.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use glam::Vec3;
use serde_json::Value;

use ornis_animation::{Animator, JointPose, SkelClip, SkelPlayer, Skeleton, SkinnedMesh};
use ornis_core::mutation::{Mutation, MutationBus, MutationPlugin, apply_mutations};
use ornis_core::units::{Clamped01, PositiveF32};
use ornis_core::{
    ComponentMeta, ComponentRegistry, Entity, InputState, SceneVersion, Seconds, SmartStore, World,
};
use ornis_editor::{EditorMarks, install_editor, is_editor_only};
use ornis_gameplay::{Position, Velocity, install_gameplay};
use ornis_physics::RigidBody;

use crate::anim_wiring::{GltfSpawn, wire_loaded_animation};
use crate::physics_runtime::{PhysicsRuntime, install_physics};
use crate::{
    GameWorld, install_gameplay_physics_bridge, install_object_animation,
    install_skeletal_animation,
};
use ornis_assets::collider::ColliderDesc;
use ornis_assets::scene::{
    CameraDesc, CameraProjection, EntityDesc, LightDesc, MaterialDesc, MeshDesc, Scene,
    TransformDesc,
};
use ornis_audio::{AudioPlugin, bridge::install_gameplay_audio_bridge};

use editor_backend::ipc::{EditorCommand, GameEvent, RequestId, UiCommand};
use editor_backend::remote::{ScenePath, ScenePathError, SceneRoots};

/// Editor-side name. The component is [`ornis_core::Name`] so glTF and
/// asset nodes share one type; its JSON form is still a plain string.
pub use ornis_core::Name;

/// Components editable through the generic protocol (F0; audit §10 D2).
/// Built once: registry ops cover `set_component`, scene snapshots and
/// `create_entity` overrides — no per-type command code in the engine.
/// Registration order is the snapshot order; treat it as protocol.
static REGISTRY: LazyLock<ComponentRegistry> = LazyLock::new(|| {
    let mut registry = ComponentRegistry::new();
    registry.register::<Name>("Name");
    registry.register::<TransformDesc>("Transform");
    registry.register::<MeshDesc>("Mesh");
    registry.register::<MaterialDesc>("Material");
    registry.register::<ColliderDesc>("Collider");
    registry.register_component::<Velocity>();
    registry.register_component::<Position>();
    registry
});

/// Idle cadence of the editor-world loop: when no command arrives within
/// this window the loop wakes anyway so the simulation and the scene-file
/// watcher advance at ~60 Hz even with no traffic.
const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(16);
/// Midpoint / half-extent scale.
const HALF: f32 = 0.5;

/// Upper bound for a measured wall-clock frame delta fed to the simulation.
///
/// Both host loops (native `render_frame`, editor-world [`run`]) measure the
/// real time between frames and clamp it here: a hitch (debugger stop,
/// backgrounded window) must cost at most ~6 fixed steps, never a
/// catch-up spiral. The bound sits inside the engine's own hitch budget
/// (`FixedTime` drops anything past 8 steps of 1/60 s), so the clamp only
/// trims what the accumulator would discard anyway.
pub const MAX_FRAME_DT: Seconds = Seconds(0.1);

/// Clamps a measured wall-clock delta into a simulation step.
///
/// Non-negative by construction (`Duration`), finite, at most
/// [`MAX_FRAME_DT`]. Shared by the native frame and the editor-world loop
/// so both hosts apply the same anti-spiral policy.
pub fn clamp_frame_dt(elapsed: Duration) -> Seconds {
    Seconds(elapsed.as_secs_f32().clamp(0.0, MAX_FRAME_DT.get()))
}

/// Outcome of one editor frame: whether observers should publish fresh
/// snapshots. Replaces the legacy `bool` on the typed [`Seconds`] path;
/// [`EditorSession::tick`] keeps returning `bool` for the deterministic
/// test pin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickOutcome {
    /// Physics, queued mutations or asset reloads changed the world (or a
    /// scene replacement did): the caller should publish `status`/`scene`.
    Changed,
    /// The frame advanced the clock only; cached snapshots stay valid.
    Unchanged,
}

impl TickOutcome {
    /// Legacy `bool` view (`true` = changed).
    pub fn changed(self) -> bool {
        matches!(self, Self::Changed)
    }
}

/// Outcome of one watched-scene reload check: whether the live world was
/// replaced. A host save that the [`FileWatch`] poll observes reports
/// [`ReloadDecision::SkippedSelfSave`] instead of rebuilding the world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReloadDecision {
    /// The file differed from the live world and replaced it (fresh
    /// snapshots were published).
    Reloaded,
    /// The observed change is the session's own save (see
    /// [`EditorSession::save_scene_file`]): the world is untouched.
    SkippedSelfSave,
    /// The file could not replace the world (missing or malformed): the
    /// live world is untouched, the cause went to stderr.
    Failed,
}

/// Scene-file I/O failure: reading, writing or (de)serializing the RON
/// scene behind `save_scene`/`load_scene` and the hot-reload watcher.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SceneFileError {
    /// The world did not serialize to RON.
    #[error("scene RON serialization: {0}")]
    Serialize(String),
    /// The sibling `*.tmp` file could not be written.
    #[error("write {path}: {reason}")]
    Write {
        /// Target scene path (not the `*.tmp` sibling).
        path: String,
        /// Underlying I/O cause.
        reason: String,
    },
    /// The atomic rename over the target failed.
    #[error("rename to {path}: {reason}")]
    Rename {
        /// Target scene path.
        path: String,
        /// Underlying I/O cause.
        reason: String,
    },
    /// The scene file could not be read back.
    #[error("read {path}: {reason}")]
    Read {
        /// Scene path that failed to read.
        path: String,
        /// Underlying I/O cause.
        reason: String,
    },
    /// The file content is not a valid scene; the world is untouched.
    #[error("{0}")]
    Parse(String),
}

/// Fingerprint of a scene file at one instant: the `(path, mtime/size)`
/// the watcher compares to tell the session's own save apart from an
/// external edit. `None` metadata means the file was missing when sampled.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SavedFingerprint {
    path: PathBuf,
    mtime: Option<SystemTime>,
    size: Option<u64>,
}

impl SavedFingerprint {
    fn capture(path: &Path) -> Self {
        let meta = fs::metadata(path).ok();
        Self {
            path: path.into(),
            mtime: meta.as_ref().and_then(|m| m.modified().ok()),
            size: meta.map(|m| m.len()),
        }
    }
}

/// World resource: lighting, camera and ambient light of the scene.
#[derive(Debug, Clone)]
pub struct SceneEnvironment {
    /// All scene lights (currently directional only).
    pub lights: Vec<LightDesc>,
    /// The single viewing camera.
    pub camera: CameraDesc,
    /// Ambient light RGB multiplier.
    pub ambient: [f32; 3],
}

impl Default for SceneEnvironment {
    fn default() -> Self {
        Self {
            lights: Vec::new(),
            camera: CameraDesc {
                position: glam::Vec3::new(0.0, 2.5, 9.0),
                target: glam::Vec3::ZERO,
                up: ornis_core::units::UnitVec3::Y,
                fov: ornis_core::units::Degrees::new(60.0),
                near: ornis_core::units::Meters::new(0.1),
                far: ornis_core::units::Meters::new(100.0),
                projection: CameraProjection::Perspective,
            },
            ambient: [0.10, 0.10, 0.15],
        }
    }
}

/// Live renderable scene: the single [`GameWorld`] plus editor-side
/// bookkeeping (protocol entity list, environment resource, SI physics
/// bindings, the scene label and a mutation version counter).
///
/// `world` owns the authoritative engine and its frame host;
/// `alive` is the protocol entity list (every spawn/despawn goes through
/// it, so editor-only components such as names and colliders stay in one
/// place). The world's own scene-entity list is only populated by the
/// `replace_scene` path, which the session never uses — loads go through
/// [`EditorSession::load_scene`] to attach names and physics bodies.
pub struct EditorSession {
    world: GameWorld,
    alive: Vec<Entity>,
    /// Scene label round-tripped through `Scene::name` on save/load.
    scene_name: String,
    /// Monotonic scene-mutation counter: bumped by every entity or scene
    /// mutation, never by frame execution alone. Serializes as the plain
    /// underlying number, so `/api/status` and `/api/scene` payloads keep
    /// their `"version": <n>` shape.
    version: SceneVersion,
    /// Fingerprint of the last host-initiated [`EditorSession::save_scene_file`]
    /// write, sample by sample. The hot-reload gate compares the watched file
    /// against it and swallows exactly the save's own mtime bump instead of
    /// rebuilding the world (fresh physics/audio) from the bytes just written.
    last_saved: Option<SavedFingerprint>,
    /// Scene-file sandbox roots for `save_scene`/`load_scene`: production
    /// code uses the workspace defaults, tests inject a temp dir.
    scene_roots: SceneRoots,
}

/// Earth-surface gravity magnitude along −Y (m/s²).
const DEFAULT_GRAVITY_Y: f32 = -9.81;
/// Default procedural sphere sector count for editor placeholders.
const DEFAULT_SPHERE_SEGMENTS: u32 = 32;
/// Default procedural sphere stack count for editor placeholders.
const DEFAULT_SPHERE_RINGS: u32 = 24;

impl Default for EditorSession {
    fn default() -> Self {
        let mut world = GameWorld::new();
        let engine = world.engine_mut();
        let _ = engine.world_mut().insert(SceneEnvironment::default());
        install_physics(engine, Vec3::new(0.0, DEFAULT_GRAVITY_Y, 0.0));
        install_gameplay(engine);
        install_gameplay_physics_bridge(engine);
        // Editor viewport domain (PLAN §i, E0): marker lanes + skeleton
        // systems on the same authoritative world. The single install point
        // for every construction route (`new`/`Default` delegate here, and
        // `load_scene` rebuilds through `new`), so editor entities always
        // have their lanes and filters.
        install_editor(engine);
        // Audio mirrors the showcase runtime: real output when a device
        // exists, silent otherwise; the bridge is a no-op without a host.
        if let Some(audio) = AudioPlugin::try_default() {
            audio.install(engine);
        }
        install_gameplay_audio_bridge(engine);
        // The mutation bus is the single write protocol for world
        // content: compute producers (scripting languages, tools)
        // register between frames and drain on `tick`; the built-in
        // editor below applies the same `Mutation` values synchronously.
        // An empty bus is one atomic counter plus an empty loop per frame.
        MutationPlugin::new().install(engine);
        // Object animation in the same DAG (after body poses): no-op
        // until an entity carries animation lanes.
        install_object_animation(engine);
        // Skeletal sampling + CPU skinning, same DAG position contract
        // (`anim_sample → skel_sample → skel_skin_cpu`): no-op until a
        // glTF load wires skeletal lanes.
        install_skeletal_animation(engine);
        // Skeletal lanes ride the same store (samplers read them when
        // present): registered up front like the hot lanes, so glTF wiring
        // only inserts. `AnimPlayer`/`AnimClip` are already covered by
        // `install_object_animation`.
        if let Some(store) = engine.world_mut().store_mut() {
            ensure_anim_lanes(store);
        }
        Self {
            world,
            alive: Vec::new(),
            scene_name: "scene".into(),
            version: SceneVersion::ZERO,
            last_saved: None,
            scene_roots: SceneRoots::workspace_defaults(),
        }
    }
}

impl EditorSession {
    /// An empty world with the default environment (default camera,
    /// no lights).
    pub fn new() -> Self {
        Self::default()
    }

    /// Test seam: a session resolving scene paths against `roots` instead
    /// of the workspace defaults, so tests never touch the real
    /// `<workspace>/editor` or `<workspace>/assets` trees.
    #[cfg(test)]
    fn with_roots(roots: SceneRoots) -> Self {
        Self {
            scene_roots: roots,
            ..Self::new()
        }
    }

    /// Returns the shared logical world backing the editor facade.
    ///
    /// The editor keeps protocol metadata (`alive`, scene name and version)
    /// beside the world, but ECS components and singleton domain resources
    /// live in this single `ornis_core::World`.
    pub fn world(&self) -> &World {
        self.world.engine().world()
    }

    /// Returns the shared logical world for setup and command processing.
    pub fn world_mut(&mut self) -> &mut World {
        self.world.engine_mut().world_mut()
    }

    /// Monotonic scene-mutation counter: bumped by every entity or scene
    /// mutation, never by frame execution alone. Clients polling the
    /// `version` field of `/api/status` or `/api/scene` observe it as a
    /// plain JSON number.
    pub fn version(&self) -> SceneVersion {
        self.version
    }

    /// Advances the editor's domain schedule by one deterministic step.
    ///
    /// Legacy `f32` pin for tests: delegates to [`EditorSession::tick_secs`].
    /// Test call sites keep passing the constant `1.0 / 60.0` step so frame
    /// counts stay reproducible; production loops pass a measured
    /// [`clamp_frame_dt`] delta through `tick_secs` instead.
    pub fn tick(&mut self, delta_seconds: f32) -> bool {
        self.tick_secs(Seconds::new(delta_seconds)).changed()
    }

    /// Advances the editor's domain schedule by one frame.
    ///
    /// Asset reloads queued via `request_reload` replace the world first;
    /// physics is intentionally opt-in per component: only entities with a
    /// `RigidBody` lane entry participate. The delta is forwarded as-is to
    /// `GameWorld::frame_secs`, whose engine runs a bounded fixed
    /// accumulator (1/60 s steps, at most 8 per frame, excess hitch time
    /// dropped) followed by the once-per-frame schedule — so a measured
    /// wall-clock delta stays stable and a constant test delta stays
    /// deterministic. Returns [`TickOutcome::Changed`] when anything changed
    /// and the caller should publish a fresh scene snapshot.
    pub fn tick_secs(&mut self, delta: Seconds) -> TickOutcome {
        // Asset reloads replace the whole world (fresh engine included),
        // so they run before the frame — never on a discarded world.
        let assets_changed = self.drain_assets();
        let _ = self.world.frame_secs(delta);
        let changed = self
            .world
            .engine_mut()
            .world_mut()
            .resources_mut()
            .get_mut::<Mutex<PhysicsRuntime>>()
            .map(|runtime| {
                runtime
                    .get_mut()
                    .unwrap_or_else(|e| e.into_inner())
                    .take_changed()
            })
            .unwrap_or(false);
        let applied = self.drain_mutations();
        if changed || applied > 0 {
            self.version.bump();
        }
        if changed || applied > 0 || assets_changed {
            TickOutcome::Changed
        } else {
            TickOutcome::Unchanged
        }
    }

    /// Applies asset-server reloads queued via `request_reload`, in id
    /// order through the normal replace path ([`EditorSession::load_scene`]).
    /// Second input channel next to [`EditorSession::drain_mutations`]:
    /// file changes arrive here after the watcher re-reads them, and
    /// programmatic hosts mark dirty assets directly. Returns true when
    /// the world was replaced.
    fn drain_assets(&mut self) -> bool {
        let dirty = self.world.assets_mut().take_dirty();
        if dirty.is_empty() {
            return false;
        }
        let mut changed = false;
        for id in dirty {
            let scene = self.world.assets().get_scene(id).cloned();
            if let Some(scene) = scene {
                self.load_scene(scene);
                changed = true;
            }
        }
        changed
    }

    /// Drains producer mutations queued by `mutation_tick` and writes them
    /// into the store through [`REGISTRY`]; returns writes performed.
    /// Sequential borrows (drain, then write) keep the bus lock clear of
    /// the store borrow.
    fn drain_mutations(&mut self) -> usize {
        let pending: Vec<Mutation> = self
            .mutation_bus()
            .map(|bus| bus.drain())
            .unwrap_or_default();
        if pending.is_empty() {
            return 0;
        }
        let Some(store) = self.world.engine_mut().world_mut().store_mut() else {
            return 0;
        };
        apply_mutations(store, &REGISTRY, &pending).applied()
    }

    /// The single write protocol for world content: compute producers
    /// attach here between frames, and custom editors drive the same bus
    /// the frame tick drains.
    fn mutation_bus(&self) -> Option<&MutationBus> {
        self.world.engine().world().resources().get::<MutationBus>()
    }

    fn store(&self) -> Option<&SmartStore> {
        self.world.engine().world().store()
    }

    fn store_mut(&mut self) -> Option<&mut SmartStore> {
        self.world.engine_mut().world_mut().store_mut()
    }

    fn environment(&self) -> Option<&SceneEnvironment> {
        self.world
            .engine()
            .world()
            .resources()
            .get::<SceneEnvironment>()
    }

    fn environment_mut(&mut self) -> Option<&mut SceneEnvironment> {
        self.world
            .engine_mut()
            .world_mut()
            .resources_mut()
            .get_mut::<SceneEnvironment>()
    }

    /// Number of currently alive entities.
    pub fn entity_count(&self) -> usize {
        self.alive.len()
    }

    /// Spawn with default components: gray dielectric sphere (r=1) at origin.
    pub fn spawn(&mut self, name: Option<String>) -> Entity {
        self.spawn_with(
            name,
            default_transform(),
            default_mesh(),
            default_material(),
        )
    }

    /// Spawn an entity with explicit components and an optional name
    /// (defaults to "Entity <id>"); bumps the version counter.
    pub fn spawn_with(
        &mut self,
        name: Option<String>,
        transform: TransformDesc,
        mesh: MeshDesc,
        material: MaterialDesc,
    ) -> Entity {
        let physics_body = ornis_physics::colliders::body_for(&transform, &mesh, None, 0.0);
        let Some(store) = self.store() else {
            return Entity::new(0);
        };
        let entity = store.create_entity();
        self.alive.push(entity);
        let name = name.unwrap_or_else(|| format!("Entity {}", entity.id()));
        let Some(store) = self.store_mut() else {
            return entity;
        };
        store.insert(entity, Name(name));
        store.insert(entity, transform.clone());
        crate::insert_flat_pose(store, entity, &transform);
        store.insert(entity, mesh);
        store.insert(entity, material);
        // A broken collider (`Err`) spawns without a body, like the
        // no-recipe (`Ok(None)`) case — transport never invents colliders.
        if let Ok(Some(body)) = physics_body {
            store.insert(entity, body);
        }
        self.version.bump();
        entity
    }

    /// Despawn by id/generation. Returns the entity if it was alive.
    pub fn despawn(&mut self, id: u32, generation: u32) -> Option<Entity> {
        let entity = Entity::new_with_gen(id, generation);
        {
            let store = self.store()?;
            if !store.is_alive(entity) {
                return None;
            }
            ornis_core::despawn_recursive(store, entity);
        }
        let still_alive: Vec<Entity> = {
            let store = self.store();
            self.alive
                .iter()
                .copied()
                .filter(|entity| store.is_some_and(|store| store.is_alive(*entity)))
                .collect()
        };
        self.alive = still_alive;
        self.version.bump();
        Some(entity)
    }

    /// Display name of `entity`, if alive and named.
    pub fn name_of(&self, entity: Entity) -> Option<String> {
        self.store()
            .and_then(|store| store.read_lane::<Name>())
            .and_then(|lane| lane.get(entity).map(|name| name.0.clone()))
    }

    /// Snapshot the world as a [`Scene`]: every alive entity becomes an
    /// [`EntityDesc`] (missing components fall back to the spawn defaults),
    /// lights/camera/ambient come from the environment resource.
    pub fn to_scene(&self) -> Scene {
        to_scene(self)
    }

    /// Replace the world with `scene`: each `EntityDesc` becomes an entity,
    /// lights/camera/ambient replace the environment resource. The version
    /// stays monotonic (`max(loaded entity count, old version + 1)`) so
    /// clients polling `version` always observe the replacement.
    /// Returns the number of entities loaded.
    pub fn load_scene(&mut self, scene: Scene) -> usize {
        let count = scene.entities.len();
        let mut fresh = EditorSession::new();
        for e in scene.entities {
            fresh.spawn_with(Some(e.name), e.transform, e.mesh, e.material);
        }
        if let Some(env) = fresh.environment_mut() {
            *env = SceneEnvironment {
                lights: scene.lights,
                camera: scene.camera,
                ambient: scene.ambient,
            };
        }
        fresh.scene_name = scene.name;
        fresh.version = fresh.version.max(self.version.bumped());
        // The asset registry (load history, retained sources) lives on the
        // game world and survives the replacement — it describes files, not
        // the live entities. The pending self-save fingerprint survives with
        // it: the file on disk is still the bytes the session wrote, so the
        // next watcher poll must keep swallowing the save's own mtime bump
        // instead of reloading over the freshly loaded world.
        *fresh.world.assets_mut() = std::mem::take(self.world.assets_mut());
        fresh.last_saved = self.last_saved.take();
        fresh.scene_roots = self.scene_roots.clone();
        *self = fresh;
        count
    }

    /// Parse a RON scene and load it (replacing the world, see
    /// [`EditorSession::load_scene`]). An invalid RON string leaves the world
    /// untouched. Parsing goes through the asset server so the source is
    /// retained for round-trips.
    ///
    /// # Errors
    ///
    /// [`SceneLoadError`](ornis_assets::SceneLoadError) when the text is
    /// not a valid scene; the world is untouched.
    pub fn load_scene_ron(&mut self, ron_str: &str) -> Result<usize, ornis_assets::SceneLoadError> {
        let id = self.world.assets_mut().load_scene_ron(ron_str)?;
        let Some(scene) = self.world.assets().get_scene(id).cloned() else {
            return Ok(0);
        };
        Ok(self.load_scene(scene))
    }

    /// Serialize the world to RON and write it to `path` **atomically**
    /// (sibling `*.tmp` file + rename): a crash or I/O error mid-write can
    /// never leave a truncated scene file behind. The world is not mutated.
    ///
    /// On success the file's `(mtime, size)` fingerprint is remembered in
    /// `last_saved`: the next hot-reload poll compares the watched file
    /// against it and swallows the save's own mtime bump instead of
    /// rebuilding the world from the bytes just written. Invariant: only an
    /// actually different file (external edit) may trigger a reload — a
    /// matching fingerprint is consumed one-shot by the reload gate (see
    /// `reload_watched_scene`), so a later external edit still reloads.
    ///
    /// # Errors
    ///
    /// [`SceneFileError`] when the world does not serialize or the atomic
    /// write fails; the previous scene file (if any) stays intact.
    ///
    /// The `path` is a validated [`ScenePath`]: only files inside
    /// `<workspace>/assets` or `<workspace>/editor` can be named — the type,
    /// not a runtime check at the call site, enforces the sandbox (the HTTP
    /// backend and [`EditorSession::scene_path`] resolve user input into it).
    pub fn save_scene_file(&mut self, path: &ScenePath) -> Result<(), SceneFileError> {
        let resolved = path.as_path();
        let ron = self
            .to_scene()
            .to_ron()
            .map_err(|e| SceneFileError::Serialize(e.to_string()))?;
        atomic_write(resolved, &ron)?;
        self.last_saved = Some(SavedFingerprint::capture(resolved));
        Ok(())
    }

    /// Read `path` and replace the world with its scene. `.ron` files go
    /// through the asset server (retained as sources); `.glb`/`.gltf`
    /// files load geometry + scalar materials through the same replace
    /// path, then wire skeletal/object animation clips (named skeletal
    /// playback, see [`EditorSession::wire_gltf_animation`]).
    /// Any error (missing file, invalid content) leaves the world untouched.
    ///
    /// # Errors
    ///
    /// [`SceneFileError`] when the file cannot be read or parsed; the
    /// world is untouched.
    ///
    /// Like [`EditorSession::save_scene_file`], the `path` is a validated
    /// [`ScenePath`] confined to `<workspace>/assets` or `<workspace>/editor`.
    pub fn load_scene_file(&mut self, path: &ScenePath) -> Result<usize, SceneFileError> {
        let resolved = path.as_path();
        let is_gltf = resolved
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("glb") || ext.eq_ignore_ascii_case("gltf"));
        if is_gltf {
            let id = self
                .world
                .assets_mut()
                .load_gltf_file(resolved)
                .map_err(|e| SceneFileError::Parse(e.to_string()))?;
            let Some(model) = self.world.assets().model(id).cloned() else {
                return Ok(0);
            };
            // The editor instantiates a flat scene (world TRS per primitive).
            // The retained model keeps the node tree for animation wiring.
            let scene = ornis_assets::scene_from_model(&model);
            let count = self.load_scene(scene);
            // `load_scene` moves the asset server onto the fresh world. The
            // cloned model is the node tree animation wiring reads.
            let added = self.wire_gltf_animation(&model);
            return Ok(count + added);
        }
        let ron = fs::read_to_string(resolved).map_err(|e| SceneFileError::Read {
            path: resolved.display().to_string(),
            reason: e.to_string(),
        })?;
        self.load_scene_ron(&ron)
            .map_err(|e| SceneFileError::Parse(e.to_string()))
    }

    /// Wires animation from a retained glTF [`Model`](ornis_gltf::Model) into
    /// the live world (thin wrapper over
    /// [`wire_loaded_animation`](crate::anim_wiring::wire_loaded_animation)).
    ///
    /// Mesh entities are already spawned in primitive order (see
    /// [`scene_from_model`](ornis_assets::scene_from_model)), so
    /// `loaded.primitives` zips with the pre-wire [`EditorSession::alive`]
    /// prefix. The first primitive of a node wins in the node map.
    /// Skeletal clips land on an [`Animator`] on the first mesh
    /// entity (stand-in until a scene root is supplied). No skeletal
    /// cursor exists until `Animator::play`. Object players are inserted
    /// paused on the entities their tracks name. Created roots and
    /// playlists join `alive`.
    ///
    /// Returns the number of spawned animation entities (roots + playlists).
    fn wire_gltf_animation(&mut self, loaded: &ornis_gltf::Model) -> usize {
        if loaded.skins.is_empty() && loaded.skel_clips.is_empty() && loaded.anim_clips.is_empty() {
            return 0;
        }
        // Mesh-entity snapshot: `alive` grows below (roots + playlists), so
        // the primitive-order zip must pin the pre-wire prefix.
        let mesh_entities: Vec<Entity> = self.alive.clone();
        // First mesh entity of a node wins (flat spawn has no mesh-less nodes).
        let mut node_to_entity: HashMap<ornis_gltf::NodeIdx, Entity> = HashMap::new();
        for (primitive, entity) in loaded.primitives.iter().zip(mesh_entities.iter()) {
            node_to_entity.entry(primitive.node).or_insert(*entity);
        }
        let spawn = GltfSpawn {
            entities: mesh_entities,
            node_to_entity,
            scene_root: None,
        };
        let Some(store) = self.store_mut() else {
            return 0;
        };
        let wiring = wire_loaded_animation(store, loaded, &spawn);
        let added = wiring.added();
        for entity in wiring
            .roots
            .iter()
            .chain(wiring.skel_playlists.iter())
            .chain(wiring.anim_playlists.iter())
        {
            self.alive.push(*entity);
        }
        if added > 0 {
            self.version.bump();
        }
        added
    }

    /// Resolve a user-supplied scene `path` against this session's sandbox
    /// roots: the defense-in-depth check behind `save_scene`/`load_scene`
    /// (the HTTP backend already drops escaping paths fail-fast in
    /// `build_command`). `..` and absolute paths escaping
    /// `<workspace>/assets` or `<workspace>/editor` are a typed
    /// [`ScenePathError`], never a panic.
    ///
    /// Test-only shorthand: production paths flow through [`command_path`].
    #[cfg(test)]
    fn scene_path(&self, raw: &str) -> Result<ScenePath, ScenePathError> {
        ScenePath::resolve(&self.scene_roots, raw)
    }

    /// JSON snapshot for `GET /api/scene` (see the module docs for the contract).
    pub fn scene_json(&self) -> String {
        scene_json(self)
    }

    /// JSON payload for `GET /api/status` (cached by the HTTP server).
    pub fn status_json(&self) -> String {
        status_json(self)
    }

    /// Execute one command, emitting state and completion events through `ev_tx`.
    pub fn handle_command(&mut self, cmd: &UiCommand, ev_tx: &Sender<GameEvent>) {
        handle_command(self, cmd, ev_tx);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Snapshots and command handling — free functions over [`EditorSession`]
// ═══════════════════════════════════════════════════════════════════════════
// The bulk of the snapshot/command logic lives here, not in
// `impl EditorSession`, to keep the type's method count within the structural
// gate's thresholds; the public methods above are thin delegates.

/// Snapshot `world` as a [`Scene`]: every alive entity becomes an
/// [`EntityDesc`] (missing components fall back to the spawn defaults),
/// lights/camera/ambient come from the environment resource.
///
/// Editor chrome ([`EditorOnly`](ornis_editor::EditorOnly)) is excluded:
/// gizmos and selection proxies replicate through [`scene_json`] but must
/// never persist into the scene file. The check is the shared
/// [`is_editor_only`](ornis_editor::is_editor_only) predicate — the same
/// one physics sync-in uses.
fn to_scene(world: &EditorSession) -> Scene {
    let entities = world
        .alive
        .iter()
        .filter(|entity| {
            !world
                .store()
                .is_some_and(|store| is_editor_only(store, **entity))
        })
        .map(|&e| entity_desc(world, e))
        .collect();
    let env = world.environment().cloned().unwrap_or_default();
    Scene {
        name: world.scene_name.clone(),
        entities,
        lights: env.lights,
        camera: env.camera,
        ambient: env.ambient,
    }
}

/// One alive entity as an [`EntityDesc`] for [`to_scene`].
fn entity_desc(world: &EditorSession, entity: Entity) -> EntityDesc {
    EntityDesc {
        name: world
            .name_of(entity)
            .unwrap_or_else(|| format!("Entity {}", entity.id())),
        transform: world
            .store()
            .and_then(|store| read_component(store, entity))
            .unwrap_or_else(default_transform),
        mesh: world
            .store()
            .and_then(|store| read_component(store, entity))
            .unwrap_or_else(default_mesh),
        material: world
            .store()
            .and_then(|store| read_component(store, entity))
            .unwrap_or_else(default_material),
    }
}

/// JSON snapshot for `GET /api/scene` (see the module docs for the contract).
fn scene_json(world: &EditorSession) -> String {
    let entities: Vec<Value> = world
        .alive
        .iter()
        .filter_map(|&e| world.store().map(|store| entity_json(store, e)))
        .collect();
    let env = world.environment().cloned().unwrap_or_default();
    let lights = serde_json::to_value(&env.lights).unwrap_or(Value::Null);
    let camera = serde_json::to_value(&env.camera).unwrap_or(Value::Null);
    serde_json::json!({
        "version": world.version(),
        "entity_count": world.entity_count(),
        "entities": entities,
        "lights": lights,
        "camera": camera,
        "ambient": env.ambient,
    })
    .to_string()
}

/// JSON payload for `GET /api/status` (cached by the HTTP server).
fn status_json(world: &EditorSession) -> String {
    serde_json::json!({
        "entity_count": world.entity_count(),
        "name": "Ornis Engine",
        "version": world.version(),
    })
    .to_string()
}

/// Publish `status` + `scene` snapshots so the HTTP server's caches
/// (`GET /api/status`, `GET /api/scene`) reflect the current world.
fn publish_state(world: &EditorSession, ev_tx: &Sender<GameEvent>) {
    emit(ev_tx, "status", world.status_json());
    emit(ev_tx, "scene", world.scene_json());
}

fn emit(ev_tx: &Sender<GameEvent>, cmd_type: &str, payload: String) {
    ev_tx
        .send(GameEvent::CustomEvent {
            cmd_type: cmd_type.into(),
            json_data: payload,
        })
        .ok();
}

/// Invalid commands become `error` events instead of panics.
fn emit_error(ev_tx: &Sender<GameEvent>, command: &str, message: &str) {
    emit(
        ev_tx,
        "error",
        serde_json::json!({"command": command, "message": message}).to_string(),
    );
}

/// Result of executing a command on the authoritative editor world.
struct CommandOutcome {
    success: bool,
    error: Option<String>,
}

impl CommandOutcome {
    fn success() -> Self {
        Self {
            success: true,
            error: None,
        }
    }

    fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            error: Some(error.into()),
        }
    }
}

/// Execute one command from the HTTP server, emitting the corresponding
/// events (`entity_created`/`entity_destroyed`/`entity_list`,
/// `ComponentUpdated`, `status`/`scene` snapshots or `error`). A transport
/// wrapped command additionally receives a correlated completion event.
fn handle_command(world: &mut EditorSession, cmd: &UiCommand, ev_tx: &Sender<GameEvent>) {
    // Browser input channel (WS + POST /api/input): replace authoritative
    // InputState in the unified World. No polling / scene.ron fallback.
    // Handles both bare Input and wrapped WithRequestId(Input).
    if let Some(input) = match cmd {
        UiCommand::Input { input } => Some(input),
        UiCommand::WithRequestId { command, .. } => match &**command {
            UiCommand::Input { input } => Some(input),
            _ => None,
        },
        _ => None,
    } {
        apply_browser_input(world, input);
        return;
    }
    if let UiCommand::WithRequestId {
        request_id,
        command,
    } = cmd
    {
        let outcome = execute_command(world, command, ev_tx);
        emit_command_completed(ev_tx, *request_id, command_name(command), outcome);
    } else {
        let _ = execute_command(world, cmd, ev_tx);
    }
}

fn apply_browser_input(world: &mut EditorSession, input: &editor_backend::ipc::BrowserInput) {
    let world_mut = world.world.engine_mut().world_mut();
    let state = world_mut.resources_mut().get_mut::<InputState>();
    let state = if let Some(s) = state {
        s
    } else {
        world_mut.resources_mut().insert(InputState::default());
        let Some(s) = world_mut.resources_mut().get_mut::<InputState>() else {
            return;
        };
        s
    };
    state.apply_snapshot(
        &input.pressed_keys,
        &input.pressed_mouse_buttons,
        input.pointer_position,
        input.pointer_delta,
        input.wheel_delta,
    );
}

fn execute_command(
    world: &mut EditorSession,
    cmd: &UiCommand,
    ev_tx: &Sender<GameEvent>,
) -> CommandOutcome {
    match cmd {
        UiCommand::CreateEntity => {
            world.spawn(None);
            publish_state(world, ev_tx);
            CommandOutcome::success()
        }
        UiCommand::DestroyEntity { entity_id } => {
            // The typed variant carries no generation; match any alive
            // entity with this id.
            if let Ok(entity) = resolve_alive(&world.alive, *entity_id, None) {
                world.despawn(entity.id(), entity.generation());
                publish_state(world, ev_tx);
                CommandOutcome::success()
            } else {
                CommandOutcome::failure(format!("entity {entity_id} not found"))
            }
        }
        UiCommand::Custom {
            cmd_type,
            json_data,
        } => handle_custom(world, cmd_type, json_data, ev_tx),
        UiCommand::SetComponent {
            entity_id,
            generation,
            type_name,
            json_data,
        } => handle_set_component(world, *entity_id, *generation, type_name, json_data, ev_tx),
        UiCommand::WithRequestId { .. } => {
            CommandOutcome::failure("nested request-id command is not supported")
        }
        UiCommand::Input { .. } => {
            // Already handled in handle_command; direct calls are no-ops.
            CommandOutcome::success()
        }
    }
}

fn command_name(cmd: &UiCommand) -> EditorCommand {
    match cmd {
        UiCommand::CreateEntity => EditorCommand::CreateEntity,
        UiCommand::DestroyEntity { .. } => EditorCommand::DestroyEntity,
        UiCommand::SetComponent { .. } => EditorCommand::SetComponent,
        UiCommand::Custom { cmd_type, .. } => cmd_type.clone(),
        UiCommand::Input { .. } => EditorCommand::Input,
        UiCommand::WithRequestId { command, .. } => command_name(command),
    }
}

fn emit_command_completed(
    ev_tx: &Sender<GameEvent>,
    request_id: RequestId,
    command: EditorCommand,
    outcome: CommandOutcome,
) {
    ev_tx
        .send(GameEvent::CommandCompleted {
            request_id,
            command,
            success: outcome.success,
            error: outcome.error,
        })
        .ok();
}

/// Typed `SetComponent` (remote maps the `set_component` POST here):
/// generic upsert through the component registry. Success emits the
/// typed `ComponentUpdated` event and publishes fresh snapshots; any
/// error (unknown entity/component, malformed JSON) is an `error`
/// event with the world left untouched.
fn handle_set_component(
    world: &mut EditorSession,
    entity_id: u32,
    generation: Option<u32>,
    type_name: &editor_backend::ipc::ComponentName,
    json_data: &str,
    ev_tx: &Sender<GameEvent>,
) -> CommandOutcome {
    match set_component(world, entity_id, generation, type_name.as_str(), json_data) {
        Ok(value) => {
            ev_tx
                .send(GameEvent::ComponentUpdated {
                    entity_id,
                    type_name: type_name.clone(),
                    json_data: value.to_string(),
                })
                .ok();
            publish_state(world, ev_tx);
            CommandOutcome::success()
        }
        Err(e) => {
            emit_error(ev_tx, "set_component", &e);
            CommandOutcome::failure(e)
        }
    }
}

/// Validate and apply the upsert through the shared [`Mutation`] path;
/// returns the applied payload. Protocol errors (unknown entity /
/// component, malformed JSON) keep the exact legacy messages; the write
/// itself goes through [`apply_mutations`] like every other producer.
fn set_component(
    world: &mut EditorSession,
    entity_id: u32,
    generation: Option<u32>,
    type_name: &str,
    json_data: &str,
) -> Result<Value, String> {
    let entity = resolve_alive(&world.alive, entity_id, generation)?;
    if REGISTRY.by_name(type_name).is_none() {
        return Err(format!("unknown component '{type_name}'"));
    }
    let value: Value = serde_json::from_str(json_data).map_err(|e| format!("invalid JSON: {e}"))?;
    let Some(store) = world.store_mut() else {
        return Err("SmartStore not registered".into());
    };
    let report = apply_mutations(
        store,
        &REGISTRY,
        &[Mutation::Set {
            entity,
            component: ornis_core::ComponentName::from(type_name),
            value: value.clone(),
        }],
    );
    if let Some(entry) = report.entries.first()
        && !entry.errors.is_empty()
    {
        return Err(entry.error_messages().join("; "));
    }
    if type_name == "Collider" {
        // An explicit collider redefines collision: rebuild the lane body
        // from the current lanes (velocity state is preserved).
        if let Some(store) = world.store_mut() {
            resync_collider_body(store, entity);
        }
    }
    world.version.bump();
    Ok(value)
}

fn handle_custom(
    world: &mut EditorSession,
    cmd_type: &EditorCommand,
    json_data: &str,
    ev_tx: &Sender<GameEvent>,
) -> CommandOutcome {
    let data = match parse_data(json_data) {
        Ok(data) => data,
        Err(e) => {
            emit_error(ev_tx, cmd_type.as_str(), &e);
            return CommandOutcome::failure(e);
        }
    };
    match cmd_type.as_str() {
        "create_entity" => match cmd_create_entity(world, &data) {
            Ok(payload) => {
                emit(ev_tx, "entity_created", payload);
                publish_state(world, ev_tx);
                CommandOutcome::success()
            }
            Err(e) => {
                emit_error(ev_tx, cmd_type.as_str(), &e);
                CommandOutcome::failure(e)
            }
        },
        "destroy_entity" => match cmd_destroy_entity(world, &data) {
            Ok(payload) => {
                emit(ev_tx, "entity_destroyed", payload);
                publish_state(world, ev_tx);
                CommandOutcome::success()
            }
            Err(e) => {
                emit_error(ev_tx, cmd_type.as_str(), &e);
                CommandOutcome::failure(e)
            }
        },
        "list_entities" => {
            let payload = list_entities_json(world);
            emit(ev_tx, "entity_list", payload);
            CommandOutcome::success()
        }
        "save_scene" => {
            let path = match command_path(&world.scene_roots, &data) {
                Ok(path) => path,
                Err(e) => {
                    let message = e.to_string();
                    emit_error(ev_tx, cmd_type.as_str(), &message);
                    return CommandOutcome::failure(message);
                }
            };
            match world.save_scene_file(&path) {
                Ok(()) => {
                    emit(
                        ev_tx,
                        "scene_saved",
                        serde_json::json!({
                            "path": path.as_path().display().to_string(),
                            "version": world.version,
                        })
                        .to_string(),
                    );
                    CommandOutcome::success()
                }
                Err(e) => {
                    emit_error(ev_tx, cmd_type.as_str(), &e.to_string());
                    CommandOutcome::failure(e.to_string())
                }
            }
        }
        "load_scene" => {
            let path = match command_path(&world.scene_roots, &data) {
                Ok(path) => path,
                Err(e) => {
                    let message = e.to_string();
                    emit_error(ev_tx, cmd_type.as_str(), &message);
                    return CommandOutcome::failure(message);
                }
            };
            match world.load_scene_file(&path) {
                Ok(count) => {
                    emit(
                        ev_tx,
                        "scene_loaded",
                        serde_json::json!({
                            "path": path.as_path().display().to_string(),
                            "version": world.version,
                            "entity_count": count,
                        })
                        .to_string(),
                    );
                    publish_state(world, ev_tx);
                    CommandOutcome::success()
                }
                Err(e) => {
                    emit_error(ev_tx, cmd_type.as_str(), &e.to_string());
                    CommandOutcome::failure(e.to_string())
                }
            }
        }
        other => {
            emit_error(ev_tx, other, "unknown command");
            CommandOutcome::failure("unknown command")
        }
    }
}

fn cmd_create_entity(world: &mut EditorSession, data: &Value) -> Result<String, String> {
    let name = opt_string(data, "name")?;
    // Optional component overrides by registry name. Everything is
    // parsed BEFORE the spawn so a bad payload leaves the world (and
    // its version) untouched.
    let overrides = match data.get("components") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Object(map)) => parse_overrides(map)?,
        Some(_) => return Err("'components': expected an object".into()),
    };
    let entity = world.spawn(name);
    {
        let Some(store) = world.store_mut() else {
            return Err("SmartStore not registered".into());
        };
        for (meta, boxed) in overrides {
            // Parsed from the same meta — the box type always matches.
            meta.insert_any(store, entity, boxed);
        }
        // Overrides may have replaced the mesh or set an explicit collider:
        // rebuild the body from the final lanes (velocity is fresh here).
        resync_collider_body(store, entity);
    }
    Ok(serde_json::json!({
        "id": entity.id(),
        "generation": entity.generation(),
        "name": world.name_of(entity),
    })
    .to_string())
}

fn cmd_destroy_entity(world: &mut EditorSession, data: &Value) -> Result<String, String> {
    let entity = resolve_entity(world, data)?;
    world.despawn(entity.id(), entity.generation());
    Ok(serde_json::json!({"id": entity.id(), "generation": entity.generation()}).to_string())
}

/// Validate `id` + `generation` against the store's allocator.
fn resolve_entity(world: &EditorSession, data: &Value) -> Result<Entity, String> {
    let id = data
        .get("id")
        .and_then(Value::as_u64)
        .ok_or("missing or invalid 'id'")? as u32;
    let generation = data
        .get("generation")
        .and_then(Value::as_u64)
        .ok_or("missing or invalid 'generation'")? as u32;
    let entity = Entity::new_with_gen(id, generation);
    let alive = world.store().is_some_and(|store| store.is_alive(entity));
    if !alive {
        return Err(format!("entity {id}:{generation} not found"));
    }
    Ok(entity)
}

/// One entity entry: `id`/`generation` plus a map
/// «registry name → serde-canonical component JSON» — generic over the
/// registered component set (registry ops, no per-type code) — plus the
/// `editor` marks object ([`EditorMarks`]: `{"editor_only": …,
/// `"selected": …, `"hovered": …}`, always present). Editor chrome
/// ([`EditorOnly`](ornis_editor::EditorOnly)) is listed here like any
/// entity — the replica draws gizmos and selection from these labels —
/// while [`to_scene`] filters it out of the saved scene.
fn entity_json(store: &SmartStore, entity: Entity) -> Value {
    let mut components = serde_json::Map::new();
    for meta in REGISTRY.iter() {
        // A serialization error is unreachable for plain data structs.
        if let Ok(Some(value)) = meta.get_json(store, entity) {
            components.insert(meta.name().to_string(), value);
        }
    }
    let editor = serde_json::to_value(EditorMarks::of(store, entity)).unwrap_or(Value::Null);
    serde_json::json!({
        "id": entity.id(),
        "generation": entity.generation(),
        "components": components,
        "editor": editor,
    })
}

/// Read a typed component of `entity` from the store.
fn read_component<T: 'static + Clone + Send + Sync>(
    store: &SmartStore,
    entity: Entity,
) -> Option<T> {
    store
        .read_lane::<T>()
        .and_then(|lane| lane.get(entity).cloned())
}

/// Typed commands resolve an entity by id among the alive ones;
/// a supplied generation must match (id-only matches any generation).
fn resolve_alive(alive: &[Entity], id: u32, generation: Option<u32>) -> Result<Entity, String> {
    alive
        .iter()
        .find(|e| e.id() == id && generation.is_none_or(|g| e.generation() == g))
        .copied()
        .ok_or_else(|| format!("entity {id} not found"))
}

/// `list_entities` payload: entity count plus `{id, generation, name}` rows.
fn list_entities_json(world: &EditorSession) -> String {
    let entities: Vec<Value> = world
        .alive
        .iter()
        .map(|&e| {
            serde_json::json!({
                "id": e.id(),
                "generation": e.generation(),
                "name": world.name_of(e),
            })
        })
        .collect();
    serde_json::json!({"count": entities.len(), "entities": entities}).to_string()
}

// ═══════════════════════════════════════════════════════════════════════════
// Component defaults / JSON (de)serialization helpers
// ═══════════════════════════════════════════════════════════════════════════

/// (Re)builds the entity's solver body from its current lanes through
/// the shared projection ([`ornis_physics::colliders::body_for`]):
/// an explicit [`ColliderDesc`] lane entry wins, else the exact auto
/// recipe applies. The lane body's velocity state survives the rebuild;
/// when nothing builds the lane is left as-is (there is no lane-remove
/// API — spawn paths start clean, so this only affects live edits that
/// remove collision).
fn resync_collider_body(store: &mut SmartStore, entity: Entity) {
    let Some(transform) = store
        .read_lane::<TransformDesc>()
        .and_then(|lane| lane.get(entity).cloned())
    else {
        return;
    };
    let Some(mesh) = store
        .read_lane::<MeshDesc>()
        .and_then(|lane| lane.get(entity).cloned())
    else {
        return;
    };
    let collider = store
        .read_lane::<ColliderDesc>()
        .and_then(|lane| lane.get(entity).cloned());
    let previous = store
        .read_lane::<RigidBody>()
        .and_then(|lane| lane.get(entity).cloned());
    // A broken collider (`Err`) leaves the lane as-is, like the
    // no-recipe (`Ok(None)`) case — failed edits never clear physics.
    if let Ok(Some(mut body)) =
        ornis_physics::colliders::body_for(&transform, &mesh, collider.as_ref(), 0.0)
    {
        if let Some(prev) = previous {
            body.velocity = prev.velocity;
            body.angular_velocity = prev.angular_velocity;
        }
        store.insert(entity, body);
    }
}

fn default_transform() -> TransformDesc {
    TransformDesc::IDENTITY
}

fn default_mesh() -> MeshDesc {
    MeshDesc::Sphere {
        radius: PositiveF32::expect_valid(1.0),
        segments: DEFAULT_SPHERE_SEGMENTS,
        rings: DEFAULT_SPHERE_RINGS,
    }
}

fn default_material() -> MaterialDesc {
    MaterialDesc::Dielectric {
        base_color: [HALF, HALF, HALF],
        roughness: Clamped01::new(HALF),
        emission: [0.0, 0.0, 0.0],
        metallic: ornis_core::Metallic::new(0.0),
    }
}

/// Registers the skeletal animation lanes up front (hot
/// [`Skeleton`]/[`JointPose`]/[`SkinnedMesh`]/[`SkelPlayer`]/[`Animator`],
/// cold [`SkelClip`]): mirrors the
/// [`install_object_animation`](crate::install_object_animation) pattern
/// (`register` for hot lanes, `register_cold` for cold ones), so glTF
/// wiring only inserts. `AnimPlayer`/`AnimClip` stay with
/// `install_object_animation`.
fn ensure_anim_lanes(store: &mut SmartStore) {
    store.register::<Skeleton>();
    store.register::<JointPose>();
    store.register::<SkinnedMesh>();
    store.register::<SkelPlayer>();
    store.register::<Animator>();
    store.register_cold::<SkelClip>();
}

/// Scene path override from a `save_scene`/`load_scene` payload
/// (`{"path": "…"}`); defaults to `editor/scene.ron` — the file the WASM
/// viewport renders. A non-string `path` is ignored (falls back to the
/// default) like any other soft payload flaw.
///
/// Resolution is the session-side (defense-in-depth) sandbox check: `..`
/// and absolute paths escaping `<workspace>/assets` or
/// `<workspace>/editor` are a typed [`ScenePathError`], never a panic.
/// The HTTP backend already drops escaping paths fail-fast in
/// `build_command`, so a rejection here only triggers for direct
/// `UiCommand` senders.
fn command_path(roots: &SceneRoots, data: &Value) -> Result<ScenePath, ScenePathError> {
    match data.get("path").and_then(Value::as_str) {
        Some(raw) => ScenePath::resolve(roots, raw),
        None => ScenePath::resolve(roots, "editor/scene.ron"),
    }
}

/// Parse the command payload; an empty body means `{}`.
fn parse_data(json_data: &str) -> Result<Value, String> {
    if json_data.trim().is_empty() {
        return Ok(Value::Object(Default::default()));
    }
    let v: Value = serde_json::from_str(json_data).map_err(|e| format!("invalid JSON: {e}"))?;
    if !v.is_object() {
        return Err("command data must be a JSON object".into());
    }
    Ok(v)
}

fn opt_string(data: &Value, key: &str) -> Result<Option<String>, String> {
    match data.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_string()))
            .ok_or_else(|| format!("'{key}': expected a string")),
    }
}

/// A validated `create_entity` override: registry entry + boxed component.
type ParsedOverrides = Vec<(&'static ComponentMeta, Box<dyn std::any::Any>)>;

/// Deserialize component overrides of `create_entity`, registry-keyed.
/// Everything is validated here — before the entity is spawned — so a
/// bad payload leaves the world untouched (module-doc invariant).
fn parse_overrides(map: &serde_json::Map<String, Value>) -> Result<ParsedOverrides, String> {
    let mut parsed = Vec::with_capacity(map.len());
    for (type_name, payload) in map {
        let meta = REGISTRY
            .by_name(type_name)
            .ok_or_else(|| format!("unknown component '{type_name}'"))?;
        let boxed = meta.parse_json(payload).map_err(|e| e.to_string())?;
        parsed.push((meta, boxed));
    }
    Ok(parsed)
}

// ═══════════════════════════════════════════════════════════════════════════
// Startup
// ═══════════════════════════════════════════════════════════════════════════

/// Workspace root anchor for scene files: this crate lives at
/// `crates/app`, so its manifest dir is two levels below the root (the
/// same `../../` convention as `editor-backend`'s asset root). Ancestor
/// indexing keeps the path free of `..` segments, so default
/// `scene_saved`/`scene_loaded` payloads report clean paths.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

/// Default scene file for the `save_scene`/`load_scene` commands:
/// `editor/scene.ron` — the scene the WASM viewport renders at startup.
/// Test-only: production resolution goes through the sandboxed
/// [`ScenePath`](editor_backend::remote::ScenePath).
#[cfg(test)]
fn scene_file_path() -> PathBuf {
    workspace_root().join("editor/scene.ron")
}

/// Write `contents` to `path` atomically: a sibling `<name>.tmp` file is
/// written first and renamed over the target, so a failed write leaves the
/// previous scene file intact (or no file at all).
fn atomic_write(path: &Path, contents: &str) -> Result<(), SceneFileError> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    fs::write(&tmp, contents).map_err(|e| SceneFileError::Write {
        path: path.display().to_string(),
        reason: e.to_string(),
    })?;
    fs::rename(&tmp, path).map_err(|e| SceneFileError::Rename {
        path: path.display().to_string(),
        reason: e.to_string(),
    })
}

/// Startup scene RON: `editor/scene.ron` (the initial scene for the live
/// editor/WASM snapshot), falling back to `assets/scene.ron`.
fn startup_scene_ron() -> Option<String> {
    let root = workspace_root();
    ["editor/scene.ron", "assets/scene.ron"]
        .iter()
        .find_map(|rel| fs::read_to_string(root.join(rel)).ok())
}

/// Watches one file for external edits (phase 7, minimal): the
/// editor-world loop already wakes every 16 ms, so a cheap mtime poll
/// needs no `notify` dependency. Serves the scene file (frontend clients
/// pick the reload up through the normal versioned `/api/scene`
/// snapshots).
struct FileWatch {
    path: PathBuf,
    last_mtime: Option<SystemTime>,
}

impl FileWatch {
    /// Starts watching `path`, baselining the current mtime so the
    /// already-loaded content does not trigger an instant reload.
    fn new(path: PathBuf) -> Self {
        Self {
            last_mtime: fs::metadata(&path).and_then(|m| m.modified()).ok(),
            path,
        }
    }

    /// Returns `true` once when the file changed since the last poll: a
    /// newer mtime, or a file that appeared after being missing.
    fn poll(&mut self) -> bool {
        let current = fs::metadata(&self.path).and_then(|m| m.modified()).ok();
        if current != self.last_mtime {
            self.last_mtime = current;
            return current.is_some();
        }
        false
    }
}

/// Resolves the scene file the world loads from: `editor/scene.ron`
/// preferred, `assets/scene.ron` fallback — mirrors `startup_scene_ron`.
fn watched_scene_path() -> Option<PathBuf> {
    let root = workspace_root();
    ["editor/scene.ron", "assets/scene.ron"]
        .iter()
        .map(|rel| root.join(rel))
        .find(|path| path.is_file())
}

/// Reloads the watched file into the world and publishes fresh snapshots.
/// A missing or malformed file keeps the live world untouched.
///
/// Self-save suppression: when the file's current `(mtime, size)` matches
/// the fingerprint [`EditorSession::save_scene_file`] remembered, the
/// observed change is the session's own save — the fingerprint is consumed
/// one-shot and the world (version, entity handles, physics/audio state)
/// is left untouched. Any actually different file reloads as before.
fn reload_watched_scene(
    world: &mut EditorSession,
    path: &ScenePath,
    ev_tx: &Sender<GameEvent>,
) -> ReloadDecision {
    let resolved = path.as_path();
    let current = SavedFingerprint::capture(resolved);
    if world.last_saved.as_ref() == Some(&current) {
        world.last_saved = None;
        return ReloadDecision::SkippedSelfSave;
    }
    match world.load_scene_file(path) {
        Ok(count) => {
            publish_state(world, ev_tx);
            eprintln!(
                "ornis: hot-reloaded {} ({} entities)",
                resolved.display(),
                count
            );
            ReloadDecision::Reloaded
        }
        Err(e) => {
            eprintln!("ornis: scene hot-reload skipped: {e}");
            ReloadDecision::Failed
        }
    }
}

/// Spawn the `editor-world` thread: owns the world, loads the startup scene,
/// executes commands from `cmd_rx`, and advances the fixed-rate domain frame
/// host between commands until the HTTP server side drops its sender.
///
/// Loop shape (one iteration):
///
/// ```text
/// recv first command (up to ~16 ms idle wait) -> drain ready burst ->
/// poll scene watcher -> tick with measured dt -> publish when changed
/// ```
///
/// Every wake — command or idle timeout — falls through to the watcher poll
/// and the tick, so a dense command burst can delay but never starve the
/// simulation or the file watcher. The tick delta is the real wall-clock
/// time since the previous tick clamped by [`clamp_frame_dt`]; an idle loop
/// keeps its historical ~16 ms cadence.
pub fn run(cmd_rx: Receiver<UiCommand>, ev_tx: Sender<GameEvent>) -> JoinHandle<()> {
    // Prefer a named thread; fall back to an anonymous one if the OS rejects
    // the name (values move into at most one closure).
    let named = thread::Builder::new().name("editor-world".into());
    match named.spawn({
        let cmd_rx = cmd_rx.clone();
        let ev_tx = ev_tx.clone();
        move || editor_world_loop(cmd_rx, ev_tx)
    }) {
        Ok(handle) => handle,
        Err(_) => thread::spawn(move || editor_world_loop(cmd_rx, ev_tx)),
    }
}

/// Editor-world tick loop: commands, scene hot-reload, and wall-clock frames.
fn editor_world_loop(cmd_rx: Receiver<UiCommand>, ev_tx: Sender<GameEvent>) {
    let mut world = EditorSession::new();
    match startup_scene_ron() {
        Some(ron) => {
            if let Err(e) = world.load_scene_ron(&ron) {
                eprintln!("ornis: failed to load startup scene: {e}");
            }
        }
        None => eprintln!("ornis: no startup scene found, starting with an empty world"),
    }
    // Publish the initial state so the HTTP caches are live
    // before the first command arrives.
    publish_state(&world, &ev_tx);
    let mut scene_watch = watched_scene_path().map(FileWatch::new);
    // Wall clock for the measured tick delta (see [`clamp_frame_dt`]).
    let mut last_tick = Instant::now();
    loop {
        match cmd_rx.recv_timeout(IDLE_POLL_INTERVAL) {
            Ok(first) => world.handle_command(&first, &ev_tx),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        // Drain the burst that queued while handling, without
        // blocking: the tick below still runs this iteration.
        let mut disconnected = false;
        loop {
            match cmd_rx.try_recv() {
                Ok(cmd) => world.handle_command(&cmd, &ev_tx),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if let Some(watch) = scene_watch.as_mut()
            && watch.poll()
        {
            // The watcher only observes workspace scene files, so this
            // resolve is a fail-closed formality (defense in depth).
            let watched = watch.path.clone();
            match ScenePath::resolve(&world.scene_roots, &watched.to_string_lossy()) {
                Ok(path) => {
                    reload_watched_scene(&mut world, &path, &ev_tx);
                }
                Err(e) => eprintln!("ornis: scene hot-reload skipped: {e}"),
            }
        }
        if world
            .tick_secs(clamp_frame_dt(last_tick.elapsed()))
            .changed()
        {
            publish_state(&world, &ev_tx);
        }
        last_tick = Instant::now();
        if disconnected {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::unbounded;

    fn world_and_events() -> (EditorSession, Sender<GameEvent>, Receiver<GameEvent>) {
        let (ev_tx, ev_rx) = unbounded();
        (EditorSession::new(), ev_tx, ev_rx)
    }

    fn custom(cmd_type: &str, json_data: &str) -> UiCommand {
        UiCommand::Custom {
            cmd_type: cmd_type.into(),
            json_data: json_data.into(),
        }
    }

    /// Typed generic upsert (remote maps `set_component` POSTs here).
    fn set_component(
        entity_id: u32,
        generation: Option<u32>,
        type_name: &str,
        json: &str,
    ) -> UiCommand {
        UiCommand::SetComponent {
            entity_id,
            generation,
            type_name: type_name.into(),
            json_data: json.into(),
        }
    }

    /// Drain all pending events from the channel.
    fn drain_all(rx: &Receiver<GameEvent>) -> Vec<GameEvent> {
        std::iter::from_fn(|| rx.try_recv().ok()).collect()
    }

    /// Extract `json_data` payloads of `CustomEvent`s with the given type.
    fn custom_events(events: &[GameEvent], cmd_type: &str) -> Vec<String> {
        events
            .iter()
            .filter_map(|ev| match ev {
                GameEvent::CustomEvent {
                    cmd_type: t,
                    json_data,
                } if t.as_str() == cmd_type => Some(json_data.clone()),
                _ => None,
            })
            .collect()
    }

    /// Typed `ComponentUpdated` events as (entity_id, type_name, json_data).
    fn component_updates(
        events: &[GameEvent],
    ) -> Vec<(u32, editor_backend::ipc::ComponentName, String)> {
        events
            .iter()
            .filter_map(|ev| match ev {
                GameEvent::ComponentUpdated {
                    entity_id,
                    type_name,
                    json_data,
                } => Some((*entity_id, type_name.clone(), json_data.clone())),
                _ => None,
            })
            .collect()
    }

    /// Float comparison: scene values are f32, JSON literals are f64.
    fn assert_f32_seq(v: &Value, expected: &[f32]) {
        let arr = v.as_array().expect("expected an array");
        assert_eq!(arr.len(), expected.len());
        for (a, b) in arr.iter().zip(expected) {
            let a = a.as_f64().expect("expected a number");
            assert!((a - f64::from(*b)).abs() < 1e-6, "{a} != {b}");
        }
    }

    fn assert_f32(v: &Value, expected: f32) {
        let a = v.as_f64().expect("expected a number");
        assert!((a - f64::from(expected)).abs() < 1e-6, "{a} != {expected}");
    }

    const FULL_TRANSFORM: &str = r#"{"translation":[1,2,3],"rotation":[0,0,0,1],"scale":[1,1,1]}"#;

    #[test]
    fn editor_session_uses_core_world_for_components_and_environment() {
        let mut world = EditorSession::new();
        let entity = world.spawn(Some("Hero".into()));

        assert!(world.world().store().is_some());
        assert!(
            world
                .world()
                .resources()
                .get::<SceneEnvironment>()
                .is_some()
        );
        assert!(
            world
                .world()
                .store()
                .expect("core World store")
                .read_lane::<TransformDesc>()
                .expect("Transform lane")
                .get(entity)
                .is_some()
        );
    }

    #[test]
    fn editor_tick_synchronizes_dynamic_physics_pose() {
        let mut world = EditorSession::new();
        let entity = world.spawn(None);
        world
            .world_mut()
            .store_mut()
            .expect("world store")
            .insert(entity, RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0));

        assert!(world.tick(1.0 / 60.0));

        assert!(world.to_scene().entities[0].transform.translation[1] < 0.0);
    }

    #[test]
    fn browser_wasd_input_drives_player_through_gameplay() {
        use editor_backend::ipc::BrowserInput;
        use ornis_gameplay::{Player, Position};

        let (ev_tx, _ev_rx) = unbounded();
        let mut world = EditorSession::new();
        let entity = world.spawn(None);
        world
            .world_mut()
            .store_mut()
            .expect("world store")
            .insert(entity, Player);
        world
            .world_mut()
            .store_mut()
            .expect("world store")
            .insert(entity, Position(Vec3::ZERO));
        // `spawn` attaches a static body (mass 0): the bridge skips static
        // bodies and `body_to_transform` would pin `Position` back. Gameplay
        // intent needs a dynamic body to reach the solver.
        world
            .world_mut()
            .store_mut()
            .expect("world store")
            .insert(entity, RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0));

        // Same channel as `POST /api/input` and the WS input frames.
        handle_command(
            &mut world,
            &UiCommand::Input {
                input: BrowserInput {
                    pressed_keys: vec![87],
                    ..BrowserInput::default()
                },
            },
            &ev_tx,
        );
        world.tick(1.0 / 60.0);
        // The fixed schedule runs before the variable one: tick 1 writes
        // gameplay intent (`player_input` → `Velocity`), tick 2 carries it
        // through the bridge (`velocity_to_body` → physics → `body_to_transform`).
        world.tick(1.0 / 60.0);

        let z = world
            .world()
            .store()
            .expect("world store")
            .read_lane::<Position>()
            .expect("Position lane")
            .get(entity)
            .expect("player position")
            .0
            .z;
        assert!(z < 0.0, "W must move the player forward (-z)");
    }

    fn temp_scene_path(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ornis-watch-{}-{}", std::process::id(), tag));
        fs::create_dir_all(&dir).expect("temp dir");
        dir.join("scene.ron")
    }

    /// Temp sandbox workspace (`<dir>/editor`, `<dir>/assets`) plus a
    /// session rooted at it: scene I/O tests never touch the real
    /// workspace trees. Returns the session and the dir (for cleanup).
    fn sandboxed_session(tag: &str) -> (EditorSession, PathBuf) {
        let dir = std::env::temp_dir().join(format!("ornis-sandbox-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("editor")).expect("sandbox editor");
        fs::create_dir_all(dir.join("assets")).expect("sandbox assets");
        let world = EditorSession::with_roots(SceneRoots::new(&dir));
        (world, dir)
    }

    #[test]
    fn scene_watch_fires_once_on_external_change() {
        let path = temp_scene_path("once");
        // Missing file: no fire, but appearance fires exactly once.
        let mut watch = FileWatch::new(path.clone());
        assert!(!watch.poll());
        fs::write(&path, "v: 1").expect("write temp scene");
        assert!(watch.poll());
        assert!(!watch.poll());
        // A watch created over existing content is baselined: no fire.
        let mut watch = FileWatch::new(path.clone());
        assert!(!watch.poll());
        let _ = fs::remove_file(&path);
    }

    /// End-to-end mutation bus: a producer emits `Mutation::Set` patches,
    /// `tick` drains them into the world and bumps the version — the same
    /// path `set_component` takes, without a scripting language involved.
    #[test]
    fn producer_tick_applies_mutations_to_world() {
        use ornis_core::mutation::{Mutation, MutationProducer};
        struct Rename {
            entity: Entity,
        }
        impl MutationProducer for Rename {
            fn produce(&mut self, _dt: f32, _tick: u64) -> Vec<Mutation> {
                vec![Mutation::Set {
                    entity: self.entity,
                    component: "Name".into(),
                    value: serde_json::json!("renamed"),
                }]
            }
        }
        let mut world = EditorSession::new();
        let entity = world.spawn(Some("ticked".into()));
        let version_before = world.version;
        world
            .mutation_bus()
            .expect("bus installed")
            .add_producer(Box::new(Rename { entity }));
        assert!(
            world.tick(1.0 / 60.0),
            "producer mutation must mark the world changed"
        );
        assert!(
            world.version > version_before,
            "apply must bump the version"
        );
        let name: Option<Name> = world
            .store()
            .and_then(|store| read_component(store, entity));
        assert_eq!(name.expect("Name lane").0, "renamed");
    }

    #[test]
    fn clamp_frame_dt_bounds_hitches_but_keeps_idle_steps() {
        let idle = clamp_frame_dt(Duration::from_millis(16)).get();
        assert!(
            (idle - 0.016).abs() < 1e-6,
            "idle step passes through: {idle}"
        );
        assert_eq!(clamp_frame_dt(Duration::from_secs(10)), MAX_FRAME_DT);
        assert_eq!(clamp_frame_dt(Duration::ZERO), Seconds::ZERO);
    }

    #[test]
    fn tick_secs_reports_typed_outcome() {
        let mut world = EditorSession::new();
        assert_eq!(
            world.tick_secs(Seconds::new(1.0 / 60.0)),
            TickOutcome::Unchanged
        );
        let entity = world.spawn(None);
        world
            .world_mut()
            .store_mut()
            .expect("world store")
            .insert(entity, RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0));
        assert_eq!(
            world.tick_secs(Seconds::new(1.0 / 60.0)),
            TickOutcome::Changed
        );
    }

    /// The session's own save must not hot-reload the world back onto
    /// itself (that path rebuilds physics/audio from scratch), while a
    /// genuine external edit still replaces it. Mirrors the production
    /// order in [`run`]: save -> tick -> watcher poll -> reload gate.
    #[test]
    fn save_suppresses_self_reload_but_external_edit_reloads() {
        let (ev_tx, _ev_rx) = unbounded();
        let (mut world, dir) = sandboxed_session("self-save");
        world.spawn(Some("Hero".into()));
        let version = world.version;
        // The watcher is born before the save, like the loop's startup watch.
        let path = world.scene_path("editor/self-save.ron").expect("sandboxed");
        let _ = fs::remove_file(path.as_path());
        let mut watch = FileWatch::new(path.as_path().to_path_buf());

        world.save_scene_file(&path).expect("save");
        world.tick(1.0 / 60.0);
        assert!(watch.poll(), "the save bumps mtime, the poll observes it");
        assert_eq!(
            reload_watched_scene(&mut world, &path, &ev_tx),
            ReloadDecision::SkippedSelfSave
        );
        assert_eq!(world.version, version, "skipped reload bumps nothing");
        assert_eq!(world.entity_count(), 1);
        assert_eq!(
            world.name_of(world.alive[0]).as_deref(),
            Some("Hero"),
            "handles stay stable"
        );

        // External edit (removal first, so even coarse-mtime filesystems
        // observe a change): the gate must reload now.
        let edited = fs::read_to_string(path.as_path())
            .expect("saved ron")
            .replacen("Hero", "Outsider", 1);
        let _ = fs::remove_file(path.as_path());
        assert!(!watch.poll(), "disappearance alone never reloads");
        fs::write(path.as_path(), edited).expect("external edit");
        assert!(watch.poll(), "external edit fires the watcher");
        assert_eq!(
            reload_watched_scene(&mut world, &path, &ev_tx),
            ReloadDecision::Reloaded
        );
        assert!(world.version > version, "reload bumps the version");
        assert_eq!(
            world.name_of(world.alive[0]).as_deref(),
            Some("Outsider"),
            "external content replaced the world"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn hot_reload_replaces_world_and_survives_garbage() {
        let (ev_tx, _ev_rx) = unbounded();
        let (mut world, dir) = sandboxed_session("reload");
        let ron = fs::read_to_string(scene_file_path()).expect("editor/scene.ron readable");
        let path = world.scene_path("editor/scene.ron").expect("sandboxed");
        fs::write(path.as_path(), &ron).expect("write temp scene");

        reload_watched_scene(&mut world, &path, &ev_tx);
        assert_eq!(world.entity_count(), 5);

        fs::write(path.as_path(), "not ron {{{").expect("write garbage");
        reload_watched_scene(&mut world, &path, &ev_tx);
        assert_eq!(world.entity_count(), 5);
        let _ = fs::remove_dir_all(&dir);
    }

    /// `.glb` dispatch rejects garbage without touching the world; the
    /// positive bytes→`Scene` path is pinned in `ornis-assets`.
    #[test]
    fn load_scene_file_rejects_gltf_garbage() {
        let (mut world, dir) = sandboxed_session("garbage");
        let version = world.version;
        let path = world.scene_path("editor/garbage.glb").expect("sandboxed");
        fs::write(path.as_path(), "not a glb at all").expect("write garbage");
        assert!(world.load_scene_file(&path).is_err());
        assert_eq!(world.version, version);
        assert_eq!(world.entity_count(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Queued asset reloads replace the world on the next tick, in id
    /// order, with a version bump — the dirty-set is a live input
    /// channel, not a dead API.
    #[test]
    fn drain_assets_replaces_world_on_tick() {
        let mut world = EditorSession::new();
        assert_eq!(world.entity_count(), 0);
        let a = world
            .world
            .assets_mut()
            .load_scene_ron("Scene(name: \"a\", entities: [], lights: [], camera: (position: (0.0, 2.5, 9.0), target: (0.0, 0.0, 0.0), up: (0.0, 1.0, 0.0), fov: 60.0, near: 0.1, far: 100.0), ambient: (0.1, 0.1, 0.1))")
            .expect("scene a loads");
        let version_before = world.version;
        assert!(world.world.assets_mut().request_reload(a));
        assert!(world.tick(1.0 / 60.0), "asset reload must mark changed");
        assert!(world.version > version_before);
        assert_eq!(world.scene_name, "a");
        assert!(world.world.assets_mut().take_dirty().is_empty());
    }

    /// Invalid RON is a typed [`SceneLoadError`](ornis_assets::SceneLoadError),
    /// not a `String`: the world is untouched and the message survives.
    #[test]
    fn load_scene_ron_reports_typed_errors() {
        let mut world = EditorSession::new();
        let error = world
            .load_scene_ron("Scene(name: 42)")
            .expect_err("malformed RON fails");
        assert!(!error.message().is_empty());
        assert_eq!(world.entity_count(), 0);
    }

    /// The session installs object animation into the frame schedule
    /// (physics bodies skipped by the sampler itself).
    #[test]
    fn session_installs_object_animation() {
        let world = EditorSession::new();
        let mermaid = world.world.engine().schedule().mermaid();
        assert!(mermaid.contains("anim_sample"), "anim wired:\n{mermaid}");
    }

    #[test]
    fn spawn_assigns_names_and_counts() {
        let mut world = EditorSession::new();
        let a = world.spawn(None);
        let b = world.spawn(Some("Hero".into()));
        assert_eq!(world.entity_count(), 2);
        assert_eq!(world.name_of(a).as_deref(), Some("Entity 0"));
        assert_eq!(world.name_of(b).as_deref(), Some("Hero"));
        assert_ne!(a.id(), b.id());
    }

    /// Auto recipes build exact bodies at spawn: box meshes get box
    /// bodies (the old helper only knew spheres), planes get none.
    #[test]
    fn spawn_builds_exact_bodies_from_auto_recipes() {
        use ornis_physics::Shape;
        let mut world = EditorSession::new();
        let sphere = world.spawn(None);
        let lane = world
            .store()
            .and_then(|store| store.read_lane::<RigidBody>())
            .expect("body lane");
        assert!(matches!(
            lane.get(sphere).expect("sphere body").shape,
            Shape::Sphere { .. }
        ));
        drop(lane);
        world.spawn_with(
            Some("box".into()),
            TransformDesc::IDENTITY,
            MeshDesc::Box {
                size: [
                    PositiveF32::expect_valid(2.0),
                    PositiveF32::expect_valid(4.0),
                    PositiveF32::expect_valid(6.0),
                ],
            },
            default_material(),
        );
        let lane = world
            .store()
            .and_then(|store| store.read_lane::<RigidBody>())
            .expect("body lane");
        let boxed = world
            .alive
            .iter()
            .find(|e| world.name_of(**e).as_deref() == Some("box"))
            .expect("box entity");
        assert!(matches!(
            lane.get(*boxed).expect("box body").shape,
            Shape::Box { .. }
        ));
    }

    #[test]
    fn despawn_recycles_ids_with_new_generation() {
        let mut world = EditorSession::new();
        let a = world.spawn(None);
        let b = world.spawn(None);
        // Stale generation must not despawn.
        assert!(world.despawn(a.id(), a.generation() + 1).is_none());
        assert_eq!(world.entity_count(), 2);
        assert_eq!(world.despawn(a.id(), a.generation()), Some(a));
        assert_eq!(world.entity_count(), 1);
        let c = world.spawn(None);
        assert_eq!(c.id(), a.id());
        assert_ne!(c.generation(), a.generation());
        assert_eq!(world.name_of(b).as_deref(), Some("Entity 1"));
    }

    #[test]
    fn scene_ron_round_trip() {
        let ron = fs::read_to_string(scene_file_path()).expect("editor/scene.ron readable");
        let mut world = EditorSession::new();
        let loaded = world.load_scene_ron(&ron).expect("scene loads");
        assert_eq!(loaded, 5);
        assert_eq!(world.entity_count(), 5);
        assert_eq!(world.version, 5);

        let scene: Value = serde_json::from_str(&world.scene_json()).unwrap();
        assert_eq!(scene["version"], 5);
        assert_eq!(scene["entity_count"], 5);
        let entities = scene["entities"].as_array().unwrap();
        assert_eq!(entities.len(), 5);

        // Canonical serde shapes: components keyed by registry name,
        // enums externally tagged.
        let red = &entities[0];
        assert_eq!(red["id"], 0);
        assert_eq!(red["generation"], 0);
        let red_components = &red["components"];
        assert_eq!(red_components.as_object().unwrap().len(), 4);
        assert_eq!(red_components["Name"], "Red Sphere");
        assert_f32_seq(
            &red_components["Transform"]["translation"],
            &[-5.6, 0.0, 0.0],
        );
        assert_f32_seq(
            &red_components["Transform"]["rotation"],
            &[0.0, 0.0, 0.0, 1.0],
        );
        assert_f32_seq(&red_components["Transform"]["scale"], &[1.0, 1.0, 1.0]);
        assert_f32(&red_components["Mesh"]["Sphere"]["radius"], 1.0);
        assert_eq!(red_components["Mesh"]["Sphere"]["segments"], 32);
        assert_eq!(red_components["Mesh"]["Sphere"]["rings"], 24);
        assert_f32_seq(
            &red_components["Material"]["Dielectric"]["base_color"],
            &[0.8, 0.2, 0.2],
        );
        assert_f32(&red_components["Material"]["Dielectric"]["roughness"], HALF);

        // Material variants survive the round trip.
        let gold = &entities[3]["components"];
        assert_eq!(gold["Name"], "Gold Sphere");
        assert_f32_seq(&gold["Material"]["Metal"]["base_color"], &[0.9, 0.7, 0.1]);
        let ceramic = &entities[4]["components"];
        assert_eq!(ceramic["Name"], "Ceramic Sphere");
        assert_f32(&ceramic["Material"]["Coat"]["coat_weight"], 1.0);
        assert_f32(&ceramic["Material"]["Coat"]["coat_roughness"], 0.1);

        // Environment resource: serde-canonical enum tagging here too.
        let lights = scene["lights"].as_array().unwrap();
        assert_eq!(lights.len(), 2);
        // Directions are unit vectors once loaded: `(1, 1, 1)` in the file
        // reads back normalized (same light, the renderer normalized anyway).
        assert_f32_seq(
            &lights[0]["Directional"]["direction"],
            &[0.577_350_26, 0.577_350_26, 0.577_350_26],
        );
        assert_f32(&lights[0]["Directional"]["intensity"], 0.6);
        assert_f32_seq(&lights[1]["Directional"]["color"], &[0.8, 0.8, 1.0]);
        assert_f32_seq(&scene["camera"]["position"], &[0.0, 2.5, 9.0]);
        assert_f32(&scene["camera"]["fov"], 60.0);
        assert_f32(&scene["camera"]["near"], 0.1);
        assert_f32(&scene["camera"]["far"], 100.0);
        assert_f32_seq(&scene["ambient"], &[0.10, 0.10, 0.15]);
    }

    #[test]
    fn scene_json_lists_entities_with_full_components() {
        let mut world = EditorSession::new();
        world.spawn(None);
        world.spawn(Some("Hero".into()));
        let scene: Value = serde_json::from_str(&world.scene_json()).unwrap();
        assert_eq!(scene["entity_count"], 2);
        assert_eq!(scene["version"], 2);
        let entities = scene["entities"].as_array().unwrap();
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0]["id"], 0);
        assert_eq!(entities[0]["generation"], 0);

        let components = &entities[0]["components"];
        assert_eq!(components.as_object().unwrap().len(), 4);
        assert_eq!(components["Name"], "Entity 0");
        // Default components: unit sphere at the origin, gray dielectric.
        assert_f32_seq(&components["Transform"]["translation"], &[0.0, 0.0, 0.0]);
        assert_f32_seq(&components["Transform"]["rotation"], &[0.0, 0.0, 0.0, 1.0]);
        assert!(components["Mesh"]["Sphere"].is_object());
        assert_f32_seq(
            &components["Material"]["Dielectric"]["base_color"],
            &[HALF, HALF, HALF],
        );
        assert_eq!(entities[1]["components"]["Name"], "Hero");
    }

    #[test]
    fn scene_json_empty_world() {
        let world = EditorSession::new();
        let scene: Value = serde_json::from_str(&world.scene_json()).unwrap();
        assert_eq!(scene["version"], 0);
        assert_eq!(scene["entity_count"], 0);
        assert_eq!(scene["entities"], serde_json::json!([]));
        assert_eq!(scene["lights"], serde_json::json!([]));
        assert!(scene["camera"].is_object());
        assert!(scene["ambient"].is_array());
    }

    #[test]
    fn version_increments_only_on_mutations() {
        let (mut world, ev_tx, _ev_rx) = world_and_events();
        assert_eq!(world.version, 0);
        world.handle_command(&custom("create_entity", ""), &ev_tx);
        assert_eq!(world.version, 1);
        world.handle_command(
            &set_component(0, Some(0), "Transform", FULL_TRANSFORM),
            &ev_tx,
        );
        assert_eq!(world.version, 2);
        // Failed command (no such entity): no bump.
        world.handle_command(
            &set_component(9, Some(0), "Transform", FULL_TRANSFORM),
            &ev_tx,
        );
        assert_eq!(world.version, 2);
        world.handle_command(&set_component(0, Some(0), "Name", r#""X""#), &ev_tx);
        assert_eq!(world.version, 3);
        world.handle_command(
            &set_component(
                0,
                Some(0),
                "Material",
                r#"{"Metal":{"base_color":[0.9,0.7,0.1],"roughness":0.2}}"#,
            ),
            &ev_tx,
        );
        assert_eq!(world.version, 4);
        world.handle_command(
            &custom("destroy_entity", r#"{"id":0,"generation":0}"#),
            &ev_tx,
        );
        assert_eq!(world.version, 5);
        let scene: Value = serde_json::from_str(&world.scene_json()).unwrap();
        assert_eq!(scene["version"], 5);
    }

    #[test]
    fn request_id_command_emits_correlated_completion() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(
            &UiCommand::WithRequestId {
                request_id: RequestId::new(77),
                command: Box::new(custom("create_entity", r#"{"name":"Hero"}"#)),
            },
            &ev_tx,
        );

        let events = drain_all(&ev_rx);
        let completion = events.iter().find_map(|event| match event {
            GameEvent::CommandCompleted {
                request_id,
                command,
                success,
                error,
            } => Some((*request_id, command.clone(), *success, error.clone())),
            _ => None,
        });
        assert_eq!(
            completion,
            Some((RequestId::new(77), EditorCommand::CreateEntity, true, None))
        );
    }

    #[test]
    fn create_entity_command_emits_events_and_updates_state() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", r#"{"name":"Hero"}"#), &ev_tx);
        assert_eq!(world.entity_count(), 1);

        let events = drain_all(&ev_rx);

        let created = custom_events(&events, "entity_created");
        assert_eq!(created.len(), 1);
        let created: Value = serde_json::from_str(&created[0]).unwrap();
        assert_eq!(created["id"], 0);
        assert_eq!(created["generation"], 0);
        assert_eq!(created["name"], "Hero");

        let statuses = custom_events(&events, "status");
        assert_eq!(statuses.len(), 1);
        let status: Value = serde_json::from_str(&statuses[0]).unwrap();
        assert_eq!(status["entity_count"], 1);
        assert_eq!(status["version"], 1);

        let scenes = custom_events(&events, "scene");
        assert_eq!(scenes.len(), 1);
        let scene: Value = serde_json::from_str(&scenes[0]).unwrap();
        assert_eq!(scene["entity_count"], 1);
        assert_eq!(scene["entities"][0]["components"]["Name"], "Hero");

        assert!(ev_rx.try_recv().is_err(), "no leftover events");
    }

    #[test]
    fn create_entity_with_component_overrides() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(
            &custom(
                "create_entity",
                r#"{
                    "name": "Metal Ball",
                    "components": {
                        "Transform": {"translation":[1,2,3],"rotation":[0,0,0,1],"scale":[2,2,2]},
                        "Mesh": {"Sphere":{"radius":2.0,"segments":16,"rings":8}},
                        "Material": {"Metal":{"base_color":[0.9,0.7,0.1],"roughness":0.2}}
                    }
                }"#,
            ),
            &ev_tx,
        );
        assert_eq!(world.entity_count(), 1);

        let scene: Value = serde_json::from_str(&world.scene_json()).unwrap();
        let components = &scene["entities"][0]["components"];
        assert_eq!(components["Name"], "Metal Ball");
        assert_f32_seq(&components["Transform"]["translation"], &[1.0, 2.0, 3.0]);
        assert_f32_seq(&components["Transform"]["scale"], &[2.0, 2.0, 2.0]);
        assert_f32(&components["Mesh"]["Sphere"]["radius"], 2.0);
        assert_eq!(components["Mesh"]["Sphere"]["segments"], 16);
        assert_f32_seq(
            &components["Material"]["Metal"]["base_color"],
            &[0.9, 0.7, 0.1],
        );
        assert_f32(&components["Material"]["Metal"]["roughness"], 0.2);

        let created = custom_events(&drain_all(&ev_rx), "entity_created");
        assert_eq!(created.len(), 1);
        let created: Value = serde_json::from_str(&created[0]).unwrap();
        assert_eq!(created["id"], 0);
        assert_eq!(created["name"], "Metal Ball");
    }

    #[test]
    fn create_entity_without_name_uses_default() {
        let (mut world, ev_tx, _ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", ""), &ev_tx);
        let scene: Value = serde_json::from_str(&world.scene_json()).unwrap();
        assert_eq!(scene["entities"][0]["components"]["Name"], "Entity 0");
    }

    #[test]
    fn set_component_replaces_component_generically() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", ""), &ev_tx);
        // Full replace: the payload is the whole component, field-level
        // merging is the client's job (editor.js keeps the snapshot).
        world.handle_command(
            &set_component(0, Some(0), "Transform", FULL_TRANSFORM),
            &ev_tx,
        );

        let scene: Value = serde_json::from_str(&world.scene_json()).unwrap();
        let transform = &scene["entities"][0]["components"]["Transform"];
        assert_f32_seq(&transform["translation"], &[1.0, 2.0, 3.0]);
        assert_f32_seq(&transform["rotation"], &[0.0, 0.0, 0.0, 1.0]);
        assert_f32_seq(&transform["scale"], &[1.0, 1.0, 1.0]);

        let events = drain_all(&ev_rx);
        let updated = component_updates(&events);
        assert_eq!(updated.len(), 1);
        assert_eq!(updated[0].0, 0);
        assert_eq!(updated[0].1, "Transform");
        assert!(updated[0].2.contains("translation"));
    }

    #[test]
    fn set_component_material_and_name() {
        let (mut world, ev_tx, _ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", ""), &ev_tx);
        world.handle_command(
            &set_component(
                0,
                Some(0),
                "Material",
                r#"{"Coat":{"base_color":[1,1,1],"coat_weight":1.0,"coat_roughness":0.1}}"#,
            ),
            &ev_tx,
        );
        world.handle_command(&set_component(0, Some(0), "Name", r#""Warden""#), &ev_tx);

        let scene: Value = serde_json::from_str(&world.scene_json()).unwrap();
        let components = &scene["entities"][0]["components"];
        assert_f32(&components["Material"]["Coat"]["coat_weight"], 1.0);
        assert_f32(&components["Material"]["Coat"]["coat_roughness"], 0.1);
        assert_eq!(components["Name"], "Warden");
    }

    #[test]
    fn set_component_errors_leave_world_untouched() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", ""), &ev_tx);
        let version = world.version;

        // Unknown component type.
        world.handle_command(&set_component(0, Some(0), "NoSuchComponent", "{}"), &ev_tx);
        // Malformed component JSON.
        world.handle_command(&set_component(0, Some(0), "Transform", "{broken"), &ev_tx);
        // Schema mismatch (rotation is missing).
        world.handle_command(
            &set_component(0, Some(0), "Transform", r#"{"translation":[1,2,3]}"#),
            &ev_tx,
        );
        // Stale generation does not match the alive entity.
        world.handle_command(&set_component(0, Some(7), "Name", r#""Ghost""#), &ev_tx);
        // Generation omitted: matches any alive entity with the id —
        // this one SUCCEEDS, hence the separate version check below.
        world.handle_command(&set_component(0, None, "Name", r#""Real""#), &ev_tx);

        let events = drain_all(&ev_rx);
        assert_eq!(custom_events(&events, "error").len(), 4);
        assert_eq!(component_updates(&events).len(), 1);
        assert_eq!(world.version, version.bumped());

        let scene: Value = serde_json::from_str(&world.scene_json()).unwrap();
        let components = &scene["entities"][0]["components"];
        assert_eq!(components["Name"], "Real");
        // The failed Transform write must not have landed.
        assert_f32_seq(&components["Transform"]["translation"], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn destroy_entity_command_removes_entity() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", ""), &ev_tx);
        world.handle_command(&custom("create_entity", ""), &ev_tx);
        while ev_rx.try_recv().is_ok() {}

        world.handle_command(
            &custom("destroy_entity", r#"{"id":0,"generation":0}"#),
            &ev_tx,
        );
        assert_eq!(world.entity_count(), 1);
        let events = drain_all(&ev_rx);
        let destroyed = custom_events(&events, "entity_destroyed");
        assert_eq!(destroyed.len(), 1);
        let destroyed: Value = serde_json::from_str(&destroyed[0]).unwrap();
        assert_eq!(destroyed["id"], 0);
        assert_eq!(destroyed["generation"], 0);

        // Wrong generation: error event, world untouched.
        let version = world.version;
        world.handle_command(
            &custom("destroy_entity", r#"{"id":1,"generation":9}"#),
            &ev_tx,
        );
        assert_eq!(world.entity_count(), 1);
        assert_eq!(world.version, version);
        let events = drain_all(&ev_rx);
        assert_eq!(custom_events(&events, "entity_destroyed").len(), 0);
        assert_eq!(custom_events(&events, "error").len(), 1);
    }

    #[test]
    fn invalid_commands_emit_error_events_without_mutating() {
        let (mut world, ev_tx, ev_rx) = world_and_events();

        // Broken JSON body.
        world.handle_command(&custom("create_entity", "{not json"), &ev_tx);
        // Unknown command type.
        world.handle_command(&custom("nonsense", ""), &ev_tx);
        // set_component on a non-existent entity.
        world.handle_command(
            &set_component(
                3,
                None,
                "Material",
                r#"{"Metal":{"base_color":[1,1,1],"roughness":0.2}}"#,
            ),
            &ev_tx,
        );
        // Unknown component in create overrides: no entity must appear.
        world.handle_command(
            &custom(
                "create_entity",
                r#"{"components":{"Unobtainium":{"density":1}}}"#,
            ),
            &ev_tx,
        );
        // Payload that is valid JSON but not an object.
        world.handle_command(&custom("create_entity", r#""oops""#), &ev_tx);

        assert_eq!(world.entity_count(), 0);
        assert_eq!(world.version, 0);

        let errors = custom_events(&drain_all(&ev_rx), "error");
        assert_eq!(errors.len(), 5);
        for e in &errors {
            let e: Value = serde_json::from_str(e).unwrap();
            assert!(e["command"].is_string());
            assert!(e["message"].is_string());
        }
    }

    #[test]
    fn list_entities_command_reports_entities() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", r#"{"name":"A"}"#), &ev_tx);
        world.handle_command(&custom("create_entity", r#"{"name":"B"}"#), &ev_tx);
        while ev_rx.try_recv().is_ok() {}

        world.handle_command(&custom("list_entities", ""), &ev_tx);
        let lists = custom_events(&drain_all(&ev_rx), "entity_list");
        assert_eq!(lists.len(), 1);
        let list: Value = serde_json::from_str(&lists[0]).unwrap();
        assert_eq!(list["count"], 2);
        assert_eq!(list["entities"][0]["id"], 0);
        assert_eq!(list["entities"][0]["name"], "A");
        assert_eq!(list["entities"][1]["id"], 1);
        assert_eq!(list["entities"][1]["name"], "B");
    }

    #[test]
    fn typed_create_and_destroy_variants() {
        let (mut world, ev_tx, _ev_rx) = world_and_events();
        world.handle_command(&UiCommand::CreateEntity, &ev_tx);
        assert_eq!(world.entity_count(), 1);
        world.handle_command(&UiCommand::DestroyEntity { entity_id: 0 }, &ev_tx);
        assert_eq!(world.entity_count(), 0);
        // Destroying an unknown entity is a no-op.
        world.handle_command(&UiCommand::DestroyEntity { entity_id: 42 }, &ev_tx);
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn run_thread_loads_startup_scene_and_processes_commands() {
        let (cmd_tx, cmd_rx) = unbounded();
        let (ev_tx, ev_rx) = unbounded();
        let handle = run(cmd_rx, ev_tx);

        cmd_tx
            .send(custom("create_entity", r#"{"name":"Hero"}"#))
            .unwrap();
        // Wait for the scene snapshot reflecting the new entity.
        let mut seen_scene = None;
        for _ in 0..100 {
            while let Ok(ev) = ev_rx.try_recv() {
                if let GameEvent::CustomEvent {
                    cmd_type,
                    json_data,
                } = ev
                    && cmd_type == "scene"
                {
                    seen_scene = Some(json_data);
                }
            }
            if let Some(scene) = &seen_scene
                && scene.contains("Hero")
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let scene = seen_scene.expect("scene snapshot expected");
        assert!(scene.contains("Hero"));
        // The startup scene (5 spheres from editor/scene.ron) was loaded too.
        assert!(scene.contains("Red Sphere"));

        drop(cmd_tx);
        handle.join().expect("editor-world thread must finish");
    }

    // ── save/load scene ────────────────────────────────────────────────────

    /// Snapshot JSON with `version` stripped: two worlds compare by content.
    fn scene_value(world: &EditorSession) -> Value {
        let mut v: Value = serde_json::from_str(&world.scene_json()).unwrap();
        v.as_object_mut().unwrap().remove("version");
        v
    }

    #[test]
    fn to_scene_round_trip_through_ron_preserves_world() {
        let ron = fs::read_to_string(scene_file_path()).expect("editor/scene.ron readable");
        let mut world = EditorSession::new();
        world.load_scene_ron(&ron).expect("scene loads");
        // A runtime-created entity must round-trip too.
        world.spawn_with(
            Some("Extra".into()),
            TransformDesc::from_arrays([9.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 2.0]),
            MeshDesc::Sphere {
                radius: PositiveF32::expect_valid(HALF),
                segments: 8,
                rings: 4,
            },
            MaterialDesc::Metal {
                base_color: [1.0, 0.0, 0.0],
                roughness: Clamped01::new(0.3),
                emission: [0.0, 0.0, 0.0],
                metallic: ornis_core::Metallic::new(1.0),
            },
        );

        let serialized = world.to_scene().to_ron().expect("serialize");
        let reparsed = Scene::from_ron(&serialized).expect("re-parse");

        let mut restored = EditorSession::new();
        let loaded = restored.load_scene(reparsed);
        assert_eq!(loaded, 6);
        assert_eq!(scene_value(&restored), scene_value(&world));
        // Version: max(loaded entity count, old version + 1) — here the
        // 6 spawns dominate over the fresh world's `0 + 1`.
        assert_eq!(restored.version, world.version);
    }

    #[test]
    fn save_and_load_file_round_trip() {
        let (mut world, dir) = sandboxed_session("save-load");
        let path = world.scene_path("editor/scene.ron").expect("sandboxed");

        world.spawn(Some("Hero".into()));
        world.save_scene_file(&path).expect("save");

        // The file on disk is a valid scene with the world's content.
        let on_disk =
            Scene::from_ron(&fs::read_to_string(path.as_path()).unwrap()).expect("valid RON");
        assert_eq!(on_disk.entities.len(), 1);
        assert_eq!(on_disk.entities[0].name, "Hero");

        let mut restored = EditorSession::with_roots(SceneRoots::new(&dir));
        let loaded = restored.load_scene_file(&path).expect("load");
        assert_eq!(loaded, 1);
        assert_eq!(scene_value(&restored), scene_value(&world));
        // The temp file was renamed away — no litter.
        assert!(
            !dir.join("editor/scene.ron.tmp").exists(),
            "no temp file left"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_failure_never_leaves_partial_files() {
        // The sandboxed target directory does not exist: the lexical resolve
        // still passes (nothing to canonicalize), but the write fails cleanly.
        let dir = std::env::temp_dir().join(format!("ornis-save-fail-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut world = EditorSession::with_roots(SceneRoots::new(&dir));
        let path = world.scene_path("editor/scene.ron").expect("sandboxed");

        assert!(world.save_scene_file(&path).is_err());
        assert!(!path.as_path().exists(), "no partial scene file");
        assert!(
            !dir.join("editor/scene.ron.tmp").exists(),
            "no temp file left"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_broken_or_missing_file_keeps_world() {
        let (ev_tx, ev_rx) = unbounded();
        let (mut world, dir) = sandboxed_session("load-broken");
        world.handle_command(&custom("create_entity", r#"{"name":"Keep"}"#), &ev_tx);
        while ev_rx.try_recv().is_ok() {}
        let before = scene_value(&world);
        let version = world.version;

        fs::write(dir.join("editor/broken.ron"), "Scene(name: 42)").unwrap();

        world.handle_command(
            &custom("load_scene", r#"{"path":"editor/broken.ron"}"#),
            &ev_tx,
        );
        world.handle_command(
            &custom("load_scene", r#"{"path":"editor/nope.ron"}"#),
            &ev_tx,
        );

        let events = drain_all(&ev_rx);
        assert_eq!(custom_events(&events, "error").len(), 2);
        assert_eq!(custom_events(&events, "scene_loaded").len(), 0);
        assert_eq!(scene_value(&world), before, "world untouched");
        assert_eq!(world.version, version, "version untouched");
        let _ = fs::remove_dir_all(&dir);
    }

    /// Escaping `save_scene`/`load_scene` paths are rejected with `error`
    /// events: `..` traversal, absolute paths outside the sandbox, and the
    /// sibling-prefix trap (`editor-evil/`). The world and the version stay
    /// untouched, and no file is created outside the sandbox.
    #[test]
    fn scene_commands_reject_paths_outside_sandbox() {
        let (ev_tx, ev_rx) = unbounded();
        let (mut world, dir) = sandboxed_session("sandbox-cmds");
        world.handle_command(&custom("create_entity", r#"{"name":"Keep"}"#), &ev_tx);
        while ev_rx.try_recv().is_ok() {}
        let before = scene_value(&world);
        let version = world.version;

        // Legit sandboxed paths resolve: default, `editor/`, `assets/`.
        let defaults =
            command_path(&world.scene_roots, &serde_json::json!({})).expect("default resolves");
        assert!(defaults.as_path().ends_with("editor/scene.ron"));
        for raw in ["editor/scene.ron", "assets/custom.ron"] {
            let resolved = world.scene_path(raw).expect("legit path resolves");
            assert!(resolved.as_path().starts_with(&dir), "{raw}");
        }

        let evil = dir.join("editor-evil/x.ron").to_string_lossy().into_owned();
        for raw in [
            "../../evil.ron",
            "editor/../../evil.ron",
            "/etc/passwd",
            evil.as_str(),
        ] {
            assert!(world.scene_path(raw).is_err(), "{raw} must not resolve");
            world.handle_command(
                &custom("save_scene", &format!(r#"{{"path":"{raw}"}}"#)),
                &ev_tx,
            );
            world.handle_command(
                &custom("load_scene", &format!(r#"{{"path":"{raw}"}}"#)),
                &ev_tx,
            );
        }

        let events = drain_all(&ev_rx);
        assert_eq!(custom_events(&events, "error").len(), 8);
        assert_eq!(custom_events(&events, "scene_saved").len(), 0);
        assert_eq!(custom_events(&events, "scene_loaded").len(), 0);
        assert_eq!(scene_value(&world), before, "world untouched");
        assert_eq!(world.version, version, "version untouched");
        assert!(!dir.join("evil.ron").exists(), "no escape write");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_load_commands_emit_events_and_restore_state() {
        let (ev_tx, ev_rx) = unbounded();
        let (mut world, dir) = sandboxed_session("cmds");
        let arg = r#"{"path":"editor/scene.ron"}"#;
        let path = dir.join("editor/scene.ron");

        world.handle_command(&custom("create_entity", r#"{"name":"Hero"}"#), &ev_tx);
        world.handle_command(&custom("save_scene", arg), &ev_tx);

        let events = drain_all(&ev_rx);
        let saved = custom_events(&events, "scene_saved");
        assert_eq!(saved.len(), 1);
        let saved: Value = serde_json::from_str(&saved[0]).unwrap();
        assert!(saved["path"].as_str().unwrap().ends_with("scene.ron"));
        assert_eq!(saved["version"], 1);
        assert!(path.exists());

        // Mutate: create + destroy; then load brings the saved state back.
        world.handle_command(&custom("create_entity", r#"{"name":"Temp"}"#), &ev_tx);
        let hero = world.alive[0];
        world.handle_command(
            &custom(
                "destroy_entity",
                &format!(
                    r#"{{"id":{},"generation":{}}}"#,
                    hero.id(),
                    hero.generation()
                ),
            ),
            &ev_tx,
        );
        assert_eq!(world.entity_count(), 1);
        assert_eq!(world.name_of(world.alive[0]).as_deref(), Some("Temp"));
        while ev_rx.try_recv().is_ok() {}

        world.handle_command(&custom("load_scene", arg), &ev_tx);
        assert_eq!(world.entity_count(), 1);

        let events = drain_all(&ev_rx);
        let loaded = custom_events(&events, "scene_loaded");
        assert_eq!(loaded.len(), 1);
        let loaded: Value = serde_json::from_str(&loaded[0]).unwrap();
        assert_eq!(loaded["entity_count"], 1);
        // Fresh snapshots were published after the load.
        let scenes = custom_events(&events, "scene");
        assert_eq!(scenes.len(), 1);
        let scene: Value = serde_json::from_str(&scenes[0]).unwrap();
        assert_eq!(scene["entities"][0]["components"]["Name"], "Hero");
        let _ = fs::remove_dir_all(&dir);
    }

    // ── E0 editor domain (PLAN §i) ───────────────────────────────────────

    /// Every construction route installs the editor domain (marker lanes +
    /// skeleton system); `load_scene` rebuilds through `new`, so the
    /// install survives world replacement. An empty tick never fails.
    #[test]
    fn editor_domain_installed_on_every_construction_route() {
        for mut world in [EditorSession::new(), EditorSession::default()] {
            let store = world.world().store().expect("world store");
            assert!(store.read_lane::<ornis_editor::EditorOnly>().is_some());
            assert!(store.read_lane::<ornis_editor::Selected>().is_some());
            assert!(store.read_lane::<ornis_editor::Hovered>().is_some());
            assert!(
                world
                    .world
                    .engine()
                    .schedule()
                    .mermaid()
                    .contains("editor_maintain"),
                "skeleton system wired"
            );
            world.tick(1.0 / 60.0);
        }

        let mut world = EditorSession::new();
        world.spawn(Some("Hero".into()));
        world.load_scene(world.to_scene());
        assert_eq!(world.entity_count(), 1);
        assert!(
            world
                .world()
                .store()
                .expect("world store")
                .read_lane::<ornis_editor::EditorOnly>()
                .is_some(),
            "install survives load_scene replacement"
        );
    }

    /// Editor chrome replicates through the snapshot (with marks) but never
    /// persists: save/load round-trips contain only scene entities, while
    /// `scene_json` labels both. Handles stay stable throughout.
    #[test]
    fn editor_entities_excluded_from_save_but_visible_in_snapshot() {
        let (mut world, dir) = sandboxed_session("editor-filter");
        let hero = world.spawn(Some("Hero".into()));
        let gizmo = world
            .world_mut()
            .store_mut()
            .expect("store")
            .create_entity();
        {
            let store = world.world_mut().store_mut().expect("store");
            store.insert(gizmo, Name("Gizmo".into()));
            store.insert(gizmo, default_transform());
            store.insert(gizmo, default_mesh());
            store.insert(gizmo, default_material());
            store.insert(gizmo, ornis_editor::EditorOnly);
            store.insert(gizmo, ornis_editor::Selected);
        }
        world.alive.push(gizmo);
        assert_eq!(world.entity_count(), 2);

        // Snapshot: both entities, editor labels always present.
        let scene: Value = serde_json::from_str(&world.scene_json()).unwrap();
        assert_eq!(scene["entity_count"], 2);
        let entities = scene["entities"].as_array().unwrap();
        assert_eq!(entities.len(), 2);
        let by_id = |id: u32| {
            entities
                .iter()
                .find(|e| e["id"] == id)
                .expect("entity in snapshot")
        };
        assert_eq!(
            by_id(hero.id())["editor"],
            serde_json::json!({
                "editor_only": false, "selected": false, "hovered": false,
            })
        );
        assert_eq!(
            by_id(gizmo.id())["editor"],
            serde_json::json!({
                "editor_only": true, "selected": true, "hovered": false,
            })
        );
        assert_eq!(by_id(gizmo.id())["components"]["Name"], "Gizmo");

        // Save path: chrome filtered out.
        assert_eq!(world.to_scene().entities.len(), 1);
        assert_eq!(world.to_scene().entities[0].name, "Hero");
        let path = world.scene_path("editor/filter.ron").expect("sandboxed");
        world.save_scene_file(&path).expect("save");
        let on_disk =
            Scene::from_ron(&fs::read_to_string(path.as_path()).unwrap()).expect("valid RON");
        assert_eq!(on_disk.entities.len(), 1);
        assert_eq!(on_disk.entities[0].name, "Hero");

        // Load path: only scene entities come back; handles restart at zero.
        let mut restored = EditorSession::with_roots(SceneRoots::new(&dir));
        assert_eq!(restored.load_scene_file(&path).expect("load"), 1);
        assert_eq!(restored.entity_count(), 1);
        assert_eq!(restored.name_of(restored.alive[0]).as_deref(), Some("Hero"));
        // The live world (and its handles) is untouched by save/load.
        assert_eq!(world.entity_count(), 2);
        assert_eq!(world.name_of(hero).as_deref(), Some("Hero"));
        assert_eq!(world.name_of(gizmo).as_deref(), Some("Gizmo"));
        let _ = fs::remove_dir_all(&dir);
    }

    /// `EditorOnly` bodies never reach the solver — neither at bind time
    /// nor when marked after binding (the binding is evicted on next sync,
    /// the pose freezes instead of simulating once more).
    #[test]
    fn editor_only_bodies_never_reach_solver() {
        let mut world = EditorSession::new();
        let plain = world.spawn(None);
        world
            .world_mut()
            .store_mut()
            .expect("store")
            .insert(plain, RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0));
        let gizmo = world
            .world_mut()
            .store_mut()
            .expect("store")
            .create_entity();
        {
            let store = world.world_mut().store_mut().expect("store");
            store.insert(gizmo, Name("Gizmo".into()));
            store.insert(gizmo, default_transform());
            store.insert(gizmo, ornis_editor::EditorOnly);
            store.insert(gizmo, RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0));
        }
        world.alive.push(gizmo);

        world.tick(1.0 / 60.0);
        let lane = |world: &EditorSession| {
            world
                .store()
                .and_then(|store| read_component::<TransformDesc>(store, plain))
                .expect("plain transform")
        };
        assert!(
            lane(&world).translation[1] < 0.0,
            "scene body falls under gravity"
        );
        let gizmo_pose: TransformDesc = world
            .store()
            .and_then(|store| read_component(store, gizmo))
            .expect("gizmo transform");
        assert_eq!(
            gizmo_pose.translation.to_array(),
            [0.0, 0.0, 0.0],
            "chrome body never bound, pose untouched"
        );

        // Late marking evicts the live binding: the pose freezes.
        world
            .world_mut()
            .store_mut()
            .expect("store")
            .insert(plain, ornis_editor::EditorOnly);
        let frozen = lane(&world).translation[1];
        world.tick(1.0 / 60.0);
        assert_eq!(
            lane(&world).translation[1],
            frozen,
            "evicted body no longer simulates"
        );
    }

    // ── glTF animation wiring ────────────────────────────────────────────

    /// Cold object clips live here in tests (the lib import omits the name:
    /// only the cold lane — registered by `install_object_animation` — holds
    /// it at runtime).
    use ornis_animation::{AnimClip, AnimPlayer, try_animator};

    /// Render scene matching [`animated_loaded`] 1:1 (two mesh entities).
    fn animated_scene() -> Scene {
        fn desc(name: &str) -> EntityDesc {
            EntityDesc {
                name: name.into(),
                transform: default_transform(),
                mesh: MeshDesc::Custom {
                    positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    indices: vec![0, 1, 2],
                },
                material: default_material(),
            }
        }
        Scene {
            name: "anim-fixture".into(),
            entities: vec![desc("skinned"), desc("plain")],
            lights: Vec::new(),
            camera: CameraDesc {
                position: glam::Vec3::new(0.0, 2.5, 9.0),
                target: glam::Vec3::ZERO,
                up: ornis_core::units::UnitVec3::Y,
                fov: ornis_core::units::Degrees::new(60.0),
                near: ornis_core::units::Meters::new(0.1),
                far: ornis_core::units::Meters::new(100.0),
                projection: CameraProjection::Perspective,
            },
            ambient: [0.1, 0.1, 0.1],
        }
    }

    /// Hand-built animated source: node 0 skinned (single-joint skin), node
    /// 1 plain; one skeletal clip plus one object clip carrying an extra
    /// unmapped track (node 99) that the converter must drop.
    fn animated_loaded() -> ornis_gltf::Model {
        use ornis_core::Transform;
        use ornis_gltf::{
            LoadedAnimClip, LoadedAnimTrack, LoadedJointTrack, LoadedKey, LoadedKeyTrack,
            LoadedMaterial, LoadedMesh, LoadedSkelClip, LoadedSkin, ModelNode, ModelPrimitive,
            NodeIdx,
        };
        fn mesh(skinned: bool) -> LoadedMesh {
            LoadedMesh {
                positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                indices: vec![0, 1, 2],
                normals: None,
                uvs: None,
                joints: skinned.then(|| vec![[0, 0, 0, 0]; 3]),
                weights: skinned.then(|| vec![[1.0, 0.0, 0.0, 0.0]; 3]),
            }
        }
        fn material() -> LoadedMaterial {
            LoadedMaterial {
                base_color: [0.8, 0.2, 0.2],
                metallic: 0.0,
                roughness: 0.5,
                emission: [0.0, 0.0, 0.0],
                base_color_texture: None,
                metallic_roughness_texture: None,
                emissive_texture: None,
            }
        }
        fn node(name: &str, primitives: Vec<usize>, skin: Option<usize>) -> ModelNode {
            ModelNode {
                name: Some(name.into()),
                parent: None,
                local: Transform::IDENTITY,
                primitives,
                skin,
            }
        }
        ornis_gltf::Model {
            name: "anim-fixture".into(),
            nodes: vec![
                node("skinned", vec![0], Some(0)),
                node("plain", vec![1], None),
            ],
            roots: vec![NodeIdx(0), NodeIdx(1)],
            primitives: vec![
                ModelPrimitive {
                    node: NodeIdx(0),
                    mesh: mesh(true),
                    material: material(),
                },
                ModelPrimitive {
                    node: NodeIdx(1),
                    mesh: mesh(false),
                    material: material(),
                },
            ],
            skins: vec![LoadedSkin {
                parents: vec![-1],
                inverse_bind: vec![[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]],
                joint_names: vec!["j0".into()],
            }],
            skel_clips: vec![LoadedSkelClip {
                name: String::new(),
                duration: 1.0,
                tracks: vec![LoadedJointTrack {
                    joint: 0,
                    node: NodeIdx(0),
                    translation: LoadedKeyTrack::linear(vec![
                        LoadedKey {
                            time: 0.0,
                            value: [0.0, 0.0, 0.0],
                        },
                        LoadedKey {
                            time: 1.0,
                            value: [1.0, 0.0, 0.0],
                        },
                    ]),
                    rotation: LoadedKeyTrack::linear(Vec::new()),
                    scale: LoadedKeyTrack::linear(Vec::new()),
                }],
            }],
            anim_clips: vec![LoadedAnimClip {
                name: "walk".into(),
                duration: 2.0,
                looping: true,
                tracks: vec![
                    LoadedAnimTrack {
                        node: NodeIdx(1),
                        translation: LoadedKeyTrack::linear(vec![LoadedKey {
                            time: 0.0,
                            value: [0.0, 0.0, 0.0],
                        }]),
                        rotation: LoadedKeyTrack::linear(Vec::new()),
                        scale: LoadedKeyTrack::linear(Vec::new()),
                    },
                    LoadedAnimTrack {
                        node: NodeIdx(99),
                        translation: LoadedKeyTrack::linear(vec![LoadedKey {
                            time: 0.0,
                            value: [5.0, 5.0, 5.0],
                        }]),
                        rotation: LoadedKeyTrack::linear(Vec::new()),
                        scale: LoadedKeyTrack::linear(Vec::new()),
                    },
                ],
            }],
            stats: ornis_gltf::ImportStats::default(),
        }
    }

    /// Skeletal wiring indexes the empty-name clip as `skel_clip_0` on the
    /// first mesh and leaves the skeleton root without a cursor. Object
    /// players are inserted paused on the mapped entity (the unmapped
    /// track drops), and a `SkinnedMesh` points at the root.
    #[test]
    fn gltf_animation_wiring_indexes_clips_without_a_skeletal_player() {
        let mut world = EditorSession::new();
        let loaded = animated_loaded();
        assert_eq!(world.load_scene(animated_scene()), 2);
        let version_before = world.version;
        assert_eq!(world.wire_gltf_animation(&loaded), 3);
        assert!(world.version > version_before);
        assert_eq!(world.entity_count(), 5);

        let store = world.world().store().expect("world store");
        let skeletons = store.read_lane::<Skeleton>().expect("skeleton lane");
        assert_eq!(skeletons.len(), 1);
        let root = skeletons.entities[0];
        drop(skeletons);

        let poses = store.read_lane::<JointPose>().expect("pose lane");
        assert_eq!(
            poses.get(root).expect("root pose").matrices.len(),
            1,
            "pose length == joint count"
        );
        drop(poses);

        let skel_clips = store.read_cold_lane::<SkelClip>().expect("skel cold lane");
        assert_eq!(skel_clips.len(), 1);
        let anim_clips = store.read_cold_lane::<AnimClip>().expect("anim cold lane");
        assert_eq!(anim_clips.len(), 1);

        let hero = world.alive[0];
        let playlist = store
            .read_lane::<Animator>()
            .expect("animator lane")
            .get(hero)
            .expect("animator on the first mesh")
            .clip("skel_clip_0")
            .expect("empty glTF name");
        assert!(
            skel_clips.get(playlist.0).is_some(),
            "animator points at the skeletal playlist"
        );
        assert!(
            store
                .read_lane::<SkelPlayer>()
                .is_none_or(|lane| lane.get(root).is_none()),
            "skeletal cursors stay empty until play"
        );

        let skinned = store.read_lane::<SkinnedMesh>().expect("skinned lane");
        assert_eq!(
            skinned.get(world.alive[0]).expect("skinned mesh").skeleton,
            root
        );
        assert!(skinned.get(world.alive[1]).is_none());
        drop(skinned);

        let animated = world.alive[1];
        let object_players = store.read_lane::<AnimPlayer>().expect("anim players");
        let object = object_players.get(animated).expect("object player");
        assert!(!object.playing, "object players stay paused");
        let playlist = anim_clips.get(object.clip.0).expect("object playlist");
        assert_eq!(playlist.tracks.len(), 1, "unmapped track dropped");
        assert_eq!(playlist.tracks[0].entity, animated);
    }

    /// A source without skins or clips wires nothing: the `.ron` shape is
    /// untouched and no animation lanes are populated.
    #[test]
    fn gltf_wiring_without_animation_adds_nothing() {
        let mut world = EditorSession::new();
        assert_eq!(world.load_scene(animated_scene()), 2);
        let mut bare = animated_loaded();
        bare.skins.clear();
        bare.skel_clips.clear();
        bare.anim_clips.clear();
        for node in &mut bare.nodes {
            node.skin = None;
        }
        assert_eq!(world.wire_gltf_animation(&bare), 0);
        assert_eq!(world.entity_count(), 2);

        let store = world.world().store().expect("world store");
        assert!(
            store
                .read_cold_lane::<SkelClip>()
                .is_none_or(|lane| lane.is_empty())
        );
        assert!(
            store
                .read_cold_lane::<AnimClip>()
                .is_none_or(|lane| lane.is_empty())
        );
        assert!(
            store
                .read_lane::<SkelPlayer>()
                .is_none_or(|lane| lane.is_empty())
        );
        assert!(
            store
                .read_lane::<AnimPlayer>()
                .is_none_or(|lane| lane.is_empty())
        );
        assert!(
            store
                .read_lane::<SkinnedMesh>()
                .is_none_or(|lane| lane.is_empty())
        );
    }

    // Starter-content playback: the vendored no-root-motion UAL pack
    // plays `Walk_Loop` by name (load → named play → joints move).
    // Read-only over the committed fixture; nothing is written back.
    #[test]
    fn starter_pack_playback_moves_joints() {
        let mut session = EditorSession::new();
        let path = session
            .scene_path("assets/starter/ual1_standard.glb")
            .unwrap();
        let count = session.load_scene_file(&path).unwrap();
        assert!(count > 0, "nothing spawned");

        let hero = session.alive[0];
        {
            let store = session.store_mut().expect("store");
            try_animator(store, hero)
                .expect("animator on the first mesh")
                .play("Walk_Loop")
                .expect("Walk_Loop");
        }

        let store = session.world().store().unwrap();
        let players = store.read_lane::<SkelPlayer>().unwrap();
        assert!(!players.entities.is_empty(), "play inserts cursors");
        assert!(
            players
                .entities
                .iter()
                .all(|e| players.get(*e).is_some_and(|p| p.playing)),
            "Walk_Loop is playing"
        );
        let clip = players.get(players.entities[0]).expect("cursor").clip;
        assert_eq!(
            store
                .read_cold_lane::<SkelClip>()
                .unwrap()
                .get(clip.0)
                .expect("playlist")
                .name,
            "Walk_Loop"
        );
        let before = store
            .read_lane::<JointPose>()
            .unwrap()
            .entities
            .first()
            .and_then(|e| {
                store
                    .read_lane::<JointPose>()
                    .unwrap()
                    .get(*e)
                    .map(|p| p.matrices.clone())
            })
            .expect("joint pose");
        drop(players);

        for _ in 0..30 {
            session.tick_secs(Seconds::new(1.0 / 60.0));
        }
        let store = session.world().store().unwrap();
        let after = store
            .read_lane::<JointPose>()
            .unwrap()
            .entities
            .first()
            .and_then(|e| {
                store
                    .read_lane::<JointPose>()
                    .unwrap()
                    .get(*e)
                    .map(|p| p.matrices.clone())
            })
            .expect("joint pose");
        assert_eq!(before.len(), after.len());
        assert!(before != after, "30 frames ran but no joint moved");
    }
}
