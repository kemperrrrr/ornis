//! Server-side ECS world for `editor-only` mode.
//!
//! In `editor-only` there is no native winit loop to consume `UiCommand`s,
//! so [`run`] spawns an `editor-world` thread that owns an [`EditorWorld`]
//! (ornis-core `SmartStore` + the component registry), executes commands
//! from `POST /api/command` and publishes `GameEvent`s back to the HTTP
//! server (`status`/`scene` snapshots are cached by `remote.rs` for
//! `GET /api/status` and `GET /api/scene`; the rest reach `GET /api/events`).
//!
//! At startup the world loads `editor/scene.ron` (via
//! `ornis_render::scene::Scene::from_ron`), so the live world matches what
//! the WASM viewport renders statically. Component payloads reuse the
//! `ornis_render::scene` description types — **serde-canonical** JSON
//! (externally-tagged enums), served generically through the component
//! registry (F0, audit §10 D2). The JSON contract of
//! [`EditorWorld::scene_json`] is:
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
//!     }
//!   }],
//!   "lights": [{"Directional": {"direction": [1, 1, 1], "intensity": 0.6, "color": [1, 1, 1]}}],
//!   "camera": {"position": [0, 2.5, 9], "target": [0, 0, 0], "up": [0, 1, 0], "fov": 60.0, "near": 0.1, "far": 100.0},
//!   "ambient": [0.10, 0.10, 0.15]
//! }
//! ```
//!
//! Commands (`POST /api/command`, body `{"type": …, "data": …}`):
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
//!   emitting `scene_saved {path, version}`. The world is not mutated;
//! * `load_scene` — `{"path"?: string}`; replaces the world with the scene
//!   read back from `path`, emitting `scene_loaded {path, version,
//!   entity_count}` plus fresh `status`/`scene` snapshots. A missing or
//!   malformed file emits `error` and leaves the world untouched.
//!
//! `version` is incremented on every mutation so clients can cheaply detect
//! changes. Invalid commands never panic: they produce an `error` event and
//! leave the world (and its version) untouched.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use ornis_core::{ComponentMeta, ComponentRegistry, Entity, SmartStore};
use ornis_render::scene::{
    CameraDesc, EntityDesc, LightDesc, MaterialDesc, MeshDesc, Scene, TransformDesc,
};

use editor_backend::ipc::{GameEvent, UiCommand};

/// Editor-side name component attached to every spawned entity.
/// Newtype over `String`: its serde-canonical JSON is a plain string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Name(pub String);

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
    registry
});

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
                position: [0.0, 2.5, 9.0],
                target: [0.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
                fov: 60.0,
                near: 0.1,
                far: 100.0,
            },
            ambient: [0.10, 0.10, 0.15],
        }
    }
}

/// Live renderable scene: alive entities plus Name/Transform/Mesh/Material
/// components, the environment resource, the scene label and a mutation
/// version counter.
pub struct EditorWorld {
    store: SmartStore,
    alive: Vec<Entity>,
    environment: SceneEnvironment,
    /// Scene label round-tripped through `Scene::name` on save/load.
    scene_name: String,
    version: u64,
}

impl Default for EditorWorld {
    fn default() -> Self {
        Self {
            store: SmartStore::default(),
            alive: Vec::new(),
            environment: SceneEnvironment::default(),
            scene_name: "scene".into(),
            version: 0,
        }
    }
}

impl EditorWorld {
    /// An empty world with the default environment (default camera,
    /// no lights).
    pub fn new() -> Self {
        Self::default()
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
        let entity = self.store.create_entity();
        self.alive.push(entity);
        let name = name.unwrap_or_else(|| format!("Entity {}", entity.id()));
        self.store.insert(entity, Name(name));
        self.store.insert(entity, transform);
        self.store.insert(entity, mesh);
        self.store.insert(entity, material);
        self.version += 1;
        entity
    }

    /// Despawn by id/generation. Returns the entity if it was alive.
    pub fn despawn(&mut self, id: u32, generation: u32) -> Option<Entity> {
        let entity = Entity::new_with_gen(id, generation);
        if !self.store.is_alive(entity) {
            return None;
        }
        self.alive.retain(|e| *e != entity);
        self.store.destroy_entity(entity);
        self.version += 1;
        Some(entity)
    }

    /// Display name of `entity`, if alive and named.
    pub fn name_of(&self, entity: Entity) -> Option<String> {
        self.store
            .read_lane::<Name>()
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
        let mut fresh = EditorWorld::new();
        for e in scene.entities {
            fresh.spawn_with(Some(e.name), e.transform, e.mesh, e.material);
        }
        fresh.environment = SceneEnvironment {
            lights: scene.lights,
            camera: scene.camera,
            ambient: scene.ambient,
        };
        fresh.scene_name = scene.name;
        fresh.version = fresh.version.max(self.version + 1);
        *self = fresh;
        count
    }

    /// Parse a RON scene and load it (replacing the world, see
    /// [`EditorWorld::load_scene`]). An invalid RON string leaves the world
    /// untouched.
    pub fn load_scene_ron(&mut self, ron_str: &str) -> Result<usize, String> {
        let scene = Scene::from_ron(ron_str).map_err(|e| format!("invalid scene RON: {e}"))?;
        Ok(self.load_scene(scene))
    }

    /// Serialize the world to RON and write it to `path` **atomically**
    /// (sibling `*.tmp` file + rename): a crash or I/O error mid-write can
    /// never leave a truncated scene file behind. The world is not mutated.
    pub fn save_scene_file(&self, path: &Path) -> Result<(), String> {
        let ron = self
            .to_scene()
            .to_ron()
            .map_err(|e| format!("scene RON serialization: {e}"))?;
        atomic_write(path, &ron)
    }

    /// Read `path` and replace the world with its scene. Any error (missing
    /// file, invalid RON) leaves the world untouched.
    pub fn load_scene_file(&mut self, path: &Path) -> Result<usize, String> {
        let ron = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        self.load_scene_ron(&ron)
    }

    /// JSON snapshot for `GET /api/scene` (see the module docs for the contract).
    pub fn scene_json(&self) -> String {
        scene_json(self)
    }

    /// JSON payload for `GET /api/status` (cached by the HTTP server).
    pub fn status_json(&self) -> String {
        status_json(self)
    }

    /// Execute one command from the HTTP server, emitting the corresponding
    /// events (`entity_created`/`entity_destroyed`/`entity_list`,
    /// `ComponentUpdated`, `status`/`scene` snapshots or `error`). Invalid
    /// commands never mutate the world — they produce an `error` event.
    pub fn handle_command(&mut self, cmd: &UiCommand, ev_tx: &Sender<GameEvent>) {
        handle_command(self, cmd, ev_tx);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Snapshots and command handling — free functions over [`EditorWorld`]
// ═══════════════════════════════════════════════════════════════════════════
// The bulk of the snapshot/command logic lives here, not in
// `impl EditorWorld`, to keep the type under the bca number-of-methods
// gate; the public methods above are thin delegates.

/// Snapshot `world` as a [`Scene`]: every alive entity becomes an
/// [`EntityDesc`] (missing components fall back to the spawn defaults),
/// lights/camera/ambient come from the environment resource.
fn to_scene(world: &EditorWorld) -> Scene {
    let entities = world.alive.iter().map(|&e| entity_desc(world, e)).collect();
    Scene {
        name: world.scene_name.clone(),
        entities,
        lights: world.environment.lights.clone(),
        camera: world.environment.camera.clone(),
        ambient: world.environment.ambient,
    }
}

/// One alive entity as an [`EntityDesc`] for [`to_scene`].
fn entity_desc(world: &EditorWorld, entity: Entity) -> EntityDesc {
    EntityDesc {
        name: world
            .name_of(entity)
            .unwrap_or_else(|| format!("Entity {}", entity.id())),
        transform: read_component(&world.store, entity).unwrap_or_else(default_transform),
        mesh: read_component(&world.store, entity).unwrap_or_else(default_mesh),
        material: read_component(&world.store, entity).unwrap_or_else(default_material),
    }
}

/// JSON snapshot for `GET /api/scene` (see the module docs for the contract).
fn scene_json(world: &EditorWorld) -> String {
    let entities: Vec<Value> = world
        .alive
        .iter()
        .map(|&e| entity_json(&world.store, e))
        .collect();
    let lights = serde_json::to_value(&world.environment.lights).expect("LightDesc serializes");
    let camera = serde_json::to_value(&world.environment.camera).expect("CameraDesc serializes");
    serde_json::json!({
        "version": world.version,
        "entity_count": world.entity_count(),
        "entities": entities,
        "lights": lights,
        "camera": camera,
        "ambient": world.environment.ambient,
    })
    .to_string()
}

/// JSON payload for `GET /api/status` (cached by the HTTP server).
fn status_json(world: &EditorWorld) -> String {
    serde_json::json!({
        "entity_count": world.entity_count(),
        "name": "Ornis Engine",
        "version": world.version,
    })
    .to_string()
}

/// Publish `status` + `scene` snapshots so the HTTP server's caches
/// (`GET /api/status`, `GET /api/scene`) reflect the current world.
fn publish_state(world: &EditorWorld, ev_tx: &Sender<GameEvent>) {
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

/// Execute one command from the HTTP server, emitting the corresponding
/// events (`entity_created`/`entity_destroyed`/`entity_list`,
/// `ComponentUpdated`, `status`/`scene` snapshots or `error`). Invalid
/// commands never mutate the world — they produce an `error` event.
fn handle_command(world: &mut EditorWorld, cmd: &UiCommand, ev_tx: &Sender<GameEvent>) {
    match cmd {
        UiCommand::CreateEntity => {
            world.spawn(None);
            publish_state(world, ev_tx);
        }
        UiCommand::DestroyEntity { entity_id } => {
            // The typed variant carries no generation; match any alive
            // entity with this id.
            if let Ok(entity) = resolve_alive(&world.alive, *entity_id, None) {
                world.despawn(entity.id(), entity.generation());
                publish_state(world, ev_tx);
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
    }
}

/// Typed `SetComponent` (remote maps the `set_component` POST here):
/// generic upsert through the component registry. Success emits the
/// typed `ComponentUpdated` event and publishes fresh snapshots; any
/// error (unknown entity/component, malformed JSON) is an `error`
/// event with the world left untouched.
fn handle_set_component(
    world: &mut EditorWorld,
    entity_id: u32,
    generation: Option<u32>,
    type_name: &str,
    json_data: &str,
    ev_tx: &Sender<GameEvent>,
) {
    match set_component(world, entity_id, generation, type_name, json_data) {
        Ok(value) => {
            ev_tx
                .send(GameEvent::ComponentUpdated {
                    entity_id,
                    type_name: type_name.into(),
                    json_data: value.to_string(),
                })
                .ok();
            publish_state(world, ev_tx);
        }
        Err(e) => emit_error(ev_tx, "set_component", &e),
    }
}

/// Validate and apply the upsert; returns the applied payload.
fn set_component(
    world: &mut EditorWorld,
    entity_id: u32,
    generation: Option<u32>,
    type_name: &str,
    json_data: &str,
) -> Result<Value, String> {
    let entity = resolve_alive(&world.alive, entity_id, generation)?;
    let meta = REGISTRY
        .by_name(type_name)
        .ok_or_else(|| format!("unknown component '{type_name}'"))?;
    let value: Value = serde_json::from_str(json_data).map_err(|e| format!("invalid JSON: {e}"))?;
    meta.set_json(&mut world.store, entity, &value)
        .map_err(|e| e.to_string())?;
    world.version += 1;
    Ok(value)
}

fn handle_custom(
    world: &mut EditorWorld,
    cmd_type: &str,
    json_data: &str,
    ev_tx: &Sender<GameEvent>,
) {
    let data = match parse_data(json_data) {
        Ok(data) => data,
        Err(e) => {
            emit_error(ev_tx, cmd_type, &e);
            return;
        }
    };
    match cmd_type {
        "create_entity" => match cmd_create_entity(world, &data) {
            Ok(payload) => {
                emit(ev_tx, "entity_created", payload);
                publish_state(world, ev_tx);
            }
            Err(e) => emit_error(ev_tx, cmd_type, &e),
        },
        "destroy_entity" => match cmd_destroy_entity(world, &data) {
            Ok(payload) => {
                emit(ev_tx, "entity_destroyed", payload);
                publish_state(world, ev_tx);
            }
            Err(e) => emit_error(ev_tx, cmd_type, &e),
        },
        "list_entities" => {
            let payload = list_entities_json(world);
            emit(ev_tx, "entity_list", payload);
        }
        "save_scene" => {
            let path = command_path(&data);
            match world.save_scene_file(&path) {
                Ok(()) => emit(
                    ev_tx,
                    "scene_saved",
                    serde_json::json!({
                        "path": path.display().to_string(),
                        "version": world.version,
                    })
                    .to_string(),
                ),
                Err(e) => emit_error(ev_tx, cmd_type, &e),
            }
        }
        "load_scene" => {
            let path = command_path(&data);
            match world.load_scene_file(&path) {
                Ok(count) => {
                    emit(
                        ev_tx,
                        "scene_loaded",
                        serde_json::json!({
                            "path": path.display().to_string(),
                            "version": world.version,
                            "entity_count": count,
                        })
                        .to_string(),
                    );
                    publish_state(world, ev_tx);
                }
                Err(e) => emit_error(ev_tx, cmd_type, &e),
            }
        }
        other => emit_error(ev_tx, other, "unknown command"),
    }
}

fn cmd_create_entity(world: &mut EditorWorld, data: &Value) -> Result<String, String> {
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
    for (meta, boxed) in overrides {
        // Parsed from the same meta — the box type always matches.
        meta.insert_any(&mut world.store, entity, boxed);
    }
    Ok(serde_json::json!({
        "id": entity.id(),
        "generation": entity.generation(),
        "name": world.name_of(entity),
    })
    .to_string())
}

fn cmd_destroy_entity(world: &mut EditorWorld, data: &Value) -> Result<String, String> {
    let entity = resolve_entity(world, data)?;
    world.despawn(entity.id(), entity.generation());
    Ok(serde_json::json!({"id": entity.id(), "generation": entity.generation()}).to_string())
}

/// Validate `id` + `generation` against the store's allocator.
fn resolve_entity(world: &EditorWorld, data: &Value) -> Result<Entity, String> {
    let id = data
        .get("id")
        .and_then(Value::as_u64)
        .ok_or("missing or invalid 'id'")? as u32;
    let generation = data
        .get("generation")
        .and_then(Value::as_u64)
        .ok_or("missing or invalid 'generation'")? as u32;
    let entity = Entity::new_with_gen(id, generation);
    if !world.store.is_alive(entity) {
        return Err(format!("entity {id}:{generation} not found"));
    }
    Ok(entity)
}

/// One entity entry: `id`/`generation` plus a map
/// «registry name → serde-canonical component JSON» — generic over the
/// registered component set (registry ops, no per-type code).
fn entity_json(store: &SmartStore, entity: Entity) -> Value {
    let mut components = serde_json::Map::new();
    for meta in REGISTRY.iter() {
        // A serialization error is unreachable for plain data structs.
        if let Ok(Some(value)) = meta.get_json(store, entity) {
            components.insert(meta.name().to_string(), value);
        }
    }
    serde_json::json!({
        "id": entity.id(),
        "generation": entity.generation(),
        "components": components,
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
fn list_entities_json(world: &EditorWorld) -> String {
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

fn default_transform() -> TransformDesc {
    TransformDesc {
        translation: [0.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }
}

fn default_mesh() -> MeshDesc {
    MeshDesc::Sphere {
        radius: 1.0,
        segments: 32,
        rings: 24,
    }
}

fn default_material() -> MaterialDesc {
    MaterialDesc::Dielectric {
        base_color: [0.5, 0.5, 0.5],
        roughness: 0.5,
    }
}

/// Scene path override from a `save_scene`/`load_scene` payload
/// (`{"path": "…"}`); defaults to [`scene_file_path`]. A non-string `path`
/// is ignored (falls back to the default) like any other soft payload flaw.
fn command_path(data: &Value) -> PathBuf {
    data.get("path")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .unwrap_or_else(scene_file_path)
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

/// Default scene file for the `save_scene`/`load_scene` commands:
/// `editor/scene.ron` — the scene the WASM viewport renders at startup
/// (CARGO_MANIFEST_DIR for the `ornis` binary points at the workspace root).
fn scene_file_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("editor/scene.ron")
}

/// Write `contents` to `path` atomically: a sibling `<name>.tmp` file is
/// written first and renamed over the target, so a failed write leaves the
/// previous scene file intact (or no file at all).
fn atomic_write(path: &Path, contents: &str) -> Result<(), String> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    fs::write(&tmp, contents).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|e| format!("rename to {}: {e}", path.display()))
}

/// Startup scene RON: `editor/scene.ron` (what the WASM viewport renders
/// statically), falling back to `assets/scene.ron`.
fn startup_scene_ron() -> Option<String> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    ["editor/scene.ron", "assets/scene.ron"]
        .iter()
        .find_map(|rel| fs::read_to_string(manifest.join(rel)).ok())
}

/// Spawn the `editor-world` thread: owns the world, loads the startup scene,
/// blocks on `cmd_rx`, executes commands until the HTTP server side drops
/// its sender.
pub fn run(cmd_rx: Receiver<UiCommand>, ev_tx: Sender<GameEvent>) -> JoinHandle<()> {
    thread::Builder::new()
        .name("editor-world".into())
        .spawn(move || {
            let mut world = EditorWorld::new();
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
            while let Ok(cmd) = cmd_rx.recv() {
                world.handle_command(&cmd, &ev_tx);
            }
        })
        .expect("spawn editor-world thread")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::unbounded;

    fn world_and_events() -> (EditorWorld, Sender<GameEvent>, Receiver<GameEvent>) {
        let (ev_tx, ev_rx) = unbounded();
        (EditorWorld::new(), ev_tx, ev_rx)
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
                } if t == cmd_type => Some(json_data.clone()),
                _ => None,
            })
            .collect()
    }

    /// Typed `ComponentUpdated` events as (entity_id, type_name, json_data).
    fn component_updates(events: &[GameEvent]) -> Vec<(u32, String, String)> {
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
    fn spawn_assigns_names_and_counts() {
        let mut world = EditorWorld::new();
        let a = world.spawn(None);
        let b = world.spawn(Some("Hero".into()));
        assert_eq!(world.entity_count(), 2);
        assert_eq!(world.name_of(a).as_deref(), Some("Entity 0"));
        assert_eq!(world.name_of(b).as_deref(), Some("Hero"));
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn despawn_recycles_ids_with_new_generation() {
        let mut world = EditorWorld::new();
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
        let ron =
            fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("editor/scene.ron"))
                .expect("editor/scene.ron readable");
        let mut world = EditorWorld::new();
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
        assert_f32(&red_components["Material"]["Dielectric"]["roughness"], 0.5);

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
        assert_f32_seq(&lights[0]["Directional"]["direction"], &[1.0, 1.0, 1.0]);
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
        let mut world = EditorWorld::new();
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
            &[0.5, 0.5, 0.5],
        );
        assert_eq!(entities[1]["components"]["Name"], "Hero");
    }

    #[test]
    fn scene_json_empty_world() {
        let world = EditorWorld::new();
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
        world.handle_command(&set_component(0, Some(0), "Collider", "{}"), &ev_tx);
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
        assert_eq!(world.version, version + 1);

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
    fn scene_value(world: &EditorWorld) -> Value {
        let mut v: Value = serde_json::from_str(&world.scene_json()).unwrap();
        v.as_object_mut().unwrap().remove("version");
        v
    }

    /// Fresh temp dir per test (removed first, so no stale state).
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn to_scene_round_trip_through_ron_preserves_world() {
        let ron =
            fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("editor/scene.ron"))
                .expect("editor/scene.ron readable");
        let mut world = EditorWorld::new();
        world.load_scene_ron(&ron).expect("scene loads");
        // A runtime-created entity must round-trip too.
        world.spawn_with(
            Some("Extra".into()),
            TransformDesc {
                translation: [9.0, 1.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [2.0, 2.0, 2.0],
            },
            MeshDesc::Sphere {
                radius: 0.5,
                segments: 8,
                rings: 4,
            },
            MaterialDesc::Metal {
                base_color: [1.0, 0.0, 0.0],
                roughness: 0.3,
            },
        );

        let serialized = world.to_scene().to_ron().expect("serialize");
        let reparsed = Scene::from_ron(&serialized).expect("re-parse");

        let mut restored = EditorWorld::new();
        let loaded = restored.load_scene(reparsed);
        assert_eq!(loaded, 6);
        assert_eq!(scene_value(&restored), scene_value(&world));
        // Version: max(loaded entity count, old version + 1) — here the
        // 6 spawns dominate over the fresh world's `0 + 1`.
        assert_eq!(restored.version, world.version);
    }

    #[test]
    fn save_and_load_file_round_trip() {
        let dir = temp_dir("ornis_editor_world_save_load");
        let path = dir.join("scene.ron");

        let mut world = EditorWorld::new();
        world.spawn(Some("Hero".into()));
        world.save_scene_file(&path).expect("save");

        // The file on disk is a valid scene with the world's content.
        let on_disk = Scene::from_ron(&fs::read_to_string(&path).unwrap()).expect("valid RON");
        assert_eq!(on_disk.entities.len(), 1);
        assert_eq!(on_disk.entities[0].name, "Hero");

        let mut restored = EditorWorld::new();
        let loaded = restored.load_scene_file(&path).expect("load");
        assert_eq!(loaded, 1);
        assert_eq!(scene_value(&restored), scene_value(&world));
        // The temp file was renamed away — no litter.
        assert!(!dir.join("scene.ron.tmp").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_failure_never_leaves_partial_files() {
        // The target directory does not exist: the write must fail cleanly.
        let dir = std::env::temp_dir().join("ornis_editor_world_save_fail");
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("scene.ron");

        let world = EditorWorld::new();
        assert!(world.save_scene_file(&path).is_err());
        assert!(!path.exists(), "no partial scene file");
        assert!(!dir.join("scene.ron.tmp").exists(), "no temp file left");
    }

    #[test]
    fn load_broken_or_missing_file_keeps_world() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", r#"{"name":"Keep"}"#), &ev_tx);
        while ev_rx.try_recv().is_ok() {}
        let before = scene_value(&world);
        let version = world.version;

        let dir = temp_dir("ornis_editor_world_load_broken");
        let broken = dir.join("broken.ron");
        fs::write(&broken, "Scene(name: 42)").unwrap();

        let load =
            |path: &Path| custom("load_scene", &format!(r#"{{"path":"{}"}}"#, path.display()));
        world.handle_command(&load(&broken), &ev_tx);
        world.handle_command(&load(&dir.join("nope.ron")), &ev_tx);

        let events = drain_all(&ev_rx);
        assert_eq!(custom_events(&events, "error").len(), 2);
        assert_eq!(custom_events(&events, "scene_loaded").len(), 0);
        assert_eq!(scene_value(&world), before, "world untouched");
        assert_eq!(world.version, version, "version untouched");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_load_commands_emit_events_and_restore_state() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        let dir = temp_dir("ornis_editor_world_cmds");
        let path = dir.join("scene.ron");
        let arg = format!(r#"{{"path":"{}"}}"#, path.display());

        world.handle_command(&custom("create_entity", r#"{"name":"Hero"}"#), &ev_tx);
        world.handle_command(&custom("save_scene", &arg), &ev_tx);

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

        world.handle_command(&custom("load_scene", &arg), &ev_tx);
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
}
