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
//! * `list_entities` — no payload.
//!
//! `version` is incremented on every mutation so clients can cheaply detect
//! changes. Invalid commands never panic: they produce an `error` event and
//! leave the world (and its version) untouched.

use std::fs;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use ornis_core::{ComponentMeta, ComponentRegistry, Entity, SmartStore};
use ornis_render::scene::{CameraDesc, LightDesc, MaterialDesc, MeshDesc, Scene, TransformDesc};

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
    pub lights: Vec<LightDesc>,
    pub camera: CameraDesc,
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
/// components, the environment resource and a mutation version counter.
#[derive(Default)]
pub struct EditorWorld {
    store: SmartStore,
    alive: Vec<Entity>,
    environment: SceneEnvironment,
    version: u64,
}

impl EditorWorld {
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn name_of(&self, entity: Entity) -> Option<String> {
        self.store
            .read_lane::<Name>()
            .and_then(|lane| lane.get(entity).map(|name| name.0.clone()))
    }

    /// Load a RON scene into the world: each `EntityDesc` becomes an entity,
    /// lights/camera/ambient replace the environment resource.
    /// Returns the number of entities loaded.
    pub fn load_scene_ron(&mut self, ron_str: &str) -> Result<usize, String> {
        let scene = Scene::from_ron(ron_str).map_err(|e| format!("invalid scene RON: {e}"))?;
        let count = scene.entities.len();
        for e in scene.entities {
            self.spawn_with(Some(e.name), e.transform, e.mesh, e.material);
        }
        self.environment = SceneEnvironment {
            lights: scene.lights,
            camera: scene.camera,
            ambient: scene.ambient,
        };
        Ok(count)
    }

    /// JSON snapshot for `GET /api/scene` (see the module docs for the contract).
    pub fn scene_json(&self) -> String {
        let entities: Vec<Value> = self
            .alive
            .iter()
            .map(|&e| entity_json(&self.store, e))
            .collect();
        let lights = serde_json::to_value(&self.environment.lights).expect("LightDesc serializes");
        let camera = serde_json::to_value(&self.environment.camera).expect("CameraDesc serializes");
        serde_json::json!({
            "version": self.version,
            "entity_count": self.entity_count(),
            "entities": entities,
            "lights": lights,
            "camera": camera,
            "ambient": self.environment.ambient,
        })
        .to_string()
    }

    /// JSON payload for `GET /api/status` (cached by the HTTP server).
    pub fn status_json(&self) -> String {
        serde_json::json!({
            "entity_count": self.entity_count(),
            "name": "Ornis Engine",
            "version": self.version,
        })
        .to_string()
    }

    /// Publish `status` + `scene` snapshots so the HTTP server's caches
    /// (`GET /api/status`, `GET /api/scene`) reflect the current world.
    fn publish_state(&self, ev_tx: &Sender<GameEvent>) {
        self.emit(ev_tx, "status", self.status_json());
        self.emit(ev_tx, "scene", self.scene_json());
    }

    fn emit(&self, ev_tx: &Sender<GameEvent>, cmd_type: &str, payload: String) {
        ev_tx
            .send(GameEvent::CustomEvent {
                cmd_type: cmd_type.into(),
                json_data: payload,
            })
            .ok();
    }

    /// Invalid commands become `error` events instead of panics.
    fn emit_error(&self, ev_tx: &Sender<GameEvent>, command: &str, message: &str) {
        self.emit(
            ev_tx,
            "error",
            serde_json::json!({"command": command, "message": message}).to_string(),
        );
    }

    /// Execute one command, emitting the corresponding events.
    pub fn handle_command(&mut self, cmd: &UiCommand, ev_tx: &Sender<GameEvent>) {
        match cmd {
            UiCommand::CreateEntity => {
                self.spawn(None);
                self.publish_state(ev_tx);
            }
            UiCommand::DestroyEntity { entity_id } => {
                // The typed variant carries no generation; match any alive
                // entity with this id.
                if let Ok(entity) = resolve_alive(&self.alive, *entity_id, None) {
                    self.despawn(entity.id(), entity.generation());
                    self.publish_state(ev_tx);
                }
            }
            UiCommand::Custom {
                cmd_type,
                json_data,
            } => self.handle_custom(cmd_type, json_data, ev_tx),
            UiCommand::SetComponent {
                entity_id,
                generation,
                type_name,
                json_data,
            } => self.handle_set_component(*entity_id, *generation, type_name, json_data, ev_tx),
        }
    }

    /// Typed `SetComponent` (remote maps the `set_component` POST here):
    /// generic upsert through the component registry. Success emits the
    /// typed `ComponentUpdated` event and publishes fresh snapshots; any
    /// error (unknown entity/component, malformed JSON) is an `error`
    /// event with the world left untouched.
    fn handle_set_component(
        &mut self,
        entity_id: u32,
        generation: Option<u32>,
        type_name: &str,
        json_data: &str,
        ev_tx: &Sender<GameEvent>,
    ) {
        match self.set_component(entity_id, generation, type_name, json_data) {
            Ok(value) => {
                ev_tx
                    .send(GameEvent::ComponentUpdated {
                        entity_id,
                        type_name: type_name.into(),
                        json_data: value.to_string(),
                    })
                    .ok();
                self.publish_state(ev_tx);
            }
            Err(e) => self.emit_error(ev_tx, "set_component", &e),
        }
    }

    /// Validate and apply the upsert; returns the applied payload.
    fn set_component(
        &mut self,
        entity_id: u32,
        generation: Option<u32>,
        type_name: &str,
        json_data: &str,
    ) -> Result<Value, String> {
        let entity = resolve_alive(&self.alive, entity_id, generation)?;
        let meta = REGISTRY
            .by_name(type_name)
            .ok_or_else(|| format!("unknown component '{type_name}'"))?;
        let value: Value =
            serde_json::from_str(json_data).map_err(|e| format!("invalid JSON: {e}"))?;
        meta.set_json(&mut self.store, entity, &value)
            .map_err(|e| e.to_string())?;
        self.version += 1;
        Ok(value)
    }

    fn handle_custom(&mut self, cmd_type: &str, json_data: &str, ev_tx: &Sender<GameEvent>) {
        let data = match parse_data(json_data) {
            Ok(data) => data,
            Err(e) => {
                self.emit_error(ev_tx, cmd_type, &e);
                return;
            }
        };
        match cmd_type {
            "create_entity" => match self.cmd_create_entity(&data) {
                Ok(payload) => {
                    self.emit(ev_tx, "entity_created", payload);
                    self.publish_state(ev_tx);
                }
                Err(e) => self.emit_error(ev_tx, cmd_type, &e),
            },
            "destroy_entity" => match self.cmd_destroy_entity(&data) {
                Ok(payload) => {
                    self.emit(ev_tx, "entity_destroyed", payload);
                    self.publish_state(ev_tx);
                }
                Err(e) => self.emit_error(ev_tx, cmd_type, &e),
            },
            "list_entities" => {
                let payload = list_entities_json(self);
                self.emit(ev_tx, "entity_list", payload);
            }
            other => self.emit_error(ev_tx, other, "unknown command"),
        }
    }

    fn cmd_create_entity(&mut self, data: &Value) -> Result<String, String> {
        let name = opt_string(data, "name")?;
        // Optional component overrides by registry name. Everything is
        // parsed BEFORE the spawn so a bad payload leaves the world (and
        // its version) untouched.
        let overrides = match data.get("components") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Object(map)) => parse_overrides(map)?,
            Some(_) => return Err("'components': expected an object".into()),
        };
        let entity = self.spawn(name);
        for (meta, boxed) in overrides {
            // Parsed from the same meta — the box type always matches.
            meta.insert_any(&mut self.store, entity, boxed);
        }
        Ok(serde_json::json!({
            "id": entity.id(),
            "generation": entity.generation(),
            "name": self.name_of(entity),
        })
        .to_string())
    }

    fn cmd_destroy_entity(&mut self, data: &Value) -> Result<String, String> {
        let entity = self.resolve_entity(data)?;
        self.despawn(entity.id(), entity.generation());
        Ok(serde_json::json!({"id": entity.id(), "generation": entity.generation()}).to_string())
    }

    /// Validate `id` + `generation` against the store's allocator.
    fn resolve_entity(&self, data: &Value) -> Result<Entity, String> {
        let id = data
            .get("id")
            .and_then(Value::as_u64)
            .ok_or("missing or invalid 'id'")? as u32;
        let generation = data
            .get("generation")
            .and_then(Value::as_u64)
            .ok_or("missing or invalid 'generation'")? as u32;
        let entity = Entity::new_with_gen(id, generation);
        if !self.store.is_alive(entity) {
            return Err(format!("entity {id}:{generation} not found"));
        }
        Ok(entity)
    }
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
            world.publish_state(&ev_tx);
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
}
