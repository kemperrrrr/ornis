//! Server-side ECS world for `editor-only` mode.
//!
//! In `editor-only` there is no native winit loop to consume `UiCommand`s,
//! so [`run`] spawns an `editor-world` thread that owns an [`EditorWorld`]
//! (ornis-core `EntityAllocator` + `ComponentStore`s), executes commands
//! from `POST /api/command` and publishes `GameEvent`s back to the HTTP
//! server (`status`/`scene` snapshots are cached by `remote.rs` for
//! `GET /api/status` and `GET /api/scene`; the rest reach `GET /api/events`).
//!
//! At startup the world loads `editor/scene.ron` (via
//! `ornis_render::scene::Scene::from_ron`), so the live world matches what
//! the WASM viewport renders statically. Component payloads reuse the
//! `ornis_render::scene` description types; the JSON contract of
//! [`EditorWorld::scene_json`] is:
//!
//! ```json
//! {
//!   "version": 5, "entity_count": 2,
//!   "entities": [{
//!     "id": 0, "generation": 0, "name": "Red Sphere",
//!     "components": ["Name", "Transform", "Mesh", "Material"],
//!     "transform": {"translation": [-5.6, 0, 0], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
//!     "mesh": {"kind": "sphere", "radius": 1.0, "segments": 32, "rings": 24},
//!     "material": {"kind": "dielectric", "base_color": [0.8, 0.2, 0.2], "roughness": 0.5}
//!   }],
//!   "lights": [{"kind": "directional", "direction": [1, 1, 1], "intensity": 0.6, "color": [1, 1, 1]}],
//!   "camera": {"position": [0, 2.5, 9], "target": [0, 0, 0], "up": [0, 1, 0], "fov": 60.0, "near": 0.1, "far": 100.0},
//!   "ambient": [0.10, 0.10, 0.15]
//! }
//! ```
//!
//! `version` is incremented on every mutation so clients can cheaply detect
//! changes. Invalid commands never panic: they produce an `error` event and
//! leave the world (and its version) untouched.

use std::fs;
use std::path::PathBuf;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, Sender};
use serde_json::Value;

use ornis_core::{ComponentStore, Entity, EntityAllocator};
use ornis_render::scene::{CameraDesc, LightDesc, MaterialDesc, MeshDesc, Scene, TransformDesc};

use crate::ipc::{GameEvent, UiCommand};

/// Editor-side name component attached to every spawned entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Name(pub String);

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
    allocator: EntityAllocator,
    alive: Vec<Entity>,
    names: ComponentStore<Name>,
    transforms: ComponentStore<TransformDesc>,
    meshes: ComponentStore<MeshDesc>,
    materials: ComponentStore<MaterialDesc>,
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
        self.spawn_with(name, default_transform(), default_mesh(), default_material())
    }

    pub fn spawn_with(
        &mut self,
        name: Option<String>,
        transform: TransformDesc,
        mesh: MeshDesc,
        material: MaterialDesc,
    ) -> Entity {
        let entity = self.allocator.allocate();
        self.alive.push(entity);
        let name = name.unwrap_or_else(|| format!("Entity {}", entity.id()));
        self.names.insert(entity, Name(name));
        self.transforms.insert(entity, transform);
        self.meshes.insert(entity, mesh);
        self.materials.insert(entity, material);
        self.version += 1;
        entity
    }

    /// Despawn by id/generation. Returns the entity if it was alive.
    pub fn despawn(&mut self, id: u32, generation: u32) -> Option<Entity> {
        let entity = Entity::new_with_gen(id, generation);
        if !self.allocator.is_alive(entity) {
            return None;
        }
        self.alive.retain(|e| *e != entity);
        self.names.remove(entity);
        self.transforms.remove(entity);
        self.meshes.remove(entity);
        self.materials.remove(entity);
        self.allocator.deallocate(entity);
        self.version += 1;
        Some(entity)
    }

    pub fn name_of(&self, entity: Entity) -> Option<&str> {
        self.names.get(entity).map(|n| n.0.as_str())
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
        let mut entities: Vec<Value> = Vec::with_capacity(self.alive.len());
        for &entity in &self.alive {
            let mut components = Vec::new();
            let mut name = Value::Null;
            if let Some(n) = self.name_of(entity) {
                components.push("Name");
                name = Value::String(n.to_string());
            }
            let transform = self.transforms.get(entity).map(transform_json);
            if transform.is_some() {
                components.push("Transform");
            }
            let mesh = self.meshes.get(entity).map(mesh_json);
            if mesh.is_some() {
                components.push("Mesh");
            }
            let material = self.materials.get(entity).map(material_json);
            if material.is_some() {
                components.push("Material");
            }
            entities.push(serde_json::json!({
                "id": entity.id(),
                "generation": entity.generation(),
                "name": name,
                "components": components,
                "transform": transform,
                "mesh": mesh,
                "material": material,
            }));
        }
        let lights: Vec<Value> = self.environment.lights.iter().map(light_json).collect();
        serde_json::json!({
            "version": self.version,
            "entity_count": self.entity_count(),
            "entities": entities,
            "lights": lights,
            "camera": camera_json(&self.environment.camera),
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
                let target = self.alive.iter().find(|e| e.id() == *entity_id).copied();
                if let Some(entity) = target {
                    self.despawn(entity.id(), entity.generation());
                    self.publish_state(ev_tx);
                }
            }
            UiCommand::Custom {
                cmd_type,
                json_data,
            } => self.handle_custom(cmd_type, json_data, ev_tx),
            // SetComponent has no editable components yet on the server side.
            UiCommand::SetComponent { .. } => {}
        }
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
            "set_transform" => match self.cmd_set_transform(&data) {
                Ok(payload) => {
                    self.emit(ev_tx, "entity_updated", payload);
                    self.publish_state(ev_tx);
                }
                Err(e) => self.emit_error(ev_tx, cmd_type, &e),
            },
            "set_material" => match self.cmd_set_material(&data) {
                Ok(payload) => {
                    self.emit(ev_tx, "entity_updated", payload);
                    self.publish_state(ev_tx);
                }
                Err(e) => self.emit_error(ev_tx, cmd_type, &e),
            },
            "rename_entity" => match self.cmd_rename_entity(&data) {
                Ok(payload) => {
                    self.emit(ev_tx, "entity_updated", payload);
                    self.publish_state(ev_tx);
                }
                Err(e) => self.emit_error(ev_tx, cmd_type, &e),
            },
            "list_entities" => {
                let payload = self.cmd_list_entities();
                self.emit(ev_tx, "entity_list", payload);
            }
            other => self.emit_error(ev_tx, other, "unknown command"),
        }
    }

    fn cmd_create_entity(&mut self, data: &Value) -> Result<String, String> {
        let name = opt_string(data, "name")?;
        let transform = match data.get("transform") {
            None | Some(Value::Null) => default_transform(),
            Some(v) => parse_transform(v)?,
        };
        let mesh = match data.get("mesh") {
            None | Some(Value::Null) => default_mesh(),
            Some(v) => parse_mesh(v)?,
        };
        let material = match data.get("material") {
            None | Some(Value::Null) => default_material(),
            Some(v) => parse_material(v)?,
        };
        let entity = self.spawn_with(name, transform, mesh, material);
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

    fn cmd_set_transform(&mut self, data: &Value) -> Result<String, String> {
        let entity = self.resolve_entity(data)?;
        // Parse first so a malformed payload changes nothing.
        let translation = opt_f32s(data, "translation")?;
        let rotation = opt_f32s(data, "rotation")?;
        let scale = opt_f32s(data, "scale")?;
        let t = self
            .transforms
            .get_mut(entity)
            .ok_or("entity has no Transform")?;
        if let Some(v) = translation {
            t.translation = v;
        }
        if let Some(v) = rotation {
            t.rotation = v;
        }
        if let Some(v) = scale {
            t.scale = v;
        }
        self.version += 1;
        Ok(serde_json::json!({
            "id": entity.id(),
            "generation": entity.generation(),
            "component": "transform",
            "transform": transform_json(t),
        })
        .to_string())
    }

    fn cmd_set_material(&mut self, data: &Value) -> Result<String, String> {
        let entity = self.resolve_entity(data)?;
        let material = parse_material(data.get("material").ok_or("missing 'material'")?)?;
        let payload = material_json(&material);
        *self.materials.get_mut(entity).ok_or("entity has no Material")? = material;
        self.version += 1;
        Ok(serde_json::json!({
            "id": entity.id(),
            "generation": entity.generation(),
            "component": "material",
            "material": payload,
        })
        .to_string())
    }

    fn cmd_rename_entity(&mut self, data: &Value) -> Result<String, String> {
        let entity = self.resolve_entity(data)?;
        let name = data
            .get("name")
            .and_then(Value::as_str)
            .ok_or("missing or invalid 'name'")?
            .to_string();
        self.names.get_mut(entity).ok_or("entity has no Name")?.0 = name.clone();
        self.version += 1;
        Ok(serde_json::json!({
            "id": entity.id(),
            "generation": entity.generation(),
            "component": "name",
            "name": name,
        })
        .to_string())
    }

    fn cmd_list_entities(&self) -> String {
        let entities: Vec<Value> = self
            .alive
            .iter()
            .map(|&e| {
                serde_json::json!({
                    "id": e.id(),
                    "generation": e.generation(),
                    "name": self.name_of(e),
                })
            })
            .collect();
        serde_json::json!({"count": entities.len(), "entities": entities}).to_string()
    }

    /// Validate `id` + `generation` against the allocator.
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
        if !self.allocator.is_alive(entity) {
            return Err(format!("entity {id}:{generation} not found"));
        }
        Ok(entity)
    }
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

fn transform_json(t: &TransformDesc) -> Value {
    serde_json::json!({
        "translation": t.translation,
        "rotation": t.rotation,
        "scale": t.scale,
    })
}

fn mesh_json(m: &MeshDesc) -> Value {
    match m {
        MeshDesc::Sphere {
            radius,
            segments,
            rings,
        } => serde_json::json!({
            "kind": "sphere",
            "radius": radius,
            "segments": segments,
            "rings": rings,
        }),
    }
}

fn material_json(m: &MaterialDesc) -> Value {
    match m {
        MaterialDesc::Dielectric {
            base_color,
            roughness,
        } => serde_json::json!({
            "kind": "dielectric",
            "base_color": base_color,
            "roughness": roughness,
        }),
        MaterialDesc::Metal {
            base_color,
            roughness,
        } => serde_json::json!({
            "kind": "metal",
            "base_color": base_color,
            "roughness": roughness,
        }),
        MaterialDesc::Coat {
            base_color,
            coat_weight,
            coat_roughness,
        } => serde_json::json!({
            "kind": "coat",
            "base_color": base_color,
            "coat_weight": coat_weight,
            "coat_roughness": coat_roughness,
        }),
    }
}

fn light_json(l: &LightDesc) -> Value {
    match l {
        LightDesc::Directional {
            direction,
            intensity,
            color,
        } => serde_json::json!({
            "kind": "directional",
            "direction": direction,
            "intensity": intensity,
            "color": color,
        }),
    }
}

fn camera_json(c: &CameraDesc) -> Value {
    serde_json::json!({
        "position": c.position,
        "target": c.target,
        "up": c.up,
        "fov": c.fov,
        "near": c.near,
        "far": c.far,
    })
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

fn parse_f32s<const N: usize>(v: &Value) -> Result<[f32; N], String> {
    let arr = v.as_array().ok_or("expected an array")?;
    if arr.len() != N {
        return Err(format!("expected {N} elements, got {}", arr.len()));
    }
    let mut out = [0.0; N];
    for (i, x) in arr.iter().enumerate() {
        out[i] = x.as_f64().ok_or("expected a number")? as f32;
    }
    Ok(out)
}

fn opt_f32s<const N: usize>(data: &Value, key: &str) -> Result<Option<[f32; N]>, String> {
    match data.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => parse_f32s(v)
            .map(Some)
            .map_err(|e| format!("'{key}': {e}")),
    }
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

fn f32_field(v: &Value, key: &str, default: f32) -> Result<f32, String> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(x) => x
            .as_f64()
            .map(|f| f as f32)
            .ok_or_else(|| format!("'{key}': expected a number")),
    }
}

fn u32_field(v: &Value, key: &str, default: u32) -> Result<u32, String> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(x) => x
            .as_u64()
            .map(|f| f as u32)
            .ok_or_else(|| format!("'{key}': expected a non-negative integer")),
    }
}

fn parse_transform(v: &Value) -> Result<TransformDesc, String> {
    let mut t = default_transform();
    if let Some(x) = opt_f32s(v, "translation")? {
        t.translation = x;
    }
    if let Some(x) = opt_f32s(v, "rotation")? {
        t.rotation = x;
    }
    if let Some(x) = opt_f32s(v, "scale")? {
        t.scale = x;
    }
    Ok(t)
}

fn parse_mesh(v: &Value) -> Result<MeshDesc, String> {
    let kind = opt_string(v, "kind")?.unwrap_or_else(|| "sphere".into());
    match kind.as_str() {
        "sphere" => Ok(MeshDesc::Sphere {
            radius: f32_field(v, "radius", 1.0)?,
            segments: u32_field(v, "segments", 32)?,
            rings: u32_field(v, "rings", 24)?,
        }),
        other => Err(format!("unknown mesh kind '{other}'")),
    }
}

fn parse_material(v: &Value) -> Result<MaterialDesc, String> {
    let kind = opt_string(v, "kind")?.unwrap_or_else(|| "dielectric".into());
    let base_color = opt_f32s(v, "base_color")?.unwrap_or([0.5, 0.5, 0.5]);
    match kind.as_str() {
        "dielectric" => Ok(MaterialDesc::Dielectric {
            base_color,
            roughness: f32_field(v, "roughness", 0.5)?,
        }),
        "metal" => Ok(MaterialDesc::Metal {
            base_color,
            roughness: f32_field(v, "roughness", 0.5)?,
        }),
        "coat" => Ok(MaterialDesc::Coat {
            base_color,
            coat_weight: f32_field(v, "coat_weight", 1.0)?,
            coat_roughness: f32_field(v, "coat_roughness", 0.1)?,
        }),
        other => Err(format!("unknown material kind '{other}'")),
    }
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

    #[test]
    fn spawn_assigns_names_and_counts() {
        let mut world = EditorWorld::new();
        let a = world.spawn(None);
        let b = world.spawn(Some("Hero".into()));
        assert_eq!(world.entity_count(), 2);
        assert_eq!(world.name_of(a), Some("Entity 0"));
        assert_eq!(world.name_of(b), Some("Hero"));
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
        assert_eq!(world.name_of(b), Some("Entity 1"));
    }

    #[test]
    fn scene_ron_round_trip() {
        let ron = fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("editor/scene.ron"),
        )
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

        let red = &entities[0];
        assert_eq!(red["id"], 0);
        assert_eq!(red["generation"], 0);
        assert_eq!(red["name"], "Red Sphere");
        assert_eq!(
            red["components"],
            serde_json::json!(["Name", "Transform", "Mesh", "Material"])
        );
        assert_f32_seq(&red["transform"]["translation"], &[-5.6, 0.0, 0.0]);
        assert_f32_seq(&red["transform"]["rotation"], &[0.0, 0.0, 0.0, 1.0]);
        assert_f32_seq(&red["transform"]["scale"], &[1.0, 1.0, 1.0]);
        assert_eq!(red["mesh"]["kind"], "sphere");
        assert_f32(&red["mesh"]["radius"], 1.0);
        assert_eq!(red["mesh"]["segments"], 32);
        assert_eq!(red["mesh"]["rings"], 24);
        assert_eq!(red["material"]["kind"], "dielectric");
        assert_f32_seq(&red["material"]["base_color"], &[0.8, 0.2, 0.2]);
        assert_f32(&red["material"]["roughness"], 0.5);

        // Material variants survive the round trip.
        assert_eq!(entities[3]["name"], "Gold Sphere");
        assert_eq!(entities[3]["material"]["kind"], "metal");
        assert_f32_seq(&entities[3]["material"]["base_color"], &[0.9, 0.7, 0.1]);
        assert_eq!(entities[4]["name"], "Ceramic Sphere");
        assert_eq!(entities[4]["material"]["kind"], "coat");
        assert_f32(&entities[4]["material"]["coat_weight"], 1.0);
        assert_f32(&entities[4]["material"]["coat_roughness"], 0.1);

        // Environment resource.
        let lights = scene["lights"].as_array().unwrap();
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0]["kind"], "directional");
        assert_f32_seq(&lights[0]["direction"], &[1.0, 1.0, 1.0]);
        assert_f32(&lights[0]["intensity"], 0.6);
        assert_f32_seq(&lights[1]["color"], &[0.8, 0.8, 1.0]);
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
        assert_eq!(entities[0]["name"], "Entity 0");
        assert_eq!(
            entities[0]["components"],
            serde_json::json!(["Name", "Transform", "Mesh", "Material"])
        );
        // Default components: unit sphere at the origin, gray dielectric.
        assert_f32_seq(&entities[0]["transform"]["translation"], &[0.0, 0.0, 0.0]);
        assert_f32_seq(&entities[0]["transform"]["rotation"], &[0.0, 0.0, 0.0, 1.0]);
        assert_eq!(entities[0]["mesh"]["kind"], "sphere");
        assert_eq!(entities[0]["material"]["kind"], "dielectric");
        assert_f32_seq(&entities[0]["material"]["base_color"], &[0.5, 0.5, 0.5]);
        assert_eq!(entities[1]["name"], "Hero");
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
            &custom("set_transform", r#"{"id":0,"generation":0,"translation":[1,2,3]}"#),
            &ev_tx,
        );
        assert_eq!(world.version, 2);
        // Failed command: no bump.
        world.handle_command(&custom("set_transform", r#"{"id":9,"generation":0}"#), &ev_tx);
        assert_eq!(world.version, 2);
        world.handle_command(
            &custom("rename_entity", r#"{"id":0,"generation":0,"name":"X"}"#),
            &ev_tx,
        );
        assert_eq!(world.version, 3);
        world.handle_command(
            &custom("set_material", r#"{"id":0,"generation":0,"material":{"kind":"metal"}}"#),
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
        assert_eq!(scene["entities"][0]["name"], "Hero");

        assert!(ev_rx.try_recv().is_err(), "no leftover events");
    }

    #[test]
    fn create_entity_with_full_payload() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(
            &custom(
                "create_entity",
                r#"{
                    "name": "Metal Ball",
                    "transform": {"translation": [1, 2, 3], "scale": [2, 2, 2]},
                    "mesh": {"kind": "sphere", "radius": 2.0, "segments": 16, "rings": 8},
                    "material": {"kind": "metal", "base_color": [0.9, 0.7, 0.1], "roughness": 0.2}
                }"#,
            ),
            &ev_tx,
        );
        assert_eq!(world.entity_count(), 1);

        let scene: Value = serde_json::from_str(&world.scene_json()).unwrap();
        let e = &scene["entities"][0];
        assert_eq!(e["name"], "Metal Ball");
        assert_f32_seq(&e["transform"]["translation"], &[1.0, 2.0, 3.0]);
        // Unspecified transform fields keep their defaults.
        assert_f32_seq(&e["transform"]["rotation"], &[0.0, 0.0, 0.0, 1.0]);
        assert_f32_seq(&e["transform"]["scale"], &[2.0, 2.0, 2.0]);
        assert_eq!(e["mesh"]["kind"], "sphere");
        assert_f32(&e["mesh"]["radius"], 2.0);
        assert_eq!(e["mesh"]["segments"], 16);
        assert_eq!(e["mesh"]["rings"], 8);
        assert_eq!(e["material"]["kind"], "metal");
        assert_f32_seq(&e["material"]["base_color"], &[0.9, 0.7, 0.1]);
        assert_f32(&e["material"]["roughness"], 0.2);

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
        assert_eq!(scene["entities"][0]["name"], "Entity 0");
    }

    #[test]
    fn set_transform_updates_only_given_fields() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", ""), &ev_tx);
        world.handle_command(
            &custom("set_transform", r#"{"id":0,"generation":0,"translation":[3,0,0]}"#),
            &ev_tx,
        );
        let scene: Value = serde_json::from_str(&world.scene_json()).unwrap();
        let t = &scene["entities"][0]["transform"];
        assert_f32_seq(&t["translation"], &[3.0, 0.0, 0.0]);
        assert_f32_seq(&t["rotation"], &[0.0, 0.0, 0.0, 1.0]);
        assert_f32_seq(&t["scale"], &[1.0, 1.0, 1.0]);

        let updated = custom_events(&drain_all(&ev_rx), "entity_updated");
        assert_eq!(updated.len(), 1);
        let updated: Value = serde_json::from_str(&updated[0]).unwrap();
        assert_eq!(updated["id"], 0);
        assert_eq!(updated["component"], "transform");
        assert_f32_seq(&updated["transform"]["translation"], &[3.0, 0.0, 0.0]);
    }

    #[test]
    fn set_material_replaces_material() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", ""), &ev_tx);
        world.handle_command(
            &custom(
                "set_material",
                r#"{"id":0,"generation":0,"material":{"kind":"coat","base_color":[1,1,1],
                    "coat_weight":1.0,"coat_roughness":0.1}}"#,
            ),
            &ev_tx,
        );
        let scene: Value = serde_json::from_str(&world.scene_json()).unwrap();
        let m = &scene["entities"][0]["material"];
        assert_eq!(m["kind"], "coat");
        assert_f32(&m["coat_weight"], 1.0);
        assert_f32(&m["coat_roughness"], 0.1);

        let updated = custom_events(&drain_all(&ev_rx), "entity_updated");
        assert_eq!(updated.len(), 1);
        let updated: Value = serde_json::from_str(&updated[0]).unwrap();
        assert_eq!(updated["component"], "material");
        assert_eq!(updated["material"]["kind"], "coat");
    }

    #[test]
    fn rename_entity_updates_name() {
        let (mut world, ev_tx, ev_rx) = world_and_events();
        world.handle_command(&custom("create_entity", r#"{"name":"Old"}"#), &ev_tx);
        world.handle_command(
            &custom("rename_entity", r#"{"id":0,"generation":0,"name":"New"}"#),
            &ev_tx,
        );
        let scene: Value = serde_json::from_str(&world.scene_json()).unwrap();
        assert_eq!(scene["entities"][0]["name"], "New");

        let updated = custom_events(&drain_all(&ev_rx), "entity_updated");
        assert_eq!(updated.len(), 1);
        let updated: Value = serde_json::from_str(&updated[0]).unwrap();
        assert_eq!(updated["component"], "name");
        assert_eq!(updated["name"], "New");
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
        // Non-existent entity.
        world.handle_command(
            &custom("set_material", r#"{"id":3,"generation":0,"material":{"kind":"metal"}}"#),
            &ev_tx,
        );
        // Unknown material kind: the entity must not be created.
        world.handle_command(
            &custom("create_entity", r#"{"material":{"kind":"unobtainium"}}"#),
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

        cmd_tx.send(custom("create_entity", r#"{"name":"Hero"}"#)).unwrap();
        // Wait for the scene snapshot reflecting the new entity.
        let mut seen_scene = None;
        for _ in 0..100 {
            while let Ok(ev) = ev_rx.try_recv() {
                if let GameEvent::CustomEvent { cmd_type, json_data } = ev
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
