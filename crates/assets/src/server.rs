//! Minimal typed asset registry over the existing scene contract.
//!
//! [`AssetServer`] owns CPU-side scene assets ([`Scene`](crate::scene::Scene)) behind typed
//! [`AssetId`]s and emits [`AssetEvent`] load/error events. Instantiated
//! entities carry the inline [`MeshDesc`](crate::scene::MeshDesc) /
//! [`MaterialDesc`](crate::scene::MaterialDesc) lanes that extraction
//! reads directly; no separate handle lanes exist (checked 2026-09-23:
//! `MeshHandle`/`MaterialHandle` were written by `instantiate` but never
//! read by extraction, so they were removed rather than kept as dead weight).
//!
//! Residency reuses existing mechanisms instead of a new manager island:
//! CPU residency is the server-owned [`Scene`](crate::scene::Scene) plus the [`SmartStore`](ornis_core::SmartStore)
//! lanes written by [`AssetServer::instantiate`]; GPU residency stays with
//! the platform upload path (no render dependency: this crate never names
//! GPU types). Hot reload is a dirty-set plan: [`AssetServer::request_reload`]
//! marks, [`AssetServer::take_dirty`] drains, and the owner re-imports the
//! dirty sources and applies them as world mutations; file watching stays
//! with the owner (editor `SceneFileWatch`). Loaders today: scene `.ron`
//! ([`AssetServer::load_scene_ron`], wrapping [`Scene::from_ron`](crate::scene::Scene::from_ron))
//! and glTF geometry ([`AssetServer::load_gltf`], via [`crate::import`]).

use std::collections::{HashMap, HashSet};

use crate::scene::{EntityDesc, Scene};
use ornis_core::{Engine, Entity, SmartStore};

/// Opaque asset generation counter: every loaded asset gets a fresh id.
const FIRST_ASSET_INDEX: u64 = 1;

/// Opaque handle of one loaded asset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AssetId {
    index: u64,
}

impl AssetId {
    /// Raw generation counter, useful for deterministic ordering in tests.
    pub fn index(self) -> u64 {
        self.index
    }
}

/// Asset kinds the server can load (today only scenes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetKind {
    /// Scene `.ron` asset ([`Scene`]).
    Scene,
}

impl AssetKind {
    /// Short stable label of the kind.
    pub fn name(self) -> &'static str {
        match self {
            AssetKind::Scene => "scene",
        }
    }
}

/// Load/error notifications drained via [`AssetServer::take_events`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssetEvent {
    /// An asset finished loading and is addressable by `id`.
    Loaded {
        /// Handle of the loaded asset.
        id: AssetId,
        /// Kind that was loaded.
        kind: AssetKind,
    },
    /// A load failed; the world is untouched.
    Failed {
        /// Kind that failed to load.
        kind: AssetKind,
        /// Typed reason (scene parse failure; glTF failures return
        /// directly from `load_gltf` without emitting an event).
        error: SceneLoadError,
    },
}

/// Scene `.ron` parse failure.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct SceneLoadError {
    message: String,
}

impl SceneLoadError {
    /// Human-readable reason.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Parses a scene `.ron` asset without touching any world.
///
/// Thin wrapper over [`Scene::from_ron`]; the editor routes its scene-file
/// parsing through here so native, editor and tests share one loader.
///
/// # Errors
///
/// [`SceneLoadError`] when the text is not a valid scene.
pub fn parse_scene_ron(ron_str: &str) -> Result<Scene, SceneLoadError> {
    Scene::from_ron(ron_str).map_err(|error| SceneLoadError {
        message: error.to_string(),
    })
}

/// Minimal typed asset registry: owns scene sources and emits events.
pub struct AssetServer {
    next: u64,
    scenes: HashMap<AssetId, Scene>,
    sources: HashMap<AssetId, String>,
    dirty: HashSet<AssetId>,
    events: Vec<AssetEvent>,
}

impl Default for AssetServer {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetServer {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            next: FIRST_ASSET_INDEX,
            scenes: HashMap::new(),
            sources: HashMap::new(),
            dirty: HashSet::new(),
            events: Vec::new(),
        }
    }

    /// Whether no assets are loaded.
    pub fn is_empty(&self) -> bool {
        self.scenes.is_empty()
    }

    /// Number of loaded scene assets.
    pub fn scene_count(&self) -> usize {
        self.scenes.len()
    }

    /// Whether `id` addresses a loaded scene.
    pub fn contains(&self, id: AssetId) -> bool {
        self.scenes.contains_key(&id)
    }

    /// Stores an already-parsed scene and emits [`AssetEvent::Loaded`].
    ///
    /// `source` keeps the originating `.ron` text for round-trip and
    /// hot-reload plan; `None` stores no text.
    pub fn load_scene(&mut self, scene: Scene, source: Option<String>) -> AssetId {
        let id = AssetId { index: self.next };
        self.next = self.next.saturating_add(1);
        if let Some(source) = source {
            self.sources.insert(id, source);
        }
        self.scenes.insert(id, scene);
        self.events.push(AssetEvent::Loaded {
            id,
            kind: AssetKind::Scene,
        });
        id
    }

    /// Parses and stores a scene `.ron` asset.
    ///
    /// On success emits [`AssetEvent::Loaded`] and returns the id; on
    /// failure emits [`AssetEvent::Failed`] and returns the parse error.
    pub fn load_scene_ron(&mut self, ron_str: &str) -> Result<AssetId, SceneLoadError> {
        match parse_scene_ron(ron_str) {
            Ok(scene) => Ok(self.load_scene(scene, Some(ron_str.to_owned()))),
            Err(error) => {
                self.events.push(AssetEvent::Failed {
                    kind: AssetKind::Scene,
                    error: error.clone(),
                });
                Err(error)
            }
        }
    }

    /// Borrows a loaded scene by id.
    pub fn get_scene(&self, id: AssetId) -> Option<&Scene> {
        self.scenes.get(&id)
    }

    /// Re-serializes a loaded scene to `.ron`.
    ///
    /// Round-trip proof: `parse(load(x).ron) == load(parse(x))` up to RON
    /// formatting. Returns `None` for unknown ids.
    pub fn scene_ron(&self, id: AssetId) -> Option<String> {
        self.scenes.get(&id)?.to_ron().ok()
    }

    /// Parses glTF bytes (`.glb` or `.gltf`) into a [`Scene`] via
    /// [`crate::import`] and stores it. Textured slots keep their scalar
    /// fallback until the GPU upload step learns images.
    ///
    /// # Errors
    ///
    /// Returns the typed [`ornis_gltf::ImportError`]; the registry is untouched.
    pub fn load_gltf(&mut self, bytes: &[u8]) -> Result<AssetId, ornis_gltf::ImportError> {
        let loaded = ornis_gltf::load_slice(bytes)?;
        Ok(self.load_scene(crate::import::scene_from_gltf(&loaded), None))
    }

    /// Reads a glTF file (resolving sibling `.bin` like
    /// [`ornis_gltf::load_path`]) and stores it as a scene.
    ///
    /// # Errors
    ///
    /// Returns the IO/import error typed as [`ornis_gltf::ImportError`];
    /// the registry is untouched.
    pub fn load_gltf_file(
        &mut self,
        path: &std::path::Path,
    ) -> Result<AssetId, ornis_gltf::ImportError> {
        let loaded = ornis_gltf::load_path(path)?;
        Ok(self.load_scene(crate::import::scene_from_gltf(&loaded), None))
    }

    /// Marks an asset dirty for hot reload; `false` for unknown ids.
    ///
    /// Programmatic invalidation API. The editor's file watcher drives
    /// reloads by re-reading files (see below); this set is for hosts
    /// that detect staleness another way.
    pub fn request_reload(&mut self, id: AssetId) -> bool {
        if !self.scenes.contains_key(&id) {
            return false;
        }
        self.dirty.insert(id);
        true
    }

    /// Drains dirty assets in id order; the second call is empty.
    ///
    /// Honest boundary: a dirty scene re-applies as a whole-world
    /// replace (same as file reload), not as per-entity mutations —
    /// allocator `id`/`generation` handles are unstable across loads,
    /// so there is no identity to patch by. Per-entity reload streams
    /// await stable asset-entity ids (explicit next step, not a stub:
    /// the replace path is real and tested).
    pub fn take_dirty(&mut self) -> Vec<AssetId> {
        let mut out: Vec<AssetId> = self.dirty.iter().copied().collect();
        out.sort_by_key(|id| id.index());
        self.dirty.clear();
        out
    }

    /// Drains pending load/error events in emission order.
    pub fn take_events(&mut self) -> Vec<AssetEvent> {
        std::mem::take(&mut self.events)
    }

    /// Spawns every entity of scene `id` into `engine`.
    ///
    /// Writes the existing `Transform`/`Mesh`/`Material` lanes that
    /// extraction reads directly; returns `None` for unknown ids.
    /// Physics, gameplay and GPU state stay with the caller.
    pub fn instantiate(&self, engine: &mut Engine, id: AssetId) -> Option<Vec<Entity>> {
        let scene = self.scenes.get(&id)?;
        let store = engine.world_mut().store_mut()?;
        let mut out = Vec::with_capacity(scene.entities.len());
        for desc in scene.entities.iter() {
            out.push(insert_asset_entity(store, desc));
        }
        Some(out)
    }
}

fn insert_asset_entity(store: &mut SmartStore, desc: &EntityDesc) -> Entity {
    let entity = store.create_entity();
    store.insert(entity, desc.transform.clone());
    store.insert(entity, desc.mesh.clone());
    store.insert(entity, desc.material.clone());
    entity
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{CameraDesc, EntityDesc, MaterialDesc, MeshDesc, TransformDesc};
    use ornis_core::Engine;

    /// Shipped demo scene: five spheres over two directional lights.
    const DEMO_RON: &str = include_str!("../../../assets/scene.ron");
    /// Showcase demo scene: floor + wall + tower + rolling ball.
    const SHOWCASE_RON: &str = include_str!("../../../assets/demo_scene.ron");

    fn entity_desc(name: &str, x: f32) -> EntityDesc {
        EntityDesc {
            name: name.into(),
            transform: TransformDesc {
                translation: [x, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
            mesh: MeshDesc::Sphere {
                radius: ornis_core::units::PositiveF32::expect_valid(1.0),
                segments: 16,
                rings: 8,
            },
            material: MaterialDesc::Metal {
                base_color: [0.9, 0.7, 0.1],
                roughness: ornis_core::units::Clamped01::new(0.2),
                emission: [0.0, 0.0, 0.0],
            },
        }
    }

    fn two_entity_scene() -> Scene {
        Scene {
            name: "assets-test".into(),
            entities: vec![entity_desc("a", -1.0), entity_desc("b", 1.0)],
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
    fn loads_demo_scene_and_emits_loaded() {
        let mut server = AssetServer::new();
        assert!(server.is_empty());
        let id = server.load_scene_ron(DEMO_RON).expect("demo scene loads");
        assert_eq!(server.scene_count(), 1);
        assert!(server.contains(id));
        assert_eq!(server.get_scene(id).expect("stored").entities.len(), 5);
        assert_eq!(
            server.take_events(),
            vec![AssetEvent::Loaded {
                id,
                kind: AssetKind::Scene
            }]
        );
        assert!(server.take_events().is_empty());
    }

    #[test]
    fn loads_showcase_scene_with_floor_wall_tower_and_ball() {
        let mut server = AssetServer::new();
        let id = server
            .load_scene_ron(SHOWCASE_RON)
            .expect("showcase scene loads");
        let scene = server.get_scene(id).expect("stored");
        assert_eq!(scene.name, "demo_showcase");
        assert_eq!(scene.entities.len(), 7);
        let names: Vec<&str> = scene
            .entities
            .iter()
            .map(|entity| entity.name.as_str())
            .collect();
        for expected in [
            "Rolling Ball",
            "Floor",
            "Wall",
            "Tower Base",
            "Tower Mid",
            "Tower Top",
        ] {
            assert!(names.contains(&expected), "missing {expected}: {names:?}");
        }
        let mut engine = Engine::new();
        let entities = server
            .instantiate(&mut engine, id)
            .expect("showcase instantiates");
        assert_eq!(entities.len(), 7);
        let store = engine.world().store().expect("store");
        assert_eq!(
            store
                .read_lane::<crate::scene::TransformDesc>()
                .expect("lane")
                .len(),
            7
        );
        assert_eq!(
            store
                .read_lane::<crate::scene::MeshDesc>()
                .expect("lane")
                .len(),
            7
        );
        assert_eq!(
            store
                .read_lane::<crate::scene::MaterialDesc>()
                .expect("lane")
                .len(),
            7
        );
    }

    #[test]
    fn invalid_ron_emits_failed_and_leaves_world_untouched() {
        let mut server = AssetServer::new();
        let error = server
            .load_scene_ron("Scene(name: 42)")
            .expect_err("malformed RON fails");
        assert!(!error.message().is_empty());
        assert!(server.is_empty());
        let events = server.take_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], AssetEvent::Failed { .. }));
        assert!(server.take_events().is_empty());
    }

    #[test]
    fn failed_event_carries_kind_and_typed_reason() {
        let mut server = AssetServer::new();
        let error = server
            .load_scene_ron("not a scene at all")
            .expect_err("malformed RON fails");
        let events = server.take_events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            AssetEvent::Failed {
                kind,
                error: event_error,
            } => {
                assert_eq!(*kind, AssetKind::Scene);
                assert_eq!(*event_error, error);
                assert!(!event_error.message().is_empty());
            }
            AssetEvent::Loaded { .. } => panic!("expected Failed, got Loaded"),
        }
    }

    #[test]
    fn stored_scene_round_trips_through_ron() {
        let mut server = AssetServer::new();
        let id = server.load_scene_ron(DEMO_RON).expect("demo scene loads");
        let serialized = server.scene_ron(id).expect("re-serialize");
        let reparsed = parse_scene_ron(&serialized).expect("re-parse");
        assert_eq!(reparsed.entities.len(), 5);
        assert_eq!(reparsed.name, "demo");
        let reserialized = reparsed.to_ron().expect("re-serialize twice");
        assert_eq!(serialized, reserialized);
    }

    #[test]
    fn instantiate_populates_lanes() {
        let mut server = AssetServer::new();
        let scene = two_entity_scene();
        let id = server.load_scene(scene, None);
        let mut engine = Engine::new();
        let entities = server
            .instantiate(&mut engine, id)
            .expect("known asset instantiates");
        assert_eq!(entities.len(), 2);

        let store = engine.world().store().expect("store");
        assert_eq!(
            store
                .read_lane::<crate::scene::TransformDesc>()
                .expect("lane")
                .len(),
            2
        );
        assert_eq!(
            store
                .read_lane::<crate::scene::MeshDesc>()
                .expect("lane")
                .len(),
            2
        );
        assert_eq!(
            store
                .read_lane::<crate::scene::MaterialDesc>()
                .expect("lane")
                .len(),
            2
        );
    }

    #[test]
    fn instantiate_unknown_id_returns_none() {
        let server = AssetServer::new();
        let mut engine = Engine::new();
        let unknown = AssetId { index: 999 };
        assert!(server.instantiate(&mut engine, unknown).is_none());
        assert!(server.scene_ron(unknown).is_none());
    }

    #[test]
    fn hot_reload_dirty_cycle_is_stable() {
        let mut server = AssetServer::new();
        let id = server.load_scene(two_entity_scene(), None);
        assert!(!server.request_reload(AssetId { index: 999 }));
        assert!(server.request_reload(id));
        assert!(server.request_reload(id));
        assert_eq!(server.take_dirty(), vec![id]);
        assert!(server.take_dirty().is_empty());
    }

    #[test]
    fn asset_kinds_and_ids_have_stable_labels() {
        assert_eq!(AssetKind::Scene.name(), "scene");
        let mut server = AssetServer::new();
        let first = server.load_scene(two_entity_scene(), None);
        let second = server.load_scene(two_entity_scene(), None);
        assert_ne!(first, second);
        assert!(second.index() > first.index());
    }

    /// One triangle (positions + indices) as base64 buffer bytes.
    const TRIANGLE_B64: &str = "AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAEAAAACAAAA";

    fn triangle_gltf_json() -> String {
        let mut json = serde_json::json!({
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "tri"}],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
                "material": 0,
            }]}],
            "materials": [{"pbrMetallicRoughness": {
                "baseColorFactor": [0.9, 0.8, 0.2, 1.0],
                "metallicFactor": 1.0,
                "roughnessFactor": 0.2,
            }}],
            "buffers": [{"byteLength": 48, "uri": "data:application/octet-stream;base64,__B64__"}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 36},
                {"buffer": 0, "byteOffset": 36, "byteLength": 12},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                 "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
                {"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"},
            ],
        });
        let uri = format!("data:application/octet-stream;base64,{TRIANGLE_B64}");
        json["buffers"][0]["uri"] = serde_json::Value::String(uri);
        serde_json::to_string(&json).expect("fixture serializes")
    }

    #[test]
    fn load_gltf_stores_scene_end_to_end() {
        // Real bytes (not hand-built structs): JSON + base64 buffer.
        let mut server = AssetServer::new();
        let id = server
            .load_gltf(triangle_gltf_json().as_bytes())
            .expect("triangle gltf loads");
        let scene = server.get_scene(id).expect("stored");
        assert_eq!(scene.entities.len(), 1);
        assert_eq!(scene.entities[0].name, "tri");
        assert_eq!(scene.entities[0].transform.translation, [0.0, 0.0, 0.0]);
        assert!(matches!(
            scene.entities[0].mesh,
            crate::scene::MeshDesc::Custom { .. }
        ));
        assert!(matches!(
            scene.entities[0].material,
            crate::scene::MaterialDesc::Metal { .. }
        ));
        assert_eq!(
            server.take_events(),
            vec![AssetEvent::Loaded {
                id,
                kind: AssetKind::Scene
            }]
        );
    }

    #[test]
    fn load_gltf_garbage_is_a_typed_import_error() {
        // Broken bytes are `ImportError`, not a `String`: the registry
        // stays untouched and the reason keeps its variant.
        let mut server = AssetServer::new();
        let error = server
            .load_gltf(b"definitely not gltf")
            .expect_err("garbage bytes fail");
        assert!(
            matches!(error, ornis_gltf::ImportError::Parse { .. }),
            "{error:?}"
        );
        assert!(server.is_empty());
        assert!(server.take_events().is_empty());
    }
}
