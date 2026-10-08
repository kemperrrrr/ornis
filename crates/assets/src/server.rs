//! Typed asset registry over the existing scene contract.
//!
//! [`AssetServer`] owns CPU-side scene assets ([`Scene`](crate::scene::Scene)) behind typed
//! [`Handle`]s / [`AssetId`]s and emits [`AssetEvent`] load/error events. Instantiated
//! entities carry the inline [`MeshDesc`](crate::scene::MeshDesc) /
//! [`MaterialDesc`](crate::scene::MaterialDesc) lanes that extraction
//! reads directly; no separate handle lanes exist (checked 2026-09-23:
//! `MeshHandle`/`MaterialHandle` were written by `instantiate` but never
//! read by extraction, so they were removed rather than kept as dead weight).
//!
//! Loading goes through one generic entry point,
//! [`AssetServer::load::<T>`](AssetServer::load): the
//! [`ImporterRegistry`] picks an [`Importer`](crate::Importer) by file
//! extension (formats are cargo features: `gltf` default, `fbx`
//! placeholder; `.ron` always), the same path returns the same handle
//! (dedup), every file-backed asset remembers its path, and every failure
//! is one format-neutral [`AssetError`] that is also broadcast as
//! [`AssetEvent::Failed`]. The older id-based entry points
//! ([`AssetServer::load_scene_ron`], [`AssetServer::load_gltf`],
//! [`AssetServer::load_gltf_file`], ...) stay for the editor and route
//! through the same storage and events.
//!
//! Residency reuses existing mechanisms instead of a new manager island:
//! CPU residency is the server-owned [`Scene`](crate::scene::Scene) plus the [`SmartStore`](ornis_core::SmartStore)
//! lanes written by [`AssetServer::instantiate`]; GPU residency stays with
//! the platform upload path (no render dependency: this crate never names
//! GPU types). Hot reload is a dirty-set plan: [`AssetServer::request_reload`]
//! marks, [`AssetServer::take_dirty`] drains, and the owner re-imports the
//! dirty sources and applies them as world mutations; file watching stays
//! with the owner (editor `SceneFileWatch`). [`AssetServer::reload`]
//! re-imports a file-backed asset in place (same handle).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::error::AssetError;
use crate::handle::{Asset, Handle};
use crate::importer::{ImportedAsset, Importer, ImporterRegistry, SceneImport};
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

    /// Builds an id from its raw counter. Live ids start at 1.
    pub(crate) const fn from_raw(index: u64) -> Self {
        Self { index }
    }
}

/// Asset kinds the server can load.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AssetKind {
    /// Scene asset ([`Scene`]): `.ron`.
    Scene,
    /// glTF model ([`crate::Model`]): `.gltf`/`.glb`.
    #[cfg(feature = "gltf")]
    Model,
}

impl AssetKind {
    /// Short stable label of the kind.
    pub fn name(self) -> &'static str {
        match self {
            AssetKind::Scene => "scene",
            #[cfg(feature = "gltf")]
            AssetKind::Model => "model",
        }
    }
}

impl std::fmt::Display for AssetKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Load/error notifications drained via [`AssetServer::take_events`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssetEvent {
    /// An asset finished loading (or reloading in place) and is
    /// addressable by `id`.
    Loaded {
        /// Handle of the loaded asset.
        id: AssetId,
        /// Kind that was loaded.
        kind: AssetKind,
    },
    /// A load failed; the registry and any world are untouched. Emitted
    /// for every format (RON and glTF alike) and for unsupported
    /// extensions requested through [`AssetServer::load`].
    Failed {
        /// Kind that failed to load.
        kind: AssetKind,
        /// Requested file (`None` for in-memory loads).
        path: Option<PathBuf>,
        /// Typed, format-tagged reason (same value the call returned).
        error: AssetError,
    },
}

/// Scene `.ron` parse failure.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct SceneLoadError {
    message: String,
}

impl SceneLoadError {
    /// Wraps a human-readable reason.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

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

/// Dedup key of a path: canonical when the file exists, verbatim otherwise.
fn path_key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Typed asset registry: owns asset sources, dispatches importers and
/// emits events.
pub struct AssetServer {
    next: u64,
    registry: ImporterRegistry,
    kinds: HashMap<AssetId, AssetKind>,
    scenes: HashMap<AssetId, Scene>,
    /// glTF models ([`crate::Model`]). `.ron` loads have no entry.
    #[cfg(feature = "gltf")]
    models: HashMap<AssetId, crate::Model>,
    sources: HashMap<AssetId, String>,
    /// Requested file per file-backed asset (as given by the caller).
    paths: HashMap<AssetId, PathBuf>,
    /// Dedup index: [`path_key`] → newest asset loaded from that file.
    by_path: HashMap<PathBuf, AssetId>,
    dirty: HashSet<AssetId>,
    events: Vec<AssetEvent>,
}

impl Default for AssetServer {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetServer {
    /// Creates an empty registry with every feature-enabled built-in
    /// importer ([`ImporterRegistry::with_builtins`]).
    pub fn new() -> Self {
        Self::with_registry(ImporterRegistry::with_builtins())
    }

    /// Creates an empty registry dispatching through `registry`.
    pub fn with_registry(registry: ImporterRegistry) -> Self {
        Self {
            next: FIRST_ASSET_INDEX,
            registry,
            kinds: HashMap::new(),
            scenes: HashMap::new(),
            #[cfg(feature = "gltf")]
            models: HashMap::new(),
            sources: HashMap::new(),
            paths: HashMap::new(),
            by_path: HashMap::new(),
            dirty: HashSet::new(),
            events: Vec::new(),
        }
    }

    /// Registers an extra importer (overrides built-ins for shared
    /// extensions, see [`ImporterRegistry::register`]).
    pub fn register_importer(&mut self, importer: impl Importer) {
        self.registry.register(importer);
    }

    /// The extension → importer registry.
    pub fn importers(&self) -> &ImporterRegistry {
        &self.registry
    }

    /// Whether no assets are loaded.
    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    /// Number of loaded scene assets.
    pub fn scene_count(&self) -> usize {
        self.scenes.len()
    }

    /// Whether `id` addresses a loaded scene.
    pub fn contains(&self, id: AssetId) -> bool {
        self.kinds.contains_key(&id)
    }

    /// Loads the asset at `path` as `T`, dispatching by extension.
    ///
    /// The same file (canonical path) returns the same handle without
    /// re-importing and without a new event; use [`AssetServer::reload`]
    /// to pick up changes. On success emits [`AssetEvent::Loaded`]; on
    /// failure emits [`AssetEvent::Failed`] and stores nothing.
    ///
    /// # Errors
    ///
    /// [`AssetError::UnsupportedExtension`] when no enabled importer claims
    /// the extension, [`AssetError::WrongKind`] when the importer (or the
    /// already-loaded asset) is not a `T`, [`AssetError::NotFound`] /
    /// [`AssetError::Io`] for filesystem failures and
    /// [`AssetError::Import`] / [`AssetError::UnsupportedFormat`] from the
    /// importer.
    pub fn load<T: Asset>(&mut self, path: impl AsRef<Path>) -> Result<Handle<T>, AssetError> {
        let path = path.as_ref();
        let key = path_key(path);
        if let Some(&id) = self.by_path.get(&key) {
            let found = self.kinds.get(&id).copied().unwrap_or(T::KIND);
            if found != T::KIND {
                return Err(self.fail(
                    T::KIND,
                    Some(path),
                    AssetError::WrongKind {
                        path: Some(path.to_path_buf()),
                        expected: T::KIND,
                        found,
                    },
                ));
            }
            return Ok(Handle::from_id(id));
        }
        let imported = self.import_path_as(T::KIND, path)?;
        let id = self.store(imported, Some(path));
        Ok(Handle::from_id(id))
    }

    /// Borrows the asset behind a typed handle (`None` once unloaded).
    pub fn get<T: Asset>(&self, handle: &Handle<T>) -> Option<&T> {
        T::get(self, handle.id())
    }

    /// Re-imports a file-backed asset in place: the handle stays valid and
    /// [`AssetEvent::Loaded`] fires again. On failure the previous asset
    /// is kept and [`AssetEvent::Failed`] fires.
    ///
    /// # Errors
    ///
    /// [`AssetError::UnknownHandle`] for unknown or non-file-backed
    /// handles, otherwise the same as [`AssetServer::load`].
    pub fn reload<T: Asset>(&mut self, handle: &Handle<T>) -> Result<(), AssetError> {
        let id = handle.id();
        let Some(path) = self.paths.get(&id).cloned() else {
            return Err(AssetError::UnknownHandle { index: id.index() });
        };
        let imported = self.import_path_as(T::KIND, &path)?;
        self.insert_at(id, imported);
        self.events.push(AssetEvent::Loaded { id, kind: T::KIND });
        Ok(())
    }

    /// Drops an asset and everything retained for it (source text, model,
    /// path index, dirty mark). Returns whether it existed;
    /// handles to it resolve to `None` afterwards.
    pub fn unload<T: Asset>(&mut self, handle: &Handle<T>) -> bool {
        let id = handle.id();
        let existed = self.kinds.remove(&id).is_some();
        self.scenes.remove(&id);
        #[cfg(feature = "gltf")]
        self.models.remove(&id);
        self.sources.remove(&id);
        self.dirty.remove(&id);
        if let Some(path) = self.paths.remove(&id) {
            let key = path_key(&path);
            if self.by_path.get(&key) == Some(&id) {
                self.by_path.remove(&key);
            }
        }
        existed
    }

    /// File the asset was loaded from (`None` for in-memory loads).
    pub fn path(&self, id: AssetId) -> Option<&Path> {
        self.paths.get(&id).map(PathBuf::as_path)
    }

    /// Stores an already-parsed scene and emits [`AssetEvent::Loaded`].
    ///
    /// `source` keeps the originating `.ron` text for round-trip and
    /// hot-reload plan; `None` stores no text.
    pub fn load_scene(&mut self, scene: Scene, source: Option<String>) -> AssetId {
        let mut import = SceneImport::new(scene);
        import.source_text = source;
        self.store(ImportedAsset::Scene(import), None)
    }

    /// Parses and stores a scene `.ron` asset.
    ///
    /// On success emits [`AssetEvent::Loaded`] and returns the id; on
    /// failure emits [`AssetEvent::Failed`] (carrying the
    /// [`AssetError`] form) and returns the parse error.
    ///
    /// # Errors
    ///
    /// [`SceneLoadError`] when the text is not a valid scene (kept as the
    /// return type for the editor's typed `load_scene_ron`).
    pub fn load_scene_ron(&mut self, ron_str: &str) -> Result<AssetId, SceneLoadError> {
        match parse_scene_ron(ron_str) {
            Ok(scene) => Ok(self.load_scene(scene, Some(ron_str.to_owned()))),
            Err(error) => {
                self.fail(AssetKind::Scene, None, AssetError::from(error.clone()));
                Err(error)
            }
        }
    }

    /// Borrows a loaded scene by id.
    pub fn get_scene(&self, id: AssetId) -> Option<&Scene> {
        self.scenes.get(&id)
    }

    /// Borrows a loaded glTF [`Model`](crate::Model) by id.
    ///
    /// Only glTF loads ([`AssetServer::load`] of a `.gltf`/`.glb`,
    /// [`AssetServer::load_gltf`]/[`load_gltf_file`](AssetServer::load_gltf_file))
    /// store a model. `.ron` loads, manual scenes, and unknown ids return
    /// `None`. A flat editor scene is [`scene_from_model`](crate::scene_from_model).
    #[cfg(feature = "gltf")]
    pub fn model(&self, id: AssetId) -> Option<&crate::Model> {
        self.models.get(&id)
    }

    /// Re-serializes a loaded scene to `.ron`.
    ///
    /// Round-trip proof: `parse(load(x).ron) == load(parse(x))` up to RON
    /// formatting. Returns `None` for unknown ids.
    pub fn scene_ron(&self, id: AssetId) -> Option<String> {
        self.scenes.get(&id)?.to_ron().ok()
    }

    /// Parses glTF bytes (`.glb` or `.gltf`) into a [`Model`](crate::Model)
    /// and stores it. Textured slots keep their scalar fallback until the
    /// GPU upload step learns images. The editor flattens with
    /// [`scene_from_model`](crate::scene_from_model) when it needs a
    /// [`Scene`].
    ///
    /// # Errors
    ///
    /// [`AssetError::Import`] tagged `"gltf"` (the typed
    /// [`ornis_gltf::ImportError`] is its source); emits
    /// [`AssetEvent::Failed`] and leaves the registry untouched.
    #[cfg(feature = "gltf")]
    pub fn load_gltf(&mut self, bytes: &[u8]) -> Result<AssetId, AssetError> {
        match crate::importer::GltfImporter.import_slice(bytes) {
            Ok(imported) => Ok(self.store(imported, None)),
            Err(error) => Err(self.fail(AssetKind::Model, None, error)),
        }
    }

    /// Reads a glTF file (resolving sibling `.bin` like
    /// [`ornis_gltf::load_path`]) and stores it as a [`Model`](crate::Model),
    /// remembering the path. Unlike [`AssetServer::load`] this always
    /// re-imports (the editor re-reads files on hot reload) and returns a
    /// fresh id; the path index then points at the newest id.
    ///
    /// # Errors
    ///
    /// [`AssetError`] (not found, IO, `"gltf"` import failure); emits
    /// [`AssetEvent::Failed`] and leaves the registry untouched.
    #[cfg(feature = "gltf")]
    pub fn load_gltf_file(&mut self, path: &Path) -> Result<AssetId, AssetError> {
        match crate::importer::GltfImporter.import_path(path) {
            Ok(imported) => Ok(self.store(imported, Some(path))),
            Err(error) => Err(self.fail(AssetKind::Model, Some(path), error)),
        }
    }

    /// Marks an asset dirty for hot reload; `false` for unknown ids.
    ///
    /// Programmatic invalidation API. The editor's file watcher drives
    /// reloads by re-reading files (see below); this set is for hosts
    /// that detect staleness another way.
    pub fn request_reload(&mut self, id: AssetId) -> bool {
        if !self.kinds.contains_key(&id) {
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

    /// Dispatches `path` to its importer and checks the produced kind;
    /// failures are recorded as [`AssetEvent::Failed`].
    fn import_path_as(
        &mut self,
        kind: AssetKind,
        path: &Path,
    ) -> Result<ImportedAsset, AssetError> {
        let result = self.registry.for_path(path).and_then(|importer| {
            if importer.kind() != kind {
                return Err(AssetError::WrongKind {
                    path: Some(path.to_path_buf()),
                    expected: kind,
                    found: importer.kind(),
                });
            }
            importer.import_path(path)
        });
        result.map_err(|error| self.fail(kind, Some(path), error))
    }

    /// Records a failure event and hands the error back.
    fn fail(&mut self, kind: AssetKind, path: Option<&Path>, error: AssetError) -> AssetError {
        self.events.push(AssetEvent::Failed {
            kind,
            path: path.map(Path::to_path_buf),
            error: error.clone(),
        });
        error
    }

    /// Stores a fresh asset under a new id, indexes its path and emits
    /// [`AssetEvent::Loaded`].
    fn store(&mut self, imported: ImportedAsset, path: Option<&Path>) -> AssetId {
        let id = AssetId { index: self.next };
        self.next = self.next.saturating_add(1);
        let kind = imported.kind();
        self.insert_at(id, imported);
        if let Some(path) = path {
            self.paths.insert(id, path.to_path_buf());
            self.by_path.insert(path_key(path), id);
        }
        self.events.push(AssetEvent::Loaded { id, kind });
        id
    }

    /// Writes (or overwrites) every storage lane of `id`.
    fn insert_at(&mut self, id: AssetId, imported: ImportedAsset) {
        self.kinds.insert(id, imported.kind());
        match imported {
            ImportedAsset::Scene(import) => {
                match import.source_text {
                    Some(text) => {
                        self.sources.insert(id, text);
                    }
                    None => {
                        self.sources.remove(&id);
                    }
                };
                #[cfg(feature = "gltf")]
                self.models.remove(&id);
                self.scenes.insert(id, import.scene);
            }
            #[cfg(feature = "gltf")]
            ImportedAsset::Model(model) => {
                self.sources.remove(&id);
                self.scenes.remove(&id);
                self.models.insert(id, model);
            }
        }
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
    use crate::scene::{
        CameraDesc, CameraProjection, EntityDesc, MaterialDesc, MeshDesc, TransformDesc,
    };
    use ornis_core::Engine;

    /// Shipped demo scene: five spheres over two directional lights.
    const DEMO_RON: &str = include_str!("../../../assets/scene.ron");
    /// Showcase demo scene: floor + wall + tower + rolling ball.
    const SHOWCASE_RON: &str = include_str!("../../../assets/demo_scene.ron");

    fn entity_desc(name: &str, x: f32) -> EntityDesc {
        EntityDesc {
            name: name.into(),
            transform: TransformDesc {
                translation: glam::Vec3::new(x, 0.0, 0.0),
                ..TransformDesc::IDENTITY
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
                metallic: ornis_core::Metallic::new(1.0),
            },
        }
    }

    fn two_entity_scene() -> Scene {
        Scene {
            name: "assets-test".into(),
            entities: vec![entity_desc("a", -1.0), entity_desc("b", 1.0)],
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
                path,
                error: event_error,
            } => {
                assert_eq!(*kind, AssetKind::Scene);
                assert!(path.is_none(), "in-memory load has no path");
                assert_eq!(*event_error, AssetError::from(error.clone()));
                assert_eq!(event_error.format(), Some("ron"));
                let source = std::error::Error::source(event_error).expect("typed source");
                assert_eq!(
                    source.downcast_ref::<SceneLoadError>(),
                    Some(&error),
                    "the typed RON error survives inside the event"
                );
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
    #[cfg(feature = "gltf")]
    const TRIANGLE_B64: &str = "AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAEAAAACAAAA";

    #[cfg(feature = "gltf")]
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

    #[cfg(feature = "gltf")]
    #[test]
    fn load_gltf_stores_model_end_to_end() {
        // Real bytes (not hand-built structs): JSON + base64 buffer.
        let mut server = AssetServer::new();
        let id = server
            .load_gltf(triangle_gltf_json().as_bytes())
            .expect("triangle gltf loads");
        let model = server.model(id).expect("stored model");
        assert_eq!(model.nodes.len(), 1);
        assert_eq!(model.nodes[0].name.as_deref(), Some("tri"));
        assert_eq!(model.primitives.len(), 1);
        let scene = crate::scene_from_model(model);
        assert_eq!(scene.entities.len(), 1);
        assert_eq!(scene.entities[0].name, "tri");
        assert_eq!(scene.entities[0].transform.translation, glam::Vec3::ZERO);
        assert!(matches!(
            scene.entities[0].mesh,
            crate::scene::MeshDesc::Custom { .. }
        ));
        assert!(matches!(
            scene.entities[0].material,
            crate::scene::MaterialDesc::Metal { .. }
        ));
        assert!(server.get_scene(id).is_none(), "glTF is not a Scene");
        assert_eq!(
            server.take_events(),
            vec![AssetEvent::Loaded {
                id,
                kind: AssetKind::Model
            }]
        );
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn load_gltf_keeps_the_node_on_the_model() {
        let mut server = AssetServer::new();
        let id = server
            .load_gltf(triangle_gltf_json().as_bytes())
            .expect("triangle gltf loads");
        let model = server.model(id).expect("model");
        assert_eq!(model.primitives.len(), 1);
        assert_eq!(model.primitives[0].node, crate::NodeIdx(0));
        assert_eq!(model.nodes[0].name.as_deref(), Some("tri"));
        assert!(model.skins.is_empty());
        assert!(server.get_scene(id).is_none());
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn ron_and_manual_loads_store_no_model() {
        let mut server = AssetServer::new();
        let ron = server.load_scene_ron(DEMO_RON).expect("demo scene loads");
        assert!(server.model(ron).is_none());
        let manual = server.load_scene(two_entity_scene(), None);
        assert!(server.model(manual).is_none());
        assert!(server.model(AssetId { index: 999 }).is_none());
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn load_gltf_garbage_is_a_typed_import_error() {
        // Broken bytes are a format-tagged `AssetError`, not a `String`:
        // the registry stays untouched, the typed `ImportError` survives
        // as the source and a `Failed` event fires (like RON).
        let mut server = AssetServer::new();
        let error = server
            .load_gltf(b"definitely not gltf")
            .expect_err("garbage bytes fail");
        assert_eq!(error.format(), Some("gltf"), "{error:?}");
        let source = std::error::Error::source(&error).expect("typed source");
        assert!(
            matches!(
                source.downcast_ref::<ornis_gltf::ImportError>(),
                Some(ornis_gltf::ImportError::Parse { .. })
            ),
            "{error:?}"
        );
        assert!(server.is_empty());
        assert_eq!(
            server.take_events(),
            vec![AssetEvent::Failed {
                kind: AssetKind::Model,
                path: None,
                error,
            }]
        );
    }

    /// Shipped starter mannequin (Quaternius UAL-1, CC0).
    #[cfg(feature = "gltf")]
    fn starter_glb() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/starter/ual1_standard.glb")
    }

    /// Writes `contents` into a fresh temp dir under `name`.
    fn temp_file(name: &str, contents: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ornis-assets-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(name);
        std::fs::write(&path, contents).expect("write temp file");
        path
    }

    #[test]
    fn generic_load_dispatches_ron_by_extension_and_dedups() {
        let path = temp_file("scene.RON", DEMO_RON.as_bytes());
        let mut server = AssetServer::new();
        let handle: Handle<Scene> = server.load(&path).expect("ron loads");
        assert_eq!(server.get(&handle).expect("stored").entities.len(), 5);
        assert_eq!(server.path(handle.id()), Some(path.as_path()));
        assert!(server.scene_ron(handle.id()).is_some());
        // Same path → same handle, no re-import and no second event.
        let again: Handle<Scene> = server.load(&path).expect("dedup");
        assert_eq!(again, handle);
        assert_eq!(server.scene_count(), 1);
        assert_eq!(
            server.take_events(),
            vec![AssetEvent::Loaded {
                id: handle.id(),
                kind: AssetKind::Scene
            }]
        );
        std::fs::remove_dir_all(path.parent().expect("dir")).expect("cleanup");
    }

    #[test]
    fn generic_load_rejects_unknown_extension_with_failed_event() {
        let mut server = AssetServer::new();
        let error = server
            .load::<Scene>("model.obj")
            .expect_err("no .obj importer");
        assert!(
            matches!(&error, AssetError::UnsupportedExtension { extension, .. } if extension == "obj"),
            "{error:?}"
        );
        assert!(server.is_empty());
        let events = server.take_events();
        assert!(
            matches!(&events[..], [AssetEvent::Failed { path: Some(_), .. }]),
            "{events:?}"
        );
    }

    #[test]
    fn generic_load_missing_file_is_not_found() {
        let mut server = AssetServer::new();
        let error = server
            .load::<Scene>("definitely/missing/scene.ron")
            .expect_err("missing file");
        assert!(matches!(error, AssetError::NotFound { .. }), "{error:?}");
        assert_eq!(server.take_events().len(), 1);
    }

    #[test]
    fn generic_load_parse_failure_is_format_tagged() {
        let path = temp_file("broken.ron", b"Scene(name: 42)");
        let mut server = AssetServer::new();
        let error = server.load::<Scene>(&path).expect_err("broken ron");
        assert_eq!(error.format(), Some("ron"));
        assert_eq!(error.path(), Some(path.as_path()));
        assert!(server.is_empty());
        std::fs::remove_dir_all(path.parent().expect("dir")).expect("cleanup");
    }

    #[test]
    fn reload_reimports_in_place_and_unload_releases() {
        let path = temp_file("scene.ron", DEMO_RON.as_bytes());
        let mut server = AssetServer::new();
        let handle: Handle<Scene> = server.load(&path).expect("ron loads");
        std::fs::write(&path, SHOWCASE_RON).expect("edit file");
        server.reload(&handle).expect("reload");
        assert_eq!(
            server.get(&handle).expect("same handle").name,
            "demo_showcase"
        );
        // A broken edit keeps the previous asset and reports Failed.
        std::fs::write(&path, "garbage").expect("break file");
        assert!(server.reload(&handle).is_err());
        assert_eq!(server.get(&handle).expect("kept").name, "demo_showcase");

        assert!(server.unload(&handle));
        assert!(server.get(&handle).is_none());
        assert!(server.path(handle.id()).is_none());
        assert!(!server.unload(&handle));
        // After unload the path loads fresh under a new id.
        std::fs::write(&path, DEMO_RON).expect("restore file");
        let fresh: Handle<Scene> = server.load(&path).expect("reload after unload");
        assert_ne!(fresh, handle);
        std::fs::remove_dir_all(path.parent().expect("dir")).expect("cleanup");
    }

    #[test]
    fn manual_loads_cannot_reload() {
        let mut server = AssetServer::new();
        let id = server.load_scene(two_entity_scene(), None);
        let handle = Handle::<Scene>::from_id(id);
        assert_eq!(
            server.reload(&handle),
            Err(AssetError::UnknownHandle { index: id.index() })
        );
    }

    #[test]
    fn handle_debug_names_kind() {
        let handle = Handle::<Scene>::from_id(AssetId { index: 7 });
        assert_eq!(format!("{handle:?}"), "Handle<scene>(7)");
        assert_eq!(AssetId::from(handle).index(), 7);
    }

    #[test]
    fn custom_importer_extends_the_registry() {
        struct Stub;
        impl Importer for Stub {
            fn format(&self) -> &'static str {
                "stub"
            }
            fn extensions(&self) -> &'static [&'static str] {
                &["stub"]
            }
            fn kind(&self) -> AssetKind {
                AssetKind::Scene
            }
            fn import_path(&self, _path: &Path) -> Result<ImportedAsset, AssetError> {
                Ok(ImportedAsset::Scene(SceneImport::new(two_entity_scene())))
            }
        }
        let mut server = AssetServer::new();
        server.register_importer(Stub);
        let handle: Handle<Scene> = server.load("anything.stub").expect("stub imports");
        assert_eq!(server.get(&handle).expect("stored").entities.len(), 2);
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn generic_load_glb_retains_source_and_path() {
        let path = starter_glb();
        let mut server = AssetServer::new();
        let handle: Handle<crate::Model> = server.load(&path).expect("starter loads");
        let model = server.get(&handle).expect("model");
        assert!(!model.nodes.is_empty());
        assert!(!model.skel_clips.is_empty(), "starter ships clips");
        assert_eq!(server.path(handle.id()), Some(path.as_path()));
        assert_eq!(server.load::<crate::Model>(&path).expect("dedup"), handle);
        let wrong = server.load::<Scene>(&path).expect_err("glb is not a scene");
        assert!(matches!(wrong, AssetError::WrongKind { .. }), "{wrong:?}");
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn load_gltf_file_records_path_and_emits_failed() {
        let mut server = AssetServer::new();
        let path = starter_glb();
        let id = server.load_gltf_file(&path).expect("starter loads");
        assert_eq!(server.path(id), Some(path.as_path()));
        // The generic loader reuses the legacy load through the path index.
        assert_eq!(server.load::<crate::Model>(&path).expect("dedup").id(), id);
        server.take_events();

        let missing = Path::new("definitely/missing.glb");
        let error = server.load_gltf_file(missing).expect_err("missing");
        assert!(matches!(error, AssetError::NotFound { .. }), "{error:?}");
        assert_eq!(
            server.take_events(),
            vec![AssetEvent::Failed {
                kind: AssetKind::Model,
                path: Some(missing.to_path_buf()),
                error,
            }]
        );
    }

    #[cfg(feature = "fbx")]
    #[test]
    fn fbx_feature_reports_unsupported_format() {
        let mut server = AssetServer::new();
        let error = server.load::<Scene>("model.fbx").expect_err("placeholder");
        assert!(
            matches!(error, AssetError::UnsupportedFormat { format: "fbx", .. }),
            "{error:?}"
        );
    }

    #[cfg(not(feature = "fbx"))]
    #[test]
    fn fbx_without_feature_is_unknown_extension() {
        let mut server = AssetServer::new();
        let error = server.load::<Scene>("model.fbx").expect_err("no importer");
        assert!(
            matches!(error, AssetError::UnsupportedExtension { .. }),
            "{error:?}"
        );
    }
}
