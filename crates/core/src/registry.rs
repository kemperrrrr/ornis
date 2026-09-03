//! Component registry — foundation of audit 2026-08-22 F0
//! (document `docs/quality/audit-2026-08-22.md`, §10): name ↔ [`TypeId`] ↔
//! type-erased operations over [`SmartStore`] lanes.
//!
//! Serves tooling paths: the editor's generic `SetComponent` (D2),
//! the scripting batch API (D1), scene serialization (phase 7), scheduler
//! lane granularity (`lane_id` — dense index for future access bitsets).
//! Hot per-frame loops **do not touch** the registry — they stay typed
//! (SoA lanes, `#[smart_pipeline]`); the boundary is the same as Bevy's
//! `bevy_reflect` vs typed queries.
//!
//! Thunks are monomorphized via plain generic registration — no procedural
//! macro required; derive sugar (`#[derive(RegisterComponent)]`) is an
//! optional next step that does not change the registry API.
//!
//! # Example
//!
//! ```rust
//! use ornis_core::{ComponentRegistry, SmartStore};
//!
//! let mut registry = ComponentRegistry::new();
//! registry.register::<f32>("health");
//!
//! let mut world = SmartStore::new();
//! let hero = world.create_entity();
//!
//! let meta = registry.by_name("health").unwrap();
//! meta.set_json(&mut world, hero, &serde_json::json!(100.0))
//!     .unwrap();
//! assert_eq!(
//!     meta.get_json(&world, hero).unwrap(),
//!     Some(serde_json::json!(100.0))
//! );
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

use serde::{Serialize, de::DeserializeOwned};

use crate::entity::Entity;
use crate::smart_store::SmartStore;

/// Dense lane index in the registry (0..len). Reserved for scheduler access
/// bitsets (audit §3.6) — stable within a single registry.
pub type LaneId = u32;

/// Error of a type-erased registry operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// JSON does not match the component schema (`set_json`) or the
    /// component is not serializable (`get_json`; practically unreachable
    /// for ordinary structs).
    Json(String),
}

impl RegistryError {
    fn from_json(error: serde_json::Error) -> Self {
        Self::Json(error.to_string())
    }
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::Json(message) => write!(f, "component JSON error: {message}"),
        }
    }
}

impl std::error::Error for RegistryError {}

type RegisterLaneFn = fn(&mut SmartStore);
type InsertAnyFn = fn(&mut SmartStore, Entity, Box<dyn Any>) -> bool;
type ContainsFn = fn(&SmartStore, Entity) -> bool;
type LaneLenFn = fn(&SmartStore) -> usize;
type RemoveFn = fn(&mut SmartStore, Entity) -> Option<Box<dyn Any>>;
type GetJsonFn = fn(&SmartStore, Entity) -> Result<Option<serde_json::Value>, RegistryError>;
type SetJsonFn = fn(&mut SmartStore, Entity, &serde_json::Value) -> Result<(), RegistryError>;
type ParseJsonFn = fn(&serde_json::Value) -> Result<Box<dyn Any>, RegistryError>;

fn register_lane_thunk<T>(store: &mut SmartStore)
where
    T: 'static + Send + Sync,
{
    store.register::<T>();
}

fn insert_any_thunk<T>(store: &mut SmartStore, entity: Entity, boxed: Box<dyn Any>) -> bool
where
    T: 'static + Clone + Send + Sync,
{
    let Ok(component) = boxed.downcast::<T>() else {
        return false;
    };
    store.insert(entity, *component);
    true
}

fn contains_thunk<T>(store: &SmartStore, entity: Entity) -> bool
where
    T: 'static + Send + Sync,
{
    store
        .read_lane::<T>()
        .is_some_and(|lane| lane.contains(entity))
}

fn lane_len_thunk<T>(store: &SmartStore) -> usize
where
    T: 'static + Send + Sync,
{
    store.read_lane::<T>().map_or(0, |lane| lane.len())
}

fn remove_thunk<T>(store: &mut SmartStore, entity: Entity) -> Option<Box<dyn Any>>
where
    T: 'static + Send + Sync,
{
    store
        .write_lane::<T>()
        .and_then(|mut lane| lane.remove(entity))
        .map(|component| Box::new(component) as Box<dyn Any>)
}

fn get_json_thunk<T>(store: &SmartStore, entity: Entity) -> GetJsonResult
where
    T: 'static + Send + Sync + Serialize,
{
    let Some(lane) = store.read_lane::<T>() else {
        return Ok(None);
    };
    let Some(component) = lane.get(entity) else {
        return Ok(None);
    };
    serde_json::to_value(component)
        .map(Some)
        .map_err(RegistryError::from_json)
}

fn set_json_thunk<T>(store: &mut SmartStore, entity: Entity, value: &serde_json::Value) -> SetResult
where
    T: 'static + Clone + Send + Sync + DeserializeOwned,
{
    let component: T = serde_json::from_value(value.clone()).map_err(RegistryError::from_json)?;
    store.insert(entity, component);
    Ok(())
}

fn parse_json_thunk<T>(value: &serde_json::Value) -> Result<Box<dyn Any>, RegistryError>
where
    T: 'static + DeserializeOwned,
{
    let component: T = serde_json::from_value(value.clone()).map_err(RegistryError::from_json)?;
    Ok(Box::new(component))
}

type GetJsonResult = Result<Option<serde_json::Value>, RegistryError>;
type SetResult = Result<(), RegistryError>;

/// Type-erased component record: name ↔ type ↔ operations over its lane.
///
/// All operations delegate to monomorphic thunks created at
/// [`ComponentRegistry::register`]; the struct is `Send + Sync` (fn pointers
/// and `&'static str`), so the registry can be shared across threads (`Arc`).
pub struct ComponentMeta {
    name: &'static str,
    type_name: &'static str,
    type_id: TypeId,
    lane_id: LaneId,
    register_lane: RegisterLaneFn,
    insert_any: InsertAnyFn,
    contains: ContainsFn,
    lane_len: LaneLenFn,
    remove: RemoveFn,
    get_json: GetJsonFn,
    set_json: SetJsonFn,
    parse_json: ParseJsonFn,
}

impl ComponentMeta {
    /// Short name from registration (protocol key: JSON/FFI/scenes).
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Full Rust type path (diagnostics, not a protocol key).
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// [`TypeId`] of the component — lane key in [`SmartStore`].
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// Dense lane index in the registry (see [`LaneId`]).
    pub fn lane_id(&self) -> LaneId {
        self.lane_id
    }

    /// Creates an empty lane in the world if it does not exist yet.
    pub fn register_lane(&self, store: &mut SmartStore) {
        (self.register_lane)(store)
    }

    /// Inserts a boxed component. `false` if the boxed type is not `T`
    /// (caller contract violation — the registry itself never creates such a call).
    pub fn insert_any(&self, store: &mut SmartStore, entity: Entity, boxed: Box<dyn Any>) -> bool {
        (self.insert_any)(store, entity, boxed)
    }

    /// Whether the entity has the component (taking the handle generation into account).
    pub fn contains(&self, store: &SmartStore, entity: Entity) -> bool {
        (self.contains)(store, entity)
    }

    /// Number of live components in the lane (0 if the lane does not exist yet).
    pub fn lane_len(&self, store: &SmartStore) -> usize {
        (self.lane_len)(store)
    }

    /// Removes and returns the component as `Box<dyn Any>` (None — not present).
    pub fn remove(&self, store: &mut SmartStore, entity: Entity) -> Option<Box<dyn Any>> {
        (self.remove)(store, entity)
    }

    /// Snapshot of the component as JSON (None — the entity has none).
    pub fn get_json(
        &self,
        store: &SmartStore,
        entity: Entity,
    ) -> Result<Option<serde_json::Value>, RegistryError> {
        (self.get_json)(store, entity)
    }

    /// Upserts the component from JSON: deserializes and inserts (semantics
    /// of `SmartStore::insert` — an existing component is overwritten).
    pub fn set_json(
        &self,
        store: &mut SmartStore,
        entity: Entity,
        value: &serde_json::Value,
    ) -> Result<(), RegistryError> {
        (self.set_json)(store, entity, value)
    }

    /// Deserializes the component from JSON into `Box<dyn Any>` — without
    /// touching the world. Paired with [`ComponentMeta::insert_any`] it
    /// provides "parse first, then mutate" semantics: the caller validates
    /// all command payloads before any single world write (the editor
    /// protocol invariant "command error does not touch the world").
    pub fn parse_json(&self, value: &serde_json::Value) -> Result<Box<dyn Any>, RegistryError> {
        (self.parse_json)(value)
    }
}

/// Component registry: built once at startup (`register::<T>(name)`
/// for each type), then read-only and shareable (`Arc`).
///
/// Registration order determines [`LaneId`] — for reproducible protocols
/// register in a fixed order.
#[derive(Default)]
pub struct ComponentRegistry {
    by_id: HashMap<TypeId, LaneId>,
    by_name: HashMap<&'static str, LaneId>,
    entries: Vec<ComponentMeta>,
}

impl ComponentRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a component type under the protocol name `name`.
    ///
    /// Core operations (lane, contains, remove) do not require serde;
    /// `get_json`/`set_json` are monomorphized over `Serialize`/
    /// `DeserializeOwned` of the same type — the "reflection only for
    /// tooling" boundary is enforced by the caller's bounds.
    ///
    /// # Panics
    /// Panics on duplicate registration of the same type or an occupied
    /// name — this is a configuration error, not a runtime condition.
    pub fn register<T>(&mut self, name: &'static str) -> &mut Self
    where
        T: 'static + Clone + Send + Sync + Serialize + DeserializeOwned,
    {
        let type_id = TypeId::of::<T>();
        assert!(
            !self.by_id.contains_key(&type_id),
            "component type `{}` is already registered",
            std::any::type_name::<T>()
        );
        assert!(
            !self.by_name.contains_key(name),
            "component name `{name}` is already registered"
        );

        let lane_id = self.entries.len() as LaneId;
        self.entries.push(ComponentMeta {
            name,
            type_name: std::any::type_name::<T>(),
            type_id,
            lane_id,
            register_lane: register_lane_thunk::<T>,
            insert_any: insert_any_thunk::<T>,
            contains: contains_thunk::<T>,
            lane_len: lane_len_thunk::<T>,
            remove: remove_thunk::<T>,
            get_json: get_json_thunk::<T>,
            set_json: set_json_thunk::<T>,
            parse_json: parse_json_thunk::<T>,
        });
        self.by_id.insert(type_id, lane_id);
        self.by_name.insert(name, lane_id);
        self
    }

    /// Entry by type.
    pub fn by_id(&self, type_id: TypeId) -> Option<&ComponentMeta> {
        self.by_id
            .get(&type_id)
            .map(|&id| &self.entries[id as usize])
    }

    /// Entry by protocol name.
    pub fn by_name(&self, name: &str) -> Option<&ComponentMeta> {
        self.by_name.get(name).map(|&id| &self.entries[id as usize])
    }

    /// Entry by dense lane index.
    pub fn by_lane_id(&self, lane_id: LaneId) -> Option<&ComponentMeta> {
        self.entries.get(lane_id as usize)
    }

    /// All entries in registration order.
    pub fn iter(&self) -> std::slice::Iter<'_, ComponentMeta> {
        self.entries.iter()
    }

    /// Number of registered types.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Health {
        hp: u32,
    }

    fn registry_with_two() -> ComponentRegistry {
        let mut registry = ComponentRegistry::new();
        registry.register::<Position>("position");
        registry.register::<Health>("health");
        registry
    }

    #[test]
    fn lookup_by_name_and_id_and_lane_id() {
        let registry = registry_with_two();

        let pos = registry.by_name("position").expect("position");
        assert_eq!(pos.type_id(), TypeId::of::<Position>());
        assert_eq!(pos.type_name(), std::any::type_name::<Position>());
        assert_eq!(pos.lane_id(), 0);

        let health = registry.by_id(TypeId::of::<Health>()).expect("health");
        assert_eq!(health.name(), "health");
        assert_eq!(health.lane_id(), 1);
        assert!(registry.by_lane_id(1).is_some());

        // LaneId is a dense projection: by_lane_id and lookup coincide.
        assert!(std::ptr::eq(registry.by_lane_id(0).unwrap(), pos));
        assert!(registry.by_name("ghost").is_none());
        assert!(registry.by_id(TypeId::of::<u8>()).is_none());
        assert!(registry.by_lane_id(2).is_none());

        assert_eq!(registry.len(), 2);
        assert!(!registry.is_empty());
        let names: Vec<_> = registry.iter().map(|meta| meta.name()).collect();
        assert_eq!(names, vec!["position", "health"]);
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn duplicate_type_panics() {
        let mut registry = ComponentRegistry::new();
        registry.register::<Position>("position");
        registry.register::<Position>("pos2");
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn duplicate_name_panics() {
        let mut registry = ComponentRegistry::new();
        registry.register::<Position>("component");
        registry.register::<Health>("component");
    }

    #[test]
    fn register_lane_creates_empty_lane_eagerly() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let meta = registry.by_name("position").unwrap();

        meta.register_lane(&mut store);
        assert_eq!(meta.lane_len(&store), 0);
        // The lane was actually created: typed access is already possible.
        assert!(store.read_lane::<Position>().is_some());
    }

    #[test]
    fn insert_any_then_contains_and_len() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("position").unwrap();

        assert!(!meta.contains(&store, entity));
        assert_eq!(meta.lane_len(&store), 0);

        let inserted = meta.insert_any(&mut store, entity, Box::new(Position { x: 1.0, y: 2.0 }));
        assert!(inserted);
        assert!(meta.contains(&store, entity));
        assert_eq!(meta.lane_len(&store), 1);

        // The typed path sees the same value.
        let lane = store.read_lane::<Position>().unwrap();
        assert_eq!(lane.get(entity), Some(&Position { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn insert_any_with_wrong_box_type_returns_false() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("position").unwrap();

        // Box of a different type: insert must not happen.
        let inserted = meta.insert_any(&mut store, entity, Box::new(Health { hp: 5 }));
        assert!(!inserted);
        assert!(!meta.contains(&store, entity));
        assert_eq!(meta.lane_len(&store), 0);
        // And the Health lane was not touched by the foreign insert.
        let health = registry.by_name("health").unwrap();
        assert_eq!(health.lane_len(&store), 0);
    }

    #[test]
    fn set_json_upserts_and_get_json_roundtrips() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("position").unwrap();

        meta.set_json(&mut store, entity, &json!({"x": 1.5, "y": -2.0}))
            .unwrap();
        assert_eq!(
            meta.get_json(&store, entity).unwrap(),
            Some(json!({"x": 1.5, "y": -2.0}))
        );
        assert_eq!(meta.lane_len(&store), 1);

        // Repeated set_json — overwrite without growing the lane.
        meta.set_json(&mut store, entity, &json!({"x": 0.0, "y": 7.25}))
            .unwrap();
        assert_eq!(
            meta.get_json(&store, entity).unwrap(),
            Some(json!({"x": 0.0, "y": 7.25}))
        );
        assert_eq!(meta.lane_len(&store), 1);
    }

    #[test]
    fn set_json_schema_mismatch_is_json_error() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("health").unwrap();

        // Missing field `hp`.
        let missing = meta.set_json(&mut store, entity, &json!({"mana": 5}));
        assert!(matches!(missing, Err(RegistryError::Json(_))));
        // Field type mismatch.
        let wrong_type = meta.set_json(&mut store, entity, &json!({"hp": "full"}));
        assert!(matches!(wrong_type, Err(RegistryError::Json(_))));
        // i32 does not fit into u32.
        let negative = meta.set_json(&mut store, entity, &json!({"hp": -1}));
        assert!(matches!(negative, Err(RegistryError::Json(_))));

        assert!(!meta.contains(&store, entity));
    }

    #[test]
    fn parse_json_validates_before_insert_any() {
        let registry = registry_with_two();
        let position = registry.by_name("position").unwrap();
        let mut store = SmartStore::new();
        let entity = store.create_entity();

        // The parsed box is inserted and read back.
        let boxed = position.parse_json(&json!({"x": 1.0, "y": 2.0})).unwrap();
        assert!(position.insert_any(&mut store, entity, boxed));
        let lane = store.read_lane::<Position>().unwrap();
        assert_eq!(lane.get(entity), Some(&Position { x: 1.0, y: 2.0 }));

        // Schema mismatch — error before any world mutation.
        let bad = position.parse_json(&json!({"x": "left", "y": 0.0}));
        assert!(matches!(bad, Err(RegistryError::Json(_))));
        assert_eq!(position.lane_len(&store), 1);
    }

    #[test]
    fn get_json_on_absent_component_is_ok_none() {
        let registry = registry_with_two();
        let store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("position").unwrap();

        assert_eq!(meta.get_json(&store, entity).unwrap(), None);
    }

    #[test]
    fn remove_returns_boxed_component_and_clears() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("position").unwrap();

        meta.insert_any(&mut store, entity, Box::new(Position { x: 3.0, y: 4.0 }));
        let boxed = meta.remove(&mut store, entity).expect("component");
        let position = boxed.downcast::<Position>().expect("position type");
        assert_eq!(*position, Position { x: 3.0, y: 4.0 });

        assert!(!meta.contains(&store, entity));
        assert!(meta.remove(&mut store, entity).is_none());
    }

    #[test]
    fn destroyed_entity_has_no_components() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("position").unwrap();
        meta.insert_any(&mut store, entity, Box::new(Position { x: 1.0, y: 1.0 }));

        store.destroy_entity(entity);
        assert!(!meta.contains(&store, entity));

        // Fresh entity with a recycled id — empty.
        let recycled = store.create_entity();
        assert_ne!(recycled.generation(), entity.generation());
        assert!(!meta.contains(&store, recycled));
    }

    #[test]
    fn components_of_different_types_are_isolated() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let pos = registry.by_name("position").unwrap();
        let health = registry.by_name("health").unwrap();

        pos.set_json(&mut store, entity, &json!({"x": 1.0, "y": 2.0}))
            .unwrap();
        health
            .set_json(&mut store, entity, &json!({"hp": 100}))
            .unwrap();

        assert_eq!(pos.lane_len(&store), 1);
        assert_eq!(health.lane_len(&store), 1);

        // Removing one type does not touch the other.
        health.remove(&mut store, entity);
        assert!(!health.contains(&store, entity));
        assert!(pos.contains(&store, entity));
    }
}
