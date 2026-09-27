//! Component registry — foundation of audit 2026-08-22 F0
//! (document `docs/quality/audit-2026-08-22.md`, §10): name ↔ [`TypeId`] ↔
//! type-erased operations over [`SmartStore`] lanes.
//!
//! Serves tooling paths: the editor's generic `SetComponent` (D2),
//! the mutation-producer batch protocol, scene serialization (phase 7), scheduler
//! lane granularity (`lane_id` — dense index for future access bitsets).
//! Hot per-frame loops **do not touch** the registry — they stay typed
//! (SoA lanes, `#[smart_pipeline]`); the boundary is the same as Bevy's
//! `bevy_reflect` vs typed queries.
//!
//! Thunks are monomorphized via plain generic registration; derive sugar
//! (`#[derive(RegisterComponent)]` from `ornis-macros`) implements
//! [`RegisterComponent`] and lets callers use `register_component::<T>()`
//! without repeating the protocol name.
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

use serde::{Serialize, de::DeserializeOwned};

use crate::entity::Entity;
use crate::smart_store::SmartStore;

/// Dense lane index in the registry (0..len). Reserved for scheduler access
/// bitsets (audit §3.6) — stable within a single registry.
///
/// Newtype (not a `u32` alias) so lane indices never mix with entity ids,
/// generations or resource ordinals at the type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LaneId(pub u32);

impl LaneId {
    /// Wraps a raw dense index.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Raw dense index.
    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// Raw dense index as `usize` (registry table lookup).
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for LaneId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<LaneId> for u32 {
    fn from(id: LaneId) -> Self {
        id.0
    }
}

impl From<LaneId> for usize {
    fn from(id: LaneId) -> Self {
        id.0 as usize
    }
}

impl std::fmt::Display for LaneId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lane{}", self.0)
    }
}

/// Canonical component protocol name: the `by_name` registry key.
///
/// Owned (`Box<str>`) so both `'static` registrations and dynamic
/// wire/JSON names share one type; `by_name` lookups accept `&str`
/// through the [`std::borrow::Borrow`] impl, so existing call sites keep
/// working while producers (`Mutation::Set`, editor payloads) carry the
/// typed name instead of a bare `String`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComponentName(pub(crate) Box<str>);

impl ComponentName {
    /// Wraps a `'static` protocol name without allocating.
    pub fn from_static(name: &'static str) -> Self {
        Self(name.into())
    }

    /// Protocol name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::borrow::Borrow<str> for ComponentName {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for ComponentName {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for ComponentName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for ComponentName {
    fn from(name: &str) -> Self {
        Self(name.into())
    }
}

impl From<String> for ComponentName {
    fn from(name: String) -> Self {
        Self(name.into_boxed_str())
    }
}

impl From<&String> for ComponentName {
    fn from(name: &String) -> Self {
        Self(name.as_str().into())
    }
}

impl serde::Serialize for ComponentName {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> serde::Deserialize<'de> for ComponentName {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer).map(Self::from)
    }
}

/// Validated dotted field path (`position`, `velocity.0`): the address of one
/// value inside a component's canonical JSON form.
///
/// Newtype (not a `String` alias) so paths never mix with component names or
/// raw wire strings at the type level. Wire input enters through
/// [`FieldPath::parse`]; compile-time constants use
/// [`FieldPath::from_static`] (trusted, like `COMPONENT_NAME`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldPath(pub(crate) Box<str>);

impl FieldPath {
    /// Parses and validates a wire path: non-empty, dot-separated segments,
    /// each either an identifier (`[A-Za-z_][A-Za-z0-9_]*`, a JSON object
    /// key) or an array index (digits, no leading zeros — a JSON array slot).
    pub fn parse(path: &str) -> Result<Self, FieldPathError> {
        if path.is_empty() {
            return Err(FieldPathError::Empty);
        }
        for segment in path.split('.') {
            if !is_path_segment(segment) {
                return Err(FieldPathError::BadSegment {
                    path: path.to_string(),
                    segment: segment.to_string(),
                });
            }
        }
        Ok(Self(path.into()))
    }

    /// Wraps a compile-time path without allocating. Trusted input only —
    /// debug builds re-validate.
    pub fn from_static(path: &'static str) -> Self {
        debug_assert!(
            Self::parse(path).is_ok(),
            "invalid static field path `{path}`"
        );
        Self(path.into())
    }

    /// Path as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Segments left to right, classified as object keys or array indices.
    pub fn segments(&self) -> impl Iterator<Item = FieldSegment<'_>> {
        self.as_str().split('.').map(|segment| {
            if segment.bytes().next().is_some_and(|b| b.is_ascii_digit()) {
                FieldSegment::Index(segment.parse().unwrap_or(usize::MAX))
            } else {
                FieldSegment::Field(segment)
            }
        })
    }
}

impl std::fmt::Display for FieldPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::borrow::Borrow<str> for FieldPath {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for FieldPath {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl serde::Serialize for FieldPath {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> serde::Deserialize<'de> for FieldPath {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).map_err(serde::de::Error::custom)
    }
}

/// One segment of a [`FieldPath`]: a JSON object key or an array index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldSegment<'a> {
    /// Object key.
    Field(&'a str),
    /// Array index.
    Index(usize),
}

/// Rejection of [`FieldPath::parse`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FieldPathError {
    /// Path is empty.
    #[error("field path is empty")]
    Empty,
    /// A dot-separated segment is neither an identifier nor an index.
    #[error("invalid segment `{segment}` in field path `{path}`")]
    BadSegment {
        /// Whole offending path.
        path: String,
        /// Offending segment.
        segment: String,
    },
}

fn is_path_segment(segment: &str) -> bool {
    if segment.is_empty() {
        return false;
    }
    if segment.bytes().all(|b| b.is_ascii_digit()) {
        return segment.len() == 1 || !segment.starts_with('0');
    }
    let mut bytes = segment.bytes();
    let first = bytes.next().unwrap_or(b' ');
    (first.is_ascii_alphabetic() || first == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// One addressable field of a registered component: schema entry for
/// field-level reflection (listing, granular edits, sync mappings).
///
/// Emitted by `#[derive(RegisterComponent)]` for named-field structs;
/// hand-written for custom field surfaces (non-serde types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldMeta {
    /// Top-level field name (canonical JSON object key).
    pub name: &'static str,
    /// Canonical dotted path from the component root (equals `name` for
    /// top-level fields).
    pub path: &'static str,
    /// Rust type of the field, `stringify!`d at derive time (diagnostics only).
    pub type_name: &'static str,
    /// Whether the field accepts writes through the field surface.
    /// Solver-owned state (positions, masses) is readable but not writable.
    pub writable: bool,
}

/// Error of a type-erased registry operation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RegistryError {
    /// JSON does not match the component schema (`set_json`) or the
    /// component is not serializable (`get_json`; practically unreachable
    /// for ordinary structs).
    #[error("component JSON error: {0}")]
    Json(String),
    /// Component name is not registered.
    #[error("unknown component `{0}`")]
    UnknownComponent(ComponentName),
    /// The entity has no such component — field access cannot construct
    /// the missing remainder (lifecycle stays with whole-component writes).
    #[error("entity {id}g{generation} has no `{component}` component")]
    MissingComponent {
        /// Protocol name of the absent component.
        component: ComponentName,
        /// Entity id.
        id: u32,
        /// Entity generation.
        generation: u32,
    },
    /// No value lives at the path (absent key, out-of-bounds index, or a
    /// guard rejected the address, e.g. a non-whitelisted custom-surface path).
    #[error("unknown field `{path}` on component `{component}`")]
    UnknownField {
        /// Protocol name of the component.
        component: ComponentName,
        /// Requested path.
        path: FieldPath,
    },
    /// The lane was never created in this world — field access (unlike
    /// whole-value writes) never creates lanes. Sync harnesses treat it
    /// as "nothing to carry yet"; explicit edits should register the
    /// lane first (or create the component with a whole-value write).
    #[error("lane `{component}` is not registered in this world")]
    MissingLane {
        /// Protocol name of the component.
        component: ComponentName,
    },
}

impl RegistryError {
    /// Wraps a `serde_json` failure (custom field surfaces reuse it so
    /// leaf-type mismatches report identically to the JSON surface).
    pub fn from_json(error: serde_json::Error) -> Self {
        Self::Json(error.to_string())
    }
}

/// Marker for components that have a canonical registry name.
///
/// Implemented by `#[derive(RegisterComponent)]` (with optional
/// `#[component(name = "Custom")]`). Lets the registry infer the protocol
/// name via [`ComponentRegistry::register_component`].
pub trait RegisterComponent: 'static + Clone + Send + Sync + Serialize + DeserializeOwned {
    /// Canonical protocol name for this component (`by_name` key).
    const COMPONENT_NAME: &'static str;
    /// Addressable top-level fields, emitted by the derive for named-field
    /// structs. Empty by default (manually registered or opaque types):
    /// listing degrades to nothing, navigation still works.
    const FIELDS: &'static [FieldMeta] = &[];
}

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

/// Iterates the entities present in the lane, in dense order (deterministic).
type VisitFn = fn(&SmartStore, &mut dyn FnMut(Entity));
/// Reads one field value (`None` — the entity has no such component).
/// The component name travels along because monomorphic thunks cannot
/// name their own component for diagnostics.
type GetFieldFn = fn(
    &SmartStore,
    Entity,
    &ComponentName,
    &FieldPath,
) -> Result<Option<serde_json::Value>, RegistryError>;
/// Writes one field value, returning whether a write happened (`false` — a
/// surface guard skipped it, e.g. solver-owned state on a static body).
/// Shared store access (lane `RwLock` guards) so schedule systems can call
/// it — unlike whole-value writes, field access never creates lanes.
type SetFieldFn = fn(
    &SmartStore,
    Entity,
    &ComponentName,
    &FieldPath,
    &serde_json::Value,
) -> Result<bool, RegistryError>;

fn visit_thunk<T>(store: &SmartStore, visit: &mut dyn FnMut(Entity))
where
    T: 'static + Send + Sync,
{
    if let Some(lane) = store.read_lane::<T>() {
        for &entity in &lane.entities {
            visit(entity);
        }
    }
}

/// Navigates the canonical JSON form along `path` (read direction).
fn navigate<'v>(value: &'v serde_json::Value, path: &FieldPath) -> Option<&'v serde_json::Value> {
    let mut current = value;
    for segment in path.segments() {
        current = match (segment, current) {
            (FieldSegment::Field(key), serde_json::Value::Object(map)) => map.get(key)?,
            (FieldSegment::Index(i), serde_json::Value::Array(items)) => items.get(i)?,
            _ => return None,
        };
    }
    Some(current)
}

/// Navigates the canonical JSON form along `path` (write direction, strict:
/// missing intermediates are an error, never auto-created).
fn navigate_mut<'v>(
    value: &'v mut serde_json::Value,
    path: &FieldPath,
) -> Option<&'v mut serde_json::Value> {
    let mut current = value;
    for segment in path.segments() {
        current = match (segment, current) {
            (FieldSegment::Field(key), serde_json::Value::Object(map)) => map.get_mut(key)?,
            (FieldSegment::Index(i), serde_json::Value::Array(items)) => items.get_mut(i)?,
            _ => return None,
        };
    }
    Some(current)
}

fn missing_lane(component: &ComponentName) -> RegistryError {
    RegistryError::MissingLane {
        component: component.clone(),
    }
}

fn get_field_json_thunk<T>(
    store: &SmartStore,
    entity: Entity,
    component: &ComponentName,
    path: &FieldPath,
) -> Result<Option<serde_json::Value>, RegistryError>
where
    T: 'static + Send + Sync + Serialize,
{
    let Some(root) = get_json_thunk::<T>(store, entity)? else {
        return Ok(None);
    };
    navigate(&root, path)
        .cloned()
        .map(Some)
        .ok_or(RegistryError::UnknownField {
            component: component.clone(),
            path: path.clone(),
        })
}

fn set_field_json_thunk<T>(
    store: &SmartStore,
    entity: Entity,
    component: &ComponentName,
    path: &FieldPath,
    value: &serde_json::Value,
) -> Result<bool, RegistryError>
where
    T: 'static + Clone + Send + Sync + Serialize + DeserializeOwned,
{
    let Some(lane) = store.read_lane::<T>() else {
        return Err(missing_lane(component));
    };
    let Some(current) = lane.get(entity) else {
        return Err(RegistryError::MissingComponent {
            component: component.clone(),
            id: entity.id(),
            generation: entity.generation(),
        });
    };
    let mut root = serde_json::to_value(current).map_err(RegistryError::from_json)?;
    let slot = navigate_mut(&mut root, path).ok_or(RegistryError::UnknownField {
        component: component.clone(),
        path: path.clone(),
    })?;
    *slot = value.clone();
    // Full re-parse before any write: a mistyped leaf rejects here.
    let parsed: T = serde_json::from_value(root).map_err(RegistryError::from_json)?;
    drop(lane);
    let Some(mut lane) = store.write_lane::<T>() else {
        return Err(missing_lane(component));
    };
    lane.insert(entity, parsed);
    Ok(true)
}

fn get_json_unsupported(
    _store: &SmartStore,
    _entity: Entity,
) -> Result<Option<serde_json::Value>, RegistryError> {
    Err(RegistryError::Json(
        "component has no JSON form (custom field surface registered; use field paths)".to_string(),
    ))
}

fn set_json_unsupported(
    _store: &mut SmartStore,
    _entity: Entity,
    _value: &serde_json::Value,
) -> Result<(), RegistryError> {
    Err(RegistryError::Json(
        "component has no JSON form (custom field surface registered; use field paths)".to_string(),
    ))
}

fn parse_json_unsupported(_value: &serde_json::Value) -> Result<Box<dyn Any>, RegistryError> {
    Err(RegistryError::Json(
        "component has no JSON form (custom field surface registered; use field paths)".to_string(),
    ))
}

/// Field surface for a non-serde type (solver state with invariants, e.g.
/// body lanes): schema plus hand-written addressable access over the typed
/// lane. Guards live in the thunks (a static body skips velocity writes
/// with `Ok(false)`), so one harness serves every engine behind the same
/// vocabulary. Whole-value JSON stays unsupported for such types.
pub struct FieldSurface {
    /// Addressable fields (readable, and writable where flagged).
    pub fields: &'static [FieldMeta],
    /// Reads one field value (`None` — the entity has no such component).
    pub get_field: GetFieldFn,
    /// Writes one field value (`false` — skipped by a surface guard).
    pub set_field: SetFieldFn,
}

type GetJsonResult = Result<Option<serde_json::Value>, RegistryError>;
type SetResult = Result<(), RegistryError>;

/// Type-erased component record: name ↔ type ↔ operations over its lane.
///
/// All operations delegate to monomorphic thunks created at
/// [`ComponentRegistry::register`]; the struct is `Send + Sync` (fn pointers
/// and owned [`ComponentName`]), so the registry can be shared across threads (`Arc`).
pub struct ComponentMeta {
    name: ComponentName,
    type_name: &'static str,
    type_id: TypeId,
    lane_id: LaneId,
    fields: &'static [FieldMeta],
    visit: VisitFn,
    register_lane: RegisterLaneFn,
    insert_any: InsertAnyFn,
    contains: ContainsFn,
    lane_len: LaneLenFn,
    remove: RemoveFn,
    get_json: GetJsonFn,
    set_json: SetJsonFn,
    parse_json: ParseJsonFn,
    get_field: GetFieldFn,
    set_field: SetFieldFn,
}

impl ComponentMeta {
    /// Short name from registration (protocol key: JSON/FFI/scenes).
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Typed protocol name (same key as [`ComponentMeta::name`]).
    pub fn component_name(&self) -> &ComponentName {
        &self.name
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

    /// Addressable field schema. Empty for manually registered or opaque
    /// types (listing degrades to nothing — navigation still works for
    /// JSON-backed components).
    pub fn fields(&self) -> &[FieldMeta] {
        self.fields
    }

    /// Calls `visit` for every entity carrying the component, in dense
    /// order (deterministic — the harness relies on it).
    pub fn visit_entities(&self, store: &SmartStore, visit: &mut dyn FnMut(Entity)) {
        (self.visit)(store, visit);
    }

    /// Reads one field value through the field surface
    /// (`None` — the entity has no such component). Shared store access,
    /// so schedule systems can call it.
    pub fn get_field(
        &self,
        store: &SmartStore,
        entity: Entity,
        path: &FieldPath,
    ) -> Result<Option<serde_json::Value>, RegistryError> {
        (self.get_field)(store, entity, &self.name, path)
    }

    /// Writes one field value, returning whether a write happened (`false`
    /// — a surface guard skipped it). Shared store access (lane `RwLock`
    /// guards), so schedule systems can call it; unlike whole-value
    /// writes, field access never creates lanes
    /// ([`RegistryError::MissingLane`]).
    ///
    /// # Errors
    /// [`RegistryError::MissingComponent`] when the entity lacks the
    /// component — field access never constructs the missing remainder
    /// (lifecycle stays with whole-component writes).
    pub fn set_field(
        &self,
        store: &SmartStore,
        entity: Entity,
        path: &FieldPath,
        value: &serde_json::Value,
    ) -> Result<bool, RegistryError> {
        (self.set_field)(store, entity, &self.name, path, value)
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
    by_name: HashMap<ComponentName, LaneId>,
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
        let key = ComponentName::from_static(name);
        assert!(
            !self.by_id.contains_key(&type_id),
            "component type `{}` is already registered",
            std::any::type_name::<T>()
        );
        assert!(
            !self.by_name.contains_key(key.as_str()),
            "component name `{name}` is already registered"
        );

        let lane_id = LaneId(self.entries.len() as u32);
        self.entries.push(ComponentMeta {
            name: key.clone(),
            type_name: std::any::type_name::<T>(),
            type_id,
            lane_id,
            fields: &[],
            visit: visit_thunk::<T>,
            register_lane: register_lane_thunk::<T>,
            insert_any: insert_any_thunk::<T>,
            contains: contains_thunk::<T>,
            lane_len: lane_len_thunk::<T>,
            remove: remove_thunk::<T>,
            get_json: get_json_thunk::<T>,
            set_json: set_json_thunk::<T>,
            parse_json: parse_json_thunk::<T>,
            get_field: get_field_json_thunk::<T>,
            set_field: set_field_json_thunk::<T>,
        });
        self.by_id.insert(type_id, lane_id);
        self.by_name.insert(key, lane_id);
        self
    }

    /// Registers a component type with an explicit field schema (for
    /// schema-first types whose derive is unavailable). Navigation works
    /// identically to [`ComponentRegistry::register`].
    ///
    /// # Panics
    /// Same duplicate-registration panics as [`ComponentRegistry::register`].
    pub fn register_with_fields<T>(
        &mut self,
        name: &'static str,
        fields: &'static [FieldMeta],
    ) -> &mut Self
    where
        T: 'static + Clone + Send + Sync + Serialize + DeserializeOwned,
    {
        self.register::<T>(name);
        if let Some(meta) = self.entries.last_mut() {
            meta.fields = fields;
        }
        self
    }

    /// Registers a non-serde type behind a hand-written [`FieldSurface`]
    /// (solver state with invariants): lane operations stay generic over
    /// `T`, whole-value JSON reports "no JSON form", field access goes to
    /// the surface thunks.
    ///
    /// # Panics
    /// Same duplicate-registration panics as [`ComponentRegistry::register`].
    pub fn register_field_surface<T>(
        &mut self,
        name: &'static str,
        surface: FieldSurface,
    ) -> &mut Self
    where
        T: 'static + Clone + Send + Sync,
    {
        let type_id = TypeId::of::<T>();
        let key = ComponentName::from_static(name);
        assert!(
            !self.by_id.contains_key(&type_id),
            "component type `{}` is already registered",
            std::any::type_name::<T>()
        );
        assert!(
            !self.by_name.contains_key(key.as_str()),
            "component name `{name}` is already registered"
        );

        let lane_id = LaneId(self.entries.len() as u32);
        self.entries.push(ComponentMeta {
            name: key.clone(),
            type_name: std::any::type_name::<T>(),
            type_id,
            lane_id,
            fields: surface.fields,
            visit: visit_thunk::<T>,
            register_lane: register_lane_thunk::<T>,
            insert_any: insert_any_thunk::<T>,
            contains: contains_thunk::<T>,
            lane_len: lane_len_thunk::<T>,
            remove: remove_thunk::<T>,
            get_json: get_json_unsupported,
            set_json: set_json_unsupported,
            parse_json: parse_json_unsupported,
            get_field: surface.get_field,
            set_field: surface.set_field,
        });
        self.by_id.insert(type_id, lane_id);
        self.by_name.insert(key, lane_id);
        self
    }

    /// Registers a [`RegisterComponent`] under its canonical name.
    ///
    /// Sugar over [`ComponentRegistry::register`] — the name comes from
    /// `T::COMPONENT_NAME` (generated by `#[derive(RegisterComponent)]`),
    /// the field schema from `T::FIELDS`.
    pub fn register_component<T: RegisterComponent>(&mut self) -> &mut Self {
        self.register_with_fields::<T>(T::COMPONENT_NAME, T::FIELDS)
    }

    /// Entry by type.
    pub fn by_id(&self, type_id: TypeId) -> Option<&ComponentMeta> {
        self.by_id
            .get(&type_id)
            .map(|&id| &self.entries[id.as_usize()])
    }

    /// Entry by protocol name.
    pub fn by_name(&self, name: &str) -> Option<&ComponentMeta> {
        self.by_name
            .get(name)
            .map(|&id| &self.entries[id.as_usize()])
    }

    /// Entry by typed protocol name.
    pub fn by_component(&self, name: &ComponentName) -> Option<&ComponentMeta> {
        self.by_name
            .get(name.as_str())
            .map(|&id| &self.entries[id.as_usize()])
    }

    /// Entry by dense lane index.
    pub fn by_lane_id(&self, lane_id: LaneId) -> Option<&ComponentMeta> {
        self.entries.get(lane_id.as_usize())
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
        assert_eq!(pos.lane_id(), LaneId(0));

        let health = registry.by_id(TypeId::of::<Health>()).expect("health");
        assert_eq!(health.name(), "health");
        assert_eq!(health.lane_id(), LaneId(1));
        assert!(registry.by_lane_id(LaneId(1)).is_some());

        // LaneId is a dense projection: by_lane_id and lookup coincide.
        assert!(std::ptr::eq(registry.by_lane_id(LaneId(0)).unwrap(), pos));
        assert!(registry.by_name("ghost").is_none());
        assert!(registry.by_id(TypeId::of::<u8>()).is_none());
        assert!(registry.by_lane_id(LaneId(2)).is_none());

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

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Rigid {
        pose: Position,
        tags: [f32; 3],
    }

    fn registry_with_rigid() -> ComponentRegistry {
        let mut registry = ComponentRegistry::new();
        registry.register::<Rigid>("rigid");
        registry
    }

    #[test]
    fn field_path_parse_accepts_idents_and_indices() {
        for valid in ["x", "velocity.0", "a.b2.c_3", "_p", "translation.10"] {
            let path = FieldPath::parse(valid).expect("valid path");
            assert_eq!(path.as_str(), valid);
        }
        assert_eq!(FieldPath::parse(""), Err(FieldPathError::Empty));
        for invalid in [".x", "x.", "x..y", "0x", "01", "x y", "x-y", "v.é"] {
            let error = FieldPath::parse(invalid).expect_err("invalid path");
            assert!(
                matches!(error, FieldPathError::BadSegment { .. }),
                "path `{invalid}`"
            );
        }
    }

    #[test]
    fn field_path_display_and_serde_roundtrip() {
        let path = FieldPath::from_static("velocity.2");
        assert_eq!(path.to_string(), "velocity.2");
        let wire = serde_json::to_value(&path).unwrap();
        assert_eq!(wire, json!("velocity.2"));
        let back: FieldPath = serde_json::from_value(wire).unwrap();
        assert_eq!(back, path);
        assert!(serde_json::from_value::<FieldPath>(json!("x..y")).is_err());
    }

    #[test]
    fn field_segments_classify_keys_and_indices() {
        let path = FieldPath::from_static("pose.x");
        let segments: Vec<_> = path.segments().collect();
        assert_eq!(
            segments,
            vec![FieldSegment::Field("pose"), FieldSegment::Field("x"),]
        );
        let indexed = FieldPath::from_static("tags.2");
        let segments: Vec<_> = indexed.segments().collect();
        assert_eq!(
            segments,
            vec![FieldSegment::Field("tags"), FieldSegment::Index(2)]
        );
    }

    #[test]
    fn get_set_field_roundtrip_nested_and_indexed() {
        let registry = registry_with_rigid();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("rigid").unwrap();
        meta.set_json(
            &mut store,
            entity,
            &json!({"pose": {"x": 1.0, "y": 2.0}, "tags": [10.0, 20.0, 30.0]}),
        )
        .unwrap();

        assert_eq!(
            meta.get_field(&store, entity, &FieldPath::from_static("pose.x"))
                .unwrap(),
            Some(json!(1.0))
        );
        assert_eq!(
            meta.get_field(&store, entity, &FieldPath::from_static("tags.1"))
                .unwrap(),
            Some(json!(20.0))
        );

        assert!(
            meta.set_field(
                &store,
                entity,
                &FieldPath::from_static("pose.y"),
                &json!(9.0)
            )
            .unwrap()
        );
        assert!(
            meta.set_field(
                &store,
                entity,
                &FieldPath::from_static("tags.0"),
                &json!(-1.0)
            )
            .unwrap()
        );
        // Siblings are untouched by the granular writes.
        assert_eq!(
            meta.get_json(&store, entity).unwrap(),
            Some(json!({"pose": {"x": 1.0, "y": 9.0}, "tags": [-1.0, 20.0, 30.0]}))
        );
    }

    #[test]
    fn set_field_rejects_mistyped_leaf_without_touching_world() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("position").unwrap();
        meta.set_json(&mut store, entity, &json!({"x": 1.0, "y": 2.0}))
            .unwrap();

        let bad = meta.set_field(&store, entity, &FieldPath::from_static("x"), &json!("left"));
        assert!(matches!(bad, Err(RegistryError::Json(_))));
        assert_eq!(
            meta.get_json(&store, entity).unwrap(),
            Some(json!({"x": 1.0, "y": 2.0}))
        );
    }

    #[test]
    fn field_errors_name_component_and_path() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("health").unwrap();
        meta.set_json(&mut store, entity, &json!({"hp": 7}))
            .unwrap();

        let unknown = meta
            .set_field(&store, entity, &FieldPath::from_static("mana"), &json!(1))
            .expect_err("unknown field");
        match unknown {
            RegistryError::UnknownField { component, path } => {
                assert_eq!(component.as_str(), "health");
                assert_eq!(path.as_str(), "mana");
            }
            other => panic!("wrong error: {other:?}"),
        }

        // Out-of-bounds index is an unknown field, not a panic.
        let oob = meta
            .get_field(&store, entity, &FieldPath::from_static("hp.3"))
            .expect_err("index into a scalar");
        assert!(matches!(oob, RegistryError::UnknownField { .. }));

        // Entity without the component.
        let stranger = store.create_entity();
        assert_eq!(
            meta.get_field(&store, stranger, &FieldPath::from_static("hp"))
                .unwrap(),
            None
        );
        let missing = meta
            .set_field(&store, stranger, &FieldPath::from_static("hp"), &json!(1))
            .expect_err("missing component");
        assert!(matches!(missing, RegistryError::MissingComponent { .. }));

        // Lane never created in this world.
        let fresh = SmartStore::new();
        let nowhere = fresh.create_entity();
        let no_lane = meta
            .set_field(&fresh, nowhere, &FieldPath::from_static("hp"), &json!(1))
            .expect_err("missing lane");
        assert!(matches!(no_lane, RegistryError::MissingLane { .. }));
    }

    #[test]
    fn visit_entities_yields_dense_order() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let meta = registry.by_name("health").unwrap();
        let mut expected = Vec::new();
        for i in 0..3u32 {
            let entity = store.create_entity();
            meta.set_json(&mut store, entity, &json!({"hp": i}))
                .unwrap();
            expected.push(entity);
        }
        let mut visited = Vec::new();
        meta.visit_entities(&store, &mut |entity| visited.push(entity));
        assert_eq!(visited, expected);
    }

    #[test]
    fn register_with_fields_stores_schema() {
        use crate::FieldMeta;

        static SCHEMA: &[FieldMeta] = &[FieldMeta {
            name: "hp",
            path: "hp",
            type_name: "u32",
            writable: true,
        }];
        let mut registry = ComponentRegistry::new();
        registry.register_with_fields::<Health>("health", SCHEMA);
        let meta = registry.by_name("health").unwrap();
        assert_eq!(meta.fields(), SCHEMA);
        // Unschemaed registration lists nothing but still navigates.
        let mut plain = ComponentRegistry::new();
        plain.register::<Health>("health");
        assert!(plain.by_name("health").unwrap().fields().is_empty());
    }
}
