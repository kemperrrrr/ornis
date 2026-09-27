//! World mutation bus: the single protocol for changing world content.
//!
//! Editors (built-in or custom) and compute producers (scripting languages,
//! procedural tools, future network replication) all speak one language:
//! [`Mutation`] values applied through [`apply_mutations`]. A scripting
//! language attaches as a [`MutationProducer`] on the shared [`MutationBus`];
//! a custom editor pushes the same [`Mutation`] values. There is no second,
//! scripting-specific write path.
//!
//! Values stay JSON at the boundary because the [`ComponentRegistry`] speaks
//! JSON (`parse_json`/`insert_any`); producers receive `dt`/`tick` as plain
//! numbers, so no per-call codec exists on this seam. Hot ECS loops stay
//! typed; this module is the tooling/editing seam, mirroring
//! `PhysicsEngine` and `RenderBackend` as plugin traits.
//!
//! Timing split (deliberate, one vocabulary): frame producers drain through
//! [`MutationTick`] into the bus and apply between frames, while synchronous
//! producers (the built-in editor) call [`apply_mutations`] directly for an
//! immediate report. Both paths share the type, the applier and the report.

use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};

use crate::{
    ComponentMeta, ComponentName, ComponentRegistry, Engine, Entity, FieldPath, Resources,
    SmartStore, System, SystemAccess, Time,
};

/// Typed mutation failure: one rejection reason per entry.
///
/// Replaces the former `Vec<String>` free-form messages so producers can
/// match on the reason while humans still get the same text via
/// [`std::fmt::Display`] (legacy messages preserved verbatim).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MutationError {
    /// Component name is not registered.
    #[error("unknown component `{0}`")]
    UnknownComponent(ComponentName),
    /// Target entity is not alive (id + generation for diagnostics).
    #[error("entity {id}g{generation} is not alive")]
    EntityNotAlive {
        /// Entity id.
        id: u32,
        /// Entity generation.
        generation: u32,
    },
    /// Registry value failed to parse (`parse_json` message preserved).
    #[error("{0}")]
    Parse(String),
    /// Parsed value failed to insert (internal type mismatch).
    #[error("mutation for `{0}` failed to insert (internal type mismatch)")]
    InsertMismatch(ComponentName),
    /// Field path addresses nothing on the component.
    #[error("unknown field `{path}` on component `{component}`")]
    UnknownField {
        /// Registry name of the component.
        component: ComponentName,
        /// Requested path.
        path: FieldPath,
    },
    /// The entity lacks the component — field access never constructs
    /// the missing remainder (create it with [`Mutation::Set`] first).
    #[error("entity {id}g{generation} has no `{component}` component")]
    MissingComponent {
        /// Registry name of the absent component.
        component: ComponentName,
        /// Entity id.
        id: u32,
        /// Entity generation.
        generation: u32,
    },
}

/// One world-content change, producer-neutral.
///
/// The built-in editor translates `set_component` into [`Mutation::Set`];
/// a scripting language emits the same values from its tick function.
/// Reporting back is [`MutationReport`], shared by every producer.
#[derive(Debug, Clone)]
pub enum Mutation {
    /// Upsert `component` on `entity` from a registry-canonical JSON value
    /// (same encoding the editor protocol and scene snapshots use).
    Set {
        /// Target entity, generation included.
        entity: Entity,
        /// Registry name (see [`ComponentRegistry::by_name`]).
        component: ComponentName,
        /// New component value.
        value: serde_json::Value,
    },
    /// Write one field of `component` on `entity` (granular inspector
    /// edits, sync mappings). The component must already exist — field
    /// access never constructs it.
    SetField {
        /// Target entity, generation included.
        entity: Entity,
        /// Registry name (see [`ComponentRegistry::by_name`]).
        component: ComponentName,
        /// Dotted path inside the canonical JSON form (`position.x`).
        path: FieldPath,
        /// New field value.
        value: serde_json::Value,
    },
}

/// Application report of one [`apply_mutations`] pass, shared by every
/// producer path (frame drain and synchronous editor writes alike).
#[derive(Debug, Clone, Default)]
pub struct MutationReport {
    /// Per-mutation results, in input order.
    pub entries: Vec<AppliedMutation>,
}

impl MutationReport {
    /// Total component writes across all mutations.
    pub fn applied(&self) -> usize {
        self.entries.iter().map(|entry| entry.applied).sum()
    }

    /// Total rejections across all mutations.
    pub fn error_count(&self) -> usize {
        self.entries.iter().map(|entry| entry.errors.len()).sum()
    }
}

/// Per-mutation slice of a [`MutationReport`].
#[derive(Debug, Clone)]
pub struct AppliedMutation {
    /// Index into the input slice passed to [`apply_mutations`].
    pub index: usize,
    /// Component writes performed. Zero with empty `errors` means
    /// "nothing to do"; zero with errors means "rejected".
    pub applied: usize,
    /// One typed rejection per failure.
    pub errors: Vec<MutationError>,
}

impl AppliedMutation {
    /// Legacy human-readable messages (verbatim [`MutationError`] texts).
    pub fn error_messages(&self) -> Vec<String> {
        self.errors.iter().map(|e| e.to_string()).collect()
    }
}

/// Applies `mutations` to the world through `registry`.
///
/// Each mutation validates fully before its write (entity alive with the
/// right generation, known component, value parses), so one bad mutation
/// rejects only its own entry while the rest still apply, in input order.
/// Call between frames with exclusive store access (for example right
/// after [`Engine::run_frame`]); never from inside a system.
pub fn apply_mutations(
    store: &mut SmartStore,
    registry: &ComponentRegistry,
    mutations: &[Mutation],
) -> MutationReport {
    let entries = mutations
        .iter()
        .enumerate()
        .map(|(index, mutation)| apply_one(store, registry, index, mutation))
        .collect();
    MutationReport { entries }
}

/// Validates one mutation, then writes it — or rejects it untouched.
fn apply_one(
    store: &mut SmartStore,
    registry: &ComponentRegistry,
    index: usize,
    mutation: &Mutation,
) -> AppliedMutation {
    match mutation {
        Mutation::Set {
            entity,
            component,
            value,
        } => {
            let meta = match registry.by_component(component) {
                Some(meta) => meta,
                None => {
                    return AppliedMutation {
                        index,
                        applied: 0,
                        errors: vec![MutationError::UnknownComponent(component.clone())],
                    };
                }
            };
            if !store.is_alive(*entity) {
                return AppliedMutation {
                    index,
                    applied: 0,
                    errors: vec![MutationError::EntityNotAlive {
                        id: entity.id(),
                        generation: entity.generation(),
                    }],
                };
            }
            match parse_and_insert(store, meta, *entity, value) {
                Ok(()) => AppliedMutation {
                    index,
                    applied: 1,
                    errors: Vec::new(),
                },
                Err(error) => AppliedMutation {
                    index,
                    applied: 0,
                    errors: vec![error],
                },
            }
        }
        Mutation::SetField {
            entity,
            component,
            path,
            value,
        } => {
            let meta = match registry.by_component(component) {
                Some(meta) => meta,
                None => {
                    return AppliedMutation {
                        index,
                        applied: 0,
                        errors: vec![MutationError::UnknownComponent(component.clone())],
                    };
                }
            };
            if !store.is_alive(*entity) {
                return AppliedMutation {
                    index,
                    applied: 0,
                    errors: vec![MutationError::EntityNotAlive {
                        id: entity.id(),
                        generation: entity.generation(),
                    }],
                };
            }
            match meta.set_field(store, *entity, path, value) {
                Ok(written) => AppliedMutation {
                    index,
                    applied: usize::from(written),
                    errors: Vec::new(),
                },
                Err(error) => AppliedMutation {
                    index,
                    applied: 0,
                    errors: vec![field_error_to_mutation(component, path, *entity, error)],
                },
            }
        }
    }
}

/// Maps a field-surface failure onto the mutation report vocabulary.
/// A guard skip never reaches here (`set_field` reports it as `Ok(false)`).
fn field_error_to_mutation(
    component: &ComponentName,
    path: &FieldPath,
    entity: Entity,
    error: crate::RegistryError,
) -> MutationError {
    match error {
        crate::RegistryError::UnknownField { .. } => MutationError::UnknownField {
            component: component.clone(),
            path: path.clone(),
        },
        crate::RegistryError::MissingComponent { .. }
        | crate::RegistryError::MissingLane { .. } => MutationError::MissingComponent {
            component: component.clone(),
            id: entity.id(),
            generation: entity.generation(),
        },
        crate::RegistryError::UnknownComponent(_) | crate::RegistryError::Json(_) => {
            MutationError::Parse(error.to_string())
        }
    }
}

/// Parses `value` first and inserts only on success, so a schema failure
/// never half-writes the world.
fn parse_and_insert(
    store: &mut SmartStore,
    meta: &ComponentMeta,
    entity: Entity,
    value: &serde_json::Value,
) -> Result<(), MutationError> {
    let boxed = meta
        .parse_json(value)
        .map_err(|error| MutationError::Parse(error.to_string()))?;
    if meta.insert_any(store, entity, boxed) {
        Ok(())
    } else {
        Err(MutationError::InsertMismatch(meta.component_name().clone()))
    }
}

/// A per-frame source of [`Mutation`] values: a scripting language, a
/// procedural tool, or any future producer that computes world edits.
///
/// Producers run through [`MutationTick`] once per frame and never touch
/// the store themselves; the host applies what they emit via
/// [`apply_mutations`]. Synchronous producers (the built-in editor) skip
/// the bus and call [`apply_mutations`] directly for an immediate report.
pub trait MutationProducer: Send + Sync {
    /// Computes this frame's mutations. `dt_seconds` is the variable frame
    /// delta and `tick` the zero-based frame counter of the bus.
    fn produce(&mut self, dt_seconds: f32, tick: u64) -> Vec<Mutation>;
}

/// Shared home for [`MutationProducer`]s inside an [`Engine`].
///
/// The seam needs `&mut` producers, but systems only receive `&Resources` —
/// hence the interior behind a [`Mutex`]. Producers are registered between
/// frames and run in registration order; pending mutations wait for the
/// host drain (see [`MutationTick`]), so the engine lock never meets the
/// store borrow.
pub struct MutationBus {
    inner: Mutex<BusInner>,
    ticks: AtomicU64,
}

#[derive(Default)]
struct BusInner {
    producers: Vec<Box<dyn MutationProducer>>,
    pending: Vec<Mutation>,
}

impl Default for MutationBus {
    fn default() -> Self {
        Self::new()
    }
}

impl MutationBus {
    /// Creates an empty bus: no producers, no pending mutations.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(BusInner::default()),
            ticks: AtomicU64::new(0),
        }
    }

    /// Registers a producer; it runs from the next [`MutationTick`] on,
    /// after previously registered producers.
    pub fn add_producer(&self, producer: Box<dyn MutationProducer>) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.producers.push(producer);
        }
    }

    /// Pushes one mutation from a synchronous producer (a custom editor
    /// driving the same bus the frame tick drains).
    pub fn push(&self, mutation: Mutation) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.pending.push(mutation);
        }
    }

    /// Number of registered producers.
    pub fn producer_count(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.producers.len())
            .unwrap_or(0)
    }

    /// Mutations waiting for the host drain.
    pub fn pending_count(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.pending.len())
            .unwrap_or(0)
    }

    /// Frames produced so far.
    pub fn ticks(&self) -> u64 {
        self.ticks.load(Ordering::SeqCst)
    }

    /// Runs every producer with the frame delta and queues what they emit.
    /// A failing producer contributes nothing and never blocks the rest
    /// (producers return values, not results, by contract).
    fn produce_all(&self, dt_seconds: f32) {
        let tick = self.ticks.fetch_add(1, Ordering::SeqCst);
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        let mut fresh = Vec::new();
        for producer in inner.producers.iter_mut() {
            fresh.extend(producer.produce(dt_seconds, tick));
        }
        inner.pending.extend(fresh);
    }

    /// Takes all queued mutations for the host apply step; the queue is
    /// empty afterwards, so every mutation applies at most once.
    pub fn drain(&self) -> Vec<Mutation> {
        self.inner
            .lock()
            .map(|mut inner| std::mem::take(&mut inner.pending))
            .unwrap_or_default()
    }
}

/// Once-per-frame driver for [`MutationBus`] producers.
///
/// Reads [`Time`] for the frame delta and declares no lanes. Producers run
/// once per frame (never per fixed substep — their output would multiply
/// with the substep count), so this system belongs in the variable
/// schedule ([`Engine::schedule_mut`], the [`crate::Stage::PostFrame`]
/// storage); it also reads the variable [`Time`] delta, not the fixed step.
pub struct MutationTick;

impl System for MutationTick {
    fn name(&self) -> &'static str {
        "mutation_tick"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new().reads::<Time>().reads::<MutationBus>()
    }

    fn run(&self, resources: &Resources) {
        let dt = resources
            .get::<Time>()
            .map(|t| t.delta_seconds())
            .unwrap_or(0.0);
        if let Some(bus) = resources.get::<MutationBus>() {
            bus.produce_all(dt);
        }
    }
}

/// Installs mutation ticking into an [`Engine`], mirroring `GameplayPlugin`.
///
/// ```
/// # use ornis_core::mutation::{Mutation, MutationBus, MutationPlugin, MutationProducer};
/// # use ornis_core::Engine;
/// struct PushOne(Option<Mutation>);
/// impl MutationProducer for PushOne {
///     fn produce(&mut self, _dt: f32, _tick: u64) -> Vec<Mutation> {
///         self.0.take().into_iter().collect()
///     }
/// }
/// let mut engine = Engine::new();
/// MutationPlugin::new().install(&mut engine);
/// engine.world().resources().get::<MutationBus>().expect("bus installed")
///     .add_producer(Box::new(PushOne(None)));
/// engine.run_frame(1.0 / 60.0);
/// ```
pub struct MutationPlugin {
    bus: MutationBus,
}

impl MutationPlugin {
    /// Starts a plugin with an empty bus.
    pub fn new() -> Self {
        Self {
            bus: MutationBus::new(),
        }
    }

    /// Inserts the bus resource and the `mutation_tick` system into the
    /// variable schedule (the [`crate::Stage::PostFrame`] storage: once
    /// per frame, after the fixed loop — see [`MutationTick`]).
    pub fn install(self, engine: &mut Engine) {
        engine.world_mut().insert(self.bus);
        engine.schedule_mut().add_system(MutationTick);
    }
}

impl Default for MutationPlugin {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    struct Mana {
        mana: u32,
    }

    struct EchoProducer {
        calls: Vec<(f32, u64)>,
        scripted: Vec<Vec<Mutation>>,
    }

    impl EchoProducer {
        fn new(scripted: Vec<Vec<Mutation>>) -> Self {
            Self {
                calls: Vec::new(),
                scripted,
            }
        }
    }

    impl MutationProducer for EchoProducer {
        fn produce(&mut self, dt_seconds: f32, tick: u64) -> Vec<Mutation> {
            self.calls.push((dt_seconds, tick));
            if self.scripted.is_empty() {
                Vec::new()
            } else {
                self.scripted.remove(0)
            }
        }
    }

    fn test_registry() -> ComponentRegistry {
        let mut registry = ComponentRegistry::new();
        registry.register::<Mana>("mana");
        registry
    }

    #[test]
    fn tick_produces_once_per_frame_not_per_fixed_substep() {
        // Placement pin: PostFrame storage only, so producers run once per
        // frame instead of multiplying with the fixed step count.
        let mut engine = crate::Engine::new();
        MutationPlugin::new().install(&mut engine);
        let bus = engine
            .world()
            .resources()
            .get::<MutationBus>()
            .expect("bus installed");
        assert_eq!(engine.schedule().len(), 1);
        assert!(engine.fixed_schedule().is_empty());
        assert!(engine.stage_schedule(crate::Stage::PreUpdate).is_empty());
        assert!(engine.stage_schedule(crate::Stage::Input).is_empty());
        bus.add_producer(Box::new(EchoProducer::new(vec![Vec::new()])));

        // Catch-up frame: three fixed steps, exactly one produce call
        // carrying the variable frame delta.
        let delta = crate::FixedTime::default().delta_seconds();
        engine.run_frame(delta * 3.0);

        let fixed = *engine
            .world()
            .resources()
            .get::<crate::FixedTime>()
            .expect("fixed clock");
        assert_eq!(fixed.steps_this_frame(), 3);
        assert_eq!(fixed.tick(), 3);
        let bus = engine
            .world()
            .resources()
            .get::<MutationBus>()
            .expect("bus installed");
        assert_eq!(bus.ticks(), 1);
        assert_eq!(bus.pending_count(), 0);
        assert_eq!(bus.producer_count(), 1);
    }

    #[test]
    fn produce_calls_carry_variable_dt_and_tick_index() {
        let mut engine = crate::Engine::new();
        MutationPlugin::new().install(&mut engine);
        // Keep the producer handle outside: calls are recorded through it.
        let probe = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        struct Probe {
            log: std::sync::Arc<std::sync::Mutex<Vec<(f32, u64)>>>,
        }
        impl MutationProducer for Probe {
            fn produce(&mut self, dt_seconds: f32, tick: u64) -> Vec<Mutation> {
                self.log.lock().expect("log").push((dt_seconds, tick));
                Vec::new()
            }
        }
        engine
            .world()
            .resources()
            .get::<MutationBus>()
            .expect("bus")
            .add_producer(Box::new(Probe { log: probe.clone() }));
        engine.run_frame(0.25);
        engine.run_frame(0.5);
        let log = probe.lock().expect("log");
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].0, 0.25);
        assert_eq!(log[0].1, 0);
        assert_eq!(log[1].0, 0.5);
        assert_eq!(log[1].1, 1);
    }

    #[test]
    fn producers_run_in_registration_order() {
        let mut engine = crate::Engine::new();
        MutationPlugin::new().install(&mut engine);
        let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        struct Tag {
            id: u32,
            order: std::sync::Arc<std::sync::Mutex<Vec<u32>>>,
        }
        impl MutationProducer for Tag {
            fn produce(&mut self, _dt: f32, _tick: u64) -> Vec<Mutation> {
                self.order.lock().expect("order").push(self.id);
                Vec::new()
            }
        }
        let bus = engine
            .world()
            .resources()
            .get::<MutationBus>()
            .expect("bus");
        for id in [1, 2, 3] {
            bus.add_producer(Box::new(Tag {
                id,
                order: order.clone(),
            }));
        }
        engine.run_frame(1.0 / 60.0);
        assert_eq!(*order.lock().expect("order"), vec![1, 2, 3]);
    }

    #[test]
    fn tick_without_bus_is_noop() {
        let resources = Resources::new();
        MutationTick.run(&resources);
    }

    fn apply_setup() -> (SmartStore, ComponentRegistry) {
        let mut store = SmartStore::new();
        store.register::<Mana>();
        (store, test_registry())
    }

    fn mana_of(store: &SmartStore, entity: Entity) -> Option<u32> {
        store
            .read_lane::<Mana>()
            .expect("lane")
            .get(entity)
            .map(|mana| mana.mana)
    }

    #[test]
    fn apply_set_writes_through_registry() {
        let (mut store, registry) = apply_setup();
        let entity = store.create_entity();
        let report = apply_mutations(
            &mut store,
            &registry,
            &[Mutation::Set {
                entity,
                component: "mana".into(),
                value: serde_json::json!({"mana": 7}),
            }],
        );
        assert_eq!(report.applied(), 1);
        assert_eq!(report.error_count(), 0);
        assert_eq!(mana_of(&store, entity), Some(7));
    }

    #[test]
    fn apply_rejects_entry_atomically_and_keeps_world_untouched() {
        let (mut store, registry) = apply_setup();
        let entity = store.create_entity();
        let dead = Entity::new_with_gen(999, 0);
        let report = apply_mutations(
            &mut store,
            &registry,
            &[
                Mutation::Set {
                    entity,
                    component: "mana".into(),
                    value: serde_json::json!({"mana": 7}),
                },
                Mutation::Set {
                    entity: dead,
                    component: "mana".into(),
                    value: serde_json::json!({"mana": 9}),
                },
                Mutation::Set {
                    entity,
                    component: "nope".into(),
                    value: serde_json::json!(null),
                },
            ],
        );
        assert_eq!(report.applied(), 1);
        assert_eq!(report.error_count(), 2);
        assert_eq!(report.entries[0].index, 0);
        assert!(report.entries[0].errors.is_empty());
        assert_eq!(report.entries[1].index, 1);
        assert_eq!(report.entries[2].index, 2);
        // The good entry still wrote; the bad ones wrote nothing.
        assert_eq!(mana_of(&store, entity), Some(7));
    }

    #[test]
    fn apply_rejects_bad_value_without_writing() {
        let (mut store, registry) = apply_setup();
        let entity = store.create_entity();
        store.insert(entity, Mana { mana: 7 });
        let report = apply_mutations(
            &mut store,
            &registry,
            &[Mutation::Set {
                entity,
                component: "mana".into(),
                value: serde_json::json!({"mana": "lots"}),
            }],
        );
        assert_eq!(report.applied(), 0);
        assert_eq!(report.error_count(), 1);
        assert_eq!(mana_of(&store, entity), Some(7));
    }

    fn set_field(entity: Entity, value: serde_json::Value) -> Mutation {
        Mutation::SetField {
            entity,
            component: "mana".into(),
            path: crate::FieldPath::from_static("mana"),
            value,
        }
    }

    #[test]
    fn apply_set_field_writes_granularly() {
        let (mut store, registry) = apply_setup();
        let entity = store.create_entity();
        store.insert(entity, Mana { mana: 7 });
        let report = apply_mutations(
            &mut store,
            &registry,
            &[set_field(entity, serde_json::json!(42))],
        );
        assert_eq!(report.applied(), 1);
        assert_eq!(report.error_count(), 0);
        assert_eq!(mana_of(&store, entity), Some(42));
    }

    #[test]
    fn apply_set_field_rejects_with_typed_errors() {
        let (mut store, registry) = apply_setup();
        let entity = store.create_entity();
        store.insert(entity, Mana { mana: 7 });
        let dead = Entity::new_with_gen(999, 0);
        let report = apply_mutations(
            &mut store,
            &registry,
            &[
                set_field(entity, serde_json::json!("lots")),
                Mutation::SetField {
                    entity,
                    component: "mana".into(),
                    path: crate::FieldPath::from_static("health"),
                    value: serde_json::json!(1),
                },
                set_field(dead, serde_json::json!(1)),
            ],
        );
        assert_eq!(report.applied(), 0);
        assert_eq!(report.error_count(), 3);
        assert!(matches!(
            report.entries[0].errors[0],
            MutationError::Parse(_)
        ));
        assert!(matches!(
            report.entries[1].errors[0],
            MutationError::UnknownField { .. }
        ));
        assert!(matches!(
            report.entries[2].errors[0],
            MutationError::EntityNotAlive { .. }
        ));
        // Nothing wrote.
        assert_eq!(mana_of(&store, entity), Some(7));
        // Human-readable texts still render.
        assert!(!report.entries[1].error_messages().is_empty());
    }

    #[test]
    fn apply_set_field_needs_existing_component() {
        let (mut store, registry) = apply_setup();
        let entity = store.create_entity();
        let report = apply_mutations(
            &mut store,
            &registry,
            &[set_field(entity, serde_json::json!(1))],
        );
        assert_eq!(report.applied(), 0);
        assert_eq!(report.error_count(), 1);
        assert!(matches!(
            report.entries[0].errors[0],
            MutationError::MissingComponent { .. }
        ));
    }
}
