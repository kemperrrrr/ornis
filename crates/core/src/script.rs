//! Scripting plugin seam for Ornis (Phase 6, plan D1).
//!
//! The engine core knows only the [`ScriptEngine`] trait — concrete languages
//! (Rhai, Rune, Python, WASM components) are adapters that implement it.
//! Hot-path ECS loops stay typed; this trait is for tooling, FFI and
//! editor integration, mirroring `PhysicsEngine` and `RenderBackend`.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};

use crate::{
    ComponentMeta, ComponentRegistry, Engine, Entity, Resources, SmartStore, System, SystemAccess,
    Time,
};

/// Opaque handle to a script instance (one script file / module).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScriptHandle(pub u64);

/// Opaque handle to a component/entity batch entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchHandle(pub u64);

/// Error returned by script operations.
#[derive(Debug, Clone)]
pub struct ScriptError(pub String);

impl std::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ScriptError {}

/// Result of a batch operation.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Per-handle outcome: `Ok(())` or `Err` string.
    pub outcomes: HashMap<BatchHandle, Result<(), String>>,
}

/// Plugin trait for scripting backends.
///
/// The core engine depends only on this trait; language specifics live
/// in adapter crates (e.g. `ornis-rhai`). This keeps the hot ECS
/// path typed while allowing editor/FFI to drive scripts through
/// type-erased handles.
///
/// # Errors
///
/// All methods return [`ScriptError`] on load/compile or runtime failure.
pub trait ScriptEngine: Send + Sync {
    /// Load (or reload) a script module from source text.
    ///
    /// Returns a handle that can be used with [`ScriptEngine::call`].
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError`] if the source fails to parse/compile.
    fn load(&mut self, name: &str, source: &str) -> Result<ScriptHandle, ScriptError>;

    /// Call an exported function on a loaded script.
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError`] if the handle is unknown or the call traps.
    fn call(
        &mut self,
        handle: &ScriptHandle,
        func: &str,
        args: &[u8],
    ) -> Result<Vec<u8>, ScriptError>;

    /// Batch variant: one call instead of `N` individual calls.
    ///
    /// Handles are preferred over raw pointers into `SparseSet` cells —
    /// pointers are only valid in-process and break the WASM sandbox.
    fn batch_call(&mut self, calls: &[(ScriptHandle, String, Vec<u8>)]) -> BatchResult;

    /// Hot-reload a previously loaded module in place.
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError`] if the new source fails to compile; the
    /// previous version must remain usable.
    fn hot_reload(&mut self, handle: &ScriptHandle, new_source: &str) -> Result<(), ScriptError>;

    /// Unload a script module.
    fn unload(&mut self, handle: &ScriptHandle);
}

/// No-op engine useful for tests and as a baseline adapter.
#[derive(Debug, Default)]
pub struct NoopScriptEngine {
    next_id: u64,
    modules: HashMap<u64, String>,
}

impl ScriptEngine for NoopScriptEngine {
    fn load(&mut self, _name: &str, source: &str) -> Result<ScriptHandle, ScriptError> {
        let id = self.next_id;
        self.next_id += 1;
        self.modules.insert(id, source.to_owned());
        Ok(ScriptHandle(id))
    }

    fn call(
        &mut self,
        handle: &ScriptHandle,
        _func: &str,
        _args: &[u8],
    ) -> Result<Vec<u8>, ScriptError> {
        if self.modules.contains_key(&handle.0) {
            Ok(Vec::new())
        } else {
            Err(ScriptError(format!("unknown handle {}", handle.0)))
        }
    }

    fn batch_call(&mut self, calls: &[(ScriptHandle, String, Vec<u8>)]) -> BatchResult {
        let outcomes = calls
            .iter()
            .enumerate()
            .map(|(i, (h, _, _))| {
                let r = if self.modules.contains_key(&h.0) {
                    Ok(())
                } else {
                    Err(format!("unknown handle {}", h.0))
                };
                (BatchHandle(i as u64), r)
            })
            .collect();
        BatchResult { outcomes }
    }

    fn hot_reload(&mut self, handle: &ScriptHandle, new_source: &str) -> Result<(), ScriptError> {
        if let Some(entry) = self.modules.get_mut(&handle.0) {
            *entry = new_source.to_owned();
            Ok(())
        } else {
            Err(ScriptError(format!("unknown handle {}", handle.0)))
        }
    }

    fn unload(&mut self, handle: &ScriptHandle) {
        self.modules.remove(&handle.0);
    }
}

/// One per-frame script entry: a loaded module plus the exported
/// function the tick system calls every frame.
#[derive(Debug, Clone)]
pub struct ScriptTickEntry {
    /// Module to call into.
    pub handle: ScriptHandle,
    /// Exported function called with the tick payload.
    pub func: String,
}

/// Runtime home for a [`ScriptEngine`] inside an [`Engine`].
///
/// The seam needs `&mut self`, but systems only receive `&Resources` —
/// hence the engine lives behind a [`Mutex`]. Tick entries are
/// registered between frames and never change while systems run, so
/// they need no lock; per-entry outcomes are rewritten every frame,
/// hence theirs.
pub struct ScriptHost {
    engine: Mutex<Box<dyn ScriptEngine>>,
    entries: Vec<ScriptTickEntry>,
    outcomes: Mutex<HashMap<usize, Result<Vec<u8>, String>>>,
    ticks: AtomicU64,
}

impl ScriptHost {
    /// Wraps an engine with no tick entries yet.
    pub fn new(engine: Box<dyn ScriptEngine>) -> Self {
        Self {
            engine: Mutex::new(engine),
            entries: Vec::new(),
            outcomes: Mutex::new(HashMap::new()),
            ticks: AtomicU64::new(0),
        }
    }

    /// Registers an already-loaded module for per-frame calls.
    ///
    /// Returns the entry index used by [`ScriptHost::outcome`].
    pub fn add_tick(&mut self, handle: ScriptHandle, func: impl Into<String>) -> usize {
        let index = self.entries.len();
        self.entries.push(ScriptTickEntry {
            handle,
            func: func.into(),
        });
        index
    }

    /// Loads a module and registers it for per-frame calls.
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError`] if the source fails to parse/compile;
    /// nothing is registered in that case.
    pub fn load_and_tick(
        &mut self,
        name: &str,
        source: &str,
        func: &str,
    ) -> Result<usize, ScriptError> {
        let handle = self
            .engine
            .lock()
            .map_err(|_| ScriptError("script host lock poisoned".into()))?
            .load(name, source)?;
        Ok(self.add_tick(handle, func))
    }

    /// Tick entries in registration order.
    pub fn entries(&self) -> &[ScriptTickEntry] {
        &self.entries
    }

    /// Last outcome of entry `index`, once it has ticked at least once.
    pub fn outcome(&self, index: usize) -> Option<Result<Vec<u8>, String>> {
        self.outcomes.lock().ok()?.get(&index).cloned()
    }

    /// Frames ticked so far.
    pub fn ticks(&self) -> u64 {
        self.ticks.load(Ordering::SeqCst)
    }

    /// Calls every entry with `{"dt": dt, "tick": n}` and stores outcomes.
    ///
    /// A failing entry records its error and never blocks the rest.
    fn tick_all(&self, dt: f32) {
        let n = self.ticks.fetch_add(1, Ordering::SeqCst);
        let args = serde_json::json!({"dt": dt, "tick": n})
            .to_string()
            .into_bytes();
        let Ok(mut engine) = self.engine.lock() else {
            self.record_all(Err("script host lock poisoned".to_owned()));
            return;
        };
        for (index, entry) in self.entries.iter().enumerate() {
            let out = engine
                .call(&entry.handle, &entry.func, &args)
                .map_err(|e| e.0);
            self.record(index, out);
        }
    }

    /// Stores one entry's outcome.
    fn record(&self, index: usize, outcome: Result<Vec<u8>, String>) {
        if let Ok(mut outcomes) = self.outcomes.lock() {
            outcomes.insert(index, outcome);
        }
    }

    /// Stores the same outcome for every entry (host-level failure).
    fn record_all(&self, outcome: Result<Vec<u8>, String>) {
        if let Ok(mut outcomes) = self.outcomes.lock() {
            for index in 0..self.entries.len() {
                outcomes.insert(index, outcome.clone());
            }
        }
    }
}

/// Once-per-frame driver for [`ScriptHost`] tick entries.
///
/// Reads [`Time`] for the frame delta and declares no lanes. Every call
/// goes through the JSON codec, so this system belongs in the variable
/// schedule ([`Engine::schedule_mut`]) — never the bounded fixed one.
pub struct ScriptTickSystem;

impl System for ScriptTickSystem {
    fn name(&self) -> &'static str {
        "script_tick"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new().reads::<Time>().reads::<ScriptHost>()
    }

    fn run(&self, resources: &Resources) {
        let dt = resources
            .get::<Time>()
            .map(|t| t.delta_seconds())
            .unwrap_or(0.0);
        if let Some(host) = resources.get::<ScriptHost>() {
            host.tick_all(dt);
        }
    }
}

/// Installs script ticking into an [`Engine`], mirroring `GameplayPlugin`.
///
/// ```no_run
/// # use ornis_core::script::{NoopScriptEngine, ScriptPlugin};
/// # use ornis_core::Engine;
/// let mut engine = Engine::new();
/// ScriptPlugin::new(Box::new(NoopScriptEngine::default()))
///     .with_tick("idle", "fn tick() {}", "tick")
///     .expect("script compiles")
///     .install(&mut engine);
/// engine.run_frame(1.0 / 60.0);
/// ```
pub struct ScriptPlugin {
    host: ScriptHost,
}

impl ScriptPlugin {
    /// Starts a plugin around `engine` with no entries yet.
    pub fn new(engine: Box<dyn ScriptEngine>) -> Self {
        Self {
            host: ScriptHost::new(engine),
        }
    }

    /// Loads `source` and ticks `func` every frame.
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError`] if the source fails to parse/compile;
    /// nothing is registered in that case.
    pub fn with_tick(mut self, name: &str, source: &str, func: &str) -> Result<Self, ScriptError> {
        self.host.load_and_tick(name, source, func)?;
        Ok(self)
    }

    /// Inserts the host resource and the `script_tick` system into the
    /// variable schedule.
    pub fn install(self, engine: &mut Engine) {
        engine.world_mut().insert(self.host);
        engine.schedule_mut().add_system(ScriptTickSystem);
    }
}

/// Application report of one [`ScriptHost::apply_outcomes`] pass.
#[derive(Debug, Clone, Default)]
pub struct ApplyReport {
    /// Per-entry results for entries that had a stored outcome, in
    /// entry-index order. Entries that did not tick since the previous
    /// apply are absent.
    pub entries: Vec<AppliedEntry>,
}

impl ApplyReport {
    /// Total patches written across all entries.
    pub fn applied(&self) -> usize {
        self.entries.iter().map(|entry| entry.applied).sum()
    }

    /// Total patch errors across all entries.
    pub fn error_count(&self) -> usize {
        self.entries.iter().map(|entry| entry.errors.len()).sum()
    }
}

/// Per-entry slice of an [`ApplyReport`].
#[derive(Debug, Clone)]
pub struct AppliedEntry {
    /// Tick entry index (see [`ScriptHost::add_tick`]).
    pub index: usize,
    /// Patches written. Zero with empty `errors` means "nothing to do"
    /// (no `"set"` key); zero with errors means "rejected".
    pub applied: usize,
    /// One message per rejected patch (or a tick/outcome failure).
    pub errors: Vec<String>,
}

impl ScriptHost {
    /// Applies drained tick outcomes to the world through `registry`.
    ///
    /// Each outcome must be a JSON object; its optional `"set"` array
    /// holds patches of shape `{"entity": {"id": N, "generation": M},
    /// "component": "<name>", "value": {...}}`. Outcomes without a
    /// `"set"` key are skipped — not every tick function writes the
    /// world. Validation is two-phase per entry: every patch parses and
    /// every entity proves alive before the first write, so one bad
    /// patch rejects its whole entry while other entries still apply.
    /// Outcomes apply at most once: they are drained, and
    /// [`ScriptHost::outcome`] no longer reports them afterwards.
    ///
    /// Call between frames with exclusive store access (for example
    /// right after [`Engine::run_frame`]); never from inside a system.
    pub fn apply_outcomes(
        &self,
        store: &mut SmartStore,
        registry: &ComponentRegistry,
    ) -> ApplyReport {
        let drained = self
            .outcomes
            .lock()
            .map(|mut outcomes| std::mem::take(&mut *outcomes));
        let Ok(drained) = drained else {
            return ApplyReport::default();
        };
        let mut entries: Vec<_> = drained
            .into_iter()
            .map(|(index, outcome)| apply_one(store, registry, index, outcome))
            .collect();
        entries.sort_by_key(|entry| entry.index);
        ApplyReport { entries }
    }
}

/// Applies one entry's outcome: parses the `"set"` array, validates
/// every patch, then writes all patches or nothing.
fn apply_one(
    store: &mut SmartStore,
    registry: &ComponentRegistry,
    index: usize,
    outcome: Result<Vec<u8>, String>,
) -> AppliedEntry {
    let bytes = match outcome {
        Ok(bytes) => bytes,
        Err(message) => {
            return AppliedEntry {
                index,
                applied: 0,
                errors: vec![format!("tick failed: {message}")],
            };
        }
    };
    let value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            return AppliedEntry {
                index,
                applied: 0,
                errors: vec![format!("outcome is not JSON: {error}")],
            };
        }
    };
    let patches = match value.get("set") {
        None => {
            return AppliedEntry {
                index,
                applied: 0,
                errors: Vec::new(),
            };
        }
        Some(serde_json::Value::Array(items)) => items,
        Some(_) => {
            return AppliedEntry {
                index,
                applied: 0,
                errors: vec!["`set` must be an array".to_owned()],
            };
        }
    };
    // Phase 1: validate everything, writing nothing.
    let mut errors = Vec::new();
    let mut pending = Vec::new();
    for (n, patch) in patches.iter().enumerate() {
        match parse_patch(store, registry, patch) {
            Ok(valid) => pending.push(valid),
            Err(message) => errors.push(format!("patch {n}: {message}")),
        }
    }
    if !errors.is_empty() {
        return AppliedEntry {
            index,
            applied: 0,
            errors,
        };
    }
    // Phase 2: all valid — write.
    let mut applied = 0;
    for (entity, meta, boxed) in pending {
        if meta.insert_any(store, entity, boxed) {
            applied += 1;
        } else {
            errors.push(format!(
                "patch for `{}` failed to insert (internal type mismatch)",
                meta.name()
            ));
        }
    }
    AppliedEntry {
        index,
        applied,
        errors,
    }
}

/// Validates one patch object without touching the world: entity shape,
/// known component, liveness, and JSON schema (parsed into an untyped
/// box ready for [`ComponentMeta::insert_any`]).
fn parse_patch<'meta>(
    store: &SmartStore,
    registry: &'meta ComponentRegistry,
    patch: &serde_json::Value,
) -> Result<(Entity, &'meta ComponentMeta, Box<dyn Any>), String> {
    let id = patch["entity"]["id"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or("bad `entity.id` (need a u32)")?;
    let generation = patch["entity"]["generation"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or("bad `entity.generation` (need a u32)")?;
    let entity = Entity::new_with_gen(id, generation);
    let component = patch["component"]
        .as_str()
        .ok_or("bad `component` (need a name)")?;
    let meta = registry
        .by_name(component)
        .ok_or_else(|| format!("unknown component `{component}`"))?;
    if !store.is_alive(entity) {
        return Err(format!("entity {id}g{generation} is not alive"));
    }
    let value = patch.get("value").ok_or("missing `value`")?;
    let boxed = meta.parse_json(value).map_err(|error| error.to_string())?;
    Ok((entity, meta, boxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_load_call_reload() {
        let mut eng = NoopScriptEngine::default();
        let h = eng.load("test", "fn foo() {}").expect("load");
        eng.call(&h, "foo", &[]).expect("call");
        eng.hot_reload(&h, "fn foo() { 42 }").expect("reload");
        let r = eng.batch_call(&[(h.clone(), "foo".into(), vec![])]);
        assert_eq!(r.outcomes.len(), 1);
        eng.unload(&h);
        assert!(eng.call(&h, "foo", &[]).is_err());
    }

    /// Test double that logs every call and answers with its own call
    /// count as JSON; the function named `"boom"` always fails, so one
    /// test can cover both the happy path and entry isolation.
    type CallLog = std::sync::Arc<Mutex<Vec<(String, Vec<u8>)>>>;

    struct EchoEngine {
        calls: CallLog,
        next_id: Mutex<u64>,
        known: Mutex<HashMap<u64, String>>,
    }

    impl EchoEngine {
        fn new() -> (Self, CallLog) {
            let calls = std::sync::Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    calls: std::sync::Arc::clone(&calls),
                    next_id: Mutex::new(0),
                    known: Mutex::new(HashMap::new()),
                },
                calls,
            )
        }
    }

    impl ScriptEngine for EchoEngine {
        fn load(&mut self, name: &str, _source: &str) -> Result<ScriptHandle, ScriptError> {
            let mut next = self.next_id.lock().expect("test lock");
            let id = *next;
            *next += 1;
            self.known
                .lock()
                .expect("test lock")
                .insert(id, name.to_owned());
            Ok(ScriptHandle(id))
        }

        fn call(
            &mut self,
            handle: &ScriptHandle,
            func: &str,
            args: &[u8],
        ) -> Result<Vec<u8>, ScriptError> {
            if !self
                .known
                .lock()
                .expect("test lock")
                .contains_key(&handle.0)
            {
                return Err(ScriptError(format!("unknown handle {}", handle.0)));
            }
            if func == "boom" {
                return Err(ScriptError("boom failed".into()));
            }
            let mut calls = self.calls.lock().expect("test lock");
            calls.push((func.to_owned(), args.to_owned()));
            Ok(format!(r#"{{"call":{}}}"#, calls.len()).into_bytes())
        }

        fn batch_call(&mut self, calls: &[(ScriptHandle, String, Vec<u8>)]) -> BatchResult {
            let outcomes = calls
                .iter()
                .enumerate()
                .map(|(i, (h, func, args))| {
                    let r = self.call(h, func, args).map(|_| ()).map_err(|e| e.0);
                    (BatchHandle(i as u64), r)
                })
                .collect();
            BatchResult { outcomes }
        }

        fn hot_reload(
            &mut self,
            handle: &ScriptHandle,
            _new_source: &str,
        ) -> Result<(), ScriptError> {
            if self
                .known
                .lock()
                .expect("test lock")
                .contains_key(&handle.0)
            {
                Ok(())
            } else {
                Err(ScriptError(format!("unknown handle {}", handle.0)))
            }
        }

        fn unload(&mut self, handle: &ScriptHandle) {
            self.known.lock().expect("test lock").remove(&handle.0);
        }
    }

    fn tick_args(payload: &[u8]) -> serde_json::Value {
        serde_json::from_slice(payload).expect("tick args are JSON")
    }

    #[test]
    fn script_tick_drives_entries_with_dt_and_tick_index() {
        let (echo, log) = EchoEngine::new();
        let mut engine = crate::Engine::new();
        ScriptPlugin::new(Box::new(echo))
            .with_tick("a", "src", "tick_a")
            .expect("load a")
            .with_tick("b", "src", "tick_b")
            .expect("load b")
            .install(&mut engine);
        engine.run_frame(0.016);
        engine.run_frame(0.016);

        let host = engine
            .world()
            .resources()
            .get::<ScriptHost>()
            .expect("host installed");
        assert_eq!(host.ticks(), 2);
        assert_eq!(host.entries().len(), 2);

        // Two frames × two entries, in registration order per frame.
        let calls = log.lock().expect("test lock");
        assert_eq!(calls.len(), 4);
        for (i, (func, args)) in calls.iter().enumerate() {
            let frame = i / 2;
            assert_eq!(*func, if i % 2 == 0 { "tick_a" } else { "tick_b" });
            let v = tick_args(args);
            assert!((v["dt"].as_f64().expect("dt") - 0.016).abs() < 1e-6);
            assert_eq!(v["tick"].as_u64().expect("tick"), frame as u64);
        }

        for index in 0..2 {
            let out = host.outcome(index).expect("outcome stored");
            let v: serde_json::Value =
                serde_json::from_slice(&out.expect("call ok")).expect("outcome JSON");
            assert!(v["call"].as_u64().is_some());
        }
    }

    #[test]
    fn script_tick_isolates_failing_entries() {
        let (echo, log) = EchoEngine::new();
        let mut engine = crate::Engine::new();
        ScriptPlugin::new(Box::new(echo))
            .with_tick("ok-src", "src", "tick_ok")
            .expect("load ok")
            .with_tick("bad-src", "src", "boom")
            .expect("load bad")
            .install(&mut engine);
        engine.run_frame(0.016);

        let host = engine
            .world()
            .resources()
            .get::<ScriptHost>()
            .expect("host installed");
        assert!(host.outcome(0).expect("outcome 0").is_ok());
        assert!(host.outcome(1).expect("outcome 1").is_err());
        // The failing entry was still attempted exactly once.
        assert_eq!(log.lock().expect("test lock").len(), 1);
    }

    #[test]
    fn script_tick_without_host_is_noop() {
        let resources = Resources::new();
        ScriptTickSystem.run(&resources);
    }

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    struct Mana {
        mana: u32,
    }

    fn apply_setup() -> (SmartStore, ComponentRegistry, ScriptHost, Entity) {
        let mut store = SmartStore::new();
        store.register::<Mana>();
        let mut registry = ComponentRegistry::new();
        registry.register::<Mana>("mana");
        let entity = store.create_entity();
        let host = ScriptHost::new(Box::new(NoopScriptEngine::default()));
        (store, registry, host, entity)
    }

    fn store_outcome(host: &ScriptHost, index: usize, payload: Vec<u8>) {
        host.outcomes
            .lock()
            .expect("test lock")
            .insert(index, Ok(payload));
    }

    fn set_patch(entity: Entity, component: &str, value: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"set": [{
            "entity": {"id": entity.id(), "generation": entity.generation()},
            "component": component,
            "value": value,
        }]})
    }

    fn mana_of(store: &SmartStore, registry: &ComponentRegistry, entity: Entity) -> Option<u32> {
        let meta = registry.by_name("mana").expect("mana registered");
        let value = meta
            .get_json(store, entity)
            .expect("get_json")
            .expect("present");
        value["mana"].as_u64().and_then(|n| u32::try_from(n).ok())
    }

    #[test]
    fn apply_outcomes_sets_components_and_drains() {
        let (mut store, registry, host, entity) = apply_setup();
        store_outcome(
            &host,
            0,
            set_patch(entity, "mana", serde_json::json!({"mana": 7}))
                .to_string()
                .into_bytes(),
        );
        let report = host.apply_outcomes(&mut store, &registry);
        assert_eq!(report.applied(), 1);
        assert_eq!(report.error_count(), 0);
        assert_eq!(mana_of(&store, &registry, entity), Some(7));
        // Second apply finds nothing: outcomes drain on use.
        let again = host.apply_outcomes(&mut store, &registry);
        assert_eq!(again.applied(), 0);
        assert!(again.entries.is_empty());
        assert!(host.outcome(0).is_none());
    }

    #[test]
    fn apply_outcomes_rejects_entry_atomically() {
        let (mut store, registry, host, entity) = apply_setup();
        let payload = serde_json::json!({"set": [
            set_patch(entity, "mana", serde_json::json!({"mana": 7}))["set"][0],
            {"entity": {"id": entity.id(), "generation": entity.generation()},
             "component": "nope", "value": {}},
        ]});
        store_outcome(&host, 0, payload.to_string().into_bytes());
        let report = host.apply_outcomes(&mut store, &registry);
        assert_eq!(report.applied(), 0);
        assert_eq!(report.error_count(), 1);
        // The valid patch was not written either.
        let meta = registry.by_name("mana").expect("mana registered");
        assert!(meta.get_json(&store, entity).expect("get_json").is_none());
    }

    #[test]
    fn apply_outcomes_rejects_dead_entities_bad_shapes_and_tick_failures() {
        let (mut store, registry, host, entity) = apply_setup();
        // Unknown entity id.
        store_outcome(
            &host,
            0,
            set_patch(
                Entity::new_with_gen(999, 0),
                "mana",
                serde_json::json!({"mana": 1}),
            )
            .to_string()
            .into_bytes(),
        );
        // Destroyed entity: id recycled, old handle dead.
        let doomed = store.create_entity();
        store.destroy_entity(doomed);
        store_outcome(
            &host,
            1,
            set_patch(doomed, "mana", serde_json::json!({"mana": 1}))
                .to_string()
                .into_bytes(),
        );
        // Schema violation: mana is a u32, not a string.
        store_outcome(
            &host,
            2,
            set_patch(entity, "mana", serde_json::json!({"mana": "lots"}))
                .to_string()
                .into_bytes(),
        );
        // Not JSON at all.
        store_outcome(&host, 3, b"not json".to_vec());
        // Tick itself failed.
        host.outcomes
            .lock()
            .expect("test lock")
            .insert(4, Err("kaboom".into()));
        // No "set" key: skipped silently.
        store_outcome(&host, 5, br#"{"call": 1}"#.to_vec());

        let report = host.apply_outcomes(&mut store, &registry);
        assert_eq!(report.applied(), 0);
        assert_eq!(report.entries.len(), 6);
        assert_eq!(report.error_count(), 5);
        assert!(report.entries[5].errors.is_empty());
    }
}
