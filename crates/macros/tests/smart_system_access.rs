//! Shared probe harness for `#[smart_system]` access-set tests (see
//! `REVIEW-smart_system.md`): one world shape plus `run_enforced`, which runs
//! a system under forced access checking and reports the declaration panic
//! (if any) instead of unwinding the test.

use std::sync::Mutex;

use ornis_core::{Resources, Schedule, SmartStore, System};

/// Test position component.
#[derive(Debug, Clone, PartialEq)]
pub struct P {
    /// Horizontal coordinate.
    pub x: f32,
}

/// Test velocity component.
#[derive(Debug, Clone, PartialEq)]
pub struct V {
    /// Horizontal speed.
    pub x: f32,
}

/// Test counter resource (used behind `Mutex` for interior mutability).
#[derive(Debug, Default)]
pub struct Counter(
    /// Current count.
    pub u32,
);

/// Test tuning resource.
#[derive(Debug)]
pub struct Cfg {
    /// Gain applied by probe systems.
    pub k: f32,
}

/// Builds the shared probe world: one entity with `P`/`V`, a `Mutex<Counter>`
/// and a `Cfg` resource.
pub fn world() -> Resources {
    let mut resources = Resources::new();
    let mut store = SmartStore::new();
    store.register::<P>();
    store.register::<V>();
    let e = store.create_entity();
    store.insert(e, P { x: 0.0 });
    store.insert(e, V { x: 1.0 });
    resources.insert(store);
    resources.insert(Mutex::new(Counter::default()));
    resources.insert(Cfg { k: 2.0 });
    resources
}

/// Ok(()) — the system ran under forced access checking; Err(msg) — the
/// declaration panic (`undeclared ...`).
/// Runs one system under forced access checking, reporting the declaration
/// panic (if any) instead of unwinding the test.
pub fn run_enforced<S: System + 'static>(system: S) -> Result<(), String> {
    let resources = world();
    let mut schedule = Schedule::new();
    schedule.set_parallel(false).set_enforce_accesses(true);
    schedule.add_system(system);
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        schedule.run(&resources);
    }))
    .map_err(|payload| {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|text| text.to_string()))
            .unwrap_or_default()
    })
}
