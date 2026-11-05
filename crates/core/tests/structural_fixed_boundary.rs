//! R4 fixed-tick boundary (issue #115 review blockers 1 and 2).
//!
//! The fixed structural queue must drain on the **tick** boundary, not once
//! per frame: otherwise whether tick `k+1` sees tick `k`'s `add`/`remove`/
//! `spawn` depends on how many ticks the frame happened to run, i.e. on FPS
//! and catch-up. Replay and rollback drive the simulation through
//! `run_fixed_step`, so the same apply must live there too.
//!
//! Every assertion below fails against the original frame-only design.

use ornis_core::{
    Engine, FixedStep, Pack, Resources, SmartStore, Stage, StructuralWriter, System, SystemAccess,
    mutation::{FixedStructuralQueue, StructuralQueue},
};
use std::sync::{Arc, Mutex};

/// Test bundle: one generated lane per field.
#[derive(Clone, Debug, PartialEq, ornis_macros::Pack)]
struct Mote {
    hp: f32,
}

/// Seeds an engine with one `Mote` (hp 0) already in its store.
fn engine_with_one_mote() -> Engine {
    let mut engine = Engine::new();
    let store = engine.world_mut().store_mut().expect("world store");
    Mote::pack_register(store);
    let entity = store.create_entity();
    Mote { hp: 0.0 }.pack_insert(store, entity);
    engine
}

/// Spawns exactly one `Mote` per fixed tick through the fixed writer.
///
/// Written as a plain `System` rather than a `#[smart_system]`: a smart system
/// runs its body once per entity, which would spawn exponentially. One spawn
/// per tick is the shape that made the original design FPS-dependent — tick
/// `k+1` must see tick `k`'s entity.
struct SpawnOnePerTick;

impl System for SpawnOnePerTick {
    fn name(&self) -> &'static str {
        "spawn_one_per_tick"
    }
    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .writes::<FixedStructuralQueue>()
    }
    fn run(&self, resources: &Resources) {
        let writer = StructuralWriter::<FixedStep>::fetch(resources);
        writer.spawn(Mote { hp: 1000.0 });
    }
}

/// Records the `Mote` lane length at the start of each fixed tick.
struct ObserveTick(Arc<Mutex<Vec<usize>>>);

impl System for ObserveTick {
    fn name(&self) -> &'static str {
        "observe_tick"
    }
    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<Mote__hp__PackLane>()
    }
    fn run(&self, resources: &Resources) {
        let store = resources.get::<SmartStore>().expect("store");
        let len = store
            .read_lane::<Mote__hp__PackLane>()
            .map_or(0, |lane| lane.len());
        self.0.lock().expect("trace").push(len);
    }
}

/// Blocker 1 from all three reviews: a `Gameplay` spawn must be seen by the
/// very next fixed tick of the same frame.
#[test]
fn gameplay_spawn_is_visible_to_the_next_fixed_tick() {
    let traces = Arc::new(Mutex::new(Vec::new()));
    let mut engine = engine_with_one_mote();
    engine.add_stage_system(Stage::Gameplay, SpawnOnePerTick);
    engine.add_stage_system(Stage::Gameplay, ObserveTick(traces.clone()));

    // Two fixed ticks in one frame (delta = 2 × fixed step).
    let fixed = engine
        .world()
        .resources()
        .get::<ornis_core::FixedTime>()
        .map_or(1.0 / 60.0, |time| time.delta_seconds());
    engine.run_frame(fixed * 2.0);

    let trace = traces.lock().expect("trace");
    assert_eq!(trace.len(), 2, "the frame ran two fixed ticks");
    assert_eq!(trace[0], 1, "tick 1 sees only the seeded entity");
    assert_eq!(
        trace[1], 2,
        "tick 2 sees tick 1's spawn — the queue drained on the tick boundary"
    );
}

/// An engine whose fixed step is exactly representable in binary (`0.25`), so
/// `run_frame(k * step)` runs exactly `k` ticks with no float drift.
fn engine_with_exact_step() -> (Engine, f32) {
    let step = 0.25_f32;
    let mut engine = engine_with_one_mote();
    let _ = engine
        .world_mut()
        .insert(ornis_core::FixedTime::new(step, 8));
    (engine, step)
}

/// Frame slicing must not change the outcome.
///
/// The comparison is the **per-tick observation trace**, not just the final
/// world: with a frame-only apply both slices still end with the same
/// entities, so only the sequence of what each tick could see reveals the
/// FPS dependence. A tick must observe exactly the spawns of all earlier
/// ticks, whichever frame they fell into.
///
/// The step is exact (see [`engine_with_exact_step`]) and the test asserts the
/// achieved tick totals match before comparing traces.
#[test]
fn fixed_queue_is_independent_of_frame_slicing() {
    fn run_with_slices(slices: &[u32]) -> (u64, Vec<usize>) {
        let (mut engine, step) = engine_with_exact_step();
        let trace = Arc::new(Mutex::new(Vec::new()));
        engine.add_stage_system(Stage::Gameplay, SpawnOnePerTick);
        engine.add_stage_system(Stage::Gameplay, ObserveTick(trace.clone()));
        for &ticks in slices {
            engine.run_frame(step * ticks as f32);
        }
        let ticks_run = engine
            .world()
            .resources()
            .get::<ornis_core::FixedTime>()
            .map_or(0, |time| time.tick());
        let trace = trace.lock().expect("trace").clone();
        (ticks_run, trace)
    }

    // Same tick budget (3), different slicing: one tick per frame, versus a
    // frame with none, one, none and two — the catch-up shape that made the
    // original frame-only apply FPS-dependent.
    let (uniform_ticks, uniform) = run_with_slices(&[1, 1, 1]);
    let (ragged_ticks, ragged) = run_with_slices(&[0, 1, 0, 2]);
    assert_eq!(uniform_ticks, 3, "uniform slicing ran three ticks");
    assert_eq!(
        ragged_ticks, 3,
        "ragged slicing ran three ticks (0+1+0+2 must not lose a tick)"
    );

    // Per-tick visibility: the seed, then one spawn per completed tick.
    assert_eq!(
        uniform,
        vec![1, 2, 3],
        "each tick observes every earlier tick's spawn"
    );
    assert_eq!(
        uniform, ragged,
        "the fixed queue must not depend on how frames were sliced into ticks"
    );
}

/// The per-tick apply is what `run_fixed_step` gives replay/rollback hosts
/// (review blocker 2): the queue must drain there too, not only in `run_frame`.
#[test]
fn run_fixed_step_applies_the_fixed_queue() {
    let mut engine = engine_with_one_mote();
    engine.add_stage_system(Stage::Gameplay, SpawnOnePerTick);

    let tick = engine.run_fixed_step();
    assert_eq!(tick.get(), 1, "one replay tick ran");

    let store = engine.world().store().expect("world store");
    let lane = store.read_lane::<Mote__hp__PackLane>().expect("hp lane");
    assert_eq!(
        lane.len(),
        2,
        "the spawn queued by the replay tick applied at its boundary"
    );
    assert!(
        engine
            .world()
            .resources()
            .get::<FixedStructuralQueue>()
            .expect("queue registered")
            .is_empty(),
        "replay must not leave ops piling up in the queue"
    );
    assert_eq!(
        engine
            .world()
            .resources()
            .get::<ornis_core::FixedTime>()
            .map(|time| (time.steps_this_frame(), time.current_step(), time.alpha())),
        Some((1, 1, 0.0)),
        "the replay tick leaves the canonical clock fields"
    );
}

/// `run_fixed_step` and `run_frame` must reach the same world for the same
/// tick count — the live-recording vs headless-replay parity the networking
/// design requires.
#[test]
fn run_fixed_step_matches_run_frame_for_the_same_ticks() {
    fn world_after(mut engine: Engine, ticks: usize, via_frames: bool) -> Vec<(u32, u32)> {
        {
            let store = engine.world_mut().store_mut().expect("world store");
            Mote::pack_register(store);
            let entity = store.create_entity();
            Mote { hp: 0.0 }.pack_insert(store, entity);
        }
        engine.add_stage_system(Stage::Gameplay, SpawnOnePerTick);
        let fixed = engine
            .world()
            .resources()
            .get::<ornis_core::FixedTime>()
            .map_or(1.0 / 60.0, |time| time.delta_seconds());
        for _ in 0..ticks {
            if via_frames {
                engine.run_frame(fixed);
            } else {
                engine.run_fixed_step();
            }
        }
        let store = engine.world().store().expect("world store");
        let lane = store.read_lane::<Mote__hp__PackLane>().expect("hp lane");
        lane.entities
            .iter()
            .map(|entity| {
                (
                    entity.id(),
                    lane.get(*entity).expect("hp entry").0.to_bits(),
                )
            })
            .collect()
    }

    let frames = world_after(Engine::new(), 5, true);
    let steps = world_after(Engine::new(), 5, false);
    assert_eq!(
        frames, steps,
        "recording (run_frame) and replay (run_fixed_step) must agree"
    );
}
