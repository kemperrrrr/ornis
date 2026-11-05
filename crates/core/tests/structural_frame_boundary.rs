//! R4 frame boundary and registration contract (issue #115).
//!
//! The frame structural queue drains once per frame, between `PostFrame` and
//! `Stage::Last`, and only accepts writers registered outside the fixed plan.
//! The tick-boundary gates live in `structural_fixed_boundary.rs`.

use ornis_core::{
    Engine, Entity, FixedStep, Pack, Resources, Schedule, SmartStore, Stage, StructuralWriter,
    System, SystemAccess,
    mutation::{
        FixedStructuralQueue, FrameStructuralQueue, StructuralQueue, apply_fixed_structural_queue,
        register_structural_queues,
    },
};
use ornis_macros::smart_system;
use std::sync::{Arc, Mutex};

/// Test bundle: one generated lane per field.
#[derive(Clone, Debug, PartialEq, ornis_macros::Pack)]
struct Mote {
    hp: f32,
}

/// Component queued through `StructuralWriter::insert`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Buff {
    amount: f32,
}

/// Frame-plan writer: registered in `PostFrame`, so it addresses the frame
/// queue.
#[smart_system]
fn queue_buff_frame(u: &mut Mote, #[entity] e: Entity, ops: StructuralWriter<FrameStep>) {
    u.hp += 0.5;
    ops.insert(e, Buff { amount: u.hp * 2.0 });
}

/// Fixed-plan writer used only by the registration guard test.
#[smart_system]
fn queue_buff_fixed(u: &mut Mote, #[entity] e: Entity, ops: StructuralWriter<FixedStep>) {
    u.hp += 0.5;
    ops.insert(e, Buff { amount: u.hp * 2.0 });
}

/// Seeds a world holding exactly one `Mote` and registers both queues.
fn seeded_one_mote() -> (Resources, Entity) {
    let mut resources = Resources::new();
    let mut store = SmartStore::new();
    Mote::pack_register(&mut store);
    let entity = store.create_entity();
    Mote { hp: 1.0 }.pack_insert(&mut store, entity);
    resources.insert(store);
    register_structural_queues(&mut resources);
    (resources, entity)
}

/// Spawns exactly one `Mote` per fixed tick through the fixed writer.
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
        StructuralWriter::<FixedStep>::fetch(resources).spawn(Mote { hp: 1000.0 });
    }
}

/// Engine contract: a frame writer in `PostFrame` queues ops that apply
/// between `PostFrame` and `Stage::Last`, where an observer reads the applied
/// component (issue #115, engine integration).
#[test]
fn engine_applies_queued_ops_before_last_stage_systems() {
    /// Reads the queued component inside `Stage::Last`.
    struct ObserveBuff(Arc<Mutex<Option<Option<Buff>>>>);
    impl System for ObserveBuff {
        fn name(&self) -> &'static str {
            "observe_buff"
        }
        fn access(&self) -> SystemAccess {
            SystemAccess::new()
                .reads::<SmartStore>()
                .reads_lane::<Buff>()
        }
        fn run(&self, resources: &Resources) {
            let store = resources.get::<SmartStore>().expect("store");
            let value = store.read_lane::<Buff>().and_then(|lane| {
                let entity = lane.entities.first().copied()?;
                lane.get(entity).copied()
            });
            *self.0.lock().expect("probe lock") = Some(value);
        }
    }

    let probe = Arc::new(Mutex::new(None));
    let mut engine = Engine::new();
    let entity = {
        let store = engine.world_mut().store_mut().expect("world store");
        Mote::pack_register(store);
        let entity = store.create_entity();
        Mote { hp: 2.0 }.pack_insert(store, entity);
        entity
    };
    engine.add_stage_system(Stage::PostFrame, QueueBuffFrameSystem::new());
    engine.add_stage_system(Stage::Last, ObserveBuff(probe.clone()));

    engine.run_frame(1.0 / 60.0);

    // The PostFrame writer bumped hp to 2.5 and queued Buff { 5.0 }.
    assert_eq!(
        *probe.lock().expect("probe lock"),
        Some(Some(Buff { amount: 5.0 })),
        "Last sees the frame's deferred ops"
    );
    let store = engine.world().store().expect("world store");
    assert_eq!(
        Mote::pack_get(store, entity),
        Some(Mote { hp: 2.5 }),
        "PostFrame lane writes landed too"
    );
}

/// Ops queued inside `Last` cannot be applied by the same frame (the one
/// apply already ran); they land at the next frame's apply, in FIFO order.
#[test]
fn ops_queued_in_last_stage_apply_on_the_next_frame() {
    /// Queues a lane write with no observable side effect of its own.
    struct QueueFromLast(Arc<Mutex<usize>>);
    impl System for QueueFromLast {
        fn name(&self) -> &'static str {
            "queue_from_last"
        }
        fn access(&self) -> SystemAccess {
            SystemAccess::new()
                .reads::<SmartStore>()
                .writes::<FrameStructuralQueue>()
        }
        fn run(&self, resources: &Resources) {
            let queue = resources
                .get::<FrameStructuralQueue>()
                .expect("queue registered by World");
            let store = resources.get::<SmartStore>().expect("store");
            let entity = store.create_entity();
            ornis_core::mutation::StructuralQueue::enqueue(
                queue,
                Box::new(move |store: &mut SmartStore| {
                    store.insert(entity, 7_u32);
                }),
            );
            *self.0.lock().expect("count lock") += 1;
        }
    }

    let runs = Arc::new(Mutex::new(0));
    let mut engine = Engine::new();
    engine.add_stage_system(Stage::Last, QueueFromLast(runs.clone()));

    engine.run_frame(1.0 / 60.0);
    assert!(
        engine
            .world()
            .store()
            .expect("store")
            .read_lane::<u32>()
            .is_none(),
        "an op queued in Last waits for the next frame's apply"
    );
    assert_eq!(
        engine
            .world()
            .resources()
            .get::<FrameStructuralQueue>()
            .expect("queue registered")
            .len(),
        1
    );

    engine.run_frame(1.0 / 60.0);
    assert_eq!(
        engine
            .world()
            .store()
            .expect("store")
            .read_lane::<u32>()
            .map(|lane| lane.len()),
        // Frame 1 applied the op queued in frame 0; frame 2's op is still
        // pending, so exactly one entry landed.
        Some(1),
        "the pending op applies at the next frame boundary"
    );
    assert_eq!(*runs.lock().expect("count lock"), 2);
}

/// Registration guard: a fixed writer in a once-per-frame stage panics with a
/// message naming the system and the stage, instead of producing a
/// frame-dependent simulation.
#[test]
#[should_panic(expected = "StructuralWriter<FixedStep>")]
fn fixed_writer_outside_gameplay_is_rejected_at_registration() {
    let mut engine = Engine::new();
    engine.add_stage_system(Stage::PostFrame, QueueBuffFixedSystem::new());
}

/// Registration guard: a frame writer inside the fixed plan would apply at a
/// frame-dependent moment, so it is rejected too.
#[test]
#[should_panic(expected = "StructuralWriter<FrameStep>")]
fn frame_writer_inside_gameplay_is_rejected_at_registration() {
    #[smart_system]
    fn frame_writer(u: &mut Mote, #[entity] e: Entity, ops: StructuralWriter<FrameStep>) {
        u.hp += 1.0;
        ops.insert(e, Buff { amount: 1.0 });
    }

    let mut engine = Engine::new();
    engine.add_stage_system(Stage::Gameplay, FrameWriterSystem::new());
}

/// The two kinds address different queues, so `PostFrame` frame writers and
/// `Gameplay` fixed writers coexist and drain at their own boundaries.
#[test]
fn fixed_and_frame_writers_drain_at_their_own_boundaries() {
    /// Reads both queues' lengths after the frame.
    struct ObserveQueues(Arc<Mutex<(usize, usize)>>);
    impl System for ObserveQueues {
        fn name(&self) -> &'static str {
            "observe_queues"
        }
        fn access(&self) -> SystemAccess {
            SystemAccess::new()
                .reads::<FixedStructuralQueue>()
                .reads::<FrameStructuralQueue>()
        }
        fn run(&self, resources: &Resources) {
            let fixed = resources
                .get::<FixedStructuralQueue>()
                .map_or(0, |queue| queue.len());
            let frame = resources
                .get::<FrameStructuralQueue>()
                .map_or(0, |queue| queue.len());
            *self.0.lock().expect("queues lock") = (fixed, frame);
        }
    }

    let seen = Arc::new(Mutex::new((usize::MAX, usize::MAX)));
    let mut engine = Engine::new();
    engine.add_stage_system(Stage::Gameplay, SpawnOnePerTick);
    engine.add_stage_system(Stage::PostFrame, QueueBuffFrameSystem::new());
    engine.add_stage_system(Stage::Last, ObserveQueues(seen.clone()));

    engine.run_frame(1.0 / 60.0);

    assert_eq!(
        *seen.lock().expect("queues lock"),
        (0, 0),
        "both queues drained before Last, each at its own boundary"
    );
}

/// `spawn_deferred` resolves the handle at apply and leaves nothing
/// observable in between — the alternative to `spawn`'s live-but-empty handle.
#[test]
fn spawn_deferred_reserves_the_id_at_apply() {
    let (mut resources, _entity) = seeded_one_mote();
    let probe = DeferredProbe(Arc::new(Mutex::new(None)));
    resources.insert(probe.clone());
    let mut schedule = Schedule::new();
    schedule.add_system(DeferredSpawnSystem::new(probe));
    schedule.run(&resources);

    let store = resources.get::<SmartStore>().expect("store");
    // Only the seed exists while the op is queued.
    assert_eq!(
        store
            .read_lane::<Mote__hp__PackLane>()
            .map(|lane| lane.len()),
        Some(1),
        "nothing is created before the boundary"
    );
    {
        let probe = resources
            .get::<DeferredProbe>()
            .expect("probe inserted by system setup")
            .0
            .lock()
            .expect("probe lock");
        assert!(
            probe
                .as_ref()
                .expect("the system queued a spawn")
                .try_resolve()
                .is_none(),
            "the pending handle must be unresolved before apply"
        );
    }

    apply_fixed_structural_queue(&mut resources);

    let store = resources.get::<SmartStore>().expect("store");
    assert_eq!(
        store
            .read_lane::<Mote__hp__PackLane>()
            .map(|lane| lane.len()),
        Some(2)
    );
    let resolved = resources
        .get::<DeferredProbe>()
        .expect("probe")
        .0
        .lock()
        .expect("probe lock")
        .as_ref()
        .expect("the system queued a spawn")
        .try_resolve()
        .expect("resolved at apply");
    assert!(store.is_alive(resolved));
    assert_eq!(
        Mote::pack_get(store, resolved),
        Some(Mote { hp: 55.0 }),
        "the bundle landed on the resolved handle"
    );
}

/// Side channel for [`spawn_deferred_reserves_the_id_at_apply`]: stores the
/// pending handle so the test can check it before and after the apply.
#[derive(Clone)]
struct DeferredProbe(Arc<Mutex<Option<ornis_core::PendingEntity>>>);

#[smart_system]
fn deferred_spawn(u: &mut Mote, #[config] probe: DeferredProbe, ops: StructuralWriter) {
    u.hp += 1.0;
    let pending = ops.spawn_deferred(Mote { hp: 55.0 });
    *probe.0.lock().expect("probe lock") = Some(pending);
}

/// `remove` on a lock-free lane must panic at the call site (review note 4),
/// not at a later boundary where the offending system is no longer on the
/// stack.
#[test]
#[should_panic(expected = "cannot target a lock-free lane")]
fn remove_on_a_lock_free_lane_panics_at_the_call_site() {
    let (mut resources, entity) = seeded_one_mote();
    resources
        .get_mut::<SmartStore>()
        .expect("store")
        .register_lock_free::<u64>();
    let writer = StructuralWriter::<FixedStep>::fetch(&resources);
    // The assert fires here, before anything is queued.
    writer.remove::<u64>(entity);
}
