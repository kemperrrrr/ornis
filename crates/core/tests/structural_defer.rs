//! R4 deferred structural ops end-to-end (RAT-57, issue #115): the
//! `#[smart_system]` `StructuralWriter` parameter, the frame-boundary queue
//! apply, and the engine stage contract.
//!
//! Covers three issue criteria: parity with a manual lane loop (the queued
//! path lands the same bits as direct store writes), deterministic queue
//! order (registration-order level splits + FIFO apply + ordered spawns),
//! and the engine contract (ops queue during `PostFrame` and become visible
//! to `Stage::Last` systems).

use ornis_core::{
    Engine, Entity, Pack, Resources, Schedule, SmartStore, Stage, StructuralWriter, System,
    SystemAccess,
    mutation::{StructuralQueue, apply_structural_queue, register_structural_queue},
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

/// Generated writer: touches the bundle lane and queues a deferred insert
/// whose payload depends on the very field the body mutates.
#[smart_system]
fn queue_buff(u: &mut Mote, #[entity] e: Entity, ops: StructuralWriter) {
    u.hp += 0.5;
    ops.insert(e, Buff { amount: u.hp * 2.0 });
}

/// Three queue writers in one schedule: all conflict on the queue and the
/// `hp` lane, so their levels pin the enqueue order to registration order.
#[smart_system]
fn stamp_one(u: &mut Mote, #[entity] e: Entity, ops: StructuralWriter) {
    u.hp += 1.0;
    ops.insert(e, Buff { amount: 1.0 });
}

#[smart_system]
fn stamp_two(u: &mut Mote, #[entity] e: Entity, ops: StructuralWriter) {
    u.hp += 1.0;
    ops.insert(e, Buff { amount: 2.0 });
}

#[smart_system]
fn stamp_three(u: &mut Mote, #[entity] e: Entity, ops: StructuralWriter) {
    u.hp += 1.0;
    ops.insert(e, Buff { amount: 3.0 });
}

/// Queues two spawns per entity: dense-lane append order must mirror queue
/// order across the apply.
#[smart_system]
fn spawn_wave(u: &mut Mote, ops: StructuralWriter) {
    u.hp += 1.0;
    ops.spawn(Mote { hp: 100.0 });
    ops.spawn(Mote { hp: 200.0 });
}

fn seeded_resources(hps: &[f32]) -> (Resources, Vec<Entity>) {
    let mut resources = Resources::new();
    let mut store = SmartStore::new();
    Mote::pack_register(&mut store);
    let mut entities = Vec::with_capacity(hps.len());
    for &hp in hps {
        let entity = store.create_entity();
        Mote { hp }.pack_insert(&mut store, entity);
        entities.push(entity);
    }
    resources.insert(store);
    register_structural_queue(&mut resources);
    (resources, entities)
}

fn hp_of(resources: &Resources, entity: Entity) -> f32 {
    let store = resources.get::<SmartStore>().expect("store");
    Mote::pack_get(store, entity).expect("mote bundle").hp
}

fn buff_of(resources: &Resources, entity: Entity) -> Option<Buff> {
    let store = resources.get::<SmartStore>().expect("store");
    store
        .read_lane::<Buff>()
        .and_then(|lane| lane.get(entity).copied())
}

/// Parity: the queued `StructuralWriter` path lands byte-identical state to
/// a hand-written lane loop over the same columns (issue #115, criterion 3).
#[test]
fn generated_structural_writer_matches_manual_lane_loop() {
    let hps: Vec<f32> = (0..257).map(|i| i as f32 * 0.25).collect();

    let (mut generated, generated_entities) = seeded_resources(&hps);
    let mut schedule = Schedule::new();
    schedule.add_system(QueueBuffSystem::new());
    schedule.run(&generated);
    apply_structural_queue(&mut generated);

    let (mut manual, manual_entities) = seeded_resources(&hps);
    {
        let store = manual.get_mut::<SmartStore>().expect("store");
        let mut updates = Vec::with_capacity(manual_entities.len());
        {
            let lane = store.read_lane::<Mote__hp__PackLane>().expect("hp lane");
            for &entity in &lane.entities {
                let hp = lane.get(entity).expect("hp entry").0;
                let next = hp + 0.5;
                updates.push((entity, next, Buff { amount: next * 2.0 }));
            }
        }
        for (entity, hp, buff) in updates {
            store
                .write_lane::<Mote__hp__PackLane>()
                .expect("hp lane")
                .insert(entity, Mote__hp__PackLane(hp));
            store.insert(entity, buff);
        }
    }

    assert_eq!(generated_entities, manual_entities, "seeded ids match");
    for (left, right) in generated_entities.iter().zip(&manual_entities) {
        let a = hp_of(&generated, *left);
        let b = hp_of(&manual, *right);
        assert_eq!(a.to_bits(), b.to_bits(), "hp bits diverge at {left:?}");
        let x = buff_of(&generated, *left).expect("generated buff");
        let y = buff_of(&manual, *right).expect("manual buff");
        assert_eq!(
            x.amount.to_bits(),
            y.amount.to_bits(),
            "buff bits diverge at {left:?}"
        );
    }
}

/// Determinism: queueing systems serialize into separate levels in
/// registration order, ops apply FIFO after the run, and the last writer's
/// value wins (issue #115, criterion 2).
#[test]
fn queue_ops_apply_in_registration_order() {
    let hps = [10.0, 20.0];
    let (mut resources, entities) = seeded_resources(&hps);
    let mut schedule = Schedule::new();
    schedule.add_system(StampOneSystem::new());
    schedule.add_system(StampTwoSystem::new());
    schedule.add_system(StampThreeSystem::new());
    assert_eq!(schedule.levels(), vec![vec![0], vec![1], vec![2]]);

    schedule.run(&resources);
    assert!(
        resources
            .get::<StructuralQueue>()
            .expect("queue registered")
            .len()
            == 3 * entities.len(),
        "every writer's ops stay queued during the run"
    );
    assert!(
        buff_of(&resources, entities[0]).is_none(),
        "not applied yet"
    );

    apply_structural_queue(&mut resources);

    for (entity, &seed) in entities.iter().zip(&hps) {
        // Three FIFO insert ops of the same component: the last writer wins.
        assert_eq!(
            buff_of(&resources, *entity),
            Some(Buff { amount: 3.0 }),
            "registration order must survive apply at {entity:?}"
        );
        // Every writer's lane touch landed as well: seed + 1 + 1 + 1.
        assert_eq!(
            hp_of(&resources, *entity).to_bits(),
            (seed + 3.0f32).to_bits()
        );
    }
    assert!(
        resources
            .get::<StructuralQueue>()
            .expect("queue registered")
            .is_empty()
    );
}

/// Spawns queue their ids in call order, so the dense lane receives them in
/// exactly that order at apply time.
#[test]
fn spawn_ops_land_in_queue_order() {
    let (mut resources, entities) = seeded_resources(&[1.0]);
    let mut schedule = Schedule::new();
    schedule.add_system(SpawnWaveSystem::new());
    schedule.run(&resources);
    apply_structural_queue(&mut resources);

    assert_eq!(hp_of(&resources, entities[0]).to_bits(), (2.0f32).to_bits());
    let store = resources.get::<SmartStore>().expect("store");
    let lane = store.read_lane::<Mote__hp__PackLane>().expect("hp lane");
    assert_eq!(
        lane.entities
            .iter()
            .skip(entities.len())
            .map(|entity| lane.get(*entity).expect("hp entry").0)
            .collect::<Vec<_>>(),
        vec![100.0, 200.0],
        "spawn queue order == dense lane append order"
    );
}

/// Engine contract: a smart-system writer in `PostFrame` queues ops that
/// apply between `PostFrame` and `Stage::Last`, where an observer reads the
/// applied component (issue #115, engine integration).
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
    engine.add_stage_system(Stage::PostFrame, QueueBuffSystem::new());
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
