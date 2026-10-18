//! `#[smart_system]` v2 end-to-end (IDEAS §32, RAT-31): per-object
//! kernels with access derived from parameter types.
//!
//! Covers the scalar loop (gather/body/scatter), `if`/`return` filters,
//! `Res`/`ResMut`/event-role parameters, `#[config]` and `#[entity]`
//! parameters, deterministic event application order, and the scheduler
//! level splits (same-lane writers serialize, readers share a level,
//! resource read/write splits across disjoint lane sets).

use ornis_core::{
    Entity, EventReader, EventWriter, Pack, PlainResource, Res, ResMut, Resources, Schedule,
    SmartStore, System, SystemAccess, register_events,
};
use ornis_macros::smart_system;

/// Test bundle: one lane per field (generated wrappers).
#[derive(Clone, Debug, PartialEq, ornis_macros::Pack)]
struct Mote {
    position: f32,
    velocity: f32,
}

/// Second bundle with disjoint lanes (resource-conflict isolation).
#[derive(Clone, Debug, PartialEq, ornis_macros::Pack)]
struct Dust {
    x: f32,
}

/// Plain shared resource (structural derive, no interior mutability).
#[derive(Clone, Debug, PartialEq, ornis_macros::PlainResource)]
struct Wind {
    push: f32,
}

/// Event payload (must be `Clone` for peeking consumers).
#[derive(Clone, Debug, PartialEq)]
struct Ping {
    amount: f32,
}

/// Bundle with an entity link (`Option<Entity>` sparse lane).
#[derive(Clone, Debug, PartialEq, ornis_macros::Pack)]
struct Chaser {
    position: f32,
    target: Option<Entity>,
}

#[smart_system]
fn drift(m: &mut Mote, wind: Res<Wind>) {
    if m.velocity == 0.0 {
        return;
    }
    m.position += m.velocity + wind.push;
}

#[smart_system]
fn drift_again(m: &mut Mote) {
    m.position += 1.0;
}

#[smart_system]
fn inspect_a(m: &Mote) {
    let _ = m.position + m.velocity;
}

#[smart_system]
fn inspect_b(m: &Mote) {
    let _ = m.position - m.velocity;
}

#[smart_system]
fn stir(d: &Dust, wind: ResMut<Wind>) {
    let _ = (d.x, wind.push);
}

#[smart_system]
fn gust(m: &mut Mote, #[config] speed: f32) {
    m.position += speed;
}

#[smart_system]
fn ping(m: &Mote, #[entity] e: Entity, pings: EventWriter<Ping>) {
    if m.position > 0.0 {
        pings.send(e, Ping { amount: 1.0 });
    }
}

#[smart_system]
fn pong(m: &mut Mote, #[entity] e: Entity, pings: EventReader<Ping>) {
    for (to, ping) in pings.read_sorted() {
        if to == e {
            m.velocity += ping.amount;
        }
    }
}

#[smart_system]
fn drain_pings(m: &Mote, pings: EventReader<Ping>) {
    let _ = (m.position, pings.drain_sorted());
}

#[smart_system]
fn chase(c: &mut Chaser) {
    if let Some(t) = c.target {
        c.position += t.position * 0.5;
    }
}

#[smart_system]
fn nudge_position(m: &mut Mote) {
    m.position += 1.0;
}

#[smart_system]
fn nudge_velocity(m: &mut Mote) {
    m.velocity += 1.0;
}

#[smart_system]
fn integrate_mote(m: &mut Mote) {
    m.position += m.velocity * 0.5;
}

#[smart_system]
fn retarget(c: &mut Chaser, #[config] to: Option<Entity>) {
    if c.position >= 0.0 {
        c.target = to;
    }
}

#[smart_system]
fn detach(c: &mut Chaser) {
    if c.position >= 0.0 {
        c.target = None;
    }
}

#[smart_system]
fn forward_fields(m: &mut Mote) {
    m.position += 1.0;
    m.velocity += 1.0;
}

#[smart_system]
fn backward_fields(m: &mut Mote) {
    m.velocity += 1.0;
    m.position += 1.0;
}

fn mote(position: f32, velocity: f32) -> Mote {
    Mote { position, velocity }
}

fn resources_with_motes(motes: &[Mote]) -> (Resources, Vec<Entity>) {
    let mut resources = Resources::new();
    let mut store = SmartStore::new();
    Mote::pack_register(&mut store);
    Dust::pack_register(&mut store);
    let mut entities = Vec::new();
    for mote in motes {
        let entity = store.create_entity();
        mote.pack_insert(&mut store, entity);
        entities.push(entity);
    }
    resources.insert(store);
    resources.insert(Wind { push: 1.0 });
    register_events::<Ping>(&mut resources);
    (resources, entities)
}

fn mote_of(resources: &Resources, entity: Entity) -> Mote {
    let store = resources.get::<SmartStore>().expect("store");
    Mote::pack_get(store, entity).expect("bundle")
}

/// Scalar loop: gather/body/scatter with an `if`/`return` filter.
#[test]
fn per_object_loop_integrates_and_filters() {
    let (resources, entities) = resources_with_motes(&[
        mote(0.0, 2.0),
        mote(10.0, 0.0), // filtered out by the early return
        mote(5.0, -1.0),
    ]);
    let mut schedule = Schedule::new();
    schedule.add_system(DriftSystem::new());
    schedule.run(&resources);

    assert_eq!(mote_of(&resources, entities[0]).position, 0.0 + 2.0 + 1.0);
    assert_eq!(mote_of(&resources, entities[1]).position, 10.0);
    assert_eq!(mote_of(&resources, entities[2]).position, 5.0 - 1.0 + 1.0);
}

/// `#[config]` parameters ride on the system struct.
#[test]
fn config_parameter_applies() {
    let (resources, entities) = resources_with_motes(&[mote(1.0, 0.0)]);
    let mut schedule = Schedule::new();
    schedule.add_system(GustSystem::new(2.5));
    schedule.run(&resources);
    assert_eq!(mote_of(&resources, entities[0]).position, 3.5);
}

/// Two writers over the same bundle never share a level.
#[test]
fn writers_on_same_pack_serialize() {
    let mut schedule = Schedule::new();
    schedule.add_system(DriftSystem::new());
    schedule.add_system(DriftAgainSystem::new());
    assert_eq!(schedule.levels(), vec![vec![0], vec![1]]);
}

/// Two readers over the same bundle share a level.
#[test]
fn readers_on_same_pack_share_a_level() {
    let mut schedule = Schedule::new();
    schedule.add_system(InspectASystem::new());
    schedule.add_system(InspectBSystem::new());
    assert_eq!(schedule.levels(), vec![vec![0, 1]]);
}

/// Resource read/write splits even with disjoint lanes: the conflict
/// comes from `Wind`, not from components.
#[test]
fn resource_read_write_splits_over_disjoint_lanes() {
    let mut schedule = Schedule::new();
    schedule.add_system(DriftSystem::new());
    schedule.add_system(StirSystem::new());
    assert_eq!(schedule.levels(), vec![vec![0], vec![1]]);
}

/// Access derivation is honest: read-only systems declare no writes.
#[test]
fn access_sets_follow_parameter_types() {
    let read = InspectASystem::new().access();
    assert!(read.writes.is_empty() && read.writes_lanes.is_empty());
    assert!(!read.reads_lanes.is_empty());

    let write = DriftSystem::new().access();
    assert!(!write.writes_lanes.is_empty());

    let event_store = std::any::TypeId::of::<ornis_core::EventStore<Ping>>();
    let events_write = PingSystem::new().access();
    assert!(events_write.writes.contains(&event_store));
    let events_read = PongSystem::new().access();
    assert!(events_read.writes.contains(&event_store));
    let events_drain = DrainPingsSystem::new().access();
    assert!(events_drain.writes.contains(&event_store));
}

/// Event queue: a writer produces entries and a reader consumes the ordered
/// snapshot (sorted by entity id, independent of send order).
#[test]
fn events_send_and_apply_in_entity_id_order() {
    // Entities created in reverse position order so arrival order differs
    // from entity-id order; the application must still be deterministic.
    let (resources, entities) = resources_with_motes(&[mote(3.0, 0.0), mote(1.0, 0.0)]);
    let mut schedule = Schedule::new();
    schedule.add_system(PingSystem::new());
    schedule.add_system(PongSystem::new());
    assert_eq!(schedule.levels(), vec![vec![0], vec![1]]);
    schedule.run(&resources);

    // Both motes had position > 0, so both were pinged once.
    assert_eq!(mote_of(&resources, entities[0]).velocity, 1.0);
    assert_eq!(mote_of(&resources, entities[1]).velocity, 1.0);

    // Application order is by entity id: drain and check directly.
    let (resources, _) = resources_with_motes(&[mote(3.0, 0.0), mote(1.0, 0.0)]);
    let mut schedule = Schedule::new();
    schedule.add_system(PingSystem::new());
    schedule.run(&resources);
    let store = resources
        .get::<ornis_core::EventStore<Ping>>()
        .expect("event store");
    let drained = store.drain_sorted();
    let ids: Vec<u32> = drained.iter().map(|(e, _)| e.id()).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted, "drain order must be entity-id order");
    assert!(store.is_empty());
}

/// `System::name` is the function name (ordering/diagnostics key).
#[test]
fn system_name_is_function_name() {
    assert_eq!(DriftSystem::new().name(), "drift");
}

/// Entity links: `t.position` gathers the target's bundle; unlinked
/// entities (`None`) pass through untouched.
#[test]
fn link_read_gathers_target_bundle() {
    let mut resources = Resources::new();
    let mut store = SmartStore::new();
    Chaser::pack_register(&mut store);
    let target = store.create_entity();
    Chaser {
        position: 10.0,
        target: None,
    }
    .pack_insert(&mut store, target);
    let hunter = store.create_entity();
    Chaser {
        position: 0.0,
        target: Some(target),
    }
    .pack_insert(&mut store, hunter);
    let loner = store.create_entity();
    Chaser {
        position: 1.0,
        target: None,
    }
    .pack_insert(&mut store, loner);
    resources.insert(store);

    let mut schedule = Schedule::new();
    schedule.add_system(ChaseSystem::new());
    schedule.run(&resources);

    let store = resources.get::<SmartStore>().expect("store");
    let hunter_body = Chaser::pack_get(store, hunter).expect("hunter bundle");
    assert_eq!(hunter_body.position, 0.0 + 10.0 * 0.5);
    let target_body = Chaser::pack_get(store, target).expect("target bundle");
    assert_eq!(target_body.position, 10.0);
    let loner_body = Chaser::pack_get(store, loner).expect("loner bundle");
    assert_eq!(loner_body.position, 1.0);
}

/// The generated per-object loop produces the same bits as a hand-written
/// lane loop over the same columns.
#[test]
fn generated_loop_matches_manual_lane_loop() {
    let motes: Vec<Mote> = (0..257)
        .map(|i| mote(i as f32 * 0.5, (i % 5) as f32 - 2.0))
        .collect();
    let (generated, generated_entities) = resources_with_motes(&motes);
    let mut schedule = Schedule::new();
    schedule.add_system(IntegrateMoteSystem::new());
    schedule.run(&generated);

    let (manual, manual_entities) = resources_with_motes(&motes);
    {
        let store = manual.get::<SmartStore>().expect("store");
        let mut positions = store
            .write_lane::<Mote__position__PackLane>()
            .expect("position lane");
        let velocities = store
            .read_lane::<Mote__velocity__PackLane>()
            .expect("velocity lane");
        for (position, velocity) in positions.data.iter_mut().zip(velocities.data.iter()) {
            position.0 += velocity.0 * 0.5;
        }
    }

    let generated_store = generated.get::<SmartStore>().expect("store");
    let manual_store = manual.get::<SmartStore>().expect("store");
    for (left, right) in generated_entities.iter().zip(&manual_entities) {
        let a = Mote::pack_get(generated_store, *left).expect("bundle");
        let b = Mote::pack_get(manual_store, *right).expect("bundle");
        assert_eq!(a.position.to_bits(), b.position.to_bits());
        assert_eq!(a.velocity.to_bits(), b.velocity.to_bits());
    }
}

/// Field-level access: a body that touches only `u.position` declares only
/// that lane, so it does not conflict with a writer of `velocity`.
#[test]
fn field_access_declares_only_touched_lanes() {
    let access = NudgePositionSystem::new().access();
    let position = std::any::TypeId::of::<Mote__position__PackLane>();
    let velocity = std::any::TypeId::of::<Mote__velocity__PackLane>();
    assert_eq!(access.writes_lanes.len(), 1);
    assert!(access.writes_lanes.contains(&position));
    assert!(access.reads_lanes.is_empty());
    assert!(!access.writes_lanes.contains(&velocity));
}

/// Writers of disjoint fields of one bundle share a level.
#[test]
fn disjoint_field_writers_share_a_level() {
    let mut schedule = Schedule::new();
    schedule.add_system(NudgePositionSystem::new());
    schedule.add_system(NudgeVelocitySystem::new());
    assert_eq!(schedule.levels(), vec![vec![0, 1]]);
}

/// Both disjoint writers apply to every entity they touch.
#[test]
fn disjoint_field_writers_both_apply() {
    let (resources, entities) = resources_with_motes(&[mote(1.0, 2.0), mote(3.0, 4.0)]);
    let mut schedule = Schedule::new();
    schedule.add_system(NudgePositionSystem::new());
    schedule.add_system(NudgeVelocitySystem::new());
    schedule.run(&resources);

    assert_eq!(mote_of(&resources, entities[0]), mote(2.0, 3.0));
    assert_eq!(mote_of(&resources, entities[1]), mote(4.0, 5.0));
}

/// The touched fields define the entity set: an entity that owns the
/// `position` lane but no `velocity` is visited by a position-only body and
/// skipped by a velocity-only body.
#[test]
fn touched_fields_define_the_entity_set() {
    let (mut resources, entities) = resources_with_motes(&[mote(1.0, 0.0)]);
    let partial = {
        let store = resources.get_mut::<SmartStore>().expect("store");
        let entity = store.create_entity();
        store
            .write_lane::<Mote__position__PackLane>()
            .expect("position lane")
            .insert(entity, Mote__position__PackLane(5.0));
        entity
    };

    let mut schedule = Schedule::new();
    schedule.add_system(NudgePositionSystem::new());
    schedule.run(&resources);

    let store = resources.get::<SmartStore>().expect("store");
    let lane = store.read_lane::<Mote__position__PackLane>().expect("lane");
    assert_eq!(lane.get(partial).map(|p| p.0), Some(6.0));
    assert_eq!(lane.get(entities[0]).map(|p| p.0), Some(2.0));
    drop(lane);

    let mut schedule = Schedule::new();
    schedule.add_system(NudgeVelocitySystem::new());
    schedule.run(&resources);
    let store = resources.get::<SmartStore>().expect("store");
    let velocity = store.read_lane::<Mote__velocity__PackLane>().expect("lane");
    assert!(!velocity.contains(partial));
    assert_eq!(velocity.get(entities[0]).map(|v| v.0), Some(1.0));
}

/// Optional fields: writing `Some` inserts the sparse entry and writing
/// `None` removes it, so `pack_get` reads the absent value back as `None`.
#[test]
fn sparse_field_write_inserts_and_removes_entry() {
    let mut resources = Resources::new();
    let mut store = SmartStore::new();
    Chaser::pack_register(&mut store);
    let other = store.create_entity();
    let subject = store.create_entity();
    Chaser {
        position: 0.0,
        target: None,
    }
    .pack_insert(&mut store, subject);
    resources.insert(store);

    let mut schedule = Schedule::new();
    schedule.add_system(RetargetSystem::new(Some(other)));
    schedule.run(&resources);
    let store = resources.get::<SmartStore>().expect("store");
    assert_eq!(
        Chaser::pack_get(store, subject).expect("bundle").target,
        Some(other)
    );

    let mut schedule = Schedule::new();
    schedule.add_system(DetachSystem::new());
    schedule.run(&resources);
    let store = resources.get::<SmartStore>().expect("store");
    assert_eq!(
        Chaser::pack_get(store, subject).expect("bundle").target,
        None
    );
    let target_lane = store
        .read_lane::<Chaser__target__PackLane>()
        .expect("target lane");
    assert!(!target_lane.contains(subject));
}

/// A self-link reads the target's value before this entity's write-back,
/// the same value the pre-field-access loop observed.
#[test]
fn self_link_reads_pre_body_value() {
    let mut resources = Resources::new();
    let mut store = SmartStore::new();
    Chaser::pack_register(&mut store);
    let entity = store.create_entity();
    Chaser {
        position: 4.0,
        target: Some(entity),
    }
    .pack_insert(&mut store, entity);
    resources.insert(store);

    let mut schedule = Schedule::new();
    schedule.add_system(ChaseSystem::new());
    schedule.run(&resources);

    let store = resources.get::<SmartStore>().expect("store");
    assert_eq!(
        Chaser::pack_get(store, entity).expect("bundle").position,
        4.0 + 4.0 * 0.5
    );
}

/// `Res` requires `PlainResource`: covered at compile time by trybuild
/// (`smart_system_rejects_*`); `SystemAccess` builder parity here.
#[test]
fn plain_resource_bound_is_structural() {
    // Direct impl for coverage of the manual path.
    struct Manual;
    // SAFETY: test-only unit type — reaches no state, so sharing is sound.
    unsafe impl ornis_core::ResMutResource for Manual {}
    // SAFETY: test-only unit type — reaches no state, so sharing is sound.
    unsafe impl PlainResource for Manual {}
    let access = SystemAccess::new().reads::<Manual>();
    assert_eq!(access.reads.len(), 1);
}

/// Two systems over one bundle touch its fields in opposite orders and run
/// concurrently on the same store. Guards are taken in canonical (field-name)
/// order, so neither thread can hold one lane while waiting for the other
/// in reverse; a usage-ordered acquisition deadlocks here and trips the timeout.
#[test]
fn opposite_field_orders_run_concurrently_without_deadlock() {
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    const ITERS: usize = 200;
    let motes: Vec<Mote> = (0..64).map(|i| mote(i as f32, 0.0)).collect();
    let (resources, entities) = resources_with_motes(&motes);
    let resources = Arc::new(resources);
    let (done_tx, done_rx) = mpsc::channel();

    for forward in [true, false] {
        let resources = Arc::clone(&resources);
        let done = done_tx.clone();
        std::thread::spawn(move || {
            for _ in 0..ITERS {
                if forward {
                    ForwardFieldsSystem::new().run(&resources);
                } else {
                    BackwardFieldsSystem::new().run(&resources);
                }
            }
            let _ = done.send(());
        });
    }
    drop(done_tx);

    for _ in 0..2 {
        done_rx
            .recv_timeout(Duration::from_secs(60))
            .expect("opposite field orders deadlocked on lane guards");
    }

    let store = resources.get::<SmartStore>().expect("store");
    let expected = (2 * ITERS) as f32;
    for (i, entity) in entities.iter().enumerate() {
        let got = Mote::pack_get(store, *entity).expect("bundle");
        assert_eq!(got.position, i as f32 + expected);
        assert_eq!(got.velocity, expected);
    }
}
