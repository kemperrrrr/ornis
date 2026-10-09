//! `#[smart_system]` v2 end-to-end (IDEAS §32, RAT-31): per-object
//! kernels with access derived from parameter types.
//!
//! Covers the scalar loop (gather/body/scatter), `if`/`return` filters,
//! `Res`/`ResMut`/`Events` parameters, `#[config]` and `#[entity]`
//! parameters, deterministic event application order, and the scheduler
//! level splits (same-lane writers serialize, readers share a level,
//! resource read/write splits across disjoint lane sets).

use ornis_core::{
    Entity, Events, Pack, PlainResource, Res, ResMut, Resources, Schedule, SmartStore, System,
    SystemAccess, register_events,
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
fn ping(m: &Mote, #[entity] e: Entity, pings: Events<Ping>) {
    if m.position > 0.0 {
        pings.send(e, Ping { amount: 1.0 });
    }
}

#[smart_system]
fn pong(m: &mut Mote, #[entity] e: Entity, pings: Events<Ping>) {
    for (to, ping) in pings.read_sorted() {
        if to == e {
            m.velocity += ping.amount;
        }
    }
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

    let events_write = PingSystem::new().access();
    assert!(!events_write.writes.is_empty());
    let events_read = PongSystem::new().access();
    assert!(events_read.writes.is_empty() && !events_read.reads.is_empty());
}

/// Events: `send` in one system, deterministic application in another
/// (sorted by entity id, independent of send order).
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

/// `Res` requires `PlainResource`: covered at compile time by trybuild
/// (`smart_system_rejects_*`); `SystemAccess` builder parity here.
#[test]
fn plain_resource_bound_is_structural() {
    // Direct impl for coverage of the manual path.
    struct Manual;
    impl PlainResource for Manual {}
    let access = SystemAccess::new().reads::<Manual>();
    assert_eq!(access.reads.len(), 1);
}
