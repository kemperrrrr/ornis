//! `SmartStore` lane granularity in `Schedule` (audit §3.6, backlog #5):
//! systems declare accesses to component lanes on top of the shared
//! store resource; the plan derives conflicts by lanes, enforcement is
//! at the `read_lane`/`write_lane` boundary.

use std::any::TypeId;
use std::sync::Mutex;

use ornis_core::{Entity, Resources, Schedule, SmartStore, System, SystemAccess, compute_levels};

/// Test components (lanes).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Position(f32);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Velocity(f32);

/// Test singleton resources.
struct ResourceA;
struct ResourceB;

/// Resource holding the probe entity.
struct TestEntity(Entity);

/// Commutative event log (commutativity contract of `schedule.rs`).
type Log = Mutex<Vec<&'static str>>;

/// Stub system: name, accesses and body are test data.
struct LaneSystem {
    name: &'static str,
    access: SystemAccess,
    run: fn(&Resources),
}

impl LaneSystem {
    fn new(name: &'static str, access: SystemAccess, run: fn(&Resources)) -> Self {
        Self { name, access, run }
    }
}

impl System for LaneSystem {
    fn name(&self) -> &'static str {
        self.name
    }

    fn access(&self) -> SystemAccess {
        self.access.clone()
    }

    fn run(&self, resources: &Resources) {
        (self.run)(resources);
    }
}

fn noop(_: &Resources) {}

/// Access "read store + read lane `T`" — canonical lane-system form:
/// the store is held as a shared resource, granularity comes from the lane.
fn lane_reads<T: 'static + Send + Sync>() -> SystemAccess {
    SystemAccess::new().reads::<SmartStore>().reads_lane::<T>()
}

/// Access "read store + write lane `T`".
fn lane_writes<T: 'static + Send + Sync>() -> SystemAccess {
    SystemAccess::new().reads::<SmartStore>().writes_lane::<T>()
}

/// Negative case of audit §3.6: the shared store resource alone does not
/// serialize systems — disjoint lanes provide the granularity.
#[test]
fn disjoint_lanes_share_one_level_over_common_store() {
    let mut sched = Schedule::new();
    sched
        .add_system(LaneSystem::new(
            "pos_writer",
            lane_writes::<Position>(),
            noop,
        ))
        .add_system(LaneSystem::new(
            "vel_writer",
            lane_writes::<Velocity>(),
            noop,
        ));
    assert_eq!(sched.levels(), vec![vec![0, 1]]);
}

/// Lane conflict classes: RaW and WaW split systems into separate levels,
/// WaR puts the reader first — tie-break remains registration order.
#[test]
fn lane_conflicts_split_levels() {
    // RaW: lane writer → lane reader.
    let mut sched = Schedule::new();
    sched
        .add_system(LaneSystem::new("w", lane_writes::<Position>(), noop))
        .add_system(LaneSystem::new("r", lane_reads::<Position>(), noop));
    assert_eq!(sched.levels(), vec![vec![0], vec![1]]);

    // WaR: reader first (anti-dependency).
    let mut sched = Schedule::new();
    sched
        .add_system(LaneSystem::new("r", lane_reads::<Position>(), noop))
        .add_system(LaneSystem::new("w", lane_writes::<Position>(), noop));
    assert_eq!(sched.levels(), vec![vec![0], vec![1]]);

    // WaW: two writers to the same lane.
    let mut sched = Schedule::new();
    sched
        .add_system(LaneSystem::new("w1", lane_writes::<Position>(), noop))
        .add_system(LaneSystem::new("w2", lane_writes::<Position>(), noop));
    assert_eq!(sched.levels(), vec![vec![0], vec![1]]);
}

/// Phase B audit criterion: a system over two lanes is included in the
/// shared plan without manual `order_before`; an independent resource
/// system correctly shares a level with it.
#[test]
fn two_lane_system_plans_without_manual_edges() {
    let mut sched = Schedule::new();
    sched
        .add_system(LaneSystem::new("src", lane_writes::<Position>(), noop))
        .add_system(LaneSystem::new(
            "mid",
            SystemAccess::new()
                .reads::<SmartStore>()
                .reads_lane::<Position>()
                .writes_lane::<Velocity>(),
            noop,
        ))
        .add_system(LaneSystem::new("sink", lane_reads::<Velocity>(), noop))
        .add_system(LaneSystem::new(
            "free",
            SystemAccess::new().writes::<ResourceA>(),
            noop,
        ));
    assert_eq!(
        sched.levels(),
        vec![vec![0, 3], vec![1], vec![2]],
        "<two lanes> in a chain without manual edges; resource system is parallel to the source"
    );
}

/// Namespaces are separate: the same type as a singleton resource in one
/// system and a lane in another does not conflict.
#[test]
fn resource_and_lane_namespaces_do_not_conflict() {
    let mut sched = Schedule::new();
    sched
        .add_system(LaneSystem::new(
            "res_writer",
            SystemAccess::new().writes::<Position>(),
            noop,
        ))
        .add_system(LaneSystem::new(
            "lane_reader",
            lane_reads::<Position>(),
            noop,
        ));
    assert_eq!(sched.levels(), vec![vec![0, 1]]);
}

/// Typed and id-based builders produce identical declarations — the id path
/// used by the dynamic frontend via registry F0 (`ComponentMeta::type_id`).
#[test]
fn id_builders_match_typed_builders() {
    let typed = SystemAccess::new()
        .reads_lane::<Position>()
        .writes_lane::<Velocity>();
    let by_id = SystemAccess::new()
        .reads_lane_id(TypeId::of::<Position>())
        .writes_lane_id(TypeId::of::<Velocity>());
    assert_eq!(typed, by_id);
}

/// Differential test: the bitset plan with lanes matches the naive
/// conflict model (RaW/WaR/WaW over resources and lanes, separate
/// namespaces) on pseudo-random access mixtures.
#[test]
fn lane_plan_matches_naive_model() {
    fn naive_conflicts(a: &SystemAccess, b: &SystemAccess) -> bool {
        let writes_touch = |x: &SystemAccess, y: &SystemAccess| {
            x.writes
                .iter()
                .any(|w| y.reads.contains(w) || y.writes.contains(w))
                || x.writes_lanes
                    .iter()
                    .any(|w| y.reads_lanes.contains(w) || y.writes_lanes.contains(w))
        };
        writes_touch(a, b) || writes_touch(b, a)
    }

    let mut lcg = 0x1EAF_AE5Eu64;
    let mut next = || {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        lcg
    };
    const NAMES: [&str; 8] = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"];
    let mut sched = Schedule::new();
    let mut accesses: Vec<SystemAccess> = Vec::new();
    for &name in NAMES.iter() {
        let mut access = SystemAccess::new().reads::<SmartStore>();
        if next() % 2 == 0 {
            access = access.reads::<ResourceA>();
        }
        if next() % 3 == 0 {
            access = access.writes::<ResourceB>();
        }
        if next() % 2 == 0 {
            access = access.reads_lane::<Position>();
        }
        if next() % 3 == 0 {
            access = access.writes_lane::<Velocity>();
        }
        // One type is simultaneously a resource and a lane in different systems —
        // namespaces do not overlap.
        if next() % 4 == 0 {
            access = access.writes::<Position>();
        }
        if next() % 5 == 0 {
            access = access.reads_lane_id(TypeId::of::<ResourceA>());
        }
        sched.add_system(LaneSystem::new(name, access.clone(), noop));
        accesses.push(access);
    }
    let expected = compute_levels(accesses.len(), |i, j| {
        naive_conflicts(&accesses[i], &accesses[j])
    });
    assert_eq!(
        sched.levels(),
        expected,
        "bitset plan with lanes must match the naive model"
    );
}

/// Runs a single probe system with access enforcement enabled.
fn run_probe(access: SystemAccess, run: fn(&Resources)) {
    let mut res = Resources::new();
    let mut store = SmartStore::new();
    store.register::<Position>();
    store.register::<Velocity>();
    res.insert(store);
    let mut sched = Schedule::new();
    sched.set_parallel(false);
    sched.add_system(LaneSystem::new("probe", access, run));
    sched.set_enforce_accesses(true);
    sched.run(&res);
}

fn probe_read_position_lane(res: &Resources) {
    let store = res.get::<SmartStore>().expect("store");
    let _ = store.read_lane::<Position>();
}

fn probe_write_position_lane(res: &Resources) {
    let store = res.get::<SmartStore>().expect("store");
    let _ = store.write_lane::<Position>();
}

#[test]
#[should_panic(expected = "reads lane")]
fn undeclared_lane_read_panics_under_enforcement() {
    // Store resource is declared honestly, lane is not: catches read_lane.
    run_probe(
        SystemAccess::new().reads::<SmartStore>(),
        probe_read_position_lane,
    );
}

#[test]
#[should_panic(expected = "writes lane")]
fn undeclared_lane_write_panics_under_enforcement() {
    run_probe(
        SystemAccess::new().reads::<SmartStore>(),
        probe_write_position_lane,
    );
}

#[test]
#[should_panic(expected = "writes lane")]
fn lane_read_declaration_does_not_cover_writes() {
    // Write strictness: reads_lane does not cover write_lane.
    run_probe(
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<Position>(),
        probe_write_position_lane,
    );
}

#[test]
fn disabled_enforcement_allows_undeclared_lanes() {
    let mut res = Resources::new();
    let mut store = SmartStore::new();
    store.register::<Position>();
    res.insert(store);
    let mut sched = Schedule::new();
    sched.set_parallel(false);
    sched.add_system(LaneSystem::new(
        "probe",
        SystemAccess::new().reads::<SmartStore>(),
        probe_read_position_lane,
    ));
    sched.set_enforce_accesses(false);
    sched.run(&res);
}

/// End-to-end scenario: honest declarations pass enforcement, data flows
/// through lanes in scheduler-derived order (RaW on Velocity: writer
/// must run before reader).
#[test]
fn declared_lanes_pass_enforcement_and_data_flows() {
    let mut res = Resources::new();
    let mut store = SmartStore::new();
    store.register::<Position>();
    store.register::<Velocity>();
    let entity = store.create_entity();
    store.insert(entity, Position(1.0));
    res.insert(store);
    res.insert(TestEntity(entity));
    res.insert(Log::default());

    fn write_velocity(res: &Resources) {
        let entity = res.get::<TestEntity>().unwrap().0;
        let store = res.get::<SmartStore>().unwrap();
        store
            .write_lane::<Velocity>()
            .unwrap()
            .insert(entity, Velocity(2.5));
        res.get::<Log>().unwrap().lock().unwrap().push("w");
    }

    fn read_velocity(res: &Resources) {
        let entity = res.get::<TestEntity>().unwrap().0;
        let store = res.get::<SmartStore>().unwrap();
        let lane = store.read_lane::<Velocity>().unwrap();
        assert_eq!(lane.get(entity), Some(&Velocity(2.5)));
        drop(lane);
        res.get::<Log>().unwrap().lock().unwrap().push("r");
    }

    let mut sched = Schedule::new();
    sched
        .add_system(LaneSystem::new(
            "writer",
            SystemAccess::new()
                .reads::<SmartStore>()
                .reads::<TestEntity>()
                .reads::<Log>()
                .writes_lane::<Velocity>(),
            write_velocity,
        ))
        .add_system(LaneSystem::new(
            "reader",
            SystemAccess::new()
                .reads::<SmartStore>()
                .reads::<TestEntity>()
                .reads::<Log>()
                .reads_lane::<Velocity>(),
            read_velocity,
        ));
    sched.set_enforce_accesses(true);
    sched.run(&res);

    let log = res.get::<Log>().unwrap();
    assert_eq!(
        *log.lock().unwrap(),
        vec!["w", "r"],
        "RaW on Velocity lane forced the writer to a level before the reader"
    );
}
