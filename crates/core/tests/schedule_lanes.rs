//! Гранулярность лент `SmartStore` в `Schedule` (аудит §3.6, бэклог #5):
//! системы объявляют доступы к лентам компонентов поверх общего
//! store-ресурса; план выводит конфликты по лентам, принуждение стоит на
//! границе `read_lane`/`write_lane`.

use std::any::TypeId;
use std::sync::Mutex;

use ornis_core::{Entity, Resources, Schedule, SmartStore, System, SystemAccess, compute_levels};

/// Тестовые компоненты (ленты).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Position(f32);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Velocity(f32);

/// Тестовые singleton-ресурсы.
struct ResourceA;
struct ResourceB;

/// Ресурс с сущностью пробного прогона.
struct TestEntity(Entity);

/// Коммутативный лог событий (контракт коммутативности `schedule.rs`).
type Log = Mutex<Vec<&'static str>>;

/// Система-заглушка: имя, доступы и тело — данные теста.
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

/// Доступ «читаю store + читаю ленту `T`» — каноничная форма системы
/// над лентами: store holdится как общий ресурс, гранулярность лентой.
fn lane_reads<T: 'static + Send + Sync>() -> SystemAccess {
    SystemAccess::new().reads::<SmartStore>().reads_lane::<T>()
}

/// Доступ «читаю store + пишу ленту `T`».
fn lane_writes<T: 'static + Send + Sync>() -> SystemAccess {
    SystemAccess::new().reads::<SmartStore>().writes_lane::<T>()
}

/// Негатив аудита §3.6: общий store-ресурс сам по себе не сериализует
/// системы — гранулярность дают дизъюнктные ленты.
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

/// Conflict-классы по лентам: RaW и WaW разводят системы по уровням,
/// WaR ставит читателя первым — тайбрейк остаётся порядком регистрации.
#[test]
fn lane_conflicts_split_levels() {
    // RaW: писатель ленты → читатель ленты.
    let mut sched = Schedule::new();
    sched
        .add_system(LaneSystem::new("w", lane_writes::<Position>(), noop))
        .add_system(LaneSystem::new("r", lane_reads::<Position>(), noop));
    assert_eq!(sched.levels(), vec![vec![0], vec![1]]);

    // WaR: читатель первым (анти-зависимость).
    let mut sched = Schedule::new();
    sched
        .add_system(LaneSystem::new("r", lane_reads::<Position>(), noop))
        .add_system(LaneSystem::new("w", lane_writes::<Position>(), noop));
    assert_eq!(sched.levels(), vec![vec![0], vec![1]]);

    // WaW: два писателя одной ленты.
    let mut sched = Schedule::new();
    sched
        .add_system(LaneSystem::new("w1", lane_writes::<Position>(), noop))
        .add_system(LaneSystem::new("w2", lane_writes::<Position>(), noop));
    assert_eq!(sched.levels(), vec![vec![0], vec![1]]);
}

/// Критерий Фазы B аудита: система над двумя лентами возвращается в
/// общий план без ручных `order_before`; независимая ресурсная система
/// делит с ней уровень корректно.
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
        "<две ленты> в цепочке без ручных рёбер; ресурсная — параллельна источнику"
    );
}

/// Пространства имён раздельны: один и тот же тип — singleton-ресурс у
/// одной системы и лента у другой — конфликта нет.
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

/// Типизированные и id-билдеры строят одинаковые декларации — id-путь,
/// на который опирается динамический фронтенд через реестр F0
/// (`ComponentMeta::type_id`).
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

/// Дифференциальный тест: битсет-план с лентами совпадает с наивной
/// моделью конфликтов (RaW/WaR/WaW по ресурсам и по лентам, раздельные
/// пространства имён) на псевдослучайных смесях доступов.
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
        // Один тип одновременно ресурсом и лентой у разных систем —
        // пространства имён не пересекаются.
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
        "bitset-план с лентами обязан совпадать с наивной моделью"
    );
}

/// Прогон одной системы-зонда с включённым принуждением доступов.
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
    // Store-ресурс декларирован честно, лента — нет: ловит read_lane.
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
    // Строгость записи: reads_lane не покрывает write_lane.
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

/// Сквозной сценарий: честные декларации проходят принуждение, данные
/// текут через ленты в выведенном планировщиком порядке (RaW по
/// Velocity: писатель обязан отработать раньше читателя).
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
        "RaW по ленте Velocity обязал писателя к уровню раньше читателя"
    );
}
