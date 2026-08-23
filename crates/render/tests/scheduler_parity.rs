//! Паритет фронтендов планировщика (бэклог #19, антидрейф-канон): одна и
//! та же топология доступов на `ornis_core::Schedule` (системы, ключи
//! TypeId — ресурсы и ленты) и на `ornis_render::FramePlan` (пассы,
//! ключи ResourceId) обязана давать побитово одинаковые уровни: оба
//! потребителя считаются одним движком `ornis-schedule`. Семантический
//! дрейф любой стороны = красный CI.

use ornis_core::{Resources, Schedule, System, SystemAccess};
use ornis_render::{FramePlan, ResourceId, SizePolicy, TextureSpec};

/// Ключи ресурсного пространства имён core (в этом файле — типы-маркеры,
/// реальные singleton-ресурсы плану не нужны).
struct K0;
struct K1;
struct K2;
struct K3;

/// Ключи лент `SmartStore` — второе пространство имён core; у рендера ему
/// аналог не нужен: рендер-пространство (ResourceId) одно, ключи
/// 4..8 просто отображаются на его элементы.
struct L0;
struct L1;
struct L2;
struct L3;

/// Система-заглушка: для паритета нужны только имя и доступы.
struct Stub(&'static str, SystemAccess);

impl System for Stub {
    fn name(&self) -> &'static str {
        self.0
    }

    fn access(&self) -> SystemAccess {
        self.1.clone()
    }

    fn run(&self, _: &Resources) {}
}

fn spec() -> TextureSpec {
    TextureSpec {
        format: wgpu::TextureFormat::Rgba8Unorm,
        samples: 1,
        size: SizePolicy::Fixed {
            width: 4,
            height: 4,
        },
    }
}

/// Ключ 0..8 → декларация core: 0..4 — ресурсное пространство имён,
/// 4..8 — ленты (зеркально элементам r0..r7 на стороне рендера).
fn push_access(access: SystemAccess, key: usize, write: bool) -> SystemAccess {
    match (key, write) {
        (0, false) => access.reads::<K0>(),
        (0, true) => access.writes::<K0>(),
        (1, false) => access.reads::<K1>(),
        (1, true) => access.writes::<K1>(),
        (2, false) => access.reads::<K2>(),
        (2, true) => access.writes::<K2>(),
        (3, false) => access.reads::<K3>(),
        (3, true) => access.writes::<K3>(),
        (4, false) => access.reads_lane::<L0>(),
        (4, true) => access.writes_lane::<L0>(),
        (5, false) => access.reads_lane::<L1>(),
        (5, true) => access.writes_lane::<L1>(),
        (6, false) => access.reads_lane::<L2>(),
        (6, true) => access.writes_lane::<L2>(),
        (7, false) => access.reads_lane::<L3>(),
        (7, true) => access.writes_lane::<L3>(),
        (_, _) => unreachable!("key space is 0..8"),
    }
}

/// Строит зеркальные планы одной топологии `reads`/`writes` (ключи 0..8)
/// на обоих фронтендах и применяет явные именные рёбра; возвращает
/// уровни (core, render).
fn mirrored_levels(
    reads: &[Vec<usize>],
    writes: &[Vec<usize>],
    edges: &[(&'static str, &'static str)],
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    const NAMES: [&str; 12] = [
        "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    ];
    assert_eq!(reads.len(), writes.len(), "parallel access slices");
    // Доменное правило рендера «first touch must be a write»
    // (`FramePlan::build`): ресурс, чьё первое касание — чтение без
    // более ранней (включая собственную) записи, обязан быть импортом.
    // На уровни import не влияет (внепуловость, не семантика доступов),
    // core-сторона такого правила не знает — зеркалим честно.
    let mut import = [false; 8];
    for (key, slot) in import.iter_mut().enumerate() {
        let first_use =
            (0..reads.len()).find(|&i| reads[i].contains(&key) || writes[i].contains(&key));
        if let Some(i) = first_use {
            let written = (0..=i).any(|j| writes[j].contains(&key));
            if reads[i].contains(&key) && !written {
                *slot = true;
            }
        }
    }
    let spec = spec();
    let mut sched = Schedule::new();
    let mut plan = FramePlan::new((640, 480));
    let ids: Vec<ResourceId> = (0..8)
        .map(|i| {
            if import[i] {
                plan.import_resource(format!("r{i}"), spec)
            } else {
                plan.create_resource(format!("r{i}"), spec)
            }
        })
        .collect();
    for i in 0..reads.len() {
        let mut access = SystemAccess::new();
        for &k in &reads[i] {
            access = push_access(access, k, false);
        }
        for &k in &writes[i] {
            access = push_access(access, k, true);
        }
        sched.add_system(Stub(NAMES[i], access));
        let mut pass = plan.add_pass(NAMES[i]);
        for &k in &reads[i] {
            pass = pass.read(ids[k]);
        }
        for &k in &writes[i] {
            pass = pass.write(ids[k]);
        }
    }
    for &(before, after) in edges {
        sched.order_before(before, after);
        plan.order_before_named(before, after);
    }
    (sched.levels(), plan.build().levels())
}

/// Базовые конфликт-классы и явные рёбра: уровни фронтендов идентичны.
#[test]
fn fixed_topologies_match_across_frontends() {
    // Независимые писатели → один уровень.
    let (core, render) =
        mirrored_levels(&[vec![], vec![], vec![]], &[vec![0], vec![1], vec![2]], &[]);
    assert_eq!(core, vec![vec![0, 1, 2]]);
    assert_eq!(core, render);

    // Цепочка RaW.
    let (core, render) = mirrored_levels(
        &[vec![], vec![0], vec![1]],
        &[vec![0], vec![1], vec![]],
        &[],
    );
    assert_eq!(core, vec![vec![0], vec![1], vec![2]]);
    assert_eq!(core, render);

    // WaR (анти-зависимость): читатель первым; ключ из лентового
    // пространства core (7 → r7 у рендера).
    let (core, render) = mirrored_levels(&[vec![7], vec![]], &[vec![], vec![7]], &[]);
    assert_eq!(core, vec![vec![0], vec![1]]);
    assert_eq!(core, render);

    // Ленты (4..8) и ресурсы (0..4) — раздельные пространства имён:
    // ресурсный писатель и лентовый читатель не конфликтуют.
    let (core, render) = mirrored_levels(&[vec![], vec![4]], &[vec![0], vec![]], &[]);
    assert_eq!(core, vec![vec![0, 1]]);
    assert_eq!(core, render);

    // Явное ребро разбивает общий уровень на обоих фронтендах.
    let (core, render) = mirrored_levels(&[vec![], vec![]], &[vec![0], vec![1]], &[("s0", "s1")]);
    assert_eq!(core, vec![vec![0], vec![1]]);
    assert_eq!(core, render);
}

/// Дифференциальный паритет: псевдослучайные срезы доступов по 8 ключам
/// (оба пространства имён core) с явными рёбрами и без — уровни
/// фронтендов обязаны совпадать побитово.
#[test]
fn lcg_scenarios_match_across_frontends() {
    let mut lcg = 0x9E37_79B9u64;
    let mut next = || {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        lcg
    };
    let mut reads: Vec<Vec<usize>> = Vec::new();
    let mut writes: Vec<Vec<usize>> = Vec::new();
    for _ in 0..12 {
        let mut r = Vec::new();
        let mut w = Vec::new();
        for key in 0..8 {
            if next() % 5 == 0 {
                r.push(key);
            }
            if next() % 5 == 0 {
                w.push(key);
            }
        }
        reads.push(r);
        writes.push(w);
    }
    let (core, render) = mirrored_levels(&reads, &writes, &[]);
    assert_eq!(core, render, "паритет без явных рёбер");
    let (core, render) = mirrored_levels(&reads, &writes, &[("s2", "s8"), ("s0", "s11")]);
    assert_eq!(core, render, "паритет с явными рёбрами");
}
