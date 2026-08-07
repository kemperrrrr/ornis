//! Property-тесты ядра ECS (proptest, уровень 2 quality-гейта).
//!
//! Проверяем инварианты, а не примеры: sparse-set ComponentStore
//! (insert/remove/get/iter против модели на HashMap), generational
//! EntityAllocator (recycling слотов, протухшие handle), PageTable
//! (разреженные индексы большого диапазона) и чистую геометрию AABB/Ray.
//!
//! Количество случаев ограничено, чтобы `cargo test` не раздувался.

use std::collections::{HashMap, HashSet};

use ornis_core::{ComponentStore, Entity, EntityAllocator, PageTable};
use ornis_physics::math::{AABB, Ray};
use proptest::prelude::*;

const CASES: u32 = 64;

fn entity_strategy() -> impl Strategy<Value = Entity> {
    (0u32..512, 0u32..4).prop_map(|(id, g)| Entity::new_with_gen(id, g))
}

fn vec3_strategy() -> impl Strategy<Value = glam::Vec3> {
    (-1e4f32..1e4, -1e4f32..1e4, -1e4f32..1e4).prop_map(|(x, y, z)| glam::Vec3::new(x, y, z))
}

/// Операция над ComponentStore для model-based теста.
#[derive(Debug, Clone)]
enum StoreOp {
    Insert(Entity, u64),
    Remove(Entity),
}

fn store_ops_strategy() -> impl Strategy<Value = Vec<StoreOp>> {
    proptest::collection::vec(
        prop_oneof![
            3 => (entity_strategy(), any::<u64>())
                .prop_map(|(e, v)| StoreOp::Insert(e, v)),
            1 => entity_strategy().prop_map(StoreOp::Remove),
        ],
        0..128,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    // ── ComponentStore: sparse-set инварианты ────────────────────────

    /// После insert компонент читается обратно.
    #[test]
    fn insert_then_get_returns_value(e in entity_strategy(), v in any::<u64>()) {
        let mut store = ComponentStore::new();
        store.insert(e, v);
        prop_assert_eq!(store.get(e), Some(&v));
        prop_assert!(store.contains(e));
        prop_assert_eq!(store.len(), 1);
    }

    /// Повторный insert той же entity перезаписывает значение, len не растёт.
    #[test]
    fn insert_overwrite_keeps_len(e in entity_strategy(), v1 in any::<u64>(), v2 in any::<u64>()) {
        let mut store = ComponentStore::new();
        store.insert(e, v1);
        store.insert(e, v2);
        prop_assert_eq!(store.len(), 1);
        prop_assert_eq!(store.get(e), Some(&v2));
    }

    /// После remove компонент не читается, remove возвращает его значение.
    #[test]
    fn remove_returns_value_and_clears(e in entity_strategy(), v in any::<u64>()) {
        let mut store = ComponentStore::new();
        store.insert(e, v);
        prop_assert_eq!(store.remove(e), Some(v));
        prop_assert_eq!(store.get(e), None);
        prop_assert!(!store.contains(e));
        prop_assert_eq!(store.len(), 0);
    }

    /// Удаление отсутствующей entity — None, длина не меняется.
    #[test]
    fn remove_absent_is_none(e in entity_strategy(), absent in entity_strategy(), v in any::<u64>()) {
        prop_assume!(e != absent);
        let mut store = ComponentStore::new();
        store.insert(e, v);
        prop_assert_eq!(store.remove(absent), None);
        prop_assert_eq!(store.len(), 1);
    }

    /// Протухший handle (другая generation) не видит чужой компонент в том же слоте.
    #[test]
    fn stale_handle_is_rejected(id in 0u32..512, v in any::<u64>()) {
        let mut store = ComponentStore::new();
        let live = Entity::new_with_gen(id, 0);
        store.insert(live, v);
        let stale = Entity::new_with_gen(id, 1);
        prop_assert_eq!(store.get(stale), None);
        prop_assert!(!store.contains(stale));
    }

    /// Model-based: случайная смесь insert/remove против HashMap-модели.
    /// Сильнейшее свойство: swap-remove ничего не теряет и не дублирует,
    /// iter покрывает ровно живые entity. Модель: один слот на id,
    /// последний insert выигрывает (включая generation), remove только
    /// живым handle.
    #[test]
    fn store_matches_hashmap_model(ops in store_ops_strategy()) {
        let mut store: ComponentStore<u64> = ComponentStore::new();
        // id → (generation живого handle, значение)
        let mut model: HashMap<u32, (u32, u64)> = HashMap::new();

        for op in ops {
            match op {
                StoreOp::Insert(e, v) => {
                    store.insert(e, v);
                    model.insert(e.id(), (e.generation(), v));
                }
                StoreOp::Remove(e) => {
                    let got = store.remove(e);
                    let expected = match model.get(&e.id()) {
                        Some((g, _)) if *g == e.generation() => {
                            model.remove(&e.id()).map(|(_, v)| v)
                        }
                        _ => None,
                    };
                    prop_assert_eq!(got, expected);
                }
            }
        }

        prop_assert_eq!(store.len(), model.len());
        // iter покрывает ровно живые entity — без пропусков и дублей.
        let iter_entities: HashSet<(u32, u32)> = store
            .entities
            .iter()
            .map(|e| (e.id(), e.generation()))
            .collect();
        prop_assert_eq!(iter_entities.len(), model.len());
        for (id, (g, value)) in &model {
            prop_assert!(iter_entities.contains(&(*id, *g)));
            let e = Entity::new_with_gen(*id, *g);
            prop_assert_eq!(store.get(e), Some(value));
        }
    }

    // ── EntityAllocator: generational indices ────────────────────────

    /// Свежие аллокации живы и уникальны.
    #[test]
    fn fresh_allocations_are_alive_and_unique(n in 1usize..64) {
        let mut alloc = EntityAllocator::new();
        let entities: Vec<Entity> = (0..n).map(|_| alloc.allocate()).collect();
        let ids: HashSet<u32> = entities.iter().map(|e| e.id()).collect();
        prop_assert_eq!(ids.len(), n);
        for e in entities {
            prop_assert!(alloc.is_alive(e));
        }
    }

    /// Переиспользование слота повышает generation, старый handle мёртв.
    #[test]
    fn recycling_bumps_generation(_ in 0u32..1) {
        let mut alloc = EntityAllocator::new();
        let a = alloc.allocate();
        alloc.deallocate(a);
        prop_assert!(!alloc.is_alive(a));
        let b = alloc.allocate();
        prop_assert_eq!(b.id(), a.id());
        prop_assert_eq!(b.generation(), a.generation().wrapping_add(1));
        prop_assert!(alloc.is_alive(b));
        prop_assert!(!alloc.is_alive(a));
    }

    /// Model-based: случайные alloc/dealloc — is_alive согласован с моделью.
    #[test]
    fn allocator_matches_model(ops in proptest::collection::vec(any::<bool>(), 0..64)) {
        let mut alloc = EntityAllocator::new();
        let mut live: Vec<Entity> = Vec::new();

        for do_alloc in ops {
            if do_alloc || live.is_empty() {
                let e = alloc.allocate();
                prop_assert!(alloc.is_alive(e));
                prop_assert!(!live.contains(&e));
                live.push(e);
            } else {
                let victim = live.swap_remove(0);
                alloc.deallocate(victim);
                prop_assert!(!alloc.is_alive(victim));
            }
        }
        for e in &live {
            prop_assert!(alloc.is_alive(*e));
        }
    }

    // ── PageTable: разреженные индексы большого диапазона ────────────

    /// set→get для случайных индексов, разбросанных по многим страницам.
    #[test]
    fn page_table_roundtrip_sparse(
        entries in proptest::collection::hash_map(0usize..100_000, any::<u64>(), 1..64)
    ) {
        let mut table: PageTable<u64> = PageTable::new();
        for (i, v) in &entries {
            table.set(*i, *v);
        }
        for (i, v) in &entries {
            prop_assert_eq!(table.get(*i), Some(v));
        }
    }

    /// get на нетронутой странице — None; на тронутой, но незаписанной
    /// позиции — Some(default).
    #[test]
    fn page_table_untouched_semantics(i in 0usize..100_000, v in any::<u64>()) {
        let mut table: PageTable<u64> = PageTable::new();
        prop_assert_eq!(table.get(i), None);
        table.set(i, v);
        // Соседний слот той же страницы (та же страница, другой offset)
        // уже аллоцирован и возвращает default.
        let neighbour = i ^ 1; // меняем только младший бит — та же страница
        if neighbour != i {
            prop_assert_eq!(table.get(neighbour), Some(&u64::default()));
        }
        prop_assert_eq!(table.get(i), Some(&v));
    }

    // ── physics math: AABB / Ray ─────────────────────────────────────

    /// AABB::from_points содержит каждую точку выборки.
    #[test]
    fn aabb_from_points_contains_all(
        points in proptest::collection::vec(vec3_strategy(), 1..32)
    ) {
        let aabb = AABB::from_points(&points);
        for p in &points {
            prop_assert!(aabb.contains_point(*p));
        }
    }

    /// expand(point) делает AABB содержащим и старые точки, и новую.
    #[test]
    fn aabb_expand_keeps_contents(
        points in proptest::collection::vec(vec3_strategy(), 1..32),
        extra in vec3_strategy(),
    ) {
        let mut aabb = AABB::from_points(&points);
        aabb.expand(extra);
        for p in points.iter().chain(std::iter::once(&extra)) {
            prop_assert!(aabb.contains_point(*p));
        }
    }

    /// overlaps коммутативен.
    #[test]
    fn aabb_overlaps_is_commutative(
        a_min in vec3_strategy(),
        a_size in vec3_strategy(),
        b_min in vec3_strategy(),
        b_size in vec3_strategy(),
    ) {
        let a = AABB::new(a_min, a_min + a_size.abs());
        let b = AABB::new(b_min, b_min + b_size.abs());
        prop_assert_eq!(a.overlaps(&b), b.overlaps(&a));
    }

    /// Ray::point_at(t) == origin + direction * t.
    #[test]
    fn ray_point_at_is_linear(
        origin in vec3_strategy(),
        direction in vec3_strategy(),
        t in -1e3f32..1e3,
    ) {
        let ray = Ray::new(origin, direction);
        let expected = origin + direction * t;
        let got = ray.point_at(t);
        let eps = 1e-3 * (1.0 + expected.length());
        prop_assert!((got - expected).length() <= eps);
    }
}
