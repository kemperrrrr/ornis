//! Механика планирования уровней параллельности — engine-крейт
//! (Фаза A плана унификации, аудит 2026-08-22 §7).
//!
//! Здесь живёт только доменно-нейтральная часть: узлы с наборами
//! доступов (ключи `K` в `reads`/`writes`), явные рёбра порядка с
//! валидацией и единый [`OrderError`], уровни параллельности
//! ([`compute_levels`] и битсет-вариант [`bitset_level_plan`]), кеш
//! плана с диагностикой ([`PlanCache`]) и уровневый исполнитель
//! ([`run_levels`]: sequential / rayon / wasm-последовательность).
//!
//! Потребители держат доменные данные у себя и собирают срезы ключей:
//! `ornis-core::schedule::Schedule` планирует системы по singleton-
//! ресурсам (ключ — `TypeId`), `ornis-render::RenderGraph` — пассы по
//! текстурным ресурсам (ключ — `ResourceId`). Пул текстур, лайфтаймы,
//! бюджет S4, `Resources`/ECS остаются в доменных крейтах (анти-цели
//! Фазы A: никаких wgpu и ECS в этом крейте).
//!
//! Семантика конфликтов: узлы `i < j` упорядочены, если `i` пишет то,
//! что `j` читает или пишет (RaW/WaW), либо `j` пишет то, что читает
//! `i` (анти-зависимость, WaR); порядок регистрации — тайбрейк, явные
//! рёбра только разбивают уровни. Совпадение битсет-плана с наивной
//! Vec-моделью пинится дифференциальным тестом
//! `bitset_plan_matches_reference_model` ниже; фронтенды гоняют тот же
//! контракт на своих типах (`core::schedule` — одноимённым тестом,
//! `render` — золотыми разбиениями `levels` включая production-граф).

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use fixedbitset::FixedBitSet;
#[cfg(not(target_family = "wasm"))]
use rayon::prelude::*;

/// Движок уровней параллельности — единый для всех шедулеров проекта
/// (IDEAS §28.2: «один scheduler на всё»). `ordered(i, j)` = есть
/// зависимость i→j (i < j, порядок регистрации — тайбрейк); уровень j =
/// 1 + максимум уровней предшественников, группы идут по возрастанию,
/// внутри группы порядок регистрации.
pub fn compute_levels(n: usize, ordered: impl Fn(usize, usize) -> bool) -> Vec<Vec<usize>> {
    let mut level = vec![0usize; n];
    for j in 1..n {
        let mut best = 0usize;
        for (i, prev_level) in level.iter().enumerate().take(j) {
            if ordered(i, j) {
                best = best.max(prev_level + 1);
            }
        }
        level[j] = best;
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, l) in level.iter().enumerate() {
        if groups.len() <= *l {
            groups.resize_with(*l + 1, Vec::new);
        }
        groups[*l].push(i);
    }
    groups
}

/// Уровневый план поверх битсетов: плотный индекс на distinct ключи
/// доступа, пересечения `FixedBitSet` вместо линейных `Vec::contains` в
/// O(n²)-цикле, явные рёбра — матрица смежности. Семантика конфликтов
/// идентична наивной Vec-модели (RaW/WaW/WaR); совпадение пинится
/// тестом `bitset_plan_matches_reference_model`.
///
/// `reads`/`writes` — параллельные срезы по узлам (индекс = узел
/// регистрации). Рёбра вне диапазона узлов игнорируются (валидация
/// рёбер — на уровне API, см. [`resolve_named_edge`]).
pub fn bitset_level_plan<K: Copy + Eq + Hash>(
    reads: &[Vec<K>],
    writes: &[Vec<K>],
    ordering: &[(usize, usize)],
) -> Vec<Vec<usize>> {
    assert_eq!(reads.len(), writes.len(), "parallel access slices");
    let nodes = reads.len();
    let mut resource_ids: HashMap<K, u32> = HashMap::new();
    let mut dense = |id: K| -> usize {
        let next = resource_ids.len() as u32;
        *resource_ids.entry(id).or_insert(next) as usize
    };
    let reads: Vec<FixedBitSet> = reads
        .iter()
        .map(|a| a.iter().map(|&id| dense(id)).collect())
        .collect();
    let writes: Vec<FixedBitSet> = writes
        .iter()
        .map(|a| a.iter().map(|&id| dense(id)).collect())
        .collect();
    let mut successors = vec![FixedBitSet::with_capacity(nodes); nodes];
    for &(before, after) in ordering {
        if before < nodes && after < nodes {
            successors[before].insert(after);
        }
    }
    compute_levels(nodes, |i, j| {
        let disjoint_writes_read = writes[i].is_disjoint(&reads[j]);
        let disjoint_writes_writes = writes[i].is_disjoint(&writes[j]);
        let disjoint_read_writes = reads[i].is_disjoint(&writes[j]);
        !(disjoint_writes_read && disjoint_writes_writes && disjoint_read_writes)
            || successors[i].contains(j)
    })
}

/// Ошибка добавления явного ребра порядка — единый тип для всех
/// фронтендов (раньше почти дословные копии: `OrderError` в core и
/// `GraphOrderError` в render, аудит §4.2). Имена узлов — `String`:
/// системы и пассы равнозначны на этом уровне.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderError {
    /// Узла с таким именем нет (уникальность имён — на вызывающем).
    UnknownNode { name: String },
    /// `after` зарегистрирован раньше `before`: порядок исполнения —
    /// порядок регистрации, явные рёбра только разбивают уровни.
    BackwardEdge { before: String, after: String },
}

impl fmt::Display for OrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderError::UnknownNode { name } => write!(f, "no node named '{name}'"),
            OrderError::BackwardEdge { before, after } => write!(
                f,
                "'{after}' is registered earlier than '{before}' — register nodes in \
                 execution order; explicit edges only split parallel levels"
            ),
        }
    }
}

impl std::error::Error for OrderError {}

/// Name-based валидация ребра: имена → индексы регистрации через
/// `index_of`; неизвестное имя → [`OrderError::UnknownNode`], обратное
/// направление (b ≥ a) → [`OrderError::BackwardEdge`]. Общая часть
/// name-API обоих фронтендов (аудит §4.2).
pub fn resolve_named_edge(
    before: &str,
    after: &str,
    index_of: impl Fn(&str) -> Option<usize>,
) -> Result<(usize, usize), OrderError> {
    let Some(b) = index_of(before) else {
        return Err(OrderError::UnknownNode {
            name: before.to_owned(),
        });
    };
    let Some(a) = index_of(after) else {
        return Err(OrderError::UnknownNode {
            name: after.to_owned(),
        });
    };
    if b >= a {
        return Err(OrderError::BackwardEdge {
            before: before.to_owned(),
            after: after.to_owned(),
        });
    }
    Ok((b, a))
}

/// Index-based валидация ребра: индексы регистрации уже на руках,
/// `name_of` отдаёт имя узла для сообщений (`None` — узла нет, имя в
/// ошибке синтезируется как `"#<index>"`). Общая часть id-API обоих
/// фронтендов.
pub fn validate_indexed_edge(
    before: usize,
    after: usize,
    name_of: impl Fn(usize) -> Option<String>,
) -> Result<(), OrderError> {
    let Some(before_name) = name_of(before) else {
        return Err(OrderError::UnknownNode {
            name: format!("#{before}"),
        });
    };
    let Some(after_name) = name_of(after) else {
        return Err(OrderError::UnknownNode {
            name: format!("#{after}"),
        });
    };
    if before >= after {
        return Err(OrderError::BackwardEdge {
            before: before_name,
            after: after_name,
        });
    }
    Ok(())
}

/// Кеш уровневого плана с диагностикой (стиль S1-кешей проекта):
/// инвалидируется мутациями графа узлов, пересчитывается лениво;
/// счётчик [`PlanCache::computations`] в steady state не растёт.
/// Poisoned Mutex восстанавливается молча — план остаётся корректным
/// (это чистая функция состояния графа).
#[derive(Default)]
pub struct PlanCache {
    cached: Mutex<Option<Vec<Vec<usize>>>>,
    computations: AtomicUsize,
}

impl PlanCache {
    /// Пустой (грязный) кеш.
    pub fn new() -> Self {
        Self::default()
    }

    /// Сбрасывает кеш (вызывается мутациями фронтенда).
    pub fn invalidate(&mut self) {
        *self
            .cached
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }

    /// Кешированный план либо пересчёт `compute` с запоминанием.
    pub fn get_or_compute(&self, compute: impl FnOnce() -> Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        let mut guard = self
            .cached
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(levels) = guard.as_ref() {
            return levels.clone();
        }
        let levels = compute();
        *guard = Some(levels.clone());
        self.computations.fetch_add(1, Ordering::Relaxed);
        levels
    }

    /// Сколько раз план пересчитывался — диагностика кеша.
    pub fn computations(&self) -> usize {
        self.computations.load(Ordering::Relaxed)
    }
}

/// Уровневый исполнитель: уровни — строго последовательно, узлы внутри
/// уровня — параллельно (rayon, поэтому `run` обязан быть `Sync`).
/// `parallel = false` — строгий порядок регистрации 0..nodes (заметьте:
/// он может отличаться от порядка обхода уровней, поэтому узлов
/// посчитано отдельным параметром).
#[cfg(not(target_family = "wasm"))]
pub fn run_levels(levels: &[Vec<usize>], nodes: usize, parallel: bool, run: impl Fn(usize) + Sync) {
    if parallel {
        for group in levels {
            group.par_iter().for_each(|&i| run(i));
        }
        return;
    }
    for i in 0..nodes {
        run(i);
    }
}

/// wasm-вариант [`run_levels`]: rayon-потоков нет — всегда строгий
/// порядок регистрации 0..nodes, поэтому граница `Sync` с `run`
/// снимается (GPU-типы web-бэкенда wgpu не `Sync`; потребитель —
/// запись пассов `ormis-render::graph_frame`, бэклог #19).
#[cfg(target_family = "wasm")]
pub fn run_levels(levels: &[Vec<usize>], nodes: usize, parallel: bool, run: impl Fn(usize)) {
    let _ = (levels, parallel);
    for i in 0..nodes {
        run(i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Наивная Vec-модель конфликтов — эталон дифференциального теста:
    /// i<j упорядочены при RaW/WaW (i пишет в чтения/записи j) или WaR
    /// (j пишет в чтения i); явные рёбра накладываются сверху.
    fn naive_conflicts(a_reads: &[u8], a_writes: &[u8], b_reads: &[u8], b_writes: &[u8]) -> bool {
        a_writes
            .iter()
            .any(|w| b_reads.contains(w) || b_writes.contains(w))
            || b_writes.iter().any(|w| a_reads.contains(w))
    }

    fn plan_via_closure(
        reads: &[Vec<u8>],
        writes: &[Vec<u8>],
        ordering: &[(usize, usize)],
    ) -> Vec<Vec<usize>> {
        compute_levels(reads.len(), |i, j| {
            naive_conflicts(&reads[i], &writes[i], &reads[j], &writes[j])
                || ordering.contains(&(i, j))
        })
    }

    #[test]
    fn independent_writers_share_one_level() {
        let reads: Vec<Vec<u8>> = vec![vec![], vec![], vec![]];
        let writes = vec![vec![0], vec![1], vec![2]];
        assert_eq!(plan_via_closure(&reads, &writes, &[]), vec![vec![0, 1, 2]]);
        assert_eq!(bitset_level_plan(&reads, &writes, &[]), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn chain_is_linear_and_anti_dependency_orders_reader_first() {
        // 0 пишет k0; 1 читает k0 и пишет k1; 2 читает k1.
        let reads: Vec<Vec<u8>> = vec![vec![], vec![0], vec![1]];
        let writes = vec![vec![0], vec![1], vec![]];
        assert_eq!(
            bitset_level_plan(&reads, &writes, &[]),
            vec![vec![0], vec![1], vec![2]]
        );
        // i читает k0, j пишет k0 → сначала читатель (анти-зависимость).
        let reads: Vec<Vec<u8>> = vec![vec![0], vec![]];
        let writes = vec![vec![], vec![0]];
        assert_eq!(
            bitset_level_plan(&reads, &writes, &[]),
            vec![vec![0], vec![1]]
        );
    }

    #[test]
    fn explicit_edges_only_split_levels() {
        let reads: Vec<Vec<u8>> = vec![vec![], vec![]];
        let writes = vec![vec![0], vec![1]];
        assert_eq!(bitset_level_plan(&reads, &writes, &[]), vec![vec![0, 1]]);
        assert_eq!(
            bitset_level_plan(&reads, &writes, &[(0, 1)]),
            vec![vec![0], vec![1]]
        );
    }

    #[test]
    fn out_of_range_edges_are_ignored() {
        let reads: Vec<Vec<u8>> = vec![vec![]];
        let writes = vec![vec![0]];
        assert_eq!(bitset_level_plan(&reads, &writes, &[(3, 7)]), vec![vec![0]]);
        assert_eq!(bitset_level_plan(&reads, &writes, &[(0, 0)]), vec![vec![0]]);
    }

    /// Дифференциальный тест (критерий выхода Фазы A): псевдослучайные
    /// наборы доступов (LCG) — битсет-план обязан совпадать с наивной
    /// Vec-моделью, с явными рёбрами и без.
    #[test]
    fn bitset_plan_matches_reference_model() {
        let mut lcg = 0x5EED_600Du64;
        let mut next = move || {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            lcg
        };
        let mut reads: Vec<Vec<u8>> = Vec::new();
        let mut writes: Vec<Vec<u8>> = Vec::new();
        for _ in 0..12 {
            let mut r = Vec::new();
            let mut w = Vec::new();
            if next() % 2 == 0 {
                r.push((next() % 5) as u8);
            }
            if next() % 3 == 0 {
                r.push((next() % 5) as u8);
            }
            if next() % 2 == 0 {
                w.push((next() % 5) as u8);
            }
            if next() % 3 == 0 {
                w.push((next() % 5) as u8);
            }
            reads.push(r);
            writes.push(w);
        }
        assert_eq!(
            bitset_level_plan(&reads, &writes, &[]),
            plan_via_closure(&reads, &writes, &[]),
            "bitset plan must match the reference model without explicit edges"
        );
        let edges = [(2usize, 8usize), (0, 11)];
        assert_eq!(
            bitset_level_plan(&reads, &writes, &edges),
            plan_via_closure(&reads, &writes, &edges),
            "bitset plan must match the reference model with explicit edges"
        );
    }

    #[test]
    fn named_edge_validates_names_and_direction() {
        let names = ["a", "b"];
        let index_of = |name: &str| names.iter().position(|n| *n == name);
        assert_eq!(resolve_named_edge("a", "b", index_of), Ok((0, 1)));
        assert_eq!(
            resolve_named_edge("b", "a", index_of),
            Err(OrderError::BackwardEdge {
                before: "b".to_owned(),
                after: "a".to_owned(),
            })
        );
        assert_eq!(
            resolve_named_edge("a", "ghost", index_of),
            Err(OrderError::UnknownNode {
                name: "ghost".to_owned(),
            })
        );
    }

    #[test]
    fn indexed_edge_validates_ids_and_direction() {
        let name_of = |i: usize| (i < 2).then(|| format!("n{i}"));
        assert_eq!(validate_indexed_edge(0, 1, name_of), Ok(()));
        assert!(matches!(
            validate_indexed_edge(1, 0, name_of),
            Err(OrderError::BackwardEdge { .. })
        ));
        assert_eq!(
            validate_indexed_edge(0, 99, name_of),
            Err(OrderError::UnknownNode {
                name: "#99".to_owned(),
            })
        );
    }

    #[test]
    fn order_error_display_is_actionable() {
        let unknown = OrderError::UnknownNode {
            name: "ghost".into(),
        };
        assert_eq!(unknown.to_string(), "no node named 'ghost'");
        let backward = OrderError::BackwardEdge {
            before: "a".into(),
            after: "b".into(),
        };
        assert!(backward.to_string().contains("registered earlier"));
    }

    #[test]
    fn plan_cache_hits_and_invalidates() {
        let mut cache = PlanCache::new();
        let compute = || vec![vec![0usize]];
        assert_eq!(cache.computations(), 0);
        assert_eq!(cache.get_or_compute(compute), vec![vec![0]]);
        assert_eq!(cache.get_or_compute(compute), vec![vec![0]]);
        assert_eq!(cache.computations(), 1, "second call is a cache hit");
        cache.invalidate();
        let _ = cache.get_or_compute(compute);
        assert_eq!(cache.computations(), 2, "invalidate forces recompute");
    }

    #[test]
    fn run_levels_sequential_is_registration_order() {
        // Намеренно нехитрый случай, когда обход уровней ≠ порядок
        // регистрации: sequential режим обязан держать последний.
        let trace = Mutex::new(Vec::new());
        run_levels(&[vec![0, 2], vec![1]], 3, false, |i| {
            trace.lock().unwrap().push(i);
        });
        assert_eq!(*trace.lock().unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn run_levels_parallel_covers_same_set() {
        let levels = vec![vec![0, 2], vec![1, 3]];
        let par_trace = Mutex::new(Vec::new());
        run_levels(&levels, 4, true, |i| {
            par_trace.lock().unwrap().push(i);
        });
        let mut par = par_trace.lock().unwrap().clone();
        par.sort_unstable();
        assert_eq!(par, vec![0, 1, 2, 3]);
    }
}
