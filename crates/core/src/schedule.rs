//! Unified system scheduler — S5a foundation (PLAN Приложение C, IDEAS §28.2).
//!
//! Один планировщик на всё: системы объявляют доступы к одиночным
//! ресурсам («миру» из [`Resources`]), планировщик выводит конфликты
//! (read-after-write, write-after-read, write-after-write), раскладывает
//! системы по уровням параллельности и исполняет уровни последовательно,
//! а системы внутри уровня — параллельно (rayon).
//!
//! Детерминизм (Strong Confluence): уровни выведены из порядка
//! регистрации; внутри уровня системы не конфликтуют по *объявленным*
//! доступам (записи дизъюнктны с чужими чтениями/записями), а любые две
//! конфликтующие системы исполняются в порядке регистрации. При честных
//! декларациях это гарантирует отсутствие гонок данных, но **не**
//! побитовую идентичность параллельного и последовательного прогонов —
//! для неё нужен контракт коммутативности ниже.
//!
//! # Контракт коммутативности
//!
//! `reads`-доступ разрешает совместное использование ресурса системами
//! одного уровня, а внутренняя изменяемость (`Mutex`/атомики) делает его
//! **безопасным для памяти — но не детерминированным**: порядок, в
//! котором системы уровня захватывают общий `Mutex<Vec>`, зависит от
//! планировки потоков и различается от прогона к прогону. Поэтому
//! ресурс, объявленный `reads` и мутируемый изнутри, обязан быть
//! **коммутативным аккумулятором**: наблюдаемый результат не должен
//! зависеть от порядка операций (атомичные счётчики с коммутативными
//! операциями; append-мультимножества, сравниваемые с точностью до
//! перестановки). Некоммутативную мутацию («последний писатель
//! выиграл», порядкозависимый `push`, стек) ресурс обязан отражать в
//! `writes` — тогда конфликт виден планировщику и системы разводятся по
//! уровням в порядке регистрации. Границу контракта иллюстрирует тест
//! `parallel_matches_sequential`: события сравниваются отсортированными,
//! то есть перестановка внутри уровня — допустимая часть семантики.
//!
//! Порядок регистрации — тайбрейк, как у пассов рендер-графа: любые два
//! конфликтующих процесса идут в порядке добавления.
//!
//! # Принуждение объявленных доступов
//!
//! Условие «не трогать чужие ресурсы» больше не только на честном слове:
//! пока исполняется система, [`Resources::get`]/[`Resources::contains`]
//! проверяют, что ресурс объявлен в `access()` этой системы (чтение или
//! запись; собственная запись покрывает чтение). Нарушение паникует с
//! именем системы и типом ресурса. Проверка — thread-local стек активных
//! деклараций (RAII, корректно откатывается при паниках и вложенных
//! шедулерах); вне [`Schedule::run`] доступ свободен. По умолчанию
//! включена в debug-сборках и выключена в release
//! ([`Schedule::set_enforce_accesses`] переопределяет).
//!
//! Механика уровней и конфликтов (включая битсет-план и кеш плана с
//! диагностикой), единый [`OrderError`] и уровневый исполнитель живут в
//! крейте `ornis-schedule` (Фаза A, аудит §7); здесь остаётся
//! ECS-фронтенд: [`Resources`], [`SystemAccess`], [`System`] и
//! TLS-enforcement объявленных доступов.

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;

pub use ornis_schedule::{OrderError, compute_levels};
use ornis_schedule::{PlanCache, bitset_level_plan, resolve_named_edge, run_levels};

/// Singleton resource container («мир»): одно значение на тип.
/// Мутация через внутреннюю изменяемость (`Mutex<T>`, атомики) —
/// параллельные системы получают `&Resources`.
#[derive(Default)]
pub struct Resources {
    map: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Resources {
    pub fn new() -> Self {
        Self::default()
    }

    /// Вставляет/заменяет singleton типа `R`; возвращает прежний, если был.
    pub fn insert<R: Any + Send + Sync>(&mut self, resource: R) -> Option<R> {
        self.map
            .insert(TypeId::of::<R>(), Box::new(resource))
            .and_then(|boxed| boxed.downcast::<R>().ok())
            .map(|boxed| *boxed)
    }

    /// Есть ли singleton типа `R`.
    ///
    /// # Panics
    /// Паникует, если вызвана из системы, не декларировавшей `R`
    /// (включённое принуждение доступов; см. модульную документацию).
    pub fn contains<R: Any + Send + Sync>(&self) -> bool {
        assert_access_declared::<R>();
        self.map.contains_key(&TypeId::of::<R>())
    }

    /// Общий доступ к singleton типа `R`.
    ///
    /// # Panics
    /// Паникует, если вызвана из системы, не декларировавшей `R`
    /// (чтение или запись; включённое принуждение доступов).
    pub fn get<R: Any + Send + Sync>(&self) -> Option<&R> {
        assert_access_declared::<R>();
        self.map
            .get(&TypeId::of::<R>())
            .and_then(|boxed| boxed.downcast_ref::<R>())
    }

    /// Мутабельный доступ (только между запусками шедулера, не из
    /// параллельных систем).
    pub fn get_mut<R: Any + Send + Sync>(&mut self) -> Option<&mut R> {
        self.map
            .get_mut(&TypeId::of::<R>())
            .and_then(|boxed| boxed.downcast_mut::<R>())
    }

    /// Удаляет singleton типа `R`.
    pub fn remove<R: Any + Send + Sync>(&mut self) -> Option<R> {
        self.map
            .remove(&TypeId::of::<R>())
            .and_then(|boxed| boxed.downcast::<R>().ok())
            .map(|boxed| *boxed)
    }

    /// Число singleton-ресурсов.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Пуст ли мир.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Объявленные доступы системы: чтения/записи singleton-ресурсов по типам.
/// Раньше назывался `Access` — переименован, чтобы не конфликтовать с
/// типовым `ornis_render::Access` (ZST-маркеры `Read`/`Write` графа).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SystemAccess {
    pub reads: Vec<TypeId>,
    pub writes: Vec<TypeId>,
}

impl SystemAccess {
    pub fn new() -> Self {
        Self::default()
    }

    /// Добавляет `R` в набор чтений.
    pub fn reads<R: Any + Send + Sync>(mut self) -> Self {
        self.reads.push(TypeId::of::<R>());
        self
    }

    /// Добавляет `R` в набор записей.
    pub fn writes<R: Any + Send + Sync>(mut self) -> Self {
        self.writes.push(TypeId::of::<R>());
        self
    }

    /// Конфликтует ли запись с (чтение ∪ запись) другого доступа.
    #[cfg(test)]
    fn writes_touch(&self, other: &SystemAccess) -> bool {
        self.writes
            .iter()
            .any(|w| other.reads.contains(w) || other.writes.contains(w))
    }
}

/// Система единого шедулера: доступы — данные, исполнение — над миром.
pub trait System: Send + Sync {
    fn name(&self) -> &'static str;
    fn access(&self) -> SystemAccess;
    fn run(&self, resources: &Resources);
}

/// Конфликт двух систем (в порядке регистрации i < j): упорядочены, если
/// i пишет то, что j читает/пишет (RaW/WaW), или j пишет то, что читает
/// i (анти-зависимость, WaR).
#[cfg(test)]
fn conflicts(a: &SystemAccess, b: &SystemAccess) -> bool {
    a.writes_touch(b) || b.writes_touch(a)
}

/// Уровни параллельности: системы без конфликтов внутри уровня;
/// уровни упорядочены зависимостями. Детерминировано порядком регистрации.
///
/// Эталонная Vec-реализация: продакшн-путь [`Schedule`] использует
/// битсет-проекцию тех же конфликтов; тест `bitset_plan_matches_reference_model`
/// пинит совпадение.
#[cfg(test)]
fn reference_level_groups(
    accesses: &[SystemAccess],
    ordering: &[(usize, usize)],
) -> Vec<Vec<usize>> {
    compute_levels(accesses.len(), |i, j| {
        conflicts(&accesses[i], &accesses[j]) || ordering.contains(&(i, j))
    })
}

/// Активная декларация доступов исполняемой системы (принуждение).
struct AccessFrame {
    system: &'static str,
    reads: Vec<TypeId>,
    writes: Vec<TypeId>,
}

thread_local! {
    /// Стек активных деклараций: пуст вне `Schedule::run` (или когда
    /// принуждение выключено) — тогда `Resources` unrestricted.
    static ACCESS_FRAMES: RefCell<Vec<AccessFrame>> = const { RefCell::new(Vec::new()) };
}

/// RAII-пуш декларации в [`ACCESS_FRAMES`]; снимается при выходе из
/// области видимости, включая раскрутку паники (rayon переиспользует
/// потоки — без RAII стек протухал бы).
struct AccessFrameGuard;

impl AccessFrameGuard {
    fn push(system: &'static str, access: &SystemAccess) -> Self {
        ACCESS_FRAMES.with(|frames| {
            frames.borrow_mut().push(AccessFrame {
                system,
                reads: access.reads.clone(),
                writes: access.writes.clone(),
            });
        });
        AccessFrameGuard
    }
}

impl Drop for AccessFrameGuard {
    fn drop(&mut self) {
        ACCESS_FRAMES.with(|frames| {
            frames.borrow_mut().pop();
        });
    }
}

/// Проверяет `R` против декларации текущей исполняемой системы; вне
/// [`Schedule::run`] — no-op. Нарушение = нарушение контракта
/// детерминизма: недекларированный доступ может гоняться с системами
/// того же параллельного уровня.
fn assert_access_declared<R: Any + Send + Sync>() {
    ACCESS_FRAMES.with(|frames| {
        let frames = frames.borrow();
        let Some(frame) = frames.last() else {
            return;
        };
        let id = TypeId::of::<R>();
        let declared = frame.reads.contains(&id) || frame.writes.contains(&id);
        if !declared {
            panic!(
                "system '{}' reads resource '{}' that is not declared in its access set \
                 (SystemAccess::reads/writes) — undeclared access breaks the deterministic \
                 schedule contract",
                frame.system,
                std::any::type_name::<R>()
            );
        }
    });
}

/// Расписание систем: порядок регистрации — тайбрейк конфликтов.
pub struct Schedule {
    systems: Vec<Box<dyn System>>,
    accesses: Vec<SystemAccess>,
    /// S5c: явные рёбра порядка (индексы регистрации i < j) поверх
    /// выведенных из доступов — например, для скрытых зависимостей
    /// (общие queue-буферы), которых не видно в множествах доступа.
    ordering: Vec<(usize, usize)>,
    parallel: bool,
    enforce_accesses: bool,
    /// Кеш уровневого плана с диагностикой (зеркалит S1-кеш
    /// `GraphLayout` рендера): пересчитывается только после
    /// `add_system`/`order_before`.
    plan: PlanCache,
}

impl Default for Schedule {
    fn default() -> Self {
        Self::new()
    }
}

impl Schedule {
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            accesses: Vec::new(),
            ordering: Vec::new(),
            parallel: true,
            enforce_accesses: cfg!(debug_assertions),
            plan: PlanCache::new(),
        }
    }

    /// Добавляет систему (порядок = приоритет при конфликтах).
    pub fn add_system<S: System + 'static>(&mut self, system: S) -> &mut Self {
        self.accesses.push(system.access());
        self.systems.push(Box::new(system));
        self.invalidate_plan();
        self
    }

    fn invalidate_plan(&mut self) {
        self.plan.invalidate();
    }

    /// S5c: объявляет, что система `before` обязана выполниться раньше
    /// системы `after`, даже если их доступы не конфликтуют (скрытая
    /// зависимость). Обе ищутся по `name()`.
    ///
    /// # Panics
    /// Паникует, если имя не найдено (уникальность имён — на вызывающем)
    /// или `after` зарегистрирована раньше `before`: порядок исполнения —
    /// порядок регистрации (S3), явные рёбра только разбивают уровни.
    pub fn order_before(&mut self, before: &str, after: &str) -> &mut Self {
        self.try_order_before(before, after)
            .unwrap_or_else(|error| panic!("order_before('{before}', '{after}'): {error}"))
    }

    /// Мягкая [`Schedule::order_before`]: ошибка — возвращаемое
    /// [`OrderError`], не паника (для динамических фронтендов фазы 6).
    pub fn try_order_before(&mut self, before: &str, after: &str) -> Result<&mut Self, OrderError> {
        let (b, a) = resolve_named_edge(before, after, |name| self.try_system_index(name))?;
        if !self.ordering.contains(&(b, a)) {
            self.ordering.push((b, a));
            self.invalidate_plan();
        }
        Ok(self)
    }

    /// S5c: зеркальный [`Schedule::order_before`].
    pub fn order_after(&mut self, after: &str, before: &str) -> &mut Self {
        self.order_before(before, after)
    }

    /// Мягкая [`Schedule::order_after`].
    pub fn try_order_after(&mut self, after: &str, before: &str) -> Result<&mut Self, OrderError> {
        self.try_order_before(before, after)
    }

    fn try_system_index(&self, name: &str) -> Option<usize> {
        self.systems.iter().position(|sys| sys.name() == name)
    }

    /// Параллельное (true, по умолчанию) или строго последовательное
    /// (bit-identical порядок регистрации) исполнение.
    pub fn set_parallel(&mut self, parallel: bool) -> &mut Self {
        self.parallel = parallel;
        self
    }

    /// Принуждение объявленных доступов (см. модульную документацию):
    /// по умолчанию включено в debug-сборках, выключено в release.
    pub fn set_enforce_accesses(&mut self, enforce: bool) -> &mut Self {
        self.enforce_accesses = enforce;
        self
    }

    /// Число систем.
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// Пусто ли расписание.
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    /// Уровни параллельности (индексы систем в порядке регистрации).
    pub fn level_groups(&self) -> Vec<Vec<usize>> {
        self.cached_levels()
    }

    /// Сколько раз уровневый план пересчитан — диагностика кеша
    /// (в steady state не растёт).
    pub fn level_computations(&self) -> usize {
        self.plan.computations()
    }

    fn cached_levels(&self) -> Vec<Vec<usize>> {
        let reads: Vec<Vec<TypeId>> = self.accesses.iter().map(|a| a.reads.clone()).collect();
        let writes: Vec<Vec<TypeId>> = self.accesses.iter().map(|a| a.writes.clone()).collect();
        self.plan.get_or_compute(|| bitset_level_plan(&reads, &writes, &self.ordering))
    }

    /// Исполняет расписание над миром.
    pub fn run(&self, resources: &Resources) {
        // Sequential режим не считает план вовсе (как и раньше).
        let levels = if self.parallel {
            self.cached_levels()
        } else {
            Vec::new()
        };
        run_levels(&levels, self.systems.len(), self.parallel, |i| {
            self.run_system(i, resources);
        });
    }

    fn run_system(&self, i: usize, resources: &Resources) {
        let system = &self.systems[i];
        if self.enforce_accesses {
            let _frame = AccessFrameGuard::push(system.name(), &self.accesses[i]);
            system.run(resources);
        } else {
            system.run(resources);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct A;
    struct B;
    struct C;

    fn reads_writes<R: Any + Send + Sync>(reads: bool, writes: bool) -> SystemAccess {
        let mut a = SystemAccess::new();
        if reads {
            a = a.reads::<R>();
        }
        if writes {
            a = a.writes::<R>();
        }
        a
    }

    /// Тестовая система-заглушка с фиксированным именем и доступами.
    struct NamedNoop(&'static str, SystemAccess);

    impl System for NamedNoop {
        fn name(&self) -> &'static str {
            self.0
        }
        fn access(&self) -> SystemAccess {
            self.1.clone()
        }
        fn run(&self, _: &Resources) {}
    }

    #[test]
    fn resources_singleton_per_type() {
        let mut res = Resources::new();
        assert!(res.is_empty());
        res.insert(7u32);
        assert_eq!(res.insert(9u32), Some(7));
        assert_eq!(res.get::<u32>(), Some(&9));
        *res.get_mut::<u32>().unwrap() = 11;
        assert_eq!(res.remove::<u32>(), Some(11));
        assert!(!res.contains::<u32>());
    }

    #[test]
    fn diamond_yields_parallel_level() {
        // A пишет X; B и C читают X (пишут Y/Z); D читает Y и Z.
        let accesses = vec![
            reads_writes::<A>(false, true),
            reads_writes::<B>(false, true),
            reads_writes::<C>(false, true),
        ];
        // независимые писатели → один уровень
        assert_eq!(reference_level_groups(&accesses, &[]), vec![vec![0, 1, 2]]);

        let chain = vec![
            SystemAccess::new().writes::<A>(),
            SystemAccess::new().reads::<A>().writes::<B>(),
            SystemAccess::new().reads::<B>().writes::<C>(),
        ];
        assert_eq!(
            reference_level_groups(&chain, &[]),
            vec![vec![0], vec![1], vec![2]]
        );
    }

    #[test]
    fn anti_dependency_orders_reader_first() {
        // i читает X, j пишет X → i раньше j (анти-зависимость).
        let accesses = vec![
            SystemAccess::new().reads::<A>(),
            SystemAccess::new().writes::<A>(),
        ];
        assert_eq!(
            reference_level_groups(&accesses, &[]),
            vec![vec![0], vec![1]]
        );
    }

    struct Bump {
        name: &'static str,
        access: SystemAccess,
        target: &'static str,
    }

    impl System for Bump {
        fn name(&self) -> &'static str {
            self.name
        }
        fn access(&self) -> SystemAccess {
            self.access.clone()
        }
        fn run(&self, resources: &Resources) {
            let counter = resources
                .get::<Mutex<Vec<&'static str>>>()
                .expect("log resource");
            let mut log = counter.lock().expect("log lock");
            log.push(self.target);
        }
    }

    /// Тип лог-ресурса тестов ниже (декларируется как чтение).
    type Log = Mutex<Vec<&'static str>>;

    #[test]
    fn sequential_mode_is_registration_order() {
        let mut res = Resources::new();
        res.insert(Log::default());
        let mut sched = Schedule::new();
        sched
            .add_system(Bump {
                name: "w",
                access: SystemAccess::new().writes::<A>().reads::<Log>(),
                target: "w",
            })
            .add_system(Bump {
                name: "r",
                access: SystemAccess::new()
                    .reads::<A>()
                    .writes::<B>()
                    .reads::<Log>(),
                target: "r",
            });
        sched.set_parallel(false);
        sched.run(&res);
        let log = res.get::<Log>().unwrap();
        assert_eq!(*log.lock().unwrap(), vec!["w", "r"]);
    }

    /// Strong Confluence: параллельный прогон даёт тот же результат, что
    /// последовательный, при любом числе потоков.
    #[test]
    fn parallel_matches_sequential() {
        #[derive(Default)]
        struct Counters {
            sum: AtomicU64,
            events: Mutex<Vec<&'static str>>,
        }
        struct Add {
            access: SystemAccess,
            name: &'static str,
        }
        impl System for Add {
            fn name(&self) -> &'static str {
                self.name
            }
            fn access(&self) -> SystemAccess {
                self.access.clone()
            }
            fn run(&self, resources: &Resources) {
                let c = resources.get::<Counters>().expect("counters");
                c.sum.fetch_add(1, Ordering::Relaxed);
                c.events.lock().unwrap().push(self.name);
            }
        }

        let build = |parallel: bool| -> (u64, Vec<&'static str>) {
            let mut res = Resources::new();
            res.insert(Counters::default());
            let mut sched = Schedule::new();
            // ромб: источник → два независимых узла → сборщик
            sched
                .add_system(Add {
                    access: SystemAccess::new().writes::<A>().reads::<Counters>(),
                    name: "src",
                })
                .add_system(Add {
                    access: SystemAccess::new()
                        .reads::<A>()
                        .writes::<B>()
                        .reads::<Counters>(),
                    name: "left",
                })
                .add_system(Add {
                    access: SystemAccess::new()
                        .reads::<A>()
                        .writes::<C>()
                        .reads::<Counters>(),
                    name: "right",
                })
                .add_system(Add {
                    access: SystemAccess::new()
                        .reads::<B>()
                        .reads::<C>()
                        .writes::<A>()
                        .reads::<Counters>(),
                    name: "sink",
                });
            sched.set_parallel(parallel);
            sched.run(&res);
            let c = res.remove::<Counters>().unwrap();
            let mut events = c.events.into_inner().unwrap();
            events.sort();
            (c.sum.into_inner(), events)
        };

        let (seq_sum, seq_events) = build(false);
        let (par_sum, par_events) = build(true);
        assert_eq!(seq_sum, 4);
        assert_eq!(par_sum, 4);
        // Мультимножество событий совпадает; порядок внутри уровня
        // недетерминирован — потому сравниваем отсортированными.
        assert_eq!(seq_events, par_events);
    }

    #[test]
    fn explicit_ordering_splits_a_level() {
        // Два независимых писателя делят уровень; явное ребро разводит их.
        let mut sched = Schedule::new();
        sched
            .add_system(NamedNoop("first", SystemAccess::new().writes::<A>()))
            .add_system(NamedNoop("second", SystemAccess::new().writes::<B>()));
        assert_eq!(sched.level_groups(), vec![vec![0, 1]]);
        sched.order_before("first", "second");
        assert_eq!(sched.level_groups(), vec![vec![0], vec![1]]);
    }

    #[test]
    #[should_panic(expected = "registered")]
    fn explicit_ordering_rejects_backward_direction() {
        let mut sched = Schedule::new();
        sched
            .add_system(NamedNoop("a", SystemAccess::new()))
            .add_system(NamedNoop("b", SystemAccess::new()));
        sched.order_before("b", "a");
    }

    #[test]
    #[should_panic(expected = "no node named")]
    fn explicit_ordering_unknown_name_panics() {
        let mut sched = Schedule::new();
        sched.add_system(NamedNoop("only", SystemAccess::new()));
        sched.order_before("only", "ghost");
    }

    #[test]
    fn try_order_before_reports_errors_without_panicking() {
        let mut sched = Schedule::new();
        sched
            .add_system(NamedNoop("a", SystemAccess::new().writes::<A>()))
            .add_system(NamedNoop("b", SystemAccess::new().writes::<B>()));
        assert!(matches!(
            sched.try_order_before("b", "a"),
            Err(OrderError::BackwardEdge { .. })
        ));
        assert_eq!(
            sched.try_order_before("a", "ghost").map(|_| ()),
            Err(OrderError::UnknownNode {
                name: "ghost".to_owned(),
            })
        );
        // План не тронут ошибками; успешное ребро разбивает уровень.
        assert_eq!(sched.level_groups(), vec![vec![0, 1]]);
        assert!(sched.try_order_before("a", "b").is_ok());
        assert_eq!(sched.level_groups(), vec![vec![0], vec![1]]);
    }

    #[test]
    fn schedule_levels_diamond() {
        let mut sched = Schedule::new();
        sched
            .add_system(NamedNoop("src", SystemAccess::new().writes::<A>()))
            .add_system(NamedNoop(
                "left",
                SystemAccess::new().reads::<A>().writes::<B>(),
            ))
            .add_system(NamedNoop(
                "right",
                SystemAccess::new().reads::<A>().writes::<C>(),
            ))
            .add_system(NamedNoop(
                "sink",
                SystemAccess::new().reads::<B>().reads::<C>().writes::<A>(),
            ));
        assert_eq!(sched.level_groups(), vec![vec![0], vec![1, 2], vec![3]]);
    }

    #[test]
    fn level_plan_is_cached_until_mutation() {
        let mut sched = Schedule::new();
        for name in ["a", "b", "c"] {
            sched.add_system(NamedNoop(name, SystemAccess::new().writes::<A>()));
        }
        let res = Resources::new();
        assert_eq!(sched.level_computations(), 0);
        sched.run(&res);
        assert_eq!(sched.level_computations(), 1);
        sched.run(&res);
        sched.run(&res);
        assert_eq!(
            sched.level_computations(),
            1,
            "steady state reuses the cached plan"
        );
        sched.add_system(NamedNoop("d", SystemAccess::new().reads::<A>()));
        sched.run(&res);
        assert_eq!(
            sched.level_computations(),
            2,
            "add_system invalidates the plan"
        );
        assert_eq!(
            sched.level_groups(),
            vec![vec![0], vec![1], vec![2], vec![3]]
        );
    }

    #[test]
    fn bitset_plan_matches_reference_model() {
        // Псевдослучайные наборы доступов (LCG): битсет-план обязан
        // совпадать с эталонной Vec-реализацией (конфликты → уровни).
        let mut lcg = 0x5EED_600Du64;
        let mut next = move || {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            lcg
        };
        const NAMES: [&str; 12] = [
            "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
        ];
        let mut sched = Schedule::new();
        let mut accesses: Vec<SystemAccess> = Vec::new();
        for &name in NAMES.iter() {
            let mut access = SystemAccess::new();
            if next() % 2 == 0 {
                access = access.reads::<A>();
            }
            if next() % 3 == 0 {
                access = access.reads::<B>();
            }
            if next() % 2 == 0 {
                access = access.writes::<C>();
            }
            if next() % 3 == 0 {
                access = access.writes::<A>();
            }
            sched.add_system(NamedNoop(name, access.clone()));
            accesses.push(access);
        }
        assert_eq!(
            sched.level_groups(),
            reference_level_groups(&accesses, &[]),
            "bitset plan must match the reference model without explicit edges"
        );
        sched.order_before("s2", "s8");
        assert_eq!(
            sched.level_groups(),
            reference_level_groups(&accesses, &[(2, 8)]),
            "bitset plan must match the reference model with an explicit edge"
        );
    }

    #[test]
    #[should_panic(expected = "not declared")]
    fn undeclared_read_panics_under_enforcement() {
        struct Sneaky;
        impl System for Sneaky {
            fn name(&self) -> &'static str {
                "sneaky"
            }
            fn access(&self) -> SystemAccess {
                SystemAccess::new().writes::<A>()
            }
            fn run(&self, resources: &Resources) {
                // Читает B, не декларировав его, — нарушение контракта.
                let _ = resources.get::<B>();
            }
        }
        let mut res = Resources::new();
        res.insert(B);
        let mut sched = Schedule::new();
        sched.add_system(Sneaky).set_enforce_accesses(true);
        sched.run(&res);
    }

    #[test]
    fn declared_access_passes_enforcement() {
        struct Honest;
        impl System for Honest {
            fn name(&self) -> &'static str {
                "honest"
            }
            fn access(&self) -> SystemAccess {
                SystemAccess::new().reads::<A>().writes::<B>()
            }
            fn run(&self, resources: &Resources) {
                // B декларирован на запись — собственное чтение разрешено.
                assert!(resources.get::<A>().is_some());
                assert!(resources.get::<B>().is_some());
            }
        }
        let mut res = Resources::new();
        res.insert(A);
        res.insert(B);
        let mut sched = Schedule::new();
        sched.add_system(Honest).set_enforce_accesses(true);
        sched.run(&res);
    }

    #[test]
    fn resources_are_unrestricted_outside_schedule() {
        let mut res = Resources::new();
        res.insert(B);
        // Вне Schedule::run активных деклараций нет — доступ свободен.
        assert!(res.contains::<B>());
        assert!(res.get::<B>().is_some());
    }
}
