//! Unified system scheduler — S5a foundation (PLAN Приложение C, IDEAS §28.2).
//!
//! Один планировщик на всё: системы объявляют доступы к одиночным
//! ресурсам («миру» из [`Resources`]), планировщик выводит конфликты
//! (read-after-write, write-after-read, write-after-write), раскладывает
//! системы по уровням параллельности и исполняет уровни последовательно,
//! а системы внутри уровня — параллельно (rayon).
//!
//! Детерминизм (Strong Confluence): уровни выведены из порядка
//! регистрации; внутри уровня системы не конфликтуют (записи дизъюнктны
//! с чужими чтениями/записями), значит порядок исполнения внутри уровня
//! не влияет на результат — при условии, что ресурсы-одиночки
//! используют внутреннюю изменяемость (`Mutex`/атомики), а системы не
//! трогают чужие ресурсы помимо объявленных доступов.
//!
//! Порядок регистрации — тайбрейк, как у пассов рендер-графа: любые два
//! конфликтующих процесса идут в порядке добавления.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use rayon::prelude::*;

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
    pub fn contains<R: Any + Send + Sync>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<R>())
    }

    /// Общий доступ к singleton типа `R`.
    pub fn get<R: Any + Send + Sync>(&self) -> Option<&R> {
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Access {
    pub reads: Vec<TypeId>,
    pub writes: Vec<TypeId>,
}

impl Access {
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
    fn writes_touch(&self, other: &Access) -> bool {
        self.writes
            .iter()
            .any(|w| other.reads.contains(w) || other.writes.contains(w))
    }
}

/// Система единого шедулера: доступы — данные, исполнение — над миром.
pub trait System: Send + Sync {
    fn name(&self) -> &'static str;
    fn access(&self) -> Access;
    fn run(&self, resources: &Resources);
}

/// Конфликт двух систем (в порядке регистрации i < j): упорядочены, если
/// i пишет то, что j читает/пишет (RaW/WaW), или j пишет то, что читает
/// i (анти-зависимость, WaR).
fn conflicts(a: &Access, b: &Access) -> bool {
    a.writes_touch(b) || b.writes_touch(a)
}

/// Уровни параллельности: системы без конфликтов внутри уровня;
/// уровни упорядочены зависимостями. Детерминировано порядком регистрации.
fn level_groups(accesses: &[Access], ordering: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let n = accesses.len();
    let mut level = vec![0usize; n];
    for j in 1..n {
        let mut best = 0usize;
        for i in 0..j {
            let ordered =
                conflicts(&accesses[i], &accesses[j]) || ordering.contains(&(i, j));
            if ordered {
                best = best.max(level[i] + 1);
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

/// Расписание систем: порядок регистрации — тайбрейк конфликтов.
pub struct Schedule {
    systems: Vec<Box<dyn System>>,
    accesses: Vec<Access>,
    /// S5c: явные рёбра порядка (индексы регистрации i < j) поверх
    /// выведенных из доступов — например, для скрытых зависимостей
    /// (общие queue-буферы), которых не видно в множествах доступа.
    ordering: Vec<(usize, usize)>,
    parallel: bool,
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
        }
    }

    /// Добавляет систему (порядок = приоритет при конфликтах).
    pub fn add_system<S: System + 'static>(&mut self, system: S) -> &mut Self {
        self.accesses.push(system.access());
        self.systems.push(Box::new(system));
        self
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
        let b = self.system_index(before);
        let a = self.system_index(after);
        assert!(
            b < a,
            "order_before('{before}', '{after}'): '{after}' is registered              earlier — register systems in execution order and use              order_before to split parallel levels"
        );
        if !self.ordering.contains(&(b, a)) {
            self.ordering.push((b, a));
        }
        self
    }

    /// S5c: зеркальный [`Schedule::order_before`].
    pub fn order_after(&mut self, after: &str, before: &str) -> &mut Self {
        self.order_before(before, after)
    }

    fn system_index(&self, name: &str) -> usize {
        self.systems
            .iter()
            .position(|sys| sys.name() == name)
            .unwrap_or_else(|| panic!("schedule: no system named '{name}'"))
    }

    /// Параллельное (true, по умолчанию) или строго последовательное
    /// (bit-identical порядок регистрации) исполнение.
    pub fn set_parallel(&mut self, parallel: bool) -> &mut Self {
        self.parallel = parallel;
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
        level_groups(&self.accesses, &self.ordering)
    }

    /// Исполняет расписание над миром.
    pub fn run(&self, resources: &Resources) {
        if !self.parallel {
            for system in &self.systems {
                system.run(resources);
            }
            return;
        }
        for group in self.level_groups() {
            group.par_iter().for_each(|&i| {
                self.systems[i].run(resources);
            });
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

    fn reads_writes<R: Any + Send + Sync>(reads: bool, writes: bool) -> Access {
        let mut a = Access::new();
        if reads {
            a = a.reads::<R>();
        }
        if writes {
            a = a.writes::<R>();
        }
        a
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
        assert_eq!(level_groups(&accesses, &[]), vec![vec![0, 1, 2]]);

        let chain = vec![
            Access::new().writes::<A>(),
            Access::new().reads::<A>().writes::<B>(),
            Access::new().reads::<B>().writes::<C>(),
        ];
        assert_eq!(level_groups(&chain), vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn anti_dependency_orders_reader_first() {
        // i читает X, j пишет X → i раньше j (анти-зависимость).
        let accesses = vec![Access::new().reads::<A>(), Access::new().writes::<A>()];
        assert_eq!(level_groups(&accesses, &[]), vec![vec![0], vec![1]]);
    }

    struct Bump {
        name: &'static str,
        access: Access,
        target: &'static str,
    }

    impl System for Bump {
        fn name(&self) -> &'static str {
            self.name
        }
        fn access(&self) -> Access {
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

    #[test]
    fn sequential_mode_is_registration_order() {
        let mut res = Resources::new();
        res.insert(Mutex::new(Vec::<&'static str>::new()));
        let mut sched = Schedule::new();
        sched
            .add_system(Bump {
                name: "w",
                access: Access::new().writes::<A>(),
                target: "w",
            })
            .add_system(Bump {
                name: "r",
                access: Access::new().reads::<A>().writes::<B>(),
                target: "r",
            });
        sched.set_parallel(false);
        sched.run(&res);
        let log = res.get::<Mutex<Vec<&'static str>>>().unwrap();
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
            access: Access,
            name: &'static str,
        }
        impl System for Add {
            fn name(&self) -> &'static str {
                self.name
            }
            fn access(&self) -> Access {
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
                    access: Access::new().writes::<A>(),
                    name: "src",
                })
                .add_system(Add {
                    access: Access::new().reads::<A>().writes::<B>(),
                    name: "left",
                })
                .add_system(Add {
                    access: Access::new().reads::<A>().writes::<C>(),
                    name: "right",
                })
                .add_system(Add {
                    access: Access::new().reads::<B>().reads::<C>().writes::<A>(),
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
        struct Noop(&'static str, Access);
        impl System for Noop {
            fn name(&self) -> &'static str {
                self.0
            }
            fn access(&self) -> Access {
                self.1.clone()
            }
            fn run(&self, _: &Resources) {}
        }
        sched
            .add_system(Noop("first", Access::new().writes::<A>()))
            .add_system(Noop("second", Access::new().writes::<B>()));
        assert_eq!(sched.level_groups(), vec![vec![0, 1]]);
        sched.order_before("first", "second");
        assert_eq!(sched.level_groups(), vec![vec![0], vec![1]]);
    }

    #[test]
    #[should_panic(expected = "registered")]
    fn explicit_ordering_rejects_backward_direction() {
        let mut sched = Schedule::new();
        struct Noop(&'static str);
        impl System for Noop {
            fn name(&self) -> &'static str {
                self.0
            }
            fn access(&self) -> Access {
                Access::new()
            }
            fn run(&self, _: &Resources) {}
        }
        sched.add_system(Noop("a")).add_system(Noop("b"));
        sched.order_before("b", "a");
    }

    #[test]
    #[should_panic(expected = "no system named")]
    fn explicit_ordering_unknown_name_panics() {
        let mut sched = Schedule::new();
        struct Noop;
        impl System for Noop {
            fn name(&self) -> &'static str {
                "only"
            }
            fn access(&self) -> Access {
                Access::new()
            }
            fn run(&self, _: &Resources) {}
        }
        sched.add_system(Noop);
        sched.order_before("only", "ghost");
    }

    #[test]
    fn schedule_levels_diamond() {
        let mut sched = Schedule::new();
        struct Noop(Access);
        impl System for Noop {
            fn name(&self) -> &'static str {
                "noop"
            }
            fn access(&self) -> Access {
                self.0.clone()
            }
            fn run(&self, _: &Resources) {}
        }
        sched
            .add_system(Noop(Access::new().writes::<A>()))
            .add_system(Noop(Access::new().reads::<A>().writes::<B>()))
            .add_system(Noop(Access::new().reads::<A>().writes::<C>()))
            .add_system(Noop(Access::new().reads::<B>().reads::<C>().writes::<A>()));
        assert_eq!(sched.level_groups(), vec![vec![0], vec![1, 2], vec![3]]);
    }
}
