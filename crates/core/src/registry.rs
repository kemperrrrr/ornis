//! Реестр компонентов — фундамент F0 аудита 2026-08-22
//! (документ `docs/quality/audit-2026-08-22.md`, §10): имя ↔ [`TypeId`] ↔
//! type-erased операции над лентами [`SmartStore`].
//!
//! Обслуживает tooling-пути: generic `SetComponent` редактора (D2),
//! batch-API скриптинга (D1), сериализацию сцен (фаза 7), гранулярность
//! лент шедулера (`lane_id` — плотный индекс для будущих битсетов
//! доступов). Горячие кадровые циклы реестр **не трогают** — они
//! остаются типизированными (SoA-ленты, `#[smart_pipeline]`); граница
//! та же, что у Bevy между `bevy_reflect` и типизированными квери.
//!
//! Thunk'и мономорфизируются обычной generic-регистрацией — без
//! процедурного макроса; derive-сахар (`#[derive(RegisterComponent)]`)
//! — опциональный следующий шаг, API реестра от него не меняется.
//!
//! # Пример
//!
//! ```rust
//! use ornis_core::{ComponentRegistry, SmartStore};
//!
//! let mut registry = ComponentRegistry::new();
//! registry.register::<f32>("health");
//!
//! let mut world = SmartStore::new();
//! let hero = world.create_entity();
//!
//! let meta = registry.by_name("health").unwrap();
//! meta.set_json(&mut world, hero, &serde_json::json!(100.0))
//!     .unwrap();
//! assert_eq!(
//!     meta.get_json(&world, hero).unwrap(),
//!     Some(serde_json::json!(100.0))
//! );
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

use serde::{Serialize, de::DeserializeOwned};

use crate::entity::Entity;
use crate::smart_store::SmartStore;

/// Плотный индекс ленты в реестре (0..len). Зарезервирован под битсеты
/// доступов шедулера (аудит §3.6) — стабилен в пределах одного реестра.
pub type LaneId = u32;

/// Ошибка type-erased операции реестра.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// JSON не соответствует схеме компонента (`set_json`) либо
    /// компонент не сериализуется (`get_json`; для обычных struct-ов
    /// практически недостижимо).
    Json(String),
}

impl RegistryError {
    fn from_json(error: serde_json::Error) -> Self {
        Self::Json(error.to_string())
    }
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::Json(message) => write!(f, "component JSON error: {message}"),
        }
    }
}

impl std::error::Error for RegistryError {}

type RegisterLaneFn = fn(&mut SmartStore);
type InsertAnyFn = fn(&mut SmartStore, Entity, Box<dyn Any>) -> bool;
type ContainsFn = fn(&SmartStore, Entity) -> bool;
type LaneLenFn = fn(&SmartStore) -> usize;
type RemoveFn = fn(&mut SmartStore, Entity) -> Option<Box<dyn Any>>;
type GetJsonFn = fn(&SmartStore, Entity) -> Result<Option<serde_json::Value>, RegistryError>;
type SetJsonFn = fn(&mut SmartStore, Entity, &serde_json::Value) -> Result<(), RegistryError>;
type ParseJsonFn = fn(&serde_json::Value) -> Result<Box<dyn Any>, RegistryError>;

fn register_lane_thunk<T>(store: &mut SmartStore)
where
    T: 'static + Send + Sync,
{
    store.register::<T>();
}

fn insert_any_thunk<T>(store: &mut SmartStore, entity: Entity, boxed: Box<dyn Any>) -> bool
where
    T: 'static + Clone + Send + Sync,
{
    let Ok(component) = boxed.downcast::<T>() else {
        return false;
    };
    store.insert(entity, *component);
    true
}

fn contains_thunk<T>(store: &SmartStore, entity: Entity) -> bool
where
    T: 'static + Send + Sync,
{
    store
        .read_lane::<T>()
        .is_some_and(|lane| lane.contains(entity))
}

fn lane_len_thunk<T>(store: &SmartStore) -> usize
where
    T: 'static + Send + Sync,
{
    store.read_lane::<T>().map_or(0, |lane| lane.len())
}

fn remove_thunk<T>(store: &mut SmartStore, entity: Entity) -> Option<Box<dyn Any>>
where
    T: 'static + Send + Sync,
{
    store
        .write_lane::<T>()
        .and_then(|mut lane| lane.remove(entity))
        .map(|component| Box::new(component) as Box<dyn Any>)
}

fn get_json_thunk<T>(store: &SmartStore, entity: Entity) -> GetJsonResult
where
    T: 'static + Send + Sync + Serialize,
{
    let Some(lane) = store.read_lane::<T>() else {
        return Ok(None);
    };
    let Some(component) = lane.get(entity) else {
        return Ok(None);
    };
    serde_json::to_value(component)
        .map(Some)
        .map_err(RegistryError::from_json)
}

fn set_json_thunk<T>(store: &mut SmartStore, entity: Entity, value: &serde_json::Value) -> SetResult
where
    T: 'static + Clone + Send + Sync + DeserializeOwned,
{
    let component: T = serde_json::from_value(value.clone()).map_err(RegistryError::from_json)?;
    store.insert(entity, component);
    Ok(())
}

fn parse_json_thunk<T>(value: &serde_json::Value) -> Result<Box<dyn Any>, RegistryError>
where
    T: 'static + DeserializeOwned,
{
    let component: T = serde_json::from_value(value.clone()).map_err(RegistryError::from_json)?;
    Ok(Box::new(component))
}

type GetJsonResult = Result<Option<serde_json::Value>, RegistryError>;
type SetResult = Result<(), RegistryError>;

/// Type-erased запись о компоненте: имя ↔ тип ↔ операции над его лентой.
///
/// Все операции делегируются мономорфным thunk'ам, созданным при
/// [`ComponentRegistry::register`]; структура `Send + Sync` (fn-pointer'ы
/// и `&'static str`), реестр можно разделять между потоками (`Arc`).
pub struct ComponentMeta {
    name: &'static str,
    type_name: &'static str,
    type_id: TypeId,
    lane_id: LaneId,
    register_lane: RegisterLaneFn,
    insert_any: InsertAnyFn,
    contains: ContainsFn,
    lane_len: LaneLenFn,
    remove: RemoveFn,
    get_json: GetJsonFn,
    set_json: SetJsonFn,
    parse_json: ParseJsonFn,
}

impl ComponentMeta {
    /// Короткое имя из регистрации (ключ протоколов: JSON/FFI/сцены).
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Полный Rust-путь типа (диагностика, не ключ протокола).
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// [`TypeId`] компонента — ключ ленты в [`SmartStore`].
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// Плотный индекс ленты в реестре (см. [`LaneId`]).
    pub fn lane_id(&self) -> LaneId {
        self.lane_id
    }

    /// Создаёт пустую ленту в мире, если её ещё нет.
    pub fn register_lane(&self, store: &mut SmartStore) {
        (self.register_lane)(store)
    }

    /// Вставляет boxed-компонент. `false`, если тип бокса не `T`
    /// (нарушение вызывающего — реестр сам такой вызов не создаёт).
    pub fn insert_any(&self, store: &mut SmartStore, entity: Entity, boxed: Box<dyn Any>) -> bool {
        (self.insert_any)(store, entity, boxed)
    }

    /// Есть ли у сущности компонент (с учётом generation хендла).
    pub fn contains(&self, store: &SmartStore, entity: Entity) -> bool {
        (self.contains)(store, entity)
    }

    /// Число живых компонентов в ленте (0, если ленты ещё нет).
    pub fn lane_len(&self, store: &SmartStore) -> usize {
        (self.lane_len)(store)
    }

    /// Удаляет и возвращает компонент как `Box<dyn Any>` (None — его нет).
    pub fn remove(&self, store: &mut SmartStore, entity: Entity) -> Option<Box<dyn Any>> {
        (self.remove)(store, entity)
    }

    /// Снапшот компонента в JSON (None — у сущности его нет).
    pub fn get_json(
        &self,
        store: &SmartStore,
        entity: Entity,
    ) -> Result<Option<serde_json::Value>, RegistryError> {
        (self.get_json)(store, entity)
    }

    /// Upsert компонента из JSON: десериализует и вставляет (семантика
    /// `SmartStore::insert` — существующий компонент перезаписывается).
    pub fn set_json(
        &self,
        store: &mut SmartStore,
        entity: Entity,
        value: &serde_json::Value,
    ) -> Result<(), RegistryError> {
        (self.set_json)(store, entity, value)
    }

    /// Десериализует компонент из JSON в `Box<dyn Any>` — без доступа к
    /// миру. Пара с [`ComponentMeta::insert_any`] даёт семантику
    /// «сначала разобрать, потом мутировать»: вызывающий валидирует все
    /// payload'ы команды до единой записи в мир (инвариант «ошибка
    /// команды не трогает мир» редакторского протокола).
    pub fn parse_json(&self, value: &serde_json::Value) -> Result<Box<dyn Any>, RegistryError> {
        (self.parse_json)(value)
    }
}

/// Реестр компонентов: строится один раз на старте (`register::<T>(name)`
/// для каждого типа), дальше — read-only и разделяемый (`Arc`).
///
/// Порядок регистрации определяет [`LaneId`] — для воспроизводимых
/// протоколов регистрируйте в фиксированном порядке.
#[derive(Default)]
pub struct ComponentRegistry {
    by_id: HashMap<TypeId, LaneId>,
    by_name: HashMap<&'static str, LaneId>,
    entries: Vec<ComponentMeta>,
}

impl ComponentRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Регистрирует тип компонента под протокольным именем `name`.
    ///
    /// Базовые операции (лента, contains, remove) не требуют serde;
    /// `get_json`/`set_json` мономорфизируются под `Serialize`/
    /// `DeserializeOwned` того же типа — граница «reflection только для
    /// tooling» соблюдается требованием bound'ов у вызывающего.
    ///
    /// # Panics
    /// Паникует при повторной регистрации того же типа или занятом
    /// имени — это ошибка конфигурации, а не рантайм-ситуация.
    pub fn register<T>(&mut self, name: &'static str) -> &mut Self
    where
        T: 'static + Clone + Send + Sync + Serialize + DeserializeOwned,
    {
        let type_id = TypeId::of::<T>();
        assert!(
            !self.by_id.contains_key(&type_id),
            "component type `{}` is already registered",
            std::any::type_name::<T>()
        );
        assert!(
            !self.by_name.contains_key(name),
            "component name `{name}` is already registered"
        );

        let lane_id = self.entries.len() as LaneId;
        self.entries.push(ComponentMeta {
            name,
            type_name: std::any::type_name::<T>(),
            type_id,
            lane_id,
            register_lane: register_lane_thunk::<T>,
            insert_any: insert_any_thunk::<T>,
            contains: contains_thunk::<T>,
            lane_len: lane_len_thunk::<T>,
            remove: remove_thunk::<T>,
            get_json: get_json_thunk::<T>,
            set_json: set_json_thunk::<T>,
            parse_json: parse_json_thunk::<T>,
        });
        self.by_id.insert(type_id, lane_id);
        self.by_name.insert(name, lane_id);
        self
    }

    /// Запись по типу.
    pub fn by_id(&self, type_id: TypeId) -> Option<&ComponentMeta> {
        self.by_id
            .get(&type_id)
            .map(|&id| &self.entries[id as usize])
    }

    /// Запись по протокольному имени.
    pub fn by_name(&self, name: &str) -> Option<&ComponentMeta> {
        self.by_name.get(name).map(|&id| &self.entries[id as usize])
    }

    /// Запись по плотному индексу ленты.
    pub fn by_lane_id(&self, lane_id: LaneId) -> Option<&ComponentMeta> {
        self.entries.get(lane_id as usize)
    }

    /// Все записи в порядке регистрации.
    pub fn iter(&self) -> std::slice::Iter<'_, ComponentMeta> {
        self.entries.iter()
    }

    /// Число зарегистрированных типов.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Пуст ли реестр.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Health {
        hp: u32,
    }

    fn registry_with_two() -> ComponentRegistry {
        let mut registry = ComponentRegistry::new();
        registry.register::<Position>("position");
        registry.register::<Health>("health");
        registry
    }

    #[test]
    fn lookup_by_name_and_id_and_lane_id() {
        let registry = registry_with_two();

        let pos = registry.by_name("position").expect("position");
        assert_eq!(pos.type_id(), TypeId::of::<Position>());
        assert_eq!(pos.type_name(), std::any::type_name::<Position>());
        assert_eq!(pos.lane_id(), 0);

        let health = registry.by_id(TypeId::of::<Health>()).expect("health");
        assert_eq!(health.name(), "health");
        assert_eq!(health.lane_id(), 1);
        assert!(registry.by_lane_id(1).is_some());

        // LaneId — плотная проекция: by_lane_id и поиск совпадают.
        assert!(std::ptr::eq(registry.by_lane_id(0).unwrap(), pos));
        assert!(registry.by_name("ghost").is_none());
        assert!(registry.by_id(TypeId::of::<u8>()).is_none());
        assert!(registry.by_lane_id(2).is_none());

        assert_eq!(registry.len(), 2);
        assert!(!registry.is_empty());
        let names: Vec<_> = registry.iter().map(|meta| meta.name()).collect();
        assert_eq!(names, vec!["position", "health"]);
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn duplicate_type_panics() {
        let mut registry = ComponentRegistry::new();
        registry.register::<Position>("position");
        registry.register::<Position>("pos2");
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn duplicate_name_panics() {
        let mut registry = ComponentRegistry::new();
        registry.register::<Position>("component");
        registry.register::<Health>("component");
    }

    #[test]
    fn register_lane_creates_empty_lane_eagerly() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let meta = registry.by_name("position").unwrap();

        meta.register_lane(&mut store);
        assert_eq!(meta.lane_len(&store), 0);
        // Лента реально создана: типизированный доступ уже возможен.
        assert!(store.read_lane::<Position>().is_some());
    }

    #[test]
    fn insert_any_then_contains_and_len() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("position").unwrap();

        assert!(!meta.contains(&store, entity));
        assert_eq!(meta.lane_len(&store), 0);

        let inserted = meta.insert_any(&mut store, entity, Box::new(Position { x: 1.0, y: 2.0 }));
        assert!(inserted);
        assert!(meta.contains(&store, entity));
        assert_eq!(meta.lane_len(&store), 1);

        // Типизированный путь видит то же значение.
        let lane = store.read_lane::<Position>().unwrap();
        assert_eq!(lane.get(entity), Some(&Position { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn insert_any_with_wrong_box_type_returns_false() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("position").unwrap();

        // Бокс другого типа: insert не должен случиться.
        let inserted = meta.insert_any(&mut store, entity, Box::new(Health { hp: 5 }));
        assert!(!inserted);
        assert!(!meta.contains(&store, entity));
        assert_eq!(meta.lane_len(&store), 0);
        // А лента Health не была тронута чужим insert.
        let health = registry.by_name("health").unwrap();
        assert_eq!(health.lane_len(&store), 0);
    }

    #[test]
    fn set_json_upserts_and_get_json_roundtrips() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("position").unwrap();

        meta.set_json(&mut store, entity, &json!({"x": 1.5, "y": -2.0}))
            .unwrap();
        assert_eq!(
            meta.get_json(&store, entity).unwrap(),
            Some(json!({"x": 1.5, "y": -2.0}))
        );
        assert_eq!(meta.lane_len(&store), 1);

        // Повторный set_json — перезапись без роста ленты.
        meta.set_json(&mut store, entity, &json!({"x": 0.0, "y": 7.25}))
            .unwrap();
        assert_eq!(
            meta.get_json(&store, entity).unwrap(),
            Some(json!({"x": 0.0, "y": 7.25}))
        );
        assert_eq!(meta.lane_len(&store), 1);
    }

    #[test]
    fn set_json_schema_mismatch_is_json_error() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("health").unwrap();

        // Нет поля `hp`.
        let missing = meta.set_json(&mut store, entity, &json!({"mana": 5}));
        assert!(matches!(missing, Err(RegistryError::Json(_))));
        // Тип поля не сходится.
        let wrong_type = meta.set_json(&mut store, entity, &json!({"hp": "full"}));
        assert!(matches!(wrong_type, Err(RegistryError::Json(_))));
        // i32 в u32 не лезет.
        let negative = meta.set_json(&mut store, entity, &json!({"hp": -1}));
        assert!(matches!(negative, Err(RegistryError::Json(_))));

        assert!(!meta.contains(&store, entity));
    }

    #[test]
    fn parse_json_validates_before_insert_any() {
        let registry = registry_with_two();
        let position = registry.by_name("position").unwrap();
        let mut store = SmartStore::new();
        let entity = store.create_entity();

        // Разобранный бокс вставляется и читается обратно.
        let boxed = position.parse_json(&json!({"x": 1.0, "y": 2.0})).unwrap();
        assert!(position.insert_any(&mut store, entity, boxed));
        let lane = store.read_lane::<Position>().unwrap();
        assert_eq!(lane.get(entity), Some(&Position { x: 1.0, y: 2.0 }));

        // Схема не сошлась — ошибка до всякой мутации мира.
        let bad = position.parse_json(&json!({"x": "left", "y": 0.0}));
        assert!(matches!(bad, Err(RegistryError::Json(_))));
        assert_eq!(position.lane_len(&store), 1);
    }

    #[test]
    fn get_json_on_absent_component_is_ok_none() {
        let registry = registry_with_two();
        let store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("position").unwrap();

        assert_eq!(meta.get_json(&store, entity).unwrap(), None);
    }

    #[test]
    fn remove_returns_boxed_component_and_clears() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("position").unwrap();

        meta.insert_any(&mut store, entity, Box::new(Position { x: 3.0, y: 4.0 }));
        let boxed = meta.remove(&mut store, entity).expect("component");
        let position = boxed.downcast::<Position>().expect("position type");
        assert_eq!(*position, Position { x: 3.0, y: 4.0 });

        assert!(!meta.contains(&store, entity));
        assert!(meta.remove(&mut store, entity).is_none());
    }

    #[test]
    fn destroyed_entity_has_no_components() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let meta = registry.by_name("position").unwrap();
        meta.insert_any(&mut store, entity, Box::new(Position { x: 1.0, y: 1.0 }));

        store.destroy_entity(entity);
        assert!(!meta.contains(&store, entity));

        // Свежая сущность с переиспользованным id — пустая.
        let recycled = store.create_entity();
        assert_ne!(recycled.generation(), entity.generation());
        assert!(!meta.contains(&store, recycled));
    }

    #[test]
    fn components_of_different_types_are_isolated() {
        let registry = registry_with_two();
        let mut store = SmartStore::new();
        let entity = store.create_entity();
        let pos = registry.by_name("position").unwrap();
        let health = registry.by_name("health").unwrap();

        pos.set_json(&mut store, entity, &json!({"x": 1.0, "y": 2.0}))
            .unwrap();
        health
            .set_json(&mut store, entity, &json!({"hp": 100}))
            .unwrap();

        assert_eq!(pos.lane_len(&store), 1);
        assert_eq!(health.lane_len(&store), 1);

        // Удаление одного типа не трогает другой.
        health.remove(&mut store, entity);
        assert!(!health.contains(&store, entity));
        assert!(pos.contains(&store, entity));
    }
}
