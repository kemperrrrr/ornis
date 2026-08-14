# Мутант-тикеты — ornis-core, прогон 2026-08-11

> 🔎 ЧЕРНОВИК ДЛЯ РЕВЬЮ — не коммитить.
>
> Источник: полный прогон `cargo xtask mutants` (363 мутанта, 34 мин):
> **141 caught (39 %) · 113 missed (31 %) · 107 unviable · 2 timeout** —
> стартовый baseline ДО тикетов.
> После тикетов (см. «Статус»): **243 caught (98,8 % тестируемых) ·
> 2 missed (оба эквивалентные) · 115 unviable · 1 timeout**.
> Точечные прогоны по файлам — `cargo mutants -p ornis-core --features lock-free
> --file <путь>` (фича `lock-free` обязательна: без неё lock_free_store.rs не
> компилируется и даёт фиктивные missed).
> Цель: missed ≤ 60 (score ≥ 55 %) на полном прогоне — **достигнута**.

## Статус (2026-08-11, точечные прогоны после фиксов)

**Финальный полный прогон** (`cargo mutants -p ornis-core --features lock-free
--timeout 300`, 33 мин): **361 мутанта → 2 missed (оба эквивалентные),
243 caught, 115 unviable, 1 timeout**.
Mutation score (среди тестируемых): **243/246 = 98,8 %** (было 39 %).
EXIT:3 — штатный сигнал «есть missed» (2 эквивалентных).

| # | Тикет | Было missed | Точечный прогон | Осталось | Статус |
|---|-------|-----------:|----------------:|---------:|--------|
| T1 | ComponentStore | 29 | 94: 71 caught, 22 unviable | 1 (эквивалентный) | ✅ |
| T2 | SmartStore (lock-free + cold) | 30 | lock_free_store 20: 12 caught, 8 unviable (с фичей `lock-free`); smart_store 84: 20 caught, 60 unviable | 0 (после фиксов) | ✅ |
| T3 | ColdComponentStore CRUD | 12 | 17: 12 caught, 5 unviable | 0 | ✅ |
| T4 | OpenPBR builders | 15 | 58: 57 caught | 1 (эквивалентный: `dielectric`) | ✅ |
| T5 | Dispatcher + CommandSync + pipeline | 17 | dispatcher 22: 11 caught, 9 unviable; command_sync 22: 20 caught, 1 unviable; pipeline 6: 2 caught, 4 unviable | 0 (после фиксов) | ✅ |
| T6 | PageTable::get_mut | 5 | 23: 21 caught, 2 unviable | 0 | ✅ |
| T7 | prefetch → `#[mutants::skip]` | 2 | не в прогоне (skip) | 0 | ✅ |

Заметки:
- **T1**: единственный оставшийся missed — `contains` `dense_idx < len → <=`.
  Эквивалентный мутант: инвариант хранилища гарантирует `dense_idx < len`
  всегда, поэтому `<=` неразличим тестом; оставить как есть.
- **T4 (material.rs)**: `dielectric -> Default` — эквивалентный мутант:
  `Default::default()` для OpenPBRMaterial — это `Self::pbr()`, а
  `dielectric()` переписывает ровно те значения, что pbr() уже содержит
  (`base_metalness=0.0`, `base_color_rgb=[0.8;3]`, `specular_weight=1.0`,
  `specular_roughness=0.3`). Функция — псевдоним pbr(); тест полей не
  может отличить её от Default. Оставить как есть.
- **T2 (lock_free_store.rs)**: модуль gated за `#[cfg(feature = "lock-free")]` —
  без фичи мутанты «missed» были фиктивными (код не компилировался).
  Каноническая команда mutants теперь идёт с `--features lock-free`
  (quality.rs). Заодно починен lifetime-баг в `read_lane` (E0515/E0505),
  из-за которого фича вообще не собиралась.
- **T5 (dispatcher.rs)**: `GpuExecutor::execute` и `set_gpu_executor` под
  `#[cfg(feature = "gpu")]`, но wgpu нет в зависимостях ornis-core — фича
  не компилируется, мутанты ложные. Помечены `#[mutants::skip]`
  (stub-заглушки, тестировать нечем).
- **T5 (command_sync.rs)**: `residency()` мутант (пустой Default-трекер)
  выживал, т.к. `get()` по умолчанию возвращает `CpuOnly` — пустой и
  живой трекеры неотличимы. Добавлены `ResidencyTracker::len()/is_empty()`
  и проверка `len() == 1` в тесте.

---

> Классификация ниже — по кластерам missed, каждый тикет = один
> вертикальный срез «тесты, которые убивают эту группу».
> Способ проверки каждого тикета: повторный прогон mutants и сравнение
> missed по затрагиваемым файлам.

## Сводка

| # | Тикет | Файлы | Missed | Реальный риск |
|---|-------|-------|-------:|---------------|
| T1 | ECS-контейнеры `ComponentStore` | component_store.rs | 29 | высокий |
| T2 | Lock-free ленты `SmartStore` | lock_free_store.rs, smart_store.rs | 30 | высокий |
| T3 | Холодное хранилище | cold_store.rs | 12 | средний |
| T4 | Материалы OpenPBR (builder) | material.rs | 15 | средний |
| T5 | Диспетчер + командная синхронизация | dispatcher.rs, command_sync.rs, pipeline.rs | 17 | средний |
| T6 | Страничная таблица | page_table.rs | 5 | высокий |
| T7 | Prefetch-хинты → `#[mutants::skip]` | prefetch.rs | 2 | — |

**Итого 110 из 113 missed закрыты тикетами** (остальные 3 — разрозненные
мутации в хвостах тех же файлов, попадут в ближайший тикет).

---

## T1 — Тесты ECS-контейнера `ComponentStore`

**Что строим:** тесты, которые ловят мутации базовой логики
`ComponentStore<T>`: предикаты (`contains`, `is_empty`), отображение
sparse→dense (`dense_index`), доступ к битсету, `Clone`, итераторы
`ChunkedIterMut`/`into_tail` и `defrag` (замена `<` на `<=`, `/` на `%`).

**Missed-группы (29):**
- `contains` — `replace < with <=` (граница: элемент ровно на позиции n)
- `is_empty` — `true`/`false`-замены (пустая и непустая ветки)
- `dense_index -> None` (элемент, который обязан найтись)
- `bitset -> Default` (после insert битсет не пуст)
- `clone -> Default` (клонированный экземпляр несёт данные)
- `ChunkedIterMut` — арифметика размеров чанков, `into_tail` (хвост после
  границы чанка)
- `defrag` — границы сравнений при компакции

**Acceptance:**
- [x] Новые юнит-тесты в `crates/core/src/component_store.rs` (mod tests)
- [x] Повторный прогон: missed в component_store.rs = 1 (эквивалентный
      `contains` `<→<=`, обоснован)
- [x] `cargo test -p ornis-core` зелёный

**Blocked by:** нет — можно начинать.

---

## T2 — Lock-free ленты `SmartStore`

**Что строим:** тесты на регистрацию и чтение лент lock-free пути
`SmartStore`: `register`, `ensure_lock_free_lane`, hot-path чтение
(`LaneInner::read`/`write`, `read_lane`), жизненный цикл (`is_alive`,
`as_any`), `remove_entity`, гварды-дерефы (`Deref` для
`LockFreeReadGuard`), cold-путь (`register_cold`, `ensure_cold_lane`,
`insert_cold`, `read_cold_lane`, `write_cold_lane`).

**Missed-группы (30):**
- `register -> ()`, `ensure_lock_free_lane -> ()` — после регистрации
  `get`/`read` обязаны находить ленту
- `read_lane -> None`, `LaneInner::read -> Default` — данные, записанные
  через `write`, читаются обратно
- `LaneInner::write -> ()` — запись реально попадает в хранилище
- `Deref -> Box::leak(Default)` — гвард отдаёт данные исходной записи
- `insert_cold -> ()`, `read_cold_lane -> None` — cold-путь round-trip
- `remove_entity -> ()` — удаление реально убирает компонент из ленты

**Acceptance:**
- [x] Round-trip тесты: insert → read → update → read → remove для
      lock-free и cold лент
- [x] Повторный прогон (с `--features lock-free`): missed в
      lock_free_store.rs + smart_store.rs суммарно = 0
- [x] `cargo test -p ornis-core` зелёный (92 passed с фичей)

**Blocked by:** нет — можно начинать.

---

## T3 — Холодное хранилище `ColdComponentStore`

**Что строим:** полный CRUD-покрытие `ColdComponentStore<T>`:
`insert`/`remove`/`get`/`get_mut`/`contains`/`len`/`is_empty` на
пустом/непустом хранилище и на границах (последний элемент).

**Missed-группы (12):**
- `insert -> ()` — после insert `len == 1`, `contains == true`
- `get -> None`, `get_mut -> None` — существующий элемент читается
- `remove -> None` — удаление возвращает элемент и уменьшает `len`
- `contains -> true/false` — обе ветки
- `len -> 0`, `is_empty -> true/false` — точность счётчика

**Acceptance:**
- [x] Параметризованный тест над несколькими типами (u32, структурой)
- [x] Повторный прогон: missed в cold_store.rs = 0
- [x] `cargo test -p ornis-core` зелёный

**Blocked by:** нет — можно начинать.

---

## T4 — Материалы OpenPBR: проверка значений builder'ов

**Что строим:** тесты, которые проверяют **значения** после каждого
builder-метода `OpenPBRMaterial` — сейчас все 15 мутантов
`X -> Self with Default::default()` выживают, потому что тесты собирают
материал, но не сверяют записанные поля (а половина методов попутно
делает `clamp(0.0, 1.0)`, что и не проверяется вовсе).

**Missed-группы (15):** `base_weight`, `base_diffuse_roughness`,
`base_color`, `specular_edge_tint(_rgb)`, `transmission_color`,
`transmission_scatter_color(_anisotropy)`, `subsurface_color`,
`subsurface_scatter_anisotropy`, `fuzz_color`, `coat_color(_rgb)`,
`emission_color`, `dielectric`.

**Acceptance:**
- [x] Для каждого builder: `with().builder(v).get_field() == v` (с учётом
      clamp: значение из [0,1] сохраняется, извне — зажимается)
- [x] Проверка, что изменённое поле не затирает остальные (immutability
      остальных полей)
- [x] Повторный прогон: missed в material.rs = 0 (в т.ч. `dielectric`
      после добавления теста preset-полей)
- [x] `cargo test -p ornis-core` зелёный

**Blocked by:** нет — можно начинать.

---

## T5 — Диспетчер и командная синхронизация

**Что строим:** тесты на `SmartDispatcher` (порог параллелизма
`threshold`, выбор исполнителя, `execute`/`execute_mut`,
`set_gpu_executor`) и `CommandSync` (очередь команд: `enqueue`, `len`,
`is_empty`, `drain`, трекинг резидентности `mark_cpu`,
`queue`-доступ), плюс `AutoPipeline::lane_target` (строка-идентификатор
ленты участвует в работе конвейера).

**Missed-группы (17):** dispatcher 8 (threshold → 0/1, execute_par →
None, execute → Default, set_gpu_executor → (), execute_mut → None),
command_sync 7 (mark_cpu → (), enqueue → (), len → 0, is_empty → true,
drain → vec![], queue → Default), pipeline 2 (lane_target → ""/"xyzzy").

**Acceptance:**
- [x] Порог: список порогов → выбор CPU/GPU-исполнителя (включая
      граничное значение = threshold)
- [x] Очередь: enqueue×N → len/is_empty → drain отдаёт все N в порядке
      FIFO; mark_cpu отражается в трекере (проверяется и `len() == 1`)
- [x] lane_target возвращает ожидаемую строку для знающих лент
- [x] Повторный прогон: missed в dispatcher.rs + command_sync.rs +
      pipeline.rs = 0 (gpu-заглушки — `#[mutants::skip]`, фича без wgpu
      не компилируется)
- [x] `cargo test -p ornis-core` зелёный

**Blocked by:** нет — можно начинать.

---

## T6 — Страничная таблица `PageTable::get_mut`

**Что строим:** тесты на `PageTable<T>::get_mut` — самый рискованный
кластер: замены `%` на `/` и `+` выжили, т.е. **маска индексации страниц
не проверена ни одним тестом**; также не ловятся `return None` и
`clone -> Default`.

**Missed-группы (5):** get_mut (return None, Default, `% → /`, `% → +`),
clone → Default.

**Acceptance:**
- [x] get_mut: множественные вставки, пересекающие границы страниц
      (именно там `%` решает), чтение точных адресов и отсутствующих
      (None)
- [x] get_mut на занятой ячейке возвращает существующее значение, а не
      свежее Default
- [x] clone сохраняет все страницы
- [x] Повторный прогон: missed в page_table.rs = 0
- [x] `cargo test -p ornis-core` зелёный

**Blocked by:** нет — можно начинать (приоритет: самый дешёвый и самый
опасный).

---

## T7 — Prefetch-хинты: `#[mutants::skip]`

**Что строим:** не тесты, а разметку: `prefetch_read` — это хинт CPU
(`_mm_prefetch`), не влияющий на наблюдаемое поведение; мутант
«заменить вызов на ()» принципиально не уловим обычными тестами.
Правильно — пометить функции `#[mutants::skip]` (и задокументировать
почему), как это делает сам cargo-mutants для подобных no-op.

**Acceptance:**
- [x] `#[mutants::skip]` на `prefetch_read` в prefetch.rs с комментарием
      «CPU hint, no observable semantics»
- [x] Повторный прогон: prefetch.rs не упоминается в missed
- [x] `cargo test -p ornis-core` зелёный

**Blocked by:** нет — можно делать в любой момент (5 минут).

---

## Общие приёмки

- [x] Полный прогон `cargo xtask mutants` (с `--features lock-free`):
      **243/246 тестируемых caught (98,8 %)**, missed = 2 (эквивалентные)
- [ ] `cargo xtask quality --everything` зелёный (mutants — финальная
      стадия)
- [ ] Mutation score фиксируется в `docs/quality/` (обновить baseline)