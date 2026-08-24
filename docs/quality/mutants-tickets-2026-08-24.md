# Мутант-тикеты — ornis-physics, прогон 2026-08-24

> 🔎 ЧЕРНОВИК ДЛЯ РЕВЬЮ — не коммитить.
>
> Источник: `cargo mutants --package ornis-physics --timeout 60`
> (verify-heavy workflow, ручной запуск): **2351 мутант** —
> **706 caught (30%) · 1462 missed (62%) · 33 timeout · 149 unviable**.
> Базовый прогон, тикеты пока **все открыты**.
>
> Это **ornis-physics**, не ornis-core как в `mutants-tickets-2026-08-11.md` —
> тикеты продолжают сквозную нумерацию с T8.

## Статус (2026-08-24, первый прогон, ничего не закрыто)

| # | Тикет | Файлы | Missed / TO | Реальный риск | Статус |
|---|-------|-------|------------:|---------------|--------|
| T8  | Engine core: солвер острова + collision detection | engine.rs | 728 missed | высокий | 🟡 открыт |
| T9  | GPU-имплементация солвера | gpu.rs | 274 missed | высокий | 🟡 открыт |
| T10 | SIMD-wide солвер (G7) | wide.rs | 166 missed | средний | 🟡 открыт |
| T11 | Геометрические distance-функции | distance.rs | 98 missed | средний | 🟡 открыт |
| T12 | Shape::inertia | shape.rs | 60 missed | средний | 🟡 открыт |
| T13 | Hardening публичных API физики (timeout'ы) | engine.rs | 33 timeout | **высокий** | 🔴 открыт |
| T14 | `RigidBody::set_orientation` (no-op) + мелочи | body.rs | 2 missed | **высокий** | 🔴 открыт |
| —   | Хвосты в `math.rs` (22), `body.rs` остаток | math.rs, body.rs | 24 missed | низкий | ⚪️ отложен |

**Итого покрыто тикетами: ~1438 missed + 33 timeout из 1495 непокрытых.
Остаток ~24 missed — в хвостах, попадут в ближайший затрагиваемый тикет.**

> Главное здесь не «закрыть все 1462 missed» — в численной физике с
> tolerance'ами это нереалистично. Главное — **T13 (hardening)** и
> **T14 (set_orientation no-op)**, это **реальные баги**, не тестовые.
> Остальное — план на итерации, не «срочно».

---

> Классификация ниже — по кластерам missed, каждый тикет = один
> вертикальный срез «тесты, которые убивают эту группу».
> Способ проверки: повторный `cargo mutants --package ornis-physics
> --timeout 60` и сравнение missed по затрагиваемым файлам.

## Сводка

| # | Тикет | Missed | Топ-функции | Что предлагается |
|---|-------|-------:|-------------|-------------------|
| T8  | Engine core | 728 | `solve_island_velocity` (66), `build_manifold_state` (63), `capsule_vs_capsule` (81), `box_manifold` (44), `solve_normal_block` (35), `solve_scalar_friction` (31), `sphere_vs_capsule` (33) | property-тесты на инварианты солвера: импульс до/после, energy bound, позиционная коррекция ≤ толщины манифольда |
| T9  | GPU-солвер | 274 | `contact_solver` (156), `world_inertia_matrix` (51), `GpuBatch::fill_lane` (39) | сравнение CPU и GPU результата на одних входных данных (golden parity); рефактор длинной функции в под-функции для покрытия |
| T10 | SIMD-wide солвер | 166 | `WideBatch::solve_iteration` (81), `WideBatch::world_inertia` (33) | тесты на `Batch<LANES>` API; reference-значения для маленьких (2–4 lane) batches |
| T11 | Distance-функции | 98 | `obb_corners` (48), `seg_seg_closest` (32) | exact-assert'ы на целочисленных ветках + reference values для float-веток; tolerance 0.001 не ловит `*` ↔ `/` |
| T12 | Shape::inertia | 60 | `Shape::inertia` (50) | reference values для каждой формы (sphere, box, capsule, cylinder, mesh-bounds) |
| T13 | Hardening API | 33 TO | `mul_inv_inertia`, `k_entry`, `solve_small`, `apply_impulse`, `solve_normal_block`, `solve_continuous`, `solve_scalar_friction` | `assert!(x.is_finite())` и проверки диапазонов в публичных функциях; **это не тест, это hardening** |
| T14 | set_orientation no-op | 2 | `RigidBody::set_orientation` (no-op мутация) | один точечный `assert_eq!(body.orientation, q)` после `set_orientation(q)` |

---

## T8 · Engine core (728 missed)

`crates/physics/src/engine.rs`. Самый большой кластер — основной солвер
и collision detection. Топ пропущенных мутаций:

- `solve_island_velocity` (66) — `BuiltinPhysicsEngine::solve_island_velocity`
- `build_manifold_state` (63) — формирование контактного манифольда
- `capsule_vs_capsule` (81) — collision narrow phase
- `sphere_vs_capsule` (33) — collision narrow phase
- `box_manifold` (44) — генерация контактов box-vs-X
- `solve_normal_block` (35) — нормальный constraint solver
- `solve_scalar_friction` (31) — friction solver

Преобладают **численные** мутации (`*` ↔ `/`, `+` ↔ `-`, `<` ↔ `<=`).
Tolerance в assert'ах (~0.001) пропускает их — это **известная проблема
численного кода**.

**Что предлагается**: property-тесты на инварианты солвера, а не на
конкретные числовые значения. Например:

- `Σ(масса × скорость)` до и после шага — сохраняется в пределах 1e-3.
- Относительная кинетическая энергия ≤ начальной (нет «взрыва»).
- После `step` позиции тел — конечные (не NaN/inf).
- Пары контактирующих тел не проникают друг в друга больше, чем
  `restitution * penetration_depth`.

Это **свойства**, а не golden values — они гораздо устойчивее к
численным мутациям и реально ловят регрессии.

---

## T9 · GPU-солвер (274 missed)

`crates/physics/src/gpu.rs`. Свежий код из `9ce2ba0 G7: SIMD-wide
контактный солвер + GPU-перенос (physics)`. Топ:

- `contact_solver` (156) — ядро GPU-солвера
- `world_inertia_matrix` (51) — вычисление инерции в GPU layout
- `GpuBatch::fill_lane` (39) — заполнение SIMD-lane

**Что предлагается**:

1. **CPU/GPU parity** — на одинаковых входных данных прогнать CPU
   солвер и GPU солвер, сверить позиции/скорости тел после N шагов
   в пределах tolerance. Это сразу ловит 90% missed в `contact_solver`
   и одновременно страхует от регрессий между CPU и GPU.
2. **Рефактор длинной функции** — `contact_solver` 156 missed'ов —
   это, вероятно, **одна большая функция**, которую стоит разбить
   на под-функции (как `solve_small`, `apply_impulse`,
   `solve_normal_block` уже сделано). После разбиения — точечные
   тесты на каждую.

---

## T10 · SIMD-wide солвер (166 missed)

`crates/physics/src/wide.rs`. Тоже из G7:

- `WideBatch::solve_iteration` (81)
- `WideBatch::world_inertia` (33)

**Что предлагается**: тесты на публичный API `Batch<LANES>` — для
маленьких LANES (2, 4) reference-значения вычислимы руками, для
больших — parity с CPU солвером. Та же стратегия, что и T9.

---

## T11 · Distance-функции (98 missed)

`crates/physics/src/distance.rs`. Топ:

- `obb_corners` (48) — углы oriented bounding box
- `seg_seg_closest` (32) — ближайшая точка между двумя отрезками

**Что предлагается**:

- **Целочисленные ветки** — exact `assert_eq!` (например, на degenerate
  cases, когда отрезки параллельны или пересекаются).
- **Float-ветки** — reference values, **посчитанные вручную или
  независимой реализацией** (например, numpy/scipy). Tolerance 0.001
  в assert'ах **не работает** для мутаций `*` ↔ `/` — нужно либо
  сравнение с **независимой формулой**, либо exact на целочисленных
  кратных значениях.

---

## T12 · Shape::inertia (60 missed)

`crates/physics/src/shape.rs`. `Shape::inertia` (50) — инерция формы.

**Что предлагается**: reference values для каждого варианта `Shape`:

- `Sphere(r)` — `(2/5) * m * r²` — можно exact-assert на формулу
- `Box(half_extents)` — диагональная матрица `m/12 * (h² + d²)` и т.д.
- `Capsule(r, h)`, `Cylinder(r, h)`, `Mesh(bounds)` — аналогично

Это **golden values**, и они должны быть exact (float rounding
учитывается в литералах). Ловят мутации вроде `*` ↔ `/`, `+` ↔ `-`.

---

> 🔴 **ДИСКЛЕЙМЕР:** раздел **§T13 ниже содержит ЛОЖНЫЙ диагноз**. Утверждения
> про «бесконечный цикл» при деградированных входах — **не верны**: в
> `crates/physics/src/engine.rs` **нет** `while`/`loop` (все циклы — `for`
> с фиксированными границами), поэтому бесконечный цикл физически
> невозможен. Реальная природа «timeout'ов» — **медленно падающие**
> stability-тесты при мутантном коде (система становится жёсткой/
> осциллирующей, тесты на сотнях шагов не конвергируют за `--timeout 60s`,
> но падают позже). Подробности и честный итог — в `t13-followups.md`,
> раздел «РЕВЬЮ-ЗАМЕТКИ».

## T13 · Hardening публичных API физики (33 timeout)

**Это не про тесты — это про код.** Все 33 timeout'а в `engine.rs`.

> ⚠️ **[ЛОЖЬ В ПЕРВОНАЧАЛЬНОМ ТЕКСТЕ]** Далее было: «Конкретные функции,
> где мутация приводит к **зависанию** на 60s» и «код не защищён от
> деградированных входов: NaN, infinity ... **уходят в бесконечный цикл**»
> и «**знаковые мутации в итерационных циклах → бесконечный цикл**». —
> **ошибочно.** Бесконечных циклов нет (см. ДИСКЛЕЙМЕР выше). Это
> **медленно падающие** тесты, а не зависания.

Конкретные функции, где мутация приводит к **таймауту раннера тестов (60s)**:

- `mul_inv_inertia` — `*` → `+` или `/` приводит к расходящейся физике
- `k_entry` — замена `f32` константы на `1.0` или `-1.0` приводит к расхождению
- `solve_small`, `apply_impulse`, `solve_normal_block` — знаковые мутации
  меняют алгебру решателя → система не конвергирует за 60s (тесты падают позже)
- `solve_continuous` (строка 1631) — `*` → `/` в CCD-проходе
- `solve_scalar_friction` (строки 2582, 2617, 2620) — деление меняет сходимость

> ⚠️ **[ЛОЖЬ]** Утверждение «код не защищён от деградированных входов
> (NaN, infinity, нулевая инерция, деление на ноль **уходят в бесконечный
> цикл**), поэтому нужны debug_assert!» — **неверно по факту**: (а)
> бесконечного цикла нет; (б) debug_assert! на finiteness **не ловят**
> мутации `*→+`/`+=↔-=` (те конечны, но неверны). Assert'ы полезны как
> защитное программирование, но они **не** то, что закрывает timeout'ы —
> это делает поднятый `--timeout` + regression-тесты с независимыми
> оракулами (glam).

**Что предлагается** (это не тест, это **защитное программирование**):

```rust
pub fn mul_inv_inertia(...) -> ... {
    debug_assert!(inv_inertia.iter().all(|x| x.is_finite()));
    debug_assert!(velocity.iter().all(|x| x.is_finite()));
    // ...
}
```

И аналогично для других публичных API. `debug_assert!` — потому что
в release'ах мы не хотим платить за проверки, а в debug/test —
хотим. Если хочется runtime — `assert!` или явный error path.

**Это самый приоритетный тикет из списка**: один хороший `debug_assert!`
здесь стоит десятков новых тестов.

---

## T14 · `RigidBody::set_orientation` (no-op) + мелочи

**Самый дешёвый тикет в списке — и самый поучительный.** В `missed.txt`:

```
crates/physics/src/body.rs:83:9: replace RigidBody::set_orientation with ()
```

То есть мутатор взял **всю функцию** и заменил на `()` — функция
**перестала делать что-либо** — и тест остался зелёным. Это значит,
что **ни один тест не проверяет, что после `set_orientation(q)`
ориентация тела реально изменилась**. Серьёзный зомби-тест.

Дополнительно в `body.rs`:

- `RigidBody::build:35:44` — `replace / with *` в построении тела
- `RigidBody::apply_torque:92:21` — `replace += with -=` (знак
  приложения torque)

**Что предлагается**: 5-строчный focused тест:

```rust
#[test]
fn set_orientation_changes_orientation() {
    let mut body = RigidBody::default();
    let q = Quat::from_rotation_y(1.0);
    body.set_orientation(q);
    assert_eq!(body.orientation(), q);
}
```

Закрывает критический зомби-тест. Аналогично — assert на
`apply_torque` (что скорость изменилась) и `build` (что итоговое
тело соответствует входным данным).

---

## Что **не** пытаемся закрыть полностью

- 1462 missed — это **план на итерации**, не «закрыть за раз». Каждый
  прогон mutants — это снимок, не «закрыл и забыл».
- `math.rs` (22 missed) — мелкие численные функции; в хвосте T12.
- Численные мутации в геометрии (`*` ↔ `/` в `seg_seg_closest`)
  частично неизбежны при tolerance-based assert'ах — **T11 решает
  это reference values, не «закрыть все»**.

## Способ проверки каждого тикета

После фикса тестов по тикету — повторный прогон:

```bash
cargo mutants --package ornis-physics --timeout 60
```

Сравнить `missed` и `timeout` по затрагиваемым файлам. Цель по T13 —
**33 timeout → 0** (после добавления `debug_assert!`). Цель по T14 —
**3 missed → 0** в `body.rs`. По остальным — снижение, не ноль.

---

## Где смотреть в артефакте

- `mutants.out/summary.txt` — финальная сводка
- `mutants.out/missed.txt` — список зомби-тестов (главное)
- `mutants.out/timeout.txt` — список зависаний
- `mutants.out/diff/<file>/*.diff` — git-style diff каждого мутанта
- `mutants.out/log/<file>/*.log` — лог `cargo test` для каждого мутанта
