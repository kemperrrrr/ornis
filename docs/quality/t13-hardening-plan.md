# T13 Hardening Plan — публичные API физики (33 timeout)

> 🔴 **ДИСКЛЕЙМЕР (по факту выполнения 2026-08-24):** этот план строится на
> **ЛОЖНОЙ предпосылке**. Утверждения про «уходят в infinite loop при
> деградированных входах», «debug_assert! прерывает цикл при зависании»,
> «бесконечная итерация солвера» — **не верны**. В `engine.rs` нет
> `while`/`loop` (все циклы — `for` с фиксированными границами), так что
> бесконечный цикл невозможен. То, что mutants видит как «timeout», — это
> **медленно падающие** stability-тесты при мутантном коде (система
> расходится, тесты на сотнях шагов не конвергируют за 60s, но падают
> позже), а не зависание. `debug_assert!` на finiteness **не ловит** мутации
> `*→+`/`+=↔-=` (те конечны, но неверны) и **не прерывает** никакой цикл.
> Реальное закрытие timeout'ов = поднятый `--timeout` + regression-тесты с
> независимыми оракулами (glam). См. `t13-followups.md`, «РЕВЬЮ-ЗАМЕТКИ».

> 🔎 ЧЕРНОВИК ДЛЯ РЕВЬЮ — не коммитить до согласования.
>
> Источник: `cargo mutants --package ornis-physics --timeout 60`,
> прогон 2026-08-23, `mutants.out/timeout.txt` (33 строки в `engine.rs`).
> Это **не про тесты** (T8-T12, T14), а про **защитное программирование**.
>
> ⚠️ **[ЛОЖЬ В ПЕРВОНАЧАЛЬНОМ ТЕКСТЕ]** Далее было: «в проде эти места
> потенциально уходят в infinite loop при деградированных входах (NaN, inf,
> нулевая инерция, повреждённый save-файл). На mutants мы это видим как
> зависание; на реальных данных — как зависший игровой кадр.» — **ошибочно**
> (см. ДИСКЛЕЙМЕР выше: бесконечного цикла нет, это медленно падающие
> тесты под мутантным кодом, а не зависание в проде).

## Цель

> ⚠️ **[ЛОЖЬ В ПЕРВОНАЧАЛЬНОМ ТЕКСТЕ]** Ниже цель сформулирована как
> «debug_assert! падает ДО того, как цикл зависнет» и «33 timeout'а → 0
> при `--timeout 60`». **Ошибочно:** (а) бесконечного цикла нет; (б) assert
> на finiteness не ловит алгебраические мутации; (в) `--timeout 60` давал
> 14 timeout (на 5 функциях), не 0. Честная цель — защитное программирование
> ПЛЮС поднятый `--timeout` ПЛЮС regression-тесты с независимыми оракулами
> для переклассификации TIMEOUT→CAUGHT.

Добавить `debug_assert!` на входах и в ключевых точках 10 функций так, чтобы:

1. **В debug/test сборке**: деградированный вход → `debug_assert!` падает
   с понятной диагностикой.
2. **В release сборке**: zero-cost (компилятор выкидывает `debug_assert!`).
3. **В cargo mutants**: те же мутации приводят к **caught** (тест падает)
   или **missed** — но без ложных `timeout` нужен **поднятый `--timeout`**
   (60→300+) и **regression-тесты с независимыми оракулами** (assert'ы
   входа сами по себе этого не дают).

## Метрика

- **До**: 33 timeout'а в `mutants.out/timeout.txt`.
- ⚠️ **[ЛОЖЬ]** «После: 0 timeout'ов при `--timeout 60`» — **неверно**.
  Без поднятого `--timeout` и regression-тестов timeout'ы остаются
  (медленно падающие тесты). Честно: цель — переклассификация в CAUGHT.

## Общий паттерн

```rust
// На входе публичных методов — finiteness + разумные диапазоны:
debug_assert!(dt.is_finite() && dt > 0.0, "dt must be finite and positive, got {dt}");
debug_assert!(imp.is_finite(), "impulse must be finite, got {imp}");
debug_assert!(self.bodies.iter().all(|b| b.inertia.is_finite()),
              "inertia tensor must be finite");
```

Почему `debug_assert!` (а не `assert!`):

> ⚠️ **[ЛОЖЬ В ПЕРВОНАЧАЛЬНОМ ТЕКСТЕ]** Далее было: «В debug/test — ловит
> баги **до** зависания» и «`debug_assert!` достаточно для **прерывания
> цикла при зависании**». — **ошибочно**: бесконечного цикла нет, а
> assert'ы на finiteness не ловят алгебраические мутации и ничего не
> «прерывают».

- Game engine — hot path критичен, в release нельзя платить.
- В debug/test — ловит баги (но **не** алгебраические мутации `*→+`/`+=↔-=`).
- Для закрытия timeout'ов нужны отдельные regression-тесты с независимыми
  оракулами + поднятый `--timeout`, а не только debug_assert!.

## Карта по функциям (10 шт., 33 мутации)

### 1. `mul_inv_inertia` (lines 91, 92) — `pub(crate)`

3 мутации: `*` → `+`/`/` в inv_inertia_axis умножении.

**Что ломается**: если `inv_inertia_axis(inertia.x)` мутирован (или `body.x` после `orientation.inverse() * v` становится inf/NaN), результат `orientation * scaled` = NaN, downstream вечно считает с NaN.

**Assert** на входе:
```rust
debug_assert!(inertia.is_finite(), "inertia must be finite, got {inertia:?}");
debug_assert!(v.is_finite(), "v must be finite, got {v:?}");
debug_assert!(orientation.is_finite(), "orientation must be finite");
```

---

### 2. `effective_mass` (lines 131-135) — `private fn`

5 мутаций: замена константы `f32` на `1.0`/`-1.0`, `+` → `*`/`-` в выражении.

**Что ломается**: `b.inv_mass` — если в мутации стал `1.0` (а должен быть `0.0` для статика), `effective_mass` не учитывает статическое тело, и солвер думает, что оба тела динамические → бессмысленная итерация.

**Assert** на входе:
```rust
debug_assert!(bodies[i].inv_mass.is_finite() && bodies[i].inv_mass >= 0.0);
debug_assert!(bodies[j].inv_mass.is_finite() && bodies[j].inv_mass >= 0.0);
debug_assert!(bodies[i].inertia.is_finite());
debug_assert!(bodies[j].inertia.is_finite());
debug_assert!(n.is_finite());
```

---

### 3. `solve_small` (lines 162-175) — `private fn`

8 мутаций: `*` → `+`/`/`, `/` → `*`/`%`, `-=` → `+=`/`/=`.

**Что ломается**: backward substitution `s / m[r][r]` — если `m[r][r]` стал `0` или near-zero после мутации, деление не уменьшает `s`, цикл **не сходится**.

**Assert**:
```rust
// Перед backward substitution:
for r in 0..n {
    debug_assert!(m[r][r].is_finite() && m[r][r].abs() > 1e-12,
                  "solve_small: pivot m[{r}][{r}] = {} — singular or NaN",
                  m[r][r]);
}
```

Альтернатива — clamp `m[r][r]` к `eps` перед делением (тогда assert не нужен). Решаем по согласованию.

---

### 4. `apply_impulse` (lines 203-205) — тело `private fn` на ~line 194

3 мутации: `-=` → `/=`, `+=` → `-=`, `-=` → `+=`.

**Что ломается**: `a.velocity -= imp * a.inv_mass` — если `a.inv_mass` мутирован (статическое тело с `inv_mass=0` не должно получать velocity), то либо ничего не меняется, либо velocity взрывается.

**Assert** на входе:
```rust
debug_assert!(i != j, "apply_impulse: i == j");
debug_assert!(imp.is_finite(), "imp must be finite, got {imp}");
debug_assert!(bodies[i].inv_mass.is_finite() && bodies[i].inv_mass >= 0.0);
debug_assert!(bodies[j].inv_mass.is_finite() && bodies[j].inv_mass >= 0.0);
```

---

### 5. `solve_normal_block` (line 297, мутации в 312, 349, 388, 390) — `private fn`

6 мутаций: `-` → `+`/`/` в арифметике шага солвера.

**Что ломается**: знаковая путаница в `lambda = -accum / k_eff` (или аналогично) — если знак инвертирован, `accum` и `k_eff` оба одного знака → `lambda` не уменьшает constraint violation → **система не конвергирует** (тесты падают медленно, не бесконечный цикл).

> ⚠️ **[ЛОЖЬ]** Оригинал: «→ **бесконечная итерация солвера**» — **неверно** (нет while/loop; циклы конечны, просто результат расходится).

**Assert**:
```rust
// Внутри цикла по итерациям солвера:
debug_assert!(accum.is_finite() && k_eff.is_finite(),
              "solve_normal_block: non-finite accum={} k_eff={}", accum, k_eff);
```

Плюс `assert!(iterations < MAX_ITERATIONS)` чтобы цикл гарантированно завершался.

---

### 6. `BuiltinPhysicsEngine::integrate_velocities` (line 1222) — **pub fn**

1 мутация: `*` → `/` в `body.torque * dt`.

**Что ломается**: `body.torque * dt` → `body.torque / dt` (если `dt` мутирован или очень мал) → inf, после `mul_inv_inertia` → NaN.

**Assert**:
```rust
debug_assert!(dt.is_finite() && dt > 0.0, "dt must be positive finite, got {dt}");
// После умножения torque * dt:
let delta = body.torque * dt;
debug_assert!(delta.is_finite(), "torque*dt overflowed: torque={:?} dt={}", body.torque, dt);
```

---

### 7. `BuiltinPhysicsEngine::rebuild_islands::find` (line 1264) — **inner fn, не pub**

1 мутация: `!=` → `==` в `while parent[x] != x`.

**Что ломается**: **классический infinite loop** в union-find без path compression. Любой не-корневой элемент зацикливается навсегда.

> ⚠️ **[УТОЧНЕНИЕ]** Это утверждение про `find` — **отдельный, валидный случай** (там действительно `while parent[x] != x`, и мутация `!=`→`==` даёт настоящий бесконечный цикл). Это **единственное** реальное место с бесконечным циклом во всём T13; оно НЕ относится к 5 функциям, которые фактически правились в followups (`mul_inv_inertia` и др.). Общий ДИСКЛЕЙМЕР про «нет while/loop» к `find` не применим.

**Решение**: переписать на итеративную версию с path halving:
```rust
fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];  // path halving
        x = parent[x];
    }
    x
}
```
Это **уже** итеративная с path halving, мутация `!=` → `==` превращает её в `while false` — нет, превращает **условие** в `while parent[x] == x`, что для корня x выполнится сразу (выход), а для **не-корня** — зависнет. Так что **защита**: assert, что после find мы на корне (т.е. `parent[result] == result`).
```rust
let root = find(&mut parent, x);
debug_assert!(parent[root] == root, "find did not converge: parent[{}]={}", root, parent[root]);
```

---

### 8. `BuiltinPhysicsEngine::solve_continuous` (line 1631) — **pub fn**

1 мутация: `*` → `/` в `self.bodies[h].velocity * sub_dt`.

**Что ломается**: `velocity * sub_dt` → `velocity / sub_dt` → если `sub_dt` очень мал, или `velocity` очень велик → inf.

**Assert**:
```rust
debug_assert!(sub_dt.is_finite() && sub_dt > 0.0, "sub_dt must be positive, got {sub_dt}");
// После умножения:
let disp = self.bodies[h].velocity * sub_dt;
debug_assert!(disp.is_finite(), "velocity*sub_dt overflowed");
```

---

### 9. `BuiltinPhysicsEngine::partition_into_islands::find` (line 1841) — **inner fn, не pub**

1 мутация: `!=` → `==` (тот же паттерн, что и #7).

**Решение**: то же — добавить `debug_assert!` после find. (Или, поскольку функция **дублирует** `rebuild_islands::find`, вынести в shared helper.)

---

### 10. `BuiltinPhysicsEngine::solve_scalar_friction` (lines 2582, 2617, 2620) — **pub fn**

4 мутации: `*` → `/` (×3), `-` → `/` (×1).

**Что ломается**: `acc[k] * st.mu` (или `*` на `t`) → `acc[k] / st.mu` → если `mu` мутирован (или `acc[k]` большое) → inf. Или `new_t - st.acc_friction[k]` → `new_t / st.acc_friction[k]` → division by near-zero → NaN.

**Assert**:
```rust
// В цикле по k:
debug_assert!(acc[k].is_finite(), "acc must be finite");
debug_assert!(st.mu.is_finite() && st.mu >= 0.0, "mu must be non-negative, got {}", st.mu);
// После new_t вычисления:
debug_assert!(new_t.is_finite(), "friction impulse overflowed");
```

---

## Сводная таблица (что в какой функции)

| # | Функция | pub? | Мутаций | Категория |
|---|---|:-:|---:|---|
| 1 | `mul_inv_inertia` | pub(crate) | 3 | finiteness входа |
| 2 | `effective_mass` | private | 5 | finiteness + non-negative inv_mass |
| 3 | `solve_small` | private | 8 | pivot finiteness + non-zero |
| 4 | `apply_impulse` | private | 3 | finiteness + non-negative inv_mass |
| 5 | `solve_normal_block` | private | 6 | iter finiteness + max iters |
| 6 | `integrate_velocities` | **pub** | 1 | dt finiteness + delta finiteness |
| 7 | `rebuild_islands::find` | inner | 1 | root convergence assert |
| 8 | `solve_continuous` | **pub** | 1 | sub_dt finiteness + disp finiteness |
| 9 | `partition_into_islands::find` | inner | 1 | root convergence (duplicate of #7) |
| 10 | `solve_scalar_friction` | **pub** | 4 | acc/mu/impulse finiteness |
| | **Total** | | **33** | |

**Приоритет**: сначала **pub** функции (6, 8, 10) — это **поверхность атаки** извне. Потом **inner** (1, 2, 3, 4, 5, 7, 9) — внутренние helper'ы, защищают от деградации внутри.

## План применения

**Раунд 1 — сначала самые критичные (функции 6, 8, 10) + найти грабли:**
1. `BuiltinPhysicsEngine::integrate_velocities` (1 assert)
2. `BuiltinPhysicsEngine::solve_continuous` (1 assert)
3. `BuiltinPhysicsEngine::solve_scalar_friction` (4 assert'а в цикле)
4. `rebuild_islands::find` + `partition_into_islands::find` (один паттерн, две точки)

**Раунд 2 — внутренние helper'ы:**
5. `mul_inv_inertia`
6. `effective_mass`
7. `apply_impulse`
8. `solve_small`
9. `solve_normal_block`

После каждого раунда: `cargo mutants --package ornis-physics --timeout 60` для верификации.

**Открытые вопросы** (на согласование):
1. `solve_small` — **clamp к `eps` перед делением** (чинит math) или **assert на non-zero** (только ловит) ? Clamp — хирургичнее, не падает на легитимных near-singular матрицах.
2. `solve_normal_block` — добавить `max_iterations` cap (страховка от **расхождения** на любых мутациях, не только текущих 6; не от «зависания», т.к. бесконечного цикла нет).
3. **Покрытие тестами**: после фиксов — добавить **хотя бы один regression test** для каждого из 10 мест (мутируем, проверяем что assert сработал). Это превращает T13 из «hardening» в «hardening + coverage». Обсудим — может быть overkill для раунда 1.
4. **Macro для повторяющихся assert'ов** — есть 7 мест, где assert одинаковый (`is_finite` + проверка диапазона). Вынести в `debug_assert_finite!(x)` макрос? Может быть, может быть overkill.

## Следующие шаги (на согласование)

1. **Согласовать** этот план — какие assert'ы ок, какие нет, какие вопросы (clamp, max_iter, макрос) решаем.
2. **Раунд 1** — пулл-реквест с 3 pub-функциями + 2 find'ами. ~30 строк кода.
3. **Прогон mutants** после merge'а — ожидаем `timeout.txt: 0` (по крайней мере для этих функций).
4. **Раунд 2** — оставшиеся 5 inner-функций. ~40 строк.
5. **Финальный прогон** — `mutants.out/timeout.txt` должен быть пустым.

Если **clamp vs assert** для `solve_small` не принципиально — по умолчанию предлагаю **assert** (минимум изменений в логике, только safety net). Если хочешь clamp — скажи, поправлю.

Жду фидбек: что ок, что нет, что пересмотреть. После согласования — оформлю как draft PR в Ornis.
