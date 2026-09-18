# 005: 100k real-time — СПАЙК (исследование, не прод-реализация)

## Вопрос

Given tiled-сцена 100k dynamic (+плитки пола), when меряем `step(1/60)`
через `probe_100k` с разбивкой `StepTiming` (broad/narrow/solver/island/trigger/substeps),
then виден пофазный breakdown 100k + доказан хотя бы один рычаг с выигрышем ≥2x —
да или нет, измерено?

## Почему важно

Broadphase-ядро закрыто (Auto + adaptive + incremental + bucketing + sleep):
10k tiled уже держит бюджет 60 FPS, а 100k — паритет tree/grid и ~10x лучше
исторических ~8 с, но до 16.7 мс не хватает ~36x (арифметика на числах из
постановки: 604/16.7). Доминанта по всем замерам — solver, поэтому следующий
шаг — не ещё один broadphase, а измеренный breakdown 100k + рычаги solver/narrow/sleep.

## (а) Где время сейчас — только из файлов, без выдуманных чисел

Источники: `docs/quality/perf-baseline-2026-08-27.md`,
`docs/quality/perf-baseline-2026-09-02.md`, `PLAN.md` Этап B (~строки 563–574),
`crates/physics/src/broadphase.rs:58–109` (`StepTiming`, `StepBudget`),
`crates/physics/examples/probe_100k.rs`, `crates/physics/examples/perf_probe.rs`,
`crates/physics/benches/solver_bench.rs`.

> Числа постановки (10k ~14–16 мс; 100k tree 759 мс vs grid-8 604 мс) в этих
> файлах НЕ найдены — ниже только то, что верифицировано по файлам.

**10k tiled, история ускорения (M1, release):**

| Срез | Время step | Источник |
|---|---|---|
| SAP 10k (10400 с полом), settle 30 | ~767 мс – 1.1 с | baseline-08-27 (`physics_bodies/10000` 767 мс); exploratory CI 1.1167 с / 1.0931 с / 1.1130 с |
| Grid 8.0 10k, лучший cell-size | **180.00 мс** (6.18x vs SAP, −8.5% vs grid-4) | baseline-08-27, расширенный follow-up |
| p1 incremental broadphase, Grid 4.0 | **10.1 мс** (~75x к SAP) | baseline-09-02, сводка p1+p2 |
| p2 + narrow cache, Grid 4.0 | **9.2 мс** (+~9% к p1) | baseline-09-02 |
| Этап B (PLAN.md): tiled 10k | **103.4 → 15.82 мс** (broad 4.86 / narrow 8.01 / solver 2.95, 63 FPS) | `PLAN.md:563–574` |
| solver_bench Этап B | 74 → **17.07 мс** (−77%) | `PLAN.md:570` |
| probe_100k 10k/5 шагов, M1: SAP steady | **604.88 мс** (736/757/365/729/566) | baseline-09-02 |
| probe_100k 10k/5 шагов, M1: Grid 4.0 steady | **174.31 мс** (193/193/163/179/160, ~3.5x vs SAP) | baseline-09-02 |
| solver_bench `--quick` 09-02, Grid 4.0 10k | **22.4 мс** (расхождение ~2.4x vs 9.2 мс — требует повторного прогона) | baseline-09-02 |

**100k tiled (104096 тел), единственные файлы-замеры:**

| Backend | Step 0 | Step 1 | Step 2 | Steady | Источник |
|---|---|---|---|---|---|
| Grid 8.0 | 8.1867 с | 8.0282 с | 8.0236 с | **~8.02 с** | baseline-08-27 §100k probes (runs 33245718111 / 33251548032, exploratory, разные runner'ы) |
| SAP | 97.6264 с | 79.4794 с | 79.5051 с | **~79.49 с** | там же |
| Grid diagnostics | cells 13448, raw 2 349 246, static 74 256, AABB-reject 2 074 990, candidates **100000** | | | | там же |
| SAP diagnostics | step 0: raw **5.42 млрд** (sweep-ось совпала с плоским распределением); step 2: raw ~22.5M (~9.6x больше Grid) | | | | там же |

100k намеренно вне criterion (`solver_bench.rs:156–161`: warmup step превышал
30 мин) — числа только из `probe_100k`.

**StepTiming breakdown — solver доминирует (09-02, `perf_probe`, без settle для islands):**

| Сценарий | ms/frame | broad | narrow | solver | Вывод |
|---|---|---|---|---|---|
| many_islands_256 (1024) | 41.770 | 0.701 | 4.576 | **34.740 (~83%)** | solver-доминанта |
| hetero 255 slow+1 fast (1024) | 42.147 | 0.784 | 5.855 | **33.948 (~80%)** | пер-островные iters экономят |
| islands_grid (1025, без settle) | 3.463 | 0.247 | 0.587 | 2.268 | settle меняет всё (см. ниже) |
| big_stack_32 (settle 60) | 1.853 | 0.035 | 0.168 | **1.599 (~86%)** | один остров |
| tall_stack_50 sub=12 | 7.005 | 0.069 | 0.547 | **6.247 (~89%)** | awake 48 |

Контраст settle: `islands_grid_16x16` после 60 settle — **1.925 мкс**
(острова спят), без settle — 3.463 мс. `physics_bodies/1000` settled — **1.96 мкс**.

**Регресс-маркер:** `big_stack_32` 2.06 мс (08-27) → 11.04 мс (09-02),
`deep_stack_128` 15.1 → 25.30 мс — подозревается `StepTiming`/narrow-cache
overhead на одном острове, требует профилирования (baseline-09-02, Выводы).

## (б) Топ-3 рычага — гипотезы с гейтом измерения

Порядок — по доле solver (~80–89% на dense) и мультипликатору substeps.

**1. Substep shedding через `StepBudget` + пер-островные итерации solver (ожидание: 2–6x на 100k).**
Механика: `StepBudget` (`broadphase.rs:85–109`) шеддит `P × S > max_pair_substeps`
(default 200k; калибровка: 10k settled ~14k×4=56k, cold ~14k×12=170k — на ≤10k
не кусается, на 100k шеддит к полу). Контакты не дропаются (completeness
сохранена — те же контакты меньшим числом проходов), per-island iteration
scaling не тронут. Плюс уже существующие: required substeps [4..12]
(`ceil(|v|·dt/240)`, порог max−min≥4), `last_substep_shed()`, `--kill-plane`
эксперимент в `probe_100k.rs`. Гейт: `timing ... substeps=` + `budget shed N`
до/после + stability (`awake/min_y`) не хуже.

**2. Sleep coverage — кто держит сцену бодрой (ожидание: 2–10x, бимодально).**
Механика: settled 1k спит целиком (1.96 мкс), холодная 10k бодрствует
(×12 substeps). `probe_100k` уже печатает `sleep asleep=N/total`,
`awake dynamics n/max_v/max_w`, `awake min_y` — смотреть, какой хвост
(скорости/угловые/минимальная y) мешает заснуть 100k. Гейт: доля asleep +
`awake_n/max_v/max_w` до/после твика порогов; verifier — повторный прогон
с settle (`perf_probe`-стиль) vs холодный старт.

**3. Narrow cache hit-rate + SAT cache (ожидание: 1.1–2x; p2 дал +9% на 10k).**
Механика: p2 — кэш только `cur_substep==0`, fast-miss bypass
(`rel_speed > 0.5` или `|w|² > 0.25` — мимо HashMap), эвикция stale;
10k Grid 4.0: raw 4.6M → candidates 14k — воронка узкая, но каждый промах
дорогой. SAT cache (16-шард `Vec<Mutex>` try_lock-only, `obb_sat_cached`,
по PLAN.md — 8→2 без регресса) — второй слой. Гейт: снять hit/miss/fast_miss
счётчики (их пока нет в печати probe — добавить только в throwaway-пример,
не в lib) + `narrow_phase_ms` из `StepTiming` до/после.
Вне топ-3, следом: islands parallelism (rayon narrow >256 пар; per-island
dispatch — выигрыш только на disjoint островах, на одном острове замеряет
gather/scatter overhead) и SIMD-wide lanes (friction уже имеет scalar/wide
паритет — образец для solver-пути).

## (в) План измерений — конкретные команды и метрики

```bash
# 0. Быстрый breakdown малых сцен (solver-доминанта):
cargo run -p ornis-physics --release --example perf_probe
# 1. 10k tiled, оба backend (калибровка против baseline-09-02):
cargo run -p ornis-physics --release --example probe_100k -- --bodies 10000 --steps 5
cargo run -p ornis-physics --release --example probe_100k -- --bodies 10000 --steps 5 --grid --cell-size 4
cargo run -p ornis-physics --release --example probe_100k -- --bodies 10000 --steps 5 --grid --cell-size 8
# 2. 100k tiled — главный замер спайка (долго; начинать с --steps 3):
cargo run -p ornis-physics --release --example probe_100k -- --bodies 100000 --steps 3 --grid --cell-size 8 --scene tiled
cargo run -p ornis-physics --release --example probe_100k -- --bodies 100000 --steps 3 --tree --scene tiled
cargo run -p ornis-physics --release --example probe_100k -- --bodies 100000 --steps 3 --auto --scene tiled
# 3. Рычаг substeps/shed + сон:
cargo run -p ornis-physics --release --example probe_100k -- --bodies 100000 --steps 3 --grid --cell-size 8 --scene tiled --kill-plane
cargo run -p ornis-physics --release --example probe_100k -- --bodies 100000 --steps 3 --scene sparse
cargo run -p ornis-physics --release --example probe_100k -- --bodies 100000 --steps 3 --scene giant_floor
# 4. Criterion-матрица (10k, settle 30; 100k НЕ гонять — см. solver_bench.rs:156):
cargo bench -p ornis-physics --bench solver_bench -- --quick
```

Снимать с каждого прогона: `step N: <ms>`, `stats ... pair_tests/static_skips/
aabb_rejections/candidates/cells`, `timing ... broad/narrow/solver/island/
trigger/substeps` (`step_timing()`, `broadphase.rs:58–72`), `sleep asleep=N/total`,
`awake dynamics n/max_v/max_w`, `awake min_y`, `budget shed N`, для `--auto` —
`auto active backend`. Дополнительно: `StepTiming::per_substep_ms()` (цена
одного сабстепа) и machine/toolchain/commit в шапку (как baseline-09-02:
M1/16 ГБ/rustc 1.97.0/release/`fb9d0a4`).

## (г) Критерий успеха спайка

НЕ «стало 16 мс». Успех = **измерен breakdown 100k tiled
(broad/narrow/solver/island/trigger/substeps + sleep-диагностика) на
grid-8/tree/auto + доказан один рычаг ≥2x** на том же железе/коммите
(повторный прогон, те же флаги, таблица до/после). Неудача тоже результат:
breakdown без рычага ≥2x → зафиксировать доминанту и закрыть спайк выводом
«нужен другой класс решений».

## Статус

OPEN (план измерений, замеров 100k в этом спайке ещё нет).
Правило спайка: `crates/*/src` только читать; throwaway-код — только
в `crates/physics/examples/` как новый файл (рядом с `probe_100k.rs`).
