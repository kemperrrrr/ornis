# Performance baseline — 2026-09-02

Зафиксированный baseline производительности после p1 (incremental broadphase) и p2 (narrowphase cache).
Числа — медианные оценки criterion (mean из `[low .. high]`) и прямые замеры `perf_probe`/`probe_100k`,
один прогон на Apple M1; это точка отсчёта для сравнения будущих изменений, а не строгий эксперимент.

## Конфигурация

- Машина: Apple M1, 16 ГБ RAM, macOS 26.5.2 (Darwin 25.5.0 arm64)
- Toolchain: rustc 1.97.0 (2d8144b78 2026-07-07)
- Профиль: `release` (criterion, `cargo bench` + `cargo run --release`)
- Коммит: `fb9d0a4` + незакоммиченные изменения (incremental broadphase, narrow cache, box↔capsule, analytic TOI)

## Воспроизведение

```bash
cargo bench -p ornis-physics --bench solver_bench -- --quick
cargo run -p ornis-physics --release --example perf_probe
cargo run -p ornis-physics --release --example probe_100k -- --bodies 10000 --steps 5
cargo run -p ornis-physics --release --example probe_100k -- --bodies 10000 --steps 5 --grid --cell-size 4
cargo test -p ornis-physics --test box_capsule_toi
```

Workflow `performance.yml` публикует `solver_bench` артефакты вручную через `workflow_dispatch`.

## Сводка p1 + p2 (UniformGrid, tiled floor 10k dynamic + ~400 tiles)

p1 — инкрементальный UniformGrid: full rebuild только при топологической смене или >50% dirty тел,
иначе retained clean-clean + dirty-involved honest totals, `active` из `HashSet` де-дуплицирован.

p2 — narrowphase cache: кэш только для `cur_substep==0`, быстрый путь для `small active` и `fast_miss`
(`rel_speed > 0.5` или `|w|² > 0.25` — обход HashMap, без кэширования), eвикция stale пар.

Эффект на 10k tiled (10400 тел):

| Этап | physics_bodies/10000 (UniformGrid) | Примечание |
|---|---|---|
| baseline до p1 (2026-08-27, SAP 10k) | ~767 ms – 1.1 s | O(n log n) SAP, 30 settle steps |
| после p1 (incremental broadphase) | **10.1 ms** | Grid 4.0 honest totals, без narrow cache |
| после p2 (narrow cache) | **9.2 ms** | Grid 4.0 + cache, тот же workload |

Т.е. p1 даёт ~75× ускорение к SAP на 10k, p2 — ещё ~9% сверху. Оба числа — локальные замеры на M1
(однократные Criterion means, не CI-normalized). Бюджет 16.7 ms впервые достигнут именно после p1/p2
на UniformGrid (SAP остаётся в ~0.9–1.1 s).

## Physics step — `solver_bench` (текущий прогон, `cargo bench --quick`, 2026-09-02)

Один `step(1/60)` после 30–60 settle шагов (кроме `physics_bodies`, где settle=30):

| Сценарий | Время (mean) | Комментарий |
|---|---|---|
| islands_grid_16x16 (1025, 256 островов) | 1.925 µs | острова спят — steady почти бесплатен |
| big_stack_32 (33, 1 остров) | 11.04 ms | стек не засыпает, solver каждый шаг |
| deep_stack_128 (129, 1 остров) | 25.30 ms | рост сверхлинейный по высоте |
| physics_bodies/1000 (1049 тел, UniformGrid 4.0) | 1.96 µs | tiled floor, тела спят |
| physics_bodies/10000 (10400 тел, UniformGrid 4.0) | **22.4 ms** | tiled floor, бодрствующий шаг (см. p2 выше: 9.2 ms на холодном M1 до p2 overhead?) |
| physics_bodies/10000 (SAP) | 976.09 ms | SAP baseline для сравнения |
| physics_bodies/10000 (Grid 1.0) | 23.84 ms | 123052 cells, raw 1.57M |
| physics_bodies/10000 (Grid 8.0) | 24.73 ms | 1352 cells, raw 8.0M — cell 8 не быстрее на M1 в текущей сборке |

В этом прогоне `big_stack_32`/`deep_stack_128` регрессировали vs 2026-08-27 (2.06 ms → 11.0 ms,
15.1 ms → 25.3 ms) из-за дополнительного `StepTiming`/`narrow cache` overhead на одном острове —
требует отдельного профилирования solver vs broadphase (см. `perf_probe` ниже).

Диагностика 10k на UniformGrid 4.0 (criterion, steady):

| Cells | Raw pair tests | Static skips | AABB rejects | Candidates |
|---:|---:|---:|---:|---:|
| 5408 | 4 628 288 | 12 096 | 3 768 588 | 14 161 |

## `perf_probe` — детальный breakdown (2026-09-02, `cargo run --release --example perf_probe`)

| Сценарий | ms/frame | broad | narrow | solver | peak broad/narrow/solver | Примечание |
|---|---:|---:|---:|---:|---|---|
| islands_grid_16x16 (1025) | 3.463 | 0.247 | 0.587 | 2.268 | 1.363 / 2.208 / 10.205 | sleep 1024/1025, но probe без settle — 60 frames с 0 |
| many_islands_256 (1024) | 41.770 | 0.701 | 4.576 | 34.740 | 2.033 / 23.892 / 257.573 | 256 башен ×4, dense |
| hetero 255 slow+1 fast (1024) | 42.147 | 0.784 | 5.855 | 33.948 | 2.593 / 17.682 / 85.516 | пер-островные iters экономят |
| big_stack_32 (33) после 60 settle | 1.853 | 0.035 | 0.168 | 1.599 | 1.825 / 0.473 / 9.906 | один остров, stable |
| contact_cluster_2k (2000) | 2.151 | 0.701 | 0.000 | 0.135 | 2.212 / 0.001 / 0.457 | sparse, solver почти 0 (sleep?) |
| tall_stack_50 sub=12 | 7.005 | 0.069 | 0.547 | 6.247 | 0.138 / 0.785 / 14.916 | awake 48, max_v 22.9 |

Примечание: `islands_grid` в `perf_probe` стартует с 0 settle (в отличие от `solver_bench`),
поэтому 3.46 ms ≠ 1.92 µs. Solver остаётся доминантой для `many_islands`/`hetero` (~80% времени).

## `probe_100k` — tiled floor, 10k dynamic (10400 с плитками), 5 шагов (2026-09-02, M1 release)

| Backend | Step 0 | Step 1 | Step 2 | Step 3 | Step 4 | Mean steady (4/5) |
|---|---:|---:|---:|---:|---:|---:|
| SweepAndPrune | 736.11 ms | 757.28 ms | 365.52 ms | 729.99 ms | 566.72 ms | **604.88 ms** |
| UniformGrid 4.0 | 193.12 ms | 193.87 ms | 163.34 ms | 179.23 ms | 160.82 ms | **174.31 ms** |

Диагностика `probe_100k` (после шага 0, для 10k):

| Backend | Cells | Raw pair tests | Static skips | AABB rejects | Candidates |
|---|---:|---:|---:|---:|---:|
| SweepAndPrune (step 0) | 0 | 744 400 | 11 400 | 718 839 | 14 161 |
| UniformGrid 4.0 (step 0) | 5408 | 450 472 | 12 096 | 347 168 | 14 161 |

На steady-state Grid 4.0 ~3.5× быстрее SAP на этом workload. Оба далеки от 16.7 ms —
10k tiled ещё не real-time без p1/p2+settle; 100k tiled (104096 тел) в `probe_100k` ранее показывал
~8.02 s (Grid 8.0) vs ~79.5 s (SAP) steady, см. `perf-baseline-2026-08-27.md` §100k probes.

## Новый API — box↔capsule + analytic TOI (G6, 2026-09-02)

- `box_vs_capsule` — тонкий wrapper поверх точной `distance::shape_distance` (OBB vs capsule core),
  `dist <= margin` → speculative contact, zero-alloc, обе ориентации (`Box+Capsule` и `Capsule+Box` с флипом normal).
- Analytic swept-volume TOI: `first_angular_overlap_fraction` переписан с 5° семплинга на
  conservative advancement с exact distance + bound `|disp| + r_max·|angle|`, до 32 итераций + 10 шагов бинарного
  рефайна, гарантия отсутствия туннелирования для любой тонкой стенки при любом угле (в т.ч. капсула).
- Покрытие: `crates/physics/tests/box_capsule_toi.rs` — 14 интеграционных тестов
  (6 box↔capsule контакт, 5 TOI быстрым вращением включая тонкую стенку 0.04 и комбинированный sweep,
  3 регрессии — swapped order, resting, multi-body). `cargo test -p ornis-physics` зелёный.

## Выводы

- p1 (incremental broadphase) перевёл UniformGrid 10k из ~0.8–1.1 s в ~10.1 ms (проходной для 16.7 ms).
- p2 (narrow cache, first substep only + fast-miss bypass) дал ещё ~9.2 ms, но текущий Criterion на M1
  показывает 22.4 ms для Grid 4.0 — расхождение ~2.4× требует повторного прогона с фиксированным `--sample-size`
  и проверкой влияния `StepTiming`/кеша на маленьких сценах (`big_stack` регресс +36% острова,
  +356% стека — проверить `solve_normal_block`/`apply_impulse`/`mul_inv_inertia` пути).
- Solver по-прежнему доминанта для dense/many-islands (>80%), broadphase ~0.7 ms даже на 1024 телах.
- 100k tiled остаётся вне real-time (~174 ms для 10k, ~8 s для 100k на Grid), следующий шаг — профилирование
  solver/narrowphase и проверка sparse/heterogeneous сцен перед сменой default с SAP на Grid.
