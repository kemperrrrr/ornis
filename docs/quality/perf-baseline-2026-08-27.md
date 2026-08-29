# Performance baseline — 2026-08-27

Зафиксированный baseline производительности (PROJECT_REVIEW п.4).
Числа — медианные оценки criterion (среднее из `[low mid high]`),
один прогон; это точка отсчёта для сравнения будущих изменений,
а не статистически строгий эксперимент.

## Конфигурация

- Машина: Apple M1, 16 ГБ RAM, macOS 26.5.2
- Toolchain: rustc 1.97.0 (2d8144b78 2026-07-07)
- Профиль: `release` (criterion, `cargo bench`)
- Коммит: `2465570` + незакоммиченные изменения бенчей п.4

## Воспроизведение

### Локально

```bash
cargo bench -p ornis-core        # ECS storage
cargo bench -p ornis-physics     # physics step (solver_bench)
cargo bench -p ornis-render      # frame-plan layout + запись пассов
cargo bench -p ornis-materialx   # MaterialX parse/convert
```

### На GitHub Actions

Отдельный performance workflow (`.github/workflows/performance.yml`):

- запускается вручную через `workflow_dispatch`;
- выполняет `cargo bench -p ornis-physics --bench solver_bench`;
- сохраняет результаты Criterion в артефакты;
- публикует краткую сводку в job summary;
- не влияет на основной Quality gate.

Для 100k тел зонд запускается отдельно:

```bash
cargo run -p ornis-physics --release --example probe_100k
```

## ECS storage (`crates/core`)

`store_bench` — 100k компонентов `Position` (12 байт) в `ComponentStore`:

| Сценарий | Время |
|---|---|
| insert_100k (амортизированная вставка) | 1.07 µs |
| iterate_100k | 36.6 µs |
| par_iterate_100k (rayon) | 89.3 µs |
| random_access_100k (10k выборок) | 144.2 µs |
| intersection_100k (битсет двух лент) | 357.4 µs |

`comparison_bench` — 10k сущностей, сравнение с эталонными структурами
(hybrid = `ComponentStore`):

| Сценарий | hybrid | pure_sparse | hashmap | archetype |
|---|---|---|---|---|
| insert (10k) | 119.7 µs | 116.5 µs | 1.035 ms | 141.5 µs |
| iterate | 6.70 µs | 3.55 µs | 28.9 µs | 7.04 µs |
| random_access (10k) | 25.1 µs | 20.2 µs | 467.8 µs | 20.8 µs |
| memory (на сущность) | 455 ps | 597 ps | 448 ps | 1.22 ns |

## Physics step (`crates/physics`, `solver_bench`)

Один `step(1/60)` в установившемся состоянии (сцена «успокоена»
30–60 шагами перед замером):

| Сценарий | Время | Комментарий |
|---|---|---|
| islands_grid_16x16 (1024 тела, 256 островов) | 1.19 µs | острова спят — steady state почти бесплатен |
| big_stack_32 (32 тела, 1 остров) | 2.06 ms | стек не засыпает, солвер работает каждый шаг |
| deep_stack_128 (128 тел, 1 остров) | 15.1 ms | то же, рост ~линейно-сверхлинейный по высоте стека |
| physics_bodies/1000 (сетка на тайловом полу) | 1.56 µs | тела спят |
| physics_bodies/10000 (сетка на тайловом полу) | 767 ms | тела не успевают уснуть за 30 шагов — бодрствующий шаг |
| physics_bodies/100000 | **не измерим criterion** — см. находку ниже |

### Exploratory comparison: Sweep-and-Prune vs UniformGrid (2026-08-28)

Ниже зафиксирован результат отдельного performance workflow по логу,
предоставленному пользователем. Workflow run: `33194136814`, head
`4fa10c0813f264d9df7c1b1d66002297ea9c5d28`, запуск 2026-08-28 на ветке
`arena/01a043d9-ornis`. CPU/runner metadata и версия `rustc` в raw log не
сохранены, поэтому это **exploratory comparison**, а не замена
воспроизводимому Apple M1 baseline выше.

Команда:

```bash
cargo bench -p ornis-physics --bench solver_bench -- --verbose
```

Центральная оценка Criterion (`time: [low estimate high]`):

| Сценарий | Sweep-and-Prune | UniformGrid | Сравнение |
|---|---:|---:|---|
| `physics_bodies/1000` | 1.4029 µs | 1.4075 µs | разница около +0.3% для UniformGrid, в пределах шума |
| `physics_bodies/10000` | 1.1167 s | 288.58 ms | UniformGrid примерно в 3.87 раза быстрее, −74.2% |

На 1k тел измерения практически равны. На 10k тел UniformGrid
подтверждает исходную гипотезу для tiled-floor workload, но абсолютное
время остаётся около 289 ms и всё ещё далеко от бюджета real-time кадра
16.7 ms. В 10k-прогоне было всего 10 samples, Criterion предупредил о
длительном сборе, а Sweep-and-Prune получил один outlier; относительный
вывод нужно повторить с большим числом samples.

`Gnuplot not found` не влияет на измерения: Criterion использовал Plotters
backend. Этот прогон не включал GPU physics — benchmark использовал обычный
CPU path без `--features gpu` и без подключения `WgpuContactSolver`.

**Промежуточное решение:** UniformGrid — сильный provisional candidate для
текущей CPU-сцены, но Sweep-and-Prune остаётся default до профилирования
candidate pairs/solver и отдельного прогона 100k на обоих backend'ах.
Следующий benchmark-код теперь печатает `BroadPhaseStats` для обоих
backend'ов и сравнивает grid cell size 1.0/2.0/4.0; эти счётчики помогут
отделить лишние candidate checks от solver cost, но не заменяют отдельный
timing breakdown.

### Cell-size follow-up (2026-08-29)

Успешный performance workflow run `33235046208` на head
`8183a76c462367b0783c71c362c92dfca7689f6a` завершил сравнение трёх размеров
UniformGrid. Runner/CPU metadata в raw log отсутствуют; значения ниже —
exploratory данные для выбора конфигурации, а не новый machine-normalized
baseline.

Центральная оценка Criterion (`time: [low estimate high]`):

| Сценарий | Sweep-and-Prune | Grid 1.0 | Grid 2.0 | Grid 4.0 |
|---|---:|---:|---:|---:|
| 1k тел | 1.5264 µs | 1.5268 µs | 1.5268 µs | 1.5273 µs |
| 10k тел | 1.0931 s | 542.51 ms | 273.32 ms | 198.45 ms |

На 10k тел `cell_size = 4.0` — лучший из проверенных вариантов: примерно
в 5.51 раза быстрее Sweep-and-Prune и на 27.4% быстрее `cell_size = 2.0`.
`cell_size = 1.0` оказался хуже из-за bookkeeping: 123052 ячейки против
20808 у `2.0` и 5408 у `4.0`. На 1k все варианты находятся в пределах шума.

Диагностика на 10k тел (10400 тел вместе с 400 tiles пола):

| Backend | Cells | Raw pair tests | Static skips | AABB rejects | Candidates |
|---|---:|---:|---:|---:|---:|
| Sweep-and-Prune | 0 | 599346 | 11400 | 576165 | 11781 |
| UniformGrid 1.0 | 123052 | 176672 | 63384 | 0 | 14161 |
| UniformGrid 2.0 | 20808 | 297812 | 27056 | 157468 | 14161 |
| UniformGrid 4.0 | 5408 | 298024 | 12096 | 222560 | 14161 |

Grid сокращает raw pair checks примерно вдвое, но оставляет примерно на
20.2% больше candidate pairs, чем Sweep-and-Prune. Значит, следующий шаг —
отделить timing broadphase от solver и проверить, не является ли часть
оставшихся 198.45 ms solver/narrowphase cost.

Criterion сообщал об отсутствии `base/sample.json` для чистого runner;
workflow всё равно завершился успешно, и текущие измерения были получены.
GPU physics в этом прогоне не участвовала.

### Расширенный cell-size follow-up (2026-08-29)

Workflow run `33240643444` на head
`7504d9bbe2b4d75fecb52efd14784f4aac2fdbd4` был остановлен общим лимитом job
в 60 минут (`cancelled`), но присланный benchmark output содержит измерения
всех шести конфигураций. Runner/CPU metadata в raw log отсутствуют, поэтому
это exploratory данные, а не machine-normalized baseline.

На 10k тел фактический body count равен 10400 (10000 dynamic + 400 floor
tiles). Центральные оценки Criterion:

| Backend | 1k тел | 10k тел |
|---|---:|---:|
| Sweep-and-Prune | 1.4015 µs | 1.1130 s |
| UniformGrid 1.0 | 1.4208 µs | 469.99 ms |
| UniformGrid 2.0 | 1.3809 µs | 271.36 ms |
| UniformGrid 4.0 | 1.4329 µs | 196.69 ms |
| UniformGrid 8.0 | 1.4541 µs | **180.00 ms** |
| UniformGrid 16.0 | 1.4140 µs | 198.59 ms |

`cell_size = 8.0` — лучший из проверенных вариантов: примерно 6.18x быстрее
SAP и на 8.5% быстрее `4.0`. Увеличение до `16.0` уже ухудшает результат на
10.3% относительно `8.0`. На 1k различия находятся в пределах шума.

Диагностика на 10k:

| Backend | Cells | Raw pair tests | Static skips | AABB rejects | Candidates |
|---|---:|---:|---:|---:|---:|
| Sweep-and-Prune | 0 | 599346 | 11400 | 576165 | 11781 |
| UniformGrid 1.0 | 123052 | 176672 | 63384 | 0 | 14161 |
| UniformGrid 2.0 | 20808 | 297812 | 27056 | 157468 | 14161 |
| UniformGrid 4.0 | 5408 | 298024 | 12096 | 222560 | 14161 |
| UniformGrid 8.0 | 1352 | 487298 | 7104 | 435792 | 14161 |
| UniformGrid 16.0 | 392 | 1159602 | 8704 | 1114448 | 14161 |

Это показывает, что минимальное число cells не равно минимальному wall-clock:
`8.0` обслуживает в 4 раза меньше cells, чем `4.0`, и выдерживает рост raw
pair tests; `16.0` уже создаёт примерно в 3.9 раза больше raw pair tests, чем
`8.0`. Все Grid варианты сохраняют одинаковый набор `14161` candidate pairs,
но он примерно на 20.2% больше SAP (`11781`).

`physics_bodies` измеряет полный `step`, поэтому результат не отделяет
broadphase от narrowphase/solver. `cell_size = 8.0` — текущий provisional
лучший tuning choice для этой сцены, но не production default: 100k и timing
breakdown ещё впереди. `probe_100k` поддерживает, например:

```text
cargo run -p ornis-physics --release --example probe_100k -- --sweep
cargo run -p ornis-physics --release --example probe_100k -- --grid --cell-size 8
cargo run -p ornis-physics --release --example probe_100k -- --grid --cell-size 16
```

Adaptive grid пока не объявляется следующим default. После timing breakdown
broadphase/narrowphase/solver можно реализовать редкую cost-based настройку с
hysteresis и ограниченным набором кандидатов; пересчитывать cell size каждый
кадр было бы слишком дорогим и может вызвать oscillation. После сверки с
Box3D и Jolt основной следующий архитектурный кандидат — persistent dynamic
AABB tree с moved/active-body queries, а не ещё один full-rebuild grid; разбор
ссылок находится в [`broadphase-reference-2026-08-29.md`](broadphase-reference-2026-08-29.md).

### Находка: сверхлинейный рост step на 100k тел

Ручной зонд (`crates/physics/examples/probe_100k.rs`, `step` с попешаговым
таймингом, тот же сценарий сетки; запуск: `cargo run -p ornis-physics
--release --example probe_100k`):

- Единый пол-AABB на всю сцену: **~45–56 с/шаг** — Sweep-and-Prune
  вырождается в O(n²), т.к. гигантский AABB перекрывает все тела и
  sweep держит весь набор в активном списке.
- Тайловый пол (тайлы 10×10): **80–110 с/шаг и растёт** — значит,
  квадратичная составляющая есть не только в giant-AABB случае.
  Линейная экстраполяция от 10k (767 ms) дала бы ~8 с; наблюдаемое
  в ~10 раз хуже.

Вывод: масштаб 100k тел сейчас непрактичен для реального времени;
это вход для работ по physics (PROJECT_REVIEW п.2/п.3). В criterion-бенче
100k намеренно отсутствует (см. комментарий в `solver_bench.rs`).

## Render (`crates/render`)

`layout_bench` — стоимость плана кадра (frame plan) на кадр:

| Сценарий | Время |
|---|---|
| layout/compute: forward (7 пассов) | 5.89 µs |
| layout/compute: deferred (8 пассов) | 5.33 µs |
| layout/compute: hybrid (9 пассов) | 12.1 µs |
| layout/cache_hit (steady-state кадр) | 4.4–4.9 ns |
| layout/levels (уровни параллельности) | 191–241 ns |

`recording_bench` — запись пассов в команды wgpu (цель 256×256, сцена
сфер): sequential 1.00 ms, parallel 1.28 ms (parallel на этом сценарии
не окупается — overhead на дочерние задачи; полезен на тяжёлых кадрах).

## MaterialX (`crates/materialx`, `parse_bench`)

| Сценарий | Время |
|---|---|
| parse: документ на 1000 constant-узлов | 609 µs |
| convert: math-chain на 100 узлов → `OpenPBRMaterial` | 185 µs |
