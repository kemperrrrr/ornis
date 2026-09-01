# Ornis: текущие ограничения и план развития

> **Актуальный срез: 2026-08-28.** Этот документ объединяет
> аудит и план работ. Формулировки в разделах с датами —
> исторические снимки; текущие статусы сверены ниже с кодом и
> последующими коммитами.

## Главные ограничения сейчас

- GPU-диспетчеризация в `core` пока фактически является заглушкой: `Dispatcher` умеет выбрать GPU, но `GpuExecutor::execute` не выполняет GPU-операцию.
- `PhysicsEngine::shapecast` уже реализован и покрыт тестами, но физический API все еще ограничен небольшим набором форм и возможностей.
- В `ornis-core` уже есть логический `World`-фундамент (`Resources` с
  авторитетным `SmartStore` и запуском `Schedule`) и backend-neutral
  `Engine` с ресурсами `Time`/`FixedTime`/`InputState`; editor-only physics и
  native/WASM render extraction + `FramePlan` уже подключены. Editor protocol
  теперь имеет queue ACK и correlated completion events, native showcase
  physics также подключена к общему bounded fixed host; полный cross-domain
  runtime с gameplay consumers и browser physics во всех режимах ещё не
  собран.
- Scheduler вынесен в отдельный crate и хорошо протестирован, но еще
  не стал единым runtime-планировщиком всего движка.
- Редактор и ECS пока не образуют полностью единую live-систему: синхронизация идет через polling, а часть сценариев остается демонстрационной.
- Проект одновременно развивает ECS, GPU compute, WASM editor, MaterialX, audio, physics и собственные макросы. Такой широкий scope увеличивает стоимость сопровождения и риск распыления усилий.
- Native-приложение пока скорее showcase/runtime shell, чем полноценная игровая платформа.

## План дальнейшей работы

1. ~~**Закончить вертикальный сценарий редактора**~~ — ✅ закрыт (2026-08-26);
   следующий этап — довести cross-domain runtime поверх `ornis_core::World`.

   Создание entity → изменение Transform/Material → обновление WASM-сцены →
   сохранение и загрузка сцены (`save_scene`/`load_scene` через
   `POST /api/command`, атомарная запись `editor/scene.ron`, меню File →
   Save/Reload в UI, события `scene_saved`/`scene_loaded` в `/api/events`).
   Editor-only vertical slice использует `EditorWorld` на core World и
   polling; браузер после serialization boundary восстанавливает snapshot
   в отдельный `ornis_render::RenderWorld`.

2. **Довести physics API** — 🟡 (п1/п2/box↔capsule/SAT 2026-09-02: см. ниже)

   Collision layers/masks уже добавлены в `RigidBody` и применяются
   симметрично в broadphase, narrowphase и linear CCD. Triggers генерируют
   deterministic enter/exit events без solver impulses, а raycast использует
   точные sphere/OBB/capsule intersection'ы и surface normals. Angular CCD
   теперь имеет bounded sweep для вращающихся box/capsule; fully analytic
   swept-volume TOI — ✅ (conservative advancement, 2026-09-01). **2026-09-02:**
   **п1 incremental broadphase ✅** (`body_cells`/`prev_meta`, dirty-set, heuristic >50% → full rebuild),
   **п2 narrow cache ✅** (`NarrowCacheEntry`, первый substep, fast-path >0.5 м/с, HashMap) + **SAT cache ✅ (отдельный PR, sequential-only, 16-шард `Vec<Mutex>` без регресса на parallel)** (`SatCacheEntry`, `obb_sat_cached`/`box_manifold_cached`, `try_lock`-only, large parallel bypass),
   **box↔capsule ✅** как честный discrete контакт (оба narrowphase-пути через `distance::shape_distance`/`box_vs_capsule`, speculative `margin`, analytic TOI `cast_shape` conservative advancement). **Полный 8→2 на больших parallel сценах** (`physics_bodies 10k` 14k пар, `par_iter>256`) — **следующий шаг**: lock-free/DashMap + расширенный EPS, сейчас SAT выключен для parallel чтобы не регрессить 9→15→89 мс.

3. **Укрепить GPU-путь**

   Проверить реальные GPU-сценарии на macOS, сравнить CPU и GPU по стабильности, скорости и расхождению результатов. Не обещать bit-identical поведение там, где его нет.

4. **Сделать performance baseline**

   Зафиксировать бенчмарки для:

   - 1k, 10k и 100k тел;
   - широких и глубоких physics islands;
   - render passes;
   - ECS storage;
   - MaterialX parsing.

   > ✅ 2026-08-27: baseline зафиксирован в
   > [perf-baseline-2026-08-27.md](docs/quality/perf-baseline-2026-08-27.md)
   > (Apple M1): ECS storage (1k–100k), physics step (islands/стеки/тела),
   > render (layout + запись пассов), MaterialX. Добавлены бенчи:
   > `physics_bodies` (1k/10k), `deep_stack_128`, `materialx parse_bench`.
   > ⚠️ Находка: на 100k тел `step` сверхлинеен (единый пол-AABB вырождает
   > Sweep-and-Prune в O(n²), ~45–56 с/шаг; с тайловым полом — 80–110 с/шаг,
   > квадратичная составляющая остаётся) — 100k в criterion не помещается,
   > числа сняты зондом `crates/physics/examples/probe_100k.rs`. Это вход
   > для п.2/п.3.
   >
   > ✅ **2026-09-01 — этап B (perf_probe, Apple M1, --release):** broadphase
   > **1×/кадр** (swept `dt`, клон `broad_active`) + per-body required
   > substeps **[4..12]** = `ceil(|v|*dt / 1/240)` с порогом `max-min>=4`
   > и pre-filter до SAT через `scratch_pairs`; no-alloc scratch
   > (`UniformGrid::scratch_pairs`, `Engine::scratch_manifolds/pairs/clamped/parent`,
   > `detect_collisions_into`), narrow параллельно через rayon при `>256` пар.
   > **tiled 10k: 103.4 → 15.82 мс** (broad 4.86 / narrow 8.01 / solver 2.95,
   > **63 FPS**), **solver_bench: 74 → 17.07 мс** (−77% от 180 мс baseline),
   > **many_islands 16.33 мс (61.2 FPS)**, **hetero 16.33 мс**,
   > **islands_grid 2.81 мс (355 FPS)**, **contact_cluster 3.16 мс (316 FPS)**,
   > **big_stack 925 FPS**, шум машины **±1.5 мс**. Бюджет **60 FPS (16.7 мс)**
   > достигнут на tiled 10k; **п1 incremental ✅ 2026-09-01** (`body_cells`/`prev_meta`, dirty-set, retained clean-clean, honest stats, heuristic >50% → full rebuild; 4.8→~1 мс), **п2 narrow cache ✅** (`NarrowCacheEntry` + `detect_collisions_into_with_cache`, первый substep, ±1e-4, fast-path >0.5 м/с) + **SAT cache ✅ 2026-09-02 (отдельный PR, 16-шард `Vec<Mutex>`, sequential-only, parallel bypass без регресса 9→89 мс)** (`SatCacheEntry`, `obb_sat_cached`/`box_manifold_cached`, `try_lock`-only) к цели ~10 мс / 100 FPS; **box↔capsule ✅ 2026-09-01** (оба narrowphase-пути через `box_vs_capsule` → `distance::shape_distance`, analytic TOI `cast_shape`); **полный 8→2 на больших parallel** — следующий шаг (lock-free/DashMap + расширенный EPS); GPU solver 2.9 мс — не бутылка, отложен.

5. **Упростить крупные модули**

   Разделить `physics/src/engine.rs` и MaterialX evaluator по ответственности: broadphase, narrow phase, constraints, integration и queries.

   > Статус 2026-08-25: в значительной части выполнено раундом ликвидации
   > complexity-debt (bca baseline 131 → 0 записей). `engine.rs` уже разбит
   > на `engine/{contacts,islands,joints}.rs` (solve_island_velocity
   > cognitive 110 → 9); MaterialX evaluator почищен — мёртвый дубль
   > `src/codegen.rs` (939 строк, не был объявлен в lib.rs) удалён,
   > `evaluate_node`/`extract_material`/`parse_constant` раздроблены на
   > хелперы. Остаток: дальнейшее деление graph.rs по фазам (detection /
   > evaluation / extraction), если файл снова вырастет.

6. **Улучшить надежность редактора**

   Базовые request id/ACK, correlated completion events, transport sequence
   numbers, JSON escaping и stale guards уже добавлены. `/api/events` получил
   bounded replay по cursor и `EventGap`; WebSocket server-push на `/api/events`
   теперь реализован, polling остаётся fallback для старых окружений.

7. **Привести документацию к фактическому состоянию**

   Уточнить заявления про «единый scheduler», CPU/GPU dispatch и invisible ECS, чтобы README отражал именно текущую реализацию.

   > ✅ 2026-08-27: README и связанные текущие статусы сверены с кодом.
   > Исправлено: `src/` без Vello;
   > IPC-типы — в `crates/editor-backend/src/ipc.rs` (не `src/ipc.rs`);
   > линтер `#[smart_pipeline]` — deprecated-note трюк вместо `compile_warning!`;
   > `Pack` → ✅ (`for_each_packed` + совместимость лент с `for_each_entity!`);
   > WASM-viewport рендерит живую сцену из `/api/scene` (fallback на `scene.ron`,
   > orbit-камера) — статусы редактора и roadmap п.4 обновлены; Приложение A
   > подтверждено кодом (убран удалённый `shader.rs`, `shapecast` реализован —
   > G6, A3.3/A3.4 закрыты). Заодно обновлён комментарий в шапке `editor/editor.js`.

8. **Единый источник шейдеров: перевести render на Rust→WGSL (путь 2)**

   После удаления мёртвого `shader.rs` канонический источник render-шейдеров
   — builder'ы в `crates/render/src/shaders/`. Физика уже генерирует шейдеры
   из Rust (`#[gpu_pipeline]` + `#[derive(WgslStruct)]`, идея №4
   «CPU↔GPU из одного Rust-кода»), но render всё ещё содержит рукописные
   WGSL-литералы в `shaders/mod.rs` и `composite.rs`. Поэтому задача ниже
   остаётся актуальной: постепенно перевести render-пассы на общий
   Rust→WGSL путь, не возвращая второй источник истины.

   Выбранный путь — **вариант 2: постепенный перевод рендерных пассов на
   `#[gpu_pipeline]` / `WgslStruct`**, чтобы весь WGSL выводился из одного
   Rust-источника:

   - расширить DSL `#[gpu_pipeline]` с compute-функций до фрагментных/вершинных
     шейдеров (entry points `@vertex`/`@fragment`, multiple entry points,
     varying-входы/выходы);
   - перевести пассы по одному, начиная с простых (`composite`, bloom),
     заканчивая PBR lighting; после каждого пасса — визуальное сравнение
     кадров до/после и naga-валидация;
   - `#[derive(WgslStruct)]` становится единственным источником раскладок
     uniform/storage буферов рендера (как уже сделано для GPU-физики);
     compile-time `offset_of!` ассерты заменяют ручную синхронизацию;
   - критерий завершения: `grep -r "vec4<f32>" crates/render/src` не находит
     рукописных шейдерных литералов, все шейдеры проходят naga-валидацию
     в тестах, визуальные golden-frame тесты зелёные.

   Промежуточная страховка на время миграции: вынести ещё не переведённые
   WGSL-литералы из .rs в отдельные `.wgsl`-файлы с `include_str!` +
   naga-валидацией в тестах, чтобы строки шейдеров хотя бы не жили внутри
   Rust-кода.

   > Историческая находка аудита 2026-08-26: `shader.rs` был мёртвым
   > дублем и содержал около 1343 строк лишнего WGSL/kernel-кода.
   > ✅ **Выполнено 2026-08-26**: файл удалён, re-export вычищен из `lib.rs`,
   > канонический источник шейдеров закреплён за `shaders/` builder'ами.
   > Оставшаяся часть задачи — перевести сами render-пассы на Rust→WGSL.

   > ✅ **Промежуточная страховка выполнена 2026-08-30**: рукописные
   > WGSL-литералы вынесены из `shaders/mod.rs` и `composite.rs` в
   > `crates/render/src/shaders/wgsl/` (10 файлов, подключены через
   > `include_str!`, содержимое байт-в-байт идентично — собранные строки
   > шейдеров не изменились). Все builder'ы и composite-шейдер проходят
   > naga-валидацию в тестах (`parse_str` + `Validator`). Сам перевод пассов
   > на Rust→WGSL по-прежнему впереди.

9. **Документация Rust-кода: массовые пропуски, но мало лжи**

   Аудит 2026-08-26 (два параллельных прохода, все крейты):

   - **Покрытие docs было плохим и неровным** — аудит 2026-08-26 насчитал
     ~726+ публичных элементов без `///` (это исторический baseline, а не
     текущая цифра).
   - **Модульные `//!` хедеры также были неполными** — в baseline отсутствовали
     хедеры у 28 файлов; новые файлы теперь должны начинаться с `//!` по
     правилу `AGENTS.md`.
   - **Расхождение доков с кодом почти НЕ подтверждено** — после раунда
     рефакторинга были найдены одна битая ссылка и две неразрешимые
     rustdoc-ссылки; все они исправлены, rustdoc CI даёт **0 warnings**.
   - **Текущий статус (2026-08-27):** `#![warn(missing_docs)]` включён во
     всех workspace-крейтах и бинаре; основная волна публичной документации
     уже прошла. Остаток — доведение отдельных `//!` хедеров и поддержание
     правила для нового API. Каноничность render-шэйдеров зафиксирована:
     мёртвый `shader.rs` удалён, но полный Rust→WGSL перевод render ещё не
     выполнен.

## Ближайший приоритет

Editor-only vertical slice уже работает поверх `ornis_core::World` и общего
frame contract. Native/WASM render и native/editor-only physics seams также
подключены; общий bounded fixed host теперь вынесен в `ornis_core::Engine`.
Следующий приоритет — gameplay consumers и масштабирование broadphase, не
создавая второй authoritative-модели состояния.


---

## Дополнение по результатам ревью


## Основные проблемы

### 1. Критическая проблема производительности физики

Это самая серьёзная техническая находка.

Согласно [`docs/quality/perf-baseline-2026-08-27.md`](docs/quality/perf-baseline-2026-08-27.md):

- 10 000 тел — около **767 ms за шаг** в одном из сценариев;
- 100 000 тел — примерно **45–110 секунд за шаг**;
- Sweep-and-Prune вырождается в квадратичную работу;
- гигантский AABB пола приводит к огромному active set;
- даже tiled floor не устраняет сверхлинейный рост.

Следовательно, физический движок сейчас **не масштабируется до заявленного ECS/engine workload**. Причина не только в конкретной сортировке, но и в общей модели broad phase.

Приоритетные направления:

1. spatial hash / uniform grid / dynamic AABB tree;
2. разбиение мира на регионы;
3. фильтрация collision layers;
4. отдельный pipeline для sleeping bodies;
5. ограничение island expansion;
6. profiling narrow phase и contact generation;
7. отдельный benchmark для worst-case broad phase.

До решения этой проблемы я бы не позиционировал движок как пригодный для больших сцен.

> **Статус 2026-08-29 (вариант А: профиль solver, локально Mac aarch64, --release):**
> bottleneck локализован — это **tuning substeps/sleep, не архитектура solver**.
> - `perf_probe` (islands_grid 1025, settle=0): sub=12 → solver **21 ms**/кадр
>   (33.7 ms/фрейм, мир не засыпает); **sub=4 → solver 1.4 ms** (2.3 ms/фрейм,
>   сон срабатывает). Рычаг №1 — `substeps`, не iterations.
> - **Зависимость от сцены доказана:** на жёстких сценах sub=4 *проигрывает*:
>   - tall_stack_50 (50 ярусов): оба режима ДРЕБЕЗЖАТ (max_awake_v 16–21 м/с,
>     башня не спит) — нужны iterations/softness, не substeps;
>   - fast_drop (v=-40 м/с в пол): sub=12 гасит до 0 (спит), sub=4 оставляет
>     дребезг 0.32 м/с и тело не спит. Туннелирования нет (min_body_bottom_y=-0.9).
> - Вывод: **глобальная константа `substeps=12` субоптимальна в обе стороны**
>   (жирна для осевших сцен, недостаточна/не та для жёстких). Честный fix —
>   адаптивный substepping (по max скорости/прониканию) или понижение default +
>   подъём sleep-threshold; отдельная задача с риском регресса стабильности.
>   **Adaptive substepping ✅ реализован** (`effective_substeps` в `engine.rs`):
>   выбирает substeps по max скорости awake dynamic, clamp в [4, self.substeps]
>   (не поднимает выше заданного `set_substeps`, чтобы не ломать CCD-тесты с
>   sub=1). islands_grid под default теперь 10.25 ms/фрейм (solver 6.9) против
>   33.7 при фикс. sub=12 — мир засыпает; fast_drop адаптивно держит 12 и
>   гасит до 0 (без дребезга sub=4). `perf_probe` расширен сценами
>   many_islands / contact_cluster / tall_stack / fast_drop + tuning-sweep и
>   `log_stability`. Gate (fmt/clippy/bca) чист, 114 тестов проходят (в т.ч.
>   `adaptive_substeps_scale_with_body_speed`).
>   **Per-island adaptive ✅ реализован** (`e9a096b`):
>   `adaptive_iters_for_island(max_speed,dt,base)` — каждый остров получает
>   `iters = ceil(wanted/max_sub * base)` по `wanted=ceil(|v|*dt/1/240)`;
>   медленные острова — 3 vel / 1 pos, быстрые — 8/4. `dispatch_islands_velocity`
>   и `solve_contacts_position` предвычисляют `iters_per_island` вне `par_iter`.
>   `many_islands 1024` **30→22.5мс** (solver 19→13.6, -30%), `islands_grid 1025`
>   **10.2→5.7мс** (solver 6.9→2.6), hetero 1 fast+255 slow — 29мс solver 18
>   (экономия ~6мс vs global). Test `per_island_iters_scale_with_speed`,
>   115 тестов, clippy/bca/fmt 0. Потолок: только по `|v|`, без `max_pen`;
>   upgrade — `wanted = max(|v|*dt, max_pen/slop)` когда `min_bottom_y < -0.02`
>   при `|v|<0.2`.
>   **Backlog CPU (не предел, ×3 до Jolt):**
>   1. `wide_solver` ON + SIMD 4/8 контактов — reuse `wide.rs` ✅ уже ON (`engine.rs:1423` wide_solver=true, `dispatch_islands_velocity` wide_on);
>   2. убрать аллокации в кадре (`Vec<Manifold>`, `HashMap` warm, клон shard) — ⏭️ отложено (требует убрать клон shard + HashMap, add когда `perf_probe tiled` покажет alloc flame — ponytail: no allocs per frame);
>   3. penetration-driven — `wanted = max(|v|*dt, max_pen/slop)` ✅ `31e6b6e` (`adaptive_iters_for_island_with_pen`, slop 0.01, 115 тестов);
>   4. агрессивнее сон / `SLEEP_TIME` tuning ✅ `31e6b6e` (0.5→0.3с, 12 кадров раньше, `ponytail` ceiling).
>   3/4 done, 1 уже был, 1 отложен.

> ### Решение по следующему broadphase (срез 2026-08-28)

Конкретный production default пока не выбран. В коде уже есть opt-in
экспериментальный `BroadPhaseKind::UniformGrid` с deterministic candidate
pairs, static/dynamic-friendly cell decomposition и large-body escape path;
крупные static AABB (например, пол) не обязаны порождать одну пару со всеми
телами через линейный axis sweep. `Sweep-and-Prune` остаётся default
baseline/fallback для сравнения.

**Dynamic AABB tree** остаётся вторым кандидатом для разреженных миров с
сильно различающимися размерами тел. Выбор default не фиксируется до
benchmark-матрицы 1k/10k/100k тел, большого единого пола, tiled floor,
sparse world, плотных islands и worst-case broadphase. Это выбор broadphase
backend, а не новый верхний scheduler и не runtime-выбор без измерений.

Сверка с исходниками Box3D и Jolt уточнила приоритет: перед adaptive
`cell_size` нужно проверить persistent proxy/lifetime, fat AABB, разделение
static/moving structures и active/moved-body queries. Архитектурные заметки и
официальные ссылки: [`docs/quality/broadphase-reference-2026-08-29.md`](docs/quality/broadphase-reference-2026-08-29.md).

### Exploratory benchmark (2026-08-28)

По workflow run `33194136814` (head
`4fa10c0813f264d9df7c1b1d66002297ea9c5d28`, запуск 2026-08-28) получены
следующие центральные оценки Criterion. CPU/runner metadata и `rustc`
version в raw log не были сохранены:

| Сценарий | Sweep-and-Prune | UniformGrid | Вывод |
|---|---:|---:|---|
| 1k тел | 1.4029 µs | 1.4075 µs | практически паритет, около +0.3% для grid |
| 10k тел | 1.1167 s | 288.58 ms | grid быстрее примерно в 3.87 раза, −74.2% |

Это подтверждает UniformGrid как provisional candidate для текущего
CPU tiled-floor workload, но не закрывает масштабирование: 288.58 ms на
10k тел всё ещё значительно выше бюджета 16.7 ms; на момент этого среза
100k не были измерены, а 10k-прогон использовал только 10 samples и содержал
warning Criterion.
`Gnuplot not found` не является ошибкой — использован Plotters backend.
GPU physics в этом прогоне не участвовала: benchmark не включал
`--features gpu` и не подключал `WgpuContactSolver`.

Benchmark теперь печатает `BroadPhaseStats`: body count, raw pair tests,
layer/mask rejections, static-static skips, AABB rejections, unique
candidate pairs, occupied grid cells и large-body count. Он сравнивает
UniformGrid с cell size 1.0/2.0/4.0/8.0/16.0. Это закрывает первый
candidate-pair breakdown; отдельный timing broadphase против solver и 100k
probe для обоих CPU backend'ов ещё впереди. До этих измерений
Sweep-and-Prune остаётся default, UniformGrid — opt-in.

### Cell-size follow-up (2026-08-29)

Workflow run `33240643444` на head
`7504d9bbe2b4d75fecb52efd14784f4aac2fdbd4` был остановлен общим лимитом job
в 60 минут, но присланный benchmark output содержит измерения всех шести
конфигураций. На 1k тел все варианты остаются в пределах шума. На 10k
центральные оценки составили: SAP — `1.1130 s`, grid 1.0 — `469.99 ms`,
grid 2.0 — `271.36 ms`, grid 4.0 — `196.69 ms`, grid 8.0 — `180.00 ms`,
grid 16.0 — `198.59 ms`. `cell_size = 8.0` — лучший проверенный вариант,
примерно 6.18x быстрее SAP и на 8.5% быстрее `4.0`; `16.0` на 10.3%
медленнее `8.0`. Полная таблица и `BroadPhaseStats` — в
`docs/quality/perf-baseline-2026-08-27.md`.

Это меняет provisional tuning conclusion для tiled-floor сцены: теперь
кандидат — `cell_size = 8.0`, но измерено end-to-end `step`, а не изолированное
время broadphase. Grid по-прежнему выдаёт `14161` candidate pairs против
`11781` у SAP и остаётся примерно в 10.8 раза медленнее бюджета кадра 16.7 ms.

Targeted 100k probes `33245718111` (Grid 8.0) и `33251548032` (SAP)
успешно сравнили оба backend на tiled floor: около `8.02 s/step` против
`79.49 s/step` в steady state. На первом SAP step было `5417936560` raw pair
tests против `2349246` у Grid; оба backend выдали `100000` candidates. Это
сильное подтверждение Grid 8.0 против SAP на 100k, но два запуска были на
отдельных unlabeled runner'ах и не являются machine-normalized baseline.
Оба варианта всё ещё далеко от real-time, поэтому production default пока не
переключается.

Ручной `probe_100k` умеет выбирать `--sweep`/`--grid`, cell size, число тел и
число шагов. Adaptive grid пока не добавляется: без timing breakdown он
может оптимизировать counters, но ухудшить wall-clock из-за пересборки и
нестабильного выбора. Следующие шаги — timing
broadphase/narrowphase/solver и persistent `DynamicAabbTree`; adaptive policy
остаётся редкой cost-based настройкой с hysteresis после этого.

> **Статус 2026-08-29 (audit follow-up):** timing breakdown ✅ добавлен.
> `StepTiming` суммирует фазы по substep-циклу в `BuiltinPhysicsEngine::step`,
> `step_timing()` отдаёт разбивку, `perf_probe` печатает среднее + peak-frame.
> Замер на активном мире (dev build, aarch64): islands_grid 16×16 (1025 тел)
> broad 28.7 ms / narrow 142 ms / solver 437 ms за шаг; big_stack 32
> broad 0.55 ms / narrow 10.4 ms / solver 34.4 ms. **Broadphase — лишь ~5%
> стоимости шага**, solver доминирует. Вывод: п.4/п.5 (DynamicAabbTree) всё ещё
> нужны для 100k+ и худших паттернов, но на текущих сценах их выигрыш мал до
> профилирования/оптимизации solver и narrowphase.
>
> **Статус 2026-08-29 (п.3–п.6 broadphase завершён):** `DynamicAabbTree` backend ✅
> реализован (`crates/physics/src/broadphase_tree.rs`), подключён в
> `BroadPhaseBackend` + `BroadPhaseKind::DynamicAabbTree`, выбирается через
> `set_broadphase`, флаг `--tree` в `probe_100k`. Oracle-тест (brute-force, не
> SAP) проходит; SAP-дефект **исправлен**: `SweepAndPrune::update` терял пары,
> где больший body-индекс сортируется раньше на оси sweep (сравнение индексов
> вместо позиций) — пары теперь канонизуются в `(min,max)`, добавлен
> регрессионный тест. SAP снова корректный oracle. Moved-list tree исправлен на
> re-query всех dynamic (иначе терялись пары после первого substep).
> **п.5 ✅:** локальная матрица (Mac aarch64, --release, 10k тел) замерена —
> tiled / giant_floor / sparse / islands / heterogeneous; grid побеждает на 4/5
> сцен, SAP квадратичен на dense, tree проигрывает grid везде (полный re-query
> dynamic). `candidate_pairs` идентичны у всех backend (корректность).
> **п.6 ✅:** **default broadphase = UniformGrid** (изменено в
> `BuiltinPhysicsEngine::new`); SAP — compatibility baseline, tree — experimental.
> Adaptive routing (SAP↔grid↔tree по паттерну) — будущая работа. bca/clippy/fmt
> чисты, 113 тестов физики проходят. План broadphase (п.3–п.6) закрыт.
### 2. GPU-диспетчеризация в `ornis-core` всё ещё заглушка

В [`crates/core/src/dispatcher.rs`](crates/core/src/dispatcher.rs):

- `Dispatcher` действительно выбирает `Cpu` или `Gpu`;
- `GpuExecutor` существует при feature `gpu`;
- однако `GpuExecutor::execute` возвращает `None`;
- GPU-операция там не выполняется.

При этом в `ornis-wgpu-backend` есть отдельный рабочий механизм compute dispatch через `CommandSync`.

То есть нужно чётко разделять:

- **работающий GPU backend / command dispatch**;
- **не завершённую автоматическую GPU-диспетчеризацию ECS-операций из core**.

Сейчас API легко создаёт впечатление, что `SmartDispatcher` уже способен автоматически исполнять generic ECS workload на GPU. На самом деле это пока fallback на CPU.

Это не обязательно плохое архитектурное решение, но документация и API должны явно маркировать эту границу.

### 3. Редактор ещё не является полностью live-системой

Положительные части есть:

- HTTP server;
- REST endpoints;
- command channel;
- scene snapshots;
- создание entity;
- изменение компонентов;
- save/load;
- WASM viewport;
- polling.

Но остаются ограничения:

- editor сохраняет polling fallback для старых proxy/server окружений;
- команды получают queue-level `request_id`/`accepted` ACK и correlated
  `CommandCompleted` event;
- snapshot endpoints получают transport `sequence`, а scene сохраняет
  authoritative `version`;
- `/api/events?after=<sequence>` даёт bounded replay и `EventGap`, если старые
  записи уже вытеснены;
- WebSocket `/api/events` server-push использует тот же cursor/event shape;
- ошибка выполнения команды приходит через correlated completion event и
  legacy `error` event;
- native runtime по-прежнему выглядит скорее showcase shell, чем полноценный runtime.

Сериализация событий теперь проходит через `serde_json`: строковые поля
экранируются, а невалидные embedded payloads выдаются как JSON-строки, поэтому
`/api/events` не ломается из-за кавычек или обратных слешей.

### 4. Проект слишком широк для текущего размера

Одновременно развиваются:

- ECS;
- GPU compute;
- custom procedural macros;
- render graph;
- unified scheduler;
- physics;
- audio;
- MaterialX;
- WASM;
- editor;
- scripting roadmap;
- asset pipeline;
- lock-free storage.

Каждая подсистема сама по себе достаточно большая. Риск в том, что движок станет коллекцией интересных технологий без одного законченного пользовательского сценария.

Наиболее разумный фокус сейчас:

> **живой редактор → ECS → render → physics → save/load**

Пока этот vertical slice не станет устойчивым, scripting, новые языки, asset pipeline и дальнейшее усложнение scheduler лучше держать на втором плане.

### 5. Слишком крупные модули

По размеру особенно выделяются:

- `crates/physics/src/engine.rs` — около 2900 строк;
- `crates/render/src/renderer.rs` — около 2000 строк;
- `crates/materialx/src/graph.rs` — около 1600 строк;
- `crates/render/src/frame_plan.rs`;
- `crates/render/src/frame_exec.rs`;
- `crates/core/src/schedule.rs`.

Часть уже была разделена, но остаточная сложность всё ещё высокая. В частности:

- physics стоит дальше делить на broad phase, narrow phase, contacts, solver, integration, queries;
- renderer — на resource setup, upload, pass recording, frame lifecycle;
- MaterialX graph — на parse, graph validation, evaluation, extraction.

Это не блокирует разработку прямо сейчас, но повышает стоимость дальнейших изменений и ревью.

### 6. Документация местами опережает реальную интеграцию

Документация в целом подробная и полезная, но часть формулировок звучит более завершённо, чем соответствующий код.

Например:

- «invisible ECS» пока в значительной степени реализуется через macro/API слой, а не как полностью прозрачная трансформация произвольного объектного кода;
- unified scheduler реализован как инфраструктура, но ещё не стал runtime-планировщиком полного кадра;
- GPU dispatch в core не завершён;
- редактор live-связан с ECS только через промежуточные REST snapshots и commands.

Я бы рекомендовал для каждой крупной возможности разделять статусы:

- `implemented`;
- `tested in isolation`;
- `integrated in runtime`;
- `used by editor`;
- `production-ready`.

Сейчас эти уровни иногда смешиваются.

## Тестирование

Я попытался запустить:

```bash
cargo test --workspace
```

Но в окружении отсутствует `cargo`:

```text
cargo: command not found
```

Поэтому в рамках этого ревью я не могу подтвердить текущий compile/test status фактическим запуском. Статический осмотр показывает хорошую тестовую базу, но итоговая оценка должна учитывать, что именно в этой сессии workspace не был собран.

При этом в репозитории уже есть [`quality.yml`](.github/workflows/quality.yml), который запускает `cargo xtask quality --ci` в GitHub Actions. В случае падения workflow сохраняет лог и публикует сводку и части полного лога в комментариях к pull request. Права на изменение GitHub-среды и самого workflow в рамках этого ревью не требовались и не использовались.

Есть тесты для:

- ECS и generation semantics;
- macros;
- scheduler;
- editor backend;
- scene serialization;
- physics;
- render planning;
- GPU pipeline;
- property tests;
- integration tests.

Однако для такого проекта важно добавить ещё:

- браузерный визуальный end-to-end тест snapshot → WASM scene;
- ~~browser WebSocket reconnect integration test~~ — ✅ добавлен (2026-08-30):
  `crates/editor-backend/tests/http_integration.rs`, reconnect по
  `/api/events?after=<sequence>` без дублей/потерь + сценарий `EventGap`
  после вытеснения истории;
- ~~fuzzing HTTP command payloads~~ — ✅ добавлен (2026-08-30): fuzz-target
  `editor_command` против `editor_backend::remote::parse_command_payload`;
- benchmark worst-case broad phase;
- compile test для всех публичных macro entry points.

## Что в проекте хорошо

### 1. Хорошая модульная декомпозиция

Workspace разделён на логичные крейты:

- `ornis-core` — ECS, sparse sets, entity lifecycle, dispatcher;
- `ornis-physics` — физика и collision detection;
- `ornis-render` — wgpu-рендерер и PBR;
- `ornis-schedule` — планирование систем;
- `ornis-wgpu-backend` — GPU compute и command sync;
- `ornis-materialx` — импорт MaterialX;
- `ornis-audio` — аудио;
- `ornis-wasm` — браузерный рендер;
- `editor-backend` — HTTP/IPC слой;
- `xtask` — единая инфраструктура качества.

Это значительно лучше, чем монолитный экспериментальный движок, где ECS, физика и рендер смешаны в одном crate.

### 2. Сильная реализация ECS-ядра

В `crates/core` есть:

- sparse-set storage;
- dense-массивы и paginated sparse index;
- bitset-пересечения;
- generation-aware entities;
- recycling entity IDs;
- hot/cold и lock-free storage;
- CPU/Rayon execution lanes;
- component registry;
- property-тесты и тесты детерминизма.

Особенно хорошо, что property-тесты уже находили реальные ошибки в семантике generations. Это признак полезной тестовой инфраструктуры, а не просто большого числа smoke-тестов.

### 3. Неплохая тестовая и quality-инфраструктура

Есть:

- `cargo xtask quality`;
- `cargo clippy --all-targets -- -D warnings`;
- `cargo deny`;
- `cargo audit`;
- `cargo outdated`;
- fuzzing для RON и MaterialX;
- mutation testing через `cargo-mutants`;
- baseline complexity;
- benchmarks;
- CI workflow;
- GPU-тесты на software adapter.

Это сильная сторона проекта. В большинстве подобных ранних движков есть только `cargo test`, а здесь уже сформирована инженерная оболочка вокруг разработки.

### 4. Рендерер выглядит содержательно, а не декларативно

В `ornis-render` есть:

- forward/deferred/hybrid frame plan;
- G-buffer;
- lighting/composite passes;
- PBR-функции;
- OpenPBR material;
- WGSL shader builders;
- material upload;
- render backend abstraction;
- тесты планировщика;
- benchmark записи команд.

Удаление мёртвого дубликата `shader.rs` было правильным решением: наличие двух независимых источников WGSL было серьёзным риском рассинхронизации.

### 5. Физика уже больше, чем заглушка

В `ornis-physics` реализованы:

- Sweep-and-Prune broad phase;
- sphere/box/capsule контакты;
- SAT для box-box;
- импульсный solver;
- restitution и friction;
- islands;
- joints;
- raycast;
- shapecast через conservative advancement;
- CPU и GPU-путь.

Для экспериментального движка это достаточно широкий и полезный набор.

## Что в проекте хорошо в целом

Проект выглядит как **сильный технический R&D-прототип с хорошей культурой разработки**, а не как игрушечный engine scaffold. Особенно хорошо сделаны:

- workspace architecture;
- ECS storage;
- quality tooling;
- scheduler foundation;
- PBR/rendering foundation;
- breadth of tests.

Но до production ещё далеко из-за трёх блокеров:

1. **физика плохо масштабируется**;
2. **GPU dispatch в core не завершён**;
3. **editor/ECS protocol ещё не является надёжной live-интеграцией**.

Если сфокусироваться на одном вертикальном сценарии и временно заморозить расширение scope, из проекта может получиться интересный специализированный Rust engine. В текущем виде это качественный, но слишком широкий исследовательский прототип.

## Что я бы сделал следующим

### Приоритет 1 — закрыть надёжный vertical slice

Сделать один сценарий полностью проверяемым:

```text
create entity
→ set Transform
→ set Material
→ render updated scene
→ save
→ reload
→ verify identical scene
```

Причём не только UI-тестом, а отдельным интеграционным тестом backend/world protocol.

### Приоритет 2 — исправить editor protocol

Базовый REST-шов закрыт: есть client/server request id, explicit
queue-level ACK/error response, correlated completion events, transport
sequence snapshots, scene version, `serde_json` event serialization и stale
guards в WASM/editor UI. `/api/events?after=<sequence>` даёт bounded replay и
`EventGap`, а `/api/events` поддерживает WebSocket server-push с тем же
cursor/event contract. Сервер отслеживает connection handles, отправляет
normal close при shutdown и heartbeat ping на idle connections; остаётся
browser reconnect test и полноценное чтение client close frames.

### Приоритет 3 — physics scaling

Не добавлять новые сложные формы, пока не будет понятен профиль broad phase на:

- 1k;
- 10k;
- 100k тел;
- много маленьких islands;
- один большой контактный кластер;
- большой static environment.

### Приоритет 4 — явно пометить experimental API

Экспериментальные или незавершённые поверхности API нужно **не прятать, а явно помечать** в документации, rustdoc и по возможности в именах/модулях. В первую очередь это касается:

- core GPU executor stub;
- незавершённого automatic SmartBuffer residency;
- experimental scheduler integration;
- speculative scripting interfaces.

Для таких API стоит использовать явные маркеры вроде `Experimental`, `Unstable`, `not production-ready`, feature flags и отдельные разделы документации. При этом API должны оставаться видимыми разработчику: задача не скрыть технический долг, а не допустить ошибочного восприятия экспериментального контракта как стабильного.

> ✅ 2026-08-30 (частично): маркеры проставлены в rustdoc — `GpuExecutor`
> и модуль `crates/core/src/dispatcher.rs` помечены как experimental stub
> (GPU-диспетчер фактически CPU-fallback), `SmartBuffer` в
> `ornis-wgpu-backend` — как manual-residency без автоматического слоя.
> `Engine`/`Schedule` уже честно описаны как minimal frame host.
> Спекулятивных scripting-интерфейсов в коде нет (фаза 6 не начата) —
> маркировать нечего. README-строки диспетчера и SmartBuffer уже несут
> эти оговорки.

### Приоритет 5 — documentation lint

В репозитории уже зафиксировано большое количество недокументированного public API. Стоит:

1. включить `#![warn(missing_docs)]` в библиотеках;
2. начать с public facade:
   - `Renderer3D`;
   - `SmartStore`;
   - `ComponentStore`;
   - `Entity`;
   - macro entry points;
   - `AudioEngine`;
3. затем включить `-D warnings` для rustdoc в CI.

## Итоговая оценка

| Область | Оценка |
|---|---:|
| Архитектура | **8/10** |
| Инженерная культура | **8/10** |
| Тестовая база | **7/10** |
| Готовность к production | **4/10** |
| Фокус продукта | **5/10** |
| Потенциал | **8/10** |

**Ornis — амбициозный экспериментальный игровой движок на Rust, а не готовый production engine.** Архитектурно проект уже достаточно серьёзный: около 10 крейтов, ECS, физика, wgpu-рендер, MaterialX, WASM/WebGPU-редактор, планировщики и quality tooling.

Главная проблема проекта — не отсутствие интересных технологий, а **широкий scope и незавершённая интеграция между подсистемами**.

---

## Дополнение: фактическая интеграция Scheduler, World и игрового цикла

### Краткий ответ

Сейчас у Ornis **нет единого runtime-мирового цикла, который связывает Scheduler, ECS, Physics, Render и CPU/GPU**. Есть отдельные хорошо проработанные механизмы и несколько параллельных «скелетов» интеграции, но они пока не собраны в одну работающую систему.

### 1. Используется ли Scheduler?

#### `ornis-core::Schedule`

Структура существует:

```rust
ornis_core::Schedule
ornis_core::Resources
ornis_core::System
ornis_core::SystemAccess
```

Она умеет:

- регистрировать системы;
- анализировать `reads`/`writes`;
- строить уровни параллельного выполнения;
- соблюдать `order_before`;
- запускать системы последовательно или через Rayon;
- проверять объявленные доступы;
- работать с access-декларациями для `SmartStore`-лент.

Реализация находится в:

```text
crates/core/src/schedule.rs
crates/schedule/src/lib.rs
```

Но **в основном runtime этот Scheduler сейчас не вызывается**.

По фактическому использованию:

- в `crates/core` он тестируется и исполняется через `Engine::run_frame`;
- в `crates/render/tests/scheduler_parity.rs` проверяется соответствие render-планировщика;
- в `crates/render/src/frame_exec.rs` используется общий `ornis_schedule::run_levels` для render-пассов;
- editor-only physics systems исполняются core `Schedule` через `EditorWorld::tick`;
- native loop запускает `RenderWorld::run_frame`, а WASM loop — тот же library-level host;
- полный scheduler всех доменов (input, physics, render и gameplay) ещё не собран.

То есть `ornis-core::Schedule` уже является frame executor для подключённых
runtime-доменов, но пока не единым главным планировщиком всего движка:
`FramePlan` сохраняет специализированное управление render resources/pass'ами.

#### Render Scheduler / FramePlan

У рендера есть отдельный механизм:

```text
FramePlan
FrameLayout
RenderFrame3D
FrameExecutor
```

Он используется в:

- `crates/render/src/frame_plan.rs`;
- `crates/render/src/frame_exec.rs`;
- render-тестах;
- render benchmarks;
- render examples.

Он умеет:

- строить порядок render passes;
- вычислять lifetime ресурсов;
- делать aliasing текстур;
- строить уровни параллельной записи команд;
- исполнять pass'ы через `FrameExecutor`;
- использовать общий `ornis_schedule::run_levels`.

Но это **не общий игровой Scheduler**. Это scheduler именно для render frame
и его transient GPU resources; он остаётся специализированным внутренним
планом, пока верхний `Engine` orchestrates domain systems.

Текущие production render paths используют один и тот же concrete plan-capable
pipeline:

Native:

```text
src/main.rs
GameApp::render_frame
RenderWorld::run_frame
→ RenderExtract
→ Renderer3D uploads
→ RenderFrame3D::render / FramePlan
```

WASM:

```text
crates/wasm/src/lib.rs
requestAnimationFrame
RenderWorld::run_frame
→ RenderExtract
→ Renderer3D uploads
→ RenderFrame3D::render / FramePlan
```

`RenderBackend::render_scene` остаётся compatibility/plugin и reference API;
основные native/WASM loops больше не вызывают этот legacy shortcut и не
дублируют pass recording logic.

#### Physics Scheduler

`ornis-physics` сохраняет собственный оптимизированный внутренний pipeline и
вызывает `BuiltinPhysicsEngine::step(fixed_delta)` как одну доменную
операцию. В editor-only и native showcase этот шаг обёрнут core systems
`PhysicsSyncIn` → `PhysicsStep` → `PhysicsSyncOut`, зарегистрированными в
`Engine::fixed_schedule`; `Engine::run_frame` выбирает bounded число fixed
updates и запускает frame schedule после них. WASM physics намеренно не
подключается за serialization boundary.

В `ornis-physics` есть собственный внутренний pipeline:

```text
integration
→ broad phase
→ narrow phase
→ islands
→ velocity solving
→ positional solving
```

Это внутренняя последовательность физического движка, а не `ornis_core::Schedule`.

Правильная формулировка сейчас такая:

> Physics имеет собственный внутренний solver pipeline, но не является системой, зарегистрированной в общем engine Scheduler.

### 2. Существует ли единый мир?

#### Формально — частично

В `ornis-core` есть:

```text
Resources
SmartStore
Schedule
```

`Resources` — это type-erased singleton container:

```rust
HashMap<TypeId, Box<dyn Any + Send + Sync>>
```

Системы получают:

```rust
fn run(&self, resources: &Resources)
```

А `SmartStore` содержит ECS-компоненты.

Это уже реализованный фундамент логического `ornis_core::World`: он объединяет `SmartStore` и singleton-ресурсы через `Resources` и умеет запускать `Schedule`. Дополнительно `ornis_core::Engine` публикует `Time`, `FixedTime` и `InputState`, запускает bounded fixed schedule, затем один once-per-frame schedule, после чего очищает transient input deltas. Native winit adapter заполняет ресурс, но **engine-level runtime**, связывающий этот World с Physics, Renderer, GPU context, browser input consumers и полным frame lifecycle, пока не собран.

#### Что реально существует

##### Editor World

В `src/editor_world.rs` есть:

```rust
pub struct EditorWorld {
    engine: ornis_core::Engine,
    alive: Vec<Entity>,
    scene_name: String,
    version: u64,
}
```

Это editor-only facade над общим core runtime host'ом.

Он содержит:

- `ornis_core::Engine` с `World`, `SmartStore` и `SceneEnvironment`-ресурсом;
- список живых entities;
- Transform/Mesh/Material/Name в ECS-лентах;
- scene version;
- команды редактора.

Он умеет:

- создать entity;
- удалить entity;
- изменить компонент;
- сериализовать сцену;
- загружать сцену;
- сохранять сцену;
- публиковать snapshots.

Но:

- содержит `BuiltinPhysicsEngine` через ресурс `PhysicsRuntime` и три
  physics systems (`sync_in`/`step`/`sync_out`);
- не содержит `Renderer3D` и не является полным native/WASM frame loop;
- запускает physics frame через `Engine::run_frame` при timeout между командами;
- WASM получает от него JSON snapshot через HTTP.

То есть `EditorWorld` уже использует **общий core World**, но остаётся
editor-only facade, а не полным engine runtime.

##### Native GameContext

В `src/main.rs` native режим использует:

```rust
struct GameContext {
    window,
    device,
    queue,
    surface,
    renderer3d: Renderer3D,
    frame_plan: RenderFrame3D,
    sphere_mesh,
    render_world: ornis_render::RenderWorld,
    orbit: ornis_render::OrbitCamera,
    remote_cmd_rx,
    remote_ev_tx,
    entity_count,
}
```

Это всё ещё минимальный showcase context, но теперь он:

- содержит `ornis_render::RenderWorld` (внутри `ornis_core::Engine`) с
  ECS-компонентами сцены;
- запускает общий `RenderExtract` через `RenderWorld::run_frame`;
- хранит wgpu surface/renderer/mesh отдельно как backend-ресурсы;
- использует общий `RenderFrame3D`/`FramePlan` для native frame recording;
- содержит shared `OrbitCamera`, зарегистрированный как once-per-frame
  `InputState` consumer в Engine schedule;
- содержит `PhysicsRuntime` и скрытый static floor с одним dynamic showcase body;
- рисует фиксированную showcase-сцену из пяти ECS-сущностей.

Native rendering и physics уже получают данные через ECS/World schedule;
`Engine` владеет bounded fixed 60 Hz accumulator, а physics systems получают
фиксированный шаг через `FixedTime`. Полный набор gameplay stages и
cross-domain frame contract ещё не подключён; отдельный render extraction и
backend-owned GPU lifecycle остаются переходными границами.

##### WASM RenderWorld и GPU adapter

В WASM серверный `Scene` / `LiveScene`, полученный через JSON, сначала
проходит serialization boundary и восстанавливается в общий с native
library-level `ornis_render::RenderWorld`:

```text
EditorWorld
→ /api/scene
→ RenderWorld (Engine + SmartStore + Schedule)
→ RenderExtracted
→ Renderer3D / RenderFrame3D
```

Внутренний `GpuScene` WASM теперь содержит только mesh, extracted snapshot
и light tuples — он не повторяет ECS-to-material/instance conversion.
`RenderWorld::run_frame` запускает тот же `Engine`/`RenderExtract` контракт,
а `RenderFrame3D` записывает тот же typed `FramePlan`, что и native.

Это не общий in-process world между сервером и браузером:

```text
server-side authoritative world
→ versioned serialized snapshot
→ browser-side RenderWorld copy
```

Между ними нет общей памяти и общей ECS-ссылки; это намеренная
serialization boundary из IDEAS §28.

### 3. Откуда сейчас берёт данные рендеринг?

Есть три разных источника.

#### Native runtime

В `src/main.rs` showcase-сцена теперь создаётся как ECS-компоненты
`TransformDesc`/`MeshDesc`/`MaterialDesc` внутри `ornis_core::Engine`.
`RenderExtract` запускается через `Engine::run_frame` и формирует
backend-neutral `RenderExtracted`, после чего native renderer загружает
полученные материалы и instances в GPU.

`Renderer3D` и `sphere_mesh` пока остаются native-owned ресурсами, поэтому
это уже ECS-backed extraction, но ещё не полный FramePlan/runtime pipeline.

#### Editor/WASM runtime

Источник такой:

```text
editor/scene.ron
или
/api/scene
```

Дальше:

```text
Scene / LiveScene
→ RenderWorld::replace_scene
→ Engine::run_frame
→ RenderExtracted
→ mesh/material/instance uploads
→ Renderer3D
```

Это сохраняет scene serialization boundary, но conversion logic живёт в
`crates/render/src/extraction.rs`, а не в WASM adapter.

#### FramePlan rendering

`RenderFrame3D` получает render-specific pass data и GPU resources. Native и
WASM runtime вызывают его после `RenderWorld::run_frame`/`RenderExtract`:
ECS-компоненты превращаются в общий `RenderExtracted`, затем загружаются в
`Renderer3D` и записываются через один и тот же typed `FramePlan`.

`RenderBackend::render_scene` остаётся legacy compatibility path для
плагинов, тестов и reference probes; production native/WASM loops его не
вызывают.

### 4. Откуда сейчас берёт данные физика?

Физика всё ещё владеет оптимизированным внутренним представлением
`BuiltinPhysicsEngine`, editor-only и native showcase runtime подключают его
как доменный ресурс `PhysicsRuntime`; browser-side physics намеренно остаётся
за serialization boundary.

В editor-only путь выглядит так:

```text
ECS RigidBody + TransformDesc
→ PhysicsSyncIn
→ BuiltinPhysicsEngine::step
→ PhysicsSyncOut
→ ECS RigidBody + TransformDesc
```

Синхронизация и системы находятся в `src/engine_runtime.rs`. `RigidBody`
остаётся внутренним physics-компонентом runtime facade и не входит в текущий
serde scene snapshot. Native showcase подключает скрытый static floor и один
dynamic body, а WASM physics не запускает: браузер остаётся snapshot client.

То есть первый ECS ↔ physics шов уже есть, но полноценный physics pipeline
для всех runtime-платформ и единый Collider/Transform data lifecycle ещё
не реализованы.

### 5. Есть ли общие CPU/GPU данные?

Полной автоматической CPU/GPU data model пока нет, но общий логический
render input уже существует: `RenderWorld`/`SmartStore` → `RenderExtracted` →
platform-owned GPU buffers. Физика и GPU residency пока имеют отдельные
physical representations.

Существуют следующие механизмы.

#### CPU ECS storage

```text
ComponentStore
SmartStore
ColdComponentStore
Lock-free store
```

#### GPU command/data path

```text
CommandQueue
CommandSync
DataResidency
ResidencyTracker
SmartBuffer
GpuCommand
```

#### Render GPU resources

```text
wgpu::Buffer
wgpu::Texture
wgpu::TextureView
Renderer3D
FrameExecutor
```

#### Physics GPU path

В `ornis-physics` есть отдельный GPU solver.

Core `World` и backend-neutral `Engine` уже существуют, но пока не
содержат одновременно domain resources physics/render/GPU и не являются
владельцами полного production frame lifecycle.

И нет единой автоматической схемы:

```text
ECS component changed
→ residency tracker notices it
→ CPU/GPU synchronization
→ physics or render consumes same authoritative data
```

`SmartBuffer` и residency infrastructure существуют, но автоматическая полноценная синхронизация пока не завершена.

### 6. Существует ли игровой цикл?

#### Native mode — да, но минимальный и не engine-level

В `src/main.rs` есть winit lifecycle:

```text
main()
→ EventLoop::new()
→ run_app()
→ GameApp::resumed()
→ GameApp::about_to_wait()
→ GameApp::window_event()
```

Инициализация:

```text
GameApp::resumed
→ create window
→ create wgpu instance
→ request adapter
→ create device/queue
→ create surface
→ create Renderer3D + RenderFrame3D
→ create RenderWorld from `assets/scene.ron`
→ run initial RenderExtract
```

Каждый кадр:

```text
about_to_wait
→ process_remote_commands
→ request_redraw

RedrawRequested
→ consume InputState with OrbitCamera
→ Engine::run_frame
   ├── RenderExtract
   └── native PhysicsSyncIn / PhysicsStep / PhysicsSyncOut
→ acquire surface texture
→ set camera/lights
→ upload extracted materials/instances
→ FramePlan / RenderFrame3D
→ submit
→ present
```

В этом цикле уже есть ECS systems execution, общий render extraction,
shared input consumer, native showcase physics и native/WASM `FramePlan`.
`Engine` теперь предоставляет отдельный bounded `FixedTime` host: fixed
systems выполняются перед once-per-frame schedule, а `RenderExtract` не
повторяется для каждого substep. Полноценные именованные gameplay stages
пока не выделены:

```text
fixed schedule (physics + future fixed gameplay)
→ once-per-frame schedule (input consumers/render extraction)
→ backend render/present
```

Также нет:

- полного набора gameplay systems, читающих `InputState` и меняющих ECS;
- physics step в browser-side RenderWorld (это намеренно отдельный snapshot client);
- post-frame systems;
- frame statistics;
- deterministic cross-domain update stage beyond the bounded fixed host.

#### Editor-only mode

В `editor-only`:

```text
main
→ создать command/event channels
→ запустить EditorWorld thread
→ запустить HTTP server
→ park main thread
```

Внутри editor thread:

```text
load editor/scene.ron
→ publish initial snapshot
→ ждать UiCommand с timeout
→ выполнить команду
→ Engine::run_frame (physics sync/step/sync-out)
→ при изменении позы отправить snapshot
```

Это пока не полный игровой цикл: editor-only runtime имеет physics tick,
но не рендерит кадр и не подключает `FramePlan`; WASM получает состояние
через HTTP snapshot'ы.

#### WASM mode

В WASM игровой/рендерный цикл существует в виде:

```text
start_renderer
→ init WebGPU
→ load scene
→ create RenderWorld (Engine + SmartStore + Schedule)
→ initial Engine::run_frame / RenderExtract
→ create Renderer3D + RenderFrame3D
→ spawn_render_loop
→ requestAnimationFrame
```

Каждый кадр выполняется примерно:

```text
resize
→ иногда poll /api/scene
→ RenderWorld::replace_scene
→ Engine::run_frame / RenderExtract
→ upload extracted data
→ update camera
→ acquire surface texture
→ RenderFrame3D::render / FramePlan
→ present
→ requestAnimationFrame
```

Это настоящий render loop с ECS extraction и render plan, но он:

- не является общим in-process world с сервером — состояние приходит
  versioned snapshot'ами;
- не запускает Physics;
- не содержит input/gameplay systems;
- содержит только client-side camera update и rendering после boundary.

### Итоговая схема текущего состояния

Сейчас архитектура выглядит так:

```text
                    ┌──────────────────────────────┐
                    │ Ornis-core Engine             │
                    │ World + Schedule + Time       │
                    └──────────────┬───────────────┘
                                   │
             ┌─────────────────────┴─────────────────────┐
             │                                           │
┌────────────▼────────────┐                 ┌────────────▼────────────┐
│ EditorWorld              │                 │ Native GameContext      │
│ core World + physics     │                 │ core Engine +            │
│ sync/step/sync-out       │                 │ RenderExtract            │
└────────────┬────────────┘                 └────────────┬────────────┘
             │ REST/JSON                                 │ GPU upload
             ▼                                           ▼
┌────────────────────────┐                 ┌──────────────────────────┐
│ WASM RenderWorld        │                 │ Native RenderWorld        │
│ snapshot → Engine       │                 │ assets → Engine           │
│ → RenderExtract         │                 │ → RenderExtract           │
└────────────┬───────────┘                 └────────────┬─────────────┘
             │                                          │
             └──────────────┬───────────────────────────┘
                            ▼
                 ┌──────────────────────────┐
                 │ Renderer3D + FramePlan    │
                 │ shared typed pass path    │
                 └──────────────────────────┘
```

То есть:

- **Core `Engine`/`Schedule` уже исполняется** для подключённых render- и
  editor-only/native showcase physics-систем;
- **Render `FramePlan` остаётся отдельным render scheduler** для lifetime,
  aliasing и typed pass execution;
- **Physics подключена как systems в editor-only и native showcase runtime**;
- **RenderExtract, shared OrbitCamera и `FramePlan` подключены к native и WASM render loops**;
- **EditorWorld использует core `World`, но остаётся server-side facade**;
- **Native loop всё ещё showcase loop**, несмотря на ECS-backed extraction;
- **WASM loop — browser-side snapshot client** с собственным `RenderWorld`;
- **единый authoritative world между server и browser не требуется**:
  serialization boundary сохраняется по IDEAS §28;
- **единого CPU/GPU data lifecycle нет**;
- **полный native/WASM physics/render/ECS кадр не проходит через один
  cross-domain schedule**.

### Что нужно сделать, чтобы появилась настоящая единая архитектура

`ornis_core::World` уже существует как логический контейнер
`Resources` + `SmartStore`, а `ornis_core::Engine` — как минимальный
backend-neutral frame host с `Time`/`FixedTime` и двумя schedule-планами.
Editor-only physics, native showcase physics и native/WASM render extraction
уже используют этот host. Нужен следующий слой, который доведёт
интеграцию до общего cross-domain physics/render/input pipeline;
serialization boundary server↔browser при этом сохраняется:

```rust
pub struct GameRuntime {
    frame_host: ornis_core::Engine,
    physics: PhysicsRuntime,
    renderer: RendererRuntime,
    render_frame: RenderFrame3D,
}
```

Цикл:

```rust
fn frame(&mut self, dt: Duration) {
    // Fixed systems (physics/gameplay) run first; the once-per-frame
    // schedule then performs extraction before the backend presents.
    self.frame_host.run_frame(dt.as_secs_f32());
    self.render_frame();
}
```

Более правильный вариант с именованными фазами (следующий слой поверх
текущего fixed/frame разделения):

```text
PreUpdate
→ Input
→ Gameplay systems
→ FixedUpdate / Physics
→ Transform propagation
→ Render extraction
→ Render schedule
→ Present
→ PostFrame
```

Сейчас реализованы `Engine::fixed_schedule`/`FixedTime` и
`Engine::schedule` для once-per-frame работы; отдельные `PreUpdate`,
`Gameplay`, `PostFrame` и backend render stages ещё не выделены.

При этом Physics и Render должны быть **потребителями данных из World**, а не независимыми владельцами параллельных копий состояния.

Именно это — следующий большой архитектурный шаг проекта. Сейчас все необходимые строительные блоки уже существуют, но **их интеграция в единый runtime ещё не выполнена**.

---

## Дополнение: единый Scheduler, единый World и единый execution model

### Уточнение целевой архитектуры

В контексте задумки Ornis более логичной целью является не набор независимых планировщиков для разных доменов, а:

> **один Scheduler + единый World + общая модель типизированных доступов, где render passes, physics и обычные системы являются одной категорией вычислений.**

Предыдущая формулировка о `FramePlan` как о постоянном отдельном render sub-planner была слишком консервативной. В `IDEAS.md`, особенно в секции 28, заложена более сильная идея: Scheduler должен решать общую задачу для всех вычислений движка:

- видеть зависимости по данным;
- распределять порядок выполнения;
- определять параллельные участки;
- выбирать CPU или GPU;
- управлять residency;
- вычислять lifetime ресурсов;
- переиспользовать память;
- строить команды;
- не заставлять программиста вручную разделять ECS, системы и render passes.

Render, Physics и Gameplay — это разные алгоритмы, работающие поверх **одной execution/data model**.

### Единый Scheduler, но не один гигантский алгоритм

Целевая архитектура должна выглядеть не так:

```text
Engine Scheduler
├── Physics
├── RenderFramePlan
└── AssetManager
```

а так:

```text
Engine Scheduler
├── Gameplay systems
├── Physics systems
├── Audio systems
├── Asset systems
├── Render extraction systems
└── Render pass systems
```

Все системы:

- объявляют доступы к данным;
- становятся узлами одного DAG;
- могут выполняться на CPU или GPU;
- используют один World;
- участвуют в одном плане зависимостей.

Domain-specific код при этом содержит только алгоритм:

```text
physics algorithm
render algorithm
audio algorithm
material algorithm
```

Общие механизмы движка предоставляют:

```text
storage
query
dependency analysis
CPU/GPU routing
residency
parallel execution
resource lifetime
buffer/texture allocation
command recording
```

Это соответствует идее «обычный Rust-код → умный конвейер», а не классическому ECS, где разработчик вручную раскладывает всё по системам.

### Render pass как обычная система

Целевая форма может выглядеть примерно так:

```rust
#[smart_pipeline]
fn gbuffer(
    entities: Query<(&Transform, &MeshHandle, &MaterialHandle)>,
    camera: Res<Camera>,
    target: Write<GBuffer>,
) {
    // Rust-код алгоритма рендера.
}
```

Или более низкоуровнево:

```rust
#[smart_pipeline]
fn lighting(
    gbuffer: Read<GBuffer>,
    lights: Read<Lights>,
    output: Write<HdrTarget>,
) {
    // Общий Rust DSL, который может попасть на CPU или GPU.
}
```

Тогда `gbuffer` — не особый объект `FramePlan::add_pass(...)`, а обычный узел Scheduler с типизированными доступами.

Scheduler видит:

```text
gbuffer:
  reads  Transform, MeshHandle, MaterialHandle, Camera
  writes GBuffer

lighting:
  reads  GBuffer, Lights
  writes HdrTarget
```

И строит:

```text
gbuffer → lighting
```

Если две системы используют независимые ресурсы, они могут выполняться параллельно. Если доступы конфликтуют, Scheduler строит зависимость.

### Render resource lifetime как часть общего Scheduler

Render resource lifetime не является аргументом в пользу отдельного архитектурного Scheduler. Это аргумент в пользу того, чтобы общий Scheduler стал достаточно мощным и понимал ресурсы первого класса.

Он должен уметь не только:

```text
system A before system B
```

но и:

```text
resource GBuffer:
  created by GBufferPass
  alive through LightingPass
  dead after CompositePass
```

Тогда текущие возможности `FramePlan`:

- lifetime windows;
- transient textures;
- aliasing;
- texture pool;
- memory budget;
- pass culling;
- render ordering;

становятся не отдельной системой, а частью общего планировщика.

`FramePlan` в переходной архитектуре может оставаться:

```text
scheduler backend / compiled execution plan
```

А в долгосрочной архитектуре его API должен постепенно исчезнуть или стать внутренним implementation detail. Это совпадает с `IDEAS.md §28.1`: pass становится системой с типизированной сигнатурой, а не отдельным imperative builder.

### Physics на том же механизме

Физика не должна навсегда оставаться непрозрачным вызовом:

```rust
physics.step(dt)
```

Внешний physics pipeline должен стать набором систем:

```text
PhysicsSyncIn
PhysicsBroadPhase
PhysicsNarrowPhase
PhysicsIslandBuild
PhysicsVelocitySolve
PhysicsPositionSolve
PhysicsSyncOut
```

Их внутренние алгоритмы остаются domain-specific, но планирование, доступы и CPU/GPU routing становятся общими.

Например:

```text
PhysicsSyncIn:
  reads  Transform, Collider, RigidBody
  writes PhysicsBodies

BroadPhase:
  reads  PhysicsBodies
  writes BroadPhasePairs

NarrowPhase:
  reads  BroadPhasePairs, Shapes
  writes ContactManifolds

Solver:
  reads  ContactManifolds, RigidBodies
  writes RigidBodies

PhysicsSyncOut:
  reads  RigidBodies
  writes Transform
```

Scheduler получает возможность видеть зависимости не как одну непрозрачную функцию `physics.step`, а как реальный pipeline.

Отдельный physics solver может внутри использовать SIMD или GPU-алгоритм, но снаружи подключается через общую модель систем и ресурсов.

### Взаимодействие областей

Единая data/scheduling модель особенно важна для взаимодействия подсистем.

Например, отладка коллайдеров может быть обычным междоменным pipeline:

```text
Physics systems
  writes: ColliderDebugGeometry

DebugDraw system
  reads: ColliderDebugGeometry
  writes: DebugRenderBuffer

Render system
  reads: DebugRenderBuffer
  writes: RenderTarget
```

Scheduler автоматически строит:

```text
Physics
→ ColliderDebugGeometry
→ DebugDraw
→ Render
```

То же самое можно делать для:

- collision contacts;
- navmesh visualization;
- audio emitters;
- particle systems;
- GPU profiler overlays;
- shadow debug;
- skeletal bones;
- editor gizmos;
- physics islands;
- GPU residency diagnostics.

Без общей data/scheduling модели каждая такая связь превращается в отдельный integration layer и увеличивает сложность проекта.

### «Невидимый ECS» — не классический ECS API

Цель Ornis не в том, чтобы пользователь вручную писал классический ECS-код:

```rust
for (entity, transform, velocity) in query.iter_mut() {
    // manual ECS-oriented code
}
```

Цель — позволить писать привычный объектный или предметный код:

```rust
for entity in entities {
    entity.position += entity.velocity;
}
```

А pipeline сам:

1. определяет используемые поля;
2. раскладывает данные по sparse sets;
3. выводит lane-доступы;
4. выбирает CPU/Rayon или GPU/WGSL;
5. строит зависимости;
6. планирует residency;
7. группирует работу;
8. исполняет вычисление.

Поэтому правильный вопрос — не «как подключить Physics и Render к ECS?», а:

> «Как сделать так, чтобы Physics и Render были алгоритмами, которые используют тот же скрытый data/execution pipeline, что и обычный пользовательский код?»

Это существенно более сильная постановка, чем классический явный ECS API.

### Единый World должен быть логическим

«Единый World» не означает, что данные всегда находятся только в одном физическом буфере памяти. Один логический компонент может иметь несколько представлений:

```text
logical component
├── CPU dense storage
├── GPU storage buffer
├── render representation
└── physics representation
```

Но это должны быть представления одной логической сущности, которыми управляет общий residency/ownership механизм.

Например:

```text
Position
→ authoritative logical component
→ CPU lane, если нужен gameplay
→ GPU lane, если выполняется particle/update kernel
→ render read, когда строится кадр
```

Scheduler и residency tracker должны понимать:

- кто последним писал данные;
- где находится актуальная версия;
- нужна ли синхронизация;
- можно ли передать команду вместо копирования;
- когда необходимо materialize CPU/GPU view.

Это соответствует идеям Ornis о Data Residency, Instructions Instead of Data, ZST CPU/GPU routing и Command-Based Sync.

### Единый Asset Pipeline

Asset pipeline также должен быть частью той же модели, а не отдельным изолированным менеджером.

Целевая схема:

```text
AssetServer
→ AssetRegistry
→ typed handles
→ CPU/GPU residency
→ loader/importer systems
→ asset events
→ render/physics consumers
```

В ECS хранятся handles:

```rust
MeshHandle
MaterialHandle
ColliderHandle
AudioClipHandle
```

а не копии самих больших ассетов.

Общий Scheduler может запускать:

```text
AssetScan
→ ParseMaterialX
→ BuildOpenPBR
→ UploadMaterialGPU
→ CookPhysicsShape
→ MarkAssetReady
```

Типизированные загрузчики остаются специализированными:

```text
.ron       → SceneLoader
.mtlx      → MaterialXLoader
.gltf      → MeshLoader
.png/.jpg  → TextureLoader
.wav/.ogg  → AudioLoader
```

Единый asset lifecycle должен обеспечивать:

- единые `AssetId` и handles;
- async loading;
- dependency tracking;
- cache;
- hot reload;
- load/error events;
- CPU/GPU residency;
- versioning;
- safe lifetime management;
- одинаковое поведение native и WASM.

Один asset может иметь несколько runtime-представлений:

```text
MeshAsset
├── CPU mesh
├── GPU mesh
└── Physics collision mesh
```

Или:

```text
MaterialX document
├── OpenPBR material for renderer
├── GPU material buffer
└── editor inspection data
```

### EditorWorld в единой архитектуре

`EditorWorld` должен эволюционировать в frontend над тем же `World`:

```text
Editor command
→ EngineCommand
→ World mutation
→ Schedule / command application
→ events
→ render extraction
```

Целевая модель не такая:

```text
EditorWorld
→ JSON
→ отдельный ad-hoc GpuScene conversion
```

а такая:

```text
Engine World (server authoritative)
├── ECS state
├── Physics state
├── Asset state
├── Render extraction state
└── Editor protocol
              │ versioned serialization boundary
              ▼
Browser RenderWorld (Engine + SmartStore + RenderExtract)
→ Renderer3D + FramePlan
```

Для браузера serialization boundary остаётся обязательной, если WASM и
сервер живут в разных контекстах. Authoritative state находится на стороне
server engine world; browser-side `RenderWorld` — typed snapshot replica, а
не второй источник истины. Общими остаются contract и data-flow, но не
память и не физический solver state.

### Целевая архитектура

```text
                         ┌─────────────────────┐
                         │   Engine Scheduler  │
                         │  unified dependency │
                         │   + CPU/GPU router  │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │        World        │
                         │                     │
                         │ SmartStore          │
                         │ Resources           │
                         │ Assets              │
                         │ Time/Input          │
                         │ Residency           │
                         └──────┬───────┬──────┘
                                │       │
                 ┌──────────────▼─┐   ┌─▼──────────────┐
                 │ Physics systems │   │ Render systems │
                 │                 │   │               │
                 │ broad phase     │   │ extract       │
                 │ contacts        │   │ gbuffer       │
                 │ solver           │   │ lighting      │
                 │ sync in/out      │   │ bloom         │
                 └──────────────┬──┘   └──────┬────────┘
                                │             │
                                └──────┬──────┘
                                       │
                              ┌────────▼────────┐
                              │ Common execution │
                              │ CPU / Rayon      │
                              │ GPU / WGSL       │
                              │ buffers / pools  │
                              │ command sync     │
                              └──────────────────┘
```

Целевая структура может выглядеть так:

`ornis_core::World` уже предоставляет логический контейнер ресурсов и
`SmartStore`. Runtime-обвязка должна добавлять специализированные ресурсы,
не дублируя ECS-состояние:

```rust
pub struct GameRuntime {
    pub frame_host: ornis_core::Engine,
    pub physics: PhysicsRuntime,
    pub renderer: RendererRuntime,
    pub render_frame: RenderFrame3D,
    pub gpu: GpuContext,
}
```

### Что делать с текущими структурами

#### `ornis_core::Schedule`

Развивать в сторону настоящего общего Scheduler:

- typed resource access;
- component/lane access;
- CPU/GPU execution target;
- resource lifetime;
- transient resource declarations;
- command recording;
- residency dependencies;
- stage/phase support;
- unified diagnostics.

#### `FramePlan`

Не выбрасывать сразу. Использовать как промежуточную реализацию:

```text
FramePlan
→ адаптировать под общий Scheduler
→ сделать pass системой
→ перенести resource lifetime в общую модель
→ оставить FrameExecutor backend-specific
```

Текущий `FramePlan` — не неправильное решение, а **первый специализированный прототип будущего unified scheduler**.

#### Physics

Разделить физический `step` на системы, сохранив внутренние оптимизированные kernels там, где это выгодно.

#### `SmartStore`

Сделать его не просто storage-компонентов, а основой для прозрачного доступа:

- queries;
- lanes;
- generated access metadata;
- CPU/GPU route;
- packed iteration;
- residency-aware views.

#### World

`ornis_core::World` уже создан как логический контейнер `Resources` с авторитетным `SmartStore`; не превращать его в огромный god-object. Специализированные physics/render/assets ресурсы должны быть доступны единому планировщику через общий контракт.

### Эволюционный план

1. ✅ создать фундамент `ornis_core::World` и backend-neutral `Engine` с `Time`/`FixedTime`/`InputState`, fixed schedule и bounded accumulator (`crates/core/src/{world.rs,engine.rs,input.rs}`);
2. ✅ зарегистрировать `PhysicsRuntime` в editor-only и native showcase World (Time, InputState и SmartStore предоставляет core foundation); asset resources ещё впереди;
3. ✅ добавить `PhysicsSyncIn`, `PhysicsStep`, `PhysicsSyncOut` для editor-only и native showcase runtime; browser-side physics намеренно не запускается;
4. ✅ добавить backend-neutral `RenderExtract` и вынести его в `ornis-render::extraction`;
5. ✅ перевести native showcase loop на общий `RenderWorld::run_frame` для ECS-backed extraction и shared OrbitCamera;
6. ✅ подключить `RenderFrame3D`/`FramePlan` к native render path;
7. ✅ перевести WASM loop на тот же `RenderWorld`/`Engine`/`RenderExtract`/`FramePlan` contract после serialization boundary;
8. завершить browser/gameplay input consumers и собрать полноценный cross-domain runtime поверх общего fixed host; browser physics остаётся за serialization boundary;
9. добавить `AssetServer` и handles;
10. только после этого развивать WebSocket, scripting и сложный hot reload.

### Quality gate: BCA → rustqual (2026-08-30)

**Решение:** BCA заменяется на rustqual.

Аргументы: BCA MPL-2.0 — нельзя встраивать в код как lib без лицензионных последствий; rustqual MIT — можно. Ornis на 100% Rust, фронтенд JS будет заменён Rust (blitz, Boa незрелая) — мульти-язык BCA не нужен. rustqual покрывает BCA + структурные измерения (IOSP, DRY, SRP, Coupling, Test Quality, Architecture call_parity).

**Фильтр вкуса (нарезка + magic) зафиксирован в `rustqual.toml`:**
- `max_function_lines = 80` (было 60) — режем только если можно дать доменное имя `calc_total/save`; нет имени → `// qual:allow(iosp) reason: match-dispatcher`.
- `max_cognitive = 20`, `max_cyclomatic = 20` — как в `bca.toml` (hard), не 15/10 default.
- magic: пока без allow-листа (725 находок) — доменные числа в `const` (`MAX_WS_PAYLOAD`, `SLEEP_THR`), тривиальные `0/1` глушить `// qual:allow(complexity, magic)` до патча allow-листа; не плодить `const ZERO = 0`.
- `max_suppression_ratio = 0.10` — клапан: если `qual:allow` >10% функций → warning, ratchet а не цель.
- IOSP leniency: `strict_closures=false`, `strict_iterator_chains=false`, `strict_error_propagation=false`.

**План перехода (ladder rung 5):**
1. `cargo run --manifest-path /tmp/rustqual -- . --save-baseline baseline.json` — baseline текущий Score 11% / 1707 findings.
2. В CI `quality.yml`: `rustqual --compare baseline.json --fail-on-regression` рядом с `bca check` (две недели параллельно).
3. Поднимать `min-quality-score` по 5% за итерацию, чиня только новые нарушения.
4. `cargo uninstall big-code-analysis && rm bca.toml .bca-baseline.toml` когда Score ≥80% и 2 недели без регресса.
5. `bca.toml` до удаления остаётся tiered fallback, уже `nexits=9` для `poll_one_frame`.

> ponytail: `rustqual.toml` — единственный source of truth, не дублировать пороги в `xtask/quality.rs`.

### Итог

Более логичная цель для Ornis:

> **один Scheduler, один логический World, один Asset Pipeline и единый CPU/GPU execution model.**

При этом:

- Physics, Render, Audio и Gameplay реализуют свои алгоритмы;
- общий движок отвечает за хранение, зависимости, параллелизм, выбор CPU/GPU и синхронизацию;
- render lifetime/aliasing не исчезают, а становятся частью общего resource planner;
- ECS не должен быть классическим явным ECS API;
- sparse sets и macro-based hidden ECS должны скрывать от пользователя механическую часть распределения данных и систем;
- взаимодействия между доменами должны выражаться обычными общими компонентами и ресурсами.

Итоговая формула:

> **Доменная специализация должна существовать на уровне алгоритмов и backend execution, но не на уровне разрозненных моделей World, Scheduler и data flow.**

Это одна из самых сильных и отличительных идей Ornis. Текущая кодовая база находится на переходной стадии: `Schedule`, `FramePlan`, `SmartStore`, `CommandSync` и registry уже являются строительными блоками, но unified runtime ещё предстоит собрать.
