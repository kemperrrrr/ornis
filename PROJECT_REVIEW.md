# Ornis: текущие ограничения и план развития

> **Актуальный срез: 2026-08-27.** Этот документ объединяет
> аудит и план работ. Формулировки в разделах с датами —
> исторические снимки; текущие статусы сверены ниже с кодом и
> последующими коммитами.

## Главные ограничения сейчас

- GPU-диспетчеризация в `core` пока фактически является заглушкой: `Dispatcher` умеет выбрать GPU, но `GpuExecutor::execute` не выполняет GPU-операцию.
- `PhysicsEngine::shapecast` уже реализован и покрыт тестами, но физический API все еще ограничен небольшим набором форм и возможностей.
- В `ornis-core` уже есть логический `World`-фундамент (`Resources` с
  авторитетным `SmartStore` и запуском `Schedule`) и backend-neutral
  `Engine` с ресурсом `Time`, но они ещё не подключены к единому
  runtime-циклу physics/render/editor.
- Scheduler вынесен в отдельный crate и хорошо протестирован, но еще
  не стал единым runtime-планировщиком всего движка.
- Редактор и ECS пока не образуют полностью единую live-систему: синхронизация идет через polling, а часть сценариев остается демонстрационной.
- Проект одновременно развивает ECS, GPU compute, WASM editor, MaterialX, audio, physics и собственные макросы. Такой широкий scope увеличивает стоимость сопровождения и риск распыления усилий.
- Native-приложение пока скорее showcase/runtime shell, чем полноценная игровая платформа.

## План дальнейшей работы

1. ~~**Закончить вертикальный сценарий редактора**~~ — ✅ закрыт (2026-08-26);
   следующий этап — интеграция этого сценария с `ornis_core::World`.

   Создание entity → изменение Transform/Material → обновление WASM-сцены →
   сохранение и загрузка сцены (`save_scene`/`load_scene` через
   `POST /api/command`, атомарная запись `editor/scene.ron`, меню File →
   Save/Reload в UI, события `scene_saved`/`scene_loaded` в `/api/events`).
   Это завершает editor-only vertical slice; он пока использует отдельный
   `EditorWorld` и polling, а не общий runtime `World`.

2. **Довести physics API**

   Добавить фильтры столкновений, collision layers, триггеры, CCD-ротацию и более точный `raycast` для OBB и capsule.

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

   Заменить polling на WebSocket или хотя бы добавить sequence numbers, acknowledgements, обработку ошибок команд и защиту от устаревших snapshots.

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

Editor-only vertical slice уже работает. Следующий приоритет — подключить его к добавленному логическому `ornis_core::World` и общему frame contract, не создавая вторую authoritative-модель состояния; затем заняться масштабированием broadphase physics и надёжностью editor-протокола.


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

- polling вместо WebSocket;
- fire-and-forget команды;
- `POST /api/command` возвращает `{}` даже для некорректной команды;
- нет нормального request ID / acknowledgement;
- нет sequence numbers для snapshot'ов;
- ошибка команды приходит отдельно через event;
- редактор и движок не используют единый надёжный live-протокол;
- native runtime по-прежнему выглядит скорее showcase shell, чем полноценный runtime.

Особенно неприятный момент — ручная сериализация событий в `format_events` в [`crates/editor-backend/src/remote.rs`](crates/editor-backend/src/remote.rs). Поля вроде `cmd_type` и `type_name` вставляются в JSON через `format!`, без JSON escaping. Если туда попадёт кавычка, обратный слеш или управляющий символ, endpoint может вернуть невалидный JSON.

Лучше сериализовать структуры через `serde_json`, а не собирать JSON вручную.

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

- end-to-end тест editor command → ECS mutation → scene snapshot;
- end-to-end тест snapshot → WASM scene;
- тест stale snapshot / sequence number;
- fuzzing HTTP command payloads;
- benchmark worst-case broad phase;
- regression test на JSON escaping событий;
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

Минимальный набор:

- request ID;
- explicit ACK/error response;
- sequence number у snapshots;
- version у scene state;
- typed event serialization через Serde;
- защита от stale updates;
- WebSocket после стабилизации REST-контракта.

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

- в `crates/core` он преимущественно тестируется;
- в `crates/render/tests/scheduler_parity.rs` проверяется соответствие render-планировщика;
- в `crates/render/src/frame_exec.rs` используется общий `ornis_schedule::run_levels`;
- физика его не использует;
- native game loop его не использует;
- WASM render loop его не использует;
- editor world его не использует.

То есть `ornis-core::Schedule` сейчас — это **готовая инфраструктура и API**, но не главный исполнитель кадра движка.

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

Но это **не общий игровой Scheduler**. Это scheduler именно для render frame.

Более того, существующие native и WASM entry points сейчас используют в основном старый прямой путь:

```rust
renderer.render_scene(...)
```

Native:

```text
src/main.rs
GameApp::about_to_wait
GameApp::window_event
render_frame
Renderer3D::render_scene
```

WASM:

```text
crates/wasm/src/lib.rs
requestAnimationFrame
FrameState::draw
Renderer3D / RenderBackend
```

`RenderFrame3D` полноценно используется в тестах, benchmark'ах и examples, но **не является пока центральным render pipeline основного native/WASM runtime**.

#### Physics Scheduler

Physics Scheduler не использует.

Физика вызывается напрямую:

```rust
physics.step(delta_time)
```

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

Это уже реализованный фундамент логического `ornis_core::World`: он объединяет `SmartStore` и singleton-ресурсы через `Resources` и умеет запускать `Schedule`. Дополнительно `ornis_core::Engine` публикует `Time` и запускает один frame schedule. Но **engine-level runtime**, связывающий этот World с Physics, Renderer, GPU context, input и полным frame lifecycle, пока не собран.

#### Что реально существует

##### Editor World

В `src/editor_world.rs` есть:

```rust
pub struct EditorWorld {
    world: World,
    alive: Vec<Entity>,
    scene_name: String,
    version: u64,
}
```

Это настоящий мир, но только для editor-only режима.

Он содержит:

- `ornis_core::World` с `SmartStore` и `SceneEnvironment`-ресурсом;
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

- в нём нет `BuiltinPhysicsEngine`;
- в нём нет `Renderer3D`;
- он не регистрирует системы в `Schedule` и не запускает игровой frame loop;
- WASM получает от него JSON snapshot через HTTP.

То есть `EditorWorld` — это **мир редактора**, а не общий engine world.

##### Native GameContext

В `src/main.rs` native режим использует:

```rust
struct GameContext {
    window,
    device,
    queue,
    surface,
    renderer3d,
    sphere_mesh,
    materials,
    instance_data,
    remote_cmd_rx,
    remote_ev_tx,
    entity_count,
}
```

Это уже больше похоже на runtime context, но он:

- не содержит `SmartStore`;
- не содержит Physics;
- не содержит `Schedule`;
- не использует `ornis_core::World`;
- хранит `materials` и `instance_data` отдельно;
- рисует фиксированную showcase-сцену из пяти сфер.

Native rendering сейчас не берёт данные из ECS.

##### WASM GpuScene

В WASM создаётся отдельная структура:

```rust
struct GpuScene {
    mesh,
    mesh_params,
    materials,
    instances,
    lights,
}
```

Она строится из `Scene` / `LiveScene`, полученной через JSON.

Это уже связь:

```text
EditorWorld
→ /api/scene
→ WASM
→ GpuScene
→ Renderer3D
```

Но это не общий in-process world. Это:

```text
server-side world
→ serialized snapshot
→ browser-side copy
```

Между ними нет общей памяти и нет общей ECS-ссылки.

### 3. Откуда сейчас берёт данные рендеринг?

Есть три разных источника.

#### Native runtime

В `src/main.rs` данные создаются вручную:

```rust
materials = vec![...]
instance_data = ...
sphere_mesh = create_sphere(...)
```

Рендеринг получает данные из `GameContext`.

Это демонстрационная сцена, не ECS.

#### Editor/WASM runtime

Источник такой:

```text
editor/scene.ron
или
/api/scene
```

Дальше:

```text
Scene
→ build_gpu_scene
→ materials
→ instances
→ lights
→ Renderer3D
```

Это работает как scene serialization pipeline.

#### FramePlan rendering

`RenderFrame3D` получает render-specific pass data и GPU resources. Он не получает напрямую `SmartStore` и не извлекает автоматически ECS-компоненты.

То есть рендер пока не делает:

```text
world.query::<Transform, Mesh, Material>()
→ render instances
```

Такого общего extraction слоя нет.

### 4. Откуда сейчас берёт данные физика?

Физика живёт полностью отдельно.

`BuiltinPhysicsEngine` владеет своими:

- rigid bodies;
- shapes;
- handles;
- velocities;
- contacts;
- joints;
- islands.

Она не читает автоматически:

```rust
SmartStore<Transform>
SmartStore<RigidBody>
SmartStore<Collider>
```

и не записывает автоматически результаты обратно в ECS.

В текущем коде нет интеграции вида:

```text
ECS Transform + Collider
→ Physics step
→ ECS Transform update
```

Есть physics API и собственный physics state, но нет physics system, зарегистрированной в общем мире.

### 5. Есть ли общие CPU/GPU данные?

Пока нет единой data model.

Существуют отдельные механизмы.

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

Но сейчас нет единого объекта вроде:

```rust
struct World {
    ecs: SmartStore,
    physics: PhysicsWorld,
    renderer: Renderer,
    resources: Resources,
    schedule: Schedule,
    gpu: GpuContext,
}
```

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
→ create Renderer3D
→ create mesh/materials/instances
```

Каждый кадр:

```text
about_to_wait
→ process_remote_commands
→ request_redraw

RedrawRequested
→ render_frame
→ acquire surface texture
→ set camera
→ set lights
→ upload materials
→ upload instances
→ render_scene
→ submit
→ present
```

Но в этом цикле отсутствуют отдельные стадии:

```text
pre_update
input
fixed_update
physics
post_update
extract
render
cleanup
```

Фактически есть:

```text
process remote commands
→ render
```

Также нет:

- фиксированного physics timestep;
- accumulator;
- `delta_time` для gameplay;
- ECS systems execution;
- physics step в игровом цикле;
- render extraction;
- post-frame systems;
- frame statistics;
- deterministic update stage.

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
→ ждать UiCommand
→ выполнить команду
→ отправить GameEvent/snapshot
```

Это **command-processing loop**, но не игровой цикл.

Там нет render tick и physics tick.

#### WASM mode

В WASM игровой/рендерный цикл существует в виде:

```text
start_renderer
→ init WebGPU
→ load scene
→ create GpuScene
→ create Renderer
→ spawn_render_loop
→ requestAnimationFrame
```

Каждый кадр выполняется примерно:

```text
resize
→ иногда poll /api/scene
→ применить live scene
→ update camera
→ acquire surface texture
→ draw
→ present
→ requestAnimationFrame
```

Это настоящий render loop, но он:

- не вызывает общий Scheduler;
- не запускает Physics;
- не запускает ECS systems;
- получает состояние snapshot'ами;
- содержит только client-side camera update и rendering.

### Итоговая схема текущего состояния

Сейчас архитектура выглядит так:

```text
                    ┌────────────────────┐
                    │ Ornis-core Schedule│
                    │  готов, но не wired │
                    └────────────────────┘

┌────────────────┐       REST/JSON       ┌────────────────┐
│  EditorWorld   │ ────────────────────> │  WASM GpuScene │
│ SmartStore     │                        │  Renderer3D    │
│ scene state    │                        │ requestFrame   │
└────────────────┘                        └────────────────┘


┌────────────────┐
│ Native GameApp │
│ GameContext    │
│ direct render  │
│ fixed showcase │
└────────────────┘


┌─────────────────────────┐
│ BuiltinPhysicsEngine    │
│ own bodies/shapes/state  │
│ direct physics.step()   │
└─────────────────────────┘


┌─────────────────────────┐
│ RenderFrame3D/FramePlan │
│ render-only scheduler   │
│ tests/examples/benches  │
└─────────────────────────┘
```

То есть:

- **Scheduler есть**, но не является главным scheduler'ом движка;
- **Render FramePlan есть**, но это отдельный render scheduler;
- **Physics есть**, но он не подключён как system;
- **EditorWorld есть**, но он не является общим World;
- **Native loop есть**, но это showcase loop;
- **WASM loop есть**, но это browser render loop;
- **единый runtime authoritative world ещё не wired**: core `World` существует, но editor/native/physics/render используют отдельные контейнеры;
- **единого CPU/GPU data lifecycle нет**;
- **physics/render/ECS не проходят через один frame schedule**.

### Что нужно сделать, чтобы появилась настоящая единая архитектура

`ornis_core::World` уже существует как логический контейнер
`Resources` + `SmartStore`, а `ornis_core::Engine` — как минимальный
backend-neutral frame host с `Time` и `Schedule`. Нужен следующий слой,
который зарегистрирует в World доменные ресурсы и подключит physics/render
к этому frame runner:

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
    self.frame_host.run_frame(dt.as_secs_f32()); // publishes Time + runs scheduled systems
    self.physics_step();
    self.extract_render_data();
    self.render_frame();
}
```

Более правильный вариант с фазами:

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
→ отдельный WASM state
```

а такая:

```text
Engine World
├── ECS state
├── Physics state
├── Asset state
├── Render extraction state
└── Editor protocol
```

Для браузера всё равно останется serialization boundary, если WASM и сервер живут в разных контекстах. Но authoritative state должен быть один — на стороне engine world, а браузер должен получать versioned snapshots/events.

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

1. ✅ создать фундамент `ornis_core::World` и backend-neutral `Engine` с `Time` (`crates/core/src/{world.rs,engine.rs}`);
2. зарегистрировать в World `PhysicsRuntime`, input и asset resources (Time и SmartStore предоставляет core foundation);
3. добавить `PhysicsSyncIn`, `PhysicsStep`, `PhysicsSyncOut`;
4. добавить `RenderExtract`;
5. перевести native loop на `Engine::run_frame`;
6. подключить `RenderFrame3D` как внутренность `RenderSystem`;
7. перевести WASM loop на тот же frame contract;
8. добавить `AssetServer` и handles;
9. только после этого развивать WebSocket, scripting и сложный hot reload.

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
