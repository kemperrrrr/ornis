# PLAN — план реализации Ornis

Рабочий документ, синхронизированный с кодом (не переписан из старых
планов). Датированные блоки сохраняют хронологию; текущие статусы берутся
из позднейших пометок и сверенного [`README.md`](README.md). Документ
дополняет [`IDEAS.md`](IDEAS.md) (архитектурные идеи).

Легенда: ✅ сделано и верифицировано · 🟡 частично · ❌ не начато ·
❄️ заморожено/удалено

## ✅ Сделано

### Ядро и рендер (фазы 0–5 старого плана)

- **Фазы 0–1, ядро ECS**: Sparse Sets (`ComponentStore`), Entity с
  генерационными индексами, bitset-пересечения, страничный sparse,
  ZST-диспетчеризация лент (`b39db39`, `b8e0a54`).
- **Процедурные макросы**: `smart_pipeline` (авто-параллелизация,
  `2b85990`), `derive(PipelineConfig)` (статический профайлер, `afc5fab`),
  `derive(Pack)` (SoA-пакинг, `14bca2e`), `kernel`, `for_each_entity`.
- **Ядро — прочее**: lock-free SmartStore (`26bde36`), hot/cold split,
  тесты детерминизма Strong Confluence (`8ac94ca`), физика
  (Sweep-and-Prune, импульсный солвер, raycast за трейтом `PhysicsEngine`).
- **База аудио**: `AudioSource`/`AudioListener`, декодер symphonia,
  бэкенды cpal и Web Audio (`b2ed884`, `75b16ae`).
- **RenderBackend + Renderer3D**: forward-рендер, WGSL PBR
  (GGX/Smith/Fresnel/ACES), OpenPBR-материал (20 vec4, `05a823e`,
  `3c6bf0f`), трейт `RenderBackend` с фабрикой; MaterialX-парсер
  `.mtlx` → `OpenPBRMaterial`.

### Браузерный поворот (июль 2026)

- **WASM/WebGPU рендер `scene.ron` в браузере** — viewport scaffold +
  загрузчик (`e4723cf`, 24.07), рендер через `Renderer3D` (`d3e1350`,
  24.07); pixel-identical нативному эталону (проверено
  headless-скриншотами).
- **Редактор как статика `editor/` + REST**: HTTP-сервер на 3420
  (`GET /`, `GET /api/status`, `GET /api/events`, `POST /api/command`,
  раздача фронтенда); `cargo xtask editor` (`6585c5b`, 26.07).
- **Удаление нативного UI** (`29e3547`, 01.08): `crates/ui*`,
  `src/bin_blitz.rs`, форки `forks/` (blitz, boa_engine, icu_normalizer).
  IPC-типы `UiCommand`/`GameEvent` сохранены в `crates/editor-backend/src/ipc.rs`.

### Качество (август 2026)

- Baseline-аудит и конфиги: `deny.toml`, `rustfmt.toml`,
  `[workspace.lints]` (`7b529da`); снимки в `docs/quality/`.
- Полная чистка warnings: 349 clippy + 128 rustc + 4 high-уязвимости
  (quick-xml) → **0/0/0** (серия коммитов 01.08, от `6fb9d0c` до
  `a69c4c5`).
- **`cargo xtask quality`** — единый гейт уровня 1/2: fmt, clippy
  `-D warnings`, test, audit, deny, outdated; `--full` добавляет
  llvm-cov и bench compile-check (`03450b6`).
- **proptest** — 15 свойств ядра; нашёл 2 реальных бага
  generation-семантики в `ComponentStore` (insert/remove), оба
  исправлены (`791b6d9`). Тесты: 162/162.
- **Фаззинг парсеров** (scene.ron, .mtlx) — cargo-fuzz scaffolding,
  оба таргета по 100 прогонов без крашей; **cargo-mutants**
  инфраструктура (777 мутантов в core, `dc75cd5`).
- **CI** `.github/workflows/quality.yml`: quality / wasm-check /
  supply-chain (`cfa482d`).
- Консолидация документации: README + PLAN + IDEAS, архив удалён
  (этот коммит). Покрытие строк: 45.77% — историческая базовая точка;
  актуальное значение — 83.32% (см. `docs/quality/coverage-2026-08-25.md`).

## 🟡 Частично — что достроить

- **Логический `World` + frame host (foundation)**: `ornis_core::World`
  объединяет `Resources` и авторитетный `SmartStore`, а `ornis_core::Engine`
  добавляет `Time`/`FixedTime`, два rate-плана и `run_frame` поверх
  `Schedule`. Editor-only facade уже использует этот host и physics
  sync/step/sync-out; native и WASM render используют общий library-level
  `RenderWorld`/`RenderExtract`/`FramePlan` contract. Backend-neutral
  `InputState` уже публикуется Engine и native winit/WASM orbit adapters его
  заполняют; native showcase и editor-only physics подключены к общему
  bounded 60 Hz fixed host, browser gameplay consumers и полный cross-domain
  runtime пока не завершены.
- **Remote Editor**: REST, обработчик команд в `editor-only`, `GET /api/scene`,
  подключение `editor.js` к `/api/scene`/`/api/status`, generic
  `set_component`, а также save/load сцены — ✅. Рендер получает живые
  snapshot'ы через polling; WASM отбрасывает устаревшие версии до применения,
  API возвращает explicit `request_id`/`accepted` ACK, а engine emits
  correlated `CommandCompleted` events; snapshots получают transport
  `sequence`; `/api/events?after=<sequence>` даёт bounded replay и `EventGap`.
  WebSocket upgrade на `/api/events` реализован для server-push; editor UI
  предпочитает его и сохраняет cursor polling как fallback.
  Editor-only `EditorWorld` использует `ornis_core::World`,
  а браузер восстанавливает snapshot в отдельном `RenderWorld` после
  serialization boundary — общей памяти между ними нет.
- **Command-Based Sync**: CPU-очередь + residency tracker и базовое
  GPU-исполнение (compute dispatch + flush, есть тест) — есть;
  автоматической data residency «копировать только при необходимости»
  (SmartBuffer) — нет.
- **Линтер параллелизуемости**: `#[smart_pipeline]` выдаёт предупреждения
  через deprecated-note трюк; IDE-интеграции и расширяемого набора правил нет.
- **WASM**: viewport получает актуальные scene snapshot'ы из `/api/scene`
  через polling (~1/с), имеет fallback на `scene.ron` и orbit-камеру; каждый
  snapshot восстанавливается в общий для native/WASM library-level
  `RenderWorld`, проходит `Engine::run_frame`/`RenderExtract` и записывается
  через `RenderFrame3D`/`FramePlan`. Ввода из браузера обратно в движок и
  WebSocket-синхронизации пока нет.
- **Component Packing**: `#[derive(Pack)]` генерирует wrapper-ленты и
  `Pack::for_each_packed`; wrapper-ленты совместимы с `for_each_entity!`,
  интеграция покрыта `crates/core/tests/pack_integration.rs` — ✅.
- **GPU-физика (G7)**: single-point контакты решаются через GPU velocity
  path, после чего проходят общий CPU NGS position solve; тест
  `gpu_solver_tracks_cpu_engine` подтверждает согласованное состояние с
  CPU-путём в заданном допуске на Metal и lavapipe. GPU-путь — Jacobi/GS
  hybrid и осознанно не bit-identical CPU-пути; для реиграемых сцен нужен
  CPU-only или отдельный fixed-point путь. Остатки G7: angular sweep в CCD,
  per-iteration interleaving joints/contacts и полная масштабируемость
  broadphase.

## Дорожная карта (по приоритетам)

### a. Протокол движок ↔ редактор

✅ Реализованы обработчик команд в `editor-only`, `GET /api/scene`,
подключение `editor.js` к REST, generic `set_component`, сохранение/загрузка
сцены, explicit command ACK (`request_id`/`accepted`), коррелированные
`CommandCompleted` events и transport sequence для snapshots. `/api/events`
поддерживает bounded replay по cursor и сообщает об eviction через `EventGap`;
WebSocket server-push реализован; connection handles отслеживаются, idle
соединения получают heartbeat ping, shutdown отправляет normal close. Polling
остаётся fallback для старых proxy/server окружений, чтение client close frames
пока не завершено.

### b. Живой ECS в браузере

✅ WASM-canvas рендерит актуальные snapshot'ы editor-only ECS через
`/api/scene`, имеет fallback на `scene.ron` и orbit-камеру. После границы
serialization snapshot восстанавливается в `ornis_render::RenderWorld`,
где `Engine` запускает общий `RenderExtract`, публикует `InputState`, а
`RenderFrame3D` исполняет `FramePlan`. Orbit pointer/wheel input уже проходит
через этот ресурс и scheduled `OrbitCamera` consumer. Общий `Engine` уже
предоставляет `FixedTime` и
bounded fixed schedule; WebSocket server-push уже добавлен. Остаются
browser/gameplay consumers и полный cross-domain runtime; серверный
`EditorWorld` и browser-side copy намеренно не делят память.

### c. Фаза 6 — Скриптинг (пересмотрена 2026-08-22, решение D1 аудита)

Не «лесенка языков», а плагинный шов + первый адаптер
([`docs/quality/audit-2026-08-22.md`](docs/quality/audit-2026-08-22.md),
решения F0/D1):

1. **Реестр компонентов (F0, фундамент)** — имя ↔ TypeId ↔
   type-erased операции (insert/get/remove в `SmartStore`, serde JSON
   в/из компонента); только tooling-пути (FFI/редактор/сериализация),
   горячие циклы остаются типизированными.
   🟡 **2026-08-22**: реализовано в `crates/core/src/registry.rs`
   (`ComponentRegistry`/`ComponentMeta`; thunk'и мономорфизирует
   generic-регистрация — процедурный макрос не понадобился,
   derive-сахар опционально позже); юнит-тесты + doc-пример.
   Верифицировано последующим CI quality-прогоном; generic-регистрация
   остаётся tooling-путём, горячие циклы не затрагивает.
2. **`ScriptEngine`-трейт** — третий плагинный трейт рядом с
   `PhysicsEngine` и `RenderBackend`: ядро знает только трейт
   (load/call/batch/hot reload), языки — адаптеры.
3. **Batch API по хендлам** (`engine.batch_add(...)` — один вызов
   вместо 100k): хендлы вместо прямых указателей на ячейки Sparse
   Set — указатели совместимы только с in-process FFI и ломают
   WASM-сценарий (sandbox).
4. **Первый адаптер — Rhai** → hot reload → прочие языки (Rune,
   Python/rustpython, WASM-компоненты) отдельными адаптерами после
   проверки шва минимум двумя реализациями (правило трёх); WASM —
   исследовательский трек (языко-нейтральность + sandbox).

### d. Фаза 7 — Asset Pipeline (браузерная интерпретация)

Hot reload сцен/мешей/`.mtlx`: сервер следит за `assets/` и `editor/`
(`notify`), фронтенд перезагружает сцену через REST/WebSocket.
Build-time генерация бинарных слепков для Sparse Sets — позже.

### e. Качество (продолжение)

Покрытие workspace уже **83.32%** (см.
`docs/quality/coverage-2026-08-25.md`; 45.77% от 2026-08-01 — историческая
базовая точка). Полный прогон mutants для `ornis-core` завершён с
98.8% mutation score среди тестируемых мутантов; для physics зафиксирован
первый большой прогон и отдельные T8–T14 ограничения. Остаются ночные
fuzz-прогоны (растить corpus, краши → регрессионные тесты),
criterion/performance-профилирование и flamegraph/dhat на perf-спринте.

### f. Дальше по старому плану

Deferred/Forward hybrid рендер и B1-R7 уже реализованы; подробности и
пиксельные проверки записаны в [`docs/rendering/render-graph.md`](docs/rendering/render-graph.md).
Остаются NUMA-aware allocation → кроссплатформенные прогоны
(Linux/Windows CI, miri) → адаптеры Rapier/Jolt за `PhysicsEngine` →
документация API и релизная упаковка.

### g. Unified Scheduler (IDEAS §28, долгосрочно)

Эволюция render graph в «третий путь» (scheduler как у Bevy + lifetime/aliasing
как у Frostbite): кеш `GraphLayout` → пасс как типизированная система →
бюджет памяти как ограничение → один scheduler без extract-фазы. Не
вытесняет приоритеты a–d: первые этапы (S0–S1) дёшевы и могут идти
параллельно с «a» — **S0–S6: см. Приложение C** (S1–S6 ✅; Приложение C
закрыто целиком).
**Контракт шедулера** (5 правил: доступы-данные, единый
`compute_levels`, тайбрейк регистрацией, `order_before` только разбивает
уровни, новый фронтенд лишь со вторым потребителем) —
`docs/rendering/unified-scheduler.md`. Пост-S5 hardening (2026-08-21):
принуждение объявленных доступов в debug-сборках, кеш уровневого плана с
битсет-конфликтами (`ornis_core::SystemAccess`), мягкий `try_order_before`
(`OrderError`) в ядре и графе. **Веха интеграции** (после S6,
вместе с приоритетом «a»): кадр через верхний `Schedule` над
`Resources`-миром (`Res<Device>`/`Res<Queue>`/время); физика и рендер —
системы-домены (внутренние острова/уровни — внутренность); критерий:
главный цикл (натив и wasm) исполняет render-кадр через `Engine`, а
render extraction остаётся явной переходной boundary-стадией до полного
cross-domain scheduler. Общий `FixedTime`/fixed schedule уже является
host-level orchestration для подключённых physics systems; полноценные
системы gameplay и остальные cross-domain consumers ещё впереди.
Physics живёт в production editor-only и native showcase циклах;
browser-side physics остаётся за serialization boundary. Полный unified
runtime без отдельной extract-фазы — будущая цель, не текущий статус.

> **Прогресс 2026-08-28:** `ornis_core::Engine` с `Time`/`FixedTime`/`InputState`
> исполняет native showcase и browser-side `RenderWorld` frames; общий
> backend-neutral `RenderExtract` находится в `ornis-render`, а native и WASM
> используют `RenderFrame3D`/`FramePlan`. Engine bounded fixed schedule теперь
> является общим host-level accumulator'ом; editor-only `EditorWorld` и
> native showcase исполняют physics sync/step/sync-out через него. Native
> winit и WASM orbit adapters записывают input в ресурс. Остаются полноценные
> gameplay consumers, расширение orchestration на остальные домены и единый
> cross-domain runtime; serialization boundary между сервером и браузером
> сохраняется намеренно.

## ❌ Не делать / отложено (решения владельца)

- **Нативный UI** — удалён (`29e3547`): доведение собственного
  UI-движка до production сопоставимо с командой браузерного движка;
  редактор живёт в браузере. Не возвращаться.
- **HVM2/Bend как compute-бэкенд** — идея на далёкое будущее,
  активной работы нет.
- **Формальная верификация** — отложено бессрочно; вместо неё
  proptest + mutants + fuzz.

---
## Приложение B — Рендерер и физика: план работ

> Архивный план, сверенный с реализованными фазами B1-R7 и G1-G7;
> незавершённые пункты явно отмечены ниже.

### B1. Рендерер (`crates/render`) — план

- **R1.** Решить судьбу G-buffer/Lighting-путей: в коде есть структуры
  `GBufferTextures`/`LightingPass`/`CompositePass`, но README помечает
  Deferred/Forward hybrid как ❌. Либо довести gbuffer-путь до активного
  (G-buffer → lighting pass → composite), либо удалить как спекулятивный
  каркас. Быстрый шаг: поставить README-статус точно по коду.
  - **→ Решено (2026-08-10, design note):** доводить gbuffer-путь
    до активного гибрида, не удалять. Способ — render graph (см. R7).
    README-статус «Deferred/Forward hybrid» актуализировать по коду
    (гибрид уже работает императивно).
- **R2.** `Renderer3D` под живой ECS: поднять бюджет с `max_objects=256`,
  перейти на инстансинг через `InstanceData`, поддержать динамические
  меши/трансформы без пересоздания буферов.
- **R3.** Материалы: масштабировать OpenPBR-бюджет (`max_materials=64`),
  вынести параметры в общий descriptor set, готовить пайплайн под
  переключение материалов без rebind.
- **R4.** Свет: сейчас только направленные (до 4, `GpuLight`) — добавить
  point/spot + shadow maps (когда понадобится).
- **R5.** Довести backend abstraction до plan-capable API: native и WASM
  уже используют один `Renderer3D` + `RenderFrame3D` путь, а
  `RenderBackend::render_scene` сохраняется как compatibility/plugin API и
  reference path без дублирования pass logic.
- **R6.** Связать рендер с ECS-сценой в браузере: snapshot уже
  восстанавливается в shared library-level `RenderWorld` и проходит
  `Engine`/`RenderExtract`/`FramePlan`; остаются ввод/камера из браузера в
  движок и live event transport (перекликается с «b. Живой ECS в браузере»).
- **R7.** Render Graph — слой оркестрации пассов (design/implementation note,
  2026-08-10). Гибрид forward/deferred уже работает императивно в
  `renderer.rs` (gbuffer 5 MRT → lighting → forward → composite); render
  graph делает выбор техники конфигурацией графа. Оговорка wgpu: барьеры
  и layout transitions wgpu делает сам — берём lifetime ресурсов
  (gbuffer → transient), пул текстур, culling пассов; memory aliasing на
  уровне драйвера недоступен (пул объектов вместо него). План — лёгкий
  immediate-граф (Frostbite/Ponies&Light), фазы:
  0 — каркас `render_graph` (юнит-тесты на lifetime ресурсов), 1 —
  обёртка 4 существующих пассов как узлов (попиксельная проверка через
  `render_probe`), 2 — gbuffer в transient-пул (замер пиковой памяти GPU
  через profiler), 3 — первый новый узел (SSAO/блум), 4 — переключатель
  forward/hybrid/deferred как конфигурация графа.
  Разбор: [`docs/rendering/render-graph.md`](docs/rendering/render-graph.md);
  идея: IDEAS.md §27.
  **Статус (2026-08-11, подтверждено ревью): фазы 0–4 выполнены** —
  каркас `render_graph.rs` + исполнитель `graph_frame.rs` (`GraphExecutor`,
  `RenderGraph3D`; 4 пасса как узлы, pass-тела на `GbufferTargets`).
  Фаза 2: `texture_budget()` — legacy 8 постоянных текстур vs пул 7
  слотов, −20,0% на 1280×720; мёртвые depth-текстуры удалены; пул
  стабилен 16 кадров. Верификация `render_graph_probe`: legacy vs graph
  пути пиксельно идентичны, 9 ресурсов → 7 слотов пула.
  Фаза 3 — блум как первый новый узел: каскад down0(½, bright-pass 0.7)→
  down1(¼)→down2(⅛)→up1→up0 (ADD поверх Load), composite смешивает bloom
  перед tonemap; bloom-off пиксельно = legacy, bloom-on: 267 103 px
  изменены, слоты 7→10 (+3 уровня), бюджет +3,8 MB. Фаза 4 —
  переключатель техники как конфигурация графа: `Technique { Forward,
  Deferred, Hybrid }` → `RenderGraph3D::new_with(format, size, technique,
  bloom)`; узлы gbuffer/lighting/forward добавляются по флагу,
  composite-шейдер смешивает по `mode` uniform (0/1/2), в forward-only
  пасс сам владеет depth, блум читает живую HDR-текстуру. Probe:
  deferred-only == legacy (0 отличий), forward-only: 137 164 px отличий,
  2 слота / 10,6 MB (−62%), свой блум активен; hybrid без регрессий.
  Тесты 30/30, quality PASS. Статус-описание: §9
  `docs/rendering/render-graph.md`.

### B2. Физика (`crates/physics`) — план работ

> Архитектура сверена с реальными исходниками **Box3D** (`github.com/erincatto/box3d`,
> soltimeer 3D-преемник Box2D, Catto, июнь 2026) и **Jolt** (`github.com/jrouwe/JoltPhysics`):
> multisolver = joint-солверы + широкий (SIMD) контактный солвер + manifold-солвер,
> стадии WarmStart→Solve→IntegratePositions→Relax→Restitution, warm starting,
> split impulse (velocity/position раздельно), constraint graph → острова + sleeping,
> speculative CCD, мягкие констрейнты.
> В текущем коде G1-G7 уже реализованы: orientation, manifolds, split impulse,
> islands/sleeping, joints, linear CCD/shapecast, CPU/SIMD/GPU paths. Следующий
> практический риск — масштабирование broadphase и оставшиеся ограничения G7;
> исторические описания стартового состояния ниже сохранены как хронология.
>
> Отдельная сверка broadphase с актуальными исходниками Box3D/Jolt показывает,
> что следующий риск — не только подбор `UniformGrid` cell size: зрелые
> реализации используют persistent proxies, fat AABB, static/dynamic layers и
> active/moved-body queries. Зафиксированные выводы и ссылки:
> [`docs/quality/broadphase-reference-2026-08-29.md`](docs/quality/broadphase-reference-2026-08-29.md).

| Этап | Что делаем | Основание (реальный код) | Статус |
|---|---|---|---|
| **G1** | **Ориентация + угловая динамика**: `RigidBody` с `orientation: Quat`, `angular_velocity`, инерцией (тензор), интеграция (semi-implicit Euler) + вращение; коллизия с учётом ориентации (sphere↔OBB, OBB↔OBB по SAT, капсула по оси тела) | Box3D `b3BodyState` (linear+angular velocity), Jolt `Body` (инерция) | ✅ Реализовано |
| **G2** | Контактные манифолды (несколько точек на пару) + кэш + warm starting. **G2a**: структуры (до 4 точек), OBB↔OBB vertex-face манифолд, солвер по точкам с warm-start кэшем импульсов. **G2b**: стабильность покоя — отдельная стадия WarmStart, накопление импульсов, внешние итерации по всем манифолдам, стабильный 4-точечный манифолд. | Box3D `b3ContactConstraintWide`/`ManifoldConstraint` (симметрический `cached_manifold`), Jolt `mContactPoints` | ✅ G2a+G2b реализованы |
| **G3** | Split impulse: раздельные velocity/position проходы + мягкие констрейнты (`b3MakeSoft`-аналог) | Box3D стадии `IntegrateVelocities`/`IntegratePositions`, `b3Softness`; Jolt `ContactConstraintPart` | ✅ Реализовано (остаток по стекам ≥4 закрыт в G4) |
| **G4** | Constraint graph → острова + sleeping (awake/static кэш). **+ Block solver (b2Solve22-аналог) для точек манифолда**: скалярный PGS + NGS расходится в rocking-моде на длинных цепочках (измерено в G3) — лечится одновременным решением связанных точек и островно-кохерентным sleep/wake (per-body sleeping уже есть из G3) | Box3D `b3ConstraintGraph`/`b3SolverSet`/`sleepVelocity`, Jolt islands | ✅ Реализовано |
| **G5** | Joints: ball (spherical), revolute; через под-солверы | Box3D `spherical_joint`/`revolute_joint`, Jolt `ConstraintPart/*` | ✅ Реализовано |
| **G6** | CCD (speculative контакты + TOI `b3SolveContinuous`) и честный `shapecast` | Box3D `B3_SPECULATIVE_DISTANCE`, `b3SolveContinuous` | ✅ Реализовано |
| **G7** | Производительность: wide/SIMD-контактный путь + параллельные таски (граф-раскраска, CAS-блоки). **GPU-перенос wide-контактного солвера через существующий `CommandSync.dispatch_gpu`** (паттерн «команды туда, где живут данные», AABB/контакты в residency `GpuOnly`) — осмысленно только после G4 (острова = единица независимого dispatch; sleeping сокращает сам объём работы). Требует осознанного решения про детерминизм: GPU float не даст bit-identical результатов CPU-пути (конфликт с Strong Confluence; варианты — GPU только для визуально-массовых тел или fixed-point) | Box3D `solver.h` (WideContact, per-block syncIndex, bepu-inspired) | ✅ CPU (острова+rayon, сон) + SIMD-wide (`wide.rs`) + GPU (`gpu.rs`, feature `gpu`). Детерминизм: CPU-путь остаётся Strong-Confluence; GPU — опт-ин Jacobi/GS hybrid, не бит-идентичен CPU (документировано) |

**Качество на каждом этапе (обязательный гейт):** сценарии (стек из N бокстов стоит N сек без дрифта; сфера спокойно покоится; sleep/пробуждение корректны), proptest-инварианты (после settle нет проникновения; сохранение импульса; детерминизм `RAYON_NUM_THREADS=1` vs `=32`); fuzz (нет NaN/паник); mutants на солвер. Согласуется с философией Strong Confluence.

**Логика порядка**: G1 (ориентация) — фундамент, без него невозможны ни joints, ни настоящие манифолды; G2-G3 — стабильность/качество; G4 — производительность; G5-G6 — поверхность; G7 — peak-производ.

**G1 (2026-08-07, реализовано; остатки дочищены в тот же день)**: добавлены `orientation: Quat`, `angular_velocity`, инерция (диагональный тензор через `Shape::inertia`), `torque`; полуимплицитная интеграция линейной + угловой динамики (вращение кватерниона через `from_scaled_axis`); коллизии с учётом ориентации: sphere↔OBB (в локальном фрейме), OBB↔OBB (полный SAT по 15 осям), капсула по оси тела, **sphere↔capsule**; resolve: угловой импульс (world-инерция через `mul_inv_inertia`) **и вращательная позиционная коррекция** (эффективная масса c rotation + slop/релаксация). **`shapecast` реализован** (консервативный свип пере-юзом узкой фазы, 64 шага). Тесты: 11 (к прежним +4 добавлены `sphere_capsule_collision`, `shapecast_hits_body`). **Остаток → G2**: пара box↔capsule (и капсула↔бокс) всё ещё `None`.

**G2b (2026-08-08, решён)**: бокс на статичном полу улетал (энергетическая инъекция, y≈1020 за 4 с). Оказалось три наложенных дефекта. (1) **Warm start без отдельной стадии**: кэшированный импульс никогда не применялся к скоростям, но прибавлялся к аккумулятору каждый проход — накопление росло бесконечно. Исправлено паттерном Box2D: стадия `WarmStart` (применить кэш один раз) + velocity solve с накоплением от уже применённого, 8 итераций, реституция как velocity bias (с порогом 1 м/с), накопленное трение с клампом ±µ·λn. (2) **Нестабильный манифолд**: углы обоих боксов давали до 8 кандидатов на 4 контактных зоны, дедуп по 3D-близости их не склеивал, набор точек мигал кадр к кадру — теперь dedup в касательной плоскости (4 стабильные точки), а warm-кэш хранит мировые координаты точек и матчится по близости (≤5 см, one-to-one), а не по индексу. (3) **Выпадение манифолда при микро-наклонах**: строгий тест «угол внутри бокса» (eps=1e-3) при наклоне ~0.002 рад отбрасывал все 8 углов → fallback в 1 центральную точку → бокс вставал на ребро и раскачивался; SAT при микро-наклонах выбирал шумные edge-edge оси. Исправлено: speculative margin (глубина до −0.01) + касательный slack 0.05 в генерации точек, face-preference (1e-3) в SAT — edge-ось побеждает только с заметным запасом. Также: внешние Гаусс-Зейдель итерации по всем манифолдам (раньше — вложенные per-manifold, стек не сходился), инвалидация warm-кэша при `remove_body` (swap_remove сдвигает индексы). Тесты: `box_rests_on_static_floor` разблокирован, добавлены `sphere_rests_on_static_floor` и `two_box_stack_stays_stable` (стек 2 боксов стоит 5 с без дрифта). 15/15, clippy/fmt чисто. Осталось: box↔capsule → G2c.

**G3 (2026-08-08, 🟡 реализовано с остатком)**: split impulse по мотивам Box3D-стадий `Solve`/`IntegratePositions`. Сделано: (1) **Якорный NGS**: позиционный проход хранит body-frame якоря (la/lb) и глубину на момент детекции (pen0) по каждой точке и итерируемо (4 прохода, β=0.2, slop 0.02, cap 0.25) перемеряет ЖИВОЕ separation — коррекции распределяются по всему набору манифолдов, а не одноразовыми толчками; `make_soft(k,cfm)` оставлен как точка расширения, но для контактов CFM=0 — мягкость позиционного прохода сама оказалась резонатором (измерено бисекцией). (2) **Реституция вынесена в одноразовую стадию** (аналог `b3SolverStage_Restitution`): bias = −e·vn0 только на первом подшаге шага, только для НЕсматченных точек, vn0 < −1.0, pen0 ≤ 0.05; импульс не накапливается и не warm-стартится — persistent-контакт никогда не ре-реституирует (иначе NGS подпитывает отскок — накачка). (3) **Кап warm-импульса**: `min(warm, −vn_pre/k_eff)` — устаревший кэшированный импульс в разделяющийся контакт был чистой инъекцией энергии 480 раз/с. (4) **Трение в фиксированном касательном базисе** (t1,t2 от нормали, круговой кламп к µ·λn) — пересчёт касательной по мгновенной скорости проскальзывания «гулял» и водил стеки вбок. (5) **SAT-антиблик**: порог вырождения edge-edge осей 1e-6→1e-3 (при 1e-6 f32-шум давал ложный вердикт «separated» на микро-наклонах — манифолд мигал на 1 подшаг, warm-кэш сбрасывался, тело в свободном падении набирало −0.02 м/с за цикл — тихая накачка), плюс спекулятивный запас 2e-3 на вердикт разделения. (6) **8 подшагов** (было 4) — удар наклонного бокса о ребро со скалярным PGS не сходился (λ≈11.7 при норме ~0.1 концентрировался на одной точке и раскручивал тело до w≈400); на 8 подшагах бокс ложится плашмя идеально. (7) **Sleeping (перенесён из G4)**: тело со скоростями < 0.15 м/с и < 0.15 рад/с в течение 0.5 с засыпает (пропуск интеграции и контактной работы), будится контактом с бодрствующим телом. Это штатный ответ Jolt/Box3D на микроджиттер покоящихся стеков — у нас тоже сработал: `two_box_stack` зелёный. Тесты: 16 passed + `tilted_box_falls_flat` (бокс под 20° ложится плашмя) + `two_box_stack_stays_stable`; clippy/fmt чисто. **Остаток → G4**: `four_box_stack_stays_stable` помечен `#[ignore]` — в цепочках ≥3 контактов скалярный PGS + NGS расходится в rocking-моде (показано: без NGS накачки нет, демпфирование β/warm/CFM не лечит, e=0 ограничивает амплитуду ~0.05) — нужен block solver по точкам манифолда и островно-кохерентный sleep/wake.

**G4 (2026-08-09, реализовано)**: constraint graph + острова + block solver. Сделано: (1) **Block solver нормалей** (аналог b2Solve22 из Box2D/box3D-подхода «решай связанные точки вместе»): для манифолда с ≥2 точками собирается K-матрица манифолда (`inv_mass` + угловые члены через `mul_inv_inertia`) и решается точный LCP перебором активных множеств по убыванию размера (маски до 2^count, K_S·acc'_S = −vn_S + K_{S,all}·acc, валидность: acc' ≥ 0 на активных, vn' ≥ 0 на неактивных); системы ≤4×4 — Гаусс с частичным пивотом (`solve_small`, None при сингулярности < 1e-12), при отказе — fallback в скалярный PGS. Это убило взрывную накачку 4-стека: раньше точечный PGS по очереди качал энергию между 4 точками опоры (rocking-мода), теперь они сходятся совместно. (2) **Острова**: union-find по dynamic-dynamic контактам последнего подшага (`island: Vec<u32>`, статика = u32::MAX). Ключевой инвариант — **заморозка спящих островов**: пары с обоими asleep и одинаковым старым island union'ятся принудительно, иначе блик детекции на 1 подшаг растворял остров и будил его по частям. (3) **Островно-кохерентный sleep/wake**: остров засыпает, только когда ВСЕ бодрствующие члены < 0.15 м/с и < 0.15 рад/с в течение 0.5 с (засыпание обнуляет скорости); `wake_island` будит целиком. (4) **Фикс пробуждения от статики**: спящие пары (и пары с не-dynamic) пропускаются в солвере, `wake_island` вызывается только от бодрствующего DYNAMIC партнёра — до фикса пол «будил» спящие тела каждый подшаг и солвер крутил импульсы в спящее тело («спит со скоростью −0.226»). (5) **12 подшагов** (было 8) — дожало остаточный limit-cycle джиттер верха 4-стека (|v| ≈ 0.16 держал остров над порогом сна; на 12 подшагах стек успокаивается под гейт < 0.08). Тесты: `four_box_stack_stays_stable` разблокирован и зелёный (300 кадров, высоты и дрифт в допусках), 17/17, clippy/fmt чисто. Остаток: islands пока используются для sleep/wake, но не для параллелизации солвера (per-island dispatch) — это материал G7 (вместе с GPU-переносом wide-контактного пути через `CommandSync.dispatch_gpu`).

**G5 (2026-08-15, реализовано)**: joints (ball/revolute) через под-солверы. Сделано: (1) **API**: `JointKind::{Ball, Revolute}` (новый модуль `joint.rs`), `add_joint(body_a, body_b, kind) -> Option<JointHandle>` (отклоняет self-joint и невалидные хендлы, нормализует оси шарнира на создании, будит спящий остров), `remove_joint`; методы добавлены в трейт `PhysicsEngine`. (2) **Ball joint**: 3 линейных equality-констрейнта по мировым осям на якорях (point-to-point, аналог `b3SphericalJoint`): velocity-проход с накоплением без клампа (equality), позиционный проход — Baumgarte β=0.2 с cap 0.25, общий с контактами. (3) **Revolute**: ball + 2 угловых equality-констрейнта по осям ⊥ шарниру (t1/t2 от мировой оси тела A, пересчёт каждую итерацию); вращение вокруг оси свободно. (4) **Warm start** из аккумуляторов прошлого подшага (тот же паттерн, что контактный кэш G2b) — и линейный, и угловой. (5) **Джойнты — рёбра constraint graph**: входят в union-find островов (jointed-пара спит и будится вместе), `remove_body` дропает зависимые джойнты и ремапит индексы swap_remove. (6) Джойнт-солвер чередуется с контактным на гранулярности подшага (12 подшагов ≈ 720 Гц); истинный per-iteration interleaving контактов и джойнтов — осознанно отложен. **Найдены и исправлены два бага**: (а) **латентный баг G1 в `mul_inv_inertia`**: тензор применялся как Rᵀ·v·I⁻¹ без обратного поворота R — правильно I⁻¹_world = R·I⁻¹_body·Rᵀ; был невидим, потому что все тесты до G5 использовали изотропную инерцию (кубы/сферы: I = c·E инвариантно к повороту), а revolute-рукоятка 0.1×2×0.1 анизотропна — баг давал нестабильность вплоть до NaN; (б) **знак позиционной угловой коррекции**: δ, выравнивающий wb на wa, равен −(wa×wb), а не +(wa×wb) (тройное произведение: (wa×wb)×wb = wb·cosθ − wa) — с неверным знаком коррекция превращалась в экспоненциальную накачку (~180° за 300 кадров). Тесты: +4 (`ball_joint_pendulum_holds_anchor` — маятник качается, якорь не расходится > 0.05; `ball_joint_chain_hangs` — цепь из 3 звеньев висит; `revolute_hinge_rotates_about_axis_only` — шарнир вращается о Z, наклон по X/Y < ~1°, якорь на месте; `remove_body_drops_dependent_joints`). 21/21, clippy/fmt чисто. Остаток → G5b (не в плане, по потребности): лимиты/мотор шарнира (`b3RevoluteJoint` limit/motor), per-iteration interleaving с контактами, мягкость джойнтов через `make_soft`.


**G6 (2026-08-15, реализовано)**: CCD (speculative контакты + TOI) и честный `shapecast`. Сделано: (1) **Модуль `distance.rs`**: точные аналитические расстояния между всеми парами фигур (сфера/бокс/капсула) — сознательно НЕ GJK: у нас всего 3 типа фигур, а для OBB точное расстояние получается перебором фич (16 vertex-face + 144 edge-edge кандидата) без итераций и без проблем сходимости GJK на гранях. (2) **`shapecast` через conservative advancement** (`cast_shape`): итеративный свип по точному расстоянию (TOUCH=1e-3, до 24 итераций, ход = dist − TOUCH/2), туннель-фри для стен любой толщины; `CastHit.t` — АБСОЛЮТНАЯ дистанция вдоль свипа, не фракция (путаница фракция/абсолют уже дала один красный тест). (3) **Speculative контакты** (аналог `B3_SPECULATIVE_DISTANCE`): всем функциям узкой фазы добавлен параметр `margin` (per-pair = SPEC_BASE 0.05 + rel_speed·sub_dt), penetration больше не клампится к ≥0 (отрицательная = зазор); `SweepAndPrune::update` строит swept AABB для dynamic тел + инфляцию всех AABB на 0.025; в солвере у точек с зазором velocity-target = pen0/sub_dt (импульс ровно на «долететь до касания за подшаг», не больше — и в скалярном пути, и в block solver, и в warm-cap), реституция пропускает speculative-точки, которые в этом подшаге не долетят. (4) **TOI-проход `solve_continuous`** (линейный аналог `b3SolveContinuous`): тело с предсказанным смещением за подшаг > половины минимального размера кастится вдоль смещения и клампится к первому удару (отступ 1e-3), скорость корректируется (отскок при |vn| > 1 с min-реституцией пары, иначе гашение нормальной составляющей), тело помечается и не интегрируется повторно. Угловой свип в касте не учитывается (linear cast) — осознанный остаток. (5) **Перестройка стадий в порядок Box3D `IntegrateVelocities → Solve → IntegratePositions`** — ключевой структурный фикс этапа: раньше `step()` двигал тела и только потом решал констрейнты, т.е. каждый подшаг был «свободное падение → неупругий снап обратно», что при гравитации квадратично съедало энергию (измерено на маятнике); теперь гравитация гасится контактным импульсом ДО движения позиций, покой не «дышит», а джойнт-маятник держит энергию с точностью < 0.5% за 5 с (было: потеря ~87% за период). `integrate` разбит на `integrate_velocities`/`integrate_positions`, `solve_joints` — на velocity/position стадии, контактный солвер — на `solve_contacts_velocity` (возвращает per-manifold state) + `solve_contacts_position` (NGS после движения). (6) **Jointed-пары не коллидируют** (`joint_pairs: HashSet`, фильтр перед узкой фазой; Box2D `collide_connected = false` по умолчанию): палка шарнира на тангенциальном качании законно заметает углы внутрь корпуса опоры — без фильтра контактное трение работало фантомным тормом (это и была причина затухания маятника 2.5×/период, а не сам солвер); `remove_joint`/`remove_body` корректно обновляют набор пар. Тесты: +3 (`fast_sphere_does_not_tunnel` — сфера r=0.1 на −80 м/с против плиты 0.1 м: смещение/подшаг 0.111 > толщины, без CCD проходила насквозь, теперь садится на плиту; `shapecast_exact_hit_distance` — аналитическая дистанция 3.5 ± 1e-2; `shapecast_thin_wall_no_tunnel` — стена 4 см насквозь, хит на 1.88 ± 1e-2), `revolute_hinge_rotates_about_axis_only` сделан фазо-устойчивым (экстремумы качания вместо финального кадра) и заодно исправлена геометрия якоря (ошибка 1.0 м маскировала затухание до G6). 24/24, clippy/fmt чисто. Остатки: реституция в TOI упрощена (одноразовая, без учёта касательной); joints/contacts чередуются на гранулярности подшага, не per-iteration; angular sweep не учитывается.

**G7 (2026-08-16, полностью):** производительность.
- **CPU-часть**: замеры, фиксы сна, per-island параллельный солвер (rayon), детерминизм (25/25 тестов) — реализовано ранее (2026-08-15).
- **SIMD-wide контактный путь** (2026-08-16): `WideBatch` — SoA-пакет из ≤4 single-point контактов с дизъюнктными телами, решаемых лейнами в порядке глобального GS. Прекомпьютинг K⁻¹, матриц инерции, факторов применения; каждая лейна — скалярное арифметическое выражение, идентичное скалярному пути в пределах ±1 ulp (матричная инерция вместо кватернионных вращений). Встраивание: `build_solver_steps(bodies, manifolds, states) → Vec<SolverStep>` — жадная упаковка последовательных контактов с непересекающимися телами в батчи по 4; шаги исполняются в исходном порядке манифолдов. Реституция: `WideBatch::solve_restitution()` — одноразовый поститерационный пасс на тех же лейнах. Config: `set_wide_solver(bool)` (default true); отключение возвращает чистый скалярный путь для бит-точного воспроизведения. Бенчмарк: `cargo bench -p ornis-physics` на сцене spheres_grid (оценка SPEEDUP wide vs scalar). Тесты: `wide_batch_matches_scalar_single_point` (две независимые пары, точность 1e-4), `batch_groups_disjoint_consecutive_contacts`.
- **GPU-перенос wide-контактного солвера** (2026-08-16, обновлён 2026-08-17): модуль `gpu.rs` (`#[cfg(feature = "gpu")]`). Compute шейдер (4 лейны/workgroup, один батч/диспатч): нормальный + фрикционный импульс, реституция. Шейдер **написан на Rust**: `#[gpu_pipeline(...)]` генерирует WGSL из тела Rust-функции (bindings/builtins/workgroup size — в атрибутах макроса), `#[derive(WgslStruct)]` генерирует WGSL-структуры из Rust-раскладок и compile-time ассертит `offset_of!`/`size_of` против правил WGSL — рукописного WGSL в физике больше нет (идея №4 «CPU↔GPU из одного Rust-кода»). Заодно исправлены расхождения раскладок, накопленные при ручной синхронизации: явный паддинг после `vec3` (WGSL выравнивает `vec3<f32>` на 16 байт) и страйды буферов из `size_of::<T>()` вместо устаревшей константы 2144. `WgpuContactSolver` — wgpu-буферы (body state 32 байта/тело, batch 1248 байт/батч), пайплайн, `upload/download/solve` методы. Пэкер `pack_single_point_batches` — CPU-side упаковка состояний в GPU-батчи (та же стратегия дизъюнктных тел). Интеграция через `solve_contacts_velocity_gpu` (гибрид: single-point → GPU, multi-point → CPU острова, Jacobi/GS hybrid — не бит-идентичен CPU). Аттач: `BuiltinPhysicsEngine::set_gpu_solver(solver)`. Детерминизм: GPU-путь не бит-идентичен CPU (разные ассоциации fma/rounding); документировано. Тесты: `gpu_pack_produces_disjoint_batches` (чистый CPU), валидация сгенерированного WGSL через naga (без устройства), `gpu_solver_single_contact_matches_analytic` и `gpu_solver_tracks_cpu_engine` (прогон на реальном адаптере; на CI — mesa/lavapipe, без адаптера — skip). Зависимости: wgpu, bytemuck (feature gate `gpu`), ornis-macros; dev: naga, pollster. `cargo test -p ornis-physics --features gpu` + clippy — отдельные стадии гейта `cargo xtask quality`.
- **Gather/scatter**: инструкции `Vec::with_capacity` добавлены; пулинг аллокаций не реализован (тривиально, даёт <1% прироста).
- **High-stack rocking** (качество, не производительность): 32-стек подрагивает, 24/33 спит — остаётся как issue для будущей работы (физически корректно, визуально приемлемо).
- **Извлечение хелперов**: `build_manifold_state` (единый преамбул), `partition_into_islands`, `dispatch_islands_velocity` — переиспользуются CPU и GPU путём. Итог: 25 тестов, clippy/fmt/AST проверено (в среде без Rust toolchain — код написан, полная компиляция при следующем `cargo test -p ornis-physics`).

**Physics API follow-up (2026-08-28):** `RigidBody` теперь имеет
взаимную фильтрацию `collision_layer`/`collision_mask`; broadphase,
narrowphase и linear CCD не создают пары для несовместимых фильтров.
Triggers имеют `Entered`/`Exited` events и не применяют solver/CCD impulses.
Raycast теперь использует точные sphere/OBB/capsule intersections с
корректными surface normals. Angular CCD получил bounded sweep по углу для
box/capsule с binary search первого sampled impact; fully analytic
swept-volume TOI остаётся дальнейшим улучшением.

**Broadphase decision boundary (2026-08-28):** текущий Sweep-and-Prune
остаётся default baseline/fallback. В коде появился opt-in
`BroadPhaseKind::UniformGrid`: deterministic grid candidate pairs,
static/dynamic cell decomposition, large-body escape path и layer/mask
filtering до narrowphase. Dynamic AABB tree остаётся вторым кандидатом для
sparse/heterogeneous worlds. Production default откладывается до матрицы
1k/10k/100k тел, giant floor/tiled floor, sparse world, dense islands и
worst-case broadphase.

**Exploratory benchmark (2026-08-28):** workflow run `33194136814`
(head `4fa10c0813f264d9df7c1b1d66002297ea9c5d28`) показал, что UniformGrid
и Sweep-and-Prune практически равны на 1k тел (1.4075 vs 1.4029 µs), но на
10k tiled-floor тел UniformGrid быстрее примерно в 3.87 раза (288.58 ms vs
1.1167 s). CPU/runner metadata и `rustc` version в raw log отсутствуют; 10k
использовал 10 samples и дал warning о длительном сборе, поэтому результат
фиксируется как направляющий, не как полный baseline. GPU path в этом
прогоне не участвовал. Абсолютные 288.58 ms всё ещё не соответствуют
real-time бюджету 16.7 ms.

**Диагностический benchmark-срез:** benchmark печатает `BroadPhaseStats`
(body count, raw pair tests, layer/mask rejections, static-static skips,
AABB rejections, unique candidates, occupied grid cells и large bodies) и
сравнивает UniformGrid с cell size 1.0/2.0/4.0/8.0/16.0. Это даёт
candidate-pair breakdown; отдельный timing broadphase против solver и 100k
probe для обоих backend'ов ещё впереди. UniformGrid пока остаётся opt-in
provisional candidate, а Sweep-and-Prune — default до этих измерений.

**Cell-size follow-up (2026-08-29):** workflow run `33240643444` на head
`7504d9bbe2b4d75fecb52efd14784f4aac2fdbd4` был остановлен общим лимитом job
в 60 минут, но присланный benchmark output содержит измерения всех шести
конфигураций. На 10k тел: SAP — `1.1130 s`, grid 1.0 — `469.99 ms`, grid
2.0 — `271.36 ms`, grid 4.0 — `196.69 ms`, grid 8.0 — `180.00 ms`, grid
16.0 — `198.59 ms`. `cell_size = 8.0` — лучший проверенный вариант,
примерно 6.18x быстрее SAP и на 8.5% быстрее `4.0`; `16.0` на 10.3%
медленнее `8.0`. На 1k все варианты остаются в пределах шума. Runner
metadata отсутствуют; targeted 100k Grid 8.0 уже измерен, но SAP на том же
probe ещё не запускался, поэтому production default не переключается.

Текущий результат меняет provisional tuning conclusion: для tiled-floor
сцены лучший проверенный cell size — `8.0`, но это end-to-end `step` время,
а не изолированный broadphase timing. Grid по-прежнему создаёт `14161`
candidate pairs против `11781` у SAP на 10k. Targeted 100k probes
`33245718111` (Grid 8.0) и `33251548032` (SAP) сравнили оба backend: около
`8.02 s/step` против `79.49 s/step` в steady state; на первом SAP step было
`5417936560` raw pair tests против `2349246` у Grid. Оба запуска используют
отдельные runner'ы и свежую 100k сцену, поэтому это exploratory comparison.
Далее нужны timing breakdown broadphase/narrowphase/solver и persistent
`DynamicAabbTree`; adaptive cell size пока не реализован.

---
## Приложение C — Unified Scheduler (IDEAS №28): план реализации

> **Актуальная оговорка (2026-08-27):** S0–S6 описывают завершённую
> инфраструктурную ветку scheduler/render-plan. `ornis_core::World` и
> `Engine` теперь являются frame host'ом для native и browser-side render
> extraction; `RenderWorld`/`RenderExtract` вынесены в `ornis-render`, а
> `RenderFrame3D`/`FramePlan` используются обоими render loops. Общий
> `Engine` fixed host уже подключает physics systems в editor-only и native
> showcase; полный cross-domain runtime с gameplay/input consumers всё ещё не
> подключён.

> План по [`IDEAS.md`](IDEAS.md) §28 («третий путь» между Frostbite
> FrameGraph и Bevy 0.19: один scheduler + автоматический lifetime/aliasing +
> типизированные ресурсы в сигнатурах). Режим «в долгую»: каждый этап —
> самостоятельный выигрыш, откатываемый и не зависящий от следующего (§28,
> «эволюционный путь»). Горизонты — оценки, не обязательства. Хронологическая
> таблица ниже отражает срез 2026-08-18; актуальная оговорка 2026-08-27
> находится перед C0. Приоритеты a–d (редактор, живой ECS, скриптинг, ассеты)
> остаются выше; S0–S1 дёшевы и могут идти параллельно с «a».

### C0. Отправная точка (снимок 2026-08-18)

> Это исторический срез до выполнения S1–S6. Актуальные статусы и
> оставшиеся пробелы указаны в таблице этапов и в блоке интеграции
> `World` ниже; старые имена файлов в этом срезе сохранены только
> для объяснения хронологии.

Из §28 уже частично есть (работает, протестировано):

| Компонент §28 | Что есть в коде | Где |
|---|---|---|
| Lifetime-окна ресурсов | `first_use..last_use` по enabled-пассам | `frame_plan.rs` (`FramePlan::compute_layout`) |
| Пул/aliasing непересекающихся ресурсов | greedy first-fit interval partitioning по слотам с равным `TextureSpec` | там же |
| Culling пассов | `set_pass_enabled`: disabled-пасс выпадает из layout, ресурсы не получают слотов | `frame_plan.rs` |
| Метрика памяти | `FrameExecutor::texture_budget()` — сумма байтов пула | `frame_exec.rs` |
| Пиксельная верификация | `render_probe` (legacy vs graph по техникам) | `crates/render/examples/render_probe.rs` |

Пробелы, которые закрывает план:

1. `build()`/`compute_layout()` выполняется **каждый кадр**
   (`graph_frame.rs:492` в `RenderGraph3D::render`; ещё `layout_dump` :452) —
   аллокации и сортировка на горячем пути (§28, шаг 1 эволюции).
2. Пасс — builder + **строки**: тело пассов — `match pass.pass().name.as_str()`
   в `RenderGraph3D::render` (`graph_frame.rs:472`), `unreachable!` на
   неизвестном имени. Read/write — рантайм-данные, а не типы сигнатуры.
3. Порядок пассов — порядок вставки; «read-before-write» проверяется
   паникой в рантайме (`compute_layout`), конфликт двух писателей не
   проверяется вовсе.
4. Бюджет — метрика пост-фактум: `build()` не знает бюджет и не умеет
   отказывать внятно (§28.3).
5. Рендер живёт в отдельных контейнерах (`RenderGraph3D` поверх
   `RenderGraph`+`GraphExecutor`, `RenderContext`): нет «одного мира»,
   автопараллелизма пассов и единого scheduler'а с ECS (§28.2).

### Этапы

| Этап | Горизонт (по §28) | Суть | Статус |
|---|---|---|---|
| **S0** | недели | Базлайн-метрики: стоимость `build()` на кадр, память пула по техникам | 🟡 benchmark-числа записаны в `perf-baseline-2026-08-27.md`; полная матрица texture budget/probe ещё не архивирована |
| **S1** | недели | Кеш `GraphLayout` с инвалидацией по сигнатуре | ✅ верифицировано CI (PR #4: fmt/clippy/bca/test/doc/wasm зелёные, 2026-08-19) |
| **S2** | месяцы | Пасс = типизированная система (`Reads`/`Writes` в типах), роспуск `match` по именам | ✅ S2a+S2b верифицировано CI (2026-08-19, прогон 32270386050) |
| **S3** | месяцы | Layout из типов; `PassBuilder` → deprecated-шим; конфликт писателей — ошибка | ✅ верифицировано CI (2026-08-19, прогон 32284997326); конфликт писателей снят (порядок = регистрация) |
| **S4** | месяцы | `Budget` как вход планировщика: валидация на `build()`, укладка в бюджет | ✅ верифицировано CI (2026-08-19, прогон 32337861190) |
| **S5** | кварталы | Один scheduler: ресурсы рендера в ECS, автопараллелизм по доступам, без extract | ✅ S5a/b/c ✅ CI + закрытие: **единый `compute_levels`** для графа и ядра (дубли устранены), bench записи на lavapipe (compile-checked; числа — ручной прогон на любой машине), регресс-гейт по построению (parallel opt-in). Extract-free `Res<Device>`-мир — вместе с приоритетом «a» |
| **S6** | годы | Решение: роспуск `RenderGraph` как структуры данных либо фиксация «почему нет» | ✅ ратифицировано 2026-08-19: реестр + `mermaid()`-проекция; полный роспуск отклонён (4 причины) |

Порядок S0→S4 строгий; S5 требует S2–S4; S6 — только по итогам S5.

### S0 — базлайн-метрики

Цель: зафиксировать числа до любых изменений (паттерн G-этапов физики).
Шаги:

- criterion-бенч `compute_layout` на трёх графах: Forward+блум (7 пассов),
  Deferred+блум (8), Hybrid+блум (9) — метрики: время вызова, аллокации.
- Таблица `texture_budget()` по всем техникам × {блум on/off} ×
  {720p, 1080p, 1440p}.
- Probe-диффы всех техник — 0 отличий (точка отсчёта для регрессий).

Гейт: числа в `docs/rendering/unified-scheduler.md` (новый файл) + ссылка
отсюда; бенч входит в `cargo bench -p ornis-render`.
Критерий выхода: таблицы сняты и записаны.

### S1 — кеш GraphLayout (§28, «ближайшее»)

Цель: убрать `compute_layout` с горячего пути кадра.
Шаги:

- `RenderGraph` хранит `layout: Option<GraphLayout>` + `dirty: bool`;
  мутации (`set_surface_size`, `set_pass_enabled`, любой вызов builder'а
  у `add_pass`) ставят `dirty`.
- `build()` → `layout() -> &GraphLayout`: пересчёт только при `dirty`;
  сигнатура кеша по сути = `(surface_size, множество enabled-пассов,
  ресурсы+спеки, read/write множества)` — но вместо хеша-ключа проще
  честный dirty-флаг: мутаций графа в steady-state нет.
- `RenderGraph3D::render` (`graph_frame.rs:492`) и `layout_dump` (:452)
  используют кеш.

Гейт:

- юнит-тест: N кадров без мутаций → ровно один `compute_layout`
  (счётчик за cfg(test)-хуком или injectable-часы);
- инвалидация: resize / `set_pass_enabled` / новые ресурсы → пересчёт;
- probe: пиксельно 0 отличий на всех техниках;
- bench: время кадра до/после (числа — в док S0).

Откат: `layout()` деградирует в пересчёт каждый вызов — эквивалент
сегодняшнего поведения.

**Статус (сверка 2026-08-27; код и основной quality-гейт проверены CI):**

- **S0**: бенч `crates/render/benches/layout_bench.rs` — группы
  `layout/compute/*` (Forward 7 / Deferred 8 / Hybrid 9 пассов, блум,
  1920×1080) и `layout/cache_hit/*`; числа — в
  [`docs/rendering/unified-scheduler.md`](docs/rendering/unified-scheduler.md);
  значения сняты в baseline `docs/quality/perf-baseline-2026-08-27.md`;
  полная матрица texture budget/probe остаётся отдельным ручным прогоном.
- **S1 (реализовано)**: `RenderGraph.cached: Option<GraphLayout>` +
  dirty-флаг на всех мутаторах (`set_surface_size`,
  `create/import/external`-ресурс, `add_pass`, `set_pass_enabled`,
  `PassBuilder::{read,write,write_clear}`). Кеш-доступ — новый
  `layout() -> &GraphLayout` (пересчёт только при dirty); `build()` —
  owned-снимок кеша (клон, для тестов); `invalidate()` — явная
  инвалидация; `layout_computations()` — счетчик пересчётов (диагностика).
  `RenderGraph3D::render`/`layout_dump` переведены на `layout()`;
  добавлены `RenderGraph3D::{graph, graph_mut}` для бенчей/проб.
  От сигнатурного хеша из §28 осознанно отказались в пользу честного
  dirty-флага: мутаций графа в steady state нет, хеш дороже и сложнее в
  поддержке. Benchmark-числа S0/S1 зафиксированы в
  `docs/quality/perf-baseline-2026-08-27.md`; ручной GPU probe остаётся
  дополнительной верификацией. Тесты (5 новых): `layout_is_cached_until_mutation`,
  `every_mutation_invalidates_cache`, `build_snapshot_matches_cached_layout`
  (render_graph), `layout_cache_reused_across_frames` (graph_frame,
  уровень RenderGraph3D). `layout_dump` теперь `&mut self` (единственные
  внешние вызовы — `render_graph_probe`, биндинги уже `mut`). Остаток:
  полная матрица texture budget и ручные probe-диффы на GPU; основной
  benchmark и CI-гейт уже записаны.

### S2 — пасс как типизированная система (§28.1)

Цель: «добавить пасс = написать систему с правильной сигнатурой»; доступы —
в типах, диспетчеризация — без строк.
Шаги:

- Трейт `GraphPass` с типовыми множествами доступа: `type Reads:
  ResourceSet`, `type Writes: ResourceSet`. `ResourceSet` — трейты на
  кортежах ZST-маркеров `Read<R>` / `Write<R>` (в духе ZST-лейнов, идея
  №2.1). Принципиально **без syn-разбора имён** — урок хрупкости
  `smart_pipeline` (строковый матчинг методов).
- Зависимости выводятся из множеств: read-after-write / write-after-write →
  порядок; на этом этапе порядок остаётся явным списком (insertion order),
  `.before()/.after()` — не раньше S5.
- Конфликт двух писателей одного ресурса без порядка — диагностика
  (в S2 — ошибка в `build()` с именами систем; в S3 — по возможности
  compile-time).
- Миграция существующих пассов (gbuffer, lighting, forward,
  bloom_down0/1/2, bloom_up1/up0, composite) с `match` по именам в
  `RenderGraph3D::render` на `impl GraphPass`; ветка `unreachable!`
  умирает.

Гейт: существующие тесты графа зелёные (render_graph + graph_frame);
probe 3 техники × {блум} = 0 отличий; grep-гейт: в исполнителе нет
строковых имён пассов.
Откат: старый `match`-путь за feature-флагом до конца S3.

**Статус S2a (2026-08-18, код + тесты написаны, компиляция при ближайшем
`cargo xtask quality` — среда без toolchain, прецедент G7):**

- Инфраструктура — новый модуль `crates/render/src/system.rs`:
  `GraphResource` (идентичность ресурса = тип: `NAME`/`kind()`/
  `spec(fmt)`), ZST-маркеры `Read<R>`/`Write<R>`/`WriteClear<R, C>`
  (`ClearBlack/White/Transparent` как ассоциированные константы),
  `AccessSet`/`ViewsFor` на кортежах 1..=6 (macro_rules; `Views` =
  кортеж `&TextureView`), трейт `GraphPass { Reads, Writes, name(),
  run(SystemViews, &mut Frame) }`, `SystemSet` (реестр `TypeId →
  ResourceId` + стёртые раннеры; `add_system` выводит проводку из
  типов: read/write/write_clear автоматически), `Frame` (контекст
  кадра). Без syn и строк — анти-цель Приложения C соблюдена.
- Мигрированы пассы со статичными доступами (6 из 10): `GbufferPass`,
  `LightingPass`, `BloomDown1/2Pass`, `BloomUp1/0Pass` (модуль
  `graph_passes.rs`);
  12 типизированных ресурсов (`Albedo`..`Bloom2`), спеки/имена/порядок
  `ResourceId` 1:1 со старой проводкой. `RenderGraph3D::render`:
  диспетчеризация typed-систем по `PassId`, fallback-`match` сжат с 10
  веток до 3.
- Тесты (+6): сбор доступов (порядок/цвета), паритет типизированной и
  builder-проводки (layout-дампы), external-output не пулится, паника
  на незарегистрированный ресурс; в graph_frame —
  `typed_wiring_matches_imperative_reference` (3 техники × {блум on/off}
  против дословной копии старой проводки).
- **S2b (спроектировано 2026-08-19, не реализовано)**: у
  `forward`/`bloom_down0`/`composite` доступы условные (зависят от
  `Technique`/блума). Отброшены: (a) «вариантные типы» — точность без
  дублирования тел; (b) «instance-фильтр» — union доступов врёт
  планировщику (продлевает lifetime мёртвых слоёв, ломает тесты
  пулуализации). Принят синтез — **вариантные режимы + типизированный
  fetch**: конфигурация = singleton-тип, реализующий трейт-таблицу
  фактов (множества доступов + const-ручки + сборка входов 3–6 строк),
  тело пасса одно, generic по режиму (`Composite<M: CompositeMode>`,
  6 режимов; `Forward<OwnsDepth|SharedDepth>`;
  `BloomBright<DeferredInput|ForwardInput>`); опционально
  `views.read::<R>()`/`write::<R>()` — доступ по типу ресурса вместо
  позиционных кортежей (первый шаг: резолв через `Resolver` +
  debug-assert членства — compile-time membership упирается в
  когерентность/специализацию; API стабилен, внутренности ужесточаются
  позже без смены кол-сайтов). Комбинаторика переезжает из кода в
  таблицу режимов: новая опция = строка-факт, не новое тело.
  Эволюция спроектирована и на известные пределы: скриптинг (фаза 6) —
  data-фронтенд `DynamicPass` («манифест + граница»: доступы данными
  до первого исполнения, `validate` при регистрации, обращение вне
  манифеста — ошибка; множества заморожены при регистрации —
  планировщику без разницы, layout/пул/S5 не меняются); масштаб —
  роли/алиасы (пасс пишется против роли, граф разрешает её в ресурсы
  при построении: таблица линейна по измерениям, не мультипликативна;
  composite → пасс-рецепт, режимы-поведения остаются только у forward)
  + бюджетный гейт S4 как предохранитель. Внедрение поэтапное по
  триггерам: data-фронтенд — с первым скриптовым пассом (фаза 6),
  роли — с третьим измерением конфигурации (~SSAO, таблица режимов
  у ~10).
- **S2b реализовано и верифицировано CI (2026-08-19; найдено и исправлено по мере прогонов: E0423 на путях-конструкторах → `new()` у семейств; неявный `Sized` на generic-параметрах структур → супер-трейт `Sized` у трейтов режимов; состав composite-регистрации — replace_migration промахнулся мимо new_with из-за комментариев и попал в тест-эталон, восстановлен; clippy: Default/dead-code)**: `Forward<OwnsDepth |
  SharedDepth>`, `BloomBright<FromDeferred | FromForward>`,
  `Composite<6 режимов>`; тела по одному на семейство, входы composite —
  в `inputs()` режимов (мёртвые слои биндятся на живой вид, выбор по
  `SHADER_MODE`); fetch — единый `SystemViews::get::<R>()` с
  debug-assert членства (пара read/write не понадобилась: у forward
  depth читается в одном режиме и пишется в другом, а тело одно);
  `run_conditional_pass` и `unreachable!`-ветка по именам удалены —
  исполнитель свободен от строковых имён пассов. Паритет: тест
  `typed_wiring_matches_imperative_reference` (дословная старая
  проводка) — oracle всех 6 конфигураций.
  Детали и код-скетч —
  [`docs/rendering/unified-scheduler.md`](docs/rendering/unified-scheduler.md).

### S3 — layout из типов; builder в шим

Цель: единственный источник правды о доступах — типы систем; резких
ломок API нет.
Шаги:

- `compute_layout` потребляет `Reads`/`Writes` из типов; `PassBuilder`
  остаётся как совместимый шим, генерирующий те же данные (deprecated,
  с подсказками миграции).
- Рантайм-проверки «first touch must be write» переносятся на регистрацию
  систем (для builder-пути — сохраняются).
- Golden-тесты layout'ов: число слотов/лтаймы для фиксированных техник —
  против текущих чисел (слоты 7/10 — из B1-R7), чтобы миграция на типы
  не изменила пул тихо.

Гейт: golden-тесты; обновлён `docs/rendering/render-graph.md`
(сигнатуры пассов, пример «как объявить свой пасс»).

### S4 — бюджет памяти как first-class (§28.3)

Цель: планировщик принимает `Budget { gpu_textures: bytes,
transient_pool: bytes, … }` и либо укладывается, либо внятно отказывает.
Шаги:

- `Budget` — параметр `build()`; пик пула ≤ бюджета, иначе ошибка с
  конкретикой: «пассу X нужен ресурс Y (Z MB), бюджет исчерпан — уменьши
  размер / отключи пасс».
- Стратегия пулуализации: текущий greedy first-fit (детерминированный)
  остаётся дефолтом; опционально — минимизация пика (interval partitioning
  + цветовое кодирование; для дерева фаз с фиксированным порядком —
  полиномиально, NP-трудность общего случая обходим ограничением задачи).
- `texture_budget()` (`graph_frame.rs:75`) из метрики становится
  проверяемым ограничением в тестах.

Гейт: proptest — случайные подмножества enabled-пассов: пик никогда не
превышает бюджет либо ошибка с именем ресурса; юнит — бюджет превышен →
ошибка, а не паника/OOM.
Откат: `Budget::unbounded()` → поведение S3.

### S5 — один scheduler, без extract-фазы (§28.2)

Цель: рендер — ECS-системы над лентами (`Res<Device>`, `Res<Queue>`,
`Res<SurfaceConfig>`), один мир, автопараллелизм по непересекающимся
доступам; CPU↔GPU — команды через `command_sync`, а не второй мир.
Подэтапы (каждый с собственным гейтом):

- **S5a**: device/queue/surface и пул текстур — singleton-компоненты в
  ECS; `RenderContext` разбирается на систем-параметры; `RenderGraph3D`
  худеет до конфигурации. Гейт: probe 0 отличий, тесты графа зелёные.
- **S5b**: автопараллелизм независимых пассов (rayon) по множествам
  доступов — паттерн «острова» из физики (G4/G7). Гейт: детерминизм
  Strong Confluence (`RAYON_NUM_THREADS=1` vs `=32`), probe 0 отличий.
- **S5c**: `.before()/.after()`-ордеринг поверх выведенных зависимостей.
  Гейт: bench против S1-пути на 9-пассовом графе — без регрессии.

Предпосылки: S2–S4, `command_sync` (уже есть), опыт GPU-пути физики G7.
Барьеры/layout transitions — wgpu делает сам (оговорка B1-R7): наш слой
отвечает за порядок, lifetime, пул и бюджет, не за синхронизацию.
GPU-бит-идентичность не обещается (урок G7 — документированная норма).

### S6 — решение о роспуске графа — ✅ ратифицировано (2026-08-19)

**Решение (вариант «реестр + отладочная проекция»)**: `RenderGraph`
понижен в статусе — из публичного способа объявлять пассы (это теперь
типы, S2–S5) во внутренний реестр объявлений + движок вывода layout +
носитель рантайм-состояния. Публичная ценность графа как артефакта —
**отладочная проекция**: `GraphLayout::mermaid()` рендерит уровни/пассы/
ресурсы/потоки, GitHub отрисовывает нативно (```mermaid в PR-ревью).
Компиляторная метафора: типы — исходный язык, `AccessDesc` — IR,
`RenderGraph` — таблица символов + планировщик, `GraphExecutor` —
бэкенд; builder — «ассемблер», задокументирован шимом с S3.

**Полный роспуск (builder → cfg(test), реестр в SystemSet, удаление
типа) отклонён**, причины:

1. **Паритет-оракул**: эталон-тест — дословный старый продакшн-код на
   builder'е; роспуск либо убивает его, либо оставляет «музейный»
   второй движок layout с двойной поддержкой навсегда, либо деградирует
   до golden-снапшотов без независимости.
2. **Роспуск иллюзорен**: остаток графа — кеш (S1), бюджет (S4), пул
   текстур, external views, culling — рантайм-состояние, которое не
   бывает проекцией типов; при роспуске оно переезжает, а не исчезает.
3. **Цель §28.1 уже достигнута**: у builder/culling/order_before ноль
   продакшн-потребителей (ревизия 2026-08-19) — ратифицируем факт,
   нечего «дораспускать».
4. **Цена/риск**: большой рефакторинг всей страховочной обвязки
   (паритет, golden, proptest, probe, бенчи) при нулевом
   функциональном выигрыше.

Дверь открыта: если фаза 6 (data-фронтенд) потребует консолидации
реестров — у неё будет второй живой житель и естественная форма
(`AccessDesc` как валюта). Контракт шедулера (5 правил) —
`docs/rendering/unified-scheduler.md`.

Уточнение (2026-08-23, после бэклога #4 аудита): с выносом механики в
`crates/schedule` вопрос «ликвидировать граф» распался на два разных.
**Граф-планировщик ликвидирован** — в `render_graph` не осталось своих
уровней/конфликтов/рёбер/`OrderError` (`layout_levels` — адаптер вызова
`bitset_level_plan`). Причины #2 и #4 выше относятся теперь только к
**оболочке**: доменное состояние (пул, лайфтаймы `[first_use, last_use]`,
бюджет S4, external views, encoders) и debug-проекция
`mermaid()`/`debug_dump()`; её сворачивание в набор систем с
типизированными ресурсами — задача Фазы C плана аудита
(`docs/quality/audit-2026-08-22.md`, §7), а причина #1 («музейный второй
движок layout») снята — движок один.

Доводка имени (2026-08-23, тот же день): оболочка получила
Фаза-C-долговечные имена — `render_graph.rs` → `frame_plan.rs`
(`RenderGraph` → `FramePlan`, `GraphLayout` → `FrameLayout`),
`graph_frame.rs` → `frame_exec.rs` (`GraphExecutor` → `FrameExecutor`,
`RenderGraph3D` → `RenderFrame3D`, `GraphIds` → `FrameIds`),
`graph_passes.rs` → `frame_passes.rs` (`GraphPass` → `FramePass`,
`GraphResource` → `FrameResource`, `ResourceKind::GraphOwned` →
`FrameOwned`); пример — `frame_plan_probe`. Карта имён —
`docs/rendering/unified-scheduler.md` (блок «Переименование» в шапке).
Датированные разделы этого документа (S1–S6) пишут именами своего дня.

Срез 1b (2026-08-23, тот же день): из перечисления доменов оболочки в
уточнении выше обобщена mermaid-половина debug-проекции — доменно-
нейтральный проектор `ornis_schedule::MermaidDiagram` (уровни
подграфами, потоки рёбрами, строковые id/метки от фронтенда). Поверх
него два адаптера: `FrameLayout::mermaid()` (байтовый формат прежний,
пинится golden-тестом `mermaid_is_a_valid_projection`) и новая
`Schedule::mermaid()` (системы `S{i}` подграфами уровней, рёбра
`order_before` — стрелками). На оболочке остаются `debug_dump()`
(пул/слоты/спеки — домен) и runtime-домен из причины #2; причина #4
(цена/риск обвязки) не затронута — обвязка нетронута, формат не менялся.

### Anti-цели и оговорки

- **Не greenfield**: никакого нового крейта «scheduler» с нуля — только
  эволюция существующего графа (§28, «эволюционный путь»).
- **Не дублировать wgpu**: барьеры и переходы layout'ов уже делает wgpu;
  наш планировщик — порядок, lifetime, пул, бюджет.
- **Без syn-строк**: доступы — трейты и ZST-маркеры, не разбор имён
  методов в макросе (хрупкость `smart_pipeline`).
- **Детерминизм**: Strong Confluence гарантируется на CPU-путях; GPU —
  опционально, не бит-идентичен (норма G7).
- **Каждый этап откатываем** и не блокирует приоритеты a–d.

### Общий гейт каждого этапа

`cargo xtask quality` (fmt, clippy −D warnings, bca, test, audit, deny) +
probe-диффы (0 отличий для эквивалентных конфигураций) + обновление
этого приложения статусами и числами по факту (паттерн B1-R7/B2).
