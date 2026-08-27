# Ornis Engine

Игровой движок на Rust с «невидимым ECS»: вы пишете обычный объектный код,
а процедурные макросы на этапе компиляции раскладывают данные в
SoA-хранилища (Sparse Sets) и направляют вычисления на CPU (rayon) или
GPU (wgpu compute). Редактор — браузерный: сцена рендерится в `<canvas>`
через WASM + WebGPU, UI — обычное веб-приложение.

> Основные документы проекта — этот README (что есть сейчас, статусы
> верифицированы по коду), [`PLAN.md`](PLAN.md) (план реализации,
> синхронизированный с кодом) и [`IDEAS.md`](IDEAS.md) (архитектурные
> идеи). Дополнительные implementation notes находятся в
> [`docs/quality/`](docs/quality/) и [`docs/rendering/`](docs/rendering/).
> Прежние документы (`STRATEGY_PIVOT.md`, `implementation_plan.md` и др.)
> удалены из дерева — история сохранена в git.

---

## Быстрый старт

```bash
cargo xtask editor        # или: cargo editor
```

Затем откройте **http://127.0.0.1:3420** в Chrome/Edge с включённым WebGPU.

Требования:

- Rust toolchain (stable)
- `wasm-pack` (сборка `crates/wasm`)
- Chrome/Edge с WebGPU (на macOS работает из коробки)

Что происходит при запуске: xtask собирает WASM-пакет в `editor/pkg/`,
запускает бинарь `ornis` в режиме `editor-only`, который поднимает
HTTP-сервер на порту 3420 и раздаёт фронтенд из `editor/`.

Нативный режим (`cargo run` без фичи) сервер **не поднимает** — движку
редактор не нужен; браузерный редактор рядом с нативным окном доступен
по явному флагу: `cargo run -- --remote-editor`.

## Качество

Единая точка входа — `cargo xtask quality`:

```bash
cargo xtask quality           # уровень 1: fmt, clippy, bca, test, audit, deny, outdated
cargo xtask quality --ci      # + rustdoc и wasm32 check — ровно то, что гоняет GitHub CI
cargo xtask quality --full    # + покрытие (llvm-cov → target/llvm-cov/html) и bench compile-check
cargo xtask quality --bench   # + полный прогон criterion-бенчмарков (долго)
cargo xtask quality --everything  # всё сразу: --ci + --full + --bench + mutants + fuzz smoke
cargo xtask fuzz <target>     # фаззинг парсеров: scene_ron, materialx_parse (через +nightly)
cargo xtask mutants           # мутационное тестирование ornis-core (cargo-mutants, долго)

# complexity gate (опционально, внешний бинарь) — обёртка над ручной последовательностью:
cargo xtask bca --full        # = install (если нет) + baseline + report + quality
cargo xtask bca --install     # cargo install big-code-analysis-cli --locked
cargo xtask bca --write-baseline
cargo xtask bca --report      # html to target/bca/index.html

# ручной эквивалент (как было ранее):
# cargo install big-code-analysis-cli --locked
# bca check --write-baseline
# git add .bca-baseline.toml
# bca report -O html -o target/bca/index.html
# cargo xtask quality
```

- Каждая стадия печатает PASS/FAIL/SKIP/INFO; команда не прерывается на первом
  падении, в конце — сводная таблица, exit code по худшей стадии.
- Отсутствующие инструменты (cargo-audit, cargo-deny, cargo-outdated,
  cargo-llvm-cov, bca, cargo-fuzz, cargo-mutants) — SKIP с подсказкой
  `cargo install ... --locked`.
- Property-тесты ядра (proptest): `crates/core/tests/property_tests.rs` —
  входят в обычный `cargo test`.
- Фаззинг: `fuzz/` — независимый cargo-fuzz крейт (не в workspace),
  запуск `cargo +nightly fuzz run <target>` (rust-toolchain.toml пинит
  stable, поэтому `+nightly` указывается явно).
- Complexity: [`bca.toml`](bca.toml) + [`.bcaignore`](.bcaignore) + [`.bca-baseline.toml`](.bca-baseline.toml)
  — гейт `bca` (https://github.com/dekobon/big-code-analysis) измеряет
  cognitive/cyclomatic/Halstead/ABC. Лицензия CLI — MPL-2.0, но как внешний
  бинарь он **не влияет** на лицензию Ornis (MIT OR Apache-2.0). Не добавлять
  как либу в Cargo — только `cargo install big-code-analysis-cli`.
  Подробности: [`docs/quality/bca.md`](docs/quality/bca.md).
- CI: `.github/workflows/quality.yml` — одна job на push/PR в `master`:
  только установка окружения (системные пакеты, toolchain 1.97, wasm
  target, cargo-deny/audit/outdated, bca — опционально, если установка
  не удалась, гейт SKIP-нет complexity-стадию) и один шаг
  `cargo xtask quality --ci`.
  xtask — единственный источник правды о составе гейта: локально и в CI
  выполняется одна и та же команда.

Подробности: [`docs/quality/report-2026-08-01.md`](docs/quality/report-2026-08-01.md),
[`docs/quality/baseline-2026-08-01.md`](docs/quality/baseline-2026-08-01.md)
и [`docs/quality/bca.md`](docs/quality/bca.md).

## Структура репозитория

| Путь | Назначение | Статус |
|---|---|---|
| `src/` | Бинарь `ornis`: нативный режим (winit + wgpu) и `editor-only` (HTTP-сервер) | Активен |
| `editor/` | Фронтенд редактора: `index.html`, `css/`, `js/`, `icons/`, `scene.ron` | Активен |
| `xtask/` | Команды `cargo xtask`: `editor`, `quality`, `fuzz`, `mutants` | Активен |
| `crates/core` | Sparse Sets, Entity (генерационные индексы), диспетчер, Command Sync | Активен |
| `crates/physics` | Физика: трейт `PhysicsEngine`, Sweep-and-Prune, `RigidBody`, raycast | Активен (вынесен из core в августе 2026) |
| `crates/macros` | Процедурные макросы: `smart_pipeline`, `for_each_entity`, `kernel`, `Pack` и др. | Активен |
| `crates/render` | `Renderer3D`, OpenPBR-материал, WGSL-шейдеры, трейт `RenderBackend` | Активен |
| `crates/schedule` | Механика планировщика: `compute_levels`, битсет-план конфликтов, единый `OrderError`, кеш `PlanCache`, исполнитель `run_levels` | Активен (Фаза A аудита, август 2026) |
| `crates/wgpu_backend` | GPU-исполнение: command sync, smart buffer, PSO-кэш, роутер | Активен |
| `crates/materialx` | Парсер `.mtlx` и конвертация в OpenPBR | Активен |
| `crates/wasm` | WASM-обёртка для рендера сцены в браузере | Активен |
| `crates/audio` | Аудио: `AudioSource`/`AudioListener`, бэкенды cpal / Web Audio | Активен |
| `assets/` | Ассеты (шрифты Inter и пр.) | Служебное |

> Удалено (август 2026): нативный UI-стек `crates/ui`, `crates/ui-blitz`,
> `crates/ui-gosub`, `crates/ui-core`, `src/bin_blitz.rs` и локальные форки
> `forks/` (blitz, boa_engine, icu_normalizer). Решение: писать собственный
> отрисовщик фронтенда нецелесообразно — редактор живёт в браузере (`editor/`),
> сцена рендерится через WASM/WebGPU. IPC-типы `UiCommand`/`GameEvent`
> переехали в `crates/editor-backend/src/ipc.rs`. Планы и контекст — в git-истории.

## Текущее состояние (верифицировано по коду)

Легенда: ✅ — реализовано и проверено в коде · 🟡 — частично · ❌ — не реализовано · ❄️ — заморожено · ❓ — не верифицировано в этом аудите

### Ядро ECS

| Фича | Статус | Комментарий |
|---|---|---|
| Sparse Sets (`ComponentStore`: dense + entities + paginated sparse + bitset) | ✅ | `crates/core/src/component_store.rs` |
| Логический `World` (общие `Resources` + `SmartStore` + запуск `Schedule`) | 🟡 | `crates/core/src/world.rs`; фундамент есть, но physics/render/editor и основной frame loop ещё не подключены |
| Entity Recycling + генерационные индексы | ✅ | `crates/core/src/entity.rs` |
| Bitset-пересечения, страничные sparse-массивы, cache-line alignment | ✅ | в `ComponentStore` |
| Lock-free store, hot/cold split, temporal sort (`defrag`) | ✅ | `lock_free_store.rs`, `cold_store.rs` |
| ZST-диспетчеризация (`GpuLane`/`CpuLane`/`HybridLane`, `LaneTarget`) | ✅ | `crates/core/src/pipeline.rs` |
| Макросы (`smart_pipeline`, `for_each_entity`, `kernel`, `gpu_pipeline`, `WgslStruct`, `Pack`, `PipelineConfig`, `AutoPipeline`) | ✅ | `crates/macros/src/` |
| Runtime-диспетчер CPU/GPU (`Dispatcher`, `SmartDispatcher`, `decide(element_count)`) | 🟡 | `crates/core/src/dispatcher.rs`; выбор по порогу работает, но `GpuExecutor` в core пока CPU-fallback/stub |
| Command-Based Sync: CPU-side очередь команд + residency tracker | ✅ | `crates/core/src/command_sync.rs` |
| Command-Based Sync: реальное GPU-исполнение (compute dispatch + flush) | ✅ | `crates/wgpu_backend/src/command_sync.rs`, есть тест `gpu_dispatch_records_and_flushes` |
| Линтер: compile-time предупреждения при непараллелизуемых паттернах | 🟡 | `#[smart_pipeline]` помечает такие циклы через deprecated-note трюк (видно в IDE и терминале; `crates/macros/src/smart_pipeline.rs`), но нет расширяемого набора правил |
| Component Packing (`#[derive(Pack)]`) | ✅ | генерируются wrapper-ленты и `Pack::for_each_packed` (packed-аналог `for_each_entity!`); wrapper-ленты — обычные компоненты и напрямую совместимы с `for_each_entity!` (`crates/core/tests/pack_integration.rs`) |
| SmartBuffer (автоматическая data residency CPU↔GPU) | 🟡 | dirty-флаги есть; автоматического копирования «только при необходимости» нет |

### Рендер и материалы

| Фича | Статус | Комментарий |
|---|---|---|
| `Renderer3D` + WGSL PBR (GGX, Smith-G, Fresnel-Schlick, ACES) | ✅ | `crates/render/src/` |
| OpenPBR-материал (20 vec4 параметров, все BSDF) | ✅ | `crates/render/src/material.rs` |
| MaterialX: парсер `.mtlx` → AST → `OpenPBRMaterial` | ✅ | `crates/materialx/src/` |
| Трейт `RenderBackend` + фабрика `create_render_backend` | ✅ | `crates/render/src/render_backend.rs` |
| Frame Plan (бывш. Render Graph; модули `frame_plan.rs`/`frame_exec.rs`, rename от 2026-08-23): `RenderFrame3D` + `Technique` (forward/deferred/hybrid как конфигурация плана) + блум-каскад | ✅ |
| Unified Scheduler (IDEAS §28, PLAN Прил. C): кеш layout (S1), пассы-системы с типизированными доступами и режимами (S2), golden-тесты пула (S3), бюджет памяти (S4), уровни параллельности + параллельная запись команд opt-in (S5), `order_before`, общий `mermaid()`-проектор отладки обоих планировщиков (`ornis-schedule::MermaidDiagram`; S6-проекция + срез 1b: `Schedule::mermaid`); `ornis-core::Schedule` + контракт шедулера, hardening: debug-принуждение объявленных доступов систем и пассов (пассы — на выдаче view по `ResourceId`, бэклог #6; кадр систем переносится в дочерние параллельные задачи, `#[smart_pipeline]` — автоматически, бэклог #7), кеш уровневого плана (битсеты), `try_order_before`, гранулярность лент `SmartStore` в декларациях систем (S5d) | 🟡 | инфраструктура планирования реализована, но единый runtime-кадр native/WASM ещё не wired; `crates/render/src/frame_plan.rs`, `frame_exec.rs`; детали: `docs/rendering/render-graph.md` |

### Платформы и редактор

| Фича | Статус | Комментарий |
|---|---|---|
| Desktop: winit + wgpu (Vulkan/Metal/DX12) | ✅ | `src/main.rs`, нативный режим |
| WASM + WebGPU в браузере | ✅ | `crates/wasm`; рендер `editor/scene.ron` проверен headless-скриншотами (5 сфер, pixel-identical нативному эталону) |
| Браузерный редактор: фронтенд (панели, иконки, раскладка) | 🟡 | `editor/` отдаётся сервером и отрисовывается; WASM-canvas рендерит живую сцену из `/api/scene` (polling ~1/с) с fallback на `scene.ron`, есть orbit-камера |
| Браузерный редактор: связь с живым движком | 🟡 | В режиме `editor-only` сервер держит отдельный editor-only ECS-мир (`src/editor_world.rs`), пока не общий `ornis_core::World`: при старте мир загружает `editor/scene.ron` (5 сфер + свет/камера/ambient как ресурс), у сущностей компоненты Name/Transform/Mesh/Material, есть `version` (инкремент на мутацию). Иерархия и счётчик сущностей в футере обновляются из `/api/scene`/`/api/status`, создание сущности из UI работает; через `POST /api/command` принимаются `create_entity`/`destroy_entity` и generic `set_component` (любой компонент из реестра, serde-каноничный JSON), невалидные команды → событие `error`. Сохранение и загрузка сцены — команды `save_scene`/`load_scene` (см. следующую строку) |
| Сохранение/загрузка сцены (save/load) | ✅ | `save_scene`/`load_scene` через `POST /api/command` (опциональный `{"path": …}`, по умолчанию `editor/scene.ron`): мир сериализуется в RON и пишется атомарно (sibling `*.tmp` + rename), загрузка заменяет мир из файла; результаты — события `scene_saved {path, version}` / `scene_loaded {path, version, entity_count}` / `error` в `GET /api/events`. В UI — меню File → Save/Reload, результат в футере. WASM-canvas рендерит живую сцену: polling `/api/scene` (~1/с), при недоступности сервера — fallback на `scene.ron` |
| Remote API (HTTP, порт 3420) | 🟡 | `GET /`, `GET /api/status`, `GET /api/scene`, `GET /api/events`, `POST /api/command`, статика из `editor/`. WebSocket нет. В режиме `editor-only` команды исполняются ECS-миром (`editor-world` поток); в нативном режиме сервер opt-in (`cargo run -- --remote-editor`), а команды там исполняет заглушка-счётчик в игровом цикле |
| `GET /api/scene` (выгрузка сцены из живого ECS) | ✅ | полный снапшот: `version`, `entity_count`, сущности (id, генерация, имя, компоненты, transform/mesh/material), `lights`, `camera`, `ambient`; снапшот публикуется после каждой команды |
| Нативный UI-крейт | 🗑️ | удалён (август 2026): `crates/ui*`, форки, vello/boa-стек; нативный режим рендерит 3D-сцену без UI-overlay |

### Аудио, физика, прочее

| Фича | Статус | Комментарий |
|---|---|---|
| Аудио-база: `AudioSource`/`AudioListener`, декодер (symphonia), бэкенды cpal / Web Audio | ✅ | `crates/audio/`; в настоящий момент файл активно дорабатывается |
| DSP на GPU, процедурный звук | ❌ | |
| `PhysicsEngine` trait + встроенный движок (Sweep-and-Prune, импульсный солвер, raycast, SIMD-wide батч-солвер G7) | ✅ | `crates/physics/`; G7: `wide.rs` (SIMD-wide CPU), `gpu.rs` (GPU, feature `gpu`; шейдер написан на Rust и транслируется в WGSL макросами `gpu_pipeline`/`WgslStruct`, проверен naga и lavapipe-тестами в quality-гейте) |
| Подключение Rapier/Jolt через трейт | ❌ | трейт есть, адаптеров нет |

### Не начато

- **Скриптинг (фаза 6)**: реестр компонентов (F0) ✅; `ScriptEngine`-трейт,
  Rhai-адаптер, Batch API и hot reload — ❌ (рамка 2026-08-22: плагинный
  шов + адаптеры вместо лесенки языков — см. PLAN.md и
  [audit-2026-08-22](docs/quality/audit-2026-08-22.md), решения F0/D1)
- **Asset Pipeline (фаза 7)**: build-time сканирование ассетов, hot reload — ❌
- **NUMA-aware allocation** — ❌
- **HVM2/Bend как compute-бэкенд** — ❌ (идея на будущее)
- **Мультиплатформенные тесты/miri (фаза 11)** — ❓ не верифицировано

## Roadmap

### Следующий интеграционный этап

`ornis_core::World` уже даёт общий логический контейнер `Resources` с
`SmartStore` и `Schedule`. Следующий шаг — подключить к нему time/input,
physics и render как доменные системы, а затем перевести native и WASM
frame loops на единый контракт. Это не означает немедленно удалять
`FramePlan`: он остаётся переходным render/backend-планом.

Полный план (сделано / частично / приоритеты / анти-цели) — в [`PLAN.md`](PLAN.md).

### Ближайшее: оживить браузерный редактор

1. ~~**Обработчик команд engine ↔ editor**~~ — ✅ сделано: в режиме
   `editor-only` поток `editor-world` (`src/editor_world.rs`) читает `cmd_rx`
   и исполняет команды из `POST /api/command` на живом editor-only ECS-мире.
2. ~~**`GET /api/scene`**~~ — ✅ сделано: сцена сериализуется в JSON
   (version/сущности с transform/mesh/material/lights/camera/ambient),
   снапшот кешируется сервером; при старте мир загружает `editor/scene.ron`.
3. **Связь `editor.js` ↔ REST** — 🟡 частично: иерархия и футер живут на
   `/api/scene` и `/api/status` (polling), создание из UI работает;
   редактирование компонентов — generic-командой `set_component` через
   реестр компонентов (решение D2, реализовано; см.
   [audit-2026-08-22](docs/quality/audit-2026-08-22.md)): сервер и UI
   обмениваются serde-каноничным JSON, новый компонент движка становится
   редактируемым регистрацией в реестре — без правок engine-side кода.
   Инспектор UI уже сейчас правит name/transform/material через неё.
   ~~Сохранение/загрузка сцены~~ — ✅ сделано: меню File → Save/Reload шлёт
   `save_scene`/`load_scene` (атомарная запись `editor/scene.ron`, события
   `scene_saved`/`scene_loaded`/`error`, результат показан в футере).
4. **WASM-canvas ↔ живой ECS** — 🟡 частично: viewport рендерит актуальное
   состояние из `/api/scene` (polling ~1/с, fallback на `scene.ron`), есть
   orbit-камера; впереди — ввод (мышь/клавиатура) из браузера в движок.
5. Дальше: WebSocket-канал для событий и live-синхронизации вместо polling'а
   `/api/events`.

### Фаза 6 — Скриптинг (рамка пересмотрена 2026-08-22)

Реестр компонентов (F0) → `ScriptEngine`-трейт (плагинный, как
`PhysicsEngine`/`RenderBackend`) → Batch API по хендлам → первый
адаптер Rhai → hot reload → прочие языки отдельными адаптерами
(Rune/Python/WASM-компоненты) по правилу трёх. Подробно: фаза 6 в
[`PLAN.md`](PLAN.md), решения F0/D1/D2 в
[audit-2026-08-22](docs/quality/audit-2026-08-22.md).

### Фаза 7 — Asset Pipeline

Build-time сканирование `/assets` (парсинг CSS/SVG/HTML/MTLX, генерация
Rust-структур и бинарных слепков для Sparse Sets) + runtime hot reload
через `notify`.

## Ключевые архитектурные идеи

Полная версия — в [`IDEAS.md`](IDEAS.md) (28 пронумерованных разделов идей). Суть:

1. **Невидимый ECS.** Пользователь пишет `entity.position += entity.velocity`,
   макросы превращают AoS-код в SoA-хранилища (Sparse Sets: плотный `data` +
   страничный sparse-индексатор + bitset). Мутации O(1), без Archetype Move.
2. **Компилятор как оптимизатор.** ZST-маркеры (`GpuLane`/`CpuLane`) задают
   статический типовой маршрут там, где он уже известен; runtime
   `SmartDispatcher` пока выбирает по порогу, а `ornis-core::GpuExecutor`
   остаётся CPU-fallback/stub. Статический профайлер анализирует AST (размер
   типа, ветвления, access pattern) и генерирует пороги.
3. **Инструкции вместо данных.** CPU шлёт GPU не массивы float'ов, а команды
   («примени гравитацию к Position»); данные живут там, где созданы
   (Command-Based Sync + Data Residency).
4. **Детерминизм (Strong Confluence).** CPU-пути с честными декларациями и
   коммутативными аккумуляторами должны давать побитово одинаковый результат
   при любом числе потоков (тесты с `RAYON_NUM_THREADS=1` и `=32`); GPU-путь
   сознательно не обещает bit-identical результат.
5. **Открытые стандарты.** OpenPBR как модель затенения, MaterialX как формат
   материалов — совместимость с VFX/AAA-пайплайнами, без изобретения своего PBR.
6. **Плагинные трейты.** `PhysicsEngine` (своя лёгкая физика + Rapier/Jolt/PhysX)
   и `RenderBackend` (свой wgpu-рендер + возможность замены).
7. **Браузерный редактор.** Нативный UI-движок удалён (август 2026): доведение
   его до production сопоставимо с командой браузерного движка. Редактор —
   веб-приложение, сцена — WASM/WebGPU в `<canvas>`. История нативного стека —
   в git-истории.

## Документация

Основные документы и implementation notes:

- [`README.md`](README.md) — текущее состояние, верифицированное по коду (этот файл)
- [`PLAN.md`](PLAN.md) — план реализации: сделано / частично / дорожная карта
- [`IDEAS.md`](IDEAS.md) — 28 пронумерованных архитектурных разделов (включая исторический §5)
- [`docs/quality/`](docs/quality/) — аудит-снимки качества: baseline/report, coverage, performance и audit;
- [`docs/rendering/`](docs/rendering/) — implementation notes по FramePlan и Unified Scheduler.

Прежние документы (`STRATEGY_PIVOT.md`, `implementation_plan.md`, `SUMMARY.md`,
`ANALYSIS_DOCS_VS_CODE.md`, `GOSUB_INTEGRATION.md`) удалены из дерева при
консолидации (август 2026) — история сохранена в git. При расхождении
любых старых источников с кодом верить коду.

---
## Приложение A — Движок рендеринга и физический движок

> Сверено с кодом (`crates/render`, `crates/physics`) 2026-08-27.

### A1. Движок рендеринга (`crates/render`)

#### Что есть в коде
- **`Renderer3D`** (`renderer.rs`) — основной рендерер на wgpu. Содержит:
  - **forward-проход** + структуры **G-buffer** (`GBufferTextures`: albedo,
    normal, material_id, world_position, material_params, depth) и pipeline'ы
    `ForwardPass` / `LightingPass` / `CompositePass`.
    ✅ *Спор forward vs deferred закрыт (см. раздел «Рендер»): техника — это
    конфигурация render graph (`Technique`), все три ветки верифицированы
    probe-диффами (гибрид == legacy пиксель-в-пиксель).*
  - Uniform'ы: `CameraUniform` (view_proj, inv_view_proj, cam_pos),
    `PerObjectGpu` (model, normal_matrix, material_index), `GpuLight`+`LightingUniform`
    (ambient + до 4 направленных источников), `InstanceData`.
  - Бюджеты: `max_objects=256`, `max_materials=64`.
- **Материалы**: `OpenPBRMaterial` (20 vec4-параметра, все BSDF) из `ornis_core`, константа `OPENPBR_MATERIAL_SIZE`.
- **`RenderBackend`** (`render_backend.rs`) — трейт + фабрика `create_render_backend` (плагинная точка смены бэкенда).
- **`scene.rs`** — загрузка сцены, **`mesh.rs`** (`Mesh`/`Vertex`), **`shaders/`** (WGSL, `math.rs` — math-хелперы шейдеров), **`transform.rs`**, **`composite.rs`**.

### A2. Физический движок (`crates/physics`)

#### Что есть в коде
- **Трейт `PhysicsEngine`** (Send+Sync): `step`, `add_body`/`remove_body`/`get_body(_mut)`, `raycast`, `shapecast` — точка подключения внешних движков (Rapier/Jolt за тем же трейтом).
- **`BuiltinPhysicsEngine`** (`engine.rs`):
  - **Sweep-and-Prune** широкофазный: swept AABB для движущихся тел, сортировка по сменяющейся оси (x→y→z), active-пары.
  - **Узкая фаза**: sphere/sphere, sphere/box, box/box, sphere/capsule и
    capsule/capsule; OBB SAT и контактные манифолды до 4 точек. Пара
    box/capsule в дискретном contact path пока не реализована, хотя
    `distance.rs` поддерживает её для shapecast.
  - **Солвер**: warm-start, friction/restitution, split velocity/position stages, block solver, constraint islands, coherent sleeping/wake и ball/revolute joints; для single-point контактов доступны SIMD-wide и opt-in GPU пути.
  - **`step`** по умолчанию делится на 12 подшагов; в каждом подшаге идут интеграция скоростей → broadphase/narrowphase → velocity solve → linear CCD → интеграция позиций → NGS position solve.
  - **Raycast** — t-slab против ориентированного shape AABB с приближённой нормалью. **`shapecast`** — conservative advancement по точным попарным дистанциям (`distance.rs`), есть тесты на попадание, точную дистанцию и тонкую стену без туннелирования.
  - `BodyType` dynamic/static/kinematic, `RigidBody` с ориентацией, угловой скоростью и диагональным body-space inertia.
- Тесты находятся в `engine.rs`, `engine/contacts.rs`, `engine/islands.rs`, `engine/joints.rs` и `gpu.rs`; physics пока не синхронизирована автоматически с ECS `World`.

### A3. Открытые вопросы для ревью
1. ~~Рендер: активен ли G-buffer/lighting-путь или только forward?~~ **Закрыто в Фазе 4** — render graph c `Technique` (forward/deferred/hybrid), гибрид == legacy пиксель-в-пиксель.
2. ~~Физика: какие связки (joints) нужны в первую очередь и нужен ли CCD для быстрых тел?~~ Закрыто G5/G6: ball/revolute joints и linear CCD реализованы; дальше нужны joint limits/motors и angular CCD.
3. ~~`shapecast` — пустая заглушка~~ **Закрыто (G6)** — честный shapecast через conservative advancement (`distance.rs`), покрыт тестами.
4. ~~Движок рендера не связан с ECS-сценой в браузере~~ **Частично закрыто** — WASM-viewport рендерит живую сцену из `/api/scene`; физика со сценой браузера по-прежнему не связана (см. План B в PLAN.md).
