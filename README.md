# Ornis Engine

Игровой движок на Rust с «невидимым ECS»: вы пишете обычный объектный код,
а процедурные макросы на этапе компиляции раскладывают данные в
SoA-хранилища (Sparse Sets) и направляют вычисления на CPU (rayon) или
GPU (wgpu compute). Редактор — браузерный: сцена рендерится в `<canvas>`
через WASM + WebGPU, UI — обычное веб-приложение.

> Документация проекта — три файла: этот README (что есть сейчас,
> статусы верифицированы по коду), [`PLAN.md`](PLAN.md) (план
> реализации, синхронизированный с кодом) и [`IDEAS.md`](IDEAS.md)
> (архитектурные идеи). Плюс аудит-снимки в [`docs/quality/`](docs/quality/).
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

## Качество

Единая точка входа — `cargo xtask quality`:

```bash
cargo xtask quality           # уровень 1: fmt, clippy -D warnings, test, audit, deny, outdated
cargo xtask quality --full    # + покрытие (llvm-cov → target/llvm-cov/html) и bench compile-check
cargo xtask quality --bench   # + полный прогон criterion-бенчмарков (долго)
cargo xtask fuzz <target>     # фаззинг парсеров: scene_ron, materialx_parse (через +nightly)
cargo xtask mutants           # мутационное тестирование ornis-core (cargo-mutants, долго)
```

- Каждая стадия печатает PASS/FAIL; команда не прерывается на первом
  падении, в конце — сводная таблица, exit code по худшей стадии.
- Отсутствующие инструменты (cargo-audit, cargo-deny, cargo-outdated,
  cargo-llvm-cov, cargo-fuzz, cargo-mutants) — SKIP с подсказкой
  `cargo install ... --locked`.
- Property-тесты ядра (proptest): `crates/core/tests/property_tests.rs` —
  входят в обычный `cargo test`.
- Фаззинг: `fuzz/` — независимый cargo-fuzz крейт (не в workspace),
  запуск `cargo +nightly fuzz run <target>` (rust-toolchain.toml пинит
  stable, поэтому `+nightly` указывается явно).
- CI: `.github/workflows/quality.yml` — три job (quality, wasm-check,
  supply-chain) на push/PR в `master`.

Подробности: [`docs/quality/report-2026-08-01.md`](docs/quality/report-2026-08-01.md)
и [`docs/quality/baseline-2026-08-01.md`](docs/quality/baseline-2026-08-01.md).

## Структура репозитория

| Путь | Назначение | Статус |
|---|---|---|
| `src/` | Бинарь `ornis`: нативный режим (winit + wgpu + Vello) и `editor-only` (HTTP-сервер) | Активен |
| `editor/` | Фронтенд редактора: `index.html`, `css/`, `js/`, `icons/`, `scene.ron` | Активен |
| `xtask/` | Команды `cargo xtask`: `editor`, `quality`, `fuzz`, `mutants` | Активен |
| `crates/core` | Sparse Sets, Entity (генерационные индексы), диспетчер, Command Sync | Активен |
| `crates/physics` | Физика: трейт `PhysicsEngine`, Sweep-and-Prune, `RigidBody`, raycast | Активен (вынесен из core в августе 2026) |
| `crates/macros` | Процедурные макросы: `smart_pipeline`, `for_each_entity`, `kernel`, `Pack` и др. | Активен |
| `crates/render` | `Renderer3D`, OpenPBR-материал, WGSL-шейдеры, трейт `RenderBackend` | Активен |
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
> переехали в `src/ipc.rs`. Планы и контекст — в git-истории.

## Текущее состояние (верифицировано по коду)

Легенда: ✅ — реализовано и проверено в коде · 🟡 — частично · ❌ — не реализовано · ❄️ — заморожено · ❓ — не верифицировано в этом аудите

### Ядро ECS

| Фича | Статус | Комментарий |
|---|---|---|
| Sparse Sets (`ComponentStore`: dense + entities + paginated sparse + bitset) | ✅ | `crates/core/src/component_store.rs` |
| Entity Recycling + генерационные индексы | ✅ | `crates/core/src/entity.rs` |
| Bitset-пересечения, страничные sparse-массивы, cache-line alignment | ✅ | в `ComponentStore` |
| Lock-free store, hot/cold split, temporal sort (`defrag`) | ✅ | `lock_free_store.rs`, `cold_store.rs` |
| ZST-диспетчеризация (`GpuLane`/`CpuLane`/`HybridLane`, `LaneTarget`) | ✅ | `crates/core/src/pipeline.rs` |
| Макросы (`smart_pipeline`, `for_each_entity`, `kernel`, `gpu_pipeline`, `Pack`, `PipelineConfig`, `AutoPipeline`) | ✅ | `crates/macros/src/` |
| Runtime-диспетчер CPU/GPU (`Dispatcher`, `SmartDispatcher`, `decide(element_count)`) | ✅ | `crates/core/src/dispatcher.rs` |
| Command-Based Sync: CPU-side очередь команд + residency tracker | ✅ | `crates/core/src/command_sync.rs` |
| Command-Based Sync: реальное GPU-исполнение (compute dispatch + flush) | ✅ | `crates/wgpu_backend/src/command_sync.rs`, есть тест `gpu_dispatch_records_and_flushes` |
| Линтер: `compile_warning!` при непараллелизуемых паттернах | 🟡 | `#[smart_pipeline]` генерирует `compile_warning!` (`crates/macros/src/smart_pipeline.rs`), но нет интеграции с IDE и расширяемого набора правил |
| Component Packing (`#[derive(Pack)]`) | 🟡 | wrapper-типы генерируются, но не интегрированы автоматически в `for_each_entity`/`smart_pipeline` |
| SmartBuffer (автоматическая data residency CPU↔GPU) | 🟡 | dirty-флаги есть; автоматического копирования «только при необходимости» нет |

### Рендер и материалы

| Фича | Статус | Комментарий |
|---|---|---|
| `Renderer3D` + WGSL PBR (GGX, Smith-G, Fresnel-Schlick, ACES) | ✅ | `crates/render/src/` |
| OpenPBR-материал (20 vec4 параметров, все BSDF) | ✅ | `crates/render/src/material.rs` |
| MaterialX: парсер `.mtlx` → AST → `OpenPBRMaterial` | ✅ | `crates/materialx/src/` |
| Трейт `RenderBackend` + фабрика `create_render_backend` | ✅ | `crates/render/src/render_backend.rs` |
| Render Graph: `RenderGraph3D` + `Technique` (forward/deferred/hybrid как конфигурация графа) + блум-каскад | ✅ | `crates/render/src/render_graph.rs`, `graph_frame.rs`; детали: `docs/rendering/render-graph.md` |

### Платформы и редактор

| Фича | Статус | Комментарий |
|---|---|---|
| Desktop: winit + wgpu (Vulkan/Metal/DX12) | ✅ | `src/main.rs`, нативный режим |
| WASM + WebGPU в браузере | ✅ | `crates/wasm`; рендер `editor/scene.ron` проверен headless-скриншотами (5 сфер, pixel-identical нативному эталону) |
| Браузерный редактор: фронтенд (панели, иконки, раскладка) | 🟡 | `editor/` отдаётся сервером и отрисовывается; WASM-canvas рендерит статичную сцену из `scene.ron` |
| Браузерный редактор: связь с живым движком | ❌ | WASM-canvas не подключён к ECS; нет ввода, редактирования, live-обновления сцены |
| Remote API (HTTP, порт 3420) | 🟡 | `GET /`, `GET /api/status`, `GET /api/events`, `POST /api/command`, статика из `editor/`. WebSocket нет. В режиме `editor-only` команды из `POST /api/command` уходят в канал, который никто не читает (`_cmd_rx` в `src/main.rs`); в нативном режиме они обрабатываются |
| `GET /api/scene` (выгрузка сцены из живого ECS) | ❌ | не реализовано |
| Нативный UI-крейт | 🗑️ | удалён (август 2026): `crates/ui*`, форки, vello/boa-стек; нативный режим рендерит 3D-сцену без UI-overlay |

### Аудио, физика, прочее

| Фича | Статус | Комментарий |
|---|---|---|
| Аудио-база: `AudioSource`/`AudioListener`, декодер (symphonia), бэкенды cpal / Web Audio | ✅ | `crates/audio/`; в настоящий момент файл активно дорабатывается |
| DSP на GPU, процедурный звук | ❌ | |
| `PhysicsEngine` trait + встроенный движок (Sweep-and-Prune, импульсный солвер, raycast) | ✅ | `crates/physics/` |
| Подключение Rapier/Jolt через трейт | ❌ | трейт есть, адаптеров нет |

### Не начато

- **Скриптинг (фаза 6)**: Rhai/Rune/Python, Batch API, hot reload — ❌
- **Asset Pipeline (фаза 7)**: build-time сканирование ассетов, hot reload — ❌
- **NUMA-aware allocation** — ❌
- **HVM2/Bend как compute-бэкенд** — ❌ (идея на будущее)
- **Мультиплатформенные тесты/miri (фаза 11)** — ❓ не верифицировано

## Roadmap

Полный план (сделано / частично / приоритеты / анти-цели) — в [`PLAN.md`](PLAN.md).

### Ближайшее: оживить браузерный редактор

1. **Обработчик команд engine ↔ editor** — в режиме `editor-only` читать
   `cmd_rx` и исполнять команды из `POST /api/command` (сейчас канал
   создаётся и сразу отбрасывается).
2. **`GET /api/scene`** — сериализация текущей сцены из ECS в JSON/RON,
   чтобы фронтенд мог отображать реальную иерархию сущностей.
3. **Связь `editor.js` ↔ REST** — инспектор и иерархия поверх `/api/scene`,
   редактирование компонентов через `/api/command`.
4. **WASM-canvas ↔ живой ECS** — рендер не статичного `scene.ron`,
   а актуального состояния; ввод (мышь/клавиатура) из браузера в движок.
5. Дальше: WebSocket-канал для событий и live-синхронизации вместо polling'а
   `/api/events`.

### Фаза 6 — Скриптинг

Rhai → Batch API (`engine.batch_add(...)` — один FFI-вызов вместо 100k) →
hot reload → Rune → Python (PyO3/RustPython). FFI-биндинги оборачивают
переменные скриптов в прямые указатели на ячейки Sparse Set.

### Фаза 7 — Asset Pipeline

Build-time сканирование `/assets` (парсинг CSS/SVG/HTML/MTLX, генерация
Rust-структур и бинарных слепков для Sparse Sets) + runtime hot reload
через `notify`.

## Ключевые архитектурные идеи

Полная версия — в [`IDEAS.md`](IDEAS.md) (26 идей). Суть:

1. **Невидимый ECS.** Пользователь пишет `entity.position += entity.velocity`,
   макросы превращают AoS-код в SoA-хранилища (Sparse Sets: плотный `data` +
   страничный sparse-индексатор + bitset). Мутации O(1), без Archetype Move.
2. **Компилятор как оптимизатор.** ZST-маркеры (`GpuLane`/`CpuLane`) выбирают
   исполнителя на этапе компиляции — в рантайме нет ни одного `if` для выбора
   CPU/GPU. Статический профайлер анализирует AST (размер типа, ветвления,
   access pattern) и генерирует пороги.
3. **Инструкции вместо данных.** CPU шлёт GPU не массивы float'ов, а команды
   («примени гравитацию к Position»); данные живут там, где созданы
   (Command-Based Sync + Data Residency).
4. **Детерминизм (Strong Confluence).** Параллельные системы обязаны давать
   побитово одинаковый результат при любом числе потоков (тесты с
   `RAYON_NUM_THREADS=1` и `=32`).
5. **Открытые стандарты.** OpenPBR как модель затенения, MaterialX как формат
   материалов — совместимость с VFX/AAA-пайплайнами, без изобретения своего PBR.
6. **Плагинные трейты.** `PhysicsEngine` (своя лёгкая физика + Rapier/Jolt/PhysX)
   и `RenderBackend` (свой wgpu-рендер + возможность замены).
7. **Браузерный редактор.** Нативный UI-движок удалён (август 2026): доведение
   его до production сопоставимо с командой браузерного движка. Редактор —
   веб-приложение, сцена — WASM/WebGPU в `<canvas>`. История нативного стека —
   в git-истории.

## Документация

Структура документации — три файла плюс аудит-снимки:

- [`README.md`](README.md) — текущее состояние, верифицированное по коду (этот файл)
- [`PLAN.md`](PLAN.md) — план реализации: сделано / частично / дорожная карта
- [`IDEAS.md`](IDEAS.md) — 26 архитектурных идей (перенесён без изменений)
- [`docs/quality/`](docs/quality/) — аудит-снимки качества: baseline и report от 2026-08-01

Прежние документы (`STRATEGY_PIVOT.md`, `implementation_plan.md`, `SUMMARY.md`,
`ANALYSIS_DOCS_VS_CODE.md`, `GOSUB_INTEGRATION.md`) удалены из дерева при
консолидации (август 2026) — история сохранена в git. При расхождении
любых старых источников с кодом верить коду.

---
## Приложение A — Движок рендеринга и физический движок (черновик для ревью)

> 🔎 Черновик, написан по коду (`crates/render`, `crates/physics`).
> Нужен ваш ревью: там, где формулировки неточны, — поправьте.

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
- **`scene.rs`** — загрузка сцены, **`mesh.rs`** (`Mesh`/`Vertex`), **`shader.rs`**/`shaders/` (WGSL, `math.rs` — math-хелперы шейдеров), **`transform.rs`**, **`composite.rs`**.

### A2. Физический движок (`crates/physics`)

#### Что есть в коде
- **Трейт `PhysicsEngine`** (Send+Sync): `step`, `add_body`/`remove_body`/`get_body(_mut)`, `raycast`, `shapecast` — точка подключения внешних движков (Rapier/Jolt за тем же трейтом).
- **`BuiltinPhysicsEngine`** (`mod.rs`):
  - **Sweep-and-Prune** широкофазный: сортировка AABB-ов по сменяющейся оси (x→y→z), active-пары.
  - **Узкая фаза**: контакты сфера/сфера, сфера/бокс, бокс/бокс (минимальная ось SAT), капсула/капсула.
  - **Разрешение**: позиционная коррекция по проникновению + импульс (реституция) + **трение Кулона**.
  - **Substeps** = 4: внутри `step` цикл `integrate → broadphase → detect → resolve`.
  - **Raycast** — t-slab по AABB, возвращает ближайшее. **`shapecast` — заглушка (`None`)**.
  - `BodyType` dynamic/static, `RigidBody` (position, velocity, inv_mass, restitution, friction), `Shape` (sphere/box/capsule).
- Тесты в `mod.rs`: падение сферы, статика не падает, сфера-сфера, box-box, raycast на сферу.

### A3. Открытые вопросы для ревью
1. ~~Рендер: активен ли G-buffer/lighting-путь или только forward?~~ **Закрыто в Фазе 4** — render graph c `Technique` (forward/deferred/hybrid), гибрид == legacy пиксель-в-пиксель.
2. Физика: какие связки (joints) нужны в первую очередь (revolute/ball), нужен ли CCD для быстрых тел.
3. `shapecast` — пустая заглушка; планируется ли честная реализация или достаточно raycast+sphere-cast.
4. Движок рендера и физики не связаны с ECS-сценой в браузере (см. План B в PLAN.md).
