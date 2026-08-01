# Ornis Engine

Игровой движок на Rust с «невидимым ECS»: вы пишете обычный объектный код,
а процедурные макросы на этапе компиляции раскладывают данные в
SoA-хранилища (Sparse Sets) и направляют вычисления на CPU (rayon) или
GPU (wgpu compute). Редактор — браузерный: сцена рендерится в `<canvas>`
через WASM + WebGPU, UI — обычное веб-приложение.

> Этот README заменяет прежние документы `STRATEGY_PIVOT.md`, `SUMMARY.md`,
> `implementation_plan.md`, `key_ideas.md`, `ANALYSIS_DOCS_VS_CODE.md`,
> `GOSUB_INTEGRATION.md`. Они сохранены в [`docs/archive/`](docs/archive/)
> и могут расходиться с текущим состоянием кода. Статусы ниже
> верифицированы по исходникам, а не переписаны из старых документов.

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
| `xtask/` | Команда `cargo xtask editor` | Активен |
| `crates/core` | Sparse Sets, Entity (генерационные индексы), диспетчер, физика, Command Sync | Активен |
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
> переехали в `src/ipc.rs`. Планы и контекст — в `docs/archive/`.

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
| `Renderer3D` (forward) + WGSL PBR (GGX, Smith-G, Fresnel-Schlick, ACES) | ✅ | `crates/render/src/` |
| OpenPBR-материал (20 vec4 параметров, все BSDF) | ✅ | `crates/render/src/material.rs` |
| MaterialX: парсер `.mtlx` → AST → `OpenPBRMaterial` | ✅ | `crates/materialx/src/` |
| Трейт `RenderBackend` + фабрика `create_render_backend` | ✅ | `crates/render/src/render_backend.rs` |
| Deferred/Forward hybrid (G-buffer, lighting pass, переключение бэкендов) | ❌ | есть только forward-рендерер + composite-проход |

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
| `PhysicsEngine` trait + встроенный движок (Sweep-and-Prune, PBD, raycast) | ✅ | `crates/core/src/physics/` |
| Подключение Rapier/Jolt через трейт | ❌ | трейт есть, адаптеров нет |

### Не начато

- **Скриптинг (фаза 6)**: Rhai/Rune/Python, Batch API, hot reload — ❌
- **Asset Pipeline (фаза 7)**: build-time сканирование ассетов, hot reload — ❌
- **NUMA-aware allocation** — ❌
- **HVM2/Bend как compute-бэкенд** — ❌ (идея на будущее)
- **Мультиплатформенные тесты/miri (фаза 11)** — ❓ не верифицировано

## Roadmap

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

Полная версия — в `docs/archive/key_ideas.md` (26 идей). Суть:

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
   в git-истории и `docs/archive/`.

## Архив документации

Прежние документы лежат в [`docs/archive/`](docs/archive/):

- `STRATEGY_PIVOT.md` — решение о переходе от нативного UI к браузерному редактору
- `implementation_plan.md` — фазовый план (фазы 0–11; статусы местами устарели)
- `key_ideas.md` — 26 архитектурных идей
- `SUMMARY.md` — автосводка по фазам (содержала дубли и конфликты статусов)
- `ANALYSIS_DOCS_VS_CODE.md` — аудит «документы vs код» от 12 июля 2026 (устарел: с тех пор `RenderBackend`, GPU-исполнение Command Sync и `compile_warning!` появились)
- `GOSUB_INTEGRATION.md` — план интеграции Gosub (эксперимент заморожен)

При расхождении архива с этим README верить README и коду.
