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
cargo xtask quality           # уровень 1: fmt, clippy, rustqual, smoke, test, audit, deny, outdated
cargo xtask quality --ci      # + rustdoc и wasm32 check — ровно то, что гоняет GitHub CI
cargo xtask quality --full    # + покрытие (llvm-cov → target/llvm-cov/html) и bench compile-check
cargo xtask quality --bench   # + полный прогон criterion-бенчмарков (долго)
cargo xtask quality --everything  # всё сразу: --ci + --full + --bench + mutants + fuzz smoke
cargo xtask fuzz <target>     # фаззинг парсеров: scene_ron, materialx_parse (через +nightly)
cargo xtask mutants           # мутационное тестирование ornis-core (cargo-mutants, долго)

# обновление structural baseline (ratchet):
rustqual --save-baseline baseline.json   # локально после осознанного роста сложности
git add baseline.json
cargo xtask quality            # регресс-гейт: падает только если Score/находки ухудшились
```

- Каждая стадия печатает PASS/FAIL/SKIP/INFO; команда не прерывается на первом
  падении, в конце — сводная таблица, exit code по худшей стадии.
- Отсутствующие инструменты (cargo-audit, cargo-deny, cargo-outdated,
  cargo-llvm-cov, cargo-fuzz, cargo-mutants) — SKIP с подсказкой
  `cargo install ... --locked`; `rustqual` — аналогично SKIP, если не установлен.
- Property-тесты ядра (proptest): `crates/core/tests/property_tests.rs` —
  входят в обычный `cargo test`.
- Фаззинг: `fuzz/` — независимый cargo-fuzz крейт (не в workspace),
  запуск `cargo +nightly fuzz run <target>` (rust-toolchain.toml пинит
  stable, поэтому `+nightly` указывается явно).
- Structural gate: [`rustqual.toml`](rustqual.toml) + [`baseline.json`](baseline.json)
  — гейт `rustqual` (MIT, https://github.com/SaschaOnTour/rustqual) измеряет
  IOSP/complexity/DRY/SRP/coupling/test-quality. Ratchet: `rustqual --compare baseline.json --fail-on-regression --no-fail` падает только при регрессе, baseline обновляется осознанно (`rustqual --save-baseline baseline.json`).
- CI: `.github/workflows/quality.yml` — одна job на push/PR в `master`:
  только установка окружения (системные пакеты, toolchain 1.97, wasm
  target, cargo-deny/audit/outdated, rustqual — опционально, если установка
  не удалась, гейт SKIP-нет structural-стадию) и один шаг
  `cargo xtask quality --ci`.
  xtask — единственный источник правды о составе гейта: локально и в CI
  выполняется одна и та же команда.

- **Performance benchmarks** (`.github/workflows/performance.yml`) — отдельный
  workflow для criterion-бенчмарков, не входящий в основной quality gate:
  - запуск вручную через `workflow_dispatch` в Actions;
  - `cargo bench -p ornis-physics --bench solver_bench` — сравнение
    SweepAndPrune / UniformGrid на 1k и 10k телах;
  - benchmark также печатает `BroadPhaseStats` (pair tests, filtering,
    static-static skips, cells, large bodies и unique candidates) для
    breakdown candidate generation и сравнивает grid cell size 1.0/2.0/4.0/8.0/16.0;
  - результаты сохраняются в артефакты `target/criterion/`, сводка — в job summary;
  - workflow не влияет на основной quality gate;
  - 100k body зонд: `cargo run -p ornis-physics --release --example probe_100k -- --grid --cell-size 8` (или `--sweep`, `--bodies`, `--steps`);
  - **Актуально на 2026-09-01 (оптимизации):** `UniformGrid cell 8.0` — default (6.18× vs SAP: `180мс` vs `1.113с` на 10k в `solver_bench`; no-alloc scratch, broadphase 1×/кадр, per-body substeps → фильтр до SAT, narrow — rayon). `tiled 10k` — `~15.8мс` / 63 FPS (`solver_bench: 17.07мс` / 58.6 FPS) vs старые `~103мс/74мс/180мс` — бюджет 60 FPS (`16.67мс`) достигнут. `many_islands`/`hetero 10k` — 61 FPS, `islands_grid 10k` — 355 FPS;
  - история замеров и `BroadPhaseStats`: [`docs/quality/perf-baseline-2026-08-27.md`](docs/quality/perf-baseline-2026-08-27.md).

Подробности: [`docs/quality/report-2026-08-01.md`](docs/quality/report-2026-08-01.md)
и [`docs/quality/baseline-2026-08-01.md`](docs/quality/baseline-2026-08-01.md).

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
| Логический `World` (общие `Resources` + `SmartStore` + запуск `Schedule`) | ✅ | `crates/core/src/world.rs`; native, WASM и editor-only используют единый `ornis_core::World/Engine/Schedule` (unified runtime `crates/app::install_unified_runtime`), serialization boundary только для транспорта |
| Backend-neutral `Engine` (`World` + variable/fixed `Schedule` + `Time`/`FixedTime`) | ✅ | `crates/core/src/engine.rs`; `run_frame` публикует оба clock-ресурса; native showcase, WASM и editor-only подключены к единому `install_gameplay`/`install_unified_runtime` (60 Hz fixed host, InputState→Velocity→Position→RigidBody) |
| Backend-neutral `InputState` resource | ✅ | `crates/core/src/input.rs`; `Engine` + `InputState::apply_snapshot` — единый источник; native winit, WASM orbit + browser `BrowserInput`→`UiCommand::Input`→WS/`POST /api/input`→`apply_snapshot` (remote.rs, editor_world.rs, main.rs) |
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
| Unified Scheduler (IDEAS §28, PLAN Прил. C): кеш layout (S1), пассы-системы с типизированными доступами и режимами (S2), golden-тесты пула (S3), бюджет памяти (S4), уровни параллельности + параллельная запись команд opt-in (S5), `order_before`, общий `mermaid()`-проектор отладки обоих планировщиков (`ornis-schedule::MermaidDiagram`; S6-проекция + срез 1b: `Schedule::mermaid`); `ornis-core::Schedule` + контракт шедулера, hardening: debug-принуждение объявленных доступов систем и пассов (пассы — на выдаче view по `ResourceId`, бэклог #6; кадр систем переносится в дочерние параллельные задачи, `#[smart_pipeline]` — автоматически, бэклог #7), кеш уровневого плана (битсеты), `try_order_before`, гранулярность лент `SmartStore` в декларациях систем (S5d), backend-neutral fixed schedule/accumulator (`FixedTime`) для domain orchestration | 🟡 | инфраструктура и render frame contract реализованы; `Engine` теперь разделяет fixed systems и once-per-frame systems: native/editor-only physics используют общий bounded 60 Hz host, native и WASM render loops проходят через `Engine`/`RenderWorld`/`RenderExtract`/`FramePlan`; единый cross-domain runtime с полноценными gameplay systems всё ещё не собран; `crates/render/src/{extraction.rs,frame_plan.rs,frame_exec.rs}`; детали: `docs/rendering/render-graph.md` |

### Платформы и редактор

| Фича | Статус | Комментарий |
|---|---|---|
| Desktop: winit + wgpu (Vulkan/Metal/DX12) | ✅ | `src/main.rs`, нативный режим |
| WASM + WebGPU в браузере | ✅ | `crates/wasm`; `/api/scene` или `scene.ron` проходит через `RenderWorld`/`Engine`/`RenderExtract`, затем `RenderFrame3D`/`FramePlan`; orbit-камера остаётся client-side |
| Браузерный редактор: фронтенд (панели, иконки, раскладка) | ✅ | `editor/` отдаётся сервером; WASM-canvas рендерит живую сцену из `/api/scene` через единый runtime (без fallback), orbit-камера через InputState |
| Браузерный редактор: связь с живым движком | ✅ | Единый `Engine/World/Schedule` (`crates/app::install_unified_runtime` + `gameplay::install_gameplay`): native showcase, `editor-only` и WASM — один authoritative `World`; browser шлёт `BrowserInput`→`UiCommand::Input` по WS `POST /api/input` → `InputState::apply_snapshot`; `RenderWorld` — view без второго World; `editor/scene.ron` (5 сфер) загружается при старте, `version` инкремент, `POST /api/command` `create_entity/destroy_entity/set_component`, `save_scene/load_scene`; события `CommandCompleted/error/scene_saved` в `/api/events` |
| Сохранение/загрузка сцены (save/load) | ✅ | `save_scene`/`load_scene` через `POST /api/command` (опциональный `{"path": …}`, по умолчанию `editor/scene.ron`): мир сериализуется в RON и пишется атомарно (sibling `*.tmp` + rename), загрузка заменяет мир из файла; результаты — события `scene_saved {path, version}` / `scene_loaded {path, version, entity_count}` / `error` в `GET /api/events`. В UI — меню File → Save/Reload, результат в футере. WASM-canvas рендерит живую сцену: polling `/api/scene` (~1/с), при недоступности сервера — ошибка (требуется `cargo run --features editor-only`) |
| Remote API (HTTP + WebSocket, порт 3420) | ✅ | `GET /`, `GET /api/status`, `GET /api/scene`, `GET /api/events?after=<sequence>`, WebSocket upgrade на `/api/events`, `POST /api/command`, статика из `editor/`. `POST /api/command` возвращает `request_id` + `accepted` ACK; engine-wrapped commands завершаются коррелированным `CommandCompleted`; snapshot endpoints получают transport `sequence`; `/api/events` хранит bounded replay window и сообщает `EventGap`; editor предпочитает WebSocket и откатывается к cursor polling; сервер отправляет heartbeat ping и normal close при shutdown. В режиме `editor-only` команды исполняются ECS-миром (`editor-world` поток); в нативном режиме сервер opt-in (`cargo run -- --remote-editor`), а команды там исполняет заглушка-счётчик в игровом цикле |
| `GET /api/scene` (выгрузка сцены из живого ECS) | ✅ | полный снапшот: `version`, transport `sequence`, `entity_count`, сущности (id, генерация, имя, компоненты, transform/mesh/material), `lights`, `camera`, `ambient`; снапшот публикуется после каждой команды |
| Нативный UI-крейт | 🗑️ | удалён (август 2026): `crates/ui*`, форки, vello/boa-стек; нативный режим рендерит 3D-сцену без UI-overlay |

### Аудио, физика, прочее

| Фича | Статус | Комментарий |
|---|---|---|
| Аудио-база: `AudioSource`/`AudioListener`, декодер (symphonia), бэкенды cpal / Web Audio | ✅ | `crates/audio/`; в настоящий момент файл активно дорабатывается |
| DSP на GPU, процедурный звук | ❌ | |
| `PhysicsEngine` trait + встроенный движок (Sweep-and-Prune, импульсный солвер, collision layers/masks, triggers, точный raycast, SIMD-wide батч-солвер G7) | ✅ | `crates/physics/`; `RigidBody` поддерживает взаимную фильтрацию layer/mask, trigger bodies дают deterministic enter/exit events без импульсов, raycast точно пересекает sphere/OBB/capsule, а angular CCD имеет bounded box/capsule sweep; G7: `wide.rs` (SIMD-wide CPU), `gpu.rs` (GPU, feature `gpu`; шейдер написан на Rust и транслируется в WGSL макросами `gpu_pipeline`/`WgslStruct`, проверен naga и lavapipe-тестами в quality-гейте) |
| Подключение Rapier/Jolt через трейт | ❌ | трейт есть, адаптеров нет |

### Не начато

- **Скриптинг (фаза 6)**: реестр компонентов (F0) ✅; `ScriptEngine`-трейт ✅ (`crates/core/src/script.rs`, `2026-09-05`: `load/call/batch_call/hot_reload/unload` + `NoopScriptEngine`); Batch API по хендлам и hot reload — ✅ как методы трейта; первый адаптер Rhai и прочие языки — ❌; **Mojo — один из официально поддерживаемых адаптеров (решение 2026-09-05, до появления `wasm32`/`WASI` в Mojo — `modular#19`/`#5367` открыты — единственным не становится)** (рамка 2026-08-22: плагинный
  шов + адаптеры вместо лесенки языков — см. PLAN.md и
  [audit-2026-08-22](docs/quality/audit-2026-08-22.md), решения F0/D1)
- **Asset Pipeline (фаза 7)**: build-time сканирование ассетов, hot reload — ❌
- **NUMA-aware allocation** — ❌
- **HVM2/Bend как compute-бэкенд** — ❌ (идея на будущее)
- **Мультиплатформенные тесты/miri (фаза 11)** — ❓ не верифицировано

## Roadmap

### Следующий интеграционный этап

`ornis_core::World` уже даёт общий логический контейнер `Resources` с
`SmartStore` и `Schedule`, а `ornis_core::Engine` — frame host с ресурсами
`Time`/`FixedTime`: fixed schedule выполняется bounded 60 Hz accumulator'ом,
после чего once-per-frame schedule запускается один раз. Native и WASM
render loops уже используют общий `RenderWorld`/`RenderExtract`/`FramePlan`
контракт после serialization boundary. `InputState` теперь является
backend-neutral resource; native winit и WASM orbit adapters записывают
keyboard, mouse, pointer and wheel input, а browser render frame публикует
его через `Engine`. Следующий шаг — подключить полноценные gameplay
consumers и расширить orchestration на остальные домены, а затем собрать
полный cross-domain runtime. Native showcase и editor-only physics уже
подключены к общему fixed host; server/browser physics остаётся отдельным
вопросом из-за serialization boundary. Это не означает немедленно удалять
`FramePlan`: он остаётся переходным render/backend-планом.

Полный план (сделано / частично / приоритеты / анти-цели) — в [`PLAN.md`](PLAN.md).

### Ближайшее: оживить браузерный редактор

1. ~~**Обработчик команд engine ↔ editor**~~ — ✅ сделано: в режиме
   `editor-only` поток `editor-world` (`src/editor_world.rs`) читает `cmd_rx`
   и исполняет команды из `POST /api/command` на живом editor-only ECS-мире.
2. ~~**`GET /api/scene`**~~ — ✅ сделано: сцена сериализуется в JSON
   (version/сущности с transform/mesh/material/lights/camera/ambient),
   снапшот кешируется сервером; при старте мир загружает `editor/scene.ron`.
3. **Связь `editor.js` ↔ REST** — ✅: иерархия и футер живут на
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
4. **WASM-canvas ↔ живой ECS** — 🟡 частично: после serialization boundary
   viewport восстанавливает snapshot в общем library-level `RenderWorld`,
   запускает `Engine`/`RenderExtract` и рисует через `FramePlan`; источник —
   `/api/scene` (polling ~1/с, без fallback) через единый runtime, есть orbit-камера.
   Browser pointer/wheel input уже проходит через `InputState`, а
   `RenderWorld` получает общий `Engine`/`FixedTime` host; впереди —
   gameplay consumers и physics.
5. ~~WebSocket server-push для `/api/events`~~ — ✅ реализован: upgrade на `/api/events`, heartbeat ping, reconnect; polling остаётся fallback.

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

> Сверено с кодом (`crates/render`, `crates/physics`) 2026-09-01.

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
  - **Broadphase**: `UniformGrid cell 8.0` — default (6.18× vs SAP; SAP — fallback), **incremental п1 ✅ готов** (`body_cells`/`prev_meta`, dirty-set + retained clean-clean + heuristic >50% → full rebuild, honest `BroadPhaseStats`), no-alloc scratch, broadphase 1×/кадр, per-body substeps — фильтр `swept AABB vs AABB` до SAT, large-body escape path.
  - **Узкая фаза**: sphere/sphere, sphere/box, box/box, sphere/capsule,
    capsule/capsule и **box↔capsule ✅** (через `distance::shape_distance`/`box_vs_capsule`, speculative `margin`, оба направления, оба narrowphase-пути `detect_collisions_into`/`*_fallback`); OBB SAT и контактные манифолды до 4 точек; narrow — rayon + **narrow cache п2 ✅** (первый substep, `(pos,rot,margin)` ±1e-4, fast-path >0.5 м/с bypass, `HashMap<(usize,usize),NarrowCacheEntry>`) + **SAT cache ✅ 2026-09-02 (отдельный PR, 16-шард `Vec<Mutex>` sequential-only, `obb_sat_cached`/`box_manifold_cached`, `try_lock`-only, large parallel bypass без регресса 9→89 мс, инвалидация по генерациям)**.
  - **Солвер**: warm-start, friction/restitution, split velocity/position stages, block solver, constraint islands, coherent sleeping/wake и ball/revolute joints; для single-point контактов доступны SIMD-wide и opt-in GPU пути.
  - **`step`** — 12 substeps (per-body фильтр), velocity интеграция → broadphase (1×/кадр) → narrowphase → velocity solve → linear/bounded angular CCD → position интеграция → NGS position solve. Бюджет 60 FPS (`16.67мс`) на `tiled 10k`: `~15.8мс` / 63 FPS (`solver_bench 17.07мс` / 58.6 FPS) vs старые `103/74/180мс`; `many_islands`/`hetero` — 61 FPS, `islands_grid` — 355 FPS.
  - **Raycast** — точные локальные sphere/OBB/capsule intersection'ы с surface normals. **`shapecast`** — conservative advancement по точным попарным дистанциям (`distance.rs`), есть тесты на попадание, точную дистанцию и тонкую стену без туннелирования.
  - `BodyType` dynamic/static/kinematic, `RigidBody` с ориентацией, угловой скоростью, collision layer/mask и trigger-флагом; trigger transitions доступны через `TriggerEvent`.
- Тесты находятся в `engine.rs`, `engine/contacts.rs`, `engine/islands.rs`, `engine/joints.rs` и `gpu.rs`; editor-only и native showcase используют ECS↔physics sync systems, browser-side physics намеренно не запускается поверх server snapshot.

### A3. Открытые вопросы для ревью
1. ~~Рендер: активен ли G-buffer/lighting-путь или только forward?~~ **Закрыто в Фазе 4** — render graph c `Technique` (forward/deferred/hybrid), гибрид == legacy пиксель-в-пиксель.
2. ~~Физика: какие связки (joints) нужны в первую очередь и нужен ли CCD для быстрых тел?~~ Закрыто G5/G6 и 2026-08-28: ball/revolute joints, linear CCD и bounded angular CCD реализованы; **2026-09-01 — box↔capsule как честный контакт (G1-остаток закрыт) и fully analytic swept-volume TOI ✅** (`distance::shape_distance` + `box_vs_capsule` в обоих narrowphase-путях, `cast_shape` conservative advancement по точной distance, `shape_max_radius` gap/(|disp|+r·angle) + binary 10); **2026-09-02 — SAT cache ✅ (16-шард, sequential-only)**. Далее нужны joint limits/motors и **полный 8→2 на больших parallel** (lock-free/DashMap).
3. ~~`shapecast` — пустая заглушка~~ **Закрыто (G6)** — честный shapecast через conservative advancement (`distance.rs`), покрыт тестами.
4. ~~Движок рендера не связан с ECS-сценой в браузере~~ **Частично закрыто** — WASM-viewport рендерит живую сцену из `/api/scene`; физика со сценой браузера по-прежнему не связана (см. План B в PLAN.md).
5. **Broadphase scaling** — закрыто на 2026-09-01: `UniformGrid cell 8.0` — production default (6.18× vs SAP, no-alloc, 1×/кадр, per-body substeps фильтр), `tiled 10k ~15.8мс` — бюджет 60 FPS достигнут. SAP — fallback. **п1 incremental ✅ 2026-09-01** (`body_cells`/`prev_meta`, dirty-set + retained, heuristic >50% → full rebuild). Следующий кандидат — persistent `DynamicAabbTree` с moved/active-body queries; adaptive cell size — опционально. **п2 narrow cache ✅ + SAT cache ✅ 2026-09-02 (16-шард sequential-only, parallel bypass без регресса)**. Разбор: [`docs/quality/broadphase-reference-2026-08-29.md`](docs/quality/broadphase-reference-2026-08-29.md). **Полный 8→2 на больших parallel — следующий шаг (lock-free/DashMap + расширенный EPS, сейчас 9→89 мс регресс избежан).**
