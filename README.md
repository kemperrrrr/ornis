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
  - **Актуально на 2026-09-09 (adaptive broadphase):** `BroadPhaseKind::Auto` — analytic SAP↔Grid↔Tree routing с гистерезисом (≤512 тел → sweep, sparse → tree, ≥1500 плотные → grid cell 8.0); `tiled 10k` → grid (~14–16мс steady-state, бюджет 60 FPS держится), `sparse 10k` → tree (10.4мс), settled `tiled 10k` tree 20.2→~12мс (бьёт grid-8 16.1мс); липкая cell 8.0 + уход по плотности (`raw<3` → 2.0, `raw>12` → 16.0). 100k tiled — вне real-time (~8 с/шаг на Grid): следующий шаг — модульные солверы (AVBD-спайк M0, см. `PLAN.md`);
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
| `crates/physics` | Физика: трейт `PhysicsEngine`, `BroadPhaseKind::Auto` (SAP/Grid/Tree + adaptive), GJK/EPA-формы (cylinder/cone/hull/TriMesh/heightfield), CCD, joints с limits/motors, `RigidBody`, raycast/shapecast | Активен (вынесен из core в августе 2026) |
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
| SmartBuffer (ручной примитив residency CPU↔GPU) + `AutoLane`/`GpuLanes` (автослой) | 🟡 | `SmartBuffer` — ручной примитив (dirty-флаги); автоматический слой — `AutoLane` (`ornis-wgpu-backend`: политика + residency + CPU-фолбэк) и typed `GpuLanes`-мост; покрытие вне доказанных kernel'ов — открыто |

### Рендер и материалы

| Фича | Статус | Комментарий |
|---|---|---|
| `Renderer3D` + WGSL PBR (GGX, Smith-G, Fresnel-Schlick, ACES) | ✅ | `crates/render/src/` |
| OpenPBR-материал (20 vec4 параметров, все BSDF) | ✅ | `crates/render/src/material.rs` |
| MaterialX: парсер `.mtlx` → AST → `OpenPBRMaterial` | ✅ | `crates/materialx/src/` |
| Трейт `RenderBackend` + фабрика `create_render_backend` | ✅ | `crates/render/src/render_backend.rs` |
| Frame Plan (бывш. Render Graph; `RenderFrame3D` + `Technique` (forward/deferred/hybrid как конфигурация плана) + блум-каскад): модуль `frame_plan.rs` **растворен в `transient_pool.rs` (d2, 2026-09-07) и `system.rs` (d3, 2026-09-07)**; реестр деклараций — единый `SystemSet` (resources/passes/ordering/budget/pool/generation), компилятор лайфтаймов и пула — `TransientPool` (`FrameExecutor::ensure_layout(&SystemSet)` ключ — `SystemSet::generation`). Полное удаление `FramePlan` (d4, 2026-09-07): паритет-оракул `Schedule` vs `SystemSet` подтверждает две формы декларации (типизированная `add_system<P: FramePass>` + императивная `add_pass().read/write`) на одном типе | ✅ |
| Unified Scheduler (IDEAS §28, PLAN Прил. C): кеш layout (S1), пассы-системы с типизированными доступами и режимами (S2), golden-тесты пула (S3), бюджет памяти (S4), уровни параллельности + параллельная запись команд opt-in (S5), `order_before`, общий `mermaid()`-проектор отладки обоих планировщиков (`ornis-schedule::MermaidDiagram`; S6-проекция + срез 1b: `Schedule::mermaid`); `ornis-core::Schedule` + контракт шедулера, hardening: debug-принуждение объявленных доступов систем и пассов (пассы — на выдаче view по `ResourceId`, бэклог #6; кадр систем переносится в дочерние параллельные задачи, `#[smart_pipeline]` — автоматически, бэклог #7), кеш уровневого плана (битсеты), `try_order_before`, гранулярность лент `SmartStore` в декларациях систем (S5d), backend-neutral fixed schedule/accumulator (`FixedTime`) для domain orchestration; GPU как системы (S7) — `GpuDevice/Queue/Surface/SurfaceState/GpuFrameState` как `Resources` (`crates/render/src/gpu_resources.rs`), `RenderSubmit` (S5e X4, 2026-09-07: без `Mutex<RenderExtracted>`-снимка, читает лейны/ресурсы напрямую) делает `set_camera/upload_*`, `RenderPresent` делает `acquire → frame3d.render → queue.submit/present` в том же `Engine::schedule` (`RenderExtract → OrbitCamera → RenderSubmit → RenderPresent` на едином `bitset_level_plan` с физикой), `GameApp::render_frame` — только `run_frame`; cross-domain bridge `Velocity→RigidBody` (fixed `velocity_to_body`) + `RigidBody→Position/TransformDesc` (frame `body_to_transform`) в `ornis-app::install_gameplay_physics_bridge` (идемпотентен, добавлен в `EditorWorld` и `GameApp::showcase_engine`) — browser `WASD`/`InputState` теперь ведёт `Player` через `player_input → physics_push/physics_step → render extract` в одном DAG | ✅ | `Engine` разделяет fixed/once-per-frame; native render теперь полностью в `Engine::schedule` (upload + acquire/present — системы), `Resized` реконфигурирует `GpuSurface` из ресурса; WASM и editor-world уже на том же `Engine`/`RenderExtract`/`RenderFrame3D` + gameplay/physics bridge; serialization boundary между сервером и браузером сохраняется намеренно; `gpu_resources` — native-only (`#[cfg(not(target_arch = \"wasm32\"))]`, 2026-09-07: wgpu web-типы `!Send`/`!Sync`, несовместимы с `Send + Sync`-бинареми `World`/`Resources`; wasm рендерит через extraction + `ornis-wasm`); S5e-E1 ✅ 2026-09-07 (`schedule_bridge`: пассы — `System`-близнецы в core `Schedule`, `render_schedule` — уровни `Schedule` + borrowed encoder; гейты — паритет уровней в `scheduler_parity` + пиксельный паритет `tests/schedule_render.rs`); S5e-E2 ✅ 2026-09-07 (`render_to_buffers` — per-pass encoders → `FrameCommandBuffers` (registration order) + `FramePresentTarget` handover; нативный `RenderPresent` не субмитит — `RenderFlush` делает ordered submit + present; гейт — третий пиксельный тест `tests/schedule_render.rs`); X1 ✅ 2026-09-07 (`RenderSubmit` — прямое чтение `TransformDesc`/`MeshDesc`/`MaterialDesc`-лейн (`reads_lane`, S5d) через канон `extract_render_data` вместо клона `Mutex<RenderExtracted>`; оракул-гейт — прямой вызов побайтово равен снапшоту); X2 ✅ 2026-09-07 (`GpuMesh`-ресурс + система `RenderMesh`: пересоздание сферы по `max_mesh_params` из лейна (канон `extract_render_data`, пол (32,24)), `RenderSubmit` меш не трогает, `RenderPresent` читает `Mutex<GpuMesh>`; гейты — canon-tie + tessellation-probe `tests/mesh_resource.rs`); X3 ✅ 2026-09-07 (`RenderLights`-ресурс: ambient + `Vec<LightDesc>`, пишется `RenderWorld::replace_scene`, дефолт = legacy-риг (const), `RenderSubmit` читает ресурс вместо хардкода `set_lights`, конверсия — `set_lights_args()`; гейты — юнит-равенство legacy + пиксельный probe `tests/light_resource.rs`); X4 ✅ 2026-09-07 (`RenderExtracted` → `FrameUpload`, снапшот-система/ресурс удалены, `RenderPresent` на прямое чтение лейнов, wasm на `frame_upload()`, корневой showcase на `GpuMesh`-ресурс + прямые лейны); E3 ✅ no-op (пул render-side, golden-тесты не менялись) — S5e+Extract-free закрыта; `crates/render/src/{extraction.rs,transient_pool.rs,system.rs,frame_exec.rs,gpu_resources.rs,schedule_bridge.rs}`, `crates/app/src/lib.rs`; `docs/rendering/unified-scheduler.md` |

### Платформы и редактор

| Фича | Статус | Комментарий |
|---|---|---|
| Desktop: winit + wgpu (Vulkan/Metal/DX12) | ✅ | `src/main.rs`, нативный режим |
| WASM + WebGPU в браузере | ✅ | `crates/wasm`; `/api/scene` или `scene.ron` проходит через `RenderWorld`/`Engine`/`RenderExtract`, затем `RenderFrame3D`; orbit-камера остаётся client-side |
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
| `PhysicsEngine` trait + встроенный движок (Sweep-and-Prune, импульсный солвер, collision layers/masks, triggers, точный raycast, SIMD-wide батч-солвер G7) | ✅ | `crates/physics/`; `RigidBody` поддерживает взаимную фильтрацию layer/mask, trigger bodies дают deterministic enter/exit events без импульсов, raycast точно пересекает sphere/OBB/capsule, а angular CCD имеет bounded box/capsule sweep; G7: `wide.rs` (SIMD-wide CPU), `gpu.rs` (GPU, feature `gpu`; шейдер написан на Rust и транслируется в WGSL макросами `gpu_pipeline`/`WgslStruct`, проверен naga и lavapipe-тестами в quality-гейте); второй impl — `AvbdEngine` (`avbd.rs`, AVBD M1 ✅ 2026-09-12: стек боксов, Ball/Revolute, триггеры/contact-события, raycast-паритет, детерминизм; gaps: sphere-stacking, часть джойнтов, CCD/sleep/islands) |
| Подключение Rapier/Jolt через трейт | ❌ | трейт есть, адаптеров нет |

### Не начато

- **Скриптинг (фаза 6)**: реестр компонентов (F0) ✅ (`ComponentRegistry` + `#[derive(RegisterComponent)]`/`register_component` `2026-09-06`); `ScriptEngine`-трейт ✅ (`crates/core/src/script.rs`, `2026-09-05`: `load/call/batch_call/hot_reload/unload` + `NoopScriptEngine`); Batch API по хендлам и hot reload — ✅ как методы трейта; первый адаптер Rhai ✅ 2026-09-05 (`crates/rhai`: `RhaiScriptEngine`, JSON-кодек args/return, 6 тестов; deny/advisories/outdated чисто), второй адаптер Rune ✅ 2026-09-06 (`crates/rune`: `RuneScriptEngine`, rune 0.14, 7 тестов зеркально Rhai; шов проверен правилом трёх), третий адаптер Python ✅ 2026-09-06 (`crates/python`: `PythonScriptEngine`, rustpython-vm 0.5 на worker-thread, 8 тестов; deny/advisories чисто; 2026-09-07: rustpython-vm вендорен в `third_party/rustpython-vm` — `[patch.crates-io]` + однострочный фикс RustPython#8343, без него крейт не собирается с libc ≥ 0.2.187 при гейте «все зависимости latest»), интеграция в рантайм ✅ 2026-09-06 (`ScriptHost`-ресурс + `script_tick` во variable schedule + `ScriptPlugin`; тик `[{"dt","tick"}]` (массив — требование кодека адаптеров) → outcomes per entry; `apply_outcomes` пишет `{"set": [...]}` в мир двухфазно через реестр), editor-интеграция ✅ 2026-09-06 (`script_load/call/hot_reload/unload/list` + file-watch `.rhai` с hot-reload на тике + автоверсия; 3 e2e-теста), WASM — ❌; **Mojo — один из официально поддерживаемых адаптеров (решение 2026-09-05, до появления `wasm32`/`WASI` в Mojo — `modular#19`/`#5367` открыты — единственным не становится)** (рамка 2026-08-22: плагинный
  шов + адаптеры вместо лесенки языков — см. PLAN.md и
  [audit-2026-08-22](docs/quality/audit-2026-08-22.md), решения F0/D1)
- **Asset Pipeline (фаза 7)**: build-time сканирование ассетов — ❌; hot reload сцены ✅ (`FileWatch`, mtime `editor/scene.ron`, без `notify`)
- **NUMA-aware allocation** — ❌
- **HVM2/Bend как compute-бэкенд** — ❌ (идея на будущее)
- **Мультиплатформенные тесты/miri (фаза 11)** — ❓ не верифицировано

## Roadmap

### Следующий интеграционный этап

`ornis_core::World` уже даёт общий логический контейнер `Resources` с
`SmartStore` и `Schedule`, а `ornis_core::Engine` — frame host с ресурсами
`Time`/`FixedTime`: fixed schedule выполняется bounded 60 Hz accumulator'ом,
после чего once-per-frame schedule запускается один раз. Native и WASM
render loops уже используют общий `RenderWorld`/`RenderExtract`/`RenderFrame3D`
контракт после serialization boundary. `InputState` теперь является
backend-neutral resource; native winit и WASM orbit adapters записывают
keyboard, mouse, pointer and wheel input, а browser render frame публикует
его через `Engine`. Gameplay consumers подключены через
`ornis-app::install_gameplay_physics_bridge` (`Velocity→RigidBody` fixed +
`RigidBody→Position/TransformDesc` frame) в `EditorWorld` и native showcase:
browser `WASD`/`InputState` (WS `POST /api/input` → `apply_browser_input`)
ведёт `Player` в том же DAG, что физика и render extraction. Остаётся
расширить orchestration на прочие домены (audio/script) — полный
единый runtime без отдельной render-фазы extract остаётся будущей
целью.

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
4. **WASM-canvas ↔ живой ECS** — ✅: после serialization boundary
   viewport восстанавливает snapshot в общем library-level `RenderWorld`,
   запускает `Engine`/`RenderExtract` и рисует через `RenderFrame3D`; источник —
   `/api/scene` (polling ~1/с, без fallback) через единый runtime, есть orbit-камера.
   Browser `WASD`/`InputState` теперь проходит через cross-domain bridge
   `ornis-app::install_gameplay_physics_bridge` (`player_input` → `Velocity` →
   `velocity_to_body` → physics solver → `body_to_transform` → `Position`/
   `TransformDesc`) в том же `Engine::run_frame` DAG, что и editor/native;
   `browser_wasd_input_drives_player_through_gameplay` верифицирует 2-тактный
   `Input→Velocity→RigidBody→Position` путь. Остаётся расширить bridge на
   audio/script домены.
5. ~~WebSocket server-push для `/api/events`~~ — ✅ реализован: upgrade на `/api/events`, heartbeat ping, reconnect; polling остаётся fallback.

### Фаза 6 — Скриптинг (рамка пересмотрена 2026-08-22)

Реестр компонентов (F0) → `ScriptEngine`-трейт (плагинный, как
`PhysicsEngine`/`RenderBackend`) → Batch API по хендлам → первый
адаптер Rhai ✅ → hot reload → прочие языки отдельными адаптерами
(Rune/Python/WASM-компоненты) по правилу трёх. Подробно: фаза 6 в
[`PLAN.md`](PLAN.md), решения F0/D1/D2 в
[audit-2026-08-22](docs/quality/audit-2026-08-22.md).

### Фаза 7 — Asset Pipeline

Build-time сканирование `/assets` (парсинг CSS/SVG/HTML/MTLX, генерация
Rust-структур и бинарных слепков для Sparse Sets) — позже. Runtime hot
reload сцены уже есть: editor-world следит за mtime `editor/scene.ron`
(`SceneFileWatch`, без `notify`) и перезагружает мир; фронтенд подхватывает
новую версию через `/api/scene`.

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
- [`docs/rendering/`](docs/rendering/) — implementation notes по render-graph, transient pool и Unified Scheduler.

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
- **`scene.rs`** — загрузка сцены, **`mesh.rs`** (`Mesh`/`Vertex`), **`shaders/`** (ноль рукописного WGSL: `*_generated.rs` из `#[stage]`/`#[gpu_pipeline]` + `helpers.rs`/`math.rs`), **`transform.rs`**, **`composite.rs`**.

### A2. Физический движок (`crates/physics`)

#### Что есть в коде
- **Трейт `PhysicsEngine`** (Send+Sync): `step`, `add_body`/`remove_body`/`get_body(_mut)`, `raycast`, `shapecast` — точка подключения внешних движков (Rapier/Jolt за тем же трейтом).
- **`BuiltinPhysicsEngine`** (`engine.rs`):
  - **Broadphase**: `UniformGrid cell 8.0` — default (6.18× vs SAP; SAP — fallback), **incremental п1 ✅ готов** (`body_cells`/`prev_meta`, dirty-set + retained clean-clean + heuristic >50% → full rebuild, honest `BroadPhaseStats`), no-alloc scratch, broadphase 1×/кадр, per-body substeps — фильтр `swept AABB vs AABB` до SAT, large-body escape path.
  - **Узкая фаза**: sphere/sphere, sphere/box, box/box, sphere/capsule,
    capsule/capsule и **box↔capsule ✅** (через `distance::shape_distance`/`box_vs_capsule`, speculative `margin`, оба направления, оба narrowphase-пути `detect_collisions_into`/`*_fallback`); **cylinder/cone/convex-hull ✅ 2026-09-11** — GJK (64 итерации, свидетели) + EPA (32, horizon через edge-counting) в `gjk.rs`, generic single-contact через `distance_contact` (surface-anchored, hemisphere-rule при пересечении свидетелей); свидетели чинятся alternating closest-point проекциями (`Shape::closest_point`, рестарт из центров при dist < 1мм, interior → nearest-face); hull-тела несут rolling/torsion дефолты 0.2/0.05 (иначе тетраэдр качается на вершинах вечно — замерено); **trimesh ✅ 2026-09-12** — `Shape::TriMesh` из сырого `vertices/indices` (`TriMesh::from_indexed`, без лоадеров): треугольники как предсобранные centroid-relative hull-примитивы под median-split AABB BVH (свой, без зависимостей), narrow = минимум GJK по overlapped листьям, raycast — BVH walk + slab prune, инерция точная (Mirtich, bbox-fallback), mesh-vs-mesh без контакта (concave-concave не определён); **heightfield ✅ 2026-09-11** — минимум по overlapped колонкам (solid box от global min + one-cell skirt на плоских), DDA-raycast; OBB SAT и контактные манифолды до 4 точек; narrow — rayon + **narrow cache п2 ✅** (первый substep, `(pos,rot,margin)` ±1e-4, fast-path >0.5 м/с bypass, `HashMap<(usize,usize),NarrowCacheEntry>`) + **SAT cache ✅ 2026-09-02, lock-free 2026-09-05** (`obb_sat_cached`/`box_manifold_cached`, общий `DashMap` для parallel+sequential без `try_lock`-потерь, переиспользуется только ось SAT + расширенный EPS ~1 мм/~0.26°, narrow −12…−18% на parallel-сценах без регресса).
  - **Солвер**: warm-start, friction/restitution, split velocity/position stages, block solver, constraint islands, coherent sleeping/wake и joints ball/revolute/**prismatic ✅/fixed ✅/distance ✅/wheel ✅/gear ✅/sixdof ✅ 2026-09-11** (все с лимитами/моторами где уместно, gear — координатное `coord_a+ratio·coord_b=const` с remap, sixdof — per-axis Locked/Limited/Free+drive); для single-point контактов доступны SIMD-wide и opt-in GPU пути.
  - **`step`** — 12 substeps (per-body фильтр), velocity интеграция → broadphase (1×/кадр) → narrowphase → velocity solve → linear/analytic angular CCD → position интеграция → NGS position solve. Бюджет 60 FPS (`16.67мс`) на `tiled 10k`: `~15.8мс` / 63 FPS (`solver_bench 17.07мс` / 58.6 FPS) vs старые `103/74/180мс`; `many_islands`/`hetero` — 61 FPS, `islands_grid` — 355 FPS.
  - **Raycast** — точные локальные sphere/OBB/capsule/**cylinder/cone/hull ✅/heightfield-DDA ✅ 2026-09-11**/**trimesh-BVH ✅ 2026-09-12** intersection'ы с surface normals. **`shapecast`** — conservative advancement по точным попарным дистанциям (`distance.rs`), есть тесты на попадание, точную дистанцию и тонкую стену без туннелирования. **Kinematic CCD ✅ 2026-09-11 (п.1):** контракт драйвера (телепорт = смена позы при нулевой скорости), `PrevPose` + swept-AABB union, implied motion выше travel-gate (половина min-dim) ставится в velocity-поля на шаг (save/restore), ниже — positional settle; signed SAT-separation для box-box (unsigned oracle слеп к пересечениям). **Angular CCD тесты ✅ 2026-09-11:** 3 интеграционных (`box_capsule_toi`: box/capsule/thin-wall) перебазированы со старого full-stop контракта на задокументированный unified impulse — клэмп (нетуннелирование) + energy-cap (`E_after ≤ E_before`, спин cheaper-оси не растёт); тангенциальный спин гасится фрикцией следующих сабстепов по дизайну.
  - `BodyType` dynamic/static/kinematic, `RigidBody` с ориентацией, угловой скоростью, collision layer/mask и trigger-флагом; trigger transitions доступны через `TriggerEvent`.
- Тесты находятся в `engine.rs`, `engine/contacts.rs`, `engine/islands.rs`, `engine/joints.rs`, `gjk.rs` (GJK/EPA: разделение/проникновение/свидетели/регрессия inside-test) и `gpu.rs`; editor-only и native showcase используют ECS↔physics sync systems, browser-side physics намеренно не запускается поверх server snapshot.

### A3. Открытые вопросы для ревью
1. ~~Рендер: активен ли G-buffer/lighting-путь или только forward?~~ **Закрыто в Фазе 4** — render graph c `Technique` (forward/deferred/hybrid), гибрид == legacy пиксель-в-пиксель.
2. ~~Физика: какие связки (joints) нужны в первую очередь и нужен ли CCD для быстрых тел?~~ Закрыто G5/G6 и 2026-08-28: ball/revolute joints, linear CCD и analytic angular CCD реализованы; **Friction ✅ 2026-09-11 (ODE fdir1/mu/mu2 + MuJoCo slide/torsion/rolling):** `friction_dir` (локальный fdir1, приоритет тела A, вырождение → дефолтный базис) + `friction_transverse` (mu2; по умолчанию зеркалит `friction`); изотропные пары идут legacy circular-cone дословно (снапшот цел), анизотропные — joint elliptical projection (осознанное отклонение от ODE-пирамиды: эллипс — точное обобщение Кулона); `rolling_friction`/`torsion_friction` в метрах (torque cap = μ·λn, чистые пары без линейной части), оба пути (scalar + wide lanes) с предподсчитанными угловыми массами, нулевые коэффициенты — zero-cost skip; тесты: канал скольжения (3,0,0 на полу), остановка мяча роллингом (0.11 vs 5.71 контроль, 240 шагов), убийство спина торсией, wide↔scalar паритет на эллипсе, сходимость в чистое качение (slip=0, v=8·5/7). **Half-depth contact point ✅ 2026-09-11 (найден через friction-тесты):** `sphere_vs_obb` ставил точку как середину (центр шара, поверхность) — на глубине r/2; все рычаги трения/моменты были половинными, катящийся шар сходился в фантомное half-rolling v = ω·r/2 с живым проскальзыванием (измерено точно: v=1.923295 при ω=−7.69318, якоря кэша на глубине 0.25). Исправлено на конвенцию `sphere_vs_sphere` (поверхность + pen/2); теперь чистое качение сходится в slip=0 и v=8·5/7 — аналитика твёрдого шара. Тест с негативным контролем (на старом коде slip ровно −v). Снапшот цел (сфер в нём нет). Роллинг-тест перекалиброван на честную геометрию (240 шагов). **GPU bulk v2 ✅ 2026-09-11:** `WgpuContactSolver::solve` — один энкодер вместо сабмита+блокирующего wait на итерацию (back-to-back compute passes, per-pass params через dynamic uniform offsets из params-таблицы; границы диспатчей те же → Jacobi-модель неизменна, шейдер не тронут, изотропный путь как на CPU). Убран главный балк-налог: было ~100+ CPU round-trip'ов на шаг (итерации × сабстепы), стал 1 upload + 1 submit + 1 wait на solve. Настоящий AVBD-порт (affine bodies, Hessian+LDL в шейдере) — отдельно: нужны matrix/scratch-хелперы в kernel DSL, их нет. Верификация честно: локальный `check --features gpu` убит OOM (чужой rustpython-билд съел 16ГБ), API сверен с сорцами wgpu 30.0.1 + ин-репо использованием; в CI покроют `solve_params`-тест, naga-валидация (шейдер тот же) и оба `gpu_solver_*` (гоняют новый solve). **Contact events ✅ 2026-09-11 (Box3D `b3ContactEvents` parity):** begin/end по касанию со slop 5мм (Box2D linearSlop parity; speculative near-miss молчит), hit по approach > 1м/с pre-solve каждый сабстеп с per-step dedupe (включая speculative с подтверждённым импульсом); frozen-пары держат состояние без churn'а; drain `drain_contact_events()` в детерминированном порядке; тесты: begin/hit/end/gap-silence/frozen-silence/run-determinism. Снапшот не дрогнул. **Prismatic ✅ 2026-09-11 (Box2D point-to-line):** 2 perp-линейных + 2 угловых равенства, лимит/мотор вдоль оси (та же one-sided + force-clamp дисциплина, что у revolute), reference_length с позы сборки, warm start через мировые компоненты (фрейм-независим); тесты: слайдер без провисания, лимит, мотор, реалignment. Снапшот не дрогнул. **Angular CCD rework ✅ 2026-09-10:** гейт-зеркало linear (`R·angle ≤ 0.5·min_dim` вместо плоских 15° — тонкие тела раньше, кубы реже), frictionless-отклик (`Δω = J·I⁻¹(r_c×n̂)`, тангенциальный спин выживает; плоские удары останавливаются как раньше) + energy-cap для жёстких рычагов (блендер 500 рад/с пойман тестом), uniform bound оставлен осознанно (доказан straddle-free); снапшот не дрогнул, tiled/hetero без регресса. **Unified CCD impulse ✅ 2026-09-10 (п.2):** angular-удары бьются одним контактным импульсом `J = −(1+e)·vn_c/denom` по скорости точки контакта (спин считается, угловая реституция есть, тангенциальное сохраняется, спин-часть через energy-cap); linear-путь не тронут. Тесты: точные числа связанного удара, inelastic, stiff-диссипация, движковый спин-баунс вверх. Найдено и зафиксировано взаимодействие: speculative margin считается только по центрам (спин не входит) — поэтому чистый спин-TOI всегда доходит до CCD, а внутри 5 см margin дискретка может превентивно съесть спин и разоружить гейт (тест держит зазор 10 см). **Sleep foundation ✅ 2026-09-10 (v1 мирового сна):** статика рождается спящей — все frozen-skip'ы narrowphase действуют с первого шага (settled tiled 10k: narrow ~9→~3мс, пары те же); `wake_island` no-op для не-динамики; active-manifold фильтр упрощён до чистых asleep-флагов (попутно починен призрак kinematic-vs-sleeper — кинематика всегда бодрая и будит спящих); penetration-wake 1 см против телепортов/спавнов с нулевой скоростью; early-out учитывает driven kinematic (иначе платформа-призрак при спящем мире); контракт драйвера как в Box2D: телепорт обязан нести скорость. Тесты: born-asleep, floor-stays-asleep, kinematic-plow, teleport-wake. Снапшот не дрогнул. **100k verdict ✅ 2026-09-10 (п.1):** settled-100k = 0.84мс/шаг (early-out; O(n)-проверки сна); active-100k (islands) ≈ 74мс: 2× `broadphase.update` ~62мс + island/sleep ~6мс + narrow/solver ~4мс (по активности); cold-transient 100k ~1.9с разово (solver cold-resolution). Полная атрибуция — новые поля `StepTiming::{island_ms, trigger_ms}`; trigger-detect загейчен при отсутствии триггеров (−11мс). Вывод: chunk-skip бэкендов ОТКЛОНЁН с данными — спящие миры уже early-out'ят за 0.84мс, бодрствующим скипать нечего; остаточный рычаг под 60fps на 100k-active — параллельный broadphase (не взят). **Pair bucketing ✅ 2026-09-10:** per-substep фильтр пар превращён из 12 перепроверок в один стабильный counting-sort по req за шаг (бакет s = суффикс class>s, побитово тот же вход narrowphase); settled tiled 10k: narrow ~9→0.7мс, hetero-пары те же (5642), homogeneous-сцены не затронуты (гейт), снапшот не дрогнул. **Kinematic CCD отложен с доказательством:** тонкая спящая жертва (4 см) против стены 10 м/с едет на плуге 7.5 м без туннелирования — speculative margin покрывает travel всегда, отдельный свип драйверов дал бы только качество отклика (тест `fast_kinematic_plow_carries_thin_sleeper`). **2026-09-01 — box↔capsule как честный контакт (G1-остаток закрыт) и fully analytic swept-volume TOI ✅** (`distance::shape_distance` + `box_vs_capsule` в обоих narrowphase-путях, `cast_shape` conservative advancement по точной distance, `shape_max_radius` gap/(|disp|+r·angle) + binary 10); **2026-09-02 — SAT cache ✅, 2026-09-05 — lock-free `DashMap` shared parallel+sequential + расширенный EPS; joint limits/motors ✅ 2026-09-05** (`RevoluteLimit`/`RevoluteMotor`, one-sided velocity limit + torque-clamped motor).
3. ~~`shapecast` — пустая заглушка~~ **Закрыто (G6)** — честный shapecast через conservative advancement (`distance.rs`), покрыт тестами.
4. ~~Движок рендера не связан с ECS-сценой в браузере~~ **Частично закрыто** — WASM-viewport рендерит живую сцену из `/api/scene`; физика со сценой браузера по-прежнему не связана (см. План B в PLAN.md).
5. **Broadphase scaling** — закрыто на 2026-09-01: `UniformGrid cell 8.0` — production default (6.18× vs SAP, no-alloc, 1×/кадр, per-body substeps фильтр), `tiled 10k ~15.8мс` — бюджет 60 FPS достигнут. SAP — fallback. **п1 incremental ✅ 2026-09-01** (`body_cells`/`prev_meta`, dirty-set + retained, heuristic >50% → full rebuild). Следующий кандидат — persistent `DynamicAabbTree` с moved/active-body queries; adaptive cell size — опционально. **п2 narrow cache ✅ + SAT cache ✅ 2026-09-02, lock-free ✅ 2026-09-05** (общий `DashMap`, axis-only reuse + расширенный EPS; замер `perf_probe`: narrow many_islands 1.98→1.62 мс, hetero 1.87→1.63 мс, islands_grid 0.77→0.68 мс; tiled 10k steady-state ~14–15 мс/кадр). Разбор: [`docs/quality/broadphase-reference-2026-08-29.md`](docs/quality/broadphase-reference-2026-08-29.md). **Adaptive broadphase ✅ 2026-09-09:** `BroadPhaseKind::Auto` — analytic SAP↔Grid↔Tree routing с гистерезисом (≤512 тел → sweep, sparse → tree, ≥1500 плотные → grid cell 8.0, middle band → grid только при coarse-телах >64; 3 голоса + cooldown 60), победитель виден через `auto_active_broadphase()`; probe: tiled 10k → grid, sparse 10k → tree (10.4мс). **Adaptive cell ✅ 2026-09-09:** липкая 8.0 (единственная settled-валидированная) + уход по сильному плотностному сигналу (`raw<3` → 2.0 для плотных 3D-упаковок, `raw>12` → 16.0 для sparse; coarse-тела исключены из оценки), переоценка ≤1/60 апдейтов или при росте >25%. **Tree pair buffering ✅ 2026-09-09:** персистентный буфер пар + точный dirty (swept/meta), full-rebuild при смене числа тел; settled tiled 10k: tree 20.2→~12мс (бьёт grid-8 16.1мс). **Tree correctness ✅ 2026-09-09:** найден и закрыт пропуск пар быстрыми телами (fat строился от base, а фильтр — от swept; негативный контроль: 2 теста валятся на старом коде) — теперь fat от swept + реинсерт телепортированной статики и смена static/dynamic + scratch-буфер запросов; пары == grid на матрице 5×10k. **Tree rotations ✅ 2026-09-09:** AVL-сингл-ротации на подъёме + `height` в `refit_node` (глубина 87→≤24 на 2k); матрица 10k теперь бьёт grid на tiled/giant/islands и вровень на sparse/hetero; 100k tiled: tree 759 мс vs grid-8 604 мс (паритет, ~10x лучше исторических 8 с). Явный `DynamicAabbTree` остаётся opt-in для dense-миров; Auto ведёт туда sparse-миры. **Step budget ✅ 2026-09-09:** `StepBudget` (default 200k pair×substeps, floor 4) — детерминированный shed сабстепов, никогда пар и никогда по wall-clock; калибровка держится в стороне от всех ≤10k сцен (tiled cold 170k), бьёт только на 100k-масштабе; shed виден через `last_substep_shed()` (seam-маркер) и в `probe_100k`; пары при shed побайтово те же (тест). Честный предел: покоящийся cold-start гигант режется только слипом/кэшами, бюджет bound'ит скоростной множитель. **Scheduler narrowphase ✅ 2026-09-09:** spike доказал паритет (`narrow_pair` выделен как shared kernel): flat rayon 7.8мс vs `run_levels`/32 шарда 6.1мс на 3k пар (расхождение было гранулярностью, не оверхедом). Порт: `detect_collisions_into` идёт через `run_levels` (K≈4×потоков, пул `NarrowShardPool`, порядок склейки = детерминизм); settled tiled 10k: **16.1→9.8мс (−38%, criterion p<0.05, A/B тоггл)**; cold без изменений (80мс). Всплеск 300мс по дороге — термалка под чужим билдом (load 10.6), не код. **Scheduler islands ✅ 2026-09-09:** velocity+position диспатчи идут через общий `dispatch_islands` helper (один scheduler-уровень, нода на остров, uncontended Mutex-refs, порядок фиксирован = детерминизм); A/B тоггл: 120 vs 124мс (паритет — распределение то же, выигрыш архитектурный: в физике больше нет прямого rayon, весь параллелизм через `run_levels`). **Determinism (Box3D-level) ✅ 2026-09-09:** (1) детерминированные хэшеры везде на sim-путях (`rustc-hash` Fx, также `DashMap` с `FxBuildHasher`; fma-интринсиков нет — аудит); (2) thread-тест побитово 1-vs-32 + новый run-to-run тест; (3) canonical snapshot `tests/data/determinism_snapshot.hex` (ARM), CI на x86_64 ловит contraction/codegen-дрейф; детерминированный режим = wide off (±1ulp). Rollback-детерминизма нет — как у Box3D (кэши), зафиксировано честно. **Gyroscope ✅ 2026-09-10:** `IntegrateVelocities` после торка решает Ньютон-Рафсон `I·(w2−w1) + h·(w2×I·w2) = 0` в body-frame (идея Box3D `b3IntegrateVelocitiesTask`, реализация своя diagonal-only: 3 фикс. итерации через `solve_small`, точные skip-гейты для изотропных тел и покоя). Тесты: флип Джанибекова (min_dot −1.0, dL 5.5%, dE 10.8% за 300 шагов), стабильность старшей оси, побитовая нетронутость куба; canonical snapshot не дрогнул (там все тела изотропные), tiled/hetero 10k без регресса.
