# PLAN — план реализации Ornis

Рабочий документ, синхронизированный с кодом (не переписан из старых
планов). Дополняет [`README.md`](README.md) (текущее состояние по
компонентам) и [`IDEAS.md`](IDEAS.md) (архитектурные идеи).

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
  IPC-типы `UiCommand`/`GameEvent` сохранены в `src/ipc.rs`.

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
  (этот коммит). Покрытие строк: 45.77% (базовая точка).

## 🟡 Частично — что достроить

- **Remote Editor**: REST есть; WebSocket — нет; в режиме `editor-only`
  обработчик команд не читает канал (`_cmd_rx` в `src/main.rs`) —
  команды из `POST /api/command` теряются; `editor.js` не подключён
  к REST (иерархия/инспектор статичны).
- **Command-Based Sync**: CPU-очередь + residency tracker и базовое
  GPU-исполнение (compute dispatch + flush, есть тест) — есть;
  автоматической data residency «копировать только при необходимости»
  (SmartBuffer) — нет.
- **Линтер параллелизуемости**: `compile_warning!` от `#[smart_pipeline]`
  есть; IDE-интеграции и расширяемого набора правил нет.
- **WASM**: сцена статичная (однократная загрузка `scene.ron`); нет
  связи с живым ECS и ввода (камера, гизмо).
- **Component Packing**: `#[derive(Pack)]` генерирует wrapper-типы,
  но не интегрирован автоматически в `for_each_entity`/`smart_pipeline`.

## Дорожная карта (по приоритетам)

### a. Протокол движок ↔ редактор

Обработчик `cmd_rx` в `editor-only` режиме (команды из
`POST /api/command` исполняются, а не теряются) → `GET /api/scene`
(сериализация иерархии/компонентов из живого ECS) → подключение
`editor.js` к REST (иерархия, инспектор, редактирование через
`/api/command`) → WebSocket вместо polling `/api/events`.

### b. Живой ECS в браузере

WASM-canvas рендерит актуальное состояние ECS (двусторонняя
синхронизация сцены), ввод из браузера в движок, орбитальная камера.

### c. Фаза 6 — Скриптинг

Rhai → Batch API (`engine.batch_add(...)` — один FFI-вызов вместо
100k) → hot reload → Rune → Python (rustpython). FFI-биндинги
оборачивают переменные скриптов в прямые указатели на ячейки
Sparse Set.

### d. Фаза 7 — Asset Pipeline (браузерная интерпретация)

Hot reload сцен/мешей/`.mtlx`: сервер следит за `assets/` и `editor/`
(`notify`), фронтенд перезагружает сцену через REST/WebSocket.
Build-time генерация бинарных слепков для Sparse Sets — позже.

### e. Качество (продолжение)

Покрытие ядра до 60%+ (сейчас 45.77% по workspace); первый полный
mutants-прогон на ornis-core с фиксацией mutation score; ночные
fuzz-прогоны (corpus растить, краши → регрессионные тесты);
criterion-baseline как точка отсчёта производительности;
flamegraph/dhat — на perf-спринт.

### f. Дальше по старому плану

Deferred/Forward hybrid рендер (G-buffer, lighting pass) →
NUMA-aware allocation → кроссплатформенные прогоны (Linux/Windows
CI, miri) → адаптеры Rapier/Jolt за `PhysicsEngine` → документация
API (rustdoc) и релизная упаковка.

## ❌ Не делать / отложено (решения владельца)

- **Нативный UI** — удалён (`29e3547`): доведение собственного
  UI-движка до production сопоставимо с командой браузерного движка;
  редактор живёт в браузере. Не возвращаться.
- **HVM2/Bend как compute-бэкенд** — идея на далёкое будущее,
  активной работы нет.
- **Формальная верификация** — отложено бессрочно; вместо неё
  proptest + mutants + fuzz.

---
## Приложение B — Рендерер и физика: план работ (черновик для ревью)

> 🔎 Черновик по двум движкам. Номера пунктов — для ваших корректировок.

### B1. Рендерер (`crates/render`) — план

- **R1.** Решить судьбу G-buffer/Lighting-путей: в коде есть структуры
  `GBufferTextures`/`LightingPass`/`CompositePass`, но README помечает
  Deferred/Forward hybrid как ❌. Либо довести gbuffer-путь до активного
  (G-buffer → lighting pass → composite), либо удалить как спекулятивный
  каркас. Быстрый шаг: поставить README-статус точно по коду.
- **R2.** `Renderer3D` под живой ECS: поднять бюджет с `max_objects=256`,
  перейти на инстансинг через `InstanceData`, поддержать динамические
  меши/трансформы без пересоздания буферов.
- **R3.** Материалы: масштабировать OpenPBR-бюджет (`max_materials=64`),
  вынести параметры в общий descriptor set, готовить пайплайн под
  переключение материалов без rebind.
- **R4.** Свет: сейчас только направленные (до 4, `GpuLight`) — добавить
  point/spot + shadow maps (когда понадобится).
- **R5.** Единый путь нативных и WASM-бэкендов через `RenderBackend`
  (недублирование wgpu-кода для браузера).
- **R6.** Связать рендер с ECS-сценой в браузере: WASM-canvas рендерит
  актуальное состояние, ввод/камера из браузера (перекликается с
  «b. Живой ECS в браузере»).

### B2. Физика (`crates/physics`) — план работ

> Архитектура сверена с реальными исходниками **Box3D** (`github.com/erincatto/box3d`,
> soltimeer 3D-преемник Box2D, Catto, июнь 2026) и **Jolt** (`github.com/jrouwe/JoltPhysics`):
> multisolver = joint-солверы + широкий (SIMD) контактный солвер + manifold-солвер,
> стадии WarmStart→Solve→IntegratePositions→Relax→Restitution, warm starting,
> split impulse (velocity/position раздельно), constraint graph → острова + sleeping,
> speculative CCD, мягкие констрейнты.
> Наш текущий солвер — одиночный проход sequential impulse, по 1 контакту на пару,
> без ориентации; это стартовый стиль. Ниже — поэтапный апгрейд, каждый этап с гейтом тестов.

| Этап | Что делаем | Основание (реальный код) | Статус |
|---|---|---|---|
| **G1** | **Ориентация + угловая динамика**: `RigidBody` с `orientation: Quat`, `angular_velocity`, инерцией (тензор), интеграция (semi-implicit Euler) + вращение; коллизия с учётом ориентации (sphere↔OBB, OBB↔OBB по SAT, капсула по оси тела) | Box3D `b3BodyState` (linear+angular velocity), Jolt `Body` (инерция) | ✅ Реализовано |
| **G2** | Контактные манифолды (несколько точек на пару) + кэш + warm starting. **G2a**: структуры (до 4 точек), OBB↔OBB vertex-face манифолд, солвер по точкам с warm-start кэшем импульсов. **G2b**: стабильность покоя (нужны итерации солвера/velocity bias — сейчас бокс на статичном полу может провалиться), кэш контактных точек. | Box3D `b3ContactConstraintWide`/`ManifoldConstraint` (симметрический `cached_manifold`), Jolt `mContactPoints` | 🟡 G2a ✅, G2b в работе |
| **G3** | Split impulse: раздельные velocity/position проходы + мягкие констрейнты (`b3MakeSoft`-аналог) | Box3D стадии `IntegrateVelocities`/`IntegratePositions`, `b3Softness`; Jolt `ContactConstraintPart` | ❌ |
| **G4** | Constraint graph → острова + sleeping (awake/static кэш) | Box3D `b3ConstraintGraph`/`b3SolverSet`/`sleepVelocity`, Jolt islands | ❌ |
| **G5** | Joints: ball (spherical), revolute; через под-солверы | Box3D `spherical_joint`/`revolute_joint`, Jolt `ConstraintPart/*` | ❌ |
| **G6** | CCD (speculative контакты + TOI `b3SolveContinuous`) и честный `shapecast` | Box3D `B3_SPECULATIVE_DISTANCE`, `b3SolveContinuous` | ❌ |
| **G7** | Производительность: wide/SIMD-контактный путь + параллельные таски (граф-раскраска, CAS-блоки) | Box3D `solver.h` (WideContact, per-block syncIndex, bepu-inspired) | ❌ |

**Качество на каждом этапе (обязательный гейт):** сценарии (стек из N бокстов стоит N сек без дрифта; сфера спокойно покоится; sleep/пробуждение корректны), proptest-инварианты (после settle нет проникновения; сохранение импульса; детерминизм `RAYON_NUM_THREADS=1` vs `=32`); fuzz (нет NaN/паник); mutants на солвер. Согласуется с философией Strong Confluence.

**Логика порядка**: G1 (ориентация) — фундамент, без него невозможны ни joints, ни настоящие манифолды; G2-G3 — стабильность/качество; G4 — производительность; G5-G6 — поверхность; G7 — peak-производ.

**G1 (2026-08-07, реализовано; остатки дочищены в тот же день)**: добавлены `orientation: Quat`, `angular_velocity`, инерция (диагональный тензор через `Shape::inertia`), `torque`; полуимплицитная интеграция линейной + угловой динамики (вращение кватерниона через `from_scaled_axis`); коллизии с учётом ориентации: sphere↔OBB (в локальном фрейме), OBB↔OBB (полный SAT по 15 осям), капсула по оси тела, **sphere↔capsule**; resolve: угловой импульс (world-инерция через `mul_inv_inertia`) **и вращательная позиционная коррекция** (эффективная масса c rotation + slop/релаксация). **`shapecast` реализован** (консервативный свип пере-юзом узкой фазы, 64 шага). Тесты: 11 (к прежним +4 добавлены `sphere_capsule_collision`, `shapecast_hits_body`). **Остаток → G2**: пара box↔capsule (и капсула↔бокс) всё ещё `None`.
