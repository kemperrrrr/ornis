# План реализации игрового движка на Rust

> Пошаговый план на основе концептуального диалога.  
> Философия: двигаться от ядра к оболочке, от простого к сложному, MVP-first.

---

## Фаза 0. Инфраструктура и подготовка

**Цель**: создать репозиторий, настроить CI/CD, выбрать и зафиксировать версии ключевых зависимостей.

| # | Задача | Технологии | Критерий завершения | Статус |
|---|--------|------------|---------------------|--------|
| 0.1 | Инициализация workspace Cargo (монорепозиторий) | `cargo workspace` | Структура `crates/core`, `crates/macros`, `crates/ui`, `crates/wgpu_backend`, `examples/` | ✅ |
| 0.2 | Зафиксировать toolchain (MSRV) | `rustup`, `cargo` | `rust-version = "1.85"` в корневом `Cargo.toml` | ✅ |
| 0.3 | Настроить CI: clippy, fmt, tests, docs | GitHub Actions | Красный билд при warning'ах | ✅ |
| 0.4 | Выбрать и протестировать зависимости ядра | `rayon 1.10`, `wgpu`, `syn 2.0`, `quote`, `proc-macro2` | `cargo check` проходит на всех таргетах | ✅ |

---

## Фаза 1. Ядро движка — Sparse Sets + SmartStore (MVP)

**Цель**: создать сверхскоростное, потокобезопасное хранилище данных, готовое для CPU и GPU.

| # | Задача | Технологии | Статус |
|---|--------|------------|--------|
| 1.1 | Реализовать `Entity` (Id) и `ComponentStore<T>` (Sparse Set) | Чистый Rust (`Vec<T>`, `Vec<usize>`) | ✅ |
| 1.2 | Реализовать `SmartStore` — менеджер всех лент | `HashMap<TypeId, Box<dyn Lane>>`, `RwLock` | ✅ |
| 1.3 | Методы `create_entity`, `insert<T>`, `read_lane<T>`, `write_lane<T>` | `downcast_ref`, `RwLockReadGuard`/`WriteGuard` | ✅ |
| 1.4 | Интеграция с `rayon` | `rayon::prelude::*` | ✅ |
| 1.5 | Бенчмарки ядра | `criterion` | ✅ |
| 1.6 | **Bitset Acceleration** | `fixedbitset` intersection/difference | ✅ |
| 1.7 | **Paginated Sparse Arrays** | `PageTable<T>` (страницы по 4K) | ✅ |
| 1.8 | Комплексный бенчмарк гибридной структуры | `criterion`, 4 подхода | ✅ |
| 1.9 | **Entity Recycling + Generational Indices** | `free_list`, gen-guard | ✅ |
| 1.10 | **Cache-Line Alignment** | `#[repr(align(64))]` на `ComponentStore` | ✅ |
| 1.11 | **Chunked Iteration** | `chunks_exact_mut(4)` + `into_tail()` | ✅ |
| 1.12 | **Lock-Free SmartStore** | `crossbeam-epoch`, `Atomic`, feature `lock-free` | ✅ |
| 1.13 | **Prefetch Intrinsics** | `_mm_prefetch` + `prefetch_iter!` макрос | ✅ |
| 1.14 | **Temporal Coherency Sort** | `defrag()` по entity ID | ✅ |
| 1.15 | **Hot/Cold Data Splitting** | `ColdComponentStore<T>` + `register_cold` | ✅ |
| 1.16 | **Встроенный физический движок** | Sweep-and-Prune + PBD solver + raycast | ✅ |
| 1.17 | **PhysicsEngine trait** | `step(dt)`, `raycast`, `add_body` | ✅ |

> **Принцип: Strong Confluence** — все параллельные системы движка ([`#[smart_pipeline]`, `#[gpu_pipeline]`, `for_each_entity!`) должны быть strongly confluent: результат не зависит от числа потоков или порядка обработки. Тесты с `RAYON_NUM_THREADS=1` и `RAYON_NUM_THREADS=32` — побитово одинаковый output. Критично для replay, сети и отладки.

**Ключевой код (целевой)**:
```rust
pub struct ComponentStore<T> {
    pub data: Vec<T>,           // Dense — итерация, GPU, prefetcher
    pub entities: Vec<Entity>,  // Обратная карта dense → entity
    sparse: PageTable<usize>,   // Paginated: страницы по 4K, ленивое выделение
    bitset: FixedBitSet,        // Битовая маска для SIMD intersection
}

**Paginated PageTable**:
```rust
const PAGE_SIZE: usize = 4096;

pub struct PageTable<T> {
    pages: Vec<Option<Box<[T; PAGE_SIZE]>>>, // ленивые страницы
}
```


---

## Фаза 2. Процедурные макросы — «невидимый ECS»

**Цель**: автоматизировать SOA-разложение структур и параллельные циклы без ручного кода.

| # | Задача | Технологии | Статус |
|---|--------|------------|--------|
| 2.1 | Создать крейт `smart-pipeline-macro` | `proc-macro = true`, `syn`, `quote` | ✅ |
| 2.2 | Макрос `#[derive(AutoPipeline)]` над структурой | AST-трансформация (`syn::DeriveInput`) | ✅ |
| 2.3 | Макрос `#[smart_pipeline]` над функцией | `syn::visit`, `quote` | ✅ |
| 2.4 | Макрос `for_each_entity!(store, \|a: &A, b: &mut B\| { ... })` | `proc_macro::TokenStream` | ✅ |
| 2.5 | Поддержка проекций (View-структуры) | Анализ полей в AST | ✅ (через `#[pack]`) |
| 2.6 | Анализ зависимостей для параллелизма задач | Граф чтения/записи компонентов | ⏳ отложено |
| 2.7 | ZST-маркеры `GpuLane`, `CpuLane`, `HybridLane` | Zero-sized types | ✅ |
| 2.8 | Трейт `LaneTarget` и его генерация | `syn` + `quote` | ✅ |
| 2.9 | `derive(PipelineConfig)` — статический профайлер | `syn::visit`, `quote` | ✅ |
| 2.10 | **Component Packing / SoA внутри компонента** | `#[pack]`, автогенерация `pack_register`/`pack_insert` | ✅ |

---

## Фаза 3. Автоматический CPU/GPU диспетчер

**Цель**: макрос сам решает, где выполнять код — на CPU или GPU.

| # | Задача | Технологии | Статус |
|---|--------|------------|--------|
| 3.1 | Интеграция `wgpu` | `wgpu` crate | ✅ |
| 3.2 | Автоматическое создание `wgpu::Buffer` из `ComponentStore.data` | `bytemuck`, `unsafe { std::slice::from_raw_parts }` | ✅ |
| 3.3 | Генерация WGSL из простых математических выражений | `syn` → строка WGSL | ✅ |
| 3.4 | Макрос `#[gpu_pipeline]` | Компиляция шейдера в `wgpu::ShaderModule` | ✅ |
| 3.5 | Smart Buffer (Data Residency) | Флаги `DirtyCPU` / `DirtyGPU` | ✅ |
| 3.6 | Динамический выбор CPU/GPU (Runtime) | Эвристика: размер массива + сложность функции | ✅ |
| 3.7 | Автопрофилировщик при старте | Микро-бенчмарки (100, 1k, 10k, 100k элементов) | ✅ |
| 3.8 | DSL-подмножество Rust для GPU (Compute-Subset) | `syn` + AST-валидация | ✅ |
| 3.9 | Статический профайлер функций (compile-time) | `syn::visit`, подсчёт `if`/`match` | ✅ |
| 3.10 | Сортировщик конвейеров (Pipeline Router) | ZST + `LaneTarget` | ✅ |
| 3.11 | Инструкции вместо данных (Command-Based Sync) | `wgpu::CommandBuffer`, `ComputePass` | ✅ |
| 3.12 | ZST-диспетчеризация `ExecuteLane` | ZST (`GpuExecutor`, `CpuExecutor`) | ✅ |
| 3.13 | **Compute Shader Pre-warm** (PSO-кэш) | `HashMap<(TypeId, TypeId), wgpu::ComputePipeline>`, сериализация на диск | ✅ |
| 3.14 | **LEAK-паттерн для GPU-блоков** (из HVM2) | Shared memory, `wgpu::ComputePass` на блок 64–256 элементов | ✅ |
| 3.15 | **HVM2 как опциональный compute-бэкенд** (future work) | HVM2 runtime, Bend → HVM2 → C/CUDA | ⏳ отложено |

**Ограничения GPU-подмножества (Compute-Subset / DSL)**:
- Разрешено: арифметика, векторная математика, простые условия (`if` без сложного `else`/`match`).
- Запрещено: `Vec`, `String`, `Box`, динамическая аллокация.
- Запрещено: рекурсия (ошибка компиляции).
- Запрещено: сложный control flow (глубокие вложенные `match`, циклы `while` с неизвестными границами).
- Макрос выдаёт понятное сообщение: *«операция X не поддерживается в контексте kernel»*.

---

## Фаза 4. UI-система + Материалы — Headless DOM + WGPU-рендеринг + OpenPBR/MaterialX

**Цель**: рендерить HTML/CSS через GPU, а JS-логику выполнять внутри движка. Поддержка OpenPBR и MaterialX для материалов.

| # | Задача | Технологии | Критерий завершения |
|---|--------|------------|---------------------|
| 4.1 | Парсинг HTML/CSS в Rust + интеграция Servo | `html5ever`, `lightningcss`, `taffy`, `servo-core` (опционально) | Полноценный парсинг; либо через `html5ever`/`lightningcss` (легковесно), либо через `Servo` (полная совместимость) |
| 4.1a | Интеграция Servo как библиотеки | `servo`, `webrender` | HTML/CSS/JS исполняется полноценно; layout-результаты попадают в Sparse Sets, а рендеринг — в `wgpu` |
| 4.2 | Векторный рендеринг (SVG + UI) | `vello`, `usvg`, `resvg` | SVG-иконка и прямоугольник кнопки рисуются через `wgpu` в 144+ FPS |
| 4.3 | Встроенный JS-интерпретатор | `boa_engine` или `rquickjs` | JS-скрипт выполняется внутри Rust-процесса, вызывает `globalThis.document.createElement` |
| 4.4 | Headless DOM (шим для JS) | Реализация `VirtualElement`, `document`, `window` | React запускается без реального браузера, создавая виртуальные узлы |
| 4.5 | Связь JS ↔ Rust (Sparse Sets) | FFI, бинарные вызовы, `crossbeam-channel` / `flume` | Движение ползунка в JS мгновенно обновляет `ComponentStore<UIStyle>` |
| 4.6 | Двухсторонний IPC (UI-поток ↔ игровой поток) | Lock-free MPMC канал | UI не блокирует игровой цикл; 60+ FPS игры + 144 FPS UI |
| 4.7 | In-Game Editor | Overlay Render Pass | По кнопке (например, `~`) открывается редактор повью игрового мира |
| 4.8 | Удалённый редактор (Web) | WebSocket / HTTP | Игра на ПК, редактор на планшете/ноутбуке в браузере |
| 4.9 | **Система материалов: OpenPBR Surface** | `naga`, WGSL-генерация | Поддержка OpenPBR (металличность, шероховатость, clearcoat, sheen, subsurface); материал → `ComponentStore<Material>` → `wgpu::BindGroup` |
| 4.10 | **Поддержка MaterialX** | `materialx` crate или кастомный парсер `.mtlx` в WGSL | Импорт MaterialX-нод-графа; компиляция в шейдеры wgpu; совместимость с VFX-пайплайном |
| 4.11 | **Рендер-пайплайн материалов** | Deferred + Forward hybrid | OpenPBR-шейдеры рендерятся одним compute pass на GPU; переключение между собственным рендером и внешним — через `RenderBackend` trait |

---

## Фаза 5. Скриптовые языки (мультискриптинг)

**Цель**: дать геймдизайнерам выбор языка без потери производительности.

| # | Задача | Технологии | Критерий завершения |
|---|--------|------------|---------------------|
| 5.1 | Интеграция `Rhai` | `rhai` crate | Скрипт `.rhai` может читать/писать `Position` из SmartStore |
| 5.2 | Интеграция `Rune` | `rune` crate | Скрипт `.rn` с синтаксисом Rust компилируется в байт-код |
| 5.3 | Интеграция Python (опционально) | `PyO3` или `rustpython` | Python-скрипт `entity.speed = 10` мгновенно записывается в Sparse Set |
| 5.4 | FFI-оптимизация для скриптов | Прямые указатели на ячейки `ComponentStore.data` | Нет boxing'а; скриптовая переменная = обёртка над `&mut T` |
| 5.5 | Hot Reload скриптов | Файловый вотчер (`notify`) | Изменение `.rhai` → перезагрузка без пересборки движка |
| 5.6 | **Batch API для скриптов** | `engine.batch_add("Position", "x", 1.0)` | 1 FFI-вызов вместо 100k; внутри `rayon par_iter_mut` |
| 5.7 | **Bend/HVM2 как скриптовый язык** (future work) | `hvm2` + `bend-lang` crates | Pure-функции на Bend автоматически параллелятся на CPU/GPU; требуется FFI в HVM2 (ожидается) |

---

## Фаза 6. Конвейер ресурсов (Asset Pipeline)

**Цель**: автоматически глотать UI-ресурсы без ручных макросов.

| # | Задача | Технологии | Критерий завершения |
|---|--------|------------|---------------------|
| 6.1 | Скрипт сборки `build.rs` | `std::fs`, `walkdir` | При `cargo build` автоматически сканируется папка `assets/ui/` |
| 6.2 | Компиляция CSS в структуры | `lightningcss` | `.css` → `ui_styles.rs` со структурами для `ComponentStore<UIStyle>` |
| 6.3 | Компиляция SVG в геометрию | `usvg` | `.svg` → массив векторных путей для `Vello` |
| 6.4 | Вшивание JS в бинарник | `include_bytes!` | `node_modules/react/index.js` доступен в рантайме как `&[u8]` |
| 6.5 | Макрос одной папки `#[compile_frontend_pipeline!("ui/src/")]` | `proc_macro` + `std::fs` | Одна строка в `main.rs` загружает всё UI |
| 6.6 | Hot Reload в dev-режиме | `notify` crate | Изменение `button.css` → мгновенное обновление `ComponentStore<UIStyle>` и перерисовка экрана |

---

## Фаза 7. Адаптер для сторонних библиотек

**Цель**: использовать `crates.io` без модификации исходников.

| # | Задача | Технологии | Критерий завершения |
|---|--------|------------|---------------------|
| 7.1 | Auto-Generated Proxy Views | Временные стек-структуры в L1-кэше | Сторонняя функция `update_position(body: &RigidBody, vel: &mut Velocity)` получает данные из плотных массивов без cache miss |
| 7.2 | SIMD/векторизация через макрос | `std::simd` (или `packed_simd`) | Макрос пакует 4/8/16 элементов в регистры AVX2/NEON перед вызовом |
| 7.3 | Inlining-оптимизация | `#[inline]` + `rustc` LTO | Чужая функция «растворяется» в `zip`-цикле по Sparse Sets при `--release` |
| 7.4 | Graceful degradation (CPU-fallback) | Эвристика ветвлений | Если сторонняя функция содержит сложные `if/else` → выполняется на CPU, но всё равно в Rayon-конвейере |
| 7.5 | Паттерн «Границы данных» (Data Boundaries) | Borrow Checker | Чужая функция физически не может трогать данные вне выделенных лент |
| 7.6 | **Трейт `PhysicsEngine`** — абстракция физики | `trait PhysicsEngine { fn step(&mut self, dt: f32); fn raycast(&self, ...) -> Option<Hit>; }` | Собственный движок (Sweep-and-Prune + constraints) — impl по умолчанию; Rapier, Jolt, PhysX — опциональные реализации |
| 7.7 | **Трейт `RenderBackend`** — абстракция рендера | `trait RenderBackend { fn begin_frame(&mut self); fn draw_mesh(&mut self, ...); fn end_frame(&mut self); }` | `WgpuBackend` — impl по умолчанию; возможность подключить альтернативный рендер (консоли, HDRP) |
| 7.8 | **Расширение физики для производительности** (future) | `#[cfg(feature = "physics-perf")]`, SIMD-солвер, IPC-связь с выделенным physics-тредом | Физика выносится в отдельный поток/процесс (IPC через shared memory); бенчмарк: 10k тел с коллизиями <1 ms |

---

## Фаза 8. Встроенный линтер (Compile-Time Warnings)

**Цель**: обучать разработчика и предотвращать неэффективные паттерны.

| # | Задача | Технологии | Критерий завершения |
|---|--------|------------|---------------------|
| 8.1 | AST-Visitor для поиска медленных циклов | `syn::visit::Visit` | Макрос `#[smart_pipeline]` находит `for` с `store.get()` / `store.get_mut()` внутри |
| 8.2 | Генерация `compile_warning!` | `#[deprecated]` trick | При сборке выводится яркое предупреждение с рекомендацией использовать `for_each_entity!` |
| 8.3 | Интеграция с IDE | Rust Analyzer | Предупреждение подсвечивается в VS Code / Vim / Emacs |
| 8.4 | Расширяемость линтера | Плагинная система | Новые правила добавляются как отдельные `Visitor`'s |

---

## Фаза 9. Кроссплатформенность (Desktop + WebAssembly)

**Цель**: один код — все платформы.

| # | Задача | Технологии | Критерий завершения |
|---|--------|------------|---------------------|
| 9.1 | Таргеты Desktop | `wgpu` (Vulkan/Metal/DX12) | `.exe` (Windows), `.app` (macOS), бинарник (Linux) |
| 9.2 | Таргет WebAssembly | `wasm32-unknown-unknown`, `wasm-bindgen` | Браузер запускает `.wasm`, использует WebGPU через `<canvas>` |
| 9.3 | Единый рендер-пасс (игра + UI) | `wgpu::RenderPass` | UI и мир рисуются в одном вызове, шейдеры могут накладываться друг на друга |
| 9.4 | Условная компиляция платформ | `#[cfg(target_arch = "wasm32")]` | Desktop использует нативные потоки, WASM — Web Workers |
| 9.5 | Размер бинарника | `strip`, `upx`, `wasm-opt` | Desktop: <20 МБ, WASM: <5 МБ (без текстур) |
| 9.6 | **NUMA-Aware Allocation** | `libc::mbind`, `VirtualAllocExNuma` | `ComponentStore<T>::with_numa_affinity(node_id)`; критично для MMO-серверов на 32+ ядрах |

---

## Фаза 10. Полировка, документация, релиз

| # | Задача | Технологии | Критерий завершения |
|---|--------|------------|---------------------|
| 10.1 | Полное покрытие тестами | `cargo test`, `miri` (для unsafe) | >80% покрытие ядра, все edge cases Sparse Set |
| 10.2 | Документация (rustdoc + book) | `mdbook` | Пошаговый tutorial «Создай свою первую игру за 30 минут» |
| 10.3 | Примеры и шаблоны | `cargo generate` | Шаблоны: 2D-шутер, 3D-платформер, UI-редактор |
| 10.4 | Benchmark suite | `criterion`, custom harness | Сравнение с Bevy, Unity DOTS, bare metal на типовых сценариях |
| 10.5 | Публикация на crates.io | `cargo publish` | `smart-engine = "0.1.0"` доступен для установки |

---

## Критический путь (Minimal Viable Product)

Чтобы получить **работающий прототип**, достаточно фаз 1–3 + 4.1–4.3:

```
Фаза 0 (1 неделя) → Фаза 1 (2 недели) → Фаза 2 (3 недели) → Фаза 3 (4 недели) → Фаза 4.1–4.3 (2 недели)
```

**Итого MVP**: ~12 недель (3 месяца) для одного разработчика на полную ставку.

После MVP можно демонстрировать:
- 100k частиц с физикой, автоматически улетевших на GPU.
- Простой UI-инспектор, написанный на React, но рендерящийся через WGPU внутри игры.
- Изменение ползунка в UI → мгновенное обновление данных в Sparse Set.

---

## Риски и их митигация

| Риск | Митигация |
|------|-----------|
| Генерация WGSL из сложного Rust-кода слишком сложна | **DSL-подмножество**: ограничить `#[kernel]` чистой математикой, без циклов и рекурсии; макрос выдаёт понятную ошибку компиляции |
| Borrow Checker мешает автоматическим прокси-структурам | **ZST + сложные абстракции**: `EntityToken`, `View<'a, T>`, `PhantomData` — Borrow Checker становится союзником, а не врагом; `unsafe` только внутри макроса, обёрнутое в safe API |
| Headless DOM не совместим со всеми JS-фреймворками | **Servo как библиотека**: использовать `servo-core`/`webrender` для полноценного парсинга и исполнения, но перенаправлять геометрию в `wgpu`-конвейер; вес 50–80 МБ приемлем для AAA-целей |
| Копирование данных CPU↔GPU убивает производительность | **Data Residency + Инструкции вместо данных**: копировать только при `Dirty`-флаге; CPU отправляет команды (`wgpu::CommandBuffer`), а не массивы, туда где данные уже лежат |
| Сторонние библиотеки используют `std::collections` (HashMap) | **Proxy-views + ZST-адаптер**: временные структуры в L1-кэше; `LaneTarget` маркирует чужие типы как CPU-only, а математические — GPU |
| Runtime-эвристики CPU/GPU ошибочны | **Статический профайлер**: решение принимается на этапе компиляции (ZST + `const fn`), а runtime только подставляет `n` в готовую формулу |

---

## Используемые библиотеки (итоговый стек)

| Слой | Библиотека | Назначение |
|------|------------|------------|
| Ядро | `fixedbitset` | Битовые маски для SIMD intersection запросов |
| Ядро | `rayon` | Параллельные итераторы CPU |
| Ядро | `crossbeam-channel` / `flume` | Lock-free каналы между потоками |
| Графика | `wgpu` | Унифицированный API (Desktop + WebGPU) |
| Графика | `vello` | Векторный рендеринг на GPU |
| Графика | `taffy` | Расчёт CSS-разметки (flexbox/grid) |
| Парсинг | `html5ever` | Парсинг HTML |
| Парсинг | `lightningcss` | Парсинг и минификация CSS |
| SVG | `usvg`, `resvg` | Парсинг и растеризация SVG |
| JS | `boa_engine` / `rquickjs` | Интерпретатор JavaScript в Rust |
| Python | `PyO3` / `rustpython` | Интерпретатор Python в Rust |
| Скриптинг | `rhai`, `rune` | Встроенные Rust-подобные языки |
| Макросы | `syn`, `quote`, `proc-macro2` | Процедурные макросы, AST-анализ |
| Сериализация | `bincode`, `messagepack` | Бинарный обмен данными UI↔Rust |
| Файловый вотчер | `notify` | Hot Reload ресурсов |
| Бенчмарки | `criterion` | Измерение производительности |
| Биндинги | `wasm-bindgen` | Мост Rust ↔ JavaScript в WASM |

---

*План составлен на основе анализа 74-страничного PDF-диалога, датированного 24–25 июня 2026 г.*
