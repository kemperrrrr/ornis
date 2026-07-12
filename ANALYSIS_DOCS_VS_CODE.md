# Анализ соответствия документации (MD) и исходного кода Ornis

> Дата анализа: 12 июля 2026  
> Исходные документы: [key_ideas.md](key_ideas.md), [implementation_plan.md](implementation_plan.md), [SUMMARY.md](SUMMARY.md)

---

## Сводная таблица статусов

| Категория | Документация | Реальность в коде | Статус |
|---|---|---|---|
| **Ядро (Sparse Sets)** | Фаза 1 ✅ | Полностью реализовано | ✅ |
| **Процедурные макросы** | Фаза 2 (9/10) | Все макросы работают | ✅ |
| **CPU/GPU диспетчер** | Фаза 3 (14/15) | Runtime диспетчер реализован в `dispatcher.rs` | ✅ |
| **Аудио** | Фаза 4 (6/8) | База работает, GPU DSP нет | ✅ База |
| **UI + Материалы** | Фаза 5 (9/12) | OpenPBR ✅, MaterialX ✅, RenderBackend ✅ | ✅ |
| **Скриптинг** | Фаза 6 | Не начато | ❌ |
| **Asset Pipeline** | Фаза 7 | Не начато | ❌ |
| **Адаптеры библиотек** | Фаза 8 | PhysicsEngine ✅, RenderBackend ✅ | ✅ |
| **Линтер** | Фаза 9 | Анализ есть, warnings нет | ❌ |
| **Кроссплатформенность** | Фаза 10 | Desktop + WASM работает | ✅ |

---

## ✅ Полностью соответствует (реализовано и работает)

### 1. Ядро Sparse Sets — `crates/core/src/component_store.rs`
- **Документация**: [key_ideas.md:16-77](key_ideas.md#L16-L77), [implementation_plan.md:27-43](implementation_plan.md#L27-L43)
- **Код**: `ComponentStore<T>` с `data`, `entities`, `PageTable`, `FixedBitSet`
- **Фичи**: Bitset acceleration, Paginated Sparse Arrays, Cache-Line Alignment (`#[repr(align(64))]`), Chunked Iteration, Prefetch Intrinsics, Temporal Coherency Sort (`defrag()`), Hot/Cold Splitting (`ColdComponentStore`), Lock-Free (`lock_free_store.rs`), Entity Recycling + Generational Indices (`entity.rs`)

### 2. Процедурные макросы — `crates/macros/src/`
- **Документация**: [key_ideas.md:81-90](key_ideas.md#L81-L90), [implementation_plan.md:74-84](implementation_plan.md#L74-L84)
- **Макросы**: `AutoPipeline`, `Pack`, `for_each_entity`, `smart_pipeline`, `PipelineConfig`, `kernel`, `gpu_pipeline`
- **ZST диспетчеризация**: `GpuLane`, `CpuLane`, `HybridLane`, `LaneTarget`, `Route` в [pipeline.rs](crates/core/src/pipeline.rs)

### 3. OpenPBR Material — `crates/core/src/material.rs`
- **Документация**: [key_ideas.md:347-352](key_ideas.md#L347-L352), [implementation_plan.md:150](implementation_plan.md#L150)
- **Код**: 20 vec4 параметров, все BSDF (specular, transmission, subsurface, coat, fuzz, thin_film, emission), builder-паттерн, `bytemuck::Pod`
- **Примечание**: Перенесён в `ornis-core` для устранения циклической зависимости render ↔ materialx

### 4. WGSL PBR шейдеры — `crates/render/src/shaders/` + `shader.rs`
- **Документация**: [key_ideas.md:349-350](key_ideas.md#L349-L350), [implementation_plan.md:150](implementation_plan.md#L150)
- **Реализация**: GGX NDF, Smith-G masking-shadowing, Fresnel-Schlick, multiple lights, tone mapping (ACES), octahedral normal encoding

### 5. MaterialX Парсер — `crates/materialx/src/parser.rs`
- **Документация**: [key_ideas.md:353-357](key_ideas.md#L353-L357), [implementation_plan.md:151](implementation_plan.md#L151)
- **Код**: XML → AST (`MaterialXDocument`, `NodeGraph`, `NodeDef`, `Node`, `Input`, `Output`)

### 6. MaterialX → OpenPBR конвертер — `crates/materialx/src/codegen.rs`
- **Документация**: [key_ideas.md:356](key_ideas.md#L356), [implementation_plan.md:151](implementation_plan.md#L151)
- **Код**: `MaterialXConverter::to_openpbr()` — ищет граф `ND_open_pbr_surface_surfaceshader`, извлекает выходы в `OpenPBRMaterial`

### 7. MaterialX интеграция — `crates/materialx/` + `crates/render/src/render_backend.rs`
- **Документация**: [key_ideas.md:353-357](key_ideas.md#L353-L357), [implementation_plan.md:151](implementation_plan.md#L151)
- **Код**: 
  - Парсер `.mtlx` → AST (`crates/materialx/src/parser.rs`)
  - Конвертер в `OpenPBRMaterial` (`crates/materialx/src/codegen.rs:67-223`)
  - Factory-функция `create_render_backend()` в `crates/render/src/render_backend.rs:65-71`
  - `OpenPBRMaterial` вынесен в `ornis-core` для разрыва цикла зависимостей

### 8. Кроссплатформенность — Desktop + WASM
- **Документация**: [key_ideas.md:224-229](key_ideas.md#L224-L229), [implementation_plan.md:221-228](implementation_plan.md#L221-L228)
- **Работает**: `wgpu` + `winit` (Desktop), `wasm32-unknown-unknown` + WebGPU (Web)

---

## ⚠️ Частично реализовано / требует доработки

| Фича | Документация | Что есть в коде | Что не хватает |
|---|---|---|---|
| **CPU/GPU автодиспетчер** | [key_ideas.md:92-105](key_ideas.md#L92-L105), [implementation_plan.md:93-104](implementation_plan.md#L93-L104) | `Dispatcher`, `SmartDispatcher`, `CpuExecutor`, `GpuExecutor` в `crates/core/src/dispatcher.rs` | Полностью реализован: `decide(element_count)`, `CpuExecutor::execute`, `SmartDispatcher::execute_read`, тесты |
| **Command-Based Sync** | [key_ideas.md:136-146](key_ideas.md#L136-L146), [implementation_plan.md:103](implementation_plan.md#L103) | `wgpu::CommandBuffer` в `gpu_pipeline.rs` | Нет системы "инструкций вместо данных" — данные копируются через SmartBuffer, нет трекинга residency. |
| **Component Packing / SoA** | [key_ideas.md:295-299](key_ideas.md#L295-L299), [implementation_plan.md:83](implementation_plan.md#L83) | `#[derive(Pack)]` генерирует wrapper-типы для каждого поля в `pack.rs` | Не интегрирован в `for_each_entity` / `smart_pipeline` автоматически. |
| **SmartBuffer (Data Residency)** | [key_ideas.md:103](key_ideas.md#L103), [implementation_plan.md:97](implementation_plan.md#L97) | `DirtyCPU`/`DirtyGPU` флаги в макросах | Нет автоматического копирования только при необходимости. |

---

## ❌ Заявлено как готовое (✅ в docs), но НЕ РЕАЛИЗОВАНО в коде

### 1. Рендер-пайплайн материалов — Фаза 5.11
- **Документация**: [implementation_plan.md:152](implementation_plan.md#L152) — "Deferred + Forward hybrid, G-buffer pass + lighting pass + forward pass + composite pass; переключение между рендерами через `RenderBackend` trait"
- **Реальность**: Только `Renderer3D` (forward) + `CompositePass` (Vello UI overlay). **Нет `RenderBackend` trait**, нет deferred/forward разделения, нет lighting pass, нет переключения бэкендов.

### 2. In-Game Editor — Фаза 5.7
- **Документация**: [implementation_plan.md:148](implementation_plan.md#L148) ✅
- **Реальность**: Есть `EditorOverlay` с Vello примитивами (панель, Scene Stats, UIStyle Editor). **Это debug overlay, а не редактор**: нет инспектора сущностей, иерархии сцены, gizmos, редактирования компонентов.

### 3. Remote Editor (Web) — Фаза 5.8
- **Документация**: [implementation_plan.md:149](implementation_plan.md#L149) ✅
- **Реальность**: HTTP сервер на `tiny_http` (порт 3420) + статичная страница. **Нет реального редактирования**: только entity_count, event-лог, создание entity. Нет синхронизации состояния, нет UI для редактирования.

### 4. Static Analyzer / Линтер — Фаза 9
- **Документация**: [key_ideas.md:216-221](key_ideas.md#L216-L221), [implementation_plan.md:208-213](implementation_plan.md#L208-L213) ⚠️
- **Реальность**: `smart_pipeline` делает базовый анализ AST. **НО `compile_warning!` не генерируется**, интеграции с IDE (Rust Analyzer) нет, расширяемости нет.

### 5. Batch API для скриптов — Фаза 6.6
- **Документация**: [key_ideas.md:289-293](key_ideas.md#L289-L293), [implementation_plan.md:167](implementation_plan.md#L167) ❌
- **Реальность**: Нет реализации.

### 6. DSP на GPU / Процедурный звук — Фаза 4.7-4.8
- **Документация**: [implementation_plan.md:130-131](implementation_plan.md#L130-L131) ⏳
- **Реальность**: Нет.

### 7. NUMA-Aware Allocation — Фаза 10.6
- **Документация**: [key_ideas.md:301-305](key_ideas.md#L301-L305), [implementation_plan.md:228](implementation_plan.md#L228) ❌
- **Реальность**: Нет.

### 8. HVM2/Bend бэкенд — Фаза 3.15, 6.7
- **Документация**: [key_ideas.md:336-341](key_ideas.md#L336-L341), [implementation_plan.md:107,168](implementation_plan.md#L107)
- **Реальность**: Только в документации как "future work".

---

## 🔗 Ссылки на исходные MD-файлы

| Файл | Описание |
|---|---|
| [key_ideas.md](key_ideas.md) | Концептуальные идеи (418 строк) — архитектура, паттерны, философия |
| [implementation_plan.md](implementation_plan.md) | План реализации (301 строка) — фазы, задачи, статусы, стек |
| [SUMMARY.md](SUMMARY.md) | Автообновляемая сводка (101 строка) — текущее состояние по фазам |

---

## 📋 Рекомендации

### Приоритет 1: Закрыть критические расхождения
1. **Command-Based Sync** — система "инструкций вместо данных" для CPU↔GPU синхронизации

### Приоритет 2: Довести до mind-соответствия
2. **Сделать настоящий In-Game Editor** (инспектор, иерархия, gizmos)
3. **Добавить редактирование в Remote Editor** (WebSocket + синхронизация состояния)
4. **Реализовать `compile_warning!` в `smart_pipeline`** — обучение разработчика

### Приоритет 3: Актуализировать документацию
- Пометить в `implementation_plan.md` и `SUMMARY.md` задачи как ⏳/❌ там, где код не реализован
- Убрать ✅ у Editor, DSP на GPU, Batch API, Линтер
- Добавить раздел "Known Gaps" или "Architecture Debt"

---

## 📊 Метрика соответствия

```
Реализовано полностью:           12/25 (48%)
Частично / требует доработки:     3/25 (12%)
Заявлено ✅, но отсутствует:      8/25 (32%)
Не начато (фазы 6-11):            2/25  (8%)
```

> **Вывод**: Ядро (Sparse Sets, макросы, OpenPBR, WGSL, UI/DOM, MaterialX, RenderBackend, PhysicsEngine, Runtime CPU/GPU диспетчер) — production-ready. Command-Based Sync, редакторы, линтер, DSP на GPU — требуют реализации. Документация опережает код на ~1-2 фазы.