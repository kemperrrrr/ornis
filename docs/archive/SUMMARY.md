# Сводка по проекту Ornis

> Автоматически обновляемый файл — актуальное состояние проекта.

## ⚠️ Стратегический поворот (июль 2026)

**Нативный UI-крейт (`crates/ui`) заморожен.** Редактор теперь работает в браузере, сцена выводится в окно браузера (WebGPU/WASM). Мы ждём зрелости `blitz` и `boa_engine`, чтобы вернуться к нативному UI. Подробнее — [`STRATEGY_PIVOT.md`](./STRATEGY_PIVOT.md).

Текущий режим запуска: `cargo run --features editor-only` — только браузерный редактор (HTTP-сервер на порту 3420), без нативного winit-окна.

---

## Общее состояние

- **Всего фаз**: 11 (0–11)
- **Полностью завершено**: Фазы 0, 1
- **В работе**: Фаза 2 (9/10 ✅), Фаза 3 (14/15 ✅), Фаза 4 (6/8 ✅)
- **Заморожена**: Фаза 5 — UI-система (нативный рендеринг)
- **Не начаты**: Фазы 6–11

> Автоматически обновляемый файл — актуальное состояние проекта.

## Общее состояние

- **Всего фаз**: 11 (0–11)
- **Полностью завершено**: Фазы 0, 1
- **В работе**: Фаза 2 (9/10 ✅), Фаза 3 (14/15 ✅), Фаза 4 (6/8 ✅), Фаза 5 (9/12 ✅)
- **Не начаты**: Фазы 6–11

---

## Фаза 0. Инфраструктура и подготовка ✅

| # | Статус |
|---|--------|
| 0.1 | ✅ Workspace Cargo настроен |
| 0.2 | ✅ MSRV зафиксирован |
| 0.3 | ✅ CI настроен |
| 0.4 | ✅ Зависимости зафиксированы |

---

## Фаза 1. Ядро — Sparse Sets + SmartStore ✅

Все 17 задач выполнены. Реализовано:
- Entity, ComponentStore, SmartStore
- Bitset acceleration, paginated sparse arrays
- Entity recycling, cache-line alignment, chunked iteration
- Lock-free SmartStore, prefetch intrinsics
- Temporal coherency sort, hot/cold data splitting
- Physics engine (Sweep-and-Prune + PBD + raycast)

---

## Фаза 2. Процедурные макросы (9/10 ✅, 1 ⏳)

| # | Статус |
|---|--------|
| 2.1–2.5 | ✅ AutoPipeline, smart_pipeline, for_each_entity, проекции, pack |
| 2.6 | ⏳ Анализ зависимостей — отложено |
| 2.7–2.10 | ✅ ZST-маркеры, LaneTarget, PipelineConfig, Component Packing |

---

## Фаза 3. CPU/GPU диспетчер ✅ (14/15)

Реализовано:
- wgpu-интеграция, GPU-буферы, WGSL-генерация
- gpu_pipeline, Smart Buffer, runtime-выбор CPU/GPU
- Автопрофилировщик, DSL-подмножество, статический профайлер
- Pipeline Router, Command-Based Sync, ZST-диспетчеризация
- PSO-кэш, LEAK-паттерн

⏳ Отложено: 3.15 HVM2-бэкенд (нет кода, только документация)

---

## Фаза 4. Аудиосистема ✅ (6/8)

| # | Статус |
|---|--------|
| 4.1–4.5 | ✅ AudioSource/L listener, cpal, symphonia, микшер, spatial audio |
| 4.6 | ✅ Web Audio API бэкенд |
| 4.7–4.8 | ⏳ DSP-эффекты на GPU, процедурный звук (future) |

---

## Фаза 5. UI-система ❄️ ЗАМОРОЖЕНА (стратегический поворот)

> См. [`STRATEGY_PIVOT.md`](./STRATEGY_PIVOT.md). Нативный UI-крейт (`crates/ui`) не удаляется, но активная разработка приостановлена. Редактор теперь — браузерное приложение. Сцена рендерится в браузерный canvas через WebGPU (WASM).

| # | Статус | Примечание |
|---|--------|------------|
| 5.1 | ✅ Парсинг HTML/CSS + flexbox layout | Сохранено в `crates/ui` |
| 5.1a | ❄️ Интеграция Servo | Отложена до зрелости экосистемы |
| 5.2 | ✅ Векторный рендеринг + текст | Сохранено в `crates/ui` |
| 5.3 | ✅ JS-интерпретатор (boa_engine) | Сохранено, но Boa ещё не зрел для production UI |
| 5.4 | ✅ Headless DOM | Сохранено в `crates/ui` |
| 5.5 | ✅ JS ↔ Rust ECS bridge | Сохранено в `crates/ui` |
| 5.6 | ✅ Двухсторонний IPC | Переиспользуется для связи браузер ↔ движок |
| 5.7 | ❄️ In-Game Editor | Заменён на Remote Editor в браузере |
| 5.8 | ✅ **Remote Editor (Web)** | **Стал основным режимом редактора** |
| 5.9 | ✅ Система материалов: OpenPBR Surface | Активно, в `crates/render` |
| 5.10 | ⏳ Поддержка MaterialX | Активно, в `crates/materialx` |
| 5.11 | ⏳ Рендер-пайплайн материалов | Активно, в `crates/render` |

| # | Статус |
|---|--------|
| 5.1 | ✅ Парсинг HTML/CSS + flexbox layout (html5ever + lightningcss + taffy) |
| 5.1a | ⏳ Интеграция Servo (отложено) |
| 5.2 | ✅ Векторный рендеринг + текст (vello + skrifa) |
| 5.3 | ✅ JS-интерпретатор (boa_engine) |
| 5.4 | ✅ Headless DOM (Element, classList, style, console) |
| 5.5 | ✅ JS ↔ Rust ECS bridge (EcsBridge, UIStyle) |
| 5.6 | ✅ Двухсторонний IPC (crossbeam-channel, UiCommand/GameEvent) |
| 5.7 | ✅ In-Game Editor (Vello overlay, F1/~ toggle) |
| 5.8 | ✅ Remote Editor (Web) — HTTP-сервер + веб-страница на порту 3420 |
| 5.9 | ✅ Система материалов: OpenPBR Surface |

Создан крейт `crates/render` с:
- `Material` — ECS-компонент с OpenPBR-параметрами (base_color, emission, metalness, roughness, specular, subsurface, sheen, coat), `bytemuck::Pod` для GPU-загрузки
- `Mesh` + `Vertex` — геометрия (вершины/индексы) + `create_sphere()`
- `Transform` — позиция/вращение/масштаб (`glam`)
- WGSL PBR-шейдеры: GGX microfacet (NDF, Smith-G, Fresnel-Schlick), multiple lights, ambient, tone-mapping
- `Renderer3D` — wgpu-пайплайн с storage buffer на per-object данные (мат. инстансы)
- `CompositePass` — полный экран для наложения Vello UI поверх 3D с alpha-блендингом
- Интеграция в основной цикл: 5 сфер с разными материалами (red dielectric, green rough, blue smooth, gold metallic, white ceramic), одна дир. лампа + fill light

| 5.10 | ⏳ Поддержка MaterialX |
| 5.11 | ⏳ Рендер-пайплайн материалов |

---

## Фазы 6–11 ❌

Не начаты.
