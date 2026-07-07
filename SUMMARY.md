# Сводка по проекту Ornis

> Автоматически обновляемый файл — актуальное состояние проекта.

## Общее состояние

- **Всего фаз**: 11 (0–11)
- **Полностью завершено**: Фазы 0, 1, 3
- **В работе**: Фаза 2 (9/10 задач), Фаза 4 (5/8 задач)
- **Не начаты**: Фазы 5–11

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

## Фаза 4. Аудиосистема ✅ (5/8)

| # | Статус |
|---|--------|
| 4.1–4.5 | ✅ AudioSource, cpal, symphonia, микшер, spatial audio |
| 4.6 | ❌ Web Audio API бэкенд |
| 4.7–4.8 | ⏳ DSP-эффекты на GPU, процедурный звук (future) |

---

## Фазы 5–11 ❌

Не начаты. Приоритет: Фаза 5 (UI + материалы) при появлении ресурсов.
