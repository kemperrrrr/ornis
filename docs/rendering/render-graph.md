# Render Graph для гибридного рендеринга — черновик для ревью

> ⚠️ **Статус: ЧЕРНОВИК ДЛЯ РЕВЬЮ.** Не коммитить до ревью. Правки — дописывать в конец.
> Дата: 2026-08-10.
> Источник: исследование Hermes (дисциплина web-research), первоисточники в разделе «Источники».

---

## 1. Проблема: Forward vs Deferred

Долгое время не могли определиться, на какой технике рендерить:

- **Forward** — прост, дёшев на слабом железе, естественно работает с прозрачностью и MSAA, но платит за каждый источник света (пере-шейдинг вершин). На много света — просадки.
- **Deferred** — свет считается на экранном G-buffer (MRT), стоимость света не зависит от сложности сцены, но: сложно прозрачное, сложно MSAA, нужен G-buffer (память/полоса), banding по нормалям.

Вывод исследования: **вопрос «что выбрать» устарел**. Индустрия (Frostbite, Granite, Doom 2016, Spider-Man) давно рендерит гибридно: неопрачные в deferred (G-buffer + lighting), прозрачные/спецматериалы — forward поверх. А инструмент, на котором такой гибрид собирается без боли — **render graph**. Это не третья техника рендеринга, а **слой оркестрации пассов** над любой техникой.

## 2. Что такое render graph

Направленный ацикличный граф (DAG): **узлы** = render-пассы, **рёбра** = зависимости по ресурсам (пасс читает/пишет ресурс X). Модули рендера описывают свои входы/выходы локально, а граф видит весь кадр целиком и на глобальном знании автоматически делает:

| Возможность | Что даёт |
|---|---|
| Топологическая сортировка + реордеринг | Порядок исполнения с максимальным overlap пассов, минимум «жёстких» барьеров |
| **Transient-ресурсы** | Ресурс живёт только между первым и последним использованием; вне этого окна не выделен |
| Memory aliasing | Непересекающиеся по времени текстуры делят одну физическую память (на голом Vulkan/D3D12) |
| Автобарьеры / layout transitions | Снимает главную боль низкоуровневых API (Vulkan/D3D12) |
| Async compute | Независимые подграфы автоматически уходят на compute queue |
| Culling пассов | Неиспользуемые ветви графа не исполняются; пасс может сам решить, нужен ли он в этом кадре |

Два стиля реализации:

- **Retained-граф** — строится один раз, переиспользуется (Bevy RenderGraph). Хорош, когда топология кадра стабильна.
- **Immediate-граф** — пересобирается каждый кадр с нуля; лёгкий, предсказуемый (Frostbite FrameGraph, Granite, Ponies & Light). Гибче для движка со скриптуемым редактором.

## 3. Почему это закрывает спор Forward/Deferred

Render graph не выбирает технику — он делает выбор **конфигурацией графа**, а не переписыванием рендера. Пример гибрида (по сути — Granite):

```cpp
// Deferred-ветка
auto &gbuffer  = graph.add_pass("gbuffer", ALL_GRAPHICS);
gbuffer.add_color_output("emissive", emissive);   // 4 MRT
gbuffer.add_color_output("albedo",   albedo);
gbuffer.add_color_output("normal",   normal);
gbuffer.set_depth_stencil_output("depth", depth);

auto &lighting = graph.add_pass("lighting", ALL_GRAPHICS);
lighting.add_color_output("HDR", emissive, "emissive"); // fullscreen-quad
lighting.add_attachment_input("albedo");
lighting.add_attachment_input("normal");

// Forward-ветка (прозрачные) — пишет в тот же HDR
auto &fwd = graph.add_pass("forward", ALL_GRAPHICS);
fwd.add_color_output("HDR", emissive, "HDR");
fwd.set_depth_stencil_input("depth");
```

- Хочешь чистый forward (мало света/слабое железо) — `gbuffer`-узел просто не добавляется в граф.
- Хочешь SSAO/тени/TAA — узлы вставляются между `gbuffer` и `lighting`.
- Прозрачные уходят от проблем deferred естественным образом (отдельный forward-узел).

## 4. Текущее состояние Ornis: гибрид уже есть, но императивный

В `crates/render/src/renderer.rs` (`Renderer3D`, ~1550 строк) уже живут оба мира, связанные вручную в одном `render()`:

| Пасс | Функция | Выход |
|---|---|---|
| G-buffer | `render_gbuffer` | 5 MRT: albedo, normal, material_id, world_position, material_params + depth |
| Lighting | `render_lighting` | fullscreen-quad, читает G-buffer → HDR |
| Forward | `render_forward` | прозрачные поверх HDR |
| Composite | (в `composite.rs`) | тонирование → swapchain |

Болевые точки текущей схемы:

- G-buffer текстуры **постоянные**, живут всю жизнь `Renderer3D` (не transient).
- Последовательность пассов хардкодом; новый пасс = новые поля структуры + правка `render()`.
- Нет места, куда «вставить» SSAO/тени/TAA между gbuffer и lighting без роста спагетти.

**Вывод:** выбор forward/deferred в Ornis уже фактически сделан в сторону гибрида — не хватает слоя оркестрации, чтобы это стало чистым и расширяемым.

## 5. Оговорка про wgpu: что граф даёт, а что уже делает wgpu

Канонические статьи (Granite, Frostbite) написаны под голый Vulkan/D3D12, где главная ценность графа — автоматические барьеры и layout transitions. **wgpu уже делает это сам** (внутренний usage tracker). Поэтому для Ornis ценность графа смещается:

| Механика | На голом Vulkan/D3D12 | На wgpu |
|---|---|---|
| Барьеры/layout transitions | Главная боль, снимает граф | Уже в wgpu |
| Переиспользование памяти | Memory aliasing на уровне драйвера | Ограничено — но пул текстур-объектов даёт ту же экономию на уровне объектов |
| Время жизни ресурсов | — | **Главный выигрыш**: gbuffer становится transient, пул временных текстур |
| Декларативность/расширяемость | — | **Главный выигрыш**: пассы — данные, а не код |
| Async compute / мульти-queue | Да, через граф | wgpu поддерживает ограниченно; не цель первого этапа |
| WASM (редактор в браузере) | — | Граф — чистый Rust без GPU-специфики; рабочий прецедент: `wgpu-rendergraph` (native + webgpu + webgl) |

## 6. Предлагаемый дизайн: лёгкий immediate-граф

Паттерн Frostbite/Ponies&Light, без тяжёлого bake-конвейера Granite (барьеры, алиасинг, async compute — на wgpu не переносится буквально). Ориентир API:

```rust
// Крейт: crates/render/src/render_graph.rs (или отдельный крейт render_graph)

pub struct RenderGraph {
    passes: Vec<RenderPassNode>,
    resources: ResourceRegistry,
    frames_in_flight: usize,
}

pub struct RenderPassNode {
    name: &'static str,
    inputs: Vec<ResourceHandle>,   // чтение
    outputs: Vec<AttachmentSpec>,  // запись (цвет/глубина)
    execute: Box<dyn Fn(&mut wgpu::CommandEncoder, &PassContext)>,
}

pub struct AttachmentSpec {
    handle: ResourceHandle,
    format: wgpu::TextureFormat,
    size: SizePolicy,          // MatchSurface / Relative / Fixed
    sample_count: u32,
    clear: Option<wgpu::Color>,
}

// Пасс объявляет зависимости и получает «живые» view-ы только на время исполнения:
pub struct PassContext<'a> {
    pub views: HashMap<ResourceHandle, ResourceView<'a>>, // текстуры, созданные/полученные из пула
}
```

Ключевые решения:

1. **ResourceRegistry + пул текстур.** Логический ресурс = спецификация (формат/размер/samples). Реальная `wgpu::Texture` выдаётся из пула (или создаётся) на время жизни пасса. Спецификация совпадает → переиспользование объекта. G-buffer объявляется transient: живёт от `gbuffer` до `lighting`.
2. **Pass culling.** Пасс может вернуть `Skip`, если не нужен в этом кадре (например, рендер теней при нуле теневых источников).
3. **Swapchain = корень графа.** Граф завершается узлом, который пишет в поверхность (`Composite`).
4. **Порядок = порядок добавления** (immediate-граф): топологическая сортировка не нужна, если декларировать пассы в правильном порядке; зависимости проверяются assert-ами (ресурс читается раньше, чем записан — ошибка).

### Этапы внедрения (без переписывания пайплайнов)

1. **Фаза 0 — каркас.** Крейт/модуль `render_graph` с регистририей ресурсов и пулом текстур; пассы-колбэки. Юнит-тесты на lifetime (ресурс не выдан вне окна жизни) и на переиспользование пула.
2. **Фаза 1 — перенос существующих пассов.** Обернуть 4 текущих пасса (`gbuffer`, `lighting`, `forward`, `composite`) в узлы. Поведение должно совпасть попиксельно (сравнение скриншотов через `render_probe`).
3. **Фаза 2 — gbuffer transient.** Перевести текстуры G-buffer в пул; замерить пиковую память GPU до/после через profiler.
4. **Фаза 3 — первый новый узел.** Добавить что-то простое (например, блум или SSAO) — доказательство расширяемости.
5. **Фаза 4 (future)** — переключатель «forward-only / hybrid / deferred» как конфигурация графа (для слабого железа и редактора).

## 7. Открытые вопросы для ревью

- [ ] Нужен ли отдельный крейт `render_graph` или модуль внутри `render`? (пока нет внешних зависимостей — склоняюсь к модулю)
- [ ] Retained vs immediate: для редактора (пересборка при изменении сцены) immediate выглядит правильнее — подтвердить.
- [ ] Насколько глубоко делать пул: только текстуры одного размера или full-хеш спецификации?
- [ ] Стоит ли вводить понятие «суб-пасс» для будущих input-attachments (на wgpu это ограничено) — или отложить.
- [ ] Нужен ли JSON/Viz-дамп графа для дебага (как отладочный инструмент в духе существующего JSON-diff подхода)?

## 8. Источники

- [FrameGraph: Extensible Rendering Architecture in Frostbite — Yuriy O'Donnell, GDC 2017](https://gdcvault.com/play/1024612/FrameGraph-Extensible-Rendering-Architecture-in) — первоисточник техники (D3D12).
- [Render graphs and Vulkan — a deep dive — Hans-Kristian Arntzen (Granite), 2017](https://themaister.net/blog/2017/08/15/render-graphs-and-vulkan-a-deep-dive/) — самый полный разбор: bake, реордеринг, transient, алиасинг, async compute.
- [Rendergraphs and how to implement one — Ponies & Light, 2022](https://poniesandlight.co.uk/reflect/island_rendergraph_1/) — лёгкая immediate-реализация, bitset-слияние подграфов.
- [rafx Render Graph (Rust)](https://aclysma.github.io/rafx/docs/framework/render_graph.html) — API рендер-графа на Rust с примерами.
- [Bevy render_graph (Rust/wgpu)](https://docs.rs/bevy_render/latest/bevy_render/render_graph/) + [обзор архитектуры (DeepWiki)](https://deepwiki.com/bevyengine/bevy/5.2-rendergraph-and-execution) — retained-граф в продакшне.
- [wgpu-rendergraph — matthewjberger](https://github.com/matthewjberger/wgpu-rendergraph) — рабочий рендерграф на wgpu: native + webgpu + webgl (wasm) — прецедент для редактора.
- [Clustered Forward vs Deferred — Yosoygames, 2016](https://www.yosoygames.com.ar/wp/2016/11/clustered-forward-vs-deferred-shading/) — гибрид depth-prepass + малый G-buffer (Doom 2016).
- [Writing an efficient Vulkan renderer — zeux.io, 2020](https://zeux.io/2020/02/27/writing-an-efficient-vulkan-renderer/) — обзорная рекомендация по графам.

---

## 9. Статус реализации (2026-08-10, подтверждено ревью)

- **Фаза 0 — каркас** ✅: `render_graph.rs` — lifetime-окна, пул слотов (жадная интервальная раскладка), culling пассов, импортированные и внешние ресурсы; 10 юнит-тестов.
- **Фаза 1 — исполнитель** ✅: `graph_frame.rs` — `GraphExecutor` (пул → реальные `wgpu::Texture`, внешние view для swapchain), `PassViews` (view_of с проверкой живости), `RenderGraph3D` (gbuffer → lighting → forward → composite как узлы). Pass-тела в `renderer.rs` параметризованы `GbufferTargets`; lighting собирает бинд-группу per-frame (gbuffer теперь transient-доступен).
- **Верификация**: `examples/render_graph_probe.rs` — legacy и graph пути байт-в-байт идентичны (1280×720, 0 отличий из 3 686 400 байт); 9 ресурсов → 7 слотов (алиасинг `material_params` + `hdr_fwd` в слоте #4). `cargo xtask quality` — PASS.
- Осталось: **Фаза 2** — замер пиковой памяти GPU до/после (профиль уже даёт число слотов), **Фаза 3** — первый новый узел (SSAO/блум), **Фаза 4** — переключатель forward/hybrid/deferred как конфигурация графа.