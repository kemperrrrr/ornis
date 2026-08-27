# Render Graph для гибридного рендеринга — design/implementation note

> **Статус (2026-08-27):** это зафиксированная design/implementation
> note, а не незакоммиченный черновик. Фазы 0–4 реализованы и
> верифицированы; будущие изменения должны синхронизироваться с
> `PLAN.md`, `README.md` и актуальными именами `FramePlan`/`FrameExecutor`.
> Дата исходного дизайна: 2026-08-10.
> Источник: исследование Hermes (дисциплина web-research), первоисточники в разделе «Источники».
>
> **Интеграция 2026-08-27:** native и WASM render loops используют
> `ornis_render::RenderWorld`/`RenderExtracted` для ECS-backed extraction и
> `RenderFrame3D`/`FramePlan` для записи кадра. `RenderBackend::render_scene`
> остаётся compatibility/reference API; server↔browser serialization boundary
> сохраняется.
>
> **Переименование 2026-08-23**: модули/типы — `render_graph.rs` → `frame_plan.rs`
> (`RenderGraph` → `FramePlan`, `GraphLayout` → `FrameLayout`), `graph_frame.rs` →
> `frame_exec.rs` (`GraphExecutor` → `FrameExecutor`, `RenderGraph3D` → `RenderFrame3D`),
> `graph_passes.rs` → `frame_passes.rs` (`GraphPass` → `FramePass`, `GraphResource` →
> `FrameResource`); пример — `frame_plan_probe`. Датированные фаза-логи ниже используют
> имена своего дня; живой рецепт §10 переведён на новые.

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

## 4. Историческое состояние Ornis до FramePlan (снимок 2026-08-10)

> Следующее описание фиксирует проблему, которую закрыли фазы 0–4;
> оно не является текущим состоянием кода.


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

## 9. Статус реализации (2026-08-11, подтверждено ревью)

- **Фаза 0 — каркас** ✅: `render_graph.rs` — lifetime-окна, пул слотов (жадная интервальная раскладка), culling пассов, импортированные и внешние ресурсы; 10 юнит-тестов.
- **Фаза 1 — исполнитель** ✅: `graph_frame.rs` — `GraphExecutor` (пул → реальные `wgpu::Texture`, внешние view для swapchain), `PassViews` (view_of с проверкой живости), `RenderGraph3D` (gbuffer → lighting → forward → composite как узлы). Pass-тела в `renderer.rs` параметризованы `GbufferTargets`; lighting собирает бинд-группу per-frame (gbuffer теперь transient-доступен).
- **Фаза 2 — budget памяти GPU** ✅: `Renderer3D::texture_budget()` (legacy: 8 постоянных текстур) vs `GraphExecutor`/`RenderGraph3D::texture_budget()` (пул: 7 слотов); `format_bytes_per_pixel()` (таблица форматов, 2 юнит-теста). Удалены мёртвые `depth_texture`/`depth_view` из `Renderer3D` (рабочий depth живёт в gbuffer MRT). 1280×720: legacy 36 864 000 B (35,2 MB) → graph 29 491 200 B (28,1 MB), **−20,0%** (ровно один `Rgba16Float`, слот #4).
- **Верификация**: `examples/render_graph_probe.rs` — legacy и graph пути байт-в-байт идентичны (0 отличий из 3 686 400); 9 ресурсов → 7 слотов (алиасинг `material_params` + `hdr_fwd` в слоте #4); 16 кадров — пул стабилен, кадры идентичны. `cargo xtask quality` — PASS (тесты 26/26).
- **Фаза 3 — блум** ✅ (первый новый узел, доказательство расширяемости): каскад `bloom_down0` (bright-pass ½, порог `BLOOM_BRIGHT_THRESHOLD = 0.7`, мягкое колено `smoothstep`), `bloom_down1` (¼), `bloom_down2` (⅛), `bloom_up1` (upsample ⅛→¼, ADD поверх Load), `bloom_up0` (¼→½, ADD поверх Load); up-пассы переиспользуют down-слоты (`SizePolicy::Fraction(u32)`), финальный уровень — в `bloom0`. Composite: новые входы `bloom_tex` (binding 3) + `BloomParams` (binding 4), `mix(combined, bloom*intensity)` перед `aces_tonemap`. Culling: при `bloom=false` bloom-ресурсы в пул не попадают (first_use == usize::MAX), composite биндит заглушку c intensity 0.
- **Верификация блума**: bloom-off — 0 отличий legacy↔graph (16 кадров); bloom-on — **267 103 px изменены** (specular-блики > порога в `scene.ron`); слоты 7 → **10** (7 базовых + 3 bloom-уровня), бюджет 28 107 264 → **31 910 400 B** (+3,8 MB); PNG `target/render_graph_probe_bloom.png`.
- **Фаза 4 — переключатель forward/hybrid/deferred как конфигурация графа** ✅: `Technique { Forward, Deferred, Hybrid }` (`graph_frame.rs`, ре-экспорт из `lib.rs`); `RenderGraph3D::new_with(format, size, technique, bloom)`. Проводка узлов: `gbuffer`/`lighting` при `has_deferred()`, `forward` при `has_forward()`; composite читает только живые слои, мёртвый слой дублируется живым view, шейдер различает по `mode` uniform (0/1/2). Forward-only: пасс сам владеет depth (`write_clear`, LoadOp::Clear(1.0)), блум читает живую HDR-текстуру (`hdr` для deferred/hybrid, `hdr_fwd` для forward-only).
- **Верификация техник** (probe, 1280×720, 16 кадров): hybrid совпадает с legacy попиксельно; **deferred-only == legacy** (0 отличий — классический путь и есть deferred-цепочка); forward-only — **137 164 px** отличий, **2 слота / 10 598 400 B (−62%)**, собственный блум активен; бюджеты hybrid 29 491 200 B ≥ deferred 29 491 200 B ≥ forward 10 598 400 B; юнит-тесты **30/30** (+4: флаги техник, состав пассов, вход блума по слою, владение depth и мёртвые ресурсы при forward-only).
- **Наблюдение**: deferred-only == legacy означает, что forward-слой в классическом пути ничего не добавляет на текущей сцене (все материалы opaque); прозрачность поверх deferred — отдельный будущий узел.

## 10. Типизированные пассы (S2a/S2b/S3, 2026-08-19)

Пасс объявляется системой с доступами в сигнатуре; проводка (reads/
writes/clear), layout и пул выводятся из типов. Условные доступы —
«режимами» (singleton-типы с таблицей фактов), тело у семейства одно.

Как объявить свой пасс:

```rust
use ornis_render::frame_passes::{Bloom0, Bloom1};
use ornis_render::system::{self, Frame, FramePass, Read, SystemViews, Write};

struct BloomMid;

impl FramePass for BloomMid {
    type Reads = (Read<Bloom0>,);
    type Writes = (Write<Bloom1>,);
    fn name(&self) -> &'static str { "bloom_mid" }
    fn run(&mut self, views: SystemViews<'_, Self>, frame: &mut Frame<'_>) {
        let (input,) = views.reads;
        let (output,) = views.writes;
        frame.renderer.render_bloom_down(
            frame.device, frame.queue, frame.encoder, input, output, 0.0,
        );
    }
}

// регистрация (порядок = порядок исполнения):
systems.add_system(&mut plan, BloomMid);
```

Диспетчеризация в `RenderFrame3D::render` — по `PassId`, строковых имён
в исполнителе больше нет. Виды можно получать и по типу ресурса:
`views.get::<Bloom0>()` (debug-проверка членства в объявленном наборе).

`FramePlan::add_pass`/`PassBuilder` — шим совместимости для тестов и
инструментов; production-код объявляет пассы типами.

Защита от тихих изменений пула — golden-тесты (`golden_pool_slots_per_
technique`, `golden_bloom_adds_exactly_three_slots`,
`golden_dead_layers_are_unpooled`, `golden_hybrid_lifetimes`): 9
ресурсов → 7 слотов на deferred/hybrid, блум добавляет ровно свои три
уровня (у bloom0/1/2 разные TextureSpec-ключи), мёртвые слои не
пул-ятся. `planned_pool_bytes(layout)` считает байты пула без GPU —
база для S0-метрик и S4-бюджета.
