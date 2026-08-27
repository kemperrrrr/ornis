# Unified Scheduler — S0–S6 (кеш FrameLayout и общая механика)

> Рабочий документ этапов S0–S6 из **Приложения C** [`PLAN.md`](../../PLAN.md)
> (идея — [`IDEAS.md`](../../IDEAS.md) §28). Каждый прогон бенчей/тестов
> обновляет числа в этом файле.

> **Переименование 2026-08-23** (срез 1 приближения к Фазе C): модули и
> типы получили Фаза-C-долговечные имена — `render_graph.rs` →
> `frame_plan.rs` (`RenderGraph` → `FramePlan`, `GraphLayout` →
> `FrameLayout`), `graph_frame.rs` → `frame_exec.rs` (`GraphExecutor` →
> `FrameExecutor`, `RenderGraph3D` → `RenderFrame3D`, `GraphIds` →
> `FrameIds`), `graph_passes.rs` → `frame_passes.rs` (`GraphPass` →
> `FramePass`, `GraphResource` → `FrameResource`, `ResourceKind::GraphOwned`
> → `FrameOwned`), пример → `frame_plan_probe`. Датированные секции
> хронологии ниже (S0–S5, Hardening 2026-08-21…) используют имена своего
> дня; канон имён — карта и глоссарий под этим указателем.

> Датированные исторические секции ниже могут использовать прежние
> имена `RenderGraph`/`GraphLayout`/`GraphExecutor`; актуальный код
> называется `FramePlan`/`FrameLayout`/`FrameExecutor`.

> **Срез 1b (2026-08-23)**: отладочная mermaid-проекция обобщена в
> движок — `ornis-schedule::MermaidDiagram` (уровни подграфами, потоки
> рёбрами; доменно-нейтральные строковые id/метки собирает фронтенд).
> Поверх него два адаптера: `FrameLayout::mermaid()` (байтовый формат
> прежний, пинится golden-тестом `mermaid_is_a_valid_projection`) и
> новая `Schedule::mermaid()` (системы `S{i}` подграфами уровней, рёбра
> `order_before` — стрелками). У оболочки остаётся доменный
> `debug_dump()` (пул/слоты/спеки).

## Карта планировщика (2026-08-23, бэклог #19)

> Этот блок — точка входа; ниже по файлу — хронология этапов (S0→S6).

**Движок один** — крейт `crates/schedule` (`ornis-schedule`): уровни
(`compute_levels`, `bitset_level_plan<K>` с generic-ключом),
единый `OrderError` + валидация рёбер (`resolve_named_edge`,
`validate_indexed_edge`), кеш уровенного плана (`PlanCache`), исполнитель
уровней (`run_levels` — rayon на нативе, последовательный на wasm),
mermaid-проектор отладочных диаграмм (`MermaidDiagram`, срез 1b).
Доменов в нём нет: ни wgpu, ни ECS.

**Фронтенда два** (осознанно, решение S6 ниже — полный роспуск
`FramePlan` (тогдашний `RenderGraph`) отклонён с причинами):

- `crates/core/src/schedule.rs` (`Schedule`) — системы над `Resources`:
  ключи — `TypeId` singleton-ресурса и `TypeId` ленты `SmartStore`
  (`reads_lane`/`writes_lane`, раздельные пространства имён);
  TLS-enforcement объявленных доступов; кеш — `PlanCache`
  (ленивая инвалидация); отладочная проекция — `Schedule::mermaid()`
  (срез 1b).
- `crates/render/src/frame_plan.rs` (`FramePlan`) — пассы над пулом
  текстур: ключи — `ResourceId`; пул/лайфтаймы/бюджет S4/layout-кеш S1 —
  доменные данные, уровни и рёбра — движок; исполнение записи команд —
  `run_levels` на обоих таргетах (на wasm движковый путь последователен,
  0..nodes; выключенные пассы отсутствуют в `FrameLayout`, поэтому
  порядок регистрации корректен и совпадает с нативным); debug-enforcement
  объявленных доступов — на выдаче view (`PassViews::view_of` →
  `assert_pass_access_declared`, бэклог #6): «sneaky pass» паникует в
  debug, release-путь без стоимости; debug-проекция
  `FrameLayout::mermaid()` — адаптер общего проектора среза 1b
  (формат байт-в-байт прежний, пинится golden-тестом).

**Куда класть новое** (антидрейф): семантика, общая для обоих
фронтендов — только в движок; доменные данные (текстурный пул, ленты,
physics-агрегаты) — только во фронтенд. Исполняемая страховка от
дрейфа — паритет-тест `crates/render/tests/scheduler_parity.rs`:
зеркальные топологии обязаны давать побитово одинаковые уровни.

**Глоссарий двойных имён** (одно понятие — одно слово в каждом
фронтенде; внутри фронтенда всегда его слово):

| Понятие | `Schedule` (core) | `FramePlan` (render) | Движок |
|---|---|---|---|
| Узел плана | система | пасс | node |
| Доступы узла | `SystemAccess` (reads/writes + ленты) | типизированные `Access`-наборы (ZST-маркеры), проекция в layout | срезы `reads`/`writes` ключей `K` |
| Ключ доступа | `TypeId` (ресурс / лента, раздельные пространства) | `ResourceId` | `K: Copy + Eq + Hash` |
| Явные рёбра | `order_before(name, name)` | `order_before(PassId, PassId)` / `_named` | `resolve_named_edge` / `validate_indexed_edge` |
| Уровни | `Schedule::levels()` | `FrameLayout::levels()` | `compute_levels` / `bitset_level_plan` |
| Явные рёбра (данные) | `Schedule::ordering` | `FramePlan::ordering` | матрица смежности в плане |
| Кеш плана | `PlanCache` + `level_computations()` | S1-кеш layout (уровни вычисляются при build) + `layout_computations()` | `PlanCache` (политика на фронтенде) |
| Исполнитель | `run_levels` (оба таргета) | `run_levels` (оба таргета; wasm — последовательный, без `Sync`-границы) | `run_levels` (cfg-пара сигнатур) |
| Ошибка рёбер | `OrderError` (реэкспорт) | `OrderError` (реэкспорт) | `OrderError` |

## Что сделано (2026-08-18)

### S0 — базлайн-метрики

- Бенч `crates/render/benches/layout_bench.rs` (`cargo bench -p ornis-render`):
  - `layout/compute/*` — стоимость одного `compute_layout` на трёх
    производственных графах (Forward 7 пассов / Deferred 8 / Hybrid 9,
    блум включён, 1920×1080);
  - `layout/cache_hit/*` — стоимость `layout()` при готовом кеше
    (steady-state кадр после S1).

### S1 — кеш `GraphLayout`

- `RenderGraph` хранит `cached: Option<GraphLayout>` (`None` = dirty);
  любая мутация (`set_surface_size`, `create/import/external`-ресурсы,
  `add_pass`, `set_pass_enabled`, `PassBuilder::{read,write,write_clear}`)
  сбрасывает кеш.
- Новый горячий метод `RenderGraph::layout() -> &GraphLayout` — пересчёт
  только при dirty; `build()` стал owned-снимком кеша (клонирует — для
  тестов/диагностики), `invalidate()` — явная инвалидация (бенчи),
  `layout_computations()` — счётчик пересчётов (диагностика кеша).
- `RenderGraph3D::render` и `layout_dump` переведены на `layout()`:
  `compute_layout` больше не выполняется каждый кадр (в steady state —
  ровно один раз на конфигурацию).
- `RenderGraph3D::{graph, graph_mut}` — доступ к графу для бенчей/проб.

Кеш не хеширует ключ: мутаций графа в steady state нет, поэтому
инвалидация — честный dirty-флаг на каждом мутаторе (дешевле и проще
поддерживать, чем сигнатурный хеш из §28-шага 1).

## Числа

### `compute_layout` (S0 baseline)

| Граф | Пассов | Ресурсов | Время/вызов | Примечание |
|---|---|---|---|---|
| Forward+блум | 7 | 12 | **5.89 µs** | Apple M1, release, один criterion-прогон |
| Deferred+блум | 8 | 12 | **5.33 µs** | Apple M1, release, один criterion-прогон |
| Hybrid+блум | 9 | 12 | **12.1 µs** | Apple M1, release, один criterion-прогон |

### Кеш-попадание (S1)

| Граф | Время `layout()` (cache hit) | Выигрыш vs compute | Примечание |
|---|---|---|---|
| Forward/Deferred/Hybrid (диапазон замера) | **4.4–4.9 ns** | примерно **1.1–2.7 тыс. раз** | В baseline сохранён общий диапазон `layout/cache_hit/*`, без разбивки по техникам |

> Числа сняты 2026-08-27 на Apple M1 (release, criterion; один прогон) и
> вынесены в [`docs/quality/perf-baseline-2026-08-27.md`](../quality/perf-baseline-2026-08-27.md).
> Среда, в которой писался S0/S1 (2026-08-18), действительно не имела Rust
> toolchain — это историческое объяснение прежних плейсхолдеров. Полную
> матрицу texture budget и отдельные GPU probe-диффы по всем конфигурациям
> нужно снять отдельным ручным прогоном.

### Пул текстур по техникам (`texture_budget`, lavapipe/Metal)

Частично зафиксировано в [`render-graph.md`](render-graph.md): без блума
9 ресурсов → 7 слотов и −20,0% на 1280×720; с блумом добавляются 3 слота
(+3,8 MB). Полная матрица техник × bloom × разрешение пока не архивирована.

## Верификация

> Первый CI-прогон (PR #4): fmt/clippy/компиляция чисто; гейт bca нашёл
> 5 новых нарушений лимитов сложности — устранены (модуль `graph_passes.rs`,
> `run_conditional_pass`, разбор `imperative_wiring`, baseline-запись
> `RenderGraph: nom=31`). Прогон №2 застрял на `apt-get install` раннера
> (инфраструктура GitHub, ~37 мин) — перезапущен новым коммитом.

- `cargo test -p ornis-render`:
  - `layout_is_cached_until_mutation` — повторный доступ без мутаций = 1
    пересчёт;
  - `every_mutation_invalidates_cache` — resize / add_pass+read / toggle
    пасса / create_resource / import_resource / invalidate — каждый
    мутатор инвалидирует;
  - `build_snapshot_matches_cached_layout` — `build()` == кеш;
  - `layout_cache_reused_across_frames` (уровень `RenderGraph3D`) — два
    «кадра» = 1 пересчёт, resize = второй пересчёт, дальше снова кеш.
- `cargo run --example render_graph_probe` — probe-диффы (все техники,
  блум on/off) обязаны остаться пиксельно идентичными: S1 меняет только
  частоту вычисления layout, не его результат.
- `cargo xtask quality` — весь гейт.

## Что дальше (из Приложения C)

- S2 — пасс как типизированная система (`Reads`/`Writes` в типах),
  роспуск `match` по именам пассов в `RenderGraph3D::render`.

## S2a — пасс как типизированная система (2026-08-18, код написан)

Что сделано:

- **Новый модуль `crates/render/src/system.rs`** — инфраструктура
  типизированных доступов (без строк и syn, анти-цель Приложения C):
  - `GraphResource` — идентичность ресурса = тип (`NAME`, `kind()`,
    `spec(surface_format)`), `ResourceKind` {GraphOwned, Imported,
    ExternalOutput};
  - `Read<R>` / `Write<R>` / `WriteClear<R, C>` — ZST-маркеры доступов
    (`ClearBlack/White/Transparent` — значения очистки как ассоциированные
    константы);
  - `AccessSet` / `ViewsFor` на кортежах арности 1..=6 (macro_rules;
    `Views` = кортеж `&TextureView` той же арности);
  - `GraphPass { type Reads; type Writes; name(); run(SystemViews, &mut Frame) }`
    — сигнатура пасса: доступы в типах, тело получает типизированные виды;
  - `SystemSet` — реестр `TypeId → ResourceId` + стёртые раннеры;
    `add_system` выводит проводку пасса из `Reads`/`Writes`
    (read/write/write_clear — автоматически);
  - `Frame` — контекст кадра (device/queue/encoder/renderer/mesh/instances).
- **Мигрированы 6 из 10 пассов** (статичные множества доступа):
  `GbufferPass`, `LightingPass`, `BloomDown1/2Pass`, `BloomUp1/0Pass` —
  объявлены в `graph_passes.rs` как `impl GraphPass`; их проводка в
  `new_with` теперь `systems.add_system(...)`.
- **Диспетчеризация**: `RenderGraph3D::render` сначала пробует typed-system
  по `PassId`, затем fallback на `match` по имени. Ветка `match` сжалась с
  10 до 3 (`forward`, `bloom_down0`, `composite`).
- **12 типизированных ресурсов** (`Albedo`, `Normal`, …, `Bloom2`) —
  спецификации и имена 1:1 со старой проводкой (паритет `ResourceId`).

Тесты (+6): в `system.rs` — порядок сбора доступов, перенос clear-цвета,
**паритет типизированной и builder-проводки** (равенство layout-дампов),
external-output не пулится, паника на незарегистрированный ресурс; в
`graph_frame.rs` — `typed_wiring_matches_imperative_reference`: все
3 техники × {блум on/off} против дословной копии старой проводки.

### S2b — дизайн «вариантные режимы + типизированный fetch» (2026-08-19)

У `forward`/`bloom_down0`/`composite` доступы зависят от конфигурации.
Два наивных пути отвергнуты: (a) вариантные типы — 9 структур с
дублирующимися телами; (b) instance-фильтр — union доступов врёт
планировщику: продлевает lifetime мёртвых слоёв (`hdr` в forward-only),
ломает тесты пулуализации (`technique_forward_owns_depth…`), возвращает
рантайм-условия в layout.

Синтез — разделить «факты о конфигурации» и «исполнение»:

1. **Режим = singleton-тип с таблицей фактов** (только данные, без логики):

   ```rust
   pub trait CompositeMode {
       type Reads: AccessSet + for<'a> ViewsFor<'a>;
       const SHADER_MODE: u32;
       const BLOOM: bool;
       fn inputs<'a>(views: SystemViews<'a, Composite<Self>>) -> CompositeInputs<'a>;
   }
   pub struct HybridBloom;
   impl CompositeMode for HybridBloom {
       type Reads = (Read<Hdr>, Read<HdrFwd>, Read<Bloom0>);
       const SHADER_MODE: u32 = 2;
       const BLOOM: bool = true;
       fn inputs<'a>(v: SystemViews<'a, Composite<Self>>) -> CompositeInputs<'a> {
           let (hdr, hdr_fwd, bloom) = v.reads;
           CompositeInputs { hdr, hdr_fwd, bloom, bloom_intensity: 1.0, /* … */ }
       }
   }
   // ещё 5 строк-фактов: DeferredBloom, Deferred, Hybrid, ForwardBloom, Forward

   pub struct Composite<M: CompositeMode>(PhantomData<fn() -> M>);
   impl<M: CompositeMode> GraphPass for Composite<M> {
       type Reads = M::Reads;
       type Writes = (Write<Target>,);
       fn name(&self) -> &'static str { "composite" }
       fn run(&mut self, views: SystemViews<'_, Self>, frame: &mut Frame<'_>) {
           // ОДНО тело на все конфигурации
           frame.renderer.render_composite(/* … */, M::inputs(views));
       }
   }
   ```

   Трюк «мёртвых слоёв» (deferred-only дублирует вид `hdr` в слот
   `hdr_fwd`) живёт в `inputs()` своего режима — локально и явно.
   Аналогично: `Forward<OwnsDepth|SharedDepth>` (2 режима),
   `BloomBright<DeferredInput|ForwardInput>` (2 режима).

2. **Типизированный fetch** (опциональный слой): `views.read::<R>()` /
   `views.write::<R>()` — доступ по типу ресурса вместо позиционного
   кортежа. Тела и `inputs()` перестают быть хрупкими к форме кортежа.
   Ограничение: compile-time проверка членства (`Read<R> ∈ Reads`)
   упирается в когерентность Rust (перекрытие blanket-имплов,
   специализация нестабильна) — поэтому первый шаг: резолв через
   `Resolver` + `debug_assert` членства (линейный поиск по ≤8
   элементам). API стабилен: внутренности позже ужесточаются до
   compile-time без смены кол-сайтов.

Свойства: планировщик получает точные статические множества (как (a));
тело одно на семейство (как (b)); использование ресурса, не
объявленного в наборе, невозможно; новая опция (SSAO on/off) — строка
в таблице режимов, не новое тело. Комбинаторика не исчезает (6
комбинаций техники×блума — математика), но переезжает из кода в таблицу
конфигурации. Регистрация: `new_with` выбирает режим в существующих
if-ветках (`add_system(Composite::<HybridBloom>)`), после чего
`run_conditional_pass` и ветка `unreachable!` удаляются — исполнитель
становится свободным от строковых имён (grep-гейт S2 enforceable),
открывая S3 (builder → deprecated-шим).

### Эволюция дизайна: скриптинг и масштаб (2026-08-19)

Оба известных предела решаются одним ходом — **поднять рантайм-проекцию
(`AccessDesc`) до первого сорта** и развести слои: фронтенд объявления
(типы / данные / рецепты) → валидация (компилятор или проверка при
регистрации) → планировщик (потребляет только проекцию).

**Скриптинг (фаза 6) — паттерн «манифест + граница»**: `DynamicPass`
объявляет доступы данными до первого исполнения (`reads`/`writes` +
`validate(&registry)` при регистрации: существование ресурсов,
read-before-write, конфликты писателей, совпадение с манифестом
скрипта); исполнение — через границу: `get(resource)` вне манифеста —
жёсткая ошибка (модель permissions браузера/Android). Множества
заморожены при регистрации — layout/пул/S5 не отличают скриптовый пасс
от типизированного. Compile-time-правды для скриптов не будет никогда
(определение скриптинга); манифест — индустриальный максимум.

**Масштаб — разделить именование и поведение**: рост таблицы режимов
лечится ролями/алиасами — пасс пишется против роли («HDR-слои»), граф
при построении разрешает роль в конкретное множество (таблица линейна
по измерениям, не мультипликативна по пассам). Composite вырождается в
один пасс-рецепт (набор собирается из ролей при регистрации), у
`bloom_down0` остаётся алиас входа; настоящие режимы-поведения — только
`OwnsDepth`/`SharedDepth` у forward. Предохранитель — бюджетный гейт
S4: невлезающие комбинации ловятся при сборке независимо от числа
режимов.

**Поэтапность (YAGNI)**: S2b — как спроектировано (6 строк-фактов);
data-фронтенд — с первым скриптовым пассом (фаза 6); роли — когда
появится третье измерение конфигурации (~SSAO) и таблица приблизится к
10 режимам. Стыки предсказаны, а не обнаружены постфактум.

## S2b — реализовано и верифицировано CI (2026-08-19, зелёный прогон 32270386050)

По дизайну выше: `Forward<OwnsDepth|SharedDepth>`,
`BloomBright<FromDeferred|FromForward>`, `Composite<M>` с 6 режимами
(`CompositeDeferredBloom/Deferred/HybridBloom/Hybrid/ForwardBloom/
Forward`). Отступление от скетча: fetch — единый
`SystemViews::get::<R>()` (debug-assert членства по обоим множествам),
а не пара `read`/`write` — у forward `depth` читается в одном режиме и
пишется в другом при одном теле; на уровне wgpu-видов различия нет.
`run_conditional_pass` и `unreachable!`-ветка по именам удалены:
`RenderGraph3D::render` диспетчеризует только по `PassId`. Паритет —
тест против дословной старой проводки (все 6 конфигураций).

## Верификация — ✅ пройдена (CI, PR #4, 2026-08-19)

Полный гейт `cargo xtask quality --ci` зелёный на Linux/lavapipe:
fmt, clippy `-D warnings`, bca, **все тесты** (включая 11 новых:
кеш-инвалидация, паритет проводки ×6 конфигураций, типизированные
системы), audit, deny, outdated, rustdoc, wasm32 check.

По ходу верификации найдены и исправлены (видно по цепочке прогонов):
HRTB-грани `for<'a> ViewsFor<'a>` в трейте `GraphPass` + `where Self: Sized`
на `run` (by-value `SystemViews`); ряд fmt-расхождений (fn_call_width=60,
порядок use, склейка 100-колоночных строк); clippy type_complexity.
CI-телеметрия xtask (аннотации со strip_ansi) — попутный вклад: гейт
теперь сам печатает таблицу стадий и ошибки как аннотации GitHub.

Осталось вне CI: пиксельные probe-диффы (`frame_plan_probe`/
`render_probe` — нужен GPU-адаптер) и полная матрица texture budget.
Основные benchmark-числа S0/S1 уже зафиксированы выше и в
`docs/quality/perf-baseline-2026-08-27.md`.

```
cargo test -p ornis-render          # +6 тестов к существующим
cargo run -p ornis-render --example render_graph_probe   # пиксельные диффы = 0
cargo xtask quality
```

Пиксельный паритет ожидается: проводка идентична (паритетный тест),
исполнение тел пассов не изменилось (те же вызовы `Renderer3D`).

## S3 — layout из типов; builder в шим (2026-08-19)

Типы — единственный источник правды о доступах production-пассов (S2a+
S2b закрыли все десять). `add_pass`/`PassBuilder` задокументированы как
шим совместимости (rustdoc), тесты/инструменты продолжают их
использовать. Golden-тесты пин-ят пул против тихих изменений: слоты
2/5/7/7/10/10 по (техника × блум), блум = ровно +3 слота, мёртвые слои
без слотов, окна lifetime depth/hdr/hdr_fwd/bloom0. Добавлен
`planned_pool_bytes(&GraphLayout)` — байты пула без устройства (S0/
S4-базис). «Конфликт писателей» сознательно не реализован: порядок
пассов = порядок регистрации, write-write между пассами легален и
секвенирован.

## S4 — бюджет памяти как first-class (✅ CI, 2026-08-19, прогон 32337861190)

- `Budget { gpu_textures: Option<u64> }` (`unbounded()` — поведение S3);
  `RenderGraph::set_budget` / `RenderGraph3D::set_budget`, инвалидация
  кеша при смене бюджета.
- `try_layout() -> Result<&GraphLayout, BudgetExceeded>` —
  невыполнимый бюджет = **возвращаемая ошибка**, не паника; `layout()`
  паникует с тем же сообщением (горячий путь). Нарушивший layout не
  кешируется.
- `BudgetExceeded` — actionable: `required`/`budget` + топ-3 слота по
  байтам (имена ресурсов, формат, размер) + подсказка («уменьши размеры
  или отключи пассы, напр. блум»).
- `planned_pool_bytes()` переехал в `GraphLayout` (метод);
  `format_bytes_per_pixel` — в `render_graph` (общая точка).
- **Оптимальность пула доказана структурно**: greedy first-fit по
  `first_use` на интервальном графе даёт ω (максимум одновременных
  перекрытий) — это и есть минимальное число слотов (= «interval
  partitioning» из §28.3); отдельный «режим минимизации пика» не нужен.
- Гейт: proptest `budget_holds_or_refuses` (техники × блум × culling
  хвоста × размеры: точный бюджет всегда ок; на байт меньше — всегда
  отказ с `required == planned`; `unbounded` возвращает S3) + юнит
  `budget_exceeded_is_actionable`.

## S5 — один scheduler: фундамент (S5a, 2026-08-19)

- **`ornis-core::schedule`** — системный шедулер «один на всё»
  (§28.2): `Resources` — контейнер singleton-ресурсов по типам
  («мир»; Device/Queue/конфиги — такие же жители, как ленты данных);
  `System { name, access(), run(&Resources) }` — доступы данными
  (чтения/записи по типам); `Schedule` — выводит конфликты
  (RaW/WaR/WaW; тайбрейк — порядок регистрации), раскладывает системы
  по уровням параллельности, исполняет уровни последовательно, системы
  внутри уровня — параллельно (rayon); `set_parallel(false)` — строго
  последовательный bit-identical режим.
- **Детерминизм**: уровни детерминированы порядком регистрации; внутри
  уровня системы бесконфликтны (записи дизъюнктны с чужими доступами) —
  порядок внутри уровня не влияет на результат. Требование к системам:
  мутация ресурсов через внутреннюю изменяемость (Mutex/атомики),
  никакого доступа вне объявленных множеств. Тест
  `parallel_matches_sequential` — параллельный прогон ==omu
  последовательному (суммы + мультимножество событий).
- **`GraphLayout::levels()`** — те же уровни для пассов рендера, из
  declared accesses layout'а. Независимые ветви делят уровень
  (`independent_branches_share_levels`). Текущий конвейер — НЕ строгая
  цепочка (тест-ожидание «цепочка» опровергнуто первым же прогоном):
  уровни hybrid+bloom = `gbuffer → {lighting ∥ forward} → цепочка
  блума → composite` — deferred-слои и forward-путь не делят ресурсов,
  это первый реальный параллелизм конвейера (`production_graph_levels`).
  Параллельная запись команд (encoder на пасс) — S5b.
- Остатки S5: миграция исполнения рендера на `Schedule`
  (encoder-per-pass), `.before()/.after()` поверх выведенных
  зависимостей, bench против S1-пути.

## S5c — явный порядок поверх выведенных зависимостей (✅ CI, 2026-08-19, ffef5e9)

- `RenderGraph::order_before(PassId, PassId)` / `order_before_named(name,
  name)`: объявляет скрытую зависимость (например, общий queue-записанный
  uniform-буфер — invariant S5b), невидимую множествам доступа. Рёбра
  хранятся на графе, снимок попадает в `GraphLayout::ordering`,
  `levels()` разводит пассы по разным уровням. Порядок ИСПОЛНЕНИЯ
  остаётся порядком регистрации (решение S3): ребро назад — паника с
  подсказкой. Рёбра через выключенный пасс игнорируются.
- `ornis-core::Schedule::order_before(name, name)` / `order_after` —
  тот же контракт для систем: рёбра только разбивают уровни параллельности.
- Гейты: ребро разводит общий уровень (граф и ядро), обратное
  направление и неизвестное имя паникуют, golden-тесты S3 (без рёбер —
  layout не меняется) остаются зелёными. Bench `layout/levels` добавлен
  в `layout_bench` (criterion, ручной прогон — как числа S0).
- Осознанное ограничение: топологическая пересортировка исполнения
  отложена (регистрация = порядок); потребность появится с динамическими
  пассами (data-фронтенд фазы 6).

## S5d — гранулярность лент SmartStore в плане систем (2026-08-23, бэклог #5 аудита)

- `SystemAccess::reads_lane/writes_lane::<T>()` — декларации доступа к
  горячим лентам `SmartStore` по `TypeId` компонента; `*_lane_id(TypeId)`
  — варианты под динамические фронтенды через реестр F0
  (`ComponentMeta::type_id`). Протокол решает негатив §3.6
  (`docs/quality/audit-2026-08-22.md`): `SmartStore` как один
  singleton-ресурс сериализовал бы все системы, а без деклараций
  зависимости по компонентам были невидимы планировщику.
- Ключевое пространство планировщика раздельно: `TypeId` ресурса и
  `TypeId` ленты — разные ключи (`AccessKey` в `schedule.rs`), ложного
  конфликта «один тип как ресурс и как компонент» нет. Битсет-план
  `ornis-schedule` generic и не менялся.
- Каноничная форма системы над лентами: `.reads::<SmartStore>()` (сам
  store — общий singleton, read-read не конфликтует) + ленты по
  компонентам. Дизъюнктные ленты — параллельные системы без ручных
  `order_before` (критерий Фазы B аудита; тест
  `two_lane_system_plans_without_manual_edges`).
- Enforcement на границе `SmartStore::read_lane/write_lane` (TLS-стек
  активных деклараций): чтение покрывается `reads_lane` или
  `writes_lane`, запись — строго `writes_lane` (write-гард доказывает
  намерение, в отличие от незримой мутации через `&Resources`). Cold- и
  lock-free ленты — отдельные пространства имён, протоколом не покрыты.
  Тесты: `crates/core/tests/schedule_lanes.rs` — RaW/WaR/WaW по лентам,
  разделение пространств, id-эквивалентность типизированным билдерам,
  LCG-дифференциальный план против наивной модели, паники enforcement.

## S5 — закрытие (2026-08-19, без GPU-машины)

- **Единый движок уровней**: `ornis-core::compute_levels(n, ordered(i,j))`
  — одна функция считает уровни параллельности и для `Schedule` (системы,
  ключи TypeId), и для `GraphLayout::levels` (пассы, ключи ResourceId).
  Дублирование S5a/S5c устранено: «один scheduler на всё» теперь буква
  кода, а не только паттерн. Два фронтенда остаются осознанно:
  texture-ключевой (пул видов графа) и generic-ресурсный (`Resources`).
  > 2026-08-23 (Фаза A аудита): движок вынесен в крейт `ornis-schedule`
  > (`compute_levels`, битсет-план, единый `OrderError`, `PlanCache`,
  > исполнитель `run_levels`); `ornis_core::compute_levels` остаётся
  > реэкспортом, контракт ниже действует для нового пути.
- **Bench записи**: `recording_bench` (lavapipe/headless) — sequential vs
  parallel `render()` + одинаковый submit; меряет CPU-сторону записи,
  которую и оптимизирует S5b. Compile-checked в гейте; числа — ручной
  прогон (`cargo bench -p ornis-render --bench recording_bench`) на любой
  машине. На текущем графе (2 пасса в параллельном уровне) ожидаемо
  в пределах шума — выигрыш проявится с тяжёлыми независимыми пассами.
- **Регресс-гейт без bench**: параллельная запись opt-in (по умолчанию
  выключена) → дефолтный путь бит-идентичен прежнему по построению,
  пиксельный паритет на lavapipe пиняет оба пути.
- Extract-free «один мир» (`Res<Device>` и т.п.) — следующий шаг за S5:
  паттерн `Resources`-одиночек показан в тестах ядра; вживление в
  главный цикл — вместе с живым редактором (приоритет «a»), где
  появляется настоящий второй потребитель кадровых систем.

## Контракт шедулера (2026-08-19, фиксация после S5)

Единство планировщика — это **модель и движок, а не одна структура
данных**: доступы-декларации → выведенные зависимости → уровни → парал-
лельное исполнение. Любой фронтенд, желающий исполняться «под единым
scheduler'ом», соблюдает контракт:

1. **Доступы — данные**: множества чтений/записей декларируются как
   `AccessDesc`-образные данные (типовой фронтенд: типы → `AccessDesc`;
   скриптовый фазы 6: манифест → `AccessDesc`), а не похоронены в коде
   систем.
2. **Уровни — только через общий движок** `ornis-schedule`
   (`compute_levels`/`bitset_level_plan`; реэкспорт
   `ornis-core::compute_levels` сохраняется); собственная реализация
   уровней запрещена (движок один с закрытия S5).
3. **Тайбрейк конфликтов — порядок регистрации** (RaW/WaR/WaW поверх).
4. **Явный порядок (`order_before`) только разбивает уровни
   параллельности**, исполнение не пересортировывает.
5. **Новый фронтенд — только со вторым реальным потребителем** (YAGNI):
   расширение пространства ключей (TypeId/ResourceId/…) требует живого
   кейса, а не гипотетического.

Соответствие сегодня: рендер (Resource-ключевой фронтенд) — все пять
пунктов; `Schedule` ядра (generic, TypeId) — эталонный житель. Физика
встраивается крупной системой («шаг физики»: пишет в хранилища тел),
её внутренние острова/rayon — внутренность системы, как уровни пассов
у рендера. Иерархия: верхний `Schedule` планирует домены, домен
планирует своё нутро — вложенность, а не конкуренция.

**Веха интеграции** (после S6, вместе с приоритетом «a»): кадр гоняется
через верхний `Schedule` над `Resources`-миром (`Res<Device>`,
`Res<Queue>`, время, ввод): физика и рендер — системы-домены, скрипты
фазы 6 — data-фронтенд. Критерий приёмки: главный цикл (натив и wasm)
исполняет кадр через `Schedule`; физика впервые живёт в продакшн-цикле;
extract-фаза отсутствует by construction.

## S6 — ратификация (✅ 2026-08-19): реестр + отладочная проекция

`RenderGraph` понижен до внутреннего реестра объявлений + движка layout
+ рантайм-состояния; интерфейс объявлений — типы (S2–S5). Публичная
проекция графа — `GraphLayout::mermaid()`: уровни подграфами, пассы и
ресурсы узлами, потоки рёбрами; GitHub рендерит нативно — вставка в
PR-ревью превращает layout-дамп в картинку конвейера. Мёртвые ресурсы в
проекцию не входят.

Полный роспуск отклонён (паритет-оракул, нераспускаемое рантайм-ядро,
нулевые потребители императивного API, цена/риск) — полный список в
PLAN, Приложение C / S6. Оба исхода плана — успех; зафиксирован второй.

**Приложение C закрыто**: S1–S6 ✅, контракт шедулера и веха интеграции
(кадр через верхний `Schedule`, вместе с приоритетом «a») записаны.
Вне CI остаются: числа S0/recording-bench (ручной прогон на любой
машине) и probe-диффы на дискретном GPU.

## Hardening (2026-08-21): принуждение доступов, кеш уровневого плана, мягкий порядок

Пост-S5 доработка контракта по итогам внешнего ревью шедулера; код —
`ornis-core::schedule` + `RenderGraph`.

- **Принуждение объявленных доступов** (правило контракта «доступы —
  данные» получило зубы): пока исполняется система, `Resources::get`/
  `contains` проверяют ресурс против её `access()`; недекларированное
  чтение паникует с именем системы и типом ресурса. Собственная запись
  покрывает чтение (own-write read разрешён). Механика — thread-local
  стек активных деклараций (RAII-откат при паниках, корректен при
  вложенных шедулерах и rayon-переиспользовании потоков); вне
  `Schedule::run` доступ свободен. По умолчанию включено в debug,
  выключено в release (`set_enforce_accesses` переопределяет) —
  release-путь без накладных расходов.
- **Кеш уровневого плана + битсеты**: `Schedule` больше не пересчитывает
  уровни на каждый `run` — план кешируется и инвалидируется
  `add_system`/`order_before` (зеркалит S1-кеш `GraphLayout`; диагностика
  `level_computations()`). Внутри плана доступы спроецированы в
  `FixedBitSet` (плотный индекс по `TypeId`) — конфликты считаются
  пересечениями, а не линейными `Vec::contains` в O(n²)-цикле; явные
  рёбра — матрица смежности. Эталонная Vec-реализация оставлена как
  модель и пинится тестом `bitset_plan_matches_reference_model`
  (псевдослучайные доступы, LCG).
- **Мягкий явный порядок**: `Schedule::try_order_before`/`try_order_after`
  и `RenderGraph::try_order_before[_named]` возвращают `OrderError` /
  `GraphOrderError` (`UnknownSystem`/`UnknownPass`, `BackwardEdge`)
  вместо паники — паттерн S4 («невыполнимое — возвращаемая ошибка»).
  Паникующие варианты сохранены как тонкие обёртки (сообщения
  совместимы с тестами). Побочно: `try_order_before` в рендер-графе
  валидирует `PassId` — раньше ребро с неизвестным id добавлялось молча.
- **Переименование**: `ornis_core::Access` → `SystemAccess` — снята
  коллизия с типовым `ornis_render::Access` (`Read`/`Write`-маркеры
  графа); два разных понятия больше не называются одинаково.

Тесты (+6): `level_plan_is_cached_until_mutation`,
`bitset_plan_matches_reference_model`, `undeclared_read_panics_under_
enforcement` / `declared_access_passes_enforcement` /
`resources_are_unrestricted_outside_schedule`,
`try_order_before_reports_errors_without_panicking` (ядро и граф);
существующие тесты шедулера обновлены под декларирование лог-ресурсов
(они и раньше читали его не декларировав — принуждение сразу нашло
нарушителей в собственных тестах).

### Пасс-сторона принуждения (2026-08-23, бэклог #6)

Симметрия с TLS-enforcement систем: `PassViews::view_of` — единая воронка
выдачи view по `ResourceId` (типизированные `SystemViews` и императивные
run-замыкания сходятся здесь) — в debug проверяет `id` против declared
reads/writes пасса (`assert_pass_access_declared`, `frame_plan.rs`);
нарушение — паника с именем пасса и ресурса, аналог `sneaky system` —
`sneaky pass`. Release: `#[cfg(debug_assertions)]`, нулевая стоимость.
Честный лимит: `queue.write_buffer` в renderer-uniforms не проходит через
`view_of`, поэтому инвариант queue-backed буферов одного уровня — авторский
контракт (класс ограничения — как у rayon-границы систем ниже). Тесты:
`sneaky_pass_undeclared_access_panics`,
`declared_pass_access_passes_enforcement` (`frame_plan.rs`),
`pass_views_undeclared_view_panics_in_debug` (`frame_exec.rs`).

### Системная сторона на rayon-границе (2026-08-23, бэклог #7)

TLS-кадр принуждения действовал только в потоке `System::run`; дочерние
задачи системы стартовали с пустым стеком (аудит §3.3) — брешь ровно на
главном паттерне движка (`#[smart_pipeline]` генерирует `par_iter`).
Закрыто переносом кадра: `capture_access_frame` захватывает верхнюю
декларацию до входа в параллельную секцию, `AccessFrameCapture::install`
перевешивает её в рабочий поток (RAII, пустой снимок — no-op); макрос
генерирует пару вокруг `par_iter`-тел автоматически (обе ветви — одна
лента и zip), вложенные циклы наследуют кадр транзитивно. Ручной
`rayon`/`std`-параллелизм без захвата — задокументированный лимит (как и
debug-only по умолчанию). Тесты:
`undeclared_access_in_child_thread_panics_with_captured_frame`,
`declared_access_in_child_thread_passes_with_captured_frame`,
`capture_outside_schedule_run_is_noop`. Фаза B аудита с этим закрыта
целиком: ленты (#5) ✅, пассы (#6) ✅, rayon (#7) ✅.
