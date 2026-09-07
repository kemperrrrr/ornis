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
> → `FrameOwned`), пример → `frame3d_probe`. Датированные секции
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

Осталось вне CI: пиксельные probe-диффы (`frame3d_probe`/
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

## S5d — гранулярность лент SmartStore в плане систем (✅ закрыт; 2026-08-23, бэклог #5 аудита; верифицировано 2026-09-07: `SystemAccess::reads_lane/writes_lane`, `FrameResource::kind()`, typed `SystemSet` — тесты `schedule_lanes.rs` + реестровые тесты `system.rs` зелёные)

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

**Веха интеграции** (срез 2026-08-28): native и WASM render loops уже
проходят через верхний `Engine`/`Schedule` frame host, общий
`ornis-render::RenderExtract` и `RenderFrame3D`/`FramePlan`. `Engine` также
предоставляет bounded `FixedTime` schedule; physics systems в editor-only и
native showcase используют этот host-level accumulator. Gameplay systems и
cross-domain physics/render/input ещё впереди. `RenderExtract` остаётся
явной переходной serialization/ECS boundary — критерий «без отдельной
extract-фазы» относится только к будущему полному unified scheduler, а не к
текущему runtime.

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

## S7 — GPU как системы (2026-09-06, native, шаг 1+2)

`GpuDevice`/`GpuQueue`/`GpuSurface`/`GpuSurfaceState` + `GpuFrameState{Renderer3D, RenderFrame3D, Mesh}` как ECS-ресурсы
(`crates/render/src/gpu_resources.rs`): `install_gpu_resources(device, queue, surface, surface_state, frame_state)` вставляет их в `World` и добавляет `RenderSubmit` + `RenderPresent`.

Модуль `gpu_resources` — native-only: `#[cfg(not(target_arch = "wasm32"))]` в `render/src/lib.rs` (2026-09-07) — wgpu web-типы `!Send`/`!Sync` (`Rc`/JS-колбэки), а `World::insert`/`Resources::get` требуют `Send + Sync`; wasm-путь рендерит через `RenderWorld` extraction + `ornis-wasm`, как и раньше.

- `RenderSubmit` (`reads Mutex<RenderExtracted>/Mutex<OrbitCamera>/GpuDevice/GpuQueue/GpuSurfaceState, writes Mutex<GpuFrameState>`): пересоздаёт `Mesh` по `mesh_params`, считает `view_proj` из `OrbitCamera + GpuSurfaceState.size`, `set_camera/set_lights/upload_materials/upload_instances` на `Queue`.
- `RenderPresent` (`reads GpuDevice/GpuQueue/GpuSurface/GpuSurfaceState/Mutex<RenderExtracted>, writes Mutex<GpuFrameState>`): `surface.get_current_texture → create_view → frame_plan.render → queue.submit/present` (основание — `RenderContext` из `render_backend.rs`). `Outdated`/`Lost` — реконфигурирует `Surface` на месте; `Occluded`/`Timeout`/`Validation` — пропускает кадр; `Suboptimal` как `Success`.

Интеграция native: `GameApp::initialize` клонирует `Device/Queue` и отдаёт `Surface` в ресурсы, `GameContext` больше не хранит `Renderer3D/FramePlan/Mesh/Surface/SurfaceConfig` отдельно; `GameApp::render_frame` → только `render_world.run_frame` (в `Engine::schedule` уже `RenderExtract → OrbitCamera → RenderSubmit → RenderPresent` на едином `bitset_level_plan`, уровни `Extract → Submit → Present`: RaW по `Mutex<RenderExtracted>` + WaW по `Mutex<GpuFrameState>`). `Resized` реконфигурирует `GpuSurface` (`Mutex<Surface>.configure`) по `GpuDevice` + обновлённому `GpuSurfaceState` и синхронит `GpuFrameState{renderer.resize, frame_plan.set_surface_size}`. `cargo check/clippy` чисто, `cargo test -p ornis-render --lib` 99 + `ornis-core` 142 зелёные.

## Роспуск оболочки FramePlan — стадия 1 (2026-09-06)

Первый шаг роспуска по рецепту `references/frameplan-dissolution.md` (перемещение, не удаление): hot-path handle layout переехал из плана в исполнитель.

- `FramePlan::generation: u64` — монотонный счётчик деклараций; каждая мутация (`create/import/external/add_pass/order/set_enabled/set_surface_size/set_budget/invalidate`, включая `PassBuilder::read/write/write_clear` — раньше сбрасывали только кеш в обход счётчика) идёт через `touch()` (сброс кеша + бамп).
- `FrameExecutor::ensure_layout(plan) -> Arc<FrameLayout>` — мемоизированный shared-снапшот по generation; steady-state — `Arc`-клон без рекомпута и без поколоночного клона векторов; `invalidate_layout()` — точечный сброс без трогания пула. `RenderFrame3D::render` идёт через него (паника при превышении бюджета — как у `layout()`).
- Удалены мёртвые `FramePlan::execute`/`PassContext` (нулевые потребители; тест `execute_delivers_live_resources_and_slots` переписан на прямой обход таблиц layout). `PassBuilder` задокументирован как тест/паритет-воронка; единственный prod-путь проводки — `SystemSet::{register_resource, add_system}`.
- Полное `cfg(test)`-гейтирование builder'а отклонено: его требует интеграционный `scheduler_parity.rs` (паритет-оракул `Schedule` vs `FramePlan`), внешнему крейту `cfg(test)`-API недоступен без новой фичи — цена без выигрыша (нарушает S6-причину #4).

Гейты стадии: `cargo check --workspace --all-targets` чисто, `cargo test -p ornis-render` зелено (lib 100: +1 `executor_memoizes_layout_across_frames`, golden/probe/proptest/parity без изменений — пул 7/10 слотов и бюджет-пины нетронуты). Производительность: steady-state кадр — одно сравнение `u64` + `Arc`-клон вместо `layout().clone()`; пул/алиасинг и кеш S1 сохранены — регресса нет по построению.

## Роспуск оболочки FramePlan — стадии 2+3 (d2/d3, 2026-09-07)

Динамическая половина `FramePlan` (то, что осталось от S1) вынесена в
`crates/render/src/transient_pool.rs`, а `SystemSet` стал единым prod-реестром
деклараций (d3-консолидация).

- **`transient_pool.rs` (d2)** — компилятор деклараций: `PoolInput` /
  `ResourceNode` / `PassNode` (pub(crate) input-снимки реестра),
  `TransientPool` с `ensure(generation, &input)` (мемоизация по
  `Arc<FrameLayout>`), `FrameLayout` / `ResourceLayout` / `PassLayout` /
  `PoolSlot` (выход компиляции), `Budget` / `BudgetExceeded` / `SizePolicy` /
  `TextureSpec` (+ `TextureSpec::external()` для external-output'ов),
  `format_bytes_per_pixel` / `budget_exceeded` / `assert_pass_access_declared`.
  Реэкспорт через `lib.rs` сохраняет обратную совместимость со всеми
  downstream-потребителями типов.
- **`SystemSet` (d3)** — теперь сам реестр: `resources` / `passes` / `ordering`
  / `budget` / `pool` / `generation` живут здесь. `FrameExecutor::ensure_layout`
  принимает `&SystemSet` (а не `&mut FramePlan`) и идёт через `pool_input()`
  + `TransientPool::ensure`. `pool_input` помечен `#[allow(dead_code)]` (mirror
  на `FramePlan::pool_input`) — на случай будущих cold-path потребителей.
  `Debug` impl ручной: `TypeId` и `Box<dyn FnMut>` не `Debug`, поля `ids` /
  `systems` скипаются с подсчётом длины (диспатч-интернал намеренно opaque в
  тестах).
- **`FramePlan` остаётся** как имperative/parity-фронтенд с собственным
  `TransientPool` — cold-path `FramePlan::layout()` для инструментов и тестов;
  продакт идёт через `SystemSet`. Полное удаление `FramePlan` — решение
  владельца (стадия 4).
- **Тесты** — `frame_exec.rs` (lib) + `tests/{budget_proptest,parallel_render,
  scheduler_parity}.rs` обновлены под `(&mut .systems)` / `systems_mut()` /
  `(&g3.systems)` / `.plan → .systems` (поле `RenderFrame3D` переименовано);
  `executor_memoizes_layout_across_frames` переписан на `SystemSet` (d3 —
  продакт-путь). `layout_bench.rs` (bench) переведён на `systems_mut()`.

Гейты стадий 2+3: `cargo check --workspace --all-targets` чисто; `cargo test
-p ornis-render` зелено (lib 100 + integration 4 = 104, без регрессий);
`cargo clippy --workspace --all-targets -- -D warnings` чисто. Производительность
не затронута: hot-path контракт (одно `u64`-сравнение + `Arc`-клон) сохранён;
пул/алиасинг/кеш S1 — без изменений.

## Роспуск оболочки FramePlan — стадия 4 (d4, 2026-09-07)

`FramePlan` удалён полностью. `SystemSet` — единственный реестр деклараций
и единственный prod-потребитель `TransientPool`; паритет-оракул в
`scheduler_parity.rs` теперь сверяет `ornis_core::Schedule` против
`ornis_render::SystemSet` (а не против удалённого `FramePlan`).

- `crates/render/src/frame_plan.rs` (806 строк) удалён целиком вместе с
  его `#[cfg(test)] mod tests` (19 тестов).
- `crates/render/src/lib.rs`: `pub mod frame_plan` и `pub use frame_plan::
  FramePlan` сняты; реэкспорты типов пула через `transient_pool`
  сохранены (публичный API рендера стабилен для downstream).
- `crates/render/src/frame_exec.rs` тесты: `imperative_resources`/
  `imperative_passes` принимают `&mut SystemSet`; `imperative_wiring`
  возвращает `SystemSet`; `FramePlan::new(size)` →
  `SystemSet::new()` + `set_surface_size(size)`; тест
  `pass_views_undeclared_view_panics_in_debug` (sneaky-pass) переведён
  на `SystemSet`.
- `crates/render/tests/scheduler_parity.rs`: импорт `SystemSet` вместо
  `FramePlan`; `FramePlan::new((w,h))` → `SystemSet::new()` +
  `set_surface_size((w,h))`; имена ресурсов `"r0".."r7"` литералами
  (контракт `create_resource(&'static str, _)`).
- `crates/render/src/system.rs::tests`: добавлены 6 реестровых тестов,
  переехавших из удалённого `frame_plan.rs::tests`:
  `unknown_resource_panics`, `explicit_ordering_rejects_backward`,
  `explicit_ordering_unknown_name`, `try_order_before_reports_errors_without_panicking`,
  debug-only `sneaky_pass_undeclared_access_panics` и
  `declared_pass_access_passes_enforcement`.
- `crates/render/src/transient_pool.rs::tests`: добавлены 14 тестов
  уровня пула с прямым `PoolInput` (без реестра): `lifetime_window_basic`,
  `transient_slot_reuse_same_spec`, `overlapping_resources_need_distinct_slots`,
  `read_before_write_panics`, `imported_resource_may_be_read_first`,
  `disabled_pass_culls_its_resources` (двойная проверка — через
  `SystemSet::set_pass_enabled` и через прямой `PassNode { enabled: false }`),
  `independent_branches_share_levels`, `explicit_ordering_splits_shared_level`,
  `layout_tables_walk_for_each_pass` (бывший `execute_delivers_live_…`),
  `mermaid_is_a_valid_projection`, `debug_dump_lists_structure`,
  `clear_value_is_carried_to_layout`, `layout_is_cached_until_mutation`,
  `generation_bump_invalidates_cache`, `build_snapshot_matches_cached_layout`.
  `ResourceNode`/`PassNode` получили `#[derive(Clone)]` (внутренние типы,
  API не затронут).
- Комментарии в `gpu_resources.rs`, `wasm/src/lib.rs`,
  `schedule/src/lib.rs`, `frame_exec.rs`, `system.rs`,
  `transient_pool.rs` обновлены на актуальные имена.

Гейты: `cargo check --workspace --all-targets` чисто;
`cargo test -p ornis-render` зелено (lib 100 + integration 4 = 104,
без регрессий против стадий 2+3 — каждое assertion из удалённых
19 тестов `frame_plan::tests` либо переехало в `system::tests`/
`transient_pool::tests`, либо покрыто существующим тестом в
`frame_exec::tests`); `cargo clippy --workspace --all-targets
-- -D warnings` чисто.

Следствие для канона: паритет-оракул теперь подтверждает **один
фронтенд** (`SystemSet`) в двух формах декларации — типизированной
(`add_system<P: FramePass>`) и императивной (`add_pass().read/write`).
Это сильнее прежнего «два разных фронтенда» (`FramePlan` vs
`Schedule`): движок уровней один (`ornis-schedule::bitset_level_plan`
— его зовут и `core::Schedule`, и `TransientPool`), а `Schedule` и
`SystemSet` остаются двумя фронтендами над ним; внутри рендера
поверхность декларации едина.

## S6 — пересмотр после d4 (✅ решение подтверждено, 2026-09-07)

`FramePlan` удалён (d4, −806 строк), `SystemSet` — единственный
реестр деклараций рендера. Вопрос S6 «распускать дальше или оставить
как render-aware проекцию над `Schedule`» пересмотрен на новой картине;
решение: **оставить `SystemSet`+`TransientPool` как render-aware
проекцию**, не вливать в `core::Schedule`. Причины:

1. **Доменное состояние не проецируется**: пул текстур, lifetime-окна
   `[first_use, last_use]`, бюджет S4, external views, pooled GPU-объекты
   `FrameExecutor` — рантайм-состояние кадра, а не декларации доступов.
   При «роспуске» оно переезжает в `Schedule`, а не исчезает (причина #2
   из 2026-08-19 действует и после d4).
2. **Пространства ключей разные по делу**: `ResourceId` (текстуры пула,
   sharing по `TextureSpec`) vs `TypeId` ресурса/ленты `Schedule`
   (синглтоны + `SmartStore`-лейны). Склейка дала бы либо строки-ключи,
   либо потерю sharing-проверок пула.
3. **Исполнение разное**: пассы пишут в `wgpu::CommandEncoder`
   (последовательно — один encoder, параллельно — per-pass encoders +
   submit в порядке регистрации, `FrameExecutor::execute[_parallel]`);
   системы `Schedule` исполняют `run(&Resources)` через rayon без
   encoder-контекста. Общий знаменатель — только уровни
   (`bitset_level_plan`), и он уже общий.

Следствие для S5e: сближение идёт через валюту `AccessDesc` и общие
уровни (паритет-оракул `scheduler_parity.rs`), а не через ликвидацию
типа. Паритет подтверждает один фронтенд (`SystemSet`) в двух формах
декларации (типизированная + императивная) поверх одного движка с
`Schedule`. Пересмотр — только со вторым живым потребителем
data-фронтенда (фаза 6) или сменой ключа пула.

## S5e + Extract-free — декомпозиция (2026-09-07, закрыто; E1–E2, X1–X4, E3 ✅)

Честная оценка: это месяцы, не один коммит. Ниже — фазы с собственными
гейтами; каждая фаза — самостоятельный выигрыш и откатываема. База:
движок уровней уже один (`bitset_level_plan`), `FrameCommandBuffers`
(стадия 1 handover, `gpu_resources.rs`) доказывает `Send + Sync`
завершённых буферов, `execute_parallel` доказывает per-pass encoders +
submit в порядке регистрации.

### S5e: пассы как обычные `Schedule`-системы

- **E1 ✅ — пасс как `System`-адаптер**: каждая `FramePass`-реализация
  получает тонкий `System`-близнец с тем же `AccessDesc` (проекция
  `ResourceId`-доступов в `SystemAccess` через реестр `SystemSet`).
  Исполнение — уровни `Schedule`, запись — через borrowed encoder
  (как сегодня `Frame { encoder }`). Гейт: уровни адаптеров ==
  `FrameLayout::levels()` (расширение `scheduler_parity.rs`), probe
  0 отличий.
- **E2 ✅ — encoder-контекст как frame-ресурс**: `RenderPresent` пишет
  per-pass/per-level encoders, завершённые буферы складывает в
  `FrameCommandBuffers`, отдельная система сливает их в порядке
  регистрации (механика уже проверена `execute_parallel`).
  Гейт: sequential vs schedule-driven пути пиксельно идентичны.
- **E3 ✅ — пул/бюджет остаются render-side**: `TransientPool`-компиляция
  (lifetime/слоты/бюджет) не переезжает в `Schedule` — `Schedule`
  потребляет только уровни, пул остаётся проекцией (решение S6 выше).
  No-op: `transient_pool.rs` не менялся, golden-тесты слотов/бюджета
  зелёные без изменений.

### Extract-free: от `Mutex<RenderExtracted>` к прямым `Res`/лейнам

- **X1 ✅ — upload-системы читают лейны**: `RenderSubmit` сегодня клонирует
  `Mutex<RenderExtracted>`; перевести `upload_instances/upload_materials`
  на прямое чтение `TransformDesc`/`MeshDesc`/`MaterialDesc`-лейн
  (`reads_lane`, канон S5d). Гейт: `extract_render_data` остаётся
  оракулом — прямое чтение побайтово равно снапшоту.
- **X2 ✅ — меш как ресурс**: `mesh_params`/`Mesh` пересоздание — из лейн,
  а не из клона снапшота. Гейт: probe сцен с разной тесселяцией.
- **X3 ✅ — свет/камера как ресурсы**: `OrbitCamera` уже `Mutex`-ресурс;
  захардкоженные `set_lights` в `RenderSubmit` перевести на `LightDesc`
  из мира. Гейт: probe освещения 0 отличий.
- **X4 ✅ — удаление `Mutex<RenderExtracted>`**: последний читатель
  мигрирует, тип удаляется, `RenderWorld` остаётся только
  сцено-загрузчиком (serialization boundary), не кадровым снапшотом.
  Гейт: `grep RenderExtracted` пуст вне истории; весь `cargo test
  -p ornis-render` зелен.

Порядок: E1 → E2 → X1 → X2 → X3 → E3/X4. E1 разблокирует всё остальное;
X1–X3 независимы между собой после E2. Все пункты закрыты (2026-09-07).

### X4 — удаление `Mutex<RenderExtracted>` ✅ (2026-09-07)

Закрыт шестым шагом декомпозиции — Extract-free завершён.

- `RenderExtracted` переименован в `FrameUpload` (payload прямого
  чтения, не снапшот): снапшот-система `RenderExtract`, ресурс
  `Mutex<RenderExtracted>` и `install_render_extract` удалены;
  `RenderWorld::extracted()` → `frame_upload()` (прямой
  `extract_render_data` на сторе мира). `RenderWorld` — только
  сцено-загрузчик (serialization boundary) + хост `Engine`-кадра.
- `RenderPresent` (последний читатель) мигрировал: instance count —
  прямой `extract_render_data(store)`, в access — `SmartStore` + три
  `reads_lane` (как у `RenderSubmit`/`RenderMesh`).
- wasm: `GpuScene.extracted: FrameUpload`, `build_gpu_scene` и тест
  контракта — на `frame_upload()`; в расписании `RenderWorld` больше
  нет систем (`schedule().len() == 0`).
- Гейт: `grep RenderExtracted` — пуст (вне истории/доки); весь
  `cargo test -p ornis-render` должен быть зелёным (CI).

### X3 — свет как ресурс ✅ (2026-09-07)

Закрыт пятым шагом декомпозиции. Камера уже ресурс (`Mutex<OrbitCamera>`
с S7); мигрировал свет.

- Новый ресурс `RenderLights` (`extraction.rs`): ambient + `Vec<LightDesc>`.
  Пишется сцено-загрузчиком между кадрами (`RenderWorld::replace_scene`
  публикует `scene.lights`/`scene.ambient`), в расписании только читается —
  без `Mutex`, как `GpuSurfaceState`. Дефолт — legacy-риг через `const`
  (тот самый, что был захардкожен в `RenderSubmit`), `install_gpu_resources`
  вставляет его: рантайм без сцены рендерит ровно как раньше.
- `RenderLights::set_lights_args()` — единственная конверсия
  `LightDesc → (direction, intensity, color)` (канон для системы и тестов);
  wasm (`build_gpu_scene`) мигрировал на неё — inline-`match` в
  `ornis-wasm` удалён.
- `RenderSubmit`: `reads RenderLights`, хардкод `set_lights` удалён.
- Гейты: юнит (дефолт == legacy-аргументы побайтово; `replace_scene`
  публикует свет сцены) + пиксельный probe `tests/light_resource.rs` на
  общем GPU-харнессе: рендер с legacy-аргументами vs ресурс через
  `set_lights_args` — 0 отличий.

### X2 — меш как ресурс ✅ (2026-09-07)

Закрыт четвёртым шагом декомпозиции.

- `Mesh` + кеш тесселяции вынесены из `GpuFrameState` в отдельный
  ресурс `Mutex<GpuMesh>`: пересоздание меша больше не держит лок
  renderer'а/frame-plan'а. Порядок локов задокументирован: `GpuMesh`
  раньше `GpuFrameState` (иначе его держит только `RenderPresent`).
- Новая система `RenderMesh` (`install_render_mesh`, X2): читает те же
  три лейна (`reads_lane`), пишет только `Mutex<GpuMesh>`; критерий —
  `max_mesh_params` из `extraction.rs`: максимум тесселяции по ПОЛНЫМ
  сущностям с полом (32, 24) — канон выделен из `extract_render_data`
  (не клона снапшота). `RenderSubmit` больше не трогает меш (и не
  читает `GpuDevice`); `RenderPresent` берёт меш из `GpuMesh`-ресурса
  (RaW после `RenderMesh`), unsafe-блок сузился до renderer+frame3d.
- Гейты: canon-tie в `extraction.rs` (`max_mesh_params` ==
  `extract_render_data().mesh_params`; неполная сущность 96/64 не
  двигает максимум) + probe-тест `tests/mesh_resource.rs` на общем
  GPU-харнессе — три сцены (48/32 → +16/12 без изменений → +64/40 с
  пересозданием), vertex/index counts сверяются с эталонной сферой.

### X1 — upload-системы читают лейны напрямую ✅ (2026-09-07)

Закрыт третьим шагом декомпозиции (первый Extract-free).

- `RenderSubmit` больше не читает `Mutex<RenderExtracted>`: данные
  материалов/инстансов — прямое чтение `TransformDesc`/`MeshDesc`/
  `MaterialDesc`-лейн (`reads_lane`, канон S5d) через тот же канон
  `extract_render_data`, что и у оракула. Зависимость от пишущих лейны
  систем — RaW по лейнам (тоньше RaW по всему снапшоту): upload стоит
  после реальных писателей, но в одном уровне с другими читателями.
- `RenderExtract`/`Mutex<RenderExtracted>` остаются оракулом: гейт —
  прямой `extract_render_data(store)` после `run_frame` побайтово равен
  снапшоту, опубликованному системой (материалы — по байтам `bytemuck`,
  инстансы — по полям; сцена с тремя видами материалов и разной
  тесселяцией). Последний читатель снапшота — `RenderPresent`
  (instance count) — мигрировал в X4.

### E2 — encoder-контекст как frame-ресурс ✅ (2026-09-07)

Закрыт вторым шагом декомпозиции.

- `FrameExecutor::record_in_order` — последовательная запись механики,
  уже доказанной `execute_parallel`: по энкодеру на pass/level,
  завершённые `CommandBuffer` уходят в общий sink в порядке регистрации.
- `RenderFrame3D::render_to_buffers` — вторая форма записи поверх
  общего `projected_order`: без заимствованного `RenderContext` и без
  submit. Заимствованный `render()` остаётся для wasm/examples/benches.
- Нативный `RenderPresent` больше не субмитит сам: acquired frame
  уходит в `FramePresentTarget`, записи — в `FrameCommandBuffers`
  (оба — `Mutex`-handover, `Send + Sync`, чтобы жить в `Resources`).
- `RenderFlush` (регистрация следом → WaW по обоим handover-ресурсам)
  делает один ordered submit (`FrameCommandBuffers::flush` = drain +
  submit; система не видит internals) и present.
- Гейт — третий тест на общем GPU-харнессе (`schedule_render.rs`):
  sequential reference vs `render_to_buffers` + ordered flush —
  пиксельно идентичны.

### E1 — пасс как `System`-адаптер ✅ (2026-09-07)

Закрыт первым шагом декомпозиции. Мост `crates/render/src/schedule_bridge.rs`:

- `try_project_schedule(&SystemSet) -> Schedule` — каждый включённый пасс
  становится `PassSystem`-близнецом: то же имя, доступы спроецированы из
  `ResourceId` в `TypeId` через реестр (`register_resource::<R>`).
  Disabled-пассы и их рёбра выпадают ровно как в `layout_levels`.
  Нетипизированный ресурс (`create_resource` без типа) — честный
  `ProjectionError::UntypedResource`: ядру нечего зеркалить.
- `run` у близнеца — no-op по дизайну: E1 исполняет пассы через
  borrowed-encoder dispatch, близнец управляет levelling'ом. Запись внутрь
  систем через frame-ресурс — это E2 (`FrameCommandBuffers`), не этот шаг.
- `RenderFrame3D::render_schedule` — рендер кадра, ordenированный
  уровнями спроецированного `Schedule` (flatten уровней →
  `FrameExecutor::execute_in_order`, тот же `dispatch_pass`, что у
  sequential/parallel путей); debug-assert на каждый кадр проверяет
  уровни == `FrameLayout::levels()`. Fallible: возвращает
  `ProjectionError` при нетипизированном ресурсе. Проекция строится на
  вызов — E2 поднимет владение schedule в рантайм.

Гейты: `scheduler_parity.rs` расширен каноном E1 (проекция ==
`FrameLayout::levels()` на цепочках/shared-levels/edge-splits,
disabled-culling, untyped-error; плюс production-матрица Technique×bloom
в юнит-тестах моста) и `tests/schedule_render.rs` — пиксельный паритет
schedule-driven vs sequential на lavapipe (0 отличий). GPU-harness двух
гейтов (S5b и E1) дедуплицирован в `tests/common/mod.rs` — один
headless-сценарий, мишени и readback вместо копий в каждом гейте
(ratchet-clean по findings).

## WASD-мост: фикс шва sync + порядок (2026-09-06)

Тест `browser_wasd_input_drives_player_through_gameplay` падал на чистом master (предсуществующее, не регресс роспуска). Две наложенные причины:

- **Шов `sync_external_pose` (`src/engine_runtime.rs`)** не пробрасывал скорость связанным dynamic-телам (solver-авторитетность): intent из `velocity_to_body` умирал в ECS-лейне, `sync_out` прикалывал позу обратно каждый тик. Фикс: dynamic-телам форвардятся `velocity` + `angular_velocity` из ECS-источника (для тел без intent ECS-копия совпадает с solver-состоянием с прошлого `sync_out` — тождественно, безопасно).
- **Порядок в fixed-расписании**: мост регистрировался после физики, а явные рёбра действуют только вперёд по порядку регистрации (S3-контракт) — backward-ребро молча игнорируется, intent опаздывал на тик. Фикс: `VelocityToBodySystem` ставится через `prepend_system` + best-effort `try_order_before("velocity_to_body", "physics_sync_in")`. Порядок стал `velocity_to_body → sync_in → step → sync_out`, задокументированная 2-тиковая латентность `Input→Position` восстановлена.
- Тест обновлён под новый DAG: игроку нужно dynamic-тело (`spawn` даёт static с массой 0 — мост его осознанно пропускает, `body_to_transform` прикалывает позу).
