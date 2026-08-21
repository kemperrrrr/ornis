# Unified Scheduler — S0/S1 (кеш GraphLayout)

> Рабочий документ этапов S0–S1 из **Приложения C** [`PLAN.md`](../../PLAN.md)
> (идея — [`IDEAS.md`](../../IDEAS.md) §28). Каждый прогон бенчей/тестов
> обновляет числа в этом файле.

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
| Forward+блум | 7 | 12 | _заполнить после первого `cargo bench -p ornis-render`_ | |
| Deferred+блум | 8 | 12 | _—_ | |
| Hybrid+блум | 9 | 12 | _—_ | |

### Кеш-попадание (S1)

| Граф | Время `layout()` (cache hit) | Выигрыш vs compute | Примечание |
|---|---|---|---|
| Forward+блум | _—_ | _—_ | |
| Deferred+блум | _—_ | _—_ | |
| Hybrid+блум | _—_ | _—_ | |

> ⚠️ Окружение, в котором написан S0/S1 (2026-08-18), не имеет Rust
> toolchain — числа появятся после первого прогона
> `cargo bench -p ornis-render` и `cargo xtask quality`. Прецедент
> оформления: заметка G7 в PLAN.md.

### Пул текстур по техникам (`texture_budget`, lavapipe/Metal)

_Заполнить с первым прогоном `cargo run --example render_graph_probe`_:
слоты/байты уже печатаются примером. Точка сравнения из B1-R7: 9 ресурсов →
7 слотов, −20,0% на 1280×720 (без блума); +3 слота и +3,8 MB с блумом.

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

Осталось вне CI: пиксельные probe-диффы (`render_graph_probe` — нужен
GPU-адаптер; ожидаемо 0 отличий: проводка идентична, тела пассов не
менялись) и числа `cargo bench -p ornis-render` для таблиц S0.

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

## S5 — закрытие (2026-08-19, без GPU-машины)

- **Единый движок уровней**: `ornis-core::compute_levels(n, ordered(i,j))`
  — одна функция считает уровни параллельности и для `Schedule` (системы,
  ключи TypeId), и для `GraphLayout::levels` (пассы, ключи ResourceId).
  Дублирование S5a/S5c устранено: «один scheduler на всё» теперь буква
  кода, а не только паттерн. Два фронтенда остаются осознанно:
  texture-ключевой (пул видов графа) и generic-ресурсный (`Resources`).
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
