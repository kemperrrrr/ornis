# Архитектурное решение для перехода Rust → WGSL.

## Главная цель
Я бы не стремился сразу к «любой Rust-функции → WGSL». Лучше формализовать ограниченный **shader Rust**:
Rust
```
#[shader]
fn lighting(input: FragmentInput, resources: Resources) -> GBufferOutput {
    let normal = decode_normal(input.normal);
    let color = evaluate_material(resources.materials[input.material_id], normal);

    GBufferOutput {
        albedo: color,
        normal: encode_normal(normal),
        material_id: input.material_id,
    }
}


```
Это должен быть настоящий Rust-синтаксис, но с ограниченным типовым миром и семантикой GPU.
## Что стоит изменить архитектурно
### 1. Ввести собственный промежуточный IR
Сейчас `syn` почти сразу превращается в WGSL-текст. Лучше сделать:
Plain text
```
Rust AST
  ↓
Shader IR
  ↓
WGSL backend


```
Например:
Rust
```
enum Expr {
    Literal(Literal),
    Variable(Symbol),
    Binary { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Call { function: FunctionId, args: Vec<Expr> },
    Construct { ty: TypeId, fields: Vec<Expr> },
    Field { base: Box<Expr>, field: FieldId },
}

enum Stmt {
    Let { name: Symbol, value: Expr },
    Assign { target: Expr, value: Expr },
    If { condition: Expr, then_branch: Block, else_branch: Option<Block> },
    Return(Option<Expr>),
}


```
Тогда генератор WGSL не будет разбросан по десяткам `format!`:
Rust
```
format!("{}.{}", Self::expr(&f.base), member)
format!("{}({})", name, args.join(", "))
format!("{} = {}", lhs, rhs)


```
AST-конвертер строит IR, а отдельный `wgsl::Writer` печатает его. Позже можно добавить другой backend или более качественную диагностику.
### 2. Перестать моделировать shader через `String`
Особенно проблемные места сейчас:
- `lighting_wgsl_header() -> String`;
- `wgsl_decl`;
- `param_prefix`;
- `struct_lit`, который дописывает комментарий `/* field1, field2 */`;
- `write_module`, который после `naga` делает строковый `.replace(...)`.
Вместо этого нужен объект уровня модуля:
Rust
```
struct ShaderModule {
    types: Vec<TypeDecl>,
    globals: Vec<GlobalDecl>,
    constants: Vec<ConstDecl>,
    functions: Vec<Function>,
    entries: Vec<EntryPoint>,
}


```
А затем:
Rust
```
let shader = ShaderModule::new()
    .add_type(CameraUniform::shader_type())
    .add_type(LightingUniform::shader_type())
    .add_resource(camera_resource())
    .add_entry(lighting_entry());

shader.to_wgsl()


```
Если `naga` остаётся backend для деклараций — хорошо. Но не смешивай `naga`—вывод с постобработкой текста без необходимости. Исключение — workaround для `var<storage>`: это баг writer'а naga 30 (`address_space_str` безусловно печатает голый `var<storage>` для `Storage{LOAD}` без `STORE`, а по спекту WGSL это читается как read-write). На уровне построения `naga::GlobalVariable` это не решается — ручки в IR нет, наш `GlobalVariable` уже корректен. Опции: текущая постобработка с тестом-пином, патч вендора, issue апстриму.
### 3. Сделать ресурсы частью декларативного описания pass
Сейчас для каждого pass есть ручной массив:
Rust
```
pub const LIGHTING_RESOURCES: [Resource; 10] = [...]


```
И отдельно:
- `naga`-header;
- bind group layout;
- WGSL declarations;
- visibility;
- runtime binding code.
Лучше описывать pass одним объектом:
Rust
```
#[shader_pass]
mod lighting {
    #[uniform(group = 0, binding = 0, visibility = "vertex_fragment")]
    type Camera = CameraUniform;

    #[storage(group = 0, binding = 2, read_only, visibility = "fragment")]
    type Materials = [OpenPbrMaterial];

    #[texture(group = 0, binding = 3)]
    static ALBEDO_TEX: texture_2d<f32>;

    #[sampler(group = 0, binding = 9)]
    static LIGHTING_SAMPLER: sampler;
}


```
Из него макрос может генерировать:
- WGSL globals;
- `wgpu::BindGroupLayoutEntry`;
- resource metadata;
- проверку конфликтов binding;
- список доступных shader-идентификаторов.
Это уберёт рассинхронизацию между Rust runtime и WGSL header.
### 4. Убрать свободные идентификаторы из `#[stage]` — выполнено (bundle-форма)

Свободные идентификаторы убраны из всех 12 entry: ресурсы передаются
явно — либо одиночным `#[wgsl(global = "camera")] camera: ...`, либо (2+
ресурса) bundle-параметром `#[wgsl(context)] ctx: LightingContext`:

```rust
#[stage(fragment, entry = "fs_main", returns = "@location(0) vec4<f32>")]
fn lighting_fragment_entry(
    #[wgsl(location = 0)] uv: glam::Vec2,
    #[wgsl(context)] ctx: LightingContext,
) -> glam::Vec4 {
    let depth = textureLoad(ctx.depth_tex, /* … */ 0);
    // …
}
```

Правила:
- bundle-параметр уходит из WGSL-сигнатуры; `ctx.field` понижается до
  глобала `field` (имя поля === имя глобала, без маппинга);
- контракт — сама структура + `#[derive(ShaderContext)]` (`GLOBALS`),
  тест `stage_globals_declared` сверяет все 30 имён со собранными
  шейдерами;
- WGSL-константы переименованы в lowercase (`const quad`/`const uvs`),
  чтобы Rust-поле и глобал совпадали буквально;
- типы полей — настоящие Rust-типы (`CameraUniform`, `Vec<PerObjectGpu>`)
  либо номинальные DSL-маркеры (`Texture2d`, `Sampler`), которые никогда
  не инстанцируются.

Остаток свободных имён: константы `EPS`/`PI` в `#[wgsl_fn]`-хелперах и
имена kernel-функций — шаги 2–3 плана (единый `ShaderType`,
типизированное разрешение имён).

Честная граница: proc-макрос не видит определение структуры из `#[stage]`
(cross-item информации нет), поэтому соответствие «поле → глобал» —
конвенция, enforced тестами, а не макросом. Макрос проверяет только
комбинации (`context` несовместим с `builtin`/`location`/`global`) и
валидность синтаксиса опций.
### 5. Разделить shader-типы и CPU-типы
`glam::Vec3` удобен для CPU, но он не должен быть единственным источником shader-семантики. Нужен явный типовой слой:
Rust
```
struct F32;
struct U32;
struct Vec2<T>(T, T);
struct Vec3<T>(T, T, T);
struct Vec4<T>(T, T, T, T);


```
Или хотя бы traits:
Rust
```
trait ShaderType {
    const WGSL: &'static str;
}


```
Проблема текущего подхода — генератор угадывает семантику по именам:
Rust
```
"Vec3" => "vec3<f32>"
"new" => constructor
"lerp" => "mix"


```
Это работает, пока DSL небольшой, но быстро превращается в таблицу исключений. Лучше иметь явную семантику операций:
Rust
```
ShaderBuiltin::Mix
ShaderBuiltin::Dot
ShaderBuiltin::Normalize


```
А mapping Rust-метода в builtin делать на этапе lowering, с ошибкой у исходного токена.
## Что делать с `naga`
Есть два реалистичных пути.
### Вариант A — собственный IR + WGSL writer
Лучший вариант для твоего проекта сейчас.
Плюсы:
- сохраняется полный контроль над Rust-подобным DSL;
- хорошие source span-диагностики;
- можно генерировать именно читаемый WGSL;
- проще поддержать нестандартные ограничения проекта.
`naga` при этом использовать для:
- финального парсинга;
- валидации;
- reflection;
- проверки типов и binding-ов;
- round-trip тестов.
### Вариант B — строить весь shader через `naga::Module`
Это красиво для структур, глобальных ресурсов и констант, но заметно сложнее для полноценного тела функций и control flow. Можно прийти к этому постепенно, но не стоит насильно переводить весь проект на ручную сборку `naga` IR, если основной источник всё равно Rust AST.
Практичный компромисс:
Plain text
```
Rust declarations → naga IR
Rust expressions/functions → собственный Shader IR → WGSL
naga → финальная валидация всего WGSL


```
Позже функции тоже можно переводить в `naga` IR, если появится выгода.
## Как приблизиться к «настоящему Rust»
Порядок важен: сначала дешёвые шаги с compile-time эффектом, собственный
IR-писатель — последним, когда DSL упрётся (а не наоборот).
1. **Типизированный контекст stage-функций**
Свободные идентификаторы уходят первыми: DSL уже умеет `ctx.field[index]`,
не хватает только отображения «поле контекста → глобал» (набросок:
`#[wgsl(global = "QUAD")]`). Маленькая фича, compile-time проверка
существования ресурса — сразу.
2. **Стабилизировать типы и имена**
Убрать угадывание WGSL-типов по строковым именам, ввести единый `ShaderType`/`TypeId` и enum builtin'ов с lowering'ом вместо таблицы исключений.
3. **Типизированное разрешение имён**
Поля структур, функции, ресурсы и builtins разрешаются в compile-time symbols, а не остаются строками. Ошибка указывает на Rust-выражение.
4. **Вынести генерацию текста из `syn`-обходчика**
`syn` только строит IR; никакого `format!` внутри разбора выражений.
5. **Заменить сборку shader-модуля на декларативный builder**
Убрать ручные `header`, `wgsl_decl`, массивы snippets.
6. **Собственный IR + writer для тел** — только когда шаги 1–5 исчерпаны.
7. **Затем уже расширять язык**
`for`, `match`, массивы, mutable locals, helper calls, `discard`, texture sampling и т. д.
## Самая важная мысль
Не надо пытаться сделать компилятор «Rust напрямую в WGSL» через всё больше таблиц и `format!`. Следующий архитектурный шаг — не новый набор mappings, а **семантический слой между Rust AST и WGSL**.
Конкретно для текущего кода я бы первым большим рефакторингом сделал:
Plain text
```
crates/macros/src/wgsl.rs
    Rust AST → Shader IR

crates/macros/src/wgsl_ir.rs
    Expr / Stmt / Type / Function / Module

crates/macros/src/wgsl_writer.rs
    Shader IR → форматированный WGSL

crates/render/src/shaders/
    декларативные pass descriptions


```
После этого `lighting_wgsl_header`, `pbr_fragment`, `wgsl_decl` и большая часть ручной склейки либо исчезнут, либо станут очень маленькими декларативными описаниями. Ваши текущие `WgslStruct`, `WgslInterface` и `naga_ir` уже хорошо подходят как фундамент для такого перехода.

# Rust → WGSL: типизированный контекст и собственный IR

> **Статус:** принятое архитектурное направление  
> **Дата фиксации:** 2026-09-09

Этот документ фиксирует направление развития транслятора Rust → WGSL:
shader-код должен оставаться обычным, читаемым Rust-подмножеством, а WGSL
должен генерироваться из типизированного промежуточного представления без
ручных header-ов и склеек строк.

## Принятые решения

### Типизированный shader-контекст — реализован (плоская bundle-форма)

Stage-функции получают ресурсы через `#[wgsl(context)] ctx: Bundle`
(плюс одиночные `#[wgsl(global)]`-параметры). Реализованный стиль —
плоский, а не вложенный как в исходном наброске (`ctx.quad.positions`
ниже заменено на `ctx.quad`: вложенные структуры контекста — будущее,
когда появится типизированное разрешение имён):

```rust
#[stage(vertex, entry = "vs_main")]
fn vertex(
    vertex_index: super::VertexIndex,
    ctx: Context<super::QuadContext>,
) -> CompositeVertexOutput {
    CompositeVertexOutput {
        clip_position: ctx.quad[vertex_index],
        uv: ctx.uvs[vertex_index],
    }
}
```

Контекст содержит только ресурсы своей стадии; соответствие
поле → глобал — конвенция (имя совпадает буквально), enforced тестом
`stage_globals_declared`, а не макросом (proc-макрос не видит чужого
item'а). Проверка типов полей, binding-ов и stage-доступности —
шаги 2–3 плана миграции.

### Вариант A: собственный промежуточный IR

Выбран гибкий вариант с собственным shader IR:

```text
Rust AST → Shader IR → WGSL writer → naga validation
```

`syn` разбирает Rust и строит семантический IR, но не печатает WGSL
непосредственно. Отдельный writer отвечает за форматированный WGSL.
`naga` остаётся backend-инструментом для деклараций, финальной валидации,
reflection и round-trip тестов.

Это не попытка транслировать весь Rust. Поддерживается явное GPU-подмножество:
фиксированные типы, предсказуемый control flow, статически известные ресурсы
и отсутствие динамической аллокации.

## Что должно измениться

- Rust-код остаётся главным источником истины для layout-ов, интерфейсов,
  ресурсов, функций и entry points.
- Ручные `format!`, `push_str`, `wgsl_decl` и строковые header-ы уходят из
  семантического слоя компилятора.
- Постобработка готового WGSL через `.replace(...)` не используется.
- Типы и built-in-функции разрешаются через symbols и типы IR, а не через
  таблицы строковых совпадений.
- Ошибки указывают на исходное Rust-выражение: неизвестный ресурс,
  несовместимый тип, недоступный ресурс stage-функции или неподдерживаемая
  операция.
- До печати WGSL можно выполнять отдельные passes оптимизации и валидации.

## Целевая архитектура

```text
Rust declarations ───────┐
                          ├─> Shader module ─> WGSL writer ─> naga
Rust functions ─> AST ───┘       (IR)
```

Предполагаемые компоненты:

- `ShaderType`, `TypeId` и таблица symbols для типов и имён;
- `Expr`, `Stmt`, `Block`, `Function`, `EntryPoint`, `Global` и `Module`;
- lowering из `syn` в IR с сохранением `Span`;
- отдельный `wgsl_writer`;
- декларативное описание shader-pass, из которого генерируются WGSL globals,
  `wgpu::BindGroupLayoutEntry`, resource metadata и проверки.

## План миграции

1. Ввести типизированный контекст stage-функций — выполнено:
   `ctx: Context<Bundle>` + `#[derive(ShaderContext)]`
   (плоская bundle-форма, 12 entry, 30 имён под `stage_globals_declared`);
   одиночные ресурсы — `#[wgsl(global = "...")]`; builtin-индексы —
   newtypes `VertexIndex`/`InstanceIndex` без атрибутов; location'ы
   объявлены до функций — на полях `WgslInterface`-структур, entry берут
   готовый struct-input (`input: QuadVertexOutput`, а не голый
   `uv + location`); located-возвраты — типом
   `-> Location<0, glam::Vec4>` вместо строки `returns = "…"`
   (число + настоящий тип, оба проверяет rustc). Сигнатуры entry —
   чистый Rust без `#[wgsl(...)]` на bundle/newtype-параметрах: маркер
   bundle — сама обёртка `Context<…>` (bare-структура `input: VertexInput`
   остаётся настоящим WGSL-параметром).
2. Ввести `ShaderType`/`TypeId` и enum builtin'ов вместо угадывания по именам.
3. Вынести `Expr`, `Stmt`, `Type`, `Function`, `Module` и связанные symbols
   в отдельный Shader IR.
4. Разделить lowering Rust AST и печать WGSL: `syn`-обходчик больше не
   формирует WGSL через `format!`.
5. Ввести декларативное описание shader-pass.
6. Оставить `naga` финальным валидатором результата.
7. Постепенно расширять поддерживаемое подмножество Rust, сохраняя
   compile-time диагностику и round-trip тесты.

## Отклонённые альтернативы (разбор от 2026-09-09)

Рассмотрено предложение «type-level дескрипторы + build-time compiler
(build.rs) + убрать макросы/аннотации». Частично принято, частично
отклонено — зафиксировано, чтобы не возвращаться.

**Принято:**
- newtypes для builtin'ов (`VertexIndex`, `InstanceIndex`) — убирают
  аннотацию по-настоящему, а не перекладывают: `u32` неоднозначен
  (vertex/instance/location), newtype однозначен. Маппинг по имени типа —
  тот же механизм, что у glam (`Vec2` → `vec2<f32>`), до шага 2 плана
  (`ShaderType`).
- rustc-as-checker вместо «умного макроса»: proc-макрос cross-item не
  видит в принципе, проверка перекладывается на типы (bounds,
  `VisibleAt<S>`-маркеры — позже).
- Таблица трёх сущностей (layout / interface / context) — сходится с
  текущей, курс подтверждён.

**Отклонено:**
- Build-time compiler (build.rs с парсингом исходников): дублирует работу
  proc-макросов без инкрементальности cargo, хрупок вокруг
  cfg/алиасов/re-export/macro-expansion. Всё, что он даёт (проверка полей,
  сборка модуля, stage-visibility), достижимо иначе: компиляция entry
  против mock-контекстов в тестах, `wgsl_source()`-билдеры (обычные
  функции с полной видимостью), marker-traits. Цена — аддитивное время
  сборки + отдельный проект кэширования ради паритета с тем, что уже
  бесплатно.
- Const-generic биндинги в типах (`Uniform<0, 0, T>`): group/binding
  сейчас живут в одном массиве на пасс; зашивка их в тип превращает
  правку layout'а в правку сигнатур. Хуже для итерации.
- «Избавиться от макросов» как цель: закон сохранения метаданных —
  она переезжает, а не исчезает (в предложенном билдере имя ресурса и
  вовсе возвращается строкой `.resource::<…>("camera")`; поле структуры
  даёт настоящий идентификатор). Derive и lowering тел остаются
  макросами; цель — убрать их из семантического кода, что уже почти так.
- Налог `.0` (`ctx.vertex_index.0`) — шум в каждом шейдере + отдельное
  правило lowering'а; newtypes живут только в сигнатурной позиции, тела
  используют голый идентификатор.

**Цель вместо build.rs:** «валидный Rust в телах» — entry компилируются
против mock-контекстов (rustc проверяет поля), DSL понижает, naga
валидирует. Косметическое разделение `#[wgsl(…)]` на
`#[resource]`/`#[context]`/`#[builtin]` — отложено, рабочее не трогаем.

### Контракт entry-имён — один источник, не типы

`entry = "vs_main"` остаётся строкой сознательно: маркер-типы надёжности
не добавляют. Макросу нужна строка в момент экспансии, а `impl EntryPoint
for …` чужого типа он прочитать не может — строка осталась бы в `entry`
anyway, тип стал бы вторым написанием того же (больше дрейфа, не меньше).
Вместо этого renderer и `composite.rs` просят entry через сгенерированный
`entry_point()`-аксессор напрямую (14 литералов убраны): одно написание
кормит WGSL-функцию, аксессуар и запрос пайплайна; опечатка
самоконсистентна (безвредна), переименование распространяется само.
Внешний неймспейс `{vs_main, fs_main, vs, fs}` замкнут тестом
`entry_namespace_is_closed` — эти имена смотрят наружу (кэши пайплайнов,
снепшоты), их смена должна быть осознанной.

## Связанные документы
- [Идеи движка и долгосрочное направление](../../IDEAS.md), §4.5.
- [План реализации](../../PLAN.md), раздел «Rust → WGSL».
