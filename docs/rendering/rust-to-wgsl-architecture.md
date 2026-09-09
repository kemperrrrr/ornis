//! Архитектурное решение для перехода Rust → WGSL.

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
Если `naga` остаётся backend для деклараций — хорошо. Но не смешивай `naga`-вывод с постобработкой текста. Например, workaround для `var<storage>` лучше решить на уровне построения `naga::GlobalVariable` или отдельного writer-а, а не заменой строк после генерации.
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
### 4. Убрать свободные идентификаторы из `#[stage]`
Сейчас stage-тело может использовать `QUAD`, `UVS`, `per_objects` и другие имена, которых Rust-компилятор не видит. Это удобно для прототипа, но сильно ухудшает читаемость и диагностику.
Вместо этого передавай ресурсы явно:
Rust
```
#[stage(vertex, entry = "vs_main")]
fn composite_vertex(
    #[builtin(vertex_index)] index: u32,
    quad: &QuadData,
) -> CompositeVertexOutput {
    CompositeVertexOutput {
        clip_position: quad.positions[index],
        uv: quad.uvs[index],
    }
}


```
Или введи специальный тип контекста:
Rust
```
fn vertex(ctx: VertexContext, index: u32) -> CompositeVertexOutput


```
Макрос сможет проверить:
- существует ли ресурс;
- совпадает ли тип;
- разрешён ли ресурс на данной стадии;
- не используется ли texture в vertex stage без соответствующей возможности.
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
Я бы мигрировал по этапам:
1. **Стабилизировать типы и имена**
Убрать угадывание WGSL-типов по строковым именам, ввести единый `ShaderType`/`TypeId`.
2. **Вынести генерацию текста из `syn`-обходчика**
`syn` только строит IR; никакого `format!` внутри разбора выражений.
3. **Заменить сборку shader-модуля на декларативный builder**
Убрать ручные `header`, `wgsl_decl`, массивы snippets и постобработку `.replace`.
4. **Сделать stage-функции валидным Rust-кодом**
Никаких свободных WGSL-ресурсов. Все inputs/resources должны быть видны макросу и типизированы.
5. **Добавить typed name resolution**
Поля структур, функции, ресурсы и builtins должны разрешаться в compile-time symbols, а не оставаться строками.
6. **Добавить нормальные diagnostics**
Ошибка должна указывать на Rust-выражение:
- «`Vec3::lerp` не поддерживается»;
- «resource `materials` недоступен во vertex stage»;
- «поле `material_id` отсутствует в `GBufferOutput`».
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

### Типизированный shader-контекст

Stage-функции получают ресурсы, входы и выходы через Rust-типы контекста.
Свободные WGSL-идентификаторы (`QUAD`, `UVS`, `materials` и подобные) не
должны быть основным API. Контекст делает зависимости функции явными и
позволяет проверять их на этапе компиляции.

Целевой стиль:

```rust
#[shader_stage(vertex, entry = "vs_main")]
fn vertex(
    ctx: VertexContext,
    #[builtin(vertex_index)] index: u32,
) -> CompositeVertexOutput {
    CompositeVertexOutput {
        clip_position: ctx.quad.positions[index],
        uv: ctx.quad.uvs[index],
    }
}
```

Контекст должен содержать только разрешённые для конкретной стадии ресурсы.
Макрос проверяет типы полей, binding-и, доступность ресурса и соответствие
интерфейсов; генератор не получает имена через неявные строки.

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

1. Вынести `Expr`, `Stmt`, `Type`, `Function`, `Module` и связанные symbols
   в отдельный Shader IR.
2. Разделить lowering Rust AST и печать WGSL: `syn`-обходчик больше не
   формирует WGSL через `format!`.
3. Ввести декларативное описание shader-pass.
4. Перевести stage-функции со свободных binding-ов на типизированный context.
5. Оставить `naga` финальным валидатором результата.
6. Постепенно расширять поддерживаемое подмножество Rust, сохраняя
   compile-time диагностику и round-trip тесты.

## Связанные документы

- [Идеи движка и долгосрочное направление](../../IDEAS.md), §4.5.
- [План реализации](../../PLAN.md), раздел «Rust → WGSL».
