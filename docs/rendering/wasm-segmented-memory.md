# Wasm → GPU: сегментированная виртуальная память

Статус: research note (2026-10-25; срез вызовов, `switch` и compute — 2026-10-29; потолок специализаций считается по уникальным входам, не по инструкциям — 2026-10-29; указатели на storage через `storage_base` + `i32.load` / `i32.store` — 2026-10-29).
Не принято как замена текущему пути Rust → Shader IR → WGSL
(`rust-to-wgsl-architecture.md`). Фиксирует идею «Virtual Memory
Space Translation / Software-Managed Memory Paging» поверх Wasm linear
memory и предложения по ABI надстройки. Research-прототип (не
production path) — `crates/gpu-abi` (`GpuRef` + `GpuSafeLayout`,
`unsafe` только в `gpu_abi`) и offline `crates/wasm2gpu`.
Toy-проход: vertex+fragment, uniform и `texture_sample`. Storage
`storage_read` / `storage_write` покрывает скаляры (`f32` / `u32` /
`i32`), `vec2` / `vec3` / `vec4<f32>` и простые структуры. Uniform
на том же контракте: `uniform_read_struct`, `uniform_field_*` (индекс
поля только константа) и `uniform_read_*` / `uniform_index_struct`
(длина константа, индекс элемента может быть динамическим, если WGSL
это принимает). Раскладка структуры — `place_fields` с выравниванием
uniform (16), та же, что у хостового `GpuRef` / `gpu_struct!`. Поля
задаёт custom section `ornis_gpu.types`. Group и binding остаются
immediate-операндами импорта; индекс элемента может быть динамическим
и опускается в access chain (`OpAccessChain` / WGSL `array[index]`).
Структурный control flow (`block` / `loop` / `if` / `br` / `br_if` /
`br_table`) строится в CFG; абстрактная интерпретация несёт provenance
указателя и offset через ветки и join (`phi` / `select`, §3.4). Два
разных binding на одном join — отказ. `br_table` опускается в WGSL
`switch`, поэтому Rust `match` доходит до GPU. Прямой `call`
определённой функции модуля становится функцией naga: константный
group/binding запекается в специализацию, динамический offset проходит
аргументом или результатом. Одинаковый набор константных аргументов
берётся из кэша. Рекурсия, `call_indirect` и выход за потолок
специализаций — явная ошибка.
Прямой `if` без побочных эффектов опускается в WGSL `select`, остальной
— в `if` / `loop`. Неподдерживаемый тип — явная ошибка. WGSL/SPIR-V
через naga. Bindless и descriptor indexing не default. Экспорт `shade`
и, если он есть, `compute` в `crates/wasm2gpu-shader` помечены
`#[unsafe(no_mangle)]`: лифтер gpu-теста ищет эти имена, а в Rust 2024
атрибут unsafe. Это не блок `unsafe` и не код `gpu-abi`. `compute`
лифтер переименовывает в `cs_main`: `@workgroup_size(4, 1, 1)`, builtin
`global_invocation_id`, запись в storage. Упаковка
space+binding+offset в каждый `i32` по-прежнему не default ABI.

Связано: `rust-to-wgsl-architecture.md`, обсуждения Wasm GC / DWARF /
Kongruent как носителей типов для шейдерного бэкенда.

## 1. Мотивация

`rustc` для `wasm32` выдаёт плоский массив байт (MVP linear memory) и
инструкции вроде `i32.load` / `i32.store`. GPU же требует раздельных
адресных пространств (uniform / storage / private / push-constant) и
привязок `@group` / `@binding`, а не одной кучи.

Если не менять факт плоской памяти, можно перехватить **правила
адресации**: объявить, что часть бит указателя — пространство и
ресурс, а нижние — offset, и в трансляторе `wasm2gpu` поднимать это
в нативные `OpAccessChain` / WGSL-глобалы.

Цель надстройки — превратить плоскую Wasm-модель в структурированную
GPU-модель без отказа от `rustc` как фронтенда тел шейдеров.

## 2. Исходный концепт (идея)

Нарезка `i32`-указателя (пример):

| Биты   | Поле        | Смысл                                      |
|--------|-------------|--------------------------------------------|
| 28–31  | space       | 0 Private, 1 Uniform, 2 Storage, 3 PushConst |
| 16–27  | resource id | номер `@binding` / дескриптора (до 4096)   |
| 0–15   | offset      | смещение внутри буфера (до 64 KiB)         |

На стороне Rust SDK — смарт-указатель, который в wasm остаётся обычным
`i32.load` по «магическому» адресу:

```rust
pub struct GpuPointer<T> {
    encoded_address: u32,
    _marker: core::marker::PhantomData<T>,
}

impl<T> GpuPointer<T> {
    pub fn read(&self) -> T {
        // Адрес вида 0x10020010: space=Uniform, binding=2, offset=16.
        // rustc → валидный MVP wasm (`i32.load`).
        unsafe { core::ptr::read_volatile(self.encoded_address as *const T) }
    }
}
```

В `wasm2gpu` при разборе байткода: вместо слепого load из кучи —
декод адреса (static analysis / constant propagation) и эмит
`OpAccessChain` к переменной с нужным binding, без единой linear memory
на GPU.

### 2.1. Две проблемы из исходной формулировки

**Динамические индексы.** `buf[local_index]` делает адрес
неконстантным. Наивный декод при компиляции невозможен.

- Исходное предложение: на GPU генерировать сдвиги/маски и descriptor
  indexing / runtime arrays.
- Уточнение (см. §3): для индекса **внутри одного** буфера достаточно
  `OpAccessChain(base_var, i)` — demask binding на GPU не нужен.
  Runtime demask — только когда сам binding динамический (bindless).

**Layout / packing.** В Wasm `struct { a: vec3, b: f32 }` плотный
(offset `b` = 12); в uniform/`std140` после `vec3` дыра до 16.
SDK обязан форсировать GPU-layout (`encase`-подобное, `#[repr(align(16))]`
и т.п.), чтобы байты в Wasm совпали с ожиданиями SPIR-V/WGSL.

## 3. Предложения: как должна выглядеть надстройка

Packing `space + binding + offset` в **каждый** `i32` — слабое место
(лимит 64 KiB на storage, хрупкая арифметика, конфликт с LLVM).
Рабочая модель: **binding в provenance указателя** (тип / SSA /
custom section), в битах адреса — в основном offset; bindless — редкий
отдельный случай.

### 3.1. Три слоя

```text
Shader Rust (no_std SDK: Uniform/Storage/GpuPtr, GPU-layout)
        │  rustc → wasm32-unknown-unknown
        ▼
WASM MVP-модуль
  • i32.load/store от typed base + offset
  • импорты: texture_sample, barrier, …
  • без общего heap-аллокатора; скаляры — wasm locals
        │  wasm2gpu (офлайн)
        ▼
Abstract interpretation адресов по CFG
  → Naga IR / SPIR-V / WGSL
```

1. **SDK (`gpu_abi`, `no_std`)** — единственный API автора шейдера.
   Типы вроде `Uniform<G, B, T>`, `Storage<G, B, T>`,
   `GpuPtr<T, Space>`. Layout принудительно GPU. Текстуры/семплеры —
   не load из памяти, а методы → wasm-импорты.
2. **Контракт с rustc** — без аллокатора и сырого `&[u8]` по куче.
   Арифметика указателей только через SDK (`ptr.offset(i)`), без
   переноса в high-биты. Иначе LLVM считает адрес плоским и ломает
   инварианты; одного `read_volatile` мало.
3. **`wasm2gpu`** — офлайн-транслятор, не «рантайм на GPU». CFG +
   решётка адресов → нативные access chain. Real linear memory, если
   остаётся, — только Private/scratch.

### 3.2. Схема адресации

| Случай | Где binding | Что в `i32`/`u64` | Эмит |
|--------|-------------|-------------------|------|
| Обычный (≈90%) | Тип / SSA provenance (`Storage<0,3,T>`) | offset (± индекс элемента) | `OpAccessChain` к фиксированному global |
| Bindless | Отдельный `DynHandle` | handle + offset | descriptor indexing / runtime array |
| Private | — | обычный offset в scratch memory или wasm local | private var / local |

- Uniform на WebGPU и так ≤ 64 KiB → 16-бит offset ок.
- Storage часто ≫ 64 KiB → **нарезка 4+12+16 убивает storage**; offset
  для storage должен быть шире, а binding — не из тех же бит.
- Space в high-битах годится как assert/debug (Private vs Buffer), но
  не как единственный способ узнать `@binding`.

### 3.3. Динамический индекс

`buf[i]` → `OpAccessChain(base_var, i)`, где `base_var` уже привязан к
конкретному ресурсу. Битовая маска на GPU нужна только при динамическом
выборе binding (bindless). На WebGPU/wgpu портативного bindless «как
DX12» нет — не обещать как default.

### 3.4. Дополнительные риски (кроме двух исходных)

1. **rustc/LLVM** — слияние load’ов, перенос при `add` в high-биты,
   spills в linear memory. Лечение: узкий SDK, запрет сырых `*const T`
   вне него.
2. **Текстуры/семплеры** — не `i32.load`; только импорты / `call`.
3. **Abstract interpretation по CFG** — через `phi`/`select` provenance
   должен стыковаться; Unknown → reject или тяжёлый runtime decode.
4. **Модель wgpu** — динамический выбор binding не портативен; закладывать
   только как opt-in при явной поддержке.

## 4. Минимальный ABI (эскиз)

Не спецификация реализации — ориентир для прототипа.

```rust
// Псевдо-API SDK
pub struct Uniform<const G: u32, const B: u32, T>(GpuPtr<T, UniformSpace>);
pub struct Storage<const G: u32, const B: u32, T>(GpuPtr<T, StorageSpace>);

impl<T, S> GpuPtr<T, S> {
    pub fn offset(self, i: u32) -> Self { /* checked, не трогает space */ }
    pub fn read(self) -> T { /* → i32.load / lowering */ }
    pub fn write(self, v: T) { /* storage only */ }
}

// Текстуры — не память
impl Texture2d {
    pub fn sample(&self, s: Sampler, uv: Vec2) -> Vec4 {
        // → call $texture_sample_2d
        unimplemented!()
    }
}
```

Транслятор читает:

- таблицу импортов → GPU-опкоды;
- custom section или имена глобалов → карта `(G,B) → переменная`;
- решётку `AddrValue = ConstOffset(base) | Dyn(base, index, stride) | Unknown`.

Критерий приёмки toy-прототипа: один vertex + один fragment с
`Uniform` + `texture_sample`, round-trip в валидный WGSL/SPIR-V через
naga, без descriptor indexing.

Срез после toy (тот же транслятор, не новый ABI): `storage_read` /
`storage_write` читают и пишут элемент массива. Элемент — скаляр
(`f32`, `u32`, `i32`), `vec2` / `vec3` / `vec4<f32>` или простая
структура из этих полей. Индекс — отдельный операнд (`i32.const` или
runtime `i32`), group и binding — только `i32.const`. Динамический
индекс становится access chain к уже выбранному global. Динамический
group/binding по-прежнему ошибка (`DynamicBinding`): это bindless, он
вне default. Один и тот же `(group, binding)` как два разных типа
storage — ошибка.

Структуры объявляются в custom section `ornis_gpu.types` (magic
`OGT1`, затем число структур и теги полей). Идентификатор структуры —
четвёртый immediate импорта `storage_read_struct` /
`storage_write_struct`. Lowerer кладёт поля через `place_fields(...,
uniform: true)` и `storage_stride` с выравниванием 16, так что
смещения совпадают с хостовым `gpu_struct!` (`vec3` затем `f32`:
смещения 0 и 12, stride 16; два `vec3` — через 16). Вложенные
структуры, незнакомый тег поля и ширина вне 1..=8 полей отвергаются.

Uniform использует ту же раскладку и ту же custom section. Импорты:
`uniform_read_struct` (group, binding, struct id) читает структуру
целиком; `uniform_field_*` читает одно поле, индекс поля — `i32.const`
(`AccessIndex`); динамический индекс поля — ошибка `uniform field
index must be a constant`. `uniform_read_*` и `uniform_index_struct`
читают элемент фиксированного массива: длина — `i32.const`, индекс
элемента может быть runtime и становится `Access`. Длина 0,
неконстантная длина и binding больше 64 KiB отвергаются. Stride
элемента — `uniform_stride` (`f32` — 16 байт, не storage stride 4).
WGSL-писатель naga не сохраняет `@align`, `@size` и `@stride`. Скаляр
и `vec2` дополняются lane-структурой до 16 байт: значение остаётся в
поле 0, байт 0 элемента совпадает с хостовым `index(i, UNIFORM_STRIDE)`.
Элемент-структура получает хвост из скалярных `u32` внутри самой
структуры, пока записанный размер не равен span хоста. У
`struct { f32 }`, `struct { f32, f32 }` и `struct { vec2 }` этот span
и stride равны 16, как у `gpu_struct!`. `vec3` уже выровнен на 16;
хвост дописывается, когда конец последнего поля короче span. Глобал
uniform — структура (WGSL не
принимает голый массив в этом адресном пространстве); массив — её
единственное поле со смещением 0. Один `(group, binding)` как два
uniform-типа или сразу как uniform и storage — ошибка. `br_table`
опускается в WGSL `switch` (`case` / `default`, без fall-through).
Селектор 0, 1, … — это метки `br_table` по порядку, последняя метка —
`default`. `br_table` с глубиной 0 в `loop`, который опущен как naga `loop`, —
`continue`, в том числе если тело цикла перед этим делает `i32.add`.
`br_table` с глубиной 0 в область, которую транслятор не обернул в naga
`loop`, — ошибка `operator is outside the toy subset`. `block` и `loop`
оборачиваются; statement-`if` без такой обёртки — нет. `return` из
функции без результатов — naga
`return`. `return` со значением и `br` на неявный блок функции
по-прежнему ошибка `branch to the function is not lowered`.

Прямой `call` функции модуля — отдельная функция naga, не инлайн.
Константный `i32` (group и binding) не становится параметром: каллея
читает его как immediate, поэтому provenance остаётся константой внутри
специализации. Ключ кэша — функция и полный набор этих констант (до
`MAX_PARAMS`, без отбрасывания хвоста): тот же ключ не опускается
второй раз, другой ключ — другая функция. Динамический индекс и любой
`f32`, включая `f32.const`, — аргументы, а не часть ключа, поэтому два
вызова с разными float делят одну функцию. Константный результат
кладётся на стек вызывающего как immediate (им может быть binding
следующего импорта); динамический результат — значение вызова, а
несколько таких результатов — анонимная структура. Абстрактная
интерпретация запоминает результат по (функция, абстрактные аргументы)
и не обходит тело повторно. Потолок — 64 специализации на модуль и
8192 уникальных входа (функция, абстрактные аргументы) на точку входа.
Инструкции тела и итерации цикла этот счётчик не тратят: фрагмент без
вызовов и цикл с большим телом проходят. Сверх потолка — ошибка
`specialization budget exceeded`. Рекурсия, в том числе взаимная, —
ошибка `recursive call is not lowered`. `call_indirect` и секции
table/element — `indirect call is not lowered`.

Экспорт `cs_main` необязателен: модуль только с `vs_main` и `fs_main`
по-прежнему опускается. Сигнатура compute — ровно один `i32` и ноль
результатов. Аргумент naga — `vec3<u32>` с builtin
`global_invocation_id`; wasm-параметр 0 читает компоненту `x`.
`@workgroup_size` — 4, 1, 1. Запись в storage — тот же
`storage_write_*`, что у фрагмента.

Control flow шире прямой линии `vs_main` / `fs_main`. Парсер строит
CFG из Wasm structured control flow (`block` / `loop` / `if` / `br` /
`br_if` / `br_table`). Абстрактная интерпретация ведёт provenance (group, binding)
и offset/индекс через ветки; на join одинаковые константы остаются
константой, разные константы provenance становятся phi. Если этот phi
затем читается как group или binding — отказ (`ambiguous provenance
at a join`). `i32.add` / `i32.eqz` сохраняют флаг joined, поэтому
`(select c 0 1) + 0` в роли binding — та же ошибка, а не
`DynamicBinding`. Phi индекса разрешён и опускается в WGSL `select`
либо в присваивание внутри `if`. Непараметрические локалы в анализе и
в lowering начинаются с нуля, как в Wasm.

Цикл с динамическим индексом остаётся `loop`. `br` глубже одного
уровня несётся флагами `break` / `continue`, потому что у WGSL нет
многоуровневого `br`. После вложенного `block` флаги проверяются у
ближайшего naga-`loop`, в том числе когда этот `block` стоит внутри
statement-`if`: хвост ветки после `br` (например `storage_write`) не
выполняется. `br` / `br_if` на неявный блок функции (это `return`) не
опускается: ошибка `branch to the function is not lowered`. Явный
`return` без результатов опускается. `if` без
результатов и без побочных эффектов выбрасывается целиком.

Срез указателей (2026-10-29). Импорт `storage_base(group, binding)`
возвращает base-тег пары: анализ несёт `Ptr(group, binding, offset)`
через арифметику (`i32.add` / `i32.sub`, константы складываются) и
через локалы (shadow-слот помнит provenance). `i32.load`, `f32.load`,
`i32.store` и `f32.store` с immediate-offset принимают только такой
адрес: константное смещение проверяется на кратность 4 байтам
(`UnsupportedType`), индекс элемента — `bytes >> 2` (константа
сворачивается в `array[i]`), динамический сдвигает на 2. Первое
применение биндинга выбирает элемент (`u32` для `i32.*`, `f32` для
`f32.*`), дальше тот же контракт «два разных типа — ошибка», что у
`storage_read`. Не-указательный адрес — `LinearMemory`. Join двух
указателей с разными парами — `AmbiguousProvenance`; с одинаковыми —
offset phi. Через границу функции указатели v1 не идут: аргумент и
результат — отказ. Для реального кода добавлены `i32.shl` и
`i32.shr_u` (rustc усиливает умножение на сдвиг). SDK: `GpuRef::addr`,
`GpuRef::load`, `GpuRef::store` (только `wasm32`, скаляр 4 байт) —
`unsafe` остаётся в `gpu_abi::copy`.

## 5. Отношение к текущему пути ornis

Текущий принятый путь: Rust declarations → naga IR, тела → Shader IR →
WGSL, naga как валидатор (`rust-to-wgsl-architecture.md`). Это не
transpiler из исполняемого Rust и не Wasm.

Сегментированная Wasm-память — **другой фронтенд**: «настоящий rustc в
телах» ценой своего ABI и транслятора. Имеет смысл как research / этап 2,
если DSL упрётся в язык (generics, сложный control flow).

Практически ближе и дешевле сейчас: усиливать DSL + Naga и явные
`Context` / bindings. Если идти в Wasm-путь — начинать с typed
provenance + offset-only pointers + imports для texture, а не с нарезки
32 бит на space/id/offset.

## 6. Вердикт

| Вариант | Оценка |
|---------|--------|
| Packing space+binding+offset в каждый i32 | Идея-катализатор; как default ABI — нет (storage offset, LLVM, wgpu bindless) |
| Provenance в типе/SSA, offset в указателе, texture-импорты | Рабочая форма надстройки для прототипа |
| Замена текущего DSL-пути | Не сейчас; research note до явного решения |

Сделано как research-прототип, не как замена DSL: toy ABI (§4),
storage read/write (скаляры, векторы, простые структуры, индекс
элемента — access chain) и CFG с абстрактной интерпретацией
provenance. Bindless по-прежнему не default.

Предпочтительная форма SDK — `GpuRef` + явный `read` (§7).

## 7. Safe Abstractions: с сырого unsafe на type-driven SDK

Статус дополнения: 2026-10-25. Уточняет §3–4: как именно должен
выглядеть SDK, чтобы provenance и layout держались в Safe Rust, а
`unsafe` был сжат в один модуль `gpu_abi`.

### 7.1. Зачем сдвиг

Когда SDK строится на сырых `*const T` / `*mut T` и «магических»
адресах, для LLVM это классическая плоская CPU-память: TBAA, слияние и
перестановка load/store легко ломают инварианты шейдера. Перенос модели
на **безопасные абстракции** (typed descriptors / ref-tokens) решает
три задачи сразу:

1. **Provenance** — `const GROUP` / `const BINDING` протаскиваются через
   `.field()` / `.index()` и не «испаряются» из типа.
2. **Предсказуемость LLVM** — нет пользовательских pointer casts и
   произвольной арифметики; оптимизатор не видит «обычную» кучу.
3. **Layout в compile-time** — маркер вроде `GpuSafeLayout` + derive
   запрещают передать в Uniform/Storage тип без std140/std430.

Это не альтернатива сегментированной памяти, а **форма SDK** поверх
модели «binding в типе, offset в значении» (§3.2).

### 7.2. Zero-cost токены вместо указателей

Конечный автор шейдера не трогает `unsafe`. SDK отдаёт структуры,
которые в wasm становятся простым `i32` (offset), а в Rust несут
жёсткие generic-ограничения:

```rust
/// Привязка к буферу зашита в const-generics. В wasm — только offset.
pub struct GpuRef<'a, const GROUP: u32, const BINDING: u32, T: ?Sized> {
    offset: u32,
    _marker: core::marker::PhantomData<&'a T>,
}

impl<'a, const G: u32, const B: u32, T> GpuRef<'a, G, B, T> {
    /// Доступ к полю: новый типизированный токен, без чтения GPU-памяти.
    /// Offsets — через `offset_of!` / derive, не через пользовательские
    /// замыкания на `*const T` (хрупкий `f(null)` не использовать).
    pub fn field<U>(&self, field_offset: u32) -> GpuRef<'a, G, B, U> {
        GpuRef {
            offset: self.offset + field_offset,
            _marker: core::marker::PhantomData,
        }
    }
}
```

`field` / `index` на этапе вызова не делают load: только считают offset.
Фактическое чтение — отдельный I/O-барьер (§7.3).

### 7.3. Явный I/O вместо `Deref`

В шейдерах нельзя отдавать `&T` через `Deref`: обычная ссылка Rust
обязана указывать на объект в linear memory CPU; GPU-буфер так не
работает. Чтение/запись — явные методы:

```rust
impl<'a, const G: u32, const B: u32, T: Copy> GpuRef<'a, G, B, T> {
    pub fn read(&self) -> T {
        // Непрозрачный импорт / intrinsic: wasm2gpu узнаёт вызов в SSA
        // и не разворачивает его в «обычный» i32.load по куче до GPU-pass.
        unsafe { gpu_abi::raw_read::<G, B, T>(self.offset) }
    }
}
```

Контракт с транслятором: `raw_read` / `raw_write` — импорты или
жёстко узнаваемые символы, не inline-load, который LLVM/TBAA могут
переставить как CPU-память.

### 7.4. Три плюса для `wasm2gpu`

| Плюс | Механика |
|------|----------|
| Неразрывный provenance | `G`/`B` в типе на всей цепочке методов → binding из типа/SSA, не demask адреса |
| Меньше фокусов LLVM | Нет кастов и нелегальной арифметики снаружи `gpu_abi` |
| Layout compile-time | `T: GpuSafeLayout` (+ макрос padding) — неверный тип не собирается |

### 7.5. Что Safe API не снимает

- `unsafe` остаётся **внутри** `gpu_abi`; обход SDK снова открывает TBAA.
- Рамки GPU (нет кучи, нет `dyn`, текстуры ≠ load) те же, что в §3.4.
- Тело `raw_read` должно оставаться непрозрачным для CPU-семантики до
  GPU-lowering.

### 7.6. Связь с вердиктом

Рабочий прототип SDK уже в этом стиле: `GpuRef<G, B, T>` плюс явные
`read` / `write` / `field` / `index` и `GpuSafeLayout`. Токены binding'ов
нулевой стоимости, сырые импорты остаются под ними, `unsafe` только в
`gpu_abi` (`copy.rs`). Packing space/id/offset в каждый `i32` (§2)
остаётся отклонённым default ABI.

### 7.7. Отказ лифтера: ворота должны свернуться при компиляции

Проверки token-строительства (`stride % ALIGN`, `field_offset % ALIGN`,
итоговый `offset % ALIGN`) под `wasm32` остаются в госте ветками к
`contract_breach`. Если такая ветка не сворачивается при компиляции
гостя — stride или смещение не являются константными для LLVM — в
reachable-граф лифтера вместе с ней попадает panic-механика ядра
(`fmt`, `i64`, `global.get` стек-пойнтера), и лифт экспорта отказывает:
`compute reaches an op the lifter will not keep`.

Практическое следствие для авторов шейдерных гостей: токены
строятся только от константных stride/offset (например `index_storage`
с `T::STORAGE_STRIDE`), а каждая новая ворота должна сворачиваться в
`true` на этапе компиляции. Тестовый лифтер
(`wasm2gpu-shader/tests/gpu_e2e/compute_lift.rs`) намеренно отказывает
такой экспорт, а не трактует panic-ветку как шейдерный код; хост при этом
проверяет её рантаймом (`abi_ptr_tests.rs`).
