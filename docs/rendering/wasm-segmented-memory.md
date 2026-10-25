# Wasm → GPU: сегментированная виртуальная память

Статус: research note (2026-10-25). Не принято как замена текущему
пути Rust → Shader IR → WGSL (`rust-to-wgsl-architecture.md`).
Фиксирует идею «Virtual Memory Space Translation / Software-Managed
Memory Paging» поверх Wasm linear memory и предложения по ABI
надстройки.

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

Следующий шаг при зелёном свете: toy ABI (§4) + минимальный `wasm2gpu`
на один проход с uniform + sample, без bindless.
