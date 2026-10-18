# Сглаживание

Выбор сглаживания разделён между frame plan и renderer: план регистрирует
нужный resolve-пасс, а renderer выполняет соответствующий режим. Для TAA режим
должен быть задан **и** в `RenderFrame3D`, **и** через
`Renderer3D::set_anti_alias`; иначе план не включает jitter-камеру. Native
showcase читает `--aa off|msaa|taa|fxaa|smaa`; браузерный renderer по умолчанию
использует одно-сэмпловый FXAA.

```rust,ignore
let mode = AntiAliasMode::Taa;
renderer.set_anti_alias(mode);
let frame = RenderFrame3D::new_with_aa(
    surface_format,
    size,
    Technique::Hybrid,
    Bloom::Off,
    AaWiring::new(mode),
);
```

`new_with_aa_samples` принимает negotiated sample count для `Msaa`; `Off`
всегда принудительно использует 1×. TAA, FXAA и SMAA также используют 1×
geometry и заменяют MSAA, а не складываются с ним. TAA history создаётся
лениво при первом resolve, поэтому не расходует память в остальных режимах.

## Режимы

- **Off** — 1× без пост-фильтра.
- **MSAA** — существующий negotiated 4× path (fallback на 1×, если adapter не
  поддерживает нужные форматы/resolve).
- **TAA** — камера получает циклический Halton(2,3) jitter; g-buffer пишет
  velocity `(uv_current - uv_previous, previous_ndc_depth, valid)` в отдельную
  RGBA16F attachment. Предыдущий clip transform учитывает transform instance и
  (для GPU skinning) предыдущий joint palette. Previous models/palettes keyed
  by stable ECS entity, so dense-lane or palette-slot reordering cannot borrow
  another object's history; first-seen entities invalidate motion for one frame.
  CPU-pre-skinned geometry and custom draw paths without a retained previous
  vertex stream use `valid = 0`; static custom geometry retains its previous model.
  Resolve перепроецирует history
  по velocity, отбрасывает фон, выход за границу и depth-disocclusion, затем
  ограничивает history YCoCg-диапазоном текущего neighbourhood и смешивает его
  с текущим кадром. History ping-pong сбрасывается при смене режима и resize.
- **FXAA** — одно-проходный лума edge filter с 5 соседними выборками; выбран
  для WASM как более дешёвый пространственный fallback.
- **SMAA** — одно-проходный локальный morphological edge filter с лума-весами;
  он использует те же пять выборок без отдельной edge/search lookup-текстуры,
  поэтому это облегчённый вариант SMAA 1x, а не reference-пайплайн с area/search
  текстурами.

## Стоимость

Цены ниже — число fullscreen выборок, заданное шейдером, без стоимости
composite/lighting/geometry:

| Режим | Fullscreen pass | Выборки на пиксель | Дополнительная память |
|---|---:|---:|---:|
| FXAA | 1 | 5 | intermediate LDR layer |
| SMAA-lite | 1 | 5 | intermediate LDR layer |
| TAA | 1 | 13 по умолчанию; до 29 при максимальном 5×5 clamp (центр, depth, motion, history, neighbourhood) | intermediate LDR + 2×RGBA16F history; RG16F velocity + R16Uint packed previous depth/validity |

MSAA остаётся частью geometry attachments и не запускает AA post-pass. Цифры
выборок — статическая оценка из шейдера, не GPU timing; GPU pass timing следует
снять на CI/целевом adapter.

## Гейты

- CPU-reference tests проверяют Halton offsets, YCoCg bounds, статическую
  checkerboard-конвергенцию, отбрасывание stale history при движении и то, что
  FXAA/SMAA оставляют плоский цвет неизменным.
- Naga валидирует TAA/FXAA/SMAA WGSL; g-buffer/shader tests покрывают motion
  outputs и previous-palette binding.
