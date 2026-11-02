## Этап 1 — аудит GPU-шейдеров и разрывов wasm2gpu

Снимок `master` после #229. Инвентаризация по исходникам `crates/render/src/shaders`, рендер-пайплайнам и `crates/physics/src/gpu`; строковые WGSL-источники тоже включены, а не только макросы.

### Инвентарь точек входа

**Render, vertex/fragment**

- Геометрия: `shaders/gbuffer_generated.rs` — `vs_main` / `fs_main`; отдельная skinned-версия vertex собирается в `skinning.rs`. Forward PBR переиспользует G-buffer vertex (`pbr_generated.rs::fs_main`), есть textured material-вариант (`material_textures.rs::fs_main_textured`). Тени — vertex-only варианты в `renderer/shadows.rs` (обычный и skinned; fragment нет, depth-only).
- Fullscreen/post: `composite_generated.rs` (`vs_main`, `fs_main`), `hdr_composite_generated.rs` (`vs_main`, `fs_main`), `bloom_generated.rs` (`vs_main`, `fs_main`), `fog_generated.rs` (`vs_main`, `fs_main`).
- Deferred lighting: `lighting_generated.rs` — fullscreen `vs_main`, `fs_main` и MSAA-вариант `fs_main_msaa` (экспортируется под WGSL entry `fs_main`).
- AA: `aa_common.rs` задаёт общий vertex; `aa_edge_generated.rs` и `aa_taa_generated.rs` собирают fragment `fs_main` из WGSL-тел.
- AO: `ao_generated.rs` — fullscreen vertex плюс raw SSAO/GTAO fragment и bilateral-blur fragment; генерация меняется по методу и sample count.
- Значительная часть Rust entry/helper-тела уже создаётся `#[stage]`/`#[kernel]`, но источники AA/AO и крупные части lighting/post остаются строковыми WGSL-фрагментами/сборкой. Переход означает смену источника исполнения на wasm2gpu, а не просто замену макроса.

**Compute в render**

- `gpu_culling.rs::cull_instances`: ручной WGSL, compaction + `atomicAdd` indirect args.
- `renderer/clusters/wgsl.rs::assign_lights_to_clusters`: ручной WGSL, атомарные счётчики и storage-массивы.
- `auto_exposure/wgsl.rs::build_histogram` и `adapt_exposure`: ручной WGSL; histogram использует workgroup memory, atomics, барьеры и `textureLoad`.
- `wasm2gpu-shader/src/skinning.rs::skin`: уже wasm2gpu после #229; production GPU-путь и parity-бенч/тест существуют.
- Auto-exposure имеет и wasm2gpu-гостей/харнесс; это существующий прототип, отдельно нужно сверить, какой вариант реально выбран в штатном frame path.

**Остальной GPU-код/генерация**

- `crates/physics/src/gpu/si_batches.rs::contact_solver`, `avbd.rs::avbd_stub_kernel` и `avbd_row_kernel` — три активных `#[gpu_pipeline]` compute-кернела; `avbd.rs` также имеет три `#[wgsl_fn]` helper-а. `contact_math.rs` содержит два `#[kernel]` helper-а.
- `crates/render/src/shaders/math.rs` содержит 26 `#[kernel]` math-функций, переиспользуемых несколькими проходами. `crates/macros` реализует `#[kernel]`, `#[wgsl_fn]`, `#[stage]`, `#[gpu_pipeline]`, resource/layout packing и helper stitching; после миграции production-потребителей остаётся также удалить эти генераторы и связанные зависимости/тесты.
- Основные runtime-точки создания пайплайнов: `renderer/{meshes,gbuffer,forward,shadows,post,deferred,construct}.rs`, `renderer/{aa,ao}/build.rs`, `renderer/clusters/gpu.rs`, `gpu_culling/culler.rs`, `auto_exposure.rs`; важная часть parity-golden тестов уже расположена рядом с shader-модулями.

### Текущие возможности и конкретные блокеры

1. **Модель entry points сейчас игрушечная и фиксирована.** `wasm2gpu::lower` всегда требует `vs_main` и `fs_main`, плюс опциональный единственный `cs_main`; vertex/fragment сигнатуры зашиты как фиксированные количества скалярных lanes (`VS 4→6`, `FS 2→4`). Metadata `#[shader::*]` уже записывает стадии и compute workgroup size, но translator использует её только для размера `cs_main`. Нужны произвольный набор экспортированных entry points, stage-specific ABI, user locations/struct IO и renderer builtins (`vertex_index`, `instance_index`, position/depth, fragment position/front-facing, sample index); compute dispatch builtins помимо текущего invocation-id.x, включая local invocation/workgroup ids.
2. **Ресурсы/текстуры неполны.** ABI сейчас даёт scalar/vector/struct storage, ограниченные uniform-struct/fixed-array обращения и только `texture_sample_2d` с filtering sampler. Целевым проходам нужны `textureLoad`/dimensions, explicit LOD, integer и depth-текстуры, multisampled/depth-array/cube/cube-array, comparison sampler и compare sampling, а также больше форматов/видов доступа и layout-вариантов. Биндлесс-массивы намеренно отвергаются; их необходимость нужно оценивать отдельно по фактическим shader-источникам.
3. **Compute-синхронизация и atomics отсутствуют в Wasm GPU ABI/IR.** Нет workgroup address-space globals, `workgroupBarrier`, atomic типовых операций и соответствующего memory ordering. Это непосредственный блокер для histogram, culling/cluster compaction и части physics. Нужно определить безопасный перенос WGSL execution/memory semantics, а не только добавить названия импортов.
4. **Математический/векторный язык слишком мал.** IR покрывает scalar f32 арифметику/сравнения, ограниченные integer ops и `sqrt`; крупные проходы требуют векторных/матричных операций и WGSL builtins: dot/cross/normalize/length, min/max/clamp/mix/select, exp/log/pow/trig, reflect/refract, derivatives (`dpdx`/`dpdy`/`fwidth`) и связанные edge semantics. Нужна матрица opcode/builtin → потребитель и parity/precision-политика (в частности wasm-vs-WGSL точность sqrt уже задокументирована).
5. **Raster semantics и результат требуют расширения.** Нужны несколько color outputs/locations, integer attachments, discard/alpha test, depth output, interpolation qualifiers, multisample semantics и правильные stage интерфейсы. Иначе сначала блокируются G-buffer, PBR/lighting и temporal/AO passes даже после добавления текстур.
6. **Layout/host integration и тестирование — отдельная работа.** Следует зафиксировать единый ABI для uniform/storage (включая матрицы, массивы и атомики) и генерировать/проверять host bind-group layouts. Для каждого перевода нужны WGSL snapshots, CPU/reference или old-vs-new GPU parity; golden-кадры должны покрывать материалы/тени/MSAA/AO/AA/post, а не только отдельные kernels. Нужны A/B на M1 по проходам; lavapipe CI пригоден для корректности, не для целевого performance-решения.

### Рекомендуемое разбиение следующих этапов

Сначала расширить IR/ABI и snapshots на: произвольные entry points + stage IO/builtins; textureLoad/dimensions и нужные типы текстур; векторно-матричные builtins; затем atomics/workgroup memory/barriers. Каждый набор подтверждать минимальным гостем и naga WGSL snapshots. После этого переносить compute по одному проходу (culling, clusters, AO/bloom/TAA и auto-exposure), затем raster-кластеры (G-buffer/PBR/lighting, shadows, post), сохраняя старый путь до parity и M1 A/B. Physics и полное удаление `ornis-shader-lang`/макросов — финальная волна; физический GPU-код за пределами render включён в аудит, хотя он не перечислен в render-подразделе issue.

Это аудит, не изменение runtime-пути: DSL/WGSL-производство и production frame path не менялись.