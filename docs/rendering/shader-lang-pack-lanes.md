# Pack-лейны: `#[pack(gpu)]` + `#[gpu_pipeline(pack(T))]` (RAT-50, feature 2)

Задача: https://github.com/kemperrrrr/ornis/issues/110 (часть 2 из 3).

## Проблема

`#[derive(Pack)]` хранит бандл компонентов в SoA-лейнах `SmartStore`,
но GPU-путь этого не видел: каждая пара `(лейн, буфер)` описывалась
вручную в `#[gpu_pipeline(storage(...))]`, а колонки сериализовались
руками. Расхождение имени, номера биндинга или порядка сущностей между
Rust-лейном и WGSL-массивом — это не ошибка компиляции, а тихое
несоответствие данных на GPU.

Feature 1 (RAT-49, `docs/rendering/shader-lang-entity-gather.md`) дал
`#[kernel]` entity-gathers, но `derive(Pack)` не интегрирован —
сериализация колонок в лейн-буферы оставалась ручной.

## Модель

Три согласованные части:

1. **`#[pack(gpu)]` на бандле** — derive видит определение структуры,
   поэтому сам описывает каждую колонку: `ornis_core::PackGpu` даёт
   `pack_gpu_lanes()` (по дескриптору на поле, в порядке объявления) и
   `pack_gpu_upload(&SmartStore)` — сериализацию колонок в байты.
   Без атрибута бандл остаётся чисто CPU: `PackGpu` не реализуется
   (trybuild `pack_gpu_requires_opt_in`).

2. **`#[gpu_pipeline(pack(T))]`** — опция макроса: каждый перечисленный
   бандл добавляет по одному `var<storage, read>`-лейну на поле. Макрос
   не видит чужой derive-expansion (как и `helpers(...)`), поэтому пути
   к бандлам резолвятся в модуле-вызывающем: генерируются loader-функции
   рядом с модулем пайплайна, внутри модуля — только `super::`-вызовы.

3. **`create_pack_buffers(device, &store)`** — генерируемый хелпер:
   вызывает `pack_gpu_upload` и создаёт по storage-буферу на лейн
   (label = имя поля), в порядке `pack(...)`. `None`, если стор
   бандлов не несёт.

## Контракт раскладки

- **Имена**: WGSL-имя лейна — имя поля бандла. Элемент —
  pass-through: `f32`/`u32`/`i32` — сами себе, всё остальное
  зеркалится по листовому идентификатору (`glam::Vec3` → `Vec3`).
  Регитри-маппинг (`Vec3` → `vec3<f32>`) сознательно запрещён: storage-
  массив `vec3<f32>` имеет stride 16, у glam `Vec3` — 12; угаданный
  spelling соврал бы о раскладке. Написание зеркала проверяет naga
  (валидация модуля); соответствие `size_of::<T>()` и WGSL-stride
  закреплено `size_of`-ассертами `#[derive(WgslStruct)]`, а у ручной
  WGSL-структуры без `WgslStruct` это соответствие — контракт автора.
  Зарезервированные в WGSL имена полей (`ref`, `type`, `loop`, `filter`,
  …) и префикс `__` (зарезервирован самой грамматикой) — compile error
  на спане поля, а не naga-падение: имя поля и есть имя storage-
  переменной (trybuild `pack_gpu_rejects_reserved_field_name`,
  `pack_gpu_rejects_dunder_field`). `bool`/`BVecN` и вложенный
  `Option` — compile error (trybuild `pack_gpu_rejects_bool_lane`,
  `…_nested_option`).
- **Номера биндингов**: лейны идут после явных
  `storage`/`uniform`/`texture`/`sampler`-опций: `base = число явных
  биндингов`, лейн `i` — биндинг `base + i`. `pack(...)` **обязан**
  стоять после явных опций (compile error, trybuild
  `gpu_pipeline_rejects_pack_before_storage`) — иначе база не
  константа на этапе макроса.
- **Индексация**: общий индекс-пространство всех колонок одного бандла —
  восходящий порядок id сущностей первого **не-sparse**-лейна (первого
  поля без `Option`): ведущее спарс-поле с `None` не вправе выкидывать
  сущность из остальных колонок (тест
  `sparse_first_field_keeps_dense_entities_in_every_column`). Если все
  поля спарс-овые (dense-якоря нет), индекс-пространство —
  отсортированное объединение всех лейнов (тест
  `all_sparse_bundle_uses_sorted_union_index_space`: лейны пересекаются
  по сущности и вставляются вне порядка id, так что удаление
  сортировки или дедупликации в генерации ломает тест). CPU-правило
  `Pack::pack_entities` («первый лейн») не менялось — при спарс-первом
  поле порядки сознательно различаются. Каждый лейн ищет сущность по
  id; после swap-remove dense-порядок «дышит», но колонки не расходятся
  (тест `upload_survives_dense_order_churn`).
- **Спарс-лейны**: `Option<T>`-поле — спарс-колонка; отсутствующая у
  сущности запись заливается элементом-нулём (`Pod ⇒ Zeroable`).
- **Пустой стор**: `len == 0` — не ошибка: каждая колонка несёт один
  элемент-ноль, иначе wgpu отвергнет нулевой буферный биндинг; пустая
  сцена при старте должна биндиться (тест
  `empty_store_lane_buffers_bind_without_validation_error`). Внимание
  ядрам: при пустом сторе WGSL `arrayLength` вернёт 1 (построчный
  паддинг), поэтому перебирать нужно по `len` (или uniform-длине), а
  не по `arrayLength` — иначе ядро обработает фантомную нулевую
  сущность.
- **Дубликаты**: имена лейнов не должны пересекаться между бандлами и
  с явными биндингами/структурами (WGSL module-scope — одно имя).
  Макрос не видит чужих имён, поэтому коллизия падает в
  `wgsl_source()` с именем виновника — до создания любого GPU-объекта
  (тест `duplicate_lane_name_panics_at_stitch`).

## Раскладка CPU/GPU-дескрипторов

`ornis_core::PackGpuLane` (runtime, в ядре — core не может зависеть от
shader-lang) и `ornis_shader_lang::PackLaneDesc` (каноническое WGSL-
написание, `decl(group, binding)`) — два типа по необходимости. Их
сшивка закреплена пин-тестом
`pack_lane_decl_matches_shader_lang_descriptor`: сгенерированное
объявление обязано буквально содержать `PackLaneDesc::decl` — дрейф
написаний падает тестом.

## `create_pipeline` и раскладка

`#[gpu_pipeline]` в full-shader compute-режиме раньше не порождал
`create_bind_group_layout`/`create_pipeline` вообще. С `pack(...)`
генерируется полный набор: лейаут покрывает явные ресурсы (по IR:
storage/uniform/texture/sampler; `var<private>` слота не занимает) плюс
цикл по лейнам; `create_pipeline` строится против этого лейаута в форме
wgpu 30 (`bind_group_layouts: &[Some(&bgl)]`, `immediate_size: 0`).
Отклонения раскладки (лейн без буфера, длина колонки ≠ числу сущностей)
останавливаются валидацией wgpu, а UB не возникает: буфер создаётся ровно
под байты колонки, robust-access читает выровненный хвост.

`#[gpu_pipeline]` с `vertex`/`fragment` + `pack(...)` — compile error
(лейн-загрузка сегодня только compute).

## Проверка

`crates/macros/tests/pack_gpu_lanes.rs`:

- дескрипторы лейнов (порядок, spelling, спарс-флаг);
- выравнивание колонок: shared-order, churn после swap-remove,
  нулевой плейсхолдер спарс-лейна; спарс-первое поле не теряет
  сущности в других колонках; все-спарс-бандл берёт объединение
  лейнов;
- склейка `wgsl_source()`: явный биндинг → лейны по номерам, naga-
  валидация, пин против `PackLaneDesc::decl`;
- паника дубликата;
- runtime-негативы раскладки (wgpu-ошибка, не UB): bind group без
  лейн-биндингов отвергается валидацией
  (`missing_lane_bindings_are_a_validation_error_not_ub`); слот
  `texture_depth_cube_array` требует view размерности `CubeArray` —
  неверная размерность ловится при создании bind group
  (`cube_array_texture_slot_accepts_a_cube_view`); пустой стор биндится
  без ошибок валидации (`empty_store_lane_buffers_bind_without_validation_error`);
- device-gated (Metal локально; CI `test-physics,smoke` — lavapipe):
  `create_pack_buffers` + `create_pipeline` + dispatch → readback
  явного выхода == CPU-вычисление того же стора.

Отклонения закреплены trybuild (`tests/ui/pack_gpu_*.rs` — opt-in,
`bool`/вложенный `Option`, зарезервированное WGSL-имя поля, префикс
`__`, non-Pod колонка, конфликт `#[derive(AutoPipeline, Pack)]` с одной
`#[pack]`-атрибутом; `gpu_pipeline_rejects_pack_before_storage.rs`).

## Честные границы v1

- Производственного пользователя `derive(Pack)` в рендер-слое ещё нет
  (все `Pack`-бандлы — тесты/бенчи), поэтому `create_pipeline`-обвязка
  рендер-слоя (`crates/render`) не переписана — она приедет с первым
  производственным бандлом. Контракт и макросная сторона закреплены
  тестами уже сейчас.
- `no_wgpu_in_logic` (rustqual) запрещает `wgpu::` в `crates/core/**`
  и `crates/shader-lang/**` — дескрипторы поэтому чистые данные,
  wgpu-типы только в сгенерированном коде потребителя.
- `create_pack_buffers` создаёт буферы с usage `STORAGE`; для
  readback-тестов копирование идёт через отдельный staging буфер.
