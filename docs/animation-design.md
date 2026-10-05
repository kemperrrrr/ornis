# Дизайн системы анимации (объектная + скелетная)

> Статус: только дизайн, кода нет. Новый файл, соседние треки работают
> параллельно: glTF-лоадер — отдельный трек, здесь описан только стык.
> Стиль ссылок — `файл:строка` от корня репозитория.

## 0. Что уже есть (точки опоры)

- `crates/render/src/scene.rs:39-46` — `TransformDesc { translation, rotation (x,y,z,w), scale }`.
  Единственный placement, который видит рендер.
- `crates/render/src/scene.rs:50-91` — `MeshDesc` (Sphere/Box/Plane/Cylinder/Custom);
  `as_custom` (`scene.rs:100-108`) — единственный путь нестандартной геометрии.
- `crates/render/src/extraction.rs:332-342` — экстракция читает три лейна
  (`TransformDesc`, `MeshDesc`, `MaterialDesc`); некомплект скипается, стаба нет.
  `model = T·R·S` собирается в `extraction.rs:378-382,429-433`,
  rotation нормализуется (`normalized_rotation`, `extraction.rs:673-681`).
- `crates/render/src/renderer.rs:198-205` — `InstanceData { model_matrix, normal_matrix, material_index }`:
  объектная анимация укладывается в существующий инстансинг без изменений шейдеров.
- `crates/core/src/gameplay.rs:46-52` — `Position(Vec3)` (gameplay-перемещение),
  `Velocity` (`gameplay.rs:31-37`); `physics_push` (`gameplay.rs:268-296`, fixed)
  и `transform_update` (`gameplay.rs:305-351`, frame, сейчас residual=0) пишут
  только `Position`, никогда `TransformDesc`.
- `crates/app/src/lib.rs:202-249` — `BodyToTransformSystem`: единственный писатель
  `TransformDesc` сегодня; читает `RigidBody`, пишет `Position` + `TransformDesc`
  (доступы — `lib.rs:209-215`). Кадр: `GameWorld::frame` (`crates/app/src/game_world.rs:135-138`)
  = `run_frame` + `frame_upload`; вставка сцены — три лейна (`game_world.rs:141-152`).
- Скедулер: `SystemAccess` (`crates/core/src/schedule.rs:168-177`),
  `reads_lane/writes_lane` (`schedule.rs:197-207`), enforcement на границе
  `read_lane/write_lane` (`smart_store.rs:285-320`, проверки `schedule.rs:415-439`).
  Стадии `PreUpdate → Input → Gameplay(fixed) → PostFrame` (`crates/core/src/engine.rs:221-257`),
  порядок `run_frame` (`engine.rs:436-470`); fixed-кап `DEFAULT_MAX_FIXED_STEPS_PER_FRAME=8`
  (`engine.rs:11-15`).
- `crates/core/src/mutation.rs:37-48` — `Mutation::Set` (JSON через реестр);
  `apply_mutations` (`mutation.rs:89-100`) — только между кадрами, не из систем.
  `MutationProducer` (`mutation.rs:178-182`), `MutationBus` (`mutation.rs:191-277`),
  `MutationTick` (`mutation.rs:286-306`) — раз в кадр, `PostFrame`, читает `Time`.
- Residency: `DataResidency { CpuOnly, GpuOnly, Both }` + `ResidencyTracker`
  (`crates/core/src/command_sync.rs:13-88`) — переиспользовать для skinning-буферов.
- Аплоад мешей: `upload_mesh_data` (`crates/render/src/mesh_upload.rs:85-106`),
  `custom_vertices` (`mesh_upload.rs:147-151`), per-frame кеш по `soup_hash`
  (`mesh_upload.rs:159-197`). CPU-скиннинг ляжет на этот же путь.
- Слот лоадера: `PROJECT_REVIEW.md:2184` — `.gltf → MeshLoader` (пока пусто;
  `gltf`/`Skeleton`/`skin` в `crates/` отсутствуют — проверено grep).
- Хранилище: `SmartStore` (`crates/core/src/smart_store.rs:119-123`) —
  hot-лейны (`RwLock<ComponentStore>`, dense `data/entities`, `component_store.rs:36-44`),
  cold-лейны (`smart_store.rs:180-238`), `Pack` (`smart_store.rs:363-384`).

## 1. Объектная анимация

### 1.1. Данные

- `AnimClip` — **холодный** компонент/ресурс (клип меняется редко, сэмплируется часто):
  имя, длительность, `loop: bool`, треки `Vec<AnimTrack>`.
  `AnimTrack { entity: Entity, translation: KeyTrack<Vec3>, rotation: KeyTrack<Quat>, scale: KeyTrack<Vec3> }`.
  Ключи: `(time: f32, value)` отсортированы; интерполяция — lerp для T/S, slerp для R.
  Хранить в cold-лейне (`register_cold`, ср. `smart_store.rs:180-192`) или ресурсе
  `Assets<AnimClip>`; решение — cold-лейн на сущность-плейлист + ресурс-таблица для
  разделяемых клипов (как `materials` dedup в `extraction.rs:580-593`).
- `AnimPlayer` — **горячий** лейн на анимируемой сущности:
  `{ clip: ClipId, time: f32, speed: f32, weight: f32, playing: bool }`.
  `Clone + Send + Sync` (требование `smart_store.rs:114-116,144-149`).

### 1.2. Сэмплирование и куда пишется

- Система `anim_sample`, **только `Stage::PostFrame`** (variable `Time`, не `FixedTime`):
  визуальная поза не должна умножаться на число fixed-сабстепов
  (тот же аргумент, что у `MutationTick`, `mutation.rs:279-286`).
  Читает `Time::elapsed_seconds/delta_seconds` (`engine.rs:23-64`), продвигает
  `player.time += dt * speed`, сэмплирует трек, пишет результат.
- **Куда пишется — решение:**
  - рендеримые сущности (есть `MeshDesc`): писать **`TransformDesc.translation/rotation`**
    напрямую (scale — только если трек его содержит, иначе не трогать, чтобы не
    затирать размер из `EntityDesc`). Причина: экстракция читает только
    `TransformDesc` (`extraction.rs:332-342`); `Position` рендер не видит.
  - зеркало в `Position`: писать тогда и только тогда, когда лейн уже существует
    на сущности (как делает `body_to_transform`, `app/src/lib.rs:231-239`:
    `get_mut` → обновить, иначе `insert`). Не создавать `Position` ради анимации —
    иначе раздуваем лейн и ломаем `position_count`/снапшоты (`gameplay.rs:72-78`).
  - сущности без `MeshDesc` (чистый gameplay-маркер): писать `Position`.
- **fixed vs frame:** объектная анимация — frame-only. Исключение — root motion,
  влияющий на геймплей/физику: он идёт не через `anim_sample`, а через существующий
  fixed-путь (`Velocity` → `RigidBody`, `app/src/lib.rs:150-198`): сэмплер пишет
  displacement в `Velocity`, интеграция остаётся в `physics_push`/солвере.
  Визуальный остаток (качание без перемещения) — в `TransformDesc` поверх.
- Порядок: после `body_to_transform`. Сущность с `RigidBody` — физика авторитетна:
  `anim_sample` пропускает такие сущности (проверка наличия `RigidBody` в лейне),
  кроме root-motion-канала (см. выше). Явное ребро
  `order_before("body_to_transform", "anim_sample")` для детерминизма уровней
  (механика рёбер — `schedule.rs:506-539`).

### 1.3. Доступы `anim_sample`

```text
reads:  Time, SmartStore
reads_lane:  AnimPlayer, AnimClip (или ресурс клипов), RigidBody (только наличие)
writes_lane: TransformDesc, Position (условный mirror — декларировать оба,
             писать по наличию лейна; enforcement требует writes_lane для записи,
             schedule.rs:410-439)
```

Конфликты: WaW с `body_to_transform` по `TransformDesc`/`Position` → разные уровни,
порядок — явным ребром. С `audio_listener_sync` (уже упорядочен после
`body_to_transform`, `crates/audio/src/bridge.rs:110`) — добавить ребро
`anim_sample → audio_listener_sync`, чтобы слушатель слышал финальную позу.

## 2. Скелетная анимация

### 2.1. Данные (новые лейны)

- `Skeleton` (hot, на корне скелета): `{ parents: Vec<i32>, inverse_bind: Vec<Mat4>, joint_names: Vec<String> }`.
  Топология неизменяема после загрузки; хранить `parents`/`inverse_bind` в `Arc<...>`
  внутри компонента, чтобы клонирование лейна (`ComponentStore: Clone`) не копировало массивы.
- `JointPose` (hot, на корне): `Vec<Mat4>` — локальные→модельные матрицы суставов,
  выход сэмплера, вход скининга. Длина = числу суставов; кап `MAX_JOINTS=128`
  (uniform/storage-лимит будущего GPU-пути; превышение — ошибка загрузки, не тихий дроп).
- `SkinnedMesh` (hot, на меше): `{ skeleton: Entity, joints: Vec<[u16; 4]>, weights: Vec<[f32; 4]>, positions/normals/uvs/indices (Arc) }`.
  Веса нормализованы при загрузке (`sum=1`, иначе нормировать; нулевая сумма → `(1,0,0,0)` на joint 0).
  Лимит — 4 влияния (glTF-стандарт `JOINTS_0/WEIGHTS_0`).
- `SkelClip` (cold): `{ duration, tracks: Vec<JointTrack> }`,
  `JointTrack { joint: u32, t: KeyTrack<Vec3>, r: KeyTrack<Quat>, s: KeyTrack<Vec3> }`.
- `SkelPlayer` (hot): как `AnimPlayer` + `clip: ClipId`; бленд двух клипов — фаза 3,
  не фаза 1 (поле `weight` зарезервировать сразу).

Отдельный `joints-лейн` на каждый сустав-сущность — **отклонено**: суставы как сущности
раздувают аллокатор и экстракцию (каждый сустав — не рендеримая сущность, но попадает
в итерацию `transforms`, `extraction.rs:351`); плоский `JointPose`-вектор на корне
дешевле и дружит с одним GPU-буфером.

### 2.2. Сэмплер-система `skel_sample` (`PostFrame`)

1. Прочитать `SkelPlayer.time`, сэмплировать каждый `JointTrack` (lerp/slerp).
2. Композиция `local = T·R·S` (та же конвенция, что `Transform::matrix`,
   `render/src/transform.rs:35-39`), затем проход по `parents`:
   `model[j] = model[parent] * local[j]`, корень — `* TransformDesc` сущности.
3. Записать `JointPose` целиком (swap вектора, не поэлементно — один `write_lane`).
4. Финальные skinning-матрицы `joint_matrix[j] = model[j] * inverse_bind[j]` —
   считать в скининг-проходе, не в сэмплере (сэмплер — поза, скининг — вершины).

Доступы: `reads Time, SmartStore; reads_lane: SkelPlayer, SkelClip, Skeleton, TransformDesc;
writes_lane: JointPose`. Уровни: после `anim_sample` (разные лейны — параллельны,
но явное ребро для стабильности mermaid), до скининга.

### 2.3. CPU vs GPU скининг (решение с учётом residency)

- **Фаза 1 — CPU-скининг** (единственный путь до GPU-фазы):
  система `skel_skin_cpu` (`PostFrame`, после `skel_sample`) читает `JointPose` +
  `SkinnedMesh`, считает `v' = Σ w·M[j]·v` (позиции и нормали — `M` для нормалей
  через inverse-transpose 3×3, как `normal_matrix` в `extraction.rs:387`),
  кладёт результат в per-frame `custom_cache`-подобный буфер и публикует через
  существующий `CustomMeshEntry`-путь (`extraction.rs:51-64`): вершины уже
  в мировых координатах, `instance.model_matrix = IDENTITY`.
  Плюс: ноль изменений шейдеров/пайплайнов; минус: upload каждый кадр
  (`create_buffer_init` цена, ср. `mesh_upload.rs:90-99`).
- **Фаза 2 — GPU-скининг:** `joint_matrix` публикуется как storage-буфер
  (один на скелет, `MAX_JOINTS` записей), вершинный шейдер делает
  `Σ w·M[j]·v` для позиций и тангентов; нормали — `Σ w·inverse_transpose(M[j])·n`
  (тот же вырожденный fallback, что CPU). Residency-протокол (`command_sync.rs:13-88`):
  `skel_sample` → `mark_cpu::<JointPose>()`; render-submit при `needs_cpu_to_gpu`
  заливает буфер и `mark_both`; CPU-скининг при `Both` пропускается.
  Пока GPU-пути нет — весь `SkinnedMesh`-лейн `CpuOnly` по умолчанию
  (`ResidencyTracker::get`, `command_sync.rs:62-67`), что честно отражает реальность.
- Выбор пути — per-skeleton флаг, не глобальный: маленькие скелеты могут оставаться
  на CPU (меньше latency одного кадра), толпа — на GPU. Эвристика по умолчанию:
  CPU при `joints ≤ 32 && vertices ≤ 4k`, иначе GPU (когда появится).

Экстракция: `SkinnedMesh`-сущности **не** идут в shared `instances`
(размер нельзя запечь в `model_matrix`, ср. `extraction.rs:401-441`) —
только в `custom_meshes`-лейн с флагом `skinned: bool` (расширение `CustomMeshEntry`,
аддитивно). `max_mesh_params` их игнорирует (как `Custom`, `extraction.rs:472-474`).
Неполный скелет (нет `Skeleton`/`JointPose`/`SkelClip`) — скип со счётчиком
в `ExtractionStats` (новый `skipped_bad_skin`, рядом с `skipped_bad_custom`,
`extraction.rs:290-311`), без стабов.

## 3. Интеграция с schedule

| Система | Стадия | reads | reads_lane | writes_lane |
|---|---|---|---|---|
| `anim_sample` | PostFrame | `Time`, `SmartStore` | `AnimPlayer`, клип, `RigidBody` (наличие) | `TransformDesc`, `Position` |
| `skel_sample` | PostFrame | `Time`, `SmartStore` | `SkelPlayer`, `SkelClip`, `Skeleton`, `TransformDesc` | `JointPose` |
| `skel_skin_cpu` | PostFrame | `SmartStore` | `JointPose`, `SkinnedMesh` | (публикация в frame-payload, не лейн; либо `writes` на `Mutex<SkinUpload>`-ресурс) |

- Все три — `engine.schedule_mut()` / `add_stage_system(PostFrame)` (ср. `MutationTick`,
  `mutation.rs:340-343`). Никаких fixed-систем: поза — variable.
- Рёбра: `body_to_transform → anim_sample → skel_sample → skel_skin_cpu →
  audio_listener_sync`. Регистрация в этом порядке + `try_order_before` страховка
  (паттерн `gameplay.rs:187-190`, `audio/bridge.rs:110`).
- Уровни выводит планировщик из доступов (`Schedule::levels`, `schedule.rs:569-574`);
  конфликты только по `TransformDesc`/`Position` (anim vs body) — skel-системы
  по своим лейнам параллельны со всем остальным.
- Детерминизм: сэмплирование — чистая функция `(clip, time)`; `time` из
  `Time::elapsed_seconds` (монотонно, `engine.rs:55-64`). Параллелизм внутри системы —
  только через `#[smart_pipeline]`-паттерн с `capture_access_frame`
  (`schedule.rs:348-381`), ручной `rayon::scope` без capture запрещён.
- GameWorld: `frame` без изменений — экстракция после `run_frame`
  автоматически видит финальные позы. Регистрация — через
  `Engine::add_stage_system(Stage::PostFrame, …)` (словарь `GameStage`
  удалён 2026-09-22, остался только `CoreStage`).

## 4. Стык с glTF-треком (что должен дать лоадер)

Лоадер (слот `PROJECT_REVIEW.md:2184`) — единственный, кто парсит glTF; анимация
потребляет только готовые лейны. Контракт загрузки:

1. **Меш:** `positions`/`indices` (+ `NORMAL`/`TEXCOORD_0` если есть, иначе как
   `custom_mesh_data`: нормали пересчитать, uv — box-проекция, `mesh_upload.rs:108-134`)
   → `SkinnedMesh` (+ `MeshDesc::Custom`-совместимый soup для CPU-фолбэка).
2. **Скелет:** `skins[0]` → `Skeleton { parents (из `nodes` иерархии),
   inverse_bind (из `inverseBindMatrices`, при отсутствии — identity),
   joint_names }`; `skeleton` — сущность-корень (обычно glTF-`node` со `skin`).
3. **Веса:** `JOINTS_0` (u8/u16 → u16) + `WEIGHTS_0` → `joints`/`weights`,
   нормализовать; >4 влияний — оставить топ-4 с перенормировкой + warn в stderr
   (паттерн `RenderLights::from_scene`, `extraction.rs:153-168`: лог раз при загрузке,
   не per-frame).
4. **Клипы:** `animations[]` → `SkelClip` каждый: `input` (times) + `output`
   (translation/quaternion/scale) по `channel.target.path`; интерполяция:
   фаза 1 — `LINEAR` (+ `STEP` как вырожденный lerp), `CUBICSPLINE` — фаза 3.
   Объектные треки (анимация `node` без `skin`) → `AnimClip`/`AnimTrack`
   с привязкой к сущности лоадера.
5. **Сущности:** лоадер создаёт: корень (`TransformDesc` из node-TRS + `Skeleton` +
   `JointPose::identity` + `SkelPlayer` на клип 0 paused) и меш-сущности
   (`SkinnedMesh { skeleton: root }`). Маппинг `node_index → Entity/joint`
   лоадер держит у себя, наружу не торчит.
6. **Ошибки:** нет `inverseBindMatrices` → identity; нет `skin` но есть анимация
   нод → только `AnimClip`; меш без `JOINTS_0` → обычный `Custom` (скип скининга,
   не ошибка). Невалидный soup — та же честность, что `UploadError`
   (`mesh_upload.rs:17-25`): скип сущности, счётчик, без стабов.

## 5. Фазы внедрения (оценка)

- **Фаза A (S): объектная анимация.** Лейны `AnimClip`(cold)+`AnimPlayer`(hot),
  `anim_sample` (PostFrame, доступы §1.3), рёбра `body_to_transform → anim_sample →
   audio_listener_sync` в плане `CoreStage::PostFrame`. Критерий: сфера ездит по ключам,
  физическая сущность игнорируется аниматором, экстракция без новых счётчиков.
- **Фаза B (M): скелет + CPU-скининг.** Лейны `Skeleton`/`JointPose`/`SkinnedMesh`/
  `SkelClip`/`SkelPlayer`, `skel_sample` + `skel_skin_cpu`, `CustomMeshEntry.skinned`,
  `ExtractionStats.skipped_bad_skin`, кап `MAX_JOINTS=128`. Критерий: glTF-куб
  с 2 костями гнётся, bad-skin скипается со счётчиком.
- **Фаза C (M): glTF-стык.** Лоадер реализует контракт §4 (в своём треке);
  здесь — только приёмка: скелет/веса/клипы из файла доезжают до фаз A/B.
  Критерий: один настоящий `.gltf` (робот/куб) играет `LINEAR`-клип.
- **Фаза D (L): GPU-скининг.** Joint-storage-буфер, вершинный шейдер
  `Σ w·M·v`, residency-протокол §2.3, per-skeleton выбор CPU/GPU, удаление
  per-frame upload с hot-пути. Критерий: паритет картинки CPU vs GPU
  (tolerance как `docs/WASM_PIXEL_E2E.md:44-48`: mean ≤ 2/255), upload/кадр → 0
  при статичной позе (`Both`).
- **Фаза E (M, опционально): бленд/маски.** `weight` в плеерах, кроссфейд N→M кадров,
  верх/низ маски; `CUBICSPLINE`; иерархия `Parent` + `GlobalTransform` для
  не-скелетных ригов. Только после D.

## 6. Осознанно НЕ делаем

- Суставы как ECS-сущности (см. §2.1 — раздувание итерации экстракции).
- Запись анимации в fixed-расписание (кроме root-motion через `Velocity`;
  поза — variable, иначе умножение на сабстепы).
- Анимационные выходы через `MutationBus`: JSON-кодек (`mutation.rs:10-14,89-100`)
  — тул/скрипт-шов, не hot-loop; сэмплер пишет лейны напрямую типизированно.
- Свой `Transform` (`render/src/transform.rs:7-14`) как лейн: рендер читает
  `TransformDesc`, второй placement-источник = рассинхрон.
- Бленд-дерево/IK/ретаргет в фазах A–D; морфы (blend shapes) — только после E.
- Изменение `InstanceData`/шейдеров ради фаз A–C (объектная ложится в `model_matrix`,
  скиннинг — в `custom_meshes`).
- Тихие фолбэки: незагрузившийся скелет/вес/клип = скип + счётчик, никогда сфера-заглушка
  (честность как `skipped_bad_custom`, `extraction.rs:368-376`).
