# Дизайн системы анимации (объектная + скелетная)

> Статус: фазы A–E в коде (`crates/animation`): веса, кроссфейд и маски слоёв
> — для скелета и для объектных треков (см. §5).
> Стык с glTF-лоадером описан в §4; соседние треки работают параллельно.
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
  `AnimTrack { entity: Entity, rest: JointTrs, translation: KeyTrack<Vec3>, rotation: KeyTrack<Quat>, scale: KeyTrack<Vec3> }`,
  где `rest` — авторский TRS узла (RESET, фаза E).
  Ключи: `(time: f32, value)` отсортированы; интерполяция — lerp для T/S, slerp для R.
  Хранить в cold-лейне (`register_cold`, ср. `smart_store.rs:180-192`) или ресурсе
  `Assets<AnimClip>`; решение — cold-лейн на сущность-плейлист + ресурс-таблица для
  разделяемых клипов (как `materials` dedup в `extraction.rs:580-593`).
- `AnimPlayer` — **горячий** лейн на анимируемой сущности:
  `{ clip: ClipId, time: f32, speed: f32, weight: f32, playing: bool }`; `weight` — фаза E:
  поза `rest + weight·(поза − rest)`.
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
   joint_names, rest (TRS узла скина) }`; `skeleton` — сущность-корень (обычно glTF-`node` со `skin`).
3. **Веса:** `JOINTS_0` (u8/u16 → u16) + `WEIGHTS_0` → `joints`/`weights`,
   нормализовать; >4 влияний — оставить топ-4 с перенормировкой + warn в stderr
   (паттерн `RenderLights::from_scene`, `extraction.rs:153-168`: лог раз при загрузке,
   не per-frame).
4. **Клипы:** `animations[]` → `SkelClip` каждый: `input` (times) + `output`
   (translation/quaternion/scale) по `channel.target.path`; интерполяция:
   фаза 1 — `LINEAR` (+ `STEP` как вырожденный lerp), `CUBICSPLINE` — фаза 3.
   Объектные треки (анимация `node` без `skin`) → `AnimClip`/`AnimTrack`
   с привязкой к сущности лоадера; `AnimTrack::rest` ← TRS узла (RESET, фаза E).
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

  **Статус (скелет, сделано):** модель детерминированная, как в Godot: отсутствующий
  канал читается как покой кости — её локальный TRS из узла скина (как
  rest-поза Skeleton в Godot; `Skeleton::rest`, при импорте из `LoadedSkin.rest`),
  игрок с весом `w` даёт
  `покой + w·(поза − покой)`, без состояния между кадрами. Кроссфейд — `SkelFade`
  на корне: внешний игрок живёт своими часами, прогресс линейный,
  `Animator::play_crossfade`; `play` и `stop` отменяют фейд. Слои — стек
  `SkelLayerStack` на корень (снизу вверх) и холодный `JointMask` по суставам:
  каждый слой лерпит поверх позы ниже с весом `layer_weight·mask[j]`;
  длина маски не совпадает с числом суставов → скип со
  счётчиком. Скелетный покой — identity только для скелетов без авторского rest
  (`Skeleton::new` по умолчанию).

  **Объектные треки (`AnimPlayer`), сделано:** вариант A — явный RESET (T/R/S узла)
  на каждом `AnimTrack::rest`, как RESET в Godot; импорт glTF заполняет его из узла
  (`LoadedAnimTrack::rest`). `AnimPlayer::weight` работает по тому же правилу, что и
  скелет: `rest + w·(поза − rest)`. Кроссфейд объекта — горячий `AnimFade` на
  сущности (внешний `AnimPlayer` — `from`, его часы идут до конца фейда). Канал, которого
  не дал ни один игрок, не пишется (как раньше у объектов). Слои и фейды входят
  типизированно через фасад `anim_wiring` (`set_skel_layers` ставит весь стек
  `SkelLayerStack` сразу — замена, не добавление; `fade_to_clip`).

  **Дальше (сверено по коду 2026-10-19 на `55f1b0a`):**

  1. **Прод-доступность фазы E.** У `Animator::play_crossfade`, `SkelLayer`,
     `JointMask` и `AnimFade` нет ни одного нетестового вызывателя — только
     регистрация лейнов (`crates/app/src/anim_wiring.rs:395-400`). Пока так,
     `qual:api` на `Animator::play_crossfade` остаётся обещанием, а не фактом.
     **Шаги 1–2 сделаны (2026-10-21, #179):** `play_crossfade` и `AnimFade`
     получили прод-вызов (`GameWorld::play_animation` → фасад
     `anim_wiring::play_crossfade`), `qual:api` с `Animator::play_crossfade`
     снят; `SkelLayer`/`JointMask` прод-вызывателя пока не имеют (ждут шага 3).
     **Шаг 3 (2026-10-22, #180):** стек `SkelLayerStack` сделан (п. 3),
     `set_skel_layers` ставит весь стек через типизированный вход фасада;
     прод-вызыватель у слоёв по-прежнему не появился — `set_skel_layers`
     остаётся хост-API с `qual:api`.
  2. **Общий `Animator`-фасад.** `Animator` (`crates/animation/src/animator.rs:105-223`)
     ведёт только скелетный плейлист (`play`/`play_crossfade`/`pause`/`stop`/`set_looping`);
     у объектов — лишь `set_playing`/`rewind_player` (`crates/app/src/anim_wiring.rs:325,343`).
     Нужен `play_crossfade` для `AnimPlayer` и типизированный вход для `SkelLayer`/`AnimFade`,
     чтобы §5 не требовал прямой вставки в лейн. Решение (2026-10-19): фасад в
     `crates/app/src/anim_wiring.rs`, а не расширение `Animator` — см. п. 10.
     **Сделано (2026-10-21, #179):** фасад `anim_wiring::play_crossfade`
     (обе топологии; имена объектных клипов — по `Name` плейлиста),
     `fade_to_clip` (типизированный `AnimFade`), `set_skel_layers`
     (типизированный вход слоёв; с шага 3 — стек `SkelLayerStack`); `GameWorld::play_animation(entity, name,
     duration)` — прод-точка входа (первый нетестовый вызыватель
     `AnimatorMut::play_crossfade`; у самой цепочки внутридеревных
     вызывателей пока нет), резюме/пауза/стоп — `resume_animation`/
     `pause_animation`/`stop_animation`.
  3. **Стек слоёв. ✅ Сделано (2026-10-22, #180).** `SkelLayerStack` — стек
     `SkelLayer` на корень: смешивается снизу вверх, upper-body поверх
     locomotion и любые комбинации из нескольких слоёв; `set_skel_layers`
     ставит весь стек сразу (замена, не добавление), вход типизированный
     через фасад (`anim_wiring`).
  4. **Осознанно отложено** (см. §6): бленд-дерево, IK, ретаргет, анимационный
     state machine, события/уведомления по ключам, root motion (кроме `Velocity`).
     Гейт — сначала пункты 1–3, морфы (#153) — отдельным PR.
  5. **Перф-бэклог горячего пути.** `tick_fades` идёт по `frozen.contains` на
     каждую запись, O(fades × physics bodies) на кадр
     (`crates/animation/src/lib.rs:1891`); скелетный путь даёт ~5–7 пер-кадровых
     `Vec` (`collect_pose_work`/`sample_entity`/`node_poses`), объектный — 2 `Vec`
     на сущность (`track_of`, `side_pose`, `crates/animation/src/lib.rs:672-684`).
     Нужны битовая маска/`HashSet` для frozen и переиспользование буферов.
  6. **Ragdoll — не начато** (в основном физический трек, но шов нужен и со
     стороны анимации; упоминаний в `PLAN.md`/`IDEAS.md`/`docs/` нет). База в
     физике уже есть: `crates/physics/src/articulated.rs` — Featherstone ABA
     (R6, reduced coordinates; `FeatherstoneEngine`, `BaseJoint`,
     `set_joint_motor`/`set_joint_damping`/`set_joint_friction`) и
     maximal-coordinate constraints `crates/physics/src/joint.rs` (G5) для
     сцен с контактами. Чего нет — шва «физика ↔ скелет»: физика не знает
     `Skeleton`/`JointPose`, анимация про физику знает только пропуск
     `RigidBody`-сущностей в `anim_sample` (`crates/app/src/lib.rs:104`).
     Ragdoll = скин + суставы физики + blending «физика ↔ анимация», по
     образцу `SkinningMode::Cpu/Gpu` (см. §2.3).
  7. **Animation tree (бленд-дерево / граф переходов) — отложено осознанно**
     (§6). Реализации нет; как отдельный пункт в планах он не стоял — есть
     только строка «не делаем в фазах A–D». Текущая модель закрывает простые
     случаи: база + стек слоёв (`SkelLayerStack`) + один кроссфейд на корень
     (`SkelFade`). Дерево — произвольный граф нод (blend1d/2d, state machine с
     переходами по условию, синхронизация по меткам). Гейт — фоллоу-апы 1–3
     (прод-доступность, общий `Animator`-фасад, стек слоёв).
  8. **Motion matching — не начато.** Упоминаний в коде, `PLAN.md`, `IDEAS.md`
     и `docs/` нет. Состав: база движений (клипы + метки контактов),
     feature-выборка позы/скоростей, детерминированный выбор кадра по
     дистанции, инерциализация переходов. Опора у фазы E уже есть —
     `SkelClip`/`KeyTrack` как источник поз, а детерминированная модель бленда
     (без состояния между кадрами, §5) как раз нужна для повторяемого выбора.
  9. **Distance matching / motion warping — не начато.** Упоминаний нет.
     Distance matching — привязка фазы цикла не только ко времени, но и к
     пройденной дистанции/моменту контакта (лечит «скольжение ног» при разной
     скорости); motion warping — деформация позы под целевую точку или
     траекторию (шаг на платформу, прыжок). Оба ложатся поверх сэмплера
     (доступ к позам по времени плюс коррекция), то есть — после слоёв и
     общего API, а warping ещё и после IK (§6).
  10. **Шаги разблокировки гейта (порядок; решение 2026-10-19).**
     - **Шаг 1 — фасад в `crates/app/src/anim_wiring.rs`** ✅ (2026-10-21,
       #179), рядом с
       `set_playing`/`rewind_player` (`:325,343`): `play_crossfade(store,
       entity, name, duration)` плюс типизированный вход для `SkelLayer`/`AnimFade`.
       Скелетная ветка делегирует в `AnimatorMut::play_crossfade` через
       `try_animator` (одна реализация фейда), объектная пишет `AnimFade`
       (и `AnimPlayer`) прямо на сущности. Топологии разные: каталог на корне
       с `targets: Vec<Entity>` (`crates/animation/src/animator.rs:26-30`)
       против плеера на самой анимируемой сущности, — поэтому один тип не
       объединяем, а ставим драйвер поверх каталога, как Godot `AnimationTree`
       над `AnimationPlayer` (общий базовый `AnimationMixer`) или Bevy
       `AnimationTransitions` отдельным компонентом от `AnimationPlayer`.
       Вход остаётся `anim_wiring`-ным и потому, что знание «обоих видов»
       уже там (`set_playing` пробует `SkelPlayer`, затем `AnimPlayer`).
     - **Шаг 2 — первый прод-вызывающий:** ✅ (2026-10-21, #179)
       `GameWorld::play_animation`
       (`crates/app/src/game_world.rs:131`) сейчас зовёт только `set_playing`
       (без имени и без фейда); после шага 1 он же снимает `qual:api`
       с `Animator::play_crossfade` (`crates/animation/src/animator.rs:119`).
       Точка входа `EntityMut::animator()` уже есть
       (`crates/app/src/game_world.rs:831`).
     - **Шаг 3 — стек слоёв** (п. 3) тем же типизированным входом фасада.
       ✅ Сделано (2026-10-22, #180): `SkelLayerStack` (`crates/animation/src/blend.rs`)
       — стек `SkelLayer` на корень, снизу вверх, `set_skel_layers` заменяет
       стек целиком. Слои остаются только скелетными: `SkelLayer` —
       единственный тип слоя, объектного в модели нет. Пересмотр — когда появится анимационный граф
       (п. 7): вход может уйти в него.

## 6. Осознанно НЕ делаем

- Суставы как ECS-сущности (см. §2.1 — раздувание итерации экстракции).
- Запись анимации в fixed-расписание (кроме root-motion через `Velocity`;
  поза — variable, иначе умножение на сабстепы).
- Анимационные выходы через `MutationBus`: JSON-кодек (`mutation.rs:10-14,89-100`)
  — тул/скрипт-шов, не hot-loop; сэмплер пишет лейны напрямую типизированно.
- Свой `Transform` (`render/src/transform.rs:7-14`) как лейн: рендер читает
  `TransformDesc`, второй placement-источник = рассинхрон.
- IK/ретаргет в фазах A–D; бленд-дерево/state machine — §7 (поверх E);
  морфы (blend shapes) — только после E.
  Дерево анимаций поверх стека слоёв начинаем после E — модель в §7;
  IK/ретаргет, события по ключам, motion matching и distance matching /
  motion warping остаются отложенными до прод-доступности общего
  `Animator`-фасада (§5, «Дальше»).
- Изменение `InstanceData`/шейдеров ради фаз A–C (объектная ложится в `model_matrix`,
  скиннинг — в `custom_meshes`).
- Тихие фолбэки: незагрузившийся скелет/вес/клип = скип + счётчик, никогда сфера-заглушка
  (честность как `skipped_bad_custom`, `extraction.rs:368-376`).

## 7. Дерево анимаций (модель)

Бленд-дерево/state machine живёт **поверх** фазы E: те же клипы, те же
правила смешивания, только вместо одного `SkelPlayer` на корень — граф
(`SkelTree`) и его горячее состояние (`SkelTreePlayer`).

### 7.1 Модель

- `TreeNodeDef` — нода графа (авторские данные, холодный лейн `SkelTree`):
  - `Clip { clip }` — лист: один клип, сэмплится по своему курсору;
  - `Blend1d { axis, children }` — 1-D бленд вдоль параметра `axis`
    (`TreeParam`: имя + default), `children` — позиции на оси + индексы
    дочерних нод.
- Порядок нод = порядок вычисления: `Blend1d` ссылается только на более
  ранние ноды, последняя нода — корень дерева. Так обход без рекурсии и
  без стека в горячем пути.
- `SkelTreePlayer` — горячий лейн на том же корне: значения параметров +
  курсоры листьев (по одному на `Clip`-ноду, в порядке нод). Единственное
  состояние дерева — часы курсоров и выставленные хостом параметры; поза
  никогда не возвращается в следующий кадр (контракт фазы E).

### 7.2 Веса blend1d и фолд поз

- `blend1d_weights(children, value, out) -> n` — детерминированные веса:
  активны не более двух соседей, веса в сумме дают 1; вне диапазона оси
  позу целиком берёт ближайший конец; ровно на позиции — её нода; при
  равных соседних позициях выигрывает более поздняя. Пустые `children` =
  rest-поза.
- N-way фолд (`BlendedPose::fold_pose`): аккумулятор стартует из rest и
  подмешивает позы по очереди с перенормировкой `w / Σw` — двухвходовой
  случай точно совпадает с `mix_towards` (тот же lerp/slerp), один вход
  читается как поза целиком. Порядок фолда фиксирован (порядок нод), итог
  детерминирован.

### 7.3 Горячий путь

Оценка дерева (`eval_tree`) пишет по одной scratch-позе на ноду (буферы
переиспользуются между кадрами, `fit_tree_poses` — только рост), корень
(последняя нода) становится базовой позой: поверх неё идёт обычный стек
`SkelLayerStack` (общий `overlay_layers` с player-путём), затем без
изменений `compose_model_matrices` + `node_poses` (контрибьюторы — клипы
всех листьев). Управление — типизированный фасад `set_skel_tree` /
`set_tree_param` / `clear_skel_tree`; корень с деревом игнорирует
`SkelPlayer` на той же сущности (его часы даже не шагают), снимает
летящий `SkelFade` при установке, а playback-входы
(`fade_to_clip`/`play_crossfade`/`play_animation` и статусы
`resume_animation`/`pause_animation`/`stop_animation`) отвечают
`PlaybackError::TreeDriven`, пока `clear_skel_tree` не вернёт корень
проигрывателю. `set_skel_tree` отклоняет битое дерево сразу (неизвестный
`ClipId` листья — проверка как у `fade_to_clip`, ось без параметра,
не строго возрастающие или не-конечные позиции, не-конечные default'ы); то же
нарушение, найденное в горячем пути (включая не-конечный параметр),
читается как bad skin: корень не публикуется, счётчик растёт — никаких
усечённых смесей и отравленных NaN-весов. Контракт детей проверяется
полностью (все дети, не только активные), чтобы кривое дерево не меняло
вердикт от значения параметра.

### 7.4 State machine

Состояние — это нода дерева: `TreeNodeDef::Machine` держит `states`
(индексы более ранних нод), `initial` и список `TreeTransition`. Переход
`{from, to, when, fade}` срабатывает, когда машина стоит на `from` и
`when` (`TreeCond`: сравнение живого параметра с порогом через
`TreeCmp::{Greater, Less}`) истинно; первый совпавший переход в списке
побеждает. Кроссфейд — механика `SkelFade`: линейный прогресс, часы
ушедшего состояния продолжают идти, неположительный `fade` режет сразу.
Летящий кроссфейд **никогда не прерывается** — условия перепроверяются
только после его завершения (стабильность входа в состояние).

Горячее состояние — `SkelTreePlayer::machines` (курсор на ноду-машину:
текущее состояние, уходящее состояние и часы фейда), шагается в общем
`advance_tree_players`; установка дерева (`set_skel_tree`) сбрасывает
машины на `initial`. Петля `from == to` разрешена: перезапускает
кроссфейд состояния в себя после каждого завершения (пустая смесь —
вреда нет, условие остаётся живым). Оценка (`eval_tree`) читает готовые
позы состояний и
выдаёт позу машины: чистое состояние целиком или их смесь по прогрессу
фейда. Контракт машины (непустые состояния, ссылки на более ранние ноды,
`initial` и переходы в пределах `states`, конечные пороги и `fade`,
конечный живой параметр условия) проверяется целиком каждый кадр и при
установке — нарушение читается как bad skin. Дерево с машиной может
смешивать бленды и машины произвольно: ноды выполняются в порядке
нумерации, корень — последняя нода.

### 7.5 Метки и синхронизация

Именованные метки клипа (`SyncMark`: имя + нормализованная позиция `at` в
цикле `[0, 1)`; каталог `ClipSyncMarks` — холодный лейн по `ClipId`)
выставляются типизированным входом `set_clip_sync_marks(store, clip, marks)`
(пустой список — снять метки). Метки — авторские данные: хост подписывает
контакты шага (`foot`), и переход машины выравнивает фазы циклов.

Когда переход машины срабатывает (семантика §7.4), каждый зацикленный лист
входящего состояния берёт фазу у первого листа уходящего состояния (в
порядке нод), чей клип разделяет с ним имя метки: курсор входящего листа
переставляется так, что следующие попадания в метки общего имени у обоих
клипов случаются в один момент времени (сетка — первая метка имени в
порядке автора; уходящий клип задаёт, сколько времени осталось до его
ближайшего попадания). Так walk→run не дёргает шаг: оба цикла держат общий
контакт.

Правила и границы:

- выравнивание — однократно, в момент срабатывания перехода (включая
  мгновенный рез при неположительном `fade`); летящий кроссфейд часы не
  переставляет;
- лист без пары (нет общего имени, нет меток, не зациклен, нет живого
  клипа или его цикл неположителен) не трогается; лист, живущий в обоих
  состояниях, тоже не трогается — это те же часы, self-loop (`from == to`)
  фазу поэтому не меняет;
- метки живут только на пути деревьев: `fade_to_clip`/`play_crossfade` часы
  не переставляют;
- состояние по-прежнему — только часы курсоров: перестановка времени —
  чистая функция меток и текущих часов, кадры повторяемы.

`set_clip_sync_marks` отклоняет неизвестный `ClipId` и кривые метки (пустое
имя, `at` вне `[0, 1)` или не-конечный) и ничего не пишет.

### 7.6 Дальше

1. ✅ Модель и blend1d-примитивы (веса, фолд поз) + blend1d в горячем пути.
2. ✅ State machine: состояния = ноды, переходы по условиям на параметрах,
   кроссфейд между состояниями (семантика `SkelFade`).
3. ✅ Метки/синхронизация (фаза клипа по меткам при переходах машин).
4. blend2d — по надобности.

