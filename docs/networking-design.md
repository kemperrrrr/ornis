# Дизайн мультиплеера ornis (срез L-design)

Трек net. Код этим документом не пишется.
Основание: `PLAN.md`, раздел «l. Мультиплеер».
Утверждения о коде сверены с master `7fcdf8a` (ревью PR #51, issue #108).
Язык документа — русский. Каждое утверждение о коде ornis подкреплено
путём к файлу и функции.

---

## 1. Изучение практики: что берём, чего избегаем

Для каждого источника — вердикт «берём / избегаем / недоступен» и почему.
Ссылки ведут на публичные документы; характеристики ниже сверены с ними,
а не пересказаны по памяти.

### 1.1. `docs/PLAN-networking.md` — НЕДОСТУПЕН

- Файла `docs/PLAN-networking.md` в репозитории ornis нет. В `docs/` лежат
  другие материалы (`animation-design.md`, `WASM_PIXEL_E2E.md`,
  `rendering/`, `quality/`, этот файл); сетевого плана Balaur среди них нет.
- Доступа к репозиторию Balaur из этого среза нет, найти файл там невозможно.
- Вердикт: **недоступен**. Если Юра даст доступ к Balaur или сам файл —
  следующий срез сверит дизайн с ним и дополнит документ.

### 1.2. Rollback-модуль и Recordings-панель Balaur — НЕДОСТУПНЫ

- По `PLAN.md` §l это образцы для L1 (rollback) и L2 (запись сессий).
- Доступа к коду Balaur нет, разобрать устройство модуля и панели невозможно.
- Вердикт: **недоступны**. Компенсация: L1 проектируем по GGPO/ggrs
  (открытая модель, §1.4), L2 — по формату «ввод + сид + версия + per-tick digest»
  (§4.3), что покрывает ту же функциональность (повтор + проверка).
  При появлении доступа — сверить и зафиксировать расхождения отдельным коммитом в док.

### 1.3. Photon Quantum — БЕРЁМ input-only ядро, ИЗБЕГАЕМ закрытость и привязку к движку

Источники: [обзор Quantum](https://doc.photonengine.com/quantum/current/quantum-intro),
[кадры](https://doc.photonengine.com/quantum/current/manual/frames),
[ввод](https://doc.photonengine.com/quantum/v3/manual/input).

Сервер Quantum game-agnostic: симуляцию по умолчанию не гоняет. Он
синхронизирует часы и рассылает **server-confirmed tick-set** — полный
набор вводов всех игроков на тик. Опоздавший ввод сервер подменяет.
Клиент различает verified-кадр (все вводы тика подтверждены, и все
предыдущие тики тоже verified) и predicted-кадр (локальная голова без
подтверждения). «Авторитет сервера» в смысле «сервер симулирует мир и
шлёт состояние» — неверная формулировка для Quantum.

- **Берём:** обмен только вводом. Одинаковый подтверждённый набор вводов
  на одинаковом тике даёт одинаковое состояние у всех пиров. Это модель
  v1 ornis (§3).
- **Берём:** канонический набор вводов на тик задаёт одна согласованная
  сторона, а не порядок прибытия пакетов. У Quantum это сервер. В ornis v1
  симуляционного сервера нет (§3), поэтому ту же роль играет полный набор
  пиров сессии, упорядоченный по `PeerId` (§4.1): отсутствующий ввод —
  predicted, а не «применим то, что успело прийти».
- **Берём:** предсказание локального ввода и откат, когда приходит
  подтверждённый набор. Локальный игрок не ждёт сеть. Verified и predicted
  — разные кадры, как в Quantum; смешивать их в одном тике нельзя (§1.8 п.1).
- **Берём:** квантование непрерывного ввода до того, как он попадает в
  общую симуляцию. В Quantum Unity-`float` переводится в fixed-point
  (`ToFPVector3`) в `PollInput`, ещё до `SetInput`. Для ornis это касается
  непрерывных полей `InputState` (`pointer_delta: [f32; 2]`,
  `wheel_delta: f32`, `crates/input/src/state.rs`). В v1 геймплей их не
  читает: `player_input` берёт только дискретные действия `InputMap`,
  указатель остаётся у орбитальной камеры (§2.5). Если непрерывная ось
  когда-нибудь войдёт в тик, квантовать её нужно до записи в `InputBundle`,
  тем же приёмом.
- **Берём:** детерминированные строительные блоки (математика, физика) как
  предусловие — совпадает с Strong Confluence + G8, `PLAN.md` §l.
- **Избегаем:** закрытый код и привязку netcode к движку и облаку Photon.
  ornis делает открытый `crates/net` с трейтом транспорта (§4.5).
- **Избегаем:** «магию без цены». Явно фиксируем границы солверов, одну
  архитектуру пиров в v1 и реакцию на десинк (§4.2, §4.4).

### 1.4. GGPO / ggrs — БЕРЁМ rollback-скелет целиком

Источники: [GGPO](https://github.com/pond3r/ggpo),
[ggrs](https://github.com/gschup/ggrs),
[`SyncTestSession`](https://docs.rs/ggrs/latest/ggrs/struct.SyncTestSession.html),
[bevy_ggrs](https://github.com/gschup/bevy_ggrs).

GGPO — эталон rollback для файтингов. ggrs — Rust-реализация того же
скелета: сессии P2P, задержка ввода, checksum на кадр, save / load / advance.
bevy_ggrs сажает этот скелет на расписание Bevy: локальный ввод снимается
в отдельном расписании `ReadInputs`, симуляция идёт в `GgrsSchedule` и
читает уже подтверждённые `PlayerInputs`, а не живой `ButtonInput`.

- **Берём:** три операции как интерфейс отката — сохранить состояние тика,
  загрузить состояние тика, продвинуть симуляцию на один тик с заданным
  вводом. Это ядро L1 (§4.2).
- **Берём:** настраиваемую задержку ввода (input delay). Локальный ввод
  применяется через N тиков, чтобы реже предсказывать. Параметр сессии,
  не константа.
- **Берём:** per-tick checksum. Расхождение ловится на тике, а не «когда
  разъехалась картинка». Состав digest — §4.2. Checksum считает симуляция
  (так устроен ggrs); алгоритм ornis фиксируем там же, без новой зависимости.
- **Берём:** `SyncTestSession` как гейт снапшота (§6.2). Каждый
  `advance_frame` сохраняет состояние, продвигает один кадр, откатывается
  на `check_distance` кадров, пересимулирует и сравнивает checksum.
  Расхождение — ошибка гейта. Sparse saving с этим гейтом несовместим:
  сохранять нужно каждый тик окна. Крейт ggrs в workspace не тащим
  (hard-gate `cargo outdated`, `AGENTS.md`); повторяем процедуру в тесте ornis.
- **Берём:** разделение «сеть приносит пачки ввода, симуляция тикает своим
  шагом». Это L0 (§4.1).
- **Избегаем:** допущения «состояние маленькое и сериализуется бесплатно».
  Мир ornis (ECS + ресурсы + физика) большой; снапшот всего мира — отдельный
  срез со своим форматом и бюджетом (§5, §6.2).

### 1.5. Amethyst — ИЗБЕГАЕМ как анти-паттерн процесса

Что известно: Amethyst — открытый ECS-движок на Rust; разработка ушла
в бесконечный рерайт архитектуры вместо стабилизации релизов, проект
фактически заглох как игровой движок.

- **Берём (единственное):** ECS-организацию логики и data-driven подход —
  у ornis это уже есть (`crates/core`, `Schedule`, `ComponentRegistry`).
- **Избегаем:** «бесконечный рерайт вместо релизов». Прямое следствие для срезов:
  каждый срез L0/L1/L2 и срез снапшота — независимо мержимый, со своим гейтом (§6),
  без перепроектирования ядра ради сети. Сеть адаптируется к `Engine`/`Schedule`,
  а не наоборот.
- **Избегаем:** расширение скоупа v1 (MMO-репликация, RPC, WebRTC) —
  см. «вне скоупа» в `PLAN.md` §l и §3.

### 1.6. Godot — КОНТРАСТ, вне скоупа v1

Что известно (Godot 4: авторитет узлов, `MultiplayerSynchronizer` для
репликации свойств, RPC-вызовы, server/client authority):

- **Понимаем как контраст:** модель Godot — репликация состояния
  (свойства узлов синхронизируются хостом) + RPC для событий.
  Это удобно для кооператива с толерантностью к расхождениям,
  но требует доверия к состоянию и не даёт бит-детерминизма.
- **В v1 не берём:** ни репликацию состояния, ни RPC. Причины:
  1) без детерминированного фундамента (он ещё строится: G8 не начат)
     репликация скрыла бы десинки вместо их ловли;
  2) физика ornis (Sequential Impulse, спящие тела, warm-start) не переживёт
     наложение чужого состояния без отката — нужен именно input-only + rollback;
  3) скоуп v1 — малые сессии lockstep/rollback (`PLAN.md` §l).
- **Возможно позже:** RPC-подобные команды лобби/мета-уровня (старт, пауза)
  поверх того же транспорта — но это уже за пределами v1.

### 1.7. Экосистема Bevy: renet, lightyear, naia, matchbox — БЕРЁМ разделение слоёв

- **[renet](https://github.com/lucaspoffo/renet)** — client/server библиотека
  поверх UDP с протоколом netcode (`renet_netcode`). Это не QUIC и не P2P.
  Для модели v1 (P2P, без симуляционного сервера, §3) renet как транспорт
  не подходит. Берём только приём: каналы с разной надёжностью живут под
  симуляцией и не диктуют ей порядок тиков.
- **[lightyear](https://github.com/cBournhonesque/lightyear)** — ближайший
  аналог v1. Режим deterministic replication передаёт только ввод и
  совместим и с lockstep, и с prediction/rollback
  (`crates/deterministic`, пример `examples/deterministic_replication`).
  Ввод буферизуется на тик: сборка локального ввода — в `FixedPreUpdate`
  (`WriteClientInputs` / `BufferClientInputs`), симуляция читает его в
  `FixedUpdate`, приём чужих пакетов — в `PreUpdate`, до отката.
  Описание: [Inputs](https://cbournhonesque.github.io/lightyear/book/concepts/advanced_replication/inputs.html).
  Репликация мира, interest management и серверный авторитет у lightyear
  тоже есть — в v1 их не берём.
- **[naia](https://github.com/naia-lib/naia)** — server-authoritative
  репликация сущностей и сообщений (UDP нативно, WebRTC в браузере).
  Это стек репликации, а не «альтернативный транспорт» под нашу симуляцию.
  В v1 не берём: по проводу пошло бы состояние (§1.8 п.4).
- **[matchbox](https://github.com/johanhelsing/matchbox)** — WebRTC-транспорт
  и matchmaking, в том числе для связки с ggrs. Это как раз транспорт.
  WebRTC в v1 вне скоупа (`PLAN.md` §l); matchbox остаётся кандидатом
  wasm-транспорта за пределами v1, рядом с WebTransport (§4.7).

Выводы для ornis:

- **Берём:** жёсткое разделение «транспорт vs симуляция», как у lightyear
  и bevy_ggrs. В `crates/net` транспорт — трейт (§4.5). Симуляция знает
  `Tick` и `InputBundle` текущего тика. Ошибка «смешение транспорта и
  симуляции» из `PLAN.md` закрывается на уровне типов.
- **Берём:** явный шаг симуляции, отдельный от рендер-кадра. Длительность
  шага уже есть: `FixedTime::delta_seconds()` (§2.1). Сетевой и записываемый
  путь продвигает ровно один `Tick` на один `InputBundle` и не выводит число
  тиков из wall-clock (§1.8 п.3).
- **Берём:** каналы с разной надёжностью (ввод — надёжно, по номерам тиков;
  digest — ненадёжно, агрегатом) при выборе QUIC-каналов (§4.7).
- **Избегаем:** копирования lightyear целиком (репликация мира, interest
  management, серверный авторитет) — вне скоупа v1.
- **Избегаем:** привязки к одному транспорту. QUIC нативно в v1, wasm-вариант
  отдельно (§4.7), смена — заменой имплементации трейта.

### 1.8. Критические ошибки из `PLAN.md` — разбор каждой

`PLAN.md` §l перечисляет четыре ошибки. Для каждой: в чём суть и чем дизайн закрывает.

1. **Полу-состояния.**
   Суть: применение части сетевого обновления (половина ввода тика,
   половина снапшота) с продолжением симуляции «как-нибудь».
   Закрытие: ввод тика применяется атомарно — либо полный набор вводов
   всех пиров на тик, либо предсказание с последующим откатом (L1).
   Инвариант фиксируется тестом: тик никогда не видит «половину» `InputBundle`.
   Снапшот — версионированный целиком (`WorldSnapshot` + версия формата),
   частичный restore запрещён типом (restore возвращает `Result`, ошибка —
   десинк, а не продолжение).

2. **Смешение транспорта и симуляции.**
   Суть: сокеты, ретраи и таймауты внутри систем симуляции; тик зависит
   от состояния сети.
   Закрытие: `crates/net` делится на `Transport` (байты, тики не знает)
   и адаптер, который раскладывает arrivals по `Tick` (§4.1, §4.5).
   Приём сокета — раз в кадр, на variable-границе (как `PreUpdate` у
   lightyear). Применение — в начале каждого fixed-шага, из ресурса
   текущего `Tick`. Системы симуляции сокет не видят.
   Зависимости: `crates/net` зависит от `ornis-core` и `ornis-input`,
   не наоборот и не от `ornis-app` (§4.5).

3. **Wall-clock в тике.**
   Суть: `Instant::now()` или дельта кадра выбирает, какой ввод съест
   детерминированный шаг, или сколько шагов «считается миром». Пиры с
   разным FPS расходятся.
   Что код делает сейчас и чем это закрывается — §2.5 и §2.6. Коротко:
   длительность шага уже фиксирована (`FixedTime::delta_seconds()`),
   но число шагов кадра и то, какой `InputState` они видят, задаёт
   wall-clock. Пока `player_input` читает этот снимок с variable-расписания,
   replay L2 не воспроизводится.
   Закрытие, первая задача среза L2 (§6.1): ввод сэмплируется на `Tick`,
   `player_input` переезжает в `fixed_schedule` и читает `InputBundle`
   этого тика, а сетевой и записываемый прогон шагает тики по журналу,
   не по `Instant`. Гейт: два прогона с разным wall-расписанием кадров
   и одним журналом тиков дают один digest-лог.

4. **Состояние вместо инпутов на проводе.**
   Суть: пересылка позиций и скоростей и наложение их на мир — скрытые
   десинки, читерство, отсутствие повтора.
   Закрытие: на проводе в v1 только ввод (`InputBundle` по тикам) и digest
   для проверки. Состояние по сети не ходит. Исключение одно: bootstrap
   сцены (идентификатор сцены, сид, стартовое состояние `EntityAllocator`)
   — это начальные условия, не состояние посреди сессии. L2-запись — тот
   же формат, что провод: файл сессии можно скормить replay без сети.
   Реакция на десинк в v1 — остановка и локальный дамп, не запрос чужого
   снапшота (§4.2).

---

## 2. Что уже есть в ornis (с путями к файлам)

### 2.1. Фиксированный шаг и тик

- `FixedTime`: `crates/core/src/engine.rs`. Константы
  `DEFAULT_FIXED_DELTA_SECONDS = 1.0 / 60.0` и
  `DEFAULT_MAX_FIXED_STEPS_PER_FRAME = 8`. Поля структуры:
  `delta_seconds`, `accumulator_seconds`, `elapsed_seconds`, `tick`,
  `steps_this_frame`, `current_step`, `alpha`, `dropped_seconds`,
  `max_steps_per_frame`.
- `FixedTime::begin_frame(frame_delta)` прибавляет дельту кадра к
  аккумулятору и выбирает, сколько fixed-шагов выполнить (не больше 8).
  Избыток списывается в `dropped_seconds`, спирали догона нет.
  `FixedTime::start_step` увеличивает `tick` на 1 и вычитает ровно
  `delta_seconds`. `alpha()` — дробный остаток аккумулятора для
  интерполяции картинки; комментарий у метода прямо говорит, что fixed-системы
  должны брать `delta_seconds`, а не `alpha`.
- `Engine::run_frame` (`crates/core/src/engine.rs`): публикует `Time` из
  дельты кадра, вызывает `begin_frame` этой же дельтой, затем гоняет
  `PreUpdate` → `Input` → цикл `Gameplay` (`fixed_schedule`, N раз) →
  `PostFrame` (хранилище `schedule`). Документация метода: каждый fixed-шаг
  кадра видит один и тот же снимок ввода; после `PostFrame` вызывается
  `InputState::clear_frame_transients`.
- Тип `Tick(u64)` (newtype, ручной `Serialize` как `u64`):
  `crates/core/src/typestate.rs`, рядом `Seconds`, `Frame`, `FixedSteps`.
  `FixedTime::tick_id()` возвращает этот `Tick`.
- Физика на том же fixed-пути: `install_physics`
  (`crates/app/src/physics_runtime.rs`) ставит sync-in → soft-sync-in →
  step → sync-out → soft-sync-out в `fixed_schedule_mut`. `PhysicsStep`
  читает `FixedTime::delta_seconds()` и вызывает `solver.step`. Wall-clock
  солвер не читает.

Вывод: длительность сетевого тика — `FixedTime::delta_seconds()`,
идентичность тика — `Tick`. Отдельных «сетевых часов» не вводим.
Число тиков за кадр сейчас считает `begin_frame` из дельты кадра; для
записи и сети это не источник тиков (§2.6, §4.1).

### 2.2. Strong Confluence (детерминизм)

- Планировщик: `crates/core/src/schedule.rs`.
- Тесты: `crates/core/tests/confluence_tests.rs` (1-vs-32 паритет).
- Ночной гейт: `.github/workflows/determinism-nightly.yml` запускает
  `cargo test -p ornis-physics -- --ignored --skip regenerate --skip known_limit_`.
  Другие крейты, включая будущий `crates/net`, этот workflow не гоняет.

Вывод: фундамент детерминизма физики уже есть и держится ночью.
Rollback без него бессмыслен. Ночной workflow не заменяет гейт
`record → replay --verify` (§6.1).

### 2.3. Роли, мутации, реестр

- Роли `Authoritative` / `Replica` (phantom-параметры, default — Authoritative):
  `crates/core/src/typestate.rs`; использование — `crates/app/src/game_world.rs`
  (`GameWorld<Role>`, `ReplicaGameWorld = GameWorld<Replica>`; реплика
  наполняется из сериализованных снапшотов, раскладка мира та же).
- `MutationBus` (`crates/core/src/mutation.rs`): продьюсеры, `drain()`,
  `ticks()`, `MutationTick`, `MutationPlugin`.
  - `Mutation` — это `Set` / `SetField` с `serde_json::Value` для правки
    компонента по реестру (инспектор, скрипт), не снимок клавиатуры.
  - `MutationProducer::produce(&mut self, dt_seconds, tick)` получает
    variable-дельту кадра. `tick` здесь — счётчик кадров шины:
    `MutationBus::ticks` подписан как «Frames produced so far»,
    `produce_all` делает `fetch_add` раз за вызов.
  - `MutationTick` стоит в variable-расписании (`MutationPlugin::install`
    вызывает `schedule_mut().add_system(MutationTick)`, это `Stage::PostFrame`).
    Документация типа: producers run once per frame, never per fixed substep,
    и читают `Time`, не `FixedTime`. `MutationTick::run` берёт
    `Time::delta_seconds()`.
  - `MutationBus::drain` только забирает очередь `Vec<Mutation>`.
    Хост применяет её после кадра: `EditorSession::tick_secs`
    (`crates/app/src/session.rs`) сначала вызывает `GameWorld::frame_secs`,
    и только потом `drain_mutations` → `apply_mutations`.
- `ComponentRegistry` (имена, поля, `RegisterComponent`, сериализация
  компонентов): `crates/core/src/registry.rs`.

Вывод: роли уже различают «кто мутирует мир» на уровне типов. Для сети
их можно переиспользовать как «авторитетная симуляция» и «вьюпорт»,
не заводя вторую иерархию ролей. `MutationBus::drain()` точкой сетевого
ввода не является: это кадровая очередь JSON-правок редактора и скриптов,
её применяют после fixed-цикла. Сетевой ввод — отдельный ресурс на `Tick`,
который fixed-системы читают в начале шага (§4.1). Реестр пригодится
снапшоту компонентов (§6.2); к идентичности `Entity` он отношения не имеет (§4.6).

### 2.4. `WorldSnapshot` физики

- Формат: `crates/physics/src/snapshot.rs` (`WorldSnapshot`, `BodySnapshot`,
  `JointSnapshot`, события, warm-пары; `to_ron` / `from_ron`,
  `instantiate_si()`).
- Тест: `crates/physics/tests/world_snapshot.rs`, функция `scene()`
  (пол, ящики, соединённые ball joint, сфера с порогом силы контакта;
  снимок на шаге 90, продолжение 60 шагов, сравнение бит float через
  `body_bits`).
- Ограничение: только Sequential Impulse и `RoutingKind::Single`.
  Остальное — `SnapshotError::Unsupported` (`crates/physics/src/errors.rs`),
  не молчаливая потеря. Модульный док `snapshot.rs` прямо отказывает
  AVBD, XPBD и `RoutingKind::Islands`.

Вывод: снапшот физики — готовый кусок среза снапшота, но только для
SI-Single. Границы сетевых сцен (§4.4) отсюда и следуют. Полный снапшот
мира (ECS + ресурсы + физика + аллокатор) — отдельный срез (§5, §6.2).

### 2.5. Ввод: где wall-clock входит в геймплей

Крейт `crates/input`:

- `InputState` (`crates/input/src/state.rs`) — снимок кадра: `pressed_keys`,
  `pressed_mouse_buttons`, `pointer_position`, `pointer_delta`, `wheel_delta`.
  Модульный док: платформенные адаптеры пишут его между кадрами, системы
  читают тот же ресурс в течение кадра.
- `InputState::apply_snapshot` заменяет снимок целиком (ключи, кнопки,
  указатель, колесо). Так устроен провод браузера: `pressed_keys: Vec<u32>`,
  `pressed_mouse_buttons: Vec<u8>` по WS и `POST /api/input`.
- `InputState::clear_frame_transients` обнуляет `pointer_delta` и
  `wheel_delta` один раз после кадра. Удержание клавиш сохраняется.

Кто пишет `InputState` (всё это между кадрами, не на `Tick`):

- Нативный winit в `crates/runner/src/lib.rs`: обработчики `WindowEvent`
  зовут `GameApp::update_input` и правят ресурс напрямую.
- Браузерный ввод, тот же runner: `GameApp::about_to_wait` вызывает
  `process_remote_commands`, тот на `UiCommand::Input` вызывает
  `GameApp::apply_browser_input` → `InputState::apply_snapshot`.
  `RedrawRequested` после этого вызывает `render_frame`.
- Редакторская сессия: `apply_browser_input` в `crates/app/src/session.rs`
  (из `handle_command`) и `ornis_app::apply_browser_input` в
  `crates/app/src/lib.rs`. Обе заменяют ресурс целиком.

Кто читает, и в каком расписании (`GameplayPlugin::install`,
`crates/gameplay/src/lib.rs`):

- `player_input` (`PlayerInputSystem`) висит на `engine.schedule_mut()` —
  это `Stage::PostFrame`, один раз за кадр, после fixed-цикла.
  Функция читает `InputState` и `InputMap` и пишет горизонтальный
  `Velocity` всем сущностям с `Player`. Указатель она не потребляет:
  комментарий у функции оставляет его орбитальной камере
  (`OrbitCameraSystem` в `crates/render/src/camera.rs`, тоже variable-расписание).
- `physics_push` (`PhysicsPushSystem`) висит на `fixed_schedule_mut()`.
  Интегрирует уже записанный `Velocity` через `FixedTime::delta()`
  (`Position::advanced_by_secs`). Сама дельта шага — фиксированная.
  Но `Velocity` к этому моменту написан `player_input` предыдущего кадра:
  в этом кадре `player_input` ещё не бежал. Все до 8 fixed-шагов кадра
  делят один и тот же `Velocity`.
- `transform_update` (`TransformUpdateSystem`) тоже на `schedule_mut()`,
  после `player_input` (`try_order_before("player_input", "transform_update")`).
  Функция читает `Time::delta_seconds()` (дельта кадра) и `FixedTime::alpha()`,
  собирает приращение `velocity * dt * alpha * 0.0` и выбрасывает его
  (`let _ = residual`). Позицию она сейчас не пишет. Комментарий в конце
  функции предлагает «подмешать сюда намерение, пришедшее после fixed-фазы
  (например networked input)». Это пост-fixed сглаживание по `alpha`,
  и сейчас оно ещё и занулено. Точкой входа `NetAdapter` этот код не является.

Итог по геймплею. Fixed-путь физики (`physics_push`, `PhysicsStep`) шагает
на `FixedTime::delta_seconds()`. Wall-clock входит рядом:

1. Один `InputState` на кадр обслуживает до `DEFAULT_MAX_FIXED_STEPS_PER_FRAME`
   fixed-шагов (`Engine::run_frame` это документирует). Граница кадра,
   а не `Tick`, решает, на каком тике сменится `Velocity`.
2. `player_input` стоит в `PostFrame`, поэтому смена намерения видна
   только следующим кадрам.
3. `transform_update` читает wall-дельту и `alpha`. Записи в мир нет,
   но хук, на который ссылается комментарий, лежит после fixed-цикла.

Закрытие — задачи среза L2, до записи (§6.1), по образцу bevy_ggrs и lightyear:

1. Сэмпл локального ввода — раз в кадр, в `Stage::Input` (это расписание
   уже есть и выполняется до fixed-цикла; геймплей его сейчас не занимает).
   Сэмпл читает `InputState` и складывает дискретные действия `InputMap`
   (`MOVE_FORWARD` / `BACK` / `LEFT` / `RIGHT`) в значение на кадр.
   Внутри fixed-шага `InputState` больше не читают. Непрерывные поля
   (`pointer_delta`, `wheel_delta`) в `InputBundle` v1 не входят: их
   потребители — камера и прочий variable-путь. Если такой оси понадобится
   симуляция, её квантуют до записи в журнал (§1.3).
2. В начале каждого fixed-шага адаптер кладёт в ресурс текущего `Tick`
   полный `InputBundle`. Удержанные действия повторяются на каждый тик
   кадра (флаг repeatable у Quantum, повтор последнего ввода у lightyear).
   Одноразовые края, если когда-нибудь войдут в бандл, занимают ровно один
   тик: сегодня `clear_frame_transients` привязан к кадру, не к тику.
3. `player_input` переезжает в `fixed_schedule` (`Stage::Gameplay`) и
   встаёт перед `physics_push`. Читает `InputBundle` тика
   (`FixedTime::tick_id()`), пишет `Velocity`. `physics_push` остаётся
   на `FixedTime::delta()`.
4. `transform_update` остаётся в `PostFrame` и не пишет авторитетные
   `Position` и `Velocity`. Комментарий про networked input в этой функции
   снимается, когда срез L2 переносит потребление ввода. `NetAdapter`
   сюда не подключается (§6.3).

### 2.6. `runner`, сессия редактора и wall-clock

Два хоста меряют `Instant` и кормят им кадр:

- `GameApp::render_frame` (`crates/runner/src/lib.rs`) хранит
  `GameContext.last_frame: Instant`, берёт `elapsed()`, тут же пишет
  новый `Instant::now()` и вызывает
  `GameWorld::frame_secs(clamp_frame_dt(elapsed))`.
- `editor_world_loop` (`crates/app/src/session.rs`) так же хранит
  `last_tick: Instant` и передаёт `clamp_frame_dt(last_tick.elapsed())`
  в `EditorSession::tick_secs`.

`clamp_frame_dt` ограничивает замер сверху константой `MAX_FRAME_DT`
(0.1 с). `GameWorld::frame_secs` вызывает `Engine::run_frame_secs`,
а тот — `FixedTime::begin_frame`. Дельта кадра выбирает число fixed-шагов
и остаток `alpha`. При длинном кадре шаги сверх восьми выбрасываются
(`dropped_seconds`): два пира с разным FPS могут выполнить разное число
тиков за один и тот же стенный интервал.

Это нормальный аккумулятор для локальной картинки (та же роль, что
`Time<Virtual>` у Bevy: сколько раз звать fixed-расписание в этом кадре).
Для записи и для пиров источником тиков он быть не может: пропущенный
из-за hitch тик на одной машине — уже другой журнал.

Вывод: wall-clock законно живёт на границе кадра и только отвечает на
вопрос «сколько локальных шагов успеть нарисовать». Сетевой и
записываемый прогон на этот вопрос не опирается. Он продвигает ровно
один `Tick` на один записанный `InputBundle`, с дельтой
`FixedTime::delta_seconds()`, без `Instant` и без `dropped_seconds`.
Локальный `run_frame` для одиночной игры остаётся как есть.

### 2.7. `Entity` и аллокатор

- `crates/core/src/entity.rs`: `Entity` — пара `EntityId(u32)` +
  `Generation(u32)`. `Serialize` / `Deserialize` / `Ord` на типе нет.
- `EntityAllocator` (`next_id`, `free_list`, `generations`):
  `allocate` снимает id с хвоста `free_list` (LIFO) либо выдаёт
  `next_id` и растит его; поколение берётся из `generations[id]`.
  `deallocate` увеличивает поколение и кладёт id в `free_list`.
  Одинаковая последовательность `allocate` / `deallocate` даёт одинаковые
  `Entity`. Формулировка «спавн порядок не гарантирует» для самого
  аллокатора неверна.

Вывод: lockstep с одинаковой симуляцией получает одинаковые id, если
стартовое состояние аллокатора одинаково и спавны — функция тика и
канонического ввода. В снапшот входят `next_id`, `free_list` и
`generations` (§6.2). Отдельный `NetEntityId` в v1 не заводим (§4.6).

### 2.8. Только текстовая сериализация

- В `crates/core` — `serde_json`; снапшоты физики — RON
  (`WorldSnapshot::to_ron` / `from_ron`, `crates/physics/src/snapshot.rs`).
- Бинарного wire-формата нет.

Вывод: v1-проводу нужен компактный бинарный кодек (bincode/postcard-подобный,
решение — за L0-срезом, здесь фиксируем требование: детерминированный,
версионированный, без JSON на горячем пути).

### 2.9. Снапшоты детерминизма и G8

- Канонические снапшоты отдельные: `crates/physics/tests/data/`
  (`determinism_snapshot.x86_64.hex`, `determinism_snapshot.aarch64.hex`,
  плюс `avbd_snapshot.hex`).
- Канон-тесты хешируют сырые биты: `f32::to_bits` в
  `crates/physics/tests/avbd_engine.rs` и в `crates/physics/tests/world_snapshot.rs`.
- G8 (детерминизм CPU+GPU) не начат. GPU-путь (`WgpuContactSolver` /
  `GpuSequentialImpulse`, фича `gpu`) не бит-идентичен CPU по построению.
- Ресурса RNG в ornis нет: крейта `rand` в workspace-зависимостях нет
  (`Cargo.toml`). Псевдослучайность в тестах физики собрана без `rand`
  (`crates/physics/tests/math_props.rs`).

Вывод: в v1 пиры обязаны быть на одной архитектуре (§4.4). Кросс-платформа
и G8-сцены — вне скоупа v1. Детерминированный RNG — отдельная задача,
не часть digest, пока ресурса нет (§4.2).

### 2.10. Сетевой код редактора — не основа мультиплеера

- `crates/editor-backend/src/remote.rs`: нативный loopback-транспорт —
  `TcpListener` на `127.0.0.1:{port}`, `RemoteEditor::start/stop`,
  loopback-only hardening (проверки `Host`/`Origin`, лимит тела).
- `crates/editor-backend/src/ipc.rs`: IPC того же loopback-контура редактора.

Почему не основа:

1. Это транспорт «редактор ↔ локальная игра» (команды UI, снапшоты сцены),
   а не «пир ↔ пир» (ввод по тикам, digest, откат).
2. Привязка к `127.0.0.1` и HTTP/WS-семантика — не игровой провод
   (нет тиков, каналов надёжности, QUIC).
3. Переиспользование создало бы ту самую ошибку «смешение транспорта
   и симуляции». Единственное, что можно позаимствовать, — практику
   loopback-тестов (in-process транспорт для тестов `crates/net`, §4.5).

---

## 3. Модель v1

- **Сессии:** малые (2–4 пира, P2P полносвязно или через релей; выделенного
  симуляционного сервера в v1 нет).
- **Симуляция:** lockstep с предсказанием и rollback (по GGPO-скелету, §1.4).
- **На проводе — только ввод:** на каждый `Tick` каждый пир публикует свой
  ввод. `InputBundle` тика — полный набор. Состояние не пересылается.
- **Без репликации состояния, RPC и WebRTC.** `ReplicaGameWorld` из
  `game_world.rs` остаётся браузерным вьюпортом, а не сетевой репликой.
  Это же ограничение записано в `PLAN.md` §l («вне скоупа v1»).
- **Предусловие:** Strong Confluence зелёный и сетевые сцены в границах §4.4.
  Без детерминизма rollback невозможен (зафиксировано в `PLAN.md` §l).

---

## 4. Дизайн по уровням

### 4.1. L0 — delivery model (граница ввода)

Правила доставки — инварианты, а не рекомендации. Образец — связка
lightyear (приём в `PreUpdate`, буфер на тик, чтение в `FixedUpdate`) и
bevy_ggrs (`ReadInputs` отдельно от `GgrsSchedule`).

1. Сокет читается **раз в кадр**, на variable-границе (до fixed-цикла).
   В симуляцию ввод попадает **на каждый `Tick`**, в начале fixed-шага
   этого тика, до `player_input` и `physics_push`. За кадр таких шагов
   бывает от 0 до `DEFAULT_MAX_FIXED_STEPS_PER_FRAME` (8). «Раз в тик» и
   «раз в кадр» поэтому разные места.
2. Порядок прибытия пишется в отладочный лог приёма и больше никуда.
   В симуляцию `InputBundle` тика подаётся в каноническом порядке
   `(Tick, PeerId)`: пиры сессии заранее упорядочены по `PeerId`,
   как player handle в ggrs. У разных пиров порядок пакетов разный,
   порядок в бандле — один. Это тот же вывод, что server-confirmed
   tick-set у Quantum, только подтверждающая сторона — полный набор
   пиров, а не симуляционный сервер (§3).
3. Тик записи и сети **не выводится из wall-clock**. Arrivals ждут своего
   `Tick`. Опоздавший ввод этого тика — предсказание и пометка для отката
   L1, не повод применить «сколько успело прийти» и не повод пропустить
   тик из-за `dropped_seconds`.
4. Хранилище: `Tick → InputBundle`. Бандл полон (по слоту на каждого пира);
   отсутствующий слот помечен `predicted`. Системы видят бандл целиком.
5. Реплей идёт **без сети**: журнал тиков + сид + версия + старт сцены (§4.3).
6. Атомарность тика: система видит либо полный `InputBundle`, либо
   предсказанный с флагом. «Половин» не бывает (§1.8 п.1).

Ресурс ввода, не `MutationBus`. На время fixed-шага в мире лежит ресурс
текущего тика (рабочее имя `TickInput`: `Tick` + `InputBundle`). Его
заполняет адаптер L0 в начале шага. `player_input` и прочие fixed-системы
читают его и про сокет не знают. `MutationBus::drain()` сюда не входит:
после кадра он отдаёт JSON-`Mutation` редактора (§2.3). Класть туда
сетевой ввод значило бы снова привязать тик к кадру и к `Time::delta_seconds()`.

Локальный сэмпл кадра (клавиши, которые игрок держит прямо сейчас) живёт
в `Stage::Input` и только кормит слот этого пира в бандлах тех тиков,
которые кадр собирается выполнить. Удержание повторяется на каждый такой
тик. Это не чтение `InputState` из `physics_push`.

### 4.2. L1 — rollback (снапшот, digest, десинк)

Состав:

1. **Снапшот всего мира** делает срез §6.2: ECS (компоненты через
   `ComponentRegistry`, `Entity` как пара id+generation, §4.6), ресурсы
   (весь `FixedTime`, не один `tick`), состояние `EntityAllocator`,
   физика (`WorldSnapshot`, сейчас только SI-Single) и привязки
   `PhysicsRuntime`. Формат версионирован. Restore целиком или никак
   (`Result`, не полу-состояние).
2. **Restore и пересимуляция** по GGPO-скелету (§1.4):
   `save(tick) / load(tick) / advance(tick, inputs)`. Окно хранения —
   параметр сессии (по умолчанию несколько секунд тиков; точное число —
   после замера памяти в §6.2).
3. **Per-tick digest.** На каждый тик в канонический поток байт входят:
   - номер тика и версия формата;
   - канонический `InputBundle` тика (после разрешения предсказаний,
     пиры по возрастанию `PeerId`);
   - поля тела, которые `body_bits` в `crates/physics/tests/world_snapshot.rs`
     уже собирает в поток бит: поза, ориентация, линейная и угловая
     скорости, масса, обратная масса, момент силы, restitution, friction,
     инерция, тип тела. Кодировка — сырые `f32::to_bits`, как там и в
     `crates/physics/tests/avbd_engine.rs`. Одной позы мало: миры с
     одинаковой позой и разной массой не должны сходиться в один digest;
   - число живых сущностей и таблица id в порядке аллокатора
     (`next_id`, затем `free_list`, затем `generations`), не в порядке
     обхода `HashMap`.

   Квантовать float перед хешем нельзя: на одной архитектуре расхождение
   в младших битах и есть десинк, квантование его спрячет. Единственная
   канонизация бит: `-0.0` записывается как `+0.0`, NaN в хешируемое
   состояние не допускается (это ошибка состояния, не значение).
   RNG в поток не входит: ресурса RNG нет (§2.9). Появится детерминированный
   генератор — его состояние добавит отдельная задача, вместе с ресурсом.

   Хеш считает симуляция, как checksum в ggrs. `std::collections::hash_map::DefaultHasher`
   не берём: стабильность `std::hash::Hash` между версиями rustc не обещана,
   а `RandomState` у `HashMap` ещё и разный между процессами.
   Новую зависимость (`xxhash`, `twox-hash` и аналоги) не заводим: в
   workspace хеш-крейта нет, а `cargo outdated --exit-code 1` — hard gate
   (`AGENTS.md`). Алгоритм фиксированный, без крейта: FNV-1a 64
   ([спецификация](http://www.isthe.com/chongo/tech/comp/fnv/)) по этому
   little-endian потоку. Offset basis `14695981039346656037`,
   prime `1099511628211`. Это детект десинка, не защита от читов.
   Ширина 64 бита: узкого 16-битного checksum на весь мир мало,
   случайные совпадения начали бы прятать расхождения.
   Пиры обмениваются digest на тик агрегатом, не обязательно отдельным
   пакетом на каждый тик.

4. **Реакция на десинк в v1.** Несовпадение digest → состояние
   `Desynced(tick, digest_a, digest_b)`, сессия останавливается.
   Рядом пишутся локальный `.ornis-rec` и локальный снапшот этого тика,
   чтобы `replay --verify` повторил расхождение без сети. Запрос полного
   снапшота у большинства или у хоста в v1 запрещён: это пересылка
   состояния по проводу (§1.8 п.4, §3). Продолжение «как-нибудь» тоже
   запрещено. Античит не делаем (малые сессии, доверие пирам), только детект.
   Ресинк чужим состоянием — вопрос за пределами v1 (§7).

### 4.3. L2 — запись и replay

Формат записи сессии (файл `.ornis-rec`, версия в заголовке):

```
header: magic + format_version + engine_version + scene_id + seed
        + tick_rate(60) + peer_ids + input_delay
        + стартовое состояние EntityAllocator
frames: повтор для каждого тика: tick_no + InputBundle (пиры по PeerId) + digest
footer: final_digest + tick_count
```

- Кодировка v1: компактный бинарный кодек (не JSON/RON — они остаются
  для сцен и отладки, не для горячего пути).
- CLI: `record` (писать во время сессии) и `replay --verify`
  (прогнать без сети и без `Instant`, сверить per-tick digest, ненулевой
  код возврата при расхождении). Образец UX — панель Recordings в Balaur
  (§1.2, недоступна; делаем по этому описанию).
- Запись в сетевых сессиях включена всегда (дёшево: только ввод).
  Кольцо или отдельные файлы — решает срез L2.
- Отладочный лог порядка прибытия в `.ornis-rec` не входит. Иначе replay
  зависел бы от расписания сокета. Этот лог можно писать рядом, отдельным
  файлом, для разбора сети; симуляция его не читает.

### 4.4. Границы v1 (что допустимо в сетевых сценах)

1. **Солвер и маршрутизация:** только Sequential Impulse и
   `RoutingKind::Single` — единственное, что умеет `WorldSnapshot` (§2.4).
   Запрещены также `RoutingKind::Islands` и любой `SoftBody`.
   Первый компонент `SoftBody` сам переводит мир на `SolverKind::Xpbd`:
   `PhysicsRuntime::sync_soft_in` (`crates/app/src/physics_runtime.rs`)
   вызывает `set_solver_kind(SolverKind::Xpbd, ...)`, когда текущий солвер
   ещё не XPBD. Проверки «какой солвер выбран в конфиге» недостаточно:
   сцена без явного XPBD всё равно уедет туда, как только появится soft-тело.
   XPBD, AVBD, GPU-пути и multi-island — вне сетевых сцен, пока их нет
   в снапшоте. Нарушение — ошибка загрузки сцены, не десинк в рантайме.
2. **CPU-only.** GPU-солверы не бит-идентичны CPU, в сетевых сценах их нет.
   G8 не начат (§2.9) и является предусловием, чтобы это правило ослабить,
   а не частью v1.
3. **Одна архитектура у пиров:** x86_64 с x86_64, aarch64 с aarch64
   (канонические снапшоты разные, §2.9). Кросс-архитектура — будущая работа.
4. **Шаг 60 Гц**, `Tick` с нуля сессии после детерминированного старта (§6.1).
   Другой шаг — вне v1.

### 4.5. Крейты `crates/net` и `crates/net-quic`

В `rustqual.toml` у `[architecture.layers]` стоит
`unmatched_behavior = "strict_error"`. Новый крейт без записи в
`paths` слоя и в `[architecture.external_crates]` роняет fast-гейт.
Один крейт сразу с `quinn` (адаптер) и с журналом ввода (домен) в один
слой не кладётся. Слои ornis, изнутри наружу: `foundation`, `domain`,
`application`, `adapters`.

- `crates/net` — слой `domain`. Трейт транспорта, loopback, `InputBundle`,
  журнал тиков, digest, record/replay. Зависит от `ornis-core` и
  `ornis-input`. Не зависит от `quinn` и не зависит от `ornis-app`.
- `crates/net-quic` — слой `adapters`, рядом с `crates/wasm` и
  `editor-backend`. Единственная имплементация `Transport` на `quinn`.
  Появляется не в срезе L2, а вместе с сокетами (§6.4).
- Save/load мира в `crates/net` не помещается. `PhysicsRuntime.bindings`
  и `soft_bindings` — приватные `HashMap` внутри `ornis-app`
  (`crates/app/src/physics_runtime.rs`), fixed-расписание физики собирает
  `install_physics` там же. Порт объявляет `crates/net`, реализацию пишет
  `ornis-app` (слой `application` имеет право зависеть от `domain`):

```rust
// Эскиз. Кода нет. Tick не объявляем второй раз.
pub use ornis_core::Tick;

pub struct PeerId(pub u64);

pub enum Channel {
    ReliableInputs,
    UnreliableDigests,
}

pub struct Datagram {
    pub peer: PeerId,
    pub channel: Channel,
    pub bytes: Vec<u8>,
}

/// Байт-труба. Без `Send` и без `Sync`: wasm-бэкенд держит типы `web-sys`,
/// они не `Send`. Нативный quinn остаётся `Send` внутри `crates/net-quic`,
/// это свойство адаптера, не трейта. `&mut self` у методов и так
/// однопоточный вызов с границы кадра.
pub trait Transport {
    fn send(&mut self, peer: PeerId, channel: Channel, bytes: &[u8]);
    fn recv(&mut self) -> Vec<Datagram>;
    fn peers(&self) -> Vec<PeerId>;
}

pub struct LoopbackTransport;

/// Полный набор вводов одного тика. В симуляцию подаётся по возрастанию PeerId.
pub struct InputBundle {
    pub tick: Tick,
    pub peers: Vec<(PeerId, PeerInput, Prediction)>,
}

pub enum Prediction { Confirmed, Predicted }

/// Дискретные действия, которые сегодня читает `player_input` через InputMap.
/// Не сырой InputState: указатель и колесо в тик v1 не входят (§2.5).
pub struct PeerInput { /* биты MOVE_* */ }

/// Ресурс на время одного fixed-шага. Читают системы Stage::Gameplay.
pub struct TickInput { pub bundle: InputBundle }

/// Порт снапшота. Реализация — в ornis-app: там PhysicsRuntime и сборка fixed-расписания.
pub trait WorldCheckpoint {
    type Error;
    fn save(&self) -> Result<Checkpoint, Self::Error>;
    fn load(&mut self, saved: &Checkpoint) -> Result<(), Self::Error>;
}
```

- `LoopbackTransport` — in-process, для тестов без сокетов.
- `NetAdapter` (L0) живёт в `ornis-app`: только там можно вставить систему
  в `fixed_schedule` рядом с `physics_push` и писать ресурс `TickInput`.
  Логику журнала он берёт из `crates/net`.
- `Rollback` вызывает `WorldCheckpoint` и `advance` журнала. Политика
  отката — в `crates/net`, конкретный save/load мира — в `ornis-app`.
- Обратной зависимости «симуляция знает про сокет» нет: `player_input`
  читает `TickInput`, не `Transport`.

Регистрация обоих крейтов — часть того среза, который крейт создаёт:
`[workspace].members` в корневом `Cargo.toml`, путь в
`[architecture.layers.*]` и ключ в `[architecture.external_crates]`
(`ornis_net`, `ornis_net_quic` — через подчёркивание, как соседние ключи).

### 4.6. Сериализуемый `Entity`

Проблема (§2.7): `Entity` не сериализуется. Сам аллокатор детерминирован.

Решение v1:

1. Снапшот хранит `EntityAllocator` целиком: `next_id`, `free_list`,
   `generations`. Restore поднимает аллокатор до следующей `allocate`,
   а не раздаёт id заново по порядку сущностей в файле.
2. `Entity` в снапшоте — пара `(id, generation)`. По проводу в v1 entity
   не ходят: `PeerInput` — дискретные действия, без ссылок на сущности.
3. `NetEntityId` в v1 не вводим. Он понадобится, только если ввод начнёт
   называть сущность («взаимодействовать с этим id») или если появится
   вход в сессию не с нулевого тика. Второе — это ресинк состояния,
   он вне v1 (§4.2, §7). Пока симуляция у всех пиров одна и та же,
   локального `Entity` достаточно, и второй пространство id только
   разъедется с аллокатором.
4. Обход `HashMap<Entity, _>` (`PhysicsRuntime.bindings`,
   `soft_bindings`) в снапшоте и в digest идёт по каноническому ключу
   `(EntityId, Generation)`. `HashMap` использует `RandomState`, порядок
   итерации между процессами разный; `Ord` у `Entity` сейчас нет, ключ
   задаёт срез снапшота явно.

### 4.7. Транспорт: QUIC (quinn) нативно, вариант для wasm

Источник: [quinn](https://github.com/quinn-rs/quinn).

- **v1 нативно:** QUIC через `quinn`, крейт `crates/net-quic` (§4.5).
  Причины: один порт, TLS из коробки, мультиплексирование (ввод —
  надёжный поток, digest — ненадёжные датаграммы), миграция соединений.
  Отдельный raw UDP не нужен (так уже записано в `PLAN.md` §l).
- **wasm:** QUIC в браузере недоступен. WebTransport-клиент садится на
  тот же `Transport` (тот же поток `InputBundle`, другой драйвер байтов)
  и живёт в слое adapters, без границы `Send`. Сам бэкенд — за пределами
  v1 (`PLAN.md` §l). Трейт это уже учитывает, ядро ради wasm не переписывается.
- **В v1 не делаем:** WebRTC, raw UDP, MMO-репликацию, серверный авторитет
  над состоянием.

---

## 5. Порядок срезов (с обоснованием)

Порядок: **L2 (запись и replay) → снапшот всего мира → L0 → L1**.

Снапшот — отдельный срез со своими задачами и гейтом (§6.2). У среза
один критерий и один гейт, из него можно завести отдельную issue.

1. **L2 — запись и `replay --verify` — первым.**
   Причина: replay проверяет, что журнал вводов вообще воспроизводит мир,
   и для этого не нужен сокет. Если локальный replay расходится, сеть это
   только спрячет. С первого среза есть файл для баг-репорта.
   Первые задачи среза — не кодек, а перенос ввода на `Tick` (§6.1):
   без этого критерий «10 минут → тот же digest» недостижим.
2. **Снапшот всего мира — вторым.**
   Причина: save/load — самая рискованная часть (размер, канонический
   порядок, аллокатор, `FixedTime` целиком). Делать её до сокетов значит
   прийти к L0 с уже проверенным откатом состояния. Гейт — локальная
   процедура `SyncTestSession` (§6.2), без сети.
3. **L0 — третьим.**
   Причина: журнал и снапшот уже есть. L0 добавляет приём arrivals,
   раскладку по `Tick` и `LoopbackTransport`. Тест детерминированный:
   один и тот же журнал при разном порядке пакетов.
4. **L1 — последним.**
   Причина: L1 складывает готовые части (снапшот + журнал + digest)
   в политику окна, предсказания и остановки на десинке. Это самый
   трудный в отладке слой, поэтому он идёт, когда всё под ним уже
   проверяется.

Вариант «сначала L0» отвергнут: сеть без replay и без save/load даёт
десинки, которые нечем поймать и нечем повторить. Каждый следующий срез
стоит на гейте предыдущего.

---

## 6. Задачи, критерии, тесты и гейты по срезам

Каждый срез — независимо мержимая issue со своим гейтом
(урок Amethyst, §1.5: без рерайтов ядра).

### 6.1. Срез L2 — запись и replay (первый)

Предусловия, без которых критерий «10 минут → побитово тот же digest»
недостижим. Это первые задачи среза, не хвост:

1. Перенос потребления ввода и интеграции намерения на fixed-путь
   и сэмпл на `Tick` (§2.5): `player_input` в `fixed_schedule` перед
   `physics_push`, ресурс `TickInput`, `transform_update` не пишет
   авторитетную позу. Сетевой прогон и replay шагают тики по журналу,
   не через `Instant` / `begin_frame` (§2.6).
2. Детерминированный старт. В репозитории нет `.ron`-сцены с компонентом
   `Player` (он объявлен в `crates/gameplay/src/lib.rs` и создаётся в
   тестах этого файла). Гейт собирает сцену в тесте:
   - физика — `scene()` из `crates/physics/tests/world_snapshot.rs`
     (пол, ящики на ball joint, сфера с порогом силы, SI-Single);
   - плюс одна сущность с `Player` и скриптованный `InputBundle` на тик,
     не живой `InputState`.
   Старт: свежий `EntityAllocator` (`EntityAllocator::new`) и загрузка
   одних и тех же байт сцены одним и тем же путём. Заголовок `.ornis-rec`
   хранит это стартовое состояние аллокатора (§4.3), чтобы replay не
   зависел от того, что успели выделить до записи.
3. Крейт `crates/net` в `[workspace].members` и в `rustqual.toml`
   (слой `domain`, ключ `ornis_net`, §4.5). В этом срезе крейт умеет
   запись, журнал и digest. `quinn` и `crates/net-quic` здесь не появляются.

Дальше по файлам:

4. Формат `.ornis-rec` (§4.3): header / frames / footer, версия, бинарный кодек.
   Модуль — `crates/net` (запись без сокетов).
5. Сериализация `InputBundle`: канонический порядок пиров, дискретные
   действия из §2.5. Опора на смысл `InputMap`, не на
   `InputState::apply_snapshot` (тот остаётся кадрированным входом
   браузера и winit).
6. CLI `record` / `replay --verify`. Место бинаря решает срез; кандидат —
   существующий runner или xtask, без нового сокета.
7. Per-tick digest v0 по §4.2 п.3 (FNV-1a 64, `to_bits`, нормализация
   `-0.0`, запрет NaN).

Критерии готовности:

- Запись сессии на гейт-сцене (пункт 2) открывается replay без сети и без
  wall-clock и даёт побитово тот же digest-лог. Длина прогона для критерия
  «длинная сессия» — 10 минут симулированного времени (36 000 тиков по 1/60 с),
  не 10 минут стенных часов: иначе в критерий снова попадает `Instant`.
- `replay --verify` возвращает ненулевой код, если подменить один байт ввода.
- Формат версионирован: чтение будущей версии — явная ошибка, не тихий дрейф.
- Два прогона с разным разбиением тех же тиков по кадрам (разные дельты
  `run_frame`, один журнал) дают один digest.

Тесты и гейт:

- Юнит: round-trip кодека (пустой ввод, полный ввод, граничные тики).
- Интеграционный: «записал → проиграл → digest совпал» на сцене из пункта 2.
- Негативный: подмена ввода или версии → явная ошибка.
- Гейт PR: `cargo test -p ornis-net` и clippy/fmt по `AGENTS.md`.
  `determinism-nightly.yml` этот тест не запускает (он гоняет только
  `ornis-physics`, §2.2), поэтому `replay --verify` обязан быть в PR-CI
  среза, а не «и так покроется ночью». Ночной гейт физики остаётся зелёным
  отдельно; расширять его на `ornis-net` можно, но это не замена PR-тесту.

### 6.2. Срез «снапшот всего мира» (второй)

Задачи:

1. `WorldSnapshot` (`crates/physics/src/snapshot.rs`) остаётся в границах
   §4.4. Чужой солвер и `RoutingKind::Islands` по-прежнему
   `SnapshotError::Unsupported`.
2. Снапшот ECS: компоненты через `ComponentRegistry`
   (`crates/core/src/registry.rs`), `Entity` как `(id, generation)` (§4.6).
3. Состояние `EntityAllocator`: `next_id`, `free_list`, `generations`
   (`crates/core/src/entity.rs`).
4. `PhysicsRuntime.bindings` и `soft_bindings`
   (`crates/app/src/physics_runtime.rs`). Для сетевой сцены
   `soft_bindings` пуст (запрет `SoftBody`, §4.4); непустая карта — ошибка
   загрузки, не пропуск поля. Итерация и запись обеих карт — по ключу
   `(EntityId, Generation)`, потому что `HashMap` + `RandomState` даёт
   разный порядок в разных процессах.
5. Ресурс `FixedTime` целиком, не только `tick`: `delta_seconds`,
   `accumulator_seconds`, `elapsed_seconds`, `tick`, `steps_this_frame`,
   `current_step`, `alpha`, `dropped_seconds`, `max_steps_per_frame`
   (`crates/core/src/engine.rs`). `alpha` функция остатка и шага, но
   снимок между шагами кадра без `current_step` и аккумулятора не
   восстановить. Сида RNG в снимке нет (§2.9).
6. `save(tick) / load(tick)` целиком или `Err` (§1.8 п.1). Реализация
   порта `WorldCheckpoint` — в `ornis-app` (§4.5).

Критерии готовности:

- `save → N тиков → load → те же N тиков` даёт тот же digest, что
  непрерывный прогон, без сокета.
- Сцена с GPU-солвером, не-SI режимом, `RoutingKind::Islands` или
  `SoftBody` не загружается как сетевая: понятная ошибка, не десинк.
- Размер снапшота и время save/load записаны в документ среза. Из них
  считается окно L1.

Тесты и гейт:

- Юнит: round-trip каждого куска снимка, включая аллокатор и порядок ключей
  `bindings` (два процесса с разным `RandomState` дают один байтовый снимок).
- Интеграционный: save/load на сцене `scene()` из
  `crates/physics/tests/world_snapshot.rs` и на гейт-сцене L2 с `Player`.
- Гейт в духе ggrs `SyncTestSession`: на каждом тике сохранить checksum,
  откатиться на N тиков назад, пересимулировать, сравнить checksum с
  первым прогоном. N меньше окна. Сохранять каждый тик окна (sparse saving
  этот гейт не проходит, так же как в ggrs). Зависимости на крейт ggrs нет.
- Плюс отсутствие регресса существующих тестов `world_snapshot`.

### 6.3. Срез L0 — delivery model (третий)

Задачи:

1. `Transport` и `LoopbackTransport` в `crates/net` (§4.5). Сокетов в срезе нет.
2. `NetAdapter` в `ornis-app`: раз в кадр забирает `Datagram`, раскладывает
   по `Tick → InputBundle`; в начале каждого fixed-шага публикует `TickInput`
   этого тика. Точка входа — новая система `fixed_schedule` перед
   `player_input`, не хвост `transform_update` и не `MutationBus::drain`.
   `InputState::apply_snapshot` остаётся тем, чем он является: заменой
   кадрового снимка из браузера (`GameApp::apply_browser_input`,
   `apply_browser_input` в `session.rs`). Сэмпл в `Stage::Input` читает
   уже этот снимок и копирует дискретные действия в слот локального пира.
3. Журнал тиков пишется форматом `.ornis-rec` (срез L2). Порядок прибытия
   в этот файл не попадает (§4.1, §4.3).
4. Продвижение тиков записи не зависит от расписания arrivals и от `Instant`.

Критерии готовности:

- Два пира на `LoopbackTransport` с задержкой и перемешиванием пакетов
  дают тот же digest, что локальный прогон того же журнала.
- Тик не видит неполный `InputBundle` (тест на инвариант).
- Один журнал тиков, прогнанный двумя разными последовательностями
  `run_frame` (разные wall-дельты, те же тики), даёт один digest.
  `player_input` в этом прогоне не читает `Time` и не читает `InputState`.

Тесты и гейт:

- Имитация сети поверх `LoopbackTransport`: задержка, дубли, перестановка.
- Паритет «loopback против локального журнала» по digest.
- Гейт: `cargo test -p ornis-net` и тесты `ornis-app`, которые покрывают
  адаптер; clippy/fmt; confluence физики зелёный.

### 6.4. Срез L1 — rollback (последний)

Задачи:

1. Политика `Rollback` в `crates/net`: окно (параметр сессии), предсказание
   опоздавшего ввода, `load` + повторный `advance`. Save/load мира — через
   `WorldCheckpoint` из среза снапшота, не второй реализацией в `net`.
2. Обмен digest (канал `UnreliableDigests`), сравнение, переход в
   `Desynced(tick, a, b)`, остановка сессии и локальный дамп `.ornis-rec`
   плюс снапшот (§4.2). Запроса состояния по проводу нет.
3. Параметр `input_delay_ticks` с дефолтом по замерам.
4. `crates/net-quic` (`quinn`) — этим срезом или следующим, отдельным PR.
   Пока его нет, loopback уже покрывает логику. Регистрация крейта в
   workspace и в слое `adapters` — в том же PR, что и крейт (§4.5).

Критерии готовности:

- Потеря или задержка ввода на loopback приводит к откату, и digest после
  пересимуляции совпадает с эталоном.
- Подмена digest ловится на тике расхождения. Сессия переходит в
  `Desynced` с номером тика и обеими суммами и останавливается.
- Окно отката покрывает 99-й перцентиль измеренной задержки тестовой сети.
  Число фиксируется в issue по замерам.

Тесты и гейт:

- Хаос-тест: случайные задержки и потери на loopback, сверка digest.
- Инъекция десинка: порча состояния одного пира → детект на ожидаемом тике,
  без попытки догнать чужим снапшотом.
- Замер времени пересимуляции окна (бюджет кадра не превышен).
- Гейт: полный набор тестов среза и зелёный ночной determinism физики.
  PR — draft, пока Юра явно не примет.

---

## 7. Риски и открытые вопросы для Юры

Риски:

1. **Размер снапшота мира.** Полный ECS и физика могут не влезть в окно
   на несколько секунд тиков. Митигация: замер в §6.2 до фиксации окна.
   Дельта-снапшоты или урезанный набор компонентов — уже другой дизайн,
   отдельная issue.
2. **Канонизация float.** На одной архитектуре digest берёт сырые
   `f32::to_bits` (§4.2): квантование спрятало бы десинк. Отдельно
   нормализуется только `-0.0`, NaN запрещён. Кросс-архитектурная
   канонизация — не этот риск, она вне v1 (§4.4).
3. **Кросс-архитектура.** Канонические снапшоты x86_64 и aarch64 разные.
   v1 — одна архитектура у пиров сессии (§4.4).
4. **Зависимость `quinn`.** Тяжёлая зависимость с транзитивами; гейт
   `cargo outdated --exit-code 1` (`AGENTS.md`) может покраснеть.
   Митигация: сокеты в отдельном крейте `crates/net-quic` и отдельным шагом.
   Loopback покрывает логику без `quinn`. Тот же гейт — причина, почему
   digest не тянет `xxhash` (§4.2).
5. **Расползание скоупа.** Репликация и RPC «по пути» запрещены
   (`PLAN.md` §l, §1.5, §3). Такие предложения — отдельные issue за v1.

Открытые вопросы:

1. Доступ к Balaur (`docs/PLAN-networking.md`, rollback-модуль,
   панель Recordings): дать доступ или подтвердить, что проектируем без них?
2. Дефолт `input_delay_ticks` и размер окна — после замеров §6.4 или
   нужен бюджет уже сейчас (например, окно в 120 тиков)?
3. QUIC в срезе L1 или отдельным срезом после L1 на loopback?
4. Формат `.ornis-rec` — отдельный файл в `docs/` или достаточно §4.3?
5. Нужен ли в v1 релей для пиров за NAT, или достаточно прямого соединения
   в одной сети?
6. Подтвердить ограничение v1 «одна архитектура в сессии»
   (x86_64 с x86_64, aarch64 с aarch64).
7. Ресинк сессии чужим состоянием после `Desynced` — за v1. В v1 сессия
   останавливается и пишет локальный дамп (§4.2). Нужен ли этому отдельный
   будущий срез, или десинк всегда заканчивает сессию?
