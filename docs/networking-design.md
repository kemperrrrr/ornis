# Дизайн мультиплеера ornis (срез L-design)

Трек net. Код этим документом не пишется.
Основание: `PLAN.md`, раздел «l. Мультиплеер».
Утверждения о коде сверены с master `f5f8947` (ревью PR #51 и PR #138, issue #108).
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

Источники: пост [Amethyst — Starting fresh](https://amethyst.rs/posts/amethyst--starting-fresh)
(октябрь 2021), репозиторий [amethyst/amethyst](https://github.com/amethyst/amethyst)
(заархивирован владельцем 18.04.2022), список релиза
[0.16](https://github.com/amethyst/amethyst/issues/2308).

В октябре 2021 разработку официально остановили: и стабильную 0.15, и
готовящуюся 0.16. В посте 0.16 названа Legion-версией — это переход
ECS со `specs` на `legion`. В чеклисте issue 2308 пункты «Merge Legion
into master» и «Release & Publish 0.16» так и не закрыты, релиза 0.16 нет.
18.04.2022 репозиторий переведён в архив, только чтение.

- **Берём (единственное):** ECS-организацию логики и data-driven подход —
  у ornis это уже есть (`crates/core`, `Schedule`, `ComponentRegistry`).
- **Избегаем:** бесконечный рерайт ядра вместо срезов с релизом. Прямое
  следствие: каждый срез L0/L1/L2 и срез снапшота мержится отдельно, со
  своим гейтом (§6). Планировщик, `System::run` и аккумулятор кадра ради
  сети не переписываем. Два заранее названных метода на
  `Engine<Running>` всё же нужны (§6.1): `run_fixed_step` шагает один
  тик журнала, `run_variable_frame` проводит кадр рендера и ввода без
  fixed-цикла. Одного `run_fixed_step` мало живому раннеру с записью и
  L0: публичный `run_frame` вместе с variable-стадиями сам исполняет
  fixed-шаги из аккумулятора. Это не рерайт планировщика.
- **Избегаем:** расширение скоупа v1 (MMO-репликация, RPC, WebRTC) —
  см. «вне скоупа» в `PLAN.md` §l и §3.

### 1.6. Godot — КОНТРАСТ, вне скоупа v1

Источники: [High-level multiplayer](https://docs.godotengine.org/en/stable/tutorials/networking/high_level_multiplayer.html)
(Godot 4: `MultiplayerAPI`, авторитет узла, `MultiplayerSynchronizer`
для репликации свойств, RPC).

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
   не наоборот и не от `ornis-app` (§4.5). `player_input` читает ресурс
   `TickInput`, объявленный в `ornis-input`, а не в `crates/net`.
   `ornis-gameplay` от `ornis-net` не зависит: иначе доменный геймплей
   потянул бы транспорт. `PeerInput` лежит там же. Хост (`ornis-app`)
   копирует в ресурс уже упорядоченные действия; `PeerId` и `InputBundle`
   остаются в `crates/net` (§4.5).

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
   `player_input` переезжает в `fixed_schedule` и читает `TickInput`
   этого тика. Запись и replay двигают мир новым `Engine::run_fixed_step`,
   по одному тику журнала, не через `Instant` и не через `begin_frame`.
   Гейт: два прогона одного журнала дают один digest-лог, каким бы ни
   было стенное расписание кадров.

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
- `FixedTime::begin_frame` и `FixedTime::start_step` приватные (`fn`, не
  `pub`, `crates/core/src/engine.rs`). Снаружи их не вызвать. Единственный
  публичный прогон кадра — `Engine::run_frame` / `run_frame_secs`; число
  тиков внутри выбирает `begin_frame` из переданной дельты. Раздельных
  публичных входов сейчас нет; их контракт задаёт §6.1
  (`Engine::run_fixed_step` и `Engine::run_variable_frame`).
  `begin_frame` ограничивает аккумулятор сверху
  `(max_steps + 1) × delta_seconds`: один лишний шаг держится как
  hitch-буфер и влияет на число шагов уже следующего кадра. Исполняется
  не больше `max_steps` (8); лишние шаги сверх этого капа списываются в
  `dropped_seconds`. `start_step` увеличивает `tick` на 1 и вычитает
  `delta_seconds` из аккумулятора. Он ожидает, что аккумулятор уже не
  меньше шага (`debug_assert`, `crates/core/src/engine.rs`); в release
  отдельной ветки «не вычитать» нет, разность зажата
  `(accumulator_seconds - delta_seconds).max(0.0)`. Адаптер сети и
  replay эти методы не зовут. `alpha()` — дробный остаток аккумулятора для интерполяции
  картинки; комментарий у метода прямо говорит, что fixed-системы
  должны брать `delta_seconds`, а не `alpha`.
- Поставить систему в fixed-расписание можно из любого крейта, у которого
  есть `&mut Engine`: `Engine::fixed_schedule_mut` публичный.
  `GameplayPlugin::install` в `ornis-gameplay` уже так и делает.
  Ограничение «fixed-систему вставляет только `ornis-app`» неверно.
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
- Тест: `crates/physics/tests/world_snapshot.rs`. Приватная `fn scene()`
  собирает голый `ornis_physics::Engine`: пол, ящики, ball joint через
  `add_joint`, сфера с порогом силы контакта. ECS, `PhysicsRuntime` и
  `Player` там нет, из `ornis-app` и из будущего `crates/net` функцию
  не импортировать. Снимок на шаге 90, продолжение 60 шагов, сравнение
  бит float через `body_bits`. Это регресс солвера и образец раскладки
  тел, не сцена гейта L2 (§6.1).
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
  в этом кадре `player_input` ещё не бежал. Все fixed-шаги кадра делят
  один и тот же `Velocity`. Через живые хосты их не больше примерно шести:
  `clamp_frame_dt` режет дельту до 0.1 с раньше, чем `begin_frame` (§2.6).
  Кап 8 — только внутри `begin_frame`, если в `run_frame` попала дельта
  без этого клампа.
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

1. Сэмпл локального ввода снимает хост, не система `Stage::Input`.
   Это расписание есть и в `run_frame` выполняется до fixed-цикла, но
   запись и replay в `run_frame` не входят (§6.1), поэтому система там
   на шаге журнала не вызовется. Хост читает `InputState` напрямую и
   складывает дискретные действия `InputMap` (`MOVE_FORWARD` / `BACK` /
   `LEFT` / `RIGHT`). Внутри fixed-шага `InputState` больше не читают.
   Непрерывные поля (`pointer_delta`, `wheel_delta`) в `InputBundle` v1
   не входят: их потребители — камера и прочий variable-путь. Если такой
   оси понадобится симуляция, её квантуют до записи в журнал (§1.3).
2. Перед каждым тиком хост, у которого есть `&mut Engine`, кладёт в ресурс
   `TickInput` (`ornis-input`) упорядоченный `Vec<PeerInput>` этого тика
   и вызывает один шаг (§6.1). Журнал хранит `InputBundle` (`crates/net`);
   в ресурс бандл не кладётся: `ornis-input` — foundation и не называет
   типы domain. В срезе L2 это ещё не `NetAdapter` (он появляется в L0):
   при записи — локальный сэмплер, при `replay` — драйвер журнала.
   Удержанные действия повторяются на каждый тик (флаг repeatable у
   Quantum, повтор последнего ввода у lightyear). Одноразовые края, если
   когда-нибудь войдут в бандл, занимают ровно один тик: сегодня
   `clear_frame_transients` привязан к кадру, не к тику.
3. `player_input` переезжает в `fixed_schedule` (`Stage::Gameplay`) и
   исполняется перед `velocity_to_body`. Читает `TickInput`, пишет
   `Velocity`. Сам ресурс он не пишет: система получает только
   `&Resources` (§6.1). `TickInput.tick` — это `u64`, равный
   `FixedTime::tick()`, который системы видят уже внутри шага.
   Нет ресурса или числа не сходятся — `player_input` падает явно и не
   подставляет пустой ввод. Хост после шага снимает ресурс
   (`World::remove`, `crates/core/src/world.rs`). Слот ввода — не «все
   `Player` подряд». Сейчас `Player` — пустой маркер, и `player_input`
   пишет `Velocity` каждой такой сущности (`crates/gameplay/src/lib.rs`).
   Срез добавляет `Player.slot: u32`: это индекс в `TickInput.players`,
   позиция пира в каноническом порядке, без `PeerId` на сущности.
   Скорость пишется только сущности с этим слотом. Два маркера с одним
   слотом или индекс длиннее вектора — явная ошибка. `physics_push` остаётся
   на `FixedTime::delta()`
   и интегрирует уже записанную скорость; в солвер намерение доводит
   мост, не `physics_push`.
   `install_gameplay_physics_bridge` ставит `VelocityToBodySystem`
   через `prepend_system`, то есть первым в fixed-расписании
   (`crates/app/src/lib.rs`). Явное ребро режет уровни только вперёд по
   регистрации: `try_order_before("player_input", "velocity_to_body")`
   после обычного `add_system` — это `OrderError::BackwardEdge`,
   `Schedule::order_before` такое ребро игнорирует (тест
   `explicit_ordering_rejects_backward_direction` в
   `crates/core/src/schedule.rs`). Конфликт на `Velocity` тогда тоже
   идёт по регистрации, и мост читает скорость прошлого тика. Срез
   регистрирует `player_input` тем же `prepend_system` уже после моста:
   индекс `player_input` меньше, чем у `velocity_to_body`, RaW по
   `Velocity` держит порядок `player_input` → `velocity_to_body` →
   `physics_sync_in`.
4. `transform_update` остаётся в `PostFrame` и не пишет авторитетные
   `Position` и `Velocity`. Комментарий про networked input в этой функции
   снимается, когда срез L2 переносит потребление ввода. Ни сэмплер L2,
   ни `NetAdapter` сюда не подключаются (§6.3).

### 2.6. `runner`, сессия редактора и wall-clock

Два хоста меряют `Instant` и кормят им кадр:

- `GameApp::render_frame` (`crates/runner/src/lib.rs`) хранит
  `GameContext.last_frame: Instant`, берёт `elapsed()`, тут же пишет
  новый `Instant::now()` и вызывает
  `GameWorld::frame_secs(clamp_frame_dt(elapsed))`.
- `editor_world_loop` (`crates/app/src/session.rs`) так же хранит
  `last_tick: Instant` и передаёт `clamp_frame_dt(last_tick.elapsed())`
  в `EditorSession::tick_secs`.

При длинном кадре хост сначала режет дельту до 0.1 с. `clamp_frame_dt`
(`crates/app/src/session.rs`) ограничивает замер константой `MAX_FRAME_DT`;
комментарий у константы и у `GameApp::render_frame` называет это «не больше
~6 fixed-шагов» при 1/60. Отрезанное стенное время пропадает молча и в
`dropped_seconds` не попадает. `GameWorld::frame_secs` передаёт уже
обрезанную дельту в `Engine::run_frame_secs`, а тот — в приватный
`begin_frame`. Кап `DEFAULT_MAX_FIXED_STEPS_PER_FRAME` (8) и
`dropped_seconds` — вторая линия, внутри `begin_frame`. Они срабатывают
только если в `run_frame` попала дельта без этого клампа. Вывод тот же:
два пира с разным FPS выполняют разное число тиков за один и тот же
стенный интервал, потому что число шагов считает кадр, а не журнал.

Это нормальный аккумулятор для локальной картинки (та же роль, что
`Time<Virtual>` у Bevy: сколько раз звать fixed-расписание в этом кадре).
Для записи и для пиров источником тиков он быть не может: пропущенный
из-за hitch тик на одной машине — уже другой журнал.

Вывод: wall-clock законно живёт на границе кадра и только отвечает на
вопрос «сколько локальных шагов успеть нарисовать». Сетевой и
записываемый прогон на этот вопрос не опирается. Он продвигает ровно
один `Tick` на один записанный `InputBundle` методом `run_fixed_step`
(§6.1), с дельтой `FixedTime::delta_seconds()`, без `Instant`, без
`clamp_frame_dt` и без `dropped_seconds`. Локальный `run_frame` для
одиночной игры без записи остаётся как есть. Живой раннер, который
пишет журнал и рисует кадр, `run_frame` не зовёт: variable-стадии идут
через `run_variable_frame`, тики — через `run_fixed_step`. Смешивать
эти два входа с `run_frame` на одном `Engine` нельзя (§6.1).

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
- Ресурса RNG в мире нет. Секции `[workspace.dependencies]` в корневом
  `Cargo.toml` нет вообще. Ни один крейт ornis не зависит от `rand`
  напрямую. В `Cargo.lock` `rand` 0.10.3 есть транзитивно: dev-зависимость
  `proptest` (например `crates/core/Cargo.toml`) тянет его сама.
  Псевдослучайность тестов физики собрана без `rand`
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
   этого тика, до `player_input`, а тот — до `velocity_to_body` (§2.5).
   Сколько тиков сделать за кадр, выбирает хост вызовами
   `run_fixed_step`, а не `begin_frame`. Диапазон «от 0 до 8» — свойство
   `run_frame`: столько шагов `begin_frame` может набрать за одну дельту,
   и живые хосты до этого капа не доходят из-за клампа 0.1 с (§2.6).
   «Раз в тик» и «раз в кадр» поэтому разные места.
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
4. Хранилище журнала и адаптера: `Tick → InputBundle`. Бандл полон (по
   слоту на каждого пира); отсутствующий слот помечен `predicted`.
   Системы этот тип не видят. В мир на шаг попадает `TickInput` со всеми
   `PeerInput` бандла, без `PeerId` и без флага предсказания. Флаг
   `Prediction` видят только хост и журнал: симуляция подтверждённый и
   предсказанный ввод не различает, она применяет уже скопированные
   действия.
5. Реплей идёт **без сети**: журнал тиков + сид + версия + старт сцены (§4.3).
6. Атомарность тика: хост кладёт в `TickInput` все `PeerInput` бандла,
   полного или предсказанного. Флаг предсказания остаётся на
   `InputBundle`, в ресурс и в системы он не копируется. Система видит
   уже собранный вектор. «Половин» не бывает (§1.8 п.1).

Ресурс ввода, не `MutationBus`. На время одного тика в мире лежит ресурс
`TickInput` из `ornis-input`: `players: Vec<PeerInput>` в каноническом
порядке пиров и `tick: u64` — номер, который системы увидят внутри шага.
Это не newtype `Tick` и не `PeerId`: `ornis-input` (foundation) на
domain не ссылается. `Prediction` в ресурсе нет, его держит
`InputBundle`. Пишет ресурс хост до шага, не система:
у системы только `&Resources` (§6.1). Хост копирует из бандла одни
`PeerInput`, уже отсортированные по `PeerId`. В L2 хост — локальный
сэмплер при записи и драйвер журнала при `replay`. В L0 тот же ресурс
пишет `NetAdapter`, тем же шагом `run_fixed_step`. `player_input` и
остальные fixed-системы ресурс только читают и про сокет не знают.
`MutationBus::drain()` сюда не входит: после кадра он отдаёт JSON-`Mutation`
редактора (§2.3). Класть туда сетевой ввод значило бы снова привязать тик
к кадру и к `Time::delta_seconds()`.

Локальный сэмпл кадра (клавиши, которые игрок держит прямо сейчас) снимается
раз в кадр, до цикла тиков. Сэмплер повторяет удержание на каждый тик,
который драйвер сейчас исполняет, и кладёт это в слот локального пира
внутри `TickInput`. Это не чтение `InputState` из `physics_push` и не
система внутри `fixed_schedule`.

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

   Лейн `Position` в digest не входит. Его из позы тела переписывает
   `BodyToTransformSystem` (`body_to_transform`,
   `crates/app/src/lib.rs`), а эта система висит на `schedule_mut` —
   `PostFrame`. `run_fixed_step` `PostFrame` не гоняет, поэтому лейн
   не функция тика: `physics_push` пишет туда интеграл `Velocity`, а
   солверную позу туда кладёт только variable-кадр. Тот же запрет у
   `TransformDesc`, его пишет та же система. Снапшот сессии с журналом
   эти два лейна тоже не хранит (§6.2). Поля тела в digest остаются.

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
  `ornis-input`. Не зависит от `quinn`, не зависит от `ornis-app` и не
  зависит от `ornis-gameplay`.
- `crates/net-quic` — слой `adapters`, рядом с `crates/wasm` и
  `editor-backend`. Единственная имплементация `Transport` на `quinn`.
  Появляется не в срезе L2, а вместе с сокетами (§6.4).
- `PeerInput` и `TickInput` живут в `ornis-input` (слой `foundation`,
  ключ `ornis_input`). Крейт сейчас без зависимостей
  (`crates/input/Cargo.toml`): `Tick`, `PeerId` и `InputBundle` в эти
  типы не входят. `ornis-gameplay` уже зависит от `ornis-input` и
  `player_input` читает `TickInput` оттуда. Ребра `ornis-gameplay` →
  `ornis-net` срез не добавляет.

Финальный граф: `ornis-input` (`PeerInput`, `TickInput`) ← `ornis-gameplay` и ← `ornis-net`; `ornis-gameplay` не зависит от `ornis-net`, `ornis-net` не зависит от `ornis-gameplay`; `ornis-app` зависит от обоих и пишет `TickInput`; `ornis-net-quic` (adapters) зависит от `ornis-net` и в срезе L2 не появляется.

Ключи `rustqual.toml`: `ornis_input` = `foundation`, `ornis_gameplay` =
`domain`, `ornis_net` = `domain`, `ornis_app` = `application`,
`ornis_net_quic` = `adapters`. Два крейта `domain` могут зависеть друг
от друга, только если нет цикла; здесь цикла нет, потому что геймплей
про сеть не знает.

Типы, которые читает симуляция, объявляет `ornis-input`:

```rust
// Эскиз ornis-input. Кода нет.
// crates/input/Cargo.toml зависимостей не имеет; Tick и PeerId сюда не входят.

/// Дискретные действия одного пира. Те же слова, что InputMap::MOVE_*.
/// Не сырой InputState: указатель и колесо в тик v1 не входят (§2.5).
pub struct PeerInput {
    pub move_forward: bool,
    pub move_back: bool,
    pub move_left: bool,
    pub move_right: bool,
}

/// Ресурс на время одного fixed-шага. Пишет хост до run_fixed_step,
/// читает player_input. tick — значение FixedTime::tick(), которое
/// системы увидят внутри шага (u64, не newtype Tick: этот крейт на
/// ornis-core не зависит). players — канонический порядок пиров.
pub struct TickInput {
    pub tick: u64,
    pub players: Vec<PeerInput>,
}
```

- Save/load мира в `crates/net` не помещается. `PhysicsRuntime.bindings`
  и `soft_bindings` — приватные `HashMap` внутри `ornis-app`
  (`crates/app/src/physics_runtime.rs`), fixed-расписание физики собирает
  `install_physics` там же. Порт объявляет `crates/net`, реализацию пишет
  `ornis-app` (слой `application` имеет право зависеть от `domain`):

```rust
// Эскиз crates/net. Кода нет. Tick не объявляем второй раз.
pub use ornis_core::Tick;
use ornis_input::PeerInput;

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

/// Полный набор вводов одного тика. Журнал и адаптер хранят его.
/// В мир копируются только PeerInput, по возрастанию PeerId (§4.1).
pub struct InputBundle {
    pub tick: Tick,
    pub peers: Vec<(PeerId, PeerInput, Prediction)>,
}

pub enum Prediction { Confirmed, Predicted }

/// Непрозрачный снимок мира на тик. Формат `bytes` задаёт срез снапшота
/// (§6.2). `crates/net` байты не разбирает.
pub struct Checkpoint {
    pub tick: Tick,
    pub bytes: Vec<u8>,
}

/// Порт снапшота. Реализация — в ornis-app: там PhysicsRuntime.
pub trait WorldCheckpoint {
    type Error;
    fn save(&self) -> Result<Checkpoint, Self::Error>;
    fn load(&mut self, saved: &Checkpoint) -> Result<(), Self::Error>;
}
```

- `LoopbackTransport` — in-process, для тестов без сокетов.
- `NetAdapter` (L0) живёт в `ornis-app`, потому что там `PhysicsRuntime`,
  реализация `WorldCheckpoint` и сборка игрового мира. Не потому, что
  только app умеет регистрировать fixed-системы: `fixed_schedule_mut`
  публичный, им уже пользуется `ornis-gameplay` (§2.1). Адаптер не система
  внутри расписания. Перед каждым `run_fixed_step` он заменяет `TickInput`
  через `world_mut`. Журнал и разбор пакетов он берёт из `crates/net`.
- `Rollback` вызывает `WorldCheckpoint` и `advance` журнала. Политика
  отката — в `crates/net`, конкретный save/load мира — в `ornis-app`.
- Обратной зависимости «симуляция знает про сокет» нет: `player_input`
  читает `TickInput` из `ornis-input`, не `Transport` и не `InputBundle`.
  `ornis-gameplay` от `crates/net` не зависит.

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
(урок Amethyst, §1.5: без рерайта планировщика). Изменения API ядра
названы в задаче 1 среза L2: `run_fixed_step` и `run_variable_frame`.

### 6.1. Срез L2 — запись и replay (первый)

Предусловия, без которых критерий «10 минут → побитово тот же digest»
недостижим. Это первые задачи среза, не хвост:

1. Потиковый ввод и шаг по журналу. Сейчас этого API нет, и три
   обходных пути не годятся. Сравнение и выбор — ниже. Задача среза:
   добавить в `ornis-core` `run_fixed_step` и `run_variable_frame`.
   Журнал гоняется только первым. Живой раннер с записью и L0 на кадр
   зовут второй, без `run_frame` (§ ниже).
2. Детерминированный старт на ECS-сцене в `ornis-app` (состав — ниже,
   это не `scene()` из теста физики). В репозитории нет `.ron` с
   компонентом `Player` (он объявлен в `crates/gameplay/src/lib.rs`).
   Старт: свежий `EntityAllocator` (`EntityAllocator::new`) и одна и та
   же сборка мира. Заголовок `.ornis-rec` хранит стартовое состояние
   аллокатора (§4.3), чтобы replay не зависел от того, что успели
   выделить до записи.

**Как подавать ввод на тик и как шагать журнал.** Ориентир — bevy_ggrs,
не выдуманный цикл. Там плагин сам кладёт `PlayerInputs` и один раз
гоняет `GgrsSchedule`; аккумулятор `Time<Virtual>` кадр отката не
считает. lightyear пишет ввод системой в `FixedPreUpdate`, потому что
система Bevy может менять мир. В ornis система так не может.

Факты текущего API (`crates/core/src/engine.rs`):

- публичный прогон — только `Engine::run_frame(delta)`; число тиков
  выбирает приватный `begin_frame`;
- `start_step` тоже приватный. Он ожидает аккумулятор не меньше шага
  (`debug_assert`); в release вычитание зажато `max(0.0)`, это не
  ранний выход (§2.1);
- `System::run` получает `&Resources`. Комментарий `run_frame` (около
  строк 725–737) прямо говорит: из системы нельзя сделать `get_mut` на
  обычный ресурс, `World::run` держит `&self`. Запись из системы — это
  либо `Mutex`, либо смена `System::run` на `&mut`, ломающая параллельный
  исполнитель уровней.

Три варианта:

- **A. `Mutex`-ресурс и fixed-система, которая читает буфер.** Так уже
  устроен `Mutex<PhysicsRuntime>`. Система в начале `fixed_schedule`
  берёт бандл по `FixedTime::tick_id()`. Ядро не меняется, и писать
  обычный `TickInput` из системы не нужно. Для replay этого мало: сколько
  раз система вызовется, по-прежнему решает `run_frame` и приватный
  аккумулятор. Журнал «один кадр файла = один тик» так не исполнить.
- **B. Replay вызывает `run_frame(DEFAULT_FIXED_DELTA_SECONDS)` и проверяет,
  что `tick` вырос на 1.** Тоже без нового API. Если в аккумуляторе
  остаток `r < delta`, аргумент ровно `delta` превращает его в
  `r + delta`, и цикл `begin_frame` (`crates/core/src/engine.rs`)
  вычитает шаг один раз: остаток снова `r`, шагов ровно один. Ноль
  шагов при дельте ровно `delta` и остатке `r < delta` не получается.
  Два шага здесь возможны только из-за округления `f32` на самой границе
  `r + delta`, не из-за hitch-буфера `(max_steps + 1) × delta`: тот в
  этот вызов не входит. Один шаг — неявное свойство `begin_frame`, не
  контракт журнала. Плюс каждый такой вызов гоняет `PreUpdate`, `Input`
  и `PostFrame`, двигает `Time` и зовёт `clear_frame_transients`
  (`Engine::run_frame`). Replay тогда зависит от variable-систем
  (`MutationTick`, `transform_update`, `body_to_transform`, камера), а
  не только от `fixed_schedule`. `run_frame(0.0)` этот путь тоже не
  чинит: после `accumulator_seconds = 0` он даёт 0 шагов, но ставит
  `Time` в дельту 0 и в `begin_frame` затирает `steps_this_frame` и
  `current_step` нулями. Гейт на этом не стоит.
- **C. Публичный метод одного тика; `TickInput` пишет хост снаружи.**
  Хост уже имеет `&mut Engine` (`world_mut`) и может заменить ресурс до
  запуска расписания. Системы ресурс только читают, `Mutex` и смена
  `System::run` не нужны. Шаг не смотрит на `Instant` и не зовёт
  `begin_frame`. Это тот же разрез, что у bevy_ggrs: ввод кладут до
  расписания, расписание исполняют ровно один раз на тик.

**Выбор — C.** Срезу нужны два метода на `impl Engine<Running>`
(`crates/core/src/engine.rs`, рядом с `run_frame`). Оба вызываются между
кадрами, не из системы: то же ограничение, что у `world_mut`.

Первый:

```rust
pub fn run_fixed_step(&mut self)
```

Аргумента дельты нет: длина шага уже лежит в ресурсе
(`FixedTime::delta_seconds`). Если ресурса нет, метод вставляет
`FixedTime::default()` до расписания, как `run_frame` вставляет его до
`begin_frame` (`crates/core/src/engine.rs`). Иначе `PhysicsStep` молча
выходит (`crates/app/src/physics_runtime.rs`). До запуска расписания
метод переписывает часы одного тика и только потом один раз выполняет
`fixed_schedule`. Системы видят уже новый тик: `FixedTime::tick()`
внутри шага на единицу больше, чем до вызова.

Что метод делает с полями `FixedTime`:

- `tick` — `saturating_add(1)`, как `start_step`;
- `elapsed_seconds` — прибавляет `f64::from(delta_seconds)`;
- `accumulator_seconds` — записывает `0`. Шаг не вычитается, остаток не
  хранится. Дробь, оставшаяся от предыдущего `run_frame`, этим вызовом
  сбрасывается: шаг журнала не продолжает wall-clock аккумулятор;
- `steps_this_frame` — записывает `1`;
- `current_step` — записывает `1`. Это не `saturating_add` к значению,
  которое оставил `begin_frame`: у потикового входа кадра нет;
- `alpha` — записывает `0`. Это не `update_alpha`: тот ставит
  `(accumulator_seconds / delta_seconds).clamp(0, 1)` и после
  `start_step` равен доле остатка. Здесь остатка нет;
- `delta_seconds`, `dropped_seconds` и `max_steps_per_frame` метод не
  меняет. `dropped_seconds` копит только `begin_frame`.

Метод не вызывает `begin_frame` и не вызывает `start_step`.
`start_step` в debug ожидает аккумулятор не меньше шага и в release
зажимает вычитание `max(0.0)` (§2.1); `alpha` он считает из остатка.
Для журнала это другой путь. Не двигает variable-`Time`, не гоняет
`PreUpdate`, `Input` и `PostFrame`, не вызывает `clear_frame_transients`.

После вызова четыре кадровых поля всегда одни и те же:
`accumulator_seconds = 0`, `steps_this_frame = 1`, `current_step = 1`,
`alpha = 0`. У `run_frame` они зависят от числа шагов кадра и от
остатка `begin_frame` (§2.1): кадр из двух ровных шагов оставляет
`steps_this_frame = 2` и `current_step = 2` при том же `tick`, кадр с
дробным хвостом оставляет ненулевой аккумулятор и `alpha`. Снимок
приводит эти поля к кортежу `run_fixed_step` (§6.2), поэтому байты
`FixedTime` не зависят от того, каким путём прошёл тик.

`begin_frame` и `start_step` остаются приватными. `run_frame` остаётся
входом локальной игры без записи. На одном `Engine` его не смешивают с
`run_fixed_step`: остаток аккумулятора после `run_frame` добавил бы
тики, которых в журнале нет. После `run_fixed_step` аккумулятор уже
0, поэтому `run_frame(0.0)` даёт 0 шагов и не является пустым кадром:
`Time` получает дельту 0, а `begin_frame` заново пишет
`steps_this_frame = 0`, `current_step = 0` и пересчитывает `alpha`.
Канонические единицы после тика затираются. Если аккумулятор ещё не
меньше шага, тот же вызов исполнит лишний тик. `System::run` не меняется. Это не перенос
планировщика в `crates/net` и не вторая модель систем. Граница из §1.5
такая: планировщик и `System::run` не трогаем; добавляются два метода,
названные здесь, и ничего сверх.

Второй метод — кадр без симуляции. Живому раннеру с записью и
`NetAdapter` раз в кадр нужны `PreUpdate`, `Input` и `PostFrame`
(рендер, камера, `body_to_transform`, `MutationTick`). Звать ради этого
`run_frame` нельзя: он сам наберёт fixed-шаги поверх
`run_fixed_step`.

```rust
pub fn run_variable_frame(&mut self, delta_seconds: f32)
```

Дельта конечная и неотрицательная, как у `run_frame`. Метод публикует
`Time` (вставляет ресурс, если его нет, и двигает той же дельтой),
гоняет `PreUpdate` → `Input` → `PostFrame` и при наличии `InputState`
зовёт `clear_frame_transients`. `begin_frame`, `start_step` и
`fixed_schedule` он не вызывает и `FixedTime` не вставляет и не меняет.
Headless-replay L2 этот метод не зовёт: гейт журнала зависит только от
`fixed_schedule`. На кадр живого хоста порядок такой:
`run_variable_frame` один раз, затем `run_fixed_step` столько раз,
сколько тиков этот кадр симулирует. Число выбирает хост, не
`begin_frame`.

Кто заполняет `TickInput`, пока нет `NetAdapter`:

- **Запись.** Локальный сэмплер в хосте L2 (тест или раннер в
  `ornis-app`, не система). Headless-тест зовёт только `run_fixed_step`.
  Раннер, который ещё и рисует кадр, вместо `run_frame` зовёт
  `run_variable_frame`, затем шаги. Перед каждым `run_fixed_step`
  сэмплер берёт дискретные действия (`InputMap`: `MOVE_FORWARD` /
  `BACK` / `LEFT` / `RIGHT`), собирает `InputBundle` для журнала и
  пишет `TickInput` через `world_mut`. Поле `tick` — это
  `FixedTime::tick().saturating_add(1)` до вызова (если ресурса ещё
  нет, база — 0, как у `FixedTime::default`). `InputBundle.tick`
  (`Tick::get`) обязан совпасть с этим `u64`, иначе шаг не начинается.
  Удержание повторяется на каждый следующий тик, пока снимок кадра тот
  же. После шага бандл дописывается в журнал, и хост делает
  `world_mut().remove::<TickInput>()` (`World::remove`). Неснятый
  ресурс со старым `tick` на следующем шаге не повторится молча:
  `player_input` сверяет число с `FixedTime::tick()` и падает.
- **`replay`.** Драйвер журнала читает следующий кадр `.ornis-rec`
  (`InputBundle`), так же заполняет `TickInput` и зовёт только
  `run_fixed_step`. `InputState`, `Instant` и `run_variable_frame` он
  не трогает. После шага ресурс снимается тем же `remove`.
- **L0.** `NetAdapter` подменяет источник бандла. На кадр —
  `run_variable_frame`, на тик — тот же `run_fixed_step` (§6.3).

`player_input` в `fixed_schedule` стоит перед `velocity_to_body` и
только читает `TickInput` из `ornis-input`, пишет `Velocity`.
Регистрация — `prepend_system` после `install_gameplay_physics_bridge`,
не `add_system`: мост уже поставил `velocity_to_body` первым, и
обратное `order_before` планировщик игнорирует (§2.5). Порядок
`player_input` → `velocity_to_body` → `physics_sync_in`.
`physics_push` этот порядок не задаёт. Нет `TickInput` или
`TickInput.tick != FixedTime::tick()` — явная паника, не пустой ввод
и не повтор прошлого вектора. Индекс слота: сегодня `Player` — пустой
маркер, и `player_input` пишет `Velocity` всем таким сущностям.
Срез кладёт на маркер `slot: u32` и пишет скорость только той сущности,
чей `slot` равен индексу в `TickInput.players`. Индекс — позиция в
каноническом порядке `PeerId`, сам `PeerId` геймплей не хранит. Два
`Player` с одним слотом или слот за концом вектора — явная ошибка.
В гейте L2 один `Player` со `slot = 0` и вектор из одного `PeerInput`.
`transform_update` не пишет авторитетную позу (§2.5).

**Гейт-сцена собирается в `ornis-app`, не вызовом `scene()`.**
`scene()` — приватная функция интеграционного теста физики и возвращает
голый `ornis_physics::Engine` (§2.4). Добавить туда `Player` нельзя.

Тест гейта в `ornis-app` (там же, где будет `WorldCheckpoint`) делает так:

- `install_physics` (`crates/app/src/physics_runtime.rs`) — SI, fixed-системы
  sync/step, гравитация как у `scene()` (`Vec3::new(0.0, -9.81, 0.0)`);
- `install_gameplay` (`crates/gameplay/src/lib.rs`) и
  `install_gameplay_physics_bridge` (`crates/app/src/lib.rs`): без моста
  `Velocity`, который пишет `player_input`, не доходит до `RigidBody`.
  После моста `player_input` ставится `prepend_system`: мост уже сделал
  `velocity_to_body` первым, и без этого ввод отстаёт на тик (§2.5);
- сущности с компонентом `RigidBody` в том же духе, что тела `scene()`:
  статический пол, динамические ящики, сфера с порогом силы контакта.
  Это образец раскладки, не вызов функции. Ball joint из `scene()` в гейт
  не переносится: `add_joint` есть только у голого солвера, компонента
  сустава в `ornis-app` нет;
- на одном динамическом теле ещё `Player` со `slot = 0`, `Position` и
  `Velocity`. Вектор `TickInput.players` на этом гейте из одного
  элемента: `players[0]` и есть ввод этой сущности;
- ввод — скриптованный `InputBundle` на тик в журнале; драйвер копирует
  его `PeerInput` в ресурс `TickInput` и зовёт `run_fixed_step`, не
  живой `InputState` и не `run_frame`.

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
- Два прогона одного журнала через `run_fixed_step` (по одному вызову на
  кадр файла, без `run_frame` и без `run_variable_frame`) дают один digest.

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
   Лейны `Position` и `TransformDesc` в байты сессии с журналом не
   входят. Их солверная поза появляется только в `PostFrame`
   (`BodyToTransformSystem`), а `run_fixed_step` эту стадию не гоняет
   (§4.2). Иначе save после одного `run_fixed_step` и save после
   следующего `run_variable_frame` разойдутся при тех же телах.
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
   (`crates/core/src/engine.rs`). Сида RNG в снимке нет (§2.9).
   В байты чекпоинта кадровые поля пишутся не «как лежат в живом
   ресурсе». После `run_frame` живые `accumulator_seconds`,
   `steps_this_frame`, `current_step` и `alpha` — это кадр, в котором
   тик случился (два ровных шага оставляют `current_step = 2`, дробный
   хвост — ненулевой аккумулятор и `alpha`). После `run_fixed_step` те
   же четыре поля всегда `(0, 1, 1, 0)` (§6.1). `save` подставляет этот
   кортеж в копию и пишет `dropped_seconds = 0`, но только когда
   `tick >= 1`. Тик 0, до первого шага, устроен иначе. Свежий
   `FixedTime::default()` и кадр, в котором `begin_frame` не набрал ни
   одного шага, оставляют `tick = 0`, `elapsed_seconds = 0`,
   `steps_this_frame = 0`, `current_step = 0`. Кортеж `(0, 1, 1, 0)`
   для них врал бы, что шаг уже был. Для `tick == 0` `save` пишет
   `accumulator_seconds = 0`, `steps_this_frame = 0`, `current_step = 0`,
   `alpha = 0`, `elapsed_seconds = 0`, `dropped_seconds = 0`. Для
   `tick >= 1` — кортеж после шага. `delta_seconds` и
   `max_steps_per_frame` в обоих случаях копируются как есть;
   `elapsed_seconds` и `tick` для `tick >= 1` тоже, на одном числе шагов
   одной дельты они у обоих путей совпадают. Живой ресурс `save` не
   переписывает: `alpha` ещё нужен интерполяции кадра.
   `dropped_seconds` в идентичность тика не входит — его увеличивает
   только hitch-кап `begin_frame`, `run_fixed_step` его не трогает.
   `load` поднимает ресурс уже в этом каноническом виде. Два снимка
   одного `tick` — после `run_frame` и после `run_fixed_step` — дают
   одни байты `FixedTime`, включая тик 0.
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
- Интеграционный: save/load на гейт-сцене L2 (§6.1). Существующие тесты
  `world_snapshot` остаются регрессом голого солвера; их `scene()` в
  этот гейт не импортируется.
- Гейт в духе ggrs `SyncTestSession`: на каждом тике сохранить checksum,
  откатиться на N тиков назад, пересимулировать, сравнить checksum с
  первым прогоном. N меньше окна. Сохранять каждый тик окна (sparse saving
  этот гейт не проходит, так же как в ggrs). Зависимости на крейт ggrs нет.
- Плюс отсутствие регресса существующих тестов `world_snapshot`.

### 6.3. Срез L0 — delivery model (третий)

Задачи:

1. `Transport` и `LoopbackTransport` в `crates/net` (§4.5). Сокетов в срезе нет.
2. `NetAdapter` в `ornis-app` подменяет сэмплер L2 как источник бандла.
   Раз в кадр зовёт `Engine::run_variable_frame` (§6.1), забирает
   `Datagram` и раскладывает по `Tick → InputBundle` в каноническом
   порядке `PeerId`. На каждый тик этого кадра копирует `PeerInput` в
   `TickInput` (поле `tick` — номер, который шаг покажет системам),
   зовёт `Engine::run_fixed_step` и снимает `TickInput` через
   `World::remove`. `run_frame` адаптер не зовёт. Это код хоста, не
   система `fixed_schedule`: система не может записать ресурс.
   `player_input` только читает уже положенный `TickInput` и сверяет
   `tick` с `FixedTime::tick()`. Не хвост `transform_update` и не
   `MutationBus::drain`.
   `InputState::apply_snapshot` остаётся заменой кадрового снимка из
   браузера (`GameApp::apply_browser_input`, `apply_browser_input` в
   `session.rs`). Локальный слот бандла по-прежнему заполняет сэмплер
   из этого снимка, до шага, как в L2.
3. Журнал тиков пишется форматом `.ornis-rec` (срез L2). Порядок прибытия
   в этот файл не попадает (§4.1, §4.3).
4. Продвижение тиков записи не зависит от расписания arrivals и от `Instant`.

Критерии готовности:

- Два пира на `LoopbackTransport` с задержкой и перемешиванием пакетов
  дают тот же digest, что локальный прогон того же журнала.
- Тик не видит неполный `InputBundle` (тест на инвариант).
- Один журнал, прогнанный дважды через `run_fixed_step`, даёт один digest.
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
