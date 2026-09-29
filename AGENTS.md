# AGENTS.md — правила для контрибьюторов и агентов

## Документация кода

- **Новый публичный API — только с `///`.** Во всех крейтах включён
  `#![warn(missing_docs)]`; не добавляйте предупреждений.
- Новый файл начинайте с `//!` — что это и зачем, одним абзацем.
- Стиль: краткое поведенческое описание (что делает, инварианты,
  ограничения), а не пересказ сигнатуры. `# Safety` для unsafe,
  `# Errors` для fallible API.
- Язык — как у соседнего кода в том же файле.

## Качество

- Единая точка входа — `cargo xtask quality` (локально и в CI одно и то же).
- Перед коммитом минимум: `cargo test --workspace` и
  `cargo clippy --all-targets -- -D warnings`.
- Перед пушем — предпролёт по осям CI (иначе «локально зелёное, в CI
  красное»): точечные проверки крейта НЕ эквивалентны гейту, гейт шире
  сразу по трём осям — стадии × фичи × платформа. Обязательно:
  `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets
  -- -D warnings`, тесты тронутых крейтов с теми же фичами, что
  CI-шарды (`--features gpu` для физики — шейдерные гейты живут только
  там), `rustqual --compare baseline.json --fail-on-regression --no-fail`
  и `cargo outdated --workspace --exit-code 1` (hard gate: зависимости
  обязаны быть latest — иначе `cargo update`). Цена ~5–15 минут против
  нескольких итераций «пуш → красный → чини». Тяжёлые крейты
  (wasm/physics) проверять через CI, но компилируемость их таргетов
  (`--all-targets`, wasm32) обязательна до пуша.
- Complexity-гейт: `rustqual` против `baseline.json` (ratchet: `rustqual --compare baseline.json --fail-on-regression --no-fail`). Рост сложности — осознанно: обновить baseline точечно (`rustqual --save-baseline baseline.json`), с проверкой `rustqual` Score не регрессирует.

## Границы изменений

- Минимальный дифф: не трогайте файлы вне задачи; параллельные агенты
  могут работать в соседних крейтах.
- Мёртвый/«ещё не применённый» код не удалять без решения владельца —
  помечать в отчёте.
- При расхождении документации с кодом верить коду и чинить документацию.

## Проектирование на уровне типов

- Инварианты — в типах, а не в комментариях: newtype-обёртки
  (`BodyHandle`, `EntityId`, `Meters`, `Seconds`, `LinearRgb`,
  `UnitVec3`, `PositiveF32`, `Clamped01`), `enum` вместо `bool`/`u32`-флагов
  (`BodyRole`, `ExecMode`, `CompositeMode`), `Result` с `thiserror`
  вместо `String`/`Option`/`bool` для ошибок.
- GPU-контракт — подмена типа в DSL: Rust хранит `MaterialIdx`/`TextureHandle`,
  `#[derive(WgslStruct)]`/`WgslInterface` транслируют их в стандартные `u32`
  (прецедент — `GpuBool` → `u32`); `repr(transparent)` + `Pod` дают нулевую
  стоимость на CPU, layout проверяется `offset_of!`/`size_of` + naga.
- Фазы — typestate (`World<Building/Running>`, `Engine<Building/Running>`,
  `GameWorld<Authoritative/Replica>`); `schedule_mut` — только построение,
  транспорт (`/api/*`, WASM) остаётся `u64`/`String`-совместимым.

## Документы

- `README.md` — текущее состояние (верифицировано по коду).
- `PLAN.md` — план реализации. `IDEAS.md` — архитектурные идеи.
- `PROJECT_REVIEW.md` — текущие ограничения и активный план работ.
- Меняете поведение/статусы — синхронизируйте README в том же коммите.

## Cursor Cloud specific instructions

- Образ уже содержит Rust из `rust-toolchain.toml` (сейчас 1.97) и таргет
  `wasm32-unknown-unknown`. Поверх него ставятся `libasound2-dev`,
  `libudev-dev`, `pkg-config`, `mesa-vulkan-drivers` (lavapipe: без этого
  ICD нативные wgpu-тесты не находят адаптер), `mold` и бинарь `wasm-pack`.
- Редактор — `cargo xtask editor`, затем http://127.0.0.1:3420.
  Живое состояние: `GET /api/status` и `GET /api/scene`. `POST /api/command`
  принимает JSON только с `Content-Type: application/json` и
  `Origin: http://127.0.0.1:3420` (иначе 415/403).
- `RUSTFLAGS=-C link-arg=-fuse-ld=mold` — только нативные таргеты. Тот же
  флаг на `wasm32` ломает линковку; wasm-сборку (`wasm-pack`, `wasm-check`)
  гоняйте без него. `.cargo/config.toml` ограничивает `jobs = 2`.
- Звуковой карты в VM нет: поток cpal пишет ошибки ALSA и на редактор это
  не влияет. У Chrome в этой VM нет WebGPU, поэтому 3D-viewport может
  остаться пустым; оболочка редактора и HTTP API при этом работают.
  Нативные GPU-тесты идут через lavapipe, не через браузер.
