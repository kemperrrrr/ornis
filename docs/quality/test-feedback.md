# Быстрый тестовый фидбек: `ornis-physics`

Дефолтный `cargo test -p ornis-physics` гоняет **быстрый subset** (~5 с
тест-рантайма): два тяжёлых детерминизм-гейта помечены `#[ignore]`.
Ассерты и поведение не менялись — только пометки + этот документ.

## Команды

```sh
# Быстро (дефолт, gate): ~5 с тест-рантайма, 337 passed / 4 ignored
# (включая `math_props`: 4 детерминированных AABB/Ray-инварианта,
# переехавших из `ornis-core/tests/property_tests.rs`)
cargo test -p ornis-physics

# Только ignored-набор (2 тяжёлых гейта + 2 regenerate-хелпера)
cargo test -p ornis-physics -- --ignored

# Вообще всё, включая ignored (быстрые + тяжёлые + regenerate)
cargo test -p ornis-physics -- --include-ignored
```

## Что ушло в `#[ignore]` и почему

Оба — потоки-сходимость гейты «1 поток vs 32 потока, бит-в-бит» на сценах
за параллельным порогом (>256 тел): сцена строится дважды, каждый прогон
тащит свой `rayon::ThreadPool`, AVBD-степы дорогие. Вместе давали ~95%
рантайма дефолтного сьюта.

| Тест | Файл | Замер (соло / в сьюте) | Причина |
|---|---|---|---|
| `routed_world_is_bit_identical_with_one_or_many_workers` | `crates/physics/tests/solver_split_invariants.rs` | ~51 с / 64–99 с | 256 боксов, 16 шагов ×2 конфига пулов |
| `avbd_confluence_one_vs_many_threads` | `crates/physics/tests/avbd_engine.rs` | ~37 с / 36–42 с | 265 тел, 80 шагов ×2 конфига пулов |

Остальное всё лёгкое: lib (138 тестов) ~1.4–1.8 с, `si_step` (51 тест)
~0.9–1.6 с, остальные файлы ≤0.8 с каждый. `gpu_batches` в дефолте пуст
(0 тестов — файл под `#![cfg(feature = "gpu")]`, гоняется только в
`test-physics` стадии гейта).

Замеры: Apple Silicon, 8 CPU, дебаг-профиль, параллельно шли чужие
`cargo` (lock-wait и шум CPU) — абсолютные секунды плавают ±30–50%,
соотношение быстро/полно устойчиво.

До/после (`cargo test -p ornis-physics`, зелёный в обоих):

| | wall | сумма `finished in` | итог |
|---|---|---|---|
| До | 2:00.76 | ~111 с | 335 passed / 2 ignored |
| После | 1:34.38* | ~4.8 с (~×23) | 333 passed / 4 ignored |
| `-- --ignored` | 2:34.21 | ~135 с | 4 passed / 0 failed |

\* В wall «после» сидит разовая перекомпиляция двух тронутых таргетов и
ожидание общего cargo-lock; steady-state wall упирается в ~5 с тестов +
спавн 13 бинарников.

## Осторожно: `*_regenerate` в ignored-наборе

`-- --ignored` / `-- --include-ignored` заодно запускают
`avbd_snapshot_regenerate` и `determinism_snapshot_regenerate` — они
**перезаписывают** `crates/physics/tests/data/*.hex`. При неизменном
солвере байты идентичны (проверено: `git status` чист по `tests/data/`
после прогона). Если после полного прогона `git diff` показывает
`tests/data/*.hex` — это сигнал дрейфа солвера/кодогена, смотреть diff
осознанно, не коммитить вслепую.

## sccache локально (опционально, только ускорение сборки)

CI уже использует sccache в shard'ах (`RUSTC_WRAPPER=sccache`,
`CARGO_INCREMENTAL=0`, см. `.github/workflows/quality.yml`). Локально:

```sh
cargo install sccache
export RUSTC_WRAPPER=sccache
export CARGO_INCREMENTAL=0   # нужен для cache hits
```

`.cargo/config.toml` (`jobs = 2`) не трогать — это локальный дефолт
против распухания памяти линковщика; на рантайм тестов не влияет.

## nextest (опционально)

Конфиг: `.config/nextest.toml`. Гейтом остаётся `cargo test`
(`cargo xtask quality --only test` / `--only test-physics`).

```sh
cargo install cargo-nextest
cargo nextest run -p ornis-physics                 # быстрый subset
cargo nextest run -p ornis-physics --profile full  # + ignored-гейты
```

## CI

Воркфлоу не менялся осознанно: `test` (workspace) и `test-physics`
(`--features gpu -- --test-threads=1`, serial из-за общего lavapipe)
гоняют быстрый subset и остаются зелёными. Полный профиль с тяжёлыми
гейтами — локально (`-- --include-ignored`) или `nextest --profile full`.
Известный медленный профиль: `test-physics` идёт в один поток —
долго даже без ignored-гейтов, это отдельная тема (не этот трек).
