# CPU vs GPU parity — macOS, 2026-09-20

Закрывает частично п.3 активного плана
([`PROJECT_REVIEW.md`](../../PROJECT_REVIEW.md), § «Укрепить GPU-путь»):
проверить реальные GPU-сценарии на macOS, сравнить CPU и GPU по
стабильности, скорости и расхождению результатов, не обещая
bit-identical поведение там, где его нет.

## Методика

Только измерения существующим кодом, новых бенчей и правок кода нет:

- tolerance-тесты `ornis-wgpu-backend` (kernels `scale`/`madd`, CPU↔GPU
  agreement внутри допуска);
- `cargo test -p ornis-physics --features gpu --test gpu_batches`
  (согласование GPU-решателя с CPU-путём и с аналитикой);
- один дешёвый существующий критерий-бенч
  (`solver_bench physics_step/big_stack_32`, release) как CPU-точка
  скорости. Полный criterion-сьют осознанно не гонялся (см. § «Скорость»).

## Машина

- Apple M1, 16 GB RAM, macOS 26.6.2;
- GPU: Apple M1 8-core, Metal 4 (единственный не-fallback backend при
  `force_fallback_adapter: false`; тесты просят HighPerformance-адаптер);
- профиль тестов — debug (`test`), бенч — release (`bench`).

## Стабильность (что прогнано)

| Сьют | Результат | Время |
|---|---|---|
| `cargo test -p ornis-wgpu-backend` (42 lib + 4 integration) | 46/46 ✅ | 55.18 с |
| `cargo test -p ornis-physics --features gpu --test gpu_batches` | 17/17 ✅ | 75.67 с тестов (2м17 с со сборкой) |
| `gpu_solver_tracks_cpu_engine` отдельно, `--nocapture` | ok, без `no wgpu adapter — skipped` | 65.78 с |

Доказательство реального исполнения на GPU (а не honest-skip):
тесты `try_device()`/`create_test_device()` возвращают `None` без
адаптера и тогда тесты — мгновенные no-op; здесь device-тесты шли
десятки секунд каждый. Дополнительно в `auto_lane` CPU-замыкания —
`panic!("must run on GPU")`: прохождение означает, что считал именно
GPU-кернел (результат `x*2+1`, а не CPU `x*5`).

## Расхождение (допуски, все зелёные на Metal)

- `auto_lane::cpu_and_gpu_agree_within_tolerance_not_bit_identical`
  (madd, 64 элемента): поэлементно `|gpu−cpu| ≤ 1e-5·max(|cpu|, 1)`,
  контрольные суммы `≤ 1e-3·max(|sum|, 1)`.
- `gpu_solver_single_contact_matches_analytic` (сфера в статику):
  нормальная скорость погашена (`< 1e-3`), импульс = массе (`±1e-2`).
- `gpu_contact_row_matches_cpu_kernels` (фрикционный контакт,
  GPU vs CPU wide batch): `|dv| < 1e-4`, `|dw| < 1e-4`,
  нормальный импульс `|gpu−cpu| < 1e-4`; поведение: остановка
  нормального сближения, тангенциальный слип `< 1e-3`.
- `gpu_solver_tracks_cpu_engine` (стабильный стек сфер, 60 шагов =
  1 с симуляции): позиции и скорости CPU vs GPU `< 0.05` по всем телам.

## Скорость

- CPU-точка (release, M1): `physics_step/big_stack_32` (33 тела) —
  mean **5.65 мс/шаг**, median **5.28 мс/шаг** (std 0.96 мс).
- GPU-тайминги отдельно не снимались: device-тесты идут в debug
  (60+ с на тест — доминируют upload/download round-trip'ы и debug-ALU,
  а не шейдер), прямое сравнение «GPU-кернел vs CPU-кернел по времени»
  существующим кодом не покрыто (см. «Не покрыто»).
- Полный `solver_bench` не гонялся: это не «дёшево» — одна только
  регистрация criterion выполняет 14 конфигов `settled_body_grid`
  (до 10k тел × 30 шагов прогрева в release) до исполнения выбранного
  бенча. Существующие baseline-цифры — в
  `perf-baseline-2026-08-27.md` / `perf-baseline-2026-09-02.md`.

## Явный отказ от bit-identical

GPU-путь — Jacobi/GS-hybrid и осознанно не bit-identical CPU-пути
(документировано в PLAN, G7, и в именах тестов:
`cpu_and_gpu_agree_within_tolerance_not_bit_identical`,
«device float contraction may differ ±1 ulp per op»). Контракт —
согласие в допуске на стабильных сценах; для реиграемых сцен нужен
CPU-only или отдельный fixed-point путь (остаток G7, не предмет
этого отчёта).

## Что покрыто / что нет

Покрыто: madd/scale agreement CPU↔GPU в допуске; GPU single-contact
vs CPU wide batch и vs аналитика; engine-level settled-state
согласие стабильного стека; WGSL-layout пины и CPU-ступени AVBD
(без девайса); одна CPU-точка скорости.

Не покрыто (честно, не этот заезд): хаотичные/опрокидывающиеся стеки
(гибрид расходится сверх любого жёсткого допуска — зафиксировано в
комментарии к `gpu_solver_tracks_cpu_engine`, проверяется только
стабильный стек); прямое GPU-vs-CPU сравнение времени; device-путь
AVBD (пока только CPU-stub + layout); GPU-масштабирование broadphase;
рендерный GPU-паритет (чужой трек).

## Вывод

п.3 закрыт частично: стабильность (63 device-зависимых проверки
зелёные на реальном Metal-адаптере M1) и расхождение (допуски
1e-5…0.05 по уровням) измерены существующими тестами и зафиксированы
выше; скорость дана одной CPU-точкой, GPU-тайминги и хаотичные сцены —
открытые подпункты.
