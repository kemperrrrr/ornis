# 006: GPU AVBD primal sweep (DESIGN SPIKE)

> Дизайн-спайк: только документ, прод-кода нет. Правило: `crates/*/src` не трогать.

## Вопрос

Given закрытые в M2 GPU-предусловия AVBD, when оформляем один primal
sweep одного тела как compute kernel (вход — собранные 6×6 блоки / J / F,
выход — dq; dual отдельно), then фиксируем минимальную единицу переноса,
список препятствий и гейт спайка — да или нет, проверяемо?

## Контекст: что уже закрыто, что нет

- M2 закрыл только предусловия DSL (PLAN.md:648-662): `Mat3` в
  `ShaderType` (`mat3x3<f32>`, `Mat3::from_cols`, `IDENTITY`/`ZERO`),
  локальные фикс-массивы под гессиан (включая вложенные; эффектные
  repeat громко отклоняются), `helpers(...)` в `#[gpu_pipeline]`
  (стыковка `wgsl_source()` впереди entry). Пины:
  `macros/tests/compute_dsl.rs` (включая naga-валидацию LDL 3×3) и
  physics `helpers_stitch_ahead_of_main_and_validate`.
- Реального GPU AVBD-солвера нет. Референс CPU-пути —
  `AvbdEngine::solve_body` (`crates/physics/src/avbd.rs`): сборка 6×6
  (`m_dt2` + world-инерция + contact/joint rows), плотный LDL 6×6 без
  пивотинга (`solve_6x6`, `None` при breakdown), primal/dual-цикл,
  последовательный sweep по телам — как авторский `solver.cpp`
  (`savant117/avbd-demo3d`, Giles SIGGRAPH'25), портированный в M0
  (`spikes/001-avbd-stack`).
- В проекте уже есть GPU bulk v2 для builtin (`crates/physics/src/gpu.rs`,
  шапка + `WgpuContactSolver::solve`): все итерации — back-to-back
  compute-проходы в ОДНОМ энкодере (per-pass params через dynamic uniform
  offsets, один upload / один submit / один blocking wait), модель —
  Jacobi/GS hybrid (single-point → GPU, multi-point block-LCP → CPU),
  осознанно не bit-identical CPU-пути. Плюс `AutoLane`/`GpuLanes`
  (`crates/wgpu_backend`) как готовая lane-инфраструктура.

## Минимальная единица переноса

Один primal sweep одного тела как один compute kernel, без сцены:

- Вход (CPU-сборка, как сейчас в `solve_body`): 6×6 lhs-блок тела,
  6-вектор rhs (инерциальный член + contact/joint rows уже свёрнуты),
  penalty/lambda состояние пар — read-only на время sweep.
- Kernel: плотный LDL 6×6 без пивотинга + прямые/обратные подстановки,
  выход — dq (6: dx + dtheta). Один workgroup (или одна invocation)
  на тело; межтельные зависимости — только через sequential sweep
  на хосте, как в `solver.cpp`.
- Dual (`updateDual`: penalty-ramp, lambda-память, separated-damper /
  imminence-gate правила из `solve_body`) — ОТДЕЛЬНО: либо второй
  kernel, либо остаётся на CPU в первой итерации спайка. Смешивать
  primal и dual в одном kernel не предлагать — dual-правила портируются
  только дословно (урок M0: угаданные dual-правила дают
  coupling-неустойчивость).

Что единица НЕ включает: island routing, broadphase, warm-start
миграцию, multi-point rolling — только `lhs/rhs → dq` на одном теле.

## Что мешает (фиксируем как риски, не решаем здесь)

1. **LDL 6×6 в WGSL.** Плотный LDL без пивотинга — вложенные циклы с
   ранним выходом при breakdown (`s <= 1e-12 → None`). В шейдере нужен
   тот же контракт отказа (тело скипает обновление, а не пишет NaN),
   иначе один вырожденный блок травит весь sweep. CPU-фолбэк при
   breakdown обязателен по построению.
2. **Гессиан-вложенность.** Primal-сборка держит вложенные фикс-массивы
   (3×3 блоки внутри 6×6). DSL-предусловие закрыто, но первая же
   эффектно-мутабельная итерация по ним будет громко отклонена —
   формулировать сборку сразу в принимаемом DSL-подмножестве
   (по образцу пина LDL 3×3 в `compute_dsl.rs`).
3. **Детерминизм осознанно не bit-identical.** GPU float (fma/rounding,
   порядок sweep при параллелизации тел) не даст побитового паритета с
   CPU Strong-Confluence путём — та же оговорка, что у bulk v2 в шапке
   `gpu.rs`. Авторитетный путь — CPU; GPU AVBD — опт-ин для
   визуально-массовых тел, как задокументировано для G7. Поведенческий
   паритет (позы покоя), никогда побитовый.
4. **Sweep vs bulk.** Bulk v2 выигрывает на независимых батчах
   (дизъюнктные тела, Jacobi). AVBD sweep по построению
   Gauss-Seidel-последователен (тело видит свежие соседи). Параллелить
   тела внутри sweep — менять математику сходимости; спайк держит sweep
   последовательным, параллелизм только внутри тела (6×6), иначе это
   уже другой солвер.

## Гейт спайка (проверяемо, без сцены)

1. Сгенерированный WGSL primal-kernel валидируется naga без устройства
   (тот же приём, что bulk v2).
2. Паритет на ОДНОМ теле под lavapipe: CPU `solve_6x6` vs GPU kernel на
   поведенческом допуске (не побитово) — вход один и тот же `lhs/rhs`,
   сравнивается `dq`. Сцены, стеки, итерации до покоя — вне гейта.
3. Breakdown-контракт: вырожденный `lhs` даёт пропуск обновления, а не
   NaN/взрыв — явный кейс в спайке.

## Рекомендация

Держать спайк на одном теле (kernel + naga + lavapipe-паритет), не
расползаться на сцену/острова/coupling. При зелёном гейте — следующий
шаг: dual как второй kernel с дословным портом damper/gate-правил,
и только потом разговор о sweep-параллелизме.
