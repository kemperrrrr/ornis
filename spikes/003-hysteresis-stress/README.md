# 003: hysteresis under periodic driving (THROWAWAY SPIKE)

## Question
Given a scripted periodic kick (every 90 steps) that re-wakes a settled
ball, when islands route with hysteresis (unanimous calm → Builtin,
any fast body → Avbd immediately), then migrations stay at ~2 per
drive cycle (wake + sleep) with no within-cycle thrash, rest states
stay correct, and reruns are identical — yes or no, measured?

## Why it matters
002 proved one migrate–settle–migrate cycle. The hysteresis design
must also survive sustained periodic driving without thrashing
(rebuild storms) or state corruption across ~10 cycles.

## Approach
- Same M3 harness as 002 (fixed-order registry, rebuild migration,
  zombie-guard mass restore, island union-find on proximity).
- Migrant ball kicked to v=(0,4,0) every 90 steps when calm (900 steps,
  10 cycles); control ball settles once and never moves.
- The two balls are 5 m apart: independent islands, independent routing.

## Run
`cargo test -p ornis-physics --test m3_spike_003`

## Verdict: VALIDATED

Measured: 19 rebuilds over 900 steps / 10 drive cycles — ~2 per cycle
(wake + sleep) plus the initial settle, zero within-cycle thrash. Both
balls end calm in Builtin; rerun bit-identical.

### What worked
- Hysteresis shape holds under sustained periodic driving: unanimous
  calm migrates down, any fast body migrates up immediately, and the
  30-step calm requirement plus the 0.2/0.5 band absorb boundary jitter.
- No state corruption across ~10 rebuild cycles (poses/velocities sane,
  control rests untouched throughout).

### What didn't
- Nothing failed. Residual risk (not covered): a body hovering exactly
  inside the (0.2, 0.5) deadband resets its calm counter forever and
  never migrates down — by design (deadband = don't-care), but a scene
  engineered at v≈0.3 would pin to AVBD. Acceptable: AVBD is the
  always-correct fallback, Builtin is the optimization.

### Surprises
- None — the predicted 2/cycle matched exactly (19 measured).

### Recommendation for the real build
- Keep the hysteresis shape as-is (per-body counters, unanimous-island
  calm, immediate wake). Add a migration counter metric (coupling tax
  goes in the books per PLAN M3).
- Next: 004 (assignment-invariance of rest state).
