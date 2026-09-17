# 004: routing-invariant rest state (THROWAWAY SPIKE)

## Question
Given the same scene started under different initial assignments
(all-AVBD vs all-Builtin), when both run 600 steps with island routing
active, then both settle to the same rest poses within behavioral
tolerance — yes or no, measured?

## Why it matters
The M3 correctness contract: the routing choice must not change physics
outcomes. Trajectories WILL differ (different solvers integrate
differently), but rest states must agree. If they don't, routing is not
an optimization — it's a different simulation.

## Approach
- Same M3 harness (fixed registry, rebuild migration, zombie-guard,
  island union-find, hysteresis).
- Run A: everything starts AVBD. Run B: everything starts Builtin.
- 2-stack + migrant, 600 steps, no kicks (routing + settling only).
- Assert rest heights within tolerance (behavioral, never bitwise).

## Run
`cargo test -p ornis-physics --test m3_spike_004`

## Verdict: VALIDATED

Same rest poses within 0.1 m from all-AVBD vs all-Builtin starts
(2-stack + migrant, 600 steps, routing active throughout).

### What worked
- Routing is an optimization, not a different simulation: rest states
  agree behaviorally despite different integration paths and
  migration histories.

### What didn't
- Nothing failed. Note the tolerance is behavioral (0.1 m), never
  bitwise — trajectories legitimately differ by solver.

### Surprises
- None — first-run green.

### Recommendation for the real build
- Promote this shape to a suite gate in the M3 implementation
  (drop-settle scene × initial assignments).
- Spikes complete: 001 PARTIAL (plow documented), 002/003/004
  VALIDATED. Ready for M3 design on `Engine`.
