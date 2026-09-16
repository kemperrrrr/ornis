# 001: cross-solver contact (THROWAWAY SPIKE)

## Question
Given two dynamic boxes owned by DIFFERENT solvers (AVBD + Builtin),
when they stack via a staggered Legacy-style boundary exchange
(foreign body mirrored as kinematic, one step lag),
then the stack rests at the same heights as single-solver within
behavioral tolerance, with no energy growth — yes or no, measured?

## Why it matters
M3 killer risk: PLAN.md lists cross-engine collision detection as a hard
part. Genesis (see `/tmp/genesis-ref`, `simulator.py` + `couplers/`)
answers it WITHOUT migrating bodies: static ownership by material at add
time, per-substep boundary exchange (Legacy kernels: foreign side as a
kinematic boundary + reaction back; SAP: joint solve). Our case is two
rigid solvers over the same body type, so the spike mirrors Genesis
Legacy-style: each solver owns its bodies, cross pairs couple through
kinematic mirrors.

## Approach
- Shared static floor created once in BOTH engines (statics never sync).
- AVBD owns top box (dynamic) + kinematic mirror of bottom.
- Builtin owns bottom box (dynamic) + kinematic mirror of top.
- Per step: copy owner pose+velocity into mirrors, step AVBD, step
  Builtin (staggered: each solver sees the foreign body one step stale).
- No joints across solvers in this spike (hard part #2, spike 002).

## Run
`cargo test -p ornis-physics --test m3_spike_001`

## Verdict: PARTIAL

### What worked
- One-directional coupling rests: AVBD top on kinematic mirror of the
  Builtin bottom settles at y≈1.5, bottom at y≈0.5, velocities ~0.
- Determinism: two identical coupled runs are bit-identical.
- Overhead (debug build, 60 steps): coupled 784ms vs single-solver
  160ms ≈ 5x — dominated by stepping two engines, not mirror sync.

### What didn't
- Symmetric staggered mirrors BULLDOZE: both boxes ejected sideways
  (z≈3-7) and rest on the floor, not stacked. The mirror tracks its
  owner with one step lag; while bodies approach, the teleported
  kinematic mirror overlaps the other body and plows through it, and
  SAT resolves the deep overlap along ±z. Velocities end ~0 — a quiet
  wrong answer, the worst kind.

### Surprises
- The failure is silent (no explosion, no tunneling) — ejection +
  rest looks plausible per-step. A cross-solver stack test must assert
  rest POSES, never velocity alone (same lesson as the retouch pins).
- Weight transfer is missing one-directional: the Builtin bottom never
  feels the AVBD top. Fine for rest heights, wrong for heavy-top
  stacks and friction chains.

### Recommendation for the real build
- Do NOT ship symmetric teleported mirrors. Two honest options:
  (a) force-level reaction exchange Genesis-Legacy-style (cross impulse
  applied to BOTH sides, not just the owner's solver);
  (b) route by ISLANDS — a contact-connected component always solves
  in one solver, cross pairs exist only at island boundaries during
  transitions. (b) sidesteps the plow almost entirely and matches the
  existing island infrastructure (`engine/islands.rs`).
- Next: spike 002 (per-step ownership transfer) should prototype (b).
