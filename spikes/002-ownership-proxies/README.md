# 002: island ownership + relaxed proxies (THROWAWAY SPIKE)

## Question
Given per-step solver assignment with hysteresis, when bodies migrate
between AVBD and Builtin via rebuild (M2 machinery) and cross pairs
couple through DYNAMIC proxies with relaxed pose/velocity tracking
(no pose teleport), then a mixed stack + migrant ball settles, stays
settled across migrations, handles never break, and migration count
stays bounded — yes or no, measured?

## Why it matters
001 killed pose-teleport mirrors (kinematic plow → silent side-eject).
Newton's proxy answer: finite virtual inertia + lagged/staggered
transfer + relaxed feedback. This spike ports that: proxies are full
dynamics with real mass; per step `v += RV*(v_owner - v)` and
`p += RP*(p_owner - p)` with small RP (mm-scale tracking absorbed by
contact margins, no plow-through). Ownership (Newton: explicit subsets
+ global↔local maps) is a fixed-order global registry here; migration
is a deterministic rebuild, hysteresis bounds its frequency.

## Approach
- Globals: floor (static in both, never migrates) + bottom (pinned
  Builtin) + top (pinned AVBD) + migrant (routed).
- Routing: awake/fast → AVBD immediately; sleepy (v<0.15 for 30 steps)
  → Builtin. Migrant kicked at step 300 to force wake-migrate-settle.
- Cross proxies: every foreign-owned dynamic gets a dynamic proxy in
  the other engine (RV=0.5, RP=0.1, hardcoded tuning surface).
- No cross-solver joints (documented M3 hard part, out of scope).

## Run
`cargo test -p ornis-physics --test m3_spike_002`

## Verdict: VALIDATED (with two mandatory reconcile rules)

### What worked
- Island routing end-to-end: 2-stack settles (0.50/1.49), migrant
  settles, kick at step 300 re-flies and re-settles, migrations bounded
  (4 rebuilds over 600 steps incl. kick cycle), rerun bit-identical.
- Cross pairs vanish as a class: an island always steps in one solver.
  No proxies, no plow, no weight-transfer gap.
- Hysteresis works: unanimous-island sleep migrates down, any fast
  body wakes the island up immediately.

### What didn't (and what it taught)
- v1 (staggered dynamic proxies) PUMPED energy through the floor
  (bottom tunneled to y=-360): two lagged pairs chasing each other is
  unstable for resting contact. Killed, documented here for the record.
- Zombie bodies: a snapshot taken from a sleeping engine carries
  inv_mass=0 AND zero inertia; rebuilt awake it is unsolvable (frozen
  forever) or tumbles corner-first through the floor. reconcile MUST
  restore the mass model like wake_body. M2 already does this inside
  `bodies_snapshot` — the spike re-derived the rule at the harness
  level; any M3 registry must apply it on every migration.
- Host edits into sleeping engines freeze (Builtin G7 skips the world;
  AVBD needs its velocity-field wake check to see the edit). Migration
  via rebuild sidesteps it (fresh engines awake), but direct host
  writes need an explicit wake path in the real build.
- AVBD rest velocity floor is ~0.16 (g*dt), so a 0.15 sleep threshold
  never fires; routing thresholds must sit above it (used 0.2/0.5) or
  read engine sleep flags authoritatively.

### Surprises
- The migrant-frozen-with-v=6 symptom looked like a wake bug but was
  the zombie (inv_mass=0 + awake). Two stacked causes, one symptom.

### Recommendation for the real build
- M3 = fixed-order global registry (stable handles) + island routing
  with hysteresis + rebuild migration with mass-model restore +
  static floor natively in both engines. No cross-solver joints yet
  (still the documented hard part).
- Next: 003 (hysteresis stress: thrash scene at the sleep boundary),
  004 (determinism gate — already green here, promote to suite).
