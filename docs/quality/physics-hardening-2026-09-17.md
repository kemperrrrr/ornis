# Physics hardening and M3 verification — 2026-09-17

Work is on PR #12 (`arena/01a0a609-ornis`); no merge is authorized.

## Changes under verification

- AVBD: local angular errors / world gradients, including Fixed and SixDof;
  nonzero reactions survive zero-error and small-inertia states.
- Joint lifecycle: preserve physical assembly references and continuous gear
  phase across solver/routing changes; remove dependent gears before dense remaps.
- Native solver: retain joints on a swapped-in tail body; quaternion-sign-invariant
  angular correction; continuous gear coordinates.
- M3: shared fixed clock, swept pre-step ownership, angular-aware hysteresis,
  global queries before deferred rebuilds, event/driver baselines, remapped
  fracture events, per-solver and routing/rebuild timing metrics.
- Geometry: actual full-side cuboid inertia (old value was 4x too small), signed
  sphere/box and box/box containment, face-interior capsule-spine intersections.
- Fracture: velocities sampled from the parent's rigid velocity field, preserving
  linear/angular momentum and kinetic energy rather than cloning COM velocity.

## CI evidence so far

- `35187504511`: root compile blocked by two new `MeshDesc::Custom` match omissions;
  fixed without inventing a sphere collider for unimplemented Custom transport.
- `35190724680`: the angular-regression tests passed; an existing step-budget test
  compared *end-of-step* backend counters, not the input candidate set. The test
  now compares complete sorted candidate lists (stronger than count equality).
- `35197777773`: duplicate `joint_count` API, corrected.
- `35199760179`: 238 library tests passed; fracture conservation exposed the
  cuboid inertia bug, and the old gear test read wrapped rather than continuous
  rotation. Both causes were corrected, not hidden by looser thresholds.
- `35207392615`: **239 physics library tests passed**, one expected old canonical
  snapshot mismatch, one ignored regeneration test. The new snapshot is the
  exact x86-64 CI output after the intentional inertia/geometry changes.

The full gate is NOT yet green. Integration suites and final CPU/GPU validation
must complete before marking M3 closed. ARM replay of the new canonical is not
claimed by this x86-64 run.

## Baseline calibration, not suppression

The checked-in rustqual baseline (score 0.1786838, 128 violations, 1807 findings)
was not the state of current master. An isolated archive of `c97089a` was measured
by the same rustqual 1.8.2 in the same CI job: score **0**, 275 violations, 2946
findings. The reviewed implementation measured score **0**, 277 violations, 2958
findings, with improved IOSP and one fewer nesting/long-function warning.

The refresh records this deliberate physics/diagnostics scope (+2 violations,
+12 findings against actual master), not a claim that master was at 17.8% when
this task began. No rustqual thresholds, weights, exclusions or quality failure
conditions were weakened. The exact comparison is retained in
`physics-baseline-calibration-2026-09-17.json`; `baseline.json` is the complete
measured CI export. Further changes remain ratchet-checked.

## CI transport and resource limits

Several runners terminated with SIGTERM/exit 143 before failure-comment steps;
these were not successful test runs. Workflow edits were refused by the GitHub
App's `workflows` permission and were not pushed. Full diagnostics are available
through bounded check annotations; stage output is now streamed as well.

Compiler parallelism is bounded to two jobs, GPU test processes run serially,
and debug line tables/assertions remain enabled. Smoke separates build from its
unchanged 90-second runtime readiness deadline and drains output without an
unread stderr pipe. No test or mandatory stage is disabled.

## 2026-09-18 merged (appendix; history above is unchanged)

- PR #12 (`fix(physics): AVBD correctness and M3 multisolver lifecycle`) is
  MERGED as `d6a59b0` (mergedAt 2026-09-18T05:52:57Z). PR #13
  (`fix(physics+quality): finish AVBD/M3 work of PR #12 with green quality
  gate`) is MERGED as `0a26089` (mergedAt 2026-09-18T05:52:55Z). Note:
  `0a26089` is the PR #13 merge, not the PR #12 merge.
- Green evidence for the merges: Quality run `35312558704` (push of
  `0a26089`) is `completed/success`; Quality run `35257932070` (PR #13
  pull_request) is `completed/success`. Re-check with `gh run list` and the
  `cargo xtask quality` gate.
- HEAD caveat (2026-09-18T07:00Z): post-merge `28e7301` run `35316617236`
  is `completed/failure` and `d227e11` run `35317406118` was `in_progress`;
  the gate is therefore NOT claimed green at HEAD.
- Baseline comparison: `docs/quality/physics-baseline-calibration-2026-09-17.json`.
- The 2026-09-17 statements above ("Work is on PR #12; no merge is
  authorized", "The full gate is NOT yet green") are retained as history and
  are SUPERSEDED by this section for the merge status.
