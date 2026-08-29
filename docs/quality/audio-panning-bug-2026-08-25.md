# Audio spatial panning: azimuth contract mismatch (fixed)

> **Статус:** исправлено в коммитах `925d2c5` и `06fd801`. Таблица и
> первоначальный вариант решения ниже сохранены как история диагностики;
> финальный контракт и реализация указаны в отдельном разделе.

**Date:** 2026-08-25
**Final status:** fixed; the pre-fix diagnosis and final contract are
recorded below.

## Summary of the pre-fix defect

`SpatialParams.azimuth` had **no documented contract**, and the three
places that produced / consumed it disagreed on its meaning. This caused a
real, audible defect (channel inversion at wide angles) plus a silent
geometry loss.

## What each site did before the fix

| Site | Treatment of `azimuth` | Issue |
|------|------------------------|-------|
| `engine.rs:72-76` | `azimuth = (diff.x / distance).asin().clamp(-1.0, 1.0)` | `asin` returns **radians** (∈ [−π/2, π/2] ≈ [−1.57, 1.57]); the `.clamp(-1.0, 1.0)` then **truncates** angles 57°–90°, discarding real geometry. |
| `native.rs:159` `pan_sample` | `angle = (azimuth + 1.0) * FRAC_PI_4` | **expects `azimuth ∈ [-1, 1]`** (normalized pan). Feeding radians (up to 1.57) gives, at `azimuth = 1.57`, `angle ≈ 2.0 rad` → `cos(2.0) ≈ −0.42` → **left channel inverted / anti-phase** for a source on the right. |
| `wasm.rs:86` | `panner.position_x().set_value(sp.azimuth)` | Web Audio `PannerNode.position_x` is a **position in meters**, not an angle. Source "1 radian to the right" is placed "1 meter to the right" — geometrically wrong. |

## Why this is a bug, not a feature

- The **center-is-−3 dB** behaviour of `pan_sample` is *correct* equal-power
  panning and is **not** the bug. The bug is the **contract mismatch**.
- Native inverts the left channel for wide-right sources (audible artifact).
- `engine.rs` discards angles beyond 57° via the bogus `clamp`.
- The two backends interpret the same field differently (normalized pan
  vs. world-space X coordinate), so the browser and native builds pan
  **differently** for the same scene — a determinism / correctness break.

## Final contract and implementation

Rationale: a 3D engine's spatial audio should reflect the real direction
of the source relative to the listener. `engine.rs` computes the true
azimuth (`asin(diff.x/dist)`) in radians; both backends now consume that
same geometric contract.

1. **`engine.rs`** — the bogus clamp was removed; `asin` is already bounded:
   ```rust
   let azimuth = if distance > 0.001 {
       (diff.x / distance).asin()
   } else {
       0.0
   };
   ```
2. **`native.rs::pan_sample`** — the final product choice is equal-gain
   linear panning (not the equal-power formula from the initial fix):
   ```rust
   let t = (azimuth / std::f32::consts::FRAC_PI_2).clamp(-1.0, 1.0);
   let left = ((1.0 - t).clamp(0.0, 1.0) * sample).clamp(-1.0, 1.0);
   let right = ((1.0 + t).clamp(0.0, 1.0) * sample).clamp(-1.0, 1.0);
   ```
   Checks: `θ=−π/2` → `(1, 0)`; `θ=0` → `(1, 1)`; `θ=+π/2` →
   `(0, 1)` for a normalized positive sample, with no channel inversion.
3. **`wasm.rs`** — `PannerNode` receives the same angle-derived position,
   including elevation, and is configured with the linear distance model,
   reference distance and rolloff factor:
   ```rust
   let x = sp.distance * sp.azimuth.sin() * sp.elevation.cos();
   let y = sp.distance * sp.elevation.sin();
   let z = -sp.distance * sp.azimuth.cos() * sp.elevation.cos();
   ```

## Verification

Added unit tests in `crates/audio/src/backend/native.rs` (the `pan_sample_`
suite) asserting the three cardinal directions, monotonic linear gains and
that **no channel inverts** for any `azimuth ∈ [-π/2, π/2]`. Regression:
the old formula failed `azimuth = π/2` (expected pure right, got
`L≈−0.42·s`). The final center behavior is intentionally equal-gain, so it
is not an equal-power energy invariant.

`wasm.rs` now consumes elevation and configures the browser panner's linear
distance model with `rolloff_factor` and `reference_distance`. The current
`AudioEngine` derives `elevation = 0` and fixed rolloff/reference values from
its 3D position input; richer listener/source orientation is future work.
