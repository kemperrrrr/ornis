# Audio spatial panning: azimuth contract mismatch (bug)

**Date:** 2026-08-25
**Status:** found during coverage work on `crates/audio`; fix applied in
this session (see commit referenced in changelog). Documented for review.

## Summary

`SpatialParams.azimuth` has **no documented contract**, and the three
places that produce / consume it disagree on its meaning. This causes a
real, audible defect (channel inversion at wide angles) plus a silent
geometry loss.

## What each site does

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

## Chosen fix (variant Y: true geometric angle in radians)

Rationale: a 3D engine's spatial audio should reflect the real direction
of the source relative to the listener. `engine.rs` already computes the
true azimuth (`asin(diff.x/dist)`); we keep that and make both backends
consume radians consistently.

1. **`engine.rs`** — drop the bogus clamp; `asin` is already bounded:
   ```rust
   let azimuth = if distance > 0.001 {
       (diff.x / distance).asin()
   } else {
       0.0
   };
   ```
2. **`native.rs::pan_sample`** — map radians to equal-power:
   ```rust
   let phi = azimuth / 2.0 + std::f32::consts::FRAC_PI_4;
   let left = (phi.cos() * sample).clamp(-1.0, 1.0);
   let right = (phi.sin() * sample).clamp(-1.0, 1.0);
   ```
   Checks: `θ=−π/2` → `(1, 0)`; `θ=0` → `(1/√2, 1/√2)` (equal-power center);
   `θ=+π/2` → `(0, 1)`; no inversion at any angle.
3. **`wasm.rs`** — derive a world position from angle + distance:
   ```rust
   p.position_x().set_value(sp.distance * sp.azimuth.sin());
   p.position_z().set_value(-sp.distance * sp.azimuth.cos());
   ```

## Verification

Added unit tests in `crates/audio/src/backend/native.rs` (the `pan_sample_*`
suite) asserting the three cardinal directions and that **no channel
inverts** for any `azimuth ∈ [-π/2, π/2]`. Regression: the old formula
failed `azimuth = π/2` (expected pure right, got `L≈−0.42·s`).

## Out of scope

`wasm.rs` panner also ignores `elevation`, `rolloff_factor`,
`reference_distance` (only `azimuth`/`distance` wired). Distance attenuation
(`distance_attenuation` in native) is separate and correct; wasm delegates
attenuation to the browser `PannerNode` (acceptable). Documented here so the
gap is visible, not silently dropped.
