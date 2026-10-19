//! 2D plane grid as pure math (PLAN §k, slice K2).
//!
//! [`plane_grid`] turns a [`PlaneCamera`] into the numbers a web overlay
//! needs to draw the sheet grid: a minor step from the `1/2/5·10^k` ladder
//! (the smallest whose on-screen size clears `min_step_px`), the major-line
//! cadence, the step in pixels and the screen position of the world origin.
//! Drawing stays web-side (there is no line pass in the renderer, and
//! grid-as-entities is E4) — this module never touches DOM or GPU state.

use ornis_core::PositiveF32;

use crate::plane::PlaneCamera;

/// Grid numbers for one frame: step, cadence and origin for the overlay.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct PlaneGrid {
    /// Minor step in world units, from the `1/2/5·10^k` ladder.
    pub minor_step: f32,
    /// Minor steps per major line: `10` for mantissas `1`/`5`, `5` for `2`.
    pub major_every: u32,
    /// Minor step on screen in pixels (`>= min_step_px`, `< 2.5·min_step_px`).
    pub minor_step_px: f32,
    /// Screen position of the world origin (may sit outside the viewport).
    pub origin_px: [f32; 2],
}

/// Mantissas of the decade ladder (named consts: bare literals trip the
/// complexity gate's magic-number rule).
const LADDER_MANTISSAS: [f32; 3] = [1.0, 2.0, 5.0];
/// Decade base of the ladder.
const LADDER_DECADE: f32 = 10.0;
/// Minor steps per major line for the middle mantissa.
const MAJOR_EVERY_MID: u32 = 5;
/// Minor steps per major line for the outer mantissas.
const MAJOR_EVERY_DECADE: u32 = 10;

/// Picks the grid step for `camera`: the smallest `1/2/5·10^k` step whose
/// on-screen size clears `min_step_px`, plus the major cadence and the
/// screen position of the world origin (`plane_to_screen(0, 0)`).
#[must_use]
pub fn plane_grid(camera: &PlaneCamera, min_step_px: PositiveF32) -> PlaneGrid {
    let ppu = camera.pixels_per_unit();
    let floor = min_step_px.get();
    // Ladder walk: 1/2/5 mantissas across decades until the on-screen step
    // clears the floor. `ppu > 0` by construction (`viewport_h > 0`,
    // `hh > 0`), so the loop always terminates.
    let mut magnitude = (floor / ppu).log10().floor() as i32 - 1;
    loop {
        for mantissa in LADDER_MANTISSAS {
            let step = mantissa * LADDER_DECADE.powi(magnitude);
            if step * ppu >= floor {
                let major_every = if mantissa == LADDER_MANTISSAS[1] {
                    MAJOR_EVERY_MID
                } else {
                    MAJOR_EVERY_DECADE
                };
                return PlaneGrid {
                    minor_step: step,
                    major_every,
                    minor_step_px: step * ppu,
                    origin_px: camera.plane_to_screen(glam::Vec2::ZERO),
                };
            }
        }
        magnitude += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;
    use ornis_core::Meters;

    use crate::plane::PlaneCamera;

    /// Rig with `60 px` per unit (`hh = 5` on a `600 px` tall viewport).
    fn camera() -> PlaneCamera {
        PlaneCamera::new(
            Vec2::ZERO,
            Meters::new(5.0),
            Meters::new(10.0),
            Meters::new(0.1),
            Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("valid test camera")
    }

    /// The ladder picks the smallest step clearing the floor, with the
    /// documented major cadence per mantissa.
    #[test]
    fn ladder_picks_smallest_clearing_step() {
        // `ppu = 60`: `0.5` (30 px) misses, `1.0` (60 px) clears.
        let grid = plane_grid(&camera(), PositiveF32::expect_valid(48.0));
        assert_eq!(grid.minor_step, 1.0);
        assert_eq!(grid.major_every, 10);
        assert_eq!(grid.minor_step_px, 60.0);
        // A higher floor climbs the ladder: `1.0` (60 px) misses `100 px`,
        // `2.0` (120 px) clears with cadence 5.
        let grid = plane_grid(&camera(), PositiveF32::expect_valid(100.0));
        assert_eq!(grid.minor_step, 2.0);
        assert_eq!(grid.major_every, 5);
        assert_eq!(grid.minor_step_px, 120.0);
        // Exact-boundary floors hold the step: `5.0` at 300 px clears 300.
        let grid = plane_grid(&camera(), PositiveF32::expect_valid(300.0));
        assert_eq!(grid.minor_step, 5.0);
        assert_eq!(grid.major_every, 10);
    }

    /// Gate: `min_step_px ≤ minor_step_px < 2.5·min_step_px`, and the origin
    /// matches `plane_to_screen(0, 0)` — across zoom levels.
    #[test]
    fn step_bounds_and_origin_hold_across_zoom() {
        let mut camera = camera();
        for floor in [8.0, 24.0, 48.0, 100.0, 200.0] {
            for _ in 0..3 {
                let grid = plane_grid(&camera, PositiveF32::expect_valid(floor));
                assert!(
                    grid.minor_step_px >= floor && grid.minor_step_px < 2.5 * floor,
                    "floor {floor}: got {}",
                    grid.minor_step_px
                );
                assert_eq!(grid.origin_px, camera.plane_to_screen(Vec2::ZERO));
                camera.zoom_at(250.0, [400.0, 300.0]);
            }
        }
    }

    /// Panning moves the origin with the world: a 60 px drag at 60 ppu
    /// shifts the origin by exactly 60 px.
    #[test]
    fn origin_tracks_pan() {
        let mut camera = camera();
        let before = plane_grid(&camera, PositiveF32::expect_valid(48.0)).origin_px;
        camera.pan_pixels([60.0, 0.0]);
        let after = plane_grid(&camera, PositiveF32::expect_valid(48.0)).origin_px;
        assert!((after[0] - before[0] - 60.0).abs() < 1e-4);
        assert!((after[1] - before[1]).abs() < 1e-4);
    }
}
