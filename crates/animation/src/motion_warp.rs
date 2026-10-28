//! Root-trajectory warping (§8.5): the root's per-frame delta bends the
//! authored trajectory onto a moving goal over a [`WarpWindow`] — «шаг на
//! платформу», приземление. Pure functions on data, no allocations.

use glam::Vec3;

use crate::motion_match::WarpWindow;

/// Progress of the root-trajectory warp in `0.0..=1.0` at clip `time`:
/// linear from `0` at `window.start` to `1` at `window.contact`, held at
/// `1` afterwards. Unlike the limb pull of [`crate::warp_weight`] — which
/// relaxes back down around the contact — the trajectory redirect is
/// complete at the contact and persists: the root lands on the goal and
/// keeps walking from there (§8.5).
///
/// `0` outside `start..=contact` and on a non-finite or unordered window
/// (`start <= contact <= end` is required, and `end` must be strictly
/// after `start`).
#[must_use]
fn trajectory_progress(time: f32, window: WarpWindow) -> f32 {
    let WarpWindow {
        start,
        contact,
        end,
    } = window;
    let ordered = start.is_finite()
        && contact.is_finite()
        && end.is_finite()
        && time.is_finite()
        && start <= contact
        && contact <= end
        && end > start
        && time > start;
    if !ordered {
        return 0.0;
    }
    if time >= contact {
        return 1.0;
    }
    // `start < time < contact` holds here, so the span is positive.
    ((time - start) / (contact - start)).clamp(0.0, 1.0)
}

/// Root-trajectory warp of one frame's root delta (§8.5): `delta` picks
/// up the frame's share of `goal_gap` — the vector from where the root
/// would land at `window.contact` without the warp to where the host
/// wants it («шаг на платформу», приземление). The shares sum to
/// `goal_gap` over the window, so the redirected root arrives at the
/// goal exactly at the contact and keeps the offset afterwards. The host
/// passes its mixed [`crate::RootMotionDelta`] through this before
/// applying it (§8.4).
///
/// `from` and `to` are clip times of the same clock the window was
/// authored on — the previous frame's time and the current one. The
/// share is the difference of the two ramp samples, so any span and
/// order are consistent (a scrub backwards unwinds the warp). Outside
/// the window the delta passes through untouched, and so does it on a
/// non-finite `goal_gap`.
#[must_use]
pub fn warp_root_delta(
    delta: Vec3,
    from: f32,
    to: f32,
    window: WarpWindow,
    goal_gap: Vec3,
) -> Vec3 {
    if !goal_gap.is_finite() {
        return delta;
    }
    let share = trajectory_progress(to, window) - trajectory_progress(from, window);
    delta + goal_gap * share
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window() -> WarpWindow {
        WarpWindow {
            start: 0.2,
            contact: 0.5,
            end: 0.8,
        }
    }

    /// The per-frame shares of [`warp_root_delta`] sum to `goal_gap`
    /// across the window: the redirected root arrives at the goal
    /// exactly at the contact.
    #[test]
    fn trajectory_warp_sums_the_gap_by_the_contact() {
        let gap = Vec3::new(3.0, 0.0, 4.0);
        let mut landed = Vec3::ZERO;
        for frame in 0..5_u16 {
            let from = 0.2 + f32::from(frame) * 0.06;
            landed += warp_root_delta(Vec3::ZERO, from, from + 0.06, window(), gap);
        }
        assert!((landed - gap).length() < 1e-5, "{landed} vs {gap}");
    }

    /// Outside the window the delta passes through untouched, before
    /// the start and after the contact alike.
    #[test]
    fn trajectory_warp_is_flat_outside_the_window() {
        let gap = Vec3::new(1.0, 2.0, 3.0);
        let walk = Vec3::new(0.5, 0.0, 0.0);
        assert_eq!(warp_root_delta(walk, 0.0, 0.1, window(), gap), walk);
        let late = warp_root_delta(walk, 0.85, 0.95, window(), gap);
        assert_eq!(late, walk, "after the contact");
    }

    /// An unordered window or a non-finite gap leave the delta alone.
    #[test]
    fn trajectory_warp_rejects_bad_input() {
        let broken = WarpWindow {
            start: 0.5,
            contact: 0.2,
            end: 0.8,
        };
        let walk = Vec3::new(1.0, 0.0, 0.0);
        let gap = Vec3::new(3.0, 0.0, 4.0);
        assert_eq!(warp_root_delta(walk, 0.2, 0.4, broken, gap), walk);
        let sick = Vec3::new(f32::NAN, 0.0, 0.0);
        assert_eq!(warp_root_delta(walk, 0.2, 0.4, window(), sick), walk);
    }

    /// Scrubbing backwards unwinds the warp: the share is the signed
    /// difference of the ramp samples, so `from > to` subtracts.
    #[test]
    fn trajectory_warp_unwinds_backwards() {
        let gap = Vec3::new(2.0, 0.0, 0.0);
        let forward = warp_root_delta(Vec3::ZERO, 0.2, 0.5, window(), gap);
        let back = warp_root_delta(forward, 0.5, 0.2, window(), gap);
        assert!(back.length() < 1e-6, "{back}");
    }
}
