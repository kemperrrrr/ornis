//! Platform-independent viewport logic: what a bounds post means, which
//! content height wins, how a scene body is parsed, how a frame step is
//! clamped, and what a new placement requires.
//!
//! None of this needs AppKit, a `wgpu::Device` or a window, so it is **not**
//! `cfg`-gated and every function here is exercised by CI on Linux. The macOS
//! modules (`frame`, `platform`, `host`) are then thin: they take the decision
//! this module produced and apply it to an `NSView` and a swap chain.
//!
//! Keeping the split here — rather than inside the `cfg`-gated modules — is
//! deliberate: ~1250 lines of macOS code never see a Linux compiler, so
//! anything that *can* be checked everywhere must live where it is checked.

use crate::PhysicalRect;
use crate::native_viewport::layout::ViewportPlacement;

/// Longest frame step fed to the viewport simulation, in seconds.
///
/// Same idea as the engine's own `clamp_frame_dt`: a hitch (window drag, tab
/// switch back) must not fast-forward the scene, and a long stall must not make
/// the orbit camera jump. One frame at 15 fps is the ceiling.
pub const MAX_FRAME_STEP: f32 = 1.0 / 15.0;

/// Clamps a measured frame interval into the range the simulation accepts.
///
/// Non-finite input (a clock that went backwards, a NaN from a broken timer)
/// becomes `0.0` rather than propagating: the scene simply does not advance for
/// that frame. Negative values are impossible in practice but treated the same
/// way, because `Duration::as_secs_f32` on a huge interval can round to
/// infinity and poison the whole camera path.
pub fn clamp_frame_step(elapsed_secs: f32) -> f32 {
    if !elapsed_secs.is_finite() || elapsed_secs <= 0.0 {
        return 0.0;
    }
    elapsed_secs.min(MAX_FRAME_STEP)
}

/// Parses one `/api/scene` event body, or `None` when it is not usable.
///
/// The routing decision ([`crate::scene_snapshot_json`]) picks the event; this
/// turns its body into the renderable snapshot. A truncated or malformed body
/// yields `None` so the caller keeps the scene it already has — one bad event
/// must not blank the viewport.
pub fn parse_scene_body(json: &str) -> Option<ornis_app::SceneSnapshot> {
    ornis_app::parse_scene_json(json).ok()
}

/// Which content height a bounds post is clipped against.
///
/// A page post carries the height the window had when the page measured; when
/// that number is unusable (`0` while minimized, `NaN` from a stale layout) the
/// last good height is kept, so a stale post cannot shrink the viewport to
/// nothing on its own.
pub fn resolve_content_height(reported: f64, previous: f64) -> f64 {
    if reported.is_finite() && reported > 0.0 {
        reported
    } else {
        previous
    }
}

/// The decision one bounds post produces, before anything touches AppKit.
///
/// Splitting the decision from its application is what makes the interesting
/// parts testable without a window: which height wins, whether the box is
/// visible at all, and whether the swap chain has to be reconfigured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundsUpdate {
    /// Height to remember for the next post (see [`resolve_content_height`]).
    pub content_height: f64,
    /// Placement to apply, or `None` when nothing visible remains.
    pub placement: Option<ViewportPlacement>,
    /// Whether the drawable size changed (a reconfigure is needed).
    pub resized: bool,
}

impl BoundsUpdate {
    /// A post that leaves nothing visible: the viewport hides and no surface
    /// work is needed.
    fn hidden(content_height: f64) -> Self {
        Self {
            content_height,
            placement: None,
            resized: false,
        }
    }

    /// A post that positions the viewport, carrying whether the drawable size
    /// changed (`ViewportPlacement::needs_surface_resize`).
    fn visible(
        content_height: f64,
        placement: ViewportPlacement,
        previous: Option<ViewportPlacement>,
    ) -> Self {
        Self {
            content_height,
            placement: Some(placement),
            resized: previous.is_none_or(|previous| previous.needs_surface_resize(placement)),
        }
    }

    /// Decides what a bounds post means, given the previous state.
    ///
    /// `previous_placement` is `None` before the first post and after a
    /// hidden period, which makes the first visible post count as a resize.
    pub fn decide(
        rect: PhysicalRect,
        scale: f32,
        reported_height: f64,
        previous_height: f64,
        previous_placement: Option<ViewportPlacement>,
    ) -> Self {
        let content_height = resolve_content_height(reported_height, previous_height);
        match ViewportPlacement::new(rect, scale, content_height) {
            Some(placement) => Self::visible(content_height, placement, previous_placement),
            None => Self::hidden(content_height),
        }
    }
}

/// What a new placement requires from the AppKit view and the swap chain.
///
/// The interesting rule — "a move must not reallocate GPU targets" — is a pure
/// decision over two placements, so it is testable without a window, a GPU or
/// an `NSView`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementStep {
    /// Same rectangle: nothing to do (a repeated bounds post).
    Unchanged,
    /// Moved, same size: only the view frame changes.
    MoveOnly,
    /// Size changed: view frame and swap chain both move.
    Resize,
}

impl PlacementStep {
    /// Decides the step between the last applied placement and a new one.
    ///
    /// `previous: None` means nothing is placed yet (first post after startup
    /// or after the viewport was hidden), which is always a full `Resize`.
    pub fn decide(previous: Option<ViewportPlacement>, next: ViewportPlacement) -> Self {
        let Some(previous) = previous else {
            return Self::Resize;
        };
        if previous.needs_surface_resize(next) {
            Self::Resize
        } else if previous.needs_frame_change(next) {
            Self::MoveOnly
        } else {
            Self::Unchanged
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: u32, y: u32, width: u32, height: u32) -> PhysicalRect {
        PhysicalRect {
            x,
            y,
            width,
            height,
        }
    }

    fn placed(x: u32, y: u32, width: u32, height: u32) -> ViewportPlacement {
        ViewportPlacement::new(rect(x, y, width, height), 1.0, 800.0).expect("placement")
    }

    #[test]
    fn frame_steps_are_clamped_and_non_finite_input_is_dropped() {
        // A normal 60 fps frame passes through untouched.
        assert!((clamp_frame_step(1.0 / 60.0) - 1.0 / 60.0).abs() < f32::EPSILON);
        // A hitch is capped, not fast-forwarded.
        assert_eq!(clamp_frame_step(5.0), MAX_FRAME_STEP);
        assert_eq!(clamp_frame_step(MAX_FRAME_STEP), MAX_FRAME_STEP);
        // Degenerate clocks advance nothing.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -1.0] {
            assert_eq!(clamp_frame_step(bad), 0.0, "elapsed {bad}");
        }
    }

    #[test]
    fn content_height_falls_back_to_the_last_good_value() {
        // A usable post wins: the window changed size and the page saw it.
        assert_eq!(resolve_content_height(900.0, 800.0), 900.0);
        // A minimized window reports 0, a stale layout may report NaN: neither
        // may shrink the clip box, so the previous height stays.
        assert_eq!(resolve_content_height(0.0, 800.0), 800.0);
        assert_eq!(resolve_content_height(-5.0, 800.0), 800.0);
        assert_eq!(resolve_content_height(f64::NAN, 800.0), 800.0);
        assert_eq!(resolve_content_height(f64::INFINITY, 800.0), 800.0);
        // Nothing known yet: zero, which makes `ViewportPlacement::new` refuse
        // and the caller hide the view instead of guessing a size.
        assert_eq!(resolve_content_height(0.0, 0.0), 0.0);
    }

    #[test]
    fn a_first_bounds_post_is_always_a_resize() {
        let update = BoundsUpdate::decide(rect(0, 0, 800, 600), 2.0, 800.0, 0.0, None);
        assert!(update.resized, "the first placement has no drawable yet");
        assert_eq!(update.content_height, 800.0);
        assert_eq!(
            update.placement.map(|placement| placement.rect),
            Some(rect(0, 0, 800, 600))
        );
    }

    #[test]
    fn a_moved_bounds_post_does_not_reallocate_the_surface() {
        let first = BoundsUpdate::decide(rect(10, 50, 400, 300), 1.0, 800.0, 800.0, None);
        let placed = first.placement.expect("first placement");
        let moved = BoundsUpdate::decide(
            rect(20, 60, 400, 300),
            1.0,
            800.0,
            first.content_height,
            Some(placed),
        );
        assert!(!moved.resized, "a move must not reconfigure the swap chain");
        assert_ne!(
            moved.placement.map(|placement| placement.frame),
            Some(placed.frame),
            "the view frame still moves"
        );

        // A size change does reconfigure, and clipping by a smaller window
        // counts as a size change too.
        let resized = BoundsUpdate::decide(
            rect(10, 50, 401, 300),
            1.0,
            800.0,
            first.content_height,
            Some(placed),
        );
        assert!(resized.resized);
        let clipped = BoundsUpdate::decide(
            rect(10, 50, 400, 300),
            1.0,
            200.0,
            first.content_height,
            Some(placed),
        );
        assert!(clipped.resized);
    }

    #[test]
    fn a_bounds_post_that_leaves_nothing_visible_hides_the_viewport() {
        // Below the window: hidden, and the remembered height is still updated
        // so the next good post clips against the right window.
        let update = BoundsUpdate::decide(rect(0, 900, 800, 100), 1.0, 700.0, 800.0, None);
        assert_eq!(update.placement, None);
        assert!(!update.resized);
        assert_eq!(update.content_height, 700.0);
        // A stale post with an unusable height keeps the last good one.
        let stale = BoundsUpdate::decide(rect(0, 900, 800, 100), 1.0, 0.0, 700.0, None);
        assert_eq!(stale.content_height, 700.0);
    }

    #[test]
    fn a_repeated_bounds_post_is_a_no_op() {
        let first = placed(10, 50, 400, 300);
        assert_eq!(
            PlacementStep::decide(Some(first), first),
            PlacementStep::Unchanged
        );
        // Nothing placed yet: the first post must configure the surface.
        assert_eq!(PlacementStep::decide(None, first), PlacementStep::Resize);
    }

    #[test]
    fn a_move_only_updates_the_frame_and_a_resize_reconfigures() {
        let first = placed(10, 50, 400, 300);
        assert_eq!(
            PlacementStep::decide(Some(first), placed(20, 60, 400, 300)),
            PlacementStep::MoveOnly,
            "a dock drag must not reallocate GPU targets"
        );
        assert_eq!(
            PlacementStep::decide(Some(first), placed(10, 50, 401, 300)),
            PlacementStep::Resize
        );
    }

    #[test]
    fn scene_bodies_parse_or_are_rejected_without_a_gpu() {
        // A real `/api/scene` payload is what the engine publishes on every
        // mutation: the viewport must accept it.
        let snapshot =
            parse_scene_body(CONTRACT).expect("the authoritative contract body must parse");
        assert_eq!(snapshot.scene.entities.len(), 1);
        // A truncated or malformed event leaves the current scene alone
        // rather than blanking the viewport.
        for bad in ["", "{}", "not json", r#"{"version":1}"#] {
            assert!(parse_scene_body(bad).is_none(), "must reject {bad:?}");
        }
        assert!(
            parse_scene_body(&CONTRACT[..CONTRACT.len() / 2]).is_none(),
            "a truncated event must be rejected"
        );
    }

    /// The minimal `/api/scene` body: one renderable entity, a camera and one
    /// light. Kept beside the parser test so a contract change breaks here
    /// first.
    const CONTRACT: &str = r#"{
        "version": 3, "sequence": 7,
        "entities": [{
            "id": 1, "generation": 1,
            "components": {
                "Transform": {"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]},
                "Mesh": {"Sphere": {"radius":1.0,"segments":16,"rings":12}},
                "Material": {"Dielectric": {"base_color":[0.8,0.2,0.2],"roughness":0.5}}
            }
        }],
        "lights": [{"Directional": {"direction":[1,1,1],"intensity":0.6,"color":[1,1,1]}}],
        "camera": {"position":[0,2.5,9],"target":[0,0,0],"up":[0,1,0],"fov":60.0,"near":0.1,"far":100.0},
        "ambient": [0.1,0.1,0.15]
    }"#;
}
