//! The native viewport as the shell's binary uses it: attach, place, feed,
//! draw.
//!
//! [`ViewportHost`] is the seam between `winit`/`wry` and the viewport: it owns
//! the [`NativeViewport`] (or nothing at all, when this machine has no GPU) and
//! answers the four things the shell receives — a page-reported rectangle,
//! forwarded input, an authoritative scene snapshot, and a frame tick.
//!
//! The scene comes from the *same* event stream the page consumes
//! (`GameEvent::CustomEvent { cmd_type: "scene" }`, published by the engine
//! thread), so the native viewport and the browser viewport can never show
//! different scenes: one source, one parser, two renderers.
//!
//! macOS-only, like the surface underneath it.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2_app_kit::{NSView, NSWindow};
use wry::WebViewExtMacOS;

use super::frame::FrameStats;
use super::layout::ViewportPlacement;
use super::platform::{NativeViewport, ViewportError};
use crate::{PhysicalRect, ViewportInput};

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
/// The routing decision (`scene_snapshot_json` in `crate::viewport_ipc`) picks
/// the event; this turns its body into the renderable snapshot. Separated from
/// [`ViewportHost::apply_scene_json`] so the accept/reject contract — including
/// "a truncated event leaves the current scene alone" — is testable without a
/// GPU, a window or an AppKit view.
pub fn parse_scene_body(json: &str) -> Option<ornis_app::SceneSnapshot> {
    ornis_app::parse_scene_json(json).ok()
}

/// The decision one bounds post produces, before anything touches AppKit.
///
/// Splitting the decision from its application is what makes the interesting
/// parts testable without a window: which height wins, whether the box is
/// visible at all, and whether the swap chain has to be reconfigured. The
/// `ViewportHost` then only applies `placement` to the view and surface.
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

/// Why the shell could not enable the native viewport.
///
/// The shell treats all of these the same way — keep the page, show its
/// written notice — but reports the specific cause, because "no GPU" and "the
/// window went away" want different fixes.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HostError {
    /// No AppKit view on the window (not macOS, or the window is gone).
    #[error("window has no AppKit view")]
    NoView,
    /// The GPU side refused (see [`ViewportError`]).
    #[error("{0}")]
    Viewport(#[from] ViewportError),
}

/// The native viewport plus the state the shell feeds it.
pub struct ViewportHost {
    viewport: NativeViewport,
    /// Last placement applied, so a repeated bounds post is a no-op.
    placement: Option<ViewportPlacement>,
    /// Content-view height in points, for vertical clipping.
    content_height: f64,
}

impl ViewportHost {
    /// Creates the layer under `webview` and the GPU state for it.
    ///
    /// The view is inserted into the webview's *superview* ordered below the
    /// webview, which is what makes the page paint over the layer everywhere
    /// except its transparent viewport rectangle.
    ///
    /// # Errors
    ///
    /// [`HostError::NoView`] when the webview has no AppKit sibling to order
    /// against; otherwise the GPU refusal from [`NativeViewport::new`].
    // qual:allow(unsafe) reason: "AppKit object graph access — each block below carries its own SAFETY note; there is no safe wrapper for view/subview pointers"
    // qual:allow(test_quality, untested) reason: "AppKit/window edge; the error vocabulary (HostError) and every geometry decision are host-tested, creating an NSView needs macOS"
    pub fn attach(webview: &wry::WebView, scale: f64) -> Result<Self, HostError> {
        // `WryWebView` is wry's `WKWebView` subclass; only the `NSView` half is
        // needed (parent, sibling, window).
        // Two class hops: `WryWebView` -> `WKWebView` -> `NSView`.
        let wry_view = webview.webview();
        let webview_view: Retained<objc2_web_kit::WKWebView> = Retained::into_super(wry_view);
        let ns_view: Retained<NSView> = Retained::into_super(webview_view);
        // SAFETY: `ns_view` is the live webview's `NSView`; `superview` only
        // reads the (nullable) parent pointer AppKit already holds.
        let Some(parent) = (unsafe { ns_view.superview() }) else {
            return Err(HostError::NoView);
        };
        let content_height = ns_view.window().map_or(0.0, |window: Retained<NSWindow>| {
            window.contentLayoutRect().size.height
        });
        // SAFETY: `parent` and `ns_view` come from the live webview, and both
        // outlive the viewport (it is inserted into `parent`).
        let viewport = unsafe {
            NativeViewport::new(
                NonNull::from(&*parent),
                Some(NonNull::from(&*ns_view)),
                scale,
            )
        }?;
        Ok(Self {
            viewport,
            placement: None,
            content_height,
        })
    }

    /// Applies a page-reported rectangle (device pixels) measured inside a
    /// window whose content view is `content_height` points tall.
    ///
    /// Both numbers come from the same moment — the page posts after layout,
    /// the binary knows the window size — so the host keeps no stale height
    /// state. A rectangle that no longer intersects the window (a collapsed
    /// dock, a scrolled-away box, a shrunken window) hides the viewport rather
    /// than configuring an empty surface: that is a normal state, not an error.
    ///
    /// Returns whether the drawable was (re)configured, so the caller can tell
    /// a move-only update from a real resize.
    // qual:allow(test_quality, untested) reason: "applies BoundsUpdate to a live NSView + surface; the whole decision it wraps is host-tested (a_first_bounds_post_is_always_a_resize, a_moved_bounds_post_does_not_reallocate_the_surface)"
    pub fn set_bounds(&mut self, rect: PhysicalRect, scale: f32, content_height: f64) -> bool {
        let update = BoundsUpdate::decide(
            rect,
            scale,
            content_height,
            self.content_height,
            self.placement,
        );
        self.content_height = update.content_height;
        let Some(placement) = update.placement else {
            self.viewport.hide();
            self.placement = None;
            return false;
        };
        self.viewport.set_placement(placement);
        self.placement = Some(placement);
        update.resized
    }

    /// Forwards one validated input snapshot.
    pub fn apply_input(&mut self, input: &ViewportInput) {
        if let Some(browser) = input.to_browser_input() {
            self.viewport.apply_input(&browser);
        }
    }

    /// Applies one authoritative `/api/scene` snapshot body.
    ///
    /// A malformed body is ignored (the previous scene stays): the page reports
    /// parse errors on its own channel, and a viewport must never go blank
    /// because one event was truncated.
    // qual:allow(test_quality, untested) reason: "the parse half (parse_scene_body) is host-tested; this only hands the snapshot to the GPU frame"
    pub fn apply_scene_json(&mut self, json: &str) -> bool {
        let Some(snapshot) = parse_scene_body(json) else {
            return false;
        };
        self.viewport.apply_snapshot(&snapshot)
    }

    /// Draws one frame; returns `false` when there was nothing to present.
    pub fn render(&mut self, delta: f32) -> bool {
        self.viewport.render(delta)
    }

    /// Diagnostics of the last applied frame (status line, logs).
    pub fn stats(&self) -> FrameStats {
        self.viewport.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Attaching needs a window and a GPU, so CI pins the error vocabulary and
    // the "a collapsed box hides instead of configuring a 0-pixel surface"
    // rule (the latter through `ViewportPlacement`, which is pure).
    #[test]
    fn host_errors_are_typed() {
        let error = HostError::NoView;
        assert_eq!(error.to_string(), "window has no AppKit view");
        let wrapped = HostError::from(ViewportError::NoAdapter);
        assert!(wrapped.to_string().contains("no wgpu adapter"));
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
    fn a_collapsed_rectangle_has_no_placement() {
        assert_eq!(
            ViewportPlacement::new(
                PhysicalRect {
                    x: 0,
                    y: 0,
                    width: 0,
                    height: 100
                },
                1.0,
                800.0
            ),
            None
        );
    }

    fn rect(x: u32, y: u32, width: u32, height: u32) -> PhysicalRect {
        PhysicalRect {
            x,
            y,
            width,
            height,
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
