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
use super::logic::{BoundsUpdate, parse_scene_body};
use super::platform::{NativeViewport, ViewportError};
use crate::{PhysicalRect, ViewportInput};

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

    // macOS-only: the GPU/AppKit paths cannot run on CI, so what is pinned
    // here is the error vocabulary the binary branches on. Every decision this
    // module applies is tested in `super::logic`, which compiles everywhere.
    #[test]
    fn host_errors_are_typed() {
        let error = HostError::NoView;
        assert_eq!(error.to_string(), "window has no AppKit view");
        let wrapped = HostError::from(ViewportError::NoAdapter);
        assert!(wrapped.to_string().contains("no wgpu adapter"));
        assert_ne!(HostError::NoView, wrapped);
    }
}
