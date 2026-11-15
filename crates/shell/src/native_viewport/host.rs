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

    /// Applies a page-reported rectangle (device pixels) and returns whether
    /// the drawable was (re)configured.
    ///
    /// A rectangle that no longer intersects the window hides the viewport
    /// instead of configuring an empty surface — a collapsed or scrolled-away
    /// box is a normal state, not an error.
    pub fn set_bounds(&mut self, rect: PhysicalRect, scale: f32) -> bool {
        let Some(placement) = ViewportPlacement::new(rect, scale, self.content_height) else {
            self.viewport.hide();
            self.placement = None;
            return false;
        };
        let resized = self
            .placement
            .is_none_or(|previous| previous.needs_surface_resize(placement));
        self.viewport.set_placement(placement);
        self.placement = Some(placement);
        resized
    }

    /// Keeps the placement valid when the window resizes: the page re-posts its
    /// rectangle, and until it does the last one is clipped to the new height.
    pub fn set_content_height(&mut self, height: f64) {
        if height.is_nan() || height <= 0.0 {
            return;
        }
        self.content_height = height;
        if let Some(placement) = self.placement
            && let Some(clipped) = ViewportPlacement::new(placement.rect, 1.0, height)
        {
            self.viewport.set_placement(clipped);
            self.placement = Some(clipped);
        }
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
    pub fn apply_scene_json(&mut self, json: &str) -> bool {
        match ornis_app::parse_scene_json(json) {
            Ok(snapshot) => self.viewport.apply_snapshot(&snapshot),
            Err(_) => false,
        }
    }

    /// Draws one frame; returns `false` when there was nothing to present.
    pub fn render(&mut self, delta: f32) -> bool {
        self.viewport.render(delta)
    }

    /// Whether an input snapshot changes nothing the viewport draws.
    pub fn stats(&self) -> FrameStats {
        self.viewport.stats()
    }
}

/// AppKit view of the window, for hosts that need it directly.
pub fn window_view(window: &winit::window::Window) -> Option<NonNull<NSView>> {
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let handle = window.window_handle().ok()?;
    match handle.as_raw() {
        RawWindowHandle::AppKit(appkit) => Some(appkit.ns_view.cast()),
        _ => None,
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
}
