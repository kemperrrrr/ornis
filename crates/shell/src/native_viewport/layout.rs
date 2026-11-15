//! Where the native viewport view sits and how big its drawable is.
//!
//! Pure geometry, so the whole placement contract is testable on a host with
//! no window and no GPU. Two units meet here and must not be confused:
//!
//! * **points** — what AppKit frames are expressed in. The page measures its
//!   viewport in CSS pixels, which are points on macOS, so the page's
//!   rectangle *is* the view frame (no conversion, no rounding).
//! * **device pixels** — what a `CAMetalLayer` drawable and wgpu's surface
//!   configuration take. Points × backing scale, rounded to whole pixels.
//!
//! The layer itself is never positioned: it is the view's backing layer, so
//! AppKit keeps it equal to the view's bounds and only the view frame moves.
//!
//! Coordinates are the parent view's, which winit makes flipped (origin at
//! the top-left, exactly like a page's `getBoundingClientRect`), so the page's
//! numbers need no vertical flip. The drawable size is derived from the same
//! rectangle, which is what keeps the rendered picture aligned with the
//! transparent hole in the page.

use crate::PhysicalRect;

/// Placement of the native viewport inside the window's content view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewportPlacement {
    /// Drawable rectangle in device pixels (top-left origin).
    pub rect: PhysicalRect,
    /// View frame in points: `(x, y, width, height)`.
    pub frame: ViewFrame,
    /// Validated device-pixel ratio the rectangle was measured at.
    ///
    /// Kept on the placement (rather than re-read from the page) so the
    /// renderer's pointer scale is derived from the exact number the geometry
    /// used: a camera reading a different ratio moves at the wrong speed.
    pub scale: f32,
}

/// AppKit view frame in points (top-left origin, flipped parent).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewFrame {
    /// Horizontal offset, points.
    pub x: f64,
    /// Vertical offset from the top, points.
    pub y: f64,
    /// Width, points.
    pub width: f64,
    /// Height, points.
    pub height: f64,
}

impl ViewportPlacement {
    /// Placement for a page-reported rectangle inside a window whose content
    /// view is `content_height` points tall.
    ///
    /// Returns `None` when nothing visible remains — the caller then hides the
    /// view instead of configuring a zero-size surface. A box that overhangs
    /// the window (a dock mid-animation, a stale layout after a window resize)
    /// is clipped here, not by the caller.
    pub fn new(rect: PhysicalRect, scale: f32, content_height: f64) -> Option<Self> {
        let ratio = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        let scale = f64::from(ratio);
        if rect.width == 0 || rect.height == 0 || content_height.is_nan() || content_height <= 0.0 {
            return None;
        }
        // Back to points for the frame, then clip vertically to the content
        // area so the view never extends past the window's top or bottom.
        let width = f64::from(rect.width) / scale;
        let height = f64::from(rect.height) / scale;
        let top = (f64::from(rect.y) / scale).max(0.0);
        let visible = (top + height).min(content_height) - top;
        if !visible.is_finite() || visible <= 0.0 {
            return None;
        }
        let frame = ViewFrame {
            x: f64::from(rect.x) / scale,
            y: top,
            width,
            height: visible,
        };
        Some(Self {
            rect: PhysicalRect {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                // Clipping to the content area can only shrink the box.
                height: ((visible * scale).round() as u32).clamp(1, rect.height),
            },
            frame,
            scale: ratio,
        })
    }

    /// Whether the swap chain must be reconfigured for `other`.
    ///
    /// Moving alone does not resize the drawable, so a dock drag only moves
    /// the view instead of reallocating every GPU target per mouse move.
    pub fn needs_surface_resize(self, other: Self) -> bool {
        self.rect.width != other.rect.width || self.rect.height != other.rect.height
    }

    /// Whether the view must be moved (position or size changed).
    pub fn needs_frame_change(self, other: Self) -> bool {
        self.frame != other.frame
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

    #[test]
    fn frame_is_the_page_rectangle_in_points() {
        // Retina: the drawable is twice the frame, the frame is untouched.
        let placement =
            ViewportPlacement::new(rect(640, 96, 1280, 960), 2.0, 800.0).expect("placement");
        assert_eq!(
            placement.frame,
            ViewFrame {
                x: 320.0,
                y: 48.0,
                width: 640.0,
                height: 480.0
            }
        );
        assert_eq!(
            placement.rect,
            rect(640, 96, 1280, 960),
            "the drawable keeps the device-pixel size"
        );
        // 1x: points and pixels agree.
        let plain = ViewportPlacement::new(rect(10, 20, 100, 50), 1.0, 800.0).expect("placement");
        assert_eq!(
            plain.frame,
            ViewFrame {
                x: 10.0,
                y: 20.0,
                width: 100.0,
                height: 50.0
            }
        );
    }

    #[test]
    fn a_box_overhanging_the_window_is_clipped() {
        // 300 points tall, top at 700 in an 800-point window: 100 visible.
        let clipped = ViewportPlacement::new(rect(0, 700, 800, 300), 1.0, 800.0)
            .expect("partial intersection");
        assert_eq!(clipped.frame.height, 100.0);
        assert_eq!(clipped.rect.height, 100);
        // Entirely below the window, or a collapsed box: no placement.
        assert_eq!(
            ViewportPlacement::new(rect(0, 900, 800, 100), 1.0, 800.0),
            None
        );
        assert_eq!(ViewportPlacement::new(rect(0, 0, 0, 100), 1.0, 800.0), None);
        assert_eq!(ViewportPlacement::new(rect(0, 0, 100, 0), 1.0, 800.0), None);
        // A minimized window has no content area at all.
        assert_eq!(ViewportPlacement::new(rect(0, 0, 100, 100), 1.0, 0.0), None);
    }

    #[test]
    fn degenerate_scale_falls_back_to_one() {
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let placement =
                ViewportPlacement::new(rect(10, 20, 100, 50), scale, 800.0).expect("placement");
            assert_eq!(placement.frame.width, 100.0, "scale {scale} must not scale");
        }
    }

    #[test]
    fn moving_does_not_reallocate_but_resizing_does() {
        let first = ViewportPlacement::new(rect(10, 50, 400, 300), 1.0, 800.0).expect("placement");
        let moved = ViewportPlacement::new(rect(20, 60, 400, 300), 1.0, 800.0).expect("placement");
        assert!(!first.needs_surface_resize(moved));
        assert!(first.needs_frame_change(moved));
        let resized =
            ViewportPlacement::new(rect(10, 50, 401, 300), 1.0, 800.0).expect("placement");
        assert!(first.needs_surface_resize(resized));
        // Clipping by a shrinking window is a resize too.
        let clipped =
            ViewportPlacement::new(rect(10, 50, 400, 300), 1.0, 200.0).expect("placement");
        assert!(first.needs_surface_resize(clipped));
        // A repeated, unchanged post is a full no-op.
        let again = ViewportPlacement::new(rect(10, 50, 400, 300), 1.0, 800.0).expect("placement");
        assert!(!first.needs_frame_change(again));
        assert!(!first.needs_surface_resize(again));
    }
}
