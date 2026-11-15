//! Page → shell viewport protocol: the viewport rectangle and forwarded input.
//!
//! The embedded page owns the layout: only it knows where the `#viewport` box
//! sits once panels, docks and tabs are laid out, so the shell never guesses.
//! Every layout change posts `{"type":"viewport_bounds","data":{…}}` and the
//! native viewport re-frames itself; pointer/wheel/key state over the same box
//! arrives as `{"type":"viewport_input","data":{…}}` in the documented
//! `POST /api/input` snapshot shape, so one wire shape serves the browser
//! transport, the engine queue and the native viewport.
//!
//! Everything here is pure — no window, no GPU, no channels — so the whole
//! protocol is unit-tested on CI hosts without a display, and the `wry`/AppKit
//! wiring stays in the binary.
//!
//! Unknown `type` tags are not shell messages: [`parse_viewport_message`]
//! returns `None` and the caller keeps the ordinary command path, so a page
//! built before this protocol keeps working.

use editor_backend::BrowserInput;
use serde::Deserialize;

/// Smallest viewport edge the shell accepts, in CSS pixels. A collapsed box
/// (`display:none`, a drag that temporarily zeroes the dock) hides the native
/// viewport instead of configuring a 0-pixel surface.
pub const MIN_VIEWPORT_EDGE: f32 = 1.0;

/// Largest viewport edge the shell accepts, in CSS pixels. Bounds are a sanity
/// clamp against a broken/replayed post, not a layout opinion.
pub const MAX_VIEWPORT_EDGE: f32 = 65_536.0;

/// Lowest accepted device-pixel ratio (browser zoom-out).
pub const MIN_SCALE: f32 = 0.5;

/// Highest accepted device-pixel ratio.
pub const MAX_SCALE: f32 = 8.0;

/// Largest accepted `pressed_*` list; a snapshot is the pressed set, not a
/// log, so a longer list is a malformed post rather than input.
pub const MAX_PRESSED: usize = 64;

/// Viewport rectangle in CSS pixels plus the device-pixel ratio it was
/// measured at.
///
/// Constructed only through [`ViewportBounds::new`], so a value that exists is
/// already finite, non-degenerate and inside the sanity clamps; the physical
/// form the renderer consumes comes from [`ViewportBounds::physical`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewportBounds {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    scale: f32,
}

/// Integer device-pixel rectangle relative to the window's top-left corner.
///
/// This is the value the native viewport applies: whole device pixels, so an
/// unchanged layout yields an equal rectangle and a repeated bounds post is a
/// cheap no-op instead of a surface reconfigure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalRect {
    /// Left edge, device pixels.
    pub x: u32,
    /// Top edge, device pixels.
    pub y: u32,
    /// Width, device pixels (at least 1).
    pub width: u32,
    /// Height, device pixels (at least 1).
    pub height: u32,
}

impl ViewportBounds {
    /// Validates a page-reported rectangle; `None` when it cannot be rendered.
    ///
    /// Rejects non-finite values (a `1e40` post parses as infinity), edges
    /// outside [`MIN_VIEWPORT_EDGE`]..=[`MAX_VIEWPORT_EDGE`] and ratios outside
    /// [`MIN_SCALE`]..=[`MAX_SCALE`]. Negative origins are legal (a dock can
    /// slide the box left before the layout settles) and are clamped by
    /// [`ViewportBounds::clamped_to`].
    pub fn new(x: f32, y: f32, width: f32, height: f32, scale: f32) -> Option<Self> {
        let finite = x.is_finite() && y.is_finite() && width.is_finite() && height.is_finite();
        let sized = (MIN_VIEWPORT_EDGE..=MAX_VIEWPORT_EDGE).contains(&width)
            && (MIN_VIEWPORT_EDGE..=MAX_VIEWPORT_EDGE).contains(&height);
        let scaled = scale.is_finite() && (MIN_SCALE..=MAX_SCALE).contains(&scale);
        (finite && sized && scaled).then_some(Self {
            x,
            y,
            width,
            height,
            scale,
        })
    }

    /// Device-pixel ratio the rectangle was measured at.
    pub fn scale(self) -> f32 {
        self.scale
    }

    /// Device-pixel rectangle, rounded to whole pixels (at least 1×1).
    ///
    /// Origin and size round independently: the box the page reports is the
    /// box the DOM painted, and truncating both edges can lose a pixel column
    /// on fractional layouts.
    pub fn physical(self) -> PhysicalRect {
        PhysicalRect {
            x: self.x.max(0.0).mul_add(self.scale, 0.0).round() as u32,
            y: self.y.max(0.0).mul_add(self.scale, 0.0).round() as u32,
            width: ((self.width * self.scale).round() as u32).max(1),
            height: ((self.height * self.scale).round() as u32).max(1),
        }
    }

    /// Intersects the rectangle with a `window_width`×`window_height` CSS-pixel
    /// window; `None` when nothing visible remains.
    ///
    /// The page can legitimately report a box that overhangs the window (a
    /// dock animation, a stale layout after a window resize). Rendering the
    /// overhang would place device pixels outside the surface, so the shell
    /// clips instead of trusting the post.
    pub fn clamped_to(self, window_width: f32, window_height: f32) -> Option<Self> {
        if !(window_width.is_finite() && window_height.is_finite()) {
            return None;
        }
        let left = self.x.max(0.0);
        let top = self.y.max(0.0);
        let right = (self.x + self.width).min(window_width);
        let bottom = (self.y + self.height).min(window_height);
        let (width, height) = (right - left, bottom - top);
        if !(width.is_finite() && height.is_finite()) {
            return None;
        }
        Self::new(left, top, width, height, self.scale)
    }
}

/// One forwarded input snapshot (`POST /api/input` shape).
///
/// The pressed sets are the authoritative state for the frame — not deltas —
/// so a dropped post cannot leave a key or button stuck; the shell forwards
/// the whole snapshot to the engine and to the native viewport alike.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct ViewportInput {
    /// Pressed key codes (raw `u32`, the platform-neutral wire codes).
    pub pressed_keys: Vec<u32>,
    /// Pressed mouse-button codes.
    pub pressed_mouse_buttons: Vec<u8>,
    /// Pointer position over the viewport, CSS pixels.
    pub pointer_position: [f32; 2],
    /// Pointer movement since the previous snapshot, CSS pixels.
    pub pointer_delta: [f32; 2],
    /// Wheel delta since the previous snapshot (lines).
    pub wheel_delta: f32,
}

impl ViewportInput {
    /// Whether the snapshot is well formed: finite pointer/wheel values and
    /// pressed sets that fit [`MAX_PRESSED`].
    ///
    /// Non-finite deltas are the interesting case — JSON cannot spell `NaN`,
    /// but `1e40` parses to infinity and would poison the camera for the rest
    /// of the session.
    pub fn is_valid(&self) -> bool {
        self.pointer_position.iter().all(|v| v.is_finite())
            && self
                .pointer_delta
                .iter()
                .all(|value| value.is_finite())
            && self.wheel_delta.is_finite()
            && self.pressed_keys.len() <= MAX_PRESSED
            && self.pressed_mouse_buttons.len() <= MAX_PRESSED
    }

    /// Converts a validated snapshot into the engine's browser-input type.
    ///
    /// Returns `None` for an invalid snapshot (see
    /// [`ViewportInput::is_valid`]) instead of repairing it: a repaired
    /// snapshot would silently drop the user's drag.
    pub fn to_browser_input(&self) -> Option<BrowserInput> {
        self.is_valid().then(|| BrowserInput {
            pressed_keys: self.pressed_keys.clone(),
            pressed_mouse_buttons: self.pressed_mouse_buttons.clone(),
            pointer_position: self.pointer_position,
            pointer_delta: self.pointer_delta,
            wheel_delta: self.wheel_delta,
        })
    }
}

/// A parsed shell-level message from the page.
#[derive(Debug, Clone, PartialEq)]
pub enum ViewportMessage {
    /// The viewport box moved or resized.
    Bounds(PhysicalRect),
    /// Pointer/wheel/key state over the viewport box.
    Input(ViewportInput),
}

/// Raw envelope: `{"type": <tag>, "data": <payload>}`.
#[derive(Debug, Deserialize)]
struct Envelope {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    data: serde_json::Value,
}

/// Bounds payload: `{"x","y","width","height","scale"}` in CSS pixels.
#[derive(Debug, Deserialize)]
struct BoundsPayload {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    scale: f32,
}

/// Parses one `window.ipc.postMessage` payload as a shell viewport message.
///
/// Returns `None` for anything that is not this protocol — an engine command,
/// malformed JSON, a stale shape — so the caller's ordinary command path is
/// untouched. A bounds post that fails validation also returns `None`: the
/// native viewport keeps its last good rectangle rather than shrinking to
/// nothing.
pub fn parse_viewport_message(body: &str) -> Option<ViewportMessage> {
    let envelope: Envelope = serde_json::from_str(body).ok()?;
    match envelope.kind.as_str() {
        "viewport_bounds" => {
            let bounds: BoundsPayload = serde_json::from_value(envelope.data).ok()?;
            let bounds = ViewportBounds::new(
                bounds.x,
                bounds.y,
                bounds.width,
                bounds.height,
                bounds.scale,
            )?;
            Some(ViewportMessage::Bounds(bounds.physical()))
        }
        "viewport_input" => {
            let input: ViewportInput = serde_json::from_value(envelope.data).ok()?;
            input.is_valid().then_some(ViewportMessage::Input(input))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(body: &str) -> Option<PhysicalRect> {
        match parse_viewport_message(body) {
            Some(ViewportMessage::Bounds(rect)) => Some(rect),
            _ => None,
        }
    }

    #[test]
    fn bounds_post_parses_and_scales_to_device_pixels() {
        let rect = bounds(
            r#"{"type":"viewport_bounds","data":{"x":320,"y":48,"width":640,"height":480,"scale":2}}"#,
        )
        .expect("valid bounds post parses");
        assert_eq!(
            rect,
            PhysicalRect {
                x: 640,
                y: 96,
                width: 1280,
                height: 960
            }
        );
        // Retina off: CSS pixels are device pixels.
        let plain = bounds(
            r#"{"type":"viewport_bounds","data":{"x":10,"y":20,"width":100,"height":50,"scale":1}}"#,
        )
        .expect("1x bounds post parses");
        assert_eq!(
            plain,
            PhysicalRect {
                x: 10,
                y: 20,
                width: 100,
                height: 50
            }
        );
    }

    #[test]
    fn bounds_post_rejects_non_finite_and_degenerate_values() {
        // `1e40` is valid JSON and parses to infinity: the overflow must not
        // reach a surface configuration.
        for data in [
            r#"{"x":1e40,"y":0,"width":10,"height":10,"scale":1}"#,
            r#"{"x":0,"y":0,"width":0,"height":10,"scale":1}"#,
            r#"{"x":0,"y":0,"width":10,"height":10,"scale":0}"#,
            r#"{"x":0,"y":0,"width":10,"height":10,"scale":1e40}"#,
            r#"{"x":0,"y":0,"width":1e40,"height":10,"scale":1}"#,
        ] {
            let body = format!(r#"{{"type":"viewport_bounds","data":{data}}}"#);
            assert!(bounds(&body).is_none(), "must reject {data}");
        }
        // Unknown / malformed envelopes fall through to the command path.
        assert!(parse_viewport_message(r#"{"type":"ping"}"#).is_none());
        assert!(parse_viewport_message(r#"{"type":"viewport_bounds"}"#).is_none());
        assert!(parse_viewport_message("not json").is_none());
    }

    #[test]
    fn fractional_layouts_round_edges_independently() {
        // 1392.9 CSS px at 1.25 give a 1741.125 device-pixel box: the size
        // rounds up instead of truncating a column away.
        let bounds = ViewportBounds::new(0.5, 0.5, 1392.9, 100.4, 1.25).expect("valid bounds");
        assert_eq!(
            bounds.physical(),
            PhysicalRect {
                x: 1,
                y: 1,
                width: 1741,
                height: 126
            }
        );
    }

    #[test]
    fn clamping_clips_an_overhanging_box_to_the_window() {
        let bounds = ViewportBounds::new(-40.0, -10.0, 300.0, 200.0, 1.0).expect("valid bounds");
        let clipped = bounds.clamped_to(200.0, 100.0).expect("overlap remains");
        assert_eq!(
            clipped.physical(),
            PhysicalRect {
                x: 0,
                y: 0,
                width: 200,
                height: 100
            }
        );
        // Fully outside the window: hidden, never a negative-size surface.
        let outside = ViewportBounds::new(400.0, 400.0, 100.0, 100.0, 1.0).expect("valid bounds");
        assert_eq!(outside.clamped_to(200.0, 200.0), None);
        // A zero-size window (minimized) hides the viewport too.
        assert_eq!(bounds.clamped_to(0.0, 0.0), None);
    }

    #[test]
    fn repeated_bounds_posts_compare_equal() {
        let first = bounds(
            r#"{"type":"viewport_bounds","data":{"x":12,"y":34,"width":500,"height":400,"scale":2}}"#,
        )
        .expect("parses");
        let again = bounds(
            r#"{"type":"viewport_bounds","data":{"x":12,"y":34,"width":500,"height":400,"scale":2}}"#,
        )
        .expect("parses");
        assert_eq!(first, again, "unchanged layout must be a cheap no-op");
    }

    #[test]
    fn input_post_parses_into_the_engine_snapshot() {
        let body = r#"{"type":"viewport_input","data":{
            "pressed_keys":[87,65],
            "pressed_mouse_buttons":[0],
            "pointer_position":[120.5,80.25],
            "pointer_delta":[3.0,-2.0],
            "wheel_delta":1.0
        }}"#;
        let Some(ViewportMessage::Input(input)) = parse_viewport_message(body) else {
            panic!("input post must parse");
        };
        let browser = input.to_browser_input().expect("valid snapshot converts");
        assert_eq!(browser.pressed_keys, vec![87, 65]);
        assert_eq!(browser.pressed_mouse_buttons, vec![0]);
        assert_eq!(browser.pointer_position, [120.5, 80.25]);
        assert_eq!(browser.pointer_delta, [3.0, -2.0]);
        assert_eq!(browser.wheel_delta, 1.0);
    }

    #[test]
    fn input_post_defaults_absent_fields_and_rejects_infinities() {
        let empty = r#"{"type":"viewport_input","data":{}}"#;
        let Some(ViewportMessage::Input(input)) = parse_viewport_message(empty) else {
            panic!("an empty snapshot is a release-all post");
        };
        assert_eq!(input, ViewportInput::default());
        assert!(input.to_browser_input().is_some());

        for data in [
            r#"{"pointer_position":[1e40,0]}"#,
            r#"{"pointer_delta":[0,-1e40]}"#,
            r#"{"wheel_delta":1e40}"#,
        ] {
            let body = format!(r#"{{"type":"viewport_input","data":{data}}}"#);
            assert!(
                parse_viewport_message(&body).is_none(),
                "must reject {data}"
            );
        }
    }

    #[test]
    fn input_post_bounds_the_pressed_sets() {
        let keys: Vec<String> = (0..=MAX_PRESSED).map(|k| k.to_string()).collect();
        let body = format!(
            r#"{{"type":"viewport_input","data":{{"pressed_keys":[{}]}}}}"#,
            keys.join(",")
        );
        assert!(
            parse_viewport_message(&body).is_none(),
            "a snapshot is the pressed set, not a key log"
        );
    }
}
