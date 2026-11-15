//! Native wgpu viewport under the transparent webview (PLAN.md §j, вариант 1).
//!
//! The webview still owns the whole editor UI; the shell reserves a
//! *transparent* rectangle where `#viewport` sits and draws the scene into a
//! `CAMetalLayer`-backed `NSView` placed *below* the webview. Nothing is
//! copied through the page: the GPU surface presents straight to the window,
//! so the frame path is the same one `ornis-runner` uses.
//!
//! Split by design, because CI has no GPU and no window:
//!
//! * [`layout`] — where the view goes and how big its drawable is. Pure
//!   arithmetic over [`PhysicalRect`](crate::PhysicalRect); unit-tested.
//! * `frame` (macOS) — the scene→GPU frame state (replica world, renderer,
//!   plan, camera). Needs a `wgpu::Device`, so it lives behind `#[cfg(target_os
//!   = "macos")]` like the surface it drives.
//! * `platform` (macOS) — the AppKit `NSView`/`CAMetalLayer` plumbing.
//! * `host` (macOS) — attaching, placing and feeding the viewport from the
//!   shell binary.
//!
//! Only macOS has the surface path here: the problem being solved is
//! WKWebView-on-macOS, and a Linux/Windows child-window equivalent is a
//! different surface target (`build_as_child` + `X11`/`HWND`) that this spike
//! deliberately does not guess at. On other hosts the module compiles down to
//! the pure layout helpers, so the shell still builds and tests everywhere.

pub mod layout;
pub mod logic;

#[cfg(target_os = "macos")]
pub mod frame;
#[cfg(target_os = "macos")]
pub mod host;
#[cfg(target_os = "macos")]
pub mod platform;

#[cfg(target_os = "macos")]
pub use frame::{FrameStats, SurfaceSpec, ViewportFrame};
#[cfg(target_os = "macos")]
pub use host::{HostError, ViewportHost};
pub use layout::{ViewFrame, ViewportPlacement};
pub use logic::{BoundsUpdate, MAX_FRAME_STEP, PlacementStep, clamp_frame_step};
#[cfg(target_os = "macos")]
pub use logic::{parse_scene_body, resolve_content_height};
#[cfg(target_os = "macos")]
pub use platform::{NativeViewport, ViewportError};
