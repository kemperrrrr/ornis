//! AppKit plumbing: the `CAMetalLayer`-backed view under the webview.
//!
//! The webview stays the window's UI surface. This module inserts one `NSView`
//! as a sibling *below* it and gives that view a `CAMetalLayer` as its backing
//! layer: wgpu renders the editor scene into the layer, the page paints its
//! viewport rectangle transparent, and every other page pixel keeps covering
//! the layer. That ordering is the whole trick (PLAN.md §j, вариант 1).
//!
//! Why a sibling and not a child of the webview: wry's `build_as_child` view
//! is winit's own view (winit#4203/wry#1477), and the page must never draw
//! *over* the layer — the UI overlay requirement is exactly "webview above,
//! viewport below". A sibling ordered `Below` gives both for free.
//!
//! Every Objective-C call is `unsafe` and carries its own SAFETY note; the
//! geometry it applies was validated in [`super::layout`] without any of this.

use std::ptr::NonNull;

use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2_app_kit::{NSView, NSWindowOrderingMode};
use objc2_foundation::{NSPoint, NSRect, NSSize};
use objc2_quartz_core::CAMetalLayer;
use raw_window_handle::{
    AppKitDisplayHandle, AppKitWindowHandle, RawDisplayHandle, RawWindowHandle,
};

use super::frame::{FrameStats, SurfaceSpec, ViewportFrame};
use super::layout::ViewportPlacement;
use super::logic::PlacementStep;

/// Storage buffers the render shader stages require (same as the runner).
const MAX_STORAGE_BUFFERS_PER_STAGE: u32 = 8;

/// Why the native viewport could not be created or reconfigured.
///
/// Typed rather than a string so the binary can decide what to do: a missing
/// adapter means "leave the page's written notice in place", a failed surface
/// means "report it and keep the webview-only shell".
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ViewportError {
    /// The AppKit view was not reachable on the main thread.
    #[error("native viewport must be created on the main thread")]
    OffMainThread,
    /// No wgpu adapter on this machine.
    #[error("no wgpu adapter for the native viewport")]
    NoAdapter,
    /// The adapter refused a device.
    #[error("wgpu device request failed: {reason}")]
    NoDevice {
        /// Underlying failure.
        reason: String,
    },
    /// A Metal surface could not be created for the layer.
    #[error("wgpu surface creation failed: {reason}")]
    Surface {
        /// Underlying failure.
        reason: String,
    },
}

/// GPU half of the viewport: the surface for one view plus the device, queue,
/// swap-chain configuration and frame state that draw into it.
///
/// Split from [`NativeViewport`] so neither the AppKit setup nor the adapter
/// negotiation is one long function; the fields are private and move together.
struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    frame: ViewportFrame,
}

impl Gpu {
    /// Creates the Metal surface for `view`, negotiates a device over it and
    /// builds the frame state. The 1×1 seed configuration is replaced by the
    /// first page-reported placement.
    ///
    /// # Safety
    ///
    /// `view` must be a live `NSView` on the main thread that outlives the
    /// returned surface.
    // qual:allow(unsafe) reason: "Metal surface creation takes a raw NSView handle by contract; the block below cannot be made safe and carries its SAFETY note"
    unsafe fn for_view(view: &NSView) -> Result<Self, ViewportError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            flags: wgpu::InstanceFlags::empty(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        // SAFETY: the caller guarantees `view` is live and outlives the
        // surface. `Layer::from_ns_view` (inside wgpu) makes the view
        // layer-backed and attaches its `CAMetalLayer` as the root layer.
        let surface = unsafe {
            let handle = AppKitWindowHandle::new(NonNull::from(view).cast());
            instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(RawDisplayHandle::AppKit(AppKitDisplayHandle::new())),
                raw_window_handle: RawWindowHandle::AppKit(handle),
            })
        }
        .map_err(|e| ViewportError::Surface {
            reason: e.to_string(),
        })?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }))
        .map_err(|_| ViewportError::NoAdapter)?;
        let mut limits = adapter.limits();
        limits.max_storage_buffers_per_shader_stage = MAX_STORAGE_BUFFERS_PER_STAGE;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("shell viewport device"),
            required_features: ornis_render::texture_upload::texture_compression_features(
                adapter.features(),
            ),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| ViewportError::NoDevice {
            reason: e.to_string(),
        })?;
        let format = choose_format(&surface.get_capabilities(&adapter).formats);
        let config = swap_chain_config(format, 1, 1);
        surface.configure(&device, &config);
        let frame = ViewportFrame::new(
            device.clone(),
            queue.clone(),
            SurfaceSpec {
                format,
                width: 1,
                height: 1,
            },
        );
        Ok(Self {
            surface,
            device,
            queue,
            config,
            frame,
        })
    }
}

/// Swap-chain format: sRGB first (the renderer's output is sRGB-encoded), then
/// whatever the surface offers, then the platform default.
fn choose_format(formats: &[wgpu::TextureFormat]) -> wgpu::TextureFormat {
    formats
        .iter()
        .copied()
        .find(wgpu::TextureFormat::is_srgb)
        .or_else(|| formats.first().copied())
        .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb)
}

/// Swap-chain configuration for one drawable size.
///
/// Present mode follows [`ornis_render::surface_present_mode`], so the viewport
/// obeys `ORNIS_NO_VSYNC` exactly like the runner's own window.
fn swap_chain_config(
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: width.max(1),
        height: height.max(1),
        present_mode: ornis_render::surface_present_mode(),
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    }
}

/// Native viewport: one layer-backed `NSView` below the webview plus the GPU
/// state that draws into it.
///
/// # Drop order
///
/// Fields are dropped in declaration order, so `view` comes **last**: the
/// `wgpu::Surface` was created from a raw handle into that view's layer
/// ([`NativeViewport::new`]) and must be destroyed first. Reading top to
/// bottom, everything a surface depends on (itself, the device, the queue, the
/// frame state) goes before the `NSView` it borrows from — that is the
/// invariant `create_surface_unsafe` was called under, and keeping `view` last
/// is what makes it hold at teardown instead of relying on the superview to
/// outlive the struct by luck.
///
/// `Drop` additionally detaches the view from its superview, so a destroyed
/// viewport leaves no hidden layer-backed `NSView` under the webview.
pub struct NativeViewport {
    /// Surface created from the view's layer. Declared first: dropped first.
    surface: wgpu::Surface<'static>,
    /// Device the surface and the frame state share.
    device: wgpu::Device,
    /// Queue that uploads and presents (the frame state submits through it).
    queue: wgpu::Queue,
    /// Swap-chain configuration applied to `surface`.
    config: wgpu::SurfaceConfiguration,
    /// Frame state (renderer, plan, replica world).
    frame: ViewportFrame,
    /// Placement last applied, to skip a redundant move/resize.
    placement: Option<ViewportPlacement>,
    /// The inserted view. **Last on purpose** (see the type docs): the surface
    /// above is built on this view's layer, so it must outlive it at teardown.
    /// `NSView` is also retained by its superview, but the field is what keeps
    /// the raw pointer wgpu was handed valid for the surface's lifetime.
    view: Retained<NSView>,
}

impl NativeViewport {
    /// Creates the layer view under `parent` and the GPU state that draws into
    /// it.
    ///
    /// `parent` is the view wry installed the webview into. The new view is
    /// ordered *below* it, so every page pixel keeps covering the layer.
    /// Returns [`ViewportError::NoAdapter`] on a host without a GPU instead of
    /// panicking: the shell then leaves the page's written notice in place.
    ///
    /// # Safety
    ///
    /// `parent` and `sibling` must be live `NSView`s on the main thread that
    /// share a window, and `parent` must outlive the returned viewport (the
    /// view is inserted into it).
    // qual:allow(unsafe) reason: "AppKit object graph only exposes raw view pointers; each block below carries its own SAFETY note"
    pub unsafe fn new(
        parent: NonNull<NSView>,
        sibling: Option<NonNull<NSView>>,
        scale: f64,
    ) -> Result<Self, ViewportError> {
        let mtm = MainThreadMarker::new().ok_or(ViewportError::OffMainThread)?;
        // SAFETY: the caller guarantees both views are live `NSView`s on the
        // main thread (checked above).
        let parent: &NSView = unsafe { parent.as_ref() };
        let scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };

        let view = NSView::new(mtm);
        view.setWantsLayer(true);
        let layer = CAMetalLayer::new();
        layer.setContentsScale(scale);
        // `setLayer` needs objc2-app-kit's `objc2-quartz-core` feature (the
        // `CALayer` bridge), so the layer is attached through the message-send
        // path the renderer's own layer setup uses.
        // SAFETY: `view` is a live `NSView`, `layer` is a live `CAMetalLayer`,
        // and `setLayer:` is an AppKit setter taking one object argument.
        unsafe {
            let _: () = objc2::msg_send![&view, setLayer: &*layer];
        }
        view.setHidden(true);
        // Below the webview: page pixels cover the layer everywhere except the
        // transparent viewport region. Without a sibling the view still goes
        // below the parent's existing subviews, which is the same ordering for
        // a window whose content view holds only the webview.
        // SAFETY: the caller guarantees a `Some` sibling is a live `NSView`.
        let anchor = sibling.map(|sibling| unsafe { sibling.as_ref() });
        parent.addSubview_positioned_relativeTo(&view, NSWindowOrderingMode::Below, anchor);

        // SAFETY: `view` is a live, layer-backed `NSView` that outlives the
        // surface — `view` is the **last** field of `Self` (see the type docs),
        // so the surface is destroyed before it. `Gpu::for_view` only stores a
        // surface built from this view's layer.
        let gpu = unsafe { Gpu::for_view(&view)? };
        Ok(Self {
            surface: gpu.surface,
            device: gpu.device,
            queue: gpu.queue,
            config: gpu.config,
            frame: gpu.frame,
            placement: None,
            view,
        })
    }

    /// Applies a page-reported placement: moves the view, resizes the drawable
    /// only when the size actually changed, and keeps the frame in sync.
    ///
    /// Moving alone never reconfigures the swap chain (see
    /// [`ViewportPlacement::needs_surface_resize`]) — a dock drag must not
    /// reallocate GPU targets on every mouse move.
    // qual:allow(test_quality, untested) reason: "applies PlacementStep to AppKit + wgpu; the step decision and the geometry are host-tested"
    pub fn set_placement(&mut self, placement: ViewportPlacement) {
        self.view.setHidden(false);
        match PlacementStep::decide(self.placement, placement) {
            PlacementStep::Unchanged => {}
            PlacementStep::MoveOnly => {
                self.view.setFrame(frame_rect(placement));
                self.placement = Some(placement);
            }
            PlacementStep::Resize => {
                self.view.setFrame(frame_rect(placement));
                // One call keeps the drawable and the pointer scale in step:
                // both come from the same page rectangle, and a camera that
                // scaled by a stale ratio would move at the wrong speed after a
                // DPI change.
                self.frame.resize_scaled(
                    placement.rect.width.max(1),
                    placement.rect.height.max(1),
                    placement.scale,
                );
                self.config.width = placement.rect.width.max(1);
                self.config.height = placement.rect.height.max(1);
                self.surface.configure(&self.device, &self.config);
                self.placement = Some(placement);
            }
        }
    }

    /// Hides the view (the page collapsed the viewport, or fell back).
    pub fn hide(&mut self) {
        self.view.setHidden(true);
    }

    /// Applies one `/api/scene` snapshot.
    pub fn apply_snapshot(&mut self, snapshot: &ornis_app::SceneSnapshot) -> bool {
        self.frame.apply_snapshot(snapshot)
    }

    /// Forwards one input snapshot.
    pub fn apply_input(&mut self, input: &editor_backend::BrowserInput) {
        self.frame.apply_input(input);
    }

    /// Draws and presents one frame.
    ///
    /// Returns `false` when the surface was out of date (the swap chain is
    /// reconfigured and the next frame draws into the fresh one) — the same
    /// recovery contract as the runner's `RenderPresent`.
    pub fn render(&mut self, delta: f32) -> bool {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return false;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return false,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.frame.render(&view, delta);
        // wgpu 30 presents through the queue: the acquired texture is
        // returned to the swap chain and the queue flushes it.
        self.queue.present(frame);
        true
    }

    /// Diagnostics of the last applied frame.
    pub fn stats(&self) -> FrameStats {
        self.frame.stats()
    }
}

impl Drop for NativeViewport {
    /// Detaches the view from its superview before the fields go.
    ///
    /// Without this the `NSView` would stay in the webview's superview as a
    /// hidden, layer-backed sibling holding a `CAMetalLayer` and whatever the
    /// last presented drawable left in it — invisible, but still composited and
    /// still retaining GPU resources for as long as the window lives.
    ///
    /// The other fields are dropped by the compiler in declaration order right
    /// after this returns, with `view` last (see the type docs); detaching here
    /// cannot invalidate the surface, because `removeFromSuperview` only
    /// unlinks the view from its parent — the view object itself stays alive
    /// until its own field is dropped.
    fn drop(&mut self) {
        // `removeFromSuperview` unlinks the view from whatever superview holds
        // it; with no superview it is a documented no-op, so this is safe and
        // needs no SAFETY note.
        self.view.removeFromSuperview();
    }
}

/// AppKit frame for a placement, in points.
fn frame_rect(placement: ViewportPlacement) -> NSRect {
    let frame = placement.frame;
    NSRect::new(
        NSPoint::new(frame.x, frame.y),
        NSSize::new(frame.width.max(1.0), frame.height.max(1.0)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // The AppKit/surface paths need macOS, a window and a GPU, so CI can only
    // pin the parts that do not touch them. What is testable here is the error
    // vocabulary the binary branches on.
    #[test]
    fn viewport_errors_are_distinguishable() {
        assert_eq!(
            ViewportError::NoAdapter.to_string(),
            "no wgpu adapter for the native viewport"
        );
        let surface = ViewportError::Surface {
            reason: "boom".into(),
        };
        assert!(surface.to_string().contains("boom"));
        assert_ne!(ViewportError::NoAdapter, surface);
        assert_ne!(ViewportError::OffMainThread, surface);
    }

    #[test]
    fn format_selection_prefers_srgb_then_falls_back() {
        use wgpu::TextureFormat;
        // The renderer writes sRGB-encoded colour, so an sRGB swap chain is
        // the correct default.
        assert_eq!(
            choose_format(&[TextureFormat::Bgra8Unorm, TextureFormat::Bgra8UnormSrgb]),
            TextureFormat::Bgra8UnormSrgb
        );
        // No sRGB on offer: take what the surface has.
        assert_eq!(
            choose_format(&[TextureFormat::Rgba8Unorm]),
            TextureFormat::Rgba8Unorm
        );
        // No formats at all: the platform default, never a panic.
        assert_eq!(choose_format(&[]), TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn swap_chain_config_never_has_a_zero_edge() {
        let config = swap_chain_config(wgpu::TextureFormat::Bgra8UnormSrgb, 0, 0);
        assert_eq!((config.width, config.height), (1, 1));
        // Present mode is the shared policy (`ORNIS_NO_VSYNC` opt-out), not a
        // second opinion invented here.
        assert_eq!(config.present_mode, ornis_render::surface_present_mode());
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Auto);
        assert_eq!(config.desired_maximum_frame_latency, 2);
    }

    #[test]
    fn frame_rect_keeps_a_positive_size() {
        let placement = ViewportPlacement::new(
            crate::PhysicalRect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            1.0,
            800.0,
        )
        .expect("placement");
        let rect = frame_rect(placement);
        assert!(rect.size.width >= 1.0 && rect.size.height >= 1.0);
        assert_eq!(rect.origin.x, 0.0);
    }
}
