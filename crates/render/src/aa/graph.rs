//! Native frame-graph adapter; temporal history remains outside the transient pool.
use super::{AaInputs, AaResolve, AntiAliasing};
use crate::frame_passes::Hdr;
use crate::system::{Frame, FramePass, FrameResource, Read, ResourceKind, SystemViews, Write};
use crate::transient_pool::{SizePolicy, TextureSpec};

/// Imported RGBA16Float UV motion/current depth/previous depth.
/// The producer must preserve previous vertex positions, including skinning.
/// Attach its view to the executor before running the graph.
pub struct AaMotion;
impl FrameResource for AaMotion {
    const NAME: &'static str = "aa_motion";
    fn kind() -> ResourceKind {
        ResourceKind::Imported
    }
    fn spec(_: wgpu::TextureFormat) -> TextureSpec {
        TextureSpec {
            format: wgpu::TextureFormat::Rgba16Float,
            samples: 1,
            size: SizePolicy::MatchSurface,
        }
    }
}

/// Linear, single-sampled AA output; feed this to tone mapping, not back into Hdr.
pub struct AaColor;
impl FrameResource for AaColor {
    const NAME: &'static str = "aa_color";
    fn kind() -> ResourceKind {
        ResourceKind::FrameOwned
    }
    fn spec(_: wgpu::TextureFormat) -> TextureSpec {
        TextureSpec {
            format: wgpu::TextureFormat::Rgba16Float,
            samples: 1,
            size: SizePolicy::MatchSurface,
        }
    }
}

/// Opt-in deferred HDR resolve node. Register Hdr, AaMotion and AaColor,
/// add this after lighting, and consume AaColor in a downstream composite.
///
/// The default RenderFrame3D wiring is deliberately unchanged: it has no
/// motion-vector producer yet. For wasm use AaResolve directly (GPU handles
/// there are !Send, whereas the native frame scheduler requires Send passes).
pub struct AaPass {
    resolve: AaResolve,
}

impl AaPass {
    /// Creates a per-camera node with an RGBA16Float linear output.
    pub fn new(device: &wgpu::Device, extent: [u32; 2], mode: AntiAliasing) -> Self {
        Self {
            resolve: AaResolve::new(device, wgpu::TextureFormat::Rgba16Float, extent, mode),
        }
    }

    /// Resizes history and resets accumulation. Call with the frame-plan resize.
    pub fn resize(&mut self, device: &wgpu::Device, extent: [u32; 2]) {
        self.resolve.resize(device, extent);
    }

    /// Resets history for a camera cut or scene replacement.
    pub fn reset_history(&mut self) {
        self.resolve.reset_history();
    }
}

impl FramePass for AaPass {
    type Reads = (Read<Hdr>, Read<AaMotion>);
    type Writes = (Write<AaColor>,);

    fn name(&self) -> &'static str {
        "anti_aliasing"
    }

    fn run<'a>(&mut self, views: SystemViews<'a, Self>, frame: &mut Frame<'a>) {
        let (Some(color), Some(motion), Some(target)) = (
            views.get::<Hdr>(),
            views.get::<AaMotion>(),
            views.get::<AaColor>(),
        ) else {
            return;
        };
        self.resolve.render(
            frame.device,
            frame.encoder,
            AaInputs {
                color,
                motion,
                target,
            },
        );
    }
}
