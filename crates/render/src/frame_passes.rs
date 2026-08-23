//! Typed resources and static passes for the 3D frame plan — S2a
//! (IDEAS §28.1, PLAN Приложение C).
//!
//! Resource identity is a type ([`FrameResource`]); passes with static
//! access sets are [`FramePass`] implementations whose wiring is derived
//! from `Reads`/`Writes`. Specs and names mirror the imperative wiring
//! exactly, so layout dumps (and the parity test in `frame_exec.rs`)
//! stay identical. The conditional passes (forward, bloom_down0,
//! composite) remain imperative until S2b.

use crate::frame_plan::{SizePolicy, TextureSpec};
use crate::renderer::{CompositeInputs, GbufferTargets};
use crate::system::{
    AccessSet, ClearBlack, ClearTransparent, ClearWhite, Frame, FramePass, FrameResource, Read,
    ResourceKind, SystemViews, ViewsFor, Write, WriteClear,
};
use std::marker::PhantomData;
// Short alias keeps `typed_resource!` invocations under rustfmt's
// fn_call_width (60) so they stay on one line.
use wgpu::TextureFormat as F;

// ── S2: typed resources (IDEAS §28.1, PLAN Приложение C) ─────────────
// A resource's identity is its type; specs/names mirror the imperative
// wiring exactly so layout dumps (and the parity test) stay identical.

macro_rules! typed_resource {
    ($t:ident, $name:literal, owned, $format:expr) => {
        impl FrameResource for $t {
            const NAME: &'static str = $name;
            fn kind() -> ResourceKind {
                ResourceKind::FrameOwned
            }
            fn spec(_: wgpu::TextureFormat) -> TextureSpec {
                TextureSpec {
                    format: $format,
                    samples: 1,
                    size: SizePolicy::MatchSurface,
                }
            }
        }
    };
    ($t:ident, $name:literal, owned_fraction, $format:expr, $divisor:expr) => {
        impl FrameResource for $t {
            const NAME: &'static str = $name;
            fn kind() -> ResourceKind {
                ResourceKind::FrameOwned
            }
            fn spec(_: wgpu::TextureFormat) -> TextureSpec {
                TextureSpec {
                    format: $format,
                    samples: 1,
                    size: SizePolicy::Fraction($divisor),
                }
            }
        }
    };
}

/// G-buffer albedo layer.
pub struct Albedo;
typed_resource!(Albedo, "albedo", owned, F::Rgba8Unorm);

/// G-buffer world-space normal layer.
pub struct Normal;
typed_resource!(Normal, "normal", owned, F::Rg16Float);

/// G-buffer material id layer.
pub struct MaterialId;
typed_resource!(MaterialId, "material_id", owned, F::R32Uint);

/// G-buffer world-space position layer.
pub struct WorldPosition;
typed_resource!(WorldPosition, "world_position", owned, F::Rg16Float);

/// G-buffer material params layer.
pub struct MaterialParams;
typed_resource!(MaterialParams, "material_params", owned, F::Rgba16Float);

/// Depth buffer.
pub struct Depth;
typed_resource!(Depth, "depth", owned, F::Depth32Float);

/// HDR layer of the deferred path — mirrors the surface format.
pub struct Hdr;
impl FrameResource for Hdr {
    const NAME: &'static str = "hdr";
    fn kind() -> ResourceKind {
        ResourceKind::FrameOwned
    }
    fn spec(surface_format: wgpu::TextureFormat) -> TextureSpec {
        TextureSpec {
            format: surface_format,
            samples: 1,
            size: SizePolicy::MatchSurface,
        }
    }
}

/// HDR layer of the forward path.
pub struct HdrFwd;
typed_resource!(HdrFwd, "hdr_fwd", owned, F::Rgba16Float);

/// Swapchain target (externally backed view, never pooled).
pub struct Target;
impl FrameResource for Target {
    const NAME: &'static str = "target";
    fn kind() -> ResourceKind {
        ResourceKind::ExternalOutput
    }
    fn spec(_: wgpu::TextureFormat) -> TextureSpec {
        TextureSpec {
            format: wgpu::TextureFormat::Rgba8Unorm,
            samples: 1,
            size: SizePolicy::MatchSurface,
        }
    }
}

/// Bloom level at 1/2 of the surface.
pub struct Bloom0;
typed_resource!(Bloom0, "bloom0", owned_fraction, F::Rgba16Float, 2);

/// Bloom level at 1/4 of the surface.
pub struct Bloom1;
typed_resource!(Bloom1, "bloom1", owned_fraction, F::Rgba16Float, 4);

/// Bloom level at 1/8 of the surface.
pub struct Bloom2;
typed_resource!(Bloom2, "bloom2", owned_fraction, F::Rgba16Float, 8);

// ── S2: typed passes (static access sets) ────────────────────────────
// gbuffer/lighting and the four middle bloom passes have configuration-
// independent access sets, so they are pure typed systems. The conditional
// passes (forward, bloom_down0, composite) stay imperative until S2b.

/// G-buffer pass: writes all six G-buffer layers.
pub struct GbufferPass;
impl FramePass for GbufferPass {
    type Reads = ();
    type Writes = (
        Write<Albedo>,
        Write<Normal>,
        Write<MaterialId>,
        Write<WorldPosition>,
        Write<MaterialParams>,
        Write<Depth>,
    );
    fn name(&self) -> &'static str {
        "gbuffer"
    }
    fn run(&mut self, views: SystemViews<'_, Self>, frame: &mut Frame<'_>) {
        let (albedo, normal, material_id, world_position, material_params, depth) = views.writes;
        let g = GbufferTargets {
            albedo,
            normal,
            material_id,
            world_position,
            material_params,
            depth,
        };
        frame
            .renderer
            .render_gbuffer(frame.encoder, &g, frame.mesh, frame.instance_count);
    }
}

/// Lighting pass: resolves the G-buffer into the HDR layer.
pub struct LightingPass;
impl FramePass for LightingPass {
    type Reads = (
        Read<Albedo>,
        Read<Normal>,
        Read<MaterialId>,
        Read<WorldPosition>,
        Read<MaterialParams>,
        Read<Depth>,
    );
    type Writes = (WriteClear<Hdr, ClearBlack>,);
    fn name(&self) -> &'static str {
        "lighting"
    }
    fn run(&mut self, views: SystemViews<'_, Self>, frame: &mut Frame<'_>) {
        let (albedo, normal, material_id, world_position, material_params, depth) = views.reads;
        let (hdr,) = views.writes;
        let g = GbufferTargets {
            albedo,
            normal,
            material_id,
            world_position,
            material_params,
            depth,
        };
        frame
            .renderer
            .render_lighting(frame.device, frame.encoder, &g, hdr);
    }
}

/// Bloom downsample 1/2 → 1/4 (bright-pass already applied at 1/2).
pub struct BloomDown1Pass;
impl FramePass for BloomDown1Pass {
    type Reads = (Read<Bloom0>,);
    type Writes = (WriteClear<Bloom1, ClearBlack>,);
    fn name(&self) -> &'static str {
        "bloom_down1"
    }
    fn run(&mut self, views: SystemViews<'_, Self>, frame: &mut Frame<'_>) {
        let (input,) = views.reads;
        let (output,) = views.writes;
        frame.renderer.render_bloom_down(
            frame.device,
            frame.queue,
            frame.encoder,
            input,
            output,
            0.0,
        );
    }
}

/// Bloom downsample 1/4 → 1/8.
pub struct BloomDown2Pass;
impl FramePass for BloomDown2Pass {
    type Reads = (Read<Bloom1>,);
    type Writes = (WriteClear<Bloom2, ClearBlack>,);
    fn name(&self) -> &'static str {
        "bloom_down2"
    }
    fn run(&mut self, views: SystemViews<'_, Self>, frame: &mut Frame<'_>) {
        let (input,) = views.reads;
        let (output,) = views.writes;
        frame.renderer.render_bloom_down(
            frame.device,
            frame.queue,
            frame.encoder,
            input,
            output,
            0.0,
        );
    }
}

/// Bloom upsample 1/8 → 1/4.
pub struct BloomUp1Pass;
impl FramePass for BloomUp1Pass {
    type Reads = (Read<Bloom2>,);
    type Writes = (Write<Bloom1>,);
    fn name(&self) -> &'static str {
        "bloom_up1"
    }
    fn run(&mut self, views: SystemViews<'_, Self>, frame: &mut Frame<'_>) {
        let (input,) = views.reads;
        let (output,) = views.writes;
        frame
            .renderer
            .render_bloom_up(frame.device, frame.encoder, input, output);
    }
}

/// Bloom upsample 1/4 → 1/2.
pub struct BloomUp0Pass;
impl FramePass for BloomUp0Pass {
    type Reads = (Read<Bloom1>,);
    type Writes = (Write<Bloom0>,);
    fn name(&self) -> &'static str {
        "bloom_up0"
    }
    fn run(&mut self, views: SystemViews<'_, Self>, frame: &mut Frame<'_>) {
        let (input,) = views.reads;
        let (output,) = views.writes;
        frame
            .renderer
            .render_bloom_up(frame.device, frame.encoder, input, output);
    }
}

// ── S2b: conditional passes as mode families ─────────────────────────
// Access sets that depend on the plan configuration are selected at
// registration: the configuration becomes a mode TYPE (a table of
// facts), the body stays one. Design: docs/rendering/unified-scheduler.md.

/// Bright-pass threshold for the first bloom downsample (luminance gate:
/// only pixels brighter than this contribute to the bloom chain).
const BLOOM_BRIGHT_THRESHOLD: f32 = 0.7;

// ── forward: two behavior modes (owns the depth buffer or shares it) ──

/// Depth ownership of the forward pass: forward-only clears the depth
/// itself; hybrid reads the one the gbuffer pass filled.
pub trait ForwardMode: Sized + 'static {
    type Reads: AccessSet + for<'a> ViewsFor<'a>;
    type Writes: AccessSet + for<'a> ViewsFor<'a>;
    const OWNS_DEPTH: bool;
}

/// Forward-only technique: the pass owns (clears) the depth buffer.
pub struct OwnsDepth;
impl ForwardMode for OwnsDepth {
    type Reads = ();
    type Writes = (
        WriteClear<Depth, ClearWhite>,
        WriteClear<HdrFwd, ClearTransparent>,
    );
    const OWNS_DEPTH: bool = true;
}

/// Hybrid technique: the gbuffer pass owns the depth buffer.
pub struct SharedDepth;
impl ForwardMode for SharedDepth {
    type Reads = (Read<Depth>,);
    type Writes = (WriteClear<HdrFwd, ClearTransparent>,);
    const OWNS_DEPTH: bool = false;
}

/// The forward pass; `M` selects the depth-ownership mode.
pub struct Forward<M: ForwardMode>(PhantomData<fn() -> M>);
impl<M: ForwardMode> Forward<M> {
    /// Value constructor: a bare struct path is not a value (E0423).
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<M: ForwardMode> Default for Forward<M> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
impl<M: ForwardMode> FramePass for Forward<M> {
    type Reads = M::Reads;
    type Writes = M::Writes;
    fn name(&self) -> &'static str {
        "forward"
    }
    fn run(&mut self, views: SystemViews<'_, Self>, frame: &mut Frame<'_>) {
        frame.renderer.render_forward(
            frame.encoder,
            views.get::<Depth>(),
            views.get::<HdrFwd>(),
            frame.mesh,
            frame.instance_count,
            M::OWNS_DEPTH,
        );
    }
}

// ── bloom_down0: the bright-pass input follows the technique ─────────

/// Which HDR layer feeds the bright pass (the one this technique made).
pub trait BrightInput: Sized + 'static {
    type Reads: AccessSet + for<'a> ViewsFor<'a>;
    fn input<'a>(views: &SystemViews<'a, BloomBright<Self>>) -> &'a wgpu::TextureView;
}

/// Deferred/hybrid: `hdr`, filled by the lighting pass.
pub struct FromDeferred;
impl BrightInput for FromDeferred {
    type Reads = (Read<Hdr>,);
    fn input<'a>(views: &SystemViews<'a, BloomBright<Self>>) -> &'a wgpu::TextureView {
        views.get::<Hdr>()
    }
}

/// Forward-only: `hdr_fwd`, filled by the forward pass.
pub struct FromForward;
impl BrightInput for FromForward {
    type Reads = (Read<HdrFwd>,);
    fn input<'a>(views: &SystemViews<'a, BloomBright<Self>>) -> &'a wgpu::TextureView {
        views.get::<HdrFwd>()
    }
}

/// The bright-pass downsample (surface → 1/2); `I` selects the input.
pub struct BloomBright<I: BrightInput>(PhantomData<fn() -> I>);
impl<I: BrightInput> BloomBright<I> {
    /// Value constructor: a bare struct path is not a value (E0423).
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<I: BrightInput> Default for BloomBright<I> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
impl<I: BrightInput> FramePass for BloomBright<I> {
    type Reads = I::Reads;
    type Writes = (WriteClear<Bloom0, ClearBlack>,);
    fn name(&self) -> &'static str {
        "bloom_down0"
    }
    fn run(&mut self, views: SystemViews<'_, Self>, frame: &mut Frame<'_>) {
        frame.renderer.render_bloom_down(
            frame.device,
            frame.queue,
            frame.encoder,
            I::input(&views),
            views.get::<Bloom0>(),
            BLOOM_BRIGHT_THRESHOLD,
        );
    }
}

// ── composite: six modes = (technique) × (bloom on/off) ──────────────

/// Which HDR layers exist and whether the bloom chain feeds the mix.
pub trait CompositeMode: Sized + 'static {
    type Reads: AccessSet + for<'a> ViewsFor<'a>;
    const SHADER_MODE: u32;
    const BLOOM: bool;
    /// Binds the shader inputs from this mode's declared views. Dead
    /// layers (the ones this technique does not produce) are bound to a
    /// live view with zero effect — the shader picks by `SHADER_MODE`.
    fn inputs<'a>(views: &SystemViews<'a, Composite<Self>>) -> CompositeInputs<'a>;
}

/// Deferred + bloom.
pub struct CompositeDeferredBloom;
impl CompositeMode for CompositeDeferredBloom {
    type Reads = (Read<Hdr>, Read<Bloom0>);
    const SHADER_MODE: u32 = 0;
    const BLOOM: bool = true;
    fn inputs<'a>(views: &SystemViews<'a, Composite<Self>>) -> CompositeInputs<'a> {
        let hdr = views.get::<Hdr>();
        CompositeInputs {
            target: views.get::<Target>(),
            hdr,
            hdr_fwd: hdr,
            bloom: views.get::<Bloom0>(),
            bloom_intensity: 1.0,
            mode: Self::SHADER_MODE,
        }
    }
}

/// Deferred, bloom culled.
pub struct CompositeDeferred;
impl CompositeMode for CompositeDeferred {
    type Reads = (Read<Hdr>,);
    const SHADER_MODE: u32 = 0;
    const BLOOM: bool = false;
    fn inputs<'a>(views: &SystemViews<'a, Composite<Self>>) -> CompositeInputs<'a> {
        let hdr = views.get::<Hdr>();
        CompositeInputs {
            target: views.get::<Target>(),
            hdr,
            hdr_fwd: hdr,
            bloom: hdr,
            bloom_intensity: 0.0,
            mode: Self::SHADER_MODE,
        }
    }
}

/// Hybrid + bloom: both HDR layers and the bloom chain are live.
pub struct CompositeHybridBloom;
impl CompositeMode for CompositeHybridBloom {
    type Reads = (Read<Hdr>, Read<HdrFwd>, Read<Bloom0>);
    const SHADER_MODE: u32 = 2;
    const BLOOM: bool = true;
    fn inputs<'a>(views: &SystemViews<'a, Composite<Self>>) -> CompositeInputs<'a> {
        CompositeInputs {
            target: views.get::<Target>(),
            hdr: views.get::<Hdr>(),
            hdr_fwd: views.get::<HdrFwd>(),
            bloom: views.get::<Bloom0>(),
            bloom_intensity: 1.0,
            mode: Self::SHADER_MODE,
        }
    }
}

/// Hybrid, bloom culled.
pub struct CompositeHybrid;
impl CompositeMode for CompositeHybrid {
    type Reads = (Read<Hdr>, Read<HdrFwd>);
    const SHADER_MODE: u32 = 2;
    const BLOOM: bool = false;
    fn inputs<'a>(views: &SystemViews<'a, Composite<Self>>) -> CompositeInputs<'a> {
        let hdr_fwd = views.get::<HdrFwd>();
        CompositeInputs {
            target: views.get::<Target>(),
            hdr: views.get::<Hdr>(),
            hdr_fwd,
            bloom: hdr_fwd,
            bloom_intensity: 0.0,
            mode: Self::SHADER_MODE,
        }
    }
}

/// Forward-only + bloom.
pub struct CompositeForwardBloom;
impl CompositeMode for CompositeForwardBloom {
    type Reads = (Read<HdrFwd>, Read<Bloom0>);
    const SHADER_MODE: u32 = 1;
    const BLOOM: bool = true;
    fn inputs<'a>(views: &SystemViews<'a, Composite<Self>>) -> CompositeInputs<'a> {
        let hdr_fwd = views.get::<HdrFwd>();
        CompositeInputs {
            target: views.get::<Target>(),
            hdr: hdr_fwd,
            hdr_fwd,
            bloom: views.get::<Bloom0>(),
            bloom_intensity: 1.0,
            mode: Self::SHADER_MODE,
        }
    }
}

/// Forward-only, bloom culled.
pub struct CompositeForward;
impl CompositeMode for CompositeForward {
    type Reads = (Read<HdrFwd>,);
    const SHADER_MODE: u32 = 1;
    const BLOOM: bool = false;
    fn inputs<'a>(views: &SystemViews<'a, Composite<Self>>) -> CompositeInputs<'a> {
        let hdr_fwd = views.get::<HdrFwd>();
        CompositeInputs {
            target: views.get::<Target>(),
            hdr: hdr_fwd,
            hdr_fwd,
            bloom: hdr_fwd,
            bloom_intensity: 0.0,
            mode: Self::SHADER_MODE,
        }
    }
}

/// The composite pass; `M` is the (technique × bloom) mode.
pub struct Composite<M: CompositeMode>(PhantomData<fn() -> M>);
impl<M: CompositeMode> Composite<M> {
    /// Value constructor: a bare struct path is not a value (E0423).
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<M: CompositeMode> Default for Composite<M> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
impl<M: CompositeMode> FramePass for Composite<M> {
    type Reads = M::Reads;
    type Writes = (Write<Target>,);
    fn name(&self) -> &'static str {
        "composite"
    }
    fn run(&mut self, views: SystemViews<'_, Self>, frame: &mut Frame<'_>) {
        frame.renderer.render_composite(
            frame.device,
            frame.queue,
            frame.encoder,
            M::inputs(&views),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SURFACE: wgpu::TextureFormat = F::Rgba8UnormSrgb;

    fn owned_spec<R: FrameResource>(format: F, size: SizePolicy) {
        assert_eq!(R::kind(), ResourceKind::FrameOwned);
        assert_eq!(
            R::spec(SURFACE),
            TextureSpec {
                format,
                samples: 1,
                size,
            }
        );
    }

    /// Specs mirror the imperative wiring (frame_exec parity test); the
    /// dump names are part of the contract.
    #[test]
    fn resource_names_and_specs() {
        assert_eq!(Albedo::NAME, "albedo");
        owned_spec::<Albedo>(F::Rgba8Unorm, SizePolicy::MatchSurface);
        assert_eq!(Normal::NAME, "normal");
        owned_spec::<Normal>(F::Rg16Float, SizePolicy::MatchSurface);
        assert_eq!(MaterialId::NAME, "material_id");
        owned_spec::<MaterialId>(F::R32Uint, SizePolicy::MatchSurface);
        assert_eq!(WorldPosition::NAME, "world_position");
        owned_spec::<WorldPosition>(F::Rg16Float, SizePolicy::MatchSurface);
        assert_eq!(MaterialParams::NAME, "material_params");
        owned_spec::<MaterialParams>(F::Rgba16Float, SizePolicy::MatchSurface);
        assert_eq!(Depth::NAME, "depth");
        owned_spec::<Depth>(F::Depth32Float, SizePolicy::MatchSurface);
        assert_eq!(HdrFwd::NAME, "hdr_fwd");
        owned_spec::<HdrFwd>(F::Rgba16Float, SizePolicy::MatchSurface);
        assert_eq!(Bloom0::NAME, "bloom0");
        owned_spec::<Bloom0>(F::Rgba16Float, SizePolicy::Fraction(2));
        assert_eq!(Bloom1::NAME, "bloom1");
        owned_spec::<Bloom1>(F::Rgba16Float, SizePolicy::Fraction(4));
        assert_eq!(Bloom2::NAME, "bloom2");
        owned_spec::<Bloom2>(F::Rgba16Float, SizePolicy::Fraction(8));
    }

    #[test]
    fn hdr_mirrors_the_surface_format() {
        assert_eq!(Hdr::NAME, "hdr");
        assert_eq!(Hdr::kind(), ResourceKind::FrameOwned);
        assert_eq!(Hdr::spec(F::Rgba8UnormSrgb).format, F::Rgba8UnormSrgb);
        assert_eq!(Hdr::spec(F::Bgra8UnormSrgb).format, F::Bgra8UnormSrgb);
        assert_eq!(Hdr::spec(SURFACE).size, SizePolicy::MatchSurface);
    }

    #[test]
    fn target_is_external_output() {
        assert_eq!(Target::NAME, "target");
        assert_eq!(Target::kind(), ResourceKind::ExternalOutput);
        assert_eq!(Target::spec(SURFACE).format, F::Rgba8Unorm);
        assert_eq!(Target::spec(SURFACE).size, SizePolicy::MatchSurface);
    }

    #[test]
    fn static_pass_names() {
        assert_eq!(GbufferPass.name(), "gbuffer");
        assert_eq!(LightingPass.name(), "lighting");
        assert_eq!(BloomDown1Pass.name(), "bloom_down1");
        assert_eq!(BloomDown2Pass.name(), "bloom_down2");
        assert_eq!(BloomUp1Pass.name(), "bloom_up1");
        assert_eq!(BloomUp0Pass.name(), "bloom_up0");
    }

    fn reads_of<P: FramePass>() -> Vec<&'static str> {
        let mut v = Vec::new();
        P::Reads::collect_accesses(&mut v);
        assert!(v.iter().all(|a| !a.write && a.clear.is_none()));
        v.iter().map(|a| a.name).collect()
    }

    fn writes_of<P: FramePass>() -> Vec<(&'static str, Option<wgpu::Color>)> {
        let mut v = Vec::new();
        P::Writes::collect_accesses(&mut v);
        assert!(v.iter().all(|a| a.write));
        v.iter().map(|a| (a.name, a.clear)).collect()
    }

    #[test]
    fn gbuffer_writes_all_six_layers() {
        assert!(reads_of::<GbufferPass>().is_empty());
        assert_eq!(
            writes_of::<GbufferPass>(),
            vec![
                ("albedo", None),
                ("normal", None),
                ("material_id", None),
                ("world_position", None),
                ("material_params", None),
                ("depth", None),
            ]
        );
    }

    #[test]
    fn lighting_reads_gbuffer_and_clears_hdr_black() {
        assert_eq!(
            reads_of::<LightingPass>(),
            vec![
                "albedo",
                "normal",
                "material_id",
                "world_position",
                "material_params",
                "depth"
            ]
        );
        assert_eq!(
            writes_of::<LightingPass>(),
            vec![("hdr", Some(wgpu::Color::BLACK))]
        );
    }

    #[test]
    fn bloom_chain_wiring() {
        assert_eq!(reads_of::<BloomDown1Pass>(), vec!["bloom0"]);
        assert_eq!(
            writes_of::<BloomDown1Pass>(),
            vec![("bloom1", Some(wgpu::Color::BLACK))]
        );
        assert_eq!(reads_of::<BloomDown2Pass>(), vec!["bloom1"]);
        assert_eq!(
            writes_of::<BloomDown2Pass>(),
            vec![("bloom2", Some(wgpu::Color::BLACK))]
        );
        assert_eq!(reads_of::<BloomUp1Pass>(), vec!["bloom2"]);
        assert_eq!(writes_of::<BloomUp1Pass>(), vec![("bloom1", None)]);
        assert_eq!(reads_of::<BloomUp0Pass>(), vec!["bloom1"]);
        assert_eq!(writes_of::<BloomUp0Pass>(), vec![("bloom0", None)]);
        assert_eq!(BLOOM_BRIGHT_THRESHOLD, 0.7);
    }

    #[test]
    fn forward_modes_select_depth_ownership() {
        const { assert!(OwnsDepth::OWNS_DEPTH) };
        const { assert!(!SharedDepth::OWNS_DEPTH) };

        assert_eq!(Forward::<OwnsDepth>::new().name(), "forward");
        assert_eq!(Forward::<SharedDepth>::default().name(), "forward");

        // Forward-only: owns and clears the depth (white = far plane).
        assert!(reads_of::<Forward<OwnsDepth>>().is_empty());
        assert_eq!(
            writes_of::<Forward<OwnsDepth>>(),
            vec![
                ("depth", Some(wgpu::Color::WHITE)),
                ("hdr_fwd", Some(wgpu::Color::TRANSPARENT)),
            ]
        );
        // Hybrid: depth comes from the gbuffer pass.
        assert_eq!(reads_of::<Forward<SharedDepth>>(), vec!["depth"]);
        assert_eq!(
            writes_of::<Forward<SharedDepth>>(),
            vec![("hdr_fwd", Some(wgpu::Color::TRANSPARENT))]
        );
    }

    #[test]
    fn bright_pass_input_follows_technique() {
        assert_eq!(BloomBright::<FromDeferred>::new().name(), "bloom_down0");
        assert_eq!(BloomBright::<FromForward>::default().name(), "bloom_down0");
        assert_eq!(reads_of::<BloomBright<FromDeferred>>(), vec!["hdr"]);
        assert_eq!(reads_of::<BloomBright<FromForward>>(), vec!["hdr_fwd"]);
        assert_eq!(
            writes_of::<BloomBright<FromDeferred>>(),
            vec![("bloom0", Some(wgpu::Color::BLACK))]
        );
    }

    #[test]
    fn composite_modes_encode_technique_and_bloom() {
        assert_eq!(
            Composite::<CompositeDeferredBloom>::new().name(),
            "composite"
        );
        assert_eq!(Composite::<CompositeForward>::default().name(), "composite");

        // (mode constant, bloom flag, expected reads)
        let cases: [(u32, bool, &[&str]); 6] = [
            (
                CompositeDeferredBloom::SHADER_MODE,
                CompositeDeferredBloom::BLOOM,
                &["hdr", "bloom0"],
            ),
            (
                CompositeDeferred::SHADER_MODE,
                CompositeDeferred::BLOOM,
                &["hdr"],
            ),
            (
                CompositeHybridBloom::SHADER_MODE,
                CompositeHybridBloom::BLOOM,
                &["hdr", "hdr_fwd", "bloom0"],
            ),
            (
                CompositeHybrid::SHADER_MODE,
                CompositeHybrid::BLOOM,
                &["hdr", "hdr_fwd"],
            ),
            (
                CompositeForwardBloom::SHADER_MODE,
                CompositeForwardBloom::BLOOM,
                &["hdr_fwd", "bloom0"],
            ),
            (
                CompositeForward::SHADER_MODE,
                CompositeForward::BLOOM,
                &["hdr_fwd"],
            ),
        ];
        let expected_modes = [0, 0, 2, 2, 1, 1];
        let expected_bloom = [true, false, true, false, true, false];
        for (i, (mode, bloom, _)) in cases.iter().enumerate() {
            assert_eq!(*mode, expected_modes[i], "case {i}");
            assert_eq!(*bloom, expected_bloom[i], "case {i}");
        }

        assert_eq!(
            reads_of::<Composite<CompositeDeferredBloom>>(),
            vec!["hdr", "bloom0"]
        );
        assert_eq!(reads_of::<Composite<CompositeDeferred>>(), vec!["hdr"]);
        assert_eq!(
            reads_of::<Composite<CompositeHybridBloom>>(),
            vec!["hdr", "hdr_fwd", "bloom0"]
        );
        assert_eq!(
            reads_of::<Composite<CompositeHybrid>>(),
            vec!["hdr", "hdr_fwd"]
        );
        assert_eq!(
            reads_of::<Composite<CompositeForwardBloom>>(),
            vec!["hdr_fwd", "bloom0"]
        );
        assert_eq!(reads_of::<Composite<CompositeForward>>(), vec!["hdr_fwd"]);

        // Every composite mode writes the swapchain target without a clear.
        assert_eq!(
            writes_of::<Composite<CompositeHybridBloom>>(),
            vec![("target", None)]
        );
    }
}
