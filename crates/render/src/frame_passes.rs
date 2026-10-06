//! Typed resources and static passes for the 3D frame plan — S2a
//! (IDEAS §28.1, PLAN Appendix C).
//!
//! Resource identity is a type ([`FrameResource`]); passes with static
//! access sets are [`FramePass`] implementations whose wiring is derived
//! from `Reads`/`Writes`. Specs and names mirror the imperative wiring
//! exactly, so layout dumps (and the parity test in `frame_exec.rs`)
//! stay identical. The conditional passes (forward, bloom_down0,
//! composite) remain imperative until S2b.

use crate::renderer::{CompositeInputs, GbufferTargets, Renderer3D};
use crate::system::{
    AccessSet, ClearBlack, ClearTransparent, ClearWhite, Frame, FramePass, FrameResource, Read,
    ResourceKind, SystemViews, ViewsFor, Write, WriteClear,
};
use crate::transient_pool::{SizePolicy, TextureSpec};
use std::marker::PhantomData;
/// Default fog tint (linear RGB).
const DEFAULT_FOG_RGB: [f32; 3] = [0.5, 0.6, 0.7];
/// Default fog density (1/m).
const DEFAULT_FOG_DENSITY: f32 = 0.02;

// Short alias keeps `typed_resource!` invocations under rustfmt's
// fn_call_width (60) so they stay on one line.
use wgpu::TextureFormat as F;

// ── S2: typed resources (IDEAS §28.1, PLAN Appendix C) ─────────────
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
    ($t:ident, $name:literal, owned_msaa, $format:expr) => {
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
            fn multisampled() -> bool {
                true
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

/// G-buffer albedo layer (multisampled geometry target at 4x; downstream
/// passes sample the renderer's single-sample resolve).
pub struct Albedo;
typed_resource!(Albedo, "albedo", owned_msaa, F::Rgba8Unorm);

/// G-buffer world-space normal layer (multisampled geometry target at 4x).
pub struct Normal;
typed_resource!(Normal, "normal", owned_msaa, F::Rg16Float);

/// G-buffer material id layer (`R16Uint`: the spec-guaranteed
/// multisampleable integer format — `R32Uint` has no multisample flag, so
/// the legacy MSAA g-buffer could never create it; ids past `u16::MAX`
/// truncate, see [`crate::renderer::MSAA_SAMPLE_COUNT`]).
///
/// Stays multisampled at 4x with no resolve target (loaded as sample 0).
pub struct MaterialId;
typed_resource!(MaterialId, "material_id", owned_msaa, F::R16Uint);

/// G-buffer world-space position layer (multisampled geometry target at 4x).
pub struct WorldPosition;
typed_resource!(WorldPosition, "world_position", owned_msaa, F::Rg16Float);

/// G-buffer material params layer (multisampled geometry target at 4x).
pub struct MaterialParams;
typed_resource!(
    MaterialParams,
    "material_params",
    owned_msaa,
    F::Rgba16Float
);

/// Depth buffer (multisampled at 4x with no resolve target; shared with
/// the forward pass in hybrid mode).
pub struct Depth;
typed_resource!(Depth, "depth", owned_msaa, F::Depth32Float);

/// HDR layer of the deferred path.
///
/// Scene-linear [`crate::renderer::DEFERRED_HDR_FORMAT`], independent of
/// the surface: the lighting pass writes pre-tonemap color and the
/// composite applies ACES once. Matching the surface (8-bit sRGB) clamped
/// highlights and quantized the walking mannequin.
pub struct Hdr;
impl FrameResource for Hdr {
    const NAME: &'static str = "hdr";
    fn kind() -> ResourceKind {
        ResourceKind::FrameOwned
    }
    fn spec(_surface_format: wgpu::TextureFormat) -> TextureSpec {
        TextureSpec {
            format: crate::renderer::DEFERRED_HDR_FORMAT,
            samples: 1,
            size: SizePolicy::MatchSurface,
        }
    }
}

/// HDR layer of the forward path (the forward color target: multisampled
/// at 4x, resolving into the renderer's single-sample view which the
/// composite pass and the bloom bright-pass sample there).
pub struct HdrFwd;
typed_resource!(HdrFwd, "hdr_fwd", owned_msaa, F::Rgba16Float);

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
    fn run<'a>(&mut self, views: SystemViews<'a, Self>, frame: &mut Frame<'a>) {
        let (
            Some(albedo),
            Some(normal),
            Some(material_id),
            Some(world_position),
            Some(material_params),
            Some(depth),
        ) = views.writes
        else {
            return;
        };
        let g = GbufferTargets {
            albedo,
            normal,
            material_id,
            world_position,
            material_params,
            depth,
        };
        frame.renderer.render_gbuffer_with_custom(
            frame.device,
            frame.encoder,
            &g,
            frame.mesh,
            frame.instance_count,
            frame.customs,
        );
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
    fn run<'a>(&mut self, views: SystemViews<'a, Self>, frame: &mut Frame<'a>) {
        let (
            Some(albedo),
            Some(normal),
            Some(material_id),
            Some(world_position),
            Some(material_params),
            Some(depth),
        ) = views.reads
        else {
            return;
        };
        let (Some(hdr),) = views.writes else {
            return;
        };
        let g = GbufferTargets {
            albedo,
            normal,
            material_id,
            world_position,
            material_params,
            depth,
        };
        frame.renderer.render_shadows_with_custom(
            frame.device,
            frame.encoder,
            frame.mesh,
            frame.instance_count,
            frame.customs,
        );
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
    fn run<'a>(&mut self, views: SystemViews<'a, Self>, frame: &mut Frame<'a>) {
        let (Some(input),) = views.reads else {
            return;
        };
        let (Some(output),) = views.writes else {
            return;
        };
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
    fn run<'a>(&mut self, views: SystemViews<'a, Self>, frame: &mut Frame<'a>) {
        let (Some(input),) = views.reads else {
            return;
        };
        let (Some(output),) = views.writes else {
            return;
        };
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
    fn run<'a>(&mut self, views: SystemViews<'a, Self>, frame: &mut Frame<'a>) {
        let (Some(input),) = views.reads else {
            return;
        };
        let (Some(output),) = views.writes else {
            return;
        };
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
    fn run<'a>(&mut self, views: SystemViews<'a, Self>, frame: &mut Frame<'a>) {
        let (Some(input),) = views.reads else {
            return;
        };
        let (Some(output),) = views.writes else {
            return;
        };
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
    /// Accesses of the mode's forward pass.
    type Reads: AccessSet + for<'a> ViewsFor<'a>;
    /// Writes of the mode's forward pass.
    type Writes: AccessSet + for<'a> ViewsFor<'a>;
    /// Who owns (clears) the depth buffer in this technique.
    const DEPTH: crate::flags::DepthOwnership;
    /// Whether this pass renders the shadow pre-pass itself: only
    /// forward-only (no `LightingPass` runs its own).
    const SHADOWS: crate::flags::ShadowCast;
}

/// Forward-only technique: the pass owns (clears) the depth buffer.
pub struct OwnsDepth;
impl ForwardMode for OwnsDepth {
    type Reads = ();
    type Writes = (
        WriteClear<Depth, ClearWhite>,
        WriteClear<HdrFwd, ClearTransparent>,
    );
    const DEPTH: crate::flags::DepthOwnership = crate::flags::DepthOwnership::Owned;
    const SHADOWS: crate::flags::ShadowCast = crate::flags::ShadowCast::Enabled;
}

/// Hybrid technique: the gbuffer pass owns the depth buffer.
pub struct SharedDepth;
impl ForwardMode for SharedDepth {
    type Reads = (Read<Depth>,);
    type Writes = (WriteClear<HdrFwd, ClearTransparent>,);
    const DEPTH: crate::flags::DepthOwnership = crate::flags::DepthOwnership::Shared;
    const SHADOWS: crate::flags::ShadowCast = crate::flags::ShadowCast::Disabled;
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
    fn run<'a>(&mut self, views: SystemViews<'a, Self>, frame: &mut Frame<'a>) {
        // Forward-only owns its shadow pre-pass too: no LightingPass
        // runs in that technique, so without this the shadow maps stay
        // at texture-init zero and every shadowed light goes fully dark.
        if M::SHADOWS.is_enabled() {
            frame.renderer.render_shadows_with_custom(
                frame.device,
                frame.encoder,
                frame.mesh,
                frame.instance_count,
                frame.customs,
            );
        }
        let (Some(depth), Some(hdr_fwd)) = (views.get::<Depth>(), views.get::<HdrFwd>()) else {
            return;
        };
        frame.renderer.render_forward_with_custom(
            frame.encoder,
            depth,
            hdr_fwd,
            frame.mesh,
            frame.instance_count,
            M::DEPTH.clears_depth(),
            frame.customs,
        );
    }
}

// ── bloom_down0: the bright-pass input follows the technique ─────────

/// Which HDR layer feeds the bright pass (the one this technique made).
pub trait BrightInput: Sized + 'static {
    /// The layer read by the bright pass in this mode.
    type Reads: AccessSet + for<'a> ViewsFor<'a>;
    /// Borrows the HDR view this technique's bright pass reads: the pooled
    /// view at 1x, the renderer's single-sample forward resolve at 4x when
    /// this mode reads the multisampled forward layer.
    fn input<'a>(
        views: &SystemViews<'a, BloomBright<Self>>,
        renderer: &'a Renderer3D,
    ) -> Option<&'a wgpu::TextureView>;
}

/// Deferred/hybrid: `hdr` (single-sample lighting output in all modes).
pub struct FromDeferred;
impl BrightInput for FromDeferred {
    type Reads = (Read<Hdr>,);
    fn input<'a>(
        views: &SystemViews<'a, BloomBright<Self>>,
        _renderer: &'a Renderer3D,
    ) -> Option<&'a wgpu::TextureView> {
        views.get::<Hdr>()
    }
}

/// Forward-only: `hdr_fwd` (multisampled at 4x — the resolve view then).
pub struct FromForward;
impl BrightInput for FromForward {
    type Reads = (Read<HdrFwd>,);
    fn input<'a>(
        views: &SystemViews<'a, BloomBright<Self>>,
        renderer: &'a Renderer3D,
    ) -> Option<&'a wgpu::TextureView> {
        renderer
            .forward_resolve_view()
            .or_else(|| views.get::<HdrFwd>())
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
    fn run<'a>(&mut self, views: SystemViews<'a, Self>, frame: &mut Frame<'a>) {
        let (Some(input), Some(output)) = (I::input(&views, frame.renderer), views.get::<Bloom0>())
        else {
            return;
        };
        frame.renderer.render_bloom_down(
            frame.device,
            frame.queue,
            frame.encoder,
            input,
            output,
            BLOOM_BRIGHT_THRESHOLD,
        );
    }
}

// ── composite: six modes = (technique) × (bloom on/off) ──────────────

/// Which HDR layers exist and whether the bloom chain feeds the mix.
pub trait CompositeMode: Sized + 'static {
    /// This mode's live layers (dead ones bind a zeroed view instead).
    type Reads: AccessSet + for<'a> ViewsFor<'a>;
    /// Which HDR layers exist (value of the shader's layer-mix selector).
    const TECHNIQUE: crate::flags::CompositeTechnique;
    /// Whether the bloom chain contributes to the mix.
    const BLOOM: crate::flags::Bloom;
    /// Binds the shader inputs from this mode's declared views. Dead
    /// layers (the ones this technique does not produce) are bound to a
    /// live view with zero effect — the shader picks by `TECHNIQUE`.
    /// Modes reading the forward layer sample the renderer's
    /// single-sample resolve at 4x (the pooled forward layer is
    /// multisampled there); at 1x every view is the pooled one.
    fn inputs<'a>(
        views: &SystemViews<'a, Composite<Self>>,
        renderer: &'a Renderer3D,
    ) -> Option<CompositeInputs<'a>>;
}

/// Deferred + bloom.
pub struct CompositeDeferredBloom;
impl CompositeMode for CompositeDeferredBloom {
    type Reads = (Read<Hdr>, Read<Bloom0>);
    const TECHNIQUE: crate::flags::CompositeTechnique = crate::flags::CompositeTechnique::Deferred;
    const BLOOM: crate::flags::Bloom = crate::flags::Bloom::On;
    fn inputs<'a>(
        views: &SystemViews<'a, Composite<Self>>,
        _renderer: &'a Renderer3D,
    ) -> Option<CompositeInputs<'a>> {
        let hdr = views.get::<Hdr>()?;
        Some(CompositeInputs {
            target: views.get::<Target>()?,
            hdr,
            hdr_fwd: hdr,
            bloom: views.get::<Bloom0>()?,
            bloom_intensity: Self::BLOOM.intensity(),
            mode: Self::TECHNIQUE.shader_mode(),
        })
    }
}

/// Deferred, bloom culled.
pub struct CompositeDeferred;
impl CompositeMode for CompositeDeferred {
    type Reads = (Read<Hdr>,);
    const TECHNIQUE: crate::flags::CompositeTechnique = crate::flags::CompositeTechnique::Deferred;
    const BLOOM: crate::flags::Bloom = crate::flags::Bloom::Off;
    fn inputs<'a>(
        views: &SystemViews<'a, Composite<Self>>,
        _renderer: &'a Renderer3D,
    ) -> Option<CompositeInputs<'a>> {
        let hdr = views.get::<Hdr>()?;
        Some(CompositeInputs {
            target: views.get::<Target>()?,
            hdr,
            hdr_fwd: hdr,
            bloom: hdr,
            bloom_intensity: Self::BLOOM.intensity(),
            mode: Self::TECHNIQUE.shader_mode(),
        })
    }
}

/// Hybrid + bloom: both HDR layers and the bloom chain are live.
pub struct CompositeHybridBloom;
impl CompositeMode for CompositeHybridBloom {
    type Reads = (Read<Hdr>, Read<HdrFwd>, Read<Bloom0>);
    const TECHNIQUE: crate::flags::CompositeTechnique = crate::flags::CompositeTechnique::Hybrid;
    const BLOOM: crate::flags::Bloom = crate::flags::Bloom::On;
    fn inputs<'a>(
        views: &SystemViews<'a, Composite<Self>>,
        renderer: &'a Renderer3D,
    ) -> Option<CompositeInputs<'a>> {
        Some(CompositeInputs {
            target: views.get::<Target>()?,
            hdr: views.get::<Hdr>()?,
            hdr_fwd: renderer
                .forward_resolve_view()
                .or_else(|| views.get::<HdrFwd>())?,
            bloom: views.get::<Bloom0>()?,
            bloom_intensity: Self::BLOOM.intensity(),
            mode: Self::TECHNIQUE.shader_mode(),
        })
    }
}

/// Hybrid, bloom culled.
pub struct CompositeHybrid;
impl CompositeMode for CompositeHybrid {
    type Reads = (Read<Hdr>, Read<HdrFwd>);
    const TECHNIQUE: crate::flags::CompositeTechnique = crate::flags::CompositeTechnique::Hybrid;
    const BLOOM: crate::flags::Bloom = crate::flags::Bloom::Off;
    fn inputs<'a>(
        views: &SystemViews<'a, Composite<Self>>,
        renderer: &'a Renderer3D,
    ) -> Option<CompositeInputs<'a>> {
        let hdr_fwd = renderer
            .forward_resolve_view()
            .or_else(|| views.get::<HdrFwd>())?;
        Some(CompositeInputs {
            target: views.get::<Target>()?,
            hdr: views.get::<Hdr>()?,
            hdr_fwd,
            bloom: hdr_fwd,
            bloom_intensity: Self::BLOOM.intensity(),
            mode: Self::TECHNIQUE.shader_mode(),
        })
    }
}

/// Forward-only + bloom.
pub struct CompositeForwardBloom;
impl CompositeMode for CompositeForwardBloom {
    type Reads = (Read<HdrFwd>, Read<Bloom0>);
    const TECHNIQUE: crate::flags::CompositeTechnique = crate::flags::CompositeTechnique::Forward;
    const BLOOM: crate::flags::Bloom = crate::flags::Bloom::On;
    fn inputs<'a>(
        views: &SystemViews<'a, Composite<Self>>,
        renderer: &'a Renderer3D,
    ) -> Option<CompositeInputs<'a>> {
        let hdr_fwd = renderer
            .forward_resolve_view()
            .or_else(|| views.get::<HdrFwd>())?;
        Some(CompositeInputs {
            target: views.get::<Target>()?,
            hdr: hdr_fwd,
            hdr_fwd,
            bloom: views.get::<Bloom0>()?,
            bloom_intensity: Self::BLOOM.intensity(),
            mode: Self::TECHNIQUE.shader_mode(),
        })
    }
}

/// Forward-only, bloom culled.
pub struct CompositeForward;
impl CompositeMode for CompositeForward {
    type Reads = (Read<HdrFwd>,);
    const TECHNIQUE: crate::flags::CompositeTechnique = crate::flags::CompositeTechnique::Forward;
    const BLOOM: crate::flags::Bloom = crate::flags::Bloom::Off;
    fn inputs<'a>(
        views: &SystemViews<'a, Composite<Self>>,
        renderer: &'a Renderer3D,
    ) -> Option<CompositeInputs<'a>> {
        let hdr_fwd = renderer
            .forward_resolve_view()
            .or_else(|| views.get::<HdrFwd>())?;
        Some(CompositeInputs {
            target: views.get::<Target>()?,
            hdr: hdr_fwd,
            hdr_fwd,
            bloom: hdr_fwd,
            bloom_intensity: Self::BLOOM.intensity(),
            mode: Self::TECHNIQUE.shader_mode(),
        })
    }
}

/// Positive fog density in 1/m (strictly `> 0` and finite).
///
/// Newtype over [`PositiveF32`](ornis_core::units::PositiveF32): a disabled
/// fog carries no density at all ([`FogState::Disabled`]), so any live
/// density is positive by construction — no `density == 0.0` checks at use
/// sites.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogDensity(ornis_core::units::PositiveF32);

impl FogDensity {
    /// Checked constructor: `Some` only for finite values `> 0`.
    pub fn try_new(value: f32) -> Option<Self> {
        ornis_core::units::PositiveF32::try_new(value).map(Self)
    }

    /// Constant-payload constructor for literals validated by inspection.
    /// Non-finite or non-positive input falls back to
    /// [`f32::MIN_POSITIVE`] instead of panicking.
    pub const fn expect_valid(value: f32) -> Self {
        Self(ornis_core::units::PositiveF32::expect_valid(value))
    }

    /// Raw density in 1/m.
    pub const fn get(self) -> f32 {
        self.0.get()
    }
}

impl From<FogDensity> for f32 {
    /// Raw density.
    fn from(density: FogDensity) -> Self {
        density.get()
    }
}

/// Distance-fog settings for the enabled state.
///
/// `color` is the linear-space fog color ([`LinearRgb`](ornis_core::units::LinearRgb));
/// `density` is the exponential falloff rate. There is no zero-density
/// instance — disabled fog is [`FogState::Disabled`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogSettings {
    /// Fog color mixed toward with depth.
    pub color: ornis_core::units::LinearRgb,
    /// Fog density (always positive).
    pub density: FogDensity,
}

impl FogSettings {
    /// Builds settings from typed color + density.
    pub const fn new(color: ornis_core::units::LinearRgb, density: FogDensity) -> Self {
        Self { color, density }
    }

    /// Fallible raw constructor: `None` when `density` is not finite and
    /// `> 0`.
    pub fn try_from_raw(color: [f32; 3], density: f32) -> Option<Self> {
        Some(Self {
            color: ornis_core::units::LinearRgb::new(color),
            density: FogDensity::try_new(density)?,
        })
    }
}

impl Default for FogSettings {
    fn default() -> Self {
        Self {
            color: ornis_core::units::LinearRgb::new(DEFAULT_FOG_RGB),
            density: FogDensity::expect_valid(DEFAULT_FOG_DENSITY),
        }
    }
}

/// Distance-fog state: disabled (exact no-op) or enabled with settings.
///
/// Replaces `density == 0.0` bool checks: the invariant lives in the type.
/// Default is [`FogState::Disabled`], so merely registering
/// [`FogPass::default()`] leaves the frame pixel-identical.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FogState {
    /// Fog off: [`apply_fog`] is the exact identity and [`FogPass::run`]
    /// records no commands.
    #[default]
    Disabled,
    /// Fog on with the given settings.
    Enabled(FogSettings),
}

impl FogState {
    /// `true` for [`FogState::Disabled`].
    pub fn is_disabled(self) -> bool {
        matches!(self, Self::Disabled)
    }

    /// `true` for [`FogState::Enabled`].
    pub fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled(_))
    }

    /// Settings of the enabled state, `None` when disabled.
    pub fn settings(self) -> Option<FogSettings> {
        match self {
            Self::Enabled(settings) => Some(settings),
            Self::Disabled => None,
        }
    }
}

impl From<FogSettings> for FogState {
    /// Wraps enabled settings.
    fn from(settings: FogSettings) -> Self {
        Self::Enabled(settings)
    }
}

/// Exponential fog factor for `depth` (view-space distance, `>= 0`) and
/// `density`: `1 - exp(-density * depth)` in `[0, 1)`.
///
/// Non-finite or negative depths map to zero (no fog); the result is a
/// [`Clamped01`](ornis_core::units::Clamped01) by construction, so the mix
/// can never overshoot the fog color.
pub fn fog_factor(depth: f32, density: FogDensity) -> ornis_core::units::Clamped01 {
    if !depth.is_finite() {
        return ornis_core::units::Clamped01::ZERO;
    }
    let depth = depth.max(0.0);
    ornis_core::units::Clamped01::new(1.0 - (-density.get() * depth).exp())
}

/// Pure distance-fog mix: `color + (fog.color - color) * factor`.
///
/// `factor` comes from [`fog_factor`]; [`FogState::Disabled`] returns
/// `color` unchanged — the exact identity, so the disabled pass cannot
/// drift a pixel. `depth` is view-space distance (`>= 0`): on the GPU it is
/// the Euclidean distance from the eye to the world position reconstructed
/// from the g-buffer `Depth` buffer (see [`crate::shaders::fog_generated`]
/// — hardware depth is non-linear, the `world_position` layer only stores
/// xy, so depth-texture reconstruction is authoritative).
pub fn apply_fog(color: [f32; 3], depth: f32, fog: FogState) -> [f32; 3] {
    let FogState::Enabled(settings) = fog else {
        return color;
    };
    let factor = fog_factor(depth, settings.density).get();
    let fog_color = settings.color.as_array();
    [
        color[0] + (fog_color[0] - color[0]) * factor,
        color[1] + (fog_color[1] - color[1]) * factor,
        color[2] + (fog_color[2] - color[2]) * factor,
    ]
}

/// Where an opt-in [`FogPass`] node sits relative to the `composite`
/// pass: the owner's wiring decision as a type, not a bool flag.
///
/// The composite pass clears `target` and draws one fullscreen mix over
/// it, while an enabled [`FogPass`] loads `target` and draws fogged `hdr`
/// over that — so placement decides whose output is presented:
/// [`FogPlacement::AfterComposite`] (recommended) presents fogged `hdr`;
/// [`FogPlacement::BeforeComposite`] has its `target` write discarded by
/// the composite clear (node present in the plan, no visible effect).
/// Either way fog needs the deferred HDR layer: on
/// [`Technique::Forward`](crate::frame_exec::Technique::Forward) plans
/// `hdr` is never written and compiling the layout fails
/// (read-before-write). Default plans register no fog at all — see
/// [`FogWiring`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FogPlacement {
    /// Fog node before `composite`: its `target` write is discarded by
    /// the composite clear. Only plan shape changes (pass order); the
    /// presented frame is identical to no fog.
    BeforeComposite,
    /// Fog node after `composite` (recommended): the fogged-`hdr` draw
    /// is the presented frame. Forward-layer and bloom contributions
    /// already mixed into `target` are replaced by fogged `hdr` — the
    /// owner's call on hybrid/bloomed plans.
    AfterComposite,
}

/// Opt-in fog wiring for [`RenderFrame3D`](crate::frame_exec::RenderFrame3D):
/// placement plus state in one value. Dropping it registers nothing —
/// default plans stay fog-free — so the constructor is `#[must_use]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogWiring {
    /// Where the fog node sits relative to `composite`.
    pub placement: FogPlacement,
    /// Fog state carried by the node (`Disabled` records no commands:
    /// pixel-identical to no fog).
    pub state: FogState,
}

impl FogWiring {
    /// Wiring value from placement plus state.
    #[must_use]
    pub fn new(placement: FogPlacement, state: FogState) -> Self {
        Self { placement, state }
    }
}

/// Optional distance-fog pass over the deferred HDR layer.
///
/// Never registered by [`crate::frame_exec::RenderFrame3D`] by default —
/// opt in with [`FogWiring`] (placement + state as one typed decision;
/// see [`RenderFrame3D::new_with_fog`](crate::frame_exec::RenderFrame3D::new_with_fog)).
/// With [`FogState::Disabled`] (default) [`run`](FramePass::run) records
/// no commands (pixel-identical no-op, same as not registered); with
/// [`FogState::Enabled`] it runs the GPU mix from
/// [`crate::shaders::fog_generated`] (same math as [`apply_fog`]).
///
/// Depth source: the g-buffer `Depth` buffer (hardware `Depth32Float`,
/// linearized on the GPU — see [`crate::shaders::fog_generated`]).
/// Placement recommendation: [`FogPlacement::AfterComposite`] on deferred
/// plans without bloom (the composite output there derives solely from
/// `hdr`, so fog sees the same layer it mixes).
pub struct FogPass {
    state: FogState,
}

impl FogPass {
    /// Value constructor from a [`FogState`].
    pub fn new(state: FogState) -> Self {
        Self { state }
    }

    /// Value constructor from enabled [`FogSettings`].
    pub fn with_settings(settings: FogSettings) -> Self {
        Self::new(FogState::Enabled(settings))
    }

    /// Current fog state.
    pub fn state(self) -> FogState {
        self.state
    }

    /// `true` when fog is disabled: [`run`](FramePass::run) is a no-op and
    /// the frame is unchanged.
    pub fn is_disabled(&self) -> bool {
        self.state.is_disabled()
    }
}

impl Default for FogPass {
    fn default() -> Self {
        Self::new(FogState::Disabled)
    }
}

impl FramePass for FogPass {
    type Reads = (Read<Hdr>, Read<Depth>);
    type Writes = (Write<Target>,);
    fn name(&self) -> &'static str {
        "fog"
    }
    fn run<'a>(&mut self, views: SystemViews<'a, Self>, frame: &mut Frame<'a>) {
        let Some(settings) = self.state.settings() else {
            // Disabled: keep the declared wiring honest in debug builds
            // without recording any commands (exact no-op).
            let _ = views.get::<Hdr>();
            let _ = views.get::<Depth>();
            let _ = views.get::<Target>();
            return;
        };
        let (Some(hdr), Some(depth), Some(target)) = (
            views.get::<Hdr>(),
            views.get::<Depth>(),
            views.get::<Target>(),
        ) else {
            return;
        };
        frame.renderer.render_fog(
            frame.device,
            frame.queue,
            frame.encoder,
            crate::renderer::FogInputs {
                hdr,
                depth,
                target,
                color: settings.color.as_array(),
                density: settings.density.get(),
            },
        );
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
    fn run<'a>(&mut self, views: SystemViews<'a, Self>, frame: &mut Frame<'a>) {
        let Some(inputs) = M::inputs(&views, frame.renderer) else {
            return;
        };
        frame
            .renderer
            .render_composite(frame.device, frame.queue, frame.encoder, inputs);
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
        owned_spec::<MaterialId>(F::R16Uint, SizePolicy::MatchSurface);
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
    fn hdr_is_scene_linear_half_float() {
        assert_eq!(Hdr::NAME, "hdr");
        assert_eq!(Hdr::kind(), ResourceKind::FrameOwned);
        assert_eq!(Hdr::spec(F::Rgba8UnormSrgb).format, F::Rgba16Float);
        assert_eq!(Hdr::spec(F::Bgra8UnormSrgb).format, F::Rgba16Float);
        assert_eq!(Hdr::spec(SURFACE).size, SizePolicy::MatchSurface);
        assert_eq!(Hdr::spec(SURFACE).samples, 1);
    }

    #[test]
    fn multisampled_policy_marks_geometry_layers() {
        // Pool MSAA policy: geometry targets follow the plan sample count,
        // fullscreen/external layers stay single-sample (see
        // `FrameResource::multisampled`). The canonical `spec()` stays 1x —
        // the count applies at registration (`SystemSet::register_resource`).
        for msaa in [
            Albedo::multisampled(),
            Normal::multisampled(),
            MaterialId::multisampled(),
            WorldPosition::multisampled(),
            MaterialParams::multisampled(),
            Depth::multisampled(),
            HdrFwd::multisampled(),
        ] {
            assert!(msaa, "geometry layer must follow the plan count");
        }
        for single in [
            Hdr::multisampled(),
            Bloom0::multisampled(),
            Bloom1::multisampled(),
            Bloom2::multisampled(),
            Target::multisampled(),
        ] {
            assert!(!single, "fullscreen/external layer must stay 1x");
        }
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
        assert!(v.iter().all(|a| !a.write() && a.clear.is_none()));
        v.iter().map(|a| a.name).collect()
    }

    fn writes_of<P: FramePass>() -> Vec<(&'static str, Option<wgpu::Color>)> {
        let mut v = Vec::new();
        P::Writes::collect_accesses(&mut v);
        assert!(v.iter().all(|a| a.write()));
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
        assert!(matches!(
            OwnsDepth::DEPTH,
            crate::flags::DepthOwnership::Owned
        ));
        assert!(matches!(
            SharedDepth::DEPTH,
            crate::flags::DepthOwnership::Shared
        ));
        assert!(OwnsDepth::SHADOWS.is_enabled());
        assert!(!SharedDepth::SHADOWS.is_enabled());

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
    fn fog_defaults_to_disabled_identity() {
        use ornis_core::units::{Clamped01, LinearRgb};
        // Default state is Disabled (exact no-op).
        assert!(FogState::default().is_disabled());
        assert!(!FogState::default().is_enabled());
        assert_eq!(FogState::default().settings(), None);
        assert!(FogPass::default().is_disabled());
        assert_eq!(FogPass::default().name(), "fog");
        // Opt-in wiring: reads the deferred HDR layer + g-buffer depth
        // (depth is the hardware buffer, linearized on the GPU — see
        // `fog_generated`), writes the target.
        assert_eq!(reads_of::<FogPass>(), vec!["hdr", "depth"]);
        assert_eq!(writes_of::<FogPass>(), vec![("target", None)]);
        // Disabled leaves every sample unchanged: 0 differences, including
        // degenerate depths.
        let samples = [
            ([1.0, 0.0, 0.0], 0.0),
            ([0.0, 1.0, 0.0], 1.5),
            ([0.2, 0.3, 0.9], 100.0),
            ([0.0, 0.0, 0.0], 1000.0),
            ([0.4, 0.2, 0.1], -5.0),
            ([0.4, 0.2, 0.1], f32::NAN),
            ([0.4, 0.2, 0.1], f32::INFINITY),
        ];
        let mut diffs = 0usize;
        for (color, depth) in samples {
            if apply_fog(color, depth, FogState::Disabled) != color {
                diffs += 1;
            }
        }
        assert_eq!(diffs, 0, "disabled fog must be pixel-identical");
        // Density rejects non-positive input at the type level.
        assert!(FogDensity::try_new(0.0).is_none());
        assert!(FogDensity::try_new(-1.0).is_none());
        assert!(FogDensity::try_new(f32::NAN).is_none());
        assert!(FogDensity::try_new(f32::INFINITY).is_none());
        assert_eq!(FogDensity::expect_valid(0.1).get(), 0.1);
        assert!(FogSettings::try_from_raw([0.5, 0.6, 0.7], 0.0).is_none());
        // Enabled fog moves toward the fog color with depth, never past it.
        let fog = FogState::Enabled(
            FogSettings::try_from_raw([0.5, 0.6, 0.7], 0.1).expect("positive density"),
        );
        assert!(!FogPass::new(fog).is_disabled());
        assert!(fog.is_enabled());
        let near = apply_fog([0.0, 0.0, 0.0], 0.5, fog);
        let far = apply_fog([0.0, 0.0, 0.0], 50.0, fog);
        let fog_color = [0.5, 0.6, 0.7];
        for i in 0..3 {
            assert!(near[i] > 0.0 && near[i] < far[i], "{near:?} {far:?}");
            assert!(far[i] < fog_color[i], "{far:?}");
        }
        // Factor stays in [0, 1): zero depth is zero fog, far depth
        // approaches (never reaches) full fog.
        let density = FogDensity::expect_valid(0.1);
        assert_eq!(fog_factor(0.0, density), Clamped01::ZERO);
        assert_eq!(fog_factor(-3.0, density), Clamped01::ZERO);
        assert_eq!(fog_factor(f32::NAN, density), Clamped01::ZERO);
        let far_factor = fog_factor(100.0, density).get();
        assert!(far_factor > 0.999 && far_factor < 1.0, "{far_factor}");
        // At depth 0 the output equals the input exactly (factor 0).
        let settings = FogSettings::new(LinearRgb::new([0.9, 0.1, 0.1]), density);
        assert_eq!(
            apply_fog([0.2, 0.4, 0.6], 0.0, FogState::Enabled(settings)),
            [0.2, 0.4, 0.6]
        );
    }

    #[test]
    fn fog_cpu_reference_parity() {
        // Independent reference of the pinned formula (not via `apply_fog`):
        // `color + (fog.color - color) * (1 - exp(-density * depth))` with
        // negative/non-finite depths clamped to zero fog. Tolerance is tight
        // (f32 round-trip of one exp + fused multiply-add chain).
        const TOL: f32 = 1e-6;
        let densities: [f32; 4] = [0.01, 0.05, 0.2, 1.0];
        let depths: [f32; 6] = [0.0, 0.1, 0.5, 2.0, 10.0, 100.0];
        let colors = [[0.0, 0.0, 0.0], [1.0, 0.5, 0.25], [0.2, 0.8, 0.4]];
        let fog_color = [0.5, 0.6, 0.7];
        for density in densities {
            let fog = FogState::Enabled(
                FogSettings::try_from_raw(fog_color, density).expect("positive density"),
            );
            for depth in depths {
                for color in colors {
                    let reference = {
                        let d = depth.max(0.0);
                        let f = 1.0 - (-density * d).exp();
                        [
                            color[0] + (fog_color[0] - color[0]) * f,
                            color[1] + (fog_color[1] - color[1]) * f,
                            color[2] + (fog_color[2] - color[2]) * f,
                        ]
                    };
                    let actual = apply_fog(color, depth, fog);
                    for i in 0..3 {
                        assert!(
                            (actual[i] - reference[i]).abs() <= TOL,
                            "density={density} depth={depth} color={color:?}: {actual:?} vs {reference:?}"
                        );
                    }
                    // Never overshoots: each channel stays between the
                    // input and the fog color.
                    for i in 0..3 {
                        let (lo, hi) = if color[i] <= fog_color[i] {
                            (color[i], fog_color[i])
                        } else {
                            (fog_color[i], color[i])
                        };
                        assert!(
                            actual[i] >= lo - TOL && actual[i] <= hi + TOL,
                            "overshoot: {actual:?} not in [{lo}, {hi}]"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn composite_modes_encode_technique_and_bloom() {
        assert_eq!(
            Composite::<CompositeDeferredBloom>::new().name(),
            "composite"
        );
        assert_eq!(Composite::<CompositeForward>::default().name(), "composite");

        // (technique, bloom, expected reads)
        let cases: [(
            crate::flags::CompositeTechnique,
            crate::flags::Bloom,
            &[&str],
        ); 6] = [
            (
                CompositeDeferredBloom::TECHNIQUE,
                CompositeDeferredBloom::BLOOM,
                &["hdr", "bloom0"],
            ),
            (
                CompositeDeferred::TECHNIQUE,
                CompositeDeferred::BLOOM,
                &["hdr"],
            ),
            (
                CompositeHybridBloom::TECHNIQUE,
                CompositeHybridBloom::BLOOM,
                &["hdr", "hdr_fwd", "bloom0"],
            ),
            (
                CompositeHybrid::TECHNIQUE,
                CompositeHybrid::BLOOM,
                &["hdr", "hdr_fwd"],
            ),
            (
                CompositeForwardBloom::TECHNIQUE,
                CompositeForwardBloom::BLOOM,
                &["hdr_fwd", "bloom0"],
            ),
            (
                CompositeForward::TECHNIQUE,
                CompositeForward::BLOOM,
                &["hdr_fwd"],
            ),
        ];
        let expected_modes = [0, 0, 2, 2, 1, 1];
        let expected_bloom = [true, false, true, false, true, false];
        for (i, (technique, bloom, _)) in cases.iter().enumerate() {
            assert_eq!(technique.shader_mode(), expected_modes[i], "case {i}");
            assert_eq!(bloom.is_on(), expected_bloom[i], "case {i}");
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
