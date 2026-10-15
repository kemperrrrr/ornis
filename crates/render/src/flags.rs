//! Typed replacements for `bool`/`u32` flags across the frame plan.
//!
//! Each flag previously carried its meaning in the parameter name
//! (`write: bool`, `enabled: bool`, `BLOOM: bool`, `SHADER_MODE: u32`, ...),
//! forcing every call site to re-derive the polarity. The enums below make
//! the polarity explicit at the type level while keeping `From<bool>`
//! conversions so existing tests and call sites migrate incrementally.

/// Read/write access to a frame resource (typed replacement for
/// [`crate::system::AccessDesc`] `write: bool`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Access {
    /// Sample the resource's contents; never creates the texture.
    #[default]
    Read,
    /// Produce the resource's contents for downstream passes.
    Write,
}

impl Access {
    /// `true` for [`Access::Write`].
    pub fn is_write(self) -> bool {
        matches!(self, Self::Write)
    }
}

impl From<bool> for Access {
    /// Legacy `write: bool` polarity.
    fn from(write: bool) -> Self {
        if write { Self::Write } else { Self::Read }
    }
}

impl From<Access> for bool {
    /// Legacy `write: bool` polarity.
    fn from(access: Access) -> bool {
        access.is_write()
    }
}

/// Execution state of a declared pass (typed replacement for
/// `set_pass_enabled` / `PassNode` `enabled: bool`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PassState {
    /// The pass runs and owns slots in the layout.
    #[default]
    Enabled,
    /// The pass is culled: dropped from the layout, its resources get no
    /// slots unless used elsewhere.
    Disabled,
}

impl PassState {
    /// `true` for [`PassState::Enabled`].
    pub fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

impl From<bool> for PassState {
    /// Legacy `enabled: bool` polarity.
    fn from(enabled: bool) -> Self {
        if enabled {
            Self::Enabled
        } else {
            Self::Disabled
        }
    }
}

impl From<PassState> for bool {
    /// Legacy `enabled: bool` polarity.
    fn from(state: PassState) -> bool {
        state.is_enabled()
    }
}

/// Where a frame resource's storage comes from (typed replacement for
/// `ResourceNode` / `ResourceLayout` `external: bool`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ResourceBacking {
    /// Created and owned by the plan (transient, pooled).
    #[default]
    Pooled,
    /// Backed by an externally provided view (swapchain or similar);
    /// never pooled, `slot` is always `None`.
    External,
}

impl ResourceBacking {
    /// `true` for [`ResourceBacking::External`].
    pub fn is_external(self) -> bool {
        matches!(self, Self::External)
    }
}

impl From<bool> for ResourceBacking {
    /// Legacy `external: bool` polarity.
    fn from(external: bool) -> Self {
        if external {
            Self::External
        } else {
            Self::Pooled
        }
    }
}

impl From<ResourceBacking> for bool {
    /// Legacy `external: bool` polarity.
    fn from(backing: ResourceBacking) -> bool {
        backing.is_external()
    }
}

/// Depth-buffer ownership of the forward pass (typed replacement for
/// `ForwardMode::OWNS_DEPTH: bool`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DepthOwnership {
    /// The pass clears the depth buffer itself (forward-only technique).
    Owned,
    /// The pass reads the depth the gbuffer pass filled (hybrid).
    Shared,
}

impl DepthOwnership {
    /// `true` for [`DepthOwnership::Owned`] (pass clears depth itself).
    pub fn clears_depth(self) -> bool {
        matches!(self, Self::Owned)
    }
}

impl From<bool> for DepthOwnership {
    /// Legacy `OWNS_DEPTH: bool` polarity.
    fn from(owns: bool) -> Self {
        if owns { Self::Owned } else { Self::Shared }
    }
}

impl From<DepthOwnership> for bool {
    /// Legacy `OWNS_DEPTH: bool` polarity.
    fn from(ownership: DepthOwnership) -> bool {
        ownership.clears_depth()
    }
}

/// Whether a shadow map is cast (typed replacement for `shadow: bool` on
/// lights and `ForwardMode::OWNS_SHADOWS: bool` on the forward pass).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ShadowCast {
    /// No shadow map (default; absent in older scene files).
    #[default]
    Disabled,
    /// Depth pre-pass + PCF sampling in the evaluators.
    Enabled,
}

impl ShadowCast {
    /// `true` for [`ShadowCast::Enabled`].
    pub fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

impl From<bool> for ShadowCast {
    /// Legacy `shadow: bool` polarity.
    fn from(shadow: bool) -> Self {
        if shadow {
            Self::Enabled
        } else {
            Self::Disabled
        }
    }
}

impl From<ShadowCast> for bool {
    /// Legacy `shadow: bool` polarity.
    fn from(cast: ShadowCast) -> bool {
        cast.is_enabled()
    }
}

/// wgpu LOD upper clamp shared by linear samplers (covers all mip levels).
const LOD_MAX_CLAMP: f32 = 32.0;

/// Ceiling of an anisotropic request: `wgpu` clamps every `anisotropy_clamp`
/// to `1..=16`, and 16 is the `maxSamplerAnisotropy` every target driver
/// reports.
const ANISOTROPY_MAX_CLAMP: u16 = 16;

/// Anisotropic filtering level of a material sampler (`anisotropy_clamp`).
///
/// `1` is isotropic trilinear filtering: one filtered sample per pixel, taken
/// at the mip level of the footprint's *longest* axis, so a textured surface
/// seen at a grazing angle (a floor, a road) blurs well before its texels run
/// out. `2..=16` let the filter take up to that many samples along the longest
/// axis and size the mip level by the shortest one, which is what keeps those
/// surfaces sharp into the distance.
///
/// Shadow and post-process samplers are [`SamplerKind::LinearClamp`] and have
/// no level at all — their fullscreen quads never magnify a grazing surface.
///
/// `wgpu` requires linear `mag`/`min`/`mipmap` filtering for any level above
/// `1` (every material sampler here has it) and silently degrades to `1` on an
/// adapter without `DownlevelFlags::ANISOTROPIC_FILTERING`, so a level is safe
/// to request on every backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AnisotropyLevel(u16);

impl AnisotropyLevel {
    /// Isotropic trilinear filtering: anisotropy disabled (what every
    /// sampler of this crate used before `#122`).
    pub const OFF: Self = Self(1);

    /// Project default for material samplers: the driver ceiling, `16`.
    pub const DEFAULT: Self = Self(ANISOTROPY_MAX_CLAMP);

    /// Clamping constructor for a requested level.
    ///
    /// Nothing weaker than [`AnisotropyLevel::OFF`] (a `0` from a settings
    /// file, say) maps to it, and anything above [`AnisotropyLevel::DEFAULT`]
    /// saturates at the driver ceiling, so a caller needs no range check of
    /// its own.
    // qual:api — public configuration entry; in-tree callers are the unit tests.
    pub const fn new(requested: u16) -> Self {
        if requested < Self::OFF.0 {
            Self::OFF
        } else if requested > ANISOTROPY_MAX_CLAMP {
            Self::DEFAULT
        } else {
            Self(requested)
        }
    }

    /// Raw `anisotropy_clamp` handed to `wgpu`, in `1..=16`.
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// Sampler selection for frame/upload textures (typed replacement for
/// inline `wgpu::SamplerDescriptor` literals scattered across the
/// composite, renderer and texture-upload paths).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SamplerKind {
    /// Linear `mag`/`min`, nearest `mipmap`, clamp-to-edge on every axis:
    /// fullscreen blits (composite, HDR layers, shadow maps). Always
    /// isotropic ([`AnisotropyLevel::OFF`]).
    #[default]
    LinearClamp,
    /// Linear `mag`/`min`/`mipmap`, repeat on every axis (the glTF
    /// default — the import carries no wrap mode): color/data textures, at
    /// the carried [`AnisotropyLevel`].
    LinearRepeat(AnisotropyLevel),
}

impl SamplerKind {
    /// The `wgpu` descriptor for this sampler (label is `None` so the
    /// value stays `'static` and comparable in tests).
    pub fn descriptor(self) -> wgpu::SamplerDescriptor<'static> {
        match self {
            Self::LinearClamp => wgpu::SamplerDescriptor {
                label: None,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                lod_min_clamp: 0.0,
                lod_max_clamp: LOD_MAX_CLAMP,
                compare: None,
                anisotropy_clamp: AnisotropyLevel::OFF.get(),
                border_color: None,
            },
            Self::LinearRepeat(anisotropy) => wgpu::SamplerDescriptor {
                label: None,
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                lod_min_clamp: 0.0,
                lod_max_clamp: LOD_MAX_CLAMP,
                compare: None,
                anisotropy_clamp: anisotropy.get(),
                border_color: None,
            },
        }
    }
}

/// HDR compositing technique (typed replacement for
/// `CompositeMode::SHADER_MODE: u32`): which HDR layers exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompositeTechnique {
    /// Deferred path only (`hdr`).
    Deferred,
    /// Both HDR layers (`hdr` + `hdr_fwd`).
    Hybrid,
    /// Forward path only (`hdr_fwd`).
    Forward,
}

impl CompositeTechnique {
    /// Value of the shader's layer-mix selector (`CompositeInputs::mode`).
    pub fn shader_mode(self) -> u32 {
        match self {
            Self::Deferred => 0,
            Self::Forward => 1,
            Self::Hybrid => 2,
        }
    }

    /// Decodes the legacy `SHADER_MODE: u32` encoding; unknown modes
    /// fall back to deferred (the historical `0`).
    pub fn from_shader_mode(mode: u32) -> Self {
        match mode {
            1 => Self::Forward,
            2 => Self::Hybrid,
            _ => Self::Deferred,
        }
    }
}

/// Whether the bloom chain feeds the composite mix (typed replacement
/// for `CompositeMode::BLOOM: bool`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bloom {
    /// The bloom chain contributes to the mix (`bloom_intensity = 1.0`).
    On,
    /// Bloom culled: dead layers bind a live view with zero effect.
    Off,
}

impl Bloom {
    /// `true` for [`Bloom::On`].
    pub fn is_on(self) -> bool {
        matches!(self, Self::On)
    }

    /// Intensity multiplier consumed by the composite inputs.
    pub fn intensity(self) -> f32 {
        if self.is_on() { 1.0 } else { 0.0 }
    }
}

impl From<bool> for Bloom {
    /// Legacy `BLOOM: bool` polarity.
    fn from(bloom: bool) -> Self {
        if bloom { Self::On } else { Self::Off }
    }
}

impl From<Bloom> for bool {
    /// Legacy `BLOOM: bool` polarity.
    fn from(bloom: Bloom) -> bool {
        bloom.is_on()
    }
}
