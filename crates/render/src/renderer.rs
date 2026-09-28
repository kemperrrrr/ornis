//! The deferred `Renderer3D`: imperative hybrid pipeline of
//! gbuffer (5 MRT) -> lighting -> forward -> composite passes, plus the bloom
//! chain. This is the production implementation behind
//! [`crate::render_backend::RenderBackend`]; see also [`crate::frame_exec`]
//! for the render-graph-driven equivalent.

use crate::mesh::{Mesh, SkinnedVertex, Vertex};
use crate::shaders;
use crate::skinning::{
    GBUFFER_SKINNED_RESOURCES, PALETTE_BYTE_SIZE, PaletteHandle, skinned_entry_point,
    wgsl_vertex_source_skinned,
};
use crate::textures::{
    CpuImage, GpuTexture, MaterialTextureSet, TextureCache, TextureRole,
    sampler_descriptor_for_role, upload_texture,
};
use glam::Mat4;
use ornis_assets::scene::LightDesc;
use ornis_core::material::{OPENPBR_MATERIAL_SIZE, OpenPBRMaterial};
use ornis_macros::WgslStruct;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

/// Frame-global camera uniform (binding shared by every pass).
///
/// The WGSL `Camera` declaration is generated from this layout
/// ([`CameraUniform::WGSL_SOURCE`]); the field list here is the single source
/// of truth for the buffer layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "Camera")]
pub struct CameraUniform {
    /// View-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Its inverse: reconstructs world position from depth in the lighting pass.
    pub inv_view_proj: [[f32; 4]; 4],
    /// World-space eye position (`w` = 1) for specular falloff.
    pub camera_pos: [f32; 4],
}

/// GPU per-instance record mirroring CPU [`InstanceData`] with padding to 16 bytes.
///
/// The WGSL `PerObject` declaration is generated from this layout
/// ([`PerObjectGpu::WGSL_SOURCE`]); the field list here is the single source
/// of truth for the buffer layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "PerObject")]
pub struct PerObjectGpu {
    /// Local-to-world matrix.
    pub model: [[f32; 4]; 4],
    /// Inverse-transpose model matrix (normal transform).
    pub normal_matrix: [[f32; 4]; 4],
    /// Index into the material buffer uploaded by `upload_materials`.
    pub material_index: MaterialIdx,
    /// Aligns the record to 16-byte stride (padding: not shader-visible).
    #[wgsl(skip)]
    _padding: [u32; 3],
}

/// One GPU light: direction + color packed as `vec4`s.
///
/// Evaluation kind of a [`GpuLight`] entry: directional, point, or spot.
/// Float (not enum): the stage DSL compares kinds with `>`.
pub(crate) const LIGHT_KIND_DIRECTIONAL: f32 = 0.0;
/// Evaluation kind of a [`GpuLight`] entry: point light.
pub(crate) const LIGHT_KIND_POINT: f32 = 1.0;
/// Evaluation kind of a [`GpuLight`] entry: spotlight.
pub(crate) const LIGHT_KIND_SPOT: f32 = 2.0;

/// Maximum lights uploaded per frame: the [`LightingUniform`] WGSL block
/// spells `array<Light, 8>`, and [`Renderer3D::set_lights`] uploads the
/// first eight entries of any kind. Excess lights are dropped and reported
/// in [`LightUploadStats::dropped_lights`] (never silently).
pub const MAX_LIGHTS: usize = 8;
/// Compile-time pin: the WGSL derive only accepts integer literals for
/// array lengths, so [`LightingUniform::lights`] spells `8` literally —
/// this assert keeps the spell and the limit in sync.
const _: [(); MAX_LIGHTS] = [(); 8];

/// Per-`set_lights` upload report: explicit counts for what the old
/// silent-truncate path dropped (excess lights, shadow requests without a
/// free layer/cube slot). Returned by
/// [`Renderer3D::set_lights_full`] and [`count_light_drops`], and published
/// for the last upload via [`Renderer3D::light_upload_stats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightUploadStats {
    /// Lights written into the uniform (`min(len, MAX_LIGHTS)`).
    pub uploaded: u32,
    /// Scene lights beyond [`MAX_LIGHTS`] that were not uploaded.
    pub dropped_lights: u32,
    /// Shadow requests (`shadow: ShadowCast::Enabled`) that received no map slot: 2D
    /// layers ([`SHADOW_LAYERS`]) for directional/spot lights and cube
    /// slots ([`POINT_SHADOW_CUBES`]) for point lights, including requests
    /// on dropped excess lights. These lights render unshadowed.
    pub dropped_shadows: u32,
}

/// Shadow-map layers (one per light slot).
pub const SHADOW_LAYERS: usize = 4;
/// Shadow-map resolution in pixels (square).
///
/// Depth is `Depth32Float`; 1024² × 4 layers ≈ 16 MiB, allocated once
/// at construction. Only lights with `shadow: ShadowCast::Enabled` render into a
/// layer; the rest of the array stays cleared and is never sampled
/// (`params.w = -1.0` skips the lookup branchlessly).
pub const SHADOW_SIZE: u32 = 1024;
/// Directional shadow ortho box: ±half-extent around the origin, light
/// eye at `SHADOW_DIR_DIST` along the to-light direction.
pub const SHADOW_ORTHO_HALF: f32 = 12.0;
const SHADOW_DIR_DIST: f32 = 30.0;

/// Depth-bias pair for the shadow pre-pass (2D layers and cube faces
/// share it). The constant term is negligible on `Depth32Float`; the
/// slope term dominates on curved surfaces — large values erode small
/// occluder blobs in the map, small values risk acne (guarded by
/// `shadowed_directional_without_occluder_has_no_acne`).
const SHADOW_DEPTH_BIAS_CONSTANT: i32 = 2;
const SHADOW_DEPTH_BIAS_SLOPE: f32 = 1.0;

/// Point-light shadow cubes (one depth cube per slot) and resolution.
///
/// 512² × 6 faces × 2 cubes ≈ 12 MiB. Sampling is analytic: the
/// hardware picks the face from the fragment→light vector's major
/// axis, and the reference depth uses the same 90° perspective
/// formula as the face renders (near plane [`SHADOW_CUBE_NEAR`], far
/// plane = light range, so no new uniforms are needed).
pub const POINT_SHADOW_CUBES: usize = 2;
/// Face resolution of one point-shadow cube.
pub const SHADOW_CUBE_SIZE: u32 = 512;
/// Near plane shared by the cube-face renders and the analytic
/// sampling formula — change both together.
pub const SHADOW_CUBE_NEAR: f32 = 0.1;
/// Cube-face axes (direction from the light) with the ups that reproduce
/// the hardware cube-sampling frame (OpenGL/Metal convention: +X: (−z,−y),
/// −X: (+z,−y), +Y: (+x,+z), −Y: (+x,−z), +Z: (+x,−y), −Z: (−x,−y)).
/// The U direction matches by construction; the V match needs the
/// projection Y-mirror in [`point_cube_face_vp`] (rasterization puts
/// NDC y+1 at texture row 0, the sampler reads v=0 from the top).
/// Keep both together: correct depths at mirrored texels are what the
/// sampler then misses (all-lit point shadows).
const CUBE_FACES: [([f32; 3], [f32; 3]); 6] = [
    ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
    ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
    ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
    ([0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
    ([0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
];

/// GPU light entry: kind-selected evaluation in both fragment entries
/// (deferred lighting and forward PBR share the layer evaluators).
///
/// The WGSL `Light` declaration is generated from this layout
/// ([`GpuLight::WGSL_SOURCE`]). `align(16)` matches the WGSL struct alignment
/// so the derive's nested-layout check holds (cf. physics `GpuBodyState`).
///
/// Layout note: `kind` is a full vec4 (kind in `x`) rather than a scalar —
/// a bare `f32` ahead of the vec4s would insert implicit padding that
/// `bytemuck::Pod` rejects and the derive cannot spell.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "Light")]
pub(crate) struct GpuLight {
    /// Evaluation kind in `x`: 0.0 directional, 1.0 point, 2.0 spot.
    kind: [f32; 4],
    /// Direction toward the light (directional) or spot axis from the
    /// light into the scene (spot); `[0, 0, 1, 0]` for points.
    direction: [f32; 4],
    /// World-space position (point/spot); unused by directionals.
    position: [f32; 4],
    /// Emission color RGB + radiometric intensity in alpha.
    color: [f32; 4],
    /// `(range, cos_inner, cos_outer, shadow_layer)`: range cutoff for
    /// point/spot, spot cone cosines, shadow-map layer or -1.0.
    pub params: [f32; 4],
    /// Light-space clip matrix for the shadow map (`params.w` layer);
    /// identity when the light casts no shadow.
    pub shadow_vp: [[f32; 4]; 4],
}

/// Typed evaluation kind of a [`GpuLight`] entry (mirrors `LIGHT_KIND_*`
/// scalars carried in `kind.x`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum GpuLightKind {
    /// Directional light (`0.0`).
    Directional,
    /// Point light (`1.0`).
    Point,
    /// Spotlight (`2.0`).
    Spot,
}

impl GpuLightKind {
    /// Raw kind scalar (`kind.x`).
    #[allow(dead_code)]
    pub(crate) fn as_scalar(self) -> f32 {
        match self {
            Self::Directional => LIGHT_KIND_DIRECTIONAL,
            Self::Point => LIGHT_KIND_POINT,
            Self::Spot => LIGHT_KIND_SPOT,
        }
    }

    /// Classifies a raw kind scalar; `None` for unknown values.
    #[allow(dead_code)]
    pub(crate) fn from_scalar(v: f32) -> Option<Self> {
        if v == LIGHT_KIND_DIRECTIONAL {
            Some(Self::Directional)
        } else if v == LIGHT_KIND_POINT {
            Some(Self::Point)
        } else if v == LIGHT_KIND_SPOT {
            Some(Self::Spot)
        } else {
            None
        }
    }
}

impl GpuLight {
    /// Evaluation kind scalar carried in `kind.x` (0/1/2 =
    /// directional/point/spot, see `LIGHT_KIND_*`).
    #[allow(dead_code)]
    pub(crate) fn kind_scalar(&self) -> f32 {
        self.kind[0]
    }

    /// Typed evaluation kind from `kind.x`; `None` for unknown scalars.
    /// Prefer [`GpuLight::kind_scalar`] only for raw shader debugging.
    #[allow(dead_code)]
    pub(crate) fn kind_units(&self) -> Option<GpuLightKind> {
        GpuLightKind::from_scalar(self.kind[0])
    }

    /// Emission color (RGB, exposure-baked) plus radiometric intensity in
    /// alpha, as linear RGBA.
    #[allow(dead_code)]
    pub(crate) fn color_units(&self) -> ornis_core::units::LinearRgba {
        ornis_core::units::LinearRgba::new(self.color)
    }

    /// Range cutoff carried in `params.x` (point/spot).
    #[allow(dead_code)]
    pub(crate) fn range_cutoff(&self) -> f32 {
        self.params[0]
    }

    /// Range cutoff in meters (point/spot).
    #[allow(dead_code)]
    pub(crate) fn range_units(&self) -> ornis_core::units::Meters {
        ornis_core::units::Meters::new(self.params[0])
    }
}

/// Lighting uniform block: ambient + fixed light array + count.
///
/// The WGSL `Lighting` declaration is generated from this layout
/// ([`LightingUniform::WGSL_SOURCE`]); the field list here is the single
/// source of truth for the buffer layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "Lighting")]
pub(crate) struct LightingUniform {
    ambient_color: [f32; 4],
    /// Eight lights; spelled `array<Light, 8>` in WGSL (the inner struct's
    /// `name` override is not visible here, hence the explicit `to`).
    /// The evaluators iterate `0..light_count`, so directional-only scenes
    /// with ≤4 lights render pixel-identical to the old 4-wide block.
    #[wgsl(to = "Light")]
    lights: [GpuLight; 8],
    light_count: u32,
    /// Trailing pad to a 16-multiple size (padding: not shader-visible).
    #[wgsl(skip)]
    _pad: [u32; 3],
}

/// Index into the deduplicated material table ([`FrameUpload::materials`]).
///
/// Newtype over `u32` so material indices never mix with texture handles
/// or entity ids at the type level. The DSL substitutes it to WGSL `u32`
/// (like the CPU compiler erases `repr(transparent)` wrappers), so GPU
/// records ([`PerObjectGpu`], shader interfaces) spell it directly.
#[repr(transparent)]
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    bytemuck::Pod,
    bytemuck::Zeroable,
)]
pub struct MaterialIdx(u32);

impl MaterialIdx {
    /// Wraps a raw `u32` material table index.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` material table index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Material index as `usize` for table lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for MaterialIdx {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for MaterialIdx {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

impl From<MaterialIdx> for u32 {
    fn from(h: MaterialIdx) -> Self {
        h.0
    }
}

impl From<MaterialIdx> for usize {
    fn from(h: MaterialIdx) -> Self {
        h.0 as usize
    }
}

/// CPU-side description of one drawn instance.
#[derive(Debug, Clone, Copy)]
pub struct InstanceData {
    /// Local-to-world matrix.
    pub model_matrix: Mat4,
    /// Normal matrix (inverse transpose of the linear part).
    pub normal_matrix: Mat4,
    /// Index into the material table.
    pub material_index: MaterialIdx,
}

/// G-buffer texture views, fed either from persistent textures (legacy
/// path) or from render-plan pool slots (plan path).
pub struct GbufferTargets<'a> {
    /// Base albedo (sRGB) view.
    pub albedo: &'a wgpu::TextureView,
    /// View-space/world normals view.
    pub normal: &'a wgpu::TextureView,
    /// Material identifier view.
    pub material_id: &'a wgpu::TextureView,
    /// World-space positions view.
    pub world_position: &'a wgpu::TextureView,
    /// Material parameter table view.
    pub material_params: &'a wgpu::TextureView,
    /// Depth buffer view.
    pub depth: &'a wgpu::TextureView,
}

/// Persistent g-buffer textures of the legacy path (plan path draws into
/// pooled slots instead) plus their views.
pub struct GBufferTextures {
    /// Albedo/base color target (Rgba8Unorm).
    pub albedo: wgpu::Texture,
    /// View of [`GBufferTextures::albedo`].
    pub albedo_view: wgpu::TextureView,
    /// World-space normal target (Rg16Float).
    pub normal: wgpu::Texture,
    /// View of [`GBufferTextures::normal`].
    pub normal_view: wgpu::TextureView,
    /// Material id target (R32Uint).
    pub material_id: wgpu::Texture,
    /// View of [`GBufferTextures::material_id`].
    pub material_id_view: wgpu::TextureView,
    /// World-space position target (Rg16Float xy + z from depth).
    pub world_position: wgpu::Texture,
    /// View of [`GBufferTextures::world_position`].
    pub world_position_view: wgpu::TextureView,
    /// Material parameter target (Rgba16Float).
    pub material_params: wgpu::Texture,
    /// View of [`GBufferTextures::material_params`].
    pub material_params_view: wgpu::TextureView,
    /// Depth buffer (Depth32Float), reused by the forward pass.
    pub depth: wgpu::Texture,
    /// View of [`GBufferTextures::depth`].
    pub depth_view: wgpu::TextureView,
}

/// Full-screen deferred lighting pass: reads the five g-buffer targets +
/// depth, evaluates the PBR BRDF, writes the HDR color image.
pub struct LightingPass {
    /// Full-screen triangle-strip pipeline writing Rgba16Float HDR.
    pipeline: wgpu::RenderPipeline,
    /// Bindings: camera/lighting/material buffers, 5 gbuffer views, depth, sampler.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Linear sampler for gbuffer fetches (MSAA resolve handled upstream).
    sampler: wgpu::Sampler,
}

/// Blend mode of the forward HDR layer (typed replacement for the
/// `sorted_alpha: bool` flag).
///
/// [`BlendMode::Opaque`] (default) writes with `REPLACE`: the default frame
/// (all opacities at 1.0, where `REPLACE` and `ALPHA_BLENDING` coincide)
/// stays golden-pinned. [`BlendMode::Transparent`] blends with
/// `ALPHA_BLENDING` and requires back-to-front submission order — see
/// [`TransparencyOptions`]: sort with [`sort_by_depth`] (per-instance
/// depths) or [`crate::extraction::sort_by_depth`] (whole [`crate::extraction::FrameUpload`])
/// before uploading, farthest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlendMode {
    /// Opaque forward layer (`REPLACE`, default).
    #[default]
    Opaque,
    /// Sorted alpha blend (`ALPHA_BLENDING`, opt-in; back-to-front
    /// submission order — see [`TransparencyOptions`], including the
    /// single-draw order contract).
    Transparent,
}

impl BlendMode {
    /// `true` for [`BlendMode::Transparent`].
    pub fn is_transparent(self) -> bool {
        matches!(self, Self::Transparent)
    }

    /// Blend state of the forward pipeline for this mode.
    pub fn blend_state(self) -> wgpu::BlendState {
        match self {
            Self::Transparent => wgpu::BlendState::ALPHA_BLENDING,
            Self::Opaque => wgpu::BlendState::REPLACE,
        }
    }

    /// Maps a material opacity to a blend mode: finite opacities `>= 1.0`
    /// are opaque, finite opacities below that are transparent.
    ///
    /// # Errors
    ///
    /// Returns [`TransparencyError::NonFiniteOpacity`] for non-finite input.
    pub fn from_opacity(opacity: f32) -> Result<Self, TransparencyError> {
        if !opacity.is_finite() {
            return Err(TransparencyError::NonFiniteOpacity(opacity));
        }
        if opacity >= 1.0 {
            Ok(Self::Opaque)
        } else {
            Ok(Self::Transparent)
        }
    }
}

impl From<bool> for BlendMode {
    /// Legacy `sorted_alpha: bool` polarity.
    fn from(sorted_alpha: bool) -> Self {
        if sorted_alpha {
            Self::Transparent
        } else {
            Self::Opaque
        }
    }
}

impl From<BlendMode> for bool {
    /// Legacy `sorted_alpha: bool` polarity.
    fn from(mode: BlendMode) -> Self {
        mode.is_transparent()
    }
}

/// Rejected transparency input (fallible opacity classification).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum TransparencyError {
    /// Opacity must be finite to classify into a [`BlendMode`].
    #[error("opacity {0} is not finite")]
    NonFiniteOpacity(f32),
}

/// Opt-in transparency for the forward HDR layer.
///
/// The forward layer is always cleared to transparent black
/// (`ClearTransparent`); [`mode`](Self::mode) only selects the blend state
/// of the forward pipeline (see [`forward_blend_state`]) and whether
/// callers must submit instances back-to-front. With
/// [`BlendMode::Transparent`], sort before uploading: per-instance depths
/// via [`sort_by_depth`], or a whole extracted frame via
/// [`crate::extraction::sort_by_depth`] (both order farthest first,
/// stable). Off by default so the default frame stays golden-pinned.
///
/// Order contract: the current forward submit is a single instanced
/// `draw_indexed` over `0..instance_count` (see
/// [`render_forward`](Renderer3D::render_forward)), which ignores instance
/// order — sorting is a no-op today and only takes effect with future
/// multi-draw submits (one draw per instance, back-to-front).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TransparencyOptions {
    /// Blend mode of the forward pipeline (default [`BlendMode::Opaque`]).
    pub mode: BlendMode,
}

impl TransparencyOptions {
    /// Options for the given [`BlendMode`].
    pub fn new(mode: BlendMode) -> Self {
        Self { mode }
    }

    /// Legacy `sorted_alpha` polarity: `true` when transparent.
    pub fn sorted_alpha(self) -> bool {
        self.mode.is_transparent()
    }
}

impl From<bool> for TransparencyOptions {
    /// Legacy `sorted_alpha: bool` polarity.
    fn from(sorted_alpha: bool) -> Self {
        Self {
            mode: BlendMode::from(sorted_alpha),
        }
    }
}

impl From<BlendMode> for TransparencyOptions {
    /// Wraps the mode into options.
    fn from(mode: BlendMode) -> Self {
        Self { mode }
    }
}

/// Blend state of the forward pipeline for the given transparency
/// options: `REPLACE` by default, `ALPHA_BLENDING` when transparent.
/// Pure (no GPU access) so the default can be pinned without an adapter.
pub fn forward_blend_state(options: TransparencyOptions) -> wgpu::BlendState {
    options.mode.blend_state()
}

/// CPU-side back-to-front draw order for `depths` (view-space depth per
/// instance): indices sorted by descending depth, far first, stable
/// (equal depths keep submission order). `NaN` sorts as farthest
/// (`total_cmp`). Pure helper for the future sorted forward submit: the
/// current single-draw forward pass (one instanced `draw_indexed` in
/// [`render_forward`](Renderer3D::render_forward)) ignores instance order,
/// so sorting is a no-op until multi-draw submits land.
pub fn sort_by_depth(depths: &[f32]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..depths.len() as u32).collect();
    order.sort_by(|&a, &b| depths[b as usize].total_cmp(&depths[a as usize]));
    order
}

/// Forward pass for transparency-friendly objects: draws geometry with full
/// lighting into an HDR layer, testing against the gbuffer's depth.
pub struct ForwardPass {
    /// Lit-forward pipeline (same shading as the lighting pass).
    pipeline: wgpu::RenderPipeline,
    /// Bindings for the forward pipeline (layout is stable).
    bind_group_layout: wgpu::BindGroupLayout,
    /// Current bind group, rebuilt whenever a storage buffer grows.
    bind_group: std::sync::RwLock<wgpu::BindGroup>,
    /// Owned HDR color attachment.
    _color_texture: wgpu::Texture,
    /// View of `_color_texture`.
    color_view: wgpu::TextureView,
}

/// Textured-forward pass: the legacy [`ForwardPass`] evaluation plus one
/// bound [`MaterialTextureSet`] per draw (see
/// [`crate::shaders::material_textures`]).
///
/// Built on demand by
/// [`Renderer3D::ensure_textured_forward`](Renderer3D::ensure_textured_forward)
/// (the constructor takes no queue, so upload happens there, not in
/// [`new`](Renderer3D::new)); the legacy passes never reference it, so
/// untextured frames stay pixel-identical by construction. Unbound role
/// slots resolve to neutral 1x1 white fallbacks (`x * 1.0 == x`,
/// IEEE-exact): color roles share one `Rgba8UnormSrgb` fallback (hardware
/// sRGB-decodes to linear white), the data role owns one `Rgba8Unorm`
/// fallback (linear white, so roughness/metallic factors stand alone).
pub struct TexturedForwardPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    fallback_color: GpuTexture,
    fallback_data: GpuTexture,
    sampler: wgpu::Sampler,
}

/// Final blend pass mixing deferred HDR, forward HDR and bloom into the output.
pub struct CompositePass {
    /// Full-screen triangle-strip pipeline targeting the surface format.
    pipeline: wgpu::RenderPipeline,
    /// Bindings: two HDR layers, sampler, bloom view, params buffer.
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Inputs of the composite pass: the two HDR layers plus the bloom
/// contribution (view + blend intensity). Grouped so the pass signature
/// stays small as the mix gains terms. `mode` selects the blend in the
/// shader: 0 = deferred-only, 1 = forward-only, 2 = hybrid.
pub struct CompositeInputs<'a> {
    /// Output view written by the pass (usually the surface).
    pub target: &'a wgpu::TextureView,
    /// Deferred-lit HDR layer.
    pub hdr: &'a wgpu::TextureView,
    /// Forward-lit HDR layer.
    pub hdr_fwd: &'a wgpu::TextureView,
    /// Bloom contribution texture (may be black when culled).
    pub bloom: &'a wgpu::TextureView,
    /// Multiplier on the bloom contribution (0 disables it).
    pub bloom_intensity: f32,
    /// Layer mix selector in the shader: 0 = deferred-only, 1 = forward-only, 2 = hybrid.
    pub mode: u32,
}

/// Inputs of the opt-in distance-fog pass: the deferred HDR layer plus the
/// g-buffer depth it linearizes, the fog color/density, and the output view.
/// Grouped so the pass signature stays small (like [`CompositeInputs`]).
/// `density` must be finite and `> 0` — anything else records no commands
/// (exact no-op; see [`Renderer3D::render_fog`]).
pub struct FogInputs<'a> {
    /// Deferred-lit HDR layer (fogged in place conceptually; the pass
    /// writes the mix into [`target`](Self::target)).
    pub hdr: &'a wgpu::TextureView,
    /// G-buffer hardware depth buffer (linearized on the GPU).
    pub depth: &'a wgpu::TextureView,
    /// Output view written by the pass (usually the swapchain target).
    pub target: &'a wgpu::TextureView,
    /// Linear-space fog color.
    pub color: [f32; 3],
    /// Exponential density in 1/m.
    pub density: f32,
}

/// Per-frame bloom parameters shared by the bloom passes and the composite
/// pass. `threshold` gates the bright-pass (first downsample level only);
/// `intensity` scales the bloom contribution in the composite pass.
/// `mode` (composite only) picks the layer mix: 0 = deferred-only,
/// 1 = forward-only, 2 = hybrid.
///
/// The WGSL `BloomParams` declaration is generated from this layout
/// ([`BloomUniform::WGSL_SOURCE`]).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "BloomParams")]
pub(crate) struct BloomUniform {
    threshold: f32,
    intensity: f32,
    mode: u32,
    /// Trailing pad (padding: not shader-visible).
    #[wgsl(skip)]
    _pad: f32,
}

impl Default for BloomUniform {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            intensity: 1.0,
            mode: 0,
            _pad: 0.0,
        }
    }
}

/// Per-frame fog parameters for the opt-in distance-fog pass.
///
/// `color` is the linear-space fog color mixed toward with depth;
/// `density` is the exponential falloff rate (1/m, always positive —
/// [`crate::frame_passes::FogState::Disabled`] carries no density at all).
/// Depth is the view-space distance reconstructed from the g-buffer depth
/// buffer (see [`crate::shaders::fog_generated`]): hardware depth is
/// non-linear, so it is linearized through `Camera::inv_view_proj` (the
/// same [`reconstruct_world_pos`](crate::shaders::helpers) path lighting
/// uses) and the Euclidean distance to `Camera::camera_pos` feeds
/// `1 - exp(-density * depth)`.
///
/// The WGSL `FogParams` declaration is generated from this layout
/// ([`FogUniform::WGSL_SOURCE`]); the field list here is the single source
/// of truth for the buffer layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "FogParams")]
pub(crate) struct FogUniform {
    /// Linear-space fog color.
    color: [f32; 3],
    /// Exponential density (1/m, `> 0`).
    density: f32,
}

impl FogUniform {
    /// Packs raw fog color + density for upload.
    pub(crate) fn pack(color: [f32; 3], density: f32) -> Self {
        Self { color, density }
    }
}

/// Bloom pass pipelines: a downsample (replace-blend, clear) and an upsample
/// (additive blend over a loaded target) sharing one fragment shader.
pub struct BloomPass {
    down_pipeline: wgpu::RenderPipeline,
    up_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
}

/// Opt-in distance-fog fullscreen pass (see [`crate::shaders::fog_generated`]).
///
/// Reads the deferred HDR layer plus the g-buffer depth buffer, mixes toward
/// the fog color by `1 - exp(-density * depth)` (depth = view-space distance
/// reconstructed from hardware depth), and writes the swapchain target.
/// Disabled state records no commands (exact no-op); only the enabled mix
/// draws.
pub(crate) struct FogPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
}

/// Shared resources for composite/bloom sampling.
pub struct CompositeResources {
    /// Linear-filtering, clamp-to-edge sampler used by all full-screen passes.
    pub sampler: wgpu::Sampler,
}

/// The four core GPU buffers (camera / per-object / material / lighting).
struct CoreBuffers {
    camera: wgpu::Buffer,
    per_object: wgpu::Buffer,
    material: wgpu::Buffer,
    lighting: wgpu::Buffer,
}

/// The deferred 3D renderer: owns GPU buffers, pipelines and persistent
/// targets; drives the hybrid deferred+forward+bloom frame via its
/// `render_*` methods or the all-in-one [`Renderer3D::render_scene`].
///
/// Interior mutability note: `upload_*` may reallocate growable buffers
/// and rebuild their bind groups, so the buffers live behind an `RwLock`
/// — every other method only reads them, and the parallel command
/// recording path (`Sync`) keeps working. External callers keep the plain
/// `&self` signatures; no `&mut` threading through the schedule systems
/// or the WASM loop was needed.
pub struct Renderer3D {
    camera_buffer: wgpu::Buffer,
    per_object_buffer: std::sync::RwLock<wgpu::Buffer>,
    material_buffer: std::sync::RwLock<wgpu::Buffer>,
    lighting_buffer: wgpu::Buffer,
    _bind_group_layout: wgpu::BindGroupLayout,
    _bind_group: wgpu::BindGroup,
    _pipeline: wgpu::RenderPipeline,
    pbr_texture: wgpu::Texture,
    pbr_texture_view: wgpu::TextureView,
    sample_count: u32,
    /// Exposure multiplier applied by [`set_lights`](Self::set_lights);
    /// `1.0` (default) is the exact no-op. Set from
    /// [`crate::render_backend::RenderBackendConfig::exposure`].
    exposure: f32,
    /// Transparency mode of the forward pipeline (see
    /// [`TransparencyOptions`]); default off (golden-pinned `REPLACE`).
    transparency: TransparencyOptions,
    max_objects: std::sync::atomic::AtomicU32,
    max_materials: std::sync::atomic::AtomicU32,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    gbuffer: GBufferTextures,
    gbuffer_pipeline: wgpu::RenderPipeline,
    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    gbuffer_bind_group: std::sync::RwLock<wgpu::BindGroup>,
    /// Skinned g-buffer pipeline: [`wgsl_vertex_source_skinned`] vertex
    /// stage (palette blend) with the classic fragment stage and the same
    /// five MRT targets. Draws interleaved [`SkinnedVertex`] buffers; bind
    /// groups are built per draw from `skinned_bind_group_layout` (no
    /// cached group — the palette slot differs per entry).
    skinned_pipeline: wgpu::RenderPipeline,
    /// Bind-group layout of the skinned passes (see
    /// [`GBUFFER_SKINNED_RESOURCES`]: camera, per-objects, materials plus
    /// the binding-3 palette).
    skinned_bind_group_layout: wgpu::BindGroupLayout,
    /// Depth-only skinned pipeline for the 2D shadow pre-pass (same vertex
    /// stage, no fragment): skinned entries cast skinned depth, never
    /// bind-pose depth.
    skinned_shadow_pipeline: wgpu::RenderPipeline,
    /// Mirrored (`front_face: Cw`) depth-only skinned pipeline for the
    /// point-shadow cube faces (see [`point_cube_face_vp`]).
    skinned_shadow_cube_pipeline: wgpu::RenderPipeline,
    /// Packed joint-palette storage: one [`PALETTE_BYTE_SIZE`] slot per
    /// staged entry, written by [`upload_skin_palettes`](Self::upload_skin_palettes).
    palette_buffer: std::sync::RwLock<wgpu::Buffer>,
    /// Capacity of [`Renderer3D::palette_buffer`] in palette slots; grows
    /// on demand like the per-object buffer.
    max_palettes: std::sync::atomic::AtomicU32,
    /// Slots written by the last [`upload_skin_palettes`](Self::upload_skin_palettes)
    /// call: draw handles at or past this are stale and record no commands
    /// (exact no-op, never an out-of-bounds bind).
    palette_count: std::sync::atomic::AtomicU32,
    lighting_pass: LightingPass,
    forward_pass: ForwardPass,
    /// Textured-forward pipeline plus fallbacks, built on demand by
    /// [`ensure_textured_forward`](Renderer3D::ensure_textured_forward);
    /// `None` until then (legacy flows never build it).
    textured_forward: Option<TexturedForwardPass>,
    composite_pass: CompositePass,
    /// Linear sampler shared by composite/bloom full-screen passes.
    composite_sampler: wgpu::Sampler,
    /// Bloom chain pipelines and params buffer.
    bloom_pass: BloomPass,
    /// Opt-in distance-fog pass (disabled by default: never runs unless a
    /// [`crate::frame_passes::FogPass`] with an enabled state records it).
    fog: FogPipeline,
    /// Shadow-map array (one depth layer per light slot) plus per-layer
    /// views, light-space VP uniform buffers, the depth-only pipeline,
    /// and the comparison sampler used by both lighting entries.
    shadow_maps: wgpu::Texture,
    shadow_views: [wgpu::TextureView; SHADOW_LAYERS],
    /// Full-array view for sampling (`texture_depth_2d_array`).
    shadow_array_view: wgpu::TextureView,
    shadow_vp_buffers: [wgpu::Buffer; SHADOW_LAYERS],
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_sampler: wgpu::Sampler,
    /// Shadow-casting lights assigned by the last
    /// [`set_lights`](Self::set_lights) call (layers `0..count`).
    shadow_count: std::sync::atomic::AtomicU32,
    /// Directional-shadow fit: `(center, half-extent)` set via
    /// [`set_shadow_bounds`](Self::set_shadow_bounds) from the scene AABB.
    /// `None` (default) keeps the legacy ±[`SHADOW_ORTHO_HALF`] box around
    /// the origin, so scenes that never set bounds render pixel-identical.
    shadow_fit: std::sync::RwLock<Option<([f32; 3], f32)>>,
    /// Upload report of the last [`set_lights`](Self::set_lights) /
    /// [`set_lights_full`](Self::set_lights_full) call; read via
    /// [`light_upload_stats`](Self::light_upload_stats).
    last_light_stats: std::sync::RwLock<LightUploadStats>,
    /// Point-light shadow cubes: one depth cube per slot (6 faces),
    /// per-face views, sampling array view, per-face VP uniforms, and
    /// the active cube count. `params.w` on a point light indexes the
    /// cube slot (a separate index space from the 2D layers above —
    /// the evaluator picks the pool by light kind).
    shadow_cube_maps: wgpu::Texture,
    shadow_cube_views: [wgpu::TextureView; POINT_SHADOW_CUBES * 6],
    shadow_cube_array_view: wgpu::TextureView,
    shadow_cube_vp_buffers: [wgpu::Buffer; POINT_SHADOW_CUBES * 6],
    /// Mirrored depth-only pipeline for the cube faces (their VPs
    /// mirror NDC y — see [`point_cube_face_vp`] — so winding flips).
    shadow_cube_pipeline: wgpu::RenderPipeline,
    point_shadow_count: std::sync::atomic::AtomicU32,
}

/// Pick an up vector non-parallel to the given shadow axis.
fn shadow_up(axis: glam::Vec3) -> glam::Vec3 {
    if axis.y.abs() > 0.98 {
        glam::Vec3::X
    } else {
        glam::Vec3::Y
    }
}

/// Light-space clip matrix for a shadowed directional light: ortho box
/// ±[`SHADOW_ORTHO_HALF`] around the origin, eye on the light side.
/// Same `directx` depth convention as the main camera, so stored
/// depths compare directly in the evaluator.
fn dir_shadow_vp(to_light: glam::Vec3) -> [[f32; 4]; 4] {
    let view = glam::camera::rh::view::look_at_mat4(
        to_light * SHADOW_DIR_DIST,
        glam::Vec3::ZERO,
        shadow_up(to_light),
    );
    let proj = glam::camera::rh::proj::directx::orthographic(
        -SHADOW_ORTHO_HALF,
        SHADOW_ORTHO_HALF,
        -SHADOW_ORTHO_HALF,
        SHADOW_ORTHO_HALF,
        SHADOW_DIR_DIST - 20.0,
        SHADOW_DIR_DIST + 20.0,
    );
    (proj * view).to_cols_array_2d()
}

/// Scene fit for one directional shadow map: `(center, half-extent)` from
/// a scene AABB (`min`/`max` corners, e.g. over instance translations and
/// custom-geometry points). The half-extent never shrinks below
/// [`SHADOW_ORTHO_HALF`], so small scenes keep the legacy box exactly;
/// non-finite inputs fall back to the origin default instead of poisoning
/// the projection.
pub fn shadow_fit_for_bounds(min: [f32; 3], max: [f32; 3]) -> ([f32; 3], f32) {
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let half = ((max[0] - min[0]) * 0.5)
        .max((max[1] - min[1]) * 0.5)
        .max((max[2] - min[2]) * 0.5)
        .max(SHADOW_ORTHO_HALF);
    if center.iter().all(|v| v.is_finite()) && half.is_finite() {
        (center, half)
    } else {
        ([0.0, 0.0, 0.0], SHADOW_ORTHO_HALF)
    }
}

/// Light-space clip matrix for a shadowed directional light fitted to the
/// scene: ortho box ±`half` around `center`, eye at `half + 20` along the
/// to-light direction, depth range covering the box diameter plus margin.
/// Same `directx` depth convention as [`dir_shadow_vp`]; only used when a
/// scene fit was set via [`Renderer3D::set_shadow_bounds`].
fn dir_shadow_vp_fitted(to_light: glam::Vec3, center: [f32; 3], half: f32) -> [[f32; 4]; 4] {
    let dist = half + 20.0;
    let c = glam::Vec3::from_array(center);
    let view = glam::camera::rh::view::look_at_mat4(c + to_light * dist, c, shadow_up(to_light));
    let proj = glam::camera::rh::proj::directx::orthographic(
        -half,
        half,
        -half,
        half,
        1.0,
        dist + half + 10.0,
    );
    (proj * view).to_cols_array_2d()
}

/// Light-space clip matrix for a shadowed spotlight: perspective cone
/// (`2 × outer_angle`, aspect 1) from the light position along the
/// emission axis, far plane at the light range.
fn spot_shadow_vp(
    position: [f32; 3],
    axis: glam::Vec3,
    outer_angle_deg: f32,
    range: f32,
) -> [[f32; 4]; 4] {
    let eye = glam::Vec3::from_array(position);
    let view = glam::camera::rh::view::look_at_mat4(eye, eye + axis, shadow_up(axis));
    let proj = glam::camera::rh::proj::directx::perspective(
        outer_angle_deg.to_radians() * 2.0,
        1.0,
        0.5,
        range.max(1.0),
    );
    (proj * view).to_cols_array_2d()
}

/// Light-space clip matrix for one cube face of a shadowed point
/// light: 90° perspective (aspect 1) from the light position along
/// the face axis, far plane at the light range. Matches the analytic
/// sampling formula (`SHADOW_CUBE_NEAR`, `range.max(1.0)`).
///
/// The Y row of the projection is negated: rasterization maps NDC y+1
/// to texture row 0 (top), while the hardware cube-sampling frame
/// reads v=0 from the top with the GL axis convention (`CUBE_FACES`),
/// so an unmirrored render lands V-flipped versus the sampler (small
/// centered occluders vanish, large blobs only partly overlap). The
/// negation is a reflection — winding flips, so cube faces render
/// through the mirrored shadow pipeline (`front_face: Cw`).
fn point_cube_face_vp(position: [f32; 3], range: f32, face: usize) -> [[f32; 4]; 4] {
    let (dir, up) = CUBE_FACES[face % 6];
    let eye = glam::Vec3::from_array(position);
    let view = glam::camera::rh::view::look_at_mat4(
        eye,
        eye + glam::Vec3::from_array(dir),
        glam::Vec3::from_array(up),
    );
    let proj = glam::camera::rh::proj::directx::perspective(
        std::f32::consts::FRAC_PI_2,
        1.0,
        SHADOW_CUBE_NEAR,
        range.max(1.0),
    );
    let mut vp = (proj * view).to_cols_array_2d();
    // Mirror NDC y (whole row 1 — a single element would distort,
    // not reflect).
    for col in vp.iter_mut() {
        col[1] = -col[1];
    }
    vp
}

/// Upload one inline `MeshDesc::Custom` soup as its own [`Mesh`].
///
/// Per-entity path: each Custom entity owns its vertex/index buffers (the
/// shared sphere mesh never stands in for custom geometry). CPU-side
/// conversion is deduplicated upstream by the extraction's
/// [`crate::mesh_upload::SoupCache`] (identical soups convert once per
/// frame — see [`crate::mesh_upload::SoupHash`] and the per-soup budget
/// on [`crate::mesh_upload::custom_vertices`]), so this function only
/// moves the already-converted arrays into buffers created exactly like
/// [`crate::mesh::create_sphere`].
/// Edge direction stays one-way (`ornis-render` depends on
/// `ornis-mesh-editor`, never the reverse).
///
/// # Errors
///
/// Returns [`crate::mesh_upload::UploadError::EmptyMesh`] when either slice
/// is empty, [`crate::mesh_upload::UploadError::InvalidMesh`] when the soup
/// fails validation — callers skip the entity, never a stub mesh.
pub fn upload_custom_mesh(
    device: &wgpu::Device,
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Result<Mesh, crate::mesh_upload::UploadError> {
    let (vertices, indices) = crate::mesh_upload::custom_vertices(positions, indices)?;
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("custom mesh vertex buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("custom mesh index buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Ok(Mesh {
        vertex_buffer,
        index_buffer,
        num_indices: indices.len() as u32,
        vertex_count: vertices.len() as u32,
    })
}

/// Upload interleaved [`SkinnedVertex`] rows as their own [`Mesh`].
///
/// GPU-draw path for [`SkinningMode::Gpu`](ornis_animation::SkinningMode)
/// entries: the rows come from
/// [`CustomMeshEntry::skinned_gpu_vertices`](crate::extraction::CustomMeshEntry::skinned_gpu_vertices)
/// (bind-pose vertices plus staged influences) and draw through the
/// skinned pipeline, which reads locations 0–5 from this single buffer.
/// Buffers are created exactly like [`upload_custom_mesh`].
///
/// # Errors
///
/// Returns [`crate::mesh_upload::UploadError::EmptyMesh`] when either slice
/// is empty, [`crate::mesh_upload::UploadError::InvalidMesh`] when the
/// index list is not a multiple of three or points past the vertices —
/// callers skip the entry, never a stub mesh.
pub fn upload_skinned_mesh(
    device: &wgpu::Device,
    vertices: &[SkinnedVertex],
    indices: &[u32],
) -> Result<Mesh, crate::mesh_upload::UploadError> {
    use crate::mesh_upload::UploadError;
    use ornis_mesh_editor::MeshError;
    if vertices.is_empty() || indices.is_empty() {
        return Err(UploadError::EmptyMesh);
    }
    if !indices.len().is_multiple_of(3) {
        return Err(UploadError::InvalidMesh(
            MeshError::IndexCountNotMultipleOfThree,
        ));
    }
    if indices
        .iter()
        .any(|index| (*index as usize) >= vertices.len())
    {
        return Err(UploadError::InvalidMesh(MeshError::IndexOutOfBounds));
    }
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("skinned mesh vertex buffer"),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("skinned mesh index buffer"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Ok(Mesh {
        vertex_buffer,
        index_buffer,
        num_indices: indices.len() as u32,
        vertex_count: vertices.len() as u32,
    })
}

/// Identity clip matrix for lights that cast no shadow.
const NO_SHADOW_VP: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Packs a linear RGB scene color plus radiometric intensity into the GPU
/// `vec4` (RGB exposure-scaled, intensity in alpha).
fn pack_light_color(color: [f32; 3], intensity: f32, exposure: f32) -> [f32; 4] {
    let rgb = ornis_core::units::LinearRgb::new(color).as_array();
    [
        rgb[0] * exposure,
        rgb[1] * exposure,
        rgb[2] * exposure,
        intensity,
    ]
}

/// Pure result of [`build_lighting_uniform`]: the uploadable uniform, the
/// shadow VPs the depth pre-pass publishes (layers in assignment order,
/// cube faces in slot-major order), and the upload report.
struct BuiltLighting {
    uniform: LightingUniform,
    stats: LightUploadStats,
    shadow_layer_vps: Vec<[[f32; 4]; 4]>,
    cube_face_vps: Vec<[[f32; 4]; 4]>,
}

/// Pure [`LightingUniform`] construction shared by
/// [`Renderer3D::set_lights_full`] and [`count_light_drops`]: ambient RGB
/// scaled by `ambient_intensity`, light colors scaled by `exposure`
/// (CPU-baked IBL-minimum multipliers — no shader-layout change; 1.0 is
/// the exact no-op), the first [`MAX_LIGHTS`] entries uploaded, the rest
/// reported as dropped. Directional VPs use the legacy ±
/// [`SHADOW_ORTHO_HALF`] box when `fit` is `None` and the fitted box
/// otherwise. Shadow layer VPs are collected in assignment order (layer
/// `i` ↔ entry `i`).
fn build_lighting_uniform(
    ambient: [f32; 3],
    ambient_intensity: f32,
    exposure: f32,
    lights: &[LightDesc],
    fit: Option<([f32; 3], f32)>,
) -> BuiltLighting {
    /// Normalize a direction, falling back to +Z on degenerate input.
    fn norm_dir(d: [f32; 3]) -> [f32; 4] {
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len > 0.0 {
            [d[0] / len, d[1] / len, d[2] / len, 0.0]
        } else {
            [0.0, 0.0, 1.0, 0.0]
        }
    }
    /// Same as [`norm_dir`](norm_dir) as a [`glam::Vec3`].
    fn norm3(d: [f32; 3]) -> glam::Vec3 {
        let v = glam::Vec3::from_array(d);
        if v.length_squared() > 0.0 {
            v.normalize()
        } else {
            glam::Vec3::Z
        }
    }
    let count = lights.len().min(MAX_LIGHTS);
    let mut gpu_lights = [GpuLight {
        kind: [LIGHT_KIND_DIRECTIONAL, 0.0, 0.0, 0.0],
        direction: [0.0, 0.0, 1.0, 0.0],
        position: [0.0, 0.0, 0.0, 1.0],
        color: [0.0; 4],
        params: [0.0, 0.0, 0.0, -1.0],
        shadow_vp: NO_SHADOW_VP,
    }; 8];
    let mut shadow_count = 0u32;
    let mut cube_count = 0u32;
    let mut dropped_shadows = 0u32;
    let mut shadow_layer_vps: Vec<[[f32; 4]; 4]> = Vec::new();
    // (position, range, cube slot) for shadowed point lights, in
    // assignment order; face VPs are derived below.
    let mut cube_lights: Vec<([f32; 3], f32, usize)> = Vec::new();
    /// Assign the next shadow layer, or -1.0 when `wants` is false
    /// or the array is full. Returns `(layer, clip_matrix)`.
    macro_rules! shadow_layer {
        ($wants:expr, $vp:expr) => {{
            let wants: bool = $wants;
            if wants && (shadow_count as usize) < SHADOW_LAYERS {
                let layer = shadow_count;
                shadow_count += 1;
                (layer as f32, $vp)
            } else {
                (-1.0, NO_SHADOW_VP)
            }
        }};
    }
    for (i, light) in lights.iter().take(count).enumerate() {
        gpu_lights[i] = match light {
            LightDesc::Directional {
                direction,
                intensity,
                color,
                shadow,
            } => {
                let to_light = norm3(*direction);
                let vp = match fit {
                    None => dir_shadow_vp(to_light),
                    Some((center, half)) => dir_shadow_vp_fitted(to_light, center, half),
                };
                let (layer, vp) = shadow_layer!(shadow.is_enabled(), vp);
                if shadow.is_enabled() && layer < 0.0 {
                    dropped_shadows += 1;
                }
                if layer >= 0.0 {
                    shadow_layer_vps.push(vp);
                }
                GpuLight {
                    direction: norm_dir(*direction),
                    color: pack_light_color(*color, *intensity, exposure),
                    params: [0.0, 0.0, 0.0, layer],
                    shadow_vp: vp,
                    ..gpu_lights[i]
                }
            }
            LightDesc::Point {
                position,
                intensity,
                color,
                range,
                shadow,
            } => {
                // Cube slots live in a separate index space from the
                // 2D layers (the evaluator picks the pool by kind).
                let slot = if shadow.is_enabled() && (cube_count as usize) < POINT_SHADOW_CUBES {
                    let s = cube_count as usize;
                    cube_count += 1;
                    cube_lights.push((*position, *range, s));
                    s as f32
                } else {
                    -1.0
                };
                if shadow.is_enabled() && slot < 0.0 {
                    dropped_shadows += 1;
                }
                GpuLight {
                    kind: [LIGHT_KIND_POINT, 0.0, 0.0, 0.0],
                    position: [position[0], position[1], position[2], 1.0],
                    color: pack_light_color(*color, *intensity, exposure),
                    params: [range.max(1e-3), 0.0, 0.0, slot],
                    ..gpu_lights[i]
                }
            }
            LightDesc::Spot {
                position,
                direction,
                intensity,
                color,
                range,
                inner_angle,
                outer_angle,
                shadow,
            } => {
                // Cosineordered: inner must be the tighter cone.
                let ci = ornis_core::units::Degrees::new(*inner_angle)
                    .to_radians()
                    .get()
                    .cos();
                let co = ornis_core::units::Degrees::new(*outer_angle)
                    .to_radians()
                    .get()
                    .cos();
                let axis = norm3(*direction);
                let (layer, vp) = shadow_layer!(
                    shadow.is_enabled(),
                    spot_shadow_vp(*position, axis, *outer_angle, *range)
                );
                if shadow.is_enabled() && layer < 0.0 {
                    dropped_shadows += 1;
                }
                if layer >= 0.0 {
                    shadow_layer_vps.push(vp);
                }
                GpuLight {
                    kind: [LIGHT_KIND_SPOT, 0.0, 0.0, 0.0],
                    direction: norm_dir(*direction),
                    position: [position[0], position[1], position[2], 1.0],
                    color: pack_light_color(*color, *intensity, exposure),
                    params: [range.max(1e-3), ci.max(co), co.min(ci), layer],
                    shadow_vp: vp,
                }
            }
        };
    }
    // Excess lights beyond the upload window never reach the GPU; a
    // shadow request on one is a dropped shadow too.
    let mut dropped_lights = 0u32;
    for light in lights.iter().skip(count) {
        dropped_lights += 1;
        if light.shadow_cast().is_enabled() {
            dropped_shadows += 1;
        }
    }
    let mut cube_face_vps = Vec::with_capacity(cube_lights.len() * 6);
    for (position, range, _) in &cube_lights {
        for face in 0..6 {
            cube_face_vps.push(point_cube_face_vp(*position, *range, face));
        }
    }
    BuiltLighting {
        uniform: LightingUniform {
            ambient_color: [
                ambient[0] * ambient_intensity,
                ambient[1] * ambient_intensity,
                ambient[2] * ambient_intensity,
                1.0,
            ],
            lights: gpu_lights,
            light_count: count as u32,
            _pad: [0; 3],
        },
        stats: LightUploadStats {
            uploaded: count as u32,
            dropped_lights,
            dropped_shadows,
        },
        shadow_layer_vps,
        cube_face_vps,
    }
}

/// Pure preview of what [`Renderer3D::set_lights`] would drop for
/// `lights`: excess beyond [`MAX_LIGHTS`] plus shadow requests without a
/// free layer/cube slot. No GPU access — safe to call per frame; log on
/// scene change, not per frame.
pub fn count_light_drops(lights: &[LightDesc]) -> LightUploadStats {
    build_lighting_uniform([0.0; 3], 1.0, 1.0, lights, None).stats
}

/// Exact CPU staging capacity for one [`Renderer3D::upload_instances`]
/// call: the caller reserves this once up front, so the per-instance
/// conversion never reallocates mid-frame. Identity today (exact fit —
/// a repeated same-size frame reserves the same capacity, no growth);
/// kept as a named helper so the upload path reads as one reservation.
pub(crate) fn staging_capacity_for_instances(needed: usize) -> usize {
    needed
}

impl Renderer3D {
    /// Build every pipeline/target for `surface_config`'s format and extent.
    ///
    /// Capacity starts at 256 instances / 64 materials and grows on
    /// demand: [`upload_instances`](Self::upload_instances) and
    /// [`upload_materials`](Self::upload_materials) reallocate (and rebind)
    /// their buffers when a frame needs more, so scenes are never silently
    /// truncated. Zero-sized extents are clamped to 1 pixel.
    pub fn new(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
    ) -> Self {
        let max_objects = 256u32;
        let max_materials = 64u32;
        let format = surface_config.format;
        let width = surface_config.width.max(1);
        let height = surface_config.height.max(1);

        let buffers = Self::create_core_buffers(device, max_objects, max_materials);
        let (shadow_maps, shadow_views, shadow_array_view, shadow_vp_buffers, shadow_sampler) =
            Self::create_shadow_targets(device);
        let (shadow_cube_maps, shadow_cube_views, shadow_cube_array_view, shadow_cube_vp_buffers) =
            Self::create_shadow_cube_targets(device);
        let (bind_group_layout, bind_group) = Self::create_pbr_bind_group(
            device,
            &buffers,
            &shadow_array_view,
            &shadow_sampler,
            &shadow_cube_array_view,
        );
        let pipeline =
            Self::create_pbr_pipeline(device, surface_config, sample_count, &bind_group_layout);
        let (pbr_texture, pbr_texture_view) =
            Self::create_render_target(device, width, height, format, sample_count);

        let gbuffer = Self::create_gbuffer(device, width, height, sample_count);
        let (gbuffer_pipeline, gbuffer_bind_group_layout, gbuffer_bind_group) =
            Self::create_gbuffer_pipeline(
                device,
                &gbuffer,
                &buffers.camera,
                &buffers.per_object,
                &buffers.material,
                sample_count,
            );
        let shadow_pipeline =
            Self::create_shadow_pipeline(device, &gbuffer_bind_group_layout, false);
        let shadow_cube_pipeline =
            Self::create_shadow_pipeline(device, &gbuffer_bind_group_layout, true);
        let palette_buffer = Self::create_palette_buffer(device, 1);
        let (skinned_pipeline, skinned_bind_group_layout) = Self::create_skinned_pipeline(
            device,
            &buffers.camera,
            &buffers.per_object,
            &buffers.material,
            &palette_buffer,
            sample_count,
        );
        let skinned_shadow_pipeline =
            Self::create_skinned_shadow_pipeline(device, &skinned_bind_group_layout, false);
        let skinned_shadow_cube_pipeline =
            Self::create_skinned_shadow_pipeline(device, &skinned_bind_group_layout, true);
        let lighting_pass = Self::create_lighting_pass(device, &pbr_texture_view, sample_count);
        let forward_pass = Self::create_forward_pass(
            device,
            &buffers.camera,
            &buffers.per_object,
            &buffers.material,
            &buffers.lighting,
            &shadow_array_view,
            &shadow_sampler,
            &shadow_cube_array_view,
            width,
            height,
            sample_count,
            TransparencyOptions::default(),
        );
        let composite_pass = Self::create_composite_pass(device, format);
        let bloom_pass = Self::create_bloom_pass(device);
        let fog = Self::create_fog_pass(device, format);
        let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("composite sampler"),
            ..crate::flags::SamplerKind::LinearClamp.descriptor()
        });

        Self {
            camera_buffer: buffers.camera,
            per_object_buffer: std::sync::RwLock::new(buffers.per_object),
            material_buffer: std::sync::RwLock::new(buffers.material),
            lighting_buffer: buffers.lighting,
            _bind_group_layout: bind_group_layout,
            _bind_group: bind_group,
            _pipeline: pipeline,
            pbr_texture,
            pbr_texture_view,
            sample_count,
            exposure: 1.0,
            transparency: TransparencyOptions::default(),
            max_objects: std::sync::atomic::AtomicU32::new(max_objects),
            max_materials: std::sync::atomic::AtomicU32::new(max_materials),
            format,
            width,
            height,
            gbuffer,
            gbuffer_pipeline,
            gbuffer_bind_group_layout,
            gbuffer_bind_group: std::sync::RwLock::new(gbuffer_bind_group),
            skinned_pipeline,
            skinned_bind_group_layout,
            skinned_shadow_pipeline,
            skinned_shadow_cube_pipeline,
            palette_buffer: std::sync::RwLock::new(palette_buffer),
            max_palettes: std::sync::atomic::AtomicU32::new(1),
            palette_count: std::sync::atomic::AtomicU32::new(0),
            lighting_pass,
            forward_pass,
            textured_forward: None,
            composite_pass,
            composite_sampler,
            bloom_pass,
            fog,
            shadow_maps,
            shadow_views,
            shadow_array_view,
            shadow_vp_buffers,
            shadow_pipeline,
            shadow_sampler,
            shadow_count: std::sync::atomic::AtomicU32::new(0),
            shadow_fit: std::sync::RwLock::new(None),
            last_light_stats: std::sync::RwLock::new(LightUploadStats {
                uploaded: 0,
                dropped_lights: 0,
                dropped_shadows: 0,
            }),
            shadow_cube_maps,
            shadow_cube_views,
            shadow_cube_array_view,
            shadow_cube_vp_buffers,
            shadow_cube_pipeline,
            point_shadow_count: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Like [`new`](Self::new) with an explicit forward-layer
    /// transparency mode: rebuilds only the forward pipeline with
    /// [`forward_blend_state`] for `transparency`. The default
    /// ([`BlendMode::Opaque`]) builds the same `REPLACE` pipeline as
    /// [`new`](Self::new), so the default frame is unchanged. With
    /// [`BlendMode::Transparent`], submit instances back-to-front
    /// ([`sort_by_depth`] or [`crate::extraction::sort_by_depth`]).
    pub fn new_with_transparency(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
        transparency: TransparencyOptions,
    ) -> Self {
        let mut this = Self::new(device, surface_config, sample_count);
        this.transparency = transparency;
        {
            let per_object = this
                .per_object_buffer
                .read()
                .expect("per-object buffer lock");
            let material = this.material_buffer.read().expect("material buffer lock");
            this.forward_pass = Self::create_forward_pass(
                device,
                &this.camera_buffer,
                &per_object,
                &material,
                &this.lighting_buffer,
                &this.shadow_array_view,
                &this.shadow_sampler,
                &this.shadow_cube_array_view,
                this.width,
                this.height,
                this.sample_count,
                transparency,
            );
        }
        this
    }

    /// MSAA sample count this renderer was built with (see
    /// [`crate::render_backend::RenderBackendConfig::sample_count`]).
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Exposure multiplier applied by [`set_lights`](Self::set_lights);
    /// `1.0` (default) is the exact no-op. Set from
    /// [`crate::render_backend::RenderBackendConfig::exposure`].
    pub fn exposure(&self) -> f32 {
        self.exposure
    }

    /// Replace the exposure multiplier applied by future
    /// [`set_lights`](Self::set_lights) calls (default `1.0`).
    pub fn set_exposure(&mut self, exposure: f32) {
        self.exposure = exposure;
    }

    /// Transparency mode of the forward pipeline (see
    /// [`TransparencyOptions`]).
    pub fn transparency(&self) -> TransparencyOptions {
        self.transparency
    }

    fn create_core_buffers(
        device: &wgpu::Device,
        max_objects: u32,
        max_materials: u32,
    ) -> CoreBuffers {
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera buffer"),
            contents: bytemuck::bytes_of(&CameraUniform {
                view_proj: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                inv_view_proj: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                camera_pos: [0.0, 0.0, 0.0, 1.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let per_object_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("per-object buffer"),
            size: (std::mem::size_of::<PerObjectGpu>() * max_objects as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let material_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material buffer"),
            size: (OPENPBR_MATERIAL_SIZE * max_materials as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let default_lighting = LightingUniform {
            ambient_color: [0.03, 0.03, 0.05, 1.0],
            lights: [GpuLight {
                kind: [LIGHT_KIND_DIRECTIONAL, 0.0, 0.0, 0.0],
                direction: [0.0; 4],
                position: [0.0; 4],
                color: [0.0; 4],
                params: [0.0, 0.0, 0.0, -1.0],
                shadow_vp: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }; 8],
            light_count: 0,
            _pad: [0; 3],
        };
        let lighting_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lighting buffer"),
            contents: bytemuck::bytes_of(&default_lighting),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        CoreBuffers {
            camera: camera_buffer,
            per_object: per_object_buffer,
            material: material_buffer,
            lighting: lighting_buffer,
        }
    }

    fn create_pbr_bind_group(
        device: &wgpu::Device,
        buffers: &CoreBuffers,
        shadow_array_view: &wgpu::TextureView,
        shadow_sampler: &wgpu::Sampler,
        shadow_cube_array_view: &wgpu::TextureView,
    ) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        // Layout entries come from the pass resource table.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = shaders::pbr_generated::PBR_RESOURCES
            .iter()
            .map(|r| shaders::bgl_entry(r, false))
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbr bind group layout"),
            entries: &bgl_entries,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pbr bind group"),
            layout: &bind_group_layout,
            // Binding numbers come from the table; only the name →
            // buffer mapping lives here.
            entries: &shaders::bind_group_entries(&shaders::pbr_generated::PBR_RESOURCES, |r| {
                match r.name {
                    "camera" => buffers.camera.as_entire_binding(),
                    "per_objects" => buffers.per_object.as_entire_binding(),
                    "materials" => buffers.material.as_entire_binding(),
                    "lighting" => buffers.lighting.as_entire_binding(),
                    "shadow_tex" => wgpu::BindingResource::TextureView(shadow_array_view),
                    "shadow_sampler" => wgpu::BindingResource::Sampler(shadow_sampler),
                    "shadow_cube_tex" => wgpu::BindingResource::TextureView(shadow_cube_array_view),
                    other => panic!("pbr bind group has no resource for `{other}`"),
                }
            }),
        });

        (bind_group_layout, bind_group)
    }

    fn create_pbr_pipeline(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbr vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::pbr_vertex())),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbr fragment"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::pbr_fragment())),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbr pipeline layout"),
            bind_group_layouts: &[Some(bind_group_layout)],
            immediate_size: 0,
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pbr render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(shaders::gbuffer_generated::vs_main::entry_point()),
                buffers: &[Some(Vertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some(shaders::pbr_generated::fs_main::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }

    /// Allocate the joint-palette storage buffer: `slots` packed
    /// [`PALETTE_BYTE_SIZE`] slots, grown by
    /// [`upload_skin_palettes`](Self::upload_skin_palettes). Starts
    /// zeroed; unwritten slots are never indexed (joint indices are
    /// validated `< joint count` before staging).
    fn create_palette_buffer(device: &wgpu::Device, slots: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skin palette buffer"),
            size: (PALETTE_BYTE_SIZE as u64) * (slots.max(1) as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Skinned g-buffer pipeline: the palette-blend vertex stage
    /// ([`wgsl_vertex_source_skinned`]) with the classic fragment stage and
    /// the same five MRT targets. The bind-group layout carries the extra
    /// binding-3 palette (see [`GBUFFER_SKINNED_RESOURCES`]); bind groups
    /// are built per draw (the palette slot differs per entry), so only
    /// the pipeline and layout are retained.
    fn create_skinned_pipeline(
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        per_object_buffer: &wgpu::Buffer,
        material_buffer: &wgpu::Buffer,
        palette_buffer: &wgpu::Buffer,
        sample_count: u32,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
        // Layout entries come from the skinned pass resource table; the
        // probe bind group below pins the name → buffer mapping (a table
        // gain without an update panics loudly here, never at draw time).
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = GBUFFER_SKINNED_RESOURCES
            .iter()
            .map(|r| shaders::bgl_entry(r, sample_count > 1))
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skinned gbuffer bind group layout"),
            entries: &bgl_entries,
        });
        // Probe: every table row resolves to a live buffer (same mapping
        // the per-draw groups use).
        let _ = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinned gbuffer bind group (probe)"),
            layout: &bind_group_layout,
            entries: &shaders::bind_group_entries(&GBUFFER_SKINNED_RESOURCES, |r| match r.name {
                "camera" => camera_buffer.as_entire_binding(),
                "per_objects" => per_object_buffer.as_entire_binding(),
                "materials" => material_buffer.as_entire_binding(),
                "palette" => palette_buffer.as_entire_binding(),
                other => panic!("skinned bind group has no resource for `{other}`"),
            }),
        });

        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinned gbuffer vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(wgsl_vertex_source_skinned())),
        });
        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinned gbuffer fragment"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::gbuffer_fragment())),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinned gbuffer pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skinned gbuffer pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(skinned_entry_point()),
                buffers: &[Some(SkinnedVertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some(shaders::gbuffer_generated::fs_main::entry_point()),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    /// Depth-only skinned pipeline for the shadow pre-pass: the same
    /// palette-blend vertex stage over the skinned bind-group layout, no
    /// fragment stage (varyings are discarded, like the classic shadow
    /// pipeline).
    ///
    /// `mirror_y` selects the cube-face variant (`front_face: Cw`), same
    /// convention as [`create_shadow_pipeline`](Self::create_shadow_pipeline).
    fn create_skinned_shadow_pipeline(
        device: &wgpu::Device,
        skinned_bind_group_layout: &wgpu::BindGroupLayout,
        mirror_y: bool,
    ) -> wgpu::RenderPipeline {
        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinned shadow vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(wgsl_vertex_source_skinned())),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinned shadow pipeline layout"),
            bind_group_layouts: &[Some(skinned_bind_group_layout)],
            immediate_size: 0,
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skinned shadow pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(skinned_entry_point()),
                buffers: &[Some(SkinnedVertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            // Depth-only: no color targets, varyings are discarded.
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: if mirror_y {
                    wgpu::FrontFace::Cw
                } else {
                    wgpu::FrontFace::Ccw
                },
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: SHADOW_DEPTH_BIAS_CONSTANT,
                    slope_scale: SHADOW_DEPTH_BIAS_SLOPE,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }

    fn create_render_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let pbr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pbr render target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = pbr_texture.create_view(&wgpu::TextureViewDescriptor::default());
        (pbr_texture, view)
    }

    fn create_gbuffer(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> GBufferTextures {
        let albedo = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer albedo"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let albedo_view = albedo.create_view(&wgpu::TextureViewDescriptor::default());

        let normal = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer normal"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let normal_view = normal.create_view(&wgpu::TextureViewDescriptor::default());

        let material_id = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer material_id"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let material_id_view = material_id.create_view(&wgpu::TextureViewDescriptor::default());

        let world_position = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer world_position"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let world_position_view =
            world_position.create_view(&wgpu::TextureViewDescriptor::default());

        let material_params = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer material_params"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let material_params_view =
            material_params.create_view(&wgpu::TextureViewDescriptor::default());

        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

        GBufferTextures {
            albedo,
            albedo_view,
            normal,
            normal_view,
            material_id,
            material_id_view,
            world_position,
            world_position_view,
            material_params,
            material_params_view,
            depth,
            depth_view,
        }
    }

    fn create_gbuffer_pipeline(
        device: &wgpu::Device,
        _gbuffer: &GBufferTextures,
        camera_buffer: &wgpu::Buffer,
        per_object_buffer: &wgpu::Buffer,
        material_buffer: &wgpu::Buffer,
        sample_count: u32,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::BindGroup) {
        // Layout entries come from the pass resource table.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> =
            shaders::gbuffer_generated::GBUFFER_RESOURCES
                .iter()
                .map(|r| shaders::bgl_entry(r, sample_count > 1))
                .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gbuffer bind group layout"),
            entries: &bgl_entries,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gbuffer bind group"),
            layout: &bind_group_layout,
            entries: &shaders::bind_group_entries(
                &shaders::gbuffer_generated::GBUFFER_RESOURCES,
                |r| match r.name {
                    "camera" => camera_buffer.as_entire_binding(),
                    "per_objects" => per_object_buffer.as_entire_binding(),
                    "materials" => material_buffer.as_entire_binding(),
                    other => panic!("gbuffer bind group has no resource for `{other}`"),
                },
            ),
        });

        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gbuffer vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::gbuffer_vertex())),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gbuffer fragment"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::gbuffer_fragment())),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gbuffer pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gbuffer pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(shaders::gbuffer_generated::vs_main::entry_point()),
                buffers: &[Some(Vertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some(shaders::gbuffer_generated::fs_main::entry_point()),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        (pipeline, bind_group_layout, bind_group)
    }

    /// Allocate the shadow-map array, per-layer views, the sampling
    /// array view, VP uniform buffers and the comparison sampler.
    #[allow(clippy::type_complexity)]
    fn create_shadow_targets(
        device: &wgpu::Device,
    ) -> (
        wgpu::Texture,
        [wgpu::TextureView; SHADOW_LAYERS],
        wgpu::TextureView,
        [wgpu::Buffer; SHADOW_LAYERS],
        wgpu::Sampler,
    ) {
        let shadow_maps = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow maps"),
            size: wgpu::Extent3d {
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
                depth_or_array_layers: SHADOW_LAYERS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let shadow_views = std::array::from_fn(|layer| {
            shadow_maps.create_view(&wgpu::TextureViewDescriptor {
                label: Some("shadow map layer"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });
        let shadow_array_view = shadow_maps.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow map array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let shadow_vp_buffers = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shadow VP buffer"),
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow comparison sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..crate::flags::SamplerKind::LinearClamp.descriptor()
        });
        (
            shadow_maps,
            shadow_views,
            shadow_array_view,
            shadow_vp_buffers,
            shadow_sampler,
        )
    }

    /// Depth-only pipeline for the shadow pre-pass. It reuses the
    /// gbuffer vertex shader (and its bind group layout): each layer
    /// binds its light-space VP buffer into the `camera` slot, so no
    /// new shader or layout is needed. A depth bias (constant + slope)
    /// fights acne; the evaluator adds a small reference bias on top.
    ///
    /// `mirror_y` selects the cube-face variant (`front_face: Cw`):
    /// cube VPs mirror NDC y (see [`point_cube_face_vp`]), which flips
    /// winding, so faces must cull the mirrored side.
    fn create_shadow_pipeline(
        device: &wgpu::Device,
        gbuffer_bind_group_layout: &wgpu::BindGroupLayout,
        mirror_y: bool,
    ) -> wgpu::RenderPipeline {
        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::gbuffer_vertex())),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow pipeline layout"),
            bind_group_layouts: &[Some(gbuffer_bind_group_layout)],
            immediate_size: 0,
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(shaders::gbuffer_generated::vs_main::entry_point()),
                buffers: &[Some(Vertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            // Depth-only: no color targets, varyings are discarded.
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: if mirror_y {
                    wgpu::FrontFace::Cw
                } else {
                    wgpu::FrontFace::Ccw
                },
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: SHADOW_DEPTH_BIAS_CONSTANT,
                    slope_scale: SHADOW_DEPTH_BIAS_SLOPE,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }

    /// Allocate the point-shadow cube array, per-face views, the
    /// sampling cube-array view and per-face VP uniform buffers.
    /// Rendering reuses the depth-only shadow pipeline (same gbuffer
    /// vertex layout); only the bound VP buffer and target view change
    /// per face.
    #[allow(clippy::type_complexity)]
    fn create_shadow_cube_targets(
        device: &wgpu::Device,
    ) -> (
        wgpu::Texture,
        [wgpu::TextureView; POINT_SHADOW_CUBES * 6],
        wgpu::TextureView,
        [wgpu::Buffer; POINT_SHADOW_CUBES * 6],
    ) {
        let shadow_cube_maps = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("point shadow cubes"),
            size: wgpu::Extent3d {
                width: SHADOW_CUBE_SIZE,
                height: SHADOW_CUBE_SIZE,
                depth_or_array_layers: (POINT_SHADOW_CUBES * 6) as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let shadow_cube_views = std::array::from_fn(|layer| {
            shadow_cube_maps.create_view(&wgpu::TextureViewDescriptor {
                label: Some("point shadow cube face"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });
        let shadow_cube_array_view = shadow_cube_maps.create_view(&wgpu::TextureViewDescriptor {
            label: Some("point shadow cube array"),
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        });
        let shadow_cube_vp_buffers = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("point shadow face VP buffer"),
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        (
            shadow_cube_maps,
            shadow_cube_views,
            shadow_cube_array_view,
            shadow_cube_vp_buffers,
        )
    }

    /// Render depth pre-passes for the `0..shadow_count` layers assigned
    /// by [`set_lights`](Self::set_lights); a no-op without shadowed
    /// lights. Runs before lighting/forward in every frame path
    /// (legacy `render_scene` and the `LightingPass` plan pass call it
    /// explicitly).
    pub fn render_shadows(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &Mesh,
        instance_count: u32,
    ) {
        if instance_count == 0 {
            return;
        }
        let count = self
            .shadow_count
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(SHADOW_LAYERS as u32);
        let cubes = self
            .point_shadow_count
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(POINT_SHADOW_CUBES as u32);
        if count == 0 && cubes == 0 {
            return;
        }
        let per_object = self.per_object_buffer.read().unwrap();
        let material = self.material_buffer.read().unwrap();
        for layer in 0..count as usize {
            self.render_shadow_layer(
                device,
                encoder,
                mesh,
                instance_count,
                &self.shadow_pipeline,
                &self.shadow_vp_buffers[layer],
                &self.shadow_views[layer],
                &per_object,
                &material,
            );
        }
        for cube in 0..cubes as usize {
            for face in 0..6 {
                let idx = cube * 6 + face;
                self.render_shadow_layer(
                    device,
                    encoder,
                    mesh,
                    instance_count,
                    &self.shadow_cube_pipeline,
                    &self.shadow_cube_vp_buffers[idx],
                    &self.shadow_cube_views[idx],
                    &per_object,
                    &material,
                );
            }
        }
    }

    /// One depth-only draw into a shadow view with the given light-space
    /// VP bound into the gbuffer `camera` slot. Shared by 2D layers and
    /// cube faces (same layout, mirrored pipeline for faces).
    #[allow(clippy::too_many_arguments)]
    fn render_shadow_layer(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &Mesh,
        instance_count: u32,
        pipeline: &wgpu::RenderPipeline,
        vp_buffer: &wgpu::Buffer,
        view: &wgpu::TextureView,
        per_object: &wgpu::Buffer,
        material: &wgpu::Buffer,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow bind group"),
            layout: &self.gbuffer_bind_group_layout,
            entries: &shaders::bind_group_entries(
                &shaders::gbuffer_generated::GBUFFER_RESOURCES,
                |r| match r.name {
                    "camera" => vp_buffer.as_entire_binding(),
                    "per_objects" => per_object.as_entire_binding(),
                    "materials" => material.as_entire_binding(),
                    other => panic!("shadow bind group has no resource for `{other}`"),
                },
            ),
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.num_indices, 0, 0..instance_count);
    }

    /// Render skinned depth pre-passes for one skinned entry: the same
    /// layers and cube faces [`render_shadows`](Self::render_shadows)
    /// covers, drawn with the skinned depth pipelines and the entry's
    /// palette slot — skinned depth, never bind-pose depth.
    ///
    /// The entry instance must already sit in per-object slot 0 (via
    /// [`upload_instances`](Self::upload_instances) or
    /// [`render_skinned_entry`](Self::render_skinned_entry)); a stale
    /// handle records no commands. A no-op without shadowed lights.
    pub fn render_skinned_shadows(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &Mesh,
        handle: PaletteHandle,
    ) {
        if handle.index()
            >= self
                .palette_count
                .load(std::sync::atomic::Ordering::Relaxed) as usize
        {
            return;
        }
        let count = self
            .shadow_count
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(SHADOW_LAYERS as u32);
        let cubes = self
            .point_shadow_count
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(POINT_SHADOW_CUBES as u32);
        if count == 0 && cubes == 0 {
            return;
        }
        let per_object = self.per_object_buffer.read().unwrap();
        let material = self.material_buffer.read().unwrap();
        let palette = self.palette_buffer.read().expect("palette buffer lock");
        for layer in 0..count as usize {
            self.render_skinned_shadow_layer(
                device,
                encoder,
                mesh,
                handle,
                &self.skinned_shadow_pipeline,
                &self.shadow_vp_buffers[layer],
                &self.shadow_views[layer],
                &per_object,
                &material,
                &palette,
            );
        }
        for cube in 0..cubes as usize {
            for face in 0..6 {
                let idx = cube * 6 + face;
                self.render_skinned_shadow_layer(
                    device,
                    encoder,
                    mesh,
                    handle,
                    &self.skinned_shadow_cube_pipeline,
                    &self.shadow_cube_vp_buffers[idx],
                    &self.shadow_cube_views[idx],
                    &per_object,
                    &material,
                    &palette,
                );
            }
        }
    }

    /// One skinned depth-only draw into a shadow view: the light-space VP
    /// goes into the `camera` slot and the entry palette into binding 3.
    /// Shared by 2D layers and cube faces (same layout, mirrored pipeline
    /// for faces).
    #[allow(clippy::too_many_arguments)]
    fn render_skinned_shadow_layer(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &Mesh,
        handle: PaletteHandle,
        pipeline: &wgpu::RenderPipeline,
        vp_buffer: &wgpu::Buffer,
        view: &wgpu::TextureView,
        per_object: &wgpu::Buffer,
        material: &wgpu::Buffer,
        palette: &wgpu::Buffer,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinned shadow bind group"),
            layout: &self.skinned_bind_group_layout,
            entries: &shaders::bind_group_entries(&GBUFFER_SKINNED_RESOURCES, |r| match r.name {
                "camera" => vp_buffer.as_entire_binding(),
                "per_objects" => per_object.as_entire_binding(),
                "materials" => material.as_entire_binding(),
                "palette" => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: palette,
                    offset: handle.byte_offset(),
                    size: std::num::NonZeroU64::new(PALETTE_BYTE_SIZE as u64),
                }),
                other => panic!("skinned shadow bind group has no resource for `{other}`"),
            }),
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("skinned shadow pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
    }

    fn create_lighting_pass(
        device: &wgpu::Device,
        output_view: &wgpu::TextureView,
        sample_count: u32,
    ) -> LightingPass {
        // Layout entries come from the pass resource table — the same table
        // that generates the WGSL declarations, so shader and layout agree.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> =
            shaders::lighting_generated::LIGHTING_RESOURCES
                .iter()
                .map(|r| shaders::bgl_entry(r, sample_count > 1))
                .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lighting bind group layout"),
            entries: &bgl_entries,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lighting sampler"),
            ..crate::flags::SamplerKind::LinearClamp.descriptor()
        });

        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lighting vertex (generated)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(
                shaders::lighting_generated::wgsl_vertex_source(),
            )),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lighting fragment (generated)"),
            source: wgpu::ShaderSource::Wgsl(
                Cow::Owned(shaders::lighting_generated::wgsl_source()),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lighting pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lighting pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(shaders::lighting_generated::vs_main::entry_point()),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some(shaders::lighting_generated::fs_main::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_view.texture().format(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        LightingPass {
            pipeline,
            bind_group_layout,
            sampler,
        }
    }

    // Internal pass constructor: 4 buffers + surface parameters —
    // grouping them into a struct would not improve call readability.
    #[allow(clippy::too_many_arguments)]
    fn create_forward_pass(
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        per_object_buffer: &wgpu::Buffer,
        material_buffer: &wgpu::Buffer,
        lighting_buffer: &wgpu::Buffer,
        shadow_array_view: &wgpu::TextureView,
        shadow_sampler: &wgpu::Sampler,
        shadow_cube_array_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        sample_count: u32,
        transparency: TransparencyOptions,
    ) -> ForwardPass {
        // Same layout as the PBR bind group: entries from the shared table.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = shaders::pbr_generated::PBR_RESOURCES
            .iter()
            .map(|r| shaders::bgl_entry(r, false))
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forward bind group layout"),
            entries: &bgl_entries,
        });

        let bind_group =
            std::sync::RwLock::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("forward bind group"),
                layout: &bind_group_layout,
                entries: &shaders::bind_group_entries(
                    &shaders::pbr_generated::PBR_RESOURCES,
                    |r| match r.name {
                        "camera" => camera_buffer.as_entire_binding(),
                        "per_objects" => per_object_buffer.as_entire_binding(),
                        "materials" => material_buffer.as_entire_binding(),
                        "lighting" => lighting_buffer.as_entire_binding(),
                        "shadow_tex" => wgpu::BindingResource::TextureView(shadow_array_view),
                        "shadow_sampler" => wgpu::BindingResource::Sampler(shadow_sampler),
                        "shadow_cube_tex" => {
                            wgpu::BindingResource::TextureView(shadow_cube_array_view)
                        }
                        other => panic!("forward bind group has no resource for `{other}`"),
                    },
                ),
            }));

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("forward color target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::pbr_vertex())),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward fragment"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::pbr_fragment())),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forward pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("forward pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(shaders::gbuffer_generated::vs_main::entry_point()),
                buffers: &[Some(Vertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some(shaders::pbr_generated::fs_main::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(forward_blend_state(transparency)),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        ForwardPass {
            pipeline,
            bind_group_layout,
            bind_group,
            _color_texture: color_texture,
            color_view,
        }
    }

    /// Resolves the view sampled for `role`: the cache entry bound in
    /// `textures`, or the neutral fallback (color roles share
    /// `fallback_color`, the data role uses `fallback_data`) when the slot
    /// is unbound or the handle is stale.
    fn material_view<'a>(
        textures: &MaterialTextureSet,
        cache: &'a TextureCache,
        fallback_color: &'a wgpu::TextureView,
        fallback_data: &'a wgpu::TextureView,
        role: TextureRole,
    ) -> &'a wgpu::TextureView {
        textures
            .binding(role)
            .and_then(|handle| cache.get(handle))
            .map_or(
                match role {
                    TextureRole::MetallicRoughness => fallback_data,
                    TextureRole::BaseColor | TextureRole::Emissive => fallback_color,
                },
                |texture| &texture.view,
            )
    }

    /// Build the textured-forward pipeline and its neutral fallback
    /// textures; idempotent (later calls are no-ops).
    ///
    /// The fallbacks are 1x1 white images uploaded through
    /// [`upload_texture`](crate::textures::upload_texture) under their
    /// roles (sRGB for color, linear for data), and the sampler carries
    /// [`sampler_descriptor_for_role`](crate::textures::sampler_descriptor_for_role)
    /// defaults. The fragment stage is
    /// [`wgsl_source_textured`](crate::shaders::material_textures::wgsl_source_textured);
    /// everything else (vertex entry, targets, depth, MSAA) mirrors
    /// [`TransparencyOptions`]-aware [`ForwardPass`] construction, so only
    /// textured draws change appearance.
    pub fn ensure_textured_forward(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.textured_forward.is_some() {
            return;
        }
        let white = CpuImage::from_rgba8(1, 1, vec![255; 4]).expect("1x1 white image validates");
        let fallback_color = upload_texture(device, queue, &white, TextureRole::BaseColor)
            .expect("1x1 fallback color texture uploads");
        let fallback_data = upload_texture(device, queue, &white, TextureRole::MetallicRoughness)
            .expect("1x1 fallback data texture uploads");
        let sampler = device.create_sampler(&sampler_descriptor_for_role(TextureRole::BaseColor));

        // Same table-driven layout as every other pass: entries from the
        // textured resource table, so WGSL declarations and layout agree.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> =
            shaders::material_textures::TEXTURED_PBR_RESOURCES
                .iter()
                .map(|r| shaders::bgl_entry(r, false))
                .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("textured forward bind group layout"),
            entries: &bgl_entries,
        });

        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("textured forward vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::pbr_vertex())),
        });
        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("textured forward fragment"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(
                shaders::material_textures::wgsl_source_textured(),
            )),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("textured forward pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("textured forward pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(shaders::gbuffer_generated::vs_main::entry_point()),
                buffers: &[Some(Vertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some(shaders::material_textures::fs_main_textured::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(forward_blend_state(self.transparency)),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: self.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        self.textured_forward = Some(TexturedForwardPass {
            pipeline,
            bind_group_layout,
            fallback_color,
            fallback_data,
            sampler,
        });
    }

    fn create_composite_pass(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> CompositePass {
        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::composite_vertex())),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite fragment"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::composite_fragment())),
        });

        // The composite pass runs the HDR shaders: entries from that table.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> =
            shaders::hdr_composite_generated::HDR_RESOURCES
                .iter()
                .map(|r| shaders::bgl_entry(r, false))
                .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite bind group layout"),
            entries: &bgl_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(shaders::hdr_composite_generated::vs_main::entry_point()),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some(shaders::hdr_composite_generated::fs_main::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        CompositePass {
            pipeline,
            bind_group_layout,
        }
    }

    fn create_bloom_pass(device: &wgpu::Device) -> BloomPass {
        // Bloom WGSL is now generated from Rust (path 2) — single
        // source of truth `shaders::bloom_generated::wgsl_source()`.
        let bloom_source = shaders::bloom_generated::wgsl_source();
        let bloom_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom shader (generated)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(bloom_source)),
        });
        // Vertex and fragment are one module with two entry points `vs_main`/`fs_main`.
        // Two variables point to the same module to preserve the signature
        // `bloom_pipeline(vertex, fragment, ...)`.
        let fs_module = &bloom_module;
        let vs_module = &bloom_module;

        // Layout entries come from the pass resource table.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> =
            shaders::bloom_generated::BLOOM_RESOURCES
                .iter()
                .map(|r| shaders::bgl_entry(r, false))
                .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom bind group layout"),
            entries: &bgl_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bloom params buffer"),
            contents: bytemuck::bytes_of(&BloomUniform::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Downsample: replace-blend, target is cleared first.
        let down_pipeline =
            Self::bloom_pipeline(device, &pipeline_layout, fs_module, vs_module, None);
        // Upsample: additive blend over the loaded previous level.
        let up_pipeline = Self::bloom_pipeline(
            device,
            &pipeline_layout,
            fs_module,
            vs_module,
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
        );

        BloomPass {
            down_pipeline,
            up_pipeline,
            bind_group_layout,
            params_buffer,
        }
    }

    fn bloom_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        fs_module: &wgpu::ShaderModule,
        vs_module: &wgpu::ShaderModule,
        blend: Option<wgpu::BlendState>,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: vs_module,
                entry_point: Some(shaders::bloom_generated::vs_main::entry_point()),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: fs_module,
                entry_point: Some(shaders::bloom_generated::fs_main::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }

    /// Builds the opt-in distance-fog fullscreen pass for `surface_format`.
    ///
    /// The shader is generated from Rust
    /// ([`crate::shaders::fog_generated`], `#[stage]` entries — no
    /// handwritten WGSL); the layout comes from its `FOG_RESOURCES` table.
    /// The params buffer starts at a benign enabled-neutral value (black,
    /// zero density is never drawn — [`render_fog`](Self::render_fog)
    /// returns early on non-positive densities, so the disabled pass is an
    /// exact no-op).
    fn create_fog_pass(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> FogPipeline {
        let fog_source = shaders::fog_generated::wgsl_source();
        let fog_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fog shader (generated)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(fog_source)),
        });
        let vertex_source = shaders::fog_generated::wgsl_vertex_source();
        let vertex_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fog vertex (generated)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(vertex_source)),
        });

        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = shaders::fog_generated::FOG_RESOURCES
            .iter()
            .map(|r| shaders::bgl_entry(r, false))
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fog bind group layout"),
            entries: &bgl_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fog pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fog params buffer"),
            contents: bytemuck::bytes_of(&FogUniform::pack([0.0; 3], 1.0)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fog pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_module,
                entry_point: Some(shaders::fog_generated::vs_main::entry_point()),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fog_module,
                entry_point: Some(shaders::fog_generated::fs_main::entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        FogPipeline {
            pipeline,
            bind_group_layout,
            params_buffer,
        }
    }

    /// Reallocate all size-dependent textures and re-record dependent
    /// pipelines after the output extent changed. Extents are clamped to >= 1.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        self.width = width;
        self.height = height;

        self.pbr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pbr render target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: self.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.pbr_texture_view = self
            .pbr_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.gbuffer = Self::create_gbuffer(device, width, height, self.sample_count);
        let (gbuffer_pipeline, gbuffer_bind_group_layout, gbuffer_bind_group) =
            Self::create_gbuffer_pipeline(
                device,
                &self.gbuffer,
                &self.camera_buffer,
                &self
                    .per_object_buffer
                    .read()
                    .expect("per-object buffer lock"),
                &self.material_buffer.read().expect("material buffer lock"),
                self.sample_count,
            );
        self.gbuffer_pipeline = gbuffer_pipeline;
        self.gbuffer_bind_group_layout = gbuffer_bind_group_layout;
        *self
            .gbuffer_bind_group
            .write()
            .expect("gbuffer bind group lock") = gbuffer_bind_group;

        self.lighting_pass =
            Self::create_lighting_pass(device, &self.pbr_texture_view, self.sample_count);

        self.forward_pass = Self::create_forward_pass(
            device,
            &self.camera_buffer,
            &self
                .per_object_buffer
                .read()
                .expect("per-object buffer lock"),
            &self.material_buffer.read().expect("material buffer lock"),
            &self.lighting_buffer,
            &self.shadow_array_view,
            &self.shadow_sampler,
            &self.shadow_cube_array_view,
            width,
            height,
            self.sample_count,
            self.transparency,
        );

        self.composite_pass = Self::create_composite_pass(device, self.format);
    }

    /// View of the final lit HDR image produced by the legacy-path lighting pass.
    pub fn pbr_view(&self) -> &wgpu::TextureView {
        &self.pbr_texture_view
    }

    /// Bytes allocated by the persistent textures of the legacy path
    /// (5 g-buffer MRTs + g-buffer depth + lighting target + forward color
    /// + shadow-map array + point-shadow cubes).
    pub fn texture_budget(&self) -> u64 {
        let bpp = crate::transient_pool::format_bytes_per_pixel;
        let w = self.width as u64;
        let h = self.height as u64;
        let s = self.sample_count as u64;
        let gbuffer = (bpp(wgpu::TextureFormat::Rgba8Unorm)
            + bpp(wgpu::TextureFormat::Rg16Float)
            + bpp(wgpu::TextureFormat::R32Uint)
            + bpp(wgpu::TextureFormat::Rg16Float)
            + bpp(wgpu::TextureFormat::Rgba16Float)
            + bpp(wgpu::TextureFormat::Depth32Float)) as u64
            * w
            * h
            * s;
        let pbr = bpp(self.format) as u64 * w * h * s;
        let forward = bpp(wgpu::TextureFormat::Rgba16Float) as u64 * w * h * s;
        let shadow_size = self.shadow_maps.size();
        let shadow = bpp(wgpu::TextureFormat::Depth32Float) as u64
            * shadow_size.width as u64
            * shadow_size.height as u64
            * shadow_size.depth_or_array_layers as u64;
        let cube_size = self.shadow_cube_maps.size();
        let cubes = bpp(wgpu::TextureFormat::Depth32Float) as u64
            * cube_size.width as u64
            * cube_size.height as u64
            * cube_size.depth_or_array_layers as u64;
        gbuffer + pbr + forward + shadow + cubes
    }

    /// Upload the camera uniform: view-projection, its inverse (computed here)
    /// and eye position. Call once per frame before rendering.
    /// Test-only readback of one 2D shadow layer as row-major depths
    /// (row 0 first): regression pin for the raster↔sampler V convention.
    #[cfg(test)]
    pub(crate) fn read_shadow_layer_for_tests(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layer: u32,
    ) -> Vec<f32> {
        let size = SHADOW_SIZE;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test shadow layer readback"),
            size: (size * size * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test shadow layer readback"),
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.shadow_maps,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size * 4),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = slice.get_mapped_range().expect("range");
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buf.unmap();
        out
    }
    /// Upload the camera uniform: view-projection, its inverse (computed here)
    /// and eye position. Call once per frame before rendering.
    pub fn set_camera(&self, queue: &wgpu::Queue, view_proj: &[[f32; 4]; 4], camera_pos: [f32; 3]) {
        let inv_view_proj = glam::Mat4::from_cols_array_2d(view_proj)
            .inverse()
            .to_cols_array_2d();
        let uniform = CameraUniform {
            view_proj: *view_proj,
            inv_view_proj,
            camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Upload ambient RGB plus up to eight scene lights of any kind
    /// ([`MAX_LIGHTS`]); excess lights are dropped and the drop is
    /// published via [`light_upload_stats`](Self::light_upload_stats),
    /// never silently. Directionals map exactly as before, so
    /// directional-only scenes render pixel-identical to the legacy rig.
    /// Identical to [`set_lights_full`](Self::set_lights_full) with
    /// `ambient_intensity = 1.0` and `exposure = [`exposure`](Self::exposure)
    /// (`1.0` by default — the exact no-op).
    ///
    /// Shadowed lights (directional/spot with `shadow: ShadowCast::Enabled`) are
    /// assigned map layers `0..shadow_count` (`params.w`); their
    /// light-space clip matrices go both into [`GpuLight::shadow_vp`]
    /// (sampled by the evaluators) and into the per-layer VP uniform
    /// buffers the depth pre-pass reuses through the gbuffer vertex
    /// shader's `camera` slot. Directional VPs use the legacy
    /// ±[`SHADOW_ORTHO_HALF`] box around the origin unless a scene fit
    /// was set via [`set_shadow_bounds`](Self::set_shadow_bounds).
    pub fn set_lights(&self, queue: &wgpu::Queue, ambient: [f32; 3], lights: &[LightDesc]) {
        self.set_lights_full(queue, ambient, 1.0, self.exposure, lights);
    }

    /// [`set_lights`](Self::set_lights) with IBL-minimum multipliers and
    /// an explicit upload report. `ambient_intensity` scales the ambient
    /// RGB and `exposure` scales every light color; both are baked on the
    /// CPU (no shader-layout change) and both are the exact no-op at
    /// `1.0`. Returns the [`LightUploadStats`] for this upload and
    /// publishes it for [`light_upload_stats`](Self::light_upload_stats).
    pub fn set_lights_full(
        &self,
        queue: &wgpu::Queue,
        ambient: [f32; 3],
        ambient_intensity: f32,
        exposure: f32,
        lights: &[LightDesc],
    ) -> LightUploadStats {
        let fit = *self.shadow_fit.read().expect("shadow fit lock");
        let built = build_lighting_uniform(ambient, ambient_intensity, exposure, lights, fit);
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&built.uniform));
        // Publish the light-space VPs for the depth pre-pass (as camera
        // uniforms: the shadow pipeline reuses the gbuffer vertex shader,
        // which only reads `view_proj`) and the active layer count.
        // Layers publish in assignment order (layer `i` ↔ entry `i`).
        for (layer, vp) in built.shadow_layer_vps.iter().enumerate() {
            let uniform = CameraUniform {
                view_proj: *vp,
                inv_view_proj: NO_SHADOW_VP,
                camera_pos: [0.0, 0.0, 0.0, 1.0],
            };
            queue.write_buffer(
                &self.shadow_vp_buffers[layer],
                0,
                bytemuck::bytes_of(&uniform),
            );
        }
        self.shadow_count.store(
            built.shadow_layer_vps.len() as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        // Publish the cube-face VPs (as camera uniforms, like the 2D
        // layers) and the active cube count.
        for (index, vp) in built.cube_face_vps.iter().enumerate() {
            let uniform = CameraUniform {
                view_proj: *vp,
                inv_view_proj: NO_SHADOW_VP,
                camera_pos: [0.0, 0.0, 0.0, 1.0],
            };
            queue.write_buffer(
                &self.shadow_cube_vp_buffers[index],
                0,
                bytemuck::bytes_of(&uniform),
            );
        }
        self.point_shadow_count.store(
            (built.cube_face_vps.len() / 6) as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        *self.last_light_stats.write().expect("light stats lock") = built.stats;
        built.stats
    }

    /// Upload report of the last [`set_lights`](Self::set_lights) /
    /// [`set_lights_full`](Self::set_lights_full) call: what was uploaded
    /// and what was dropped (excess lights, shadow requests without a
    /// slot). Starts at zero before the first upload.
    pub fn light_upload_stats(&self) -> LightUploadStats {
        *self.last_light_stats.read().expect("light stats lock")
    }

    /// Fit the directional shadow frustum to a scene AABB (`min`/`max`
    /// corners over instance translations and custom-geometry points).
    /// The ortho half-extent grows from the ±[`SHADOW_ORTHO_HALF`]
    /// default to cover the box (see [`shadow_fit_for_bounds`]); pass the
    /// scene bounds once per scene, not per frame.
    pub fn set_shadow_bounds(&self, min: [f32; 3], max: [f32; 3]) {
        *self.shadow_fit.write().expect("shadow fit lock") = Some(shadow_fit_for_bounds(min, max));
    }

    /// Drop the scene fit and return to the legacy ±[`SHADOW_ORTHO_HALF`]
    /// box around the origin.
    pub fn clear_shadow_bounds(&self) {
        *self.shadow_fit.write().expect("shadow fit lock") = None;
    }

    /// Current directional-shadow ortho half-extent: the fitted value
    /// after [`set_shadow_bounds`](Self::set_shadow_bounds), else the
    /// ±[`SHADOW_ORTHO_HALF`] default.
    pub fn shadow_half_extent(&self) -> f32 {
        self.shadow_fit
            .read()
            .expect("shadow fit lock")
            .map_or(SHADOW_ORTHO_HALF, |(_, half)| half)
    }

    /// Grow a storage buffer when `needed` exceeds `capacity`, doubling
    /// until it fits (amortized O(1) across frames).
    fn grown_storage_buffer(
        device: &wgpu::Device,
        label: &str,
        old: &wgpu::Buffer,
        element_bytes: usize,
        needed: usize,
        capacity: &std::sync::atomic::AtomicU32,
    ) -> Option<wgpu::Buffer> {
        if needed <= capacity.load(std::sync::atomic::Ordering::Relaxed) as usize {
            return None;
        }
        let mut grown = capacity.load(std::sync::atomic::Ordering::Relaxed).max(1);
        while (grown as usize) < needed {
            grown = grown.saturating_mul(2);
        }
        capacity.store(grown, std::sync::atomic::Ordering::Relaxed);
        // Destroying the buffer under a live bind group would fault the
        // next submit; wgpu defers actual destruction until the GPU is
        // done, so replace-then-rebind inside the same frame is safe.
        old.destroy();
        Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (element_bytes * grown as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    /// Rebind every pass that reads the grown storage buffers: each pass
    /// holds its own bind group over the same layout, so all three are
    /// rebuilt together whenever either buffer moves.
    fn rebind_storage_buffers(&self, device: &wgpu::Device) {
        let per_object = self
            .per_object_buffer
            .read()
            .expect("per-object buffer lock");
        let material = self.material_buffer.read().expect("material buffer lock");
        *self
            .gbuffer_bind_group
            .write()
            .expect("gbuffer bind group lock") =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gbuffer bind group (grown)"),
                layout: &self.gbuffer_bind_group_layout,
                entries: &shaders::bind_group_entries(
                    &shaders::gbuffer_generated::GBUFFER_RESOURCES,
                    |r| match r.name {
                        "camera" => self.camera_buffer.as_entire_binding(),
                        "per_objects" => per_object.as_entire_binding(),
                        "materials" => material.as_entire_binding(),
                        other => panic!("gbuffer bind group has no resource for `{other}`"),
                    },
                ),
            });
        *self
            .forward_pass
            .bind_group
            .write()
            .expect("forward bind group lock") =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("forward bind group (grown)"),
                layout: &self.forward_pass.bind_group_layout,
                entries: &shaders::bind_group_entries(
                    &shaders::pbr_generated::PBR_RESOURCES,
                    |r| match r.name {
                        "camera" => self.camera_buffer.as_entire_binding(),
                        "per_objects" => per_object.as_entire_binding(),
                        "materials" => material.as_entire_binding(),
                        "lighting" => self.lighting_buffer.as_entire_binding(),
                        "shadow_tex" => wgpu::BindingResource::TextureView(&self.shadow_array_view),
                        "shadow_sampler" => wgpu::BindingResource::Sampler(&self.shadow_sampler),
                        "shadow_cube_tex" => {
                            wgpu::BindingResource::TextureView(&self.shadow_cube_array_view)
                        }
                        other => panic!("forward bind group has no resource for `{other}`"),
                    },
                ),
            });
    }

    /// Ensure the per-object buffer fits `needed` instances, growing and
    /// rebinding when it does not. The lighting frame bind group is built
    /// per frame from the live buffer, so it needs no rebuild here.
    fn ensure_instance_capacity(&self, device: &wgpu::Device, needed: usize) {
        let grown = {
            let old = self
                .per_object_buffer
                .read()
                .expect("per-object buffer lock");
            Self::grown_storage_buffer(
                device,
                "per-object buffer (grown)",
                &old,
                std::mem::size_of::<PerObjectGpu>(),
                needed,
                &self.max_objects,
            )
        };
        if let Some(buffer) = grown {
            *self
                .per_object_buffer
                .write()
                .expect("per-object buffer lock") = buffer;
            self.rebind_storage_buffers(device);
        }
    }

    /// Ensure the material buffer fits `needed` entries, growing and
    /// rebinding when it does not.
    fn ensure_material_capacity(&self, device: &wgpu::Device, needed: usize) {
        let grown = {
            let old = self.material_buffer.read().expect("material buffer lock");
            Self::grown_storage_buffer(
                device,
                "material buffer (grown)",
                &old,
                OPENPBR_MATERIAL_SIZE,
                needed,
                &self.max_materials,
            )
        };
        if let Some(buffer) = grown {
            *self.material_buffer.write().expect("material buffer lock") = buffer;
            self.rebind_storage_buffers(device);
        }
    }

    /// Replace the GPU material table, growing the storage buffer (and
    /// the passes' bind groups) when `materials` exceeds current capacity;
    /// instances reference entries by index.
    pub fn upload_materials(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        materials: &[OpenPBRMaterial],
    ) {
        self.ensure_material_capacity(device, materials.len());
        let count = materials.len().min(
            self.max_materials
                .load(std::sync::atomic::Ordering::Relaxed) as usize,
        );
        queue.write_buffer(
            &self.material_buffer.read().expect("material buffer lock"),
            0,
            bytemuck::cast_slice(&materials[..count]),
        );
    }

    /// Convert and upload instances into the per-object buffer used by
    /// both gbuffer and forward passes, growing the buffer (and rebinding
    /// the passes) when the frame needs more than the current capacity.
    ///
    /// The CPU staging vector reserves `instances.len()` exactly up front,
    /// so the conversion never reallocates mid-frame; a repeated
    /// same-size frame reuses the GPU buffer as-is
    /// ([`Self::ensure_instance_capacity`] no-ops when the data fits).
    pub fn upload_instances(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[InstanceData],
    ) {
        self.ensure_instance_capacity(device, instances.len());
        let count = instances
            .len()
            .min(self.max_objects.load(std::sync::atomic::Ordering::Relaxed) as usize);
        let mut gpu_objects: Vec<PerObjectGpu> =
            Vec::with_capacity(staging_capacity_for_instances(count));
        for inst in instances.iter().take(count) {
            let model_arr: [[f32; 4]; 4] = inst.model_matrix.to_cols_array_2d();
            let normal_arr: [[f32; 4]; 4] = inst.normal_matrix.to_cols_array_2d();
            gpu_objects.push(PerObjectGpu {
                model: model_arr,
                normal_matrix: normal_arr,
                material_index: inst.material_index,
                _padding: [0; 3],
            });
        }
        queue.write_buffer(
            &self
                .per_object_buffer
                .read()
                .expect("per-object buffer lock"),
            0,
            bytemuck::cast_slice(&gpu_objects),
        );
    }

    /// Ensure the palette buffer fits `needed` slots, growing it (and only
    /// it — skinned bind groups are built per draw from the live buffer, so
    /// no pass needs rebinding here) when it does not.
    fn ensure_palette_capacity(&self, device: &wgpu::Device, needed: usize) {
        let grown = {
            let old = self.palette_buffer.read().expect("palette buffer lock");
            Self::grown_storage_buffer(
                device,
                "skin palette buffer (grown)",
                &old,
                PALETTE_BYTE_SIZE,
                needed,
                &self.max_palettes,
            )
        };
        if let Some(buffer) = grown {
            *self.palette_buffer.write().expect("palette buffer lock") = buffer;
        }
    }

    /// Pack one frame's joint palettes into the palette storage buffer and
    /// return one [`PaletteHandle`] per entry, in order.
    ///
    /// Each palette must be one full [`PALETTE_BYTE_SIZE`] slot (see
    /// [`crate::skinning::palette_upload_bytes`]): short slots are
    /// zero-padded (padding is never indexed), over-long slots are
    /// truncated — both defensively, producers stage exact slots. The
    /// buffer grows when the frame needs more slots than the current
    /// capacity; [`Renderer3D::palette_count`] records the staged count so
    /// stale handles record no commands instead of binding out of range.
    pub fn upload_skin_palettes(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        palettes: &[Vec<u8>],
    ) -> Vec<PaletteHandle> {
        self.ensure_palette_capacity(device, palettes.len());
        let count = palettes
            .len()
            .min(self.max_palettes.load(std::sync::atomic::Ordering::Relaxed) as usize);
        let mut bytes: Vec<u8> = Vec::with_capacity(count * PALETTE_BYTE_SIZE);
        for (slot, palette) in palettes.iter().take(count).enumerate() {
            debug_assert_eq!(
                palette.len(),
                PALETTE_BYTE_SIZE,
                "palette slot {slot} must be one full upload slot"
            );
            bytes.extend_from_slice(&palette[..palette.len().min(PALETTE_BYTE_SIZE)]);
            bytes.resize((slot + 1) * PALETTE_BYTE_SIZE, 0);
        }
        queue.write_buffer(
            &self.palette_buffer.read().expect("palette buffer lock"),
            0,
            &bytes,
        );
        self.palette_count
            .store(count as u32, std::sync::atomic::Ordering::Relaxed);
        (0..count as u32).map(PaletteHandle::from_raw).collect()
    }

    /// Draw one per-entity custom mesh with its own instance.
    ///
    /// The caller uploads the merged material table once (a custom entry's
    /// `instance.material_index` points into it), draws the shared sphere
    /// batch first, then calls this once per custom entry: the single
    /// instance is uploaded into slot 0 and `mesh` is drawn with count 1.
    /// Rewriting slot 0 per entry is why shared draws must come first;
    /// per-frame cost is one small upload + one draw per custom entity
    /// (no refit budget yet — see PLAN §h).
    pub fn render_custom_entry(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        g: &GbufferTargets<'_>,
        mesh: &Mesh,
        instance: &InstanceData,
    ) {
        self.upload_instances(device, queue, std::slice::from_ref(instance));
        self.render_gbuffer(encoder, g, mesh, 1);
    }

    /// Draw one per-entity skinned mesh through the skinned pipeline.
    ///
    /// GPU-draw path for one entry: `mesh` holds interleaved
    /// [`SkinnedVertex`] rows (see [`upload_skinned_mesh`]), `instance`
    /// goes into per-object slot 0 (shared draws must come first, same as
    /// [`render_custom_entry`](Self::render_custom_entry)), and `handle`
    /// selects the palette slot uploaded by
    /// [`upload_skin_palettes`](Self::upload_skin_palettes) for binding 3.
    /// A stale handle (at or past the last staged count) records no
    /// commands — exact no-op, never an out-of-bounds bind. The fragment
    /// stage and targets match [`render_gbuffer`](Self::render_gbuffer),
    /// so Cpu and Gpu draws land in the same buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn render_skinned_entry(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        g: &GbufferTargets<'_>,
        mesh: &Mesh,
        instance: &InstanceData,
        handle: PaletteHandle,
    ) {
        if handle.index()
            >= self
                .palette_count
                .load(std::sync::atomic::Ordering::Relaxed) as usize
        {
            return;
        }
        self.upload_instances(device, queue, std::slice::from_ref(instance));
        let palette = self.palette_buffer.read().expect("palette buffer lock");
        let per_object = self.per_object_buffer.read().unwrap();
        let material = self.material_buffer.read().unwrap();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinned gbuffer bind group"),
            layout: &self.skinned_bind_group_layout,
            entries: &shaders::bind_group_entries(&GBUFFER_SKINNED_RESOURCES, |r| match r.name {
                "camera" => self.camera_buffer.as_entire_binding(),
                "per_objects" => per_object.as_entire_binding(),
                "materials" => material.as_entire_binding(),
                "palette" => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &palette,
                    offset: handle.byte_offset(),
                    size: std::num::NonZeroU64::new(PALETTE_BYTE_SIZE as u64),
                }),
                other => panic!("skinned bind group has no resource for `{other}`"),
            }),
        });
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("skinned gbuffer pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: g.albedo,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.normal,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.material_id,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.world_position,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.material_params,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: g.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&self.skinned_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.num_indices, 0, 0..1);
    }

    /// Record the gbuffer pass: fills the five MRT targets + depth for
    /// `instance_count` uploaded instances of `mesh`.
    pub fn render_gbuffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        g: &GbufferTargets<'_>,
        mesh: &Mesh,
        instance_count: u32,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gbuffer pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: g.albedo,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.normal,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.material_id,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.world_position,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.material_params,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: g.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&self.gbuffer_pipeline);
        {
            let bind_group = self
                .gbuffer_bind_group
                .read()
                .expect("gbuffer bind group lock");
            rpass.set_bind_group(0, &*bind_group, &[]);
        }
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.num_indices, 0, 0..instance_count);
    }

    /// Record the deferred lighting pass: reconstructs surface data from the
    /// g-buffer in `g`, evaluates the OpenPBR BRDF and writes HDR color into `output`.
    pub fn render_lighting(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        g: &GbufferTargets<'_>,
        output: &wgpu::TextureView,
    ) {
        // The bind group is rebuilt per frame: gbuffer views come from the
        // render-plan pool (transient) or from persistent textures, and the
        // material buffer may have grown since the last frame.
        // Binding numbers come from the table; only the name → live
        // resource mapping is written out here.
        let material = self.material_buffer.read().expect("material buffer lock");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting bind group (frame)"),
            layout: &self.lighting_pass.bind_group_layout,
            entries: &shaders::bind_group_entries(
                &shaders::lighting_generated::LIGHTING_RESOURCES,
                |r| match r.name {
                    "camera" => self.camera_buffer.as_entire_binding(),
                    "lighting" => self.lighting_buffer.as_entire_binding(),
                    "materials" => material.as_entire_binding(),
                    "albedo_tex" => wgpu::BindingResource::TextureView(g.albedo),
                    "normal_tex" => wgpu::BindingResource::TextureView(g.normal),
                    "material_id_tex" => wgpu::BindingResource::TextureView(g.material_id),
                    "world_pos_tex" => wgpu::BindingResource::TextureView(g.world_position),
                    "mat_params_tex" => wgpu::BindingResource::TextureView(g.material_params),
                    "depth_tex" => wgpu::BindingResource::TextureView(g.depth),
                    "lighting_sampler" => {
                        wgpu::BindingResource::Sampler(&self.lighting_pass.sampler)
                    }
                    "shadow_tex" => wgpu::BindingResource::TextureView(&self.shadow_array_view),
                    "shadow_sampler" => wgpu::BindingResource::Sampler(&self.shadow_sampler),
                    "shadow_cube_tex" => {
                        wgpu::BindingResource::TextureView(&self.shadow_cube_array_view)
                    }
                    other => panic!("lighting bind group has no resource for `{other}`"),
                },
            ),
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lighting pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&self.lighting_pass.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..4, 0..1);
    }

    /// Record the forward pass: draws lit geometry into the HDR `output`
    /// layer, depth-testing against (and optionally clearing) `depth`.
    /// `clear_depth = true` when the forward pass runs standalone; `false`
    /// when it follows the gbuffer pass and must share its depth.
    pub fn render_forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        mesh: &Mesh,
        instance_count: u32,
        clear_depth: bool,
    ) {
        let depth_ops = wgpu::Operations {
            load: if clear_depth {
                wgpu::LoadOp::Clear(1.0)
            } else {
                wgpu::LoadOp::Load
            },
            store: wgpu::StoreOp::Store,
        };
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("forward pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(depth_ops),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&self.forward_pass.pipeline);
        {
            let bind_group = self
                .forward_pass
                .bind_group
                .read()
                .expect("forward bind group lock");
            rpass.set_bind_group(0, &*bind_group, &[]);
        }
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.num_indices, 0, 0..instance_count);
    }

    /// Record the textured-forward pass: draws lit geometry with one bound
    /// [`MaterialTextureSet`] (resolved against `cache`, unbound slots fall
    /// back to neutral white) into the HDR `output` layer, depth-testing
    /// against (and optionally clearing) `depth`. `clear_depth = true` when
    /// the pass runs standalone; `false` when it follows the gbuffer pass
    /// and must share its depth. Mirrors [`render_forward`](Self::render_forward);
    /// the scalar `is_metallic` switch is untouched by bound textures.
    ///
    /// # Panics
    ///
    /// Panics when [`ensure_textured_forward`](Self::ensure_textured_forward)
    /// was not called yet — the pipeline and fallbacks do not exist until
    /// then (the constructor takes no queue, so they cannot be built there).
    #[allow(clippy::too_many_arguments)]
    pub fn render_forward_textured(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        mesh: &Mesh,
        instance_count: u32,
        clear_depth: bool,
        textures: &MaterialTextureSet,
        cache: &TextureCache,
    ) {
        let pass = self.textured_forward.as_ref().expect(
            "textured forward pass not built: call ensure_textured_forward(device, queue) first",
        );
        // The bind group is rebuilt per frame: the material buffer may have
        // grown and the bound set may have changed. Binding numbers come
        // from the table; only the name → live resource mapping is here.
        let material = self.material_buffer.read().expect("material buffer lock");
        let per_object = self
            .per_object_buffer
            .read()
            .expect("per-object buffer lock");
        let base_color_view = Self::material_view(
            textures,
            cache,
            &pass.fallback_color.view,
            &pass.fallback_data.view,
            TextureRole::BaseColor,
        );
        let data_view = Self::material_view(
            textures,
            cache,
            &pass.fallback_color.view,
            &pass.fallback_data.view,
            TextureRole::MetallicRoughness,
        );
        let emissive_view = Self::material_view(
            textures,
            cache,
            &pass.fallback_color.view,
            &pass.fallback_data.view,
            TextureRole::Emissive,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("textured forward bind group (frame)"),
            layout: &pass.bind_group_layout,
            entries: &shaders::bind_group_entries(
                &shaders::material_textures::TEXTURED_PBR_RESOURCES,
                |r| match r.name {
                    "camera" => self.camera_buffer.as_entire_binding(),
                    "per_objects" => per_object.as_entire_binding(),
                    "materials" => material.as_entire_binding(),
                    "lighting" => self.lighting_buffer.as_entire_binding(),
                    "shadow_tex" => wgpu::BindingResource::TextureView(&self.shadow_array_view),
                    "shadow_sampler" => wgpu::BindingResource::Sampler(&self.shadow_sampler),
                    "shadow_cube_tex" => {
                        wgpu::BindingResource::TextureView(&self.shadow_cube_array_view)
                    }
                    "base_color_tex" => wgpu::BindingResource::TextureView(base_color_view),
                    "metallic_roughness_tex" => wgpu::BindingResource::TextureView(data_view),
                    "emissive_tex" => wgpu::BindingResource::TextureView(emissive_view),
                    "material_sampler" => wgpu::BindingResource::Sampler(&pass.sampler),
                    other => {
                        panic!("textured forward bind group has no resource for `{other}`")
                    }
                },
            ),
        });

        let depth_ops = wgpu::Operations {
            load: if clear_depth {
                wgpu::LoadOp::Clear(1.0)
            } else {
                wgpu::LoadOp::Load
            },
            store: wgpu::StoreOp::Store,
        };
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("textured forward pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(depth_ops),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&pass.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.num_indices, 0, 0..instance_count);
    }

    /// Record the final blend into `inputs.target`: mixes deferred + forward
    /// HDR layers per `inputs.mode` and adds bloom scaled by
    /// `inputs.bloom_intensity` (0 keeps the legacy path pixel-identical).
    pub fn render_composite(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        inputs: CompositeInputs<'_>,
    ) {
        // The bloom view is bound unconditionally; a zero intensity makes the
        // contribution null, so the legacy path (`render_scene`) stays
        // pixel-identical to the plan path with bloom culled.
        queue.write_buffer(
            &self.bloom_pass.params_buffer,
            0,
            bytemuck::bytes_of(&BloomUniform {
                threshold: 0.0,
                intensity: inputs.bloom_intensity,
                mode: inputs.mode,
                _pad: 0.0,
            }),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite bind group"),
            layout: &self.composite_pass.bind_group_layout,
            // Binding numbers come from the table; only the name → live
            // resource mapping is written out here.
            entries: &shaders::bind_group_entries(
                &shaders::hdr_composite_generated::HDR_RESOURCES,
                |r| match r.name {
                    "deferred_tex" => wgpu::BindingResource::TextureView(inputs.hdr),
                    "forward_tex" => wgpu::BindingResource::TextureView(inputs.hdr_fwd),
                    "composite_sampler" => wgpu::BindingResource::Sampler(&self.composite_sampler),
                    "bloom_tex" => wgpu::BindingResource::TextureView(inputs.bloom),
                    "bloom_params" => wgpu::BindingResource::Buffer(
                        self.bloom_pass.params_buffer.as_entire_buffer_binding(),
                    ),
                    other => panic!("composite bind group has no resource for `{other}`"),
                },
            ),
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: inputs.target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&self.composite_pass.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..4, 0..1);
    }

    /// Record the opt-in distance-fog mix: `hdr` through the fog blend into
    /// `target`, with depth from the g-buffer `depth` buffer.
    ///
    /// Depth is the hardware (non-linear) depth: it is linearized through
    /// `Camera::inv_view_proj` and the Euclidean distance to the eye feeds
    /// `color + (fog.color - color) * (1 - exp(-density * depth))` — the
    /// same math as [`crate::frame_passes::apply_fog`]. Non-positive or
    /// non-finite `density` records no commands (exact no-op, so the
    /// disabled pass leaves the frame pixel-identical).
    pub fn render_fog(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        inputs: FogInputs<'_>,
    ) {
        let density = inputs.density;
        if !density.is_finite() || density <= 0.0 {
            return;
        }
        queue.write_buffer(
            &self.fog.params_buffer,
            0,
            bytemuck::bytes_of(&FogUniform::pack(inputs.color, density)),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fog bind group"),
            layout: &self.fog.bind_group_layout,
            entries: &shaders::bind_group_entries(&shaders::fog_generated::FOG_RESOURCES, |r| {
                match r.name {
                    "hdr_tex" => wgpu::BindingResource::TextureView(inputs.hdr),
                    "fog_sampler" => wgpu::BindingResource::Sampler(&self.composite_sampler),
                    "depth_tex" => wgpu::BindingResource::TextureView(inputs.depth),
                    "camera" => self.camera_buffer.as_entire_binding(),
                    "fog_params" => wgpu::BindingResource::Buffer(
                        self.fog.params_buffer.as_entire_buffer_binding(),
                    ),
                    other => panic!("fog bind group has no resource for `{other}`"),
                }
            }),
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fog pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: inputs.target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&self.fog.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..4, 0..1);
    }

    /// All-in-one legacy frame on the renderer's persistent targets:
    /// gbuffer -> lighting -> forward -> composite straight into `target`.
    /// The render-graph path (`frame_exec`) supersedes this for plan-driven
    /// execution, but it remains the reference hybrid pipeline.
    pub fn render_scene(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        mesh: &Mesh,
        instance_count: u32,
    ) {
        let g = GbufferTargets {
            albedo: &self.gbuffer.albedo_view,
            normal: &self.gbuffer.normal_view,
            material_id: &self.gbuffer.material_id_view,
            world_position: &self.gbuffer.world_position_view,
            material_params: &self.gbuffer.material_params_view,
            depth: &self.gbuffer.depth_view,
        };
        self.render_gbuffer(encoder, &g, mesh, instance_count);
        // Depth pre-passes for shadowed lights (no-op when none).
        self.render_shadows(device, encoder, mesh, instance_count);
        self.render_lighting(device, encoder, &g, &self.pbr_texture_view);
        self.render_forward(
            encoder,
            &self.gbuffer.depth_view,
            &self.forward_pass.color_view,
            mesh,
            instance_count,
            false,
        );
        self.render_composite(
            device,
            queue,
            encoder,
            CompositeInputs {
                target,
                hdr: &self.pbr_texture_view,
                hdr_fwd: &self.forward_pass.color_view,
                bloom: &self.pbr_texture_view,
                bloom_intensity: 0.0,
                // Legacy path always runs the hybrid mix.
                mode: 2,
            },
        );
    }

    /// Downsample pass of the bloom chain: thresholded for the first level,
    /// plain downsample for deeper levels (`threshold` = 0 passes everything
    /// except pure black). Writes into `dst` with a replace blend.
    pub fn render_bloom_down(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        threshold: f32,
    ) {
        queue.write_buffer(
            &self.bloom_pass.params_buffer,
            0,
            bytemuck::bytes_of(&BloomUniform {
                threshold,
                intensity: 0.0,
                mode: 0,
                _pad: 0.0,
            }),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom down bind group"),
            layout: &self.bloom_pass.bind_group_layout,
            entries: &shaders::bind_group_entries(
                &shaders::bloom_generated::BLOOM_RESOURCES,
                |r| match r.name {
                    "src_tex" => wgpu::BindingResource::TextureView(src),
                    "src_sampler" => wgpu::BindingResource::Sampler(&self.composite_sampler),
                    "bloom_params" => wgpu::BindingResource::Buffer(
                        self.bloom_pass.params_buffer.as_entire_buffer_binding(),
                    ),
                    other => panic!("bloom bind group has no resource for `{other}`"),
                },
            ),
        });
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bloom down pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&self.bloom_pass.down_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..4, 0..1);
    }

    /// Upsample pass of the bloom chain: samples `src`, adds the result over
    /// the *loaded* contents of `dst` (additive blend) — the classic
    /// "upsample with add" cascade that recombines the levels.
    pub fn render_bloom_up(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom up bind group"),
            layout: &self.bloom_pass.bind_group_layout,
            entries: &shaders::bind_group_entries(
                &shaders::bloom_generated::BLOOM_RESOURCES,
                |r| match r.name {
                    "src_tex" => wgpu::BindingResource::TextureView(src),
                    "src_sampler" => wgpu::BindingResource::Sampler(&self.composite_sampler),
                    "bloom_params" => wgpu::BindingResource::Buffer(
                        self.bloom_pass.params_buffer.as_entire_buffer_binding(),
                    ),
                    other => panic!("bloom bind group has no resource for `{other}`"),
                },
            ),
        });
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bloom up pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&self.bloom_pass.up_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..4, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_assets::scene::ShadowCast;

    fn clip_of(vp: [[f32; 4]; 4], p: [f32; 3]) -> glam::Vec4 {
        glam::Mat4::from_cols_array_2d(&vp) * glam::Vec4::new(p[0], p[1], p[2], 1.0)
    }

    fn ndc_of(vp: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
        let c = clip_of(vp, p);
        [c.x / c.w, c.y / c.w, c.z / c.w]
    }

    fn assert_valid_wgsl(name: &str, source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name} must parse: {e}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} must validate: {e}"));
    }

    #[test]
    fn skinned_pipeline_sources_validate_with_naga() {
        // The skinned pipelines assemble the skinned vertex stage with the
        // classic fragment stage: both modules the renderer compiles must
        // validate (the shadow variants reuse the same vertex module).
        assert_valid_wgsl(
            "skinned_vertex",
            &crate::skinning::wgsl_vertex_source_skinned(),
        );
        assert_valid_wgsl("skinned_fragment", &crate::shaders::gbuffer_fragment());
        assert_eq!(crate::skinning::skinned_entry_point(), "vs_main_skinned");
    }

    #[test]
    fn skinned_upload_rejects_empty_and_bad_indices() {
        // Validation needs a device: skipped headless, like every probe.
        let Some((device, _)) = try_device() else {
            eprintln!("no GPU adapter; skipping skinned upload probe");
            return;
        };
        use crate::mesh_upload::UploadError;
        let vertex = SkinnedVertex {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            joints: [0, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        };
        assert!(matches!(
            upload_skinned_mesh(&device, &[], &[0, 1, 2]),
            Err(UploadError::EmptyMesh)
        ));
        assert!(matches!(
            upload_skinned_mesh(&device, &[vertex], &[]),
            Err(UploadError::EmptyMesh)
        ));
        assert!(matches!(
            upload_skinned_mesh(&device, &[vertex], &[0, 1]),
            Err(UploadError::InvalidMesh(_))
        ));
        assert!(matches!(
            upload_skinned_mesh(&device, &[vertex], &[0, 1, 7]),
            Err(UploadError::InvalidMesh(_))
        ));
        assert!(upload_skinned_mesh(&device, &[vertex; 3], &[0, 1, 2]).is_ok());
    }

    /// Skinned color + shadow draws bind the palette and record without
    /// panicking; a stale handle is an exact no-op. Skipped when no
    /// adapter is available.
    #[test]
    fn skinned_entry_renders_and_shadows_without_panic() {
        let Some((device, queue)) = try_device() else {
            eprintln!("no GPU adapter; skipping skinned draw probe");
            return;
        };
        const W: u32 = 64;
        const H: u32 = 64;
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            width: W,
            height: H,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        let renderer = Renderer3D::new(&device, &surface_config, 1);
        // Bind-group assembly pins the layout contract: four rows off the
        // skinned resource table, palette at binding 3, vertex-only.
        assert_eq!(GBUFFER_SKINNED_RESOURCES.len(), 4);
        let palette_row = GBUFFER_SKINNED_RESOURCES
            .iter()
            .find(|r| r.binding == 3)
            .expect("palette at binding 3");
        assert_eq!(palette_row.name, "palette");
        assert_eq!(
            palette_row.visibility,
            wgpu::ShaderStages::VERTEX,
            "palette is vertex-only"
        );
        {
            let per_object = renderer
                .per_object_buffer
                .read()
                .expect("per-object buffer lock");
            let material = renderer
                .material_buffer
                .read()
                .expect("material buffer lock");
            let palette = renderer.palette_buffer.read().expect("palette buffer lock");
            let entries =
                crate::shaders::bind_group_entries(&GBUFFER_SKINNED_RESOURCES, |r| match r.name {
                    "camera" => renderer.camera_buffer.as_entire_binding(),
                    "per_objects" => per_object.as_entire_binding(),
                    "materials" => material.as_entire_binding(),
                    "palette" => palette.as_entire_binding(),
                    other => panic!("skinned bind group has no resource for `{other}`"),
                });
            assert_eq!(entries.len(), 4);
            assert_eq!(
                entries.iter().map(|e| e.binding).collect::<Vec<_>>(),
                vec![0, 1, 2, 3]
            );
        }

        let mut red = ornis_core::OpenPBRMaterial::dielectric();
        red.base.color_rgb([0.8, 0.2, 0.2]);
        red.specular.roughness(0.5);
        renderer.upload_materials(&device, &queue, &[red]);
        // Identity palette over one triangle: the vertex stage passes the
        // bind pose through (model is identity, like the extraction's
        // skinned entries).
        let row = |position: [f32; 3]| SkinnedVertex {
            position,
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            joints: [0, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        };
        let mesh = upload_skinned_mesh(
            &device,
            &[
                row([-0.5, -0.5, 0.0]),
                row([0.5, -0.5, 0.0]),
                row([0.0, 0.5, 0.0]),
            ],
            &[0, 1, 2],
        )
        .expect("triangle valid");
        let palette_bytes =
            crate::skinning::joint_palette_bytes(&[glam::Mat4::IDENTITY]).expect("fits");
        assert_eq!(palette_bytes.len(), PALETTE_BYTE_SIZE);
        let handles = renderer.upload_skin_palettes(&device, &queue, &[palette_bytes]);
        assert_eq!(handles, vec![PaletteHandle::from_raw(0)]);
        // Identity view-projection: clip = world, depth 0 < clear 1.
        renderer.set_camera(
            &queue,
            &[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [0.0, 0.0, 0.0],
        );
        // One shadowed light so the skinned depth pre-pass draws.
        renderer.set_lights_full(
            &queue,
            [0.1, 0.1, 0.15],
            1.0,
            1.0,
            &[ornis_assets::scene::LightDesc::Directional {
                direction: [0.2, 1.0, 0.3],
                intensity: 1.2,
                color: [1.0, 1.0, 1.0],
                shadow: ShadowCast::Enabled,
            }],
        );
        let g = GbufferTargets {
            albedo: &renderer.gbuffer.albedo_view,
            normal: &renderer.gbuffer.normal_view,
            material_id: &renderer.gbuffer.material_id_view,
            world_position: &renderer.gbuffer.world_position_view,
            material_params: &renderer.gbuffer.material_params_view,
            depth: &renderer.gbuffer.depth_view,
        };
        let instance = InstanceData {
            model_matrix: glam::Mat4::IDENTITY,
            normal_matrix: glam::Mat4::IDENTITY,
            material_index: MaterialIdx::from_raw(0),
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("skinned probe encoder"),
        });
        renderer.render_skinned_entry(
            &device,
            &queue,
            &mut encoder,
            &g,
            &mesh,
            &instance,
            handles[0],
        );
        renderer.render_skinned_shadows(&device, &mut encoder, &mesh, handles[0]);
        // Stale handle: exact no-op, never an out-of-bounds bind.
        renderer.render_skinned_entry(
            &device,
            &queue,
            &mut encoder,
            &g,
            &mesh,
            &instance,
            PaletteHandle::from_raw(99),
        );
        renderer.render_skinned_shadows(&device, &mut encoder, &mesh, PaletteHandle::from_raw(99));
        queue.submit(std::iter::once(encoder.finish()));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
    }

    #[test]
    fn dir_shadow_vp_centers_origin_with_light_depth_order() {
        // Light above: to-light = +Y, eye at +Y·30 looking at origin.
        let vp = dir_shadow_vp(glam::Vec3::Y);
        let center = ndc_of(vp, [0.0, 0.0, 0.0]);
        assert!(center[0].abs() < 1e-5 && center[1].abs() < 1e-5);
        assert!((0.0..=1.0).contains(&center[2]));
        // Nearer the light (higher Y) = smaller depth.
        let hi = ndc_of(vp, [0.0, 5.0, 0.0])[2];
        let lo = ndc_of(vp, [0.0, -5.0, 0.0])[2];
        assert!(hi < center[2] && center[2] < lo, "{hi} {center:?} {lo}");
        // Box rim stays inside the frustum.
        for p in [
            [12.0, 0.0, 0.0],
            [-12.0, 0.0, 0.0],
            [0.0, 0.0, 12.0],
            [0.0, 0.0, -12.0],
        ] {
            let n = ndc_of(vp, p);
            assert!(
                n[0].abs() <= 1.0 + 1e-4 && n[1].abs() <= 1.0 + 1e-4,
                "{p:?} -> {n:?}"
            );
        }
    }

    #[test]
    fn spot_shadow_vp_points_down_its_axis() {
        let vp = spot_shadow_vp([0.0, 6.0, 0.0], glam::Vec3::NEG_Y, 35.0, 40.0);
        let hit = ndc_of(vp, [0.0, 0.0, 0.0]);
        assert!(hit[0].abs() < 1e-4 && hit[1].abs() < 1e-4);
        assert!((0.0..=1.0).contains(&hit[2]));
        // Behind the light has negative clip w (mirrored projection).
        assert!(clip_of(vp, [0.0, 8.0, 0.0]).w < 0.0);
        // Off-axis outside the 35° cone (half-width ≈ 4.2 at depth 6;
        // the shadow frame maps world X onto NDC Y here).
        let side = ndc_of(vp, [5.0, 0.0, 0.0]);
        assert!(side[0].abs().max(side[1].abs()) > 1.0, "{side:?}");
    }

    #[test]
    fn point_cube_face_vp_centers_its_axis() {
        // Light at origin-ish, range 20: each face centers its axis
        // with directx depth order (nearer = smaller).
        let pos = [2.0, 3.0, 4.0];
        let axes = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for (face, axis) in axes.iter().enumerate() {
            let vp = point_cube_face_vp(pos, 20.0, face);
            let target = [
                pos[0] + axis[0] * 5.0,
                pos[1] + axis[1] * 5.0,
                pos[2] + axis[2] * 5.0,
            ];
            let hit = ndc_of(vp, target);
            assert!(
                hit[0].abs() < 1e-4 && hit[1].abs() < 1e-4,
                "face {face}: {hit:?}"
            );
            assert!((0.0..=1.0).contains(&hit[2]), "face {face}: {hit:?}");
            // Nearer along the axis = smaller depth (matches the
            // analytic sampling formula's monotonicity).
            let near = ndc_of(
                vp,
                [
                    pos[0] + axis[0] * 2.0,
                    pos[1] + axis[1] * 2.0,
                    pos[2] + axis[2] * 2.0,
                ],
            )[2];
            let far = ndc_of(
                vp,
                [
                    pos[0] + axis[0] * 10.0,
                    pos[1] + axis[1] * 10.0,
                    pos[2] + axis[2] * 10.0,
                ],
            )[2];
            assert!(
                near < hit[2] && hit[2] < far,
                "face {face}: {near} {hit:?} {far}"
            );
        }
    }

    #[test]
    fn point_cube_face_vp_matches_sampler_frame() {
        // Texel placement must match the hardware cube-sampling frame
        // (GL convention: +X: (−z,−y), −X: (+z,−y), +Y: (+x,+z),
        // −Y: (+x,−z), +Z: (+x,−y), −Z: (−x,−y)). Rasterization maps
        // NDC y+1 onto texture row 0 while the sampler reads v=0 from
        // the top, so a +u-source offset lands at positive NDC x and a
        // +v-source offset at NEGATIVE NDC y (the V-mirror in
        // `point_cube_face_vp`). Regression pin: without the mirror
        // the sampler reads mirrored texels (all-lit point shadows).
        let pos = [2.0, 3.0, 4.0];
        let axes: [[[f32; 3]; 2]; 6] = [
            [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        ];
        let face_axis = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for (face, ([u, v], axis)) in axes.iter().zip(face_axis.iter()).enumerate() {
            let vp = point_cube_face_vp(pos, 20.0, face);
            let base = [
                pos[0] + axis[0] * 5.0,
                pos[1] + axis[1] * 5.0,
                pos[2] + axis[2] * 5.0,
            ];
            let pu = [base[0] + u[0], base[1] + u[1], base[2] + u[2]];
            let pv = [base[0] + v[0], base[1] + v[1], base[2] + v[2]];
            let nu = ndc_of(vp, pu);
            let nv = ndc_of(vp, pv);
            assert!(nu[0] > 0.05, "face {face}: +u offset -> {nu:?}");
            assert!(nv[1] < -0.05, "face {face}: +v offset -> {nv:?}");
        }
    }

    fn dir_probe(direction: [f32; 3], shadow: ornis_assets::scene::ShadowCast) -> LightDesc {
        LightDesc::Directional {
            direction,
            intensity: 1.0,
            color: [1.0, 1.0, 1.0],
            shadow,
        }
    }

    #[test]
    fn per_object_material_index_substitutes_to_u32() {
        use crate::shaders::interface::{GbufferFragmentInput, GbufferVertexOutput};
        // The DSL substitutes the transparent `MaterialIdx` newtype to WGSL
        // `u32`: the declaration texts are unchanged from the raw-`u32` era.
        assert!(
            PerObjectGpu::WGSL_SOURCE.contains("material_index: u32"),
            "{}",
            PerObjectGpu::WGSL_SOURCE
        );
        assert_eq!(
            PerObjectGpu::FIELD_NAMES,
            &["model", "normal_matrix", "material_index"]
        );
        assert!(
            GbufferVertexOutput::WGSL_SOURCE.contains("material_index: u32"),
            "{}",
            GbufferVertexOutput::WGSL_SOURCE
        );
        assert!(
            GbufferFragmentInput::WGSL_SOURCE.contains("material_index: u32"),
            "{}",
            GbufferFragmentInput::WGSL_SOURCE
        );
        // CPU layout is unchanged: transparent wrapper, same offsets/size.
        assert_eq!(std::mem::size_of::<MaterialIdx>(), 4);
        assert_eq!(std::mem::size_of::<PerObjectGpu>(), 144);
        assert_eq!(std::mem::offset_of!(PerObjectGpu, material_index), 128);
        // The derived declaration validates with naga.
        let mut module = naga::Module::default();
        let handle = PerObjectGpu::naga_add_type(&mut module);
        let naga::TypeInner::Struct { members, span } = &module.types[handle].inner else {
            panic!("PerObjectGpu must lower to a naga struct");
        };
        assert_eq!(*span, 144);
        assert_eq!(members[2].offset, 128);
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("PerObjectGpu declaration must validate");
    }

    #[test]
    fn lighting_uniform_spells_eight_lights() {
        // The WGSL block must spell the widened array; the member list
        // itself is unchanged (multipliers stay CPU-baked, no new
        // shader-visible field).
        assert!(
            LightingUniform::WGSL_SOURCE.contains("array<Light, 8>"),
            "{}",
            LightingUniform::WGSL_SOURCE
        );
        assert_eq!(
            LightingUniform::FIELD_NAMES,
            &["ambient_color", "lights", "light_count"]
        );
    }

    #[test]
    fn ten_lights_upload_eight_and_drop_two() {
        let lights: Vec<LightDesc> = (0..10)
            .map(|_| dir_probe([1.0, 1.0, 1.0], ornis_assets::scene::ShadowCast::Disabled))
            .collect();
        let built = build_lighting_uniform([0.1, 0.1, 0.15], 1.0, 1.0, &lights, None);
        assert_eq!(
            built.stats,
            LightUploadStats {
                uploaded: 8,
                dropped_lights: 2,
                dropped_shadows: 0,
            }
        );
        assert_eq!(built.uniform.light_count, 8);
        // The public preview agrees with the upload path (single logic).
        assert_eq!(count_light_drops(&lights), built.stats);
    }

    #[test]
    fn shadow_overflow_counts_dropped_shadows() {
        // Five shadowed directionals over four 2D layers.
        let dirs: Vec<LightDesc> = (0..5)
            .map(|_| dir_probe([0.0, 1.0, 0.0], ornis_assets::scene::ShadowCast::Enabled))
            .collect();
        let stats = count_light_drops(&dirs);
        assert_eq!(stats.uploaded, 5);
        assert_eq!(stats.dropped_lights, 0);
        assert_eq!(stats.dropped_shadows, 1, "{stats:?}");
        // Three shadowed points over two cubes.
        let points: Vec<LightDesc> = (0..3)
            .map(|i| LightDesc::Point {
                position: [i as f32, 4.0, 6.0],
                intensity: 100.0,
                color: [1.0, 1.0, 1.0],
                range: 30.0,
                shadow: ShadowCast::Enabled,
            })
            .collect();
        let stats = count_light_drops(&points);
        assert_eq!(stats.dropped_shadows, 1, "{stats:?}");
        // A shadow request on a dropped excess light counts too.
        let mut lights: Vec<LightDesc> = (0..8)
            .map(|_| dir_probe([1.0, 1.0, 1.0], ornis_assets::scene::ShadowCast::Disabled))
            .collect();
        lights.push(dir_probe(
            [0.0, 1.0, 0.0],
            ornis_assets::scene::ShadowCast::Enabled,
        ));
        lights.push(dir_probe(
            [0.0, 1.0, 0.0],
            ornis_assets::scene::ShadowCast::Enabled,
        ));
        let stats = count_light_drops(&lights);
        assert_eq!(stats.dropped_lights, 2, "{stats:?}");
        assert_eq!(stats.dropped_shadows, 2, "{stats:?}");
    }

    #[test]
    fn legacy_shadow_path_matches_dir_shadow_vp() {
        // `fit: None` must reproduce the legacy matrix bit-for-bit:
        // directional-only scenes stay pixel-identical.
        let light = dir_probe([1.0, 1.0, 1.0], ornis_assets::scene::ShadowCast::Enabled);
        let built = build_lighting_uniform([0.1, 0.1, 0.15], 1.0, 1.0, &[light], None);
        let v = glam::Vec3::new(1.0, 1.0, 1.0).normalize();
        assert_eq!(built.uniform.lights[0].shadow_vp, dir_shadow_vp(v));
        assert_eq!(built.uniform.lights[0].params[3], 0.0);
    }

    #[test]
    fn fitted_shadow_covers_radius_50_scene() {
        let (center, half) = shadow_fit_for_bounds([-50.0; 3], [50.0; 3]);
        assert_eq!(center, [0.0, 0.0, 0.0]);
        assert!((half - 50.0).abs() < 1e-6, "{half}");
        let vp = dir_shadow_vp_fitted(glam::Vec3::Y, center, half);
        for x in [-50.0, 50.0] {
            for y in [-50.0, 50.0] {
                for z in [-50.0, 50.0] {
                    let n = ndc_of(vp, [x, y, z]);
                    assert!(
                        n[0].abs() <= 1.0 + 1e-3 && n[1].abs() <= 1.0 + 1e-3,
                        "{x},{y},{z} -> {n:?}"
                    );
                    assert!((0.0..=1.0).contains(&n[2]), "{x},{y},{z} -> {n:?}");
                }
            }
        }
    }

    #[test]
    fn default_fit_keeps_legacy_box() {
        assert_eq!(
            shadow_fit_for_bounds([-1.0; 3], [1.0; 3]),
            ([0.0; 3], SHADOW_ORTHO_HALF)
        );
        assert_eq!(SHADOW_ORTHO_HALF, 12.0);
        assert_eq!(
            shadow_fit_for_bounds([f32::NAN; 3], [0.0; 3]),
            ([0.0; 3], SHADOW_ORTHO_HALF)
        );
    }

    #[test]
    fn transparency_defaults_to_replace_and_sorts_far_first() {
        // Default mode is opaque: the forward pipeline blends with REPLACE
        // (golden-pinned); opt-in selects ALPHA_BLENDING.
        assert_eq!(TransparencyOptions::default().mode, BlendMode::Opaque);
        assert!(!TransparencyOptions::default().sorted_alpha());
        assert!(!BlendMode::Opaque.is_transparent());
        assert!(BlendMode::Transparent.is_transparent());
        assert_eq!(
            forward_blend_state(TransparencyOptions::default()),
            wgpu::BlendState::REPLACE
        );
        assert_eq!(
            forward_blend_state(TransparencyOptions::new(BlendMode::Transparent)),
            wgpu::BlendState::ALPHA_BLENDING
        );
        assert_eq!(BlendMode::Opaque.blend_state(), wgpu::BlendState::REPLACE);
        assert_eq!(
            BlendMode::Transparent.blend_state(),
            wgpu::BlendState::ALPHA_BLENDING
        );
        // Legacy bool polarity is preserved.
        assert_eq!(BlendMode::from(false), BlendMode::Opaque);
        assert_eq!(BlendMode::from(true), BlendMode::Transparent);
        assert_eq!(TransparencyOptions::from(false).mode, BlendMode::Opaque);
        // Opacity classification: 1.0+ is opaque, below is transparent,
        // non-finite is rejected.
        assert_eq!(BlendMode::from_opacity(1.0), Ok(BlendMode::Opaque));
        assert_eq!(BlendMode::from_opacity(2.0), Ok(BlendMode::Opaque));
        assert_eq!(BlendMode::from_opacity(0.5), Ok(BlendMode::Transparent));
        assert_eq!(BlendMode::from_opacity(0.0), Ok(BlendMode::Transparent));
        assert!(matches!(
            BlendMode::from_opacity(f32::NAN),
            Err(TransparencyError::NonFiniteOpacity(_))
        ));
        assert!(matches!(
            BlendMode::from_opacity(f32::INFINITY),
            Err(TransparencyError::NonFiniteOpacity(_))
        ));
        // Depth sort: far first, stable on ties, empty stays empty.
        // Transparent mode requires this order before uploading (see
        // `TransparencyOptions`; the whole-frame twin is
        // `crate::extraction::sort_by_depth`).
        assert_eq!(sort_by_depth(&[]), Vec::<u32>::new());
        assert_eq!(sort_by_depth(&[2.0]), vec![0]);
        assert_eq!(sort_by_depth(&[1.0, 5.0, 3.0]), vec![1, 2, 0]);
        assert_eq!(sort_by_depth(&[2.0, 2.0, 1.0]), vec![0, 1, 2]);
    }

    #[test]
    fn fog_uniform_layout_matches_wgsl() {
        // Layout gates (precedent: Camera/Lighting via `WgslStruct`):
        // `color: vec3<f32>` at 0, `density: f32` at 12, 16 bytes total.
        assert_eq!(std::mem::size_of::<FogUniform>(), 16);
        assert_eq!(std::mem::offset_of!(FogUniform, color), 0);
        assert_eq!(std::mem::offset_of!(FogUniform, density), 12);
        assert_eq!(FogUniform::FIELD_NAMES, &["color", "density"]);
        assert!(
            FogUniform::WGSL_SOURCE.contains("color: vec3<f32>"),
            "{}",
            FogUniform::WGSL_SOURCE
        );
        assert!(
            FogUniform::WGSL_SOURCE.contains("density: f32"),
            "{}",
            FogUniform::WGSL_SOURCE
        );
        let packed = FogUniform::pack([0.5, 0.6, 0.7], 0.1);
        let bytes = bytemuck::bytes_of(&packed);
        assert_eq!(bytes.len(), 16);
        assert_eq!(
            &bytes[0..12],
            bytemuck::cast_slice::<f32, u8>(&[0.5, 0.6, 0.7])
        );
    }

    #[test]
    fn ibl_multipliers_scale_upload_and_default_is_noop() {
        let lights = vec![LightDesc::Directional {
            direction: [1.0, 1.0, 1.0],
            intensity: 2.0,
            color: [0.5, 0.25, 0.125],
            shadow: ShadowCast::Disabled,
        }];
        let base = build_lighting_uniform([0.5, 0.25, 0.125], 1.0, 1.0, &lights, None);
        assert_eq!(base.uniform.ambient_color, [0.5, 0.25, 0.125, 1.0]);
        assert_eq!(base.uniform.lights[0].color, [0.5, 0.25, 0.125, 2.0]);
        let scaled = build_lighting_uniform([0.5, 0.25, 0.125], 2.0, 4.0, &lights, None);
        assert_eq!(scaled.uniform.ambient_color, [1.0, 0.5, 0.25, 1.0]);
        assert_eq!(scaled.uniform.lights[0].color, [2.0, 1.0, 0.5, 2.0]);
    }

    #[test]
    fn staging_reserve_is_exact_and_stable_at_same_size() {
        // The upload path reserves once up front: exact fit for the
        // frame, and re-reserving the same size never reallocates.
        assert_eq!(staging_capacity_for_instances(300), 300);
        let mut staging: Vec<PerObjectGpu> =
            Vec::with_capacity(staging_capacity_for_instances(300));
        assert!(staging.capacity() >= 300);
        let capacity = staging.capacity();
        staging.reserve(staging_capacity_for_instances(300).saturating_sub(staging.len()));
        assert_eq!(staging.capacity(), capacity, "same size: no realloc");
    }

    /// Adapter handle, or `None` on headless CI without a GPU.
    fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::empty(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok()?;
            adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .ok()
        })
    }

    /// GPU/CPU parity smoke for the enabled fog mix: a solid HDR layer over
    /// a cleared (far-plane) depth buffer, fogged on the GPU, must land
    /// within tolerance of [`crate::frame_passes::apply_fog`] fed with the
    /// same view-space distance the shader reconstructs.
    ///
    /// The fresh `Renderer3D` camera is the identity (eye at the origin),
    /// so the reconstruction is exact on paper: NDC `(u*2-1, 1-v*2, 1)`
    /// maps to itself and the distance is its length. Skipped when no
    /// adapter is available.
    #[test]
    fn fog_enabled_gpu_matches_cpu_apply_fog() {
        const W: u32 = 32;
        const H: u32 = 2;
        const BPP: u32 = 4;
        const ROW: u32 = W * BPP; // 128 — under the 256 copy alignment…
        // …so pad rows to 256 bytes for the readback copy.
        const PADDED_ROW: u32 = 256;
        const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        const INPUT: [f32; 3] = [0.2, 0.4, 0.6];
        const FOG_COLOR: [f32; 3] = [0.9, 0.1, 0.1];
        const DENSITY: f32 = 3.0;
        // Tolerance covers u8 quantization (1/255) plus f32 exp/MAD
        // ordering between CPU and GPU.
        const TOL: f32 = 0.03;

        fn run_case(density: f32) -> Option<[f32; 3]> {
            let (device, queue) = try_device()?;
            let surface_config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: FMT,
                width: W,
                height: H,
                present_mode: wgpu::PresentMode::AutoNoVsync,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
            };
            let renderer = Renderer3D::new(&device, &surface_config, 1);

            let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            let input_byte = [byte(INPUT[0]), byte(INPUT[1]), byte(INPUT[2]), 255];
            let extent = wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            };
            let hdr_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fog parity hdr"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FMT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let mut hdr_data = vec![0u8; (ROW * H) as usize];
            for px in hdr_data.chunks_exact_mut(4) {
                px.copy_from_slice(&input_byte);
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &hdr_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &hdr_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW),
                    rows_per_image: Some(H),
                },
                extent,
            );
            // Far-plane depth: clear-only pass over a depth texture.
            let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fog parity depth"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let target_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fog parity target"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FMT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let target = target_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let hdr = hdr_tex.create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fog parity encoder"),
            });
            // Initialize both attachments first (`render_fog` loads the
            // target instead of clearing it).
            {
                let clear_depth = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("fog parity init"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &clear_depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            renderer.render_fog(
                &device,
                &queue,
                &mut encoder,
                FogInputs {
                    hdr: &hdr,
                    depth: &depth_view,
                    target: &target,
                    color: FOG_COLOR,
                    density,
                },
            );
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fog parity readback"),
                size: (PADDED_ROW * H) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &target_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(PADDED_ROW),
                        rows_per_image: Some(H),
                    },
                },
                extent,
            );
            queue.submit([encoder.finish()]);

            let slice = buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .ok();
            rx.recv().ok()?.ok()?;
            let view = slice.get_mapped_range().unwrap();
            // Center-ish texel (x=16, y=1): deterministic uv, same math
            // the CPU reference uses below.
            let px = (PADDED_ROW + 16 * BPP) as usize;
            let pixel = [
                view[px] as f32 / 255.0,
                view[px + 1] as f32 / 255.0,
                view[px + 2] as f32 / 255.0,
            ];
            Some(pixel)
        }

        // Reconstructed distance for texel (16, 1) under the identity
        // camera: uv = ((16+0.5)/32, (1+0.5)/2), NDC z = 1 (cleared far).
        let u = (16.0 + 0.5) / W as f32;
        let v = (1.0 + 0.5) / H as f32;
        let dist = ((2.0 * u - 1.0).powi(2) + (1.0 - 2.0 * v).powi(2) + 1.0).sqrt();
        let fog = crate::frame_passes::FogState::Enabled(
            crate::frame_passes::FogSettings::try_from_raw(FOG_COLOR, DENSITY)
                .expect("positive density"),
        );
        let expected = crate::frame_passes::apply_fog(INPUT, dist, fog);

        let Some(px) = run_case(DENSITY) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        for i in 0..3 {
            assert!(
                (px[i] - expected[i]).abs() <= TOL,
                "channel {i}: gpu={px:?} cpu={expected:?} (dist={dist})"
            );
        }
        // High density visibly moved toward the fog color…
        for i in 0..3 {
            assert!(
                (px[i] - FOG_COLOR[i]).abs() < (INPUT[i] - FOG_COLOR[i]).abs(),
                "no fog movement: {px:?}"
            );
        }
        // …while a near-zero density keeps the input (disabled-adjacent).
        let faint = crate::frame_passes::FogState::Enabled(
            crate::frame_passes::FogSettings::try_from_raw(FOG_COLOR, 1.0e-4)
                .expect("positive density"),
        );
        let faint_expected = crate::frame_passes::apply_fog(INPUT, dist, faint);
        let Some(faint_px) = run_case(1.0e-4) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        for i in 0..3 {
            assert!(
                (faint_px[i] - faint_expected[i]).abs() <= TOL,
                "faint channel {i}: gpu={faint_px:?} cpu={faint_expected:?}"
            );
            assert!(
                (faint_px[i] - INPUT[i]).abs() < 0.02,
                "near-zero density drifted: {faint_px:?}"
            );
        }
    }
}
