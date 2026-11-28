//! Shading constants shared by the render passes.
//!
//! The values live here once so the CPU mirrors (`msaa_edge`, the cluster
//! CPU reference, the CSM tests) and the shader kernels cannot drift: the
//! translated kernel crates take the same names from their own side, and the
//! tests below pin the numeric agreement. Nothing here is printed into WGSL
//! any more — the last DSL `const` block went with the quad vertex stage.
//!
//! The layer evaluators that used to live here (`evaluate_*`,
//! `transmission_btdf`, `csm_visibility`, `octahedral_decode`,
//! `reconstruct_world_pos`, `cluster_light_span`) moved to the translated
//! kernel crates — `ornis-gpu-openpbr`, `ornis-gpu-pbr` and
//! `ornis-gpu-lighting` — so nothing in this module is a DSL body any more.

/// Single-precision π, matching the former `3.14159265359` literal bit-wise.
pub const PI: f32 = std::f32::consts::PI;
/// Epsilon guard against division by zero (former `1e-6`).
pub const EPS: f32 = 1e-6;
/// Reference-depth bias subtracted in the shadow evaluators (former
/// `0.002` literal, now shared by the 2D and cube branches of both
/// entries): keeps the lit surface's own texels on the lit side of
/// the compare. Larger values erode small blobs uniformly; smaller
/// values risk acne (guarded by
/// `shadowed_directional_without_occluder_has_no_acne`).
pub const SHADOW_REF_BIAS: f32 = 0.001;
/// Fraction of a cascade's depth range blended with the next cascade at
/// the upper seam (#121): `1` deep inside the cascade, `0` exactly on
/// the split. Small enough to keep the sharp near texels, wide enough
/// to hide the resolution step between cascades.
pub const CSM_BLEND_FRACTION: f32 = 0.08;
/// `step` edge on a light's cascade count: at or above this, the light
/// samples the CSM pool instead of the legacy single shadow map.
pub const CSM_ACTIVE_EDGE: f32 = 0.5;
/// Index of the last cascade. It is sampled once; every earlier cascade
/// may blend into the next one.
pub const CSM_LAST_CASCADE: u32 = 3;
/// Single-precision 1/π, matching the former `0.31830988618` bit-wise.
pub const INV_PI: f32 = std::f32::consts::FRAC_1_PI;
/// Cleared g-buffer depth (far plane). A sample at this depth is background:
/// its octahedral clear `(0, 0)` decodes to +Z and must not be shaded.
pub const CLEAR_DEPTH: f32 = 1.0;
/// Dot product above which two sample normals are the same surface.
pub const NORMAL_AGREE: f32 = 0.999;
/// NDC depth gap that splits two samples onto different surfaces.
pub const DEPTH_EDGE: f32 = 0.002;
/// Alpha below which a deferred pixel is absent (former `0.001` discard).
pub const ALPHA_CUTOFF: f32 = 0.001;
/// `MSAA_SAMPLE_COUNT` as `f32`, for coverage comparisons in WGSL.
pub const MSAA_SAMPLES_F: f32 = 4.0;
/// `step` edge between a directional light (kind 0) and a point or spot (kind ≥ 1).
pub const POINT_OR_SPOT_KIND_EDGE: f32 = 0.5;
/// `step` edge between a point light (kind 1) and a spot (kind 2).
pub const SPOT_KIND_EDGE: f32 = 1.5;
/// Inner fraction of a finite light's range where attenuation starts to fall.
///
/// `smoothstep` with equal edges is undefined in WGSL, so the cutoff begins
/// this far inside the light range.
pub const RANGE_CUTOFF_INNER_FRACTION: f32 = 0.99;
/// Half-extent that maps NDC `[-1, 1]` onto a texture UV `[0, 1]`.
pub const NDC_TO_UV_HALF: f32 = 0.5;
/// Threshold separating valid previous-position `w` from the invalid marker.
pub const MOTION_VALID_THRESHOLD: f32 = NDC_TO_UV_HALF;
/// Round-to-nearest bias for packing normalized depth into a 16-bit code.
pub const DEPTH_ROUNDING_BIAS: f32 = NDC_TO_UV_HALF;
/// Largest depth code that still represents a valid motion sample.
pub const PREVIOUS_DEPTH_MAX: f32 = (u16::MAX - 1) as f32;
/// Reserved 16-bit marker for pixels without a valid previous position.
pub const PREVIOUS_DEPTH_INVALID: u32 = u16::MAX as u32;
/// Near plane of a point-light cube-shadow face. The far plane is the light range.
pub const SHADOW_CUBE_NEAR: f32 = 0.1;
/// Split on a 0/1 light selector: at or below this, the light is not a point light.
pub const LIGHT_SELECTOR_SPLIT: f32 = 0.5;
/// Screen-space normal variance scale for geometric specular AA (Filament `SIGMA2`).
pub const SPECULAR_AA_VARIANCE: f32 = 0.15;
/// Scale from normal variance to the GGX kernel (`2 * variance` in Filament).
pub const SPECULAR_AA_KERNEL_SCALE: f32 = 2.0;
/// Cap on the extra GGX kernel geometric specular AA may add (Filament `KAPPA`).
pub const SPECULAR_AA_KERNEL_CAP: f32 = 0.2;
/// Bias in the Karis firefly weight `1 / (bias + luma)` used on MSAA edges.
pub const KARIS_LUMA_BIAS: f32 = 1.0;
/// Magnitude past which an HDR channel is treated as non-finite and zeroed.
pub const HDR_FINITE_LIMIT: f32 = 1.0e20;
// The `ShadingDebug` discriminants live in `renderer::lights` (one source);
// the shader side has its own copy inside `ornis-gpu-openpbr`.
/// Pre-tonemap luma above which the heat view paints the pixel red.
pub const SHADING_DEBUG_HOT_LUMA: f32 = 1.0;
/// GPU encoding threshold for the unlit shading-mode flag
/// (`geometry params[2] >= threshold` = unlit). Single source of truth
/// lives in [`ornis_core::material::UNLIT_THRESHOLD`]; this re-export
/// puts the same-named `const` into the WGSL block below, so the CPU
/// decode and both GPU evaluators agree by construction.
pub const UNLIT_THRESHOLD: f32 = ornis_core::material::UNLIT_THRESHOLD;
