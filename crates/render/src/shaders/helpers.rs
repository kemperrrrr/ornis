//! Shading constants shared by the render passes.
//!
//! The values live here once so the CPU mirrors (`msaa_edge`, the cluster
//! CPU reference, the CSM tests) and the WGSL const block cannot drift: each
//! one is declared as Rust and printed by [`wgsl_consts`] through `f32_lit`,
//! which round-trips bit-exactly.
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
/// [`crate::renderer::ShadingDebug::Beauty`].
pub const SHADING_DEBUG_BEAUTY: u32 = 0;
/// [`crate::renderer::ShadingDebug::DirectSpecular`].
pub const SHADING_DEBUG_DIRECT_SPECULAR: u32 = 1;
/// [`crate::renderer::ShadingDebug::IblSpecular`].
pub const SHADING_DEBUG_IBL_SPECULAR: u32 = 2;
/// [`crate::renderer::ShadingDebug::Diffuse`].
pub const SHADING_DEBUG_DIFFUSE: u32 = 3;
/// [`crate::renderer::ShadingDebug::Roughness`].
pub const SHADING_DEBUG_ROUGHNESS: u32 = 4;
/// [`crate::renderer::ShadingDebug::Metallic`].
pub const SHADING_DEBUG_METALLIC: u32 = 5;
/// [`crate::renderer::ShadingDebug::Shadow`].
pub const SHADING_DEBUG_SHADOW: u32 = 6;
/// [`crate::renderer::ShadingDebug::Heat`].
pub const SHADING_DEBUG_HEAT: u32 = 7;
/// Pre-tonemap luma above which the heat view paints the pixel red.
pub const SHADING_DEBUG_HOT_LUMA: f32 = 1.0;
/// GPU encoding threshold for the unlit shading-mode flag
/// (`geometry params[2] >= threshold` = unlit). Single source of truth
/// lives in [`ornis_core::material::UNLIT_THRESHOLD`]; this re-export
/// puts the same-named `const` into the WGSL block below, so the CPU
/// decode and both GPU evaluators agree by construction.
pub const UNLIT_THRESHOLD: f32 = ornis_core::material::UNLIT_THRESHOLD;

/// WGSL `const` block for the helpers, generated from the Rust constants
/// above: the name travels via `stringify!` (rename-proof), the value via
/// `f32_lit` (bit-exact, WGSL-valid). No second spelling.
pub fn wgsl_consts() -> String {
    macro_rules! decl {
        ($name:ident) => {
            format!(
                "const {}: f32 = {};\n",
                stringify!($name),
                super::f32_lit($name)
            )
        };
    }
    let floats = [
        decl!(PI),
        decl!(EPS),
        decl!(INV_PI),
        decl!(SHADOW_REF_BIAS),
        decl!(CLEAR_DEPTH),
        decl!(NORMAL_AGREE),
        decl!(DEPTH_EDGE),
        decl!(ALPHA_CUTOFF),
        decl!(MSAA_SAMPLES_F),
        decl!(POINT_OR_SPOT_KIND_EDGE),
        decl!(SPOT_KIND_EDGE),
        decl!(RANGE_CUTOFF_INNER_FRACTION),
        decl!(NDC_TO_UV_HALF),
        decl!(MOTION_VALID_THRESHOLD),
        decl!(DEPTH_ROUNDING_BIAS),
        decl!(PREVIOUS_DEPTH_MAX),
        decl!(SHADOW_CUBE_NEAR),
        decl!(CSM_BLEND_FRACTION),
        decl!(CSM_ACTIVE_EDGE),
        decl!(LIGHT_SELECTOR_SPLIT),
        decl!(SPECULAR_AA_VARIANCE),
        decl!(SPECULAR_AA_KERNEL_SCALE),
        decl!(SPECULAR_AA_KERNEL_CAP),
        decl!(KARIS_LUMA_BIAS),
        decl!(HDR_FINITE_LIMIT),
        decl!(SHADING_DEBUG_HOT_LUMA),
        decl!(UNLIT_THRESHOLD),
    ]
    .concat();
    macro_rules! decl_u32 {
        ($name:ident) => {
            format!("const {}: u32 = {}u;\n", stringify!($name), $name)
        };
    }
    let ints = [
        decl_u32!(SHADING_DEBUG_BEAUTY),
        decl_u32!(SHADING_DEBUG_DIRECT_SPECULAR),
        decl_u32!(SHADING_DEBUG_IBL_SPECULAR),
        decl_u32!(SHADING_DEBUG_DIFFUSE),
        decl_u32!(SHADING_DEBUG_ROUGHNESS),
        decl_u32!(SHADING_DEBUG_METALLIC),
        decl_u32!(SHADING_DEBUG_SHADOW),
        decl_u32!(SHADING_DEBUG_HEAT),
        decl_u32!(CSM_LAST_CASCADE),
        decl_u32!(PREVIOUS_DEPTH_INVALID),
    ]
    .concat();
    // Cluster grid constants (#130): formatted from the single renderer
    // source (`renderer::clusters`), so the CPU reference, this consts
    // block and the assignment compute shader cannot drift.
    let cluster = format!(
        "const CLUSTERS_X: u32 = {x}u;\n\
         const CLUSTERS_Y: u32 = {y}u;\n\
         const CLUSTERS_Z: u32 = {z}u;\n\
         const CLUSTER_DEPTH_NEAR: f32 = {near};\n\
         const CLUSTER_SLICE_SCALE: f32 = {scale};\n\
         const CLUSTER_SLICE_LOOKUP_EPS: f32 = {eps};\n",
        x = crate::renderer::CLUSTER_TILE_COUNT_X,
        y = crate::renderer::CLUSTER_TILE_COUNT_Y,
        z = crate::renderer::CLUSTER_SLICE_COUNT,
        near = super::f32_lit(crate::renderer::CLUSTER_DEPTH_NEAR),
        scale = super::f32_lit(crate::renderer::cluster_slice_scale()),
        eps = super::f32_lit(crate::renderer::CLUSTER_SLICE_LOOKUP_EPS),
    );
    format!("{floats}{ints}{cluster}{}", msaa_samples_wgsl())
}

/// `u32` sample count shared with [`crate::renderer::MSAA_SAMPLE_COUNT`].
fn msaa_samples_wgsl() -> String {
    format!(
        "const MSAA_SAMPLES: u32 = {}u;\n",
        crate::renderer::MSAA_SAMPLE_COUNT
    )
}
