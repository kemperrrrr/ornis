//! HDR composite pass on the stable Rust → LLVM IR → WGSL path.
//!
//! The pass draws one fullscreen quad, mixes the deferred and forward layers by
//! the bloom mode, adds the bloom layer scaled by its intensity, applies the
//! auto-exposure multiplier, and tone maps the result with the ACES fit. The
//! four modes and the sizes live in uniforms, so the shader is a straight-line
//! mix with one branch on the mode word.
//!
//! `xtask build-shaders --package ornis-gpu-hdr-composite --entry all` compiles
//! the crate for `wasm32-unknown-unknown` and writes the checked-in
//! `crates/render/src/shaders/hdr_composite_generated.wgsl`.

#![no_std]
#![warn(missing_docs)]

use ornis_gpu_quad::standard_quad;
use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::Sampler;
use ornis_gpu_shader::Texture2d;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::fragment_input_lane;
use ornis_gpu_shader::fragment_store_at;
use ornis_gpu_shader::gpu_entry;

/// Composite sampler binding.
const COMPOSITE_SAMPLER: u32 = 2;

/// Lane of the forward layer's alpha, the coverage weight.
const ALPHA_LANE: u32 = 3;

/// Field index of the bloom intensity in [`BloomParams`].
const BLOOM_INTENSITY: usize = 1;
/// Field index of the bloom mode word in [`BloomParams`].
const BLOOM_MODE: usize = 2;

/// Field index of the exposure multiplier in [`ExposureParams`].
const EXPOSURE: usize = 0;

/// ACES fit coefficients (`x * (a·x + b) / (x · (c·x + d) + e)`).
const ACES_FIT_A: f32 = 2.51;
/// See [`ACES_FIT_A`].
const ACES_FIT_B: f32 = 0.03;
/// See [`ACES_FIT_A`].
const ACES_FIT_C: f32 = 2.43;
/// See [`ACES_FIT_A`].
const ACES_FIT_D: f32 = 0.59;
/// See [`ACES_FIT_A`].
const ACES_FIT_E: f32 = 0.14;

/// Bloom mode: forward layer only, weighted by its alpha.
const BLOOM_MODE_FORWARD: u32 = 1;
/// Bloom mode: deferred plus the forward layer.
const BLOOM_MODE_ADDITIVE: u32 = 2;

/// `BloomParams`: threshold, intensity, mode word, pad.
///
/// The pad word is part of the host layout; the shader reads three fields.
pub struct BloomParams;

impl GpuStruct for BloomParams {
    const ID: u32 = 0;
    const FIELD_TAGS: &'static [u8] = &[
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_U32,
        ornis_gpu_shader::TYPE_TAG_F32,
    ];
}

/// `ExposureParams`: the multiplier plus three pad words.
pub struct ExposureParams;

impl GpuStruct for ExposureParams {
    const ID: u32 = 1;
    const FIELD_TAGS: &'static [u8] = &[
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_F32,
    ];
}

/// One lane of the ACES fit.
#[inline(always)]
fn aces_lane(value: f32) -> f32 {
    value * (ACES_FIT_A * value + ACES_FIT_B)
        / (value * (ACES_FIT_C * value + ACES_FIT_D) + ACES_FIT_E)
}

/// The layer mix for one channel, then bloom, exposure, and ACES.
///
/// `deferred_weight` and `forward_weight` are the pulse of the mode word: mode
/// 1 keeps only the forward layer, mode 2 keeps both, mode 0 keeps the deferred
/// layer alone. Both are computed once per invocation, so the branch does not
/// repeat per channel.
#[inline(always)]
fn layer_mix(deferred: f32, forward: f32, deferred_weight: f32, forward_weight: f32) -> f32 {
    deferred * deferred_weight + forward * forward_weight
}

/// One tone-mapped channel: `(combined + bloom * intensity) * exposure` through
/// the ACES fit.
#[inline(always)]
fn tone_map(combined: f32, bloom: f32, intensity: f32, exposure: f32) -> f32 {
    aces_lane((combined + bloom * intensity) * exposure)
}

/// HDR composite vertex stage: the quad, straight from `vertex_index`.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "vs_main")]`.
#[gpu_entry(vertex)]
pub fn vs_main() {
    standard_quad();
}

/// HDR composite fragment stage: layer mix, bloom, exposure, ACES.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "fs_main")]`.
// qual:allow(srp, max_parameters=6) reason: "the pass binds six resources (three layers, one sampler, two uniforms); the SDK binds each on its own parameter, and folding them into a fake bundle would only move the list"
#[gpu_entry(fragment)]
pub fn fs_main(
    #[gpu(texture_2d, group = 0, binding = 0)] deferred_tex: Texture2d,
    #[gpu(texture_2d, group = 0, binding = 1)] forward_tex: Texture2d,
    #[gpu(sampler, group = 0, binding = 2)] composite_sampler: Sampler,
    #[gpu(texture_2d, group = 0, binding = 3)] bloom_tex: Texture2d,
    #[gpu(uniform, group = 0, binding = 4)] bloom_params: Uniform<BloomParams>,
    #[gpu(uniform, group = 0, binding = 5)] exposure_params: Uniform<ExposureParams>,
) {
    let u = fragment_input_lane::<0, 2, 0>();
    let v = fragment_input_lane::<0, 2, 1>();
    let mode = bloom_params.field_u32::<BLOOM_MODE>();
    let (deferred_weight, forward_weight) = if mode == BLOOM_MODE_FORWARD {
        (0.0, 1.0)
    } else if mode == BLOOM_MODE_ADDITIVE {
        (1.0, 1.0)
    } else {
        (1.0, 0.0)
    };
    let alpha = forward_tex.sample_lane::<0, { COMPOSITE_SAMPLER }, { ALPHA_LANE }>(
        composite_sampler,
        u,
        v,
    );
    let intensity = bloom_params.field_f32::<BLOOM_INTENSITY>();
    let exposure = exposure_params.field_f32::<EXPOSURE>();
    let combined_r = layer_mix(
        deferred_tex.sample_lane::<0, { COMPOSITE_SAMPLER }, 0>(composite_sampler, u, v),
        forward_tex.sample_lane::<0, { COMPOSITE_SAMPLER }, 0>(composite_sampler, u, v) * alpha,
        deferred_weight,
        forward_weight,
    );
    let combined_g = layer_mix(
        deferred_tex.sample_lane::<0, { COMPOSITE_SAMPLER }, 1>(composite_sampler, u, v),
        forward_tex.sample_lane::<0, { COMPOSITE_SAMPLER }, 1>(composite_sampler, u, v) * alpha,
        deferred_weight,
        forward_weight,
    );
    let combined_b = layer_mix(
        deferred_tex.sample_lane::<0, { COMPOSITE_SAMPLER }, 2>(composite_sampler, u, v),
        forward_tex.sample_lane::<0, { COMPOSITE_SAMPLER }, 2>(composite_sampler, u, v) * alpha,
        deferred_weight,
        forward_weight,
    );
    let r = tone_map(
        combined_r,
        bloom_tex.sample_lane::<0, { COMPOSITE_SAMPLER }, 0>(composite_sampler, u, v),
        intensity,
        exposure,
    );
    let g = tone_map(
        combined_g,
        bloom_tex.sample_lane::<0, { COMPOSITE_SAMPLER }, 1>(composite_sampler, u, v),
        intensity,
        exposure,
    );
    let b = tone_map(
        combined_b,
        bloom_tex.sample_lane::<0, { COMPOSITE_SAMPLER }, 2>(composite_sampler, u, v),
        intensity,
        exposure,
    );
    fragment_store_at::<0>(r, g, b, 1.0);
}
