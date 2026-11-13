//! Bloom bright-pass on the stable Rust → LLVM IR → WGSL path.
//!
//! The pass keeps the part of the frame above a luminance threshold, with a
//! soft knee so the cut does not band: `smoothstep(threshold, threshold + knee,
//! luma)` scales the sampled colour. The same shader runs for the downsample
//! and upsample stages, so the threshold and knee come from the uniform the
//! renderer writes.
//!
//! `xtask build-shaders --package ornis-gpu-bloom --entry all` writes the
//! checked-in `crates/render/src/shaders/bloom_generated.wgsl`.

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
use ornis_gpu_shader::smoothstep;

/// Sampler binding of the source layer.
///
/// The `#[gpu(...)]` attribute needs a literal, so the same number appears
/// there; this constant is what the sample calls name.
const SRC_SAMPLER: u32 = 1;
/// Field of the luminance threshold in [`BloomParams`].
const THRESHOLD: usize = 0;

/// Rec. 709 luminance weights, the coefficients the host's `luminance` kernel
/// uses.
const REC709_LUMA_R: f32 = 0.2126;
/// See [`REC709_LUMA_R`].
const REC709_LUMA_G: f32 = 0.7152;
/// See [`REC709_LUMA_R`].
const REC709_LUMA_B: f32 = 0.0722;

/// Width of the bright-pass soft knee above the threshold.
const BLOOM_SOFT_KNEE: f32 = 0.05;

/// `BloomParams`: threshold, intensity, mode word, pad.
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

/// Bloom vertex stage: the shared fullscreen quad.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "vs_main")]`.
#[gpu_entry(vertex)]
pub fn vs_main() {
    standard_quad();
}

/// Bloom fragment stage: keep what is brighter than the knee, scaled by it.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "fs_main")]`.
#[gpu_entry(fragment)]
pub fn fs_main(
    #[gpu(texture_2d, group = 0, binding = 0)] src_tex: Texture2d,
    #[gpu(sampler, group = 0, binding = 1)] src_sampler: Sampler,
    #[gpu(uniform, group = 0, binding = 2)] bloom_params: Uniform<BloomParams>,
) {
    let u = fragment_input_lane::<0, 2, 0>();
    let v = fragment_input_lane::<0, 2, 1>();
    let r = src_tex.sample_lane::<0, { SRC_SAMPLER }, 0>(src_sampler, u, v);
    let g = src_tex.sample_lane::<0, { SRC_SAMPLER }, 1>(src_sampler, u, v);
    let b = src_tex.sample_lane::<0, { SRC_SAMPLER }, 2>(src_sampler, u, v);
    // Rec. 709 luminance, one `dot` like the host's `luminance` kernel.
    let luma = r * REC709_LUMA_R + g * REC709_LUMA_G + b * REC709_LUMA_B;
    let threshold = bloom_params.field_f32::<THRESHOLD>();
    let keep = smoothstep(threshold, threshold + BLOOM_SOFT_KNEE, luma);
    fragment_store_at::<0>(r * keep, g * keep, b * keep, 1.0);
}
