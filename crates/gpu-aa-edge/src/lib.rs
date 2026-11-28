//! FXAA/SMAA edge filter on the stable Rust → LLVM IR → WGSL path.
//!
//! One shader serves both filters: the luma gradient says where an edge runs,
//! the two samples across the dominant axis are blended into the centre, and
//! the mode word in the uniform picks how the taps are weighted — FXAA treats
//! them evenly, SMAA by how close each tap's luma is to the centre. A pixel
//! whose gradient is below the threshold is returned unchanged.
//!
//! `xtask build-shaders --package ornis-gpu-aa-edge --entry all` writes the
//! checked-in `crates/render/src/shaders/aa_edge_generated.wgsl`.

#![no_std]
#![warn(missing_docs)]

use ornis_gpu_quad::standard_quad;
use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::Sampler;
use ornis_gpu_shader::Texture2d;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::abs;
use ornis_gpu_shader::clamp;
use ornis_gpu_shader::fragment_input_lane;
use ornis_gpu_shader::fragment_store_at;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::mix;

/// Sampler binding of the source layer.
///
/// The `#[gpu(...)]` attribute needs a literal, so the same number appears
/// there; this constant is what the sample calls name.
const SRC_SAMPLER: u32 = 1;

/// Field of the luma threshold in [`AaParams`].
const THRESHOLD: usize = 1;
/// Field of the mode word in [`AaParams`].
const MODE: usize = 4;

/// Weight of each of the four taps in the FXAA blend: they average evenly.
const FXAA_TAP_WEIGHT: f32 = 0.25;

/// Mode word of the SMAA blend.
const MODE_SMAA: u32 = 1;

/// Luma weights of the filter (Rec. 601, the coefficients `aa_luma` uses).
const LUMA_R: f32 = 0.299;
/// See [`LUMA_R`].
const LUMA_G: f32 = 0.587;
/// See [`LUMA_R`].
const LUMA_B: f32 = 0.114;

/// `AaParams`: history feedback, threshold, clamp radius, history flag, mode,
/// pad.
pub struct AaParams;

impl GpuStruct for AaParams {
    const ID: u32 = 0;
    const FIELD_TAGS: &'static [u8] = &[
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_U32,
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_F32,
        ornis_gpu_shader::TYPE_TAG_F32,
    ];
}

/// Filter luma of one colour, the same weights `aa_luma` uses.
#[inline(always)]
fn luma(r: f32, g: f32, b: f32) -> f32 {
    LUMA_R * r + LUMA_G * g + LUMA_B * b
}

/// Larger of two values.
#[inline(always)]
fn largest(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// Edge filter vertex stage: the shared fullscreen quad.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "vs_main")]`.
#[gpu_entry(vertex)]
pub fn vs_main() {
    standard_quad();
}

/// Edge filter fragment stage: FXAA or SMAA, by the mode word.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "fs_main")]`.
#[gpu_entry(fragment)]
pub fn fs_main(
    #[gpu(texture_2d, group = 0, binding = 0)] src_tex: Texture2d,
    #[gpu(sampler, group = 0, binding = 1)] src_sampler: Sampler,
    #[gpu(uniform, group = 0, binding = 2)] aa_params: Uniform<AaParams>,
) {
    let u = fragment_input_lane::<0, 2, 0>();
    let v = fragment_input_lane::<0, 2, 1>();
    let texel_x = 1.0 / src_tex.width() as f32;
    let texel_y = 1.0 / src_tex.height() as f32;
    // Centre and the four neighbours across both axes, one lane at a time.
    let center_r = src_tex.sample_lane::<0, { SRC_SAMPLER }, 0>(src_sampler, u, v);
    let center_g = src_tex.sample_lane::<0, { SRC_SAMPLER }, 1>(src_sampler, u, v);
    let center_b = src_tex.sample_lane::<0, { SRC_SAMPLER }, 2>(src_sampler, u, v);
    let north_r = src_tex.sample_lane::<0, { SRC_SAMPLER }, 0>(src_sampler, u, v - texel_y);
    let north_g = src_tex.sample_lane::<0, { SRC_SAMPLER }, 1>(src_sampler, u, v - texel_y);
    let north_b = src_tex.sample_lane::<0, { SRC_SAMPLER }, 2>(src_sampler, u, v - texel_y);
    let south_r = src_tex.sample_lane::<0, { SRC_SAMPLER }, 0>(src_sampler, u, v + texel_y);
    let south_g = src_tex.sample_lane::<0, { SRC_SAMPLER }, 1>(src_sampler, u, v + texel_y);
    let south_b = src_tex.sample_lane::<0, { SRC_SAMPLER }, 2>(src_sampler, u, v + texel_y);
    let east_r = src_tex.sample_lane::<0, { SRC_SAMPLER }, 0>(src_sampler, u + texel_x, v);
    let east_g = src_tex.sample_lane::<0, { SRC_SAMPLER }, 1>(src_sampler, u + texel_x, v);
    let east_b = src_tex.sample_lane::<0, { SRC_SAMPLER }, 2>(src_sampler, u + texel_x, v);
    let west_r = src_tex.sample_lane::<0, { SRC_SAMPLER }, 0>(src_sampler, u - texel_x, v);
    let west_g = src_tex.sample_lane::<0, { SRC_SAMPLER }, 1>(src_sampler, u - texel_x, v);
    let west_b = src_tex.sample_lane::<0, { SRC_SAMPLER }, 2>(src_sampler, u - texel_x, v);
    let luma_center = luma(center_r, center_g, center_b);
    let luma_north = luma(north_r, north_g, north_b);
    let luma_south = luma(south_r, south_g, south_b);
    let luma_east = luma(east_r, east_g, east_b);
    let luma_west = luma(west_r, west_g, west_b);
    let gradient_x = luma_east - luma_west;
    let gradient_y = luma_south - luma_north;
    let range = largest(abs(gradient_x), abs(gradient_y));
    let threshold = aa_params.field_f32::<THRESHOLD>();
    if range <= threshold {
        fragment_store_at::<0>(center_r, center_g, center_b, 1.0);
        return;
    }
    // The edge runs across the dominant gradient: blend along it.
    let along_x = abs(gradient_x) > abs(gradient_y);
    let tap_a_r = if along_x { east_r } else { south_r };
    let tap_a_g = if along_x { east_g } else { south_g };
    let tap_a_b = if along_x { east_b } else { south_b };
    let tap_b_r = if along_x { west_r } else { north_r };
    let tap_b_g = if along_x { west_g } else { north_g };
    let tap_b_b = if along_x { west_b } else { north_b };
    let mixed_r = (center_r + center_r + tap_a_r + tap_b_r) * FXAA_TAP_WEIGHT;
    let mixed_g = (center_g + center_g + tap_a_g + tap_b_g) * FXAA_TAP_WEIGHT;
    let mixed_b = (center_b + center_b + tap_a_b + tap_b_b) * FXAA_TAP_WEIGHT;
    // SMAA weights the taps by how close their luma is to the centre, so the
    // sample on the centre's side keeps the blend and the other fades out.
    let mode = aa_params.field_u32::<MODE>();
    let (blended_r, blended_g, blended_b) = if mode == MODE_SMAA {
        let weight_a = 1.0 / (1.0 + abs(luma(tap_a_r, tap_a_g, tap_a_b) - luma_center));
        let weight_b = 1.0 / (1.0 + abs(luma(tap_b_r, tap_b_g, tap_b_b) - luma_center));
        let total = weight_a + weight_b;
        (
            (tap_a_r * weight_a + tap_b_r * weight_b) / total,
            (tap_a_g * weight_a + tap_b_g * weight_b) / total,
            (tap_a_b * weight_a + tap_b_b * weight_b) / total,
        )
    } else {
        (mixed_r, mixed_g, mixed_b)
    };
    let weight = clamp((range - threshold) / range, 0.0, 1.0);
    fragment_store_at::<0>(
        mix(center_r, blended_r, weight),
        mix(center_g, blended_g, weight),
        mix(center_b, blended_b, weight),
        1.0,
    );
}
