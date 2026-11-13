//! UI composite pass on the stable Rust → LLVM IR → WGSL path.
//!
//! The pass draws one fullscreen quad and blends the PBR colour with the UI
//! layer: the UI texel is decoded from sRGB to linear light and mixed over the
//! scene colour by its alpha. Vertices come from two module constants indexed
//! by `@builtin(vertex_index)`; the fragment reads both layers through
//! `textureSampleLevel` at level 0 and writes `@location(0)`.
//!
//! The crate is an ordinary `no_std` library: `xtask build-shaders` compiles it
//! for `wasm32-unknown-unknown` with `--emit=llvm-ir` and translates the IR into
//! the checked-in `crates/render/src/shaders/composite_generated.wgsl`.

#![no_std]
#![warn(missing_docs)]

use ornis_gpu_shader::Sampler;
use ornis_gpu_shader::Texture2d;
use ornis_gpu_shader::fragment_input_lane;
use ornis_gpu_shader::fragment_store_at;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::mix;
use ornis_gpu_shader::pow;
use ornis_gpu_shader::vertex_index;
use ornis_gpu_shader::vertex_position;
use ornis_gpu_shader::vertex_store_2;

/// Group-0 binding of the scene colour sampler.
///
/// The `#[gpu(...)]` attribute needs a literal, so the same number appears
/// there; this constant is what the sample calls name.
const PBR_SAMPLER: u32 = 1;

/// Group-0 binding of the UI layer sampler.
const UI_SAMPLER: u32 = 3;

/// Lane of the UI alpha, the mix weight.
const ALPHA_LANE: u32 = 3;

/// Vertex index of the quad's first corner, and the corner count.
const QUAD_CORNERS: usize = 4;

/// sRGB decode threshold (IEC 61966-2-1 OETF inverse).
const SRGB_LINEAR_CUTOFF: f32 = 0.04045;
/// Slope of the linear segment below the threshold.
const SRGB_LINEAR_SLOPE: f32 = 12.92;
/// Offset of the power segment.
const SRGB_OFFSET: f32 = 0.055;
/// Scale of the power segment.
const SRGB_SCALE: f32 = 1.055;
/// Exponent of the power segment.
const SRGB_GAMMA: f32 = 2.4;

/// Quad corner of the composite pass, by vertex index.
///
/// The winding differs from the shared standard quad: the UI layer is drawn
/// with this corner order, so the constants belong to the pass.
static QUAD: [[f32; 4]; 4] = [
    [-1.0, -1.0, 0.0, 1.0],
    [-1.0, 1.0, 0.0, 1.0],
    [1.0, -1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0],
];

/// UV of the same corner, in the same order.
static UVS: [[f32; 2]; 4] = [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]];

/// One lane of the sRGB → linear decode.
#[inline(always)]
fn srgb_lane(value: f32) -> f32 {
    if value <= SRGB_LINEAR_CUTOFF {
        value / SRGB_LINEAR_SLOPE
    } else {
        pow((value + SRGB_OFFSET) / SRGB_SCALE, SRGB_GAMMA)
    }
}

/// Composite vertex stage: the quad, straight from `vertex_index`.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "vs_main")]`.
#[gpu_entry(vertex)]
pub fn vs_main() {
    let index = vertex_index() as usize;
    if index >= QUAD_CORNERS {
        return;
    }
    let corner = QUAD[index];
    vertex_position(corner[0], corner[1], corner[2], corner[3]);
    let uv = UVS[index];
    vertex_store_2::<0>(uv[0], uv[1]);
}

/// Composite fragment stage: UI over the PBR layer, sRGB-decoded.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "fs_main")]`.
#[gpu_entry(fragment)]
pub fn fs_main(
    #[gpu(texture_2d, group = 0, binding = 0)] pbr_tex: Texture2d,
    #[gpu(sampler, group = 0, binding = 1)] pbr_sampler: Sampler,
    #[gpu(texture_2d, group = 0, binding = 2)] ui_tex: Texture2d,
    #[gpu(sampler, group = 0, binding = 3)] ui_sampler: Sampler,
) {
    let u = fragment_input_lane::<0, 2, 0>();
    let v = fragment_input_lane::<0, 2, 1>();
    let alpha = ui_tex.sample_lane::<0, { UI_SAMPLER }, { ALPHA_LANE }>(ui_sampler, u, v);
    let r = mix(
        pbr_tex.sample_lane::<0, { PBR_SAMPLER }, 0>(pbr_sampler, u, v),
        srgb_lane(ui_tex.sample_lane::<0, { UI_SAMPLER }, 0>(ui_sampler, u, v)),
        alpha,
    );
    let g = mix(
        pbr_tex.sample_lane::<0, { PBR_SAMPLER }, 1>(pbr_sampler, u, v),
        srgb_lane(ui_tex.sample_lane::<0, { UI_SAMPLER }, 1>(ui_sampler, u, v)),
        alpha,
    );
    let b = mix(
        pbr_tex.sample_lane::<0, { PBR_SAMPLER }, 2>(pbr_sampler, u, v),
        srgb_lane(ui_tex.sample_lane::<0, { UI_SAMPLER }, 2>(ui_sampler, u, v)),
        alpha,
    );
    fragment_store_at::<0>(r, g, b, 1.0);
}
