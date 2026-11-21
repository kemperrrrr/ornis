
struct BloomParams {
    threshold: f32,
    intensity: f32,
    mode: u32,
};
struct BloomVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> bloom_params: BloomParams;
const quad: array<vec4<f32>, 4> = array<vec4<f32>, 4>(vec4<f32>(-1f, -1f, 0f, 1f), vec4<f32>(1f, -1f, 0f, 1f), vec4<f32>(-1f, 1f, 0f, 1f), vec4<f32>(1f, 1f, 0f, 1f));
const uvs: array<vec2<f32>, 4> = array<vec2<f32>, 4>(vec2<f32>(0f, 1f), vec2<f32>(1f, 1f), vec2<f32>(0f, 0f), vec2<f32>(1f, 0f));
const REC709_LUMA_R: f32 = 0.2126;
const REC709_LUMA_G: f32 = 0.7152;
const REC709_LUMA_B: f32 = 0.0722;
const ACES_FIT_A: f32 = 2.51;
const ACES_FIT_B: f32 = 0.03;
const ACES_FIT_C: f32 = 2.43;
const ACES_FIT_D: f32 = 0.59;
const ACES_FIT_E: f32 = 0.14;
const SCHLICK_FRESNEL_POWER: f32 = 5.0;
const F82_PIVOT_COSINE: f32 = 0.14285715;
const F82_CORRECTION_POWER: f32 = 6.0;
const SMITH_GGX_VISIBILITY_SCALE: f32 = 0.5;
const OREN_NAYAR_A_NUMERATOR: f32 = 0.5;
const OREN_NAYAR_A_OFFSET: f32 = 0.57;
const OREN_NAYAR_B_NUMERATOR: f32 = 0.45;
const OREN_NAYAR_B_OFFSET: f32 = 0.09;
const MICROFACET_SPECULAR_SCALE: f32 = 4.0;
const COLOR_CHANNEL_FLOOR: f32 = 0.000001;
const SUBSURFACE_DIFFUSION: f32 = 3.0;
const LAW_OF_COSINES_HALF: f32 = 0.5;
const SRGB_LINEAR_CUTOFF: f32 = 0.04045;
const SRGB_LINEAR_SLOPE: f32 = 12.92;
const SRGB_OFFSET: f32 = 0.055;
const SRGB_SCALE: f32 = 1.055;
const SRGB_GAMMA: f32 = 2.4;
const THIN_FILM_WAVELENGTH_R_NM: f32 = 650.0;
const THIN_FILM_WAVELENGTH_G_NM: f32 = 550.0;
const THIN_FILM_WAVELENGTH_B_NM: f32 = 450.0;
const THIN_FILM_PHASE_CYCLES: f32 = 4.0;
const BLOOM_SOFT_KNEE: f32 = 0.05;
fn luminance(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(REC709_LUMA_R, REC709_LUMA_G, REC709_LUMA_B));  }
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> BloomVertexOutput {
return BloomVertexOutput(quad[vertex_index], uvs[vertex_index]) /* clip_position, uv */; 
}
@fragment
fn fs_main(input: BloomVertexOutput) -> @location(0) vec4<f32> {
let color = textureSample(src_tex, src_sampler, input.uv).rgb;  let luma = luminance(color);  let keep = smoothstep(bloom_params.threshold, bloom_params.threshold + BLOOM_SOFT_KNEE, luma);  return vec4<f32>(color * keep, 1.0); 
}
