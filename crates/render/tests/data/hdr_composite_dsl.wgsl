
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    previous_view_proj: mat4x4<f32>,
};
struct BloomParams {
    threshold: f32,
    intensity: f32,
    mode: u32,
};
struct ExposureParams {
    exposure: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}
struct QuadVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@group(0) @binding(0) var deferred_tex: texture_2d<f32>;
@group(0) @binding(1) var forward_tex: texture_2d<f32>;
@group(0) @binding(2) var composite_sampler: sampler;
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;
@group(0) @binding(4) var<uniform> bloom_params: BloomParams;
@group(0) @binding(5) var<uniform> exposure_params: ExposureParams;
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
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> { return color * (ACES_FIT_A * color + ACES_FIT_B) / (color * (ACES_FIT_C * color + ACES_FIT_D) + ACES_FIT_E);  }
fn luminance(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(REC709_LUMA_R, REC709_LUMA_G, REC709_LUMA_B));  }
@fragment
fn fs_main(input: QuadVertexOutput) -> @location(0) vec4<f32> {
let deferred_color = textureSample(deferred_tex, composite_sampler, input.uv).rgb;  let forward_color = textureSample(forward_tex, composite_sampler, input.uv).rgba;  var combined = deferred_color;  if (bloom_params.mode == 1) { combined = forward_color.rgb * forward_color.a; } else { if (bloom_params.mode == 2) { combined = deferred_color + forward_color.rgb * forward_color.a; } }  let bloom = textureSample(bloom_tex, composite_sampler, input.uv).rgb;  let hdr = (combined + bloom * bloom_params.intensity) * exposure_params.exposure;  let tonemapped = aces_tonemap(hdr);  return vec4<f32>(tonemapped, 1.0); 
}
