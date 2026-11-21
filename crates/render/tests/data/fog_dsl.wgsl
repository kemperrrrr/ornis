
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    previous_view_proj: mat4x4<f32>,
};
struct FogParams {
    color: vec3<f32>,
    density: f32,
};
struct QuadVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var fog_sampler: sampler;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var<uniform> camera: Camera;
@group(0) @binding(4) var<uniform> fog_params: FogParams;
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
fn octahedral_decode(p: vec2<f32>) -> vec3<f32> {
var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));  let t = max(-n.z, 0.0);  let neg = vec2<f32>(0.0 - t, 0.0 - t);  let pos = vec2<f32>(t, t);  let offset = select(pos, neg, n.xy >= vec2<f32>(0.0));  n.x = n.x + offset.x;  n.y = n.y + offset.y;  return normalize(n); 
}

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32, camera: Camera) -> vec3<f32> {
let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);  let clip = vec4<f32>(ndc, 1.0);  let view = camera.inv_view_proj * clip;  return view.xyz / view.w; 
}
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> { return color * (ACES_FIT_A * color + ACES_FIT_B) / (color * (ACES_FIT_C * color + ACES_FIT_D) + ACES_FIT_E);  }
@fragment
fn fs_main(input: QuadVertexOutput) -> @location(0) vec4<f32> {
let hdr = textureSampleLevel(hdr_tex, fog_sampler, input.uv, 0.0).rgb;  let depth = textureLoad(depth_tex, vec2<u32>(input.uv * vec2<f32>(textureDimensions(depth_tex))), 0);  let world_pos = reconstruct_world_pos(input.uv, depth, camera);  let dist = length(camera.camera_pos.xyz - world_pos);  let factor = 1.0 - exp(0.0 - fog_params.density * dist);  let mixed = hdr + (fog_params.color - hdr) * factor;  let tone_mapped = aces_tonemap(mixed);  return vec4<f32>(tone_mapped, 1.0); 
}
