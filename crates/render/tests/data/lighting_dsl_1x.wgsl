
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    previous_view_proj: mat4x4<f32>,
}

struct Light {
    kind: vec4<f32>,
    direction: vec4<f32>,
    position: vec4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,
    shadow_vp: mat4x4<f32>,
}

struct CsmVp {
    cols: mat4x4<f32>,
}

struct Csm {
    vps: array<CsmVp, 4>,
    view_fwd: vec4<f32>,
    splits_lo: vec4<f32>,
    split_far: f32,
    cascade_count: u32,
    pad: vec2<u32>,
}

struct Lighting {
    ambient_color: vec4<f32>,
    lights: array<Light, 8>,
    light_count: u32,
    ibl_weight: f32,
    ibl_max_mip: f32,
    debug_view: u32,
    csm: Csm,
}

struct OpenPBRMaterial {
    base_params: vec4<f32>,
    base_color: vec4<f32>,
    specular_params: vec4<f32>,
    specular_color: vec4<f32>,
    transmission_params: vec4<f32>,
    transmission_color: vec4<f32>,
    transmission_scatter: vec4<f32>,
    subsurface_params: vec4<f32>,
    subsurface_color: vec4<f32>,
    subsurface_radius_scale_gb: vec4<f32>,
    fuzz_params: vec4<f32>,
    fuzz_color: vec4<f32>,
    coat_params: vec4<f32>,
    coat_color: vec4<f32>,
    coat_ior: vec4<f32>,
    thin_film_params: vec4<f32>,
    emission_params: vec4<f32>,
    emission_color: vec4<f32>,
    geometry_params: vec4<f32>,
    geometry_params2_: vec4<f32>,
}

struct ClusterParams {
    light_count: u32,
    head_count: u32,
    clustered: u32,
    pad: u32,
}

struct ClusterSpan {
    offset: u32,
    count: u32,
}

@group(0) @binding(0) 
var<uniform> camera: Camera;
@group(0) @binding(1) 
var<uniform> lighting: Lighting;
@group(0) @binding(2) 
var<storage, read> materials: array<OpenPBRMaterial>;
@group(0) @binding(3) 
var albedo_tex: texture_2d<f32>;
@group(0) @binding(4) 
var normal_tex: texture_2d<f32>;
@group(0) @binding(5) 
var material_id_tex: texture_2d<u32>;
@group(0) @binding(6) 
var world_pos_tex: texture_2d<f32>;
@group(0) @binding(7) 
var mat_params_tex: texture_2d<f32>;
@group(0) @binding(8) 
var depth_tex: texture_depth_2d;
@group(0) @binding(9) 
var lighting_sampler: sampler;
@group(0) @binding(10) 
var shadow_tex: texture_depth_2d_array;
@group(0) @binding(11) 
var shadow_sampler: sampler_comparison;
@group(0) @binding(12) 
var shadow_cube_tex: texture_depth_cube_array;
@group(0) @binding(13) 
var prefilter_cube: texture_cube<f32>;
@group(0) @binding(14) 
var irradiance_cube: texture_cube<f32>;
@group(0) @binding(15) 
var brdf_lut: texture_2d<f32>;
@group(0) @binding(16) 
var ibl_sampler: sampler;
@group(0) @binding(17) 
var ao_tex: texture_2d<f32>;
@group(0) @binding(18) 
var csm_tex: texture_depth_2d_array;
@group(0) @binding(19) 
var<uniform> cluster_params: ClusterParams;
@group(0) @binding(20) 
var<storage, read> clustered_lights: array<Light>;
@group(0) @binding(21) 
var<storage, read> cluster_light_indices: array<u32>;
@group(0) @binding(22) 
var<storage, read> cluster_grid: array<ClusterSpan>;
struct QuadVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
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
const PI: f32 = 3.1415927;
const EPS: f32 = 0.000001;
const INV_PI: f32 = 0.31830987;
const SHADOW_REF_BIAS: f32 = 0.001;
const CLEAR_DEPTH: f32 = 1.0;
const NORMAL_AGREE: f32 = 0.999;
const DEPTH_EDGE: f32 = 0.002;
const ALPHA_CUTOFF: f32 = 0.001;
const MSAA_SAMPLES_F: f32 = 4.0;
const POINT_OR_SPOT_KIND_EDGE: f32 = 0.5;
const SPOT_KIND_EDGE: f32 = 1.5;
const RANGE_CUTOFF_INNER_FRACTION: f32 = 0.99;
const NDC_TO_UV_HALF: f32 = 0.5;
const MOTION_VALID_THRESHOLD: f32 = 0.5;
const DEPTH_ROUNDING_BIAS: f32 = 0.5;
const PREVIOUS_DEPTH_MAX: f32 = 65534.0;
const SHADOW_CUBE_NEAR: f32 = 0.1;
const CSM_BLEND_FRACTION: f32 = 0.08;
const CSM_ACTIVE_EDGE: f32 = 0.5;
const LIGHT_SELECTOR_SPLIT: f32 = 0.5;
const SPECULAR_AA_VARIANCE: f32 = 0.15;
const SPECULAR_AA_KERNEL_SCALE: f32 = 2.0;
const SPECULAR_AA_KERNEL_CAP: f32 = 0.2;
const KARIS_LUMA_BIAS: f32 = 1.0;
const HDR_FINITE_LIMIT: f32 = 100000000000000000000.0;
const SHADING_DEBUG_HOT_LUMA: f32 = 1.0;
const UNLIT_THRESHOLD: f32 = 0.5;
const SHADING_DEBUG_BEAUTY: u32 = 0u;
const SHADING_DEBUG_DIRECT_SPECULAR: u32 = 1u;
const SHADING_DEBUG_IBL_SPECULAR: u32 = 2u;
const SHADING_DEBUG_DIFFUSE: u32 = 3u;
const SHADING_DEBUG_ROUGHNESS: u32 = 4u;
const SHADING_DEBUG_METALLIC: u32 = 5u;
const SHADING_DEBUG_SHADOW: u32 = 6u;
const SHADING_DEBUG_HEAT: u32 = 7u;
const CSM_LAST_CASCADE: u32 = 3u;
const PREVIOUS_DEPTH_INVALID: u32 = 65535u;
const CLUSTERS_X: u32 = 16u;
const CLUSTERS_Y: u32 = 9u;
const CLUSTERS_Z: u32 = 24u;
const CLUSTER_DEPTH_NEAR: f32 = 0.5;
const CLUSTER_SLICE_SCALE: f32 = 3.4407384;
const CLUSTER_SLICE_LOOKUP_EPS: f32 = 0.0001;
const MSAA_SAMPLES: u32 = 4u;
fn octahedral_decode(p: vec2<f32>) -> vec3<f32> {
var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));  let t = max(-n.z, 0.0);  let neg = vec2<f32>(0.0 - t, 0.0 - t);  let pos = vec2<f32>(t, t);  let offset = select(pos, neg, n.xy >= vec2<f32>(0.0));  n.x = n.x + offset.x;  n.y = n.y + offset.y;  return normalize(n); 
}

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32, camera: Camera) -> vec3<f32> {
let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);  let clip = vec4<f32>(ndc, 1.0);  let view = camera.inv_view_proj * clip;  return view.xyz / view.w; 
}
fn cluster_light_span(world_pos: vec3<f32>) -> vec2<u32> {
let clip = camera.view_proj * vec4<f32>(world_pos.x, world_pos.y, world_pos.z, 1.0);  let w = max(clip.w, CLUSTER_DEPTH_NEAR);  let ndc = clip.xyz / w;  let tile_f = (ndc.xy * NDC_TO_UV_HALF + NDC_TO_UV_HALF) * vec2<f32>(f32(CLUSTERS_X), f32(CLUSTERS_Y));  let tile = clamp(vec2<u32>(u32(floor(tile_f.x)), u32(floor(tile_f.y))), vec2<u32>(0u, 0u), vec2<u32>(CLUSTERS_X - 1u, CLUSTERS_Y - 1u));  let slice = clamp(u32(ceil(CLUSTER_SLICE_SCALE * log(w / CLUSTER_DEPTH_NEAR) - CLUSTER_SLICE_LOOKUP_EPS)), 0u, CLUSTERS_Z - 1u);  let cluster = tile.x + tile.y * CLUSTERS_X + slice * CLUSTERS_X * CLUSTERS_Y;  let span = cluster_grid[cluster];  let clustered = cluster_params.clustered;  return vec2<u32>(span.offset * clustered, span.count * clustered);  
}

fn csm_visibility(world_pos: vec3<f32>, cam_pos: vec3<f32>, csm: Csm, csm_tex: texture_depth_2d_array, shadow_sampler: sampler_comparison) -> f32 {
let view_depth = dot(world_pos - cam_pos, csm.view_fwd.xyz);  let b0 = csm.splits_lo.x;  let b1 = csm.splits_lo.y;  let b2 = csm.splits_lo.z;  let b3 = csm.splits_lo.w;  let b4 = csm.split_far;  let past_b1 = step(b1, view_depth);  let past_b2 = step(b2, view_depth);  let past_b3 = step(b3, view_depth);  let cascade = u32(past_b1 + past_b2 + past_b3);  let upper = mix(mix(mix(b1, b2, past_b1), b3, past_b2), b4, past_b3);  let lower = mix(mix(mix(b0, b1, past_b1), b2, past_b2), b3, past_b3);  let clip0 = csm.vps[cascade].cols * vec4<f32>(world_pos.x, world_pos.y, world_pos.z, 1.0);  let ndc0 = clip0.xyz / clip0.w;  let uv0 = vec2<f32>(ndc0.x * NDC_TO_UV_HALF + NDC_TO_UV_HALF, NDC_TO_UV_HALF - ndc0.y * NDC_TO_UV_HALF);  var vis = textureSampleCompare(csm_tex, shadow_sampler, uv0, i32(cascade), ndc0.z - SHADOW_REF_BIAS);  if (cascade < CSM_LAST_CASCADE) { let band = max((upper - lower) * CSM_BLEND_FRACTION, EPS);  let t = clamp((upper - view_depth) / band, 0.0, 1.0);  let next = cascade + 1;  let clip1 = csm.vps[next].cols * vec4<f32>(world_pos.x, world_pos.y, world_pos.z, 1.0);  let ndc1 = clip1.xyz / clip1.w;  let uv1 = vec2<f32>(ndc1.x * NDC_TO_UV_HALF + NDC_TO_UV_HALF, NDC_TO_UV_HALF - ndc1.y * NDC_TO_UV_HALF);  let vis1 = textureSampleCompare(csm_tex, shadow_sampler, uv1, i32(next), ndc1.z - SHADOW_REF_BIAS);  vis = mix(vis1, vis, t); }  return vis;  
}

fn specular_aa_roughness(roughness: f32, n: vec3<f32>) -> f32 {
let du = dpdx(n);  let dv = dpdy(n);  let variance = SPECULAR_AA_VARIANCE * (dot(du, du) + dot(dv, dv));  let kernel = min(SPECULAR_AA_KERNEL_SCALE * variance, SPECULAR_AA_KERNEL_CAP);  let alpha = roughness * roughness;  let square = saturate(alpha * alpha + kernel);  return sqrt(sqrt(max(square, 0.0))); 
}

fn sanitize_hdr(color: vec3<f32>) -> vec3<f32> {
let x_ok = select(0.0, 1.0, color.x == color.x && abs(color.x) < HDR_FINITE_LIMIT);  let y_ok = select(0.0, 1.0, color.y == color.y && abs(color.y) < HDR_FINITE_LIMIT);  let z_ok = select(0.0, 1.0, color.z == color.z && abs(color.z) < HDR_FINITE_LIMIT);  return color * (x_ok * y_ok * z_ok); 
}

fn evaluate_base_specular(v: vec3<f32>, l: vec3<f32>, h: vec3<f32>, nov: f32, nol: f32, noh: f32, voh: f32, base_weight: f32, base_color: vec3<f32>, metalness: f32, specular_weight: f32, specular_roughness: f32, specular_ior: f32, specular_anisotropy: f32, specular_edge_tint: vec3<f32>, t: vec3<f32>, b: vec3<f32>, thin_film_mod: vec3<f32>) -> vec3<f32> {
let f0_dielectric = vec3<f32>(fresnel0_from_ior(specular_ior));  let f0_metal = base_color * base_weight;  let f0 = mix(f0_dielectric, f0_metal, metalness);  let f = fresnel_f82_tint(voh, f0, specular_edge_tint);  let alpha = openpbr_anisotropy(specular_roughness, specular_anisotropy);  let d = ggx_ndf_aniso(noh, h, t, b, alpha.x, alpha.y);  let g = smith_ggx_aniso(nov, nol, v, l, t, b, alpha.x, alpha.y);  let ks = f * specular_weight;  return ks * d * g * base_weight * thin_film_mod; 
}

fn select_shading_debug(view: u32, beauty: vec3<f32>, direct_spec: vec3<f32>, ibl_spec: vec3<f32>, diffuse: vec3<f32>, roughness: f32, metalness: f32, shadow: f32, raw: vec3<f32>) -> vec3<f32> {
let is_beauty = select(0.0, 1.0, view == SHADING_DEBUG_BEAUTY);  let is_direct = select(0.0, 1.0, view == SHADING_DEBUG_DIRECT_SPECULAR);  let is_ibl = select(0.0, 1.0, view == SHADING_DEBUG_IBL_SPECULAR);  let is_diffuse = select(0.0, 1.0, view == SHADING_DEBUG_DIFFUSE);  let is_rough = select(0.0, 1.0, view == SHADING_DEBUG_ROUGHNESS);  let is_metal = select(0.0, 1.0, view == SHADING_DEBUG_METALLIC);  let is_shadow = select(0.0, 1.0, view == SHADING_DEBUG_SHADOW);  let is_heat = select(0.0, 1.0, view == SHADING_DEBUG_HEAT);  let x_bad = select(0.0, 1.0, raw.x != raw.x || abs(raw.x) >= HDR_FINITE_LIMIT);  let y_bad = select(0.0, 1.0, raw.y != raw.y || abs(raw.y) >= HDR_FINITE_LIMIT);  let z_bad = select(0.0, 1.0, raw.z != raw.z || abs(raw.z) >= HDR_FINITE_LIMIT);  let bad = max(x_bad, max(y_bad, z_bad));  let hot = select(0.0, 1.0, luminance(beauty) > SHADING_DEBUG_HOT_LUMA);  let heat = mix(mix(beauty, vec3<f32>(SHADING_DEBUG_HOT_LUMA, 0.0, 0.0), hot), vec3<f32>(SHADING_DEBUG_HOT_LUMA, 0.0, SHADING_DEBUG_HOT_LUMA), bad);  return beauty * is_beauty + direct_spec * is_direct + ibl_spec * is_ibl + diffuse * is_diffuse + vec3<f32>(roughness, roughness, roughness) * is_rough + vec3<f32>(metalness, metalness, metalness) * is_metal + vec3<f32>(shadow, shadow, shadow) * is_shadow + heat * is_heat; 
}

fn evaluate_base_layer(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, h: vec3<f32>, nov: f32, nol: f32, noh: f32, voh: f32, mat: OpenPBRMaterial, base_weight: f32, base_color: vec3<f32>, metalness: f32, diffuse_roughness: f32, specular_weight: f32, specular_roughness: f32, specular_ior: f32, specular_anisotropy: f32, specular_edge_tint: vec3<f32>, t: vec3<f32>, b: vec3<f32>, thin_film_mod: vec3<f32>) -> vec3<f32> {
let f0_dielectric = vec3<f32>(fresnel0_from_ior(specular_ior));  let f0_metal = base_color * base_weight;  let f0 = mix(f0_dielectric, f0_metal, metalness);  let f = fresnel_f82_tint(voh, f0, specular_edge_tint);  let spec = evaluate_base_specular(v, l, h, nov, nol, noh, voh, base_weight, base_color, metalness, specular_weight, specular_roughness, specular_ior, specular_anisotropy, specular_edge_tint, t, b, thin_film_mod);  let diffuse_color = base_color * (1.0 - metalness);  let diff_roughness = max(diffuse_roughness, specular_roughness);  let diff_alpha = diff_roughness * diff_roughness;  let cos_phi = max(dot(normalize(v - n * nov), normalize(l - n * nol)), 0.0);  let diff_brdf = oren_nayar_brdf(nov, nol, cos_phi, diff_alpha);  let ks = f * specular_weight;  let kd = vec3<f32>(base_diffuse_energy(luminance(ks)));  let diff = kd * diff_brdf * diffuse_color * base_weight * thin_film_mod;  return spec + diff; 
}

fn evaluate_coat_layer(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, h: vec3<f32>, nov: f32, nol: f32, noh: f32, voh: f32, mat: OpenPBRMaterial, coat_weight: f32, coat_roughness: f32, coat_anisotropy: f32, coat_dark: f32, coat_ior: f32, coat_color: vec3<f32>, base_metalness: f32, base_color: vec3<f32>, base_weight: f32, specular_weight: f32, subsurface_weight: f32, subsurface_color: vec3<f32>, t: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
if (coat_weight <= 0.0) { return vec3<f32>(0.0); }  let coat_f0 = vec3<f32>(fresnel0_from_ior(coat_ior));  let coat_f = fresnel_schlick_vec(voh, coat_f0);  let coat_alpha = openpbr_anisotropy(coat_roughness, coat_anisotropy);  let coat_alpha_u = coat_alpha.x;  let coat_alpha_v = coat_alpha.y;  let coat_d = ggx_ndf_aniso(noh, h, t, b, coat_alpha_u, coat_alpha_v);  let coat_g = smith_ggx_aniso(nov, nol, v, l, t, b, coat_alpha_u, coat_alpha_v);  let coat_brdf = coat_d * coat_g * coat_f;  return coat_color * coat_brdf * coat_weight; 
}

fn evaluate_coat_darkening(coat_weight: f32, coat_dark: f32, coat_ior: f32, base_metalness: f32, base_color: vec3<f32>, base_weight: f32, specular_weight: f32, subsurface_weight: f32, subsurface_color: vec3<f32>) -> vec3<f32> {
return coat_blend_darkened(coat_base_darkening(coat_ior, base_metalness, base_color, base_weight, specular_weight, subsurface_weight, subsurface_color), coat_weight, coat_dark); 
}

fn evaluate_fuzz_layer(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, h: vec3<f32>, nov: f32, nol: f32, noh: f32, voh: f32, fuzz_weight: f32, fuzz_roughness: f32, fuzz_color: vec3<f32>) -> vec3<f32> {
if (fuzz_weight <= 0.0) { return vec3<f32>(0.0); }  let sheen = sheen_brdf(nov, nol, noh, voh, fuzz_roughness);  return fuzz_color * sheen * fuzz_weight; 
}

fn evaluate_transmission_layer(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, h: vec3<f32>, nov: f32, nol: f32, noh: f32, voh: f32, mat: OpenPBRMaterial, transmission_weight: f32, transmission_depth: f32, transmission_dispersion_scale: f32, transmission_dispersion_abbe: f32, transmission_color: vec3<f32>, transmission_scatter: vec3<f32>, transmission_scatter_anisotropy: f32, specular_ior: f32, specular_roughness: f32, specular_anisotropy: f32, thin_walled: f32) -> vec3<f32> {
if (transmission_weight <= 0.0) { return vec3<f32>(0.0); }  let ior_out = 1.0;  let ior_in = specular_ior;  let eta = ior_in / ior_out;  let alpha = openpbr_anisotropy(specular_roughness, specular_anisotropy);  let extinction = transmission_color_to_extinction(transmission_color, transmission_depth);  let distance = transmission_depth;  let btdf = transmission_btdf(nov, nol, noh, voh, ior_in, ior_out, alpha.x, extinction, distance);  return transmission_color * btdf * transmission_weight; 
}

fn evaluate_subsurface_layer(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, nov: f32, nol: f32, subsurface_weight: f32, subsurface_radius: f32, subsurface_radius_scale_r: f32, subsurface_radius_scale_g: f32, subsurface_radius_scale_b: f32, subsurface_scatter_anisotropy: f32, subsurface_color: vec3<f32>) -> vec3<f32> {
if (subsurface_weight <= 0.0) { return vec3<f32>(0.0); }  let radius = vec3<f32>(subsurface_radius * subsurface_radius_scale_r, subsurface_radius * subsurface_radius_scale_g, subsurface_radius * subsurface_radius_scale_b);  let v_proj = v - n * nov;  let l_proj = l - n * nol;  let distance = length(v_proj - l_proj);  let ss_brdf = subsurface_brdf(nov, nol, distance, radius, subsurface_scatter_anisotropy);  return subsurface_color * ss_brdf * subsurface_weight; 
}

fn evaluate_emission(emission_luminance: f32, emission_color: vec3<f32>, coat_weight: f32, coat_color: vec3<f32>, nov: f32) -> vec3<f32> {
return coated_emission(emission_color, emission_luminance, coat_weight, coat_color, nov); 
}

fn transmission_btdf(nov: f32, nol: f32, noh: f32, voh: f32, ior_in: f32, ior_out: f32, alpha: f32, extinction: vec3<f32>, distance: f32) -> vec3<f32> {
let eta = ior_in / ior_out;  let cos_theta_t = sqrt(max(1.0 - eta * eta * (1.0 - nov * nov), 0.0));  let cos_theta_i = nov;  let f = fresnel_schlick(max(cos_theta_i, EPS), fresnel0_from_ior(ior_in));  let t = 1.0 - f;  let d = ggx_ndf(noh, alpha);  let g = smith_ggx_correlated(nov, nol, alpha);  let extinction_factor = exp(-extinction * distance);  return vec3<f32>(d * g * t) * extinction_factor; 
}
fn shade_lit(n: vec3<f32>, world_pos: vec3<f32>, v: vec3<f32>, alpha_ao: vec2<f32>, mat: OpenPBRMaterial) -> vec4<f32> {
let albedo_a = alpha_ao.x;  let ao = alpha_ao.y;  let specular_roughness = specular_aa_roughness(mat.specular_params.y, n);  if (albedo_a < ALPHA_CUTOFF) { discard; }  let nov = max(dot(n, v), EPS);  let base_weight = mat.base_params.x;  let base_color = mat.base_color.rgb;  let metalness = mat.base_params.z;  let diffuse_roughness = mat.base_params.y;  let specular_weight = mat.specular_params.x;  let specular_ior = mat.specular_params.z;  let specular_anisotropy = mat.specular_params.w;  let specular_edge_tint = mat.specular_color.rgb;  let transmission_weight = mat.transmission_params.x;  let transmission_depth = mat.transmission_params.y;  let transmission_dispersion_scale = mat.transmission_params.z;  let transmission_dispersion_abbe = mat.transmission_params.w;  let transmission_color = mat.transmission_color.rgb;  let transmission_scatter = mat.transmission_scatter.rgb;  let transmission_scatter_anisotropy = mat.transmission_scatter.a;  let subsurface_weight = mat.subsurface_params.x;  let subsurface_radius = mat.subsurface_params.y;  let subsurface_radius_scale_r = mat.subsurface_params.z;  let subsurface_scatter_anisotropy = mat.subsurface_params.w;  let subsurface_color = mat.subsurface_color.rgb;  let subsurface_radius_scale_g = mat.subsurface_radius_scale_gb.x;  let subsurface_radius_scale_b = mat.subsurface_radius_scale_gb.y;  let fuzz_weight = mat.fuzz_params.x;  let fuzz_roughness = mat.fuzz_params.y;  let fuzz_color = mat.fuzz_color.rgb;  let coat_weight = mat.coat_params.x;  let coat_roughness = mat.coat_params.y;  let coat_anisotropy = mat.coat_params.z;  let coat_darkening = mat.coat_params.w;  let coat_color = mat.coat_color.rgb;  let coat_ior = mat.coat_ior.x;  let thin_film_weight = mat.thin_film_params.x;  let thin_film_thickness_um = mat.thin_film_params.y;  let thin_film_ior = mat.thin_film_params.z;  let emission_luminance = mat.emission_params.x;  let emission_color = mat.emission_color.rgb;  let opacity = mat.geometry_params.x;  let thin_walled = mat.geometry_params.y;  var lo = vec3<f32>(0.0, 0.0, 0.0);  var spec_acc = vec3<f32>(0.0, 0.0, 0.0);  var diff_acc = vec3<f32>(0.0, 0.0, 0.0);  var shadow_vis = 1.0;  let thin_film_mod = thin_film_weight_mix(thin_film_weight, thin_film_modulation(nov, thin_film_ior, thin_film_thickness_um, 1.0));  let coat_darken = evaluate_coat_darkening(coat_weight, coat_darkening, coat_ior, metalness, base_color, base_weight, specular_weight, subsurface_weight, subsurface_color);  let t = normalize(cross(n, vec3<f32>(0.0, 1.0, 0.0)) + vec3<f32>(0.0, 0.0, EPS));  let b = cross(n, t);  let span = cluster_light_span(world_pos);  let total = lighting.light_count + span.y;  for (var k: u32 = 0; k < total; k = k + 1) { var light = clustered_lights[k];  if (k >= lighting.light_count) { light = clustered_lights[cluster_light_indices[span.x + (k - lighting.light_count)]]; }  let kind = light.kind.x;  let to_light = light.position.xyz - world_pos;  let dist = length(to_light);  let l_point = to_light / max(dist, EPS);  let use_point = step(POINT_OR_SPOT_KIND_EDGE, kind);  let l_dir = normalize(mix(normalize(light.direction.xyz), l_point, use_point));  let edge0 = light.params.x * RANGE_CUTOFF_INNER_FRACTION;  let range_cut = mix(1.0, 1.0 - smoothstep(edge0, light.params.x, dist), use_point);  let l = l_dir;  let h = normalize(v + l + vec3<f32>(0.0, 0.0, EPS));  let light_color = light.color.rgb;  let intensity = light.color.w;  let cos_theta = dot(normalize(light.direction.xyz), normalize(world_pos - light.position.xyz));  let cone = mix(1.0, smoothstep(light.params.z, light.params.y, cos_theta), step(SPOT_KIND_EDGE, kind));  let attenuation = mix(1.0, 1.0 / max(dist * dist, EPS), use_point);  var radiance = light_color * intensity * attenuation * range_cut * cone;  let is_point = step(POINT_OR_SPOT_KIND_EDGE, kind) * (1.0 - step(SPOT_KIND_EDGE, kind));  let is_csm = (1.0 - step(POINT_OR_SPOT_KIND_EDGE, kind)) * step(CSM_ACTIVE_EDGE, light.params.x);  if (is_csm > LIGHT_SELECTOR_SPLIT) { let vis = csm_visibility(world_pos, camera.camera_pos.xyz, lighting.csm, csm_tex, shadow_sampler);  radiance = radiance * vis;  shadow_vis = min(shadow_vis, vis); }  if (light.params.w >= 0.0) { if (is_point <= LIGHT_SELECTOR_SPLIT) { let shadow_clip = light.shadow_vp * vec4<f32>(world_pos.x, world_pos.y, world_pos.z, 1.0);  let shadow_ndc = shadow_clip.xyz / shadow_clip.w;  let shadow_uv = vec2<f32>(shadow_ndc.x * NDC_TO_UV_HALF + NDC_TO_UV_HALF, NDC_TO_UV_HALF - shadow_ndc.y * NDC_TO_UV_HALF);  let vis = textureSampleCompare(shadow_tex, shadow_sampler, shadow_uv, i32(light.params.w), shadow_ndc.z - SHADOW_REF_BIAS);  radiance = radiance * vis;  shadow_vis = min(shadow_vis, vis); }  if (is_point > LIGHT_SELECTOR_SPLIT) { let to_frag = world_pos - light.position.xyz;  let major = max(max(abs(to_frag.x), abs(to_frag.y)), abs(to_frag.z));  let far = max(light.params.x, 1.0);  let denom = far - SHADOW_CUBE_NEAR;  let cube_ref = (far / denom) - (SHADOW_CUBE_NEAR * far) / (denom * major) - SHADOW_REF_BIAS;  let vis = textureSampleCompare(shadow_cube_tex, shadow_sampler, to_frag, i32(light.params.w), cube_ref);  radiance = radiance * vis;  shadow_vis = min(shadow_vis, vis); } }  let nol = max(dot(n, l), EPS);  let noh = max(dot(n, h), EPS);  let voh = max(dot(v, h), EPS);  if (nol <= EPS) { continue; }  let base_bsdf = evaluate_base_layer(n, v, l, h, nov, nol, noh, voh, mat, base_weight, base_color, metalness, diffuse_roughness, specular_weight, specular_roughness, specular_ior, specular_anisotropy, specular_edge_tint, t, b, thin_film_mod) * coat_darken;  if (lighting.debug_view != SHADING_DEBUG_BEAUTY) { let spec_lobe = evaluate_base_specular(v, l, h, nov, nol, noh, voh, base_weight, base_color, metalness, specular_weight, specular_roughness, specular_ior, specular_anisotropy, specular_edge_tint, t, b, thin_film_mod) * coat_darken;  spec_acc = spec_acc + spec_lobe * radiance * nol;  diff_acc = diff_acc + (base_bsdf - spec_lobe) * radiance * nol; }  let coat_bsdf = evaluate_coat_layer(n, v, l, h, nov, nol, noh, voh, mat, coat_weight, coat_roughness, coat_anisotropy, coat_darkening, coat_ior, coat_color, metalness, base_color, base_weight, specular_weight, subsurface_weight, subsurface_color, t, b);  let fuzz_bsdf = evaluate_fuzz_layer(n, v, l, h, nov, nol, noh, voh, fuzz_weight, fuzz_roughness, fuzz_color);  let trans_bsdf = evaluate_transmission_layer(n, v, l, h, nov, nol, noh, voh, mat, transmission_weight, transmission_depth, transmission_dispersion_scale, transmission_dispersion_abbe, transmission_color, transmission_scatter, transmission_scatter_anisotropy, specular_ior, specular_roughness, specular_anisotropy, thin_walled);  let ss_bsdf = evaluate_subsurface_layer(n, v, l, nov, nol, subsurface_weight, subsurface_radius, subsurface_radius_scale_r, subsurface_radius_scale_g, subsurface_radius_scale_b, subsurface_scatter_anisotropy, subsurface_color);  let layer_bsdf = base_bsdf + coat_bsdf + fuzz_bsdf + trans_bsdf + ss_bsdf;  lo = lo + layer_bsdf * radiance * nol;  }  let ambient = lighting.ambient_color.rgb * mix(base_color, base_color * specular_weight, metalness) * ao;  let emission = evaluate_emission(emission_luminance, emission_color, coat_weight, coat_color, nov);  let f0_dielectric = fresnel0_from_ior(specular_ior);  let f0 = mix(vec3<f32>(f0_dielectric, f0_dielectric, f0_dielectric), base_color * base_weight, metalness);  let reflect_dir = n * (2.0 * dot(n, v)) - v;  let prefiltered = textureSampleLevel(prefilter_cube, ibl_sampler, reflect_dir, specular_roughness * lighting.ibl_max_mip);  let irradiance = textureSample(irradiance_cube, ibl_sampler, n);  let lut = textureSample(brdf_lut, ibl_sampler, vec2<f32>(nov, specular_roughness));  let ibl = evaluate_ibl(prefiltered.rgb, irradiance.rgb, lut.r, lut.g, f0, base_color * (1.0 - metalness), lighting.ibl_weight) * ao;  let full = ambient + lo + emission + ibl;  let raw = select(full, emission, mat.geometry_params.z >= UNLIT_THRESHOLD);  let beauty = sanitize_hdr(raw);  var ibl_spec = vec3<f32>(0.0, 0.0, 0.0);  if (lighting.debug_view == SHADING_DEBUG_IBL_SPECULAR || lighting.debug_view == SHADING_DEBUG_DIFFUSE) { ibl_spec = evaluate_ibl_specular(prefiltered.rgb, lut.r, lut.g, f0, lighting.ibl_weight); }  let diffuse = diff_acc + ambient + (ibl - ibl_spec);  let color = select_shading_debug(lighting.debug_view, beauty, spec_acc, ibl_spec, diffuse, specular_roughness, metalness, shadow_vis, raw);  return vec4<f32>(color, opacity); 
}
fn luminance(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(REC709_LUMA_R, REC709_LUMA_G, REC709_LUMA_B));  }
fn fresnel0_from_ior(ior: f32) -> f32 { let f = (ior - 1.0) / (ior + 1.0);  return f * f;  }
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 { return f0 + (1.0 - f0) * pow((1.0 - cos_theta), SCHLICK_FRESNEL_POWER);  }
fn fresnel_schlick_vec(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> { return f0 + (vec3<f32>(1.0) - f0) * pow((1.0 - cos_theta), SCHLICK_FRESNEL_POWER);  }
fn fresnel_f82_tint(cos_theta: f32, f0: vec3<f32>, f82_tint: vec3<f32>) -> vec3<f32> { let mu_bar = F82_PIVOT_COSINE;  let schlick_at_mu_bar = f0 + (vec3<f32>(1.0) - f0) * pow((1.0 - mu_bar), SCHLICK_FRESNEL_POWER);  let f82 = f82_tint * schlick_at_mu_bar;  let numerator = cos_theta * pow((1.0 - cos_theta), F82_CORRECTION_POWER);  let denominator = mu_bar * pow((1.0 - mu_bar), F82_CORRECTION_POWER);  let scale = numerator / denominator;  let f_schlick = f0 + (vec3<f32>(1.0) - f0) * pow((1.0 - cos_theta), SCHLICK_FRESNEL_POWER);  let f82_correction = f_schlick - vec3<f32>(scale) * (schlick_at_mu_bar - f82);  return max(f82_correction, vec3<f32>(0.0));  }
fn ggx_ndf(NoH: f32, alpha: f32) -> f32 { let a2 = alpha * alpha;  let denom = NoH * NoH * (a2 - 1.0) + 1.0;  return a2 / (PI * denom * denom);  }
fn ggx_ndf_aniso(NoH: f32, H: vec3<f32>, T: vec3<f32>, B: vec3<f32>, alpha_u: f32, alpha_v: f32) -> f32 { let Hu = dot(H, T);  let Hv = dot(H, B);  let denom = (Hu * Hu) / (alpha_u * alpha_u) + (Hv * Hv) / (alpha_v * alpha_v) + NoH * NoH;  return 1.0 / (PI * alpha_u * alpha_v * denom * denom);  }
fn openpbr_anisotropy(roughness: f32, anisotropy: f32) -> vec2<f32> { let r2 = roughness * roughness;  let aniso_inv = 1.0 - anisotropy;  let aniso_inv_sq = aniso_inv * aniso_inv;  let denom = aniso_inv_sq + 1.0;  let fraction = 2.0 / denom;  let sqrt_frac = sqrt(fraction);  let alpha_u = r2 * sqrt_frac;  let alpha_v = aniso_inv * alpha_u;  return vec2<f32>(alpha_u, alpha_v);  }
fn smith_ggx_correlated(NoV: f32, NoL: f32, alpha: f32) -> f32 { let a2 = alpha * alpha;  let ggxv = NoV * sqrt(max((NoL * NoL * (1.0 - a2) + a2), EPS));  let ggxl = NoL * sqrt(max((NoV * NoV * (1.0 - a2) + a2), EPS));  return SMITH_GGX_VISIBILITY_SCALE / max((ggxv + ggxl), EPS);  }
fn smith_ggx_aniso(NoV: f32, NoL: f32, V: vec3<f32>, L: vec3<f32>, T: vec3<f32>, B: vec3<f32>, alpha_u: f32, alpha_v: f32) -> f32 { let Vu = dot(V, T);  let Vv = dot(V, B);  let Lu = dot(L, T);  let Lv = dot(L, B);  let a2u = alpha_u * alpha_u;  let a2v = alpha_v * alpha_v;  let ggxv = NoV * sqrt(max((Lu * Lu * a2u + Lv * Lv * a2v + NoL * NoL), EPS));  let ggxl = NoL * sqrt(max((Vu * Vu * a2u + Vv * Vv * a2v + NoV * NoV), EPS));  return SMITH_GGX_VISIBILITY_SCALE / max((ggxv + ggxl), EPS);  }
fn oren_nayar_brdf(NoV: f32, NoL: f32, cos_phi: f32, alpha: f32) -> f32 { let sigma = max(alpha, EPS);  let sigma2 = sigma * sigma;  let A = 1.0 - OREN_NAYAR_A_NUMERATOR * sigma2 / (sigma2 + OREN_NAYAR_A_OFFSET);  let B = OREN_NAYAR_B_NUMERATOR * sigma2 / (sigma2 + OREN_NAYAR_B_OFFSET);  let theta_v = acos(max(NoV, 0.0));  let theta_l = acos(max(NoL, 0.0));  let alpha_max = max(theta_v, theta_l);  let beta_min = min(theta_v, theta_l);  let tan_beta = tan(beta_min);  return (A + B * cos_phi * sin(alpha_max) * tan_beta) * INV_PI;  }
fn base_diffuse_energy(specular_luma: f32) -> f32 { return max((1.0 - specular_luma), 0.0);  }
fn thin_film_weight_mix(weight: f32, modulation: vec3<f32>) -> vec3<f32> { return mix(vec3<f32>(1.0), modulation, clamp(weight, 0.0, 1.0));  }
fn coated_emission(emission_color: vec3<f32>, emission_luminance: f32, coat_weight: f32, coat_color: vec3<f32>, nov: f32) -> vec3<f32> { if (emission_luminance <= 0.0) { return vec3<f32>(0.0); }  let base_emission = emission_color * emission_luminance;  let coat_fresnel = pow((1.0 - nov), SCHLICK_FRESNEL_POWER);  let transmit = (1.0 - coat_fresnel) * coat_weight + (1.0 - coat_weight);  let coat_emission = coat_color * base_emission * transmit;  return mix(base_emission, coat_emission, coat_weight);  }
fn coat_base_darkening(coat_ior: f32, base_metalness: f32, base_color: vec3<f32>, base_weight: f32, specular_weight: f32, subsurface_weight: f32, subsurface_color: vec3<f32>) -> vec3<f32> { let coat_f0 = pow(((coat_ior - 1.0) / (coat_ior + 1.0)), 2.0);  let one_minus_coat_f0 = 1.0 - coat_f0;  let coat_ior_sq = coat_ior * coat_ior;  let Kcoat = 1.0 - one_minus_coat_f0 / coat_ior_sq;  let Emetal = base_color * base_weight * specular_weight;  let Edielectric = mix(subsurface_color, base_color, subsurface_weight);  let Ebase = mix(Emetal, Edielectric, base_metalness);  let Ebase_Kcoat = Ebase * Kcoat;  let one_minus_Kcoat = 1.0 - Kcoat;  let one_minus_Ebase_Kcoat = vec3<f32>(1.0) - Ebase_Kcoat;  return vec3<f32>(one_minus_Kcoat) / max(one_minus_Ebase_Kcoat, vec3<f32>(COLOR_CHANNEL_FLOOR));  }
fn coat_blend_darkened(base_darkening: vec3<f32>, coat_weight: f32, coat_dark: f32) -> vec3<f32> { if (coat_weight <= 0.0) { return vec3<f32>(1.0); }  let mix_factor = coat_weight * coat_dark;  return mix(vec3<f32>(1.0), base_darkening, mix_factor);  }
fn thin_film_modulation(cos_theta: f32, film_ior: f32, thickness_nm: f32, ior_outside: f32) -> vec3<f32> { let sin_theta_film = ior_outside * sqrt(max((1.0 - cos_theta * cos_theta), 0.0)) / film_ior;  let cos_theta_film = sqrt(max((1.0 - sin_theta_film * sin_theta_film), 0.0));  let lambda = vec3<f32>(THIN_FILM_WAVELENGTH_R_NM, THIN_FILM_WAVELENGTH_G_NM, THIN_FILM_WAVELENGTH_B_NM);  let phase = THIN_FILM_PHASE_CYCLES * PI * film_ior * thickness_nm * cos_theta_film / lambda;  let r0 = pow(((film_ior - ior_outside) / (film_ior + ior_outside)), 2.0);  return vec3<f32>(1.0) + vec3<f32>(2.0 * r0) * vec3<f32>(cos(phase.x), cos(phase.y), cos(phase.z)) / vec3<f32>(1.0 - r0 * r0);  }
fn sheen_brdf(NoV: f32, NoL: f32, NoH: f32, VoH: f32, roughness: f32) -> f32 { let alpha = roughness * roughness;  let D = alpha / (PI * pow((NoH * NoH * (alpha - 1.0) + 1.0), 2.0));  let G = 1.0 / (1.0 + alpha * (1.0 / NoV + 1.0 / NoL - 2.0));  let F = VoH;  return D * G * F / max((MICROFACET_SPECULAR_SCALE * NoV * NoL), EPS);  }
fn transmission_color_to_extinction(transmission_color: vec3<f32>, transmission_depth: f32) -> vec3<f32> { if (transmission_depth <= 0.0) { return vec3<f32>(0.0); }  let c = max(transmission_color, vec3<f32>(COLOR_CHANNEL_FLOOR));  return -log(c) / transmission_depth;  }
fn subsurface_brdf(NoV: f32, NoL: f32, distance: f32, radius: vec3<f32>, anisotropy: f32) -> vec3<f32> { let sigma_tr = sqrt(SUBSURFACE_DIFFUSION) / max(radius, vec3<f32>(EPS));  let profile = exp((-distance * sigma_tr));  let v_len2 = max((1.0 - NoV * NoV), 0.0);  let l_len2 = max((1.0 - NoL * NoL), 0.0);  let tangent_dot = (v_len2 + l_len2 - distance * distance) * LAW_OF_COSINES_HALF;  let cos_scatter = clamp((NoV * NoL + tangent_dot), -1.0, 1.0);  let phase = max((1.0 + anisotropy * cos_scatter), 0.0);  return profile * phase * INV_PI;  }
fn evaluate_ibl(prefiltered: vec3<f32>, irradiance: vec3<f32>, lut_scale: f32, lut_bias: f32, f0: vec3<f32>, diffuse_color: vec3<f32>, weight: f32) -> vec3<f32> { let fresnel = f0 * lut_scale + vec3<f32>(lut_bias);  let specular = prefiltered * fresnel;  let diffuse_weight = max((vec3<f32>(1.0) - vec3<f32>(lut_bias)), vec3<f32>(0.0));  let diffuse = irradiance * diffuse_color * diffuse_weight;  return (specular + diffuse) * weight;  }
fn evaluate_ibl_specular(prefiltered: vec3<f32>, lut_scale: f32, lut_bias: f32, f0: vec3<f32>, weight: f32) -> vec3<f32> { let fresnel = f0 * lut_scale + vec3<f32>(lut_bias);  return prefiltered * fresnel * weight;  }
fn evaluate_ibl_diffuse(irradiance: vec3<f32>, lut_bias: f32, diffuse_color: vec3<f32>, weight: f32) -> vec3<f32> { let diffuse_weight = max((vec3<f32>(1.0) - vec3<f32>(lut_bias)), vec3<f32>(0.0));  return irradiance * diffuse_color * diffuse_weight * weight;  }
@fragment
fn fs_main(input: QuadVertexOutput) -> @location(0) vec4<f32> {
let depth = textureLoad(depth_tex, vec2<u32>(input.uv * vec2<f32>(textureDimensions(depth_tex))), 0);  let albedo = textureSampleLevel(albedo_tex, lighting_sampler, input.uv, 0.0);  let normal_enc = textureSampleLevel(normal_tex, lighting_sampler, input.uv, 0.0);  let material_id = textureLoad(material_id_tex, vec2<u32>(input.uv * vec2<f32>(textureDimensions(material_id_tex))), 0).r;  let world_pos_enc = textureSampleLevel(world_pos_tex, lighting_sampler, input.uv, 0.0);  let mat_params = textureSampleLevel(mat_params_tex, lighting_sampler, input.uv, 0.0);  let mat = materials[material_id];  let n = octahedral_decode(normal_enc.rg);  let world_pos = reconstruct_world_pos(input.uv, depth, camera);  let v = normalize(camera.camera_pos.xyz - world_pos);  let ao = textureSampleLevel(ao_tex, lighting_sampler, input.uv, 0.0).r;  return shade_lit(n, world_pos, v, vec2<f32>(albedo.a, ao), mat); 
}
