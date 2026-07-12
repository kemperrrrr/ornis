pub const PBR_VERTEX: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
}

struct PerObject {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    material_index: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> per_objects: array<PerObject>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) material_index: u32,
};

@vertex
fn vs_main(
    input: VertexInput,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    let obj = per_objects[instance];
    let world_pos = obj.model * vec4<f32>(input.position, 1.0);
    var world_normal = (obj.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    world_normal = normalize(world_normal);
    var world_tangent = (obj.normal_matrix * vec4<f32>(input.tangent, 0.0)).xyz;
    world_tangent = normalize(world_tangent);

    var output: VertexOutput;
    output.clip_position = camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = world_normal;
    output.uv = input.uv;
    output.world_tangent = world_tangent;
    output.material_index = obj.material_index;
    return output;
}
"#;

pub const GBUFFER_VERTEX: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
}

struct PerObject {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    material_index: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> per_objects: array<PerObject>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) material_index: u32,
};

@vertex
fn vs_main(
    input: VertexInput,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    let obj = per_objects[instance];
    let world_pos = obj.model * vec4<f32>(input.position, 1.0);
    var world_normal = (obj.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    world_normal = normalize(world_normal);
    var world_tangent = (obj.normal_matrix * vec4<f32>(input.tangent, 0.0)).xyz;
    world_tangent = normalize(world_tangent);

    var output: VertexOutput;
    output.clip_position = camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = world_normal;
    output.uv = input.uv;
    output.world_tangent = world_tangent;
    output.material_index = obj.material_index;
    return output;
}
"#;

pub const GBUFFER_FRAGMENT: &str = r#"
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
    geometry_params2: vec4<f32>,
};

@group(0) @binding(2) var<storage, read> materials: array<OpenPBRMaterial>;

struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) material_index: u32,
};

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) material_id: u32,
    @location(3) world_pos: vec2<f32>,
    @location(4) mat_params: vec4<f32>,
};

fn octahedral_encode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    return select(p, (1.0 - abs(p.yx)) * sign(p), n.z < 0.0);
}

@fragment
fn fs_main(input: FragmentInput) -> GBufferOutput {
    let mat = materials[input.material_index];
    
    let N = normalize(input.world_normal);
    let world_pos = input.world_position;
    let base_color = mat.base_color.rgb;
    let opacity = mat.base_color.a;
    
    let normal_enc = octahedral_encode(N);
    
    let material_id = input.material_index;
    
    let world_pos_enc = vec2<f32>(world_pos.x, world_pos.y);
    
    let roughness = mat.specular_params.y;
    let metalness = mat.base_params.z;
    let specular_weight = mat.specular_params.x;
    let specular_ior = mat.specular_params.z;
    let coat_weight = mat.coat_params.x;
    let coat_roughness = mat.coat_params.y;
    let subsurface_weight = mat.subsurface_params.x;
    let transmission_weight = mat.transmission_params.x;
    let fuzz_weight = mat.fuzz_params.x;
    let thin_film_weight = mat.thin_film_params.x;
    let emission_luminance = mat.emission_params.x;
    
    let mat_params = vec4<f32>(roughness, metalness, specular_ior, coat_weight);
    
    var output: GBufferOutput;
    output.albedo = vec4<f32>(base_color, opacity);
    output.normal = normal_enc;
    output.material_id = material_id;
    output.world_pos = world_pos_enc;
    output.mat_params = mat_params;
    return output;
}
"#;

pub const LIGHTING_VERTEX: &str = r#"
const QUAD: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(-1.0, -1.0, 0.0, 1.0),
    vec4<f32>( 1.0, -1.0, 0.0, 1.0),
    vec4<f32>(-1.0,  1.0, 0.0, 1.0),
    vec4<f32>( 1.0,  1.0, 0.0, 1.0),
);

const UVS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
);

struct LightingVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> LightingVertexOutput {
    return LightingVertexOutput(QUAD[idx], UVS[idx]);
}
"#;

pub const LIGHTING_FRAGMENT: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

struct Light {
    direction: vec4<f32>,
    color: vec4<f32>,
};

struct Lighting {
    ambient_color: vec4<f32>,
    lights: array<Light, 4>,
    light_count: u32,
};

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
    geometry_params2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;
@group(0) @binding(2) var<storage, read> materials: array<OpenPBRMaterial>;
@group(0) @binding(3) var albedo_tex: texture_2d<f32>;
@group(0) @binding(4) var normal_tex: texture_2d<f32>;
@group(0) @binding(5) var material_id_tex: texture_2d<u32>;
@group(0) @binding(6) var world_pos_tex: texture_2d<f32>;
@group(0) @binding(7) var mat_params_tex: texture_2d<f32>;
@group(0) @binding(8) var depth_tex: texture_depth_2d;
@group(0) @binding(9) var lighting_sampler: sampler;

const PI: f32 = 3.14159265359;
const EPS: f32 = 1e-6;
const INV_PI: f32 = 0.31830988618;

fn fresnel0_from_ior(ior: f32) -> f32 {
    let f = (ior - 1.0) / (ior + 1.0);
    return f * f;
}

fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

fn fresnel_schlick_vec(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

fn fresnel_f82_tint(cos_theta: f32, f0: vec3<f32>, f82_tint: vec3<f32>) -> vec3<f32> {
    let mu_bar = 1.0 / 7.0;
    let schlick_at_mu_bar = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - mu_bar, 5.0);
    let f82 = f82_tint * schlick_at_mu_bar;
    let numerator = cos_theta * pow(1.0 - cos_theta, 6.0);
    let denominator = mu_bar * pow(1.0 - mu_bar, 6.0);
    let scale = numerator / denominator;
    let f_schlick = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
    let f82_correction = f_schlick - vec3<f32>(scale) * (schlick_at_mu_bar - f82);
    return max(f82_correction, vec3<f32>(0.0));
}

fn ggx_ndf(NoH: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn ggx_ndf_aniso(NoH: f32, H: vec3<f32>, T: vec3<f32>, B: vec3<f32>, alpha_u: f32, alpha_v: f32) -> f32 {
    let Hu = dot(H, T);
    let Hv = dot(H, B);
    let a2u = alpha_u * alpha_u;
    let a2v = alpha_v * alpha_v;
    let denom = 1.0 + (Hu * Hu) / a2u + (Hv * Hv) / a2v;
    return 1.0 / (PI * a2u * a2v * denom * denom);
}

fn openpbr_anisotropy(roughness: f32, anisotropy: f32) -> vec2<f32> {
    let r2 = roughness * roughness;
    let aniso_inv = 1.0 - anisotropy;
    let aniso_inv_sq = aniso_inv * aniso_inv;
    let denom = aniso_inv_sq + 1.0;
    let fraction = 2.0 / denom;
    let sqrt_frac = sqrt(fraction);
    let alpha_u = r2 * sqrt_frac;
    let alpha_v = aniso_inv * alpha_u;
    return vec2<f32>(alpha_u, alpha_v);
}

fn smith_ggx_correlated(NoV: f32, NoL: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ggxv = NoV * sqrt(max(NoL * NoL * (1.0 - a2) + a2, EPS));
    let ggxl = NoL * sqrt(max(NoV * NoV * (1.0 - a2) + a2, EPS));
    return 0.5 / max(ggxv + ggxl, EPS);
}

fn smith_ggx_aniso(NoV: f32, NoL: f32, V: vec3<f32>, L: vec3<f32>, T: vec3<f32>, B: vec3<f32>, alpha_u: f32, alpha_v: f32) -> f32 {
    let Vu = dot(V, T);
    let Vv = dot(V, B);
    let Lu = dot(L, T);
    let Lv = dot(L, B);
    let a2u = alpha_u * alpha_u;
    let a2v = alpha_v * alpha_v;
    let ggxv = NoV * sqrt(max(Lu * Lu * a2u + Lv * Lv * a2v + NoL * NoL, EPS));
    let ggxl = NoL * sqrt(max(Vu * Vu * a2u + Vv * Vv * a2v + NoV * NoV, EPS));
    return 0.5 / max(ggxv + ggxl, EPS);
}

fn oren_nayar_brdf(NoV: f32, NoL: f32, cos_phi: f32, alpha: f32) -> f32 {
    let sigma = max(alpha, EPS);
    let sigma2 = sigma * sigma;
    let A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.57);
    let B = 0.45 * sigma2 / (sigma2 + 0.09);
    let theta_v = acos(max(NoV, 0.0));
    let theta_l = acos(max(NoL, 0.0));
    let alpha_max = max(theta_v, theta_l);
    let beta_min = min(theta_v, theta_l);
    let tan_beta = tan(beta_min);
    return (A + B * cos_phi * sin(alpha_max) * tan_beta) * INV_PI;
}

fn coat_darkening(
    coat_ior: f32,
    coat_weight: f32,
    coat_darkening: f32,
    base_metalness: f32,
    base_color: vec3<f32>,
    base_weight: f32,
    specular_weight: f32,
    subsurface_weight: f32,
    subsurface_color: vec3<f32>
) -> vec3<f32> {
    let coat_f0 = fresnel0_from_ior(coat_ior);
    let one_minus_coat_f0 = 1.0 - coat_f0;
    let coat_ior_sq = coat_ior * coat_ior;
    let Kcoat = 1.0 - one_minus_coat_f0 / coat_ior_sq;

    let Emetal = base_color * base_weight * specular_weight;
    let Edielectric = mix(subsurface_color, base_color, subsurface_weight);
    let Ebase = mix(Emetal, Edielectric, base_metalness);

    let Ebase_Kcoat = Ebase * Kcoat;
    let one_minus_Kcoat = 1.0 - Kcoat;
    let one_minus_Ebase_Kcoat = vec3<f32>(1.0) - Ebase_Kcoat;
    let base_darkening = vec3<f32>(one_minus_Kcoat) / max(one_minus_Ebase_Kcoat, vec3<f32>(1e-6));

    let mix_factor = coat_weight * coat_darkening;
    return mix(vec3<f32>(1.0), base_darkening, vec3<f32>(mix_factor));
}

fn thin_film_modulation(cos_theta: f32, film_ior: f32, thickness_nm: f32, ior_outside: f32) -> vec3<f32> {
    let sin_theta_film = ior_outside * sqrt(max(1.0 - cos_theta * cos_theta, 0.0)) / film_ior;
    let cos_theta_film = sqrt(max(1.0 - sin_theta_film * sin_theta_film, 0.0));
    let lambda = vec3<f32>(650.0, 550.0, 450.0);
    let phase = 4.0 * PI * film_ior * thickness_nm * cos_theta_film / lambda;
    let r0 = pow((film_ior - ior_outside) / (film_ior + ior_outside), 2.0);
    let modulation = 1.0 + 2.0 * r0 * cos(phase) / (1.0 - r0 * r0);
    return modulation;
}

fn sheen_brdf(NoV: f32, NoL: f32, NoH: f32, VoH: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let D = alpha / (PI * pow(NoH * NoH * (alpha - 1.0) + 1.0, 2.0));
    let G = 1.0 / (1.0 + alpha * (1.0 / NoV + 1.0 / NoL - 2.0));
    let F = VoH;
    return D * G * F / max(4.0 * NoV * NoL, EPS);
}

fn transmission_btdf(
    NoV: f32, NoL: f32, VoH: f32,
    ior_in: f32, ior_out: f32,
    alpha: f32, extinction: vec3<f32>, distance: f32
) -> vec3<f32> {
    let eta = ior_in / ior_out;
    let cos_theta_t = sqrt(max(1.0 - eta * eta * (1.0 - NoV * NoV), 0.0));
    let cos_theta_i = NoV;
    let f = fresnel_schlick(max(cos_theta_i, EPS), fresnel0_from_ior(ior_in));
    let T = 1.0 - f;
    let D = ggx_ndf(VoH, alpha);
    let G = smith_ggx_correlated(NoV, NoL, alpha);
    let extinction_factor = exp(-extinction * distance);
    return vec3<f32>(D * G * T / max(4.0 * NoV * NoL, EPS)) * extinction_factor;
}

fn transmission_color_to_extinction(transmission_color: vec3<f32>, transmission_depth: f32) -> vec3<f32> {
    if transmission_depth <= 0.0 { return vec3<f32>(0.0); }
    let c = max(transmission_color, vec3<f32>(1e-6));
    return -log(c) / transmission_depth;
}

fn subsurface_brdf(
    NoV: f32, NoL: f32, distance: f32,
    radius: vec3<f32>, anisotropy: f32
) -> vec3<f32> {
    let sigma_tr = sqrt(3.0) / radius;
    let profile = exp(-distance * sigma_tr) / max(distance, EPS);
    let phase = 1.0 + anisotropy * cos(distance);
    return profile * phase * INV_PI;
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn octahedral_decode(p: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    let t = max(-n.z, 0.0);
    let offset = select(-n.yx, n.yx, n.xy >= vec2<f32>(0.0)) * t;
    n.x += offset.x;
    n.y += offset.y;
    return normalize(n);
}

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32, camera: Camera) -> vec3<f32> {
    let ndc = vec3<f32>(uv * 2.0 - 1.0, depth * 2.0 - 1.0);
    let clip = vec4<f32>(ndc, 1.0);
    let view = camera.inv_view_proj * clip;
    return view.xyz / view.w;
}

fn evaluate_base_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, H: vec3<f32>,
    NoV: f32, NoL: f32, NoH: f32, VoH: f32,
    mat: OpenPBRMaterial,
    base_weight: f32, base_color: vec3<f32>, metalness: f32,
    diffuse_roughness: f32,
    specular_weight: f32, specular_roughness: f32, specular_ior: f32, specular_anisotropy: f32, specular_edge_tint: vec3<f32>,
    T: vec3<f32>, B: vec3<f32>,
    thin_film_mod: vec3<f32>
) -> vec3<f32> {
    let F0_dielectric = vec3<f32>(fresnel0_from_ior(specular_ior));
    let F0_metal = base_color * base_weight;
    let F0 = mix(F0_dielectric, F0_metal, metalness);

    let F = fresnel_f82_tint(VoH, F0, specular_edge_tint);

    let alpha = openpbr_anisotropy(specular_roughness, specular_anisotropy);
    let alpha_u = alpha.x;
    let alpha_v = alpha.y;

    let D = ggx_ndf_aniso(NoH, H, T, B, alpha_u, alpha_v);
    let G = smith_ggx_aniso(NoV, NoL, V, L, T, B, alpha_u, alpha_v);
    let spec_brdf = D * G * F / max(4.0 * NoV * NoL, EPS);

    let diffuse_color = base_color * (1.0 - metalness);
    let diff_roughness = max(diffuse_roughness, specular_roughness);
    let diff_alpha = diff_roughness * diff_roughness;
    let cos_phi = max(dot(normalize(V - N * NoV), normalize(L - N * NoL)), 0.0);
    let diff_brdf = oren_nayar_brdf(NoV, NoL, cos_phi, diff_alpha);

    let kS = F * specular_weight;
    let kD = (vec3<f32>(1.0) - luminance(kS)) * (1.0 - metalness);
    let base_bsdf = kD * diff_brdf * diffuse_color + kS * spec_brdf;

    return base_bsdf * base_weight * thin_film_mod;
}

fn evaluate_coat_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, H: vec3<f32>,
    NoV: f32, NoL: f32, NoH: f32, VoH: f32,
    mat: OpenPBRMaterial,
    coat_weight: f32, coat_roughness: f32, coat_anisotropy: f32, coat_darkening: f32, coat_ior: f32, coat_color: vec3<f32>,
    base_metalness: f32, base_color: vec3<f32>, base_weight: f32, specular_weight: f32,
    subsurface_weight: f32, subsurface_color: vec3<f32>,
    T: vec3<f32>, B: vec3<f32>
) -> vec3<f32> {
    if coat_weight <= 0.0 { return vec3<f32>(0.0); }

    let coat_F0 = vec3<f32>(fresnel0_from_ior(coat_ior));
    let coat_F = fresnel_schlick_vec(VoH, coat_F0);

    let coat_alpha = openpbr_anisotropy(coat_roughness, coat_anisotropy);
    let coat_alpha_u = coat_alpha.x;
    let coat_alpha_v = coat_alpha.y;

    let coat_D = ggx_ndf_aniso(NoH, H, T, B, coat_alpha_u, coat_alpha_v);
    let coat_G = smith_ggx_aniso(NoV, NoL, V, L, T, B, coat_alpha_u, coat_alpha_v);
    let coat_brdf = coat_D * coat_G * coat_F / max(4.0 * NoV * NoL, EPS);

    let darkening = coat_darkening(
        coat_ior, coat_weight, coat_darkening,
        base_metalness, base_color, base_weight, specular_weight,
        subsurface_weight, subsurface_color
    );

    let coat_albedo_approx = coat_color * coat_weight * luminance(coat_F0);
    return coat_color * coat_brdf * coat_weight + darkening * coat_albedo_approx;
}

fn evaluate_fuzz_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, H: vec3<f32>,
    NoV: f32, NoL: f32, NoH: f32, VoH: f32,
    fuzz_weight: f32, fuzz_roughness: f32, fuzz_color: vec3<f32>
) -> vec3<f32> {
    if fuzz_weight <= 0.0 { return vec3<f32>(0.0); }
    let sheen = sheen_brdf(NoV, NoL, NoH, VoH, fuzz_roughness);
    return fuzz_color * sheen * fuzz_weight;
}

fn evaluate_transmission_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, H: vec3<f32>,
    NoV: f32, NoL: f32, NoH: f32, VoH: f32,
    mat: OpenPBRMaterial,
    transmission_weight: f32, transmission_depth: f32, transmission_dispersion_scale: f32, transmission_dispersion_abbe: f32,
    transmission_color: vec3<f32>, transmission_scatter: vec3<f32>, transmission_scatter_anisotropy: f32,
    specular_ior: f32, specular_roughness: f32, specular_anisotropy: f32,
    thin_walled: f32
) -> vec3<f32> {
    if transmission_weight <= 0.0 { return vec3<f32>(0.0); }

    let ior_out = 1.0;
    let ior_in = specular_ior;
    let eta = ior_in / ior_out;

    let alpha = openpbr_anisotropy(specular_roughness, specular_anisotropy);
    let extinction = transmission_color_to_extinction(transmission_color, transmission_depth);

    let distance = transmission_depth;
    let btdf = transmission_btdf(NoV, NoL, VoH, ior_in, ior_out, alpha.x, extinction, distance);

    return transmission_color * btdf * transmission_weight;
}

fn evaluate_subsurface_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>,
    NoV: f32, NoL: f32,
    subsurface_weight: f32, subsurface_radius: f32, subsurface_radius_scale_r: f32,
    subsurface_radius_scale_g: f32, subsurface_radius_scale_b: f32,
    subsurface_scatter_anisotropy: f32, subsurface_color: vec3<f32>
) -> vec3<f32> {
    if subsurface_weight <= 0.0 { return vec3<f32>(0.0); }
    let radius = vec3<f32>(
        subsurface_radius * subsurface_radius_scale_r,
        subsurface_radius * subsurface_radius_scale_g,
        subsurface_radius * subsurface_radius_scale_b
    );
    let V_proj = V - N * NoV;
    let L_proj = L - N * NoL;
    let distance = length(V_proj - L_proj);
    let ss_brdf = subsurface_brdf(NoV, NoL, distance, radius, subsurface_scatter_anisotropy);
    return subsurface_color * ss_brdf * subsurface_weight;
}

fn evaluate_emission(
    emission_luminance: f32, emission_color: vec3<f32>,
    coat_weight: f32, coat_color: vec3<f32>,
    NoV: f32
) -> vec3<f32> {
    if emission_luminance <= 0.0 { return vec3<f32>(0.0); }
    let base_emission = emission_color * emission_luminance * INV_PI;
    let coat_emission = coat_color * base_emission * (pow(1.0 - NoV, 5.0) * coat_weight + (1.0 - coat_weight));
    return mix(base_emission, coat_emission, coat_weight);
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let depth = textureLoad(depth_tex, vec2<u32>(uv * vec2<f32>(textureDimensions(depth_tex))), 0);
    
    let albedo = textureSampleLevel(albedo_tex, lighting_sampler, uv, 0.0);
    let normal_enc = textureSampleLevel(normal_tex, lighting_sampler, uv, 0.0);
    let material_id = textureLoad(material_id_tex, vec2<u32>(uv * vec2<f32>(textureDimensions(material_id_tex))), 0).r;
    let world_pos_enc = textureSampleLevel(world_pos_tex, lighting_sampler, uv, 0.0);
    let mat_params = textureSampleLevel(mat_params_tex, lighting_sampler, uv, 0.0);
    
    let mat = materials[material_id];
    
    if albedo.a < 0.001 {
        discard;
    }
    
    let N = octahedral_decode(normal_enc.rg);
    let world_pos = reconstruct_world_pos(uv, depth, camera);
    let V = normalize(camera.camera_pos.xyz - world_pos);
    let NoV = max(dot(N, V), EPS);
    
    let base_weight = mat.base_params.x;
    let base_color = mat.base_color.rgb;
    let metalness = mat.base_params.z;
    let diffuse_roughness = mat.base_params.y;
    
    let specular_weight = mat.specular_params.x;
    let specular_roughness = mat.specular_params.y;
    let specular_ior = mat.specular_params.z;
    let specular_anisotropy = mat.specular_params.w;
    let specular_edge_tint = mat.specular_color.rgb;
    
    let transmission_weight = mat.transmission_params.x;
    let transmission_depth = mat.transmission_params.y;
    let transmission_dispersion_scale = mat.transmission_params.z;
    let transmission_dispersion_abbe = mat.transmission_params.w;
    let transmission_color = mat.transmission_color.rgb;
    let transmission_scatter = mat.transmission_scatter.rgb;
    let transmission_scatter_anisotropy = mat.transmission_scatter.a;
    
    let subsurface_weight = mat.subsurface_params.x;
    let subsurface_radius = mat.subsurface_params.y;
    let subsurface_radius_scale_r = mat.subsurface_params.z;
    let subsurface_scatter_anisotropy = mat.subsurface_params.w;
    let subsurface_color = mat.subsurface_color.rgb;
    let subsurface_radius_scale_g = mat.subsurface_radius_scale_gb.x;
    let subsurface_radius_scale_b = mat.subsurface_radius_scale_gb.y;
    
    let fuzz_weight = mat.fuzz_params.x;
    let fuzz_roughness = mat.fuzz_params.y;
    let fuzz_color = mat.fuzz_color.rgb;
    
    let coat_weight = mat.coat_params.x;
    let coat_roughness = mat.coat_params.y;
    let coat_anisotropy = mat.coat_params.z;
    let coat_darkening = mat.coat_params.w;
    let coat_color = mat.coat_color.rgb;
    let coat_ior = mat.coat_ior.x;
    
    let thin_film_weight = mat.thin_film_params.x;
    let thin_film_thickness_um = mat.thin_film_params.y;
    let thin_film_ior = mat.thin_film_params.z;
    
    let emission_luminance = mat.emission_params.x;
    let emission_color = mat.emission_color.rgb;
    
    let opacity = mat.geometry_params.x;
    let thin_walled = mat.geometry_params.y;
    
    var Lo = vec3<f32>(0.0);
    
    let thin_film_mod = thin_film_modulation(NoV, thin_film_ior, thin_film_thickness_um, 1.0);
    
    let T = normalize(cross(N, vec3<f32>(0.0, 1.0, 0.0)));
    let B = cross(N, T);
    
    for (var i = 0u; i < lighting.light_count; i = i + 1u) {
        let L = normalize(lighting.lights[i].direction.xyz);
        let H = normalize(V + L);
        let light_color = lighting.lights[i].color.rgb;
        let intensity = lighting.lights[i].color.w;
        let radiance = light_color * intensity;
        
        let NoL = max(dot(N, L), EPS);
        let NoH = max(dot(N, H), EPS);
        let VoH = max(dot(V, H), EPS);
        
        if (NoL <= EPS) { continue; }
        
        let base_bsdf = evaluate_base_layer(
            N, V, L, H, NoV, NoL, NoH, VoH, mat,
            base_weight, base_color, metalness, diffuse_roughness,
            specular_weight, specular_roughness, specular_ior, specular_anisotropy, specular_edge_tint,
            T, B, thin_film_mod
        );
        
        let coat_bsdf = evaluate_coat_layer(
            N, V, L, H, NoV, NoL, NoH, VoH, mat,
            coat_weight, coat_roughness, coat_anisotropy, coat_darkening, coat_ior, coat_color,
            metalness, base_color, base_weight, specular_weight,
            subsurface_weight, subsurface_color,
            T, B
        );
        
        let fuzz_bsdf = evaluate_fuzz_layer(
            N, V, L, H, NoV, NoL, NoH, VoH,
            fuzz_weight, fuzz_roughness, fuzz_color
        );
        
        let trans_bsdf = evaluate_transmission_layer(
            N, V, L, H, NoV, NoL, NoH, VoH, mat,
            transmission_weight, transmission_depth, transmission_dispersion_scale, transmission_dispersion_abbe,
            transmission_color, transmission_scatter, transmission_scatter_anisotropy,
            specular_ior, specular_roughness, specular_anisotropy,
            thin_walled
        );
        
        let ss_bsdf = evaluate_subsurface_layer(
            N, V, L, NoV, NoL,
            subsurface_weight, subsurface_radius, subsurface_radius_scale_r,
            subsurface_radius_scale_g, subsurface_radius_scale_b,
            subsurface_scatter_anisotropy, subsurface_color
        );
        
        let layer_bsdf = base_bsdf + coat_bsdf + fuzz_bsdf + trans_bsdf + ss_bsdf;
        Lo += layer_bsdf * radiance * NoL;
    }
    
    let ambient = lighting.ambient_color.rgb * mix(base_color, base_color * specular_weight, metalness);
    let emission = evaluate_emission(emission_luminance, emission_color, coat_weight, coat_color, NoV);
    
    let color = ambient + Lo + emission;
    let tone_mapped = aces_tonemap(color);
    return vec4<f32>(tone_mapped, opacity);
}
"#;

pub const COMPOSITE_FRAGMENT: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

@group(0) @binding(0) var deferred_tex: texture_2d<f32>;
@group(0) @binding(1) var forward_tex: texture_2d<f32>;
@group(0) @binding(2) var composite_sampler: sampler;

const QUAD: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(-1.0, -1.0, 0.0, 1.0),
    vec4<f32>( 1.0, -1.0, 0.0, 1.0),
    vec4<f32>(-1.0,  1.0, 0.0, 1.0),
    vec4<f32>( 1.0,  1.0, 0.0, 1.0),
);

const UVS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
);

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

struct QuadVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> QuadVertexOutput {
    return QuadVertexOutput(QUAD[idx], UVS[idx]);
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let deferred_color = textureSample(deferred_tex, composite_sampler, uv).rgb;
    let forward_color = textureSample(forward_tex, composite_sampler, uv).rgba;
    
    let combined = deferred_color + forward_color.rgb * forward_color.a;
    let tonemapped = aces_tonemap(combined);
    
    return vec4<f32>(tonemapped, forward_color.a);
}
"#;

pub const PBR_FRAGMENT: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
}

struct Light {
    direction: vec4<f32>,
    color: vec4<f32>,
};

struct Lighting {
    ambient_color: vec4<f32>,
    lights: array<Light, 4>,
    light_count: u32,
};

struct OpenPBRMaterial {
    // Base
    base_params: vec4<f32>,       // weight, diffuse_roughness, metalness, _
    base_color: vec4<f32>,        // rgb, a=opacity

    // Specular
    specular_params: vec4<f32>,   // weight, roughness, ior, anisotropy
    specular_color: vec4<f32>,    // edge_tint, _

    // Transmission
    transmission_params: vec4<f32>,  // weight, depth, disp_scale, disp_abbe
    transmission_color: vec4<f32>,   // rgb, _
    transmission_scatter: vec4<f32>, // rgb=scatter_color, a=scatter_anisotropy

    // Subsurface
    subsurface_params: vec4<f32>,    // weight, radius, radius_scale_r, scatter_aniso
    subsurface_color: vec4<f32>,     // rgb, _
    subsurface_radius_scale_gb: vec4<f32>, // g, b, _, _

    // Fuzz
    fuzz_params: vec4<f32>,          // weight, roughness, _, _
    fuzz_color: vec4<f32>,           // rgb, _

    // Coat
    coat_params: vec4<f32>,          // weight, roughness, anisotropy, darkening
    coat_color: vec4<f32>,           // rgb, _
    coat_ior: vec4<f32>,             // x=ior, yzw=unused

    // Thin Film
    thin_film_params: vec4<f32>,     // weight, thickness_um, ior, _

    // Emission
    emission_params: vec4<f32>,      // luminance_nits, _, _, _
    emission_color: vec4<f32>,       // rgb, _

    // Geometry
    geometry_params: vec4<f32>,      // opacity, thin_walled, _, _
    geometry_params2: vec4<f32>,     // reserved for future use
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> materials: array<OpenPBRMaterial>;
@group(0) @binding(3) var<uniform> lighting: Lighting;

struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) material_index: u32,
};

const PI: f32 = 3.14159265359;
const EPS: f32 = 1e-6;
const INV_PI: f32 = 0.31830988618;

fn fresnel0_from_ior(ior: f32) -> f32 {
    let f = (ior - 1.0) / (ior + 1.0);
    return f * f;
}

fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

fn fresnel_schlick_vec(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

fn fresnel_f82_tint(cos_theta: f32, f0: vec3<f32>, f82_tint: vec3<f32>) -> vec3<f32> {
    let mu_bar = 1.0 / 7.0;
    let schlick_at_mu_bar = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - mu_bar, 5.0);
    let f82 = f82_tint * schlick_at_mu_bar;
    let numerator = cos_theta * pow(1.0 - cos_theta, 6.0);
    let denominator = mu_bar * pow(1.0 - mu_bar, 6.0);
    let scale = numerator / denominator;
    let f_schlick = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
    let f82_correction = f_schlick - vec3<f32>(scale) * (schlick_at_mu_bar - f82);
    return max(f82_correction, vec3<f32>(0.0));
}

fn ggx_ndf(NoH: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn ggx_ndf_aniso(NoH: f32, H: vec3<f32>, T: vec3<f32>, B: vec3<f32>, alpha_u: f32, alpha_v: f32) -> f32 {
    let Hu = dot(H, T);
    let Hv = dot(H, B);
    let a2u = alpha_u * alpha_u;
    let a2v = alpha_v * alpha_v;
    let denom = 1.0 + (Hu * Hu) / a2u + (Hv * Hv) / a2v;
    return 1.0 / (PI * a2u * a2v * denom * denom);
}

fn openpbr_anisotropy(roughness: f32, anisotropy: f32) -> vec2<f32> {
    let r2 = roughness * roughness;
    let aniso_inv = 1.0 - anisotropy;
    let aniso_inv_sq = aniso_inv * aniso_inv;
    let denom = aniso_inv_sq + 1.0;
    let fraction = 2.0 / denom;
    let sqrt_frac = sqrt(fraction);
    let alpha_u = r2 * sqrt_frac;
    let alpha_v = aniso_inv * alpha_u;
    return vec2<f32>(alpha_u, alpha_v);
}

fn smith_ggx_correlated(NoV: f32, NoL: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ggxv = NoV * sqrt(max(NoL * NoL * (1.0 - a2) + a2, EPS));
    let ggxl = NoL * sqrt(max(NoV * NoV * (1.0 - a2) + a2, EPS));
    return 0.5 / max(ggxv + ggxl, EPS);
}

fn smith_ggx_aniso(NoV: f32, NoL: f32, V: vec3<f32>, L: vec3<f32>, T: vec3<f32>, B: vec3<f32>, alpha_u: f32, alpha_v: f32) -> f32 {
    let Vu = dot(V, T);
    let Vv = dot(V, B);
    let Lu = dot(L, T);
    let Lv = dot(L, B);
    let a2u = alpha_u * alpha_u;
    let a2v = alpha_v * alpha_v;
    let ggxv = NoV * sqrt(max(Lu * Lu * a2u + Lv * Lv * a2v + NoL * NoL, EPS));
    let ggxl = NoL * sqrt(max(Vu * Vu * a2u + Vv * Vv * a2v + NoV * NoV, EPS));
    return 0.5 / max(ggxv + ggxl, EPS);
}

fn oren_nayar_brdf(NoV: f32, NoL: f32, cos_phi: f32, alpha: f32) -> f32 {
    let sigma = max(alpha, EPS);
    let sigma2 = sigma * sigma;
    let A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.57);
    let B = 0.45 * sigma2 / (sigma2 + 0.09);
    let theta_v = acos(max(NoV, 0.0));
    let theta_l = acos(max(NoL, 0.0));
    let alpha_max = max(theta_v, theta_l);
    let beta_min = min(theta_v, theta_l);
    let tan_beta = tan(beta_min);
    return (A + B * cos_phi * sin(alpha_max) * tan_beta) * INV_PI;
}

fn coat_darkening(
    coat_ior: f32,
    coat_weight: f32,
    coat_darkening: f32,
    base_metalness: f32,
    base_color: vec3<f32>,
    base_weight: f32,
    specular_weight: f32,
    subsurface_weight: f32,
    subsurface_color: vec3<f32>
) -> vec3<f32> {
    let coat_f0 = fresnel0_from_ior(coat_ior);
    let one_minus_coat_f0 = 1.0 - coat_f0;
    let coat_ior_sq = coat_ior * coat_ior;
    let Kcoat = 1.0 - one_minus_coat_f0 / coat_ior_sq;

    let Emetal = base_color * base_weight * specular_weight;
    let Edielectric = mix(subsurface_color, base_color, subsurface_weight);
    let Ebase = mix(Emetal, Edielectric, base_metalness);

    let Ebase_Kcoat = Ebase * Kcoat;
    let one_minus_Kcoat = 1.0 - Kcoat;
    let one_minus_Ebase_Kcoat = vec3<f32>(1.0) - Ebase_Kcoat;
    let base_darkening = vec3<f32>(one_minus_Kcoat) / max(one_minus_Ebase_Kcoat, vec3<f32>(1e-6));

    let mix_factor = coat_weight * coat_darkening;
    return mix(vec3<f32>(1.0), base_darkening, vec3<f32>(mix_factor));
}

fn thin_film_modulation(cos_theta: f32, film_ior: f32, thickness_nm: f32, ior_outside: f32) -> vec3<f32> {
    let sin_theta_film = ior_outside * sqrt(max(1.0 - cos_theta * cos_theta, 0.0)) / film_ior;
    let cos_theta_film = sqrt(max(1.0 - sin_theta_film * sin_theta_film, 0.0));
    let lambda = vec3<f32>(650.0, 550.0, 450.0);
    let phase = 4.0 * PI * film_ior * thickness_nm * cos_theta_film / lambda;
    let r0 = pow((film_ior - ior_outside) / (film_ior + ior_outside), 2.0);
    let modulation = 1.0 + 2.0 * r0 * cos(phase) / (1.0 - r0 * r0);
    return modulation;
}

fn sheen_brdf(NoV: f32, NoL: f32, NoH: f32, VoH: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let D = alpha / (PI * pow(NoH * NoH * (alpha - 1.0) + 1.0, 2.0));
    let G = 1.0 / (1.0 + alpha * (1.0 / NoV + 1.0 / NoL - 2.0));
    let F = VoH;
    return D * G * F / max(4.0 * NoV * NoL, EPS);
}

fn transmission_btdf(
    NoV: f32, NoL: f32, VoH: f32,
    ior_in: f32, ior_out: f32,
    alpha: f32, extinction: vec3<f32>, distance: f32
) -> vec3<f32> {
    let eta = ior_in / ior_out;
    let cos_theta_t = sqrt(max(1.0 - eta * eta * (1.0 - NoV * NoV), 0.0));
    let cos_theta_i = NoV;
    let f = fresnel_schlick(max(cos_theta_i, EPS), fresnel0_from_ior(ior_in));
    let T = 1.0 - f;
    let D = ggx_ndf(VoH, alpha);
    let G = smith_ggx_correlated(NoV, NoL, alpha);
    let extinction_factor = exp(-extinction * distance);
    return vec3<f32>(D * G * T / max(4.0 * NoV * NoL, EPS)) * extinction_factor;
}

fn subsurface_brdf(
    NoV: f32, NoL: f32, distance: f32,
    radius: vec3<f32>, anisotropy: f32
) -> vec3<f32> {
    let sigma_tr = sqrt(3.0) / radius;
    let profile = exp(-distance * sigma_tr) / max(distance, EPS);
    let phase = 1.0 + anisotropy * cos(distance);
    return profile * phase * INV_PI;
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        select(c.x / 12.92, pow((c.x + 0.055) / 1.055, 2.4), c.x > 0.04045),
        select(c.y / 12.92, pow((c.y + 0.055) / 1.055, 2.4), c.y > 0.04045),
        select(c.z / 12.92, pow((c.z + 0.055) / 1.055, 2.4), c.z > 0.04045)
    );
}

fn evaluate_base_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, H: vec3<f32>,
    NoV: f32, NoL: f32, NoH: f32, VoH: f32,
    mat: OpenPBRMaterial,
    base_weight: f32, base_color: vec3<f32>, metalness: f32,
    diffuse_roughness: f32,
    specular_weight: f32, specular_roughness: f32, specular_ior: f32, specular_anisotropy: f32, specular_edge_tint: vec3<f32>,
    T: vec3<f32>, B: vec3<f32>,
    thin_film_mod: vec3<f32>
) -> vec3<f32> {
    let F0_dielectric = vec3<f32>(fresnel0_from_ior(specular_ior));
    let F0_metal = base_color * base_weight;
    let F0 = mix(F0_dielectric, F0_metal, metalness);

    let F = fresnel_f82_tint(VoH, F0, specular_edge_tint);

    let alpha = openpbr_anisotropy(specular_roughness, specular_anisotropy);
    let alpha_u = alpha.x;
    let alpha_v = alpha.y;

    let D = ggx_ndf_aniso(NoH, H, T, B, alpha_u, alpha_v);
    let G = smith_ggx_aniso(NoV, NoL, V, L, T, B, alpha_u, alpha_v);
    let spec_brdf = D * G * F / max(4.0 * NoV * NoL, EPS);

    let diffuse_color = base_color * (1.0 - metalness);
    let diff_roughness = max(diffuse_roughness, specular_roughness);
    let diff_alpha = diff_roughness * diff_roughness;
    let cos_phi = max(dot(normalize(V - N * NoV), normalize(L - N * NoL)), 0.0);
    let diff_brdf = oren_nayar_brdf(NoV, NoL, cos_phi, diff_alpha);

    let kS = F * specular_weight;
    let kD = (vec3<f32>(1.0) - luminance(kS)) * (1.0 - metalness);
    let base_bsdf = kD * diff_brdf * diffuse_color + kS * spec_brdf;

    return base_bsdf * base_weight * thin_film_mod;
}

fn evaluate_coat_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, H: vec3<f32>,
    NoV: f32, NoL: f32, NoH: f32, VoH: f32,
    mat: OpenPBRMaterial,
    coat_weight: f32, coat_roughness: f32, coat_anisotropy: f32, coat_darkening: f32, coat_ior: f32, coat_color: vec3<f32>,
    base_metalness: f32, base_color: vec3<f32>, base_weight: f32, specular_weight: f32,
    subsurface_weight: f32, subsurface_color: vec3<f32>,
    T: vec3<f32>, B: vec3<f32>
) -> vec3<f32> {
    if coat_weight <= 0.0 { return vec3<f32>(0.0); }

    let coat_F0 = vec3<f32>(fresnel0_from_ior(coat_ior));
    let coat_F = fresnel_schlick_vec(VoH, coat_F0);

    let coat_alpha = openpbr_anisotropy(coat_roughness, coat_anisotropy);
    let coat_alpha_u = coat_alpha.x;
    let coat_alpha_v = coat_alpha.y;

    let coat_D = ggx_ndf_aniso(NoH, H, T, B, coat_alpha_u, coat_alpha_v);
    let coat_G = smith_ggx_aniso(NoV, NoL, V, L, T, B, coat_alpha_u, coat_alpha_v);
    let coat_brdf = coat_D * coat_G * coat_F / max(4.0 * NoV * NoL, EPS);

    let darkening = coat_darkening(
        coat_ior, coat_weight, coat_darkening,
        base_metalness, base_color, base_weight, specular_weight,
        subsurface_weight, subsurface_color
    );

    let coat_albedo_approx = coat_color * coat_weight * luminance(coat_F0);
    return coat_color * coat_brdf * coat_weight + darkening * coat_albedo_approx;
}

fn evaluate_fuzz_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, H: vec3<f32>,
    NoV: f32, NoL: f32, NoH: f32, VoH: f32,
    fuzz_weight: f32, fuzz_roughness: f32, fuzz_color: vec3<f32>
) -> vec3<f32> {
    if fuzz_weight <= 0.0 { return vec3<f32>(0.0); }
    let sheen = sheen_brdf(NoV, NoL, NoH, VoH, fuzz_roughness);
    return fuzz_color * sheen * fuzz_weight;
}

fn evaluate_transmission_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, H: vec3<f32>,
    NoV: f32, NoL: f32, NoH: f32, VoH: f32,
    mat: OpenPBRMaterial,
    transmission_weight: f32, transmission_depth: f32, transmission_dispersion_scale: f32, transmission_dispersion_abbe: f32,
    transmission_color: vec3<f32>, transmission_scatter: vec3<f32>, transmission_scatter_anisotropy: f32,
    specular_ior: f32, specular_roughness: f32, specular_anisotropy: f32,
    thin_walled: f32
) -> vec3<f32> {
    if transmission_weight <= 0.0 { return vec3<f32>(0.0); }

    let ior_out = 1.0;
    let ior_in = specular_ior;
    let eta = ior_in / ior_out;

    let alpha = openpbr_anisotropy(specular_roughness, specular_anisotropy);
    let extinction = transmission_color_to_extinction(transmission_color, transmission_depth);

    let distance = transmission_depth;
    let btdf = transmission_btdf(NoV, NoL, VoH, ior_in, ior_out, alpha.x, extinction, distance);

    return transmission_color * btdf * transmission_weight;
}

fn transmission_color_to_extinction(transmission_color: vec3<f32>, transmission_depth: f32) -> vec3<f32> {
    if transmission_depth <= 0.0 { return vec3<f32>(0.0); }
    let c = max(transmission_color, vec3<f32>(1e-6));
    return -log(c) / transmission_depth;
}

fn evaluate_subsurface_layer(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>,
    NoV: f32, NoL: f32,
    subsurface_weight: f32, subsurface_radius: f32, subsurface_radius_scale_r: f32,
    subsurface_radius_scale_g: f32, subsurface_radius_scale_b: f32,
    subsurface_scatter_anisotropy: f32, subsurface_color: vec3<f32>
) -> vec3<f32> {
    if subsurface_weight <= 0.0 { return vec3<f32>(0.0); }
    let radius = vec3<f32>(
        subsurface_radius * subsurface_radius_scale_r,
        subsurface_radius * subsurface_radius_scale_g,
        subsurface_radius * subsurface_radius_scale_b
    );
    let V_proj = V - N * NoV;
    let L_proj = L - N * NoL;
    let distance = length(V_proj - L_proj);
    let ss_brdf = subsurface_brdf(NoV, NoL, distance, radius, subsurface_scatter_anisotropy);
    return subsurface_color * ss_brdf * subsurface_weight;
}

fn evaluate_emission(
    emission_luminance: f32, emission_color: vec3<f32>,
    coat_weight: f32, coat_color: vec3<f32>,
    NoV: f32
) -> vec3<f32> {
    if emission_luminance <= 0.0 { return vec3<f32>(0.0); }
    let base_emission = emission_color * emission_luminance * INV_PI;
    let coat_emission = coat_color * base_emission * (pow(1.0 - NoV, 5.0) * coat_weight + (1.0 - coat_weight));
    return mix(base_emission, coat_emission, coat_weight);
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let mat = materials[input.material_index];
    let N = normalize(input.world_normal);
    let V = normalize(camera.camera_pos.xyz - input.world_position);
    let NoV = max(dot(N, V), EPS);

    let T = normalize(input.world_tangent);
    let B = cross(N, T);

    let base_weight = mat.base_params.x;
    let base_color = mat.base_color.rgb;
    let metalness = mat.base_params.z;
    let diffuse_roughness = mat.base_params.y;

    let specular_weight = mat.specular_params.x;
    let specular_roughness = mat.specular_params.y;
    let specular_ior = mat.specular_params.z;
    let specular_anisotropy = mat.specular_params.w;
    let specular_edge_tint = mat.specular_color.rgb;

    let transmission_weight = mat.transmission_params.x;
    let transmission_depth = mat.transmission_params.y;
    let transmission_dispersion_scale = mat.transmission_params.z;
    let transmission_dispersion_abbe = mat.transmission_params.w;
    let transmission_color = mat.transmission_color.rgb;
    let transmission_scatter = mat.transmission_scatter.rgb;
    let transmission_scatter_anisotropy = mat.transmission_scatter.a;

    let subsurface_weight = mat.subsurface_params.x;
    let subsurface_radius = mat.subsurface_params.y;
    let subsurface_radius_scale_r = mat.subsurface_params.z;
    let subsurface_scatter_anisotropy = mat.subsurface_params.w;
    let subsurface_color = mat.subsurface_color.rgb;
    let subsurface_radius_scale_g = mat.subsurface_radius_scale_gb.x;
    let subsurface_radius_scale_b = mat.subsurface_radius_scale_gb.y;

    let fuzz_weight = mat.fuzz_params.x;
    let fuzz_roughness = mat.fuzz_params.y;
    let fuzz_color = mat.fuzz_color.rgb;

    let coat_weight = mat.coat_params.x;
    let coat_roughness = mat.coat_params.y;
    let coat_anisotropy = mat.coat_params.z;
    let coat_darkening = mat.coat_params.w;
    let coat_color = mat.coat_color.rgb;
    let coat_ior = mat.coat_ior.x;

    let thin_film_weight = mat.thin_film_params.x;
    let thin_film_thickness_um = mat.thin_film_params.y;
    let thin_film_ior = mat.thin_film_params.z;

    let emission_luminance = mat.emission_params.x;
    let emission_color = mat.emission_color.rgb;

    let opacity = mat.geometry_params.x;
    let thin_walled = mat.geometry_params.y;

    var Lo = vec3<f32>(0.0);

    let thin_film_mod = thin_film_modulation(NoV, thin_film_ior, thin_film_thickness_um, 1.0);

    for (var i = 0u; i < lighting.light_count; i = i + 1u) {
        let L = normalize(lighting.lights[i].direction.xyz);
        let H = normalize(V + L);
        let light_color = lighting.lights[i].color.rgb;
        let intensity = lighting.lights[i].color.w;
        let radiance = light_color * intensity;

        let NoL = max(dot(N, L), EPS);
        let NoH = max(dot(N, H), EPS);
        let VoH = max(dot(V, H), EPS);

        if (NoL <= EPS) { continue; }

        let base_bsdf = evaluate_base_layer(
            N, V, L, H, NoV, NoL, NoH, VoH, mat,
            base_weight, base_color, metalness, diffuse_roughness,
            specular_weight, specular_roughness, specular_ior, specular_anisotropy, specular_edge_tint,
            T, B, thin_film_mod
        );

        let coat_bsdf = evaluate_coat_layer(
            N, V, L, H, NoV, NoL, NoH, VoH, mat,
            coat_weight, coat_roughness, coat_anisotropy, coat_darkening, coat_ior, coat_color,
            metalness, base_color, base_weight, specular_weight,
            subsurface_weight, subsurface_color,
            T, B
        );

        let fuzz_bsdf = evaluate_fuzz_layer(
            N, V, L, H, NoV, NoL, NoH, VoH,
            fuzz_weight, fuzz_roughness, fuzz_color
        );

        let trans_bsdf = evaluate_transmission_layer(
            N, V, L, H, NoV, NoL, NoH, VoH, mat,
            transmission_weight, transmission_depth, transmission_dispersion_scale, transmission_dispersion_abbe,
            transmission_color, transmission_scatter, transmission_scatter_anisotropy,
            specular_ior, specular_roughness, specular_anisotropy,
            thin_walled
        );

        let ss_bsdf = evaluate_subsurface_layer(
            N, V, L, NoV, NoL,
            subsurface_weight, subsurface_radius, subsurface_radius_scale_r,
            subsurface_radius_scale_g, subsurface_radius_scale_b,
            subsurface_scatter_anisotropy, subsurface_color
        );

        let layer_bsdf = base_bsdf + coat_bsdf + fuzz_bsdf + trans_bsdf + ss_bsdf;
        Lo += layer_bsdf * radiance * NoL;
    }

    let ambient = lighting.ambient_color.rgb * mix(base_color, base_color * specular_weight, metalness);
    let emission = evaluate_emission(emission_luminance, emission_color, coat_weight, coat_color, NoV);

    let color = ambient + Lo + emission;
    let tone_mapped = aces_tonemap(color);
    return vec4<f32>(tone_mapped, opacity);
}
"#;

pub const COMPOSITE_VERTEX: &str = r#"
const QUAD: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(-1.0, -1.0, 0.0, 1.0),
    vec4<f32>( 1.0, -1.0, 0.0, 1.0),
    vec4<f32>(-1.0,  1.0, 0.0, 1.0),
    vec4<f32>( 1.0,  1.0, 0.0, 1.0),
);

const UVS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
);

struct CompositeVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> CompositeVertexOutput {
    return CompositeVertexOutput(QUAD[idx], UVS[idx]);
}
"#;

/// Rust version of the WGSL octahedral_decode for testing.
pub fn octahedral_decode_rust(p: glam::Vec2) -> glam::Vec3 {
    let mut n = glam::Vec3::new(p.x, p.y, 1.0 - p.x.abs() - p.y.abs());
    let t = (-n.z).max(0.0);
    // Matches WGSL: select(-n.yx, n.yx, n.xy >= vec2<f32>(0.0)) * t
    let ox = if n.x >= 0.0 { n.y } else { -n.y } * t;
    let oy = if n.y >= 0.0 { n.x } else { -n.x } * t;
    n.x += ox;
    n.y += oy;
    n.normalize()
}