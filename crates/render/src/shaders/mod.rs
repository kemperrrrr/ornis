pub mod math;

/// ── COMPOSITE_VERTEX ────────────────────────────────────────────────
const COMPOSITE_VERTEX_BOILERPLATE: &str = r#"
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

pub fn composite_vertex() -> String {
    COMPOSITE_VERTEX_BOILERPLATE.to_string()
}

/// ── COMPOSITE_FRAGMENT ──────────────────────────────────────────────
const COMPOSITE_FRAGMENT_BOILERPLATE: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

struct BloomParams {
    threshold: f32,
    intensity: f32,
    mode: u32,
    _pad: f32,
};

@group(0) @binding(0) var deferred_tex: texture_2d<f32>;
@group(0) @binding(1) var forward_tex: texture_2d<f32>;
@group(0) @binding(2) var composite_sampler: sampler;
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;
@group(0) @binding(4) var<uniform> bloom_params: BloomParams;

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

    // Layer mix depends on the technique: 0 = deferred-only, 1 = forward-only,
    // 2 = hybrid. The dead layer is bound to the live one, so the mode is
    // what disambiguates the two inputs.
    var combined = deferred_color;
    if (bloom_params.mode == 1u) {
        combined = forward_color.rgb * forward_color.a;
    } else if (bloom_params.mode == 2u) {
        combined = deferred_color + forward_color.rgb * forward_color.a;
    }
    let bloom = textureSample(bloom_tex, composite_sampler, uv).rgb;
    let tonemapped = aces_tonemap(combined + bloom * bloom_params.intensity);

    // The composited scene is opaque; forward_color.a is 0 where no forward
    // geometry was drawn, which would make the whole frame transparent on a
    // canvas/surface. Native compositing (LegacyCompositePass) also forces 1.0.
    return vec4<f32>(tonemapped, 1.0);
}
"#;

pub fn composite_fragment() -> String {
    format!(
        "{}\n{}\n{}",
        COMPOSITE_FRAGMENT_BOILERPLATE,
        math::aces_tonemap::wgsl_source(),
        math::luminance::wgsl_source(),
    )
}

/// ── BLOOM ───────────────────────────────────────────────────────────
///
/// A single quad shader used by the whole bloom chain. Each pass samples
/// the previous (smaller or larger) level and applies a soft threshold on
/// luminance: the first downsample keeps only the bright pixels
/// (threshold ≈ 0.6-0.7), later levels pass everything (threshold = 0),
/// and the upsample passes re-add the level's own content via additive
/// blending (dst = src + previous), recreating the classic Frostbite
/// "downsample chain, upsample with add" cascade.
const BLOOM_FRAGMENT_BOILERPLATE: &str = r#"
struct BloomParams {
    threshold: f32,
    intensity: f32,
    mode: u32,
    _pad: f32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> bloom_params: BloomParams;

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

struct BloomVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> BloomVertexOutput {
    return BloomVertexOutput(QUAD[idx], UVS[idx]);
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let color = textureSample(src_tex, src_sampler, uv).rgb;
    let luma = luminance(color);
    // Soft knee: pixels above `threshold` pass fully, a thin band below it
    // fades out. With threshold = 0 everything except pure black passes.
    let keep = smoothstep(bloom_params.threshold, bloom_params.threshold + 0.05, luma);
    return vec4<f32>(color * keep, 1.0);
}
"#;

pub fn bloom_fragment() -> String {
    format!(
        "{}\n{}",
        BLOOM_FRAGMENT_BOILERPLATE,
        math::luminance::wgsl_source(),
    )
}

/// ── GBUFFER_VERTEX ──────────────────────────────────────────────────
const GBUFFER_VERTEX_BOILERPLATE: &str = r#"
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
    @location(4) @interpolate(flat) material_index: u32,
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

pub fn gbuffer_vertex() -> String {
    GBUFFER_VERTEX_BOILERPLATE.to_string()
}

/// ── GBUFFER_FRAGMENT ────────────────────────────────────────────────
const GBUFFER_FRAGMENT_BOILERPLATE: &str = r#"
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
    @location(4) @interpolate(flat) material_index: u32,
};

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) material_id: u32,
    @location(3) world_pos: vec2<f32>,
    @location(4) mat_params: vec4<f32>,
};

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

pub fn gbuffer_fragment() -> String {
    format!(
        "{}\n{}",
        GBUFFER_FRAGMENT_BOILERPLATE,
        math::octahedral_encode::wgsl_source(),
    )
}

/// ── LIGHTING_VERTEX ─────────────────────────────────────────────────
const LIGHTING_VERTEX_BOILERPLATE: &str = r#"
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

pub fn lighting_vertex() -> String {
    LIGHTING_VERTEX_BOILERPLATE.to_string()
}

/// ── LIGHTING_FRAGMENT ───────────────────────────────────────────────
const LIGHTING_BOILERPLATE: &str = r#"
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
    coat_weight: f32, coat_roughness: f32, coat_anisotropy: f32, coat_dark: f32, coat_ior: f32, coat_color: vec3<f32>,
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
        coat_ior, coat_weight, coat_dark,
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

pub fn lighting_fragment() -> String {
    let kernels = [
        math::luminance::wgsl_source(),
        math::aces_tonemap::wgsl_source(),
        math::fresnel0_from_ior::wgsl_source(),
        math::fresnel_schlick::wgsl_source(),
        math::fresnel_schlick_vec::wgsl_source(),
        math::fresnel_f82_tint::wgsl_source(),
        math::ggx_ndf::wgsl_source(),
        math::ggx_ndf_aniso::wgsl_source(),
        math::openpbr_anisotropy::wgsl_source(),
        math::smith_ggx_correlated::wgsl_source(),
        math::smith_ggx_aniso::wgsl_source(),
        math::oren_nayar_brdf::wgsl_source(),
        math::coat_darkening::wgsl_source(),
        math::thin_film_modulation::wgsl_source(),
        math::sheen_brdf::wgsl_source(),
        math::transmission_color_to_extinction::wgsl_source(),
        math::subsurface_brdf::wgsl_source(),
    ];
    let mut src = LIGHTING_BOILERPLATE.to_string();
    for k in &kernels {
        src.push('\n');
        src.push_str(k);
    }
    src
}

/// ── PBR_VERTEX ──────────────────────────────────────────────────────
const PBR_VERTEX_BOILERPLATE: &str = r#"
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
    @location(4) @interpolate(flat) material_index: u32,
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

pub fn pbr_vertex() -> String {
    PBR_VERTEX_BOILERPLATE.to_string()
}

/// ── PBR_FRAGMENT ────────────────────────────────────────────────────
const PBR_FRAGMENT_BOILERPLATE: &str = r#"
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
@group(0) @binding(2) var<storage, read> materials: array<OpenPBRMaterial>;
@group(0) @binding(3) var<uniform> lighting: Lighting;

struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) @interpolate(flat) material_index: u32,
};

const PI: f32 = 3.14159265359;
const EPS: f32 = 1e-6;
const INV_PI: f32 = 0.31830988618;

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
    coat_weight: f32, coat_roughness: f32, coat_anisotropy: f32, coat_dark: f32, coat_ior: f32, coat_color: vec3<f32>,
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
        coat_ior, coat_weight, coat_dark,
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

pub fn pbr_fragment() -> String {
    let kernels = [
        math::luminance::wgsl_source(),
        math::aces_tonemap::wgsl_source(),
        math::fresnel0_from_ior::wgsl_source(),
        math::fresnel_schlick::wgsl_source(),
        math::fresnel_schlick_vec::wgsl_source(),
        math::fresnel_f82_tint::wgsl_source(),
        math::ggx_ndf::wgsl_source(),
        math::ggx_ndf_aniso::wgsl_source(),
        math::openpbr_anisotropy::wgsl_source(),
        math::smith_ggx_correlated::wgsl_source(),
        math::smith_ggx_aniso::wgsl_source(),
        math::oren_nayar_brdf::wgsl_source(),
        math::coat_darkening::wgsl_source(),
        math::thin_film_modulation::wgsl_source(),
        math::sheen_brdf::wgsl_source(),
        math::transmission_color_to_extinction::wgsl_source(),
        math::subsurface_brdf::wgsl_source(),
        math::srgb_to_linear::wgsl_source(),
    ];
    let mut src = PBR_FRAGMENT_BOILERPLATE.to_string();
    for k in &kernels {
        src.push('\n');
        src.push_str(k);
    }
    src
}
