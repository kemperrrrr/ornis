pub const PBR_VERTEX: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
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
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_index: u32,
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

    var output: VertexOutput;
    output.clip_position = camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = world_normal;
    output.uv = input.uv;
    output.material_index = obj.material_index;
    return output;
}
"#;

pub const PBR_FRAGMENT: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
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

struct MaterialData {
    base_color: vec4<f32>,
    emission_color: vec4<f32>,
    pbr_params: vec4<f32>,
    ior_params: vec4<f32>,
    subsurface_color: vec4<f32>,
    coat_color: vec4<f32>,
    sheen_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> materials: array<MaterialData>;
@group(0) @binding(3) var<uniform> lighting: Lighting;

struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_index: u32,
};

const PI: f32 = 3.14159265;
const EPS: f32 = 1e-6;

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let mat = materials[input.material_index];
    let N = normalize(input.world_normal);
    let V = normalize(camera.camera_pos.xyz - input.world_position);
    let NoV = max(dot(N, V), EPS);

    let base_color = mat.base_color.rgb;
    let emission = mat.emission_color.rgb * mat.emission_color.w;
    let metalness = mat.pbr_params.x;
    let base_roughness = mat.pbr_params.y;
    let specular_weight = mat.pbr_params.z;
    let coat_weight = mat.pbr_params.w;
    let specular_ior = mat.ior_params.x;

    let F0 = mix(fresnel0_from_ior(specular_ior), base_color, metalness);
    let diffuse_color = base_color * (1.0 - metalness);
    let alpha = max(base_roughness * base_roughness, EPS);

    var Lo = vec3<f32>(0.0);

    for (var i = 0u; i < lighting.light_count; i = i + 1u) {
        let L = normalize(lighting.lights[i].direction.xyz);
        let H = normalize(V + L);
        let light_color = lighting.lights[i].color.rgb;
        let intensity = lighting.lights[i].color.w;
        let radiance = light_color * intensity;

        let NoL = max(dot(N, L), EPS);
        let NoH = max(dot(N, H), EPS);
        let HoV = max(dot(H, V), EPS);

        let D = ggx_ndf(NoH, alpha);
        let G = smith_ggx_correlated(NoV, NoL, alpha);
        let F = fresnel_schlick(HoV, F0);

        let spec_brdf = D * G * F / max(4.0 * NoV * NoL, EPS);
        let diff_brdf = oren_nayar(N, V, L, NoV, NoL, alpha) * diffuse_color;

        let kS = F * specular_weight;
        let kD = (1.0 - luminance(kS)) * (1.0 - metalness);

        Lo += (kD * diff_brdf + spec_brdf) * radiance * NoL;
    }

    let ambient = lighting.ambient_color.rgb * mix(diffuse_color, F0, metalness);
    let color = ambient + Lo + emission;

    let tone_mapped = aces_tonemap(color);
    return vec4<f32>(tone_mapped, mat.base_color.a);
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn fresnel0_from_ior(ior: f32) -> vec3<f32> {
    let f = (ior - 1.0) / (ior + 1.0);
    return vec3<f32>(f * f);
}

fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

fn ggx_ndf(NoH: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn smith_ggx_correlated(NoV: f32, NoL: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ggxv = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    let ggxl = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    return 0.5 / max(ggxv + ggxl, EPS);
}

fn oren_nayar(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>,
              NoV: f32, NoL: f32, alpha: f32) -> vec3<f32> {
    let sigma = max(alpha, EPS);
    let sigma2 = sigma * sigma;
    // Energy-preserving Oren-Nayar (EON) — OpenPBR reference:
    // A uses 0.57 instead of 0.33 to guarantee ∫f_r·cosθ dω ≤ ρ
    let A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.57);
    let B = 0.45 * sigma2 / (sigma2 + 0.09);

    let V_proj = normalize(V - N * NoV);
    let L_proj = normalize(L - N * NoL);
    let cos_phi = max(dot(V_proj, L_proj), 0.0);

    let theta_v = acos(NoV);
    let theta_l = acos(NoL);
    let alpha_on = max(theta_v, theta_l);
    let beta = min(theta_v, theta_l);
    let tan_beta = sqrt(1.0 - cos(beta) * cos(beta)) / max(cos(beta), EPS);

    return vec3<f32>(A + B * cos_phi * sin(alpha_on) * tan_beta) / PI;
}
"#;
