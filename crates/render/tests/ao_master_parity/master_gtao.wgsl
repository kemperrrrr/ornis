// Hand-written GTAO from master `ao_gtao.rs` / `ao_generated.rs`.
// The horizon texel is `min(u32(uv * dims), dims - 1)`.

struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    previous_view_proj: mat4x4<f32>,
}

struct AoParams {
    radius: f32,
    intensity: f32,
    bias: f32,
    blur_sigma: f32,
}

struct QuadVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(3) var<uniform> ao_params: AoParams;

const CLEAR_DEPTH: f32 = 1.0;
const EPS: f32 = 1e-6;

fn octahedral_decode(p: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    let t = max(-n.z, 0.0);
    let neg = vec2<f32>(0.0 - t, 0.0 - t);
    let pos = vec2<f32>(t, t);
    let offset = select(pos, neg, n.xy >= vec2<f32>(0.0));
    n.x = n.x + offset.x;
    n.y = n.y + offset.y;
    return normalize(n);
}

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view = camera.inv_view_proj * clip;
    return view.xyz / view.w;
}

fn ao_interleaved_gradient(pixel: vec2<u32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(vec2<f32>(pixel), magic.xy)));
}

fn gtao_arc(horizon: f32, normal_angle: f32) -> f32 {
    let h = horizon;
    let n = normal_angle;
    return 0.25 * (-cos(2.0 * h - n) + cos(n) + 2.0 * h * sin(n));
}

fn gtao_horizon_cos(uv: vec2<f32>, dims: vec2<f32>, center: vec3<f32>, view_vec: vec3<f32>, min_cos: f32, radius: f32) -> f32 {
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
        return -1.0;
    }
    let dims_u = vec2<u32>(dims);
    let texel = min(vec2<u32>(uv * dims), dims_u - vec2<u32>(1u));
    let sdepth = textureLoad(depth_tex, texel, 0);
    if (sdepth >= CLEAR_DEPTH) {
        return -1.0;
    }
    let sample_pos = reconstruct_world_pos(uv, sdepth);
    let delta = sample_pos - center;
    let r = length(delta);
    if (r <= EPS || r >= radius) {
        return -1.0;
    }
    let falloff = clamp(1.0 - r / max(radius, EPS), 0.0, 1.0);
    let cos_h = dot(delta / r, view_vec);
    return mix(min_cos, cos_h, falloff);
}

@fragment
fn fs_main(input: QuadVertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(depth_tex));
    let coord = vec2<u32>(input.uv * dims);
    let depth = textureLoad(depth_tex, coord, 0);
    if (depth >= CLEAR_DEPTH) {
        return vec4<f32>(1.0);
    }
    let normal_enc = textureLoad(normal_tex, coord, 0);
    let n = octahedral_decode(normal_enc.rg);
    let center = reconstruct_world_pos(input.uv, depth);
    let to_eye = camera.camera_pos.xyz - center;
    let view_vec = to_eye / max(length(to_eye), EPS);
    let texel_x = vec2<f32>(1.0 / dims.x, 0.0);
    let texel_y = vec2<f32>(0.0, 1.0 / dims.y);
    let screen_x = normalize(reconstruct_world_pos(input.uv + texel_x, depth) - center + vec3<f32>(EPS, 0.0, 0.0));
    let screen_y = normalize(reconstruct_world_pos(input.uv + texel_y, depth) - center + vec3<f32>(0.0, EPS, 0.0));
    let edge = center + screen_x * ao_params.radius;
    let edge_clip = camera.view_proj * vec4<f32>(edge, 1.0);
    let edge_w = max(edge_clip.w, EPS);
    let edge_uv = vec2<f32>(edge_clip.x / edge_w * 0.5 + 0.5, 0.5 - edge_clip.y / edge_w * 0.5);
    let uv_radius = max(length(edge_uv - input.uv), 1.0 / dims.x);
    let slice_base = ao_interleaved_gradient(coord) * 3.14159265359;
    var visibility = 0.0;
    for (var s = 0u; s < 2u; s = s + 1u) {
        let phi = slice_base + f32(s) * 1.57079632679;
        let cos_phi = cos(phi);
        let sin_phi = sin(phi);
        let slice_dir = normalize(screen_x * cos_phi + screen_y * sin_phi);
        let axis = normalize(cross(slice_dir, view_vec) + vec3<f32>(EPS, 0.0, 0.0));
        let projected = n - axis * dot(n, axis);
        let proj_len = max(length(projected), EPS);
        let cos_norm = clamp(dot(projected, view_vec) / proj_len, 0.0, 1.0);
        let normal_angle = sign(dot(slice_dir, projected)) * acos(cos_norm);
        let min_cos_pos = cos(normal_angle + 1.57079632679);
        let min_cos_neg = cos(normal_angle - 1.57079632679);
        var horizon_cos_pos = min_cos_pos;
        var horizon_cos_neg = min_cos_neg;
        let slice_uv = vec2<f32>(cos_phi, sin_phi);
        for (var j = 1u; j <= 4u; j = j + 1u) {
            let off = slice_uv * (uv_radius * f32(j) / 4.0);
            horizon_cos_pos = max(horizon_cos_pos, gtao_horizon_cos(input.uv + off, dims, center, view_vec, min_cos_pos, ao_params.radius));
            horizon_cos_neg = max(horizon_cos_neg, gtao_horizon_cos(input.uv - off, dims, center, view_vec, min_cos_neg, ao_params.radius));
        }
        let h_pos = acos(clamp(horizon_cos_pos, -1.0, 1.0));
        let h_neg = -acos(clamp(horizon_cos_neg, -1.0, 1.0));
        visibility += proj_len * (gtao_arc(h_pos, normal_angle) + gtao_arc(h_neg, normal_angle));
    }
    let ao = clamp(mix(1.0, visibility / 2.0, ao_params.intensity), 0.0, 1.0);
    return vec4<f32>(ao, ao, ao, 1.0);
}
