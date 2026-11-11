// Hand-written SSAO from master `ao_generated.rs`.
// The tap index is `vec2<u32>(suv * dims)` with no min: `suv == 1` is out of bounds.

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
    let t = normalize(cross(n, vec3<f32>(0.0, 1.0, 0.0)) + vec3<f32>(0.0, 0.0, EPS));
    let b = cross(n, t);
    let angle = ao_interleaved_gradient(coord) * 6.28318530718;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    var occluded = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let fi = f32(i) + 0.5;
        let tap_r = fi / 8.0;
        let tap_a = fi * 2.399963;
        var tap = vec2<f32>(cos(tap_a), sin(tap_a)) * tap_r;
        tap = vec2<f32>(tap.x * cos_a - tap.y * sin_a, tap.x * sin_a + tap.y * cos_a);
        let dir = normalize(t * tap.x + b * tap.y + n * 0.75);
        let probe = center + dir * ao_params.radius;
        let clip = camera.view_proj * vec4<f32>(probe, 1.0);
        let ndc = clip.xyz / max(clip.w, EPS);
        if (ndc.z > 1.0) {
            continue;
        }
        let suv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) {
            continue;
        }
        let sdepth = textureLoad(depth_tex, vec2<u32>(suv * dims), 0);
        if (sdepth >= CLEAR_DEPTH) {
            continue;
        }
        let sample_pos = reconstruct_world_pos(suv, sdepth);
        let range = length(probe - sample_pos);
        if (range < EPS) {
            continue;
        }
        let eye_probe = length(camera.camera_pos.xyz - probe);
        let eye_sample = length(camera.camera_pos.xyz - sample_pos);
        if (eye_sample < eye_probe - ao_params.bias) {
            occluded += max(0.0, 1.0 - range / ao_params.radius);
        }
    }
    let ao = clamp(1.0 - (occluded / 8.0) * ao_params.intensity, 0.0, 1.0);
    return vec4<f32>(ao, ao, ao, 1.0);
}
