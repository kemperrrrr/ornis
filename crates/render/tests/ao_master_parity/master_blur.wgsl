// Hand-written bilateral blur from master `ao_generated.rs`.
// Depth taps clamp with `min(vec2<u32>(uv * dims), dims - 1)`.

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

@group(0) @binding(0) var ao_raw_tex: texture_2d<f32>;
@group(0) @binding(1) var blur_sampler: sampler;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var<uniform> ao_params: AoParams;
@group(0) @binding(4) var<uniform> camera: Camera;

const EPS: f32 = 1e-6;

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view = camera.inv_view_proj * clip;
    return view.xyz / view.w;
}

fn ao_linear_depth(uv: vec2<f32>, depth: f32) -> f32 {
    let world = reconstruct_world_pos(uv, depth);
    return length(camera.camera_pos.xyz - world);
}

@fragment
fn fs_main(input: QuadVertexOutput) -> @location(0) vec4<f32> {
    let raw = textureSampleLevel(ao_raw_tex, blur_sampler, input.uv, 0.0).r;
    let dims_u = textureDimensions(depth_tex);
    let dims = vec2<f32>(dims_u);
    let center_coord = min(vec2<u32>(input.uv * dims), dims_u - vec2<u32>(1u));
    let center_z = ao_linear_depth(input.uv, textureLoad(depth_tex, center_coord, 0));
    let texel = 1.0 / vec2<f32>(textureDimensions(ao_raw_tex));
    let sigma = max(ao_params.blur_sigma, EPS);
    var acc = raw;
    var weight = 1.0;
    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(1.5, 0.0),
        vec2<f32>(-1.5, 0.0),
        vec2<f32>(0.0, 1.5),
        vec2<f32>(0.0, -1.5),
    );
    for (var i = 0u; i < 4u; i = i + 1u) {
        let off = offsets[i] * texel;
        let sample_uv = input.uv + off;
        let clamped = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
        let sample_coord = min(vec2<u32>(clamped * dims), dims_u - vec2<u32>(1u));
        let sample_z = ao_linear_depth(clamped, textureLoad(depth_tex, sample_coord, 0));
        let delta = (center_z - sample_z) / sigma;
        let w = exp(-delta * delta);
        acc += textureSampleLevel(ao_raw_tex, blur_sampler, sample_uv, 0.0).r * w;
        weight += w;
    }
    let ao = acc / weight;
    return vec4<f32>(ao, ao, ao, 1.0);
}
