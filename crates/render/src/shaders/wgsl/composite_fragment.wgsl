
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
