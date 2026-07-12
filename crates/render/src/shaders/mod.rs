pub mod math;

/// Boilerplate for COMPOSITE_FRAGMENT: structs, bindings, constants, entry points.
const COMPOSITE_BOILERPLATE: &str = r#"
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

pub fn composite_fragment() -> String {
    format!(
        "{}\n{}",
        COMPOSITE_BOILERPLATE,
        math::aces_tonemap::wgsl_source(),
    )
}
