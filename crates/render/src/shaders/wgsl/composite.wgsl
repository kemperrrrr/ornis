
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var pbr_tex: texture_2d<f32>;
@group(0) @binding(1) var pbr_sampler: sampler;
@group(0) @binding(2) var ui_tex: texture_2d<f32>;
@group(0) @binding(3) var ui_sampler: sampler;

const QUAD: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(-1.0, -1.0, 0.0, 1.0),
    vec4<f32>(-1.0,  1.0, 0.0, 1.0),
    vec4<f32>( 1.0, -1.0, 0.0, 1.0),
    vec4<f32>( 1.0,  1.0, 0.0, 1.0),
);
const UVS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0),
);

@vertex
fn vs(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    out.position = QUAD[idx];
    out.uv = UVS[idx];
    return out;
}

// Converts an sRGB-encoded channel to linear light. Vello renders into a
// non-srgb Rgba8Unorm target and applies the sRGB OETF itself, so the UI
// texture holds sRGB-encoded bytes. The surface is an *_Srgb format, so the
// hardware re-applies the OETF on write. Without decoding here the UI would be
// sRGB-encoded twice -> visibly washed out / too light. Decoding to linear
// before the mix lets the hardware's single re-encode land on the right value.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let cutoff = c <= vec3<f32>(0.04045);
    let low = c / 12.92;
    let high = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(high, low, cutoff);
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    // pbr_tex is an *_Srgb texture -> the sampler already returns linear light.
    let bg = textureSampleLevel(pbr_tex, pbr_sampler, input.uv, 0.0);
    // ui_tex is Rgba8Unorm holding sRGB-encoded bytes -> decode to linear.
    let ui = textureSampleLevel(ui_tex, ui_sampler, input.uv, 0.0);
    let ui_linear = srgb_to_linear(ui.rgb);
    return vec4<f32>(mix(bg.rgb, ui_linear, ui.a), 1.0);
}
