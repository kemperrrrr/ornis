
struct AaParams {
    feedback: f32,
    threshold: f32,
    clamp_radius: f32,
    history_valid: f32,
    mode: u32,
};
struct QuadVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> aa_params: AaParams;
const quad: array<vec4<f32>, 4> = array<vec4<f32>, 4>(vec4<f32>(-1f, -1f, 0f, 1f), vec4<f32>(1f, -1f, 0f, 1f), vec4<f32>(-1f, 1f, 0f, 1f), vec4<f32>(1f, 1f, 0f, 1f));
const uvs: array<vec2<f32>, 4> = array<vec2<f32>, 4>(vec2<f32>(0f, 1f), vec2<f32>(1f, 1f), vec2<f32>(0f, 0f), vec2<f32>(1f, 0f));
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
fn aa_luma(c: vec3<f32>) -> f32 {
    return 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
}
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> QuadVertexOutput {
    return QuadVertexOutput(quad[vertex_index], uvs[vertex_index]);
}
@fragment
fn fs_main(input: QuadVertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let dims = vec2<f32>(textureDimensions(src_tex));
    let texel = 1.0 / dims;
    let center = textureSampleLevel(src_tex, src_sampler, uv, 0.0).rgb;
    let luma_center = aa_luma(center);
    let north = textureSampleLevel(src_tex, src_sampler, uv - vec2<f32>(0.0, texel.y), 0.0).rgb;
    let south = textureSampleLevel(src_tex, src_sampler, uv + vec2<f32>(0.0, texel.y), 0.0).rgb;
    let east = textureSampleLevel(src_tex, src_sampler, uv + vec2<f32>(texel.x, 0.0), 0.0).rgb;
    let west = textureSampleLevel(src_tex, src_sampler, uv - vec2<f32>(texel.x, 0.0), 0.0).rgb;
    let gradient_x = aa_luma(east) - aa_luma(west);
    let gradient_y = aa_luma(south) - aa_luma(north);
    let range = max(abs(gradient_x), abs(gradient_y));
    if (range <= aa_params.threshold) {
        return vec4<f32>(center, 1.0);
    }
    // The edge runs across the dominant gradient: blend along it.
    var tap_a = center;
    var tap_b = center;
    if (abs(gradient_x) > abs(gradient_y)) {
        tap_a = east;
        tap_b = west;
    } else {
        tap_a = south;
        tap_b = north;
    }
    var blended = (center + center + tap_a + tap_b) * 0.25;
    if (aa_params.mode == 1u) {
        // SMAA: weight the two taps by their luma similarity to the
        // centre, so the sample on the centre's side keeps the blend and
        // the other one fades out.
        let weight_a = 1.0 / (1.0 + abs(aa_luma(tap_a) - luma_center));
        let weight_b = 1.0 / (1.0 + abs(aa_luma(tap_b) - luma_center));
        blended = (tap_a * weight_a + tap_b * weight_b) / (weight_a + weight_b);
    }
    let weight = clamp((range - aa_params.threshold) / range, 0.0, 1.0);
    return vec4<f32>(mix(center, blended, weight), 1.0);
}
