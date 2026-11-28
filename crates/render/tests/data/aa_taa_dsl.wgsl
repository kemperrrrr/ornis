
struct AaParams {
    feedback: f32,
    threshold: f32,
    clamp_radius: f32,
    history_valid: f32,
    mode: u32,
};
struct TaaOutput {
    @location(0) color: vec4<f32>,
    @location(1) history: vec4<f32>,
}
struct QuadVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var motion_tex: texture_2d<f32>;
@group(0) @binding(4) var previous_depth_tex: texture_2d<u32>;
@group(0) @binding(5) var history_tex: texture_2d<f32>;
@group(0) @binding(6) var<uniform> aa_params: AaParams;
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
fn aa_ycocg_encode(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b,
    );
}

fn aa_ycocg_decode(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> QuadVertexOutput {
    return QuadVertexOutput(quad[vertex_index], uvs[vertex_index]);
}
@fragment
fn fs_main(input: QuadVertexOutput) -> TaaOutput {
    let uv = input.uv;
    let dims_u = textureDimensions(src_tex);
    let dims = vec2<f32>(dims_u);
    let coord_u = min(vec2<u32>(uv * dims), dims_u - vec2<u32>(1u));
    let coord = vec2<i32>(coord_u);
    let current = textureSampleLevel(src_tex, src_sampler, uv, 0.0).rgb;
    let depth = textureLoad(depth_tex, coord, 0);
    if (depth >= CLEAR_DEPTH) {
        return TaaOutput(vec4<f32>(current, 1.0), vec4<f32>(current, 1.0));
    }
    // The g-buffer stores velocity in RG16F and packs previous NDC depth
    // plus validity into an R16Uint value. 65535 denotes invalid motion.
    let motion = textureLoad(motion_tex, coord, 0).rg;
    let packed_previous_depth = textureLoad(previous_depth_tex, coord, 0).r;
    if (packed_previous_depth == 65535u) {
        return TaaOutput(vec4<f32>(current, 1.0), vec4<f32>(current, depth));
    }
    let previous_depth = f32(packed_previous_depth) / 65534.0;
    let prev_uv = uv - motion;
    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0) {
        return TaaOutput(vec4<f32>(current, 1.0), vec4<f32>(current, depth));
    }
    // The colour range of the current neighbourhood: the history is
    // clamped into it, so a stale sample cannot drag its colour along.
    let radius = i32(aa_params.clamp_radius);
    var lower = vec3<f32>(1.0e30);
    var upper = vec3<f32>(-1.0e30);
    var sum = vec3<f32>(0.0);
    var sum_squares = vec3<f32>(0.0);
    var count = 0.0;
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let tap_uv = clamp(uv + vec2<f32>(f32(dx), f32(dy)) / dims, vec2<f32>(0.0), vec2<f32>(1.0));
            let tap = aa_ycocg_encode(textureSampleLevel(src_tex, src_sampler, tap_uv, 0.0).rgb);
            lower = min(lower, tap);
            upper = max(upper, tap);
            sum = sum + tap;
            sum_squares = sum_squares + tap * tap;
            count = count + 1.0;
        }
    }
    let mean = sum / count;
    let sigma = sqrt(max(sum_squares / count - mean * mean, vec3<f32>(0.0)));
    let bounds_low = max(lower, mean - 1.25 * sigma);
    let bounds_high = min(upper, mean + 1.25 * sigma);
    var result = current;
    if (aa_params.history_valid > 0.5) {
        let history = textureSampleLevel(history_tex, src_sampler, prev_uv, 0.0);
        // A mismatch means the previous surface was occluded or this is a
        // disocclusion: rejecting it keeps moving objects from leaving
        // long history trails.
        let depth_gap = abs(history.a - previous_depth);
        if (depth_gap < 0.002) {
            let clamped = aa_ycocg_decode(
                clamp(aa_ycocg_encode(history.rgb), bounds_low, bounds_high)
            );
            result = mix(current, clamped, aa_params.feedback);
        }
    }
    return TaaOutput(vec4<f32>(result, 1.0), vec4<f32>(result, depth));
}
