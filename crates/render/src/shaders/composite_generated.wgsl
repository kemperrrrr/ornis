struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) o0_: vec2<f32>,
}

@group(0) @binding(2) 
var tex_0_2_: texture_2d<f32>;
@group(0) @binding(3) 
var sampler_0_3_: sampler;
@group(0) @binding(0) 
var tex_0_0_: texture_2d<f32>;
@group(0) @binding(1) 
var sampler_0_1_: sampler;
var<private> const_RNvCslBLCZBlLSfH_19ornis_gpu_composite4QUAD: array<u32, 16> = array<u32, 16>(3212836864u, 3212836864u, 0u, 1065353216u, 3212836864u, 1065353216u, 0u, 1065353216u, 1065353216u, 3212836864u, 0u, 1065353216u, 1065353216u, 1065353216u, 0u, 1065353216u);
var<private> const_RNvCslBLCZBlLSfH_19ornis_gpu_composite3UVS: array<u32, 8> = array<u32, 8>(0u, 1065353216u, 0u, 0u, 1065353216u, 1065353216u, 1065353216u, 0u);

@fragment 
fn fs_main(@builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var fragment_out: vec4<f32>;
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_i11_: f32;
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_i18_: f32;

    let _e8 = textureSampleLevel(tex_0_2_, sampler_0_3_, vec2<f32>(uv.x, uv.y), 0f);
    let _e14 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e20 = textureSampleLevel(tex_0_2_, sampler_0_3_, vec2<f32>(uv.x, uv.y), 0f);
    if !((_e20.x <= 0.04045f)) {
        _0_u002e_sroa_u002e_0_u002e_0_u002e_i18_ = pow(((_e20.x + 0.055f) / 1.055f), 2.4f);
    } else {
        _0_u002e_sroa_u002e_0_u002e_0_u002e_i18_ = (_e20.x / 12.92f);
    }
    let _e36 = _0_u002e_sroa_u002e_0_u002e_0_u002e_i18_;
    let _e42 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e48 = textureSampleLevel(tex_0_2_, sampler_0_3_, vec2<f32>(uv.x, uv.y), 0f);
    if !((_e48.y <= 0.04045f)) {
        _0_u002e_sroa_u002e_0_u002e_0_u002e_i11_ = pow(((_e48.y + 0.055f) / 1.055f), 2.4f);
    } else {
        _0_u002e_sroa_u002e_0_u002e_0_u002e_i11_ = (_e48.y / 12.92f);
    }
    let _e64 = _0_u002e_sroa_u002e_0_u002e_0_u002e_i11_;
    let _e70 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e76 = textureSampleLevel(tex_0_2_, sampler_0_3_, vec2<f32>(uv.x, uv.y), 0f);
    if !((_e76.z <= 0.04045f)) {
        _0_u002e_sroa_u002e_0_u002e_0_u002e_i = pow(((_e76.z + 0.055f) / 1.055f), 2.4f);
    } else {
        _0_u002e_sroa_u002e_0_u002e_0_u002e_i = (_e76.z / 12.92f);
    }
    let _e92 = _0_u002e_sroa_u002e_0_u002e_0_u002e_i;
    fragment_out = vec4<f32>(mix(_e14.x, _e36, _e8.w), mix(_e42.y, _e64, _e8.w), mix(_e70.z, _e92, _e8.w), 1f);
    let _e98 = fragment_out;
    return _e98;
}

@vertex 
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var vertex_out: VertexOut;

    if (vertex_index > 3u) {
        let _e4 = vertex_out;
        return _e4;
    }
    let _e8 = (0u + (vertex_index * 4u));
    let _e11 = const_RNvCslBLCZBlLSfH_19ornis_gpu_composite4QUAD[_e8];
    let _e17 = const_RNvCslBLCZBlLSfH_19ornis_gpu_composite4QUAD[(_e8 + 1u)];
    let _e23 = const_RNvCslBLCZBlLSfH_19ornis_gpu_composite4QUAD[(_e8 + 2u)];
    let _e29 = const_RNvCslBLCZBlLSfH_19ornis_gpu_composite4QUAD[(_e8 + 3u)];
    vertex_out.position = vec4<f32>(bitcast<f32>(_e11), bitcast<f32>(_e17), bitcast<f32>(_e23), bitcast<f32>(_e29));
    let _e37 = (0u + (vertex_index * 2u));
    let _e40 = const_RNvCslBLCZBlLSfH_19ornis_gpu_composite3UVS[_e37];
    let _e46 = const_RNvCslBLCZBlLSfH_19ornis_gpu_composite3UVS[(_e37 + 1u)];
    vertex_out.o0_ = vec2<f32>(bitcast<f32>(_e40), bitcast<f32>(_e46));
    let _e52 = vertex_out;
    return _e52;
}
