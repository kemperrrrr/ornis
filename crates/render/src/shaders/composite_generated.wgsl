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
    var _11_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i: f32;
    var _10_u002e_sroa_u002e_0_u002e_2_u002e_i_u002e_i: f32;

    let _e6 = ((vertex_index & 1u) == 0u);
    if ((vertex_index & 2u) == 0u) {
        if _e6 {
            vertex_out.position = vec4<f32>(-1f, -1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i = 1f;
            _10_u002e_sroa_u002e_0_u002e_2_u002e_i_u002e_i = 0f;
        } else {
            vertex_out.position = vec4<f32>(-1f, 1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i = 0f;
            _10_u002e_sroa_u002e_0_u002e_2_u002e_i_u002e_i = 0f;
        }
    } else {
        if _e6 {
            vertex_out.position = vec4<f32>(1f, -1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i = 1f;
            _10_u002e_sroa_u002e_0_u002e_2_u002e_i_u002e_i = 1f;
        } else {
            vertex_out.position = vec4<f32>(1f, 1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i = 0f;
            _10_u002e_sroa_u002e_0_u002e_2_u002e_i_u002e_i = 1f;
        }
    }
    let _e54 = _11_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i;
    let _e56 = _10_u002e_sroa_u002e_0_u002e_2_u002e_i_u002e_i;
    vertex_out.o0_ = vec2<f32>(_e56, _e54);
    let _e61 = vertex_out;
    return _e61;
}
