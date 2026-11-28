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
    var _0_u002e_sroa_u002e_0_u002e_0_: f32;
    var _0_u002e_sroa_u002e_0_u002e_0_1: f32;
    var _0_u002e_sroa_u002e_0_u002e_0_2: f32;

    let _e4 = vec2<f32>(uv.x, uv.y);
    let _e8 = textureSampleLevel(tex_0_2_, sampler_0_3_, _e4, 0f);
    let _e13 = textureSampleLevel(tex_0_0_, sampler_0_1_, _e4, 0f);
    if !((_e8.x <= 0.04045f)) {
        _0_u002e_sroa_u002e_0_u002e_0_2 = pow(((_e8.x + 0.055f) / 1.055f), 2.4f);
    } else {
        _0_u002e_sroa_u002e_0_u002e_0_2 = (_e8.x / 12.92f);
    }
    let _e31 = _0_u002e_sroa_u002e_0_u002e_0_2;
    let _e33 = vec2<f32>(uv.x, uv.y);
    let _e37 = textureSampleLevel(tex_0_0_, sampler_0_1_, _e33, 0f);
    let _e42 = textureSampleLevel(tex_0_2_, sampler_0_3_, _e33, 0f);
    if !((_e42.y <= 0.04045f)) {
        _0_u002e_sroa_u002e_0_u002e_0_1 = pow(((_e42.y + 0.055f) / 1.055f), 2.4f);
    } else {
        _0_u002e_sroa_u002e_0_u002e_0_1 = (_e42.y / 12.92f);
    }
    let _e58 = _0_u002e_sroa_u002e_0_u002e_0_1;
    let _e60 = vec2<f32>(uv.x, uv.y);
    let _e64 = textureSampleLevel(tex_0_0_, sampler_0_1_, _e60, 0f);
    let _e69 = textureSampleLevel(tex_0_2_, sampler_0_3_, _e60, 0f);
    if !((_e69.z <= 0.04045f)) {
        _0_u002e_sroa_u002e_0_u002e_0_ = pow(((_e69.z + 0.055f) / 1.055f), 2.4f);
    } else {
        _0_u002e_sroa_u002e_0_u002e_0_ = (_e69.z / 12.92f);
    }
    let _e85 = _0_u002e_sroa_u002e_0_u002e_0_;
    fragment_out = vec4<f32>(mix(_e13.x, _e31, _e8.w), mix(_e37.y, _e58, _e8.w), mix(_e64.z, _e85, _e8.w), 1f);
    let _e91 = fragment_out;
    return _e91;
}

@vertex 
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var vertex_out: VertexOut;
    var _11_u002e_sroa_u002e_0_u002e_0_: f32;
    var _10_u002e_sroa_u002e_0_u002e_2_: f32;

    let _e6 = ((vertex_index & 1u) == 0u);
    if ((vertex_index & 2u) == 0u) {
        if _e6 {
            vertex_out.position = vec4<f32>(-1f, -1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_ = 1f;
            _10_u002e_sroa_u002e_0_u002e_2_ = 0f;
        } else {
            vertex_out.position = vec4<f32>(-1f, 1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_ = 0f;
            _10_u002e_sroa_u002e_0_u002e_2_ = 0f;
        }
    } else {
        if _e6 {
            vertex_out.position = vec4<f32>(1f, -1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_ = 1f;
            _10_u002e_sroa_u002e_0_u002e_2_ = 1f;
        } else {
            vertex_out.position = vec4<f32>(1f, 1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_ = 0f;
            _10_u002e_sroa_u002e_0_u002e_2_ = 1f;
        }
    }
    let _e54 = _11_u002e_sroa_u002e_0_u002e_0_;
    let _e56 = _10_u002e_sroa_u002e_0_u002e_2_;
    vertex_out.o0_ = vec2<f32>(_e56, _e54);
    let _e61 = vertex_out;
    return _e61;
}
