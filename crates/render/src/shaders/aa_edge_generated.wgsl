struct UniformStruct0_ {
    f0_: f32,
    f1_: f32,
    f2_: f32,
    f3_: f32,
    f4_: u32,
    f5_: f32,
    f6_: f32,
    f7_: f32,
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) o0_: vec2<f32>,
}

@group(0) @binding(0) 
var tex_0_0_: texture_2d<f32>;
@group(0) @binding(1) 
var sampler_0_1_: sampler;
@group(0) @binding(2) 
var<uniform> uniform_struct_0_2_: UniformStruct0_;

@fragment 
fn fs_main(@builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var fragment_out: vec4<f32>;
    var west_b_u002e_sroa_u002e_0_u002e_0_: f32;
    var west_g_u002e_sroa_u002e_0_u002e_0_: f32;
    var west_r_u002e_sroa_u002e_0_u002e_0_: f32;
    var east_b_u002e_sroa_u002e_0_u002e_0_: f32;
    var east_g_u002e_sroa_u002e_0_u002e_0_: f32;
    var east_r_u002e_sroa_u002e_0_u002e_0_: f32;
    var mixed_b_u002e_sroa_u002e_0_u002e_0_: f32;
    var mixed_g_u002e_sroa_u002e_0_u002e_0_: f32;
    var mixed_r_u002e_sroa_u002e_0_u002e_0_: f32;

    let _e5 = textureDimensions(tex_0_0_);
    let _e10 = (1f / f32(bitcast<i32>(_e5.x)));
    let _e15 = (1f / f32(bitcast<i32>(_e5.y)));
    let _e19 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e28 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, (uv.y - _e15)), 0f);
    let _e37 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, (uv.y + _e15)), 0f);
    let _e46 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>((uv.x + _e10), uv.y), 0f);
    let _e55 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>((uv.x - _e10), uv.y), 0f);
    let _e68 = (((_e19.x * 0.299f) + (_e19.y * 0.587f)) + (_e19.z * 0.114f));
    let _e101 = ((((_e46.x * 0.299f) + (_e46.y * 0.587f)) + (_e46.z * 0.114f)) - (((_e55.x * 0.299f) + (_e55.y * 0.587f)) + (_e55.z * 0.114f)));
    let _e102 = ((((_e37.x * 0.299f) + (_e37.y * 0.587f)) + (_e37.z * 0.114f)) - (((_e28.x * 0.299f) + (_e28.y * 0.587f)) + (_e28.z * 0.114f)));
    let _e103 = abs(_e101);
    let _e104 = abs(_e102);
    let _e106 = select(_e104, _e103, (_e103 > _e104));
    let _e109 = uniform_struct_0_2_.f1_;
    if !((_e106 <= _e109)) {
        if (abs(_e101) > abs(_e102)) {
            west_b_u002e_sroa_u002e_0_u002e_0_ = _e55.z;
            west_g_u002e_sroa_u002e_0_u002e_0_ = _e55.y;
            west_r_u002e_sroa_u002e_0_u002e_0_ = _e55.x;
            east_b_u002e_sroa_u002e_0_u002e_0_ = _e46.z;
            east_g_u002e_sroa_u002e_0_u002e_0_ = _e46.y;
            east_r_u002e_sroa_u002e_0_u002e_0_ = _e46.x;
        } else {
            west_b_u002e_sroa_u002e_0_u002e_0_ = _e28.z;
            west_g_u002e_sroa_u002e_0_u002e_0_ = _e28.y;
            west_r_u002e_sroa_u002e_0_u002e_0_ = _e28.x;
            east_b_u002e_sroa_u002e_0_u002e_0_ = _e37.z;
            east_g_u002e_sroa_u002e_0_u002e_0_ = _e37.y;
            east_r_u002e_sroa_u002e_0_u002e_0_ = _e37.x;
        }
        let _e128 = west_b_u002e_sroa_u002e_0_u002e_0_;
        let _e130 = west_g_u002e_sroa_u002e_0_u002e_0_;
        let _e132 = west_r_u002e_sroa_u002e_0_u002e_0_;
        let _e134 = east_b_u002e_sroa_u002e_0_u002e_0_;
        let _e136 = east_g_u002e_sroa_u002e_0_u002e_0_;
        let _e138 = east_r_u002e_sroa_u002e_0_u002e_0_;
        let _e156 = uniform_struct_0_2_.f4_;
        if (_e156 == 1u) {
            let _e172 = (1f / (abs((((_e134 * 0.114f) + ((_e136 * 0.587f) + (_e138 * 0.299f))) - _e68)) + 1f));
            let _e186 = (1f / (abs((((_e128 * 0.114f) + ((_e130 * 0.587f) + (_e132 * 0.299f))) - _e68)) + 1f));
            let _e187 = (_e172 + _e186);
            mixed_b_u002e_sroa_u002e_0_u002e_0_ = (((_e134 * _e172) + (_e128 * _e186)) / _e187);
            mixed_g_u002e_sroa_u002e_0_u002e_0_ = (((_e136 * _e172) + (_e130 * _e186)) / _e187);
            mixed_r_u002e_sroa_u002e_0_u002e_0_ = (((_e138 * _e172) + (_e132 * _e186)) / _e187);
        } else {
            mixed_b_u002e_sroa_u002e_0_u002e_0_ = ((_e128 + ((_e19.z + _e19.z) + _e134)) * 0.25f);
            mixed_g_u002e_sroa_u002e_0_u002e_0_ = ((_e130 + ((_e19.y + _e19.y) + _e136)) * 0.25f);
            mixed_r_u002e_sroa_u002e_0_u002e_0_ = ((_e132 + ((_e19.x + _e19.x) + _e138)) * 0.25f);
        }
        let _e207 = mixed_b_u002e_sroa_u002e_0_u002e_0_;
        let _e209 = mixed_g_u002e_sroa_u002e_0_u002e_0_;
        let _e211 = mixed_r_u002e_sroa_u002e_0_u002e_0_;
        let _e216 = clamp(((_e106 - _e109) / _e106), 0f, 1f);
        fragment_out = vec4<f32>(mix(_e19.x, _e211, _e216), mix(_e19.y, _e209, _e216), mix(_e19.z, _e207, _e216), 1f);
    } else {
        fragment_out = vec4<f32>(_e19.x, _e19.y, _e19.z, 1f);
    }
    let _e227 = fragment_out;
    return _e227;
}

@vertex 
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var vertex_out: VertexOut;
    var _11_u002e_sroa_u002e_0_u002e_0_: f32;
    var _10_u002e_sroa_u002e_0_u002e_2_: f32;

    let _e6 = ((vertex_index & 2u) == 0u);
    if ((vertex_index & 1u) == 0u) {
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
