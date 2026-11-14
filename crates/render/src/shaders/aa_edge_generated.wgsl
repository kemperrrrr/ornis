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
    let _e12 = textureDimensions(tex_0_0_);
    let _e17 = (1f / f32(bitcast<i32>(_e12.y)));
    let _e22 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e28 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e34 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e36 = (uv.y - _e17);
    let _e41 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, _e36), 0f);
    let _e47 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, _e36), 0f);
    let _e53 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, _e36), 0f);
    let _e55 = (uv.y + _e17);
    let _e60 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, _e55), 0f);
    let _e66 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, _e55), 0f);
    let _e72 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, _e55), 0f);
    let _e74 = (uv.x + _e10);
    let _e79 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(_e74, uv.y), 0f);
    let _e85 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(_e74, uv.y), 0f);
    let _e91 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(_e74, uv.y), 0f);
    let _e93 = (uv.x - _e10);
    let _e98 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(_e93, uv.y), 0f);
    let _e104 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(_e93, uv.y), 0f);
    let _e110 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(_e93, uv.y), 0f);
    let _e119 = (((_e22.x * 0.299f) + (_e28.y * 0.587f)) + (_e34.z * 0.114f));
    let _e152 = ((((_e79.x * 0.299f) + (_e85.y * 0.587f)) + (_e91.z * 0.114f)) - (((_e98.x * 0.299f) + (_e104.y * 0.587f)) + (_e110.z * 0.114f)));
    let _e153 = ((((_e60.x * 0.299f) + (_e66.y * 0.587f)) + (_e72.z * 0.114f)) - (((_e41.x * 0.299f) + (_e47.y * 0.587f)) + (_e53.z * 0.114f)));
    let _e154 = abs(_e152);
    let _e155 = abs(_e153);
    let _e157 = select(_e155, _e154, (_e154 > _e155));
    let _e160 = uniform_struct_0_2_.f1_;
    if !((_e157 <= _e160)) {
        if (abs(_e152) > abs(_e153)) {
            west_b_u002e_sroa_u002e_0_u002e_0_ = _e110.z;
            west_g_u002e_sroa_u002e_0_u002e_0_ = _e104.y;
            west_r_u002e_sroa_u002e_0_u002e_0_ = _e98.x;
            east_b_u002e_sroa_u002e_0_u002e_0_ = _e91.z;
            east_g_u002e_sroa_u002e_0_u002e_0_ = _e85.y;
            east_r_u002e_sroa_u002e_0_u002e_0_ = _e79.x;
        } else {
            west_b_u002e_sroa_u002e_0_u002e_0_ = _e53.z;
            west_g_u002e_sroa_u002e_0_u002e_0_ = _e47.y;
            west_r_u002e_sroa_u002e_0_u002e_0_ = _e41.x;
            east_b_u002e_sroa_u002e_0_u002e_0_ = _e72.z;
            east_g_u002e_sroa_u002e_0_u002e_0_ = _e66.y;
            east_r_u002e_sroa_u002e_0_u002e_0_ = _e60.x;
        }
        let _e179 = west_b_u002e_sroa_u002e_0_u002e_0_;
        let _e181 = west_g_u002e_sroa_u002e_0_u002e_0_;
        let _e183 = west_r_u002e_sroa_u002e_0_u002e_0_;
        let _e185 = east_b_u002e_sroa_u002e_0_u002e_0_;
        let _e187 = east_g_u002e_sroa_u002e_0_u002e_0_;
        let _e189 = east_r_u002e_sroa_u002e_0_u002e_0_;
        let _e207 = uniform_struct_0_2_.f4_;
        if (_e207 == 1u) {
            let _e223 = (1f / (abs((((_e185 * 0.114f) + ((_e187 * 0.587f) + (_e189 * 0.299f))) - _e119)) + 1f));
            let _e237 = (1f / (abs((((_e179 * 0.114f) + ((_e181 * 0.587f) + (_e183 * 0.299f))) - _e119)) + 1f));
            let _e238 = (_e223 + _e237);
            mixed_b_u002e_sroa_u002e_0_u002e_0_ = (((_e185 * _e223) + (_e179 * _e237)) / _e238);
            mixed_g_u002e_sroa_u002e_0_u002e_0_ = (((_e187 * _e223) + (_e181 * _e237)) / _e238);
            mixed_r_u002e_sroa_u002e_0_u002e_0_ = (((_e189 * _e223) + (_e183 * _e237)) / _e238);
        } else {
            mixed_b_u002e_sroa_u002e_0_u002e_0_ = ((_e179 + ((_e34.z + _e34.z) + _e185)) * 0.25f);
            mixed_g_u002e_sroa_u002e_0_u002e_0_ = ((_e181 + ((_e28.y + _e28.y) + _e187)) * 0.25f);
            mixed_r_u002e_sroa_u002e_0_u002e_0_ = ((_e183 + ((_e22.x + _e22.x) + _e189)) * 0.25f);
        }
        let _e258 = mixed_b_u002e_sroa_u002e_0_u002e_0_;
        let _e260 = mixed_g_u002e_sroa_u002e_0_u002e_0_;
        let _e262 = mixed_r_u002e_sroa_u002e_0_u002e_0_;
        let _e267 = clamp(((_e157 - _e160) / _e157), 0f, 1f);
        fragment_out = vec4<f32>(mix(_e22.x, _e262, _e267), mix(_e28.y, _e260, _e267), mix(_e34.z, _e258, _e267), 1f);
    } else {
        fragment_out = vec4<f32>(_e22.x, _e28.y, _e34.z, 1f);
    }
    let _e278 = fragment_out;
    return _e278;
}

@vertex 
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var vertex_out: VertexOut;
    var _11_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var _10_u002e_sroa_u002e_0_u002e_2_u002e_i: f32;

    let _e6 = ((vertex_index & 2u) == 0u);
    if ((vertex_index & 1u) == 0u) {
        if _e6 {
            vertex_out.position = vec4<f32>(-1f, -1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_u002e_i = 1f;
            _10_u002e_sroa_u002e_0_u002e_2_u002e_i = 0f;
        } else {
            vertex_out.position = vec4<f32>(-1f, 1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
            _10_u002e_sroa_u002e_0_u002e_2_u002e_i = 0f;
        }
    } else {
        if _e6 {
            vertex_out.position = vec4<f32>(1f, -1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_u002e_i = 1f;
            _10_u002e_sroa_u002e_0_u002e_2_u002e_i = 1f;
        } else {
            vertex_out.position = vec4<f32>(1f, 1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
            _10_u002e_sroa_u002e_0_u002e_2_u002e_i = 1f;
        }
    }
    let _e54 = _11_u002e_sroa_u002e_0_u002e_0_u002e_i;
    let _e56 = _10_u002e_sroa_u002e_0_u002e_2_u002e_i;
    vertex_out.o0_ = vec2<f32>(_e56, _e54);
    let _e61 = vertex_out;
    return _e61;
}
