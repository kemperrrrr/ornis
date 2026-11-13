struct UniformStruct0_ {
    f0_: vec4<f32>,
    f1_: vec4<f32>,
    f2_: vec4<f32>,
    f3_: vec4<f32>,
    f4_: vec4<f32>,
    f5_: vec4<f32>,
    f6_: vec4<f32>,
    f7_: vec4<f32>,
    f8_: vec4<f32>,
    f9_: vec4<f32>,
    f10_: vec4<f32>,
    f11_: vec4<f32>,
    f12_: vec4<f32>,
}

struct UniformStruct1_ {
    f0_: vec4<f32>,
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) o0_: vec2<f32>,
}

@group(0) @binding(2) 
var tex_0_2_: texture_depth_multisampled_2d;
@group(0) @binding(3) 
var<uniform> uniform_struct_0_3_: UniformStruct0_;
@group(0) @binding(4) 
var<uniform> uniform_struct_0_4_: UniformStruct1_;
@group(0) @binding(0) 
var tex_0_0_: texture_2d<f32>;
@group(0) @binding(1) 
var sampler_0_1_: sampler;

@fragment 
fn fs_main(@builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var fragment_out: vec4<f32>;

    let _e5 = textureDimensions(tex_0_2_);
    let _e10 = textureDimensions(tex_0_2_);
    let _e23 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(u32((uv.x * f32(bitcast<i32>(_e5.x))))), bitcast<i32>(u32((uv.y * f32(bitcast<i32>(_e10.y)))))), 0u);
    let _e27 = ((uv.x * 2f) + -1f);
    let _e31 = (1f - (uv.y * 2f));
    let _e35 = uniform_struct_0_3_.f4_.w;
    let _e40 = uniform_struct_0_3_.f5_.w;
    let _e46 = uniform_struct_0_3_.f6_.w;
    let _e52 = uniform_struct_0_3_.f7_.w;
    let _e53 = (_e52 + (((_e27 * _e35) + (_e31 * _e40)) + (_e23 * _e46)));
    let _e57 = uniform_struct_0_3_.f4_.x;
    let _e62 = uniform_struct_0_3_.f5_.x;
    let _e68 = uniform_struct_0_3_.f6_.x;
    let _e74 = uniform_struct_0_3_.f7_.x;
    let _e80 = uniform_struct_0_3_.f4_.y;
    let _e85 = uniform_struct_0_3_.f5_.y;
    let _e91 = uniform_struct_0_3_.f6_.y;
    let _e97 = uniform_struct_0_3_.f7_.y;
    let _e103 = uniform_struct_0_3_.f4_.z;
    let _e108 = uniform_struct_0_3_.f5_.z;
    let _e114 = uniform_struct_0_3_.f6_.z;
    let _e120 = uniform_struct_0_3_.f7_.z;
    let _e126 = uniform_struct_0_3_.f8_.x;
    let _e131 = uniform_struct_0_3_.f8_.y;
    let _e136 = uniform_struct_0_3_.f8_.z;
    let _e143 = uniform_struct_0_4_.f0_.w;
    let _e149 = (1f - exp((0f - (length(vec3<f32>((_e126 - ((_e74 + (((_e27 * _e57) + (_e31 * _e62)) + (_e23 * _e68))) / _e53)), (_e131 - ((_e97 + (((_e27 * _e80) + (_e31 * _e85)) + (_e23 * _e91))) / _e53)), (_e136 - ((_e120 + (((_e27 * _e103) + (_e31 * _e108)) + (_e23 * _e114))) / _e53)))) * _e143))));
    let _e153 = uniform_struct_0_4_.f0_.x;
    let _e157 = uniform_struct_0_4_.f0_.y;
    let _e161 = uniform_struct_0_4_.f0_.z;
    let _e166 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e172 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e178 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e182 = (_e166.x + (_e149 * (_e153 - _e166.x)));
    let _e198 = (_e172.y + (_e149 * (_e157 - _e172.y)));
    let _e214 = (_e178.z + (_e149 * (_e161 - _e178.z)));
    fragment_out = vec4<f32>(((_e182 * ((_e182 * 2.51f) + 0.03f)) / ((_e182 * ((_e182 * 2.43f) + 0.59f)) + 0.14f)), ((_e198 * ((_e198 * 2.51f) + 0.03f)) / ((_e198 * ((_e198 * 2.43f) + 0.59f)) + 0.14f)), ((_e214 * ((_e214 * 2.51f) + 0.03f)) / ((_e214 * ((_e214 * 2.43f) + 0.59f)) + 0.14f)), 1f);
    let _e232 = fragment_out;
    return _e232;
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
