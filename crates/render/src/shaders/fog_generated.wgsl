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
var tex_0_2_: texture_depth_2d;
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
    let _e20 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(u32((uv.x * f32(bitcast<i32>(_e5.x))))), bitcast<i32>(u32((uv.y * f32(bitcast<i32>(_e5.y)))))), 0i);
    let _e24 = ((uv.x * 2f) + -1f);
    let _e28 = (1f - (uv.y * 2f));
    let _e32 = uniform_struct_0_3_.f4_.w;
    let _e37 = uniform_struct_0_3_.f5_.w;
    let _e43 = uniform_struct_0_3_.f6_.w;
    let _e49 = uniform_struct_0_3_.f7_.w;
    let _e50 = (_e49 + (((_e24 * _e32) + (_e28 * _e37)) + (_e20 * _e43)));
    let _e54 = uniform_struct_0_3_.f4_.x;
    let _e59 = uniform_struct_0_3_.f5_.x;
    let _e65 = uniform_struct_0_3_.f6_.x;
    let _e71 = uniform_struct_0_3_.f7_.x;
    let _e77 = uniform_struct_0_3_.f4_.y;
    let _e82 = uniform_struct_0_3_.f5_.y;
    let _e88 = uniform_struct_0_3_.f6_.y;
    let _e94 = uniform_struct_0_3_.f7_.y;
    let _e100 = uniform_struct_0_3_.f4_.z;
    let _e105 = uniform_struct_0_3_.f5_.z;
    let _e111 = uniform_struct_0_3_.f6_.z;
    let _e117 = uniform_struct_0_3_.f7_.z;
    let _e123 = uniform_struct_0_3_.f8_.x;
    let _e128 = uniform_struct_0_3_.f8_.y;
    let _e133 = uniform_struct_0_3_.f8_.z;
    let _e140 = uniform_struct_0_4_.f0_.w;
    let _e146 = (1f - exp((0f - (length(vec3<f32>((_e123 - ((_e71 + (((_e24 * _e54) + (_e28 * _e59)) + (_e20 * _e65))) / _e50)), (_e128 - ((_e94 + (((_e24 * _e77) + (_e28 * _e82)) + (_e20 * _e88))) / _e50)), (_e133 - ((_e117 + (((_e24 * _e100) + (_e28 * _e105)) + (_e20 * _e111))) / _e50)))) * _e140))));
    let _e150 = uniform_struct_0_4_.f0_.x;
    let _e154 = uniform_struct_0_4_.f0_.y;
    let _e158 = uniform_struct_0_4_.f0_.z;
    let _e163 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e171 = (_e163.x + (_e146 * (_e150 - _e163.x)));
    let _e187 = (_e163.y + (_e146 * (_e154 - _e163.y)));
    let _e203 = (_e163.z + (_e146 * (_e158 - _e163.z)));
    fragment_out = vec4<f32>(((_e171 * ((_e171 * 2.51f) + 0.03f)) / ((_e171 * ((_e171 * 2.43f) + 0.59f)) + 0.14f)), ((_e187 * ((_e187 * 2.51f) + 0.03f)) / ((_e187 * ((_e187 * 2.43f) + 0.59f)) + 0.14f)), ((_e203 * ((_e203 * 2.51f) + 0.03f)) / ((_e203 * ((_e203 * 2.43f) + 0.59f)) + 0.14f)), 1f);
    let _e221 = fragment_out;
    return _e221;
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
