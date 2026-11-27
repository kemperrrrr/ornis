struct UniformStruct0_ {
    f0_: f32,
    f1_: f32,
    f2_: u32,
    f3_: f32,
}

struct UniformStruct1_ {
    f0_: f32,
    f1_: f32,
    f2_: f32,
    f3_: f32,
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) o0_: vec2<f32>,
}

@group(0) @binding(4) 
var<uniform> uniform_struct_0_4_: UniformStruct0_;
@group(0) @binding(1) 
var tex_0_1_: texture_2d<f32>;
@group(0) @binding(2) 
var sampler_0_2_: sampler;
@group(0) @binding(5) 
var<uniform> uniform_struct_0_5_: UniformStruct1_;
@group(0) @binding(0) 
var tex_0_0_: texture_2d<f32>;
@group(0) @binding(3) 
var tex_0_3_: texture_2d<f32>;

@fragment 
fn fs_main(@builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var fragment_out: vec4<f32>;
    var forward_weight_u002e_sroa_u002e_0_u002e_0_: f32;
    var deferred_weight_u002e_sroa_u002e_0_u002e_0_: f32;

    let _e6 = uniform_struct_0_4_.f2_;
    switch _e6 {
        case 1u: {
            forward_weight_u002e_sroa_u002e_0_u002e_0_ = 1f;
            deferred_weight_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
        case 2u: {
            forward_weight_u002e_sroa_u002e_0_u002e_0_ = 1f;
            deferred_weight_u002e_sroa_u002e_0_u002e_0_ = 1f;
        }
        default: {
            forward_weight_u002e_sroa_u002e_0_u002e_0_ = 0f;
            deferred_weight_u002e_sroa_u002e_0_u002e_0_ = 1f;
        }
    }
    let _e20 = forward_weight_u002e_sroa_u002e_0_u002e_0_;
    let _e22 = deferred_weight_u002e_sroa_u002e_0_u002e_0_;
    let _e23 = vec2<f32>(uv.x, uv.y);
    let _e27 = textureSampleLevel(tex_0_1_, sampler_0_2_, _e23, 0f);
    let _e31 = uniform_struct_0_4_.f1_;
    let _e34 = uniform_struct_0_5_.f0_;
    let _e37 = textureSampleLevel(tex_0_0_, sampler_0_2_, _e23, 0f);
    let _e63 = textureSampleLevel(tex_0_3_, sampler_0_2_, _e23, 0f);
    let _e67 = (_e34 * (((_e22 * _e37.x) + (_e20 * (_e27.w * _e27.x))) + (_e31 * _e63.x)));
    let _e85 = (_e34 * (((_e22 * _e37.y) + (_e20 * (_e27.w * _e27.y))) + (_e31 * _e63.y)));
    let _e103 = (_e34 * (((_e22 * _e37.z) + (_e20 * (_e27.w * _e27.z))) + (_e31 * _e63.z)));
    fragment_out = vec4<f32>(((_e67 * ((_e67 * 2.51f) + 0.03f)) / ((_e67 * ((_e67 * 2.43f) + 0.59f)) + 0.14f)), ((_e85 * ((_e85 * 2.51f) + 0.03f)) / ((_e85 * ((_e85 * 2.43f) + 0.59f)) + 0.14f)), ((_e103 * ((_e103 * 2.51f) + 0.03f)) / ((_e103 * ((_e103 * 2.43f) + 0.59f)) + 0.14f)), 1f);
    let _e121 = fragment_out;
    return _e121;
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
