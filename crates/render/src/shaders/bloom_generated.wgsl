struct UniformStruct0_ {
    f0_: f32,
    f1_: f32,
    f2_: u32,
    f3_: f32,
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

    let _e8 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e14 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e20 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e32 = uniform_struct_0_2_.f0_;
    let _e35 = smoothstep(_e32, (_e32 + 0.05f), (((_e8.x * 0.2126f) + (_e14.y * 0.7152f)) + (_e20.z * 0.0722f)));
    fragment_out = vec4<f32>((_e8.x * _e35), (_e14.y * _e35), (_e20.z * _e35), 1f);
    let _e43 = fragment_out;
    return _e43;
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
