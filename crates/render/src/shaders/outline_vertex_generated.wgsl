struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) o0_: vec3<f32>,
}

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

@group(0) @binding(2) 
var<storage> storage_0_2_: array<u32>;
@group(0) @binding(1) 
var<storage> storage_0_1_: array<u32>;
@group(0) @binding(0) 
var<uniform> uniform_struct_0_0_: UniformStruct0_;
@group(0) @binding(3) 
var<storage> storage_0_3_: array<u32>;

@vertex 
fn vs_main(@builtin(instance_index) instance_index: u32, @location(0) in0_: vec3<f32>, @location(1) in1_: vec3<f32>) -> VertexOut {
    var vertex_out: VertexOut;

    let _e5 = storage_0_2_[instance_index];
    let _e7 = (_e5 * 52u);
    let _e25 = storage_0_1_[_e7];
    let _e31 = storage_0_1_[(_e7 + 4u)];
    let _e37 = storage_0_1_[(_e7 + 8u)];
    let _e43 = storage_0_1_[(_e7 + 12u)];
    let _e50 = (((((in0_.x + (in1_.x * 0.05f)) * bitcast<f32>(_e25)) + ((in0_.y + (in1_.y * 0.05f)) * bitcast<f32>(_e31))) + ((in0_.z + (in1_.z * 0.05f)) * bitcast<f32>(_e37))) + bitcast<f32>(_e43));
    let _e70 = storage_0_1_[(_e7 | 1u)];
    let _e76 = storage_0_1_[(_e7 + 5u)];
    let _e82 = storage_0_1_[(_e7 + 9u)];
    let _e88 = storage_0_1_[(_e7 + 13u)];
    let _e95 = (((((in0_.x + (in1_.x * 0.05f)) * bitcast<f32>(_e70)) + ((in0_.y + (in1_.y * 0.05f)) * bitcast<f32>(_e76))) + ((in0_.z + (in1_.z * 0.05f)) * bitcast<f32>(_e82))) + bitcast<f32>(_e88));
    let _e115 = storage_0_1_[(_e7 | 2u)];
    let _e121 = storage_0_1_[(_e7 + 6u)];
    let _e127 = storage_0_1_[(_e7 + 10u)];
    let _e133 = storage_0_1_[(_e7 + 14u)];
    let _e140 = (((((in0_.x + (in1_.x * 0.05f)) * bitcast<f32>(_e115)) + ((in0_.y + (in1_.y * 0.05f)) * bitcast<f32>(_e121))) + ((in0_.z + (in1_.z * 0.05f)) * bitcast<f32>(_e127))) + bitcast<f32>(_e133));
    let _e144 = uniform_struct_0_0_.f0_.x;
    let _e149 = uniform_struct_0_0_.f1_.x;
    let _e154 = uniform_struct_0_0_.f2_.x;
    let _e159 = uniform_struct_0_0_.f3_.x;
    let _e166 = uniform_struct_0_0_.f0_.y;
    let _e171 = uniform_struct_0_0_.f1_.y;
    let _e176 = uniform_struct_0_0_.f2_.y;
    let _e181 = uniform_struct_0_0_.f3_.y;
    let _e188 = uniform_struct_0_0_.f0_.z;
    let _e193 = uniform_struct_0_0_.f1_.z;
    let _e198 = uniform_struct_0_0_.f2_.z;
    let _e203 = uniform_struct_0_0_.f3_.z;
    let _e210 = uniform_struct_0_0_.f0_.w;
    let _e215 = uniform_struct_0_0_.f1_.w;
    let _e220 = uniform_struct_0_0_.f2_.w;
    let _e225 = uniform_struct_0_0_.f3_.w;
    vertex_out.position = vec4<f32>((_e159 + (((_e50 * _e144) + (_e95 * _e149)) + (_e154 * _e140))), (_e181 + (((_e50 * _e166) + (_e95 * _e171)) + (_e140 * _e176))), (_e203 + (((_e50 * _e188) + (_e95 * _e193)) + (_e140 * _e198))), (_e225 + (((_e50 * _e210) + (_e95 * _e215)) + (_e140 * _e220))));
    let _e234 = storage_0_3_[instance_index];
    let _e236 = (_e234 == 2u);
    vertex_out.o0_ = vec3<f32>(select(0.55f, 0.16f, _e236), select(0.75f, 0.55f, _e236), 1f);
    let _e248 = vertex_out;
    return _e248;
}
