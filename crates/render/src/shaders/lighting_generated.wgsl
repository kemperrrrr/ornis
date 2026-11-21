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

struct UniformStruct5_ {
    f0_: u32,
    f1_: u32,
    f2_: u32,
    f3_: u32,
}

struct UniformStruct1_ {
    f0_: vec4<f32>,
    f1_: vec4<f32>,
    f2_: vec4<f32>,
    f3_: vec4<f32>,
    f4_: vec4<f32>,
    f5_: vec4<f32>,
    f6_: vec4<f32>,
    f7_: vec4<f32>,
    f8_: vec4<f32>,
}

struct UniformStruct2_ {
    f0_: vec4<f32>,
    f1_: vec4<f32>,
    f2_: vec4<f32>,
    f3_: vec4<f32>,
}

struct UniformStruct3_ {
    f0_: array<UniformStruct2_, 4>,
    f1_: vec4<f32>,
    f2_: vec4<f32>,
    f3_: f32,
}

struct UniformStruct4_ {
    f0_: vec4<f32>,
    f1_: array<UniformStruct1_, 8>,
    f2_: u32,
    f3_: f32,
    f4_: f32,
    f5_: u32,
    f6_: UniformStruct3_,
}

@group(0) @binding(8) 
var tex_0_8_: texture_depth_2d;
@group(0) @binding(3) 
var tex_0_3_: texture_2d<f32>;
@group(0) @binding(9) 
var sampler_0_9_: sampler;
@group(0) @binding(4) 
var tex_0_4_: texture_2d<f32>;
@group(0) @binding(0) 
var<uniform> uniform_struct_0_0_: UniformStruct0_;
@group(0) @binding(5) 
var tex_0_5_: texture_2d<u32>;
@group(0) @binding(2) 
var<storage> storage_0_2_: array<u32>;
@group(0) @binding(19) 
var<uniform> uniform_struct_0_19_: UniformStruct5_;
@group(0) @binding(17) 
var tex_0_17_: texture_2d<f32>;
@group(0) @binding(6) 
var tex_0_6_: texture_2d<f32>;
@group(0) @binding(7) 
var tex_0_7_: texture_2d<f32>;
@group(0) @binding(1) 
var<uniform> uniform_struct_0_1_: UniformStruct4_;
@group(0) @binding(22) 
var<storage> storage_0_22_: array<u32>;
@group(0) @binding(21) 
var<storage> storage_0_21_: array<u32>;
@group(0) @binding(20) 
var<storage> storage_0_20_: array<u32>;
@group(0) @binding(18) 
var tex_0_18_: texture_depth_2d_array;
@group(0) @binding(11) 
var sampler_0_11_: sampler_comparison;
@group(0) @binding(10) 
var tex_0_10_: texture_depth_2d_array;
@group(0) @binding(12) 
var tex_0_12_: texture_depth_cube_array;
@group(0) @binding(13) 
var tex_0_13_: texture_cube<f32>;
@group(0) @binding(16) 
var sampler_0_16_: sampler;
@group(0) @binding(14) 
var tex_0_14_: texture_cube<f32>;
@group(0) @binding(15) 
var tex_0_15_: texture_2d<f32>;

@fragment 
fn fs_main_lighting(@builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var fragment_out: vec4<f32>;
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i_u002e_i: u32;
    var radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_12_u002e_2_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_20_u002e_2_: f32;
    var vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i: f32;
    var _0_u002e_sroa_u002e_0_u002e_0_: f32;
    var extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_: f32;
    var extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_: f32;
    var emission_u002e_i_u002e_sroa_u002e_0_u002e_1_: f32;
    var emission_u002e_i_u002e_sroa_u002e_8_u002e_1_: f32;
    var emission_u002e_i_u002e_sroa_u002e_11_u002e_1_: f32;
    var ss_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var ss_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_: f32;
    var ss_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_: f32;
    var trans_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var trans_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_: f32;
    var trans_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_: f32;
    var _4_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var coat_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var coat_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_: f32;
    var coat_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_: f32;
    var fuzz_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var fuzz_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_: f32;
    var fuzz_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_: f32;
    var z_bad_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var _15_u002e_i77_u002e_sroa_u002e_0_u002e_0_: f32;
    var _15_u002e_i77_u002e_sroa_u002e_5_u002e_0_: f32;
    var _15_u002e_i77_u002e_sroa_u002e_8_u002e_0_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_12_u002e_1_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_20_u002e_1_: f32;
    var shadow_vis_u002e_sroa_u002e_0_u002e_1_u002e_i: f32;
    var _2_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var x_bad_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var y_bad_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var spec_acc_u002e_i_u002e_sroa_u002e_0_u002e_2_: f32;
    var spec_acc_u002e_i_u002e_sroa_u002e_6_u002e_2_: f32;
    var spec_acc_u002e_i_u002e_sroa_u002e_9_u002e_2_: f32;
    var diff_acc_u002e_i_u002e_sroa_u002e_0_u002e_2_: f32;
    var diff_acc_u002e_i_u002e_sroa_u002e_6_u002e_2_: f32;
    var diff_acc_u002e_i_u002e_sroa_u002e_9_u002e_2_: f32;
    var shadow_vis_u002e_sroa_u002e_0_u002e_0_: f32;
    var k_u002e_sroa_u002e_0_u002e_0_: u32;
    var diff_acc_u002e_i_u002e_sroa_u002e_9_: f32;
    var diff_acc_u002e_i_u002e_sroa_u002e_6_: f32;
    var diff_acc_u002e_i_u002e_sroa_u002e_0_: f32;
    var spec_acc_u002e_i_u002e_sroa_u002e_9_: f32;
    var spec_acc_u002e_i_u002e_sroa_u002e_6_: f32;
    var spec_acc_u002e_i_u002e_sroa_u002e_0_: f32;
    var lo_u002e_i_u002e_sroa_u002e_9_: f32;
    var lo_u002e_i_u002e_sroa_u002e_6_: f32;
    var lo_u002e_i_u002e_sroa_u002e_0_: f32;
    var lo_u002e_i_u002e_sroa_u002e_0_u002e_1_: f32;
    var lo_u002e_i_u002e_sroa_u002e_6_u002e_1_: f32;
    var lo_u002e_i_u002e_sroa_u002e_9_u002e_1_: f32;
    var spec_acc_u002e_i_u002e_sroa_u002e_0_u002e_1_: f32;
    var spec_acc_u002e_i_u002e_sroa_u002e_6_u002e_1_: f32;
    var spec_acc_u002e_i_u002e_sroa_u002e_9_u002e_1_: f32;
    var diff_acc_u002e_i_u002e_sroa_u002e_0_u002e_1_: f32;
    var diff_acc_u002e_i_u002e_sroa_u002e_6_u002e_1_: f32;
    var diff_acc_u002e_i_u002e_sroa_u002e_9_u002e_1_: f32;
    var lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa: f32;
    var lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa: f32;
    var spec_acc_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var spec_acc_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa: f32;
    var spec_acc_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa: f32;
    var diff_acc_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var diff_acc_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa: f32;
    var diff_acc_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa: f32;
    var shadow_vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa: f32;
    var radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_12_u002e_0_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_20_u002e_0_: f32;
    var _3_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var out_u002e_lo_u002e_i_u002e_sroa_u002e_0_u002e_1_: f32;
    var out_u002e_lo_u002e_i_u002e_sroa_u002e_6_u002e_1_: f32;
    var out_u002e_lo_u002e_i_u002e_sroa_u002e_9_u002e_1_: f32;
    var out_u002e_spec_acc_u002e_i_u002e_sroa_u002e_0_u002e_1_: f32;
    var out_u002e_spec_acc_u002e_i_u002e_sroa_u002e_6_u002e_1_: f32;
    var out_u002e_spec_acc_u002e_i_u002e_sroa_u002e_9_u002e_1_: f32;
    var out_u002e_diff_acc_u002e_i_u002e_sroa_u002e_0_u002e_1_: f32;
    var out_u002e_diff_acc_u002e_i_u002e_sroa_u002e_6_u002e_1_: f32;
    var out_u002e_diff_acc_u002e_i_u002e_sroa_u002e_9_u002e_1_: f32;
    var out_u002e_shadow_vis_u002e_sroa_u002e_0_u002e_1_u002e_i: f32;

    let _e5 = textureDimensions(tex_0_8_);
    let _e8 = textureDimensions(tex_0_8_);
    let _e13 = u32((uv.x * f32(bitcast<i32>(_e5.x))));
    let _e17 = u32((uv.y * f32(bitcast<i32>(_e8.y))));
    let _e23 = textureLoad(tex_0_8_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e17)), 0i);
    let _e28 = textureSampleLevel(tex_0_3_, sampler_0_9_, vec2<f32>(uv.x, uv.y), 0f);
    let _e34 = textureSampleLevel(tex_0_4_, sampler_0_9_, vec2<f32>(uv.x, uv.y), 0f);
    let _e40 = textureSampleLevel(tex_0_4_, sampler_0_9_, vec2<f32>(uv.x, uv.y), 0f);
    let _e45 = ((uv.x * 2f) + -1f);
    let _e49 = (1f - (uv.y * 2f));
    let _e53 = uniform_struct_0_0_.f4_.w;
    let _e57 = uniform_struct_0_0_.f5_.w;
    let _e61 = uniform_struct_0_0_.f6_.w;
    let _e65 = uniform_struct_0_0_.f7_.w;
    let _e71 = (_e65 + (((_e45 * _e53) + (_e49 * _e57)) + (_e23 * _e61)));
    let _e75 = uniform_struct_0_0_.f4_.x;
    let _e79 = uniform_struct_0_0_.f5_.x;
    let _e83 = uniform_struct_0_0_.f6_.x;
    let _e87 = uniform_struct_0_0_.f7_.x;
    let _e97 = uniform_struct_0_0_.f4_.y;
    let _e101 = uniform_struct_0_0_.f5_.y;
    let _e105 = uniform_struct_0_0_.f6_.y;
    let _e109 = uniform_struct_0_0_.f7_.y;
    let _e119 = uniform_struct_0_0_.f4_.z;
    let _e123 = uniform_struct_0_0_.f5_.z;
    let _e127 = uniform_struct_0_0_.f6_.z;
    let _e131 = uniform_struct_0_0_.f7_.z;
    let _e138 = ((_e87 + (((_e45 * _e75) + (_e49 * _e79)) + (_e23 * _e83))) / _e71);
    let _e139 = ((_e109 + (((_e45 * _e97) + (_e49 * _e101)) + (_e23 * _e105))) / _e71);
    let _e140 = ((_e131 + (((_e45 * _e119) + (_e49 * _e123)) + (_e23 * _e127))) / _e71);
    let _e146 = textureLoad(tex_0_5_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e17)), 0i);
    let _e152 = ((1f - abs(_e34.x)) - abs(_e40.y));
    let _e157 = select(0f, -(_e152), (_e152 < 0f));
    let _e161 = -(_e157);
    let _e167 = (_e34.x + select(_e161, _e157, !((_e34.x >= 0f))));
    let _e168 = (_e40.y + select(_e161, _e157, !((_e40.y >= 0f))));
    let _e172 = (1f / length(vec3<f32>(_e167, _e168, _e152)));
    let _e173 = (_e172 * _e167);
    let _e174 = (_e172 * _e168);
    let _e175 = (_e152 * _e172);
    let _e179 = uniform_struct_0_0_.f8_.x;
    let _e183 = uniform_struct_0_0_.f8_.y;
    let _e187 = uniform_struct_0_0_.f8_.z;
    let _e188 = (_e179 - _e138);
    let _e189 = (_e183 - _e139);
    let _e190 = (_e187 - _e140);
    let _e194 = (1f / length(vec3<f32>(_e188, _e189, _e190)));
    let _e195 = (_e188 * _e194);
    let _e196 = (_e189 * _e194);
    let _e197 = (_e190 * _e194);
    let _e200 = dot(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e195, _e196, _e197));
    let _e202 = (_e200 > 0.000001f);
    let _e204 = select(0.000001f, _e200, _e202);
    let _e206 = (_e146.x * 80u);
    let _e209 = storage_0_2_[_e206];
    let _e214 = storage_0_2_[(_e206 | 1u)];
    let _e219 = storage_0_2_[(_e206 | 2u)];
    let _e224 = storage_0_2_[(_e206 | 4u)];
    let _e229 = storage_0_2_[(_e206 | 5u)];
    let _e234 = storage_0_2_[(_e206 | 6u)];
    let _e239 = storage_0_2_[(_e206 | 8u)];
    let _e241 = (_e206 | 9u);
    let _e244 = storage_0_2_[_e241];
    let _e245 = bitcast<f32>(_e244);
    let _e246 = dpdx(_e173);
    let _e247 = dpdx(_e174);
    let _e248 = dpdx(_e175);
    let _e249 = dpdy(_e173);
    let _e250 = dpdy(_e174);
    let _e251 = dpdy(_e175);
    let _e262 = (((dot(vec3<f32>(_e246, _e247, _e248), vec3<f32>(_e246, _e247, _e248)) + dot(vec3<f32>(_e249, _e250, _e251), vec3<f32>(_e249, _e250, _e251))) * 0.15f) * 2f);
    let _e267 = (_e245 * _e245);
    let _e269 = ((_e267 * _e267) + select(0.2f, _e262, (_e262 < 0.2f)));
    let _e273 = select(_e269, 0f, (_e269 < 0f));
    let _e277 = select(_e273, 1f, (_e273 > 1f));
    let _e283 = sqrt(sqrt(select(0f, _e277, (_e277 > 0f))));
    let _e291 = storage_0_2_[(_e206 | 10u)];
    let _e296 = storage_0_2_[(_e206 | 11u)];
    let _e301 = storage_0_2_[(_e206 | 12u)];
    let _e306 = storage_0_2_[(_e206 | 13u)];
    let _e311 = storage_0_2_[(_e206 | 14u)];
    let _e316 = storage_0_2_[(_e206 + 16u)];
    let _e321 = storage_0_2_[(_e206 + 17u)];
    let _e336 = storage_0_2_[(_e206 + 20u)];
    let _e341 = storage_0_2_[(_e206 + 21u)];
    let _e346 = storage_0_2_[(_e206 + 22u)];
    let _e371 = storage_0_2_[(_e206 + 28u)];
    let _e376 = storage_0_2_[(_e206 + 29u)];
    let _e381 = storage_0_2_[(_e206 + 30u)];
    let _e386 = storage_0_2_[(_e206 + 31u)];
    let _e391 = storage_0_2_[(_e206 + 32u)];
    let _e396 = storage_0_2_[(_e206 + 33u)];
    let _e401 = storage_0_2_[(_e206 + 34u)];
    let _e406 = storage_0_2_[(_e206 + 36u)];
    let _e411 = storage_0_2_[(_e206 + 37u)];
    let _e416 = storage_0_2_[(_e206 + 40u)];
    let _e421 = storage_0_2_[(_e206 + 41u)];
    let _e426 = storage_0_2_[(_e206 + 44u)];
    let _e431 = storage_0_2_[(_e206 + 45u)];
    let _e436 = storage_0_2_[(_e206 + 46u)];
    let _e441 = storage_0_2_[(_e206 + 48u)];
    let _e446 = storage_0_2_[(_e206 + 49u)];
    let _e451 = storage_0_2_[(_e206 + 50u)];
    let _e456 = storage_0_2_[(_e206 + 51u)];
    let _e461 = storage_0_2_[(_e206 + 52u)];
    let _e466 = storage_0_2_[(_e206 + 53u)];
    let _e471 = storage_0_2_[(_e206 + 54u)];
    let _e476 = storage_0_2_[(_e206 + 56u)];
    let _e481 = storage_0_2_[(_e206 + 60u)];
    let _e486 = storage_0_2_[(_e206 + 61u)];
    let _e491 = storage_0_2_[(_e206 + 62u)];
    let _e496 = storage_0_2_[(_e206 + 64u)];
    let _e501 = storage_0_2_[(_e206 + 68u)];
    let _e506 = storage_0_2_[(_e206 + 69u)];
    let _e511 = storage_0_2_[(_e206 + 70u)];
    let _e516 = storage_0_2_[(_e206 + 72u)];
    let _e526 = storage_0_2_[(_e206 + 74u)];
    let _e549 = (cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(0f, 1f, 0f)).x + 0f);
    let _e551 = (cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(0f, 1f, 0f)).y + 0f);
    let _e553 = (cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(0f, 1f, 0f)).z + 0.000001f);
    let _e557 = (1f / length(vec3<f32>(_e549, _e551, _e553)));
    let _e558 = (_e549 * _e557);
    let _e559 = (_e551 * _e557);
    let _e560 = (_e553 * _e557);
    let _e576 = uniform_struct_0_0_.f0_.x;
    let _e580 = uniform_struct_0_0_.f1_.x;
    let _e584 = uniform_struct_0_0_.f2_.x;
    let _e588 = uniform_struct_0_0_.f3_.x;
    let _e598 = uniform_struct_0_0_.f0_.y;
    let _e602 = uniform_struct_0_0_.f1_.y;
    let _e606 = uniform_struct_0_0_.f2_.y;
    let _e610 = uniform_struct_0_0_.f3_.y;
    let _e620 = uniform_struct_0_0_.f0_.w;
    let _e624 = uniform_struct_0_0_.f1_.w;
    let _e628 = uniform_struct_0_0_.f2_.w;
    let _e632 = uniform_struct_0_0_.f3_.w;
    let _e638 = (_e632 + (((_e138 * _e620) + (_e139 * _e624)) + (_e140 * _e628)));
    let _e642 = select(0.5f, _e638, (_e638 > 0.5f));
    let _e683 = ((((min(u32(floor((((((_e610 + (((_e138 * _e598) + (_e139 * _e602)) + (_e140 * _e606))) / _e642) * 0.5f) + 0.5f) * 9f))), 8u) << 4u) | min(u32(floor((((((_e588 + (((_e138 * _e576) + (_e139 * _e580)) + (_e140 * _e584))) / _e642) * 0.5f) + 0.5f) * 16f))), 15u)) + (min(u32(ceil(((log((_e642 * 2f)) * 3.4407384f) + -0.0001f))), 23u) * 144u)) << 1u);
    let _e686 = uniform_struct_0_19_.f2_;
    let _e691 = textureSampleLevel(tex_0_17_, sampler_0_9_, vec2<f32>(uv.x, uv.y), 0f);
    let _e697 = textureSampleLevel(tex_0_6_, sampler_0_9_, vec2<f32>(uv.x, uv.y), 0f);
    let _e703 = textureSampleLevel(tex_0_7_, sampler_0_9_, vec2<f32>(uv.x, uv.y), 0f);
    let _e706 = bitcast<f32>(_e491);
    let _e710 = (_e204 * _e204);
    let _e714 = max((1f - _e710), 0f);
    let _e716 = (sqrt(_e714) / _e706);
    let _e729 = pow(((_e706 + -1f) / (_e706 + 1f)), 2f);
    let _e733 = (((bitcast<f32>(_e486) * 1000f) * (_e706 * 12.566371f)) * sqrt(max((1f - (_e716 * _e716)), 0f)));
    let _e744 = (_e729 * 2f);
    let _e750 = (1f - (_e729 * _e729));
    let _e762 = clamp(bitcast<f32>(_e481), 0f, 1f);
    let _e767 = ((_e762 * ((((_e744 * cos((_e733 / 650f))) / _e750) + 1f) + -1f)) + 1f);
    let _e772 = ((_e762 * ((((_e744 * cos((_e733 / 550f))) / _e750) + 1f) + -1f)) + 1f);
    let _e777 = ((_e762 * ((((_e744 * cos((_e733 / 450f))) / _e750) + 1f) + -1f)) + 1f);
    let _e778 = bitcast<f32>(_e441);
    let _e779 = bitcast<f32>(_e476);
    let _e780 = bitcast<f32>(_e219);
    let _e781 = bitcast<f32>(_e209);
    let _e782 = bitcast<f32>(_e239);
    let _e783 = bitcast<f32>(_e371);
    let _e788 = ((_e779 + -1f) / (_e779 + 1f));
    let _e791 = bitcast<f32>(_e224);
    let _e792 = bitcast<f32>(_e229);
    let _e793 = bitcast<f32>(_e234);
    let _e794 = (_e781 * _e791);
    let _e795 = (_e781 * _e792);
    let _e796 = (_e781 * _e793);
    let _e797 = bitcast<f32>(_e391);
    let _e798 = bitcast<f32>(_e396);
    let _e799 = bitcast<f32>(_e401);
    let _e802 = !((_e778 <= 0f));
    if _e802 {
        let _e808 = (1f - ((1f - pow(_e788, 2f)) / (_e779 * _e779)));
        let _e810 = (1f - _e808);
        let _e811 = (_e796 * _e782);
        let _e824 = (_e795 * _e782);
        let _e837 = (_e794 * _e782);
        let _e851 = (_e778 * bitcast<f32>(_e456));
        _15_u002e_i77_u002e_sroa_u002e_0_u002e_0_ = ((_e851 * ((_e810 / max((1f - ((_e837 + (((((_e791 - _e797) * _e783) + _e797) - _e837) * _e780)) * _e808)), 0.000001f)) + -1f)) + 1f);
        _15_u002e_i77_u002e_sroa_u002e_5_u002e_0_ = ((_e851 * ((_e810 / max((1f - ((_e824 + (((((_e792 - _e798) * _e783) + _e798) - _e824) * _e780)) * _e808)), 0.000001f)) + -1f)) + 1f);
        _15_u002e_i77_u002e_sroa_u002e_8_u002e_0_ = ((_e851 * ((_e810 / max((1f - ((_e811 + (((((_e793 - _e799) * _e783) + _e799) - _e811) * _e780)) * _e808)), 0.000001f)) + -1f)) + 1f);
    } else {
        _15_u002e_i77_u002e_sroa_u002e_0_u002e_0_ = 1f;
        _15_u002e_i77_u002e_sroa_u002e_5_u002e_0_ = 1f;
        _15_u002e_i77_u002e_sroa_u002e_8_u002e_0_ = 1f;
    }
    let _e877 = _15_u002e_i77_u002e_sroa_u002e_0_u002e_0_;
    let _e879 = _15_u002e_i77_u002e_sroa_u002e_5_u002e_0_;
    let _e881 = _15_u002e_i77_u002e_sroa_u002e_8_u002e_0_;
    let _e884 = uniform_struct_0_1_.f5_;
    let _e887 = uniform_struct_0_1_.f2_;
    let _e890 = storage_0_22_[_e683];
    let _e896 = storage_0_22_[(_e683 | 1u)];
    if (_e28.w < 0.001f) {
        discard;
    }
    let _e900 = ((_e896 * _e686) + _e887);
    if (_e900 == 0u) {
        lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
        lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = 0f;
        lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = 0f;
        spec_acc_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
        spec_acc_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = 0f;
        spec_acc_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = 0f;
        diff_acc_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
        diff_acc_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = 0f;
        diff_acc_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = 0f;
        shadow_vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = 1f;
    } else {
        let _e932 = bitcast<f32>(_e214);
        let _e933 = bitcast<f32>(_e291);
        let _e939 = ((_e933 + -1f) / (_e933 + 1f));
        let _e940 = (_e939 * _e939);
        let _e941 = bitcast<f32>(_e301);
        let _e942 = bitcast<f32>(_e306);
        let _e943 = bitcast<f32>(_e311);
        let _e944 = (_e283 * _e283);
        let _e946 = (1f - bitcast<f32>(_e296));
        let _e951 = (2f / ((_e946 * _e946) + 1f));
        let _e953 = select(_e283, _e932, (_e283 < _e932));
        let _e957 = (_e195 - (_e173 * _e204));
        let _e958 = (_e196 - (_e174 * _e204));
        let _e959 = (_e197 - (_e175 * _e204));
        let _e962 = (1f - _e780);
        let _e967 = max((_e953 * _e953), 0.000001f);
        let _e968 = (_e967 * _e967);
        let _e986 = bitcast<f32>(_e446);
        let _e987 = (_e788 * _e788);
        let _e990 = (_e986 * _e986);
        let _e992 = (1f - bitcast<f32>(_e451));
        let _e997 = (2f / ((_e992 * _e992) + 1f));
        let _e1001 = bitcast<f32>(_e416);
        let _e1005 = bitcast<f32>(_e421);
        let _e1006 = (_e1005 * _e1005);
        let _e1016 = bitcast<f32>(_e316);
        let _e1017 = bitcast<f32>(_e321);
        let _e1049 = bitcast<f32>(_e376);
        shadow_vis_u002e_sroa_u002e_0_u002e_0_ = 1f;
        k_u002e_sroa_u002e_0_u002e_0_ = 0u;
        diff_acc_u002e_i_u002e_sroa_u002e_9_ = 0f;
        diff_acc_u002e_i_u002e_sroa_u002e_6_ = 0f;
        diff_acc_u002e_i_u002e_sroa_u002e_0_ = 0f;
        spec_acc_u002e_i_u002e_sroa_u002e_9_ = 0f;
        spec_acc_u002e_i_u002e_sroa_u002e_6_ = 0f;
        spec_acc_u002e_i_u002e_sroa_u002e_0_ = 0f;
        lo_u002e_i_u002e_sroa_u002e_9_ = 0f;
        lo_u002e_i_u002e_sroa_u002e_6_ = 0f;
        lo_u002e_i_u002e_sroa_u002e_0_ = 0f;
        loop {
            let _e1088 = shadow_vis_u002e_sroa_u002e_0_u002e_0_;
            let _e1090 = k_u002e_sroa_u002e_0_u002e_0_;
            let _e1092 = diff_acc_u002e_i_u002e_sroa_u002e_9_;
            let _e1094 = diff_acc_u002e_i_u002e_sroa_u002e_6_;
            let _e1096 = diff_acc_u002e_i_u002e_sroa_u002e_0_;
            let _e1098 = spec_acc_u002e_i_u002e_sroa_u002e_9_;
            let _e1100 = spec_acc_u002e_i_u002e_sroa_u002e_6_;
            let _e1102 = spec_acc_u002e_i_u002e_sroa_u002e_0_;
            let _e1104 = lo_u002e_i_u002e_sroa_u002e_9_;
            let _e1106 = lo_u002e_i_u002e_sroa_u002e_6_;
            let _e1108 = lo_u002e_i_u002e_sroa_u002e_0_;
            if (_e1090 < _e887) {
                _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i_u002e_i = _e1090;
            } else {
                let _e1114 = storage_0_21_[(((_e890 * _e686) - _e887) + _e1090)];
                _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i_u002e_i = _e1114;
            }
            let _e1117 = _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i_u002e_i;
            let _e1119 = (_e1117 * 36u);
            let _e1122 = storage_0_20_[_e1119];
            let _e1123 = bitcast<f32>(_e1122);
            let _e1125 = (_e1119 + 8u);
            let _e1128 = storage_0_20_[_e1125];
            let _e1130 = (_e1119 + 9u);
            let _e1133 = storage_0_20_[_e1130];
            let _e1135 = (_e1119 + 10u);
            let _e1138 = storage_0_20_[_e1135];
            let _e1146 = length(vec3<f32>((bitcast<f32>(_e1128) - _e138), (bitcast<f32>(_e1133) - _e139), (bitcast<f32>(_e1138) - _e140)));
            let _e1148 = (_e1123 < 0.5f);
            let _e1151 = select(1f, 0f, _e1148);
            let _e1153 = (_e1119 + 16u);
            let _e1156 = storage_0_20_[_e1153];
            let _e1157 = bitcast<f32>(_e1156);
            let _e1169 = storage_0_20_[(_e1119 + 12u)];
            let _e1174 = storage_0_20_[(_e1119 + 13u)];
            let _e1179 = storage_0_20_[(_e1119 + 14u)];
            let _e1184 = storage_0_20_[(_e1119 + 15u)];
            let _e1185 = bitcast<f32>(_e1184);
            let _e1188 = storage_0_20_[_e1125];
            let _e1191 = storage_0_20_[_e1130];
            let _e1194 = storage_0_20_[_e1135];
            let _e1196 = (_e138 - bitcast<f32>(_e1188));
            let _e1198 = (_e139 - bitcast<f32>(_e1191));
            let _e1200 = (_e140 - bitcast<f32>(_e1194));
            let _e1204 = (1f / length(vec3<f32>(_e1196, _e1198, _e1200)));
            let _e1209 = (_e1119 + 4u);
            let _e1212 = storage_0_20_[_e1209];
            let _e1214 = (_e1119 + 5u);
            let _e1217 = storage_0_20_[_e1214];
            let _e1219 = (_e1119 + 6u);
            let _e1222 = storage_0_20_[_e1219];
            let _e1223 = bitcast<f32>(_e1212);
            let _e1224 = bitcast<f32>(_e1217);
            let _e1225 = bitcast<f32>(_e1222);
            let _e1229 = (1f / length(vec3<f32>(_e1223, _e1224, _e1225)));
            let _e1240 = storage_0_20_[(_e1119 + 18u)];
            let _e1246 = storage_0_20_[(_e1119 + 17u)];
            let _e1253 = select(1f, 0f, (_e1123 < 1.5f));
            let _e1255 = mix(1f, smoothstep(bitcast<f32>(_e1240), bitcast<f32>(_e1246), dot(vec3<f32>((_e1229 * _e1223), (_e1229 * _e1224), (_e1229 * _e1225)), vec3<f32>((_e1196 * _e1204), (_e1198 * _e1204), (_e1200 * _e1204)))), _e1253);
            let _e1256 = (_e1146 * _e1146);
            let _e1271 = (mix(1f, (1f - smoothstep((_e1157 * 0.99f), _e1157, _e1146)), _e1151) * mix(1f, (1f / select(_e1256, 0.000001f, (_e1256 < 0.000001f))), _e1151));
            let _e1275 = (_e1255 * ((bitcast<f32>(_e1169) * _e1185) * _e1271));
            let _e1276 = (_e1255 * ((bitcast<f32>(_e1174) * _e1185) * _e1271));
            let _e1277 = (_e1255 * ((bitcast<f32>(_e1179) * _e1185) * _e1271));
            let _e1281 = select((1f - _e1253), 0f, _e1148);
            if (((1f - _e1151) * select(1f, 0f, (_e1157 < 0.5f))) > 0.5f) {
                let _e1296 = uniform_struct_0_1_.f6_.f1_.x;
                let _e1301 = uniform_struct_0_1_.f6_.f1_.y;
                let _e1306 = uniform_struct_0_1_.f6_.f1_.z;
                let _e1309 = dot(vec3<f32>((_e138 - _e179), (_e139 - _e183), (_e140 - _e187)), vec3<f32>(_e1296, _e1301, _e1306));
                let _e1314 = uniform_struct_0_1_.f6_.f2_.x;
                let _e1319 = uniform_struct_0_1_.f6_.f2_.y;
                let _e1324 = uniform_struct_0_1_.f6_.f2_.z;
                let _e1329 = uniform_struct_0_1_.f6_.f2_.w;
                let _e1333 = uniform_struct_0_1_.f6_.f3_;
                let _e1337 = select(1f, 0f, (_e1309 < _e1319));
                let _e1341 = select(1f, 0f, (_e1309 < _e1324));
                let _e1345 = select(1f, 0f, (_e1309 < _e1329));
                let _e1348 = u32(((_e1337 + _e1341) + _e1345));
                let _e1351 = mix(mix(mix(_e1319, _e1324, _e1337), _e1329, _e1341), _e1333, _e1345);
                let _e1361 = uniform_struct_0_1_.f6_.f0_[_e1348].f0_.x;
                let _e1368 = uniform_struct_0_1_.f6_.f0_[_e1348].f1_.x;
                let _e1375 = uniform_struct_0_1_.f6_.f0_[_e1348].f2_.x;
                let _e1382 = uniform_struct_0_1_.f6_.f0_[_e1348].f3_.x;
                let _e1395 = uniform_struct_0_1_.f6_.f0_[_e1348].f0_.y;
                let _e1402 = uniform_struct_0_1_.f6_.f0_[_e1348].f1_.y;
                let _e1409 = uniform_struct_0_1_.f6_.f0_[_e1348].f2_.y;
                let _e1416 = uniform_struct_0_1_.f6_.f0_[_e1348].f3_.y;
                let _e1429 = uniform_struct_0_1_.f6_.f0_[_e1348].f0_.z;
                let _e1436 = uniform_struct_0_1_.f6_.f0_[_e1348].f1_.z;
                let _e1443 = uniform_struct_0_1_.f6_.f0_[_e1348].f2_.z;
                let _e1450 = uniform_struct_0_1_.f6_.f0_[_e1348].f3_.z;
                let _e1463 = uniform_struct_0_1_.f6_.f0_[_e1348].f0_.w;
                let _e1470 = uniform_struct_0_1_.f6_.f0_[_e1348].f1_.w;
                let _e1477 = uniform_struct_0_1_.f6_.f0_[_e1348].f2_.w;
                let _e1484 = uniform_struct_0_1_.f6_.f0_[_e1348].f3_.w;
                let _e1490 = (_e1484 + (((_e138 * _e1463) + (_e139 * _e1470)) + (_e140 * _e1477)));
                let _e1507 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e1382 + (((_e138 * _e1361) + (_e139 * _e1368)) + (_e140 * _e1375))) / _e1490) * 0.5f) + 0.5f), (0.5f - (((_e1416 + (((_e138 * _e1395) + (_e139 * _e1402)) + (_e140 * _e1409))) / _e1490) * 0.5f))), _e1348, (((_e1450 + (((_e138 * _e1429) + (_e139 * _e1436)) + (_e140 * _e1443))) / _e1490) + -0.001f));
                if (_e1348 < 3u) {
                    let _e1512 = ((_e1351 - mix(mix(mix(_e1314, _e1319, _e1337), _e1324, _e1341), _e1329, _e1345)) * 0.08f);
                    let _e1518 = ((_e1351 - _e1309) / select(_e1512, 0.000001f, (_e1512 < 0.000001f)));
                    let _e1522 = select(_e1518, 0f, (_e1518 < 0f));
                    let _e1528 = (_e1348 + 1u);
                    let _e1535 = uniform_struct_0_1_.f6_.f0_[_e1528].f0_.x;
                    let _e1542 = uniform_struct_0_1_.f6_.f0_[_e1528].f1_.x;
                    let _e1549 = uniform_struct_0_1_.f6_.f0_[_e1528].f2_.x;
                    let _e1556 = uniform_struct_0_1_.f6_.f0_[_e1528].f3_.x;
                    let _e1569 = uniform_struct_0_1_.f6_.f0_[_e1528].f0_.y;
                    let _e1576 = uniform_struct_0_1_.f6_.f0_[_e1528].f1_.y;
                    let _e1583 = uniform_struct_0_1_.f6_.f0_[_e1528].f2_.y;
                    let _e1590 = uniform_struct_0_1_.f6_.f0_[_e1528].f3_.y;
                    let _e1603 = uniform_struct_0_1_.f6_.f0_[_e1528].f0_.z;
                    let _e1610 = uniform_struct_0_1_.f6_.f0_[_e1528].f1_.z;
                    let _e1617 = uniform_struct_0_1_.f6_.f0_[_e1528].f2_.z;
                    let _e1624 = uniform_struct_0_1_.f6_.f0_[_e1528].f3_.z;
                    let _e1637 = uniform_struct_0_1_.f6_.f0_[_e1528].f0_.w;
                    let _e1644 = uniform_struct_0_1_.f6_.f0_[_e1528].f1_.w;
                    let _e1651 = uniform_struct_0_1_.f6_.f0_[_e1528].f2_.w;
                    let _e1658 = uniform_struct_0_1_.f6_.f0_[_e1528].f3_.w;
                    let _e1664 = (_e1658 + (((_e138 * _e1637) + (_e139 * _e1644)) + (_e140 * _e1651)));
                    let _e1681 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e1556 + (((_e138 * _e1535) + (_e139 * _e1542)) + (_e140 * _e1549))) / _e1664) * 0.5f) + 0.5f), (0.5f - (((_e1590 + (((_e138 * _e1569) + (_e139 * _e1576)) + (_e140 * _e1583))) / _e1664) * 0.5f))), _e1528, (((_e1624 + (((_e138 * _e1603) + (_e139 * _e1610)) + (_e140 * _e1617))) / _e1664) + -0.001f));
                    vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i = mix(_e1681, _e1507, select(_e1522, 1f, (_e1522 > 1f)));
                } else {
                    vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i = _e1507;
                }
                let _e1686 = vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i;
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (_e1275 * _e1686);
                radiance_u002e_i_u002e_sroa_u002e_12_u002e_0_ = (_e1276 * _e1686);
                radiance_u002e_i_u002e_sroa_u002e_20_u002e_0_ = (_e1277 * _e1686);
            } else {
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_ = _e1275;
                radiance_u002e_i_u002e_sroa_u002e_12_u002e_0_ = _e1276;
                radiance_u002e_i_u002e_sroa_u002e_20_u002e_0_ = _e1277;
            }
            let _e1697 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_;
            let _e1699 = radiance_u002e_i_u002e_sroa_u002e_12_u002e_0_;
            let _e1701 = radiance_u002e_i_u002e_sroa_u002e_20_u002e_0_;
            let _e1703 = (_e1119 + 19u);
            let _e1706 = storage_0_20_[_e1703];
            if !((bitcast<f32>(_e1706) >= 0f)) {
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = _e1697;
                radiance_u002e_i_u002e_sroa_u002e_12_u002e_2_ = _e1699;
                radiance_u002e_i_u002e_sroa_u002e_20_u002e_2_ = _e1701;
            } else {
                if !((_e1281 <= 0.5f)) {
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e1697;
                    radiance_u002e_i_u002e_sroa_u002e_12_u002e_1_ = _e1699;
                    radiance_u002e_i_u002e_sroa_u002e_20_u002e_1_ = _e1701;
                } else {
                    let _e1724 = storage_0_20_[(_e1119 + 20u)];
                    let _e1730 = storage_0_20_[(_e1119 + 24u)];
                    let _e1736 = storage_0_20_[(_e1119 + 28u)];
                    let _e1742 = storage_0_20_[(_e1119 + 32u)];
                    let _e1754 = storage_0_20_[(_e1119 + 21u)];
                    let _e1760 = storage_0_20_[(_e1119 + 25u)];
                    let _e1766 = storage_0_20_[(_e1119 + 29u)];
                    let _e1772 = storage_0_20_[(_e1119 + 33u)];
                    let _e1784 = storage_0_20_[(_e1119 + 22u)];
                    let _e1790 = storage_0_20_[(_e1119 + 26u)];
                    let _e1796 = storage_0_20_[(_e1119 + 30u)];
                    let _e1802 = storage_0_20_[(_e1119 + 34u)];
                    let _e1814 = storage_0_20_[(_e1119 + 23u)];
                    let _e1820 = storage_0_20_[(_e1119 + 27u)];
                    let _e1826 = storage_0_20_[(_e1119 + 31u)];
                    let _e1832 = storage_0_20_[(_e1119 + 35u)];
                    let _e1839 = ((((_e138 * bitcast<f32>(_e1814)) + (_e139 * bitcast<f32>(_e1820))) + (_e140 * bitcast<f32>(_e1826))) + bitcast<f32>(_e1832));
                    let _e1852 = storage_0_20_[_e1703];
                    let _e1861 = textureSampleCompareLevel(tex_0_10_, sampler_0_11_, vec2<f32>((((((((_e138 * bitcast<f32>(_e1724)) + (_e139 * bitcast<f32>(_e1730))) + (_e140 * bitcast<f32>(_e1736))) + bitcast<f32>(_e1742)) / _e1839) * 0.5f) + 0.5f), (0.5f - ((((((_e138 * bitcast<f32>(_e1754)) + (_e139 * bitcast<f32>(_e1760))) + (_e140 * bitcast<f32>(_e1766))) + bitcast<f32>(_e1772)) / _e1839) * 0.5f))), u32(bitcast<f32>(_e1852)), ((((((_e138 * bitcast<f32>(_e1784)) + (_e139 * bitcast<f32>(_e1790))) + (_e140 * bitcast<f32>(_e1796))) + bitcast<f32>(_e1802)) / _e1839) + -0.001f));
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_ = (_e1697 * _e1861);
                    radiance_u002e_i_u002e_sroa_u002e_12_u002e_1_ = (_e1699 * _e1861);
                    radiance_u002e_i_u002e_sroa_u002e_20_u002e_1_ = (_e1701 * _e1861);
                }
                let _e1869 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_;
                let _e1871 = radiance_u002e_i_u002e_sroa_u002e_12_u002e_1_;
                let _e1873 = radiance_u002e_i_u002e_sroa_u002e_20_u002e_1_;
                if (_e1281 > 0.5f) {
                    let _e1878 = storage_0_20_[_e1125];
                    let _e1881 = storage_0_20_[_e1130];
                    let _e1884 = storage_0_20_[_e1135];
                    let _e1886 = (_e138 - bitcast<f32>(_e1878));
                    let _e1888 = (_e139 - bitcast<f32>(_e1881));
                    let _e1890 = (_e140 - bitcast<f32>(_e1884));
                    let _e1891 = abs(_e1886);
                    let _e1892 = abs(_e1888);
                    let _e1894 = select(_e1891, _e1892, (_e1892 > _e1891));
                    let _e1895 = abs(_e1890);
                    let _e1900 = storage_0_20_[_e1153];
                    let _e1906 = select(bitcast<f32>(_e1900), 1f, (bitcast<f32>(_e1900) < 1f));
                    let _e1908 = (_e1906 + -0.1f);
                    let _e1919 = storage_0_20_[_e1703];
                    let _e1925 = textureSampleCompareLevel(tex_0_12_, sampler_0_11_, vec3<f32>(_e1886, _e1888, _e1890), u32(bitcast<f32>(_e1919)), (((_e1906 / _e1908) - ((_e1906 * 0.1f) / (select(_e1894, _e1895, (_e1895 > _e1894)) * _e1908))) + -0.001f));
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = (_e1869 * _e1925);
                    radiance_u002e_i_u002e_sroa_u002e_12_u002e_2_ = (_e1871 * _e1925);
                    radiance_u002e_i_u002e_sroa_u002e_20_u002e_2_ = (_e1873 * _e1925);
                } else {
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = _e1869;
                    radiance_u002e_i_u002e_sroa_u002e_12_u002e_2_ = _e1871;
                    radiance_u002e_i_u002e_sroa_u002e_20_u002e_2_ = _e1873;
                }
            }
            let _e1936 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_;
            let _e1938 = radiance_u002e_i_u002e_sroa_u002e_12_u002e_2_;
            let _e1940 = radiance_u002e_i_u002e_sroa_u002e_20_u002e_2_;
            if (_e884 == 6u) {
                let _e1943 = storage_0_20_[_e1119];
                let _e1944 = bitcast<f32>(_e1943);
                let _e1947 = storage_0_20_[_e1153];
                let _e1951 = storage_0_20_[_e1703];
                let _e1956 = select(1f, 0f, (_e1944 < 0.5f));
                if (((1f - _e1956) * select(1f, 0f, (bitcast<f32>(_e1947) < 0.5f))) > 0.5f) {
                    let _e1973 = uniform_struct_0_1_.f6_.f1_.x;
                    let _e1978 = uniform_struct_0_1_.f6_.f1_.y;
                    let _e1983 = uniform_struct_0_1_.f6_.f1_.z;
                    let _e1986 = dot(vec3<f32>((_e138 - _e179), (_e139 - _e183), (_e140 - _e187)), vec3<f32>(_e1973, _e1978, _e1983));
                    let _e1991 = uniform_struct_0_1_.f6_.f2_.x;
                    let _e1996 = uniform_struct_0_1_.f6_.f2_.y;
                    let _e2001 = uniform_struct_0_1_.f6_.f2_.z;
                    let _e2006 = uniform_struct_0_1_.f6_.f2_.w;
                    let _e2010 = uniform_struct_0_1_.f6_.f3_;
                    let _e2014 = select(1f, 0f, (_e1986 < _e1996));
                    let _e2018 = select(1f, 0f, (_e1986 < _e2001));
                    let _e2022 = select(1f, 0f, (_e1986 < _e2006));
                    let _e2025 = u32(((_e2014 + _e2018) + _e2022));
                    let _e2028 = mix(mix(mix(_e1996, _e2001, _e2014), _e2006, _e2018), _e2010, _e2022);
                    let _e2038 = uniform_struct_0_1_.f6_.f0_[_e2025].f0_.x;
                    let _e2045 = uniform_struct_0_1_.f6_.f0_[_e2025].f1_.x;
                    let _e2052 = uniform_struct_0_1_.f6_.f0_[_e2025].f2_.x;
                    let _e2059 = uniform_struct_0_1_.f6_.f0_[_e2025].f3_.x;
                    let _e2072 = uniform_struct_0_1_.f6_.f0_[_e2025].f0_.y;
                    let _e2079 = uniform_struct_0_1_.f6_.f0_[_e2025].f1_.y;
                    let _e2086 = uniform_struct_0_1_.f6_.f0_[_e2025].f2_.y;
                    let _e2093 = uniform_struct_0_1_.f6_.f0_[_e2025].f3_.y;
                    let _e2106 = uniform_struct_0_1_.f6_.f0_[_e2025].f0_.z;
                    let _e2113 = uniform_struct_0_1_.f6_.f0_[_e2025].f1_.z;
                    let _e2120 = uniform_struct_0_1_.f6_.f0_[_e2025].f2_.z;
                    let _e2127 = uniform_struct_0_1_.f6_.f0_[_e2025].f3_.z;
                    let _e2140 = uniform_struct_0_1_.f6_.f0_[_e2025].f0_.w;
                    let _e2147 = uniform_struct_0_1_.f6_.f0_[_e2025].f1_.w;
                    let _e2154 = uniform_struct_0_1_.f6_.f0_[_e2025].f2_.w;
                    let _e2161 = uniform_struct_0_1_.f6_.f0_[_e2025].f3_.w;
                    let _e2167 = (_e2161 + (((_e138 * _e2140) + (_e139 * _e2147)) + (_e140 * _e2154)));
                    let _e2184 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e2059 + (((_e138 * _e2038) + (_e139 * _e2045)) + (_e140 * _e2052))) / _e2167) * 0.5f) + 0.5f), (0.5f - (((_e2093 + (((_e138 * _e2072) + (_e139 * _e2079)) + (_e140 * _e2086))) / _e2167) * 0.5f))), _e2025, (((_e2127 + (((_e138 * _e2106) + (_e139 * _e2113)) + (_e140 * _e2120))) / _e2167) + -0.001f));
                    if (_e2025 < 3u) {
                        let _e2189 = ((_e2028 - mix(mix(mix(_e1991, _e1996, _e2014), _e2001, _e2018), _e2006, _e2022)) * 0.08f);
                        let _e2195 = ((_e2028 - _e1986) / select(_e2189, 0.000001f, (_e2189 < 0.000001f)));
                        let _e2199 = select(_e2195, 0f, (_e2195 < 0f));
                        let _e2205 = (_e2025 + 1u);
                        let _e2212 = uniform_struct_0_1_.f6_.f0_[_e2205].f0_.x;
                        let _e2219 = uniform_struct_0_1_.f6_.f0_[_e2205].f1_.x;
                        let _e2226 = uniform_struct_0_1_.f6_.f0_[_e2205].f2_.x;
                        let _e2233 = uniform_struct_0_1_.f6_.f0_[_e2205].f3_.x;
                        let _e2246 = uniform_struct_0_1_.f6_.f0_[_e2205].f0_.y;
                        let _e2253 = uniform_struct_0_1_.f6_.f0_[_e2205].f1_.y;
                        let _e2260 = uniform_struct_0_1_.f6_.f0_[_e2205].f2_.y;
                        let _e2267 = uniform_struct_0_1_.f6_.f0_[_e2205].f3_.y;
                        let _e2280 = uniform_struct_0_1_.f6_.f0_[_e2205].f0_.z;
                        let _e2287 = uniform_struct_0_1_.f6_.f0_[_e2205].f1_.z;
                        let _e2294 = uniform_struct_0_1_.f6_.f0_[_e2205].f2_.z;
                        let _e2301 = uniform_struct_0_1_.f6_.f0_[_e2205].f3_.z;
                        let _e2314 = uniform_struct_0_1_.f6_.f0_[_e2205].f0_.w;
                        let _e2321 = uniform_struct_0_1_.f6_.f0_[_e2205].f1_.w;
                        let _e2328 = uniform_struct_0_1_.f6_.f0_[_e2205].f2_.w;
                        let _e2335 = uniform_struct_0_1_.f6_.f0_[_e2205].f3_.w;
                        let _e2341 = (_e2335 + (((_e138 * _e2314) + (_e139 * _e2321)) + (_e140 * _e2328)));
                        let _e2358 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e2233 + (((_e138 * _e2212) + (_e139 * _e2219)) + (_e140 * _e2226))) / _e2341) * 0.5f) + 0.5f), (0.5f - (((_e2267 + (((_e138 * _e2246) + (_e139 * _e2253)) + (_e140 * _e2260))) / _e2341) * 0.5f))), _e2205, (((_e2301 + (((_e138 * _e2280) + (_e139 * _e2287)) + (_e140 * _e2294))) / _e2341) + -0.001f));
                        _0_u002e_sroa_u002e_0_u002e_0_ = mix(_e2358, _e2184, select(_e2199, 1f, (_e2199 > 1f)));
                    } else {
                        _0_u002e_sroa_u002e_0_u002e_0_ = _e2184;
                    }
                } else {
                    if (bitcast<f32>(_e1951) < 0f) {
                        _0_u002e_sroa_u002e_0_u002e_0_ = 1f;
                    } else {
                        if ((_e1944 < 1.5f) & (_e1956 > 0.5f)) {
                            let _e2372 = storage_0_20_[_e1125];
                            let _e2375 = storage_0_20_[_e1130];
                            let _e2378 = storage_0_20_[_e1135];
                            let _e2380 = (_e138 - bitcast<f32>(_e2372));
                            let _e2382 = (_e139 - bitcast<f32>(_e2375));
                            let _e2384 = (_e140 - bitcast<f32>(_e2378));
                            let _e2385 = abs(_e2380);
                            let _e2386 = abs(_e2382);
                            let _e2388 = select(_e2385, _e2386, (_e2386 > _e2385));
                            let _e2389 = abs(_e2384);
                            let _e2394 = storage_0_20_[_e1153];
                            let _e2400 = select(bitcast<f32>(_e2394), 1f, (bitcast<f32>(_e2394) < 1f));
                            let _e2402 = (_e2400 + -0.1f);
                            let _e2413 = storage_0_20_[_e1703];
                            let _e2419 = textureSampleCompareLevel(tex_0_12_, sampler_0_11_, vec3<f32>(_e2380, _e2382, _e2384), u32(bitcast<f32>(_e2413)), (((_e2400 / _e2402) - ((_e2400 * 0.1f) / (select(_e2388, _e2389, (_e2389 > _e2388)) * _e2402))) + -0.001f));
                            _0_u002e_sroa_u002e_0_u002e_0_ = _e2419;
                        } else {
                            let _e2425 = storage_0_20_[(_e1119 + 20u)];
                            let _e2431 = storage_0_20_[(_e1119 + 24u)];
                            let _e2437 = storage_0_20_[(_e1119 + 28u)];
                            let _e2443 = storage_0_20_[(_e1119 + 32u)];
                            let _e2455 = storage_0_20_[(_e1119 + 21u)];
                            let _e2461 = storage_0_20_[(_e1119 + 25u)];
                            let _e2467 = storage_0_20_[(_e1119 + 29u)];
                            let _e2473 = storage_0_20_[(_e1119 + 33u)];
                            let _e2485 = storage_0_20_[(_e1119 + 22u)];
                            let _e2491 = storage_0_20_[(_e1119 + 26u)];
                            let _e2497 = storage_0_20_[(_e1119 + 30u)];
                            let _e2503 = storage_0_20_[(_e1119 + 34u)];
                            let _e2515 = storage_0_20_[(_e1119 + 23u)];
                            let _e2521 = storage_0_20_[(_e1119 + 27u)];
                            let _e2527 = storage_0_20_[(_e1119 + 31u)];
                            let _e2533 = storage_0_20_[(_e1119 + 35u)];
                            let _e2540 = ((((_e138 * bitcast<f32>(_e2515)) + (_e139 * bitcast<f32>(_e2521))) + (_e140 * bitcast<f32>(_e2527))) + bitcast<f32>(_e2533));
                            let _e2553 = storage_0_20_[_e1703];
                            let _e2562 = textureSampleCompareLevel(tex_0_10_, sampler_0_11_, vec2<f32>((((((((_e138 * bitcast<f32>(_e2425)) + (_e139 * bitcast<f32>(_e2431))) + (_e140 * bitcast<f32>(_e2437))) + bitcast<f32>(_e2443)) / _e2540) * 0.5f) + 0.5f), (0.5f - ((((((_e138 * bitcast<f32>(_e2455)) + (_e139 * bitcast<f32>(_e2461))) + (_e140 * bitcast<f32>(_e2467))) + bitcast<f32>(_e2473)) / _e2540) * 0.5f))), u32(bitcast<f32>(_e2553)), ((((((_e138 * bitcast<f32>(_e2485)) + (_e139 * bitcast<f32>(_e2491))) + (_e140 * bitcast<f32>(_e2497))) + bitcast<f32>(_e2503)) / _e2540) + -0.001f));
                            _0_u002e_sroa_u002e_0_u002e_0_ = _e2562;
                        }
                    }
                }
                let _e2565 = _0_u002e_sroa_u002e_0_u002e_0_;
                shadow_vis_u002e_sroa_u002e_0_u002e_1_u002e_i = select(_e2565, _e1088, (_e1088 < _e2565));
            } else {
                shadow_vis_u002e_sroa_u002e_0_u002e_1_u002e_i = _e1088;
            }
            let _e2571 = shadow_vis_u002e_sroa_u002e_0_u002e_1_u002e_i;
            out_u002e_shadow_vis_u002e_sroa_u002e_0_u002e_1_u002e_i = _e2571;
            let _e2575 = storage_0_20_[_e1125];
            let _e2578 = storage_0_20_[_e1130];
            let _e2581 = storage_0_20_[_e1135];
            let _e2583 = (bitcast<f32>(_e2575) - _e138);
            let _e2585 = (bitcast<f32>(_e2578) - _e139);
            let _e2587 = (bitcast<f32>(_e2581) - _e140);
            let _e2590 = storage_0_20_[_e1119];
            let _e2596 = select(1f, 0f, (bitcast<f32>(_e2590) < 0.5f));
            let _e2599 = storage_0_20_[_e1209];
            let _e2602 = storage_0_20_[_e1214];
            let _e2605 = storage_0_20_[_e1219];
            let _e2606 = bitcast<f32>(_e2599);
            let _e2607 = bitcast<f32>(_e2602);
            let _e2608 = bitcast<f32>(_e2605);
            let _e2612 = (1f / length(vec3<f32>(_e2606, _e2607, _e2608)));
            let _e2619 = (1f / length(vec3<f32>(_e2583, _e2585, _e2587)));
            let _e2623 = mix((_e2612 * _e2606), (_e2583 * _e2619), _e2596);
            let _e2624 = mix((_e2612 * _e2607), (_e2585 * _e2619), _e2596);
            let _e2625 = mix((_e2612 * _e2608), (_e2587 * _e2619), _e2596);
            let _e2629 = (1f / length(vec3<f32>(_e2623, _e2624, _e2625)));
            let _e2630 = (_e2623 * _e2629);
            let _e2631 = (_e2624 * _e2629);
            let _e2632 = (_e2625 * _e2629);
            let _e2637 = ((_e195 + _e2630) + 0f);
            let _e2639 = ((_e196 + _e2631) + 0f);
            let _e2641 = ((_e197 + _e2632) + 0.000001f);
            let _e2645 = (1f / length(vec3<f32>(_e2637, _e2639, _e2641)));
            let _e2646 = (_e2645 * _e2637);
            let _e2647 = (_e2645 * _e2639);
            let _e2648 = (_e2645 * _e2641);
            let _e2651 = dot(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e2630, _e2631, _e2632));
            let _e2655 = select(_e2651, 0.000001f, (_e2651 < 0.000001f));
            if (_e2655 > 0.000001f) {
                let _e2660 = dot(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e2646, _e2647, _e2648));
                let _e2664 = select(_e2660, 0.000001f, (_e2660 < 0.000001f));
                let _e2667 = dot(vec3<f32>(_e195, _e196, _e197), vec3<f32>(_e2646, _e2647, _e2648));
                let _e2671 = select(_e2667, 0.000001f, (_e2667 < 0.000001f));
                let _e2672 = mix(_e940, _e794, _e780);
                let _e2673 = mix(_e940, _e795, _e780);
                let _e2674 = mix(_e940, _e796, _e780);
                let _e2677 = pow(0.85714287f, 5f);
                let _e2679 = (1f - _e2672);
                let _e2681 = (_e2672 + (_e2679 * _e2677));
                let _e2683 = (1f - _e2673);
                let _e2685 = (_e2673 + (_e2683 * _e2677));
                let _e2687 = (1f - _e2674);
                let _e2689 = (_e2674 + (_e2687 * _e2677));
                let _e2694 = (1f - _e2671);
                let _e2703 = ((_e2671 * pow(_e2694, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e2705 = pow(_e2694, 5f);
                let _e2727 = mix(_e940, _e794, _e780);
                let _e2728 = mix(_e940, _e795, _e780);
                let _e2729 = mix(_e940, _e796, _e780);
                let _e2732 = pow(0.85714287f, 5f);
                let _e2734 = (1f - _e2727);
                let _e2736 = (_e2727 + (_e2734 * _e2732));
                let _e2738 = (1f - _e2728);
                let _e2740 = (_e2728 + (_e2738 * _e2732));
                let _e2742 = (1f - _e2729);
                let _e2744 = (_e2729 + (_e2742 * _e2732));
                let _e2756 = ((_e2671 * pow(_e2694, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e2758 = pow(_e2694, 5f);
                let _e2781 = (_e944 * sqrt(_e951));
                let _e2784 = (_e946 * (_e944 * sqrt(_e951)));
                let _e2787 = dot(vec3<f32>(_e2646, _e2647, _e2648), vec3<f32>(_e558, _e559, _e560));
                let _e2790 = dot(vec3<f32>(_e2646, _e2647, _e2648), vec3<f32>(cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).x, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).y, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).z));
                let _e2792 = (_e2781 * _e2781);
                let _e2795 = (_e2784 * _e2784);
                let _e2798 = (_e2664 * _e2664);
                let _e2799 = (_e2798 + (((_e2787 * _e2787) / _e2792) + ((_e2790 * _e2790) / _e2795)));
                let _e2806 = (1f / (_e2799 * (((_e2781 * 3.1415927f) * _e2784) * _e2799)));
                let _e2809 = dot(vec3<f32>(_e195, _e196, _e197), vec3<f32>(_e558, _e559, _e560));
                let _e2812 = dot(vec3<f32>(_e195, _e196, _e197), vec3<f32>(cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).x, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).y, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).z));
                let _e2815 = dot(vec3<f32>(_e2630, _e2631, _e2632), vec3<f32>(_e558, _e559, _e560));
                let _e2818 = dot(vec3<f32>(_e2630, _e2631, _e2632), vec3<f32>(cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).x, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).y, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).z));
                let _e2824 = (_e2655 * _e2655);
                let _e2844 = (0.5f / max(((_e204 * sqrt(max((_e2824 + ((_e2792 * (_e2815 * _e2815)) + (_e2795 * (_e2818 * _e2818)))), 0.000001f))) + (_e2655 * sqrt(max((_e710 + ((_e2792 * (_e2809 * _e2809)) + (_e2795 * (_e2812 * _e2812)))), 0.000001f)))), 0.000001f));
                let _e2863 = (1f / length(vec3<f32>(_e957, _e958, _e959)));
                let _e2870 = (_e2630 - (_e173 * _e2655));
                let _e2871 = (_e2631 - (_e174 * _e2655));
                let _e2872 = (_e2632 - (_e175 * _e2655));
                let _e2876 = (1f / length(vec3<f32>(_e2870, _e2871, _e2872)));
                let _e2882 = dot(vec3<f32>((_e957 * _e2863), (_e958 * _e2863), (_e959 * _e2863)), vec3<f32>((_e2870 * _e2876), (_e2871 * _e2876), (_e2872 * _e2876)));
                let _e2887 = acos(max(_e204, 0f));
                let _e2890 = acos(max(_e2655, 0f));
                let _e2914 = (max((1f - dot(vec3<f32>((max(((_e2672 + (_e2679 * _e2705)) - ((_e2681 - (_e2681 * _e941)) * _e2703)), 0f) * _e782), (max(((_e2673 + (_e2683 * _e2705)) - ((_e2685 - (_e2685 * _e942)) * _e2703)), 0f) * _e782), (max(((_e2674 + (_e2687 * _e2705)) - ((_e2689 - (_e2689 * _e943)) * _e2703)), 0f) * _e782)), vec3<f32>(0.2126f, 0.7152f, 0.0722f))), 0f) * (((1f - ((_e968 * 0.5f) / (_e968 + 0.57f))) + (tan(min(_e2887, _e2890)) * ((((_e968 * 0.45f) / (_e968 + 0.09f)) * select(0f, _e2882, (_e2882 > 0f))) * sin(max(_e2887, _e2890))))) * 0.31830987f));
                let _e2927 = (_e877 * ((_e767 * ((((max(((_e2727 + (_e2734 * _e2758)) - ((_e2736 - (_e2736 * _e941)) * _e2756)), 0f) * _e782) * _e2806) * _e2844) * _e781)) + (_e767 * (((_e962 * _e791) * _e2914) * _e781))));
                let _e2928 = (_e879 * ((_e772 * ((((max(((_e2728 + (_e2738 * _e2758)) - ((_e2740 - (_e2740 * _e942)) * _e2756)), 0f) * _e782) * _e2806) * _e2844) * _e781)) + (_e772 * (((_e962 * _e792) * _e2914) * _e781))));
                let _e2929 = (_e881 * ((_e777 * ((((max(((_e2729 + (_e2742 * _e2758)) - ((_e2744 - (_e2744 * _e943)) * _e2756)), 0f) * _e782) * _e2806) * _e2844) * _e781)) + (_e777 * (((_e962 * _e793) * _e2914) * _e781))));
                if (_e884 == 0u) {
                    spec_acc_u002e_i_u002e_sroa_u002e_0_u002e_2_ = _e1102;
                    spec_acc_u002e_i_u002e_sroa_u002e_6_u002e_2_ = _e1100;
                    spec_acc_u002e_i_u002e_sroa_u002e_9_u002e_2_ = _e1098;
                    diff_acc_u002e_i_u002e_sroa_u002e_0_u002e_2_ = _e1096;
                    diff_acc_u002e_i_u002e_sroa_u002e_6_u002e_2_ = _e1094;
                    diff_acc_u002e_i_u002e_sroa_u002e_9_u002e_2_ = _e1092;
                } else {
                    let _e2936 = mix(_e940, _e794, _e780);
                    let _e2937 = mix(_e940, _e795, _e780);
                    let _e2938 = mix(_e940, _e796, _e780);
                    let _e2941 = pow(0.85714287f, 5f);
                    let _e2943 = (1f - _e2936);
                    let _e2945 = (_e2936 + (_e2943 * _e2941));
                    let _e2947 = (1f - _e2937);
                    let _e2949 = (_e2937 + (_e2947 * _e2941));
                    let _e2951 = (1f - _e2938);
                    let _e2953 = (_e2938 + (_e2951 * _e2941));
                    let _e2965 = ((_e2671 * pow(_e2694, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                    let _e2967 = pow(_e2694, 5f);
                    let _e2990 = (_e944 * sqrt(_e951));
                    let _e2993 = (_e946 * (_e944 * sqrt(_e951)));
                    let _e2996 = dot(vec3<f32>(_e2646, _e2647, _e2648), vec3<f32>(_e558, _e559, _e560));
                    let _e2999 = dot(vec3<f32>(_e2646, _e2647, _e2648), vec3<f32>(cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).x, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).y, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).z));
                    let _e3001 = (_e2990 * _e2990);
                    let _e3004 = (_e2993 * _e2993);
                    let _e3007 = (_e2798 + (((_e2996 * _e2996) / _e3001) + ((_e2999 * _e2999) / _e3004)));
                    let _e3014 = (1f / (_e3007 * (((_e2990 * 3.1415927f) * _e2993) * _e3007)));
                    let _e3017 = dot(vec3<f32>(_e195, _e196, _e197), vec3<f32>(_e558, _e559, _e560));
                    let _e3020 = dot(vec3<f32>(_e195, _e196, _e197), vec3<f32>(cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).x, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).y, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).z));
                    let _e3023 = dot(vec3<f32>(_e2630, _e2631, _e2632), vec3<f32>(_e558, _e559, _e560));
                    let _e3026 = dot(vec3<f32>(_e2630, _e2631, _e2632), vec3<f32>(cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).x, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).y, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).z));
                    let _e3051 = (0.5f / max(((_e204 * sqrt(max((_e2824 + ((_e3001 * (_e3023 * _e3023)) + (_e3004 * (_e3026 * _e3026)))), 0.000001f))) + (_e2655 * sqrt(max((_e710 + ((_e3001 * (_e3017 * _e3017)) + (_e3004 * (_e3020 * _e3020)))), 0.000001f)))), 0.000001f));
                    let _e3067 = (_e877 * (_e767 * ((((max(((_e2936 + (_e2943 * _e2967)) - ((_e2945 - (_e2945 * _e941)) * _e2965)), 0f) * _e782) * _e3014) * _e3051) * _e781)));
                    let _e3068 = (_e879 * (_e772 * ((((max(((_e2937 + (_e2947 * _e2967)) - ((_e2949 - (_e2949 * _e942)) * _e2965)), 0f) * _e782) * _e3014) * _e3051) * _e781)));
                    let _e3069 = (_e881 * (_e777 * ((((max(((_e2938 + (_e2951 * _e2967)) - ((_e2953 - (_e2953 * _e943)) * _e2965)), 0f) * _e782) * _e3014) * _e3051) * _e781)));
                    spec_acc_u002e_i_u002e_sroa_u002e_0_u002e_2_ = (_e1102 + (_e1936 * (_e2655 * _e3067)));
                    spec_acc_u002e_i_u002e_sroa_u002e_6_u002e_2_ = (_e1100 + (_e1938 * (_e2655 * _e3068)));
                    spec_acc_u002e_i_u002e_sroa_u002e_9_u002e_2_ = (_e1098 + (_e1940 * (_e2655 * _e3069)));
                    diff_acc_u002e_i_u002e_sroa_u002e_0_u002e_2_ = (_e1096 + (_e1936 * (_e2655 * (_e2927 - _e3067))));
                    diff_acc_u002e_i_u002e_sroa_u002e_6_u002e_2_ = (_e1094 + (_e1938 * (_e2655 * (_e2928 - _e3068))));
                    diff_acc_u002e_i_u002e_sroa_u002e_9_u002e_2_ = (_e1092 + (_e1940 * (_e2655 * (_e2929 - _e3069))));
                }
                let _e3098 = spec_acc_u002e_i_u002e_sroa_u002e_0_u002e_2_;
                let _e3100 = spec_acc_u002e_i_u002e_sroa_u002e_6_u002e_2_;
                let _e3102 = spec_acc_u002e_i_u002e_sroa_u002e_9_u002e_2_;
                let _e3104 = diff_acc_u002e_i_u002e_sroa_u002e_0_u002e_2_;
                let _e3106 = diff_acc_u002e_i_u002e_sroa_u002e_6_u002e_2_;
                let _e3108 = diff_acc_u002e_i_u002e_sroa_u002e_9_u002e_2_;
                if _e802 {
                    let _e3114 = (_e990 * sqrt(_e997));
                    let _e3117 = (_e992 * (_e990 * sqrt(_e997)));
                    let _e3120 = dot(vec3<f32>(_e2646, _e2647, _e2648), vec3<f32>(_e558, _e559, _e560));
                    let _e3123 = dot(vec3<f32>(_e2646, _e2647, _e2648), vec3<f32>(cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).x, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).y, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).z));
                    let _e3125 = (_e3114 * _e3114);
                    let _e3128 = (_e3117 * _e3117);
                    let _e3131 = (_e2798 + (((_e3120 * _e3120) / _e3125) + ((_e3123 * _e3123) / _e3128)));
                    let _e3141 = dot(vec3<f32>(_e195, _e196, _e197), vec3<f32>(_e558, _e559, _e560));
                    let _e3144 = dot(vec3<f32>(_e195, _e196, _e197), vec3<f32>(cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).x, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).y, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).z));
                    let _e3147 = dot(vec3<f32>(_e2630, _e2631, _e2632), vec3<f32>(_e558, _e559, _e560));
                    let _e3150 = dot(vec3<f32>(_e2630, _e2631, _e2632), vec3<f32>(cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).x, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).y, cross(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e558, _e559, _e560)).z));
                    let _e3177 = ((_e987 + ((1f - _e987) * pow(_e2694, 5f))) * ((1f / (_e3131 * (((_e3114 * 3.1415927f) * _e3117) * _e3131))) * (0.5f / max(((_e204 * sqrt(max((_e2824 + ((_e3125 * (_e3147 * _e3147)) + (_e3128 * (_e3150 * _e3150)))), 0.000001f))) + (_e2655 * sqrt(max((_e710 + ((_e3125 * (_e3141 * _e3141)) + (_e3128 * (_e3144 * _e3144)))), 0.000001f)))), 0.000001f))));
                    coat_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e3177 * bitcast<f32>(_e461)) * _e778);
                    coat_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e3177 * bitcast<f32>(_e466)) * _e778);
                    coat_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_ = ((_e3177 * bitcast<f32>(_e471)) * _e778);
                } else {
                    coat_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    coat_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    coat_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e3194 = coat_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e3196 = coat_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e3198 = coat_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e1001 <= 0f)) {
                    let _e3222 = ((_e2671 * ((1f / ((_e1006 * (((1f / _e204) + (1f / _e2655)) + -2f)) + 1f)) * (_e1006 / (pow((((_e1006 + -1f) * _e2798) + 1f), 2f) * 3.1415927f)))) / max(((_e204 * 4f) * _e2655), 0.000001f));
                    fuzz_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e3222 * bitcast<f32>(_e426)) * _e1001);
                    fuzz_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e3222 * bitcast<f32>(_e431)) * _e1001);
                    fuzz_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_ = ((_e3222 * bitcast<f32>(_e436)) * _e1001);
                } else {
                    fuzz_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    fuzz_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    fuzz_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e3239 = fuzz_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e3241 = fuzz_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e3243 = fuzz_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e1016 <= 0f)) {
                    let _e3245 = (_e944 * sqrt(_e951));
                    if !((_e1017 <= 0f)) {
                        extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (-(log(max(bitcast<f32>(_e336), 0.000001f))) / _e1017);
                        extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (-(log(max(bitcast<f32>(_e341), 0.000001f))) / _e1017);
                        extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (-(log(max(bitcast<f32>(_e346), 0.000001f))) / _e1017);
                    } else {
                        extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e3265 = extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                    let _e3267 = extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                    let _e3269 = extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                    let _e3276 = (_e3245 * _e3245);
                    let _e3281 = ((_e2798 * (_e3276 + -1f)) + 1f);
                    let _e3287 = (1f - _e3276);
                    let _e3306 = ((1f - (_e940 + ((1f - _e940) * pow(select(0.999999f, (1f - _e200), _e202), 5f)))) * ((_e3276 / (_e3281 * (_e3281 * 3.1415927f))) * (0.5f / max(((_e204 * sqrt(max((_e3276 + (_e2824 * _e3287)), 0.000001f))) + (_e2655 * sqrt(max((_e3276 + (_e710 * _e3287)), 0.000001f)))), 0.000001f))));
                    trans_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (((exp((_e1017 * -(_e3265))) * _e3306) * bitcast<f32>(_e336)) * _e1016);
                    trans_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (((exp((_e1017 * -(_e3267))) * _e3306) * bitcast<f32>(_e341)) * _e1016);
                    trans_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (((exp((_e1017 * -(_e3269))) * _e3306) * bitcast<f32>(_e346)) * _e1016);
                } else {
                    trans_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    trans_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    trans_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e3335 = trans_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e3337 = trans_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e3339 = trans_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e783 <= 0f)) {
                    let _e3344 = length(vec3<f32>((_e957 - _e2870), (_e958 - _e2871), (_e959 - _e2872)));
                    let _e3346 = sqrt(3f);
                    let _e3369 = -(_e3344);
                    let _e3377 = (max(((clamp(((_e204 * _e2655) + (((_e714 + max((1f - _e2824), 0f)) - (_e3344 * _e3344)) * 0.5f)), -1f, 1f) * bitcast<f32>(_e386)) + 1f), 0f) * 0.31830987f);
                    ss_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (((exp((((1f / max((_e1049 * bitcast<f32>(_e381)), 0.000001f)) * _e3346) * _e3369)) * _e3377) * _e797) * _e783);
                    ss_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (((exp((((1f / max((_e1049 * bitcast<f32>(_e406)), 0.000001f)) * _e3346) * _e3369)) * _e3377) * _e798) * _e783);
                    ss_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (((exp((((1f / max((_e1049 * bitcast<f32>(_e411)), 0.000001f)) * _e3346) * _e3369)) * _e3377) * _e799) * _e783);
                } else {
                    ss_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    ss_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    ss_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e3397 = ss_bsdf_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e3399 = ss_bsdf_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e3401 = ss_bsdf_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                lo_u002e_i_u002e_sroa_u002e_0_u002e_1_ = (_e1108 + (_e1936 * (_e2655 * (((_e2927 + _e3194) + (_e3239 + _e3335)) + _e3397))));
                lo_u002e_i_u002e_sroa_u002e_6_u002e_1_ = (_e1106 + (_e1938 * (_e2655 * (((_e2928 + _e3196) + (_e3241 + _e3337)) + _e3399))));
                lo_u002e_i_u002e_sroa_u002e_9_u002e_1_ = (_e1104 + (_e1940 * (_e2655 * (((_e2929 + _e3198) + (_e3243 + _e3339)) + _e3401))));
                spec_acc_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e3098;
                spec_acc_u002e_i_u002e_sroa_u002e_6_u002e_1_ = _e3100;
                spec_acc_u002e_i_u002e_sroa_u002e_9_u002e_1_ = _e3102;
                diff_acc_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e3104;
                diff_acc_u002e_i_u002e_sroa_u002e_6_u002e_1_ = _e3106;
                diff_acc_u002e_i_u002e_sroa_u002e_9_u002e_1_ = _e3108;
            } else {
                lo_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e1108;
                lo_u002e_i_u002e_sroa_u002e_6_u002e_1_ = _e1106;
                lo_u002e_i_u002e_sroa_u002e_9_u002e_1_ = _e1104;
                spec_acc_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e1102;
                spec_acc_u002e_i_u002e_sroa_u002e_6_u002e_1_ = _e1100;
                spec_acc_u002e_i_u002e_sroa_u002e_9_u002e_1_ = _e1098;
                diff_acc_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e1096;
                diff_acc_u002e_i_u002e_sroa_u002e_6_u002e_1_ = _e1094;
                diff_acc_u002e_i_u002e_sroa_u002e_9_u002e_1_ = _e1092;
            }
            let _e3442 = lo_u002e_i_u002e_sroa_u002e_0_u002e_1_;
            out_u002e_lo_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e3442;
            let _e3445 = lo_u002e_i_u002e_sroa_u002e_6_u002e_1_;
            out_u002e_lo_u002e_i_u002e_sroa_u002e_6_u002e_1_ = _e3445;
            let _e3448 = lo_u002e_i_u002e_sroa_u002e_9_u002e_1_;
            out_u002e_lo_u002e_i_u002e_sroa_u002e_9_u002e_1_ = _e3448;
            let _e3451 = spec_acc_u002e_i_u002e_sroa_u002e_0_u002e_1_;
            out_u002e_spec_acc_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e3451;
            let _e3454 = spec_acc_u002e_i_u002e_sroa_u002e_6_u002e_1_;
            out_u002e_spec_acc_u002e_i_u002e_sroa_u002e_6_u002e_1_ = _e3454;
            let _e3457 = spec_acc_u002e_i_u002e_sroa_u002e_9_u002e_1_;
            out_u002e_spec_acc_u002e_i_u002e_sroa_u002e_9_u002e_1_ = _e3457;
            let _e3460 = diff_acc_u002e_i_u002e_sroa_u002e_0_u002e_1_;
            out_u002e_diff_acc_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e3460;
            let _e3463 = diff_acc_u002e_i_u002e_sroa_u002e_6_u002e_1_;
            out_u002e_diff_acc_u002e_i_u002e_sroa_u002e_6_u002e_1_ = _e3463;
            let _e3466 = diff_acc_u002e_i_u002e_sroa_u002e_9_u002e_1_;
            out_u002e_diff_acc_u002e_i_u002e_sroa_u002e_9_u002e_1_ = _e3466;
            let _e3469 = (_e1090 + 1u);
            if (_e3469 == _e900) {
                lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e3442;
                lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e3445;
                lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e3448;
                spec_acc_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e3451;
                spec_acc_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e3454;
                spec_acc_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e3457;
                diff_acc_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e3460;
                diff_acc_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e3463;
                diff_acc_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e3466;
                shadow_vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = _e2571;
                break;
            } else {
                shadow_vis_u002e_sroa_u002e_0_u002e_0_ = _e2571;
                k_u002e_sroa_u002e_0_u002e_0_ = _e3469;
                diff_acc_u002e_i_u002e_sroa_u002e_9_ = _e3466;
                diff_acc_u002e_i_u002e_sroa_u002e_6_ = _e3463;
                diff_acc_u002e_i_u002e_sroa_u002e_0_ = _e3460;
                spec_acc_u002e_i_u002e_sroa_u002e_9_ = _e3457;
                spec_acc_u002e_i_u002e_sroa_u002e_6_ = _e3454;
                spec_acc_u002e_i_u002e_sroa_u002e_0_ = _e3451;
                lo_u002e_i_u002e_sroa_u002e_9_ = _e3448;
                lo_u002e_i_u002e_sroa_u002e_6_ = _e3445;
                lo_u002e_i_u002e_sroa_u002e_0_ = _e3442;
                continue;
            }
        }
    }
    let _e3513 = lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
    let _e3515 = lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
    let _e3517 = lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
    let _e3519 = spec_acc_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
    let _e3521 = spec_acc_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
    let _e3523 = spec_acc_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
    let _e3525 = diff_acc_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
    let _e3527 = diff_acc_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
    let _e3529 = diff_acc_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
    let _e3531 = shadow_vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa;
    let _e3541 = uniform_struct_0_1_.f0_.x;
    let _e3545 = uniform_struct_0_1_.f0_.y;
    let _e3549 = uniform_struct_0_1_.f0_.z;
    let _e3553 = (_e691.x * (mix(_e791, (_e791 * _e782), _e780) * _e3541));
    let _e3554 = (_e691.x * (mix(_e792, (_e792 * _e782), _e780) * _e3545));
    let _e3555 = (_e691.x * (mix(_e793, (_e793 * _e782), _e780) * _e3549));
    let _e3556 = bitcast<f32>(_e496);
    if !((_e3556 <= 0f)) {
        let _e3563 = (_e3556 * bitcast<f32>(_e501));
        let _e3564 = (_e3556 * bitcast<f32>(_e506));
        let _e3565 = (_e3556 * bitcast<f32>(_e511));
        let _e3575 = ((1f - _e778) + ((1f - pow((1f - _e204), 5f)) * _e778));
        emission_u002e_i_u002e_sroa_u002e_0_u002e_1_ = (_e3563 + ((((_e3563 * bitcast<f32>(_e461)) * _e3575) - _e3563) * _e778));
        emission_u002e_i_u002e_sroa_u002e_8_u002e_1_ = (_e3564 + ((((_e3564 * bitcast<f32>(_e466)) * _e3575) - _e3564) * _e778));
        emission_u002e_i_u002e_sroa_u002e_11_u002e_1_ = (_e3565 + ((((_e3565 * bitcast<f32>(_e471)) * _e3575) - _e3565) * _e778));
    } else {
        emission_u002e_i_u002e_sroa_u002e_0_u002e_1_ = 0f;
        emission_u002e_i_u002e_sroa_u002e_8_u002e_1_ = 0f;
        emission_u002e_i_u002e_sroa_u002e_11_u002e_1_ = 0f;
    }
    let _e3604 = emission_u002e_i_u002e_sroa_u002e_0_u002e_1_;
    let _e3606 = emission_u002e_i_u002e_sroa_u002e_8_u002e_1_;
    let _e3608 = emission_u002e_i_u002e_sroa_u002e_11_u002e_1_;
    let _e3609 = bitcast<f32>(_e291);
    let _e3614 = ((_e3609 + -1f) / (_e3609 + 1f));
    let _e3615 = (_e3614 * _e3614);
    let _e3623 = (dot(vec3<f32>(_e173, _e174, _e175), vec3<f32>(_e195, _e196, _e197)) * 2f);
    let _e3627 = ((_e173 * _e3623) - _e195);
    let _e3628 = ((_e174 * _e3623) - _e196);
    let _e3629 = ((_e175 * _e3623) - _e197);
    let _e3632 = uniform_struct_0_1_.f4_;
    let _e3633 = (_e283 * _e3632);
    let _e3637 = textureSampleLevel(tex_0_13_, sampler_0_16_, vec3<f32>(_e3627, _e3628, _e3629), _e3633);
    let _e3642 = textureSampleLevel(tex_0_13_, sampler_0_16_, vec3<f32>(_e3627, _e3628, _e3629), _e3633);
    let _e3647 = textureSampleLevel(tex_0_13_, sampler_0_16_, vec3<f32>(_e3627, _e3628, _e3629), _e3633);
    let _e3652 = textureSample(tex_0_14_, sampler_0_16_, vec3<f32>(_e173, _e174, _e175));
    let _e3657 = textureSample(tex_0_14_, sampler_0_16_, vec3<f32>(_e173, _e174, _e175));
    let _e3662 = textureSample(tex_0_14_, sampler_0_16_, vec3<f32>(_e173, _e174, _e175));
    let _e3667 = textureSample(tex_0_15_, sampler_0_16_, vec2<f32>(_e204, _e283));
    let _e3672 = textureSample(tex_0_15_, sampler_0_16_, vec2<f32>(_e204, _e283));
    let _e3676 = uniform_struct_0_1_.f3_;
    let _e3678 = (1f - _e780);
    let _e3688 = (_e3637.x * ((mix(_e3615, _e794, _e780) * _e3667.x) + _e3672.y));
    let _e3689 = (_e3642.y * ((mix(_e3615, _e795, _e780) * _e3667.x) + _e3672.y));
    let _e3690 = (_e3647.z * ((mix(_e3615, _e796, _e780) * _e3667.x) + _e3672.y));
    let _e3694 = max((1f - _e3672.y), 0f);
    let _e3707 = (_e691.x * (_e3676 * (_e3688 + (((_e3678 * _e791) * _e3652.x) * _e3694))));
    let _e3708 = (_e691.x * (_e3676 * (_e3689 + (((_e3678 * _e792) * _e3657.y) * _e3694))));
    let _e3709 = (_e691.x * (_e3676 * (_e3690 + (((_e3678 * _e793) * _e3662.z) * _e3694))));
    let _e3722 = !((bitcast<f32>(_e526) >= 0.5f));
    let _e3723 = select(_e3604, (((_e3513 + _e3553) + _e3604) + _e3707), _e3722);
    let _e3724 = select(_e3606, (((_e3515 + _e3554) + _e3606) + _e3708), _e3722);
    let _e3725 = select(_e3608, (((_e3517 + _e3555) + _e3608) + _e3709), _e3722);
    let _e3729 = ((_e3723 <= _e3723) && (0f <= 0f));
    if _e3729 {
        if (abs(_e3723) < 100000000000000000000f) {
            _2_u002e_sroa_u002e_0_u002e_0_u002e_i = _e3723;
        } else {
            _2_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        _2_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e3739 = _2_u002e_sroa_u002e_0_u002e_0_u002e_i;
    let _e3743 = ((_e3724 <= _e3724) && (0f <= 0f));
    if _e3743 {
        if (abs(_e3724) < 100000000000000000000f) {
            _3_u002e_sroa_u002e_0_u002e_0_u002e_i = _e3724;
        } else {
            _3_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        _3_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e3753 = _3_u002e_sroa_u002e_0_u002e_0_u002e_i;
    let _e3757 = ((_e3725 <= _e3725) && (0f <= 0f));
    if _e3757 {
        if (abs(_e3725) < 100000000000000000000f) {
            _4_u002e_sroa_u002e_0_u002e_0_u002e_i = _e3725;
        } else {
            _4_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        _4_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e3767 = _4_u002e_sroa_u002e_0_u002e_0_u002e_i;
    if _e3729 {
        if !((abs(_e3723) < 100000000000000000000f)) {
            x_bad_u002e_sroa_u002e_0_u002e_0_u002e_i = 1f;
        } else {
            x_bad_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        x_bad_u002e_sroa_u002e_0_u002e_0_u002e_i = 1f;
    }
    let _e3779 = x_bad_u002e_sroa_u002e_0_u002e_0_u002e_i;
    if _e3743 {
        if !((abs(_e3724) < 100000000000000000000f)) {
            y_bad_u002e_sroa_u002e_0_u002e_0_u002e_i = 1f;
        } else {
            y_bad_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        y_bad_u002e_sroa_u002e_0_u002e_0_u002e_i = 1f;
    }
    let _e3791 = y_bad_u002e_sroa_u002e_0_u002e_0_u002e_i;
    if _e3757 {
    } else {
        z_bad_u002e_sroa_u002e_0_u002e_0_u002e_i = 1f;
        let _e3795 = z_bad_u002e_sroa_u002e_0_u002e_0_u002e_i;
        let _e3800 = select(0f, 1f, (_e884 == 7u));
        let _e3820 = select(0f, 1f, (_e884 == 3u));
        let _e3825 = select(0f, 1f, (_e884 == 2u));
        let _e3830 = select(0f, 1f, (_e884 == 1u));
        let _e3835 = select(0f, 1f, (_e884 == 0u));
        let _e3837 = (_e3676 * _e3690);
        let _e3841 = (_e3676 * _e3689);
        let _e3845 = (_e3676 * _e3688);
        let _e3850 = select(_e3795, _e3791, (_e3791 > _e3795));
        let _e3852 = select(_e3850, _e3779, (_e3779 > _e3850));
        let _e3863 = select(0f, 1f, (dot(vec3<f32>(_e3739, _e3753, _e3767), vec3<f32>(0.2126f, 0.7152f, 0.0722f)) > 1f));
        let _e3897 = (_e283 * select(0f, 1f, (_e884 == 4u)));
        let _e3901 = (select(0f, 1f, (_e884 == 5u)) * _e780);
        let _e3905 = (select(0f, 1f, (_e884 == 6u)) * _e3531);
        fragment_out = vec4<f32>(((_e3905 + (_e3901 + (_e3897 + ((_e3820 * ((_e3525 + _e3553) + (_e3707 - _e3845))) + ((_e3825 * _e3845) + ((_e3830 * _e3519) + (_e3835 * _e3739))))))) + (_e3800 * mix(mix(_e3739, 1f, _e3863), 1f, _e3852))), ((_e3905 + (_e3901 + (_e3897 + ((_e3820 * ((_e3527 + _e3554) + (_e3708 - _e3841))) + ((_e3825 * _e3841) + ((_e3830 * _e3521) + (_e3835 * _e3753))))))) + (_e3800 * mix(mix(_e3753, 0f, _e3863), 0f, _e3852))), ((_e3905 + (_e3901 + (_e3897 + ((_e3820 * ((_e3529 + _e3555) + (_e3709 - _e3837))) + ((_e3825 * _e3837) + ((_e3830 * _e3523) + (_e3835 * _e3767))))))) + (_e3800 * mix(mix(_e3767, 0f, _e3863), 1f, _e3852))), bitcast<f32>(_e516));
        let _e3918 = fragment_out;
        return _e3918;
    }
    if !((abs(_e3725) < 100000000000000000000f)) {
        z_bad_u002e_sroa_u002e_0_u002e_0_u002e_i = 1f;
        let _e3926 = z_bad_u002e_sroa_u002e_0_u002e_0_u002e_i;
        let _e3931 = select(0f, 1f, (_e884 == 7u));
        let _e3951 = select(0f, 1f, (_e884 == 3u));
        let _e3956 = select(0f, 1f, (_e884 == 2u));
        let _e3961 = select(0f, 1f, (_e884 == 1u));
        let _e3966 = select(0f, 1f, (_e884 == 0u));
        let _e3968 = (_e3676 * _e3690);
        let _e3972 = (_e3676 * _e3689);
        let _e3976 = (_e3676 * _e3688);
        let _e3981 = select(_e3926, _e3791, (_e3791 > _e3926));
        let _e3983 = select(_e3981, _e3779, (_e3779 > _e3981));
        let _e3994 = select(0f, 1f, (dot(vec3<f32>(_e3739, _e3753, _e3767), vec3<f32>(0.2126f, 0.7152f, 0.0722f)) > 1f));
        let _e4028 = (_e283 * select(0f, 1f, (_e884 == 4u)));
        let _e4032 = (select(0f, 1f, (_e884 == 5u)) * _e780);
        let _e4036 = (select(0f, 1f, (_e884 == 6u)) * _e3531);
        fragment_out = vec4<f32>(((_e4036 + (_e4032 + (_e4028 + ((_e3951 * ((_e3525 + _e3553) + (_e3707 - _e3976))) + ((_e3956 * _e3976) + ((_e3961 * _e3519) + (_e3966 * _e3739))))))) + (_e3931 * mix(mix(_e3739, 1f, _e3994), 1f, _e3983))), ((_e4036 + (_e4032 + (_e4028 + ((_e3951 * ((_e3527 + _e3554) + (_e3708 - _e3972))) + ((_e3956 * _e3972) + ((_e3961 * _e3521) + (_e3966 * _e3753))))))) + (_e3931 * mix(mix(_e3753, 0f, _e3994), 0f, _e3983))), ((_e4036 + (_e4032 + (_e4028 + ((_e3951 * ((_e3529 + _e3555) + (_e3709 - _e3968))) + ((_e3956 * _e3968) + ((_e3961 * _e3523) + (_e3966 * _e3767))))))) + (_e3931 * mix(mix(_e3767, 0f, _e3994), 1f, _e3983))), bitcast<f32>(_e516));
        let _e4049 = fragment_out;
        return _e4049;
    }
    z_bad_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    let _e4053 = z_bad_u002e_sroa_u002e_0_u002e_0_u002e_i;
    let _e4058 = select(0f, 1f, (_e884 == 7u));
    let _e4078 = select(0f, 1f, (_e884 == 3u));
    let _e4083 = select(0f, 1f, (_e884 == 2u));
    let _e4088 = select(0f, 1f, (_e884 == 1u));
    let _e4093 = select(0f, 1f, (_e884 == 0u));
    let _e4095 = (_e3676 * _e3690);
    let _e4099 = (_e3676 * _e3689);
    let _e4103 = (_e3676 * _e3688);
    let _e4108 = select(_e4053, _e3791, (_e3791 > _e4053));
    let _e4110 = select(_e4108, _e3779, (_e3779 > _e4108));
    let _e4121 = select(0f, 1f, (dot(vec3<f32>(_e3739, _e3753, _e3767), vec3<f32>(0.2126f, 0.7152f, 0.0722f)) > 1f));
    let _e4155 = (_e283 * select(0f, 1f, (_e884 == 4u)));
    let _e4159 = (select(0f, 1f, (_e884 == 5u)) * _e780);
    let _e4163 = (select(0f, 1f, (_e884 == 6u)) * _e3531);
    fragment_out = vec4<f32>(((_e4163 + (_e4159 + (_e4155 + ((_e4078 * ((_e3525 + _e3553) + (_e3707 - _e4103))) + ((_e4083 * _e4103) + ((_e4088 * _e3519) + (_e4093 * _e3739))))))) + (_e4058 * mix(mix(_e3739, 1f, _e4121), 1f, _e4110))), ((_e4163 + (_e4159 + (_e4155 + ((_e4078 * ((_e3527 + _e3554) + (_e3708 - _e4099))) + ((_e4083 * _e4099) + ((_e4088 * _e3521) + (_e4093 * _e3753))))))) + (_e4058 * mix(mix(_e3753, 0f, _e4121), 0f, _e4110))), ((_e4163 + (_e4159 + (_e4155 + ((_e4078 * ((_e3529 + _e3555) + (_e3709 - _e4095))) + ((_e4083 * _e4095) + ((_e4088 * _e3523) + (_e4093 * _e3767))))))) + (_e4058 * mix(mix(_e3767, 0f, _e4121), 1f, _e4110))), bitcast<f32>(_e516));
    let _e4176 = fragment_out;
    return _e4176;
}
