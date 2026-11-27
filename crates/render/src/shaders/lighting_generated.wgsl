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
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_in: u32;
    var radiance_u002e_sroa_u002e_0_u002e_2_: f32;
    var radiance_u002e_sroa_u002e_12_u002e_2_: f32;
    var radiance_u002e_sroa_u002e_20_u002e_2_: f32;
    var vis_u002e_sroa_u002e_0_u002e_0_: f32;
    var _0_u002e_sroa_u002e_0_u002e_0_: f32;
    var extinction_u002e_sroa_u002e_0_u002e_0_: f32;
    var extinction_u002e_sroa_u002e_5_u002e_0_: f32;
    var extinction_u002e_sroa_u002e_8_u002e_0_: f32;
    var emission_u002e_sroa_u002e_0_u002e_1_: f32;
    var emission_u002e_sroa_u002e_8_u002e_1_: f32;
    var emission_u002e_sroa_u002e_11_u002e_1_: f32;
    var ss_bsdf_u002e_sroa_u002e_0_u002e_0_: f32;
    var ss_bsdf_u002e_sroa_u002e_5_u002e_0_: f32;
    var ss_bsdf_u002e_sroa_u002e_8_u002e_0_: f32;
    var trans_bsdf_u002e_sroa_u002e_0_u002e_0_: f32;
    var trans_bsdf_u002e_sroa_u002e_5_u002e_0_: f32;
    var trans_bsdf_u002e_sroa_u002e_8_u002e_0_: f32;
    var _4_u002e_sroa_u002e_0_u002e_0_: f32;
    var coat_bsdf_u002e_sroa_u002e_0_u002e_0_: f32;
    var coat_bsdf_u002e_sroa_u002e_5_u002e_0_: f32;
    var coat_bsdf_u002e_sroa_u002e_8_u002e_0_: f32;
    var fuzz_bsdf_u002e_sroa_u002e_0_u002e_0_: f32;
    var fuzz_bsdf_u002e_sroa_u002e_5_u002e_0_: f32;
    var fuzz_bsdf_u002e_sroa_u002e_8_u002e_0_: f32;
    var z_bad_u002e_sroa_u002e_0_u002e_0_: f32;
    var _15_u002e_sroa_u002e_0_u002e_0_: f32;
    var _15_u002e_sroa_u002e_5_u002e_0_: f32;
    var _15_u002e_sroa_u002e_8_u002e_0_: f32;
    var radiance_u002e_sroa_u002e_0_u002e_1_: f32;
    var radiance_u002e_sroa_u002e_12_u002e_1_: f32;
    var radiance_u002e_sroa_u002e_20_u002e_1_: f32;
    var shadow_vis_u002e_sroa_u002e_0_u002e_1_: f32;
    var _2_u002e_sroa_u002e_0_u002e_0_: f32;
    var x_bad_u002e_sroa_u002e_0_u002e_0_: f32;
    var y_bad_u002e_sroa_u002e_0_u002e_0_: f32;
    var spec_acc_u002e_sroa_u002e_0_u002e_2_: f32;
    var spec_acc_u002e_sroa_u002e_6_u002e_2_: f32;
    var spec_acc_u002e_sroa_u002e_9_u002e_2_: f32;
    var diff_acc_u002e_sroa_u002e_0_u002e_2_: f32;
    var diff_acc_u002e_sroa_u002e_6_u002e_2_: f32;
    var diff_acc_u002e_sroa_u002e_9_u002e_2_: f32;
    var shadow_vis_u002e_sroa_u002e_0_u002e_0_: f32;
    var k_u002e_sroa_u002e_0_u002e_0_: u32;
    var diff_acc_u002e_sroa_u002e_9_: f32;
    var diff_acc_u002e_sroa_u002e_6_: f32;
    var diff_acc_u002e_sroa_u002e_0_: f32;
    var spec_acc_u002e_sroa_u002e_9_: f32;
    var spec_acc_u002e_sroa_u002e_6_: f32;
    var spec_acc_u002e_sroa_u002e_0_: f32;
    var lo_u002e_sroa_u002e_9_: f32;
    var lo_u002e_sroa_u002e_6_: f32;
    var lo_u002e_sroa_u002e_0_: f32;
    var lo_u002e_sroa_u002e_0_u002e_1_: f32;
    var lo_u002e_sroa_u002e_6_u002e_1_: f32;
    var lo_u002e_sroa_u002e_9_u002e_1_: f32;
    var spec_acc_u002e_sroa_u002e_0_u002e_1_: f32;
    var spec_acc_u002e_sroa_u002e_6_u002e_1_: f32;
    var spec_acc_u002e_sroa_u002e_9_u002e_1_: f32;
    var diff_acc_u002e_sroa_u002e_0_u002e_1_: f32;
    var diff_acc_u002e_sroa_u002e_6_u002e_1_: f32;
    var diff_acc_u002e_sroa_u002e_9_u002e_1_: f32;
    var lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa: f32;
    var lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa: f32;
    var spec_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var spec_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa: f32;
    var spec_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa: f32;
    var diff_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var diff_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa: f32;
    var diff_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa: f32;
    var shadow_vis_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var radiance_u002e_sroa_u002e_0_u002e_0_: f32;
    var radiance_u002e_sroa_u002e_12_u002e_0_: f32;
    var radiance_u002e_sroa_u002e_20_u002e_0_: f32;
    var _3_u002e_sroa_u002e_0_u002e_0_: f32;
    var out_u002e_lo_u002e_sroa_u002e_0_u002e_1_: f32;
    var out_u002e_lo_u002e_sroa_u002e_6_u002e_1_: f32;
    var out_u002e_lo_u002e_sroa_u002e_9_u002e_1_: f32;
    var out_u002e_spec_acc_u002e_sroa_u002e_0_u002e_1_: f32;
    var out_u002e_spec_acc_u002e_sroa_u002e_6_u002e_1_: f32;
    var out_u002e_spec_acc_u002e_sroa_u002e_9_u002e_1_: f32;
    var out_u002e_diff_acc_u002e_sroa_u002e_0_u002e_1_: f32;
    var out_u002e_diff_acc_u002e_sroa_u002e_6_u002e_1_: f32;
    var out_u002e_diff_acc_u002e_sroa_u002e_9_u002e_1_: f32;
    var out_u002e_shadow_vis_u002e_sroa_u002e_0_u002e_1_: f32;

    let _e5 = textureDimensions(tex_0_8_);
    let _e11 = u32((uv.x * f32(bitcast<i32>(_e5.x))));
    let _e15 = u32((uv.y * f32(bitcast<i32>(_e5.y))));
    let _e20 = textureLoad(tex_0_8_, vec2<i32>(bitcast<i32>(_e11), bitcast<i32>(_e15)), 0i);
    let _e21 = vec2<f32>(uv.x, uv.y);
    let _e25 = textureSampleLevel(tex_0_3_, sampler_0_9_, _e21, 0f);
    let _e29 = textureSampleLevel(tex_0_4_, sampler_0_9_, _e21, 0f);
    let _e36 = ((uv.x * 2f) + -1f);
    let _e40 = (1f - (uv.y * 2f));
    let _e44 = uniform_struct_0_0_.f4_.w;
    let _e48 = uniform_struct_0_0_.f5_.w;
    let _e52 = uniform_struct_0_0_.f6_.w;
    let _e56 = uniform_struct_0_0_.f7_.w;
    let _e62 = (_e56 + (((_e36 * _e44) + (_e40 * _e48)) + (_e20 * _e52)));
    let _e66 = uniform_struct_0_0_.f4_.x;
    let _e70 = uniform_struct_0_0_.f5_.x;
    let _e74 = uniform_struct_0_0_.f6_.x;
    let _e78 = uniform_struct_0_0_.f7_.x;
    let _e88 = uniform_struct_0_0_.f4_.y;
    let _e92 = uniform_struct_0_0_.f5_.y;
    let _e96 = uniform_struct_0_0_.f6_.y;
    let _e100 = uniform_struct_0_0_.f7_.y;
    let _e110 = uniform_struct_0_0_.f4_.z;
    let _e114 = uniform_struct_0_0_.f5_.z;
    let _e118 = uniform_struct_0_0_.f6_.z;
    let _e122 = uniform_struct_0_0_.f7_.z;
    let _e129 = ((_e78 + (((_e36 * _e66) + (_e40 * _e70)) + (_e20 * _e74))) / _e62);
    let _e130 = ((_e100 + (((_e36 * _e88) + (_e40 * _e92)) + (_e20 * _e96))) / _e62);
    let _e131 = ((_e122 + (((_e36 * _e110) + (_e40 * _e114)) + (_e20 * _e118))) / _e62);
    let _e137 = textureLoad(tex_0_5_, vec2<i32>(bitcast<i32>(_e11), bitcast<i32>(_e15)), 0i);
    let _e143 = ((1f - abs(_e29.x)) - abs(_e29.y));
    let _e148 = select(0f, -(_e143), (_e143 < 0f));
    let _e152 = -(_e148);
    let _e158 = (_e29.x + select(_e152, _e148, !((_e29.x >= 0f))));
    let _e159 = (_e29.y + select(_e152, _e148, !((_e29.y >= 0f))));
    let _e163 = (1f / length(vec3<f32>(_e158, _e159, _e143)));
    let _e164 = (_e163 * _e158);
    let _e165 = (_e163 * _e159);
    let _e166 = (_e143 * _e163);
    let _e170 = uniform_struct_0_0_.f8_.x;
    let _e174 = uniform_struct_0_0_.f8_.y;
    let _e178 = uniform_struct_0_0_.f8_.z;
    let _e179 = (_e170 - _e129);
    let _e180 = (_e174 - _e130);
    let _e181 = (_e178 - _e131);
    let _e185 = (1f / length(vec3<f32>(_e179, _e180, _e181)));
    let _e186 = (_e179 * _e185);
    let _e187 = (_e180 * _e185);
    let _e188 = (_e181 * _e185);
    let _e191 = dot(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e186, _e187, _e188));
    let _e193 = (_e191 > 0.000001f);
    let _e195 = select(0.000001f, _e191, _e193);
    let _e197 = (_e137.x * 80u);
    let _e200 = storage_0_2_[_e197];
    let _e205 = storage_0_2_[(_e197 | 1u)];
    let _e210 = storage_0_2_[(_e197 | 2u)];
    let _e215 = storage_0_2_[(_e197 | 4u)];
    let _e220 = storage_0_2_[(_e197 | 5u)];
    let _e225 = storage_0_2_[(_e197 | 6u)];
    let _e230 = storage_0_2_[(_e197 | 8u)];
    let _e232 = (_e197 | 9u);
    let _e235 = storage_0_2_[_e232];
    let _e236 = bitcast<f32>(_e235);
    let _e237 = dpdx(_e164);
    let _e238 = dpdx(_e165);
    let _e239 = dpdx(_e166);
    let _e240 = dpdy(_e164);
    let _e241 = dpdy(_e165);
    let _e242 = dpdy(_e166);
    let _e253 = (((dot(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e237, _e238, _e239)) + dot(vec3<f32>(_e240, _e241, _e242), vec3<f32>(_e240, _e241, _e242))) * 0.15f) * 2f);
    let _e258 = (_e236 * _e236);
    let _e260 = ((_e258 * _e258) + select(0.2f, _e253, (_e253 < 0.2f)));
    let _e264 = select(_e260, 0f, (_e260 < 0f));
    let _e268 = select(_e264, 1f, (_e264 > 1f));
    let _e274 = sqrt(sqrt(select(0f, _e268, (_e268 > 0f))));
    let _e282 = storage_0_2_[(_e197 | 10u)];
    let _e287 = storage_0_2_[(_e197 | 11u)];
    let _e292 = storage_0_2_[(_e197 | 12u)];
    let _e297 = storage_0_2_[(_e197 | 13u)];
    let _e302 = storage_0_2_[(_e197 | 14u)];
    let _e307 = storage_0_2_[(_e197 + 16u)];
    let _e312 = storage_0_2_[(_e197 + 17u)];
    let _e327 = storage_0_2_[(_e197 + 20u)];
    let _e332 = storage_0_2_[(_e197 + 21u)];
    let _e337 = storage_0_2_[(_e197 + 22u)];
    let _e362 = storage_0_2_[(_e197 + 28u)];
    let _e367 = storage_0_2_[(_e197 + 29u)];
    let _e372 = storage_0_2_[(_e197 + 30u)];
    let _e377 = storage_0_2_[(_e197 + 31u)];
    let _e382 = storage_0_2_[(_e197 + 32u)];
    let _e387 = storage_0_2_[(_e197 + 33u)];
    let _e392 = storage_0_2_[(_e197 + 34u)];
    let _e397 = storage_0_2_[(_e197 + 36u)];
    let _e402 = storage_0_2_[(_e197 + 37u)];
    let _e407 = storage_0_2_[(_e197 + 40u)];
    let _e412 = storage_0_2_[(_e197 + 41u)];
    let _e417 = storage_0_2_[(_e197 + 44u)];
    let _e422 = storage_0_2_[(_e197 + 45u)];
    let _e427 = storage_0_2_[(_e197 + 46u)];
    let _e432 = storage_0_2_[(_e197 + 48u)];
    let _e437 = storage_0_2_[(_e197 + 49u)];
    let _e442 = storage_0_2_[(_e197 + 50u)];
    let _e447 = storage_0_2_[(_e197 + 51u)];
    let _e452 = storage_0_2_[(_e197 + 52u)];
    let _e457 = storage_0_2_[(_e197 + 53u)];
    let _e462 = storage_0_2_[(_e197 + 54u)];
    let _e467 = storage_0_2_[(_e197 + 56u)];
    let _e472 = storage_0_2_[(_e197 + 60u)];
    let _e477 = storage_0_2_[(_e197 + 61u)];
    let _e482 = storage_0_2_[(_e197 + 62u)];
    let _e487 = storage_0_2_[(_e197 + 64u)];
    let _e492 = storage_0_2_[(_e197 + 68u)];
    let _e497 = storage_0_2_[(_e197 + 69u)];
    let _e502 = storage_0_2_[(_e197 + 70u)];
    let _e507 = storage_0_2_[(_e197 + 72u)];
    let _e517 = storage_0_2_[(_e197 + 74u)];
    let _e540 = (cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(0f, 1f, 0f)).x + 0f);
    let _e542 = (cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(0f, 1f, 0f)).y + 0f);
    let _e544 = (cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(0f, 1f, 0f)).z + 0.000001f);
    let _e548 = (1f / length(vec3<f32>(_e540, _e542, _e544)));
    let _e549 = (_e540 * _e548);
    let _e550 = (_e542 * _e548);
    let _e551 = (_e544 * _e548);
    let _e567 = uniform_struct_0_0_.f0_.x;
    let _e571 = uniform_struct_0_0_.f1_.x;
    let _e575 = uniform_struct_0_0_.f2_.x;
    let _e579 = uniform_struct_0_0_.f3_.x;
    let _e589 = uniform_struct_0_0_.f0_.y;
    let _e593 = uniform_struct_0_0_.f1_.y;
    let _e597 = uniform_struct_0_0_.f2_.y;
    let _e601 = uniform_struct_0_0_.f3_.y;
    let _e611 = uniform_struct_0_0_.f0_.w;
    let _e615 = uniform_struct_0_0_.f1_.w;
    let _e619 = uniform_struct_0_0_.f2_.w;
    let _e623 = uniform_struct_0_0_.f3_.w;
    let _e629 = (_e623 + (((_e129 * _e611) + (_e130 * _e615)) + (_e131 * _e619)));
    let _e633 = select(0.5f, _e629, (_e629 > 0.5f));
    let _e674 = ((((min(u32(floor((((((_e601 + (((_e129 * _e589) + (_e130 * _e593)) + (_e131 * _e597))) / _e633) * 0.5f) + 0.5f) * 9f))), 8u) << 4u) | min(u32(floor((((((_e579 + (((_e129 * _e567) + (_e130 * _e571)) + (_e131 * _e575))) / _e633) * 0.5f) + 0.5f) * 16f))), 15u)) + (min(u32(ceil(((log((_e633 * 2f)) * 3.4407384f) + -0.0001f))), 23u) * 144u)) << 1u);
    let _e677 = uniform_struct_0_19_.f2_;
    let _e680 = textureSampleLevel(tex_0_17_, sampler_0_9_, _e21, 0f);
    let _e684 = textureSampleLevel(tex_0_6_, sampler_0_9_, _e21, 0f);
    let _e688 = textureSampleLevel(tex_0_7_, sampler_0_9_, _e21, 0f);
    let _e691 = bitcast<f32>(_e482);
    let _e695 = (_e195 * _e195);
    let _e699 = max((1f - _e695), 0f);
    let _e701 = (sqrt(_e699) / _e691);
    let _e714 = pow(((_e691 + -1f) / (_e691 + 1f)), 2f);
    let _e718 = (((bitcast<f32>(_e477) * 1000f) * (_e691 * 12.566371f)) * sqrt(max((1f - (_e701 * _e701)), 0f)));
    let _e729 = (_e714 * 2f);
    let _e735 = (1f - (_e714 * _e714));
    let _e747 = clamp(bitcast<f32>(_e472), 0f, 1f);
    let _e752 = ((_e747 * ((((_e729 * cos((_e718 / 650f))) / _e735) + 1f) + -1f)) + 1f);
    let _e757 = ((_e747 * ((((_e729 * cos((_e718 / 550f))) / _e735) + 1f) + -1f)) + 1f);
    let _e762 = ((_e747 * ((((_e729 * cos((_e718 / 450f))) / _e735) + 1f) + -1f)) + 1f);
    let _e763 = bitcast<f32>(_e432);
    let _e764 = bitcast<f32>(_e467);
    let _e765 = bitcast<f32>(_e210);
    let _e766 = bitcast<f32>(_e200);
    let _e767 = bitcast<f32>(_e230);
    let _e768 = bitcast<f32>(_e362);
    let _e773 = ((_e764 + -1f) / (_e764 + 1f));
    let _e776 = bitcast<f32>(_e215);
    let _e777 = bitcast<f32>(_e220);
    let _e778 = bitcast<f32>(_e225);
    let _e779 = (_e766 * _e776);
    let _e780 = (_e766 * _e777);
    let _e781 = (_e766 * _e778);
    let _e782 = bitcast<f32>(_e382);
    let _e783 = bitcast<f32>(_e387);
    let _e784 = bitcast<f32>(_e392);
    let _e787 = !((_e763 <= 0f));
    if _e787 {
        let _e793 = (1f - ((1f - pow(_e773, 2f)) / (_e764 * _e764)));
        let _e795 = (1f - _e793);
        let _e796 = (_e781 * _e767);
        let _e809 = (_e780 * _e767);
        let _e822 = (_e779 * _e767);
        let _e836 = (_e763 * bitcast<f32>(_e447));
        _15_u002e_sroa_u002e_0_u002e_0_ = ((_e836 * ((_e795 / max((1f - ((_e822 + (((((_e776 - _e782) * _e768) + _e782) - _e822) * _e765)) * _e793)), 0.000001f)) + -1f)) + 1f);
        _15_u002e_sroa_u002e_5_u002e_0_ = ((_e836 * ((_e795 / max((1f - ((_e809 + (((((_e777 - _e783) * _e768) + _e783) - _e809) * _e765)) * _e793)), 0.000001f)) + -1f)) + 1f);
        _15_u002e_sroa_u002e_8_u002e_0_ = ((_e836 * ((_e795 / max((1f - ((_e796 + (((((_e778 - _e784) * _e768) + _e784) - _e796) * _e765)) * _e793)), 0.000001f)) + -1f)) + 1f);
    } else {
        _15_u002e_sroa_u002e_0_u002e_0_ = 1f;
        _15_u002e_sroa_u002e_5_u002e_0_ = 1f;
        _15_u002e_sroa_u002e_8_u002e_0_ = 1f;
    }
    let _e862 = _15_u002e_sroa_u002e_0_u002e_0_;
    let _e864 = _15_u002e_sroa_u002e_5_u002e_0_;
    let _e866 = _15_u002e_sroa_u002e_8_u002e_0_;
    let _e869 = uniform_struct_0_1_.f5_;
    let _e872 = uniform_struct_0_1_.f2_;
    let _e875 = storage_0_22_[_e674];
    let _e881 = storage_0_22_[(_e674 | 1u)];
    if (_e25.w < 0.001f) {
        discard;
    }
    let _e885 = ((_e881 * _e677) + _e872);
    if (_e885 == 0u) {
        lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
        lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = 0f;
        lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = 0f;
        spec_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
        spec_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = 0f;
        spec_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = 0f;
        diff_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
        diff_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = 0f;
        diff_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = 0f;
        shadow_vis_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 1f;
    } else {
        let _e917 = bitcast<f32>(_e205);
        let _e918 = bitcast<f32>(_e282);
        let _e924 = ((_e918 + -1f) / (_e918 + 1f));
        let _e925 = (_e924 * _e924);
        let _e926 = bitcast<f32>(_e292);
        let _e927 = bitcast<f32>(_e297);
        let _e928 = bitcast<f32>(_e302);
        let _e929 = (_e274 * _e274);
        let _e931 = (1f - bitcast<f32>(_e287));
        let _e936 = (2f / ((_e931 * _e931) + 1f));
        let _e938 = select(_e274, _e917, (_e274 < _e917));
        let _e942 = (_e186 - (_e164 * _e195));
        let _e943 = (_e187 - (_e165 * _e195));
        let _e944 = (_e188 - (_e166 * _e195));
        let _e947 = (1f - _e765);
        let _e952 = max((_e938 * _e938), 0.000001f);
        let _e953 = (_e952 * _e952);
        let _e971 = bitcast<f32>(_e437);
        let _e972 = (_e773 * _e773);
        let _e975 = (_e971 * _e971);
        let _e977 = (1f - bitcast<f32>(_e442));
        let _e982 = (2f / ((_e977 * _e977) + 1f));
        let _e986 = bitcast<f32>(_e407);
        let _e990 = bitcast<f32>(_e412);
        let _e991 = (_e990 * _e990);
        let _e1001 = bitcast<f32>(_e307);
        let _e1002 = bitcast<f32>(_e312);
        let _e1034 = bitcast<f32>(_e367);
        shadow_vis_u002e_sroa_u002e_0_u002e_0_ = 1f;
        k_u002e_sroa_u002e_0_u002e_0_ = 0u;
        diff_acc_u002e_sroa_u002e_9_ = 0f;
        diff_acc_u002e_sroa_u002e_6_ = 0f;
        diff_acc_u002e_sroa_u002e_0_ = 0f;
        spec_acc_u002e_sroa_u002e_9_ = 0f;
        spec_acc_u002e_sroa_u002e_6_ = 0f;
        spec_acc_u002e_sroa_u002e_0_ = 0f;
        lo_u002e_sroa_u002e_9_ = 0f;
        lo_u002e_sroa_u002e_6_ = 0f;
        lo_u002e_sroa_u002e_0_ = 0f;
        loop {
            let _e1073 = shadow_vis_u002e_sroa_u002e_0_u002e_0_;
            let _e1075 = k_u002e_sroa_u002e_0_u002e_0_;
            let _e1077 = diff_acc_u002e_sroa_u002e_9_;
            let _e1079 = diff_acc_u002e_sroa_u002e_6_;
            let _e1081 = diff_acc_u002e_sroa_u002e_0_;
            let _e1083 = spec_acc_u002e_sroa_u002e_9_;
            let _e1085 = spec_acc_u002e_sroa_u002e_6_;
            let _e1087 = spec_acc_u002e_sroa_u002e_0_;
            let _e1089 = lo_u002e_sroa_u002e_9_;
            let _e1091 = lo_u002e_sroa_u002e_6_;
            let _e1093 = lo_u002e_sroa_u002e_0_;
            if (_e1075 < _e872) {
                _0_u002e_sroa_u002e_0_u002e_0_u002e_in = _e1075;
            } else {
                let _e1099 = storage_0_21_[(((_e875 * _e677) - _e872) + _e1075)];
                _0_u002e_sroa_u002e_0_u002e_0_u002e_in = _e1099;
            }
            let _e1102 = _0_u002e_sroa_u002e_0_u002e_0_u002e_in;
            let _e1104 = (_e1102 * 36u);
            let _e1107 = storage_0_20_[_e1104];
            let _e1108 = bitcast<f32>(_e1107);
            let _e1110 = (_e1104 + 8u);
            let _e1113 = storage_0_20_[_e1110];
            let _e1115 = (_e1104 + 9u);
            let _e1118 = storage_0_20_[_e1115];
            let _e1120 = (_e1104 + 10u);
            let _e1123 = storage_0_20_[_e1120];
            let _e1131 = length(vec3<f32>((bitcast<f32>(_e1113) - _e129), (bitcast<f32>(_e1118) - _e130), (bitcast<f32>(_e1123) - _e131)));
            let _e1133 = (_e1108 < 0.5f);
            let _e1136 = select(1f, 0f, _e1133);
            let _e1138 = (_e1104 + 16u);
            let _e1141 = storage_0_20_[_e1138];
            let _e1142 = bitcast<f32>(_e1141);
            let _e1154 = storage_0_20_[(_e1104 + 12u)];
            let _e1159 = storage_0_20_[(_e1104 + 13u)];
            let _e1164 = storage_0_20_[(_e1104 + 14u)];
            let _e1169 = storage_0_20_[(_e1104 + 15u)];
            let _e1170 = bitcast<f32>(_e1169);
            let _e1173 = storage_0_20_[_e1110];
            let _e1176 = storage_0_20_[_e1115];
            let _e1179 = storage_0_20_[_e1120];
            let _e1181 = (_e129 - bitcast<f32>(_e1173));
            let _e1183 = (_e130 - bitcast<f32>(_e1176));
            let _e1185 = (_e131 - bitcast<f32>(_e1179));
            let _e1189 = (1f / length(vec3<f32>(_e1181, _e1183, _e1185)));
            let _e1194 = (_e1104 + 4u);
            let _e1197 = storage_0_20_[_e1194];
            let _e1199 = (_e1104 + 5u);
            let _e1202 = storage_0_20_[_e1199];
            let _e1204 = (_e1104 + 6u);
            let _e1207 = storage_0_20_[_e1204];
            let _e1208 = bitcast<f32>(_e1197);
            let _e1209 = bitcast<f32>(_e1202);
            let _e1210 = bitcast<f32>(_e1207);
            let _e1214 = (1f / length(vec3<f32>(_e1208, _e1209, _e1210)));
            let _e1225 = storage_0_20_[(_e1104 + 18u)];
            let _e1231 = storage_0_20_[(_e1104 + 17u)];
            let _e1238 = select(1f, 0f, (_e1108 < 1.5f));
            let _e1240 = mix(1f, smoothstep(bitcast<f32>(_e1225), bitcast<f32>(_e1231), dot(vec3<f32>((_e1214 * _e1208), (_e1214 * _e1209), (_e1214 * _e1210)), vec3<f32>((_e1181 * _e1189), (_e1183 * _e1189), (_e1185 * _e1189)))), _e1238);
            let _e1241 = (_e1131 * _e1131);
            let _e1256 = (mix(1f, (1f - smoothstep((_e1142 * 0.99f), _e1142, _e1131)), _e1136) * mix(1f, (1f / select(_e1241, 0.000001f, (_e1241 < 0.000001f))), _e1136));
            let _e1260 = (_e1240 * ((bitcast<f32>(_e1154) * _e1170) * _e1256));
            let _e1261 = (_e1240 * ((bitcast<f32>(_e1159) * _e1170) * _e1256));
            let _e1262 = (_e1240 * ((bitcast<f32>(_e1164) * _e1170) * _e1256));
            let _e1266 = select((1f - _e1238), 0f, _e1133);
            if (((1f - _e1136) * select(1f, 0f, (_e1142 < 0.5f))) > 0.5f) {
                let _e1281 = uniform_struct_0_1_.f6_.f1_.x;
                let _e1286 = uniform_struct_0_1_.f6_.f1_.y;
                let _e1291 = uniform_struct_0_1_.f6_.f1_.z;
                let _e1294 = dot(vec3<f32>((_e129 - _e170), (_e130 - _e174), (_e131 - _e178)), vec3<f32>(_e1281, _e1286, _e1291));
                let _e1299 = uniform_struct_0_1_.f6_.f2_.x;
                let _e1304 = uniform_struct_0_1_.f6_.f2_.y;
                let _e1309 = uniform_struct_0_1_.f6_.f2_.z;
                let _e1314 = uniform_struct_0_1_.f6_.f2_.w;
                let _e1318 = uniform_struct_0_1_.f6_.f3_;
                let _e1322 = select(1f, 0f, (_e1294 < _e1304));
                let _e1326 = select(1f, 0f, (_e1294 < _e1309));
                let _e1330 = select(1f, 0f, (_e1294 < _e1314));
                let _e1333 = u32(((_e1322 + _e1326) + _e1330));
                let _e1336 = mix(mix(mix(_e1304, _e1309, _e1322), _e1314, _e1326), _e1318, _e1330);
                let _e1346 = uniform_struct_0_1_.f6_.f0_[_e1333].f0_.x;
                let _e1353 = uniform_struct_0_1_.f6_.f0_[_e1333].f1_.x;
                let _e1360 = uniform_struct_0_1_.f6_.f0_[_e1333].f2_.x;
                let _e1367 = uniform_struct_0_1_.f6_.f0_[_e1333].f3_.x;
                let _e1380 = uniform_struct_0_1_.f6_.f0_[_e1333].f0_.y;
                let _e1387 = uniform_struct_0_1_.f6_.f0_[_e1333].f1_.y;
                let _e1394 = uniform_struct_0_1_.f6_.f0_[_e1333].f2_.y;
                let _e1401 = uniform_struct_0_1_.f6_.f0_[_e1333].f3_.y;
                let _e1414 = uniform_struct_0_1_.f6_.f0_[_e1333].f0_.z;
                let _e1421 = uniform_struct_0_1_.f6_.f0_[_e1333].f1_.z;
                let _e1428 = uniform_struct_0_1_.f6_.f0_[_e1333].f2_.z;
                let _e1435 = uniform_struct_0_1_.f6_.f0_[_e1333].f3_.z;
                let _e1448 = uniform_struct_0_1_.f6_.f0_[_e1333].f0_.w;
                let _e1455 = uniform_struct_0_1_.f6_.f0_[_e1333].f1_.w;
                let _e1462 = uniform_struct_0_1_.f6_.f0_[_e1333].f2_.w;
                let _e1469 = uniform_struct_0_1_.f6_.f0_[_e1333].f3_.w;
                let _e1475 = (_e1469 + (((_e129 * _e1448) + (_e130 * _e1455)) + (_e131 * _e1462)));
                let _e1492 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e1367 + (((_e129 * _e1346) + (_e130 * _e1353)) + (_e131 * _e1360))) / _e1475) * 0.5f) + 0.5f), (0.5f - (((_e1401 + (((_e129 * _e1380) + (_e130 * _e1387)) + (_e131 * _e1394))) / _e1475) * 0.5f))), _e1333, (((_e1435 + (((_e129 * _e1414) + (_e130 * _e1421)) + (_e131 * _e1428))) / _e1475) + -0.001f));
                if (_e1333 < 3u) {
                    let _e1497 = ((_e1336 - mix(mix(mix(_e1299, _e1304, _e1322), _e1309, _e1326), _e1314, _e1330)) * 0.08f);
                    let _e1503 = ((_e1336 - _e1294) / select(_e1497, 0.000001f, (_e1497 < 0.000001f)));
                    let _e1507 = select(_e1503, 0f, (_e1503 < 0f));
                    let _e1513 = (_e1333 + 1u);
                    let _e1520 = uniform_struct_0_1_.f6_.f0_[_e1513].f0_.x;
                    let _e1527 = uniform_struct_0_1_.f6_.f0_[_e1513].f1_.x;
                    let _e1534 = uniform_struct_0_1_.f6_.f0_[_e1513].f2_.x;
                    let _e1541 = uniform_struct_0_1_.f6_.f0_[_e1513].f3_.x;
                    let _e1554 = uniform_struct_0_1_.f6_.f0_[_e1513].f0_.y;
                    let _e1561 = uniform_struct_0_1_.f6_.f0_[_e1513].f1_.y;
                    let _e1568 = uniform_struct_0_1_.f6_.f0_[_e1513].f2_.y;
                    let _e1575 = uniform_struct_0_1_.f6_.f0_[_e1513].f3_.y;
                    let _e1588 = uniform_struct_0_1_.f6_.f0_[_e1513].f0_.z;
                    let _e1595 = uniform_struct_0_1_.f6_.f0_[_e1513].f1_.z;
                    let _e1602 = uniform_struct_0_1_.f6_.f0_[_e1513].f2_.z;
                    let _e1609 = uniform_struct_0_1_.f6_.f0_[_e1513].f3_.z;
                    let _e1622 = uniform_struct_0_1_.f6_.f0_[_e1513].f0_.w;
                    let _e1629 = uniform_struct_0_1_.f6_.f0_[_e1513].f1_.w;
                    let _e1636 = uniform_struct_0_1_.f6_.f0_[_e1513].f2_.w;
                    let _e1643 = uniform_struct_0_1_.f6_.f0_[_e1513].f3_.w;
                    let _e1649 = (_e1643 + (((_e129 * _e1622) + (_e130 * _e1629)) + (_e131 * _e1636)));
                    let _e1666 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e1541 + (((_e129 * _e1520) + (_e130 * _e1527)) + (_e131 * _e1534))) / _e1649) * 0.5f) + 0.5f), (0.5f - (((_e1575 + (((_e129 * _e1554) + (_e130 * _e1561)) + (_e131 * _e1568))) / _e1649) * 0.5f))), _e1513, (((_e1609 + (((_e129 * _e1588) + (_e130 * _e1595)) + (_e131 * _e1602))) / _e1649) + -0.001f));
                    vis_u002e_sroa_u002e_0_u002e_0_ = mix(_e1666, _e1492, select(_e1507, 1f, (_e1507 > 1f)));
                } else {
                    vis_u002e_sroa_u002e_0_u002e_0_ = _e1492;
                }
                let _e1671 = vis_u002e_sroa_u002e_0_u002e_0_;
                radiance_u002e_sroa_u002e_0_u002e_0_ = (_e1260 * _e1671);
                radiance_u002e_sroa_u002e_12_u002e_0_ = (_e1261 * _e1671);
                radiance_u002e_sroa_u002e_20_u002e_0_ = (_e1262 * _e1671);
            } else {
                radiance_u002e_sroa_u002e_0_u002e_0_ = _e1260;
                radiance_u002e_sroa_u002e_12_u002e_0_ = _e1261;
                radiance_u002e_sroa_u002e_20_u002e_0_ = _e1262;
            }
            let _e1682 = radiance_u002e_sroa_u002e_0_u002e_0_;
            let _e1684 = radiance_u002e_sroa_u002e_12_u002e_0_;
            let _e1686 = radiance_u002e_sroa_u002e_20_u002e_0_;
            let _e1688 = (_e1104 + 19u);
            let _e1691 = storage_0_20_[_e1688];
            if !((bitcast<f32>(_e1691) >= 0f)) {
                radiance_u002e_sroa_u002e_0_u002e_2_ = _e1682;
                radiance_u002e_sroa_u002e_12_u002e_2_ = _e1684;
                radiance_u002e_sroa_u002e_20_u002e_2_ = _e1686;
            } else {
                if !((_e1266 <= 0.5f)) {
                    radiance_u002e_sroa_u002e_0_u002e_1_ = _e1682;
                    radiance_u002e_sroa_u002e_12_u002e_1_ = _e1684;
                    radiance_u002e_sroa_u002e_20_u002e_1_ = _e1686;
                } else {
                    let _e1709 = storage_0_20_[(_e1104 + 20u)];
                    let _e1715 = storage_0_20_[(_e1104 + 24u)];
                    let _e1721 = storage_0_20_[(_e1104 + 28u)];
                    let _e1727 = storage_0_20_[(_e1104 + 32u)];
                    let _e1739 = storage_0_20_[(_e1104 + 21u)];
                    let _e1745 = storage_0_20_[(_e1104 + 25u)];
                    let _e1751 = storage_0_20_[(_e1104 + 29u)];
                    let _e1757 = storage_0_20_[(_e1104 + 33u)];
                    let _e1769 = storage_0_20_[(_e1104 + 22u)];
                    let _e1775 = storage_0_20_[(_e1104 + 26u)];
                    let _e1781 = storage_0_20_[(_e1104 + 30u)];
                    let _e1787 = storage_0_20_[(_e1104 + 34u)];
                    let _e1799 = storage_0_20_[(_e1104 + 23u)];
                    let _e1805 = storage_0_20_[(_e1104 + 27u)];
                    let _e1811 = storage_0_20_[(_e1104 + 31u)];
                    let _e1817 = storage_0_20_[(_e1104 + 35u)];
                    let _e1824 = ((((_e129 * bitcast<f32>(_e1799)) + (_e130 * bitcast<f32>(_e1805))) + (_e131 * bitcast<f32>(_e1811))) + bitcast<f32>(_e1817));
                    let _e1837 = storage_0_20_[_e1688];
                    let _e1846 = textureSampleCompareLevel(tex_0_10_, sampler_0_11_, vec2<f32>((((((((_e129 * bitcast<f32>(_e1709)) + (_e130 * bitcast<f32>(_e1715))) + (_e131 * bitcast<f32>(_e1721))) + bitcast<f32>(_e1727)) / _e1824) * 0.5f) + 0.5f), (0.5f - ((((((_e129 * bitcast<f32>(_e1739)) + (_e130 * bitcast<f32>(_e1745))) + (_e131 * bitcast<f32>(_e1751))) + bitcast<f32>(_e1757)) / _e1824) * 0.5f))), u32(bitcast<f32>(_e1837)), ((((((_e129 * bitcast<f32>(_e1769)) + (_e130 * bitcast<f32>(_e1775))) + (_e131 * bitcast<f32>(_e1781))) + bitcast<f32>(_e1787)) / _e1824) + -0.001f));
                    radiance_u002e_sroa_u002e_0_u002e_1_ = (_e1682 * _e1846);
                    radiance_u002e_sroa_u002e_12_u002e_1_ = (_e1684 * _e1846);
                    radiance_u002e_sroa_u002e_20_u002e_1_ = (_e1686 * _e1846);
                }
                let _e1854 = radiance_u002e_sroa_u002e_0_u002e_1_;
                let _e1856 = radiance_u002e_sroa_u002e_12_u002e_1_;
                let _e1858 = radiance_u002e_sroa_u002e_20_u002e_1_;
                if (_e1266 > 0.5f) {
                    let _e1863 = storage_0_20_[_e1110];
                    let _e1866 = storage_0_20_[_e1115];
                    let _e1869 = storage_0_20_[_e1120];
                    let _e1871 = (_e129 - bitcast<f32>(_e1863));
                    let _e1873 = (_e130 - bitcast<f32>(_e1866));
                    let _e1875 = (_e131 - bitcast<f32>(_e1869));
                    let _e1876 = abs(_e1871);
                    let _e1877 = abs(_e1873);
                    let _e1879 = select(_e1876, _e1877, (_e1877 > _e1876));
                    let _e1880 = abs(_e1875);
                    let _e1885 = storage_0_20_[_e1138];
                    let _e1891 = select(bitcast<f32>(_e1885), 1f, (bitcast<f32>(_e1885) < 1f));
                    let _e1893 = (_e1891 + -0.1f);
                    let _e1904 = storage_0_20_[_e1688];
                    let _e1910 = textureSampleCompareLevel(tex_0_12_, sampler_0_11_, vec3<f32>(_e1871, _e1873, _e1875), u32(bitcast<f32>(_e1904)), (((_e1891 / _e1893) - ((_e1891 * 0.1f) / (select(_e1879, _e1880, (_e1880 > _e1879)) * _e1893))) + -0.001f));
                    radiance_u002e_sroa_u002e_0_u002e_2_ = (_e1854 * _e1910);
                    radiance_u002e_sroa_u002e_12_u002e_2_ = (_e1856 * _e1910);
                    radiance_u002e_sroa_u002e_20_u002e_2_ = (_e1858 * _e1910);
                } else {
                    radiance_u002e_sroa_u002e_0_u002e_2_ = _e1854;
                    radiance_u002e_sroa_u002e_12_u002e_2_ = _e1856;
                    radiance_u002e_sroa_u002e_20_u002e_2_ = _e1858;
                }
            }
            let _e1921 = radiance_u002e_sroa_u002e_0_u002e_2_;
            let _e1923 = radiance_u002e_sroa_u002e_12_u002e_2_;
            let _e1925 = radiance_u002e_sroa_u002e_20_u002e_2_;
            if (_e869 == 6u) {
                let _e1928 = storage_0_20_[_e1104];
                let _e1929 = bitcast<f32>(_e1928);
                let _e1932 = storage_0_20_[_e1138];
                let _e1936 = storage_0_20_[_e1688];
                let _e1941 = select(1f, 0f, (_e1929 < 0.5f));
                if (((1f - _e1941) * select(1f, 0f, (bitcast<f32>(_e1932) < 0.5f))) > 0.5f) {
                    let _e1958 = uniform_struct_0_1_.f6_.f1_.x;
                    let _e1963 = uniform_struct_0_1_.f6_.f1_.y;
                    let _e1968 = uniform_struct_0_1_.f6_.f1_.z;
                    let _e1971 = dot(vec3<f32>((_e129 - _e170), (_e130 - _e174), (_e131 - _e178)), vec3<f32>(_e1958, _e1963, _e1968));
                    let _e1976 = uniform_struct_0_1_.f6_.f2_.x;
                    let _e1981 = uniform_struct_0_1_.f6_.f2_.y;
                    let _e1986 = uniform_struct_0_1_.f6_.f2_.z;
                    let _e1991 = uniform_struct_0_1_.f6_.f2_.w;
                    let _e1995 = uniform_struct_0_1_.f6_.f3_;
                    let _e1999 = select(1f, 0f, (_e1971 < _e1981));
                    let _e2003 = select(1f, 0f, (_e1971 < _e1986));
                    let _e2007 = select(1f, 0f, (_e1971 < _e1991));
                    let _e2010 = u32(((_e1999 + _e2003) + _e2007));
                    let _e2013 = mix(mix(mix(_e1981, _e1986, _e1999), _e1991, _e2003), _e1995, _e2007);
                    let _e2023 = uniform_struct_0_1_.f6_.f0_[_e2010].f0_.x;
                    let _e2030 = uniform_struct_0_1_.f6_.f0_[_e2010].f1_.x;
                    let _e2037 = uniform_struct_0_1_.f6_.f0_[_e2010].f2_.x;
                    let _e2044 = uniform_struct_0_1_.f6_.f0_[_e2010].f3_.x;
                    let _e2057 = uniform_struct_0_1_.f6_.f0_[_e2010].f0_.y;
                    let _e2064 = uniform_struct_0_1_.f6_.f0_[_e2010].f1_.y;
                    let _e2071 = uniform_struct_0_1_.f6_.f0_[_e2010].f2_.y;
                    let _e2078 = uniform_struct_0_1_.f6_.f0_[_e2010].f3_.y;
                    let _e2091 = uniform_struct_0_1_.f6_.f0_[_e2010].f0_.z;
                    let _e2098 = uniform_struct_0_1_.f6_.f0_[_e2010].f1_.z;
                    let _e2105 = uniform_struct_0_1_.f6_.f0_[_e2010].f2_.z;
                    let _e2112 = uniform_struct_0_1_.f6_.f0_[_e2010].f3_.z;
                    let _e2125 = uniform_struct_0_1_.f6_.f0_[_e2010].f0_.w;
                    let _e2132 = uniform_struct_0_1_.f6_.f0_[_e2010].f1_.w;
                    let _e2139 = uniform_struct_0_1_.f6_.f0_[_e2010].f2_.w;
                    let _e2146 = uniform_struct_0_1_.f6_.f0_[_e2010].f3_.w;
                    let _e2152 = (_e2146 + (((_e129 * _e2125) + (_e130 * _e2132)) + (_e131 * _e2139)));
                    let _e2169 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e2044 + (((_e129 * _e2023) + (_e130 * _e2030)) + (_e131 * _e2037))) / _e2152) * 0.5f) + 0.5f), (0.5f - (((_e2078 + (((_e129 * _e2057) + (_e130 * _e2064)) + (_e131 * _e2071))) / _e2152) * 0.5f))), _e2010, (((_e2112 + (((_e129 * _e2091) + (_e130 * _e2098)) + (_e131 * _e2105))) / _e2152) + -0.001f));
                    if (_e2010 < 3u) {
                        let _e2174 = ((_e2013 - mix(mix(mix(_e1976, _e1981, _e1999), _e1986, _e2003), _e1991, _e2007)) * 0.08f);
                        let _e2180 = ((_e2013 - _e1971) / select(_e2174, 0.000001f, (_e2174 < 0.000001f)));
                        let _e2184 = select(_e2180, 0f, (_e2180 < 0f));
                        let _e2190 = (_e2010 + 1u);
                        let _e2197 = uniform_struct_0_1_.f6_.f0_[_e2190].f0_.x;
                        let _e2204 = uniform_struct_0_1_.f6_.f0_[_e2190].f1_.x;
                        let _e2211 = uniform_struct_0_1_.f6_.f0_[_e2190].f2_.x;
                        let _e2218 = uniform_struct_0_1_.f6_.f0_[_e2190].f3_.x;
                        let _e2231 = uniform_struct_0_1_.f6_.f0_[_e2190].f0_.y;
                        let _e2238 = uniform_struct_0_1_.f6_.f0_[_e2190].f1_.y;
                        let _e2245 = uniform_struct_0_1_.f6_.f0_[_e2190].f2_.y;
                        let _e2252 = uniform_struct_0_1_.f6_.f0_[_e2190].f3_.y;
                        let _e2265 = uniform_struct_0_1_.f6_.f0_[_e2190].f0_.z;
                        let _e2272 = uniform_struct_0_1_.f6_.f0_[_e2190].f1_.z;
                        let _e2279 = uniform_struct_0_1_.f6_.f0_[_e2190].f2_.z;
                        let _e2286 = uniform_struct_0_1_.f6_.f0_[_e2190].f3_.z;
                        let _e2299 = uniform_struct_0_1_.f6_.f0_[_e2190].f0_.w;
                        let _e2306 = uniform_struct_0_1_.f6_.f0_[_e2190].f1_.w;
                        let _e2313 = uniform_struct_0_1_.f6_.f0_[_e2190].f2_.w;
                        let _e2320 = uniform_struct_0_1_.f6_.f0_[_e2190].f3_.w;
                        let _e2326 = (_e2320 + (((_e129 * _e2299) + (_e130 * _e2306)) + (_e131 * _e2313)));
                        let _e2343 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e2218 + (((_e129 * _e2197) + (_e130 * _e2204)) + (_e131 * _e2211))) / _e2326) * 0.5f) + 0.5f), (0.5f - (((_e2252 + (((_e129 * _e2231) + (_e130 * _e2238)) + (_e131 * _e2245))) / _e2326) * 0.5f))), _e2190, (((_e2286 + (((_e129 * _e2265) + (_e130 * _e2272)) + (_e131 * _e2279))) / _e2326) + -0.001f));
                        _0_u002e_sroa_u002e_0_u002e_0_ = mix(_e2343, _e2169, select(_e2184, 1f, (_e2184 > 1f)));
                    } else {
                        _0_u002e_sroa_u002e_0_u002e_0_ = _e2169;
                    }
                } else {
                    if (bitcast<f32>(_e1936) < 0f) {
                        _0_u002e_sroa_u002e_0_u002e_0_ = 1f;
                    } else {
                        if ((_e1929 < 1.5f) & (_e1941 > 0.5f)) {
                            let _e2357 = storage_0_20_[_e1110];
                            let _e2360 = storage_0_20_[_e1115];
                            let _e2363 = storage_0_20_[_e1120];
                            let _e2365 = (_e129 - bitcast<f32>(_e2357));
                            let _e2367 = (_e130 - bitcast<f32>(_e2360));
                            let _e2369 = (_e131 - bitcast<f32>(_e2363));
                            let _e2370 = abs(_e2365);
                            let _e2371 = abs(_e2367);
                            let _e2373 = select(_e2370, _e2371, (_e2371 > _e2370));
                            let _e2374 = abs(_e2369);
                            let _e2379 = storage_0_20_[_e1138];
                            let _e2385 = select(bitcast<f32>(_e2379), 1f, (bitcast<f32>(_e2379) < 1f));
                            let _e2387 = (_e2385 + -0.1f);
                            let _e2398 = storage_0_20_[_e1688];
                            let _e2404 = textureSampleCompareLevel(tex_0_12_, sampler_0_11_, vec3<f32>(_e2365, _e2367, _e2369), u32(bitcast<f32>(_e2398)), (((_e2385 / _e2387) - ((_e2385 * 0.1f) / (select(_e2373, _e2374, (_e2374 > _e2373)) * _e2387))) + -0.001f));
                            _0_u002e_sroa_u002e_0_u002e_0_ = _e2404;
                        } else {
                            let _e2410 = storage_0_20_[(_e1104 + 20u)];
                            let _e2416 = storage_0_20_[(_e1104 + 24u)];
                            let _e2422 = storage_0_20_[(_e1104 + 28u)];
                            let _e2428 = storage_0_20_[(_e1104 + 32u)];
                            let _e2440 = storage_0_20_[(_e1104 + 21u)];
                            let _e2446 = storage_0_20_[(_e1104 + 25u)];
                            let _e2452 = storage_0_20_[(_e1104 + 29u)];
                            let _e2458 = storage_0_20_[(_e1104 + 33u)];
                            let _e2470 = storage_0_20_[(_e1104 + 22u)];
                            let _e2476 = storage_0_20_[(_e1104 + 26u)];
                            let _e2482 = storage_0_20_[(_e1104 + 30u)];
                            let _e2488 = storage_0_20_[(_e1104 + 34u)];
                            let _e2500 = storage_0_20_[(_e1104 + 23u)];
                            let _e2506 = storage_0_20_[(_e1104 + 27u)];
                            let _e2512 = storage_0_20_[(_e1104 + 31u)];
                            let _e2518 = storage_0_20_[(_e1104 + 35u)];
                            let _e2525 = ((((_e129 * bitcast<f32>(_e2500)) + (_e130 * bitcast<f32>(_e2506))) + (_e131 * bitcast<f32>(_e2512))) + bitcast<f32>(_e2518));
                            let _e2538 = storage_0_20_[_e1688];
                            let _e2547 = textureSampleCompareLevel(tex_0_10_, sampler_0_11_, vec2<f32>((((((((_e129 * bitcast<f32>(_e2410)) + (_e130 * bitcast<f32>(_e2416))) + (_e131 * bitcast<f32>(_e2422))) + bitcast<f32>(_e2428)) / _e2525) * 0.5f) + 0.5f), (0.5f - ((((((_e129 * bitcast<f32>(_e2440)) + (_e130 * bitcast<f32>(_e2446))) + (_e131 * bitcast<f32>(_e2452))) + bitcast<f32>(_e2458)) / _e2525) * 0.5f))), u32(bitcast<f32>(_e2538)), ((((((_e129 * bitcast<f32>(_e2470)) + (_e130 * bitcast<f32>(_e2476))) + (_e131 * bitcast<f32>(_e2482))) + bitcast<f32>(_e2488)) / _e2525) + -0.001f));
                            _0_u002e_sroa_u002e_0_u002e_0_ = _e2547;
                        }
                    }
                }
                let _e2550 = _0_u002e_sroa_u002e_0_u002e_0_;
                shadow_vis_u002e_sroa_u002e_0_u002e_1_ = select(_e2550, _e1073, (_e1073 < _e2550));
            } else {
                shadow_vis_u002e_sroa_u002e_0_u002e_1_ = _e1073;
            }
            let _e2556 = shadow_vis_u002e_sroa_u002e_0_u002e_1_;
            out_u002e_shadow_vis_u002e_sroa_u002e_0_u002e_1_ = _e2556;
            let _e2560 = storage_0_20_[_e1110];
            let _e2563 = storage_0_20_[_e1115];
            let _e2566 = storage_0_20_[_e1120];
            let _e2568 = (bitcast<f32>(_e2560) - _e129);
            let _e2570 = (bitcast<f32>(_e2563) - _e130);
            let _e2572 = (bitcast<f32>(_e2566) - _e131);
            let _e2575 = storage_0_20_[_e1104];
            let _e2581 = select(1f, 0f, (bitcast<f32>(_e2575) < 0.5f));
            let _e2584 = storage_0_20_[_e1194];
            let _e2587 = storage_0_20_[_e1199];
            let _e2590 = storage_0_20_[_e1204];
            let _e2591 = bitcast<f32>(_e2584);
            let _e2592 = bitcast<f32>(_e2587);
            let _e2593 = bitcast<f32>(_e2590);
            let _e2597 = (1f / length(vec3<f32>(_e2591, _e2592, _e2593)));
            let _e2604 = (1f / length(vec3<f32>(_e2568, _e2570, _e2572)));
            let _e2608 = mix((_e2597 * _e2591), (_e2568 * _e2604), _e2581);
            let _e2609 = mix((_e2597 * _e2592), (_e2570 * _e2604), _e2581);
            let _e2610 = mix((_e2597 * _e2593), (_e2572 * _e2604), _e2581);
            let _e2614 = (1f / length(vec3<f32>(_e2608, _e2609, _e2610)));
            let _e2615 = (_e2608 * _e2614);
            let _e2616 = (_e2609 * _e2614);
            let _e2617 = (_e2610 * _e2614);
            let _e2622 = ((_e186 + _e2615) + 0f);
            let _e2624 = ((_e187 + _e2616) + 0f);
            let _e2626 = ((_e188 + _e2617) + 0.000001f);
            let _e2630 = (1f / length(vec3<f32>(_e2622, _e2624, _e2626)));
            let _e2631 = (_e2630 * _e2622);
            let _e2632 = (_e2630 * _e2624);
            let _e2633 = (_e2630 * _e2626);
            let _e2636 = dot(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e2615, _e2616, _e2617));
            let _e2640 = select(_e2636, 0.000001f, (_e2636 < 0.000001f));
            if (_e2640 > 0.000001f) {
                let _e2645 = dot(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e2631, _e2632, _e2633));
                let _e2649 = select(_e2645, 0.000001f, (_e2645 < 0.000001f));
                let _e2652 = dot(vec3<f32>(_e186, _e187, _e188), vec3<f32>(_e2631, _e2632, _e2633));
                let _e2656 = select(_e2652, 0.000001f, (_e2652 < 0.000001f));
                let _e2657 = mix(_e925, _e779, _e765);
                let _e2658 = mix(_e925, _e780, _e765);
                let _e2659 = mix(_e925, _e781, _e765);
                let _e2662 = pow(0.85714287f, 5f);
                let _e2664 = (1f - _e2657);
                let _e2666 = (_e2657 + (_e2664 * _e2662));
                let _e2668 = (1f - _e2658);
                let _e2670 = (_e2658 + (_e2668 * _e2662));
                let _e2672 = (1f - _e2659);
                let _e2674 = (_e2659 + (_e2672 * _e2662));
                let _e2679 = (1f - _e2656);
                let _e2688 = ((_e2656 * pow(_e2679, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e2690 = pow(_e2679, 5f);
                let _e2712 = mix(_e925, _e779, _e765);
                let _e2713 = mix(_e925, _e780, _e765);
                let _e2714 = mix(_e925, _e781, _e765);
                let _e2717 = pow(0.85714287f, 5f);
                let _e2719 = (1f - _e2712);
                let _e2721 = (_e2712 + (_e2719 * _e2717));
                let _e2723 = (1f - _e2713);
                let _e2725 = (_e2713 + (_e2723 * _e2717));
                let _e2727 = (1f - _e2714);
                let _e2729 = (_e2714 + (_e2727 * _e2717));
                let _e2741 = ((_e2656 * pow(_e2679, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e2743 = pow(_e2679, 5f);
                let _e2766 = (_e929 * sqrt(_e936));
                let _e2769 = (_e931 * (_e929 * sqrt(_e936)));
                let _e2772 = dot(vec3<f32>(_e2631, _e2632, _e2633), vec3<f32>(_e549, _e550, _e551));
                let _e2775 = dot(vec3<f32>(_e2631, _e2632, _e2633), vec3<f32>(cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).x, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).y, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).z));
                let _e2777 = (_e2766 * _e2766);
                let _e2780 = (_e2769 * _e2769);
                let _e2783 = (_e2649 * _e2649);
                let _e2784 = (_e2783 + (((_e2772 * _e2772) / _e2777) + ((_e2775 * _e2775) / _e2780)));
                let _e2791 = (1f / (_e2784 * (((_e2766 * 3.1415927f) * _e2769) * _e2784)));
                let _e2794 = dot(vec3<f32>(_e186, _e187, _e188), vec3<f32>(_e549, _e550, _e551));
                let _e2797 = dot(vec3<f32>(_e186, _e187, _e188), vec3<f32>(cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).x, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).y, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).z));
                let _e2800 = dot(vec3<f32>(_e2615, _e2616, _e2617), vec3<f32>(_e549, _e550, _e551));
                let _e2803 = dot(vec3<f32>(_e2615, _e2616, _e2617), vec3<f32>(cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).x, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).y, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).z));
                let _e2809 = (_e2640 * _e2640);
                let _e2829 = (0.5f / max(((_e195 * sqrt(max((_e2809 + ((_e2777 * (_e2800 * _e2800)) + (_e2780 * (_e2803 * _e2803)))), 0.000001f))) + (_e2640 * sqrt(max((_e695 + ((_e2777 * (_e2794 * _e2794)) + (_e2780 * (_e2797 * _e2797)))), 0.000001f)))), 0.000001f));
                let _e2848 = (1f / length(vec3<f32>(_e942, _e943, _e944)));
                let _e2855 = (_e2615 - (_e164 * _e2640));
                let _e2856 = (_e2616 - (_e165 * _e2640));
                let _e2857 = (_e2617 - (_e166 * _e2640));
                let _e2861 = (1f / length(vec3<f32>(_e2855, _e2856, _e2857)));
                let _e2867 = dot(vec3<f32>((_e942 * _e2848), (_e943 * _e2848), (_e944 * _e2848)), vec3<f32>((_e2855 * _e2861), (_e2856 * _e2861), (_e2857 * _e2861)));
                let _e2872 = acos(max(_e195, 0f));
                let _e2875 = acos(max(_e2640, 0f));
                let _e2899 = (max((1f - dot(vec3<f32>((max(((_e2657 + (_e2664 * _e2690)) - ((_e2666 - (_e2666 * _e926)) * _e2688)), 0f) * _e767), (max(((_e2658 + (_e2668 * _e2690)) - ((_e2670 - (_e2670 * _e927)) * _e2688)), 0f) * _e767), (max(((_e2659 + (_e2672 * _e2690)) - ((_e2674 - (_e2674 * _e928)) * _e2688)), 0f) * _e767)), vec3<f32>(0.2126f, 0.7152f, 0.0722f))), 0f) * (((1f - ((_e953 * 0.5f) / (_e953 + 0.57f))) + (tan(min(_e2872, _e2875)) * ((((_e953 * 0.45f) / (_e953 + 0.09f)) * select(0f, _e2867, (_e2867 > 0f))) * sin(max(_e2872, _e2875))))) * 0.31830987f));
                let _e2912 = (_e862 * ((_e752 * ((((max(((_e2712 + (_e2719 * _e2743)) - ((_e2721 - (_e2721 * _e926)) * _e2741)), 0f) * _e767) * _e2791) * _e2829) * _e766)) + (_e752 * (((_e947 * _e776) * _e2899) * _e766))));
                let _e2913 = (_e864 * ((_e757 * ((((max(((_e2713 + (_e2723 * _e2743)) - ((_e2725 - (_e2725 * _e927)) * _e2741)), 0f) * _e767) * _e2791) * _e2829) * _e766)) + (_e757 * (((_e947 * _e777) * _e2899) * _e766))));
                let _e2914 = (_e866 * ((_e762 * ((((max(((_e2714 + (_e2727 * _e2743)) - ((_e2729 - (_e2729 * _e928)) * _e2741)), 0f) * _e767) * _e2791) * _e2829) * _e766)) + (_e762 * (((_e947 * _e778) * _e2899) * _e766))));
                if (_e869 == 0u) {
                    spec_acc_u002e_sroa_u002e_0_u002e_2_ = _e1087;
                    spec_acc_u002e_sroa_u002e_6_u002e_2_ = _e1085;
                    spec_acc_u002e_sroa_u002e_9_u002e_2_ = _e1083;
                    diff_acc_u002e_sroa_u002e_0_u002e_2_ = _e1081;
                    diff_acc_u002e_sroa_u002e_6_u002e_2_ = _e1079;
                    diff_acc_u002e_sroa_u002e_9_u002e_2_ = _e1077;
                } else {
                    let _e2921 = mix(_e925, _e779, _e765);
                    let _e2922 = mix(_e925, _e780, _e765);
                    let _e2923 = mix(_e925, _e781, _e765);
                    let _e2926 = pow(0.85714287f, 5f);
                    let _e2928 = (1f - _e2921);
                    let _e2930 = (_e2921 + (_e2928 * _e2926));
                    let _e2932 = (1f - _e2922);
                    let _e2934 = (_e2922 + (_e2932 * _e2926));
                    let _e2936 = (1f - _e2923);
                    let _e2938 = (_e2923 + (_e2936 * _e2926));
                    let _e2950 = ((_e2656 * pow(_e2679, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                    let _e2952 = pow(_e2679, 5f);
                    let _e2975 = (_e929 * sqrt(_e936));
                    let _e2978 = (_e931 * (_e929 * sqrt(_e936)));
                    let _e2981 = dot(vec3<f32>(_e2631, _e2632, _e2633), vec3<f32>(_e549, _e550, _e551));
                    let _e2984 = dot(vec3<f32>(_e2631, _e2632, _e2633), vec3<f32>(cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).x, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).y, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).z));
                    let _e2986 = (_e2975 * _e2975);
                    let _e2989 = (_e2978 * _e2978);
                    let _e2992 = (_e2783 + (((_e2981 * _e2981) / _e2986) + ((_e2984 * _e2984) / _e2989)));
                    let _e2999 = (1f / (_e2992 * (((_e2975 * 3.1415927f) * _e2978) * _e2992)));
                    let _e3002 = dot(vec3<f32>(_e186, _e187, _e188), vec3<f32>(_e549, _e550, _e551));
                    let _e3005 = dot(vec3<f32>(_e186, _e187, _e188), vec3<f32>(cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).x, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).y, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).z));
                    let _e3008 = dot(vec3<f32>(_e2615, _e2616, _e2617), vec3<f32>(_e549, _e550, _e551));
                    let _e3011 = dot(vec3<f32>(_e2615, _e2616, _e2617), vec3<f32>(cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).x, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).y, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).z));
                    let _e3036 = (0.5f / max(((_e195 * sqrt(max((_e2809 + ((_e2986 * (_e3008 * _e3008)) + (_e2989 * (_e3011 * _e3011)))), 0.000001f))) + (_e2640 * sqrt(max((_e695 + ((_e2986 * (_e3002 * _e3002)) + (_e2989 * (_e3005 * _e3005)))), 0.000001f)))), 0.000001f));
                    let _e3052 = (_e862 * (_e752 * ((((max(((_e2921 + (_e2928 * _e2952)) - ((_e2930 - (_e2930 * _e926)) * _e2950)), 0f) * _e767) * _e2999) * _e3036) * _e766)));
                    let _e3053 = (_e864 * (_e757 * ((((max(((_e2922 + (_e2932 * _e2952)) - ((_e2934 - (_e2934 * _e927)) * _e2950)), 0f) * _e767) * _e2999) * _e3036) * _e766)));
                    let _e3054 = (_e866 * (_e762 * ((((max(((_e2923 + (_e2936 * _e2952)) - ((_e2938 - (_e2938 * _e928)) * _e2950)), 0f) * _e767) * _e2999) * _e3036) * _e766)));
                    spec_acc_u002e_sroa_u002e_0_u002e_2_ = (_e1087 + (_e1921 * (_e2640 * _e3052)));
                    spec_acc_u002e_sroa_u002e_6_u002e_2_ = (_e1085 + (_e1923 * (_e2640 * _e3053)));
                    spec_acc_u002e_sroa_u002e_9_u002e_2_ = (_e1083 + (_e1925 * (_e2640 * _e3054)));
                    diff_acc_u002e_sroa_u002e_0_u002e_2_ = (_e1081 + (_e1921 * (_e2640 * (_e2912 - _e3052))));
                    diff_acc_u002e_sroa_u002e_6_u002e_2_ = (_e1079 + (_e1923 * (_e2640 * (_e2913 - _e3053))));
                    diff_acc_u002e_sroa_u002e_9_u002e_2_ = (_e1077 + (_e1925 * (_e2640 * (_e2914 - _e3054))));
                }
                let _e3083 = spec_acc_u002e_sroa_u002e_0_u002e_2_;
                let _e3085 = spec_acc_u002e_sroa_u002e_6_u002e_2_;
                let _e3087 = spec_acc_u002e_sroa_u002e_9_u002e_2_;
                let _e3089 = diff_acc_u002e_sroa_u002e_0_u002e_2_;
                let _e3091 = diff_acc_u002e_sroa_u002e_6_u002e_2_;
                let _e3093 = diff_acc_u002e_sroa_u002e_9_u002e_2_;
                if _e787 {
                    let _e3099 = (_e975 * sqrt(_e982));
                    let _e3102 = (_e977 * (_e975 * sqrt(_e982)));
                    let _e3105 = dot(vec3<f32>(_e2631, _e2632, _e2633), vec3<f32>(_e549, _e550, _e551));
                    let _e3108 = dot(vec3<f32>(_e2631, _e2632, _e2633), vec3<f32>(cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).x, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).y, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).z));
                    let _e3110 = (_e3099 * _e3099);
                    let _e3113 = (_e3102 * _e3102);
                    let _e3116 = (_e2783 + (((_e3105 * _e3105) / _e3110) + ((_e3108 * _e3108) / _e3113)));
                    let _e3126 = dot(vec3<f32>(_e186, _e187, _e188), vec3<f32>(_e549, _e550, _e551));
                    let _e3129 = dot(vec3<f32>(_e186, _e187, _e188), vec3<f32>(cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).x, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).y, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).z));
                    let _e3132 = dot(vec3<f32>(_e2615, _e2616, _e2617), vec3<f32>(_e549, _e550, _e551));
                    let _e3135 = dot(vec3<f32>(_e2615, _e2616, _e2617), vec3<f32>(cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).x, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).y, cross(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e549, _e550, _e551)).z));
                    let _e3162 = ((_e972 + ((1f - _e972) * pow(_e2679, 5f))) * ((1f / (_e3116 * (((_e3099 * 3.1415927f) * _e3102) * _e3116))) * (0.5f / max(((_e195 * sqrt(max((_e2809 + ((_e3110 * (_e3132 * _e3132)) + (_e3113 * (_e3135 * _e3135)))), 0.000001f))) + (_e2640 * sqrt(max((_e695 + ((_e3110 * (_e3126 * _e3126)) + (_e3113 * (_e3129 * _e3129)))), 0.000001f)))), 0.000001f))));
                    coat_bsdf_u002e_sroa_u002e_0_u002e_0_ = ((_e3162 * bitcast<f32>(_e452)) * _e763);
                    coat_bsdf_u002e_sroa_u002e_5_u002e_0_ = ((_e3162 * bitcast<f32>(_e457)) * _e763);
                    coat_bsdf_u002e_sroa_u002e_8_u002e_0_ = ((_e3162 * bitcast<f32>(_e462)) * _e763);
                } else {
                    coat_bsdf_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    coat_bsdf_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    coat_bsdf_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e3179 = coat_bsdf_u002e_sroa_u002e_0_u002e_0_;
                let _e3181 = coat_bsdf_u002e_sroa_u002e_5_u002e_0_;
                let _e3183 = coat_bsdf_u002e_sroa_u002e_8_u002e_0_;
                if !((_e986 <= 0f)) {
                    let _e3207 = ((_e2656 * ((1f / ((_e991 * (((1f / _e195) + (1f / _e2640)) + -2f)) + 1f)) * (_e991 / (pow((((_e991 + -1f) * _e2783) + 1f), 2f) * 3.1415927f)))) / max(((_e195 * 4f) * _e2640), 0.000001f));
                    fuzz_bsdf_u002e_sroa_u002e_0_u002e_0_ = ((_e3207 * bitcast<f32>(_e417)) * _e986);
                    fuzz_bsdf_u002e_sroa_u002e_5_u002e_0_ = ((_e3207 * bitcast<f32>(_e422)) * _e986);
                    fuzz_bsdf_u002e_sroa_u002e_8_u002e_0_ = ((_e3207 * bitcast<f32>(_e427)) * _e986);
                } else {
                    fuzz_bsdf_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    fuzz_bsdf_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    fuzz_bsdf_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e3224 = fuzz_bsdf_u002e_sroa_u002e_0_u002e_0_;
                let _e3226 = fuzz_bsdf_u002e_sroa_u002e_5_u002e_0_;
                let _e3228 = fuzz_bsdf_u002e_sroa_u002e_8_u002e_0_;
                if !((_e1001 <= 0f)) {
                    let _e3230 = (_e929 * sqrt(_e936));
                    if !((_e1002 <= 0f)) {
                        extinction_u002e_sroa_u002e_0_u002e_0_ = (-(log(max(bitcast<f32>(_e327), 0.000001f))) / _e1002);
                        extinction_u002e_sroa_u002e_5_u002e_0_ = (-(log(max(bitcast<f32>(_e332), 0.000001f))) / _e1002);
                        extinction_u002e_sroa_u002e_8_u002e_0_ = (-(log(max(bitcast<f32>(_e337), 0.000001f))) / _e1002);
                    } else {
                        extinction_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        extinction_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        extinction_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e3250 = extinction_u002e_sroa_u002e_0_u002e_0_;
                    let _e3252 = extinction_u002e_sroa_u002e_5_u002e_0_;
                    let _e3254 = extinction_u002e_sroa_u002e_8_u002e_0_;
                    let _e3261 = (_e3230 * _e3230);
                    let _e3266 = ((_e2783 * (_e3261 + -1f)) + 1f);
                    let _e3272 = (1f - _e3261);
                    let _e3291 = ((1f - (_e925 + ((1f - _e925) * pow(select(0.999999f, (1f - _e191), _e193), 5f)))) * ((_e3261 / (_e3266 * (_e3266 * 3.1415927f))) * (0.5f / max(((_e195 * sqrt(max((_e3261 + (_e2809 * _e3272)), 0.000001f))) + (_e2640 * sqrt(max((_e3261 + (_e695 * _e3272)), 0.000001f)))), 0.000001f))));
                    trans_bsdf_u002e_sroa_u002e_0_u002e_0_ = (((exp((_e1002 * -(_e3250))) * _e3291) * bitcast<f32>(_e327)) * _e1001);
                    trans_bsdf_u002e_sroa_u002e_5_u002e_0_ = (((exp((_e1002 * -(_e3252))) * _e3291) * bitcast<f32>(_e332)) * _e1001);
                    trans_bsdf_u002e_sroa_u002e_8_u002e_0_ = (((exp((_e1002 * -(_e3254))) * _e3291) * bitcast<f32>(_e337)) * _e1001);
                } else {
                    trans_bsdf_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    trans_bsdf_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    trans_bsdf_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e3320 = trans_bsdf_u002e_sroa_u002e_0_u002e_0_;
                let _e3322 = trans_bsdf_u002e_sroa_u002e_5_u002e_0_;
                let _e3324 = trans_bsdf_u002e_sroa_u002e_8_u002e_0_;
                if !((_e768 <= 0f)) {
                    let _e3329 = length(vec3<f32>((_e942 - _e2855), (_e943 - _e2856), (_e944 - _e2857)));
                    let _e3331 = sqrt(3f);
                    let _e3354 = -(_e3329);
                    let _e3362 = (max(((clamp(((_e195 * _e2640) + (((_e699 + max((1f - _e2809), 0f)) - (_e3329 * _e3329)) * 0.5f)), -1f, 1f) * bitcast<f32>(_e377)) + 1f), 0f) * 0.31830987f);
                    ss_bsdf_u002e_sroa_u002e_0_u002e_0_ = (((exp((((1f / max((_e1034 * bitcast<f32>(_e372)), 0.000001f)) * _e3331) * _e3354)) * _e3362) * _e782) * _e768);
                    ss_bsdf_u002e_sroa_u002e_5_u002e_0_ = (((exp((((1f / max((_e1034 * bitcast<f32>(_e397)), 0.000001f)) * _e3331) * _e3354)) * _e3362) * _e783) * _e768);
                    ss_bsdf_u002e_sroa_u002e_8_u002e_0_ = (((exp((((1f / max((_e1034 * bitcast<f32>(_e402)), 0.000001f)) * _e3331) * _e3354)) * _e3362) * _e784) * _e768);
                } else {
                    ss_bsdf_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    ss_bsdf_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    ss_bsdf_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e3382 = ss_bsdf_u002e_sroa_u002e_0_u002e_0_;
                let _e3384 = ss_bsdf_u002e_sroa_u002e_5_u002e_0_;
                let _e3386 = ss_bsdf_u002e_sroa_u002e_8_u002e_0_;
                lo_u002e_sroa_u002e_0_u002e_1_ = (_e1093 + (_e1921 * (_e2640 * (((_e2912 + _e3179) + (_e3224 + _e3320)) + _e3382))));
                lo_u002e_sroa_u002e_6_u002e_1_ = (_e1091 + (_e1923 * (_e2640 * (((_e2913 + _e3181) + (_e3226 + _e3322)) + _e3384))));
                lo_u002e_sroa_u002e_9_u002e_1_ = (_e1089 + (_e1925 * (_e2640 * (((_e2914 + _e3183) + (_e3228 + _e3324)) + _e3386))));
                spec_acc_u002e_sroa_u002e_0_u002e_1_ = _e3083;
                spec_acc_u002e_sroa_u002e_6_u002e_1_ = _e3085;
                spec_acc_u002e_sroa_u002e_9_u002e_1_ = _e3087;
                diff_acc_u002e_sroa_u002e_0_u002e_1_ = _e3089;
                diff_acc_u002e_sroa_u002e_6_u002e_1_ = _e3091;
                diff_acc_u002e_sroa_u002e_9_u002e_1_ = _e3093;
            } else {
                lo_u002e_sroa_u002e_0_u002e_1_ = _e1093;
                lo_u002e_sroa_u002e_6_u002e_1_ = _e1091;
                lo_u002e_sroa_u002e_9_u002e_1_ = _e1089;
                spec_acc_u002e_sroa_u002e_0_u002e_1_ = _e1087;
                spec_acc_u002e_sroa_u002e_6_u002e_1_ = _e1085;
                spec_acc_u002e_sroa_u002e_9_u002e_1_ = _e1083;
                diff_acc_u002e_sroa_u002e_0_u002e_1_ = _e1081;
                diff_acc_u002e_sroa_u002e_6_u002e_1_ = _e1079;
                diff_acc_u002e_sroa_u002e_9_u002e_1_ = _e1077;
            }
            let _e3427 = lo_u002e_sroa_u002e_0_u002e_1_;
            out_u002e_lo_u002e_sroa_u002e_0_u002e_1_ = _e3427;
            let _e3430 = lo_u002e_sroa_u002e_6_u002e_1_;
            out_u002e_lo_u002e_sroa_u002e_6_u002e_1_ = _e3430;
            let _e3433 = lo_u002e_sroa_u002e_9_u002e_1_;
            out_u002e_lo_u002e_sroa_u002e_9_u002e_1_ = _e3433;
            let _e3436 = spec_acc_u002e_sroa_u002e_0_u002e_1_;
            out_u002e_spec_acc_u002e_sroa_u002e_0_u002e_1_ = _e3436;
            let _e3439 = spec_acc_u002e_sroa_u002e_6_u002e_1_;
            out_u002e_spec_acc_u002e_sroa_u002e_6_u002e_1_ = _e3439;
            let _e3442 = spec_acc_u002e_sroa_u002e_9_u002e_1_;
            out_u002e_spec_acc_u002e_sroa_u002e_9_u002e_1_ = _e3442;
            let _e3445 = diff_acc_u002e_sroa_u002e_0_u002e_1_;
            out_u002e_diff_acc_u002e_sroa_u002e_0_u002e_1_ = _e3445;
            let _e3448 = diff_acc_u002e_sroa_u002e_6_u002e_1_;
            out_u002e_diff_acc_u002e_sroa_u002e_6_u002e_1_ = _e3448;
            let _e3451 = diff_acc_u002e_sroa_u002e_9_u002e_1_;
            out_u002e_diff_acc_u002e_sroa_u002e_9_u002e_1_ = _e3451;
            let _e3454 = (_e1075 + 1u);
            if (_e3454 == _e885) {
                lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e3427;
                lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e3430;
                lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e3433;
                spec_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e3436;
                spec_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e3439;
                spec_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e3442;
                diff_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e3445;
                diff_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e3448;
                diff_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e3451;
                shadow_vis_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e2556;
                break;
            } else {
                shadow_vis_u002e_sroa_u002e_0_u002e_0_ = _e2556;
                k_u002e_sroa_u002e_0_u002e_0_ = _e3454;
                diff_acc_u002e_sroa_u002e_9_ = _e3451;
                diff_acc_u002e_sroa_u002e_6_ = _e3448;
                diff_acc_u002e_sroa_u002e_0_ = _e3445;
                spec_acc_u002e_sroa_u002e_9_ = _e3442;
                spec_acc_u002e_sroa_u002e_6_ = _e3439;
                spec_acc_u002e_sroa_u002e_0_ = _e3436;
                lo_u002e_sroa_u002e_9_ = _e3433;
                lo_u002e_sroa_u002e_6_ = _e3430;
                lo_u002e_sroa_u002e_0_ = _e3427;
                continue;
            }
        }
    }
    let _e3498 = lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
    let _e3500 = lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
    let _e3502 = lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
    let _e3504 = spec_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
    let _e3506 = spec_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
    let _e3508 = spec_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
    let _e3510 = diff_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
    let _e3512 = diff_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
    let _e3514 = diff_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
    let _e3516 = shadow_vis_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
    let _e3526 = uniform_struct_0_1_.f0_.x;
    let _e3530 = uniform_struct_0_1_.f0_.y;
    let _e3534 = uniform_struct_0_1_.f0_.z;
    let _e3538 = (_e680.x * (mix(_e776, (_e776 * _e767), _e765) * _e3526));
    let _e3539 = (_e680.x * (mix(_e777, (_e777 * _e767), _e765) * _e3530));
    let _e3540 = (_e680.x * (mix(_e778, (_e778 * _e767), _e765) * _e3534));
    let _e3541 = bitcast<f32>(_e487);
    if !((_e3541 <= 0f)) {
        let _e3548 = (_e3541 * bitcast<f32>(_e492));
        let _e3549 = (_e3541 * bitcast<f32>(_e497));
        let _e3550 = (_e3541 * bitcast<f32>(_e502));
        let _e3560 = ((1f - _e763) + ((1f - pow((1f - _e195), 5f)) * _e763));
        emission_u002e_sroa_u002e_0_u002e_1_ = (_e3548 + ((((_e3548 * bitcast<f32>(_e452)) * _e3560) - _e3548) * _e763));
        emission_u002e_sroa_u002e_8_u002e_1_ = (_e3549 + ((((_e3549 * bitcast<f32>(_e457)) * _e3560) - _e3549) * _e763));
        emission_u002e_sroa_u002e_11_u002e_1_ = (_e3550 + ((((_e3550 * bitcast<f32>(_e462)) * _e3560) - _e3550) * _e763));
    } else {
        emission_u002e_sroa_u002e_0_u002e_1_ = 0f;
        emission_u002e_sroa_u002e_8_u002e_1_ = 0f;
        emission_u002e_sroa_u002e_11_u002e_1_ = 0f;
    }
    let _e3589 = emission_u002e_sroa_u002e_0_u002e_1_;
    let _e3591 = emission_u002e_sroa_u002e_8_u002e_1_;
    let _e3593 = emission_u002e_sroa_u002e_11_u002e_1_;
    let _e3594 = bitcast<f32>(_e282);
    let _e3599 = ((_e3594 + -1f) / (_e3594 + 1f));
    let _e3600 = (_e3599 * _e3599);
    let _e3608 = (dot(vec3<f32>(_e164, _e165, _e166), vec3<f32>(_e186, _e187, _e188)) * 2f);
    let _e3617 = uniform_struct_0_1_.f4_;
    let _e3622 = textureSampleLevel(tex_0_13_, sampler_0_16_, vec3<f32>(((_e164 * _e3608) - _e186), ((_e165 * _e3608) - _e187), ((_e166 * _e3608) - _e188)), (_e274 * _e3617));
    let _e3628 = textureSample(tex_0_14_, sampler_0_16_, vec3<f32>(_e164, _e165, _e166));
    let _e3634 = textureSample(tex_0_15_, sampler_0_16_, vec2<f32>(_e195, _e274));
    let _e3639 = uniform_struct_0_1_.f3_;
    let _e3641 = (1f - _e765);
    let _e3651 = (_e3622.x * ((mix(_e3600, _e779, _e765) * _e3634.x) + _e3634.y));
    let _e3652 = (_e3622.y * ((mix(_e3600, _e780, _e765) * _e3634.x) + _e3634.y));
    let _e3653 = (_e3622.z * ((mix(_e3600, _e781, _e765) * _e3634.x) + _e3634.y));
    let _e3657 = max((1f - _e3634.y), 0f);
    let _e3670 = (_e680.x * (_e3639 * (_e3651 + (((_e3641 * _e776) * _e3628.x) * _e3657))));
    let _e3671 = (_e680.x * (_e3639 * (_e3652 + (((_e3641 * _e777) * _e3628.y) * _e3657))));
    let _e3672 = (_e680.x * (_e3639 * (_e3653 + (((_e3641 * _e778) * _e3628.z) * _e3657))));
    let _e3685 = !((bitcast<f32>(_e517) >= 0.5f));
    let _e3686 = select(_e3589, (((_e3498 + _e3538) + _e3589) + _e3670), _e3685);
    let _e3687 = select(_e3591, (((_e3500 + _e3539) + _e3591) + _e3671), _e3685);
    let _e3688 = select(_e3593, (((_e3502 + _e3540) + _e3593) + _e3672), _e3685);
    let _e3692 = ((_e3686 <= _e3686) && (0f <= 0f));
    if _e3692 {
        if (abs(_e3686) < 100000000000000000000f) {
            _2_u002e_sroa_u002e_0_u002e_0_ = _e3686;
        } else {
            _2_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
    } else {
        _2_u002e_sroa_u002e_0_u002e_0_ = 0f;
    }
    let _e3702 = _2_u002e_sroa_u002e_0_u002e_0_;
    let _e3706 = ((_e3687 <= _e3687) && (0f <= 0f));
    if _e3706 {
        if (abs(_e3687) < 100000000000000000000f) {
            _3_u002e_sroa_u002e_0_u002e_0_ = _e3687;
        } else {
            _3_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
    } else {
        _3_u002e_sroa_u002e_0_u002e_0_ = 0f;
    }
    let _e3716 = _3_u002e_sroa_u002e_0_u002e_0_;
    let _e3720 = ((_e3688 <= _e3688) && (0f <= 0f));
    if _e3720 {
        if (abs(_e3688) < 100000000000000000000f) {
            _4_u002e_sroa_u002e_0_u002e_0_ = _e3688;
        } else {
            _4_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
    } else {
        _4_u002e_sroa_u002e_0_u002e_0_ = 0f;
    }
    let _e3730 = _4_u002e_sroa_u002e_0_u002e_0_;
    if _e3692 {
        if !((abs(_e3686) < 100000000000000000000f)) {
            x_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
        } else {
            x_bad_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
    } else {
        x_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
    }
    let _e3742 = x_bad_u002e_sroa_u002e_0_u002e_0_;
    if _e3706 {
        if !((abs(_e3687) < 100000000000000000000f)) {
            y_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
        } else {
            y_bad_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
    } else {
        y_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
    }
    let _e3754 = y_bad_u002e_sroa_u002e_0_u002e_0_;
    if _e3720 {
    } else {
        z_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
        let _e3758 = z_bad_u002e_sroa_u002e_0_u002e_0_;
        let _e3763 = select(0f, 1f, (_e869 == 7u));
        let _e3783 = select(0f, 1f, (_e869 == 3u));
        let _e3788 = select(0f, 1f, (_e869 == 2u));
        let _e3793 = select(0f, 1f, (_e869 == 1u));
        let _e3798 = select(0f, 1f, (_e869 == 0u));
        let _e3800 = (_e3639 * _e3653);
        let _e3804 = (_e3639 * _e3652);
        let _e3808 = (_e3639 * _e3651);
        let _e3813 = select(_e3758, _e3754, (_e3754 > _e3758));
        let _e3815 = select(_e3813, _e3742, (_e3742 > _e3813));
        let _e3826 = select(0f, 1f, (dot(vec3<f32>(_e3702, _e3716, _e3730), vec3<f32>(0.2126f, 0.7152f, 0.0722f)) > 1f));
        let _e3860 = (_e274 * select(0f, 1f, (_e869 == 4u)));
        let _e3864 = (select(0f, 1f, (_e869 == 5u)) * _e765);
        let _e3868 = (select(0f, 1f, (_e869 == 6u)) * _e3516);
        fragment_out = vec4<f32>(((_e3868 + (_e3864 + (_e3860 + ((_e3783 * ((_e3510 + _e3538) + (_e3670 - _e3808))) + ((_e3788 * _e3808) + ((_e3793 * _e3504) + (_e3798 * _e3702))))))) + (_e3763 * mix(mix(_e3702, 1f, _e3826), 1f, _e3815))), ((_e3868 + (_e3864 + (_e3860 + ((_e3783 * ((_e3512 + _e3539) + (_e3671 - _e3804))) + ((_e3788 * _e3804) + ((_e3793 * _e3506) + (_e3798 * _e3716))))))) + (_e3763 * mix(mix(_e3716, 0f, _e3826), 0f, _e3815))), ((_e3868 + (_e3864 + (_e3860 + ((_e3783 * ((_e3514 + _e3540) + (_e3672 - _e3800))) + ((_e3788 * _e3800) + ((_e3793 * _e3508) + (_e3798 * _e3730))))))) + (_e3763 * mix(mix(_e3730, 0f, _e3826), 1f, _e3815))), bitcast<f32>(_e507));
        let _e3881 = fragment_out;
        return _e3881;
    }
    if !((abs(_e3688) < 100000000000000000000f)) {
        z_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
        let _e3889 = z_bad_u002e_sroa_u002e_0_u002e_0_;
        let _e3894 = select(0f, 1f, (_e869 == 7u));
        let _e3914 = select(0f, 1f, (_e869 == 3u));
        let _e3919 = select(0f, 1f, (_e869 == 2u));
        let _e3924 = select(0f, 1f, (_e869 == 1u));
        let _e3929 = select(0f, 1f, (_e869 == 0u));
        let _e3931 = (_e3639 * _e3653);
        let _e3935 = (_e3639 * _e3652);
        let _e3939 = (_e3639 * _e3651);
        let _e3944 = select(_e3889, _e3754, (_e3754 > _e3889));
        let _e3946 = select(_e3944, _e3742, (_e3742 > _e3944));
        let _e3957 = select(0f, 1f, (dot(vec3<f32>(_e3702, _e3716, _e3730), vec3<f32>(0.2126f, 0.7152f, 0.0722f)) > 1f));
        let _e3991 = (_e274 * select(0f, 1f, (_e869 == 4u)));
        let _e3995 = (select(0f, 1f, (_e869 == 5u)) * _e765);
        let _e3999 = (select(0f, 1f, (_e869 == 6u)) * _e3516);
        fragment_out = vec4<f32>(((_e3999 + (_e3995 + (_e3991 + ((_e3914 * ((_e3510 + _e3538) + (_e3670 - _e3939))) + ((_e3919 * _e3939) + ((_e3924 * _e3504) + (_e3929 * _e3702))))))) + (_e3894 * mix(mix(_e3702, 1f, _e3957), 1f, _e3946))), ((_e3999 + (_e3995 + (_e3991 + ((_e3914 * ((_e3512 + _e3539) + (_e3671 - _e3935))) + ((_e3919 * _e3935) + ((_e3924 * _e3506) + (_e3929 * _e3716))))))) + (_e3894 * mix(mix(_e3716, 0f, _e3957), 0f, _e3946))), ((_e3999 + (_e3995 + (_e3991 + ((_e3914 * ((_e3514 + _e3540) + (_e3672 - _e3931))) + ((_e3919 * _e3931) + ((_e3924 * _e3508) + (_e3929 * _e3730))))))) + (_e3894 * mix(mix(_e3730, 0f, _e3957), 1f, _e3946))), bitcast<f32>(_e507));
        let _e4012 = fragment_out;
        return _e4012;
    }
    z_bad_u002e_sroa_u002e_0_u002e_0_ = 0f;
    let _e4016 = z_bad_u002e_sroa_u002e_0_u002e_0_;
    let _e4021 = select(0f, 1f, (_e869 == 7u));
    let _e4041 = select(0f, 1f, (_e869 == 3u));
    let _e4046 = select(0f, 1f, (_e869 == 2u));
    let _e4051 = select(0f, 1f, (_e869 == 1u));
    let _e4056 = select(0f, 1f, (_e869 == 0u));
    let _e4058 = (_e3639 * _e3653);
    let _e4062 = (_e3639 * _e3652);
    let _e4066 = (_e3639 * _e3651);
    let _e4071 = select(_e4016, _e3754, (_e3754 > _e4016));
    let _e4073 = select(_e4071, _e3742, (_e3742 > _e4071));
    let _e4084 = select(0f, 1f, (dot(vec3<f32>(_e3702, _e3716, _e3730), vec3<f32>(0.2126f, 0.7152f, 0.0722f)) > 1f));
    let _e4118 = (_e274 * select(0f, 1f, (_e869 == 4u)));
    let _e4122 = (select(0f, 1f, (_e869 == 5u)) * _e765);
    let _e4126 = (select(0f, 1f, (_e869 == 6u)) * _e3516);
    fragment_out = vec4<f32>(((_e4126 + (_e4122 + (_e4118 + ((_e4041 * ((_e3510 + _e3538) + (_e3670 - _e4066))) + ((_e4046 * _e4066) + ((_e4051 * _e3504) + (_e4056 * _e3702))))))) + (_e4021 * mix(mix(_e3702, 1f, _e4084), 1f, _e4073))), ((_e4126 + (_e4122 + (_e4118 + ((_e4041 * ((_e3512 + _e3539) + (_e3671 - _e4062))) + ((_e4046 * _e4062) + ((_e4051 * _e3506) + (_e4056 * _e3716))))))) + (_e4021 * mix(mix(_e3716, 0f, _e4084), 0f, _e4073))), ((_e4126 + (_e4122 + (_e4118 + ((_e4041 * ((_e3514 + _e3540) + (_e3672 - _e4058))) + ((_e4046 * _e4058) + ((_e4051 * _e3508) + (_e4056 * _e3730))))))) + (_e4021 * mix(mix(_e3730, 0f, _e4084), 1f, _e4073))), bitcast<f32>(_e507));
    let _e4139 = fragment_out;
    return _e4139;
}
