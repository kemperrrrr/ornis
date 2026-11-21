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

@group(0) @binding(0) 
var<uniform> uniform_struct_0_0_: UniformStruct0_;
@group(0) @binding(2) 
var<storage> storage_0_2_: array<u32>;
@group(0) @binding(16) 
var tex_0_16_: texture_2d<f32>;
@group(0) @binding(19) 
var sampler_0_19_: sampler;
@group(0) @binding(17) 
var tex_0_17_: texture_2d<f32>;
@group(0) @binding(18) 
var tex_0_18_: texture_2d<f32>;
@group(0) @binding(12) 
var<uniform> uniform_struct_0_12_: UniformStruct5_;
@group(0) @binding(3) 
var<uniform> uniform_struct_0_3_: UniformStruct4_;
@group(0) @binding(15) 
var<storage> storage_0_15_: array<u32>;
@group(0) @binding(14) 
var<storage> storage_0_14_: array<u32>;
@group(0) @binding(13) 
var<storage> storage_0_13_: array<u32>;
@group(0) @binding(11) 
var tex_0_11_: texture_depth_2d_array;
@group(0) @binding(5) 
var sampler_0_5_: sampler_comparison;
@group(0) @binding(4) 
var tex_0_4_: texture_depth_2d_array;
@group(0) @binding(6) 
var tex_0_6_: texture_depth_cube_array;
@group(0) @binding(7) 
var tex_0_7_: texture_cube<f32>;
@group(0) @binding(10) 
var sampler_0_10_: sampler;
@group(0) @binding(8) 
var tex_0_8_: texture_cube<f32>;
@group(0) @binding(9) 
var tex_0_9_: texture_2d<f32>;

@fragment 
fn fs_main_textured(@builtin(position) position: vec4<f32>, @location(0) in0_: vec3<f32>, @location(1) in1_: vec3<f32>, @location(2) in2_: vec2<f32>, @location(3) in3_: vec3<f32>, @location(4) @interpolate(flat) in4_: u32) -> @location(0) vec4<f32> {
    var fragment_out: vec4<f32>;
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i_u002e_i: u32;
    var radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_: f32;
    var vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i: f32;
    var lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa: f32;
    var lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa: f32;
    var extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_: f32;
    var extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_: f32;
    var emission_u002e_i_u002e_sroa_u002e_0_u002e_1_: f32;
    var emission_u002e_i_u002e_sroa_u002e_7_u002e_1_: f32;
    var emission_u002e_i_u002e_sroa_u002e_10_u002e_1_: f32;
    var subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_: f32;
    var subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_: f32;
    var transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_: f32;
    var transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_: f32;
    var _4_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var coat_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var coat_u002e_i_u002e_sroa_u002e_5_u002e_0_: f32;
    var coat_u002e_i_u002e_sroa_u002e_8_u002e_0_: f32;
    var fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_: f32;
    var fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_: f32;
    var coat_darken_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var coat_darken_u002e_i_u002e_sroa_u002e_5_u002e_0_: f32;
    var coat_darken_u002e_i_u002e_sroa_u002e_7_u002e_0_: f32;
    var k_u002e_sroa_u002e_0_u002e_0_: u32;
    var lo_u002e_i_u002e_sroa_u002e_9_: f32;
    var lo_u002e_i_u002e_sroa_u002e_6_: f32;
    var lo_u002e_i_u002e_sroa_u002e_0_: f32;
    var _2_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var lo_u002e_i_u002e_sroa_u002e_0_u002e_1_: f32;
    var lo_u002e_i_u002e_sroa_u002e_6_u002e_1_: f32;
    var lo_u002e_i_u002e_sroa_u002e_9_u002e_1_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_10_u002e_0_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_15_u002e_0_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_10_u002e_1_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_15_u002e_1_: f32;
    var _3_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var out_u002e_lo_u002e_i_u002e_sroa_u002e_0_u002e_1_: f32;
    var out_u002e_lo_u002e_i_u002e_sroa_u002e_6_u002e_1_: f32;
    var out_u002e_lo_u002e_i_u002e_sroa_u002e_9_u002e_1_: f32;

    let _e15 = (1f / length(vec3<f32>(in1_.x, in1_.y, in1_.z)));
    let _e16 = (in1_.x * _e15);
    let _e17 = (in1_.y * _e15);
    let _e18 = (in1_.z * _e15);
    let _e22 = uniform_struct_0_0_.f8_.x;
    let _e26 = uniform_struct_0_0_.f8_.y;
    let _e30 = uniform_struct_0_0_.f8_.z;
    let _e31 = (_e22 - in0_.x);
    let _e32 = (_e26 - in0_.y);
    let _e33 = (_e30 - in0_.z);
    let _e37 = (1f / length(vec3<f32>(_e31, _e32, _e33)));
    let _e38 = (_e31 * _e37);
    let _e39 = (_e32 * _e37);
    let _e40 = (_e33 * _e37);
    let _e43 = dot(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e38, _e39, _e40));
    let _e47 = select(_e43, 0.000001f, (_e43 < 0.000001f));
    let _e54 = (1f / length(vec3<f32>(in3_.x, in3_.y, in3_.z)));
    let _e55 = (in3_.x * _e54);
    let _e56 = (in3_.y * _e54);
    let _e57 = (in3_.z * _e54);
    let _e71 = (in4_ * 80u);
    let _e74 = storage_0_2_[_e71];
    let _e75 = bitcast<f32>(_e74);
    let _e80 = storage_0_2_[(_e71 | 1u)];
    let _e81 = bitcast<f32>(_e80);
    let _e86 = storage_0_2_[(_e71 | 2u)];
    let _e92 = storage_0_2_[(_e71 | 4u)];
    let _e98 = storage_0_2_[(_e71 | 5u)];
    let _e104 = storage_0_2_[(_e71 | 6u)];
    let _e110 = storage_0_2_[(_e71 | 8u)];
    let _e111 = bitcast<f32>(_e110);
    let _e113 = (_e71 | 9u);
    let _e116 = storage_0_2_[_e113];
    let _e117 = bitcast<f32>(_e116);
    let _e118 = dpdx(_e16);
    let _e119 = dpdx(_e17);
    let _e120 = dpdx(_e18);
    let _e121 = dpdy(_e16);
    let _e122 = dpdy(_e17);
    let _e123 = dpdy(_e18);
    let _e134 = (((dot(vec3<f32>(_e118, _e119, _e120), vec3<f32>(_e118, _e119, _e120)) + dot(vec3<f32>(_e121, _e122, _e123), vec3<f32>(_e121, _e122, _e123))) * 0.15f) * 2f);
    let _e139 = (_e117 * _e117);
    let _e141 = ((_e139 * _e139) + select(0.2f, _e134, (_e134 < 0.2f)));
    let _e145 = select(_e141, 0f, (_e141 < 0f));
    let _e149 = select(_e145, 1f, (_e145 > 1f));
    let _e158 = storage_0_2_[_e113];
    let _e164 = storage_0_2_[(_e71 | 10u)];
    let _e165 = bitcast<f32>(_e164);
    let _e170 = storage_0_2_[(_e71 | 11u)];
    let _e176 = storage_0_2_[(_e71 | 12u)];
    let _e177 = bitcast<f32>(_e176);
    let _e182 = storage_0_2_[(_e71 | 13u)];
    let _e183 = bitcast<f32>(_e182);
    let _e188 = storage_0_2_[(_e71 | 14u)];
    let _e189 = bitcast<f32>(_e188);
    let _e194 = storage_0_2_[(_e71 + 16u)];
    let _e195 = bitcast<f32>(_e194);
    let _e200 = storage_0_2_[(_e71 + 17u)];
    let _e201 = bitcast<f32>(_e200);
    let _e216 = storage_0_2_[(_e71 + 20u)];
    let _e217 = bitcast<f32>(_e216);
    let _e222 = storage_0_2_[(_e71 + 21u)];
    let _e223 = bitcast<f32>(_e222);
    let _e228 = storage_0_2_[(_e71 + 22u)];
    let _e229 = bitcast<f32>(_e228);
    let _e254 = storage_0_2_[(_e71 + 28u)];
    let _e255 = bitcast<f32>(_e254);
    let _e260 = storage_0_2_[(_e71 + 29u)];
    let _e261 = bitcast<f32>(_e260);
    let _e266 = storage_0_2_[(_e71 + 30u)];
    let _e272 = storage_0_2_[(_e71 + 31u)];
    let _e278 = storage_0_2_[(_e71 + 32u)];
    let _e279 = bitcast<f32>(_e278);
    let _e284 = storage_0_2_[(_e71 + 33u)];
    let _e285 = bitcast<f32>(_e284);
    let _e290 = storage_0_2_[(_e71 + 34u)];
    let _e291 = bitcast<f32>(_e290);
    let _e296 = storage_0_2_[(_e71 + 36u)];
    let _e302 = storage_0_2_[(_e71 + 37u)];
    let _e308 = storage_0_2_[(_e71 + 40u)];
    let _e309 = bitcast<f32>(_e308);
    let _e314 = storage_0_2_[(_e71 + 41u)];
    let _e315 = bitcast<f32>(_e314);
    let _e320 = storage_0_2_[(_e71 + 44u)];
    let _e326 = storage_0_2_[(_e71 + 45u)];
    let _e332 = storage_0_2_[(_e71 + 46u)];
    let _e338 = storage_0_2_[(_e71 + 48u)];
    let _e339 = bitcast<f32>(_e338);
    let _e344 = storage_0_2_[(_e71 + 49u)];
    let _e345 = bitcast<f32>(_e344);
    let _e350 = storage_0_2_[(_e71 + 50u)];
    let _e356 = storage_0_2_[(_e71 + 51u)];
    let _e361 = storage_0_2_[(_e71 + 52u)];
    let _e362 = bitcast<f32>(_e361);
    let _e367 = storage_0_2_[(_e71 + 53u)];
    let _e368 = bitcast<f32>(_e367);
    let _e373 = storage_0_2_[(_e71 + 54u)];
    let _e374 = bitcast<f32>(_e373);
    let _e379 = storage_0_2_[(_e71 + 56u)];
    let _e380 = bitcast<f32>(_e379);
    let _e385 = storage_0_2_[(_e71 + 60u)];
    let _e391 = storage_0_2_[(_e71 + 61u)];
    let _e397 = storage_0_2_[(_e71 + 62u)];
    let _e398 = bitcast<f32>(_e397);
    let _e403 = storage_0_2_[(_e71 + 64u)];
    let _e404 = bitcast<f32>(_e403);
    let _e409 = storage_0_2_[(_e71 + 68u)];
    let _e415 = storage_0_2_[(_e71 + 69u)];
    let _e421 = storage_0_2_[(_e71 + 70u)];
    let _e427 = storage_0_2_[(_e71 + 72u)];
    let _e437 = storage_0_2_[(_e71 + 74u)];
    let _e444 = textureSample(tex_0_16_, sampler_0_19_, vec2<f32>(in2_.x, in2_.y));
    let _e449 = textureSample(tex_0_16_, sampler_0_19_, vec2<f32>(in2_.x, in2_.y));
    let _e454 = textureSample(tex_0_16_, sampler_0_19_, vec2<f32>(in2_.x, in2_.y));
    let _e459 = textureSample(tex_0_17_, sampler_0_19_, vec2<f32>(in2_.x, in2_.y));
    let _e464 = textureSample(tex_0_17_, sampler_0_19_, vec2<f32>(in2_.x, in2_.y));
    let _e469 = textureSample(tex_0_18_, sampler_0_19_, vec2<f32>(in2_.x, in2_.y));
    let _e474 = textureSample(tex_0_18_, sampler_0_19_, vec2<f32>(in2_.x, in2_.y));
    let _e479 = textureSample(tex_0_18_, sampler_0_19_, vec2<f32>(in2_.x, in2_.y));
    let _e481 = (_e444.x * bitcast<f32>(_e92));
    let _e482 = (_e449.y * bitcast<f32>(_e98));
    let _e483 = (_e454.z * bitcast<f32>(_e104));
    let _e484 = (_e464.z * bitcast<f32>(_e86));
    let _e485 = (_e459.y * bitcast<f32>(_e158));
    let _e486 = dpdx(_e16);
    let _e487 = dpdx(_e17);
    let _e488 = dpdx(_e18);
    let _e489 = dpdy(_e16);
    let _e490 = dpdy(_e17);
    let _e491 = dpdy(_e18);
    let _e502 = (((dot(vec3<f32>(_e486, _e487, _e488), vec3<f32>(_e486, _e487, _e488)) + dot(vec3<f32>(_e489, _e490, _e491), vec3<f32>(_e489, _e490, _e491))) * 0.15f) * 2f);
    let _e507 = (_e485 * _e485);
    let _e509 = ((_e507 * _e507) + select(0.2f, _e502, (_e502 < 0.2f)));
    let _e513 = select(_e509, 0f, (_e509 < 0f));
    let _e517 = select(_e513, 1f, (_e513 > 1f));
    let _e523 = sqrt(sqrt(select(0f, _e517, (_e517 > 0f))));
    let _e530 = uniform_struct_0_0_.f0_.x;
    let _e534 = uniform_struct_0_0_.f1_.x;
    let _e538 = uniform_struct_0_0_.f2_.x;
    let _e542 = uniform_struct_0_0_.f3_.x;
    let _e552 = uniform_struct_0_0_.f0_.y;
    let _e556 = uniform_struct_0_0_.f1_.y;
    let _e560 = uniform_struct_0_0_.f2_.y;
    let _e564 = uniform_struct_0_0_.f3_.y;
    let _e574 = uniform_struct_0_0_.f0_.w;
    let _e578 = uniform_struct_0_0_.f1_.w;
    let _e582 = uniform_struct_0_0_.f2_.w;
    let _e586 = uniform_struct_0_0_.f3_.w;
    let _e592 = (_e586 + (((in0_.x * _e574) + (in0_.y * _e578)) + (in0_.z * _e582)));
    let _e596 = select(0.5f, _e592, (_e592 > 0.5f));
    let _e637 = ((((min(u32(floor((((((_e564 + (((in0_.x * _e552) + (in0_.y * _e556)) + (in0_.z * _e560))) / _e596) * 0.5f) + 0.5f) * 9f))), 8u) << 4u) | min(u32(floor((((((_e542 + (((in0_.x * _e530) + (in0_.y * _e534)) + (in0_.z * _e538))) / _e596) * 0.5f) + 0.5f) * 16f))), 15u)) + (min(u32(ceil(((log((_e596 * 2f)) * 3.4407384f) + -0.0001f))), 23u) * 144u)) << 1u);
    let _e640 = uniform_struct_0_12_.f2_;
    let _e643 = uniform_struct_0_3_.f2_;
    let _e646 = storage_0_15_[_e637];
    let _e652 = storage_0_15_[(_e637 | 1u)];
    let _e656 = (_e47 * _e47);
    let _e660 = max((1f - _e656), 0f);
    let _e662 = (sqrt(_e660) / _e398);
    let _e675 = pow(((_e398 + -1f) / (_e398 + 1f)), 2f);
    let _e679 = (((bitcast<f32>(_e391) * 1000f) * (_e398 * 12.566371f)) * sqrt(max((1f - (_e662 * _e662)), 0f)));
    let _e690 = (_e675 * 2f);
    let _e696 = (1f - (_e675 * _e675));
    let _e708 = clamp(bitcast<f32>(_e385), 0f, 1f);
    let _e713 = ((_e708 * ((((_e690 * cos((_e679 / 650f))) / _e696) + 1f) + -1f)) + 1f);
    let _e718 = ((_e708 * ((((_e690 * cos((_e679 / 550f))) / _e696) + 1f) + -1f)) + 1f);
    let _e723 = ((_e708 * ((((_e690 * cos((_e679 / 450f))) / _e696) + 1f) + -1f)) + 1f);
    let _e728 = ((_e380 + -1f) / (_e380 + 1f));
    let _e731 = (_e481 * _e75);
    let _e732 = (_e482 * _e75);
    let _e733 = (_e483 * _e75);
    let _e736 = !((_e339 <= 0f));
    if _e736 {
        let _e742 = (1f - ((1f - pow(_e728, 2f)) / (_e380 * _e380)));
        let _e744 = (1f - _e742);
        let _e745 = (_e733 * _e111);
        let _e758 = (_e732 * _e111);
        let _e771 = (_e731 * _e111);
        let _e785 = (_e339 * bitcast<f32>(_e356));
        coat_darken_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e785 * ((_e744 / max((1f - ((_e771 + (((((_e481 - _e279) * _e255) + _e279) - _e771) * _e484)) * _e742)), 0.000001f)) + -1f)) + 1f);
        coat_darken_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e785 * ((_e744 / max((1f - ((_e758 + (_e484 * ((((_e482 - _e285) * _e255) + _e285) - _e758))) * _e742)), 0.000001f)) + -1f)) + 1f);
        coat_darken_u002e_i_u002e_sroa_u002e_7_u002e_0_ = ((_e785 * ((_e744 / max((1f - ((_e745 + (_e484 * ((((_e483 - _e291) * _e255) + _e291) - _e745))) * _e742)), 0.000001f)) + -1f)) + 1f);
    } else {
        coat_darken_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 1f;
        coat_darken_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 1f;
        coat_darken_u002e_i_u002e_sroa_u002e_7_u002e_0_ = 1f;
    }
    let _e811 = coat_darken_u002e_i_u002e_sroa_u002e_0_u002e_0_;
    let _e813 = coat_darken_u002e_i_u002e_sroa_u002e_5_u002e_0_;
    let _e815 = coat_darken_u002e_i_u002e_sroa_u002e_7_u002e_0_;
    let _e816 = ((_e652 * _e640) + _e643);
    if (_e816 == 0u) {
        lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
        lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = 0f;
        lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = 0f;
    } else {
        let _e833 = ((_e165 + -1f) / (_e165 + 1f));
        let _e834 = (_e833 * _e833);
        let _e835 = (_e523 * _e523);
        let _e837 = (1f - bitcast<f32>(_e170));
        let _e842 = (2f / ((_e837 * _e837) + 1f));
        let _e844 = select(_e523, _e81, (_e523 < _e81));
        let _e848 = (_e38 - (_e16 * _e47));
        let _e849 = (_e39 - (_e17 * _e47));
        let _e850 = (_e40 - (_e18 * _e47));
        let _e853 = (1f - _e484);
        let _e858 = max((_e844 * _e844), 0.000001f);
        let _e859 = (_e858 * _e858);
        let _e874 = (_e728 * _e728);
        let _e877 = (_e345 * _e345);
        let _e879 = (1f - bitcast<f32>(_e350));
        let _e884 = (2f / ((_e879 * _e879) + 1f));
        let _e888 = (_e315 * _e315);
        k_u002e_sroa_u002e_0_u002e_0_ = 0u;
        lo_u002e_i_u002e_sroa_u002e_9_ = 0f;
        lo_u002e_i_u002e_sroa_u002e_6_ = 0f;
        lo_u002e_i_u002e_sroa_u002e_0_ = 0f;
        loop {
            let _e938 = k_u002e_sroa_u002e_0_u002e_0_;
            let _e940 = lo_u002e_i_u002e_sroa_u002e_9_;
            let _e942 = lo_u002e_i_u002e_sroa_u002e_6_;
            let _e944 = lo_u002e_i_u002e_sroa_u002e_0_;
            if (_e938 < _e643) {
                _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i_u002e_i = _e938;
            } else {
                let _e950 = storage_0_14_[(((_e646 * _e640) - _e643) + _e938)];
                _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i_u002e_i = _e950;
            }
            let _e953 = _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i_u002e_i;
            let _e955 = (_e953 * 36u);
            let _e958 = storage_0_13_[_e955];
            let _e959 = bitcast<f32>(_e958);
            let _e961 = (_e955 + 8u);
            let _e964 = storage_0_13_[_e961];
            let _e967 = (_e955 + 9u);
            let _e970 = storage_0_13_[_e967];
            let _e973 = (_e955 + 10u);
            let _e976 = storage_0_13_[_e973];
            let _e982 = length(vec3<f32>((bitcast<f32>(_e964) - in0_.x), (bitcast<f32>(_e970) - in0_.y), (bitcast<f32>(_e976) - in0_.z)));
            let _e987 = select(1f, 0f, (_e959 < 0.5f));
            let _e989 = (_e955 + 16u);
            let _e992 = storage_0_13_[_e989];
            let _e993 = bitcast<f32>(_e992);
            let _e1005 = storage_0_13_[(_e955 + 12u)];
            let _e1011 = storage_0_13_[(_e955 + 13u)];
            let _e1017 = storage_0_13_[(_e955 + 14u)];
            let _e1023 = storage_0_13_[(_e955 + 15u)];
            let _e1024 = bitcast<f32>(_e1023);
            let _e1027 = storage_0_13_[_e961];
            let _e1031 = storage_0_13_[_e967];
            let _e1035 = storage_0_13_[_e973];
            let _e1037 = (in0_.x - bitcast<f32>(_e1027));
            let _e1038 = (in0_.y - bitcast<f32>(_e1031));
            let _e1039 = (in0_.z - bitcast<f32>(_e1035));
            let _e1043 = (1f / length(vec3<f32>(_e1037, _e1038, _e1039)));
            let _e1048 = (_e955 + 4u);
            let _e1051 = storage_0_13_[_e1048];
            let _e1052 = bitcast<f32>(_e1051);
            let _e1054 = (_e955 + 5u);
            let _e1057 = storage_0_13_[_e1054];
            let _e1058 = bitcast<f32>(_e1057);
            let _e1060 = (_e955 + 6u);
            let _e1063 = storage_0_13_[_e1060];
            let _e1064 = bitcast<f32>(_e1063);
            let _e1068 = (1f / length(vec3<f32>(_e1052, _e1058, _e1064)));
            let _e1079 = storage_0_13_[(_e955 + 18u)];
            let _e1085 = storage_0_13_[(_e955 + 17u)];
            let _e1092 = select(1f, 0f, (_e959 < 1.5f));
            let _e1094 = mix(1f, smoothstep(bitcast<f32>(_e1079), bitcast<f32>(_e1085), dot(vec3<f32>((_e1068 * _e1052), (_e1068 * _e1058), (_e1068 * _e1064)), vec3<f32>((_e1037 * _e1043), (_e1038 * _e1043), (_e1039 * _e1043)))), _e1092);
            let _e1095 = (_e982 * _e982);
            let _e1107 = (mix(1f, (1f - smoothstep((_e993 * 0.99f), _e993, _e982)), _e987) * mix(1f, (1f / select(_e1095, 0.000001f, (_e1095 < 0.000001f))), _e987));
            let _e1111 = (_e1094 * ((bitcast<f32>(_e1005) * _e1024) * _e1107));
            let _e1112 = (_e1094 * ((bitcast<f32>(_e1011) * _e1024) * _e1107));
            let _e1113 = (_e1094 * ((bitcast<f32>(_e1017) * _e1024) * _e1107));
            let _e1116 = (_e987 * (1f - _e1092));
            if (((1f - _e987) * select(1f, 0f, (_e993 < 0.5f))) > 0.5f) {
                let _e1131 = uniform_struct_0_3_.f6_.f1_.x;
                let _e1136 = uniform_struct_0_3_.f6_.f1_.y;
                let _e1141 = uniform_struct_0_3_.f6_.f1_.z;
                let _e1144 = dot(vec3<f32>((in0_.x - _e22), (in0_.y - _e26), (in0_.z - _e30)), vec3<f32>(_e1131, _e1136, _e1141));
                let _e1149 = uniform_struct_0_3_.f6_.f2_.x;
                let _e1154 = uniform_struct_0_3_.f6_.f2_.y;
                let _e1159 = uniform_struct_0_3_.f6_.f2_.z;
                let _e1164 = uniform_struct_0_3_.f6_.f2_.w;
                let _e1168 = uniform_struct_0_3_.f6_.f3_;
                let _e1172 = select(1f, 0f, (_e1144 < _e1154));
                let _e1176 = select(1f, 0f, (_e1144 < _e1159));
                let _e1180 = select(1f, 0f, (_e1144 < _e1164));
                let _e1183 = u32(((_e1172 + _e1176) + _e1180));
                let _e1186 = mix(mix(mix(_e1154, _e1159, _e1172), _e1164, _e1176), _e1168, _e1180);
                let _e1196 = uniform_struct_0_3_.f6_.f0_[_e1183].f0_.x;
                let _e1203 = uniform_struct_0_3_.f6_.f0_[_e1183].f1_.x;
                let _e1210 = uniform_struct_0_3_.f6_.f0_[_e1183].f2_.x;
                let _e1217 = uniform_struct_0_3_.f6_.f0_[_e1183].f3_.x;
                let _e1230 = uniform_struct_0_3_.f6_.f0_[_e1183].f0_.y;
                let _e1237 = uniform_struct_0_3_.f6_.f0_[_e1183].f1_.y;
                let _e1244 = uniform_struct_0_3_.f6_.f0_[_e1183].f2_.y;
                let _e1251 = uniform_struct_0_3_.f6_.f0_[_e1183].f3_.y;
                let _e1264 = uniform_struct_0_3_.f6_.f0_[_e1183].f0_.z;
                let _e1271 = uniform_struct_0_3_.f6_.f0_[_e1183].f1_.z;
                let _e1278 = uniform_struct_0_3_.f6_.f0_[_e1183].f2_.z;
                let _e1285 = uniform_struct_0_3_.f6_.f0_[_e1183].f3_.z;
                let _e1298 = uniform_struct_0_3_.f6_.f0_[_e1183].f0_.w;
                let _e1305 = uniform_struct_0_3_.f6_.f0_[_e1183].f1_.w;
                let _e1312 = uniform_struct_0_3_.f6_.f0_[_e1183].f2_.w;
                let _e1319 = uniform_struct_0_3_.f6_.f0_[_e1183].f3_.w;
                let _e1325 = (_e1319 + (((in0_.x * _e1298) + (in0_.y * _e1305)) + (in0_.z * _e1312)));
                let _e1342 = textureSampleCompareLevel(tex_0_11_, sampler_0_5_, vec2<f32>(((((_e1217 + (((in0_.x * _e1196) + (in0_.y * _e1203)) + (in0_.z * _e1210))) / _e1325) * 0.5f) + 0.5f), (0.5f - (((_e1251 + (((in0_.x * _e1230) + (in0_.y * _e1237)) + (in0_.z * _e1244))) / _e1325) * 0.5f))), _e1183, (((_e1285 + (((in0_.x * _e1264) + (in0_.y * _e1271)) + (in0_.z * _e1278))) / _e1325) + -0.001f));
                if (_e1183 < 3u) {
                    let _e1347 = ((_e1186 - mix(mix(mix(_e1149, _e1154, _e1172), _e1159, _e1176), _e1164, _e1180)) * 0.08f);
                    let _e1353 = ((_e1186 - _e1144) / select(_e1347, 0.000001f, (_e1347 < 0.000001f)));
                    let _e1357 = select(_e1353, 0f, (_e1353 < 0f));
                    let _e1363 = (_e1183 + 1u);
                    let _e1370 = uniform_struct_0_3_.f6_.f0_[_e1363].f0_.x;
                    let _e1377 = uniform_struct_0_3_.f6_.f0_[_e1363].f1_.x;
                    let _e1384 = uniform_struct_0_3_.f6_.f0_[_e1363].f2_.x;
                    let _e1391 = uniform_struct_0_3_.f6_.f0_[_e1363].f3_.x;
                    let _e1404 = uniform_struct_0_3_.f6_.f0_[_e1363].f0_.y;
                    let _e1411 = uniform_struct_0_3_.f6_.f0_[_e1363].f1_.y;
                    let _e1418 = uniform_struct_0_3_.f6_.f0_[_e1363].f2_.y;
                    let _e1425 = uniform_struct_0_3_.f6_.f0_[_e1363].f3_.y;
                    let _e1438 = uniform_struct_0_3_.f6_.f0_[_e1363].f0_.z;
                    let _e1445 = uniform_struct_0_3_.f6_.f0_[_e1363].f1_.z;
                    let _e1452 = uniform_struct_0_3_.f6_.f0_[_e1363].f2_.z;
                    let _e1459 = uniform_struct_0_3_.f6_.f0_[_e1363].f3_.z;
                    let _e1472 = uniform_struct_0_3_.f6_.f0_[_e1363].f0_.w;
                    let _e1479 = uniform_struct_0_3_.f6_.f0_[_e1363].f1_.w;
                    let _e1486 = uniform_struct_0_3_.f6_.f0_[_e1363].f2_.w;
                    let _e1493 = uniform_struct_0_3_.f6_.f0_[_e1363].f3_.w;
                    let _e1499 = (_e1493 + (((in0_.x * _e1472) + (in0_.y * _e1479)) + (in0_.z * _e1486)));
                    let _e1516 = textureSampleCompareLevel(tex_0_11_, sampler_0_5_, vec2<f32>(((((_e1391 + (((in0_.x * _e1370) + (in0_.y * _e1377)) + (in0_.z * _e1384))) / _e1499) * 0.5f) + 0.5f), (0.5f - (((_e1425 + (((in0_.x * _e1404) + (in0_.y * _e1411)) + (in0_.z * _e1418))) / _e1499) * 0.5f))), _e1363, (((_e1459 + (((in0_.x * _e1438) + (in0_.y * _e1445)) + (in0_.z * _e1452))) / _e1499) + -0.001f));
                    vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i = mix(_e1516, _e1342, select(_e1357, 1f, (_e1357 > 1f)));
                } else {
                    vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i = _e1342;
                }
                let _e1521 = vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i;
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (_e1111 * _e1521);
                radiance_u002e_i_u002e_sroa_u002e_10_u002e_0_ = (_e1112 * _e1521);
                radiance_u002e_i_u002e_sroa_u002e_15_u002e_0_ = (_e1113 * _e1521);
            } else {
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_ = _e1111;
                radiance_u002e_i_u002e_sroa_u002e_10_u002e_0_ = _e1112;
                radiance_u002e_i_u002e_sroa_u002e_15_u002e_0_ = _e1113;
            }
            let _e1532 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_;
            let _e1534 = radiance_u002e_i_u002e_sroa_u002e_10_u002e_0_;
            let _e1536 = radiance_u002e_i_u002e_sroa_u002e_15_u002e_0_;
            let _e1538 = (_e955 + 19u);
            let _e1541 = storage_0_13_[_e1538];
            if !((bitcast<f32>(_e1541) >= 0f)) {
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = _e1532;
                radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_ = _e1534;
                radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_ = _e1536;
            } else {
                if !((_e1116 <= 0.5f)) {
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e1532;
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_1_ = _e1534;
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_1_ = _e1536;
                } else {
                    let _e1559 = storage_0_13_[(_e955 + 20u)];
                    let _e1565 = storage_0_13_[(_e955 + 24u)];
                    let _e1571 = storage_0_13_[(_e955 + 28u)];
                    let _e1577 = storage_0_13_[(_e955 + 32u)];
                    let _e1589 = storage_0_13_[(_e955 + 21u)];
                    let _e1595 = storage_0_13_[(_e955 + 25u)];
                    let _e1601 = storage_0_13_[(_e955 + 29u)];
                    let _e1607 = storage_0_13_[(_e955 + 33u)];
                    let _e1619 = storage_0_13_[(_e955 + 22u)];
                    let _e1625 = storage_0_13_[(_e955 + 26u)];
                    let _e1631 = storage_0_13_[(_e955 + 30u)];
                    let _e1637 = storage_0_13_[(_e955 + 34u)];
                    let _e1649 = storage_0_13_[(_e955 + 23u)];
                    let _e1655 = storage_0_13_[(_e955 + 27u)];
                    let _e1661 = storage_0_13_[(_e955 + 31u)];
                    let _e1667 = storage_0_13_[(_e955 + 35u)];
                    let _e1674 = ((((in0_.x * bitcast<f32>(_e1649)) + (in0_.y * bitcast<f32>(_e1655))) + (in0_.z * bitcast<f32>(_e1661))) + bitcast<f32>(_e1667));
                    let _e1687 = storage_0_13_[_e1538];
                    let _e1696 = textureSampleCompareLevel(tex_0_4_, sampler_0_5_, vec2<f32>((((((((in0_.x * bitcast<f32>(_e1559)) + (in0_.y * bitcast<f32>(_e1565))) + (in0_.z * bitcast<f32>(_e1571))) + bitcast<f32>(_e1577)) / _e1674) * 0.5f) + 0.5f), (0.5f - ((((((in0_.x * bitcast<f32>(_e1589)) + (in0_.y * bitcast<f32>(_e1595))) + (in0_.z * bitcast<f32>(_e1601))) + bitcast<f32>(_e1607)) / _e1674) * 0.5f))), u32(bitcast<f32>(_e1687)), ((((((in0_.x * bitcast<f32>(_e1619)) + (in0_.y * bitcast<f32>(_e1625))) + (in0_.z * bitcast<f32>(_e1631))) + bitcast<f32>(_e1637)) / _e1674) + -0.001f));
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_ = (_e1532 * _e1696);
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_1_ = (_e1534 * _e1696);
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_1_ = (_e1536 * _e1696);
                }
                let _e1704 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_;
                let _e1706 = radiance_u002e_i_u002e_sroa_u002e_10_u002e_1_;
                let _e1708 = radiance_u002e_i_u002e_sroa_u002e_15_u002e_1_;
                if (_e1116 > 0.5f) {
                    let _e1713 = storage_0_13_[_e961];
                    let _e1717 = storage_0_13_[_e967];
                    let _e1721 = storage_0_13_[_e973];
                    let _e1723 = (in0_.x - bitcast<f32>(_e1713));
                    let _e1724 = (in0_.y - bitcast<f32>(_e1717));
                    let _e1725 = (in0_.z - bitcast<f32>(_e1721));
                    let _e1726 = abs(_e1723);
                    let _e1727 = abs(_e1724);
                    let _e1729 = select(_e1726, _e1727, (_e1727 > _e1726));
                    let _e1730 = abs(_e1725);
                    let _e1735 = storage_0_13_[_e989];
                    let _e1736 = bitcast<f32>(_e1735);
                    let _e1740 = select(_e1736, 1f, (_e1736 < 1f));
                    let _e1742 = (_e1740 + -0.1f);
                    let _e1753 = storage_0_13_[_e1538];
                    let _e1759 = textureSampleCompareLevel(tex_0_6_, sampler_0_5_, vec3<f32>(_e1723, _e1724, _e1725), u32(bitcast<f32>(_e1753)), (((_e1740 / _e1742) - ((_e1740 * 0.1f) / (select(_e1729, _e1730, (_e1730 > _e1729)) * _e1742))) + -0.001f));
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = (_e1704 * _e1759);
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_ = (_e1706 * _e1759);
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_ = (_e1708 * _e1759);
                } else {
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = _e1704;
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_ = _e1706;
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_ = _e1708;
                }
            }
            let _e1770 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_;
            let _e1772 = radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_;
            let _e1774 = radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_;
            let _e1777 = storage_0_13_[_e955];
            let _e1781 = storage_0_13_[_e961];
            let _e1785 = storage_0_13_[_e967];
            let _e1789 = storage_0_13_[_e973];
            let _e1791 = (bitcast<f32>(_e1781) - in0_.x);
            let _e1792 = (bitcast<f32>(_e1785) - in0_.y);
            let _e1793 = (bitcast<f32>(_e1789) - in0_.z);
            let _e1795 = length(vec3<f32>(_e1791, _e1792, _e1793));
            let _e1801 = (1f / select(_e1795, 0.000001f, (_e1795 < 0.000001f)));
            let _e1809 = select(1f, 0f, (bitcast<f32>(_e1777) < 0.5f));
            let _e1812 = storage_0_13_[_e1048];
            let _e1813 = bitcast<f32>(_e1812);
            let _e1816 = storage_0_13_[_e1054];
            let _e1817 = bitcast<f32>(_e1816);
            let _e1820 = storage_0_13_[_e1060];
            let _e1821 = bitcast<f32>(_e1820);
            let _e1825 = (1f / length(vec3<f32>(_e1813, _e1817, _e1821)));
            let _e1829 = mix((_e1825 * _e1813), (_e1791 * _e1801), _e1809);
            let _e1830 = mix((_e1825 * _e1817), (_e1792 * _e1801), _e1809);
            let _e1831 = mix((_e1825 * _e1821), (_e1793 * _e1801), _e1809);
            let _e1835 = (1f / length(vec3<f32>(_e1829, _e1830, _e1831)));
            let _e1836 = (_e1829 * _e1835);
            let _e1837 = (_e1830 * _e1835);
            let _e1838 = (_e1831 * _e1835);
            let _e1843 = ((_e38 + _e1836) + 0f);
            let _e1845 = ((_e39 + _e1837) + 0f);
            let _e1847 = ((_e40 + _e1838) + 0.000001f);
            let _e1851 = (1f / length(vec3<f32>(_e1843, _e1845, _e1847)));
            let _e1852 = (_e1851 * _e1843);
            let _e1853 = (_e1851 * _e1845);
            let _e1854 = (_e1851 * _e1847);
            let _e1857 = dot(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e1836, _e1837, _e1838));
            let _e1861 = select(_e1857, 0.000001f, (_e1857 < 0.000001f));
            let _e1864 = dot(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e1852, _e1853, _e1854));
            let _e1867 = dot(vec3<f32>(_e38, _e39, _e40), vec3<f32>(_e1852, _e1853, _e1854));
            let _e1871 = select(_e1867, 0.000001f, (_e1867 < 0.000001f));
            if (_e1861 > 0.000001f) {
                let _e1877 = select(_e1864, 0.000001f, (_e1864 < 0.000001f));
                let _e1878 = mix(_e834, _e731, _e484);
                let _e1879 = mix(_e834, _e732, _e484);
                let _e1880 = mix(_e834, _e733, _e484);
                let _e1883 = pow(0.85714287f, 5f);
                let _e1885 = (1f - _e1878);
                let _e1887 = (_e1878 + (_e1885 * _e1883));
                let _e1889 = (1f - _e1879);
                let _e1891 = (_e1879 + (_e1889 * _e1883));
                let _e1893 = (1f - _e1880);
                let _e1895 = (_e1880 + (_e1893 * _e1883));
                let _e1900 = (1f - _e1871);
                let _e1909 = ((_e1871 * pow(_e1900, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e1911 = pow(_e1900, 5f);
                let _e1933 = mix(_e834, _e731, _e484);
                let _e1934 = mix(_e834, _e732, _e484);
                let _e1935 = mix(_e834, _e733, _e484);
                let _e1938 = pow(0.85714287f, 5f);
                let _e1940 = (1f - _e1933);
                let _e1942 = (_e1933 + (_e1940 * _e1938));
                let _e1944 = (1f - _e1934);
                let _e1946 = (_e1934 + (_e1944 * _e1938));
                let _e1948 = (1f - _e1935);
                let _e1950 = (_e1935 + (_e1948 * _e1938));
                let _e1962 = ((_e1871 * pow(_e1900, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e1964 = pow(_e1900, 5f);
                let _e1987 = (_e835 * sqrt(_e842));
                let _e1990 = (_e837 * (_e835 * sqrt(_e842)));
                let _e1993 = dot(vec3<f32>(_e1852, _e1853, _e1854), vec3<f32>(_e55, _e56, _e57));
                let _e1996 = dot(vec3<f32>(_e1852, _e1853, _e1854), vec3<f32>(cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).x, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).y, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).z));
                let _e1998 = (_e1987 * _e1987);
                let _e2001 = (_e1990 * _e1990);
                let _e2004 = (_e1877 * _e1877);
                let _e2005 = (_e2004 + (((_e1993 * _e1993) / _e1998) + ((_e1996 * _e1996) / _e2001)));
                let _e2012 = (1f / (_e2005 * (((_e1987 * 3.1415927f) * _e1990) * _e2005)));
                let _e2015 = dot(vec3<f32>(_e38, _e39, _e40), vec3<f32>(_e55, _e56, _e57));
                let _e2018 = dot(vec3<f32>(_e38, _e39, _e40), vec3<f32>(cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).x, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).y, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).z));
                let _e2021 = dot(vec3<f32>(_e1836, _e1837, _e1838), vec3<f32>(_e55, _e56, _e57));
                let _e2024 = dot(vec3<f32>(_e1836, _e1837, _e1838), vec3<f32>(cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).x, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).y, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).z));
                let _e2030 = (_e1861 * _e1861);
                let _e2050 = (0.5f / max(((_e47 * sqrt(max((_e2030 + ((_e1998 * (_e2021 * _e2021)) + (_e2001 * (_e2024 * _e2024)))), 0.000001f))) + (_e1861 * sqrt(max((_e656 + ((_e1998 * (_e2015 * _e2015)) + (_e2001 * (_e2018 * _e2018)))), 0.000001f)))), 0.000001f));
                let _e2069 = (1f / length(vec3<f32>(_e848, _e849, _e850)));
                let _e2076 = (_e1836 - (_e16 * _e1861));
                let _e2077 = (_e1837 - (_e17 * _e1861));
                let _e2078 = (_e1838 - (_e18 * _e1861));
                let _e2082 = (1f / length(vec3<f32>(_e2076, _e2077, _e2078)));
                let _e2088 = dot(vec3<f32>((_e848 * _e2069), (_e849 * _e2069), (_e850 * _e2069)), vec3<f32>((_e2076 * _e2082), (_e2077 * _e2082), (_e2078 * _e2082)));
                let _e2093 = acos(max(_e47, 0f));
                let _e2096 = acos(max(_e1861, 0f));
                let _e2120 = (max((1f - dot(vec3<f32>((max(((_e1878 + (_e1885 * _e1911)) - ((_e1887 - (_e1887 * _e177)) * _e1909)), 0f) * _e111), (max(((_e1879 + (_e1889 * _e1911)) - ((_e1891 - (_e1891 * _e183)) * _e1909)), 0f) * _e111), (max(((_e1880 + (_e1893 * _e1911)) - ((_e1895 - (_e1895 * _e189)) * _e1909)), 0f) * _e111)), vec3<f32>(0.2126f, 0.7152f, 0.0722f))), 0f) * (((1f - ((_e859 * 0.5f) / (_e859 + 0.57f))) + (tan(min(_e2093, _e2096)) * ((((_e859 * 0.45f) / (_e859 + 0.09f)) * select(0f, _e2088, (_e2088 > 0f))) * sin(max(_e2093, _e2096))))) * 0.31830987f));
                if _e736 {
                    let _e2141 = (_e877 * sqrt(_e884));
                    let _e2144 = (_e879 * (_e877 * sqrt(_e884)));
                    let _e2147 = dot(vec3<f32>(_e1852, _e1853, _e1854), vec3<f32>(_e55, _e56, _e57));
                    let _e2150 = dot(vec3<f32>(_e1852, _e1853, _e1854), vec3<f32>(cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).x, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).y, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).z));
                    let _e2152 = (_e2141 * _e2141);
                    let _e2155 = (_e2144 * _e2144);
                    let _e2158 = (_e2004 + (((_e2147 * _e2147) / _e2152) + ((_e2150 * _e2150) / _e2155)));
                    let _e2168 = dot(vec3<f32>(_e38, _e39, _e40), vec3<f32>(_e55, _e56, _e57));
                    let _e2171 = dot(vec3<f32>(_e38, _e39, _e40), vec3<f32>(cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).x, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).y, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).z));
                    let _e2174 = dot(vec3<f32>(_e1836, _e1837, _e1838), vec3<f32>(_e55, _e56, _e57));
                    let _e2177 = dot(vec3<f32>(_e1836, _e1837, _e1838), vec3<f32>(cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).x, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).y, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).z));
                    let _e2204 = ((_e874 + ((1f - _e874) * pow(_e1900, 5f))) * ((1f / (_e2158 * (((_e2141 * 3.1415927f) * _e2144) * _e2158))) * (0.5f / max(((_e47 * sqrt(max((_e2030 + ((_e2152 * (_e2174 * _e2174)) + (_e2155 * (_e2177 * _e2177)))), 0.000001f))) + (_e1861 * sqrt(max((_e656 + ((_e2152 * (_e2168 * _e2168)) + (_e2155 * (_e2171 * _e2171)))), 0.000001f)))), 0.000001f))));
                    coat_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e2204 * _e362) * _e339);
                    coat_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e2204 * _e368) * _e339);
                    coat_u002e_i_u002e_sroa_u002e_8_u002e_0_ = ((_e2204 * _e374) * _e339);
                } else {
                    coat_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    coat_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    coat_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2221 = coat_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2223 = coat_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2225 = coat_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e309 <= 0f)) {
                    let _e2249 = ((_e1871 * ((1f / ((_e888 * (((1f / _e47) + (1f / _e1861)) + -2f)) + 1f)) * (_e888 / (pow((((_e888 + -1f) * _e2004) + 1f), 2f) * 3.1415927f)))) / max(((_e47 * 4f) * _e1861), 0.000001f));
                    fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e2249 * bitcast<f32>(_e320)) * _e309);
                    fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e2249 * bitcast<f32>(_e326)) * _e309);
                    fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_ = ((_e2249 * bitcast<f32>(_e332)) * _e309);
                } else {
                    fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2266 = fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2268 = fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2270 = fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e195 <= 0f)) {
                    let _e2272 = (_e835 * sqrt(_e842));
                    if !((_e201 <= 0f)) {
                        extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (-(log(max(_e217, 0.000001f))) / _e201);
                        extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (-(log(max(_e223, 0.000001f))) / _e201);
                        extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (-(log(max(_e229, 0.000001f))) / _e201);
                    } else {
                        extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e2292 = extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                    let _e2294 = extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                    let _e2296 = extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                    let _e2303 = (_e2272 * _e2272);
                    let _e2308 = ((_e2004 * (_e2303 + -1f)) + 1f);
                    let _e2314 = (1f - _e2303);
                    let _e2333 = ((1f - (_e834 + ((1f - _e834) * pow((1f - _e47), 5f)))) * ((_e2303 / (_e2308 * (_e2308 * 3.1415927f))) * (0.5f / max(((_e47 * sqrt(max((_e2303 + (_e2030 * _e2314)), 0.000001f))) + (_e1861 * sqrt(max((_e2303 + (_e656 * _e2314)), 0.000001f)))), 0.000001f))));
                    transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (((exp((_e201 * -(_e2292))) * _e2333) * _e217) * _e195);
                    transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (((exp((_e201 * -(_e2294))) * _e2333) * _e223) * _e195);
                    transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (((exp((_e201 * -(_e2296))) * _e2333) * _e229) * _e195);
                } else {
                    transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2362 = transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2364 = transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2366 = transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e255 <= 0f)) {
                    let _e2371 = length(vec3<f32>((_e848 - _e2076), (_e849 - _e2077), (_e850 - _e2078)));
                    let _e2373 = sqrt(3f);
                    let _e2396 = -(_e2371);
                    let _e2404 = (max(((clamp(((_e47 * _e1861) + (((_e660 + max((1f - _e2030), 0f)) - (_e2371 * _e2371)) * 0.5f)), -1f, 1f) * bitcast<f32>(_e272)) + 1f), 0f) * 0.31830987f);
                    subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (((exp((((1f / max((_e261 * bitcast<f32>(_e266)), 0.000001f)) * _e2373) * _e2396)) * _e2404) * _e279) * _e255);
                    subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (((exp((((1f / max((_e261 * bitcast<f32>(_e296)), 0.000001f)) * _e2373) * _e2396)) * _e2404) * _e285) * _e255);
                    subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (((exp((((1f / max((_e261 * bitcast<f32>(_e302)), 0.000001f)) * _e2373) * _e2396)) * _e2404) * _e291) * _e255);
                } else {
                    subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2424 = subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2426 = subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2428 = subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                lo_u002e_i_u002e_sroa_u002e_0_u002e_1_ = (_e944 + (_e1770 * (_e1861 * ((((_e811 * ((_e713 * ((((max(((_e1933 + (_e1940 * _e1964)) - ((_e1942 - (_e1942 * _e177)) * _e1962)), 0f) * _e111) * _e2012) * _e2050) * _e75)) + (_e713 * (((_e481 * _e853) * _e2120) * _e75)))) + _e2221) + (_e2266 + _e2362)) + _e2424))));
                lo_u002e_i_u002e_sroa_u002e_6_u002e_1_ = (_e942 + (_e1772 * (_e1861 * ((((_e813 * ((_e718 * ((((max(((_e1934 + (_e1944 * _e1964)) - ((_e1946 - (_e1946 * _e183)) * _e1962)), 0f) * _e111) * _e2012) * _e2050) * _e75)) + (_e718 * (((_e482 * _e853) * _e2120) * _e75)))) + _e2223) + (_e2268 + _e2364)) + _e2426))));
                lo_u002e_i_u002e_sroa_u002e_9_u002e_1_ = (_e940 + (_e1774 * (_e1861 * ((((_e815 * ((_e723 * ((((max(((_e1935 + (_e1948 * _e1964)) - ((_e1950 - (_e1950 * _e189)) * _e1962)), 0f) * _e111) * _e2012) * _e2050) * _e75)) + (_e723 * (((_e483 * _e853) * _e2120) * _e75)))) + _e2225) + (_e2270 + _e2366)) + _e2428))));
            } else {
                lo_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e944;
                lo_u002e_i_u002e_sroa_u002e_6_u002e_1_ = _e942;
                lo_u002e_i_u002e_sroa_u002e_9_u002e_1_ = _e940;
            }
            let _e2457 = lo_u002e_i_u002e_sroa_u002e_0_u002e_1_;
            out_u002e_lo_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e2457;
            let _e2460 = lo_u002e_i_u002e_sroa_u002e_6_u002e_1_;
            out_u002e_lo_u002e_i_u002e_sroa_u002e_6_u002e_1_ = _e2460;
            let _e2463 = lo_u002e_i_u002e_sroa_u002e_9_u002e_1_;
            out_u002e_lo_u002e_i_u002e_sroa_u002e_9_u002e_1_ = _e2463;
            let _e2466 = (_e938 + 1u);
            if (_e2466 == _e816) {
                lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e2457;
                lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e2460;
                lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e2463;
                break;
            } else {
                k_u002e_sroa_u002e_0_u002e_0_ = _e2466;
                lo_u002e_i_u002e_sroa_u002e_9_ = _e2463;
                lo_u002e_i_u002e_sroa_u002e_6_ = _e2460;
                lo_u002e_i_u002e_sroa_u002e_0_ = _e2457;
                continue;
            }
        }
    }
    let _e2482 = lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
    let _e2484 = lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
    let _e2486 = lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
    let _e2496 = uniform_struct_0_3_.f0_.x;
    let _e2500 = uniform_struct_0_3_.f0_.y;
    let _e2504 = uniform_struct_0_3_.f0_.z;
    if !((_e404 <= 0f)) {
        let _e2511 = ((_e469.x * bitcast<f32>(_e409)) * _e404);
        let _e2512 = ((_e474.y * bitcast<f32>(_e415)) * _e404);
        let _e2513 = ((_e479.z * bitcast<f32>(_e421)) * _e404);
        let _e2523 = ((1f - _e339) + ((1f - pow((1f - _e47), 5f)) * _e339));
        emission_u002e_i_u002e_sroa_u002e_0_u002e_1_ = (_e2511 + ((((_e2511 * _e362) * _e2523) - _e2511) * _e339));
        emission_u002e_i_u002e_sroa_u002e_7_u002e_1_ = (_e2512 + ((((_e2512 * _e368) * _e2523) - _e2512) * _e339));
        emission_u002e_i_u002e_sroa_u002e_10_u002e_1_ = (_e2513 + ((((_e2513 * _e374) * _e2523) - _e2513) * _e339));
    } else {
        emission_u002e_i_u002e_sroa_u002e_0_u002e_1_ = 0f;
        emission_u002e_i_u002e_sroa_u002e_7_u002e_1_ = 0f;
        emission_u002e_i_u002e_sroa_u002e_10_u002e_1_ = 0f;
    }
    let _e2549 = emission_u002e_i_u002e_sroa_u002e_0_u002e_1_;
    let _e2551 = emission_u002e_i_u002e_sroa_u002e_7_u002e_1_;
    let _e2553 = emission_u002e_i_u002e_sroa_u002e_10_u002e_1_;
    let _e2558 = ((_e165 + -1f) / (_e165 + 1f));
    let _e2559 = (_e2558 * _e2558);
    let _e2567 = (dot(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e38, _e39, _e40)) * 2f);
    let _e2571 = ((_e16 * _e2567) - _e38);
    let _e2572 = ((_e17 * _e2567) - _e39);
    let _e2573 = ((_e18 * _e2567) - _e40);
    let _e2576 = uniform_struct_0_3_.f4_;
    let _e2577 = (_e523 * _e2576);
    let _e2581 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2571, _e2572, _e2573), _e2577);
    let _e2586 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2571, _e2572, _e2573), _e2577);
    let _e2591 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2571, _e2572, _e2573), _e2577);
    let _e2596 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e16, _e17, _e18));
    let _e2601 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e16, _e17, _e18));
    let _e2606 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e16, _e17, _e18));
    let _e2611 = textureSample(tex_0_9_, sampler_0_10_, vec2<f32>(_e47, _e523));
    let _e2616 = textureSample(tex_0_9_, sampler_0_10_, vec2<f32>(_e47, _e523));
    let _e2619 = (1f - _e484);
    let _e2625 = uniform_struct_0_3_.f3_;
    let _e2638 = max((1f - _e2616.y), 0f);
    let _e2662 = !((bitcast<f32>(_e437) >= 0.5f));
    let _e2663 = select(_e2549, (((_e2482 + (mix(_e481, (_e481 * _e111), _e484) * _e2496)) + _e2549) + (_e2625 * ((_e2581.x * ((mix(_e2559, _e731, _e484) * _e2611.x) + _e2616.y)) + (((_e481 * _e2619) * _e2596.x) * _e2638)))), _e2662);
    let _e2664 = select(_e2551, (((_e2484 + (mix(_e482, (_e482 * _e111), _e484) * _e2500)) + _e2551) + (_e2625 * ((_e2586.y * ((mix(_e2559, _e732, _e484) * _e2611.x) + _e2616.y)) + (((_e482 * _e2619) * _e2601.y) * _e2638)))), _e2662);
    let _e2665 = select(_e2553, (((_e2486 + (mix(_e483, (_e483 * _e111), _e484) * _e2504)) + _e2553) + (_e2625 * ((_e2591.z * ((mix(_e2559, _e733, _e484) * _e2611.x) + _e2616.y)) + (((_e483 * _e2619) * _e2606.z) * _e2638)))), _e2662);
    if ((_e2663 <= _e2663) && (0f <= 0f)) {
        if (abs(_e2663) < 100000000000000000000f) {
            _2_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2663;
        } else {
            _2_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        _2_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2679 = _2_u002e_sroa_u002e_0_u002e_0_u002e_i;
    if ((_e2664 <= _e2664) && (0f <= 0f)) {
        if (abs(_e2664) < 100000000000000000000f) {
            _3_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2664;
        } else {
            _3_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        _3_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2693 = _3_u002e_sroa_u002e_0_u002e_0_u002e_i;
    if ((_e2665 <= _e2665) && (0f <= 0f)) {
        if (abs(_e2665) < 100000000000000000000f) {
            _4_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2665;
            let _e2703 = _4_u002e_sroa_u002e_0_u002e_0_u002e_i;
            fragment_out = vec4<f32>(_e2679, _e2693, _e2703, bitcast<f32>(_e427));
            let _e2708 = fragment_out;
            return _e2708;
        }
        _4_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    } else {
        _4_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2714 = _4_u002e_sroa_u002e_0_u002e_0_u002e_i;
    fragment_out = vec4<f32>(_e2679, _e2693, _e2714, bitcast<f32>(_e427));
    let _e2719 = fragment_out;
    return _e2719;
}
