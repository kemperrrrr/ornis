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

struct UniformStruct5_ {
    f0_: u32,
    f1_: u32,
    f2_: u32,
    f3_: u32,
}

@group(0) @binding(0) 
var<uniform> uniform_struct_0_0_: UniformStruct0_;
@group(0) @binding(2) 
var<storage> storage_0_2_: array<u32>;
@group(0) @binding(3) 
var<uniform> uniform_struct_0_3_: UniformStruct4_;
@group(0) @binding(12) 
var<uniform> uniform_struct_0_12_: UniformStruct5_;
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
fn fs_main(@builtin(position) position: vec4<f32>, @location(0) in0_: vec3<f32>, @location(1) in1_: vec3<f32>, @location(3) in3_: vec3<f32>, @location(4) @interpolate(flat) in4_: u32) -> @location(0) vec4<f32> {
    var fragment_out: vec4<f32>;
    var lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa: f32;
    var lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa: f32;
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i: u32;
    var radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_: f32;
    var vis_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_: f32;
    var extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_: f32;
    var emission_u002e_sroa_u002e_0_u002e_1_: f32;
    var emission_u002e_sroa_u002e_5_u002e_1_: f32;
    var emission_u002e_sroa_u002e_8_u002e_1_: f32;
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
    var coat_darken_u002e_sroa_u002e_0_u002e_0_: f32;
    var coat_darken_u002e_sroa_u002e_5_u002e_0_: f32;
    var coat_darken_u002e_sroa_u002e_7_u002e_0_: f32;
    var k_u002e_sroa_u002e_0_u002e_0_: u32;
    var lo_u002e_sroa_u002e_9_: f32;
    var lo_u002e_sroa_u002e_6_: f32;
    var lo_u002e_sroa_u002e_0_: f32;
    var _2_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var lo_u002e_sroa_u002e_0_u002e_1_: f32;
    var lo_u002e_sroa_u002e_6_u002e_1_: f32;
    var lo_u002e_sroa_u002e_9_u002e_1_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_10_u002e_0_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_15_u002e_0_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_10_u002e_1_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_15_u002e_1_: f32;
    var _3_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var out_u002e_lo_u002e_sroa_u002e_0_u002e_1_: f32;
    var out_u002e_lo_u002e_sroa_u002e_6_u002e_1_: f32;
    var out_u002e_lo_u002e_sroa_u002e_9_u002e_1_: f32;

    let _e14 = (1f / length(vec3<f32>(in1_.x, in1_.y, in1_.z)));
    let _e15 = (in1_.x * _e14);
    let _e16 = (in1_.y * _e14);
    let _e17 = (in1_.z * _e14);
    let _e21 = uniform_struct_0_0_.f8_.x;
    let _e25 = uniform_struct_0_0_.f8_.y;
    let _e29 = uniform_struct_0_0_.f8_.z;
    let _e30 = (_e21 - in0_.x);
    let _e31 = (_e25 - in0_.y);
    let _e32 = (_e29 - in0_.z);
    let _e36 = (1f / length(vec3<f32>(_e30, _e31, _e32)));
    let _e37 = (_e30 * _e36);
    let _e38 = (_e31 * _e36);
    let _e39 = (_e32 * _e36);
    let _e42 = dot(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e37, _e38, _e39));
    let _e46 = select(_e42, 0.000001f, (_e42 < 0.000001f));
    let _e53 = (1f / length(vec3<f32>(in3_.x, in3_.y, in3_.z)));
    let _e54 = (in3_.x * _e53);
    let _e55 = (in3_.y * _e53);
    let _e56 = (in3_.z * _e53);
    let _e70 = (in4_ * 80u);
    let _e73 = storage_0_2_[_e70];
    let _e78 = storage_0_2_[(_e70 | 1u)];
    let _e83 = storage_0_2_[(_e70 | 2u)];
    let _e88 = storage_0_2_[(_e70 | 4u)];
    let _e93 = storage_0_2_[(_e70 | 5u)];
    let _e98 = storage_0_2_[(_e70 | 6u)];
    let _e103 = storage_0_2_[(_e70 | 8u)];
    let _e108 = storage_0_2_[(_e70 | 9u)];
    let _e109 = bitcast<f32>(_e108);
    let _e110 = dpdx(_e15);
    let _e111 = dpdx(_e16);
    let _e112 = dpdx(_e17);
    let _e113 = dpdy(_e15);
    let _e114 = dpdy(_e16);
    let _e115 = dpdy(_e17);
    let _e126 = (((dot(vec3<f32>(_e110, _e111, _e112), vec3<f32>(_e110, _e111, _e112)) + dot(vec3<f32>(_e113, _e114, _e115), vec3<f32>(_e113, _e114, _e115))) * 0.15f) * 2f);
    let _e131 = (_e109 * _e109);
    let _e133 = ((_e131 * _e131) + select(0.2f, _e126, (_e126 < 0.2f)));
    let _e137 = select(_e133, 0f, (_e133 < 0f));
    let _e141 = select(_e137, 1f, (_e137 > 1f));
    let _e147 = sqrt(sqrt(select(0f, _e141, (_e141 > 0f))));
    let _e152 = storage_0_2_[(_e70 | 10u)];
    let _e157 = storage_0_2_[(_e70 | 11u)];
    let _e162 = storage_0_2_[(_e70 | 12u)];
    let _e167 = storage_0_2_[(_e70 | 13u)];
    let _e172 = storage_0_2_[(_e70 | 14u)];
    let _e177 = storage_0_2_[(_e70 + 16u)];
    let _e182 = storage_0_2_[(_e70 + 17u)];
    let _e197 = storage_0_2_[(_e70 + 20u)];
    let _e202 = storage_0_2_[(_e70 + 21u)];
    let _e207 = storage_0_2_[(_e70 + 22u)];
    let _e232 = storage_0_2_[(_e70 + 28u)];
    let _e237 = storage_0_2_[(_e70 + 29u)];
    let _e242 = storage_0_2_[(_e70 + 30u)];
    let _e247 = storage_0_2_[(_e70 + 31u)];
    let _e252 = storage_0_2_[(_e70 + 32u)];
    let _e257 = storage_0_2_[(_e70 + 33u)];
    let _e262 = storage_0_2_[(_e70 + 34u)];
    let _e267 = storage_0_2_[(_e70 + 36u)];
    let _e272 = storage_0_2_[(_e70 + 37u)];
    let _e277 = storage_0_2_[(_e70 + 40u)];
    let _e282 = storage_0_2_[(_e70 + 41u)];
    let _e287 = storage_0_2_[(_e70 + 44u)];
    let _e292 = storage_0_2_[(_e70 + 45u)];
    let _e297 = storage_0_2_[(_e70 + 46u)];
    let _e302 = storage_0_2_[(_e70 + 48u)];
    let _e307 = storage_0_2_[(_e70 + 49u)];
    let _e312 = storage_0_2_[(_e70 + 50u)];
    let _e317 = storage_0_2_[(_e70 + 51u)];
    let _e322 = storage_0_2_[(_e70 + 52u)];
    let _e327 = storage_0_2_[(_e70 + 53u)];
    let _e332 = storage_0_2_[(_e70 + 54u)];
    let _e337 = storage_0_2_[(_e70 + 56u)];
    let _e342 = storage_0_2_[(_e70 + 60u)];
    let _e347 = storage_0_2_[(_e70 + 61u)];
    let _e352 = storage_0_2_[(_e70 + 62u)];
    let _e357 = storage_0_2_[(_e70 + 64u)];
    let _e362 = storage_0_2_[(_e70 + 68u)];
    let _e367 = storage_0_2_[(_e70 + 69u)];
    let _e372 = storage_0_2_[(_e70 + 70u)];
    let _e377 = storage_0_2_[(_e70 + 72u)];
    let _e387 = storage_0_2_[(_e70 + 74u)];
    let _e389 = bitcast<f32>(_e352);
    let _e393 = (_e46 * _e46);
    let _e397 = max((1f - _e393), 0f);
    let _e399 = (sqrt(_e397) / _e389);
    let _e412 = pow(((_e389 + -1f) / (_e389 + 1f)), 2f);
    let _e416 = (((bitcast<f32>(_e347) * 1000f) * (_e389 * 12.566371f)) * sqrt(max((1f - (_e399 * _e399)), 0f)));
    let _e427 = (_e412 * 2f);
    let _e433 = (1f - (_e412 * _e412));
    let _e445 = clamp(bitcast<f32>(_e342), 0f, 1f);
    let _e450 = ((_e445 * ((((_e427 * cos((_e416 / 650f))) / _e433) + 1f) + -1f)) + 1f);
    let _e455 = ((_e445 * ((((_e427 * cos((_e416 / 550f))) / _e433) + 1f) + -1f)) + 1f);
    let _e460 = ((_e445 * ((((_e427 * cos((_e416 / 450f))) / _e433) + 1f) + -1f)) + 1f);
    let _e461 = bitcast<f32>(_e302);
    let _e462 = bitcast<f32>(_e337);
    let _e463 = bitcast<f32>(_e83);
    let _e464 = bitcast<f32>(_e73);
    let _e465 = bitcast<f32>(_e103);
    let _e466 = bitcast<f32>(_e232);
    let _e471 = ((_e462 + -1f) / (_e462 + 1f));
    let _e474 = bitcast<f32>(_e88);
    let _e475 = bitcast<f32>(_e93);
    let _e476 = bitcast<f32>(_e98);
    let _e477 = (_e464 * _e474);
    let _e478 = (_e464 * _e475);
    let _e479 = (_e464 * _e476);
    let _e480 = bitcast<f32>(_e252);
    let _e481 = bitcast<f32>(_e257);
    let _e482 = bitcast<f32>(_e262);
    let _e485 = !((_e461 <= 0f));
    if _e485 {
        let _e491 = (1f - ((1f - pow(_e471, 2f)) / (_e462 * _e462)));
        let _e493 = (1f - _e491);
        let _e494 = (_e479 * _e465);
        let _e507 = (_e478 * _e465);
        let _e520 = (_e477 * _e465);
        let _e534 = (_e461 * bitcast<f32>(_e317));
        coat_darken_u002e_sroa_u002e_0_u002e_0_ = ((_e534 * ((_e493 / max((1f - ((_e520 + (((((_e474 - _e480) * _e466) + _e480) - _e520) * _e463)) * _e491)), 0.000001f)) + -1f)) + 1f);
        coat_darken_u002e_sroa_u002e_5_u002e_0_ = ((_e534 * ((_e493 / max((1f - ((_e507 + (((((_e475 - _e481) * _e466) + _e481) - _e507) * _e463)) * _e491)), 0.000001f)) + -1f)) + 1f);
        coat_darken_u002e_sroa_u002e_7_u002e_0_ = ((_e534 * ((_e493 / max((1f - ((_e494 + (((((_e476 - _e482) * _e466) + _e482) - _e494) * _e463)) * _e491)), 0.000001f)) + -1f)) + 1f);
    } else {
        coat_darken_u002e_sroa_u002e_0_u002e_0_ = 1f;
        coat_darken_u002e_sroa_u002e_5_u002e_0_ = 1f;
        coat_darken_u002e_sroa_u002e_7_u002e_0_ = 1f;
    }
    let _e560 = coat_darken_u002e_sroa_u002e_0_u002e_0_;
    let _e562 = coat_darken_u002e_sroa_u002e_5_u002e_0_;
    let _e564 = coat_darken_u002e_sroa_u002e_7_u002e_0_;
    let _e567 = uniform_struct_0_3_.f2_;
    let _e571 = uniform_struct_0_0_.f0_.x;
    let _e575 = uniform_struct_0_0_.f1_.x;
    let _e579 = uniform_struct_0_0_.f2_.x;
    let _e583 = uniform_struct_0_0_.f3_.x;
    let _e593 = uniform_struct_0_0_.f0_.y;
    let _e597 = uniform_struct_0_0_.f1_.y;
    let _e601 = uniform_struct_0_0_.f2_.y;
    let _e605 = uniform_struct_0_0_.f3_.y;
    let _e615 = uniform_struct_0_0_.f0_.w;
    let _e619 = uniform_struct_0_0_.f1_.w;
    let _e623 = uniform_struct_0_0_.f2_.w;
    let _e627 = uniform_struct_0_0_.f3_.w;
    let _e633 = (_e627 + (((in0_.x * _e615) + (in0_.y * _e619)) + (in0_.z * _e623)));
    let _e637 = select(0.5f, _e633, (_e633 > 0.5f));
    let _e678 = ((((min(u32(floor((((((_e605 + (((in0_.x * _e593) + (in0_.y * _e597)) + (in0_.z * _e601))) / _e637) * 0.5f) + 0.5f) * 9f))), 8u) << 4u) | min(u32(floor((((((_e583 + (((in0_.x * _e571) + (in0_.y * _e575)) + (in0_.z * _e579))) / _e637) * 0.5f) + 0.5f) * 16f))), 15u)) + (min(u32(ceil(((log((_e637 * 2f)) * 3.4407384f) + -0.0001f))), 23u) * 144u)) << 1u);
    let _e681 = uniform_struct_0_12_.f2_;
    let _e684 = storage_0_15_[_e678];
    let _e689 = storage_0_15_[(_e678 | 1u)];
    let _e691 = ((_e689 * _e681) + _e567);
    if (_e691 == 0u) {
        lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
        lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = 0f;
        lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = 0f;
    } else {
        let _e705 = bitcast<f32>(_e78);
        let _e706 = bitcast<f32>(_e152);
        let _e712 = ((_e706 + -1f) / (_e706 + 1f));
        let _e713 = (_e712 * _e712);
        let _e714 = bitcast<f32>(_e162);
        let _e715 = bitcast<f32>(_e167);
        let _e716 = bitcast<f32>(_e172);
        let _e717 = (_e147 * _e147);
        let _e719 = (1f - bitcast<f32>(_e157));
        let _e724 = (2f / ((_e719 * _e719) + 1f));
        let _e726 = select(_e147, _e705, (_e147 < _e705));
        let _e730 = (_e37 - (_e15 * _e46));
        let _e731 = (_e38 - (_e16 * _e46));
        let _e732 = (_e39 - (_e17 * _e46));
        let _e735 = (1f - _e463);
        let _e740 = max((_e726 * _e726), 0.000001f);
        let _e741 = (_e740 * _e740);
        let _e757 = bitcast<f32>(_e307);
        let _e758 = (_e471 * _e471);
        let _e761 = (_e757 * _e757);
        let _e763 = (1f - bitcast<f32>(_e312));
        let _e768 = (2f / ((_e763 * _e763) + 1f));
        let _e772 = bitcast<f32>(_e277);
        let _e776 = bitcast<f32>(_e282);
        let _e777 = (_e776 * _e776);
        let _e787 = bitcast<f32>(_e177);
        let _e788 = bitcast<f32>(_e182);
        let _e818 = bitcast<f32>(_e237);
        k_u002e_sroa_u002e_0_u002e_0_ = 0u;
        lo_u002e_sroa_u002e_9_ = 0f;
        lo_u002e_sroa_u002e_6_ = 0f;
        lo_u002e_sroa_u002e_0_ = 0f;
        loop {
            let _e843 = k_u002e_sroa_u002e_0_u002e_0_;
            let _e845 = lo_u002e_sroa_u002e_9_;
            let _e847 = lo_u002e_sroa_u002e_6_;
            let _e849 = lo_u002e_sroa_u002e_0_;
            if (_e843 < _e567) {
                _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i = _e843;
            } else {
                let _e855 = storage_0_14_[(((_e684 * _e681) - _e567) + _e843)];
                _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i = _e855;
            }
            let _e858 = _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i;
            let _e860 = (_e858 * 36u);
            let _e863 = storage_0_13_[_e860];
            let _e864 = bitcast<f32>(_e863);
            let _e866 = (_e860 + 8u);
            let _e869 = storage_0_13_[_e866];
            let _e872 = (_e860 + 9u);
            let _e875 = storage_0_13_[_e872];
            let _e878 = (_e860 + 10u);
            let _e881 = storage_0_13_[_e878];
            let _e887 = length(vec3<f32>((bitcast<f32>(_e869) - in0_.x), (bitcast<f32>(_e875) - in0_.y), (bitcast<f32>(_e881) - in0_.z)));
            let _e892 = select(1f, 0f, (_e864 < 0.5f));
            let _e894 = (_e860 + 16u);
            let _e897 = storage_0_13_[_e894];
            let _e898 = bitcast<f32>(_e897);
            let _e910 = storage_0_13_[(_e860 + 12u)];
            let _e916 = storage_0_13_[(_e860 + 13u)];
            let _e922 = storage_0_13_[(_e860 + 14u)];
            let _e928 = storage_0_13_[(_e860 + 15u)];
            let _e929 = bitcast<f32>(_e928);
            let _e932 = storage_0_13_[_e866];
            let _e936 = storage_0_13_[_e872];
            let _e940 = storage_0_13_[_e878];
            let _e942 = (in0_.x - bitcast<f32>(_e932));
            let _e943 = (in0_.y - bitcast<f32>(_e936));
            let _e944 = (in0_.z - bitcast<f32>(_e940));
            let _e948 = (1f / length(vec3<f32>(_e942, _e943, _e944)));
            let _e953 = (_e860 + 4u);
            let _e956 = storage_0_13_[_e953];
            let _e957 = bitcast<f32>(_e956);
            let _e959 = (_e860 + 5u);
            let _e962 = storage_0_13_[_e959];
            let _e963 = bitcast<f32>(_e962);
            let _e965 = (_e860 + 6u);
            let _e968 = storage_0_13_[_e965];
            let _e969 = bitcast<f32>(_e968);
            let _e973 = (1f / length(vec3<f32>(_e957, _e963, _e969)));
            let _e984 = storage_0_13_[(_e860 + 18u)];
            let _e990 = storage_0_13_[(_e860 + 17u)];
            let _e997 = select(1f, 0f, (_e864 < 1.5f));
            let _e999 = mix(1f, smoothstep(bitcast<f32>(_e984), bitcast<f32>(_e990), dot(vec3<f32>((_e973 * _e957), (_e973 * _e963), (_e973 * _e969)), vec3<f32>((_e942 * _e948), (_e943 * _e948), (_e944 * _e948)))), _e997);
            let _e1000 = (_e887 * _e887);
            let _e1012 = (mix(1f, (1f - smoothstep((_e898 * 0.99f), _e898, _e887)), _e892) * mix(1f, (1f / select(_e1000, 0.000001f, (_e1000 < 0.000001f))), _e892));
            let _e1016 = (_e999 * ((bitcast<f32>(_e910) * _e929) * _e1012));
            let _e1017 = (_e999 * ((bitcast<f32>(_e916) * _e929) * _e1012));
            let _e1018 = (_e999 * ((bitcast<f32>(_e922) * _e929) * _e1012));
            let _e1021 = (_e892 * (1f - _e997));
            if (((1f - _e892) * select(1f, 0f, (_e898 < 0.5f))) > 0.5f) {
                let _e1036 = uniform_struct_0_3_.f6_.f1_.x;
                let _e1041 = uniform_struct_0_3_.f6_.f1_.y;
                let _e1046 = uniform_struct_0_3_.f6_.f1_.z;
                let _e1049 = dot(vec3<f32>((in0_.x - _e21), (in0_.y - _e25), (in0_.z - _e29)), vec3<f32>(_e1036, _e1041, _e1046));
                let _e1054 = uniform_struct_0_3_.f6_.f2_.x;
                let _e1059 = uniform_struct_0_3_.f6_.f2_.y;
                let _e1064 = uniform_struct_0_3_.f6_.f2_.z;
                let _e1069 = uniform_struct_0_3_.f6_.f2_.w;
                let _e1073 = uniform_struct_0_3_.f6_.f3_;
                let _e1077 = select(1f, 0f, (_e1049 < _e1059));
                let _e1081 = select(1f, 0f, (_e1049 < _e1064));
                let _e1085 = select(1f, 0f, (_e1049 < _e1069));
                let _e1088 = u32(((_e1077 + _e1081) + _e1085));
                let _e1091 = mix(mix(mix(_e1059, _e1064, _e1077), _e1069, _e1081), _e1073, _e1085);
                let _e1101 = uniform_struct_0_3_.f6_.f0_[_e1088].f0_.x;
                let _e1108 = uniform_struct_0_3_.f6_.f0_[_e1088].f1_.x;
                let _e1115 = uniform_struct_0_3_.f6_.f0_[_e1088].f2_.x;
                let _e1122 = uniform_struct_0_3_.f6_.f0_[_e1088].f3_.x;
                let _e1135 = uniform_struct_0_3_.f6_.f0_[_e1088].f0_.y;
                let _e1142 = uniform_struct_0_3_.f6_.f0_[_e1088].f1_.y;
                let _e1149 = uniform_struct_0_3_.f6_.f0_[_e1088].f2_.y;
                let _e1156 = uniform_struct_0_3_.f6_.f0_[_e1088].f3_.y;
                let _e1169 = uniform_struct_0_3_.f6_.f0_[_e1088].f0_.z;
                let _e1176 = uniform_struct_0_3_.f6_.f0_[_e1088].f1_.z;
                let _e1183 = uniform_struct_0_3_.f6_.f0_[_e1088].f2_.z;
                let _e1190 = uniform_struct_0_3_.f6_.f0_[_e1088].f3_.z;
                let _e1203 = uniform_struct_0_3_.f6_.f0_[_e1088].f0_.w;
                let _e1210 = uniform_struct_0_3_.f6_.f0_[_e1088].f1_.w;
                let _e1217 = uniform_struct_0_3_.f6_.f0_[_e1088].f2_.w;
                let _e1224 = uniform_struct_0_3_.f6_.f0_[_e1088].f3_.w;
                let _e1230 = (_e1224 + (((in0_.x * _e1203) + (in0_.y * _e1210)) + (in0_.z * _e1217)));
                let _e1247 = textureSampleCompareLevel(tex_0_11_, sampler_0_5_, vec2<f32>(((((_e1122 + (((in0_.x * _e1101) + (in0_.y * _e1108)) + (in0_.z * _e1115))) / _e1230) * 0.5f) + 0.5f), (0.5f - (((_e1156 + (((in0_.x * _e1135) + (in0_.y * _e1142)) + (in0_.z * _e1149))) / _e1230) * 0.5f))), _e1088, (((_e1190 + (((in0_.x * _e1169) + (in0_.y * _e1176)) + (in0_.z * _e1183))) / _e1230) + -0.001f));
                if (_e1088 < 3u) {
                    let _e1252 = ((_e1091 - mix(mix(mix(_e1054, _e1059, _e1077), _e1064, _e1081), _e1069, _e1085)) * 0.08f);
                    let _e1258 = ((_e1091 - _e1049) / select(_e1252, 0.000001f, (_e1252 < 0.000001f)));
                    let _e1262 = select(_e1258, 0f, (_e1258 < 0f));
                    let _e1268 = (_e1088 + 1u);
                    let _e1275 = uniform_struct_0_3_.f6_.f0_[_e1268].f0_.x;
                    let _e1282 = uniform_struct_0_3_.f6_.f0_[_e1268].f1_.x;
                    let _e1289 = uniform_struct_0_3_.f6_.f0_[_e1268].f2_.x;
                    let _e1296 = uniform_struct_0_3_.f6_.f0_[_e1268].f3_.x;
                    let _e1309 = uniform_struct_0_3_.f6_.f0_[_e1268].f0_.y;
                    let _e1316 = uniform_struct_0_3_.f6_.f0_[_e1268].f1_.y;
                    let _e1323 = uniform_struct_0_3_.f6_.f0_[_e1268].f2_.y;
                    let _e1330 = uniform_struct_0_3_.f6_.f0_[_e1268].f3_.y;
                    let _e1343 = uniform_struct_0_3_.f6_.f0_[_e1268].f0_.z;
                    let _e1350 = uniform_struct_0_3_.f6_.f0_[_e1268].f1_.z;
                    let _e1357 = uniform_struct_0_3_.f6_.f0_[_e1268].f2_.z;
                    let _e1364 = uniform_struct_0_3_.f6_.f0_[_e1268].f3_.z;
                    let _e1377 = uniform_struct_0_3_.f6_.f0_[_e1268].f0_.w;
                    let _e1384 = uniform_struct_0_3_.f6_.f0_[_e1268].f1_.w;
                    let _e1391 = uniform_struct_0_3_.f6_.f0_[_e1268].f2_.w;
                    let _e1398 = uniform_struct_0_3_.f6_.f0_[_e1268].f3_.w;
                    let _e1404 = (_e1398 + (((in0_.x * _e1377) + (in0_.y * _e1384)) + (in0_.z * _e1391)));
                    let _e1421 = textureSampleCompareLevel(tex_0_11_, sampler_0_5_, vec2<f32>(((((_e1296 + (((in0_.x * _e1275) + (in0_.y * _e1282)) + (in0_.z * _e1289))) / _e1404) * 0.5f) + 0.5f), (0.5f - (((_e1330 + (((in0_.x * _e1309) + (in0_.y * _e1316)) + (in0_.z * _e1323))) / _e1404) * 0.5f))), _e1268, (((_e1364 + (((in0_.x * _e1343) + (in0_.y * _e1350)) + (in0_.z * _e1357))) / _e1404) + -0.001f));
                    vis_u002e_sroa_u002e_0_u002e_0_u002e_i = mix(_e1421, _e1247, select(_e1262, 1f, (_e1262 > 1f)));
                } else {
                    vis_u002e_sroa_u002e_0_u002e_0_u002e_i = _e1247;
                }
                let _e1426 = vis_u002e_sroa_u002e_0_u002e_0_u002e_i;
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (_e1016 * _e1426);
                radiance_u002e_i_u002e_sroa_u002e_10_u002e_0_ = (_e1017 * _e1426);
                radiance_u002e_i_u002e_sroa_u002e_15_u002e_0_ = (_e1018 * _e1426);
            } else {
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_ = _e1016;
                radiance_u002e_i_u002e_sroa_u002e_10_u002e_0_ = _e1017;
                radiance_u002e_i_u002e_sroa_u002e_15_u002e_0_ = _e1018;
            }
            let _e1437 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_;
            let _e1439 = radiance_u002e_i_u002e_sroa_u002e_10_u002e_0_;
            let _e1441 = radiance_u002e_i_u002e_sroa_u002e_15_u002e_0_;
            let _e1443 = (_e860 + 19u);
            let _e1446 = storage_0_13_[_e1443];
            if !((bitcast<f32>(_e1446) >= 0f)) {
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = _e1437;
                radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_ = _e1439;
                radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_ = _e1441;
            } else {
                if !((_e1021 <= 0.5f)) {
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e1437;
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_1_ = _e1439;
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_1_ = _e1441;
                } else {
                    let _e1464 = storage_0_13_[(_e860 + 20u)];
                    let _e1470 = storage_0_13_[(_e860 + 24u)];
                    let _e1476 = storage_0_13_[(_e860 + 28u)];
                    let _e1482 = storage_0_13_[(_e860 + 32u)];
                    let _e1494 = storage_0_13_[(_e860 + 21u)];
                    let _e1500 = storage_0_13_[(_e860 + 25u)];
                    let _e1506 = storage_0_13_[(_e860 + 29u)];
                    let _e1512 = storage_0_13_[(_e860 + 33u)];
                    let _e1524 = storage_0_13_[(_e860 + 22u)];
                    let _e1530 = storage_0_13_[(_e860 + 26u)];
                    let _e1536 = storage_0_13_[(_e860 + 30u)];
                    let _e1542 = storage_0_13_[(_e860 + 34u)];
                    let _e1554 = storage_0_13_[(_e860 + 23u)];
                    let _e1560 = storage_0_13_[(_e860 + 27u)];
                    let _e1566 = storage_0_13_[(_e860 + 31u)];
                    let _e1572 = storage_0_13_[(_e860 + 35u)];
                    let _e1579 = ((((in0_.x * bitcast<f32>(_e1554)) + (in0_.y * bitcast<f32>(_e1560))) + (in0_.z * bitcast<f32>(_e1566))) + bitcast<f32>(_e1572));
                    let _e1592 = storage_0_13_[_e1443];
                    let _e1601 = textureSampleCompareLevel(tex_0_4_, sampler_0_5_, vec2<f32>((((((((in0_.x * bitcast<f32>(_e1464)) + (in0_.y * bitcast<f32>(_e1470))) + (in0_.z * bitcast<f32>(_e1476))) + bitcast<f32>(_e1482)) / _e1579) * 0.5f) + 0.5f), (0.5f - ((((((in0_.x * bitcast<f32>(_e1494)) + (in0_.y * bitcast<f32>(_e1500))) + (in0_.z * bitcast<f32>(_e1506))) + bitcast<f32>(_e1512)) / _e1579) * 0.5f))), u32(bitcast<f32>(_e1592)), ((((((in0_.x * bitcast<f32>(_e1524)) + (in0_.y * bitcast<f32>(_e1530))) + (in0_.z * bitcast<f32>(_e1536))) + bitcast<f32>(_e1542)) / _e1579) + -0.001f));
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_ = (_e1437 * _e1601);
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_1_ = (_e1439 * _e1601);
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_1_ = (_e1441 * _e1601);
                }
                let _e1609 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_;
                let _e1611 = radiance_u002e_i_u002e_sroa_u002e_10_u002e_1_;
                let _e1613 = radiance_u002e_i_u002e_sroa_u002e_15_u002e_1_;
                if (_e1021 > 0.5f) {
                    let _e1618 = storage_0_13_[_e866];
                    let _e1622 = storage_0_13_[_e872];
                    let _e1626 = storage_0_13_[_e878];
                    let _e1628 = (in0_.x - bitcast<f32>(_e1618));
                    let _e1629 = (in0_.y - bitcast<f32>(_e1622));
                    let _e1630 = (in0_.z - bitcast<f32>(_e1626));
                    let _e1631 = abs(_e1628);
                    let _e1632 = abs(_e1629);
                    let _e1634 = select(_e1631, _e1632, (_e1632 > _e1631));
                    let _e1635 = abs(_e1630);
                    let _e1640 = storage_0_13_[_e894];
                    let _e1641 = bitcast<f32>(_e1640);
                    let _e1645 = select(_e1641, 1f, (_e1641 < 1f));
                    let _e1647 = (_e1645 + -0.1f);
                    let _e1658 = storage_0_13_[_e1443];
                    let _e1664 = textureSampleCompareLevel(tex_0_6_, sampler_0_5_, vec3<f32>(_e1628, _e1629, _e1630), u32(bitcast<f32>(_e1658)), (((_e1645 / _e1647) - ((_e1645 * 0.1f) / (select(_e1634, _e1635, (_e1635 > _e1634)) * _e1647))) + -0.001f));
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = (_e1609 * _e1664);
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_ = (_e1611 * _e1664);
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_ = (_e1613 * _e1664);
                } else {
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = _e1609;
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_ = _e1611;
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_ = _e1613;
                }
            }
            let _e1675 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_;
            let _e1677 = radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_;
            let _e1679 = radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_;
            let _e1682 = storage_0_13_[_e860];
            let _e1686 = storage_0_13_[_e866];
            let _e1690 = storage_0_13_[_e872];
            let _e1694 = storage_0_13_[_e878];
            let _e1696 = (bitcast<f32>(_e1686) - in0_.x);
            let _e1697 = (bitcast<f32>(_e1690) - in0_.y);
            let _e1698 = (bitcast<f32>(_e1694) - in0_.z);
            let _e1700 = length(vec3<f32>(_e1696, _e1697, _e1698));
            let _e1706 = (1f / select(_e1700, 0.000001f, (_e1700 < 0.000001f)));
            let _e1714 = select(1f, 0f, (bitcast<f32>(_e1682) < 0.5f));
            let _e1717 = storage_0_13_[_e953];
            let _e1718 = bitcast<f32>(_e1717);
            let _e1721 = storage_0_13_[_e959];
            let _e1722 = bitcast<f32>(_e1721);
            let _e1725 = storage_0_13_[_e965];
            let _e1726 = bitcast<f32>(_e1725);
            let _e1730 = (1f / length(vec3<f32>(_e1718, _e1722, _e1726)));
            let _e1734 = mix((_e1730 * _e1718), (_e1696 * _e1706), _e1714);
            let _e1735 = mix((_e1730 * _e1722), (_e1697 * _e1706), _e1714);
            let _e1736 = mix((_e1730 * _e1726), (_e1698 * _e1706), _e1714);
            let _e1740 = (1f / length(vec3<f32>(_e1734, _e1735, _e1736)));
            let _e1741 = (_e1734 * _e1740);
            let _e1742 = (_e1735 * _e1740);
            let _e1743 = (_e1736 * _e1740);
            let _e1748 = ((_e37 + _e1741) + 0f);
            let _e1750 = ((_e38 + _e1742) + 0f);
            let _e1752 = ((_e39 + _e1743) + 0.000001f);
            let _e1756 = (1f / length(vec3<f32>(_e1748, _e1750, _e1752)));
            let _e1757 = (_e1756 * _e1748);
            let _e1758 = (_e1756 * _e1750);
            let _e1759 = (_e1756 * _e1752);
            let _e1762 = dot(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e1741, _e1742, _e1743));
            let _e1766 = select(_e1762, 0.000001f, (_e1762 < 0.000001f));
            let _e1769 = dot(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e1757, _e1758, _e1759));
            let _e1772 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(_e1757, _e1758, _e1759));
            let _e1776 = select(_e1772, 0.000001f, (_e1772 < 0.000001f));
            if (_e1766 > 0.000001f) {
                let _e1782 = select(_e1769, 0.000001f, (_e1769 < 0.000001f));
                let _e1783 = mix(_e713, _e477, _e463);
                let _e1784 = mix(_e713, _e478, _e463);
                let _e1785 = mix(_e713, _e479, _e463);
                let _e1788 = pow(0.85714287f, 5f);
                let _e1790 = (1f - _e1783);
                let _e1792 = (_e1783 + (_e1790 * _e1788));
                let _e1794 = (1f - _e1784);
                let _e1796 = (_e1784 + (_e1794 * _e1788));
                let _e1798 = (1f - _e1785);
                let _e1800 = (_e1785 + (_e1798 * _e1788));
                let _e1805 = (1f - _e1776);
                let _e1814 = ((_e1776 * pow(_e1805, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e1816 = pow(_e1805, 5f);
                let _e1838 = mix(_e713, _e477, _e463);
                let _e1839 = mix(_e713, _e478, _e463);
                let _e1840 = mix(_e713, _e479, _e463);
                let _e1843 = pow(0.85714287f, 5f);
                let _e1845 = (1f - _e1838);
                let _e1847 = (_e1838 + (_e1845 * _e1843));
                let _e1849 = (1f - _e1839);
                let _e1851 = (_e1839 + (_e1849 * _e1843));
                let _e1853 = (1f - _e1840);
                let _e1855 = (_e1840 + (_e1853 * _e1843));
                let _e1867 = ((_e1776 * pow(_e1805, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e1869 = pow(_e1805, 5f);
                let _e1892 = (_e717 * sqrt(_e724));
                let _e1895 = (_e719 * (_e717 * sqrt(_e724)));
                let _e1898 = dot(vec3<f32>(_e1757, _e1758, _e1759), vec3<f32>(_e54, _e55, _e56));
                let _e1901 = dot(vec3<f32>(_e1757, _e1758, _e1759), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                let _e1903 = (_e1892 * _e1892);
                let _e1906 = (_e1895 * _e1895);
                let _e1909 = (_e1782 * _e1782);
                let _e1910 = (_e1909 + (((_e1898 * _e1898) / _e1903) + ((_e1901 * _e1901) / _e1906)));
                let _e1917 = (1f / (_e1910 * (((_e1892 * 3.1415927f) * _e1895) * _e1910)));
                let _e1920 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(_e54, _e55, _e56));
                let _e1923 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                let _e1926 = dot(vec3<f32>(_e1741, _e1742, _e1743), vec3<f32>(_e54, _e55, _e56));
                let _e1929 = dot(vec3<f32>(_e1741, _e1742, _e1743), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                let _e1935 = (_e1766 * _e1766);
                let _e1955 = (0.5f / max(((_e46 * sqrt(max((_e1935 + ((_e1903 * (_e1926 * _e1926)) + (_e1906 * (_e1929 * _e1929)))), 0.000001f))) + (_e1766 * sqrt(max((_e393 + ((_e1903 * (_e1920 * _e1920)) + (_e1906 * (_e1923 * _e1923)))), 0.000001f)))), 0.000001f));
                let _e1974 = (1f / length(vec3<f32>(_e730, _e731, _e732)));
                let _e1981 = (_e1741 - (_e15 * _e1766));
                let _e1982 = (_e1742 - (_e16 * _e1766));
                let _e1983 = (_e1743 - (_e17 * _e1766));
                let _e1987 = (1f / length(vec3<f32>(_e1981, _e1982, _e1983)));
                let _e1993 = dot(vec3<f32>((_e730 * _e1974), (_e731 * _e1974), (_e732 * _e1974)), vec3<f32>((_e1981 * _e1987), (_e1982 * _e1987), (_e1983 * _e1987)));
                let _e1998 = acos(max(_e46, 0f));
                let _e2001 = acos(max(_e1766, 0f));
                let _e2025 = (max((1f - dot(vec3<f32>((max(((_e1783 + (_e1790 * _e1816)) - ((_e1792 - (_e1792 * _e714)) * _e1814)), 0f) * _e465), (max(((_e1784 + (_e1794 * _e1816)) - ((_e1796 - (_e1796 * _e715)) * _e1814)), 0f) * _e465), (max(((_e1785 + (_e1798 * _e1816)) - ((_e1800 - (_e1800 * _e716)) * _e1814)), 0f) * _e465)), vec3<f32>(0.2126f, 0.7152f, 0.0722f))), 0f) * (((1f - ((_e741 * 0.5f) / (_e741 + 0.57f))) + (tan(min(_e1998, _e2001)) * ((((_e741 * 0.45f) / (_e741 + 0.09f)) * select(0f, _e1993, (_e1993 > 0f))) * sin(max(_e1998, _e2001))))) * 0.31830987f));
                if _e485 {
                    let _e2046 = (_e761 * sqrt(_e768));
                    let _e2049 = (_e763 * (_e761 * sqrt(_e768)));
                    let _e2052 = dot(vec3<f32>(_e1757, _e1758, _e1759), vec3<f32>(_e54, _e55, _e56));
                    let _e2055 = dot(vec3<f32>(_e1757, _e1758, _e1759), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                    let _e2057 = (_e2046 * _e2046);
                    let _e2060 = (_e2049 * _e2049);
                    let _e2063 = (_e1909 + (((_e2052 * _e2052) / _e2057) + ((_e2055 * _e2055) / _e2060)));
                    let _e2073 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(_e54, _e55, _e56));
                    let _e2076 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                    let _e2079 = dot(vec3<f32>(_e1741, _e1742, _e1743), vec3<f32>(_e54, _e55, _e56));
                    let _e2082 = dot(vec3<f32>(_e1741, _e1742, _e1743), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                    let _e2109 = ((_e758 + ((1f - _e758) * pow(_e1805, 5f))) * ((1f / (_e2063 * (((_e2046 * 3.1415927f) * _e2049) * _e2063))) * (0.5f / max(((_e46 * sqrt(max((_e1935 + ((_e2057 * (_e2079 * _e2079)) + (_e2060 * (_e2082 * _e2082)))), 0.000001f))) + (_e1766 * sqrt(max((_e393 + ((_e2057 * (_e2073 * _e2073)) + (_e2060 * (_e2076 * _e2076)))), 0.000001f)))), 0.000001f))));
                    coat_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e2109 * bitcast<f32>(_e322)) * _e461);
                    coat_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e2109 * bitcast<f32>(_e327)) * _e461);
                    coat_u002e_i_u002e_sroa_u002e_8_u002e_0_ = ((_e2109 * bitcast<f32>(_e332)) * _e461);
                } else {
                    coat_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    coat_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    coat_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2126 = coat_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2128 = coat_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2130 = coat_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e772 <= 0f)) {
                    let _e2154 = ((_e1776 * ((1f / ((_e777 * (((1f / _e46) + (1f / _e1766)) + -2f)) + 1f)) * (_e777 / (pow((((_e777 + -1f) * _e1909) + 1f), 2f) * 3.1415927f)))) / max(((_e46 * 4f) * _e1766), 0.000001f));
                    fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e2154 * bitcast<f32>(_e287)) * _e772);
                    fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e2154 * bitcast<f32>(_e292)) * _e772);
                    fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_ = ((_e2154 * bitcast<f32>(_e297)) * _e772);
                } else {
                    fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2171 = fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2173 = fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2175 = fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e787 <= 0f)) {
                    let _e2177 = (_e717 * sqrt(_e724));
                    if !((_e788 <= 0f)) {
                        extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (-(log(max(bitcast<f32>(_e197), 0.000001f))) / _e788);
                        extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (-(log(max(bitcast<f32>(_e202), 0.000001f))) / _e788);
                        extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (-(log(max(bitcast<f32>(_e207), 0.000001f))) / _e788);
                    } else {
                        extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e2197 = extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                    let _e2199 = extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                    let _e2201 = extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                    let _e2208 = (_e2177 * _e2177);
                    let _e2213 = ((_e1909 * (_e2208 + -1f)) + 1f);
                    let _e2219 = (1f - _e2208);
                    let _e2238 = ((1f - (_e713 + ((1f - _e713) * pow((1f - _e46), 5f)))) * ((_e2208 / (_e2213 * (_e2213 * 3.1415927f))) * (0.5f / max(((_e46 * sqrt(max((_e2208 + (_e1935 * _e2219)), 0.000001f))) + (_e1766 * sqrt(max((_e2208 + (_e393 * _e2219)), 0.000001f)))), 0.000001f))));
                    transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (((exp((_e788 * -(_e2197))) * _e2238) * bitcast<f32>(_e197)) * _e787);
                    transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (((exp((_e788 * -(_e2199))) * _e2238) * bitcast<f32>(_e202)) * _e787);
                    transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (((exp((_e788 * -(_e2201))) * _e2238) * bitcast<f32>(_e207)) * _e787);
                } else {
                    transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2267 = transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2269 = transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2271 = transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e466 <= 0f)) {
                    let _e2276 = length(vec3<f32>((_e730 - _e1981), (_e731 - _e1982), (_e732 - _e1983)));
                    let _e2278 = sqrt(3f);
                    let _e2301 = -(_e2276);
                    let _e2309 = (max(((clamp(((_e46 * _e1766) + (((_e397 + max((1f - _e1935), 0f)) - (_e2276 * _e2276)) * 0.5f)), -1f, 1f) * bitcast<f32>(_e247)) + 1f), 0f) * 0.31830987f);
                    subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (((exp((((1f / max((_e818 * bitcast<f32>(_e242)), 0.000001f)) * _e2278) * _e2301)) * _e2309) * _e480) * _e466);
                    subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (((exp((((1f / max((_e818 * bitcast<f32>(_e267)), 0.000001f)) * _e2278) * _e2301)) * _e2309) * _e481) * _e466);
                    subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (((exp((((1f / max((_e818 * bitcast<f32>(_e272)), 0.000001f)) * _e2278) * _e2301)) * _e2309) * _e482) * _e466);
                } else {
                    subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2329 = subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2331 = subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2333 = subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                lo_u002e_sroa_u002e_0_u002e_1_ = (_e849 + (_e1675 * (_e1766 * ((((_e560 * ((_e450 * ((((max(((_e1838 + (_e1845 * _e1869)) - ((_e1847 - (_e1847 * _e714)) * _e1867)), 0f) * _e465) * _e1917) * _e1955) * _e464)) + (_e450 * (((_e735 * _e474) * _e2025) * _e464)))) + _e2126) + (_e2171 + _e2267)) + _e2329))));
                lo_u002e_sroa_u002e_6_u002e_1_ = (_e847 + (_e1677 * (_e1766 * ((((_e562 * ((_e455 * ((((max(((_e1839 + (_e1849 * _e1869)) - ((_e1851 - (_e1851 * _e715)) * _e1867)), 0f) * _e465) * _e1917) * _e1955) * _e464)) + (_e455 * (((_e735 * _e475) * _e2025) * _e464)))) + _e2128) + (_e2173 + _e2269)) + _e2331))));
                lo_u002e_sroa_u002e_9_u002e_1_ = (_e845 + (_e1679 * (_e1766 * ((((_e564 * ((_e460 * ((((max(((_e1840 + (_e1853 * _e1869)) - ((_e1855 - (_e1855 * _e716)) * _e1867)), 0f) * _e465) * _e1917) * _e1955) * _e464)) + (_e460 * (((_e735 * _e476) * _e2025) * _e464)))) + _e2130) + (_e2175 + _e2271)) + _e2333))));
            } else {
                lo_u002e_sroa_u002e_0_u002e_1_ = _e849;
                lo_u002e_sroa_u002e_6_u002e_1_ = _e847;
                lo_u002e_sroa_u002e_9_u002e_1_ = _e845;
            }
            let _e2362 = lo_u002e_sroa_u002e_0_u002e_1_;
            out_u002e_lo_u002e_sroa_u002e_0_u002e_1_ = _e2362;
            let _e2365 = lo_u002e_sroa_u002e_6_u002e_1_;
            out_u002e_lo_u002e_sroa_u002e_6_u002e_1_ = _e2365;
            let _e2368 = lo_u002e_sroa_u002e_9_u002e_1_;
            out_u002e_lo_u002e_sroa_u002e_9_u002e_1_ = _e2368;
            let _e2371 = (_e843 + 1u);
            if (_e2371 == _e691) {
                lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e2362;
                lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e2365;
                lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e2368;
                break;
            } else {
                k_u002e_sroa_u002e_0_u002e_0_ = _e2371;
                lo_u002e_sroa_u002e_9_ = _e2368;
                lo_u002e_sroa_u002e_6_ = _e2365;
                lo_u002e_sroa_u002e_0_ = _e2362;
                continue;
            }
        }
    }
    let _e2387 = lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
    let _e2389 = lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
    let _e2391 = lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
    let _e2401 = uniform_struct_0_3_.f0_.x;
    let _e2405 = uniform_struct_0_3_.f0_.y;
    let _e2409 = uniform_struct_0_3_.f0_.z;
    let _e2413 = bitcast<f32>(_e357);
    if !((_e2413 <= 0f)) {
        let _e2420 = (_e2413 * bitcast<f32>(_e362));
        let _e2421 = (_e2413 * bitcast<f32>(_e367));
        let _e2422 = (_e2413 * bitcast<f32>(_e372));
        let _e2432 = ((1f - _e461) + ((1f - pow((1f - _e46), 5f)) * _e461));
        emission_u002e_sroa_u002e_0_u002e_1_ = (_e2420 + ((((_e2420 * bitcast<f32>(_e322)) * _e2432) - _e2420) * _e461));
        emission_u002e_sroa_u002e_5_u002e_1_ = (_e2421 + ((((_e2421 * bitcast<f32>(_e327)) * _e2432) - _e2421) * _e461));
        emission_u002e_sroa_u002e_8_u002e_1_ = (_e2422 + ((((_e2422 * bitcast<f32>(_e332)) * _e2432) - _e2422) * _e461));
    } else {
        emission_u002e_sroa_u002e_0_u002e_1_ = 0f;
        emission_u002e_sroa_u002e_5_u002e_1_ = 0f;
        emission_u002e_sroa_u002e_8_u002e_1_ = 0f;
    }
    let _e2461 = emission_u002e_sroa_u002e_0_u002e_1_;
    let _e2463 = emission_u002e_sroa_u002e_5_u002e_1_;
    let _e2465 = emission_u002e_sroa_u002e_8_u002e_1_;
    let _e2466 = bitcast<f32>(_e152);
    let _e2471 = ((_e2466 + -1f) / (_e2466 + 1f));
    let _e2472 = (_e2471 * _e2471);
    let _e2480 = (dot(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e37, _e38, _e39)) * 2f);
    let _e2484 = ((_e15 * _e2480) - _e37);
    let _e2485 = ((_e16 * _e2480) - _e38);
    let _e2486 = ((_e17 * _e2480) - _e39);
    let _e2489 = uniform_struct_0_3_.f4_;
    let _e2490 = (_e147 * _e2489);
    let _e2494 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2484, _e2485, _e2486), _e2490);
    let _e2499 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2484, _e2485, _e2486), _e2490);
    let _e2504 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2484, _e2485, _e2486), _e2490);
    let _e2509 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e15, _e16, _e17));
    let _e2514 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e15, _e16, _e17));
    let _e2519 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e15, _e16, _e17));
    let _e2524 = textureSample(tex_0_9_, sampler_0_10_, vec2<f32>(_e46, _e147));
    let _e2529 = textureSample(tex_0_9_, sampler_0_10_, vec2<f32>(_e46, _e147));
    let _e2532 = (1f - _e463);
    let _e2538 = uniform_struct_0_3_.f3_;
    let _e2551 = max((1f - _e2529.y), 0f);
    let _e2576 = !((bitcast<f32>(_e387) >= 0.5f));
    let _e2577 = select(_e2461, (((_e2387 + (mix(_e474, (_e474 * _e465), _e463) * _e2401)) + _e2461) + (_e2538 * ((_e2494.x * ((mix(_e2472, _e477, _e463) * _e2524.x) + _e2529.y)) + (((_e2532 * _e474) * _e2509.x) * _e2551)))), _e2576);
    let _e2578 = select(_e2463, (((_e2389 + (mix(_e475, (_e475 * _e465), _e463) * _e2405)) + _e2463) + (_e2538 * ((_e2499.y * ((mix(_e2472, _e478, _e463) * _e2524.x) + _e2529.y)) + (((_e2532 * _e475) * _e2514.y) * _e2551)))), _e2576);
    let _e2579 = select(_e2465, (((_e2391 + (mix(_e476, (_e476 * _e465), _e463) * _e2409)) + _e2465) + (_e2538 * ((_e2504.z * ((mix(_e2472, _e479, _e463) * _e2524.x) + _e2529.y)) + (((_e2532 * _e476) * _e2519.z) * _e2551)))), _e2576);
    if ((_e2577 <= _e2577) && (0f <= 0f)) {
        if (abs(_e2577) < 100000000000000000000f) {
            _2_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2577;
        } else {
            _2_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        _2_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2593 = _2_u002e_sroa_u002e_0_u002e_0_u002e_i;
    if ((_e2578 <= _e2578) && (0f <= 0f)) {
        if (abs(_e2578) < 100000000000000000000f) {
            _3_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2578;
        } else {
            _3_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        _3_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2607 = _3_u002e_sroa_u002e_0_u002e_0_u002e_i;
    if ((_e2579 <= _e2579) && (0f <= 0f)) {
        if (abs(_e2579) < 100000000000000000000f) {
            _4_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2579;
            let _e2617 = _4_u002e_sroa_u002e_0_u002e_0_u002e_i;
            fragment_out = vec4<f32>(_e2593, _e2607, _e2617, bitcast<f32>(_e377));
            let _e2622 = fragment_out;
            return _e2622;
        }
        _4_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    } else {
        _4_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2628 = _4_u002e_sroa_u002e_0_u002e_0_u002e_i;
    fragment_out = vec4<f32>(_e2593, _e2607, _e2628, bitcast<f32>(_e377));
    let _e2633 = fragment_out;
    return _e2633;
}
