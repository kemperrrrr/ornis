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
    var k_u002e_sroa_u002e_0_u002e_0_u002e_i1522_: u32;
    var lo_u002e_sroa_u002e_9_u002e_01521_: f32;
    var lo_u002e_sroa_u002e_6_u002e_01520_: f32;
    var lo_u002e_sroa_u002e_0_u002e_01519_: f32;
    var _2_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var lo_u002e_sroa_u002e_0_u002e_1_: f32;
    var lo_u002e_sroa_u002e_6_u002e_1_: f32;
    var lo_u002e_sroa_u002e_9_u002e_1_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_: f32;
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
        k_u002e_sroa_u002e_0_u002e_0_u002e_i1522_ = 0u;
        lo_u002e_sroa_u002e_9_u002e_01521_ = 0f;
        lo_u002e_sroa_u002e_6_u002e_01520_ = 0f;
        lo_u002e_sroa_u002e_0_u002e_01519_ = 0f;
        loop {
            let _e843 = k_u002e_sroa_u002e_0_u002e_0_u002e_i1522_;
            let _e845 = lo_u002e_sroa_u002e_9_u002e_01521_;
            let _e847 = lo_u002e_sroa_u002e_6_u002e_01520_;
            let _e849 = lo_u002e_sroa_u002e_0_u002e_01519_;
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
            let _e926 = storage_0_13_[(_e860 + 15u)];
            let _e930 = storage_0_13_[_e866];
            let _e934 = storage_0_13_[_e872];
            let _e938 = storage_0_13_[_e878];
            let _e940 = (in0_.x - bitcast<f32>(_e930));
            let _e941 = (in0_.y - bitcast<f32>(_e934));
            let _e942 = (in0_.z - bitcast<f32>(_e938));
            let _e946 = (1f / length(vec3<f32>(_e940, _e941, _e942)));
            let _e951 = (_e860 + 4u);
            let _e954 = storage_0_13_[_e951];
            let _e955 = bitcast<f32>(_e954);
            let _e957 = (_e860 + 5u);
            let _e960 = storage_0_13_[_e957];
            let _e961 = bitcast<f32>(_e960);
            let _e963 = (_e860 + 6u);
            let _e966 = storage_0_13_[_e963];
            let _e967 = bitcast<f32>(_e966);
            let _e971 = (1f / length(vec3<f32>(_e955, _e961, _e967)));
            let _e982 = storage_0_13_[(_e860 + 18u)];
            let _e988 = storage_0_13_[(_e860 + 17u)];
            let _e995 = select(1f, 0f, (_e864 < 1.5f));
            let _e998 = (_e887 * _e887);
            let _e1010 = (mix(1f, smoothstep(bitcast<f32>(_e982), bitcast<f32>(_e988), dot(vec3<f32>((_e971 * _e955), (_e971 * _e961), (_e971 * _e967)), vec3<f32>((_e940 * _e946), (_e941 * _e946), (_e942 * _e946)))), _e995) * ((bitcast<f32>(_e910) * bitcast<f32>(_e926)) * (mix(1f, (1f - smoothstep((_e898 * 0.99f), _e898, _e887)), _e892) * mix(1f, (1f / select(_e998, 0.000001f, (_e998 < 0.000001f))), _e892))));
            let _e1013 = (_e892 * (1f - _e995));
            if (((1f - _e892) * select(1f, 0f, (_e898 < 0.5f))) > 0.5f) {
                let _e1028 = uniform_struct_0_3_.f6_.f1_.x;
                let _e1033 = uniform_struct_0_3_.f6_.f1_.y;
                let _e1038 = uniform_struct_0_3_.f6_.f1_.z;
                let _e1041 = dot(vec3<f32>((in0_.x - _e21), (in0_.y - _e25), (in0_.z - _e29)), vec3<f32>(_e1028, _e1033, _e1038));
                let _e1046 = uniform_struct_0_3_.f6_.f2_.x;
                let _e1051 = uniform_struct_0_3_.f6_.f2_.y;
                let _e1056 = uniform_struct_0_3_.f6_.f2_.z;
                let _e1061 = uniform_struct_0_3_.f6_.f2_.w;
                let _e1065 = uniform_struct_0_3_.f6_.f3_;
                let _e1069 = select(1f, 0f, (_e1041 < _e1051));
                let _e1073 = select(1f, 0f, (_e1041 < _e1056));
                let _e1077 = select(1f, 0f, (_e1041 < _e1061));
                let _e1080 = u32(((_e1069 + _e1073) + _e1077));
                let _e1083 = mix(mix(mix(_e1051, _e1056, _e1069), _e1061, _e1073), _e1065, _e1077);
                let _e1093 = uniform_struct_0_3_.f6_.f0_[_e1080].f0_.x;
                let _e1100 = uniform_struct_0_3_.f6_.f0_[_e1080].f1_.x;
                let _e1107 = uniform_struct_0_3_.f6_.f0_[_e1080].f2_.x;
                let _e1114 = uniform_struct_0_3_.f6_.f0_[_e1080].f3_.x;
                let _e1127 = uniform_struct_0_3_.f6_.f0_[_e1080].f0_.y;
                let _e1134 = uniform_struct_0_3_.f6_.f0_[_e1080].f1_.y;
                let _e1141 = uniform_struct_0_3_.f6_.f0_[_e1080].f2_.y;
                let _e1148 = uniform_struct_0_3_.f6_.f0_[_e1080].f3_.y;
                let _e1161 = uniform_struct_0_3_.f6_.f0_[_e1080].f0_.z;
                let _e1168 = uniform_struct_0_3_.f6_.f0_[_e1080].f1_.z;
                let _e1175 = uniform_struct_0_3_.f6_.f0_[_e1080].f2_.z;
                let _e1182 = uniform_struct_0_3_.f6_.f0_[_e1080].f3_.z;
                let _e1195 = uniform_struct_0_3_.f6_.f0_[_e1080].f0_.w;
                let _e1202 = uniform_struct_0_3_.f6_.f0_[_e1080].f1_.w;
                let _e1209 = uniform_struct_0_3_.f6_.f0_[_e1080].f2_.w;
                let _e1216 = uniform_struct_0_3_.f6_.f0_[_e1080].f3_.w;
                let _e1222 = (_e1216 + (((in0_.x * _e1195) + (in0_.y * _e1202)) + (in0_.z * _e1209)));
                let _e1239 = textureSampleCompareLevel(tex_0_11_, sampler_0_5_, vec2<f32>(((((_e1114 + (((in0_.x * _e1093) + (in0_.y * _e1100)) + (in0_.z * _e1107))) / _e1222) * 0.5f) + 0.5f), (0.5f - (((_e1148 + (((in0_.x * _e1127) + (in0_.y * _e1134)) + (in0_.z * _e1141))) / _e1222) * 0.5f))), _e1080, (((_e1182 + (((in0_.x * _e1161) + (in0_.y * _e1168)) + (in0_.z * _e1175))) / _e1222) + -0.001f));
                if (_e1080 < 3u) {
                    let _e1244 = ((_e1083 - mix(mix(mix(_e1046, _e1051, _e1069), _e1056, _e1073), _e1061, _e1077)) * 0.08f);
                    let _e1250 = ((_e1083 - _e1041) / select(_e1244, 0.000001f, (_e1244 < 0.000001f)));
                    let _e1254 = select(_e1250, 0f, (_e1250 < 0f));
                    let _e1260 = (_e1080 + 1u);
                    let _e1267 = uniform_struct_0_3_.f6_.f0_[_e1260].f0_.x;
                    let _e1274 = uniform_struct_0_3_.f6_.f0_[_e1260].f1_.x;
                    let _e1281 = uniform_struct_0_3_.f6_.f0_[_e1260].f2_.x;
                    let _e1288 = uniform_struct_0_3_.f6_.f0_[_e1260].f3_.x;
                    let _e1301 = uniform_struct_0_3_.f6_.f0_[_e1260].f0_.y;
                    let _e1308 = uniform_struct_0_3_.f6_.f0_[_e1260].f1_.y;
                    let _e1315 = uniform_struct_0_3_.f6_.f0_[_e1260].f2_.y;
                    let _e1322 = uniform_struct_0_3_.f6_.f0_[_e1260].f3_.y;
                    let _e1335 = uniform_struct_0_3_.f6_.f0_[_e1260].f0_.z;
                    let _e1342 = uniform_struct_0_3_.f6_.f0_[_e1260].f1_.z;
                    let _e1349 = uniform_struct_0_3_.f6_.f0_[_e1260].f2_.z;
                    let _e1356 = uniform_struct_0_3_.f6_.f0_[_e1260].f3_.z;
                    let _e1369 = uniform_struct_0_3_.f6_.f0_[_e1260].f0_.w;
                    let _e1376 = uniform_struct_0_3_.f6_.f0_[_e1260].f1_.w;
                    let _e1383 = uniform_struct_0_3_.f6_.f0_[_e1260].f2_.w;
                    let _e1390 = uniform_struct_0_3_.f6_.f0_[_e1260].f3_.w;
                    let _e1396 = (_e1390 + (((in0_.x * _e1369) + (in0_.y * _e1376)) + (in0_.z * _e1383)));
                    let _e1413 = textureSampleCompareLevel(tex_0_11_, sampler_0_5_, vec2<f32>(((((_e1288 + (((in0_.x * _e1267) + (in0_.y * _e1274)) + (in0_.z * _e1281))) / _e1396) * 0.5f) + 0.5f), (0.5f - (((_e1322 + (((in0_.x * _e1301) + (in0_.y * _e1308)) + (in0_.z * _e1315))) / _e1396) * 0.5f))), _e1260, (((_e1356 + (((in0_.x * _e1335) + (in0_.y * _e1342)) + (in0_.z * _e1349))) / _e1396) + -0.001f));
                    vis_u002e_sroa_u002e_0_u002e_0_u002e_i = mix(_e1413, _e1239, select(_e1254, 1f, (_e1254 > 1f)));
                } else {
                    vis_u002e_sroa_u002e_0_u002e_0_u002e_i = _e1239;
                }
                let _e1418 = vis_u002e_sroa_u002e_0_u002e_0_u002e_i;
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (_e1010 * _e1418);
            } else {
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_ = _e1010;
            }
            let _e1423 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_;
            let _e1425 = (_e860 + 19u);
            let _e1428 = storage_0_13_[_e1425];
            if !((bitcast<f32>(_e1428) >= 0f)) {
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = _e1423;
            } else {
                if !((_e1013 <= 0.5f)) {
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e1423;
                } else {
                    let _e1442 = storage_0_13_[(_e860 + 20u)];
                    let _e1448 = storage_0_13_[(_e860 + 24u)];
                    let _e1454 = storage_0_13_[(_e860 + 28u)];
                    let _e1460 = storage_0_13_[(_e860 + 32u)];
                    let _e1472 = storage_0_13_[(_e860 + 21u)];
                    let _e1478 = storage_0_13_[(_e860 + 25u)];
                    let _e1484 = storage_0_13_[(_e860 + 29u)];
                    let _e1490 = storage_0_13_[(_e860 + 33u)];
                    let _e1502 = storage_0_13_[(_e860 + 22u)];
                    let _e1508 = storage_0_13_[(_e860 + 26u)];
                    let _e1514 = storage_0_13_[(_e860 + 30u)];
                    let _e1520 = storage_0_13_[(_e860 + 34u)];
                    let _e1532 = storage_0_13_[(_e860 + 23u)];
                    let _e1538 = storage_0_13_[(_e860 + 27u)];
                    let _e1544 = storage_0_13_[(_e860 + 31u)];
                    let _e1550 = storage_0_13_[(_e860 + 35u)];
                    let _e1557 = ((((in0_.x * bitcast<f32>(_e1532)) + (in0_.y * bitcast<f32>(_e1538))) + (in0_.z * bitcast<f32>(_e1544))) + bitcast<f32>(_e1550));
                    let _e1570 = storage_0_13_[_e1425];
                    let _e1579 = textureSampleCompareLevel(tex_0_4_, sampler_0_5_, vec2<f32>((((((((in0_.x * bitcast<f32>(_e1442)) + (in0_.y * bitcast<f32>(_e1448))) + (in0_.z * bitcast<f32>(_e1454))) + bitcast<f32>(_e1460)) / _e1557) * 0.5f) + 0.5f), (0.5f - ((((((in0_.x * bitcast<f32>(_e1472)) + (in0_.y * bitcast<f32>(_e1478))) + (in0_.z * bitcast<f32>(_e1484))) + bitcast<f32>(_e1490)) / _e1557) * 0.5f))), u32(bitcast<f32>(_e1570)), ((((((in0_.x * bitcast<f32>(_e1502)) + (in0_.y * bitcast<f32>(_e1508))) + (in0_.z * bitcast<f32>(_e1514))) + bitcast<f32>(_e1520)) / _e1557) + -0.001f));
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_ = (_e1423 * _e1579);
                }
                let _e1583 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_;
                if (_e1013 > 0.5f) {
                    let _e1588 = storage_0_13_[_e866];
                    let _e1592 = storage_0_13_[_e872];
                    let _e1596 = storage_0_13_[_e878];
                    let _e1598 = (in0_.x - bitcast<f32>(_e1588));
                    let _e1599 = (in0_.y - bitcast<f32>(_e1592));
                    let _e1600 = (in0_.z - bitcast<f32>(_e1596));
                    let _e1601 = abs(_e1598);
                    let _e1602 = abs(_e1599);
                    let _e1604 = select(_e1601, _e1602, (_e1602 > _e1601));
                    let _e1605 = abs(_e1600);
                    let _e1610 = storage_0_13_[_e894];
                    let _e1611 = bitcast<f32>(_e1610);
                    let _e1615 = select(_e1611, 1f, (_e1611 < 1f));
                    let _e1617 = (_e1615 + -0.1f);
                    let _e1628 = storage_0_13_[_e1425];
                    let _e1634 = textureSampleCompareLevel(tex_0_6_, sampler_0_5_, vec3<f32>(_e1598, _e1599, _e1600), u32(bitcast<f32>(_e1628)), (((_e1615 / _e1617) - ((_e1615 * 0.1f) / (select(_e1604, _e1605, (_e1605 > _e1604)) * _e1617))) + -0.001f));
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = (_e1583 * _e1634);
                } else {
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = _e1583;
                }
            }
            let _e1639 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_;
            let _e1642 = storage_0_13_[_e860];
            let _e1646 = storage_0_13_[_e866];
            let _e1650 = storage_0_13_[_e872];
            let _e1654 = storage_0_13_[_e878];
            let _e1656 = (bitcast<f32>(_e1646) - in0_.x);
            let _e1657 = (bitcast<f32>(_e1650) - in0_.y);
            let _e1658 = (bitcast<f32>(_e1654) - in0_.z);
            let _e1660 = length(vec3<f32>(_e1656, _e1657, _e1658));
            let _e1666 = (1f / select(_e1660, 0.000001f, (_e1660 < 0.000001f)));
            let _e1674 = select(1f, 0f, (bitcast<f32>(_e1642) < 0.5f));
            let _e1677 = storage_0_13_[_e951];
            let _e1678 = bitcast<f32>(_e1677);
            let _e1681 = storage_0_13_[_e957];
            let _e1682 = bitcast<f32>(_e1681);
            let _e1685 = storage_0_13_[_e963];
            let _e1686 = bitcast<f32>(_e1685);
            let _e1690 = (1f / length(vec3<f32>(_e1678, _e1682, _e1686)));
            let _e1694 = mix((_e1690 * _e1678), (_e1656 * _e1666), _e1674);
            let _e1695 = mix((_e1690 * _e1682), (_e1657 * _e1666), _e1674);
            let _e1696 = mix((_e1690 * _e1686), (_e1658 * _e1666), _e1674);
            let _e1700 = (1f / length(vec3<f32>(_e1694, _e1695, _e1696)));
            let _e1701 = (_e1694 * _e1700);
            let _e1702 = (_e1695 * _e1700);
            let _e1703 = (_e1696 * _e1700);
            let _e1708 = ((_e37 + _e1701) + 0f);
            let _e1710 = ((_e38 + _e1702) + 0f);
            let _e1712 = ((_e39 + _e1703) + 0.000001f);
            let _e1716 = (1f / length(vec3<f32>(_e1708, _e1710, _e1712)));
            let _e1717 = (_e1716 * _e1708);
            let _e1718 = (_e1716 * _e1710);
            let _e1719 = (_e1716 * _e1712);
            let _e1722 = dot(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e1701, _e1702, _e1703));
            let _e1726 = select(_e1722, 0.000001f, (_e1722 < 0.000001f));
            let _e1729 = dot(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e1717, _e1718, _e1719));
            let _e1732 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(_e1717, _e1718, _e1719));
            let _e1736 = select(_e1732, 0.000001f, (_e1732 < 0.000001f));
            if (_e1726 > 0.000001f) {
                let _e1742 = select(_e1729, 0.000001f, (_e1729 < 0.000001f));
                let _e1743 = mix(_e713, _e477, _e463);
                let _e1744 = mix(_e713, _e478, _e463);
                let _e1745 = mix(_e713, _e479, _e463);
                let _e1748 = pow(0.85714287f, 5f);
                let _e1750 = (1f - _e1743);
                let _e1752 = (_e1743 + (_e1750 * _e1748));
                let _e1754 = (1f - _e1744);
                let _e1756 = (_e1744 + (_e1754 * _e1748));
                let _e1758 = (1f - _e1745);
                let _e1760 = (_e1745 + (_e1758 * _e1748));
                let _e1765 = (1f - _e1736);
                let _e1774 = ((_e1736 * pow(_e1765, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e1776 = pow(_e1765, 5f);
                let _e1798 = mix(_e713, _e477, _e463);
                let _e1799 = mix(_e713, _e478, _e463);
                let _e1800 = mix(_e713, _e479, _e463);
                let _e1803 = pow(0.85714287f, 5f);
                let _e1805 = (1f - _e1798);
                let _e1807 = (_e1798 + (_e1805 * _e1803));
                let _e1809 = (1f - _e1799);
                let _e1811 = (_e1799 + (_e1809 * _e1803));
                let _e1813 = (1f - _e1800);
                let _e1815 = (_e1800 + (_e1813 * _e1803));
                let _e1827 = ((_e1736 * pow(_e1765, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e1829 = pow(_e1765, 5f);
                let _e1852 = (_e717 * sqrt(_e724));
                let _e1855 = (_e719 * (_e717 * sqrt(_e724)));
                let _e1858 = dot(vec3<f32>(_e1717, _e1718, _e1719), vec3<f32>(_e54, _e55, _e56));
                let _e1861 = dot(vec3<f32>(_e1717, _e1718, _e1719), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                let _e1863 = (_e1852 * _e1852);
                let _e1866 = (_e1855 * _e1855);
                let _e1869 = (_e1742 * _e1742);
                let _e1870 = (_e1869 + (((_e1858 * _e1858) / _e1863) + ((_e1861 * _e1861) / _e1866)));
                let _e1877 = (1f / (_e1870 * (((_e1852 * 3.1415927f) * _e1855) * _e1870)));
                let _e1880 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(_e54, _e55, _e56));
                let _e1883 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                let _e1886 = dot(vec3<f32>(_e1701, _e1702, _e1703), vec3<f32>(_e54, _e55, _e56));
                let _e1889 = dot(vec3<f32>(_e1701, _e1702, _e1703), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                let _e1895 = (_e1726 * _e1726);
                let _e1915 = (0.5f / max(((_e46 * sqrt(max((_e1895 + ((_e1863 * (_e1886 * _e1886)) + (_e1866 * (_e1889 * _e1889)))), 0.000001f))) + (_e1726 * sqrt(max((_e393 + ((_e1863 * (_e1880 * _e1880)) + (_e1866 * (_e1883 * _e1883)))), 0.000001f)))), 0.000001f));
                let _e1934 = (1f / length(vec3<f32>(_e730, _e731, _e732)));
                let _e1941 = (_e1701 - (_e15 * _e1726));
                let _e1942 = (_e1702 - (_e16 * _e1726));
                let _e1943 = (_e1703 - (_e17 * _e1726));
                let _e1947 = (1f / length(vec3<f32>(_e1941, _e1942, _e1943)));
                let _e1953 = dot(vec3<f32>((_e730 * _e1934), (_e731 * _e1934), (_e732 * _e1934)), vec3<f32>((_e1941 * _e1947), (_e1942 * _e1947), (_e1943 * _e1947)));
                let _e1958 = acos(max(_e46, 0f));
                let _e1961 = acos(max(_e1726, 0f));
                let _e1985 = (max((1f - dot(vec3<f32>((max(((_e1743 + (_e1750 * _e1776)) - ((_e1752 - (_e1752 * _e714)) * _e1774)), 0f) * _e465), (max(((_e1744 + (_e1754 * _e1776)) - ((_e1756 - (_e1756 * _e715)) * _e1774)), 0f) * _e465), (max(((_e1745 + (_e1758 * _e1776)) - ((_e1760 - (_e1760 * _e716)) * _e1774)), 0f) * _e465)), vec3<f32>(0.2126f, 0.7152f, 0.0722f))), 0f) * (((1f - ((_e741 * 0.5f) / (_e741 + 0.57f))) + (tan(min(_e1958, _e1961)) * ((((_e741 * 0.45f) / (_e741 + 0.09f)) * select(0f, _e1953, (_e1953 > 0f))) * sin(max(_e1958, _e1961))))) * 0.31830987f));
                if _e485 {
                    let _e2006 = (_e761 * sqrt(_e768));
                    let _e2009 = (_e763 * (_e761 * sqrt(_e768)));
                    let _e2012 = dot(vec3<f32>(_e1717, _e1718, _e1719), vec3<f32>(_e54, _e55, _e56));
                    let _e2015 = dot(vec3<f32>(_e1717, _e1718, _e1719), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                    let _e2017 = (_e2006 * _e2006);
                    let _e2020 = (_e2009 * _e2009);
                    let _e2023 = (_e1869 + (((_e2012 * _e2012) / _e2017) + ((_e2015 * _e2015) / _e2020)));
                    let _e2033 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(_e54, _e55, _e56));
                    let _e2036 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                    let _e2039 = dot(vec3<f32>(_e1701, _e1702, _e1703), vec3<f32>(_e54, _e55, _e56));
                    let _e2042 = dot(vec3<f32>(_e1701, _e1702, _e1703), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                    let _e2069 = ((_e758 + ((1f - _e758) * pow(_e1765, 5f))) * ((1f / (_e2023 * (((_e2006 * 3.1415927f) * _e2009) * _e2023))) * (0.5f / max(((_e46 * sqrt(max((_e1895 + ((_e2017 * (_e2039 * _e2039)) + (_e2020 * (_e2042 * _e2042)))), 0.000001f))) + (_e1726 * sqrt(max((_e393 + ((_e2017 * (_e2033 * _e2033)) + (_e2020 * (_e2036 * _e2036)))), 0.000001f)))), 0.000001f))));
                    coat_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e2069 * bitcast<f32>(_e322)) * _e461);
                    coat_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e2069 * bitcast<f32>(_e327)) * _e461);
                    coat_u002e_i_u002e_sroa_u002e_8_u002e_0_ = ((_e2069 * bitcast<f32>(_e332)) * _e461);
                } else {
                    coat_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    coat_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    coat_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2086 = coat_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2088 = coat_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2090 = coat_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e772 <= 0f)) {
                    let _e2114 = ((_e1736 * ((1f / ((_e777 * (((1f / _e46) + (1f / _e1726)) + -2f)) + 1f)) * (_e777 / (pow((((_e777 + -1f) * _e1869) + 1f), 2f) * 3.1415927f)))) / max(((_e46 * 4f) * _e1726), 0.000001f));
                    fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e2114 * bitcast<f32>(_e287)) * _e772);
                    fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e2114 * bitcast<f32>(_e292)) * _e772);
                    fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_ = ((_e2114 * bitcast<f32>(_e297)) * _e772);
                } else {
                    fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2131 = fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2133 = fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2135 = fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e787 <= 0f)) {
                    let _e2137 = (_e717 * sqrt(_e724));
                    if !((_e788 <= 0f)) {
                        extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (-(log(max(bitcast<f32>(_e197), 0.000001f))) / _e788);
                        extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (-(log(max(bitcast<f32>(_e202), 0.000001f))) / _e788);
                        extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (-(log(max(bitcast<f32>(_e207), 0.000001f))) / _e788);
                    } else {
                        extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e2157 = extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                    let _e2159 = extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                    let _e2161 = extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                    let _e2168 = (_e2137 * _e2137);
                    let _e2173 = ((_e1869 * (_e2168 + -1f)) + 1f);
                    let _e2179 = (1f - _e2168);
                    let _e2198 = ((1f - (_e713 + ((1f - _e713) * pow((1f - _e46), 5f)))) * ((_e2168 / (_e2173 * (_e2173 * 3.1415927f))) * (0.5f / max(((_e46 * sqrt(max((_e2168 + (_e1895 * _e2179)), 0.000001f))) + (_e1726 * sqrt(max((_e2168 + (_e393 * _e2179)), 0.000001f)))), 0.000001f))));
                    transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (((exp((_e788 * -(_e2157))) * _e2198) * bitcast<f32>(_e197)) * _e787);
                    transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (((exp((_e788 * -(_e2159))) * _e2198) * bitcast<f32>(_e202)) * _e787);
                    transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (((exp((_e788 * -(_e2161))) * _e2198) * bitcast<f32>(_e207)) * _e787);
                } else {
                    transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2227 = transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2229 = transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2231 = transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e466 <= 0f)) {
                    let _e2236 = length(vec3<f32>((_e730 - _e1941), (_e731 - _e1942), (_e732 - _e1943)));
                    let _e2238 = sqrt(3f);
                    let _e2261 = -(_e2236);
                    let _e2269 = (max(((clamp(((_e46 * _e1726) + (((_e397 + max((1f - _e1895), 0f)) - (_e2236 * _e2236)) * 0.5f)), -1f, 1f) * bitcast<f32>(_e247)) + 1f), 0f) * 0.31830987f);
                    subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (((exp((((1f / max((_e818 * bitcast<f32>(_e242)), 0.000001f)) * _e2238) * _e2261)) * _e2269) * _e480) * _e466);
                    subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (((exp((((1f / max((_e818 * bitcast<f32>(_e267)), 0.000001f)) * _e2238) * _e2261)) * _e2269) * _e481) * _e466);
                    subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (((exp((((1f / max((_e818 * bitcast<f32>(_e272)), 0.000001f)) * _e2238) * _e2261)) * _e2269) * _e482) * _e466);
                } else {
                    subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2289 = subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2291 = subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2293 = subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                let _e2306 = (_e1639 * _e1726);
                lo_u002e_sroa_u002e_0_u002e_1_ = (_e849 + (_e2306 * ((((_e560 * ((_e450 * ((((max(((_e1798 + (_e1805 * _e1829)) - ((_e1807 - (_e1807 * _e714)) * _e1827)), 0f) * _e465) * _e1877) * _e1915) * _e464)) + (_e450 * (((_e735 * _e474) * _e1985) * _e464)))) + _e2086) + (_e2131 + _e2227)) + _e2289)));
                lo_u002e_sroa_u002e_6_u002e_1_ = (_e847 + (_e2306 * ((((_e562 * ((_e455 * ((((max(((_e1799 + (_e1809 * _e1829)) - ((_e1811 - (_e1811 * _e715)) * _e1827)), 0f) * _e465) * _e1877) * _e1915) * _e464)) + (_e455 * (((_e735 * _e475) * _e1985) * _e464)))) + _e2088) + (_e2133 + _e2229)) + _e2291)));
                lo_u002e_sroa_u002e_9_u002e_1_ = (_e845 + (_e2306 * ((((_e564 * ((_e460 * ((((max(((_e1800 + (_e1813 * _e1829)) - ((_e1815 - (_e1815 * _e716)) * _e1827)), 0f) * _e465) * _e1877) * _e1915) * _e464)) + (_e460 * (((_e735 * _e476) * _e1985) * _e464)))) + _e2090) + (_e2135 + _e2231)) + _e2293)));
            } else {
                lo_u002e_sroa_u002e_0_u002e_1_ = _e849;
                lo_u002e_sroa_u002e_6_u002e_1_ = _e847;
                lo_u002e_sroa_u002e_9_u002e_1_ = _e845;
            }
            let _e2320 = lo_u002e_sroa_u002e_0_u002e_1_;
            out_u002e_lo_u002e_sroa_u002e_0_u002e_1_ = _e2320;
            let _e2323 = lo_u002e_sroa_u002e_6_u002e_1_;
            out_u002e_lo_u002e_sroa_u002e_6_u002e_1_ = _e2323;
            let _e2326 = lo_u002e_sroa_u002e_9_u002e_1_;
            out_u002e_lo_u002e_sroa_u002e_9_u002e_1_ = _e2326;
            let _e2329 = (_e843 + 1u);
            if (_e2329 == _e691) {
                lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e2320;
                lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e2323;
                lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e2326;
                break;
            } else {
                k_u002e_sroa_u002e_0_u002e_0_u002e_i1522_ = _e2329;
                lo_u002e_sroa_u002e_9_u002e_01521_ = _e2326;
                lo_u002e_sroa_u002e_6_u002e_01520_ = _e2323;
                lo_u002e_sroa_u002e_0_u002e_01519_ = _e2320;
                continue;
            }
        }
    }
    let _e2345 = lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
    let _e2347 = lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
    let _e2349 = lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
    let _e2359 = uniform_struct_0_3_.f0_.x;
    let _e2363 = uniform_struct_0_3_.f0_.y;
    let _e2367 = uniform_struct_0_3_.f0_.z;
    let _e2371 = bitcast<f32>(_e357);
    if !((_e2371 <= 0f)) {
        let _e2378 = (_e2371 * bitcast<f32>(_e362));
        let _e2379 = (_e2371 * bitcast<f32>(_e367));
        let _e2380 = (_e2371 * bitcast<f32>(_e372));
        let _e2390 = ((1f - _e461) + ((1f - pow((1f - _e46), 5f)) * _e461));
        emission_u002e_sroa_u002e_0_u002e_1_ = (_e2378 + ((((_e2378 * bitcast<f32>(_e322)) * _e2390) - _e2378) * _e461));
        emission_u002e_sroa_u002e_5_u002e_1_ = (_e2379 + ((((_e2379 * bitcast<f32>(_e327)) * _e2390) - _e2379) * _e461));
        emission_u002e_sroa_u002e_8_u002e_1_ = (_e2380 + ((((_e2380 * bitcast<f32>(_e332)) * _e2390) - _e2380) * _e461));
    } else {
        emission_u002e_sroa_u002e_0_u002e_1_ = 0f;
        emission_u002e_sroa_u002e_5_u002e_1_ = 0f;
        emission_u002e_sroa_u002e_8_u002e_1_ = 0f;
    }
    let _e2419 = emission_u002e_sroa_u002e_0_u002e_1_;
    let _e2421 = emission_u002e_sroa_u002e_5_u002e_1_;
    let _e2423 = emission_u002e_sroa_u002e_8_u002e_1_;
    let _e2424 = bitcast<f32>(_e152);
    let _e2429 = ((_e2424 + -1f) / (_e2424 + 1f));
    let _e2430 = (_e2429 * _e2429);
    let _e2438 = (dot(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e37, _e38, _e39)) * 2f);
    let _e2442 = ((_e15 * _e2438) - _e37);
    let _e2443 = ((_e16 * _e2438) - _e38);
    let _e2444 = ((_e17 * _e2438) - _e39);
    let _e2447 = uniform_struct_0_3_.f4_;
    let _e2448 = (_e147 * _e2447);
    let _e2452 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2442, _e2443, _e2444), _e2448);
    let _e2457 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2442, _e2443, _e2444), _e2448);
    let _e2462 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2442, _e2443, _e2444), _e2448);
    let _e2467 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e15, _e16, _e17));
    let _e2472 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e15, _e16, _e17));
    let _e2477 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e15, _e16, _e17));
    let _e2482 = textureSample(tex_0_9_, sampler_0_10_, vec2<f32>(_e46, _e147));
    let _e2487 = textureSample(tex_0_9_, sampler_0_10_, vec2<f32>(_e46, _e147));
    let _e2490 = (1f - _e463);
    let _e2496 = uniform_struct_0_3_.f3_;
    let _e2509 = max((1f - _e2487.y), 0f);
    let _e2534 = !((bitcast<f32>(_e387) >= 0.5f));
    let _e2535 = select(_e2419, (((_e2345 + (mix(_e474, (_e474 * _e465), _e463) * _e2359)) + _e2419) + (_e2496 * ((_e2452.x * ((mix(_e2430, _e477, _e463) * _e2482.x) + _e2487.y)) + (((_e2490 * _e474) * _e2467.x) * _e2509)))), _e2534);
    let _e2536 = select(_e2421, (((_e2347 + (mix(_e475, (_e475 * _e465), _e463) * _e2363)) + _e2421) + (_e2496 * ((_e2457.y * ((mix(_e2430, _e478, _e463) * _e2482.x) + _e2487.y)) + (((_e2490 * _e475) * _e2472.y) * _e2509)))), _e2534);
    let _e2537 = select(_e2423, (((_e2349 + (mix(_e476, (_e476 * _e465), _e463) * _e2367)) + _e2423) + (_e2496 * ((_e2462.z * ((mix(_e2430, _e479, _e463) * _e2482.x) + _e2487.y)) + (((_e2490 * _e476) * _e2477.z) * _e2509)))), _e2534);
    if ((_e2535 <= _e2535) && (0f <= 0f)) {
        if (abs(_e2535) < 100000000000000000000f) {
            _2_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2535;
        } else {
            _2_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        _2_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2551 = _2_u002e_sroa_u002e_0_u002e_0_u002e_i;
    if ((_e2536 <= _e2536) && (0f <= 0f)) {
        if (abs(_e2536) < 100000000000000000000f) {
            _3_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2536;
        } else {
            _3_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        _3_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2565 = _3_u002e_sroa_u002e_0_u002e_0_u002e_i;
    if ((_e2537 <= _e2537) && (0f <= 0f)) {
        if (abs(_e2537) < 100000000000000000000f) {
            _4_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2537;
            let _e2575 = _4_u002e_sroa_u002e_0_u002e_0_u002e_i;
            fragment_out = vec4<f32>(_e2551, _e2565, _e2575, bitcast<f32>(_e377));
            let _e2580 = fragment_out;
            return _e2580;
        }
        _4_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    } else {
        _4_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2586 = _4_u002e_sroa_u002e_0_u002e_0_u002e_i;
    fragment_out = vec4<f32>(_e2551, _e2565, _e2586, bitcast<f32>(_e377));
    let _e2591 = fragment_out;
    return _e2591;
}
