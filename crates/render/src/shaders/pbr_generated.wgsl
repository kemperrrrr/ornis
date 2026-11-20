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
fn fs_main(@builtin(position) position: vec4<f32>, @location(0) in0_: vec3<f32>, @location(1) in1_: vec3<f32>, @location(3) in3_: vec3<f32>, @location(4) @interpolate(flat) in4_: u32) -> @location(0) vec4<f32> {
    var fragment_out: vec4<f32>;
    var radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_: f32;
    var vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i: f32;
    var lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa: f32;
    var lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa: f32;
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i: u32;
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
    let _e74 = bitcast<f32>(_e73);
    let _e79 = storage_0_2_[(_e70 | 1u)];
    let _e80 = bitcast<f32>(_e79);
    let _e85 = storage_0_2_[(_e70 | 2u)];
    let _e86 = bitcast<f32>(_e85);
    let _e91 = storage_0_2_[(_e70 | 4u)];
    let _e92 = bitcast<f32>(_e91);
    let _e97 = storage_0_2_[(_e70 | 5u)];
    let _e98 = bitcast<f32>(_e97);
    let _e103 = storage_0_2_[(_e70 | 6u)];
    let _e104 = bitcast<f32>(_e103);
    let _e109 = storage_0_2_[(_e70 | 8u)];
    let _e110 = bitcast<f32>(_e109);
    let _e112 = (_e70 | 9u);
    let _e115 = storage_0_2_[_e112];
    let _e116 = bitcast<f32>(_e115);
    let _e117 = dpdx(_e15);
    let _e118 = dpdx(_e16);
    let _e119 = dpdx(_e17);
    let _e120 = dpdy(_e15);
    let _e121 = dpdy(_e16);
    let _e122 = dpdy(_e17);
    let _e133 = (((dot(vec3<f32>(_e117, _e118, _e119), vec3<f32>(_e117, _e118, _e119)) + dot(vec3<f32>(_e120, _e121, _e122), vec3<f32>(_e120, _e121, _e122))) * 0.15f) * 2f);
    let _e138 = (_e116 * _e116);
    let _e140 = ((_e138 * _e138) + select(0.2f, _e133, (_e133 < 0.2f)));
    let _e144 = select(_e140, 0f, (_e140 < 0f));
    let _e148 = select(_e144, 1f, (_e144 > 1f));
    let _e154 = sqrt(sqrt(select(0f, _e148, (_e148 > 0f))));
    let _e162 = storage_0_2_[(_e70 | 10u)];
    let _e163 = bitcast<f32>(_e162);
    let _e168 = storage_0_2_[(_e70 | 11u)];
    let _e174 = storage_0_2_[(_e70 | 12u)];
    let _e175 = bitcast<f32>(_e174);
    let _e180 = storage_0_2_[(_e70 | 13u)];
    let _e181 = bitcast<f32>(_e180);
    let _e186 = storage_0_2_[(_e70 | 14u)];
    let _e187 = bitcast<f32>(_e186);
    let _e192 = storage_0_2_[(_e70 + 16u)];
    let _e193 = bitcast<f32>(_e192);
    let _e198 = storage_0_2_[(_e70 + 17u)];
    let _e199 = bitcast<f32>(_e198);
    let _e214 = storage_0_2_[(_e70 + 20u)];
    let _e215 = bitcast<f32>(_e214);
    let _e220 = storage_0_2_[(_e70 + 21u)];
    let _e221 = bitcast<f32>(_e220);
    let _e226 = storage_0_2_[(_e70 + 22u)];
    let _e227 = bitcast<f32>(_e226);
    let _e252 = storage_0_2_[(_e70 + 28u)];
    let _e253 = bitcast<f32>(_e252);
    let _e258 = storage_0_2_[(_e70 + 29u)];
    let _e259 = bitcast<f32>(_e258);
    let _e264 = storage_0_2_[(_e70 + 30u)];
    let _e270 = storage_0_2_[(_e70 + 31u)];
    let _e276 = storage_0_2_[(_e70 + 32u)];
    let _e277 = bitcast<f32>(_e276);
    let _e282 = storage_0_2_[(_e70 + 33u)];
    let _e283 = bitcast<f32>(_e282);
    let _e288 = storage_0_2_[(_e70 + 34u)];
    let _e289 = bitcast<f32>(_e288);
    let _e294 = storage_0_2_[(_e70 + 36u)];
    let _e300 = storage_0_2_[(_e70 + 37u)];
    let _e306 = storage_0_2_[(_e70 + 40u)];
    let _e307 = bitcast<f32>(_e306);
    let _e312 = storage_0_2_[(_e70 + 41u)];
    let _e313 = bitcast<f32>(_e312);
    let _e318 = storage_0_2_[(_e70 + 44u)];
    let _e324 = storage_0_2_[(_e70 + 45u)];
    let _e330 = storage_0_2_[(_e70 + 46u)];
    let _e336 = storage_0_2_[(_e70 + 48u)];
    let _e337 = bitcast<f32>(_e336);
    let _e342 = storage_0_2_[(_e70 + 49u)];
    let _e343 = bitcast<f32>(_e342);
    let _e348 = storage_0_2_[(_e70 + 50u)];
    let _e354 = storage_0_2_[(_e70 + 51u)];
    let _e359 = storage_0_2_[(_e70 + 52u)];
    let _e360 = bitcast<f32>(_e359);
    let _e365 = storage_0_2_[(_e70 + 53u)];
    let _e366 = bitcast<f32>(_e365);
    let _e371 = storage_0_2_[(_e70 + 54u)];
    let _e372 = bitcast<f32>(_e371);
    let _e377 = storage_0_2_[(_e70 + 56u)];
    let _e378 = bitcast<f32>(_e377);
    let _e383 = storage_0_2_[(_e70 + 60u)];
    let _e389 = storage_0_2_[(_e70 + 61u)];
    let _e395 = storage_0_2_[(_e70 + 62u)];
    let _e396 = bitcast<f32>(_e395);
    let _e401 = storage_0_2_[(_e70 + 64u)];
    let _e402 = bitcast<f32>(_e401);
    let _e407 = storage_0_2_[(_e70 + 68u)];
    let _e413 = storage_0_2_[(_e70 + 69u)];
    let _e419 = storage_0_2_[(_e70 + 70u)];
    let _e425 = storage_0_2_[(_e70 + 72u)];
    let _e435 = storage_0_2_[(_e70 + 74u)];
    let _e440 = uniform_struct_0_0_.f0_.x;
    let _e444 = uniform_struct_0_0_.f1_.x;
    let _e448 = uniform_struct_0_0_.f2_.x;
    let _e452 = uniform_struct_0_0_.f3_.x;
    let _e462 = uniform_struct_0_0_.f0_.y;
    let _e466 = uniform_struct_0_0_.f1_.y;
    let _e470 = uniform_struct_0_0_.f2_.y;
    let _e474 = uniform_struct_0_0_.f3_.y;
    let _e484 = uniform_struct_0_0_.f0_.w;
    let _e488 = uniform_struct_0_0_.f1_.w;
    let _e492 = uniform_struct_0_0_.f2_.w;
    let _e496 = uniform_struct_0_0_.f3_.w;
    let _e502 = (_e496 + (((in0_.x * _e484) + (in0_.y * _e488)) + (in0_.z * _e492)));
    let _e506 = select(0.5f, _e502, (_e502 > 0.5f));
    let _e547 = ((((min(u32(floor((((((_e474 + (((in0_.x * _e462) + (in0_.y * _e466)) + (in0_.z * _e470))) / _e506) * 0.5f) + 0.5f) * 9f))), 8u) << 4u) | min(u32(floor((((((_e452 + (((in0_.x * _e440) + (in0_.y * _e444)) + (in0_.z * _e448))) / _e506) * 0.5f) + 0.5f) * 16f))), 15u)) + (min(u32(ceil(((log((_e506 * 2f)) * 3.4407384f) + -0.0001f))), 23u) * 144u)) << 1u);
    let _e550 = uniform_struct_0_12_.f2_;
    let _e553 = uniform_struct_0_3_.f2_;
    let _e556 = storage_0_15_[_e547];
    let _e562 = storage_0_15_[(_e547 | 1u)];
    let _e566 = (_e46 * _e46);
    let _e570 = max((1f - _e566), 0f);
    let _e572 = (sqrt(_e570) / _e396);
    let _e585 = pow(((_e396 + -1f) / (_e396 + 1f)), 2f);
    let _e589 = (((bitcast<f32>(_e389) * 1000f) * (_e396 * 12.566371f)) * sqrt(max((1f - (_e572 * _e572)), 0f)));
    let _e600 = (_e585 * 2f);
    let _e606 = (1f - (_e585 * _e585));
    let _e618 = clamp(bitcast<f32>(_e383), 0f, 1f);
    let _e623 = ((_e618 * ((((_e600 * cos((_e589 / 650f))) / _e606) + 1f) + -1f)) + 1f);
    let _e628 = ((_e618 * ((((_e600 * cos((_e589 / 550f))) / _e606) + 1f) + -1f)) + 1f);
    let _e633 = ((_e618 * ((((_e600 * cos((_e589 / 450f))) / _e606) + 1f) + -1f)) + 1f);
    let _e638 = ((_e378 + -1f) / (_e378 + 1f));
    let _e641 = (_e74 * _e92);
    let _e642 = (_e74 * _e98);
    let _e643 = (_e74 * _e104);
    let _e646 = !((_e337 <= 0f));
    if _e646 {
        let _e652 = (1f - ((1f - pow(_e638, 2f)) / (_e378 * _e378)));
        let _e654 = (1f - _e652);
        let _e655 = (_e643 * _e110);
        let _e668 = (_e642 * _e110);
        let _e681 = (_e641 * _e110);
        let _e695 = (_e337 * bitcast<f32>(_e354));
        coat_darken_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e695 * ((_e654 / max((1f - ((_e681 + (((((_e92 - _e277) * _e253) + _e277) - _e681) * _e86)) * _e652)), 0.000001f)) + -1f)) + 1f);
        coat_darken_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e695 * ((_e654 / max((1f - ((_e668 + (((((_e98 - _e283) * _e253) + _e283) - _e668) * _e86)) * _e652)), 0.000001f)) + -1f)) + 1f);
        coat_darken_u002e_i_u002e_sroa_u002e_7_u002e_0_ = ((_e695 * ((_e654 / max((1f - ((_e655 + (((((_e104 - _e289) * _e253) + _e289) - _e655) * _e86)) * _e652)), 0.000001f)) + -1f)) + 1f);
    } else {
        coat_darken_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 1f;
        coat_darken_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 1f;
        coat_darken_u002e_i_u002e_sroa_u002e_7_u002e_0_ = 1f;
    }
    let _e721 = coat_darken_u002e_i_u002e_sroa_u002e_0_u002e_0_;
    let _e723 = coat_darken_u002e_i_u002e_sroa_u002e_5_u002e_0_;
    let _e725 = coat_darken_u002e_i_u002e_sroa_u002e_7_u002e_0_;
    let _e726 = ((_e562 * _e550) + _e553);
    if (_e726 == 0u) {
        lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
        lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = 0f;
        lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = 0f;
    } else {
        let _e743 = ((_e163 + -1f) / (_e163 + 1f));
        let _e744 = (_e743 * _e743);
        let _e745 = (_e154 * _e154);
        let _e747 = (1f - bitcast<f32>(_e168));
        let _e752 = (2f / ((_e747 * _e747) + 1f));
        let _e754 = select(_e154, _e80, (_e154 < _e80));
        let _e758 = (_e37 - (_e15 * _e46));
        let _e759 = (_e38 - (_e16 * _e46));
        let _e760 = (_e39 - (_e17 * _e46));
        let _e763 = (1f - _e86);
        let _e768 = max((_e754 * _e754), 0.000001f);
        let _e769 = (_e768 * _e768);
        let _e784 = (_e638 * _e638);
        let _e787 = (_e343 * _e343);
        let _e789 = (1f - bitcast<f32>(_e348));
        let _e794 = (2f / ((_e789 * _e789) + 1f));
        let _e798 = (_e313 * _e313);
        k_u002e_sroa_u002e_0_u002e_0_ = 0u;
        lo_u002e_i_u002e_sroa_u002e_9_ = 0f;
        lo_u002e_i_u002e_sroa_u002e_6_ = 0f;
        lo_u002e_i_u002e_sroa_u002e_0_ = 0f;
        loop {
            let _e848 = k_u002e_sroa_u002e_0_u002e_0_;
            let _e850 = lo_u002e_i_u002e_sroa_u002e_9_;
            let _e852 = lo_u002e_i_u002e_sroa_u002e_6_;
            let _e854 = lo_u002e_i_u002e_sroa_u002e_0_;
            if (_e848 < _e553) {
                _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i = _e848;
            } else {
                let _e860 = storage_0_14_[(((_e556 * _e550) - _e553) + _e848)];
                _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i = _e860;
            }
            let _e863 = _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i;
            let _e865 = (_e863 * 36u);
            let _e868 = storage_0_13_[_e865];
            let _e869 = bitcast<f32>(_e868);
            let _e871 = (_e865 + 8u);
            let _e874 = storage_0_13_[_e871];
            let _e877 = (_e865 + 9u);
            let _e880 = storage_0_13_[_e877];
            let _e883 = (_e865 + 10u);
            let _e886 = storage_0_13_[_e883];
            let _e892 = length(vec3<f32>((bitcast<f32>(_e874) - in0_.x), (bitcast<f32>(_e880) - in0_.y), (bitcast<f32>(_e886) - in0_.z)));
            let _e897 = select(1f, 0f, (_e869 < 0.5f));
            let _e899 = (_e865 + 16u);
            let _e902 = storage_0_13_[_e899];
            let _e903 = bitcast<f32>(_e902);
            let _e915 = storage_0_13_[(_e865 + 12u)];
            let _e921 = storage_0_13_[(_e865 + 13u)];
            let _e927 = storage_0_13_[(_e865 + 14u)];
            let _e933 = storage_0_13_[(_e865 + 15u)];
            let _e934 = bitcast<f32>(_e933);
            let _e937 = storage_0_13_[_e871];
            let _e941 = storage_0_13_[_e877];
            let _e945 = storage_0_13_[_e883];
            let _e947 = (in0_.x - bitcast<f32>(_e937));
            let _e948 = (in0_.y - bitcast<f32>(_e941));
            let _e949 = (in0_.z - bitcast<f32>(_e945));
            let _e953 = (1f / length(vec3<f32>(_e947, _e948, _e949)));
            let _e958 = (_e865 + 4u);
            let _e961 = storage_0_13_[_e958];
            let _e962 = bitcast<f32>(_e961);
            let _e964 = (_e865 + 5u);
            let _e967 = storage_0_13_[_e964];
            let _e968 = bitcast<f32>(_e967);
            let _e970 = (_e865 + 6u);
            let _e973 = storage_0_13_[_e970];
            let _e974 = bitcast<f32>(_e973);
            let _e978 = (1f / length(vec3<f32>(_e962, _e968, _e974)));
            let _e989 = storage_0_13_[(_e865 + 18u)];
            let _e995 = storage_0_13_[(_e865 + 17u)];
            let _e1002 = select(1f, 0f, (_e869 < 1.5f));
            let _e1004 = mix(1f, smoothstep(bitcast<f32>(_e989), bitcast<f32>(_e995), dot(vec3<f32>((_e978 * _e962), (_e978 * _e968), (_e978 * _e974)), vec3<f32>((_e947 * _e953), (_e948 * _e953), (_e949 * _e953)))), _e1002);
            let _e1005 = (_e892 * _e892);
            let _e1017 = (mix(1f, (1f - smoothstep((_e903 * 0.99f), _e903, _e892)), _e897) * mix(1f, (1f / select(_e1005, 0.000001f, (_e1005 < 0.000001f))), _e897));
            let _e1021 = (_e1004 * ((bitcast<f32>(_e915) * _e934) * _e1017));
            let _e1022 = (_e1004 * ((bitcast<f32>(_e921) * _e934) * _e1017));
            let _e1023 = (_e1004 * ((bitcast<f32>(_e927) * _e934) * _e1017));
            let _e1026 = (_e897 * (1f - _e1002));
            if (((1f - _e897) * select(1f, 0f, (_e903 < 0.5f))) > 0.5f) {
                let _e1041 = uniform_struct_0_3_.f6_.f1_.x;
                let _e1046 = uniform_struct_0_3_.f6_.f1_.y;
                let _e1051 = uniform_struct_0_3_.f6_.f1_.z;
                let _e1054 = dot(vec3<f32>((in0_.x - _e21), (in0_.y - _e25), (in0_.z - _e29)), vec3<f32>(_e1041, _e1046, _e1051));
                let _e1059 = uniform_struct_0_3_.f6_.f2_.x;
                let _e1064 = uniform_struct_0_3_.f6_.f2_.y;
                let _e1069 = uniform_struct_0_3_.f6_.f2_.z;
                let _e1074 = uniform_struct_0_3_.f6_.f2_.w;
                let _e1078 = uniform_struct_0_3_.f6_.f3_;
                let _e1082 = select(1f, 0f, (_e1054 < _e1064));
                let _e1086 = select(1f, 0f, (_e1054 < _e1069));
                let _e1090 = select(1f, 0f, (_e1054 < _e1074));
                let _e1093 = u32(((_e1082 + _e1086) + _e1090));
                let _e1096 = mix(mix(mix(_e1064, _e1069, _e1082), _e1074, _e1086), _e1078, _e1090);
                let _e1106 = uniform_struct_0_3_.f6_.f0_[_e1093].f0_.x;
                let _e1113 = uniform_struct_0_3_.f6_.f0_[_e1093].f1_.x;
                let _e1120 = uniform_struct_0_3_.f6_.f0_[_e1093].f2_.x;
                let _e1127 = uniform_struct_0_3_.f6_.f0_[_e1093].f3_.x;
                let _e1140 = uniform_struct_0_3_.f6_.f0_[_e1093].f0_.y;
                let _e1147 = uniform_struct_0_3_.f6_.f0_[_e1093].f1_.y;
                let _e1154 = uniform_struct_0_3_.f6_.f0_[_e1093].f2_.y;
                let _e1161 = uniform_struct_0_3_.f6_.f0_[_e1093].f3_.y;
                let _e1174 = uniform_struct_0_3_.f6_.f0_[_e1093].f0_.z;
                let _e1181 = uniform_struct_0_3_.f6_.f0_[_e1093].f1_.z;
                let _e1188 = uniform_struct_0_3_.f6_.f0_[_e1093].f2_.z;
                let _e1195 = uniform_struct_0_3_.f6_.f0_[_e1093].f3_.z;
                let _e1208 = uniform_struct_0_3_.f6_.f0_[_e1093].f0_.w;
                let _e1215 = uniform_struct_0_3_.f6_.f0_[_e1093].f1_.w;
                let _e1222 = uniform_struct_0_3_.f6_.f0_[_e1093].f2_.w;
                let _e1229 = uniform_struct_0_3_.f6_.f0_[_e1093].f3_.w;
                let _e1235 = (_e1229 + (((in0_.x * _e1208) + (in0_.y * _e1215)) + (in0_.z * _e1222)));
                let _e1252 = textureSampleCompareLevel(tex_0_11_, sampler_0_5_, vec2<f32>(((((_e1127 + (((in0_.x * _e1106) + (in0_.y * _e1113)) + (in0_.z * _e1120))) / _e1235) * 0.5f) + 0.5f), (0.5f - (((_e1161 + (((in0_.x * _e1140) + (in0_.y * _e1147)) + (in0_.z * _e1154))) / _e1235) * 0.5f))), _e1093, (((_e1195 + (((in0_.x * _e1174) + (in0_.y * _e1181)) + (in0_.z * _e1188))) / _e1235) + -0.001f));
                if (_e1093 < 3u) {
                    let _e1257 = ((_e1096 - mix(mix(mix(_e1059, _e1064, _e1082), _e1069, _e1086), _e1074, _e1090)) * 0.08f);
                    let _e1263 = ((_e1096 - _e1054) / select(_e1257, 0.000001f, (_e1257 < 0.000001f)));
                    let _e1267 = select(_e1263, 0f, (_e1263 < 0f));
                    let _e1273 = (_e1093 + 1u);
                    let _e1280 = uniform_struct_0_3_.f6_.f0_[_e1273].f0_.x;
                    let _e1287 = uniform_struct_0_3_.f6_.f0_[_e1273].f1_.x;
                    let _e1294 = uniform_struct_0_3_.f6_.f0_[_e1273].f2_.x;
                    let _e1301 = uniform_struct_0_3_.f6_.f0_[_e1273].f3_.x;
                    let _e1314 = uniform_struct_0_3_.f6_.f0_[_e1273].f0_.y;
                    let _e1321 = uniform_struct_0_3_.f6_.f0_[_e1273].f1_.y;
                    let _e1328 = uniform_struct_0_3_.f6_.f0_[_e1273].f2_.y;
                    let _e1335 = uniform_struct_0_3_.f6_.f0_[_e1273].f3_.y;
                    let _e1348 = uniform_struct_0_3_.f6_.f0_[_e1273].f0_.z;
                    let _e1355 = uniform_struct_0_3_.f6_.f0_[_e1273].f1_.z;
                    let _e1362 = uniform_struct_0_3_.f6_.f0_[_e1273].f2_.z;
                    let _e1369 = uniform_struct_0_3_.f6_.f0_[_e1273].f3_.z;
                    let _e1382 = uniform_struct_0_3_.f6_.f0_[_e1273].f0_.w;
                    let _e1389 = uniform_struct_0_3_.f6_.f0_[_e1273].f1_.w;
                    let _e1396 = uniform_struct_0_3_.f6_.f0_[_e1273].f2_.w;
                    let _e1403 = uniform_struct_0_3_.f6_.f0_[_e1273].f3_.w;
                    let _e1409 = (_e1403 + (((in0_.x * _e1382) + (in0_.y * _e1389)) + (in0_.z * _e1396)));
                    let _e1426 = textureSampleCompareLevel(tex_0_11_, sampler_0_5_, vec2<f32>(((((_e1301 + (((in0_.x * _e1280) + (in0_.y * _e1287)) + (in0_.z * _e1294))) / _e1409) * 0.5f) + 0.5f), (0.5f - (((_e1335 + (((in0_.x * _e1314) + (in0_.y * _e1321)) + (in0_.z * _e1328))) / _e1409) * 0.5f))), _e1273, (((_e1369 + (((in0_.x * _e1348) + (in0_.y * _e1355)) + (in0_.z * _e1362))) / _e1409) + -0.001f));
                    vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i = mix(_e1426, _e1252, select(_e1267, 1f, (_e1267 > 1f)));
                } else {
                    vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i = _e1252;
                }
                let _e1431 = vis_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_i;
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (_e1021 * _e1431);
                radiance_u002e_i_u002e_sroa_u002e_10_u002e_0_ = (_e1022 * _e1431);
                radiance_u002e_i_u002e_sroa_u002e_15_u002e_0_ = (_e1023 * _e1431);
            } else {
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_ = _e1021;
                radiance_u002e_i_u002e_sroa_u002e_10_u002e_0_ = _e1022;
                radiance_u002e_i_u002e_sroa_u002e_15_u002e_0_ = _e1023;
            }
            let _e1442 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_;
            let _e1444 = radiance_u002e_i_u002e_sroa_u002e_10_u002e_0_;
            let _e1446 = radiance_u002e_i_u002e_sroa_u002e_15_u002e_0_;
            let _e1448 = (_e865 + 19u);
            let _e1451 = storage_0_13_[_e1448];
            if !((bitcast<f32>(_e1451) >= 0f)) {
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = _e1442;
                radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_ = _e1444;
                radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_ = _e1446;
            } else {
                if !((_e1026 <= 0.5f)) {
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e1442;
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_1_ = _e1444;
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_1_ = _e1446;
                } else {
                    let _e1469 = storage_0_13_[(_e865 + 20u)];
                    let _e1475 = storage_0_13_[(_e865 + 24u)];
                    let _e1481 = storage_0_13_[(_e865 + 28u)];
                    let _e1487 = storage_0_13_[(_e865 + 32u)];
                    let _e1499 = storage_0_13_[(_e865 + 21u)];
                    let _e1505 = storage_0_13_[(_e865 + 25u)];
                    let _e1511 = storage_0_13_[(_e865 + 29u)];
                    let _e1517 = storage_0_13_[(_e865 + 33u)];
                    let _e1529 = storage_0_13_[(_e865 + 22u)];
                    let _e1535 = storage_0_13_[(_e865 + 26u)];
                    let _e1541 = storage_0_13_[(_e865 + 30u)];
                    let _e1547 = storage_0_13_[(_e865 + 34u)];
                    let _e1559 = storage_0_13_[(_e865 + 23u)];
                    let _e1565 = storage_0_13_[(_e865 + 27u)];
                    let _e1571 = storage_0_13_[(_e865 + 31u)];
                    let _e1577 = storage_0_13_[(_e865 + 35u)];
                    let _e1584 = ((((in0_.x * bitcast<f32>(_e1559)) + (in0_.y * bitcast<f32>(_e1565))) + (in0_.z * bitcast<f32>(_e1571))) + bitcast<f32>(_e1577));
                    let _e1597 = storage_0_13_[_e1448];
                    let _e1606 = textureSampleCompareLevel(tex_0_4_, sampler_0_5_, vec2<f32>((((((((in0_.x * bitcast<f32>(_e1469)) + (in0_.y * bitcast<f32>(_e1475))) + (in0_.z * bitcast<f32>(_e1481))) + bitcast<f32>(_e1487)) / _e1584) * 0.5f) + 0.5f), (0.5f - ((((((in0_.x * bitcast<f32>(_e1499)) + (in0_.y * bitcast<f32>(_e1505))) + (in0_.z * bitcast<f32>(_e1511))) + bitcast<f32>(_e1517)) / _e1584) * 0.5f))), u32(bitcast<f32>(_e1597)), ((((((in0_.x * bitcast<f32>(_e1529)) + (in0_.y * bitcast<f32>(_e1535))) + (in0_.z * bitcast<f32>(_e1541))) + bitcast<f32>(_e1547)) / _e1584) + -0.001f));
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_ = (_e1442 * _e1606);
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_1_ = (_e1444 * _e1606);
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_1_ = (_e1446 * _e1606);
                }
                let _e1614 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_;
                let _e1616 = radiance_u002e_i_u002e_sroa_u002e_10_u002e_1_;
                let _e1618 = radiance_u002e_i_u002e_sroa_u002e_15_u002e_1_;
                if (_e1026 > 0.5f) {
                    let _e1623 = storage_0_13_[_e871];
                    let _e1627 = storage_0_13_[_e877];
                    let _e1631 = storage_0_13_[_e883];
                    let _e1633 = (in0_.x - bitcast<f32>(_e1623));
                    let _e1634 = (in0_.y - bitcast<f32>(_e1627));
                    let _e1635 = (in0_.z - bitcast<f32>(_e1631));
                    let _e1636 = abs(_e1633);
                    let _e1637 = abs(_e1634);
                    let _e1639 = select(_e1636, _e1637, (_e1637 > _e1636));
                    let _e1640 = abs(_e1635);
                    let _e1645 = storage_0_13_[_e899];
                    let _e1646 = bitcast<f32>(_e1645);
                    let _e1650 = select(_e1646, 1f, (_e1646 < 1f));
                    let _e1652 = (_e1650 + -0.1f);
                    let _e1663 = storage_0_13_[_e1448];
                    let _e1669 = textureSampleCompareLevel(tex_0_6_, sampler_0_5_, vec3<f32>(_e1633, _e1634, _e1635), u32(bitcast<f32>(_e1663)), (((_e1650 / _e1652) - ((_e1650 * 0.1f) / (select(_e1639, _e1640, (_e1640 > _e1639)) * _e1652))) + -0.001f));
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = (_e1614 * _e1669);
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_ = (_e1616 * _e1669);
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_ = (_e1618 * _e1669);
                } else {
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = _e1614;
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_ = _e1616;
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_ = _e1618;
                }
            }
            let _e1680 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_;
            let _e1682 = radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_;
            let _e1684 = radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_;
            let _e1687 = storage_0_13_[_e865];
            let _e1691 = storage_0_13_[_e871];
            let _e1695 = storage_0_13_[_e877];
            let _e1699 = storage_0_13_[_e883];
            let _e1701 = (bitcast<f32>(_e1691) - in0_.x);
            let _e1702 = (bitcast<f32>(_e1695) - in0_.y);
            let _e1703 = (bitcast<f32>(_e1699) - in0_.z);
            let _e1705 = length(vec3<f32>(_e1701, _e1702, _e1703));
            let _e1711 = (1f / select(_e1705, 0.000001f, (_e1705 < 0.000001f)));
            let _e1719 = select(1f, 0f, (bitcast<f32>(_e1687) < 0.5f));
            let _e1722 = storage_0_13_[_e958];
            let _e1723 = bitcast<f32>(_e1722);
            let _e1726 = storage_0_13_[_e964];
            let _e1727 = bitcast<f32>(_e1726);
            let _e1730 = storage_0_13_[_e970];
            let _e1731 = bitcast<f32>(_e1730);
            let _e1735 = (1f / length(vec3<f32>(_e1723, _e1727, _e1731)));
            let _e1739 = mix((_e1735 * _e1723), (_e1701 * _e1711), _e1719);
            let _e1740 = mix((_e1735 * _e1727), (_e1702 * _e1711), _e1719);
            let _e1741 = mix((_e1735 * _e1731), (_e1703 * _e1711), _e1719);
            let _e1745 = (1f / length(vec3<f32>(_e1739, _e1740, _e1741)));
            let _e1746 = (_e1739 * _e1745);
            let _e1747 = (_e1740 * _e1745);
            let _e1748 = (_e1741 * _e1745);
            let _e1753 = ((_e37 + _e1746) + 0f);
            let _e1755 = ((_e38 + _e1747) + 0f);
            let _e1757 = ((_e39 + _e1748) + 0.000001f);
            let _e1761 = (1f / length(vec3<f32>(_e1753, _e1755, _e1757)));
            let _e1762 = (_e1761 * _e1753);
            let _e1763 = (_e1761 * _e1755);
            let _e1764 = (_e1761 * _e1757);
            let _e1767 = dot(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e1746, _e1747, _e1748));
            let _e1771 = select(_e1767, 0.000001f, (_e1767 < 0.000001f));
            let _e1774 = dot(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e1762, _e1763, _e1764));
            let _e1777 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(_e1762, _e1763, _e1764));
            let _e1781 = select(_e1777, 0.000001f, (_e1777 < 0.000001f));
            if (_e1771 > 0.000001f) {
                let _e1787 = select(_e1774, 0.000001f, (_e1774 < 0.000001f));
                let _e1788 = mix(_e744, _e641, _e86);
                let _e1789 = mix(_e744, _e642, _e86);
                let _e1790 = mix(_e744, _e643, _e86);
                let _e1793 = pow(0.85714287f, 5f);
                let _e1795 = (1f - _e1788);
                let _e1797 = (_e1788 + (_e1795 * _e1793));
                let _e1799 = (1f - _e1789);
                let _e1801 = (_e1789 + (_e1799 * _e1793));
                let _e1803 = (1f - _e1790);
                let _e1805 = (_e1790 + (_e1803 * _e1793));
                let _e1810 = (1f - _e1781);
                let _e1819 = ((_e1781 * pow(_e1810, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e1821 = pow(_e1810, 5f);
                let _e1843 = mix(_e744, _e641, _e86);
                let _e1844 = mix(_e744, _e642, _e86);
                let _e1845 = mix(_e744, _e643, _e86);
                let _e1848 = pow(0.85714287f, 5f);
                let _e1850 = (1f - _e1843);
                let _e1852 = (_e1843 + (_e1850 * _e1848));
                let _e1854 = (1f - _e1844);
                let _e1856 = (_e1844 + (_e1854 * _e1848));
                let _e1858 = (1f - _e1845);
                let _e1860 = (_e1845 + (_e1858 * _e1848));
                let _e1872 = ((_e1781 * pow(_e1810, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e1874 = pow(_e1810, 5f);
                let _e1897 = (_e745 * sqrt(_e752));
                let _e1900 = (_e747 * (_e745 * sqrt(_e752)));
                let _e1903 = dot(vec3<f32>(_e1762, _e1763, _e1764), vec3<f32>(_e54, _e55, _e56));
                let _e1906 = dot(vec3<f32>(_e1762, _e1763, _e1764), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                let _e1908 = (_e1897 * _e1897);
                let _e1911 = (_e1900 * _e1900);
                let _e1914 = (_e1787 * _e1787);
                let _e1915 = (_e1914 + (((_e1903 * _e1903) / _e1908) + ((_e1906 * _e1906) / _e1911)));
                let _e1922 = (1f / (_e1915 * (((_e1897 * 3.1415927f) * _e1900) * _e1915)));
                let _e1925 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(_e54, _e55, _e56));
                let _e1928 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                let _e1931 = dot(vec3<f32>(_e1746, _e1747, _e1748), vec3<f32>(_e54, _e55, _e56));
                let _e1934 = dot(vec3<f32>(_e1746, _e1747, _e1748), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                let _e1940 = (_e1771 * _e1771);
                let _e1960 = (0.5f / max(((_e46 * sqrt(max((_e1940 + ((_e1908 * (_e1931 * _e1931)) + (_e1911 * (_e1934 * _e1934)))), 0.000001f))) + (_e1771 * sqrt(max((_e566 + ((_e1908 * (_e1925 * _e1925)) + (_e1911 * (_e1928 * _e1928)))), 0.000001f)))), 0.000001f));
                let _e1979 = (1f / length(vec3<f32>(_e758, _e759, _e760)));
                let _e1986 = (_e1746 - (_e15 * _e1771));
                let _e1987 = (_e1747 - (_e16 * _e1771));
                let _e1988 = (_e1748 - (_e17 * _e1771));
                let _e1992 = (1f / length(vec3<f32>(_e1986, _e1987, _e1988)));
                let _e1998 = dot(vec3<f32>((_e758 * _e1979), (_e759 * _e1979), (_e760 * _e1979)), vec3<f32>((_e1986 * _e1992), (_e1987 * _e1992), (_e1988 * _e1992)));
                let _e2003 = acos(max(_e46, 0f));
                let _e2006 = acos(max(_e1771, 0f));
                let _e2030 = (max((1f - dot(vec3<f32>((max(((_e1788 + (_e1795 * _e1821)) - ((_e1797 - (_e1797 * _e175)) * _e1819)), 0f) * _e110), (max(((_e1789 + (_e1799 * _e1821)) - ((_e1801 - (_e1801 * _e181)) * _e1819)), 0f) * _e110), (max(((_e1790 + (_e1803 * _e1821)) - ((_e1805 - (_e1805 * _e187)) * _e1819)), 0f) * _e110)), vec3<f32>(0.2126f, 0.7152f, 0.0722f))), 0f) * (((1f - ((_e769 * 0.5f) / (_e769 + 0.57f))) + (tan(min(_e2003, _e2006)) * ((((_e769 * 0.45f) / (_e769 + 0.09f)) * select(0f, _e1998, (_e1998 > 0f))) * sin(max(_e2003, _e2006))))) * 0.31830987f));
                if _e646 {
                    let _e2051 = (_e787 * sqrt(_e794));
                    let _e2054 = (_e789 * (_e787 * sqrt(_e794)));
                    let _e2057 = dot(vec3<f32>(_e1762, _e1763, _e1764), vec3<f32>(_e54, _e55, _e56));
                    let _e2060 = dot(vec3<f32>(_e1762, _e1763, _e1764), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                    let _e2062 = (_e2051 * _e2051);
                    let _e2065 = (_e2054 * _e2054);
                    let _e2068 = (_e1914 + (((_e2057 * _e2057) / _e2062) + ((_e2060 * _e2060) / _e2065)));
                    let _e2078 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(_e54, _e55, _e56));
                    let _e2081 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                    let _e2084 = dot(vec3<f32>(_e1746, _e1747, _e1748), vec3<f32>(_e54, _e55, _e56));
                    let _e2087 = dot(vec3<f32>(_e1746, _e1747, _e1748), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                    let _e2114 = ((_e784 + ((1f - _e784) * pow(_e1810, 5f))) * ((1f / (_e2068 * (((_e2051 * 3.1415927f) * _e2054) * _e2068))) * (0.5f / max(((_e46 * sqrt(max((_e1940 + ((_e2062 * (_e2084 * _e2084)) + (_e2065 * (_e2087 * _e2087)))), 0.000001f))) + (_e1771 * sqrt(max((_e566 + ((_e2062 * (_e2078 * _e2078)) + (_e2065 * (_e2081 * _e2081)))), 0.000001f)))), 0.000001f))));
                    coat_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e2114 * _e360) * _e337);
                    coat_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e2114 * _e366) * _e337);
                    coat_u002e_i_u002e_sroa_u002e_8_u002e_0_ = ((_e2114 * _e372) * _e337);
                } else {
                    coat_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    coat_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    coat_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2131 = coat_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2133 = coat_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2135 = coat_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e307 <= 0f)) {
                    let _e2159 = ((_e1781 * ((1f / ((_e798 * (((1f / _e46) + (1f / _e1771)) + -2f)) + 1f)) * (_e798 / (pow((((_e798 + -1f) * _e1914) + 1f), 2f) * 3.1415927f)))) / max(((_e46 * 4f) * _e1771), 0.000001f));
                    fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e2159 * bitcast<f32>(_e318)) * _e307);
                    fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e2159 * bitcast<f32>(_e324)) * _e307);
                    fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_ = ((_e2159 * bitcast<f32>(_e330)) * _e307);
                } else {
                    fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2176 = fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2178 = fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2180 = fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e193 <= 0f)) {
                    let _e2182 = (_e745 * sqrt(_e752));
                    if !((_e199 <= 0f)) {
                        extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (-(log(max(_e215, 0.000001f))) / _e199);
                        extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (-(log(max(_e221, 0.000001f))) / _e199);
                        extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (-(log(max(_e227, 0.000001f))) / _e199);
                    } else {
                        extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e2202 = extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                    let _e2204 = extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                    let _e2206 = extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                    let _e2213 = (_e2182 * _e2182);
                    let _e2218 = ((_e1914 * (_e2213 + -1f)) + 1f);
                    let _e2224 = (1f - _e2213);
                    let _e2243 = ((1f - (_e744 + ((1f - _e744) * pow((1f - _e46), 5f)))) * ((_e2213 / (_e2218 * (_e2218 * 3.1415927f))) * (0.5f / max(((_e46 * sqrt(max((_e2213 + (_e1940 * _e2224)), 0.000001f))) + (_e1771 * sqrt(max((_e2213 + (_e566 * _e2224)), 0.000001f)))), 0.000001f))));
                    transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (((exp((_e199 * -(_e2202))) * _e2243) * _e215) * _e193);
                    transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (((exp((_e199 * -(_e2204))) * _e2243) * _e221) * _e193);
                    transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (((exp((_e199 * -(_e2206))) * _e2243) * _e227) * _e193);
                } else {
                    transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2272 = transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2274 = transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2276 = transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e253 <= 0f)) {
                    let _e2281 = length(vec3<f32>((_e758 - _e1986), (_e759 - _e1987), (_e760 - _e1988)));
                    let _e2283 = sqrt(3f);
                    let _e2306 = -(_e2281);
                    let _e2314 = (max(((clamp(((_e46 * _e1771) + (((_e570 + max((1f - _e1940), 0f)) - (_e2281 * _e2281)) * 0.5f)), -1f, 1f) * bitcast<f32>(_e270)) + 1f), 0f) * 0.31830987f);
                    subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (((exp((((1f / max((_e259 * bitcast<f32>(_e264)), 0.000001f)) * _e2283) * _e2306)) * _e2314) * _e277) * _e253);
                    subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (((exp((((1f / max((_e259 * bitcast<f32>(_e294)), 0.000001f)) * _e2283) * _e2306)) * _e2314) * _e283) * _e253);
                    subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (((exp((((1f / max((_e259 * bitcast<f32>(_e300)), 0.000001f)) * _e2283) * _e2306)) * _e2314) * _e289) * _e253);
                } else {
                    subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2334 = subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2336 = subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2338 = subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                lo_u002e_i_u002e_sroa_u002e_0_u002e_1_ = (_e854 + (_e1680 * (_e1771 * ((((_e721 * ((_e623 * ((((max(((_e1843 + (_e1850 * _e1874)) - ((_e1852 - (_e1852 * _e175)) * _e1872)), 0f) * _e110) * _e1922) * _e1960) * _e74)) + (_e623 * (((_e763 * _e92) * _e2030) * _e74)))) + _e2131) + (_e2176 + _e2272)) + _e2334))));
                lo_u002e_i_u002e_sroa_u002e_6_u002e_1_ = (_e852 + (_e1682 * (_e1771 * ((((_e723 * ((_e628 * ((((max(((_e1844 + (_e1854 * _e1874)) - ((_e1856 - (_e1856 * _e181)) * _e1872)), 0f) * _e110) * _e1922) * _e1960) * _e74)) + (_e628 * (((_e763 * _e98) * _e2030) * _e74)))) + _e2133) + (_e2178 + _e2274)) + _e2336))));
                lo_u002e_i_u002e_sroa_u002e_9_u002e_1_ = (_e850 + (_e1684 * (_e1771 * ((((_e725 * ((_e633 * ((((max(((_e1845 + (_e1858 * _e1874)) - ((_e1860 - (_e1860 * _e187)) * _e1872)), 0f) * _e110) * _e1922) * _e1960) * _e74)) + (_e633 * (((_e763 * _e104) * _e2030) * _e74)))) + _e2135) + (_e2180 + _e2276)) + _e2338))));
            } else {
                lo_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e854;
                lo_u002e_i_u002e_sroa_u002e_6_u002e_1_ = _e852;
                lo_u002e_i_u002e_sroa_u002e_9_u002e_1_ = _e850;
            }
            let _e2367 = lo_u002e_i_u002e_sroa_u002e_0_u002e_1_;
            out_u002e_lo_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e2367;
            let _e2370 = lo_u002e_i_u002e_sroa_u002e_6_u002e_1_;
            out_u002e_lo_u002e_i_u002e_sroa_u002e_6_u002e_1_ = _e2370;
            let _e2373 = lo_u002e_i_u002e_sroa_u002e_9_u002e_1_;
            out_u002e_lo_u002e_i_u002e_sroa_u002e_9_u002e_1_ = _e2373;
            let _e2376 = (_e848 + 1u);
            if (_e2376 == _e726) {
                lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e2367;
                lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e2370;
                lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e2373;
                break;
            } else {
                k_u002e_sroa_u002e_0_u002e_0_ = _e2376;
                lo_u002e_i_u002e_sroa_u002e_9_ = _e2373;
                lo_u002e_i_u002e_sroa_u002e_6_ = _e2370;
                lo_u002e_i_u002e_sroa_u002e_0_ = _e2367;
                continue;
            }
        }
    }
    let _e2392 = lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
    let _e2394 = lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
    let _e2396 = lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
    let _e2406 = uniform_struct_0_3_.f0_.x;
    let _e2410 = uniform_struct_0_3_.f0_.y;
    let _e2414 = uniform_struct_0_3_.f0_.z;
    if !((_e402 <= 0f)) {
        let _e2421 = (_e402 * bitcast<f32>(_e407));
        let _e2422 = (_e402 * bitcast<f32>(_e413));
        let _e2423 = (_e402 * bitcast<f32>(_e419));
        let _e2433 = ((1f - _e337) + ((1f - pow((1f - _e46), 5f)) * _e337));
        emission_u002e_i_u002e_sroa_u002e_0_u002e_1_ = (_e2421 + ((((_e2421 * _e360) * _e2433) - _e2421) * _e337));
        emission_u002e_i_u002e_sroa_u002e_7_u002e_1_ = (_e2422 + ((((_e2422 * _e366) * _e2433) - _e2422) * _e337));
        emission_u002e_i_u002e_sroa_u002e_10_u002e_1_ = (_e2423 + ((((_e2423 * _e372) * _e2433) - _e2423) * _e337));
    } else {
        emission_u002e_i_u002e_sroa_u002e_0_u002e_1_ = 0f;
        emission_u002e_i_u002e_sroa_u002e_7_u002e_1_ = 0f;
        emission_u002e_i_u002e_sroa_u002e_10_u002e_1_ = 0f;
    }
    let _e2459 = emission_u002e_i_u002e_sroa_u002e_0_u002e_1_;
    let _e2461 = emission_u002e_i_u002e_sroa_u002e_7_u002e_1_;
    let _e2463 = emission_u002e_i_u002e_sroa_u002e_10_u002e_1_;
    let _e2468 = ((_e163 + -1f) / (_e163 + 1f));
    let _e2469 = (_e2468 * _e2468);
    let _e2477 = (dot(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e37, _e38, _e39)) * 2f);
    let _e2481 = ((_e15 * _e2477) - _e37);
    let _e2482 = ((_e16 * _e2477) - _e38);
    let _e2483 = ((_e17 * _e2477) - _e39);
    let _e2486 = uniform_struct_0_3_.f4_;
    let _e2487 = (_e154 * _e2486);
    let _e2491 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2481, _e2482, _e2483), _e2487);
    let _e2496 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2481, _e2482, _e2483), _e2487);
    let _e2501 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2481, _e2482, _e2483), _e2487);
    let _e2506 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e15, _e16, _e17));
    let _e2511 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e15, _e16, _e17));
    let _e2516 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e15, _e16, _e17));
    let _e2521 = textureSample(tex_0_9_, sampler_0_10_, vec2<f32>(_e46, _e154));
    let _e2526 = textureSample(tex_0_9_, sampler_0_10_, vec2<f32>(_e46, _e154));
    let _e2529 = (1f - _e86);
    let _e2535 = uniform_struct_0_3_.f3_;
    let _e2548 = max((1f - _e2526.y), 0f);
    let _e2572 = !((bitcast<f32>(_e435) >= 0.5f));
    let _e2573 = select(_e2459, (((_e2392 + (mix(_e92, (_e92 * _e110), _e86) * _e2406)) + _e2459) + (_e2535 * ((_e2491.x * ((mix(_e2469, _e641, _e86) * _e2521.x) + _e2526.y)) + (((_e2529 * _e92) * _e2506.x) * _e2548)))), _e2572);
    let _e2574 = select(_e2461, (((_e2394 + (mix(_e98, (_e98 * _e110), _e86) * _e2410)) + _e2461) + (_e2535 * ((_e2496.y * ((mix(_e2469, _e642, _e86) * _e2521.x) + _e2526.y)) + (((_e2529 * _e98) * _e2511.y) * _e2548)))), _e2572);
    let _e2575 = select(_e2463, (((_e2396 + (mix(_e104, (_e104 * _e110), _e86) * _e2414)) + _e2463) + (_e2535 * ((_e2501.z * ((mix(_e2469, _e643, _e86) * _e2521.x) + _e2526.y)) + (((_e2529 * _e104) * _e2516.z) * _e2548)))), _e2572);
    if ((_e2573 <= _e2573) && (0f <= 0f)) {
        if (abs(_e2573) < 100000000000000000000f) {
            _2_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2573;
        } else {
            _2_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        _2_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2589 = _2_u002e_sroa_u002e_0_u002e_0_u002e_i;
    if ((_e2574 <= _e2574) && (0f <= 0f)) {
        if (abs(_e2574) < 100000000000000000000f) {
            _3_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2574;
        } else {
            _3_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        _3_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2603 = _3_u002e_sroa_u002e_0_u002e_0_u002e_i;
    if ((_e2575 <= _e2575) && (0f <= 0f)) {
        if (abs(_e2575) < 100000000000000000000f) {
            _4_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2575;
            let _e2613 = _4_u002e_sroa_u002e_0_u002e_0_u002e_i;
            fragment_out = vec4<f32>(_e2589, _e2603, _e2613, bitcast<f32>(_e425));
            let _e2618 = fragment_out;
            return _e2618;
        }
        _4_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    } else {
        _4_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2624 = _4_u002e_sroa_u002e_0_u002e_0_u002e_i;
    fragment_out = vec4<f32>(_e2589, _e2603, _e2624, bitcast<f32>(_e425));
    let _e2629 = fragment_out;
    return _e2629;
}
