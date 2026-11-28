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
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_in: u32;
    var radiance_u002e_sroa_u002e_0_u002e_2_: f32;
    var radiance_u002e_sroa_u002e_10_u002e_2_: f32;
    var radiance_u002e_sroa_u002e_15_u002e_2_: f32;
    var vis_u002e_sroa_u002e_0_u002e_0_: f32;
    var lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa: f32;
    var lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa: f32;
    var extinction_u002e_sroa_u002e_0_u002e_0_: f32;
    var extinction_u002e_sroa_u002e_5_u002e_0_: f32;
    var extinction_u002e_sroa_u002e_8_u002e_0_: f32;
    var emission_u002e_sroa_u002e_0_u002e_1_: f32;
    var emission_u002e_sroa_u002e_7_u002e_1_: f32;
    var emission_u002e_sroa_u002e_10_u002e_1_: f32;
    var subsurface_u002e_sroa_u002e_0_u002e_0_: f32;
    var subsurface_u002e_sroa_u002e_5_u002e_0_: f32;
    var subsurface_u002e_sroa_u002e_8_u002e_0_: f32;
    var transmission_u002e_sroa_u002e_0_u002e_0_: f32;
    var transmission_u002e_sroa_u002e_5_u002e_0_: f32;
    var transmission_u002e_sroa_u002e_8_u002e_0_: f32;
    var _4_u002e_sroa_u002e_0_u002e_0_: f32;
    var coat_u002e_sroa_u002e_0_u002e_0_: f32;
    var coat_u002e_sroa_u002e_5_u002e_0_: f32;
    var coat_u002e_sroa_u002e_8_u002e_0_: f32;
    var fuzz_u002e_sroa_u002e_0_u002e_0_: f32;
    var fuzz_u002e_sroa_u002e_5_u002e_0_: f32;
    var fuzz_u002e_sroa_u002e_8_u002e_0_: f32;
    var coat_darken_u002e_sroa_u002e_0_u002e_0_: f32;
    var coat_darken_u002e_sroa_u002e_5_u002e_0_: f32;
    var coat_darken_u002e_sroa_u002e_7_u002e_0_: f32;
    var k_u002e_sroa_u002e_0_u002e_0_: u32;
    var lo_u002e_sroa_u002e_9_: f32;
    var lo_u002e_sroa_u002e_6_: f32;
    var lo_u002e_sroa_u002e_0_: f32;
    var _2_u002e_sroa_u002e_0_u002e_0_: f32;
    var lo_u002e_sroa_u002e_0_u002e_1_: f32;
    var lo_u002e_sroa_u002e_6_u002e_1_: f32;
    var lo_u002e_sroa_u002e_9_u002e_1_: f32;
    var radiance_u002e_sroa_u002e_0_u002e_0_: f32;
    var radiance_u002e_sroa_u002e_10_u002e_0_: f32;
    var radiance_u002e_sroa_u002e_15_u002e_0_: f32;
    var radiance_u002e_sroa_u002e_0_u002e_1_: f32;
    var radiance_u002e_sroa_u002e_10_u002e_1_: f32;
    var radiance_u002e_sroa_u002e_15_u002e_1_: f32;
    var _3_u002e_sroa_u002e_0_u002e_0_: f32;
    var out_u002e_lo_u002e_sroa_u002e_0_u002e_1_: f32;
    var out_u002e_lo_u002e_sroa_u002e_6_u002e_1_: f32;
    var out_u002e_lo_u002e_sroa_u002e_9_u002e_1_: f32;

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
    let _e441 = vec2<f32>(in2_.x, in2_.y);
    let _e444 = textureSample(tex_0_16_, sampler_0_19_, _e441);
    let _e449 = textureSample(tex_0_17_, sampler_0_19_, _e441);
    let _e453 = textureSample(tex_0_18_, sampler_0_19_, _e441);
    let _e457 = (_e444.x * bitcast<f32>(_e92));
    let _e458 = (_e444.y * bitcast<f32>(_e98));
    let _e459 = (_e444.z * bitcast<f32>(_e104));
    let _e460 = (_e449.z * bitcast<f32>(_e86));
    let _e461 = (_e449.y * bitcast<f32>(_e158));
    let _e462 = dpdx(_e16);
    let _e463 = dpdx(_e17);
    let _e464 = dpdx(_e18);
    let _e465 = dpdy(_e16);
    let _e466 = dpdy(_e17);
    let _e467 = dpdy(_e18);
    let _e478 = (((dot(vec3<f32>(_e462, _e463, _e464), vec3<f32>(_e462, _e463, _e464)) + dot(vec3<f32>(_e465, _e466, _e467), vec3<f32>(_e465, _e466, _e467))) * 0.15f) * 2f);
    let _e483 = (_e461 * _e461);
    let _e485 = ((_e483 * _e483) + select(0.2f, _e478, (_e478 < 0.2f)));
    let _e489 = select(_e485, 0f, (_e485 < 0f));
    let _e493 = select(_e489, 1f, (_e489 > 1f));
    let _e499 = sqrt(sqrt(select(0f, _e493, (_e493 > 0f))));
    let _e506 = uniform_struct_0_0_.f0_.x;
    let _e510 = uniform_struct_0_0_.f1_.x;
    let _e514 = uniform_struct_0_0_.f2_.x;
    let _e518 = uniform_struct_0_0_.f3_.x;
    let _e528 = uniform_struct_0_0_.f0_.y;
    let _e532 = uniform_struct_0_0_.f1_.y;
    let _e536 = uniform_struct_0_0_.f2_.y;
    let _e540 = uniform_struct_0_0_.f3_.y;
    let _e550 = uniform_struct_0_0_.f0_.w;
    let _e554 = uniform_struct_0_0_.f1_.w;
    let _e558 = uniform_struct_0_0_.f2_.w;
    let _e562 = uniform_struct_0_0_.f3_.w;
    let _e568 = (_e562 + (((in0_.x * _e550) + (in0_.y * _e554)) + (in0_.z * _e558)));
    let _e572 = select(0.5f, _e568, (_e568 > 0.5f));
    let _e613 = ((((min(u32(floor((((((_e540 + (((in0_.x * _e528) + (in0_.y * _e532)) + (in0_.z * _e536))) / _e572) * 0.5f) + 0.5f) * 9f))), 8u) << 4u) | min(u32(floor((((((_e518 + (((in0_.x * _e506) + (in0_.y * _e510)) + (in0_.z * _e514))) / _e572) * 0.5f) + 0.5f) * 16f))), 15u)) + (min(u32(ceil(((log((_e572 * 2f)) * 3.4407384f) + -0.0001f))), 23u) * 144u)) << 1u);
    let _e616 = uniform_struct_0_12_.f2_;
    let _e619 = uniform_struct_0_3_.f2_;
    let _e622 = storage_0_15_[_e613];
    let _e628 = storage_0_15_[(_e613 | 1u)];
    let _e632 = (_e47 * _e47);
    let _e636 = max((1f - _e632), 0f);
    let _e638 = (sqrt(_e636) / _e398);
    let _e651 = pow(((_e398 + -1f) / (_e398 + 1f)), 2f);
    let _e655 = (((bitcast<f32>(_e391) * 1000f) * (_e398 * 12.566371f)) * sqrt(max((1f - (_e638 * _e638)), 0f)));
    let _e666 = (_e651 * 2f);
    let _e672 = (1f - (_e651 * _e651));
    let _e684 = clamp(bitcast<f32>(_e385), 0f, 1f);
    let _e689 = ((_e684 * ((((_e666 * cos((_e655 / 650f))) / _e672) + 1f) + -1f)) + 1f);
    let _e694 = ((_e684 * ((((_e666 * cos((_e655 / 550f))) / _e672) + 1f) + -1f)) + 1f);
    let _e699 = ((_e684 * ((((_e666 * cos((_e655 / 450f))) / _e672) + 1f) + -1f)) + 1f);
    let _e704 = ((_e380 + -1f) / (_e380 + 1f));
    let _e707 = (_e457 * _e75);
    let _e708 = (_e458 * _e75);
    let _e709 = (_e459 * _e75);
    let _e712 = !((_e339 <= 0f));
    if _e712 {
        let _e718 = (1f - ((1f - pow(_e704, 2f)) / (_e380 * _e380)));
        let _e720 = (1f - _e718);
        let _e721 = (_e709 * _e111);
        let _e734 = (_e708 * _e111);
        let _e747 = (_e707 * _e111);
        let _e761 = (_e339 * bitcast<f32>(_e356));
        coat_darken_u002e_sroa_u002e_0_u002e_0_ = ((_e761 * ((_e720 / max((1f - ((_e747 + (((((_e457 - _e279) * _e255) + _e279) - _e747) * _e460)) * _e718)), 0.000001f)) + -1f)) + 1f);
        coat_darken_u002e_sroa_u002e_5_u002e_0_ = ((_e761 * ((_e720 / max((1f - ((_e734 + (_e460 * ((((_e458 - _e285) * _e255) + _e285) - _e734))) * _e718)), 0.000001f)) + -1f)) + 1f);
        coat_darken_u002e_sroa_u002e_7_u002e_0_ = ((_e761 * ((_e720 / max((1f - ((_e721 + (_e460 * ((((_e459 - _e291) * _e255) + _e291) - _e721))) * _e718)), 0.000001f)) + -1f)) + 1f);
    } else {
        coat_darken_u002e_sroa_u002e_0_u002e_0_ = 1f;
        coat_darken_u002e_sroa_u002e_5_u002e_0_ = 1f;
        coat_darken_u002e_sroa_u002e_7_u002e_0_ = 1f;
    }
    let _e787 = coat_darken_u002e_sroa_u002e_0_u002e_0_;
    let _e789 = coat_darken_u002e_sroa_u002e_5_u002e_0_;
    let _e791 = coat_darken_u002e_sroa_u002e_7_u002e_0_;
    let _e792 = ((_e628 * _e616) + _e619);
    if (_e792 == 0u) {
        lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
        lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = 0f;
        lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = 0f;
    } else {
        let _e809 = ((_e165 + -1f) / (_e165 + 1f));
        let _e810 = (_e809 * _e809);
        let _e811 = (_e499 * _e499);
        let _e813 = (1f - bitcast<f32>(_e170));
        let _e818 = (2f / ((_e813 * _e813) + 1f));
        let _e820 = select(_e499, _e81, (_e499 < _e81));
        let _e824 = (_e38 - (_e16 * _e47));
        let _e825 = (_e39 - (_e17 * _e47));
        let _e826 = (_e40 - (_e18 * _e47));
        let _e829 = (1f - _e460);
        let _e834 = max((_e820 * _e820), 0.000001f);
        let _e835 = (_e834 * _e834);
        let _e850 = (_e704 * _e704);
        let _e853 = (_e345 * _e345);
        let _e855 = (1f - bitcast<f32>(_e350));
        let _e860 = (2f / ((_e855 * _e855) + 1f));
        let _e864 = (_e315 * _e315);
        k_u002e_sroa_u002e_0_u002e_0_ = 0u;
        lo_u002e_sroa_u002e_9_ = 0f;
        lo_u002e_sroa_u002e_6_ = 0f;
        lo_u002e_sroa_u002e_0_ = 0f;
        loop {
            let _e914 = k_u002e_sroa_u002e_0_u002e_0_;
            let _e916 = lo_u002e_sroa_u002e_9_;
            let _e918 = lo_u002e_sroa_u002e_6_;
            let _e920 = lo_u002e_sroa_u002e_0_;
            if (_e914 < _e619) {
                _0_u002e_sroa_u002e_0_u002e_0_u002e_in = _e914;
            } else {
                let _e926 = storage_0_14_[(((_e622 * _e616) - _e619) + _e914)];
                _0_u002e_sroa_u002e_0_u002e_0_u002e_in = _e926;
            }
            let _e929 = _0_u002e_sroa_u002e_0_u002e_0_u002e_in;
            let _e931 = (_e929 * 36u);
            let _e934 = storage_0_13_[_e931];
            let _e935 = bitcast<f32>(_e934);
            let _e937 = (_e931 + 8u);
            let _e940 = storage_0_13_[_e937];
            let _e943 = (_e931 + 9u);
            let _e946 = storage_0_13_[_e943];
            let _e949 = (_e931 + 10u);
            let _e952 = storage_0_13_[_e949];
            let _e958 = length(vec3<f32>((bitcast<f32>(_e940) - in0_.x), (bitcast<f32>(_e946) - in0_.y), (bitcast<f32>(_e952) - in0_.z)));
            let _e963 = select(1f, 0f, (_e935 < 0.5f));
            let _e965 = (_e931 + 16u);
            let _e968 = storage_0_13_[_e965];
            let _e969 = bitcast<f32>(_e968);
            let _e981 = storage_0_13_[(_e931 + 12u)];
            let _e987 = storage_0_13_[(_e931 + 13u)];
            let _e993 = storage_0_13_[(_e931 + 14u)];
            let _e999 = storage_0_13_[(_e931 + 15u)];
            let _e1000 = bitcast<f32>(_e999);
            let _e1003 = storage_0_13_[_e937];
            let _e1007 = storage_0_13_[_e943];
            let _e1011 = storage_0_13_[_e949];
            let _e1013 = (in0_.x - bitcast<f32>(_e1003));
            let _e1014 = (in0_.y - bitcast<f32>(_e1007));
            let _e1015 = (in0_.z - bitcast<f32>(_e1011));
            let _e1019 = (1f / length(vec3<f32>(_e1013, _e1014, _e1015)));
            let _e1024 = (_e931 + 4u);
            let _e1027 = storage_0_13_[_e1024];
            let _e1028 = bitcast<f32>(_e1027);
            let _e1030 = (_e931 + 5u);
            let _e1033 = storage_0_13_[_e1030];
            let _e1034 = bitcast<f32>(_e1033);
            let _e1036 = (_e931 + 6u);
            let _e1039 = storage_0_13_[_e1036];
            let _e1040 = bitcast<f32>(_e1039);
            let _e1044 = (1f / length(vec3<f32>(_e1028, _e1034, _e1040)));
            let _e1055 = storage_0_13_[(_e931 + 18u)];
            let _e1061 = storage_0_13_[(_e931 + 17u)];
            let _e1068 = select(1f, 0f, (_e935 < 1.5f));
            let _e1070 = mix(1f, smoothstep(bitcast<f32>(_e1055), bitcast<f32>(_e1061), dot(vec3<f32>((_e1044 * _e1028), (_e1044 * _e1034), (_e1044 * _e1040)), vec3<f32>((_e1013 * _e1019), (_e1014 * _e1019), (_e1015 * _e1019)))), _e1068);
            let _e1071 = (_e958 * _e958);
            let _e1083 = (mix(1f, (1f - smoothstep((_e969 * 0.99f), _e969, _e958)), _e963) * mix(1f, (1f / select(_e1071, 0.000001f, (_e1071 < 0.000001f))), _e963));
            let _e1087 = (_e1070 * ((bitcast<f32>(_e981) * _e1000) * _e1083));
            let _e1088 = (_e1070 * ((bitcast<f32>(_e987) * _e1000) * _e1083));
            let _e1089 = (_e1070 * ((bitcast<f32>(_e993) * _e1000) * _e1083));
            let _e1092 = (_e963 * (1f - _e1068));
            if (((1f - _e963) * select(1f, 0f, (_e969 < 0.5f))) > 0.5f) {
                let _e1107 = uniform_struct_0_3_.f6_.f1_.x;
                let _e1112 = uniform_struct_0_3_.f6_.f1_.y;
                let _e1117 = uniform_struct_0_3_.f6_.f1_.z;
                let _e1120 = dot(vec3<f32>((in0_.x - _e22), (in0_.y - _e26), (in0_.z - _e30)), vec3<f32>(_e1107, _e1112, _e1117));
                let _e1125 = uniform_struct_0_3_.f6_.f2_.x;
                let _e1130 = uniform_struct_0_3_.f6_.f2_.y;
                let _e1135 = uniform_struct_0_3_.f6_.f2_.z;
                let _e1140 = uniform_struct_0_3_.f6_.f2_.w;
                let _e1144 = uniform_struct_0_3_.f6_.f3_;
                let _e1148 = select(1f, 0f, (_e1120 < _e1130));
                let _e1152 = select(1f, 0f, (_e1120 < _e1135));
                let _e1156 = select(1f, 0f, (_e1120 < _e1140));
                let _e1159 = u32(((_e1148 + _e1152) + _e1156));
                let _e1162 = mix(mix(mix(_e1130, _e1135, _e1148), _e1140, _e1152), _e1144, _e1156);
                let _e1172 = uniform_struct_0_3_.f6_.f0_[_e1159].f0_.x;
                let _e1179 = uniform_struct_0_3_.f6_.f0_[_e1159].f1_.x;
                let _e1186 = uniform_struct_0_3_.f6_.f0_[_e1159].f2_.x;
                let _e1193 = uniform_struct_0_3_.f6_.f0_[_e1159].f3_.x;
                let _e1206 = uniform_struct_0_3_.f6_.f0_[_e1159].f0_.y;
                let _e1213 = uniform_struct_0_3_.f6_.f0_[_e1159].f1_.y;
                let _e1220 = uniform_struct_0_3_.f6_.f0_[_e1159].f2_.y;
                let _e1227 = uniform_struct_0_3_.f6_.f0_[_e1159].f3_.y;
                let _e1240 = uniform_struct_0_3_.f6_.f0_[_e1159].f0_.z;
                let _e1247 = uniform_struct_0_3_.f6_.f0_[_e1159].f1_.z;
                let _e1254 = uniform_struct_0_3_.f6_.f0_[_e1159].f2_.z;
                let _e1261 = uniform_struct_0_3_.f6_.f0_[_e1159].f3_.z;
                let _e1274 = uniform_struct_0_3_.f6_.f0_[_e1159].f0_.w;
                let _e1281 = uniform_struct_0_3_.f6_.f0_[_e1159].f1_.w;
                let _e1288 = uniform_struct_0_3_.f6_.f0_[_e1159].f2_.w;
                let _e1295 = uniform_struct_0_3_.f6_.f0_[_e1159].f3_.w;
                let _e1301 = (_e1295 + (((in0_.x * _e1274) + (in0_.y * _e1281)) + (in0_.z * _e1288)));
                let _e1318 = textureSampleCompareLevel(tex_0_11_, sampler_0_5_, vec2<f32>(((((_e1193 + (((in0_.x * _e1172) + (in0_.y * _e1179)) + (in0_.z * _e1186))) / _e1301) * 0.5f) + 0.5f), (0.5f - (((_e1227 + (((in0_.x * _e1206) + (in0_.y * _e1213)) + (in0_.z * _e1220))) / _e1301) * 0.5f))), _e1159, (((_e1261 + (((in0_.x * _e1240) + (in0_.y * _e1247)) + (in0_.z * _e1254))) / _e1301) + -0.001f));
                if (_e1159 < 3u) {
                    let _e1323 = ((_e1162 - mix(mix(mix(_e1125, _e1130, _e1148), _e1135, _e1152), _e1140, _e1156)) * 0.08f);
                    let _e1329 = ((_e1162 - _e1120) / select(_e1323, 0.000001f, (_e1323 < 0.000001f)));
                    let _e1333 = select(_e1329, 0f, (_e1329 < 0f));
                    let _e1339 = (_e1159 + 1u);
                    let _e1346 = uniform_struct_0_3_.f6_.f0_[_e1339].f0_.x;
                    let _e1353 = uniform_struct_0_3_.f6_.f0_[_e1339].f1_.x;
                    let _e1360 = uniform_struct_0_3_.f6_.f0_[_e1339].f2_.x;
                    let _e1367 = uniform_struct_0_3_.f6_.f0_[_e1339].f3_.x;
                    let _e1380 = uniform_struct_0_3_.f6_.f0_[_e1339].f0_.y;
                    let _e1387 = uniform_struct_0_3_.f6_.f0_[_e1339].f1_.y;
                    let _e1394 = uniform_struct_0_3_.f6_.f0_[_e1339].f2_.y;
                    let _e1401 = uniform_struct_0_3_.f6_.f0_[_e1339].f3_.y;
                    let _e1414 = uniform_struct_0_3_.f6_.f0_[_e1339].f0_.z;
                    let _e1421 = uniform_struct_0_3_.f6_.f0_[_e1339].f1_.z;
                    let _e1428 = uniform_struct_0_3_.f6_.f0_[_e1339].f2_.z;
                    let _e1435 = uniform_struct_0_3_.f6_.f0_[_e1339].f3_.z;
                    let _e1448 = uniform_struct_0_3_.f6_.f0_[_e1339].f0_.w;
                    let _e1455 = uniform_struct_0_3_.f6_.f0_[_e1339].f1_.w;
                    let _e1462 = uniform_struct_0_3_.f6_.f0_[_e1339].f2_.w;
                    let _e1469 = uniform_struct_0_3_.f6_.f0_[_e1339].f3_.w;
                    let _e1475 = (_e1469 + (((in0_.x * _e1448) + (in0_.y * _e1455)) + (in0_.z * _e1462)));
                    let _e1492 = textureSampleCompareLevel(tex_0_11_, sampler_0_5_, vec2<f32>(((((_e1367 + (((in0_.x * _e1346) + (in0_.y * _e1353)) + (in0_.z * _e1360))) / _e1475) * 0.5f) + 0.5f), (0.5f - (((_e1401 + (((in0_.x * _e1380) + (in0_.y * _e1387)) + (in0_.z * _e1394))) / _e1475) * 0.5f))), _e1339, (((_e1435 + (((in0_.x * _e1414) + (in0_.y * _e1421)) + (in0_.z * _e1428))) / _e1475) + -0.001f));
                    vis_u002e_sroa_u002e_0_u002e_0_ = mix(_e1492, _e1318, select(_e1333, 1f, (_e1333 > 1f)));
                } else {
                    vis_u002e_sroa_u002e_0_u002e_0_ = _e1318;
                }
                let _e1497 = vis_u002e_sroa_u002e_0_u002e_0_;
                radiance_u002e_sroa_u002e_0_u002e_0_ = (_e1087 * _e1497);
                radiance_u002e_sroa_u002e_10_u002e_0_ = (_e1088 * _e1497);
                radiance_u002e_sroa_u002e_15_u002e_0_ = (_e1089 * _e1497);
            } else {
                radiance_u002e_sroa_u002e_0_u002e_0_ = _e1087;
                radiance_u002e_sroa_u002e_10_u002e_0_ = _e1088;
                radiance_u002e_sroa_u002e_15_u002e_0_ = _e1089;
            }
            let _e1508 = radiance_u002e_sroa_u002e_0_u002e_0_;
            let _e1510 = radiance_u002e_sroa_u002e_10_u002e_0_;
            let _e1512 = radiance_u002e_sroa_u002e_15_u002e_0_;
            let _e1514 = (_e931 + 19u);
            let _e1517 = storage_0_13_[_e1514];
            if !((bitcast<f32>(_e1517) >= 0f)) {
                radiance_u002e_sroa_u002e_0_u002e_2_ = _e1508;
                radiance_u002e_sroa_u002e_10_u002e_2_ = _e1510;
                radiance_u002e_sroa_u002e_15_u002e_2_ = _e1512;
            } else {
                if !((_e1092 <= 0.5f)) {
                    radiance_u002e_sroa_u002e_0_u002e_1_ = _e1508;
                    radiance_u002e_sroa_u002e_10_u002e_1_ = _e1510;
                    radiance_u002e_sroa_u002e_15_u002e_1_ = _e1512;
                } else {
                    let _e1535 = storage_0_13_[(_e931 + 20u)];
                    let _e1541 = storage_0_13_[(_e931 + 24u)];
                    let _e1547 = storage_0_13_[(_e931 + 28u)];
                    let _e1553 = storage_0_13_[(_e931 + 32u)];
                    let _e1565 = storage_0_13_[(_e931 + 21u)];
                    let _e1571 = storage_0_13_[(_e931 + 25u)];
                    let _e1577 = storage_0_13_[(_e931 + 29u)];
                    let _e1583 = storage_0_13_[(_e931 + 33u)];
                    let _e1595 = storage_0_13_[(_e931 + 22u)];
                    let _e1601 = storage_0_13_[(_e931 + 26u)];
                    let _e1607 = storage_0_13_[(_e931 + 30u)];
                    let _e1613 = storage_0_13_[(_e931 + 34u)];
                    let _e1625 = storage_0_13_[(_e931 + 23u)];
                    let _e1631 = storage_0_13_[(_e931 + 27u)];
                    let _e1637 = storage_0_13_[(_e931 + 31u)];
                    let _e1643 = storage_0_13_[(_e931 + 35u)];
                    let _e1650 = ((((in0_.x * bitcast<f32>(_e1625)) + (in0_.y * bitcast<f32>(_e1631))) + (in0_.z * bitcast<f32>(_e1637))) + bitcast<f32>(_e1643));
                    let _e1663 = storage_0_13_[_e1514];
                    let _e1672 = textureSampleCompareLevel(tex_0_4_, sampler_0_5_, vec2<f32>((((((((in0_.x * bitcast<f32>(_e1535)) + (in0_.y * bitcast<f32>(_e1541))) + (in0_.z * bitcast<f32>(_e1547))) + bitcast<f32>(_e1553)) / _e1650) * 0.5f) + 0.5f), (0.5f - ((((((in0_.x * bitcast<f32>(_e1565)) + (in0_.y * bitcast<f32>(_e1571))) + (in0_.z * bitcast<f32>(_e1577))) + bitcast<f32>(_e1583)) / _e1650) * 0.5f))), u32(bitcast<f32>(_e1663)), ((((((in0_.x * bitcast<f32>(_e1595)) + (in0_.y * bitcast<f32>(_e1601))) + (in0_.z * bitcast<f32>(_e1607))) + bitcast<f32>(_e1613)) / _e1650) + -0.001f));
                    radiance_u002e_sroa_u002e_0_u002e_1_ = (_e1508 * _e1672);
                    radiance_u002e_sroa_u002e_10_u002e_1_ = (_e1510 * _e1672);
                    radiance_u002e_sroa_u002e_15_u002e_1_ = (_e1512 * _e1672);
                }
                let _e1680 = radiance_u002e_sroa_u002e_0_u002e_1_;
                let _e1682 = radiance_u002e_sroa_u002e_10_u002e_1_;
                let _e1684 = radiance_u002e_sroa_u002e_15_u002e_1_;
                if (_e1092 > 0.5f) {
                    let _e1689 = storage_0_13_[_e937];
                    let _e1693 = storage_0_13_[_e943];
                    let _e1697 = storage_0_13_[_e949];
                    let _e1699 = (in0_.x - bitcast<f32>(_e1689));
                    let _e1700 = (in0_.y - bitcast<f32>(_e1693));
                    let _e1701 = (in0_.z - bitcast<f32>(_e1697));
                    let _e1702 = abs(_e1699);
                    let _e1703 = abs(_e1700);
                    let _e1705 = select(_e1702, _e1703, (_e1703 > _e1702));
                    let _e1706 = abs(_e1701);
                    let _e1711 = storage_0_13_[_e965];
                    let _e1712 = bitcast<f32>(_e1711);
                    let _e1716 = select(_e1712, 1f, (_e1712 < 1f));
                    let _e1718 = (_e1716 + -0.1f);
                    let _e1729 = storage_0_13_[_e1514];
                    let _e1735 = textureSampleCompareLevel(tex_0_6_, sampler_0_5_, vec3<f32>(_e1699, _e1700, _e1701), u32(bitcast<f32>(_e1729)), (((_e1716 / _e1718) - ((_e1716 * 0.1f) / (select(_e1705, _e1706, (_e1706 > _e1705)) * _e1718))) + -0.001f));
                    radiance_u002e_sroa_u002e_0_u002e_2_ = (_e1680 * _e1735);
                    radiance_u002e_sroa_u002e_10_u002e_2_ = (_e1682 * _e1735);
                    radiance_u002e_sroa_u002e_15_u002e_2_ = (_e1684 * _e1735);
                } else {
                    radiance_u002e_sroa_u002e_0_u002e_2_ = _e1680;
                    radiance_u002e_sroa_u002e_10_u002e_2_ = _e1682;
                    radiance_u002e_sroa_u002e_15_u002e_2_ = _e1684;
                }
            }
            let _e1746 = radiance_u002e_sroa_u002e_0_u002e_2_;
            let _e1748 = radiance_u002e_sroa_u002e_10_u002e_2_;
            let _e1750 = radiance_u002e_sroa_u002e_15_u002e_2_;
            let _e1753 = storage_0_13_[_e931];
            let _e1757 = storage_0_13_[_e937];
            let _e1761 = storage_0_13_[_e943];
            let _e1765 = storage_0_13_[_e949];
            let _e1767 = (bitcast<f32>(_e1757) - in0_.x);
            let _e1768 = (bitcast<f32>(_e1761) - in0_.y);
            let _e1769 = (bitcast<f32>(_e1765) - in0_.z);
            let _e1771 = length(vec3<f32>(_e1767, _e1768, _e1769));
            let _e1777 = (1f / select(_e1771, 0.000001f, (_e1771 < 0.000001f)));
            let _e1785 = select(1f, 0f, (bitcast<f32>(_e1753) < 0.5f));
            let _e1788 = storage_0_13_[_e1024];
            let _e1789 = bitcast<f32>(_e1788);
            let _e1792 = storage_0_13_[_e1030];
            let _e1793 = bitcast<f32>(_e1792);
            let _e1796 = storage_0_13_[_e1036];
            let _e1797 = bitcast<f32>(_e1796);
            let _e1801 = (1f / length(vec3<f32>(_e1789, _e1793, _e1797)));
            let _e1805 = mix((_e1801 * _e1789), (_e1767 * _e1777), _e1785);
            let _e1806 = mix((_e1801 * _e1793), (_e1768 * _e1777), _e1785);
            let _e1807 = mix((_e1801 * _e1797), (_e1769 * _e1777), _e1785);
            let _e1811 = (1f / length(vec3<f32>(_e1805, _e1806, _e1807)));
            let _e1812 = (_e1805 * _e1811);
            let _e1813 = (_e1806 * _e1811);
            let _e1814 = (_e1807 * _e1811);
            let _e1819 = ((_e38 + _e1812) + 0f);
            let _e1821 = ((_e39 + _e1813) + 0f);
            let _e1823 = ((_e40 + _e1814) + 0.000001f);
            let _e1827 = (1f / length(vec3<f32>(_e1819, _e1821, _e1823)));
            let _e1828 = (_e1827 * _e1819);
            let _e1829 = (_e1827 * _e1821);
            let _e1830 = (_e1827 * _e1823);
            let _e1833 = dot(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e1812, _e1813, _e1814));
            let _e1837 = select(_e1833, 0.000001f, (_e1833 < 0.000001f));
            let _e1840 = dot(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e1828, _e1829, _e1830));
            let _e1843 = dot(vec3<f32>(_e38, _e39, _e40), vec3<f32>(_e1828, _e1829, _e1830));
            let _e1847 = select(_e1843, 0.000001f, (_e1843 < 0.000001f));
            if (_e1837 > 0.000001f) {
                let _e1853 = select(_e1840, 0.000001f, (_e1840 < 0.000001f));
                let _e1854 = mix(_e810, _e707, _e460);
                let _e1855 = mix(_e810, _e708, _e460);
                let _e1856 = mix(_e810, _e709, _e460);
                let _e1859 = pow(0.85714287f, 5f);
                let _e1861 = (1f - _e1854);
                let _e1863 = (_e1854 + (_e1861 * _e1859));
                let _e1865 = (1f - _e1855);
                let _e1867 = (_e1855 + (_e1865 * _e1859));
                let _e1869 = (1f - _e1856);
                let _e1871 = (_e1856 + (_e1869 * _e1859));
                let _e1876 = (1f - _e1847);
                let _e1885 = ((_e1847 * pow(_e1876, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e1887 = pow(_e1876, 5f);
                let _e1909 = mix(_e810, _e707, _e460);
                let _e1910 = mix(_e810, _e708, _e460);
                let _e1911 = mix(_e810, _e709, _e460);
                let _e1914 = pow(0.85714287f, 5f);
                let _e1916 = (1f - _e1909);
                let _e1918 = (_e1909 + (_e1916 * _e1914));
                let _e1920 = (1f - _e1910);
                let _e1922 = (_e1910 + (_e1920 * _e1914));
                let _e1924 = (1f - _e1911);
                let _e1926 = (_e1911 + (_e1924 * _e1914));
                let _e1938 = ((_e1847 * pow(_e1876, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e1940 = pow(_e1876, 5f);
                let _e1963 = (_e811 * sqrt(_e818));
                let _e1966 = (_e813 * (_e811 * sqrt(_e818)));
                let _e1969 = dot(vec3<f32>(_e1828, _e1829, _e1830), vec3<f32>(_e55, _e56, _e57));
                let _e1972 = dot(vec3<f32>(_e1828, _e1829, _e1830), vec3<f32>(cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).x, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).y, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).z));
                let _e1974 = (_e1963 * _e1963);
                let _e1977 = (_e1966 * _e1966);
                let _e1980 = (_e1853 * _e1853);
                let _e1981 = (_e1980 + (((_e1969 * _e1969) / _e1974) + ((_e1972 * _e1972) / _e1977)));
                let _e1988 = (1f / (_e1981 * (((_e1963 * 3.1415927f) * _e1966) * _e1981)));
                let _e1991 = dot(vec3<f32>(_e38, _e39, _e40), vec3<f32>(_e55, _e56, _e57));
                let _e1994 = dot(vec3<f32>(_e38, _e39, _e40), vec3<f32>(cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).x, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).y, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).z));
                let _e1997 = dot(vec3<f32>(_e1812, _e1813, _e1814), vec3<f32>(_e55, _e56, _e57));
                let _e2000 = dot(vec3<f32>(_e1812, _e1813, _e1814), vec3<f32>(cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).x, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).y, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).z));
                let _e2006 = (_e1837 * _e1837);
                let _e2026 = (0.5f / max(((_e47 * sqrt(max((_e2006 + ((_e1974 * (_e1997 * _e1997)) + (_e1977 * (_e2000 * _e2000)))), 0.000001f))) + (_e1837 * sqrt(max((_e632 + ((_e1974 * (_e1991 * _e1991)) + (_e1977 * (_e1994 * _e1994)))), 0.000001f)))), 0.000001f));
                let _e2045 = (1f / length(vec3<f32>(_e824, _e825, _e826)));
                let _e2052 = (_e1812 - (_e16 * _e1837));
                let _e2053 = (_e1813 - (_e17 * _e1837));
                let _e2054 = (_e1814 - (_e18 * _e1837));
                let _e2058 = (1f / length(vec3<f32>(_e2052, _e2053, _e2054)));
                let _e2064 = dot(vec3<f32>((_e824 * _e2045), (_e825 * _e2045), (_e826 * _e2045)), vec3<f32>((_e2052 * _e2058), (_e2053 * _e2058), (_e2054 * _e2058)));
                let _e2069 = acos(max(_e47, 0f));
                let _e2072 = acos(max(_e1837, 0f));
                let _e2096 = (max((1f - dot(vec3<f32>((max(((_e1854 + (_e1861 * _e1887)) - ((_e1863 - (_e1863 * _e177)) * _e1885)), 0f) * _e111), (max(((_e1855 + (_e1865 * _e1887)) - ((_e1867 - (_e1867 * _e183)) * _e1885)), 0f) * _e111), (max(((_e1856 + (_e1869 * _e1887)) - ((_e1871 - (_e1871 * _e189)) * _e1885)), 0f) * _e111)), vec3<f32>(0.2126f, 0.7152f, 0.0722f))), 0f) * (((1f - ((_e835 * 0.5f) / (_e835 + 0.57f))) + (tan(min(_e2069, _e2072)) * ((((_e835 * 0.45f) / (_e835 + 0.09f)) * select(0f, _e2064, (_e2064 > 0f))) * sin(max(_e2069, _e2072))))) * 0.31830987f));
                if _e712 {
                    let _e2117 = (_e853 * sqrt(_e860));
                    let _e2120 = (_e855 * (_e853 * sqrt(_e860)));
                    let _e2123 = dot(vec3<f32>(_e1828, _e1829, _e1830), vec3<f32>(_e55, _e56, _e57));
                    let _e2126 = dot(vec3<f32>(_e1828, _e1829, _e1830), vec3<f32>(cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).x, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).y, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).z));
                    let _e2128 = (_e2117 * _e2117);
                    let _e2131 = (_e2120 * _e2120);
                    let _e2134 = (_e1980 + (((_e2123 * _e2123) / _e2128) + ((_e2126 * _e2126) / _e2131)));
                    let _e2144 = dot(vec3<f32>(_e38, _e39, _e40), vec3<f32>(_e55, _e56, _e57));
                    let _e2147 = dot(vec3<f32>(_e38, _e39, _e40), vec3<f32>(cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).x, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).y, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).z));
                    let _e2150 = dot(vec3<f32>(_e1812, _e1813, _e1814), vec3<f32>(_e55, _e56, _e57));
                    let _e2153 = dot(vec3<f32>(_e1812, _e1813, _e1814), vec3<f32>(cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).x, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).y, cross(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e55, _e56, _e57)).z));
                    let _e2180 = ((_e850 + ((1f - _e850) * pow(_e1876, 5f))) * ((1f / (_e2134 * (((_e2117 * 3.1415927f) * _e2120) * _e2134))) * (0.5f / max(((_e47 * sqrt(max((_e2006 + ((_e2128 * (_e2150 * _e2150)) + (_e2131 * (_e2153 * _e2153)))), 0.000001f))) + (_e1837 * sqrt(max((_e632 + ((_e2128 * (_e2144 * _e2144)) + (_e2131 * (_e2147 * _e2147)))), 0.000001f)))), 0.000001f))));
                    coat_u002e_sroa_u002e_0_u002e_0_ = ((_e2180 * _e362) * _e339);
                    coat_u002e_sroa_u002e_5_u002e_0_ = ((_e2180 * _e368) * _e339);
                    coat_u002e_sroa_u002e_8_u002e_0_ = ((_e2180 * _e374) * _e339);
                } else {
                    coat_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    coat_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    coat_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2197 = coat_u002e_sroa_u002e_0_u002e_0_;
                let _e2199 = coat_u002e_sroa_u002e_5_u002e_0_;
                let _e2201 = coat_u002e_sroa_u002e_8_u002e_0_;
                if !((_e309 <= 0f)) {
                    let _e2225 = ((_e1847 * ((1f / ((_e864 * (((1f / _e47) + (1f / _e1837)) + -2f)) + 1f)) * (_e864 / (pow((((_e864 + -1f) * _e1980) + 1f), 2f) * 3.1415927f)))) / max(((_e47 * 4f) * _e1837), 0.000001f));
                    fuzz_u002e_sroa_u002e_0_u002e_0_ = ((_e2225 * bitcast<f32>(_e320)) * _e309);
                    fuzz_u002e_sroa_u002e_5_u002e_0_ = ((_e2225 * bitcast<f32>(_e326)) * _e309);
                    fuzz_u002e_sroa_u002e_8_u002e_0_ = ((_e2225 * bitcast<f32>(_e332)) * _e309);
                } else {
                    fuzz_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    fuzz_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    fuzz_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2242 = fuzz_u002e_sroa_u002e_0_u002e_0_;
                let _e2244 = fuzz_u002e_sroa_u002e_5_u002e_0_;
                let _e2246 = fuzz_u002e_sroa_u002e_8_u002e_0_;
                if !((_e195 <= 0f)) {
                    let _e2248 = (_e811 * sqrt(_e818));
                    if !((_e201 <= 0f)) {
                        extinction_u002e_sroa_u002e_0_u002e_0_ = (-(log(max(_e217, 0.000001f))) / _e201);
                        extinction_u002e_sroa_u002e_5_u002e_0_ = (-(log(max(_e223, 0.000001f))) / _e201);
                        extinction_u002e_sroa_u002e_8_u002e_0_ = (-(log(max(_e229, 0.000001f))) / _e201);
                    } else {
                        extinction_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        extinction_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        extinction_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e2268 = extinction_u002e_sroa_u002e_0_u002e_0_;
                    let _e2270 = extinction_u002e_sroa_u002e_5_u002e_0_;
                    let _e2272 = extinction_u002e_sroa_u002e_8_u002e_0_;
                    let _e2279 = (_e2248 * _e2248);
                    let _e2284 = ((_e1980 * (_e2279 + -1f)) + 1f);
                    let _e2290 = (1f - _e2279);
                    let _e2309 = ((1f - (_e810 + ((1f - _e810) * pow((1f - _e47), 5f)))) * ((_e2279 / (_e2284 * (_e2284 * 3.1415927f))) * (0.5f / max(((_e47 * sqrt(max((_e2279 + (_e2006 * _e2290)), 0.000001f))) + (_e1837 * sqrt(max((_e2279 + (_e632 * _e2290)), 0.000001f)))), 0.000001f))));
                    transmission_u002e_sroa_u002e_0_u002e_0_ = (((exp((_e201 * -(_e2268))) * _e2309) * _e217) * _e195);
                    transmission_u002e_sroa_u002e_5_u002e_0_ = (((exp((_e201 * -(_e2270))) * _e2309) * _e223) * _e195);
                    transmission_u002e_sroa_u002e_8_u002e_0_ = (((exp((_e201 * -(_e2272))) * _e2309) * _e229) * _e195);
                } else {
                    transmission_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    transmission_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    transmission_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2338 = transmission_u002e_sroa_u002e_0_u002e_0_;
                let _e2340 = transmission_u002e_sroa_u002e_5_u002e_0_;
                let _e2342 = transmission_u002e_sroa_u002e_8_u002e_0_;
                if !((_e255 <= 0f)) {
                    let _e2347 = length(vec3<f32>((_e824 - _e2052), (_e825 - _e2053), (_e826 - _e2054)));
                    let _e2349 = sqrt(3f);
                    let _e2372 = -(_e2347);
                    let _e2380 = (max(((clamp(((_e47 * _e1837) + (((_e636 + max((1f - _e2006), 0f)) - (_e2347 * _e2347)) * 0.5f)), -1f, 1f) * bitcast<f32>(_e272)) + 1f), 0f) * 0.31830987f);
                    subsurface_u002e_sroa_u002e_0_u002e_0_ = (((exp((((1f / max((_e261 * bitcast<f32>(_e266)), 0.000001f)) * _e2349) * _e2372)) * _e2380) * _e279) * _e255);
                    subsurface_u002e_sroa_u002e_5_u002e_0_ = (((exp((((1f / max((_e261 * bitcast<f32>(_e296)), 0.000001f)) * _e2349) * _e2372)) * _e2380) * _e285) * _e255);
                    subsurface_u002e_sroa_u002e_8_u002e_0_ = (((exp((((1f / max((_e261 * bitcast<f32>(_e302)), 0.000001f)) * _e2349) * _e2372)) * _e2380) * _e291) * _e255);
                } else {
                    subsurface_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    subsurface_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    subsurface_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2400 = subsurface_u002e_sroa_u002e_0_u002e_0_;
                let _e2402 = subsurface_u002e_sroa_u002e_5_u002e_0_;
                let _e2404 = subsurface_u002e_sroa_u002e_8_u002e_0_;
                lo_u002e_sroa_u002e_0_u002e_1_ = (_e920 + (_e1746 * (_e1837 * ((((_e787 * ((_e689 * ((((max(((_e1909 + (_e1916 * _e1940)) - ((_e1918 - (_e1918 * _e177)) * _e1938)), 0f) * _e111) * _e1988) * _e2026) * _e75)) + (_e689 * (((_e457 * _e829) * _e2096) * _e75)))) + _e2197) + (_e2242 + _e2338)) + _e2400))));
                lo_u002e_sroa_u002e_6_u002e_1_ = (_e918 + (_e1748 * (_e1837 * ((((_e789 * ((_e694 * ((((max(((_e1910 + (_e1920 * _e1940)) - ((_e1922 - (_e1922 * _e183)) * _e1938)), 0f) * _e111) * _e1988) * _e2026) * _e75)) + (_e694 * (((_e458 * _e829) * _e2096) * _e75)))) + _e2199) + (_e2244 + _e2340)) + _e2402))));
                lo_u002e_sroa_u002e_9_u002e_1_ = (_e916 + (_e1750 * (_e1837 * ((((_e791 * ((_e699 * ((((max(((_e1911 + (_e1924 * _e1940)) - ((_e1926 - (_e1926 * _e189)) * _e1938)), 0f) * _e111) * _e1988) * _e2026) * _e75)) + (_e699 * (((_e459 * _e829) * _e2096) * _e75)))) + _e2201) + (_e2246 + _e2342)) + _e2404))));
            } else {
                lo_u002e_sroa_u002e_0_u002e_1_ = _e920;
                lo_u002e_sroa_u002e_6_u002e_1_ = _e918;
                lo_u002e_sroa_u002e_9_u002e_1_ = _e916;
            }
            let _e2433 = lo_u002e_sroa_u002e_0_u002e_1_;
            out_u002e_lo_u002e_sroa_u002e_0_u002e_1_ = _e2433;
            let _e2436 = lo_u002e_sroa_u002e_6_u002e_1_;
            out_u002e_lo_u002e_sroa_u002e_6_u002e_1_ = _e2436;
            let _e2439 = lo_u002e_sroa_u002e_9_u002e_1_;
            out_u002e_lo_u002e_sroa_u002e_9_u002e_1_ = _e2439;
            let _e2442 = (_e914 + 1u);
            if (_e2442 == _e792) {
                lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e2433;
                lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e2436;
                lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e2439;
                break;
            } else {
                k_u002e_sroa_u002e_0_u002e_0_ = _e2442;
                lo_u002e_sroa_u002e_9_ = _e2439;
                lo_u002e_sroa_u002e_6_ = _e2436;
                lo_u002e_sroa_u002e_0_ = _e2433;
                continue;
            }
        }
    }
    let _e2458 = lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
    let _e2460 = lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
    let _e2462 = lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
    let _e2472 = uniform_struct_0_3_.f0_.x;
    let _e2476 = uniform_struct_0_3_.f0_.y;
    let _e2480 = uniform_struct_0_3_.f0_.z;
    if !((_e404 <= 0f)) {
        let _e2487 = ((_e453.x * bitcast<f32>(_e409)) * _e404);
        let _e2488 = ((_e453.y * bitcast<f32>(_e415)) * _e404);
        let _e2489 = ((_e453.z * bitcast<f32>(_e421)) * _e404);
        let _e2499 = ((1f - _e339) + ((1f - pow((1f - _e47), 5f)) * _e339));
        emission_u002e_sroa_u002e_0_u002e_1_ = (_e2487 + ((((_e2487 * _e362) * _e2499) - _e2487) * _e339));
        emission_u002e_sroa_u002e_7_u002e_1_ = (_e2488 + ((((_e2488 * _e368) * _e2499) - _e2488) * _e339));
        emission_u002e_sroa_u002e_10_u002e_1_ = (_e2489 + ((((_e2489 * _e374) * _e2499) - _e2489) * _e339));
    } else {
        emission_u002e_sroa_u002e_0_u002e_1_ = 0f;
        emission_u002e_sroa_u002e_7_u002e_1_ = 0f;
        emission_u002e_sroa_u002e_10_u002e_1_ = 0f;
    }
    let _e2525 = emission_u002e_sroa_u002e_0_u002e_1_;
    let _e2527 = emission_u002e_sroa_u002e_7_u002e_1_;
    let _e2529 = emission_u002e_sroa_u002e_10_u002e_1_;
    let _e2534 = ((_e165 + -1f) / (_e165 + 1f));
    let _e2535 = (_e2534 * _e2534);
    let _e2543 = (dot(vec3<f32>(_e16, _e17, _e18), vec3<f32>(_e38, _e39, _e40)) * 2f);
    let _e2552 = uniform_struct_0_3_.f4_;
    let _e2557 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(((_e16 * _e2543) - _e38), ((_e17 * _e2543) - _e39), ((_e18 * _e2543) - _e40)), (_e499 * _e2552));
    let _e2563 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e16, _e17, _e18));
    let _e2569 = textureSample(tex_0_9_, sampler_0_10_, vec2<f32>(_e47, _e499));
    let _e2573 = (1f - _e460);
    let _e2579 = uniform_struct_0_3_.f3_;
    let _e2592 = max((1f - _e2569.y), 0f);
    let _e2616 = !((bitcast<f32>(_e437) >= 0.5f));
    let _e2617 = select(_e2525, (((_e2458 + (mix(_e457, (_e457 * _e111), _e460) * _e2472)) + _e2525) + (_e2579 * ((_e2557.x * ((mix(_e2535, _e707, _e460) * _e2569.x) + _e2569.y)) + (((_e457 * _e2573) * _e2563.x) * _e2592)))), _e2616);
    let _e2618 = select(_e2527, (((_e2460 + (mix(_e458, (_e458 * _e111), _e460) * _e2476)) + _e2527) + (_e2579 * ((_e2557.y * ((mix(_e2535, _e708, _e460) * _e2569.x) + _e2569.y)) + (((_e458 * _e2573) * _e2563.y) * _e2592)))), _e2616);
    let _e2619 = select(_e2529, (((_e2462 + (mix(_e459, (_e459 * _e111), _e460) * _e2480)) + _e2529) + (_e2579 * ((_e2557.z * ((mix(_e2535, _e709, _e460) * _e2569.x) + _e2569.y)) + (((_e459 * _e2573) * _e2563.z) * _e2592)))), _e2616);
    if ((_e2617 <= _e2617) && (0f <= 0f)) {
        if (abs(_e2617) < 100000000000000000000f) {
            _2_u002e_sroa_u002e_0_u002e_0_ = _e2617;
        } else {
            _2_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
    } else {
        _2_u002e_sroa_u002e_0_u002e_0_ = 0f;
    }
    let _e2633 = _2_u002e_sroa_u002e_0_u002e_0_;
    if ((_e2618 <= _e2618) && (0f <= 0f)) {
        if (abs(_e2618) < 100000000000000000000f) {
            _3_u002e_sroa_u002e_0_u002e_0_ = _e2618;
        } else {
            _3_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
    } else {
        _3_u002e_sroa_u002e_0_u002e_0_ = 0f;
    }
    let _e2647 = _3_u002e_sroa_u002e_0_u002e_0_;
    if ((_e2619 <= _e2619) && (0f <= 0f)) {
        if (abs(_e2619) < 100000000000000000000f) {
            _4_u002e_sroa_u002e_0_u002e_0_ = _e2619;
            let _e2657 = _4_u002e_sroa_u002e_0_u002e_0_;
            fragment_out = vec4<f32>(_e2633, _e2647, _e2657, bitcast<f32>(_e427));
            let _e2662 = fragment_out;
            return _e2662;
        }
        _4_u002e_sroa_u002e_0_u002e_0_ = 0f;
    } else {
        _4_u002e_sroa_u002e_0_u002e_0_ = 0f;
    }
    let _e2668 = _4_u002e_sroa_u002e_0_u002e_0_;
    fragment_out = vec4<f32>(_e2633, _e2647, _e2668, bitcast<f32>(_e427));
    let _e2673 = fragment_out;
    return _e2673;
}
