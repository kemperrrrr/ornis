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
    var lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa: f32;
    var lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa: f32;
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i: u32;
    var radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_: f32;
    var radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_: f32;
    var vis_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
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
    let _e115 = storage_0_2_[(_e70 | 9u)];
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
    let _e159 = storage_0_2_[(_e70 | 10u)];
    let _e160 = bitcast<f32>(_e159);
    let _e165 = storage_0_2_[(_e70 | 11u)];
    let _e171 = storage_0_2_[(_e70 | 12u)];
    let _e172 = bitcast<f32>(_e171);
    let _e177 = storage_0_2_[(_e70 | 13u)];
    let _e178 = bitcast<f32>(_e177);
    let _e183 = storage_0_2_[(_e70 | 14u)];
    let _e184 = bitcast<f32>(_e183);
    let _e189 = storage_0_2_[(_e70 + 16u)];
    let _e190 = bitcast<f32>(_e189);
    let _e195 = storage_0_2_[(_e70 + 17u)];
    let _e196 = bitcast<f32>(_e195);
    let _e211 = storage_0_2_[(_e70 + 20u)];
    let _e212 = bitcast<f32>(_e211);
    let _e217 = storage_0_2_[(_e70 + 21u)];
    let _e218 = bitcast<f32>(_e217);
    let _e223 = storage_0_2_[(_e70 + 22u)];
    let _e224 = bitcast<f32>(_e223);
    let _e249 = storage_0_2_[(_e70 + 28u)];
    let _e250 = bitcast<f32>(_e249);
    let _e255 = storage_0_2_[(_e70 + 29u)];
    let _e256 = bitcast<f32>(_e255);
    let _e261 = storage_0_2_[(_e70 + 30u)];
    let _e267 = storage_0_2_[(_e70 + 31u)];
    let _e273 = storage_0_2_[(_e70 + 32u)];
    let _e274 = bitcast<f32>(_e273);
    let _e279 = storage_0_2_[(_e70 + 33u)];
    let _e280 = bitcast<f32>(_e279);
    let _e285 = storage_0_2_[(_e70 + 34u)];
    let _e286 = bitcast<f32>(_e285);
    let _e291 = storage_0_2_[(_e70 + 36u)];
    let _e297 = storage_0_2_[(_e70 + 37u)];
    let _e303 = storage_0_2_[(_e70 + 40u)];
    let _e304 = bitcast<f32>(_e303);
    let _e309 = storage_0_2_[(_e70 + 41u)];
    let _e310 = bitcast<f32>(_e309);
    let _e315 = storage_0_2_[(_e70 + 44u)];
    let _e321 = storage_0_2_[(_e70 + 45u)];
    let _e327 = storage_0_2_[(_e70 + 46u)];
    let _e333 = storage_0_2_[(_e70 + 48u)];
    let _e334 = bitcast<f32>(_e333);
    let _e339 = storage_0_2_[(_e70 + 49u)];
    let _e340 = bitcast<f32>(_e339);
    let _e345 = storage_0_2_[(_e70 + 50u)];
    let _e351 = storage_0_2_[(_e70 + 51u)];
    let _e356 = storage_0_2_[(_e70 + 52u)];
    let _e357 = bitcast<f32>(_e356);
    let _e362 = storage_0_2_[(_e70 + 53u)];
    let _e363 = bitcast<f32>(_e362);
    let _e368 = storage_0_2_[(_e70 + 54u)];
    let _e369 = bitcast<f32>(_e368);
    let _e374 = storage_0_2_[(_e70 + 56u)];
    let _e375 = bitcast<f32>(_e374);
    let _e380 = storage_0_2_[(_e70 + 60u)];
    let _e386 = storage_0_2_[(_e70 + 61u)];
    let _e392 = storage_0_2_[(_e70 + 62u)];
    let _e393 = bitcast<f32>(_e392);
    let _e398 = storage_0_2_[(_e70 + 64u)];
    let _e399 = bitcast<f32>(_e398);
    let _e404 = storage_0_2_[(_e70 + 68u)];
    let _e410 = storage_0_2_[(_e70 + 69u)];
    let _e416 = storage_0_2_[(_e70 + 70u)];
    let _e422 = storage_0_2_[(_e70 + 72u)];
    let _e432 = storage_0_2_[(_e70 + 74u)];
    let _e437 = uniform_struct_0_0_.f0_.x;
    let _e441 = uniform_struct_0_0_.f1_.x;
    let _e445 = uniform_struct_0_0_.f2_.x;
    let _e449 = uniform_struct_0_0_.f3_.x;
    let _e459 = uniform_struct_0_0_.f0_.y;
    let _e463 = uniform_struct_0_0_.f1_.y;
    let _e467 = uniform_struct_0_0_.f2_.y;
    let _e471 = uniform_struct_0_0_.f3_.y;
    let _e481 = uniform_struct_0_0_.f0_.w;
    let _e485 = uniform_struct_0_0_.f1_.w;
    let _e489 = uniform_struct_0_0_.f2_.w;
    let _e493 = uniform_struct_0_0_.f3_.w;
    let _e499 = (_e493 + (((in0_.x * _e481) + (in0_.y * _e485)) + (in0_.z * _e489)));
    let _e503 = select(0.5f, _e499, (_e499 > 0.5f));
    let _e544 = ((((min(u32(floor((((((_e471 + (((in0_.x * _e459) + (in0_.y * _e463)) + (in0_.z * _e467))) / _e503) * 0.5f) + 0.5f) * 9f))), 8u) << 4u) | min(u32(floor((((((_e449 + (((in0_.x * _e437) + (in0_.y * _e441)) + (in0_.z * _e445))) / _e503) * 0.5f) + 0.5f) * 16f))), 15u)) + (min(u32(ceil(((log((_e503 * 2f)) * 3.4407384f) + -0.0001f))), 23u) * 144u)) << 1u);
    let _e547 = uniform_struct_0_12_.f2_;
    let _e550 = uniform_struct_0_3_.f2_;
    let _e553 = storage_0_15_[_e544];
    let _e559 = storage_0_15_[(_e544 | 1u)];
    let _e563 = (_e46 * _e46);
    let _e567 = max((1f - _e563), 0f);
    let _e569 = (sqrt(_e567) / _e393);
    let _e582 = pow(((_e393 + -1f) / (_e393 + 1f)), 2f);
    let _e586 = (((bitcast<f32>(_e386) * 1000f) * (_e393 * 12.566371f)) * sqrt(max((1f - (_e569 * _e569)), 0f)));
    let _e597 = (_e582 * 2f);
    let _e603 = (1f - (_e582 * _e582));
    let _e615 = clamp(bitcast<f32>(_e380), 0f, 1f);
    let _e620 = ((_e615 * ((((_e597 * cos((_e586 / 650f))) / _e603) + 1f) + -1f)) + 1f);
    let _e625 = ((_e615 * ((((_e597 * cos((_e586 / 550f))) / _e603) + 1f) + -1f)) + 1f);
    let _e630 = ((_e615 * ((((_e597 * cos((_e586 / 450f))) / _e603) + 1f) + -1f)) + 1f);
    let _e635 = ((_e375 + -1f) / (_e375 + 1f));
    let _e638 = (_e74 * _e92);
    let _e639 = (_e74 * _e98);
    let _e640 = (_e74 * _e104);
    let _e643 = !((_e334 <= 0f));
    if _e643 {
        let _e649 = (1f - ((1f - pow(_e635, 2f)) / (_e375 * _e375)));
        let _e651 = (1f - _e649);
        let _e652 = (_e640 * _e110);
        let _e665 = (_e639 * _e110);
        let _e678 = (_e638 * _e110);
        let _e692 = (_e334 * bitcast<f32>(_e351));
        coat_darken_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e692 * ((_e651 / max((1f - ((_e678 + (((((_e92 - _e274) * _e250) + _e274) - _e678) * _e86)) * _e649)), 0.000001f)) + -1f)) + 1f);
        coat_darken_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e692 * ((_e651 / max((1f - ((_e665 + (((((_e98 - _e280) * _e250) + _e280) - _e665) * _e86)) * _e649)), 0.000001f)) + -1f)) + 1f);
        coat_darken_u002e_i_u002e_sroa_u002e_7_u002e_0_ = ((_e692 * ((_e651 / max((1f - ((_e652 + (((((_e104 - _e286) * _e250) + _e286) - _e652) * _e86)) * _e649)), 0.000001f)) + -1f)) + 1f);
    } else {
        coat_darken_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 1f;
        coat_darken_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 1f;
        coat_darken_u002e_i_u002e_sroa_u002e_7_u002e_0_ = 1f;
    }
    let _e718 = coat_darken_u002e_i_u002e_sroa_u002e_0_u002e_0_;
    let _e720 = coat_darken_u002e_i_u002e_sroa_u002e_5_u002e_0_;
    let _e722 = coat_darken_u002e_i_u002e_sroa_u002e_7_u002e_0_;
    let _e723 = ((_e559 * _e547) + _e550);
    if (_e723 == 0u) {
        lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
        lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = 0f;
        lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = 0f;
    } else {
        let _e740 = ((_e160 + -1f) / (_e160 + 1f));
        let _e741 = (_e740 * _e740);
        let _e742 = (_e154 * _e154);
        let _e744 = (1f - bitcast<f32>(_e165));
        let _e749 = (2f / ((_e744 * _e744) + 1f));
        let _e751 = select(_e154, _e80, (_e154 < _e80));
        let _e755 = (_e37 - (_e15 * _e46));
        let _e756 = (_e38 - (_e16 * _e46));
        let _e757 = (_e39 - (_e17 * _e46));
        let _e760 = (1f - _e86);
        let _e765 = max((_e751 * _e751), 0.000001f);
        let _e766 = (_e765 * _e765);
        let _e781 = (_e635 * _e635);
        let _e784 = (_e340 * _e340);
        let _e786 = (1f - bitcast<f32>(_e345));
        let _e791 = (2f / ((_e786 * _e786) + 1f));
        let _e795 = (_e310 * _e310);
        k_u002e_sroa_u002e_0_u002e_0_ = 0u;
        lo_u002e_i_u002e_sroa_u002e_9_ = 0f;
        lo_u002e_i_u002e_sroa_u002e_6_ = 0f;
        lo_u002e_i_u002e_sroa_u002e_0_ = 0f;
        loop {
            let _e845 = k_u002e_sroa_u002e_0_u002e_0_;
            let _e847 = lo_u002e_i_u002e_sroa_u002e_9_;
            let _e849 = lo_u002e_i_u002e_sroa_u002e_6_;
            let _e851 = lo_u002e_i_u002e_sroa_u002e_0_;
            if (_e845 < _e550) {
                _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i = _e845;
            } else {
                let _e857 = storage_0_14_[(((_e553 * _e547) - _e550) + _e845)];
                _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i = _e857;
            }
            let _e860 = _0_u002e_sroa_u002e_0_u002e_0_u002e_in_u002e_i;
            let _e862 = (_e860 * 36u);
            let _e865 = storage_0_13_[_e862];
            let _e866 = bitcast<f32>(_e865);
            let _e868 = (_e862 + 8u);
            let _e871 = storage_0_13_[_e868];
            let _e874 = (_e862 + 9u);
            let _e877 = storage_0_13_[_e874];
            let _e880 = (_e862 + 10u);
            let _e883 = storage_0_13_[_e880];
            let _e889 = length(vec3<f32>((bitcast<f32>(_e871) - in0_.x), (bitcast<f32>(_e877) - in0_.y), (bitcast<f32>(_e883) - in0_.z)));
            let _e894 = select(1f, 0f, (_e866 < 0.5f));
            let _e896 = (_e862 + 16u);
            let _e899 = storage_0_13_[_e896];
            let _e900 = bitcast<f32>(_e899);
            let _e912 = storage_0_13_[(_e862 + 12u)];
            let _e918 = storage_0_13_[(_e862 + 13u)];
            let _e924 = storage_0_13_[(_e862 + 14u)];
            let _e930 = storage_0_13_[(_e862 + 15u)];
            let _e931 = bitcast<f32>(_e930);
            let _e934 = storage_0_13_[_e868];
            let _e938 = storage_0_13_[_e874];
            let _e942 = storage_0_13_[_e880];
            let _e944 = (in0_.x - bitcast<f32>(_e934));
            let _e945 = (in0_.y - bitcast<f32>(_e938));
            let _e946 = (in0_.z - bitcast<f32>(_e942));
            let _e950 = (1f / length(vec3<f32>(_e944, _e945, _e946)));
            let _e955 = (_e862 + 4u);
            let _e958 = storage_0_13_[_e955];
            let _e959 = bitcast<f32>(_e958);
            let _e961 = (_e862 + 5u);
            let _e964 = storage_0_13_[_e961];
            let _e965 = bitcast<f32>(_e964);
            let _e967 = (_e862 + 6u);
            let _e970 = storage_0_13_[_e967];
            let _e971 = bitcast<f32>(_e970);
            let _e975 = (1f / length(vec3<f32>(_e959, _e965, _e971)));
            let _e986 = storage_0_13_[(_e862 + 18u)];
            let _e992 = storage_0_13_[(_e862 + 17u)];
            let _e999 = select(1f, 0f, (_e866 < 1.5f));
            let _e1001 = mix(1f, smoothstep(bitcast<f32>(_e986), bitcast<f32>(_e992), dot(vec3<f32>((_e975 * _e959), (_e975 * _e965), (_e975 * _e971)), vec3<f32>((_e944 * _e950), (_e945 * _e950), (_e946 * _e950)))), _e999);
            let _e1002 = (_e889 * _e889);
            let _e1014 = (mix(1f, (1f - smoothstep((_e900 * 0.99f), _e900, _e889)), _e894) * mix(1f, (1f / select(_e1002, 0.000001f, (_e1002 < 0.000001f))), _e894));
            let _e1018 = (_e1001 * ((bitcast<f32>(_e912) * _e931) * _e1014));
            let _e1019 = (_e1001 * ((bitcast<f32>(_e918) * _e931) * _e1014));
            let _e1020 = (_e1001 * ((bitcast<f32>(_e924) * _e931) * _e1014));
            let _e1023 = (_e894 * (1f - _e999));
            if (((1f - _e894) * select(1f, 0f, (_e900 < 0.5f))) > 0.5f) {
                let _e1038 = uniform_struct_0_3_.f6_.f1_.x;
                let _e1043 = uniform_struct_0_3_.f6_.f1_.y;
                let _e1048 = uniform_struct_0_3_.f6_.f1_.z;
                let _e1051 = dot(vec3<f32>((in0_.x - _e21), (in0_.y - _e25), (in0_.z - _e29)), vec3<f32>(_e1038, _e1043, _e1048));
                let _e1056 = uniform_struct_0_3_.f6_.f2_.x;
                let _e1061 = uniform_struct_0_3_.f6_.f2_.y;
                let _e1066 = uniform_struct_0_3_.f6_.f2_.z;
                let _e1071 = uniform_struct_0_3_.f6_.f2_.w;
                let _e1075 = uniform_struct_0_3_.f6_.f3_;
                let _e1079 = select(1f, 0f, (_e1051 < _e1061));
                let _e1083 = select(1f, 0f, (_e1051 < _e1066));
                let _e1087 = select(1f, 0f, (_e1051 < _e1071));
                let _e1090 = u32(((_e1079 + _e1083) + _e1087));
                let _e1093 = mix(mix(mix(_e1061, _e1066, _e1079), _e1071, _e1083), _e1075, _e1087);
                let _e1103 = uniform_struct_0_3_.f6_.f0_[_e1090].f0_.x;
                let _e1110 = uniform_struct_0_3_.f6_.f0_[_e1090].f1_.x;
                let _e1117 = uniform_struct_0_3_.f6_.f0_[_e1090].f2_.x;
                let _e1124 = uniform_struct_0_3_.f6_.f0_[_e1090].f3_.x;
                let _e1137 = uniform_struct_0_3_.f6_.f0_[_e1090].f0_.y;
                let _e1144 = uniform_struct_0_3_.f6_.f0_[_e1090].f1_.y;
                let _e1151 = uniform_struct_0_3_.f6_.f0_[_e1090].f2_.y;
                let _e1158 = uniform_struct_0_3_.f6_.f0_[_e1090].f3_.y;
                let _e1171 = uniform_struct_0_3_.f6_.f0_[_e1090].f0_.z;
                let _e1178 = uniform_struct_0_3_.f6_.f0_[_e1090].f1_.z;
                let _e1185 = uniform_struct_0_3_.f6_.f0_[_e1090].f2_.z;
                let _e1192 = uniform_struct_0_3_.f6_.f0_[_e1090].f3_.z;
                let _e1205 = uniform_struct_0_3_.f6_.f0_[_e1090].f0_.w;
                let _e1212 = uniform_struct_0_3_.f6_.f0_[_e1090].f1_.w;
                let _e1219 = uniform_struct_0_3_.f6_.f0_[_e1090].f2_.w;
                let _e1226 = uniform_struct_0_3_.f6_.f0_[_e1090].f3_.w;
                let _e1232 = (_e1226 + (((in0_.x * _e1205) + (in0_.y * _e1212)) + (in0_.z * _e1219)));
                let _e1249 = textureSampleCompareLevel(tex_0_11_, sampler_0_5_, vec2<f32>(((((_e1124 + (((in0_.x * _e1103) + (in0_.y * _e1110)) + (in0_.z * _e1117))) / _e1232) * 0.5f) + 0.5f), (0.5f - (((_e1158 + (((in0_.x * _e1137) + (in0_.y * _e1144)) + (in0_.z * _e1151))) / _e1232) * 0.5f))), _e1090, (((_e1192 + (((in0_.x * _e1171) + (in0_.y * _e1178)) + (in0_.z * _e1185))) / _e1232) + -0.001f));
                if (_e1090 < 3u) {
                    let _e1254 = ((_e1093 - mix(mix(mix(_e1056, _e1061, _e1079), _e1066, _e1083), _e1071, _e1087)) * 0.08f);
                    let _e1260 = ((_e1093 - _e1051) / select(_e1254, 0.000001f, (_e1254 < 0.000001f)));
                    let _e1264 = select(_e1260, 0f, (_e1260 < 0f));
                    let _e1270 = (_e1090 + 1u);
                    let _e1277 = uniform_struct_0_3_.f6_.f0_[_e1270].f0_.x;
                    let _e1284 = uniform_struct_0_3_.f6_.f0_[_e1270].f1_.x;
                    let _e1291 = uniform_struct_0_3_.f6_.f0_[_e1270].f2_.x;
                    let _e1298 = uniform_struct_0_3_.f6_.f0_[_e1270].f3_.x;
                    let _e1311 = uniform_struct_0_3_.f6_.f0_[_e1270].f0_.y;
                    let _e1318 = uniform_struct_0_3_.f6_.f0_[_e1270].f1_.y;
                    let _e1325 = uniform_struct_0_3_.f6_.f0_[_e1270].f2_.y;
                    let _e1332 = uniform_struct_0_3_.f6_.f0_[_e1270].f3_.y;
                    let _e1345 = uniform_struct_0_3_.f6_.f0_[_e1270].f0_.z;
                    let _e1352 = uniform_struct_0_3_.f6_.f0_[_e1270].f1_.z;
                    let _e1359 = uniform_struct_0_3_.f6_.f0_[_e1270].f2_.z;
                    let _e1366 = uniform_struct_0_3_.f6_.f0_[_e1270].f3_.z;
                    let _e1379 = uniform_struct_0_3_.f6_.f0_[_e1270].f0_.w;
                    let _e1386 = uniform_struct_0_3_.f6_.f0_[_e1270].f1_.w;
                    let _e1393 = uniform_struct_0_3_.f6_.f0_[_e1270].f2_.w;
                    let _e1400 = uniform_struct_0_3_.f6_.f0_[_e1270].f3_.w;
                    let _e1406 = (_e1400 + (((in0_.x * _e1379) + (in0_.y * _e1386)) + (in0_.z * _e1393)));
                    let _e1423 = textureSampleCompareLevel(tex_0_11_, sampler_0_5_, vec2<f32>(((((_e1298 + (((in0_.x * _e1277) + (in0_.y * _e1284)) + (in0_.z * _e1291))) / _e1406) * 0.5f) + 0.5f), (0.5f - (((_e1332 + (((in0_.x * _e1311) + (in0_.y * _e1318)) + (in0_.z * _e1325))) / _e1406) * 0.5f))), _e1270, (((_e1366 + (((in0_.x * _e1345) + (in0_.y * _e1352)) + (in0_.z * _e1359))) / _e1406) + -0.001f));
                    vis_u002e_sroa_u002e_0_u002e_0_u002e_i = mix(_e1423, _e1249, select(_e1264, 1f, (_e1264 > 1f)));
                } else {
                    vis_u002e_sroa_u002e_0_u002e_0_u002e_i = _e1249;
                }
                let _e1428 = vis_u002e_sroa_u002e_0_u002e_0_u002e_i;
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (_e1018 * _e1428);
                radiance_u002e_i_u002e_sroa_u002e_10_u002e_0_ = (_e1019 * _e1428);
                radiance_u002e_i_u002e_sroa_u002e_15_u002e_0_ = (_e1020 * _e1428);
            } else {
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_ = _e1018;
                radiance_u002e_i_u002e_sroa_u002e_10_u002e_0_ = _e1019;
                radiance_u002e_i_u002e_sroa_u002e_15_u002e_0_ = _e1020;
            }
            let _e1439 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_0_;
            let _e1441 = radiance_u002e_i_u002e_sroa_u002e_10_u002e_0_;
            let _e1443 = radiance_u002e_i_u002e_sroa_u002e_15_u002e_0_;
            let _e1445 = (_e862 + 19u);
            let _e1448 = storage_0_13_[_e1445];
            if !((bitcast<f32>(_e1448) >= 0f)) {
                radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = _e1439;
                radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_ = _e1441;
                radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_ = _e1443;
            } else {
                if !((_e1023 <= 0.5f)) {
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e1439;
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_1_ = _e1441;
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_1_ = _e1443;
                } else {
                    let _e1466 = storage_0_13_[(_e862 + 20u)];
                    let _e1472 = storage_0_13_[(_e862 + 24u)];
                    let _e1478 = storage_0_13_[(_e862 + 28u)];
                    let _e1484 = storage_0_13_[(_e862 + 32u)];
                    let _e1496 = storage_0_13_[(_e862 + 21u)];
                    let _e1502 = storage_0_13_[(_e862 + 25u)];
                    let _e1508 = storage_0_13_[(_e862 + 29u)];
                    let _e1514 = storage_0_13_[(_e862 + 33u)];
                    let _e1526 = storage_0_13_[(_e862 + 22u)];
                    let _e1532 = storage_0_13_[(_e862 + 26u)];
                    let _e1538 = storage_0_13_[(_e862 + 30u)];
                    let _e1544 = storage_0_13_[(_e862 + 34u)];
                    let _e1556 = storage_0_13_[(_e862 + 23u)];
                    let _e1562 = storage_0_13_[(_e862 + 27u)];
                    let _e1568 = storage_0_13_[(_e862 + 31u)];
                    let _e1574 = storage_0_13_[(_e862 + 35u)];
                    let _e1581 = ((((in0_.x * bitcast<f32>(_e1556)) + (in0_.y * bitcast<f32>(_e1562))) + (in0_.z * bitcast<f32>(_e1568))) + bitcast<f32>(_e1574));
                    let _e1594 = storage_0_13_[_e1445];
                    let _e1603 = textureSampleCompareLevel(tex_0_4_, sampler_0_5_, vec2<f32>((((((((in0_.x * bitcast<f32>(_e1466)) + (in0_.y * bitcast<f32>(_e1472))) + (in0_.z * bitcast<f32>(_e1478))) + bitcast<f32>(_e1484)) / _e1581) * 0.5f) + 0.5f), (0.5f - ((((((in0_.x * bitcast<f32>(_e1496)) + (in0_.y * bitcast<f32>(_e1502))) + (in0_.z * bitcast<f32>(_e1508))) + bitcast<f32>(_e1514)) / _e1581) * 0.5f))), u32(bitcast<f32>(_e1594)), ((((((in0_.x * bitcast<f32>(_e1526)) + (in0_.y * bitcast<f32>(_e1532))) + (in0_.z * bitcast<f32>(_e1538))) + bitcast<f32>(_e1544)) / _e1581) + -0.001f));
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_ = (_e1439 * _e1603);
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_1_ = (_e1441 * _e1603);
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_1_ = (_e1443 * _e1603);
                }
                let _e1611 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_1_;
                let _e1613 = radiance_u002e_i_u002e_sroa_u002e_10_u002e_1_;
                let _e1615 = radiance_u002e_i_u002e_sroa_u002e_15_u002e_1_;
                if (_e1023 > 0.5f) {
                    let _e1620 = storage_0_13_[_e868];
                    let _e1624 = storage_0_13_[_e874];
                    let _e1628 = storage_0_13_[_e880];
                    let _e1630 = (in0_.x - bitcast<f32>(_e1620));
                    let _e1631 = (in0_.y - bitcast<f32>(_e1624));
                    let _e1632 = (in0_.z - bitcast<f32>(_e1628));
                    let _e1633 = abs(_e1630);
                    let _e1634 = abs(_e1631);
                    let _e1636 = select(_e1633, _e1634, (_e1634 > _e1633));
                    let _e1637 = abs(_e1632);
                    let _e1642 = storage_0_13_[_e896];
                    let _e1643 = bitcast<f32>(_e1642);
                    let _e1647 = select(_e1643, 1f, (_e1643 < 1f));
                    let _e1649 = (_e1647 + -0.1f);
                    let _e1660 = storage_0_13_[_e1445];
                    let _e1666 = textureSampleCompareLevel(tex_0_6_, sampler_0_5_, vec3<f32>(_e1630, _e1631, _e1632), u32(bitcast<f32>(_e1660)), (((_e1647 / _e1649) - ((_e1647 * 0.1f) / (select(_e1636, _e1637, (_e1637 > _e1636)) * _e1649))) + -0.001f));
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = (_e1611 * _e1666);
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_ = (_e1613 * _e1666);
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_ = (_e1615 * _e1666);
                } else {
                    radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_ = _e1611;
                    radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_ = _e1613;
                    radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_ = _e1615;
                }
            }
            let _e1677 = radiance_u002e_i_u002e_sroa_u002e_0_u002e_2_;
            let _e1679 = radiance_u002e_i_u002e_sroa_u002e_10_u002e_2_;
            let _e1681 = radiance_u002e_i_u002e_sroa_u002e_15_u002e_2_;
            let _e1684 = storage_0_13_[_e862];
            let _e1688 = storage_0_13_[_e868];
            let _e1692 = storage_0_13_[_e874];
            let _e1696 = storage_0_13_[_e880];
            let _e1698 = (bitcast<f32>(_e1688) - in0_.x);
            let _e1699 = (bitcast<f32>(_e1692) - in0_.y);
            let _e1700 = (bitcast<f32>(_e1696) - in0_.z);
            let _e1702 = length(vec3<f32>(_e1698, _e1699, _e1700));
            let _e1708 = (1f / select(_e1702, 0.000001f, (_e1702 < 0.000001f)));
            let _e1716 = select(1f, 0f, (bitcast<f32>(_e1684) < 0.5f));
            let _e1719 = storage_0_13_[_e955];
            let _e1720 = bitcast<f32>(_e1719);
            let _e1723 = storage_0_13_[_e961];
            let _e1724 = bitcast<f32>(_e1723);
            let _e1727 = storage_0_13_[_e967];
            let _e1728 = bitcast<f32>(_e1727);
            let _e1732 = (1f / length(vec3<f32>(_e1720, _e1724, _e1728)));
            let _e1736 = mix((_e1732 * _e1720), (_e1698 * _e1708), _e1716);
            let _e1737 = mix((_e1732 * _e1724), (_e1699 * _e1708), _e1716);
            let _e1738 = mix((_e1732 * _e1728), (_e1700 * _e1708), _e1716);
            let _e1742 = (1f / length(vec3<f32>(_e1736, _e1737, _e1738)));
            let _e1743 = (_e1736 * _e1742);
            let _e1744 = (_e1737 * _e1742);
            let _e1745 = (_e1738 * _e1742);
            let _e1750 = ((_e37 + _e1743) + 0f);
            let _e1752 = ((_e38 + _e1744) + 0f);
            let _e1754 = ((_e39 + _e1745) + 0.000001f);
            let _e1758 = (1f / length(vec3<f32>(_e1750, _e1752, _e1754)));
            let _e1759 = (_e1758 * _e1750);
            let _e1760 = (_e1758 * _e1752);
            let _e1761 = (_e1758 * _e1754);
            let _e1764 = dot(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e1743, _e1744, _e1745));
            let _e1768 = select(_e1764, 0.000001f, (_e1764 < 0.000001f));
            let _e1771 = dot(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e1759, _e1760, _e1761));
            let _e1774 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(_e1759, _e1760, _e1761));
            let _e1778 = select(_e1774, 0.000001f, (_e1774 < 0.000001f));
            if (_e1768 > 0.000001f) {
                let _e1784 = select(_e1771, 0.000001f, (_e1771 < 0.000001f));
                let _e1785 = mix(_e741, _e638, _e86);
                let _e1786 = mix(_e741, _e639, _e86);
                let _e1787 = mix(_e741, _e640, _e86);
                let _e1790 = pow(0.85714287f, 5f);
                let _e1792 = (1f - _e1785);
                let _e1794 = (_e1785 + (_e1792 * _e1790));
                let _e1796 = (1f - _e1786);
                let _e1798 = (_e1786 + (_e1796 * _e1790));
                let _e1800 = (1f - _e1787);
                let _e1802 = (_e1787 + (_e1800 * _e1790));
                let _e1807 = (1f - _e1778);
                let _e1816 = ((_e1778 * pow(_e1807, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e1818 = pow(_e1807, 5f);
                let _e1840 = mix(_e741, _e638, _e86);
                let _e1841 = mix(_e741, _e639, _e86);
                let _e1842 = mix(_e741, _e640, _e86);
                let _e1845 = pow(0.85714287f, 5f);
                let _e1847 = (1f - _e1840);
                let _e1849 = (_e1840 + (_e1847 * _e1845));
                let _e1851 = (1f - _e1841);
                let _e1853 = (_e1841 + (_e1851 * _e1845));
                let _e1855 = (1f - _e1842);
                let _e1857 = (_e1842 + (_e1855 * _e1845));
                let _e1869 = ((_e1778 * pow(_e1807, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                let _e1871 = pow(_e1807, 5f);
                let _e1894 = (_e742 * sqrt(_e749));
                let _e1897 = (_e744 * (_e742 * sqrt(_e749)));
                let _e1900 = dot(vec3<f32>(_e1759, _e1760, _e1761), vec3<f32>(_e54, _e55, _e56));
                let _e1903 = dot(vec3<f32>(_e1759, _e1760, _e1761), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                let _e1905 = (_e1894 * _e1894);
                let _e1908 = (_e1897 * _e1897);
                let _e1911 = (_e1784 * _e1784);
                let _e1912 = (_e1911 + (((_e1900 * _e1900) / _e1905) + ((_e1903 * _e1903) / _e1908)));
                let _e1919 = (1f / (_e1912 * (((_e1894 * 3.1415927f) * _e1897) * _e1912)));
                let _e1922 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(_e54, _e55, _e56));
                let _e1925 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                let _e1928 = dot(vec3<f32>(_e1743, _e1744, _e1745), vec3<f32>(_e54, _e55, _e56));
                let _e1931 = dot(vec3<f32>(_e1743, _e1744, _e1745), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                let _e1937 = (_e1768 * _e1768);
                let _e1957 = (0.5f / max(((_e46 * sqrt(max((_e1937 + ((_e1905 * (_e1928 * _e1928)) + (_e1908 * (_e1931 * _e1931)))), 0.000001f))) + (_e1768 * sqrt(max((_e563 + ((_e1905 * (_e1922 * _e1922)) + (_e1908 * (_e1925 * _e1925)))), 0.000001f)))), 0.000001f));
                let _e1976 = (1f / length(vec3<f32>(_e755, _e756, _e757)));
                let _e1983 = (_e1743 - (_e15 * _e1768));
                let _e1984 = (_e1744 - (_e16 * _e1768));
                let _e1985 = (_e1745 - (_e17 * _e1768));
                let _e1989 = (1f / length(vec3<f32>(_e1983, _e1984, _e1985)));
                let _e1995 = dot(vec3<f32>((_e755 * _e1976), (_e756 * _e1976), (_e757 * _e1976)), vec3<f32>((_e1983 * _e1989), (_e1984 * _e1989), (_e1985 * _e1989)));
                let _e2000 = acos(max(_e46, 0f));
                let _e2003 = acos(max(_e1768, 0f));
                let _e2027 = (max((1f - dot(vec3<f32>((max(((_e1785 + (_e1792 * _e1818)) - ((_e1794 - (_e1794 * _e172)) * _e1816)), 0f) * _e110), (max(((_e1786 + (_e1796 * _e1818)) - ((_e1798 - (_e1798 * _e178)) * _e1816)), 0f) * _e110), (max(((_e1787 + (_e1800 * _e1818)) - ((_e1802 - (_e1802 * _e184)) * _e1816)), 0f) * _e110)), vec3<f32>(0.2126f, 0.7152f, 0.0722f))), 0f) * (((1f - ((_e766 * 0.5f) / (_e766 + 0.57f))) + (tan(min(_e2000, _e2003)) * ((((_e766 * 0.45f) / (_e766 + 0.09f)) * select(0f, _e1995, (_e1995 > 0f))) * sin(max(_e2000, _e2003))))) * 0.31830987f));
                if _e643 {
                    let _e2048 = (_e784 * sqrt(_e791));
                    let _e2051 = (_e786 * (_e784 * sqrt(_e791)));
                    let _e2054 = dot(vec3<f32>(_e1759, _e1760, _e1761), vec3<f32>(_e54, _e55, _e56));
                    let _e2057 = dot(vec3<f32>(_e1759, _e1760, _e1761), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                    let _e2059 = (_e2048 * _e2048);
                    let _e2062 = (_e2051 * _e2051);
                    let _e2065 = (_e1911 + (((_e2054 * _e2054) / _e2059) + ((_e2057 * _e2057) / _e2062)));
                    let _e2075 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(_e54, _e55, _e56));
                    let _e2078 = dot(vec3<f32>(_e37, _e38, _e39), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                    let _e2081 = dot(vec3<f32>(_e1743, _e1744, _e1745), vec3<f32>(_e54, _e55, _e56));
                    let _e2084 = dot(vec3<f32>(_e1743, _e1744, _e1745), vec3<f32>(cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).x, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).y, cross(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e54, _e55, _e56)).z));
                    let _e2111 = ((_e781 + ((1f - _e781) * pow(_e1807, 5f))) * ((1f / (_e2065 * (((_e2048 * 3.1415927f) * _e2051) * _e2065))) * (0.5f / max(((_e46 * sqrt(max((_e1937 + ((_e2059 * (_e2081 * _e2081)) + (_e2062 * (_e2084 * _e2084)))), 0.000001f))) + (_e1768 * sqrt(max((_e563 + ((_e2059 * (_e2075 * _e2075)) + (_e2062 * (_e2078 * _e2078)))), 0.000001f)))), 0.000001f))));
                    coat_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e2111 * _e357) * _e334);
                    coat_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e2111 * _e363) * _e334);
                    coat_u002e_i_u002e_sroa_u002e_8_u002e_0_ = ((_e2111 * _e369) * _e334);
                } else {
                    coat_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    coat_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    coat_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2128 = coat_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2130 = coat_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2132 = coat_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e304 <= 0f)) {
                    let _e2156 = ((_e1778 * ((1f / ((_e795 * (((1f / _e46) + (1f / _e1768)) + -2f)) + 1f)) * (_e795 / (pow((((_e795 + -1f) * _e1911) + 1f), 2f) * 3.1415927f)))) / max(((_e46 * 4f) * _e1768), 0.000001f));
                    fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_ = ((_e2156 * bitcast<f32>(_e315)) * _e304);
                    fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_ = ((_e2156 * bitcast<f32>(_e321)) * _e304);
                    fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_ = ((_e2156 * bitcast<f32>(_e327)) * _e304);
                } else {
                    fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2173 = fuzz_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2175 = fuzz_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2177 = fuzz_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e190 <= 0f)) {
                    let _e2179 = (_e742 * sqrt(_e749));
                    if !((_e196 <= 0f)) {
                        extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (-(log(max(_e212, 0.000001f))) / _e196);
                        extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (-(log(max(_e218, 0.000001f))) / _e196);
                        extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (-(log(max(_e224, 0.000001f))) / _e196);
                    } else {
                        extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e2199 = extinction_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                    let _e2201 = extinction_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                    let _e2203 = extinction_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                    let _e2210 = (_e2179 * _e2179);
                    let _e2215 = ((_e1911 * (_e2210 + -1f)) + 1f);
                    let _e2221 = (1f - _e2210);
                    let _e2240 = ((1f - (_e741 + ((1f - _e741) * pow((1f - _e46), 5f)))) * ((_e2210 / (_e2215 * (_e2215 * 3.1415927f))) * (0.5f / max(((_e46 * sqrt(max((_e2210 + (_e1937 * _e2221)), 0.000001f))) + (_e1768 * sqrt(max((_e2210 + (_e563 * _e2221)), 0.000001f)))), 0.000001f))));
                    transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (((exp((_e196 * -(_e2199))) * _e2240) * _e212) * _e190);
                    transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (((exp((_e196 * -(_e2201))) * _e2240) * _e218) * _e190);
                    transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (((exp((_e196 * -(_e2203))) * _e2240) * _e224) * _e190);
                } else {
                    transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2269 = transmission_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2271 = transmission_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2273 = transmission_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                if !((_e250 <= 0f)) {
                    let _e2278 = length(vec3<f32>((_e755 - _e1983), (_e756 - _e1984), (_e757 - _e1985)));
                    let _e2280 = sqrt(3f);
                    let _e2303 = -(_e2278);
                    let _e2311 = (max(((clamp(((_e46 * _e1768) + (((_e567 + max((1f - _e1937), 0f)) - (_e2278 * _e2278)) * 0.5f)), -1f, 1f) * bitcast<f32>(_e267)) + 1f), 0f) * 0.31830987f);
                    subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (((exp((((1f / max((_e256 * bitcast<f32>(_e261)), 0.000001f)) * _e2280) * _e2303)) * _e2311) * _e274) * _e250);
                    subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (((exp((((1f / max((_e256 * bitcast<f32>(_e291)), 0.000001f)) * _e2280) * _e2303)) * _e2311) * _e280) * _e250);
                    subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_ = (((exp((((1f / max((_e256 * bitcast<f32>(_e297)), 0.000001f)) * _e2280) * _e2303)) * _e2311) * _e286) * _e250);
                } else {
                    subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_ = 0f;
                    subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_ = 0f;
                }
                let _e2331 = subsurface_u002e_i_u002e_sroa_u002e_0_u002e_0_;
                let _e2333 = subsurface_u002e_i_u002e_sroa_u002e_5_u002e_0_;
                let _e2335 = subsurface_u002e_i_u002e_sroa_u002e_8_u002e_0_;
                lo_u002e_i_u002e_sroa_u002e_0_u002e_1_ = (_e851 + (_e1677 * (_e1768 * ((((_e718 * ((_e620 * ((((max(((_e1840 + (_e1847 * _e1871)) - ((_e1849 - (_e1849 * _e172)) * _e1869)), 0f) * _e110) * _e1919) * _e1957) * _e74)) + (_e620 * (((_e760 * _e92) * _e2027) * _e74)))) + _e2128) + (_e2173 + _e2269)) + _e2331))));
                lo_u002e_i_u002e_sroa_u002e_6_u002e_1_ = (_e849 + (_e1679 * (_e1768 * ((((_e720 * ((_e625 * ((((max(((_e1841 + (_e1851 * _e1871)) - ((_e1853 - (_e1853 * _e178)) * _e1869)), 0f) * _e110) * _e1919) * _e1957) * _e74)) + (_e625 * (((_e760 * _e98) * _e2027) * _e74)))) + _e2130) + (_e2175 + _e2271)) + _e2333))));
                lo_u002e_i_u002e_sroa_u002e_9_u002e_1_ = (_e847 + (_e1681 * (_e1768 * ((((_e722 * ((_e630 * ((((max(((_e1842 + (_e1855 * _e1871)) - ((_e1857 - (_e1857 * _e184)) * _e1869)), 0f) * _e110) * _e1919) * _e1957) * _e74)) + (_e630 * (((_e760 * _e104) * _e2027) * _e74)))) + _e2132) + (_e2177 + _e2273)) + _e2335))));
            } else {
                lo_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e851;
                lo_u002e_i_u002e_sroa_u002e_6_u002e_1_ = _e849;
                lo_u002e_i_u002e_sroa_u002e_9_u002e_1_ = _e847;
            }
            let _e2364 = lo_u002e_i_u002e_sroa_u002e_0_u002e_1_;
            out_u002e_lo_u002e_i_u002e_sroa_u002e_0_u002e_1_ = _e2364;
            let _e2367 = lo_u002e_i_u002e_sroa_u002e_6_u002e_1_;
            out_u002e_lo_u002e_i_u002e_sroa_u002e_6_u002e_1_ = _e2367;
            let _e2370 = lo_u002e_i_u002e_sroa_u002e_9_u002e_1_;
            out_u002e_lo_u002e_i_u002e_sroa_u002e_9_u002e_1_ = _e2370;
            let _e2373 = (_e845 + 1u);
            if (_e2373 == _e723) {
                lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e2364;
                lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e2367;
                lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e2370;
                break;
            } else {
                k_u002e_sroa_u002e_0_u002e_0_ = _e2373;
                lo_u002e_i_u002e_sroa_u002e_9_ = _e2370;
                lo_u002e_i_u002e_sroa_u002e_6_ = _e2367;
                lo_u002e_i_u002e_sroa_u002e_0_ = _e2364;
                continue;
            }
        }
    }
    let _e2389 = lo_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
    let _e2391 = lo_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
    let _e2393 = lo_u002e_i_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
    let _e2403 = uniform_struct_0_3_.f0_.x;
    let _e2407 = uniform_struct_0_3_.f0_.y;
    let _e2411 = uniform_struct_0_3_.f0_.z;
    if !((_e399 <= 0f)) {
        let _e2418 = (_e399 * bitcast<f32>(_e404));
        let _e2419 = (_e399 * bitcast<f32>(_e410));
        let _e2420 = (_e399 * bitcast<f32>(_e416));
        let _e2430 = ((1f - _e334) + ((1f - pow((1f - _e46), 5f)) * _e334));
        emission_u002e_i_u002e_sroa_u002e_0_u002e_1_ = (_e2418 + ((((_e2418 * _e357) * _e2430) - _e2418) * _e334));
        emission_u002e_i_u002e_sroa_u002e_7_u002e_1_ = (_e2419 + ((((_e2419 * _e363) * _e2430) - _e2419) * _e334));
        emission_u002e_i_u002e_sroa_u002e_10_u002e_1_ = (_e2420 + ((((_e2420 * _e369) * _e2430) - _e2420) * _e334));
    } else {
        emission_u002e_i_u002e_sroa_u002e_0_u002e_1_ = 0f;
        emission_u002e_i_u002e_sroa_u002e_7_u002e_1_ = 0f;
        emission_u002e_i_u002e_sroa_u002e_10_u002e_1_ = 0f;
    }
    let _e2456 = emission_u002e_i_u002e_sroa_u002e_0_u002e_1_;
    let _e2458 = emission_u002e_i_u002e_sroa_u002e_7_u002e_1_;
    let _e2460 = emission_u002e_i_u002e_sroa_u002e_10_u002e_1_;
    let _e2465 = ((_e160 + -1f) / (_e160 + 1f));
    let _e2466 = (_e2465 * _e2465);
    let _e2474 = (dot(vec3<f32>(_e15, _e16, _e17), vec3<f32>(_e37, _e38, _e39)) * 2f);
    let _e2478 = ((_e15 * _e2474) - _e37);
    let _e2479 = ((_e16 * _e2474) - _e38);
    let _e2480 = ((_e17 * _e2474) - _e39);
    let _e2483 = uniform_struct_0_3_.f4_;
    let _e2484 = (_e154 * _e2483);
    let _e2488 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2478, _e2479, _e2480), _e2484);
    let _e2493 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2478, _e2479, _e2480), _e2484);
    let _e2498 = textureSampleLevel(tex_0_7_, sampler_0_10_, vec3<f32>(_e2478, _e2479, _e2480), _e2484);
    let _e2503 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e15, _e16, _e17));
    let _e2508 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e15, _e16, _e17));
    let _e2513 = textureSample(tex_0_8_, sampler_0_10_, vec3<f32>(_e15, _e16, _e17));
    let _e2518 = textureSample(tex_0_9_, sampler_0_10_, vec2<f32>(_e46, _e154));
    let _e2523 = textureSample(tex_0_9_, sampler_0_10_, vec2<f32>(_e46, _e154));
    let _e2526 = (1f - _e86);
    let _e2532 = uniform_struct_0_3_.f3_;
    let _e2545 = max((1f - _e2523.y), 0f);
    let _e2569 = !((bitcast<f32>(_e432) >= 0.5f));
    let _e2570 = select(_e2456, (((_e2389 + (mix(_e92, (_e92 * _e110), _e86) * _e2403)) + _e2456) + (_e2532 * ((_e2488.x * ((mix(_e2466, _e638, _e86) * _e2518.x) + _e2523.y)) + (((_e2526 * _e92) * _e2503.x) * _e2545)))), _e2569);
    let _e2571 = select(_e2458, (((_e2391 + (mix(_e98, (_e98 * _e110), _e86) * _e2407)) + _e2458) + (_e2532 * ((_e2493.y * ((mix(_e2466, _e639, _e86) * _e2518.x) + _e2523.y)) + (((_e2526 * _e98) * _e2508.y) * _e2545)))), _e2569);
    let _e2572 = select(_e2460, (((_e2393 + (mix(_e104, (_e104 * _e110), _e86) * _e2411)) + _e2460) + (_e2532 * ((_e2498.z * ((mix(_e2466, _e640, _e86) * _e2518.x) + _e2523.y)) + (((_e2526 * _e104) * _e2513.z) * _e2545)))), _e2569);
    if ((_e2570 <= _e2570) && (0f <= 0f)) {
        if (abs(_e2570) < 100000000000000000000f) {
            _2_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2570;
        } else {
            _2_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        _2_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2586 = _2_u002e_sroa_u002e_0_u002e_0_u002e_i;
    if ((_e2571 <= _e2571) && (0f <= 0f)) {
        if (abs(_e2571) < 100000000000000000000f) {
            _3_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2571;
        } else {
            _3_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        }
    } else {
        _3_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2600 = _3_u002e_sroa_u002e_0_u002e_0_u002e_i;
    if ((_e2572 <= _e2572) && (0f <= 0f)) {
        if (abs(_e2572) < 100000000000000000000f) {
            _4_u002e_sroa_u002e_0_u002e_0_u002e_i = _e2572;
            let _e2610 = _4_u002e_sroa_u002e_0_u002e_0_u002e_i;
            fragment_out = vec4<f32>(_e2586, _e2600, _e2610, bitcast<f32>(_e422));
            let _e2615 = fragment_out;
            return _e2615;
        }
        _4_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    } else {
        _4_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    }
    let _e2621 = _4_u002e_sroa_u002e_0_u002e_0_u002e_i;
    fragment_out = vec4<f32>(_e2586, _e2600, _e2621, bitcast<f32>(_e422));
    let _e2626 = fragment_out;
    return _e2626;
}
