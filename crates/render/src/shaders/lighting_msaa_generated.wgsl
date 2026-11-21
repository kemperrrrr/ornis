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
var tex_0_8_: texture_depth_multisampled_2d;
@group(0) @binding(3) 
var tex_0_3_: texture_2d<f32>;
@group(0) @binding(9) 
var sampler_0_9_: sampler;
@group(0) @binding(17) 
var tex_0_17_: texture_2d<f32>;
@group(0) @binding(6) 
var tex_0_6_: texture_2d<f32>;
@group(0) @binding(7) 
var tex_0_7_: texture_2d<f32>;
@group(0) @binding(0) 
var<uniform> uniform_struct_0_0_: UniformStruct0_;
@group(0) @binding(19) 
var<uniform> uniform_struct_0_19_: UniformStruct5_;
@group(0) @binding(1) 
var<uniform> uniform_struct_0_1_: UniformStruct4_;
@group(0) @binding(5) 
var tex_0_5_: texture_multisampled_2d<u32>;
@group(0) @binding(4) 
var tex_0_4_: texture_multisampled_2d<f32>;
@group(0) @binding(2) 
var<storage> storage_0_2_: array<u32>;
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
fn fs_main_lighting_msaa(@builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
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
    var alpha_acc_u002e_sroa_u002e_0_: f32;
    var weight_u002e_sroa_u002e_0_: f32;
    var single_alpha_u002e_sroa_u002e_0_: f32;
    var covered_count_u002e_sroa_u002e_0_: f32;
    var have_ref_u002e_sroa_u002e_0_: f32;
    var disagree_acc_u002e_sroa_u002e_0_: f32;
    var ref_depth_u002e_sroa_u002e_0_: f32;
    var ref_id_u002e_sroa_u002e_0_: u32;
    var s_u002e_sroa_u002e_0_: u32;
    var acc_u002e_sroa_u002e_0_: f32;
    var acc_u002e_sroa_u002e_4_: f32;
    var acc_u002e_sroa_u002e_7_: f32;
    var single_u002e_sroa_u002e_0_: f32;
    var single_u002e_sroa_u002e_4_: f32;
    var single_u002e_sroa_u002e_7_: f32;
    var ref_normal_u002e_sroa_u002e_0_: f32;
    var ref_normal_u002e_sroa_u002e_4_: f32;
    var ref_normal_u002e_sroa_u002e_6_: f32;
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
    var out_u002e_16_: f32;
    var out_u002e_15_: f32;
    var out_u002e_164_: f32;
    var out_u002e_3_: f32;
    var out_u002e_6_: f32;
    var out_u002e_9_: f32;
    var out_u002e_0_: f32;
    var out_u002e_0_1: f32;
    var out_u002e_0_2: f32;
    var out_u002e_163_: f32;
    var out_u002e_0_3: f32;
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
    let _e8 = textureDimensions(tex_0_8_);
    let _e13 = u32((uv.x * f32(bitcast<i32>(_e5.x))));
    let _e17 = u32((uv.y * f32(bitcast<i32>(_e8.y))));
    let _e22 = textureSampleLevel(tex_0_3_, sampler_0_9_, vec2<f32>(uv.x, uv.y), 0f);
    let _e28 = textureSampleLevel(tex_0_17_, sampler_0_9_, vec2<f32>(uv.x, uv.y), 0f);
    let _e34 = textureSampleLevel(tex_0_6_, sampler_0_9_, vec2<f32>(uv.x, uv.y), 0f);
    let _e40 = textureSampleLevel(tex_0_7_, sampler_0_9_, vec2<f32>(uv.x, uv.y), 0f);
    let _e45 = uniform_struct_0_0_.f8_.x;
    let _e49 = uniform_struct_0_0_.f8_.y;
    let _e53 = uniform_struct_0_0_.f8_.z;
    let _e56 = uniform_struct_0_19_.f2_;
    let _e59 = uniform_struct_0_1_.f5_;
    let _e62 = uniform_struct_0_1_.f2_;
    if (_e22.w < 0.001f) {
        discard;
    }
    let _e68 = ((uv.x * 2f) + -1f);
    let _e72 = (1f - (uv.y * 2f));
    let _e81 = select(0f, 1f, (_e59 == 0u));
    let _e86 = select(0f, 1f, (_e59 == 1u));
    let _e91 = select(0f, 1f, (_e59 == 2u));
    let _e96 = select(0f, 1f, (_e59 == 3u));
    let _e116 = select(0f, 1f, (_e59 == 7u));
    alpha_acc_u002e_sroa_u002e_0_ = 0f;
    weight_u002e_sroa_u002e_0_ = 0f;
    single_alpha_u002e_sroa_u002e_0_ = 0f;
    covered_count_u002e_sroa_u002e_0_ = 0f;
    have_ref_u002e_sroa_u002e_0_ = 0f;
    disagree_acc_u002e_sroa_u002e_0_ = 0f;
    ref_depth_u002e_sroa_u002e_0_ = 1f;
    ref_id_u002e_sroa_u002e_0_ = 0u;
    s_u002e_sroa_u002e_0_ = 0u;
    acc_u002e_sroa_u002e_0_ = 0f;
    acc_u002e_sroa_u002e_4_ = 0f;
    acc_u002e_sroa_u002e_7_ = 0f;
    single_u002e_sroa_u002e_0_ = 0f;
    single_u002e_sroa_u002e_4_ = 0f;
    single_u002e_sroa_u002e_7_ = 0f;
    ref_normal_u002e_sroa_u002e_0_ = 0f;
    ref_normal_u002e_sroa_u002e_4_ = 0f;
    ref_normal_u002e_sroa_u002e_6_ = 1f;
    loop {
        let _e154 = alpha_acc_u002e_sroa_u002e_0_;
        let _e156 = weight_u002e_sroa_u002e_0_;
        let _e158 = single_alpha_u002e_sroa_u002e_0_;
        let _e160 = covered_count_u002e_sroa_u002e_0_;
        let _e162 = have_ref_u002e_sroa_u002e_0_;
        let _e164 = disagree_acc_u002e_sroa_u002e_0_;
        let _e166 = ref_depth_u002e_sroa_u002e_0_;
        let _e168 = ref_id_u002e_sroa_u002e_0_;
        let _e170 = s_u002e_sroa_u002e_0_;
        let _e172 = acc_u002e_sroa_u002e_0_;
        let _e174 = acc_u002e_sroa_u002e_4_;
        let _e176 = acc_u002e_sroa_u002e_7_;
        let _e178 = single_u002e_sroa_u002e_0_;
        let _e180 = single_u002e_sroa_u002e_4_;
        let _e182 = single_u002e_sroa_u002e_7_;
        let _e184 = ref_normal_u002e_sroa_u002e_0_;
        let _e186 = ref_normal_u002e_sroa_u002e_4_;
        let _e188 = ref_normal_u002e_sroa_u002e_6_;
        let _e193 = textureLoad(tex_0_8_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e17)), _e170);
        let _e198 = textureLoad(tex_0_5_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e17)), _e170);
        let _e204 = textureLoad(tex_0_4_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e17)), _e170);
        let _e210 = textureLoad(tex_0_4_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e17)), _e170);
        let _e216 = ((1f - abs(_e204.x)) - abs(_e210.y));
        let _e221 = select(0f, -(_e216), (_e216 < 0f));
        let _e225 = -(_e221);
        let _e231 = (_e204.x + select(_e225, _e221, !((_e204.x >= 0f))));
        let _e232 = (_e210.y + select(_e225, _e221, !((_e210.y >= 0f))));
        let _e236 = (1f / length(vec3<f32>(_e231, _e232, _e216)));
        let _e237 = (_e236 * _e231);
        let _e238 = (_e236 * _e232);
        let _e239 = (_e216 * _e236);
        let _e241 = (_e193 < 1f);
        let _e244 = select(0f, 1f, _e241);
        let _e247 = ((_e162 == 0f) & _e241);
        let _e250 = select(0f, 1f, _e247);
        let _e251 = (_e193 - _e166);
        let _e257 = select(0f, 1f, (abs(_e251) > 0.002f));
        let _e265 = select(0f, 1f, (dot(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e184, _e186, _e188)) < 0.999f));
        let _e269 = select(1f, 0f, (_e198.x == _e168));
        let _e271 = select(_e265, _e269, (_e269 > _e265));
        let _e276 = (_e164 + (_e162 * (_e244 * select(_e257, _e271, (_e271 > _e257)))));
        out_u002e_15_ = _e276;
        let _e286 = (_e160 + _e244);
        out_u002e_16_ = _e286;
        let _e291 = uniform_struct_0_0_.f4_.w;
        let _e295 = uniform_struct_0_0_.f5_.w;
        let _e299 = uniform_struct_0_0_.f6_.w;
        let _e303 = uniform_struct_0_0_.f7_.w;
        let _e309 = (_e303 + (((_e68 * _e291) + (_e72 * _e295)) + (_e193 * _e299)));
        let _e313 = uniform_struct_0_0_.f4_.x;
        let _e317 = uniform_struct_0_0_.f5_.x;
        let _e321 = uniform_struct_0_0_.f6_.x;
        let _e325 = uniform_struct_0_0_.f7_.x;
        let _e335 = uniform_struct_0_0_.f4_.y;
        let _e339 = uniform_struct_0_0_.f5_.y;
        let _e343 = uniform_struct_0_0_.f6_.y;
        let _e347 = uniform_struct_0_0_.f7_.y;
        let _e357 = uniform_struct_0_0_.f4_.z;
        let _e361 = uniform_struct_0_0_.f5_.z;
        let _e365 = uniform_struct_0_0_.f6_.z;
        let _e369 = uniform_struct_0_0_.f7_.z;
        let _e376 = ((_e325 + (((_e68 * _e313) + (_e72 * _e317)) + (_e193 * _e321))) / _e309);
        let _e377 = ((_e347 + (((_e68 * _e335) + (_e72 * _e339)) + (_e193 * _e343))) / _e309);
        let _e378 = ((_e369 + (((_e68 * _e357) + (_e72 * _e361)) + (_e193 * _e365))) / _e309);
        let _e379 = (_e45 - _e376);
        let _e380 = (_e49 - _e377);
        let _e381 = (_e53 - _e378);
        let _e385 = (1f / length(vec3<f32>(_e379, _e380, _e381)));
        let _e386 = (_e379 * _e385);
        let _e387 = (_e380 * _e385);
        let _e388 = (_e385 * _e381);
        let _e391 = dot(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e386, _e387, _e388));
        let _e393 = (_e391 > 0.000001f);
        let _e395 = select(0.000001f, _e391, _e393);
        let _e397 = (_e198.x * 80u);
        let _e400 = storage_0_2_[_e397];
        let _e405 = storage_0_2_[(_e397 | 1u)];
        let _e410 = storage_0_2_[(_e397 | 2u)];
        let _e415 = storage_0_2_[(_e397 | 4u)];
        let _e420 = storage_0_2_[(_e397 | 5u)];
        let _e425 = storage_0_2_[(_e397 | 6u)];
        let _e430 = storage_0_2_[(_e397 | 8u)];
        let _e432 = (_e397 | 9u);
        let _e435 = storage_0_2_[_e432];
        let _e436 = bitcast<f32>(_e435);
        let _e437 = dpdx(_e237);
        let _e438 = dpdx(_e238);
        let _e439 = dpdx(_e239);
        let _e440 = dpdy(_e237);
        let _e441 = dpdy(_e238);
        let _e442 = dpdy(_e239);
        let _e453 = (((dot(vec3<f32>(_e437, _e438, _e439), vec3<f32>(_e437, _e438, _e439)) + dot(vec3<f32>(_e440, _e441, _e442), vec3<f32>(_e440, _e441, _e442))) * 0.15f) * 2f);
        let _e458 = (_e436 * _e436);
        let _e460 = ((_e458 * _e458) + select(0.2f, _e453, (_e453 < 0.2f)));
        let _e464 = select(_e460, 0f, (_e460 < 0f));
        let _e468 = select(_e464, 1f, (_e464 > 1f));
        let _e474 = sqrt(sqrt(select(0f, _e468, (_e468 > 0f))));
        let _e482 = storage_0_2_[(_e397 | 10u)];
        let _e487 = storage_0_2_[(_e397 | 11u)];
        let _e492 = storage_0_2_[(_e397 | 12u)];
        let _e497 = storage_0_2_[(_e397 | 13u)];
        let _e502 = storage_0_2_[(_e397 | 14u)];
        let _e507 = storage_0_2_[(_e397 + 16u)];
        let _e512 = storage_0_2_[(_e397 + 17u)];
        let _e527 = storage_0_2_[(_e397 + 20u)];
        let _e532 = storage_0_2_[(_e397 + 21u)];
        let _e537 = storage_0_2_[(_e397 + 22u)];
        let _e562 = storage_0_2_[(_e397 + 28u)];
        let _e567 = storage_0_2_[(_e397 + 29u)];
        let _e572 = storage_0_2_[(_e397 + 30u)];
        let _e577 = storage_0_2_[(_e397 + 31u)];
        let _e582 = storage_0_2_[(_e397 + 32u)];
        let _e587 = storage_0_2_[(_e397 + 33u)];
        let _e592 = storage_0_2_[(_e397 + 34u)];
        let _e597 = storage_0_2_[(_e397 + 36u)];
        let _e602 = storage_0_2_[(_e397 + 37u)];
        let _e607 = storage_0_2_[(_e397 + 40u)];
        let _e612 = storage_0_2_[(_e397 + 41u)];
        let _e617 = storage_0_2_[(_e397 + 44u)];
        let _e622 = storage_0_2_[(_e397 + 45u)];
        let _e627 = storage_0_2_[(_e397 + 46u)];
        let _e632 = storage_0_2_[(_e397 + 48u)];
        let _e637 = storage_0_2_[(_e397 + 49u)];
        let _e642 = storage_0_2_[(_e397 + 50u)];
        let _e647 = storage_0_2_[(_e397 + 51u)];
        let _e652 = storage_0_2_[(_e397 + 52u)];
        let _e657 = storage_0_2_[(_e397 + 53u)];
        let _e662 = storage_0_2_[(_e397 + 54u)];
        let _e667 = storage_0_2_[(_e397 + 56u)];
        let _e672 = storage_0_2_[(_e397 + 60u)];
        let _e677 = storage_0_2_[(_e397 + 61u)];
        let _e682 = storage_0_2_[(_e397 + 62u)];
        let _e687 = storage_0_2_[(_e397 + 64u)];
        let _e692 = storage_0_2_[(_e397 + 68u)];
        let _e697 = storage_0_2_[(_e397 + 69u)];
        let _e702 = storage_0_2_[(_e397 + 70u)];
        let _e707 = storage_0_2_[(_e397 + 72u)];
        let _e717 = storage_0_2_[(_e397 + 74u)];
        let _e718 = bitcast<f32>(_e707);
        let _e741 = (cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(0f, 1f, 0f)).x + 0f);
        let _e743 = (cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(0f, 1f, 0f)).y + 0f);
        let _e745 = (cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(0f, 1f, 0f)).z + 0.000001f);
        let _e749 = (1f / length(vec3<f32>(_e741, _e743, _e745)));
        let _e750 = (_e741 * _e749);
        let _e751 = (_e743 * _e749);
        let _e752 = (_e745 * _e749);
        let _e768 = uniform_struct_0_0_.f0_.x;
        let _e772 = uniform_struct_0_0_.f1_.x;
        let _e776 = uniform_struct_0_0_.f2_.x;
        let _e780 = uniform_struct_0_0_.f3_.x;
        let _e790 = uniform_struct_0_0_.f0_.y;
        let _e794 = uniform_struct_0_0_.f1_.y;
        let _e798 = uniform_struct_0_0_.f2_.y;
        let _e802 = uniform_struct_0_0_.f3_.y;
        let _e812 = uniform_struct_0_0_.f0_.w;
        let _e816 = uniform_struct_0_0_.f1_.w;
        let _e820 = uniform_struct_0_0_.f2_.w;
        let _e824 = uniform_struct_0_0_.f3_.w;
        let _e830 = (_e824 + (((_e376 * _e812) + (_e377 * _e816)) + (_e378 * _e820)));
        let _e834 = select(0.5f, _e830, (_e830 > 0.5f));
        let _e875 = ((((min(u32(floor((((((_e802 + (((_e376 * _e790) + (_e377 * _e794)) + (_e378 * _e798))) / _e834) * 0.5f) + 0.5f) * 9f))), 8u) << 4u) | min(u32(floor((((((_e780 + (((_e376 * _e768) + (_e377 * _e772)) + (_e378 * _e776))) / _e834) * 0.5f) + 0.5f) * 16f))), 15u)) + (min(u32(ceil(((log((_e834 * 2f)) * 3.4407384f) + -0.0001f))), 23u) * 144u)) << 1u);
        let _e877 = bitcast<f32>(_e682);
        let _e881 = (_e395 * _e395);
        let _e885 = max((1f - _e881), 0f);
        let _e887 = (sqrt(_e885) / _e877);
        let _e900 = pow(((_e877 + -1f) / (_e877 + 1f)), 2f);
        let _e904 = (((bitcast<f32>(_e677) * 1000f) * (_e877 * 12.566371f)) * sqrt(max((1f - (_e887 * _e887)), 0f)));
        let _e915 = (_e900 * 2f);
        let _e921 = (1f - (_e900 * _e900));
        let _e933 = clamp(bitcast<f32>(_e672), 0f, 1f);
        let _e938 = ((_e933 * ((((_e915 * cos((_e904 / 650f))) / _e921) + 1f) + -1f)) + 1f);
        let _e943 = ((_e933 * ((((_e915 * cos((_e904 / 550f))) / _e921) + 1f) + -1f)) + 1f);
        let _e948 = ((_e933 * ((((_e915 * cos((_e904 / 450f))) / _e921) + 1f) + -1f)) + 1f);
        let _e949 = bitcast<f32>(_e632);
        let _e950 = bitcast<f32>(_e667);
        let _e951 = bitcast<f32>(_e410);
        let _e952 = bitcast<f32>(_e400);
        let _e953 = bitcast<f32>(_e430);
        let _e954 = bitcast<f32>(_e562);
        let _e959 = ((_e950 + -1f) / (_e950 + 1f));
        let _e962 = bitcast<f32>(_e415);
        let _e963 = bitcast<f32>(_e420);
        let _e964 = bitcast<f32>(_e425);
        let _e965 = (_e952 * _e962);
        let _e966 = (_e952 * _e963);
        let _e967 = (_e952 * _e964);
        let _e968 = bitcast<f32>(_e582);
        let _e969 = bitcast<f32>(_e587);
        let _e970 = bitcast<f32>(_e592);
        let _e973 = !((_e949 <= 0f));
        if _e973 {
            let _e979 = (1f - ((1f - pow(_e959, 2f)) / (_e950 * _e950)));
            let _e981 = (1f - _e979);
            let _e982 = (_e967 * _e953);
            let _e995 = (_e966 * _e953);
            let _e1008 = (_e965 * _e953);
            let _e1022 = (_e949 * bitcast<f32>(_e647));
            _15_u002e_sroa_u002e_0_u002e_0_ = ((_e1022 * ((_e981 / max((1f - ((_e1008 + (((((_e962 - _e968) * _e954) + _e968) - _e1008) * _e951)) * _e979)), 0.000001f)) + -1f)) + 1f);
            _15_u002e_sroa_u002e_5_u002e_0_ = ((_e1022 * ((_e981 / max((1f - ((_e995 + (((((_e963 - _e969) * _e954) + _e969) - _e995) * _e951)) * _e979)), 0.000001f)) + -1f)) + 1f);
            _15_u002e_sroa_u002e_8_u002e_0_ = ((_e1022 * ((_e981 / max((1f - ((_e982 + (((((_e964 - _e970) * _e954) + _e970) - _e982) * _e951)) * _e979)), 0.000001f)) + -1f)) + 1f);
        } else {
            _15_u002e_sroa_u002e_0_u002e_0_ = 1f;
            _15_u002e_sroa_u002e_5_u002e_0_ = 1f;
            _15_u002e_sroa_u002e_8_u002e_0_ = 1f;
        }
        let _e1048 = _15_u002e_sroa_u002e_0_u002e_0_;
        let _e1050 = _15_u002e_sroa_u002e_5_u002e_0_;
        let _e1052 = _15_u002e_sroa_u002e_8_u002e_0_;
        let _e1055 = storage_0_22_[_e875];
        let _e1060 = storage_0_22_[(_e875 | 1u)];
        let _e1062 = ((_e1060 * _e56) + _e62);
        if (_e1062 == 0u) {
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
            let _e1093 = bitcast<f32>(_e405);
            let _e1094 = bitcast<f32>(_e482);
            let _e1100 = ((_e1094 + -1f) / (_e1094 + 1f));
            let _e1101 = (_e1100 * _e1100);
            let _e1102 = bitcast<f32>(_e492);
            let _e1103 = bitcast<f32>(_e497);
            let _e1104 = bitcast<f32>(_e502);
            let _e1105 = (_e474 * _e474);
            let _e1107 = (1f - bitcast<f32>(_e487));
            let _e1112 = (2f / ((_e1107 * _e1107) + 1f));
            let _e1114 = select(_e474, _e1093, (_e474 < _e1093));
            let _e1118 = (_e386 - (_e237 * _e395));
            let _e1119 = (_e387 - (_e238 * _e395));
            let _e1120 = (_e388 - (_e239 * _e395));
            let _e1123 = (1f - _e951);
            let _e1128 = max((_e1114 * _e1114), 0.000001f);
            let _e1129 = (_e1128 * _e1128);
            let _e1145 = bitcast<f32>(_e637);
            let _e1146 = (_e959 * _e959);
            let _e1149 = (_e1145 * _e1145);
            let _e1151 = (1f - bitcast<f32>(_e642));
            let _e1156 = (2f / ((_e1151 * _e1151) + 1f));
            let _e1160 = bitcast<f32>(_e607);
            let _e1164 = bitcast<f32>(_e612);
            let _e1165 = (_e1164 * _e1164);
            let _e1175 = bitcast<f32>(_e507);
            let _e1176 = bitcast<f32>(_e512);
            let _e1208 = bitcast<f32>(_e567);
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
                let _e1247 = shadow_vis_u002e_sroa_u002e_0_u002e_0_;
                let _e1249 = k_u002e_sroa_u002e_0_u002e_0_;
                let _e1251 = diff_acc_u002e_sroa_u002e_9_;
                let _e1253 = diff_acc_u002e_sroa_u002e_6_;
                let _e1255 = diff_acc_u002e_sroa_u002e_0_;
                let _e1257 = spec_acc_u002e_sroa_u002e_9_;
                let _e1259 = spec_acc_u002e_sroa_u002e_6_;
                let _e1261 = spec_acc_u002e_sroa_u002e_0_;
                let _e1263 = lo_u002e_sroa_u002e_9_;
                let _e1265 = lo_u002e_sroa_u002e_6_;
                let _e1267 = lo_u002e_sroa_u002e_0_;
                if (_e1249 < _e62) {
                    _0_u002e_sroa_u002e_0_u002e_0_u002e_in = _e1249;
                } else {
                    let _e1273 = storage_0_21_[(((_e1055 * _e56) - _e62) + _e1249)];
                    _0_u002e_sroa_u002e_0_u002e_0_u002e_in = _e1273;
                }
                let _e1276 = _0_u002e_sroa_u002e_0_u002e_0_u002e_in;
                let _e1278 = (_e1276 * 36u);
                let _e1281 = storage_0_20_[_e1278];
                let _e1282 = bitcast<f32>(_e1281);
                let _e1284 = (_e1278 + 8u);
                let _e1287 = storage_0_20_[_e1284];
                let _e1289 = (_e1278 + 9u);
                let _e1292 = storage_0_20_[_e1289];
                let _e1294 = (_e1278 + 10u);
                let _e1297 = storage_0_20_[_e1294];
                let _e1305 = length(vec3<f32>((bitcast<f32>(_e1287) - _e376), (bitcast<f32>(_e1292) - _e377), (bitcast<f32>(_e1297) - _e378)));
                let _e1307 = (_e1282 < 0.5f);
                let _e1310 = select(1f, 0f, _e1307);
                let _e1312 = (_e1278 + 16u);
                let _e1315 = storage_0_20_[_e1312];
                let _e1316 = bitcast<f32>(_e1315);
                let _e1328 = storage_0_20_[(_e1278 + 12u)];
                let _e1333 = storage_0_20_[(_e1278 + 13u)];
                let _e1338 = storage_0_20_[(_e1278 + 14u)];
                let _e1343 = storage_0_20_[(_e1278 + 15u)];
                let _e1344 = bitcast<f32>(_e1343);
                let _e1347 = storage_0_20_[_e1284];
                let _e1350 = storage_0_20_[_e1289];
                let _e1353 = storage_0_20_[_e1294];
                let _e1355 = (_e376 - bitcast<f32>(_e1347));
                let _e1357 = (_e377 - bitcast<f32>(_e1350));
                let _e1359 = (_e378 - bitcast<f32>(_e1353));
                let _e1363 = (1f / length(vec3<f32>(_e1355, _e1357, _e1359)));
                let _e1368 = (_e1278 + 4u);
                let _e1371 = storage_0_20_[_e1368];
                let _e1373 = (_e1278 + 5u);
                let _e1376 = storage_0_20_[_e1373];
                let _e1378 = (_e1278 + 6u);
                let _e1381 = storage_0_20_[_e1378];
                let _e1382 = bitcast<f32>(_e1371);
                let _e1383 = bitcast<f32>(_e1376);
                let _e1384 = bitcast<f32>(_e1381);
                let _e1388 = (1f / length(vec3<f32>(_e1382, _e1383, _e1384)));
                let _e1399 = storage_0_20_[(_e1278 + 18u)];
                let _e1405 = storage_0_20_[(_e1278 + 17u)];
                let _e1412 = select(1f, 0f, (_e1282 < 1.5f));
                let _e1414 = mix(1f, smoothstep(bitcast<f32>(_e1399), bitcast<f32>(_e1405), dot(vec3<f32>((_e1388 * _e1382), (_e1388 * _e1383), (_e1388 * _e1384)), vec3<f32>((_e1355 * _e1363), (_e1357 * _e1363), (_e1359 * _e1363)))), _e1412);
                let _e1415 = (_e1305 * _e1305);
                let _e1430 = (mix(1f, (1f - smoothstep((_e1316 * 0.99f), _e1316, _e1305)), _e1310) * mix(1f, (1f / select(_e1415, 0.000001f, (_e1415 < 0.000001f))), _e1310));
                let _e1434 = (_e1414 * ((bitcast<f32>(_e1328) * _e1344) * _e1430));
                let _e1435 = (_e1414 * ((bitcast<f32>(_e1333) * _e1344) * _e1430));
                let _e1436 = (_e1414 * ((bitcast<f32>(_e1338) * _e1344) * _e1430));
                let _e1440 = select((1f - _e1412), 0f, _e1307);
                if (((1f - _e1310) * select(1f, 0f, (_e1316 < 0.5f))) > 0.5f) {
                    let _e1455 = uniform_struct_0_1_.f6_.f1_.x;
                    let _e1460 = uniform_struct_0_1_.f6_.f1_.y;
                    let _e1465 = uniform_struct_0_1_.f6_.f1_.z;
                    let _e1468 = dot(vec3<f32>((_e376 - _e45), (_e377 - _e49), (_e378 - _e53)), vec3<f32>(_e1455, _e1460, _e1465));
                    let _e1473 = uniform_struct_0_1_.f6_.f2_.x;
                    let _e1478 = uniform_struct_0_1_.f6_.f2_.y;
                    let _e1483 = uniform_struct_0_1_.f6_.f2_.z;
                    let _e1488 = uniform_struct_0_1_.f6_.f2_.w;
                    let _e1492 = uniform_struct_0_1_.f6_.f3_;
                    let _e1496 = select(1f, 0f, (_e1468 < _e1478));
                    let _e1500 = select(1f, 0f, (_e1468 < _e1483));
                    let _e1504 = select(1f, 0f, (_e1468 < _e1488));
                    let _e1507 = u32(((_e1496 + _e1500) + _e1504));
                    let _e1510 = mix(mix(mix(_e1478, _e1483, _e1496), _e1488, _e1500), _e1492, _e1504);
                    let _e1520 = uniform_struct_0_1_.f6_.f0_[_e1507].f0_.x;
                    let _e1527 = uniform_struct_0_1_.f6_.f0_[_e1507].f1_.x;
                    let _e1534 = uniform_struct_0_1_.f6_.f0_[_e1507].f2_.x;
                    let _e1541 = uniform_struct_0_1_.f6_.f0_[_e1507].f3_.x;
                    let _e1554 = uniform_struct_0_1_.f6_.f0_[_e1507].f0_.y;
                    let _e1561 = uniform_struct_0_1_.f6_.f0_[_e1507].f1_.y;
                    let _e1568 = uniform_struct_0_1_.f6_.f0_[_e1507].f2_.y;
                    let _e1575 = uniform_struct_0_1_.f6_.f0_[_e1507].f3_.y;
                    let _e1588 = uniform_struct_0_1_.f6_.f0_[_e1507].f0_.z;
                    let _e1595 = uniform_struct_0_1_.f6_.f0_[_e1507].f1_.z;
                    let _e1602 = uniform_struct_0_1_.f6_.f0_[_e1507].f2_.z;
                    let _e1609 = uniform_struct_0_1_.f6_.f0_[_e1507].f3_.z;
                    let _e1622 = uniform_struct_0_1_.f6_.f0_[_e1507].f0_.w;
                    let _e1629 = uniform_struct_0_1_.f6_.f0_[_e1507].f1_.w;
                    let _e1636 = uniform_struct_0_1_.f6_.f0_[_e1507].f2_.w;
                    let _e1643 = uniform_struct_0_1_.f6_.f0_[_e1507].f3_.w;
                    let _e1649 = (_e1643 + (((_e376 * _e1622) + (_e377 * _e1629)) + (_e378 * _e1636)));
                    let _e1666 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e1541 + (((_e376 * _e1520) + (_e377 * _e1527)) + (_e378 * _e1534))) / _e1649) * 0.5f) + 0.5f), (0.5f - (((_e1575 + (((_e376 * _e1554) + (_e377 * _e1561)) + (_e378 * _e1568))) / _e1649) * 0.5f))), _e1507, (((_e1609 + (((_e376 * _e1588) + (_e377 * _e1595)) + (_e378 * _e1602))) / _e1649) + -0.001f));
                    if (_e1507 < 3u) {
                        let _e1671 = ((_e1510 - mix(mix(mix(_e1473, _e1478, _e1496), _e1483, _e1500), _e1488, _e1504)) * 0.08f);
                        let _e1677 = ((_e1510 - _e1468) / select(_e1671, 0.000001f, (_e1671 < 0.000001f)));
                        let _e1681 = select(_e1677, 0f, (_e1677 < 0f));
                        let _e1687 = (_e1507 + 1u);
                        let _e1694 = uniform_struct_0_1_.f6_.f0_[_e1687].f0_.x;
                        let _e1701 = uniform_struct_0_1_.f6_.f0_[_e1687].f1_.x;
                        let _e1708 = uniform_struct_0_1_.f6_.f0_[_e1687].f2_.x;
                        let _e1715 = uniform_struct_0_1_.f6_.f0_[_e1687].f3_.x;
                        let _e1728 = uniform_struct_0_1_.f6_.f0_[_e1687].f0_.y;
                        let _e1735 = uniform_struct_0_1_.f6_.f0_[_e1687].f1_.y;
                        let _e1742 = uniform_struct_0_1_.f6_.f0_[_e1687].f2_.y;
                        let _e1749 = uniform_struct_0_1_.f6_.f0_[_e1687].f3_.y;
                        let _e1762 = uniform_struct_0_1_.f6_.f0_[_e1687].f0_.z;
                        let _e1769 = uniform_struct_0_1_.f6_.f0_[_e1687].f1_.z;
                        let _e1776 = uniform_struct_0_1_.f6_.f0_[_e1687].f2_.z;
                        let _e1783 = uniform_struct_0_1_.f6_.f0_[_e1687].f3_.z;
                        let _e1796 = uniform_struct_0_1_.f6_.f0_[_e1687].f0_.w;
                        let _e1803 = uniform_struct_0_1_.f6_.f0_[_e1687].f1_.w;
                        let _e1810 = uniform_struct_0_1_.f6_.f0_[_e1687].f2_.w;
                        let _e1817 = uniform_struct_0_1_.f6_.f0_[_e1687].f3_.w;
                        let _e1823 = (_e1817 + (((_e376 * _e1796) + (_e377 * _e1803)) + (_e378 * _e1810)));
                        let _e1840 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e1715 + (((_e376 * _e1694) + (_e377 * _e1701)) + (_e378 * _e1708))) / _e1823) * 0.5f) + 0.5f), (0.5f - (((_e1749 + (((_e376 * _e1728) + (_e377 * _e1735)) + (_e378 * _e1742))) / _e1823) * 0.5f))), _e1687, (((_e1783 + (((_e376 * _e1762) + (_e377 * _e1769)) + (_e378 * _e1776))) / _e1823) + -0.001f));
                        vis_u002e_sroa_u002e_0_u002e_0_ = mix(_e1840, _e1666, select(_e1681, 1f, (_e1681 > 1f)));
                    } else {
                        vis_u002e_sroa_u002e_0_u002e_0_ = _e1666;
                    }
                    let _e1845 = vis_u002e_sroa_u002e_0_u002e_0_;
                    radiance_u002e_sroa_u002e_0_u002e_0_ = (_e1434 * _e1845);
                    radiance_u002e_sroa_u002e_12_u002e_0_ = (_e1435 * _e1845);
                    radiance_u002e_sroa_u002e_20_u002e_0_ = (_e1436 * _e1845);
                } else {
                    radiance_u002e_sroa_u002e_0_u002e_0_ = _e1434;
                    radiance_u002e_sroa_u002e_12_u002e_0_ = _e1435;
                    radiance_u002e_sroa_u002e_20_u002e_0_ = _e1436;
                }
                let _e1856 = radiance_u002e_sroa_u002e_0_u002e_0_;
                let _e1858 = radiance_u002e_sroa_u002e_12_u002e_0_;
                let _e1860 = radiance_u002e_sroa_u002e_20_u002e_0_;
                let _e1862 = (_e1278 + 19u);
                let _e1865 = storage_0_20_[_e1862];
                if !((bitcast<f32>(_e1865) >= 0f)) {
                    radiance_u002e_sroa_u002e_0_u002e_2_ = _e1856;
                    radiance_u002e_sroa_u002e_12_u002e_2_ = _e1858;
                    radiance_u002e_sroa_u002e_20_u002e_2_ = _e1860;
                } else {
                    if !((_e1440 <= 0.5f)) {
                        radiance_u002e_sroa_u002e_0_u002e_1_ = _e1856;
                        radiance_u002e_sroa_u002e_12_u002e_1_ = _e1858;
                        radiance_u002e_sroa_u002e_20_u002e_1_ = _e1860;
                    } else {
                        let _e1883 = storage_0_20_[(_e1278 + 20u)];
                        let _e1889 = storage_0_20_[(_e1278 + 24u)];
                        let _e1895 = storage_0_20_[(_e1278 + 28u)];
                        let _e1901 = storage_0_20_[(_e1278 + 32u)];
                        let _e1913 = storage_0_20_[(_e1278 + 21u)];
                        let _e1919 = storage_0_20_[(_e1278 + 25u)];
                        let _e1925 = storage_0_20_[(_e1278 + 29u)];
                        let _e1931 = storage_0_20_[(_e1278 + 33u)];
                        let _e1943 = storage_0_20_[(_e1278 + 22u)];
                        let _e1949 = storage_0_20_[(_e1278 + 26u)];
                        let _e1955 = storage_0_20_[(_e1278 + 30u)];
                        let _e1961 = storage_0_20_[(_e1278 + 34u)];
                        let _e1973 = storage_0_20_[(_e1278 + 23u)];
                        let _e1979 = storage_0_20_[(_e1278 + 27u)];
                        let _e1985 = storage_0_20_[(_e1278 + 31u)];
                        let _e1991 = storage_0_20_[(_e1278 + 35u)];
                        let _e1998 = ((((_e376 * bitcast<f32>(_e1973)) + (_e377 * bitcast<f32>(_e1979))) + (_e378 * bitcast<f32>(_e1985))) + bitcast<f32>(_e1991));
                        let _e2011 = storage_0_20_[_e1862];
                        let _e2020 = textureSampleCompareLevel(tex_0_10_, sampler_0_11_, vec2<f32>((((((((_e376 * bitcast<f32>(_e1883)) + (_e377 * bitcast<f32>(_e1889))) + (_e378 * bitcast<f32>(_e1895))) + bitcast<f32>(_e1901)) / _e1998) * 0.5f) + 0.5f), (0.5f - ((((((_e376 * bitcast<f32>(_e1913)) + (_e377 * bitcast<f32>(_e1919))) + (_e378 * bitcast<f32>(_e1925))) + bitcast<f32>(_e1931)) / _e1998) * 0.5f))), u32(bitcast<f32>(_e2011)), ((((((_e376 * bitcast<f32>(_e1943)) + (_e377 * bitcast<f32>(_e1949))) + (_e378 * bitcast<f32>(_e1955))) + bitcast<f32>(_e1961)) / _e1998) + -0.001f));
                        radiance_u002e_sroa_u002e_0_u002e_1_ = (_e1856 * _e2020);
                        radiance_u002e_sroa_u002e_12_u002e_1_ = (_e1858 * _e2020);
                        radiance_u002e_sroa_u002e_20_u002e_1_ = (_e1860 * _e2020);
                    }
                    let _e2028 = radiance_u002e_sroa_u002e_0_u002e_1_;
                    let _e2030 = radiance_u002e_sroa_u002e_12_u002e_1_;
                    let _e2032 = radiance_u002e_sroa_u002e_20_u002e_1_;
                    if (_e1440 > 0.5f) {
                        let _e2037 = storage_0_20_[_e1284];
                        let _e2040 = storage_0_20_[_e1289];
                        let _e2043 = storage_0_20_[_e1294];
                        let _e2045 = (_e376 - bitcast<f32>(_e2037));
                        let _e2047 = (_e377 - bitcast<f32>(_e2040));
                        let _e2049 = (_e378 - bitcast<f32>(_e2043));
                        let _e2050 = abs(_e2045);
                        let _e2051 = abs(_e2047);
                        let _e2053 = select(_e2050, _e2051, (_e2051 > _e2050));
                        let _e2054 = abs(_e2049);
                        let _e2059 = storage_0_20_[_e1312];
                        let _e2065 = select(bitcast<f32>(_e2059), 1f, (bitcast<f32>(_e2059) < 1f));
                        let _e2067 = (_e2065 + -0.1f);
                        let _e2078 = storage_0_20_[_e1862];
                        let _e2084 = textureSampleCompareLevel(tex_0_12_, sampler_0_11_, vec3<f32>(_e2045, _e2047, _e2049), u32(bitcast<f32>(_e2078)), (((_e2065 / _e2067) - ((_e2065 * 0.1f) / (select(_e2053, _e2054, (_e2054 > _e2053)) * _e2067))) + -0.001f));
                        radiance_u002e_sroa_u002e_0_u002e_2_ = (_e2028 * _e2084);
                        radiance_u002e_sroa_u002e_12_u002e_2_ = (_e2030 * _e2084);
                        radiance_u002e_sroa_u002e_20_u002e_2_ = (_e2032 * _e2084);
                    } else {
                        radiance_u002e_sroa_u002e_0_u002e_2_ = _e2028;
                        radiance_u002e_sroa_u002e_12_u002e_2_ = _e2030;
                        radiance_u002e_sroa_u002e_20_u002e_2_ = _e2032;
                    }
                }
                let _e2095 = radiance_u002e_sroa_u002e_0_u002e_2_;
                let _e2097 = radiance_u002e_sroa_u002e_12_u002e_2_;
                let _e2099 = radiance_u002e_sroa_u002e_20_u002e_2_;
                if (_e59 == 6u) {
                    let _e2102 = storage_0_20_[_e1278];
                    let _e2103 = bitcast<f32>(_e2102);
                    let _e2106 = storage_0_20_[_e1312];
                    let _e2110 = storage_0_20_[_e1862];
                    let _e2115 = select(1f, 0f, (_e2103 < 0.5f));
                    if (((1f - _e2115) * select(1f, 0f, (bitcast<f32>(_e2106) < 0.5f))) > 0.5f) {
                        let _e2132 = uniform_struct_0_1_.f6_.f1_.x;
                        let _e2137 = uniform_struct_0_1_.f6_.f1_.y;
                        let _e2142 = uniform_struct_0_1_.f6_.f1_.z;
                        let _e2145 = dot(vec3<f32>((_e376 - _e45), (_e377 - _e49), (_e378 - _e53)), vec3<f32>(_e2132, _e2137, _e2142));
                        let _e2150 = uniform_struct_0_1_.f6_.f2_.x;
                        let _e2155 = uniform_struct_0_1_.f6_.f2_.y;
                        let _e2160 = uniform_struct_0_1_.f6_.f2_.z;
                        let _e2165 = uniform_struct_0_1_.f6_.f2_.w;
                        let _e2169 = uniform_struct_0_1_.f6_.f3_;
                        let _e2173 = select(1f, 0f, (_e2145 < _e2155));
                        let _e2177 = select(1f, 0f, (_e2145 < _e2160));
                        let _e2181 = select(1f, 0f, (_e2145 < _e2165));
                        let _e2184 = u32(((_e2173 + _e2177) + _e2181));
                        let _e2187 = mix(mix(mix(_e2155, _e2160, _e2173), _e2165, _e2177), _e2169, _e2181);
                        let _e2197 = uniform_struct_0_1_.f6_.f0_[_e2184].f0_.x;
                        let _e2204 = uniform_struct_0_1_.f6_.f0_[_e2184].f1_.x;
                        let _e2211 = uniform_struct_0_1_.f6_.f0_[_e2184].f2_.x;
                        let _e2218 = uniform_struct_0_1_.f6_.f0_[_e2184].f3_.x;
                        let _e2231 = uniform_struct_0_1_.f6_.f0_[_e2184].f0_.y;
                        let _e2238 = uniform_struct_0_1_.f6_.f0_[_e2184].f1_.y;
                        let _e2245 = uniform_struct_0_1_.f6_.f0_[_e2184].f2_.y;
                        let _e2252 = uniform_struct_0_1_.f6_.f0_[_e2184].f3_.y;
                        let _e2265 = uniform_struct_0_1_.f6_.f0_[_e2184].f0_.z;
                        let _e2272 = uniform_struct_0_1_.f6_.f0_[_e2184].f1_.z;
                        let _e2279 = uniform_struct_0_1_.f6_.f0_[_e2184].f2_.z;
                        let _e2286 = uniform_struct_0_1_.f6_.f0_[_e2184].f3_.z;
                        let _e2299 = uniform_struct_0_1_.f6_.f0_[_e2184].f0_.w;
                        let _e2306 = uniform_struct_0_1_.f6_.f0_[_e2184].f1_.w;
                        let _e2313 = uniform_struct_0_1_.f6_.f0_[_e2184].f2_.w;
                        let _e2320 = uniform_struct_0_1_.f6_.f0_[_e2184].f3_.w;
                        let _e2326 = (_e2320 + (((_e376 * _e2299) + (_e377 * _e2306)) + (_e378 * _e2313)));
                        let _e2343 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e2218 + (((_e376 * _e2197) + (_e377 * _e2204)) + (_e378 * _e2211))) / _e2326) * 0.5f) + 0.5f), (0.5f - (((_e2252 + (((_e376 * _e2231) + (_e377 * _e2238)) + (_e378 * _e2245))) / _e2326) * 0.5f))), _e2184, (((_e2286 + (((_e376 * _e2265) + (_e377 * _e2272)) + (_e378 * _e2279))) / _e2326) + -0.001f));
                        if (_e2184 < 3u) {
                            let _e2348 = ((_e2187 - mix(mix(mix(_e2150, _e2155, _e2173), _e2160, _e2177), _e2165, _e2181)) * 0.08f);
                            let _e2354 = ((_e2187 - _e2145) / select(_e2348, 0.000001f, (_e2348 < 0.000001f)));
                            let _e2358 = select(_e2354, 0f, (_e2354 < 0f));
                            let _e2364 = (_e2184 + 1u);
                            let _e2371 = uniform_struct_0_1_.f6_.f0_[_e2364].f0_.x;
                            let _e2378 = uniform_struct_0_1_.f6_.f0_[_e2364].f1_.x;
                            let _e2385 = uniform_struct_0_1_.f6_.f0_[_e2364].f2_.x;
                            let _e2392 = uniform_struct_0_1_.f6_.f0_[_e2364].f3_.x;
                            let _e2405 = uniform_struct_0_1_.f6_.f0_[_e2364].f0_.y;
                            let _e2412 = uniform_struct_0_1_.f6_.f0_[_e2364].f1_.y;
                            let _e2419 = uniform_struct_0_1_.f6_.f0_[_e2364].f2_.y;
                            let _e2426 = uniform_struct_0_1_.f6_.f0_[_e2364].f3_.y;
                            let _e2439 = uniform_struct_0_1_.f6_.f0_[_e2364].f0_.z;
                            let _e2446 = uniform_struct_0_1_.f6_.f0_[_e2364].f1_.z;
                            let _e2453 = uniform_struct_0_1_.f6_.f0_[_e2364].f2_.z;
                            let _e2460 = uniform_struct_0_1_.f6_.f0_[_e2364].f3_.z;
                            let _e2473 = uniform_struct_0_1_.f6_.f0_[_e2364].f0_.w;
                            let _e2480 = uniform_struct_0_1_.f6_.f0_[_e2364].f1_.w;
                            let _e2487 = uniform_struct_0_1_.f6_.f0_[_e2364].f2_.w;
                            let _e2494 = uniform_struct_0_1_.f6_.f0_[_e2364].f3_.w;
                            let _e2500 = (_e2494 + (((_e376 * _e2473) + (_e377 * _e2480)) + (_e378 * _e2487)));
                            let _e2517 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e2392 + (((_e376 * _e2371) + (_e377 * _e2378)) + (_e378 * _e2385))) / _e2500) * 0.5f) + 0.5f), (0.5f - (((_e2426 + (((_e376 * _e2405) + (_e377 * _e2412)) + (_e378 * _e2419))) / _e2500) * 0.5f))), _e2364, (((_e2460 + (((_e376 * _e2439) + (_e377 * _e2446)) + (_e378 * _e2453))) / _e2500) + -0.001f));
                            _0_u002e_sroa_u002e_0_u002e_0_ = mix(_e2517, _e2343, select(_e2358, 1f, (_e2358 > 1f)));
                        } else {
                            _0_u002e_sroa_u002e_0_u002e_0_ = _e2343;
                        }
                    } else {
                        if (bitcast<f32>(_e2110) < 0f) {
                            _0_u002e_sroa_u002e_0_u002e_0_ = 1f;
                        } else {
                            if ((_e2103 < 1.5f) & (_e2115 > 0.5f)) {
                                let _e2531 = storage_0_20_[_e1284];
                                let _e2534 = storage_0_20_[_e1289];
                                let _e2537 = storage_0_20_[_e1294];
                                let _e2539 = (_e376 - bitcast<f32>(_e2531));
                                let _e2541 = (_e377 - bitcast<f32>(_e2534));
                                let _e2543 = (_e378 - bitcast<f32>(_e2537));
                                let _e2544 = abs(_e2539);
                                let _e2545 = abs(_e2541);
                                let _e2547 = select(_e2544, _e2545, (_e2545 > _e2544));
                                let _e2548 = abs(_e2543);
                                let _e2553 = storage_0_20_[_e1312];
                                let _e2559 = select(bitcast<f32>(_e2553), 1f, (bitcast<f32>(_e2553) < 1f));
                                let _e2561 = (_e2559 + -0.1f);
                                let _e2572 = storage_0_20_[_e1862];
                                let _e2578 = textureSampleCompareLevel(tex_0_12_, sampler_0_11_, vec3<f32>(_e2539, _e2541, _e2543), u32(bitcast<f32>(_e2572)), (((_e2559 / _e2561) - ((_e2559 * 0.1f) / (select(_e2547, _e2548, (_e2548 > _e2547)) * _e2561))) + -0.001f));
                                _0_u002e_sroa_u002e_0_u002e_0_ = _e2578;
                            } else {
                                let _e2584 = storage_0_20_[(_e1278 + 20u)];
                                let _e2590 = storage_0_20_[(_e1278 + 24u)];
                                let _e2596 = storage_0_20_[(_e1278 + 28u)];
                                let _e2602 = storage_0_20_[(_e1278 + 32u)];
                                let _e2614 = storage_0_20_[(_e1278 + 21u)];
                                let _e2620 = storage_0_20_[(_e1278 + 25u)];
                                let _e2626 = storage_0_20_[(_e1278 + 29u)];
                                let _e2632 = storage_0_20_[(_e1278 + 33u)];
                                let _e2644 = storage_0_20_[(_e1278 + 22u)];
                                let _e2650 = storage_0_20_[(_e1278 + 26u)];
                                let _e2656 = storage_0_20_[(_e1278 + 30u)];
                                let _e2662 = storage_0_20_[(_e1278 + 34u)];
                                let _e2674 = storage_0_20_[(_e1278 + 23u)];
                                let _e2680 = storage_0_20_[(_e1278 + 27u)];
                                let _e2686 = storage_0_20_[(_e1278 + 31u)];
                                let _e2692 = storage_0_20_[(_e1278 + 35u)];
                                let _e2699 = ((((_e376 * bitcast<f32>(_e2674)) + (_e377 * bitcast<f32>(_e2680))) + (_e378 * bitcast<f32>(_e2686))) + bitcast<f32>(_e2692));
                                let _e2712 = storage_0_20_[_e1862];
                                let _e2721 = textureSampleCompareLevel(tex_0_10_, sampler_0_11_, vec2<f32>((((((((_e376 * bitcast<f32>(_e2584)) + (_e377 * bitcast<f32>(_e2590))) + (_e378 * bitcast<f32>(_e2596))) + bitcast<f32>(_e2602)) / _e2699) * 0.5f) + 0.5f), (0.5f - ((((((_e376 * bitcast<f32>(_e2614)) + (_e377 * bitcast<f32>(_e2620))) + (_e378 * bitcast<f32>(_e2626))) + bitcast<f32>(_e2632)) / _e2699) * 0.5f))), u32(bitcast<f32>(_e2712)), ((((((_e376 * bitcast<f32>(_e2644)) + (_e377 * bitcast<f32>(_e2650))) + (_e378 * bitcast<f32>(_e2656))) + bitcast<f32>(_e2662)) / _e2699) + -0.001f));
                                _0_u002e_sroa_u002e_0_u002e_0_ = _e2721;
                            }
                        }
                    }
                    let _e2724 = _0_u002e_sroa_u002e_0_u002e_0_;
                    shadow_vis_u002e_sroa_u002e_0_u002e_1_ = select(_e2724, _e1247, (_e1247 < _e2724));
                } else {
                    shadow_vis_u002e_sroa_u002e_0_u002e_1_ = _e1247;
                }
                let _e2730 = shadow_vis_u002e_sroa_u002e_0_u002e_1_;
                out_u002e_shadow_vis_u002e_sroa_u002e_0_u002e_1_ = _e2730;
                let _e2734 = storage_0_20_[_e1284];
                let _e2737 = storage_0_20_[_e1289];
                let _e2740 = storage_0_20_[_e1294];
                let _e2742 = (bitcast<f32>(_e2734) - _e376);
                let _e2744 = (bitcast<f32>(_e2737) - _e377);
                let _e2746 = (bitcast<f32>(_e2740) - _e378);
                let _e2749 = storage_0_20_[_e1278];
                let _e2755 = select(1f, 0f, (bitcast<f32>(_e2749) < 0.5f));
                let _e2758 = storage_0_20_[_e1368];
                let _e2761 = storage_0_20_[_e1373];
                let _e2764 = storage_0_20_[_e1378];
                let _e2765 = bitcast<f32>(_e2758);
                let _e2766 = bitcast<f32>(_e2761);
                let _e2767 = bitcast<f32>(_e2764);
                let _e2771 = (1f / length(vec3<f32>(_e2765, _e2766, _e2767)));
                let _e2778 = (1f / length(vec3<f32>(_e2742, _e2744, _e2746)));
                let _e2782 = mix((_e2771 * _e2765), (_e2742 * _e2778), _e2755);
                let _e2783 = mix((_e2771 * _e2766), (_e2744 * _e2778), _e2755);
                let _e2784 = mix((_e2771 * _e2767), (_e2746 * _e2778), _e2755);
                let _e2788 = (1f / length(vec3<f32>(_e2782, _e2783, _e2784)));
                let _e2789 = (_e2782 * _e2788);
                let _e2790 = (_e2783 * _e2788);
                let _e2791 = (_e2784 * _e2788);
                let _e2796 = ((_e386 + _e2789) + 0f);
                let _e2798 = ((_e387 + _e2790) + 0f);
                let _e2800 = ((_e388 + _e2791) + 0.000001f);
                let _e2804 = (1f / length(vec3<f32>(_e2796, _e2798, _e2800)));
                let _e2805 = (_e2804 * _e2796);
                let _e2806 = (_e2804 * _e2798);
                let _e2807 = (_e2804 * _e2800);
                let _e2810 = dot(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e2789, _e2790, _e2791));
                let _e2814 = select(_e2810, 0.000001f, (_e2810 < 0.000001f));
                if (_e2814 > 0.000001f) {
                    let _e2819 = dot(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e2805, _e2806, _e2807));
                    let _e2823 = select(_e2819, 0.000001f, (_e2819 < 0.000001f));
                    let _e2826 = dot(vec3<f32>(_e386, _e387, _e388), vec3<f32>(_e2805, _e2806, _e2807));
                    let _e2830 = select(_e2826, 0.000001f, (_e2826 < 0.000001f));
                    let _e2831 = mix(_e1101, _e965, _e951);
                    let _e2832 = mix(_e1101, _e966, _e951);
                    let _e2833 = mix(_e1101, _e967, _e951);
                    let _e2836 = pow(0.85714287f, 5f);
                    let _e2838 = (1f - _e2831);
                    let _e2840 = (_e2831 + (_e2838 * _e2836));
                    let _e2842 = (1f - _e2832);
                    let _e2844 = (_e2832 + (_e2842 * _e2836));
                    let _e2846 = (1f - _e2833);
                    let _e2848 = (_e2833 + (_e2846 * _e2836));
                    let _e2853 = (1f - _e2830);
                    let _e2862 = ((_e2830 * pow(_e2853, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                    let _e2864 = pow(_e2853, 5f);
                    let _e2886 = mix(_e1101, _e965, _e951);
                    let _e2887 = mix(_e1101, _e966, _e951);
                    let _e2888 = mix(_e1101, _e967, _e951);
                    let _e2891 = pow(0.85714287f, 5f);
                    let _e2893 = (1f - _e2886);
                    let _e2895 = (_e2886 + (_e2893 * _e2891));
                    let _e2897 = (1f - _e2887);
                    let _e2899 = (_e2887 + (_e2897 * _e2891));
                    let _e2901 = (1f - _e2888);
                    let _e2903 = (_e2888 + (_e2901 * _e2891));
                    let _e2915 = ((_e2830 * pow(_e2853, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                    let _e2917 = pow(_e2853, 5f);
                    let _e2940 = (_e1105 * sqrt(_e1112));
                    let _e2943 = (_e1107 * (_e1105 * sqrt(_e1112)));
                    let _e2946 = dot(vec3<f32>(_e2805, _e2806, _e2807), vec3<f32>(_e750, _e751, _e752));
                    let _e2949 = dot(vec3<f32>(_e2805, _e2806, _e2807), vec3<f32>(cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).x, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).y, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).z));
                    let _e2951 = (_e2940 * _e2940);
                    let _e2954 = (_e2943 * _e2943);
                    let _e2957 = (_e2823 * _e2823);
                    let _e2958 = (_e2957 + (((_e2946 * _e2946) / _e2951) + ((_e2949 * _e2949) / _e2954)));
                    let _e2965 = (1f / (_e2958 * (((_e2940 * 3.1415927f) * _e2943) * _e2958)));
                    let _e2968 = dot(vec3<f32>(_e386, _e387, _e388), vec3<f32>(_e750, _e751, _e752));
                    let _e2971 = dot(vec3<f32>(_e386, _e387, _e388), vec3<f32>(cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).x, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).y, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).z));
                    let _e2974 = dot(vec3<f32>(_e2789, _e2790, _e2791), vec3<f32>(_e750, _e751, _e752));
                    let _e2977 = dot(vec3<f32>(_e2789, _e2790, _e2791), vec3<f32>(cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).x, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).y, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).z));
                    let _e2983 = (_e2814 * _e2814);
                    let _e3003 = (0.5f / max(((_e395 * sqrt(max((_e2983 + ((_e2951 * (_e2974 * _e2974)) + (_e2954 * (_e2977 * _e2977)))), 0.000001f))) + (_e2814 * sqrt(max((_e881 + ((_e2951 * (_e2968 * _e2968)) + (_e2954 * (_e2971 * _e2971)))), 0.000001f)))), 0.000001f));
                    let _e3022 = (1f / length(vec3<f32>(_e1118, _e1119, _e1120)));
                    let _e3029 = (_e2789 - (_e237 * _e2814));
                    let _e3030 = (_e2790 - (_e238 * _e2814));
                    let _e3031 = (_e2791 - (_e239 * _e2814));
                    let _e3035 = (1f / length(vec3<f32>(_e3029, _e3030, _e3031)));
                    let _e3041 = dot(vec3<f32>((_e1118 * _e3022), (_e1119 * _e3022), (_e1120 * _e3022)), vec3<f32>((_e3029 * _e3035), (_e3030 * _e3035), (_e3031 * _e3035)));
                    let _e3046 = acos(max(_e395, 0f));
                    let _e3049 = acos(max(_e2814, 0f));
                    let _e3073 = (max((1f - dot(vec3<f32>((max(((_e2831 + (_e2838 * _e2864)) - ((_e2840 - (_e2840 * _e1102)) * _e2862)), 0f) * _e953), (max(((_e2832 + (_e2842 * _e2864)) - ((_e2844 - (_e2844 * _e1103)) * _e2862)), 0f) * _e953), (max(((_e2833 + (_e2846 * _e2864)) - ((_e2848 - (_e2848 * _e1104)) * _e2862)), 0f) * _e953)), vec3<f32>(0.2126f, 0.7152f, 0.0722f))), 0f) * (((1f - ((_e1129 * 0.5f) / (_e1129 + 0.57f))) + (tan(min(_e3046, _e3049)) * ((((_e1129 * 0.45f) / (_e1129 + 0.09f)) * select(0f, _e3041, (_e3041 > 0f))) * sin(max(_e3046, _e3049))))) * 0.31830987f));
                    let _e3086 = (_e1048 * ((_e938 * ((((max(((_e2886 + (_e2893 * _e2917)) - ((_e2895 - (_e2895 * _e1102)) * _e2915)), 0f) * _e953) * _e2965) * _e3003) * _e952)) + (_e938 * (((_e1123 * _e962) * _e3073) * _e952))));
                    let _e3087 = (_e1050 * ((_e943 * ((((max(((_e2887 + (_e2897 * _e2917)) - ((_e2899 - (_e2899 * _e1103)) * _e2915)), 0f) * _e953) * _e2965) * _e3003) * _e952)) + (_e943 * (((_e1123 * _e963) * _e3073) * _e952))));
                    let _e3088 = (_e1052 * ((_e948 * ((((max(((_e2888 + (_e2901 * _e2917)) - ((_e2903 - (_e2903 * _e1104)) * _e2915)), 0f) * _e953) * _e2965) * _e3003) * _e952)) + (_e948 * (((_e1123 * _e964) * _e3073) * _e952))));
                    if (_e59 == 0u) {
                        spec_acc_u002e_sroa_u002e_0_u002e_2_ = _e1261;
                        spec_acc_u002e_sroa_u002e_6_u002e_2_ = _e1259;
                        spec_acc_u002e_sroa_u002e_9_u002e_2_ = _e1257;
                        diff_acc_u002e_sroa_u002e_0_u002e_2_ = _e1255;
                        diff_acc_u002e_sroa_u002e_6_u002e_2_ = _e1253;
                        diff_acc_u002e_sroa_u002e_9_u002e_2_ = _e1251;
                    } else {
                        let _e3095 = mix(_e1101, _e965, _e951);
                        let _e3096 = mix(_e1101, _e966, _e951);
                        let _e3097 = mix(_e1101, _e967, _e951);
                        let _e3100 = pow(0.85714287f, 5f);
                        let _e3102 = (1f - _e3095);
                        let _e3104 = (_e3095 + (_e3102 * _e3100));
                        let _e3106 = (1f - _e3096);
                        let _e3108 = (_e3096 + (_e3106 * _e3100));
                        let _e3110 = (1f - _e3097);
                        let _e3112 = (_e3097 + (_e3110 * _e3100));
                        let _e3124 = ((_e2830 * pow(_e2853, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                        let _e3126 = pow(_e2853, 5f);
                        let _e3149 = (_e1105 * sqrt(_e1112));
                        let _e3152 = (_e1107 * (_e1105 * sqrt(_e1112)));
                        let _e3155 = dot(vec3<f32>(_e2805, _e2806, _e2807), vec3<f32>(_e750, _e751, _e752));
                        let _e3158 = dot(vec3<f32>(_e2805, _e2806, _e2807), vec3<f32>(cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).x, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).y, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).z));
                        let _e3160 = (_e3149 * _e3149);
                        let _e3163 = (_e3152 * _e3152);
                        let _e3166 = (_e2957 + (((_e3155 * _e3155) / _e3160) + ((_e3158 * _e3158) / _e3163)));
                        let _e3173 = (1f / (_e3166 * (((_e3149 * 3.1415927f) * _e3152) * _e3166)));
                        let _e3176 = dot(vec3<f32>(_e386, _e387, _e388), vec3<f32>(_e750, _e751, _e752));
                        let _e3179 = dot(vec3<f32>(_e386, _e387, _e388), vec3<f32>(cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).x, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).y, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).z));
                        let _e3182 = dot(vec3<f32>(_e2789, _e2790, _e2791), vec3<f32>(_e750, _e751, _e752));
                        let _e3185 = dot(vec3<f32>(_e2789, _e2790, _e2791), vec3<f32>(cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).x, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).y, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).z));
                        let _e3210 = (0.5f / max(((_e395 * sqrt(max((_e2983 + ((_e3160 * (_e3182 * _e3182)) + (_e3163 * (_e3185 * _e3185)))), 0.000001f))) + (_e2814 * sqrt(max((_e881 + ((_e3160 * (_e3176 * _e3176)) + (_e3163 * (_e3179 * _e3179)))), 0.000001f)))), 0.000001f));
                        let _e3226 = (_e1048 * (_e938 * ((((max(((_e3095 + (_e3102 * _e3126)) - ((_e3104 - (_e3104 * _e1102)) * _e3124)), 0f) * _e953) * _e3173) * _e3210) * _e952)));
                        let _e3227 = (_e1050 * (_e943 * ((((max(((_e3096 + (_e3106 * _e3126)) - ((_e3108 - (_e3108 * _e1103)) * _e3124)), 0f) * _e953) * _e3173) * _e3210) * _e952)));
                        let _e3228 = (_e1052 * (_e948 * ((((max(((_e3097 + (_e3110 * _e3126)) - ((_e3112 - (_e3112 * _e1104)) * _e3124)), 0f) * _e953) * _e3173) * _e3210) * _e952)));
                        spec_acc_u002e_sroa_u002e_0_u002e_2_ = (_e1261 + (_e2095 * (_e2814 * _e3226)));
                        spec_acc_u002e_sroa_u002e_6_u002e_2_ = (_e1259 + (_e2097 * (_e2814 * _e3227)));
                        spec_acc_u002e_sroa_u002e_9_u002e_2_ = (_e1257 + (_e2099 * (_e2814 * _e3228)));
                        diff_acc_u002e_sroa_u002e_0_u002e_2_ = (_e1255 + (_e2095 * (_e2814 * (_e3086 - _e3226))));
                        diff_acc_u002e_sroa_u002e_6_u002e_2_ = (_e1253 + (_e2097 * (_e2814 * (_e3087 - _e3227))));
                        diff_acc_u002e_sroa_u002e_9_u002e_2_ = (_e1251 + (_e2099 * (_e2814 * (_e3088 - _e3228))));
                    }
                    let _e3257 = spec_acc_u002e_sroa_u002e_0_u002e_2_;
                    let _e3259 = spec_acc_u002e_sroa_u002e_6_u002e_2_;
                    let _e3261 = spec_acc_u002e_sroa_u002e_9_u002e_2_;
                    let _e3263 = diff_acc_u002e_sroa_u002e_0_u002e_2_;
                    let _e3265 = diff_acc_u002e_sroa_u002e_6_u002e_2_;
                    let _e3267 = diff_acc_u002e_sroa_u002e_9_u002e_2_;
                    if _e973 {
                        let _e3273 = (_e1149 * sqrt(_e1156));
                        let _e3276 = (_e1151 * (_e1149 * sqrt(_e1156)));
                        let _e3279 = dot(vec3<f32>(_e2805, _e2806, _e2807), vec3<f32>(_e750, _e751, _e752));
                        let _e3282 = dot(vec3<f32>(_e2805, _e2806, _e2807), vec3<f32>(cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).x, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).y, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).z));
                        let _e3284 = (_e3273 * _e3273);
                        let _e3287 = (_e3276 * _e3276);
                        let _e3290 = (_e2957 + (((_e3279 * _e3279) / _e3284) + ((_e3282 * _e3282) / _e3287)));
                        let _e3300 = dot(vec3<f32>(_e386, _e387, _e388), vec3<f32>(_e750, _e751, _e752));
                        let _e3303 = dot(vec3<f32>(_e386, _e387, _e388), vec3<f32>(cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).x, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).y, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).z));
                        let _e3306 = dot(vec3<f32>(_e2789, _e2790, _e2791), vec3<f32>(_e750, _e751, _e752));
                        let _e3309 = dot(vec3<f32>(_e2789, _e2790, _e2791), vec3<f32>(cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).x, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).y, cross(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e750, _e751, _e752)).z));
                        let _e3336 = ((_e1146 + ((1f - _e1146) * pow(_e2853, 5f))) * ((1f / (_e3290 * (((_e3273 * 3.1415927f) * _e3276) * _e3290))) * (0.5f / max(((_e395 * sqrt(max((_e2983 + ((_e3284 * (_e3306 * _e3306)) + (_e3287 * (_e3309 * _e3309)))), 0.000001f))) + (_e2814 * sqrt(max((_e881 + ((_e3284 * (_e3300 * _e3300)) + (_e3287 * (_e3303 * _e3303)))), 0.000001f)))), 0.000001f))));
                        coat_bsdf_u002e_sroa_u002e_0_u002e_0_ = ((_e3336 * bitcast<f32>(_e652)) * _e949);
                        coat_bsdf_u002e_sroa_u002e_5_u002e_0_ = ((_e3336 * bitcast<f32>(_e657)) * _e949);
                        coat_bsdf_u002e_sroa_u002e_8_u002e_0_ = ((_e3336 * bitcast<f32>(_e662)) * _e949);
                    } else {
                        coat_bsdf_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        coat_bsdf_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        coat_bsdf_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e3353 = coat_bsdf_u002e_sroa_u002e_0_u002e_0_;
                    let _e3355 = coat_bsdf_u002e_sroa_u002e_5_u002e_0_;
                    let _e3357 = coat_bsdf_u002e_sroa_u002e_8_u002e_0_;
                    if !((_e1160 <= 0f)) {
                        let _e3381 = ((_e2830 * ((1f / ((_e1165 * (((1f / _e395) + (1f / _e2814)) + -2f)) + 1f)) * (_e1165 / (pow((((_e1165 + -1f) * _e2957) + 1f), 2f) * 3.1415927f)))) / max(((_e395 * 4f) * _e2814), 0.000001f));
                        fuzz_bsdf_u002e_sroa_u002e_0_u002e_0_ = ((_e3381 * bitcast<f32>(_e617)) * _e1160);
                        fuzz_bsdf_u002e_sroa_u002e_5_u002e_0_ = ((_e3381 * bitcast<f32>(_e622)) * _e1160);
                        fuzz_bsdf_u002e_sroa_u002e_8_u002e_0_ = ((_e3381 * bitcast<f32>(_e627)) * _e1160);
                    } else {
                        fuzz_bsdf_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        fuzz_bsdf_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        fuzz_bsdf_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e3398 = fuzz_bsdf_u002e_sroa_u002e_0_u002e_0_;
                    let _e3400 = fuzz_bsdf_u002e_sroa_u002e_5_u002e_0_;
                    let _e3402 = fuzz_bsdf_u002e_sroa_u002e_8_u002e_0_;
                    if !((_e1175 <= 0f)) {
                        let _e3404 = (_e1105 * sqrt(_e1112));
                        if !((_e1176 <= 0f)) {
                            extinction_u002e_sroa_u002e_0_u002e_0_ = (-(log(max(bitcast<f32>(_e527), 0.000001f))) / _e1176);
                            extinction_u002e_sroa_u002e_5_u002e_0_ = (-(log(max(bitcast<f32>(_e532), 0.000001f))) / _e1176);
                            extinction_u002e_sroa_u002e_8_u002e_0_ = (-(log(max(bitcast<f32>(_e537), 0.000001f))) / _e1176);
                        } else {
                            extinction_u002e_sroa_u002e_0_u002e_0_ = 0f;
                            extinction_u002e_sroa_u002e_5_u002e_0_ = 0f;
                            extinction_u002e_sroa_u002e_8_u002e_0_ = 0f;
                        }
                        let _e3424 = extinction_u002e_sroa_u002e_0_u002e_0_;
                        let _e3426 = extinction_u002e_sroa_u002e_5_u002e_0_;
                        let _e3428 = extinction_u002e_sroa_u002e_8_u002e_0_;
                        let _e3435 = (_e3404 * _e3404);
                        let _e3440 = ((_e2957 * (_e3435 + -1f)) + 1f);
                        let _e3446 = (1f - _e3435);
                        let _e3465 = ((1f - (_e1101 + ((1f - _e1101) * pow(select(0.999999f, (1f - _e391), _e393), 5f)))) * ((_e3435 / (_e3440 * (_e3440 * 3.1415927f))) * (0.5f / max(((_e395 * sqrt(max((_e3435 + (_e2983 * _e3446)), 0.000001f))) + (_e2814 * sqrt(max((_e3435 + (_e881 * _e3446)), 0.000001f)))), 0.000001f))));
                        trans_bsdf_u002e_sroa_u002e_0_u002e_0_ = (((exp((_e1176 * -(_e3424))) * _e3465) * bitcast<f32>(_e527)) * _e1175);
                        trans_bsdf_u002e_sroa_u002e_5_u002e_0_ = (((exp((_e1176 * -(_e3426))) * _e3465) * bitcast<f32>(_e532)) * _e1175);
                        trans_bsdf_u002e_sroa_u002e_8_u002e_0_ = (((exp((_e1176 * -(_e3428))) * _e3465) * bitcast<f32>(_e537)) * _e1175);
                    } else {
                        trans_bsdf_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        trans_bsdf_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        trans_bsdf_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e3494 = trans_bsdf_u002e_sroa_u002e_0_u002e_0_;
                    let _e3496 = trans_bsdf_u002e_sroa_u002e_5_u002e_0_;
                    let _e3498 = trans_bsdf_u002e_sroa_u002e_8_u002e_0_;
                    if !((_e954 <= 0f)) {
                        let _e3503 = length(vec3<f32>((_e1118 - _e3029), (_e1119 - _e3030), (_e1120 - _e3031)));
                        let _e3505 = sqrt(3f);
                        let _e3528 = -(_e3503);
                        let _e3536 = (max(((clamp(((_e395 * _e2814) + (((_e885 + max((1f - _e2983), 0f)) - (_e3503 * _e3503)) * 0.5f)), -1f, 1f) * bitcast<f32>(_e577)) + 1f), 0f) * 0.31830987f);
                        ss_bsdf_u002e_sroa_u002e_0_u002e_0_ = (((exp((((1f / max((_e1208 * bitcast<f32>(_e572)), 0.000001f)) * _e3505) * _e3528)) * _e3536) * _e968) * _e954);
                        ss_bsdf_u002e_sroa_u002e_5_u002e_0_ = (((exp((((1f / max((_e1208 * bitcast<f32>(_e597)), 0.000001f)) * _e3505) * _e3528)) * _e3536) * _e969) * _e954);
                        ss_bsdf_u002e_sroa_u002e_8_u002e_0_ = (((exp((((1f / max((_e1208 * bitcast<f32>(_e602)), 0.000001f)) * _e3505) * _e3528)) * _e3536) * _e970) * _e954);
                    } else {
                        ss_bsdf_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        ss_bsdf_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        ss_bsdf_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e3556 = ss_bsdf_u002e_sroa_u002e_0_u002e_0_;
                    let _e3558 = ss_bsdf_u002e_sroa_u002e_5_u002e_0_;
                    let _e3560 = ss_bsdf_u002e_sroa_u002e_8_u002e_0_;
                    lo_u002e_sroa_u002e_0_u002e_1_ = (_e1267 + (_e2095 * (_e2814 * (((_e3086 + _e3353) + (_e3398 + _e3494)) + _e3556))));
                    lo_u002e_sroa_u002e_6_u002e_1_ = (_e1265 + (_e2097 * (_e2814 * (((_e3087 + _e3355) + (_e3400 + _e3496)) + _e3558))));
                    lo_u002e_sroa_u002e_9_u002e_1_ = (_e1263 + (_e2099 * (_e2814 * (((_e3088 + _e3357) + (_e3402 + _e3498)) + _e3560))));
                    spec_acc_u002e_sroa_u002e_0_u002e_1_ = _e3257;
                    spec_acc_u002e_sroa_u002e_6_u002e_1_ = _e3259;
                    spec_acc_u002e_sroa_u002e_9_u002e_1_ = _e3261;
                    diff_acc_u002e_sroa_u002e_0_u002e_1_ = _e3263;
                    diff_acc_u002e_sroa_u002e_6_u002e_1_ = _e3265;
                    diff_acc_u002e_sroa_u002e_9_u002e_1_ = _e3267;
                } else {
                    lo_u002e_sroa_u002e_0_u002e_1_ = _e1267;
                    lo_u002e_sroa_u002e_6_u002e_1_ = _e1265;
                    lo_u002e_sroa_u002e_9_u002e_1_ = _e1263;
                    spec_acc_u002e_sroa_u002e_0_u002e_1_ = _e1261;
                    spec_acc_u002e_sroa_u002e_6_u002e_1_ = _e1259;
                    spec_acc_u002e_sroa_u002e_9_u002e_1_ = _e1257;
                    diff_acc_u002e_sroa_u002e_0_u002e_1_ = _e1255;
                    diff_acc_u002e_sroa_u002e_6_u002e_1_ = _e1253;
                    diff_acc_u002e_sroa_u002e_9_u002e_1_ = _e1251;
                }
                let _e3601 = lo_u002e_sroa_u002e_0_u002e_1_;
                out_u002e_lo_u002e_sroa_u002e_0_u002e_1_ = _e3601;
                let _e3604 = lo_u002e_sroa_u002e_6_u002e_1_;
                out_u002e_lo_u002e_sroa_u002e_6_u002e_1_ = _e3604;
                let _e3607 = lo_u002e_sroa_u002e_9_u002e_1_;
                out_u002e_lo_u002e_sroa_u002e_9_u002e_1_ = _e3607;
                let _e3610 = spec_acc_u002e_sroa_u002e_0_u002e_1_;
                out_u002e_spec_acc_u002e_sroa_u002e_0_u002e_1_ = _e3610;
                let _e3613 = spec_acc_u002e_sroa_u002e_6_u002e_1_;
                out_u002e_spec_acc_u002e_sroa_u002e_6_u002e_1_ = _e3613;
                let _e3616 = spec_acc_u002e_sroa_u002e_9_u002e_1_;
                out_u002e_spec_acc_u002e_sroa_u002e_9_u002e_1_ = _e3616;
                let _e3619 = diff_acc_u002e_sroa_u002e_0_u002e_1_;
                out_u002e_diff_acc_u002e_sroa_u002e_0_u002e_1_ = _e3619;
                let _e3622 = diff_acc_u002e_sroa_u002e_6_u002e_1_;
                out_u002e_diff_acc_u002e_sroa_u002e_6_u002e_1_ = _e3622;
                let _e3625 = diff_acc_u002e_sroa_u002e_9_u002e_1_;
                out_u002e_diff_acc_u002e_sroa_u002e_9_u002e_1_ = _e3625;
                let _e3628 = (_e1249 + 1u);
                if (_e3628 == _e1062) {
                    lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e3601;
                    lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e3604;
                    lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e3607;
                    spec_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e3610;
                    spec_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e3613;
                    spec_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e3616;
                    diff_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e3619;
                    diff_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e3622;
                    diff_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e3625;
                    shadow_vis_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e2730;
                    break;
                } else {
                    shadow_vis_u002e_sroa_u002e_0_u002e_0_ = _e2730;
                    k_u002e_sroa_u002e_0_u002e_0_ = _e3628;
                    diff_acc_u002e_sroa_u002e_9_ = _e3625;
                    diff_acc_u002e_sroa_u002e_6_ = _e3622;
                    diff_acc_u002e_sroa_u002e_0_ = _e3619;
                    spec_acc_u002e_sroa_u002e_9_ = _e3616;
                    spec_acc_u002e_sroa_u002e_6_ = _e3613;
                    spec_acc_u002e_sroa_u002e_0_ = _e3610;
                    lo_u002e_sroa_u002e_9_ = _e3607;
                    lo_u002e_sroa_u002e_6_ = _e3604;
                    lo_u002e_sroa_u002e_0_ = _e3601;
                    continue;
                }
            }
        }
        let _e3690 = lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
        let _e3692 = lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
        let _e3694 = lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
        let _e3696 = spec_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
        let _e3698 = spec_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
        let _e3700 = spec_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
        let _e3702 = diff_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
        let _e3704 = diff_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
        let _e3706 = diff_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
        let _e3708 = shadow_vis_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
        let _e3718 = uniform_struct_0_1_.f0_.x;
        let _e3722 = uniform_struct_0_1_.f0_.y;
        let _e3726 = uniform_struct_0_1_.f0_.z;
        let _e3730 = (_e28.x * (mix(_e962, (_e962 * _e953), _e951) * _e3718));
        let _e3731 = (_e28.x * (mix(_e963, (_e963 * _e953), _e951) * _e3722));
        let _e3732 = (_e28.x * (mix(_e964, (_e964 * _e953), _e951) * _e3726));
        let _e3733 = bitcast<f32>(_e687);
        if !((_e3733 <= 0f)) {
            let _e3740 = (_e3733 * bitcast<f32>(_e692));
            let _e3741 = (_e3733 * bitcast<f32>(_e697));
            let _e3742 = (_e3733 * bitcast<f32>(_e702));
            let _e3752 = ((1f - _e949) + ((1f - pow((1f - _e395), 5f)) * _e949));
            emission_u002e_sroa_u002e_0_u002e_1_ = (_e3740 + ((((_e3740 * bitcast<f32>(_e652)) * _e3752) - _e3740) * _e949));
            emission_u002e_sroa_u002e_8_u002e_1_ = (_e3741 + ((((_e3741 * bitcast<f32>(_e657)) * _e3752) - _e3741) * _e949));
            emission_u002e_sroa_u002e_11_u002e_1_ = (_e3742 + ((((_e3742 * bitcast<f32>(_e662)) * _e3752) - _e3742) * _e949));
        } else {
            emission_u002e_sroa_u002e_0_u002e_1_ = 0f;
            emission_u002e_sroa_u002e_8_u002e_1_ = 0f;
            emission_u002e_sroa_u002e_11_u002e_1_ = 0f;
        }
        let _e3781 = emission_u002e_sroa_u002e_0_u002e_1_;
        let _e3783 = emission_u002e_sroa_u002e_8_u002e_1_;
        let _e3785 = emission_u002e_sroa_u002e_11_u002e_1_;
        let _e3786 = bitcast<f32>(_e482);
        let _e3791 = ((_e3786 + -1f) / (_e3786 + 1f));
        let _e3792 = (_e3791 * _e3791);
        let _e3800 = (dot(vec3<f32>(_e237, _e238, _e239), vec3<f32>(_e386, _e387, _e388)) * 2f);
        let _e3804 = ((_e237 * _e3800) - _e386);
        let _e3805 = ((_e238 * _e3800) - _e387);
        let _e3806 = ((_e239 * _e3800) - _e388);
        let _e3809 = uniform_struct_0_1_.f4_;
        let _e3810 = (_e474 * _e3809);
        let _e3814 = textureSampleLevel(tex_0_13_, sampler_0_16_, vec3<f32>(_e3804, _e3805, _e3806), _e3810);
        let _e3819 = textureSampleLevel(tex_0_13_, sampler_0_16_, vec3<f32>(_e3804, _e3805, _e3806), _e3810);
        let _e3824 = textureSampleLevel(tex_0_13_, sampler_0_16_, vec3<f32>(_e3804, _e3805, _e3806), _e3810);
        let _e3829 = textureSample(tex_0_14_, sampler_0_16_, vec3<f32>(_e237, _e238, _e239));
        let _e3834 = textureSample(tex_0_14_, sampler_0_16_, vec3<f32>(_e237, _e238, _e239));
        let _e3839 = textureSample(tex_0_14_, sampler_0_16_, vec3<f32>(_e237, _e238, _e239));
        let _e3844 = textureSample(tex_0_15_, sampler_0_16_, vec2<f32>(_e395, _e474));
        let _e3849 = textureSample(tex_0_15_, sampler_0_16_, vec2<f32>(_e395, _e474));
        let _e3853 = uniform_struct_0_1_.f3_;
        let _e3855 = (1f - _e951);
        let _e3865 = (_e3814.x * ((mix(_e3792, _e965, _e951) * _e3844.x) + _e3849.y));
        let _e3866 = (_e3819.y * ((mix(_e3792, _e966, _e951) * _e3844.x) + _e3849.y));
        let _e3867 = (_e3824.z * ((mix(_e3792, _e967, _e951) * _e3844.x) + _e3849.y));
        let _e3871 = max((1f - _e3849.y), 0f);
        let _e3884 = (_e28.x * (_e3853 * (_e3865 + (((_e3855 * _e962) * _e3829.x) * _e3871))));
        let _e3885 = (_e28.x * (_e3853 * (_e3866 + (((_e3855 * _e963) * _e3834.y) * _e3871))));
        let _e3886 = (_e28.x * (_e3853 * (_e3867 + (((_e3855 * _e964) * _e3839.z) * _e3871))));
        let _e3887 = (_e3853 * _e3865);
        let _e3888 = (_e3853 * _e3866);
        let _e3889 = (_e3853 * _e3867);
        let _e3902 = !((bitcast<f32>(_e717) >= 0.5f));
        let _e3903 = select(_e3781, (((_e3690 + _e3730) + _e3781) + _e3884), _e3902);
        let _e3904 = select(_e3783, (((_e3692 + _e3731) + _e3783) + _e3885), _e3902);
        let _e3905 = select(_e3785, (((_e3694 + _e3732) + _e3785) + _e3886), _e3902);
        let _e3909 = ((_e3903 <= _e3903) && (0f <= 0f));
        if _e3909 {
            if (abs(_e3903) < 100000000000000000000f) {
                _2_u002e_sroa_u002e_0_u002e_0_ = _e3903;
            } else {
                _2_u002e_sroa_u002e_0_u002e_0_ = 0f;
            }
        } else {
            _2_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
        let _e3919 = _2_u002e_sroa_u002e_0_u002e_0_;
        let _e3923 = ((_e3904 <= _e3904) && (0f <= 0f));
        if _e3923 {
            if (abs(_e3904) < 100000000000000000000f) {
                _3_u002e_sroa_u002e_0_u002e_0_ = _e3904;
            } else {
                _3_u002e_sroa_u002e_0_u002e_0_ = 0f;
            }
        } else {
            _3_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
        let _e3933 = _3_u002e_sroa_u002e_0_u002e_0_;
        let _e3937 = ((_e3905 <= _e3905) && (0f <= 0f));
        if _e3937 {
            if (abs(_e3905) < 100000000000000000000f) {
                _4_u002e_sroa_u002e_0_u002e_0_ = _e3905;
            } else {
                _4_u002e_sroa_u002e_0_u002e_0_ = 0f;
            }
        } else {
            _4_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
        let _e3947 = _4_u002e_sroa_u002e_0_u002e_0_;
        if _e3909 {
            if !((abs(_e3903) < 100000000000000000000f)) {
                x_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
            } else {
                x_bad_u002e_sroa_u002e_0_u002e_0_ = 0f;
            }
        } else {
            x_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
        }
        let _e3968 = x_bad_u002e_sroa_u002e_0_u002e_0_;
        if _e3923 {
            if !((abs(_e3904) < 100000000000000000000f)) {
                y_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
            } else {
                y_bad_u002e_sroa_u002e_0_u002e_0_ = 0f;
            }
        } else {
            y_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
        }
        let _e3980 = y_bad_u002e_sroa_u002e_0_u002e_0_;
        if _e3937 {
            if !((abs(_e3905) < 100000000000000000000f)) {
                z_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
            } else {
                z_bad_u002e_sroa_u002e_0_u002e_0_ = 0f;
            }
        } else {
            z_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
        }
        let _e3992 = z_bad_u002e_sroa_u002e_0_u002e_0_;
        let _e3994 = select(_e3992, _e3980, (_e3980 > _e3992));
        let _e3996 = select(_e3994, _e3968, (_e3968 > _e3994));
        let _e4007 = select(0f, 1f, (dot(vec3<f32>(_e3919, _e3933, _e3947), vec3<f32>(0.2126f, 0.7152f, 0.0722f)) > 1f));
        let _e4041 = (select(0f, 1f, (_e59 == 4u)) * _e474);
        let _e4045 = (select(0f, 1f, (_e59 == 5u)) * _e951);
        let _e4049 = (select(0f, 1f, (_e59 == 6u)) * _e3708);
        let _e4056 = ((_e4049 + (_e4045 + (_e4041 + ((_e96 * ((_e3702 + _e3730) + (_e3884 - _e3887))) + ((_e91 * _e3887) + ((_e86 * _e3696) + (_e81 * _e3919))))))) + (_e116 * mix(mix(_e3919, 1f, _e4007), 1f, _e3996)));
        let _e4057 = ((_e4049 + (_e4045 + (_e4041 + ((_e96 * ((_e3704 + _e3731) + (_e3885 - _e3888))) + ((_e91 * _e3888) + ((_e86 * _e3698) + (_e81 * _e3933))))))) + (_e116 * mix(mix(_e3933, 0f, _e4007), 0f, _e3996)));
        let _e4058 = ((_e4049 + (_e4045 + (_e4041 + ((_e96 * ((_e3706 + _e3732) + (_e3886 - _e3889))) + ((_e91 * _e3889) + ((_e86 * _e3700) + (_e81 * _e3947))))))) + (_e116 * mix(mix(_e3947, 0f, _e4007), 1f, _e3996)));
        let _e4066 = (((_e4056 * 0.2126f) + (_e4057 * 0.7152f)) + (_e4058 * 0.0722f));
        let _e4073 = (_e244 / (select(_e4066, 0f, (_e4066 < 0f)) + 1f));
        let _e4077 = (_e172 + (_e4056 * _e4073));
        out_u002e_3_ = _e4077;
        let _e4079 = (_e174 + (_e4057 * _e4073));
        out_u002e_6_ = _e4079;
        let _e4081 = (_e176 + (_e4058 * _e4073));
        out_u002e_9_ = _e4081;
        let _e4084 = (_e154 + (_e244 * _e718));
        out_u002e_163_ = _e4084;
        let _e4086 = (_e156 + _e4073);
        out_u002e_164_ = _e4086;
        let _e4092 = select(0f, 1f, (_e170 == 0u));
        let _e4093 = mix(_e178, _e4056, _e4092);
        out_u002e_0_ = _e4093;
        let _e4095 = mix(_e180, _e4057, _e4092);
        out_u002e_0_1 = _e4095;
        let _e4097 = mix(_e182, _e4058, _e4092);
        out_u002e_0_2 = _e4097;
        let _e4101 = (_e158 + (_e4092 * (_e718 - _e158)));
        out_u002e_0_3 = _e4101;
        let _e4104 = (_e170 + 1u);
        if (_e4104 == 4u) {
            break;
        } else {
            alpha_acc_u002e_sroa_u002e_0_ = _e4084;
            weight_u002e_sroa_u002e_0_ = _e4086;
            single_alpha_u002e_sroa_u002e_0_ = _e4101;
            covered_count_u002e_sroa_u002e_0_ = _e286;
            have_ref_u002e_sroa_u002e_0_ = select(_e162, _e244, (_e244 > _e162));
            disagree_acc_u002e_sroa_u002e_0_ = _e276;
            ref_depth_u002e_sroa_u002e_0_ = (_e166 + (_e251 * _e250));
            ref_id_u002e_sroa_u002e_0_ = select(_e168, _e198.x, _e247);
            s_u002e_sroa_u002e_0_ = _e4104;
            acc_u002e_sroa_u002e_0_ = _e4077;
            acc_u002e_sroa_u002e_4_ = _e4079;
            acc_u002e_sroa_u002e_7_ = _e4081;
            single_u002e_sroa_u002e_0_ = _e4093;
            single_u002e_sroa_u002e_4_ = _e4095;
            single_u002e_sroa_u002e_7_ = _e4097;
            ref_normal_u002e_sroa_u002e_0_ = mix(_e184, _e237, _e250);
            ref_normal_u002e_sroa_u002e_4_ = mix(_e186, _e238, _e250);
            ref_normal_u002e_sroa_u002e_6_ = mix(_e188, _e239, _e250);
            continue;
        }
    }
    let _e4126 = out_u002e_16_;
    let _e4128 = out_u002e_15_;
    let _e4130 = out_u002e_164_;
    let _e4132 = out_u002e_3_;
    let _e4134 = out_u002e_6_;
    let _e4136 = out_u002e_9_;
    let _e4138 = out_u002e_0_;
    let _e4140 = out_u002e_0_1;
    let _e4142 = out_u002e_0_2;
    let _e4144 = out_u002e_163_;
    let _e4146 = out_u002e_0_3;
    let _e4151 = select(0f, 1f, (_e4126 == 0f));
    let _e4164 = select(0f, 1f, select(false, (select(1f, _e4128, (_e4128 < 1f)) == 0f), (_e4126 == 4f)));
    let _e4168 = (1f - select(_e4151, _e4164, (_e4164 > _e4151)));
    let _e4172 = select(_e4130, 0.000001f, (_e4130 < 0.000001f));
    let _e4174 = (1f / _e4172);
    fragment_out = vec4<f32>(mix(_e4138, (_e4132 * _e4174), _e4168), mix(_e4140, (_e4134 * _e4174), _e4168), mix(_e4142, (_e4136 * _e4174), _e4168), (_e4146 + (((_e4144 / _e4172) - _e4146) * _e4168)));
    let _e4188 = fragment_out;
    return _e4188;
}
