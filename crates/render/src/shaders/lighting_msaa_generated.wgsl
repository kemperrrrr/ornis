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
    let _e11 = u32((uv.x * f32(bitcast<i32>(_e5.x))));
    let _e15 = u32((uv.y * f32(bitcast<i32>(_e5.y))));
    let _e16 = vec2<f32>(uv.x, uv.y);
    let _e20 = textureSampleLevel(tex_0_3_, sampler_0_9_, _e16, 0f);
    let _e24 = textureSampleLevel(tex_0_17_, sampler_0_9_, _e16, 0f);
    let _e28 = textureSampleLevel(tex_0_6_, sampler_0_9_, _e16, 0f);
    let _e32 = textureSampleLevel(tex_0_7_, sampler_0_9_, _e16, 0f);
    let _e37 = uniform_struct_0_0_.f8_.x;
    let _e41 = uniform_struct_0_0_.f8_.y;
    let _e45 = uniform_struct_0_0_.f8_.z;
    let _e48 = uniform_struct_0_19_.f2_;
    let _e51 = uniform_struct_0_1_.f5_;
    let _e54 = uniform_struct_0_1_.f2_;
    if (_e20.w < 0.001f) {
        discard;
    }
    let _e60 = ((uv.x * 2f) + -1f);
    let _e64 = (1f - (uv.y * 2f));
    let _e73 = select(0f, 1f, (_e51 == 0u));
    let _e78 = select(0f, 1f, (_e51 == 1u));
    let _e83 = select(0f, 1f, (_e51 == 2u));
    let _e88 = select(0f, 1f, (_e51 == 3u));
    let _e108 = select(0f, 1f, (_e51 == 7u));
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
        let _e146 = alpha_acc_u002e_sroa_u002e_0_;
        let _e148 = weight_u002e_sroa_u002e_0_;
        let _e150 = single_alpha_u002e_sroa_u002e_0_;
        let _e152 = covered_count_u002e_sroa_u002e_0_;
        let _e154 = have_ref_u002e_sroa_u002e_0_;
        let _e156 = disagree_acc_u002e_sroa_u002e_0_;
        let _e158 = ref_depth_u002e_sroa_u002e_0_;
        let _e160 = ref_id_u002e_sroa_u002e_0_;
        let _e162 = s_u002e_sroa_u002e_0_;
        let _e164 = acc_u002e_sroa_u002e_0_;
        let _e166 = acc_u002e_sroa_u002e_4_;
        let _e168 = acc_u002e_sroa_u002e_7_;
        let _e170 = single_u002e_sroa_u002e_0_;
        let _e172 = single_u002e_sroa_u002e_4_;
        let _e174 = single_u002e_sroa_u002e_7_;
        let _e176 = ref_normal_u002e_sroa_u002e_0_;
        let _e178 = ref_normal_u002e_sroa_u002e_4_;
        let _e180 = ref_normal_u002e_sroa_u002e_6_;
        let _e185 = textureLoad(tex_0_8_, vec2<i32>(bitcast<i32>(_e11), bitcast<i32>(_e15)), _e162);
        let _e190 = textureLoad(tex_0_5_, vec2<i32>(bitcast<i32>(_e11), bitcast<i32>(_e15)), _e162);
        let _e196 = textureLoad(tex_0_4_, vec2<i32>(bitcast<i32>(_e11), bitcast<i32>(_e15)), _e162);
        let _e201 = textureLoad(tex_0_4_, vec2<i32>(bitcast<i32>(_e11), bitcast<i32>(_e15)), _e162);
        let _e207 = ((1f - abs(_e196.x)) - abs(_e201.y));
        let _e212 = select(0f, -(_e207), (_e207 < 0f));
        let _e216 = -(_e212);
        let _e222 = (_e196.x + select(_e216, _e212, !((_e196.x >= 0f))));
        let _e223 = (_e201.y + select(_e216, _e212, !((_e201.y >= 0f))));
        let _e227 = (1f / length(vec3<f32>(_e222, _e223, _e207)));
        let _e228 = (_e227 * _e222);
        let _e229 = (_e227 * _e223);
        let _e230 = (_e207 * _e227);
        let _e232 = (_e185 < 1f);
        let _e235 = select(0f, 1f, _e232);
        let _e238 = ((_e154 == 0f) & _e232);
        let _e241 = select(0f, 1f, _e238);
        let _e242 = (_e185 - _e158);
        let _e248 = select(0f, 1f, (abs(_e242) > 0.002f));
        let _e256 = select(0f, 1f, (dot(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e176, _e178, _e180)) < 0.999f));
        let _e260 = select(1f, 0f, (_e190.x == _e160));
        let _e262 = select(_e256, _e260, (_e260 > _e256));
        let _e267 = (_e156 + (_e154 * (_e235 * select(_e248, _e262, (_e262 > _e248)))));
        out_u002e_15_ = _e267;
        let _e277 = (_e152 + _e235);
        out_u002e_16_ = _e277;
        let _e282 = uniform_struct_0_0_.f4_.w;
        let _e286 = uniform_struct_0_0_.f5_.w;
        let _e290 = uniform_struct_0_0_.f6_.w;
        let _e294 = uniform_struct_0_0_.f7_.w;
        let _e300 = (_e294 + (((_e60 * _e282) + (_e64 * _e286)) + (_e185 * _e290)));
        let _e304 = uniform_struct_0_0_.f4_.x;
        let _e308 = uniform_struct_0_0_.f5_.x;
        let _e312 = uniform_struct_0_0_.f6_.x;
        let _e316 = uniform_struct_0_0_.f7_.x;
        let _e326 = uniform_struct_0_0_.f4_.y;
        let _e330 = uniform_struct_0_0_.f5_.y;
        let _e334 = uniform_struct_0_0_.f6_.y;
        let _e338 = uniform_struct_0_0_.f7_.y;
        let _e348 = uniform_struct_0_0_.f4_.z;
        let _e352 = uniform_struct_0_0_.f5_.z;
        let _e356 = uniform_struct_0_0_.f6_.z;
        let _e360 = uniform_struct_0_0_.f7_.z;
        let _e367 = ((_e316 + (((_e60 * _e304) + (_e64 * _e308)) + (_e185 * _e312))) / _e300);
        let _e368 = ((_e338 + (((_e60 * _e326) + (_e64 * _e330)) + (_e185 * _e334))) / _e300);
        let _e369 = ((_e360 + (((_e60 * _e348) + (_e64 * _e352)) + (_e185 * _e356))) / _e300);
        let _e370 = (_e37 - _e367);
        let _e371 = (_e41 - _e368);
        let _e372 = (_e45 - _e369);
        let _e376 = (1f / length(vec3<f32>(_e370, _e371, _e372)));
        let _e377 = (_e370 * _e376);
        let _e378 = (_e371 * _e376);
        let _e379 = (_e376 * _e372);
        let _e382 = dot(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e377, _e378, _e379));
        let _e384 = (_e382 > 0.000001f);
        let _e386 = select(0.000001f, _e382, _e384);
        let _e388 = (_e190.x * 80u);
        let _e391 = storage_0_2_[_e388];
        let _e396 = storage_0_2_[(_e388 | 1u)];
        let _e401 = storage_0_2_[(_e388 | 2u)];
        let _e406 = storage_0_2_[(_e388 | 4u)];
        let _e411 = storage_0_2_[(_e388 | 5u)];
        let _e416 = storage_0_2_[(_e388 | 6u)];
        let _e421 = storage_0_2_[(_e388 | 8u)];
        let _e423 = (_e388 | 9u);
        let _e426 = storage_0_2_[_e423];
        let _e427 = bitcast<f32>(_e426);
        let _e428 = dpdx(_e228);
        let _e429 = dpdx(_e229);
        let _e430 = dpdx(_e230);
        let _e431 = dpdy(_e228);
        let _e432 = dpdy(_e229);
        let _e433 = dpdy(_e230);
        let _e444 = (((dot(vec3<f32>(_e428, _e429, _e430), vec3<f32>(_e428, _e429, _e430)) + dot(vec3<f32>(_e431, _e432, _e433), vec3<f32>(_e431, _e432, _e433))) * 0.15f) * 2f);
        let _e449 = (_e427 * _e427);
        let _e451 = ((_e449 * _e449) + select(0.2f, _e444, (_e444 < 0.2f)));
        let _e455 = select(_e451, 0f, (_e451 < 0f));
        let _e459 = select(_e455, 1f, (_e455 > 1f));
        let _e465 = sqrt(sqrt(select(0f, _e459, (_e459 > 0f))));
        let _e473 = storage_0_2_[(_e388 | 10u)];
        let _e478 = storage_0_2_[(_e388 | 11u)];
        let _e483 = storage_0_2_[(_e388 | 12u)];
        let _e488 = storage_0_2_[(_e388 | 13u)];
        let _e493 = storage_0_2_[(_e388 | 14u)];
        let _e498 = storage_0_2_[(_e388 + 16u)];
        let _e503 = storage_0_2_[(_e388 + 17u)];
        let _e518 = storage_0_2_[(_e388 + 20u)];
        let _e523 = storage_0_2_[(_e388 + 21u)];
        let _e528 = storage_0_2_[(_e388 + 22u)];
        let _e553 = storage_0_2_[(_e388 + 28u)];
        let _e558 = storage_0_2_[(_e388 + 29u)];
        let _e563 = storage_0_2_[(_e388 + 30u)];
        let _e568 = storage_0_2_[(_e388 + 31u)];
        let _e573 = storage_0_2_[(_e388 + 32u)];
        let _e578 = storage_0_2_[(_e388 + 33u)];
        let _e583 = storage_0_2_[(_e388 + 34u)];
        let _e588 = storage_0_2_[(_e388 + 36u)];
        let _e593 = storage_0_2_[(_e388 + 37u)];
        let _e598 = storage_0_2_[(_e388 + 40u)];
        let _e603 = storage_0_2_[(_e388 + 41u)];
        let _e608 = storage_0_2_[(_e388 + 44u)];
        let _e613 = storage_0_2_[(_e388 + 45u)];
        let _e618 = storage_0_2_[(_e388 + 46u)];
        let _e623 = storage_0_2_[(_e388 + 48u)];
        let _e628 = storage_0_2_[(_e388 + 49u)];
        let _e633 = storage_0_2_[(_e388 + 50u)];
        let _e638 = storage_0_2_[(_e388 + 51u)];
        let _e643 = storage_0_2_[(_e388 + 52u)];
        let _e648 = storage_0_2_[(_e388 + 53u)];
        let _e653 = storage_0_2_[(_e388 + 54u)];
        let _e658 = storage_0_2_[(_e388 + 56u)];
        let _e663 = storage_0_2_[(_e388 + 60u)];
        let _e668 = storage_0_2_[(_e388 + 61u)];
        let _e673 = storage_0_2_[(_e388 + 62u)];
        let _e678 = storage_0_2_[(_e388 + 64u)];
        let _e683 = storage_0_2_[(_e388 + 68u)];
        let _e688 = storage_0_2_[(_e388 + 69u)];
        let _e693 = storage_0_2_[(_e388 + 70u)];
        let _e698 = storage_0_2_[(_e388 + 72u)];
        let _e708 = storage_0_2_[(_e388 + 74u)];
        let _e709 = bitcast<f32>(_e698);
        let _e732 = (cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(0f, 1f, 0f)).x + 0f);
        let _e734 = (cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(0f, 1f, 0f)).y + 0f);
        let _e736 = (cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(0f, 1f, 0f)).z + 0.000001f);
        let _e740 = (1f / length(vec3<f32>(_e732, _e734, _e736)));
        let _e741 = (_e732 * _e740);
        let _e742 = (_e734 * _e740);
        let _e743 = (_e736 * _e740);
        let _e759 = uniform_struct_0_0_.f0_.x;
        let _e763 = uniform_struct_0_0_.f1_.x;
        let _e767 = uniform_struct_0_0_.f2_.x;
        let _e771 = uniform_struct_0_0_.f3_.x;
        let _e781 = uniform_struct_0_0_.f0_.y;
        let _e785 = uniform_struct_0_0_.f1_.y;
        let _e789 = uniform_struct_0_0_.f2_.y;
        let _e793 = uniform_struct_0_0_.f3_.y;
        let _e803 = uniform_struct_0_0_.f0_.w;
        let _e807 = uniform_struct_0_0_.f1_.w;
        let _e811 = uniform_struct_0_0_.f2_.w;
        let _e815 = uniform_struct_0_0_.f3_.w;
        let _e821 = (_e815 + (((_e367 * _e803) + (_e368 * _e807)) + (_e369 * _e811)));
        let _e825 = select(0.5f, _e821, (_e821 > 0.5f));
        let _e866 = ((((min(u32(floor((((((_e793 + (((_e367 * _e781) + (_e368 * _e785)) + (_e369 * _e789))) / _e825) * 0.5f) + 0.5f) * 9f))), 8u) << 4u) | min(u32(floor((((((_e771 + (((_e367 * _e759) + (_e368 * _e763)) + (_e369 * _e767))) / _e825) * 0.5f) + 0.5f) * 16f))), 15u)) + (min(u32(ceil(((log((_e825 * 2f)) * 3.4407384f) + -0.0001f))), 23u) * 144u)) << 1u);
        let _e868 = bitcast<f32>(_e673);
        let _e872 = (_e386 * _e386);
        let _e876 = max((1f - _e872), 0f);
        let _e878 = (sqrt(_e876) / _e868);
        let _e891 = pow(((_e868 + -1f) / (_e868 + 1f)), 2f);
        let _e895 = (((bitcast<f32>(_e668) * 1000f) * (_e868 * 12.566371f)) * sqrt(max((1f - (_e878 * _e878)), 0f)));
        let _e906 = (_e891 * 2f);
        let _e912 = (1f - (_e891 * _e891));
        let _e924 = clamp(bitcast<f32>(_e663), 0f, 1f);
        let _e929 = ((_e924 * ((((_e906 * cos((_e895 / 650f))) / _e912) + 1f) + -1f)) + 1f);
        let _e934 = ((_e924 * ((((_e906 * cos((_e895 / 550f))) / _e912) + 1f) + -1f)) + 1f);
        let _e939 = ((_e924 * ((((_e906 * cos((_e895 / 450f))) / _e912) + 1f) + -1f)) + 1f);
        let _e940 = bitcast<f32>(_e623);
        let _e941 = bitcast<f32>(_e658);
        let _e942 = bitcast<f32>(_e401);
        let _e943 = bitcast<f32>(_e391);
        let _e944 = bitcast<f32>(_e421);
        let _e945 = bitcast<f32>(_e553);
        let _e950 = ((_e941 + -1f) / (_e941 + 1f));
        let _e953 = bitcast<f32>(_e406);
        let _e954 = bitcast<f32>(_e411);
        let _e955 = bitcast<f32>(_e416);
        let _e956 = (_e943 * _e953);
        let _e957 = (_e943 * _e954);
        let _e958 = (_e943 * _e955);
        let _e959 = bitcast<f32>(_e573);
        let _e960 = bitcast<f32>(_e578);
        let _e961 = bitcast<f32>(_e583);
        let _e964 = !((_e940 <= 0f));
        if _e964 {
            let _e970 = (1f - ((1f - pow(_e950, 2f)) / (_e941 * _e941)));
            let _e972 = (1f - _e970);
            let _e973 = (_e958 * _e944);
            let _e986 = (_e957 * _e944);
            let _e999 = (_e956 * _e944);
            let _e1013 = (_e940 * bitcast<f32>(_e638));
            _15_u002e_sroa_u002e_0_u002e_0_ = ((_e1013 * ((_e972 / max((1f - ((_e999 + (((((_e953 - _e959) * _e945) + _e959) - _e999) * _e942)) * _e970)), 0.000001f)) + -1f)) + 1f);
            _15_u002e_sroa_u002e_5_u002e_0_ = ((_e1013 * ((_e972 / max((1f - ((_e986 + (((((_e954 - _e960) * _e945) + _e960) - _e986) * _e942)) * _e970)), 0.000001f)) + -1f)) + 1f);
            _15_u002e_sroa_u002e_8_u002e_0_ = ((_e1013 * ((_e972 / max((1f - ((_e973 + (((((_e955 - _e961) * _e945) + _e961) - _e973) * _e942)) * _e970)), 0.000001f)) + -1f)) + 1f);
        } else {
            _15_u002e_sroa_u002e_0_u002e_0_ = 1f;
            _15_u002e_sroa_u002e_5_u002e_0_ = 1f;
            _15_u002e_sroa_u002e_8_u002e_0_ = 1f;
        }
        let _e1039 = _15_u002e_sroa_u002e_0_u002e_0_;
        let _e1041 = _15_u002e_sroa_u002e_5_u002e_0_;
        let _e1043 = _15_u002e_sroa_u002e_8_u002e_0_;
        let _e1046 = storage_0_22_[_e866];
        let _e1051 = storage_0_22_[(_e866 | 1u)];
        let _e1053 = ((_e1051 * _e48) + _e54);
        if (_e1053 == 0u) {
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
            let _e1084 = bitcast<f32>(_e396);
            let _e1085 = bitcast<f32>(_e473);
            let _e1091 = ((_e1085 + -1f) / (_e1085 + 1f));
            let _e1092 = (_e1091 * _e1091);
            let _e1093 = bitcast<f32>(_e483);
            let _e1094 = bitcast<f32>(_e488);
            let _e1095 = bitcast<f32>(_e493);
            let _e1096 = (_e465 * _e465);
            let _e1098 = (1f - bitcast<f32>(_e478));
            let _e1103 = (2f / ((_e1098 * _e1098) + 1f));
            let _e1105 = select(_e465, _e1084, (_e465 < _e1084));
            let _e1109 = (_e377 - (_e228 * _e386));
            let _e1110 = (_e378 - (_e229 * _e386));
            let _e1111 = (_e379 - (_e230 * _e386));
            let _e1114 = (1f - _e942);
            let _e1119 = max((_e1105 * _e1105), 0.000001f);
            let _e1120 = (_e1119 * _e1119);
            let _e1136 = bitcast<f32>(_e628);
            let _e1137 = (_e950 * _e950);
            let _e1140 = (_e1136 * _e1136);
            let _e1142 = (1f - bitcast<f32>(_e633));
            let _e1147 = (2f / ((_e1142 * _e1142) + 1f));
            let _e1151 = bitcast<f32>(_e598);
            let _e1155 = bitcast<f32>(_e603);
            let _e1156 = (_e1155 * _e1155);
            let _e1166 = bitcast<f32>(_e498);
            let _e1167 = bitcast<f32>(_e503);
            let _e1199 = bitcast<f32>(_e558);
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
                let _e1238 = shadow_vis_u002e_sroa_u002e_0_u002e_0_;
                let _e1240 = k_u002e_sroa_u002e_0_u002e_0_;
                let _e1242 = diff_acc_u002e_sroa_u002e_9_;
                let _e1244 = diff_acc_u002e_sroa_u002e_6_;
                let _e1246 = diff_acc_u002e_sroa_u002e_0_;
                let _e1248 = spec_acc_u002e_sroa_u002e_9_;
                let _e1250 = spec_acc_u002e_sroa_u002e_6_;
                let _e1252 = spec_acc_u002e_sroa_u002e_0_;
                let _e1254 = lo_u002e_sroa_u002e_9_;
                let _e1256 = lo_u002e_sroa_u002e_6_;
                let _e1258 = lo_u002e_sroa_u002e_0_;
                if (_e1240 < _e54) {
                    _0_u002e_sroa_u002e_0_u002e_0_u002e_in = _e1240;
                } else {
                    let _e1264 = storage_0_21_[(((_e1046 * _e48) - _e54) + _e1240)];
                    _0_u002e_sroa_u002e_0_u002e_0_u002e_in = _e1264;
                }
                let _e1267 = _0_u002e_sroa_u002e_0_u002e_0_u002e_in;
                let _e1269 = (_e1267 * 36u);
                let _e1272 = storage_0_20_[_e1269];
                let _e1273 = bitcast<f32>(_e1272);
                let _e1275 = (_e1269 + 8u);
                let _e1278 = storage_0_20_[_e1275];
                let _e1280 = (_e1269 + 9u);
                let _e1283 = storage_0_20_[_e1280];
                let _e1285 = (_e1269 + 10u);
                let _e1288 = storage_0_20_[_e1285];
                let _e1296 = length(vec3<f32>((bitcast<f32>(_e1278) - _e367), (bitcast<f32>(_e1283) - _e368), (bitcast<f32>(_e1288) - _e369)));
                let _e1298 = (_e1273 < 0.5f);
                let _e1301 = select(1f, 0f, _e1298);
                let _e1303 = (_e1269 + 16u);
                let _e1306 = storage_0_20_[_e1303];
                let _e1307 = bitcast<f32>(_e1306);
                let _e1319 = storage_0_20_[(_e1269 + 12u)];
                let _e1324 = storage_0_20_[(_e1269 + 13u)];
                let _e1329 = storage_0_20_[(_e1269 + 14u)];
                let _e1334 = storage_0_20_[(_e1269 + 15u)];
                let _e1335 = bitcast<f32>(_e1334);
                let _e1338 = storage_0_20_[_e1275];
                let _e1341 = storage_0_20_[_e1280];
                let _e1344 = storage_0_20_[_e1285];
                let _e1346 = (_e367 - bitcast<f32>(_e1338));
                let _e1348 = (_e368 - bitcast<f32>(_e1341));
                let _e1350 = (_e369 - bitcast<f32>(_e1344));
                let _e1354 = (1f / length(vec3<f32>(_e1346, _e1348, _e1350)));
                let _e1359 = (_e1269 + 4u);
                let _e1362 = storage_0_20_[_e1359];
                let _e1364 = (_e1269 + 5u);
                let _e1367 = storage_0_20_[_e1364];
                let _e1369 = (_e1269 + 6u);
                let _e1372 = storage_0_20_[_e1369];
                let _e1373 = bitcast<f32>(_e1362);
                let _e1374 = bitcast<f32>(_e1367);
                let _e1375 = bitcast<f32>(_e1372);
                let _e1379 = (1f / length(vec3<f32>(_e1373, _e1374, _e1375)));
                let _e1390 = storage_0_20_[(_e1269 + 18u)];
                let _e1396 = storage_0_20_[(_e1269 + 17u)];
                let _e1403 = select(1f, 0f, (_e1273 < 1.5f));
                let _e1405 = mix(1f, smoothstep(bitcast<f32>(_e1390), bitcast<f32>(_e1396), dot(vec3<f32>((_e1379 * _e1373), (_e1379 * _e1374), (_e1379 * _e1375)), vec3<f32>((_e1346 * _e1354), (_e1348 * _e1354), (_e1350 * _e1354)))), _e1403);
                let _e1406 = (_e1296 * _e1296);
                let _e1421 = (mix(1f, (1f - smoothstep((_e1307 * 0.99f), _e1307, _e1296)), _e1301) * mix(1f, (1f / select(_e1406, 0.000001f, (_e1406 < 0.000001f))), _e1301));
                let _e1425 = (_e1405 * ((bitcast<f32>(_e1319) * _e1335) * _e1421));
                let _e1426 = (_e1405 * ((bitcast<f32>(_e1324) * _e1335) * _e1421));
                let _e1427 = (_e1405 * ((bitcast<f32>(_e1329) * _e1335) * _e1421));
                let _e1431 = select((1f - _e1403), 0f, _e1298);
                if (((1f - _e1301) * select(1f, 0f, (_e1307 < 0.5f))) > 0.5f) {
                    let _e1446 = uniform_struct_0_1_.f6_.f1_.x;
                    let _e1451 = uniform_struct_0_1_.f6_.f1_.y;
                    let _e1456 = uniform_struct_0_1_.f6_.f1_.z;
                    let _e1459 = dot(vec3<f32>((_e367 - _e37), (_e368 - _e41), (_e369 - _e45)), vec3<f32>(_e1446, _e1451, _e1456));
                    let _e1464 = uniform_struct_0_1_.f6_.f2_.x;
                    let _e1469 = uniform_struct_0_1_.f6_.f2_.y;
                    let _e1474 = uniform_struct_0_1_.f6_.f2_.z;
                    let _e1479 = uniform_struct_0_1_.f6_.f2_.w;
                    let _e1483 = uniform_struct_0_1_.f6_.f3_;
                    let _e1487 = select(1f, 0f, (_e1459 < _e1469));
                    let _e1491 = select(1f, 0f, (_e1459 < _e1474));
                    let _e1495 = select(1f, 0f, (_e1459 < _e1479));
                    let _e1498 = u32(((_e1487 + _e1491) + _e1495));
                    let _e1501 = mix(mix(mix(_e1469, _e1474, _e1487), _e1479, _e1491), _e1483, _e1495);
                    let _e1511 = uniform_struct_0_1_.f6_.f0_[_e1498].f0_.x;
                    let _e1518 = uniform_struct_0_1_.f6_.f0_[_e1498].f1_.x;
                    let _e1525 = uniform_struct_0_1_.f6_.f0_[_e1498].f2_.x;
                    let _e1532 = uniform_struct_0_1_.f6_.f0_[_e1498].f3_.x;
                    let _e1545 = uniform_struct_0_1_.f6_.f0_[_e1498].f0_.y;
                    let _e1552 = uniform_struct_0_1_.f6_.f0_[_e1498].f1_.y;
                    let _e1559 = uniform_struct_0_1_.f6_.f0_[_e1498].f2_.y;
                    let _e1566 = uniform_struct_0_1_.f6_.f0_[_e1498].f3_.y;
                    let _e1579 = uniform_struct_0_1_.f6_.f0_[_e1498].f0_.z;
                    let _e1586 = uniform_struct_0_1_.f6_.f0_[_e1498].f1_.z;
                    let _e1593 = uniform_struct_0_1_.f6_.f0_[_e1498].f2_.z;
                    let _e1600 = uniform_struct_0_1_.f6_.f0_[_e1498].f3_.z;
                    let _e1613 = uniform_struct_0_1_.f6_.f0_[_e1498].f0_.w;
                    let _e1620 = uniform_struct_0_1_.f6_.f0_[_e1498].f1_.w;
                    let _e1627 = uniform_struct_0_1_.f6_.f0_[_e1498].f2_.w;
                    let _e1634 = uniform_struct_0_1_.f6_.f0_[_e1498].f3_.w;
                    let _e1640 = (_e1634 + (((_e367 * _e1613) + (_e368 * _e1620)) + (_e369 * _e1627)));
                    let _e1657 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e1532 + (((_e367 * _e1511) + (_e368 * _e1518)) + (_e369 * _e1525))) / _e1640) * 0.5f) + 0.5f), (0.5f - (((_e1566 + (((_e367 * _e1545) + (_e368 * _e1552)) + (_e369 * _e1559))) / _e1640) * 0.5f))), _e1498, (((_e1600 + (((_e367 * _e1579) + (_e368 * _e1586)) + (_e369 * _e1593))) / _e1640) + -0.001f));
                    if (_e1498 < 3u) {
                        let _e1662 = ((_e1501 - mix(mix(mix(_e1464, _e1469, _e1487), _e1474, _e1491), _e1479, _e1495)) * 0.08f);
                        let _e1668 = ((_e1501 - _e1459) / select(_e1662, 0.000001f, (_e1662 < 0.000001f)));
                        let _e1672 = select(_e1668, 0f, (_e1668 < 0f));
                        let _e1678 = (_e1498 + 1u);
                        let _e1685 = uniform_struct_0_1_.f6_.f0_[_e1678].f0_.x;
                        let _e1692 = uniform_struct_0_1_.f6_.f0_[_e1678].f1_.x;
                        let _e1699 = uniform_struct_0_1_.f6_.f0_[_e1678].f2_.x;
                        let _e1706 = uniform_struct_0_1_.f6_.f0_[_e1678].f3_.x;
                        let _e1719 = uniform_struct_0_1_.f6_.f0_[_e1678].f0_.y;
                        let _e1726 = uniform_struct_0_1_.f6_.f0_[_e1678].f1_.y;
                        let _e1733 = uniform_struct_0_1_.f6_.f0_[_e1678].f2_.y;
                        let _e1740 = uniform_struct_0_1_.f6_.f0_[_e1678].f3_.y;
                        let _e1753 = uniform_struct_0_1_.f6_.f0_[_e1678].f0_.z;
                        let _e1760 = uniform_struct_0_1_.f6_.f0_[_e1678].f1_.z;
                        let _e1767 = uniform_struct_0_1_.f6_.f0_[_e1678].f2_.z;
                        let _e1774 = uniform_struct_0_1_.f6_.f0_[_e1678].f3_.z;
                        let _e1787 = uniform_struct_0_1_.f6_.f0_[_e1678].f0_.w;
                        let _e1794 = uniform_struct_0_1_.f6_.f0_[_e1678].f1_.w;
                        let _e1801 = uniform_struct_0_1_.f6_.f0_[_e1678].f2_.w;
                        let _e1808 = uniform_struct_0_1_.f6_.f0_[_e1678].f3_.w;
                        let _e1814 = (_e1808 + (((_e367 * _e1787) + (_e368 * _e1794)) + (_e369 * _e1801)));
                        let _e1831 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e1706 + (((_e367 * _e1685) + (_e368 * _e1692)) + (_e369 * _e1699))) / _e1814) * 0.5f) + 0.5f), (0.5f - (((_e1740 + (((_e367 * _e1719) + (_e368 * _e1726)) + (_e369 * _e1733))) / _e1814) * 0.5f))), _e1678, (((_e1774 + (((_e367 * _e1753) + (_e368 * _e1760)) + (_e369 * _e1767))) / _e1814) + -0.001f));
                        vis_u002e_sroa_u002e_0_u002e_0_ = mix(_e1831, _e1657, select(_e1672, 1f, (_e1672 > 1f)));
                    } else {
                        vis_u002e_sroa_u002e_0_u002e_0_ = _e1657;
                    }
                    let _e1836 = vis_u002e_sroa_u002e_0_u002e_0_;
                    radiance_u002e_sroa_u002e_0_u002e_0_ = (_e1425 * _e1836);
                    radiance_u002e_sroa_u002e_12_u002e_0_ = (_e1426 * _e1836);
                    radiance_u002e_sroa_u002e_20_u002e_0_ = (_e1427 * _e1836);
                } else {
                    radiance_u002e_sroa_u002e_0_u002e_0_ = _e1425;
                    radiance_u002e_sroa_u002e_12_u002e_0_ = _e1426;
                    radiance_u002e_sroa_u002e_20_u002e_0_ = _e1427;
                }
                let _e1847 = radiance_u002e_sroa_u002e_0_u002e_0_;
                let _e1849 = radiance_u002e_sroa_u002e_12_u002e_0_;
                let _e1851 = radiance_u002e_sroa_u002e_20_u002e_0_;
                let _e1853 = (_e1269 + 19u);
                let _e1856 = storage_0_20_[_e1853];
                if !((bitcast<f32>(_e1856) >= 0f)) {
                    radiance_u002e_sroa_u002e_0_u002e_2_ = _e1847;
                    radiance_u002e_sroa_u002e_12_u002e_2_ = _e1849;
                    radiance_u002e_sroa_u002e_20_u002e_2_ = _e1851;
                } else {
                    if !((_e1431 <= 0.5f)) {
                        radiance_u002e_sroa_u002e_0_u002e_1_ = _e1847;
                        radiance_u002e_sroa_u002e_12_u002e_1_ = _e1849;
                        radiance_u002e_sroa_u002e_20_u002e_1_ = _e1851;
                    } else {
                        let _e1874 = storage_0_20_[(_e1269 + 20u)];
                        let _e1880 = storage_0_20_[(_e1269 + 24u)];
                        let _e1886 = storage_0_20_[(_e1269 + 28u)];
                        let _e1892 = storage_0_20_[(_e1269 + 32u)];
                        let _e1904 = storage_0_20_[(_e1269 + 21u)];
                        let _e1910 = storage_0_20_[(_e1269 + 25u)];
                        let _e1916 = storage_0_20_[(_e1269 + 29u)];
                        let _e1922 = storage_0_20_[(_e1269 + 33u)];
                        let _e1934 = storage_0_20_[(_e1269 + 22u)];
                        let _e1940 = storage_0_20_[(_e1269 + 26u)];
                        let _e1946 = storage_0_20_[(_e1269 + 30u)];
                        let _e1952 = storage_0_20_[(_e1269 + 34u)];
                        let _e1964 = storage_0_20_[(_e1269 + 23u)];
                        let _e1970 = storage_0_20_[(_e1269 + 27u)];
                        let _e1976 = storage_0_20_[(_e1269 + 31u)];
                        let _e1982 = storage_0_20_[(_e1269 + 35u)];
                        let _e1989 = ((((_e367 * bitcast<f32>(_e1964)) + (_e368 * bitcast<f32>(_e1970))) + (_e369 * bitcast<f32>(_e1976))) + bitcast<f32>(_e1982));
                        let _e2002 = storage_0_20_[_e1853];
                        let _e2011 = textureSampleCompareLevel(tex_0_10_, sampler_0_11_, vec2<f32>((((((((_e367 * bitcast<f32>(_e1874)) + (_e368 * bitcast<f32>(_e1880))) + (_e369 * bitcast<f32>(_e1886))) + bitcast<f32>(_e1892)) / _e1989) * 0.5f) + 0.5f), (0.5f - ((((((_e367 * bitcast<f32>(_e1904)) + (_e368 * bitcast<f32>(_e1910))) + (_e369 * bitcast<f32>(_e1916))) + bitcast<f32>(_e1922)) / _e1989) * 0.5f))), u32(bitcast<f32>(_e2002)), ((((((_e367 * bitcast<f32>(_e1934)) + (_e368 * bitcast<f32>(_e1940))) + (_e369 * bitcast<f32>(_e1946))) + bitcast<f32>(_e1952)) / _e1989) + -0.001f));
                        radiance_u002e_sroa_u002e_0_u002e_1_ = (_e1847 * _e2011);
                        radiance_u002e_sroa_u002e_12_u002e_1_ = (_e1849 * _e2011);
                        radiance_u002e_sroa_u002e_20_u002e_1_ = (_e1851 * _e2011);
                    }
                    let _e2019 = radiance_u002e_sroa_u002e_0_u002e_1_;
                    let _e2021 = radiance_u002e_sroa_u002e_12_u002e_1_;
                    let _e2023 = radiance_u002e_sroa_u002e_20_u002e_1_;
                    if (_e1431 > 0.5f) {
                        let _e2028 = storage_0_20_[_e1275];
                        let _e2031 = storage_0_20_[_e1280];
                        let _e2034 = storage_0_20_[_e1285];
                        let _e2036 = (_e367 - bitcast<f32>(_e2028));
                        let _e2038 = (_e368 - bitcast<f32>(_e2031));
                        let _e2040 = (_e369 - bitcast<f32>(_e2034));
                        let _e2041 = abs(_e2036);
                        let _e2042 = abs(_e2038);
                        let _e2044 = select(_e2041, _e2042, (_e2042 > _e2041));
                        let _e2045 = abs(_e2040);
                        let _e2050 = storage_0_20_[_e1303];
                        let _e2056 = select(bitcast<f32>(_e2050), 1f, (bitcast<f32>(_e2050) < 1f));
                        let _e2058 = (_e2056 + -0.1f);
                        let _e2069 = storage_0_20_[_e1853];
                        let _e2075 = textureSampleCompareLevel(tex_0_12_, sampler_0_11_, vec3<f32>(_e2036, _e2038, _e2040), u32(bitcast<f32>(_e2069)), (((_e2056 / _e2058) - ((_e2056 * 0.1f) / (select(_e2044, _e2045, (_e2045 > _e2044)) * _e2058))) + -0.001f));
                        radiance_u002e_sroa_u002e_0_u002e_2_ = (_e2019 * _e2075);
                        radiance_u002e_sroa_u002e_12_u002e_2_ = (_e2021 * _e2075);
                        radiance_u002e_sroa_u002e_20_u002e_2_ = (_e2023 * _e2075);
                    } else {
                        radiance_u002e_sroa_u002e_0_u002e_2_ = _e2019;
                        radiance_u002e_sroa_u002e_12_u002e_2_ = _e2021;
                        radiance_u002e_sroa_u002e_20_u002e_2_ = _e2023;
                    }
                }
                let _e2086 = radiance_u002e_sroa_u002e_0_u002e_2_;
                let _e2088 = radiance_u002e_sroa_u002e_12_u002e_2_;
                let _e2090 = radiance_u002e_sroa_u002e_20_u002e_2_;
                if (_e51 == 6u) {
                    let _e2093 = storage_0_20_[_e1269];
                    let _e2094 = bitcast<f32>(_e2093);
                    let _e2097 = storage_0_20_[_e1303];
                    let _e2101 = storage_0_20_[_e1853];
                    let _e2106 = select(1f, 0f, (_e2094 < 0.5f));
                    if (((1f - _e2106) * select(1f, 0f, (bitcast<f32>(_e2097) < 0.5f))) > 0.5f) {
                        let _e2123 = uniform_struct_0_1_.f6_.f1_.x;
                        let _e2128 = uniform_struct_0_1_.f6_.f1_.y;
                        let _e2133 = uniform_struct_0_1_.f6_.f1_.z;
                        let _e2136 = dot(vec3<f32>((_e367 - _e37), (_e368 - _e41), (_e369 - _e45)), vec3<f32>(_e2123, _e2128, _e2133));
                        let _e2141 = uniform_struct_0_1_.f6_.f2_.x;
                        let _e2146 = uniform_struct_0_1_.f6_.f2_.y;
                        let _e2151 = uniform_struct_0_1_.f6_.f2_.z;
                        let _e2156 = uniform_struct_0_1_.f6_.f2_.w;
                        let _e2160 = uniform_struct_0_1_.f6_.f3_;
                        let _e2164 = select(1f, 0f, (_e2136 < _e2146));
                        let _e2168 = select(1f, 0f, (_e2136 < _e2151));
                        let _e2172 = select(1f, 0f, (_e2136 < _e2156));
                        let _e2175 = u32(((_e2164 + _e2168) + _e2172));
                        let _e2178 = mix(mix(mix(_e2146, _e2151, _e2164), _e2156, _e2168), _e2160, _e2172);
                        let _e2188 = uniform_struct_0_1_.f6_.f0_[_e2175].f0_.x;
                        let _e2195 = uniform_struct_0_1_.f6_.f0_[_e2175].f1_.x;
                        let _e2202 = uniform_struct_0_1_.f6_.f0_[_e2175].f2_.x;
                        let _e2209 = uniform_struct_0_1_.f6_.f0_[_e2175].f3_.x;
                        let _e2222 = uniform_struct_0_1_.f6_.f0_[_e2175].f0_.y;
                        let _e2229 = uniform_struct_0_1_.f6_.f0_[_e2175].f1_.y;
                        let _e2236 = uniform_struct_0_1_.f6_.f0_[_e2175].f2_.y;
                        let _e2243 = uniform_struct_0_1_.f6_.f0_[_e2175].f3_.y;
                        let _e2256 = uniform_struct_0_1_.f6_.f0_[_e2175].f0_.z;
                        let _e2263 = uniform_struct_0_1_.f6_.f0_[_e2175].f1_.z;
                        let _e2270 = uniform_struct_0_1_.f6_.f0_[_e2175].f2_.z;
                        let _e2277 = uniform_struct_0_1_.f6_.f0_[_e2175].f3_.z;
                        let _e2290 = uniform_struct_0_1_.f6_.f0_[_e2175].f0_.w;
                        let _e2297 = uniform_struct_0_1_.f6_.f0_[_e2175].f1_.w;
                        let _e2304 = uniform_struct_0_1_.f6_.f0_[_e2175].f2_.w;
                        let _e2311 = uniform_struct_0_1_.f6_.f0_[_e2175].f3_.w;
                        let _e2317 = (_e2311 + (((_e367 * _e2290) + (_e368 * _e2297)) + (_e369 * _e2304)));
                        let _e2334 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e2209 + (((_e367 * _e2188) + (_e368 * _e2195)) + (_e369 * _e2202))) / _e2317) * 0.5f) + 0.5f), (0.5f - (((_e2243 + (((_e367 * _e2222) + (_e368 * _e2229)) + (_e369 * _e2236))) / _e2317) * 0.5f))), _e2175, (((_e2277 + (((_e367 * _e2256) + (_e368 * _e2263)) + (_e369 * _e2270))) / _e2317) + -0.001f));
                        if (_e2175 < 3u) {
                            let _e2339 = ((_e2178 - mix(mix(mix(_e2141, _e2146, _e2164), _e2151, _e2168), _e2156, _e2172)) * 0.08f);
                            let _e2345 = ((_e2178 - _e2136) / select(_e2339, 0.000001f, (_e2339 < 0.000001f)));
                            let _e2349 = select(_e2345, 0f, (_e2345 < 0f));
                            let _e2355 = (_e2175 + 1u);
                            let _e2362 = uniform_struct_0_1_.f6_.f0_[_e2355].f0_.x;
                            let _e2369 = uniform_struct_0_1_.f6_.f0_[_e2355].f1_.x;
                            let _e2376 = uniform_struct_0_1_.f6_.f0_[_e2355].f2_.x;
                            let _e2383 = uniform_struct_0_1_.f6_.f0_[_e2355].f3_.x;
                            let _e2396 = uniform_struct_0_1_.f6_.f0_[_e2355].f0_.y;
                            let _e2403 = uniform_struct_0_1_.f6_.f0_[_e2355].f1_.y;
                            let _e2410 = uniform_struct_0_1_.f6_.f0_[_e2355].f2_.y;
                            let _e2417 = uniform_struct_0_1_.f6_.f0_[_e2355].f3_.y;
                            let _e2430 = uniform_struct_0_1_.f6_.f0_[_e2355].f0_.z;
                            let _e2437 = uniform_struct_0_1_.f6_.f0_[_e2355].f1_.z;
                            let _e2444 = uniform_struct_0_1_.f6_.f0_[_e2355].f2_.z;
                            let _e2451 = uniform_struct_0_1_.f6_.f0_[_e2355].f3_.z;
                            let _e2464 = uniform_struct_0_1_.f6_.f0_[_e2355].f0_.w;
                            let _e2471 = uniform_struct_0_1_.f6_.f0_[_e2355].f1_.w;
                            let _e2478 = uniform_struct_0_1_.f6_.f0_[_e2355].f2_.w;
                            let _e2485 = uniform_struct_0_1_.f6_.f0_[_e2355].f3_.w;
                            let _e2491 = (_e2485 + (((_e367 * _e2464) + (_e368 * _e2471)) + (_e369 * _e2478)));
                            let _e2508 = textureSampleCompareLevel(tex_0_18_, sampler_0_11_, vec2<f32>(((((_e2383 + (((_e367 * _e2362) + (_e368 * _e2369)) + (_e369 * _e2376))) / _e2491) * 0.5f) + 0.5f), (0.5f - (((_e2417 + (((_e367 * _e2396) + (_e368 * _e2403)) + (_e369 * _e2410))) / _e2491) * 0.5f))), _e2355, (((_e2451 + (((_e367 * _e2430) + (_e368 * _e2437)) + (_e369 * _e2444))) / _e2491) + -0.001f));
                            _0_u002e_sroa_u002e_0_u002e_0_ = mix(_e2508, _e2334, select(_e2349, 1f, (_e2349 > 1f)));
                        } else {
                            _0_u002e_sroa_u002e_0_u002e_0_ = _e2334;
                        }
                    } else {
                        if (bitcast<f32>(_e2101) < 0f) {
                            _0_u002e_sroa_u002e_0_u002e_0_ = 1f;
                        } else {
                            if ((_e2094 < 1.5f) & (_e2106 > 0.5f)) {
                                let _e2522 = storage_0_20_[_e1275];
                                let _e2525 = storage_0_20_[_e1280];
                                let _e2528 = storage_0_20_[_e1285];
                                let _e2530 = (_e367 - bitcast<f32>(_e2522));
                                let _e2532 = (_e368 - bitcast<f32>(_e2525));
                                let _e2534 = (_e369 - bitcast<f32>(_e2528));
                                let _e2535 = abs(_e2530);
                                let _e2536 = abs(_e2532);
                                let _e2538 = select(_e2535, _e2536, (_e2536 > _e2535));
                                let _e2539 = abs(_e2534);
                                let _e2544 = storage_0_20_[_e1303];
                                let _e2550 = select(bitcast<f32>(_e2544), 1f, (bitcast<f32>(_e2544) < 1f));
                                let _e2552 = (_e2550 + -0.1f);
                                let _e2563 = storage_0_20_[_e1853];
                                let _e2569 = textureSampleCompareLevel(tex_0_12_, sampler_0_11_, vec3<f32>(_e2530, _e2532, _e2534), u32(bitcast<f32>(_e2563)), (((_e2550 / _e2552) - ((_e2550 * 0.1f) / (select(_e2538, _e2539, (_e2539 > _e2538)) * _e2552))) + -0.001f));
                                _0_u002e_sroa_u002e_0_u002e_0_ = _e2569;
                            } else {
                                let _e2575 = storage_0_20_[(_e1269 + 20u)];
                                let _e2581 = storage_0_20_[(_e1269 + 24u)];
                                let _e2587 = storage_0_20_[(_e1269 + 28u)];
                                let _e2593 = storage_0_20_[(_e1269 + 32u)];
                                let _e2605 = storage_0_20_[(_e1269 + 21u)];
                                let _e2611 = storage_0_20_[(_e1269 + 25u)];
                                let _e2617 = storage_0_20_[(_e1269 + 29u)];
                                let _e2623 = storage_0_20_[(_e1269 + 33u)];
                                let _e2635 = storage_0_20_[(_e1269 + 22u)];
                                let _e2641 = storage_0_20_[(_e1269 + 26u)];
                                let _e2647 = storage_0_20_[(_e1269 + 30u)];
                                let _e2653 = storage_0_20_[(_e1269 + 34u)];
                                let _e2665 = storage_0_20_[(_e1269 + 23u)];
                                let _e2671 = storage_0_20_[(_e1269 + 27u)];
                                let _e2677 = storage_0_20_[(_e1269 + 31u)];
                                let _e2683 = storage_0_20_[(_e1269 + 35u)];
                                let _e2690 = ((((_e367 * bitcast<f32>(_e2665)) + (_e368 * bitcast<f32>(_e2671))) + (_e369 * bitcast<f32>(_e2677))) + bitcast<f32>(_e2683));
                                let _e2703 = storage_0_20_[_e1853];
                                let _e2712 = textureSampleCompareLevel(tex_0_10_, sampler_0_11_, vec2<f32>((((((((_e367 * bitcast<f32>(_e2575)) + (_e368 * bitcast<f32>(_e2581))) + (_e369 * bitcast<f32>(_e2587))) + bitcast<f32>(_e2593)) / _e2690) * 0.5f) + 0.5f), (0.5f - ((((((_e367 * bitcast<f32>(_e2605)) + (_e368 * bitcast<f32>(_e2611))) + (_e369 * bitcast<f32>(_e2617))) + bitcast<f32>(_e2623)) / _e2690) * 0.5f))), u32(bitcast<f32>(_e2703)), ((((((_e367 * bitcast<f32>(_e2635)) + (_e368 * bitcast<f32>(_e2641))) + (_e369 * bitcast<f32>(_e2647))) + bitcast<f32>(_e2653)) / _e2690) + -0.001f));
                                _0_u002e_sroa_u002e_0_u002e_0_ = _e2712;
                            }
                        }
                    }
                    let _e2715 = _0_u002e_sroa_u002e_0_u002e_0_;
                    shadow_vis_u002e_sroa_u002e_0_u002e_1_ = select(_e2715, _e1238, (_e1238 < _e2715));
                } else {
                    shadow_vis_u002e_sroa_u002e_0_u002e_1_ = _e1238;
                }
                let _e2721 = shadow_vis_u002e_sroa_u002e_0_u002e_1_;
                out_u002e_shadow_vis_u002e_sroa_u002e_0_u002e_1_ = _e2721;
                let _e2725 = storage_0_20_[_e1275];
                let _e2728 = storage_0_20_[_e1280];
                let _e2731 = storage_0_20_[_e1285];
                let _e2733 = (bitcast<f32>(_e2725) - _e367);
                let _e2735 = (bitcast<f32>(_e2728) - _e368);
                let _e2737 = (bitcast<f32>(_e2731) - _e369);
                let _e2740 = storage_0_20_[_e1269];
                let _e2746 = select(1f, 0f, (bitcast<f32>(_e2740) < 0.5f));
                let _e2749 = storage_0_20_[_e1359];
                let _e2752 = storage_0_20_[_e1364];
                let _e2755 = storage_0_20_[_e1369];
                let _e2756 = bitcast<f32>(_e2749);
                let _e2757 = bitcast<f32>(_e2752);
                let _e2758 = bitcast<f32>(_e2755);
                let _e2762 = (1f / length(vec3<f32>(_e2756, _e2757, _e2758)));
                let _e2769 = (1f / length(vec3<f32>(_e2733, _e2735, _e2737)));
                let _e2773 = mix((_e2762 * _e2756), (_e2733 * _e2769), _e2746);
                let _e2774 = mix((_e2762 * _e2757), (_e2735 * _e2769), _e2746);
                let _e2775 = mix((_e2762 * _e2758), (_e2737 * _e2769), _e2746);
                let _e2779 = (1f / length(vec3<f32>(_e2773, _e2774, _e2775)));
                let _e2780 = (_e2773 * _e2779);
                let _e2781 = (_e2774 * _e2779);
                let _e2782 = (_e2775 * _e2779);
                let _e2787 = ((_e377 + _e2780) + 0f);
                let _e2789 = ((_e378 + _e2781) + 0f);
                let _e2791 = ((_e379 + _e2782) + 0.000001f);
                let _e2795 = (1f / length(vec3<f32>(_e2787, _e2789, _e2791)));
                let _e2796 = (_e2795 * _e2787);
                let _e2797 = (_e2795 * _e2789);
                let _e2798 = (_e2795 * _e2791);
                let _e2801 = dot(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e2780, _e2781, _e2782));
                let _e2805 = select(_e2801, 0.000001f, (_e2801 < 0.000001f));
                if (_e2805 > 0.000001f) {
                    let _e2810 = dot(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e2796, _e2797, _e2798));
                    let _e2814 = select(_e2810, 0.000001f, (_e2810 < 0.000001f));
                    let _e2817 = dot(vec3<f32>(_e377, _e378, _e379), vec3<f32>(_e2796, _e2797, _e2798));
                    let _e2821 = select(_e2817, 0.000001f, (_e2817 < 0.000001f));
                    let _e2822 = mix(_e1092, _e956, _e942);
                    let _e2823 = mix(_e1092, _e957, _e942);
                    let _e2824 = mix(_e1092, _e958, _e942);
                    let _e2827 = pow(0.85714287f, 5f);
                    let _e2829 = (1f - _e2822);
                    let _e2831 = (_e2822 + (_e2829 * _e2827));
                    let _e2833 = (1f - _e2823);
                    let _e2835 = (_e2823 + (_e2833 * _e2827));
                    let _e2837 = (1f - _e2824);
                    let _e2839 = (_e2824 + (_e2837 * _e2827));
                    let _e2844 = (1f - _e2821);
                    let _e2853 = ((_e2821 * pow(_e2844, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                    let _e2855 = pow(_e2844, 5f);
                    let _e2877 = mix(_e1092, _e956, _e942);
                    let _e2878 = mix(_e1092, _e957, _e942);
                    let _e2879 = mix(_e1092, _e958, _e942);
                    let _e2882 = pow(0.85714287f, 5f);
                    let _e2884 = (1f - _e2877);
                    let _e2886 = (_e2877 + (_e2884 * _e2882));
                    let _e2888 = (1f - _e2878);
                    let _e2890 = (_e2878 + (_e2888 * _e2882));
                    let _e2892 = (1f - _e2879);
                    let _e2894 = (_e2879 + (_e2892 * _e2882));
                    let _e2906 = ((_e2821 * pow(_e2844, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                    let _e2908 = pow(_e2844, 5f);
                    let _e2931 = (_e1096 * sqrt(_e1103));
                    let _e2934 = (_e1098 * (_e1096 * sqrt(_e1103)));
                    let _e2937 = dot(vec3<f32>(_e2796, _e2797, _e2798), vec3<f32>(_e741, _e742, _e743));
                    let _e2940 = dot(vec3<f32>(_e2796, _e2797, _e2798), vec3<f32>(cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).x, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).y, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).z));
                    let _e2942 = (_e2931 * _e2931);
                    let _e2945 = (_e2934 * _e2934);
                    let _e2948 = (_e2814 * _e2814);
                    let _e2949 = (_e2948 + (((_e2937 * _e2937) / _e2942) + ((_e2940 * _e2940) / _e2945)));
                    let _e2956 = (1f / (_e2949 * (((_e2931 * 3.1415927f) * _e2934) * _e2949)));
                    let _e2959 = dot(vec3<f32>(_e377, _e378, _e379), vec3<f32>(_e741, _e742, _e743));
                    let _e2962 = dot(vec3<f32>(_e377, _e378, _e379), vec3<f32>(cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).x, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).y, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).z));
                    let _e2965 = dot(vec3<f32>(_e2780, _e2781, _e2782), vec3<f32>(_e741, _e742, _e743));
                    let _e2968 = dot(vec3<f32>(_e2780, _e2781, _e2782), vec3<f32>(cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).x, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).y, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).z));
                    let _e2974 = (_e2805 * _e2805);
                    let _e2994 = (0.5f / max(((_e386 * sqrt(max((_e2974 + ((_e2942 * (_e2965 * _e2965)) + (_e2945 * (_e2968 * _e2968)))), 0.000001f))) + (_e2805 * sqrt(max((_e872 + ((_e2942 * (_e2959 * _e2959)) + (_e2945 * (_e2962 * _e2962)))), 0.000001f)))), 0.000001f));
                    let _e3013 = (1f / length(vec3<f32>(_e1109, _e1110, _e1111)));
                    let _e3020 = (_e2780 - (_e228 * _e2805));
                    let _e3021 = (_e2781 - (_e229 * _e2805));
                    let _e3022 = (_e2782 - (_e230 * _e2805));
                    let _e3026 = (1f / length(vec3<f32>(_e3020, _e3021, _e3022)));
                    let _e3032 = dot(vec3<f32>((_e1109 * _e3013), (_e1110 * _e3013), (_e1111 * _e3013)), vec3<f32>((_e3020 * _e3026), (_e3021 * _e3026), (_e3022 * _e3026)));
                    let _e3037 = acos(max(_e386, 0f));
                    let _e3040 = acos(max(_e2805, 0f));
                    let _e3064 = (max((1f - dot(vec3<f32>((max(((_e2822 + (_e2829 * _e2855)) - ((_e2831 - (_e2831 * _e1093)) * _e2853)), 0f) * _e944), (max(((_e2823 + (_e2833 * _e2855)) - ((_e2835 - (_e2835 * _e1094)) * _e2853)), 0f) * _e944), (max(((_e2824 + (_e2837 * _e2855)) - ((_e2839 - (_e2839 * _e1095)) * _e2853)), 0f) * _e944)), vec3<f32>(0.2126f, 0.7152f, 0.0722f))), 0f) * (((1f - ((_e1120 * 0.5f) / (_e1120 + 0.57f))) + (tan(min(_e3037, _e3040)) * ((((_e1120 * 0.45f) / (_e1120 + 0.09f)) * select(0f, _e3032, (_e3032 > 0f))) * sin(max(_e3037, _e3040))))) * 0.31830987f));
                    let _e3077 = (_e1039 * ((_e929 * ((((max(((_e2877 + (_e2884 * _e2908)) - ((_e2886 - (_e2886 * _e1093)) * _e2906)), 0f) * _e944) * _e2956) * _e2994) * _e943)) + (_e929 * (((_e1114 * _e953) * _e3064) * _e943))));
                    let _e3078 = (_e1041 * ((_e934 * ((((max(((_e2878 + (_e2888 * _e2908)) - ((_e2890 - (_e2890 * _e1094)) * _e2906)), 0f) * _e944) * _e2956) * _e2994) * _e943)) + (_e934 * (((_e1114 * _e954) * _e3064) * _e943))));
                    let _e3079 = (_e1043 * ((_e939 * ((((max(((_e2879 + (_e2892 * _e2908)) - ((_e2894 - (_e2894 * _e1095)) * _e2906)), 0f) * _e944) * _e2956) * _e2994) * _e943)) + (_e939 * (((_e1114 * _e955) * _e3064) * _e943))));
                    if (_e51 == 0u) {
                        spec_acc_u002e_sroa_u002e_0_u002e_2_ = _e1252;
                        spec_acc_u002e_sroa_u002e_6_u002e_2_ = _e1250;
                        spec_acc_u002e_sroa_u002e_9_u002e_2_ = _e1248;
                        diff_acc_u002e_sroa_u002e_0_u002e_2_ = _e1246;
                        diff_acc_u002e_sroa_u002e_6_u002e_2_ = _e1244;
                        diff_acc_u002e_sroa_u002e_9_u002e_2_ = _e1242;
                    } else {
                        let _e3086 = mix(_e1092, _e956, _e942);
                        let _e3087 = mix(_e1092, _e957, _e942);
                        let _e3088 = mix(_e1092, _e958, _e942);
                        let _e3091 = pow(0.85714287f, 5f);
                        let _e3093 = (1f - _e3086);
                        let _e3095 = (_e3086 + (_e3093 * _e3091));
                        let _e3097 = (1f - _e3087);
                        let _e3099 = (_e3087 + (_e3097 * _e3091));
                        let _e3101 = (1f - _e3088);
                        let _e3103 = (_e3088 + (_e3101 * _e3091));
                        let _e3115 = ((_e2821 * pow(_e2844, 6f)) / (pow(0.85714287f, 6f) * 0.14285715f));
                        let _e3117 = pow(_e2844, 5f);
                        let _e3140 = (_e1096 * sqrt(_e1103));
                        let _e3143 = (_e1098 * (_e1096 * sqrt(_e1103)));
                        let _e3146 = dot(vec3<f32>(_e2796, _e2797, _e2798), vec3<f32>(_e741, _e742, _e743));
                        let _e3149 = dot(vec3<f32>(_e2796, _e2797, _e2798), vec3<f32>(cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).x, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).y, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).z));
                        let _e3151 = (_e3140 * _e3140);
                        let _e3154 = (_e3143 * _e3143);
                        let _e3157 = (_e2948 + (((_e3146 * _e3146) / _e3151) + ((_e3149 * _e3149) / _e3154)));
                        let _e3164 = (1f / (_e3157 * (((_e3140 * 3.1415927f) * _e3143) * _e3157)));
                        let _e3167 = dot(vec3<f32>(_e377, _e378, _e379), vec3<f32>(_e741, _e742, _e743));
                        let _e3170 = dot(vec3<f32>(_e377, _e378, _e379), vec3<f32>(cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).x, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).y, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).z));
                        let _e3173 = dot(vec3<f32>(_e2780, _e2781, _e2782), vec3<f32>(_e741, _e742, _e743));
                        let _e3176 = dot(vec3<f32>(_e2780, _e2781, _e2782), vec3<f32>(cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).x, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).y, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).z));
                        let _e3201 = (0.5f / max(((_e386 * sqrt(max((_e2974 + ((_e3151 * (_e3173 * _e3173)) + (_e3154 * (_e3176 * _e3176)))), 0.000001f))) + (_e2805 * sqrt(max((_e872 + ((_e3151 * (_e3167 * _e3167)) + (_e3154 * (_e3170 * _e3170)))), 0.000001f)))), 0.000001f));
                        let _e3217 = (_e1039 * (_e929 * ((((max(((_e3086 + (_e3093 * _e3117)) - ((_e3095 - (_e3095 * _e1093)) * _e3115)), 0f) * _e944) * _e3164) * _e3201) * _e943)));
                        let _e3218 = (_e1041 * (_e934 * ((((max(((_e3087 + (_e3097 * _e3117)) - ((_e3099 - (_e3099 * _e1094)) * _e3115)), 0f) * _e944) * _e3164) * _e3201) * _e943)));
                        let _e3219 = (_e1043 * (_e939 * ((((max(((_e3088 + (_e3101 * _e3117)) - ((_e3103 - (_e3103 * _e1095)) * _e3115)), 0f) * _e944) * _e3164) * _e3201) * _e943)));
                        spec_acc_u002e_sroa_u002e_0_u002e_2_ = (_e1252 + (_e2086 * (_e2805 * _e3217)));
                        spec_acc_u002e_sroa_u002e_6_u002e_2_ = (_e1250 + (_e2088 * (_e2805 * _e3218)));
                        spec_acc_u002e_sroa_u002e_9_u002e_2_ = (_e1248 + (_e2090 * (_e2805 * _e3219)));
                        diff_acc_u002e_sroa_u002e_0_u002e_2_ = (_e1246 + (_e2086 * (_e2805 * (_e3077 - _e3217))));
                        diff_acc_u002e_sroa_u002e_6_u002e_2_ = (_e1244 + (_e2088 * (_e2805 * (_e3078 - _e3218))));
                        diff_acc_u002e_sroa_u002e_9_u002e_2_ = (_e1242 + (_e2090 * (_e2805 * (_e3079 - _e3219))));
                    }
                    let _e3248 = spec_acc_u002e_sroa_u002e_0_u002e_2_;
                    let _e3250 = spec_acc_u002e_sroa_u002e_6_u002e_2_;
                    let _e3252 = spec_acc_u002e_sroa_u002e_9_u002e_2_;
                    let _e3254 = diff_acc_u002e_sroa_u002e_0_u002e_2_;
                    let _e3256 = diff_acc_u002e_sroa_u002e_6_u002e_2_;
                    let _e3258 = diff_acc_u002e_sroa_u002e_9_u002e_2_;
                    if _e964 {
                        let _e3264 = (_e1140 * sqrt(_e1147));
                        let _e3267 = (_e1142 * (_e1140 * sqrt(_e1147)));
                        let _e3270 = dot(vec3<f32>(_e2796, _e2797, _e2798), vec3<f32>(_e741, _e742, _e743));
                        let _e3273 = dot(vec3<f32>(_e2796, _e2797, _e2798), vec3<f32>(cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).x, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).y, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).z));
                        let _e3275 = (_e3264 * _e3264);
                        let _e3278 = (_e3267 * _e3267);
                        let _e3281 = (_e2948 + (((_e3270 * _e3270) / _e3275) + ((_e3273 * _e3273) / _e3278)));
                        let _e3291 = dot(vec3<f32>(_e377, _e378, _e379), vec3<f32>(_e741, _e742, _e743));
                        let _e3294 = dot(vec3<f32>(_e377, _e378, _e379), vec3<f32>(cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).x, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).y, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).z));
                        let _e3297 = dot(vec3<f32>(_e2780, _e2781, _e2782), vec3<f32>(_e741, _e742, _e743));
                        let _e3300 = dot(vec3<f32>(_e2780, _e2781, _e2782), vec3<f32>(cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).x, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).y, cross(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e741, _e742, _e743)).z));
                        let _e3327 = ((_e1137 + ((1f - _e1137) * pow(_e2844, 5f))) * ((1f / (_e3281 * (((_e3264 * 3.1415927f) * _e3267) * _e3281))) * (0.5f / max(((_e386 * sqrt(max((_e2974 + ((_e3275 * (_e3297 * _e3297)) + (_e3278 * (_e3300 * _e3300)))), 0.000001f))) + (_e2805 * sqrt(max((_e872 + ((_e3275 * (_e3291 * _e3291)) + (_e3278 * (_e3294 * _e3294)))), 0.000001f)))), 0.000001f))));
                        coat_bsdf_u002e_sroa_u002e_0_u002e_0_ = ((_e3327 * bitcast<f32>(_e643)) * _e940);
                        coat_bsdf_u002e_sroa_u002e_5_u002e_0_ = ((_e3327 * bitcast<f32>(_e648)) * _e940);
                        coat_bsdf_u002e_sroa_u002e_8_u002e_0_ = ((_e3327 * bitcast<f32>(_e653)) * _e940);
                    } else {
                        coat_bsdf_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        coat_bsdf_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        coat_bsdf_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e3344 = coat_bsdf_u002e_sroa_u002e_0_u002e_0_;
                    let _e3346 = coat_bsdf_u002e_sroa_u002e_5_u002e_0_;
                    let _e3348 = coat_bsdf_u002e_sroa_u002e_8_u002e_0_;
                    if !((_e1151 <= 0f)) {
                        let _e3372 = ((_e2821 * ((1f / ((_e1156 * (((1f / _e386) + (1f / _e2805)) + -2f)) + 1f)) * (_e1156 / (pow((((_e1156 + -1f) * _e2948) + 1f), 2f) * 3.1415927f)))) / max(((_e386 * 4f) * _e2805), 0.000001f));
                        fuzz_bsdf_u002e_sroa_u002e_0_u002e_0_ = ((_e3372 * bitcast<f32>(_e608)) * _e1151);
                        fuzz_bsdf_u002e_sroa_u002e_5_u002e_0_ = ((_e3372 * bitcast<f32>(_e613)) * _e1151);
                        fuzz_bsdf_u002e_sroa_u002e_8_u002e_0_ = ((_e3372 * bitcast<f32>(_e618)) * _e1151);
                    } else {
                        fuzz_bsdf_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        fuzz_bsdf_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        fuzz_bsdf_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e3389 = fuzz_bsdf_u002e_sroa_u002e_0_u002e_0_;
                    let _e3391 = fuzz_bsdf_u002e_sroa_u002e_5_u002e_0_;
                    let _e3393 = fuzz_bsdf_u002e_sroa_u002e_8_u002e_0_;
                    if !((_e1166 <= 0f)) {
                        let _e3395 = (_e1096 * sqrt(_e1103));
                        if !((_e1167 <= 0f)) {
                            extinction_u002e_sroa_u002e_0_u002e_0_ = (-(log(max(bitcast<f32>(_e518), 0.000001f))) / _e1167);
                            extinction_u002e_sroa_u002e_5_u002e_0_ = (-(log(max(bitcast<f32>(_e523), 0.000001f))) / _e1167);
                            extinction_u002e_sroa_u002e_8_u002e_0_ = (-(log(max(bitcast<f32>(_e528), 0.000001f))) / _e1167);
                        } else {
                            extinction_u002e_sroa_u002e_0_u002e_0_ = 0f;
                            extinction_u002e_sroa_u002e_5_u002e_0_ = 0f;
                            extinction_u002e_sroa_u002e_8_u002e_0_ = 0f;
                        }
                        let _e3415 = extinction_u002e_sroa_u002e_0_u002e_0_;
                        let _e3417 = extinction_u002e_sroa_u002e_5_u002e_0_;
                        let _e3419 = extinction_u002e_sroa_u002e_8_u002e_0_;
                        let _e3426 = (_e3395 * _e3395);
                        let _e3431 = ((_e2948 * (_e3426 + -1f)) + 1f);
                        let _e3437 = (1f - _e3426);
                        let _e3456 = ((1f - (_e1092 + ((1f - _e1092) * pow(select(0.999999f, (1f - _e382), _e384), 5f)))) * ((_e3426 / (_e3431 * (_e3431 * 3.1415927f))) * (0.5f / max(((_e386 * sqrt(max((_e3426 + (_e2974 * _e3437)), 0.000001f))) + (_e2805 * sqrt(max((_e3426 + (_e872 * _e3437)), 0.000001f)))), 0.000001f))));
                        trans_bsdf_u002e_sroa_u002e_0_u002e_0_ = (((exp((_e1167 * -(_e3415))) * _e3456) * bitcast<f32>(_e518)) * _e1166);
                        trans_bsdf_u002e_sroa_u002e_5_u002e_0_ = (((exp((_e1167 * -(_e3417))) * _e3456) * bitcast<f32>(_e523)) * _e1166);
                        trans_bsdf_u002e_sroa_u002e_8_u002e_0_ = (((exp((_e1167 * -(_e3419))) * _e3456) * bitcast<f32>(_e528)) * _e1166);
                    } else {
                        trans_bsdf_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        trans_bsdf_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        trans_bsdf_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e3485 = trans_bsdf_u002e_sroa_u002e_0_u002e_0_;
                    let _e3487 = trans_bsdf_u002e_sroa_u002e_5_u002e_0_;
                    let _e3489 = trans_bsdf_u002e_sroa_u002e_8_u002e_0_;
                    if !((_e945 <= 0f)) {
                        let _e3494 = length(vec3<f32>((_e1109 - _e3020), (_e1110 - _e3021), (_e1111 - _e3022)));
                        let _e3496 = sqrt(3f);
                        let _e3519 = -(_e3494);
                        let _e3527 = (max(((clamp(((_e386 * _e2805) + (((_e876 + max((1f - _e2974), 0f)) - (_e3494 * _e3494)) * 0.5f)), -1f, 1f) * bitcast<f32>(_e568)) + 1f), 0f) * 0.31830987f);
                        ss_bsdf_u002e_sroa_u002e_0_u002e_0_ = (((exp((((1f / max((_e1199 * bitcast<f32>(_e563)), 0.000001f)) * _e3496) * _e3519)) * _e3527) * _e959) * _e945);
                        ss_bsdf_u002e_sroa_u002e_5_u002e_0_ = (((exp((((1f / max((_e1199 * bitcast<f32>(_e588)), 0.000001f)) * _e3496) * _e3519)) * _e3527) * _e960) * _e945);
                        ss_bsdf_u002e_sroa_u002e_8_u002e_0_ = (((exp((((1f / max((_e1199 * bitcast<f32>(_e593)), 0.000001f)) * _e3496) * _e3519)) * _e3527) * _e961) * _e945);
                    } else {
                        ss_bsdf_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        ss_bsdf_u002e_sroa_u002e_5_u002e_0_ = 0f;
                        ss_bsdf_u002e_sroa_u002e_8_u002e_0_ = 0f;
                    }
                    let _e3547 = ss_bsdf_u002e_sroa_u002e_0_u002e_0_;
                    let _e3549 = ss_bsdf_u002e_sroa_u002e_5_u002e_0_;
                    let _e3551 = ss_bsdf_u002e_sroa_u002e_8_u002e_0_;
                    lo_u002e_sroa_u002e_0_u002e_1_ = (_e1258 + (_e2086 * (_e2805 * (((_e3077 + _e3344) + (_e3389 + _e3485)) + _e3547))));
                    lo_u002e_sroa_u002e_6_u002e_1_ = (_e1256 + (_e2088 * (_e2805 * (((_e3078 + _e3346) + (_e3391 + _e3487)) + _e3549))));
                    lo_u002e_sroa_u002e_9_u002e_1_ = (_e1254 + (_e2090 * (_e2805 * (((_e3079 + _e3348) + (_e3393 + _e3489)) + _e3551))));
                    spec_acc_u002e_sroa_u002e_0_u002e_1_ = _e3248;
                    spec_acc_u002e_sroa_u002e_6_u002e_1_ = _e3250;
                    spec_acc_u002e_sroa_u002e_9_u002e_1_ = _e3252;
                    diff_acc_u002e_sroa_u002e_0_u002e_1_ = _e3254;
                    diff_acc_u002e_sroa_u002e_6_u002e_1_ = _e3256;
                    diff_acc_u002e_sroa_u002e_9_u002e_1_ = _e3258;
                } else {
                    lo_u002e_sroa_u002e_0_u002e_1_ = _e1258;
                    lo_u002e_sroa_u002e_6_u002e_1_ = _e1256;
                    lo_u002e_sroa_u002e_9_u002e_1_ = _e1254;
                    spec_acc_u002e_sroa_u002e_0_u002e_1_ = _e1252;
                    spec_acc_u002e_sroa_u002e_6_u002e_1_ = _e1250;
                    spec_acc_u002e_sroa_u002e_9_u002e_1_ = _e1248;
                    diff_acc_u002e_sroa_u002e_0_u002e_1_ = _e1246;
                    diff_acc_u002e_sroa_u002e_6_u002e_1_ = _e1244;
                    diff_acc_u002e_sroa_u002e_9_u002e_1_ = _e1242;
                }
                let _e3592 = lo_u002e_sroa_u002e_0_u002e_1_;
                out_u002e_lo_u002e_sroa_u002e_0_u002e_1_ = _e3592;
                let _e3595 = lo_u002e_sroa_u002e_6_u002e_1_;
                out_u002e_lo_u002e_sroa_u002e_6_u002e_1_ = _e3595;
                let _e3598 = lo_u002e_sroa_u002e_9_u002e_1_;
                out_u002e_lo_u002e_sroa_u002e_9_u002e_1_ = _e3598;
                let _e3601 = spec_acc_u002e_sroa_u002e_0_u002e_1_;
                out_u002e_spec_acc_u002e_sroa_u002e_0_u002e_1_ = _e3601;
                let _e3604 = spec_acc_u002e_sroa_u002e_6_u002e_1_;
                out_u002e_spec_acc_u002e_sroa_u002e_6_u002e_1_ = _e3604;
                let _e3607 = spec_acc_u002e_sroa_u002e_9_u002e_1_;
                out_u002e_spec_acc_u002e_sroa_u002e_9_u002e_1_ = _e3607;
                let _e3610 = diff_acc_u002e_sroa_u002e_0_u002e_1_;
                out_u002e_diff_acc_u002e_sroa_u002e_0_u002e_1_ = _e3610;
                let _e3613 = diff_acc_u002e_sroa_u002e_6_u002e_1_;
                out_u002e_diff_acc_u002e_sroa_u002e_6_u002e_1_ = _e3613;
                let _e3616 = diff_acc_u002e_sroa_u002e_9_u002e_1_;
                out_u002e_diff_acc_u002e_sroa_u002e_9_u002e_1_ = _e3616;
                let _e3619 = (_e1240 + 1u);
                if (_e3619 == _e1053) {
                    lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e3592;
                    lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e3595;
                    lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e3598;
                    spec_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e3601;
                    spec_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e3604;
                    spec_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e3607;
                    diff_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e3610;
                    diff_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e3613;
                    diff_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa = _e3616;
                    shadow_vis_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e2721;
                    break;
                } else {
                    shadow_vis_u002e_sroa_u002e_0_u002e_0_ = _e2721;
                    k_u002e_sroa_u002e_0_u002e_0_ = _e3619;
                    diff_acc_u002e_sroa_u002e_9_ = _e3616;
                    diff_acc_u002e_sroa_u002e_6_ = _e3613;
                    diff_acc_u002e_sroa_u002e_0_ = _e3610;
                    spec_acc_u002e_sroa_u002e_9_ = _e3607;
                    spec_acc_u002e_sroa_u002e_6_ = _e3604;
                    spec_acc_u002e_sroa_u002e_0_ = _e3601;
                    lo_u002e_sroa_u002e_9_ = _e3598;
                    lo_u002e_sroa_u002e_6_ = _e3595;
                    lo_u002e_sroa_u002e_0_ = _e3592;
                    continue;
                }
            }
        }
        let _e3681 = lo_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
        let _e3683 = lo_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
        let _e3685 = lo_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
        let _e3687 = spec_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
        let _e3689 = spec_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
        let _e3691 = spec_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
        let _e3693 = diff_acc_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
        let _e3695 = diff_acc_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
        let _e3697 = diff_acc_u002e_sroa_u002e_9_u002e_0_u002e_lcssa;
        let _e3699 = shadow_vis_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
        let _e3709 = uniform_struct_0_1_.f0_.x;
        let _e3713 = uniform_struct_0_1_.f0_.y;
        let _e3717 = uniform_struct_0_1_.f0_.z;
        let _e3721 = (_e24.x * (mix(_e953, (_e953 * _e944), _e942) * _e3709));
        let _e3722 = (_e24.x * (mix(_e954, (_e954 * _e944), _e942) * _e3713));
        let _e3723 = (_e24.x * (mix(_e955, (_e955 * _e944), _e942) * _e3717));
        let _e3724 = bitcast<f32>(_e678);
        if !((_e3724 <= 0f)) {
            let _e3731 = (_e3724 * bitcast<f32>(_e683));
            let _e3732 = (_e3724 * bitcast<f32>(_e688));
            let _e3733 = (_e3724 * bitcast<f32>(_e693));
            let _e3743 = ((1f - _e940) + ((1f - pow((1f - _e386), 5f)) * _e940));
            emission_u002e_sroa_u002e_0_u002e_1_ = (_e3731 + ((((_e3731 * bitcast<f32>(_e643)) * _e3743) - _e3731) * _e940));
            emission_u002e_sroa_u002e_8_u002e_1_ = (_e3732 + ((((_e3732 * bitcast<f32>(_e648)) * _e3743) - _e3732) * _e940));
            emission_u002e_sroa_u002e_11_u002e_1_ = (_e3733 + ((((_e3733 * bitcast<f32>(_e653)) * _e3743) - _e3733) * _e940));
        } else {
            emission_u002e_sroa_u002e_0_u002e_1_ = 0f;
            emission_u002e_sroa_u002e_8_u002e_1_ = 0f;
            emission_u002e_sroa_u002e_11_u002e_1_ = 0f;
        }
        let _e3772 = emission_u002e_sroa_u002e_0_u002e_1_;
        let _e3774 = emission_u002e_sroa_u002e_8_u002e_1_;
        let _e3776 = emission_u002e_sroa_u002e_11_u002e_1_;
        let _e3777 = bitcast<f32>(_e473);
        let _e3782 = ((_e3777 + -1f) / (_e3777 + 1f));
        let _e3783 = (_e3782 * _e3782);
        let _e3791 = (dot(vec3<f32>(_e228, _e229, _e230), vec3<f32>(_e377, _e378, _e379)) * 2f);
        let _e3800 = uniform_struct_0_1_.f4_;
        let _e3805 = textureSampleLevel(tex_0_13_, sampler_0_16_, vec3<f32>(((_e228 * _e3791) - _e377), ((_e229 * _e3791) - _e378), ((_e230 * _e3791) - _e379)), (_e465 * _e3800));
        let _e3811 = textureSample(tex_0_14_, sampler_0_16_, vec3<f32>(_e228, _e229, _e230));
        let _e3817 = textureSample(tex_0_15_, sampler_0_16_, vec2<f32>(_e386, _e465));
        let _e3822 = uniform_struct_0_1_.f3_;
        let _e3824 = (1f - _e942);
        let _e3834 = (_e3805.x * ((mix(_e3783, _e956, _e942) * _e3817.x) + _e3817.y));
        let _e3835 = (_e3805.y * ((mix(_e3783, _e957, _e942) * _e3817.x) + _e3817.y));
        let _e3836 = (_e3805.z * ((mix(_e3783, _e958, _e942) * _e3817.x) + _e3817.y));
        let _e3840 = max((1f - _e3817.y), 0f);
        let _e3853 = (_e24.x * (_e3822 * (_e3834 + (((_e3824 * _e953) * _e3811.x) * _e3840))));
        let _e3854 = (_e24.x * (_e3822 * (_e3835 + (((_e3824 * _e954) * _e3811.y) * _e3840))));
        let _e3855 = (_e24.x * (_e3822 * (_e3836 + (((_e3824 * _e955) * _e3811.z) * _e3840))));
        let _e3856 = (_e3822 * _e3834);
        let _e3857 = (_e3822 * _e3835);
        let _e3858 = (_e3822 * _e3836);
        let _e3871 = !((bitcast<f32>(_e708) >= 0.5f));
        let _e3872 = select(_e3772, (((_e3681 + _e3721) + _e3772) + _e3853), _e3871);
        let _e3873 = select(_e3774, (((_e3683 + _e3722) + _e3774) + _e3854), _e3871);
        let _e3874 = select(_e3776, (((_e3685 + _e3723) + _e3776) + _e3855), _e3871);
        let _e3878 = ((_e3872 <= _e3872) && (0f <= 0f));
        if _e3878 {
            if (abs(_e3872) < 100000000000000000000f) {
                _2_u002e_sroa_u002e_0_u002e_0_ = _e3872;
            } else {
                _2_u002e_sroa_u002e_0_u002e_0_ = 0f;
            }
        } else {
            _2_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
        let _e3888 = _2_u002e_sroa_u002e_0_u002e_0_;
        let _e3892 = ((_e3873 <= _e3873) && (0f <= 0f));
        if _e3892 {
            if (abs(_e3873) < 100000000000000000000f) {
                _3_u002e_sroa_u002e_0_u002e_0_ = _e3873;
            } else {
                _3_u002e_sroa_u002e_0_u002e_0_ = 0f;
            }
        } else {
            _3_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
        let _e3902 = _3_u002e_sroa_u002e_0_u002e_0_;
        let _e3906 = ((_e3874 <= _e3874) && (0f <= 0f));
        if _e3906 {
            if (abs(_e3874) < 100000000000000000000f) {
                _4_u002e_sroa_u002e_0_u002e_0_ = _e3874;
            } else {
                _4_u002e_sroa_u002e_0_u002e_0_ = 0f;
            }
        } else {
            _4_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
        let _e3916 = _4_u002e_sroa_u002e_0_u002e_0_;
        if _e3878 {
            if !((abs(_e3872) < 100000000000000000000f)) {
                x_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
            } else {
                x_bad_u002e_sroa_u002e_0_u002e_0_ = 0f;
            }
        } else {
            x_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
        }
        let _e3937 = x_bad_u002e_sroa_u002e_0_u002e_0_;
        if _e3892 {
            if !((abs(_e3873) < 100000000000000000000f)) {
                y_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
            } else {
                y_bad_u002e_sroa_u002e_0_u002e_0_ = 0f;
            }
        } else {
            y_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
        }
        let _e3949 = y_bad_u002e_sroa_u002e_0_u002e_0_;
        if _e3906 {
            if !((abs(_e3874) < 100000000000000000000f)) {
                z_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
            } else {
                z_bad_u002e_sroa_u002e_0_u002e_0_ = 0f;
            }
        } else {
            z_bad_u002e_sroa_u002e_0_u002e_0_ = 1f;
        }
        let _e3961 = z_bad_u002e_sroa_u002e_0_u002e_0_;
        let _e3963 = select(_e3961, _e3949, (_e3949 > _e3961));
        let _e3965 = select(_e3963, _e3937, (_e3937 > _e3963));
        let _e3976 = select(0f, 1f, (dot(vec3<f32>(_e3888, _e3902, _e3916), vec3<f32>(0.2126f, 0.7152f, 0.0722f)) > 1f));
        let _e4010 = (select(0f, 1f, (_e51 == 4u)) * _e465);
        let _e4014 = (select(0f, 1f, (_e51 == 5u)) * _e942);
        let _e4018 = (select(0f, 1f, (_e51 == 6u)) * _e3699);
        let _e4025 = ((_e4018 + (_e4014 + (_e4010 + ((_e88 * ((_e3693 + _e3721) + (_e3853 - _e3856))) + ((_e83 * _e3856) + ((_e78 * _e3687) + (_e73 * _e3888))))))) + (_e108 * mix(mix(_e3888, 1f, _e3976), 1f, _e3965)));
        let _e4026 = ((_e4018 + (_e4014 + (_e4010 + ((_e88 * ((_e3695 + _e3722) + (_e3854 - _e3857))) + ((_e83 * _e3857) + ((_e78 * _e3689) + (_e73 * _e3902))))))) + (_e108 * mix(mix(_e3902, 0f, _e3976), 0f, _e3965)));
        let _e4027 = ((_e4018 + (_e4014 + (_e4010 + ((_e88 * ((_e3697 + _e3723) + (_e3855 - _e3858))) + ((_e83 * _e3858) + ((_e78 * _e3691) + (_e73 * _e3916))))))) + (_e108 * mix(mix(_e3916, 0f, _e3976), 1f, _e3965)));
        let _e4035 = (((_e4025 * 0.2126f) + (_e4026 * 0.7152f)) + (_e4027 * 0.0722f));
        let _e4042 = (_e235 / (select(_e4035, 0f, (_e4035 < 0f)) + 1f));
        let _e4046 = (_e164 + (_e4025 * _e4042));
        out_u002e_3_ = _e4046;
        let _e4048 = (_e166 + (_e4026 * _e4042));
        out_u002e_6_ = _e4048;
        let _e4050 = (_e168 + (_e4027 * _e4042));
        out_u002e_9_ = _e4050;
        let _e4053 = (_e146 + (_e235 * _e709));
        out_u002e_163_ = _e4053;
        let _e4055 = (_e148 + _e4042);
        out_u002e_164_ = _e4055;
        let _e4061 = select(0f, 1f, (_e162 == 0u));
        let _e4062 = mix(_e170, _e4025, _e4061);
        out_u002e_0_ = _e4062;
        let _e4064 = mix(_e172, _e4026, _e4061);
        out_u002e_0_1 = _e4064;
        let _e4066 = mix(_e174, _e4027, _e4061);
        out_u002e_0_2 = _e4066;
        let _e4070 = (_e150 + (_e4061 * (_e709 - _e150)));
        out_u002e_0_3 = _e4070;
        let _e4073 = (_e162 + 1u);
        if (_e4073 == 4u) {
            break;
        } else {
            alpha_acc_u002e_sroa_u002e_0_ = _e4053;
            weight_u002e_sroa_u002e_0_ = _e4055;
            single_alpha_u002e_sroa_u002e_0_ = _e4070;
            covered_count_u002e_sroa_u002e_0_ = _e277;
            have_ref_u002e_sroa_u002e_0_ = select(_e154, _e235, (_e235 > _e154));
            disagree_acc_u002e_sroa_u002e_0_ = _e267;
            ref_depth_u002e_sroa_u002e_0_ = (_e158 + (_e242 * _e241));
            ref_id_u002e_sroa_u002e_0_ = select(_e160, _e190.x, _e238);
            s_u002e_sroa_u002e_0_ = _e4073;
            acc_u002e_sroa_u002e_0_ = _e4046;
            acc_u002e_sroa_u002e_4_ = _e4048;
            acc_u002e_sroa_u002e_7_ = _e4050;
            single_u002e_sroa_u002e_0_ = _e4062;
            single_u002e_sroa_u002e_4_ = _e4064;
            single_u002e_sroa_u002e_7_ = _e4066;
            ref_normal_u002e_sroa_u002e_0_ = mix(_e176, _e228, _e241);
            ref_normal_u002e_sroa_u002e_4_ = mix(_e178, _e229, _e241);
            ref_normal_u002e_sroa_u002e_6_ = mix(_e180, _e230, _e241);
            continue;
        }
    }
    let _e4095 = out_u002e_16_;
    let _e4097 = out_u002e_15_;
    let _e4099 = out_u002e_164_;
    let _e4101 = out_u002e_3_;
    let _e4103 = out_u002e_6_;
    let _e4105 = out_u002e_9_;
    let _e4107 = out_u002e_0_;
    let _e4109 = out_u002e_0_1;
    let _e4111 = out_u002e_0_2;
    let _e4113 = out_u002e_163_;
    let _e4115 = out_u002e_0_3;
    let _e4120 = select(0f, 1f, (_e4095 == 0f));
    let _e4133 = select(0f, 1f, select(false, (select(1f, _e4097, (_e4097 < 1f)) == 0f), (_e4095 == 4f)));
    let _e4137 = (1f - select(_e4120, _e4133, (_e4133 > _e4120)));
    let _e4141 = select(_e4099, 0.000001f, (_e4099 < 0.000001f));
    let _e4143 = (1f / _e4141);
    fragment_out = vec4<f32>(mix(_e4107, (_e4101 * _e4143), _e4137), mix(_e4109, (_e4103 * _e4143), _e4137), mix(_e4111, (_e4105 * _e4143), _e4137), (_e4115 + (((_e4113 / _e4141) - _e4115) * _e4137)));
    let _e4157 = fragment_out;
    return _e4157;
}
