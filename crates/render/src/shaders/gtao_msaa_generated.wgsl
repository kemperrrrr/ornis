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
    f0_: f32,
    f1_: f32,
    f2_: f32,
    f3_: f32,
}

@group(0) @binding(0) 
var tex_0_0_: texture_depth_multisampled_2d;
@group(0) @binding(1) 
var tex_0_1_: texture_multisampled_2d<f32>;
@group(0) @binding(2) 
var<uniform> uniform_struct_0_2_: UniformStruct0_;
@group(0) @binding(3) 
var<uniform> uniform_struct_0_3_: UniformStruct1_;

@fragment 
fn fs_main(@builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var fragment_out: vec4<f32>;
    var cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_: f32;
    var visibility_u002e_sroa_u002e_0_u002e_0565_: f32;
    var slice_u002e_sroa_u002e_0_u002e_0564_: u32;
    var horizon_pos_u002e_sroa_u002e_0_u002e_0563_: f32;
    var horizon_neg_u002e_sroa_u002e_0_u002e_0562_: f32;
    var step_u002e_sroa_u002e_0_u002e_0561_: u32;
    var out_u002e_17_: f32;
    var out_u002e_22_: f32;
    var out_u002e_26_: f32;

    let _e5 = textureDimensions(tex_0_0_);
    let _e7 = f32(_e5.x);
    let _e9 = textureDimensions(tex_0_0_);
    let _e11 = f32(_e9.y);
    let _e13 = u32((uv.x * _e7));
    let _e15 = u32((uv.y * _e11));
    let _e21 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0u);
    if (_e21 < 1f) {
        let _e29 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0u);
        let _e36 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0u);
        let _e42 = ((1f - abs(_e29.x)) - abs(_e36.y));
        let _e45 = max(-(_e42), 0f);
        let _e48 = -(_e45);
        let _e50 = (_e29.x + select(_e48, _e45, (_e29.x < 0f)));
        let _e54 = (_e36.y + select(_e48, _e45, (_e36.y < 0f)));
        let _e67 = ((uv.x * 2f) + -1f);
        let _e71 = (1f - (uv.y * 2f));
        let _e75 = uniform_struct_0_2_.f4_.x;
        let _e80 = uniform_struct_0_2_.f5_.x;
        let _e86 = uniform_struct_0_2_.f6_.x;
        let _e92 = uniform_struct_0_2_.f7_.x;
        let _e97 = uniform_struct_0_2_.f4_.w;
        let _e102 = uniform_struct_0_2_.f5_.w;
        let _e108 = uniform_struct_0_2_.f6_.w;
        let _e114 = uniform_struct_0_2_.f7_.w;
        let _e118 = ((_e92 + (((_e67 * _e75) + (_e71 * _e80)) + (_e21 * _e86))) / max((_e114 + (((_e67 * _e97) + (_e71 * _e102)) + (_e21 * _e108))), 0.000001f));
        let _e122 = uniform_struct_0_2_.f4_.y;
        let _e127 = uniform_struct_0_2_.f5_.y;
        let _e133 = uniform_struct_0_2_.f6_.y;
        let _e139 = uniform_struct_0_2_.f7_.y;
        let _e144 = uniform_struct_0_2_.f4_.w;
        let _e149 = uniform_struct_0_2_.f5_.w;
        let _e155 = uniform_struct_0_2_.f6_.w;
        let _e161 = uniform_struct_0_2_.f7_.w;
        let _e165 = ((_e139 + (((_e67 * _e122) + (_e71 * _e127)) + (_e21 * _e133))) / max((_e161 + (((_e67 * _e144) + (_e71 * _e149)) + (_e21 * _e155))), 0.000001f));
        let _e169 = uniform_struct_0_2_.f4_.z;
        let _e174 = uniform_struct_0_2_.f5_.z;
        let _e180 = uniform_struct_0_2_.f6_.z;
        let _e186 = uniform_struct_0_2_.f7_.z;
        let _e191 = uniform_struct_0_2_.f4_.w;
        let _e196 = uniform_struct_0_2_.f5_.w;
        let _e202 = uniform_struct_0_2_.f6_.w;
        let _e208 = uniform_struct_0_2_.f7_.w;
        let _e212 = ((_e186 + (((_e67 * _e169) + (_e71 * _e174)) + (_e21 * _e180))) / max((_e208 + (((_e67 * _e191) + (_e71 * _e196)) + (_e21 * _e202))), 0.000001f));
        let _e216 = uniform_struct_0_2_.f8_.x;
        let _e220 = uniform_struct_0_2_.f8_.y;
        let _e224 = uniform_struct_0_2_.f8_.z;
        let _e225 = (_e216 - _e118);
        let _e226 = (_e220 - _e165);
        let _e227 = (_e224 - _e212);
        let _e231 = max(length(vec3<f32>(_e225, _e226, _e227)), 0.000001f);
        let _e232 = (_e225 / _e231);
        let _e233 = (_e226 / _e231);
        let _e234 = (_e227 / _e231);
        let _e236 = (1f / _e7);
        let _e241 = (((uv.x + _e236) * 2f) + -1f);
        let _e245 = uniform_struct_0_2_.f4_.x;
        let _e250 = uniform_struct_0_2_.f5_.x;
        let _e256 = uniform_struct_0_2_.f6_.x;
        let _e262 = uniform_struct_0_2_.f7_.x;
        let _e267 = uniform_struct_0_2_.f4_.w;
        let _e272 = uniform_struct_0_2_.f5_.w;
        let _e278 = uniform_struct_0_2_.f6_.w;
        let _e284 = uniform_struct_0_2_.f7_.w;
        let _e292 = uniform_struct_0_2_.f4_.y;
        let _e297 = uniform_struct_0_2_.f5_.y;
        let _e303 = uniform_struct_0_2_.f6_.y;
        let _e309 = uniform_struct_0_2_.f7_.y;
        let _e314 = uniform_struct_0_2_.f4_.w;
        let _e319 = uniform_struct_0_2_.f5_.w;
        let _e325 = uniform_struct_0_2_.f6_.w;
        let _e331 = uniform_struct_0_2_.f7_.w;
        let _e339 = uniform_struct_0_2_.f4_.z;
        let _e344 = uniform_struct_0_2_.f5_.z;
        let _e350 = uniform_struct_0_2_.f6_.z;
        let _e356 = uniform_struct_0_2_.f7_.z;
        let _e361 = uniform_struct_0_2_.f4_.w;
        let _e366 = uniform_struct_0_2_.f5_.w;
        let _e372 = uniform_struct_0_2_.f6_.w;
        let _e378 = uniform_struct_0_2_.f7_.w;
        let _e389 = (1f - ((uv.y + (1f / _e11)) * 2f));
        let _e393 = uniform_struct_0_2_.f4_.x;
        let _e398 = uniform_struct_0_2_.f5_.x;
        let _e404 = uniform_struct_0_2_.f6_.x;
        let _e410 = uniform_struct_0_2_.f7_.x;
        let _e415 = uniform_struct_0_2_.f4_.w;
        let _e420 = uniform_struct_0_2_.f5_.w;
        let _e426 = uniform_struct_0_2_.f6_.w;
        let _e432 = uniform_struct_0_2_.f7_.w;
        let _e440 = uniform_struct_0_2_.f4_.y;
        let _e445 = uniform_struct_0_2_.f5_.y;
        let _e451 = uniform_struct_0_2_.f6_.y;
        let _e457 = uniform_struct_0_2_.f7_.y;
        let _e462 = uniform_struct_0_2_.f4_.w;
        let _e467 = uniform_struct_0_2_.f5_.w;
        let _e473 = uniform_struct_0_2_.f6_.w;
        let _e479 = uniform_struct_0_2_.f7_.w;
        let _e487 = uniform_struct_0_2_.f4_.z;
        let _e492 = uniform_struct_0_2_.f5_.z;
        let _e498 = uniform_struct_0_2_.f6_.z;
        let _e504 = uniform_struct_0_2_.f7_.z;
        let _e509 = uniform_struct_0_2_.f4_.w;
        let _e514 = uniform_struct_0_2_.f5_.w;
        let _e520 = uniform_struct_0_2_.f6_.w;
        let _e526 = uniform_struct_0_2_.f7_.w;
        let _e533 = ((((_e262 + (((_e241 * _e245) + (_e71 * _e250)) + (_e21 * _e256))) / max((_e284 + (((_e241 * _e267) + (_e71 * _e272)) + (_e21 * _e278))), 0.000001f)) - _e118) + 0.000001f);
        let _e534 = (((_e309 + (((_e241 * _e292) + (_e71 * _e297)) + (_e21 * _e303))) / max((_e331 + (((_e241 * _e314) + (_e71 * _e319)) + (_e21 * _e325))), 0.000001f)) - _e165);
        let _e535 = (((_e356 + (((_e241 * _e339) + (_e71 * _e344)) + (_e21 * _e350))) / max((_e378 + (((_e241 * _e361) + (_e71 * _e366)) + (_e21 * _e372))), 0.000001f)) - _e212);
        let _e545 = (((_e410 + (((_e67 * _e393) + (_e389 * _e398)) + (_e21 * _e404))) / max((_e432 + (((_e67 * _e415) + (_e389 * _e420)) + (_e21 * _e426))), 0.000001f)) - _e118);
        let _e548 = ((((_e457 + (((_e67 * _e440) + (_e389 * _e445)) + (_e21 * _e451))) / max((_e479 + (((_e67 * _e462) + (_e389 * _e467)) + (_e21 * _e473))), 0.000001f)) - _e165) + 0.000001f);
        let _e549 = (((_e504 + (((_e67 * _e487) + (_e389 * _e492)) + (_e21 * _e498))) / max((_e526 + (((_e67 * _e509) + (_e389 * _e514)) + (_e21 * _e520))), 0.000001f)) - _e212);
        let _e561 = uniform_struct_0_3_.f0_;
        let _e564 = uniform_struct_0_3_.f1_;
        let _e566 = (_e118 + (normalize(vec3<f32>(_e533, _e534, _e535)).x * _e561));
        let _e568 = (_e165 + (normalize(vec3<f32>(_e533, _e534, _e535)).y * _e561));
        let _e570 = (_e212 + (normalize(vec3<f32>(_e533, _e534, _e535)).z * _e561));
        let _e574 = uniform_struct_0_2_.f0_.x;
        let _e579 = uniform_struct_0_2_.f1_.x;
        let _e585 = uniform_struct_0_2_.f2_.x;
        let _e591 = uniform_struct_0_2_.f3_.x;
        let _e596 = uniform_struct_0_2_.f0_.y;
        let _e601 = uniform_struct_0_2_.f1_.y;
        let _e607 = uniform_struct_0_2_.f2_.y;
        let _e613 = uniform_struct_0_2_.f3_.y;
        let _e618 = uniform_struct_0_2_.f0_.w;
        let _e623 = uniform_struct_0_2_.f1_.w;
        let _e629 = uniform_struct_0_2_.f2_.w;
        let _e635 = uniform_struct_0_2_.f3_.w;
        let _e638 = max((_e635 + (((_e566 * _e618) + (_e568 * _e623)) + (_e570 * _e629))), 0.000001f);
        visibility_u002e_sroa_u002e_0_u002e_0565_ = 0f;
        slice_u002e_sroa_u002e_0_u002e_0564_ = 0u;
        loop {
            let _e677 = visibility_u002e_sroa_u002e_0_u002e_0565_;
            let _e679 = slice_u002e_sroa_u002e_0_u002e_0564_;
            let _e683 = ((fract((fract(((f32(_e13) * 0.06711056f) + (f32(_e15) * 0.00583715f))) * 52.982918f)) * 3.1415927f) + (f32(_e679) * 1.5707964f));
            let _e684 = cos(_e683);
            let _e685 = sin(_e683);
            let _e688 = ((normalize(vec3<f32>(_e533, _e534, _e535)).x * _e684) + (normalize(vec3<f32>(_e545, _e548, _e549)).x * _e685));
            let _e691 = ((normalize(vec3<f32>(_e533, _e534, _e535)).y * _e684) + (normalize(vec3<f32>(_e545, _e548, _e549)).y * _e685));
            let _e694 = ((normalize(vec3<f32>(_e533, _e534, _e535)).z * _e684) + (normalize(vec3<f32>(_e545, _e548, _e549)).z * _e685));
            let _e709 = (cross(vec3<f32>(normalize(vec3<f32>(_e688, _e691, _e694)).x, normalize(vec3<f32>(_e688, _e691, _e694)).y, normalize(vec3<f32>(_e688, _e691, _e694)).z), vec3<f32>(_e232, _e233, _e234)).x + 0.000001f);
            let _e729 = dot(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(normalize(vec3<f32>(_e709, cross(vec3<f32>(normalize(vec3<f32>(_e688, _e691, _e694)).x, normalize(vec3<f32>(_e688, _e691, _e694)).y, normalize(vec3<f32>(_e688, _e691, _e694)).z), vec3<f32>(_e232, _e233, _e234)).y, cross(vec3<f32>(normalize(vec3<f32>(_e688, _e691, _e694)).x, normalize(vec3<f32>(_e688, _e691, _e694)).y, normalize(vec3<f32>(_e688, _e691, _e694)).z), vec3<f32>(_e232, _e233, _e234)).z)).x, normalize(vec3<f32>(_e709, cross(vec3<f32>(normalize(vec3<f32>(_e688, _e691, _e694)).x, normalize(vec3<f32>(_e688, _e691, _e694)).y, normalize(vec3<f32>(_e688, _e691, _e694)).z), vec3<f32>(_e232, _e233, _e234)).y, cross(vec3<f32>(normalize(vec3<f32>(_e688, _e691, _e694)).x, normalize(vec3<f32>(_e688, _e691, _e694)).y, normalize(vec3<f32>(_e688, _e691, _e694)).z), vec3<f32>(_e232, _e233, _e234)).z)).y, normalize(vec3<f32>(_e709, cross(vec3<f32>(normalize(vec3<f32>(_e688, _e691, _e694)).x, normalize(vec3<f32>(_e688, _e691, _e694)).y, normalize(vec3<f32>(_e688, _e691, _e694)).z), vec3<f32>(_e232, _e233, _e234)).y, cross(vec3<f32>(normalize(vec3<f32>(_e688, _e691, _e694)).x, normalize(vec3<f32>(_e688, _e691, _e694)).y, normalize(vec3<f32>(_e688, _e691, _e694)).z), vec3<f32>(_e232, _e233, _e234)).z)).z));
            let _e731 = (normalize(vec3<f32>(_e50, _e54, _e42)).x - (normalize(vec3<f32>(_e709, cross(vec3<f32>(normalize(vec3<f32>(_e688, _e691, _e694)).x, normalize(vec3<f32>(_e688, _e691, _e694)).y, normalize(vec3<f32>(_e688, _e691, _e694)).z), vec3<f32>(_e232, _e233, _e234)).y, cross(vec3<f32>(normalize(vec3<f32>(_e688, _e691, _e694)).x, normalize(vec3<f32>(_e688, _e691, _e694)).y, normalize(vec3<f32>(_e688, _e691, _e694)).z), vec3<f32>(_e232, _e233, _e234)).z)).x * _e729));
            let _e733 = (normalize(vec3<f32>(_e50, _e54, _e42)).y - (normalize(vec3<f32>(_e709, cross(vec3<f32>(normalize(vec3<f32>(_e688, _e691, _e694)).x, normalize(vec3<f32>(_e688, _e691, _e694)).y, normalize(vec3<f32>(_e688, _e691, _e694)).z), vec3<f32>(_e232, _e233, _e234)).y, cross(vec3<f32>(normalize(vec3<f32>(_e688, _e691, _e694)).x, normalize(vec3<f32>(_e688, _e691, _e694)).y, normalize(vec3<f32>(_e688, _e691, _e694)).z), vec3<f32>(_e232, _e233, _e234)).z)).y * _e729));
            let _e735 = (normalize(vec3<f32>(_e50, _e54, _e42)).z - (normalize(vec3<f32>(_e709, cross(vec3<f32>(normalize(vec3<f32>(_e688, _e691, _e694)).x, normalize(vec3<f32>(_e688, _e691, _e694)).y, normalize(vec3<f32>(_e688, _e691, _e694)).z), vec3<f32>(_e232, _e233, _e234)).y, cross(vec3<f32>(normalize(vec3<f32>(_e688, _e691, _e694)).x, normalize(vec3<f32>(_e688, _e691, _e694)).y, normalize(vec3<f32>(_e688, _e691, _e694)).z), vec3<f32>(_e232, _e233, _e234)).z)).z * _e729));
            let _e739 = max(length(vec3<f32>(_e731, _e733, _e735)), 0.000001f);
            let _e752 = (sign(dot(vec3<f32>(normalize(vec3<f32>(_e688, _e691, _e694)).x, normalize(vec3<f32>(_e688, _e691, _e694)).y, normalize(vec3<f32>(_e688, _e691, _e694)).z), vec3<f32>(_e731, _e733, _e735))) * acos(clamp((dot(vec3<f32>(_e731, _e733, _e735), vec3<f32>(_e232, _e233, _e234)) / _e739), 0f, 1f)));
            let _e755 = cos((_e752 + 1.5707964f));
            let _e758 = cos((_e752 + -1.5707964f));
            horizon_pos_u002e_sroa_u002e_0_u002e_0563_ = _e755;
            horizon_neg_u002e_sroa_u002e_0_u002e_0562_ = _e758;
            step_u002e_sroa_u002e_0_u002e_0561_ = 1u;
            loop {
                let _e764 = horizon_pos_u002e_sroa_u002e_0_u002e_0563_;
                let _e766 = horizon_neg_u002e_sroa_u002e_0_u002e_0562_;
                let _e768 = step_u002e_sroa_u002e_0_u002e_0561_;
                let _e772 = ((max(length(vec3<f32>((((((_e591 + (((_e574 * _e566) + (_e568 * _e579)) + (_e570 * _e585))) / _e638) * 0.5f) + 0.5f) - uv.x), ((0.5f - (((_e613 + (((_e566 * _e596) + (_e568 * _e601)) + (_e570 * _e607))) / _e638) * 0.5f)) - uv.y), 0f)), _e236) * f32(_e768)) * 0.25f);
                let _e773 = (_e684 * _e772);
                let _e774 = (_e685 * _e772);
                let _e775 = (uv.x + _e773);
                let _e776 = (uv.y + _e774);
                if ((_e776 <= 1f) & ((_e775 <= 1f) & ((_e775 >= 0f) & (_e776 >= 0f)))) {
                    let _e797 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(u32((_e775 * _e7))), bitcast<i32>(u32((_e776 * _e11)))), 0u);
                    if (_e797 < 1f) {
                        let _e803 = ((_e775 * 2f) + -1f);
                        let _e807 = (1f - (_e776 * 2f));
                        let _e811 = uniform_struct_0_2_.f4_.x;
                        let _e816 = uniform_struct_0_2_.f5_.x;
                        let _e822 = uniform_struct_0_2_.f6_.x;
                        let _e828 = uniform_struct_0_2_.f7_.x;
                        let _e833 = uniform_struct_0_2_.f4_.w;
                        let _e838 = uniform_struct_0_2_.f5_.w;
                        let _e844 = uniform_struct_0_2_.f6_.w;
                        let _e850 = uniform_struct_0_2_.f7_.w;
                        let _e858 = uniform_struct_0_2_.f4_.y;
                        let _e863 = uniform_struct_0_2_.f5_.y;
                        let _e869 = uniform_struct_0_2_.f6_.y;
                        let _e875 = uniform_struct_0_2_.f7_.y;
                        let _e880 = uniform_struct_0_2_.f4_.w;
                        let _e885 = uniform_struct_0_2_.f5_.w;
                        let _e891 = uniform_struct_0_2_.f6_.w;
                        let _e897 = uniform_struct_0_2_.f7_.w;
                        let _e905 = uniform_struct_0_2_.f4_.z;
                        let _e910 = uniform_struct_0_2_.f5_.z;
                        let _e916 = uniform_struct_0_2_.f6_.z;
                        let _e922 = uniform_struct_0_2_.f7_.z;
                        let _e927 = uniform_struct_0_2_.f4_.w;
                        let _e932 = uniform_struct_0_2_.f5_.w;
                        let _e938 = uniform_struct_0_2_.f6_.w;
                        let _e944 = uniform_struct_0_2_.f7_.w;
                        let _e949 = (((_e828 + (((_e803 * _e811) + (_e807 * _e816)) + (_e797 * _e822))) / max((_e850 + (((_e803 * _e833) + (_e807 * _e838)) + (_e797 * _e844))), 0.000001f)) - _e118);
                        let _e950 = (((_e875 + (((_e803 * _e858) + (_e807 * _e863)) + (_e797 * _e869))) / max((_e897 + (((_e803 * _e880) + (_e807 * _e885)) + (_e797 * _e891))), 0.000001f)) - _e165);
                        let _e951 = (((_e922 + (((_e803 * _e905) + (_e807 * _e910)) + (_e797 * _e916))) / max((_e944 + (((_e803 * _e927) + (_e807 * _e932)) + (_e797 * _e938))), 0.000001f)) - _e212);
                        let _e953 = length(vec3<f32>(_e949, _e950, _e951));
                        if ((_e953 > 0.000001f) & (_e953 < _e561)) {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_ = mix(_e755, dot(vec3<f32>((_e949 / _e953), (_e950 / _e953), (_e951 / _e953)), vec3<f32>(_e232, _e233, _e234)), clamp((1f - (_e953 / max(_e561, 0.000001f))), 0f, 1f));
                        } else {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_ = -1f;
                        }
                    } else {
                        cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_ = -1f;
                    }
                } else {
                    cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_ = -1f;
                }
                let _e979 = cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_;
                let _e980 = max(_e764, _e979);
                out_u002e_22_ = _e980;
                let _e982 = (uv.x - _e773);
                let _e983 = (uv.y - _e774);
                if ((_e983 <= 1f) & ((_e982 <= 1f) & ((_e982 >= 0f) & (_e983 >= 0f)))) {
                    let _e1004 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(u32((_e982 * _e7))), bitcast<i32>(u32((_e983 * _e11)))), 0u);
                    if (_e1004 < 1f) {
                        let _e1010 = ((_e982 * 2f) + -1f);
                        let _e1014 = (1f - (_e983 * 2f));
                        let _e1018 = uniform_struct_0_2_.f4_.x;
                        let _e1023 = uniform_struct_0_2_.f5_.x;
                        let _e1029 = uniform_struct_0_2_.f6_.x;
                        let _e1035 = uniform_struct_0_2_.f7_.x;
                        let _e1040 = uniform_struct_0_2_.f4_.w;
                        let _e1045 = uniform_struct_0_2_.f5_.w;
                        let _e1051 = uniform_struct_0_2_.f6_.w;
                        let _e1057 = uniform_struct_0_2_.f7_.w;
                        let _e1065 = uniform_struct_0_2_.f4_.y;
                        let _e1070 = uniform_struct_0_2_.f5_.y;
                        let _e1076 = uniform_struct_0_2_.f6_.y;
                        let _e1082 = uniform_struct_0_2_.f7_.y;
                        let _e1087 = uniform_struct_0_2_.f4_.w;
                        let _e1092 = uniform_struct_0_2_.f5_.w;
                        let _e1098 = uniform_struct_0_2_.f6_.w;
                        let _e1104 = uniform_struct_0_2_.f7_.w;
                        let _e1112 = uniform_struct_0_2_.f4_.z;
                        let _e1117 = uniform_struct_0_2_.f5_.z;
                        let _e1123 = uniform_struct_0_2_.f6_.z;
                        let _e1129 = uniform_struct_0_2_.f7_.z;
                        let _e1134 = uniform_struct_0_2_.f4_.w;
                        let _e1139 = uniform_struct_0_2_.f5_.w;
                        let _e1145 = uniform_struct_0_2_.f6_.w;
                        let _e1151 = uniform_struct_0_2_.f7_.w;
                        let _e1156 = (((_e1035 + (((_e1010 * _e1018) + (_e1014 * _e1023)) + (_e1004 * _e1029))) / max((_e1057 + (((_e1010 * _e1040) + (_e1014 * _e1045)) + (_e1004 * _e1051))), 0.000001f)) - _e118);
                        let _e1157 = (((_e1082 + (((_e1010 * _e1065) + (_e1014 * _e1070)) + (_e1004 * _e1076))) / max((_e1104 + (((_e1010 * _e1087) + (_e1014 * _e1092)) + (_e1004 * _e1098))), 0.000001f)) - _e165);
                        let _e1158 = (((_e1129 + (((_e1010 * _e1112) + (_e1014 * _e1117)) + (_e1004 * _e1123))) / max((_e1151 + (((_e1010 * _e1134) + (_e1014 * _e1139)) + (_e1004 * _e1145))), 0.000001f)) - _e212);
                        let _e1160 = length(vec3<f32>(_e1156, _e1157, _e1158));
                        if ((_e1160 > 0.000001f) & (_e1160 < _e561)) {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = mix(_e758, dot(vec3<f32>((_e1156 / _e1160), (_e1157 / _e1160), (_e1158 / _e1160)), vec3<f32>(_e232, _e233, _e234)), clamp((1f - (_e1160 / max(_e561, 0.000001f))), 0f, 1f));
                        } else {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = -1f;
                        }
                    } else {
                        cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = -1f;
                    }
                } else {
                    cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = -1f;
                }
                let _e1186 = cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i;
                let _e1187 = max(_e766, _e1186);
                out_u002e_26_ = _e1187;
                let _e1190 = (_e768 + 1u);
                if (_e1190 == 5u) {
                    break;
                } else {
                    horizon_pos_u002e_sroa_u002e_0_u002e_0563_ = _e980;
                    horizon_neg_u002e_sroa_u002e_0_u002e_0562_ = _e1187;
                    step_u002e_sroa_u002e_0_u002e_0561_ = _e1190;
                    continue;
                }
            }
            let _e1199 = out_u002e_22_;
            let _e1201 = out_u002e_26_;
            let _e1211 = (acos(clamp(_e1199, -1f, 1f)) * 2f);
            let _e1222 = (acos(clamp(_e1201, -1f, 1f)) * -2f);
            let _e1234 = (_e677 + (_e739 * ((((cos(_e752) - cos((_e1211 - _e752))) + (_e1211 * sin(_e752))) * 0.25f) + (((cos(_e752) - cos((_e1222 - _e752))) + (_e1222 * sin(_e752))) * 0.25f))));
            out_u002e_17_ = _e1234;
            if (_e679 == 0u) {
                visibility_u002e_sroa_u002e_0_u002e_0565_ = _e1234;
                slice_u002e_sroa_u002e_0_u002e_0564_ = (_e679 + 1u);
                continue;
            } else {
                break;
            }
        }
        let _e1243 = out_u002e_17_;
        let _e1250 = clamp(mix(1f, (_e1243 * 0.5f), _e564), 0f, 1f);
        fragment_out = vec4<f32>(_e1250, _e1250, _e1250, 1f);
    } else {
        fragment_out = vec4<f32>(1f, 1f, 1f, 1f);
    }
    let _e1261 = fragment_out;
    return _e1261;
}
