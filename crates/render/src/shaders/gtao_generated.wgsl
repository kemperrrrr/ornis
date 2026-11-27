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
var tex_0_0_: texture_depth_2d;
@group(0) @binding(1) 
var tex_0_1_: texture_2d<f32>;
@group(0) @binding(2) 
var<uniform> uniform_struct_0_2_: UniformStruct0_;
@group(0) @binding(3) 
var<uniform> uniform_struct_0_3_: UniformStruct1_;

@fragment 
fn fs_main(@builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var fragment_out: vec4<f32>;
    var cos_h_u002e_sroa_u002e_0_u002e_0_: f32;
    var cos_h_u002e_sroa_u002e_0_u002e_0_1: f32;
    var visibility_u002e_sroa_u002e_0_: f32;
    var slice_u002e_sroa_u002e_0_: u32;
    var horizon_pos_u002e_sroa_u002e_0_: f32;
    var horizon_neg_u002e_sroa_u002e_0_: f32;
    var step_u002e_sroa_u002e_0_: u32;
    var out_u002e_8_: f32;
    var out_u002e_10_: f32;
    var out_u002e_11_: f32;

    let _e5 = textureDimensions(tex_0_0_);
    let _e8 = f32(bitcast<i32>(_e5.x));
    let _e11 = f32(bitcast<i32>(_e5.y));
    let _e13 = u32((uv.x * _e8));
    let _e15 = u32((uv.y * _e11));
    let _e20 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0i);
    if !((_e20 >= 1f)) {
        let _e29 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0i);
        let _e35 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0i);
        let _e41 = ((1f - abs(_e29.x)) - abs(_e35.y));
        let _e44 = max(-(_e41), 0f);
        let _e48 = -(_e44);
        let _e50 = (_e29.x + select(_e48, _e44, !((_e29.x >= 0f))));
        let _e55 = (_e35.y + select(_e48, _e44, !((_e35.y >= 0f))));
        let _e68 = ((uv.x * 2f) + -1f);
        let _e72 = (1f - (uv.y * 2f));
        let _e76 = uniform_struct_0_2_.f4_.x;
        let _e81 = uniform_struct_0_2_.f5_.x;
        let _e87 = uniform_struct_0_2_.f6_.x;
        let _e93 = uniform_struct_0_2_.f7_.x;
        let _e98 = uniform_struct_0_2_.f4_.w;
        let _e103 = uniform_struct_0_2_.f5_.w;
        let _e109 = uniform_struct_0_2_.f6_.w;
        let _e115 = uniform_struct_0_2_.f7_.w;
        let _e117 = ((_e93 + (((_e68 * _e76) + (_e72 * _e81)) + (_e20 * _e87))) / (_e115 + (((_e68 * _e98) + (_e72 * _e103)) + (_e20 * _e109))));
        let _e121 = uniform_struct_0_2_.f4_.y;
        let _e126 = uniform_struct_0_2_.f5_.y;
        let _e132 = uniform_struct_0_2_.f6_.y;
        let _e138 = uniform_struct_0_2_.f7_.y;
        let _e143 = uniform_struct_0_2_.f4_.w;
        let _e148 = uniform_struct_0_2_.f5_.w;
        let _e154 = uniform_struct_0_2_.f6_.w;
        let _e160 = uniform_struct_0_2_.f7_.w;
        let _e162 = ((_e138 + (((_e68 * _e121) + (_e72 * _e126)) + (_e20 * _e132))) / (_e160 + (((_e68 * _e143) + (_e72 * _e148)) + (_e20 * _e154))));
        let _e166 = uniform_struct_0_2_.f4_.z;
        let _e171 = uniform_struct_0_2_.f5_.z;
        let _e177 = uniform_struct_0_2_.f6_.z;
        let _e183 = uniform_struct_0_2_.f7_.z;
        let _e188 = uniform_struct_0_2_.f4_.w;
        let _e193 = uniform_struct_0_2_.f5_.w;
        let _e199 = uniform_struct_0_2_.f6_.w;
        let _e205 = uniform_struct_0_2_.f7_.w;
        let _e207 = ((_e183 + (((_e68 * _e166) + (_e72 * _e171)) + (_e20 * _e177))) / (_e205 + (((_e68 * _e188) + (_e72 * _e193)) + (_e20 * _e199))));
        let _e211 = uniform_struct_0_2_.f8_.x;
        let _e215 = uniform_struct_0_2_.f8_.y;
        let _e219 = uniform_struct_0_2_.f8_.z;
        let _e220 = (_e211 - _e117);
        let _e221 = (_e215 - _e162);
        let _e222 = (_e219 - _e207);
        let _e226 = max(length(vec3<f32>(_e220, _e221, _e222)), 0.000001f);
        let _e227 = (_e220 / _e226);
        let _e228 = (_e221 / _e226);
        let _e229 = (_e222 / _e226);
        let _e231 = (1f / _e8);
        let _e236 = (((uv.x + _e231) * 2f) + -1f);
        let _e240 = uniform_struct_0_2_.f4_.x;
        let _e245 = uniform_struct_0_2_.f5_.x;
        let _e251 = uniform_struct_0_2_.f6_.x;
        let _e257 = uniform_struct_0_2_.f7_.x;
        let _e262 = uniform_struct_0_2_.f4_.w;
        let _e267 = uniform_struct_0_2_.f5_.w;
        let _e273 = uniform_struct_0_2_.f6_.w;
        let _e279 = uniform_struct_0_2_.f7_.w;
        let _e285 = uniform_struct_0_2_.f4_.y;
        let _e290 = uniform_struct_0_2_.f5_.y;
        let _e296 = uniform_struct_0_2_.f6_.y;
        let _e302 = uniform_struct_0_2_.f7_.y;
        let _e307 = uniform_struct_0_2_.f4_.w;
        let _e312 = uniform_struct_0_2_.f5_.w;
        let _e318 = uniform_struct_0_2_.f6_.w;
        let _e324 = uniform_struct_0_2_.f7_.w;
        let _e330 = uniform_struct_0_2_.f4_.z;
        let _e335 = uniform_struct_0_2_.f5_.z;
        let _e341 = uniform_struct_0_2_.f6_.z;
        let _e347 = uniform_struct_0_2_.f7_.z;
        let _e352 = uniform_struct_0_2_.f4_.w;
        let _e357 = uniform_struct_0_2_.f5_.w;
        let _e363 = uniform_struct_0_2_.f6_.w;
        let _e369 = uniform_struct_0_2_.f7_.w;
        let _e378 = (1f - ((uv.y + (1f / _e11)) * 2f));
        let _e382 = uniform_struct_0_2_.f4_.x;
        let _e387 = uniform_struct_0_2_.f5_.x;
        let _e393 = uniform_struct_0_2_.f6_.x;
        let _e399 = uniform_struct_0_2_.f7_.x;
        let _e404 = uniform_struct_0_2_.f4_.w;
        let _e409 = uniform_struct_0_2_.f5_.w;
        let _e415 = uniform_struct_0_2_.f6_.w;
        let _e421 = uniform_struct_0_2_.f7_.w;
        let _e427 = uniform_struct_0_2_.f4_.y;
        let _e432 = uniform_struct_0_2_.f5_.y;
        let _e438 = uniform_struct_0_2_.f6_.y;
        let _e444 = uniform_struct_0_2_.f7_.y;
        let _e449 = uniform_struct_0_2_.f4_.w;
        let _e454 = uniform_struct_0_2_.f5_.w;
        let _e460 = uniform_struct_0_2_.f6_.w;
        let _e466 = uniform_struct_0_2_.f7_.w;
        let _e472 = uniform_struct_0_2_.f4_.z;
        let _e477 = uniform_struct_0_2_.f5_.z;
        let _e483 = uniform_struct_0_2_.f6_.z;
        let _e489 = uniform_struct_0_2_.f7_.z;
        let _e494 = uniform_struct_0_2_.f4_.w;
        let _e499 = uniform_struct_0_2_.f5_.w;
        let _e505 = uniform_struct_0_2_.f6_.w;
        let _e511 = uniform_struct_0_2_.f7_.w;
        let _e516 = ((((_e257 + (((_e236 * _e240) + (_e72 * _e245)) + (_e20 * _e251))) / (_e279 + (((_e236 * _e262) + (_e72 * _e267)) + (_e20 * _e273)))) - _e117) + 0.000001f);
        let _e517 = (((_e302 + (((_e236 * _e285) + (_e72 * _e290)) + (_e20 * _e296))) / (_e324 + (((_e236 * _e307) + (_e72 * _e312)) + (_e20 * _e318)))) - _e162);
        let _e518 = (((_e347 + (((_e236 * _e330) + (_e72 * _e335)) + (_e20 * _e341))) / (_e369 + (((_e236 * _e352) + (_e72 * _e357)) + (_e20 * _e363)))) - _e207);
        let _e528 = (((_e399 + (((_e68 * _e382) + (_e378 * _e387)) + (_e20 * _e393))) / (_e421 + (((_e68 * _e404) + (_e378 * _e409)) + (_e20 * _e415)))) - _e117);
        let _e531 = ((((_e444 + (((_e68 * _e427) + (_e378 * _e432)) + (_e20 * _e438))) / (_e466 + (((_e68 * _e449) + (_e378 * _e454)) + (_e20 * _e460)))) - _e162) + 0.000001f);
        let _e532 = (((_e489 + (((_e68 * _e472) + (_e378 * _e477)) + (_e20 * _e483))) / (_e511 + (((_e68 * _e494) + (_e378 * _e499)) + (_e20 * _e505)))) - _e207);
        let _e544 = uniform_struct_0_3_.f0_;
        let _e547 = uniform_struct_0_3_.f1_;
        let _e549 = (_e117 + (normalize(vec3<f32>(_e516, _e517, _e518)).x * _e544));
        let _e551 = (_e162 + (normalize(vec3<f32>(_e516, _e517, _e518)).y * _e544));
        let _e553 = (_e207 + (normalize(vec3<f32>(_e516, _e517, _e518)).z * _e544));
        let _e557 = uniform_struct_0_2_.f0_.x;
        let _e562 = uniform_struct_0_2_.f1_.x;
        let _e568 = uniform_struct_0_2_.f2_.x;
        let _e574 = uniform_struct_0_2_.f3_.x;
        let _e579 = uniform_struct_0_2_.f0_.y;
        let _e584 = uniform_struct_0_2_.f1_.y;
        let _e590 = uniform_struct_0_2_.f2_.y;
        let _e596 = uniform_struct_0_2_.f3_.y;
        let _e601 = uniform_struct_0_2_.f0_.w;
        let _e606 = uniform_struct_0_2_.f1_.w;
        let _e612 = uniform_struct_0_2_.f2_.w;
        let _e618 = uniform_struct_0_2_.f3_.w;
        let _e621 = max((_e618 + (((_e549 * _e601) + (_e551 * _e606)) + (_e553 * _e612))), 0.000001f);
        visibility_u002e_sroa_u002e_0_ = 0f;
        slice_u002e_sroa_u002e_0_ = 0u;
        loop {
            let _e672 = visibility_u002e_sroa_u002e_0_;
            let _e674 = slice_u002e_sroa_u002e_0_;
            let _e678 = ((fract((fract(dot(vec2<f32>(f32(_e13), f32(_e15)), vec2<f32>(0.06711056f, 0.00583715f))) * 52.982918f)) * 3.1415927f) + (f32(_e674) * 1.5707964f));
            let _e679 = cos(_e678);
            let _e680 = sin(_e678);
            let _e683 = ((normalize(vec3<f32>(_e516, _e517, _e518)).x * _e679) + (normalize(vec3<f32>(_e528, _e531, _e532)).x * _e680));
            let _e686 = ((normalize(vec3<f32>(_e516, _e517, _e518)).y * _e679) + (normalize(vec3<f32>(_e528, _e531, _e532)).y * _e680));
            let _e689 = ((normalize(vec3<f32>(_e516, _e517, _e518)).z * _e679) + (normalize(vec3<f32>(_e528, _e531, _e532)).z * _e680));
            let _e704 = (cross(vec3<f32>(normalize(vec3<f32>(_e683, _e686, _e689)).x, normalize(vec3<f32>(_e683, _e686, _e689)).y, normalize(vec3<f32>(_e683, _e686, _e689)).z), vec3<f32>(_e227, _e228, _e229)).x + 0.000001f);
            let _e724 = dot(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(normalize(vec3<f32>(_e704, cross(vec3<f32>(normalize(vec3<f32>(_e683, _e686, _e689)).x, normalize(vec3<f32>(_e683, _e686, _e689)).y, normalize(vec3<f32>(_e683, _e686, _e689)).z), vec3<f32>(_e227, _e228, _e229)).y, cross(vec3<f32>(normalize(vec3<f32>(_e683, _e686, _e689)).x, normalize(vec3<f32>(_e683, _e686, _e689)).y, normalize(vec3<f32>(_e683, _e686, _e689)).z), vec3<f32>(_e227, _e228, _e229)).z)).x, normalize(vec3<f32>(_e704, cross(vec3<f32>(normalize(vec3<f32>(_e683, _e686, _e689)).x, normalize(vec3<f32>(_e683, _e686, _e689)).y, normalize(vec3<f32>(_e683, _e686, _e689)).z), vec3<f32>(_e227, _e228, _e229)).y, cross(vec3<f32>(normalize(vec3<f32>(_e683, _e686, _e689)).x, normalize(vec3<f32>(_e683, _e686, _e689)).y, normalize(vec3<f32>(_e683, _e686, _e689)).z), vec3<f32>(_e227, _e228, _e229)).z)).y, normalize(vec3<f32>(_e704, cross(vec3<f32>(normalize(vec3<f32>(_e683, _e686, _e689)).x, normalize(vec3<f32>(_e683, _e686, _e689)).y, normalize(vec3<f32>(_e683, _e686, _e689)).z), vec3<f32>(_e227, _e228, _e229)).y, cross(vec3<f32>(normalize(vec3<f32>(_e683, _e686, _e689)).x, normalize(vec3<f32>(_e683, _e686, _e689)).y, normalize(vec3<f32>(_e683, _e686, _e689)).z), vec3<f32>(_e227, _e228, _e229)).z)).z));
            let _e726 = (normalize(vec3<f32>(_e50, _e55, _e41)).x - (normalize(vec3<f32>(_e704, cross(vec3<f32>(normalize(vec3<f32>(_e683, _e686, _e689)).x, normalize(vec3<f32>(_e683, _e686, _e689)).y, normalize(vec3<f32>(_e683, _e686, _e689)).z), vec3<f32>(_e227, _e228, _e229)).y, cross(vec3<f32>(normalize(vec3<f32>(_e683, _e686, _e689)).x, normalize(vec3<f32>(_e683, _e686, _e689)).y, normalize(vec3<f32>(_e683, _e686, _e689)).z), vec3<f32>(_e227, _e228, _e229)).z)).x * _e724));
            let _e728 = (normalize(vec3<f32>(_e50, _e55, _e41)).y - (normalize(vec3<f32>(_e704, cross(vec3<f32>(normalize(vec3<f32>(_e683, _e686, _e689)).x, normalize(vec3<f32>(_e683, _e686, _e689)).y, normalize(vec3<f32>(_e683, _e686, _e689)).z), vec3<f32>(_e227, _e228, _e229)).y, cross(vec3<f32>(normalize(vec3<f32>(_e683, _e686, _e689)).x, normalize(vec3<f32>(_e683, _e686, _e689)).y, normalize(vec3<f32>(_e683, _e686, _e689)).z), vec3<f32>(_e227, _e228, _e229)).z)).y * _e724));
            let _e730 = (normalize(vec3<f32>(_e50, _e55, _e41)).z - (normalize(vec3<f32>(_e704, cross(vec3<f32>(normalize(vec3<f32>(_e683, _e686, _e689)).x, normalize(vec3<f32>(_e683, _e686, _e689)).y, normalize(vec3<f32>(_e683, _e686, _e689)).z), vec3<f32>(_e227, _e228, _e229)).y, cross(vec3<f32>(normalize(vec3<f32>(_e683, _e686, _e689)).x, normalize(vec3<f32>(_e683, _e686, _e689)).y, normalize(vec3<f32>(_e683, _e686, _e689)).z), vec3<f32>(_e227, _e228, _e229)).z)).z * _e724));
            let _e734 = max(length(vec3<f32>(_e726, _e728, _e730)), 0.000001f);
            let _e747 = (sign(dot(vec3<f32>(normalize(vec3<f32>(_e683, _e686, _e689)).x, normalize(vec3<f32>(_e683, _e686, _e689)).y, normalize(vec3<f32>(_e683, _e686, _e689)).z), vec3<f32>(_e726, _e728, _e730))) * acos(clamp((dot(vec3<f32>(_e726, _e728, _e730), vec3<f32>(_e227, _e228, _e229)) / _e734), 0f, 1f)));
            let _e750 = cos((_e747 + 1.5707964f));
            let _e753 = cos((_e747 + -1.5707964f));
            horizon_pos_u002e_sroa_u002e_0_ = _e750;
            horizon_neg_u002e_sroa_u002e_0_ = _e753;
            step_u002e_sroa_u002e_0_ = 1u;
            loop {
                let _e759 = horizon_pos_u002e_sroa_u002e_0_;
                let _e761 = horizon_neg_u002e_sroa_u002e_0_;
                let _e763 = step_u002e_sroa_u002e_0_;
                let _e767 = ((max(length(vec3<f32>((((((_e574 + (((_e557 * _e549) + (_e551 * _e562)) + (_e553 * _e568))) / _e621) * 0.5f) + 0.5f) - uv.x), ((0.5f - (((_e596 + (((_e549 * _e579) + (_e551 * _e584)) + (_e553 * _e590))) / _e621) * 0.5f)) - uv.y), 0f)), _e231) * f32(_e763)) * 0.25f);
                let _e768 = (_e679 * _e767);
                let _e769 = (_e680 * _e767);
                let _e770 = (uv.x + _e768);
                let _e771 = (uv.y + _e769);
                if ((_e771 <= 1f) & ((_e770 <= 1f) & ((_e770 >= 0f) & (_e771 >= 0f)))) {
                    let _e794 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(min(u32((_e770 * _e8)), u32((_e8 + -1f)))), bitcast<i32>(min(u32((_e771 * _e11)), u32((_e11 + -1f))))), 0i);
                    if (_e794 < 1f) {
                        let _e800 = ((_e770 * 2f) + -1f);
                        let _e804 = (1f - (_e771 * 2f));
                        let _e808 = uniform_struct_0_2_.f4_.x;
                        let _e813 = uniform_struct_0_2_.f5_.x;
                        let _e819 = uniform_struct_0_2_.f6_.x;
                        let _e825 = uniform_struct_0_2_.f7_.x;
                        let _e830 = uniform_struct_0_2_.f4_.w;
                        let _e835 = uniform_struct_0_2_.f5_.w;
                        let _e841 = uniform_struct_0_2_.f6_.w;
                        let _e847 = uniform_struct_0_2_.f7_.w;
                        let _e853 = uniform_struct_0_2_.f4_.y;
                        let _e858 = uniform_struct_0_2_.f5_.y;
                        let _e864 = uniform_struct_0_2_.f6_.y;
                        let _e870 = uniform_struct_0_2_.f7_.y;
                        let _e875 = uniform_struct_0_2_.f4_.w;
                        let _e880 = uniform_struct_0_2_.f5_.w;
                        let _e886 = uniform_struct_0_2_.f6_.w;
                        let _e892 = uniform_struct_0_2_.f7_.w;
                        let _e898 = uniform_struct_0_2_.f4_.z;
                        let _e903 = uniform_struct_0_2_.f5_.z;
                        let _e909 = uniform_struct_0_2_.f6_.z;
                        let _e915 = uniform_struct_0_2_.f7_.z;
                        let _e920 = uniform_struct_0_2_.f4_.w;
                        let _e925 = uniform_struct_0_2_.f5_.w;
                        let _e931 = uniform_struct_0_2_.f6_.w;
                        let _e937 = uniform_struct_0_2_.f7_.w;
                        let _e940 = (((_e825 + (((_e800 * _e808) + (_e804 * _e813)) + (_e794 * _e819))) / (_e847 + (((_e800 * _e830) + (_e804 * _e835)) + (_e794 * _e841)))) - _e117);
                        let _e941 = (((_e870 + (((_e800 * _e853) + (_e804 * _e858)) + (_e794 * _e864))) / (_e892 + (((_e800 * _e875) + (_e804 * _e880)) + (_e794 * _e886)))) - _e162);
                        let _e942 = (((_e915 + (((_e800 * _e898) + (_e804 * _e903)) + (_e794 * _e909))) / (_e937 + (((_e800 * _e920) + (_e804 * _e925)) + (_e794 * _e931)))) - _e207);
                        let _e944 = length(vec3<f32>(_e940, _e941, _e942));
                        if ((_e944 > 0.000001f) & (_e944 < _e544)) {
                            cos_h_u002e_sroa_u002e_0_u002e_0_1 = mix(_e750, dot(vec3<f32>((_e940 / _e944), (_e941 / _e944), (_e942 / _e944)), vec3<f32>(_e227, _e228, _e229)), clamp((1f - (_e944 / max(_e544, 0.000001f))), 0f, 1f));
                        } else {
                            cos_h_u002e_sroa_u002e_0_u002e_0_1 = -1f;
                        }
                    } else {
                        cos_h_u002e_sroa_u002e_0_u002e_0_1 = -1f;
                    }
                } else {
                    cos_h_u002e_sroa_u002e_0_u002e_0_1 = -1f;
                }
                let _e970 = cos_h_u002e_sroa_u002e_0_u002e_0_1;
                let _e971 = max(_e759, _e970);
                out_u002e_10_ = _e971;
                let _e973 = (uv.x - _e768);
                let _e974 = (uv.y - _e769);
                if ((_e974 <= 1f) & ((_e973 <= 1f) & ((_e973 >= 0f) & (_e974 >= 0f)))) {
                    let _e997 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(min(u32((_e973 * _e8)), u32((_e8 + -1f)))), bitcast<i32>(min(u32((_e974 * _e11)), u32((_e11 + -1f))))), 0i);
                    if (_e997 < 1f) {
                        let _e1003 = ((_e973 * 2f) + -1f);
                        let _e1007 = (1f - (_e974 * 2f));
                        let _e1011 = uniform_struct_0_2_.f4_.x;
                        let _e1016 = uniform_struct_0_2_.f5_.x;
                        let _e1022 = uniform_struct_0_2_.f6_.x;
                        let _e1028 = uniform_struct_0_2_.f7_.x;
                        let _e1033 = uniform_struct_0_2_.f4_.w;
                        let _e1038 = uniform_struct_0_2_.f5_.w;
                        let _e1044 = uniform_struct_0_2_.f6_.w;
                        let _e1050 = uniform_struct_0_2_.f7_.w;
                        let _e1056 = uniform_struct_0_2_.f4_.y;
                        let _e1061 = uniform_struct_0_2_.f5_.y;
                        let _e1067 = uniform_struct_0_2_.f6_.y;
                        let _e1073 = uniform_struct_0_2_.f7_.y;
                        let _e1078 = uniform_struct_0_2_.f4_.w;
                        let _e1083 = uniform_struct_0_2_.f5_.w;
                        let _e1089 = uniform_struct_0_2_.f6_.w;
                        let _e1095 = uniform_struct_0_2_.f7_.w;
                        let _e1101 = uniform_struct_0_2_.f4_.z;
                        let _e1106 = uniform_struct_0_2_.f5_.z;
                        let _e1112 = uniform_struct_0_2_.f6_.z;
                        let _e1118 = uniform_struct_0_2_.f7_.z;
                        let _e1123 = uniform_struct_0_2_.f4_.w;
                        let _e1128 = uniform_struct_0_2_.f5_.w;
                        let _e1134 = uniform_struct_0_2_.f6_.w;
                        let _e1140 = uniform_struct_0_2_.f7_.w;
                        let _e1143 = (((_e1028 + (((_e1003 * _e1011) + (_e1007 * _e1016)) + (_e997 * _e1022))) / (_e1050 + (((_e1003 * _e1033) + (_e1007 * _e1038)) + (_e997 * _e1044)))) - _e117);
                        let _e1144 = (((_e1073 + (((_e1003 * _e1056) + (_e1007 * _e1061)) + (_e997 * _e1067))) / (_e1095 + (((_e1003 * _e1078) + (_e1007 * _e1083)) + (_e997 * _e1089)))) - _e162);
                        let _e1145 = (((_e1118 + (((_e1003 * _e1101) + (_e1007 * _e1106)) + (_e997 * _e1112))) / (_e1140 + (((_e1003 * _e1123) + (_e1007 * _e1128)) + (_e997 * _e1134)))) - _e207);
                        let _e1147 = length(vec3<f32>(_e1143, _e1144, _e1145));
                        if ((_e1147 > 0.000001f) & (_e1147 < _e544)) {
                            cos_h_u002e_sroa_u002e_0_u002e_0_ = mix(_e753, dot(vec3<f32>((_e1143 / _e1147), (_e1144 / _e1147), (_e1145 / _e1147)), vec3<f32>(_e227, _e228, _e229)), clamp((1f - (_e1147 / max(_e544, 0.000001f))), 0f, 1f));
                        } else {
                            cos_h_u002e_sroa_u002e_0_u002e_0_ = -1f;
                        }
                    } else {
                        cos_h_u002e_sroa_u002e_0_u002e_0_ = -1f;
                    }
                } else {
                    cos_h_u002e_sroa_u002e_0_u002e_0_ = -1f;
                }
                let _e1173 = cos_h_u002e_sroa_u002e_0_u002e_0_;
                let _e1174 = max(_e761, _e1173);
                out_u002e_11_ = _e1174;
                let _e1177 = (_e763 + 1u);
                if (_e1177 == 5u) {
                    break;
                } else {
                    horizon_pos_u002e_sroa_u002e_0_ = _e971;
                    horizon_neg_u002e_sroa_u002e_0_ = _e1174;
                    step_u002e_sroa_u002e_0_ = _e1177;
                    continue;
                }
            }
            let _e1186 = out_u002e_10_;
            let _e1188 = out_u002e_11_;
            let _e1198 = (acos(clamp(_e1186, -1f, 1f)) * 2f);
            let _e1209 = (acos(clamp(_e1188, -1f, 1f)) * -2f);
            let _e1221 = (_e672 + (_e734 * ((((cos(_e747) - cos((_e1198 - _e747))) + (_e1198 * sin(_e747))) * 0.25f) + (((cos(_e747) - cos((_e1209 - _e747))) + (_e1209 * sin(_e747))) * 0.25f))));
            out_u002e_8_ = _e1221;
            if (_e674 == 0u) {
                visibility_u002e_sroa_u002e_0_ = _e1221;
                slice_u002e_sroa_u002e_0_ = (_e674 + 1u);
                continue;
            } else {
                break;
            }
        }
        let _e1230 = out_u002e_8_;
        let _e1237 = clamp(mix(1f, (_e1230 * 0.5f), _e547), 0f, 1f);
        fragment_out = vec4<f32>(_e1237, _e1237, _e1237, 1f);
    } else {
        fragment_out = vec4<f32>(1f, 1f, 1f, 1f);
    }
    let _e1248 = fragment_out;
    return _e1248;
}
