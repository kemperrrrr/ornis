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
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var occluded_u002e_sroa_u002e_0_u002e_0139_: f32;
    var index_u002e_sroa_u002e_0_u002e_0138_: u32;
    var out_u002e_10_: f32;

    let _e5 = textureDimensions(tex_0_0_);
    let _e8 = f32(bitcast<i32>(_e5.x));
    let _e10 = textureDimensions(tex_0_0_);
    let _e13 = f32(bitcast<i32>(_e10.y));
    let _e15 = u32((uv.x * _e8));
    let _e17 = u32((uv.y * _e13));
    let _e23 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(_e15), bitcast<i32>(_e17)), 0i);
    if !((_e23 >= 1f)) {
        let _e32 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(_e15), bitcast<i32>(_e17)), 0i);
        let _e39 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(_e15), bitcast<i32>(_e17)), 0i);
        let _e45 = ((1f - abs(_e32.x)) - abs(_e39.y));
        let _e48 = max(-(_e45), 0f);
        let _e52 = -(_e48);
        let _e54 = (_e32.x + select(_e52, _e48, !((_e32.x >= 0f))));
        let _e59 = (_e39.y + select(_e52, _e48, !((_e39.y >= 0f))));
        let _e72 = ((uv.x * 2f) + -1f);
        let _e76 = (1f - (uv.y * 2f));
        let _e80 = uniform_struct_0_2_.f4_.x;
        let _e85 = uniform_struct_0_2_.f5_.x;
        let _e91 = uniform_struct_0_2_.f6_.x;
        let _e97 = uniform_struct_0_2_.f7_.x;
        let _e102 = uniform_struct_0_2_.f4_.w;
        let _e107 = uniform_struct_0_2_.f5_.w;
        let _e113 = uniform_struct_0_2_.f6_.w;
        let _e119 = uniform_struct_0_2_.f7_.w;
        let _e127 = uniform_struct_0_2_.f4_.y;
        let _e132 = uniform_struct_0_2_.f5_.y;
        let _e138 = uniform_struct_0_2_.f6_.y;
        let _e144 = uniform_struct_0_2_.f7_.y;
        let _e149 = uniform_struct_0_2_.f4_.w;
        let _e154 = uniform_struct_0_2_.f5_.w;
        let _e160 = uniform_struct_0_2_.f6_.w;
        let _e166 = uniform_struct_0_2_.f7_.w;
        let _e174 = uniform_struct_0_2_.f4_.z;
        let _e179 = uniform_struct_0_2_.f5_.z;
        let _e185 = uniform_struct_0_2_.f6_.z;
        let _e191 = uniform_struct_0_2_.f7_.z;
        let _e196 = uniform_struct_0_2_.f4_.w;
        let _e201 = uniform_struct_0_2_.f5_.w;
        let _e207 = uniform_struct_0_2_.f6_.w;
        let _e213 = uniform_struct_0_2_.f7_.w;
        let _e240 = (cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).z + 0.000001f);
        let _e274 = (fract((fract(((f32(_e15) * 0.06711056f) + (f32(_e17) * 0.00583715f))) * 52.982918f)) * 6.2831855f);
        let _e275 = cos(_e274);
        let _e276 = sin(_e274);
        let _e279 = uniform_struct_0_3_.f0_;
        let _e282 = uniform_struct_0_3_.f1_;
        let _e285 = uniform_struct_0_3_.f2_;
        occluded_u002e_sroa_u002e_0_u002e_0139_ = 0f;
        index_u002e_sroa_u002e_0_u002e_0138_ = 0u;
        loop {
            let _e297 = occluded_u002e_sroa_u002e_0_u002e_0139_;
            let _e299 = index_u002e_sroa_u002e_0_u002e_0138_;
            let _e302 = (f32(_e299) + 0.5f);
            let _e304 = (_e302 * 0.125f);
            let _e306 = (_e302 * 2.399963f);
            let _e308 = (_e304 * cos(_e306));
            let _e310 = (_e304 * sin(_e306));
            let _e313 = ((_e275 * _e308) - (_e276 * _e310));
            let _e316 = ((_e276 * _e308) + (_e275 * _e310));
            let _e320 = ((normalize(vec3<f32>(_e54, _e59, _e45)).x * 0.75f) + ((normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e240)).x * _e313) + (cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e240)).x, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e240)).y, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e240)).z)).x * _e316)));
            let _e324 = ((normalize(vec3<f32>(_e54, _e59, _e45)).y * 0.75f) + ((normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e240)).y * _e313) + (cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e240)).x, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e240)).y, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e240)).z)).y * _e316)));
            let _e328 = ((normalize(vec3<f32>(_e54, _e59, _e45)).z * 0.75f) + ((normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e240)).z * _e313) + (cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e240)).x, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e240)).y, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e240)).z)).z * _e316)));
            let _e339 = (((_e97 + (((_e72 * _e80) + (_e76 * _e85)) + (_e23 * _e91))) / max((_e119 + (((_e72 * _e102) + (_e76 * _e107)) + (_e23 * _e113))), 0.000001f)) + (_e279 * normalize(vec3<f32>(_e320, _e324, _e328)).x));
            let _e341 = (((_e144 + (((_e72 * _e127) + (_e76 * _e132)) + (_e23 * _e138))) / max((_e166 + (((_e72 * _e149) + (_e76 * _e154)) + (_e23 * _e160))), 0.000001f)) + (_e279 * normalize(vec3<f32>(_e320, _e324, _e328)).y));
            let _e343 = (((_e191 + (((_e72 * _e174) + (_e76 * _e179)) + (_e23 * _e185))) / max((_e213 + (((_e72 * _e196) + (_e76 * _e201)) + (_e23 * _e207))), 0.000001f)) + (_e279 * normalize(vec3<f32>(_e320, _e324, _e328)).z));
            let _e347 = uniform_struct_0_2_.f0_.w;
            let _e352 = uniform_struct_0_2_.f1_.w;
            let _e358 = uniform_struct_0_2_.f2_.w;
            let _e364 = uniform_struct_0_2_.f3_.w;
            let _e367 = max((_e364 + (((_e339 * _e347) + (_e341 * _e352)) + (_e343 * _e358))), 0.000001f);
            let _e371 = uniform_struct_0_2_.f0_.z;
            let _e376 = uniform_struct_0_2_.f1_.z;
            let _e382 = uniform_struct_0_2_.f2_.z;
            let _e388 = uniform_struct_0_2_.f3_.z;
            if (((_e388 + (((_e339 * _e371) + (_e341 * _e376)) + (_e343 * _e382))) / _e367) > 1f) {
                _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
            } else {
                let _e398 = uniform_struct_0_2_.f0_.x;
                let _e403 = uniform_struct_0_2_.f1_.x;
                let _e409 = uniform_struct_0_2_.f2_.x;
                let _e415 = uniform_struct_0_2_.f3_.x;
                let _e421 = ((((_e415 + (((_e339 * _e398) + (_e341 * _e403)) + (_e343 * _e409))) / _e367) * 0.5f) + 0.5f);
                let _e425 = uniform_struct_0_2_.f0_.y;
                let _e430 = uniform_struct_0_2_.f1_.y;
                let _e436 = uniform_struct_0_2_.f2_.y;
                let _e442 = uniform_struct_0_2_.f3_.y;
                let _e448 = (0.5f - (((_e442 + (((_e339 * _e425) + (_e341 * _e430)) + (_e343 * _e436))) / _e367) * 0.5f));
                if ((_e448 > 1f) | (((_e421 < 0f) | (_e421 > 1f)) | (_e448 < 0f))) {
                    _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
                } else {
                    let _e471 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(u32((_e421 * _e8))), bitcast<i32>(u32((_e448 * _e13)))), 0i);
                    if !((_e471 >= 1f)) {
                        let _e478 = ((_e421 * 2f) + -1f);
                        let _e482 = (1f - (_e448 * 2f));
                        let _e486 = uniform_struct_0_2_.f4_.x;
                        let _e491 = uniform_struct_0_2_.f5_.x;
                        let _e497 = uniform_struct_0_2_.f6_.x;
                        let _e503 = uniform_struct_0_2_.f7_.x;
                        let _e508 = uniform_struct_0_2_.f4_.w;
                        let _e513 = uniform_struct_0_2_.f5_.w;
                        let _e519 = uniform_struct_0_2_.f6_.w;
                        let _e525 = uniform_struct_0_2_.f7_.w;
                        let _e529 = ((_e503 + (((_e478 * _e486) + (_e482 * _e491)) + (_e471 * _e497))) / max((_e525 + (((_e478 * _e508) + (_e482 * _e513)) + (_e471 * _e519))), 0.000001f));
                        let _e533 = uniform_struct_0_2_.f4_.y;
                        let _e538 = uniform_struct_0_2_.f5_.y;
                        let _e544 = uniform_struct_0_2_.f6_.y;
                        let _e550 = uniform_struct_0_2_.f7_.y;
                        let _e555 = uniform_struct_0_2_.f4_.w;
                        let _e560 = uniform_struct_0_2_.f5_.w;
                        let _e566 = uniform_struct_0_2_.f6_.w;
                        let _e572 = uniform_struct_0_2_.f7_.w;
                        let _e576 = ((_e550 + (((_e478 * _e533) + (_e482 * _e538)) + (_e471 * _e544))) / max((_e572 + (((_e478 * _e555) + (_e482 * _e560)) + (_e471 * _e566))), 0.000001f));
                        let _e580 = uniform_struct_0_2_.f4_.z;
                        let _e585 = uniform_struct_0_2_.f5_.z;
                        let _e591 = uniform_struct_0_2_.f6_.z;
                        let _e597 = uniform_struct_0_2_.f7_.z;
                        let _e602 = uniform_struct_0_2_.f4_.w;
                        let _e607 = uniform_struct_0_2_.f5_.w;
                        let _e613 = uniform_struct_0_2_.f6_.w;
                        let _e619 = uniform_struct_0_2_.f7_.w;
                        let _e623 = ((_e597 + (((_e478 * _e580) + (_e482 * _e585)) + (_e471 * _e591))) / max((_e619 + (((_e478 * _e602) + (_e482 * _e607)) + (_e471 * _e613))), 0.000001f));
                        let _e628 = length(vec3<f32>((_e339 - _e529), (_e341 - _e576), (_e343 - _e623)));
                        if (_e628 < 0.000001f) {
                            _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
                        } else {
                            let _e636 = uniform_struct_0_2_.f8_.x;
                            let _e640 = uniform_struct_0_2_.f8_.y;
                            let _e644 = uniform_struct_0_2_.f8_.z;
                            if !((length(vec3<f32>((_e636 - _e529), (_e640 - _e576), (_e644 - _e623))) >= (length(vec3<f32>((_e636 - _e339), (_e640 - _e341), (_e644 - _e343))) - _e285))) {
                                _0_u002e_sroa_u002e_0_u002e_0_u002e_i = max((1f - (_e628 / _e279)), 0f);
                            } else {
                                _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
                            }
                        }
                    } else {
                        _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
                    }
                }
            }
            let _e669 = _0_u002e_sroa_u002e_0_u002e_0_u002e_i;
            let _e670 = (_e297 + _e669);
            out_u002e_10_ = _e670;
            let _e673 = (_e299 + 1u);
            if (_e673 == 8u) {
                break;
            } else {
                occluded_u002e_sroa_u002e_0_u002e_0139_ = _e670;
                index_u002e_sroa_u002e_0_u002e_0138_ = _e673;
                continue;
            }
        }
        let _e679 = out_u002e_10_;
        let _e687 = clamp((1f - (_e282 * (_e679 * 0.125f))), 0f, 1f);
        fragment_out = vec4<f32>(_e687, _e687, _e687, 1f);
    } else {
        fragment_out = vec4<f32>(1f, 1f, 1f, 1f);
    }
    let _e698 = fragment_out;
    return _e698;
}
