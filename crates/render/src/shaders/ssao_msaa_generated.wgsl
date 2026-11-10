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
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var occluded_u002e_sroa_u002e_0_u002e_0146_: f32;
    var index_u002e_sroa_u002e_0_u002e_0145_: u32;
    var out_u002e_4_: f32;

    let _e5 = textureDimensions(tex_0_0_);
    let _e8 = f32(bitcast<i32>(_e5.x));
    let _e10 = textureDimensions(tex_0_0_);
    let _e13 = f32(bitcast<i32>(_e10.y));
    let _e15 = u32((uv.x * _e8));
    let _e17 = u32((uv.y * _e13));
    let _e23 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(_e15), bitcast<i32>(_e17)), 0u);
    if !((_e23 >= 1f)) {
        let _e32 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(_e15), bitcast<i32>(_e17)), 0u);
        let _e39 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(_e15), bitcast<i32>(_e17)), 0u);
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
        let _e125 = uniform_struct_0_2_.f4_.y;
        let _e130 = uniform_struct_0_2_.f5_.y;
        let _e136 = uniform_struct_0_2_.f6_.y;
        let _e142 = uniform_struct_0_2_.f7_.y;
        let _e147 = uniform_struct_0_2_.f4_.w;
        let _e152 = uniform_struct_0_2_.f5_.w;
        let _e158 = uniform_struct_0_2_.f6_.w;
        let _e164 = uniform_struct_0_2_.f7_.w;
        let _e170 = uniform_struct_0_2_.f4_.z;
        let _e175 = uniform_struct_0_2_.f5_.z;
        let _e181 = uniform_struct_0_2_.f6_.z;
        let _e187 = uniform_struct_0_2_.f7_.z;
        let _e192 = uniform_struct_0_2_.f4_.w;
        let _e197 = uniform_struct_0_2_.f5_.w;
        let _e203 = uniform_struct_0_2_.f6_.w;
        let _e209 = uniform_struct_0_2_.f7_.w;
        let _e234 = (cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).z + 0.000001f);
        let _e268 = (fract((fract(dot(vec2<f32>(f32(_e15), f32(_e17)), vec2<f32>(0.06711056f, 0.00583715f))) * 52.982918f)) * 6.2831855f);
        let _e269 = cos(_e268);
        let _e270 = sin(_e268);
        let _e273 = uniform_struct_0_3_.f0_;
        let _e276 = uniform_struct_0_3_.f1_;
        let _e279 = uniform_struct_0_3_.f2_;
        occluded_u002e_sroa_u002e_0_u002e_0146_ = 0f;
        index_u002e_sroa_u002e_0_u002e_0145_ = 0u;
        loop {
            let _e291 = occluded_u002e_sroa_u002e_0_u002e_0146_;
            let _e293 = index_u002e_sroa_u002e_0_u002e_0145_;
            let _e296 = (f32(_e293) + 0.5f);
            let _e298 = (_e296 * 0.125f);
            let _e300 = (_e296 * 2.399963f);
            let _e302 = (_e298 * cos(_e300));
            let _e304 = (_e298 * sin(_e300));
            let _e307 = ((_e269 * _e302) - (_e270 * _e304));
            let _e310 = ((_e270 * _e302) + (_e269 * _e304));
            let _e314 = ((normalize(vec3<f32>(_e54, _e59, _e45)).x * 0.75f) + ((normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e234)).x * _e307) + (cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e234)).x, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e234)).y, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e234)).z)).x * _e310)));
            let _e318 = ((normalize(vec3<f32>(_e54, _e59, _e45)).y * 0.75f) + ((normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e234)).y * _e307) + (cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e234)).x, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e234)).y, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e234)).z)).y * _e310)));
            let _e322 = ((normalize(vec3<f32>(_e54, _e59, _e45)).z * 0.75f) + ((normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e234)).z * _e307) + (cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e234)).x, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e234)).y, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(0f, 1f, 0f)).y, _e234)).z)).z * _e310)));
            let _e333 = (((_e97 + (((_e72 * _e80) + (_e76 * _e85)) + (_e23 * _e91))) / (_e119 + (((_e72 * _e102) + (_e76 * _e107)) + (_e23 * _e113)))) + (_e273 * normalize(vec3<f32>(_e314, _e318, _e322)).x));
            let _e335 = (((_e142 + (((_e72 * _e125) + (_e76 * _e130)) + (_e23 * _e136))) / (_e164 + (((_e72 * _e147) + (_e76 * _e152)) + (_e23 * _e158)))) + (_e273 * normalize(vec3<f32>(_e314, _e318, _e322)).y));
            let _e337 = (((_e187 + (((_e72 * _e170) + (_e76 * _e175)) + (_e23 * _e181))) / (_e209 + (((_e72 * _e192) + (_e76 * _e197)) + (_e23 * _e203)))) + (_e273 * normalize(vec3<f32>(_e314, _e318, _e322)).z));
            let _e341 = uniform_struct_0_2_.f0_.w;
            let _e346 = uniform_struct_0_2_.f1_.w;
            let _e352 = uniform_struct_0_2_.f2_.w;
            let _e358 = uniform_struct_0_2_.f3_.w;
            let _e361 = max((_e358 + (((_e333 * _e341) + (_e335 * _e346)) + (_e337 * _e352))), 0.000001f);
            let _e365 = uniform_struct_0_2_.f0_.z;
            let _e370 = uniform_struct_0_2_.f1_.z;
            let _e376 = uniform_struct_0_2_.f2_.z;
            let _e382 = uniform_struct_0_2_.f3_.z;
            if (((_e382 + (((_e333 * _e365) + (_e335 * _e370)) + (_e337 * _e376))) / _e361) > 1f) {
                _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
            } else {
                let _e392 = uniform_struct_0_2_.f0_.x;
                let _e397 = uniform_struct_0_2_.f1_.x;
                let _e403 = uniform_struct_0_2_.f2_.x;
                let _e409 = uniform_struct_0_2_.f3_.x;
                let _e415 = ((((_e409 + (((_e333 * _e392) + (_e335 * _e397)) + (_e337 * _e403))) / _e361) * 0.5f) + 0.5f);
                let _e419 = uniform_struct_0_2_.f0_.y;
                let _e424 = uniform_struct_0_2_.f1_.y;
                let _e430 = uniform_struct_0_2_.f2_.y;
                let _e436 = uniform_struct_0_2_.f3_.y;
                let _e442 = (0.5f - (((_e436 + (((_e333 * _e419) + (_e335 * _e424)) + (_e337 * _e430))) / _e361) * 0.5f));
                if ((_e442 > 1f) | (((_e415 < 0f) | (_e415 > 1f)) | (_e442 < 0f))) {
                    _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
                } else {
                    let _e465 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(u32((_e415 * _e8))), bitcast<i32>(u32((_e442 * _e13)))), 0u);
                    if !((_e465 >= 1f)) {
                        let _e472 = ((_e415 * 2f) + -1f);
                        let _e476 = (1f - (_e442 * 2f));
                        let _e480 = uniform_struct_0_2_.f4_.x;
                        let _e485 = uniform_struct_0_2_.f5_.x;
                        let _e491 = uniform_struct_0_2_.f6_.x;
                        let _e497 = uniform_struct_0_2_.f7_.x;
                        let _e502 = uniform_struct_0_2_.f4_.w;
                        let _e507 = uniform_struct_0_2_.f5_.w;
                        let _e513 = uniform_struct_0_2_.f6_.w;
                        let _e519 = uniform_struct_0_2_.f7_.w;
                        let _e521 = ((_e497 + (((_e472 * _e480) + (_e476 * _e485)) + (_e465 * _e491))) / (_e519 + (((_e472 * _e502) + (_e476 * _e507)) + (_e465 * _e513))));
                        let _e525 = uniform_struct_0_2_.f4_.y;
                        let _e530 = uniform_struct_0_2_.f5_.y;
                        let _e536 = uniform_struct_0_2_.f6_.y;
                        let _e542 = uniform_struct_0_2_.f7_.y;
                        let _e547 = uniform_struct_0_2_.f4_.w;
                        let _e552 = uniform_struct_0_2_.f5_.w;
                        let _e558 = uniform_struct_0_2_.f6_.w;
                        let _e564 = uniform_struct_0_2_.f7_.w;
                        let _e566 = ((_e542 + (((_e472 * _e525) + (_e476 * _e530)) + (_e465 * _e536))) / (_e564 + (((_e472 * _e547) + (_e476 * _e552)) + (_e465 * _e558))));
                        let _e570 = uniform_struct_0_2_.f4_.z;
                        let _e575 = uniform_struct_0_2_.f5_.z;
                        let _e581 = uniform_struct_0_2_.f6_.z;
                        let _e587 = uniform_struct_0_2_.f7_.z;
                        let _e592 = uniform_struct_0_2_.f4_.w;
                        let _e597 = uniform_struct_0_2_.f5_.w;
                        let _e603 = uniform_struct_0_2_.f6_.w;
                        let _e609 = uniform_struct_0_2_.f7_.w;
                        let _e611 = ((_e587 + (((_e472 * _e570) + (_e476 * _e575)) + (_e465 * _e581))) / (_e609 + (((_e472 * _e592) + (_e476 * _e597)) + (_e465 * _e603))));
                        let _e616 = length(vec3<f32>((_e333 - _e521), (_e335 - _e566), (_e337 - _e611)));
                        if (_e616 < 0.000001f) {
                            _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
                        } else {
                            let _e624 = uniform_struct_0_2_.f8_.x;
                            let _e628 = uniform_struct_0_2_.f8_.y;
                            let _e632 = uniform_struct_0_2_.f8_.z;
                            if !((length(vec3<f32>((_e624 - _e521), (_e628 - _e566), (_e632 - _e611))) >= (length(vec3<f32>((_e624 - _e333), (_e628 - _e335), (_e632 - _e337))) - _e279))) {
                                _0_u002e_sroa_u002e_0_u002e_0_u002e_i = max((1f - (_e616 / _e273)), 0f);
                            } else {
                                _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
                            }
                        }
                    } else {
                        _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
                    }
                }
            }
            let _e657 = _0_u002e_sroa_u002e_0_u002e_0_u002e_i;
            let _e658 = (_e291 + _e657);
            out_u002e_4_ = _e658;
            let _e661 = (_e293 + 1u);
            if (_e661 == 8u) {
                break;
            } else {
                occluded_u002e_sroa_u002e_0_u002e_0146_ = _e658;
                index_u002e_sroa_u002e_0_u002e_0145_ = _e661;
                continue;
            }
        }
        let _e667 = out_u002e_4_;
        let _e675 = clamp((1f - (_e276 * (_e667 * 0.125f))), 0f, 1f);
        fragment_out = vec4<f32>(_e675, _e675, _e675, 1f);
    } else {
        fragment_out = vec4<f32>(1f, 1f, 1f, 1f);
    }
    let _e686 = fragment_out;
    return _e686;
}
