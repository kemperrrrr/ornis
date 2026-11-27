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
    var _0_u002e_sroa_u002e_0_u002e_0_: f32;
    var occluded_u002e_sroa_u002e_0_: f32;
    var index_u002e_sroa_u002e_0_: u32;
    var out_u002e_4_: f32;

    let _e5 = textureDimensions(tex_0_0_);
    let _e8 = f32(bitcast<i32>(_e5.x));
    let _e11 = f32(bitcast<i32>(_e5.y));
    let _e13 = u32((uv.x * _e8));
    let _e15 = u32((uv.y * _e11));
    let _e20 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0u);
    if !((_e20 >= 1f)) {
        let _e29 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0u);
        let _e35 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0u);
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
        let _e121 = uniform_struct_0_2_.f4_.y;
        let _e126 = uniform_struct_0_2_.f5_.y;
        let _e132 = uniform_struct_0_2_.f6_.y;
        let _e138 = uniform_struct_0_2_.f7_.y;
        let _e143 = uniform_struct_0_2_.f4_.w;
        let _e148 = uniform_struct_0_2_.f5_.w;
        let _e154 = uniform_struct_0_2_.f6_.w;
        let _e160 = uniform_struct_0_2_.f7_.w;
        let _e166 = uniform_struct_0_2_.f4_.z;
        let _e171 = uniform_struct_0_2_.f5_.z;
        let _e177 = uniform_struct_0_2_.f6_.z;
        let _e183 = uniform_struct_0_2_.f7_.z;
        let _e188 = uniform_struct_0_2_.f4_.w;
        let _e193 = uniform_struct_0_2_.f5_.w;
        let _e199 = uniform_struct_0_2_.f6_.w;
        let _e205 = uniform_struct_0_2_.f7_.w;
        let _e230 = (cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).z + 0.000001f);
        let _e264 = (fract((fract(dot(vec2<f32>(f32(_e13), f32(_e15)), vec2<f32>(0.06711056f, 0.00583715f))) * 52.982918f)) * 6.2831855f);
        let _e265 = cos(_e264);
        let _e266 = sin(_e264);
        let _e269 = uniform_struct_0_3_.f0_;
        let _e272 = uniform_struct_0_3_.f1_;
        let _e275 = uniform_struct_0_3_.f2_;
        occluded_u002e_sroa_u002e_0_ = 0f;
        index_u002e_sroa_u002e_0_ = 0u;
        loop {
            let _e287 = occluded_u002e_sroa_u002e_0_;
            let _e289 = index_u002e_sroa_u002e_0_;
            let _e292 = (f32(_e289) + 0.5f);
            let _e294 = (_e292 * 0.125f);
            let _e296 = (_e292 * 2.399963f);
            let _e298 = (_e294 * cos(_e296));
            let _e300 = (_e294 * sin(_e296));
            let _e303 = ((_e265 * _e298) - (_e266 * _e300));
            let _e306 = ((_e266 * _e298) + (_e265 * _e300));
            let _e310 = ((normalize(vec3<f32>(_e50, _e55, _e41)).x * 0.75f) + ((normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).y, _e230)).x * _e303) + (cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).y, _e230)).x, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).y, _e230)).y, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).y, _e230)).z)).x * _e306)));
            let _e314 = ((normalize(vec3<f32>(_e50, _e55, _e41)).y * 0.75f) + ((normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).y, _e230)).y * _e303) + (cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).y, _e230)).x, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).y, _e230)).y, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).y, _e230)).z)).y * _e306)));
            let _e318 = ((normalize(vec3<f32>(_e50, _e55, _e41)).z * 0.75f) + ((normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).y, _e230)).z * _e303) + (cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).y, _e230)).x, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).y, _e230)).y, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e55, _e41)).x, normalize(vec3<f32>(_e50, _e55, _e41)).y, normalize(vec3<f32>(_e50, _e55, _e41)).z), vec3<f32>(0f, 1f, 0f)).y, _e230)).z)).z * _e306)));
            let _e329 = (((_e93 + (((_e68 * _e76) + (_e72 * _e81)) + (_e20 * _e87))) / (_e115 + (((_e68 * _e98) + (_e72 * _e103)) + (_e20 * _e109)))) + (_e269 * normalize(vec3<f32>(_e310, _e314, _e318)).x));
            let _e331 = (((_e138 + (((_e68 * _e121) + (_e72 * _e126)) + (_e20 * _e132))) / (_e160 + (((_e68 * _e143) + (_e72 * _e148)) + (_e20 * _e154)))) + (_e269 * normalize(vec3<f32>(_e310, _e314, _e318)).y));
            let _e333 = (((_e183 + (((_e68 * _e166) + (_e72 * _e171)) + (_e20 * _e177))) / (_e205 + (((_e68 * _e188) + (_e72 * _e193)) + (_e20 * _e199)))) + (_e269 * normalize(vec3<f32>(_e310, _e314, _e318)).z));
            let _e337 = uniform_struct_0_2_.f0_.w;
            let _e342 = uniform_struct_0_2_.f1_.w;
            let _e348 = uniform_struct_0_2_.f2_.w;
            let _e354 = uniform_struct_0_2_.f3_.w;
            let _e357 = max((_e354 + (((_e329 * _e337) + (_e331 * _e342)) + (_e333 * _e348))), 0.000001f);
            let _e361 = uniform_struct_0_2_.f0_.z;
            let _e366 = uniform_struct_0_2_.f1_.z;
            let _e372 = uniform_struct_0_2_.f2_.z;
            let _e378 = uniform_struct_0_2_.f3_.z;
            if (((_e378 + (((_e329 * _e361) + (_e331 * _e366)) + (_e333 * _e372))) / _e357) > 1f) {
                _0_u002e_sroa_u002e_0_u002e_0_ = 0f;
            } else {
                let _e388 = uniform_struct_0_2_.f0_.x;
                let _e393 = uniform_struct_0_2_.f1_.x;
                let _e399 = uniform_struct_0_2_.f2_.x;
                let _e405 = uniform_struct_0_2_.f3_.x;
                let _e411 = ((((_e405 + (((_e329 * _e388) + (_e331 * _e393)) + (_e333 * _e399))) / _e357) * 0.5f) + 0.5f);
                let _e415 = uniform_struct_0_2_.f0_.y;
                let _e420 = uniform_struct_0_2_.f1_.y;
                let _e426 = uniform_struct_0_2_.f2_.y;
                let _e432 = uniform_struct_0_2_.f3_.y;
                let _e438 = (0.5f - (((_e432 + (((_e329 * _e415) + (_e331 * _e420)) + (_e333 * _e426))) / _e357) * 0.5f));
                if ((_e438 > 1f) | (((_e411 < 0f) | (_e411 > 1f)) | (_e438 < 0f))) {
                    _0_u002e_sroa_u002e_0_u002e_0_ = 0f;
                } else {
                    let _e461 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(u32((_e411 * _e8))), bitcast<i32>(u32((_e438 * _e11)))), 0u);
                    if !((_e461 >= 1f)) {
                        let _e468 = ((_e411 * 2f) + -1f);
                        let _e472 = (1f - (_e438 * 2f));
                        let _e476 = uniform_struct_0_2_.f4_.x;
                        let _e481 = uniform_struct_0_2_.f5_.x;
                        let _e487 = uniform_struct_0_2_.f6_.x;
                        let _e493 = uniform_struct_0_2_.f7_.x;
                        let _e498 = uniform_struct_0_2_.f4_.w;
                        let _e503 = uniform_struct_0_2_.f5_.w;
                        let _e509 = uniform_struct_0_2_.f6_.w;
                        let _e515 = uniform_struct_0_2_.f7_.w;
                        let _e517 = ((_e493 + (((_e468 * _e476) + (_e472 * _e481)) + (_e461 * _e487))) / (_e515 + (((_e468 * _e498) + (_e472 * _e503)) + (_e461 * _e509))));
                        let _e521 = uniform_struct_0_2_.f4_.y;
                        let _e526 = uniform_struct_0_2_.f5_.y;
                        let _e532 = uniform_struct_0_2_.f6_.y;
                        let _e538 = uniform_struct_0_2_.f7_.y;
                        let _e543 = uniform_struct_0_2_.f4_.w;
                        let _e548 = uniform_struct_0_2_.f5_.w;
                        let _e554 = uniform_struct_0_2_.f6_.w;
                        let _e560 = uniform_struct_0_2_.f7_.w;
                        let _e562 = ((_e538 + (((_e468 * _e521) + (_e472 * _e526)) + (_e461 * _e532))) / (_e560 + (((_e468 * _e543) + (_e472 * _e548)) + (_e461 * _e554))));
                        let _e566 = uniform_struct_0_2_.f4_.z;
                        let _e571 = uniform_struct_0_2_.f5_.z;
                        let _e577 = uniform_struct_0_2_.f6_.z;
                        let _e583 = uniform_struct_0_2_.f7_.z;
                        let _e588 = uniform_struct_0_2_.f4_.w;
                        let _e593 = uniform_struct_0_2_.f5_.w;
                        let _e599 = uniform_struct_0_2_.f6_.w;
                        let _e605 = uniform_struct_0_2_.f7_.w;
                        let _e607 = ((_e583 + (((_e468 * _e566) + (_e472 * _e571)) + (_e461 * _e577))) / (_e605 + (((_e468 * _e588) + (_e472 * _e593)) + (_e461 * _e599))));
                        let _e612 = length(vec3<f32>((_e329 - _e517), (_e331 - _e562), (_e333 - _e607)));
                        if (_e612 < 0.000001f) {
                            _0_u002e_sroa_u002e_0_u002e_0_ = 0f;
                        } else {
                            let _e620 = uniform_struct_0_2_.f8_.x;
                            let _e624 = uniform_struct_0_2_.f8_.y;
                            let _e628 = uniform_struct_0_2_.f8_.z;
                            if !((length(vec3<f32>((_e620 - _e517), (_e624 - _e562), (_e628 - _e607))) >= (length(vec3<f32>((_e620 - _e329), (_e624 - _e331), (_e628 - _e333))) - _e275))) {
                                _0_u002e_sroa_u002e_0_u002e_0_ = max((1f - (_e612 / _e269)), 0f);
                            } else {
                                _0_u002e_sroa_u002e_0_u002e_0_ = 0f;
                            }
                        }
                    } else {
                        _0_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    }
                }
            }
            let _e653 = _0_u002e_sroa_u002e_0_u002e_0_;
            let _e654 = (_e287 + _e653);
            out_u002e_4_ = _e654;
            let _e657 = (_e289 + 1u);
            if (_e657 == 8u) {
                break;
            } else {
                occluded_u002e_sroa_u002e_0_ = _e654;
                index_u002e_sroa_u002e_0_ = _e657;
                continue;
            }
        }
        let _e663 = out_u002e_4_;
        let _e671 = clamp((1f - (_e272 * (_e663 * 0.125f))), 0f, 1f);
        fragment_out = vec4<f32>(_e671, _e671, _e671, 1f);
    } else {
        fragment_out = vec4<f32>(1f, 1f, 1f, 1f);
    }
    let _e682 = fragment_out;
    return _e682;
}
