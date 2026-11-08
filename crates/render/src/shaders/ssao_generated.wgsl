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
    let _e7 = f32(_e5.x);
    let _e9 = textureDimensions(tex_0_0_);
    let _e11 = f32(_e9.y);
    let _e13 = u32((uv.x * _e7));
    let _e15 = u32((uv.y * _e11));
    let _e21 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0i);
    if (_e21 < 1f) {
        let _e29 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0i);
        let _e36 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0i);
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
        let _e122 = uniform_struct_0_2_.f4_.y;
        let _e127 = uniform_struct_0_2_.f5_.y;
        let _e133 = uniform_struct_0_2_.f6_.y;
        let _e139 = uniform_struct_0_2_.f7_.y;
        let _e144 = uniform_struct_0_2_.f4_.w;
        let _e149 = uniform_struct_0_2_.f5_.w;
        let _e155 = uniform_struct_0_2_.f6_.w;
        let _e161 = uniform_struct_0_2_.f7_.w;
        let _e169 = uniform_struct_0_2_.f4_.z;
        let _e174 = uniform_struct_0_2_.f5_.z;
        let _e180 = uniform_struct_0_2_.f6_.z;
        let _e186 = uniform_struct_0_2_.f7_.z;
        let _e191 = uniform_struct_0_2_.f4_.w;
        let _e196 = uniform_struct_0_2_.f5_.w;
        let _e202 = uniform_struct_0_2_.f6_.w;
        let _e208 = uniform_struct_0_2_.f7_.w;
        let _e235 = (cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).z + 0.000001f);
        let _e269 = (fract((fract(((f32(_e13) * 0.06711056f) + (f32(_e15) * 0.00583715f))) * 52.982918f)) * 6.2831855f);
        let _e270 = cos(_e269);
        let _e271 = sin(_e269);
        let _e274 = uniform_struct_0_3_.f0_;
        let _e277 = uniform_struct_0_3_.f1_;
        let _e280 = uniform_struct_0_3_.f2_;
        occluded_u002e_sroa_u002e_0_u002e_0139_ = 0f;
        index_u002e_sroa_u002e_0_u002e_0138_ = 0u;
        loop {
            let _e292 = occluded_u002e_sroa_u002e_0_u002e_0139_;
            let _e294 = index_u002e_sroa_u002e_0_u002e_0138_;
            let _e297 = (f32(_e294) + 0.5f);
            let _e299 = (_e297 * 0.125f);
            let _e301 = (_e297 * 2.399963f);
            let _e303 = (_e299 * cos(_e301));
            let _e305 = (_e299 * sin(_e301));
            let _e308 = ((_e270 * _e303) - (_e271 * _e305));
            let _e311 = ((_e271 * _e303) + (_e270 * _e305));
            let _e315 = ((normalize(vec3<f32>(_e50, _e54, _e42)).x * 0.75f) + ((normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).y, _e235)).x * _e308) + (cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).y, _e235)).x, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).y, _e235)).y, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).y, _e235)).z)).x * _e311)));
            let _e319 = ((normalize(vec3<f32>(_e50, _e54, _e42)).y * 0.75f) + ((normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).y, _e235)).y * _e308) + (cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).y, _e235)).x, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).y, _e235)).y, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).y, _e235)).z)).y * _e311)));
            let _e323 = ((normalize(vec3<f32>(_e50, _e54, _e42)).z * 0.75f) + ((normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).y, _e235)).z * _e308) + (cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).y, _e235)).x, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).y, _e235)).y, normalize(vec3<f32>(cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).x, cross(vec3<f32>(normalize(vec3<f32>(_e50, _e54, _e42)).x, normalize(vec3<f32>(_e50, _e54, _e42)).y, normalize(vec3<f32>(_e50, _e54, _e42)).z), vec3<f32>(0f, 1f, 0f)).y, _e235)).z)).z * _e311)));
            let _e334 = (((_e92 + (((_e67 * _e75) + (_e71 * _e80)) + (_e21 * _e86))) / max((_e114 + (((_e67 * _e97) + (_e71 * _e102)) + (_e21 * _e108))), 0.000001f)) + (_e274 * normalize(vec3<f32>(_e315, _e319, _e323)).x));
            let _e336 = (((_e139 + (((_e67 * _e122) + (_e71 * _e127)) + (_e21 * _e133))) / max((_e161 + (((_e67 * _e144) + (_e71 * _e149)) + (_e21 * _e155))), 0.000001f)) + (_e274 * normalize(vec3<f32>(_e315, _e319, _e323)).y));
            let _e338 = (((_e186 + (((_e67 * _e169) + (_e71 * _e174)) + (_e21 * _e180))) / max((_e208 + (((_e67 * _e191) + (_e71 * _e196)) + (_e21 * _e202))), 0.000001f)) + (_e274 * normalize(vec3<f32>(_e315, _e319, _e323)).z));
            let _e342 = uniform_struct_0_2_.f0_.w;
            let _e347 = uniform_struct_0_2_.f1_.w;
            let _e353 = uniform_struct_0_2_.f2_.w;
            let _e359 = uniform_struct_0_2_.f3_.w;
            let _e362 = max((_e359 + (((_e334 * _e342) + (_e336 * _e347)) + (_e338 * _e353))), 0.000001f);
            let _e366 = uniform_struct_0_2_.f0_.z;
            let _e371 = uniform_struct_0_2_.f1_.z;
            let _e377 = uniform_struct_0_2_.f2_.z;
            let _e383 = uniform_struct_0_2_.f3_.z;
            if (((_e383 + (((_e334 * _e366) + (_e336 * _e371)) + (_e338 * _e377))) / _e362) > 1f) {
                _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
            } else {
                let _e393 = uniform_struct_0_2_.f0_.x;
                let _e398 = uniform_struct_0_2_.f1_.x;
                let _e404 = uniform_struct_0_2_.f2_.x;
                let _e410 = uniform_struct_0_2_.f3_.x;
                let _e416 = ((((_e410 + (((_e334 * _e393) + (_e336 * _e398)) + (_e338 * _e404))) / _e362) * 0.5f) + 0.5f);
                let _e420 = uniform_struct_0_2_.f0_.y;
                let _e425 = uniform_struct_0_2_.f1_.y;
                let _e431 = uniform_struct_0_2_.f2_.y;
                let _e437 = uniform_struct_0_2_.f3_.y;
                let _e443 = (0.5f - (((_e437 + (((_e334 * _e420) + (_e336 * _e425)) + (_e338 * _e431))) / _e362) * 0.5f));
                if ((_e443 > 1f) | (((_e416 < 0f) | (_e416 > 1f)) | (_e443 < 0f))) {
                    _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
                } else {
                    let _e466 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(u32((_e416 * _e7))), bitcast<i32>(u32((_e443 * _e11)))), 0i);
                    if (_e466 < 1f) {
                        let _e472 = ((_e416 * 2f) + -1f);
                        let _e476 = (1f - (_e443 * 2f));
                        let _e480 = uniform_struct_0_2_.f4_.x;
                        let _e485 = uniform_struct_0_2_.f5_.x;
                        let _e491 = uniform_struct_0_2_.f6_.x;
                        let _e497 = uniform_struct_0_2_.f7_.x;
                        let _e502 = uniform_struct_0_2_.f4_.w;
                        let _e507 = uniform_struct_0_2_.f5_.w;
                        let _e513 = uniform_struct_0_2_.f6_.w;
                        let _e519 = uniform_struct_0_2_.f7_.w;
                        let _e523 = ((_e497 + (((_e472 * _e480) + (_e476 * _e485)) + (_e466 * _e491))) / max((_e519 + (((_e472 * _e502) + (_e476 * _e507)) + (_e466 * _e513))), 0.000001f));
                        let _e527 = uniform_struct_0_2_.f4_.y;
                        let _e532 = uniform_struct_0_2_.f5_.y;
                        let _e538 = uniform_struct_0_2_.f6_.y;
                        let _e544 = uniform_struct_0_2_.f7_.y;
                        let _e549 = uniform_struct_0_2_.f4_.w;
                        let _e554 = uniform_struct_0_2_.f5_.w;
                        let _e560 = uniform_struct_0_2_.f6_.w;
                        let _e566 = uniform_struct_0_2_.f7_.w;
                        let _e570 = ((_e544 + (((_e472 * _e527) + (_e476 * _e532)) + (_e466 * _e538))) / max((_e566 + (((_e472 * _e549) + (_e476 * _e554)) + (_e466 * _e560))), 0.000001f));
                        let _e574 = uniform_struct_0_2_.f4_.z;
                        let _e579 = uniform_struct_0_2_.f5_.z;
                        let _e585 = uniform_struct_0_2_.f6_.z;
                        let _e591 = uniform_struct_0_2_.f7_.z;
                        let _e596 = uniform_struct_0_2_.f4_.w;
                        let _e601 = uniform_struct_0_2_.f5_.w;
                        let _e607 = uniform_struct_0_2_.f6_.w;
                        let _e613 = uniform_struct_0_2_.f7_.w;
                        let _e617 = ((_e591 + (((_e472 * _e574) + (_e476 * _e579)) + (_e466 * _e585))) / max((_e613 + (((_e472 * _e596) + (_e476 * _e601)) + (_e466 * _e607))), 0.000001f));
                        let _e622 = length(vec3<f32>((_e334 - _e523), (_e336 - _e570), (_e338 - _e617)));
                        if (_e622 < 0.000001f) {
                            _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
                        } else {
                            let _e630 = uniform_struct_0_2_.f8_.x;
                            let _e634 = uniform_struct_0_2_.f8_.y;
                            let _e638 = uniform_struct_0_2_.f8_.z;
                            if (length(vec3<f32>((_e630 - _e523), (_e634 - _e570), (_e638 - _e617))) < (length(vec3<f32>((_e630 - _e334), (_e634 - _e336), (_e638 - _e338))) - _e280)) {
                                _0_u002e_sroa_u002e_0_u002e_0_u002e_i = max((1f - (_e622 / _e274)), 0f);
                            } else {
                                _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
                            }
                        }
                    } else {
                        _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
                    }
                }
            }
            let _e662 = _0_u002e_sroa_u002e_0_u002e_0_u002e_i;
            let _e663 = (_e292 + _e662);
            out_u002e_10_ = _e663;
            let _e666 = (_e294 + 1u);
            if (_e666 == 8u) {
                break;
            } else {
                occluded_u002e_sroa_u002e_0_u002e_0139_ = _e663;
                index_u002e_sroa_u002e_0_u002e_0138_ = _e666;
                continue;
            }
        }
        let _e672 = out_u002e_10_;
        let _e680 = clamp((1f - (_e277 * (_e672 * 0.125f))), 0f, 1f);
        fragment_out = vec4<f32>(_e680, _e680, _e680, 1f);
    } else {
        fragment_out = vec4<f32>(1f, 1f, 1f, 1f);
    }
    let _e691 = fragment_out;
    return _e691;
}
