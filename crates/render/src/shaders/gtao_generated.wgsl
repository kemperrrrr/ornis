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
    var cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i419_: f32;
    var visibility_u002e_sroa_u002e_0_u002e_0578_: f32;
    var slice_u002e_sroa_u002e_0_u002e_0577_: u32;
    var horizon_pos_u002e_sroa_u002e_0_u002e_0576_: f32;
    var horizon_neg_u002e_sroa_u002e_0_u002e_0575_: f32;
    var step_u002e_sroa_u002e_0_u002e_0574_: u32;
    var out_u002e_8_: f32;
    var out_u002e_10_: f32;
    var out_u002e_11_: f32;

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
        let _e121 = ((_e97 + (((_e72 * _e80) + (_e76 * _e85)) + (_e23 * _e91))) / (_e119 + (((_e72 * _e102) + (_e76 * _e107)) + (_e23 * _e113))));
        let _e125 = uniform_struct_0_2_.f4_.y;
        let _e130 = uniform_struct_0_2_.f5_.y;
        let _e136 = uniform_struct_0_2_.f6_.y;
        let _e142 = uniform_struct_0_2_.f7_.y;
        let _e147 = uniform_struct_0_2_.f4_.w;
        let _e152 = uniform_struct_0_2_.f5_.w;
        let _e158 = uniform_struct_0_2_.f6_.w;
        let _e164 = uniform_struct_0_2_.f7_.w;
        let _e166 = ((_e142 + (((_e72 * _e125) + (_e76 * _e130)) + (_e23 * _e136))) / (_e164 + (((_e72 * _e147) + (_e76 * _e152)) + (_e23 * _e158))));
        let _e170 = uniform_struct_0_2_.f4_.z;
        let _e175 = uniform_struct_0_2_.f5_.z;
        let _e181 = uniform_struct_0_2_.f6_.z;
        let _e187 = uniform_struct_0_2_.f7_.z;
        let _e192 = uniform_struct_0_2_.f4_.w;
        let _e197 = uniform_struct_0_2_.f5_.w;
        let _e203 = uniform_struct_0_2_.f6_.w;
        let _e209 = uniform_struct_0_2_.f7_.w;
        let _e211 = ((_e187 + (((_e72 * _e170) + (_e76 * _e175)) + (_e23 * _e181))) / (_e209 + (((_e72 * _e192) + (_e76 * _e197)) + (_e23 * _e203))));
        let _e215 = uniform_struct_0_2_.f8_.x;
        let _e219 = uniform_struct_0_2_.f8_.y;
        let _e223 = uniform_struct_0_2_.f8_.z;
        let _e224 = (_e215 - _e121);
        let _e225 = (_e219 - _e166);
        let _e226 = (_e223 - _e211);
        let _e230 = max(length(vec3<f32>(_e224, _e225, _e226)), 0.000001f);
        let _e231 = (_e224 / _e230);
        let _e232 = (_e225 / _e230);
        let _e233 = (_e226 / _e230);
        let _e235 = (1f / _e8);
        let _e240 = (((uv.x + _e235) * 2f) + -1f);
        let _e244 = uniform_struct_0_2_.f4_.x;
        let _e249 = uniform_struct_0_2_.f5_.x;
        let _e255 = uniform_struct_0_2_.f6_.x;
        let _e261 = uniform_struct_0_2_.f7_.x;
        let _e266 = uniform_struct_0_2_.f4_.w;
        let _e271 = uniform_struct_0_2_.f5_.w;
        let _e277 = uniform_struct_0_2_.f6_.w;
        let _e283 = uniform_struct_0_2_.f7_.w;
        let _e289 = uniform_struct_0_2_.f4_.y;
        let _e294 = uniform_struct_0_2_.f5_.y;
        let _e300 = uniform_struct_0_2_.f6_.y;
        let _e306 = uniform_struct_0_2_.f7_.y;
        let _e311 = uniform_struct_0_2_.f4_.w;
        let _e316 = uniform_struct_0_2_.f5_.w;
        let _e322 = uniform_struct_0_2_.f6_.w;
        let _e328 = uniform_struct_0_2_.f7_.w;
        let _e334 = uniform_struct_0_2_.f4_.z;
        let _e339 = uniform_struct_0_2_.f5_.z;
        let _e345 = uniform_struct_0_2_.f6_.z;
        let _e351 = uniform_struct_0_2_.f7_.z;
        let _e356 = uniform_struct_0_2_.f4_.w;
        let _e361 = uniform_struct_0_2_.f5_.w;
        let _e367 = uniform_struct_0_2_.f6_.w;
        let _e373 = uniform_struct_0_2_.f7_.w;
        let _e382 = (1f - ((uv.y + (1f / _e13)) * 2f));
        let _e386 = uniform_struct_0_2_.f4_.x;
        let _e391 = uniform_struct_0_2_.f5_.x;
        let _e397 = uniform_struct_0_2_.f6_.x;
        let _e403 = uniform_struct_0_2_.f7_.x;
        let _e408 = uniform_struct_0_2_.f4_.w;
        let _e413 = uniform_struct_0_2_.f5_.w;
        let _e419 = uniform_struct_0_2_.f6_.w;
        let _e425 = uniform_struct_0_2_.f7_.w;
        let _e431 = uniform_struct_0_2_.f4_.y;
        let _e436 = uniform_struct_0_2_.f5_.y;
        let _e442 = uniform_struct_0_2_.f6_.y;
        let _e448 = uniform_struct_0_2_.f7_.y;
        let _e453 = uniform_struct_0_2_.f4_.w;
        let _e458 = uniform_struct_0_2_.f5_.w;
        let _e464 = uniform_struct_0_2_.f6_.w;
        let _e470 = uniform_struct_0_2_.f7_.w;
        let _e476 = uniform_struct_0_2_.f4_.z;
        let _e481 = uniform_struct_0_2_.f5_.z;
        let _e487 = uniform_struct_0_2_.f6_.z;
        let _e493 = uniform_struct_0_2_.f7_.z;
        let _e498 = uniform_struct_0_2_.f4_.w;
        let _e503 = uniform_struct_0_2_.f5_.w;
        let _e509 = uniform_struct_0_2_.f6_.w;
        let _e515 = uniform_struct_0_2_.f7_.w;
        let _e520 = ((((_e261 + (((_e240 * _e244) + (_e76 * _e249)) + (_e23 * _e255))) / (_e283 + (((_e240 * _e266) + (_e76 * _e271)) + (_e23 * _e277)))) - _e121) + 0.000001f);
        let _e521 = (((_e306 + (((_e240 * _e289) + (_e76 * _e294)) + (_e23 * _e300))) / (_e328 + (((_e240 * _e311) + (_e76 * _e316)) + (_e23 * _e322)))) - _e166);
        let _e522 = (((_e351 + (((_e240 * _e334) + (_e76 * _e339)) + (_e23 * _e345))) / (_e373 + (((_e240 * _e356) + (_e76 * _e361)) + (_e23 * _e367)))) - _e211);
        let _e532 = (((_e403 + (((_e72 * _e386) + (_e382 * _e391)) + (_e23 * _e397))) / (_e425 + (((_e72 * _e408) + (_e382 * _e413)) + (_e23 * _e419)))) - _e121);
        let _e535 = ((((_e448 + (((_e72 * _e431) + (_e382 * _e436)) + (_e23 * _e442))) / (_e470 + (((_e72 * _e453) + (_e382 * _e458)) + (_e23 * _e464)))) - _e166) + 0.000001f);
        let _e536 = (((_e493 + (((_e72 * _e476) + (_e382 * _e481)) + (_e23 * _e487))) / (_e515 + (((_e72 * _e498) + (_e382 * _e503)) + (_e23 * _e509)))) - _e211);
        let _e548 = uniform_struct_0_3_.f0_;
        let _e551 = uniform_struct_0_3_.f1_;
        let _e553 = (_e121 + (normalize(vec3<f32>(_e520, _e521, _e522)).x * _e548));
        let _e555 = (_e166 + (normalize(vec3<f32>(_e520, _e521, _e522)).y * _e548));
        let _e557 = (_e211 + (normalize(vec3<f32>(_e520, _e521, _e522)).z * _e548));
        let _e561 = uniform_struct_0_2_.f0_.x;
        let _e566 = uniform_struct_0_2_.f1_.x;
        let _e572 = uniform_struct_0_2_.f2_.x;
        let _e578 = uniform_struct_0_2_.f3_.x;
        let _e583 = uniform_struct_0_2_.f0_.y;
        let _e588 = uniform_struct_0_2_.f1_.y;
        let _e594 = uniform_struct_0_2_.f2_.y;
        let _e600 = uniform_struct_0_2_.f3_.y;
        let _e605 = uniform_struct_0_2_.f0_.w;
        let _e610 = uniform_struct_0_2_.f1_.w;
        let _e616 = uniform_struct_0_2_.f2_.w;
        let _e622 = uniform_struct_0_2_.f3_.w;
        let _e625 = max((_e622 + (((_e553 * _e605) + (_e555 * _e610)) + (_e557 * _e616))), 0.000001f);
        visibility_u002e_sroa_u002e_0_u002e_0578_ = 0f;
        slice_u002e_sroa_u002e_0_u002e_0577_ = 0u;
        loop {
            let _e676 = visibility_u002e_sroa_u002e_0_u002e_0578_;
            let _e678 = slice_u002e_sroa_u002e_0_u002e_0577_;
            let _e682 = ((fract((fract(dot(vec2<f32>(f32(_e15), f32(_e17)), vec2<f32>(0.06711056f, 0.00583715f))) * 52.982918f)) * 3.1415927f) + (f32(_e678) * 1.5707964f));
            let _e683 = cos(_e682);
            let _e684 = sin(_e682);
            let _e687 = ((normalize(vec3<f32>(_e520, _e521, _e522)).x * _e683) + (normalize(vec3<f32>(_e532, _e535, _e536)).x * _e684));
            let _e690 = ((normalize(vec3<f32>(_e520, _e521, _e522)).y * _e683) + (normalize(vec3<f32>(_e532, _e535, _e536)).y * _e684));
            let _e693 = ((normalize(vec3<f32>(_e520, _e521, _e522)).z * _e683) + (normalize(vec3<f32>(_e532, _e535, _e536)).z * _e684));
            let _e708 = (cross(vec3<f32>(normalize(vec3<f32>(_e687, _e690, _e693)).x, normalize(vec3<f32>(_e687, _e690, _e693)).y, normalize(vec3<f32>(_e687, _e690, _e693)).z), vec3<f32>(_e231, _e232, _e233)).x + 0.000001f);
            let _e728 = dot(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(normalize(vec3<f32>(_e708, cross(vec3<f32>(normalize(vec3<f32>(_e687, _e690, _e693)).x, normalize(vec3<f32>(_e687, _e690, _e693)).y, normalize(vec3<f32>(_e687, _e690, _e693)).z), vec3<f32>(_e231, _e232, _e233)).y, cross(vec3<f32>(normalize(vec3<f32>(_e687, _e690, _e693)).x, normalize(vec3<f32>(_e687, _e690, _e693)).y, normalize(vec3<f32>(_e687, _e690, _e693)).z), vec3<f32>(_e231, _e232, _e233)).z)).x, normalize(vec3<f32>(_e708, cross(vec3<f32>(normalize(vec3<f32>(_e687, _e690, _e693)).x, normalize(vec3<f32>(_e687, _e690, _e693)).y, normalize(vec3<f32>(_e687, _e690, _e693)).z), vec3<f32>(_e231, _e232, _e233)).y, cross(vec3<f32>(normalize(vec3<f32>(_e687, _e690, _e693)).x, normalize(vec3<f32>(_e687, _e690, _e693)).y, normalize(vec3<f32>(_e687, _e690, _e693)).z), vec3<f32>(_e231, _e232, _e233)).z)).y, normalize(vec3<f32>(_e708, cross(vec3<f32>(normalize(vec3<f32>(_e687, _e690, _e693)).x, normalize(vec3<f32>(_e687, _e690, _e693)).y, normalize(vec3<f32>(_e687, _e690, _e693)).z), vec3<f32>(_e231, _e232, _e233)).y, cross(vec3<f32>(normalize(vec3<f32>(_e687, _e690, _e693)).x, normalize(vec3<f32>(_e687, _e690, _e693)).y, normalize(vec3<f32>(_e687, _e690, _e693)).z), vec3<f32>(_e231, _e232, _e233)).z)).z));
            let _e730 = (normalize(vec3<f32>(_e54, _e59, _e45)).x - (normalize(vec3<f32>(_e708, cross(vec3<f32>(normalize(vec3<f32>(_e687, _e690, _e693)).x, normalize(vec3<f32>(_e687, _e690, _e693)).y, normalize(vec3<f32>(_e687, _e690, _e693)).z), vec3<f32>(_e231, _e232, _e233)).y, cross(vec3<f32>(normalize(vec3<f32>(_e687, _e690, _e693)).x, normalize(vec3<f32>(_e687, _e690, _e693)).y, normalize(vec3<f32>(_e687, _e690, _e693)).z), vec3<f32>(_e231, _e232, _e233)).z)).x * _e728));
            let _e732 = (normalize(vec3<f32>(_e54, _e59, _e45)).y - (normalize(vec3<f32>(_e708, cross(vec3<f32>(normalize(vec3<f32>(_e687, _e690, _e693)).x, normalize(vec3<f32>(_e687, _e690, _e693)).y, normalize(vec3<f32>(_e687, _e690, _e693)).z), vec3<f32>(_e231, _e232, _e233)).y, cross(vec3<f32>(normalize(vec3<f32>(_e687, _e690, _e693)).x, normalize(vec3<f32>(_e687, _e690, _e693)).y, normalize(vec3<f32>(_e687, _e690, _e693)).z), vec3<f32>(_e231, _e232, _e233)).z)).y * _e728));
            let _e734 = (normalize(vec3<f32>(_e54, _e59, _e45)).z - (normalize(vec3<f32>(_e708, cross(vec3<f32>(normalize(vec3<f32>(_e687, _e690, _e693)).x, normalize(vec3<f32>(_e687, _e690, _e693)).y, normalize(vec3<f32>(_e687, _e690, _e693)).z), vec3<f32>(_e231, _e232, _e233)).y, cross(vec3<f32>(normalize(vec3<f32>(_e687, _e690, _e693)).x, normalize(vec3<f32>(_e687, _e690, _e693)).y, normalize(vec3<f32>(_e687, _e690, _e693)).z), vec3<f32>(_e231, _e232, _e233)).z)).z * _e728));
            let _e738 = max(length(vec3<f32>(_e730, _e732, _e734)), 0.000001f);
            let _e751 = (sign(dot(vec3<f32>(normalize(vec3<f32>(_e687, _e690, _e693)).x, normalize(vec3<f32>(_e687, _e690, _e693)).y, normalize(vec3<f32>(_e687, _e690, _e693)).z), vec3<f32>(_e730, _e732, _e734))) * acos(clamp((dot(vec3<f32>(_e730, _e732, _e734), vec3<f32>(_e231, _e232, _e233)) / _e738), 0f, 1f)));
            let _e754 = cos((_e751 + 1.5707964f));
            let _e757 = cos((_e751 + -1.5707964f));
            horizon_pos_u002e_sroa_u002e_0_u002e_0576_ = _e754;
            horizon_neg_u002e_sroa_u002e_0_u002e_0575_ = _e757;
            step_u002e_sroa_u002e_0_u002e_0574_ = 1u;
            loop {
                let _e763 = horizon_pos_u002e_sroa_u002e_0_u002e_0576_;
                let _e765 = horizon_neg_u002e_sroa_u002e_0_u002e_0575_;
                let _e767 = step_u002e_sroa_u002e_0_u002e_0574_;
                let _e771 = ((max(length(vec3<f32>((((((_e578 + (((_e561 * _e553) + (_e555 * _e566)) + (_e557 * _e572))) / _e625) * 0.5f) + 0.5f) - uv.x), ((0.5f - (((_e600 + (((_e553 * _e583) + (_e555 * _e588)) + (_e557 * _e594))) / _e625) * 0.5f)) - uv.y), 0f)), _e235) * f32(_e767)) * 0.25f);
                let _e772 = (_e683 * _e771);
                let _e773 = (_e684 * _e771);
                let _e774 = (uv.x + _e772);
                let _e775 = (uv.y + _e773);
                if ((_e775 <= 1f) & ((_e774 <= 1f) & ((_e774 >= 0f) & (_e775 >= 0f)))) {
                    let _e798 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(min(u32((_e774 * _e8)), u32((_e8 + -1f)))), bitcast<i32>(min(u32((_e775 * _e13)), u32((_e13 + -1f))))), 0i);
                    if (_e798 < 1f) {
                        let _e804 = ((_e774 * 2f) + -1f);
                        let _e808 = (1f - (_e775 * 2f));
                        let _e812 = uniform_struct_0_2_.f4_.x;
                        let _e817 = uniform_struct_0_2_.f5_.x;
                        let _e823 = uniform_struct_0_2_.f6_.x;
                        let _e829 = uniform_struct_0_2_.f7_.x;
                        let _e834 = uniform_struct_0_2_.f4_.w;
                        let _e839 = uniform_struct_0_2_.f5_.w;
                        let _e845 = uniform_struct_0_2_.f6_.w;
                        let _e851 = uniform_struct_0_2_.f7_.w;
                        let _e857 = uniform_struct_0_2_.f4_.y;
                        let _e862 = uniform_struct_0_2_.f5_.y;
                        let _e868 = uniform_struct_0_2_.f6_.y;
                        let _e874 = uniform_struct_0_2_.f7_.y;
                        let _e879 = uniform_struct_0_2_.f4_.w;
                        let _e884 = uniform_struct_0_2_.f5_.w;
                        let _e890 = uniform_struct_0_2_.f6_.w;
                        let _e896 = uniform_struct_0_2_.f7_.w;
                        let _e902 = uniform_struct_0_2_.f4_.z;
                        let _e907 = uniform_struct_0_2_.f5_.z;
                        let _e913 = uniform_struct_0_2_.f6_.z;
                        let _e919 = uniform_struct_0_2_.f7_.z;
                        let _e924 = uniform_struct_0_2_.f4_.w;
                        let _e929 = uniform_struct_0_2_.f5_.w;
                        let _e935 = uniform_struct_0_2_.f6_.w;
                        let _e941 = uniform_struct_0_2_.f7_.w;
                        let _e944 = (((_e829 + (((_e804 * _e812) + (_e808 * _e817)) + (_e798 * _e823))) / (_e851 + (((_e804 * _e834) + (_e808 * _e839)) + (_e798 * _e845)))) - _e121);
                        let _e945 = (((_e874 + (((_e804 * _e857) + (_e808 * _e862)) + (_e798 * _e868))) / (_e896 + (((_e804 * _e879) + (_e808 * _e884)) + (_e798 * _e890)))) - _e166);
                        let _e946 = (((_e919 + (((_e804 * _e902) + (_e808 * _e907)) + (_e798 * _e913))) / (_e941 + (((_e804 * _e924) + (_e808 * _e929)) + (_e798 * _e935)))) - _e211);
                        let _e948 = length(vec3<f32>(_e944, _e945, _e946));
                        if ((_e948 > 0.000001f) & (_e948 < _e548)) {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i419_ = mix(_e754, dot(vec3<f32>((_e944 / _e948), (_e945 / _e948), (_e946 / _e948)), vec3<f32>(_e231, _e232, _e233)), clamp((1f - (_e948 / max(_e548, 0.000001f))), 0f, 1f));
                        } else {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i419_ = -1f;
                        }
                    } else {
                        cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i419_ = -1f;
                    }
                } else {
                    cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i419_ = -1f;
                }
                let _e974 = cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i419_;
                let _e975 = max(_e763, _e974);
                out_u002e_10_ = _e975;
                let _e977 = (uv.x - _e772);
                let _e978 = (uv.y - _e773);
                if ((_e978 <= 1f) & ((_e977 <= 1f) & ((_e977 >= 0f) & (_e978 >= 0f)))) {
                    let _e1001 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(min(u32((_e977 * _e8)), u32((_e8 + -1f)))), bitcast<i32>(min(u32((_e978 * _e13)), u32((_e13 + -1f))))), 0i);
                    if (_e1001 < 1f) {
                        let _e1007 = ((_e977 * 2f) + -1f);
                        let _e1011 = (1f - (_e978 * 2f));
                        let _e1015 = uniform_struct_0_2_.f4_.x;
                        let _e1020 = uniform_struct_0_2_.f5_.x;
                        let _e1026 = uniform_struct_0_2_.f6_.x;
                        let _e1032 = uniform_struct_0_2_.f7_.x;
                        let _e1037 = uniform_struct_0_2_.f4_.w;
                        let _e1042 = uniform_struct_0_2_.f5_.w;
                        let _e1048 = uniform_struct_0_2_.f6_.w;
                        let _e1054 = uniform_struct_0_2_.f7_.w;
                        let _e1060 = uniform_struct_0_2_.f4_.y;
                        let _e1065 = uniform_struct_0_2_.f5_.y;
                        let _e1071 = uniform_struct_0_2_.f6_.y;
                        let _e1077 = uniform_struct_0_2_.f7_.y;
                        let _e1082 = uniform_struct_0_2_.f4_.w;
                        let _e1087 = uniform_struct_0_2_.f5_.w;
                        let _e1093 = uniform_struct_0_2_.f6_.w;
                        let _e1099 = uniform_struct_0_2_.f7_.w;
                        let _e1105 = uniform_struct_0_2_.f4_.z;
                        let _e1110 = uniform_struct_0_2_.f5_.z;
                        let _e1116 = uniform_struct_0_2_.f6_.z;
                        let _e1122 = uniform_struct_0_2_.f7_.z;
                        let _e1127 = uniform_struct_0_2_.f4_.w;
                        let _e1132 = uniform_struct_0_2_.f5_.w;
                        let _e1138 = uniform_struct_0_2_.f6_.w;
                        let _e1144 = uniform_struct_0_2_.f7_.w;
                        let _e1147 = (((_e1032 + (((_e1007 * _e1015) + (_e1011 * _e1020)) + (_e1001 * _e1026))) / (_e1054 + (((_e1007 * _e1037) + (_e1011 * _e1042)) + (_e1001 * _e1048)))) - _e121);
                        let _e1148 = (((_e1077 + (((_e1007 * _e1060) + (_e1011 * _e1065)) + (_e1001 * _e1071))) / (_e1099 + (((_e1007 * _e1082) + (_e1011 * _e1087)) + (_e1001 * _e1093)))) - _e166);
                        let _e1149 = (((_e1122 + (((_e1007 * _e1105) + (_e1011 * _e1110)) + (_e1001 * _e1116))) / (_e1144 + (((_e1007 * _e1127) + (_e1011 * _e1132)) + (_e1001 * _e1138)))) - _e211);
                        let _e1151 = length(vec3<f32>(_e1147, _e1148, _e1149));
                        if ((_e1151 > 0.000001f) & (_e1151 < _e548)) {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = mix(_e757, dot(vec3<f32>((_e1147 / _e1151), (_e1148 / _e1151), (_e1149 / _e1151)), vec3<f32>(_e231, _e232, _e233)), clamp((1f - (_e1151 / max(_e548, 0.000001f))), 0f, 1f));
                        } else {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = -1f;
                        }
                    } else {
                        cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = -1f;
                    }
                } else {
                    cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = -1f;
                }
                let _e1177 = cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i;
                let _e1178 = max(_e765, _e1177);
                out_u002e_11_ = _e1178;
                let _e1181 = (_e767 + 1u);
                if (_e1181 == 5u) {
                    break;
                } else {
                    horizon_pos_u002e_sroa_u002e_0_u002e_0576_ = _e975;
                    horizon_neg_u002e_sroa_u002e_0_u002e_0575_ = _e1178;
                    step_u002e_sroa_u002e_0_u002e_0574_ = _e1181;
                    continue;
                }
            }
            let _e1190 = out_u002e_10_;
            let _e1192 = out_u002e_11_;
            let _e1202 = (acos(clamp(_e1190, -1f, 1f)) * 2f);
            let _e1213 = (acos(clamp(_e1192, -1f, 1f)) * -2f);
            let _e1225 = (_e676 + (_e738 * ((((cos(_e751) - cos((_e1202 - _e751))) + (_e1202 * sin(_e751))) * 0.25f) + (((cos(_e751) - cos((_e1213 - _e751))) + (_e1213 * sin(_e751))) * 0.25f))));
            out_u002e_8_ = _e1225;
            if (_e678 == 0u) {
                visibility_u002e_sroa_u002e_0_u002e_0578_ = _e1225;
                slice_u002e_sroa_u002e_0_u002e_0577_ = (_e678 + 1u);
                continue;
            } else {
                break;
            }
        }
        let _e1234 = out_u002e_8_;
        let _e1241 = clamp(mix(1f, (_e1234 * 0.5f), _e551), 0f, 1f);
        fragment_out = vec4<f32>(_e1241, _e1241, _e1241, 1f);
    } else {
        fragment_out = vec4<f32>(1f, 1f, 1f, 1f);
    }
    let _e1252 = fragment_out;
    return _e1252;
}
