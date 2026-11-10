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
    var cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_: f32;
    var visibility_u002e_sroa_u002e_0_u002e_0577_: f32;
    var slice_u002e_sroa_u002e_0_u002e_0576_: u32;
    var horizon_pos_u002e_sroa_u002e_0_u002e_0575_: f32;
    var horizon_neg_u002e_sroa_u002e_0_u002e_0574_: f32;
    var step_u002e_sroa_u002e_0_u002e_0573_: u32;
    var out_u002e_17_: f32;
    var out_u002e_22_: f32;
    var out_u002e_26_: f32;

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
        let _e123 = ((_e97 + (((_e72 * _e80) + (_e76 * _e85)) + (_e23 * _e91))) / max((_e119 + (((_e72 * _e102) + (_e76 * _e107)) + (_e23 * _e113))), 0.000001f));
        let _e127 = uniform_struct_0_2_.f4_.y;
        let _e132 = uniform_struct_0_2_.f5_.y;
        let _e138 = uniform_struct_0_2_.f6_.y;
        let _e144 = uniform_struct_0_2_.f7_.y;
        let _e149 = uniform_struct_0_2_.f4_.w;
        let _e154 = uniform_struct_0_2_.f5_.w;
        let _e160 = uniform_struct_0_2_.f6_.w;
        let _e166 = uniform_struct_0_2_.f7_.w;
        let _e170 = ((_e144 + (((_e72 * _e127) + (_e76 * _e132)) + (_e23 * _e138))) / max((_e166 + (((_e72 * _e149) + (_e76 * _e154)) + (_e23 * _e160))), 0.000001f));
        let _e174 = uniform_struct_0_2_.f4_.z;
        let _e179 = uniform_struct_0_2_.f5_.z;
        let _e185 = uniform_struct_0_2_.f6_.z;
        let _e191 = uniform_struct_0_2_.f7_.z;
        let _e196 = uniform_struct_0_2_.f4_.w;
        let _e201 = uniform_struct_0_2_.f5_.w;
        let _e207 = uniform_struct_0_2_.f6_.w;
        let _e213 = uniform_struct_0_2_.f7_.w;
        let _e217 = ((_e191 + (((_e72 * _e174) + (_e76 * _e179)) + (_e23 * _e185))) / max((_e213 + (((_e72 * _e196) + (_e76 * _e201)) + (_e23 * _e207))), 0.000001f));
        let _e221 = uniform_struct_0_2_.f8_.x;
        let _e225 = uniform_struct_0_2_.f8_.y;
        let _e229 = uniform_struct_0_2_.f8_.z;
        let _e230 = (_e221 - _e123);
        let _e231 = (_e225 - _e170);
        let _e232 = (_e229 - _e217);
        let _e236 = max(length(vec3<f32>(_e230, _e231, _e232)), 0.000001f);
        let _e237 = (_e230 / _e236);
        let _e238 = (_e231 / _e236);
        let _e239 = (_e232 / _e236);
        let _e241 = (1f / _e8);
        let _e246 = (((uv.x + _e241) * 2f) + -1f);
        let _e250 = uniform_struct_0_2_.f4_.x;
        let _e255 = uniform_struct_0_2_.f5_.x;
        let _e261 = uniform_struct_0_2_.f6_.x;
        let _e267 = uniform_struct_0_2_.f7_.x;
        let _e272 = uniform_struct_0_2_.f4_.w;
        let _e277 = uniform_struct_0_2_.f5_.w;
        let _e283 = uniform_struct_0_2_.f6_.w;
        let _e289 = uniform_struct_0_2_.f7_.w;
        let _e297 = uniform_struct_0_2_.f4_.y;
        let _e302 = uniform_struct_0_2_.f5_.y;
        let _e308 = uniform_struct_0_2_.f6_.y;
        let _e314 = uniform_struct_0_2_.f7_.y;
        let _e319 = uniform_struct_0_2_.f4_.w;
        let _e324 = uniform_struct_0_2_.f5_.w;
        let _e330 = uniform_struct_0_2_.f6_.w;
        let _e336 = uniform_struct_0_2_.f7_.w;
        let _e344 = uniform_struct_0_2_.f4_.z;
        let _e349 = uniform_struct_0_2_.f5_.z;
        let _e355 = uniform_struct_0_2_.f6_.z;
        let _e361 = uniform_struct_0_2_.f7_.z;
        let _e366 = uniform_struct_0_2_.f4_.w;
        let _e371 = uniform_struct_0_2_.f5_.w;
        let _e377 = uniform_struct_0_2_.f6_.w;
        let _e383 = uniform_struct_0_2_.f7_.w;
        let _e394 = (1f - ((uv.y + (1f / _e13)) * 2f));
        let _e398 = uniform_struct_0_2_.f4_.x;
        let _e403 = uniform_struct_0_2_.f5_.x;
        let _e409 = uniform_struct_0_2_.f6_.x;
        let _e415 = uniform_struct_0_2_.f7_.x;
        let _e420 = uniform_struct_0_2_.f4_.w;
        let _e425 = uniform_struct_0_2_.f5_.w;
        let _e431 = uniform_struct_0_2_.f6_.w;
        let _e437 = uniform_struct_0_2_.f7_.w;
        let _e445 = uniform_struct_0_2_.f4_.y;
        let _e450 = uniform_struct_0_2_.f5_.y;
        let _e456 = uniform_struct_0_2_.f6_.y;
        let _e462 = uniform_struct_0_2_.f7_.y;
        let _e467 = uniform_struct_0_2_.f4_.w;
        let _e472 = uniform_struct_0_2_.f5_.w;
        let _e478 = uniform_struct_0_2_.f6_.w;
        let _e484 = uniform_struct_0_2_.f7_.w;
        let _e492 = uniform_struct_0_2_.f4_.z;
        let _e497 = uniform_struct_0_2_.f5_.z;
        let _e503 = uniform_struct_0_2_.f6_.z;
        let _e509 = uniform_struct_0_2_.f7_.z;
        let _e514 = uniform_struct_0_2_.f4_.w;
        let _e519 = uniform_struct_0_2_.f5_.w;
        let _e525 = uniform_struct_0_2_.f6_.w;
        let _e531 = uniform_struct_0_2_.f7_.w;
        let _e538 = ((((_e267 + (((_e246 * _e250) + (_e76 * _e255)) + (_e23 * _e261))) / max((_e289 + (((_e246 * _e272) + (_e76 * _e277)) + (_e23 * _e283))), 0.000001f)) - _e123) + 0.000001f);
        let _e539 = (((_e314 + (((_e246 * _e297) + (_e76 * _e302)) + (_e23 * _e308))) / max((_e336 + (((_e246 * _e319) + (_e76 * _e324)) + (_e23 * _e330))), 0.000001f)) - _e170);
        let _e540 = (((_e361 + (((_e246 * _e344) + (_e76 * _e349)) + (_e23 * _e355))) / max((_e383 + (((_e246 * _e366) + (_e76 * _e371)) + (_e23 * _e377))), 0.000001f)) - _e217);
        let _e550 = (((_e415 + (((_e72 * _e398) + (_e394 * _e403)) + (_e23 * _e409))) / max((_e437 + (((_e72 * _e420) + (_e394 * _e425)) + (_e23 * _e431))), 0.000001f)) - _e123);
        let _e553 = ((((_e462 + (((_e72 * _e445) + (_e394 * _e450)) + (_e23 * _e456))) / max((_e484 + (((_e72 * _e467) + (_e394 * _e472)) + (_e23 * _e478))), 0.000001f)) - _e170) + 0.000001f);
        let _e554 = (((_e509 + (((_e72 * _e492) + (_e394 * _e497)) + (_e23 * _e503))) / max((_e531 + (((_e72 * _e514) + (_e394 * _e519)) + (_e23 * _e525))), 0.000001f)) - _e217);
        let _e566 = uniform_struct_0_3_.f0_;
        let _e569 = uniform_struct_0_3_.f1_;
        let _e571 = (_e123 + (normalize(vec3<f32>(_e538, _e539, _e540)).x * _e566));
        let _e573 = (_e170 + (normalize(vec3<f32>(_e538, _e539, _e540)).y * _e566));
        let _e575 = (_e217 + (normalize(vec3<f32>(_e538, _e539, _e540)).z * _e566));
        let _e579 = uniform_struct_0_2_.f0_.x;
        let _e584 = uniform_struct_0_2_.f1_.x;
        let _e590 = uniform_struct_0_2_.f2_.x;
        let _e596 = uniform_struct_0_2_.f3_.x;
        let _e601 = uniform_struct_0_2_.f0_.y;
        let _e606 = uniform_struct_0_2_.f1_.y;
        let _e612 = uniform_struct_0_2_.f2_.y;
        let _e618 = uniform_struct_0_2_.f3_.y;
        let _e623 = uniform_struct_0_2_.f0_.w;
        let _e628 = uniform_struct_0_2_.f1_.w;
        let _e634 = uniform_struct_0_2_.f2_.w;
        let _e640 = uniform_struct_0_2_.f3_.w;
        let _e643 = max((_e640 + (((_e571 * _e623) + (_e573 * _e628)) + (_e575 * _e634))), 0.000001f);
        visibility_u002e_sroa_u002e_0_u002e_0577_ = 0f;
        slice_u002e_sroa_u002e_0_u002e_0576_ = 0u;
        loop {
            let _e694 = visibility_u002e_sroa_u002e_0_u002e_0577_;
            let _e696 = slice_u002e_sroa_u002e_0_u002e_0576_;
            let _e700 = ((fract((fract(((f32(_e15) * 0.06711056f) + (f32(_e17) * 0.00583715f))) * 52.982918f)) * 3.1415927f) + (f32(_e696) * 1.5707964f));
            let _e701 = cos(_e700);
            let _e702 = sin(_e700);
            let _e705 = ((normalize(vec3<f32>(_e538, _e539, _e540)).x * _e701) + (normalize(vec3<f32>(_e550, _e553, _e554)).x * _e702));
            let _e708 = ((normalize(vec3<f32>(_e538, _e539, _e540)).y * _e701) + (normalize(vec3<f32>(_e550, _e553, _e554)).y * _e702));
            let _e711 = ((normalize(vec3<f32>(_e538, _e539, _e540)).z * _e701) + (normalize(vec3<f32>(_e550, _e553, _e554)).z * _e702));
            let _e726 = (cross(vec3<f32>(normalize(vec3<f32>(_e705, _e708, _e711)).x, normalize(vec3<f32>(_e705, _e708, _e711)).y, normalize(vec3<f32>(_e705, _e708, _e711)).z), vec3<f32>(_e237, _e238, _e239)).x + 0.000001f);
            let _e746 = dot(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(normalize(vec3<f32>(_e726, cross(vec3<f32>(normalize(vec3<f32>(_e705, _e708, _e711)).x, normalize(vec3<f32>(_e705, _e708, _e711)).y, normalize(vec3<f32>(_e705, _e708, _e711)).z), vec3<f32>(_e237, _e238, _e239)).y, cross(vec3<f32>(normalize(vec3<f32>(_e705, _e708, _e711)).x, normalize(vec3<f32>(_e705, _e708, _e711)).y, normalize(vec3<f32>(_e705, _e708, _e711)).z), vec3<f32>(_e237, _e238, _e239)).z)).x, normalize(vec3<f32>(_e726, cross(vec3<f32>(normalize(vec3<f32>(_e705, _e708, _e711)).x, normalize(vec3<f32>(_e705, _e708, _e711)).y, normalize(vec3<f32>(_e705, _e708, _e711)).z), vec3<f32>(_e237, _e238, _e239)).y, cross(vec3<f32>(normalize(vec3<f32>(_e705, _e708, _e711)).x, normalize(vec3<f32>(_e705, _e708, _e711)).y, normalize(vec3<f32>(_e705, _e708, _e711)).z), vec3<f32>(_e237, _e238, _e239)).z)).y, normalize(vec3<f32>(_e726, cross(vec3<f32>(normalize(vec3<f32>(_e705, _e708, _e711)).x, normalize(vec3<f32>(_e705, _e708, _e711)).y, normalize(vec3<f32>(_e705, _e708, _e711)).z), vec3<f32>(_e237, _e238, _e239)).y, cross(vec3<f32>(normalize(vec3<f32>(_e705, _e708, _e711)).x, normalize(vec3<f32>(_e705, _e708, _e711)).y, normalize(vec3<f32>(_e705, _e708, _e711)).z), vec3<f32>(_e237, _e238, _e239)).z)).z));
            let _e748 = (normalize(vec3<f32>(_e54, _e59, _e45)).x - (normalize(vec3<f32>(_e726, cross(vec3<f32>(normalize(vec3<f32>(_e705, _e708, _e711)).x, normalize(vec3<f32>(_e705, _e708, _e711)).y, normalize(vec3<f32>(_e705, _e708, _e711)).z), vec3<f32>(_e237, _e238, _e239)).y, cross(vec3<f32>(normalize(vec3<f32>(_e705, _e708, _e711)).x, normalize(vec3<f32>(_e705, _e708, _e711)).y, normalize(vec3<f32>(_e705, _e708, _e711)).z), vec3<f32>(_e237, _e238, _e239)).z)).x * _e746));
            let _e750 = (normalize(vec3<f32>(_e54, _e59, _e45)).y - (normalize(vec3<f32>(_e726, cross(vec3<f32>(normalize(vec3<f32>(_e705, _e708, _e711)).x, normalize(vec3<f32>(_e705, _e708, _e711)).y, normalize(vec3<f32>(_e705, _e708, _e711)).z), vec3<f32>(_e237, _e238, _e239)).y, cross(vec3<f32>(normalize(vec3<f32>(_e705, _e708, _e711)).x, normalize(vec3<f32>(_e705, _e708, _e711)).y, normalize(vec3<f32>(_e705, _e708, _e711)).z), vec3<f32>(_e237, _e238, _e239)).z)).y * _e746));
            let _e752 = (normalize(vec3<f32>(_e54, _e59, _e45)).z - (normalize(vec3<f32>(_e726, cross(vec3<f32>(normalize(vec3<f32>(_e705, _e708, _e711)).x, normalize(vec3<f32>(_e705, _e708, _e711)).y, normalize(vec3<f32>(_e705, _e708, _e711)).z), vec3<f32>(_e237, _e238, _e239)).y, cross(vec3<f32>(normalize(vec3<f32>(_e705, _e708, _e711)).x, normalize(vec3<f32>(_e705, _e708, _e711)).y, normalize(vec3<f32>(_e705, _e708, _e711)).z), vec3<f32>(_e237, _e238, _e239)).z)).z * _e746));
            let _e756 = max(length(vec3<f32>(_e748, _e750, _e752)), 0.000001f);
            let _e769 = (sign(dot(vec3<f32>(normalize(vec3<f32>(_e705, _e708, _e711)).x, normalize(vec3<f32>(_e705, _e708, _e711)).y, normalize(vec3<f32>(_e705, _e708, _e711)).z), vec3<f32>(_e748, _e750, _e752))) * acos(clamp((dot(vec3<f32>(_e748, _e750, _e752), vec3<f32>(_e237, _e238, _e239)) / _e756), 0f, 1f)));
            let _e772 = cos((_e769 + 1.5707964f));
            let _e775 = cos((_e769 + -1.5707964f));
            horizon_pos_u002e_sroa_u002e_0_u002e_0575_ = _e772;
            horizon_neg_u002e_sroa_u002e_0_u002e_0574_ = _e775;
            step_u002e_sroa_u002e_0_u002e_0573_ = 1u;
            loop {
                let _e781 = horizon_pos_u002e_sroa_u002e_0_u002e_0575_;
                let _e783 = horizon_neg_u002e_sroa_u002e_0_u002e_0574_;
                let _e785 = step_u002e_sroa_u002e_0_u002e_0573_;
                let _e789 = ((max(length(vec3<f32>((((((_e596 + (((_e579 * _e571) + (_e573 * _e584)) + (_e575 * _e590))) / _e643) * 0.5f) + 0.5f) - uv.x), ((0.5f - (((_e618 + (((_e571 * _e601) + (_e573 * _e606)) + (_e575 * _e612))) / _e643) * 0.5f)) - uv.y), 0f)), _e241) * f32(_e785)) * 0.25f);
                let _e790 = (_e701 * _e789);
                let _e791 = (_e702 * _e789);
                let _e792 = (uv.x + _e790);
                let _e793 = (uv.y + _e791);
                if ((_e793 <= 1f) & ((_e792 <= 1f) & ((_e792 >= 0f) & (_e793 >= 0f)))) {
                    let _e816 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(min(u32((_e792 * _e8)), u32((_e8 + -1f)))), bitcast<i32>(min(u32((_e793 * _e13)), u32((_e13 + -1f))))), 0i);
                    if (_e816 < 1f) {
                        let _e822 = ((_e792 * 2f) + -1f);
                        let _e826 = (1f - (_e793 * 2f));
                        let _e830 = uniform_struct_0_2_.f4_.x;
                        let _e835 = uniform_struct_0_2_.f5_.x;
                        let _e841 = uniform_struct_0_2_.f6_.x;
                        let _e847 = uniform_struct_0_2_.f7_.x;
                        let _e852 = uniform_struct_0_2_.f4_.w;
                        let _e857 = uniform_struct_0_2_.f5_.w;
                        let _e863 = uniform_struct_0_2_.f6_.w;
                        let _e869 = uniform_struct_0_2_.f7_.w;
                        let _e877 = uniform_struct_0_2_.f4_.y;
                        let _e882 = uniform_struct_0_2_.f5_.y;
                        let _e888 = uniform_struct_0_2_.f6_.y;
                        let _e894 = uniform_struct_0_2_.f7_.y;
                        let _e899 = uniform_struct_0_2_.f4_.w;
                        let _e904 = uniform_struct_0_2_.f5_.w;
                        let _e910 = uniform_struct_0_2_.f6_.w;
                        let _e916 = uniform_struct_0_2_.f7_.w;
                        let _e924 = uniform_struct_0_2_.f4_.z;
                        let _e929 = uniform_struct_0_2_.f5_.z;
                        let _e935 = uniform_struct_0_2_.f6_.z;
                        let _e941 = uniform_struct_0_2_.f7_.z;
                        let _e946 = uniform_struct_0_2_.f4_.w;
                        let _e951 = uniform_struct_0_2_.f5_.w;
                        let _e957 = uniform_struct_0_2_.f6_.w;
                        let _e963 = uniform_struct_0_2_.f7_.w;
                        let _e968 = (((_e847 + (((_e822 * _e830) + (_e826 * _e835)) + (_e816 * _e841))) / max((_e869 + (((_e822 * _e852) + (_e826 * _e857)) + (_e816 * _e863))), 0.000001f)) - _e123);
                        let _e969 = (((_e894 + (((_e822 * _e877) + (_e826 * _e882)) + (_e816 * _e888))) / max((_e916 + (((_e822 * _e899) + (_e826 * _e904)) + (_e816 * _e910))), 0.000001f)) - _e170);
                        let _e970 = (((_e941 + (((_e822 * _e924) + (_e826 * _e929)) + (_e816 * _e935))) / max((_e963 + (((_e822 * _e946) + (_e826 * _e951)) + (_e816 * _e957))), 0.000001f)) - _e217);
                        let _e972 = length(vec3<f32>(_e968, _e969, _e970));
                        if ((_e972 > 0.000001f) & (_e972 < _e566)) {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_ = mix(_e772, dot(vec3<f32>((_e968 / _e972), (_e969 / _e972), (_e970 / _e972)), vec3<f32>(_e237, _e238, _e239)), clamp((1f - (_e972 / max(_e566, 0.000001f))), 0f, 1f));
                        } else {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_ = -1f;
                        }
                    } else {
                        cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_ = -1f;
                    }
                } else {
                    cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_ = -1f;
                }
                let _e998 = cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_;
                let _e999 = max(_e781, _e998);
                out_u002e_22_ = _e999;
                let _e1001 = (uv.x - _e790);
                let _e1002 = (uv.y - _e791);
                if ((_e1002 <= 1f) & ((_e1001 <= 1f) & ((_e1001 >= 0f) & (_e1002 >= 0f)))) {
                    let _e1025 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(min(u32((_e1001 * _e8)), u32((_e8 + -1f)))), bitcast<i32>(min(u32((_e1002 * _e13)), u32((_e13 + -1f))))), 0i);
                    if (_e1025 < 1f) {
                        let _e1031 = ((_e1001 * 2f) + -1f);
                        let _e1035 = (1f - (_e1002 * 2f));
                        let _e1039 = uniform_struct_0_2_.f4_.x;
                        let _e1044 = uniform_struct_0_2_.f5_.x;
                        let _e1050 = uniform_struct_0_2_.f6_.x;
                        let _e1056 = uniform_struct_0_2_.f7_.x;
                        let _e1061 = uniform_struct_0_2_.f4_.w;
                        let _e1066 = uniform_struct_0_2_.f5_.w;
                        let _e1072 = uniform_struct_0_2_.f6_.w;
                        let _e1078 = uniform_struct_0_2_.f7_.w;
                        let _e1086 = uniform_struct_0_2_.f4_.y;
                        let _e1091 = uniform_struct_0_2_.f5_.y;
                        let _e1097 = uniform_struct_0_2_.f6_.y;
                        let _e1103 = uniform_struct_0_2_.f7_.y;
                        let _e1108 = uniform_struct_0_2_.f4_.w;
                        let _e1113 = uniform_struct_0_2_.f5_.w;
                        let _e1119 = uniform_struct_0_2_.f6_.w;
                        let _e1125 = uniform_struct_0_2_.f7_.w;
                        let _e1133 = uniform_struct_0_2_.f4_.z;
                        let _e1138 = uniform_struct_0_2_.f5_.z;
                        let _e1144 = uniform_struct_0_2_.f6_.z;
                        let _e1150 = uniform_struct_0_2_.f7_.z;
                        let _e1155 = uniform_struct_0_2_.f4_.w;
                        let _e1160 = uniform_struct_0_2_.f5_.w;
                        let _e1166 = uniform_struct_0_2_.f6_.w;
                        let _e1172 = uniform_struct_0_2_.f7_.w;
                        let _e1177 = (((_e1056 + (((_e1031 * _e1039) + (_e1035 * _e1044)) + (_e1025 * _e1050))) / max((_e1078 + (((_e1031 * _e1061) + (_e1035 * _e1066)) + (_e1025 * _e1072))), 0.000001f)) - _e123);
                        let _e1178 = (((_e1103 + (((_e1031 * _e1086) + (_e1035 * _e1091)) + (_e1025 * _e1097))) / max((_e1125 + (((_e1031 * _e1108) + (_e1035 * _e1113)) + (_e1025 * _e1119))), 0.000001f)) - _e170);
                        let _e1179 = (((_e1150 + (((_e1031 * _e1133) + (_e1035 * _e1138)) + (_e1025 * _e1144))) / max((_e1172 + (((_e1031 * _e1155) + (_e1035 * _e1160)) + (_e1025 * _e1166))), 0.000001f)) - _e217);
                        let _e1181 = length(vec3<f32>(_e1177, _e1178, _e1179));
                        if ((_e1181 > 0.000001f) & (_e1181 < _e566)) {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = mix(_e775, dot(vec3<f32>((_e1177 / _e1181), (_e1178 / _e1181), (_e1179 / _e1181)), vec3<f32>(_e237, _e238, _e239)), clamp((1f - (_e1181 / max(_e566, 0.000001f))), 0f, 1f));
                        } else {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = -1f;
                        }
                    } else {
                        cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = -1f;
                    }
                } else {
                    cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = -1f;
                }
                let _e1207 = cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i;
                let _e1208 = max(_e783, _e1207);
                out_u002e_26_ = _e1208;
                let _e1211 = (_e785 + 1u);
                if (_e1211 == 5u) {
                    break;
                } else {
                    horizon_pos_u002e_sroa_u002e_0_u002e_0575_ = _e999;
                    horizon_neg_u002e_sroa_u002e_0_u002e_0574_ = _e1208;
                    step_u002e_sroa_u002e_0_u002e_0573_ = _e1211;
                    continue;
                }
            }
            let _e1220 = out_u002e_22_;
            let _e1222 = out_u002e_26_;
            let _e1232 = (acos(clamp(_e1220, -1f, 1f)) * 2f);
            let _e1243 = (acos(clamp(_e1222, -1f, 1f)) * -2f);
            let _e1255 = (_e694 + (_e756 * ((((cos(_e769) - cos((_e1232 - _e769))) + (_e1232 * sin(_e769))) * 0.25f) + (((cos(_e769) - cos((_e1243 - _e769))) + (_e1243 * sin(_e769))) * 0.25f))));
            out_u002e_17_ = _e1255;
            if (_e696 == 0u) {
                visibility_u002e_sroa_u002e_0_u002e_0577_ = _e1255;
                slice_u002e_sroa_u002e_0_u002e_0576_ = (_e696 + 1u);
                continue;
            } else {
                break;
            }
        }
        let _e1264 = out_u002e_17_;
        let _e1271 = clamp(mix(1f, (_e1264 * 0.5f), _e569), 0f, 1f);
        fragment_out = vec4<f32>(_e1271, _e1271, _e1271, 1f);
    } else {
        fragment_out = vec4<f32>(1f, 1f, 1f, 1f);
    }
    let _e1282 = fragment_out;
    return _e1282;
}
