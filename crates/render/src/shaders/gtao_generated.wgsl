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
    var visibility_u002e_sroa_u002e_0_u002e_0565_: f32;
    var slice_u002e_sroa_u002e_0_u002e_0564_: u32;
    var horizon_pos_u002e_sroa_u002e_0_u002e_0563_: f32;
    var horizon_neg_u002e_sroa_u002e_0_u002e_0562_: f32;
    var step_u002e_sroa_u002e_0_u002e_0561_: u32;
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
        visibility_u002e_sroa_u002e_0_u002e_0565_ = 0f;
        slice_u002e_sroa_u002e_0_u002e_0564_ = 0u;
        loop {
            let _e682 = visibility_u002e_sroa_u002e_0_u002e_0565_;
            let _e684 = slice_u002e_sroa_u002e_0_u002e_0564_;
            let _e688 = ((fract((fract(((f32(_e15) * 0.06711056f) + (f32(_e17) * 0.00583715f))) * 52.982918f)) * 3.1415927f) + (f32(_e684) * 1.5707964f));
            let _e689 = cos(_e688);
            let _e690 = sin(_e688);
            let _e693 = ((normalize(vec3<f32>(_e538, _e539, _e540)).x * _e689) + (normalize(vec3<f32>(_e550, _e553, _e554)).x * _e690));
            let _e696 = ((normalize(vec3<f32>(_e538, _e539, _e540)).y * _e689) + (normalize(vec3<f32>(_e550, _e553, _e554)).y * _e690));
            let _e699 = ((normalize(vec3<f32>(_e538, _e539, _e540)).z * _e689) + (normalize(vec3<f32>(_e550, _e553, _e554)).z * _e690));
            let _e714 = (cross(vec3<f32>(normalize(vec3<f32>(_e693, _e696, _e699)).x, normalize(vec3<f32>(_e693, _e696, _e699)).y, normalize(vec3<f32>(_e693, _e696, _e699)).z), vec3<f32>(_e237, _e238, _e239)).x + 0.000001f);
            let _e734 = dot(vec3<f32>(normalize(vec3<f32>(_e54, _e59, _e45)).x, normalize(vec3<f32>(_e54, _e59, _e45)).y, normalize(vec3<f32>(_e54, _e59, _e45)).z), vec3<f32>(normalize(vec3<f32>(_e714, cross(vec3<f32>(normalize(vec3<f32>(_e693, _e696, _e699)).x, normalize(vec3<f32>(_e693, _e696, _e699)).y, normalize(vec3<f32>(_e693, _e696, _e699)).z), vec3<f32>(_e237, _e238, _e239)).y, cross(vec3<f32>(normalize(vec3<f32>(_e693, _e696, _e699)).x, normalize(vec3<f32>(_e693, _e696, _e699)).y, normalize(vec3<f32>(_e693, _e696, _e699)).z), vec3<f32>(_e237, _e238, _e239)).z)).x, normalize(vec3<f32>(_e714, cross(vec3<f32>(normalize(vec3<f32>(_e693, _e696, _e699)).x, normalize(vec3<f32>(_e693, _e696, _e699)).y, normalize(vec3<f32>(_e693, _e696, _e699)).z), vec3<f32>(_e237, _e238, _e239)).y, cross(vec3<f32>(normalize(vec3<f32>(_e693, _e696, _e699)).x, normalize(vec3<f32>(_e693, _e696, _e699)).y, normalize(vec3<f32>(_e693, _e696, _e699)).z), vec3<f32>(_e237, _e238, _e239)).z)).y, normalize(vec3<f32>(_e714, cross(vec3<f32>(normalize(vec3<f32>(_e693, _e696, _e699)).x, normalize(vec3<f32>(_e693, _e696, _e699)).y, normalize(vec3<f32>(_e693, _e696, _e699)).z), vec3<f32>(_e237, _e238, _e239)).y, cross(vec3<f32>(normalize(vec3<f32>(_e693, _e696, _e699)).x, normalize(vec3<f32>(_e693, _e696, _e699)).y, normalize(vec3<f32>(_e693, _e696, _e699)).z), vec3<f32>(_e237, _e238, _e239)).z)).z));
            let _e736 = (normalize(vec3<f32>(_e54, _e59, _e45)).x - (normalize(vec3<f32>(_e714, cross(vec3<f32>(normalize(vec3<f32>(_e693, _e696, _e699)).x, normalize(vec3<f32>(_e693, _e696, _e699)).y, normalize(vec3<f32>(_e693, _e696, _e699)).z), vec3<f32>(_e237, _e238, _e239)).y, cross(vec3<f32>(normalize(vec3<f32>(_e693, _e696, _e699)).x, normalize(vec3<f32>(_e693, _e696, _e699)).y, normalize(vec3<f32>(_e693, _e696, _e699)).z), vec3<f32>(_e237, _e238, _e239)).z)).x * _e734));
            let _e738 = (normalize(vec3<f32>(_e54, _e59, _e45)).y - (normalize(vec3<f32>(_e714, cross(vec3<f32>(normalize(vec3<f32>(_e693, _e696, _e699)).x, normalize(vec3<f32>(_e693, _e696, _e699)).y, normalize(vec3<f32>(_e693, _e696, _e699)).z), vec3<f32>(_e237, _e238, _e239)).y, cross(vec3<f32>(normalize(vec3<f32>(_e693, _e696, _e699)).x, normalize(vec3<f32>(_e693, _e696, _e699)).y, normalize(vec3<f32>(_e693, _e696, _e699)).z), vec3<f32>(_e237, _e238, _e239)).z)).y * _e734));
            let _e740 = (normalize(vec3<f32>(_e54, _e59, _e45)).z - (normalize(vec3<f32>(_e714, cross(vec3<f32>(normalize(vec3<f32>(_e693, _e696, _e699)).x, normalize(vec3<f32>(_e693, _e696, _e699)).y, normalize(vec3<f32>(_e693, _e696, _e699)).z), vec3<f32>(_e237, _e238, _e239)).y, cross(vec3<f32>(normalize(vec3<f32>(_e693, _e696, _e699)).x, normalize(vec3<f32>(_e693, _e696, _e699)).y, normalize(vec3<f32>(_e693, _e696, _e699)).z), vec3<f32>(_e237, _e238, _e239)).z)).z * _e734));
            let _e744 = max(length(vec3<f32>(_e736, _e738, _e740)), 0.000001f);
            let _e757 = (sign(dot(vec3<f32>(normalize(vec3<f32>(_e693, _e696, _e699)).x, normalize(vec3<f32>(_e693, _e696, _e699)).y, normalize(vec3<f32>(_e693, _e696, _e699)).z), vec3<f32>(_e736, _e738, _e740))) * acos(clamp((dot(vec3<f32>(_e736, _e738, _e740), vec3<f32>(_e237, _e238, _e239)) / _e744), 0f, 1f)));
            let _e760 = cos((_e757 + 1.5707964f));
            let _e763 = cos((_e757 + -1.5707964f));
            horizon_pos_u002e_sroa_u002e_0_u002e_0563_ = _e760;
            horizon_neg_u002e_sroa_u002e_0_u002e_0562_ = _e763;
            step_u002e_sroa_u002e_0_u002e_0561_ = 1u;
            loop {
                let _e769 = horizon_pos_u002e_sroa_u002e_0_u002e_0563_;
                let _e771 = horizon_neg_u002e_sroa_u002e_0_u002e_0562_;
                let _e773 = step_u002e_sroa_u002e_0_u002e_0561_;
                let _e777 = ((max(length(vec3<f32>((((((_e596 + (((_e579 * _e571) + (_e573 * _e584)) + (_e575 * _e590))) / _e643) * 0.5f) + 0.5f) - uv.x), ((0.5f - (((_e618 + (((_e571 * _e601) + (_e573 * _e606)) + (_e575 * _e612))) / _e643) * 0.5f)) - uv.y), 0f)), _e241) * f32(_e773)) * 0.25f);
                let _e778 = (_e689 * _e777);
                let _e779 = (_e690 * _e777);
                let _e780 = (uv.x + _e778);
                let _e781 = (uv.y + _e779);
                if ((_e781 <= 1f) & ((_e780 <= 1f) & ((_e780 >= 0f) & (_e781 >= 0f)))) {
                    let _e802 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(u32((_e780 * _e8))), bitcast<i32>(u32((_e781 * _e13)))), 0i);
                    if (_e802 < 1f) {
                        let _e808 = ((_e780 * 2f) + -1f);
                        let _e812 = (1f - (_e781 * 2f));
                        let _e816 = uniform_struct_0_2_.f4_.x;
                        let _e821 = uniform_struct_0_2_.f5_.x;
                        let _e827 = uniform_struct_0_2_.f6_.x;
                        let _e833 = uniform_struct_0_2_.f7_.x;
                        let _e838 = uniform_struct_0_2_.f4_.w;
                        let _e843 = uniform_struct_0_2_.f5_.w;
                        let _e849 = uniform_struct_0_2_.f6_.w;
                        let _e855 = uniform_struct_0_2_.f7_.w;
                        let _e863 = uniform_struct_0_2_.f4_.y;
                        let _e868 = uniform_struct_0_2_.f5_.y;
                        let _e874 = uniform_struct_0_2_.f6_.y;
                        let _e880 = uniform_struct_0_2_.f7_.y;
                        let _e885 = uniform_struct_0_2_.f4_.w;
                        let _e890 = uniform_struct_0_2_.f5_.w;
                        let _e896 = uniform_struct_0_2_.f6_.w;
                        let _e902 = uniform_struct_0_2_.f7_.w;
                        let _e910 = uniform_struct_0_2_.f4_.z;
                        let _e915 = uniform_struct_0_2_.f5_.z;
                        let _e921 = uniform_struct_0_2_.f6_.z;
                        let _e927 = uniform_struct_0_2_.f7_.z;
                        let _e932 = uniform_struct_0_2_.f4_.w;
                        let _e937 = uniform_struct_0_2_.f5_.w;
                        let _e943 = uniform_struct_0_2_.f6_.w;
                        let _e949 = uniform_struct_0_2_.f7_.w;
                        let _e954 = (((_e833 + (((_e808 * _e816) + (_e812 * _e821)) + (_e802 * _e827))) / max((_e855 + (((_e808 * _e838) + (_e812 * _e843)) + (_e802 * _e849))), 0.000001f)) - _e123);
                        let _e955 = (((_e880 + (((_e808 * _e863) + (_e812 * _e868)) + (_e802 * _e874))) / max((_e902 + (((_e808 * _e885) + (_e812 * _e890)) + (_e802 * _e896))), 0.000001f)) - _e170);
                        let _e956 = (((_e927 + (((_e808 * _e910) + (_e812 * _e915)) + (_e802 * _e921))) / max((_e949 + (((_e808 * _e932) + (_e812 * _e937)) + (_e802 * _e943))), 0.000001f)) - _e217);
                        let _e958 = length(vec3<f32>(_e954, _e955, _e956));
                        if ((_e958 > 0.000001f) & (_e958 < _e566)) {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_ = mix(_e760, dot(vec3<f32>((_e954 / _e958), (_e955 / _e958), (_e956 / _e958)), vec3<f32>(_e237, _e238, _e239)), clamp((1f - (_e958 / max(_e566, 0.000001f))), 0f, 1f));
                        } else {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_ = -1f;
                        }
                    } else {
                        cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_ = -1f;
                    }
                } else {
                    cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_ = -1f;
                }
                let _e984 = cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i415_;
                let _e985 = max(_e769, _e984);
                out_u002e_22_ = _e985;
                let _e987 = (uv.x - _e778);
                let _e988 = (uv.y - _e779);
                if ((_e988 <= 1f) & ((_e987 <= 1f) & ((_e987 >= 0f) & (_e988 >= 0f)))) {
                    let _e1009 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(u32((_e987 * _e8))), bitcast<i32>(u32((_e988 * _e13)))), 0i);
                    if (_e1009 < 1f) {
                        let _e1015 = ((_e987 * 2f) + -1f);
                        let _e1019 = (1f - (_e988 * 2f));
                        let _e1023 = uniform_struct_0_2_.f4_.x;
                        let _e1028 = uniform_struct_0_2_.f5_.x;
                        let _e1034 = uniform_struct_0_2_.f6_.x;
                        let _e1040 = uniform_struct_0_2_.f7_.x;
                        let _e1045 = uniform_struct_0_2_.f4_.w;
                        let _e1050 = uniform_struct_0_2_.f5_.w;
                        let _e1056 = uniform_struct_0_2_.f6_.w;
                        let _e1062 = uniform_struct_0_2_.f7_.w;
                        let _e1070 = uniform_struct_0_2_.f4_.y;
                        let _e1075 = uniform_struct_0_2_.f5_.y;
                        let _e1081 = uniform_struct_0_2_.f6_.y;
                        let _e1087 = uniform_struct_0_2_.f7_.y;
                        let _e1092 = uniform_struct_0_2_.f4_.w;
                        let _e1097 = uniform_struct_0_2_.f5_.w;
                        let _e1103 = uniform_struct_0_2_.f6_.w;
                        let _e1109 = uniform_struct_0_2_.f7_.w;
                        let _e1117 = uniform_struct_0_2_.f4_.z;
                        let _e1122 = uniform_struct_0_2_.f5_.z;
                        let _e1128 = uniform_struct_0_2_.f6_.z;
                        let _e1134 = uniform_struct_0_2_.f7_.z;
                        let _e1139 = uniform_struct_0_2_.f4_.w;
                        let _e1144 = uniform_struct_0_2_.f5_.w;
                        let _e1150 = uniform_struct_0_2_.f6_.w;
                        let _e1156 = uniform_struct_0_2_.f7_.w;
                        let _e1161 = (((_e1040 + (((_e1015 * _e1023) + (_e1019 * _e1028)) + (_e1009 * _e1034))) / max((_e1062 + (((_e1015 * _e1045) + (_e1019 * _e1050)) + (_e1009 * _e1056))), 0.000001f)) - _e123);
                        let _e1162 = (((_e1087 + (((_e1015 * _e1070) + (_e1019 * _e1075)) + (_e1009 * _e1081))) / max((_e1109 + (((_e1015 * _e1092) + (_e1019 * _e1097)) + (_e1009 * _e1103))), 0.000001f)) - _e170);
                        let _e1163 = (((_e1134 + (((_e1015 * _e1117) + (_e1019 * _e1122)) + (_e1009 * _e1128))) / max((_e1156 + (((_e1015 * _e1139) + (_e1019 * _e1144)) + (_e1009 * _e1150))), 0.000001f)) - _e217);
                        let _e1165 = length(vec3<f32>(_e1161, _e1162, _e1163));
                        if ((_e1165 > 0.000001f) & (_e1165 < _e566)) {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = mix(_e763, dot(vec3<f32>((_e1161 / _e1165), (_e1162 / _e1165), (_e1163 / _e1165)), vec3<f32>(_e237, _e238, _e239)), clamp((1f - (_e1165 / max(_e566, 0.000001f))), 0f, 1f));
                        } else {
                            cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = -1f;
                        }
                    } else {
                        cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = -1f;
                    }
                } else {
                    cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i = -1f;
                }
                let _e1191 = cos_h_u002e_sroa_u002e_0_u002e_0_u002e_i;
                let _e1192 = max(_e771, _e1191);
                out_u002e_26_ = _e1192;
                let _e1195 = (_e773 + 1u);
                if (_e1195 == 5u) {
                    break;
                } else {
                    horizon_pos_u002e_sroa_u002e_0_u002e_0563_ = _e985;
                    horizon_neg_u002e_sroa_u002e_0_u002e_0562_ = _e1192;
                    step_u002e_sroa_u002e_0_u002e_0561_ = _e1195;
                    continue;
                }
            }
            let _e1204 = out_u002e_22_;
            let _e1206 = out_u002e_26_;
            let _e1216 = (acos(clamp(_e1204, -1f, 1f)) * 2f);
            let _e1227 = (acos(clamp(_e1206, -1f, 1f)) * -2f);
            let _e1239 = (_e682 + (_e744 * ((((cos(_e757) - cos((_e1216 - _e757))) + (_e1216 * sin(_e757))) * 0.25f) + (((cos(_e757) - cos((_e1227 - _e757))) + (_e1227 * sin(_e757))) * 0.25f))));
            out_u002e_17_ = _e1239;
            if (_e684 == 0u) {
                visibility_u002e_sroa_u002e_0_u002e_0565_ = _e1239;
                slice_u002e_sroa_u002e_0_u002e_0564_ = (_e684 + 1u);
                continue;
            } else {
                break;
            }
        }
        let _e1248 = out_u002e_17_;
        let _e1255 = clamp(mix(1f, (_e1248 * 0.5f), _e569), 0f, 1f);
        fragment_out = vec4<f32>(_e1255, _e1255, _e1255, 1f);
    } else {
        fragment_out = vec4<f32>(1f, 1f, 1f, 1f);
    }
    let _e1266 = fragment_out;
    return _e1266;
}
