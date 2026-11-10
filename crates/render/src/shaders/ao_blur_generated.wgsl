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
var tex_0_0_: texture_2d<f32>;
@group(0) @binding(1) 
var sampler_0_1_: sampler;
@group(0) @binding(2) 
var tex_0_2_: texture_depth_2d;
@group(0) @binding(4) 
var<uniform> uniform_struct_0_4_: UniformStruct0_;
@group(0) @binding(3) 
var<uniform> uniform_struct_0_3_: UniformStruct1_;

@fragment 
fn fs_main(@builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var fragment_out: vec4<f32>;

    let _e8 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e11 = textureDimensions(tex_0_2_);
    let _e14 = f32(bitcast<i32>(_e11.x));
    let _e16 = textureDimensions(tex_0_2_);
    let _e19 = f32(bitcast<i32>(_e16.y));
    let _e24 = u32((_e14 + -1f));
    let _e30 = u32((_e19 + -1f));
    let _e37 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((uv.x * _e14)), _e24)), bitcast<i32>(min(u32((uv.y * _e19)), _e30))), 0i);
    let _e41 = ((uv.x * 2f) + -1f);
    let _e45 = (1f - (uv.y * 2f));
    let _e49 = uniform_struct_0_4_.f4_.x;
    let _e54 = uniform_struct_0_4_.f5_.x;
    let _e60 = uniform_struct_0_4_.f6_.x;
    let _e66 = uniform_struct_0_4_.f7_.x;
    let _e71 = uniform_struct_0_4_.f4_.w;
    let _e76 = uniform_struct_0_4_.f5_.w;
    let _e82 = uniform_struct_0_4_.f6_.w;
    let _e88 = uniform_struct_0_4_.f7_.w;
    let _e94 = uniform_struct_0_4_.f4_.y;
    let _e99 = uniform_struct_0_4_.f5_.y;
    let _e105 = uniform_struct_0_4_.f6_.y;
    let _e111 = uniform_struct_0_4_.f7_.y;
    let _e116 = uniform_struct_0_4_.f4_.w;
    let _e121 = uniform_struct_0_4_.f5_.w;
    let _e127 = uniform_struct_0_4_.f6_.w;
    let _e133 = uniform_struct_0_4_.f7_.w;
    let _e139 = uniform_struct_0_4_.f4_.z;
    let _e144 = uniform_struct_0_4_.f5_.z;
    let _e150 = uniform_struct_0_4_.f6_.z;
    let _e156 = uniform_struct_0_4_.f7_.z;
    let _e161 = uniform_struct_0_4_.f4_.w;
    let _e166 = uniform_struct_0_4_.f5_.w;
    let _e172 = uniform_struct_0_4_.f6_.w;
    let _e178 = uniform_struct_0_4_.f7_.w;
    let _e184 = uniform_struct_0_4_.f8_.x;
    let _e188 = uniform_struct_0_4_.f8_.y;
    let _e192 = uniform_struct_0_4_.f8_.z;
    let _e197 = length(vec3<f32>((_e184 - ((_e66 + (((_e41 * _e49) + (_e45 * _e54)) + (_e37 * _e60))) / (_e88 + (((_e41 * _e71) + (_e45 * _e76)) + (_e37 * _e82))))), (_e188 - ((_e111 + (((_e41 * _e94) + (_e45 * _e99)) + (_e37 * _e105))) / (_e133 + (((_e41 * _e116) + (_e45 * _e121)) + (_e37 * _e127))))), (_e192 - ((_e156 + (((_e41 * _e139) + (_e45 * _e144)) + (_e37 * _e150))) / (_e178 + (((_e41 * _e161) + (_e45 * _e166)) + (_e37 * _e172)))))));
    let _e199 = textureDimensions(tex_0_0_);
    let _e204 = (1f / f32(bitcast<i32>(_e199.x)));
    let _e206 = textureDimensions(tex_0_0_);
    let _e211 = (1f / f32(bitcast<i32>(_e206.y)));
    let _e214 = uniform_struct_0_3_.f3_;
    let _e216 = max(_e214, 0.000001f);
    let _e218 = (_e204 * 1.5f);
    let _e219 = (uv.x + _e218);
    let _e222 = (uv.y + (_e211 * 0f));
    let _e225 = clamp(_e219, 0f, 1f);
    let _e228 = clamp(_e222, 0f, 1f);
    let _e240 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e225 * _e14)), _e24)), bitcast<i32>(min(u32((_e228 * _e19)), _e30))), 0i);
    let _e244 = ((_e225 * 2f) + -1f);
    let _e248 = (1f - (_e228 * 2f));
    let _e252 = uniform_struct_0_4_.f4_.x;
    let _e257 = uniform_struct_0_4_.f5_.x;
    let _e263 = uniform_struct_0_4_.f6_.x;
    let _e269 = uniform_struct_0_4_.f7_.x;
    let _e274 = uniform_struct_0_4_.f4_.w;
    let _e279 = uniform_struct_0_4_.f5_.w;
    let _e285 = uniform_struct_0_4_.f6_.w;
    let _e291 = uniform_struct_0_4_.f7_.w;
    let _e297 = uniform_struct_0_4_.f4_.y;
    let _e302 = uniform_struct_0_4_.f5_.y;
    let _e308 = uniform_struct_0_4_.f6_.y;
    let _e314 = uniform_struct_0_4_.f7_.y;
    let _e319 = uniform_struct_0_4_.f4_.w;
    let _e324 = uniform_struct_0_4_.f5_.w;
    let _e330 = uniform_struct_0_4_.f6_.w;
    let _e336 = uniform_struct_0_4_.f7_.w;
    let _e342 = uniform_struct_0_4_.f4_.z;
    let _e347 = uniform_struct_0_4_.f5_.z;
    let _e353 = uniform_struct_0_4_.f6_.z;
    let _e359 = uniform_struct_0_4_.f7_.z;
    let _e364 = uniform_struct_0_4_.f4_.w;
    let _e369 = uniform_struct_0_4_.f5_.w;
    let _e375 = uniform_struct_0_4_.f6_.w;
    let _e381 = uniform_struct_0_4_.f7_.w;
    let _e387 = uniform_struct_0_4_.f8_.x;
    let _e391 = uniform_struct_0_4_.f8_.y;
    let _e395 = uniform_struct_0_4_.f8_.z;
    let _e402 = ((_e197 - length(vec3<f32>((_e387 - ((_e269 + (((_e244 * _e252) + (_e248 * _e257)) + (_e240 * _e263))) / (_e291 + (((_e244 * _e274) + (_e248 * _e279)) + (_e240 * _e285))))), (_e391 - ((_e314 + (((_e244 * _e297) + (_e248 * _e302)) + (_e240 * _e308))) / (_e336 + (((_e244 * _e319) + (_e248 * _e324)) + (_e240 * _e330))))), (_e395 - ((_e359 + (((_e244 * _e342) + (_e248 * _e347)) + (_e240 * _e353))) / (_e381 + (((_e244 * _e364) + (_e248 * _e369)) + (_e240 * _e375)))))))) / _e216);
    let _e405 = exp((_e402 * -(_e402)));
    let _e410 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(_e219, uv.y), 0f);
    let _e421 = clamp((uv.x - (_e204 * 1.5f)), 0f, 1f);
    let _e424 = clamp(_e222, 0f, 1f);
    let _e436 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e421 * _e14)), _e24)), bitcast<i32>(min(u32((_e424 * _e19)), _e30))), 0i);
    let _e440 = ((_e421 * 2f) + -1f);
    let _e444 = (1f - (_e424 * 2f));
    let _e448 = uniform_struct_0_4_.f4_.x;
    let _e453 = uniform_struct_0_4_.f5_.x;
    let _e459 = uniform_struct_0_4_.f6_.x;
    let _e465 = uniform_struct_0_4_.f7_.x;
    let _e470 = uniform_struct_0_4_.f4_.w;
    let _e475 = uniform_struct_0_4_.f5_.w;
    let _e481 = uniform_struct_0_4_.f6_.w;
    let _e487 = uniform_struct_0_4_.f7_.w;
    let _e493 = uniform_struct_0_4_.f4_.y;
    let _e498 = uniform_struct_0_4_.f5_.y;
    let _e504 = uniform_struct_0_4_.f6_.y;
    let _e510 = uniform_struct_0_4_.f7_.y;
    let _e515 = uniform_struct_0_4_.f4_.w;
    let _e520 = uniform_struct_0_4_.f5_.w;
    let _e526 = uniform_struct_0_4_.f6_.w;
    let _e532 = uniform_struct_0_4_.f7_.w;
    let _e538 = uniform_struct_0_4_.f4_.z;
    let _e543 = uniform_struct_0_4_.f5_.z;
    let _e549 = uniform_struct_0_4_.f6_.z;
    let _e555 = uniform_struct_0_4_.f7_.z;
    let _e560 = uniform_struct_0_4_.f4_.w;
    let _e565 = uniform_struct_0_4_.f5_.w;
    let _e571 = uniform_struct_0_4_.f6_.w;
    let _e577 = uniform_struct_0_4_.f7_.w;
    let _e583 = uniform_struct_0_4_.f8_.x;
    let _e587 = uniform_struct_0_4_.f8_.y;
    let _e591 = uniform_struct_0_4_.f8_.z;
    let _e598 = ((_e197 - length(vec3<f32>((_e583 - ((_e465 + (((_e440 * _e448) + (_e444 * _e453)) + (_e436 * _e459))) / (_e487 + (((_e440 * _e470) + (_e444 * _e475)) + (_e436 * _e481))))), (_e587 - ((_e510 + (((_e440 * _e493) + (_e444 * _e498)) + (_e436 * _e504))) / (_e532 + (((_e440 * _e515) + (_e444 * _e520)) + (_e436 * _e526))))), (_e591 - ((_e555 + (((_e440 * _e538) + (_e444 * _e543)) + (_e436 * _e549))) / (_e577 + (((_e440 * _e560) + (_e444 * _e565)) + (_e436 * _e571)))))))) / _e216);
    let _e601 = exp((_e598 * -(_e598)));
    let _e607 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>((uv.x - _e218), uv.y), 0f);
    let _e614 = (uv.x + (_e204 * 0f));
    let _e616 = (_e211 * 1.5f);
    let _e617 = (uv.y + _e616);
    let _e620 = clamp(_e614, 0f, 1f);
    let _e623 = clamp(_e617, 0f, 1f);
    let _e635 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e620 * _e14)), _e24)), bitcast<i32>(min(u32((_e623 * _e19)), _e30))), 0i);
    let _e639 = ((_e620 * 2f) + -1f);
    let _e643 = (1f - (_e623 * 2f));
    let _e647 = uniform_struct_0_4_.f4_.x;
    let _e652 = uniform_struct_0_4_.f5_.x;
    let _e658 = uniform_struct_0_4_.f6_.x;
    let _e664 = uniform_struct_0_4_.f7_.x;
    let _e669 = uniform_struct_0_4_.f4_.w;
    let _e674 = uniform_struct_0_4_.f5_.w;
    let _e680 = uniform_struct_0_4_.f6_.w;
    let _e686 = uniform_struct_0_4_.f7_.w;
    let _e692 = uniform_struct_0_4_.f4_.y;
    let _e697 = uniform_struct_0_4_.f5_.y;
    let _e703 = uniform_struct_0_4_.f6_.y;
    let _e709 = uniform_struct_0_4_.f7_.y;
    let _e714 = uniform_struct_0_4_.f4_.w;
    let _e719 = uniform_struct_0_4_.f5_.w;
    let _e725 = uniform_struct_0_4_.f6_.w;
    let _e731 = uniform_struct_0_4_.f7_.w;
    let _e737 = uniform_struct_0_4_.f4_.z;
    let _e742 = uniform_struct_0_4_.f5_.z;
    let _e748 = uniform_struct_0_4_.f6_.z;
    let _e754 = uniform_struct_0_4_.f7_.z;
    let _e759 = uniform_struct_0_4_.f4_.w;
    let _e764 = uniform_struct_0_4_.f5_.w;
    let _e770 = uniform_struct_0_4_.f6_.w;
    let _e776 = uniform_struct_0_4_.f7_.w;
    let _e782 = uniform_struct_0_4_.f8_.x;
    let _e786 = uniform_struct_0_4_.f8_.y;
    let _e790 = uniform_struct_0_4_.f8_.z;
    let _e797 = ((_e197 - length(vec3<f32>((_e782 - ((_e664 + (((_e639 * _e647) + (_e643 * _e652)) + (_e635 * _e658))) / (_e686 + (((_e639 * _e669) + (_e643 * _e674)) + (_e635 * _e680))))), (_e786 - ((_e709 + (((_e639 * _e692) + (_e643 * _e697)) + (_e635 * _e703))) / (_e731 + (((_e639 * _e714) + (_e643 * _e719)) + (_e635 * _e725))))), (_e790 - ((_e754 + (((_e639 * _e737) + (_e643 * _e742)) + (_e635 * _e748))) / (_e776 + (((_e639 * _e759) + (_e643 * _e764)) + (_e635 * _e770)))))))) / _e216);
    let _e800 = exp((_e797 * -(_e797)));
    let _e805 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, _e617), 0f);
    let _e815 = clamp(_e614, 0f, 1f);
    let _e818 = clamp((uv.y - (_e211 * 1.5f)), 0f, 1f);
    let _e830 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e815 * _e14)), _e24)), bitcast<i32>(min(u32((_e818 * _e19)), _e30))), 0i);
    let _e834 = ((_e815 * 2f) + -1f);
    let _e838 = (1f - (_e818 * 2f));
    let _e842 = uniform_struct_0_4_.f4_.x;
    let _e847 = uniform_struct_0_4_.f5_.x;
    let _e853 = uniform_struct_0_4_.f6_.x;
    let _e859 = uniform_struct_0_4_.f7_.x;
    let _e864 = uniform_struct_0_4_.f4_.w;
    let _e869 = uniform_struct_0_4_.f5_.w;
    let _e875 = uniform_struct_0_4_.f6_.w;
    let _e881 = uniform_struct_0_4_.f7_.w;
    let _e887 = uniform_struct_0_4_.f4_.y;
    let _e892 = uniform_struct_0_4_.f5_.y;
    let _e898 = uniform_struct_0_4_.f6_.y;
    let _e904 = uniform_struct_0_4_.f7_.y;
    let _e909 = uniform_struct_0_4_.f4_.w;
    let _e914 = uniform_struct_0_4_.f5_.w;
    let _e920 = uniform_struct_0_4_.f6_.w;
    let _e926 = uniform_struct_0_4_.f7_.w;
    let _e932 = uniform_struct_0_4_.f4_.z;
    let _e937 = uniform_struct_0_4_.f5_.z;
    let _e943 = uniform_struct_0_4_.f6_.z;
    let _e949 = uniform_struct_0_4_.f7_.z;
    let _e954 = uniform_struct_0_4_.f4_.w;
    let _e959 = uniform_struct_0_4_.f5_.w;
    let _e965 = uniform_struct_0_4_.f6_.w;
    let _e971 = uniform_struct_0_4_.f7_.w;
    let _e977 = uniform_struct_0_4_.f8_.x;
    let _e981 = uniform_struct_0_4_.f8_.y;
    let _e985 = uniform_struct_0_4_.f8_.z;
    let _e992 = ((_e197 - length(vec3<f32>((_e977 - ((_e859 + (((_e834 * _e842) + (_e838 * _e847)) + (_e830 * _e853))) / (_e881 + (((_e834 * _e864) + (_e838 * _e869)) + (_e830 * _e875))))), (_e981 - ((_e904 + (((_e834 * _e887) + (_e838 * _e892)) + (_e830 * _e898))) / (_e926 + (((_e834 * _e909) + (_e838 * _e914)) + (_e830 * _e920))))), (_e985 - ((_e949 + (((_e834 * _e932) + (_e838 * _e937)) + (_e830 * _e943))) / (_e971 + (((_e834 * _e954) + (_e838 * _e959)) + (_e830 * _e965)))))))) / _e216);
    let _e995 = exp((_e992 * -(_e992)));
    let _e1001 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, (uv.y - _e616)), 0f);
    let _e1006 = (((((_e8.x + (_e405 * _e410.x)) + (_e601 * _e607.x)) + (_e800 * _e805.x)) + (_e995 * _e1001.x)) / ((((_e405 + 1f) + _e601) + _e800) + _e995));
    fragment_out = vec4<f32>(_e1006, _e1006, _e1006, 1f);
    let _e1011 = fragment_out;
    return _e1011;
}
