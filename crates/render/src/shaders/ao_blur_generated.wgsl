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
    let _e13 = f32(_e11.x);
    let _e15 = textureDimensions(tex_0_2_);
    let _e17 = f32(_e15.y);
    let _e22 = u32((_e13 + -1f));
    let _e28 = u32((_e17 + -1f));
    let _e35 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((uv.x * _e13)), _e22)), bitcast<i32>(min(u32((uv.y * _e17)), _e28))), 0i);
    let _e39 = ((uv.x * 2f) + -1f);
    let _e43 = (1f - (uv.y * 2f));
    let _e47 = uniform_struct_0_4_.f4_.x;
    let _e52 = uniform_struct_0_4_.f5_.x;
    let _e58 = uniform_struct_0_4_.f6_.x;
    let _e64 = uniform_struct_0_4_.f7_.x;
    let _e69 = uniform_struct_0_4_.f4_.w;
    let _e74 = uniform_struct_0_4_.f5_.w;
    let _e80 = uniform_struct_0_4_.f6_.w;
    let _e86 = uniform_struct_0_4_.f7_.w;
    let _e94 = uniform_struct_0_4_.f4_.y;
    let _e99 = uniform_struct_0_4_.f5_.y;
    let _e105 = uniform_struct_0_4_.f6_.y;
    let _e111 = uniform_struct_0_4_.f7_.y;
    let _e116 = uniform_struct_0_4_.f4_.w;
    let _e121 = uniform_struct_0_4_.f5_.w;
    let _e127 = uniform_struct_0_4_.f6_.w;
    let _e133 = uniform_struct_0_4_.f7_.w;
    let _e141 = uniform_struct_0_4_.f4_.z;
    let _e146 = uniform_struct_0_4_.f5_.z;
    let _e152 = uniform_struct_0_4_.f6_.z;
    let _e158 = uniform_struct_0_4_.f7_.z;
    let _e163 = uniform_struct_0_4_.f4_.w;
    let _e168 = uniform_struct_0_4_.f5_.w;
    let _e174 = uniform_struct_0_4_.f6_.w;
    let _e180 = uniform_struct_0_4_.f7_.w;
    let _e188 = uniform_struct_0_4_.f8_.x;
    let _e192 = uniform_struct_0_4_.f8_.y;
    let _e196 = uniform_struct_0_4_.f8_.z;
    let _e201 = length(vec3<f32>((_e188 - ((_e64 + (((_e39 * _e47) + (_e43 * _e52)) + (_e35 * _e58))) / max((_e86 + (((_e39 * _e69) + (_e43 * _e74)) + (_e35 * _e80))), 0.000001f))), (_e192 - ((_e111 + (((_e39 * _e94) + (_e43 * _e99)) + (_e35 * _e105))) / max((_e133 + (((_e39 * _e116) + (_e43 * _e121)) + (_e35 * _e127))), 0.000001f))), (_e196 - ((_e158 + (((_e39 * _e141) + (_e43 * _e146)) + (_e35 * _e152))) / max((_e180 + (((_e39 * _e163) + (_e43 * _e168)) + (_e35 * _e174))), 0.000001f)))));
    let _e203 = textureDimensions(tex_0_0_);
    let _e207 = (1f / f32(_e203.x));
    let _e209 = textureDimensions(tex_0_0_);
    let _e213 = (1f / f32(_e209.y));
    let _e216 = uniform_struct_0_3_.f3_;
    let _e218 = max(_e216, 0.000001f);
    let _e220 = (_e207 * 1.5f);
    let _e221 = (uv.x + _e220);
    let _e224 = (uv.y + (_e213 * 0f));
    let _e227 = clamp(_e221, 0f, 1f);
    let _e230 = clamp(_e224, 0f, 1f);
    let _e242 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e227 * _e13)), _e22)), bitcast<i32>(min(u32((_e230 * _e17)), _e28))), 0i);
    let _e246 = ((_e227 * 2f) + -1f);
    let _e250 = (1f - (_e230 * 2f));
    let _e254 = uniform_struct_0_4_.f4_.x;
    let _e259 = uniform_struct_0_4_.f5_.x;
    let _e265 = uniform_struct_0_4_.f6_.x;
    let _e271 = uniform_struct_0_4_.f7_.x;
    let _e276 = uniform_struct_0_4_.f4_.w;
    let _e281 = uniform_struct_0_4_.f5_.w;
    let _e287 = uniform_struct_0_4_.f6_.w;
    let _e293 = uniform_struct_0_4_.f7_.w;
    let _e301 = uniform_struct_0_4_.f4_.y;
    let _e306 = uniform_struct_0_4_.f5_.y;
    let _e312 = uniform_struct_0_4_.f6_.y;
    let _e318 = uniform_struct_0_4_.f7_.y;
    let _e323 = uniform_struct_0_4_.f4_.w;
    let _e328 = uniform_struct_0_4_.f5_.w;
    let _e334 = uniform_struct_0_4_.f6_.w;
    let _e340 = uniform_struct_0_4_.f7_.w;
    let _e348 = uniform_struct_0_4_.f4_.z;
    let _e353 = uniform_struct_0_4_.f5_.z;
    let _e359 = uniform_struct_0_4_.f6_.z;
    let _e365 = uniform_struct_0_4_.f7_.z;
    let _e370 = uniform_struct_0_4_.f4_.w;
    let _e375 = uniform_struct_0_4_.f5_.w;
    let _e381 = uniform_struct_0_4_.f6_.w;
    let _e387 = uniform_struct_0_4_.f7_.w;
    let _e395 = uniform_struct_0_4_.f8_.x;
    let _e399 = uniform_struct_0_4_.f8_.y;
    let _e403 = uniform_struct_0_4_.f8_.z;
    let _e410 = ((_e201 - length(vec3<f32>((_e395 - ((_e271 + (((_e246 * _e254) + (_e250 * _e259)) + (_e242 * _e265))) / max((_e293 + (((_e246 * _e276) + (_e250 * _e281)) + (_e242 * _e287))), 0.000001f))), (_e399 - ((_e318 + (((_e246 * _e301) + (_e250 * _e306)) + (_e242 * _e312))) / max((_e340 + (((_e246 * _e323) + (_e250 * _e328)) + (_e242 * _e334))), 0.000001f))), (_e403 - ((_e365 + (((_e246 * _e348) + (_e250 * _e353)) + (_e242 * _e359))) / max((_e387 + (((_e246 * _e370) + (_e250 * _e375)) + (_e242 * _e381))), 0.000001f)))))) / _e218);
    let _e413 = exp((_e410 * -(_e410)));
    let _e418 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(_e221, uv.y), 0f);
    let _e429 = clamp((uv.x - (_e207 * 1.5f)), 0f, 1f);
    let _e432 = clamp(_e224, 0f, 1f);
    let _e444 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e429 * _e13)), _e22)), bitcast<i32>(min(u32((_e432 * _e17)), _e28))), 0i);
    let _e448 = ((_e429 * 2f) + -1f);
    let _e452 = (1f - (_e432 * 2f));
    let _e456 = uniform_struct_0_4_.f4_.x;
    let _e461 = uniform_struct_0_4_.f5_.x;
    let _e467 = uniform_struct_0_4_.f6_.x;
    let _e473 = uniform_struct_0_4_.f7_.x;
    let _e478 = uniform_struct_0_4_.f4_.w;
    let _e483 = uniform_struct_0_4_.f5_.w;
    let _e489 = uniform_struct_0_4_.f6_.w;
    let _e495 = uniform_struct_0_4_.f7_.w;
    let _e503 = uniform_struct_0_4_.f4_.y;
    let _e508 = uniform_struct_0_4_.f5_.y;
    let _e514 = uniform_struct_0_4_.f6_.y;
    let _e520 = uniform_struct_0_4_.f7_.y;
    let _e525 = uniform_struct_0_4_.f4_.w;
    let _e530 = uniform_struct_0_4_.f5_.w;
    let _e536 = uniform_struct_0_4_.f6_.w;
    let _e542 = uniform_struct_0_4_.f7_.w;
    let _e550 = uniform_struct_0_4_.f4_.z;
    let _e555 = uniform_struct_0_4_.f5_.z;
    let _e561 = uniform_struct_0_4_.f6_.z;
    let _e567 = uniform_struct_0_4_.f7_.z;
    let _e572 = uniform_struct_0_4_.f4_.w;
    let _e577 = uniform_struct_0_4_.f5_.w;
    let _e583 = uniform_struct_0_4_.f6_.w;
    let _e589 = uniform_struct_0_4_.f7_.w;
    let _e597 = uniform_struct_0_4_.f8_.x;
    let _e601 = uniform_struct_0_4_.f8_.y;
    let _e605 = uniform_struct_0_4_.f8_.z;
    let _e612 = ((_e201 - length(vec3<f32>((_e597 - ((_e473 + (((_e448 * _e456) + (_e452 * _e461)) + (_e444 * _e467))) / max((_e495 + (((_e448 * _e478) + (_e452 * _e483)) + (_e444 * _e489))), 0.000001f))), (_e601 - ((_e520 + (((_e448 * _e503) + (_e452 * _e508)) + (_e444 * _e514))) / max((_e542 + (((_e448 * _e525) + (_e452 * _e530)) + (_e444 * _e536))), 0.000001f))), (_e605 - ((_e567 + (((_e448 * _e550) + (_e452 * _e555)) + (_e444 * _e561))) / max((_e589 + (((_e448 * _e572) + (_e452 * _e577)) + (_e444 * _e583))), 0.000001f)))))) / _e218);
    let _e615 = exp((_e612 * -(_e612)));
    let _e621 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>((uv.x - _e220), uv.y), 0f);
    let _e628 = (uv.x + (_e207 * 0f));
    let _e630 = (_e213 * 1.5f);
    let _e631 = (uv.y + _e630);
    let _e634 = clamp(_e628, 0f, 1f);
    let _e637 = clamp(_e631, 0f, 1f);
    let _e649 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e634 * _e13)), _e22)), bitcast<i32>(min(u32((_e637 * _e17)), _e28))), 0i);
    let _e653 = ((_e634 * 2f) + -1f);
    let _e657 = (1f - (_e637 * 2f));
    let _e661 = uniform_struct_0_4_.f4_.x;
    let _e666 = uniform_struct_0_4_.f5_.x;
    let _e672 = uniform_struct_0_4_.f6_.x;
    let _e678 = uniform_struct_0_4_.f7_.x;
    let _e683 = uniform_struct_0_4_.f4_.w;
    let _e688 = uniform_struct_0_4_.f5_.w;
    let _e694 = uniform_struct_0_4_.f6_.w;
    let _e700 = uniform_struct_0_4_.f7_.w;
    let _e708 = uniform_struct_0_4_.f4_.y;
    let _e713 = uniform_struct_0_4_.f5_.y;
    let _e719 = uniform_struct_0_4_.f6_.y;
    let _e725 = uniform_struct_0_4_.f7_.y;
    let _e730 = uniform_struct_0_4_.f4_.w;
    let _e735 = uniform_struct_0_4_.f5_.w;
    let _e741 = uniform_struct_0_4_.f6_.w;
    let _e747 = uniform_struct_0_4_.f7_.w;
    let _e755 = uniform_struct_0_4_.f4_.z;
    let _e760 = uniform_struct_0_4_.f5_.z;
    let _e766 = uniform_struct_0_4_.f6_.z;
    let _e772 = uniform_struct_0_4_.f7_.z;
    let _e777 = uniform_struct_0_4_.f4_.w;
    let _e782 = uniform_struct_0_4_.f5_.w;
    let _e788 = uniform_struct_0_4_.f6_.w;
    let _e794 = uniform_struct_0_4_.f7_.w;
    let _e802 = uniform_struct_0_4_.f8_.x;
    let _e806 = uniform_struct_0_4_.f8_.y;
    let _e810 = uniform_struct_0_4_.f8_.z;
    let _e817 = ((_e201 - length(vec3<f32>((_e802 - ((_e678 + (((_e653 * _e661) + (_e657 * _e666)) + (_e649 * _e672))) / max((_e700 + (((_e653 * _e683) + (_e657 * _e688)) + (_e649 * _e694))), 0.000001f))), (_e806 - ((_e725 + (((_e653 * _e708) + (_e657 * _e713)) + (_e649 * _e719))) / max((_e747 + (((_e653 * _e730) + (_e657 * _e735)) + (_e649 * _e741))), 0.000001f))), (_e810 - ((_e772 + (((_e653 * _e755) + (_e657 * _e760)) + (_e649 * _e766))) / max((_e794 + (((_e653 * _e777) + (_e657 * _e782)) + (_e649 * _e788))), 0.000001f)))))) / _e218);
    let _e820 = exp((_e817 * -(_e817)));
    let _e825 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, _e631), 0f);
    let _e835 = clamp(_e628, 0f, 1f);
    let _e838 = clamp((uv.y - (_e213 * 1.5f)), 0f, 1f);
    let _e850 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e835 * _e13)), _e22)), bitcast<i32>(min(u32((_e838 * _e17)), _e28))), 0i);
    let _e854 = ((_e835 * 2f) + -1f);
    let _e858 = (1f - (_e838 * 2f));
    let _e862 = uniform_struct_0_4_.f4_.x;
    let _e867 = uniform_struct_0_4_.f5_.x;
    let _e873 = uniform_struct_0_4_.f6_.x;
    let _e879 = uniform_struct_0_4_.f7_.x;
    let _e884 = uniform_struct_0_4_.f4_.w;
    let _e889 = uniform_struct_0_4_.f5_.w;
    let _e895 = uniform_struct_0_4_.f6_.w;
    let _e901 = uniform_struct_0_4_.f7_.w;
    let _e909 = uniform_struct_0_4_.f4_.y;
    let _e914 = uniform_struct_0_4_.f5_.y;
    let _e920 = uniform_struct_0_4_.f6_.y;
    let _e926 = uniform_struct_0_4_.f7_.y;
    let _e931 = uniform_struct_0_4_.f4_.w;
    let _e936 = uniform_struct_0_4_.f5_.w;
    let _e942 = uniform_struct_0_4_.f6_.w;
    let _e948 = uniform_struct_0_4_.f7_.w;
    let _e956 = uniform_struct_0_4_.f4_.z;
    let _e961 = uniform_struct_0_4_.f5_.z;
    let _e967 = uniform_struct_0_4_.f6_.z;
    let _e973 = uniform_struct_0_4_.f7_.z;
    let _e978 = uniform_struct_0_4_.f4_.w;
    let _e983 = uniform_struct_0_4_.f5_.w;
    let _e989 = uniform_struct_0_4_.f6_.w;
    let _e995 = uniform_struct_0_4_.f7_.w;
    let _e1003 = uniform_struct_0_4_.f8_.x;
    let _e1007 = uniform_struct_0_4_.f8_.y;
    let _e1011 = uniform_struct_0_4_.f8_.z;
    let _e1018 = ((_e201 - length(vec3<f32>((_e1003 - ((_e879 + (((_e854 * _e862) + (_e858 * _e867)) + (_e850 * _e873))) / max((_e901 + (((_e854 * _e884) + (_e858 * _e889)) + (_e850 * _e895))), 0.000001f))), (_e1007 - ((_e926 + (((_e854 * _e909) + (_e858 * _e914)) + (_e850 * _e920))) / max((_e948 + (((_e854 * _e931) + (_e858 * _e936)) + (_e850 * _e942))), 0.000001f))), (_e1011 - ((_e973 + (((_e854 * _e956) + (_e858 * _e961)) + (_e850 * _e967))) / max((_e995 + (((_e854 * _e978) + (_e858 * _e983)) + (_e850 * _e989))), 0.000001f)))))) / _e218);
    let _e1021 = exp((_e1018 * -(_e1018)));
    let _e1027 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, (uv.y - _e630)), 0f);
    let _e1032 = (((((_e8.x + (_e413 * _e418.x)) + (_e615 * _e621.x)) + (_e820 * _e825.x)) + (_e1021 * _e1027.x)) / ((((_e413 + 1f) + _e615) + _e820) + _e1021));
    fragment_out = vec4<f32>(_e1032, _e1032, _e1032, 1f);
    let _e1037 = fragment_out;
    return _e1037;
}
