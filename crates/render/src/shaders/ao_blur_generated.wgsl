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
    let _e17 = f32(bitcast<i32>(_e11.y));
    let _e22 = u32((_e14 + -1f));
    let _e28 = u32((_e17 + -1f));
    let _e34 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((uv.x * _e14)), _e22)), bitcast<i32>(min(u32((uv.y * _e17)), _e28))), 0i);
    let _e38 = ((uv.x * 2f) + -1f);
    let _e42 = (1f - (uv.y * 2f));
    let _e46 = uniform_struct_0_4_.f4_.x;
    let _e51 = uniform_struct_0_4_.f5_.x;
    let _e57 = uniform_struct_0_4_.f6_.x;
    let _e63 = uniform_struct_0_4_.f7_.x;
    let _e68 = uniform_struct_0_4_.f4_.w;
    let _e73 = uniform_struct_0_4_.f5_.w;
    let _e79 = uniform_struct_0_4_.f6_.w;
    let _e85 = uniform_struct_0_4_.f7_.w;
    let _e91 = uniform_struct_0_4_.f4_.y;
    let _e96 = uniform_struct_0_4_.f5_.y;
    let _e102 = uniform_struct_0_4_.f6_.y;
    let _e108 = uniform_struct_0_4_.f7_.y;
    let _e113 = uniform_struct_0_4_.f4_.w;
    let _e118 = uniform_struct_0_4_.f5_.w;
    let _e124 = uniform_struct_0_4_.f6_.w;
    let _e130 = uniform_struct_0_4_.f7_.w;
    let _e136 = uniform_struct_0_4_.f4_.z;
    let _e141 = uniform_struct_0_4_.f5_.z;
    let _e147 = uniform_struct_0_4_.f6_.z;
    let _e153 = uniform_struct_0_4_.f7_.z;
    let _e158 = uniform_struct_0_4_.f4_.w;
    let _e163 = uniform_struct_0_4_.f5_.w;
    let _e169 = uniform_struct_0_4_.f6_.w;
    let _e175 = uniform_struct_0_4_.f7_.w;
    let _e181 = uniform_struct_0_4_.f8_.x;
    let _e185 = uniform_struct_0_4_.f8_.y;
    let _e189 = uniform_struct_0_4_.f8_.z;
    let _e194 = length(vec3<f32>((_e181 - ((_e63 + (((_e38 * _e46) + (_e42 * _e51)) + (_e34 * _e57))) / (_e85 + (((_e38 * _e68) + (_e42 * _e73)) + (_e34 * _e79))))), (_e185 - ((_e108 + (((_e38 * _e91) + (_e42 * _e96)) + (_e34 * _e102))) / (_e130 + (((_e38 * _e113) + (_e42 * _e118)) + (_e34 * _e124))))), (_e189 - ((_e153 + (((_e38 * _e136) + (_e42 * _e141)) + (_e34 * _e147))) / (_e175 + (((_e38 * _e158) + (_e42 * _e163)) + (_e34 * _e169)))))));
    let _e195 = textureDimensions(tex_0_0_);
    let _e200 = (1f / f32(bitcast<i32>(_e195.x)));
    let _e205 = (1f / f32(bitcast<i32>(_e195.y)));
    let _e208 = uniform_struct_0_3_.f3_;
    let _e210 = max(_e208, 0.000001f);
    let _e212 = (_e200 * 1.5f);
    let _e213 = (uv.x + _e212);
    let _e216 = (uv.y + (_e205 * 0f));
    let _e219 = clamp(_e213, 0f, 1f);
    let _e222 = clamp(_e216, 0f, 1f);
    let _e233 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e219 * _e14)), _e22)), bitcast<i32>(min(u32((_e222 * _e17)), _e28))), 0i);
    let _e237 = ((_e219 * 2f) + -1f);
    let _e241 = (1f - (_e222 * 2f));
    let _e245 = uniform_struct_0_4_.f4_.x;
    let _e250 = uniform_struct_0_4_.f5_.x;
    let _e256 = uniform_struct_0_4_.f6_.x;
    let _e262 = uniform_struct_0_4_.f7_.x;
    let _e267 = uniform_struct_0_4_.f4_.w;
    let _e272 = uniform_struct_0_4_.f5_.w;
    let _e278 = uniform_struct_0_4_.f6_.w;
    let _e284 = uniform_struct_0_4_.f7_.w;
    let _e290 = uniform_struct_0_4_.f4_.y;
    let _e295 = uniform_struct_0_4_.f5_.y;
    let _e301 = uniform_struct_0_4_.f6_.y;
    let _e307 = uniform_struct_0_4_.f7_.y;
    let _e312 = uniform_struct_0_4_.f4_.w;
    let _e317 = uniform_struct_0_4_.f5_.w;
    let _e323 = uniform_struct_0_4_.f6_.w;
    let _e329 = uniform_struct_0_4_.f7_.w;
    let _e335 = uniform_struct_0_4_.f4_.z;
    let _e340 = uniform_struct_0_4_.f5_.z;
    let _e346 = uniform_struct_0_4_.f6_.z;
    let _e352 = uniform_struct_0_4_.f7_.z;
    let _e357 = uniform_struct_0_4_.f4_.w;
    let _e362 = uniform_struct_0_4_.f5_.w;
    let _e368 = uniform_struct_0_4_.f6_.w;
    let _e374 = uniform_struct_0_4_.f7_.w;
    let _e380 = uniform_struct_0_4_.f8_.x;
    let _e384 = uniform_struct_0_4_.f8_.y;
    let _e388 = uniform_struct_0_4_.f8_.z;
    let _e395 = ((_e194 - length(vec3<f32>((_e380 - ((_e262 + (((_e237 * _e245) + (_e241 * _e250)) + (_e233 * _e256))) / (_e284 + (((_e237 * _e267) + (_e241 * _e272)) + (_e233 * _e278))))), (_e384 - ((_e307 + (((_e237 * _e290) + (_e241 * _e295)) + (_e233 * _e301))) / (_e329 + (((_e237 * _e312) + (_e241 * _e317)) + (_e233 * _e323))))), (_e388 - ((_e352 + (((_e237 * _e335) + (_e241 * _e340)) + (_e233 * _e346))) / (_e374 + (((_e237 * _e357) + (_e241 * _e362)) + (_e233 * _e368)))))))) / _e210);
    let _e398 = exp((_e395 * -(_e395)));
    let _e401 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(_e213, uv.y), 0f);
    let _e412 = clamp((uv.x - (_e200 * 1.5f)), 0f, 1f);
    let _e415 = clamp(_e216, 0f, 1f);
    let _e426 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e412 * _e14)), _e22)), bitcast<i32>(min(u32((_e415 * _e17)), _e28))), 0i);
    let _e430 = ((_e412 * 2f) + -1f);
    let _e434 = (1f - (_e415 * 2f));
    let _e438 = uniform_struct_0_4_.f4_.x;
    let _e443 = uniform_struct_0_4_.f5_.x;
    let _e449 = uniform_struct_0_4_.f6_.x;
    let _e455 = uniform_struct_0_4_.f7_.x;
    let _e460 = uniform_struct_0_4_.f4_.w;
    let _e465 = uniform_struct_0_4_.f5_.w;
    let _e471 = uniform_struct_0_4_.f6_.w;
    let _e477 = uniform_struct_0_4_.f7_.w;
    let _e483 = uniform_struct_0_4_.f4_.y;
    let _e488 = uniform_struct_0_4_.f5_.y;
    let _e494 = uniform_struct_0_4_.f6_.y;
    let _e500 = uniform_struct_0_4_.f7_.y;
    let _e505 = uniform_struct_0_4_.f4_.w;
    let _e510 = uniform_struct_0_4_.f5_.w;
    let _e516 = uniform_struct_0_4_.f6_.w;
    let _e522 = uniform_struct_0_4_.f7_.w;
    let _e528 = uniform_struct_0_4_.f4_.z;
    let _e533 = uniform_struct_0_4_.f5_.z;
    let _e539 = uniform_struct_0_4_.f6_.z;
    let _e545 = uniform_struct_0_4_.f7_.z;
    let _e550 = uniform_struct_0_4_.f4_.w;
    let _e555 = uniform_struct_0_4_.f5_.w;
    let _e561 = uniform_struct_0_4_.f6_.w;
    let _e567 = uniform_struct_0_4_.f7_.w;
    let _e573 = uniform_struct_0_4_.f8_.x;
    let _e577 = uniform_struct_0_4_.f8_.y;
    let _e581 = uniform_struct_0_4_.f8_.z;
    let _e588 = ((_e194 - length(vec3<f32>((_e573 - ((_e455 + (((_e430 * _e438) + (_e434 * _e443)) + (_e426 * _e449))) / (_e477 + (((_e430 * _e460) + (_e434 * _e465)) + (_e426 * _e471))))), (_e577 - ((_e500 + (((_e430 * _e483) + (_e434 * _e488)) + (_e426 * _e494))) / (_e522 + (((_e430 * _e505) + (_e434 * _e510)) + (_e426 * _e516))))), (_e581 - ((_e545 + (((_e430 * _e528) + (_e434 * _e533)) + (_e426 * _e539))) / (_e567 + (((_e430 * _e550) + (_e434 * _e555)) + (_e426 * _e561)))))))) / _e210);
    let _e591 = exp((_e588 * -(_e588)));
    let _e595 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>((uv.x - _e212), uv.y), 0f);
    let _e602 = (uv.x + (_e200 * 0f));
    let _e604 = (_e205 * 1.5f);
    let _e605 = (uv.y + _e604);
    let _e608 = clamp(_e602, 0f, 1f);
    let _e611 = clamp(_e605, 0f, 1f);
    let _e622 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e608 * _e14)), _e22)), bitcast<i32>(min(u32((_e611 * _e17)), _e28))), 0i);
    let _e626 = ((_e608 * 2f) + -1f);
    let _e630 = (1f - (_e611 * 2f));
    let _e634 = uniform_struct_0_4_.f4_.x;
    let _e639 = uniform_struct_0_4_.f5_.x;
    let _e645 = uniform_struct_0_4_.f6_.x;
    let _e651 = uniform_struct_0_4_.f7_.x;
    let _e656 = uniform_struct_0_4_.f4_.w;
    let _e661 = uniform_struct_0_4_.f5_.w;
    let _e667 = uniform_struct_0_4_.f6_.w;
    let _e673 = uniform_struct_0_4_.f7_.w;
    let _e679 = uniform_struct_0_4_.f4_.y;
    let _e684 = uniform_struct_0_4_.f5_.y;
    let _e690 = uniform_struct_0_4_.f6_.y;
    let _e696 = uniform_struct_0_4_.f7_.y;
    let _e701 = uniform_struct_0_4_.f4_.w;
    let _e706 = uniform_struct_0_4_.f5_.w;
    let _e712 = uniform_struct_0_4_.f6_.w;
    let _e718 = uniform_struct_0_4_.f7_.w;
    let _e724 = uniform_struct_0_4_.f4_.z;
    let _e729 = uniform_struct_0_4_.f5_.z;
    let _e735 = uniform_struct_0_4_.f6_.z;
    let _e741 = uniform_struct_0_4_.f7_.z;
    let _e746 = uniform_struct_0_4_.f4_.w;
    let _e751 = uniform_struct_0_4_.f5_.w;
    let _e757 = uniform_struct_0_4_.f6_.w;
    let _e763 = uniform_struct_0_4_.f7_.w;
    let _e769 = uniform_struct_0_4_.f8_.x;
    let _e773 = uniform_struct_0_4_.f8_.y;
    let _e777 = uniform_struct_0_4_.f8_.z;
    let _e784 = ((_e194 - length(vec3<f32>((_e769 - ((_e651 + (((_e626 * _e634) + (_e630 * _e639)) + (_e622 * _e645))) / (_e673 + (((_e626 * _e656) + (_e630 * _e661)) + (_e622 * _e667))))), (_e773 - ((_e696 + (((_e626 * _e679) + (_e630 * _e684)) + (_e622 * _e690))) / (_e718 + (((_e626 * _e701) + (_e630 * _e706)) + (_e622 * _e712))))), (_e777 - ((_e741 + (((_e626 * _e724) + (_e630 * _e729)) + (_e622 * _e735))) / (_e763 + (((_e626 * _e746) + (_e630 * _e751)) + (_e622 * _e757)))))))) / _e210);
    let _e787 = exp((_e784 * -(_e784)));
    let _e790 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, _e605), 0f);
    let _e800 = clamp(_e602, 0f, 1f);
    let _e803 = clamp((uv.y - (_e205 * 1.5f)), 0f, 1f);
    let _e814 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e800 * _e14)), _e22)), bitcast<i32>(min(u32((_e803 * _e17)), _e28))), 0i);
    let _e818 = ((_e800 * 2f) + -1f);
    let _e822 = (1f - (_e803 * 2f));
    let _e826 = uniform_struct_0_4_.f4_.x;
    let _e831 = uniform_struct_0_4_.f5_.x;
    let _e837 = uniform_struct_0_4_.f6_.x;
    let _e843 = uniform_struct_0_4_.f7_.x;
    let _e848 = uniform_struct_0_4_.f4_.w;
    let _e853 = uniform_struct_0_4_.f5_.w;
    let _e859 = uniform_struct_0_4_.f6_.w;
    let _e865 = uniform_struct_0_4_.f7_.w;
    let _e871 = uniform_struct_0_4_.f4_.y;
    let _e876 = uniform_struct_0_4_.f5_.y;
    let _e882 = uniform_struct_0_4_.f6_.y;
    let _e888 = uniform_struct_0_4_.f7_.y;
    let _e893 = uniform_struct_0_4_.f4_.w;
    let _e898 = uniform_struct_0_4_.f5_.w;
    let _e904 = uniform_struct_0_4_.f6_.w;
    let _e910 = uniform_struct_0_4_.f7_.w;
    let _e916 = uniform_struct_0_4_.f4_.z;
    let _e921 = uniform_struct_0_4_.f5_.z;
    let _e927 = uniform_struct_0_4_.f6_.z;
    let _e933 = uniform_struct_0_4_.f7_.z;
    let _e938 = uniform_struct_0_4_.f4_.w;
    let _e943 = uniform_struct_0_4_.f5_.w;
    let _e949 = uniform_struct_0_4_.f6_.w;
    let _e955 = uniform_struct_0_4_.f7_.w;
    let _e961 = uniform_struct_0_4_.f8_.x;
    let _e965 = uniform_struct_0_4_.f8_.y;
    let _e969 = uniform_struct_0_4_.f8_.z;
    let _e976 = ((_e194 - length(vec3<f32>((_e961 - ((_e843 + (((_e818 * _e826) + (_e822 * _e831)) + (_e814 * _e837))) / (_e865 + (((_e818 * _e848) + (_e822 * _e853)) + (_e814 * _e859))))), (_e965 - ((_e888 + (((_e818 * _e871) + (_e822 * _e876)) + (_e814 * _e882))) / (_e910 + (((_e818 * _e893) + (_e822 * _e898)) + (_e814 * _e904))))), (_e969 - ((_e933 + (((_e818 * _e916) + (_e822 * _e921)) + (_e814 * _e927))) / (_e955 + (((_e818 * _e938) + (_e822 * _e943)) + (_e814 * _e949)))))))) / _e210);
    let _e979 = exp((_e976 * -(_e976)));
    let _e983 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, (uv.y - _e604)), 0f);
    let _e988 = (((((_e8.x + (_e398 * _e401.x)) + (_e591 * _e595.x)) + (_e787 * _e790.x)) + (_e979 * _e983.x)) / ((((_e398 + 1f) + _e591) + _e787) + _e979));
    fragment_out = vec4<f32>(_e988, _e988, _e988, 1f);
    let _e993 = fragment_out;
    return _e993;
}
