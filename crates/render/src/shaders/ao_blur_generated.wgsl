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
    let _e96 = uniform_struct_0_4_.f4_.y;
    let _e101 = uniform_struct_0_4_.f5_.y;
    let _e107 = uniform_struct_0_4_.f6_.y;
    let _e113 = uniform_struct_0_4_.f7_.y;
    let _e118 = uniform_struct_0_4_.f4_.w;
    let _e123 = uniform_struct_0_4_.f5_.w;
    let _e129 = uniform_struct_0_4_.f6_.w;
    let _e135 = uniform_struct_0_4_.f7_.w;
    let _e143 = uniform_struct_0_4_.f4_.z;
    let _e148 = uniform_struct_0_4_.f5_.z;
    let _e154 = uniform_struct_0_4_.f6_.z;
    let _e160 = uniform_struct_0_4_.f7_.z;
    let _e165 = uniform_struct_0_4_.f4_.w;
    let _e170 = uniform_struct_0_4_.f5_.w;
    let _e176 = uniform_struct_0_4_.f6_.w;
    let _e182 = uniform_struct_0_4_.f7_.w;
    let _e190 = uniform_struct_0_4_.f8_.x;
    let _e194 = uniform_struct_0_4_.f8_.y;
    let _e198 = uniform_struct_0_4_.f8_.z;
    let _e203 = length(vec3<f32>((_e190 - ((_e66 + (((_e41 * _e49) + (_e45 * _e54)) + (_e37 * _e60))) / max((_e88 + (((_e41 * _e71) + (_e45 * _e76)) + (_e37 * _e82))), 0.000001f))), (_e194 - ((_e113 + (((_e41 * _e96) + (_e45 * _e101)) + (_e37 * _e107))) / max((_e135 + (((_e41 * _e118) + (_e45 * _e123)) + (_e37 * _e129))), 0.000001f))), (_e198 - ((_e160 + (((_e41 * _e143) + (_e45 * _e148)) + (_e37 * _e154))) / max((_e182 + (((_e41 * _e165) + (_e45 * _e170)) + (_e37 * _e176))), 0.000001f)))));
    let _e205 = textureDimensions(tex_0_0_);
    let _e210 = (1f / f32(bitcast<i32>(_e205.x)));
    let _e212 = textureDimensions(tex_0_0_);
    let _e217 = (1f / f32(bitcast<i32>(_e212.y)));
    let _e220 = uniform_struct_0_3_.f3_;
    let _e222 = max(_e220, 0.000001f);
    let _e224 = (_e210 * 1.5f);
    let _e225 = (uv.x + _e224);
    let _e228 = (uv.y + (_e217 * 0f));
    let _e231 = clamp(_e225, 0f, 1f);
    let _e234 = clamp(_e228, 0f, 1f);
    let _e246 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e231 * _e14)), _e24)), bitcast<i32>(min(u32((_e234 * _e19)), _e30))), 0i);
    let _e250 = ((_e231 * 2f) + -1f);
    let _e254 = (1f - (_e234 * 2f));
    let _e258 = uniform_struct_0_4_.f4_.x;
    let _e263 = uniform_struct_0_4_.f5_.x;
    let _e269 = uniform_struct_0_4_.f6_.x;
    let _e275 = uniform_struct_0_4_.f7_.x;
    let _e280 = uniform_struct_0_4_.f4_.w;
    let _e285 = uniform_struct_0_4_.f5_.w;
    let _e291 = uniform_struct_0_4_.f6_.w;
    let _e297 = uniform_struct_0_4_.f7_.w;
    let _e305 = uniform_struct_0_4_.f4_.y;
    let _e310 = uniform_struct_0_4_.f5_.y;
    let _e316 = uniform_struct_0_4_.f6_.y;
    let _e322 = uniform_struct_0_4_.f7_.y;
    let _e327 = uniform_struct_0_4_.f4_.w;
    let _e332 = uniform_struct_0_4_.f5_.w;
    let _e338 = uniform_struct_0_4_.f6_.w;
    let _e344 = uniform_struct_0_4_.f7_.w;
    let _e352 = uniform_struct_0_4_.f4_.z;
    let _e357 = uniform_struct_0_4_.f5_.z;
    let _e363 = uniform_struct_0_4_.f6_.z;
    let _e369 = uniform_struct_0_4_.f7_.z;
    let _e374 = uniform_struct_0_4_.f4_.w;
    let _e379 = uniform_struct_0_4_.f5_.w;
    let _e385 = uniform_struct_0_4_.f6_.w;
    let _e391 = uniform_struct_0_4_.f7_.w;
    let _e399 = uniform_struct_0_4_.f8_.x;
    let _e403 = uniform_struct_0_4_.f8_.y;
    let _e407 = uniform_struct_0_4_.f8_.z;
    let _e414 = ((_e203 - length(vec3<f32>((_e399 - ((_e275 + (((_e250 * _e258) + (_e254 * _e263)) + (_e246 * _e269))) / max((_e297 + (((_e250 * _e280) + (_e254 * _e285)) + (_e246 * _e291))), 0.000001f))), (_e403 - ((_e322 + (((_e250 * _e305) + (_e254 * _e310)) + (_e246 * _e316))) / max((_e344 + (((_e250 * _e327) + (_e254 * _e332)) + (_e246 * _e338))), 0.000001f))), (_e407 - ((_e369 + (((_e250 * _e352) + (_e254 * _e357)) + (_e246 * _e363))) / max((_e391 + (((_e250 * _e374) + (_e254 * _e379)) + (_e246 * _e385))), 0.000001f)))))) / _e222);
    let _e417 = exp((_e414 * -(_e414)));
    let _e422 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(_e225, uv.y), 0f);
    let _e433 = clamp((uv.x - (_e210 * 1.5f)), 0f, 1f);
    let _e436 = clamp(_e228, 0f, 1f);
    let _e448 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e433 * _e14)), _e24)), bitcast<i32>(min(u32((_e436 * _e19)), _e30))), 0i);
    let _e452 = ((_e433 * 2f) + -1f);
    let _e456 = (1f - (_e436 * 2f));
    let _e460 = uniform_struct_0_4_.f4_.x;
    let _e465 = uniform_struct_0_4_.f5_.x;
    let _e471 = uniform_struct_0_4_.f6_.x;
    let _e477 = uniform_struct_0_4_.f7_.x;
    let _e482 = uniform_struct_0_4_.f4_.w;
    let _e487 = uniform_struct_0_4_.f5_.w;
    let _e493 = uniform_struct_0_4_.f6_.w;
    let _e499 = uniform_struct_0_4_.f7_.w;
    let _e507 = uniform_struct_0_4_.f4_.y;
    let _e512 = uniform_struct_0_4_.f5_.y;
    let _e518 = uniform_struct_0_4_.f6_.y;
    let _e524 = uniform_struct_0_4_.f7_.y;
    let _e529 = uniform_struct_0_4_.f4_.w;
    let _e534 = uniform_struct_0_4_.f5_.w;
    let _e540 = uniform_struct_0_4_.f6_.w;
    let _e546 = uniform_struct_0_4_.f7_.w;
    let _e554 = uniform_struct_0_4_.f4_.z;
    let _e559 = uniform_struct_0_4_.f5_.z;
    let _e565 = uniform_struct_0_4_.f6_.z;
    let _e571 = uniform_struct_0_4_.f7_.z;
    let _e576 = uniform_struct_0_4_.f4_.w;
    let _e581 = uniform_struct_0_4_.f5_.w;
    let _e587 = uniform_struct_0_4_.f6_.w;
    let _e593 = uniform_struct_0_4_.f7_.w;
    let _e601 = uniform_struct_0_4_.f8_.x;
    let _e605 = uniform_struct_0_4_.f8_.y;
    let _e609 = uniform_struct_0_4_.f8_.z;
    let _e616 = ((_e203 - length(vec3<f32>((_e601 - ((_e477 + (((_e452 * _e460) + (_e456 * _e465)) + (_e448 * _e471))) / max((_e499 + (((_e452 * _e482) + (_e456 * _e487)) + (_e448 * _e493))), 0.000001f))), (_e605 - ((_e524 + (((_e452 * _e507) + (_e456 * _e512)) + (_e448 * _e518))) / max((_e546 + (((_e452 * _e529) + (_e456 * _e534)) + (_e448 * _e540))), 0.000001f))), (_e609 - ((_e571 + (((_e452 * _e554) + (_e456 * _e559)) + (_e448 * _e565))) / max((_e593 + (((_e452 * _e576) + (_e456 * _e581)) + (_e448 * _e587))), 0.000001f)))))) / _e222);
    let _e619 = exp((_e616 * -(_e616)));
    let _e625 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>((uv.x - _e224), uv.y), 0f);
    let _e632 = (uv.x + (_e210 * 0f));
    let _e634 = (_e217 * 1.5f);
    let _e635 = (uv.y + _e634);
    let _e638 = clamp(_e632, 0f, 1f);
    let _e641 = clamp(_e635, 0f, 1f);
    let _e653 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e638 * _e14)), _e24)), bitcast<i32>(min(u32((_e641 * _e19)), _e30))), 0i);
    let _e657 = ((_e638 * 2f) + -1f);
    let _e661 = (1f - (_e641 * 2f));
    let _e665 = uniform_struct_0_4_.f4_.x;
    let _e670 = uniform_struct_0_4_.f5_.x;
    let _e676 = uniform_struct_0_4_.f6_.x;
    let _e682 = uniform_struct_0_4_.f7_.x;
    let _e687 = uniform_struct_0_4_.f4_.w;
    let _e692 = uniform_struct_0_4_.f5_.w;
    let _e698 = uniform_struct_0_4_.f6_.w;
    let _e704 = uniform_struct_0_4_.f7_.w;
    let _e712 = uniform_struct_0_4_.f4_.y;
    let _e717 = uniform_struct_0_4_.f5_.y;
    let _e723 = uniform_struct_0_4_.f6_.y;
    let _e729 = uniform_struct_0_4_.f7_.y;
    let _e734 = uniform_struct_0_4_.f4_.w;
    let _e739 = uniform_struct_0_4_.f5_.w;
    let _e745 = uniform_struct_0_4_.f6_.w;
    let _e751 = uniform_struct_0_4_.f7_.w;
    let _e759 = uniform_struct_0_4_.f4_.z;
    let _e764 = uniform_struct_0_4_.f5_.z;
    let _e770 = uniform_struct_0_4_.f6_.z;
    let _e776 = uniform_struct_0_4_.f7_.z;
    let _e781 = uniform_struct_0_4_.f4_.w;
    let _e786 = uniform_struct_0_4_.f5_.w;
    let _e792 = uniform_struct_0_4_.f6_.w;
    let _e798 = uniform_struct_0_4_.f7_.w;
    let _e806 = uniform_struct_0_4_.f8_.x;
    let _e810 = uniform_struct_0_4_.f8_.y;
    let _e814 = uniform_struct_0_4_.f8_.z;
    let _e821 = ((_e203 - length(vec3<f32>((_e806 - ((_e682 + (((_e657 * _e665) + (_e661 * _e670)) + (_e653 * _e676))) / max((_e704 + (((_e657 * _e687) + (_e661 * _e692)) + (_e653 * _e698))), 0.000001f))), (_e810 - ((_e729 + (((_e657 * _e712) + (_e661 * _e717)) + (_e653 * _e723))) / max((_e751 + (((_e657 * _e734) + (_e661 * _e739)) + (_e653 * _e745))), 0.000001f))), (_e814 - ((_e776 + (((_e657 * _e759) + (_e661 * _e764)) + (_e653 * _e770))) / max((_e798 + (((_e657 * _e781) + (_e661 * _e786)) + (_e653 * _e792))), 0.000001f)))))) / _e222);
    let _e824 = exp((_e821 * -(_e821)));
    let _e829 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, _e635), 0f);
    let _e839 = clamp(_e632, 0f, 1f);
    let _e842 = clamp((uv.y - (_e217 * 1.5f)), 0f, 1f);
    let _e854 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(min(u32((_e839 * _e14)), _e24)), bitcast<i32>(min(u32((_e842 * _e19)), _e30))), 0i);
    let _e858 = ((_e839 * 2f) + -1f);
    let _e862 = (1f - (_e842 * 2f));
    let _e866 = uniform_struct_0_4_.f4_.x;
    let _e871 = uniform_struct_0_4_.f5_.x;
    let _e877 = uniform_struct_0_4_.f6_.x;
    let _e883 = uniform_struct_0_4_.f7_.x;
    let _e888 = uniform_struct_0_4_.f4_.w;
    let _e893 = uniform_struct_0_4_.f5_.w;
    let _e899 = uniform_struct_0_4_.f6_.w;
    let _e905 = uniform_struct_0_4_.f7_.w;
    let _e913 = uniform_struct_0_4_.f4_.y;
    let _e918 = uniform_struct_0_4_.f5_.y;
    let _e924 = uniform_struct_0_4_.f6_.y;
    let _e930 = uniform_struct_0_4_.f7_.y;
    let _e935 = uniform_struct_0_4_.f4_.w;
    let _e940 = uniform_struct_0_4_.f5_.w;
    let _e946 = uniform_struct_0_4_.f6_.w;
    let _e952 = uniform_struct_0_4_.f7_.w;
    let _e960 = uniform_struct_0_4_.f4_.z;
    let _e965 = uniform_struct_0_4_.f5_.z;
    let _e971 = uniform_struct_0_4_.f6_.z;
    let _e977 = uniform_struct_0_4_.f7_.z;
    let _e982 = uniform_struct_0_4_.f4_.w;
    let _e987 = uniform_struct_0_4_.f5_.w;
    let _e993 = uniform_struct_0_4_.f6_.w;
    let _e999 = uniform_struct_0_4_.f7_.w;
    let _e1007 = uniform_struct_0_4_.f8_.x;
    let _e1011 = uniform_struct_0_4_.f8_.y;
    let _e1015 = uniform_struct_0_4_.f8_.z;
    let _e1022 = ((_e203 - length(vec3<f32>((_e1007 - ((_e883 + (((_e858 * _e866) + (_e862 * _e871)) + (_e854 * _e877))) / max((_e905 + (((_e858 * _e888) + (_e862 * _e893)) + (_e854 * _e899))), 0.000001f))), (_e1011 - ((_e930 + (((_e858 * _e913) + (_e862 * _e918)) + (_e854 * _e924))) / max((_e952 + (((_e858 * _e935) + (_e862 * _e940)) + (_e854 * _e946))), 0.000001f))), (_e1015 - ((_e977 + (((_e858 * _e960) + (_e862 * _e965)) + (_e854 * _e971))) / max((_e999 + (((_e858 * _e982) + (_e862 * _e987)) + (_e854 * _e993))), 0.000001f)))))) / _e222);
    let _e1025 = exp((_e1022 * -(_e1022)));
    let _e1031 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, (uv.y - _e634)), 0f);
    let _e1036 = (((((_e8.x + (_e417 * _e422.x)) + (_e619 * _e625.x)) + (_e824 * _e829.x)) + (_e1025 * _e1031.x)) / ((((_e417 + 1f) + _e619) + _e824) + _e1025));
    fragment_out = vec4<f32>(_e1036, _e1036, _e1036, 1f);
    let _e1041 = fragment_out;
    return _e1041;
}
