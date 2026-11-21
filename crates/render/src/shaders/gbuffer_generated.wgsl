struct FragmentOut {
    @location(0) o0_: vec4<f32>,
    @location(1) o1_: vec2<f32>,
    @location(2) o2_: u32,
    @location(3) o3_: vec2<f32>,
    @location(4) o4_: vec4<f32>,
    @location(5) o5_: vec2<f32>,
    @location(6) o6_: u32,
}

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

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) o0_: vec3<f32>,
    @location(1) o1_: vec3<f32>,
    @location(2) o2_: vec2<f32>,
    @location(3) o3_: vec3<f32>,
    @location(4) @interpolate(flat) o4_: u32,
    @location(5) o5_: vec4<f32>,
}

@group(0) @binding(2) 
var<storage> storage_0_2_: array<f32>;
@group(0) @binding(0) 
var<uniform> uniform_struct_0_0_: UniformStruct0_;
@group(0) @binding(1) 
var<storage> storage_0_1_: array<u32>;
@group(1) @binding(0) 
var<storage> storage_1_0_: array<u32>;

@fragment 
fn fs_main(@builtin(position) position: vec4<f32>, @location(0) in0_: vec3<f32>, @location(1) in1_: vec3<f32>, @location(4) @interpolate(flat) in4_: u32, @location(5) in5_: vec4<f32>) -> FragmentOut {
    var fragment_out: FragmentOut;
    var encoded_y_u002e_sroa_u002e_0_u002e_0_: f32;
    var encoded_x_u002e_sroa_u002e_0_u002e_0_: f32;
    var previous_depth_u002e_sroa_u002e_0_u002e_0_: u32;
    var motion_y_u002e_sroa_u002e_0_u002e_0_: f32;
    var motion_x_u002e_sroa_u002e_0_u002e_0_: f32;

    let _e6 = (in4_ * 80u);
    let _e11 = storage_0_2_[(_e6 | 4u)];
    let _e16 = storage_0_2_[(_e6 | 5u)];
    let _e21 = storage_0_2_[(_e6 | 6u)];
    let _e26 = storage_0_2_[(_e6 | 7u)];
    let _e31 = storage_0_2_[(_e6 | 9u)];
    let _e36 = storage_0_2_[(_e6 | 10u)];
    let _e41 = storage_0_2_[(_e6 | 2u)];
    let _e46 = storage_0_2_[(_e6 + 48u)];
    let _e59 = uniform_struct_0_0_.f0_.x;
    let _e64 = uniform_struct_0_0_.f1_.x;
    let _e70 = uniform_struct_0_0_.f2_.x;
    let _e76 = uniform_struct_0_0_.f3_.x;
    let _e81 = uniform_struct_0_0_.f0_.y;
    let _e86 = uniform_struct_0_0_.f1_.y;
    let _e92 = uniform_struct_0_0_.f2_.y;
    let _e98 = uniform_struct_0_0_.f3_.y;
    let _e103 = uniform_struct_0_0_.f0_.w;
    let _e108 = uniform_struct_0_0_.f1_.w;
    let _e114 = uniform_struct_0_0_.f2_.w;
    let _e120 = uniform_struct_0_0_.f3_.w;
    let _e121 = (_e120 + (((in0_.x * _e103) + (in0_.y * _e108)) + (in0_.z * _e114)));
    let _e125 = uniform_struct_0_0_.f9_.x;
    let _e130 = uniform_struct_0_0_.f10_.x;
    let _e136 = uniform_struct_0_0_.f11_.x;
    let _e142 = uniform_struct_0_0_.f12_.x;
    let _e147 = uniform_struct_0_0_.f9_.y;
    let _e152 = uniform_struct_0_0_.f10_.y;
    let _e158 = uniform_struct_0_0_.f11_.y;
    let _e164 = uniform_struct_0_0_.f12_.y;
    let _e169 = uniform_struct_0_0_.f9_.z;
    let _e174 = uniform_struct_0_0_.f10_.z;
    let _e180 = uniform_struct_0_0_.f11_.z;
    let _e186 = uniform_struct_0_0_.f12_.z;
    let _e191 = uniform_struct_0_0_.f9_.w;
    let _e196 = uniform_struct_0_0_.f10_.w;
    let _e202 = uniform_struct_0_0_.f11_.w;
    let _e208 = uniform_struct_0_0_.f12_.w;
    let _e209 = (_e208 + (((in5_.x * _e191) + (in5_.y * _e196)) + (in5_.z * _e202)));
    let _e216 = (1f / ((abs(in1_.x) + abs(in1_.y)) + abs(in1_.z)));
    let _e217 = (in1_.x * _e216);
    let _e218 = (in1_.y * _e216);
    if (in1_.z < 0f) {
        let _e224 = sign(_e217);
        let _e233 = sign(_e218);
        encoded_y_u002e_sroa_u002e_0_u002e_0_ = ((1f - abs(_e217)) * select(_e233, 1f, (_e233 == 0f)));
        encoded_x_u002e_sroa_u002e_0_u002e_0_ = ((1f - abs(_e218)) * select(_e224, 1f, (_e224 == 0f)));
    } else {
        encoded_y_u002e_sroa_u002e_0_u002e_0_ = _e218;
        encoded_x_u002e_sroa_u002e_0_u002e_0_ = _e217;
    }
    let _e244 = encoded_y_u002e_sroa_u002e_0_u002e_0_;
    let _e246 = encoded_x_u002e_sroa_u002e_0_u002e_0_;
    if ((_e121 > 0f) & (_e209 > 0f)) {
    } else {
        previous_depth_u002e_sroa_u002e_0_u002e_0_ = 65535u;
        motion_y_u002e_sroa_u002e_0_u002e_0_ = 0f;
        motion_x_u002e_sroa_u002e_0_u002e_0_ = 0f;
        let _e259 = previous_depth_u002e_sroa_u002e_0_u002e_0_;
        let _e261 = motion_y_u002e_sroa_u002e_0_u002e_0_;
        let _e263 = motion_x_u002e_sroa_u002e_0_u002e_0_;
        fragment_out.o0_ = vec4<f32>(_e11, _e16, _e21, _e26);
        fragment_out.o1_ = vec2<f32>(_e246, _e244);
        fragment_out.o2_ = in4_;
        fragment_out.o3_ = vec2<f32>(in0_.x, in0_.y);
        fragment_out.o4_ = vec4<f32>(_e31, _e41, _e36, _e46);
        fragment_out.o5_ = vec2<f32>(_e263, _e261);
        fragment_out.o6_ = _e259;
        let _e284 = fragment_out;
        return _e284;
    }
    if (in5_.w > 0.5f) {
    } else {
        previous_depth_u002e_sroa_u002e_0_u002e_0_ = 65535u;
        motion_y_u002e_sroa_u002e_0_u002e_0_ = 0f;
        motion_x_u002e_sroa_u002e_0_u002e_0_ = 0f;
        let _e295 = previous_depth_u002e_sroa_u002e_0_u002e_0_;
        let _e297 = motion_y_u002e_sroa_u002e_0_u002e_0_;
        let _e299 = motion_x_u002e_sroa_u002e_0_u002e_0_;
        fragment_out.o0_ = vec4<f32>(_e11, _e16, _e21, _e26);
        fragment_out.o1_ = vec2<f32>(_e246, _e244);
        fragment_out.o2_ = in4_;
        fragment_out.o3_ = vec2<f32>(in0_.x, in0_.y);
        fragment_out.o4_ = vec4<f32>(_e31, _e41, _e36, _e46);
        fragment_out.o5_ = vec2<f32>(_e299, _e297);
        fragment_out.o6_ = _e295;
        let _e320 = fragment_out;
        return _e320;
    }
    previous_depth_u002e_sroa_u002e_0_u002e_0_ = u32(((clamp(((_e186 + (((in5_.x * _e169) + (in5_.y * _e174)) + (in5_.z * _e180))) / _e209), 0f, 1f) * 65534f) + 0.5f));
    motion_y_u002e_sroa_u002e_0_u002e_0_ = ((0.5f - (((_e98 + (((in0_.x * _e81) + (in0_.y * _e86)) + (in0_.z * _e92))) / _e121) * 0.5f)) + ((((_e164 + (((in5_.x * _e147) + (in5_.y * _e152)) + (in5_.z * _e158))) / _e209) * 0.5f) + -0.5f));
    motion_x_u002e_sroa_u002e_0_u002e_0_ = (((((_e76 + (((in0_.x * _e59) + (in0_.y * _e64)) + (in0_.z * _e70))) / _e121) * 0.5f) + 0.5f) - ((((_e142 + (((in5_.x * _e125) + (in5_.y * _e130)) + (in5_.z * _e136))) / _e209) * 0.5f) + 0.5f));
    let _e356 = previous_depth_u002e_sroa_u002e_0_u002e_0_;
    let _e358 = motion_y_u002e_sroa_u002e_0_u002e_0_;
    let _e360 = motion_x_u002e_sroa_u002e_0_u002e_0_;
    fragment_out.o0_ = vec4<f32>(_e11, _e16, _e21, _e26);
    fragment_out.o1_ = vec2<f32>(_e246, _e244);
    fragment_out.o2_ = in4_;
    fragment_out.o3_ = vec2<f32>(in0_.x, in0_.y);
    fragment_out.o4_ = vec4<f32>(_e31, _e41, _e36, _e46);
    fragment_out.o5_ = vec2<f32>(_e360, _e358);
    fragment_out.o6_ = _e356;
    let _e381 = fragment_out;
    return _e381;
}

@vertex 
fn vs_main(@builtin(instance_index) instance_index: u32, @location(0) in0_1: vec3<f32>, @location(1) in1_1: vec3<f32>, @location(2) in2_: vec2<f32>, @location(3) in3_: vec3<f32>) -> VertexOut {
    var vertex_out: VertexOut;
    var previous_w_u002e_sroa_u002e_0_u002e_0_: f32;

    let _e6 = (instance_index * 52u);
    let _e12 = storage_0_1_[_e6];
    let _e16 = (_e6 + 4u);
    let _e19 = storage_0_1_[_e16];
    let _e24 = (_e6 + 8u);
    let _e27 = storage_0_1_[_e24];
    let _e32 = (_e6 + 12u);
    let _e35 = storage_0_1_[_e32];
    let _e42 = (_e6 | 1u);
    let _e45 = storage_0_1_[_e42];
    let _e49 = (_e6 + 5u);
    let _e52 = storage_0_1_[_e49];
    let _e57 = (_e6 + 9u);
    let _e60 = storage_0_1_[_e57];
    let _e65 = (_e6 + 13u);
    let _e68 = storage_0_1_[_e65];
    let _e75 = (_e6 | 2u);
    let _e78 = storage_0_1_[_e75];
    let _e82 = (_e6 + 6u);
    let _e85 = storage_0_1_[_e82];
    let _e90 = (_e6 + 10u);
    let _e93 = storage_0_1_[_e90];
    let _e98 = (_e6 + 14u);
    let _e101 = storage_0_1_[_e98];
    vertex_out.o0_ = vec3<f32>(((((in0_1.x * bitcast<f32>(_e12)) + (in0_1.y * bitcast<f32>(_e19))) + (in0_1.z * bitcast<f32>(_e27))) + bitcast<f32>(_e35)), ((((in0_1.x * bitcast<f32>(_e45)) + (in0_1.y * bitcast<f32>(_e52))) + (in0_1.z * bitcast<f32>(_e60))) + bitcast<f32>(_e68)), ((((in0_1.x * bitcast<f32>(_e78)) + (in0_1.y * bitcast<f32>(_e85))) + (in0_1.z * bitcast<f32>(_e93))) + bitcast<f32>(_e101)));
    let _e114 = storage_0_1_[(_e6 + 16u)];
    let _e121 = storage_0_1_[(_e6 + 20u)];
    let _e129 = storage_0_1_[(_e6 + 24u)];
    let _e137 = storage_0_1_[(_e6 + 28u)];
    let _e141 = ((((in1_1.x * bitcast<f32>(_e114)) + (in1_1.y * bitcast<f32>(_e121))) + (in1_1.z * bitcast<f32>(_e129))) + (bitcast<f32>(_e137) * 0f));
    let _e149 = storage_0_1_[(_e6 + 17u)];
    let _e156 = storage_0_1_[(_e6 + 21u)];
    let _e164 = storage_0_1_[(_e6 + 25u)];
    let _e172 = storage_0_1_[(_e6 + 29u)];
    let _e176 = ((((in1_1.x * bitcast<f32>(_e149)) + (in1_1.y * bitcast<f32>(_e156))) + (in1_1.z * bitcast<f32>(_e164))) + (bitcast<f32>(_e172) * 0f));
    let _e184 = storage_0_1_[(_e6 + 18u)];
    let _e191 = storage_0_1_[(_e6 + 22u)];
    let _e199 = storage_0_1_[(_e6 + 26u)];
    let _e207 = storage_0_1_[(_e6 + 30u)];
    let _e211 = ((((in1_1.x * bitcast<f32>(_e184)) + (in1_1.y * bitcast<f32>(_e191))) + (in1_1.z * bitcast<f32>(_e199))) + (bitcast<f32>(_e207) * 0f));
    vertex_out.o1_ = vec3<f32>(normalize(vec3<f32>(_e141, _e176, _e211)).x, normalize(vec3<f32>(_e141, _e176, _e211)).y, normalize(vec3<f32>(_e141, _e176, _e211)).z);
    vertex_out.o2_ = vec2<f32>(in2_.x, in2_.y);
    let _e234 = storage_0_1_[_e6];
    let _e239 = storage_0_1_[_e16];
    let _e245 = storage_0_1_[_e24];
    let _e251 = storage_0_1_[_e32];
    let _e255 = ((((in3_.x * bitcast<f32>(_e234)) + (in3_.y * bitcast<f32>(_e239))) + (in3_.z * bitcast<f32>(_e245))) + (bitcast<f32>(_e251) * 0f));
    let _e261 = storage_0_1_[_e42];
    let _e266 = storage_0_1_[_e49];
    let _e272 = storage_0_1_[_e57];
    let _e278 = storage_0_1_[_e65];
    let _e282 = ((((in3_.x * bitcast<f32>(_e261)) + (in3_.y * bitcast<f32>(_e266))) + (in3_.z * bitcast<f32>(_e272))) + (bitcast<f32>(_e278) * 0f));
    let _e288 = storage_0_1_[_e75];
    let _e293 = storage_0_1_[_e82];
    let _e299 = storage_0_1_[_e90];
    let _e305 = storage_0_1_[_e98];
    let _e309 = ((((in3_.x * bitcast<f32>(_e288)) + (in3_.y * bitcast<f32>(_e293))) + (in3_.z * bitcast<f32>(_e299))) + (bitcast<f32>(_e305) * 0f));
    vertex_out.o3_ = vec3<f32>(normalize(vec3<f32>(_e255, _e282, _e309)).x, normalize(vec3<f32>(_e255, _e282, _e309)).y, normalize(vec3<f32>(_e255, _e282, _e309)).z);
    let _e326 = storage_0_1_[(_e6 + 48u)];
    vertex_out.o4_ = _e326;
    let _e333 = storage_0_1_[(_e6 + 49u)];
    if (_e333 == 0u) {
        previous_w_u002e_sroa_u002e_0_u002e_0_ = 0f;
        let _e339 = previous_w_u002e_sroa_u002e_0_u002e_0_;
        let _e347 = storage_0_1_[(_e6 + 32u)];
        let _e354 = storage_0_1_[(_e6 + 36u)];
        let _e362 = storage_0_1_[(_e6 + 40u)];
        let _e370 = storage_0_1_[(_e6 + 44u)];
        let _e380 = storage_0_1_[(_e6 + 33u)];
        let _e387 = storage_0_1_[(_e6 + 37u)];
        let _e395 = storage_0_1_[(_e6 + 41u)];
        let _e403 = storage_0_1_[(_e6 + 45u)];
        let _e413 = storage_0_1_[(_e6 + 34u)];
        let _e420 = storage_0_1_[(_e6 + 38u)];
        let _e428 = storage_0_1_[(_e6 + 42u)];
        let _e436 = storage_0_1_[(_e6 + 46u)];
        vertex_out.o5_ = vec4<f32>(((((in0_1.x * bitcast<f32>(_e347)) + (in0_1.y * bitcast<f32>(_e354))) + (in0_1.z * bitcast<f32>(_e362))) + bitcast<f32>(_e370)), ((((in0_1.x * bitcast<f32>(_e380)) + (in0_1.y * bitcast<f32>(_e387))) + (in0_1.z * bitcast<f32>(_e395))) + bitcast<f32>(_e403)), ((((in0_1.x * bitcast<f32>(_e413)) + (in0_1.y * bitcast<f32>(_e420))) + (in0_1.z * bitcast<f32>(_e428))) + bitcast<f32>(_e436)), _e339);
        let _e447 = storage_0_1_[_e6];
        let _e452 = storage_0_1_[_e16];
        let _e458 = storage_0_1_[_e24];
        let _e464 = storage_0_1_[_e32];
        let _e466 = ((((in0_1.x * bitcast<f32>(_e447)) + (in0_1.y * bitcast<f32>(_e452))) + (in0_1.z * bitcast<f32>(_e458))) + bitcast<f32>(_e464));
        let _e472 = storage_0_1_[_e42];
        let _e477 = storage_0_1_[_e49];
        let _e483 = storage_0_1_[_e57];
        let _e489 = storage_0_1_[_e65];
        let _e491 = ((((in0_1.x * bitcast<f32>(_e472)) + (in0_1.y * bitcast<f32>(_e477))) + (in0_1.z * bitcast<f32>(_e483))) + bitcast<f32>(_e489));
        let _e497 = storage_0_1_[_e75];
        let _e502 = storage_0_1_[_e82];
        let _e508 = storage_0_1_[_e90];
        let _e514 = storage_0_1_[_e98];
        let _e516 = ((((in0_1.x * bitcast<f32>(_e497)) + (in0_1.y * bitcast<f32>(_e502))) + (in0_1.z * bitcast<f32>(_e508))) + bitcast<f32>(_e514));
        let _e520 = uniform_struct_0_0_.f0_.x;
        let _e525 = uniform_struct_0_0_.f1_.x;
        let _e531 = uniform_struct_0_0_.f2_.x;
        let _e537 = uniform_struct_0_0_.f3_.x;
        let _e542 = uniform_struct_0_0_.f0_.y;
        let _e547 = uniform_struct_0_0_.f1_.y;
        let _e553 = uniform_struct_0_0_.f2_.y;
        let _e559 = uniform_struct_0_0_.f3_.y;
        let _e564 = uniform_struct_0_0_.f0_.z;
        let _e569 = uniform_struct_0_0_.f1_.z;
        let _e575 = uniform_struct_0_0_.f2_.z;
        let _e581 = uniform_struct_0_0_.f3_.z;
        let _e586 = uniform_struct_0_0_.f0_.w;
        let _e591 = uniform_struct_0_0_.f1_.w;
        let _e597 = uniform_struct_0_0_.f2_.w;
        let _e603 = uniform_struct_0_0_.f3_.w;
        vertex_out.position = vec4<f32>((_e537 + (((_e466 * _e520) + (_e491 * _e525)) + (_e531 * _e516))), (_e559 + (((_e466 * _e542) + (_e491 * _e547)) + (_e516 * _e553))), (_e581 + (((_e466 * _e564) + (_e491 * _e569)) + (_e516 * _e575))), (_e603 + (((_e466 * _e586) + (_e491 * _e591)) + (_e516 * _e597))));
        let _e609 = vertex_out;
        return _e609;
    }
    let _e617 = storage_0_1_[(_e6 + 35u)];
    let _e624 = storage_0_1_[(_e6 + 39u)];
    let _e632 = storage_0_1_[(_e6 + 43u)];
    let _e640 = storage_0_1_[(_e6 + 47u)];
    previous_w_u002e_sroa_u002e_0_u002e_0_ = ((((in0_1.x * bitcast<f32>(_e617)) + (in0_1.y * bitcast<f32>(_e624))) + (in0_1.z * bitcast<f32>(_e632))) + bitcast<f32>(_e640));
    let _e645 = previous_w_u002e_sroa_u002e_0_u002e_0_;
    let _e653 = storage_0_1_[(_e6 + 32u)];
    let _e660 = storage_0_1_[(_e6 + 36u)];
    let _e668 = storage_0_1_[(_e6 + 40u)];
    let _e676 = storage_0_1_[(_e6 + 44u)];
    let _e686 = storage_0_1_[(_e6 + 33u)];
    let _e693 = storage_0_1_[(_e6 + 37u)];
    let _e701 = storage_0_1_[(_e6 + 41u)];
    let _e709 = storage_0_1_[(_e6 + 45u)];
    let _e719 = storage_0_1_[(_e6 + 34u)];
    let _e726 = storage_0_1_[(_e6 + 38u)];
    let _e734 = storage_0_1_[(_e6 + 42u)];
    let _e742 = storage_0_1_[(_e6 + 46u)];
    vertex_out.o5_ = vec4<f32>(((((in0_1.x * bitcast<f32>(_e653)) + (in0_1.y * bitcast<f32>(_e660))) + (in0_1.z * bitcast<f32>(_e668))) + bitcast<f32>(_e676)), ((((in0_1.x * bitcast<f32>(_e686)) + (in0_1.y * bitcast<f32>(_e693))) + (in0_1.z * bitcast<f32>(_e701))) + bitcast<f32>(_e709)), ((((in0_1.x * bitcast<f32>(_e719)) + (in0_1.y * bitcast<f32>(_e726))) + (in0_1.z * bitcast<f32>(_e734))) + bitcast<f32>(_e742)), _e645);
    let _e753 = storage_0_1_[_e6];
    let _e758 = storage_0_1_[_e16];
    let _e764 = storage_0_1_[_e24];
    let _e770 = storage_0_1_[_e32];
    let _e772 = ((((in0_1.x * bitcast<f32>(_e753)) + (in0_1.y * bitcast<f32>(_e758))) + (in0_1.z * bitcast<f32>(_e764))) + bitcast<f32>(_e770));
    let _e778 = storage_0_1_[_e42];
    let _e783 = storage_0_1_[_e49];
    let _e789 = storage_0_1_[_e57];
    let _e795 = storage_0_1_[_e65];
    let _e797 = ((((in0_1.x * bitcast<f32>(_e778)) + (in0_1.y * bitcast<f32>(_e783))) + (in0_1.z * bitcast<f32>(_e789))) + bitcast<f32>(_e795));
    let _e803 = storage_0_1_[_e75];
    let _e808 = storage_0_1_[_e82];
    let _e814 = storage_0_1_[_e90];
    let _e820 = storage_0_1_[_e98];
    let _e822 = ((((in0_1.x * bitcast<f32>(_e803)) + (in0_1.y * bitcast<f32>(_e808))) + (in0_1.z * bitcast<f32>(_e814))) + bitcast<f32>(_e820));
    let _e826 = uniform_struct_0_0_.f0_.x;
    let _e831 = uniform_struct_0_0_.f1_.x;
    let _e837 = uniform_struct_0_0_.f2_.x;
    let _e843 = uniform_struct_0_0_.f3_.x;
    let _e848 = uniform_struct_0_0_.f0_.y;
    let _e853 = uniform_struct_0_0_.f1_.y;
    let _e859 = uniform_struct_0_0_.f2_.y;
    let _e865 = uniform_struct_0_0_.f3_.y;
    let _e870 = uniform_struct_0_0_.f0_.z;
    let _e875 = uniform_struct_0_0_.f1_.z;
    let _e881 = uniform_struct_0_0_.f2_.z;
    let _e887 = uniform_struct_0_0_.f3_.z;
    let _e892 = uniform_struct_0_0_.f0_.w;
    let _e897 = uniform_struct_0_0_.f1_.w;
    let _e903 = uniform_struct_0_0_.f2_.w;
    let _e909 = uniform_struct_0_0_.f3_.w;
    vertex_out.position = vec4<f32>((_e843 + (((_e772 * _e826) + (_e797 * _e831)) + (_e837 * _e822))), (_e865 + (((_e772 * _e848) + (_e797 * _e853)) + (_e822 * _e859))), (_e887 + (((_e772 * _e870) + (_e797 * _e875)) + (_e822 * _e881))), (_e909 + (((_e772 * _e892) + (_e797 * _e897)) + (_e822 * _e903))));
    let _e915 = vertex_out;
    return _e915;
}

@vertex 
fn vs_main_indirect(@builtin(instance_index) instance_index_1: u32, @location(0) in0_2: vec3<f32>, @location(1) in1_2: vec3<f32>, @location(2) in2_1: vec2<f32>, @location(3) in3_1: vec3<f32>) -> VertexOut {
    var vertex_out_1: VertexOut;
    var previous_w_u002e_sroa_u002e_0_u002e_0_1: f32;

    let _e7 = storage_1_0_[instance_index_1];
    let _e9 = (_e7 * 52u);
    let _e15 = storage_0_1_[_e9];
    let _e19 = (_e9 + 4u);
    let _e22 = storage_0_1_[_e19];
    let _e27 = (_e9 + 8u);
    let _e30 = storage_0_1_[_e27];
    let _e35 = (_e9 + 12u);
    let _e38 = storage_0_1_[_e35];
    let _e45 = (_e9 | 1u);
    let _e48 = storage_0_1_[_e45];
    let _e52 = (_e9 + 5u);
    let _e55 = storage_0_1_[_e52];
    let _e60 = (_e9 + 9u);
    let _e63 = storage_0_1_[_e60];
    let _e68 = (_e9 + 13u);
    let _e71 = storage_0_1_[_e68];
    let _e78 = (_e9 | 2u);
    let _e81 = storage_0_1_[_e78];
    let _e85 = (_e9 + 6u);
    let _e88 = storage_0_1_[_e85];
    let _e93 = (_e9 + 10u);
    let _e96 = storage_0_1_[_e93];
    let _e101 = (_e9 + 14u);
    let _e104 = storage_0_1_[_e101];
    vertex_out_1.o0_ = vec3<f32>(((((in0_2.x * bitcast<f32>(_e15)) + (in0_2.y * bitcast<f32>(_e22))) + (in0_2.z * bitcast<f32>(_e30))) + bitcast<f32>(_e38)), ((((in0_2.x * bitcast<f32>(_e48)) + (in0_2.y * bitcast<f32>(_e55))) + (in0_2.z * bitcast<f32>(_e63))) + bitcast<f32>(_e71)), ((((in0_2.x * bitcast<f32>(_e81)) + (in0_2.y * bitcast<f32>(_e88))) + (in0_2.z * bitcast<f32>(_e96))) + bitcast<f32>(_e104)));
    let _e117 = storage_0_1_[(_e9 + 16u)];
    let _e124 = storage_0_1_[(_e9 + 20u)];
    let _e132 = storage_0_1_[(_e9 + 24u)];
    let _e140 = storage_0_1_[(_e9 + 28u)];
    let _e144 = ((((in1_2.x * bitcast<f32>(_e117)) + (in1_2.y * bitcast<f32>(_e124))) + (in1_2.z * bitcast<f32>(_e132))) + (bitcast<f32>(_e140) * 0f));
    let _e152 = storage_0_1_[(_e9 + 17u)];
    let _e159 = storage_0_1_[(_e9 + 21u)];
    let _e167 = storage_0_1_[(_e9 + 25u)];
    let _e175 = storage_0_1_[(_e9 + 29u)];
    let _e179 = ((((in1_2.x * bitcast<f32>(_e152)) + (in1_2.y * bitcast<f32>(_e159))) + (in1_2.z * bitcast<f32>(_e167))) + (bitcast<f32>(_e175) * 0f));
    let _e187 = storage_0_1_[(_e9 + 18u)];
    let _e194 = storage_0_1_[(_e9 + 22u)];
    let _e202 = storage_0_1_[(_e9 + 26u)];
    let _e210 = storage_0_1_[(_e9 + 30u)];
    let _e214 = ((((in1_2.x * bitcast<f32>(_e187)) + (in1_2.y * bitcast<f32>(_e194))) + (in1_2.z * bitcast<f32>(_e202))) + (bitcast<f32>(_e210) * 0f));
    vertex_out_1.o1_ = vec3<f32>(normalize(vec3<f32>(_e144, _e179, _e214)).x, normalize(vec3<f32>(_e144, _e179, _e214)).y, normalize(vec3<f32>(_e144, _e179, _e214)).z);
    vertex_out_1.o2_ = vec2<f32>(in2_1.x, in2_1.y);
    let _e237 = storage_0_1_[_e9];
    let _e242 = storage_0_1_[_e19];
    let _e248 = storage_0_1_[_e27];
    let _e254 = storage_0_1_[_e35];
    let _e258 = ((((in3_1.x * bitcast<f32>(_e237)) + (in3_1.y * bitcast<f32>(_e242))) + (in3_1.z * bitcast<f32>(_e248))) + (bitcast<f32>(_e254) * 0f));
    let _e264 = storage_0_1_[_e45];
    let _e269 = storage_0_1_[_e52];
    let _e275 = storage_0_1_[_e60];
    let _e281 = storage_0_1_[_e68];
    let _e285 = ((((in3_1.x * bitcast<f32>(_e264)) + (in3_1.y * bitcast<f32>(_e269))) + (in3_1.z * bitcast<f32>(_e275))) + (bitcast<f32>(_e281) * 0f));
    let _e291 = storage_0_1_[_e78];
    let _e296 = storage_0_1_[_e85];
    let _e302 = storage_0_1_[_e93];
    let _e308 = storage_0_1_[_e101];
    let _e312 = ((((in3_1.x * bitcast<f32>(_e291)) + (in3_1.y * bitcast<f32>(_e296))) + (in3_1.z * bitcast<f32>(_e302))) + (bitcast<f32>(_e308) * 0f));
    vertex_out_1.o3_ = vec3<f32>(normalize(vec3<f32>(_e258, _e285, _e312)).x, normalize(vec3<f32>(_e258, _e285, _e312)).y, normalize(vec3<f32>(_e258, _e285, _e312)).z);
    let _e329 = storage_0_1_[(_e9 + 48u)];
    vertex_out_1.o4_ = _e329;
    let _e336 = storage_0_1_[(_e9 + 49u)];
    if (_e336 == 0u) {
        previous_w_u002e_sroa_u002e_0_u002e_0_1 = 0f;
        let _e342 = previous_w_u002e_sroa_u002e_0_u002e_0_1;
        let _e350 = storage_0_1_[(_e9 + 32u)];
        let _e357 = storage_0_1_[(_e9 + 36u)];
        let _e365 = storage_0_1_[(_e9 + 40u)];
        let _e373 = storage_0_1_[(_e9 + 44u)];
        let _e383 = storage_0_1_[(_e9 + 33u)];
        let _e390 = storage_0_1_[(_e9 + 37u)];
        let _e398 = storage_0_1_[(_e9 + 41u)];
        let _e406 = storage_0_1_[(_e9 + 45u)];
        let _e416 = storage_0_1_[(_e9 + 34u)];
        let _e423 = storage_0_1_[(_e9 + 38u)];
        let _e431 = storage_0_1_[(_e9 + 42u)];
        let _e439 = storage_0_1_[(_e9 + 46u)];
        vertex_out_1.o5_ = vec4<f32>(((((in0_2.x * bitcast<f32>(_e350)) + (in0_2.y * bitcast<f32>(_e357))) + (in0_2.z * bitcast<f32>(_e365))) + bitcast<f32>(_e373)), ((((in0_2.x * bitcast<f32>(_e383)) + (in0_2.y * bitcast<f32>(_e390))) + (in0_2.z * bitcast<f32>(_e398))) + bitcast<f32>(_e406)), ((((in0_2.x * bitcast<f32>(_e416)) + (in0_2.y * bitcast<f32>(_e423))) + (in0_2.z * bitcast<f32>(_e431))) + bitcast<f32>(_e439)), _e342);
        let _e450 = storage_0_1_[_e9];
        let _e455 = storage_0_1_[_e19];
        let _e461 = storage_0_1_[_e27];
        let _e467 = storage_0_1_[_e35];
        let _e469 = ((((in0_2.x * bitcast<f32>(_e450)) + (in0_2.y * bitcast<f32>(_e455))) + (in0_2.z * bitcast<f32>(_e461))) + bitcast<f32>(_e467));
        let _e475 = storage_0_1_[_e45];
        let _e480 = storage_0_1_[_e52];
        let _e486 = storage_0_1_[_e60];
        let _e492 = storage_0_1_[_e68];
        let _e494 = ((((in0_2.x * bitcast<f32>(_e475)) + (in0_2.y * bitcast<f32>(_e480))) + (in0_2.z * bitcast<f32>(_e486))) + bitcast<f32>(_e492));
        let _e500 = storage_0_1_[_e78];
        let _e505 = storage_0_1_[_e85];
        let _e511 = storage_0_1_[_e93];
        let _e517 = storage_0_1_[_e101];
        let _e519 = ((((in0_2.x * bitcast<f32>(_e500)) + (in0_2.y * bitcast<f32>(_e505))) + (in0_2.z * bitcast<f32>(_e511))) + bitcast<f32>(_e517));
        let _e523 = uniform_struct_0_0_.f0_.x;
        let _e528 = uniform_struct_0_0_.f1_.x;
        let _e534 = uniform_struct_0_0_.f2_.x;
        let _e540 = uniform_struct_0_0_.f3_.x;
        let _e545 = uniform_struct_0_0_.f0_.y;
        let _e550 = uniform_struct_0_0_.f1_.y;
        let _e556 = uniform_struct_0_0_.f2_.y;
        let _e562 = uniform_struct_0_0_.f3_.y;
        let _e567 = uniform_struct_0_0_.f0_.z;
        let _e572 = uniform_struct_0_0_.f1_.z;
        let _e578 = uniform_struct_0_0_.f2_.z;
        let _e584 = uniform_struct_0_0_.f3_.z;
        let _e589 = uniform_struct_0_0_.f0_.w;
        let _e594 = uniform_struct_0_0_.f1_.w;
        let _e600 = uniform_struct_0_0_.f2_.w;
        let _e606 = uniform_struct_0_0_.f3_.w;
        vertex_out_1.position = vec4<f32>((_e540 + (((_e469 * _e523) + (_e494 * _e528)) + (_e534 * _e519))), (_e562 + (((_e469 * _e545) + (_e494 * _e550)) + (_e519 * _e556))), (_e584 + (((_e469 * _e567) + (_e494 * _e572)) + (_e519 * _e578))), (_e606 + (((_e469 * _e589) + (_e494 * _e594)) + (_e519 * _e600))));
        let _e612 = vertex_out_1;
        return _e612;
    }
    let _e620 = storage_0_1_[(_e9 + 35u)];
    let _e627 = storage_0_1_[(_e9 + 39u)];
    let _e635 = storage_0_1_[(_e9 + 43u)];
    let _e643 = storage_0_1_[(_e9 + 47u)];
    previous_w_u002e_sroa_u002e_0_u002e_0_1 = ((((in0_2.x * bitcast<f32>(_e620)) + (in0_2.y * bitcast<f32>(_e627))) + (in0_2.z * bitcast<f32>(_e635))) + bitcast<f32>(_e643));
    let _e648 = previous_w_u002e_sroa_u002e_0_u002e_0_1;
    let _e656 = storage_0_1_[(_e9 + 32u)];
    let _e663 = storage_0_1_[(_e9 + 36u)];
    let _e671 = storage_0_1_[(_e9 + 40u)];
    let _e679 = storage_0_1_[(_e9 + 44u)];
    let _e689 = storage_0_1_[(_e9 + 33u)];
    let _e696 = storage_0_1_[(_e9 + 37u)];
    let _e704 = storage_0_1_[(_e9 + 41u)];
    let _e712 = storage_0_1_[(_e9 + 45u)];
    let _e722 = storage_0_1_[(_e9 + 34u)];
    let _e729 = storage_0_1_[(_e9 + 38u)];
    let _e737 = storage_0_1_[(_e9 + 42u)];
    let _e745 = storage_0_1_[(_e9 + 46u)];
    vertex_out_1.o5_ = vec4<f32>(((((in0_2.x * bitcast<f32>(_e656)) + (in0_2.y * bitcast<f32>(_e663))) + (in0_2.z * bitcast<f32>(_e671))) + bitcast<f32>(_e679)), ((((in0_2.x * bitcast<f32>(_e689)) + (in0_2.y * bitcast<f32>(_e696))) + (in0_2.z * bitcast<f32>(_e704))) + bitcast<f32>(_e712)), ((((in0_2.x * bitcast<f32>(_e722)) + (in0_2.y * bitcast<f32>(_e729))) + (in0_2.z * bitcast<f32>(_e737))) + bitcast<f32>(_e745)), _e648);
    let _e756 = storage_0_1_[_e9];
    let _e761 = storage_0_1_[_e19];
    let _e767 = storage_0_1_[_e27];
    let _e773 = storage_0_1_[_e35];
    let _e775 = ((((in0_2.x * bitcast<f32>(_e756)) + (in0_2.y * bitcast<f32>(_e761))) + (in0_2.z * bitcast<f32>(_e767))) + bitcast<f32>(_e773));
    let _e781 = storage_0_1_[_e45];
    let _e786 = storage_0_1_[_e52];
    let _e792 = storage_0_1_[_e60];
    let _e798 = storage_0_1_[_e68];
    let _e800 = ((((in0_2.x * bitcast<f32>(_e781)) + (in0_2.y * bitcast<f32>(_e786))) + (in0_2.z * bitcast<f32>(_e792))) + bitcast<f32>(_e798));
    let _e806 = storage_0_1_[_e78];
    let _e811 = storage_0_1_[_e85];
    let _e817 = storage_0_1_[_e93];
    let _e823 = storage_0_1_[_e101];
    let _e825 = ((((in0_2.x * bitcast<f32>(_e806)) + (in0_2.y * bitcast<f32>(_e811))) + (in0_2.z * bitcast<f32>(_e817))) + bitcast<f32>(_e823));
    let _e829 = uniform_struct_0_0_.f0_.x;
    let _e834 = uniform_struct_0_0_.f1_.x;
    let _e840 = uniform_struct_0_0_.f2_.x;
    let _e846 = uniform_struct_0_0_.f3_.x;
    let _e851 = uniform_struct_0_0_.f0_.y;
    let _e856 = uniform_struct_0_0_.f1_.y;
    let _e862 = uniform_struct_0_0_.f2_.y;
    let _e868 = uniform_struct_0_0_.f3_.y;
    let _e873 = uniform_struct_0_0_.f0_.z;
    let _e878 = uniform_struct_0_0_.f1_.z;
    let _e884 = uniform_struct_0_0_.f2_.z;
    let _e890 = uniform_struct_0_0_.f3_.z;
    let _e895 = uniform_struct_0_0_.f0_.w;
    let _e900 = uniform_struct_0_0_.f1_.w;
    let _e906 = uniform_struct_0_0_.f2_.w;
    let _e912 = uniform_struct_0_0_.f3_.w;
    vertex_out_1.position = vec4<f32>((_e846 + (((_e775 * _e829) + (_e800 * _e834)) + (_e840 * _e825))), (_e868 + (((_e775 * _e851) + (_e800 * _e856)) + (_e825 * _e862))), (_e890 + (((_e775 * _e873) + (_e800 * _e878)) + (_e825 * _e884))), (_e912 + (((_e775 * _e895) + (_e800 * _e900)) + (_e825 * _e906))));
    let _e918 = vertex_out_1;
    return _e918;
}
