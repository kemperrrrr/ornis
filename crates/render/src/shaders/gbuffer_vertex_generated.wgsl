struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) o0_: vec3<f32>,
    @location(1) o1_: vec3<f32>,
    @location(2) o2_: vec2<f32>,
    @location(3) o3_: vec3<f32>,
    @location(4) @interpolate(flat) o4_: u32,
    @location(5) o5_: vec4<f32>,
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

@group(0) @binding(1) 
var<storage> storage_0_1_: array<u32>;
@group(0) @binding(0) 
var<uniform> uniform_struct_0_0_: UniformStruct0_;

@vertex 
fn vs_main(@builtin(instance_index) instance_index: u32, @location(0) in0_: vec3<f32>, @location(1) in1_: vec3<f32>, @location(2) in2_: vec2<f32>, @location(3) in3_: vec3<f32>) -> VertexOut {
    var vertex_out: VertexOut;
    var previous_w_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;

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
    vertex_out.o0_ = vec3<f32>(((((in0_.x * bitcast<f32>(_e12)) + (in0_.y * bitcast<f32>(_e19))) + (in0_.z * bitcast<f32>(_e27))) + bitcast<f32>(_e35)), ((((in0_.x * bitcast<f32>(_e45)) + (in0_.y * bitcast<f32>(_e52))) + (in0_.z * bitcast<f32>(_e60))) + bitcast<f32>(_e68)), ((((in0_.x * bitcast<f32>(_e78)) + (in0_.y * bitcast<f32>(_e85))) + (in0_.z * bitcast<f32>(_e93))) + bitcast<f32>(_e101)));
    let _e114 = storage_0_1_[(_e6 + 16u)];
    let _e121 = storage_0_1_[(_e6 + 20u)];
    let _e129 = storage_0_1_[(_e6 + 24u)];
    let _e137 = storage_0_1_[(_e6 + 28u)];
    let _e141 = ((((in1_.x * bitcast<f32>(_e114)) + (in1_.y * bitcast<f32>(_e121))) + (in1_.z * bitcast<f32>(_e129))) + (bitcast<f32>(_e137) * 0f));
    let _e149 = storage_0_1_[(_e6 + 17u)];
    let _e156 = storage_0_1_[(_e6 + 21u)];
    let _e164 = storage_0_1_[(_e6 + 25u)];
    let _e172 = storage_0_1_[(_e6 + 29u)];
    let _e176 = ((((in1_.x * bitcast<f32>(_e149)) + (in1_.y * bitcast<f32>(_e156))) + (in1_.z * bitcast<f32>(_e164))) + (bitcast<f32>(_e172) * 0f));
    let _e184 = storage_0_1_[(_e6 + 18u)];
    let _e191 = storage_0_1_[(_e6 + 22u)];
    let _e199 = storage_0_1_[(_e6 + 26u)];
    let _e207 = storage_0_1_[(_e6 + 30u)];
    let _e211 = ((((in1_.x * bitcast<f32>(_e184)) + (in1_.y * bitcast<f32>(_e191))) + (in1_.z * bitcast<f32>(_e199))) + (bitcast<f32>(_e207) * 0f));
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
        previous_w_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
        let _e339 = previous_w_u002e_sroa_u002e_0_u002e_0_u002e_i;
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
        vertex_out.o5_ = vec4<f32>(((((in0_.x * bitcast<f32>(_e347)) + (in0_.y * bitcast<f32>(_e354))) + (in0_.z * bitcast<f32>(_e362))) + bitcast<f32>(_e370)), ((((in0_.x * bitcast<f32>(_e380)) + (in0_.y * bitcast<f32>(_e387))) + (in0_.z * bitcast<f32>(_e395))) + bitcast<f32>(_e403)), ((((in0_.x * bitcast<f32>(_e413)) + (in0_.y * bitcast<f32>(_e420))) + (in0_.z * bitcast<f32>(_e428))) + bitcast<f32>(_e436)), _e339);
        let _e447 = storage_0_1_[_e6];
        let _e452 = storage_0_1_[_e16];
        let _e458 = storage_0_1_[_e24];
        let _e464 = storage_0_1_[_e32];
        let _e466 = ((((in0_.x * bitcast<f32>(_e447)) + (in0_.y * bitcast<f32>(_e452))) + (in0_.z * bitcast<f32>(_e458))) + bitcast<f32>(_e464));
        let _e472 = storage_0_1_[_e42];
        let _e477 = storage_0_1_[_e49];
        let _e483 = storage_0_1_[_e57];
        let _e489 = storage_0_1_[_e65];
        let _e491 = ((((in0_.x * bitcast<f32>(_e472)) + (in0_.y * bitcast<f32>(_e477))) + (in0_.z * bitcast<f32>(_e483))) + bitcast<f32>(_e489));
        let _e497 = storage_0_1_[_e75];
        let _e502 = storage_0_1_[_e82];
        let _e508 = storage_0_1_[_e90];
        let _e514 = storage_0_1_[_e98];
        let _e516 = ((((in0_.x * bitcast<f32>(_e497)) + (in0_.y * bitcast<f32>(_e502))) + (in0_.z * bitcast<f32>(_e508))) + bitcast<f32>(_e514));
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
    previous_w_u002e_sroa_u002e_0_u002e_0_u002e_i = ((((in0_.x * bitcast<f32>(_e617)) + (in0_.y * bitcast<f32>(_e624))) + (in0_.z * bitcast<f32>(_e632))) + bitcast<f32>(_e640));
    let _e645 = previous_w_u002e_sroa_u002e_0_u002e_0_u002e_i;
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
    vertex_out.o5_ = vec4<f32>(((((in0_.x * bitcast<f32>(_e653)) + (in0_.y * bitcast<f32>(_e660))) + (in0_.z * bitcast<f32>(_e668))) + bitcast<f32>(_e676)), ((((in0_.x * bitcast<f32>(_e686)) + (in0_.y * bitcast<f32>(_e693))) + (in0_.z * bitcast<f32>(_e701))) + bitcast<f32>(_e709)), ((((in0_.x * bitcast<f32>(_e719)) + (in0_.y * bitcast<f32>(_e726))) + (in0_.z * bitcast<f32>(_e734))) + bitcast<f32>(_e742)), _e645);
    let _e753 = storage_0_1_[_e6];
    let _e758 = storage_0_1_[_e16];
    let _e764 = storage_0_1_[_e24];
    let _e770 = storage_0_1_[_e32];
    let _e772 = ((((in0_.x * bitcast<f32>(_e753)) + (in0_.y * bitcast<f32>(_e758))) + (in0_.z * bitcast<f32>(_e764))) + bitcast<f32>(_e770));
    let _e778 = storage_0_1_[_e42];
    let _e783 = storage_0_1_[_e49];
    let _e789 = storage_0_1_[_e57];
    let _e795 = storage_0_1_[_e65];
    let _e797 = ((((in0_.x * bitcast<f32>(_e778)) + (in0_.y * bitcast<f32>(_e783))) + (in0_.z * bitcast<f32>(_e789))) + bitcast<f32>(_e795));
    let _e803 = storage_0_1_[_e75];
    let _e808 = storage_0_1_[_e82];
    let _e814 = storage_0_1_[_e90];
    let _e820 = storage_0_1_[_e98];
    let _e822 = ((((in0_.x * bitcast<f32>(_e803)) + (in0_.y * bitcast<f32>(_e808))) + (in0_.z * bitcast<f32>(_e814))) + bitcast<f32>(_e820));
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
