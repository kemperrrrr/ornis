struct UniformStruct0_ {
    f0_: f32,
    f1_: f32,
    f2_: f32,
    f3_: f32,
    f4_: f32,
    f5_: f32,
    f6_: f32,
    f7_: f32,
    f8_: f32,
    f9_: f32,
    f10_: f32,
    f11_: f32,
}

@group(0) @binding(0) 
var<storage, read_write> storage_0_0_: array<u32>;
var<workgroup> workgroup_scratch: array<u32, 256>;
@group(0) @binding(1) 
var<uniform> uniform_struct_0_1_: UniformStruct0_;
@group(0) @binding(2) 
var<storage, read_write> storage_0_2_: array<f32>;

@compute @workgroup_size(1, 1, 1) 
fn cs_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    var total_u002e_sroa_u002e_0_u002e_0_u002e_i20_: u32;
    var index_u002e_sroa_u002e_0_u002e_0_u002e_i119_: u32;
    var cumulative_u002e_sroa_u002e_0_u002e_0_u002e_i24_: u32;
    var sum_u002e_sroa_u002e_0_u002e_0_u002e_i23_: f32;
    var sum_weight_u002e_sroa_u002e_0_u002e_0_u002e_i22_: f32;
    var index_u002e_sroa_u002e_0_u002e_0_u002e_i21_: u32;
    var out_u002e_spec_u002e_select_u002e_i: u32;
    var out_u002e_4_: f32;
    var out_u002e_3_: f32;

    let _e5 = storage_0_0_[0u];
    workgroup_scratch[0u] = _e5;
    storage_0_0_[0u] = 0u;
    total_u002e_sroa_u002e_0_u002e_0_u002e_i20_ = 0u;
    index_u002e_sroa_u002e_0_u002e_0_u002e_i119_ = 1u;
    loop {
        let _e18 = total_u002e_sroa_u002e_0_u002e_0_u002e_i20_;
        let _e20 = index_u002e_sroa_u002e_0_u002e_0_u002e_i119_;
        let _e23 = storage_0_0_[_e20];
        workgroup_scratch[_e20] = _e23;
        storage_0_0_[_e20] = 0u;
        let _e29 = (_e23 + _e18);
        out_u002e_spec_u002e_select_u002e_i = _e29;
        let _e32 = (_e20 + 1u);
        if (_e32 == 256u) {
            break;
        } else {
            total_u002e_sroa_u002e_0_u002e_0_u002e_i20_ = _e29;
            index_u002e_sroa_u002e_0_u002e_0_u002e_i119_ = _e32;
            continue;
        }
    }
    let _e38 = out_u002e_spec_u002e_select_u002e_i;
    if (_e38 == 0u) {
        return;
    }
    let _e41 = f32(_e38);
    let _e44 = uniform_struct_0_1_.f2_;
    let _e45 = (_e44 * _e41);
    let _e48 = uniform_struct_0_1_.f3_;
    let _e49 = (_e48 * _e41);
    cumulative_u002e_sroa_u002e_0_u002e_0_u002e_i24_ = 0u;
    sum_u002e_sroa_u002e_0_u002e_0_u002e_i23_ = 0f;
    sum_weight_u002e_sroa_u002e_0_u002e_0_u002e_i22_ = 0f;
    index_u002e_sroa_u002e_0_u002e_0_u002e_i21_ = 1u;
    loop {
        let _e59 = cumulative_u002e_sroa_u002e_0_u002e_0_u002e_i24_;
        let _e61 = sum_u002e_sroa_u002e_0_u002e_0_u002e_i23_;
        let _e63 = sum_weight_u002e_sroa_u002e_0_u002e_0_u002e_i22_;
        let _e65 = index_u002e_sroa_u002e_0_u002e_0_u002e_i21_;
        let _e68 = workgroup_scratch[_e65];
        let _e69 = (_e68 + _e59);
        let _e74 = (clamp(f32(_e69), _e45, _e49) - clamp(f32(_e59), _e45, _e49));
        let _e85 = uniform_struct_0_1_.f0_;
        let _e88 = uniform_struct_0_1_.f1_;
        let _e92 = (_e61 + (_e74 * (_e85 + (clamp(((f32(_e65) + -0.5f) / 254f), 0f, 1f) * _e88))));
        out_u002e_3_ = _e92;
        let _e94 = (_e63 + _e74);
        out_u002e_4_ = _e94;
        let _e97 = (_e65 + 1u);
        if (_e97 == 256u) {
            break;
        } else {
            cumulative_u002e_sroa_u002e_0_u002e_0_u002e_i24_ = _e69;
            sum_u002e_sroa_u002e_0_u002e_0_u002e_i23_ = _e92;
            sum_weight_u002e_sroa_u002e_0_u002e_0_u002e_i22_ = _e94;
            index_u002e_sroa_u002e_0_u002e_0_u002e_i21_ = _e97;
            continue;
        }
    }
    let _e105 = out_u002e_4_;
    let _e107 = out_u002e_3_;
    if (_e105 > 0f) {
    } else {
        return;
    }
    let _e114 = storage_0_2_[0u];
    let _e117 = uniform_struct_0_1_.f5_;
    let _e120 = uniform_struct_0_1_.f6_;
    let _e124 = clamp(log2(max(_e114, 0.000001f)), _e117, _e120);
    let _e127 = uniform_struct_0_1_.f4_;
    let _e130 = (clamp((_e127 - (_e107 / _e105)), _e117, _e120) - _e124);
    let _e133 = uniform_struct_0_1_.f7_;
    let _e136 = uniform_struct_0_1_.f8_;
    let _e142 = uniform_struct_0_1_.f10_;
    let _e143 = (select(_e136, _e133, (_e130 > 0f)) * _e142);
    let _e146 = uniform_struct_0_1_.f9_;
    storage_0_2_[0u] = exp2(select((_e124 + (sign(_e130) * min(min(_e143, ((_e143 * abs(_e130)) / max(_e146, 0.001f))), abs(_e130)))), _e124, (_e130 == 0f)));
    return;
}
