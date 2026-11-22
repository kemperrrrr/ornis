struct UniformStruct0_ {
    f0_: u32,
    f1_: u32,
    f2_: u32,
    f3_: u32,
}

struct UniformStruct1_ {
    f0_: f32,
    f1_: f32,
    f2_: f32,
    f3_: f32,
}

@group(0) @binding(3) 
var<uniform> uniform_struct_0_3_: UniformStruct0_;
@group(0) @binding(4) 
var<uniform> uniform_struct_0_4_: UniformStruct1_;
@group(0) @binding(0) 
var<storage> storage_0_0_: array<f32>;
@group(0) @binding(1) 
var<storage> storage_0_1_: array<f32>;
@group(0) @binding(2) 
var<storage, read_write> storage_0_2_: array<f32>;

@compute @workgroup_size(64, 1, 1) 
fn cs_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    var h_u002e_sroa_u002e_0_u002e_0_: f32;

    let _e5 = uniform_struct_0_3_.f0_;
    if (global_invocation_id.x < _e5) {
    } else {
        return;
    }
    let _e8 = (global_invocation_id.x << 2u);
    let _e11 = uniform_struct_0_4_.f0_;
    let _e16 = storage_0_0_[(_e8 | 3u)];
    if (_e16 > 0f) {
        let _e19 = (_e11 * _e11);
        if (_e19 > 0.000000000001f) {
            h_u002e_sroa_u002e_0_u002e_0_ = (1f / (_e16 * _e19));
        } else {
            h_u002e_sroa_u002e_0_u002e_0_ = 0f;
        }
    } else {
        h_u002e_sroa_u002e_0_u002e_0_ = 0f;
    }
    let _e31 = h_u002e_sroa_u002e_0_u002e_0_;
    let _e33 = (global_invocation_id.x << 3u);
    let _e36 = storage_0_1_[_e33];
    let _e38 = (_e31 > 0.000000000001f);
    storage_0_2_[_e33] = select(0f, (_e36 / _e31), _e38);
    let _e45 = (_e33 | 1u);
    let _e48 = storage_0_1_[_e45];
    storage_0_2_[_e45] = select(0f, (_e48 / _e31), _e38);
    let _e55 = (_e33 | 2u);
    let _e58 = storage_0_1_[_e55];
    storage_0_2_[_e55] = select(0f, (_e58 / _e31), _e38);
    let _e66 = storage_0_0_[_e8];
    let _e67 = (_e11 * _e11);
    let _e69 = (_e67 > 0.000000000001f);
    let _e72 = select(0f, (_e66 / _e67), _e69);
    let _e77 = storage_0_0_[(_e8 | 1u)];
    let _e80 = select(0f, (_e77 / _e67), _e69);
    let _e85 = storage_0_0_[(_e8 | 2u)];
    let _e88 = select(0f, (_e85 / _e67), _e69);
    let _e90 = (_e33 | 4u);
    let _e93 = storage_0_1_[_e90];
    storage_0_2_[_e90] = select(0f, (_e93 / _e72), (_e72 > 0.000000000001f));
    let _e102 = (_e33 | 5u);
    let _e105 = storage_0_1_[_e102];
    storage_0_2_[_e102] = select(0f, (_e105 / _e80), (_e80 > 0.000000000001f));
    let _e114 = (_e33 | 6u);
    let _e117 = storage_0_1_[_e114];
    storage_0_2_[_e114] = select(0f, (_e117 / _e88), (_e88 > 0.000000000001f));
    return;
}
