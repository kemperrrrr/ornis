struct UniformStruct0_ {
    f0_: vec4<f32>,
    f1_: vec4<f32>,
    f2_: vec4<f32>,
    f3_: vec4<f32>,
    f4_: vec4<f32>,
    f5_: vec4<f32>,
    f6_: u32,
}

@group(0) @binding(1) 
var<storage> storage_0_1_: array<f32>;
@group(0) @binding(2) 
var<storage, read_write> storage_0_2_: array<u32>;
@group(0) @binding(0) 
var<uniform> uniform_struct_0_0_: UniformStruct0_;
@group(0) @binding(3) 
var<storage, read_write> atomic_storage_0_3_: array<atomic<u32>>;

@compute @workgroup_size(64, 1, 1) 
fn cs_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let _e7 = arrayLength((&storage_0_2_));
    let _e10 = uniform_struct_0_0_.f6_;
    if ((global_invocation_id.x >= (arrayLength((&storage_0_1_)) / 52u)) | (global_invocation_id.x >= _e10)) {
        return;
    }
    let _e15 = (global_invocation_id.x * 52u);
    let _e18 = storage_0_1_[_e15];
    let _e23 = storage_0_1_[(_e15 | 1u)];
    let _e28 = storage_0_1_[(_e15 | 2u)];
    let _e33 = storage_0_1_[(_e15 + 4u)];
    let _e38 = storage_0_1_[(_e15 + 5u)];
    let _e43 = storage_0_1_[(_e15 + 6u)];
    let _e48 = storage_0_1_[(_e15 + 8u)];
    let _e53 = storage_0_1_[(_e15 + 9u)];
    let _e58 = storage_0_1_[(_e15 + 10u)];
    let _e63 = storage_0_1_[(_e15 + 12u)];
    let _e68 = storage_0_1_[(_e15 + 13u)];
    let _e73 = storage_0_1_[(_e15 + 14u)];
    let _e75 = length(vec3<f32>(_e18, _e23, _e28));
    let _e77 = length(vec3<f32>(_e33, _e38, _e43));
    let _e79 = length(vec3<f32>(_e48, _e53, _e58));
    let _e81 = select(_e77, _e75, (_e75 > _e77));
    let _e83 = select(_e79, _e81, (_e81 > _e79));
    if !((_e83 >= 0f)) {
        let _e91 = atomicAdd((&atomic_storage_0_3_[1u]), 1u);
        if (_e91 < _e7) {
        } else {
            return;
        }
        storage_0_2_[_e91] = global_invocation_id.x;
    } else {
        let _e97 = uniform_struct_0_0_.f0_;
        let _e101 = -(_e83);
        if (dot(vec4<f32>(_e63, _e68, _e73, 1f), _e97) < _e101) {
            return;
        }
        let _e105 = uniform_struct_0_0_.f1_;
        if (dot(vec4<f32>(_e63, _e68, _e73, 1f), _e105) < _e101) {
            return;
        }
        let _e112 = uniform_struct_0_0_.f2_;
        if (dot(vec4<f32>(_e63, _e68, _e73, 1f), _e112) < _e101) {
            return;
        }
        let _e119 = uniform_struct_0_0_.f3_;
        if (dot(vec4<f32>(_e63, _e68, _e73, 1f), _e119) < _e101) {
            return;
        }
        let _e126 = uniform_struct_0_0_.f4_;
        if (dot(vec4<f32>(_e63, _e68, _e73, 1f), _e126) < _e101) {
            return;
        }
        let _e133 = uniform_struct_0_0_.f5_;
        if (dot(vec4<f32>(_e63, _e68, _e73, 1f), _e133) < _e101) {
            return;
        }
        let _e142 = atomicAdd((&atomic_storage_0_3_[1u]), 1u);
        if (_e142 < _e7) {
        } else {
            return;
        }
        storage_0_2_[_e142] = global_invocation_id.x;
    }
    return;
}
