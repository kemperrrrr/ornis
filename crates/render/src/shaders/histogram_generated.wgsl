struct UniformStruct0_ {
    f0_: f32,
    f1_: f32,
    f2_: f32,
    f3_: u32,
}

var<workgroup> workgroup_atomics: array<atomic<u32>, 256>;
@group(0) @binding(0) 
var tex_0_0_: texture_2d<f32>;
@group(0) @binding(1) 
var tex_0_1_: texture_2d<f32>;
@group(0) @binding(3) 
var<uniform> uniform_struct_0_3_: UniformStruct0_;
@group(0) @binding(2) 
var<storage, read_write> atomic_storage_0_2_: array<atomic<u32>>;

@compute @workgroup_size(8, 8, 1) 
fn cs_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32) {
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_i: u32;
    var safe_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var lid_u002e_sroa_u002e_0_u002e_0_u002e_i13_: u32;
    var lid_u002e_sroa_u002e_0_u002e_0_u002e_i315_: u32;

    if (local_invocation_index < 256u) {
        lid_u002e_sroa_u002e_0_u002e_0_u002e_i13_ = local_invocation_index;
        loop {
            let _e7 = lid_u002e_sroa_u002e_0_u002e_0_u002e_i13_;
            atomicStore((&workgroup_atomics[_e7]), 0u);
            if (_e7 < 192u) {
                lid_u002e_sroa_u002e_0_u002e_0_u002e_i13_ = (_e7 + 64u);
                continue;
            } else {
                break;
            }
        }
    }
    workgroupBarrier();
    let _e19 = textureDimensions(tex_0_0_);
    let _e22 = textureDimensions(tex_0_0_);
    if ((global_invocation_id.x < _e19.x) & (global_invocation_id.y < _e22.y)) {
        let _e32 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(global_invocation_id.x), bitcast<i32>(global_invocation_id.y)), 0i);
        let _e39 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(global_invocation_id.x), bitcast<i32>(global_invocation_id.y)), 0i);
        let _e46 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(global_invocation_id.x), bitcast<i32>(global_invocation_id.y)), 0i);
        let _e53 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(global_invocation_id.x), bitcast<i32>(global_invocation_id.y)), 0i);
        let _e60 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(global_invocation_id.x), bitcast<i32>(global_invocation_id.y)), 0i);
        let _e67 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(global_invocation_id.x), bitcast<i32>(global_invocation_id.y)), 0i);
        let _e74 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(global_invocation_id.x), bitcast<i32>(global_invocation_id.y)), 0i);
        let _e78 = uniform_struct_0_3_.f3_;
        let _e83 = select(1f, 0f, (_e78 == 1u));
        let _e88 = select(1f, 0f, (_e78 == 0u));
        let _e106 = dot(vec3<f32>(((_e32.x * _e83) + ((_e53.x * _e74.w) * _e88)), ((_e39.y * _e83) + ((_e60.y * _e74.w) * _e88)), ((_e46.z * _e83) + ((_e67.z * _e74.w) * _e88))), vec3<f32>(0.2126f, 0.7152f, 0.0722f));
        if ((_e106 != _e106) | (0f != 0f)) {
            safe_u002e_sroa_u002e_0_u002e_0_u002e_i = 0.000001f;
        } else {
            if (abs(_e106) > 340000000000000000000000000000000000000f) {
                safe_u002e_sroa_u002e_0_u002e_0_u002e_i = 0.000001f;
            } else {
                safe_u002e_sroa_u002e_0_u002e_0_u002e_i = max(_e106, 0.000001f);
            }
        }
        let _e122 = safe_u002e_sroa_u002e_0_u002e_0_u002e_i;
        let _e123 = log2(_e122);
        let _e126 = uniform_struct_0_3_.f0_;
        let _e129 = uniform_struct_0_3_.f1_;
        let _e132 = uniform_struct_0_3_.f2_;
        if (_e123 < _e126) {
            _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0u;
        } else {
            _0_u002e_sroa_u002e_0_u002e_0_u002e_i = u32((((_e132 + -2f) * clamp(((_e123 - _e126) * _e129), 0f, 1f)) + 1f));
        }
        let _e149 = _0_u002e_sroa_u002e_0_u002e_0_u002e_i;
        let _e153 = atomicAdd((&workgroup_atomics[_e149]), 1u);
    }
    workgroupBarrier();
    if (local_invocation_index < 256u) {
        lid_u002e_sroa_u002e_0_u002e_0_u002e_i315_ = local_invocation_index;
    } else {
        return;
    }
    loop {
        let _e158 = lid_u002e_sroa_u002e_0_u002e_0_u002e_i315_;
        let _e161 = atomicLoad((&workgroup_atomics[_e158]));
        if (_e161 == 0u) {
        } else {
            let _e166 = atomicAdd((&atomic_storage_0_2_[_e158]), _e161);
        }
        if (_e158 < 192u) {
            lid_u002e_sroa_u002e_0_u002e_0_u002e_i315_ = (_e158 + 64u);
            continue;
        } else {
            break;
        }
    }
    return;
}
