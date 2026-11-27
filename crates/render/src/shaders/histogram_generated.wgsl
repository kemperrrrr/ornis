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
    var _0_u002e_sroa_u002e_0_u002e_0_: u32;
    var safe_u002e_sroa_u002e_0_u002e_0_: f32;
    var lid_u002e_sroa_u002e_0_u002e_0_: u32;
    var lid_u002e_sroa_u002e_0_u002e_0_1: u32;

    if (local_invocation_index < 256u) {
        lid_u002e_sroa_u002e_0_u002e_0_ = local_invocation_index;
        loop {
            let _e7 = lid_u002e_sroa_u002e_0_u002e_0_;
            atomicStore((&workgroup_atomics[_e7]), 0u);
            if (_e7 < 192u) {
                lid_u002e_sroa_u002e_0_u002e_0_ = (_e7 + 64u);
                continue;
            } else {
                break;
            }
        }
    }
    workgroupBarrier();
    let _e19 = textureDimensions(tex_0_0_);
    if ((global_invocation_id.x < _e19.x) & (global_invocation_id.y < _e19.y)) {
        let _e30 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(global_invocation_id.x), bitcast<i32>(global_invocation_id.y)), 0i);
        let _e36 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(global_invocation_id.x), bitcast<i32>(global_invocation_id.y)), 0i);
        let _e42 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(global_invocation_id.x), bitcast<i32>(global_invocation_id.y)), 0i);
        let _e49 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(global_invocation_id.x), bitcast<i32>(global_invocation_id.y)), 0i);
        let _e55 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(global_invocation_id.x), bitcast<i32>(global_invocation_id.y)), 0i);
        let _e61 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(global_invocation_id.x), bitcast<i32>(global_invocation_id.y)), 0i);
        let _e67 = textureLoad(tex_0_1_, vec2<i32>(bitcast<i32>(global_invocation_id.x), bitcast<i32>(global_invocation_id.y)), 0i);
        let _e71 = uniform_struct_0_3_.f3_;
        let _e72 = (_e49.x * _e67.w);
        let _e74 = (_e71 == 1u);
        let _e77 = (_e71 == 2u);
        let _e80 = (_e55.y * _e67.w);
        let _e84 = (_e61.z * _e67.w);
        let _e93 = dot(vec3<f32>(select(select(_e30.x, _e72, _e74), (_e30.x + _e72), _e77), select(select(_e36.y, _e80, _e74), (_e36.y + _e80), _e77), select(select(_e42.z, _e84, _e74), (_e42.z + _e84), _e77)), vec3<f32>(0.2126f, 0.7152f, 0.0722f));
        if !(((_e93 <= _e93) && (0f <= 0f))) {
            safe_u002e_sroa_u002e_0_u002e_0_ = 0.000001f;
        } else {
            if (abs(_e93) > 340000000000000000000000000000000000000f) {
                safe_u002e_sroa_u002e_0_u002e_0_ = 0.000001f;
            } else {
                safe_u002e_sroa_u002e_0_u002e_0_ = max(_e93, 0.000001f);
            }
        }
        let _e110 = safe_u002e_sroa_u002e_0_u002e_0_;
        let _e111 = log2(_e110);
        let _e114 = uniform_struct_0_3_.f0_;
        let _e117 = uniform_struct_0_3_.f1_;
        let _e120 = uniform_struct_0_3_.f2_;
        if (_e111 < _e114) {
            _0_u002e_sroa_u002e_0_u002e_0_ = 0u;
        } else {
            _0_u002e_sroa_u002e_0_u002e_0_ = u32((((_e120 + -2f) * clamp(((_e111 - _e114) * _e117), 0f, 1f)) + 1f));
        }
        let _e137 = _0_u002e_sroa_u002e_0_u002e_0_;
        let _e141 = atomicAdd((&workgroup_atomics[_e137]), 1u);
    }
    workgroupBarrier();
    if (local_invocation_index < 256u) {
        lid_u002e_sroa_u002e_0_u002e_0_1 = local_invocation_index;
    } else {
        return;
    }
    loop {
        let _e146 = lid_u002e_sroa_u002e_0_u002e_0_1;
        let _e149 = atomicLoad((&workgroup_atomics[_e146]));
        if (_e149 == 0u) {
        } else {
            let _e154 = atomicAdd((&atomic_storage_0_2_[_e146]), _e149);
        }
        if (_e146 < 192u) {
            lid_u002e_sroa_u002e_0_u002e_0_1 = (_e146 + 64u);
            continue;
        } else {
            break;
        }
    }
    return;
}
