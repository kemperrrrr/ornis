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
    var lid_u002e_sroa_u002e_0_u002e_0_u002e_i27_: u32;
    var lid_u002e_sroa_u002e_0_u002e_0_u002e_i529_: u32;

    if (local_invocation_index < 256u) {
        lid_u002e_sroa_u002e_0_u002e_0_u002e_i27_ = local_invocation_index;
        loop {
            let _e7 = lid_u002e_sroa_u002e_0_u002e_0_u002e_i27_;
            atomicStore((&workgroup_atomics[_e7]), 0u);
            if (_e7 < 192u) {
                lid_u002e_sroa_u002e_0_u002e_0_u002e_i27_ = (_e7 + 64u);
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
        let _e79 = (_e53.x * _e74.w);
        let _e81 = (_e78 == 1u);
        let _e84 = (_e78 == 2u);
        let _e87 = (_e60.y * _e74.w);
        let _e91 = (_e67.z * _e74.w);
        let _e100 = dot(vec3<f32>(select(select(_e32.x, _e79, _e81), (_e32.x + _e79), _e84), select(select(_e39.y, _e87, _e81), (_e39.y + _e87), _e84), select(select(_e46.z, _e91, _e81), (_e46.z + _e91), _e84)), vec3<f32>(0.2126f, 0.7152f, 0.0722f));
        if !(((_e100 <= _e100) && (0f <= 0f))) {
            safe_u002e_sroa_u002e_0_u002e_0_u002e_i = 0.000001f;
        } else {
            if (abs(_e100) > 340000000000000000000000000000000000000f) {
                safe_u002e_sroa_u002e_0_u002e_0_u002e_i = 0.000001f;
            } else {
                safe_u002e_sroa_u002e_0_u002e_0_u002e_i = max(_e100, 0.000001f);
            }
        }
        let _e117 = safe_u002e_sroa_u002e_0_u002e_0_u002e_i;
        let _e118 = log2(_e117);
        let _e121 = uniform_struct_0_3_.f0_;
        let _e124 = uniform_struct_0_3_.f1_;
        let _e127 = uniform_struct_0_3_.f2_;
        if (_e118 < _e121) {
            _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 0u;
        } else {
            _0_u002e_sroa_u002e_0_u002e_0_u002e_i = u32((((_e127 + -2f) * clamp(((_e118 - _e121) * _e124), 0f, 1f)) + 1f));
        }
        let _e144 = _0_u002e_sroa_u002e_0_u002e_0_u002e_i;
        let _e148 = atomicAdd((&workgroup_atomics[_e144]), 1u);
    }
    workgroupBarrier();
    if (local_invocation_index < 256u) {
        lid_u002e_sroa_u002e_0_u002e_0_u002e_i529_ = local_invocation_index;
    } else {
        return;
    }
    loop {
        let _e153 = lid_u002e_sroa_u002e_0_u002e_0_u002e_i529_;
        let _e156 = atomicLoad((&workgroup_atomics[_e153]));
        if (_e156 == 0u) {
        } else {
            let _e161 = atomicAdd((&atomic_storage_0_2_[_e153]), _e156);
        }
        if (_e153 < 192u) {
            lid_u002e_sroa_u002e_0_u002e_0_u002e_i529_ = (_e153 + 64u);
            continue;
        } else {
            break;
        }
    }
    return;
}
