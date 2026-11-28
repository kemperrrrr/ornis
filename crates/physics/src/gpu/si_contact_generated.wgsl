struct UniformStruct0_ {
    f0_: u32,
    f1_: u32,
    f2_: u32,
    f3_: u32,
}

@group(0) @binding(1) 
var<storage, read_write> storage_0_1_: array<f32>;
@group(0) @binding(2) 
var<uniform> uniform_struct_0_2_: UniformStruct0_;
@group(0) @binding(0) 
var<storage, read_write> storage_0_0_: array<f32>;

@compute @workgroup_size(4, 1, 1) 
fn cs_main_contact(@builtin(global_invocation_id) global_invocation_id: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32) {
    var clamped_u002e_sroa_u002e_0_u002e_0_: f32;
    var clamped_u002e_sroa_u002e_0_u002e_0_1: f32;
    var wb_z_u002e_sroa_u002e_0_u002e_1_: f32;
    var wb_y_u002e_sroa_u002e_0_u002e_1_: f32;
    var wb_x_u002e_sroa_u002e_0_u002e_1_: f32;
    var vb_z_u002e_sroa_u002e_0_u002e_1_: f32;
    var vb_y_u002e_sroa_u002e_0_u002e_1_: f32;
    var vb_x_u002e_sroa_u002e_0_u002e_1_: f32;
    var wa_z_u002e_sroa_u002e_0_u002e_1_: f32;
    var wa_y_u002e_sroa_u002e_0_u002e_1_: f32;
    var wa_x_u002e_sroa_u002e_0_u002e_1_: f32;
    var va_z_u002e_sroa_u002e_0_u002e_1_: f32;
    var va_y_u002e_sroa_u002e_0_u002e_1_: f32;
    var va_x_u002e_sroa_u002e_0_u002e_1_: f32;
    var wb_z_u002e_sroa_u002e_0_u002e_2_: f32;
    var wb_y_u002e_sroa_u002e_0_u002e_2_: f32;
    var wb_x_u002e_sroa_u002e_0_u002e_2_: f32;
    var vb_z_u002e_sroa_u002e_0_u002e_2_: f32;
    var vb_y_u002e_sroa_u002e_0_u002e_2_: f32;
    var vb_x_u002e_sroa_u002e_0_u002e_2_: f32;
    var wa_z_u002e_sroa_u002e_0_u002e_2_: f32;
    var wa_y_u002e_sroa_u002e_0_u002e_2_: f32;
    var wa_x_u002e_sroa_u002e_0_u002e_2_: f32;
    var va_z_u002e_sroa_u002e_0_u002e_2_: f32;
    var va_y_u002e_sroa_u002e_0_u002e_2_: f32;
    var va_x_u002e_sroa_u002e_0_u002e_2_: f32;
    var wb_z_u002e_sroa_u002e_0_u002e_0_: f32;
    var wb_y_u002e_sroa_u002e_0_u002e_0_: f32;
    var wb_x_u002e_sroa_u002e_0_u002e_0_: f32;
    var vb_z_u002e_sroa_u002e_0_u002e_0_: f32;
    var vb_y_u002e_sroa_u002e_0_u002e_0_: f32;
    var vb_x_u002e_sroa_u002e_0_u002e_0_: f32;
    var wa_z_u002e_sroa_u002e_0_u002e_0_: f32;
    var wa_y_u002e_sroa_u002e_0_u002e_0_: f32;
    var wa_x_u002e_sroa_u002e_0_u002e_0_: f32;
    var va_z_u002e_sroa_u002e_0_u002e_0_: f32;
    var va_y_u002e_sroa_u002e_0_u002e_0_: f32;
    var va_x_u002e_sroa_u002e_0_u002e_0_: f32;
    var f_z_u002e_sroa_u002e_0_u002e_0_: f32;
    var f_y_u002e_sroa_u002e_0_u002e_0_: f32;
    var f_x_u002e_sroa_u002e_0_u002e_0_: f32;
    var f1_u002e_sroa_u002e_0_u002e_0_: f32;
    var f_z_u002e_sroa_u002e_0_u002e_1_: f32;
    var f_y_u002e_sroa_u002e_0_u002e_1_: f32;
    var f_x_u002e_sroa_u002e_0_u002e_1_: f32;
    var f2_u002e_sroa_u002e_0_u002e_0_: f32;

    let _e8 = (((global_invocation_id.x - local_invocation_index) >> 2u) * 312u);
    let _e13 = storage_0_1_[(_e8 + 308u)];
    if (local_invocation_index < bitcast<u32>(_e13)) {
    } else {
        return;
    }
    let _e18 = uniform_struct_0_2_.f0_;
    let _e21 = uniform_struct_0_2_.f1_;
    let _e24 = uniform_struct_0_2_.f2_;
    let _e30 = storage_0_1_[((_e8 + local_invocation_index) + 288u)];
    let _e33 = (bitcast<u32>(_e30) << 3u);
    let _e39 = storage_0_1_[((_e8 + local_invocation_index) + 292u)];
    let _e42 = (bitcast<u32>(_e39) << 3u);
    let _e45 = storage_0_0_[_e33];
    let _e47 = (_e33 | 1u);
    let _e50 = storage_0_0_[_e47];
    let _e52 = (_e33 | 2u);
    let _e55 = storage_0_0_[_e52];
    let _e57 = (_e33 | 4u);
    let _e60 = storage_0_0_[_e57];
    let _e62 = (_e33 | 5u);
    let _e65 = storage_0_0_[_e62];
    let _e67 = (_e33 | 6u);
    let _e70 = storage_0_0_[_e67];
    let _e73 = storage_0_0_[_e42];
    let _e75 = (_e42 | 1u);
    let _e78 = storage_0_0_[_e75];
    let _e80 = (_e42 | 2u);
    let _e83 = storage_0_0_[_e80];
    let _e85 = (_e42 | 4u);
    let _e88 = storage_0_0_[_e85];
    let _e90 = (_e42 | 5u);
    let _e93 = storage_0_0_[_e90];
    let _e95 = (_e42 | 6u);
    let _e98 = storage_0_0_[_e95];
    let _e102 = storage_0_1_[(_e8 + local_invocation_index)];
    let _e108 = storage_0_1_[((_e8 + local_invocation_index) + 4u)];
    let _e114 = storage_0_1_[((_e8 + local_invocation_index) + 8u)];
    let _e120 = storage_0_1_[((_e8 + local_invocation_index) + 12u)];
    let _e126 = storage_0_1_[((_e8 + local_invocation_index) + 16u)];
    let _e132 = storage_0_1_[((_e8 + local_invocation_index) + 20u)];
    let _e138 = storage_0_1_[((_e8 + local_invocation_index) + 24u)];
    let _e144 = storage_0_1_[((_e8 + local_invocation_index) + 28u)];
    let _e150 = storage_0_1_[((_e8 + local_invocation_index) + 32u)];
    let _e156 = storage_0_1_[((_e8 + local_invocation_index) + 132u)];
    let _e162 = storage_0_1_[((_e8 + local_invocation_index) + 136u)];
    let _e168 = storage_0_1_[((_e8 + local_invocation_index) + 144u)];
    let _e174 = storage_0_1_[((_e8 + local_invocation_index) + 280u)];
    let _e177 = ((_e8 + local_invocation_index) + 296u);
    let _e180 = storage_0_1_[_e177];
    let _e219 = (_e180 + (_e168 * (_e174 - dot(vec3<f32>(((_e73 + cross(vec3<f32>(_e88, _e93, _e98), vec3<f32>(_e138, _e144, _e150)).x) - (_e45 + cross(vec3<f32>(_e60, _e65, _e70), vec3<f32>(_e120, _e126, _e132)).x)), ((_e78 + cross(vec3<f32>(_e88, _e93, _e98), vec3<f32>(_e138, _e144, _e150)).y) - (_e50 + cross(vec3<f32>(_e60, _e65, _e70), vec3<f32>(_e120, _e126, _e132)).y)), ((_e83 + cross(vec3<f32>(_e88, _e93, _e98), vec3<f32>(_e138, _e144, _e150)).z) - (_e55 + cross(vec3<f32>(_e60, _e65, _e70), vec3<f32>(_e120, _e126, _e132)).z))), vec3<f32>(_e102, _e108, _e114)))));
    let _e223 = select(0f, _e219, (_e219 > 0f));
    let _e224 = (_e223 - _e180);
    if (abs(_e224) > 0.000000000001f) {
        let _e233 = storage_0_1_[((_e8 + local_invocation_index) + 156u)];
        let _e241 = storage_0_1_[((_e8 + local_invocation_index) + 160u)];
        let _e249 = storage_0_1_[((_e8 + local_invocation_index) + 164u)];
        let _e257 = storage_0_1_[((_e8 + local_invocation_index) + 180u)];
        let _e265 = storage_0_1_[((_e8 + local_invocation_index) + 184u)];
        let _e273 = storage_0_1_[((_e8 + local_invocation_index) + 188u)];
        let _e281 = storage_0_1_[((_e8 + local_invocation_index) + 168u)];
        let _e289 = storage_0_1_[((_e8 + local_invocation_index) + 172u)];
        let _e297 = storage_0_1_[((_e8 + local_invocation_index) + 176u)];
        let _e305 = storage_0_1_[((_e8 + local_invocation_index) + 192u)];
        let _e313 = storage_0_1_[((_e8 + local_invocation_index) + 196u)];
        let _e321 = storage_0_1_[((_e8 + local_invocation_index) + 200u)];
        wb_z_u002e_sroa_u002e_0_u002e_0_ = (_e98 + (_e224 * _e321));
        wb_y_u002e_sroa_u002e_0_u002e_0_ = (_e93 + (_e224 * _e313));
        wb_x_u002e_sroa_u002e_0_u002e_0_ = (_e88 + (_e224 * _e305));
        vb_z_u002e_sroa_u002e_0_u002e_0_ = (_e83 + (_e224 * _e297));
        vb_y_u002e_sroa_u002e_0_u002e_0_ = (_e78 + (_e224 * _e289));
        vb_x_u002e_sroa_u002e_0_u002e_0_ = (_e73 + (_e224 * _e281));
        wa_z_u002e_sroa_u002e_0_u002e_0_ = (_e70 - (_e224 * _e273));
        wa_y_u002e_sroa_u002e_0_u002e_0_ = (_e65 - (_e224 * _e265));
        wa_x_u002e_sroa_u002e_0_u002e_0_ = (_e60 - (_e224 * _e257));
        va_z_u002e_sroa_u002e_0_u002e_0_ = (_e55 - (_e224 * _e249));
        va_y_u002e_sroa_u002e_0_u002e_0_ = (_e50 - (_e224 * _e241));
        va_x_u002e_sroa_u002e_0_u002e_0_ = (_e45 - (_e224 * _e233));
    } else {
        wb_z_u002e_sroa_u002e_0_u002e_0_ = _e98;
        wb_y_u002e_sroa_u002e_0_u002e_0_ = _e93;
        wb_x_u002e_sroa_u002e_0_u002e_0_ = _e88;
        vb_z_u002e_sroa_u002e_0_u002e_0_ = _e83;
        vb_y_u002e_sroa_u002e_0_u002e_0_ = _e78;
        vb_x_u002e_sroa_u002e_0_u002e_0_ = _e73;
        wa_z_u002e_sroa_u002e_0_u002e_0_ = _e70;
        wa_y_u002e_sroa_u002e_0_u002e_0_ = _e65;
        wa_x_u002e_sroa_u002e_0_u002e_0_ = _e60;
        va_z_u002e_sroa_u002e_0_u002e_0_ = _e55;
        va_y_u002e_sroa_u002e_0_u002e_0_ = _e50;
        va_x_u002e_sroa_u002e_0_u002e_0_ = _e45;
    }
    let _e349 = wb_z_u002e_sroa_u002e_0_u002e_0_;
    let _e351 = wb_y_u002e_sroa_u002e_0_u002e_0_;
    let _e353 = wb_x_u002e_sroa_u002e_0_u002e_0_;
    let _e355 = vb_z_u002e_sroa_u002e_0_u002e_0_;
    let _e357 = vb_y_u002e_sroa_u002e_0_u002e_0_;
    let _e359 = vb_x_u002e_sroa_u002e_0_u002e_0_;
    let _e361 = wa_z_u002e_sroa_u002e_0_u002e_0_;
    let _e363 = wa_y_u002e_sroa_u002e_0_u002e_0_;
    let _e365 = wa_x_u002e_sroa_u002e_0_u002e_0_;
    let _e367 = va_z_u002e_sroa_u002e_0_u002e_0_;
    let _e369 = va_y_u002e_sroa_u002e_0_u002e_0_;
    let _e371 = va_x_u002e_sroa_u002e_0_u002e_0_;
    let _e382 = ((_e359 + cross(vec3<f32>(_e353, _e351, _e349), vec3<f32>(_e138, _e144, _e150)).x) - (_e371 + cross(vec3<f32>(_e365, _e363, _e361), vec3<f32>(_e120, _e126, _e132)).x));
    let _e393 = ((_e357 + cross(vec3<f32>(_e353, _e351, _e349), vec3<f32>(_e138, _e144, _e150)).y) - (_e369 + cross(vec3<f32>(_e365, _e363, _e361), vec3<f32>(_e120, _e126, _e132)).y));
    let _e404 = ((_e355 + cross(vec3<f32>(_e353, _e351, _e349), vec3<f32>(_e138, _e144, _e150)).z) - (_e367 + cross(vec3<f32>(_e365, _e363, _e361), vec3<f32>(_e120, _e126, _e132)).z));
    let _e410 = storage_0_1_[((_e8 + local_invocation_index) + 284u)];
    let _e411 = (_e223 * _e410);
    let _e417 = storage_0_1_[((_e8 + local_invocation_index) + 60u)];
    let _e423 = storage_0_1_[((_e8 + local_invocation_index) + 64u)];
    let _e429 = storage_0_1_[((_e8 + local_invocation_index) + 68u)];
    let _e435 = storage_0_1_[((_e8 + local_invocation_index) + 72u)];
    let _e441 = storage_0_1_[((_e8 + local_invocation_index) + 76u)];
    let _e447 = storage_0_1_[((_e8 + local_invocation_index) + 80u)];
    let _e450 = ((_e8 + local_invocation_index) + 300u);
    let _e453 = storage_0_1_[_e450];
    let _e456 = ((_e8 + local_invocation_index) + 304u);
    let _e459 = storage_0_1_[_e456];
    let _e462 = ((_e8 + local_invocation_index) + 148u);
    let _e465 = storage_0_1_[_e462];
    if (_e465 > 0f) {
        let _e473 = storage_0_1_[_e462];
        let _e475 = (_e453 - (dot(vec3<f32>(_e382, _e393, _e404), vec3<f32>(_e417, _e423, _e429)) * _e473));
        let _e479 = sqrt(((_e459 * _e459) + (_e475 * _e475)));
        if ((_e479 > _e411) & (_e479 > 0.000000000001f)) {
            clamped_u002e_sroa_u002e_0_u002e_0_1 = (_e475 * (_e411 / _e479));
        } else {
            clamped_u002e_sroa_u002e_0_u002e_0_1 = _e475;
        }
        let _e489 = clamped_u002e_sroa_u002e_0_u002e_0_1;
        let _e490 = (_e489 - _e453);
        f_z_u002e_sroa_u002e_0_u002e_0_ = ((_e429 * _e490) + 0f);
        f_y_u002e_sroa_u002e_0_u002e_0_ = ((_e423 * _e490) + 0f);
        f_x_u002e_sroa_u002e_0_u002e_0_ = ((_e417 * _e490) + 0f);
        f1_u002e_sroa_u002e_0_u002e_0_ = _e489;
    } else {
        f_z_u002e_sroa_u002e_0_u002e_0_ = 0f;
        f_y_u002e_sroa_u002e_0_u002e_0_ = 0f;
        f_x_u002e_sroa_u002e_0_u002e_0_ = 0f;
        f1_u002e_sroa_u002e_0_u002e_0_ = _e453;
    }
    let _e512 = f_z_u002e_sroa_u002e_0_u002e_0_;
    let _e514 = f_y_u002e_sroa_u002e_0_u002e_0_;
    let _e516 = f_x_u002e_sroa_u002e_0_u002e_0_;
    let _e518 = f1_u002e_sroa_u002e_0_u002e_0_;
    let _e521 = ((_e8 + local_invocation_index) + 152u);
    let _e524 = storage_0_1_[_e521];
    if (_e524 > 0f) {
        let _e532 = storage_0_1_[_e521];
        let _e534 = (_e459 - (dot(vec3<f32>(_e382, _e393, _e404), vec3<f32>(_e435, _e441, _e447)) * _e532));
        let _e538 = sqrt(((_e518 * _e518) + (_e534 * _e534)));
        if ((_e538 > _e411) & (_e538 > 0.000000000001f)) {
            clamped_u002e_sroa_u002e_0_u002e_0_ = (_e534 * (_e411 / _e538));
        } else {
            clamped_u002e_sroa_u002e_0_u002e_0_ = _e534;
        }
        let _e548 = clamped_u002e_sroa_u002e_0_u002e_0_;
        let _e549 = (_e548 - _e459);
        f_z_u002e_sroa_u002e_0_u002e_1_ = (_e512 + (_e447 * _e549));
        f_y_u002e_sroa_u002e_0_u002e_1_ = (_e514 + (_e441 * _e549));
        f_x_u002e_sroa_u002e_0_u002e_1_ = (_e516 + (_e435 * _e549));
        f2_u002e_sroa_u002e_0_u002e_0_ = _e548;
    } else {
        f_z_u002e_sroa_u002e_0_u002e_1_ = _e512;
        f_y_u002e_sroa_u002e_0_u002e_1_ = _e514;
        f_x_u002e_sroa_u002e_0_u002e_1_ = _e516;
        f2_u002e_sroa_u002e_0_u002e_0_ = _e459;
    }
    let _e565 = f_z_u002e_sroa_u002e_0_u002e_1_;
    let _e567 = f_y_u002e_sroa_u002e_0_u002e_1_;
    let _e569 = f_x_u002e_sroa_u002e_0_u002e_1_;
    let _e571 = f2_u002e_sroa_u002e_0_u002e_0_;
    if (dot(vec3<f32>(_e569, _e567, _e565), vec3<f32>(_e569, _e567, _e565)) > 0.000000000000000000000001f) {
        let _e594 = storage_0_1_[((_e8 + local_invocation_index) + 204u)];
        let _e600 = storage_0_1_[((_e8 + local_invocation_index) + 208u)];
        let _e606 = storage_0_1_[((_e8 + local_invocation_index) + 212u)];
        let _e615 = storage_0_1_[((_e8 + local_invocation_index) + 216u)];
        let _e621 = storage_0_1_[((_e8 + local_invocation_index) + 220u)];
        let _e627 = storage_0_1_[((_e8 + local_invocation_index) + 224u)];
        let _e636 = storage_0_1_[((_e8 + local_invocation_index) + 228u)];
        let _e642 = storage_0_1_[((_e8 + local_invocation_index) + 232u)];
        let _e648 = storage_0_1_[((_e8 + local_invocation_index) + 236u)];
        let _e669 = storage_0_1_[((_e8 + local_invocation_index) + 240u)];
        let _e675 = storage_0_1_[((_e8 + local_invocation_index) + 244u)];
        let _e681 = storage_0_1_[((_e8 + local_invocation_index) + 248u)];
        let _e690 = storage_0_1_[((_e8 + local_invocation_index) + 252u)];
        let _e696 = storage_0_1_[((_e8 + local_invocation_index) + 256u)];
        let _e702 = storage_0_1_[((_e8 + local_invocation_index) + 260u)];
        let _e711 = storage_0_1_[((_e8 + local_invocation_index) + 264u)];
        let _e717 = storage_0_1_[((_e8 + local_invocation_index) + 268u)];
        let _e723 = storage_0_1_[((_e8 + local_invocation_index) + 272u)];
        wb_z_u002e_sroa_u002e_0_u002e_1_ = (_e349 + dot(vec3<f32>(_e711, _e717, _e723), vec3<f32>(cross(vec3<f32>(_e138, _e144, _e150), vec3<f32>(_e569, _e567, _e565)).x, cross(vec3<f32>(_e138, _e144, _e150), vec3<f32>(_e569, _e567, _e565)).y, cross(vec3<f32>(_e138, _e144, _e150), vec3<f32>(_e569, _e567, _e565)).z)));
        wb_y_u002e_sroa_u002e_0_u002e_1_ = (_e351 + dot(vec3<f32>(_e690, _e696, _e702), vec3<f32>(cross(vec3<f32>(_e138, _e144, _e150), vec3<f32>(_e569, _e567, _e565)).x, cross(vec3<f32>(_e138, _e144, _e150), vec3<f32>(_e569, _e567, _e565)).y, cross(vec3<f32>(_e138, _e144, _e150), vec3<f32>(_e569, _e567, _e565)).z)));
        wb_x_u002e_sroa_u002e_0_u002e_1_ = (_e353 + dot(vec3<f32>(_e669, _e675, _e681), vec3<f32>(cross(vec3<f32>(_e138, _e144, _e150), vec3<f32>(_e569, _e567, _e565)).x, cross(vec3<f32>(_e138, _e144, _e150), vec3<f32>(_e569, _e567, _e565)).y, cross(vec3<f32>(_e138, _e144, _e150), vec3<f32>(_e569, _e567, _e565)).z)));
        vb_z_u002e_sroa_u002e_0_u002e_1_ = (_e355 + (_e162 * _e565));
        vb_y_u002e_sroa_u002e_0_u002e_1_ = (_e357 + (_e162 * _e567));
        vb_x_u002e_sroa_u002e_0_u002e_1_ = (_e359 + (_e162 * _e569));
        wa_z_u002e_sroa_u002e_0_u002e_1_ = (_e361 - dot(vec3<f32>(_e636, _e642, _e648), vec3<f32>(cross(vec3<f32>(_e120, _e126, _e132), vec3<f32>(_e569, _e567, _e565)).x, cross(vec3<f32>(_e120, _e126, _e132), vec3<f32>(_e569, _e567, _e565)).y, cross(vec3<f32>(_e120, _e126, _e132), vec3<f32>(_e569, _e567, _e565)).z)));
        wa_y_u002e_sroa_u002e_0_u002e_1_ = (_e363 - dot(vec3<f32>(_e615, _e621, _e627), vec3<f32>(cross(vec3<f32>(_e120, _e126, _e132), vec3<f32>(_e569, _e567, _e565)).x, cross(vec3<f32>(_e120, _e126, _e132), vec3<f32>(_e569, _e567, _e565)).y, cross(vec3<f32>(_e120, _e126, _e132), vec3<f32>(_e569, _e567, _e565)).z)));
        wa_x_u002e_sroa_u002e_0_u002e_1_ = (_e365 - dot(vec3<f32>(_e594, _e600, _e606), vec3<f32>(cross(vec3<f32>(_e120, _e126, _e132), vec3<f32>(_e569, _e567, _e565)).x, cross(vec3<f32>(_e120, _e126, _e132), vec3<f32>(_e569, _e567, _e565)).y, cross(vec3<f32>(_e120, _e126, _e132), vec3<f32>(_e569, _e567, _e565)).z)));
        va_z_u002e_sroa_u002e_0_u002e_1_ = (_e367 - (_e156 * _e565));
        va_y_u002e_sroa_u002e_0_u002e_1_ = (_e369 - (_e156 * _e567));
        va_x_u002e_sroa_u002e_0_u002e_1_ = (_e371 - (_e156 * _e569));
    } else {
        wb_z_u002e_sroa_u002e_0_u002e_1_ = _e349;
        wb_y_u002e_sroa_u002e_0_u002e_1_ = _e351;
        wb_x_u002e_sroa_u002e_0_u002e_1_ = _e353;
        vb_z_u002e_sroa_u002e_0_u002e_1_ = _e355;
        vb_y_u002e_sroa_u002e_0_u002e_1_ = _e357;
        vb_x_u002e_sroa_u002e_0_u002e_1_ = _e359;
        wa_z_u002e_sroa_u002e_0_u002e_1_ = _e361;
        wa_y_u002e_sroa_u002e_0_u002e_1_ = _e363;
        wa_x_u002e_sroa_u002e_0_u002e_1_ = _e365;
        va_z_u002e_sroa_u002e_0_u002e_1_ = _e367;
        va_y_u002e_sroa_u002e_0_u002e_1_ = _e369;
        va_x_u002e_sroa_u002e_0_u002e_1_ = _e371;
    }
    let _e770 = wb_z_u002e_sroa_u002e_0_u002e_1_;
    let _e772 = wb_y_u002e_sroa_u002e_0_u002e_1_;
    let _e774 = wb_x_u002e_sroa_u002e_0_u002e_1_;
    let _e776 = vb_z_u002e_sroa_u002e_0_u002e_1_;
    let _e778 = vb_y_u002e_sroa_u002e_0_u002e_1_;
    let _e780 = vb_x_u002e_sroa_u002e_0_u002e_1_;
    let _e782 = wa_z_u002e_sroa_u002e_0_u002e_1_;
    let _e784 = wa_y_u002e_sroa_u002e_0_u002e_1_;
    let _e786 = wa_x_u002e_sroa_u002e_0_u002e_1_;
    let _e788 = va_z_u002e_sroa_u002e_0_u002e_1_;
    let _e790 = va_y_u002e_sroa_u002e_0_u002e_1_;
    let _e792 = va_x_u002e_sroa_u002e_0_u002e_1_;
    if (((_e18 + 1u) == _e21) & (_e24 != 0u)) {
        let _e804 = storage_0_1_[((_e8 + local_invocation_index) + 276u)];
        if (_e804 > 0f) {
            let _e844 = (_e168 * (_e804 - dot(vec3<f32>(((_e780 + cross(vec3<f32>(_e774, _e772, _e770), vec3<f32>(_e138, _e144, _e150)).x) - (_e792 + cross(vec3<f32>(_e786, _e784, _e782), vec3<f32>(_e120, _e126, _e132)).x)), ((_e778 + cross(vec3<f32>(_e774, _e772, _e770), vec3<f32>(_e138, _e144, _e150)).y) - (_e790 + cross(vec3<f32>(_e786, _e784, _e782), vec3<f32>(_e120, _e126, _e132)).y)), ((_e776 + cross(vec3<f32>(_e774, _e772, _e770), vec3<f32>(_e138, _e144, _e150)).z) - (_e788 + cross(vec3<f32>(_e786, _e784, _e782), vec3<f32>(_e120, _e126, _e132)).z))), vec3<f32>(_e102, _e108, _e114))));
            if (_e844 > 0f) {
                let _e852 = storage_0_1_[((_e8 + local_invocation_index) + 156u)];
                let _e860 = storage_0_1_[((_e8 + local_invocation_index) + 160u)];
                let _e868 = storage_0_1_[((_e8 + local_invocation_index) + 164u)];
                let _e876 = storage_0_1_[((_e8 + local_invocation_index) + 180u)];
                let _e884 = storage_0_1_[((_e8 + local_invocation_index) + 184u)];
                let _e892 = storage_0_1_[((_e8 + local_invocation_index) + 188u)];
                let _e900 = storage_0_1_[((_e8 + local_invocation_index) + 168u)];
                let _e908 = storage_0_1_[((_e8 + local_invocation_index) + 172u)];
                let _e916 = storage_0_1_[((_e8 + local_invocation_index) + 176u)];
                let _e924 = storage_0_1_[((_e8 + local_invocation_index) + 192u)];
                let _e932 = storage_0_1_[((_e8 + local_invocation_index) + 196u)];
                let _e940 = storage_0_1_[((_e8 + local_invocation_index) + 200u)];
                wb_z_u002e_sroa_u002e_0_u002e_2_ = (_e770 + (_e844 * _e940));
                wb_y_u002e_sroa_u002e_0_u002e_2_ = (_e772 + (_e844 * _e932));
                wb_x_u002e_sroa_u002e_0_u002e_2_ = (_e774 + (_e844 * _e924));
                vb_z_u002e_sroa_u002e_0_u002e_2_ = (_e776 + (_e844 * _e916));
                vb_y_u002e_sroa_u002e_0_u002e_2_ = (_e778 + (_e844 * _e908));
                vb_x_u002e_sroa_u002e_0_u002e_2_ = (_e780 + (_e844 * _e900));
                wa_z_u002e_sroa_u002e_0_u002e_2_ = (_e782 - (_e844 * _e892));
                wa_y_u002e_sroa_u002e_0_u002e_2_ = (_e784 - (_e844 * _e884));
                wa_x_u002e_sroa_u002e_0_u002e_2_ = (_e786 - (_e844 * _e876));
                va_z_u002e_sroa_u002e_0_u002e_2_ = (_e788 - (_e844 * _e868));
                va_y_u002e_sroa_u002e_0_u002e_2_ = (_e790 - (_e844 * _e860));
                va_x_u002e_sroa_u002e_0_u002e_2_ = (_e792 - (_e844 * _e852));
            } else {
                wb_z_u002e_sroa_u002e_0_u002e_2_ = _e770;
                wb_y_u002e_sroa_u002e_0_u002e_2_ = _e772;
                wb_x_u002e_sroa_u002e_0_u002e_2_ = _e774;
                vb_z_u002e_sroa_u002e_0_u002e_2_ = _e776;
                vb_y_u002e_sroa_u002e_0_u002e_2_ = _e778;
                vb_x_u002e_sroa_u002e_0_u002e_2_ = _e780;
                wa_z_u002e_sroa_u002e_0_u002e_2_ = _e782;
                wa_y_u002e_sroa_u002e_0_u002e_2_ = _e784;
                wa_x_u002e_sroa_u002e_0_u002e_2_ = _e786;
                va_z_u002e_sroa_u002e_0_u002e_2_ = _e788;
                va_y_u002e_sroa_u002e_0_u002e_2_ = _e790;
                va_x_u002e_sroa_u002e_0_u002e_2_ = _e792;
            }
        } else {
            wb_z_u002e_sroa_u002e_0_u002e_2_ = _e770;
            wb_y_u002e_sroa_u002e_0_u002e_2_ = _e772;
            wb_x_u002e_sroa_u002e_0_u002e_2_ = _e774;
            vb_z_u002e_sroa_u002e_0_u002e_2_ = _e776;
            vb_y_u002e_sroa_u002e_0_u002e_2_ = _e778;
            vb_x_u002e_sroa_u002e_0_u002e_2_ = _e780;
            wa_z_u002e_sroa_u002e_0_u002e_2_ = _e782;
            wa_y_u002e_sroa_u002e_0_u002e_2_ = _e784;
            wa_x_u002e_sroa_u002e_0_u002e_2_ = _e786;
            va_z_u002e_sroa_u002e_0_u002e_2_ = _e788;
            va_y_u002e_sroa_u002e_0_u002e_2_ = _e790;
            va_x_u002e_sroa_u002e_0_u002e_2_ = _e792;
        }
    } else {
        wb_z_u002e_sroa_u002e_0_u002e_2_ = _e770;
        wb_y_u002e_sroa_u002e_0_u002e_2_ = _e772;
        wb_x_u002e_sroa_u002e_0_u002e_2_ = _e774;
        vb_z_u002e_sroa_u002e_0_u002e_2_ = _e776;
        vb_y_u002e_sroa_u002e_0_u002e_2_ = _e778;
        vb_x_u002e_sroa_u002e_0_u002e_2_ = _e780;
        wa_z_u002e_sroa_u002e_0_u002e_2_ = _e782;
        wa_y_u002e_sroa_u002e_0_u002e_2_ = _e784;
        wa_x_u002e_sroa_u002e_0_u002e_2_ = _e786;
        va_z_u002e_sroa_u002e_0_u002e_2_ = _e788;
        va_y_u002e_sroa_u002e_0_u002e_2_ = _e790;
        va_x_u002e_sroa_u002e_0_u002e_2_ = _e792;
    }
    let _e992 = wb_z_u002e_sroa_u002e_0_u002e_2_;
    let _e994 = wb_y_u002e_sroa_u002e_0_u002e_2_;
    let _e996 = wb_x_u002e_sroa_u002e_0_u002e_2_;
    let _e998 = vb_z_u002e_sroa_u002e_0_u002e_2_;
    let _e1000 = vb_y_u002e_sroa_u002e_0_u002e_2_;
    let _e1002 = vb_x_u002e_sroa_u002e_0_u002e_2_;
    let _e1004 = wa_z_u002e_sroa_u002e_0_u002e_2_;
    let _e1006 = wa_y_u002e_sroa_u002e_0_u002e_2_;
    let _e1008 = wa_x_u002e_sroa_u002e_0_u002e_2_;
    let _e1010 = va_z_u002e_sroa_u002e_0_u002e_2_;
    let _e1012 = va_y_u002e_sroa_u002e_0_u002e_2_;
    let _e1014 = va_x_u002e_sroa_u002e_0_u002e_2_;
    storage_0_0_[_e33] = _e1014;
    storage_0_0_[_e47] = _e1012;
    storage_0_0_[_e52] = _e1010;
    storage_0_0_[_e57] = _e1008;
    storage_0_0_[_e62] = _e1006;
    storage_0_0_[_e67] = _e1004;
    storage_0_0_[_e42] = _e1002;
    storage_0_0_[_e75] = _e1000;
    storage_0_0_[_e80] = _e998;
    storage_0_0_[_e85] = _e996;
    storage_0_0_[_e90] = _e994;
    storage_0_0_[_e95] = _e992;
    storage_0_1_[_e177] = _e223;
    storage_0_1_[_e450] = _e518;
    storage_0_1_[_e456] = _e571;
    return;
}
