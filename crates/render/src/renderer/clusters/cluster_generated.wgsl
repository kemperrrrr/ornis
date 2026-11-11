struct UniformStruct1_ {
    f0_: u32,
    f1_: u32,
    f2_: u32,
    f3_: u32,
}

struct UniformStruct0_ {
    f0_: vec4<f32>,
    f1_: vec4<f32>,
    f2_: vec4<f32>,
    f3_: vec4<f32>,
}

@group(0) @binding(1) 
var<uniform> uniform_struct_0_1_: UniformStruct1_;
@group(0) @binding(0) 
var<uniform> uniform_struct_0_0_: UniformStruct0_;
@group(0) @binding(2) 
var<storage> storage_0_2_: array<f32>;
@group(0) @binding(5) 
var<storage, read_write> atomic_storage_0_5_: array<atomic<u32>>;
@group(0) @binding(3) 
var<storage, read_write> storage_0_3_: array<u32>;
@group(0) @binding(4) 
var<storage, read_write> storage_0_4_: array<u32>;

@compute @workgroup_size(64, 1, 1) 
fn cs_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    var count_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa: u32;
    var _9_u002e_i5_u002e_sroa_u002e_0_u002e_0_: f32;
    var _9_u002e_i5_u002e_sroa_u002e_5_u002e_0_: f32;
    var _9_u002e_i5_u002e_sroa_u002e_6_u002e_0_: f32;
    var _9_u002e_i5_u002e_sroa_u002e_7_u002e_0_: f32;
    var _15_u002e_i_u002e_sroa_u002e_0_u002e_0_: f32;
    var _15_u002e_i_u002e_sroa_u002e_5_u002e_0_: f32;
    var _15_u002e_i_u002e_sroa_u002e_6_u002e_0_: f32;
    var _15_u002e_i_u002e_sroa_u002e_7_u002e_0_: f32;
    var z_lo_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_off0_u002e_i: u32;
    var _0_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var span_u002e_sroa_u002e_0_u002e_0_: u32;
    var head_u002e_sroa_u002e_0_u002e_0_u002e_i8473_: u32;
    var count_u002e_sroa_u002e_0_u002e_0_u002e_i472_: u32;
    var head_u002e_sroa_u002e_0_u002e_0_u002e_i478_: u32;
    var cursor_u002e_sroa_u002e_0_u002e_0_u002e_i477_: u32;
    var cursor_u002e_sroa_u002e_0_u002e_1_u002e_i: u32;
    var out_u002e_spec_u002e_select_u002e_i: u32;

    if (global_invocation_id.x > 3455u) {
        return;
    }
    let _e6 = uniform_struct_0_1_.f0_;
    let _e9 = uniform_struct_0_1_.f1_;
    let _e15 = (global_invocation_id.x / 144u);
    let _e19 = ((global_invocation_id.x << 1u) & 30u);
    let _e24 = ((f32(_e19) * 0.0625f) + -1f);
    let _e31 = ((f32((_e19 + 2u)) * 0.0625f) + -1f);
    let _e33 = (((global_invocation_id.x >> 4u) % 9u) << 1u);
    let _e38 = ((f32(_e33) / 9f) + -1f);
    let _e45 = ((f32((_e33 + 2u)) / 9f) + -1f);
    let _e49 = uniform_struct_0_0_.f0_.x;
    let _e53 = uniform_struct_0_0_.f1_.x;
    let _e57 = uniform_struct_0_0_.f2_.x;
    let _e61 = uniform_struct_0_0_.f3_.x;
    let _e65 = uniform_struct_0_0_.f0_.y;
    let _e69 = uniform_struct_0_0_.f1_.y;
    let _e73 = uniform_struct_0_0_.f2_.y;
    let _e77 = uniform_struct_0_0_.f3_.y;
    let _e81 = uniform_struct_0_0_.f0_.w;
    let _e85 = uniform_struct_0_0_.f1_.w;
    let _e89 = uniform_struct_0_0_.f2_.w;
    let _e93 = uniform_struct_0_0_.f3_.w;
    if (global_invocation_id.x < 144u) {
        z_lo_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
    } else {
        z_lo_u002e_sroa_u002e_0_u002e_0_u002e_i = (exp(((f32((_e15 + 4294967295u)) / 23f) * 6.684612f)) * 0.5f);
    }
    let _e110 = z_lo_u002e_sroa_u002e_0_u002e_0_u002e_i;
    if (global_invocation_id.x > 3311u) {
        _0_u002e_sroa_u002e_0_u002e_0_u002e_i = 1000000000f;
    } else {
        _0_u002e_sroa_u002e_0_u002e_0_u002e_i = (exp(((f32(_e15) / 23f) * 6.684612f)) * 0.5f);
    }
    let _e125 = _0_u002e_sroa_u002e_0_u002e_0_u002e_i;
    let _e127 = (_e49 - (_e24 * _e81));
    let _e129 = (_e53 - (_e24 * _e85));
    let _e131 = (_e57 - (_e24 * _e89));
    let _e133 = (_e61 - (_e24 * _e93));
    let _e135 = ((_e31 * _e81) - _e49);
    let _e137 = ((_e31 * _e85) - _e53);
    let _e139 = ((_e31 * _e89) - _e57);
    let _e141 = ((_e31 * _e93) - _e61);
    let _e143 = (_e65 - (_e38 * _e81));
    let _e145 = (_e69 - (_e38 * _e85));
    let _e147 = (_e73 - (_e38 * _e89));
    let _e149 = (_e77 - (_e38 * _e93));
    let _e151 = ((_e45 * _e81) - _e65);
    let _e153 = ((_e45 * _e85) - _e69);
    let _e155 = ((_e45 * _e89) - _e73);
    let _e157 = ((_e45 * _e93) - _e77);
    let _e158 = (_e93 - _e110);
    let _e159 = -(_e81);
    let _e160 = -(_e85);
    let _e161 = -(_e89);
    let _e162 = (_e125 - _e93);
    if (_e9 < _e6) {
        head_u002e_sroa_u002e_0_u002e_0_u002e_i8473_ = _e9;
        count_u002e_sroa_u002e_0_u002e_0_u002e_i472_ = 0u;
        loop {
            let _e168 = head_u002e_sroa_u002e_0_u002e_0_u002e_i8473_;
            let _e170 = count_u002e_sroa_u002e_0_u002e_0_u002e_i472_;
            let _e172 = (_e168 * 36u);
            let _e177 = storage_0_2_[(_e172 + 8u)];
            let _e182 = storage_0_2_[(_e172 + 9u)];
            let _e187 = storage_0_2_[(_e172 + 10u)];
            let _e192 = storage_0_2_[(_e172 + 16u)];
            let _e197 = storage_0_2_[(_e172 + 18u)];
            let _e200 = storage_0_2_[_e172];
            if ((_e197 < 0.1f) | (_e200 < 2f)) {
                _9_u002e_i5_u002e_sroa_u002e_0_u002e_0_ = _e177;
                _9_u002e_i5_u002e_sroa_u002e_5_u002e_0_ = _e182;
                _9_u002e_i5_u002e_sroa_u002e_6_u002e_0_ = _e187;
                _9_u002e_i5_u002e_sroa_u002e_7_u002e_0_ = _e192;
            } else {
                let _e210 = (_e197 * _e197);
                let _e218 = ((_e192 * (((1f - _e210) / _e210) + 1f)) * 0.5f);
                let _e223 = storage_0_2_[(_e172 + 4u)];
                let _e228 = storage_0_2_[(_e172 + 5u)];
                let _e233 = storage_0_2_[(_e172 + 6u)];
                let _e235 = length(vec3<f32>(_e223, _e228, _e233));
                _9_u002e_i5_u002e_sroa_u002e_0_u002e_0_ = (_e177 + (_e218 * (_e223 / _e235)));
                _9_u002e_i5_u002e_sroa_u002e_5_u002e_0_ = (_e182 + (_e218 * (_e228 / _e235)));
                _9_u002e_i5_u002e_sroa_u002e_6_u002e_0_ = (_e187 + (_e218 * (_e233 / _e235)));
                _9_u002e_i5_u002e_sroa_u002e_7_u002e_0_ = _e218;
            }
            let _e250 = _9_u002e_i5_u002e_sroa_u002e_0_u002e_0_;
            let _e252 = _9_u002e_i5_u002e_sroa_u002e_5_u002e_0_;
            let _e254 = _9_u002e_i5_u002e_sroa_u002e_6_u002e_0_;
            let _e256 = _9_u002e_i5_u002e_sroa_u002e_7_u002e_0_;
            if ((_e133 + dot(vec3<f32>(_e127, _e129, _e131), vec3<f32>(_e250, _e252, _e254))) < (_e256 * -(length(vec3<f32>(_e127, _e129, _e131))))) {
                _0_u002e_sroa_u002e_0_u002e_0_u002e_off0_u002e_i = 0u;
            } else {
                if ((_e141 + dot(vec3<f32>(_e135, _e137, _e139), vec3<f32>(_e250, _e252, _e254))) < (_e256 * -(length(vec3<f32>(_e135, _e137, _e139))))) {
                    _0_u002e_sroa_u002e_0_u002e_0_u002e_off0_u002e_i = 0u;
                } else {
                    if ((_e149 + dot(vec3<f32>(_e143, _e145, _e147), vec3<f32>(_e250, _e252, _e254))) < (_e256 * -(length(vec3<f32>(_e143, _e145, _e147))))) {
                        _0_u002e_sroa_u002e_0_u002e_0_u002e_off0_u002e_i = 0u;
                    } else {
                        if ((_e157 + dot(vec3<f32>(_e151, _e153, _e155), vec3<f32>(_e250, _e252, _e254))) < (_e256 * -(length(vec3<f32>(_e151, _e153, _e155))))) {
                            _0_u002e_sroa_u002e_0_u002e_0_u002e_off0_u002e_i = 0u;
                        } else {
                            if ((_e158 + dot(vec3<f32>(_e81, _e85, _e89), vec3<f32>(_e250, _e252, _e254))) < (_e256 * -(length(vec3<f32>(_e81, _e85, _e89))))) {
                                _0_u002e_sroa_u002e_0_u002e_0_u002e_off0_u002e_i = 0u;
                            } else {
                                _0_u002e_sroa_u002e_0_u002e_0_u002e_off0_u002e_i = select(0u, 1u, !(((_e162 + dot(vec3<f32>(_e159, _e160, _e161), vec3<f32>(_e250, _e252, _e254))) < (_e256 * -(length(vec3<f32>(_e159, _e160, _e161)))))));
                            }
                        }
                    }
                }
            }
            let _e327 = _0_u002e_sroa_u002e_0_u002e_0_u002e_off0_u002e_i;
            let _e328 = (_e327 + _e170);
            out_u002e_spec_u002e_select_u002e_i = _e328;
            let _e331 = (_e168 + 1u);
            if (_e331 == _e6) {
                break;
            } else {
                head_u002e_sroa_u002e_0_u002e_0_u002e_i8473_ = _e331;
                count_u002e_sroa_u002e_0_u002e_0_u002e_i472_ = _e328;
                continue;
            }
        }
        let _e336 = out_u002e_spec_u002e_select_u002e_i;
        count_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = min(_e336, 128u);
    } else {
        count_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = 0u;
    }
    let _e343 = count_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa;
    let _e347 = atomicAdd((&atomic_storage_0_5_[0u]), _e343);
    if ((_e347 + _e343) > 442368u) {
        let _e356 = select((442368u - _e347), 0u, (_e347 > 442367u));
        let _e361 = atomicAdd((&atomic_storage_0_5_[1u]), (_e343 - _e356));
        span_u002e_sroa_u002e_0_u002e_0_ = _e356;
    } else {
        span_u002e_sroa_u002e_0_u002e_0_ = _e343;
    }
    let _e365 = span_u002e_sroa_u002e_0_u002e_0_;
    let _e367 = (global_invocation_id.x << 1u);
    storage_0_3_[_e367] = _e347;
    storage_0_3_[(_e367 | 1u)] = _e365;
    if ((_e9 >= _e6) | (_e365 == 0u)) {
        return;
    } else {
        head_u002e_sroa_u002e_0_u002e_0_u002e_i478_ = _e9;
        cursor_u002e_sroa_u002e_0_u002e_0_u002e_i477_ = 0u;
    }
    loop {
        let _e382 = head_u002e_sroa_u002e_0_u002e_0_u002e_i478_;
        let _e384 = cursor_u002e_sroa_u002e_0_u002e_0_u002e_i477_;
        let _e386 = (_e382 * 36u);
        let _e391 = storage_0_2_[(_e386 + 8u)];
        let _e396 = storage_0_2_[(_e386 + 9u)];
        let _e401 = storage_0_2_[(_e386 + 10u)];
        let _e406 = storage_0_2_[(_e386 + 16u)];
        let _e411 = storage_0_2_[(_e386 + 18u)];
        let _e414 = storage_0_2_[_e386];
        if ((_e411 < 0.1f) | (_e414 < 2f)) {
            _15_u002e_i_u002e_sroa_u002e_0_u002e_0_ = _e391;
            _15_u002e_i_u002e_sroa_u002e_5_u002e_0_ = _e396;
            _15_u002e_i_u002e_sroa_u002e_6_u002e_0_ = _e401;
            _15_u002e_i_u002e_sroa_u002e_7_u002e_0_ = _e406;
        } else {
            let _e424 = (_e411 * _e411);
            let _e432 = ((_e406 * (((1f - _e424) / _e424) + 1f)) * 0.5f);
            let _e437 = storage_0_2_[(_e386 + 4u)];
            let _e442 = storage_0_2_[(_e386 + 5u)];
            let _e447 = storage_0_2_[(_e386 + 6u)];
            let _e449 = length(vec3<f32>(_e437, _e442, _e447));
            _15_u002e_i_u002e_sroa_u002e_0_u002e_0_ = (_e391 + (_e432 * (_e437 / _e449)));
            _15_u002e_i_u002e_sroa_u002e_5_u002e_0_ = (_e396 + (_e432 * (_e442 / _e449)));
            _15_u002e_i_u002e_sroa_u002e_6_u002e_0_ = (_e401 + (_e432 * (_e447 / _e449)));
            _15_u002e_i_u002e_sroa_u002e_7_u002e_0_ = _e432;
        }
        let _e464 = _15_u002e_i_u002e_sroa_u002e_0_u002e_0_;
        let _e466 = _15_u002e_i_u002e_sroa_u002e_5_u002e_0_;
        let _e468 = _15_u002e_i_u002e_sroa_u002e_6_u002e_0_;
        let _e470 = _15_u002e_i_u002e_sroa_u002e_7_u002e_0_;
        if ((_e133 + dot(vec3<f32>(_e127, _e129, _e131), vec3<f32>(_e464, _e466, _e468))) < (_e470 * -(length(vec3<f32>(_e127, _e129, _e131))))) {
            cursor_u002e_sroa_u002e_0_u002e_1_u002e_i = _e384;
        } else {
            if ((_e141 + dot(vec3<f32>(_e135, _e137, _e139), vec3<f32>(_e464, _e466, _e468))) < (_e470 * -(length(vec3<f32>(_e135, _e137, _e139))))) {
                cursor_u002e_sroa_u002e_0_u002e_1_u002e_i = _e384;
            } else {
                if ((_e149 + dot(vec3<f32>(_e143, _e145, _e147), vec3<f32>(_e464, _e466, _e468))) < (_e470 * -(length(vec3<f32>(_e143, _e145, _e147))))) {
                    cursor_u002e_sroa_u002e_0_u002e_1_u002e_i = _e384;
                } else {
                    if ((_e157 + dot(vec3<f32>(_e151, _e153, _e155), vec3<f32>(_e464, _e466, _e468))) < (_e470 * -(length(vec3<f32>(_e151, _e153, _e155))))) {
                        cursor_u002e_sroa_u002e_0_u002e_1_u002e_i = _e384;
                    } else {
                        if ((_e158 + dot(vec3<f32>(_e81, _e85, _e89), vec3<f32>(_e464, _e466, _e468))) < (_e470 * -(length(vec3<f32>(_e81, _e85, _e89))))) {
                            cursor_u002e_sroa_u002e_0_u002e_1_u002e_i = _e384;
                        } else {
                            if ((_e162 + dot(vec3<f32>(_e159, _e160, _e161), vec3<f32>(_e464, _e466, _e468))) < (_e470 * -(length(vec3<f32>(_e159, _e160, _e161))))) {
                                cursor_u002e_sroa_u002e_0_u002e_1_u002e_i = _e384;
                            } else {
                                storage_0_4_[(_e384 + _e347)] = _e382;
                                cursor_u002e_sroa_u002e_0_u002e_1_u002e_i = (_e384 + 1u);
                            }
                        }
                    }
                }
            }
        }
        let _e538 = cursor_u002e_sroa_u002e_0_u002e_1_u002e_i;
        let _e540 = (_e382 + 1u);
        if select((_e538 >= _e365), true, (_e540 >= _e6)) {
            break;
        } else {
            head_u002e_sroa_u002e_0_u002e_0_u002e_i478_ = _e540;
            cursor_u002e_sroa_u002e_0_u002e_0_u002e_i477_ = _e538;
            continue;
        }
    }
    return;
}
