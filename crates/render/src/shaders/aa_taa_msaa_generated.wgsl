struct FragmentOut {
    @location(0) o0_: vec4<f32>,
    @location(1) o1_: vec4<f32>,
}

struct UniformStruct0_ {
    f0_: f32,
    f1_: f32,
    f2_: f32,
    f3_: f32,
    f4_: u32,
    f5_: f32,
    f6_: f32,
    f7_: f32,
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) o0_: vec2<f32>,
}

@group(0) @binding(0) 
var tex_0_0_: texture_2d<f32>;
@group(0) @binding(1) 
var sampler_0_1_: sampler;
@group(0) @binding(2) 
var tex_0_2_: texture_depth_multisampled_2d;
@group(0) @binding(3) 
var tex_0_3_: texture_2d<f32>;
@group(0) @binding(4) 
var tex_0_4_: texture_2d<u32>;
@group(0) @binding(6) 
var<uniform> uniform_struct_0_6_: UniformStruct0_;
@group(0) @binding(5) 
var tex_0_5_: texture_2d<f32>;

@fragment 
fn fs_main(@builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>) -> FragmentOut {
    var fragment_out: FragmentOut;
    var bounds_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var bounds_u002e_sroa_u002e_6_u002e_0_u002e_lcssa: f32;
    var bounds_u002e_sroa_u002e_10_u002e_0_u002e_lcssa: f32;
    var bounds_u002e_sroa_u002e_14_u002e_0_u002e_lcssa: f32;
    var bounds_u002e_sroa_u002e_18_u002e_0_u002e_lcssa: f32;
    var bounds_u002e_sroa_u002e_22_u002e_0_u002e_lcssa: f32;
    var count_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var sum_sq_cg_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var sum_sq_co_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var sum_sq_y_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var sum_cg_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var sum_co_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var sum_y_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var sum_y_u002e_sroa_u002e_0_u002e_0_: f32;
    var sum_co_u002e_sroa_u002e_0_u002e_0_: f32;
    var sum_cg_u002e_sroa_u002e_0_u002e_0_: f32;
    var sum_sq_y_u002e_sroa_u002e_0_u002e_0_: f32;
    var sum_sq_co_u002e_sroa_u002e_0_u002e_0_: f32;
    var sum_sq_cg_u002e_sroa_u002e_0_u002e_0_: f32;
    var count_u002e_sroa_u002e_0_u002e_0_: f32;
    var dy_u002e_sroa_u002e_0_u002e_0_: u32;
    var bounds_u002e_sroa_u002e_22_u002e_093_: f32;
    var bounds_u002e_sroa_u002e_18_u002e_092_: f32;
    var bounds_u002e_sroa_u002e_14_u002e_091_: f32;
    var bounds_u002e_sroa_u002e_10_u002e_090_: f32;
    var bounds_u002e_sroa_u002e_6_u002e_089_: f32;
    var bounds_u002e_sroa_u002e_0_u002e_088_: f32;
    var result_b_u002e_sroa_u002e_0_u002e_0_: f32;
    var result_g_u002e_sroa_u002e_0_u002e_0_: f32;
    var result_r_u002e_sroa_u002e_0_u002e_0_: f32;
    var sum_y_u002e_sroa_u002e_0_u002e_1_: f32;
    var sum_co_u002e_sroa_u002e_0_u002e_1_: f32;
    var sum_cg_u002e_sroa_u002e_0_u002e_1_: f32;
    var sum_sq_y_u002e_sroa_u002e_0_u002e_1_: f32;
    var sum_sq_co_u002e_sroa_u002e_0_u002e_1_: f32;
    var sum_sq_cg_u002e_sroa_u002e_0_u002e_1_: f32;
    var count_u002e_sroa_u002e_0_u002e_1_: f32;
    var dx_u002e_sroa_u002e_0_u002e_0_: u32;
    var bounds_u002e_sroa_u002e_22_u002e_166_: f32;
    var bounds_u002e_sroa_u002e_18_u002e_165_: f32;
    var bounds_u002e_sroa_u002e_14_u002e_164_: f32;
    var bounds_u002e_sroa_u002e_10_u002e_163_: f32;
    var bounds_u002e_sroa_u002e_6_u002e_162_: f32;
    var bounds_u002e_sroa_u002e_0_u002e_161_: f32;
    var out_u002e_6_: f32;
    var out_u002e_7_: f32;
    var out_u002e_8_: f32;
    var out_u002e_9_: f32;
    var out_u002e_10_: f32;
    var out_u002e_11_: f32;
    var out_u002e_18_: f32;
    var out_u002e_17_: f32;
    var out_u002e_16_: f32;
    var out_u002e_15_: f32;
    var out_u002e_14_: f32;
    var out_u002e_13_: f32;
    var out_u002e_12_: f32;

    let _e5 = textureDimensions(tex_0_0_);
    let _e8 = f32(bitcast<i32>(_e5.x));
    let _e11 = f32(bitcast<i32>(_e5.y));
    let _e13 = u32((uv.x * _e8));
    let _e15 = u32((uv.y * _e11));
    let _e19 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e30 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0u);
    if !((_e30 >= 1f)) {
        let _e39 = textureLoad(tex_0_3_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0i);
        let _e45 = textureLoad(tex_0_3_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0i);
        let _e52 = textureLoad(tex_0_4_, vec2<i32>(bitcast<i32>(_e13), bitcast<i32>(_e15)), 0i);
        let _e54 = (uv.x - _e39.x);
        let _e55 = (uv.y - _e45.y);
        if ((_e54 >= 0f) & (_e54 <= 1f)) {
            if ((_e52.x != 65535u) & ((_e55 >= 0f) & (_e55 <= 1f))) {
                let _e71 = uniform_struct_0_6_.f2_;
                let _e73 = bitcast<u32>(i32(_e71));
                let _e75 = (0u - _e73);
                if (bitcast<i32>(_e73) < bitcast<i32>(_e75)) {
                    bounds_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 340282350000000000000000000000000000000f;
                    bounds_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = 340282350000000000000000000000000000000f;
                    bounds_u002e_sroa_u002e_10_u002e_0_u002e_lcssa = 340282350000000000000000000000000000000f;
                    bounds_u002e_sroa_u002e_14_u002e_0_u002e_lcssa = -340282350000000000000000000000000000000f;
                    bounds_u002e_sroa_u002e_18_u002e_0_u002e_lcssa = -340282350000000000000000000000000000000f;
                    bounds_u002e_sroa_u002e_22_u002e_0_u002e_lcssa = -340282350000000000000000000000000000000f;
                    count_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
                    sum_sq_cg_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
                    sum_sq_co_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
                    sum_sq_y_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
                    sum_cg_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
                    sum_co_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
                    sum_y_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 0f;
                } else {
                    sum_y_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    sum_co_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    sum_cg_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    sum_sq_y_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    sum_sq_co_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    sum_sq_cg_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    count_u002e_sroa_u002e_0_u002e_0_ = 0f;
                    dy_u002e_sroa_u002e_0_u002e_0_ = _e75;
                    bounds_u002e_sroa_u002e_22_u002e_093_ = -340282350000000000000000000000000000000f;
                    bounds_u002e_sroa_u002e_18_u002e_092_ = -340282350000000000000000000000000000000f;
                    bounds_u002e_sroa_u002e_14_u002e_091_ = -340282350000000000000000000000000000000f;
                    bounds_u002e_sroa_u002e_10_u002e_090_ = 340282350000000000000000000000000000000f;
                    bounds_u002e_sroa_u002e_6_u002e_089_ = 340282350000000000000000000000000000000f;
                    bounds_u002e_sroa_u002e_0_u002e_088_ = 340282350000000000000000000000000000000f;
                    loop {
                        let _e133 = sum_y_u002e_sroa_u002e_0_u002e_0_;
                        let _e135 = sum_co_u002e_sroa_u002e_0_u002e_0_;
                        let _e137 = sum_cg_u002e_sroa_u002e_0_u002e_0_;
                        let _e139 = sum_sq_y_u002e_sroa_u002e_0_u002e_0_;
                        let _e141 = sum_sq_co_u002e_sroa_u002e_0_u002e_0_;
                        let _e143 = sum_sq_cg_u002e_sroa_u002e_0_u002e_0_;
                        let _e145 = count_u002e_sroa_u002e_0_u002e_0_;
                        let _e147 = dy_u002e_sroa_u002e_0_u002e_0_;
                        let _e149 = bounds_u002e_sroa_u002e_22_u002e_093_;
                        let _e151 = bounds_u002e_sroa_u002e_18_u002e_092_;
                        let _e153 = bounds_u002e_sroa_u002e_14_u002e_091_;
                        let _e155 = bounds_u002e_sroa_u002e_10_u002e_090_;
                        let _e157 = bounds_u002e_sroa_u002e_6_u002e_089_;
                        let _e159 = bounds_u002e_sroa_u002e_0_u002e_088_;
                        sum_y_u002e_sroa_u002e_0_u002e_1_ = _e133;
                        sum_co_u002e_sroa_u002e_0_u002e_1_ = _e135;
                        sum_cg_u002e_sroa_u002e_0_u002e_1_ = _e137;
                        sum_sq_y_u002e_sroa_u002e_0_u002e_1_ = _e139;
                        sum_sq_co_u002e_sroa_u002e_0_u002e_1_ = _e141;
                        sum_sq_cg_u002e_sroa_u002e_0_u002e_1_ = _e143;
                        count_u002e_sroa_u002e_0_u002e_1_ = _e145;
                        dx_u002e_sroa_u002e_0_u002e_0_ = _e75;
                        bounds_u002e_sroa_u002e_22_u002e_166_ = _e149;
                        bounds_u002e_sroa_u002e_18_u002e_165_ = _e151;
                        bounds_u002e_sroa_u002e_14_u002e_164_ = _e153;
                        bounds_u002e_sroa_u002e_10_u002e_163_ = _e155;
                        bounds_u002e_sroa_u002e_6_u002e_162_ = _e157;
                        bounds_u002e_sroa_u002e_0_u002e_161_ = _e159;
                        loop {
                            let _e179 = sum_y_u002e_sroa_u002e_0_u002e_1_;
                            let _e181 = sum_co_u002e_sroa_u002e_0_u002e_1_;
                            let _e183 = sum_cg_u002e_sroa_u002e_0_u002e_1_;
                            let _e185 = sum_sq_y_u002e_sroa_u002e_0_u002e_1_;
                            let _e187 = sum_sq_co_u002e_sroa_u002e_0_u002e_1_;
                            let _e189 = sum_sq_cg_u002e_sroa_u002e_0_u002e_1_;
                            let _e191 = count_u002e_sroa_u002e_0_u002e_1_;
                            let _e193 = dx_u002e_sroa_u002e_0_u002e_0_;
                            let _e195 = bounds_u002e_sroa_u002e_22_u002e_166_;
                            let _e197 = bounds_u002e_sroa_u002e_18_u002e_165_;
                            let _e199 = bounds_u002e_sroa_u002e_14_u002e_164_;
                            let _e201 = bounds_u002e_sroa_u002e_10_u002e_163_;
                            let _e203 = bounds_u002e_sroa_u002e_6_u002e_162_;
                            let _e205 = bounds_u002e_sroa_u002e_0_u002e_161_;
                            let _e220 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(clamp((uv.x + (f32(bitcast<i32>(_e193)) / _e8)), 0f, 1f), clamp((uv.y + (f32(bitcast<i32>(_e147)) / _e11)), 0f, 1f)), 0f);
                            let _e229 = (_e220.y * 0.5f);
                            let _e232 = (_e220.z * 0.25f);
                            let _e233 = (((_e220.x * 0.25f) + _e229) + _e232);
                            let _e238 = ((_e220.x * 0.5f) - (_e220.z * 0.5f));
                            let _e242 = ((_e229 - (_e220.x * 0.25f)) - _e232);
                            let _e243 = min(_e205, _e233);
                            out_u002e_6_ = _e243;
                            let _e245 = min(_e203, _e238);
                            out_u002e_7_ = _e245;
                            let _e247 = min(_e201, _e242);
                            out_u002e_8_ = _e247;
                            let _e249 = max(_e199, _e233);
                            out_u002e_9_ = _e249;
                            let _e251 = max(_e197, _e238);
                            out_u002e_10_ = _e251;
                            let _e253 = max(_e195, _e242);
                            out_u002e_11_ = _e253;
                            let _e255 = (_e179 + _e233);
                            out_u002e_12_ = _e255;
                            let _e257 = (_e181 + _e238);
                            out_u002e_13_ = _e257;
                            let _e259 = (_e183 + _e242);
                            out_u002e_14_ = _e259;
                            let _e262 = (_e185 + (_e233 * _e233));
                            out_u002e_15_ = _e262;
                            let _e265 = (_e187 + (_e238 * _e238));
                            out_u002e_16_ = _e265;
                            let _e268 = (_e189 + (_e242 * _e242));
                            out_u002e_17_ = _e268;
                            let _e271 = (_e191 + 1f);
                            out_u002e_18_ = _e271;
                            let _e274 = (_e193 + 1u);
                            if (bitcast<i32>(_e274) > bitcast<i32>(_e73)) {
                                break;
                            } else {
                                sum_y_u002e_sroa_u002e_0_u002e_1_ = _e255;
                                sum_co_u002e_sroa_u002e_0_u002e_1_ = _e257;
                                sum_cg_u002e_sroa_u002e_0_u002e_1_ = _e259;
                                sum_sq_y_u002e_sroa_u002e_0_u002e_1_ = _e262;
                                sum_sq_co_u002e_sroa_u002e_0_u002e_1_ = _e265;
                                sum_sq_cg_u002e_sroa_u002e_0_u002e_1_ = _e268;
                                count_u002e_sroa_u002e_0_u002e_1_ = _e271;
                                dx_u002e_sroa_u002e_0_u002e_0_ = _e274;
                                bounds_u002e_sroa_u002e_22_u002e_166_ = _e253;
                                bounds_u002e_sroa_u002e_18_u002e_165_ = _e251;
                                bounds_u002e_sroa_u002e_14_u002e_164_ = _e249;
                                bounds_u002e_sroa_u002e_10_u002e_163_ = _e247;
                                bounds_u002e_sroa_u002e_6_u002e_162_ = _e245;
                                bounds_u002e_sroa_u002e_0_u002e_161_ = _e243;
                                continue;
                            }
                        }
                        let _e293 = out_u002e_12_;
                        let _e295 = out_u002e_13_;
                        let _e297 = out_u002e_14_;
                        let _e299 = out_u002e_15_;
                        let _e301 = out_u002e_16_;
                        let _e303 = out_u002e_17_;
                        let _e305 = out_u002e_18_;
                        let _e307 = out_u002e_11_;
                        let _e309 = out_u002e_10_;
                        let _e311 = out_u002e_9_;
                        let _e313 = out_u002e_8_;
                        let _e315 = out_u002e_7_;
                        let _e317 = out_u002e_6_;
                        let _e319 = (_e147 + 1u);
                        if (bitcast<i32>(_e319) > bitcast<i32>(_e73)) {
                            bounds_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e317;
                            bounds_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e315;
                            bounds_u002e_sroa_u002e_10_u002e_0_u002e_lcssa = _e313;
                            bounds_u002e_sroa_u002e_14_u002e_0_u002e_lcssa = _e311;
                            bounds_u002e_sroa_u002e_18_u002e_0_u002e_lcssa = _e309;
                            bounds_u002e_sroa_u002e_22_u002e_0_u002e_lcssa = _e307;
                            count_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e305;
                            sum_sq_cg_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e303;
                            sum_sq_co_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e301;
                            sum_sq_y_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e299;
                            sum_cg_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e297;
                            sum_co_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e295;
                            sum_y_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e293;
                            break;
                        } else {
                            sum_y_u002e_sroa_u002e_0_u002e_0_ = _e293;
                            sum_co_u002e_sroa_u002e_0_u002e_0_ = _e295;
                            sum_cg_u002e_sroa_u002e_0_u002e_0_ = _e297;
                            sum_sq_y_u002e_sroa_u002e_0_u002e_0_ = _e299;
                            sum_sq_co_u002e_sroa_u002e_0_u002e_0_ = _e301;
                            sum_sq_cg_u002e_sroa_u002e_0_u002e_0_ = _e303;
                            count_u002e_sroa_u002e_0_u002e_0_ = _e305;
                            dy_u002e_sroa_u002e_0_u002e_0_ = _e319;
                            bounds_u002e_sroa_u002e_22_u002e_093_ = _e307;
                            bounds_u002e_sroa_u002e_18_u002e_092_ = _e309;
                            bounds_u002e_sroa_u002e_14_u002e_091_ = _e311;
                            bounds_u002e_sroa_u002e_10_u002e_090_ = _e313;
                            bounds_u002e_sroa_u002e_6_u002e_089_ = _e315;
                            bounds_u002e_sroa_u002e_0_u002e_088_ = _e317;
                            continue;
                        }
                    }
                }
                let _e377 = bounds_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
                let _e379 = bounds_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
                let _e381 = bounds_u002e_sroa_u002e_10_u002e_0_u002e_lcssa;
                let _e383 = bounds_u002e_sroa_u002e_14_u002e_0_u002e_lcssa;
                let _e385 = bounds_u002e_sroa_u002e_18_u002e_0_u002e_lcssa;
                let _e387 = bounds_u002e_sroa_u002e_22_u002e_0_u002e_lcssa;
                let _e389 = count_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
                let _e391 = sum_sq_cg_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
                let _e393 = sum_sq_co_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
                let _e395 = sum_sq_y_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
                let _e397 = sum_cg_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
                let _e399 = sum_co_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
                let _e401 = sum_y_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
                let _e402 = (_e401 / _e389);
                let _e403 = (_e399 / _e389);
                let _e404 = (_e397 / _e389);
                let _e424 = (sqrt(max(((_e395 / _e389) - (_e402 * _e402)), 0f)) * 1.25f);
                let _e428 = (sqrt(max(((_e393 / _e389) - (_e403 * _e403)), 0f)) * 1.25f);
                let _e432 = (sqrt(max(((_e391 / _e389) - (_e404 * _e404)), 0f)) * 1.25f);
                let _e443 = uniform_struct_0_6_.f3_;
                if (_e443 > 0.5f) {
                    let _e453 = textureSampleLevel(tex_0_5_, sampler_0_1_, vec2<f32>(_e54, _e55), 0f);
                    if (abs((_e453.w - (f32(_e52.x) / 65534f))) < 0.002f) {
                        let _e468 = (_e453.y * 0.5f);
                        let _e471 = (_e453.z * 0.25f);
                        let _e473 = clamp((((_e453.x * 0.25f) + _e468) + _e471), max(_e377, (_e402 - _e424)), min(_e383, (_e402 + _e424)));
                        let _e479 = clamp(((_e453.x * 0.5f) - (_e453.z * 0.5f)), max(_e379, (_e403 - _e428)), min(_e385, (_e403 + _e428)));
                        let _e484 = clamp(((_e468 - (_e453.x * 0.25f)) - _e471), max(_e381, (_e404 - _e432)), min(_e387, (_e404 + _e432)));
                        let _e487 = uniform_struct_0_6_.f0_;
                        result_b_u002e_sroa_u002e_0_u002e_0_ = mix(_e19.z, ((_e473 - _e479) - _e484), _e487);
                        result_g_u002e_sroa_u002e_0_u002e_0_ = mix(_e19.y, (_e473 + _e484), _e487);
                        result_r_u002e_sroa_u002e_0_u002e_0_ = mix(_e19.x, ((_e473 + _e479) - _e484), _e487);
                    } else {
                        result_b_u002e_sroa_u002e_0_u002e_0_ = _e19.z;
                        result_g_u002e_sroa_u002e_0_u002e_0_ = _e19.y;
                        result_r_u002e_sroa_u002e_0_u002e_0_ = _e19.x;
                    }
                } else {
                    result_b_u002e_sroa_u002e_0_u002e_0_ = _e19.z;
                    result_g_u002e_sroa_u002e_0_u002e_0_ = _e19.y;
                    result_r_u002e_sroa_u002e_0_u002e_0_ = _e19.x;
                }
                let _e506 = result_b_u002e_sroa_u002e_0_u002e_0_;
                let _e508 = result_g_u002e_sroa_u002e_0_u002e_0_;
                let _e510 = result_r_u002e_sroa_u002e_0_u002e_0_;
                fragment_out.o0_ = vec4<f32>(_e510, _e508, _e506, 1f);
                fragment_out.o1_ = vec4<f32>(_e510, _e508, _e506, _e30);
            } else {
                fragment_out.o0_ = vec4<f32>(_e19.x, _e19.y, _e19.z, 1f);
                fragment_out.o1_ = vec4<f32>(_e19.x, _e19.y, _e19.z, _e30);
            }
        } else {
            fragment_out.o0_ = vec4<f32>(_e19.x, _e19.y, _e19.z, 1f);
            fragment_out.o1_ = vec4<f32>(_e19.x, _e19.y, _e19.z, _e30);
        }
    } else {
        fragment_out.o0_ = vec4<f32>(_e19.x, _e19.y, _e19.z, 1f);
        fragment_out.o1_ = vec4<f32>(_e19.x, _e19.y, _e19.z, 1f);
    }
    let _e541 = fragment_out;
    return _e541;
}

@vertex 
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var vertex_out: VertexOut;
    var _11_u002e_sroa_u002e_0_u002e_0_: f32;
    var _10_u002e_sroa_u002e_0_u002e_2_: f32;

    let _e6 = ((vertex_index & 2u) == 0u);
    if ((vertex_index & 1u) == 0u) {
        if _e6 {
            vertex_out.position = vec4<f32>(-1f, -1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_ = 1f;
            _10_u002e_sroa_u002e_0_u002e_2_ = 0f;
        } else {
            vertex_out.position = vec4<f32>(-1f, 1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_ = 0f;
            _10_u002e_sroa_u002e_0_u002e_2_ = 0f;
        }
    } else {
        if _e6 {
            vertex_out.position = vec4<f32>(1f, -1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_ = 1f;
            _10_u002e_sroa_u002e_0_u002e_2_ = 1f;
        } else {
            vertex_out.position = vec4<f32>(1f, 1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_ = 0f;
            _10_u002e_sroa_u002e_0_u002e_2_ = 1f;
        }
    }
    let _e54 = _11_u002e_sroa_u002e_0_u002e_0_;
    let _e56 = _10_u002e_sroa_u002e_0_u002e_2_;
    vertex_out.o0_ = vec2<f32>(_e56, _e54);
    let _e61 = vertex_out;
    return _e61;
}
