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
    var bounds_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa: f32;
    var bounds_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa: f32;
    var bounds_u002e_i_u002e_sroa_u002e_10_u002e_0_u002e_lcssa: f32;
    var bounds_u002e_i_u002e_sroa_u002e_14_u002e_0_u002e_lcssa: f32;
    var bounds_u002e_i_u002e_sroa_u002e_18_u002e_0_u002e_lcssa: f32;
    var bounds_u002e_i_u002e_sroa_u002e_22_u002e_0_u002e_lcssa: f32;
    var count_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa: f32;
    var sum_sq_cg_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa: f32;
    var sum_sq_co_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa: f32;
    var sum_sq_y_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa: f32;
    var sum_cg_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa: f32;
    var sum_co_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa: f32;
    var sum_y_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa: f32;
    var sum_y_u002e_sroa_u002e_0_u002e_0_u002e_i101_: f32;
    var sum_co_u002e_sroa_u002e_0_u002e_0_u002e_i100_: f32;
    var sum_cg_u002e_sroa_u002e_0_u002e_0_u002e_i99_: f32;
    var sum_sq_y_u002e_sroa_u002e_0_u002e_0_u002e_i98_: f32;
    var sum_sq_co_u002e_sroa_u002e_0_u002e_0_u002e_i97_: f32;
    var sum_sq_cg_u002e_sroa_u002e_0_u002e_0_u002e_i96_: f32;
    var count_u002e_sroa_u002e_0_u002e_0_u002e_i95_: f32;
    var dy_u002e_sroa_u002e_0_u002e_0_u002e_i94_: u32;
    var bounds_u002e_i_u002e_sroa_u002e_22_u002e_093_: f32;
    var bounds_u002e_i_u002e_sroa_u002e_18_u002e_092_: f32;
    var bounds_u002e_i_u002e_sroa_u002e_14_u002e_091_: f32;
    var bounds_u002e_i_u002e_sroa_u002e_10_u002e_090_: f32;
    var bounds_u002e_i_u002e_sroa_u002e_6_u002e_089_: f32;
    var bounds_u002e_i_u002e_sroa_u002e_0_u002e_088_: f32;
    var result_b_u002e_sroa_u002e_0_u002e_0_: f32;
    var result_g_u002e_sroa_u002e_0_u002e_0_: f32;
    var result_r_u002e_sroa_u002e_0_u002e_0_: f32;
    var sum_y_u002e_sroa_u002e_0_u002e_1_u002e_i74_: f32;
    var sum_co_u002e_sroa_u002e_0_u002e_1_u002e_i73_: f32;
    var sum_cg_u002e_sroa_u002e_0_u002e_1_u002e_i72_: f32;
    var sum_sq_y_u002e_sroa_u002e_0_u002e_1_u002e_i71_: f32;
    var sum_sq_co_u002e_sroa_u002e_0_u002e_1_u002e_i70_: f32;
    var sum_sq_cg_u002e_sroa_u002e_0_u002e_1_u002e_i69_: f32;
    var count_u002e_sroa_u002e_0_u002e_1_u002e_i68_: f32;
    var dx_u002e_sroa_u002e_0_u002e_0_u002e_i67_: u32;
    var bounds_u002e_i_u002e_sroa_u002e_22_u002e_166_: f32;
    var bounds_u002e_i_u002e_sroa_u002e_18_u002e_165_: f32;
    var bounds_u002e_i_u002e_sroa_u002e_14_u002e_164_: f32;
    var bounds_u002e_i_u002e_sroa_u002e_10_u002e_163_: f32;
    var bounds_u002e_i_u002e_sroa_u002e_6_u002e_162_: f32;
    var bounds_u002e_i_u002e_sroa_u002e_0_u002e_161_: f32;
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
    let _e10 = textureDimensions(tex_0_0_);
    let _e13 = f32(bitcast<i32>(_e10.y));
    let _e15 = u32((uv.x * _e8));
    let _e17 = u32((uv.y * _e13));
    let _e22 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e28 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e34 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(uv.x, uv.y), 0f);
    let _e41 = textureLoad(tex_0_2_, vec2<i32>(bitcast<i32>(_e15), bitcast<i32>(_e17)), 0u);
    if !((_e41 >= 1f)) {
        let _e50 = textureLoad(tex_0_3_, vec2<i32>(bitcast<i32>(_e15), bitcast<i32>(_e17)), 0i);
        let _e57 = textureLoad(tex_0_3_, vec2<i32>(bitcast<i32>(_e15), bitcast<i32>(_e17)), 0i);
        let _e64 = textureLoad(tex_0_4_, vec2<i32>(bitcast<i32>(_e15), bitcast<i32>(_e17)), 0i);
        let _e66 = (uv.x - _e50.x);
        let _e67 = (uv.y - _e57.y);
        if ((_e66 >= 0f) & (_e66 <= 1f)) {
            if ((_e64.x != 65535u) & ((_e67 >= 0f) & (_e67 <= 1f))) {
                let _e83 = uniform_struct_0_6_.f2_;
                let _e85 = bitcast<u32>(i32(_e83));
                let _e87 = (0u - _e85);
                if (bitcast<i32>(_e85) < bitcast<i32>(_e87)) {
                    bounds_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = 340282350000000000000000000000000000000f;
                    bounds_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = 340282350000000000000000000000000000000f;
                    bounds_u002e_i_u002e_sroa_u002e_10_u002e_0_u002e_lcssa = 340282350000000000000000000000000000000f;
                    bounds_u002e_i_u002e_sroa_u002e_14_u002e_0_u002e_lcssa = -340282350000000000000000000000000000000f;
                    bounds_u002e_i_u002e_sroa_u002e_18_u002e_0_u002e_lcssa = -340282350000000000000000000000000000000f;
                    bounds_u002e_i_u002e_sroa_u002e_22_u002e_0_u002e_lcssa = -340282350000000000000000000000000000000f;
                    count_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = 0f;
                    sum_sq_cg_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = 0f;
                    sum_sq_co_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = 0f;
                    sum_sq_y_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = 0f;
                    sum_cg_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = 0f;
                    sum_co_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = 0f;
                    sum_y_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = 0f;
                } else {
                    sum_y_u002e_sroa_u002e_0_u002e_0_u002e_i101_ = 0f;
                    sum_co_u002e_sroa_u002e_0_u002e_0_u002e_i100_ = 0f;
                    sum_cg_u002e_sroa_u002e_0_u002e_0_u002e_i99_ = 0f;
                    sum_sq_y_u002e_sroa_u002e_0_u002e_0_u002e_i98_ = 0f;
                    sum_sq_co_u002e_sroa_u002e_0_u002e_0_u002e_i97_ = 0f;
                    sum_sq_cg_u002e_sroa_u002e_0_u002e_0_u002e_i96_ = 0f;
                    count_u002e_sroa_u002e_0_u002e_0_u002e_i95_ = 0f;
                    dy_u002e_sroa_u002e_0_u002e_0_u002e_i94_ = _e87;
                    bounds_u002e_i_u002e_sroa_u002e_22_u002e_093_ = -340282350000000000000000000000000000000f;
                    bounds_u002e_i_u002e_sroa_u002e_18_u002e_092_ = -340282350000000000000000000000000000000f;
                    bounds_u002e_i_u002e_sroa_u002e_14_u002e_091_ = -340282350000000000000000000000000000000f;
                    bounds_u002e_i_u002e_sroa_u002e_10_u002e_090_ = 340282350000000000000000000000000000000f;
                    bounds_u002e_i_u002e_sroa_u002e_6_u002e_089_ = 340282350000000000000000000000000000000f;
                    bounds_u002e_i_u002e_sroa_u002e_0_u002e_088_ = 340282350000000000000000000000000000000f;
                    loop {
                        let _e145 = sum_y_u002e_sroa_u002e_0_u002e_0_u002e_i101_;
                        let _e147 = sum_co_u002e_sroa_u002e_0_u002e_0_u002e_i100_;
                        let _e149 = sum_cg_u002e_sroa_u002e_0_u002e_0_u002e_i99_;
                        let _e151 = sum_sq_y_u002e_sroa_u002e_0_u002e_0_u002e_i98_;
                        let _e153 = sum_sq_co_u002e_sroa_u002e_0_u002e_0_u002e_i97_;
                        let _e155 = sum_sq_cg_u002e_sroa_u002e_0_u002e_0_u002e_i96_;
                        let _e157 = count_u002e_sroa_u002e_0_u002e_0_u002e_i95_;
                        let _e159 = dy_u002e_sroa_u002e_0_u002e_0_u002e_i94_;
                        let _e161 = bounds_u002e_i_u002e_sroa_u002e_22_u002e_093_;
                        let _e163 = bounds_u002e_i_u002e_sroa_u002e_18_u002e_092_;
                        let _e165 = bounds_u002e_i_u002e_sroa_u002e_14_u002e_091_;
                        let _e167 = bounds_u002e_i_u002e_sroa_u002e_10_u002e_090_;
                        let _e169 = bounds_u002e_i_u002e_sroa_u002e_6_u002e_089_;
                        let _e171 = bounds_u002e_i_u002e_sroa_u002e_0_u002e_088_;
                        sum_y_u002e_sroa_u002e_0_u002e_1_u002e_i74_ = _e145;
                        sum_co_u002e_sroa_u002e_0_u002e_1_u002e_i73_ = _e147;
                        sum_cg_u002e_sroa_u002e_0_u002e_1_u002e_i72_ = _e149;
                        sum_sq_y_u002e_sroa_u002e_0_u002e_1_u002e_i71_ = _e151;
                        sum_sq_co_u002e_sroa_u002e_0_u002e_1_u002e_i70_ = _e153;
                        sum_sq_cg_u002e_sroa_u002e_0_u002e_1_u002e_i69_ = _e155;
                        count_u002e_sroa_u002e_0_u002e_1_u002e_i68_ = _e157;
                        dx_u002e_sroa_u002e_0_u002e_0_u002e_i67_ = _e87;
                        bounds_u002e_i_u002e_sroa_u002e_22_u002e_166_ = _e161;
                        bounds_u002e_i_u002e_sroa_u002e_18_u002e_165_ = _e163;
                        bounds_u002e_i_u002e_sroa_u002e_14_u002e_164_ = _e165;
                        bounds_u002e_i_u002e_sroa_u002e_10_u002e_163_ = _e167;
                        bounds_u002e_i_u002e_sroa_u002e_6_u002e_162_ = _e169;
                        bounds_u002e_i_u002e_sroa_u002e_0_u002e_161_ = _e171;
                        loop {
                            let _e191 = sum_y_u002e_sroa_u002e_0_u002e_1_u002e_i74_;
                            let _e193 = sum_co_u002e_sroa_u002e_0_u002e_1_u002e_i73_;
                            let _e195 = sum_cg_u002e_sroa_u002e_0_u002e_1_u002e_i72_;
                            let _e197 = sum_sq_y_u002e_sroa_u002e_0_u002e_1_u002e_i71_;
                            let _e199 = sum_sq_co_u002e_sroa_u002e_0_u002e_1_u002e_i70_;
                            let _e201 = sum_sq_cg_u002e_sroa_u002e_0_u002e_1_u002e_i69_;
                            let _e203 = count_u002e_sroa_u002e_0_u002e_1_u002e_i68_;
                            let _e205 = dx_u002e_sroa_u002e_0_u002e_0_u002e_i67_;
                            let _e207 = bounds_u002e_i_u002e_sroa_u002e_22_u002e_166_;
                            let _e209 = bounds_u002e_i_u002e_sroa_u002e_18_u002e_165_;
                            let _e211 = bounds_u002e_i_u002e_sroa_u002e_14_u002e_164_;
                            let _e213 = bounds_u002e_i_u002e_sroa_u002e_10_u002e_163_;
                            let _e215 = bounds_u002e_i_u002e_sroa_u002e_6_u002e_162_;
                            let _e217 = bounds_u002e_i_u002e_sroa_u002e_0_u002e_161_;
                            let _e224 = clamp((uv.x + (f32(bitcast<i32>(_e205)) / _e8)), 0f, 1f);
                            let _e227 = clamp((uv.y + (f32(bitcast<i32>(_e159)) / _e13)), 0f, 1f);
                            let _e232 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(_e224, _e227), 0f);
                            let _e238 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(_e224, _e227), 0f);
                            let _e244 = textureSampleLevel(tex_0_0_, sampler_0_1_, vec2<f32>(_e224, _e227), 0f);
                            let _e249 = (_e238.y * 0.5f);
                            let _e252 = (_e244.z * 0.25f);
                            let _e253 = (((_e232.x * 0.25f) + _e249) + _e252);
                            let _e258 = ((_e232.x * 0.5f) - (_e244.z * 0.5f));
                            let _e262 = ((_e249 - (_e232.x * 0.25f)) - _e252);
                            let _e263 = min(_e217, _e253);
                            out_u002e_6_ = _e263;
                            let _e265 = min(_e215, _e258);
                            out_u002e_7_ = _e265;
                            let _e267 = min(_e213, _e262);
                            out_u002e_8_ = _e267;
                            let _e269 = max(_e211, _e253);
                            out_u002e_9_ = _e269;
                            let _e271 = max(_e209, _e258);
                            out_u002e_10_ = _e271;
                            let _e273 = max(_e207, _e262);
                            out_u002e_11_ = _e273;
                            let _e275 = (_e191 + _e253);
                            out_u002e_12_ = _e275;
                            let _e277 = (_e193 + _e258);
                            out_u002e_13_ = _e277;
                            let _e279 = (_e195 + _e262);
                            out_u002e_14_ = _e279;
                            let _e282 = (_e197 + (_e253 * _e253));
                            out_u002e_15_ = _e282;
                            let _e285 = (_e199 + (_e258 * _e258));
                            out_u002e_16_ = _e285;
                            let _e288 = (_e201 + (_e262 * _e262));
                            out_u002e_17_ = _e288;
                            let _e291 = (_e203 + 1f);
                            out_u002e_18_ = _e291;
                            let _e294 = (_e205 + 1u);
                            if (bitcast<i32>(_e294) > bitcast<i32>(_e85)) {
                                break;
                            } else {
                                sum_y_u002e_sroa_u002e_0_u002e_1_u002e_i74_ = _e275;
                                sum_co_u002e_sroa_u002e_0_u002e_1_u002e_i73_ = _e277;
                                sum_cg_u002e_sroa_u002e_0_u002e_1_u002e_i72_ = _e279;
                                sum_sq_y_u002e_sroa_u002e_0_u002e_1_u002e_i71_ = _e282;
                                sum_sq_co_u002e_sroa_u002e_0_u002e_1_u002e_i70_ = _e285;
                                sum_sq_cg_u002e_sroa_u002e_0_u002e_1_u002e_i69_ = _e288;
                                count_u002e_sroa_u002e_0_u002e_1_u002e_i68_ = _e291;
                                dx_u002e_sroa_u002e_0_u002e_0_u002e_i67_ = _e294;
                                bounds_u002e_i_u002e_sroa_u002e_22_u002e_166_ = _e273;
                                bounds_u002e_i_u002e_sroa_u002e_18_u002e_165_ = _e271;
                                bounds_u002e_i_u002e_sroa_u002e_14_u002e_164_ = _e269;
                                bounds_u002e_i_u002e_sroa_u002e_10_u002e_163_ = _e267;
                                bounds_u002e_i_u002e_sroa_u002e_6_u002e_162_ = _e265;
                                bounds_u002e_i_u002e_sroa_u002e_0_u002e_161_ = _e263;
                                continue;
                            }
                        }
                        let _e313 = out_u002e_12_;
                        let _e315 = out_u002e_13_;
                        let _e317 = out_u002e_14_;
                        let _e319 = out_u002e_15_;
                        let _e321 = out_u002e_16_;
                        let _e323 = out_u002e_17_;
                        let _e325 = out_u002e_18_;
                        let _e327 = out_u002e_11_;
                        let _e329 = out_u002e_10_;
                        let _e331 = out_u002e_9_;
                        let _e333 = out_u002e_8_;
                        let _e335 = out_u002e_7_;
                        let _e337 = out_u002e_6_;
                        let _e339 = (_e159 + 1u);
                        if (bitcast<i32>(_e339) > bitcast<i32>(_e85)) {
                            bounds_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa = _e337;
                            bounds_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa = _e335;
                            bounds_u002e_i_u002e_sroa_u002e_10_u002e_0_u002e_lcssa = _e333;
                            bounds_u002e_i_u002e_sroa_u002e_14_u002e_0_u002e_lcssa = _e331;
                            bounds_u002e_i_u002e_sroa_u002e_18_u002e_0_u002e_lcssa = _e329;
                            bounds_u002e_i_u002e_sroa_u002e_22_u002e_0_u002e_lcssa = _e327;
                            count_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = _e325;
                            sum_sq_cg_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = _e323;
                            sum_sq_co_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = _e321;
                            sum_sq_y_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = _e319;
                            sum_cg_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = _e317;
                            sum_co_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = _e315;
                            sum_y_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa = _e313;
                            break;
                        } else {
                            sum_y_u002e_sroa_u002e_0_u002e_0_u002e_i101_ = _e313;
                            sum_co_u002e_sroa_u002e_0_u002e_0_u002e_i100_ = _e315;
                            sum_cg_u002e_sroa_u002e_0_u002e_0_u002e_i99_ = _e317;
                            sum_sq_y_u002e_sroa_u002e_0_u002e_0_u002e_i98_ = _e319;
                            sum_sq_co_u002e_sroa_u002e_0_u002e_0_u002e_i97_ = _e321;
                            sum_sq_cg_u002e_sroa_u002e_0_u002e_0_u002e_i96_ = _e323;
                            count_u002e_sroa_u002e_0_u002e_0_u002e_i95_ = _e325;
                            dy_u002e_sroa_u002e_0_u002e_0_u002e_i94_ = _e339;
                            bounds_u002e_i_u002e_sroa_u002e_22_u002e_093_ = _e327;
                            bounds_u002e_i_u002e_sroa_u002e_18_u002e_092_ = _e329;
                            bounds_u002e_i_u002e_sroa_u002e_14_u002e_091_ = _e331;
                            bounds_u002e_i_u002e_sroa_u002e_10_u002e_090_ = _e333;
                            bounds_u002e_i_u002e_sroa_u002e_6_u002e_089_ = _e335;
                            bounds_u002e_i_u002e_sroa_u002e_0_u002e_088_ = _e337;
                            continue;
                        }
                    }
                }
                let _e397 = bounds_u002e_i_u002e_sroa_u002e_0_u002e_0_u002e_lcssa;
                let _e399 = bounds_u002e_i_u002e_sroa_u002e_6_u002e_0_u002e_lcssa;
                let _e401 = bounds_u002e_i_u002e_sroa_u002e_10_u002e_0_u002e_lcssa;
                let _e403 = bounds_u002e_i_u002e_sroa_u002e_14_u002e_0_u002e_lcssa;
                let _e405 = bounds_u002e_i_u002e_sroa_u002e_18_u002e_0_u002e_lcssa;
                let _e407 = bounds_u002e_i_u002e_sroa_u002e_22_u002e_0_u002e_lcssa;
                let _e409 = count_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa;
                let _e411 = sum_sq_cg_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa;
                let _e413 = sum_sq_co_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa;
                let _e415 = sum_sq_y_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa;
                let _e417 = sum_cg_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa;
                let _e419 = sum_co_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa;
                let _e421 = sum_y_u002e_sroa_u002e_0_u002e_0_u002e_i_u002e_lcssa;
                let _e422 = (_e421 / _e409);
                let _e423 = (_e419 / _e409);
                let _e424 = (_e417 / _e409);
                let _e444 = (sqrt(max(((_e415 / _e409) - (_e422 * _e422)), 0f)) * 1.25f);
                let _e448 = (sqrt(max(((_e413 / _e409) - (_e423 * _e423)), 0f)) * 1.25f);
                let _e452 = (sqrt(max(((_e411 / _e409) - (_e424 * _e424)), 0f)) * 1.25f);
                let _e463 = uniform_struct_0_6_.f3_;
                if (_e463 > 0.5f) {
                    let _e473 = textureSampleLevel(tex_0_5_, sampler_0_1_, vec2<f32>(_e66, _e67), 0f);
                    let _e479 = textureSampleLevel(tex_0_5_, sampler_0_1_, vec2<f32>(_e66, _e67), 0f);
                    let _e485 = textureSampleLevel(tex_0_5_, sampler_0_1_, vec2<f32>(_e66, _e67), 0f);
                    let _e491 = textureSampleLevel(tex_0_5_, sampler_0_1_, vec2<f32>(_e66, _e67), 0f);
                    if (abs((_e491.w - (f32(_e64.x) / 65534f))) < 0.002f) {
                        let _e500 = (_e479.y * 0.5f);
                        let _e503 = (_e485.z * 0.25f);
                        let _e505 = clamp((((_e473.x * 0.25f) + _e500) + _e503), max(_e397, (_e422 - _e444)), min(_e403, (_e422 + _e444)));
                        let _e511 = clamp(((_e473.x * 0.5f) - (_e485.z * 0.5f)), max(_e399, (_e423 - _e448)), min(_e405, (_e423 + _e448)));
                        let _e516 = clamp(((_e500 - (_e473.x * 0.25f)) - _e503), max(_e401, (_e424 - _e452)), min(_e407, (_e424 + _e452)));
                        let _e519 = uniform_struct_0_6_.f0_;
                        result_b_u002e_sroa_u002e_0_u002e_0_ = mix(_e34.z, ((_e505 - _e511) - _e516), _e519);
                        result_g_u002e_sroa_u002e_0_u002e_0_ = mix(_e28.y, (_e505 + _e516), _e519);
                        result_r_u002e_sroa_u002e_0_u002e_0_ = mix(_e22.x, ((_e505 + _e511) - _e516), _e519);
                    } else {
                        result_b_u002e_sroa_u002e_0_u002e_0_ = _e34.z;
                        result_g_u002e_sroa_u002e_0_u002e_0_ = _e28.y;
                        result_r_u002e_sroa_u002e_0_u002e_0_ = _e22.x;
                    }
                } else {
                    result_b_u002e_sroa_u002e_0_u002e_0_ = _e34.z;
                    result_g_u002e_sroa_u002e_0_u002e_0_ = _e28.y;
                    result_r_u002e_sroa_u002e_0_u002e_0_ = _e22.x;
                }
                let _e538 = result_b_u002e_sroa_u002e_0_u002e_0_;
                let _e540 = result_g_u002e_sroa_u002e_0_u002e_0_;
                let _e542 = result_r_u002e_sroa_u002e_0_u002e_0_;
                fragment_out.o0_ = vec4<f32>(_e542, _e540, _e538, 1f);
                fragment_out.o1_ = vec4<f32>(_e542, _e540, _e538, _e41);
            } else {
                fragment_out.o0_ = vec4<f32>(_e22.x, _e28.y, _e34.z, 1f);
                fragment_out.o1_ = vec4<f32>(_e22.x, _e28.y, _e34.z, _e41);
            }
        } else {
            fragment_out.o0_ = vec4<f32>(_e22.x, _e28.y, _e34.z, 1f);
            fragment_out.o1_ = vec4<f32>(_e22.x, _e28.y, _e34.z, _e41);
        }
    } else {
        fragment_out.o0_ = vec4<f32>(_e22.x, _e28.y, _e34.z, 1f);
        fragment_out.o1_ = vec4<f32>(_e22.x, _e28.y, _e34.z, 1f);
    }
    let _e573 = fragment_out;
    return _e573;
}

@vertex 
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var vertex_out: VertexOut;
    var _11_u002e_sroa_u002e_0_u002e_0_u002e_i: f32;
    var _10_u002e_sroa_u002e_0_u002e_2_u002e_i: f32;

    let _e6 = ((vertex_index & 2u) == 0u);
    if ((vertex_index & 1u) == 0u) {
        if _e6 {
            vertex_out.position = vec4<f32>(-1f, -1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_u002e_i = 1f;
            _10_u002e_sroa_u002e_0_u002e_2_u002e_i = 0f;
        } else {
            vertex_out.position = vec4<f32>(-1f, 1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
            _10_u002e_sroa_u002e_0_u002e_2_u002e_i = 0f;
        }
    } else {
        if _e6 {
            vertex_out.position = vec4<f32>(1f, -1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_u002e_i = 1f;
            _10_u002e_sroa_u002e_0_u002e_2_u002e_i = 1f;
        } else {
            vertex_out.position = vec4<f32>(1f, 1f, 0f, 1f);
            _11_u002e_sroa_u002e_0_u002e_0_u002e_i = 0f;
            _10_u002e_sroa_u002e_0_u002e_2_u002e_i = 1f;
        }
    }
    let _e54 = _11_u002e_sroa_u002e_0_u002e_0_u002e_i;
    let _e56 = _10_u002e_sroa_u002e_0_u002e_2_u002e_i;
    vertex_out.o0_ = vec2<f32>(_e56, _e54);
    let _e61 = vertex_out;
    return _e61;
}
