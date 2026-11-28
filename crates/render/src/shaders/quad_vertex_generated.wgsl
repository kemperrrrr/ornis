struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) o0_: vec2<f32>,
}

@vertex 
fn vs_main_quad(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var vertex_out: VertexOut;
    var _13_u002e_sroa_u002e_0_u002e_0_: f32;
    var _12_u002e_sroa_u002e_0_u002e_2_: f32;

    let _e6 = ((vertex_index & 2u) == 0u);
    if ((vertex_index & 1u) == 0u) {
        if _e6 {
            vertex_out.position = vec4<f32>(-1f, -1f, 0f, 1f);
            _13_u002e_sroa_u002e_0_u002e_0_ = 1f;
            _12_u002e_sroa_u002e_0_u002e_2_ = 0f;
        } else {
            vertex_out.position = vec4<f32>(-1f, 1f, 0f, 1f);
            _13_u002e_sroa_u002e_0_u002e_0_ = 0f;
            _12_u002e_sroa_u002e_0_u002e_2_ = 0f;
        }
    } else {
        if _e6 {
            vertex_out.position = vec4<f32>(1f, -1f, 0f, 1f);
            _13_u002e_sroa_u002e_0_u002e_0_ = 1f;
            _12_u002e_sroa_u002e_0_u002e_2_ = 1f;
        } else {
            vertex_out.position = vec4<f32>(1f, 1f, 0f, 1f);
            _13_u002e_sroa_u002e_0_u002e_0_ = 0f;
            _12_u002e_sroa_u002e_0_u002e_2_ = 1f;
        }
    }
    let _e54 = _13_u002e_sroa_u002e_0_u002e_0_;
    let _e56 = _12_u002e_sroa_u002e_0_u002e_2_;
    vertex_out.o0_ = vec2<f32>(_e56, _e54);
    let _e61 = vertex_out;
    return _e61;
}
