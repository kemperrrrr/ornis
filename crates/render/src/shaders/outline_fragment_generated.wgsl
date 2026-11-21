@fragment 
fn fs_main(@builtin(position) position: vec4<f32>, @location(0) in0_: vec3<f32>) -> @location(0) vec4<f32> {
    var fragment_out: vec4<f32>;

    fragment_out = vec4<f32>(in0_.x, in0_.y, in0_.z, 1f);
    let _e9 = fragment_out;
    return _e9;
}
