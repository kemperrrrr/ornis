struct VertexOut {
    @builtin(position) position: vec4<f32>,
}

@group(0) @binding(0) 
var tex_0_0_: texture_2d<f32>;

@fragment 
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    var fragment_out: vec4<f32>;

    let _e3 = textureDimensions(tex_0_0_);
    let _e17 = bitcast<u32>(max(bitcast<i32>(bitcast<u32>(min(bitcast<i32>(bitcast<u32>(i32(position.x))), bitcast<i32>((_e3.x + 4294967295u))))), bitcast<i32>(0u)));
    let _e20 = textureDimensions(tex_0_0_);
    let _e34 = bitcast<u32>(max(bitcast<i32>(bitcast<u32>(min(bitcast<i32>(bitcast<u32>(i32(position.y))), bitcast<i32>((_e20.y + 4294967295u))))), bitcast<i32>(0u)));
    let _e40 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(_e17), bitcast<i32>(_e34)), 0i);
    let _e47 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(_e17), bitcast<i32>(_e34)), 0i);
    let _e54 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(_e17), bitcast<i32>(_e34)), 0i);
    let _e61 = textureLoad(tex_0_0_, vec2<i32>(bitcast<i32>(_e17), bitcast<i32>(_e34)), 0i);
    fragment_out = vec4<f32>(_e40.x, _e47.y, _e54.z, _e61.w);
    let _e66 = fragment_out;
    return _e66;
}

@vertex 
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var vertex_out: VertexOut;

    vertex_out.position = vec4<f32>(f32(bitcast<i32>((((vertex_index << 2u) & 4u) + 4294967295u))), f32(bitcast<i32>((((vertex_index << 1u) & 4u) + 4294967295u))), 0f, 1f);
    let _e23 = vertex_out;
    return _e23;
}
