
struct CompositeVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
const quad: array<vec4<f32>, 4> = array<vec4<f32>, 4>(vec4<f32>(-1f, -1f, 0f, 1f), vec4<f32>(1f, -1f, 0f, 1f), vec4<f32>(-1f, 1f, 0f, 1f), vec4<f32>(1f, 1f, 0f, 1f));
const uvs: array<vec2<f32>, 4> = array<vec2<f32>, 4>(vec2<f32>(0f, 1f), vec2<f32>(1f, 1f), vec2<f32>(0f, 0f), vec2<f32>(1f, 0f));
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> CompositeVertexOutput {
return CompositeVertexOutput(quad[vertex_index], uvs[vertex_index]) /* clip_position, uv */; 
}
