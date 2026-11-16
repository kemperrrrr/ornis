
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    previous_view_proj: mat4x4<f32>,
}
struct PerObject {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    previous_model: mat4x4<f32>,
    material_index: u32,
    motion_valid: u32,
}
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec3<f32>,
};
struct OutlineVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> per_objects: array<PerObject>;
@group(0) @binding(2) var<storage, read> outline_indices: array<u32>;
@group(0) @binding(3) var<storage, read> outline_flags: array<u32>;
const OUTLINE_WIDTH: f32 = 0.05;
const OUTLINE_HOVER: vec3<f32> = vec3<f32>(0.55, 0.75, 1.0);
const OUTLINE_SELECTED: vec3<f32> = vec3<f32>(0.16, 0.55, 1.0);
@vertex
fn vs_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> OutlineVertexOutput {
let scene_index = outline_indices[instance_index];  let obj = per_objects[scene_index];  let flag = outline_flags[instance_index];  let expanded = input.position + input.normal * OUTLINE_WIDTH;  let clip_position = camera.view_proj * (obj.model * vec4<f32>(expanded, 1.0));  var color = OUTLINE_HOVER;  if (flag == 2) { color = OUTLINE_SELECTED; }  var output: OutlineVertexOutput;  output.clip_position = clip_position;  output.color = color;  return output; 
}
