
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
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) @interpolate(flat) material_index: u32,
    @location(5) previous_world_position: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> per_objects: array<PerObject>;
@group(1) @binding(0) var<storage, read> visible_indices: array<u32>;
@vertex
fn vs_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
let obj = per_objects[instance_index];  let world_pos = obj.model * vec4<f32>(input.position, 1.0);  var previous_world_pos = obj.previous_model * vec4<f32>(input.position, 1.0);  if (obj.motion_valid == 0) { previous_world_pos = vec4<f32>(previous_world_pos.xyz, 0.0); }  var world_normal = normalize((obj.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz);  var world_tangent = normalize((obj.model * vec4<f32>(input.tangent, 0.0)).xyz);  var output: VertexOutput;  output.clip_position = camera.view_proj * world_pos;  output.world_position = world_pos.xyz;  output.previous_world_position = previous_world_pos;  output.world_normal = world_normal;  output.uv = input.uv;  output.world_tangent = world_tangent;  output.material_index = obj.material_index;  return output; 
}
@vertex
fn vs_main_indirect(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
let scene_index = visible_indices[instance_index];  let obj = per_objects[scene_index];  let world_pos = obj.model * vec4<f32>(input.position, 1.0);  var previous_world_pos = obj.previous_model * vec4<f32>(input.position, 1.0);  if (obj.motion_valid == 0) { previous_world_pos = vec4<f32>(previous_world_pos.xyz, 0.0); }  var world_normal = normalize((obj.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz);  var world_tangent = normalize((obj.model * vec4<f32>(input.tangent, 0.0)).xyz);  var output: VertexOutput;  output.clip_position = camera.view_proj * world_pos;  output.world_position = world_pos.xyz;  output.previous_world_position = previous_world_pos;  output.world_normal = world_normal;  output.uv = input.uv;  output.world_tangent = world_tangent;  output.material_index = obj.material_index;  return output; 
}
