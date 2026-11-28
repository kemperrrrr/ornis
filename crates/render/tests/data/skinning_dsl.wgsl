
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
struct SkinJoint {
    matrix: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}
struct SkinnedVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>,
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
@group(0) @binding(3) var<storage, read> palette: array<SkinJoint>;
@group(0) @binding(4) var<storage, read> previous_palette: array<SkinJoint>;
@vertex
fn vs_main_skinned(input: SkinnedVertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
let obj = per_objects[instance_index];  let joint0 = palette[input.joints.x];  let joint1 = palette[input.joints.y];  let joint2 = palette[input.joints.z];  let joint3 = palette[input.joints.w];  let previous_joint0 = previous_palette[input.joints.x];  let previous_joint1 = previous_palette[input.joints.y];  let previous_joint2 = previous_palette[input.joints.z];  let previous_joint3 = previous_palette[input.joints.w];  let weight0 = input.weights.x;  let weight1 = input.weights.y;  let weight2 = input.weights.z;  let weight3 = input.weights.w;  let skinned_pos = (joint0.matrix * vec4<f32>(input.position, 1.0)).xyz * weight0 + (joint1.matrix * vec4<f32>(input.position, 1.0)).xyz * weight1 + (joint2.matrix * vec4<f32>(input.position, 1.0)).xyz * weight2 + (joint3.matrix * vec4<f32>(input.position, 1.0)).xyz * weight3;  let previous_skinned_pos = (previous_joint0.matrix * vec4<f32>(input.position, 1.0)).xyz * weight0 + (previous_joint1.matrix * vec4<f32>(input.position, 1.0)).xyz * weight1 + (previous_joint2.matrix * vec4<f32>(input.position, 1.0)).xyz * weight2 + (previous_joint3.matrix * vec4<f32>(input.position, 1.0)).xyz * weight3;  let skinned_nrm = normalize((joint0.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz * weight0 + (joint1.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz * weight1 + (joint2.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz * weight2 + (joint3.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz * weight3);  let skinned_tan = normalize((joint0.matrix * vec4<f32>(input.tangent, 0.0)).xyz * weight0 + (joint1.matrix * vec4<f32>(input.tangent, 0.0)).xyz * weight1 + (joint2.matrix * vec4<f32>(input.tangent, 0.0)).xyz * weight2 + (joint3.matrix * vec4<f32>(input.tangent, 0.0)).xyz * weight3);  let world_pos = obj.model * vec4<f32>(skinned_pos, 1.0);  var previous_world_pos = obj.previous_model * vec4<f32>(previous_skinned_pos, 1.0);  if (obj.motion_valid == 0) { previous_world_pos = vec4<f32>(previous_world_pos.xyz, 0.0); }  var world_normal = normalize((obj.normal_matrix * vec4<f32>(skinned_nrm, 0.0)).xyz);  var world_tangent = normalize((obj.model * vec4<f32>(skinned_tan, 0.0)).xyz);  var output: VertexOutput;  output.clip_position = camera.view_proj * world_pos;  output.world_position = world_pos.xyz;  output.previous_world_position = previous_world_pos;  output.world_normal = world_normal;  output.uv = input.uv;  output.world_tangent = world_tangent;  output.material_index = obj.material_index;  return output; 
}
