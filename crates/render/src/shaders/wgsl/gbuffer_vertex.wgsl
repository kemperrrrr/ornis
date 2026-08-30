
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
}

struct PerObject {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    material_index: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> per_objects: array<PerObject>;

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
};

@vertex
fn vs_main(
    input: VertexInput,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    let obj = per_objects[instance];
    let world_pos = obj.model * vec4<f32>(input.position, 1.0);
    var world_normal = (obj.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    world_normal = normalize(world_normal);
    var world_tangent = (obj.normal_matrix * vec4<f32>(input.tangent, 0.0)).xyz;
    world_tangent = normalize(world_tangent);

    var output: VertexOutput;
    output.clip_position = camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = world_normal;
    output.uv = input.uv;
    output.world_tangent = world_tangent;
    output.material_index = obj.material_index;
    return output;
}
