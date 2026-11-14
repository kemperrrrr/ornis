
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
// ---- fragment ----

struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    previous_view_proj: mat4x4<f32>,
}
struct OpenPBRMaterial {
    base_params: vec4<f32>,
    base_color: vec4<f32>,
    specular_params: vec4<f32>,
    specular_color: vec4<f32>,
    transmission_params: vec4<f32>,
    transmission_color: vec4<f32>,
    transmission_scatter: vec4<f32>,
    subsurface_params: vec4<f32>,
    subsurface_color: vec4<f32>,
    subsurface_radius_scale_gb: vec4<f32>,
    fuzz_params: vec4<f32>,
    fuzz_color: vec4<f32>,
    coat_params: vec4<f32>,
    coat_color: vec4<f32>,
    coat_ior: vec4<f32>,
    thin_film_params: vec4<f32>,
    emission_params: vec4<f32>,
    emission_color: vec4<f32>,
    geometry_params: vec4<f32>,
    geometry_params2: vec4<f32>,
};
struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) @interpolate(flat) material_index: u32,
    @location(5) previous_world_position: vec4<f32>,
};
struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) material_id: u32,
    @location(3) world_pos: vec2<f32>,
    @location(4) mat_params: vec4<f32>,
    @location(5) motion: vec2<f32>,
    @location(6) previous_depth: u32,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> materials: array<OpenPBRMaterial>;
const PI: f32 = 3.1415927;
const EPS: f32 = 0.000001;
const INV_PI: f32 = 0.31830987;
const SHADOW_REF_BIAS: f32 = 0.001;
const CLEAR_DEPTH: f32 = 1.0;
const NORMAL_AGREE: f32 = 0.999;
const DEPTH_EDGE: f32 = 0.002;
const ALPHA_CUTOFF: f32 = 0.001;
const MSAA_SAMPLES_F: f32 = 4.0;
const POINT_OR_SPOT_KIND_EDGE: f32 = 0.5;
const SPOT_KIND_EDGE: f32 = 1.5;
const RANGE_CUTOFF_INNER_FRACTION: f32 = 0.99;
const NDC_TO_UV_HALF: f32 = 0.5;
const MOTION_VALID_THRESHOLD: f32 = 0.5;
const DEPTH_ROUNDING_BIAS: f32 = 0.5;
const PREVIOUS_DEPTH_MAX: f32 = 65534.0;
const SHADOW_CUBE_NEAR: f32 = 0.1;
const CSM_BLEND_FRACTION: f32 = 0.08;
const CSM_ACTIVE_EDGE: f32 = 0.5;
const LIGHT_SELECTOR_SPLIT: f32 = 0.5;
const SPECULAR_AA_VARIANCE: f32 = 0.15;
const SPECULAR_AA_KERNEL_SCALE: f32 = 2.0;
const SPECULAR_AA_KERNEL_CAP: f32 = 0.2;
const KARIS_LUMA_BIAS: f32 = 1.0;
const HDR_FINITE_LIMIT: f32 = 100000000000000000000.0;
const SHADING_DEBUG_HOT_LUMA: f32 = 1.0;
const UNLIT_THRESHOLD: f32 = 0.5;
const SHADING_DEBUG_BEAUTY: u32 = 0u;
const SHADING_DEBUG_DIRECT_SPECULAR: u32 = 1u;
const SHADING_DEBUG_IBL_SPECULAR: u32 = 2u;
const SHADING_DEBUG_DIFFUSE: u32 = 3u;
const SHADING_DEBUG_ROUGHNESS: u32 = 4u;
const SHADING_DEBUG_METALLIC: u32 = 5u;
const SHADING_DEBUG_SHADOW: u32 = 6u;
const SHADING_DEBUG_HEAT: u32 = 7u;
const CSM_LAST_CASCADE: u32 = 3u;
const PREVIOUS_DEPTH_INVALID: u32 = 65535u;
const CLUSTERS_X: u32 = 16u;
const CLUSTERS_Y: u32 = 9u;
const CLUSTERS_Z: u32 = 24u;
const CLUSTER_DEPTH_NEAR: f32 = 0.5;
const CLUSTER_SLICE_SCALE: f32 = 3.4407384;
const CLUSTER_SLICE_LOOKUP_EPS: f32 = 0.0001;
const MSAA_SAMPLES: u32 = 4u;
fn octahedral_encode(n: vec3<f32>) -> vec2<f32> { let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));  let q = vec2<f32>(p.x, p.y);  var sign_x = -1.0;  if (q.x >= 0.0) { sign_x = 1.0; }  var sign_y = -1.0;  if (q.y >= 0.0) { sign_y = 1.0; }  let flipped = vec2<f32>(1.0 - abs(q.y), 1.0 - abs(q.x)) * vec2<f32>(sign_x, sign_y);  if (n.z < 0.0) { return flipped; }  return q;  }
@fragment
fn fs_main(input: FragmentInput) -> GBufferOutput {
let mat = materials[input.material_index];  let n = normalize(input.world_normal);  let world_pos = input.world_position;  let base_color = mat.base_color.rgb;  let opacity = mat.base_color.a;  let normal_enc = octahedral_encode(n);  let material_id = input.material_index;  let world_pos_enc = vec2<f32>(world_pos.x, world_pos.y);  let roughness = mat.specular_params.y;  let metalness = mat.base_params.z;  let specular_weight = mat.specular_params.x;  let specular_ior = mat.specular_params.z;  let coat_weight = mat.coat_params.x;  let coat_roughness = mat.coat_params.y;  let subsurface_weight = mat.subsurface_params.x;  let transmission_weight = mat.transmission_params.x;  let fuzz_weight = mat.fuzz_params.x;  let thin_film_weight = mat.thin_film_params.x;  let emission_luminance = mat.emission_params.x;  let mat_params = vec4<f32>(roughness, metalness, specular_ior, coat_weight);  let current_clip = camera.view_proj * vec4<f32>(world_pos.xyz, 1.0);  let previous_clip = camera.previous_view_proj * vec4<f32>(input.previous_world_position.xyz, 1.0);  var motion = vec2<f32>(0.0, 0.0);  var previous_depth = PREVIOUS_DEPTH_INVALID;  if (current_clip.w > 0.0 && previous_clip.w > 0.0 && input.previous_world_position.w > MOTION_VALID_THRESHOLD) { let current_ndc = current_clip.xyz / current_clip.w;  let previous_ndc = previous_clip.xyz / previous_clip.w;  let current_uv = vec2<f32>(current_ndc.x * NDC_TO_UV_HALF + NDC_TO_UV_HALF, NDC_TO_UV_HALF - current_ndc.y * NDC_TO_UV_HALF);  let previous_uv = vec2<f32>(previous_ndc.x * NDC_TO_UV_HALF + NDC_TO_UV_HALF, NDC_TO_UV_HALF - previous_ndc.y * NDC_TO_UV_HALF);  motion = current_uv - previous_uv;  previous_depth = u32(clamp(previous_ndc.z, 0.0, 1.0) * PREVIOUS_DEPTH_MAX + DEPTH_ROUNDING_BIAS); }  var output: GBufferOutput;  output.albedo = vec4<f32>(base_color, opacity);  output.normal = normal_enc;  output.material_id = material_id;  output.world_pos = world_pos_enc;  output.mat_params = mat_params;  output.motion = motion;  output.previous_depth = previous_depth;  return output; 
}
