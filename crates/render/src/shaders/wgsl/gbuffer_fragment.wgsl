
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

@group(0) @binding(2) var<storage, read> materials: array<OpenPBRMaterial>;

struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) @interpolate(flat) material_index: u32,
};

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) material_id: u32,
    @location(3) world_pos: vec2<f32>,
    @location(4) mat_params: vec4<f32>,
};

@fragment
fn fs_main(input: FragmentInput) -> GBufferOutput {
    let mat = materials[input.material_index];

    let N = normalize(input.world_normal);
    let world_pos = input.world_position;
    let base_color = mat.base_color.rgb;
    let opacity = mat.base_color.a;

    let normal_enc = octahedral_encode(N);

    let material_id = input.material_index;

    let world_pos_enc = vec2<f32>(world_pos.x, world_pos.y);

    let roughness = mat.specular_params.y;
    let metalness = mat.base_params.z;
    let specular_weight = mat.specular_params.x;
    let specular_ior = mat.specular_params.z;
    let coat_weight = mat.coat_params.x;
    let coat_roughness = mat.coat_params.y;
    let subsurface_weight = mat.subsurface_params.x;
    let transmission_weight = mat.transmission_params.x;
    let fuzz_weight = mat.fuzz_params.x;
    let thin_film_weight = mat.thin_film_params.x;
    let emission_luminance = mat.emission_params.x;

    let mat_params = vec4<f32>(roughness, metalness, specular_ior, coat_weight);

    var output: GBufferOutput;
    output.albedo = vec4<f32>(base_color, opacity);
    output.normal = normal_enc;
    output.material_id = material_id;
    output.world_pos = world_pos_enc;
    output.mat_params = mat_params;
    return output;
}
