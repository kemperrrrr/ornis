use ornis_macros::{gpu_pipeline, kernel};

#[gpu_pipeline]
fn add_kernel(a: f32, b: f32) -> f32 {
    a + b * 2.0
}

#[gpu_pipeline]
fn vec3_kernel(pos: glam::Vec3, vel: glam::Vec3) -> glam::Vec3 {
    pos + vel * 0.016
}

// Full-shader mode: bindings, workgroup size and built-ins are declared in
// the attribute; the function body is the compute entry-point body.
#[gpu_pipeline(
    workgroup_size = 4,
    storage(body_buf: [f32; 64], read_write),
    uniform(params: [u32; 4]),
    builtin(gid: workgroup_id, lid: local_invocation_id),
)]
fn scale_kernel() {
    let i = gid.x;
    if i < params.x {
        body_buf[i] = body_buf[i] * f32(params.y) + f32(lid.x);
    }
}

#[test]
fn add_kernel_compiles() {
    let _source = add_kernel::wgsl_source();
    assert!(_source.contains("storage"));
    assert!(_source.contains("output[i]"));
    assert!(_source.contains("a + b * 2.0"));
}

#[test]
fn vec3_kernel_wgsl() {
    let source = vec3_kernel::wgsl_source();
    assert!(source.contains("vec3<f32>"));
    assert!(source.contains("pos + vel * 0.016"));
}

#[test]
fn full_shader_mode_generates_bindings_and_entry() {
    let source = scale_kernel::wgsl_source();
    assert!(source.contains(
        "@group(0) @binding(0) var<storage, read_write> body_buf: array<f32>;"
    ));
    assert!(source.contains("@group(0) @binding(1) var<uniform> params: vec4<u32>;"));
    assert!(source.contains("@compute @workgroup_size(4)"));
    assert!(source.contains("@builtin(workgroup_id) gid: vec3<u32>"));
    assert!(source.contains("@builtin(local_invocation_id) lid: vec3<u32>"));
    assert!(source.contains("fn main("));
    assert!(source.contains("if (i < params.x)"));
    assert!(source.contains("body_buf[i] = body_buf[i] * f32(params.y) + f32(lid.x);"));
}

// #[kernel] validation tests
#[kernel]
fn kernel_valid(a: f32, b: f32) -> f32 {
    a + b * 2.0
}

#[test]
fn kernel_valid_compiles() {
    let source = kernel_valid::wgsl_source();
    assert!(source.contains("a + b * 2.0"));
}

// Note: compile-failure tests for Vec/String/recursion cannot be expressed
// as positive tests here; they manifest as compile errors at the macro expansion site.
