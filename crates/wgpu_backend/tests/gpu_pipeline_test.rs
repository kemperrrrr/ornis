use ornis_macros::{gpu_pipeline, kernel};

#[gpu_pipeline]
fn add_kernel(a: f32, b: f32) -> f32 {
    a + b * 2.0
}

#[gpu_pipeline]
fn vec3_kernel(pos: glam::Vec3, vel: glam::Vec3) -> glam::Vec3 {
    pos + vel * 0.016
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
