use ornis_macros::PipelineConfig;
use ornis_core::{PipelineConfig, TargetDiscriminant};

#[derive(PipelineConfig)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[test]
fn test_simple_position_gpu() {
    assert_eq!(Position::lane_target(), TargetDiscriminant::Gpu);
    assert!(Position::THRESHOLD > 0);
}

#[derive(PipelineConfig)]
struct LargeComponent {
    data: Vec<String>,
    name: String,
    id: u64,
}

#[test]
fn test_large_struct_with_vec_cpu() {
    // Vec<String> and String are heap types -> CPU
    assert_eq!(LargeComponent::lane_target(), TargetDiscriminant::Cpu);
}

#[derive(PipelineConfig)]
struct SmallGpuStruct {
    a: f32,
    b: f32,
}

#[test]
fn test_small_gpu_struct() {
    // Small f32 fields -> GPU
    assert_eq!(SmallGpuStruct::lane_target(), TargetDiscriminant::Gpu);
}

#[derive(PipelineConfig)]
struct Vec3Component {
    pos: glam::Vec3,
    vel: glam::Vec3,
}

#[test]
fn test_glam_types_gpu() {
    // glam Vec3 -> GPU
    assert_eq!(Vec3Component::lane_target(), TargetDiscriminant::Gpu);
}

#[derive(PipelineConfig)]
struct MixedComponent {
    data: f32,
    name: String,
}

#[test]
fn test_mixed_types_cpu() {
    // String is heap type -> CPU
    assert_eq!(MixedComponent::lane_target(), TargetDiscriminant::Cpu);
}

#[derive(PipelineConfig)]
struct OptionVec3 {
    pos: Option<glam::Vec3>,
}

#[test]
fn test_option_vec3_gpu() {
    // Option<Vec3> -> GPU or Auto
    match OptionVec3::lane_target() {
        TargetDiscriminant::Gpu | TargetDiscriminant::Auto(_) => {}
        _ => panic!("Expected GPU or Auto for Option<Vec3>"),
    }
}

#[derive(PipelineConfig)]
struct GenericComponent<T: Send + Sync + Copy> {
    value: T,
}

#[test]
fn test_generic_with_bounds() {
    let _ = GenericComponent::<f32>::lane_target();
}

#[derive(PipelineConfig)]
#[gpu]
struct ForcedGpu {
    data: Vec<String>,
}

#[test]
fn test_gpu_attribute_override() {
    // #[gpu] attribute overrides to GPU even with heap types
    assert_eq!(ForcedGpu::lane_target(), TargetDiscriminant::Gpu);
}

#[derive(PipelineConfig)]
#[cpu]
struct ForcedCpu {
    x: f32,
    y: f32,
}

#[test]
fn test_cpu_attribute_override() {
    // #[cpu] attribute overrides to CPU even with primitive types
    assert_eq!(ForcedCpu::lane_target(), TargetDiscriminant::Cpu);
}

#[derive(PipelineConfig)]
#[auto]
struct HybridComp {
    x: f32,
    y: f32,
}

#[test]
fn test_hybrid_attribute() {
    // #[auto] attribute forces Auto with threshold
    match HybridComp::lane_target() {
        TargetDiscriminant::Auto(threshold) => {
            assert!(threshold > 0);
        }
        _ => panic!("Expected Auto target due to #[auto] attribute"),
    }
}

#[derive(PipelineConfig)]
struct RecursiveNode {
    value: f32,
    next: Option<Box<RecursiveNode>>,
}

#[test]
fn test_recursive_type_cpu() {
    // Recursive type -> CPU or Auto
    match RecursiveNode::lane_target() {
        TargetDiscriminant::Cpu | TargetDiscriminant::Auto(_) => {}
        _ => panic!("Expected CPU or Auto for recursive type"),
    }
}

// Test with a larger struct to verify size-based threshold
#[derive(PipelineConfig)]
struct LargeStruct {
    a: [f32; 100], // ~400 bytes
}

#[test]
fn test_large_struct_auto() {
    // Size > 256 -> threshold increased
    match LargeStruct::lane_target() {
        TargetDiscriminant::Auto(threshold) => {
            assert!(threshold > 10_000);
        }
        TargetDiscriminant::Cpu => {}
        _ => panic!("Expected CPU or Auto for large struct"),
    }
}

// Test with Option<Vec> (heap type inside Option)
#[derive(PipelineConfig)]
struct OptionVecString {
    data: Option<Vec<String>>,
}

#[test]
fn test_option_vec_string_cpu() {
    assert_eq!(OptionVecString::lane_target(), TargetDiscriminant::Cpu);
}