//! Host-shadow coverage for compute and texture ABI imports.

use ornis_gpu_abi::GpuVec2;
use ornis_gpu_abi::GpuVec4;
use ornis_gpu_abi::HostMemory;
use ornis_gpu_abi::Sampler;
use ornis_gpu_abi::Texture2d;
use ornis_gpu_abi::storage_atomic_add_u32;
use ornis_gpu_abi::storage_barrier;
use ornis_gpu_abi::storage_store_f32;
use ornis_gpu_abi::storage_store_u32;
use ornis_gpu_abi::with_host;
use ornis_gpu_abi::workgroup_atomic_add_u32;
use ornis_gpu_abi::workgroup_barrier;
use ornis_gpu_abi::workgroup_load_u32;
use ornis_gpu_abi::workgroup_store_u32;

const ABI_GROUP: u32 = 0;
const TEXTURE_BINDING: u32 = 1;
const SAMPLER_BINDING: u32 = 2;
const FLOAT_OUTPUT_BINDING: u32 = 3;
const ATOMIC_INPUT_BINDING: u32 = 4;
const U32_OUTPUT_BINDING: u32 = 5;
const TEXEL_X: i32 = 1;
const TEXEL_Y: i32 = 0;
const SAMPLE_U: f32 = 0.75;
const SAMPLE_V: f32 = 0.5;
const SAMPLE_LOD: f32 = 0.0;
const VECTOR_X_OFFSET: u32 = 0;
const VECTOR_Y_OFFSET: u32 = 1;
const VECTOR_Z_OFFSET: u32 = 2;
const VECTOR_W_OFFSET: u32 = 3;
const LOAD_OUTPUT_OFFSET: u32 = 0;
const SAMPLE_OUTPUT_OFFSET: u32 = 4;
const STORAGE_ATOMIC_INDEX: u32 = 0;
const STORAGE_ATOMIC_VALUE: u32 = 5;
const WORKGROUP_INDEX: u32 = 0;
const WORKGROUP_INITIAL_VALUE: u32 = 7;
const WORKGROUP_ATOMIC_INDEX: u32 = 1;
const WORKGROUP_ATOMIC_FIRST_VALUE: u32 = 5;
const WORKGROUP_ATOMIC_SECOND_VALUE: u32 = 7;
const STORAGE_OUTPUT_STORAGE_OLD: u32 = 0;
const STORAGE_OUTPUT_WORKGROUP_LOAD: u32 = 1;
const STORAGE_OUTPUT_WORKGROUP_ATOMIC_OLD: u32 = 2;
const STORAGE_OUTPUT_WORKGROUP_ATOMIC_NEXT: u32 = 3;
const INITIAL_ATOMIC_VALUE: u32 = 10;
const STORAGE_ATOMIC_RESULT: u32 = INITIAL_ATOMIC_VALUE + STORAGE_ATOMIC_VALUE;
const WORKGROUP_ATOMIC_OLD: u32 = 0;
const CULL_STORAGE_INPUT: [f32; 3] = [3.0, 4.0, 0.0];
const CULL_STORAGE_LENGTH: f32 = 5.0;
const CULL_TEST_OUTSIDE_CENTER: f32 = -2.0;
const CULL_TEST_INSIDE_CENTER: f32 = 0.5;
const CULL_TEST_RADIUS: f32 = 0.25;
const CULL_TEST_BOUNDARY_CENTER: f32 = -CULL_TEST_RADIUS;
const CULL_TEST_PLANE: u32 = 0;
const FLOATS: [u32; 8] = [
    0,
    0x3f80_0000,
    0,
    0x3f80_0000,
    0,
    0x3f80_0000,
    0,
    0x3f80_0000,
];
const U32S: [u32; 4] = [
    INITIAL_ATOMIC_VALUE,
    WORKGROUP_INITIAL_VALUE,
    WORKGROUP_ATOMIC_OLD,
    WORKGROUP_ATOMIC_FIRST_VALUE,
];
const TEXELS: [GpuVec4; 2] = [
    GpuVec4 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    },
    GpuVec4 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
        w: 1.0,
    },
];

#[test]
fn compute_texture_and_atomic_imports_match_the_host_shadow() {
    let mut memory = HostMemory::new();
    memory.bind_texture::<ABI_GROUP, TEXTURE_BINDING>(2, 1, &TEXELS);
    memory.bind_sampler::<ABI_GROUP, SAMPLER_BINDING>();
    memory.bind_storage::<ABI_GROUP, FLOAT_OUTPUT_BINDING, [f32; 8]>();
    memory.bind_storage::<ABI_GROUP, ATOMIC_INPUT_BINDING, [u32; 1]>();
    memory.write_storage::<ABI_GROUP, ATOMIC_INPUT_BINDING, [u32; 1]>(&[INITIAL_ATOMIC_VALUE]);
    memory.bind_storage::<ABI_GROUP, U32_OUTPUT_BINDING, [u32; 4]>();

    with_host(&mut memory, || {
        let texture = Texture2d::<ABI_GROUP, TEXTURE_BINDING>::bind();
        let sampler = Sampler::<ABI_GROUP, SAMPLER_BINDING>::bind();
        let loaded = texture.load(TEXEL_X, TEXEL_Y);
        let sampled = texture.sample_lod(
            &sampler,
            GpuVec2 {
                x: SAMPLE_U,
                y: SAMPLE_V,
            },
            SAMPLE_LOD,
        );
        storage_store_f32::<ABI_GROUP, FLOAT_OUTPUT_BINDING>(
            LOAD_OUTPUT_OFFSET + VECTOR_X_OFFSET,
            loaded.x,
        );
        storage_store_f32::<ABI_GROUP, FLOAT_OUTPUT_BINDING>(
            LOAD_OUTPUT_OFFSET + VECTOR_Y_OFFSET,
            loaded.y,
        );
        storage_store_f32::<ABI_GROUP, FLOAT_OUTPUT_BINDING>(
            LOAD_OUTPUT_OFFSET + VECTOR_Z_OFFSET,
            loaded.z,
        );
        storage_store_f32::<ABI_GROUP, FLOAT_OUTPUT_BINDING>(
            LOAD_OUTPUT_OFFSET + VECTOR_W_OFFSET,
            loaded.w,
        );
        storage_store_f32::<ABI_GROUP, FLOAT_OUTPUT_BINDING>(
            SAMPLE_OUTPUT_OFFSET + VECTOR_X_OFFSET,
            sampled.x,
        );
        storage_store_f32::<ABI_GROUP, FLOAT_OUTPUT_BINDING>(
            SAMPLE_OUTPUT_OFFSET + VECTOR_Y_OFFSET,
            sampled.y,
        );
        storage_store_f32::<ABI_GROUP, FLOAT_OUTPUT_BINDING>(
            SAMPLE_OUTPUT_OFFSET + VECTOR_Z_OFFSET,
            sampled.z,
        );
        storage_store_f32::<ABI_GROUP, FLOAT_OUTPUT_BINDING>(
            SAMPLE_OUTPUT_OFFSET + VECTOR_W_OFFSET,
            sampled.w,
        );

        let previous_storage = storage_atomic_add_u32::<ABI_GROUP, ATOMIC_INPUT_BINDING>(
            STORAGE_ATOMIC_INDEX,
            STORAGE_ATOMIC_VALUE,
        );
        storage_store_u32::<ABI_GROUP, U32_OUTPUT_BINDING>(
            STORAGE_OUTPUT_STORAGE_OLD,
            previous_storage,
        );
        workgroup_store_u32(WORKGROUP_INDEX, WORKGROUP_INITIAL_VALUE);
        workgroup_barrier();
        storage_store_u32::<ABI_GROUP, U32_OUTPUT_BINDING>(
            STORAGE_OUTPUT_WORKGROUP_LOAD,
            workgroup_load_u32(WORKGROUP_INDEX),
        );
        let previous_workgroup =
            workgroup_atomic_add_u32(WORKGROUP_ATOMIC_INDEX, WORKGROUP_ATOMIC_FIRST_VALUE);
        let previous_workgroup_after_add =
            workgroup_atomic_add_u32(WORKGROUP_ATOMIC_INDEX, WORKGROUP_ATOMIC_SECOND_VALUE);
        storage_store_u32::<ABI_GROUP, U32_OUTPUT_BINDING>(
            STORAGE_OUTPUT_WORKGROUP_ATOMIC_OLD,
            previous_workgroup,
        );
        storage_store_u32::<ABI_GROUP, U32_OUTPUT_BINDING>(
            STORAGE_OUTPUT_WORKGROUP_ATOMIC_NEXT,
            previous_workgroup_after_add,
        );
        storage_barrier();
    });

    assert_eq!(
        memory
            .read_storage::<ABI_GROUP, FLOAT_OUTPUT_BINDING, [f32; 8]>()
            .map(f32::to_bits),
        FLOATS
    );
    assert_eq!(
        memory.read_storage::<ABI_GROUP, U32_OUTPUT_BINDING, [u32; 4]>(),
        U32S
    );
    assert_eq!(
        memory.read_storage::<ABI_GROUP, ATOMIC_INPUT_BINDING, [u32; 1]>(),
        [STORAGE_ATOMIC_RESULT]
    );
}

#[test]
fn cull_helpers_read_storage_length_and_test_uniform_planes() {
    let mut memory = HostMemory::new();
    memory.bind_storage::<{ super::CULL_GROUP }, { super::SPHERE_BINDING }, [f32; 3]>();
    memory.write_storage::<{ super::CULL_GROUP }, { super::SPHERE_BINDING }, [f32; 3]>(
        &CULL_STORAGE_INPUT,
    );
    let uniform_values: [super::UniformF32Lane; super::PLANE_LEN as usize] =
        std::array::from_fn(|index| super::UniformF32Lane {
            value: if index == 0 { 1.0 } else { 0.0 },
            pad0: 0,
            pad1: 0,
            pad2: 0,
        });
    memory.bind_uniform::<
        { super::CULL_GROUP },
        { super::PLANE_BINDING },
        [super::UniformF32Lane; super::PLANE_LEN as usize],
    >();
    memory.write_uniform::<
        { super::CULL_GROUP },
        { super::PLANE_BINDING },
        [super::UniformF32Lane; super::PLANE_LEN as usize],
    >(&uniform_values);

    let (
        radius,
        outside,
        inside,
        boundary,
        boundary_visible,
        nan_axes_visible,
        mixed_nan_axis_visible,
        negative_radius_visible,
    ) = with_host(&mut memory, || {
        (
            super::cull_storage_length(0),
            super::cull_plane_outside(
                CULL_TEST_OUTSIDE_CENTER,
                0.0,
                0.0,
                CULL_TEST_RADIUS,
                CULL_TEST_PLANE,
            ),
            super::cull_plane_outside(
                CULL_TEST_INSIDE_CENTER,
                0.0,
                0.0,
                CULL_TEST_RADIUS,
                CULL_TEST_PLANE,
            ),
            super::cull_plane_outside(
                CULL_TEST_BOUNDARY_CENTER,
                0.0,
                0.0,
                CULL_TEST_RADIUS,
                CULL_TEST_PLANE,
            ),
            super::cull_record_visible(
                CULL_TEST_BOUNDARY_CENTER,
                0.0,
                0.0,
                CULL_TEST_RADIUS,
                CULL_TEST_RADIUS,
                CULL_TEST_RADIUS,
            ),
            super::cull_record_visible(
                CULL_TEST_OUTSIDE_CENTER,
                0.0,
                0.0,
                f32::NAN,
                f32::NAN,
                f32::NAN,
            ),
            super::cull_record_visible(
                CULL_TEST_OUTSIDE_CENTER,
                0.0,
                0.0,
                f32::NAN,
                CULL_TEST_RADIUS,
                CULL_TEST_RADIUS,
            ),
            super::cull_record_visible(CULL_TEST_OUTSIDE_CENTER, 0.0, 0.0, -1.0, 0.0, 0.0),
        )
    });
    assert_eq!(radius, CULL_STORAGE_LENGTH);
    assert!(outside);
    assert!(!inside);
    assert!(!boundary, "touching a plane remains visible");
    assert!(boundary_visible);
    assert!(nan_axes_visible);
    assert!(!mixed_nan_axis_visible);
    assert!(negative_radius_visible);
    assert!(super::cull_radius_invalid(f32::NAN));
    assert!(super::cull_radius_invalid(-1.0));
    assert!(!super::cull_radius_invalid(2.0));
}
