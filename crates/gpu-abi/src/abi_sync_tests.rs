//! ABI vector-math and synchronization regression tests.

use crate::GpuVec3;
use crate::GpuVec4;
use crate::HostMemory;
use crate::Texture2d;
use crate::storage_atomic_add_u32;
use crate::storage_barrier;
use crate::with_host;
use crate::workgroup_atomic_add_u32;
use crate::workgroup_barrier;
use crate::workgroup_load_u32;
use crate::workgroup_store_u32;

const ABI_MATH_A: GpuVec3 = GpuVec3 {
    x: 1.0,
    y: 2.0,
    z: 3.0,
};
const ABI_MATH_B: GpuVec3 = GpuVec3 {
    x: 4.0,
    y: 5.0,
    z: 6.0,
};
const ABI_MATH_DOT: f32 = 32.0;
const ABI_MATH_CROSS: GpuVec3 = GpuVec3 {
    x: -3.0,
    y: 6.0,
    z: -3.0,
};
const ABI_MATH_LENGTH_INPUT: GpuVec3 = GpuVec3 {
    x: 0.0,
    y: 3.0,
    z: 4.0,
};
const ABI_MATH_LENGTH: f32 = 5.0;
const ABI_MATH_NORMALIZED: GpuVec3 = GpuVec3 {
    x: 0.0,
    y: 0.6,
    z: 0.8,
};
const ATOMIC_STORAGE_WORDS: usize = 4;
const ATOMIC_STORAGE_INDEX: u32 = 2;
const ATOMIC_STORAGE_FIRST_ADD: u32 = 5;
const ATOMIC_STORAGE_SECOND_ADD: u32 = 7;
const WORKGROUP_TEST_INDEX: u32 = 4;
const WORKGROUP_INITIAL_VALUE: u32 = 3;
const WORKGROUP_ADD_VALUE: u32 = 6;
const WORKGROUP_FINAL_VALUE: u32 = 9;
const ATOMIC_STORAGE_FINAL_WORD: u32 = 12;
const ATOMIC_STORAGE_EXPECTED: [u32; ATOMIC_STORAGE_WORDS] = [0, 0, ATOMIC_STORAGE_FINAL_WORD, 0];
const TEXTURE_TEST_GROUP: u32 = 4;
const TEXTURE_TEST_BINDING: u32 = 2;
const TEXTURE_TEST_WIDTH: u32 = 2;
const TEXTURE_TEST_HEIGHT: u32 = 2;
const TEXTURE_TEST_X: i32 = 1;
const TEXTURE_TEST_Y: i32 = 0;
const TEXTURE_TEST_TEXELS: [GpuVec4; 4] = [
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
    GpuVec4 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
        w: 1.0,
    },
    GpuVec4 {
        x: 1.0,
        y: 1.0,
        z: 0.0,
        w: 1.0,
    },
];

#[test]
fn vec3_math_helpers_match_scalar_reference() {
    assert_eq!(crate::gpu_dot3(ABI_MATH_A, ABI_MATH_B), ABI_MATH_DOT);
    assert_eq!(crate::gpu_cross3(ABI_MATH_A, ABI_MATH_B), ABI_MATH_CROSS);
    assert_eq!(crate::gpu_length3(ABI_MATH_LENGTH_INPUT), ABI_MATH_LENGTH);
    assert_eq!(
        crate::gpu_normalize3(ABI_MATH_LENGTH_INPUT),
        ABI_MATH_NORMALIZED
    );
}

#[test]
fn atomics_and_workgroup_memory_have_host_shadows() {
    let mut mem = HostMemory::new();
    mem.bind_storage::<1, 2, [u32; ATOMIC_STORAGE_WORDS]>();
    with_host(&mut mem, || {
        assert_eq!(
            storage_atomic_add_u32::<1, 2>(ATOMIC_STORAGE_INDEX, ATOMIC_STORAGE_FIRST_ADD),
            0
        );
        assert_eq!(
            storage_atomic_add_u32::<1, 2>(ATOMIC_STORAGE_INDEX, ATOMIC_STORAGE_SECOND_ADD),
            ATOMIC_STORAGE_FIRST_ADD
        );
        workgroup_store_u32(WORKGROUP_TEST_INDEX, WORKGROUP_INITIAL_VALUE);
        assert_eq!(
            workgroup_load_u32(WORKGROUP_TEST_INDEX),
            WORKGROUP_INITIAL_VALUE
        );
        assert_eq!(
            workgroup_atomic_add_u32(WORKGROUP_TEST_INDEX, WORKGROUP_ADD_VALUE),
            WORKGROUP_INITIAL_VALUE
        );
        assert_eq!(
            workgroup_load_u32(WORKGROUP_TEST_INDEX),
            WORKGROUP_FINAL_VALUE
        );
        workgroup_barrier();
        storage_barrier();
    });
    assert_eq!(
        mem.read_storage::<1, 2, [u32; ATOMIC_STORAGE_WORDS]>(),
        ATOMIC_STORAGE_EXPECTED
    );
}

#[test]
fn texture_load_reads_the_requested_texel() {
    let mut mem = HostMemory::new();
    let texels = TEXTURE_TEST_TEXELS;
    mem.bind_texture::<TEXTURE_TEST_GROUP, TEXTURE_TEST_BINDING>(
        TEXTURE_TEST_WIDTH,
        TEXTURE_TEST_HEIGHT,
        &texels,
    );
    let got = with_host(&mut mem, || {
        Texture2d::<TEXTURE_TEST_GROUP, TEXTURE_TEST_BINDING>::bind()
            .load(TEXTURE_TEST_X, TEXTURE_TEST_Y)
    });
    assert_eq!(got, texels[1]);
}

#[test]
#[should_panic(expected = "workgroup index out of bounds")]
fn workgroup_shadow_rejects_out_of_range_indices() {
    let mut mem = HostMemory::new();
    with_host(&mut mem, || {
        workgroup_load_u32(crate::symbols::WORKGROUP_WORDS)
    });
}

#[test]
#[should_panic(expected = "gpu access out of bounds")]
fn atomic_storage_shadow_rejects_out_of_range_indices() {
    let mut mem = HostMemory::new();
    mem.bind_storage::<1, 2, [u32; 1]>();
    with_host(&mut mem, || storage_atomic_add_u32::<1, 2>(1, 1));
}

#[test]
#[should_panic(expected = "workgroup index out of bounds")]
fn workgroup_atomic_shadow_rejects_out_of_range_indices() {
    let mut mem = HostMemory::new();
    with_host(&mut mem, || {
        workgroup_atomic_add_u32(crate::symbols::WORKGROUP_WORDS, 1)
    });
}
