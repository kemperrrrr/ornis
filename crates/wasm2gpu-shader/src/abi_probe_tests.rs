//! Host-shadow test for the wasm compute ABI probe.

use super::*;
use ornis_gpu_abi::GpuVec4;
use ornis_gpu_abi::HostMemory;
use ornis_gpu_abi::with_host;

const INITIAL_ATOMIC_VALUE: u32 = 10;
const WORKGROUP_ATOMIC_OLD: u32 = 0;
const STORAGE_ATOMIC_RESULT: u32 = INITIAL_ATOMIC_VALUE + STORAGE_ATOMIC_VALUE;
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

#[test]
fn abi_probe_matches_the_host_shadow() {
    let mut memory = HostMemory::new();
    memory.bind_texture::<ABI_GROUP, TEXTURE_BINDING>(2, 1, &TEXELS);
    memory.bind_sampler::<ABI_GROUP, SAMPLER_BINDING>();
    memory.bind_storage::<ABI_GROUP, FLOAT_OUTPUT_BINDING, [f32; 8]>();
    memory.bind_storage::<ABI_GROUP, ATOMIC_INPUT_BINDING, [u32; 1]>();
    memory.write_storage::<ABI_GROUP, ATOMIC_INPUT_BINDING, [u32; 1]>(&[INITIAL_ATOMIC_VALUE]);
    memory.bind_storage::<ABI_GROUP, U32_OUTPUT_BINDING, [u32; 4]>();

    with_host(&mut memory, || abi_probe(0));

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
