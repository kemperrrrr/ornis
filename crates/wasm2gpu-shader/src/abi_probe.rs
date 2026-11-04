//! GPU-only fixture that exercises the new compute and texture ABI imports.

use ornis_gpu_abi::GpuVec2;
use ornis_gpu_abi::Sampler;
use ornis_gpu_abi::Texture2d;
use ornis_gpu_abi::shader;
use ornis_gpu_abi::storage_atomic_add_u32;
use ornis_gpu_abi::storage_barrier;
use ornis_gpu_abi::storage_store_f32;
use ornis_gpu_abi::storage_store_u32;
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

/// Wasm compute export for testing atomics, barriers, texture loads, and LOD sampling.
#[shader::compute(workgroup_size = 1)]
pub fn abi_probe(_invocation: u32) {
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
}
