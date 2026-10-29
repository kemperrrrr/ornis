//! Shader body compiled to wasm32 and lowered by `ornis-wasm2gpu`.
//!
//! The host build of [`shade`] is the CPU reference. On `wasm32` the same
//! function is an `extern "C"` export named `shade`. `#[unsafe(no_mangle)]`
//! keeps that export name: the gpu-test lifter looks it up, and Rust 2024
//! makes `no_mangle` an unsafe attribute. It is an export attribute, not a
//! memory-unsafe block; the only `unsafe` code stays in `gpu-abi`. Returning
//! a struct would spill through linear memory, which the translator rejects.
//! This crate is not the production shader DSL.

#![cfg_attr(target_arch = "wasm32", no_std)]
#![warn(missing_docs)]

use ornis_gpu_abi::gpu_struct;
use ornis_gpu_abi::storage_at_f32;
use ornis_gpu_abi::storage_store_f32;
use ornis_gpu_abi::uniform_array_f32;
use ornis_gpu_abi::uniform_field_f32;
use ornis_gpu_abi::uniform_field_u32;

#[cfg(target_arch = "wasm32")]
use ornis_gpu_abi::storage_at_u32;
#[cfg(target_arch = "wasm32")]
use ornis_gpu_abi::storage_store_u32;

gpu_struct! {
    /// Uniform parameters. Field 0 is the dynamic index. Field 1 is the scale.
    pub struct ShadeParams {
        /// Element index shared by the storage arrays and the uniform array.
        pub index: u32,
        /// Scale applied to the storage sample.
        pub intensity: f32,
        /// Padding so the struct span stays 16 bytes.
        pub pad0: u32,
        /// Padding so the struct span stays 16 bytes.
        pub pad1: u32,
    }
}

gpu_struct! {
    /// One `f32` at the uniform stride of 16 bytes.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub struct UniformF32Lane {
        /// Value the shader reads.
        pub value: f32,
        /// Padding to the uniform stride.
        pub pad0: u32,
        /// Padding to the uniform stride.
        pub pad1: u32,
        /// Padding to the uniform stride.
        pub pad2: u32,
    }
}

/// Length of the uniform `f32` array [`shade`] indexes.
pub const UNIFORM_LEN: u32 = 4;

/// Element count of the storage buffers [`shade`] reads and writes.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
// qual:api — the gpu e2e harness sizes storage buffers from this length.
pub const STORAGE_LEN: u32 = 4;

const OUT_BINDING: u32 = 3;

/// Storage binding the compute kernel writes. Group 0, binding 0.
pub const COMPUTE_BINDING: u32 = 0;

const THIRD: f32 = 3.0;
const REST: f32 = 4.0;
#[cfg(target_arch = "wasm32")]
const REST_ID: u32 = 3;

/// Planes of the `cull` clip test: four components per plane.
pub const PLANE_COUNT: u32 = 6;

/// Plane components as one uniform `f32` array (stride 16 bytes).
pub const PLANE_LEN: u32 = PLANE_COUNT * 4;

/// Lanes of one sphere: center xyz plus the radius.
const LANES: u32 = 4;

/// Index of the last lane: a sphere's radius, a plane's `w`.
const LAST_LANE: u32 = LANES - 1;

/// Group of every binding the `cull` kernel touches.
pub const CULL_GROUP: u32 = 0;

/// Uniform binding of the padded plane components.
pub const PLANE_BINDING: u32 = 0;

/// Storage binding of the four sphere lanes per sphere.
pub const SPHERE_BINDING: u32 = 1;

/// Storage binding the `cull` kernel writes one flag per sphere into.
pub const FLAG_BINDING: u32 = 2;

/// Storage binding whose element 0 is the guarded sphere count.
pub const COUNT_BINDING: u32 = 3;

/// Value the wasm `compute` export stores at invocation `id`.
///
/// The `match` is what wasm32 lowers with `br_table`. The host gpu test
/// compares the same function with the buffer the GPU wrote.
pub fn compute_value(id: u32) -> f32 {
    match id {
        0 => 1.0,
        1 => 2.0,
        2 => THIRD,
        _ => REST,
    }
}

/// Fragment body.
///
/// `u` and `v` match the vertex outputs and are not read. The dynamic index
/// comes from the uniform struct, so the vertex/fragment locations stay two
/// `f32`s.
#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
// qual:api — host reference for the wasm export; the gpu e2e test calls it.
pub fn shade(u: f32, v: f32) -> f32 {
    shade_body(u, v)
}

/// Wasm export named `shade`.
///
/// `#[unsafe(no_mangle)]` is the Rust 2024 spelling of the export attribute.
/// The lifter copies this body into the toy `fs_main` only when the export
/// name is `shade`.
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn shade(u: f32, v: f32) -> f32 {
    shade_body(u, v)
}

#[inline(always)]
fn shade_body(_u: f32, _v: f32) -> f32 {
    let index = uniform_field_u32::<0, 0, 0, 0, { ShadeParams::FIELD_OFFSETS[0] }>();
    let intensity = uniform_field_f32::<0, 0, 0, 1, { ShadeParams::FIELD_OFFSETS[1] }>();
    let src = storage_at_f32::<0, 1>(index);
    let extra = uniform_array_f32::<0, 2, UNIFORM_LEN>(index);
    let out = src * intensity + extra;
    storage_store_f32::<0, OUT_BINDING>(index, out);
    out
}

/// Wasm compute export. The lifter renames it to `cs_main`.
///
/// Each `match` arm calls its own `#[inline(never)]` function so wasm32 keeps
/// a multi-way branch (`br_table`) instead of a rodata load. The index is an
/// argument of that function: provenance stays in the const generics, and the
/// offset crosses the call. `#[unsafe(no_mangle)]` is the Rust 2024 export
/// attribute, same as [`shade`].
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn compute(id: u32) {
    let index = slot(id);
    match id {
        0 => arm0(index),
        1 => arm1(index),
        2 => arm2(index),
        _ => arm_rest(index),
    }
}

#[cfg(target_arch = "wasm32")]
#[inline(never)]
fn slot(index: u32) -> u32 {
    index
}

#[cfg(target_arch = "wasm32")]
#[inline(never)]
fn arm0(index: u32) {
    storage_store_f32::<0, COMPUTE_BINDING>(index, compute_value(0));
}

#[cfg(target_arch = "wasm32")]
#[inline(never)]
fn arm1(index: u32) {
    storage_store_f32::<0, COMPUTE_BINDING>(index, compute_value(1));
}

#[cfg(target_arch = "wasm32")]
#[inline(never)]
fn arm2(index: u32) {
    storage_store_f32::<0, COMPUTE_BINDING>(index, compute_value(2));
}

#[cfg(target_arch = "wasm32")]
#[inline(never)]
fn arm_rest(index: u32) {
    storage_store_f32::<0, COMPUTE_BINDING>(index, compute_value(REST_ID));
}

/// Wasm compute export: one visible flag per sphere against six planes.
///
/// The count guard wraps the whole body, so no work happens at `id >=
/// count`; [`cull_one`] stays a call under the `if`. The lifter copies
/// this body into the toy `cs_main` only when the export name is `cull`.
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn cull(id: u32) {
    let count = storage_at_u32::<CULL_GROUP, COUNT_BINDING>(0);
    if id < count {
        cull_one(id);
    }
}

/// Sphere-versus-planes flag for one sphere.
///
/// `dot + w + radius < 0.0` is the production `sphere_visible` test
/// rearranged onto one comparison; IEEE `<` fails open on NaN, like the
/// CPU path. The flag is `1` for visible and `0` for culled.
#[cfg(target_arch = "wasm32")]
#[inline(never)]
// qual:api — the gpu e2e A/B test exercises this kernel through `cull`.
fn cull_one(id: u32) {
    let base = id * LANES;
    let cx = storage_at_f32::<CULL_GROUP, SPHERE_BINDING>(base);
    let cy = storage_at_f32::<CULL_GROUP, SPHERE_BINDING>(base + 1);
    let cz = storage_at_f32::<CULL_GROUP, SPHERE_BINDING>(base + 2);
    let radius = storage_at_f32::<CULL_GROUP, SPHERE_BINDING>(base + LAST_LANE);
    let mut visible = 1u32;
    let mut plane = 0u32;
    while plane < PLANE_COUNT {
        let at = plane * LANES;
        let px = uniform_array_f32::<CULL_GROUP, PLANE_BINDING, PLANE_LEN>(at);
        let py = uniform_array_f32::<CULL_GROUP, PLANE_BINDING, PLANE_LEN>(at + 1);
        let pz = uniform_array_f32::<CULL_GROUP, PLANE_BINDING, PLANE_LEN>(at + 2);
        let pw = uniform_array_f32::<CULL_GROUP, PLANE_BINDING, PLANE_LEN>(at + LAST_LANE);
        if px * cx + py * cy + pz * cz + pw + radius < 0.0 {
            visible = 0;
        }
        plane += 1;
    }
    storage_store_u32::<CULL_GROUP, FLAG_BINDING>(id, visible);
}

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    core::arch::wasm32::unreachable()
}
