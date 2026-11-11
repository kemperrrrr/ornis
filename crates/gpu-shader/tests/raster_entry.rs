//! A raster entry point on the stable SDK.
//!
//! The shader metadata global carries a fixed export name, so one shader crate
//! holds exactly one entry point and this test is its own crate. What it pins
//! is the SDK half: `#[gpu_entry(vertex)]` accepts the stage and leaves a plain
//! safe `fn`, and the stage-IO functions type-check with their const-generic
//! location, declared width, and lane. The translated WGSL is covered by
//! `ornis-llvm2gpu`, which compiles this file's sibling kernels for `wasm32`.

use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::{
    fragment_input_lane, fragment_store_at, instance_index, vertex_index, vertex_input_lane,
    vertex_position, vertex_store_2, vertex_store_4,
};

/// Vertex entry: builtins, one `vec3` input, a `vec2` and a `vec4` output.
#[gpu_entry(vertex)]
pub fn vs_main() {
    let index = vertex_index();
    let instance = instance_index();
    let x = vertex_input_lane::<0, 3, 0>();
    let y = vertex_input_lane::<0, 3, 1>();
    let z = vertex_input_lane::<0, 3, 2>();
    vertex_position(x + index as f32, y, z, 1.0);
    vertex_store_2::<0>(x, y);
    vertex_store_4::<1>(x, y, z, instance as f32);
}

/// A fragment reading the same locations back.
///
/// The bodies never run on the host: the resource and stage-IO methods diverge
/// there, which is why this is a separate type-checking entry rather than a
/// second `#[gpu_entry]` in the same crate.
pub fn fragment_body() {
    let u = fragment_input_lane::<0, 3, 0>();
    let v = fragment_input_lane::<0, 3, 1>();
    fragment_store_at::<0>(u, v, 0.0, 1.0);
}

#[test]
#[should_panic(expected = "requires a wasm32 shader build")]
fn vertex_entry_is_a_plain_safe_function() {
    // No `extern "C"`, no `#[unsafe(no_mangle)]`: the attribute owns the export
    // name, so the kernel is an ordinary Rust function. On the host its first
    // stage-IO call diverges, which is what the panic pins.
    vs_main();
}

#[test]
#[should_panic(expected = "requires a wasm32 shader build")]
fn fragment_stage_io_type_checks() {
    fragment_body();
}
