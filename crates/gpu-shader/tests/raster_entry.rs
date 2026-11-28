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

/// A fragment entry next to the vertex one: both live in this crate.
///
/// The metadata global's export name carries the entry, so the two records do
/// not collide and the builder can translate both into one module — which is
/// what a pass with a vertex and a fragment stage needs.
#[gpu_entry(fragment)]
pub fn fs_main() {
    let u = fragment_input_lane::<0, 3, 0>();
    let v = fragment_input_lane::<0, 3, 1>();
    fragment_store_at::<0>(u, v, 0.0, 1.0);
}

/// Byte length of one record here: `entry=<name>;workgroup=1,1,1`, no NUL.
const RECORD_LEN: usize = "entry=vs_main;workgroup=1,1,1".len();

// SAFETY: the two statics are the `[u8; N]` metadata records `#[gpu_entry]`
// emits for this crate's entries. They are plain immutable bytes, and the
// lengths below are the exact record lengths asserted by the test.
unsafe extern "C" {
    static ornis_gpu_shader_meta_vs_main: [u8; RECORD_LEN];
    static ornis_gpu_shader_meta_fs_main: [u8; RECORD_LEN];
}

#[test]
fn each_entry_publishes_its_own_record() {
    let record = |bytes: &'static [u8]| std::str::from_utf8(bytes).expect("ASCII");
    // SAFETY: see the declaration above: both are readable immutable arrays of
    // `RECORD_LEN` bytes, and the borrow ends with this statement.
    let vertex = unsafe { record(&ornis_gpu_shader_meta_vs_main) };
    // SAFETY: same contract as above, one static at a time.
    let fragment = unsafe { record(&ornis_gpu_shader_meta_fs_main) };
    assert_eq!(vertex, "entry=vs_main;workgroup=1,1,1");
    assert_eq!(fragment, "entry=fs_main;workgroup=1,1,1");
}
