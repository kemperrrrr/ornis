//! The typed SDK describes a binding exactly once, and the metadata record
//! agrees with the types.
//!
//! These tests run on the host: the resource methods panic off `wasm32`, so the
//! assertions stay on the compile-time half — the `(group, binding)` pair a
//! parameter resolves to, the access mode it carries, and the record the
//! attribute emits for it.

use ornis_gpu_shader::AtomicU32;
use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::ReadWrite;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::gpu_entry;

/// One uniform struct with the `CullParams` shape.
///
/// Public because it appears in `typed_kernel`'s signature: the resource
/// parameters stay in the signature now, so their types are reachable too.
pub struct Params;

impl GpuStruct for Params {
    const ID: u32 = 0;
    const FIELD_TAGS: &'static [u8] = &[6, 6, 6, 6, 6, 6, 2];
}

/// A kernel whose body never runs on the host but must still type-check.
///
/// A plain safe `fn`: no `extern "C"`, no `#[unsafe(no_mangle)]`.
#[gpu_entry(compute(threads(64)))]
pub fn typed_kernel(
    id: u32,
    #[gpu(uniform, group = 0, binding = 0)] params: Uniform<Params>,
    #[gpu(storage_buffer, group = 0, binding = 1, read_only)] src: Storage<f32, ReadOnly>,
    #[gpu(storage_buffer, group = 0, binding = 2)] dst: Storage<u32, ReadWrite>,
    #[gpu(storage_buffer, group = 0, binding = 3)] args: Storage<AtomicU32, ReadWrite>,
) {
    let count = params.field_u32::<6>();
    if id >= count {
        return;
    }
    if id >= src.len() {
        return;
    }
    let distance = params.plane_distance::<0>(id as f32, 0.0, 0.0);
    let scale = Vec3::new(distance, 0.0, 0.0).length();
    if scale < 0.0 {
        return;
    }
    let slot = args.atomic_add(1, 1);
    if slot >= dst.len() {
        return;
    }
    dst.store(slot, id);
}

#[test]
fn binding_resolution_writes_group_and_binding_once() {
    // The attribute is the only place a binding's address is written.
    assert_eq!(Storage::<f32, 0, 1, ReadOnly>::address(), (0, 1));
    assert_eq!(Storage::<u32, 0, 2, ReadWrite>::address(), (0, 2));
}

#[test]
fn resource_parameters_are_zero_sized() {
    // A kernel is ordinary Rust: the resource parameters stay in the signature
    // and the body uses them as arguments. What the translator sees is
    // `cs_main(i32)` only because the binding types are zero-sized — rustc drops
    // ZST arguments from the compiled entry. If one of these ever grew a field,
    // the entry would start carrying an argument the translator cannot lower,
    // so the invariant is asserted here and against real IR in `abi_parity`.
    assert_eq!(
        (
            core::mem::size_of::<Storage<f32, 0, 1, ReadOnly>>(),
            core::mem::size_of::<Storage<u32, 0, 2, ReadWrite>>(),
            core::mem::size_of::<Storage<AtomicU32, 0, 3, ReadWrite>>(),
            core::mem::size_of::<Uniform<Params, 0, 0>>(),
        ),
        (0, 0, 0, 0),
        "resource parameters must be zero-sized so the entry stays `cs_main(i32)`"
    );
    assert_eq!(Storage::<u32, 0, 2, ReadWrite>::address(), (0, 2));
}

#[test]
fn vec3_length_and_dot_are_host_evaluable() {
    // The host build is the reference the GPU parity tests compare against, so
    // the math has to be exact there.
    let unit = Vec3::new(3.0, 4.0, 0.0);
    assert_eq!(unit.length(), 5.0);
    assert_eq!(unit.dot(unit), 25.0);
    assert_eq!(Vec3::new(1.0, 2.0, 3.0).dot(Vec3::new(0.0, 0.0, 1.0)), 3.0);
}

#[test]
fn record_publishes_the_type_tags_not_a_hand_written_list() {
    let record = typed_record();
    assert_eq!(
        record,
        "entry=typed_kernel;workgroup=64,1,1;uniform=0:6666662"
    );
}

/// The metadata record the entry attribute emitted for [`typed_kernel`].
fn typed_record() -> &'static str {
    std::str::from_utf8(&ORNIS_GPU_SHADER_META[..]).expect("the record is ASCII")
}
