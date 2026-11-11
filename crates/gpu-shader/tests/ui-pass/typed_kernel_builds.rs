//! A correct typed kernel builds.
//!
//! The other cases in `tests/ui/` must fail on a *const evaluation* error
//! (a tag mismatch), and `cargo check` alone does not evaluate those constants.
//! Keeping one passing case makes trybuild use `cargo build`, so the compile-fail
//! snapshots reflect the diagnostic a real build shows. This file is also the
//! reference for the accepted spelling of every binding kind.

use ornis_gpu_shader::{AtomicU32, GpuStruct, ReadOnly, ReadWrite, Storage, Uniform, gpu_entry};

pub struct Params;

impl GpuStruct for Params {
    const ID: u32 = 0;
    const FIELD_TAGS: &'static [u8] = &[6, 2];
}

#[gpu_entry(compute(threads(32)))]
pub fn cs_main(
    id: u32,
    #[gpu(uniform, group = 0, binding = 0)] params: Uniform<Params>,
    #[gpu(storage_buffer, group = 0, binding = 1, read_only)] src: Storage<f32, ReadOnly>,
    #[gpu(storage_buffer, group = 0, binding = 2)] dst: Storage<u32, ReadWrite>,
    #[gpu(storage_buffer, group = 0, binding = 3)] args: Storage<AtomicU32, ReadWrite>,
) {
    let count = params.field_u32::<1>();
    let distance = params.plane_distance::<0>(0.0, 0.0, 0.0);
    if id >= count || id >= src.len() || distance < 0.0 {
        return;
    }
    let slot = args.atomic_add(1, 1);
    if slot >= dst.len() {
        return;
    }
    dst.store(slot, id);
}

fn main() {}
