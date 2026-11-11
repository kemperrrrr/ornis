//! `#[gpu_entry]` rejects a binding it cannot translate.
//!
//! The attribute is the only place a binding is declared, so a misspelling
//! there has to be a build error rather than a silently unbound parameter: a
//! `uniform` parameter that is a `Storage`, or a `storage_buffer` whose type is
//! a `Uniform`, would otherwise reach the translator as a signature mismatch.

use ornis_gpu_shader::{gpu_entry, ReadOnly, Storage, Uniform};

pub struct Params;

impl ornis_gpu_shader::GpuStruct for Params {
    const ID: u32 = 0;
    const FIELD_TAGS: &'static [u8] = &[2];
}

#[gpu_entry(compute(threads(1)))]
pub fn cs_main(
    id: u32,
    #[gpu(uniform, group = 0, binding = 0)] wrong: Storage<u32, ReadOnly>,
    #[gpu(storage_buffer, group = 0, binding = 1)] also_wrong: Uniform<Params>,
) {
    let _ = (id, wrong, also_wrong);
}

fn main() {}
