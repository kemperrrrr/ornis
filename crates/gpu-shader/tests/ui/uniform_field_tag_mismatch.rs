//! A uniform field read is checked against the struct's declared tags.
//!
//! `BadParams` declares field 0 as a `vec4`, so `field_u32::<0>()` fails the
//! attribute's `const` assertion — reading a `u32` out of a `vec4` field would
//! otherwise only fail WGSL validation, after the kernel has been translated.

use ornis_gpu_shader::{gpu_entry, GpuStruct, Uniform};

pub struct BadParams;

impl GpuStruct for BadParams {
    const ID: u32 = 0;
    const FIELD_TAGS: &'static [u8] = &[6, 2];
}

#[gpu_entry(compute(threads(1)))]
pub fn cs_main(
    id: u32,
    #[gpu(uniform, group = 0, binding = 0)] params: Uniform<BadParams>,
) {
    let _ = params.field_u32::<0>() + id;
}

fn main() {}
