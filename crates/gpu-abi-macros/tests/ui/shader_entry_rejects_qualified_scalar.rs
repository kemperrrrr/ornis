//! UI test: only bare scalar names are accepted, not `foo::f32`.

use ornis_gpu_abi_macros::fragment;

mod foo {
    pub use core::primitive::f32;
}

#[fragment]
fn shade(v: foo::f32) {
    let _ = v;
}

fn main() {}
