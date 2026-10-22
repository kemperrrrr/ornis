//! UI test: WGSL identifiers must not start with `__` (reserved by the
//! grammar), so a pack lane field spelled `__x` is rejected at macro
//! time on the field span instead of failing naga at `wgsl_source()`.

use ornis_macros::Pack;

#[derive(Clone, Pack)]
#[pack(gpu)]
struct Dunder {
    __x: f32,
}

fn main() {}
