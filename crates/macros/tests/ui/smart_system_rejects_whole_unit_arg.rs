//! UI test: `#[smart_system]` rejects passing whole `u` to a function.

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

fn helper(_u: &Unit) {}

// Whole-`u` would hide field accesses from access derivation.
#[smart_system]
fn sys(u: &mut Unit) {
    helper(&u);
}

fn main() {}
