//! UI test: writing an own field requires `u: &mut T`.

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

#[smart_system]
fn sys(u: &Unit) {
    u.hp += 1.0;
}

fn main() {}
