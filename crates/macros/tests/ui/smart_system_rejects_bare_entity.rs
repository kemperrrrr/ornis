//! UI test: `u` may only appear as `u.field`; the whole entity is not visible.

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

#[smart_system]
fn sys(u: &Unit) {
    let _ = u.hp;
    let _whole = u;
}

fn main() {}
