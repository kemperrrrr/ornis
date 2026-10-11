//! UI test: one resource declared through two `ResMut` parameters is a signature error.

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

#[smart_system]
fn sys(u: &mut Unit, first: ResMut<u32>, second: ResMut<u32>) {
    let _ = (u.hp, first, second);
}

fn main() {}
