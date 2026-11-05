//! UI test: `#[smart_system]` rejects `ResMut<StructuralQueue>` — deferred
//! structural ops must be declared with `StructuralWriter` so duplicate and
//! conflicting declarations stay detectable.

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

#[smart_system]
fn sys(u: &mut Unit, queue: ResMut<StructuralQueue>) {
    let _ = (u.hp, queue);
}

fn main() {}
