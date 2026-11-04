//! UI test: `#[smart_system]` rejects `Res<StructuralQueue>` — deferred
//! structural ops must be declared with `StructuralWriter` so duplicate and
//! conflicting declarations stay detectable.

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

#[smart_system]
fn sys(u: &mut Unit, queue: Res<StructuralQueue>) {
    let _ = (u.hp, queue);
}

fn main() {}
