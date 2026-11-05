//! UI test: two `StructuralWriter` parameters address the same
//! `StructuralQueue` singleton — declare one writer per system.

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

#[smart_system]
fn sys(u: &mut Unit, first: StructuralWriter, second: StructuralWriter<'_>) {
    u.hp += 1.0;
    let _ = (first, second);
}

fn main() {}
