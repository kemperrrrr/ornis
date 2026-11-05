//! UI test: `StructuralWriter` takes a lifetime only — type arguments are
//! rejected (the queue carries boxed ops, not a typed payload).

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

#[smart_system]
fn sys(u: &mut Unit, ops: StructuralWriter<u32>) {
    u.hp += 1.0;
    let _ = ops;
}

fn main() {}
