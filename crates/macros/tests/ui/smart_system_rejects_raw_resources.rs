//! UI test: `#[smart_system]` rejects a raw `&Resources` parameter.

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

// Raw `&Resources` would bypass the type-derived access set.
#[smart_system]
fn sys(u: &mut Unit, resources: &Resources) {
    let _ = (u.hp, resources);
}

fn main() {}
