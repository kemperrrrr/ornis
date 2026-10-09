//! UI test: `#[smart_system]` rejects writes to another entity.

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

// `t` is bound from a `u.field` chain, so it is another entity's handle:
// `t.hp` is read-only (gather) — mutations go through `Events<E>`.
#[smart_system]
fn sys(u: &mut Unit) {
    let _ = [u.hp].iter().map(|t| {
        t.hp -= 1.0;
        t
    });
}

fn main() {}
