//! UI test: `Res` rejects interior mutability (`Mutex`).

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

// Shared `Res` must be `PlainResource`; a `Mutex` would be memory-safe
// but non-deterministic on a parallel level — use `ResMut` instead.
#[smart_system]
fn sys(u: &mut Unit, counter: Res<std::sync::Mutex<u32>>) {
    let _ = u.hp;
}

fn main() {}
