//! UI test: `#[derive(PlainResource)]` is structural — a newtype
//! holding a `Mutex` does not implement it.

use std::sync::Mutex;

use ornis_core::Res;
use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

#[derive(ornis_macros::PlainResource)]
struct Holder {
    lock: Mutex<u32>,
}

#[smart_system]
fn sys(u: &mut Unit, holder: Res<Holder>) {
    let _ = (u.hp, holder);
}

fn main() {}
