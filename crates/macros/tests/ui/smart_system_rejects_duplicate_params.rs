//! UI test: one resource declared twice (read + write) is a signature error.

use ornis_core::{Res, ResMut};
use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

#[smart_system]
fn sys(u: &mut Unit, read: Res<u32>, write: ResMut<u32>) {
    let _ = (u.hp, read, write);
}

fn main() {}
