//! UI test: a type alias cannot smuggle SmartStore through ResMut.

use ornis_core::{ResMut, SmartStore};
use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

type StoreAlias = SmartStore;

#[smart_system]
fn sys(u: &Unit, store: ResMut<StoreAlias>) {
    let _ = (u.hp, store);
}

fn main() {}
