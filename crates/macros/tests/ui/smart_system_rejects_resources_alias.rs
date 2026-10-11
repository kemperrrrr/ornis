//! UI test: a type alias cannot pass the scheduler's resource container through ResMut.

use ornis_core::{ResMut, Resources};
use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

type ResourcesAlias = Resources;

#[smart_system]
fn sys(u: &Unit, resources: ResMut<ResourcesAlias>) {
    let _ = (u.hp, resources);
}

fn main() {}
