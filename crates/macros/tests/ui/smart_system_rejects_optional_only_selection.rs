//! UI test: a body that touches only `Option` fields cannot select entities,
//! because sparse lanes list only the entries that exist.

use ornis_core::Entity;
use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Chaser {
    position: f32,
    target: Option<Entity>,
}

#[smart_system]
fn detach(c: &mut Chaser) {
    c.target = None;
}

fn main() {}
