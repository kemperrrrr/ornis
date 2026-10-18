//! UI test: macro-generated handles are hygienic and not visible to the body.

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

#[smart_system]
fn sys(u: &Unit) {
    let _ = u.hp;
    // These names are implementation details. In particular, exposing the
    // store would let the body access lanes outside its declared Pack.
    let _ = __smart_store.write_lane::<Unit>();
    let _ = (__entity, __u);
}

fn main() {}
