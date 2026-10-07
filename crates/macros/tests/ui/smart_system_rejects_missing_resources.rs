//! UI test: `#[smart_system]` rejects functions without a `&Resources`
//! parameter — there is nothing to build the system struct around.

use ornis_macros::smart_system;

#[smart_system]
fn no_resources(speed: f32) {
    let _ = speed;
}

fn main() {}
