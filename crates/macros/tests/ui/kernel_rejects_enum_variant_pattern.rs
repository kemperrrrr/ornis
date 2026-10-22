//! UI test: `kernel` rejects enum-variant paths in patterns
//! (issue #110, feature 3).

use ornis_macros::kernel;

// Fieldless enums have no WGSL type of their own. The DSL spells them
// as a `u32` tag; an `Enum::Variant` path therefore has no spelling and
// is rejected with a message that points at the tag form.
#[kernel]
fn variant(tag: u32) -> u32 {
    match tag {
        State::Idle => 1u32,
        _ => 0u32,
    }
}

enum State {
    Idle,
    Running,
}

fn main() {}
