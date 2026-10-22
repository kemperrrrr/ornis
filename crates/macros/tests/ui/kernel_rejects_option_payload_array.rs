//! UI test: `kernel` rejects array `Option` payloads (issue #110,
//! feature 3).

use ornis_macros::kernel;

// The generated `Option_<T>` struct stores a single scalar/vector
// payload. An array would need a fixed layout the struct cannot name
// portably; flatten it.
#[kernel]
fn array_payload(x: Option<[f32; 4]>) -> f32 {
    match x {
        Some(v) => v[0],
        None => 0.0,
    }
}

fn main() {}
