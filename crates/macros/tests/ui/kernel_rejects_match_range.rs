//! UI test: `kernel` rejects range patterns (issue #110, feature 3).

use ornis_macros::kernel;

// A range pattern has no WGSL spelling. Compare with explicit literal
// arms instead.
#[kernel]
fn ranged(x: u32) -> u32 {
    match x {
        0u32..=9u32 => 1u32,
        _ => 0u32,
    }
}

fn main() {}
