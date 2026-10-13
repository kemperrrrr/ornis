//! UI test: `kernel` rejects narrow/wide source types and suffixes.

use ornis_macros::kernel;

// A `u8` parameter would lower to `f32` while CPU `eval` wraps
// (`300u32 as u8 == 44`), and the `100u8` literal would lower to `u32`:
// all three are rejected at macro time.
#[kernel]
fn pu8(x: u8) -> u8 {
    x + 100u8
}

// `let` annotations and suffixed literals are checked too: `250u8`
// would lower to `250u` while `eval` holds a `u8`.
#[kernel]
fn narrow_let() -> f32 {
    let v: u8 = 250u8;
    v as f32
}

fn main() {}
