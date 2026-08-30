//! UI test: `Pack` derive rejects tuple structs.

use ornis_macros::Pack;

// #[derive(Pack)] requires named fields — tuple structs are rejected.
#[derive(Pack)]
struct TupleStruct(f32, f32);

fn main() {}
