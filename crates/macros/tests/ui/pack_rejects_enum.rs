//! UI test: `Pack` derive rejects enums.

use ornis_macros::Pack;

// #[derive(Pack)] only works on structs — enums are rejected.
#[derive(Pack)]
enum NotAStruct {
    A,
    B,
}

fn main() {}
