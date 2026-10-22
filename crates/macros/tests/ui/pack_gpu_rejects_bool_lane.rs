//! UI test: `#[pack(gpu)]` rejects `bool` columns — a storage array
//! has no bool representation.

use ornis_macros::Pack;

#[derive(Pack)]
#[pack(gpu)]
struct Bools {
    flag: bool,
}

fn main() {}
