//! UI test: a `#[pack(gpu)]` column type must be `bytemuck::Pod` —
//! the upload writes the column with `bytemuck::bytes_of`, and the
//! derive asserts the bound on the field's own span.

use ornis_macros::Pack;

#[derive(Clone, Debug, PartialEq)]
struct NotPod {
    x: u32,
}

#[derive(Clone, Pack)]
#[pack(gpu)]
struct Mirrored {
    v: NotPod,
}

fn main() {}
