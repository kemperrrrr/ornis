//! UI test: a field name the WGSL grammar reserves becomes the lane's
//! storage-variable name — `r#type` strips to `type`, which is
//! reserved, so the rejection lands on the field's span (the WGSL
//! failure it prevents would only surface at naga).

use ornis_macros::Pack;

#[derive(Pack)]
#[pack(gpu)]
struct Reserved {
    r#type: u32,
}

fn main() {}
