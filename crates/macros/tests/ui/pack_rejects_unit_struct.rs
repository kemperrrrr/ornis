use ornis_macros::Pack;

// #[derive(Pack)] requires at least one named field — unit structs are rejected.
#[derive(Pack)]
struct UnitStruct;

fn main() {}
