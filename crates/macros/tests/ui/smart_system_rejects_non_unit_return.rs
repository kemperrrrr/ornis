//! UI test: `#[smart_system]` rejects functions returning a value —
//! `System::run` returns `()`, silently dropping a result would hide bugs.

use ornis_macros::smart_system;

#[smart_system]
fn returns_value(resources: &ornis_core::Resources) -> usize {
    let _ = resources;
    42
}

fn main() {}
