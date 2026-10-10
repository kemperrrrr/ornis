//! UI test: `#[derive(PlainResource)]` is structural — a struct holding
//! a `Mutex` does not implement it, so `Res<Holder>` fails at the
//! `PlainResource` bound (the v2 `reads` gate for shared resources).

use ornis_macros::PlainResource;

#[derive(PlainResource)]
#[allow(dead_code)]
struct Holder {
    lock: std::sync::Mutex<u32>,
}

fn takes(_: ornis_core::Res<'_, Holder>) {}

fn main() {
    takes;
}
