//! UI test: untyped Events parameters are rejected in favor of explicit roles.

use std::marker::PhantomData;

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

// A local stand-in catches attempts to keep using the old body-inferred API.
#[allow(dead_code)]
struct Events<E>(PhantomData<E>);

#[smart_system]
fn sys(u: &Unit, events: Events<u32>) {
    let _ = (u.hp, events);
}

fn main() {}
