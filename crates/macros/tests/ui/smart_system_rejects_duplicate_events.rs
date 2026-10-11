//! UI test: one event queue declared through `EventWriter` + `EventReader` is a signature error.

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

struct Ping;

#[smart_system]
fn sys(u: &mut Unit, out: EventWriter<Ping>, inbox: EventReader<Ping>) {
    let _ = (u.hp, out, inbox);
}

fn main() {}
