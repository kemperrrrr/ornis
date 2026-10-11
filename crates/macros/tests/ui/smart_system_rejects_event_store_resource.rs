//! UI test: `ResMut<EventStore<E>>` + `EventReader<E>` address one queue through two
//! roles; the raw `EventStore` resource is rejected in favour of the event parameters.

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

struct Ping;

#[smart_system]
fn sys(u: &mut Unit, raw: ResMut<EventStore<Ping>>, inbox: EventReader<Ping>) {
    let _ = (u.hp, raw, inbox);
}

fn main() {}
