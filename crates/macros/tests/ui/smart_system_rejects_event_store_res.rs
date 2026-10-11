//! UI test: a read-only `Res<EventStore<E>>` is rejected as well — queues are
//! reached only through `EventWriter<E>` / `EventReader<E>`.

use ornis_macros::smart_system;

#[derive(Clone, Debug, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

struct Ping;

#[smart_system]
fn sys(u: &mut Unit, raw: Res<EventStore<Ping>>) {
    let _ = (u.hp, raw);
}

fn main() {}
