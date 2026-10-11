//! UI test (pass): two read-only `Res<C>` declarations of one resource need no
//! scheduler ordering, so the signature compiles and the system runs.

use ornis_core::{Pack, Res, Resources, Schedule, SmartStore};
use ornis_macros::smart_system;

#[derive(Clone, Debug, PartialEq, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

#[derive(Clone, Debug, ornis_macros::PlainResource)]
struct Bias {
    amount: f32,
}

#[smart_system]
fn shared_reads(u: &mut Unit, first: Res<Bias>, second: Res<Bias>) {
    u.hp += first.amount + second.amount;
}

fn main() {
    let mut store = SmartStore::new();
    Unit::pack_register(&mut store);
    let entity = store.create_entity();
    Unit { hp: 1.0 }.pack_insert(&mut store, entity);
    let mut resources = Resources::new();
    resources.insert(store);
    resources.insert(Bias { amount: 2.0 });

    let mut schedule = Schedule::new();
    schedule.add_system(SharedReadsSystem::new());
    schedule.run(&resources);

    let store = resources.get::<SmartStore>().expect("store");
    assert_eq!(Unit::pack_get(store, entity), Some(Unit { hp: 5.0 }));
}
