//! UI test (pass): bare `StructuralWriter` and `StructuralWriter<'_>` both
//! declare the deferred structural queue. Ops queued during `Schedule::run`
//! stay unapplied until the frame boundary (`apply_structural_queue`), which
//! the engine runs between `PostFrame` and `Stage::Last`.

use ornis_core::{Pack, Schedule, StructuralWriter, World};
use ornis_macros::smart_system;

#[derive(Clone, Debug, PartialEq, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

#[derive(Clone, Debug, PartialEq)]
struct Bonus {
    amount: f32,
}

#[smart_system]
fn queue_bonus(u: &mut Unit, #[entity] e: Entity, ops: StructuralWriter) {
    u.hp += 1.0;
    ops.insert(e, Bonus { amount: 5.0 });
}

#[smart_system]
fn queue_bonus_explicit(u: &mut Unit, #[entity] e: Entity, ops: StructuralWriter<'_>) {
    u.hp += 1.0;
    ops.insert(e, Bonus { amount: 7.0 });
}

fn main() {
    let mut world = World::new();
    let entity = {
        let store = world.store_mut().expect("world store");
        Unit::pack_register(store);
        let entity = store.create_entity();
        Unit { hp: 1.0 }.pack_insert(store, entity);
        entity
    };

    let mut schedule = Schedule::new();
    schedule.add_system(QueueBonusSystem::new());
    schedule.add_system(QueueBonusExplicitSystem::new());
    schedule.run(world.resources());

    // The lane-scattered field writes landed; the queued structural ops
    // did not — they wait for the frame-boundary apply.
    {
        let store = world.store().expect("world store");
        assert_eq!(Unit::pack_get(store, entity), Some(Unit { hp: 3.0 }));
        assert!(store.read_lane::<Bonus>().is_none(), "queue not applied yet");
    }

    ornis_core::mutation::apply_structural_queue(world.resources_mut());

    {
        let store = world.store().expect("world store");
        let bonus = store
            .read_lane::<Bonus>()
            .expect("insert op created the lane")
            .get(entity)
            .cloned();
        // Both systems' inserts applied in queue order: the second one won.
        assert_eq!(bonus, Some(Bonus { amount: 7.0 }));
    }
}
