//! UI test (pass): both writer kinds compile and address the queue their
//! stage drains.
//!
//! Ops queued during `Schedule::run` stay unapplied until the matching
//! boundary: the fixed queue drains after every fixed tick and the frame queue
//! between `PostFrame` and `Stage::Last`.

use ornis_core::mutation::{
    FixedStructuralQueue, FrameStructuralQueue, StructuralQueue, apply_fixed_structural_queue,
    apply_frame_structural_queue,
};
use ornis_core::{FixedStep, FrameStep, Pack, Schedule, World};
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
fn queue_bonus_explicit(u: &mut Unit, #[entity] e: Entity, ops: StructuralWriter<'_, FixedStep>) {
    u.hp += 1.0;
    ops.insert(e, Bonus { amount: 7.0 });
}

/// The frame kind is accepted too; it writes the other queue.
#[smart_system]
fn queue_bonus_frame(u: &mut Unit, #[entity] e: Entity, ops: StructuralWriter<'_, FrameStep>) {
    u.hp += 1.0;
    ops.insert(e, Bonus { amount: 11.0 });
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

    // The kinds are also usable as plain types, e.g. to name a queue in a
    // helper signature.
    let _fixed_kind = std::any::TypeId::of::<FixedStep>();
    let _frame_kind = std::any::TypeId::of::<FrameStep>();

    let mut schedule = Schedule::new();
    schedule.add_system(QueueBonusSystem::new());
    schedule.add_system(QueueBonusExplicitSystem::new());
    schedule.add_system(QueueBonusFrameSystem::new());
    schedule.run(world.resources());

    // The lane-scattered field writes landed; the queued structural ops
    // did not — they wait for their own boundary.
    {
        let store = world.store().expect("world store");
        assert_eq!(Unit::pack_get(store, entity), Some(Unit { hp: 4.0 }));
        assert!(store.read_lane::<Bonus>().is_none(), "queue not applied yet");
        assert_eq!(
            world
                .resources()
                .get::<FixedStructuralQueue>()
                .expect("fixed queue")
                .len(),
            2,
            "the two fixed writers queued two ops"
        );
        assert_eq!(
            world
                .resources()
                .get::<FrameStructuralQueue>()
                .expect("frame queue")
                .len(),
            1,
            "the frame writer queued into the other queue"
        );
    }

    // Draining the fixed queue must not touch the frame queue.
    apply_fixed_structural_queue(world.resources_mut());
    {
        let store = world.store().expect("world store");
        let bonus = store
            .read_lane::<Bonus>()
            .expect("insert op created the lane")
            .get(entity)
            .cloned();
        // Both fixed inserts applied in queue order: the second one won.
        assert_eq!(bonus, Some(Bonus { amount: 7.0 }));
        assert_eq!(
            world
                .resources()
                .get::<FrameStructuralQueue>()
                .expect("frame queue")
                .len(),
            1
        );
    }

    apply_frame_structural_queue(world.resources_mut());
    {
        let store = world.store().expect("world store");
        let bonus = store
            .read_lane::<Bonus>()
            .expect("lane")
            .get(entity)
            .cloned();
        assert_eq!(bonus, Some(Bonus { amount: 11.0 }));
    }
}
