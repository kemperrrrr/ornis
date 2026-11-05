//! UI test (pass): a fixed writer and a frame writer may coexist in one
//! system — they address different queue singletons, so neither is a
//! duplicate of the other.
//!
//! Two writers of the *same* kind stay an error
//! (`smart_system_rejects_duplicate_structural_writer`).

use ornis_core::mutation::{
    FixedStructuralQueue, FrameStructuralQueue, StructuralQueue, apply_fixed_structural_queue,
    apply_frame_structural_queue,
};
use ornis_core::{Entity, FrameStep, Pack, Resources, Schedule, SmartStore, StructuralWriter};
use ornis_macros::smart_system;

#[derive(Clone, Debug, PartialEq, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FixedMark(u32);

#[derive(Clone, Copy, Debug, PartialEq)]
struct FrameMark(u32);

/// Queues one op on each queue from the same body.
#[smart_system]
fn mixed(
    u: &mut Unit,
    #[entity] e: Entity,
    fixed: StructuralWriter<'_, ornis_core::FixedStep>,
    frame: StructuralWriter<'_, FrameStep>,
) {
    u.hp += 1.0;
    fixed.insert(e, FixedMark(1));
    frame.insert(e, FrameMark(2));
}

fn main() {
    let mut resources = Resources::new();
    let mut store = SmartStore::new();
    Unit::pack_register(&mut store);
    store.register::<FixedMark>();
    store.register::<FrameMark>();
    let entity = store.create_entity();
    Unit { hp: 0.0 }.pack_insert(&mut store, entity);
    resources.insert(store);
    ornis_core::mutation::register_structural_queues(&mut resources);

    let mut schedule = Schedule::new();
    schedule.add_system(MixedSystem::new());
    schedule.run(&resources);

    assert_eq!(
        resources
            .get::<FixedStructuralQueue>()
            .expect("fixed queue")
            .len(),
        1
    );
    assert_eq!(
        resources
            .get::<FrameStructuralQueue>()
            .expect("frame queue")
            .len(),
        1
    );

    apply_fixed_structural_queue(&mut resources);
    {
        let store = resources.get::<SmartStore>().expect("store");
        let fixed = store
            .read_lane::<FixedMark>()
            .and_then(|lane| lane.get(entity).copied());
        let frame = store
            .read_lane::<FrameMark>()
            .and_then(|lane| lane.get(entity).copied());
        assert_eq!(fixed, Some(FixedMark(1)), "fixed op applied");
        assert_eq!(frame, None, "frame op waits for its own boundary");
    }

    apply_frame_structural_queue(&mut resources);
    let store = resources.get::<SmartStore>().expect("store");
    assert_eq!(
        store
            .read_lane::<FrameMark>()
            .and_then(|lane| lane.get(entity).copied()),
        Some(FrameMark(2))
    );
}
