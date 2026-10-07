//! `#[smart_system]` (PLAN m, path 3): the macro generates the whole
//! `ornis_core::System` implementation from a plain function — struct,
//! `name()`, body-derived `access()`, and `run()` — so the declaration
//! cannot desync from the implementation.

use ornis_core::{Resources, Schedule, SmartStore, System, SystemAccess};
use ornis_macros::smart_system;

#[derive(Debug, Clone, PartialEq)]
struct DrivePos {
    x: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct DriveVel {
    x: f32,
}

#[smart_system]
fn drive(resources: &Resources, speed: f32) {
    let store = resources.get::<SmartStore>().expect("store");
    let mut pos = store.write_lane::<DrivePos>().expect("pos");
    for p in pos.iter_mut() {
        p.x += speed;
    }
}

#[test]
fn smart_system_name_defaults_to_fn_name() {
    assert_eq!(DriveSystem::new(1.0).name(), "drive");
}

#[test]
fn smart_system_access_matches_body() {
    // Sorted resources, then read lanes, then write lanes — the macro's
    // canonical order (a write covers a read of the same lane).
    assert_eq!(
        DriveSystem::new(1.0).access(),
        SystemAccess::new()
            .reads::<SmartStore>()
            .writes_lane::<DrivePos>()
    );
}

#[test]
fn smart_system_runs_through_schedule() {
    let mut resources = Resources::new();
    let mut store = SmartStore::new();
    store.register::<DrivePos>();
    for _ in 0..3 {
        let e = store.create_entity();
        store.insert(e, DrivePos { x: 0.0 });
    }
    resources.insert(store);
    let mut sched = Schedule::new();
    sched.set_parallel(false);
    sched.add_system(DriveSystem::new(0.5));
    // Debug access enforcement is active in tests: an access set missing
    // any touched resource or lane would panic here.
    sched.run(&resources);
    let store = resources.get::<SmartStore>().expect("store");
    let lane = store.read_lane::<DrivePos>().expect("pos");
    for pos in lane.iter() {
        assert_eq!(pos.x, 0.5);
    }
}

#[smart_system(name = "custom_drive")]
fn drive_custom(resources: &Resources) {
    let store = resources.get::<SmartStore>().expect("store");
    let vel = store.read_lane::<DriveVel>().expect("vel");
    let _ = vel.len();
}

#[test]
fn smart_system_custom_name_and_unit_struct() {
    let sys = DriveCustomSystem::new();
    assert_eq!(sys.name(), "custom_drive");
    assert_eq!(
        sys.access(),
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<DriveVel>()
    );
}
