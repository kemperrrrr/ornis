//! R4 structural-writer registration guard (issue #115 review blocker).
//!
//! A `StructuralWriter<FrameStep>` in the fixed plan — or a
//! `StructuralWriter<FixedStep>` in a once-per-frame plan — reintroduces the
//! FPS/catch-up dependence the per-tick apply removed: the frame queue drains
//! on the frame boundary, which the fixed schedule knows nothing about.
//!
//! The guard therefore has to hold on **every** registration path, not just
//! `add_stage_system`. One test per path below: the reviewers found that
//! `add_fixed_system` and `*_schedule_mut(..).add_system(..)` bypassed the
//! check entirely, so each is pinned separately — a shared helper would hide
//! exactly the regression these tests exist to catch.
//!
//! The check lives on the single choke point every path funnels through
//! (`Schedule::add_system`/`prepend_system`), driven by the `PlanRole` the
//! engine tags each of its schedules with. `enqueue` on the queues is
//! likewise not callable from this file: it is crate-private, so an external
//! system cannot bypass the declared queue access either. (Verified by
//! compiling `q.enqueue(..)` against `FixedStructuralQueue` from an
//! integration test: `no method named 'enqueue'`.)

use ornis_core::{Engine, EngineBuilder, Pack, Schedule, Stage};
use ornis_macros::smart_system;

/// Minimal bundle: the writers only need a `Pack` entity set.
#[derive(Clone, Debug, PartialEq, ornis_macros::Pack)]
struct Unit {
    hp: f32,
}

/// Uses the bundle so the derive is not dead code (the guard tests only
/// register systems, they never run them).
#[test]
fn unit_bundle_is_a_valid_pack() {
    let mut store = ornis_core::SmartStore::new();
    Unit::pack_register(&mut store);
    let entity = store.create_entity();
    Unit { hp: 1.0 }.pack_insert(&mut store, entity);
    assert_eq!(Unit::pack_get(&store, entity), Some(Unit { hp: 1.0 }));
}

// Signatures are written with fully-qualified types: `#[smart_system]`
// re-emits them that way, so importing the same names up top would leave
// unused imports behind.
#[smart_system]
fn frame_writer(
    u: &mut Unit,
    #[entity] e: ornis_core::Entity,
    ops: ornis_core::StructuralWriter<ornis_core::FrameStep>,
) {
    u.hp += 1.0;
    let _ = (e, ops);
}

#[smart_system]
fn fixed_writer(
    u: &mut Unit,
    #[entity] e: ornis_core::Entity,
    ops: ornis_core::StructuralWriter<ornis_core::FixedStep>,
) {
    u.hp += 1.0;
    let _ = (e, ops);
}

// ===== Frame writer in the fixed plan =====

#[test]
#[should_panic(expected = "StructuralWriter<FrameStep>")]
fn builder_add_fixed_system_rejects_frame_writer() {
    let mut builder = EngineBuilder::new();
    builder.add_fixed_system(FrameWriterSystem::new());
}

#[test]
#[should_panic(expected = "StructuralWriter<FrameStep>")]
fn building_add_fixed_system_rejects_frame_writer() {
    let mut engine = Engine::new_building();
    engine.add_fixed_system(FrameWriterSystem::new());
}

#[test]
#[should_panic(expected = "StructuralWriter<FrameStep>")]
fn running_add_fixed_system_rejects_frame_writer() {
    let mut engine = Engine::new();
    engine.add_fixed_system(FrameWriterSystem::new());
}

#[test]
#[should_panic(expected = "StructuralWriter<FrameStep>")]
fn add_stage_system_gameplay_rejects_frame_writer() {
    let mut engine = Engine::new();
    engine.add_stage_system(Stage::Gameplay, FrameWriterSystem::new());
}

#[test]
#[should_panic(expected = "StructuralWriter<FrameStep>")]
fn building_stage_schedule_mut_gameplay_rejects_frame_writer() {
    let mut engine = Engine::new_building();
    engine
        .stage_schedule_mut(Stage::Gameplay)
        .add_system(FrameWriterSystem::new());
}

#[test]
#[should_panic(expected = "StructuralWriter<FrameStep>")]
fn running_stage_schedule_mut_gameplay_rejects_frame_writer() {
    let mut engine = Engine::new();
    engine
        .stage_schedule_mut(Stage::Gameplay)
        .add_system(FrameWriterSystem::new());
}

#[test]
#[should_panic(expected = "StructuralWriter<FrameStep>")]
fn fixed_schedule_mut_rejects_frame_writer() {
    let mut engine = Engine::new();
    engine
        .fixed_schedule_mut()
        .add_system(FrameWriterSystem::new());
}

#[test]
#[should_panic(expected = "StructuralWriter<FrameStep>")]
fn builder_fixed_schedule_mut_rejects_frame_writer() {
    let mut builder = EngineBuilder::new();
    builder
        .fixed_schedule_mut()
        .add_system(FrameWriterSystem::new());
}

/// `prepend_system` is the other insertion path into a fixed schedule.
#[test]
#[should_panic(expected = "StructuralWriter<FrameStep>")]
fn fixed_schedule_prepend_rejects_frame_writer() {
    let mut engine = Engine::new();
    engine
        .fixed_schedule_mut()
        .prepend_system(FrameWriterSystem::new());
}

// ===== Fixed writer in a once-per-frame plan =====

#[test]
#[should_panic(expected = "StructuralWriter<FixedStep>")]
fn add_stage_system_post_frame_rejects_fixed_writer() {
    let mut engine = Engine::new();
    engine.add_stage_system(Stage::PostFrame, FixedWriterSystem::new());
}

#[test]
#[should_panic(expected = "StructuralWriter<FixedStep>")]
fn stage_schedule_mut_post_frame_rejects_fixed_writer() {
    let mut engine = Engine::new();
    engine
        .stage_schedule_mut(Stage::PostFrame)
        .add_system(FixedWriterSystem::new());
}

#[test]
#[should_panic(expected = "StructuralWriter<FixedStep>")]
fn schedule_mut_rejects_fixed_writer() {
    let mut engine = Engine::new();
    engine.schedule_mut().add_system(FixedWriterSystem::new());
}

#[test]
#[should_panic(expected = "StructuralWriter<FixedStep>")]
fn builder_schedule_mut_rejects_fixed_writer() {
    let mut builder = EngineBuilder::new();
    builder.schedule_mut().add_system(FixedWriterSystem::new());
}

// ===== Accepted pairings still register =====

/// The matching pairings must keep working on every path, otherwise the guard
/// would be "safe" only by rejecting everything.
#[test]
fn matching_writer_and_plan_register_on_every_path() {
    let mut engine = Engine::new();
    engine.add_stage_system(Stage::Gameplay, FixedWriterSystem::new());
    engine
        .stage_schedule_mut(Stage::Gameplay)
        .add_system(FixedWriterSystem::new());
    engine
        .fixed_schedule_mut()
        .add_system(FixedWriterSystem::new());
    engine.add_fixed_system(FixedWriterSystem::new());
    engine.add_stage_system(Stage::PostFrame, FrameWriterSystem::new());
    engine
        .stage_schedule_mut(Stage::PostFrame)
        .add_system(FrameWriterSystem::new());
    engine.schedule_mut().add_system(FrameWriterSystem::new());
    assert_eq!(engine.stage_schedule(Stage::Gameplay).len(), 4);
    assert_eq!(engine.stage_schedule(Stage::PostFrame).len(), 3);
}

/// A standalone `Schedule` is not driven by an engine boundary, so its host
/// chooses which queue to apply; both kinds are accepted there. The engine
/// tags every schedule it owns, which is where the guard applies.
#[test]
fn standalone_schedule_accepts_both_writer_kinds() {
    let mut schedule = Schedule::new();
    schedule.add_system(FixedWriterSystem::new());
    schedule.add_system(FrameWriterSystem::new());
    assert_eq!(schedule.len(), 2);

    let mut fixed = Schedule::new().with_plan_role(ornis_core::PlanRole::Fixed);
    fixed.add_system(FixedWriterSystem::new());
    assert_eq!(fixed.len(), 1);
}

/// Sealing: the writer guard runs against the engine's fixed schedule even
/// when a system is added through the builder, so a builder-only setup cannot
/// smuggle a frame writer in either.
#[test]
#[should_panic(expected = "StructuralWriter<FrameStep>")]
fn builder_build_keeps_the_guard() {
    let mut builder = EngineBuilder::new();
    builder
        .fixed_schedule_mut()
        .add_system(FrameWriterSystem::new());
    let _ = builder.build();
}
