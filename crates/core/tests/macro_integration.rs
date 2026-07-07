#![allow(dead_code)]

use ornis_core::{PipelineConfig as _, SmartStore};
use ornis_macros::{AutoPipeline as DeriveAutoPipeline, PipelineConfig, for_each_entity, smart_pipeline};

#[derive(Debug, Clone, DeriveAutoPipeline)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Debug, Clone, DeriveAutoPipeline)]
struct Velocity {
    x: f32,
    y: f32,
    z: f32,
}

#[test]
fn derive_auto_pipeline_registers() {
    let mut store = SmartStore::new();
    store.register::<Position>();
    store.register::<Velocity>();

    let entity = store.create_entity();
    store.insert(entity, Position { x: 1.0, y: 0.0, z: 0.0 });
    store.insert(entity, Velocity { x: 0.1, y: 0.0, z: 0.0 });

    let pos_lane = store.read_lane::<Position>().unwrap();
    assert_eq!(pos_lane.get(entity).unwrap().x, 1.0);
}

#[test]
fn for_each_entity_macro_single_lane() {
    let mut store = SmartStore::new();
    store.register::<Position>();

    let e = store.create_entity();
    store.insert(e, Position { x: 10.0, y: 20.0, z: 30.0 });

    for_each_entity!(store, |pos: &mut Position| {
        pos.x += 1.0;
    });

    let lane = store.read_lane::<Position>().unwrap();
    assert_eq!(lane.get(e).unwrap().x, 11.0);
}

#[test]
fn for_each_entity_macro_two_lanes() {
    let mut store = SmartStore::new();
    store.register::<Position>();
    store.register::<Velocity>();

    let e = store.create_entity();
    store.insert(e, Position { x: 1.0, y: 2.0, z: 3.0 });
    store.insert(e, Velocity { x: 0.5, y: 0.0, z: 0.0 });

    for_each_entity!(store, |pos: &mut Position, vel: &Velocity| {
        pos.x += vel.x;
    });

    let lane = store.read_lane::<Position>().unwrap();
    assert!((lane.get(e).unwrap().x - 1.5).abs() < 1e-6);
}

#[smart_pipeline]
fn test_pipeline_hook() {
    let _x = 42;
}

#[test]
fn smart_pipeline_attribute_compiles() {
    test_pipeline_hook();
}

#[derive(Debug, Clone)]
struct Vec3(f32, f32, f32);

#[derive(Debug, Clone, DeriveAutoPipeline)]
#[pack]
struct Transform {
    position: Vec3,
    rotation: Vec3,
    scale: Vec3,
}

#[test]
fn pack_splits_into_fields() {
    let mut store = SmartStore::new();
    Transform::pack_register(&mut store);

    let e = store.create_entity();
    Transform::pack_insert(&mut store, e, Transform {
        position: Vec3(1.0, 0.0, 0.0),
        rotation: Vec3(0.0, 1.0, 0.0),
        scale: Vec3(1.0, 1.0, 1.0),
    });

    let pos_lane = store.read_lane::<Vec3>().unwrap();
    assert_eq!(pos_lane.get(e).unwrap().0, 1.0);
}

#[derive(PipelineConfig)]
struct GpuParticle {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(PipelineConfig)]
struct CpuInventory {
    name: String,
    count: u32,
}

#[test]
fn pipeline_config_detects_gpu() {
    assert_eq!(GpuParticle::lane_target(), ornis_core::TargetDiscriminant::Gpu);
}

#[test]
fn pipeline_config_detects_cpu() {
    assert_eq!(CpuInventory::lane_target(), ornis_core::TargetDiscriminant::Cpu);
}
