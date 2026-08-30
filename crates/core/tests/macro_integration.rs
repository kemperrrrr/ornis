//! Integration tests for `ornis_macros` derives and `for_each_entity` expansion.

#![allow(dead_code)]

use ornis_core::{Pack as _, PipelineConfig as _, SmartStore};
use ornis_macros::{
    AutoPipeline as DeriveAutoPipeline, Pack, PipelineConfig, for_each_entity, smart_pipeline,
};

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

#[derive(Debug, Clone, DeriveAutoPipeline)]
struct Force {
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
    store.insert(
        entity,
        Position {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
    );
    store.insert(
        entity,
        Velocity {
            x: 0.1,
            y: 0.0,
            z: 0.0,
        },
    );

    let pos_lane = store.read_lane::<Position>().unwrap();
    assert_eq!(pos_lane.get(entity).unwrap().x, 1.0);
}

#[test]
fn for_each_entity_macro_single_lane() {
    let mut store = SmartStore::new();
    store.register::<Position>();

    let e = store.create_entity();
    store.insert(
        e,
        Position {
            x: 10.0,
            y: 20.0,
            z: 30.0,
        },
    );

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
    store.insert(
        e,
        Position {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        },
    );
    store.insert(
        e,
        Velocity {
            x: 0.5,
            y: 0.0,
            z: 0.0,
        },
    );

    for_each_entity!(store, |pos: &mut Position, vel: &Velocity| {
        pos.x += vel.x;
    });

    let lane = store.read_lane::<Position>().unwrap();
    assert!((lane.get(e).unwrap().x - 1.5).abs() < 1e-6);
}

/// Regression (audit §2.3, backlog #18): with three lanes and partial
/// ownership, an entity missing a component from lane 3 must be skipped
/// by the intersection — not panic the loop on `get(entity).unwrap()`,
/// which is what the pre-fix codegen (zip of only the first two lanes)
/// did.
#[test]
fn for_each_entity_macro_three_lanes_skips_partial_ownership() {
    let mut store = SmartStore::new();
    store.register::<Position>();
    store.register::<Velocity>();
    store.register::<Force>();

    // Full set: must be visited and updated.
    let full = store.create_entity();
    store.insert(
        full,
        Position {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
    );
    store.insert(
        full,
        Velocity {
            x: 0.5,
            y: 0.0,
            z: 0.0,
        },
    );
    store.insert(
        full,
        Force {
            x: 2.0,
            y: 0.0,
            z: 0.0,
        },
    );

    // Position+Velocity, but no Force: pre-fix this entity entered the
    // zip of the first two lanes and panicked the loop; now it must be
    // silently excluded and left untouched.
    let partial = store.create_entity();
    store.insert(
        partial,
        Position {
            x: 10.0,
            y: 0.0,
            z: 0.0,
        },
    );
    store.insert(
        partial,
        Velocity {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
    );

    let mut visited = 0usize;
    for_each_entity!(store, |pos: &mut Position,
                             vel: &mut Velocity,
                             force: &Force| {
        visited += 1;
        vel.x += force.x;
        pos.x += vel.x;
    });

    assert_eq!(visited, 1);
    let pos_lane = store.read_lane::<Position>().unwrap();
    assert_eq!(pos_lane.get(full).unwrap().x, 3.5); // 1.0 + 0.5 + 2.0
    assert_eq!(pos_lane.get(partial).unwrap().x, 10.0);
}

#[smart_pipeline]
fn test_pipeline_hook() {
    let _x = 42;
}

#[test]
fn smart_pipeline_attribute_compiles() {
    test_pipeline_hook();
}

// ===== smart_pipeline behavioral tests =====

#[smart_pipeline]
fn integrate_positions(store: &SmartStore, dt: f32) -> usize {
    // Code before the loops must be preserved by the macro.
    let started = dt > 0.0;
    assert!(started);

    let mut positions = store.write_lane::<Position>().expect("position lane");
    let velocities = store.read_lane::<Velocity>().expect("velocity lane");

    for (pos, vel) in positions.iter_mut().zip(velocities.iter()) {
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
        pos.z += vel.z * dt;
    }

    // Code after the loop must run; the tail expression must be returned
    // through the macro-generated wrapper.
    positions.len()
}

fn store_with_moving_entities(n: usize) -> SmartStore {
    let mut store = SmartStore::new();
    store.register::<Position>();
    store.register::<Velocity>();
    for _ in 0..n {
        let e = store.create_entity();
        store.insert(
            e,
            Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        );
        store.insert(
            e,
            Velocity {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
        );
    }
    store
}

#[test]
fn smart_pipeline_two_lane_loop_updates_every_entity() {
    let store = store_with_moving_entities(100);

    let processed = integrate_positions(&store, 0.5);

    assert_eq!(processed, 100);
    let lane = store.read_lane::<Position>().unwrap();
    for pos in lane.iter() {
        assert_eq!((pos.x, pos.y, pos.z), (0.5, 1.0, 1.5));
    }
}

#[smart_pipeline]
fn shift_positions_twice(store: &SmartStore) -> f32 {
    let mut positions = store.write_lane::<Position>().expect("position lane");

    for pos in positions.iter_mut() {
        pos.x += 1.0;
    }

    // Code between two loops must be preserved and executed.
    let shift = positions.len() as f32 * 10.0;

    for pos in positions.iter_mut() {
        pos.x += shift;
    }

    shift
}

#[test]
fn smart_pipeline_preserves_code_between_loops() {
    let store = store_with_moving_entities(4);

    let shift = shift_positions_twice(&store);

    assert_eq!(shift, 40.0);
    let lane = store.read_lane::<Position>().unwrap();
    for pos in lane.iter() {
        assert_eq!(pos.x, 1.0 + 40.0);
    }
}

// `total` is mutated across iterations, so the loop must stay sequential —
// and, critically, still iterate over every entity (not run its body once).
#[allow(deprecated)] // the macro intentionally warns that the loops stay sequential
#[smart_pipeline]
fn sum_positions_x(store: &SmartStore) -> f32 {
    let mut warmup = 0;
    for i in 0..3 {
        warmup += i;
    }
    assert_eq!(warmup, 3);

    let mut total = 0.0;
    let positions = store.read_lane::<Position>().expect("position lane");
    for pos in positions.iter() {
        total += pos.x;
    }
    total
}

#[test]
fn smart_pipeline_non_parallelizable_loop_still_iterates() {
    let mut store = SmartStore::new();
    store.register::<Position>();
    for i in 0..10 {
        let e = store.create_entity();
        store.insert(
            e,
            Position {
                x: i as f32,
                y: 0.0,
                z: 0.0,
            },
        );
    }

    let total = sum_positions_x(&store);

    assert_eq!(total, (0..10).sum::<i32>() as f32);
}

#[derive(Debug, Clone, DeriveAutoPipeline)]
#[pack]
struct TransformPacked {
    position: Vec3,
    rotation: Vec3,
    scale: Vec3,
}

#[test]
fn pack_splits_into_fields() {
    let mut store = SmartStore::new();
    TransformPacked::pack_register(&mut store);

    let e = store.create_entity();
    TransformPacked::pack_insert(
        &mut store,
        e,
        TransformPacked {
            position: Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            rotation: Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            scale: Vec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
        },
    );

    let pos_lane = store.read_lane::<Vec3>().unwrap();
    assert_eq!(pos_lane.get(e).unwrap().x, 1.0);
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
    assert_eq!(
        GpuParticle::lane_target(),
        ornis_core::TargetDiscriminant::Gpu
    );
}

#[test]
fn pipeline_config_detects_cpu() {
    assert_eq!(
        CpuInventory::lane_target(),
        ornis_core::TargetDiscriminant::Cpu
    );
}

// ===== Pack derive tests =====

#[derive(Debug, Clone, PartialEq, Pack)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Default for Vec3 {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Pack)]
struct Quat {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl Default for Quat {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Pack)]
struct Transform {
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
}

#[test]
fn pack_derive_simple_struct() {
    let mut store = SmartStore::new();
    Transform::pack_register(&mut store);

    let e = store.create_entity();
    let transform = Transform {
        position: Vec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        },
        rotation: Quat::default(),
        scale: Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
    };
    transform.pack_insert(&mut store, e);

    let retrieved = Transform::pack_get(&store, e).unwrap();
    assert_eq!(retrieved, transform);
}

#[test]
fn pack_derive_get_reconstructs() {
    let mut store = SmartStore::new();
    Transform::pack_register(&mut store);

    let e = store.create_entity();
    let original = Transform {
        position: Vec3 {
            x: 5.0,
            y: -1.0,
            z: 2.5,
        },
        rotation: Quat::default(),
        scale: Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        },
    };
    original.pack_insert(&mut store, e);

    let retrieved = Transform::pack_get(&store, e).unwrap();
    assert_eq!(retrieved, original);
}

#[test]
fn pack_derive_get_mut_modifies() {
    let mut store = SmartStore::new();
    Transform::pack_register(&mut store);

    let e = store.create_entity();
    let original = Transform {
        position: Vec3::default(),
        rotation: Quat::default(),
        scale: Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
    };
    original.pack_insert(&mut store, e);

    if let Some(mut pack_mut) = Transform::pack_get_mut(&mut store, e) {
        pack_mut.position().x = 10.0;
        pack_mut.scale().x = 0.5;
        pack_mut.scale().y = 0.5;
        pack_mut.scale().z = 0.5;
    }

    let retrieved = Transform::pack_get(&store, e).unwrap();
    assert_eq!(retrieved.position.x, 10.0);
    assert_eq!(retrieved.scale.x, 0.5);
}

#[test]
fn pack_derive_round_trip() {
    let mut store = SmartStore::new();
    Transform::pack_register(&mut store);

    let e = store.create_entity();
    let original = Transform {
        position: Vec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        },
        rotation: Quat {
            x: 0.1,
            y: 0.2,
            z: 0.3,
            w: 0.9,
        },
        scale: Vec3 {
            x: 1.5,
            y: 2.5,
            z: 3.5,
        },
    };
    original.pack_insert(&mut store, e);

    let retrieved = Transform::pack_get(&store, e).unwrap();
    assert_eq!(retrieved, original);

    if let Some(mut pack_mut) = Transform::pack_get_mut(&mut store, e) {
        pack_mut.position().x += 1.0;
    }
    let retrieved2 = Transform::pack_get(&store, e).unwrap();
    assert_eq!(retrieved2.position.x, 2.0);
    assert_eq!(retrieved2.position.y, 2.0);
    assert_eq!(retrieved2.position.z, 3.0);
}

// Test 2: Struct with multiple fields of same type (each gets unique lane via wrapper)
#[derive(Debug, Clone, PartialEq, Pack)]
struct PhysicsState {
    position: Vec3,
    velocity: Vec3,
    acceleration: Vec3,
    mass: f32,
    restitution: f32,
}

#[test]
fn pack_derive_duplicate_types_share_lane() {
    let mut store = SmartStore::new();
    PhysicsState::pack_register(&mut store);

    let e = store.create_entity();
    let original = PhysicsState {
        position: Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        velocity: Vec3 {
            x: 0.1,
            y: 0.0,
            z: 0.0,
        },
        acceleration: Vec3::default(),
        mass: 1.0,
        restitution: 0.5,
    };
    original.pack_insert(&mut store, e);

    let retrieved = PhysicsState::pack_get(&store, e).unwrap();
    assert_eq!(retrieved, original);
}
