// Strong Confluence Tests - Determinism across thread counts
// Tests that parallel execution produces bitwise-identical results
// regardless of RAYON_NUM_THREADS setting.

use ornis_core::SmartStore;
use ornis_macros::for_each_entity;

// ===== Test Types =====

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct Velocity {
    x: f32,
    y: f32,
    z: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct Force {
    x: f32,
    y: f32,
    z: f32,
}

// ===== Helper Functions =====

fn run_with_threads<F>(threads: usize, f: F) -> Vec<Position>
where
    F: FnOnce(&mut SmartStore) -> Vec<Position>,
{
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }

    let mut store = SmartStore::new();
    store.register::<Position>();
    store.register::<Velocity>();
    store.register::<Force>();

    f(&mut store)
}

fn collect_positions(store: &SmartStore) -> Vec<Position> {
    let lane = store.read_lane::<Position>().unwrap();
    let mut positions: Vec<_> = lane.iter().copied().collect();
    positions.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
    positions
}

fn assert_bitwise_equal(a: &[Position], b: &[Position], test_name: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", test_name);
    for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
        let a_bytes = bytemuck::bytes_of(pa);
        let b_bytes = bytemuck::bytes_of(pb);
        assert_eq!(
            a_bytes, b_bytes,
            "{}: position[{}] differs at bytes",
            test_name, i
        );
    }
}

// ===== Test Cases =====

#[test]
fn strong_confluence_for_each_entity_single_lane() {
    let threads_1 = run_with_threads(1, |store| {
        for i in 0..1000 {
            let e = store.create_entity();
            store.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            );
            store.insert(
                e,
                Velocity {
                    x: 0.1,
                    y: 0.0,
                    z: 0.0,
                },
            );
        }

        for_each_entity!(store, |pos: &mut Position, vel: &Velocity| {
            pos.x += vel.x;
            pos.y += vel.y;
            pos.z += vel.z;
        });

        collect_positions(store)
    });

    let threads_32 = run_with_threads(32, |store| {
        for i in 0..1000 {
            let e = store.create_entity();
            store.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            );
            store.insert(
                e,
                Velocity {
                    x: 0.1,
                    y: 0.0,
                    z: 0.0,
                },
            );
        }

        for_each_entity!(store, |pos: &mut Position, vel: &Velocity| {
            pos.x += vel.x;
            pos.y += vel.y;
            pos.z += vel.z;
        });

        collect_positions(store)
    });

    assert_bitwise_equal(&threads_1, &threads_32, "for_each_entity_single_lane");
}

#[test]
fn strong_confluence_for_each_entity_two_lanes() {
    let threads_1 = run_with_threads(1, |store| {
        for i in 0..500 {
            let e = store.create_entity();
            store.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            );
            store.insert(
                e,
                Velocity {
                    x: 0.5,
                    y: -0.2,
                    z: 0.1,
                },
            );
        }

        for_each_entity!(store, |pos: &mut Position, vel: &Velocity| {
            pos.x += vel.x;
            pos.y += vel.y;
            pos.z += vel.z;
        });

        collect_positions(store)
    });

    let threads_32 = run_with_threads(32, |store| {
        for i in 0..500 {
            let e = store.create_entity();
            store.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            );
            store.insert(
                e,
                Velocity {
                    x: 0.5,
                    y: -0.2,
                    z: 0.1,
                },
            );
        }

        for_each_entity!(store, |pos: &mut Position, vel: &Velocity| {
            pos.x += vel.x;
            pos.y += vel.y;
            pos.z += vel.z;
        });

        collect_positions(store)
    });

    assert_bitwise_equal(&threads_1, &threads_32, "for_each_entity_two_lanes");
}

#[test]
fn strong_confluence_three_lanes() {
    let threads_1 = run_with_threads(1, |store| {
        for i in 0..500 {
            let e = store.create_entity();
            store.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            );
            store.insert(
                e,
                Velocity {
                    x: 0.1,
                    y: 0.02,
                    z: 0.0,
                },
            );
            store.insert(
                e,
                Force {
                    x: 0.01,
                    y: 0.0,
                    z: -0.005,
                },
            );
        }

        for_each_entity!(store, |pos: &mut Position,
                                 vel: &mut Velocity,
                                 force: &Force| {
            vel.x += force.x;
            vel.y += force.y;
            vel.z += force.z;

            pos.x += vel.x;
            pos.y += vel.y;
            pos.z += vel.z;
        });

        collect_positions(store)
    });

    let threads_32 = run_with_threads(32, |store| {
        for i in 0..500 {
            let e = store.create_entity();
            store.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            );
            store.insert(
                e,
                Velocity {
                    x: 0.1,
                    y: 0.02,
                    z: 0.0,
                },
            );
            store.insert(
                e,
                Force {
                    x: 0.01,
                    y: 0.0,
                    z: -0.005,
                },
            );
        }

        for_each_entity!(store, |pos: &mut Position,
                                 vel: &mut Velocity,
                                 force: &Force| {
            vel.x += force.x;
            vel.y += force.y;
            vel.z += force.z;

            pos.x += vel.x;
            pos.y += vel.y;
            pos.z += vel.z;
        });

        collect_positions(store)
    });

    assert_bitwise_equal(&threads_1, &threads_32, "three_lanes");
}

#[test]
fn strong_confluence_smart_pipeline_branching() {
    let threads_1 = run_with_threads(1, |store| {
        for i in 0..1000 {
            let e = store.create_entity();
            store.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            );
            store.insert(
                e,
                Velocity {
                    x: if i % 2 == 0 { 0.1 } else { -0.1 },
                    y: 0.0,
                    z: 0.0,
                },
            );
        }

        for_each_entity!(store, |pos: &mut Position, vel: &Velocity| {
            if vel.x > 0.0 {
                pos.x += vel.x * 2.0;
            } else {
                pos.x += vel.x * 0.5;
            }
            pos.y += vel.y;
            pos.z += vel.z;
        });

        collect_positions(store)
    });

    let threads_32 = run_with_threads(32, |store| {
        for i in 0..1000 {
            let e = store.create_entity();
            store.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            );
            store.insert(
                e,
                Velocity {
                    x: if i % 2 == 0 { 0.1 } else { -0.1 },
                    y: 0.0,
                    z: 0.0,
                },
            );
        }

        for_each_entity!(store, |pos: &mut Position, vel: &Velocity| {
            if vel.x > 0.0 {
                pos.x += vel.x * 2.0;
            } else {
                pos.x += vel.x * 0.5;
            }
            pos.y += vel.y;
            pos.z += vel.z;
        });

        collect_positions(store)
    });

    assert_bitwise_equal(&threads_1, &threads_32, "smart_pipeline_branching");
}

#[test]
fn strong_confluence_entity_creation_order() {
    let threads_1 = run_with_threads(1, |store| {
        let mut entities = Vec::new();
        for i in 0..500 {
            let e = store.create_entity();
            entities.push(e);
            store.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            );
        }

        for e in entities.iter().rev() {
            if let Some(pos) = store.write_lane::<Position>().unwrap().get_mut(*e) {
                pos.x += 100.0;
            }
        }

        collect_positions(store)
    });

    let threads_32 = run_with_threads(32, |store| {
        let mut entities = Vec::new();
        for i in 0..500 {
            let e = store.create_entity();
            entities.push(e);
            store.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            );
        }

        for e in entities.iter().rev() {
            if let Some(pos) = store.write_lane::<Position>().unwrap().get_mut(*e) {
                pos.x += 100.0;
            }
        }

        collect_positions(store)
    });

    assert_bitwise_equal(&threads_1, &threads_32, "entity_creation_order");
}

#[test]
fn strong_confluence_defrag() {
    let threads_1 = run_with_threads(1, |store| {
        let mut entities = Vec::new();
        for i in 0..1000 {
            let e = store.create_entity();
            entities.push(e);
            store.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            );
        }

        for i in (0..1000).step_by(3) {
            store.destroy_entity(entities[i]);
        }

        if let Some(mut lane) = store.write_lane::<Position>() {
            lane.defrag();
        }

        collect_positions(store)
    });

    let threads_32 = run_with_threads(32, |store| {
        let mut entities = Vec::new();
        for i in 0..1000 {
            let e = store.create_entity();
            entities.push(e);
            store.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            );
        }

        for i in (0..1000).step_by(3) {
            store.destroy_entity(entities[i]);
        }

        if let Some(mut lane) = store.write_lane::<Position>() {
            lane.defrag();
        }

        collect_positions(store)
    });

    assert_bitwise_equal(&threads_1, &threads_32, "defrag");
}

#[test]
fn strong_confluence_for_each_entity_pure_math() {
    let threads_1 = run_with_threads(1, |store| {
        for i in 0..2000 {
            let e = store.create_entity();
            store.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            );
            store.insert(
                e,
                Velocity {
                    x: 0.01,
                    y: 0.02,
                    z: 0.03,
                },
            );
            store.insert(
                e,
                Force {
                    x: 0.001,
                    y: 0.002,
                    z: 0.003,
                },
            );
        }

        for_each_entity!(store, |pos: &mut Position,
                                 vel: &mut Velocity,
                                 force: &Force| {
            vel.x += force.x;
            vel.y += force.y;
            vel.z += force.z;

            pos.x += vel.x;
            pos.y += vel.y;
            pos.z += vel.z;
        });

        collect_positions(store)
    });

    let threads_32 = run_with_threads(32, |store| {
        for i in 0..2000 {
            let e = store.create_entity();
            store.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            );
            store.insert(
                e,
                Velocity {
                    x: 0.01,
                    y: 0.02,
                    z: 0.03,
                },
            );
            store.insert(
                e,
                Force {
                    x: 0.001,
                    y: 0.002,
                    z: 0.003,
                },
            );
        }

        for_each_entity!(store, |pos: &mut Position,
                                 vel: &mut Velocity,
                                 force: &Force| {
            vel.x += force.x;
            vel.y += force.y;
            vel.z += force.z;

            pos.x += vel.x;
            pos.y += vel.y;
            pos.z += vel.z;
        });

        collect_positions(store)
    });

    assert_bitwise_equal(&threads_1, &threads_32, "smart_pipeline_pure_math");
}
