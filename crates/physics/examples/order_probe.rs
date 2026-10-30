//! Body-order locality probe (issue #178).
//!
//! Builds the same physical scene twice — once with bodies inserted in
//! spatial (island-contiguous) order, once with the dynamic bodies shuffled
//! by a fixed-seed Fisher–Yates — then times identical `step` loops. The
//! only difference is the layout of `SequentialImpulseEngine::bodies`, i.e.
//! the memory-locality hypothesis behind sorting bodies by island/Morton.
//!
//! Run in release on an otherwise idle machine:
//!
//! ```text
//! cargo run --release -p ornis-physics --example order_probe
//! ```
//!
//! Repeat the command several times and compare medians; the probe itself
//! also interleaves repeats and prints median/min/max per scene.

use std::hint::black_box;
use std::time::Instant;

use glam::Vec3;
use ornis_physics::{BodyType, PhysicsEngine, RigidBody, SequentialImpulseEngine};

const DT: f32 = 1.0 / 60.0;
const BOX_HALF: f32 = 0.4;
const PITCH: f32 = 2.0;
const LEVEL_PITCH: f32 = 0.81;
const REST_Y: f32 = 0.4;
const FLOOR_HALF_Y: f32 = 0.5;
const TILE_HALF: f32 = 5.0;

fn keep_static(kind: BodyType, layout: Layout) -> bool {
    match layout {
        Layout::Spatial | Layout::ShuffledAll | Layout::Morton => false,
        Layout::ShuffledDynamic => kind != BodyType::Dynamic,
        Layout::ShuffledStatic => kind == BodyType::Dynamic,
    }
}

/// Reproducible Fisher–Yates over the given index positions (tiny LCG, no
/// extra dependency).
fn shuffle_positions(bodies: &mut [RigidBody], positions: &[usize], seed: u64) {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as usize
    };
    for i in (1..positions.len()).rev() {
        let j = next() % (i + 1);
        bodies.swap(positions[i], positions[j]);
    }
}

fn floor(half_xz: f32) -> RigidBody {
    RigidBody::new_box(
        Vec3::new(0.0, -FLOOR_HALF_Y, 0.0),
        Vec3::new(half_xz, FLOOR_HALF_Y, half_xz),
        0.0,
    )
}

/// G×G independent 4-box towers on one floor: many disjoint islands.
fn scene_islands_grid(g: u32) -> Vec<RigidBody> {
    let mut bodies = vec![floor(100.0)];
    let half = Vec3::splat(BOX_HALF);
    for gx in 0..g {
        for gz in 0..g {
            let x = (gx as f32 - g as f32 / 2.0) * PITCH;
            let z = (gz as f32 - g as f32 / 2.0) * PITCH;
            for level in 0..4 {
                bodies.push(RigidBody::new_box(
                    Vec3::new(x, REST_Y + 0.5 + level as f32 * LEVEL_PITCH, z),
                    half,
                    1.0,
                ));
            }
        }
    }
    bodies
}

/// N single resting boxes on tiled floors: broadphase-dominated.
fn scene_body_grid(n: u32) -> Vec<RigidBody> {
    let side = (n as f32).sqrt().ceil() as u32;
    let span = side as f32 * PITCH;
    let tiles = (span / (2.0 * TILE_HALF)).ceil() as i32;
    let mut bodies = Vec::new();
    for tx in 0..tiles {
        for tz in 0..tiles {
            let x = (tx as f32 - tiles as f32 / 2.0 + 0.5) * 2.0 * TILE_HALF;
            let z = (tz as f32 - tiles as f32 / 2.0 + 0.5) * 2.0 * TILE_HALF;
            bodies.push(RigidBody::new_box(
                Vec3::new(x, -FLOOR_HALF_Y, z),
                Vec3::new(TILE_HALF, FLOOR_HALF_Y, TILE_HALF),
                0.0,
            ));
        }
    }
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = (gx as f32 - side as f32 / 2.0) * PITCH;
        let z = (gz as f32 - side as f32 / 2.0) * PITCH;
        bodies.push(RigidBody::new_box(
            Vec3::new(x, REST_Y + 1.0, z),
            Vec3::splat(BOX_HALF),
            1.0,
        ));
    }
    bodies
}

/// One tall stack: a single island, the worst case for any island ordering.
fn scene_stack(n: u32) -> Vec<RigidBody> {
    let mut bodies = vec![floor(10.0)];
    for level in 0..n {
        bodies.push(RigidBody::new_box(
            Vec3::new(0.0, REST_Y + 3.0 + level as f32 * LEVEL_PITCH, 0.0),
            Vec3::splat(BOX_HALF),
            1.0,
        ));
    }
    bodies
}

/// N boxes falling into a heap: islands merge/split while solving, so the
/// order of `bodies` is maximally at odds with the live island layout.
fn scene_pile(n: u32) -> Vec<RigidBody> {
    let side = (n as f32).sqrt().ceil() as u32;
    let mut bodies = vec![floor(40.0)];
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = (gx as f32 - side as f32 / 2.0) * 1.1;
        let z = (gz as f32 - side as f32 / 2.0) * 1.1;
        // Deterministic height ladder so the heap is dense and churning.
        let y = 3.0 + ((i * 37) % 97) as f32 * 0.11;
        bodies.push(RigidBody::new_box(
            Vec3::new(x, y, z),
            Vec3::splat(BOX_HALF),
            1.0,
        ));
    }
    bodies
}

/// 10-bit Morton code of a world position (x, z, y), the ordering key an
/// in-engine reorder would use.
fn morton(b: &RigidBody) -> u64 {
    fn quant(v: f32) -> u64 {
        ((v + 64.0) * 8.0).clamp(0.0, 1023.0) as u64
    }
    fn spread(mut x: u64) -> u64 {
        x &= 0x3ff;
        x = (x | (x << 16)) & 0x0300_00ff_0000_00ff;
        x = (x | (x << 8)) & 0x000f_00f0_00f0_00f0;
        x = (x | (x << 4)) & 0x0c30_c30c_30c3_0c30;
        x = (x | (x << 2)) & 0x2492_4924_9249_2492;
        x
    }
    let p = b.position;
    spread(quant(p.x)) | (spread(quant(p.y)) << 1) | (spread(quant(p.z)) << 2)
}

/// Median cost of one full copy and one copy+sort of `bodies` — the lower
/// bound an in-engine physical reorder pays per refresh.
fn reorder_cost(scene: &Scene, repeats: usize) -> (f64, f64) {
    let mut copy = Vec::new();
    let mut sorted = Vec::new();
    for _ in 0..repeats {
        let t = Instant::now();
        black_box(black_box(scene.bodies.clone()));
        copy.push(t.elapsed().as_secs_f64() * 1e9);
    }
    for _ in 0..repeats {
        let t = Instant::now();
        let mut v = black_box(scene.bodies.clone());
        v.sort_by_key(morton);
        black_box(&v);
        sorted.push(t.elapsed().as_secs_f64() * 1e9);
    }
    copy.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (copy[copy.len() / 2], sorted[sorted.len() / 2])
}

struct Scene {
    name: &'static str,
    bodies: Vec<RigidBody>,
    settle: u32,
    measure_steps: u32,
}

fn build(bodies: Vec<RigidBody>) -> SequentialImpulseEngine {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    for body in bodies {
        physics.add_body(body);
    }
    physics
}

#[derive(Clone, Copy, PartialEq)]
enum Layout {
    Spatial,
    ShuffledAll,
    /// Static floors stay in place; only dynamic indices are permuted. If the
    /// layout hypothesis holds, this reproduces the full-shuffle cost.
    ShuffledDynamic,
    /// Dynamic bodies stay spatial; only statics move. Control: must be flat.
    ShuffledStatic,
    /// Morton-sorted layout: the target an in-engine spatial reorder would
    /// produce (comparison sort cost shown separately).
    Morton,
}

/// One timed run: fresh engine, optional permutation, `measure_steps` steps.
fn run_once(scene: &Scene, layout: Layout, seed: u64) -> f64 {
    let mut bodies = scene.bodies.clone();
    if layout == Layout::Morton {
        bodies.sort_by_key(morton);
    } else {
        let positions: Vec<usize> = (0..bodies.len())
            .filter(|&i| !keep_static(bodies[i].body_type, layout))
            .collect();
        if layout != Layout::Spatial {
            shuffle_positions(&mut bodies, &positions, seed);
        }
    }
    let mut physics = build(bodies);
    for _ in 0..scene.settle {
        physics.step(DT);
    }
    let start = Instant::now();
    for _ in 0..scene.measure_steps {
        black_box(&mut physics).step(black_box(DT));
    }
    start.elapsed().as_secs_f64() * 1e9 / f64::from(scene.measure_steps)
}

fn median(mut samples: Vec<f64>) -> (f64, f64, f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    let q = |p: f64| samples[((n as f64 - 1.0) * p).round() as usize];
    (q(0.5), q(0.25), q(0.75), samples[0])
}

fn main() {
    let scenes = [
        Scene {
            name: "islands_grid_16x16",
            bodies: scene_islands_grid(16),
            settle: 0,
            measure_steps: 90,
        },
        Scene {
            name: "small_grid_4x4",
            bodies: scene_islands_grid(4),
            settle: 0,
            measure_steps: 180,
        },
        Scene {
            name: "body_grid_1000",
            bodies: scene_body_grid(1_000),
            settle: 0,
            measure_steps: 60,
        },
        Scene {
            name: "body_grid_10000",
            bodies: scene_body_grid(10_000),
            settle: 0,
            measure_steps: 20,
        },
        Scene {
            name: "stack_128_single_island",
            bodies: scene_stack(128),
            settle: 0,
            measure_steps: 90,
        },
        Scene {
            name: "pile_2000_active",
            bodies: scene_pile(2_000),
            settle: 0,
            measure_steps: 60,
        },
    ];
    let repeats = 11;
    const LAYOUTS: [Layout; 5] = [
        Layout::Spatial,
        Layout::ShuffledAll,
        Layout::ShuffledDynamic,
        Layout::ShuffledStatic,
        Layout::Morton,
    ];
    println!("rigid_body_size_bytes={}", std::mem::size_of::<RigidBody>());
    println!(
        "scene,bodies,layout,runs,median_ns_per_step,q1,q3,min,ratio_over_spatial,copy_ns,sort_ns"
    );
    for scene in &scenes {
        let mut samples: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::with_capacity(repeats));
        // Interleave the layouts inside every repeat so thermal drift and
        // background load hit all variants alike. Every repeat also draws a
        // fresh permutation seed, so the shuffled medians average over
        // arrangements instead of pinning one lucky/unlucky layout.
        for repeat in 0..repeats {
            let seed = 0x9E37_79B9_7F4A_7C15u64
                .wrapping_add((repeat as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
            for (k, layout) in LAYOUTS.iter().enumerate() {
                samples[k].push(run_once(scene, *layout, seed));
            }
        }
        let (copy_ns, sort_ns) = reorder_cost(scene, repeats);
        let base = median(samples[0].clone()).0;
        let names = [
            "spatial",
            "shuffled_all",
            "shuffled_dynamic",
            "shuffled_static",
            "morton",
        ];
        for (k, name) in names.iter().enumerate() {
            let (med, q1, q3, min) = median(samples[k].clone());
            println!(
                "{},{},{},{},{:.0},{:.0},{:.0},{:.0},{:.3},{:.0},{:.0}",
                scene.name,
                scene.bodies.len(),
                name,
                repeats,
                med,
                q1,
                q3,
                min,
                med / base,
                copy_ns,
                sort_ns
            );
        }
    }
}
