//! Cross-engine physics shootout: ornis `SequentialImpulseEngine` vs
//! Rapier (native pipeline) vs Box3D (via `boxddd`).
//!
//! Every engine runs byte-identical scenes (same box geometry, gravity,
//! `dt = 1/60`): a tiled resting grid at 1k/10k bodies (broadphase-heavy),
//! a tall 28-box stack (solver-stability-heavy) and a field of independent
//! 4-box islands (islands-heavy). Each bench warms the world up, settles it,
//! then reports the steady-state single-step cost through criterion.
//!
//! Engine selection is by cargo feature: ornis always runs; Rapier needs
//! `--features rapier`, Box3D needs `--features box3d`. Those optional deps
//! are currently omitted from `Cargo.toml` (nalgebra convert-glam030/031/032
//! trips the workspace `cargo outdated` hard gate); the cfg-gated backends
//! below stay ready to restore. Jolt is deliberately
//! absent: `jolt-sys 0.1.5` hardcodes a `Visual Studio 16 2019` CMake
//! generator plus Windows-only link libs, so it cannot build on macOS/Linux
//! (see the shootout report); wiring it needs an upstream fix or a new
//! binding first.
//!
//! Fairness note (also stated in the report): the engines run different
//! solvers that need different iteration counts for equal quality, so this
//! is a "defaults vs defaults plus one tuned run" comparison, NOT an
//! equal-quality comparison. Stack drift after a fixed 600-step horizon is
//! printed next to every stack bench (`SHOOTOUT drift ...`) so time and
//! stability stay side by side.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use glam::Vec3;

use ornis_physics::{PhysicsEngine, RigidBody, SequentialImpulseEngine};

// ---------------------------------------------------------------------------
// Shared scene description (identical for every engine).
// ---------------------------------------------------------------------------

/// World-space gravity shared by all scenes and engines.
const GRAVITY_Y: f32 = -9.81;
/// Fixed step shared by all scenes and engines.
const DT: f32 = 1.0 / 60.0;
/// Dynamic box half-extents shared by all scenes and engines.
const BOX_HALF: f32 = 0.4;
/// Horizontal pitch of resting boxes in grid/island scenes.
const PITCH_XZ: f32 = 2.0;
/// Vertical pitch of stacked boxes (half 0.4 + 0.01 settle gap).
const STACK_PITCH_Y: f32 = 0.81;
/// Rest height of a box sitting on a floor whose top face is y = 0.
const REST_Y: f32 = 0.4;
/// Warmup steps for grid/island scenes (no drift metric there).
const WARMUP_STEPS: usize = 60;
/// Fixed horizon for the stack drift metric (10 s of sim time).
const DRIFT_STEPS: usize = 600;
/// Box mass shared by all engines (Box3D reaches it via density, see below).
const BOX_MASS: f32 = 1.0;
/// Box3D density giving [`BOX_MASS`] for a 0.8^3 box (mass = density*volume).
#[cfg(feature = "box3d")]
const BOX3D_DENSITY: f32 = BOX_MASS / 0.512;

// ---------------------------------------------------------------------------
// Small helpers (panics, never unwrap: keeps `-D warnings` clippy clean).
// ---------------------------------------------------------------------------

/// Unwrap a fallible setup step with a named context.
#[cfg(feature = "box3d")]
fn must<T, E: std::fmt::Debug>(result: Result<T, E>, what: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("shootout setup failed ({what}): {error:?}"),
    }
}

// ---------------------------------------------------------------------------
// ornis + Rapier: both behind `PhysicsEngine`, so one generic setup covers
// default runs of both engines.
// ---------------------------------------------------------------------------

/// Tiled static floor plus `n` dynamic boxes at rest, mirroring the
/// `solver_bench` body-grid precedent (tiling keeps broadphase honest).
fn build_grid<E: PhysicsEngine>(engine: &mut E, n: u32) {
    let side = (n as f32).sqrt().ceil() as u32;
    let span = side as f32 * PITCH_XZ;
    let tile_half = 5.0f32;
    let tiles = (span / (2.0 * tile_half)).ceil() as i32;
    for tx in 0..tiles {
        for tz in 0..tiles {
            let x = (tx as f32 - tiles as f32 / 2.0 + 0.5) * 2.0 * tile_half;
            let z = (tz as f32 - tiles as f32 / 2.0 + 0.5) * 2.0 * tile_half;
            engine.add_body(RigidBody::new_box(
                Vec3::new(x, -0.5, z),
                Vec3::new(tile_half, 0.5, tile_half),
                0.0,
            ));
        }
    }
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        let x = (gx as f32 - side as f32 / 2.0) * PITCH_XZ;
        let z = (gz as f32 - side as f32 / 2.0) * PITCH_XZ;
        engine.add_body(RigidBody::new_box(
            Vec3::new(x, REST_Y, z),
            Vec3::splat(BOX_HALF),
            BOX_MASS,
        ));
    }
}

/// One tall stack of `levels` boxes on a static floor; returns the handle
/// of the top box for the drift metric.
fn build_stack<E: PhysicsEngine>(engine: &mut E, levels: u32) -> ornis_physics::body::BodyHandle {
    engine.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    let mut top = ornis_physics::body::BodyHandle::from(0u32);
    for level in 0..levels {
        top = engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, REST_Y + level as f32 * STACK_PITCH_Y, 0.0),
            Vec3::splat(BOX_HALF),
            BOX_MASS,
        ));
    }
    top
}

/// `g x g` independent 4-box islands on one big static floor.
fn build_islands<E: PhysicsEngine>(engine: &mut E, g: u32) {
    engine.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(100.0, 0.5, 100.0),
        0.0,
    ));
    for gx in 0..g {
        for gz in 0..g {
            let x = (gx as f32 - g as f32 / 2.0) * PITCH_XZ;
            let z = (gz as f32 - g as f32 / 2.0) * PITCH_XZ;
            for level in 0..4 {
                engine.add_body(RigidBody::new_box(
                    Vec3::new(x, REST_Y + level as f32 * STACK_PITCH_Y, z),
                    Vec3::splat(BOX_HALF),
                    BOX_MASS,
                ));
            }
        }
    }
}

/// ornis engine with the requested substep count (12 = default tuning).
fn ornis_world(substeps: u32) -> SequentialImpulseEngine {
    let mut engine = SequentialImpulseEngine::new(Vec3::new(0.0, GRAVITY_Y, 0.0));
    engine.set_substeps(substeps);
    engine
}

/// ornis per-step wall-clock breakdown (diagnostic only).
fn ornis_breakdown(engine: &SequentialImpulseEngine) -> String {
    let timing = engine.step_timing();
    let stats = engine.broadphase_stats();
    format!(
        "broad={:.3}ms narrow={:.3}ms solver={:.3}ms island={:.3}ms trigger={:.3}ms substeps={} pairs={} shed={}",
        timing.broad_phase_ms,
        timing.narrow_phase_ms,
        timing.solver_ms,
        timing.island_ms,
        timing.trigger_ms,
        timing.substeps,
        stats.candidate_pairs,
        engine.last_substep_shed(),
    )
}

// ---------------------------------------------------------------------------
// Rapier tuned run: native pipeline, one knob — `num_solver_iterations`.
// ---------------------------------------------------------------------------

/// Native Rapier world mirroring the adapter's body mapping (dynamic cuboid
/// colliders, mass 1, friction 0.5, restitution 0.3), with an explicit
/// solver-iteration count (4 = Rapier default).
#[cfg(feature = "rapier")]
struct NativeRapier {
    pipeline: rapier3d::pipeline::PhysicsPipeline,
    islands: rapier3d::dynamics::IslandManager,
    broad: rapier3d::geometry::BroadPhaseBvh,
    narrow: rapier3d::geometry::NarrowPhase,
    bodies: rapier3d::dynamics::RigidBodySet,
    colliders: rapier3d::geometry::ColliderSet,
    impulse_joints: rapier3d::dynamics::ImpulseJointSet,
    multibody_joints: rapier3d::dynamics::MultibodyJointSet,
    soft_bodies: rapier3d::dynamics::SoftBodySet,
    ccd: rapier3d::dynamics::CCDSolver,
    params: rapier3d::dynamics::IntegrationParameters,
    top: Option<rapier3d::dynamics::RigidBodyHandle>,
}

#[cfg(feature = "rapier")]
impl NativeRapier {
    fn fresh(iterations: usize) -> Self {
        let params = rapier3d::dynamics::IntegrationParameters {
            dt: DT,
            num_solver_iterations: iterations,
            ..rapier3d::dynamics::IntegrationParameters::default()
        };
        Self {
            pipeline: rapier3d::pipeline::PhysicsPipeline::new(),
            islands: rapier3d::dynamics::IslandManager::new(),
            broad: rapier3d::geometry::BroadPhaseBvh::new(),
            narrow: rapier3d::geometry::NarrowPhase::new(),
            bodies: rapier3d::dynamics::RigidBodySet::new(),
            colliders: rapier3d::geometry::ColliderSet::new(),
            impulse_joints: rapier3d::dynamics::ImpulseJointSet::new(),
            multibody_joints: rapier3d::dynamics::MultibodyJointSet::new(),
            soft_bodies: rapier3d::dynamics::SoftBodySet::new(),
            ccd: rapier3d::dynamics::CCDSolver::new(),
            params,
            top: None,
        }
    }

    fn add_box(
        &mut self,
        pos: [f32; 3],
        half: [f32; 3],
        mass: f32,
    ) -> rapier3d::dynamics::RigidBodyHandle {
        use rapier3d::dynamics::RigidBodyBuilder;
        use rapier3d::geometry::ColliderBuilder;
        let body = if mass > 0.0 {
            RigidBodyBuilder::dynamic()
                .translation(rapier3d::math::Vector::new(pos[0], pos[1], pos[2]))
                .build()
        } else {
            RigidBodyBuilder::fixed()
                .translation(rapier3d::math::Vector::new(pos[0], pos[1], pos[2]))
                .build()
        };
        let handle = self.bodies.insert(body);
        let mut collider = ColliderBuilder::cuboid(half[0], half[1], half[2])
            .friction(0.5)
            .restitution(0.3);
        if mass > 0.0 {
            collider = collider.mass(mass);
        }
        self.colliders
            .insert_with_parent(collider, handle, &mut self.bodies);
        handle
    }

    fn step(&mut self) {
        let gravity = rapier3d::math::Vector::new(0.0, GRAVITY_Y, 0.0);
        self.pipeline.step(
            gravity,
            &self.params,
            &mut self.islands,
            &mut self.broad,
            &mut self.narrow,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.soft_bodies,
            &mut self.ccd,
            &(),
            &(),
        );
    }

    fn top_y(&self) -> f32 {
        self.top
            .and_then(|h| self.bodies.get(h))
            .map(|b| b.translation().y)
            .unwrap_or(f32::NAN)
    }

    fn top_xz(&self) -> f32 {
        self.top
            .and_then(|h| self.bodies.get(h))
            .map(|b| b.translation().x.hypot(b.translation().z))
            .unwrap_or(f32::NAN)
    }
}

/// Shared grid layout: static floor tiles plus dynamic box positions.
#[cfg(any(feature = "rapier", feature = "box3d"))]
type GridLayout = (Vec<([f32; 3], [f32; 3])>, Vec<[f32; 3]>);

/// Shared grid layout as (floor tiles, dynamic boxes): floors are
/// `(pos, half)` statics, boxes are positions.
#[cfg(any(feature = "rapier", feature = "box3d"))]
fn grid_layout(n: u32) -> GridLayout {
    let side = (n as f32).sqrt().ceil() as u32;
    let span = side as f32 * PITCH_XZ;
    let tile_half = 5.0f32;
    let tiles = (span / (2.0 * tile_half)).ceil() as i32;
    let mut floors = Vec::new();
    for tx in 0..tiles {
        for tz in 0..tiles {
            let x = (tx as f32 - tiles as f32 / 2.0 + 0.5) * 2.0 * tile_half;
            let z = (tz as f32 - tiles as f32 / 2.0 + 0.5) * 2.0 * tile_half;
            floors.push(([x, -0.5, z], [tile_half, 0.5, tile_half]));
        }
    }
    let mut boxes = Vec::with_capacity(n as usize);
    for i in 0..n {
        let gx = i % side;
        let gz = i / side;
        boxes.push([
            (gx as f32 - side as f32 / 2.0) * PITCH_XZ,
            REST_Y,
            (gz as f32 - side as f32 / 2.0) * PITCH_XZ,
        ]);
    }
    (floors, boxes)
}

#[cfg(feature = "rapier")]
fn native_rapier_grid(n: u32, iterations: usize) -> NativeRapier {
    let mut world = NativeRapier::fresh(iterations);
    let (floors, boxes) = grid_layout(n);
    for (pos, half) in floors {
        world.add_box(pos, half, 0.0);
    }
    for pos in boxes {
        world.add_box(pos, [BOX_HALF, BOX_HALF, BOX_HALF], BOX_MASS);
    }
    world
}

#[cfg(feature = "rapier")]
fn native_rapier_stack(levels: u32, iterations: usize) -> NativeRapier {
    let mut world = NativeRapier::fresh(iterations);
    world.add_box([0.0, -0.5, 0.0], [10.0, 0.5, 10.0], 0.0);
    for level in 0..levels {
        let handle = world.add_box(
            [0.0, REST_Y + level as f32 * STACK_PITCH_Y, 0.0],
            [BOX_HALF, BOX_HALF, BOX_HALF],
            BOX_MASS,
        );
        world.top = Some(handle);
    }
    world
}

#[cfg(feature = "rapier")]
fn native_rapier_islands(g: u32, iterations: usize) -> NativeRapier {
    let mut world = NativeRapier::fresh(iterations);
    world.add_box([0.0, -0.5, 0.0], [100.0, 0.5, 100.0], 0.0);
    for gx in 0..g {
        for gz in 0..g {
            let x = (gx as f32 - g as f32 / 2.0) * PITCH_XZ;
            let z = (gz as f32 - g as f32 / 2.0) * PITCH_XZ;
            for level in 0..4 {
                world.add_box(
                    [x, REST_Y + level as f32 * STACK_PITCH_Y, z],
                    [BOX_HALF, BOX_HALF, BOX_HALF],
                    BOX_MASS,
                );
            }
        }
    }
    world
}

// ---------------------------------------------------------------------------
// Box3D (`boxddd`): one world builder, sub-step count parametrized
// (4 = upstream sample default).
// ---------------------------------------------------------------------------

/// Box3D world with the shared scene recipe; `substeps` maps to
/// `World::step(dt, sub_step_count)`.
#[cfg(feature = "box3d")]
struct Box3dWorld {
    // Foundation is process-global (`initialize_default` is idempotent for
    // the same config); the world owns bodies, so no lifetime ties remain.
    world: boxddd::World,
    substeps: i32,
    top: Option<boxddd::BodyId>,
}

#[cfg(feature = "box3d")]
impl Box3dWorld {
    fn add_box(
        world: &mut boxddd::World,
        foundation: &boxddd::Foundation,
        pos: [f32; 3],
        half: [f32; 3],
        dynamic: bool,
    ) -> boxddd::BodyId {
        let body = must(
            world.create_body(must(
                if dynamic {
                    foundation
                        .body_def_builder()
                        .body_type(boxddd::BodyType::Dynamic)
                        .position(pos)
                        .build()
                } else {
                    foundation
                        .body_def_builder()
                        .body_type(boxddd::BodyType::Static)
                        .position(pos)
                        .build()
                },
                "box3d body def",
            )),
            "box3d create_body",
        );
        let shape_def = must(
            foundation
                .shape_def_builder()
                .density(if dynamic { BOX3D_DENSITY } else { 1.0 })
                .friction(0.5)
                .restitution(0.0)
                .build(),
            "box3d shape def",
        );
        let hull = must(
            boxddd::shapes::BoxHull::new(half[0], half[1], half[2]),
            "box3d hull",
        );
        must(
            world.create_hull_shape(body, &shape_def, &hull),
            "box3d attach hull",
        );
        body
    }

    fn fresh(substeps: i32) -> (Self, &'static boxddd::Foundation) {
        let foundation = must(boxddd::Foundation::initialize_default(), "box3d foundation");
        let world = must(
            foundation.create_world(must(
                foundation
                    .world_def_builder()
                    .gravity([0.0, GRAVITY_Y, 0.0])
                    .build(),
                "box3d world def",
            )),
            "box3d create_world",
        );
        (
            Self {
                world,
                substeps,
                top: None,
            },
            foundation,
        )
    }

    fn grid(n: u32, substeps: i32) -> Self {
        let (mut world, foundation) = Self::fresh(substeps);
        let (floors, boxes) = grid_layout(n);
        for (pos, half) in floors {
            Self::add_box(&mut world.world, foundation, pos, half, false);
        }
        for pos in boxes {
            Self::add_box(
                &mut world.world,
                foundation,
                pos,
                [BOX_HALF, BOX_HALF, BOX_HALF],
                true,
            );
        }
        world
    }

    fn stack(levels: u32, substeps: i32) -> Self {
        let (mut world, foundation) = Self::fresh(substeps);
        Self::add_box(
            &mut world.world,
            foundation,
            [0.0, -0.5, 0.0],
            [10.0, 0.5, 10.0],
            false,
        );
        for level in 0..levels {
            let body = Self::add_box(
                &mut world.world,
                foundation,
                [0.0, REST_Y + level as f32 * STACK_PITCH_Y, 0.0],
                [BOX_HALF, BOX_HALF, BOX_HALF],
                true,
            );
            world.top = Some(body);
        }
        world
    }

    fn islands(g: u32, substeps: i32) -> Self {
        let (mut world, foundation) = Self::fresh(substeps);
        Self::add_box(
            &mut world.world,
            foundation,
            [0.0, -0.5, 0.0],
            [100.0, 0.5, 100.0],
            false,
        );
        for gx in 0..g {
            for gz in 0..g {
                let x = (gx as f32 - g as f32 / 2.0) * PITCH_XZ;
                let z = (gz as f32 - g as f32 / 2.0) * PITCH_XZ;
                for level in 0..4 {
                    Self::add_box(
                        &mut world.world,
                        foundation,
                        [x, REST_Y + level as f32 * STACK_PITCH_Y, z],
                        [BOX_HALF, BOX_HALF, BOX_HALF],
                        true,
                    );
                }
            }
        }
        world
    }

    fn step(&mut self) {
        must(self.world.step(DT, self.substeps), "box3d step");
    }

    fn top_drift(&self, rest_y: f32) -> (f32, f32) {
        let Some(top) = self.top else {
            return (f32::NAN, f32::NAN);
        };
        let pos = must(self.world.body_position(top), "box3d body_position");
        (pos.y - rest_y, pos.x.hypot(pos.z))
    }
}

// ---------------------------------------------------------------------------
// Benchmark registration.
// ---------------------------------------------------------------------------

/// Stack height shared by every engine (solver-stability scene).
const STACK_LEVELS: u32 = 28;
/// Island field size shared by every engine (8x8 x 4 = 256 bodies).
const ISLAND_GRID: u32 = 8;

fn bench_grids(c: &mut Criterion) {
    let mut group = c.benchmark_group("shootout/grid");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for n in [1_000u32, 10_000] {
        group.bench_function(BenchmarkId::new("ornis_default", n), |b| {
            let mut physics = ornis_world(12);
            build_grid(&mut physics, n);
            for _ in 0..WARMUP_STEPS {
                physics.step(DT);
            }
            eprintln!(
                "SHOOTOUT ornis_default grid_{n}: {}",
                ornis_breakdown(&physics)
            );
            b.iter(|| std::hint::black_box(&mut physics).step(std::hint::black_box(DT)));
        });
        group.bench_function(BenchmarkId::new("ornis_tuned_sub4", n), |b| {
            let mut physics = ornis_world(4);
            build_grid(&mut physics, n);
            for _ in 0..WARMUP_STEPS {
                physics.step(DT);
            }
            eprintln!(
                "SHOOTOUT ornis_tuned_sub4 grid_{n}: {}",
                ornis_breakdown(&physics)
            );
            b.iter(|| std::hint::black_box(&mut physics).step(std::hint::black_box(DT)));
        });

        #[cfg(feature = "rapier")]
        group.bench_function(BenchmarkId::new("rapier_default", n), |b| {
            let mut physics = native_rapier_grid(n, 4);
            for _ in 0..WARMUP_STEPS {
                physics.step();
            }
            eprintln!("SHOOTOUT rapier_default grid_{n}: native pipeline, iterations=4");
            b.iter(|| std::hint::black_box(&mut physics).step());
        });
        #[cfg(feature = "rapier")]
        group.bench_function(BenchmarkId::new("rapier_tuned_iter8", n), |b| {
            let mut physics = native_rapier_grid(n, 8);
            for _ in 0..WARMUP_STEPS {
                physics.step();
            }
            eprintln!("SHOOTOUT rapier_tuned_iter8 grid_{n}: native pipeline, iterations=8");
            b.iter(|| std::hint::black_box(&mut physics).step());
        });
        #[cfg(feature = "box3d")]
        group.bench_function(BenchmarkId::new("box3d_default_sub4", n), |b| {
            let mut physics = Box3dWorld::grid(n, 4);
            for _ in 0..WARMUP_STEPS {
                physics.step();
            }
            b.iter(|| std::hint::black_box(&mut physics).step());
        });
        #[cfg(feature = "box3d")]
        group.bench_function(BenchmarkId::new("box3d_tuned_sub2", n), |b| {
            let mut physics = Box3dWorld::grid(n, 2);
            for _ in 0..WARMUP_STEPS {
                physics.step();
            }
            b.iter(|| std::hint::black_box(&mut physics).step());
        });
    }
    group.finish();
}

fn bench_stacks(c: &mut Criterion) {
    let mut group = c.benchmark_group("shootout/stack");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    let rest_top = REST_Y + (STACK_LEVELS - 1) as f32 * STACK_PITCH_Y;

    group.bench_function("ornis_default", |b| {
        let mut physics = ornis_world(12);
        let top = build_stack(&mut physics, STACK_LEVELS);
        for _ in 0..DRIFT_STEPS {
            physics.step(DT);
        }
        let body = physics.get_body(top).map(|body| (body.position, body.velocity));
        eprintln!("SHOOTOUT drift engine=ornis cfg=default scene=stack28 rest_top={rest_top:.4} top={body:?} {}", ornis_breakdown(&physics));
        b.iter(|| std::hint::black_box(&mut physics).step(std::hint::black_box(DT)));
    });
    group.bench_function("ornis_tuned_sub4", |b| {
        let mut physics = ornis_world(4);
        let top = build_stack(&mut physics, STACK_LEVELS);
        for _ in 0..DRIFT_STEPS {
            physics.step(DT);
        }
        let body = physics.get_body(top).map(|body| (body.position, body.velocity));
        eprintln!("SHOOTOUT drift engine=ornis cfg=tuned_sub4 scene=stack28 rest_top={rest_top:.4} top={body:?} {}", ornis_breakdown(&physics));
        b.iter(|| std::hint::black_box(&mut physics).step(std::hint::black_box(DT)));
    });

    #[cfg(feature = "rapier")]
    group.bench_function("rapier_default", |b| {
        let mut physics = native_rapier_stack(STACK_LEVELS, 4);
        for _ in 0..DRIFT_STEPS {
            physics.step();
        }
        eprintln!("SHOOTOUT drift engine=rapier cfg=default scene=stack28 rest_top={rest_top:.4} top_y={:.4} top_dxz={:.4}",
            physics.top_y(), physics.top_xz());
        b.iter(|| std::hint::black_box(&mut physics).step());
    });
    #[cfg(feature = "rapier")]
    group.bench_function("rapier_tuned_iter8", |b| {
        let mut physics = native_rapier_stack(STACK_LEVELS, 8);
        for _ in 0..DRIFT_STEPS {
            physics.step();
        }
        eprintln!("SHOOTOUT drift engine=rapier cfg=tuned_iter8 scene=stack28 rest_top={rest_top:.4} top_y={:.4} top_dxz={:.4}",
            physics.top_y(), physics.top_xz());
        b.iter(|| std::hint::black_box(&mut physics).step());
    });

    #[cfg(feature = "box3d")]
    group.bench_function("box3d_default_sub4", |b| {
        let mut physics = Box3dWorld::stack(STACK_LEVELS, 4);
        for _ in 0..DRIFT_STEPS {
            physics.step();
        }
        let (dy, dxz) = physics.top_drift(rest_top);
        eprintln!("SHOOTOUT drift engine=box3d cfg=default_sub4 scene=stack28 rest_top={rest_top:.4} top_dy={dy:.4} top_dxz={dxz:.4}");
        b.iter(|| std::hint::black_box(&mut physics).step());
    });
    #[cfg(feature = "box3d")]
    group.bench_function("box3d_tuned_sub2", |b| {
        let mut physics = Box3dWorld::stack(STACK_LEVELS, 2);
        for _ in 0..DRIFT_STEPS {
            physics.step();
        }
        let (dy, dxz) = physics.top_drift(rest_top);
        eprintln!("SHOOTOUT drift engine=box3d cfg=tuned_sub2 scene=stack28 rest_top={rest_top:.4} top_dy={dy:.4} top_dxz={dxz:.4}");
        b.iter(|| std::hint::black_box(&mut physics).step());
    });

    group.finish();
}

fn bench_islands(c: &mut Criterion) {
    let mut group = c.benchmark_group("shootout/islands");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    group.bench_function("ornis_default", |b| {
        let mut physics = ornis_world(12);
        build_islands(&mut physics, ISLAND_GRID);
        for _ in 0..WARMUP_STEPS {
            physics.step(DT);
        }
        eprintln!(
            "SHOOTOUT ornis_default islands_8x8: {}",
            ornis_breakdown(&physics)
        );
        b.iter(|| std::hint::black_box(&mut physics).step(std::hint::black_box(DT)));
    });
    group.bench_function("ornis_tuned_sub4", |b| {
        let mut physics = ornis_world(4);
        build_islands(&mut physics, ISLAND_GRID);
        for _ in 0..WARMUP_STEPS {
            physics.step(DT);
        }
        eprintln!(
            "SHOOTOUT ornis_tuned_sub4 islands_8x8: {}",
            ornis_breakdown(&physics)
        );
        b.iter(|| std::hint::black_box(&mut physics).step(std::hint::black_box(DT)));
    });

    #[cfg(feature = "rapier")]
    group.bench_function("rapier_default", |b| {
        let mut physics = native_rapier_islands(ISLAND_GRID, 4);
        for _ in 0..WARMUP_STEPS {
            physics.step();
        }
        eprintln!("SHOOTOUT rapier_default islands_8x8: native pipeline, iterations=4");
        b.iter(|| std::hint::black_box(&mut physics).step());
    });
    #[cfg(feature = "rapier")]
    group.bench_function("rapier_tuned_iter8", |b| {
        let mut physics = native_rapier_islands(ISLAND_GRID, 8);
        for _ in 0..WARMUP_STEPS {
            physics.step();
        }
        eprintln!("SHOOTOUT rapier_tuned_iter8 islands_8x8: native pipeline, iterations=8");
        b.iter(|| std::hint::black_box(&mut physics).step());
    });

    #[cfg(feature = "box3d")]
    group.bench_function("box3d_default_sub4", |b| {
        let mut physics = Box3dWorld::islands(ISLAND_GRID, 4);
        for _ in 0..WARMUP_STEPS {
            physics.step();
        }
        b.iter(|| std::hint::black_box(&mut physics).step());
    });
    #[cfg(feature = "box3d")]
    group.bench_function("box3d_tuned_sub2", |b| {
        let mut physics = Box3dWorld::islands(ISLAND_GRID, 2);
        for _ in 0..WARMUP_STEPS {
            physics.step();
        }
        b.iter(|| std::hint::black_box(&mut physics).step());
    });

    group.finish();
}

criterion_group!(benches, bench_grids, bench_stacks, bench_islands);
criterion_main!(benches);
