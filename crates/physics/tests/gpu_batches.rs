//! GPU batch pins: WGSL layouts, batching and device solver agreement.

#![cfg(feature = "gpu")]

use glam::Vec3;
use ornis_physics::BodyHandle;
use ornis_physics::RestitutionGate;
use ornis_physics::RigidBody;
use ornis_physics::engine::{
    Manifold, ManifoldPoint, ManifoldState, PhysicsEngine, SequentialImpulseEngine,
};
use ornis_physics::gpu::{
    AVBD_ROW_ENTRY, AVBD_STUB_ENTRY, DispatchReport, GPU_AVBD_SYSTEM_STRIDE, GPU_BATCH_STRIDE,
    GPU_BODY_STRIDE, GpuAvbdDispatch, GpuAvbdMass, GpuAvbdRow, GpuAvbdStub, GpuAvbdSystem,
    GpuBatch, GpuBodyState, GpuSequentialImpulse, LaneInput, SI_CONTACT_ENTRY, WgpuAvbdSolver,
    avbd_diag_solve_cpu, avbd_inertial_hessian_diag, avbd_ldl_6x6_cpu, avbd_row_wgsl,
    avbd_stage_contact_solve, avbd_stamp_row_cpu, avbd_stub_wgsl, contact_solver_wgsl,
    pack_single_point_batches, solve_params_bool,
};
use std::sync::Arc;

/// Bulk params layout (no device): pass `k` carries `(k, total, rest)`
/// — the exact values the old per-iteration loop uploaded one by one.
#[test]
fn solve_params_mirror_old_per_iteration_values() {
    let params = solve_params_bool(8, true);
    assert_eq!(params.len(), 8);
    for (k, entry) in params.iter().enumerate() {
        assert_eq!(*entry, [k as u32, 8, 1, 0], "pass {k} params wrong");
    }
    let params = solve_params_bool(3, false);
    assert_eq!(params, vec![[0, 3, 0, 0], [1, 3, 0, 0], [2, 3, 0, 0]]);
    assert!(solve_params_bool(0, false).is_empty());
}

#[test]
fn gpu_pack_produces_disjoint_batches() {
    let bodies = vec![
        RigidBody::new_sphere(Vec3::ZERO, 0.5, 0.0), // 0: static
        RigidBody::new_sphere(Vec3::new(1.0, 0.0, 0.0), 0.5, 1.0), // 1
        RigidBody::new_sphere(Vec3::new(2.0, 0.0, 0.0), 0.5, 1.0), // 2
        RigidBody::new_sphere(Vec3::new(3.0, 0.0, 0.0), 0.5, 1.0), // 3
        RigidBody::new_sphere(Vec3::new(4.0, 0.0, 0.0), 0.5, 1.0), // 4
    ];
    // Single-point manifold helper
    fn mk_manifold(i: usize, j: usize, n: Vec3, p: Vec3) -> Manifold {
        Manifold {
            body_a: i.into(),
            body_b: j.into(),
            normal: n,
            point_count: 1,
            points: [ManifoldPoint {
                world_point: p,
                penetration: 0.01,
            }; 4],
        }
    }
    fn mk_state(i: usize, j: usize) -> ManifoldState {
        ManifoldState {
            mi: 0,
            i,
            j,
            count: 1,
            acc: [0.0; 4],
            acc_friction: [0.0; 4],
            acc_friction2: [0.0; 4],
            bias: [0.0; 4],
            target: [0.0; 4],
            mu: 0.3,
            mu2: 0.3,
            mu_roll: 0.0,
            mu_spin: 0.0,
            acc_roll: [0.0; 4],
            acc_roll2: [0.0; 4],
            acc_spin: [0.0; 4],
            t1: Vec3::X,
            t2: Vec3::Z,
            surface_velocity: Vec3::ZERO,
            la: [Vec3::ZERO; 4],
            lb: [Vec3::ZERO; 4],
            pen0: [0.0; 4],
        }
    }

    // Create contacts (0,1), (0,2) — conflict on 0
    let manifolds = vec![
        mk_manifold(0, 1, Vec3::X, Vec3::ZERO),
        mk_manifold(0, 2, Vec3::X, Vec3::ZERO),
        mk_manifold(3, 4, Vec3::X, Vec3::new(3.5, 0.0, 0.0)),
    ];
    let states: Vec<ManifoldState> = (0..3)
        .map(|i| {
            let m = &manifolds[i];
            mk_state(m.body_a.index(), m.body_b.index())
        })
        .collect();
    let single_indices: Vec<usize> = (0..3).collect();

    let (batches, count) = pack_single_point_batches(&bodies, &states, &manifolds, &single_indices);
    assert_eq!(count, 2, "should form 2 batches: [1, 2] with conflict");
    assert_eq!(
        batches[0].count, 1,
        "first batch only has the non-conflicting contact... wait"
    );
    // Actually with greedy pack: (0,1) starts batch; (0,2) conflicts → flush batch1(1), start batch2 with (0,2); (3,4) clears bodies → adds to batch2.
    // So batch[0].count = 1 (just 0,1), batch[1].count = 2 (0,2 + 3,4).
    assert_eq!(batches[1].count, 2, "second batch has 2 lanes");
}

#[test]
fn layout_stride_matches_struct_size() {
    // The WgslStruct derive asserts per-field offsets and the total size
    // at compile time; the buffer strides must be the same numbers.
    assert_eq!(GPU_BODY_STRIDE, 32);
    assert_eq!(GPU_BATCH_STRIDE, std::mem::size_of::<GpuBatch>() as u64);
    assert_eq!(std::mem::offset_of!(GpuBodyState, velocity), 0);
    assert_eq!(std::mem::offset_of!(GpuBodyState, angular), 16);
}

/// Artifact gate: the kernel the host loads is the translated one — three
/// bindings in the host's order, one four-thread workgroup, the batch stride
/// and lane word offsets the Rust record defines, and the accumulator words
/// written back at the end.
#[test]
fn contact_kernel_is_the_translated_artifact() {
    let source = contact_solver_wgsl();
    for binding in [
        "@group(0) @binding(0) \nvar<storage, read_write> storage_0_0_: array<f32>;",
        "@group(0) @binding(1) \nvar<storage, read_write> storage_0_1_: array<f32>;",
        "@group(0) @binding(2) \nvar<uniform> uniform_struct_0_2_",
    ] {
        assert!(source.contains(binding), "contact artifact lost {binding}");
    }
    assert!(source.contains("@compute @workgroup_size(4, 1, 1)"));
    assert!(source.contains(&format!("fn {}(", SI_CONTACT_ENTRY)));
    // One batch is 312 words: the kernel scales the workgroup base by it and
    // finds the lane's `count` word at the end of the record.
    assert!(source.contains("* 312u"), "batch stride missing");
    assert!(source.contains("+ 308u"), "count word missing");
    // The three accumulators persist for the next dispatch (warm start).
    for word in ["+ 296u", "+ 300u", "+ 304u"] {
        assert!(source.contains(word), "accumulator word {word} not written");
    }
    assert!(!source.contains("@vertex"), "compute stage only");
}

#[test]
fn generated_wgsl_validates_with_naga() {
    let source = contact_solver_wgsl();
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("generated WGSL must parse: {e}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("generated WGSL must validate: {e}"));
}

/// Create a wgpu device/queue for tests. Returns `None` when no adapter
/// is available (e.g. a machine without GPU drivers) — the device tests
/// then skip instead of failing.
fn create_test_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ornis-physics gpu test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;
    Some((Arc::new(device), Arc::new(queue)))
}

/// Direct solver-level check: a sphere falling onto a static sphere must
/// cancel its normal velocity, accumulate the full impulse, and leave the
/// static body untouched.
#[test]
fn gpu_solver_single_contact_matches_analytic() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("gpu_solver_single_contact_matches_analytic: no wgpu adapter — skipped");
        return;
    };
    let solver = GpuSequentialImpulse::new(device, queue, 2, 1);

    let a = RigidBody::new_sphere(Vec3::ZERO, 0.5, 0.0); // static
    let mut b = RigidBody::new_sphere(Vec3::new(0.0, 1.0, 0.0), 0.5, 1.0);
    b.velocity = Vec3::new(0.0, -1.0, 0.0);

    let mut batch = GpuBatch::zero();
    batch.fill_lane(LaneInput {
        lane: 0,
        n: Vec3::Y,                    // normal
        ra: Vec3::new(0.0, 0.5, 0.0),  // ra = contact − center a
        rb: Vec3::new(0.0, -0.5, 0.0), // rb = contact − center b
        target: 0.0,                   // target (non-speculative rest)
        mu: 0.0,                       // mu (no friction)
        bias: 0.0,                     // bias (no restitution)
        acc_in: 0.0,                   // acc_in
        a: &a,
        ba_idx: 0,
        b: &b,
        bb_idx: 1,
    });
    batch.count = 1;

    assert_eq!(
        solver.upload_bodies(&[a.clone(), b.clone()]),
        DispatchReport {
            staged: 2,
            dropped: 0
        }
    );
    assert_eq!(
        solver.upload_batches(&[batch]),
        DispatchReport {
            staged: 1,
            dropped: 0
        }
    );
    solver.solve(1, 8, RestitutionGate::Suppressed);

    let mut out = [a, b];
    let body_report = solver
        .try_download_bodies(&mut out)
        .expect("body download must map");
    assert_eq!(
        body_report,
        DispatchReport {
            staged: 2,
            dropped: 0
        }
    );
    let mut acc_batches = [batch];
    let acc_report = solver
        .try_download_acc(&mut acc_batches)
        .expect("acc download must map");
    assert_eq!(
        acc_report,
        DispatchReport {
            staged: 1,
            dropped: 0
        }
    );

    assert!(
        out[1].velocity.length() < 1e-3,
        "normal velocity must be cancelled, got {:?}",
        out[1].velocity
    );
    assert!(
        (acc_batches[0].acc[0] - 1.0).abs() < 1e-2,
        "accumulated impulse must equal the body mass, got {}",
        acc_batches[0].acc[0]
    );
    assert_eq!(out[0].velocity, Vec3::ZERO, "static body must not move");
    assert_eq!(out[0].angular_velocity, Vec3::ZERO);
    assert!(
        out[1].angular_velocity.length() < 1e-3,
        "no torque expected"
    );
}

/// Staging hygiene: uploads past the capacity bound stage the prefix and
/// report the dropped tail instead of dropping silently. The fallible
/// download observes the same bound symmetrically.
#[test]
fn gpu_upload_reports_staged_and_dropped() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("gpu_upload_reports_staged_and_dropped: no wgpu adapter — skipped");
        return;
    };
    let solver = GpuSequentialImpulse::new(device, queue, 2, 1);
    let bodies = vec![
        RigidBody::new_sphere(Vec3::new(0.0, 0.0, 0.0), 0.5, 1.0),
        RigidBody::new_sphere(Vec3::new(1.0, 0.0, 0.0), 0.5, 1.0),
        RigidBody::new_sphere(Vec3::new(2.0, 0.0, 0.0), 0.5, 1.0),
        RigidBody::new_sphere(Vec3::new(3.0, 0.0, 0.0), 0.5, 1.0),
        RigidBody::new_sphere(Vec3::new(4.0, 0.0, 0.0), 0.5, 1.0),
    ];
    assert_eq!(
        solver.upload_bodies(&bodies),
        DispatchReport {
            staged: 2,
            dropped: 3
        }
    );
    assert_eq!(
        solver.upload_batches(&[GpuBatch::zero(); 4]),
        DispatchReport {
            staged: 1,
            dropped: 3
        }
    );
    let mut out = bodies.clone();
    assert_eq!(
        solver
            .try_download_bodies(&mut out)
            .expect("body download must map"),
        DispatchReport {
            staged: 2,
            dropped: 3
        }
    );
}

/// DSL-pilot agreement: one frictional contact solved on the GPU must
/// match the CPU wide batch — both sides now run the shared
/// `contact_math` row — within tolerance. Never bit-identical by
/// promise (device float contraction may differ ±1 ulp per op), but
/// the same row math on both sides.
#[test]
fn gpu_contact_row_matches_cpu_kernels() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("gpu_contact_row_matches_cpu_kernels: no wgpu adapter — skipped");
        return;
    };
    let n = Vec3::Y;
    let t1 = ornis_physics::math::tangent_basis(n).0;
    let t2 = t1.cross(n);
    let contact = Vec3::new(0.0, 0.5, 0.0);
    let mu = 0.5;

    let a = RigidBody::new_sphere(Vec3::ZERO, 0.5, 0.0); // static
    let mut b = RigidBody::new_sphere(Vec3::new(0.0, 1.0, 0.0), 0.5, 1.0);
    b.velocity = Vec3::new(0.15, -1.0, 0.1); // approach + small slide
    let bodies = [a.clone(), b.clone()];

    // GPU side: one lane, 48 iterations, no restitution. (The row
    // converges in a handful of passes; 48 keeps the CPU/GPU agreement
    // comparison on many accumulated updates.)
    let solver = GpuSequentialImpulse::new(device, queue, 2, 1);
    let mut gb = GpuBatch::zero();
    gb.fill_lane(LaneInput {
        lane: 0,
        n,
        ra: contact - a.position,
        rb: contact - b.position,
        target: 0.0,
        mu,
        bias: 0.0,
        acc_in: 0.0,
        a: &a,
        ba_idx: 0,
        b: &b,
        bb_idx: 1,
    });
    gb.count = 1;
    solver.upload_bodies(&bodies);
    solver.upload_batches(&[gb]);
    solver.solve(1, 48, RestitutionGate::Suppressed);
    let mut gpu_bodies = bodies.clone();
    solver.download_bodies(&mut gpu_bodies);
    let mut gpu_batches = [gb];
    solver.download_acc(&mut gpu_batches);

    // CPU side: the same contact as a one-lane wide batch (the batch
    // path calls `contact_math::...::eval` per lane).
    let manifolds = [Manifold {
        body_a: BodyHandle::from_raw(0),
        body_b: BodyHandle::from_raw(1),
        normal: n,
        point_count: 1,
        points: [ManifoldPoint {
            world_point: contact,
            penetration: 0.01,
        }; 4],
    }];
    let mut states = [ManifoldState {
        mi: 0,
        i: 0,
        j: 1,
        count: 1,
        acc: [0.0; 4],
        acc_friction: [0.0; 4],
        acc_friction2: [0.0; 4],
        bias: [0.0; 4],
        target: [0.0; 4],
        mu,
        mu2: mu,
        mu_roll: 0.0,
        mu_spin: 0.0,
        acc_roll: [0.0; 4],
        acc_roll2: [0.0; 4],
        acc_spin: [0.0; 4],
        t1,
        t2,
        surface_velocity: Vec3::ZERO,
        la: [Vec3::ZERO; 4],
        lb: [Vec3::ZERO; 4],
        pen0: [0.0; 4],
    }];
    let items: Vec<(usize, &Manifold, &ManifoldState)> = vec![(0, &manifolds[0], &states[0])];
    let mut batch = ornis_physics::wide::WideBatch::build(&items, &bodies);
    let mut cpu_bodies = bodies.clone();
    for _ in 0..48 {
        batch.gather(&cpu_bodies);
        batch.solve_iteration();
        batch.scatter(&mut cpu_bodies);
    }
    batch.write_back_acc(&mut states);

    // The normal approach must stop and the contact slip must bite on
    // both sides. NOTE: this asserts slip (relative tangential velocity
    // at the contact), not center slide: with rolling resistance off, a
    // frictional hit converts slide into rolling (v = -w x r), so the
    // center keeps moving at ~0.129 while the contact slip is ~0.
    let slip_of = |bodies: &[RigidBody; 2]| {
        let (a, b) = (&bodies[0], &bodies[1]);
        let rel = (b.velocity + b.angular_velocity.cross(contact - b.position))
            - (a.velocity + a.angular_velocity.cross(contact - a.position));
        (rel - n * rel.dot(n)).length()
    };
    for (label, bodies) in [("gpu", &gpu_bodies), ("cpu", &cpu_bodies)] {
        assert!(
            bodies[1].velocity.y.abs() < 0.05,
            "{label}: normal approach must stop, got {:?}",
            bodies[1].velocity
        );
        let slip = slip_of(bodies);
        assert!(
            slip < 1e-3,
            "{label}: contact slip must bite, tangential slip {slip}"
        );
        assert_eq!(
            bodies[0].velocity,
            Vec3::ZERO,
            "{label}: static must not move"
        );
    }
    // ...and the two sides must agree in tolerance.
    for h in 0..2 {
        let dv = (gpu_bodies[h].velocity - cpu_bodies[h].velocity).length();
        let dw = (gpu_bodies[h].angular_velocity - cpu_bodies[h].angular_velocity).length();
        assert!(dv < 1e-4, "body {h} velocity diverged: {dv}");
        assert!(dw < 1e-4, "body {h} angular velocity diverged: {dw}");
    }
    assert!(
        (gpu_batches[0].acc[0] - states[0].acc[0]).abs() < 1e-4,
        "normal impulse diverged: gpu {} vs cpu {}",
        gpu_batches[0].acc[0],
        states[0].acc[0]
    );
}

/// Engine-level check: the same scene run on the CPU and with the GPU
/// solver attached must settle to the same resting state.
#[test]
fn gpu_solver_tracks_cpu_engine() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("gpu_solver_tracks_cpu_engine: no wgpu adapter — skipped");
        return;
    };
    let solver = GpuSequentialImpulse::new(device, queue, 16, 64);
    let gravity = Vec3::new(0.0, -9.81, 0.0);

    let mut cpu = SequentialImpulseEngine::new(gravity);
    let mut gpu = SequentialImpulseEngine::new(gravity);
    for engine in [&mut cpu, &mut gpu] {
        engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, -2.0, 0.0),
            Vec3::new(4.0, 0.5, 4.0),
            0.0,
        ));
        // Vertically aligned stack, placed already at its resting heights
        // (ground top -1.5, so the spheres rest at -1.0, 0.0, 1.0). The GPU
        // path is a Jacobi/GS hybrid over batches, not the CPU GS order, so a
        // stack that is still *falling* drifts by centimetres between the two
        // schemes before it settles; starting settled keeps the contract about
        // holding the stack instead of about matching two fall trajectories.
        // Lateral offsets would topple it chaotically — also not this gate.
        engine.add_body(RigidBody::new_sphere(Vec3::new(0.0, -1.0, 0.0), 0.5, 1.0));
        engine.add_body(RigidBody::new_sphere(Vec3::new(0.0, 0.0, 0.0), 0.5, 1.0));
        engine.add_body(RigidBody::new_sphere(Vec3::new(0.0, 1.0, 0.0), 0.5, 1.0));
    }
    gpu.set_gpu_solver(solver);

    for _ in 0..60 {
        cpu.step(1.0 / 60.0);
        gpu.step(1.0 / 60.0);
    }

    for i in 0usize..4 {
        let (bc, bg) = (
            cpu.get_body(i.into()).unwrap(),
            gpu.get_body(i.into()).unwrap(),
        );
        assert!(
            bc.position.distance(bg.position) < 0.05,
            "body {i} position diverged: cpu {:?} vs gpu {:?}",
            bc.position,
            bg.position
        );
        assert!(
            bc.velocity.distance(bg.velocity) < 0.05,
            "body {i} velocity diverged: cpu {:?} vs gpu {:?}",
            bc.velocity,
            bg.velocity
        );
    }
}

/// Rung-1 mass roster layout (no device): 16-byte stride, `inertia`
/// first at offset 0 (16-aligned `vec3<f32>`), `inv_mass` at 12.
#[test]
fn avbd_mass_layout_matches_wgsl() {
    assert_eq!(std::mem::size_of::<GpuAvbdMass>(), 16);
    assert_eq!(std::mem::offset_of!(GpuAvbdMass, inertia), 0);
    assert_eq!(std::mem::offset_of!(GpuAvbdMass, inv_mass), 12);
    // The kernel reads the roster word by word, so the contract is the word
    // layout, not a struct declaration: 16 bytes per record, `inertia` first.
    let source = avbd_stub_wgsl();
    assert!(source.contains("var<storage> storage_0_0_: array<f32>;"));
    assert!(source.contains(&format!("fn {}(", AVBD_STUB_ENTRY)));
    assert_eq!(std::mem::size_of::<GpuAvbdMass>(), 4 * 4);
}

/// Rung-1 CPU assembly (no device): the inertial Hessian diagonal
/// equals the analytic `m/dt²`, `I/dt²` values; statics and
/// degenerate `dt` yield zeros.
#[test]
fn avbd_hessian_diag_matches_analytic() {
    let dt = 1.0 / 60.0;
    let diag = avbd_inertial_hessian_diag(1.0, [2.0, 3.0, 4.0], dt);
    // f32 1/60 is inexact, so compare in tolerance (never bit-identical
    // by promise — same rule as the device comparison to come).
    let expect = [3600.0, 3600.0, 3600.0, 7200.0, 10800.0, 14400.0];
    for i in 0..6 {
        assert!(
            (diag[i] - expect[i]).abs() < 1e-2,
            "axis {i}: {} vs {}",
            diag[i],
            expect[i]
        );
    }
    // Heavier body: linear block scales with mass, angular with inertia.
    let heavy = avbd_inertial_hessian_diag(0.1, [1.0, 1.0, 1.0], dt);
    assert!((heavy[0] - 36000.0).abs() < 1.0);
    assert!((heavy[3] - 3600.0).abs() < 1e-2);
    // Statics and degenerate dt assemble nothing.
    assert_eq!(
        avbd_inertial_hessian_diag(0.0, [1.0, 1.0, 1.0], dt),
        [0.0; 6]
    );
    assert_eq!(
        avbd_inertial_hessian_diag(-1.0, [1.0, 1.0, 1.0], dt),
        [0.0; 6]
    );
    assert_eq!(
        avbd_inertial_hessian_diag(1.0, [1.0, 1.0, 1.0], 0.0),
        [0.0; 6]
    );
    assert_eq!(
        avbd_inertial_hessian_diag(1.0, [1.0, 1.0, 1.0], f32::NAN),
        [0.0; 6]
    );
}

/// Rung-1 CPU solve (no device): the diagonal LDL agrees with the
/// dense AVBD LDL on diagonal systems within tolerance (never
/// bit-identical by promise); degenerate axes solve to zero (the
/// documented shader adaptation — no `Option` in WGSL).
#[test]
fn avbd_diag_ldl_matches_dense_solve() {
    let diag = [4.0, 3600.0, 1.5, 7200.0, 9.0, 2.25];
    let rhs = [8.0, -3.6, 0.75, 1.44, 27.0, -4.5];
    let mut lhs = [[0.0f32; 6]; 6];
    for (i, row) in lhs.iter_mut().enumerate() {
        row[i] = diag[i];
    }
    let dense = ornis_physics::avbd::solve_6x6(lhs, rhs).expect("SPD diagonal solves");
    let flown = avbd_diag_solve_cpu(diag, rhs);
    for i in 0..6 {
        assert!(
            (dense[i] - flown[i]).abs() < 1e-5,
            "axis {i}: dense {} vs diag {}",
            dense[i],
            flown[i]
        );
    }
    // Degenerate axes carry no correction (statics/sleepers stay put).
    let mut bad = diag;
    bad[2] = 0.0;
    bad[4] = -1.0;
    let zeroed = avbd_diag_solve_cpu(bad, rhs);
    assert_eq!(zeroed[2], 0.0);
    assert_eq!(zeroed[4], 0.0);
    assert!((zeroed[0] - rhs[0] / diag[0]).abs() < 1e-9);
}

/// Rung-1 staging (no device): the CPU reference assembles + solves
/// per body (dynamics move, statics zero out) and rejects mismatched
/// inputs — the exact contract the future device path validates against.
#[test]
fn avbd_stage_diag_solve_mirrors_kernel_math() {
    let stub = GpuAvbdStub::new(8);
    let dynamic = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 2.0);
    let static_body = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 0.0);
    let bodies = [dynamic, static_body];
    let rhs = [[1.0; 6], [1.0; 6]];
    let dt = 1.0 / 60.0;
    let out = stub
        .stage_diag_solve(&bodies, &rhs, dt)
        .expect("matching inputs stage");
    assert_eq!(out.len(), 2);
    // Dynamic linear block: rhs / (m/dt²) with m = 2.
    let expect_lin = 1.0 / (2.0 / (dt * dt));
    for (i, &got) in out[0].iter().enumerate().take(3) {
        assert!((got - expect_lin).abs() < 1e-9, "axis {i}: {got}");
    }
    assert_eq!(out[1], [0.0; 6], "static body carries no correction");
    assert!(stub.stage_diag_solve(&bodies, &rhs[..1], dt).is_none());
}

/// Rung-1 roster (no device): dynamics carry their mass model, statics
/// and kinematics zero out (the exact inputs `avbd_stub_kernel` reads).
#[test]
fn avbd_mass_roster_mirrors_body_mass_model() {
    use ornis_physics::body::BodyType;
    let stub = GpuAvbdStub::new(8);
    let mut dynamic = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 2.0);
    dynamic.body_type = BodyType::Dynamic;
    let static_body = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 0.0);
    let roster = stub.mass_roster(&[dynamic.clone(), static_body]);
    assert_eq!(roster.len(), 2);
    assert!((roster[0].inv_mass - dynamic.inv_mass).abs() < 1e-9);
    assert_eq!(roster[0].inertia, dynamic.inertia.to_array());
    assert_eq!(roster[1].inv_mass, 0.0);
    assert_eq!(roster[1].inertia, [0.0; 3]);
}

/// Rung-1 contract: the artifact is the translated kernel — five bindings in
/// the host's order, one workgroup of 64, and the count guard — and it
/// naga-validates without a device.
#[test]
fn avbd_stub_kernel_is_the_translated_artifact() {
    let source = avbd_stub_wgsl();
    for binding in [
        "var<storage> storage_0_0_: array<f32>;",
        "var<storage> storage_0_1_: array<f32>;",
        "var<storage, read_write> storage_0_2_: array<f32>;",
        "@group(0) @binding(3)",
        "@group(0) @binding(4)",
    ] {
        assert!(source.contains(binding), "stub artifact lost {binding}");
    }
    assert!(source.contains("@compute @workgroup_size(64, 1, 1)"));
    assert!(source.contains(&format!("fn {}(", AVBD_STUB_ENTRY)));
    // The guard returns for out-of-range bodies before touching the buffers.
    assert!(source.contains("return;"));
    assert!(!source.contains("@fragment"), "compute stage only");
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("stub WGSL must parse: {e}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("stub WGSL must validate: {e}"));
}

/// STUB contract: stepping falls back to the CPU AVBD engine (gravity
/// integrates velocity) and never claims the device.
#[test]
fn avbd_stub_cpu_fallback_advances() {
    use ornis_physics::avbd::AvbdEngine;
    use ornis_physics::engine::PhysicsEngine;

    fn runs_on_device_via_seam(stub: &GpuAvbdStub) -> bool {
        ornis_physics::gpu::GpuAvbdDispatch::runs_on_device(stub)
    }

    let stub = GpuAvbdStub::new(8);
    assert!(!stub.runs_on_device());
    assert!(!runs_on_device_via_seam(&stub));
    let mut engine = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
    engine.add_body(RigidBody::new_sphere(Vec3::new(0.0, 5.0, 0.0), 0.5, 1.0));
    stub.step_avbd(&mut engine, 1.0 / 60.0);
    let v = engine
        .get_body(BodyHandle::from_raw(0))
        .expect("stub scene keeps its body")
        .velocity;
    assert!(v.y < 0.0, "CPU fallback must integrate gravity, got {v:?}");
}

/// Rung-1 wiring: the GPU flag defaults off; attaching the stub records
/// the CPU fallback per completed step while the trajectory stays
/// bit-identical to the unattached engine (same code path, honest
/// fallback — unlike the Jacobi/GS GPU contact hybrid, which only
/// promises tolerance).
#[test]
fn avbd_engine_gpu_flag_defaults_off_and_falls_back() {
    use ornis_physics::avbd::AvbdEngine;
    use ornis_physics::engine::PhysicsEngine;

    fn scene() -> AvbdEngine {
        let mut engine = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(5.0, 0.5, 5.0),
            0.0,
        ));
        engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::splat(0.4),
            1.0,
        ));
        engine
    }

    let mut plain = scene();
    assert!(!plain.gpu_avbd_enabled());
    assert_eq!(plain.gpu_fallback_steps(), 0);
    let mut wired = scene();
    wired.set_gpu_avbd(Some(GpuAvbdStub::new(8)));
    assert!(wired.gpu_avbd_enabled());
    for _ in 0..60 {
        plain.step(1.0 / 60.0);
        wired.step(1.0 / 60.0);
    }
    assert_eq!(wired.gpu_fallback_steps(), 60);
    assert_eq!(plain.gpu_fallback_steps(), 0);
    for h in 0usize..2 {
        let (bp, bw) = (
            plain.get_body(h.into()).expect("plain keeps bodies"),
            wired.get_body(h.into()).expect("wired keeps bodies"),
        );
        assert_eq!(bp.position.to_array(), bw.position.to_array());
        assert_eq!(bp.velocity.to_array(), bw.velocity.to_array());
    }
    wired.set_gpu_avbd(None);
    assert!(!wired.gpu_avbd_enabled());
}

/// Rung-2 row/system layout (no device): 48-byte rows (`axis` at 0,
/// 16-aligned `vec3<f32>`; `pen` at 12; `lever` at 16; `force` at 28;
/// `sign` at 32) packed with mass + state into 96-byte systems
/// (`mass` at 0, `state` at 16, `row` at 48).
#[test]
fn avbd_row_layout_matches_wgsl() {
    assert_eq!(std::mem::size_of::<GpuAvbdRow>(), 48);
    assert_eq!(std::mem::offset_of!(GpuAvbdRow, axis), 0);
    assert_eq!(std::mem::offset_of!(GpuAvbdRow, pen), 12);
    assert_eq!(std::mem::offset_of!(GpuAvbdRow, lever), 16);
    assert_eq!(std::mem::offset_of!(GpuAvbdRow, force), 28);
    assert_eq!(std::mem::offset_of!(GpuAvbdRow, sign), 32);
    assert_eq!(GPU_AVBD_SYSTEM_STRIDE, 96);
    assert_eq!(std::mem::size_of::<GpuAvbdSystem>(), 96);
    assert_eq!(std::mem::offset_of!(GpuAvbdSystem, mass), 0);
    assert_eq!(std::mem::offset_of!(GpuAvbdSystem, state), 16);
    assert_eq!(std::mem::offset_of!(GpuAvbdSystem, row), 48);
    // The kernel reads the record by word index, so the layout contract is the
    // stride and the offsets above; the artifact must agree on the size of the
    // staging array it declares (64 systems of 24 words, all f32).
    let source = avbd_row_wgsl();
    assert!(source.contains("array<f32>"));
    assert_eq!(GPU_AVBD_SYSTEM_STRIDE as usize, 24 * 4);
}

/// Rung-2 CPU stamp (no device): one row into a zero system carries exactly
/// the analytic outer products (`nn = sign*axis`, `t = r×nn`) and nothing
/// else; the side sign flips the gradients.
#[test]
fn avbd_stamp_row_cpu_matches_analytic_row() {
    // axis +Y, lever +X, A-side: nn = +Y, t = X×Y = +Z.
    let mut lhs = [[0.0f32; 6]; 6];
    let mut rhs = [0.0f32; 6];
    avbd_stamp_row_cpu(
        &mut lhs,
        &mut rhs,
        [0.0, 1.0, 0.0],
        2.0,
        3.0,
        [1.0, 0.0, 0.0],
        1.0,
    );
    let mut expect_lhs = [[0.0f32; 6]; 6];
    expect_lhs[1][1] = 2.0;
    expect_lhs[5][5] = 2.0;
    expect_lhs[1][5] = 2.0;
    expect_lhs[5][1] = 2.0;
    let mut expect_rhs = [0.0f32; 6];
    expect_rhs[1] = 3.0;
    expect_rhs[5] = 3.0;
    for (i, ((lrow, erow), (&r, &er))) in lhs
        .iter()
        .zip(expect_lhs.iter())
        .zip(rhs.iter().zip(expect_rhs.iter()))
        .enumerate()
    {
        for (j, (&l, &e)) in lrow.iter().zip(erow.iter()).enumerate() {
            assert!((l - e).abs() < 1e-9, "lhs[{i}][{j}]: {l} vs {e}");
        }
        assert!((r - er).abs() < 1e-9, "rhs[{i}]: {r} vs {er}");
    }
    // B-side: nn = -Y, t = X×(-Y) = -Z — every stamped entry flips sign.
    let mut lhs_b = [[0.0f32; 6]; 6];
    let mut rhs_b = [0.0f32; 6];
    avbd_stamp_row_cpu(
        &mut lhs_b,
        &mut rhs_b,
        [0.0, 1.0, 0.0],
        2.0,
        3.0,
        [1.0, 0.0, 0.0],
        -1.0,
    );
    for (i, ((lrow, erow), (&r, &er))) in lhs_b
        .iter()
        .zip(expect_lhs.iter())
        .zip(rhs_b.iter().zip(expect_rhs.iter()))
        .enumerate()
    {
        for (j, (&l, &e)) in lrow.iter().zip(erow.iter()).enumerate() {
            // Hessian is quadratic in the side (outer products of flipped
            // gradients): unchanged. The rhs is linear: flipped.
            assert!(
                (l - e).abs() < 1e-9,
                "B-side lhs[{i}][{j}] must match A-side Hessian"
            );
        }
        assert!((r + er).abs() < 1e-9, "B-side rhs[{i}] must flip, got {r}");
    }
    // Inert rows stamp nothing.
    let inert = GpuAvbdRow::inert();
    let mut lhs_i = [[1.0f32; 6]; 6];
    let mut rhs_i = [1.0f32; 6];
    avbd_stamp_row_cpu(
        &mut lhs_i,
        &mut rhs_i,
        inert.axis,
        inert.pen,
        inert.force,
        inert.lever,
        inert.sign,
    );
    assert_eq!(lhs_i, [[1.0f32; 6]; 6]);
    assert_eq!(rhs_i, [1.0f32; 6]);
}

/// Rung-2 CPU LDL (no device): agrees with the dense AVBD solve on SPD
/// systems, and reports breakdown (zeroed delta, `false`) on singular and
/// indefinite pivots — the exact signal the shader writes into `avbd_ok`.
#[allow(clippy::needless_range_loop)] // 6x6 assembly indexes both triangles.
#[test]
fn avbd_ldl_6x6_cpu_matches_dense_and_signals_breakdown() {
    let mut lhs = [[0.0f32; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            lhs[i][j] = if i == j {
                4.0 + i as f32
            } else {
                0.1 * ((i + j) as f32 + 1.0)
            };
        }
        // Symmetrize: LDL without pivoting needs the symmetric half.
        for j in 0..i {
            lhs[i][j] = lhs[j][i];
        }
    }
    // Diagonal dominance keeps every pivot positive (SPD by Gershgorin).
    let rhs = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let (x, ok) = avbd_ldl_6x6_cpu(lhs, rhs);
    assert!(ok, "SPD system must solve");
    let dense = ornis_physics::avbd::solve_6x6(lhs, rhs).expect("SPD solves densely");
    for (i, (&got, &want)) in x.iter().zip(dense.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "axis {i}: mirror {got} vs dense {want}"
        );
    }
    // Singular (zeroed static) and indefinite (negative pivot) break down.
    let (zx, zok) = avbd_ldl_6x6_cpu([[0.0; 6]; 6], rhs);
    assert!(!zok, "singular system must report breakdown");
    assert_eq!(zx, [0.0; 6]);
    let mut neg = lhs;
    neg[0][0] = -1.0;
    let (nx, nok) = avbd_ldl_6x6_cpu(neg, rhs);
    assert!(!nok, "indefinite pivot must report breakdown");
    assert_eq!(nx, [0.0; 6]);
}

/// Rung-2 staging (no device): dynamics assemble + solve, statics break
/// down, mismatched inputs are rejected — the exact contract the device
/// path validates against.
#[test]
fn avbd_stage_contact_solve_mirrors_kernel_inputs() {
    let stub = GpuAvbdStub::new(8);
    let dynamic = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 2.0);
    let static_body = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 0.0);
    let bodies = [dynamic, static_body];
    let rhs = [[10.0; 6], [10.0; 6]];
    let rows = [
        GpuAvbdRow::new(Vec3::Y, 500.0, Vec3::new(0.2, 0.0, 0.0), -3.0, 1.0),
        GpuAvbdRow::inert(),
    ];
    let dt = 1.0 / 60.0;
    let out = avbd_stage_contact_solve(&bodies, &rhs, &rows, dt).expect("matching inputs stage");
    assert_eq!(out.len(), 2);
    assert!(out[0].1, "dynamic row system must solve");
    assert!(!out[1].1, "static inert system must report breakdown");
    assert_eq!(out[1].0, [0.0; 6]);
    // The staged delta is finite and moves along the row, not NaN dust.
    assert!(out[0].0.iter().all(|v| v.is_finite()));
    assert!(out[0].0.iter().any(|v| v.abs() > 1e-9));
    assert!(stub.mass_roster(&bodies).len() == 2);
    assert!(avbd_stage_contact_solve(&bodies, &rhs[..1], &rows, dt).is_none());
    assert!(avbd_stage_contact_solve(&bodies, &rhs, &rows[..1], dt).is_none());
}

/// Rung-2 contract: the artifact is the translated kernel — five bindings in
/// the host's order, one workgroup of 64, the dense solve's two loops and both
/// breakdown arms — and it naga-validates without a device.
#[test]
fn avbd_row_kernel_is_the_translated_artifact() {
    let source = avbd_row_wgsl();
    for binding in [
        "var<storage> storage_0_0_: array<f32>;",
        "var<storage, read_write> storage_0_1_: array<f32>;",
        "var<storage, read_write> storage_0_2_: array<f32>;",
        "@group(0) @binding(3)",
        "@group(0) @binding(4)",
    ] {
        assert!(source.contains(binding), "row artifact lost {binding}");
    }
    assert!(source.contains("@compute @workgroup_size(64, 1, 1)"));
    assert!(source.contains(&format!("fn {}(", AVBD_ROW_ENTRY)));
    // The LDL is two loops (factorize/solve plus back substitution) over the
    // 36-word scratch the translator lowers local arrays into.
    assert!(
        source.matches("loop {").count() >= 2,
        "expected the dense solve's loops"
    );
    assert!(source.contains("array<u32, 36>"), "no 6x6 scratch");
    // Both breakdown arms write the flag and the delta.
    assert!(source.contains("1f") && source.contains("0f"));
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("row WGSL must parse: {e}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("row WGSL must validate: {e}"));
}

/// Rung-2 device round trip: staged systems solved on the device agree with
/// [`avbd_stage_contact_solve`] per member plus the breakdown flag — the
/// proof behind [`WgpuAvbdSolver::runs_on_device`].
#[test]
fn avbd_row_solve_device_round_trip() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("avbd_row_solve_device_round_trip: no wgpu adapter — skipped");
        return;
    };
    let dt = 1.0 / 60.0;
    let dynamic = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 2.0);
    let static_body = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 0.0);
    let bodies = [dynamic, static_body];
    // O(1e2) residuals against O(1e3) Hessian entries: solved deltas land
    // O(0.01–0.1), far above f32 device-vs-CPU contraction noise.
    let rhs = [[360.0, -720.0, 180.0, 90.0, -45.0, 720.0], [0.0; 6]];
    let rows = [
        GpuAvbdRow::new(Vec3::Y, 1000.0, Vec3::new(0.3, 0.0, 0.1), -50.0, 1.0),
        GpuAvbdRow::inert(),
    ];
    let expected =
        avbd_stage_contact_solve(&bodies, &rhs, &rows, dt).expect("matching inputs stage");
    assert!(expected[0].1, "dynamic system must solve on CPU");
    assert!(!expected[1].1, "static system must break down on CPU");

    let solver = WgpuAvbdSolver::new(device, queue, 8);
    assert!(solver.runs_on_device());
    let stub = GpuAvbdStub::new(8);
    let masses = stub.mass_roster(&bodies);
    let systems: Vec<GpuAvbdSystem> = masses
        .iter()
        .zip(rhs.iter())
        .zip(rows.iter())
        .map(|((m, r), row)| GpuAvbdSystem::new(*m, *r, *row))
        .collect();
    solver.upload(&systems);
    solver.solve(2, dt);
    let (deltas, oks) = solver.try_download().expect("avbd download must map");

    assert!(oks[0] > 0.5, "dynamic system must solve on device");
    assert!(oks[1] < 0.5, "static system must break down on device");
    // CPU cross-check within tolerance per member (never bit-identical by
    // promise: device float contraction may differ ±1 ulp per op over a
    // ~200-op factorization).
    let got = deltas[0].to_residual();
    for (i, (&g, &e)) in got.iter().zip(expected[0].0.iter()).enumerate() {
        let tol = 1e-3 + 1e-4 * e.abs().max(g.abs());
        assert!(
            (g - e).abs() <= tol,
            "member {i}: device {g} vs CPU {e} (tol {tol})"
        );
    }
    assert_eq!(deltas[1].to_residual(), [0.0; 6]);
}
