//! Host-side execution tests for the skinning guest through the GPU ABI shadow.

use ornis_gpu_abi::HostMemory;
use ornis_gpu_abi::with_host;

use super::BIND_BINDING;
use super::JOINTS_BINDING;
use super::OUT_BINDING;
use super::PALETTE_BINDING;
use super::PARAMS_BINDING;
use super::WEIGHTS_BINDING;
use super::skin;
use super::sqrt_approx;

const VERTICES: usize = 4;
const JOINTS: usize = 2;
const PALETTE_LANES: usize = JOINTS * 32;
const VERTEX_LANES: usize = VERTICES * 6;
const INFLUENCE_LANES: usize = VERTICES * 4;
const PARAM_WORDS: usize = 2;
const TOL: f32 = 1e-6;

/// Column-major identity joint (skin matrix and normal matrix).
const IDENTITY_JOINT: [f32; 32] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// Column-major rigid joint: `Rz(90°)` with translation `(1, 0, 0)`; the
/// normal matrix drops the translation.
const ROTATED_JOINT: [f32; 32] = [
    0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

fn palette() -> [f32; PALETTE_LANES] {
    let mut lanes = [0.0; PALETTE_LANES];
    lanes[..32].copy_from_slice(&IDENTITY_JOINT);
    lanes[32..].copy_from_slice(&ROTATED_JOINT);
    lanes
}

fn bind_vertices() -> [f32; VERTEX_LANES] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, // v0: joint 1 only
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // v1: 50/50 blend
        2.0, 0.0, 0.0, 0.0, 1.0, 0.0, // v2: out-of-range joint
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, // v3: zero weights
    ]
}

fn run_skin() -> [f32; VERTEX_LANES] {
    let joints: [u32; INFLUENCE_LANES] = [1, 0, 0, 0, 0, 1, 0, 0, 7, 0, 0, 0, 1, 0, 0, 0];
    let weights: [f32; INFLUENCE_LANES] = [
        1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let params = [VERTICES as u32, JOINTS as u32];
    let mut memory = HostMemory::new();
    memory.bind_storage::<0, PALETTE_BINDING, [f32; PALETTE_LANES]>();
    memory.write_storage::<0, PALETTE_BINDING, [f32; PALETTE_LANES]>(&palette());
    memory.bind_storage::<0, BIND_BINDING, [f32; VERTEX_LANES]>();
    memory.write_storage::<0, BIND_BINDING, [f32; VERTEX_LANES]>(&bind_vertices());
    memory.bind_storage::<0, JOINTS_BINDING, [u32; INFLUENCE_LANES]>();
    memory.write_storage::<0, JOINTS_BINDING, [u32; INFLUENCE_LANES]>(&joints);
    memory.bind_storage::<0, WEIGHTS_BINDING, [f32; INFLUENCE_LANES]>();
    memory.write_storage::<0, WEIGHTS_BINDING, [f32; INFLUENCE_LANES]>(&weights);
    memory.bind_storage::<0, PARAMS_BINDING, [u32; PARAM_WORDS]>();
    memory.write_storage::<0, PARAMS_BINDING, [u32; PARAM_WORDS]>(&params);
    memory.bind_storage::<0, OUT_BINDING, [f32; VERTEX_LANES]>();
    memory.write_storage::<0, OUT_BINDING, [f32; VERTEX_LANES]>(&[0.0; VERTEX_LANES]);
    with_host(&mut memory, || {
        for id in 0..(VERTICES as u32 + 1) {
            skin(id);
        }
    });
    memory.read_storage::<0, OUT_BINDING, [f32; VERTEX_LANES]>()
}

fn assert_vertex(actual: &[f32; VERTEX_LANES], vertex: usize, expected: [f32; 6]) {
    for (lane, want) in expected.iter().enumerate() {
        let got = actual[vertex * 6 + lane];
        assert!(
            (got - want).abs() < TOL,
            "vertex {vertex} lane {lane}: expected {want}, got {got}"
        );
    }
}

#[test]
fn skin_kernel_matches_hand_computed_rigid_blend() {
    let out = run_skin();
    // Joint 1 only: Rz(90°)·(1,0,0) + (1,0,0); the +Z normal is unchanged.
    assert_vertex(&out, 0, [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    // 50/50: position midpoint, normal renormalized to the diagonal.
    let diagonal = std::f32::consts::FRAC_1_SQRT_2;
    assert_vertex(&out, 1, [1.0, 0.5, 0.0, diagonal, diagonal, 0.0]);
}

#[test]
fn skin_kernel_falls_back_like_the_cpu_path() {
    let out = run_skin();
    // Out-of-range joint reads as identity: the bind vertex passes through.
    assert_vertex(&out, 2, [2.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    // Zero weights fall back to full weight on slot 0 (joint 1 here).
    assert_vertex(&out, 3, [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn sqrt_approx_matches_the_std_reference_across_the_normal_range() {
    // Blended-normal squared lengths sit near 1, but non-uniform joint
    // scales can move them orders of magnitude either way.
    for value in [1.0e-8_f32, 0.25, 1.0, 2.0, 3.999, 4.0, 100.0, 65_536.0, 1.0e12] {
        let expected = value.sqrt();
        let relative_error = (sqrt_approx(value) - expected).abs() / expected;
        assert!(
            relative_error < TOL,
            "sqrt({value}) relative error {relative_error}"
        );
    }
}
