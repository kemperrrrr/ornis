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
    run_skin_over(&palette())
}

fn run_skin_over(palette: &[f32; PALETTE_LANES]) -> [f32; VERTEX_LANES] {
    let joints: [u32; INFLUENCE_LANES] = [1, 0, 0, 0, 0, 1, 0, 0, 7, 0, 0, 0, 1, 0, 0, 0];
    let weights: [f32; INFLUENCE_LANES] = [
        1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let params = [VERTICES as u32, JOINTS as u32];
    let mut memory = HostMemory::new();
    memory.bind_storage::<0, PALETTE_BINDING, [f32; PALETTE_LANES]>();
    memory.write_storage::<0, PALETTE_BINDING, [f32; PALETTE_LANES]>(palette);
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
    for value in [
        1.0e-45_f32,
        1.0e-8,
        0.25,
        1.0,
        2.0,
        3.999,
        4.0,
        100.0,
        65_536.0,
        1.0e12,
        f32::MAX,
    ] {
        let expected = value.sqrt();
        let relative_error = (sqrt_approx(value) - expected).abs() / expected;
        assert!(
            relative_error < TOL,
            "sqrt({value}) relative error {relative_error}"
        );
    }
}

#[test]
fn sqrt_approx_terminates_on_non_finite_inputs() {
    assert!(sqrt_approx(f32::INFINITY).is_infinite());
    assert!(sqrt_approx(f32::NAN).is_nan());
    assert!(sqrt_approx(-1.0).is_nan());
    assert_eq!(sqrt_approx(0.0), 0.0);
}

/// Joint scale whose square overflows `f32` (the review's hang scenario).
const OVERFLOW_SCALE: f32 = 1.0e20;

/// Identity joint 0 plus joint 1 scaled by [`OVERFLOW_SCALE`] along x
/// (matrix and normal matrix alike).
fn overflow_palette() -> [f32; PALETTE_LANES] {
    let mut lanes = [0.0; PALETTE_LANES];
    lanes[..32].copy_from_slice(&IDENTITY_JOINT);
    lanes[32..].copy_from_slice(&IDENTITY_JOINT);
    lanes[32] = OVERFLOW_SCALE;
    lanes[48] = OVERFLOW_SCALE;
    lanes
}

#[test]
fn skin_kernel_terminates_when_the_blended_normal_overflows() {
    // The 50/50 vertex blends normals (1,0,0) and (1e20,0,0): the squared
    // length overflows f32, so the kernel must skip the renormalize and
    // emit the blended normal as-is instead of spinning in the sqrt.
    let out = run_skin_over(&overflow_palette());
    let expected = 0.5 + 0.5 * OVERFLOW_SCALE;
    let blend_vertex = 1;
    let normal_x = out[blend_vertex * 6 + 3];
    assert!(
        (normal_x - expected).abs() / expected < TOL,
        "expected unscaled normal x {expected}, got {normal_x}"
    );
    assert_eq!(out[blend_vertex * 6 + 4], 0.0);
    assert_eq!(out[blend_vertex * 6 + 5], 0.0);
}
