//! Sequential-impulse isotropic contact row: one source of truth for CPU and GPU.
//!
//! The single-point contact row — a projected Gauss-Seidel normal update
//! plus the circular Coulomb-cone friction clamp — is written once as
//! `#[kernel]` functions below. The CPU wide path ([`crate::wide`]) and the
//! scalar island velocity solve ([`crate::engine`]) call `...::eval`
//! directly (the same Rust); the GPU `contact_solver` in `gpu.rs` (the
//! `gpu` feature) stitches the same sources ahead of its entry via
//! `helpers(...)` instead of carrying a mirror implementation.
//!
//! # DSL audit (no macro changes needed)
//!
//! The isotropic row needs scalar arithmetic plus two-way branches (`if`
//! with early `return`): no loops, no fixed-size scratch arrays, no
//! `match`. All three already lower through the kernel pipeline (`if`
//! expressions, the `max`/`sqrt` builtins, `&&`; the loop/array precedent
//! lives in `crates/macros/tests/compute_dsl.rs` and is exercised by the
//! AVBD rung-1 helpers). Nothing was extended: extending the macro is out
//! of scope for this pilot, and the row does not need it.
//!
//! # Pilot boundary (deliberate)
//!
//! Only the isotropic row is shared. Anisotropic (elliptical-cone)
//! friction, rolling/torsional resistance and the multi-point block LCP
//! stay CPU-only: they need per-point `Vec3` frames and small dense solves
//! with no lane-parallel GPU counterpart on this rung. The GPU shader keeps
//! those coefficients CPU-side, as before.
//!
//! # Packaging note
//!
//! `#[kernel]` expansions reference `wgpu` types for their pipeline
//! helpers, so `wgpu` is an unconditional dependency of this crate while
//! the `gpu` Cargo feature keeps gating device execution (the `gpu`
//! module, solver attach, GPU branches). The workspace already builds
//! `wgpu` unconditionally via `ornis-render`, so this adds no new build
//! cost there.
//!
//! CPU/GPU agreement is by tolerance, never bit-identical: the GPU pass is
//! a Jacobi/GS hybrid over wide batches (see `gpu.rs`).

use ornis_macros::kernel;

/// Projected Gauss-Seidel normal update for one isotropic contact.
///
/// Returns the new accumulated impulse: the old accumulation plus the
/// corrective impulse `(target - vn) * inv_k`, projected to the
/// non-negative (push-only) half-line. The caller applies
/// `new_acc - acc` along the contact normal.
///
/// `vn` is the current relative normal velocity (separating positive),
/// `spec_target` the speculative approach-speed limit (`0` when touching;
/// named to avoid the reserved WGSL word `target`),
/// `inv_k` the precomputed inverse effective mass (`0` for inert lanes).
#[kernel]
fn contact_normal_step(vn: f32, spec_target: f32, inv_k: f32, acc: f32) -> f32 {
    let lambda = (spec_target - vn) * inv_k;
    let new_acc = (acc + lambda).max(0.0);
    return new_acc;
}

/// Circular Coulomb-cone clamp for one isotropic friction axis.
///
/// `new_t` is the unclamped candidate for this axis (old accumulation
/// plus the tangential corrective impulse), `other` the accumulated
/// impulse on the perpendicular axis, `max_friction` the cone radius
/// (`mu * normal_impulse`). Returns the candidate projected onto the
/// disc: untouched inside, scaled onto the circle outside. A degenerate
/// (zero-length) pair passes through untouched — no division happens.
#[kernel]
fn contact_friction_clamp(new_t: f32, other: f32, max_friction: f32) -> f32 {
    let len = (new_t * new_t + other * other).sqrt();
    if len > max_friction && len > 1e-12 {
        return new_t * (max_friction / len);
    }
    return new_t;
}

#[cfg(test)]
mod tests {
    use super::{contact_friction_clamp, contact_normal_step};

    fn assert_close(got: f32, want: f32) {
        assert!(
            (got - want).abs() < 1e-6,
            "kernel eval diverged: got {got}, want {want}"
        );
    }

    #[test]
    fn normal_step_accumulates_and_projects() {
        // Approaching contact accumulates a positive impulse.
        assert_close(contact_normal_step::eval(-2.0, 0.0, 0.5, 0.0), 1.0);
        // Separating contact projects back to zero: no tensile pull.
        assert_close(contact_normal_step::eval(1.0, 0.0, 0.5, 0.0), 0.0);
        // Warm start accumulates on top of the cached impulse.
        assert_close(contact_normal_step::eval(-1.0, 0.0, 1.0, 2.0), 3.0);
        // Inert lane (zero inverse mass) keeps its accumulation.
        assert_close(contact_normal_step::eval(-2.0, 0.0, 0.0, 1.5), 1.5);
    }

    #[test]
    fn friction_clamp_projects_circle() {
        // Inside the cone: untouched.
        assert_close(contact_friction_clamp::eval(1.0, 1.0, 5.0), 1.0);
        // Outside: scaled onto the circle (3-4-5 triangle, cap 2.5).
        assert_close(contact_friction_clamp::eval(3.0, 4.0, 2.5), 1.5);
        // Degenerate zero-length pair: untouched, no divide-by-zero.
        assert_close(contact_friction_clamp::eval(0.0, 0.0, 1.0), 0.0);
        // Zero cap with a live candidate: fully clamped to zero.
        assert_close(contact_friction_clamp::eval(2.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn helper_sources_mention_the_row_and_validate_with_naga() {
        let normal = contact_normal_step::wgsl_source();
        assert!(
            normal.contains(
                "fn contact_normal_step(vn: f32, spec_target: f32, inv_k: f32, acc: f32) -> f32"
            ),
            "unexpected normal helper source: {normal}"
        );
        assert!(
            normal.contains("max((acc + lambda), 0.0)"),
            "normal projection must survive translation: {normal}"
        );
        let friction = contact_friction_clamp::wgsl_source();
        assert!(
            friction.contains(
                "fn contact_friction_clamp(new_t: f32, other: f32, max_friction: f32) -> f32"
            ),
            "unexpected friction helper source: {friction}"
        );
        assert!(
            friction.contains("sqrt((new_t * new_t + other * other))")
                || friction.contains("sqrt(new_t * new_t + other * other)"),
            "cone length must survive translation: {friction}"
        );
        for (name, src) in [
            ("contact_normal_step", normal),
            ("contact_friction_clamp", friction),
        ] {
            let module = naga::front::wgsl::parse_str(src)
                .unwrap_or_else(|e| panic!("{name} helper must parse: {e}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} helper must validate: {e:?}"));
        }
    }
}
