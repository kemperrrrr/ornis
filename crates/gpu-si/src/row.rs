//! Sequential-impulse isotropic contact row: one source of truth for CPU and GPU.
//!
//! The single-point contact row — a projected Gauss-Seidel normal update plus
//! the circular Coulomb-cone friction clamp — lives here as plain, host-runnable
//! Rust. The CPU wide path (`ornis-physics` wide/scalar islands) calls these
//! functions directly, and the GPU contact kernel (`cs_main_contact`, this
//! crate's entry) calls the same functions through the same names, so the
//! device parity gates compare the kernel against the exact arithmetic it was
//! translated from — no hand-copied second spelling that could drift on a NaN
//! or a ±0.
//!
//! # Pilot boundary (deliberate)
//!
//! Only the isotropic row is shared. Anisotropic (elliptical-cone) friction,
//! rolling/torsional resistance and the multi-point block LCP stay CPU-only:
//! they need per-point `Vec3` frames and small dense solves with no
//! lane-parallel GPU counterpart on this rung. The GPU shader keeps those
//! coefficients CPU-side, as before.
//!
//! CPU/GPU agreement is by tolerance, never bit-identical: the GPU pass is a
//! Jacobi/GS hybrid over wide batches (see `ornis-physics`'s `gpu` module).

use ornis_gpu_abi::thresholds::DEGENERATE;
use ornis_gpu_shader::sqrt;

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
///
/// # Errors
///
/// None — a pure float function.
#[must_use]
#[inline(always)]
pub fn contact_normal_step(vn: f32, spec_target: f32, inv_k: f32, acc: f32) -> f32 {
    let lambda = (spec_target - vn) * inv_k;
    let candidate = acc + lambda;
    let mut new_acc = 0.0;
    if candidate > 0.0 {
        new_acc = candidate;
    }
    new_acc
}

/// Circular Coulomb-cone clamp for one isotropic friction axis.
///
/// `new_t` is the unclamped candidate for this axis (old accumulation
/// plus the tangential corrective impulse), `other` the accumulated
/// impulse on the perpendicular axis, `max_friction` the cone radius
/// (`mu * normal_impulse`). Returns the candidate projected onto the
/// disc: untouched inside, scaled onto the circle outside. A degenerate
/// (zero-length) pair passes through untouched — no division happens.
///
/// # Errors
///
/// None — a pure float function.
#[must_use]
#[inline(always)]
pub fn contact_friction_clamp(new_t: f32, other: f32, max_friction: f32) -> f32 {
    let len = sqrt(new_t * new_t + other * other);
    let mut clamped = new_t;
    if len > max_friction && len > DEGENERATE {
        clamped = new_t * (max_friction / len);
    }
    clamped
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
        assert_close(contact_normal_step(-2.0, 0.0, 0.5, 0.0), 1.0);
        // Separating contact projects back to zero: no tensile pull.
        assert_close(contact_normal_step(1.0, 0.0, 0.5, 0.0), 0.0);
        // Warm start accumulates on top of the cached impulse.
        assert_close(contact_normal_step(-1.0, 0.0, 1.0, 2.0), 3.0);
        // Inert lane (zero inverse mass) keeps its accumulation.
        assert_close(contact_normal_step(-2.0, 0.0, 0.0, 1.5), 1.5);
    }

    #[test]
    fn friction_clamp_projects_circle() {
        // Inside the cone: untouched.
        assert_close(contact_friction_clamp(1.0, 1.0, 5.0), 1.0);
        // Outside: scaled onto the circle (3-4-5 triangle, cap 2.5).
        assert_close(contact_friction_clamp(3.0, 4.0, 2.5), 1.5);
        // Degenerate zero-length pair: untouched, no divide-by-zero.
        assert_close(contact_friction_clamp(0.0, 0.0, 1.0), 0.0);
        // Zero cap with a live candidate: fully clamped to zero.
        assert_close(contact_friction_clamp(2.0, 0.0, 0.0), 0.0);
    }
}
