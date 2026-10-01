//! Soft-body self-collision (D1 leftover #1): intra-body particle repulsion.
//!
//! Gauss–Seidel passes pushing overlapping particles of a single
//! [`crate::soft::SoftBody`] apart. Particles collide as spheres of radius
//! [`crate::soft::SoftBody::contact_radius`] — the same field that sizes the
//! deformable↔rigid coupling spheres (D1.4), hence dual-use: zero disables
//! both paths. Constrained pairs keep their distance rows as the sole
//! authority (no fighting between equality and contact solves); contacts are
//! rigid inequalities (`C = dist − (ri + rj) ≥ 0`) with a persistent
//! multiplier `λ ≥ 0` accumulated across the iterations of one substep (the
//! Small-Steps analogue of the contact warm-start cache), reset by
//! [`crate::soft::SoftBody::begin_substep`].

use std::collections::{HashMap, HashSet};

use glam::Vec3;

use crate::constants::NEAR_ZERO;
use crate::soft::SoftBody;

/// One Gauss–Seidel repulsion pass over the particles of `body`.
///
/// Uniform grid hash with cell `2 · contact_radius` (the interaction
/// diameter); `contact_radius <= 0` (or non-finite) returns early — the
/// coupling is disabled for the whole body. Each particle is a sphere of
/// radius `contact_radius` (dual-use with the rigid-coupling radius, see the
/// module docs).
///
/// Skips: pairs already linked by a [`crate::soft::DeformConstraint`]
/// (collected into a `HashSet<(min, max)>` once per call), pairs with zero
/// total inverse mass (both pinned), and separated pairs with no stored
/// multiplier. Surviving pairs solve the inequality
/// `C = dist − (ri + rj) ≥ 0` with the persistent `λ ≥ 0` kept in
/// [`crate::soft::SoftBody::self_collision_lambda`] under the canonical
/// `(min, max)` key: `Δλ = (−C − λ̃·λ)/(w + λ̃)` with zero compliance,
/// clamped to `λ ≥ 0`. Separated pairs (`C ≥ 0`) only drop a stale
/// multiplier, never pulling particles together — the pass is
/// attraction-free. The engine calls this once per solver iteration, so
/// the multipliers accumulate within one substep (a resting fold converges
/// instead of re-applying the full push every pass) and are reset by
/// [`crate::soft::SoftBody::begin_substep`].
///
/// Degenerate coincident positions use a `+X` fallback axis instead of
/// producing NaN; non-finite positions are skipped. `h` is the substep size,
/// currently unused (rigid contact, no compliance) and kept for solver-pass
/// signature symmetry.
pub(crate) fn solve_self_collision(body: &mut SoftBody, h: f32) {
    let _ = h;
    let r = body.contact_radius;
    if !r.is_finite() || r <= 0.0 {
        return;
    }
    let cell = 2.0 * r;
    if !cell.is_finite() || cell <= 0.0 {
        return;
    }
    let n = body.particles.len();
    if n < 2 {
        return;
    }
    let diameter = cell;

    let linked: HashSet<(crate::soft::ParticleIdx, crate::soft::ParticleIdx)> = body
        .constraints
        .iter()
        .filter(|c| c.a != c.b)
        .map(|c| (c.a.min(c.b), c.a.max(c.b)))
        .collect();

    let cell_of = |p: Vec3| {
        (
            (p.x / cell).floor() as i32,
            (p.y / cell).floor() as i32,
            (p.z / cell).floor() as i32,
        )
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, p) in body.particles.iter().enumerate() {
        if !p.position.is_finite() {
            continue;
        }
        grid.entry(cell_of(p.position)).or_default().push(i);
    }

    for i in 0..n {
        let origin = body.particles[i].position;
        if !origin.is_finite() {
            continue;
        }
        let (cx, cy, cz) = cell_of(origin);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    let Some(bucket) = grid.get(&key) else {
                        continue;
                    };
                    for &j in bucket.iter() {
                        if j <= i {
                            continue;
                        }
                        // Canonical pair key (`i < j` by construction).
                        if linked.contains(&(
                            crate::soft::ParticleIdx::from(i),
                            crate::soft::ParticleIdx::from(j),
                        )) {
                            continue;
                        }
                        let wa = body.particles[i].inv_mass;
                        let wb = body.particles[j].inv_mass;
                        if !body.particles[j].position.is_finite() {
                            continue;
                        }
                        let w = wa + wb;
                        if w <= 0.0 {
                            continue;
                        }
                        // Live positions: earlier pairs in this pass already moved `i`.
                        let pa = body.particles[i].position;
                        let pb = body.particles[j].position;
                        let delta = pa - pb;
                        let dist = delta.length();
                        if !dist.is_finite() {
                            continue;
                        }
                        let key = (i as u32, j as u32);
                        let lambda = body.self_collision_lambda.get(&key).copied().unwrap_or(0.0);
                        let c = dist - diameter;
                        if c >= 0.0 {
                            // Separating or clean: drop a stale multiplier
                            // without ever pulling back (no attraction).
                            if lambda > 0.0 {
                                body.self_collision_lambda.remove(&key);
                            }
                            continue;
                        }
                        let axis = if dist < NEAR_ZERO {
                            Vec3::X
                        } else {
                            delta / dist
                        };
                        let dlambda = crate::xpbd::delta_lambda(c, lambda, w, 0.0);
                        let next = (lambda + dlambda).max(0.0);
                        let applied = next - lambda;
                        if next <= 0.0 {
                            body.self_collision_lambda.remove(&key);
                        } else {
                            body.self_collision_lambda.insert(key, next);
                        }
                        if applied == 0.0 || !applied.is_finite() {
                            continue;
                        }
                        body.particles[i].position += axis * (applied * wa);
                        body.particles[j].position -= axis * (applied * wb);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soft::{DeformConstraint, DeformKind, SoftBody};
    use glam::Vec3;

    fn free_body(a: Vec3, b: Vec3, radius: f32) -> SoftBody {
        // Built via the `chain` builder (never a struct literal) so later
        // `SoftBody` field additions don't break these tests.
        let mut body = SoftBody::chain(a, Vec3::X, 2, 1.0, 1.0, 0.0);
        body.particles[0].position = a;
        body.particles[0].prev_position = a;
        body.particles[0].inv_mass = 1.0;
        body.particles[1].position = b;
        body.particles[1].prev_position = b;
        body.constraints.clear();
        body.contact_radius = radius;
        body
    }

    /// Two overlapping free particles separate to exactly the interaction
    /// diameter `2r`, split evenly for equal masses.
    #[test]
    fn overlapping_free_pair_separates_to_diameter() {
        let r = 0.1;
        let mut body = free_body(Vec3::ZERO, Vec3::new(0.05, 0.0, 0.0), r);
        solve_self_collision(&mut body, 1.0 / 60.0);
        let dist = (body.particles[0].position - body.particles[1].position).length();
        assert!(
            (dist - 2.0 * r).abs() < 1e-5,
            "separation {dist}, want {}",
            2.0 * r
        );
        for p in &body.particles {
            assert!(p.position.is_finite());
        }
    }

    /// A pair already joined by a distance row is left alone: the equality
    /// constraint owns that distance, not the contact pass.
    #[test]
    fn constrained_pair_is_untouched() {
        let r = 0.1;
        let mut body = free_body(Vec3::ZERO, Vec3::new(0.05, 0.0, 0.0), r);
        body.constraints.push(DeformConstraint {
            a: crate::soft::ParticleIdx::from_raw(0),
            b: crate::soft::ParticleIdx::from_raw(1),
            rest: 0.05,
            compliance: 0.0,
            kind: DeformKind::Structural,
            lambda: 0.0,
        });
        let before = [body.particles[0].position, body.particles[1].position];
        solve_self_collision(&mut body, 1.0 / 60.0);
        assert_eq!(body.particles[0].position, before[0]);
        assert_eq!(body.particles[1].position, before[1]);
    }

    /// The pinned particle never moves; the free one takes the full
    /// correction and ends a full diameter away.
    #[test]
    fn pinned_particle_holds_while_free_moves_away() {
        let r = 0.1;
        let mut body = free_body(Vec3::ZERO, Vec3::new(0.05, 0.0, 0.0), r);
        body.particles[0].inv_mass = 0.0;
        solve_self_collision(&mut body, 1.0 / 60.0);
        assert_eq!(body.particles[0].position, Vec3::ZERO);
        let dist = (body.particles[0].position - body.particles[1].position).length();
        assert!(
            (dist - 2.0 * r).abs() < 1e-5,
            "separation {dist}, want {}",
            2.0 * r
        );
    }

    /// Zero radius disables the coupling: overlapping particles are a no-op,
    /// including the coincident-position degenerate case (no NaN).
    #[test]
    fn zero_radius_is_noop() {
        let mut body = free_body(Vec3::ZERO, Vec3::ZERO, 0.0);
        solve_self_collision(&mut body, 1.0 / 60.0);
        assert_eq!(body.particles[0].position, Vec3::ZERO);
        assert_eq!(body.particles[1].position, Vec3::ZERO);
    }

    /// A resting fold converges across the iterations of one substep instead
    /// of vibrating: penetration never grows, later passes move less than
    /// the first, and multipliers stay `λ ≥ 0`.
    #[test]
    fn resting_fold_converges_without_vibration() {
        let r = 0.1;
        let diameter = 2.0 * r;
        let h = 1.0 / 60.0;
        // Three overlapping free particles, no distance rows (built via the
        // `chain` builder, then unpinned and unlinked).
        let mut body = SoftBody::chain(Vec3::ZERO, Vec3::X, 3, 1.0, 1.0, 0.0);
        body.constraints.clear();
        for (p, x) in body.particles.iter_mut().zip([0.0, 0.15, 0.30]) {
            p.position = Vec3::new(x, 0.0, 0.0);
            p.prev_position = p.position;
            p.inv_mass = 1.0;
        }
        body.contact_radius = r;
        let penetration = |body: &SoftBody| {
            let mut worst = 0.0f32;
            for i in 0..body.particles.len() {
                for j in (i + 1)..body.particles.len() {
                    let d = (body.particles[i].position - body.particles[j].position).length();
                    worst = worst.max(diameter - d);
                }
            }
            worst
        };
        let snapshot = |body: &SoftBody| {
            body.particles
                .iter()
                .map(|p| p.position)
                .collect::<Vec<_>>()
        };
        let travel = |a: &[Vec3], b: &[Vec3]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (*x - *y).length())
                .sum::<f32>()
        };

        let pen0 = penetration(&body);
        assert!(pen0 > 0.0, "the fold starts penetrating");
        let before = snapshot(&body);
        solve_self_collision(&mut body, h);
        let pen1 = penetration(&body);
        let after_first = snapshot(&body);
        let d1 = travel(&before, &after_first);
        assert!(
            pen1 <= pen0,
            "first pass reduces penetration ({pen1} > {pen0})"
        );
        assert!(d1 > 0.0, "first pass moves the fold apart");
        assert!(
            body.self_collision_lambda.values().all(|&l| l >= 0.0),
            "multipliers stay non-negative"
        );
        assert!(
            !body.self_collision_lambda.is_empty(),
            "active contacts accumulate lambda"
        );

        // Second iteration of the same substep (no `begin_substep` reset).
        solve_self_collision(&mut body, h);
        let pen2 = penetration(&body);
        let after_second = snapshot(&body);
        let d2 = travel(&after_first, &after_second);
        assert!(
            pen2 <= pen1 + 1e-6,
            "penetration never grows ({pen2} > {pen1})"
        );
        assert!(
            d2 <= d1 + 1e-6,
            "passes decay instead of vibrating ({d2} > {d1})"
        );

        // Further iterations settle: no re-penetration, no NaN.
        for _ in 0..8 {
            solve_self_collision(&mut body, h);
        }
        assert!(penetration(&body) <= pen1 + 1e-5, "the fold stays settled");
        for p in &body.particles {
            assert!(p.position.is_finite());
        }

        // A fresh substep resets the accumulation (Small-Steps regime).
        body.begin_substep();
        assert!(body.self_collision_lambda.is_empty());
    }
}
