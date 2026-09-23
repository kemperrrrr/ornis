//! Soft-body self-collision (D1 leftover #1): intra-body particle repulsion.
//!
//! One Gauss–Seidel pass pushing overlapping particles of a single
//! [`crate::soft::SoftBody`] apart. Particles collide as spheres of radius
//! [`crate::soft::SoftBody::contact_radius`] — the same field that sizes the
//! deformable↔rigid coupling spheres (D1.4), hence dual-use: zero disables
//! both paths. Constrained pairs keep their distance rows as the sole
//! authority (no fighting between equality and contact solves); there is no
//! persistent Lagrange multiplier, contacts are rigid inequalities resolved
//! once per call.

use std::collections::{HashMap, HashSet};

use glam::Vec3;

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
/// total inverse mass (both pinned), and separated pairs. Surviving pairs
/// solve the inequality `C = dist − (ri + rj)` only when `C < 0`, pushing
/// apart proportional to `inv_mass` with no persistent lambda (single pass).
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

    let linked: HashSet<(usize, usize)> = body
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
                        if linked.contains(&(i, j)) {
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
                        if dist >= diameter {
                            continue;
                        }
                        let axis = if dist < 1e-9 { Vec3::X } else { delta / dist };
                        let c = dist - diameter;
                        debug_assert!(c < 0.0);
                        let dlambda = crate::xpbd::delta_lambda(c, 0.0, w, 0.0);
                        if dlambda == 0.0 || !dlambda.is_finite() {
                            continue;
                        }
                        body.particles[i].position += axis * (dlambda * wa);
                        body.particles[j].position -= axis * (dlambda * wb);
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
            a: 0,
            b: 1,
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
}
