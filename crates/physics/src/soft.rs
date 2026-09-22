//! Deformable bodies (PLAN B2/D1): particles and distance constraints.
//!
//! [`Particle`] is a 3-DOF point mass — deliberately not a [`crate::body::RigidBody`]:
//! hauling a quaternion and an inertia tensor per cloth vertex is wasteful
//! and meaningless. [`SoftBody`] bundles particles with a distance-constraint
//! topology (structural/shear/bend groups, each with its own compliance).
//! The solver is XPBD on the shared [`crate::xpbd`] kernel: constraints are
//! equalities solved with `Δλ = (−C − α̃λ)/(w + α̃)`, `α̃ = α/h²`, one
//! Lagrange multiplier per scalar row, reset every substep (Small-Steps
//! regime — see [`crate::xpbd`]).
//!
//! Builders cover the first two D1 scenes: [`SoftBody::chain`] (rope/cable)
//! and [`SoftBody::cloth_grid`] (draping sheet). Volume preservation,
//! deformable↔rigid coupling and render-mesh upload are later D1 steps and
//! explicitly absent here.

use glam::Vec3;

/// Stable index of a soft body inside [`crate::xpbd::XpbdEngine`].
///
/// Dense like [`crate::body::BodyHandle`]: removal swaps the last body into
/// the freed slot, so only the moved body's handle changes.
pub type SoftHandle = usize;

/// Constraint group inside a deformable: structural/shear/bend rows share
/// one formula and differ only in topology and compliance (stiff stretch,
/// soft bend — the cloth recipe from the XPBD paper).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeformKind {
    /// Primary topology (chain links, cloth warp/weft).
    Structural,
    /// Diagonals resisting in-plane shear.
    Shear,
    /// Skip-one rows resisting bending (soft).
    Bend,
}

/// One 3-DOF point mass of a [`SoftBody`].
///
/// `inv_mass == 0` pins the particle (static anchor): the solver skips it
/// and it never integrates. `prev_position` is the substep-start pose used
/// for the BDF1-style velocity update.
#[derive(Debug, Clone)]
pub struct Particle {
    /// World-space position (solved state).
    pub position: Vec3,
    /// Position at the substep start (velocity baseline).
    pub prev_position: Vec3,
    /// Linear velocity (m/s).
    pub velocity: Vec3,
    /// Cached `1 / mass` (0 = pinned) — what the solver actually uses.
    pub inv_mass: f32,
}

impl Particle {
    /// Free particle of `mass` at `position` (`mass <= 0` pins it).
    pub fn new(position: Vec3, mass: f32) -> Self {
        Self {
            position,
            prev_position: position,
            velocity: Vec3::ZERO,
            inv_mass: if mass > 0.0 { 1.0 / mass } else { 0.0 },
        }
    }

    /// Whether the solver treats this particle as an immovable anchor.
    pub fn is_pinned(&self) -> bool {
        self.inv_mass <= 0.0
    }
}

/// One scalar distance constraint between two particles of the same body.
#[derive(Debug, Clone)]
pub struct DeformConstraint {
    /// First particle index.
    pub a: usize,
    /// Second particle index.
    pub b: usize,
    /// Rest length (m).
    pub rest: f32,
    /// Compliance `α` (inverse stiffness in m/N, 0 = inextensible).
    pub compliance: f32,
    /// Topology group (structural/shear/bend).
    pub kind: DeformKind,
    /// Accumulated multiplier, live only within one substep's iterations.
    pub lambda: f32,
}

/// Which particles of a [`SoftBody::cloth_grid`] start pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClothPin {
    /// Nothing pinned (free fall).
    None,
    /// The whole top row (`row == 0`) pinned — curtain scenario.
    TopRow,
    /// Only the two top corners pinned — drape scenario.
    Corners,
}

/// A deformable body: particles plus a distance-constraint topology.
///
/// Constraints reference particle indices of this body only (no cross-body
/// rows in D1.1); invalid indices are rejected by the builders, and the
/// solver skips degenerate rows defensively.
#[derive(Debug, Clone)]
pub struct SoftBody {
    /// Particles in index order.
    pub particles: Vec<Particle>,
    /// Distance rows over the particles.
    pub constraints: Vec<DeformConstraint>,
}

impl SoftBody {
    /// Rope/cable: `count` particles from `origin` along `dir` (normalized
    /// internally) spaced `spacing` apart, each of `mass`, linked by
    /// structural rows of `compliance`. The first particle is pinned.
    pub fn chain(
        origin: Vec3,
        dir: Vec3,
        count: usize,
        spacing: f32,
        mass: f32,
        compliance: f32,
    ) -> Self {
        let dir = dir.normalize_or(Vec3::NEG_Y);
        let mut particles = Vec::with_capacity(count);
        for i in 0..count {
            let mut p = Particle::new(origin + dir * (i as f32 * spacing), mass);
            if i == 0 {
                p.inv_mass = 0.0;
            }
            particles.push(p);
        }
        let mut constraints = Vec::with_capacity(count.saturating_sub(1));
        for i in 0..count.saturating_sub(1) {
            constraints.push(DeformConstraint {
                a: i,
                b: i + 1,
                rest: spacing,
                compliance,
                kind: DeformKind::Structural,
                lambda: 0.0,
            });
        }
        Self {
            particles,
            constraints,
        }
    }

    /// Cloth sheet in the local XY plane: `cols × rows` particles from
    /// `origin` (top-left corner), spaced `spacing` apart, each of `mass`.
    /// Structural rows link grid neighbours, shear rows link diagonals and
    /// bend rows skip one vertex — each group with its own compliance.
    /// `pin` selects the initially pinned set.
    #[allow(clippy::too_many_arguments)]
    pub fn cloth_grid(
        origin: Vec3,
        cols: usize,
        rows: usize,
        spacing: f32,
        mass: f32,
        structural: f32,
        shear: f32,
        bend: f32,
        pin: ClothPin,
    ) -> Self {
        let at = |c: usize, r: usize| r * cols + c;
        let pinned = |c: usize, r: usize| match pin {
            ClothPin::None => false,
            ClothPin::TopRow => r == 0,
            ClothPin::Corners => r == 0 && (c == 0 || c + 1 == cols),
        };
        let mut particles = Vec::with_capacity(cols * rows);
        for r in 0..rows {
            for c in 0..cols {
                let mut p = Particle::new(
                    origin + Vec3::new(c as f32 * spacing, -(r as f32) * spacing, 0.0),
                    mass,
                );
                if pinned(c, r) {
                    p.inv_mass = 0.0;
                }
                particles.push(p);
            }
        }
        let mut constraints = Vec::new();
        let mut link = |a: usize, b: usize, rest: f32, compliance: f32, kind: DeformKind| {
            if a < particles.len() && b < particles.len() && a != b {
                constraints.push(DeformConstraint {
                    a,
                    b,
                    rest,
                    compliance,
                    kind,
                    lambda: 0.0,
                });
            }
        };
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    link(
                        at(c, r),
                        at(c + 1, r),
                        spacing,
                        structural,
                        DeformKind::Structural,
                    );
                }
                if r + 1 < rows {
                    link(
                        at(c, r),
                        at(c, r + 1),
                        spacing,
                        structural,
                        DeformKind::Structural,
                    );
                }
                if c + 1 < cols && r + 1 < rows {
                    let d = spacing * std::f32::consts::SQRT_2;
                    link(at(c, r), at(c + 1, r + 1), d, shear, DeformKind::Shear);
                    link(at(c + 1, r), at(c, r + 1), d, shear, DeformKind::Shear);
                }
                if c + 2 < cols {
                    link(
                        at(c, r),
                        at(c + 2, r),
                        2.0 * spacing,
                        bend,
                        DeformKind::Bend,
                    );
                }
                if r + 2 < rows {
                    link(
                        at(c, r),
                        at(c, r + 2),
                        2.0 * spacing,
                        bend,
                        DeformKind::Bend,
                    );
                }
            }
        }
        Self {
            particles,
            constraints,
        }
    }

    /// Number of particles.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Number of distance rows.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Semi-implicit Euler prediction for one substep of size `h` under
    /// `gravity`: pinned particles only refresh their baseline.
    pub fn integrate(&mut self, h: f32, gravity: Vec3) {
        for p in &mut self.particles {
            p.prev_position = p.position;
            if p.inv_mass <= 0.0 {
                p.velocity = Vec3::ZERO;
                continue;
            }
            p.velocity += h * gravity;
            p.position += h * p.velocity;
        }
    }

    /// Gauss–Seidel sweep over the distance rows (equalities): each row
    /// evaluates `C = |pa − pb| − rest` at the live positions and applies
    /// the XPBD update. Multipliers start at 0 every substep (reset by
    /// [`SoftBody::begin_substep`], not here — iterations within one
    /// substep accumulate).
    pub fn solve_constraints(&mut self, h: f32) {
        for c in &mut self.constraints {
            let (pa, pb) = match pair(&self.particles, c.a, c.b) {
                Some(pair) => pair,
                None => continue,
            };
            let delta = pa.position - pb.position;
            let dist = delta.length();
            if dist < 1e-9 {
                continue;
            }
            let n = delta / dist;
            let w = pa.inv_mass + pb.inv_mass;
            if w <= 0.0 {
                continue;
            }
            let alpha_tilde = c.compliance / (h * h);
            let dlambda = crate::xpbd::delta_lambda(dist - c.rest, c.lambda, w, alpha_tilde);
            c.lambda += dlambda;
            let (pa, pb) = match pair_mut(&mut self.particles, c.a, c.b) {
                Some(pair) => pair,
                None => continue,
            };
            if dlambda != 0.0 {
                pa.position += n * (dlambda * pa.inv_mass);
                pb.position -= n * (dlambda * pb.inv_mass);
            }
        }
    }

    /// Reset per-substep multiplier state (Small-Steps: every substep is a
    /// fresh XPBD solve starting from `λ = 0`).
    pub fn begin_substep(&mut self) {
        for c in &mut self.constraints {
            c.lambda = 0.0;
        }
    }

    /// BDF1-style velocity update from the solved positions.
    pub fn update_velocities(&mut self, h: f32) {
        for p in &mut self.particles {
            if p.inv_mass <= 0.0 {
                p.velocity = Vec3::ZERO;
                continue;
            }
            p.velocity = (p.position - p.prev_position) / h;
        }
    }
}

/// Shared read of two distinct particles (`None` for bad indices).
fn pair(particles: &[Particle], a: usize, b: usize) -> Option<(&Particle, &Particle)> {
    if a == b || a >= particles.len() || b >= particles.len() {
        return None;
    }
    Some((&particles[a], &particles[b]))
}

/// Exclusive mutable borrow of two distinct particles.
fn pair_mut(
    particles: &mut [Particle],
    a: usize,
    b: usize,
) -> Option<(&mut Particle, &mut Particle)> {
    if a == b || a >= particles.len() || b >= particles.len() {
        return None;
    }
    if a < b {
        let (lo, hi) = particles.split_at_mut(b);
        Some((&mut lo[a], &mut hi[0]))
    } else {
        let (lo, hi) = particles.split_at_mut(a);
        Some((&mut hi[0], &mut lo[b]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builder topology: chain links neighbours, pins the first particle.
    #[test]
    fn chain_builder_links_and_pins() {
        let body = SoftBody::chain(Vec3::ZERO, Vec3::NEG_Y, 6, 0.5, 1.0, 0.0);
        assert_eq!(body.particle_count(), 6);
        assert_eq!(body.constraint_count(), 5);
        assert!(body.particles[0].is_pinned());
        assert!(
            body.constraints
                .iter()
                .all(|c| c.kind == DeformKind::Structural)
        );
        assert!(body.particles[1..].iter().all(|p| !p.is_pinned()));
    }

    /// Builder topology: grid counts per group + top-row pinning.
    #[test]
    fn cloth_grid_builder_counts_and_pins() {
        let (cols, rows) = (5, 4);
        let body = SoftBody::cloth_grid(
            Vec3::ZERO,
            cols,
            rows,
            0.25,
            1.0,
            0.0,
            0.0,
            1e-4,
            ClothPin::TopRow,
        );
        let structural = 2 * cols * rows - cols - rows;
        let shear = 2 * (cols - 1) * (rows - 1);
        let bend = (cols - 2) * rows + cols * (rows - 2);
        assert_eq!(body.particle_count(), cols * rows);
        assert_eq!(body.constraint_count(), structural + shear + bend);
        assert_eq!(
            body.particles.iter().filter(|p| p.is_pinned()).count(),
            cols,
            "whole top row pinned"
        );
    }

    /// Degenerate rows (self-links, bad indices, coincident particles) are
    /// skipped, never panicking or producing NaN.
    #[test]
    fn degenerate_rows_are_skipped() {
        let mut body = SoftBody {
            particles: vec![
                Particle::new(Vec3::ZERO, 1.0),
                Particle::new(Vec3::ZERO, 1.0),
            ],
            constraints: vec![
                DeformConstraint {
                    a: 0,
                    b: 0,
                    rest: 1.0,
                    compliance: 0.0,
                    kind: DeformKind::Structural,
                    lambda: 0.0,
                },
                DeformConstraint {
                    a: 0,
                    b: 7,
                    rest: 1.0,
                    compliance: 0.0,
                    kind: DeformKind::Structural,
                    lambda: 0.0,
                },
                DeformConstraint {
                    a: 0,
                    b: 1,
                    rest: 1.0,
                    compliance: 0.0,
                    kind: DeformKind::Structural,
                    lambda: 0.0,
                },
            ],
        };
        body.integrate(1.0 / 1200.0, Vec3::new(0.0, -9.81, 0.0));
        body.solve_constraints(1.0 / 1200.0);
        body.update_velocities(1.0 / 1200.0);
        for p in &body.particles {
            assert!(p.position.is_finite() && p.velocity.is_finite());
        }
    }
}
