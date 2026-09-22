//! Extended Position-Based Dynamics (XPBD) rigid-body engine.
//!
//! Textbook implementation of Macklin–Müller–Chentanez 2016 ("XPBD:
//! Position-Based Simulation of Compliant Constrained Dynamics", MIG'16)
//! with the rigid-body extensions from Müller et al. ("Detailed Rigid Body
//! Simulation with Extended Position Based Dynamics") and the substepping
//! regime from Macklin et al. 2019 ("Small Steps in Physics Simulation"):
//! the frame step is split into many small substeps with a single
//! constraint iteration each, which converges faster per unit of work than
//! spending the same budget on solver iterations inside one large step.
//!
//! # Method (one substep of size `h`)
//!
//! 1. Integrate: `v += h·g`, `x += h·v` (plus torque/gyroscopic angular
//!    integration); orientations advance through the exact exponential map
//!    and are renormalized.
//! 2. Solve: for every scalar constraint the XPBD Gauss–Seidel update
//!    `Δλ = (−C − α̃·λ) / (w + α̃)` with `α̃ = α/h²` is applied as
//!    `Δx = M⁻¹·∇Cᵀ·Δλ`, where `w = ∇C·M⁻¹·∇Cᵀ` is the generalized inverse
//!    mass (linear + angular terms). Contacts are inequalities (`λ ≥ 0`);
//!    joints are equalities. With `α = 0` this reduces exactly to PBD.
//! 3. Velocities are re-derived from positions (`v = (x − xₙ)/h`, angular
//!    from the quaternion delta); restitution and Coulomb friction run as a
//!    velocity pass clamped by the position-level normal force `λ/h²`.
//!
//! # Scope (deliberate, documented)
//!
//! - Discrete contacts only (no continuous collision detection): substeps
//!   shrink the tunneling window but fast/thin bodies can still tunnel.
//! - Contacts solve on the single `shape_distance` witness pair per body
//!   pair; face-stable multi-point manifolds are a non-goal here.
//! - Joints supported structurally: ball, distance, fixed, revolute and
//!   prismatic. Limits, motors and springs are NOT driven (accepted joints
//!   constrain the free axes and ignore the drive). Wheel, gear and six-DOF
//!   joints are rejected (`add_joint` returns `None`).
//! - No sleeping, no islands, single-threaded, no trigger/contact events:
//!   use [`crate::engine::SequentialImpulseEngine`] or
//!   [`crate::avbd::AvbdEngine`] when gameplay events are needed.
//! - Soft bodies ([`crate::soft::SoftBody`], PLAN B2/D1.1) step in the same
//!   substep loop (particles + distance rows only — no volume, no
//!   deformable↔rigid coupling, no render upload yet).
//! - Not wired into the [`crate::Engine`] orchestrator / [`crate::SolverKind`]
//!   switch: this engine stands alone behind [`crate::engine::PhysicsEngine`].

use std::collections::BTreeSet;

use glam::{Quat, Vec3};

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::distance::{ShapeRef, cast_shape, shape_distance};
use crate::engine::{PhysicsEngine, raycast_shape_hit};
use crate::joint::{JointHandle, JointKind, resolve_joint};
use crate::math::{Ray, RaycastHit, tangent_basis};
use crate::migration::valid_joint;
use crate::shape::Shape;
use crate::soft::{SoftBody, SoftHandle};

/// Structural joint model supported by [`XpbdEngine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XpbdJointKind {
    /// Anchor coincidence (3 positional rows).
    Ball,
    /// Ball rows plus hinge-axis alignment (spin about the axis stays free).
    Revolute,
    /// Perpendicular anchor rows plus axis alignment (slide and spin about
    /// the axis stay free).
    Prismatic,
    /// Ball rows plus full orientation lock.
    Fixed,
    /// Single distance row between the anchors (rod with ball ends).
    Distance,
}

/// Persistent joint state: structural kind plus assembly-time frames.
#[derive(Debug, Clone)]
struct XpbdJoint {
    /// First body handle.
    a: usize,
    /// Second body handle.
    b: usize,
    /// Structural model.
    kind: XpbdJointKind,
    /// Anchor in body A's local frame.
    la: Vec3,
    /// Anchor in body B's local frame.
    lb: Vec3,
    /// Hinge/slide axis in A's local frame.
    ax_a: Vec3,
    /// Hinge/slide axis in B's local frame.
    ax_b: Vec3,
    /// Rest length of a distance rod, captured at creation.
    rest_length: f32,
    /// Relative rotation `qa⁻¹·qb` at creation (fixed joints).
    q_ref: Quat,
}

/// One discrete contact for a single substep: fixed normal and body-local
/// anchors (captured at discovery), live Lagrange multiplier.
#[derive(Debug, Clone)]
struct Contact {
    /// First body handle.
    a: usize,
    /// Second body handle.
    b: usize,
    /// Contact normal from A to B.
    n: Vec3,
    /// Contact anchor in A's local frame.
    la: Vec3,
    /// Contact anchor in B's local frame.
    lb: Vec3,
    /// Accumulated normal multiplier (force estimate: `λ/h²`).
    lambda: f32,
}

/// XPBD rigid-body engine; see the module docs for formulation and scope.
///
/// Bodies live in dense handle order (`swap_remove` on removal, like the
/// other engines). Constraints are rebuilt from scratch every substep, so
/// there is no cross-step warm-start state to migrate or invalidate.
/// Soft bodies ([`crate::soft::SoftBody`]) live in a second dense registry
/// and step in the same substep loop (D1.1: particles + distance rows only).
#[derive(Debug)]
pub struct XpbdEngine {
    /// Constant world-space acceleration applied to dynamic bodies.
    gravity: Vec3,
    /// Bodies in handle order.
    bodies: Vec<RigidBody>,
    /// Live joints in handle order.
    joints: Vec<XpbdJoint>,
    /// Canonical body pairs joined by a non-fixed joint: their contacts are
    /// skipped (`collide_connected = false` parity; fixed/weld assemblies
    /// keep structural contacts).
    joint_pairs: BTreeSet<(usize, usize)>,
    /// Substeps per `step` call (Small-Steps regime).
    substeps: u32,
    /// Constraint iterations per substep (1 is the Small-Steps optimum).
    iterations: u32,
    /// Contact compliance `α` (inverse stiffness, 0 = rigid).
    contact_compliance: f32,
    /// Joint compliance `α` (inverse stiffness, 0 = rigid).
    joint_compliance: f32,
    /// Approach speed below which contacts are inelastic (m/s).
    restitution_threshold: f32,
    /// Soft bodies in handle order (PLAN B2/D1).
    soft_bodies: Vec<SoftBody>,
}

impl XpbdEngine {
    /// Empty engine with Small-Steps defaults: 20 substeps × 1 iteration,
    /// rigid contacts and joints, 1 m/s restitution threshold.
    pub fn new(gravity: Vec3) -> Self {
        Self {
            gravity,
            bodies: Vec::new(),
            joints: Vec::new(),
            joint_pairs: BTreeSet::new(),
            substeps: 20,
            iterations: 1,
            contact_compliance: 0.0,
            joint_compliance: 0.0,
            restitution_threshold: 1.0,
            soft_bodies: Vec::new(),
        }
    }

    /// Substeps per [`PhysicsEngine::step`]: the frame step is divided into
    /// this many XPBD substeps. More substeps = stiffer stacks and smaller
    /// tunneling windows at linear cost. Ignored when zero or absurd
    /// (clamped to 1..=1000).
    pub fn set_substeps(&mut self, n: u32) {
        self.substeps = n.clamp(1, 1000);
    }

    /// Constraint iterations per substep. The Small-Steps result says a
    /// fixed budget is best spent on substeps, so the default is 1;
    /// raising this stiffens individual substeps instead. Clamped to 1..=100.
    pub fn set_iterations(&mut self, n: u32) {
        self.iterations = n.clamp(1, 100);
    }

    /// Contact compliance `α` (inverse stiffness in m/N). Zero (default) is
    /// a rigid contact; positive values let contacts sink proportionally to
    /// the contact force. Must be finite; negative values are clamped to 0.
    pub fn set_contact_compliance(&mut self, alpha: f32) {
        self.contact_compliance = if alpha.is_finite() {
            alpha.max(0.0)
        } else {
            0.0
        };
    }

    /// Joint compliance `α` (inverse stiffness). Zero (default) is a rigid
    /// joint; positive values let joints stretch under load. Same guards as
    /// [`XpbdEngine::set_contact_compliance`].
    pub fn set_joint_compliance(&mut self, alpha: f32) {
        self.joint_compliance = if alpha.is_finite() {
            alpha.max(0.0)
        } else {
            0.0
        };
    }

    /// Approach speed (m/s) below which contacts are perfectly inelastic.
    /// Faster impacts bounce with the bodies' mean restitution. Must be
    /// non-negative and finite; invalid values are ignored.
    pub fn set_restitution_threshold(&mut self, threshold: f32) {
        if threshold.is_finite() && threshold >= 0.0 {
            self.restitution_threshold = threshold;
        }
    }

    /// Number of registered bodies (dense handles).
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Number of live joints (dense handles).
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    /// Register a soft body and return its handle (PLAN B2/D1).
    pub fn add_soft_body(&mut self, body: SoftBody) -> SoftHandle {
        self.soft_bodies.push(body);
        self.soft_bodies.len() - 1
    }

    /// Remove a soft body, swapping the last into its slot. Invalid
    /// handles are a no-op. No joints reference particles in D1.1, so no
    /// remap is needed beyond the swap.
    pub fn remove_soft_body(&mut self, handle: SoftHandle) {
        if handle < self.soft_bodies.len() {
            self.soft_bodies.swap_remove(handle);
        }
    }

    /// Number of registered soft bodies (dense handles).
    pub fn soft_body_count(&self) -> usize {
        self.soft_bodies.len()
    }

    /// Read-only access to a soft body, or `None` for an invalid handle.
    pub fn get_soft_body(&self, handle: SoftHandle) -> Option<&SoftBody> {
        self.soft_bodies.get(handle)
    }

    /// Mutable access to a soft body, or `None` for an invalid handle.
    /// Direct particle edits take effect at the next [`PhysicsEngine::step`].
    pub fn get_soft_body_mut(&mut self, handle: SoftHandle) -> Option<&mut SoftBody> {
        self.soft_bodies.get_mut(handle)
    }

    /// Whether the body is simulated by this engine (dynamic with mass).
    fn solvable(&self, h: usize) -> bool {
        self.bodies[h].body_type == BodyType::Dynamic && self.bodies[h].inv_mass > 0.0
    }

    /// One substep of size `h`: integrate → discover → solve → velocities.
    fn substep(&mut self, h: f32) {
        let n = self.bodies.len();
        let mut prev_pos = vec![Vec3::ZERO; n];
        let mut prev_rot = vec![Quat::IDENTITY; n];
        for (i, b) in self.bodies.iter_mut().enumerate() {
            prev_pos[i] = b.position;
            prev_rot[i] = b.orientation;
            if b.body_type != BodyType::Dynamic || b.inv_mass <= 0.0 {
                continue;
            }
            b.velocity += h * self.gravity;
            // Newton–Euler angular integration with the gyroscopic term.
            let iw = world_inertia(b.inertia, b.orientation, b.angular_velocity);
            let ang_acc = apply_inv_inertia(
                b.inertia,
                b.orientation,
                b.torque - b.angular_velocity.cross(iw),
            );
            b.angular_velocity += h * ang_acc;
            b.torque = Vec3::ZERO;
            b.position += h * b.velocity;
            b.orientation = integrate_orientation(b.orientation, b.angular_velocity, h);
        }

        let mut contacts = self.discover_contacts();
        // Time-scaled compliance: α̃ = α/h² (Macklin et al. 2016, §4).
        let alpha_c = self.contact_compliance / (h * h);
        let alpha_j = self.joint_compliance / (h * h);
        for body in &mut self.soft_bodies {
            body.integrate(h, self.gravity);
            body.begin_substep();
        }
        for _ in 0..self.iterations {
            for i in 0..contacts.len() {
                self.solve_contact(i, &mut contacts, alpha_c);
            }
            for j in 0..self.joints.len() {
                self.solve_joint(j, alpha_j);
            }
            for body in &mut self.soft_bodies {
                body.solve_constraints(h);
            }
        }
        for b in &mut self.bodies {
            b.orientation = b.orientation.normalize();
        }

        // BDF1-style velocity update from the solved positions.
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if b.body_type != BodyType::Dynamic || b.inv_mass <= 0.0 {
                continue;
            }
            b.velocity = (b.position - prev_pos[i]) / h;
            b.angular_velocity = angular_velocity_from_delta(b.orientation, prev_rot[i], h);
        }
        for body in &mut self.soft_bodies {
            body.update_velocities(h);
        }
        self.solve_velocities(&contacts, h);
    }

    /// Discrete contact discovery at the current poses: AABB prefilter plus
    /// collision filters, trigger skip and the no-collide joint set, then an
    /// exact `shape_distance` query per surviving pair.
    fn discover_contacts(&self) -> Vec<Contact> {
        let n = self.bodies.len();
        let aabbs: Vec<crate::math::AABB> = self
            .bodies
            .iter()
            .map(|b| b.shape.aabb(b.position, b.orientation))
            .collect();
        let mut out = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                let (ba, bb) = (&self.bodies[a], &self.bodies[b]);
                if !ba.can_collide_with(bb) {
                    continue;
                }
                if ba.is_trigger || bb.is_trigger {
                    continue;
                }
                // Pairs with no dynamics on either side cannot move.
                if !self.solvable(a) && !self.solvable(b) {
                    continue;
                }
                if self.joint_pairs.contains(&(a, b)) {
                    continue;
                }
                if !aabbs[a].overlaps(&aabbs[b]) {
                    continue;
                }
                let da = ShapeRef {
                    shape: &ba.shape,
                    pos: ba.position,
                    rot: ba.orientation,
                };
                let db = ShapeRef {
                    shape: &bb.shape,
                    pos: bb.position,
                    rot: bb.orientation,
                };
                let d = shape_distance(da, db);
                if !d.dist.is_finite() || d.dist > 0.0 {
                    continue;
                }
                let mut normal = d.point_b - d.point_a;
                if normal.length_squared() < 1e-12 {
                    normal = bb.position - ba.position;
                }
                let normal = normal.normalize_or(Vec3::Y);
                out.push(Contact {
                    a,
                    b,
                    n: normal,
                    la: ba.orientation.inverse() * (d.point_a - ba.position),
                    lb: bb.orientation.inverse() * (d.point_b - bb.position),
                    lambda: 0.0,
                });
            }
        }
        out
    }

    /// Position-level normal solve for one contact (inequality, `λ ≥ 0`).
    fn solve_contact(&mut self, i: usize, contacts: &mut [Contact], alpha_tilde: f32) {
        let c = &mut contacts[i];
        let (ba, bb) = pair_mut(&mut self.bodies, c.a, c.b);
        let pa = ba.position + ba.orientation * c.la;
        let pb = bb.position + bb.orientation * c.lb;
        // Signed gap: negative while penetrating, the XPBD `C(x) ≥ 0` form.
        let gap = (pa - pb).dot(c.n);
        if gap >= 0.0 && c.lambda == 0.0 {
            return;
        }
        let ra = pa - ba.position;
        let rb = pb - bb.position;
        let w = generalized_inverse_mass(ba, bb, c.n, ra, rb);
        if w <= 0.0 {
            return;
        }
        let dlambda = delta_lambda(gap, c.lambda, w, alpha_tilde);
        let next = (c.lambda + dlambda).max(0.0);
        let applied = next - c.lambda;
        c.lambda = next;
        if applied != 0.0 {
            apply_position_correction(ba, 1.0, c.n, ra, applied);
            apply_position_correction(bb, -1.0, c.n, rb, applied);
        }
    }

    /// Position-level joint solve (equalities, one `λ` per scalar row).
    fn solve_joint(&mut self, j: usize, alpha_tilde: f32) {
        let joint = self.joints[j].clone();
        match joint.kind {
            XpbdJointKind::Ball => {
                for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                    self.solve_position_row(
                        joint.a,
                        joint.b,
                        joint.la,
                        joint.lb,
                        axis,
                        0.0,
                        0.0,
                        alpha_tilde,
                    );
                }
            }
            XpbdJointKind::Distance => {
                let (ba, bb) = (&self.bodies[joint.a], &self.bodies[joint.b]);
                let pa = ba.position + ba.orientation * joint.la;
                let pb = bb.position + bb.orientation * joint.lb;
                let delta = pa - pb;
                let dist = delta.length();
                if dist < 1e-9 {
                    return;
                }
                // C = |pa − pb| − rest; λ state would need persistence
                // across iterations for exact compliant behavior — with the
                // default single iteration per substep λ starts at 0, which
                // is exactly the Small-Steps regime this engine implements.
                let w = generalized_inverse_mass(
                    ba,
                    bb,
                    delta / dist,
                    pa - ba.position,
                    pb - bb.position,
                );
                if w <= 0.0 {
                    return;
                }
                let dlambda = (joint.rest_length - dist) / (w + alpha_tilde);
                if dlambda != 0.0 {
                    let (ba, bb) = pair_mut(&mut self.bodies, joint.a, joint.b);
                    let n = delta / dist;
                    apply_position_correction(ba, 1.0, n, pa - ba.position, dlambda);
                    apply_position_correction(bb, -1.0, n, pb - bb.position, dlambda);
                }
            }
            XpbdJointKind::Fixed => {
                for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                    self.solve_position_row(
                        joint.a,
                        joint.b,
                        joint.la,
                        joint.lb,
                        axis,
                        0.0,
                        0.0,
                        alpha_tilde,
                    );
                }
                self.solve_angular_lock(&joint, alpha_tilde);
            }
            XpbdJointKind::Revolute | XpbdJointKind::Prismatic => {
                if joint.kind == XpbdJointKind::Prismatic {
                    // Anchor separation perpendicular to the slide axis.
                    let axis = {
                        let ba = &self.bodies[joint.a];
                        (ba.orientation * joint.ax_a).normalize_or(Vec3::Y)
                    };
                    let (t1, t2) = tangent_basis(axis);
                    for t in [t1, t2] {
                        self.solve_position_row(
                            joint.a,
                            joint.b,
                            joint.la,
                            joint.lb,
                            t,
                            0.0,
                            0.0,
                            alpha_tilde,
                        );
                    }
                } else {
                    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                        self.solve_position_row(
                            joint.a,
                            joint.b,
                            joint.la,
                            joint.lb,
                            axis,
                            0.0,
                            0.0,
                            alpha_tilde,
                        );
                    }
                }
                self.solve_axis_alignment(&joint, alpha_tilde);
            }
        }
    }

    /// One scalar positional row: `C = (pa − pb)·n − target`, equality.
    #[allow(clippy::too_many_arguments)]
    fn solve_position_row(
        &mut self,
        a: usize,
        b: usize,
        la: Vec3,
        lb: Vec3,
        n: Vec3,
        target: f32,
        lambda: f32,
        alpha_tilde: f32,
    ) {
        let (ba, bb) = pair_mut(&mut self.bodies, a, b);
        let pa = ba.position + ba.orientation * la;
        let pb = bb.position + bb.orientation * lb;
        let c = (pa - pb).dot(n) - target;
        let w = generalized_inverse_mass(ba, bb, n, pa - ba.position, pb - bb.position);
        if w <= 0.0 {
            return;
        }
        let dlambda = delta_lambda(c, lambda, w, alpha_tilde);
        if dlambda != 0.0 {
            apply_position_correction(ba, 1.0, n, pa - ba.position, dlambda);
            apply_position_correction(bb, -1.0, n, pb - bb.position, dlambda);
        }
    }

    /// Fixed-joint orientation lock: drive the relative-rotation vector to
    /// zero through three scalar angular rows (small-angle linearization).
    fn solve_angular_lock(&mut self, joint: &XpbdJoint, alpha_tilde: f32) {
        let theta = {
            let (ba, bb) = (&self.bodies[joint.a], &self.bodies[joint.b]);
            let mut q_err = bb.orientation * (ba.orientation * joint.q_ref).inverse();
            if q_err.w < 0.0 {
                q_err = Quat::from_xyzw(-q_err.x, -q_err.y, -q_err.z, -q_err.w);
            }
            2.0 * q_err.xyz()
        };
        if theta.length_squared() < 1e-16 {
            return;
        }
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let c = theta.dot(axis);
            if c.abs() < 1e-12 {
                continue;
            }
            let w = {
                let (ba, bb) = (&self.bodies[joint.a], &self.bodies[joint.b]);
                angular_inverse_mass(ba, bb, axis)
            };
            if w <= 0.0 {
                continue;
            }
            let dlambda = delta_lambda(c, 0.0, w, alpha_tilde);
            if dlambda != 0.0 {
                let (ba, bb) = pair_mut(&mut self.bodies, joint.a, joint.b);
                apply_angular_correction(ba, 1.0, axis, dlambda);
                apply_angular_correction(bb, -1.0, axis, dlambda);
            }
        }
    }

    /// Hinge/slide axis alignment: `a×b` with the twist about the axis
    /// projected out, so one rotational degree of freedom stays free.
    fn solve_axis_alignment(&mut self, joint: &XpbdJoint, alpha_tilde: f32) {
        let corr = {
            let (ba, bb) = (&self.bodies[joint.a], &self.bodies[joint.b]);
            let axis_a = (ba.orientation * joint.ax_a).normalize_or(Vec3::Y);
            let axis_b = (bb.orientation * joint.ax_b).normalize_or(Vec3::Y);
            let mut corr = axis_a.cross(axis_b);
            corr -= axis_a * corr.dot(axis_a);
            corr
        };
        if corr.length_squared() < 1e-16 {
            return;
        }
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let c = corr.dot(axis);
            if c.abs() < 1e-12 {
                continue;
            }
            let w = {
                let (ba, bb) = (&self.bodies[joint.a], &self.bodies[joint.b]);
                angular_inverse_mass(ba, bb, axis)
            };
            if w <= 0.0 {
                continue;
            }
            let dlambda = delta_lambda(c, 0.0, w, alpha_tilde);
            if dlambda != 0.0 {
                let (ba, bb) = pair_mut(&mut self.bodies, joint.a, joint.b);
                apply_angular_correction(ba, 1.0, axis, dlambda);
                apply_angular_correction(bb, -1.0, axis, dlambda);
            }
        }
    }

    /// Velocity pass over the substep contacts: restitution for fast
    /// approaches plus Coulomb friction clamped by the position-level
    /// normal impulse `λ/h`.
    fn solve_velocities(&mut self, contacts: &[Contact], h: f32) {
        for c in contacts {
            // Restitution first (may separate the pair); friction reads the
            // post-bounce velocities below.
            self.solve_restitution(c);
            let normal_impulse = c.lambda / h;
            if normal_impulse <= 0.0 {
                continue;
            }
            // Geometric-mean Coulomb combine (Box2D parity).
            let mu = (self.bodies[c.a].friction * self.bodies[c.b].friction).sqrt();
            if mu <= 0.0 {
                continue;
            }
            let (ba, bb) = (&self.bodies[c.a], &self.bodies[c.b]);
            let pa = ba.position + ba.orientation * c.la;
            let pb = bb.position + bb.orientation * c.lb;
            let ra = pa - ba.position;
            let rb = pb - bb.position;
            let vrel = point_velocity(ba, ra) - point_velocity(bb, rb);
            let vt = vrel - c.n * vrel.dot(c.n);
            let speed = vt.length();
            if speed < 1e-9 {
                continue;
            }
            let t = vt / speed;
            let w = generalized_inverse_mass(ba, bb, t, ra, rb);
            if w <= 0.0 {
                continue;
            }
            let max_friction = mu * normal_impulse;
            let jt = (-speed / w).clamp(-max_friction, max_friction);
            if jt != 0.0 {
                let (ba, bb) = pair_mut(&mut self.bodies, c.a, c.b);
                apply_velocity_impulse(ba, 1.0, t, ra, jt);
                apply_velocity_impulse(bb, -1.0, t, rb, jt);
            }
        }
    }

    /// One-shot bounce for a single contact when the approach speed clears
    /// the restitution threshold.
    fn solve_restitution(&mut self, c: &Contact) {
        let (ba, bb) = (&self.bodies[c.a], &self.bodies[c.b]);
        let pa = ba.position + ba.orientation * c.la;
        let pb = bb.position + bb.orientation * c.lb;
        let ra = pa - ba.position;
        let rb = pb - bb.position;
        let vn = (point_velocity(ba, ra) - point_velocity(bb, rb)).dot(c.n);
        if vn >= -self.restitution_threshold {
            return;
        }
        let w = generalized_inverse_mass(ba, bb, c.n, ra, rb);
        if w <= 0.0 {
            return;
        }
        let e = 0.5 * (ba.restitution + bb.restitution);
        let j = -(1.0 + e) * vn / w;
        if j != 0.0 {
            let (ba, bb) = pair_mut(&mut self.bodies, c.a, c.b);
            apply_velocity_impulse(ba, 1.0, c.n, ra, j);
            apply_velocity_impulse(bb, -1.0, c.n, rb, j);
        }
    }

    /// Canonical key for the no-collide joint set.
    fn pair_key(a: usize, b: usize) -> (usize, usize) {
        (a.min(b), a.max(b))
    }

    /// Rebuild the no-collide set after structural edits (fixed joints are
    /// excluded: weld assemblies keep their structural contacts).
    fn rebuild_joint_pairs(&mut self) {
        self.joint_pairs.clear();
        for j in &self.joints {
            if j.kind != XpbdJointKind::Fixed {
                self.joint_pairs.insert(Self::pair_key(j.a, j.b));
            }
        }
    }

    /// Shared raycast kernel over all bodies (local-frame query, like the
    /// sequential-impulse engine, so hits agree by construction).
    fn raycast_body(&self, ray: &Ray, handle: usize, max_dist: f32) -> Option<RaycastHit> {
        if max_dist.is_nan() || max_dist < 0.0 || !ray.direction.is_finite() {
            return None;
        }
        let body = &self.bodies[handle];
        let inverse = body.orientation.inverse();
        let origin = inverse * (ray.origin - body.position);
        let direction = inverse * ray.direction;
        let (distance, local_normal) = raycast_shape_hit(&body.shape, origin, direction, max_dist)?;
        Some(RaycastHit {
            handle,
            point: ray.point_at(distance),
            normal: (body.orientation * local_normal).normalize_or(Vec3::Y),
            distance,
        })
    }
}

impl PhysicsEngine for XpbdEngine {
    fn step(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let h = dt / self.substeps as f32;
        for _ in 0..self.substeps {
            if self.bodies.is_empty() {
                return;
            }
            self.substep(h);
        }
    }

    fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        self.bodies.push(body);
        self.bodies.len() - 1
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        if handle >= self.bodies.len() {
            return;
        }
        let last = self.bodies.len() - 1;
        self.bodies.swap_remove(handle);
        let map = |h: usize| if h == last { handle } else { h };
        self.joints.retain(|j| j.a != handle && j.b != handle);
        for j in &mut self.joints {
            j.a = map(j.a);
            j.b = map(j.b);
        }
        self.rebuild_joint_pairs();
    }

    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        self.bodies.get(handle)
    }

    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        self.bodies.get_mut(handle)
    }

    fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        kind: JointKind,
    ) -> Option<JointHandle> {
        if !valid_joint(&kind) {
            return None;
        }
        if body_a >= self.bodies.len() || body_b >= self.bodies.len() || body_a == body_b {
            return None;
        }
        let model = match kind {
            JointKind::Ball { .. } => XpbdJointKind::Ball,
            JointKind::Revolute { .. } => XpbdJointKind::Revolute,
            JointKind::Prismatic { .. } => XpbdJointKind::Prismatic,
            JointKind::Fixed { .. } => XpbdJointKind::Fixed,
            JointKind::Distance { .. } => XpbdJointKind::Distance,
            // Wheel needs a compliant suspension spring, gear couples other
            // joints and six-DOF needs per-axis configs: all rejected rather
            // than silently mis-solved.
            JointKind::Wheel { .. } | JointKind::Gear { .. } | JointKind::SixDof { .. } => {
                return None;
            }
        };
        let resolved = resolve_joint(
            &kind,
            self.bodies[body_a].position,
            self.bodies[body_a].orientation,
            self.bodies[body_b].position,
            self.bodies[body_b].orientation,
        )?;
        self.joints.push(XpbdJoint {
            a: body_a,
            b: body_b,
            kind: model,
            la: resolved.la,
            lb: resolved.lb,
            ax_a: resolved.ax_a,
            ax_b: resolved.ax_b,
            rest_length: resolved.ref_distance,
            q_ref: resolved.ref_quat,
        });
        self.rebuild_joint_pairs();
        Some(self.joints.len() - 1)
    }

    fn remove_joint(&mut self, handle: JointHandle) {
        if handle >= self.joints.len() {
            return;
        }
        self.joints.swap_remove(handle);
        self.rebuild_joint_pairs();
    }

    fn raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit> {
        let mut closest: Option<RaycastHit> = None;
        for handle in 0..self.bodies.len() {
            if let Some(hit) = self.raycast_body(&ray, handle, max_dist) {
                match &closest {
                    Some(best) if hit.distance < best.distance => closest = Some(hit),
                    None => closest = Some(hit),
                    _ => {}
                }
            }
        }
        closest
    }

    fn shapecast(&self, shape: &Shape, from: Vec3, to: Vec3) -> Option<RaycastHit> {
        let mover = ShapeRef {
            shape,
            pos: from,
            rot: Quat::IDENTITY,
        };
        let targets = self.bodies.iter().enumerate().map(|(h, b)| {
            (
                h,
                ShapeRef {
                    shape: &b.shape,
                    pos: b.position,
                    rot: b.orientation,
                },
            )
        });
        cast_shape(mover, to - from, targets).map(|h| RaycastHit {
            handle: h.handle,
            point: h.point,
            normal: h.normal,
            distance: h.t,
        })
    }
}

/// XPBD scalar update (Macklin et al. 2016, Eq. 18):
/// `Δλ = (−C − α̃·λ) / (w + α̃)` with the time-scaled compliance
/// `α̃ = α/h²` folded in by the caller. `α̃ = 0` is exactly PBD.
/// Shared with [`crate::soft`] (deformable distance rows reuse it 1:1).
pub(crate) fn delta_lambda(c: f32, lambda: f32, w_sum: f32, alpha_tilde: f32) -> f32 {
    (-c - alpha_tilde * lambda) / (w_sum + alpha_tilde)
}

/// World-space inverse inertia applied to `v`: `R·(D⁻¹·(Rᵀ·v))`.
/// Zero diagonal entries (locked/static axes) contribute nothing.
fn apply_inv_inertia(diag: Vec3, q: Quat, v: Vec3) -> Vec3 {
    let local = q.inverse() * v;
    let scaled = Vec3::new(
        if diag.x > 0.0 { local.x / diag.x } else { 0.0 },
        if diag.y > 0.0 { local.y / diag.y } else { 0.0 },
        if diag.z > 0.0 { local.z / diag.z } else { 0.0 },
    );
    q * scaled
}

/// World-space inertia applied to `v`: `R·(D·(Rᵀ·v))`.
fn world_inertia(diag: Vec3, q: Quat, v: Vec3) -> Vec3 {
    q * (diag * (q.inverse() * v))
}

/// Generalized inverse mass for a positional constraint along `n` through
/// the body-frame levers `ra`/`rb` (Müller et al., rigid-body XPBD):
/// `w = Σ (1/m + (r×n)ᵀ·I⁻¹·(r×n))`.
fn generalized_inverse_mass(a: &RigidBody, b: &RigidBody, n: Vec3, ra: Vec3, rb: Vec3) -> f32 {
    let mut w = a.inv_mass + b.inv_mass;
    let ta = ra.cross(n);
    w += ta.dot(apply_inv_inertia(a.inertia, a.orientation, ta));
    let tb = rb.cross(n);
    w += tb.dot(apply_inv_inertia(b.inertia, b.orientation, tb));
    w
}

/// Generalized inverse mass for a pure angular constraint about `n`:
/// `w = Σ nᵀ·I⁻¹·n`.
fn angular_inverse_mass(a: &RigidBody, b: &RigidBody, n: Vec3) -> f32 {
    n.dot(apply_inv_inertia(a.inertia, a.orientation, n))
        + n.dot(apply_inv_inertia(b.inertia, b.orientation, n))
}

/// Apply `Δλ` of a positional constraint to one body: translate by
/// `M⁻¹·∇Cᵀ·Δλ` and rotate by the resulting torque arm (the quaternion
/// sum is the linearized exponential map; renormalized after the solve).
fn apply_position_correction(body: &mut RigidBody, sign: f32, n: Vec3, r: Vec3, dlambda: f32) {
    if body.body_type != BodyType::Dynamic || body.inv_mass <= 0.0 {
        return;
    }
    let impulse = n * (sign * dlambda);
    body.position += impulse * body.inv_mass;
    let dtheta = apply_inv_inertia(body.inertia, body.orientation, r.cross(impulse));
    let q = body.orientation;
    let dq = Quat::from_xyzw(dtheta.x * 0.5, dtheta.y * 0.5, dtheta.z * 0.5, 0.0) * q;
    body.orientation = Quat::from_xyzw(q.x + dq.x, q.y + dq.y, q.z + dq.z, q.w + dq.w);
}

/// Apply `Δλ` of an angular constraint: pure rotation, no translation.
fn apply_angular_correction(body: &mut RigidBody, sign: f32, n: Vec3, dlambda: f32) {
    if body.body_type != BodyType::Dynamic || body.inv_mass <= 0.0 {
        return;
    }
    let dtheta = apply_inv_inertia(body.inertia, body.orientation, n * (sign * dlambda));
    let q = body.orientation;
    let dq = Quat::from_xyzw(dtheta.x * 0.5, dtheta.y * 0.5, dtheta.z * 0.5, 0.0) * q;
    body.orientation = Quat::from_xyzw(q.x + dq.x, q.y + dq.y, q.z + dq.z, q.w + dq.w);
}

/// Impulse-velocity update for one body at the lever `r`.
fn apply_velocity_impulse(body: &mut RigidBody, sign: f32, n: Vec3, r: Vec3, j: f32) {
    if body.body_type != BodyType::Dynamic || body.inv_mass <= 0.0 {
        return;
    }
    let impulse = n * (sign * j);
    body.velocity += impulse * body.inv_mass;
    body.angular_velocity += apply_inv_inertia(body.inertia, body.orientation, r.cross(impulse));
}

/// Point velocity `v + ω×r` at the world lever `r`.
fn point_velocity(body: &RigidBody, r: Vec3) -> Vec3 {
    body.velocity + body.angular_velocity.cross(r)
}

/// Exact exponential-map orientation integration, renormalized.
fn integrate_orientation(q: Quat, w: Vec3, h: f32) -> Quat {
    let half = h * 0.5;
    let dq = Quat::from_xyzw(w.x * half, w.y * half, w.z * half, 0.0) * q;
    Quat::from_xyzw(q.x + dq.x, q.y + dq.y, q.z + dq.z, q.w + dq.w).normalize()
}

/// Angular velocity from the substep rotation `q·q_prev⁻¹` (linearized
/// angle-axis recovery with a rest deadband; positions are O(1) `f32`, so
/// sub-epsilon spin is dust).
fn angular_velocity_from_delta(q: Quat, q_prev: Quat, h: f32) -> Vec3 {
    let mut dq = q * q_prev.inverse();
    if dq.w < 0.0 {
        dq = Quat::from_xyzw(-dq.x, -dq.y, -dq.z, -dq.w);
    }
    let spin = 2.0 * dq.xyz();
    if spin.length_squared() < 1e-18 {
        Vec3::ZERO
    } else {
        spin / h
    }
}

/// Two mutable body borrows by index (`a != b`, enforced by the caller —
/// joints reject self-pairs at creation and contacts pair distinct bodies).
fn pair_mut(bodies: &mut [RigidBody], a: usize, b: usize) -> (&mut RigidBody, &mut RigidBody) {
    debug_assert_ne!(a, b);
    if a < b {
        let (lo, hi) = bodies.split_at_mut(b);
        (&mut lo[a], &mut hi[0])
    } else {
        let (lo, hi) = bodies.split_at_mut(a);
        (&mut hi[0], &mut lo[b])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box dropped onto a static floor must come to rest on it: bounded
    /// penetration, near-zero velocity — the basic XPBD contact sanity check.
    #[test]
    fn box_settles_on_static_floor() {
        let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        // Floor top face at y = 0; the falling box starts fully clear of it.
        engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, -5.0, 0.0),
            Vec3::splat(5.0),
            0.0,
        ));
        let top = engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::splat(0.5),
            1.0,
        ));
        for _ in 0..180 {
            engine.step(1.0 / 60.0);
        }
        let b = engine.get_body(top).expect("box survives");
        // Floor top at y=0, box half-height 0.5: rest center ≈ 0.5.
        assert!(
            (b.position.y - 0.5).abs() < 0.05,
            "settled height {}, want ~0.5",
            b.position.y
        );
        assert!(
            b.velocity.length() < 0.25,
            "rest velocity {} too high",
            b.velocity.length()
        );
    }

    /// A ball joint must hold a pendulum bob near its anchor length: the
    /// positional rows converge under substeps instead of drifting.
    #[test]
    fn ball_joint_holds_pendulum_length() {
        let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let anchor = engine.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.1, 0.0));
        let bob = engine.add_body(RigidBody::new_sphere(Vec3::new(0.0, -2.0, 0.0), 0.2, 1.0));
        engine
            .add_joint(
                anchor,
                bob,
                JointKind::Ball {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::new(0.0, 2.0, 0.0),
                },
            )
            .expect("ball joint accepted");
        for _ in 0..120 {
            engine.step(1.0 / 60.0);
        }
        let (pa, pb) = (
            engine.get_body(anchor).expect("anchor").position,
            engine.get_body(bob).expect("bob").position,
        );
        // Anchor at origin, bob anchor 2 m above its center: coincidence
        // means the bob center hangs 2 m below the origin.
        let separation = (pb - pa).length();
        assert!(
            (separation - 2.0).abs() < 0.05,
            "anchor separation {separation}, want ~2.0"
        );
    }

    /// Positive joint compliance must visibly soften a distance rod under a
    /// static axial load compared to the rigid default: a heavy ball hangs
    /// straight below its anchor, so the rod carries `m·g` of tension and
    /// the compliant equilibrium stretch is `α·m·g`.
    #[test]
    fn compliance_softens_distance_rod() {
        fn stretch(compliance: f32) -> f32 {
            let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
            engine.set_joint_compliance(compliance);
            let a = engine.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.1, 0.0));
            let b = engine.add_body(RigidBody::new_sphere(Vec3::new(0.0, -3.0, 0.0), 0.1, 10.0));
            engine
                .add_joint(
                    a,
                    b,
                    JointKind::Distance {
                        local_anchor_a: Vec3::ZERO,
                        local_anchor_b: Vec3::ZERO,
                    },
                )
                .expect("distance joint accepted");
            for _ in 0..300 {
                engine.step(1.0 / 60.0);
            }
            let (pa, pb) = (
                engine.get_body(a).expect("a").position,
                engine.get_body(b).expect("b").position,
            );
            (pa - pb).length() - 3.0
        }
        let rigid = stretch(0.0).abs();
        let soft = stretch(2e-4).abs();
        assert!(rigid < 2e-3, "rigid rod stretch {rigid} too large");
        assert!(
            soft > 8e-3,
            "soft rod ({soft}) must stretch more than rigid ({rigid})"
        );
    }

    /// Unsupported joint models are rejected instead of mis-solved.
    #[test]
    fn unsupported_joints_return_none() {
        let mut engine = XpbdEngine::new(Vec3::ZERO);
        let a = engine.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0));
        let b = engine.add_body(RigidBody::new_sphere(Vec3::X, 0.5, 1.0));
        assert!(
            engine
                .add_joint(
                    a,
                    b,
                    JointKind::Gear {
                        joint_a: 0,
                        joint_b: 0,
                        ratio: 1.0,
                    },
                )
                .is_none(),
            "gear must be rejected"
        );
        assert_eq!(engine.joint_count(), 0);
    }

    /// A hanging chain must keep its total length under gravity (D1.1):
    /// rigid structural rows converge instead of stretching like rubber.
    #[test]
    fn chain_holds_length_under_gravity() {
        use crate::soft::SoftBody;

        let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let rope = engine.add_soft_body(SoftBody::chain(Vec3::ZERO, Vec3::NEG_Y, 6, 0.5, 1.0, 0.0));
        for _ in 0..180 {
            engine.step(1.0 / 60.0);
        }
        let body = engine.get_soft_body(rope).expect("rope survives");
        // Pinned end at the origin, 5 links of 0.5: free end hangs at ≈ −2.5.
        let end = body.particles.last().expect("nonempty").position;
        assert!(
            (end.y + 2.5).abs() < 0.1,
            "free end height {}, want ~-2.5",
            end.y
        );
        let mut worst = 0.0f32;
        for c in &body.constraints {
            let d = (body.particles[c.a].position - body.particles[c.b].position).length();
            worst = worst.max((d - c.rest).abs() / c.rest);
        }
        assert!(worst < 0.02, "max link stretch {worst}, want <2%");
        assert!(
            body.particles.iter().all(|p| p.position.is_finite()),
            "no NaN in rope"
        );
    }

    /// A top-pinned cloth grid must drape (D1.1): the free edge falls below
    /// the pins while structural stretch stays bounded — no PBD rubber.
    #[test]
    fn cloth_grid_drapes_with_bounded_stretch() {
        use crate::soft::{ClothPin, DeformKind, SoftBody};

        let mut engine = XpbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let (cols, rows, spacing) = (6, 6, 0.25);
        let sheet = engine.add_soft_body(SoftBody::cloth_grid(
            Vec3::ZERO,
            cols,
            rows,
            spacing,
            1.0,
            0.0,
            0.0,
            1e-4,
            ClothPin::TopRow,
        ));
        for _ in 0..180 {
            engine.step(1.0 / 60.0);
        }
        let body = engine.get_soft_body(sheet).expect("sheet survives");
        // Free edge (last row) must hang well below the pinned top row.
        let edge_y = body.particles[(rows - 1) * cols..]
            .iter()
            .map(|p| p.position.y)
            .fold(f32::INFINITY, f32::min);
        assert!(
            edge_y < -0.8,
            "free edge at {edge_y}, want drape below -0.8"
        );
        let mut worst = 0.0f32;
        for c in body
            .constraints
            .iter()
            .filter(|c| c.kind == DeformKind::Structural)
        {
            let d = (body.particles[c.a].position - body.particles[c.b].position).length();
            worst = worst.max((d - c.rest).abs() / c.rest);
        }
        assert!(worst < 0.08, "max structural stretch {worst}, want <8%");
        assert!(
            body.particles.iter().all(|p| p.position.is_finite()),
            "no NaN in cloth"
        );
    }
}
