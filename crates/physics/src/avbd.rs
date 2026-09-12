//! AVBD rigid-body engine: a second [`crate::engine::PhysicsEngine`]
//! implementation (M1, Genesis-style engine-level modularity).
//!
//! Ports the Augmented Vertex Block Descent update rules from
//! savant117/avbd-demo3d (MIT, Giles et al, SIGGRAPH'25): per-body 6x6 SPD
//! assembly (linear + angular + cross blocks) solved with dense LDL, plus a
//! per-constraint dual (lambda/penalty) update. Contacts use the Taylor
//! constraint `C = C0*(1-alpha) + J*dq` with push-only normal clamp and a
//! joint friction-cone clamp; joints are infinite-stiffness equalities of
//! the form `C = live - alpha*C0`.
//!
//! Validated by spike `spikes/001-avbd-stack` (4-box stack stands 300 steps,
//! rest heights +-6mm, settle on step 11). Spike lessons structural here:
//! dual rules are ported, never guessed; orientation uses the official exact
//! quaternion operators (naive xyz-add + linear-diff destabilizes levers);
//! joints carry the official geometric-stiffness term (without it ball
//! joints spin up through long levers).
//!
//! # M1 scope (deliberate gaps, all documented)
//!
//! - Shapes: all pairs query through `crate::distance::shape_distance`;
//!   face-stable multi-point manifolds are built for box-involved pairs
//!   (witness is fallback only — it slides and rocks stacks); everything
//!   else solves on the witness pair (functional but tippy).
//! - Contact frames are persistent per pair (witness normals flip sign at
//!   first touch; a flipped frame turns compressive lambda tensile).
//! - Anchors follow the official merge: frozen while static grip holds
//!   (`stick`), refreshed to live geometry on slide — but never while
//!   penetrating (support first) and never for resting pairs (empty shells
//!   beyond `GEN_MARGIN`, no stale push, no lambda burn).
//! - Friction is isotropic (`mu = sqrt(muA*muB)`); anisotropic/rolling/
//!   torsional friction stays on the builtin engine (M2).
//! - Joints: [`crate::joint::JointKind::Ball`] and free
//!   [`crate::joint::JointKind::Revolute`] (hinge axis enforced via two
//!   angular rows); limits/motors and Prismatic/Fixed/Distance/Wheel return
//!   `None`. Fracture is not implemented. Penalty joints show ~10cm dynamic
//!   stretch at swing bottom (bounded, documented in tests).
//! - Sphere-on-sphere stacking is an M1 gap: single-point contact plus free
//!   rotation needs rolling multi-point contact (M2). Spheres rest and
//!   settle on floors and boxes.
//! - No CCD, substeps, islands or sleeping: one implicit step of 10
//!   iterations (M2). Broadphase is an O(n2) bounding-sphere prefilter.
//!   High-speed impacts (5+ m/s) catch deep (~3cm) — exact TOI is M2.
//! - Orientation integrates and differentiates with the official exact
//!   quaternion operators (`normalize(q + quat(w)*q/2)`,
//!   `2*(q*q0^-1).xyz`); per-step renormalization prevents long-term drift
//!   (the demo never renormalizes).

use glam::{Mat3, Quat, Vec3};
use std::collections::BTreeSet;

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::distance::{ShapeRef, cast_shape, shape_distance};
use crate::engine::{PhysicsEngine, raycast_shape_hit};
use crate::joint::{JointHandle, JointKind};
use crate::math::{Ray, RaycastHit};
use crate::shape::Shape;
use crate::trigger::{
    CONTACT_BEGIN_SLOP, CONTACT_HIT_THRESHOLD, ContactEvent, ContactEventKind, TriggerEvent,
    TriggerEventKind,
};

/// Solver iterations per step (official default).
const ITERS: usize = 10;
/// Stabilization: only `(1-alpha)` of the step-start violation enters `C`.
const ALPHA: f32 = 0.99;
/// Warmstart decay for duals and penalties (Eq. 19).
const GAMMA: f32 = 0.999;
/// Additive penalty ramp scale (Eq. 16).
const BETA: f32 = 10000.0;
/// Speculative contact margin folded into the normal `C0`.
const MARGIN: f32 = 0.01;
/// Penalty clamp range (official `PENALTY_MIN/MAX`).
const PENALTY_MIN: f32 = 1.0;
const PENALTY_MAX: f32 = 1.0e10;
/// Penalty of a freshly created contact row (official: clamped up from 0).
const PENALTY_INIT: f32 = 1.0;
/// Penalty of a freshly created joint row (official: constructed at 0,
/// Eq. 19 clamps up to `PENALTY_MIN`; stiffness comes from lambda first).
const JOINT_PENALTY_INIT: f32 = 1.0;
/// Pair creation distance: witness gap below this opens a contact pair.
const GEN_MARGIN: f32 = 0.05;
/// Box-corner manifold expansion around the witness plane.
const EXPAND_SLOP: f32 = 0.005;
/// Contact-point dedup distance when expanding manifolds.
const POINT_MATCH_DIST: f32 = 0.03;
/// Dual-memory match distance for persistent points across steps.
const LAMBDA_MATCH_DIST: f32 = 0.05;
/// Constraint satisfaction tolerance: a row with `|C|` below this is
/// exactly satisfied — no force, no dual memory, no penalty ramp. Positions
/// are O(1) in f32 (eps ~1.2e-7), so anything smaller is rounding dust, and
/// stamping stiffness for dust lets idle rows (via long levers) ratchet
/// lambda and penalty into a slow runaway.
const C_EPS: f32 = 1e-7;

/// Static-friction position threshold (official `STICK_THRESH`): a point
/// whose tangential violation is below this kept its grip last step.
const STICK_THRESH: f32 = 0.00001;

/// Cap on contact points per pair (official manifold holds 8).
const MAX_POINTS: usize = 8;

/// Dense LDL (no pivoting) for a 6x6 SPD system. Returns `None` on breakdown.
fn solve_6x6(lhs: [[f32; 6]; 6], rhs: [f32; 6]) -> Option<[f32; 6]> {
    let mut l = [[0.0f32; 6]; 6];
    let mut d = [0.0f32; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut s = lhs[i][j];
            for k in 0..j {
                s -= l[i][k] * d[k] * l[j][k];
            }
            if i == j {
                if s <= 1e-12 {
                    return None;
                }
                d[i] = s;
                l[i][i] = 1.0;
            } else {
                l[i][j] = s / d[j];
            }
        }
    }
    let mut y = [0.0f32; 6];
    for i in 0..6 {
        let mut s = rhs[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s;
    }
    let mut z = [0.0f32; 6];
    for i in 0..6 {
        z[i] = y[i] / d[i];
    }
    let mut x = [0.0f32; 6];
    for i in (0..6).rev() {
        let mut s = z[i];
        for k in (i + 1)..6 {
            s -= l[k][i] * x[k];
        }
        x[i] = s;
    }
    Some(x)
}

/// Outer product `a (x) b` as a row-major 3x3.
fn outer(a: Vec3, b: Vec3) -> [[f32; 3]; 3] {
    let av = [a.x, a.y, a.z];
    let bv = [b.x, b.y, b.z];
    [
        [av[0] * bv[0], av[0] * bv[1], av[0] * bv[2]],
        [av[1] * bv[0], av[1] * bv[1], av[1] * bv[2]],
        [av[2] * bv[0], av[2] * bv[1], av[2] * bv[2]],
    ]
}

/// Geometric stiffness of a ball-socket anchor (official
/// `geometricStiffnessBallSocket`): the Newton correction for how the lever
/// arm itself rotates. Skipping it (as a "second-order term") leaves the
/// truncated Hessian pointing the wrong way for follower forces through
/// long levers — the joint spins up exponentially. Row-major like theirs.
fn geometric_stiffness_ball_socket(k: usize, v: Vec3) -> [[f32; 3]; 3] {
    let arr = v.to_array();
    let mut m = [[0.0f32; 3]; 3];
    m[0][0] = -arr[k];
    m[1][1] = -arr[k];
    m[2][2] = -arr[k];
    m[0][k] += arr[0];
    m[1][k] += arr[1];
    m[2][k] += arr[2];
    m
}

/// Column-length diagonal (official `diagonalize`).
fn diagonalize(m: [[f32; 3]; 3]) -> Vec3 {
    Vec3::new(
        Vec3::new(m[0][0], m[1][0], m[2][0]).length(),
        Vec3::new(m[0][1], m[1][1], m[2][1]).length(),
        Vec3::new(m[0][2], m[1][2], m[2][2]).length(),
    )
}

/// Hamilton product on raw components (row-major convention, matching the
/// official demo's `maths.h` so the rotation helpers below are exact ports).
fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

/// Proper quaternion integration (official `quat + float3`):
/// `normalize(a + quat(w)*a*0.5)`. The naive xyz-add tried earlier breaks
/// rotation kinematics and destabilizes every lever.
fn quat_integrate(q: Quat, w: Vec3) -> Quat {
    let a = q.to_array();
    let t = quat_mul([w.x, w.y, w.z, 0.0], a);
    Quat::from_xyzw(
        a[0] + t[0] * 0.5,
        a[1] + t[1] * 0.5,
        a[2] + t[2] * 0.5,
        a[3] + t[3] * 0.5,
    )
    .normalize()
}

/// Relative-rotation vector (official `quat - quat`):
/// `2*(a*inverse(b)).xyz`. Linear component differences under-report spin
/// by ~2x and mishandle large angles; this is what their Jacobians expect.
fn quat_diff_vec(a: Quat, b: Quat) -> Vec3 {
    let qa = a.to_array();
    let qb = b.to_array();
    let r = quat_mul(qa, [-qb[0], -qb[1], -qb[2], qb[3]]);
    Vec3::new(2.0 * r[0], 2.0 * r[1], 2.0 * r[2])
}

/// Deterministic tangent frame for a normal (mirror of the GPU solver's
/// basis: fixed axis choice, no exact-equality branches).
fn tangent_basis(n: Vec3) -> (Vec3, Vec3) {
    let axis = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let t1 = n.cross(axis).normalize_or(Vec3::Z);
    (t1, t1.cross(n))
}

/// World-space inertia tensor from a body-frame diagonal and orientation.
fn world_inertia(inertia: Vec3, rot: Quat) -> [[f32; 3]; 3] {
    let r = Mat3::from_quat(rot);
    let cols = [r.x_axis, r.y_axis, r.z_axis];
    let mut m = [[0.0f32; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let ca = [cols[0][a], cols[1][a], cols[2][a]];
            let cb = [cols[0][b], cols[1][b], cols[2][b]];
            m[a][b] =
                ca[0] * inertia.x * cb[0] + ca[1] * inertia.y * cb[1] + ca[2] * inertia.z * cb[2];
        }
    }
    m
}

/// Analytic inverse of a symmetric 3x3 (torque path). Falls back to the
/// guarded diagonal when near-singular.
fn inverse_symmetric(m: [[f32; 3]; 3], diag: Vec3) -> [[f32; 3]; 3] {
    let (a, b, c) = (m[0][0], m[0][1], m[0][2]);
    let (d, e, f) = (m[1][1], m[1][2], m[2][2]);
    let det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    if det.abs() < 1e-12 {
        let inv = Vec3::new(
            if diag.x > 0.0 { 1.0 / diag.x } else { 0.0 },
            if diag.y > 0.0 { 1.0 / diag.y } else { 0.0 },
            if diag.z > 0.0 { 1.0 / diag.z } else { 0.0 },
        );
        return [[inv.x, 0.0, 0.0], [0.0, inv.y, 0.0], [0.0, 0.0, inv.z]];
    }
    let id = 1.0 / det;
    [
        [
            (d * f - e * e) * id,
            (c * e - b * f) * id,
            (b * e - c * d) * id,
        ],
        [
            (c * e - b * f) * id,
            (a * f - c * c) * id,
            (b * c - a * e) * id,
        ],
        [
            (b * e - c * d) * id,
            (b * c - a * e) * id,
            (a * d - b * b) * id,
        ],
    ]
}

fn mat3_vec(m: [[f32; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

/// Bounding radius for the O(n2) prefilter (exact for the G1-G4 shapes,
/// conservative estimates elsewhere fall back to always-narrowphase).
fn bound_radius(shape: &Shape) -> f32 {
    match shape {
        Shape::Sphere { radius } => *radius,
        Shape::Box { half_extents } => half_extents.length(),
        Shape::Capsule {
            radius,
            half_height,
        } => half_height + radius,
        Shape::Cylinder {
            radius,
            half_height,
        }
        | Shape::Cone {
            radius,
            half_height,
        } => (radius * radius + half_height * half_height).sqrt(),
        Shape::ConvexHull(hull) => hull
            .vertices
            .iter()
            .map(|v| v.length())
            .fold(0.0f32, f32::max),
        Shape::Heightfield(_) | Shape::TriMesh(_) => f32::INFINITY,
    }
}

/// Local-space corners of a box (empty for every other shape).
fn box_corners(shape: &Shape) -> Vec<Vec3> {
    match shape {
        Shape::Box { half_extents: h } => {
            let mut out = Vec::with_capacity(8);
            for &sx in &[-1.0f32, 1.0] {
                for &sy in &[-1.0f32, 1.0] {
                    for &sz in &[-1.0f32, 1.0] {
                        out.push(Vec3::new(sx * h.x, sy * h.y, sz * h.z));
                    }
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Whether the pair may ever produce force (layer/mask mutual filter).
fn pair_allowed(a: &RigidBody, b: &RigidBody) -> bool {
    (a.collision_layer & b.collision_mask) != 0 && (b.collision_layer & a.collision_mask) != 0
}

/// Effective inverse mass: only true dynamics participate in the solve.
fn eff_inv_mass(b: &RigidBody) -> f32 {
    if b.body_type == BodyType::Dynamic {
        b.inv_mass
    } else {
        0.0
    }
}

/// A single contact point: material anchors on both bodies plus the dual
/// state. Anchors are body-local, fixed at detection (spike lesson).
/// Penalty is per-axis (official `penalty` float3): a shared penalty lets
/// the loaded axis stiffen the idle axes into lever instability.
#[derive(Clone, Debug)]
struct AvbdPoint {
    ra: Vec3,
    rb: Vec3,
    lam: [f32; 3],
    pen: [f32; 3],
    /// Static grip held last step (official `stick`): anchors stay frozen.
    /// Rolling/sliding points refresh anchors to live geometry (otherwise a
    /// spinning body's frozen lever orbits and pumps energy).
    stuck: bool,
}

/// Contact pair: two bodies, one normal frame, up to [`MAX_POINTS`] points.
#[derive(Clone, Debug)]
struct AvbdPair {
    a: usize,
    b: usize,
    n: Vec3,
    mu: f32,
    points: Vec<AvbdPoint>,
}

/// Equality-constraint joint (ball anchor rows + optional hinge axis rows).
/// Per-axis penalties (official float3s): sharing one penalty across rows
/// lets the loaded row drive idle rows (via long levers) unstable.
#[derive(Clone, Debug)]
struct AvbdJoint {
    a: usize,
    b: usize,
    la: Vec3,
    lb: Vec3,
    ax_a: Vec3,
    ax_b: Vec3,
    angular: bool,
    lam_l: [f32; 3],
    lam_a: [f32; 2],
    pen_l: [f32; 3],
    pen_a: [f32; 2],
}

/// AVBD rigid-body engine; see the module docs for formulation and scope.
///
/// Bodies are stored in handle order (`swap_remove` on removal, exactly like
/// [`crate::engine::BuiltinPhysicsEngine`]); contacts and joints remap the
/// same way. Single-threaded, hence deterministic by construction.
#[derive(Clone, Debug)]
pub struct AvbdEngine {
    gravity: Vec3,
    bodies: Vec<RigidBody>,
    pairs: Vec<AvbdPair>,
    joints: Vec<AvbdJoint>,
    prev_touch: BTreeSet<(usize, usize)>,
    prev_trigger: BTreeSet<(usize, usize)>,
    contact_events: Vec<ContactEvent>,
    trigger_events: Vec<TriggerEvent>,
    pos0: Vec<Vec3>,
    rot0: Vec<Quat>,
    inertial: Vec<Vec3>,
    inertial_rot: Vec<Quat>,
    pre_vel: Vec<Vec3>,
}

impl AvbdEngine {
    /// Empty engine; `gravity` is a constant world-space acceleration
    /// applied to dynamic bodies each step.
    pub fn new(gravity: Vec3) -> Self {
        Self {
            gravity,
            bodies: Vec::new(),
            pairs: Vec::new(),
            joints: Vec::new(),
            prev_touch: BTreeSet::new(),
            prev_trigger: BTreeSet::new(),
            contact_events: Vec::new(),
            trigger_events: Vec::new(),
            pos0: Vec::new(),
            rot0: Vec::new(),
            inertial: Vec::new(),
            inertial_rot: Vec::new(),
            pre_vel: Vec::new(),
        }
    }

    /// Number of registered bodies.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    fn solvable(&self, h: usize) -> bool {
        self.bodies[h].body_type == BodyType::Dynamic && self.bodies[h].inv_mass > 0.0
    }

    fn ensure_scratch(&mut self) {
        let n = self.bodies.len();
        self.pos0.resize(n, Vec3::ZERO);
        self.rot0.resize(n, Quat::IDENTITY);
        self.inertial.resize(n, Vec3::ZERO);
        self.inertial_rot.resize(n, Quat::IDENTITY);
        self.pre_vel.resize(n, Vec3::ZERO);
    }

    /// Generate/persist contact pairs for this step. Returns the set of
    /// trigger-overlapping pairs (canonical order).
    fn generate_pairs(&mut self) -> BTreeSet<(usize, usize)> {
        let n = self.bodies.len();
        let mut seen = vec![false; self.pairs.len()];
        let mut trigger_now = BTreeSet::new();
        for ia in 0..n {
            for ib in (ia + 1)..n {
                let (a, b) = (&self.bodies[ia], &self.bodies[ib]);
                if !pair_allowed(a, b) {
                    continue;
                }
                let ra = bound_radius(&a.shape);
                let rb = bound_radius(&b.shape);
                if (a.position - b.position).length() > ra + rb + GEN_MARGIN {
                    continue;
                }
                let trigger = a.is_trigger || b.is_trigger;
                let d = shape_distance(
                    ShapeRef {
                        shape: &a.shape,
                        pos: a.position,
                        rot: a.orientation,
                    },
                    ShapeRef {
                        shape: &b.shape,
                        pos: b.position,
                        rot: b.orientation,
                    },
                );
                if trigger {
                    if d.dist <= 0.0 {
                        trigger_now.insert((ia, ib));
                    }
                    continue;
                }
                let pair_exists = self.pairs.iter().any(|p| p.a == ia && p.b == ib);
                if d.dist > GEN_MARGIN && !pair_exists {
                    // Creation gate: new pairs form near touch.
                    continue;
                }
                // Separated live pairs keep an EMPTY shell (official: manifold
                // persists while spheres overlap, zero contacts when apart).
                // Keeping stale points would push bodies apart with dead
                // lambda (slow levitation); dropping the pair would burn
                // warm duals and reload every re-touch (limit cycle).
                let separated = d.dist > GEN_MARGIN;
                // Anchor refresh (official `!stick` rule) is additionally
                // gated on separation: a penetrating rolling contact must
                // keep support (frozen C0), refreshing it re-coincides C0 to
                // +margin and leaves only velocity damping -> slow sink.
                let allow_refresh = d.dist > 0.0;
                let mut normal = d.point_a - d.point_b;
                if normal.length_squared() < 1e-16 {
                    normal = a.position - b.position;
                }
                let mut normal = normal.normalize_or(Vec3::Y);
                // Persistent normal per pair: witness directions flip sign at
                // first touch (separated vs penetrating closest points), which
                // would turn a compressive lambda tensile and catapult the
                // bodies. New pairs orient by body centers (B -> A); live
                // pairs keep their stored direction (spike/official lesson:
                // the contact frame must be stable, like collide() face
                // normals, not a live witness direction).
                if let Some(old) = self.pairs.iter().find(|p| p.a == ia && p.b == ib) {
                    if normal.dot(old.n) < 0.0 {
                        normal = -normal;
                    }
                } else if normal.dot(a.position - b.position) < 0.0 {
                    normal = -normal;
                }
                let mu = (a.friction.max(0.0) * b.friction.max(0.0)).sqrt();
                // Witness point plus box-corner expansion for face stability.
                // Every point is a coincident pair on the witness plane: both
                // anchors are material points that start at the same world
                // position (official rA/rB semantics). Projecting a foreign
                // corner into the other body's frame instead would glue a
                // non-material point and pump energy (spike lesson).
                let mut fresh: Vec<(Vec3, Vec3)> = Vec::new();
                if !separated {
                    let inv_a = a.orientation.inverse();
                    let inv_b = b.orientation.inverse();
                    // Box corners first: stable material support for faces.
                    for (h, witness, sign) in [(ia, d.point_a, 1.0f32), (ib, d.point_b, -1.0f32)] {
                        let body = &self.bodies[h];
                        for corner in box_corners(&body.shape) {
                            let world = body.position + body.orientation * corner;
                            let along = (world - witness).dot(sign * normal);
                            if along.abs() > EXPAND_SLOP + (-d.dist).max(0.0) {
                                continue;
                            }
                            if fresh.len() >= MAX_POINTS {
                                break;
                            }
                            // Coincident plane point shared by both anchors.
                            let pw = world - sign * normal * along;
                            let ra_l = inv_a * (pw - a.position);
                            let rb_l = inv_b * (pw - b.position);
                            if fresh.iter().any(|(ea, eb)| {
                                (ea - ra_l).length() < POINT_MATCH_DIST
                                    && (eb - rb_l).length() < POINT_MATCH_DIST
                            }) {
                                continue;
                            }
                            fresh.push((ra_l, rb_l));
                        }
                    }
                } // end corner expansion.
                // Witness fallback: the closest-point pair slides across faces
                // (non-material churn that rocks stacks), so it is only used
                // when corners give fewer than 3 points (edge/vertex and
                // non-box contacts).
                if !separated && fresh.len() < 3 {
                    let inv_a = a.orientation.inverse();
                    let inv_b = b.orientation.inverse();
                    let ra_l = inv_a * (d.point_a - a.position);
                    let rb_l = inv_b * (d.point_b - b.position);
                    if !fresh.iter().any(|(ea, eb)| {
                        (ea - ra_l).length() < POINT_MATCH_DIST
                            && (eb - rb_l).length() < POINT_MATCH_DIST
                    }) {
                        fresh.push((ra_l, rb_l));
                    }
                }
                // Persist duals by matching material anchors (5cm window).
                if let Some(idx) = self.pairs.iter().position(|p| p.a == ia && p.b == ib) {
                    seen[idx] = true;
                    let pair = &mut self.pairs[idx];
                    pair.n = normal;
                    pair.mu = mu;
                    let mut next: Vec<AvbdPoint> = Vec::with_capacity(fresh.len());
                    for (ra_l, rb_l) in fresh {
                        if let Some(old) = pair
                            .points
                            .iter()
                            .find(|p| (p.ra - ra_l).length() < LAMBDA_MATCH_DIST)
                        {
                            // Official merge: matched points carry lambda/penalty;
                            // anchors stay frozen while the grip holds
                            // (`stick`), otherwise they refresh to the live
                            // coincident geometry (rolling without refresh
                            // orbits a frozen lever and pumps energy) —
                            // unless penetrating (support first).
                            let (ra, rb) = if old.stuck || !allow_refresh {
                                (old.ra, old.rb)
                            } else {
                                (ra_l, rb_l)
                            };
                            next.push(AvbdPoint {
                                ra,
                                rb,
                                lam: old.lam,
                                pen: old.pen,
                                stuck: old.stuck,
                            });
                        } else {
                            next.push(AvbdPoint {
                                ra: ra_l,
                                rb: rb_l,
                                lam: [0.0; 3],
                                pen: [PENALTY_INIT; 3],
                                stuck: true,
                            });
                        }
                    }
                    pair.points = next;
                } else {
                    seen.push(true);
                    self.pairs.push(AvbdPair {
                        a: ia,
                        b: ib,
                        n: normal,
                        mu,
                        points: fresh
                            .into_iter()
                            .map(|(ra_l, rb_l)| AvbdPoint {
                                ra: ra_l,
                                rb: rb_l,
                                lam: [0.0; 3],
                                pen: [PENALTY_INIT; 3],
                                stuck: true,
                            })
                            .collect(),
                    });
                }
            }
        }
        for idx in (0..seen.len()).rev() {
            if !seen[idx] {
                self.pairs.swap_remove(idx);
            }
        }
        trigger_now
    }

    /// Taylor row value `C = C0*(1-alpha) + u.(dA - dB)` plus the live lever
    /// arms, for constraint row direction `u` (A-side).
    fn row_c(&self, pair: &AvbdPair, pt: &AvbdPoint, u: Vec3, c0: f32) -> (f32, Vec3, Vec3) {
        let a = &self.bodies[pair.a];
        let b = &self.bodies[pair.b];
        let ra_w = a.orientation * pt.ra;
        let ra_w0 = self.rot0[pair.a] * pt.ra;
        let rb_w = b.orientation * pt.rb;
        let rb_w0 = self.rot0[pair.b] * pt.rb;
        let d_a = (a.position + ra_w) - (self.pos0[pair.a] + ra_w0)
            + u * (ra_w
                .cross(u)
                .dot(quat_diff_vec(a.orientation, self.rot0[pair.a])));
        let d_b = (b.position + rb_w) - (self.pos0[pair.b] + rb_w0)
            + u * (rb_w
                .cross(u)
                .dot(quat_diff_vec(b.orientation, self.rot0[pair.b])));
        (c0 * (1.0 - ALPHA) + u.dot(d_a - d_b), ra_w, rb_w)
    }

    /// Clamped contact force triple: push-only normal, joint cone on tangents.
    fn contact_force(cn: f32, pens: [f32; 3], lam: [f32; 3], ct: [f32; 2], mu: f32) -> [f32; 3] {
        let fn_c = (pens[0] * cn + lam[0]).min(0.0);
        let ft_raw = [pens[1] * ct[0] + lam[1], pens[2] * ct[1] + lam[2]];
        let scale = (ft_raw[0] * ft_raw[0] + ft_raw[1] * ft_raw[1]).sqrt();
        let bound = fn_c.abs() * mu;
        let k = if scale > bound && scale > 0.0 {
            bound / scale
        } else {
            1.0
        };
        [fn_c, ft_raw[0] * k, ft_raw[1] * k]
    }

    /// Stamp one constraint row for a single body side.
    #[allow(clippy::too_many_arguments)]
    fn stamp_row(
        lhs: &mut [[f32; 6]; 6],
        rhs: &mut [f32; 6],
        axis: Vec3,
        pen: f32,
        f: f32,
        r: Vec3,
        sign: f32,
    ) {
        let nn = sign * axis;
        let t = r.cross(nn);
        let o_nn = outer(nn, nn);
        let o_tt = outer(t, t);
        let o_nt = outer(nn, t);
        for x in 0..3 {
            for y in 0..3 {
                lhs[x][y] += pen * o_nn[x][y];
                lhs[3 + x][3 + y] += pen * o_tt[x][y];
                lhs[x][3 + y] += pen * o_nt[x][y];
                lhs[3 + x][y] += pen * o_nt[y][x];
            }
        }
        rhs[0] += f * nn.x;
        rhs[1] += f * nn.y;
        rhs[2] += f * nn.z;
        rhs[3] += f * t.x;
        rhs[4] += f * t.y;
        rhs[5] += f * t.z;
    }

    /// Solve one body against all its pairs and joints (official primal).
    fn solve_body(&mut self, h: usize) {
        let m_dt2 = 1.0 / eff_inv_mass(&self.bodies[h]) / (DT_STEP * DT_STEP);
        let iw = world_inertia(self.bodies[h].inertia, self.bodies[h].orientation);
        let mut lhs = [[0.0f32; 6]; 6];
        for (i, row) in lhs.iter_mut().enumerate().take(3) {
            row[i] = m_dt2;
        }
        for a in 0..3 {
            for b in 0..3 {
                lhs[3 + a][3 + b] = iw[a][b] / (DT_STEP * DT_STEP);
            }
        }
        let mut rhs = [0.0f32; 6];
        let rl = m_dt2 * (self.bodies[h].position - self.inertial[h]);
        rhs[0] = rl.x;
        rhs[1] = rl.y;
        rhs[2] = rl.z;
        let ra = mat3_vec(
            iw,
            quat_diff_vec(self.bodies[h].orientation, self.inertial_rot[h]),
        ) / (DT_STEP * DT_STEP);
        rhs[3] = ra.x;
        rhs[4] = ra.y;
        rhs[5] = ra.z;

        // Contact rows.
        for pi in 0..self.pairs.len() {
            let (is_a, is_b) = {
                let p = &self.pairs[pi];
                (p.a == h, p.b == h)
            };
            if !is_a && !is_b {
                continue;
            }
            let sign = if is_a { 1.0 } else { -1.0 };
            for qi in 0..self.pairs[pi].points.len() {
                let (n, mu, pt) = {
                    let p = &self.pairs[pi];
                    (p.n, p.mu, p.points[qi].clone())
                };
                let (t1, t2) = tangent_basis(n);
                let (cn, r_up, r_lo) = self.row_c(&self.pairs[pi], &pt, n, self.gap_c0(pi, qi));
                let (ct1, _, _) = self.row_c(&self.pairs[pi], &pt, t1, 0.0);
                let (ct2, _, _) = self.row_c(&self.pairs[pi], &pt, t2, 0.0);
                let f = Self::contact_force(cn, pt.pen, pt.lam, [ct1, ct2], mu);
                let r_side = if is_a { r_up } else { r_lo };
                for (row, (axis, cv, fv)) in [(n, cn, f[0]), (t1, ct1, f[1]), (t2, ct2, f[2])]
                    .into_iter()
                    .enumerate()
                {
                    // Dust guard: an exactly-satisfied row stamps nothing.
                    if cv.abs() < C_EPS {
                        continue;
                    }
                    let pen = pt.pen[row];
                    Self::stamp_row(&mut lhs, &mut rhs, axis, pen, fv, r_side, sign);
                }
            }
        }

        // Joint rows (equalities: full live violation, unclamped).
        for ji in 0..self.joints.len() {
            let (is_a, is_b) = {
                let j = &self.joints[ji];
                (j.a == h, j.b == h)
            };
            if !is_a && !is_b {
                continue;
            }
            let sign = if is_a { 1.0 } else { -1.0 };
            let j = self.joints[ji].clone();
            let a = &self.bodies[j.a];
            let b = &self.bodies[j.b];
            let pa = a.position + a.orientation * j.la;
            let pb = b.position + b.orientation * j.lb;
            let pa0 = self.pos0[j.a] + self.rot0[j.a] * j.la;
            let pb0 = self.pos0[j.b] + self.rot0[j.b] * j.lb;
            let live: Vec3 = pa - pb;
            let c0v: Vec3 = pa0 - pb0;
            for k in 0..3 {
                let axis = Vec3::from_array({
                    let mut arr = [0.0f32; 3];
                    arr[k] = 1.0;
                    arr
                });
                let c = live[k] - ALPHA * c0v[k];
                if c.abs() < C_EPS {
                    continue;
                }
                let f = j.pen_l[k] * c + j.lam_l[k];
                let r_side = if is_a {
                    a.orientation * j.la
                } else {
                    b.orientation * j.lb
                };
                Self::stamp_row(&mut lhs, &mut rhs, axis, j.pen_l[k], f, r_side, sign);
            }
            // Geometric stiffness (official): the lever rotates with the
            // body, and the truncated Hessian must know. Without this the
            // joint spins up through long levers (see module docs).
            {
                let r = if is_a {
                    a.orientation * j.la
                } else {
                    -(b.orientation * j.lb)
                };
                let f = Vec3::new(
                    j.pen_l[0] * (live[0] - ALPHA * c0v[0]) + j.lam_l[0],
                    j.pen_l[1] * (live[1] - ALPHA * c0v[1]) + j.lam_l[1],
                    j.pen_l[2] * (live[2] - ALPHA * c0v[2]) + j.lam_l[2],
                );
                let fa = f.to_array();
                let mut h_mat = [[0.0f32; 3]; 3];
                for (k, fk) in fa.iter().enumerate() {
                    let g = geometric_stiffness_ball_socket(k, r);
                    for x in 0..3 {
                        for y in 0..3 {
                            h_mat[x][y] += g[x][y] * fk;
                        }
                    }
                }
                let hd = diagonalize(h_mat);
                lhs[3][3] += hd.x;
                lhs[4][4] += hd.y;
                lhs[5][5] += hd.z;
            }
            if j.angular {
                let axa = a.orientation * j.ax_a;
                let axb = b.orientation * j.ax_b;
                let axa0 = self.rot0[j.a] * j.ax_a;
                let axb0 = self.rot0[j.b] * j.ax_b;
                let (t1, t2) = tangent_basis(axa0);
                for (t, lam, pen) in [(t1, j.lam_a[0], j.pen_a[0]), (t2, j.lam_a[1], j.pen_a[1])] {
                    let live_c = t.dot(axa - axb);
                    let c0_c = t.dot(axa0 - axb0);
                    let c = live_c - ALPHA * c0_c;
                    if c.abs() < C_EPS {
                        continue;
                    }
                    let f = pen * c + lam;
                    // Rotation-only rows: torque arms about the hinge axes.
                    let g_ang = if is_a { axa.cross(t) } else { -(axb.cross(t)) };
                    let o = outer(g_ang, g_ang);
                    for x in 0..3 {
                        for y in 0..3 {
                            lhs[3 + x][3 + y] += pen * o[x][y];
                        }
                    }
                    rhs[3] += f * g_ang.x;
                    rhs[4] += f * g_ang.y;
                    rhs[5] += f * g_ang.z;
                }
            }
        }

        let neg = [-rhs[0], -rhs[1], -rhs[2], -rhs[3], -rhs[4], -rhs[5]];
        if let Some(dx) = solve_6x6(lhs, neg) {
            let body = &mut self.bodies[h];
            body.position += Vec3::new(dx[0], dx[1], dx[2]);
            body.orientation = quat_integrate(body.orientation, Vec3::new(dx[3], dx[4], dx[5]));
        }
    }

    /// Step-start normal violation of a point (gap + margin).
    fn gap_c0(&self, pi: usize, qi: usize) -> f32 {
        let p = &self.pairs[pi];
        let pt = &p.points[qi];
        let a = &self.bodies[p.a];
        let b = &self.bodies[p.b];
        let pa = self.pos0[p.a] + self.rot0[p.a] * pt.ra;
        let pb = self.pos0[p.b] + self.rot0[p.b] * pt.rb;
        let _ = (a, b);
        p.n.dot(pa - pb) + MARGIN
    }

    /// Dual update for every pair and joint (official dual core).
    fn dual_update(&mut self) {
        for pi in 0..self.pairs.len() {
            let (n, mu) = {
                let p = &self.pairs[pi];
                (p.n, p.mu)
            };
            let (t1, t2) = tangent_basis(n);
            for qi in 0..self.pairs[pi].points.len() {
                let c0 = self.gap_c0(pi, qi);
                let (cn, _, _) = self.row_c(&self.pairs[pi], &self.pairs[pi].points[qi], n, c0);
                let (ct1, _, _) = self.row_c(&self.pairs[pi], &self.pairs[pi].points[qi], t1, 0.0);
                let (ct2, _, _) = self.row_c(&self.pairs[pi], &self.pairs[pi].points[qi], t2, 0.0);
                let (pen, lam) = {
                    let pt = &self.pairs[pi].points[qi];
                    (pt.pen, pt.lam)
                };
                let f = Self::contact_force(cn, pen, lam, [ct1, ct2], mu);
                let pt = &mut self.pairs[pi].points[qi];
                // Static-grip flag (official `stick`), set unconditionally.
                pt.stuck = (ct1 * ct1 + ct2 * ct2).sqrt() < STICK_THRESH;
                if cn.abs() >= C_EPS {
                    pt.lam[0] = f[0];
                    if f[0] < 0.0 {
                        pt.pen[0] = (pt.pen[0] + BETA * cn.abs()).min(PENALTY_MAX);
                    }
                }
                let t_scale = (f[1] * f[1] + f[2] * f[2]).sqrt() / (f[0].abs() * mu + 1e-12);
                if t_scale <= 1.0 {
                    if ct1.abs() >= C_EPS {
                        pt.lam[1] = f[1];
                        pt.pen[1] = (pt.pen[1] + BETA * ct1.abs()).min(PENALTY_MAX);
                    }
                    if ct2.abs() >= C_EPS {
                        pt.lam[2] = f[2];
                        pt.pen[2] = (pt.pen[2] + BETA * ct2.abs()).min(PENALTY_MAX);
                    }
                }
            }
        }
        for j in self.joints.iter_mut() {
            let a = &self.bodies[j.a];
            let b = &self.bodies[j.b];
            let pa = a.position + a.orientation * j.la;
            let pb = b.position + b.orientation * j.lb;
            let pa0 = self.pos0[j.a] + self.rot0[j.a] * j.la;
            let pb0 = self.pos0[j.b] + self.rot0[j.b] * j.lb;
            let live = pa - pb;
            let c0v = pa0 - pb0;
            for k in 0..3 {
                let c = live[k] - ALPHA * c0v[k];
                if c.abs() >= C_EPS {
                    j.lam_l[k] += j.pen_l[k] * c;
                    j.pen_l[k] = (j.pen_l[k] + BETA * c.abs()).min(PENALTY_MAX);
                }
            }
            if j.angular {
                let axa = a.orientation * j.ax_a;
                let axb = b.orientation * j.ax_b;
                let axa0 = self.rot0[j.a] * j.ax_a;
                let axb0 = self.rot0[j.b] * j.ax_b;
                let (t1, t2) = tangent_basis(axa0);
                for (t, li) in [(t1, 0), (t2, 1)] {
                    let c = t.dot(axa - axb) - ALPHA * t.dot(axa0 - axb0);
                    if c.abs() < C_EPS {
                        continue;
                    }
                    let f = j.pen_a[li] * c + j.lam_a[li];
                    j.lam_a[li] = f;
                    j.pen_a[li] = (j.pen_a[li] + BETA * c.abs()).min(PENALTY_MAX);
                }
            }
        }
    }
}

/// Fixed step used by the solver tuning (the formulation is dt-parametric
/// through the mass terms; the iteration count is tuned for 1/60).
const DT_STEP: f32 = 1.0 / 60.0;

impl PhysicsEngine for AvbdEngine {
    fn step(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let _ = dt;
        self.ensure_scratch();
        let n = self.bodies.len();
        // Consume torques into angular velocity (cleared each step, like the
        // builtin engine); gravity folds into the inertial position below.
        for h in 0..n {
            if !self.solvable(h) {
                continue;
            }
            let b = &mut self.bodies[h];
            let iw = world_inertia(b.inertia, b.orientation);
            let iw_inv = inverse_symmetric(
                iw,
                Vec3::new(
                    if b.inertia.x > 0.0 { b.inertia.x } else { 0.0 },
                    if b.inertia.y > 0.0 { b.inertia.y } else { 0.0 },
                    if b.inertia.z > 0.0 { b.inertia.z } else { 0.0 },
                ),
            );
            let torque = std::mem::replace(&mut b.torque, Vec3::ZERO);
            b.angular_velocity += mat3_vec(iw_inv, torque) * DT_STEP;
        }
        // Contacts, triggers, and pre-step velocities for hit events.
        let trigger_now = self.generate_pairs();
        for h in 0..n {
            self.pos0[h] = self.bodies[h].position;
            self.rot0[h] = self.bodies[h].orientation;
            self.pre_vel[h] = self.bodies[h].velocity;
            if !self.solvable(h) {
                self.inertial[h] = self.bodies[h].position;
                self.inertial_rot[h] = self.bodies[h].orientation;
                continue;
            }
            let b = &self.bodies[h];
            self.inertial[h] =
                b.position + b.velocity * DT_STEP + self.gravity * (DT_STEP * DT_STEP);
            self.inertial_rot[h] = quat_integrate(b.orientation, b.angular_velocity * DT_STEP);
        }
        // Eq. 19 warmstart decay.
        for pair in &mut self.pairs {
            for pt in &mut pair.points {
                pt.lam[0] *= ALPHA * GAMMA;
                pt.lam[1] *= ALPHA * GAMMA;
                pt.lam[2] *= ALPHA * GAMMA;
                for k in 0..3 {
                    pt.pen[k] = (pt.pen[k] * GAMMA).clamp(PENALTY_MIN, PENALTY_MAX);
                }
            }
        }
        for j in &mut self.joints {
            j.lam_l[0] *= ALPHA * GAMMA;
            j.lam_l[1] *= ALPHA * GAMMA;
            j.lam_l[2] *= ALPHA * GAMMA;
            j.lam_a[0] *= ALPHA * GAMMA;
            j.lam_a[1] *= ALPHA * GAMMA;
            for k in 0..3 {
                j.pen_l[k] = (j.pen_l[k] * GAMMA).clamp(PENALTY_MIN, PENALTY_MAX);
            }
            for k in 0..2 {
                j.pen_a[k] = (j.pen_a[k] * GAMMA).clamp(PENALTY_MIN, PENALTY_MAX);
            }
        }
        // Warmstarted positions (official adaptive weight is unity on free
        // fall; proper quaternion integration like the demo).
        for h in 0..n {
            if !self.solvable(h) {
                continue;
            }
            self.bodies[h].position = self.inertial[h];
            self.bodies[h].orientation = self.inertial_rot[h];
        }
        // Main loop: primal sweep in reverse handle order (official sweeps
        // its body list head-first, i.e. reverse creation order), then duals.
        for _ in 0..ITERS {
            for h in (0..n).rev() {
                if self.solvable(h) {
                    self.solve_body(h);
                }
            }
            self.dual_update();
        }
        // BDF1 velocities + orientation renormalization (deviation from the
        // demo, which never renormalizes: prevents long-term quat drift).
        for h in 0..n {
            if !self.solvable(h) {
                continue;
            }
            let b = &mut self.bodies[h];
            b.velocity = (b.position - self.pos0[h]) / DT_STEP;
            // Official relative-rotation velocity `2*(q*q0^-1).xyz`, with a
            // rest deadband (positions are O(1) f32: sub-epsilon spin is dust).
            let spin = quat_diff_vec(b.orientation, self.rot0[h]);
            b.angular_velocity = if spin.length() < 1e-9 {
                Vec3::ZERO
            } else {
                spin / DT_STEP
            };
            b.orientation = b.orientation.normalize();
        }
        self.emit_events(trigger_now);
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
        // Mirror the builtin: exiting trigger pairs report Exited.
        let mut exited: Vec<(usize, usize)> = self
            .prev_trigger
            .iter()
            .filter(|(a, b)| *a == handle || *b == handle)
            .map(|(a, b)| (*a, *b))
            .collect();
        exited.sort_unstable();
        for (a, b) in exited {
            self.trigger_events.push(TriggerEvent {
                body_a: a,
                body_b: b,
                kind: TriggerEventKind::Exited,
            });
        }
        self.bodies.swap_remove(handle);
        let map = |h: usize| if h == last { handle } else { h };
        self.pairs.retain_mut(|p| {
            if p.a == handle || p.b == handle {
                return false;
            }
            p.a = map(p.a);
            p.b = map(p.b);
            true
        });
        self.joints.retain_mut(|j| {
            if j.a == handle || j.b == handle {
                return false;
            }
            j.a = map(j.a);
            j.b = map(j.b);
            true
        });
        // Handle-keyed state is stale after the swap (builtin parity).
        self.prev_touch.clear();
        self.prev_trigger
            .retain(|(a, b)| *a != handle && *b != handle);
        let mut remapped = BTreeSet::new();
        for (a, b) in &self.prev_trigger {
            remapped.insert((map(*a).min(map(*b)), map(*a).max(map(*b))));
        }
        self.prev_trigger = remapped;
        self.contact_events.clear();
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
        if body_a >= self.bodies.len() || body_b >= self.bodies.len() || body_a == body_b {
            return None;
        }
        let (la, lb, ax_a, ax_b, angular) = match kind {
            JointKind::Ball {
                local_anchor_a,
                local_anchor_b,
            } => (local_anchor_a, local_anchor_b, Vec3::X, Vec3::X, false),
            JointKind::Revolute {
                local_anchor_a,
                local_anchor_b,
                local_axis_a,
                local_axis_b,
                limit,
                motor,
            } => {
                if limit.is_some() || motor.is_some() {
                    return None;
                }
                let mut ax_a = local_axis_a;
                let mut ax_b = local_axis_b;
                if ax_a.length_squared() < 1e-12 || ax_b.length_squared() < 1e-12 {
                    return None;
                }
                ax_a = ax_a.normalize();
                ax_b = ax_b.normalize();
                (local_anchor_a, local_anchor_b, ax_a, ax_b, true)
            }
            // M1 gap: Prismatic/Fixed/Distance/Wheel/Gear/SixDof need their
            // own row models (M2, intra-engine solver interface).
            JointKind::Prismatic { .. }
            | JointKind::Fixed { .. }
            | JointKind::Distance { .. }
            | JointKind::Wheel { .. }
            | JointKind::Gear { .. }
            | JointKind::SixDof { .. } => return None,
        };
        self.joints.push(AvbdJoint {
            a: body_a,
            b: body_b,
            la,
            lb,
            ax_a,
            ax_b,
            angular,
            lam_l: [0.0; 3],
            lam_a: [0.0; 2],
            pen_l: [JOINT_PENALTY_INIT; 3],
            pen_a: [JOINT_PENALTY_INIT; 2],
        });
        Some(self.joints.len() - 1)
    }

    fn remove_joint(&mut self, handle: JointHandle) {
        if handle < self.joints.len() {
            self.joints.swap_remove(handle);
        }
    }

    fn raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit> {
        if max_dist.is_nan() || max_dist < 0.0 || !ray.direction.is_finite() {
            return None;
        }
        let mut closest: Option<RaycastHit> = None;
        for (h, body) in self.bodies.iter().enumerate() {
            let inverse = body.orientation.inverse();
            let origin = inverse * (ray.origin - body.position);
            let direction = inverse * ray.direction;
            let Some((distance, local_normal)) =
                raycast_shape_hit(&body.shape, origin, direction, max_dist)
            else {
                continue;
            };
            let nearer = closest.as_ref().is_none_or(|c| distance < c.distance);
            if nearer {
                closest = Some(RaycastHit {
                    handle: h,
                    point: ray.point_at(distance),
                    normal: (body.orientation * local_normal).normalize_or(Vec3::Y),
                    distance,
                });
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

    fn drain_trigger_events(&mut self) -> Vec<TriggerEvent> {
        std::mem::take(&mut self.trigger_events)
    }

    fn drain_contact_events(&mut self) -> Vec<ContactEvent> {
        std::mem::take(&mut self.contact_events)
    }
}

impl AvbdEngine {
    /// Reconcile trigger and solid-contact transitions after a step.
    fn emit_events(&mut self, trigger_now: BTreeSet<(usize, usize)>) {
        for (a, b) in trigger_now.symmetric_difference(&self.prev_trigger) {
            let (a, b) = (*a, *b);
            let entered = trigger_now.contains(&(a, b));
            self.trigger_events.push(TriggerEvent {
                body_a: a,
                body_b: b,
                kind: if entered {
                    TriggerEventKind::Entered
                } else {
                    TriggerEventKind::Exited
                },
            });
        }
        self.prev_trigger = trigger_now;
        let mut touch_now = BTreeSet::new();
        let mut hits: Vec<ContactEvent> = Vec::new();
        for p in &self.pairs {
            let a = &self.bodies[p.a];
            let b = &self.bodies[p.b];
            let mut deepest = 0.0f32;
            let mut witness = (a.position + b.position) * 0.5;
            for pt in &p.points {
                let pa = a.position + a.orientation * pt.ra;
                let pb = b.position + b.orientation * pt.rb;
                let gap = p.n.dot(pa - pb);
                if -gap > deepest {
                    deepest = -gap;
                    witness = pa;
                }
            }
            if deepest > CONTACT_BEGIN_SLOP {
                touch_now.insert((p.a, p.b));
                if !self.prev_touch.contains(&(p.a, p.b)) {
                    let approach = -((self.pre_vel[p.a] - self.pre_vel[p.b]).dot(p.n));
                    if approach > CONTACT_HIT_THRESHOLD {
                        hits.push(ContactEvent {
                            body_a: p.a,
                            body_b: p.b,
                            kind: ContactEventKind::Hit {
                                point: witness,
                                normal: -p.n,
                                approach_speed: approach,
                            },
                        });
                    }
                }
            }
        }
        for (a, b) in touch_now.symmetric_difference(&self.prev_touch) {
            let (a, b) = (*a, *b);
            self.contact_events.push(ContactEvent {
                body_a: a,
                body_b: b,
                kind: if touch_now.contains(&(a, b)) {
                    ContactEventKind::Begin
                } else {
                    ContactEventKind::End
                },
            });
        }
        self.contact_events.append(&mut hits);
        self.prev_touch = touch_now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ldl_solves_identity() {
        let lhs = [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        ];
        let x = solve_6x6(lhs, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap();
        for (got, want) in x.iter().zip([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
            assert!((got - want).abs() < 1e-5);
        }
    }

    #[test]
    fn ldl_rejects_non_positive() {
        let lhs = [[0.0f32; 6]; 6];
        assert!(solve_6x6(lhs, [0.0; 6]).is_none());
    }

    #[test]
    fn tangent_basis_is_orthonormal() {
        for n in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(1.0, 2.0, 3.0).normalize(),
        ] {
            let (t1, t2) = tangent_basis(n);
            assert!((t1.dot(n)).abs() < 1e-6);
            assert!((t2.dot(n)).abs() < 1e-6);
            assert!((t1.dot(t2)).abs() < 1e-6);
            assert!((t1.length() - 1.0).abs() < 1e-6);
        }
    }
}
