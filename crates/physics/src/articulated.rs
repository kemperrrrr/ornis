//! Featherstone articulated-body solver (R6): reduced-coordinate forward
//! dynamics for jointed mechanisms via the Articulated Body Algorithm (ABA).
//!
//! Maximal-coordinate solvers (SI/AVBD/XPBD in this crate) hold chains with
//! per-substep constraint projections, which drift on long or precise
//! linkages. This engine integrates the mechanism in generalized coordinates
//! instead: joint constraints hold by construction (zero drift), and one
//! O(n) ABA pass yields the exact tree forward dynamics. It is opt-in next
//! to the maximal-coordinate default described in [`crate::joint`] (the P6
//! "do not build" verdict is overturned; maximal coordinates remain the
//! path for contact-coupled scenes): v1 solves free/loaded mechanisms
//! exactly and leaves contacts to the maximal-coordinate engines.
//!
//! # Notation (Featherstone 1987 / "Rigid Body Dynamics Algorithms", 2008)
//!
//! Spatial motion vectors are `v = [omega; v_O]` (angular first) and spatial
//! forces `f = [n_O; f]` (moment first), both in the link frame whose origin
//! is the link center of mass. `Xup(i)` is the Plücker transform mapping the
//! parent-link motion to link `i` (`E` rotation, `r` parent-origin to
//! link-origin in parent coordinates):
//!
//! - motion: `w_B = E w_A`, `v_B = E v_A + E (w_A x r)`;
//! - force (child to parent): `f_A = E' f_B`, `n_A = E' n_B + r x f_A`.
//!
//! `S_i` is the joint motion subspace (the free motion of the link origin
//! across the joint: `[s; p_pin x s]` for a hinge about unit axis `s` with
//! the joint center at child-frame `p_pin`), `c_Ji = v_i x v_Ji` the joint
//! velocity-product term, `I_i` the link spatial inertia, `p_i = v_i x*
//! (I_i v_i) - f_ext` the bias force (gravity and the consumed body torque
//! ride in `f_ext`). The three passes per step are:
//!
//! 1. outward kinematics: `v_i = Xup(i) v_parent + S_i qd_i` (+ bias `p_i`);
//! 2. inward articulated inertia: `IA_i`, `pA_i`, then for 1-DOF joints
//!    `U = IA S`, `D = S' U`, `u = tau - S' pA`, and the parent feels
//!    `X' (IA - U U'/D) X`, `X' (pA + IA_a c_J + U u/D)`;
//! 3. outward accelerations: `qdd_i = (u_i - U_i' (Xup a_parent + c_Ji))/D_i`
//!    (the floating root solves its 6x6 `IA` directly), then semi-implicit
//!    Euler on `(q, qd)` and forward kinematics back into the bodies.
//!
//! # Scope v1 (deliberate, documented)
//!
//! - Joints: revolute (1-DOF), fixed/weld (0-DOF), floating base (6-DOF
//!   root). Every other [`JointKind`](crate::joint::JointKind) —
//!   ball/spherical, prismatic, distance, rope, spring, wheel, gear,
//!   six-DOF, free motor — is rejected by [`PhysicsEngine::add_joint`] with
//!   [`JointError::Unsupported`](crate::errors::JointError::Unsupported),
//!   never silently mis-solved.
//! - Gravity, per-joint viscous damping plus a smooth Coulomb term, and
//!   actuation through the shared [`JointMotor`](crate::joint::JointMotor)
//!   drive (velocity/position/servo torque forms; legacy
//!   [`RevoluteMotor`](crate::joint::RevoluteMotor) converts via
//!   [`RevoluteMotor::as_general`](crate::joint::RevoluteMotor::as_general)).
//!   A position/servo motor on the floating root tethers the base to the
//!   pose captured when the motor was set (the fixed-vs-floating path).
//! - Joint frames are translation-only: the joint frame inherits the parent
//!   orientation, so hinge axes/anchor offsets are plain parent-frame
//!   vectors. Any mechanism can still be described (frames are free at
//!   build); a stored constant rotation is a follow-up, not a limit on
//!   expressible topologies.
//! - No contacts/collision: bodies never collide with anything (not even
//!   each other). [`PhysicsEngine::raycast`] works (exact shape queries);
//!   [`PhysicsEngine::shapecast`] reports no hit. Coupling articulated
//!   bodies with the SI contact world and cross-solver routing are explicit
//!   follow-ups — the orchestrator [`crate::Engine`] does not know this
//!   engine yet.
//! - Revolute limits are cheap hard stops (clamp + outward-velocity kill
//!   after integration), not constraint forces: fast slams can stick for a
//!   step. Proper limit rows are a follow-up.
//! - No sleeping: links integrate every step. Sleeping articulated islands
//!   are a follow-up.
//! - Integration is semi-implicit Euler with one ABA solve per
//!   [`PhysicsEngine::step`]: stiff position servos on long floppy chains
//!   whip (large distal accelerations are the exact dynamics, verified by a
//!   zero joint-equation residual), so drive them at small steps (240+ Hz
//!   for the gains in the arm test). Substepping inside `step` is a
//!   follow-up; the host picks `dt`.
//!
//! # Determinism
//!
//! Fixed link order, fixed 6x6 elimination order, no hash maps: rerunning
//! the same call sequence is bit-identical (covered by a rerun test).

use glam::{Mat3, Quat, Vec3};

use crate::body::{BodyHandle, BodyType, RigidBody};
use crate::engine::joints::quat_twist;
use crate::engine::{PhysicsEngine, raycast_shape_hit};
use crate::errors::{JointError, QueryError};
use crate::joint::{
    JointHandle, JointKind, JointMotor, MotorKind, MotorModel, ResolvedJoint, RevoluteLimit,
    RevoluteMotor, resolve_joint,
};
use crate::math::{Ray, RaycastHit};
use crate::migration::{JointReference, validate_joint, validate_motor};

/// Six-vector: indices `0..3` angular, `3..6` linear.
type S6 = [f32; 6];
/// Six-by-six matrix, row-major.
type M6 = [[f32; 6]; 6];

/// Plücker transform from frame A to frame B: `e` rotates A-coordinates
/// into B-coordinates, `r` is the A-origin to B-origin vector in A
/// coordinates (see the module docs for the motion/force formulas).
#[derive(Debug, Clone, Copy)]
struct SpatialTransform {
    /// Rotation A into B.
    e: Mat3,
    /// A-origin to B-origin, in A coordinates.
    r: Vec3,
}

/// Skew-symmetric matrix with `skew(v) * u == v.cross(u)`.
fn skew(v: Vec3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.0, v.z, -v.y),
        Vec3::new(-v.z, 0.0, v.x),
        Vec3::new(v.y, -v.x, 0.0),
    )
}

/// Angular part of a spatial vector.
fn ang(v: &S6) -> Vec3 {
    Vec3::new(v[0], v[1], v[2])
}

/// Linear part of a spatial vector.
fn lin(v: &S6) -> Vec3 {
    Vec3::new(v[3], v[4], v[5])
}

/// Pack a spatial vector from its angular/linear parts.
fn pack(angular: Vec3, linear: Vec3) -> S6 {
    [
        angular.x, angular.y, angular.z, linear.x, linear.y, linear.z,
    ]
}

/// Motion cross product `a x b` (both motion vectors).
fn cross_motion(a: &S6, b: &S6) -> S6 {
    let (wa, va) = (ang(a), lin(a));
    let (wb, vb) = (ang(b), lin(b));
    pack(wa.cross(wb), wa.cross(vb) + va.cross(wb))
}

/// Force cross product `v x* f`: motion `v` against force `f`.
fn cross_force(v: &S6, f: &S6) -> S6 {
    let (w, vl) = (ang(v), lin(v));
    let (n, fl) = (ang(f), lin(f));
    pack(w.cross(n) + vl.cross(fl), w.cross(fl))
}

/// Dot product of two six-vectors.
fn dot6(a: &S6, b: &S6) -> f32 {
    let mut s = 0.0;
    for k in 0..6 {
        s += a[k] * b[k];
    }
    s
}

/// Six-by-six matrix times six-vector.
fn mat_vec(m: &M6, v: &S6) -> S6 {
    let mut out = [0.0; 6];
    for i in 0..6 {
        let mut s = 0.0;
        for k in 0..6 {
            s += m[i][k] * v[k];
        }
        out[i] = s;
    }
    out
}

/// Six-by-six matrix product.
fn mat_mul(a: &M6, b: &M6) -> M6 {
    let mut out = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut s = 0.0;
            for k in 0..6 {
                s += a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

/// Six-by-six transpose product `A' * B`.
fn mat_t_mul(a: &M6, b: &M6) -> M6 {
    let mut out = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut s = 0.0;
            for k in 0..6 {
                s += a[k][i] * b[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

/// Plücker transform as a 6x6 motion matrix
/// `[[E, 0], [-E skew(r), E]]` (see [`SpatialTransform`]).
fn mat_of_transform(x: &SpatialTransform) -> M6 {
    let er = x.e * skew(x.r);
    let (c0, c1, c2) = (x.e.x_axis, x.e.y_axis, x.e.z_axis);
    let (d0, d1, d2) = (er.x_axis, er.y_axis, er.z_axis);
    [
        [c0.x, c1.x, c2.x, 0.0, 0.0, 0.0],
        [c0.y, c1.y, c2.y, 0.0, 0.0, 0.0],
        [c0.z, c1.z, c2.z, 0.0, 0.0, 0.0],
        [-d0.x, -d1.x, -d2.x, c0.x, c1.x, c2.x],
        [-d0.y, -d1.y, -d2.y, c0.y, c1.y, c2.y],
        [-d0.z, -d1.z, -d2.z, c0.z, c1.z, c2.z],
    ]
}

/// Apply to a motion vector (`v_B = X(A->B) v_A`).
fn apply_motion(x: &SpatialTransform, v: &S6) -> S6 {
    let (w, vl) = (ang(v), lin(v));
    let wb = x.e * w;
    pack(wb, x.e * vl + x.e * w.cross(x.r))
}

/// Apply the transposed transform to a force (`f_A = X' f_B`).
fn apply_force_t(x: &SpatialTransform, f: &S6) -> S6 {
    let (n, fl) = (ang(f), lin(f));
    let fa = x.e.transpose() * fl;
    pack(x.e.transpose() * n + x.r.cross(fa), fa)
}

/// Block-diagonal link spatial inertia at the center of mass:
/// `diag(ic, m * I3)` (the link frame sits at the COM, so the
/// parallel-axis terms vanish — see the module docs).
fn link_inertia(inertia_diag: Vec3, mass: f32) -> M6 {
    [
        [inertia_diag.x, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, inertia_diag.y, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, inertia_diag.z, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, mass, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, mass, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, mass],
    ]
}

/// Solve a symmetric 6x6 system by Gaussian elimination with partial
/// pivoting (fixed order, deterministic). Used only for the floating root;
/// 1-DOF joints divide by the scalar `D`.
fn solve6(a: &M6, b: &S6) -> S6 {
    let mut m = [[0.0; 7]; 6];
    for i in 0..6 {
        for j in 0..6 {
            m[i][j] = a[i][j];
        }
        m[i][6] = b[i];
    }
    for col in 0..6 {
        let mut piv = col;
        let mut best = m[col][col].abs();
        for row in (col + 1)..6 {
            let v = m[row][col].abs();
            if v > best {
                best = v;
                piv = row;
            }
        }
        if piv != col {
            m.swap(piv, col);
        }
        let d = m[col][col];
        if d.abs() < 1e-30 {
            continue;
        }
        for row in (col + 1)..6 {
            let f = m[row][col] / d;
            if f != 0.0 {
                for k in col..7 {
                    m[row][k] -= f * m[col][k];
                }
            }
        }
    }
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        let mut s = m[i][6];
        for k in (i + 1)..6 {
            s -= m[i][k] * x[k];
        }
        let d = m[i][i];
        x[i] = if d.abs() < 1e-30 { 0.0 } else { s / d };
    }
    x
}

/// Wrap an angle into `[-PI, PI]`.
fn wrap_pi(a: f32) -> f32 {
    (a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

/// Root attachment of a link: welded to the world or free-floating.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BaseJoint {
    /// Welded to the world: the link keeps its pose forever (parks static
    /// bodies and anchors tether comparisons).
    Fixed,
    /// Free 6-DOF base: position/quaternion state with full spatial
    /// velocity. Total momentum is conserved (up to gravity); a
    /// position/servo [`JointMotor`](crate::joint::JointMotor) via
    /// [`FeatherstoneEngine::set_joint_motor`] tethers it to the pose
    /// captured when the motor was set.
    Floating,
}

/// How a link attaches to its parent. Joint frames inherit the parent
/// orientation (translation-only frames, see the module docs); `q_off` is
/// the constant parent-to-child rotation at assembly.
#[derive(Debug, Clone, Copy)]
enum LinkJoint {
    /// Welded: no generalized coordinate. With a parent, the constant
    /// assembly offset; without one, the link parks at its pose.
    Fixed {
        /// Child position in the parent frame at assembly.
        rel_pos: Vec3,
        /// Child orientation in the parent frame at assembly.
        rel_quat: Quat,
    },
    /// Hinge about `axis` (unit, parent frame).
    Revolute {
        /// Hinge axis in the parent frame.
        axis: Vec3,
        /// Constant parent-to-child rotation at assembly (`qa0' qb0`).
        q_off: Quat,
        /// Joint center in the parent frame.
        r_pj: Vec3,
        /// Child-COM offset rotated into the assembly parent frame
        /// (`qa0' (child0 - joint)`).
        s0: Vec3,
        /// Joint motion subspace, angular part in the child frame
        /// (`q_off' axis`, constant).
        s_ang: Vec3,
        /// Joint motion subspace, linear part (`p_pin x s_ang`, constant).
        s_lin: Vec3,
        /// Twist at assembly (rad): `q` measures travel from here, so
        /// limits and position motors are assembly-relative like the
        /// maximal-coordinate engines.
        ref_angle: f32,
        /// Travel window (rad, relative to assembly).
        limit: Option<RevoluteLimit>,
        /// Spec velocity motor, generalized (from
        /// [`RevoluteMotor::as_general`](crate::joint::RevoluteMotor::as_general)).
        spec_motor: Option<JointMotor>,
    },
    /// Free base (roots only): no parent, 6-DOF state lives in the body.
    Floating,
}

/// One articulated link: its body (pose/velocity/mass state, also serving
/// the [`PhysicsEngine`] registry) plus the constant joint frames and the
/// generalized coordinates when jointed.
#[derive(Debug, Clone)]
struct Link {
    /// Pose/velocity/mass state (position is the COM, orientation the link
    /// frame — the ABA link frame by construction).
    body: RigidBody,
    /// Parent link, or `None` for a world-attached root.
    parent: Option<usize>,
    /// Attachment model.
    joint: LinkJoint,
    /// Original joint spec (migration payload, unused by the solve).
    spec: Option<JointKind>,
    /// Assembly reference (migration payload, unused by the solve).
    reference: JointReference,
    /// Generalized position (revolute: rad from assembly; else unused).
    q: f32,
    /// Generalized velocity (revolute: rad/s; else unused).
    qd: f32,
    /// Motor override from `set_joint_motor` (`None` = spec motor applies).
    /// On a floating root this must hold a position/servo tether.
    servo: Option<JointMotor>,
    /// Tether target position (world) for a floating root, captured when
    /// the tether motor was set.
    tether_pos: Vec3,
    /// Tether target orientation for a floating root, captured when the
    /// tether motor was set.
    tether_quat: Quat,
    /// Viscous damping torque per rad/s (`>= 0`).
    damping: f32,
    /// Smooth Coulomb friction torque budget (`>= 0`).
    friction: f32,
}

/// Coulomb smoothing scale (rad/s): `tanh(qd / SMOOTH)` blends static into
/// sliding friction without a discontinuity at zero.
const FRICTION_SMOOTH: f32 = 0.02;
/// Floor for the 1-DOF articulated denominator `D` (mass `> 0` keeps it
/// positive; the floor guards degenerate zero-inertia bodies).
const MIN_JOINT_INERTIA: f32 = 1e-12;

/// Featherstone articulated-body engine: standalone
/// [`PhysicsEngine`] implementation over reduced coordinates (see the
/// module docs for formulation, scope and determinism).
///
/// Build chains from ordinary bodies: `add_body` registers a root link
/// (floating when dynamic, parked when not), `add_joint` attaches a root
/// link under another link through a revolute/fixed joint (snapping the
/// child's subtree so the anchors coincide), or use
/// [`FeatherstoneEngine::add_root`] for an explicit fixed/floating base.
#[derive(Debug)]
pub struct FeatherstoneEngine {
    /// Links in handle order (`BodyHandle(i)` is link `i`).
    links: Vec<Link>,
    /// Children adjacency in index order, rebuilt on structural edits.
    children: Vec<Vec<usize>>,
    /// Outward traversal order (roots first), rebuilt on structural edits.
    order: Vec<usize>,
    /// World-space gravity acceleration (m/s^2).
    gravity: Vec3,
}

impl FeatherstoneEngine {
    /// Empty engine with the given world-space gravity acceleration.
    /// Non-finite gravity folds to zero (never a NaN source downstream).
    pub fn new(gravity: Vec3) -> Self {
        Self {
            links: Vec::new(),
            children: Vec::new(),
            order: Vec::new(),
            gravity: if gravity.is_finite() {
                gravity
            } else {
                Vec3::ZERO
            },
        }
    }

    /// Current world-space gravity acceleration.
    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    /// Replace gravity (non-finite input is ignored, the old value stays).
    pub fn set_gravity(&mut self, gravity: Vec3) {
        if gravity.is_finite() {
            self.gravity = gravity;
        }
    }

    /// Number of registered links (dense handles, like bodies elsewhere).
    pub fn body_count(&self) -> usize {
        self.links.len()
    }

    /// Number of live joints (links with a parent).
    pub fn joint_count(&self) -> usize {
        self.links.iter().filter(|l| l.parent.is_some()).count()
    }

    /// Register a root link with an explicit base. Non-dynamic bodies park
    /// regardless of `base` (infinite-mass links cannot join ABA dynamics;
    /// the pose freezes).
    pub fn add_root(&mut self, body: RigidBody, base: BaseJoint) -> BodyHandle {
        let dynamic = body.body_type == BodyType::Dynamic && body.inv_mass > 0.0;
        let joint = if dynamic {
            match base {
                BaseJoint::Fixed => LinkJoint::Fixed {
                    rel_pos: Vec3::ZERO,
                    rel_quat: Quat::IDENTITY,
                },
                BaseJoint::Floating => LinkJoint::Floating,
            }
        } else {
            LinkJoint::Fixed {
                rel_pos: Vec3::ZERO,
                rel_quat: Quat::IDENTITY,
            }
        };
        let tether_pos = body.position;
        let tether_quat = body.orientation;
        self.links.push(Link {
            body,
            parent: None,
            joint,
            spec: None,
            reference: JointReference::default(),
            q: 0.0,
            qd: 0.0,
            servo: None,
            tether_pos,
            tether_quat,
            damping: 0.0,
            friction: 0.0,
        });
        let handle = BodyHandle::from(self.links.len() - 1);
        self.rebuild_topology();
        handle
    }

    /// Generalized position of a joint (revolute: rad from assembly).
    /// `None` for invalid handles, fixed joints and floating roots.
    pub fn joint_angle(&self, handle: JointHandle) -> Option<f32> {
        let link = self.links.get(handle.index())?;
        match link.joint {
            LinkJoint::Revolute { .. } => Some(link.q),
            _ => None,
        }
    }

    /// Generalized velocity of a joint (revolute: rad/s).
    /// `None` for invalid handles, fixed joints and floating roots.
    pub fn joint_velocity(&self, handle: JointHandle) -> Option<f32> {
        let link = self.links.get(handle.index())?;
        match link.joint {
            LinkJoint::Revolute { .. } => Some(link.qd),
            _ => None,
        }
    }

    /// Generalized motor override on an assembled joint, mirroring the
    /// maximal-coordinate engines: `Some` replaces the spec motor,
    /// `None` clears the override. Revolute joints take any
    /// [`JointMotor`](crate::joint::JointMotor) kind; a floating root
    /// takes a position/servo tether to the pose captured by this call.
    ///
    /// # Errors
    ///
    /// [`JointError::UnknownRef`] for a stale handle, `Unsupported` for
    /// fixed joints, velocity tethers, or an invalid motor.
    pub fn set_joint_motor(
        &mut self,
        handle: JointHandle,
        motor: Option<JointMotor>,
    ) -> Result<(), JointError> {
        if let Some(m) = motor {
            validate_motor(&m).map_err(|_| JointError::NonFinite {
                field: "motor".to_string(),
            })?;
        }
        let Some(link) = self.links.get(handle.index()) else {
            return Err(JointError::UnknownRef {
                handle: handle.index(),
            });
        };
        let tetherable = matches!(link.joint, LinkJoint::Floating) && link.parent.is_none();
        let revolute = matches!(link.joint, LinkJoint::Revolute { .. });
        if revolute {
            self.links[handle.index()].servo = motor;
            return Ok(());
        }
        if tetherable {
            if let Some(m) = motor
                && matches!(m.kind, MotorKind::Velocity)
            {
                return Err(JointError::Unsupported {
                    detail: "floating tether needs a position or servo motor".to_string(),
                });
            }
            let (p, q) = (
                self.links[handle.index()].body.position,
                self.links[handle.index()].body.orientation,
            );
            let link = &mut self.links[handle.index()];
            link.servo = motor;
            if motor.is_some() {
                link.tether_pos = p;
                link.tether_quat = q;
            }
            return Ok(());
        }
        Err(JointError::Unsupported {
            detail: "set_joint_motor needs a revolute joint or a floating root".to_string(),
        })
    }

    /// Current motor override of a joint (`None` = spec motor applies, if
    /// any). `None` for an invalid handle.
    pub fn joint_motor(&self, handle: JointHandle) -> Option<JointMotor> {
        self.links.get(handle.index())?.servo
    }

    /// Viscous damping of a revolute joint (torque per rad/s). Negative or
    /// non-finite input clamps to zero. No-op for invalid handles and
    /// non-revolute links.
    pub fn set_joint_damping(&mut self, handle: JointHandle, damping: f32) {
        let c = if damping.is_finite() {
            damping.max(0.0)
        } else {
            0.0
        };
        if let Some(link) = self.links.get_mut(handle.index())
            && matches!(link.joint, LinkJoint::Revolute { .. })
        {
            link.damping = c;
        }
    }

    /// Smooth Coulomb friction budget of a revolute joint (N·m, blended by
    /// `tanh(qd / 0.02)`). Negative or non-finite input clamps to zero.
    /// No-op for invalid handles and non-revolute links.
    pub fn set_joint_friction(&mut self, handle: JointHandle, friction: f32) {
        let c = if friction.is_finite() {
            friction.max(0.0)
        } else {
            0.0
        };
        if let Some(link) = self.links.get_mut(handle.index())
            && matches!(link.joint, LinkJoint::Revolute { .. })
        {
            link.friction = c;
        }
    }

    /// Rebuild children adjacency and the outward order after structural
    /// edits (roots in index order, then breadth-first — deterministic).
    fn rebuild_topology(&mut self) {
        self.children.clear();
        self.children.resize(self.links.len(), Vec::new());
        for (i, link) in self.links.iter().enumerate() {
            if let Some(p) = link.parent
                && p < self.links.len()
            {
                self.children[p].push(i);
            }
        }
        self.order.clear();
        let mut queue: Vec<usize> = self
            .links
            .iter()
            .enumerate()
            .filter(|(_, l)| l.parent.is_none())
            .map(|(i, _)| i)
            .collect();
        self.order.extend(queue.iter().copied());
        let mut head = 0;
        while head < queue.len() {
            let i = queue[head];
            head += 1;
            for &c in &self.children[i] {
                self.order.push(c);
                queue.push(c);
            }
        }
    }

    /// Subtree of `root` (inclusive) in index order.
    fn subtree(&self, root: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            out.push(i);
            stack.extend(self.children[i].iter().copied());
        }
        out.sort_unstable();
        out
    }

    /// Pose of link `i`'s parent frame: the live parent pose, or the link's
    /// own pose for roots (identity forward kinematics — a defensive
    /// fallback; jointed links always have parents).
    fn parent_pose(&self, i: usize) -> (Vec3, Quat) {
        match self.links[i].parent {
            Some(p) => (self.links[p].body.position, self.links[p].body.orientation),
            None => (self.links[i].body.position, self.links[i].body.orientation),
        }
    }

    /// Forward kinematics of link `i` from its parent pose and `q`.
    fn fk_pose(&self, i: usize) -> (Vec3, Quat) {
        let link = &self.links[i];
        let (pp, qp) = self.parent_pose(i);
        match link.joint {
            LinkJoint::Fixed { rel_pos, rel_quat } => {
                if link.parent.is_none() {
                    (link.body.position, link.body.orientation)
                } else {
                    (pp + qp * rel_pos, qp * rel_quat)
                }
            }
            LinkJoint::Floating => (link.body.position, link.body.orientation),
            LinkJoint::Revolute {
                axis,
                q_off,
                r_pj,
                s0,
                ..
            } => {
                if link.parent.is_none() {
                    (link.body.position, link.body.orientation)
                } else {
                    let r = Quat::from_axis_angle(axis, link.q);
                    (pp + qp * (r_pj + r * s0), qp * r * q_off)
                }
            }
        }
    }

    /// Plücker parent-to-link transform at the current `q`.
    fn xup(&self, i: usize) -> SpatialTransform {
        let link = &self.links[i];
        match link.joint {
            LinkJoint::Revolute {
                axis,
                q_off,
                r_pj,
                s0,
                ..
            } => {
                let r = Quat::from_axis_angle(axis, link.q);
                let e = Mat3::from_quat(q_off.conjugate() * Quat::from_axis_angle(axis, -link.q));
                SpatialTransform {
                    e,
                    r: r_pj + r * s0,
                }
            }
            LinkJoint::Fixed { rel_pos, rel_quat } => SpatialTransform {
                e: Mat3::from_quat(rel_quat.conjugate()),
                r: rel_pos,
            },
            LinkJoint::Floating => SpatialTransform {
                e: Mat3::from_quat(link.body.orientation.conjugate()),
                r: Vec3::ZERO,
            },
        }
    }

    /// Joint motion subspace (constant per link).
    fn joint_subspace(&self, i: usize) -> S6 {
        match self.links[i].joint {
            LinkJoint::Revolute { s_ang, s_lin, .. } => pack(s_ang, s_lin),
            _ => [0.0; 6],
        }
    }

    /// Re-sync revolute `q` from the live poses (host teleports through
    /// `get_body_mut` take effect here): raw assembly-relative twist,
    /// unwrapped against the previous `q` so continuous spinning survives.
    fn resync_coordinates(&mut self) {
        for i in 0..self.links.len() {
            let (axis, reference) = match self.links[i].joint {
                LinkJoint::Revolute {
                    axis, ref_angle, ..
                } => (axis, ref_angle),
                _ => continue,
            };
            let Some(p) = self.links[i].parent else {
                continue;
            };
            let qa = self.links[p].body.orientation;
            let qb = self.links[i].body.orientation;
            let raw = quat_twist(qa.conjugate() * qb, axis) - reference;
            let prev = self.links[i].q;
            self.links[i].q = prev + wrap_pi(raw - prev);
        }
    }

    /// One ABA solve at `(q, qd)`: returns generalized accelerations in
    /// link order (revolute scalar in `[0]`; floating root full 6D).
    fn aba(&self, h: f32) -> Vec<S6> {
        let n = self.links.len();
        let identity_x = SpatialTransform {
            e: Mat3::IDENTITY,
            r: Vec3::ZERO,
        };
        let mut xup = vec![identity_x; n];
        let mut s_sub = vec![[0.0; 6]; n];
        let mut vel = vec![[0.0; 6]; n];
        let mut c_j = vec![[0.0; 6]; n];
        let mut ia = vec![[[0.0; 6]; 6]; n];
        let mut pa = vec![[0.0; 6]; n];
        let mut u_mat = vec![[0.0; 6]; n];
        let mut den = vec![0.0; n];
        let mut uu = vec![0.0; n];
        let zero = [0.0; 6];

        // Pass 1 — outward kinematics + bias forces.
        for &i in &self.order {
            xup[i] = self.xup(i);
            s_sub[i] = self.joint_subspace(i);
            let link = &self.links[i];
            let v_parent = match link.parent {
                Some(p) => vel[p],
                None => zero,
            };
            let rot = Mat3::from_quat(link.body.orientation);
            let v = match link.joint {
                // Body velocities are world-frame; the root subspace is the
                // body frame.
                LinkJoint::Floating => pack(
                    rot.transpose() * link.body.angular_velocity,
                    rot.transpose() * link.body.velocity,
                ),
                _ => {
                    let xp = apply_motion(&xup[i], &v_parent);
                    let mut s = [0.0; 6];
                    for k in 0..6 {
                        s[k] = xp[k] + s_sub[i][k] * link.qd;
                    }
                    s
                }
            };
            vel[i] = v;
            // Joint velocity across the joint (zero for welds/roots).
            let mut vj = [0.0; 6];
            if matches!(link.joint, LinkJoint::Revolute { .. }) {
                for k in 0..6 {
                    vj[k] = s_sub[i][k] * link.qd;
                }
            }
            c_j[i] = cross_motion(&v, &vj);
            let inertia = link_inertia(link.body.inertia, link.body.mass);
            let iv = mat_vec(&inertia, &v);
            let mut p = cross_force(&v, &iv);
            // External force: gravity at the COM plus the consumed torque.
            let f_ext = pack(
                rot.transpose() * link.body.torque,
                rot.transpose() * (link.body.mass * self.gravity),
            );
            for k in 0..6 {
                p[k] -= f_ext[k];
            }
            ia[i] = inertia;
            pa[i] = p;
        }

        // Pass 2 — inward articulated inertia.
        for &i in self.order.iter().rev() {
            if matches!(self.links[i].joint, LinkJoint::Revolute { .. }) {
                let u_vec = mat_vec(&ia[i], &s_sub[i]);
                let d = dot6(&s_sub[i], &u_vec).max(MIN_JOINT_INERTIA);
                let tau = self.joint_torque(i, d, h);
                let u = tau - dot6(&s_sub[i], &pa[i]);
                // Articulated update: IA -= U U'/D.
                for r in 0..6 {
                    for c in 0..6 {
                        ia[i][r][c] -= u_vec[r] * u_vec[c] / d;
                    }
                }
                // pA += IA_a c_J + U u/D.
                let iac = mat_vec(&ia[i], &c_j[i]);
                for k in 0..6 {
                    pa[i][k] += iac[k] + u_vec[k] * (u / d);
                }
                u_mat[i] = u_vec;
                den[i] = d;
                uu[i] = u;
            }
            let Some(p) = self.links[i].parent else {
                continue;
            };
            let x = mat_of_transform(&xup[i]);
            let acc = mat_t_mul(&x, &mat_mul(&ia[i], &x));
            for r in 0..6 {
                for c in 0..6 {
                    ia[p][r][c] += acc[r][c];
                }
            }
            let pf = apply_force_t(&xup[i], &pa[i]);
            for k in 0..6 {
                pa[p][k] += pf[k];
            }
        }

        // Pass 3 — outward accelerations.
        let mut acc = vec![[0.0; 6]; n];
        let mut a_link = vec![[0.0; 6]; n];
        for &i in &self.order {
            let a_in = match self.links[i].parent {
                Some(p) => apply_motion(&xup[i], &a_link[p]),
                None => zero,
            };
            match self.links[i].joint {
                LinkJoint::Revolute { .. } => {
                    let mut ac = [0.0; 6];
                    for k in 0..6 {
                        ac[k] = a_in[k] + c_j[i][k];
                    }
                    let qdd = (uu[i] - dot6(&u_mat[i], &ac)) / den[i];
                    acc[i][0] = qdd;
                    let mut a = ac;
                    for k in 0..6 {
                        a[k] += s_sub[i][k] * qdd;
                    }
                    a_link[i] = a;
                }
                LinkJoint::Fixed { .. } => {
                    a_link[i] = a_in;
                }
                LinkJoint::Floating => {
                    let tether = self.base_tether(i, &ia[i]);
                    let ia_in = mat_vec(&ia[i], &a_in);
                    let mut u = [0.0; 6];
                    for k in 0..6 {
                        u[k] = tether[k] - pa[i][k] - ia_in[k];
                    }
                    let qdd = solve6(&ia[i], &u);
                    acc[i] = qdd;
                    let mut a = a_in;
                    for k in 0..6 {
                        a[k] += qdd[k];
                    }
                    a_link[i] = a;
                }
            }
        }
        acc
    }

    /// Joint torque for a revolute link: damping + smooth friction + the
    /// effective (override-or-spec) [`JointMotor`](crate::joint::JointMotor)
    /// drive in torque form (`D` is the articulated inertia about the
    /// joint, so `1/D` is the drive's inverse effective mass).
    fn joint_torque(&self, i: usize, d: f32, h: f32) -> f32 {
        let link = &self.links[i];
        let mut tau = -link.damping * link.qd - link.friction * (link.qd / FRICTION_SMOOTH).tanh();
        let motor = link.servo.or(match link.joint {
            LinkJoint::Revolute { spec_motor, .. } => spec_motor,
            _ => None,
        });
        if let Some(m) = motor {
            let inv_eff = 1.0 / d;
            tau += match m.kind {
                MotorKind::Velocity => {
                    if h > 0.0 {
                        m.velocity_impulse(m.target_velocity - link.qd, inv_eff, h) / h
                    } else {
                        0.0
                    }
                }
                MotorKind::Position | MotorKind::Servo => {
                    let pos_err = m.target_position - link.q;
                    let vel_err = m.target_velocity - link.qd;
                    m.servo_force(pos_err, vel_err, inv_eff)
                        .clamp(-m.max_force, m.max_force)
                }
            };
        }
        tau
    }

    /// Tether wrench (body frame) for a floating root with a
    /// position/servo motor: PD pull to the pose captured when the motor
    /// was set. Zero without a tether.
    ///
    /// Acceleration-based motors (the default) scale by the root's
    /// ARTICULATED inertia `ia` — exactly the load the base acceleration
    /// sees in pass 3 — so the tether has uniform bandwidth
    /// `sqrt(stiffness)` whatever hangs below it. (A composite
    /// subtree-inertia scaling overestimates that load whenever a free
    /// hinge decouples a child from the base motion, which made the
    /// explicit tether unstable on offset chains.) Force-based motors
    /// apply absolute gains. Force and torque are each clamped to
    /// `max_force`.
    fn base_tether(&self, i: usize, ia: &M6) -> S6 {
        let link = &self.links[i];
        let Some(m) = link.servo else {
            return [0.0; 6];
        };
        if !matches!(m.kind, MotorKind::Position | MotorKind::Servo) {
            return [0.0; 6];
        }
        let rot = Mat3::from_quat(link.body.orientation);
        let lin_err = link.tether_pos - link.body.position;
        // Attitude error: shortest-arc axis-angle of `q_ref' q`, in world.
        let dq = link.tether_quat.conjugate() * link.body.orientation;
        let dq = if dq.w < 0.0 { -dq } else { dq };
        let half = dq.w.clamp(-1.0, 1.0).acos();
        let local_err = if half < 1e-6 {
            Vec3::ZERO
        } else {
            dq.xyz() * (2.0 * half / half.sin().max(1e-9))
        };
        // `dq` is in the reference frame; express the error in world.
        let ang_err = link.tether_quat * local_err;
        // `RigidBody::angular_velocity` is world-space.
        let w_world = link.body.angular_velocity;
        let (n_world, f_world) = match m.model {
            MotorModel::ForceBased => (
                ang_err * (-m.stiffness) - w_world * m.damping,
                lin_err * m.stiffness - link.body.velocity * m.damping,
            ),
            MotorModel::AccelerationBased => {
                let alpha = ang_err * (-m.stiffness) - w_world * m.damping;
                let accel = lin_err * m.stiffness - link.body.velocity * m.damping;
                let wrench = mat_vec(ia, &pack(rot.transpose() * alpha, rot.transpose() * accel));
                (rot * ang(&wrench), rot * lin(&wrench))
            }
        };
        let n_world = clamp_vec(n_world, m.max_force);
        let f_world = clamp_vec(f_world, m.max_force);
        pack(rot.transpose() * n_world, rot.transpose() * f_world)
    }

    /// Integrate `(q, qd)` semi-implicitly and write poses/velocities back.
    fn integrate(&mut self, acc: &[S6], h: f32) {
        for (i, a) in acc.iter().enumerate() {
            match self.links[i].joint {
                LinkJoint::Revolute { limit, .. } => {
                    self.links[i].qd += h * a[0];
                    self.links[i].q += h * self.links[i].qd;
                    if let Some(lim) = limit {
                        if self.links[i].q < lim.min {
                            self.links[i].q = lim.min;
                            if self.links[i].qd < 0.0 {
                                self.links[i].qd = 0.0;
                            }
                        } else if self.links[i].q > lim.max {
                            self.links[i].q = lim.max;
                            if self.links[i].qd > 0.0 {
                                self.links[i].qd = 0.0;
                            }
                        }
                    }
                }
                LinkJoint::Fixed { .. } => {}
                LinkJoint::Floating => {
                    let rot = Mat3::from_quat(self.links[i].body.orientation);
                    let w = self.links[i].body.angular_velocity + h * (rot * ang(a));
                    let v = self.links[i].body.velocity + h * (rot * lin(a));
                    self.links[i].body.angular_velocity = w;
                    self.links[i].body.velocity = v;
                    self.links[i].body.position += h * v;
                    self.links[i].body.orientation =
                        integrate_quat(self.links[i].body.orientation, w, h);
                }
            }
        }
        // Forward kinematics into poses.
        let order = self.order.clone();
        for &i in &order {
            if matches!(
                self.links[i].joint,
                LinkJoint::Fixed { .. } | LinkJoint::Revolute { .. }
            ) && self.links[i].parent.is_some()
            {
                let (p, q) = self.fk_pose(i);
                self.links[i].body.position = p;
                self.links[i].body.orientation = q;
            }
        }
        // Velocity propagation at the new state.
        let mut vel = vec![[0.0; 6]; self.links.len()];
        for &i in &order {
            let x = self.xup(i);
            let v_parent = match self.links[i].parent {
                Some(p) => vel[p],
                None => [0.0; 6],
            };
            let link = &self.links[i];
            let rot = Mat3::from_quat(link.body.orientation);
            let v = match link.joint {
                // Spatial velocities are link-frame (children propagate
                // through `xup`); the root's world velocities rotate in.
                LinkJoint::Floating => pack(
                    rot.transpose() * link.body.angular_velocity,
                    rot.transpose() * link.body.velocity,
                ),
                _ => {
                    let xp = apply_motion(&x, &v_parent);
                    let s = self.joint_subspace(i);
                    let mut out = [0.0; 6];
                    for k in 0..6 {
                        out[k] = xp[k] + s[k] * link.qd;
                    }
                    out
                }
            };
            vel[i] = v;
            // Body velocities are world-frame (like every other engine).
            self.links[i].body.angular_velocity = rot * ang(&v);
            self.links[i].body.velocity = rot * lin(&v);
        }
        // Consumed torques clear (body-torque parity with the other engines).
        for link in &mut self.links {
            link.body.torque = Vec3::ZERO;
        }
    }

    /// Admit a revolute or fixed joint between two links: resolve frames,
    /// snap the child subtree onto the parent anchor, build the constant
    /// joint frames. Returns the joint (child-side frames included).
    fn admit_joint(
        &mut self,
        ia: usize,
        ib: usize,
        kind: &JointKind,
    ) -> Result<(LinkJoint, ResolvedJoint), JointError> {
        match *kind {
            JointKind::Revolute { limit, motor, .. } => {
                let resolved = resolve_joint(
                    kind,
                    self.links[ia].body.position,
                    self.links[ia].body.orientation,
                    self.links[ib].body.position,
                    self.links[ib].body.orientation,
                )
                .ok_or_else(|| JointError::BadAxis {
                    detail: "unresolvable joint frames".to_string(),
                })?;
                if resolved.degenerate {
                    return Err(JointError::BadAxis {
                        detail: "revolute axis is degenerate".to_string(),
                    });
                }
                let qa = self.links[ia].body.orientation;
                let pa = self.links[ia].body.position;
                let qb = self.links[ib].body.orientation;
                let joint_world = pa + qa * resolved.la;
                // Snap the child subtree so the anchors coincide exactly.
                let delta = joint_world - (self.links[ib].body.position + qb * resolved.lb);
                if delta != Vec3::ZERO {
                    for j in self.subtree(ib) {
                        self.links[j].body.position += delta;
                    }
                }
                let pb = self.links[ib].body.position;
                let axis = resolved.ax_a;
                let q_off = qa.conjugate() * qb;
                let s0 = qa.conjugate() * (pb - joint_world);
                let s_ang = q_off.conjugate() * axis;
                let p_pin = qb.conjugate() * (joint_world - pb);
                let joint = LinkJoint::Revolute {
                    axis,
                    q_off,
                    r_pj: qa.conjugate() * (joint_world - pa),
                    s0,
                    s_ang,
                    s_lin: p_pin.cross(s_ang),
                    ref_angle: resolved.ref_angle,
                    limit,
                    spec_motor: motor.as_ref().map(RevoluteMotor::as_general),
                };
                Ok((joint, resolved))
            }
            JointKind::Fixed { .. } => {
                let resolved = resolve_joint(
                    kind,
                    self.links[ia].body.position,
                    self.links[ia].body.orientation,
                    self.links[ib].body.position,
                    self.links[ib].body.orientation,
                )
                .ok_or_else(|| JointError::BadAxis {
                    detail: "unresolvable joint frames".to_string(),
                })?;
                let qa = self.links[ia].body.orientation;
                let pa = self.links[ia].body.position;
                let qb = self.links[ib].body.orientation;
                let joint_world = pa + qa * resolved.la;
                let delta = joint_world - (self.links[ib].body.position + qb * resolved.lb);
                if delta != Vec3::ZERO {
                    for j in self.subtree(ib) {
                        self.links[j].body.position += delta;
                    }
                }
                let pb = self.links[ib].body.position;
                Ok((
                    LinkJoint::Fixed {
                        rel_pos: qa.conjugate() * (pb - pa),
                        rel_quat: qa.conjugate() * qb,
                    },
                    resolved,
                ))
            }
            ref other => Err(JointError::Unsupported {
                detail: format!(
                    "featherstone v1 solves revolute/fixed only, got {}",
                    joint_kind_name(other)
                ),
            }),
        }
    }
}

/// Clamp a vector's magnitude.
fn clamp_vec(v: Vec3, max: f32) -> Vec3 {
    if max <= 0.0 {
        return Vec3::ZERO;
    }
    let l = v.length();
    if l > max && l.is_finite() {
        v * (max / l)
    } else {
        v
    }
}

/// Exact exponential-map quaternion integration for world-frame angular
/// velocity `w` over `h`.
fn integrate_quat(q: Quat, w: Vec3, h: f32) -> Quat {
    let omega = w.length();
    if omega < 1e-12 {
        return q;
    }
    (Quat::from_axis_angle(w / omega, omega * h) * q).normalize()
}

/// Short kind name for `Unsupported` details.
fn joint_kind_name(kind: &JointKind) -> &'static str {
    match kind {
        JointKind::Ball { .. } => "ball",
        JointKind::Revolute { .. } => "revolute",
        JointKind::Prismatic { .. } => "prismatic",
        JointKind::Fixed { .. } => "fixed",
        JointKind::Distance { .. } => "distance",
        JointKind::Rope { .. } => "rope",
        JointKind::Spring { .. } => "spring",
        JointKind::Wheel { .. } => "wheel",
        JointKind::Gear { .. } => "gear",
        JointKind::SixDof { .. } => "sixdof",
        JointKind::Motor { .. } => "motor",
    }
}

impl PhysicsEngine for FeatherstoneEngine {
    fn step(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 || self.links.is_empty() {
            return;
        }
        self.resync_coordinates();
        let acc = self.aba(dt);
        self.integrate(&acc, dt);
    }

    fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        let base = if body.body_type == BodyType::Dynamic && body.inv_mass > 0.0 {
            BaseJoint::Floating
        } else {
            BaseJoint::Fixed
        };
        self.add_root(body, base)
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        let hi = handle.index();
        if hi >= self.links.len() {
            return;
        }
        // Children float free with their live velocities (physically
        // continuous); only this link leaves.
        let kids = self.children[hi].clone();
        for c in kids {
            self.links[c].parent = None;
            self.links[c].joint = LinkJoint::Floating;
            self.links[c].spec = None;
        }
        let last = self.links.len() - 1;
        self.links.swap_remove(hi);
        for link in &mut self.links {
            if link.parent == Some(last) {
                link.parent = Some(hi);
            }
        }
        self.rebuild_topology();
    }

    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        self.links.get(handle.index()).map(|l| &l.body)
    }

    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        self.links.get_mut(handle.index()).map(|l| &mut l.body)
    }

    fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        kind: JointKind,
    ) -> Result<JointHandle, JointError> {
        validate_joint(&kind)?;
        let (ia, ib) = (body_a.index(), body_b.index());
        if ia == ib {
            return Err(JointError::SelfJoint { handle: ia });
        }
        if ia >= self.links.len() || ib >= self.links.len() {
            return Err(JointError::InvalidHandles { a: ia, b: ib });
        }
        // Only a root link can be attached (else a cycle or a re-parent).
        if self.links[ib].parent.is_some() {
            return Err(JointError::Unsupported {
                detail: "articulated attach needs a root link as the child".to_string(),
            });
        }
        // `b` inside `a`'s ancestry would close a loop (`b` is a root, so
        // the reverse cannot happen).
        let mut cursor = Some(ia);
        while let Some(c) = cursor {
            if c == ib {
                return Err(JointError::Unsupported {
                    detail: "articulated attach would close a kinematic loop".to_string(),
                });
            }
            cursor = self.links[c].parent;
        }
        // The attached subtree must be fully dynamic (infinite-mass links
        // break the articulated inertia).
        for j in self.subtree(ib) {
            if self.links[j].body.body_type != BodyType::Dynamic {
                return Err(JointError::Unsupported {
                    detail: "articulated links must be dynamic".to_string(),
                });
            }
        }
        let (joint, resolved) = self.admit_joint(ia, ib, &kind)?;
        self.links[ib].parent = Some(ia);
        self.links[ib].joint = joint;
        self.links[ib].spec = Some(kind);
        self.links[ib].reference = JointReference::from(resolved);
        // Assembly is the zero of the new coordinates; the subtree keeps no
        // relative velocity across the fresh joint.
        self.links[ib].q = 0.0;
        self.links[ib].qd = 0.0;
        self.rebuild_topology();
        Ok(JointHandle::from(ib))
    }

    fn remove_joint(&mut self, handle: JointHandle) {
        let hi = handle.index();
        if hi >= self.links.len() || self.links[hi].parent.is_none() {
            return;
        }
        // The link (with its subtree still attached beneath it) floats free
        // with live velocities.
        self.links[hi].parent = None;
        self.links[hi].joint = LinkJoint::Floating;
        self.links[hi].spec = None;
        self.rebuild_topology();
    }

    fn raycast(&self, ray: Ray, max_dist: f32) -> Result<Option<RaycastHit>, QueryError> {
        crate::errors::check_ray_input(ray.origin, ray.direction, max_dist)?;
        let mut closest: Option<RaycastHit> = None;
        for (i, link) in self.links.iter().enumerate() {
            let inverse = link.body.orientation.conjugate();
            let origin = inverse * (ray.origin - link.body.position);
            let direction = inverse * ray.direction;
            let Some((distance, local_normal)) =
                raycast_shape_hit(&link.body.shape, origin, direction, max_dist)
            else {
                continue;
            };
            let better = closest.as_ref().is_none_or(|hit| distance < hit.distance);
            if better {
                closest = Some(RaycastHit {
                    handle: BodyHandle::from(i),
                    point: ray.point_at(distance),
                    normal: (link.body.orientation * local_normal).normalize_or(Vec3::Y),
                    distance,
                });
            }
        }
        Ok(closest)
    }

    fn shapecast(
        &self,
        _shape: &crate::shape::Shape,
        _from: Vec3,
        _to: Vec3,
    ) -> Option<RaycastHit> {
        // v1 has no collision geometry: articulated bodies never overlap
        // anything, so a sweep honestly reports no hit (documented).
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::PhysicsEngine;

    /// Test gravity (m/s^2).
    const G: f32 = 9.81;

    fn gravity() -> Vec3 {
        Vec3::new(0.0, -G, 0.0)
    }

    /// Near-point-mass link (sphere inertia `2/5 m r^2` is negligible
    /// against `m L^2`, so Lagrange closed forms apply to ~1e-4).
    fn point_mass(pos: Vec3) -> RigidBody {
        RigidBody::new_sphere(pos, 0.02, 1.0)
    }

    /// Static world anchor.
    fn anchor_body(pos: Vec3) -> RigidBody {
        RigidBody::new_sphere(pos, 0.05, 0.0)
    }

    /// Hinge about Z with the given local anchors.
    fn revolute(la: Vec3, lb: Vec3) -> JointKind {
        JointKind::Revolute {
            local_anchor_a: la,
            local_anchor_b: lb,
            local_axis_a: Vec3::Z,
            local_axis_b: Vec3::Z,
            limit: None,
            motor: None,
        }
    }

    /// Pose a link hanging from `pivot`: assembly arm `arm` (COM offset at
    /// assembly) rotated by `theta` about Z (assembly orientation identity
    /// in these tests).
    fn hang(link: &mut RigidBody, pivot: Vec3, arm: Vec3, theta: f32) {
        let r = Quat::from_rotation_z(theta);
        link.orientation = r;
        link.position = pivot + r * arm;
    }

    /// Mechanical energy of one link (translational + rotational + gravity
    /// potential against the pivot height).
    fn link_energy(body: &RigidBody, pivot_y: f32) -> f32 {
        let rot = Mat3::from_quat(body.orientation);
        let iw = rot * (body.inertia * (rot.transpose() * body.angular_velocity));
        0.5 * body.mass * body.velocity.length_squared()
            + 0.5 * body.angular_velocity.dot(iw)
            + body.mass * G * (body.position.y - pivot_y)
    }

    /// Double pendulum with unit point masses at unit length: anchor at the
    /// origin, link0 COM `(0,-1,0)`, link1 COM `(0,-2,0)`. Assembly is the
    /// vertical pose; call [`pose_double_pendulum`] to displace (posing
    /// after attach keeps the assembly reference vertical, so `q` reads
    /// true assembly-relative angles).
    fn double_pendulum() -> (
        FeatherstoneEngine,
        BodyHandle,
        BodyHandle,
        JointHandle,
        JointHandle,
    ) {
        let mut engine = FeatherstoneEngine::new(gravity());
        let anchor = engine.add_body(anchor_body(Vec3::ZERO));
        let h0 = engine.add_body(point_mass(Vec3::new(0.0, -1.0, 0.0)));
        let j0 = engine
            .add_joint(anchor, h0, revolute(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)))
            .expect("revolute admits");
        let h1 = engine.add_body(point_mass(Vec3::new(0.0, -2.0, 0.0)));
        let j1 = engine
            .add_joint(h0, h1, revolute(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)))
            .expect("revolute admits");
        (engine, h0, h1, j0, j1)
    }

    /// Displace an assembled double pendulum to `(t1, t2)` through direct
    /// pose edits (exercises the resync path): link0 twists `t1` about the
    /// pivot, link1 twists `t1 + t2` about the first joint.
    fn pose_double_pendulum(
        engine: &mut FeatherstoneEngine,
        h0: BodyHandle,
        h1: BodyHandle,
        t1: f32,
        t2: f32,
    ) {
        let r0 = Quat::from_rotation_z(t1);
        let joint1 = r0 * Vec3::new(0.0, -1.0, 0.0);
        {
            let l0 = engine.get_body_mut(h0).expect("body");
            l0.orientation = r0;
            l0.position = joint1;
        }
        let r01 = Quat::from_rotation_z(t1 + t2);
        {
            let l1 = engine.get_body_mut(h1).expect("body");
            l1.orientation = r01;
            l1.position = joint1 + r01 * Vec3::new(0.0, -1.0, 0.0);
        }
    }

    /// Lagrange closed form for the double point pendulum at `(t1, t2)`
    /// with `m1 = m2 = L1 = L2 = 1`: mass matrix + gravity torque.
    fn lagrange_qdd(t1: f32, t2: f32) -> (f32, f32) {
        let (m1, m2, l1, l2) = (1.0, 1.0, 1.0, 1.0);
        let c2 = t2.cos();
        let m11 = (m1 + m2) * l1 * l1 + m2 * l2 * l2 + 2.0 * m2 * l1 * l2 * c2;
        let m12 = m2 * l2 * l2 + m2 * l1 * l2 * c2;
        let m22 = m2 * l2 * l2;
        let tau1 = -G * ((m1 + m2) * l1 * t1.sin() + m2 * l2 * (t1 + t2).sin());
        let tau2 = -G * m2 * l2 * (t1 + t2).sin();
        let det = m11 * m22 - m12 * m12;
        (
            (m22 * tau1 - m12 * tau2) / det,
            (m11 * tau2 - m12 * tau1) / det,
        )
    }

    /// Skew is the cross product: `skew(a) b == a x b`.
    #[test]
    fn skew_matrix_is_the_cross_product() {
        let (a, b) = (Vec3::new(1.0, -2.0, 0.5), Vec3::new(0.25, 3.0, -1.5));
        assert!((skew(a) * b - a.cross(b)).length() < 1e-6);
    }

    /// Motion/force duality: `f . (X v) == (X' f) . v`, and the 6x6 matrix
    /// form agrees with the direct application.
    #[test]
    fn transform_duality_holds() {
        let x = SpatialTransform {
            e: Mat3::from_rotation_z(0.7),
            r: Vec3::new(0.3, -0.2, 0.5),
        };
        let v = [1.0, -2.0, 0.5, 3.0, 0.25, -1.5];
        let f = [0.5, 1.5, -1.0, -2.0, 0.75, 2.25];
        let lhs = dot6(&f, &apply_motion(&x, &v));
        let rhs = dot6(&apply_force_t(&x, &f), &v);
        assert!((lhs - rhs).abs() < 1e-5, "duality gap {lhs} vs {rhs}");
        let m = mat_of_transform(&x);
        let direct = apply_motion(&x, &v);
        let matrix = mat_vec(&m, &v);
        for k in 0..6 {
            assert!((direct[k] - matrix[k]).abs() < 1e-5, "6x6 mismatch at {k}");
        }
    }

    /// The 6x6 solve inverts cleanly on a PD system.
    #[test]
    fn solve6_inverts_a_pd_system() {
        let a = link_inertia(Vec3::new(0.5, 0.4, 0.3), 2.0);
        let b = [1.0, -1.0, 0.5, 2.0, 0.25, -0.75];
        let x = solve6(&a, &b);
        let back = mat_vec(&a, &x);
        for k in 0..6 {
            assert!((back[k] - b[k]).abs() < 1e-5, "residual at {k}");
        }
    }

    /// Initial joint accelerations match the Lagrange closed form (the
    /// semi-analytic check): `qdd` read off after one small step.
    #[test]
    fn double_pendulum_initial_acceleration_matches_lagrange() {
        let (mut engine, h0, h1, j0, j1) = double_pendulum();
        pose_double_pendulum(&mut engine, h0, h1, 0.05, 0.05);
        let h = 1.0 / 240.0;
        engine.step(h);
        let qdd0 = engine.joint_velocity(j0).expect("vel") / h;
        let qdd1 = engine.joint_velocity(j1).expect("vel") / h;
        let (e0, e1) = lagrange_qdd(0.05, 0.05);
        assert!((qdd0 - e0).abs() < 3e-3, "qdd0 {qdd0} vs lagrange {e0}");
        assert!((qdd1 - e1).abs() < 3e-3, "qdd1 {qdd1} vs lagrange {e1}");
    }

    /// Small-angle period within 1%: upward zero crossings over two periods
    /// against `2 PI sqrt(I_pivot / (m g L))`.
    #[test]
    fn single_pendulum_period_matches_small_angle_analytic() {
        let mut engine = FeatherstoneEngine::new(gravity());
        let anchor = engine.add_body(anchor_body(Vec3::ZERO));
        let h0 = engine.add_body(point_mass(Vec3::new(0.0, -1.0, 0.0)));
        let j = engine
            .add_joint(anchor, h0, revolute(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)))
            .expect("revolute admits");
        // Displace after assembly so `q` reads the true assembly-relative
        // angle (posing before attach would zero the coordinates there).
        {
            let link = engine.get_body_mut(h0).expect("body");
            hang(link, Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), 0.05);
        }
        let h = 1.0 / 240.0;
        let mut prev = 0.05;
        let mut crossings = Vec::new();
        let mut t = 0.0;
        for _ in 0..1500 {
            engine.step(h);
            t += h;
            let q = engine.joint_angle(j).expect("angle");
            if prev < 0.0 && q >= 0.0 {
                crossings.push(t - h + (-prev / (q - prev)) * h);
            }
            prev = q;
        }
        assert!(crossings.len() >= 3, "need two periods, got {crossings:?}");
        let avg = (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f32;
        let i_pivot = 1.0 + 0.4 * 0.02 * 0.02;
        let t0 = 2.0 * std::f32::consts::PI * (i_pivot / G).sqrt();
        assert!(
            (avg - t0).abs() / t0 < 0.01,
            "period {avg} vs analytic {t0}"
        );
    }

    /// No pumping over 600 steps: single-pendulum energy stays in a 5% band.
    #[test]
    fn single_pendulum_energy_stays_bounded() {
        let mut engine = FeatherstoneEngine::new(gravity());
        let anchor = engine.add_body(anchor_body(Vec3::ZERO));
        let h0 = engine.add_body(point_mass(Vec3::new(0.0, -1.0, 0.0)));
        engine
            .add_joint(anchor, h0, revolute(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)))
            .expect("revolute admits");
        {
            let link = engine.get_body_mut(h0).expect("body");
            hang(link, Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), 0.05);
        }
        let h = 1.0 / 120.0;
        let e0 = link_energy(engine.get_body(h0).expect("body"), 0.0);
        let mut worst = 0.0f32;
        for _ in 0..600 {
            engine.step(h);
            let e = link_energy(engine.get_body(h0).expect("body"), 0.0);
            worst = worst.max((e - e0).abs() / e0.abs());
        }
        assert!(worst < 0.05, "energy drift {worst}");
    }

    /// Double pendulum over 600 steps: bounded energy, finite state, and
    /// the anchors still coincide (zero drift by construction).
    #[test]
    fn double_pendulum_runs_600_steps_without_pumping() {
        let (mut engine, h0, h1, _, _) = double_pendulum();
        pose_double_pendulum(&mut engine, h0, h1, 0.05, 0.05);
        let h = 1.0 / 120.0;
        let bodies: Vec<BodyHandle> = (0..engine.body_count()).map(BodyHandle::from).collect();
        let e0: f32 = bodies
            .iter()
            .map(|&b| link_energy(engine.get_body(b).expect("body"), 0.0))
            .sum();
        let mut worst = 0.0f32;
        for _ in 0..600 {
            engine.step(h);
            let e: f32 = bodies
                .iter()
                .map(|&b| link_energy(engine.get_body(b).expect("body"), 0.0))
                .sum();
            worst = worst.max((e - e0).abs() / e0.abs());
            for &b in &bodies {
                let body = engine.get_body(b).expect("body");
                assert!(body.position.is_finite());
                assert!(body.velocity.is_finite());
            }
        }
        assert!(worst < 0.10, "energy drift {worst}");
    }

    /// Six-link arm holds a bent pose under gravity with position servos:
    /// bounded drift, no blowup. Stiff holding on a floppy chain excites
    /// fast whip modes (large distal accelerations are the exact dynamics),
    /// so this runs at kHz steps — explicit Euler needs it, same as any
    /// stiff drive (see the module docs; substepping is the follow-up).
    #[test]
    fn six_link_arm_holds_pose_under_gravity() {
        let mut engine = FeatherstoneEngine::new(gravity());
        let anchor = engine.add_body(anchor_body(Vec3::ZERO));
        let mut prev = anchor;
        let mut joints = Vec::new();
        for k in 0..6 {
            let com = Vec3::new(0.0, -0.25 - 0.5 * k as f32, 0.0);
            let b = engine.add_body(RigidBody::new_sphere(com, 0.05, 1.0));
            let la = if k == 0 {
                Vec3::ZERO
            } else {
                Vec3::new(0.0, -0.25, 0.0)
            };
            let j = engine
                .add_joint(prev, b, revolute(la, Vec3::new(0.0, 0.25, 0.0)))
                .expect("revolute admits");
            // Acceleration-based (default model): uniform closed-loop
            // bandwidth on every joint; the gain holds against gravity
            // through the articulated inertia (see the module docs).
            let motor = JointMotor::position(0.15, 4000.0, 150.0, 1e4).expect("motor");
            engine.set_joint_motor(j, Some(motor)).expect("servo");
            prev = b;
            joints.push((j, 0.15));
        }
        let h = 1.0 / 8000.0;
        for _ in 0..4800 {
            engine.step(h);
        }
        for (j, target) in joints {
            let q = engine.joint_angle(j).expect("angle");
            assert!(q.is_finite(), "non-finite angle");
            assert!((q - target).abs() < 0.05, "drift {q} vs target {target}");
        }
    }

    /// Free-floating two-link chain falls as a whole: COM tracks `g t^2/2`,
    /// total momentum tracks `M g t`, internal angles and spin stay zero.
    #[test]
    fn floating_base_falls_as_a_whole() {
        let mut engine = FeatherstoneEngine::new(gravity());
        let h0 = engine.add_body(point_mass(Vec3::ZERO));
        let h1 = engine.add_body(point_mass(Vec3::new(0.0, -0.6, 0.0)));
        let j = engine
            .add_joint(
                h0,
                h1,
                revolute(Vec3::new(0.0, -0.3, 0.0), Vec3::new(0.0, 0.3, 0.0)),
            )
            .expect("revolute admits");
        let h = 1.0 / 60.0;
        for _ in 0..120 {
            engine.step(h);
        }
        let t = 2.0;
        let b0 = engine.get_body(h0).expect("body").clone();
        let b1 = engine.get_body(h1).expect("body").clone();
        let com_y = 0.5 * (b0.position.y + b1.position.y);
        let expected = -0.3 - 0.5 * G * t * t;
        assert!(
            (com_y - expected).abs() / expected.abs() < 0.02,
            "com {com_y} vs free fall {expected}"
        );
        let q = engine.joint_angle(j).expect("angle");
        assert!(q.abs() < 1e-3, "self-actuation {q}");
        let p = b0.velocity + b1.velocity;
        assert!(
            (p.y + 2.0 * G * t).abs() < 0.05,
            "momentum drift {}",
            p.y + 2.0 * G * t
        );
        // Angular momentum about the COM stays zero.
        let com = 0.5 * (b0.position + b1.position);
        let mut l = Vec3::ZERO;
        for b in [&b0, &b1] {
            l += (b.position - com).cross(b.velocity) * b.mass;
        }
        assert!(l.length() < 1e-2, "spin {l:?}");
    }

    /// Determinism: two identical runs are bit-identical.
    #[test]
    fn rerun_is_bit_identical() {
        fn snapshot(engine: &FeatherstoneEngine) -> Vec<u32> {
            let mut out = Vec::new();
            for i in 0..engine.body_count() {
                let b = engine.get_body(BodyHandle::from(i)).expect("body");
                for v in [
                    b.position.x,
                    b.position.y,
                    b.position.z,
                    b.orientation.x,
                    b.orientation.y,
                    b.orientation.z,
                    b.orientation.w,
                    b.velocity.x,
                    b.velocity.y,
                    b.velocity.z,
                    b.angular_velocity.x,
                    b.angular_velocity.y,
                    b.angular_velocity.z,
                ] {
                    out.push(v.to_bits());
                }
            }
            out
        }
        let (mut a, ah0, ah1, _, _) = double_pendulum();
        pose_double_pendulum(&mut a, ah0, ah1, 0.05, 0.05);
        for _ in 0..200 {
            a.step(1.0 / 120.0);
        }
        let (mut b, bh0, bh1, _, _) = double_pendulum();
        pose_double_pendulum(&mut b, bh0, bh1, 0.05, 0.05);
        for _ in 0..200 {
            b.step(1.0 / 120.0);
        }
        assert_eq!(snapshot(&a), snapshot(&b));
    }

    /// Floating-root tether: a displaced base with velocity returns to the
    /// pose captured when the motor was set and settles there.
    #[test]
    fn tether_returns_base_to_captured_pose() {
        let mut engine = FeatherstoneEngine::new(Vec3::ZERO);
        let h0 = engine.add_body(point_mass(Vec3::ZERO));
        let tether = JointMotor::position(0.0, 200.0, 30.0, 1e5).expect("tether");
        engine
            .set_joint_motor(JointHandle::from(h0.index()), Some(tether))
            .expect("tether sets");
        {
            let link = engine.get_body_mut(h0).expect("body");
            link.position = Vec3::new(0.3, -0.2, 0.1);
            link.orientation = Quat::from_rotation_z(0.2);
            link.velocity = Vec3::new(1.0, 0.0, 0.0);
        }
        for _ in 0..240 {
            engine.step(1.0 / 120.0);
        }
        let body = engine.get_body(h0).expect("body");
        assert!(body.position.length() < 0.02, "pos {:?}", body.position);
        assert!(
            quat_twist(body.orientation, Vec3::Z).abs() < 0.02,
            "attitude {:?}",
            body.orientation
        );
        assert!(body.velocity.length() < 0.05, "vel {:?}", body.velocity);
    }

    /// Fixed (welded) base vs tethered floating base on the same offset
    /// chain agree over a short window: a stiff tether converges to a weld
    /// (the fixed-vs-floating path), so the hanging child must swing the
    /// same way in both worlds. The base is posed at 0.02 rad, which also
    /// covers floating-root velocity propagation under a non-identity
    /// root orientation (world-frame root velocities must rotate into the
    /// link frame before propagating to children).
    ///
    /// The fixed world welds the base to the anchor rather than hinging
    /// it: a hinged base is a double pendulum whose moving joint
    /// accelerates the child (~1 m lever, physically a different
    /// mechanism), so it cannot be compared with a held base.
    #[test]
    fn fixed_and_tethered_floating_bases_agree() {
        // Base COM at `r0 (0,-0.5,0)`, child joint at `r0 (0,-1,0)`, child
        // COM 0.25 below the joint; base at 0.02 rad, child at 0.04 rad
        // world (0.02 rad relative to the base).
        fn build(floating: bool) -> (FeatherstoneEngine, BodyHandle, JointHandle) {
            let mut engine = FeatherstoneEngine::new(gravity());
            let anchor = (!floating).then(|| engine.add_body(anchor_body(Vec3::ZERO)));
            let t0 = 0.02;
            let r0 = Quat::from_rotation_z(t0);
            let mut base_body = point_mass(r0 * Vec3::new(0.0, -0.5, 0.0));
            base_body.orientation = r0;
            let base = engine.add_body(base_body);
            let joint1 = r0 * Vec3::new(0.0, -1.0, 0.0);
            let r01 = Quat::from_rotation_z(2.0 * t0);
            let mut child = point_mass(joint1 + r01 * Vec3::new(0.0, -0.25, 0.0));
            child.orientation = r01;
            let h1 = engine.add_body(child);
            if let Some(anchor) = anchor {
                engine
                    .add_joint(
                        anchor,
                        base,
                        JointKind::Fixed {
                            local_anchor_a: Vec3::ZERO,
                            local_anchor_b: Vec3::new(0.0, 0.5, 0.0),
                        },
                    )
                    .expect("weld");
            }
            let j1 = engine
                .add_joint(
                    base,
                    h1,
                    revolute(Vec3::new(0.0, -0.5, 0.0), Vec3::new(0.0, 0.25, 0.0)),
                )
                .expect("joint");
            (engine, base, j1)
        }
        let (mut fixed, fh0, fj1) = build(false);
        let (mut floating, gh0, gj1) = build(true);
        let tether = JointMotor::position(0.0, 4000.0, 100.0, 1e8).expect("tether");
        floating
            .set_joint_motor(JointHandle::from(gh0.index()), Some(tether))
            .expect("tether sets");
        let h = 1.0 / 4000.0;
        let mut worst = 0.0f32;
        let mut swing = 0.0f32;
        let q1_start = fixed.joint_angle(fj1).expect("angle");
        for _ in 0..400 {
            fixed.step(h);
            floating.step(h);
            let fb = quat_twist(fixed.get_body(fh0).expect("body").orientation, Vec3::Z);
            let gb = quat_twist(floating.get_body(gh0).expect("body").orientation, Vec3::Z);
            let q1 = fixed.joint_angle(fj1).expect("angle");
            let g1 = floating.joint_angle(gj1).expect("angle");
            worst = worst.max((fb - gb).abs()).max((q1 - g1).abs());
            swing = swing.max((q1 - q1_start).abs());
        }
        // The child really swings (gravity acts), and both worlds track.
        assert!(swing > 5e-3, "child barely moved: {swing}");
        assert!(worst < 1e-3, "fixed/floating gap {worst}");
    }

    /// A rotated floating root with a child free-falls as one rigid
    /// whole: COM drop `g t^2 / 2`, no sideways drift, frozen joint angle.
    /// Regression: world-frame root velocities fed to the child
    /// propagation unrotated were re-rotated by the root attitude every
    /// step (only identity roots fell correctly).
    #[test]
    fn rotated_floating_chain_free_falls_rigidly() {
        let mut engine = FeatherstoneEngine::new(gravity());
        let r0 = Quat::from_rotation_z(0.3);
        let mut base_body = point_mass(r0 * Vec3::new(0.0, -0.5, 0.0));
        base_body.orientation = r0;
        let base = engine.add_body(base_body);
        let mut child = point_mass(r0 * Vec3::new(0.0, -1.25, 0.0));
        child.orientation = r0;
        let h1 = engine.add_body(child);
        let j1 = engine
            .add_joint(
                base,
                h1,
                revolute(Vec3::new(0.0, -0.5, 0.0), Vec3::new(0.0, 0.25, 0.0)),
            )
            .expect("joint");
        let start = engine.get_body(base).expect("body").position;
        let q_start = engine.joint_angle(j1).expect("angle");
        let h = 1.0 / 240.0;
        let steps = 60;
        for _ in 0..steps {
            engine.step(h);
        }
        let t = h * steps as f32;
        let pos = engine.get_body(base).expect("body").position;
        let drop = start.y - pos.y;
        // Semi-implicit Euler drops `g h^2 n(n+1)/2`, within 2% of `g t^2/2`.
        assert!(
            (drop - 0.5 * G * t * t).abs() < 0.02 * 0.5 * G * t * t,
            "drop {drop}"
        );
        assert!(
            (pos.x - start.x).abs() < 1e-4,
            "sideways drift {}",
            pos.x - start.x
        );
        let q = engine.joint_angle(j1).expect("angle");
        assert!((q - q_start).abs() < 1e-4, "joint moved {q_start} -> {q}");
    }

    /// Velocity motor spins the hinge to its target rate.
    #[test]
    fn velocity_motor_spins_to_target() {
        let mut engine = FeatherstoneEngine::new(Vec3::ZERO);
        let anchor = engine.add_body(anchor_body(Vec3::ZERO));
        let h0 = engine.add_body(point_mass(Vec3::new(0.0, -1.0, 0.0)));
        let j = engine
            .add_joint(anchor, h0, revolute(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)))
            .expect("joint");
        let motor = JointMotor::velocity(2.0, 10.0).expect("motor");
        engine.set_joint_motor(j, Some(motor)).expect("drive");
        for _ in 0..240 {
            engine.step(1.0 / 120.0);
        }
        let qd = engine.joint_velocity(j).expect("vel");
        assert!((qd - 2.0).abs() < 0.05, "rate {qd}");
    }

    /// Limits clamp travel: a driven hinge stops at the window edge.
    #[test]
    fn revolute_limit_clamps_travel() {
        use ornis_core::units::Radians;
        let mut engine = FeatherstoneEngine::new(Vec3::ZERO);
        let anchor = engine.add_body(anchor_body(Vec3::ZERO));
        let h0 = engine.add_body(point_mass(Vec3::new(0.0, -1.0, 0.0)));
        let limit = RevoluteLimit::try_new(Radians::new(-0.1), Radians::new(0.1)).expect("limit");
        let j = engine
            .add_joint(
                anchor,
                h0,
                JointKind::Revolute {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
                    local_axis_a: Vec3::Z,
                    local_axis_b: Vec3::Z,
                    limit: Some(limit),
                    motor: None,
                },
            )
            .expect("joint");
        let motor = JointMotor::velocity(5.0, 10.0).expect("motor");
        engine.set_joint_motor(j, Some(motor)).expect("drive");
        for _ in 0..120 {
            engine.step(1.0 / 120.0);
        }
        let q = engine.joint_angle(j).expect("angle");
        assert!(q <= 0.1 + 1e-6, "limit overrun {q}");
        assert!(q > 0.09, "never reached the stop {q}");
    }

    /// Non-revolute/non-fixed joints are explicit `Unsupported`, never
    /// silent; fixed joints admit.
    #[test]
    fn non_revolute_joints_are_unsupported() {
        let mut engine = FeatherstoneEngine::new(gravity());
        let a = engine.add_body(point_mass(Vec3::ZERO));
        let b = engine.add_body(point_mass(Vec3::new(0.0, -1.0, 0.0)));
        let c = engine.add_body(point_mass(Vec3::new(0.0, -2.0, 0.0)));
        let d = engine.add_body(point_mass(Vec3::new(0.0, -3.0, 0.0)));
        let f = engine.add_body(point_mass(Vec3::new(0.0, -4.0, 0.0)));
        let z = Vec3::ZERO;
        for kind in [
            JointKind::Ball {
                local_anchor_a: z,
                local_anchor_b: z,
            },
            JointKind::Prismatic {
                local_anchor_a: z,
                local_anchor_b: z,
                local_axis_a: Vec3::Z,
                local_axis_b: Vec3::Z,
                limit: None,
                motor: None,
            },
            JointKind::Distance {
                local_anchor_a: z,
                local_anchor_b: z,
            },
            JointKind::Motor {
                linear_target: Vec3::ZERO,
                angular_target: Vec3::ZERO,
                max_force: 1.0,
                max_torque: 1.0,
                correction: 0.3,
            },
        ] {
            let err = engine.add_joint(a, b, kind).expect_err("unsupported");
            assert!(
                matches!(err, JointError::Unsupported { .. }),
                "explicit unsupported, got {err:?}"
            );
        }
        // Rejections leave no joint behind; fixed admits.
        assert_eq!(engine.joint_count(), 0);
        engine
            .add_joint(
                c,
                d,
                JointKind::Fixed {
                    local_anchor_a: z,
                    local_anchor_b: z,
                },
            )
            .expect("fixed admits");
        engine
            .add_joint(
                c,
                f,
                revolute(Vec3::new(0.0, -0.5, 0.0), Vec3::new(0.0, 0.5, 0.0)),
            )
            .expect("second branch admits");
        assert_eq!(engine.joint_count(), 2);
    }
}
