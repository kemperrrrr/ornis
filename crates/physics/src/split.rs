//! Deterministic island ownership over a shared body/joint registry.
//! Routing precedes integration; rebuilds preserve assembly references and
//! completed-step event baselines, never use kinematic collision proxies.
//!
//! Handle spaces: [`BodyHandle`]/[`JointHandle`] are GLOBAL (slots in
//! [`SplitState::bodies`]/[`SplitState::joints`); [`SplitBody`] stores no
//! global copy — its position is its identity, so `swap_remove` cannot go
//! stale). [`LocalAvbdBody`]/[`LocalSiBody`] ([`LocalAvbdJoint`]/
//! [`LocalSiJoint`]) are engine-table indices. Conversions live only in
//! this file (plus one-line typed accessors on the engines); every other
//! `BodyHandle` in `avbd`/`sequential_impulse` is engine-local.

use std::collections::BTreeSet;
use std::time::Instant;

use glam::{Quat, Vec3};

use crate::broadphase::PrevPose;
use crate::constants::NEAR_ZERO;
use crate::distance::{ShapeRef, cast_shape};
use crate::engine::raycast_shape_hit;
use crate::flags::{RoutePhase, SolverSide};
use crate::joint::CrossRowKind;
use crate::migration::{EventState, JointSnapshot};
use crate::{
    AABB, AvbdEngine, BodyHandle, BodyType, ContactEvent, ContactEventKind, CrossJointStatus,
    JointHandle, JointKind, LocalAvbdBody, LocalAvbdJoint, LocalSiBody, LocalSiJoint,
    PhysicsEngine, Ray, RaycastHit, RigidBody, SequentialImpulseEngine, Shape, SolverKind,
    SplitTiming, TriggerEvent, TriggerEventKind, XpbdEngine,
};

pub(super) const DT: f32 = 1.0 / 60.0;
pub(super) const MAX_STEPS: usize = 4;
const SLEEP_SPEED: f32 = 0.2;
/// Midpoint / half-extent scale.
const HALF: f32 = 0.5;
const WAKE_SPEED: f32 = HALF;
/// Squared-length floor for a usable couple direction.
const MIN_DIR_LEN2: f32 = HALF;
const QUIET_STEPS: u32 = 30;
const LINK_MARGIN: f32 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SplitOwner {
    Static,
    Avbd,
    SequentialImpulse,
    /// Host-pinned XPBD bodies (never assigned by hysteresis — see
    /// [`crate::Engine::pin_body_solver`]). The split XPBD engine mirrors
    /// only these bodies, in registry order, so its dense table never
    /// double-steps a body owned elsewhere.
    Xpbd,
}

impl SplitOwner {
    fn preferred(kind: SolverKind) -> Self {
        match kind {
            SolverKind::SequentialImpulse => Self::SequentialImpulse,
            // Islands routing parks soft bodies and only ever splits rigid
            // bodies between SI and AVBD: an XPBD-kind world entering
            // Islands starts its dynamics on the calm AVBD side (same
            // default as structural adds under Islands).
            SolverKind::Avbd | SolverKind::Xpbd => Self::Avbd,
        }
    }
}

/// Dense body index into [`SplitState::xpbd`].
///
/// The XPBD engine speaks plain [`BodyHandle`] dense indices (it has no
/// local-handle newtype — its table is filled with the XPBD-owned subset
/// in registry order, so dense and global indices differ). This wrapper
/// keeps the two spaces apart at the type level inside this file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct XpbdBody(u32);

impl XpbdBody {
    fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<XpbdBody> for BodyHandle {
    fn from(h: XpbdBody) -> Self {
        Self::from_raw(h.0)
    }
}

/// Dense joint index into [`SplitState::xpbd`] (same discipline as
/// [`XpbdBody`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct XpbdJoint(u32);

impl XpbdJoint {
    fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<XpbdJoint> for JointHandle {
    fn from(h: XpbdJoint) -> Self {
        Self::from_raw(h.0)
    }
}

/// One queued cross-solver row: the two global ends, their local anchors,
/// the structural row to solve and (distance only) the assembly rest
/// length. `Copy` so sweeps can re-read it without touching the registry.
#[derive(Debug, Clone, Copy)]
struct CrossWork {
    a: BodyHandle,
    b: BodyHandle,
    la: Vec3,
    lb: Vec3,
    row: CrossRowKind,
    rest: f32,
}

/// Short solver-agnostic joint name for status/error strings.
fn joint_kind_name(kind: &JointKind) -> &'static str {
    match kind {
        JointKind::Ball { .. } => "ball",
        JointKind::Revolute { .. } => "revolute",
        JointKind::Prismatic { .. } => "prismatic",
        JointKind::Fixed { .. } => "fixed",
        JointKind::Distance { .. } => "distance",
        JointKind::Wheel { .. } => "wheel",
        JointKind::Gear { .. } => "gear",
        JointKind::SixDof { .. } => "six-dof",
    }
}

/// Apply the inverse world-space inertia tensor (`R · I⁻¹_body · Rᵀ`),
/// zero-guarded per axis. Same standard formula as the SI kernels, kept
/// local so the coupling pass depends only on [`RigidBody`] storage.
fn cross_inv_inertia(inertia: Vec3, orientation: Quat, v: Vec3) -> Vec3 {
    let body = orientation.inverse() * v;
    let scaled = Vec3::new(
        if inertia.x > 0.0 {
            body.x / inertia.x
        } else {
            0.0
        },
        if inertia.y > 0.0 {
            body.y / inertia.y
        } else {
            0.0
        },
        if inertia.z > 0.0 {
            body.z / inertia.z
        } else {
            0.0
        },
    );
    orientation * scaled
}

/// PBD inverse-mass split of a positional correction: `(wa, wb)` sum to 1
/// (a static side takes none). `(0, 0)` when both sides are infinite-mass.
fn cross_weights(a: &RigidBody, b: &RigidBody) -> (f32, f32) {
    let (ia, ib) = (a.inv_mass.max(0.0), b.inv_mass.max(0.0));
    let sum = ia + ib;
    if sum <= 0.0 {
        (0.0, 0.0)
    } else {
        (ia / sum, ib / sum)
    }
}

/// Positional projection of one cross row. Ball pulls both world anchors
/// together, distance restores the assembly rest length along the anchor
/// delta. Non-finite input never reaches the engines.
fn solve_cross_position(a: &mut RigidBody, b: &mut RigidBody, w: &CrossWork) {
    a.restore_sleep_triple();
    b.restore_sleep_triple();
    if !a.position.is_finite()
        || !b.position.is_finite()
        || !a.orientation.is_finite()
        || !b.orientation.is_finite()
    {
        return;
    }
    let ra = a.orientation * w.la;
    let rb = b.orientation * w.lb;
    let delta = (b.position + rb) - (a.position + ra);
    if !delta.is_finite() {
        return;
    }
    let (wa, wb) = cross_weights(a, b);
    match w.row {
        CrossRowKind::Ball => {
            a.position += delta * wa;
            b.position -= delta * wb;
        }
        CrossRowKind::Distance => {
            let len = delta.length();
            if len < NEAR_ZERO || !len.is_finite() || !w.rest.is_finite() {
                return;
            }
            let correction = delta / len * (len - w.rest);
            a.position += correction * wa;
            b.position -= correction * wb;
        }
    }
}

/// Point velocity of a body (`v + ω × r`).
fn cross_point_velocity(body: &RigidBody, r: Vec3) -> Vec3 {
    body.velocity + body.angular_velocity.cross(r)
}

/// Apply a linear impulse at the anchor offsets (sequential-impulse
/// sign convention: `+j` on B, `-j` on A).
fn cross_apply_impulse(a: &mut RigidBody, b: &mut RigidBody, impulse: Vec3, ra: Vec3, rb: Vec3) {
    a.velocity -= impulse * a.inv_mass;
    a.angular_velocity -= cross_inv_inertia(a.inertia, a.orientation, ra.cross(impulse));
    b.velocity += impulse * b.inv_mass;
    b.angular_velocity += cross_inv_inertia(b.inertia, b.orientation, rb.cross(impulse));
}

/// Effective inverse mass along `dir` at the anchor offsets (linear +
/// rotational terms through the world-space inertia).
fn cross_effective_mass(a: &RigidBody, b: &RigidBody, dir: Vec3, ra: Vec3, rb: Vec3) -> f32 {
    let ra_d = ra.cross(dir);
    let rb_d = rb.cross(dir);
    a.inv_mass.max(0.0)
        + b.inv_mass.max(0.0)
        + ra_d.dot(cross_inv_inertia(a.inertia, a.orientation, ra_d))
        + rb_d.dot(cross_inv_inertia(b.inertia, b.orientation, rb_d))
}

/// Velocity row of one cross joint: kills the relative anchor-point
/// velocity along every constrained axis (3 for ball, 1 along the delta
/// for distance). Equality rows, no clamping.
fn solve_cross_velocity(a: &mut RigidBody, b: &mut RigidBody, w: &CrossWork) {
    a.restore_sleep_triple();
    b.restore_sleep_triple();
    if !a.velocity.is_finite()
        || !b.velocity.is_finite()
        || !a.angular_velocity.is_finite()
        || !b.angular_velocity.is_finite()
    {
        return;
    }
    let ra = a.orientation * w.la;
    let rb = b.orientation * w.lb;
    /// Linear DOF count for ball-joint cross rows.
    const LINEAR_DOF: usize = 3;
    let dirs: [Vec3; LINEAR_DOF] = match w.row {
        CrossRowKind::Ball => [Vec3::X, Vec3::Y, Vec3::Z],
        CrossRowKind::Distance => {
            let delta = (b.position + rb) - (a.position + ra);
            let len = delta.length();
            if len < NEAR_ZERO || !len.is_finite() {
                return;
            }
            let n = delta / len;
            [n, Vec3::ZERO, Vec3::ZERO]
        }
    };
    for dir in dirs {
        if dir.length_squared() < MIN_DIR_LEN2 {
            continue;
        }
        let k = cross_effective_mass(a, b, dir, ra, rb);
        if k < NEAR_ZERO {
            continue;
        }
        let vrel = (cross_point_velocity(b, rb) - cross_point_velocity(a, ra)).dot(dir);
        cross_apply_impulse(a, b, dir * (-vrel / k), ra, rb);
    }
}

pub(super) struct SplitBody {
    pub body: RigidBody,
    pub owner: SplitOwner,
    /// Host-forced owner ([`crate::Engine::pin_body_solver`]): hysteresis
    /// never reassigns a pinned body, so a joint between differently-pinned
    /// bodies stays cross-solver by construction instead of reuniting.
    pub pinned: Option<SolverKind>,
    /// AVBD-table index, not a global handle. `None` when the body is not
    /// mirrored into AVBD (SI/XPBD-owned).
    pub local_avbd: Option<LocalAvbdBody>,
    /// SI-table index, not a global handle. `None` when the body is not
    /// mirrored into SI (AVBD/XPBD-owned).
    pub local_si: Option<LocalSiBody>,
    /// XPBD-table index (see [`XpbdBody`]). `None` unless XPBD-owned.
    pub local_xpbd: Option<XpbdBody>,
    pub sleepy: u32,
    pub previous: PrevPose,
}

/// Global handle for the registry slot at `index`: the body's position in
/// [`SplitState::bodies`] IS its global identity (stored nowhere per body
/// so `swap_remove` can never leave a stale copy behind).
pub(super) fn global_handle(index: usize) -> BodyHandle {
    BodyHandle::from(index)
}

/// Global joint handle for the registry slot at `index` (same discipline
/// as [`global_handle`]).
pub(super) fn global_joint(index: usize) -> JointHandle {
    JointHandle::from(index)
}

impl SplitBody {
    pub(super) fn new(body: RigidBody, kind: SolverKind) -> Self {
        let owner = if body.body_type == BodyType::Dynamic {
            SplitOwner::preferred(kind)
        } else {
            SplitOwner::Static
        };
        let previous = PrevPose {
            pos: body.position,
            rot: body.orientation,
        };
        Self {
            body,
            owner,
            pinned: None,
            local_avbd: None,
            local_si: None,
            local_xpbd: None,
            sleepy: 0,
            previous,
        }
    }
}

pub(super) struct SplitJoint {
    pub state: JointSnapshot,
    /// AVBD-table joint index, not a global joint handle.
    pub local_avbd: Option<LocalAvbdJoint>,
    /// SI-table joint index, not a global joint handle.
    pub local_si: Option<LocalSiJoint>,
    /// XPBD-table joint index (see [`XpbdJoint`]).
    pub local_xpbd: Option<XpbdJoint>,
}

impl SplitJoint {
    pub(super) fn new(state: JointSnapshot) -> Self {
        Self {
            state,
            local_avbd: None,
            local_si: None,
            local_xpbd: None,
        }
    }
}

pub(super) struct SplitState {
    pub si: SequentialImpulseEngine,
    pub avbd: AvbdEngine,
    /// Third split engine: mirrors XPBD-owned (host-pinned) bodies only,
    /// in registry order. Never receives soft bodies (they stay parked in
    /// the orchestrator for the whole Islands session) and is stepped only
    /// while it owns at least one body.
    pub xpbd: XpbdEngine,
    pub gravity: Vec3,
    pub bodies: Vec<SplitBody>,
    pub joints: Vec<SplitJoint>,
    pub events: EventState,
    pub time_debt: f64,
    pub timing: SplitTiming,
    pub rebuilds: u64,
    pub migrated_bodies: u64,
    pub steps: u64,
}

/// Iterative path compression avoids stack depth depending on world size.
fn find(root: &mut [usize], mut x: usize) -> usize {
    while root[x] != x {
        root[x] = root[root[x]];
        x = root[x];
    }
    x
}

fn link(root: &mut [usize], a: usize, b: usize) {
    let (a, b) = (find(root, a), find(root, b));
    root[a.max(b)] = a.min(b);
}

fn restore_mass(body: &mut RigidBody) {
    body.restore_sleep_triple();
}

/// Conservative one-tick occupancy, including acceleration and rotation.
fn swept_bounds(body: &RigidBody, gravity: Vec3, dt: f32) -> AABB {
    let mut bounds = body.shape.aabb(body.position, body.orientation);
    if body.angular_velocity.length_squared() > 0.0 || body.torque.length_squared() > 0.0 {
        let local = body.shape.aabb(Vec3::ZERO, Quat::IDENTITY);
        let radius = local.min.abs().max(local.max.abs()).length();
        bounds = AABB::new(
            body.position - Vec3::splat(radius),
            body.position + Vec3::splat(radius),
        );
    }
    let travel = body.velocity * dt;
    let margin = gravity.abs() * (dt * dt) + Vec3::splat(LINK_MARGIN);
    bounds.min += travel.min(Vec3::ZERO) - margin;
    bounds.max += travel.max(Vec3::ZERO) + margin;
    bounds
}

impl SplitState {
    pub(super) fn new(gravity: Vec3) -> Self {
        Self {
            si: SequentialImpulseEngine::new(gravity),
            avbd: AvbdEngine::new(gravity),
            xpbd: XpbdEngine::new(gravity),
            gravity,
            bodies: Vec::new(),
            joints: Vec::new(),
            events: EventState::default(),
            time_debt: 0.0,
            timing: SplitTiming::default(),
            rebuilds: 0,
            migrated_bodies: 0,
            steps: 0,
        }
    }

    /// All physical references remain in global handle space.
    pub(super) fn joint_snapshots(&self) -> Vec<JointSnapshot> {
        self.joints.iter().map(|j| j.state).collect()
    }

    /// Capture rest state at add_joint time, not at the later rebuild.
    ///
    /// # Errors
    ///
    /// [`crate::errors::JointError`] for self-joints, bad specs, unknown
    /// bodies/gears, or degenerate hinges.
    pub(super) fn add_joint(
        &mut self,
        a: BodyHandle,
        b: BodyHandle,
        spec: JointKind,
    ) -> Result<JointHandle, crate::errors::JointError> {
        use crate::errors::JointError;
        if a == b {
            return Err(JointError::SelfJoint { handle: a.index() });
        }
        crate::migration::validate_joint(&spec)?;
        let (ba, bb) = (
            &self
                .bodies
                .get(a.index())
                .ok_or(JointError::InvalidHandles {
                    a: a.index(),
                    b: b.index(),
                })?
                .body,
            &self
                .bodies
                .get(b.index())
                .ok_or(JointError::InvalidHandles {
                    a: a.index(),
                    b: b.index(),
                })?
                .body,
        );
        let reference = if let JointKind::Gear {
            joint_a,
            joint_b,
            ratio,
        } = spec
        {
            crate::migration::JointReference {
                distance: crate::invariants::Meters(
                    self.coordinate(joint_a).ok_or(JointError::UnknownRef {
                        handle: joint_a.index(),
                    })? + ratio
                        * self.coordinate(joint_b).ok_or(JointError::UnknownRef {
                            handle: joint_b.index(),
                        })?,
                ),
                ..crate::migration::JointReference::default()
            }
        } else {
            let r = crate::resolve_joint(
                &spec,
                ba.position,
                ba.orientation,
                bb.position,
                bb.orientation,
            )
            .ok_or_else(|| JointError::BadAxis {
                detail: "unresolvable joint frames".to_string(),
            })?;
            if r.degenerate
                && matches!(
                    spec,
                    JointKind::Revolute { .. } | JointKind::Prismatic { .. }
                )
            {
                return Err(JointError::BadAxis {
                    detail: "degenerate hinge/slide axis".to_string(),
                });
            }
            r.into()
        };
        let h = global_joint(self.joints.len());
        self.joints.push(SplitJoint::new(JointSnapshot {
            a,
            b,
            spec,
            reference,
        }));
        Ok(h)
    }

    fn coordinate(&self, h: JointHandle) -> Option<f32> {
        let j = &self.joints.get(h.index())?.state;
        let (a, b) = (
            &self.bodies[j.a.index()].body,
            &self.bodies[j.b.index()].body,
        );
        match j.spec {
            JointKind::Revolute { local_axis_a, .. } => Some(
                crate::engine::joints::hinge_twist(
                    a.orientation,
                    b.orientation,
                    local_axis_a.normalize_or(Vec3::Z),
                ) - j.reference.angle.0,
            ),
            JointKind::Prismatic {
                local_anchor_a,
                local_anchor_b,
                local_axis_a,
                ..
            } => {
                let axis =
                    (a.orientation * local_axis_a.normalize_or(Vec3::Z)).normalize_or(Vec3::Z);
                Some(
                    ((b.position + b.orientation * local_anchor_b)
                        - (a.position + a.orientation * local_anchor_a))
                        .dot(axis)
                        - j.reference.length.0,
                )
            }
            _ => None,
        }
    }

    /// Rebuilds retain joint rest data, driver baselines and contact history.
    ///
    /// Global handles index [`SplitState::bodies`]/[`SplitState::joints`];
    /// engine tables are filled in the same order, so each fresh local index
    /// is captured here — the only global↔local conversion site for bodies
    /// (joints convert just below).
    ///
    /// Joints whose ends landed in different solvers stay unmirrored: ball
    /// and distance rows run in the post-step coupling pass
    /// ([`SplitState::couple_cross_joints`]), every other kind is reported
    /// through [`crate::Engine::cross_joint_status`] instead of being
    /// silently dropped.
    pub(super) fn rebuild(&mut self) {
        let timer = Instant::now();
        self.avbd = AvbdEngine::new(self.gravity);
        self.si = SequentialImpulseEngine::new(self.gravity);
        self.xpbd = XpbdEngine::new(self.gravity);
        for b in &mut self.bodies {
            restore_mass(&mut b.body);
            b.local_avbd = None;
            b.local_si = None;
            b.local_xpbd = None;
            if b.owner != SplitOwner::SequentialImpulse && b.owner != SplitOwner::Xpbd {
                let h = LocalAvbdBody::from(self.avbd.add_body(b.body.clone()));
                self.avbd.restore_body_baseline_local(h, b.previous);
                b.local_avbd = Some(h);
            }
            if b.owner != SplitOwner::Avbd && b.owner != SplitOwner::Xpbd {
                let h = LocalSiBody::from(self.si.add_body(b.body.clone()));
                self.si.restore_body_baseline_local(h, b.previous);
                b.local_si = Some(h);
            }
            if b.owner == SplitOwner::Static || b.owner == SplitOwner::Xpbd {
                let h = self.xpbd.add_body(b.body.clone());
                // XPBD keeps no PrevPose baseline (`restore_body_baseline`
                // is a no-op there by design); contact Begin/End derives
                // from the global baseline in `step`, so nothing is lost.
                b.local_xpbd = Some(XpbdBody(h.as_u32()));
            }
        }
        for j in &mut self.joints {
            j.local_avbd = None;
            j.local_si = None;
            j.local_xpbd = None;
        }
        for i in 0..self.joints.len() {
            let j = self.joints[i].state;
            if let (Some(a), Some(b)) = (
                self.bodies[j.a.index()].local_avbd,
                self.bodies[j.b.index()].local_avbd,
            ) {
                let spec = self.local_spec(j.spec, SolverSide::Avbd);
                if let Some(spec) = spec
                    && let Ok(h) = self.avbd.add_joint_local(a, b, spec)
                {
                    self.avbd.restore_joint_reference_local(h, j.reference);
                    self.joints[i].local_avbd = Some(h);
                }
            }
            if let (Some(a), Some(b)) = (
                self.bodies[j.a.index()].local_si,
                self.bodies[j.b.index()].local_si,
            ) {
                let spec = self.local_spec(j.spec, SolverSide::SequentialImpulse);
                if let Some(spec) = spec
                    && let Ok(h) = self.si.add_joint_local(a, b, spec)
                {
                    self.si.restore_joint_reference_local(h, j.reference);
                    self.joints[i].local_si = Some(h);
                }
            }
            // XPBD mirror: both ends present, the kind structurally
            // supported there, gear refs resolvable. Anything else stays
            // unmirrored and is reported via `cross_joint_status` — never
            // half-solved. Cross-solver ball/distance joints intentionally
            // land here too (no engine holds both ends) for the coupling
            // pass below.
            if let (Some(a), Some(b)) = (
                self.bodies[j.a.index()].local_xpbd,
                self.bodies[j.b.index()].local_xpbd,
            ) {
                let spec = self.xpbd_local_spec(j.spec);
                if let Some(spec) = spec
                    && crate::xpbd::xpbd_supports_joint(&spec)
                    && let Ok(h) = self.xpbd.add_joint(a.into(), b.into(), spec)
                {
                    self.xpbd.restore_joint_reference(h, j.reference);
                    self.joints[i].local_xpbd = Some(XpbdJoint(h.as_u32()));
                }
            }
        }
        self.avbd
            .restore_event_state(self.local_events(SolverSide::Avbd));
        self.si
            .restore_event_state(self.local_events(SolverSide::SequentialImpulse));
        self.rebuilds += 1;
        self.timing.rebuild += timer.elapsed();
    }

    /// Remap a global-space gear spec into one engine's local joint space.
    /// Global joint refs enter here; local refs leave. Returns `None` when a
    /// referenced joint is not mirrored into this engine.
    fn local_spec(&self, spec: JointKind, side: SolverSide) -> Option<JointKind> {
        if let JointKind::Gear {
            joint_a,
            joint_b,
            ratio,
        } = spec
        {
            let local = |h: JointHandle| -> Option<JointHandle> {
                let j = self.joints.get(h.index())?;
                if side.is_avbd() {
                    j.local_avbd.map(JointHandle::from)
                } else {
                    j.local_si.map(JointHandle::from)
                }
            };
            Some(JointKind::Gear {
                joint_a: local(joint_a)?,
                joint_b: local(joint_b)?,
                ratio,
            })
        } else {
            Some(spec)
        }
    }

    /// Remap a global-space gear spec into the split XPBD table's joint
    /// space. Same contract as [`SplitState::local_spec`]: global joint refs
    /// enter, XPBD-dense refs leave, `None` when a referenced joint has no
    /// XPBD mirror (cross-solver or natively unsupported gears stay
    /// unmirrored and surface through `cross_joint_status`).
    fn xpbd_local_spec(&self, spec: JointKind) -> Option<JointKind> {
        if let JointKind::Gear {
            joint_a,
            joint_b,
            ratio,
        } = spec
        {
            let local = |h: JointHandle| -> Option<JointHandle> {
                self.joints
                    .get(h.index())?
                    .local_xpbd
                    .map(JointHandle::from)
            };
            Some(JointKind::Gear {
                joint_a: local(joint_a)?,
                joint_b: local(joint_b)?,
                ratio,
            })
        } else {
            Some(spec)
        }
    }

    /// Remap the global completed-step event baseline into one engine's
    /// local body space. Pairs touching bodies absent from this engine are
    /// dropped (the engine never sees them).
    fn local_events(&self, side: SolverSide) -> EventState {
        let local = |pairs: &BTreeSet<(BodyHandle, BodyHandle)>| {
            pairs
                .iter()
                .filter_map(|&(a, b)| {
                    let get = |h: BodyHandle| -> Option<BodyHandle> {
                        let r = self.bodies.get(h.index())?;
                        if side.is_avbd() {
                            r.local_avbd.map(BodyHandle::from)
                        } else {
                            r.local_si.map(BodyHandle::from)
                        }
                    };
                    Some((get(a)?, get(b)?))
                })
                .collect()
        };
        EventState {
            contacts: local(&self.events.contacts),
            triggers: local(&self.events.triggers),
        }
    }

    pub(super) fn pull(&mut self) {
        for b in &mut self.bodies {
            let src = match b.owner {
                SplitOwner::Avbd => b.local_avbd.and_then(|h| self.avbd.get_body_local(h)),
                SplitOwner::SequentialImpulse => b.local_si.and_then(|h| self.si.get_body_local(h)),
                SplitOwner::Xpbd => b
                    .local_xpbd
                    .and_then(|h| self.xpbd.get_body(BodyHandle::from(h))),
                SplitOwner::Static => None, // host owns non-dynamic poses/properties
            };
            if let Some(live) = src {
                b.body = live.clone();
                restore_mass(&mut b.body);
            }
            b.previous = PrevPose {
                pos: b.body.position,
                rot: b.body.orientation,
            };
        }
        let av = self.avbd.joint_snapshots();
        let si_joints = self.si.joint_snapshots();
        let xpbd_joints = self.xpbd.joint_snapshots();
        for j in &mut self.joints {
            let state = j
                .local_avbd
                .and_then(|h| av.get(h.index()))
                .or_else(|| j.local_si.and_then(|h| si_joints.get(h.index())))
                .or_else(|| j.local_xpbd.and_then(|h| xpbd_joints.get(h.index())));
            if let Some(state) = state {
                j.state.reference = state.reference;
            }
        }
    }

    /// Push a host-edited GLOBAL body into every engine mirror (local writes).
    pub(super) fn push_global(&mut self, global: BodyHandle) {
        let Some(b) = self.bodies.get(global.index()) else {
            return;
        };
        if let Some(h) = b.local_avbd {
            self.avbd.wake_body_local(h);
            if let Some(dst) = self.avbd.get_body_mut_local(h) {
                *dst = b.body.clone();
            }
        }
        if let Some(h) = b.local_si {
            self.si.wake_body_local(h);
            if let Some(dst) = self.si.get_body_mut_local(h) {
                *dst = b.body.clone();
            }
        }
        if let Some(h) = b.local_xpbd {
            self.xpbd.wake_body(BodyHandle::from(h));
            if let Some(dst) = self.xpbd.get_body_mut(BodyHandle::from(h)) {
                *dst = b.body.clone();
            }
        }
    }

    /// Wake a GLOBAL body in every engine mirror.
    pub(super) fn wake_global(&mut self, global: BodyHandle) {
        if let Some(b) = self.bodies.get_mut(global.index()) {
            b.sleepy = 0;
            if let Some(local) = b.local_avbd {
                self.avbd.wake_body_local(local);
            }
            if let Some(local) = b.local_si {
                self.si.wake_body_local(local);
            }
            if let Some(local) = b.local_xpbd {
                self.xpbd.wake_body(BodyHandle::from(local));
            }
        }
    }

    /// Ownership is decided before either solver advances. Static anchors
    /// do not join independent islands; gear dependencies join all four sides.
    pub(super) fn route(
        &mut self,
        dt: f32,
        phase: RoutePhase,
        edited: &BTreeSet<BodyHandle>,
    ) -> usize {
        let timer = Instant::now();
        let n = self.bodies.len();
        let old: Vec<_> = self.bodies.iter().map(|b| b.owner).collect();
        for (h, b) in self.bodies.iter_mut().enumerate() {
            if b.body.body_type != BodyType::Dynamic {
                b.owner = SplitOwner::Static;
                b.sleepy = 0;
            } else if let Some(pin) = b.pinned {
                // Host-forced owner: apply immediately, hysteresis never
                // reassigns below, so a joint between differently-pinned
                // bodies stays cross-solver until the host unpins.
                let want = match pin {
                    SolverKind::SequentialImpulse => SplitOwner::SequentialImpulse,
                    SolverKind::Avbd => SplitOwner::Avbd,
                    SolverKind::Xpbd => SplitOwner::Xpbd,
                };
                if b.owner != want {
                    b.owner = want;
                    b.sleepy = 0;
                    restore_mass(&mut b.body);
                }
            } else if b.owner == SplitOwner::Static || b.owner == SplitOwner::Xpbd {
                // Fresh dynamics and unpinned XPBD leftovers rejoin the
                // hysteresis pool on the calm side.
                b.owner = SplitOwner::Avbd;
                b.sleepy = 0;
                restore_mass(&mut b.body);
            }
            if phase.is_tick() && b.owner != SplitOwner::Static {
                if edited.contains(&global_handle(h))
                    || b.body.velocity.length() >= SLEEP_SPEED
                    || b.body.angular_velocity.length() >= SLEEP_SPEED
                    || b.body.torque.length_squared() > 0.0
                {
                    b.sleepy = 0;
                } else {
                    b.sleepy = b.sleepy.saturating_add(1).min(QUIET_STEPS);
                }
            }
        }
        let mut root: Vec<_> = (0..n).collect();
        let bounds: Vec<_> = self
            .bodies
            .iter()
            .map(|b| {
                (b.owner != SplitOwner::Static).then(|| swept_bounds(&b.body, self.gravity, dt))
            })
            .collect();
        for a in 0..n {
            let Some(aa) = &bounds[a] else { continue };
            for (b, bound) in bounds.iter().enumerate().skip(a + 1) {
                let Some(bb) = bound else { continue };
                let (x, y) = (&self.bodies[a].body, &self.bodies[b].body);
                if x.collision_layer & y.collision_mask != 0
                    && y.collision_layer & x.collision_mask != 0
                    && aa.overlaps(bb)
                {
                    link(&mut root, a, b);
                }
            }
        }
        for j in &self.joints {
            let participants = if let JointKind::Gear {
                joint_a, joint_b, ..
            } = j.state.spec
            {
                let a = self.joints[joint_a.index()].state;
                let b = self.joints[joint_b.index()].state;
                [a.a, a.b, b.a, b.b]
            } else {
                [j.state.a, j.state.b, j.state.a, j.state.b]
            };
            let mut first: Option<usize> = None;
            for h in participants {
                let hi = h.index();
                if self.bodies[hi].owner != SplitOwner::Static {
                    if let Some(a) = first {
                        link(&mut root, a, hi);
                    } else {
                        first = Some(hi);
                    }
                }
            }
        }
        let mut calm = vec![true; n];
        let mut active = vec![false; n];
        let mut has_avbd = vec![false; n];
        for (h, b) in self.bodies.iter().enumerate() {
            if b.owner == SplitOwner::Static {
                continue;
            }
            let r = find(&mut root, h);
            calm[r] &= b.sleepy >= QUIET_STEPS;
            active[r] |= edited.contains(&global_handle(h))
                || b.body.velocity.length() > WAKE_SPEED
                || b.body.angular_velocity.length() > WAKE_SPEED
                || b.body.torque.length_squared() > 0.0;
            has_avbd[r] |= b.owner == SplitOwner::Avbd;
        }
        for (h, b) in self.bodies.iter_mut().enumerate() {
            if b.owner == SplitOwner::Static {
                continue;
            }
            let r = find(&mut root, h);
            // Pinned bodies keep their host-forced owner (still counted in
            // the island stats above, so neighbours react to them); only
            // unpinned bodies follow hysteresis.
            if b.pinned.is_none() {
                b.owner = if active[r] {
                    SplitOwner::Avbd
                } else if calm[r] {
                    SplitOwner::SequentialImpulse
                } else if has_avbd[r] {
                    SplitOwner::Avbd
                } else {
                    SplitOwner::SequentialImpulse
                };
            }
        }
        let moved = old
            .iter()
            .zip(&self.bodies)
            .filter(|(old, b)| **old != b.owner)
            .count();
        self.migrated_bodies += moved as u64;
        self.timing.routing += timer.elapsed();
        moved
    }

    /// Drain/map before any rebuild so migration cannot swallow events.
    ///
    /// Engine event queues speak LOCAL handles; `av`/`si_map`/`xp_map`
    /// translate each local index back to its GLOBAL owner. Every other pair
    /// in this file (`now`, `contacts`, `triggers`, `self.events`) is
    /// global. After all three engines advance, cross-solver joints run in
    /// [`SplitState::couple_cross_joints`] (canonical registry order) and
    /// only then does `pull` sync the registry.
    pub(super) fn step(&mut self) -> (Vec<ContactEvent>, Vec<TriggerEvent>) {
        let timer = Instant::now();
        self.avbd.step(DT);
        self.timing.avbd += timer.elapsed();
        let timer = Instant::now();
        self.si.step(DT);
        self.timing.si += timer.elapsed();
        let timer = Instant::now();
        // An empty XPBD table has nothing to integrate (its `step`
        // early-returns anyway); skipping keeps the common two-solver path
        // bit-identical to M3.
        if self.xpbd.body_count() > 0 {
            self.xpbd.step(DT);
        }
        self.timing.xpbd += timer.elapsed();
        let mut av: Vec<Option<BodyHandle>> = vec![None; self.bodies.len()];
        let mut si_map = av.clone();
        let mut xp_map: Vec<Option<BodyHandle>> = vec![None; self.xpbd.body_count()];
        for (global, b) in self.bodies.iter().enumerate() {
            let global = global_handle(global);
            if let Some(h) = b.local_avbd {
                av[h.index()] = Some(global);
            }
            if let Some(h) = b.local_si {
                si_map[h.index()] = Some(global);
            }
            if let Some(h) = b.local_xpbd {
                xp_map[h.index()] = Some(global);
            }
        }
        let mut hits = Vec::new();
        for (events, map) in [
            (self.avbd.drain_contact_events(), &av),
            (self.si.drain_contact_events(), &si_map),
            (self.xpbd.drain_contact_events(), &xp_map),
        ] {
            for event in events {
                let ContactEventKind::Hit {
                    point,
                    mut normal,
                    approach_speed,
                } = event.kind
                else {
                    continue;
                };
                let (Some(a), Some(b)) = (map[event.body_a.index()], map[event.body_b.index()])
                else {
                    continue;
                };
                if a > b {
                    normal = -normal;
                }
                hits.push(ContactEvent {
                    body_a: a.min(b),
                    body_b: a.max(b),
                    kind: ContactEventKind::Hit {
                        point,
                        normal,
                        approach_speed,
                    },
                });
            }
        }
        self.avbd.drain_trigger_events();
        self.si.drain_trigger_events();
        self.xpbd.drain_trigger_events();
        let mut now = EventState::default();
        for (state, map) in [
            (self.avbd.event_state(), &av),
            (self.si.event_state(), &si_map),
            (self.xpbd.event_state(), &xp_map),
        ] {
            let remap = |pairs: BTreeSet<(BodyHandle, BodyHandle)>| {
                pairs
                    .into_iter()
                    .filter_map(|(a, b)| {
                        let (a, b) = (map[a.index()]?, map[b.index()]?);
                        Some((a.min(b), a.max(b)))
                    })
                    .collect::<BTreeSet<_>>()
            };
            now.contacts.extend(remap(state.contacts));
            now.triggers.extend(remap(state.triggers));
        }
        let mut contacts = Vec::new();
        for &(a, b) in now.contacts.symmetric_difference(&self.events.contacts) {
            contacts.push(ContactEvent {
                body_a: a,
                body_b: b,
                kind: if now.contacts.contains(&(a, b)) {
                    ContactEventKind::Begin
                } else {
                    ContactEventKind::End
                },
            });
        }
        contacts.extend(hits);
        contacts.sort_by_key(|e| (e.body_a, e.body_b));
        contacts.dedup_by(|a, b| a == b);
        let triggers = now
            .triggers
            .symmetric_difference(&self.events.triggers)
            .map(|&(a, b)| TriggerEvent {
                body_a: a,
                body_b: b,
                kind: if now.triggers.contains(&(a, b)) {
                    TriggerEventKind::Entered
                } else {
                    TriggerEventKind::Exited
                },
            })
            .collect();
        self.events = now;
        // Cross-solver rows run after every engine advanced and before the
        // registry sync, so `pull` picks up the corrected poses/velocities.
        self.couple_cross_joints();
        self.pull();
        self.steps += 1;
        (contacts, triggers)
    }

    /// Cross-solver coupling pass (v1): every registry joint whose dynamic
    /// ends are owned by different solvers and whose kind carries a
    /// structural row ([`CrossRowKind`]: ball, distance) is solved here —
    /// one PBD-style positional projection plus one velocity impulse row —
    /// directly between the two live engine mirrors.
    ///
    /// Canonical order: global joint index ascending (the same determinism
    /// precedent as the scheduler narrowphase's sorted contact emission),
    /// so reruns and thread-count changes are bit-identical. Both sides get
    /// mass-restore first (sleep may have zeroed the triple — the same M3
    /// rebuild discipline), so a heavy sleeper still carries its weight
    /// instead of reading as infinite mass. Engine sleep flags are left
    /// alone: poses correct every tick, velocities take effect on wake.
    ///
    /// Per-tick frequency only (no substepping — out of scope for v1, like
    /// cross-solver contacts, which would need an O(n²) cross-AABB
    /// broadphase between the tables).
    fn couple_cross_joints(&mut self) {
        let mut work: Vec<CrossWork> = Vec::new();
        for j in &self.joints {
            let (oa, ob) = (
                self.bodies[j.state.a.index()].owner,
                self.bodies[j.state.b.index()].owner,
            );
            if oa == ob || oa == SplitOwner::Static || ob == SplitOwner::Static {
                continue;
            }
            let Some(row) = crate::joint::cross_row_kind(&j.state.spec) else {
                continue;
            };
            let (la, lb) = match j.state.spec {
                JointKind::Ball {
                    local_anchor_a,
                    local_anchor_b,
                }
                | JointKind::Distance {
                    local_anchor_a,
                    local_anchor_b,
                } => (local_anchor_a, local_anchor_b),
                _ => continue,
            };
            work.push(CrossWork {
                a: j.state.a,
                b: j.state.b,
                la,
                lb,
                row,
                rest: j.state.reference.distance.0,
            });
        }
        // Relaxation sweeps over the (tiny) cross set, then one velocity
        // sweep at the final anchors.
        for _ in 0..MAX_STEPS {
            for w in &work {
                self.couple_position(w);
            }
        }
        for w in &work {
            self.couple_velocity(w);
        }
    }

    /// Live engine mirrors of two cross-owned bodies, for the coupling
    /// pass. Owners differ by construction (the caller only queues cross
    /// joints), so the two mirrors always live in different engine fields
    /// and borrow disjointly. `None` when a mirror is missing (a rebuild is
    /// pending — structural edits always set the dirty flag first).
    fn cross_mirrors(
        &mut self,
        a: BodyHandle,
        b: BodyHandle,
    ) -> Option<(&mut RigidBody, &mut RigidBody)> {
        let (oa, ob) = (
            self.bodies.get(a.index())?.owner,
            self.bodies.get(b.index())?.owner,
        );
        match (oa, ob) {
            (SplitOwner::Avbd, SplitOwner::SequentialImpulse) => {
                let la = self.bodies[a.index()].local_avbd?;
                let lb = self.bodies[b.index()].local_si?;
                Some((
                    self.avbd.get_body_mut_local(la)?,
                    self.si.get_body_mut_local(lb)?,
                ))
            }
            (SplitOwner::SequentialImpulse, SplitOwner::Avbd) => {
                let la = self.bodies[a.index()].local_si?;
                let lb = self.bodies[b.index()].local_avbd?;
                Some((
                    self.si.get_body_mut_local(la)?,
                    self.avbd.get_body_mut_local(lb)?,
                ))
            }
            (SplitOwner::Avbd, SplitOwner::Xpbd) => {
                let la = self.bodies[a.index()].local_avbd?;
                let lb = self.bodies[b.index()].local_xpbd?;
                Some((
                    self.avbd.get_body_mut_local(la)?,
                    self.xpbd.get_body_mut(BodyHandle::from(lb))?,
                ))
            }
            (SplitOwner::Xpbd, SplitOwner::Avbd) => {
                let la = self.bodies[a.index()].local_xpbd?;
                let lb = self.bodies[b.index()].local_avbd?;
                Some((
                    self.xpbd.get_body_mut(BodyHandle::from(la))?,
                    self.avbd.get_body_mut_local(lb)?,
                ))
            }
            (SplitOwner::SequentialImpulse, SplitOwner::Xpbd) => {
                let la = self.bodies[a.index()].local_si?;
                let lb = self.bodies[b.index()].local_xpbd?;
                Some((
                    self.si.get_body_mut_local(la)?,
                    self.xpbd.get_body_mut(BodyHandle::from(lb))?,
                ))
            }
            (SplitOwner::Xpbd, SplitOwner::SequentialImpulse) => {
                let la = self.bodies[a.index()].local_xpbd?;
                let lb = self.bodies[b.index()].local_si?;
                Some((
                    self.xpbd.get_body_mut(BodyHandle::from(la))?,
                    self.si.get_body_mut_local(lb)?,
                ))
            }
            _ => None,
        }
    }

    /// One positional projection of a cross row between the live mirrors
    /// (see [`SplitState::couple_cross_joints`]).
    fn couple_position(&mut self, w: &CrossWork) {
        if let Some((a, b)) = self.cross_mirrors(w.a, w.b) {
            solve_cross_position(a, b, w);
        }
    }

    /// One velocity row of a cross joint between the live mirrors.
    fn couple_velocity(&mut self, w: &CrossWork) {
        if let Some((a, b)) = self.cross_mirrors(w.a, w.b) {
            solve_cross_velocity(a, b, w);
        }
    }

    /// Classify a registry joint for
    /// [`crate::Engine::cross_joint_status`]: cross ball/distance joints
    /// couple, cross joints of any other kind are unsupported, same-solver
    /// joints are native — unless the joint holds no mirror anywhere (an
    /// XPBD-unsupported native kind or an unresolvable gear), which is
    /// unsupported too. Every unmirrored joint names its cause; nothing is
    /// silently dropped. `None` for an invalid handle.
    pub(super) fn cross_status(&self, handle: JointHandle) -> Option<CrossJointStatus> {
        let j = self.joints.get(handle.index())?;
        let (oa, ob) = (
            self.bodies.get(j.state.a.index())?.owner,
            self.bodies.get(j.state.b.index())?.owner,
        );
        let name = joint_kind_name(&j.state.spec);
        if oa != ob && oa != SplitOwner::Static && ob != SplitOwner::Static {
            return match crate::joint::cross_row_kind(&j.state.spec) {
                Some(row) => Some(CrossJointStatus::Coupled(row)),
                None => Some(CrossJointStatus::Unsupported {
                    detail: format!(
                        "{name} joint across {oa:?}/{ob:?} solvers has no cross-solver row (ball/distance only)"
                    ),
                }),
            };
        }
        let mirrored = j.local_avbd.is_some() || j.local_si.is_some() || j.local_xpbd.is_some();
        if mirrored {
            return Some(CrossJointStatus::Native);
        }
        let detail = if matches!(j.state.spec, JointKind::Gear { .. }) {
            "gear references joints with no mirror in this solver".to_string()
        } else if oa == SplitOwner::Xpbd || ob == SplitOwner::Xpbd {
            format!("{name} is not supported on the XPBD path")
        } else {
            format!("{name} joint has no solver mirror")
        };
        Some(CrossJointStatus::Unsupported { detail })
    }

    pub(super) fn raycast(
        &self,
        ray: Ray,
        max_dist: f32,
    ) -> Result<Option<RaycastHit>, crate::errors::QueryError> {
        crate::errors::check_ray_input(ray.origin, ray.direction, max_dist)?;
        let mut closest: Option<RaycastHit> = None;
        for (h, b) in self.bodies.iter().enumerate() {
            let inverse = b.body.orientation.conjugate();
            let origin = inverse * (ray.origin - b.body.position);
            let direction = inverse * ray.direction;
            if let Some((distance, normal)) =
                raycast_shape_hit(&b.body.shape, origin, direction, max_dist)
                && closest.as_ref().is_none_or(|old| distance < old.distance)
            {
                closest = Some(RaycastHit {
                    handle: global_handle(h),
                    point: ray.point_at(distance),
                    normal: (b.body.orientation * normal).normalize_or(Vec3::Y),
                    distance,
                });
            }
        }
        Ok(closest)
    }

    pub(super) fn shapecast(&self, shape: &Shape, from: Vec3, to: Vec3) -> Option<RaycastHit> {
        if !from.is_finite() || !to.is_finite() {
            return None;
        }
        let mover = ShapeRef {
            shape,
            pos: from,
            rot: Quat::IDENTITY,
        };
        let targets = self.bodies.iter().enumerate().map(|(h, b)| {
            (
                global_handle(h),
                ShapeRef {
                    shape: &b.body.shape,
                    pos: b.body.position,
                    rot: b.body.orientation,
                },
            )
        });
        cast_shape(mover, to - from, targets).map(|hit| RaycastHit {
            handle: hit.handle,
            point: hit.point,
            normal: hit.normal,
            distance: hit.t,
        })
    }
}
