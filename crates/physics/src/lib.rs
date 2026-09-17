//! Ornis builtin rigid-body physics: dynamics, collision detection, contact
//! and joint solving.
//!
//! The crate is organized as a small CPU pipeline shared by the engine trait
//! ([`engine::PhysicsEngine`]) and its reference implementation
//! ([`engine::BuiltinPhysicsEngine`]):
//!
//! - [`body`] — rigid bodies and their handles/mass model.
//! - [`shape`] — convex primitives with AABB projection and inertia tensors.
//! - [`math`] — geometric queries used by broadphase and raycasts.
//! - `broadphase` — candidate-pair backends and benchmark diagnostics.
//! - [`joint`] — persistent equality constraints (ball/revolute).
//! - [`engine`] — the step pipeline: broadphase → narrowphase → island
//!   partitioning → substepped velocity/position solving, with optional
//!   SIMD-wide (`wide` module) and GPU (`gpu` feature) solver paths.
#![warn(missing_docs)]

mod broadphase;
mod broadphase_tree;

/// AVBD rigid-body engine: second [`engine::PhysicsEngine`] implementation
/// (M1, Genesis-style engine-level modularity).
pub mod avbd;
/// Rigid bodies: [`RigidBody`], mass model and body handles/types.
pub mod body;
pub(crate) mod distance;
/// The physics step pipeline and the [`crate::engine::PhysicsEngine`] trait.
pub mod engine;
/// GJK/EPA fallback narrow phase for cylinder, cone and convex hull pairs.
pub(crate) mod gjk;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;
pub mod joint;
pub mod math;
/// Collision shapes with AABB projection and inertia tensors.
pub mod shape;
/// Trigger overlap event types emitted by the builtin physics engine.
pub mod trigger;
pub(crate) mod wide;

pub use avbd::AvbdEngine;
pub use body::{BodyHandle, BodyType, RigidBody};
pub use broadphase::{BroadPhaseKind, BroadPhaseStats, StepBudget, StepTiming};
pub use engine::{BuiltinPhysicsEngine, PhysicsEngine};
pub use joint::{
    AxisConfig, JointHandle, JointKind, PrismaticLimit, PrismaticMotor, ResolvedJoint,
    RevoluteLimit, RevoluteMotor, WheelSuspension, resolve_joint,
};
pub use math::{AABB, Ray, RaycastHit};
pub use shape::{ConvexHull, Heightfield, Shape, TriMesh};
pub use trigger::{
    CONTACT_BEGIN_SLOP, CONTACT_HIT_THRESHOLD, ContactEvent, ContactEventKind, FractureEvent,
    TriggerEvent, TriggerEventKind,
};

/// Selectable constraint solver (M2 intra-engine modularity, Genesis
/// style): the builtin sequential-impulse engine or the AVBD engine.
/// Same [`PhysicsEngine`] seam, same scenes — the orchestrator
/// ([`Engine`]) migrates bodies and joints across the switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    /// Sequential-impulse engine with islands, sleep and substeps.
    Builtin,
    /// AVBD engine (position-level, single-thread sweep).
    Avbd,
}

/// Routing policy for the M3 multi-solver scene (validated by spikes
/// 001–004, see `spikes/`): one solver per contact island, never
/// teleported cross-solver mirrors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingKind {
    /// One engine owns the whole scene (M1/M2 behavior, default).
    Single,
    /// Contact islands route between Builtin and AVBD with hysteresis
    /// (unanimous-island calm migrates down, any fast body wakes up).
    Islands,
}

/// M3 coupling metrics: the coupling tax, in the books per PLAN M3.
/// `None` from [`Engine::split_metrics`] outside Islands routing.
#[derive(Debug, Clone, Copy, Default)]
pub struct SplitMetrics {
    /// Rebuild migrations triggered by routing decisions (structural
    /// add/remove rebuilds are not counted).
    pub migrations: u64,
    /// Dynamic bodies currently owned by AVBD.
    pub avbd_bodies: usize,
    /// Dynamic bodies currently owned by Builtin.
    pub builtin_bodies: usize,
}

/// Sleep-routing thresholds (spike-validated): the AVBD rest velocity
/// floor sits at ~g*dt (0.16), so the calm band must clear it.
const SPLIT_SLEEP_V: f32 = 0.2;
/// Any body faster than this wakes its island to AVBD immediately.
const SPLIT_WAKE_V: f32 = 0.5;
/// Consecutive calm steps before an island migrates down.
const SPLIT_SLEEP_STEPS: u32 = 30;
/// Proximity band glued onto AABBs for island linkage (pre-touch band,
/// same order as the pair creation distance).
const SPLIT_LINK_MARGIN: f32 = 0.05;

/// Solver orchestrator: owns one engine behind the [`PhysicsEngine`] seam
/// and migrates the full scene (bodies 1:1 in handle order, then joints)
/// across [`SolverKind`] switches, so handles cached by the host stay
/// valid. Warm-start state (contact lambdas, island sleep) does not
/// migrate — only poses, velocities and joint specs do.
///
/// The orchestrator also owns cross-solver policy: the [`FractureEvent`]
/// pass runs inside [`PhysicsEngine::step`] (uniform for both solvers)
/// and its reports wait in [`Engine::drain_fracture_events`].
///
/// M3 adds an opt-in second mode ([`RoutingKind::Islands`]): both engines
/// stay alive and whole contact islands route between them with
/// hysteresis. Global [`BodyHandle`]s stay stable (fixed-order registry);
/// solver-local handles are an internal detail.
pub struct Engine {
    inner: EngineInner,
    fracture_events: Vec<FractureEvent>,
    /// Contact events drained from the inner engine during `step`
    /// (the fracture pass consumes the inner queue, so the orchestrator
    /// re-serves them here — see `drain_contact_events`).
    contact_events: Vec<ContactEvent>,
    /// Solver used in [`RoutingKind::Single`] (and the collapse target
    /// when leaving Islands). Unchanged by routing.
    single_kind: SolverKind,
    /// Active routing policy (M3). `Single` behaves exactly as M1/M2.
    routing: RoutingKind,
    /// Gravity this orchestrator was built (or switched) with.
    gravity: glam::Vec3,
    /// Live M3 registry, present only under Islands routing.
    split: Option<Box<SplitState>>,
    /// Structural changes (add/remove) pending engine rebuild. Single
    /// mode applies them immediately, so this only ever sets in Islands.
    structural_dirty: bool,
    /// Globals touched through `get_body_mut` since the last step; the
    /// owner engine wakes them before stepping (host edits do not wake
    /// sleeping solvers by themselves).
    wake_set: std::collections::BTreeSet<BodyHandle>,
    /// Routing-triggered rebuilds since the last routing change.
    migrations: u64,
}

/// Active solver behind the seam (boxed: both states dwarf the policy).
enum EngineInner {
    /// Sequential-impulse engine with islands, sleep and substeps.
    Builtin(Box<BuiltinPhysicsEngine>),
    /// AVBD engine (position-level, single-thread sweep).
    Avbd(Box<AvbdEngine>),
}

impl Engine {
    /// Empty orchestrator with the given solver and world-space gravity.
    pub fn new(kind: SolverKind, gravity: glam::Vec3) -> Self {
        let inner = match kind {
            SolverKind::Builtin => {
                EngineInner::Builtin(Box::new(BuiltinPhysicsEngine::new(gravity)))
            }
            SolverKind::Avbd => EngineInner::Avbd(Box::new(AvbdEngine::new(gravity))),
        };
        Self {
            inner,
            fracture_events: Vec::new(),
            contact_events: Vec::new(),
            gravity,
            single_kind: kind,
            routing: RoutingKind::Single,
            split: None,
            structural_dirty: false,
            wake_set: std::collections::BTreeSet::new(),
            migrations: 0,
        }
    }

    /// Active solver kind for [`RoutingKind::Single`] (and the collapse
    /// target when leaving Islands). Routing does not change it.
    pub fn kind(&self) -> SolverKind {
        self.single_kind
    }

    /// Active routing policy.
    pub fn routing(&self) -> RoutingKind {
        self.routing
    }

    /// M3 coupling metrics, or `None` outside Islands routing.
    pub fn split_metrics(&self) -> Option<SplitMetrics> {
        let s = self.split.as_ref()?;
        let (mut av, mut bu) = (0, 0);
        for b in &s.bodies {
            match b.owner {
                SplitOwner::Avbd => av += 1,
                SplitOwner::Builtin => bu += 1,
                SplitOwner::Static => {}
            }
        }
        Some(SplitMetrics {
            migrations: self.migrations,
            avbd_bodies: av,
            builtin_bodies: bu,
        })
    }

    /// Switch the active solver, migrating bodies and joints 1:1 (handles
    /// stay valid). Pending trigger/contact/fracture events are dropped
    /// with the old engine. No-op when already on `kind`. Under Islands
    /// routing this collapses the split registry into a fresh `Single`
    /// engine (global order, then joints).
    pub fn set_solver_kind(&mut self, kind: SolverKind, gravity: glam::Vec3) {
        if self.routing == RoutingKind::Islands {
            self.collapse_to_single(kind, gravity);
            return;
        }
        if self.kind() == kind {
            return;
        }
        let (bodies, joints) = match &self.inner {
            EngineInner::Builtin(e) => (e.bodies_snapshot(), e.joint_specs()),
            EngineInner::Avbd(e) => (e.bodies_snapshot(), e.joint_specs()),
        };
        let mut next = Self::new(kind, gravity);
        for body in bodies {
            next.add_body(body);
        }
        for (a, b, spec) in joints {
            // Gears reference joint handles, which are dense 0..n on both
            // sides here (re-added in order), so specs migrate verbatim.
            // A spec the target solver rejects (AVBD has no row model gap
            // left at M1-close, but future kinds may) is skipped, never
            // fatal — same discipline as stale gear references.
            let _ = next.add_joint(a, b, spec);
        }
        *self = next;
    }

    /// Switch the routing policy. `Single` collapses any live split
    /// registry into the current [`Engine::kind`] engine; `Islands`
    /// builds the registry from the current single engine (bodies in
    /// handle order, then joints) and starts both engines. Global
    /// handles stay valid across the switch. No-op when unchanged.
    pub fn set_routing(&mut self, routing: RoutingKind) {
        if self.routing == routing {
            return;
        }
        match routing {
            RoutingKind::Single => {
                let kind = self.single_kind;
                let gravity = self
                    .split
                    .as_ref()
                    .map(|s| s.gravity)
                    .unwrap_or(glam::Vec3::ZERO);
                self.collapse_to_single(kind, gravity);
            }
            RoutingKind::Islands => {
                let gravity = self.gravity;
                let mut split = Box::new(SplitState::new(gravity));
                // Bodies in handle order: global == local on entry.
                let bodies = match &self.inner {
                    EngineInner::Builtin(e) => e.bodies_snapshot(),
                    EngineInner::Avbd(e) => e.bodies_snapshot(),
                };
                for body in bodies {
                    let owner = SplitOwner::of(&body);
                    split.bodies.push(SplitBody {
                        body,
                        owner,
                        local_avbd: None,
                        local_builtin: None,
                        sleepy: 0,
                    });
                }
                // Joints reference dense locals == globals here.
                let specs: Vec<(BodyHandle, BodyHandle, JointKind)> = match &self.inner {
                    EngineInner::Builtin(e) => e.joint_specs(),
                    EngineInner::Avbd(e) => e.joint_specs(),
                };
                for (a, b, spec) in specs {
                    split.joints.push(SplitJoint {
                        a,
                        b,
                        spec,
                        local_avbd: None,
                        local_builtin: None,
                    });
                }
                split.rebuild();
                self.split = Some(split);
                self.routing = RoutingKind::Islands;
                self.migrations = 0;
                self.structural_dirty = false;
                self.wake_set.clear();
            }
        }
    }

    /// Collapse Islands routing into a fresh `Single(kind)` engine from
    /// the registry (global order, then joints with verbatim specs —
    /// globals equal dense locals on a fresh engine).
    fn collapse_to_single(&mut self, kind: SolverKind, gravity: glam::Vec3) {
        let mut next = Self::new(kind, gravity);
        if let Some(s) = self.split.as_ref() {
            for b in &s.bodies {
                next.add_body(b.body.clone());
            }
            for j in &s.joints {
                let _ = next.add_joint(j.a, j.b, j.spec);
            }
        } else {
            let (bodies, joints) = match &self.inner {
                EngineInner::Builtin(e) => (e.bodies_snapshot(), e.joint_specs()),
                EngineInner::Avbd(e) => (e.bodies_snapshot(), e.joint_specs()),
            };
            for body in bodies {
                next.add_body(body);
            }
            for (a, b, spec) in joints {
                let _ = next.add_joint(a, b, spec);
            }
        }
        // `*self = next` resets all orchestrator bookkeeping (events drop
        // with the old engine, same as the M2 switch above).
        *self = next;
    }

    /// Drain fracture reports since the last call (see [`FractureEvent`]).
    pub fn drain_fracture_events(&mut self) -> Vec<FractureEvent> {
        std::mem::take(&mut self.fracture_events)
    }

    /// Rebuild split engines from the registry when structural changes
    /// (add/remove) are pending. Cheap flag check on the hot path.
    fn split_ensure_built(&mut self) {
        if self.routing != RoutingKind::Islands {
            return;
        }
        if self.structural_dirty {
            if let Some(s) = self.split.as_mut() {
                s.rebuild();
            }
            self.structural_dirty = false;
        }
    }

    /// One Islands step: wake host dirties, step both engines in fixed
    /// order (AVBD then Builtin — determinism needs a fixed order, either
    /// would do), pull registry truth, route islands (rebuild + count on
    /// change), then the split fracture pass.
    fn split_step(&mut self, dt: f32) {
        self.split_ensure_built();
        let dirties: Vec<BodyHandle> = std::mem::take(&mut self.wake_set).into_iter().collect();
        let changed: bool;
        if let Some(s) = self.split.as_mut() {
            for g in dirties {
                s.push_global(g);
            }
            s.avbd.step(dt);
            s.builtin.step(dt);
            s.pull();
            if s.route() {
                s.rebuild();
                changed = true;
            } else {
                changed = false;
            }
        } else {
            return;
        }
        if changed {
            self.migrations += 1;
        }
        self.fracture_split();
    }

    /// Inverse maps local engine handle -> global, sized tightly.
    fn split_inverse(s: &SplitState) -> (Vec<Option<BodyHandle>>, Vec<Option<BodyHandle>>) {
        let mut av = vec![None; s.bodies.len()];
        let mut bu = vec![None; s.bodies.len()];
        for (g, b) in s.bodies.iter().enumerate() {
            if let Some(h) = b.local_avbd
                && h < av.len()
            {
                av[h] = Some(g);
            }
            if let Some(h) = b.local_builtin
                && h < bu.len()
            {
                bu[h] = Some(g);
            }
        }
        // Compact to dense local ranges (rebuilds add in global order, so
        // locals are dense 0..k per engine; truncate defensively).
        let trim = |v: &mut Vec<Option<BodyHandle>>| {
            while v.last() == Some(&None) {
                v.pop();
            }
        };
        trim(&mut av);
        trim(&mut bu);
        (av, bu)
    }

    fn contact_kind_rank(kind: &ContactEventKind) -> u8 {
        match kind {
            ContactEventKind::Begin => 0,
            ContactEventKind::End => 1,
            ContactEventKind::Hit { .. } => 2,
        }
    }

    /// Query both engines, remap hit handles to globals, nearest wins.
    /// Bodies with pending (unbuilt) structural changes are invisible —
    /// step first after add/remove for exact queries.
    fn split_raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit> {
        let s = self.split.as_ref()?;
        let (ia, ib) = Self::split_inverse(s);
        let global = |hit: RaycastHit, inv: &[Option<BodyHandle>]| -> Option<RaycastHit> {
            let g = inv.get(hit.handle).copied().flatten()?;
            Some(RaycastHit { handle: g, ..hit })
        };
        let ha = s.avbd.raycast(ray, max_dist).and_then(|h| global(h, &ia));
        let hb = s
            .builtin
            .raycast(ray, max_dist)
            .and_then(|h| global(h, &ib));
        match (ha, hb) {
            (Some(a), Some(b)) => Some(if a.distance <= b.distance { a } else { b }),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    fn split_shapecast(
        &self,
        shape: &Shape,
        from: glam::Vec3,
        to: glam::Vec3,
    ) -> Option<RaycastHit> {
        let s = self.split.as_ref()?;
        let (ia, ib) = Self::split_inverse(s);
        let global = |hit: RaycastHit, inv: &[Option<BodyHandle>]| -> Option<RaycastHit> {
            let g = inv.get(hit.handle).copied().flatten()?;
            Some(RaycastHit { handle: g, ..hit })
        };
        let ha = s
            .avbd
            .shapecast(shape, from, to)
            .and_then(|h| global(h, &ia));
        let hb = s
            .builtin
            .shapecast(shape, from, to)
            .and_then(|h| global(h, &ib));
        match (ha, hb) {
            (Some(a), Some(b)) => Some(if a.distance <= b.distance { a } else { b }),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    /// Split fracture pass: drain both engines, remap to globals, merge in
    /// canonical order, then run the shared Hit logic on the registry
    /// (parent removed, halves appended, immediate rebuild so halves step
    /// next round — same timing as the single-engine pass).
    fn fracture_split(&mut self) {
        let Some(s) = self.split.as_mut() else {
            return;
        };
        let (ia, ib) = Self::split_inverse(s);
        let mut contacts: Vec<ContactEvent> = Vec::new();
        for ev in s.avbd.drain_contact_events() {
            if let (Some(a), Some(b)) = (
                ia.get(ev.body_a).copied().flatten(),
                ia.get(ev.body_b).copied().flatten(),
            ) {
                contacts.push(ContactEvent {
                    body_a: a,
                    body_b: b,
                    kind: ev.kind,
                });
            }
        }
        for ev in s.builtin.drain_contact_events() {
            if let (Some(a), Some(b)) = (
                ib.get(ev.body_a).copied().flatten(),
                ib.get(ev.body_b).copied().flatten(),
            ) {
                contacts.push(ContactEvent {
                    body_a: a,
                    body_b: b,
                    kind: ev.kind,
                });
            }
        }
        contacts.sort_by(|a, b| {
            (a.body_a, a.body_b, Self::contact_kind_rank(&a.kind)).cmp(&(
                b.body_a,
                b.body_b,
                Self::contact_kind_rank(&b.kind),
            ))
        });
        // Hit logic on globals (mirrors `fracture_pass` candidate rules).
        let mut candidates = std::collections::BTreeSet::new();
        for ev in &contacts {
            if let ContactEventKind::Hit { approach_speed, .. } = ev.kind {
                for h in [ev.body_a, ev.body_b] {
                    if let Some(body) = s.bodies.get(h).map(|r| &r.body)
                        && body.body_type == BodyType::Dynamic
                        && approach_speed >= body.fracture_impact_speed
                    {
                        candidates.insert(h);
                    }
                }
            }
        }
        let mut ordered: Vec<BodyHandle> = candidates.into_iter().collect();
        ordered.sort_unstable_by(|a, b| b.cmp(a));
        let fractured = !ordered.is_empty();
        for parent in ordered {
            let Some(body) = s.bodies.get(parent).map(|r| r.body.clone()) else {
                continue;
            };
            let Some([pa, pb]) = Self::split_box(&body) else {
                continue;
            };
            Self::split_remove_body(s, parent);
            // Halves inherit the parent's owner (same island by proximity).
            let owner = SplitOwner::of(&pa);
            for half in [pa, pb] {
                s.bodies.push(SplitBody {
                    body: half,
                    owner,
                    local_avbd: None,
                    local_builtin: None,
                    sleepy: 0,
                });
            }
            let n = s.bodies.len();
            self.fracture_events.push(FractureEvent {
                parent,
                pieces: [n - 2, n - 1],
            });
        }
        if fractured {
            // Halves must exist before the next step: rebuild eagerly so
            // the fresh pieces report sane locals immediately (the lazy
            // `split_ensure_built` would also catch it).
            s.rebuild();
        }
        self.contact_events.extend(contacts);
    }

    /// Registry body removal with joint remap (swap_remove discipline,
    /// same as the engines: later handles shift, refs are patched).
    fn split_remove_body(s: &mut SplitState, handle: BodyHandle) {
        let n = s.bodies.len();
        if handle >= n {
            return;
        }
        // Drop joints touching the removed body; patch refs to the moved one.
        let mut j = 0;
        while j < s.joints.len() {
            let (a, b) = (s.joints[j].a, s.joints[j].b);
            if a == handle || b == handle {
                Self::split_remove_joint(s, j);
            } else {
                if a == n - 1 {
                    s.joints[j].a = handle;
                }
                if b == n - 1 {
                    s.joints[j].b = handle;
                }
                j += 1;
            }
        }
        s.bodies.swap_remove(handle);
    }

    /// Registry joint removal with gear-ref remap.
    fn split_remove_joint(s: &mut SplitState, handle: usize) {
        let n = s.joints.len();
        if handle >= n {
            return;
        }
        s.joints.swap_remove(handle);
        // Patch gear references to the moved joint (n-1 -> handle).
        for j in &mut s.joints {
            if let JointKind::Gear {
                joint_a,
                joint_b,
                ratio: _,
            } = &mut j.spec
            {
                if *joint_a == n - 1 {
                    *joint_a = handle;
                }
                if *joint_b == n - 1 {
                    *joint_b = handle;
                }
            }
        }
    }

    /// Split a dynamic box along its longest axis into two halves with
    /// conserved mass, velocity and material (halves inherit the fracture
    /// threshold, so rubble may fracture again). Returns `None` for
    /// non-boxes, non-dynamics and massless bodies.
    fn split_box(parent: &RigidBody) -> Option<[RigidBody; 2]> {
        let Shape::Box { half_extents: h } = parent.shape else {
            return None;
        };
        if parent.body_type != BodyType::Dynamic || parent.mass <= 0.0 {
            return None;
        }
        let (axis, ext) = if h.x >= h.y && h.x >= h.z {
            (glam::Vec3::X, h.x)
        } else if h.y >= h.z {
            (glam::Vec3::Y, h.y)
        } else {
            (glam::Vec3::Z, h.z)
        };
        let mut h2 = h.to_array();
        h2[if axis == glam::Vec3::X {
            0
        } else if axis == glam::Vec3::Y {
            1
        } else {
            2
        }] = ext * 0.5;
        let h2 = glam::Vec3::from_array(h2);
        let off = (parent.orientation * axis).normalize_or(axis) * (ext * 0.5);
        let mut halves = [parent.clone(), parent.clone()];
        for (half, s) in halves.iter_mut().zip([-1.0, 1.0]) {
            half.shape = Shape::Box { half_extents: h2 };
            half.position = parent.position + off * s;
            half.mass = parent.mass * 0.5;
            half.inv_mass = 1.0 / half.mass;
            half.inertia = half.shape.inertia(half.mass);
        }
        Some(halves)
    }

    /// Fracture pass: every [`ContactEventKind::Hit`] whose approach
    /// speed reaches a dynamic box's `fracture_impact_speed` splits it.
    /// Candidates are removed in DESCENDING handle order — `swap_remove`
    /// only ever moves the tail into the removed slot, so smaller handles
    /// (processed later) are never invalidated. Joints on the parent die
    /// with the removal on both solvers; halves start joint-free, awake,
    /// with zeroed quiet timers.
    fn fracture_pass(&mut self) {
        use ContactEventKind::Hit;
        let mut candidates = std::collections::BTreeSet::new();
        let inner = &mut self.inner;
        // The inner queue is consumed here, so every event is stashed for
        // re-serve: fracture must never swallow the host's contact stream.
        let drained: Vec<ContactEvent> = match inner {
            EngineInner::Builtin(e) => e.drain_contact_events(),
            EngineInner::Avbd(e) => e.drain_contact_events(),
        };
        self.contact_events.extend(drained.iter().cloned());
        let hits: Vec<(BodyHandle, BodyHandle, f32)> = drained
            .into_iter()
            .filter_map(|ev| match ev.kind {
                Hit { approach_speed, .. } => Some((ev.body_a, ev.body_b, approach_speed)),
                _ => None,
            })
            .collect();
        let get = |inner: &EngineInner, h: BodyHandle| -> Option<RigidBody> {
            match inner {
                EngineInner::Builtin(e) => e.get_body(h).cloned(),
                EngineInner::Avbd(e) => e.get_body(h).cloned(),
            }
        };
        for (a, b, approach) in hits {
            for h in [a, b] {
                if let Some(body) = get(inner, h)
                    && body.body_type == BodyType::Dynamic
                    && approach >= body.fracture_impact_speed
                {
                    candidates.insert(h);
                }
            }
        }
        let mut ordered: Vec<BodyHandle> = candidates.into_iter().collect();
        ordered.sort_unstable_by(|a, b| b.cmp(a));
        for parent in ordered {
            let Some(body) = get(inner, parent) else {
                continue;
            };
            let Some([pa, pb]) = Self::split_box(&body) else {
                continue;
            };
            match inner {
                EngineInner::Builtin(e) => e.remove_body(parent),
                EngineInner::Avbd(e) => e.remove_body(parent),
            }
            let (ha, hb) = match inner {
                EngineInner::Builtin(e) => (e.add_body(pa), e.add_body(pb)),
                EngineInner::Avbd(e) => (e.add_body(pa), e.add_body(pb)),
            };
            self.fracture_events.push(FractureEvent {
                parent,
                pieces: [ha, hb],
            });
        }
    }
}

impl PhysicsEngine for Engine {
    fn step(&mut self, dt: f32) {
        if self.routing == RoutingKind::Islands {
            self.split_step(dt);
            return;
        }
        match &mut self.inner {
            EngineInner::Builtin(e) => e.step(dt),
            EngineInner::Avbd(e) => e.step(dt),
        }
        self.fracture_pass();
    }

    fn add_body(&mut self, body: RigidBody) -> BodyHandle {
        if self.routing == RoutingKind::Islands {
            let s = self.split.as_mut().expect("split state live under Islands");
            let h = s.bodies.len();
            let owner = SplitOwner::of(&body);
            s.bodies.push(SplitBody {
                body,
                owner,
                local_avbd: None,
                local_builtin: None,
                sleepy: 0,
            });
            self.structural_dirty = true;
            return h;
        }
        match &mut self.inner {
            EngineInner::Builtin(e) => e.add_body(body),
            EngineInner::Avbd(e) => e.add_body(body),
        }
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        if self.routing == RoutingKind::Islands {
            if let Some(s) = self.split.as_mut() {
                let n = s.bodies.len();
                Self::split_remove_body(s, handle);
                // swap_remove shifts globals: the deleted handle's entry
                // dies, the moved tail (n-1 -> handle) keeps its entry.
                self.wake_set.remove(&handle);
                if handle < n.saturating_sub(1) && self.wake_set.remove(&(n - 1)) {
                    self.wake_set.insert(handle);
                }
            }
            self.structural_dirty = true;
            return;
        }
        match &mut self.inner {
            EngineInner::Builtin(e) => e.remove_body(handle),
            EngineInner::Avbd(e) => e.remove_body(handle),
        }
    }

    fn get_body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        if self.routing == RoutingKind::Islands {
            return self.split.as_ref()?.bodies.get(handle).map(|r| &r.body);
        }
        match &self.inner {
            EngineInner::Builtin(e) => e.get_body(handle),
            EngineInner::Avbd(e) => e.get_body(handle),
        }
    }

    fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        if self.routing == RoutingKind::Islands {
            let s = self.split.as_mut()?;
            let r = s.bodies.get_mut(handle)?;
            // Host edit: wake the owner's copy before the next step (an
            // edit alone must not leave a stale sleeping copy behind).
            self.wake_set.insert(handle);
            return Some(&mut r.body);
        }
        match &mut self.inner {
            EngineInner::Builtin(e) => e.get_body_mut(handle),
            EngineInner::Avbd(e) => e.get_body_mut(handle),
        }
    }

    fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        kind: JointKind,
    ) -> Option<JointHandle> {
        if self.routing == RoutingKind::Islands {
            let s = self.split.as_mut().expect("split state live under Islands");
            // Validate globals (gear refs must be live global joint ids).
            if body_a >= s.bodies.len() || body_b >= s.bodies.len() || body_a == body_b {
                return None;
            }
            if let JointKind::Gear {
                joint_a, joint_b, ..
            } = &kind
                && (*joint_a >= s.joints.len() || *joint_b >= s.joints.len())
            {
                return None;
            }
            let h = s.joints.len();
            s.joints.push(SplitJoint {
                a: body_a,
                b: body_b,
                spec: kind,
                local_avbd: None,
                local_builtin: None,
            });
            self.structural_dirty = true;
            return Some(h);
        }
        match &mut self.inner {
            EngineInner::Builtin(e) => e.add_joint(body_a, body_b, kind),
            EngineInner::Avbd(e) => e.add_joint(body_a, body_b, kind),
        }
    }

    fn remove_joint(&mut self, handle: JointHandle) {
        if self.routing == RoutingKind::Islands {
            if let Some(s) = self.split.as_mut() {
                Self::split_remove_joint(s, handle);
            }
            self.structural_dirty = true;
            return;
        }
        match &mut self.inner {
            EngineInner::Builtin(e) => e.remove_joint(handle),
            EngineInner::Avbd(e) => e.remove_joint(handle),
        }
    }

    fn raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit> {
        if self.routing == RoutingKind::Islands {
            // Pending structural changes would hide new bodies: build first
            // (read path; `&self` so clone-on-write is overkill — the next
            // step rebuilds anyway. Bodies added but not yet built are
            // invisible to this query, documented).
            return self.split_raycast(ray, max_dist);
        }
        match &self.inner {
            EngineInner::Builtin(e) => e.raycast(ray, max_dist),
            EngineInner::Avbd(e) => e.raycast(ray, max_dist),
        }
    }

    fn shapecast(&self, shape: &Shape, from: glam::Vec3, to: glam::Vec3) -> Option<RaycastHit> {
        if self.routing == RoutingKind::Islands {
            return self.split_shapecast(shape, from, to);
        }
        match &self.inner {
            EngineInner::Builtin(e) => e.shapecast(shape, from, to),
            EngineInner::Avbd(e) => e.shapecast(shape, from, to),
        }
    }

    fn drain_trigger_events(&mut self) -> Vec<TriggerEvent> {
        if self.routing == RoutingKind::Islands {
            // Stale engines would misreport: build pending structure first.
            self.split_ensure_built();
            if let Some(s) = self.split.as_mut() {
                let (ia, ib) = Self::split_inverse(s);
                let mut out = Vec::new();
                for ev in s.avbd.drain_trigger_events() {
                    if let (Some(a), Some(b)) = (
                        ia.get(ev.body_a).copied().flatten(),
                        ia.get(ev.body_b).copied().flatten(),
                    ) {
                        out.push(TriggerEvent {
                            body_a: a,
                            body_b: b,
                            kind: ev.kind,
                        });
                    }
                }
                for ev in s.builtin.drain_trigger_events() {
                    if let (Some(a), Some(b)) = (
                        ib.get(ev.body_a).copied().flatten(),
                        ib.get(ev.body_b).copied().flatten(),
                    ) {
                        out.push(TriggerEvent {
                            body_a: a,
                            body_b: b,
                            kind: ev.kind,
                        });
                    }
                }
                out.sort_by(|a, b| {
                    (a.body_a, a.body_b, a.kind as u8).cmp(&(b.body_a, b.body_b, b.kind as u8))
                });
                return out;
            }
            return Vec::new();
        }
        match &mut self.inner {
            EngineInner::Builtin(e) => e.drain_trigger_events(),
            EngineInner::Avbd(e) => e.drain_trigger_events(),
        }
    }

    fn drain_contact_events(&mut self) -> Vec<ContactEvent> {
        // Served from the orchestrator stash (filled by `step`): the
        // fracture pass sits between the inner queue and the host.
        std::mem::take(&mut self.contact_events)
    }

    fn wake_body(&mut self, handle: BodyHandle) {
        if self.routing == RoutingKind::Islands {
            if let Some(s) = self.split.as_mut() {
                s.wake_global(handle);
            }
            return;
        }
        match &mut self.inner {
            EngineInner::Builtin(e) => e.wake_body(handle),
            EngineInner::Avbd(e) => e.wake_body(handle),
        }
    }
}

// ---------------------------------------------------------------------------
// M3 Islands routing: fixed-order global registry, both engines alive.
// ---------------------------------------------------------------------------

/// Which solver owns a registry body. Statics live natively in both
/// engines and never migrate or need proxies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SplitOwner {
    Static,
    Avbd,
    Builtin,
}

impl SplitOwner {
    fn of(body: &RigidBody) -> Self {
        if body.body_type == BodyType::Dynamic {
            // New bodies arrive awake (the always-correct fallback is AVBD;
            // calm islands migrate down from there).
            SplitOwner::Avbd
        } else {
            SplitOwner::Static
        }
    }
}

/// One global body: index in [`SplitState::bodies`] IS the global
/// [`BodyHandle`], stable across rebuild migrations.
struct SplitBody {
    body: RigidBody,
    owner: SplitOwner,
    local_avbd: Option<BodyHandle>,
    local_builtin: Option<BodyHandle>,
    sleepy: u32,
}

/// One global joint: `a`/`b` are global body handles, gear references
/// are global JOINT ids (remapped to fresh locals on every rebuild).
struct SplitJoint {
    a: BodyHandle,
    b: BodyHandle,
    spec: JointKind,
    local_avbd: Option<JointHandle>,
    local_builtin: Option<JointHandle>,
}

/// Live M3 registry (Newton-style explicit ownership over a shared body
/// set). Bodies never move slots; engines are rebuilt deterministically
/// from it, so global handles survive migrations.
struct SplitState {
    builtin: BuiltinPhysicsEngine,
    avbd: AvbdEngine,
    gravity: glam::Vec3,
    bodies: Vec<SplitBody>,
    joints: Vec<SplitJoint>,
}

fn split_find(root: &mut [usize], x: usize) -> usize {
    if root[x] != x {
        root[x] = split_find(root, root[x]);
    }
    root[x]
}

/// Restore the mass model of a snapshot body (zombie rule): bodies
/// pulled from a sleeping engine carry zeroed inverse mass/inertia, and
/// a fresh engine starts them awake — unrestored they are unsolvable
/// forever (or tumble on zero inertia). Mirrors `wake_body`.
fn split_restore_mass(body: &mut RigidBody) {
    if body.body_type == BodyType::Dynamic && body.inv_mass <= 0.0 {
        body.inv_mass = 1.0 / body.mass;
        body.inertia = body.shape.inertia(body.mass);
    }
}

impl SplitState {
    fn new(gravity: glam::Vec3) -> Self {
        Self {
            builtin: BuiltinPhysicsEngine::new(gravity),
            avbd: AvbdEngine::new(gravity),
            gravity,
            bodies: Vec::new(),
            joints: Vec::new(),
        }
    }

    /// Deterministic rebuild from the registry (global order): statics go
    /// to both engines, dynamics to their owner; joints re-added in global
    /// order with remapped local references.
    fn rebuild(&mut self) {
        self.avbd = AvbdEngine::new(self.gravity);
        self.builtin = BuiltinPhysicsEngine::new(self.gravity);
        for b in &mut self.bodies {
            b.local_avbd = None;
            b.local_builtin = None;
        }
        for (g, b) in self.bodies.iter_mut().enumerate() {
            let _ = g;
            match b.owner {
                SplitOwner::Static => {
                    b.local_avbd = Some(self.avbd.add_body(b.body.clone()));
                    b.local_builtin = Some(self.builtin.add_body(b.body.clone()));
                }
                SplitOwner::Avbd => {
                    b.local_avbd = Some(self.avbd.add_body(b.body.clone()));
                }
                SplitOwner::Builtin => {
                    b.local_builtin = Some(self.builtin.add_body(b.body.clone()));
                }
            }
        }
        for j in &mut self.joints {
            j.local_avbd = None;
            j.local_builtin = None;
        }
        for ji in 0..self.joints.len() {
            let (a, b, spec) = {
                let j = &self.joints[ji];
                (j.a, j.b, j.spec)
            };
            // Endpoints pinned to one island share one solver by
            // construction (joint links join the union-find); a split
            // joint is skipped, never fatal (M2 discipline).
            let on_avbd = self
                .bodies
                .get(a)
                .is_some_and(|x| x.owner != SplitOwner::Builtin)
                && self
                    .bodies
                    .get(b)
                    .is_some_and(|x| x.owner != SplitOwner::Builtin);
            let on_builtin = self
                .bodies
                .get(a)
                .is_some_and(|x| x.owner != SplitOwner::Avbd)
                && self
                    .bodies
                    .get(b)
                    .is_some_and(|x| x.owner != SplitOwner::Avbd);
            if on_avbd && !on_builtin {
                let la = self.bodies[a].local_avbd.unwrap();
                let lb = self.bodies[b].local_avbd.unwrap();
                let local_spec = match spec {
                    JointKind::Gear {
                        joint_a,
                        joint_b,
                        ratio,
                    } => {
                        let la_ref = self.joints.get(joint_a).and_then(|j| j.local_avbd);
                        let lb_ref = self.joints.get(joint_b).and_then(|j| j.local_avbd);
                        match (la_ref, lb_ref) {
                            (Some(x), Some(y)) => JointKind::Gear {
                                joint_a: x,
                                joint_b: y,
                                ratio,
                            },
                            _ => continue,
                        }
                    }
                    other => other,
                };
                // A rejected spec keeps later gear refs dangling-safe:
                // gear remap above yields None and skips (same discipline
                // as stale references).
                self.joints[ji].local_avbd = self.avbd.add_joint(la, lb, local_spec);
            } else if on_builtin && !on_avbd {
                let la = self.bodies[a].local_builtin.unwrap();
                let lb = self.bodies[b].local_builtin.unwrap();
                let local_spec = match spec {
                    JointKind::Gear {
                        joint_a,
                        joint_b,
                        ratio,
                    } => {
                        let la_ref = self.joints.get(joint_a).and_then(|j| j.local_builtin);
                        let lb_ref = self.joints.get(joint_b).and_then(|j| j.local_builtin);
                        match (la_ref, lb_ref) {
                            (Some(x), Some(y)) => JointKind::Gear {
                                joint_a: x,
                                joint_b: y,
                                ratio,
                            },
                            _ => continue,
                        }
                    }
                    other => other,
                };
                self.joints[ji].local_builtin = self.builtin.add_joint(la, lb, local_spec);
            }
        }
    }

    /// Pull engine truth back into the registry (per-owner locals).
    fn pull(&mut self) {
        for b in &mut self.bodies {
            let src = match b.owner {
                SplitOwner::Static | SplitOwner::Avbd => {
                    b.local_avbd.and_then(|h| self.avbd.get_body(h).cloned())
                }
                SplitOwner::Builtin => b
                    .local_builtin
                    .and_then(|h| self.builtin.get_body(h).cloned()),
            };
            if let Some(mut live) = src {
                // Zombie rule: a sleeping engine may zero inverse mass on
                // its copy — never let that poison registry truth.
                split_restore_mass(&mut live);
                b.body = live;
            }
        }
    }

    /// Push registry truth into the owning engine copy and wake it (host
    /// edits land on the registry; without this the engines would step
    /// stale copies and `pull` would clobber the edit — the spike-002
    /// "host writes into a sleeping engine" trap, fixed structurally).
    fn push_global(&mut self, global: BodyHandle) {
        let Some(rec) = self.bodies.get(global) else {
            return;
        };
        let (owner, body) = (rec.owner, rec.body.clone());
        match owner {
            SplitOwner::Avbd => {
                if let Some(h) = rec.local_avbd {
                    if let Some(dst) = self.avbd.get_body_mut(h) {
                        *dst = body;
                    }
                    self.avbd.wake_body(h);
                }
            }
            SplitOwner::Builtin => {
                if let Some(h) = rec.local_builtin {
                    if let Some(dst) = self.builtin.get_body_mut(h) {
                        *dst = body;
                    }
                    self.builtin.wake_body(h);
                }
            }
            SplitOwner::Static => {
                // Statics live in both engines natively; push to both.
                if let Some(h) = rec.local_avbd
                    && let Some(dst) = self.avbd.get_body_mut(h)
                {
                    *dst = body.clone();
                }
                if let Some(h) = rec.local_builtin
                    && let Some(dst) = self.builtin.get_body_mut(h)
                {
                    *dst = body;
                }
            }
        }
    }

    /// Wake a global body in its owning engine (no-op for statics and
    /// unknown globals).
    fn wake_global(&mut self, global: BodyHandle) {
        let Some(b) = self.bodies.get(global) else {
            return;
        };
        match b.owner {
            SplitOwner::Avbd => {
                if let Some(h) = b.local_avbd {
                    self.avbd.wake_body(h);
                }
            }
            SplitOwner::Builtin => {
                if let Some(h) = b.local_builtin {
                    self.builtin.wake_body(h);
                }
            }
            SplitOwner::Static => {}
        }
    }

    /// Contact islands over the registry: dynamic–dynamic AABB overlap
    /// (statics never link — they live in both engines anyway) plus
    /// joint links (a joint pins its endpoints to one solver, so
    /// cross-solver joint rows never exist). Returns true when any owner
    /// changed (caller rebuilds and counts a migration).
    fn route(&mut self) -> bool {
        let n = self.bodies.len();
        let mut root: Vec<usize> = (0..n).collect();
        let link = |root: &mut Vec<usize>, a: usize, b: usize| {
            let (ra, rb) = (split_find(root, a), split_find(root, b));
            root[ra] = rb;
        };
        let aabbs: Vec<Option<AABB>> = self
            .bodies
            .iter()
            .map(|b| {
                if b.owner == SplitOwner::Static {
                    return None;
                }
                let mut ab = b.body.shape.aabb(b.body.position, b.body.orientation);
                let m = glam::Vec3::splat(SPLIT_LINK_MARGIN);
                ab.min -= m;
                ab.max += m;
                Some(ab)
            })
            .collect();
        for (a, ab) in aabbs.iter().enumerate() {
            let Some(ab) = ab else { continue };
            for (b, bb) in aabbs.iter().enumerate().skip(a + 1) {
                let Some(bb) = bb else { continue };
                if ab.overlaps(bb) {
                    link(&mut root, a, b);
                }
            }
        }
        for j in &self.joints {
            if j.a < n && j.b < n {
                link(&mut root, j.a, j.b);
            }
        }
        // Per-body calm counters (hysteresis history, rides on the record
        // so removals never desync it).
        for b in self.bodies.iter_mut() {
            if b.owner == SplitOwner::Static {
                continue;
            }
            let v = b.body.velocity.length();
            if v > SPLIT_WAKE_V {
                b.sleepy = 0;
            } else if v < SPLIT_SLEEP_V {
                b.sleepy += 1;
            } else {
                b.sleepy = 0;
            }
        }
        // Island calm <=> every dynamic member sleepy long enough. Wake
        // is per-body immediate; sleep is per-island unanimous (a calm
        // body never drags an awake island down, an awake body never
        // lets a calm island migrate).
        let mut changed = false;
        for g in 0..n {
            if self.bodies[g].owner == SplitOwner::Static {
                continue;
            }
            let r = split_find(&mut root, g);
            let mut calm = true;
            for h in 0..n {
                if self.bodies[h].owner != SplitOwner::Static
                    && split_find(&mut root, h) == r
                    && self.bodies[h].sleepy < SPLIT_SLEEP_STEPS
                {
                    calm = false;
                    break;
                }
            }
            let want = if self.bodies[g].body.velocity.length() > SPLIT_WAKE_V {
                SplitOwner::Avbd
            } else if calm {
                SplitOwner::Builtin
            } else {
                SplitOwner::Avbd
            };
            if want != self.bodies[g].owner {
                self.bodies[g].owner = want;
                changed = true;
            }
        }
        changed
    }
}
