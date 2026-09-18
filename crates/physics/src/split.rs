//! Deterministic island ownership over a shared body/joint registry.
//! Routing precedes integration; rebuilds preserve assembly references and
//! completed-step event baselines, never use kinematic collision proxies.

use std::collections::BTreeSet;
use std::time::Instant;

use glam::{Quat, Vec3};

use crate::broadphase::PrevPose;
use crate::distance::{ShapeRef, cast_shape};
use crate::engine::raycast_shape_hit;
use crate::migration::{EventState, JointSnapshot};
use crate::{
    AABB, AvbdEngine, BodyHandle, BodyType, BuiltinPhysicsEngine, ContactEvent, ContactEventKind,
    JointKind, PhysicsEngine, Ray, RaycastHit, RigidBody, Shape, SolverKind, SplitTiming,
    TriggerEvent, TriggerEventKind,
};

pub(super) const DT: f32 = 1.0 / 60.0;
pub(super) const MAX_STEPS: usize = 4;
const SLEEP_SPEED: f32 = 0.2;
const WAKE_SPEED: f32 = 0.5;
const QUIET_STEPS: u32 = 30;
const LINK_MARGIN: f32 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SplitOwner {
    Static,
    Avbd,
    Builtin,
}

impl SplitOwner {
    fn preferred(kind: SolverKind) -> Self {
        match kind {
            SolverKind::Avbd => Self::Avbd,
            SolverKind::Builtin => Self::Builtin,
        }
    }
}

pub(super) struct SplitBody {
    pub body: RigidBody,
    pub owner: SplitOwner,
    pub local_avbd: Option<BodyHandle>,
    pub local_builtin: Option<BodyHandle>,
    pub sleepy: u32,
    pub previous: PrevPose,
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
            local_avbd: None,
            local_builtin: None,
            sleepy: 0,
            previous,
        }
    }
}

pub(super) struct SplitJoint {
    pub state: JointSnapshot,
    pub local_avbd: Option<usize>,
    pub local_builtin: Option<usize>,
}

impl SplitJoint {
    pub(super) fn new(state: JointSnapshot) -> Self {
        Self {
            state,
            local_avbd: None,
            local_builtin: None,
        }
    }
}

pub(super) struct SplitState {
    pub builtin: BuiltinPhysicsEngine,
    pub avbd: AvbdEngine,
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
    if body.body_type == BodyType::Dynamic && body.inv_mass <= 0.0 {
        body.inv_mass = 1.0 / body.mass;
        body.inertia = body.shape.inertia(body.mass);
    }
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
            builtin: BuiltinPhysicsEngine::new(gravity),
            avbd: AvbdEngine::new(gravity),
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
    pub(super) fn add_joint(&mut self, a: usize, b: usize, spec: JointKind) -> Option<usize> {
        if a == b || !crate::migration::valid_joint(&spec) {
            return None;
        }
        let (ba, bb) = (&self.bodies.get(a)?.body, &self.bodies.get(b)?.body);
        let reference = if let JointKind::Gear {
            joint_a,
            joint_b,
            ratio,
        } = spec
        {
            crate::migration::JointReference {
                distance: self.coordinate(joint_a)? + ratio * self.coordinate(joint_b)?,
                ..crate::migration::JointReference::default()
            }
        } else {
            let r = crate::resolve_joint(
                &spec,
                ba.position,
                ba.orientation,
                bb.position,
                bb.orientation,
            )?;
            if r.degenerate
                && matches!(
                    spec,
                    JointKind::Revolute { .. } | JointKind::Prismatic { .. }
                )
            {
                return None;
            }
            r.into()
        };
        let h = self.joints.len();
        self.joints.push(SplitJoint::new(JointSnapshot {
            a,
            b,
            spec,
            reference,
        }));
        Some(h)
    }

    fn coordinate(&self, h: usize) -> Option<f32> {
        let j = &self.joints.get(h)?.state;
        let (a, b) = (&self.bodies[j.a].body, &self.bodies[j.b].body);
        match j.spec {
            JointKind::Revolute { local_axis_a, .. } => Some(
                crate::engine::joints::hinge_twist(
                    a.orientation,
                    b.orientation,
                    local_axis_a.normalize_or(Vec3::Z),
                ) - j.reference.angle,
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
                        - j.reference.length,
                )
            }
            _ => None,
        }
    }

    /// Rebuilds retain joint rest data, driver baselines and contact history.
    pub(super) fn rebuild(&mut self) {
        let timer = Instant::now();
        self.avbd = AvbdEngine::new(self.gravity);
        self.builtin = BuiltinPhysicsEngine::new(self.gravity);
        for b in &mut self.bodies {
            restore_mass(&mut b.body);
            b.local_avbd = None;
            b.local_builtin = None;
            if b.owner != SplitOwner::Builtin {
                let h = self.avbd.add_body(b.body.clone());
                self.avbd.restore_body_baseline(h, b.previous);
                b.local_avbd = Some(h);
            }
            if b.owner != SplitOwner::Avbd {
                let h = self.builtin.add_body(b.body.clone());
                self.builtin.restore_body_baseline(h, b.previous);
                b.local_builtin = Some(h);
            }
        }
        for j in &mut self.joints {
            j.local_avbd = None;
            j.local_builtin = None;
        }
        for i in 0..self.joints.len() {
            let j = self.joints[i].state;
            if let (Some(a), Some(b)) = (self.bodies[j.a].local_avbd, self.bodies[j.b].local_avbd) {
                let spec = self.local_spec(j.spec, true);
                if let Some(spec) = spec {
                    let h = self
                        .avbd
                        .add_joint(a, b, spec)
                        .expect("validated AVBD joint");
                    self.avbd.restore_joint_reference(h, j.reference);
                    self.joints[i].local_avbd = Some(h);
                }
            }
            if let (Some(a), Some(b)) = (
                self.bodies[j.a].local_builtin,
                self.bodies[j.b].local_builtin,
            ) {
                let spec = self.local_spec(j.spec, false);
                if let Some(spec) = spec {
                    let h = self
                        .builtin
                        .add_joint(a, b, spec)
                        .expect("validated builtin joint");
                    self.builtin.restore_joint_reference(h, j.reference);
                    self.joints[i].local_builtin = Some(h);
                }
            }
        }
        self.avbd.restore_event_state(self.local_events(true));
        self.builtin.restore_event_state(self.local_events(false));
        self.rebuilds += 1;
        self.timing.rebuild += timer.elapsed();
    }

    fn local_spec(&self, spec: JointKind, avbd: bool) -> Option<JointKind> {
        if let JointKind::Gear {
            joint_a,
            joint_b,
            ratio,
        } = spec
        {
            let local = |h: usize| {
                let j = self.joints.get(h)?;
                if avbd { j.local_avbd } else { j.local_builtin }
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

    fn local_events(&self, avbd: bool) -> EventState {
        let local = |pairs: &BTreeSet<(usize, usize)>| {
            pairs
                .iter()
                .filter_map(|&(a, b)| {
                    let get = |h: usize| {
                        let r = self.bodies.get(h)?;
                        if avbd { r.local_avbd } else { r.local_builtin }
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
                SplitOwner::Avbd => b.local_avbd.and_then(|h| self.avbd.get_body(h)),
                SplitOwner::Builtin => b.local_builtin.and_then(|h| self.builtin.get_body(h)),
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
        let bu = self.builtin.joint_snapshots();
        for j in &mut self.joints {
            let state = j
                .local_avbd
                .and_then(|h| av.get(h))
                .or_else(|| j.local_builtin.and_then(|h| bu.get(h)));
            if let Some(state) = state {
                j.state.reference = state.reference;
            }
        }
    }

    pub(super) fn push_global(&mut self, global: BodyHandle) {
        let Some(b) = self.bodies.get(global) else {
            return;
        };
        if let Some(h) = b.local_avbd {
            self.avbd.wake_body(h);
            if let Some(dst) = self.avbd.get_body_mut(h) {
                *dst = b.body.clone();
            }
        }
        if let Some(h) = b.local_builtin {
            self.builtin.wake_body(h);
            if let Some(dst) = self.builtin.get_body_mut(h) {
                *dst = b.body.clone();
            }
        }
    }

    pub(super) fn wake_global(&mut self, h: usize) {
        if let Some(b) = self.bodies.get_mut(h) {
            b.sleepy = 0;
            if let Some(local) = b.local_avbd {
                self.avbd.wake_body(local);
            }
            if let Some(local) = b.local_builtin {
                self.builtin.wake_body(local);
            }
        }
    }

    /// Ownership is decided before either solver advances. Static anchors
    /// do not join independent islands; gear dependencies join all four sides.
    pub(super) fn route(&mut self, dt: f32, tick: bool, edited: &BTreeSet<usize>) -> usize {
        let timer = Instant::now();
        let n = self.bodies.len();
        let old: Vec<_> = self.bodies.iter().map(|b| b.owner).collect();
        for (h, b) in self.bodies.iter_mut().enumerate() {
            if b.body.body_type != BodyType::Dynamic {
                b.owner = SplitOwner::Static;
                b.sleepy = 0;
            } else if b.owner == SplitOwner::Static {
                b.owner = SplitOwner::Avbd;
                b.sleepy = 0;
                restore_mass(&mut b.body);
            }
            if tick && b.owner != SplitOwner::Static {
                if edited.contains(&h)
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
                let a = self.joints[joint_a].state;
                let b = self.joints[joint_b].state;
                [a.a, a.b, b.a, b.b]
            } else {
                [j.state.a, j.state.b, j.state.a, j.state.b]
            };
            let mut first = None;
            for h in participants {
                if self.bodies[h].owner != SplitOwner::Static {
                    if let Some(a) = first {
                        link(&mut root, a, h);
                    } else {
                        first = Some(h);
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
            active[r] |= edited.contains(&h)
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
            b.owner = if active[r] {
                SplitOwner::Avbd
            } else if calm[r] {
                SplitOwner::Builtin
            } else if has_avbd[r] {
                SplitOwner::Avbd
            } else {
                SplitOwner::Builtin
            };
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
    pub(super) fn step(&mut self) -> (Vec<ContactEvent>, Vec<TriggerEvent>) {
        let timer = Instant::now();
        self.avbd.step(DT);
        self.timing.avbd += timer.elapsed();
        let timer = Instant::now();
        self.builtin.step(DT);
        self.timing.builtin += timer.elapsed();
        let mut av = vec![None; self.bodies.len()];
        let mut bu = av.clone();
        for (global, b) in self.bodies.iter().enumerate() {
            if let Some(h) = b.local_avbd {
                av[h] = Some(global);
            }
            if let Some(h) = b.local_builtin {
                bu[h] = Some(global);
            }
        }
        let mut hits = Vec::new();
        for (events, map) in [
            (self.avbd.drain_contact_events(), &av),
            (self.builtin.drain_contact_events(), &bu),
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
                let (Some(a), Some(b)) = (map[event.body_a], map[event.body_b]) else {
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
        self.builtin.drain_trigger_events();
        let mut now = EventState::default();
        for (state, map) in [
            (self.avbd.event_state(), &av),
            (self.builtin.event_state(), &bu),
        ] {
            let remap = |pairs: BTreeSet<(usize, usize)>| {
                pairs
                    .into_iter()
                    .filter_map(|(a, b)| {
                        let (a, b) = (map[a]?, map[b]?);
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
        self.pull();
        self.steps += 1;
        (contacts, triggers)
    }

    pub(super) fn raycast(&self, ray: Ray, max_dist: f32) -> Option<RaycastHit> {
        if max_dist.is_nan()
            || max_dist < 0.0
            || !ray.origin.is_finite()
            || !ray.direction.is_finite()
        {
            return None;
        }
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
                    handle: h,
                    point: ray.point_at(distance),
                    normal: (b.body.orientation * normal).normalize_or(Vec3::Y),
                    distance,
                });
            }
        }
        closest
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
                h,
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
