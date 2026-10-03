//! Contact hooks: Rapier-style `filter_contact_pair` /
//! `modify_solver_contacts` seam for the sequential-impulse pipeline.
//!
//! Games need per-pair contact control that bodies alone cannot express:
//! one-way platforms, conveyor belts, ice/mud patches, impact damage,
//! portals. The seam is three trait methods with default no-ops:
//!
//! - [`ContactHooks::filter_pair`] runs after the broadphase, before the
//!   narrowphase (the cheap point: rejected pairs never pay SAT) and
//!   returns a [`SolverFlags`] decision per solid pair: `SKIP` produces no
//!   manifold (never wakes sleepers), `READ_ONLY` builds the manifold and
//!   emits events but applies zero impulses, `COMPUTE_IMPULSES` solves
//!   normally. `SolverFlags: From<bool>` keeps the legacy polarity
//!   (`true` = compute, `false` = skip).
//! - [`ContactHooks::filter_intersection_pair`] is the same veto for
//!   sensor pairs (at least one trigger): it runs over the trigger
//!   overlaps before event reconciliation, so a rejected sensor pair emits
//!   no enter/exit events. Solid pairs never reach it and sensor pairs
//!   never reach `filter_pair`.
//! - [`ContactHooks::modify_contact`] runs before the velocity solve, once
//!   per substep per active manifold, in canonical manifold order. The hook
//!   sees a [`ContactView`] (borrow-safe: owned scalars, no `&mut` into the
//!   engine) and may override the pair normal, the combined friction, the
//!   combined restitution and the conveyor surface velocity, drop or merge
//!   individual contact points ([`ContactView::retain_points`]), plus read
//!   the warm-started accumulated impulse (contact-force estimate) and the
//!   pre-solve approach speed.
//!
//! [`SequentialImpulseEngine`](super::SequentialImpulseEngine) stores
//! `Option<Box<dyn ContactHooks>>` (default `None`). With `None` the step
//! is bit-for-bit the legacy path: both filter passes are skipped, no
//! override records are built, and the solver reads the legacy preamble. A
//! hook whose methods are all default no-ops additionally builds only
//! all-`None` overrides (change detection against the filled legacy values),
//! so it stays bit-identical too (pinned by the snapshot test in
//! `tests/contact_hooks.rs`).
//!
//! Determinism: `filter_pair` runs sequentially in broadphase pair order
//! (canonical min/max keys); `filter_intersection_pair` runs sequentially
//! in broadphase overlap order; `modify_contact` runs sequentially in
//! active manifold order, once per substep, before island partitioning —
//! never inside the parallel island solves. While hooks are attached the
//! engine forces the scalar island path (flat-singleton and GPU
//! single-point batching are skipped) so every hook effect routes through
//! the one solver that implements overrides.
//!
//! # What a hook cannot do
//!
//! Hooks receive only shared body borrows and an owned [`ContactView`]:
//! topology changes (add/remove body, add/remove joint) are impossible by
//! construction — there is no `&mut` engine handle inside the callback.
//! Hooks must not retain the borrowed bodies beyond the call. Sleep
//! invariants hold because filtering happens before manifold creation: a
//! filtered pair emits no contact, wakes nothing, and leaves no events.
//! Panicking out of a hook aborts the step mid-pipeline and may leave
//! scratch pair buffers swapped out; hooks should be infallible in
//! practice.

use rustc_hash::FxHashSet;

use glam::Vec3;

use crate::body::{BodyHandle, RigidBody};
use crate::engine::{Manifold, ManifoldPoint};

use super::MAX_MANIFOLD_POINTS;
use super::SequentialImpulseEngine;
use super::math::point_velocity;

/// Shared read-only pair data for [`ContactHooks::filter_pair`]: the two
/// bodies under their canonical handles (`body_a` belongs to `a`).
#[derive(Debug)]
pub struct PairFilterContext<'a> {
    /// Body under handle `a` (shared borrow, valid for the call only).
    pub body_a: &'a RigidBody,
    /// Body under handle `b` (shared borrow, valid for the call only).
    pub body_b: &'a RigidBody,
}

/// Shared read-only pair data for [`ContactHooks::modify_contact`].
#[derive(Debug)]
pub struct ModifyContext<'a> {
    /// First body (shared borrow, valid for the call only).
    pub body_a: &'a RigidBody,
    /// Second body (shared borrow, valid for the call only).
    pub body_b: &'a RigidBody,
    /// Current substep length in seconds (surface-velocity units reference).
    pub sub_dt: f32,
}

/// Per-pair solver decision returned by [`ContactHooks::filter_pair`]:
/// the narrow-input flag set in the spirit of Rapier's `SolverFlags`
/// (`None` = skip the pair, `Some(empty)` = read-only, `Some(COMPUTE)` =
/// solve — the three outcomes below).
///
/// Only `COMPUTE_IMPULSES` feeds the constraint solver today (soft pairs
/// do not exist in this engine); the set shape leaves room for future
/// solver-side bits without breaking the seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SolverFlags(u32);

impl SolverFlags {
    /// Solve the pair normally: manifold built, impulses applied.
    pub const COMPUTE_IMPULSES: Self = Self(0b01);
    /// Build the manifold and emit events, but apply zero impulses (the
    /// pair is sensed, not solved): the bodies pass through each other
    /// while `modify_contact` still observes every substep.
    pub const READ_ONLY: Self = Self(0b10);
    /// Drop the pair for this step: no manifold, no wake, no events.
    pub const SKIP: Self = Self(0b00);

    /// Empty flag set: [`SolverFlags::SKIP`].
    pub const fn empty() -> Self {
        Self::SKIP
    }

    /// Whether any flag bit is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether `other`'s bits are all set in `self`.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether the pair is dropped for this step ([`SolverFlags::SKIP`]).
    pub const fn is_skip(self) -> bool {
        self.0 == 0
    }

    /// Whether the solver may apply impulses to this pair.
    pub const fn computes_impulses(self) -> bool {
        self.contains(Self::COMPUTE_IMPULSES)
    }

    /// Whether the manifold is built but never solved: kept (non-`SKIP`)
    /// and impulse-free.
    pub const fn is_read_only(self) -> bool {
        !self.is_skip() && !self.computes_impulses()
    }
}

impl Default for SolverFlags {
    /// Default pair decision: solve ([`SolverFlags::COMPUTE_IMPULSES`],
    /// Rapier parity).
    fn default() -> Self {
        Self::COMPUTE_IMPULSES
    }
}

impl From<bool> for SolverFlags {
    /// Legacy `filter_pair: bool` polarity: `true` keeps and solves the
    /// pair ([`SolverFlags::COMPUTE_IMPULSES`]), `false` drops it
    /// ([`SolverFlags::SKIP`]). There is no `bool` spelling of
    /// [`SolverFlags::READ_ONLY`].
    fn from(keep: bool) -> Self {
        if keep {
            Self::COMPUTE_IMPULSES
        } else {
            Self::SKIP
        }
    }
}

impl From<SolverFlags> for bool {
    /// Legacy polarity back: a kept pair (`COMPUTE_IMPULSES` or
    /// `READ_ONLY`) reads as `true`, [`SolverFlags::SKIP`] as `false`.
    fn from(flags: SolverFlags) -> bool {
        !flags.is_skip()
    }
}

impl std::ops::BitOr for SolverFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for SolverFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// One hook-visible contact point: world-space position plus penetration
/// depth (negative = speculative gap). Copy the [`ContactView::points`]
/// entry out, or edit it in place, to merge points by hand; use
/// [`ContactView::retain_points`] for the common drop-kept-subset case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactPointView {
    /// Contact point in world space.
    pub position: Vec3,
    /// Penetration depth at the point (negative = speculative gap).
    pub penetration: f32,
}

/// Borrow-safe per-pair contact view for [`ContactHooks::modify_contact`].
///
/// Owned scalars only: mutating the view never aliases engine state. After
/// the call the engine validates the mutable fields (non-unit normals are
/// re-normalized, degenerate/non-finite normals are dropped, negative or
/// non-finite friction is dropped, restitution is clamped to `0..=1`,
/// non-finite surface velocity is dropped, non-finite points are dropped
/// and the point list is truncated to the manifold capacity) and applies
/// only the entries that differ from the filled legacy values.
#[derive(Debug, Clone)]
pub struct ContactView {
    /// First body handle.
    pub body_a: BodyHandle,
    /// Second body handle.
    pub body_b: BodyHandle,
    /// Contact normal, body A to body B. Mutable by the hook.
    pub normal: Vec3,
    /// Combined Coulomb friction (isotropic override: sets both `mu` and
    /// `mu2`, replacing any body-level anisotropy for this pair). Mutable.
    pub friction: f32,
    /// Combined restitution. Mutable.
    pub restitution: f32,
    /// Conveyor surface velocity in world space (default zero): the
    /// friction solver drives the relative contact velocity toward this
    /// instead of zero, so a static belt with velocity `(2, 0, 0)` carries
    /// resting boxes along +X. Mutable.
    pub surface_velocity: Vec3,
    /// Warm-started accumulated normal impulse over the pair (read-only
    /// snapshot of the previous substep; contact force ≈ impulse /
    /// sub-dt). Zero on first touch.
    pub accumulated_impulse: f32,
    /// Pre-solve closing speed along the normal (`> 0` approaching,
    /// `0` separating). Read-only; impact-damage threshold input.
    pub approach_speed: f32,
    /// Deepest penetration over the manifold points (negative = speculative
    /// gap). Read-only.
    pub penetration: f32,
    /// Live contact points in manifold order (at most the manifold
    /// capacity). Mutable: drop or merge entries to steer the reduction —
    /// a hook that clears every point suppresses the manifold for this
    /// substep (no impulses, no hit; begin/end follow the touching state
    /// of the surviving points). Untouched views compare equal to the
    /// manifold and are never written back, so no-op hooks stay
    /// bit-identical.
    pub points: Vec<ContactPointView>,
}

impl ContactView {
    /// Keep only the points selected by `f` (manifold order preserved):
    /// the manifold-reduction point. Reducing a 4-point face contact to
    /// its deepest point keeps a stack standing on one lane; clearing all
    /// points drops the manifold for this substep. Edits beyond dropping
    /// (merging, repositioning) go through [`ContactView::points`] directly.
    pub fn retain_points(&mut self, f: impl FnMut(&ContactPointView) -> bool) {
        self.points.retain(f);
    }
}

/// Per-pair contact-law decision: game-contact seam in the spirit of
/// Rapier's `filter_contact_pair` / `modify_solver_contacts`, shaped for
/// this engine (narrow-input filter + pre-solve manifold override).
///
/// Both methods have no-op defaults; the trait is object-safe
/// (`Box<dyn ContactHooks>`) and `Send + Sync` so steps stay thread-safe.
pub trait ContactHooks: Send + Sync {
    /// Cheap pair decision after the broadphase, before the narrowphase.
    /// `a`/`b` are dense body indices; `ctx` carries the shared bodies.
    /// Called sequentially in broadphase pair order, once per step, for
    /// solid pairs only (sensor pairs go to
    /// [`ContactHooks::filter_intersection_pair`] instead). Return
    /// [`SolverFlags::SKIP`] to drop the pair for this step (no manifold,
    /// no wake, no events), [`SolverFlags::READ_ONLY`] to sense it without
    /// solving, [`SolverFlags::COMPUTE_IMPULSES`] to solve it. `bool`
    /// converts into flags (`true` = compute), so a legacy `return expr`
    /// becomes `return SolverFlags::from(expr)`. The default solves every
    /// pair.
    fn filter_pair(
        &self,
        a: BodyHandle,
        b: BodyHandle,
        ctx: &PairFilterContext<'_>,
    ) -> SolverFlags {
        let _ = (a, b, ctx);
        SolverFlags::COMPUTE_IMPULSES
    }

    /// Sensor-pair veto over the trigger overlaps, before event
    /// reconciliation. `a`/`b` are dense body indices; `ctx` carries the
    /// shared bodies. Return `false` to suppress the pair's enter/exit
    /// events for this step (the overlap is still detected, just not
    /// reported). Called sequentially in broadphase overlap order, once
    /// per step, for pairs with at least one trigger only. The default
    /// keeps every overlap.
    fn filter_intersection_pair(
        &self,
        a: BodyHandle,
        b: BodyHandle,
        ctx: &PairFilterContext<'_>,
    ) -> bool {
        let _ = (a, b, ctx);
        true
    }

    /// Pre-solve per-manifold override, once per substep per active
    /// manifold in manifold order. Mutate `contact` to steer the solve
    /// (scalars and [`ContactView::retain_points`]); read the
    /// impulse/approach inputs for gameplay (damage, audio). `READ_ONLY`
    /// pairs observe the call with a zeroed `accumulated_impulse` but are
    /// never solved. The default changes nothing.
    fn modify_contact(&self, contact: &mut ContactView, ctx: &ModifyContext<'_>) {
        let _ = (contact, ctx);
    }
}

/// One-way platform as a ready-made [`ContactHooks`]: pairs with
/// `platform` pass through while the other body moves up faster than
/// `pass_speed`, and collide otherwise — a ball shot from below crosses
/// the plate, then lands on top when it falls back.
///
/// Velocity-gated like the classic filter-side platform (not Rapier's
/// modify-side `update_as_oneway_platform` state machine, which needs a
/// per-manifold persistent word this engine does not store): the gate
/// reads the non-platform body's world velocity, so it suits platforms
/// that are static or slow next to fast gameplay bodies.
#[derive(Debug, Clone, Copy)]
pub struct OneWayPlatform {
    /// The platform body: pairs not touching it always solve.
    pub platform: BodyHandle,
    /// Upward speed above which the other body passes through (m/s).
    pub pass_speed: f32,
}

impl OneWayPlatform {
    /// Platform gate with the default 1 m/s pass speed.
    pub fn new(platform: BodyHandle) -> Self {
        Self::with_pass_speed(platform, 1.0)
    }

    /// Platform gate with an explicit pass speed (m/s): the other body
    /// passes through while its upward velocity exceeds it.
    pub fn with_pass_speed(platform: BodyHandle, pass_speed: f32) -> Self {
        Self {
            platform,
            pass_speed,
        }
    }
}

impl ContactHooks for OneWayPlatform {
    fn filter_pair(
        &self,
        a: BodyHandle,
        b: BodyHandle,
        ctx: &PairFilterContext<'_>,
    ) -> SolverFlags {
        let other = if a == self.platform {
            ctx.body_b
        } else if b == self.platform {
            ctx.body_a
        } else {
            return SolverFlags::COMPUTE_IMPULSES;
        };
        if other.velocity.y > self.pass_speed {
            SolverFlags::SKIP
        } else {
            SolverFlags::COMPUTE_IMPULSES
        }
    }
}

/// Validated per-pair override travelling with the manifold into the
/// island solve. Each field is `Some` only when the hook changed it, so
/// untouched pairs (and no-op hooks) keep the legacy preamble exactly.
#[derive(Debug, Clone, Default)]
pub(crate) struct HookOverride {
    /// Isotropic friction override (sets both `mu` and `mu2`).
    pub friction: Option<f32>,
    /// Combined-restitution override.
    pub restitution: Option<f32>,
    /// Conveyor surface-velocity override.
    pub surface_velocity: Option<Vec3>,
}

impl SequentialImpulseEngine {
    /// Attach (`Some`) or detach (`None`) the contact-hooks seam. Default:
    /// `None` (legacy bit-exact path). Attaching or detaching clears the
    /// warm-start cache: cached impulses belong to the previous contact
    /// law, and reusing them across the switch would inject stale support.
    pub fn set_contact_hooks(&mut self, hooks: Option<Box<dyn ContactHooks>>) {
        self.contact_hooks = hooks;
        self.warm_impulses.clear();
        self.hook_read_only.clear();
    }

    /// Whether a contact-hooks object is currently attached.
    pub fn has_contact_hooks(&self) -> bool {
        self.contact_hooks.is_some()
    }

    /// H1 narrow-input filter: sequential per-pair [`SolverFlags`]
    /// decision over the broadphase pairs in place (order-preserving
    /// `retain`, so the no-op hook keeps the narrowphase input exactly).
    /// `SKIP` pairs are dropped; `READ_ONLY` pairs are kept and their
    /// canonical keys are recorded in `hook_read_only` for the solver to
    /// sense without solving. Sensor pairs are never passed to
    /// `filter_pair` (the intersection filter owns them). Clears the
    /// read-only set first, so the no-hooks path stays empty by
    /// construction.
    pub(super) fn apply_hook_filter(&mut self, pairs: &mut Vec<(usize, usize)>) {
        self.hook_read_only.clear();
        let Some(hooks) = self.contact_hooks.as_deref() else {
            return;
        };
        if pairs.is_empty() {
            return;
        }
        let mut read_only: FxHashSet<(usize, usize)> = FxHashSet::default();
        let bodies = &self.bodies;
        pairs.retain(|&(a, b)| {
            if bodies[a].is_trigger || bodies[b].is_trigger {
                return true;
            }
            let flags = hooks.filter_pair(
                BodyHandle::from(a),
                BodyHandle::from(b),
                &PairFilterContext {
                    body_a: &bodies[a],
                    body_b: &bodies[b],
                },
            );
            if flags.is_read_only() {
                read_only.insert((a.min(b), a.max(b)));
            }
            !flags.is_skip()
        });
        self.hook_read_only = read_only;
    }

    /// Intersection-pair filter: sequential veto over the detected trigger
    /// overlaps in place (order-preserving `retain`, so the no-op hook
    /// keeps the reconciled event stream exactly). Runs after the exact
    /// distance check, before enter/exit reconciliation: a rejected overlap
    /// emits no events for this step. Without hooks the input is returned
    /// untouched.
    pub(super) fn apply_hook_intersection_filter(
        &self,
        overlaps: Vec<(usize, usize)>,
    ) -> Vec<(usize, usize)> {
        let Some(hooks) = self.contact_hooks.as_deref() else {
            return overlaps;
        };
        if overlaps.is_empty() {
            return overlaps;
        }
        let bodies = &self.bodies;
        overlaps
            .into_iter()
            .filter(|&(a, b)| {
                hooks.filter_intersection_pair(
                    BodyHandle::from(a),
                    BodyHandle::from(b),
                    &PairFilterContext {
                        body_a: &bodies[a],
                        body_b: &bodies[b],
                    },
                )
            })
            .collect()
    }

    /// H2 pre-solve override: one sequential [`ContactHooks::modify_contact`]
    /// call per active manifold (manifold order, deterministic), before
    /// island partitioning. Returns one entry per global manifold
    /// (`None` = legacy preamble, untouched). Without hooks returns an
    /// empty vec (partition treats missing entries as `None`).
    pub(super) fn apply_hook_modify(
        &self,
        manifolds: &mut [Manifold],
        active: &[usize],
        sub_dt: f32,
    ) -> Vec<Option<HookOverride>> {
        let Some(hooks) = self.contact_hooks.as_deref() else {
            return Vec::new();
        };
        let mut out: Vec<Option<HookOverride>> = Vec::new();
        out.resize_with(manifolds.len(), Option::default);
        for &mi in active {
            // Copy the inputs out first: the hook write-back mutates the
            // manifold below, so no borrow of it survives the call.
            let (orig_normal, bha, bhb, count, pts) = {
                let m = &manifolds[mi];
                (m.normal, m.body_a, m.body_b, m.point_count, m.points)
            };
            let (i, j) = (bha.index(), bhb.index());
            let (ba, bb) = (&self.bodies[i], &self.bodies[j]);
            let key = (i.min(j), i.max(j));
            // Read-only pairs never carry impulses: report zero so the
            // force probe cannot observe stale support from before the
            // flag flip (the solver excludes them below, so the cache
            // could otherwise outlive the decision by a substep).
            let accumulated = if self.hook_read_only.contains(&key) {
                0.0
            } else {
                self.warm_impulses
                    .get(&key)
                    .map(|(pts, n)| pts.iter().take(*n).map(|p| p.impulse).sum())
                    .unwrap_or(0.0)
            };
            let mut best_closing = 0.0f32;
            let mut deepest = f32::NEG_INFINITY;
            for pt in pts.iter().take(count) {
                let p = pt.world_point;
                let ra = p - ba.position;
                let rb = p - bb.position;
                let vrel = point_velocity(bb, rb) - point_velocity(ba, ra);
                best_closing = best_closing.max(-vrel.dot(orig_normal));
                deepest = deepest.max(pt.penetration);
            }
            let legacy_friction = ba.friction.max(bb.friction);
            let legacy_restitution = ba.restitution.min(bb.restitution);
            let mut view = ContactView {
                body_a: bha,
                body_b: bhb,
                normal: orig_normal,
                friction: legacy_friction,
                restitution: legacy_restitution,
                surface_velocity: Vec3::ZERO,
                accumulated_impulse: accumulated,
                approach_speed: best_closing.max(0.0),
                penetration: deepest,
                points: pts
                    .iter()
                    .take(count)
                    .map(|p| ContactPointView {
                        position: p.world_point,
                        penetration: p.penetration,
                    })
                    .collect(),
            };
            hooks.modify_contact(
                &mut view,
                &ModifyContext {
                    body_a: ba,
                    body_b: bb,
                    sub_dt,
                },
            );
            // Validate + write back the normal (re-normalized when finite
            // and non-degenerate, original kept otherwise).
            if view.normal.is_finite() && view.normal.length_squared() > 1e-12 {
                manifolds[mi].normal = view.normal.normalize();
            } else {
                view.normal = orig_normal;
            }
            // Validate the scalars (invalid input falls back to legacy, so
            // a sloppy hook can never poison the solver with NaN).
            if !(view.friction.is_finite() && view.friction >= 0.0) {
                view.friction = legacy_friction;
            }
            if view.restitution.is_finite() {
                view.restitution = view.restitution.clamp(0.0, 1.0);
            } else {
                view.restitution = legacy_restitution;
            }
            if !view.surface_velocity.is_finite() {
                view.surface_velocity = Vec3::ZERO;
            }
            // Reduction point: truncate to the manifold capacity and drop
            // non-finite points (a sloppy hook can never poison the
            // solver with NaN anchors, same discipline as the scalars).
            if view.points.len() > MAX_MANIFOLD_POINTS {
                view.points.truncate(MAX_MANIFOLD_POINTS);
            }
            view.points
                .retain(|p| p.position.is_finite() && p.penetration.is_finite());
            let changed_points =
                view.points.len() != count
                    || view.points.iter().zip(pts.iter()).any(|(v, m)| {
                        v.position != m.world_point || v.penetration != m.penetration
                    });
            if changed_points {
                let m = &mut manifolds[mi];
                for (k, pv) in view.points.iter().enumerate() {
                    m.points[k] = ManifoldPoint {
                        world_point: pv.position,
                        penetration: pv.penetration,
                    };
                }
                // Zero is a valid outcome (hook cleared every point):
                // the solve stages skip invalid counts, and the event
                // pass reads `0..point_count`, so the manifold simply
                // contributes nothing this substep.
                m.point_count = view.points.len();
            }
            let changed_normal = view.normal != orig_normal;
            let changed_friction = view.friction != legacy_friction;
            let changed_restitution = view.restitution != legacy_restitution;
            let changed_surface = view.surface_velocity != Vec3::ZERO;
            if changed_normal
                || changed_friction
                || changed_restitution
                || changed_surface
                || changed_points
            {
                out[mi] = Some(HookOverride {
                    friction: changed_friction.then_some(view.friction),
                    restitution: changed_restitution.then_some(view.restitution),
                    surface_velocity: changed_surface.then_some(view.surface_velocity),
                });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::PhysicsEngine;

    struct Noop;

    impl ContactHooks for Noop {}

    /// Default `filter_pair` solves every pair.
    #[test]
    fn default_filter_keeps_pairs() {
        let hooks = Noop;
        let a = RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0);
        let b = RigidBody::new_box(Vec3::X, Vec3::splat(0.5), 1.0);
        let ctx = PairFilterContext {
            body_a: &a,
            body_b: &b,
        };
        assert_eq!(
            hooks.filter_pair(BodyHandle::from_raw(0), BodyHandle::from_raw(1), &ctx),
            SolverFlags::COMPUTE_IMPULSES
        );
    }

    /// Default `filter_intersection_pair` keeps every sensor overlap.
    #[test]
    fn default_intersection_filter_keeps_overlaps() {
        let hooks = Noop;
        let a = RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0);
        let b = RigidBody::new_box(Vec3::X, Vec3::splat(0.5), 1.0);
        let ctx = PairFilterContext {
            body_a: &a,
            body_b: &b,
        };
        assert!(hooks.filter_intersection_pair(
            BodyHandle::from_raw(0),
            BodyHandle::from_raw(1),
            &ctx
        ));
    }

    /// `SolverFlags` keeps the legacy `bool` polarity: `true` solves,
    /// `false` skips, and back.
    #[test]
    fn solver_flags_stay_bool_compatible() {
        assert_eq!(SolverFlags::from(true), SolverFlags::COMPUTE_IMPULSES);
        assert_eq!(SolverFlags::from(false), SolverFlags::SKIP);
        assert!(bool::from(SolverFlags::COMPUTE_IMPULSES));
        assert!(bool::from(SolverFlags::READ_ONLY));
        assert!(!bool::from(SolverFlags::SKIP));
        assert_eq!(SolverFlags::default(), SolverFlags::COMPUTE_IMPULSES);
        assert!(SolverFlags::READ_ONLY.is_read_only());
        assert!(!SolverFlags::COMPUTE_IMPULSES.is_read_only());
        assert!(SolverFlags::SKIP.is_skip());
    }

    /// `retain_points` keeps manifold order and drops the rest.
    #[test]
    fn retain_points_keeps_order() {
        let mut view = ContactView {
            body_a: BodyHandle::from_raw(0),
            body_b: BodyHandle::from_raw(1),
            normal: Vec3::Y,
            friction: 0.5,
            restitution: 0.3,
            surface_velocity: Vec3::ZERO,
            accumulated_impulse: 0.0,
            approach_speed: 0.0,
            penetration: 0.03,
            points: vec![
                ContactPointView {
                    position: Vec3::new(0.0, 0.0, 0.0),
                    penetration: 0.01,
                },
                ContactPointView {
                    position: Vec3::new(1.0, 0.0, 0.0),
                    penetration: 0.03,
                },
                ContactPointView {
                    position: Vec3::new(2.0, 0.0, 0.0),
                    penetration: 0.02,
                },
            ],
        };
        view.retain_points(|p| p.penetration >= 0.02);
        assert_eq!(view.points.len(), 2);
        assert_eq!(view.points[0].position, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(view.points[1].position, Vec3::new(2.0, 0.0, 0.0));
    }

    /// Default `modify_contact` leaves the view untouched.
    #[test]
    fn default_modify_leaves_view_untouched() {
        let hooks = Noop;
        let a = RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0);
        let b = RigidBody::new_box(Vec3::X, Vec3::splat(0.5), 1.0);
        let mut view = ContactView {
            body_a: BodyHandle::from_raw(0),
            body_b: BodyHandle::from_raw(1),
            normal: Vec3::Y,
            friction: 0.5,
            restitution: 0.3,
            surface_velocity: Vec3::ZERO,
            accumulated_impulse: 1.0,
            approach_speed: 2.0,
            penetration: 0.01,
            points: vec![ContactPointView {
                position: Vec3::ZERO,
                penetration: 0.01,
            }],
        };
        hooks.modify_contact(
            &mut view,
            &ModifyContext {
                body_a: &a,
                body_b: &b,
                sub_dt: 1.0 / 720.0,
            },
        );
        assert_eq!(view.normal, Vec3::Y);
        assert_eq!(view.friction, 0.5);
        assert_eq!(view.restitution, 0.3);
        assert_eq!(view.surface_velocity, Vec3::ZERO);
        assert_eq!(view.points.len(), 1);
    }

    /// [`OneWayPlatform`] lets a fast-rising body through and holds the rest.
    #[test]
    fn one_way_platform_gates_on_rise_speed() {
        let platform = BodyHandle::from_raw(0);
        let hooks = OneWayPlatform::new(platform);
        let plate = RigidBody::new_box(Vec3::ZERO, Vec3::splat(1.0), 0.0);
        let mut rising = RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0);
        rising.velocity = Vec3::new(0.0, 10.0, 0.0);
        let resting = RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0);
        let ctx_up = PairFilterContext {
            body_a: &plate,
            body_b: &rising,
        };
        let ctx_down = PairFilterContext {
            body_a: &plate,
            body_b: &resting,
        };
        assert_eq!(
            hooks.filter_pair(platform, BodyHandle::from_raw(1), &ctx_up),
            SolverFlags::SKIP
        );
        assert_eq!(
            hooks.filter_pair(platform, BodyHandle::from_raw(1), &ctx_down),
            SolverFlags::COMPUTE_IMPULSES
        );
    }

    /// Degenerate hook output falls back to the legacy values instead of
    /// poisoning the solver (NaN normal kept as original, negative
    /// friction dropped, restitution clamped, NaN belt dropped).
    #[test]
    fn invalid_hook_output_falls_back_to_legacy() {
        struct Sloppy;
        impl ContactHooks for Sloppy {
            fn modify_contact(&self, contact: &mut ContactView, _ctx: &ModifyContext<'_>) {
                contact.normal = Vec3::ZERO;
                contact.friction = -1.0;
                contact.restitution = 7.0;
                contact.surface_velocity = Vec3::NAN;
            }
        }
        let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(10.0, 1.0, 10.0),
            0.0,
        ));
        let klein = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
        physics.set_contact_hooks(Some(Box::new(Sloppy)));
        // Must not panic, produce NaN, or tunnel: the box lands and rests.
        for _ in 0..240 {
            physics.step(1.0 / 60.0);
        }
        let b = physics.get_body(klein).expect("box live");
        assert!(b.position.is_finite(), "sloppy hook must not poison poses");
        assert!(
            (b.position.y - 0.5).abs() < 0.1,
            "box rests on the floor, got y={}",
            b.position.y
        );
    }
}
