//! Contact hooks: Rapier-style `filter_contact_pair` /
//! `modify_solver_contacts` seam for the sequential-impulse pipeline.
//!
//! Games need per-pair contact control that bodies alone cannot express:
//! one-way platforms, conveyor belts, ice/mud patches, impact damage,
//! portals. The seam is two trait methods with default no-ops:
//!
//! - [`ContactHooks::filter_pair`] runs after the broadphase, before the
//!   narrowphase (the cheap point: rejected pairs never pay SAT). A
//!   rejected pair produces no manifold, so it never wakes sleepers.
//! - [`ContactHooks::modify_contact`] runs before the velocity solve, once
//!   per substep per active manifold, in canonical manifold order. The hook
//!   sees a [`ContactView`] (borrow-safe: owned scalars, no `&mut` into the
//!   engine) and may override the pair normal, the combined friction, the
//!   combined restitution and the conveyor surface velocity, plus read the
//!   warm-started accumulated impulse (contact-force estimate) and the
//!   pre-solve approach speed.
//!
//! [`SequentialImpulseEngine`](super::SequentialImpulseEngine) stores
//! `Option<Box<dyn ContactHooks>>` (default `None`). With `None` the step
//! is bit-for-bit the legacy path: the filter pass is skipped, no override
//! records are built, and the solver reads the legacy preamble. A hook
//! whose methods are all default no-ops additionally builds only
//! all-`None` overrides (change detection against the filled legacy values),
//! so it stays bit-identical too (pinned by the snapshot test in
//! `tests/contact_hooks.rs`).
//!
//! Determinism: `filter_pair` runs sequentially in broadphase pair order
//! (canonical min/max keys); `modify_contact` runs sequentially in active
//! manifold order, once per substep, before island partitioning — never
//! inside the parallel island solves. While hooks are attached the engine
//! forces the scalar island path (flat-singleton and GPU single-point
//! batching are skipped) so every hook effect routes through the one solver
//! that implements overrides.
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

use glam::Vec3;

use crate::body::{BodyHandle, RigidBody};
use crate::engine::Manifold;

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

/// Borrow-safe per-pair contact view for [`ContactHooks::modify_contact`].
///
/// Owned scalars only: mutating the view never aliases engine state. After
/// the call the engine validates the mutable fields (non-unit normals are
/// re-normalized, degenerate/non-finite normals are dropped, negative or
/// non-finite friction is dropped, restitution is clamped to `0..=1`,
/// non-finite surface velocity is dropped) and applies only the entries
/// that differ from the filled legacy values.
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
}

/// Per-pair contact-law decision: game-contact seam in the spirit of
/// Rapier's `filter_contact_pair` / `modify_solver_contacts`, shaped for
/// this engine (narrow-input filter + pre-solve manifold override).
///
/// Both methods have no-op defaults; the trait is object-safe
/// (`Box<dyn ContactHooks>`) and `Send + Sync` so steps stay thread-safe.
pub trait ContactHooks: Send + Sync {
    /// Cheap pair veto after the broadphase, before the narrowphase.
    /// `a`/`b` are dense body indices; `ctx` carries the shared bodies.
    /// Return `false` to drop the pair for this step (no manifold, no
    /// wake, no events). Called sequentially in broadphase pair order.
    /// The default keeps every pair.
    fn filter_pair(&self, a: BodyHandle, b: BodyHandle, ctx: &PairFilterContext<'_>) -> bool {
        let _ = (a, b, ctx);
        true
    }

    /// Pre-solve per-manifold override, once per substep per active
    /// manifold in manifold order. Mutate `contact` to steer the solve;
    /// read the impulse/approach inputs for gameplay (damage, audio).
    /// The default changes nothing.
    fn modify_contact(&self, contact: &mut ContactView, ctx: &ModifyContext<'_>) {
        let _ = (contact, ctx);
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
    }

    /// Whether a contact-hooks object is currently attached.
    pub fn has_contact_hooks(&self) -> bool {
        self.contact_hooks.is_some()
    }

    /// H1 narrow-input filter: sequential veto over the broadphase pairs
    /// in place (order-preserving `retain`, so the no-op hook keeps the
    /// narrowphase input exactly). Skipped entirely without hooks.
    pub(super) fn apply_hook_filter(&self, pairs: &mut Vec<(usize, usize)>) {
        let Some(hooks) = self.contact_hooks.as_deref() else {
            return;
        };
        if pairs.is_empty() {
            return;
        }
        let bodies = &self.bodies;
        pairs.retain(|&(a, b)| {
            hooks.filter_pair(
                BodyHandle::from(a),
                BodyHandle::from(b),
                &PairFilterContext {
                    body_a: &bodies[a],
                    body_b: &bodies[b],
                },
            )
        });
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
            let accumulated = self
                .warm_impulses
                .get(&key)
                .map(|(pts, n)| pts.iter().take(*n).map(|p| p.impulse).sum())
                .unwrap_or(0.0);
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
            let changed_normal = view.normal != orig_normal;
            let changed_friction = view.friction != legacy_friction;
            let changed_restitution = view.restitution != legacy_restitution;
            let changed_surface = view.surface_velocity != Vec3::ZERO;
            if changed_normal || changed_friction || changed_restitution || changed_surface {
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

    /// Default `filter_pair` keeps every pair.
    #[test]
    fn default_filter_keeps_pairs() {
        let hooks = Noop;
        let a = RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0);
        let b = RigidBody::new_box(Vec3::X, Vec3::splat(0.5), 1.0);
        let ctx = PairFilterContext {
            body_a: &a,
            body_b: &b,
        };
        assert!(hooks.filter_pair(BodyHandle::from_raw(0), BodyHandle::from_raw(1), &ctx));
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
