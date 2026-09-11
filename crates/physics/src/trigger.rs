//! Trigger overlap events emitted by the builtin physics engine.
//!
//! Triggers participate in broadphase overlap detection but never contribute
//! impulses to the solver. Events are reported as deterministic body-handle
//! pairs and are drained from the physics engine after a step.
//!
//! Triggers are filtered symmetrically in broadphase, narrowphase and the
//! linear CCD path, and share the same collision `layer`/`mask` model as
//! solid bodies (see `body::RigidBody`).
//!
//! Solid-contact events ([`ContactEvent`]) live here too: same drain-after-step
//! discipline, same deterministic ordering, but for impulse-carrying contacts.

use glam::Vec3;

use crate::body::BodyHandle;

/// Kind of transition for a trigger overlap pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerEventKind {
    /// The pair was not overlapping on the previous completed step and is now.
    Entered,
    /// The pair was overlapping on the previous completed step and is not now.
    Exited,
}

/// A transition in the overlap state of a pair containing at least one trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriggerEvent {
    /// Lower body handle of the canonical pair.
    pub body_a: BodyHandle,
    /// Higher body handle of the canonical pair.
    pub body_b: BodyHandle,
    /// Whether the pair entered or exited the trigger volume.
    pub kind: TriggerEventKind,
}

/// Kind of a solid-contact transition. Box3D `b3ContactEvents` parity:
/// begin/end track touching state, hit reports hard impacts by approach
/// speed (including speculative contacts with a confirmed impulse).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContactEventKind {
    /// The pair started touching this step (was separated before).
    /// Touching means penetration above [`CONTACT_BEGIN_SLOP`] — solver
    /// residuals included, speculative near-misses excluded.
    Begin,
    /// The pair stopped touching this step (touches no more).
    End,
    /// A hard impact: approach speed above [`CONTACT_HIT_THRESHOLD`] at a
    /// solved contact. Always positive, meters per second.
    Hit {
        /// World contact point of the hardest-approaching point.
        point: Vec3,
        /// Contact normal as solved (from `body_a` toward `body_b`).
        normal: Vec3,
        /// Closing speed along the normal. Always positive.
        approach_speed: f32,
    },
}

/// Threshold (m/s) above which an impact becomes a [`ContactEventKind::Hit`].
/// Mirrors the restitution kick-in so hits and bounces agree on "hard".
pub const CONTACT_HIT_THRESHOLD: f32 = 1.0;

/// Slop (m) below which a gap still counts as touching for begin/end.
/// Contacts live at ~zero penetration (solver residuals), so strict
/// positivity would miss real touchdowns. Box2D `linearSlop` parity.
pub const CONTACT_BEGIN_SLOP: f32 = 0.005;

/// A solid-contact transition between two non-trigger bodies, drained after
/// the step like [`TriggerEvent`]. Pairs are reported in deterministic
/// (sorted) order; orientations follow the stored pair order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactEvent {
    /// First body handle of the pair (manifold order).
    pub body_a: BodyHandle,
    /// Second body handle of the pair (manifold order).
    pub body_b: BodyHandle,
    /// What happened: begin, end, or hard hit (with impact data).
    pub kind: ContactEventKind,
}
