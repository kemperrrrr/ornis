//! Trigger overlap events emitted by the builtin physics engine.
//!
//! Triggers participate in broadphase overlap detection but never contribute
//! impulses to the solver. Events are reported as deterministic body-handle
//! pairs and are drained from the physics engine after a step.
//!
//! Triggers are filtered symmetrically in broadphase, narrowphase and the
//! linear CCD path, and share the same collision `layer`/`mask` model as
//! solid bodies (see `body::RigidBody`).

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
