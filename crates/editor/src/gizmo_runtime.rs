//! Replica-side gizmo drag state machine (PLAN §i, slice E3).
//!
//! Pure transitions over [`GizmoDrag`]: press → preview → commit or cancel,
//! plus the one-step undo bookkeeping. No engine, no GPU, no browser — the
//! viewport side owns the pointer edges and the transport, this module owns
//! the invariants:
//!
//! - a click on a handle without motion commits nothing;
//! - an abandoned drag (tool switch, rebuilt replica) can be rolled back to
//!   the transform it started from;
//! - the one-step undo of the last *committed* drag survives until a real
//!   commit replaces it.
//!
//! Keeping the machine here (instead of in the browser crate) is what makes
//! those invariants host-testable.

use crate::gizmo_drag::GizmoDrag;
use crate::picking::PickRay;
use ornis_assets::scene::TransformDesc;
use ornis_core::Entity;

/// Authoritative identity of the dragged entity plus its pre-drag transform.
///
/// The one-step undo addresses the *server* entity (id + generation) and
/// carries the transform to restore, so it survives a rebuilt replica.
#[derive(Debug, Clone, PartialEq)]
pub struct GizmoTarget {
    /// Authoritative entity id.
    pub id: u32,
    /// Authoritative entity generation.
    pub generation: u32,
    /// Transform to restore on undo.
    pub transform: TransformDesc,
}

/// The press edge of the pointer: only a fresh press may start a drag.
///
/// A held button keeps orbiting the camera without re-picking, so the
/// drag machine consumes the *edge*, not the level.
#[derive(Debug, Default)]
pub struct PressEdge {
    /// Left-button level observed on the previous frame.
    pressed: bool,
}

impl PressEdge {
    /// Whether the button went down since the last call, and records the
    /// new level.
    pub fn edge(&mut self, pressed: bool) -> bool {
        let edge = pressed && !self.pressed;
        self.pressed = pressed;
        edge
    }
}

/// Drag/undo state of the gizmo replica.
#[derive(Debug, Default)]
pub struct GizmoRuntime {
    /// The drag in flight, if any.
    drag: Option<GizmoDrag>,
    /// The entity the drag moves.
    selected: Option<Entity>,
    /// Whether the preview actually changed the transform.
    moved: bool,
    /// Undo candidate of the drag in flight (promoted on a real commit).
    candidate: Option<GizmoTarget>,
    /// One-step undo payload of the last committed drag.
    undo: Option<GizmoTarget>,
}

impl GizmoRuntime {
    /// Arms a drag on `selected`.
    ///
    /// `candidate` is the undo payload of the pre-drag transform: it is
    /// promoted to the one-step undo only when the drag really moves, so a
    /// click on a handle never eats the undo of the previous drag.
    pub fn begin(&mut self, drag: GizmoDrag, selected: Entity, candidate: Option<GizmoTarget>) {
        self.drag = Some(drag);
        self.selected = Some(selected);
        self.moved = false;
        self.candidate = candidate;
    }

    /// Records a preview transform; returns whether the drag moved.
    ///
    /// The preview is compared against the transform the drag started from,
    /// so a press without motion never counts as a move.
    fn preview(&mut self, transform: &TransformDesc) -> bool {
        let Some(drag) = self.drag.as_ref() else {
            return false;
        };
        if transform != &drag.start {
            self.moved = true;
        }
        self.moved
    }

    /// The preview transform for `ray` and whether the drag moved.
    ///
    /// `None` when no drag is in flight or the ray cannot be projected onto
    /// the handle (a parallel ray, a degenerate rotation radius).
    pub fn preview_transform(&mut self, ray: &PickRay) -> Option<TransformDesc> {
        let drag = self.drag.clone()?;
        let transform = drag.update(ray)?;
        self.preview(&transform);
        Some(transform)
    }

    /// Finishes the drag.
    ///
    /// A moved drag promotes its undo candidate to the one-step undo; a
    /// click without motion commits nothing and keeps the previous undo.
    /// Returns the promoted undo payload, if any.
    pub fn commit(&mut self) -> Option<GizmoTarget> {
        self.drag = None;
        self.selected = None;
        let moved = std::mem::take(&mut self.moved);
        let candidate = self.candidate.take();
        if !moved {
            return None;
        }
        self.undo = candidate.clone();
        candidate
    }

    /// Drops the in-flight drag and its undo candidate.
    ///
    /// The one-step undo of the last commit survives — it addresses the
    /// server entity, not the replica.
    pub fn cancel(&mut self) {
        self.drag = None;
        self.selected = None;
        self.moved = false;
        self.candidate = None;
    }

    /// The preview to roll back when the drag is abandoned: the dragged
    /// entity and the transform it had before the drag started.
    pub fn rollback(&self) -> Option<(Entity, TransformDesc)> {
        let drag = self.drag.as_ref()?;
        Some((self.selected?, drag.start.clone()))
    }

    /// Takes the one-step undo payload (the pre-drag transform of the last
    /// committed drag).
    pub fn take_undo(&mut self) -> Option<GizmoTarget> {
        self.undo.take()
    }

    /// Whether a drag is in flight.
    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// The entity the current drag moves.
    pub fn selected(&self) -> Option<Entity> {
        self.selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gizmo::{GizmoAxis, GizmoHandle, GizmoMode, GizmoSpace};
    use crate::picking::PickRay;
    use glam::Vec3;

    /// A press ray through the drag handle.
    fn ray() -> PickRay {
        PickRay::try_new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)).expect("valid ray")
    }

    /// A release ray that drags the handle one unit along the world X.
    fn release_ray() -> PickRay {
        PickRay::try_new(Vec3::new(1.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)).expect("valid ray")
    }

    /// A drag on the world X handle of an identity selection.
    fn drag() -> GizmoDrag {
        GizmoDrag::begin(
            &TransformDesc::IDENTITY,
            GizmoHandle {
                mode: GizmoMode::Move,
                space: GizmoSpace::World,
                axis: GizmoAxis::X,
            },
            &ray(),
        )
        .expect("drag starts")
    }

    /// The undo payload of an entity at the origin.
    fn target() -> GizmoTarget {
        GizmoTarget {
            id: 7,
            generation: 1,
            transform: TransformDesc::IDENTITY,
        }
    }

    /// A runtime with an armed drag on entity 1.
    fn armed() -> GizmoRuntime {
        let mut runtime = GizmoRuntime::default();
        runtime.begin(drag(), Entity::new(1), Some(target()));
        runtime
    }

    /// The transform of an entity moved one unit along X.
    fn moved_x() -> TransformDesc {
        TransformDesc {
            translation: Vec3::new(1.0, 0.0, 0.0),
            ..TransformDesc::IDENTITY
        }
    }

    /// A click on a handle without motion commits nothing: the undo of the
    /// previous drag stays available.
    #[test]
    fn click_without_motion_commits_nothing() {
        let mut runtime = GizmoRuntime::default();
        runtime.begin(drag(), Entity::new(1), Some(target()));
        assert!(!runtime.preview(&TransformDesc::IDENTITY));
        assert!(runtime.commit().is_none());
        assert!(
            runtime.take_undo().is_none(),
            "a click without motion must not create an undo"
        );
    }

    /// A moved drag promotes its undo candidate: the one-step undo restores
    /// the pre-drag transform.
    #[test]
    fn a_moved_drag_promotes_the_undo_candidate() {
        let mut runtime = armed();
        assert!(runtime.preview(&moved_x()));
        let promoted = runtime.commit().expect("a moved drag commits");
        assert_eq!(promoted.transform, TransformDesc::IDENTITY);
        let undo = runtime.take_undo().expect("the undo survives the commit");
        assert_eq!(undo.transform, TransformDesc::IDENTITY);
        assert_eq!(undo.id, 7);
    }

    /// A click on a handle after a real drag keeps the undo of that drag.
    #[test]
    fn a_later_click_does_not_eat_the_previous_undo() {
        let mut runtime = GizmoRuntime::default();
        runtime.begin(drag(), Entity::new(1), Some(target()));
        runtime
            .preview_transform(&release_ray())
            .expect("a preview");
        runtime.commit();
        runtime.begin(drag(), Entity::new(1), Some(target()));
        runtime.preview_transform(&ray()).expect("no motion");
        runtime.commit();
        assert!(
            runtime.take_undo().is_some(),
            "the undo of the committed drag must survive a later click"
        );
    }

    /// An abandoned drag (tool switch, rebuilt replica) rolls back to the
    /// pre-drag transform and drops its undo candidate.
    #[test]
    fn a_cancelled_drag_rolls_back_and_keeps_the_undo() {
        let mut runtime = armed();
        runtime
            .preview_transform(&release_ray())
            .expect("a preview");
        let (entity, transform) = runtime.rollback().expect("a rollback target");
        assert_eq!(entity, Entity::new(1));
        assert_eq!(transform, TransformDesc::IDENTITY);

        runtime.cancel();
        assert!(!runtime.is_dragging());
        assert!(runtime.rollback().is_none());
        assert!(
            runtime.take_undo().is_none(),
            "a cancelled drag never becomes an undo"
        );
    }

    /// The undo addresses the server entity, so it survives a rebuilt
    /// replica (the drag state is gone, the undo payload is not).
    #[test]
    fn the_undo_survives_a_replica_rebuild() {
        let mut runtime = armed();
        runtime
            .preview_transform(&release_ray())
            .expect("a preview");
        runtime.commit();

        // The replica is rebuilt: the drag state is dropped, the undo is not.
        runtime.cancel();
        let undo = runtime.take_undo().expect("the undo survives");
        assert_eq!(undo.id, 7);
        assert_eq!(undo.generation, 1);
        assert_eq!(undo.transform, TransformDesc::IDENTITY);
    }
}
