//! Undo history: op log with bounded, strategy-chosen state retention.
//!
//! [`UndoStack`] records every applied edit as an [`EditOp`](crate::EditOp)
//! intent plus the view state needed to restore it. [`UndoStrategy::Ops`]
//! (the default) bounds history by entry count — cheap and predictable for
//! small meshes; [`UndoStrategy::Snapshots`] bounds it by retained heap
//! bytes instead, so heavy meshes cannot blow the history budget. Redo is a
//! symmetric stack costing one extra clone per undo; any new push clears it.
//!
//! Decision (ops vs snapshots): ops win by default because intents are tiny
//! (a matrix, a face list) and topological ops are not invertible from the
//! intent alone — an extrude adds vertices no negative depth can remove, a
//! subdivide quadruples triangles, a boolean replaces the shell — so every
//! entry also keeps its pre-op view. The snapshots variant exists for heavy
//! meshes where a count cap alone would admit gigabytes: same entries,
//! evicted by bytes instead of count.

use crate::{EditOp, MeshData, PositiveUsize};

/// How undo history is retained (chosen once per stack, never a bool flag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UndoStrategy {
    /// Count-bounded op log (default): at most `cap` entries, evicted
    /// oldest-first. Cheapest when meshes are small and every drag frame
    /// records an op.
    Ops,
    /// Byte-bounded history: entries are evicted oldest-first once their
    /// retained heap bytes would exceed `max_bytes`. For heavy meshes where
    /// a count cap alone would admit gigabytes. The count `cap` still
    /// applies as a backstop.
    Snapshots {
        /// Ceiling for retained pre-op snapshot bytes.
        max_bytes: usize,
    },
}

impl Default for UndoStrategy {
    /// Ops log: the cheap default (see the module-level decision).
    fn default() -> Self {
        Self::Ops
    }
}

/// How full the undo history is (query enum, never a bare bool/len pair).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UndoDepth {
    /// No undoable entries.
    Empty,
    /// `len` of `cap` slots used.
    Available {
        /// Undoable entries currently retained.
        len: usize,
        /// History capacity.
        cap: PositiveUsize,
    },
    /// At capacity: the next push evicts the oldest entry.
    Full {
        /// History capacity.
        cap: PositiveUsize,
    },
}

/// Typed undo failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum UndoError {
    /// No undoable entries.
    #[error("nothing to undo")]
    Empty,
    /// No redone entries to restore.
    #[error("nothing to redo")]
    RedoEmpty,
    /// One snapshot alone exceeds the byte budget: pushed nothing, history
    /// untouched.
    #[error("single snapshot exceeds the {max_bytes}-byte history budget")]
    BudgetExceeded {
        /// Byte ceiling that rejected the snapshot.
        max_bytes: usize,
    },
    /// Boolean/bevel intents run on [`ExactWorker`](crate::ExactWorker), not
    /// on the preview path: state untouched, nothing logged.
    #[error("exact op needs the background worker, not the preview path")]
    RequiresExactWorker,
}

/// Bounded op log with exact restore and a symmetric redo stack.
///
/// `ops`/`before` are parallel (same length, lockstep): the intent for
/// inspection and replay, plus the pre-op view that makes undo exact even
/// for non-invertible topological ops.
#[derive(Debug, Clone)]
pub struct UndoStack {
    /// Logged intents, oldest first.
    ops: Vec<EditOp>,
    /// Pre-op view per logged intent (parallel to `ops`).
    before: Vec<MeshData>,
    /// Intents undone and available for redo, oldest first.
    redo_ops: Vec<EditOp>,
    /// View at undo time per redo intent (parallel to `redo_ops`).
    redo_after: Vec<MeshData>,
    /// Entry-count capacity (both strategies).
    cap: PositiveUsize,
    /// Retention policy (evict by count vs by bytes).
    strategy: UndoStrategy,
    /// Heap bytes currently retained in `before`.
    retained_bytes: usize,
}

impl UndoStack {
    /// Default entry-count capacity.
    pub const DEFAULT_CAP: usize = 64;

    /// Bounded history with the given capacity and retention policy.
    pub fn new(cap: PositiveUsize, strategy: UndoStrategy) -> Self {
        Self {
            ops: Vec::new(),
            before: Vec::new(),
            redo_ops: Vec::new(),
            redo_after: Vec::new(),
            cap,
            strategy,
            retained_bytes: 0,
        }
    }

    /// Bounded history with [`UndoStack::DEFAULT_CAP`] entries.
    pub fn with_default_cap(strategy: UndoStrategy) -> Self {
        Self::new(PositiveUsize::expect_valid(Self::DEFAULT_CAP), strategy)
    }

    /// Entry-count capacity.
    pub fn cap(self) -> PositiveUsize {
        self.cap
    }

    /// Retention policy chosen at construction.
    pub fn strategy(self) -> UndoStrategy {
        self.strategy
    }

    /// Undoable entries currently retained.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// True when no entries are retained.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Logged intents, oldest first (inspection/replay; not for mutation).
    pub fn ops(&self) -> &[EditOp] {
        &self.ops
    }

    /// Heap bytes currently retained in pre-op snapshots.
    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    /// Fullness query (drives "history full" UI without bool flags).
    pub fn depth(&self) -> UndoDepth {
        if self.ops.is_empty() {
            UndoDepth::Empty
        } else if self.ops.len() >= self.cap.get() {
            UndoDepth::Full { cap: self.cap }
        } else {
            UndoDepth::Available {
                len: self.ops.len(),
                cap: self.cap,
            }
        }
    }

    /// Drop all undo and redo entries.
    pub fn clear(&mut self) {
        self.ops.clear();
        self.before.clear();
        self.redo_ops.clear();
        self.redo_after.clear();
        self.retained_bytes = 0;
    }

    /// Log an applied op with its pre-op view. A new push clears redo.
    /// Consecutive [`EditOp::Transform`] entries coalesce (matrices compose,
    /// original `before` kept) so a drag records one entry, not sixty.
    ///
    /// # Errors
    ///
    /// Returns [`UndoError::BudgetExceeded`] under
    /// [`UndoStrategy::Snapshots`] when one snapshot alone exceeds the byte
    /// budget (history untouched).
    pub fn push(&mut self, op: EditOp, before: MeshData) -> Result<(), UndoError> {
        self.redo_ops.clear();
        self.redo_after.clear();
        if matches!(op, EditOp::Transform { .. })
            && matches!(self.ops.last(), Some(EditOp::Transform { .. }))
        {
            // Drag coalescing: v'' = M_new · (M_old · v).
            if let (Some(EditOp::Transform { matrix }), EditOp::Transform { matrix: next }) =
                (self.ops.last_mut(), &op)
            {
                *matrix = *next * *matrix;
            }
            return Ok(());
        }
        self.make_room(before.heap_bytes())?;
        self.retained_bytes += before.heap_bytes();
        self.ops.push(op);
        self.before.push(before);
        Ok(())
    }

    /// Restore the pre-op view of the newest entry, stashing `current`
    /// (the view at undo time) for redo.
    ///
    /// # Errors
    ///
    /// Returns [`UndoError::Empty`] when no entries are retained.
    pub fn undo(&mut self, current: &MeshData) -> Result<MeshData, UndoError> {
        let Some(op) = self.ops.pop() else {
            return Err(UndoError::Empty);
        };
        let Some(before) = self.before.pop() else {
            self.ops.push(op);
            return Err(UndoError::Empty);
        };
        self.retained_bytes = self.retained_bytes.saturating_sub(before.heap_bytes());
        self.redo_ops.push(op);
        self.redo_after.push(current.clone());
        Ok(before)
    }

    /// Re-apply the newest undone entry, stashing `current` back into undo.
    /// Never coalesces (fusing entries would skip undo steps).
    ///
    /// # Errors
    ///
    /// Returns [`UndoError::RedoEmpty`] when nothing was undone, or
    /// [`UndoError::BudgetExceeded`] when re-logging would break the byte
    /// budget (both stacks untouched).
    pub fn redo(&mut self, current: &MeshData) -> Result<MeshData, UndoError> {
        if self.redo_ops.is_empty() {
            return Err(UndoError::RedoEmpty);
        }
        self.make_room(current.heap_bytes())?;
        let Some(op) = self.redo_ops.pop() else {
            return Err(UndoError::RedoEmpty);
        };
        let Some(after) = self.redo_after.pop() else {
            self.redo_ops.push(op);
            return Err(UndoError::RedoEmpty);
        };
        self.retained_bytes += current.heap_bytes();
        self.ops.push(op);
        self.before.push(current.clone());
        Ok(after)
    }

    /// Evict oldest entries until `bytes` more fit (count cap always, byte
    /// ceiling under [`UndoStrategy::Snapshots`]).
    fn make_room(&mut self, bytes: usize) -> Result<(), UndoError> {
        if let UndoStrategy::Snapshots { max_bytes } = self.strategy
            && bytes > max_bytes
        {
            return Err(UndoError::BudgetExceeded { max_bytes });
        }
        while self.ops.len() >= self.cap.get() || self.over_bytes(bytes) {
            self.evict_oldest();
        }
        Ok(())
    }

    /// True when retaining `bytes` more would break the byte ceiling.
    fn over_bytes(&self, bytes: usize) -> bool {
        match self.strategy {
            UndoStrategy::Ops => false,
            UndoStrategy::Snapshots { max_bytes } => self.retained_bytes + bytes > max_bytes,
        }
    }

    /// Drop the oldest entry (no-op on an empty stack, so the eviction loop
    /// always terminates).
    fn evict_oldest(&mut self) {
        if self.ops.is_empty() {
            return;
        }
        self.ops.remove(0);
        let oldest = self.before.remove(0);
        self.retained_bytes = self.retained_bytes.saturating_sub(oldest.heap_bytes());
    }
}

impl Default for UndoStack {
    /// Empty ops log with default capacity (see [`UndoStack::DEFAULT_CAP`]).
    fn default() -> Self {
        Self::with_default_cap(UndoStrategy::Ops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_mesh() -> MeshData {
        MeshData::unit_box()
    }

    fn cap2() -> PositiveUsize {
        PositiveUsize::expect_valid(2)
    }

    #[test]
    fn stack_defaults_to_empty_bounded_log() {
        assert_eq!(UndoStack::DEFAULT_CAP, 64);
        let stack = UndoStack::default();
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
        assert_eq!(stack.depth(), UndoDepth::Empty);
        assert_eq!(stack.strategy(), UndoStrategy::Ops);
    }

    #[test]
    fn push_undo_redo_roundtrip_restores_views() {
        let mut stack = UndoStack::new(cap2(), UndoStrategy::Ops);
        assert_eq!(stack.depth(), UndoDepth::Empty);
        assert!(stack.is_empty());
        let before = box_mesh();
        let mut after = box_mesh();
        for p in &mut after.positions {
            p[0] += 1.0;
        }
        stack
            .push(EditOp::Subdivide { levels: 1 }, before.clone())
            .expect("push fits");
        assert_eq!(
            stack.depth(),
            UndoDepth::Available {
                len: 1,
                cap: cap2()
            }
        );
        let restored = stack.undo(&after).expect("undo pops");
        assert_eq!(restored.positions, before.positions);
        assert_eq!(restored.indices, before.indices);
        assert_eq!(stack.depth(), UndoDepth::Empty);
        let redone = stack.redo(&restored).expect("redo restores");
        assert_eq!(redone.positions, after.positions);
        assert_eq!(redone.indices, after.indices);
        assert!(matches!(stack.redo(&redone), Err(UndoError::RedoEmpty)));
        let mut empty = UndoStack::new(cap2(), UndoStrategy::Ops);
        assert!(matches!(empty.undo(&before), Err(UndoError::Empty)));
    }

    #[test]
    fn cap_evicts_oldest_first() {
        let mut stack = UndoStack::new(cap2(), UndoStrategy::Ops);
        for levels in 1..=3 {
            stack
                .push(EditOp::Subdivide { levels }, box_mesh())
                .expect("push fits");
        }
        assert_eq!(stack.len(), 2);
        assert_eq!(stack.depth(), UndoDepth::Full { cap: cap2() });
        assert!(matches!(
            stack.ops(),
            [
                EditOp::Subdivide { levels: 2 },
                EditOp::Subdivide { levels: 3 }
            ]
        ));
    }

    #[test]
    fn consecutive_transforms_coalesce_into_one_entry() {
        let mut stack = UndoStack::new(PositiveUsize::expect_valid(8), UndoStrategy::Ops);
        let before = box_mesh();
        let m1 = glam::Mat4::from_translation(glam::Vec3::new(1.0, 0.0, 0.0));
        let m2 = glam::Mat4::from_translation(glam::Vec3::new(0.0, 2.0, 0.0));
        stack
            .push(EditOp::Transform { matrix: m1 }, before.clone())
            .expect("first transform logs");
        stack
            .push(EditOp::Transform { matrix: m2 }, before.clone())
            .expect("drag frame coalesces");
        assert_eq!(stack.len(), 1);
        let EditOp::Transform { matrix } = &stack.ops()[0] else {
            panic!("coalesced entry stays a transform");
        };
        assert_eq!(*matrix, m2 * m1);
        // Undo still jumps to the pre-drag view, not the intermediate one.
        let restored = stack.undo(&before).expect("undo pops");
        assert_eq!(restored.positions, before.positions);
    }

    #[test]
    fn new_push_clears_redo() {
        let mut stack = UndoStack::new(cap2(), UndoStrategy::Ops);
        let mesh = box_mesh();
        stack
            .push(EditOp::Subdivide { levels: 1 }, mesh.clone())
            .expect("push fits");
        let _ = stack.undo(&mesh).expect("undo pops");
        stack
            .push(EditOp::Subdivide { levels: 2 }, mesh.clone())
            .expect("new edit logs");
        assert!(matches!(stack.redo(&mesh), Err(UndoError::RedoEmpty)));
        stack.clear();
        assert!(stack.is_empty());
        assert_eq!(stack.retained_bytes(), 0);
    }

    #[test]
    fn snapshots_strategy_bounds_retained_bytes() {
        let one = box_mesh().heap_bytes();
        assert!(one > 0);
        // One snapshot alone over budget: denial, history untouched.
        let mut denied = UndoStack::new(
            PositiveUsize::expect_valid(8),
            UndoStrategy::Snapshots { max_bytes: one - 1 },
        );
        assert!(matches!(
            denied.push(EditOp::Subdivide { levels: 1 }, box_mesh()),
            Err(UndoError::BudgetExceeded { .. })
        ));
        assert!(denied.is_empty());
        // Exact fit keeps one entry; the next push evicts oldest-first.
        let mut stack = UndoStack::new(
            PositiveUsize::expect_valid(8),
            UndoStrategy::Snapshots { max_bytes: one },
        );
        for levels in 1..=2 {
            stack
                .push(EditOp::Subdivide { levels }, box_mesh())
                .expect("push fits after eviction");
        }
        assert_eq!(stack.len(), 1);
        assert_eq!(stack.retained_bytes(), one);
        assert!(matches!(stack.ops(), [EditOp::Subdivide { levels: 2 }]));
        let restored = stack.undo(&box_mesh()).expect("undo pops");
        assert_eq!(restored.triangle_count(), box_mesh().triangle_count());
        assert_eq!(stack.retained_bytes(), 0);
    }
}
