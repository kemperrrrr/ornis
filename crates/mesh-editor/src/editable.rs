//! Interactive preview mesh: cheap edits applied every frame.
//!
//! [`EditableMesh`] keeps the committed `base` plus an optional `preview`
//! copy. Preview ops (transform, extrude, subdivide) run incrementally on
//! the CPU inside the frame budget; exact ops (boolean, full bevel) run on
//! [`crate::ExactWorker`] and land via [`EditableMesh::commit`]/`cancel`.

use std::collections::HashMap;
use std::time::Instant;

use crate::mesh_data::TRIANGLE_VERTS;

/// Base mesh plus optional in-progress preview and dirty tracking.
#[derive(Debug)]
pub struct EditableMesh {
    /// Last committed (exact) version.
    pub base: crate::MeshData,
    /// Working copy shown while an edit is in flight.
    pub preview: Option<crate::MeshData>,
    /// Sequence of the current preview (bumped per preview op).
    pub preview_seq: u64,
    /// Sequence of the last committed base.
    pub base_seq: u64,
    /// What changed since the last GPU upload.
    pub dirty: crate::MeshDirty,
    /// Frame-budget telemetry of the split.
    pub stats: crate::FrameStats,
}

impl EditableMesh {
    /// Wrap a committed mesh with no pending preview.
    pub fn new(base: crate::MeshData) -> Self {
        Self {
            base,
            preview: None,
            preview_seq: 0,
            base_seq: 0,
            dirty: crate::MeshDirty::new(),
            stats: crate::FrameStats::default(),
        }
    }

    /// Currently displayed mesh: preview when present, else base.
    pub fn view(&self) -> &crate::MeshData {
        self.preview.as_ref().unwrap_or(&self.base)
    }

    /// Apply a rigid transform incrementally (`O(n)` over vertices).
    ///
    /// Positions go through the full matrix, normals through its rotation
    /// part only (translation-free, renormalized).
    pub fn apply_transform(&mut self, matrix: glam::Mat4) {
        let started = Instant::now();
        let mesh = self.preview_or_clone();
        let rotation = glam::Mat3::from_mat4(matrix);
        for (p, n) in mesh.positions.iter_mut().zip(mesh.normals.iter_mut()) {
            let v = matrix.transform_point3(glam::Vec3::new(p[0], p[1], p[2]));
            *p = [v.x, v.y, v.z];
            let w = rotation * glam::Vec3::new(n[0], n[1], n[2]);
            let w = w.normalize_or_zero();
            *n = [w.x, w.y, w.z];
        }
        self.finish_preview(
            crate::MeshDirty::VERTS | crate::MeshDirty::NORMALS,
            &[],
            started,
        );
    }

    /// Push faces along their face normals, building side walls.
    ///
    /// Each extruded face is duplicated (new vertices shifted by
    /// `depth` along the face normal) and stitched to the original loop
    /// with side-wall quads, so the result stays a closed shell.
    pub fn apply_extrude(&mut self, faces: &[u32], depth: f32) {
        let started = Instant::now();
        let mesh = self.preview_or_clone();
        let tri_count = mesh.triangle_count();
        let mut affected: Vec<u32> = Vec::new();
        for &f in faces {
            let f = f as usize;
            if f >= tri_count {
                continue;
            }
            let [a, b, c] = [
                mesh.indices[TRIANGLE_VERTS * f],
                mesh.indices[TRIANGLE_VERTS * f + 1],
                mesh.indices[TRIANGLE_VERTS * f + 2],
            ];
            let normal = face_normal(mesh, a, b, c);
            // Duplicate the loop, shifted along the face normal.
            let mut loop_new = [0u32; TRIANGLE_VERTS];
            for (k, v) in [a, b, c].iter().enumerate() {
                let p = mesh.positions[*v as usize];
                mesh.positions.push([
                    p[0] + normal[0] * depth,
                    p[1] + normal[1] * depth,
                    p[2] + normal[2] * depth,
                ]);
                mesh.normals.push(mesh.normals[*v as usize]);
                mesh.uvs.push(mesh.uvs[*v as usize]);
                loop_new[k] = mesh.positions.len() as u32 - 1;
            }
            // Cap becomes the shifted loop.
            mesh.indices[TRIANGLE_VERTS * f..TRIANGLE_VERTS * f + TRIANGLE_VERTS]
                .copy_from_slice(&loop_new);
            // Side walls: one quad (two triangles) per loop edge.
            let old = [a, b, c];
            for e in 0..TRIANGLE_VERTS {
                let o0 = old[e];
                let o1 = old[(e + 1) % TRIANGLE_VERTS];
                let n0 = loop_new[e];
                let n1 = loop_new[(e + 1) % TRIANGLE_VERTS];
                affected.push(mesh.triangle_count() as u32);
                mesh.indices.extend([o0, o1, n1, o0, n1, n0]);
            }
            affected.push(f as u32);
        }
        crate::recompute_normals(mesh, None);
        self.finish_preview(
            crate::MeshDirty::TOPO | crate::MeshDirty::NORMALS,
            &affected,
            started,
        );
    }

    /// Midpoint subdivision: every triangle becomes 4 per level.
    ///
    /// Edge midpoints are deduplicated through a hash map so shared edges
    /// stay welded; shading normals are rebuilt afterwards.
    pub fn apply_subdivide(&mut self, levels: u32) {
        let started = Instant::now();
        let mut affected: Vec<u32> = Vec::new();
        for _ in 0..levels {
            let mesh = self.preview_or_clone();
            let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
            let mut new_indices = Vec::with_capacity(mesh.indices.len() * 4);
            let tris = mesh.triangle_count();
            for f in 0..tris {
                let [a, b, c] = [
                    mesh.indices[TRIANGLE_VERTS * f],
                    mesh.indices[TRIANGLE_VERTS * f + 1],
                    mesh.indices[TRIANGLE_VERTS * f + 2],
                ];
                let mab = midpoint_vertex(mesh, &mut midpoints, a, b);
                let mbc = midpoint_vertex(mesh, &mut midpoints, b, c);
                let mca = midpoint_vertex(mesh, &mut midpoints, c, a);
                new_indices.extend([a, mab, mca, mab, b, mbc, mca, mbc, c, mab, mbc, mca]);
            }
            mesh.indices = new_indices;
            crate::recompute_normals(mesh, None);
            affected.extend(0..mesh.triangle_count() as u32);
        }
        self.finish_preview(
            crate::MeshDirty::TOPO | crate::MeshDirty::NORMALS,
            &affected,
            started,
        );
    }

    /// Accept the preview as the new base version.
    pub fn commit(&mut self) {
        if let Some(mesh) = self.preview.take() {
            self.base = mesh;
            self.base_seq = self.preview_seq;
        }
        self.dirty.clear();
        self.refresh_stats();
    }

    /// Drop the preview, keep the base version.
    pub fn cancel(&mut self) {
        self.preview = None;
        self.dirty.clear();
        self.refresh_stats();
    }

    /// Apply one recorded op through the preview path, logging `(op, before)`
    /// into `undo` first. Preview-capable ops (transform/extrude/subdivide)
    /// and session ops (commit/cancel) run here; exact ops (boolean/bevel)
    /// need the background worker.
    ///
    /// # Errors
    ///
    /// Returns [`UndoError::RequiresExactWorker`](crate::UndoError) for
    /// boolean/bevel (state untouched, nothing logged), or the history
    /// denial from [`UndoStack::push`](crate::UndoStack) (state untouched —
    /// the log write happens before any mutation).
    pub fn apply_op(
        &mut self,
        op: &crate::EditOp,
        undo: &mut crate::UndoStack,
    ) -> Result<(), crate::UndoError> {
        match op {
            crate::EditOp::Transform { matrix } => {
                undo.push(op.clone(), self.view().clone())?;
                self.apply_transform(*matrix);
                Ok(())
            }
            crate::EditOp::Extrude { faces, depth } => {
                undo.push(op.clone(), self.view().clone())?;
                self.apply_extrude(faces, *depth);
                Ok(())
            }
            crate::EditOp::Subdivide { levels } => {
                undo.push(op.clone(), self.view().clone())?;
                self.apply_subdivide(*levels);
                Ok(())
            }
            crate::EditOp::CommitExact => {
                undo.push(op.clone(), self.base.clone())?;
                self.commit();
                Ok(())
            }
            crate::EditOp::CancelPreview => {
                undo.push(op.clone(), self.view().clone())?;
                self.cancel();
                Ok(())
            }
            crate::EditOp::Boolean { .. } | crate::EditOp::Bevel { .. } => {
                Err(crate::UndoError::RequiresExactWorker)
            }
        }
    }

    /// Undo the last logged op: its pre-op view becomes the new preview.
    /// The preview sequence advances, so in-flight exact results go stale
    /// (an undo is a newer edit than any pending job).
    ///
    /// # Errors
    ///
    /// Returns [`UndoError::Empty`](crate::UndoError) when no entries are
    /// retained (state untouched).
    pub fn undo(&mut self, stack: &mut crate::UndoStack) -> Result<(), crate::UndoError> {
        let restored = stack.undo(self.view())?;
        self.preview = Some(restored);
        self.preview_seq = self.preview_seq.saturating_add(1);
        self.dirty.set(
            crate::MeshDirty::VERTS
                | crate::MeshDirty::TOPO
                | crate::MeshDirty::NORMALS
                | crate::MeshDirty::GPU_UPLOAD,
            &[],
        );
        self.refresh_stats();
        Ok(())
    }

    /// Redo the newest undone op, symmetric to [`undo`](Self::undo).
    ///
    /// # Errors
    ///
    /// Returns [`UndoError::RedoEmpty`](crate::UndoError) when nothing was
    /// undone (state untouched).
    pub fn redo(&mut self, stack: &mut crate::UndoStack) -> Result<(), crate::UndoError> {
        let restored = stack.redo(self.view())?;
        self.preview = Some(restored);
        self.preview_seq = self.preview_seq.saturating_add(1);
        self.dirty.set(
            crate::MeshDirty::VERTS
                | crate::MeshDirty::TOPO
                | crate::MeshDirty::NORMALS
                | crate::MeshDirty::GPU_UPLOAD,
            &[],
        );
        self.refresh_stats();
        Ok(())
    }

    /// Submit the current view to the background worker, tagging the job
    /// with the current preview sequence. The returned [`Seq`](crate::Seq)
    /// is the freshness key [`poll_exact`](Self::poll_exact) compares
    /// against: preview edits after this call advance `preview_seq` past
    /// it and the result lands stale (counted, never swapped).
    pub fn submit_exact(&self, worker: &crate::ExactWorker, op: crate::ExactOp) -> crate::Seq {
        let seq = crate::Seq::new(self.preview_seq);
        worker.submit_seq(self.view().clone(), op, seq);
        seq
    }

    /// Poll the worker for the newest finished result and apply the swap
    /// criterion: fresh (`seq >= preview_seq`) plus a coherent preview
    /// adopts the result as the new base and clears the preview; stale
    /// keeps the preview and counts the drop. Nothing pending keeps the
    /// preview without touching state.
    pub fn poll_exact(&mut self, worker: &crate::ExactWorker) -> crate::SwapDecision {
        let Some(result) = worker.try_recv() else {
            return crate::SwapDecision::KeepPreview;
        };
        self.stats
            .observe_exact_result(&result, worker.thread_count() as u32);
        let decision = crate::decide_swap(
            crate::Seq::new(self.preview_seq),
            self.stats.preview_coherence(),
            crate::Seq::new(result.seq),
        );
        match decision {
            crate::SwapDecision::SwapExact(_) => {
                self.base = result.mesh;
                self.base_seq = result.seq;
                self.preview = None;
                self.dirty.set(
                    crate::MeshDirty::VERTS
                        | crate::MeshDirty::TOPO
                        | crate::MeshDirty::NORMALS
                        | crate::MeshDirty::GPU_UPLOAD,
                    &[],
                );
                self.refresh_stats();
            }
            crate::SwapDecision::KeepPreview => {
                self.stats.dropped_exact_seq = self.stats.dropped_exact_seq.saturating_add(1);
            }
        }
        decision
    }

    /// Preview working copy, cloning the base on first edit of a session.
    fn preview_or_clone(&mut self) -> &mut crate::MeshData {
        if self.preview.is_none() {
            self.preview = Some(self.base.clone());
        }
        self.preview.as_mut().expect("preview just created")
    }

    /// Bump the preview sequence, record dirty flags and frame time.
    fn finish_preview(&mut self, flags: u8, faces: &[u32], started: Instant) {
        self.preview_seq += 1;
        let flags = flags | crate::MeshDirty::GPU_UPLOAD;
        self.dirty.set(flags, faces);
        self.refresh_stats();
        self.stats.preview_us = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
    }

    /// Refresh displayed vertex/triangle counters.
    fn refresh_stats(&mut self) {
        let view = self.preview.as_ref().unwrap_or(&self.base);
        self.stats.verts = view.vertex_count();
        self.stats.tris = view.triangle_count();
    }
}

/// Unit face normal of triangle (`a`, `b`, `c`).
fn face_normal(mesh: &crate::MeshData, a: u32, b: u32, c: u32) -> [f32; 3] {
    let pa = mesh.positions[a as usize];
    let pb = mesh.positions[b as usize];
    let pc = mesh.positions[c as usize];
    let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
    let n = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > f32::EPSILON {
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Midpoint vertex of edge (`a`, `b`), reused across adjacent triangles.
fn midpoint_vertex(
    mesh: &mut crate::MeshData,
    cache: &mut HashMap<(u32, u32), u32>,
    a: u32,
    b: u32,
) -> u32 {
    let key = (a.min(b), a.max(b));
    if let Some(&v) = cache.get(&key) {
        return v;
    }
    let pa = mesh.positions[a as usize];
    let pb = mesh.positions[b as usize];
    mesh.positions.push([
        (pa[0] + pb[0]) * 0.5,
        (pa[1] + pb[1]) * 0.5,
        (pa[2] + pb[2]) * 0.5,
    ]);
    let na = mesh.normals[a as usize];
    let nb = mesh.normals[b as usize];
    mesh.normals.push([
        (na[0] + nb[0]) * 0.5,
        (na[1] + nb[1]) * 0.5,
        (na[2] + nb[2]) * 0.5,
    ]);
    let ua = mesh.uvs[a as usize];
    let ub = mesh.uvs[b as usize];
    mesh.uvs
        .push([(ua[0] + ub[0]) * 0.5, (ua[1] + ub[1]) * 0.5]);
    let v = mesh.positions.len() as u32 - 1;
    cache.insert(key, v);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_shifts_box() {
        let mut edit = EditableMesh::new(crate::MeshData::unit_box());
        edit.apply_transform(glam::Mat4::from_translation(glam::Vec3::new(1.0, 0.0, 0.0)));
        let view = edit.view();
        assert!(
            view.positions.iter().all(|p| p[0] >= 0.5 - 1e-5),
            "box moved +x"
        );
        assert!(!edit.dirty.is_clean());
        edit.commit();
        assert!(edit.preview.is_none());
        assert!(edit.dirty.is_clean());
    }

    #[test]
    fn subdivide_quadruples_triangles() {
        let mut edit = EditableMesh::new(crate::MeshData::unit_box());
        let before = edit.view().triangle_count();
        edit.apply_subdivide(1);
        assert_eq!(edit.view().triangle_count(), before * 4);
        edit.cancel();
        assert_eq!(edit.view().triangle_count(), before);
    }

    #[test]
    fn extrude_grows_triangle_count() {
        let mut edit = EditableMesh::new(crate::MeshData::unit_box());
        let before = edit.view().triangle_count();
        edit.apply_extrude(&[0], 0.25);
        assert!(edit.view().triangle_count() > before);
        assert!(edit.dirty.contains(crate::MeshDirty::TOPO));
        assert!(edit.dirty.contains(crate::MeshDirty::NORMALS));
        assert!(!edit.dirty.affected.is_empty());
    }

    #[test]
    fn boolean_union_through_bridge() {
        let a = crate::MeshData::unit_box();
        let mut b = crate::MeshData::unit_box();
        for p in &mut b.positions {
            p[0] += 0.5;
        }
        let out = crate::boolean(&a, &b, crate::BooleanKind::Union).expect("union works");
        assert!(out.triangle_count() > 0);
    }

    #[test]
    fn apply_op_undo_redo_roundtrip() {
        let mut edit = EditableMesh::new(crate::MeshData::unit_box());
        let mut undo = crate::UndoStack::with_default_cap(crate::UndoStrategy::Ops);
        let original = edit.view().positions.clone();
        let op = crate::EditOp::Transform {
            matrix: glam::Mat4::from_translation(glam::Vec3::new(1.0, 0.0, 0.0)),
        };
        edit.apply_op(&op, &mut undo).expect("preview op applies");
        assert!(
            edit.view().positions.iter().all(|p| p[0] >= 0.5 - 1e-5),
            "box moved +x"
        );
        edit.undo(&mut undo).expect("undo restores original");
        assert_eq!(edit.view().positions, original);
        edit.redo(&mut undo).expect("redo reapplies");
        assert!(
            edit.view().positions.iter().all(|p| p[0] >= 0.5 - 1e-5),
            "box moved +x again"
        );
    }

    #[test]
    fn exact_op_rejected_without_state_change() {
        let mut edit = EditableMesh::new(crate::MeshData::unit_box());
        let mut undo = crate::UndoStack::with_default_cap(crate::UndoStrategy::Ops);
        let original = edit.view().positions.clone();
        let op = crate::EditOp::Boolean {
            kind: crate::BooleanKind::Union,
            tool: crate::MeshData::unit_box(),
            tool_matrix: glam::Mat4::IDENTITY,
        };
        assert!(matches!(
            edit.apply_op(&op, &mut undo),
            Err(crate::UndoError::RequiresExactWorker)
        ));
        assert_eq!(edit.view().positions, original);
        assert!(undo.is_empty());
        assert!(matches!(edit.undo(&mut undo), Err(crate::UndoError::Empty)));
    }

    #[test]
    fn poll_exact_applies_fresh_result() {
        use std::time::{Duration, Instant};
        let mut edit = EditableMesh::new(crate::MeshData::unit_box());
        let worker = crate::ExactWorker::spawn();
        // `BevelAll` with non-positive radius clones the snapshot: the fast
        // worker path, no kernel timing in the assertion.
        let submitted = edit.submit_exact(&worker, crate::ExactOp::BevelAll { radius: 0.0 });
        assert_eq!(submitted, crate::Seq::new(edit.preview_seq));
        let deadline = Instant::now() + Duration::from_secs(5);
        let swapped = loop {
            match edit.poll_exact(&worker) {
                crate::SwapDecision::SwapExact(seq) => break seq,
                crate::SwapDecision::KeepPreview if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                crate::SwapDecision::KeepPreview => {
                    panic!("fresh exact result never arrived");
                }
            }
        };
        assert_eq!(swapped, submitted);
        assert!(edit.preview.is_none());
        assert_eq!(edit.base_seq, submitted.get());
    }

    #[test]
    fn poll_exact_keeps_newer_preview() {
        use std::time::{Duration, Instant};
        let mut edit = EditableMesh::new(crate::MeshData::unit_box());
        let worker = crate::ExactWorker::spawn();
        edit.submit_exact(&worker, crate::ExactOp::BevelAll { radius: 0.0 });
        // A newer preview edit supersedes the in-flight job however fast
        // the worker clones: its seq is fixed at submit time.
        edit.apply_transform(glam::Mat4::from_translation(glam::Vec3::X));
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let _ = edit.poll_exact(&worker);
            if edit.stats.dropped_exact_seq == 1 {
                break;
            }
            assert!(Instant::now() < deadline, "stale result never arrived");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(edit.preview.is_some(), "newer preview survives");
    }
}
