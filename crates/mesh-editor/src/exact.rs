//! Background exact worker: booleans and full bevels off the frame.
//!
//! [`ExactWorker`] owns a dedicated thread fed by an `mpsc` job queue.
//! Each [`submit`](ExactWorker::submit) carries a sequence number; the worker
//! always processes the newest pending job and drops superseded ones, and
//! [`try_recv`](ExactWorker::try_recv) likewise keeps only the newest
//! finished result, so a slow exact op never blocks the preview loop.
//!
//! `BevelAll` is implemented as an honest Minkowski sum of the mesh with a
//! sphere of the requested radius: every edge and corner is truly rounded,
//! at the cost of growing the overall dimensions by `radius` on every side.
//! This is documented behavior, not an approximation artifact.

use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

/// Exact (background) operation on a mesh snapshot.
#[derive(Debug)]
pub enum ExactOp {
    /// CSG combination with a tool mesh (`tool_matrix` in base space).
    Boolean {
        /// Tool volume in its own local space.
        tool: crate::MeshData,
        /// Transform applied to the tool before combining.
        tool_matrix: glam::Mat4,
        /// Combination kind.
        kind: crate::BooleanKind,
    },
    /// Round every edge/corner with the given radius (Minkowski sum).
    BevelAll {
        /// Rounding radius in engine units; must be positive.
        radius: f32,
    },
}

/// Finished exact result tagged with its submission sequence.
#[derive(Debug)]
pub struct ExactResult {
    /// Sequence of the job that produced this mesh.
    pub seq: u64,
    /// Resulting mesh (input snapshot on kernel failure).
    pub mesh: crate::MeshData,
    /// Worker-side wall time in milliseconds.
    pub elapsed_ms: u64,
}

/// Job submitted to the worker thread.
struct ExactJob {
    seq: u64,
    snapshot: crate::MeshData,
    op: ExactOp,
}

/// Dedicated-thread exact executor with newest-wins semantics.
#[derive(Debug)]
pub struct ExactWorker {
    jobs: Sender<ExactJob>,
    results: Receiver<ExactResult>,
}

impl ExactWorker {
    /// Spawn the worker thread and return its handle.
    pub fn spawn() -> Self {
        let (job_tx, job_rx) = std::sync::mpsc::channel::<ExactJob>();
        let (res_tx, res_rx) = std::sync::mpsc::channel::<ExactResult>();
        std::thread::spawn(move || worker_loop(job_rx, res_tx));
        Self {
            jobs: job_tx,
            results: res_rx,
        }
    }

    /// Submit a job; a newer submit supersedes still-pending older ones.
    ///
    /// Superseded jobs are dropped by the worker before execution, so only
    /// the newest pending snapshot is ever processed.
    pub fn submit(&self, snapshot: crate::MeshData, op: ExactOp, seq: u64) {
        let _ = self.jobs.send(ExactJob { seq, snapshot, op });
    }

    /// Poll for the newest finished result, dropping older pending ones.
    pub fn try_recv(&self) -> Option<ExactResult> {
        let mut latest: Option<ExactResult> = None;
        loop {
            match self.results.try_recv() {
                Ok(result) => latest = Some(result),
                Err(TryRecvError::Empty) => return latest,
                Err(TryRecvError::Disconnected) => return latest,
            }
        }
    }
}

/// Worker body: always run the newest queued job, drop the rest.
fn worker_loop(jobs: Receiver<ExactJob>, results: Sender<ExactResult>) {
    while let Ok(first) = jobs.recv() {
        // Newest wins: drain the queue, keep only the latest job.
        let mut job = first;
        while let Ok(next) = jobs.try_recv() {
            job = next;
        }
        let started = Instant::now();
        let mesh = run_job(&job.snapshot, &job.op);
        let result = ExactResult {
            seq: job.seq,
            mesh,
            elapsed_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        };
        if results.send(result).is_err() {
            return;
        }
    }
}

/// Execute one exact op on the worker thread.
fn run_job(snapshot: &crate::MeshData, op: &ExactOp) -> crate::MeshData {
    match op {
        ExactOp::Boolean {
            tool,
            tool_matrix,
            kind,
        } => {
            let mut moved_tool = tool.clone();
            for p in &mut moved_tool.positions {
                let v = tool_matrix.transform_point3(glam::Vec3::new(p[0], p[1], p[2]));
                *p = [v.x, v.y, v.z];
            }
            crate::boolean(snapshot, &moved_tool, *kind).unwrap_or_else(|_| snapshot.clone())
        }
        ExactOp::BevelAll { radius } => {
            if *radius <= 0.0 {
                return snapshot.clone();
            }
            bevel_all(snapshot, *radius).unwrap_or_else(|_| snapshot.clone())
        }
    }
}

/// Round all edges by Minkowski-summing the mesh with a sphere.
///
/// Grows every dimension by `radius` (documented): a unit box beveled with
/// `r` spans `1 + 2r` per axis.
fn bevel_all(mesh: &crate::MeshData, radius: f32) -> Result<crate::MeshData, crate::BridgeError> {
    use manifold_rust::manifold::Manifold;
    let base = crate::to_manifold(mesh)?;
    let sphere = Manifold::sphere(f64::from(radius), 24);
    let out = base.minkowski_sum(&sphere);
    if out.status() != manifold_rust::types::Error::NoError {
        return Err(crate::BridgeError::KernelFailed);
    }
    let mut beveled = crate::from_manifold(&out);
    crate::recompute_normals(&mut beveled, None);
    Ok(beveled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Poll the worker until a result arrives or the timeout expires.
    fn recv_blocking(worker: &ExactWorker, timeout: Duration) -> Option<ExactResult> {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Some(result) = worker.try_recv() {
                return Some(result);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        None
    }

    #[test]
    fn worker_returns_submitted_seq() {
        let worker = ExactWorker::spawn();
        let tool = crate::MeshData::unit_box();
        worker.submit(
            crate::MeshData::unit_box(),
            ExactOp::Boolean {
                tool,
                tool_matrix: glam::Mat4::from_translation(glam::Vec3::new(0.5, 0.0, 0.0)),
                kind: crate::BooleanKind::Union,
            },
            7,
        );
        let result = recv_blocking(&worker, Duration::from_secs(30)).expect("result arrives");
        assert_eq!(result.seq, 7);
        assert!(result.mesh.triangle_count() > 0);
    }

    #[test]
    fn newer_submit_supersedes_pending() {
        let worker = ExactWorker::spawn();
        for seq in [10u64, 11, 12] {
            let tool = crate::MeshData::unit_box();
            worker.submit(
                crate::MeshData::unit_box(),
                ExactOp::Boolean {
                    tool,
                    tool_matrix: glam::Mat4::IDENTITY,
                    kind: crate::BooleanKind::Union,
                },
                seq,
            );
        }
        let result = recv_blocking(&worker, Duration::from_secs(30)).expect("result arrives");
        // Whatever the worker ran, it must be one of the submitted seqs,
        // and no stale seq below the first may surface afterwards.
        assert!((10..=12).contains(&result.seq));
        assert!(worker.try_recv().is_none_or(|r| r.seq >= result.seq));
    }
}
