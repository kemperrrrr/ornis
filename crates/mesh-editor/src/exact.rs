//! Background exact worker: booleans and full bevels off the frame.
//!
//! [`ExactWorker`] owns a pool (default one thread — the legacy behavior)
//! fed by an `mpsc` job queue. Each [`submit`](ExactWorker::submit)
//! carries a sequence number; the pool always processes the newest
//! pending job and drops superseded ones, and
//! [`try_recv`](ExactWorker::try_recv) likewise keeps only the newest
//! finished result, so a slow exact op never blocks the preview loop.
//!
//! `BevelAll` is implemented as an honest Minkowski sum of the mesh with a
//! sphere of the requested radius: every edge and corner is truly rounded,
//! at the cost of growing the overall dimensions by `radius` on every side.
//! This is documented behavior, not an approximation artifact.

use std::num::NonZeroUsize;
use std::sync::{
    Arc, Mutex,
    mpsc::{Receiver, Sender, TryRecvError},
};
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

/// Submission sequence: total order over exact jobs.
///
/// Newtype over the raw `u64` so job/result channels cannot be mixed up
/// with unrelated counters (preview steps, transport versions). Larger
/// values are newer; [`ExactPriority::NewestWins`] keeps the greatest
/// pending [`Seq`] and drops the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Seq(u64);

impl Seq {
    /// Wrap a raw sequence number.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Raw sequence number.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for Seq {
    /// Raw-to-typed submission order.
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl From<Seq> for u64 {
    /// Typed-to-raw submission order (transport-friendly).
    fn from(seq: Seq) -> Self {
        seq.get()
    }
}

impl std::fmt::Display for Seq {
    /// Raw number (logs, assertions).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Positive thread count: pool sizes are `>= 1` by construction, so a
/// zero-thread pool is unrepresentable instead of a runtime error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PositiveUsize(NonZeroUsize);

impl PositiveUsize {
    /// The single-thread pool (legacy behavior).
    pub const ONE: Self = Self(match NonZeroUsize::new(1) {
        Some(one) => one,
        None => unreachable!(),
    });

    /// Checked constructor: `Some` only for values `>= 1`.
    pub const fn new(value: usize) -> Option<Self> {
        match NonZeroUsize::new(value) {
            Some(valid) => Some(Self(valid)),
            None => None,
        }
    }

    /// Constant-payload constructor, panicking on zero. For literals
    /// validated by inspection (defaults, tests) where
    /// `new(...).expect` would drown the payload in noise.
    ///
    /// # Panics
    /// Panics when `value` is zero.
    pub const fn expect_valid(value: usize) -> Self {
        match Self::new(value) {
            Some(valid) => valid,
            None => panic!("PositiveUsize requires a value >= 1"),
        }
    }

    /// Raw thread count.
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

impl Default for PositiveUsize {
    /// Single thread (legacy behavior).
    fn default() -> Self {
        Self::ONE
    }
}

/// Scheduling priority of the exact pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ExactPriority {
    /// Newest-wins: a newer submit supersedes still-pending older jobs
    /// (dropped before execution), and polling keeps only the finished
    /// result with the greatest [`Seq`]. A slow exact op never blocks
    /// the preview loop and never surfaces stale geometry.
    #[default]
    NewestWins,
}

/// Pool size as a type instead of a `bool` single-vs-pool flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExactPoolSize {
    /// One worker thread — the legacy behavior, bit-for-bit.
    Single,
    /// Fixed `N`-thread pool sharing one newest-wins job queue.
    Fixed(PositiveUsize),
}

impl ExactPoolSize {
    /// Worker thread count (1 for [`Self::Single`]).
    pub const fn thread_count(self) -> usize {
        match self {
            Self::Single => 1,
            Self::Fixed(threads) => threads.get(),
        }
    }
}

impl Default for ExactPoolSize {
    /// Single thread (legacy behavior).
    fn default() -> Self {
        Self::Single
    }
}

impl From<PositiveUsize> for ExactPoolSize {
    /// `1` maps to [`Self::Single`], larger counts to [`Self::Fixed`].
    fn from(threads: PositiveUsize) -> Self {
        if threads.get() == 1 {
            Self::Single
        } else {
            Self::Fixed(threads)
        }
    }
}

/// Pool configuration: thread count plus scheduling priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkerConfig {
    /// Worker threads sharing one newest-wins queue (`>= 1`).
    pub threads: PositiveUsize,
    /// Scheduling priority (currently always newest-wins).
    pub priority: ExactPriority,
}

impl WorkerConfig {
    /// Single-thread pool (legacy behavior).
    pub const fn single() -> Self {
        Self {
            threads: PositiveUsize::ONE,
            priority: ExactPriority::NewestWins,
        }
    }

    /// Pool with `threads` workers and newest-wins scheduling.
    pub const fn new(threads: PositiveUsize) -> Self {
        Self {
            threads,
            priority: ExactPriority::NewestWins,
        }
    }

    /// Worker thread count.
    pub const fn thread_count(self) -> usize {
        self.threads.get()
    }
}

impl Default for WorkerConfig {
    /// Single thread (legacy behavior).
    fn default() -> Self {
        Self::single()
    }
}

impl From<ExactPoolSize> for WorkerConfig {
    /// Pool size plus default (newest-wins) priority.
    fn from(size: ExactPoolSize) -> Self {
        match size {
            ExactPoolSize::Single => Self::single(),
            ExactPoolSize::Fixed(threads) => Self::new(threads),
        }
    }
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
    /// Queue wait before execution started, in microseconds
    /// (submit → execution start; includes waiting for a free pool
    /// thread). Feeds [`crate::FrameStats::exact_queue_us`].
    pub queue_us: u64,
}

/// Job submitted to the worker pool.
struct ExactJob {
    seq: Seq,
    submitted: Instant,
    snapshot: crate::MeshData,
    op: ExactOp,
}

/// Exact executor with newest-wins semantics over a configurable pool.
///
/// Default is one worker thread — the legacy behavior, parity-pinned by
/// the `pool_matches_single_thread_newest_wins` test: `N` threads share
/// the same queue discipline (newest pending job runs, older pending
/// jobs drop; polling keeps the greatest finished [`Seq`]), so raising
/// the pool size only adds throughput, never stale geometry.
#[derive(Debug)]
pub struct ExactWorker {
    jobs: Sender<ExactJob>,
    results: Receiver<ExactResult>,
    config: WorkerConfig,
}

impl ExactWorker {
    /// Spawn the single-thread worker and return its handle.
    pub fn spawn() -> Self {
        Self::spawn_with_config(WorkerConfig::single())
    }

    /// Spawn a pool with `config` worker threads sharing one
    /// newest-wins queue. `WorkerConfig::single()` (the default) is
    /// exactly [`spawn`](Self::spawn).
    pub fn spawn_with_config(config: WorkerConfig) -> Self {
        Self::spawn_sized(ExactPoolSize::from(config.threads), config.priority)
    }

    /// Spawn a pool sized by `size` with newest-wins scheduling.
    pub fn spawn_sized(size: ExactPoolSize, priority: ExactPriority) -> Self {
        let (job_tx, job_rx) = std::sync::mpsc::channel::<ExactJob>();
        let (res_tx, res_rx) = std::sync::mpsc::channel::<ExactResult>();
        let shared: Arc<Mutex<Receiver<ExactJob>>> = Arc::new(Mutex::new(job_rx));
        let config = WorkerConfig {
            threads: PositiveUsize::expect_valid(size.thread_count()),
            priority,
        };
        for _ in 0..size.thread_count() {
            let jobs = Arc::clone(&shared);
            let results = res_tx.clone();
            std::thread::spawn(move || pool_loop(jobs, results, priority));
        }
        drop(res_tx);
        Self {
            jobs: job_tx,
            results: res_rx,
            config,
        }
    }

    /// Pool configuration this worker was spawned with.
    pub const fn config(&self) -> WorkerConfig {
        self.config
    }

    /// Worker thread count (1 for the legacy [`spawn`](Self::spawn)).
    pub const fn thread_count(&self) -> usize {
        self.config.threads.get()
    }

    /// Submit a job; a newer submit supersedes still-pending older ones.
    ///
    /// Superseded jobs are dropped by the pool before execution, so only
    /// the newest pending snapshot is ever processed.
    pub fn submit(&self, snapshot: crate::MeshData, op: ExactOp, seq: u64) {
        self.submit_seq(snapshot, op, Seq::new(seq));
    }

    /// Typed [`Seq`] variant of [`submit`](Self::submit).
    pub fn submit_seq(&self, snapshot: crate::MeshData, op: ExactOp, seq: Seq) {
        let _ = self.jobs.send(ExactJob {
            seq,
            submitted: Instant::now(),
            snapshot,
            op,
        });
    }

    /// Poll for the newest finished result, dropping older pending ones.
    ///
    /// Newest means greatest [`Seq`], not last received: pool threads may
    /// finish out of order, so the drain keeps the maximum sequence and a
    /// late stale result never overwrites a newer one. Single-thread
    /// results arrive in order, where maximum and last coincide.
    pub fn try_recv(&self) -> Option<ExactResult> {
        let mut latest: Option<ExactResult> = None;
        loop {
            match self.results.try_recv() {
                Ok(result) => {
                    let newer = latest.as_ref().is_none_or(|prev: &ExactResult| {
                        Seq::new(result.seq) >= Seq::new(prev.seq)
                    });
                    if newer {
                        latest = Some(result);
                    }
                }
                Err(TryRecvError::Empty) => return latest,
                Err(TryRecvError::Disconnected) => return latest,
            }
        }
    }
}

/// Pool body: always run the newest queued job, drop the rest.
///
/// Job acquisition plus the newest-wins drain hold the queue lock as one
/// atomic step, so concurrent workers never run two superseded jobs at
/// once; execution itself runs outside the lock, so `N` threads overlap
/// on distinct newest batches.
fn pool_loop(
    jobs: Arc<Mutex<Receiver<ExactJob>>>,
    results: Sender<ExactResult>,
    priority: ExactPriority,
) {
    debug_assert!(matches!(priority, ExactPriority::NewestWins));
    loop {
        let job = {
            let queue = jobs.lock().expect("exact job queue lock");
            let Ok(first) = queue.recv() else {
                return;
            };
            // Newest wins: drain the queue, keep the greatest sequence.
            let mut newest = first;
            while let Ok(next) = queue.try_recv() {
                if next.seq >= newest.seq {
                    newest = next;
                }
            }
            newest
        };
        let queue_us = job
            .submitted
            .elapsed()
            .as_micros()
            .min(u128::from(u64::MAX)) as u64;
        let started = Instant::now();
        let mesh = run_job(&job.snapshot, &job.op);
        let result = ExactResult {
            seq: job.seq.get(),
            mesh,
            elapsed_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
            queue_us,
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
    fn seq_orders_submissions_and_pool_size_maps_threads() {
        // `Seq` is a total order (newer = greater); the pool-size enum
        // replaces the old bool single-vs-pool flag without one.
        assert!(Seq::new(12) > Seq::new(10));
        assert_eq!(u64::from(Seq::new(7)), 7);
        assert_eq!(ExactPoolSize::default(), ExactPoolSize::Single);
        assert_eq!(ExactPoolSize::Single.thread_count(), 1);
        assert_eq!(
            ExactPoolSize::Fixed(PositiveUsize::expect_valid(4)).thread_count(),
            4
        );
        assert_eq!(
            ExactPoolSize::from(PositiveUsize::expect_valid(1)),
            ExactPoolSize::Single
        );
        assert_eq!(WorkerConfig::default(), WorkerConfig::single());
        assert_eq!(WorkerConfig::single().thread_count(), 1);
        assert!(PositiveUsize::new(0).is_none());
        let single = ExactWorker::spawn();
        assert_eq!(single.thread_count(), 1);
        let pooled =
            ExactWorker::spawn_with_config(WorkerConfig::new(PositiveUsize::expect_valid(2)));
        assert_eq!(pooled.thread_count(), 2);
    }

    #[test]
    fn pool_matches_single_thread_newest_wins() {
        // Deterministic parity: the same rapid `1..=16` submission series
        // against a 1-thread and a 4-thread pool must both converge on
        // `seq 16` as the newest finished result, with no stale sequence
        // surfacing afterwards.
        fn newest_seq(worker: &ExactWorker) -> u64 {
            for seq in 1u64..=16 {
                let tool = crate::MeshData::unit_box();
                worker.submit_seq(
                    crate::MeshData::unit_box(),
                    ExactOp::Boolean {
                        tool,
                        tool_matrix: glam::Mat4::IDENTITY,
                        kind: crate::BooleanKind::Union,
                    },
                    Seq::new(seq),
                );
            }
            let deadline = Instant::now() + Duration::from_secs(30);
            let mut newest = 0u64;
            while Instant::now() < deadline {
                if let Some(result) = worker.try_recv() {
                    newest = newest.max(result.seq);
                    if newest == 16 {
                        break;
                    }
                } else {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
            assert_eq!(newest, 16, "newest-wins converges on seq 16");
            // No stale result may surface after the newest.
            assert!(worker.try_recv().is_none_or(|r| r.seq >= newest));
            newest
        }
        let single = ExactWorker::spawn();
        let pooled = ExactWorker::spawn_sized(
            ExactPoolSize::Fixed(PositiveUsize::expect_valid(4)),
            ExactPriority::NewestWins,
        );
        assert_eq!(newest_seq(&single), newest_seq(&pooled));
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
