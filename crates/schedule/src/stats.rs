//! Opt-in timing for the level executor plus the frame-time aggregation
//! the debug stats window reads: [`NodeTimings`] accumulates per-node
//! wall time lock-free (safe under the rayon executor), [`run_levels_timed`]
//! wraps [`run_levels`](crate::run_levels) with `Instant` probes, and
//! [`FrameHistory`] is a fixed-capacity ring buffer of frame durations
//! with average / recent-worst queries. Nothing here runs unless a
//! frontend opts in, so the untimed path stays exactly as before.
//!
//! `std::time::Instant` panics on `wasm32-unknown-unknown`, so the timed
//! executor exists only off wasm; the aggregation types are portable.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
#[cfg(not(target_family = "wasm"))]
use std::time::Instant;

/// Per-node wall-time accumulator for one schedule.
///
/// Slot `i` sums every run of node `i` since the last [`drain`](Self::drain)
/// (a fixed-rate schedule can run several times per frame); one extra
/// slot holds the wall time of whole executor passes. Counters are
/// relaxed atomics: nodes of one level record concurrently without locks.
#[derive(Debug, Default)]
pub struct NodeTimings {
    /// Per-node nanoseconds, followed by the pass-total slot.
    nanos: Vec<AtomicU64>,
}

/// One drained [`NodeTimings`] snapshot.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TimingSnapshot {
    /// Accumulated wall time per node (registration index).
    pub nodes: Vec<Duration>,
    /// Accumulated wall time of whole executor passes.
    pub total: Duration,
}

impl NodeTimings {
    /// Zeroed accumulator for `nodes` nodes.
    pub fn new(nodes: usize) -> Self {
        Self {
            nanos: (0..=nodes).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Number of tracked nodes.
    pub fn len(&self) -> usize {
        self.nanos.len().saturating_sub(1)
    }

    /// Whether no node is tracked.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds `elapsed` to node `index`; out-of-range indices are ignored.
    pub fn record(&self, index: usize, elapsed: Duration) {
        if index < self.len() {
            add_nanos(&self.nanos[index], elapsed);
        }
    }

    /// Adds `elapsed` to the pass-total slot.
    pub fn record_total(&self, elapsed: Duration) {
        if let Some(slot) = self.nanos.last() {
            add_nanos(slot, elapsed);
        }
    }

    /// Returns the accumulated times and resets every counter to zero.
    pub fn drain(&self) -> TimingSnapshot {
        let mut nodes: Vec<Duration> = self
            .nanos
            .iter()
            .map(|slot| Duration::from_nanos(slot.swap(0, Ordering::Relaxed)))
            .collect();
        let total = nodes.pop().unwrap_or_default();
        TimingSnapshot { nodes, total }
    }
}

fn add_nanos(slot: &AtomicU64, elapsed: Duration) {
    let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    slot.fetch_add(nanos, Ordering::Relaxed);
}

/// [`run_levels`](crate::run_levels) with every node and the whole pass
/// timed into `timings` (same ordering and parallelism semantics).
#[cfg(not(target_family = "wasm"))]
pub fn run_levels_timed(
    levels: &[Vec<usize>],
    nodes: usize,
    parallel: bool,
    timings: &NodeTimings,
    run: impl Fn(usize) + Sync,
) {
    let start = Instant::now();
    crate::run_levels(levels, nodes, parallel, |i| {
        let node_start = Instant::now();
        run(i);
        timings.record(i, node_start.elapsed());
    });
    timings.record_total(start.elapsed());
}

/// Fixed-capacity ring buffer of frame durations (oldest overwritten).
///
/// Answers what the stats window shows: the mean over the buffer, the
/// worst frame inside a recent wall-time window, and the samples in
/// chronological order for the history graph.
#[derive(Debug, Clone)]
pub struct FrameHistory {
    samples: Vec<Duration>,
    /// Index the next sample is written to.
    head: usize,
    capacity: usize,
}

impl FrameHistory {
    /// Empty history holding at most `capacity` frames (minimum one).
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            samples: Vec::with_capacity(capacity),
            head: 0,
            capacity,
        }
    }

    /// Maximum number of retained frames.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of retained frames.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether no frame was pushed yet.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Appends one frame, overwriting the oldest when full.
    pub fn push(&mut self, frame: Duration) {
        if self.samples.len() < self.capacity {
            self.samples.push(frame);
        } else {
            self.samples[self.head] = frame;
        }
        self.head = (self.head + 1) % self.capacity;
    }

    /// Frames newest first.
    pub fn iter_newest_first(&self) -> impl Iterator<Item = Duration> + '_ {
        let (newer, older) = self.samples.split_at(self.split_point());
        newer.iter().rev().chain(older.iter().rev()).copied()
    }

    /// Frames oldest first (graph order).
    pub fn iter_chronological(&self) -> impl Iterator<Item = Duration> + '_ {
        let (newer, older) = self.samples.split_at(self.split_point());
        older.iter().chain(newer.iter()).copied()
    }

    /// Mean frame time over the retained frames (zero when empty).
    pub fn average(&self) -> Duration {
        let count = u32::try_from(self.samples.len()).unwrap_or(u32::MAX);
        if count == 0 {
            return Duration::ZERO;
        }
        self.samples.iter().sum::<Duration>() / count
    }

    /// Worst frame among the newest frames that fit into `window` of
    /// wall time (the newest frame always counts, so a single long hitch
    /// is never hidden).
    pub fn worst_within(&self, window: Duration) -> Duration {
        let mut covered = Duration::ZERO;
        let mut worst = Duration::ZERO;
        for frame in self.iter_newest_first() {
            if covered >= window {
                break;
            }
            covered += frame;
            worst = worst.max(frame);
        }
        worst
    }

    /// Frames per second over the retained frames (zero when empty).
    pub fn fps(&self) -> f64 {
        let average = self.average().as_secs_f64();
        if average > 0.0 { 1.0 / average } else { 0.0 }
    }

    /// Boundary between the wrapped-around newest part and the oldest
    /// part of the backing vector.
    fn split_point(&self) -> usize {
        if self.samples.len() < self.capacity {
            self.samples.len()
        } else {
            self.head
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn history_average_and_fps() {
        let mut h = FrameHistory::new(4);
        assert_eq!(h.average(), Duration::ZERO);
        assert_eq!(h.fps(), 0.0);
        h.push(ms(10));
        h.push(ms(30));
        assert_eq!(h.average(), ms(20));
        assert!((h.fps() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn ring_buffer_overwrites_oldest_and_keeps_order() {
        let mut h = FrameHistory::new(3);
        for n in 1..=5 {
            h.push(ms(n));
        }
        assert_eq!(h.len(), 3);
        assert_eq!(h.capacity(), 3);
        let chrono: Vec<_> = h.iter_chronological().collect();
        assert_eq!(chrono, vec![ms(3), ms(4), ms(5)]);
        let newest: Vec<_> = h.iter_newest_first().collect();
        assert_eq!(newest, vec![ms(5), ms(4), ms(3)]);
        assert_eq!(h.average(), ms(4));
    }

    #[test]
    fn ring_buffer_partial_fill_order() {
        let mut h = FrameHistory::new(8);
        h.push(ms(1));
        h.push(ms(2));
        assert_eq!(
            h.iter_chronological().collect::<Vec<_>>(),
            vec![ms(1), ms(2)]
        );
        assert!(!h.is_empty());
        assert!(FrameHistory::new(0).capacity() == 1);
    }

    #[test]
    fn worst_is_limited_to_recent_window() {
        let mut h = FrameHistory::new(16);
        h.push(ms(500)); // old hitch, outside the window
        for _ in 0..10 {
            h.push(ms(100));
        }
        assert_eq!(h.worst_within(Duration::from_secs(1)), ms(100));
        h.push(ms(250));
        assert_eq!(h.worst_within(Duration::from_secs(1)), ms(250));
        // The newest frame always counts, even when longer than the window.
        h.push(ms(3000));
        assert_eq!(h.worst_within(Duration::from_secs(1)), ms(3000));
        assert_eq!(FrameHistory::new(2).worst_within(ms(10)), Duration::ZERO);
    }

    #[test]
    fn node_timings_accumulate_and_drain() {
        let t = NodeTimings::new(2);
        assert_eq!(t.len(), 2);
        assert!(!t.is_empty());
        t.record(0, ms(1));
        t.record(0, ms(2));
        t.record(1, ms(5));
        t.record(9, ms(7)); // ignored
        t.record_total(ms(9));
        let snap = t.drain();
        assert_eq!(snap.nodes, vec![ms(3), ms(5)]);
        assert_eq!(snap.total, ms(9));
        assert_eq!(t.drain().nodes, vec![Duration::ZERO; 2]);
        assert!(NodeTimings::default().is_empty());
    }

    #[test]
    fn timed_executor_runs_every_node_and_records() {
        let t = NodeTimings::new(3);
        let ran = AtomicU64::new(0);
        run_levels_timed(&[vec![0, 1], vec![2]], 3, true, &t, |i| {
            ran.fetch_add(1 << i, Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(1));
        });
        assert_eq!(ran.load(Ordering::Relaxed), 0b111);
        let snap = t.drain();
        assert!(snap.nodes.iter().all(|d| *d >= ms(1)));
        assert!(snap.total >= ms(1));
    }
}
