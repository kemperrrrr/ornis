//! Frame-budget telemetry for the preview/exact split.
//!
//! Every frame shows a coherent preview (budget ≤0.5–1 ms on M1); the exact
//! boolean converges in the background over N frames. These counters keep
//! the degradation ladder (L0 full preview → L1 decimated → L2 bounding
//! proxy → L3 frozen preview + progress) honest instead of estimated.

/// Measured per-frame cost of the mesh-editing split.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStats {
    /// Preview work done inside the frame, in microseconds.
    pub preview_us: u64,
    /// Last finished exact job wall time, in milliseconds.
    pub exact_ms: u64,
    /// Vertices of the currently displayed mesh.
    pub verts: usize,
    /// Triangles of the currently displayed mesh.
    pub tris: usize,
    /// Exact jobs dropped as stale (a newer `seq` superseded them).
    pub dropped_exact_seq: u64,
    /// Last finished exact job queue wait (submit → execution start),
    /// in microseconds (see [`crate::ExactResult::queue_us`]). Zero when
    /// no exact result has been observed yet.
    pub exact_queue_us: u64,
    /// Pool size that produced the last observed exact result. Zero
    /// means unset (no result observed); otherwise `>= 1`.
    pub exact_threads: u32,
}

impl FrameStats {
    /// Frame slice for a coherent preview, in microseconds (1 ms).
    pub const PREVIEW_BUDGET_US: u64 = 1_000;

    /// True when the preview exceeded its frame budget.
    pub fn preview_over_budget(&self, budget_us: u64) -> bool {
        self.preview_us > budget_us
    }

    /// True when the last preview op fit the coherence budget.
    pub fn preview_within_budget(&self) -> bool {
        self.preview_us <= Self::PREVIEW_BUDGET_US
    }

    /// Coherence witness for the swap gate: `Some` while the preview is
    /// live, `None` once it degrades (decimated/proxy/frozen ladder).
    pub fn preview_coherence(&self) -> Option<PreviewStats> {
        PreviewStats::try_new(self.preview_us as f32 / 1_000.0)
    }

    /// Record the timings of one finished exact result without touching
    /// the mesh counters: wall time plus queue wait come from the
    /// result, the pool size from the worker that produced it
    /// ([`crate::ExactWorker::thread_count`]).
    pub fn observe_exact_result(&mut self, result: &crate::ExactResult, threads: u32) {
        self.exact_ms = result.elapsed_ms;
        self.exact_queue_us = result.queue_us;
        self.exact_threads = threads;
    }
}

/// Preview frame proven inside the 1 ms coherence budget.
///
/// The ≤1.0 ms invariant lives in the type: the field is private and the
/// only constructor ([`PreviewStats::try_new`]) rejects anything outside
/// `0.0..=1.0` (including `NaN`/infinity). [`decide_swap`](crate::decide_swap)
/// treats the witness as the "preview is live" half of the swap criterion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreviewStats {
    coherent_ms: f32,
}

impl PreviewStats {
    /// Coherence ceiling in milliseconds: a costlier preview frame is
    /// degraded (L1+ ladder), not coherent.
    pub const BUDGET_MS: f32 = 1.0;

    /// Witness constructor: `Some` only for finite costs within budget.
    pub fn try_new(coherent_ms: f32) -> Option<Self> {
        if (0.0..=Self::BUDGET_MS).contains(&coherent_ms) {
            Some(Self { coherent_ms })
        } else {
            None
        }
    }

    /// Witnessed preview cost in milliseconds (always `≤ 1.0`).
    pub fn coherent_ms(self) -> f32 {
        self.coherent_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coherence_witness_enforces_1ms_invariant() {
        assert_eq!(FrameStats::PREVIEW_BUDGET_US, 1_000);
        assert_eq!(PreviewStats::BUDGET_MS, 1.0);
        assert!(PreviewStats::try_new(0.5).is_some());
        assert!(PreviewStats::try_new(1.0).is_some());
        for bad in [1.000_001, f32::INFINITY, f32::NAN, -0.5] {
            assert!(PreviewStats::try_new(bad).is_none(), "rejected: {bad}");
        }
        assert_eq!(
            PreviewStats::try_new(0.25)
                .expect("in budget")
                .coherent_ms(),
            0.25
        );
    }

    #[test]
    fn frame_stats_coherence_plumbing_is_deterministic() {
        // Synthetic counters only: no wall-clock read, so the test cannot
        // flake under CI load. The budget constants above carry the budget.
        let live = FrameStats {
            preview_us: 500,
            ..FrameStats::default()
        };
        assert!(live.preview_within_budget());
        assert!(!live.preview_over_budget(FrameStats::PREVIEW_BUDGET_US));
        assert!(live.preview_coherence().is_some());
        let degraded = FrameStats {
            preview_us: 1_500,
            ..FrameStats::default()
        };
        assert!(!degraded.preview_within_budget());
        assert!(degraded.preview_coherence().is_none());
    }
}
