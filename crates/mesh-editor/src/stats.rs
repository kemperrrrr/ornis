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
}

impl FrameStats {
    /// True when the preview exceeded its frame budget.
    pub fn preview_over_budget(&self, budget_us: u64) -> bool {
        self.preview_us > budget_us
    }
}
