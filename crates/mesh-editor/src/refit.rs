//! Frame-budget gates for collider refit and the exact/preview swap.
//!
//! [`RefitBudget`] is the single time+size gate in front of every collider
//! rebuild: [`RefitBudget::decide`] runs before any allocation (mesh size
//! only), [`RefitBudget::decide_with_elapsed`] folds in observed wall time
//! afterwards. Outcomes are [`RefitDecision`] (never a bool), and the swap
//! of a background exact result over the live preview goes through
//! [`decide_swap`] → [`SwapDecision`], keyed by [`Seq`] newest-wins
//! ordering plus a [`PreviewStats`](crate::PreviewStats) coherence witness.

use crate::{PreviewStats, Seq};

/// Wall-time duration in whole milliseconds (frame-budget unit).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Millis(u64);

impl Millis {
    /// Zero duration.
    pub const ZERO: Self = Self(0);

    /// Wraps a raw millisecond count.
    pub const fn new(ms: u64) -> Self {
        Self(ms)
    }

    /// Raw millisecond count.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for Millis {
    /// Wraps a raw millisecond count.
    fn from(ms: u64) -> Self {
        Self(ms)
    }
}

impl From<Millis> for u64 {
    /// Unwraps back to the raw millisecond count.
    fn from(ms: Millis) -> Self {
        ms.0
    }
}

/// Gate outcome for a collider refit (never a bool).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefitDecision {
    /// Within budget: rebuild the collider synchronously.
    RefitNow,
    /// Over budget: keep the old collider, retry later (background pass or
    /// next commit). The mesh edit itself is unaffected.
    Defer,
}

/// Time + size budget in front of every collider rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefitBudget {
    /// Largest single refit slice the frame absorbs.
    pub max_ms: Millis,
    /// Largest mesh (triangles) rebuilt synchronously.
    pub max_tris: u32,
}

impl RefitBudget {
    /// Default gate: 2 ms / 65 536 triangles. Covers a mid-size
    /// `TriMesh` + BVH rebuild inside one commit hitch; bigger meshes defer
    /// to a background pass and the caller retries.
    pub const DEFAULT: Self = Self {
        max_ms: Millis::new(2),
        max_tris: 65_536,
    };

    /// Budget with an explicit time slice and triangle ceiling.
    pub const fn new(max_ms: Millis, max_tris: u32) -> Self {
        Self { max_ms, max_tris }
    }

    /// Pre-build gate on mesh size only: runs before any allocation, so an
    /// over-budget mesh never hitches the frame building arrays it cannot
    /// use.
    pub fn decide(&self, triangle_count: usize) -> RefitDecision {
        if u64::try_from(triangle_count).unwrap_or(u64::MAX) > u64::from(self.max_tris) {
            RefitDecision::Defer
        } else {
            RefitDecision::RefitNow
        }
    }

    /// Post-measure gate: size plus observed (or estimated) wall time.
    /// Either axis over budget defers.
    pub fn decide_with_elapsed(&self, triangle_count: usize, elapsed: Millis) -> RefitDecision {
        if elapsed > self.max_ms {
            RefitDecision::Defer
        } else {
            self.decide(triangle_count)
        }
    }
}

impl Default for RefitBudget {
    /// [`RefitBudget::DEFAULT`]: 2 ms / 65 536 triangles.
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Denial from [`to_physics_arrays_gated`](crate::to_physics_arrays_gated):
/// the mesh exceeded the refit budget, nothing was allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("collider refit deferred: {tris} triangles exceed budget of {max_tris}")]
pub struct RefitDefer {
    /// Triangles of the mesh that was NOT converted.
    pub tris: usize,
    /// Budget ceiling that rejected it.
    pub max_tris: u32,
}

impl RefitDefer {
    /// Gate outcome matching this denial (always [`RefitDecision::Defer`]).
    pub fn decision(self) -> RefitDecision {
        RefitDecision::Defer
    }
}

/// Swap outcome for a finished exact result (never a bool).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapDecision {
    /// Keep showing the preview: the exact result is stale, or the preview
    /// is degraded and must not pop under the user.
    KeepPreview,
    /// The exact result is newer than the preview: adopt it as the base.
    SwapExact(Seq),
}

/// Newest-wins swap criterion: adopt the exact mesh only when its `seq`
/// covers the current preview and the preview carries a coherence witness.
///
/// Equality (`exact_seq == preview_seq`) is the fresh case, not a tie: the
/// job was computed from exactly the displayed snapshot. Less-than is
/// stale — a newer preview edit superseded the job, so swapping would
/// regress it. `None` coherence (degraded preview: decimated, proxy or
/// frozen ladder level) pins the swap even for a fresh result.
pub fn decide_swap(
    preview_seq: Seq,
    coherence: Option<PreviewStats>,
    exact_seq: Seq,
) -> SwapDecision {
    if coherence.is_some() && exact_seq >= preview_seq {
        SwapDecision::SwapExact(exact_seq)
    } else {
        SwapDecision::KeepPreview
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn millis_wraps_counts() {
        assert_eq!(Millis::ZERO.get(), 0);
        assert_eq!(Millis::new(2).get(), 2);
        assert_eq!(u64::from(Millis::new(2)), 2);
        assert!(Millis::new(1) < Millis::new(2));
    }

    #[test]
    fn budget_decides_on_tris_and_elapsed() {
        let budget = RefitBudget::new(Millis::new(2), 11);
        assert_eq!(budget.decide(11), RefitDecision::RefitNow);
        assert_eq!(budget.decide(12), RefitDecision::Defer);
        assert_eq!(
            budget.decide_with_elapsed(4, Millis::new(3)),
            RefitDecision::Defer
        );
        assert_eq!(
            budget.decide_with_elapsed(4, Millis::new(2)),
            RefitDecision::RefitNow
        );
        assert_eq!(
            budget.decide_with_elapsed(12, Millis::new(1)),
            RefitDecision::Defer
        );
        assert_eq!(RefitBudget::default(), RefitBudget::DEFAULT);
        let denial = RefitDefer {
            tris: 12,
            max_tris: 11,
        };
        assert_eq!(denial.decision(), RefitDecision::Defer);
    }

    #[test]
    fn swap_keeps_stale_or_incoherent_exact() {
        let coherent = PreviewStats::try_new(0.5);
        assert_eq!(
            decide_swap(Seq::new(5), coherent, Seq::new(5)),
            SwapDecision::SwapExact(Seq::new(5)),
            "equality is fresh: computed from the displayed snapshot"
        );
        assert_eq!(
            decide_swap(Seq::new(5), coherent, Seq::new(6)),
            SwapDecision::SwapExact(Seq::new(6))
        );
        assert_eq!(
            decide_swap(Seq::new(6), coherent, Seq::new(5)),
            SwapDecision::KeepPreview,
            "stale exact never regresses a newer preview"
        );
        assert_eq!(
            decide_swap(Seq::new(5), None, Seq::new(6)),
            SwapDecision::KeepPreview,
            "degraded preview pins even a fresh result"
        );
        assert_eq!(
            decide_swap(Seq::new(5), None, Seq::new(5)),
            SwapDecision::KeepPreview
        );
    }
}
