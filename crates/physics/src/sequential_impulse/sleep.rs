//! Island sleep queries: sleep state and island timers.

use crate::body::BodyHandle;

use super::SequentialImpulseEngine;

impl SequentialImpulseEngine {
    /// Whether the body is currently sleeping (G4/G7 diagnostics). Static
    /// bodies report true from birth — they never move, which is exactly
    /// what the frozen-pair skips in the narrow phase rely on.
    pub fn is_asleep(&self, handle: BodyHandle) -> bool {
        self.asleep.get(handle).copied().unwrap_or(false)
    }

    /// (Diagnostics) island id of the body and its current sleep timer.
    pub fn debug_island_info(&self, handle: BodyHandle) -> Option<(u32, f32)> {
        let root = *self.island.get(handle)?;
        let timer = self.island_timers.get(&root).copied().unwrap_or(0.0);
        Some((root, timer))
    }
}
