//! Indirect-batch selection for the shared mesh batch: the opt-in
//! flag, the per-frame culler lifecycle, and the staged-batch handover
//! between the submit system and the record path (slice 2c).
use super::Culler;
use super::InstanceData;
use super::Renderer3D;
use super::read_lock;
use super::write_lock;
use glam::Mat4;

impl Renderer3D {
    /// Opt-in switch for indirect submission of the shared mesh batch:
    /// when set, `prepare_indirect_batch` compacts the frame and the
    /// gbuffer node submits it with one `draw_indexed_indirect` instead
    /// of `draw_indexed`. Default off (direct path — every existing
    /// frame is unchanged). Set from
    /// [`crate::render_backend::RenderBackendConfig::indirect_draw`].
    pub fn set_indirect_draw(&self, enabled: bool) {
        self.indirect_draw
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether indirect submission is opted in (read by the submit
    /// system to skip staging without the flag).
    pub fn indirect_draw(&self) -> bool {
        self.indirect_draw
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Grows the culler to `needed` instances (doubling, like the
    /// per-object buffer), creating it on first use. A re-created
    /// culler rebuilds its pipeline and bind groups — amortized over
    /// frames, like every other growable store here.
    fn ensure_culler_capacity(&self, device: &wgpu::Device, needed: usize) {
        let fits = read_lock(&self.culler)
            .as_ref()
            .is_some_and(|culler| culler.capacity as usize >= needed);
        if fits {
            return;
        }
        let current = read_lock(&self.culler)
            .as_ref()
            .map_or(1, |culler| culler.capacity.max(1));
        let mut grown = current;
        while (grown as usize) < needed {
            grown = grown.saturating_mul(2);
        }
        // `grown` is at least one by construction, so `try_new` cannot
        // fail; on the impossible error the old culler (if any) stays
        // and the frame falls back to direct in `prepare_indirect_batch`.
        // A new culler owns new buffers, so the cached indirect bind
        // group (bound over the old `visible_buffer`) is dropped and
        // rebuilt on the next indirect draw.
        if let Ok(culler) = Culler::try_new(device, grown) {
            *write_lock(&self.culler) = Some(culler);
            *write_lock(&self.indirect_bind_group) = None;
        }
    }

    /// Runs the culling dispatch for this frame (slice 2c selection):
    /// takes the batch staged by [`stage_batch_inputs`](Self::stage_batch_inputs),
    /// compacts its instances under its `view_proj`, and arms the
    /// indirect batch when opted in via
    /// [`set_indirect_draw`](Self::set_indirect_draw). A no-op resetting
    /// to direct when opted out or when no batch was staged. When the
    /// compact reports
    /// [`CullerError::CapacityExceeded`](crate::gpu_culling::CullerError),
    /// the frame transparently falls back to the direct `draw_indexed`
    /// path ([`render_gbuffer`](Self::render_gbuffer) /
    /// [`render_shadows`](Self::render_shadows) — shadows always draw
    /// directly); the next frame retries.
    pub(crate) fn prepare_indirect_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        index_count: u32,
    ) {
        if !self
            .indirect_draw
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            self.indirect_ready
                .store(false, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        let staged = write_lock(&self.staged_batch).take();
        let Some((view_proj, instances)) = staged else {
            self.indirect_ready
                .store(false, std::sync::atomic::Ordering::Relaxed);
            return;
        };
        self.ensure_culler_capacity(device, instances.len());
        let ready = read_lock(&self.culler).as_ref().is_some_and(|culler| {
            culler
                .compact(device, queue, &view_proj, index_count, &instances)
                .is_ok()
        });
        self.indirect_ready
            .store(ready, std::sync::atomic::Ordering::Relaxed);
    }

    /// Stages the shared batch for a possible indirect submit: moves
    /// (`view_proj`, `instances`) in for the record path to take in
    /// [`prepare_indirect_batch`](Self::prepare_indirect_batch).
    /// Unconditional — without the opt-in flag the staged batch is
    /// simply never consumed.
    pub(crate) fn stage_batch_inputs(&self, view_proj: Mat4, instances: Vec<InstanceData>) {
        *write_lock(&self.staged_batch) = Some((view_proj, instances));
    }

    /// Whether the last [`prepare_indirect_batch`](Self::prepare_indirect_batch)
    /// left an indirect batch ready to submit (read by the gbuffer node).
    pub(crate) fn indirect_batch_ready(&self) -> bool {
        self.indirect_ready
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}
