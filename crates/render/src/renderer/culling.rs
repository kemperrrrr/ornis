//! Indirect-batch selection for the shared mesh batch: the opt-in
//! flag, the per-frame culler lifecycle, and the staged-batch handover
//! between the submit system and the record path (slice 2c).
use super::Culler;
use super::PerObjectGpu;
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
    /// takes the converted batch staged by
    /// [`stage_batch_inputs`](Self::stage_batch_inputs), stages it
    /// without re-converting under its `view_proj`, and arms the
    /// indirect batch when opted in via
    /// [`set_indirect_draw`](Self::set_indirect_draw). A no-op resetting
    /// to direct when opted out or when no batch was staged. When the
    /// staging reports
    /// [`CullerError::CapacityExceeded`](crate::gpu_culling::CullerError),
    /// the frame transparently falls back to the direct `draw_indexed`
    /// path ([`render_gbuffer`](Self::render_gbuffer) /
    /// [`render_shadows`](Self::render_shadows) — shadows always draw
    /// directly); the next frame retries. That branch is defensive:
    /// `ensure_culler_capacity` runs just before the compact with the
    /// same record count and grows the culler to fit (its `try_new`
    /// fails only on zero capacity, which the doubling rules out), so a
    /// staged frame never lands there in practice.
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
        let Some((view_proj, records)) = staged else {
            self.indirect_ready
                .store(false, std::sync::atomic::Ordering::Relaxed);
            return;
        };
        self.ensure_culler_capacity(device, records.len());
        let ready = read_lock(&self.culler).as_ref().is_some_and(|culler| {
            culler
                .compact_records(queue, &view_proj, index_count, &records)
                .is_ok()
        });
        self.indirect_ready
            .store(ready, std::sync::atomic::Ordering::Relaxed);
    }

    /// Stages the shared batch for a possible indirect submit: moves
    /// (`view_proj`, pre-converted `records`) in for the record path to
    /// take in [`prepare_indirect_batch`](Self::prepare_indirect_batch).
    /// Unconditional — without the opt-in flag the staged batch is
    /// simply never consumed.
    pub(crate) fn stage_batch_inputs(&self, view_proj: Mat4, records: Vec<PerObjectGpu>) {
        *write_lock(&self.staged_batch) = Some((view_proj, records));
    }

    /// Whether the last [`prepare_indirect_batch`](Self::prepare_indirect_batch)
    /// left an indirect batch ready to submit (read by the gbuffer node).
    pub(crate) fn indirect_batch_ready(&self) -> bool {
        self.indirect_ready
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_util;
    use super::super::{BatchDraw, select_shared_batch};
    use super::*;
    use crate::gpu_culling::CullerError;
    use crate::renderer::InstanceData;

    /// Minimal surface config for renderer construction in these tests
    /// (no frame is ever presented through it).
    fn test_surface_config() -> wgpu::SurfaceConfiguration {
        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            width: 64,
            height: 64,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        }
    }

    /// Over-capacity frames fall back to the direct path:
    /// [`Culler::compact`](crate::gpu_culling::Culler::compact) reports
    /// [`CullerError::CapacityExceeded`], and the frame selection maps
    /// every non-armed combination (opt-out, missed prepare, exhausted
    /// capacity) to `draw_indexed` — never to a partial indirect draw.
    #[test]
    fn capacity_exceeded_falls_back_to_direct() {
        let Some((device, queue)) = test_util::try_device() else {
            eprintln!("SKIPPED (capacity_exceeded_falls_back_to_direct): no wgpu adapter");
            return;
        };
        let culler = Culler::try_new(&device, 2).expect("non-zero capacity");
        let instances = (0..3)
            .map(|_| InstanceData {
                model_matrix: glam::Mat4::IDENTITY,
                normal_matrix: glam::Mat4::IDENTITY,
                material_index: crate::renderer::MaterialIdx::from_raw(0),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            culler.compact(&queue, &glam::Mat4::IDENTITY, 42, &instances),
            Err(CullerError::CapacityExceeded {
                count: 3,
                capacity: 2,
            })
        );
        assert!(matches!(
            select_shared_batch(true, None, 3),
            BatchDraw::Direct { .. }
        ));
        assert!(matches!(
            select_shared_batch(false, Some(&culler), 3),
            BatchDraw::Direct { .. }
        ));
        let renderer = Renderer3D::new(&device, &test_surface_config(), 1);
        renderer.prepare_indirect_batch(&device, &queue, 42);
        assert!(
            !renderer.indirect_batch_ready(),
            "an unarmed frame must stay on the direct fallback"
        );
    }
}
