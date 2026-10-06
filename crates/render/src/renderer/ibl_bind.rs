//! Split-sum IBL binding: environment upload, explicit/automatic weight resolution, and rebinding of the forward and legacy PBR bind groups.
use super::forward::{IblTail, forward_ibl_entries};
use super::*;
impl Renderer3D {
    /// Bind a split-sum environment, or clear it (`None` restores the
    /// 1×1 black cubes and a weight of 0 — the direct-light result).
    ///
    /// Forward bind groups are rebuilt so they sample the new views.
    /// Deferred lighting rebuilds its group every frame.
    pub fn set_image_based_light(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        env: Option<&crate::ibl::EnvironmentCube>,
    ) {
        self.ibl = crate::ibl::upload_targets(device, queue, env);
        self.ibl_uploaded
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.store_ibl_params(queue);
        self.rebind_ibl_groups(device);
    }

    /// Automatic cube weight, unless an explicit weight was stored.
    ///
    /// No explicit weight: `0` with the black placeholder, `1` after a
    /// cube upload. An explicit weight stays across both binds and is
    /// what [`Self::set_lights`] copies into `LightingUniform.ibl_weight`.
    fn resolve_ibl_weight(explicit: bool, explicit_weight: f32, automatic: f32) -> f32 {
        if explicit { explicit_weight } else { automatic }
    }

    /// Writes `weight` into `LightingUniform.ibl_weight` only.
    ///
    /// Does not upload or clear the environment cube, and does not change
    /// `ibl_max_mip`. Later [`Self::set_image_based_light`] and
    /// [`Self::set_lights`] keep this value.
    pub(crate) fn set_explicit_environment_weight(&self, queue: &wgpu::Queue, weight: f32) {
        self.ibl_weight_explicit
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.ibl_weight_bits
            .store(weight.to_bits(), std::sync::atomic::Ordering::Relaxed);
        queue.write_buffer(
            &self.lighting_buffer,
            std::mem::offset_of!(LightingUniform, ibl_weight) as u64,
            bytemuck::bytes_of(&weight),
        );
    }

    /// `LightingUniform.ibl_weight` last stored in the CPU mirror.
    #[cfg(test)]
    pub(crate) fn ibl_weight_for_tests(&self) -> f32 {
        f32::from_bits(
            self.ibl_weight_bits
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// `LightingUniform.ibl_max_mip` last stored in the CPU mirror.
    #[cfg(test)]
    pub(crate) fn ibl_max_mip_for_tests(&self) -> f32 {
        f32::from_bits(
            self.ibl_max_mip_bits
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    fn store_ibl_params(&self, queue: &wgpu::Queue) {
        let weight = Self::resolve_ibl_weight(
            self.ibl_weight_explicit
                .load(std::sync::atomic::Ordering::Relaxed),
            f32::from_bits(
                self.ibl_weight_bits
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            self.ibl.weight,
        );
        self.ibl_weight_bits
            .store(weight.to_bits(), std::sync::atomic::Ordering::Relaxed);
        self.ibl_max_mip_bits.store(
            self.ibl.max_mip.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        queue.write_buffer(
            &self.lighting_buffer,
            std::mem::offset_of!(LightingUniform, ibl_weight) as u64,
            bytemuck::bytes_of(&IblTail {
                weight,
                max_mip: self.ibl.max_mip,
            }),
        );
    }

    pub(super) fn encode_ibl_if_needed(&self, encoder: &mut wgpu::CommandEncoder) {
        match self
            .ibl_uploaded
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            true => {}
            false => crate::ibl::encode_initial_upload(encoder, &self.ibl_staging, &self.ibl),
        }
    }

    fn rebind_ibl_groups(&mut self, device: &wgpu::Device) {
        self.rebind_forward_ibl(device);
        self.rebind_legacy_pbr_ibl(device);
    }

    fn rebind_forward_ibl(&mut self, device: &wgpu::Device) {
        let per_object = read_lock(&self.per_object_buffer);
        let material = read_lock(&self.material_buffer);
        *write_lock(&self.forward_pass.bind_group) =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("forward bind group (ibl)"),
                layout: &self.forward_pass.bind_group_layout,
                entries: &forward_ibl_entries(
                    &self.camera_buffer,
                    &per_object,
                    &material,
                    &self.lighting_buffer,
                    &self.shadow_array_view,
                    &self.shadow_sampler,
                    &self.shadow_cube_array_view,
                    &self.ibl,
                ),
            });
    }

    fn rebind_legacy_pbr_ibl(&mut self, device: &wgpu::Device) {
        let per_object = read_lock(&self.per_object_buffer);
        let material = read_lock(&self.material_buffer);
        self._bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pbr bind group (ibl)"),
            layout: &self._bind_group_layout,
            entries: &forward_ibl_entries(
                &self.camera_buffer,
                &per_object,
                &material,
                &self.lighting_buffer,
                &self.shadow_array_view,
                &self.shadow_sampler,
                &self.shadow_cube_array_view,
                &self.ibl,
            ),
        });
    }
}
