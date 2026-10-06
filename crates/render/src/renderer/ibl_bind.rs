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

#[cfg(test)]
mod tests {
    use super::super::lights::build_lighting_uniform;
    use super::super::test_util::*;
    use super::super::*;
    use ornis_assets::scene::ShadowCast;
    #[test]
    fn ibl_multipliers_scale_upload_and_default_is_noop() {
        let lights = vec![LightDesc::Directional {
            direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::new(1.0, 1.0, 1.0))
                .expect("non-zero direction"),
            intensity: 2.0,
            color: [HALF, 0.25, 0.125],
            shadow: ShadowCast::Disabled,
        }];
        let base =
            build_lighting_uniform([HALF, 0.25, 0.125], 1.0, 1.0, &lights, None, 0.0, 0.0, 0);
        assert_eq!(base.uniform.ambient_color, [HALF, 0.25, 0.125, 1.0]);
        assert_eq!(base.uniform.lights[0].color, [HALF, 0.25, 0.125, 2.0]);
        let scaled =
            build_lighting_uniform([HALF, 0.25, 0.125], 2.0, 4.0, &lights, None, 0.0, 0.0, 0);
        assert_eq!(scaled.uniform.ambient_color, [1.0, HALF, 0.25, 1.0]);
        assert_eq!(scaled.uniform.lights[0].color, [2.0, 1.0, HALF, 2.0]);
    }

    /// Explicit IBL weight reaches the lighting uniform and survives a
    /// cube bind. Without the setter, no cube stays at 0 and a cube
    /// becomes 1. Skipped when no adapter is available.
    #[test]
    fn explicit_environment_weight_survives_cube_bind() {
        /// Side of the headless surface used only to construct the renderer.
        const SIDE: u32 = 4;
        /// Explicit weight, distinct from both automatic endpoints.
        const EXPLICIT_WEIGHT: f32 = 0.35;
        let Some((device, queue)) = try_device() else {
            return;
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: SIDE,
            height: SIDE,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        let cube = crate::ibl::EnvironmentCube::solid(ornis_core::units::Color::WHITE, SIDE);
        let mut automatic = Renderer3D::new(&device, &config, 1);
        assert_eq!(automatic.ibl_weight_for_tests(), 0.0);
        automatic.set_image_based_light(&device, &queue, None);
        assert_eq!(automatic.ibl_weight_for_tests(), 0.0);
        automatic.set_image_based_light(&device, &queue, Some(&cube));
        assert_eq!(automatic.ibl_weight_for_tests(), 1.0);
        let mip = automatic.ibl_max_mip_for_tests();
        assert!(mip > 0.0, "a bound cube publishes a prefilter mip");
        automatic.set_lights(&queue, [0.1, 0.1, 0.1], &[]);
        assert_eq!(automatic.ibl_weight_for_tests(), 1.0);
        assert_eq!(automatic.ibl_max_mip_for_tests(), mip);

        let mut explicit = Renderer3D::new(&device, &config, 1);
        explicit.set_explicit_environment_weight(&queue, EXPLICIT_WEIGHT);
        assert_eq!(explicit.ibl_weight_for_tests(), EXPLICIT_WEIGHT);
        explicit.set_image_based_light(&device, &queue, Some(&cube));
        assert_eq!(explicit.ibl_weight_for_tests(), EXPLICIT_WEIGHT);
        assert!(explicit.ibl_max_mip_for_tests() > 0.0);
        explicit.set_lights(&queue, [0.1, 0.1, 0.1], &[]);
        assert_eq!(explicit.ibl_weight_for_tests(), EXPLICIT_WEIGHT);
        let kept_mip = explicit.ibl_max_mip_for_tests();
        explicit.set_explicit_environment_weight(&queue, EXPLICIT_WEIGHT);
        assert_eq!(explicit.ibl_max_mip_for_tests(), kept_mip);
    }
}
