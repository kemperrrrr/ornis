//! Anti-aliasing passes: the temporal resolve (TAA) and the post-process
//! edge filters (FXAA/SMAA), plus the renderer state the temporal mode
//! needs (the jitter sequence and the camera matrices of the last two
//! frames).
//!
//! The temporal mode jitters the projection every frame and resolves the
//! frame against a history buffer: the current sample is reprojected into
//! the previous frame through the depth buffer, the history is clamped to
//! the colour range of the current neighbourhood and the two are blended.
//! The edge filters are single passes over the composited image.
//!
//! The pipelines are built per sample count (the temporal resolve binds
//! the multisampled depth at 4x), so [`Renderer3D::resize`] leaves them
//! alone — only the history textures follow the surface size.

mod build;
mod state;

use super::{FULLSCREEN_QUAD_VERTS, Renderer3D, WgslStruct};
use crate::antialias::{AntiAliasMode, EdgeFilterSettings, TaaSettings};
use crate::shaders;
use glam::Mat4;
pub(crate) use state::AaState;

/// Format of the temporal history buffer: a linear HDR target, half the
/// memory of an RGBA32 one and wide enough for the composited image.
pub(crate) const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Per-frame parameters of the anti-aliasing passes.
///
/// The temporal resolve reads the g-buffer motion vector and previous
/// depth of the surface under the current pixel, plus the history-valid
/// flag; the edge filters read the luma threshold and their mode selector.
///
/// The WGSL `AaParams` declaration is generated from this layout
/// ([`AaUniform::WGSL_SOURCE`]); the field list here is the single source
/// of truth for the buffer layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "AaParams")]
pub(crate) struct AaUniform {
    /// Weight of the history in the temporal blend.
    feedback: f32,
    /// Luma range below which a pixel is not filtered.
    threshold: f32,
    /// Radius of the temporal neighbourhood clamp, in pixels.
    clamp_radius: f32,
    /// `1.0` when a valid history buffer exists.
    history_valid: f32,
    /// Which edge filter the post-process pass runs: 0 = FXAA, 1 = SMAA.
    mode: u32,
    /// Trailing pad (padding: not shader-visible).
    #[wgsl(skip)]
    _pad: [f32; 3],
}

impl AaUniform {
    /// Packs the temporal resolve settings. `history_valid` says whether
    /// the history texture holds a resolved frame.
    pub(crate) fn pack_taa(settings: TaaSettings, history_valid: bool) -> Self {
        Self {
            feedback: settings.feedback(),
            threshold: EdgeFilterSettings::default().threshold(),
            clamp_radius: settings.clamp_radius() as f32,
            history_valid: if history_valid { 1.0 } else { 0.0 },
            mode: 0,
            _pad: [0.0; 3],
        }
    }

    /// Packs the parameters of the post-process edge filter.
    pub(crate) fn pack_edge(settings: EdgeFilterSettings, mode: AntiAliasMode) -> Self {
        Self {
            feedback: 0.0,
            threshold: settings.threshold(),
            clamp_radius: 0.0,
            history_valid: 0.0,
            mode: mode.shader_mode(),
            _pad: [0.0; 3],
        }
    }
}

/// Inputs of the temporal resolve: the composited image, the g-buffer
/// depth and motion vector the reprojection reads, and the presented target.
pub struct TaaInputs<'a> {
    /// The composited image the resolve reads.
    pub source: &'a wgpu::TextureView,
    /// The g-buffer depth buffer: background pixels do not reuse history.
    pub depth: &'a wgpu::TextureView,
    /// The g-buffer motion vector (current UV minus previous UV, plus the
    /// previous NDC depth in B) the reprojection follows.
    pub motion: &'a wgpu::TextureView,
    /// The presented target the resolve writes.
    pub target: &'a wgpu::TextureView,
    /// The settings of the resolve (feedback, clamp radius).
    pub settings: TaaSettings,
}

/// Inputs of the post-process edge filter: the composited image and the
/// presented target.
pub struct EdgeInputs<'a> {
    /// The composited image the filter reads.
    pub source: &'a wgpu::TextureView,
    /// The presented target the filter writes.
    pub target: &'a wgpu::TextureView,
    /// The settings of the filter (the luma threshold).
    pub settings: EdgeFilterSettings,
    /// Which filter to run (FXAA or SMAA).
    pub mode: AntiAliasMode,
}

impl Renderer3D {
    /// The view-projection a frame renders with, and the temporal state
    /// the resolve needs.
    ///
    /// Only the temporal mode jitters: the projection shifts by a
    /// sub-pixel amount every frame, so the same world point lands on a
    /// different sub-pixel position and the resolve accumulates the
    /// samples. Every other mode renders with the untouched matrix, so the
    /// frame stays pixel-identical.
    pub(super) fn aa_frame_camera(&self, base: Mat4) -> Mat4 {
        if !self.aa_state.mode().is_temporal() {
            return base;
        }
        // The first TAA frame has no accumulated sample to resolve. Render
        // it unjittered, but record its matrix so the next frame reprojects
        // into the camera that actually produced this frame's depth.
        if !self.aa_state.temporal_armed() {
            self.aa_state.record_camera_frame(base);
            return base;
        }
        let offset = self.aa_state.advance_jitter();
        let view_proj =
            crate::antialias::jittered_projection(base, offset, (self.width, self.height));
        self.aa_state.record_camera_frame(view_proj);
        view_proj
    }

    /// Select the anti-aliasing mode of the following frames.
    ///
    /// The temporal mode jitters the projection (see
    /// [`set_camera`](Self::set_camera)) and expects the history resolve
    /// pass in the frame plan; the post-process modes filter the composited
    /// image in one pass. Switching the mode resets the temporal state, so
    /// a stale history never leaks into the new mode.
    pub fn set_anti_alias(&self, mode: AntiAliasMode) {
        self.aa_state.set_mode(mode);
    }

    /// The anti-aliasing mode the renderer currently runs with.
    pub fn anti_alias_mode(&self) -> AntiAliasMode {
        self.aa_state.mode()
    }

    /// Record the temporal resolve: reads the composited image, resolves
    /// it against the history buffer and writes both the presented image
    /// and the next history.
    pub fn render_taa(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        inputs: TaaInputs<'_>,
    ) {
        let TaaInputs {
            source,
            depth,
            motion,
            target,
            settings,
        } = inputs;
        self.aa_passes
            .ensure_history(device, self.width, self.height);
        let history_guard = super::read_lock(&self.aa_passes.history);
        let history = history_guard
            .as_ref()
            .expect("TAA history is allocated before its resolve pass");
        let (read_index, write_index) = self.aa_passes.history_slots();
        let uniform = AaUniform::pack_taa(settings, self.aa_state.history_valid());
        queue.write_buffer(
            &self.aa_passes.params_buffer,
            0,
            bytemuck::bytes_of(&uniform),
        );
        let history_read = &history[read_index].view;
        let history_write = &history[write_index].view;
        let bind_group = self.aa_taa_bind_group(device, source, depth, motion, history_read);
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("taa pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: history_write,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&self.aa_passes.taa);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..FULLSCREEN_QUAD_VERTS, 0..1);
        drop(rpass);
        drop(history_guard);
        self.aa_passes.advance_history(write_index);
        self.aa_state.mark_history_valid();
        // The resolve runs from now on, so the projection may jitter.
        self.aa_state.arm_temporal();
    }

    /// The bind group of the temporal resolve: the composited image, the
    /// depth it reprojects through, the history it clamps and the
    /// per-frame parameters.
    fn aa_taa_bind_group(
        &self,
        device: &wgpu::Device,
        source: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        motion: &wgpu::TextureView,
        history: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taa bind group"),
            layout: &self.aa_passes.taa_layout,
            entries: &shaders::bind_group_entries(&shaders::aa_taa_generated::TAA_RESOURCES, |r| {
                match r.name {
                    "src_tex" => Some(wgpu::BindingResource::TextureView(source)),
                    "src_sampler" => Some(wgpu::BindingResource::Sampler(&self.aa_passes.sampler)),
                    "depth_tex" => Some(wgpu::BindingResource::TextureView(depth)),
                    "motion_tex" => Some(wgpu::BindingResource::TextureView(motion)),
                    "history_tex" => Some(wgpu::BindingResource::TextureView(history)),
                    "aa_params" => Some(wgpu::BindingResource::Buffer(
                        self.aa_passes.params_buffer.as_entire_buffer_binding(),
                    )),
                    _ => None,
                }
            })
            .unwrap_or_default(),
        })
    }

    /// Record the post-process edge filter (FXAA or SMAA) over the
    /// composited image.
    pub fn render_edge_filter(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        inputs: EdgeInputs<'_>,
    ) {
        let EdgeInputs {
            source,
            target,
            settings,
            mode,
        } = inputs;
        let uniform = AaUniform::pack_edge(settings, mode);
        queue.write_buffer(
            &self.aa_passes.params_buffer,
            0,
            bytemuck::bytes_of(&uniform),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aa edge bind group"),
            layout: &self.aa_passes.edge_layout,
            entries: &shaders::bind_group_entries(
                &shaders::aa_edge_generated::EDGE_RESOURCES,
                |r| match r.name {
                    "src_tex" => Some(wgpu::BindingResource::TextureView(source)),
                    "src_sampler" => Some(wgpu::BindingResource::Sampler(&self.aa_passes.sampler)),
                    "aa_params" => Some(wgpu::BindingResource::Buffer(
                        self.aa_passes.params_buffer.as_entire_buffer_binding(),
                    )),
                    _ => None,
                },
            )
            .unwrap_or_default(),
        });
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("aa edge pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&self.aa_passes.edge);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..FULLSCREEN_QUAD_VERTS, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aa_uniform_layout_matches_wgsl() {
        // Four floats, the mode selector and the trailing pad: 32 bytes.
        assert_eq!(std::mem::size_of::<AaUniform>(), 32);
        assert_eq!(std::mem::offset_of!(AaUniform, feedback), 0);
        assert_eq!(std::mem::offset_of!(AaUniform, threshold), 4);
        assert_eq!(std::mem::offset_of!(AaUniform, clamp_radius), 8);
        assert_eq!(std::mem::offset_of!(AaUniform, history_valid), 12);
        assert_eq!(std::mem::offset_of!(AaUniform, mode), 16);
        assert_eq!(
            AaUniform::FIELD_NAMES,
            &[
                "feedback",
                "threshold",
                "clamp_radius",
                "history_valid",
                "mode"
            ]
        );
    }
}
