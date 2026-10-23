//! GPU half of the RAT-48 culling profile: a degenerate render pipeline
//! that issues the culler's `draw_indexed_indirect`, so one frame can
//! bracket both GPU passes — the culling compute pass and the indirect
//! draw — with pass-boundary timestamps.
//!
//! The probe renders nothing: every vertex collapses to one clip-space
//! point, so the indirect-draw pass measures the pass plus the indirect
//! dispatch, not shading. `Renderer3D`'s gbuffer pass is private to the
//! renderer, so this synthetic probe is the stand-in the profile owns;
//! its cost is named as such in the report. Without
//! `wgpu::Features::TIMESTAMP_QUERY` the query set is `None`, the
//! readback yields no stamps, and the caller prints the GPU stages as
//! unmeasured — the bench still builds and runs.
use std::borrow::Cow;
use std::time::Duration;

use ornis_render::Culler;
use wgpu::util::DeviceExt;

/// Degenerate probe WGSL: one clip-space point, constant colour.
const PROBE_WGSL: &str = r#"
@vertex fn vs() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.5, 0.5, 0.5, 1.0);
}
"#;
/// The probe target size (headless, never presented); the profile's
/// renderer surface reuses it, so the headless frame is square once.
pub(super) const PROBE_WIDTH: u32 = 64;
pub(super) const PROBE_HEIGHT: u32 = 64;
/// Query-set slots: cull-pass begin/end, indirect-draw begin/end.
const QUERY_COMPUTE_BEGIN: u32 = 0;
const QUERY_COMPUTE_END: u32 = 1;
const QUERY_DRAW_BEGIN: u32 = 2;
const QUERY_DRAW_END: u32 = 3;
/// Slots per query set — the open end of the resolve range.
const QUERY_SLOTS: u32 = QUERY_DRAW_END + 1;
/// Bytes of one resolved `u64` timestamp word.
const TIMESTAMP_WORD_BYTES: usize = std::mem::size_of::<u64>();
/// The same width in the `u64` units `read_bytes_back` takes.
const TIMESTAMP_BYTES: u64 = TIMESTAMP_WORD_BYTES as u64;

/// The indirect-draw probe: pipeline, degenerate target, index buffer,
/// the resolve buffer and — when the adapter has `TIMESTAMP_QUERY` —
/// the query set that brackets the cull and draw passes.
pub(super) struct Probe {
    pipeline: wgpu::RenderPipeline,
    index_buffer: wgpu::Buffer,
    target: wgpu::TextureView,
    queries: Option<wgpu::QuerySet>,
    resolve: wgpu::Buffer,
    period_ns: f32,
}

impl Probe {
    // qual:allow(iosp) reason: "probe construction — the timestamp feature gate and GPU object setup; Self::create_pipeline and the feature probe are local wiring, not a layer crossing"
    /// Build the probe on `device`; `queue` only supplies the timestamp
    /// period. The query set exists exactly when the device carries
    /// [`wgpu::Features::TIMESTAMP_QUERY`].
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let target = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("culling profile probe target"),
                size: wgpu::Extent3d {
                    width: PROBE_WIDTH,
                    height: PROBE_HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let pipeline = Self::create_pipeline(device);
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("culling profile probe indices"),
            contents: bytemuck::cast_slice(&vec![0_u32; super::INDEX_COUNT as usize]),
            usage: wgpu::BufferUsages::INDEX,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("culling profile timestamp resolve"),
            size: u64::from(QUERY_SLOTS) * TIMESTAMP_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let queries = if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("culling profile timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERY_SLOTS,
            }))
        } else {
            None
        };
        let period_ns = queue.get_timestamp_period();
        Self {
            pipeline,
            index_buffer,
            target,
            queries,
            resolve,
            period_ns,
        }
    }

    /// The probe's fixed-function pipeline: the WGSL module, a
    /// pipeline-less (`None`) render pipeline for the degenerate draw.
    fn create_pipeline(device: &wgpu::Device) -> wgpu::RenderPipeline {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("culling profile probe shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PROBE_WGSL)),
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("culling profile probe pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }

    // qual:allow(test_quality, untested) reason: "bench-only GPU frame driver, driven by the criterion harness, never under libtest"
    /// One profiled GPU frame: the cull pass and the indirect-draw pass
    /// in one encoder, the query resolve after them, one submit, the
    /// blocking poll every measured frame needs, then the stamp readback.
    /// `(None, None)` without timestamps — the caller marks the stage
    /// unmeasured instead of guessing.
    pub(super) fn timed_passes(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        culler: &Culler,
    ) -> (Option<Duration>, Option<Duration>) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("culling profile frame"),
        });
        self.record_cull(&mut encoder, culler);
        self.record_draw(&mut encoder, culler);
        if let Some(queries) = &self.queries {
            encoder.resolve_query_set(queries, 0..QUERY_SLOTS, &self.resolve, 0);
        }
        queue.submit([encoder.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        self.read_stamps(device, queue)
    }

    // qual:allow(test_quality, untested) reason: "bench-only pass recorder, driven by the criterion harness, never under libtest"
    /// The production compaction pass into this encoder, timestamped at
    /// the pass boundary when the query set exists.
    fn record_cull(&self, encoder: &mut wgpu::CommandEncoder, culler: &Culler) {
        match &self.queries {
            Some(queries) => culler.record_compact_timed(
                encoder,
                wgpu::ComputePassTimestampWrites {
                    query_set: queries,
                    beginning_of_pass_write_index: Some(QUERY_COMPUTE_BEGIN),
                    end_of_pass_write_index: Some(QUERY_COMPUTE_END),
                },
            ),
            None => culler.record_compact(encoder),
        }
    }

    // qual:allow(test_quality, untested) reason: "bench-only pass recorder, driven by the criterion harness, never under libtest"
    /// The indirect draw the probe owns: one degenerate indexed draw
    /// straight from the culler's args, timed at the pass boundary.
    fn record_draw(&self, encoder: &mut wgpu::CommandEncoder, culler: &Culler) {
        let timestamp_writes =
            self.queries
                .as_ref()
                .map(|queries| wgpu::RenderPassTimestampWrites {
                    query_set: queries,
                    beginning_of_pass_write_index: Some(QUERY_DRAW_BEGIN),
                    end_of_pass_write_index: Some(QUERY_DRAW_END),
                });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("culling profile indirect draw pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(&culler.args_buffer, 0);
    }

    // qual:allow(test_quality, untested) reason: "bench-only stamp readback, driven by the criterion harness, never under libtest"
    /// Both pass durations from the resolve buffer: `(None, None)`
    /// without a query set or a failed readback — never a guess.
    fn read_stamps(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (Option<Duration>, Option<Duration>) {
        if self.queries.is_none() {
            return (None, None);
        }
        let bytes = u64::from(QUERY_SLOTS) * TIMESTAMP_BYTES;
        let stamps = match ornis_render::read_bytes_back(device, queue, &self.resolve, bytes) {
            Some(raw) => raw
                .chunks_exact(TIMESTAMP_WORD_BYTES)
                .filter_map(|word| word.try_into().ok().map(u64::from_le_bytes))
                .collect::<Vec<u64>>(),
            None => return (None, None),
        };
        (
            self.duration(&stamps, QUERY_COMPUTE_BEGIN, QUERY_COMPUTE_END),
            self.duration(&stamps, QUERY_DRAW_BEGIN, QUERY_DRAW_END),
        )
    }

    /// One pass duration: `(end - begin)` ticks scaled by the queue's
    /// timestamp period (`None` when the pair is missing).
    fn duration(&self, stamps: &[u64], begin: u32, end: u32) -> Option<Duration> {
        match (stamps.get(begin as usize), stamps.get(end as usize)) {
            (Some(&begin), Some(&end)) => {
                let ticks = end.saturating_sub(begin) as f64;
                Some(Duration::from_nanos(
                    (ticks * f64::from(self.period_ns)) as u64,
                ))
            }
            _ => None,
        }
    }
}
