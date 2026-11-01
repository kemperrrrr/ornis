//! Storage-buffer ABI and dispatch plumbing for the three lowered kernels.

mod data;
mod pipeline;
mod resources;

use bytemuck::cast_slice;
#[cfg(test)]
use ornis_wasm2gpu_testkit::gpu_buffer;
#[cfg(test)]
use ornis_wasm2gpu_testkit::map_bytes;
use ornis_wasm2gpu_testkit::symbols;

use crate::auto_exposure::AutoExposureSettings;
use crate::auto_exposure::HISTOGRAM_BINS;

use super::Lowered;
use data::{ADAPT_PARAMS_LEN, HISTOGRAM_PARAMS_LEN, adapt_params, histogram_params, uniform_bytes};
#[cfg(test)]
use data::output_from_bytes;
use pipeline::{dispatch, pipeline};
use resources::{bind_entry, binding, storage_buffer, uniform_buffer};

const RGBA_LANES: usize = 4;
const INPUT_LANES: usize = RGBA_LANES * 2;
const EXPOSURE_SLOTS: usize = 2;
const FRAME_WORDS: usize = 2;
const INPUT_BINDING: u32 = 0;
const LOG_LUMINANCE_BINDING: u32 = 1;
const HISTOGRAM_BINDING: u32 = 2;
const FRAME_BINDING: u32 = 3;
const HISTOGRAM_PARAMS_BINDING: u32 = 4;
const ADAPT_PARAMS_BINDING: u32 = 5;
const EXPOSURE_BINDING: u32 = 6;
#[cfg(test)]
const FLOAT_BYTES: u64 = std::mem::size_of::<f32>() as u64;
#[cfg(test)]
const HISTOGRAM_SNAPSHOT_BYTES: u64 = HISTOGRAM_BINS as u64 * std::mem::size_of::<u32>() as u64;
#[cfg(test)]
const EXPOSURE_READBACK_OFFSET: u64 = HISTOGRAM_SNAPSHOT_BYTES;
#[cfg(test)]
const EXPOSURE_OUTPUT_OFFSET: u64 = FLOAT_BYTES;
#[cfg(test)]
const STAGING_BYTES: u64 = HISTOGRAM_SNAPSHOT_BYTES + FLOAT_BYTES;

/// The two HDR layers in the production pass's compositing mode.
pub(crate) struct GuestInputs<'a> {
    /// Width of both HDR layers.
    pub(crate) width: u32,
    /// Height of both HDR layers.
    pub(crate) height: u32,
    /// Composite selector: 0 deferred, 1 forward, 2 hybrid.
    pub(crate) mode: u32,
    /// Deferred RGBA layer, one texel per element.
    pub(crate) deferred: &'a [[f32; 4]],
    /// Forward RGBA layer, one texel per element.
    pub(crate) forward: &'a [[f32; 4]],
}

/// Inputs and timing state for one guest frame.
#[cfg(test)]
pub(crate) struct GuestFrame<'a> {
    /// Packed HDR input layers and composite mode.
    pub(crate) inputs: GuestInputs<'a>,
    /// Sanitized exposure controls.
    pub(crate) settings: AutoExposureSettings,
    /// Frame delta in seconds.
    pub(crate) dt: f32,
    /// Exposure multiplier from the previous frame.
    pub(crate) previous_exposure: f32,
}

/// Histogram snapshot and smoothed exposure emitted by one guest frame.
#[cfg(test)]
pub(crate) struct GuestOutput {
    pub(crate) histogram: [u32; HISTOGRAM_BINS],
    pub(crate) exposure: f32,
}

/// Persistent guest pipelines and buffers, reusable across frames of one size.
pub(crate) struct GuestGpu {
    bind_group: wgpu::BindGroup,
    _layout: wgpu::BindGroupLayout,
    log_luma: wgpu::ComputePipeline,
    histogram_pass: wgpu::ComputePipeline,
    adapt: wgpu::ComputePipeline,
    input: wgpu::Buffer,
    _log_values: wgpu::Buffer,
    histogram: wgpu::Buffer,
    exposure: wgpu::Buffer,
    frame: wgpu::Buffer,
    histogram_params: wgpu::Buffer,
    adapt_params: wgpu::Buffer,
    #[cfg(test)]
    staging: wgpu::Buffer,
    pixel_count: u32,
}

impl GuestGpu {
    /// Builds compute pipelines and exact-sized storage/uniform buffers.
    pub(crate) fn new(device: &wgpu::Device, lowered: &Lowered, width: u32, height: u32) -> Self {
        let pixel_count = width.checked_mul(height).unwrap_or_default();
        assert!(pixel_count > 0, "guest image must contain a pixel");
        let storage = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wasm2gpu auto-exposure layout"),
            entries: &[
                binding(INPUT_BINDING, storage),
                binding(LOG_LUMINANCE_BINDING, storage),
                binding(HISTOGRAM_BINDING, storage),
                binding(FRAME_BINDING, storage),
                binding(HISTOGRAM_PARAMS_BINDING, wgpu::BufferBindingType::Uniform),
                binding(ADAPT_PARAMS_BINDING, wgpu::BufferBindingType::Uniform),
                binding(EXPOSURE_BINDING, storage),
            ],
        });
        let log_luma = pipeline(device, &layout, &lowered.log_luma, "ae_log_luma");
        let histogram_pass = pipeline(device, &layout, &lowered.histogram, "ae_histogram");
        let adapt = pipeline(device, &layout, &lowered.adapt, "ae_adapt");
        let input = storage_buffer(device, "ae packed hdr", pixel_count as usize * INPUT_LANES);
        let log_values = storage_buffer(device, "ae log luminance", pixel_count as usize);
        let histogram = storage_buffer(device, "ae u32 histogram", HISTOGRAM_BINS);
        let exposure = storage_buffer(device, "ae exposure history", EXPOSURE_SLOTS);
        let frame = storage_buffer(device, "ae frame settings", FRAME_WORDS);
        let histogram_params = uniform_buffer(device, "ae histogram params", HISTOGRAM_PARAMS_LEN);
        let adapt_params = uniform_buffer(device, "ae adapt params", ADAPT_PARAMS_LEN);
        #[cfg(test)]
        let staging = gpu_buffer(
            device,
            "ae guest readback",
            STAGING_BYTES,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wasm2gpu auto-exposure bind group"),
            layout: &layout,
            entries: &[
                bind_entry(INPUT_BINDING, &input),
                bind_entry(LOG_LUMINANCE_BINDING, &log_values),
                bind_entry(HISTOGRAM_BINDING, &histogram),
                bind_entry(FRAME_BINDING, &frame),
                bind_entry(HISTOGRAM_PARAMS_BINDING, &histogram_params),
                bind_entry(ADAPT_PARAMS_BINDING, &adapt_params),
                bind_entry(EXPOSURE_BINDING, &exposure),
            ],
        });
        Self {
            bind_group,
            _layout: layout,
            log_luma,
            histogram_pass,
            adapt,
            input,
            _log_values: log_values,
            histogram,
            exposure,
            frame,
            histogram_params,
            adapt_params,
            #[cfg(test)]
            staging,
            pixel_count,
        }
    }

    /// Uploads one storage mirror, the uniform tables, and the initial exposure.
    pub(crate) fn prepare(
        &self,
        queue: &wgpu::Queue,
        inputs: GuestInputs<'_>,
        settings: AutoExposureSettings,
        dt: f32,
        previous_exposure: f32,
    ) {
        assert_eq!(
            inputs.width.checked_mul(inputs.height),
            Some(self.pixel_count)
        );
        assert_eq!(inputs.deferred.len(), self.pixel_count as usize);
        assert_eq!(inputs.forward.len(), self.pixel_count as usize);
        let mut packed = vec![0.0_f32; self.pixel_count as usize * INPUT_LANES];
        for pixel in 0..self.pixel_count as usize {
            let at = pixel * INPUT_LANES;
            packed[at..at + RGBA_LANES].copy_from_slice(&inputs.deferred[pixel]);
            packed[at + RGBA_LANES..at + INPUT_LANES].copy_from_slice(&inputs.forward[pixel]);
        }
        queue.write_buffer(&self.input, 0, cast_slice(&packed));
        queue.write_buffer(&self.frame, 0, cast_slice(&[self.pixel_count, inputs.mode]));
        let settings = settings.sanitized();
        let histogram_params = histogram_params(&settings);
        let adapt_params = adapt_params(&settings, dt);
        queue.write_buffer(&self.histogram_params, 0, &uniform_bytes(&histogram_params));
        queue.write_buffer(&self.adapt_params, 0, &uniform_bytes(&adapt_params));
        let histogram = [0_u32; HISTOGRAM_BINS];
        queue.write_buffer(&self.histogram, 0, cast_slice(&histogram));
        queue.write_buffer(&self.exposure, 0, cast_slice(&[previous_exposure, 0.0_f32]));
    }

    /// Records log-luminance, one-bin-per-lane scan, and adaptation passes.
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_histogram(encoder);
        self.encode_adaptation(encoder);
    }

    fn encode_histogram(&self, encoder: &mut wgpu::CommandEncoder) {
        let lanes = symbols::CS_WORKGROUP[0];
        dispatch(
            encoder,
            &self.log_luma,
            &self.bind_group,
            self.pixel_count.div_ceil(lanes),
            "ae guest log luminance",
        );
        dispatch(
            encoder,
            &self.histogram_pass,
            &self.bind_group,
            (HISTOGRAM_BINS as u32).div_ceil(lanes),
            "ae guest histogram",
        );
    }

    fn encode_adaptation(&self, encoder: &mut wgpu::CommandEncoder) {
        dispatch(encoder, &self.adapt, &self.bind_group, 1, "ae guest adapt");
    }

    /// Executes one frame and snapshots the histogram before the guest clears it.
    #[cfg(test)]
    pub(crate) fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: GuestFrame<'_>,
    ) -> Option<GuestOutput> {
        self.prepare(
            queue,
            frame.inputs,
            frame.settings,
            frame.dt,
            frame.previous_exposure,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wasm2gpu auto-exposure encoder"),
        });
        self.encode_histogram(&mut encoder);
        encoder.copy_buffer_to_buffer(
            &self.histogram,
            0,
            &self.staging,
            0,
            HISTOGRAM_SNAPSHOT_BYTES,
        );
        self.encode_adaptation(&mut encoder);
        encoder.copy_buffer_to_buffer(
            &self.exposure,
            EXPOSURE_OUTPUT_OFFSET,
            &self.staging,
            EXPOSURE_READBACK_OFFSET,
            FLOAT_BYTES,
        );
        queue.submit(std::iter::once(encoder.finish()));
        output_from_bytes(&map_bytes(device, &self.staging))
    }
}
