//! GPU eye adaptation (auto-exposure) driven by a log-luminance histogram
//! ([ornis #125](https://github.com/kemperrrrr/ornis/issues/125)).
//!
//! Instead of a fixed exposure the frame can adapt to scene brightness
//! (walking out of a room into daylight or vice versa). The pass is fully
//! GPU-resident — the histogram of the lit HDR target is built with an
//! atomic-histogram compute pass, a second compute pass reduces it to a
//! clipped average log-luminance, converts that to a target *exposure value
//! in EV* (with an EV compensation) and smooths it over time, and the
//! composite tone-mapper multiplies the HDR color by the resulting exposure.
//! No value is ever read back to the CPU, so the CPU never stalls on the GPU.
//!
//! Manual mode stays available ([`ExposureMode::Manual`], exposure `1.0`):
//! it records no compute work and writes the constant multiplier into the
//! same uniform the composite reads, so the manual path is pixel-identical
//! to the pre-auto-exposure frame (the golden probes stay green).
//!
//! The recipe follows Bevy's auto exposure:
//!
//! * the histogram measures the same HDR the tone-mapper sees — the
//!   `mode`-selected mix of the deferred and forward layers, never a lone
//!   deferred layer (so forward geometry is metered too);
//! * bin `0` is reserved for texels below `2^min_log_lum` and is excluded
//!   from the average, so black backgrounds and letterbox bars do not drag
//!   the exposure up. A frame with no in-range texel holds the previous
//!   exposure;
//! * tail clipping uses a *partial* bin weight
//!   (`clamp(cum, low, high) - clamp(cum_before, low, high)`), so the
//!   clipped quantiles are honoured to sub-bin precision;
//! * smoothing runs in EV (log2 exposure): the step is
//!   `min(speed*dt, |delta| * speed*dt / transition_distance)`, clamped to
//!   `speed*dt`, which behaves symmetrically for brightening and darkening;
//! * the frame delta time is supplied by the host through
//!   [`Renderer3D::set_frame_time`](crate::renderer::Renderer3D::set_frame_time);
//!   [`DEFAULT_FRAME_DT_SECONDS`] is only a fallback used before the first
//!   host call.

mod wgsl;

#[cfg(test)]
mod tests;

pub(crate) use wgsl::{adapt_wgsl, histogram_wgsl};

use bytemuck::Pod;
use bytemuck::Zeroable;
use wgpu::util::DeviceExt as _;

/// Number of histogram bins. Bin `0` is reserved for texels below the
/// histogram window; the in-range texels land in bins `1..HISTOGRAM_BINS`.
pub const HISTOGRAM_BINS: usize = 256;

/// Threads per histogram workgroup axis: the pass is 2D (`8×8`), one
/// thread per texel.
pub const HISTOGRAM_WORKGROUP_SIZE: u32 = 8;

/// Lowest log2-luminance a texel can map to (`2^-8` ≈ 0.0039); darker
/// texels saturate into bin 0 and are excluded from the average.
pub const DEFAULT_MIN_LOG_LUM: f32 = -8.0;

/// Highest log2-luminance a texel can map to (`2^8` = 256); brighter texels
/// saturate into the last bin.
pub const DEFAULT_MAX_LOG_LUM: f32 = 8.0;

/// Floor applied to a texel's luminance before `log2`, so black pixels are
/// finite and land deterministically in bin 0.
const LUMINANCE_FLOOR: f32 = 1e-6;

/// Minimum adaptation speed (EV/s); guards the step against a zero rate.
const MIN_ADAPTATION_SPEED: f32 = 1e-3;

/// Minimum exponential transition distance (EV); guards the division.
const MIN_TRANSITION_DISTANCE: f32 = 1e-3;

/// Half a bin: the offset from a bin's lower edge to its centre.
const BIN_CENTER_OFFSET: f32 = 0.5;

/// Bind-group slot of the [`HistogramParams`] uniform in the histogram pass.
const HISTOGRAM_PARAMS_BINDING: u32 = 3;

/// Rec.709 luma weights of a linear-RGB texel (shared by the CPU reference
/// and the WGSL `log_luminance`).
const REC709_LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Number of in-range bins (every bin but the reserved bin `0`).
const IN_RANGE_BINS: f32 = HISTOGRAM_BINS as f32 - 2.0;

/// Fallback frame delta (seconds) used by the auto-exposure smoothing until
/// the host supplies a measured delta through
/// [`Renderer3D::set_frame_time`](crate::renderer::Renderer3D::set_frame_time).
///
/// The adaptation speed is expressed in F-stops per second, so it is only
/// correct when `dt` is the real elapsed time (Bevy's `globals.delta_time`).
/// This constant merely keeps the very first frames well-defined; hosts
/// should call `set_frame_time` every frame.
pub const DEFAULT_FRAME_DT_SECONDS: f32 = 1.0 / 60.0;

/// Smallest frame delta (seconds) the smoothing accepts; guards against a
/// stalled or bogus measurement freezing the adaptation.
pub const MIN_FRAME_DT_SECONDS: f32 = 1e-4;

/// Largest frame delta (seconds) the smoothing accepts; guards against a
/// long pause advancing the adaptation by an entire second in one step.
pub const MAX_FRAME_DT_SECONDS: f32 = 1.0;

/// WGSL struct name of [`ExposureParams`]; the composite resource table uses
/// it as the uniform type.
pub const EXPOSURE_WGSL_NAME: &str = "ExposureParams";

/// WGSL declaration of [`ExposureParams`].
///
/// Four `f32` (the multiplier plus three pad words) keep the struct a
/// 16-byte uniform block that both the composite fragment stage and the
/// adapt compute stage can bind.
pub const EXPOSURE_PARAMS_WGSL: &str = "\
struct ExposureParams {
    exposure: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}";

/// The post-tone-mapper exposure multiplier shared by the auto-exposure
/// adapt pass (writes it) and the HDR composite (reads it as a uniform).
///
/// The `pad` words are never read; they exist only so the uniform block is
/// 16 bytes and matches the WGSL declaration above.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct ExposureParams {
    /// Exposure multiplier applied to the HDR color before tone mapping.
    pub exposure: f32,
    /// Padding to the 16-byte uniform block (never read).
    pub pad: [f32; 3],
}

impl ExposureParams {
    /// WGSL member names the shader-lowering scanner may resolve on an
    /// `exposure_params` use (the pad words are never read).
    pub const FIELD_NAMES: &'static [&'static str] = &["exposure"];

    /// Packs an exposure multiplier (pad zeroed).
    pub fn pack(exposure: f32) -> Self {
        Self {
            exposure,
            pad: [0.0; 3],
        }
    }
}

/// How the frame's post-tone-mapper exposure is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExposureMode {
    /// Fixed multiplier (default `1.0`, the exact no-op). Keeps the manual
    /// path pixel-identical to the pre-auto-exposure frame.
    Manual {
        /// Exposure multiplier applied before tone mapping.
        multiplier: f32,
    },
    /// Eye adaptation from the log-luminance histogram of the lit HDR target.
    Auto(AutoExposureSettings),
}

impl Default for ExposureMode {
    fn default() -> Self {
        Self::Manual { multiplier: 1.0 }
    }
}

/// Tuning knobs of the auto-exposure histogram pass.
///
/// The names describe the direction of the *exposure* (and therefore the
/// rendered image), never the direction of the scene brightness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoExposureSettings {
    /// Target average scene luminance (the classic "key" value, ~`0.18`).
    pub target_luminance: f32,
    /// Exposure-compensation bias in stops (EV): positive brightens.
    pub ev_compensation: f32,
    /// Lower clamp of the exposure, in stops (`2^min_ev`).
    pub min_ev: f32,
    /// Upper clamp of the exposure, in stops (`2^max_ev`).
    pub max_ev: f32,
    /// Lowest log2-luminance bin edge.
    pub min_log_lum: f32,
    /// Highest log2-luminance bin edge.
    pub max_log_lum: f32,
    /// Lower edge of the metered quantile window: the darkest
    /// `low_percentile` of the in-range samples are ignored.
    pub low_percentile: f32,
    /// Upper edge of the metered quantile window: the brightest
    /// `1 - high_percentile` of the in-range samples are ignored. Must be
    /// strictly greater than [`low_percentile`](Self::low_percentile).
    pub high_percentile: f32,
    /// How fast the exposure **rises** (image brightens) when the scene is
    /// too dark, in EV (F-stops) per second.
    pub brighten_speed: f32,
    /// How fast the exposure **falls** (image darkens) when the scene is too
    /// bright, in EV (F-stops) per second.
    pub darken_speed: f32,
    /// Distance in EV from the target at which adaptation switches from the
    /// linear regime (`|delta| * speed * dt / distance`) to the saturated
    /// `speed * dt` regime. Dampens frame-to-frame jitter while keeping large
    /// changes fast.
    pub exponential_transition_distance: f32,
}

impl Default for AutoExposureSettings {
    fn default() -> Self {
        Self {
            target_luminance: 0.18,
            ev_compensation: 0.0,
            min_ev: -8.0,
            max_ev: 8.0,
            min_log_lum: DEFAULT_MIN_LOG_LUM,
            max_log_lum: DEFAULT_MAX_LOG_LUM,
            low_percentile: 0.10,
            high_percentile: 0.90,
            brighten_speed: 3.0,
            darken_speed: 1.0,
            exponential_transition_distance: 1.5,
        }
    }
}

/// Replaces `NaN`/infinite values with `fallback`.
fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

impl AutoExposureSettings {
    /// Sanitises every knob: `NaN`s fall back to the default, the histogram
    /// window stays ordered, the quantile window stays a non-empty
    /// `0 <= low < high <= 1`, the speeds stay positive, and the transition
    /// distance stays above the guard. The defaults already satisfy this, so
    /// the default path is unchanged.
    pub fn sanitized(self) -> Self {
        let default = Self::default();
        let min_log_lum = finite_or(self.min_log_lum, default.min_log_lum);
        let max_log_lum = finite_or(self.max_log_lum, default.max_log_lum).max(min_log_lum);
        let low = finite_or(self.low_percentile, default.low_percentile).clamp(0.0, 1.0);
        let high = finite_or(self.high_percentile, default.high_percentile).clamp(0.0, 1.0);
        let high = high.max(low + f32::EPSILON);
        let min_ev = finite_or(self.min_ev, default.min_ev);
        Self {
            target_luminance: finite_or(self.target_luminance, default.target_luminance)
                .max(f32::EPSILON),
            ev_compensation: finite_or(self.ev_compensation, default.ev_compensation),
            min_ev,
            max_ev: finite_or(self.max_ev, default.max_ev).max(min_ev),
            min_log_lum,
            max_log_lum,
            low_percentile: low,
            high_percentile: high,
            brighten_speed: finite_or(self.brighten_speed, default.brighten_speed)
                .max(MIN_ADAPTATION_SPEED),
            darken_speed: finite_or(self.darken_speed, default.darken_speed)
                .max(MIN_ADAPTATION_SPEED),
            exponential_transition_distance: finite_or(
                self.exponential_transition_distance,
                default.exponential_transition_distance,
            )
            .max(MIN_TRANSITION_DISTANCE),
        }
    }

    /// Log2 span covered by the histogram (always at least one bin wide).
    pub fn log_range(&self) -> f32 {
        (self.max_log_lum - self.min_log_lum).max(f32::EPSILON)
    }

    /// Inverse log2 span: `(log - min_log_lum) * inv_log_range` yields the
    /// normalised in-range coordinate in `[0, 1]`.
    pub fn inv_log_range(&self) -> f32 {
        1.0 / self.log_range()
    }

    /// Target exposure value in EV that maps the average scene luminance to
    /// the key luminance (plus the EV compensation).
    pub fn target_ev(&self) -> f32 {
        self.target_luminance.log2() + self.ev_compensation
    }

    /// Lower clamp of the exposure multiplier.
    pub fn min_exposure(&self) -> f32 {
        exp2(self.min_ev)
    }

    /// Upper clamp of the exposure multiplier.
    pub fn max_exposure(&self) -> f32 {
        exp2(self.max_ev)
    }
}

/// `2^x` in `f32`.
fn exp2(x: f32) -> f32 {
    (x * std::f32::consts::LN_2).exp()
}

/// Log2-luminance of a linear-RGB texel, floored so black is finite.
///
/// Mirrors the WGSL `log_luminance` in [`histogram_wgsl`].
pub fn log_luminance(rgb: [f32; 3]) -> f32 {
    let lum = REC709_LUMA[0] * rgb[0] + REC709_LUMA[1] * rgb[1] + REC709_LUMA[2] * rgb[2];
    lum.max(LUMINANCE_FLOOR).log2()
}

/// Histogram bin index of a log2-luminance, in `0..HISTOGRAM_BINS`.
///
/// Bin `0` is reserved for texels below the window (`log_lum <
/// min_log_lum`); in-range texels map into `1..HISTOGRAM_BINS`. Mirrors the
/// WGSL `histogram_bin` in [`histogram_wgsl`].
pub fn histogram_bin(log_lum: f32, settings: &AutoExposureSettings) -> usize {
    if log_lum < settings.min_log_lum {
        return 0;
    }
    let t = ((log_lum - settings.min_log_lum) * settings.inv_log_range()).clamp(0.0, 1.0);
    (t * IN_RANGE_BINS) as usize + 1
}

/// Log2-luminance represented by in-range bin `index` (`1..HISTOGRAM_BINS`),
/// i.e. the centre of the bin's normalised interval.
///
/// Mirrors the reconstruction in [`adapt_wgsl`].
pub fn bin_log_luminance(index: usize, settings: &AutoExposureSettings) -> f32 {
    let t = ((index as f32 - BIN_CENTER_OFFSET) / IN_RANGE_BINS).clamp(0.0, 1.0);
    settings.min_log_lum + t * settings.log_range()
}

/// CPU reference of the histogram pass: bin counts of the log-luminance of
/// every texel in `texels`.
// qual:test_helper
pub fn build_histogram(
    texels: &[[f32; 3]],
    settings: &AutoExposureSettings,
) -> [u32; HISTOGRAM_BINS] {
    let mut hist = [0_u32; HISTOGRAM_BINS];
    for texel in texels {
        hist[histogram_bin(log_luminance(*texel), settings)] += 1;
    }
    hist
}

/// CPU reference of the reduction: the average log2-luminance over the
/// metered quantile window `[low_percentile, high_percentile]` of the
/// *in-range* samples (bin `0` is excluded), with partial bin weights.
///
/// Returns `None` when no texel falls inside the histogram window (a black
/// frame); the caller then holds the previous exposure.
///
/// Mirrors the reduction loop in [`adapt_wgsl`].
// qual:test_helper
pub fn average_log_luminance(
    histogram: &[u32; HISTOGRAM_BINS],
    settings: &AutoExposureSettings,
) -> Option<f32> {
    let total: u64 = histogram[1..].iter().map(|c| u64::from(*c)).sum();
    if total == 0 {
        return None;
    }
    let low = f64::from(settings.low_percentile) * total as f64;
    let high = f64::from(settings.high_percentile) * total as f64;
    let mut acc = 0.0_f64;
    let mut weight_total = 0.0_f64;
    let mut cum = 0_u64;
    for (i, count) in histogram.iter().enumerate().skip(1) {
        let cum_before = cum;
        cum += u64::from(*count);
        let weight = (cum as f64).clamp(low, high) - (cum_before as f64).clamp(low, high);
        acc += weight * f64::from(bin_log_luminance(i, settings));
        weight_total += weight;
    }
    if weight_total <= 0.0 {
        return None;
    }
    Some((acc / weight_total) as f32)
}

/// Exposure value (in EV) that maps `avg_log_lum` to the target luminance
/// (plus the EV compensation), clamped to the settings' EV window.
///
/// Mirrors the target computation in [`adapt_wgsl`].
pub fn desired_exposure_ev(avg_log_lum: f32, settings: &AutoExposureSettings) -> f32 {
    (settings.target_ev() - avg_log_lum).clamp(settings.min_ev, settings.max_ev)
}

/// Exposure multiplier for `avg_log_lum` ([`desired_exposure_ev`] applied to
/// `2^x`).
// qual:test_helper
pub fn desired_exposure(avg_log_lum: f32, settings: &AutoExposureSettings) -> f32 {
    exp2(desired_exposure_ev(avg_log_lum, settings))
        .clamp(settings.min_exposure(), settings.max_exposure())
}

/// One EV smoothing step: advance `prev_ev` toward `target_ev` by
/// `min(|delta|, speed*dt, |delta| * speed*dt / transition_distance)` where
/// `speed` is `brighten_speed` when the exposure rises and `darken_speed`
/// when it falls. The `|delta|` clamp keeps the step from ever overshooting
/// the target (so a large `dt` lands exactly on it).
///
/// Mirrors the smoothing tail of [`adapt_wgsl`].
// qual:test_helper
pub fn adapt_exposure_ev(
    prev_ev: f32,
    target_ev: f32,
    dt_seconds: f32,
    settings: &AutoExposureSettings,
) -> f32 {
    let delta = target_ev - prev_ev;
    let speed = if delta > 0.0 {
        settings.brighten_speed
    } else {
        settings.darken_speed
    }
    .max(MIN_ADAPTATION_SPEED);
    let max_step = speed * dt_seconds;
    let step = max_step
        .min(delta.abs() * max_step / settings.exponential_transition_distance)
        .min(delta.abs());
    (prev_ev + delta.signum() * step).clamp(settings.min_ev, settings.max_ev)
}

/// Uniform block of the histogram pass.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
struct HistogramParams {
    min_log_lum: f32,
    inv_log_range: f32,
    bin_count: f32,
    mode: u32,
}

impl HistogramParams {
    fn from_settings(settings: &AutoExposureSettings, mode: u32) -> Self {
        Self {
            min_log_lum: settings.min_log_lum,
            inv_log_range: settings.inv_log_range(),
            bin_count: HISTOGRAM_BINS as f32,
            mode,
        }
    }
}

/// Uniform block of the adapt pass.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
struct AdaptParams {
    min_log_lum: f32,
    log_range: f32,
    low_clip: f32,
    high_clip: f32,
    target_ev: f32,
    min_ev: f32,
    max_ev: f32,
    brighten_speed: f32,
    darken_speed: f32,
    transition_distance: f32,
    dt: f32,
    _pad: f32,
}

impl AdaptParams {
    fn from_settings(settings: &AutoExposureSettings, dt: f32) -> Self {
        Self {
            min_log_lum: settings.min_log_lum,
            log_range: settings.log_range(),
            low_clip: settings.low_percentile,
            high_clip: settings.high_percentile,
            target_ev: settings.target_ev(),
            min_ev: settings.min_ev,
            max_ev: settings.max_ev,
            brighten_speed: settings.brighten_speed,
            darken_speed: settings.darken_speed,
            transition_distance: settings.exponential_transition_distance,
            dt,
            _pad: 0.0,
        }
    }
}

/// Bind-group-layout entry for one compute buffer binding.
fn compute_buffer_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Bind-group-layout entry for a histogram HDR texture input.
///
/// The pass only `textureLoad`s, so the bindings are declared non-filterable
/// and accept any float view (the `Rgba16Float` HDR target included).
fn compute_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// One frame of inputs for [`AutoExposure::encode`].
pub struct ExposureFrame<'a> {
    /// The deferred HDR layer to meter.
    pub hdr: &'a wgpu::TextureView,
    /// The forward HDR layer to meter (bound as `hdr` when `mode == 0`).
    pub hdr_fwd: &'a wgpu::TextureView,
    /// Layer mix selector matching the composite: `0` deferred-only, `1`
    /// forward-only, `2` hybrid (`deferred + forward * alpha`).
    pub mode: u32,
    /// HDR target width in texels (the bound view's extent).
    pub width: u32,
    /// HDR target height in texels (the bound view's extent).
    pub height: u32,
    /// Auto-exposure tuning for this frame.
    pub settings: AutoExposureSettings,
    /// Frame delta time in seconds, for the temporal smoothing.
    pub dt_seconds: f32,
}

/// GPU resources of the auto-exposure pass: the histogram/adapt pipelines,
/// the persistent histogram and uniform buffers, and the shared
/// [`ExposureParams`] uniform block the composite reads.
///
/// Bind groups are rebuilt per frame (like the composite/bloom passes here),
/// so a pooled HDR view can be histogrammed without recreating the pass.
pub struct AutoExposure {
    histogram_pipeline: wgpu::ComputePipeline,
    adapt_pipeline: wgpu::ComputePipeline,
    histogram_layout: wgpu::BindGroupLayout,
    adapt_layout: wgpu::BindGroupLayout,
    histogram_buffer: wgpu::Buffer,
    histogram_params: wgpu::Buffer,
    adapt_params: wgpu::Buffer,
    exposure_buffer: wgpu::Buffer,
    workgroup: u32,
}

impl AutoExposure {
    /// Builds the pipelines and buffers; the exposure uniform starts at
    /// `1.0` (the manual no-op).
    pub fn new(device: &wgpu::Device) -> Self {
        let (histogram_shader, adapt_shader) = auto_exposure_shaders(device);
        let (histogram_layout, adapt_layout) = auto_exposure_layouts(device);

        let histogram_pipeline = compute_pipeline(
            device,
            "auto-exposure histogram",
            &histogram_layout,
            &histogram_shader,
            "build_histogram",
        );
        let adapt_pipeline = compute_pipeline(
            device,
            "auto-exposure adapt",
            &adapt_layout,
            &adapt_shader,
            "adapt_exposure",
        );

        let default_settings = AutoExposureSettings::default();
        // Zero-initialised: the adapt pass clears it at the end of every
        // frame, so the build pass always accumulates onto zeros. The first
        // frame relies on this initial state.
        let histogram_buffer = init_buffer(
            device,
            "auto-exposure histogram buffer",
            &vec![0_u8; HISTOGRAM_BINS * std::mem::size_of::<u32>()],
            // `COPY_SRC` lets tests and diagnostics read the bins back.
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let histogram_params = init_buffer(
            device,
            "auto-exposure histogram params",
            bytemuck::bytes_of(&HistogramParams::from_settings(&default_settings, 0)),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let adapt_params = init_buffer(
            device,
            "auto-exposure adapt params",
            bytemuck::bytes_of(&AdaptParams::from_settings(
                &default_settings,
                DEFAULT_FRAME_DT_SECONDS,
            )),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let exposure_buffer = init_buffer(
            device,
            "auto-exposure exposure uniform",
            bytemuck::bytes_of(&ExposureParams::pack(1.0)),
            wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        Self {
            histogram_pipeline,
            adapt_pipeline,
            histogram_layout,
            adapt_layout,
            histogram_buffer,
            histogram_params,
            adapt_params,
            exposure_buffer,
            workgroup: HISTOGRAM_WORKGROUP_SIZE,
        }
    }

    /// The shared exposure uniform block, bound by the composite pass.
    pub fn exposure_buffer(&self) -> &wgpu::Buffer {
        &self.exposure_buffer
    }

    /// Writes a constant exposure multiplier (manual mode).
    pub fn write_manual(&self, queue: &wgpu::Queue, exposure: f32) {
        queue.write_buffer(
            &self.exposure_buffer,
            0,
            bytemuck::bytes_of(&ExposureParams::pack(exposure)),
        );
    }

    /// Records one auto-exposure update onto `encoder`: build the histogram
    /// of the `mode`-selected HDR mix (a non-filterable `texture_2d<f32>`
    /// view of `frame.width × frame.height`), then reduce it to the smoothed
    /// exposure. Two compute passes so the reduction sees a completed build;
    /// the adapt pass clears the histogram for the next frame.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: ExposureFrame<'_>,
    ) {
        self.write_params(queue, &frame);
        self.record_histogram(device, encoder, &frame);
        self.record_adapt(device, encoder);
    }

    /// Test-only path recording just the histogram build (no reduction, so
    /// the counts stay readable for a bin-for-bin GPU-vs-CPU comparison).
    // qual:test_helper
    pub fn encode_histogram_only(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: ExposureFrame<'_>,
    ) {
        self.write_params(queue, &frame);
        self.record_histogram(device, encoder, &frame);
    }

    /// Uploads the histogram/adapt uniforms for one frame.
    fn write_params(&self, queue: &wgpu::Queue, frame: &ExposureFrame<'_>) {
        let settings = frame.settings.sanitized();
        queue.write_buffer(
            &self.histogram_params,
            0,
            bytemuck::bytes_of(&HistogramParams::from_settings(&settings, frame.mode)),
        );
        queue.write_buffer(
            &self.adapt_params,
            0,
            bytemuck::bytes_of(&AdaptParams::from_settings(&settings, frame.dt_seconds)),
        );
    }

    /// Records the histogram build pass of one frame.
    fn record_histogram(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame: &ExposureFrame<'_>,
    ) {
        let groups = [
            frame.width.div_ceil(self.workgroup),
            frame.height.div_ceil(self.workgroup),
        ];
        let histogram_bind = bind_group(
            device,
            "auto-exposure histogram bind group",
            &self.histogram_layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(frame.hdr),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(frame.hdr_fwd),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: HISTOGRAM_PARAMS_BINDING,
                    resource: self.histogram_params.as_entire_binding(),
                },
            ],
        );
        run_pass(
            encoder,
            "auto-exposure histogram pass",
            &self.histogram_pipeline,
            &histogram_bind,
            groups,
        );
    }

    /// Records the histogram reduction pass of one frame (also clears the
    /// histogram for the next frame).
    fn record_adapt(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let adapt_bind = bind_group(
            device,
            "auto-exposure adapt bind group",
            &self.adapt_layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.adapt_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.exposure_buffer.as_entire_binding(),
                },
            ],
        );
        run_pass(
            encoder,
            "auto-exposure adapt pass",
            &self.adapt_pipeline,
            &adapt_bind,
            [1, 1],
        );
    }
}

/// Creates a WGSL compute shader module.
fn compute_shader(device: &wgpu::Device, label: &str, wgsl: String) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    })
}

/// Creates a compute bind-group layout from its entries.
fn compute_layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

/// Builds both auto-exposure shader modules (histogram, then adapt).
fn auto_exposure_shaders(device: &wgpu::Device) -> (wgpu::ShaderModule, wgpu::ShaderModule) {
    (
        compute_shader(device, "auto-exposure histogram", histogram_wgsl()),
        compute_shader(device, "auto-exposure adapt", adapt_wgsl()),
    )
}

/// Builds both bind-group layouts (histogram, then adapt).
fn auto_exposure_layouts(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    let histogram = compute_layout(
        device,
        "auto-exposure histogram layout",
        &[
            compute_texture_entry(0),
            compute_texture_entry(1),
            compute_buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            compute_buffer_entry(HISTOGRAM_PARAMS_BINDING, wgpu::BufferBindingType::Uniform),
        ],
    );
    let adapt = compute_layout(
        device,
        "auto-exposure adapt layout",
        &[
            compute_buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
            compute_buffer_entry(1, wgpu::BufferBindingType::Uniform),
            compute_buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    );
    (histogram, adapt)
}

/// Creates a device buffer pre-initialised from `contents`.
fn init_buffer(
    device: &wgpu::Device,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage,
    })
}

/// Creates a bind group from a layout and its entries.
fn bind_group<'a>(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    entries: &[wgpu::BindGroupEntry<'a>],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries,
    })
}

/// Records one compute pass dispatching an `x × y` workgroup grid.
fn run_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bind: &wgpu::BindGroup,
    groups: [u32; 2],
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    pass.dispatch_workgroups(groups[0], groups[1], 1);
}

/// Builds a single-entry compute pipeline.
fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}
