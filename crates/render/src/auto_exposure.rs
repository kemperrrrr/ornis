//! GPU eye adaptation (auto-exposure) driven by a log-luminance histogram
//! ([ornis #125](https://github.com/kemperrrrr/ornis/issues/125)).
//!
//! Instead of a fixed exposure the frame can adapt to scene brightness
//! (walking out of a room into daylight or vice versa). The pass is fully
//! GPU-resident — the histogram of the lit HDR target is built with an
//! atomic-histogram compute pass, a second compute pass reduces it to a
//! clipped average log-luminance, converts that to a target exposure (with
//! an EV compensation) and smooths it over time, and the composite
//! tone-mapper multiplies the HDR color by the resulting exposure. No
//! value is ever read back to the CPU, so the CPU never stalls on the GPU.
//!
//! Manual mode stays available ([`crate::auto_exposure::ExposureMode::Manual`], exposure `1.0`):
//! it records no compute work and writes the constant multiplier into the
//! same uniform the composite reads, so the manual path is pixel-identical
//! to the pre-auto-exposure frame (the golden probes stay green).
//!
//! Settings mirror the classic histogram eye-adaptation recipe:
//! target-key luminance, EV compensation, per-percentile tail clipping,
//! and separate up/down adaptation speeds.

// qual:allow(srp, file_length=640) reason: "one cohesive GPU feature — the histogram build, the reduction and their settings only function together; splitting them would scatter a single pass across files."

use bytemuck::Pod;
use bytemuck::Zeroable;
use wgpu::util::DeviceExt as _;

/// Number of histogram bins. Also baked into the WGSL loop bound below.
pub const HISTOGRAM_BINS: usize = 256;

/// Threads per histogram workgroup axis: the pass is 2D (`8×8`), one
/// thread per texel.
pub const HISTOGRAM_WORKGROUP_SIZE: u32 = 8;

/// Lowest log2-luminance a texel can map to (`2^-8` ≈ 0.0039); darker
/// texels saturate into bin 0.
pub const DEFAULT_MIN_LOG_LUM: f32 = -8.0;

/// Highest log2-luminance a texel can map to (`2^4` = 16); brighter texels
/// saturate into the last bin.
pub const DEFAULT_MAX_LOG_LUM: f32 = 4.0;

/// Floor applied to a texel's luminance before `log2`, so black pixels are
/// finite and land deterministically in bin 0.
const LUMINANCE_FLOOR: f32 = 1e-6;

/// Minimum adaptation speed (seconds); guards the exponential against a
/// zero rate.
const MIN_ADAPTATION_SPEED: f32 = 1e-3;

/// Rec.709 luma weights of a linear-RGB texel (shared by the CPU reference
/// and the WGSL `log_luminance`).
const REC709_LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Offset from a bin's lower edge to its centre (half a bin).
const BIN_CENTER_OFFSET: f32 = 0.5;

/// Largest fraction of either histogram tail the settings may clip.
const MAX_TAIL_CLIP: f32 = 0.5;

/// Default frame delta (seconds) packed into the adapt params at startup;
/// the renderer overwrites it with the measured frame time each frame.
const DEFAULT_FRAME_DT: f32 = 1.0 / 60.0;

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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoExposureSettings {
    /// Target average scene luminance (the classic "key" value, ~`0.18`).
    pub target_luminance: f32,
    /// Exposure-compensation bias in stops (EV): positive brightens.
    pub ev_compensation: f32,
    /// Lower clamp of the exposure multiplier, in stops (`2^min_ev`).
    pub min_ev: f32,
    /// Upper clamp of the exposure multiplier, in stops (`2^max_ev`).
    pub max_ev: f32,
    /// Adaptation time constant (seconds) when the scene brightens.
    pub speed_up: f32,
    /// Adaptation time constant (seconds) when the scene darkens.
    pub speed_down: f32,
    /// Fraction of darkest pixels clipped off before averaging.
    pub low_percentile: f32,
    /// Fraction of brightest pixels clipped off before averaging.
    pub high_percentile: f32,
    /// Lowest log2-luminance bin edge.
    pub min_log_lum: f32,
    /// Highest log2-luminance bin edge.
    pub max_log_lum: f32,
}

impl Default for AutoExposureSettings {
    fn default() -> Self {
        Self {
            target_luminance: 0.18,
            ev_compensation: 0.0,
            min_ev: -8.0,
            max_ev: 8.0,
            speed_up: 1.5,
            speed_down: 0.5,
            low_percentile: 0.005,
            high_percentile: 0.005,
            min_log_lum: DEFAULT_MIN_LOG_LUM,
            max_log_lum: DEFAULT_MAX_LOG_LUM,
        }
    }
}

impl AutoExposureSettings {
    /// Clamps every knob into a usable range: the histogram window stays
    /// ordered, the tail clips stay inside `[0, 0.5)`, and the adaptation
    /// speeds stay positive. The defaults already satisfy this, so the
    /// default path is unchanged.
    pub fn sanitized(self) -> Self {
        let (min_log_lum, max_log_lum) = (self.min_log_lum.min(self.max_log_lum), self.max_log_lum);
        Self {
            target_luminance: self.target_luminance.max(f32::EPSILON),
            ev_compensation: self.ev_compensation,
            min_ev: self.min_ev,
            max_ev: self.max_ev.max(self.min_ev),
            speed_up: self.speed_up.max(MIN_ADAPTATION_SPEED),
            speed_down: self.speed_down.max(MIN_ADAPTATION_SPEED),
            low_percentile: self.low_percentile.clamp(0.0, MAX_TAIL_CLIP),
            high_percentile: self.high_percentile.clamp(0.0, MAX_TAIL_CLIP),
            min_log_lum,
            max_log_lum: max_log_lum.max(min_log_lum),
        }
    }

    /// Log2 span covered by the histogram (always at least one bin wide).
    pub fn log_range(&self) -> f32 {
        (self.max_log_lum - self.min_log_lum).max(f32::EPSILON)
    }

    /// Inverse bin width: `(log - min_log_lum) * inv_bin_width` yields the
    /// fractional bin index.
    pub fn inv_bin_width(&self) -> f32 {
        HISTOGRAM_BINS as f32 / self.log_range()
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

/// Histogram bin index of a log2-luminance, clamped into `0..HISTOGRAM_BINS`.
///
/// Mirrors the kicker in [`histogram_wgsl`] (`inv_bin_width` is the
/// settings' [`inv_bin_width`](AutoExposureSettings::inv_bin_width)).
pub fn histogram_bin(log_lum: f32, settings: &AutoExposureSettings) -> usize {
    let t = (log_lum - settings.min_log_lum) * settings.inv_bin_width();
    let clamped = t.clamp(0.0, (HISTOGRAM_BINS as f32) - 1.0);
    clamped as usize
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

/// Log2-luminance at the center of bin `i`.
///
/// Mirrors the bin-center reconstruction in [`adapt_wgsl`].
pub fn bin_center_log_lum(i: usize, settings: &AutoExposureSettings) -> f32 {
    settings.min_log_lum + (i as f32 + BIN_CENTER_OFFSET) / settings.inv_bin_width()
}

/// CPU reference of the reduction: the average log2-luminance of the
/// histogram with `low_percentile`/`high_percentile` tails clipped.
///
/// Returns `None` when the histogram is empty or every counted bin is
/// clipped away (the caller keeps the previous exposure).
///
/// Mirrors the clipped-average loop in [`adapt_wgsl`].
// qual:test_helper
pub fn average_log_luminance(
    histogram: &[u32; HISTOGRAM_BINS],
    settings: &AutoExposureSettings,
) -> Option<f32> {
    let total: u64 = histogram.iter().map(|c| u64::from(*c)).sum();
    if total == 0 {
        return None;
    }
    let low_count = (total as f64 * f64::from(settings.low_percentile)) as u64;
    let high_count = (total as f64 * f64::from(settings.high_percentile)) as u64;
    let upper = total.saturating_sub(high_count);
    let mut acc = 0.0_f64;
    let mut acc_count = 0_u64;
    let mut cum = 0_u64;
    for (i, count) in histogram.iter().enumerate() {
        let cum_before = cum;
        cum += u64::from(*count);
        // Include a bin when its cumulative range stays inside the clipped
        // window `[low_count, upper]`.
        if cum > low_count && cum_before < upper {
            acc += f64::from(*count) * f64::from(bin_center_log_lum(i, settings));
            acc_count += u64::from(*count);
        }
    }
    if acc_count == 0 {
        return None;
    }
    Some((acc / acc_count as f64) as f32)
}

/// Exposure multiplier that maps `avg_log_lum` to the target luminance
/// (plus EV compensation), clamped to the settings' EV window.
///
/// Mirrors the target computation in [`adapt_wgsl`].
// qual:test_helper
pub fn desired_exposure(avg_log_lum: f32, settings: &AutoExposureSettings) -> f32 {
    let avg_lum = exp2(avg_log_lum).max(LUMINANCE_FLOOR);
    let target = settings.target_luminance * exp2(settings.ev_compensation) / avg_lum;
    target.clamp(settings.min_exposure(), settings.max_exposure())
}

/// One temporal-smoothing step: move `prev` toward `target` by an
/// exponential rate chosen from `speed_up` (target brighter) or
/// `speed_down` (target darker).
///
/// Mirrors the smoothing tail of [`adapt_wgsl`].
// qual:test_helper
pub fn adapt_exposure(
    prev: f32,
    target: f32,
    dt_seconds: f32,
    settings: &AutoExposureSettings,
) -> f32 {
    let rate = if target > prev {
        settings.speed_up
    } else {
        settings.speed_down
    }
    .max(MIN_ADAPTATION_SPEED);
    let alpha = 1.0 - (-dt_seconds / rate).exp();
    let adapted = prev + (target - prev) * alpha;
    adapted.clamp(settings.min_exposure(), settings.max_exposure())
}

/// Uniform block of the histogram pass.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
struct HistogramParams {
    min_log_lum: f32,
    inv_bin_width: f32,
    bin_count: f32,
    _pad: f32,
}

impl HistogramParams {
    fn from_settings(settings: &AutoExposureSettings) -> Self {
        Self {
            min_log_lum: settings.min_log_lum,
            inv_bin_width: settings.inv_bin_width(),
            bin_count: HISTOGRAM_BINS as f32,
            _pad: 0.0,
        }
    }
}

/// Uniform block of the adapt pass.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
struct AdaptParams {
    min_log_lum: f32,
    inv_bin_width: f32,
    low_clip: f32,
    high_clip: f32,
    target_luminance: f32,
    ev_compensation: f32,
    dt: f32,
    speed_up: f32,
    speed_down: f32,
    min_exposure: f32,
    max_exposure: f32,
    _pad: f32,
}

impl AdaptParams {
    fn from_settings(settings: &AutoExposureSettings, dt: f32) -> Self {
        Self {
            min_log_lum: settings.min_log_lum,
            inv_bin_width: settings.inv_bin_width(),
            low_clip: settings.low_percentile,
            high_clip: settings.high_percentile,
            target_luminance: settings.target_luminance,
            ev_compensation: settings.ev_compensation,
            dt,
            speed_up: settings.speed_up,
            speed_down: settings.speed_down,
            min_exposure: settings.min_exposure(),
            max_exposure: settings.max_exposure(),
            _pad: 0.0,
        }
    }
}

/// WGSL compute pass building the log-luminance histogram of the HDR target.
///
/// Bindings: `0` the HDR texture (`texture_2d<f32>`, sampled by load), `1`
/// the atomic histogram (`array<atomic<u32>, HISTOGRAM_BINS>`, read-write —
/// the CPU zeroes it before each dispatch), `2` the histogram params
/// uniform. One `8×8` workgroup covers 64 texels; the shader guards the
/// ragged tail against `textureDimensions`.
pub fn histogram_wgsl() -> String {
    format!(
        r#"//! GPU log-luminance histogram (auto-exposure, ornis #125).
const HISTOGRAM_BINS: u32 = {bins}u;
const LUMINANCE_FLOOR: f32 = 1e-6;

struct HistogramParams {{
    min_log_lum: f32,
    inv_bin_width: f32,
    bin_count: f32,
    pad: f32,
}};

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, {bins}u>;
@group(0) @binding(2) var<uniform> params: HistogramParams;

fn log_luminance(color: vec3<f32>) -> f32 {{
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    return log2(max(luma, LUMINANCE_FLOOR));
}}

@compute @workgroup_size({wg}, {wg}, 1)
fn build_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let dims = textureDimensions(hdr_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {{
        return;
    }}
    let color = textureLoad(hdr_tex, vec2<i32>(gid.xy), 0).rgb;
    let log_lum = log_luminance(color);
    let t = (log_lum - params.min_log_lum) * params.inv_bin_width;
    let bin = u32(clamp(t, 0.0, params.bin_count - 1.0));
    atomicAdd(&histogram[bin], 1u);
}}
"#,
        bins = HISTOGRAM_BINS,
        wg = HISTOGRAM_WORKGROUP_SIZE,
    )
}

/// WGSL compute pass reducing the histogram to the adapted exposure.
///
/// Bindings: `0` the histogram (`array<u32, HISTOGRAM_BINS>`, read), `1`
/// the adapt params uniform, `2` the [`ExposureParams`] block
/// (read-write — the smoothed exposure is read back as the composite
/// uniform). A single invocation (`workgroup_size(1)`, one workgroup)
/// clips the tails, averages the log-luminance, converts to a target
/// exposure, and advances the temporal smoothing one step.
pub fn adapt_wgsl() -> String {
    format!(
        r#"//! GPU eye adaptation (auto-exposure, ornis #125).
const HISTOGRAM_BINS: u32 = {bins}u;
const LUMINANCE_FLOOR: f32 = 1e-6;
const MIN_ADAPTATION_SPEED: f32 = 1e-3;

struct AdaptParams {{
    min_log_lum: f32,
    inv_bin_width: f32,
    low_clip: f32,
    high_clip: f32,
    target_luminance: f32,
    ev_compensation: f32,
    dt: f32,
    speed_up: f32,
    speed_down: f32,
    min_exposure: f32,
    max_exposure: f32,
    pad: f32,
}};

struct ExposureParams {{
    exposure: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}};

@group(0) @binding(0) var<storage, read> histogram: array<u32, {bins}u>;
@group(0) @binding(1) var<uniform> params: AdaptParams;
@group(0) @binding(2) var<storage, read_write> exposure_out: ExposureParams;

fn bin_center_log_lum(index: u32) -> f32 {{
    return params.min_log_lum + (f32(index) + 0.5) / params.inv_bin_width;
}}

fn desired_exposure(avg_log_lum: f32) -> f32 {{
    let avg_lum = max(exp2(avg_log_lum), LUMINANCE_FLOOR);
    let target_exposure = params.target_luminance * exp2(params.ev_compensation) / avg_lum;
    return clamp(target_exposure, params.min_exposure, params.max_exposure);
}}

@compute @workgroup_size(1)
fn adapt_exposure() {{
    var total: u32 = 0u;
    for (var i = 0u; i < HISTOGRAM_BINS; i = i + 1u) {{
        total = total + histogram[i];
    }}
    if (total == 0u) {{
        return;
    }}
    let inv_total = 1.0 / f32(total);
    let low_count = u32(f32(total) * params.low_clip);
    let upper = total - u32(f32(total) * params.high_clip);
    var acc: f32 = 0.0;
    var acc_weight: f32 = 0.0;
    var cum: u32 = 0u;
    for (var i = 0u; i < HISTOGRAM_BINS; i = i + 1u) {{
        let count = histogram[i];
        let cum_before = cum;
        cum = cum + count;
        if (cum > low_count && cum_before < upper) {{
            let weight = f32(count) * inv_total;
            acc = acc + weight * bin_center_log_lum(i);
            acc_weight = acc_weight + weight;
        }}
    }}
    if (acc_weight == 0.0) {{
        return;
    }}
    let avg_log_lum = acc / acc_weight;
    let target_exposure = desired_exposure(avg_log_lum);

    let prev = exposure_out.exposure;
    let rate = max(select(params.speed_down, params.speed_up, target_exposure > prev), MIN_ADAPTATION_SPEED);
    let alpha = 1.0 - exp(-params.dt / rate);
    let adapted = prev + (target_exposure - prev) * alpha;
    exposure_out.exposure = clamp(adapted, params.min_exposure, params.max_exposure);
}}
"#,
        bins = HISTOGRAM_BINS,
    )
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

/// Bind-group-layout entry for the histogram's HDR texture input.
///
/// The pass only `textureLoad`s, so the binding is declared non-filterable
/// and accepts any float view (the `Rgba16Float` HDR target included).
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
    /// The lit HDR target view to histogram.
    pub hdr: &'a wgpu::TextureView,
    /// HDR target width in texels.
    pub width: u32,
    /// HDR target height in texels.
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
        let histogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("auto-exposure histogram buffer"),
            size: (HISTOGRAM_BINS * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let histogram_params = init_buffer(
            device,
            "auto-exposure histogram params",
            bytemuck::bytes_of(&HistogramParams::from_settings(&default_settings)),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let adapt_params = init_buffer(
            device,
            "auto-exposure adapt params",
            bytemuck::bytes_of(&AdaptParams::from_settings(
                &default_settings,
                DEFAULT_FRAME_DT,
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

    /// Records one auto-exposure update onto `encoder`: zero the histogram,
    /// build it from `frame.hdr` (a non-filterable `texture_2d<f32>` view of
    /// `frame.width × frame.height`), then reduce it to the smoothed
    /// exposure. Two compute passes so the reduction sees a completed
    /// build.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: ExposureFrame<'_>,
    ) {
        let settings = frame.settings.sanitized();
        queue.write_buffer(
            &self.histogram_buffer,
            0,
            &vec![0_u8; HISTOGRAM_BINS * std::mem::size_of::<u32>()],
        );
        queue.write_buffer(
            &self.histogram_params,
            0,
            bytemuck::bytes_of(&HistogramParams::from_settings(&settings)),
        );
        queue.write_buffer(
            &self.adapt_params,
            0,
            bytemuck::bytes_of(&AdaptParams::from_settings(&settings, frame.dt_seconds)),
        );

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
                    resource: self.histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
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
            compute_buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
            compute_buffer_entry(2, wgpu::BufferBindingType::Uniform),
        ],
    );
    let adapt = compute_layout(
        device,
        "auto-exposure adapt layout",
        &[
            compute_buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Adapter handle, or `None` on headless hosts without a GPU (the skip is
    /// logged, never silent).
    fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let descriptor = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        };
        let instance = wgpu::Instance::new(descriptor);
        match pollster::block_on(request_device(instance)) {
            Some(pair) => Some(pair),
            None => {
                eprintln!("SKIPPED (auto_exposure): no wgpu device");
                None
            }
        }
    }

    /// Requests the default adapter/device from `instance`, logging the
    /// adapter error when the host has no GPU.
    async fn request_device(instance: wgpu::Instance) -> Option<(wgpu::Device, wgpu::Queue)> {
        let options = wgpu::RequestAdapterOptions::default();
        let adapter = match instance.request_adapter(&options).await {
            Ok(adapter) => adapter,
            Err(err) => {
                eprintln!("SKIPPED (auto_exposure): no wgpu adapter: {err}");
                return None;
            }
        };
        adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()
    }

    #[test]
    fn exposure_params_layout_is_a_16_byte_uniform_block() {
        assert_eq!(std::mem::size_of::<ExposureParams>(), 16);
        assert_eq!(std::mem::offset_of!(ExposureParams, exposure), 0);
        let packed = ExposureParams::pack(2.0);
        assert_eq!(packed.exposure, 2.0);
        assert_eq!(packed.pad, [0.0; 3]);
        assert_eq!(bytemuck::bytes_of(&packed).len(), 16);
        assert!(EXPOSURE_PARAMS_WGSL.contains("exposure: f32"));
        assert!(EXPOSURE_PARAMS_WGSL.contains(EXPOSURE_WGSL_NAME));
    }

    #[test]
    fn params_layouts_match_their_wgsl() {
        assert_eq!(std::mem::size_of::<HistogramParams>(), 16);
        assert_eq!(std::mem::size_of::<AdaptParams>(), 48);
        assert_eq!(std::mem::offset_of!(AdaptParams, min_log_lum), 0);
        assert_eq!(std::mem::offset_of!(AdaptParams, max_exposure), 40);
        let hist = histogram_wgsl();
        assert!(hist.contains("min_log_lum: f32"));
        assert!(hist.contains("array<atomic<u32>, 256u>"));
        let adapt = adapt_wgsl();
        assert!(adapt.contains("speed_up: f32"));
        assert!(adapt.contains("speed_down: f32"));
        assert!(adapt.contains("array<u32, 256u>"));
        assert!(adapt.contains("exposure_out.exposure"));
    }

    #[test]
    fn shaders_validate_with_naga() {
        for (name, source) in [("histogram", histogram_wgsl()), ("adapt", adapt_wgsl())] {
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("{name} compute shader must parse: {e}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} compute shader must validate: {e}"));
        }
    }

    #[test]
    fn manual_mode_defaults_to_the_pixel_identity_no_op() {
        assert_eq!(
            ExposureMode::default(),
            ExposureMode::Manual { multiplier: 1.0 }
        );
        assert_eq!(ExposureParams::pack(1.0).exposure, 1.0);
    }

    #[test]
    fn histogram_bins_saturate_the_dark_and_bright_tails() {
        let settings = AutoExposureSettings::default();
        // Black is far below the range: bin 0.
        assert_eq!(histogram_bin(log_luminance([0.0, 0.0, 0.0]), &settings), 0);
        // `2^20` is far above the range: the last bin.
        assert_eq!(
            histogram_bin(log_luminance([1.0e6, 1.0e6, 1.0e6]), &settings),
            HISTOGRAM_BINS - 1
        );
    }

    #[test]
    fn average_log_luminance_recovers_a_uniform_scene() {
        let settings = AutoExposureSettings::default();
        let lum = 0.5_f32;
        let texels = vec![[lum; 3]; 1024];
        let hist = build_histogram(&texels, &settings);
        let avg = average_log_luminance(&hist, &settings).expect("non-empty histogram");
        assert!(
            (avg - lum.log2()).abs() < 0.1,
            "avg={avg} expected≈{}",
            lum.log2()
        );
    }

    #[test]
    fn tail_clipping_ignores_a_bright_minority() {
        let settings = AutoExposureSettings::default();
        // A dark field with a few very bright fireflies: the clipped average
        // must not be dragged up by the outliers.
        let mut texels = vec![[0.02_f32; 3]; 1000];
        texels.extend(std::iter::repeat_n([1000.0_f32; 3], 10));
        let hist = build_histogram(&texels, &settings);
        let avg = average_log_luminance(&hist, &settings).expect("non-empty histogram");
        let fireflies = build_histogram(&texels[1000..], &settings);
        let bright_only = average_log_luminance(&fireflies, &settings).expect("bright histogram");
        assert!(
            avg < bright_only - 1.0,
            "clipped average {avg} should be far below the firefly average {bright_only}"
        );
        assert!((avg - 0.02_f32.log2()).abs() < 0.2, "avg={avg}");
    }

    #[test]
    fn desired_exposure_targets_the_key_luminance() {
        let settings = AutoExposureSettings::default();
        // A scene at the target luminance needs exposure 1.0.
        let exposure = desired_exposure(settings.target_luminance.log2(), &settings);
        assert!((exposure - 1.0).abs() < 1e-3, "{exposure}");
        // A ten-times darker scene needs a ten-times larger exposure.
        let darker = desired_exposure((settings.target_luminance / 10.0).log2(), &settings);
        assert!((darker - 10.0).abs() < 0.1, "{darker}");
    }

    #[test]
    fn desired_exposure_respects_the_ev_clamp() {
        let settings = AutoExposureSettings {
            min_ev: -2.0,
            max_ev: 2.0,
            ..Default::default()
        };
        // An almost black scene cannot push exposure past 2^2.
        assert_eq!(desired_exposure(-20.0, &settings), settings.max_exposure());
        // A blown-out scene cannot push it below 2^-2.
        assert_eq!(desired_exposure(20.0, &settings), settings.min_exposure());
    }

    #[test]
    fn adaptation_moves_toward_the_target_at_the_chosen_speed() {
        let settings = AutoExposureSettings::default();
        // Darker -> brighter uses `speed_up`; one step is strictly between.
        let step = adapt_exposure(1.0, 4.0, 1.0 / 60.0, &settings);
        assert!(step > 1.0 && step < 4.0, "{step}");
        // The two speeds must differ (down is the faster one by default).
        let up = adapt_exposure(1.0, 4.0, 0.1, &settings);
        let down = adapt_exposure(4.0, 1.0, 0.1, &settings);
        assert!(
            (down - 4.0).abs() > (up - 1.0).abs(),
            "down {down} should move further than up {up}"
        );
        // Repeated steps converge to the target.
        let mut exposure = 1.0;
        for _ in 0..600 {
            exposure = adapt_exposure(exposure, 4.0, 1.0 / 60.0, &settings);
        }
        assert!((exposure - 4.0).abs() < 0.05, "{exposure}");
    }

    /// Fills an `Rgba32Float` texture with a constant linear-RGB value.
    fn constant_hdr_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgb: [f32; 3],
    ) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("auto-exposure test hdr"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut data = Vec::with_capacity((width * height) as usize * 4);
        for _ in 0..(width * height) {
            data.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 1.0]);
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4 * 4),
                rows_per_image: Some(height),
            },
            size,
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Reads the exposure multiplier back from the shared uniform.
    fn read_exposure(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &AutoExposure,
    ) -> Option<f32> {
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("auto-exposure test readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("auto-exposure test readback encoder"),
        });
        encoder.copy_buffer_to_buffer(pass.exposure_buffer(), 0, &readback, 0, 16);
        queue.submit(std::iter::once(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let bytes = {
            let view = slice.get_mapped_range().ok()?;
            [view[0], view[1], view[2], view[3]]
        };
        readback.unmap();
        Some(f32::from_le_bytes(bytes))
    }

    /// Runs `frames` auto-exposure updates against a constant-luminance HDR
    /// target and returns the final exposure.
    fn converge(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        luminance: f32,
        settings: &AutoExposureSettings,
        frames: u32,
    ) -> f32 {
        const W: u32 = 32;
        const H: u32 = 32;
        let pass = AutoExposure::new(device);
        let view = constant_hdr_texture(device, queue, W, H, [luminance; 3]);
        let mut last = 1.0_f32;
        for _ in 0..frames {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("auto-exposure test encoder"),
            });
            pass.encode(
                device,
                queue,
                &mut encoder,
                ExposureFrame {
                    hdr: &view,
                    width: W,
                    height: H,
                    settings: *settings,
                    dt_seconds: 1.0 / 60.0,
                },
            );
            queue.submit(std::iter::once(encoder.finish()));
            last = read_exposure(device, queue, &pass).expect("exposure readback");
        }
        last
    }

    #[test]
    fn dark_and_bright_scenes_converge_to_the_target_luminance() {
        let Some((device, queue)) = try_device() else {
            return;
        };
        let settings = AutoExposureSettings::default();
        for luminance in [0.01_f32, 2.0_f32] {
            let exposure = converge(&device, &queue, luminance, &settings, 600);
            let expected = settings.target_luminance / luminance;
            let relative = (exposure - expected).abs() / expected;
            assert!(
                relative < 0.15,
                "luminance {luminance}: exposure {exposure} vs expected {expected}"
            );
        }
    }

    #[test]
    fn gpu_histogram_matches_the_cpu_reference() {
        let Some((device, queue)) = try_device() else {
            return;
        };
        // After one update the adapted exposure tracks the CPU reference
        // reduction of the same constant scene within a small tolerance.
        let settings = AutoExposureSettings::default();
        let luminance = 0.05_f32;
        let exposure = converge(&device, &queue, luminance, &settings, 1);
        let cpu = adapt_exposure(
            1.0,
            desired_exposure(luminance.log2(), &settings),
            1.0 / 60.0,
            &settings,
        );
        assert!((exposure - cpu).abs() < 1e-2, "gpu {exposure} vs cpu {cpu}");
    }
}
