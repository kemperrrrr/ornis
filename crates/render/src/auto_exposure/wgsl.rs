//! WGSL sources of the auto-exposure compute passes (histogram + reduction).

use super::HISTOGRAM_BINS;

/// WGSL compute pass building the log-luminance histogram of the HDR mix.
///
/// The source is the checked-in llvm2gpu module. Bindings match the host
/// pass: `0` the deferred HDR texture, `1` the forward HDR texture (both
/// `texture_2d<f32>`, sampled by load), `2` the atomic histogram, `3` the
/// histogram params uniform (`min_log_lum`, `inv_log_range`, `bin_count`,
/// `mode`). The selected layer mix matches the composite tone-mapper, so
/// the histogram meters the same HDR that is tone-mapped. Each 8×8
/// workgroup accumulates into a shared histogram and flushes one
/// `atomicAdd` per occupied bin. Bin `0` collects texels below the window.
///
/// The shader guards the ragged tail against `textureDimensions`. The
/// entry point is `cs_main`.
pub(crate) fn histogram_wgsl() -> String {
    include_str!("../shaders/histogram_generated.wgsl").to_string()
}

/// WGSL compute pass reducing the histogram to the adapted exposure.
///
/// Bindings: `0` the histogram (`array<u32, HISTOGRAM_BINS>`, read-write —
/// cleared for the next frame), `1` the adapt params uniform, `2` the
/// [`ExposureParams`](super::ExposureParams) block (read-write — the smoothed
/// exposure is read back as the composite uniform). A single invocation
/// (`workgroup_size(1)`, one workgroup) excludes bin `0`, clips the tails
/// with partial bin weights, averages the log-luminance, converts to a
/// target EV, and advances the EV smoothing by one step.
pub(crate) fn adapt_wgsl() -> String {
    format!(
        r#"//! GPU eye adaptation (auto-exposure, ornis #125).
const HISTOGRAM_BINS: u32 = {bins}u;
const LUMINANCE_FLOOR: f32 = 1e-6;
const MIN_ADAPTATION_SPEED: f32 = 1e-3;
const MIN_TRANSITION_DISTANCE: f32 = 1e-3;

struct AdaptParams {{
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
    pad: f32,
}};

struct ExposureParams {{
    exposure: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}};

@group(0) @binding(0) var<storage, read_write> histogram: array<u32, {bins}u>;
@group(0) @binding(1) var<uniform> params: AdaptParams;
@group(0) @binding(2) var<storage, read_write> exposure_out: ExposureParams;

var<workgroup> counts: array<u32, {bins}u>;

fn bin_log_luminance(index: u32) -> f32 {{
    let t = clamp((f32(index) - 0.5) / (f32(HISTOGRAM_BINS) - 2.0), 0.0, 1.0);
    return params.min_log_lum + t * params.log_range;
}}

@compute @workgroup_size(1)
fn adapt_exposure() {{
    // Snapshot the counts and clear the histogram (including bin 0) for the
    // next frame; the clear happens even when there is nothing to meter.
    var total: u32 = 0u;
    for (var i = 0u; i < HISTOGRAM_BINS; i = i + 1u) {{
        let count = histogram[i];
        counts[i] = count;
        histogram[i] = 0u;
        if (i > 0u) {{
            total = total + count;
        }}
    }}
    if (total == 0u) {{
        return;
    }}

    let low = f32(total) * params.low_clip;
    let high = f32(total) * params.high_clip;
    var cumulative: u32 = 0u;
    var sum: f32 = 0.0;
    var sum_weight: f32 = 0.0;
    for (var i = 1u; i < HISTOGRAM_BINS; i = i + 1u) {{
        let cum_before = cumulative;
        cumulative = cumulative + counts[i];
        let weight = clamp(f32(cumulative), low, high) - clamp(f32(cum_before), low, high);
        sum = sum + weight * bin_log_luminance(i);
        sum_weight = sum_weight + weight;
    }}
    if (sum_weight <= 0.0) {{
        return;
    }}
    let avg_log_lum = sum / sum_weight;

    let prev_ev = clamp(log2(max(exposure_out.exposure, LUMINANCE_FLOOR)), params.min_ev, params.max_ev);
    let target_ev = clamp(params.target_ev - avg_log_lum, params.min_ev, params.max_ev);
    let delta = target_ev - prev_ev;
    var adapted = prev_ev;
    if (delta != 0.0) {{
        let speed = select(params.darken_speed, params.brighten_speed, delta > 0.0);
        let max_step = speed * params.dt;
        let step = min(max_step, abs(delta) * max_step / max(params.transition_distance, MIN_TRANSITION_DISTANCE));
        // Never overshoot the target: a large `dt` lands exactly on it.
        let clamped = min(step, abs(delta));
        adapted = prev_ev + sign(delta) * clamped;
    }}
    // `adapted` is between the clamped `prev_ev` and `target_ev` (the step
    // is clamped to `|delta|`), so it is already inside the EV window and
    // needs no final clamp.
    exposure_out.exposure = exp2(adapted);
}}
"#,
        bins = HISTOGRAM_BINS,
    )
}
