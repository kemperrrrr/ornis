//! Guest ABI packing and fixed-width staging-buffer decoding.

use crate::auto_exposure::AutoExposureSettings;
use crate::auto_exposure::HISTOGRAM_BINS;

use super::EXPOSURE_READBACK_OFFSET;
use super::GuestOutput;

const IN_RANGE_BINS: f32 = HISTOGRAM_BINS as f32 - 2.0;
const BIN_CENTER_OFFSET: f32 = 0.5;
const BIN_EDGE_COUNT: usize = HISTOGRAM_BINS + 1;
pub(super) const HISTOGRAM_PARAMS_LEN: usize = HISTOGRAM_BINS + 3;
pub(super) const ADAPT_PARAMS_LEN: usize = HISTOGRAM_BINS + 9;
const HISTOGRAM_MIN_LOG_LUMINANCE_INDEX: usize = HISTOGRAM_BINS + 1;
const HISTOGRAM_INV_LOG_RANGE_INDEX: usize = HISTOGRAM_BINS + 2;
const ADAPT_LOW_CLIP_INDEX: usize = HISTOGRAM_BINS;
const ADAPT_HIGH_CLIP_INDEX: usize = HISTOGRAM_BINS + 1;
const ADAPT_TARGET_EV_INDEX: usize = HISTOGRAM_BINS + 2;
const ADAPT_MIN_EV_INDEX: usize = HISTOGRAM_BINS + 3;
const ADAPT_MAX_EV_INDEX: usize = HISTOGRAM_BINS + 4;
const ADAPT_BRIGHTEN_SPEED_INDEX: usize = HISTOGRAM_BINS + 5;
const ADAPT_DARKEN_SPEED_INDEX: usize = HISTOGRAM_BINS + 6;
const ADAPT_TRANSITION_DISTANCE_INDEX: usize = HISTOGRAM_BINS + 7;
const ADAPT_DT_INDEX: usize = HISTOGRAM_BINS + 8;
const UNIFORM_SLOT_BYTES: usize = 16;
const U32_BYTES: usize = std::mem::size_of::<u32>();
const F32_BYTES: usize = std::mem::size_of::<f32>();

pub(super) fn histogram_params(settings: &AutoExposureSettings) -> Vec<f32> {
    let mut values = vec![0.0; HISTOGRAM_PARAMS_LEN];
    for (index, value) in values.iter_mut().take(BIN_EDGE_COUNT).enumerate() {
        *value = index as f32;
    }
    values[HISTOGRAM_MIN_LOG_LUMINANCE_INDEX] = settings.min_log_lum;
    values[HISTOGRAM_INV_LOG_RANGE_INDEX] = settings.inv_log_range();
    values
}

pub(super) fn adapt_params(settings: &AutoExposureSettings, dt: f32) -> Vec<f32> {
    let mut values = vec![0.0; ADAPT_PARAMS_LEN];
    for (index, value) in values.iter_mut().take(HISTOGRAM_BINS).enumerate() {
        let t = ((index as f32 - BIN_CENTER_OFFSET) / IN_RANGE_BINS).clamp(0.0, 1.0);
        *value = settings.min_log_lum + t * settings.log_range();
    }
    values[ADAPT_LOW_CLIP_INDEX] = settings.low_percentile;
    values[ADAPT_HIGH_CLIP_INDEX] = settings.high_percentile;
    values[ADAPT_TARGET_EV_INDEX] = settings.target_ev();
    values[ADAPT_MIN_EV_INDEX] = settings.min_ev;
    values[ADAPT_MAX_EV_INDEX] = settings.max_ev;
    values[ADAPT_BRIGHTEN_SPEED_INDEX] = settings.brighten_speed;
    values[ADAPT_DARKEN_SPEED_INDEX] = settings.darken_speed;
    values[ADAPT_TRANSITION_DISTANCE_INDEX] = settings.exponential_transition_distance;
    values[ADAPT_DT_INDEX] = dt;
    values
}

pub(super) fn uniform_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0_u8; values.len() * UNIFORM_SLOT_BYTES];
    for (index, value) in values.iter().enumerate() {
        let offset = index * UNIFORM_SLOT_BYTES;
        bytes[offset..offset + F32_BYTES].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub(super) fn output_from_bytes(bytes: &[u8]) -> Option<GuestOutput> {
    let mut histogram = [0_u32; HISTOGRAM_BINS];
    for (index, count) in histogram.iter_mut().enumerate() {
        let offset = index * U32_BYTES;
        let lane = bytes.get(offset..offset + U32_BYTES)?;
        let lane = [lane[0], lane[1], lane[2], lane[3]];
        *count = u32::from_le_bytes(lane);
    }
    let exposure_offset = EXPOSURE_READBACK_OFFSET as usize;
    let exposure_lane = bytes.get(exposure_offset..exposure_offset + F32_BYTES)?;
    let exposure_lane = [
        exposure_lane[0],
        exposure_lane[1],
        exposure_lane[2],
        exposure_lane[3],
    ];
    let exposure = f32::from_le_bytes(exposure_lane);
    Some(GuestOutput {
        histogram,
        exposure,
    })
}
