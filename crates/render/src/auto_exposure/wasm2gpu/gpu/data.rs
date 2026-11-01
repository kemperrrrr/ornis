//! Guest ABI packing and fixed-width staging-buffer decoding.

use crate::auto_exposure::AutoExposureSettings;
use crate::auto_exposure::HISTOGRAM_BINS;

use super::EXPOSURE_READBACK_OFFSET;
use super::GuestOutput;

const IN_RANGE_BINS: f32 = HISTOGRAM_BINS as f32 - 2.0;

pub(super) fn histogram_params(settings: &AutoExposureSettings) -> Vec<f32> {
    let mut values = vec![0.0; 259];
    for (index, value) in values.iter_mut().take(HISTOGRAM_BINS + 1).enumerate() {
        *value = index as f32;
    }
    values[257] = settings.min_log_lum;
    values[258] = settings.inv_log_range();
    values
}

pub(super) fn adapt_params(settings: &AutoExposureSettings, dt: f32) -> Vec<f32> {
    let mut values = vec![0.0; 265];
    for (index, value) in values.iter_mut().take(HISTOGRAM_BINS).enumerate() {
        let t = ((index as f32 - 0.5) / IN_RANGE_BINS).clamp(0.0, 1.0);
        *value = settings.min_log_lum + t * settings.log_range();
    }
    values[256] = settings.low_percentile;
    values[257] = settings.high_percentile;
    values[258] = settings.target_ev();
    values[259] = settings.min_ev;
    values[260] = settings.max_ev;
    values[261] = settings.brighten_speed;
    values[262] = settings.darken_speed;
    values[263] = settings.exponential_transition_distance;
    values[264] = dt;
    values
}

pub(super) fn uniform_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0_u8; values.len() * 16];
    for (index, value) in values.iter().enumerate() {
        let offset = index * 16;
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub(super) fn output_from_bytes(bytes: &[u8]) -> GuestOutput {
    let mut histogram = [0_u32; HISTOGRAM_BINS];
    for (index, count) in histogram.iter_mut().enumerate() {
        let offset = index * 4;
        *count = u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32 bin"));
    }
    let exposure = f32::from_le_bytes(
        bytes[EXPOSURE_READBACK_OFFSET as usize..EXPOSURE_READBACK_OFFSET as usize + 4]
            .try_into()
            .expect("exposure lane"),
    );
    GuestOutput { histogram, exposure }
}
