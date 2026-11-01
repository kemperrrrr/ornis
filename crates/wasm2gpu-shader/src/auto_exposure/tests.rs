//! Host-side execution tests for the guest kernels through the GPU ABI shadow.

use crate::UniformF32Lane;
use ornis_gpu_abi::HostMemory;
use ornis_gpu_abi::with_host;

use super::ADAPT_BRIGHTEN_SPEED_INDEX;
use super::ADAPT_DARKEN_SPEED_INDEX;
use super::ADAPT_DT_INDEX;
use super::ADAPT_HIGH_CLIP_INDEX;
use super::ADAPT_LOW_CLIP_INDEX;
use super::ADAPT_MAX_EV_INDEX;
use super::ADAPT_MIN_EV_INDEX;
use super::ADAPT_PARAMS_BINDING;
use super::ADAPT_PARAMS_LEN;
use super::ADAPT_TARGET_EV_INDEX;
use super::ADAPT_TRANSITION_DISTANCE_INDEX;
use super::EXPOSURE_BINDING;
use super::FRAME_BINDING;
use super::HISTOGRAM_BINDING;
use super::HISTOGRAM_BINS;
use super::HISTOGRAM_PARAMS_BINDING;
use super::HISTOGRAM_PARAMS_LEN;
use super::HISTOGRAM_MIN_LOG_LUMINANCE_INDEX;
use super::HISTOGRAM_INV_LOG_RANGE_INDEX;
use super::INPUT_BINDING;
use super::LOG_LUMINANCE_BINDING;
use super::adapt::ae_adapt;
use super::histogram::ae_histogram;
use super::log_luma::ae_log_luma;
use super::math::log_luminance;
use super::math::selected_channel;

const INPUT_LANES: usize = 8;
const FRAME_WORDS: usize = 2;
const EXPOSURE_SLOTS: usize = 2;
const HISTOGRAM_WORDS: usize = HISTOGRAM_BINS as usize;
const HISTOGRAM_PARAMETER_WORDS: usize = HISTOGRAM_PARAMS_LEN as usize;
const ADAPT_PARAMETER_WORDS: usize = ADAPT_PARAMS_LEN as usize;
const PIXEL_COUNT: usize = 6;

fn uniform_lanes<const N: usize>(values: &[f32; N]) -> [UniformF32Lane; N] {
    std::array::from_fn(|index| UniformF32Lane {
        value: values[index],
        pad0: 0,
        pad1: 0,
        pad2: 0,
    })
}

fn log_luma_for_mode(mode: u32) -> f32 {
    let packed = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5];
    let frame = [1, mode];
    let mut memory = HostMemory::new();
    memory.bind_storage::<0, INPUT_BINDING, [f32; INPUT_LANES]>();
    memory.write_storage::<0, INPUT_BINDING, [f32; INPUT_LANES]>(&packed);
    memory.bind_storage::<0, FRAME_BINDING, [u32; FRAME_WORDS]>();
    memory.write_storage::<0, FRAME_BINDING, [u32; FRAME_WORDS]>(&frame);
    memory.bind_storage::<0, LOG_LUMINANCE_BINDING, [f32; 1]>();
    memory.write_storage::<0, LOG_LUMINANCE_BINDING, [f32; 1]>(&[0.0]);
    with_host(&mut memory, || ae_log_luma(0));
    memory.read_storage::<0, LOG_LUMINANCE_BINDING, [f32; 1]>()[0]
}

#[test]
fn log_luma_kernel_selects_the_expected_composite_layer() {
    for mode in [0, 1, 2] {
        let deferred_red = 1.0;
        let deferred_green = 0.0;
        let deferred_blue = 0.0;
        let alpha = 0.5;
        let red = selected_channel(mode, deferred_red, 0.0, alpha);
        let green = selected_channel(mode, deferred_green, 1.0, alpha);
        let blue = selected_channel(mode, deferred_blue, 0.0, alpha);
        let expected = log_luminance(red, green, blue);
        let actual = log_luma_for_mode(mode);
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "mode {mode}: expected {expected}, got {actual}"
        );
    }
}

fn histogram_params() -> [f32; HISTOGRAM_PARAMETER_WORDS] {
    let mut values = [0.0; HISTOGRAM_PARAMETER_WORDS];
    for (index, value) in values.iter_mut().take(HISTOGRAM_WORDS + 1).enumerate() {
        *value = index as f32;
    }
    values[HISTOGRAM_MIN_LOG_LUMINANCE_INDEX as usize] = -7.0;
    values[HISTOGRAM_INV_LOG_RANGE_INDEX as usize] = 1.0 / 14.0;
    values
}

fn run_histogram(log_values: &[f32; PIXEL_COUNT]) -> [u32; HISTOGRAM_WORDS] {
    let frame = [PIXEL_COUNT as u32, 0];
    let mut memory = HostMemory::new();
    memory.bind_storage::<0, FRAME_BINDING, [u32; FRAME_WORDS]>();
    memory.write_storage::<0, FRAME_BINDING, [u32; FRAME_WORDS]>(&frame);
    memory.bind_storage::<0, LOG_LUMINANCE_BINDING, [f32; PIXEL_COUNT]>();
    memory.write_storage::<0, LOG_LUMINANCE_BINDING, [f32; PIXEL_COUNT]>(log_values);
    memory.bind_storage::<0, HISTOGRAM_BINDING, [u32; HISTOGRAM_WORDS]>();
    memory.write_storage::<0, HISTOGRAM_BINDING, [u32; HISTOGRAM_WORDS]>(&[0; HISTOGRAM_WORDS]);
    let values = histogram_params();
    let lanes = uniform_lanes(&values);
    memory.bind_uniform::<
        0,
        HISTOGRAM_PARAMS_BINDING,
        [UniformF32Lane; HISTOGRAM_PARAMETER_WORDS],
    >();
    memory.write_uniform::<
        0,
        HISTOGRAM_PARAMS_BINDING,
        [UniformF32Lane; HISTOGRAM_PARAMETER_WORDS],
    >(&lanes);
    with_host(&mut memory, || {
        for bin in 0..HISTOGRAM_BINS {
            ae_histogram(bin);
        }
    });
    memory.read_storage::<0, HISTOGRAM_BINDING, [u32; HISTOGRAM_WORDS]>()
}

#[test]
fn histogram_kernel_matches_the_clamped_interval_mapping() {
    let log_values = [-8.0, -7.0, -6.0, 0.0, 7.0, 8.0];
    let actual = run_histogram(&log_values);
    let mut expected = [0_u32; HISTOGRAM_WORDS];
    for log_luminance in log_values {
        let bin = if log_luminance < -7.0 {
            0
        } else {
            let t = ((log_luminance + 7.0) / 14.0).clamp(0.0, 1.0);
            (t * (HISTOGRAM_BINS as f32 - 2.0) + 1.0) as usize
        };
        expected[bin.min(HISTOGRAM_WORDS - 1)] += 1;
    }
    assert_eq!(actual, expected);
}

fn adapt_params() -> [f32; ADAPT_PARAMETER_WORDS] {
    let mut values = [0.0; ADAPT_PARAMETER_WORDS];
    values[ADAPT_LOW_CLIP_INDEX as usize] = 0.0;
    values[ADAPT_HIGH_CLIP_INDEX as usize] = 1.0;
    values[ADAPT_TARGET_EV_INDEX as usize] = 0.0;
    values[ADAPT_MIN_EV_INDEX as usize] = -2.0;
    values[ADAPT_MAX_EV_INDEX as usize] = 2.0;
    values[ADAPT_BRIGHTEN_SPEED_INDEX as usize] = 1.0;
    values[ADAPT_DARKEN_SPEED_INDEX as usize] = 1.0;
    values[ADAPT_TRANSITION_DISTANCE_INDEX as usize] = 1.0;
    values[ADAPT_DT_INDEX as usize] = 0.5;
    values
}

fn run_adaptation(
    histogram: &[u32; HISTOGRAM_WORDS],
    exposure: &[f32; EXPOSURE_SLOTS],
) -> ([u32; HISTOGRAM_WORDS], [f32; EXPOSURE_SLOTS]) {
    let mut memory = HostMemory::new();
    memory.bind_storage::<0, HISTOGRAM_BINDING, [u32; HISTOGRAM_WORDS]>();
    memory.write_storage::<0, HISTOGRAM_BINDING, [u32; HISTOGRAM_WORDS]>(histogram);
    memory.bind_storage::<0, EXPOSURE_BINDING, [f32; EXPOSURE_SLOTS]>();
    memory.write_storage::<0, EXPOSURE_BINDING, [f32; EXPOSURE_SLOTS]>(exposure);
    let values = adapt_params();
    let lanes = uniform_lanes(&values);
    memory.bind_uniform::<
        0,
        ADAPT_PARAMS_BINDING,
        [UniformF32Lane; ADAPT_PARAMETER_WORDS],
    >();
    memory.write_uniform::<0, ADAPT_PARAMS_BINDING, [UniformF32Lane; ADAPT_PARAMETER_WORDS]>(
        &lanes,
    );
    with_host(&mut memory, || {
        ae_adapt(1);
        ae_adapt(0);
    });
    (
        memory.read_storage::<0, HISTOGRAM_BINDING, [u32; HISTOGRAM_WORDS]>(),
        memory.read_storage::<0, EXPOSURE_BINDING, [f32; EXPOSURE_SLOTS]>(),
    )
}

#[test]
fn adaptation_kernel_smooths_exposure_in_ev_and_clears_histogram() {
    let mut histogram = [0_u32; HISTOGRAM_WORDS];
    histogram[128] = 1;
    let (cleared, exposure) = run_adaptation(&histogram, &[0.5, 0.0]);
    assert_eq!(cleared, [0_u32; HISTOGRAM_WORDS]);
    assert!((exposure[0] - 0.5_f32.sqrt()).abs() < 2.0e-6);
    assert_eq!(exposure[0], exposure[1]);

    let (empty_histogram, held_exposure) = run_adaptation(&[0; HISTOGRAM_WORDS], &[0.75, 0.0]);
    assert_eq!(empty_histogram, [0_u32; HISTOGRAM_WORDS]);
    assert_eq!(held_exposure, [0.75, 0.75]);
}
