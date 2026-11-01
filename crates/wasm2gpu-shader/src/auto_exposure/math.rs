//! Scalar HDR and exposure math shared by the wasm auto-exposure kernels.

#[cfg(any(target_arch = "wasm32", test))]
use super::LUMINANCE_FLOOR;

#[cfg(any(target_arch = "wasm32", test))]
use super::HISTOGRAM_PARAMS_BINDING;
#[cfg(any(target_arch = "wasm32", test))]
use super::HISTOGRAM_PARAMS_LEN;

#[cfg(any(target_arch = "wasm32", test))]
use ornis_gpu_abi::uniform_array_f32;

#[cfg(any(target_arch = "wasm32", test))]
const IN_RANGE_BINS: f32 = super::HISTOGRAM_BINS as f32 - 2.0;
#[cfg(any(target_arch = "wasm32", test))]
const MAX_FINITE_LUMINANCE: f32 = 3.4e38;
const LN_2: f32 = core::f32::consts::LN_2;
const HALF_F32: f32 = 0.5;
const MAX_EXP2_INPUT: f32 = 127.0;
const MIN_EXP2_INPUT: f32 = -149.0;
const LOG2_FRACTION_STOP: f32 = 5.960_464_5e-8;
const EXP_SERIES_TERMS: f32 = 13.0;
#[cfg(any(target_arch = "wasm32", test))]
const REC709_RED_WEIGHT: f32 = 0.2126;
#[cfg(any(target_arch = "wasm32", test))]
const REC709_GREEN_WEIGHT: f32 = 0.7152;
#[cfg(any(target_arch = "wasm32", test))]
const REC709_BLUE_WEIGHT: f32 = 0.0722;

#[cfg(any(target_arch = "wasm32", test))]
pub(super) fn selected_channel(mode: u32, deferred: f32, forward: f32, alpha: f32) -> f32 {
    if mode == 1 {
        forward * alpha
    } else if mode == 2 {
        deferred + forward * alpha
    } else {
        deferred
    }
}

#[cfg(any(target_arch = "wasm32", test))]
pub(super) fn log_luminance(red: f32, green: f32, blue: f32) -> f32 {
    let mut luminance =
        red * REC709_RED_WEIGHT + green * REC709_GREEN_WEIGHT + blue * REC709_BLUE_WEIGHT;
    let finite_range = (-MAX_FINITE_LUMINANCE..=MAX_FINITE_LUMINANCE).contains(&luminance);
    if !finite_range || luminance < LUMINANCE_FLOOR {
        luminance = LUMINANCE_FLOOR;
    }
    log2_approx(luminance)
}

#[cfg(any(target_arch = "wasm32", test))]
pub(super) fn log_luminance_is_in_bin(
    log_lum: f32,
    bin: u32,
    min_log_lum: f32,
    inv_log_range: f32,
) -> bool {
    if bin == 0 {
        return log_lum < min_log_lum;
    }
    if log_lum < min_log_lum {
        return false;
    }
    let lower = uniform_array_f32::<0, HISTOGRAM_PARAMS_BINDING, HISTOGRAM_PARAMS_LEN>(bin);
    let upper = uniform_array_f32::<0, HISTOGRAM_PARAMS_BINDING, HISTOGRAM_PARAMS_LEN>(bin + 1);
    let t = clamp_f32((log_lum - min_log_lum) * inv_log_range, 0.0, 1.0);
    let scaled = t * IN_RANGE_BINS + 1.0;
    lower <= scaled && scaled < upper
}

#[cfg(any(target_arch = "wasm32", test))]
pub(super) fn clamp_f32(value: f32, low: f32, high: f32) -> f32 {
    if value < low {
        low
    } else if high < value {
        high
    } else {
        value
    }
}

/// Binary range-reduced log2; uses only scalar operations supported by the lifter.
pub(super) fn log2_approx(value: f32) -> f32 {
    let mut mantissa = value;
    let mut exponent = 0.0_f32;
    while mantissa < 1.0 {
        mantissa *= 2.0;
        exponent -= 1.0;
    }
    while 2.0 < mantissa {
        mantissa *= HALF_F32;
        exponent += 1.0;
    }

    let mut fraction = 0.0_f32;
    let mut bit = HALF_F32;
    while LOG2_FRACTION_STOP < bit {
        mantissa *= mantissa;
        if 2.0 < mantissa {
            mantissa *= HALF_F32;
            fraction += bit;
        }
        bit *= HALF_F32;
    }
    exponent + fraction
}

/// Range-reduced `2^x` with a short exponential series for the fractional part.
pub(super) fn exp2_approx(value: f32) -> f32 {
    if MAX_EXP2_INPUT < value {
        return f32::INFINITY;
    }
    if value < MIN_EXP2_INPUT {
        return 0.0;
    }
    let mut fraction = value;
    let mut scale = 1.0_f32;
    while 1.0 < fraction {
        scale *= 2.0;
        fraction -= 1.0;
    }
    while fraction < 0.0 {
        scale *= HALF_F32;
        fraction += 1.0;
    }

    let x = fraction * LN_2;
    let mut term = 1.0_f32;
    let mut sum = 1.0_f32;
    let mut divisor = 1.0_f32;
    while divisor < EXP_SERIES_TERMS {
        term = term * x / divisor;
        sum += term;
        divisor += 1.0;
    }
    scale * sum
}

#[cfg(test)]
mod tests {
    use super::exp2_approx;
    use super::log2_approx;

    #[test]
    fn range_reduced_log2_matches_reference_across_hdr_window() {
        for value in [
            1.0e-6_f32,
            2.0_f32.powi(-8),
            0.18,
            0.5,
            1.0,
            3.0,
            256.0,
            1.0e6,
        ] {
            let error = (log2_approx(value) - value.log2()).abs();
            assert!(error < 2.0e-6, "log2({value}) error {error}");
        }
    }

    #[test]
    fn range_reduced_exp2_matches_reference_in_exposure_window() {
        for value in [-8.0_f32, -3.25, -0.5, 0.0, 0.5, 3.25, 8.0] {
            let expected = 2.0_f32.powf(value);
            let actual = exp2_approx(value);
            let relative_error = (actual - expected).abs() / expected;
            assert!(
                relative_error < 2.0e-6,
                "exp2({value}) error {relative_error}"
            );
        }
        assert!(exp2_approx(128.0).is_infinite());
        assert_eq!(exp2_approx(-150.0), 0.0);
    }
}
