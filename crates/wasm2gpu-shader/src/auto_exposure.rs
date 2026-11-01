//! Scalarized auto-exposure guest: HDR log-luminance, histogram, and EV adaptation.
//!
//! The three compute exports preserve the production pass results without
//! workgroup synchronization: one lane writes each pixel's log luminance,
//! one lane owns each histogram bin and scans that scratch array, and one
//! lane reduces the histogram. HDR texels arrive in a storage-buffer mirror;
//! this keeps the experiment inside the current `wasm2gpu` ABI, which has no
//! `textureLoad`, atomics, workgroup memory, or barriers.

mod adapt;
mod histogram;
mod log_luma;
mod math;

#[cfg(test)]
mod tests;

/// Number of production histogram bins; bin zero is the below-window bucket.
const HISTOGRAM_BINS: u32 = 256;
/// Eight packed RGBA lanes per texel: deferred then forward.
const TEXEL_LANES: u32 = 8;
/// Storage binding with the packed deferred/forward HDR texels.
const INPUT_BINDING: u32 = 0;
/// Storage binding with one log-luminance value per texel.
const LOG_LUMINANCE_BINDING: u32 = 1;
/// Storage binding with the 256 production-style `u32` histogram counts.
const HISTOGRAM_BINDING: u32 = 2;
/// Storage binding with the previous and adapted exposure multipliers.
const EXPOSURE_BINDING: u32 = 6;
/// Storage binding holding pixel count and composite layer mode.
const FRAME_BINDING: u32 = 3;
/// Uniform binding with bin-edge values, minimum log luminance, and inverse range.
const HISTOGRAM_PARAMS_BINDING: u32 = 4;
/// Uniform binding with bin centres and adaptation settings.
const ADAPT_PARAMS_BINDING: u32 = 5;
/// Histogram parameters contain edge values 0..=256, then minimum and inverse range.
const HISTOGRAM_PARAMS_LEN: u32 = 259;
/// Adaptation parameters contain 256 centres followed by nine scalar settings.
const ADAPT_PARAMS_LEN: u32 = 265;
const MIN_LOG_LUMINANCE_INDEX: u32 = 257;
const INV_LOG_RANGE_INDEX: u32 = 258;
const ADAPT_LOW_CLIP_INDEX: u32 = 256;
const ADAPT_HIGH_CLIP_INDEX: u32 = 257;
const ADAPT_TARGET_EV_INDEX: u32 = 258;
const ADAPT_MIN_EV_INDEX: u32 = 259;
const ADAPT_MAX_EV_INDEX: u32 = 260;
const ADAPT_BRIGHTEN_SPEED_INDEX: u32 = 261;
const ADAPT_DARKEN_SPEED_INDEX: u32 = 262;
const ADAPT_TRANSITION_DISTANCE_INDEX: u32 = 263;
const ADAPT_DT_INDEX: u32 = 264;
const LUMINANCE_FLOOR: f32 = 1e-6;
