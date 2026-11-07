//! WGSL sources of the auto-exposure compute passes (histogram + reduction).

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
/// The source is the checked-in llvm2gpu module. Bindings match the host
/// pass: `0` the histogram (`array<u32>`, read-write — cleared for the next
/// frame), `1` the adapt params uniform (twelve `f32`s, the last one pad),
/// `2` the exposure block (the `f32` at index 0). A single invocation
/// (`workgroup_size(1, 1, 1)`) excludes bin `0`, clips the tails with
/// partial bin weights, averages the log-luminance, converts to a target
/// EV, and advances the EV smoothing by one step. The entry point is
/// `cs_main`.
pub(crate) fn adapt_wgsl() -> String {
    include_str!("../shaders/adapt_generated.wgsl").to_string()
}
