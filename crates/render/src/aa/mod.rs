//! Opt-in spatial/temporal AA resolve, with persistent ping-pong history.
//!
//! Run on single-sampled, linear color before tone mapping/UI. The caller
//! supplies motion/depth and applies [`crate::aa::jitter_projection`] to geometry only.
//! Temporal state belongs to a camera, never to the transient texture pool.
use glam::{Mat4, Vec2, Vec3, Vec4};
mod gpu;
pub use gpu::{AaInputs, AaResolve};

const HALTON_PERIOD: u64 = 8;
const HALTON_Y_BASE: u32 = 3;
const HALF_PIXEL: f32 = 0.5;
const PIXEL_TO_CLIP_SCALE: f32 = 2.0;

#[cfg(not(target_arch = "wasm32"))]
mod graph;
#[cfg(not(target_arch = "wasm32"))]
pub use graph::{AaColor, AaMotion, AaPass};

/// Selects geometry sampling and the screen-space resolve independently.
#[repr(u32)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AntiAliasing {
    /// Single-sample rendering without post AA.
    #[default]
    Off,
    /// Geometry-only four-sample MSAA (adapter negotiation still required).
    Msaa,
    /// Single-pass fast approximate AA, suitable for WebGPU.
    Fxaa,
    /// Three-pass morphological AA with analytic area weights.
    ///
    /// This lookup-free SMAA variant omits diagonal/corner refinement.
    Smaa,
    /// Temporal reprojection, depth rejection and neighbourhood clamp.
    Taa,
}

impl AntiAliasing {
    /// Requested geometry sample count; negotiate against the adapter.
    // qual:api — geometry configuration helper for external callers.
    pub const fn sample_count(self) -> u32 {
        match self {
            Self::Msaa => crate::renderer::MSAA_SAMPLE_COUNT,
            _ => 1,
        }
    }

    fn shader_mode(self) -> u32 {
        self as u32
    }
}

fn halton(mut index: u32, base: u32) -> f32 {
    let mut value = 0.0;
    let mut fraction = 1.0;
    while index != 0 {
        fraction /= base as f32;
        value += fraction * (index % base) as f32;
        index /= base;
    }
    value
}

/// Eight-frame Halton(2,3) sequence, in pixel units centred around zero.
/// Frame zero is the first nonzero Halton sample, not the corner (-0.5,-0.5).
pub fn temporal_jitter(frame: u64) -> Vec2 {
    let index = (frame % HALTON_PERIOD) as u32 + 1;
    Vec2::new(halton(index, 2), halton(index, HALTON_Y_BASE)) - Vec2::splat(HALF_PIXEL)
}

/// Offsets clip x/y by clip w; works for perspective and orthographic cameras.
/// Returns the original matrix for an empty viewport. UV y points down.
// qual:api — camera integration applies the resolve jitter here.
pub fn jitter_projection(view_projection: Mat4, jitter: Vec2, extent: [u32; 2]) -> Mat4 {
    if extent.contains(&0) {
        return view_projection;
    }
    let offset = Vec2::new(
        PIXEL_TO_CLIP_SCALE * jitter.x / extent[0] as f32,
        -PIXEL_TO_CLIP_SCALE * jitter.y / extent[1] as f32,
    );
    let mut shift = Mat4::IDENTITY;
    shift.w_axis.x = offset.x;
    shift.w_axis.y = offset.y;
    shift * view_projection
}

/// Builds previous-UV minus current-UV motion from *jittered* clip positions.
///
/// Use the previous skinned world position for `previous_clip`, not today's
/// position transformed by yesterday's camera. Returns `None` behind either
/// camera or for non-finite input; mark those pixels' previous depth negative.
// qual:api — velocity producers use the same UV convention.
pub fn motion_vector(current_clip: Vec4, previous_clip: Vec4) -> Option<Vec2> {
    if !current_clip.is_finite()
        || !previous_clip.is_finite()
        || current_clip.w <= 0.0
        || previous_clip.w <= 0.0
    {
        return None;
    }
    let delta = previous_clip.truncate().truncate() / previous_clip.w
        - current_clip.truncate().truncate() / current_clip.w;
    Some(delta * Vec2::new(HALF_PIXEL, -HALF_PIXEL))
}

/// CPU reference for neighbourhood clipping and temporal accumulation.
/// `feedback` is saturated; a rejected history is exactly the current sample.
// qual:api — reference for external temporal golden/ghosting tests.
pub fn temporal_blend(
    current: Vec3,
    history: Vec3,
    neighbourhood: &[Vec3],
    feedback: f32,
    accepted: bool,
) -> Vec3 {
    if !accepted || !history.is_finite() || !feedback.is_finite() {
        return current;
    }
    let (lo, hi) = neighbourhood
        .iter()
        .filter(|sample| sample.is_finite())
        .fold((current, current), |(lo, hi), sample| {
            (lo.min(*sample), hi.max(*sample))
        });
    current.lerp(history.clamp(lo, hi), feedback.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests;
