//! Edge mask for deferred 4x lighting.
//!
//! A cleared g-buffer sample stores octahedral `(0, 0)`, which decodes to
//! +Z. Box-filtering that texel into a silhouette pulls the shaded normal
//! off the surface. [`shade_mask`] drops cleared samples; an interior pixel
//! whose samples agree shades sample 0 only, matching the 1x path.

use glam::Vec3;

use crate::renderer::MSAA_SAMPLE_COUNT;
use crate::shaders::helpers::{CLEAR_DEPTH, DEPTH_EDGE, NORMAL_AGREE};

/// One g-buffer sample the deferred lighting pass may shade.
#[derive(Clone, Copy, Debug)]
pub struct DeferredSample {
    /// DirectX NDC depth. [`CLEAR_DEPTH`] is background.
    pub depth: f32,
    /// Material table index.
    pub material: u32,
    /// Decoded world normal.
    pub normal: Vec3,
}

/// Which samples contribute to one pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleCoverage {
    covered: [bool; MSAA_SAMPLE_COUNT as usize],
}

impl SampleCoverage {
    /// Whether lighting includes sample `index` (`0..4`).
    pub fn shades(self, index: u32) -> bool {
        self.covered[index as usize]
    }
}

/// Samples the MSAA lighting entry evaluates.
///
/// Mixed coverage or a disagreement (material, depth, or normal) shades
/// every covered sample. Agreement, including a fully cleared pixel, shades
/// sample 0 only.
pub fn shade_mask(samples: [DeferredSample; MSAA_SAMPLE_COUNT as usize]) -> SampleCoverage {
    let mut covered = [false; MSAA_SAMPLE_COUNT as usize];
    let mut count = 0u32;
    let mut reference = 0usize;
    let mut have_ref = false;
    let mut disagree = false;
    let mut index = 0usize;
    while index < MSAA_SAMPLE_COUNT as usize {
        let sample = samples[index];
        let hit = sample.depth < CLEAR_DEPTH;
        covered[index] = hit;
        if hit {
            count += 1;
            if !have_ref {
                reference = index;
                have_ref = true;
            } else {
                let prior = samples[reference];
                let dot = prior.normal.x * sample.normal.x
                    + prior.normal.y * sample.normal.y
                    + prior.normal.z * sample.normal.z;
                let gap = sample.depth - prior.depth;
                let depth_gap = if gap < 0.0 { 0.0 - gap } else { gap };
                if sample.material != prior.material || depth_gap > DEPTH_EDGE || dot < NORMAL_AGREE
                {
                    disagree = true;
                }
            }
        }
        index += 1;
    }
    let edge = (count > 0 && count < MSAA_SAMPLE_COUNT) || disagree;
    if !edge {
        covered = [false; MSAA_SAMPLE_COUNT as usize];
        covered[0] = true;
    }
    SampleCoverage { covered }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaders::math::octahedral_encode;
    use crate::shaders::octahedral_decode_rust;

    #[test]
    fn clear_octahedral_is_plus_z_and_is_dropped_on_an_edge() {
        let clear_n = octahedral_decode_rust(glam::Vec2::ZERO);
        assert!(clear_n.z > 0.9, "clear decode {clear_n:?}");
        assert!(clear_n.x.abs() < 0.05 && clear_n.y.abs() < 0.05);
        let surface = DeferredSample {
            depth: 0.4,
            material: 1,
            normal: Vec3::X,
        };
        let clear = DeferredSample {
            depth: CLEAR_DEPTH,
            material: 0,
            normal: clear_n,
        };
        // Sample 0 is the clear the old loader would shade; sample 2 hits.
        let mask = shade_mask([clear, clear, surface, clear]);
        assert!(!mask.shades(0), "sample 0 is background");
        assert!(mask.shades(2), "the covered sample must be shaded");
        assert!(!mask.shades(1) && !mask.shades(3));
        // Hardware resolve of one +X encoding and three clears is not +X.
        let encoded = octahedral_encode::eval(Vec3::X);
        let filtered = octahedral_decode_rust(encoded * 0.25);
        assert!(
            filtered.x < 0.5,
            "box-filtered normal {filtered:?} must leave +X"
        );
    }

    #[test]
    fn agreeing_samples_shade_only_sample_zero() {
        let sample = DeferredSample {
            depth: 0.3,
            material: 2,
            normal: Vec3::Y,
        };
        let mask = shade_mask([sample, sample, sample, sample]);
        assert!(mask.shades(0));
        assert!(!mask.shades(1) && !mask.shades(2) && !mask.shades(3));
    }
}
