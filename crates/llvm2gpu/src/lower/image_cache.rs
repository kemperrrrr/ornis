//! Deduplication of image queries: one expression per distinct request.
//!
//! Shaders read a texel one lane at a time: `sample_lane::<0>` through
//! `sample_lane::<3>` are four imports of the same sample, and each import
//! used to lower to its own `textureSample` (each axis its own
//! `textureDimensions`), so the GPU evaluated the same fetch four times. The
//! subset has no `textureStore`, image queries are pure, and a second request
//! with equal operands is the same expression: the cache returns the first
//! handle and the lane is an `AccessIndex` on it.

use std::collections::HashMap;

/// What makes two image queries the same expression.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum ImageRequest {
    /// An image or sampler global reference. Caching it keeps the operand
    /// handles of the requests below equal across call sites.
    Global(naga::Handle<naga::GlobalVariable>),
    /// `textureLoad` of one texel.
    Load {
        image: naga::Handle<naga::Expression>,
        coordinate: naga::Handle<naga::Expression>,
        array_index: Option<naga::Handle<naga::Expression>>,
        sample: Option<naga::Handle<naga::Expression>>,
        level: Option<naga::Handle<naga::Expression>>,
    },
    /// A composed coordinate vector. Sampling the same texel in four lanes
    /// builds the same `vec2`/`vec3` four times, and the sample could only
    /// key on it if the composition were shared.
    Compose {
        ty: naga::Handle<naga::Type>,
        components: Vec<naga::Handle<naga::Expression>>,
    },
    /// `textureDimensions`; the per-axis component reads share it.
    Size {
        image: naga::Handle<naga::Expression>,
        level: Option<naga::Handle<naga::Expression>>,
    },
    /// `textureSample`, `textureSampleLevel` or `textureSampleCompare`.
    Sample {
        image: naga::Handle<naga::Expression>,
        sampler: naga::Handle<naga::Expression>,
        coordinate: naga::Handle<naga::Expression>,
        array_index: Option<naga::Handle<naga::Expression>>,
        depth_ref: Option<naga::Handle<naga::Expression>>,
        level: SampleLevelKey,
        clamp_to_edge: bool,
    },
}

/// [`naga::SampleLevel`] without the payload a hash key cannot hold.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SampleLevelKey {
    /// `textureSample`: the hardware picks the mip from the derivatives.
    Auto,
    /// A fixed level of zero.
    Zero,
    /// A level the shader computes.
    Exact(naga::Handle<naga::Expression>),
}

/// Image expressions already emitted in the basic block being lowered.
///
/// The cache is cleared at the start of every block: a handle is valid only
/// inside the block that emitted it, so a later block must not reuse it.
pub(crate) type ImageCache = HashMap<ImageRequest, naga::Handle<naga::Expression>>;
