//! Texture and sampler bindings and their Naga binding shapes.

use super::VEC2_LANES;
use super::VEC3_LANES;
use crate::symbol;

/// Operand slots a texture sample spends on bindings: texture group and
/// binding, sampler group and binding.
pub(crate) const OWNED_SAMPLE_ARGS: usize = 4;

/// Operand slots a depth-compare sample spends after the coordinate: the
/// array layer and the reference value.
pub(crate) const COMPARE_VALUE_ARGS: usize = 2;

/// Coordinate shape of a texture sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SampleShape {
    /// `texture_2d<f32>`, sampled with a `vec2<f32>`.
    Plane,
    /// `texture_cube<f32>`, sampled with a direction `vec3<f32>`.
    Cube,
}

impl SampleShape {
    /// How the coordinate is read.
    pub(crate) fn dimension(self) -> naga::ImageDimension {
        match self {
            Self::Plane => naga::ImageDimension::D2,
            Self::Cube => naga::ImageDimension::Cube,
        }
    }

    /// Coordinate lanes the sampler takes.
    pub(crate) fn coordinate_lanes(self) -> usize {
        match self {
            Self::Plane => VEC2_LANES as usize,
            Self::Cube => VEC3_LANES as usize,
        }
    }
}

/// Where a sample's mip level comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SampleMip {
    /// `textureSample`: the hardware differences the coordinate across the
    /// pixel quad. Fragment stage only.
    Auto,
    /// `textureSampleLevel(.., 0.0)`: the blur and AO passes read mip 0.
    Zero,
    /// `textureSampleLevel(.., lod)`: the level is the caller's operand, the
    /// way the prefiltered IBL cube is read.
    Operand,
}

/// Shape and mip source of a texture-sample import, by its name.
///
/// The names are matched whole, so `texture_sample_level_*` and
/// `texture_sample_*` cannot answer for each other and the order of the arms
/// carries no meaning.
pub(crate) fn sample_import(callee: &str) -> Option<(SampleShape, SampleMip)> {
    use ornis_gpu_abi::symbols::{
        TEXTURE_SAMPLE_2D_F32X4, TEXTURE_SAMPLE_CUBE_F32X4, TEXTURE_SAMPLE_LEVEL_CUBE_F32X4,
    };
    const LEVEL_2D: &str = "texture_sample_level_2d_f32x4";
    match symbol::imported(callee) {
        TEXTURE_SAMPLE_LEVEL_CUBE_F32X4 => Some((SampleShape::Cube, SampleMip::Operand)),
        TEXTURE_SAMPLE_CUBE_F32X4 => Some((SampleShape::Cube, SampleMip::Auto)),
        LEVEL_2D => Some((SampleShape::Plane, SampleMip::Zero)),
        TEXTURE_SAMPLE_2D_F32X4 => Some((SampleShape::Plane, SampleMip::Auto)),
        _ => None,
    }
}

/// Coordinate shape of a depth-compare sample: an arrayed depth image, either
/// a 2D layer stack or a cube face stack.
pub(crate) fn compare_import(callee: &str) -> Option<SampleShape> {
    use ornis_gpu_abi::symbols::{
        TEXTURE_SAMPLE_COMPARE_2D_ARRAY_F32, TEXTURE_SAMPLE_COMPARE_CUBE_ARRAY_F32,
    };
    match symbol::imported(callee) {
        TEXTURE_SAMPLE_COMPARE_CUBE_ARRAY_F32 => Some(SampleShape::Cube),
        TEXTURE_SAMPLE_COMPARE_2D_ARRAY_F32 => Some(SampleShape::Plane),
        _ => None,
    }
}

/// The coordinate and the two globals a sample reads.
pub(crate) struct SampleTarget {
    /// The composed coordinate: `vec2` for a plane, `vec3` for a direction.
    pub(crate) coordinate: naga::Handle<naga::Expression>,
    /// The image global.
    pub(crate) image: naga::Handle<naga::Expression>,
    /// The sampler global.
    pub(crate) sampler: naga::Handle<naga::Expression>,
}

/// A texture binding the module declared.
///
/// The kind decides what a load returns, `dim` how the coordinate is read, and
/// `arrayed` whether a layer index comes with it — a 2D array and a cube array
/// are different types, so one binding cannot serve both.
pub(crate) struct TextureBinding {
    /// What a load or sample of the binding returns.
    pub(crate) kind: TextureKind,
    /// How the coordinate is shaped.
    pub(crate) dim: naga::ImageDimension,
    /// Whether the image carries an array layer.
    pub(crate) arrayed: bool,
    /// The global the module reads.
    pub(crate) handle: naga::Handle<naga::GlobalVariable>,
}

/// Which `texture_*` binding a load addresses.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextureKind {
    /// `texture_depth_2d`. `textureLoad` returns `f32`.
    Depth,
    /// `texture_2d<f32>`. `textureLoad` returns `vec4<f32>`.
    Sampled,
    /// `texture_2d<u32>`. `textureLoad` returns `vec4<u32>`; lane 0 is the
    /// packed word a `R16Uint`-style target carries.
    Unsigned,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Elem {
    F32,
    U32,
    AtomicU32,
    /// Bit-preserving four-word record chunk (`array<vec4<u32>>`).
    Vec4U32,
}

/// `llvm.umin` / `minnum` / `umax` / `maxnum` — the forms rustc emits for
/// `Ord::min`/`max`. WGSL `min`/`max` have the same NaN rule as `minnum`.
pub(crate) fn image_class(kind: TextureKind, multisampled: bool) -> naga::ImageClass {
    match kind {
        TextureKind::Depth => naga::ImageClass::Depth {
            multi: multisampled,
        },
        TextureKind::Sampled => naga::ImageClass::Sampled {
            kind: naga::ScalarKind::Float,
            multi: multisampled,
        },
        TextureKind::Unsigned => naga::ImageClass::Sampled {
            kind: naga::ScalarKind::Uint,
            multi: multisampled,
        },
    }
}
