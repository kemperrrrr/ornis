//! Typed texture bindings.
//!
//! A texture is a zero-sized parameter, like [`Storage`](crate::Storage): the
//! group and binding live in the type, and `#[gpu_entry]` fills them from the
//! `#[gpu(texture_2d)]` or `#[gpu(depth_texture_2d)]` attribute. Loads are mip
//! 0. A sampled texture returns one lane of `vec4<f32>`; a depth texture
//! returns the scalar depth.

use crate::Vec3;

/// Sampled `texture_2d<u32>` at `(G, B)`.
///
/// A packed integer target (motion validity plus previous depth in one word)
/// is read as `u32`: the values are flags and fixed-point depth, not colour.
pub struct Texture2dU32<const G: u32, const B: u32> {
    _binding: core::marker::PhantomData<fn() -> u32>,
}

impl<const G: u32, const B: u32> Texture2dU32<G, B> {
    /// First lane of the texel at `(x, y)`, mip 0.
    #[inline(always)]
    pub fn load(&self, x: u32, y: u32) -> u32 {
        let _ = self;
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::texture_load_u32(G, B, x, y)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (x, y);
            crate::host_unavailable()
        }
    }
}

/// Sampled `texture_2d<f32>` at `(G, B)`.
pub struct Texture2d<const G: u32, const B: u32> {
    _binding: core::marker::PhantomData<fn() -> u32>,
}

impl<const G: u32, const B: u32> Texture2d<G, B> {
    /// One lane of `textureLoad` at mip 0.
    ///
    /// The lane is a const generic: the translator only accepts a literal.
    #[inline(always)]
    pub fn load<const LANE: u32>(&self, x: u32, y: u32) -> f32 {
        const { assert!(LANE < 4, "a texel is a vec4") };
        let _ = self;
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::texture_load_f32x4(G, B, x, y, LANE)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (x, y);
            crate::host_unavailable()
        }
    }

    /// Lane `LANE` of `textureSampleLevel(tex, sampler, vec2(x, y), 0.0)`.
    ///
    /// Level 0 is explicit: a fragment has no implicit derivatives here, and
    /// the blur samples a half-res target whose mips do not exist. The sampler
    /// carries its own binding, so only the lane is a const generic.
    #[inline(always)]
    pub fn sample_lane<const SG: u32, const SB: u32, const LANE: u32>(
        &self,
        sampler: Sampler<SG, SB>,
        x: f32,
        y: f32,
    ) -> f32 {
        const { assert!(LANE < 4, "a texel is a vec4") };
        let _ = (self, sampler);
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::texture_sample_level_f32x4(G, B, SG, SB, x, y, LANE)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (x, y);
            crate::host_unavailable()
        }
    }

    /// Lane `LANE` of `textureSample(tex, sampler, vec2(x, y))`.
    ///
    /// The level comes from the screen-space derivatives of the coordinate, so
    /// this is a fragment-stage operation: naga rejects it in a vertex or
    /// compute entry. The BRDF LUT is read this way.
    #[inline(always)]
    pub fn sample_auto_lane<const SG: u32, const SB: u32, const LANE: u32>(
        &self,
        sampler: Sampler<SG, SB>,
        x: f32,
        y: f32,
    ) -> f32 {
        const { assert!(LANE < 4, "a texel is a vec4") };
        let _ = (self, sampler);
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::texture_sample_auto(G, B, SG, SB, x, y, LANE)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (x, y);
            crate::host_unavailable()
        }
    }

    /// Red lane of `textureSampleLevel(tex, sampler, vec2(x, y), 0.0)`.
    #[inline(always)]
    pub fn sample_r<const SG: u32, const SB: u32>(
        &self,
        sampler: Sampler<SG, SB>,
        x: f32,
        y: f32,
    ) -> f32 {
        self.sample_lane::<SG, SB, 0>(sampler, x, y)
    }

    /// Texture width in texels.
    #[inline(always)]
    pub fn width(&self) -> i32 {
        let _ = self;
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::texture_width_f32x4(G, B)
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::host_unavailable()
    }

    /// Texture height in texels.
    #[inline(always)]
    pub fn height(&self) -> i32 {
        let _ = self;
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::texture_height_f32x4(G, B)
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::host_unavailable()
    }
}

/// Sampled `texture_cube<f32>` at `(G, B)` — the IBL environment maps.
///
/// A cube is sampled with a direction, not a uv: the hardware picks the face
/// from the major axis, so the prefiltered radiance and the irradiance cube
/// need no per-face bookkeeping in the shader.
pub struct TextureCube<const G: u32, const B: u32> {
    _binding: core::marker::PhantomData<fn() -> u32>,
}

impl<const G: u32, const B: u32> TextureCube<G, B> {
    /// Lane `LANE` of `textureSampleLevel(cube, sampler, dir, lod)`.
    ///
    /// The level is explicit: a prefiltered mip chain is indexed by perceptual
    /// roughness, so the shader chooses the blur instead of the hardware.
    #[inline(always)]
    pub fn sample_level_lane<const SG: u32, const SB: u32, const LANE: u32>(
        &self,
        sampler: Sampler<SG, SB>,
        dir: Vec3,
        lod: f32,
    ) -> f32 {
        const { assert!(LANE < 4, "a texel is a vec4") };
        let _ = (self, sampler);
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::texture_sample_level_cube(G, B, SG, SB, dir.x, dir.y, dir.z, lod, LANE)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (dir, lod);
            crate::host_unavailable()
        }
    }

    /// Lane `LANE` of `textureSample(cube, sampler, dir)`.
    ///
    /// Fragment-only, like every implicit-level sample.
    #[inline(always)]
    pub fn sample_auto_lane<const SG: u32, const SB: u32, const LANE: u32>(
        &self,
        sampler: Sampler<SG, SB>,
        dir: Vec3,
    ) -> f32 {
        const { assert!(LANE < 4, "a texel is a vec4") };
        let _ = (self, sampler);
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::texture_sample_cube(G, B, SG, SB, dir.x, dir.y, dir.z, LANE)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = dir;
            crate::host_unavailable()
        }
    }
}

impl<const G: u32, const B: u32> Clone for TextureCube<G, B> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<const G: u32, const B: u32> Copy for TextureCube<G, B> {}

/// `texture_depth_2d_array` at `(G, B)` — the shadow-map pool.
///
/// One layer per shadow-casting light, read through a comparison sampler: the
/// reference depth is the fragment's own, and the result is the filtered
/// visibility factor rather than a raw texel.
pub struct DepthTexture2dArray<const G: u32, const B: u32> {
    _binding: core::marker::PhantomData<fn() -> u32>,
}

impl<const G: u32, const B: u32> DepthTexture2dArray<G, B> {
    /// Visibility of `reference` at `uv` in `layer`, as
    /// `textureSampleCompare(tex, sampler, uv, layer, reference)`.
    ///
    /// `reference` is the fragment's depth in the light's clip space minus the
    /// bias, so `1` means lit and `0` fully shadowed.
    #[inline(always)]
    pub fn sample_compare<const SG: u32, const SB: u32>(
        &self,
        sampler: ComparisonSampler<SG, SB>,
        x: f32,
        y: f32,
        layer: u32,
        reference: f32,
    ) -> f32 {
        let _ = (self, sampler);
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::texture_sample_compare_2d_array(G, B, SG, SB, x, y, layer, reference)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (x, y, layer, reference);
            crate::host_unavailable()
        }
    }
}

/// `texture_depth_cube_array` at `(G, B)` — the point-light shadow pool.
///
/// Sampled with the fragment→light direction, so the hardware picks the face;
/// `layer` picks the light.
pub struct DepthCubeArray<const G: u32, const B: u32> {
    _binding: core::marker::PhantomData<fn() -> u32>,
}

impl<const G: u32, const B: u32> DepthCubeArray<G, B> {
    /// Visibility of `reference` along `dir` in `layer`.
    #[inline(always)]
    pub fn sample_compare<const SG: u32, const SB: u32>(
        &self,
        sampler: ComparisonSampler<SG, SB>,
        dir: Vec3,
        layer: u32,
        reference: f32,
    ) -> f32 {
        let _ = (self, sampler);
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::texture_sample_compare_cube_array(
                G, B, SG, SB, dir.x, dir.y, dir.z, layer, reference,
            )
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (dir, layer, reference);
            crate::host_unavailable()
        }
    }
}

impl<const G: u32, const B: u32> Clone for DepthTexture2dArray<G, B> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<const G: u32, const B: u32> Copy for DepthTexture2dArray<G, B> {}

impl<const G: u32, const B: u32> Clone for DepthCubeArray<G, B> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<const G: u32, const B: u32> Copy for DepthCubeArray<G, B> {}

/// Comparison sampler at `(G, B)`.
///
/// A comparison sampler is a distinct WGSL type: a depth image sampled through
/// a filtering sampler is a validation error, not a slower path.
pub struct ComparisonSampler<const G: u32, const B: u32> {
    _binding: core::marker::PhantomData<fn() -> u32>,
}

impl<const G: u32, const B: u32> Clone for ComparisonSampler<G, B> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<const G: u32, const B: u32> Copy for ComparisonSampler<G, B> {}

/// `texture_depth_2d` at `(G, B)`.
pub struct DepthTexture2d<const G: u32, const B: u32> {
    _binding: core::marker::PhantomData<fn() -> u32>,
}

impl<const G: u32, const B: u32> DepthTexture2d<G, B> {
    /// Depth at mip 0.
    #[inline(always)]
    pub fn load(&self, x: u32, y: u32) -> f32 {
        let _ = self;
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::texture_load_depth(G, B, x, y)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (x, y);
            crate::host_unavailable()
        }
    }

    /// Texture width in texels.
    #[inline(always)]
    pub fn width(&self) -> i32 {
        let _ = self;
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::texture_width_depth(G, B)
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::host_unavailable()
    }

    /// Texture height in texels.
    #[inline(always)]
    pub fn height(&self) -> i32 {
        let _ = self;
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::texture_height_depth(G, B)
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::host_unavailable()
    }
}

impl<const G: u32, const B: u32> Clone for Texture2d<G, B> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<const G: u32, const B: u32> Copy for Texture2d<G, B> {}

impl<const G: u32, const B: u32> Clone for DepthTexture2d<G, B> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<const G: u32, const B: u32> Copy for DepthTexture2d<G, B> {}

/// Filtering sampler at `(G, B)`.
///
/// The sampler address/filter modes live on the host; the kernel only passes
/// it to [`Texture2d::sample_r`].
pub struct Sampler<const G: u32, const B: u32> {
    _binding: core::marker::PhantomData<fn() -> u32>,
}

impl<const G: u32, const B: u32> Clone for Sampler<G, B> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<const G: u32, const B: u32> Copy for Sampler<G, B> {}

const _: () = {
    assert!(core::mem::size_of::<Texture2d<0, 0>>() == 0);
    assert!(core::mem::size_of::<TextureCube<0, 0>>() == 0);
    assert!(core::mem::size_of::<DepthTexture2dArray<0, 0>>() == 0);
    assert!(core::mem::size_of::<DepthCubeArray<0, 0>>() == 0);
    assert!(core::mem::size_of::<ComparisonSampler<0, 0>>() == 0);
    assert!(core::mem::size_of::<DepthTexture2d<0, 0>>() == 0);
    assert!(core::mem::size_of::<Sampler<0, 0>>() == 0);
};
