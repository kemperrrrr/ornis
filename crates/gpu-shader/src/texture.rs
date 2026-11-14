//! Typed 2D texture bindings.
//!
//! A texture is a zero-sized parameter, like [`Storage`](crate::Storage): the
//! group and binding live in the type, and `#[gpu_entry]` fills them from the
//! `#[gpu(texture_2d)]` or `#[gpu(depth_texture_2d)]` attribute. Loads are mip
//! 0. A sampled texture returns one lane of `vec4<f32>`; a depth texture
//! returns the scalar depth.

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
    assert!(core::mem::size_of::<DepthTexture2d<0, 0>>() == 0);
    assert!(core::mem::size_of::<Sampler<0, 0>>() == 0);
};
