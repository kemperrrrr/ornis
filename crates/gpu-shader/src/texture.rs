//! Typed 2D texture bindings.
//!
//! A texture is a zero-sized parameter, like [`Storage`](crate::Storage): the
//! group and binding live in the type, and `#[gpu_entry]` fills them from the
//! `#[gpu(texture_2d)]` or `#[gpu(depth_texture_2d)]` attribute. Loads are mip
//! 0. A sampled texture returns one lane of `vec4<f32>`; a depth texture
//! returns the scalar depth.

/// Sampled `texture_2d<f32>` at `(G, B)`.
pub struct Texture2d<const G: u32, const B: u32> {
    _binding: core::marker::PhantomData<fn() -> u32>,
}

impl<const G: u32, const B: u32> Texture2d<G, B> {
    /// Lane `LANE` (`0..4`) of `textureLoad` at mip 0.
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

const _: () = {
    assert!(core::mem::size_of::<Texture2d<0, 0>>() == 0);
    assert!(core::mem::size_of::<DepthTexture2d<0, 0>>() == 0);
};
