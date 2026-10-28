//! Textures and samplers are calls, not addresses in linear memory.

use core::marker::PhantomData;

use crate::gpu_abi;
use crate::layout::GpuVec2;
use crate::layout::GpuVec4;

/// 2D texture at `(GROUP, BINDING)`. Sampling is an import, not a load.
#[derive(Clone, Copy, Debug)]
pub struct Texture2d<const GROUP: u32, const BINDING: u32> {
    _marker: PhantomData<()>,
}

impl<const G: u32, const B: u32> Texture2d<G, B> {
    /// Bind this texture. The handle carries no address.
    pub const fn bind() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Sample at `uv`. `sampler` selects the sampler binding; it is not a pointer.
    ///
    /// # Panics
    ///
    /// On the host, panics outside [`gpu_abi::with_host`](crate::with_host), or
    /// when the texture or sampler binding is missing.
    pub fn sample<const SG: u32, const SB: u32>(
        &self,
        sampler: &Sampler<SG, SB>,
        uv: GpuVec2,
    ) -> GpuVec4 {
        let _ = (self, sampler);
        gpu_abi::texture_sample_2d::<G, B, SG, SB>(uv)
    }
}

/// Sampler at `(GROUP, BINDING)`.
#[derive(Clone, Copy, Debug)]
pub struct Sampler<const GROUP: u32, const BINDING: u32> {
    _marker: PhantomData<()>,
}

impl<const G: u32, const B: u32> Sampler<G, B> {
    /// Bind this sampler. The handle carries no address.
    pub const fn bind() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}
