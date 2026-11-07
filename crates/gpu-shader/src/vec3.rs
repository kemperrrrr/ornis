//! Three-lane vector math for kernels.
//!
//! `no_std` has no `f32::sqrt`, and a software square root in a shader is worse
//! than the hardware one: these methods lower to the `ornis_gpu` vector imports
//! the translator turns into WGSL `dot`/`length`, so a length is one GPU
//! instruction rather than three multiplies and a libm call.
//!
//! Only `dot` and `length` exist on purpose; `cross` and `normalize` are not
//! lowered by `ornis-llvm2gpu` yet, and a method that cannot be translated is
//! worse than a missing one because it fails at translation time instead of at
//! compile time.

/// Three `f32` lanes, laid out exactly like a WGSL `vec3<f32>` payload.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// X lane.
    pub x: f32,
    /// Y lane.
    pub y: f32,
    /// Z lane.
    pub z: f32,
}

impl Vec3 {
    /// Build a vector from its lanes.
    #[inline(always)]
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Dot product with `other`.
    #[inline(always)]
    #[must_use]
    pub fn dot(&self, other: Vec3) -> f32 {
        let _ = (self, other);
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::dot3(self.x, self.y, self.z, other.x, other.y, other.z)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (a, b) = (*self, other);
            a.x * b.x + a.y * b.y + a.z * b.z
        }
    }

    /// Euclidean length, `sqrt(dot(self, self))`.
    #[inline(always)]
    #[must_use]
    pub fn length(&self) -> f32 {
        let _ = self;
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::length3(self.x, self.y, self.z)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let squared = self.dot(*self);
            // The host build is a reference for tests only; `sqrt` on `f32`
            // exists there because `std` is linked into the test harness.
            crate::host_sqrt(squared)
        }
    }
}
