//! Scalar GPU math for kernels.
//!
//! `no_std` has no `log2`/`exp2`/`acos`, and a software series would not match
//! the GPU instruction. Each function lowers to one `ornis_gpu` import. A
//! kernel calls these; it does not declare the import.
//!
//! The `wasm32` arm is a direct `#[inline(always)]` call. A function pointer
//! would survive into the kernel's LLVM IR, and the translator only sees calls
//! written in the entry.

macro_rules! unary {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[inline(always)]
        #[must_use]
        pub fn $name(value: f32) -> f32 {
            #[cfg(target_arch = "wasm32")]
            {
                crate::abi::$name(value)
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let _ = value;
                crate::host_unavailable()
            }
        }
    };
}

unary!(
    /// `log2(value)`.
    log2
);
unary!(
    /// `exp2(value)`.
    exp2
);
unary!(
    /// Absolute value, one WGSL `abs`.
    abs
);
unary!(
    /// Sign of `value`: `-1`, `0`, or `1`.
    sign
);
unary!(
    /// Cosine.
    cos
);
unary!(
    /// Sine.
    sin
);
unary!(
    /// Arccosine.
    acos
);
unary!(
    /// Fractional part of `value`.
    fract
);

/// Clamp `value` into `low..=high`.
#[inline(always)]
#[must_use]
pub fn clamp(value: f32, low: f32, high: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::clamp(value, low, high)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (value, low, high);
        crate::host_unavailable()
    }
}

/// Linear interpolation `start + (end - start) * factor`.
#[inline(always)]
#[must_use]
pub fn mix(start: f32, end: f32, factor: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::mix(start, end, factor)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (start, end, factor);
        crate::host_unavailable()
    }
}

/// Dot product of two scalar `vec3`s.
// qual:allow(srp, max_parameters=6) reason: "six lanes, the same operands as the import; a Vec3 argument would be a second copy of crate::Vec3::dot"
#[inline(always)]
#[must_use]
pub fn dot3(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::dot3(ax, ay, az, bx, by, bz)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (ax, ay, az, bx, by, bz);
        crate::host_unavailable()
    }
}

/// Length of a scalar `vec3`.
#[inline(always)]
#[must_use]
pub fn length3(x: f32, y: f32, z: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::length3(x, y, z)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (x, y, z);
        crate::host_unavailable()
    }
}

/// Lane `LANE` of `cross(a, b)`. `LANE` is `0..3`.
// qual:allow(srp, max_parameters=6) reason: "six vector lanes; the result lane is a const generic so the translator sees a literal"
#[inline(always)]
#[must_use]
pub fn cross_lane<const LANE: u32>(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32 {
    const { assert!(LANE < 3, "a vec3 has three lanes") };
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::cross3_lane(ax, ay, az, bx, by, bz, LANE)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (ax, ay, az, bx, by, bz);
        crate::host_unavailable()
    }
}

/// Lane `LANE` of `normalize(x, y, z)`. `LANE` is `0..3`.
#[inline(always)]
#[must_use]
pub fn normalize_lane<const LANE: u32>(x: f32, y: f32, z: f32) -> f32 {
    const { assert!(LANE < 3, "a vec3 has three lanes") };
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::normalize3_lane(x, y, z, LANE)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (x, y, z);
        crate::host_unavailable()
    }
}
