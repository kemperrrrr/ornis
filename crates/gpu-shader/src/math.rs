//! Scalar GPU math for kernels.
//!
//! `no_std` has no `log2`/`exp2`/`acos`, and a software series would not match
//! the GPU instruction. Each function lowers to one `ornis_gpu` import. A
//! kernel calls these; it does not declare the import.
//!
//! The `wasm32` arm is a direct `#[inline(always)]` call. A function pointer
//! would survive into the kernel's LLVM IR, and the translator only sees calls
//! written in the entry.
//!
//! Off `wasm32` the same functions evaluate for real through [`libm`]: the host
//! build of a shader crate is the reference the GPU parity gates compare
//! against, so it has to return values rather than panic. The two arms round
//! independently — a parity gate compares them with a tolerance, it never
//! expects a bit-exact match.

/// Coefficients of the Hermite smoothstep curve, `3t² - 2t³`.
const SMOOTHSTEP_CUBIC: f32 = 3.0;
/// Coefficient of the cubic term, `-2t³`.
const SMOOTHSTEP_CUBIC_NEGATIVE: f32 = -2.0;

/// Host-side implementations shared by the arithmetic functions below.
///
/// `wasm32` has none of these: every shader operation is an import there.
#[cfg(not(target_arch = "wasm32"))]
mod host {
    /// `-1`, `0` or `1`, the WGSL `sign` of `value`.
    #[inline(always)]
    pub(super) fn sign(value: f32) -> f32 {
        if value > 0.0 {
            1.0
        } else if value < 0.0 {
            -1.0
        } else {
            0.0
        }
    }

    /// Fractional part, `value - floor(value)`.
    #[inline(always)]
    pub(super) fn fract(value: f32) -> f32 {
        value - libm::floorf(value)
    }

    /// `start + (end - start) * factor`.
    #[inline(always)]
    pub(super) fn mix(start: f32, end: f32, factor: f32) -> f32 {
        start + (end - start) * factor
    }

    /// `smoothstep(low, high, x)`.
    ///
    /// The Hermite curve, not the instruction the GPU calls: the gate's
    /// tolerance absorbs the difference.
    #[inline(always)]
    pub(super) fn smoothstep(low: f32, high: f32, x: f32) -> f32 {
        let span = high - low;
        if span == 0.0 {
            return 0.0;
        }
        let t = ((x - low) / span).clamp(0.0, 1.0);
        t * t * (super::SMOOTHSTEP_CUBIC + super::SMOOTHSTEP_CUBIC_NEGATIVE * t)
    }
}

macro_rules! unary {
    ($(#[$meta:meta])* $name:ident => $host:path) => {
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
                $host(value)
            }
        }
    };
}

unary!(
    /// `log2(value)`.
    log2 => libm::log2f
);
unary!(
    /// `exp2(value)`.
    exp2 => libm::exp2f
);
unary!(
    /// Natural logarithm, `ln(value)` — one WGSL `log`.
    ln => libm::logf
);
unary!(
    /// `exp(value)` — one WGSL `exp`.
    exp => libm::expf
);
unary!(
    /// Absolute value, one WGSL `abs`.
    abs => libm::fabsf
);
unary!(
    /// Sign of `value`: `-1`, `0`, or `1`.
    sign => host::sign
);
unary!(
    /// Cosine.
    cos => libm::cosf
);
unary!(
    /// Sine.
    sin => libm::sinf
);
unary!(
    /// Tangent.
    tan => libm::tanf
);
unary!(
    /// Arccosine.
    acos => libm::acosf
);
unary!(
    /// Fractional part of `value`.
    fract => host::fract
);

/// `sqrt(value)` — one WGSL `sqrt`.
///
/// `no_std` has no `f32::sqrt`; a Newton series would not match the GPU
/// instruction.
#[inline(always)]
#[must_use]
pub fn sqrt(value: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::sqrt(value)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        libm::sqrtf(value)
    }
}

/// `smoothstep(low, high, x)` — one WGSL `smoothstep`.
///
/// Expanding it by hand (`clamp((x - low) / (high - low))³`) rounds differently
/// from the instruction, so the shader calls `smoothstep` itself.
#[inline(always)]
#[must_use]
pub fn smoothstep(low: f32, high: f32, x: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::smoothstep(low, high, x)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        host::smoothstep(low, high, x)
    }
}

/// `pow(base, exponent)` — one WGSL `pow`.
///
/// `exp2(exponent * log2(base))` would round twice and drift from the GPU
/// instruction, so the shader calls `pow` itself.
#[inline(always)]
#[must_use]
pub fn pow(base: f32, exponent: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::pow(base, exponent)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        libm::powf(base, exponent)
    }
}

/// `atan2(y, x)` — one WGSL `atan2`.
#[inline(always)]
#[must_use]
pub fn atan2(y: f32, x: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::atan2(y, x)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        libm::atan2f(y, x)
    }
}

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
        if value < low {
            low
        } else if value > high {
            high
        } else {
            value
        }
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
        host::mix(start, end, factor)
    }
}

/// Dot product of two scalar `vec2`s.
///
/// Lowers to WGSL `dot(vec2, vec2)`. The host value is
/// `fma(ay, by, ax * bx)`, the contraction a `vec2` dot uses when the
/// second lane is the fused addend. A scalar `ax * bx + ay * by` is the
/// opposite contraction.
#[inline(always)]
#[must_use]
pub fn dot2(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::dot2(ax, ay, bx, by)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        libm::fmaf(ay, by, ax * bx)
    }
}

/// `value / w` with the sign and magnitude of `w` left unchanged.
///
/// The hand-written `reconstruct_world_pos` returns `view.xyz / view.w`.
/// AO reconstruction divides by the raw homogeneous `w` the same way,
/// including when `|w|` is below `1e-6` or `w` is negative. Clamping `w`
/// up to `EPS` would move those reconstructions and would flip a negative
/// denominator.
#[inline(always)]
#[must_use]
pub fn homogeneous_divide(value: f32, w: f32) -> f32 {
    value / w
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
        libm::fmaf(ay, by, libm::fmaf(ax, bx, az * bz))
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
        libm::sqrtf(x * x + y * y + z * z)
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
        match LANE {
            0 => ay * bz - az * by,
            1 => az * bx - ax * bz,
            _ => ax * by - ay * bx,
        }
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
        let inverse = 1.0 / libm::sqrtf(x * x + y * y + z * z);
        match LANE {
            0 => x * inverse,
            1 => y * inverse,
            _ => z * inverse,
        }
    }
}
