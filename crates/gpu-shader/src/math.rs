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
//! Off `wasm32` the same functions evaluate for real: `sqrt` calls
//! [`f32::sqrt`] and the rest go through [`libm`]. `f32::sqrt` is implemented
//! in `std` rather than `core` on this toolchain, so the host arm links `std`;
//! the wasm arm does not. The host build is the reference the GPU parity gates
//! compare against, so it has to return values rather than panic. The two arms
//! round independently — a parity gate compares them with a tolerance, it never
//! expects a bit-exact match.

/// `f32::sqrt` lives in `std`. The wasm build never compiles this arm.
#[cfg(not(target_arch = "wasm32"))]
extern crate std;

/// Host-side implementations shared by the arithmetic functions below.
///
/// `wasm32` has none of these: every shader operation is an import there.
#[cfg(not(target_arch = "wasm32"))]
mod host {
    /// Coefficient of the quadratic term of the Hermite smoothstep, `3t²`.
    const SMOOTHSTEP_QUADRATIC: f32 = 3.0;
    /// Coefficient of the cubic term of the Hermite smoothstep, `-2t³`.
    const SMOOTHSTEP_CUBIC: f32 = -2.0;

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
        t * t * (SMOOTHSTEP_QUADRATIC + SMOOTHSTEP_CUBIC * t)
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
unary!(
    /// Largest integer not greater than `value`, one WGSL `floor`.
    floor => libm::floorf
);
unary!(
    /// Smallest integer not less than `value`, one WGSL `ceil`.
    ceil => libm::ceilf
);

/// `sqrt(value)` — one WGSL `sqrt`.
///
/// The host build calls [`f32::sqrt`]. The wasm build imports the operation,
/// so the GPU instruction is the one that runs; a Newton series would not
/// match it.
#[inline(always)]
#[must_use]
pub fn sqrt(value: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::sqrt(value)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        value.sqrt()
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

/// Screen-space derivative along x, one WGSL `dpdx`.
///
/// A host build evaluates one pixel at a time, so the derivative there is zero:
/// there is no neighbour to difference against. A shader must not depend on the
/// host value — it exists so the kernel crate is executable off the GPU.
#[inline(always)]
#[must_use]
pub fn dpdx(value: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::dpdx(value)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = value;
        0.0
    }
}

/// Screen-space derivative along y, one WGSL `dpdy`.
#[inline(always)]
#[must_use]
pub fn dpdy(value: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::dpdy(value)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = value;
        0.0
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
        sqrt(x * x + y * y + z * z)
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
        let inverse = 1.0 / sqrt(x * x + y * y + z * z);
        match LANE {
            0 => x * inverse,
            1 => y * inverse,
            _ => z * inverse,
        }
    }
}

/// The host arms, evaluated: the SDK is the reference implementation a parity
/// gate compares GPU output against, so each arm has to return the real value
/// rather than a placeholder.
#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn unary_arms_evaluate() {
        close(sqrt(4.0), 2.0);
        close(abs(-2.5), 2.5);
        close(sign(-3.0), -1.0);
        close(sign(0.0), 0.0);
        close(sign(3.0), 1.0);
        close(fract(2.25), 0.25);
        close(floor(2.75), 2.0);
        close(floor(-0.25), -1.0);
        close(ceil(2.25), 3.0);
        close(ceil(-0.25), -0.0);
        close(log2(8.0), 3.0);
        close(exp2(3.0), 8.0);
        close(ln(1.0), 0.0);
        close(exp(0.0), 1.0);
        close(cos(0.0), 1.0);
        close(sin(0.0), 0.0);
        close(tan(0.0), 0.0);
        close(acos(1.0), 0.0);
    }

    #[test]
    fn binary_and_ternary_arms_evaluate() {
        close(pow(2.0, 3.0), 8.0);
        close(atan2(1.0, 1.0), core::f32::consts::FRAC_PI_4);
        close(clamp(1.5, 0.0, 1.0), 1.0);
        close(clamp(-1.5, 0.0, 1.0), 0.0);
        close(clamp(0.25, 0.0, 1.0), 0.25);
        close(mix(2.0, 4.0, 0.5), 3.0);
        close(smoothstep(0.0, 1.0, 0.5), 0.5);
        close(smoothstep(0.0, 1.0, -1.0), 0.0);
        close(smoothstep(0.0, 1.0, 2.0), 1.0);
    }

    #[test]
    fn vector_arms_evaluate() {
        close(dot2(1.0, 2.0, 3.0, 4.0), 11.0);
        close(dot3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), 32.0);
        close(length3(3.0, 4.0, 0.0), 5.0);
        close(cross_lane::<0>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), 0.0);
        close(cross_lane::<1>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), 0.0);
        close(cross_lane::<2>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), 1.0);
        close(normalize_lane::<0>(3.0, 4.0, 0.0), 0.6);
        close(normalize_lane::<1>(3.0, 4.0, 0.0), 0.8);
        close(homogeneous_divide(6.0, 3.0), 2.0);
    }

    #[test]
    fn derivative_arms_report_no_neighbour() {
        // A host evaluation is one pixel, so there is nothing to difference
        // against: the derivative is zero, and a shader must not rely on it.
        close(dpdx(5.0), 0.0);
        close(dpdy(5.0), 0.0);
    }
}
