//! The OpenPBR shading math, shared by the forward-PBR, lighting and textured
//! material passes.
//!
//! Every function here is the shader-side twin of the reference implementation
//! in `ornis-render`'s `shaders::math`; `openpbr_math_parity` pins the two
//! together with a numeric comparison. Signatures stay in `Vec3` terms so the
//! comparison is per-lane, and the arithmetic is spelled out lane by lane
//! because the translator reads LLVM IR: a glam-style operator overload on a
//! struct is an aggregate there, three `f32` operations here.
//!
//! Rounding order matches the reference: math that is order-sensitive
//! (`dot3`, `length3`, `cross_lane`, `normalize_lane`) goes through the SDK, so
//! the GPU gets the single instruction the reference approximates.

#![no_std]
#![warn(missing_docs)]

pub mod math;
