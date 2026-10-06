//! Physical and rendering unit newtypes over raw `f32` soup.
//!
//! One paragraph: raw `f32` fields across physics, assets, gameplay and
//! render carry incompatible units (meters, radians, seconds, colors) that
//! the compiler cannot tell apart. This module gives each unit a distinct
//! transparent wrapper with `From`/`Into` conversions and a raw getter, so
//! typed constructors accept units while the stored `f32` layout, serde
//! schemas and GPU `Pod` blobs stay bit-identical. Fallible wrappers
//! ([`PositiveF32`], [`Clamped01`], [`UnitVec3`], [`UnitQuat`]) turn the
//! old comment-invariants ("must be > 0", "must be normalized") into
//! checked constructors instead of silent defaults.
//! [`Roughness`], [`Metallic`], [`Specular`] and [`EnvironmentWeight`]
//! are infallible `[0, 1]` weights (NaN becomes `0`) with the same bare
//! `f32` wire form as [`Clamped01`]. [`Surface`] bundles the three
//! material weights an authored spawn can stamp onto every primitive.
//!
//! [`Color`] stores [`LinearRgba`] and serializes as the scene file's linear
//! RGB array. [`Lux`] is the light-intensity `f32` under a newtype.

use glam::{Quat, Vec3};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

mod color;
pub use color::{Color, HexColorError, Lux};

/// Duration in seconds at the frame boundary (canonical definition).
///
/// Re-exported from [`crate::typestate`] so physics steps, engine time and
/// gameplay deltas share one type; the [`SecondsExt`] trait adds the
/// step-validity check.
pub use crate::typestate::Seconds;

/// Step-delta validation for [`Seconds`].
///
/// The engine traits document `dt > 0` and finite; this turns the comment
/// into a callable check shared by all step entry points.
pub trait SecondsExt {
    /// Whether this is a usable integration step (`finite && > 0`).
    fn is_valid_step(&self) -> bool;
}

impl SecondsExt for Seconds {
    fn is_valid_step(&self) -> bool {
        let v = self.get();
        v.is_finite() && v > 0.0
    }
}

/// Length in meters (positions, radii, ranges, cell spacing).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Meters(pub f32);

impl Meters {
    /// Zero length.
    pub const ZERO: Self = Self(0.0);

    /// Wraps a raw meter value without validation (validation lives at the
    /// typed constructors that consume it).
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    /// Raw value in meters.
    pub const fn get(self) -> f32 {
        self.0
    }

    /// Whether the value is finite.
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

impl From<f32> for Meters {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<Meters> for f32 {
    fn from(value: Meters) -> Self {
        value.get()
    }
}

/// A point in meters.
///
/// The translation stored on a local [`Transform`](crate::Transform). Not an
/// ECS component: gameplay keeps its own `Position` lane.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Position(Vec3);

impl Position {
    /// The origin.
    pub const ORIGIN: Self = Self(Vec3::ZERO);

    /// Point at `(x, y, z)` meters.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self(Vec3::new(x, y, z))
    }

    /// The raw vector, in meters.
    pub const fn get(self) -> Vec3 {
        self.0
    }
}

impl From<Vec3> for Position {
    fn from(value: Vec3) -> Self {
        Self::new(value.x, value.y, value.z)
    }
}

impl From<Position> for Vec3 {
    fn from(value: Position) -> Self {
        value.get()
    }
}

/// Angle in radians (hinge twists, limits, angular velocity integration).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Radians(pub f32);

impl Radians {
    /// Zero angle.
    pub const ZERO: Self = Self(0.0);

    /// Wraps a raw radian value without validation.
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    /// Raw value in radians.
    pub const fn get(self) -> f32 {
        self.0
    }

    /// Whether the value is finite.
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    /// Converts to degrees.
    pub fn to_degrees(self) -> Degrees {
        Degrees(self.0.to_degrees())
    }
}

impl From<f32> for Radians {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<Radians> for f32 {
    fn from(value: Radians) -> Self {
        value.get()
    }
}

impl From<Degrees> for Radians {
    fn from(value: Degrees) -> Self {
        Self::new(value.get().to_radians())
    }
}

/// Angle in degrees (scene cameras, spot cones, editor-facing fields).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Degrees(pub f32);

impl Degrees {
    /// Zero angle.
    pub const ZERO: Self = Self(0.0);

    /// Wraps a raw degree value without validation.
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    /// Raw value in degrees.
    pub const fn get(self) -> f32 {
        self.0
    }

    /// Whether the value is finite.
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    /// Converts to radians.
    pub fn to_radians(self) -> Radians {
        Radians(self.0.to_radians())
    }
}

impl From<f32> for Degrees {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<Degrees> for f32 {
    fn from(value: Degrees) -> Self {
        value.get()
    }
}

impl From<Radians> for Degrees {
    fn from(value: Radians) -> Self {
        Self::new(value.get().to_degrees())
    }
}

impl Serialize for Degrees {
    /// Wire form is the raw degree `f32` (camera `fov`, spot angles).
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(self.0)
    }
}

impl<'de> Deserialize<'de> for Degrees {
    /// Reads the raw degree `f32`.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self(f32::deserialize(deserializer)?))
    }
}

/// Frequency in hertz (wheel suspension resonance).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Hertz(pub f32);

impl Hertz {
    /// Wraps a raw hertz value without validation.
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    /// Raw value in hertz.
    pub const fn get(self) -> f32 {
        self.0
    }

    /// Whether the spring is live (`finite && > 0`).
    pub fn is_live(self) -> bool {
        self.0.is_finite() && self.0 > 0.0
    }

    /// Resonance period (`1 / f`), or `None` for a dead spring.
    pub fn period(self) -> Option<Seconds> {
        if self.is_live() {
            Some(Seconds(1.0 / self.0))
        } else {
            None
        }
    }
}

impl From<f32> for Hertz {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<Hertz> for f32 {
    fn from(value: Hertz) -> Self {
        value.get()
    }
}

/// Mass in kilograms (`0` = static/infinite-mass, `> 0` = dynamic).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Kilograms(pub f32);

impl Kilograms {
    /// Zero mass (static behavior).
    pub const ZERO: Self = Self(0.0);

    /// Wraps a raw kilogram value without validation.
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    /// Raw value in kilograms.
    pub const fn get(self) -> f32 {
        self.0
    }

    /// Whether this is a valid mass model input (finite and `>= 0`).
    pub fn is_valid(self) -> bool {
        self.0.is_finite() && self.0 >= 0.0
    }

    /// Whether this mass behaves as static (`<= 0` or non-finite).
    pub fn is_static(self) -> bool {
        !(self.0.is_finite() && self.0 > 0.0)
    }

    /// Strictly positive mass, or `None` for static/invalid input.
    pub fn as_positive(self) -> Option<PositiveF32> {
        PositiveF32::try_new(self.0)
    }
}

impl From<f32> for Kilograms {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<Kilograms> for f32 {
    fn from(value: Kilograms) -> Self {
        value.get()
    }
}

/// Angular speed in radians per second (hinge motors, spin).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RadiansPerSecond(pub f32);

impl RadiansPerSecond {
    /// Zero angular speed.
    pub const ZERO: Self = Self(0.0);

    /// Wraps a raw rad/s value without validation.
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    /// Raw value in rad/s.
    pub const fn get(self) -> f32 {
        self.0
    }

    /// Whether the value is finite.
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

impl From<f32> for RadiansPerSecond {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<RadiansPerSecond> for f32 {
    fn from(value: RadiansPerSecond) -> Self {
        value.get()
    }
}

/// Dimensionless gear transmission ratio (`coord_a + ratio * coord_b`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GearRatio(f32);

impl GearRatio {
    /// Checked constructor: `Some` only for finite non-zero ratios (a zero
    /// ratio would freeze `coord_a` at its assembly constant).
    pub fn try_new(ratio: f32) -> Option<Self> {
        if ratio.is_finite() && ratio != 0.0 {
            Some(Self(ratio))
        } else {
            None
        }
    }

    /// Raw ratio value.
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl From<GearRatio> for f32 {
    fn from(value: GearRatio) -> Self {
        value.get()
    }
}

/// Linear speed in meters per second (gameplay intent, motor targets).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MetersPerSecond(pub f32);

impl MetersPerSecond {
    /// Zero speed.
    pub const ZERO: Self = Self(0.0);

    /// Wraps a raw m/s value without validation.
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    /// Raw value in m/s.
    pub const fn get(self) -> f32 {
        self.0
    }

    /// Whether the value is finite.
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

impl From<f32> for MetersPerSecond {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<MetersPerSecond> for f32 {
    fn from(value: MetersPerSecond) -> Self {
        value.get()
    }
}

/// Rejected [`PositiveF32`] input (must be finite and `> 0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositiveF32Error;

/// Positive finite `f32` (masses, radii, spacings at checked boundaries).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositiveF32(f32);

impl PositiveF32 {
    /// Checked constructor: `Some` only for finite values `> 0`.
    pub const fn try_new(value: f32) -> Option<Self> {
        if value.is_finite() && value > 0.0 {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Constant-payload constructor for literals validated by inspection
    /// (tests, probes, default rigs). Non-finite or non-positive input
    /// falls back to [`f32::MIN_POSITIVE`] instead of panicking.
    pub const fn expect_valid(value: f32) -> Self {
        match Self::try_new(value) {
            Some(valid) => valid,
            None => Self(f32::MIN_POSITIVE),
        }
    }

    /// Raw positive value.
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl Serialize for PositiveF32 {
    /// Wire form is the raw `f32` (transport schemas stay unchanged).
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(self.0)
    }
}

impl<'de> Deserialize<'de> for PositiveF32 {
    /// Reads the raw `f32` wire form; rejects non-positive or non-finite
    /// input instead of building degenerate geometry.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = f32::deserialize(deserializer)?;
        Self::try_new(value).ok_or_else(|| {
            D::Error::custom(format!(
                "PositiveF32 requires a finite value > 0, got {value}"
            ))
        })
    }
}

impl TryFrom<f32> for PositiveF32 {
    type Error = PositiveF32Error;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::try_new(value).ok_or(PositiveF32Error)
    }
}

impl From<PositiveF32> for f32 {
    fn from(value: PositiveF32) -> Self {
        value.get()
    }
}

/// Rejected [`Clamped01`] input for the fallible path (must be in `[0, 1]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Clamped01Error;

/// `f32` clamped to `[0, 1]` (weights, roughness, opacity).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Clamped01(f32);

impl Clamped01 {
    /// Zero (minimum).
    pub const ZERO: Self = Self(0.0);

    /// One (maximum).
    pub const ONE: Self = Self(1.0);

    /// Infallible constructor: clamps into `[0, 1]` (NaN maps to `0`).
    pub fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Checked constructor: `Some` only for finite values already in `[0, 1]`.
    pub fn try_new(value: f32) -> Option<Self> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Raw value in `[0, 1]`.
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl From<Clamped01> for f32 {
    fn from(value: Clamped01) -> Self {
        value.get()
    }
}

impl Serialize for Clamped01 {
    /// Wire form is the raw `f32` (transport schemas stay unchanged).
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(self.0)
    }
}

impl<'de> Deserialize<'de> for Clamped01 {
    /// Reads the raw `f32` wire form, clamping into `[0, 1]` (same
    /// leniency as [`Clamped01::new`], so legacy payloads keep loading).
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(f32::deserialize(deserializer)?))
    }
}

/// Rejected [`Ior`] input for the fallible path (must be finite and `>= 1.0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IorError;

/// Index of refraction for transmissive surfaces (thin-walled glass).
/// Values below `1.0` are unphysical (faster than light in vacuum).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ior(f32);

impl Ior {
    /// Crown-glass default (`1.5`); also the `serde` default for older
    /// payloads without an `ior` field.
    pub const DEFAULT: Self = Self(1.5);

    /// Infallible constructor: values below `1.0` clamp up to `1.0`,
    /// non-finite input falls back to [`Ior::DEFAULT`] (same policy as the
    /// typed glass constructors).
    pub const fn new(value: f32) -> Self {
        if !value.is_finite() {
            Self::DEFAULT
        } else if value < 1.0 {
            Self(1.0)
        } else {
            Self(value)
        }
    }

    /// Checked constructor: `Some` only for finite values `>= 1.0`.
    pub const fn try_new(value: f32) -> Option<Self> {
        if value.is_finite() && value >= 1.0 {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Raw index of refraction (`>= 1.0`).
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl Default for Ior {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl From<Ior> for f32 {
    fn from(value: Ior) -> Self {
        value.get()
    }
}

impl TryFrom<f32> for Ior {
    type Error = IorError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::try_new(value).ok_or(IorError)
    }
}

impl Serialize for Ior {
    /// Wire form is the raw `f32` (transport schemas stay unchanged).
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(self.0)
    }
}

impl<'de> Deserialize<'de> for Ior {
    /// Reads the raw `f32` wire form through [`Ior::new`] (same leniency
    /// as the typed constructors, so legacy payloads keep loading).
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(f32::deserialize(deserializer)?))
    }
}

impl TryFrom<f32> for Clamped01 {
    type Error = Clamped01Error;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::try_new(value).ok_or(Clamped01Error)
    }
}

/// Maps `value` into `[0, 1]`.
///
/// `f32::clamp` leaves NaN unchanged. A missing weight is `0`.
fn clamp_unit_interval(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

/// Specular roughness in `[0, 1]` (`0` mirror-smooth, `1` fully rough).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Roughness(f32);

impl Roughness {
    /// Infallible constructor: clamps into `[0, 1]` (NaN maps to `0`).
    pub fn new(value: f32) -> Self {
        Self(clamp_unit_interval(value))
    }

    /// Raw value in `[0, 1]`.
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl From<Roughness> for f32 {
    fn from(value: Roughness) -> Self {
        value.get()
    }
}

impl Serialize for Roughness {
    /// Wire form is the raw `f32` (same as [`Clamped01`]).
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(self.0)
    }
}

impl<'de> Deserialize<'de> for Roughness {
    /// Reads the raw `f32` and clamps it ([`Roughness::new`]; NaN → `0`).
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(f32::deserialize(deserializer)?))
    }
}

/// Base metalness in `[0, 1]` (`0` dielectric, `1` metal).
///
/// The GPU mixes the two continuously: `base.params[2]` is this value,
/// not a binary preset switch.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Metallic(f32);

impl Metallic {
    /// Infallible constructor: clamps into `[0, 1]` (NaN maps to `0`).
    pub fn new(value: f32) -> Self {
        Self(clamp_unit_interval(value))
    }

    /// Raw value in `[0, 1]`.
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl From<Metallic> for f32 {
    fn from(value: Metallic) -> Self {
        value.get()
    }
}

impl Serialize for Metallic {
    /// Wire form is the raw `f32` (same as [`Clamped01`]).
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(self.0)
    }
}

impl<'de> Deserialize<'de> for Metallic {
    /// Reads the raw `f32` and clamps it ([`Metallic::new`]; NaN → `0`).
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(f32::deserialize(deserializer)?))
    }
}

/// Specular lobe weight in `[0, 1]` (`0` off, `1` full).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Specular(f32);

impl Specular {
    /// Infallible constructor: clamps into `[0, 1]` (NaN maps to `0`).
    pub fn new(value: f32) -> Self {
        Self(clamp_unit_interval(value))
    }

    /// Raw value in `[0, 1]`.
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl From<Specular> for f32 {
    fn from(value: Specular) -> Self {
        value.get()
    }
}

impl Serialize for Specular {
    /// Wire form is the raw `f32` (same as [`Clamped01`]).
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(self.0)
    }
}

impl<'de> Deserialize<'de> for Specular {
    /// Reads the raw `f32` and clamps it ([`Specular::new`]; NaN → `0`).
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(f32::deserialize(deserializer)?))
    }
}

/// Image-based light mix in `[0, 1]` (`0` direct light only, `1` full IBL).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EnvironmentWeight(f32);

impl EnvironmentWeight {
    /// Infallible constructor: clamps into `[0, 1]` (NaN maps to `0`).
    pub fn new(value: f32) -> Self {
        Self(clamp_unit_interval(value))
    }

    /// Raw value in `[0, 1]`.
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl From<EnvironmentWeight> for f32 {
    fn from(value: EnvironmentWeight) -> Self {
        value.get()
    }
}

impl Serialize for EnvironmentWeight {
    /// Wire form is the raw `f32` (same as [`Clamped01`]).
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(self.0)
    }
}

impl<'de> Deserialize<'de> for EnvironmentWeight {
    /// Reads the raw `f32` and clamps it ([`EnvironmentWeight::new`]; NaN → `0`).
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(f32::deserialize(deserializer)?))
    }
}

/// Authored override of roughness, metalness, and specular weight.
///
/// Lives in `ornis-core` so `ornis-app` can stamp it onto primitives and
/// `ornis-render` can read the same lane. Color, IOR, and emission are
/// not fields: those stay on the source material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Surface {
    /// Specular roughness written to `specular.params[1]`.
    pub roughness: Roughness,
    /// Metalness written to `base.params[2]`.
    pub metallic: Metallic,
    /// Specular lobe weight written to `specular.params[0]`.
    pub specular: Specular,
}

/// Linear-space RGB color (albedo, tint, emission; values may exceed `1`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LinearRgb(pub [f32; 3]);

impl LinearRgb {
    /// Black.
    pub const BLACK: Self = Self([0.0, 0.0, 0.0]);

    /// White.
    pub const WHITE: Self = Self([1.0, 1.0, 1.0]);

    /// Wraps raw linear RGB channels without validation.
    pub const fn new(rgb: [f32; 3]) -> Self {
        Self(rgb)
    }

    /// Raw channels.
    pub const fn as_array(self) -> [f32; 3] {
        self.0
    }

    /// Extends with an alpha channel.
    pub const fn with_alpha(self, alpha: f32) -> LinearRgba {
        LinearRgba([self.0[0], self.0[1], self.0[2], alpha])
    }
}

impl From<[f32; 3]> for LinearRgb {
    fn from(value: [f32; 3]) -> Self {
        Self::new(value)
    }
}

impl From<LinearRgb> for [f32; 3] {
    fn from(value: LinearRgb) -> Self {
        value.as_array()
    }
}

/// Linear-space RGBA color (base color, emission with alpha).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LinearRgba(pub [f32; 4]);

impl LinearRgba {
    /// Transparent black.
    pub const TRANSPARENT: Self = Self([0.0, 0.0, 0.0, 0.0]);

    /// Opaque white.
    pub const WHITE: Self = Self([1.0, 1.0, 1.0, 1.0]);

    /// Wraps raw linear RGBA channels without validation.
    pub const fn new(rgba: [f32; 4]) -> Self {
        Self(rgba)
    }

    /// Raw channels.
    pub const fn as_array(self) -> [f32; 4] {
        self.0
    }

    /// RGB channels without alpha.
    pub const fn rgb(self) -> LinearRgb {
        LinearRgb([self.0[0], self.0[1], self.0[2]])
    }

    /// Opaque version of an RGB color.
    pub const fn opaque(rgb: LinearRgb) -> Self {
        rgb.with_alpha(1.0)
    }
}

impl From<[f32; 4]> for LinearRgba {
    fn from(value: [f32; 4]) -> Self {
        Self::new(value)
    }
}

impl From<LinearRgba> for [f32; 4] {
    fn from(value: LinearRgba) -> Self {
        value.as_array()
    }
}

impl From<LinearRgb> for [f32; 4] {
    fn from(value: LinearRgb) -> Self {
        LinearRgba::opaque(value).as_array()
    }
}

impl From<LinearRgb> for LinearRgba {
    fn from(value: LinearRgb) -> Self {
        Self::opaque(value)
    }
}

impl From<Clamped01> for f64 {
    fn from(value: Clamped01) -> Self {
        value.get() as f64
    }
}

/// Squared length below which a vector/quaternion is treated as degenerate.
const DEGENERATE_LEN2: f32 = 1e-12;

/// Rejected [`UnitVec3`] input (must be finite and non-zero).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("vector must be finite and non-zero")]
pub struct UnitVec3Error;

/// Unit-length direction vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitVec3(Vec3);

impl UnitVec3 {
    /// Tolerance for the already-normalized check.
    pub const UNIT_TOL: f32 = 1e-4;

    /// Axis constants.
    pub const X: Self = Self(Vec3::X);
    /// Axis constants.
    pub const Y: Self = Self(Vec3::Y);
    /// Axis constants.
    pub const Z: Self = Self(Vec3::Z);

    /// Normalizes `value` into a unit direction.
    ///
    /// # Errors
    ///
    /// [`UnitVec3Error`] when `value` is zero or any component is non-finite.
    ///
    /// [`Self::try_from_vec`] rejects vectors that are not already unit
    /// length. This constructor rescales any finite non-zero input, which is
    /// what `UnitVec3::new(Vec3::new(-1.0, -1.0, -1.0))?` needs.
    pub fn new(value: Vec3) -> Result<Self, UnitVec3Error> {
        Self::normalize(value).ok_or(UnitVec3Error)
    }

    /// Accepts only finite vectors already of unit length (within tolerance).
    pub fn try_from_vec(value: Vec3) -> Result<Self, UnitVec3Error> {
        if !value.is_finite() {
            return Err(UnitVec3Error);
        }
        let len = value.length();
        if !len.is_finite() || (len - 1.0).abs() > Self::UNIT_TOL {
            return Err(UnitVec3Error);
        }
        Ok(Self(value / len))
    }

    /// Normalizes any finite non-zero vector; `None` on zero/non-finite input.
    pub fn normalize(value: Vec3) -> Option<Self> {
        if !value.is_finite() || value.length_squared() < DEGENERATE_LEN2 {
            return None;
        }
        Some(Self(value.normalize()))
    }

    /// Raw unit vector.
    pub const fn get(self) -> Vec3 {
        self.0
    }

    /// Raw unit vector as an array.
    pub fn as_array(self) -> [f32; 3] {
        self.0.to_array()
    }
}

impl TryFrom<Vec3> for UnitVec3 {
    type Error = UnitVec3Error;

    fn try_from(value: Vec3) -> Result<Self, Self::Error> {
        Self::try_from_vec(value)
    }
}

impl From<UnitVec3> for Vec3 {
    fn from(value: UnitVec3) -> Self {
        value.get()
    }
}

impl Serialize for UnitVec3 {
    /// Wire form is the unit vector as `[f32; 3]`.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_array().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for UnitVec3 {
    /// Reads `[f32; 3]` and normalizes it, so a scene direction such as
    /// `(1, 1, 1)` still loads. A later save writes the normalized
    /// components.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let xyz = <[f32; 3]>::deserialize(deserializer)?;
        Self::new(Vec3::from_array(xyz)).map_err(D::Error::custom)
    }
}

/// Rejected [`UnitQuat`] input (must be finite and of unit length).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitQuatError;

/// Unit rotation quaternion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitQuat(Quat);

impl UnitQuat {
    /// Tolerance for the already-normalized check.
    pub const UNIT_TOL: f32 = 1e-4;

    /// Identity rotation.
    pub const IDENTITY: Self = Self(Quat::IDENTITY);

    /// Accepts only finite quaternions already of unit length (within tolerance).
    pub fn try_from_quat(value: Quat) -> Result<Self, UnitQuatError> {
        if !value.is_finite() {
            return Err(UnitQuatError);
        }
        let len = value.length();
        if !len.is_finite() || (len - 1.0).abs() > Self::UNIT_TOL {
            return Err(UnitQuatError);
        }
        Ok(Self((value / len).normalize()))
    }

    /// Normalizes any finite non-degenerate quaternion; `None` otherwise.
    pub fn normalize(value: Quat) -> Option<Self> {
        if !value.is_finite() || value.length_squared() < DEGENERATE_LEN2 {
            return None;
        }
        Some(Self(value.normalize()))
    }

    /// Raw unit quaternion.
    pub const fn get(self) -> Quat {
        self.0
    }
}

impl TryFrom<Quat> for UnitQuat {
    type Error = UnitQuatError;

    fn try_from(value: Quat) -> Result<Self, Self::Error> {
        Self::try_from_quat(value)
    }
}

impl From<UnitQuat> for Quat {
    fn from(value: UnitQuat) -> Self {
        value.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seconds_validates_steps() {
        assert!(Seconds::new(1.0 / 60.0).is_valid_step());
        assert!(!Seconds::new(0.0).is_valid_step());
        assert!(!Seconds::new(-0.1).is_valid_step());
        assert!(!Seconds::new(f32::NAN).is_valid_step());
        assert!(!Seconds::new(f32::INFINITY).is_valid_step());
    }

    #[test]
    fn hertz_period_needs_live_spring() {
        assert_eq!(Hertz::new(2.0).period(), Some(Seconds::new(0.5)));
        assert_eq!(Hertz::new(0.0).period(), None);
        assert_eq!(Hertz::new(-1.0).period(), None);
        assert_eq!(Hertz::new(f32::NAN).period(), None);
    }

    #[test]
    fn degrees_radians_round_trip() {
        let deg = Degrees::new(180.0);
        let rad = Radians::from(deg);
        assert!((rad.get() - std::f32::consts::PI).abs() < 1e-5);
        let back = Degrees::from(rad);
        assert!((back.get() - 180.0).abs() < 1e-3);
    }

    #[test]
    fn positive_rejects_non_positive() {
        assert!(PositiveF32::try_new(1.0).is_some());
        assert!(PositiveF32::try_new(0.0).is_none());
        assert!(PositiveF32::try_new(-1.0).is_none());
        assert!(PositiveF32::try_new(f32::NAN).is_none());
        assert!(PositiveF32::try_from(2.0).is_ok());
        assert!(PositiveF32::try_from(0.0).is_err());
        assert_eq!(PositiveF32::expect_valid(2.0).get(), 2.0);
    }

    #[test]
    fn ior_clamps_up_and_defaults() {
        assert_eq!(Ior::new(1.33).get(), 1.33);
        assert_eq!(Ior::new(0.5).get(), 1.0);
        assert_eq!(Ior::new(f32::NAN).get(), 1.5);
        assert_eq!(Ior::default().get(), 1.5);
        assert!(Ior::try_new(1.0).is_some());
        assert!(Ior::try_new(0.9).is_none());
        assert!(Ior::try_from(2.0).is_ok());
        assert!(Ior::try_from(0.5).is_err());
    }

    #[test]
    fn clamped01_clamps_and_checks() {
        assert_eq!(Clamped01::new(1.5).get(), 1.0);
        assert_eq!(Clamped01::new(-0.5).get(), 0.0);
        assert!(Clamped01::try_new(0.5).is_some());
        assert!(Clamped01::try_new(1.5).is_none());
        let as_f32: f32 = Clamped01::new(0.25).into();
        assert_eq!(as_f32, 0.25);
    }

    #[test]
    fn surface_unit_interval_clamps_and_maps_nan_to_zero() {
        fn check(new: fn(f32) -> f32) {
            assert_eq!(new(0.25), 0.25);
            assert_eq!(new(0.0), 0.0);
            assert_eq!(new(1.0), 1.0);
            assert_eq!(new(1.5), 1.0);
            assert_eq!(new(-0.5), 0.0);
            assert_eq!(new(f32::NAN), 0.0);
            assert_eq!(new(f32::INFINITY), 1.0);
            assert_eq!(new(f32::NEG_INFINITY), 0.0);
        }
        check(|v| Roughness::new(v).get());
        check(|v| Metallic::new(v).get());
        check(|v| Specular::new(v).get());
        check(|v| EnvironmentWeight::new(v).get());
    }

    #[test]
    fn metallic_ron_roundtrip_is_a_bare_f32() {
        let value = Metallic::new(0.35);
        let text = ron::ser::to_string(&value).expect("serialize");
        assert_eq!(text, ron::ser::to_string(&0.35_f32).expect("f32"));
        let back: Metallic = ron::de::from_str(&text).expect("roundtrip");
        assert_eq!(back, value);
        assert_eq!(back.get(), 0.35);
        let high: Metallic = ron::de::from_str("1.7").expect("over");
        assert_eq!(high.get(), 1.0);
        let low: Metallic = ron::de::from_str("-0.2").expect("under");
        assert_eq!(low.get(), 0.0);
        let nan: Metallic = ron::de::from_str("NaN").expect("nan");
        assert_eq!(nan.get(), 0.0);
    }

    #[test]
    fn colors_convert_losslessly() {
        let rgb = LinearRgb::new([0.5, 0.25, 0.125]);
        let arr: [f32; 3] = rgb.into();
        assert_eq!(arr, [0.5, 0.25, 0.125]);
        let rgba = rgb.with_alpha(0.5);
        assert_eq!(rgba.as_array(), [0.5, 0.25, 0.125, 0.5]);
        assert_eq!(rgba.rgb(), rgb);
    }

    #[test]
    fn unit_vec3_accepts_only_unit_or_normalizable() {
        assert!(UnitVec3::try_from_vec(Vec3::X).is_ok());
        assert!(UnitVec3::try_from_vec(Vec3::ZERO).is_err());
        assert!(UnitVec3::try_from_vec(Vec3::splat(2.0)).is_err());
        assert!(UnitVec3::new(Vec3::new(2.0, 0.0, 0.0)).is_ok());
        assert!(UnitVec3::new(Vec3::ZERO).is_err());
        let diagonal = UnitVec3::new(Vec3::new(-1.0, -1.0, -1.0)).expect("non-zero");
        assert!((diagonal.get().length() - 1.0).abs() < 1e-5);
        assert!(diagonal.get().x < 0.0);
        assert!(UnitVec3::normalize(Vec3::new(2.0, 0.0, 0.0)).is_some());
        assert!(UnitVec3::normalize(Vec3::ZERO).is_none());
        let v: Vec3 = UnitVec3::X.into();
        assert_eq!(v, Vec3::X);
    }

    #[test]
    fn degrees_and_unit_vec3_match_scene_ron() {
        #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        struct Wire {
            direction: UnitVec3,
            fov: Degrees,
        }

        assert_eq!(
            ron::ser::to_string(&Degrees(45.0)).unwrap(),
            ron::ser::to_string(&45.0_f32).unwrap()
        );
        assert_eq!(
            ron::ser::to_string(&UnitVec3::X).unwrap(),
            ron::ser::to_string(&[1.0_f32, 0.0, 0.0]).unwrap()
        );
        let parsed: Wire =
            ron::de::from_str("(direction: (1.0, 1.0, 1.0), fov: 60.0)").expect("scene tuple");
        assert!((parsed.direction.get().length() - 1.0).abs() < 1e-5);
        assert_eq!(parsed.fov.get(), 60.0);
        let again: Wire = ron::de::from_str(&ron::ser::to_string(&parsed).unwrap()).unwrap();
        assert_eq!(again.fov, parsed.fov);
        assert!((again.direction.get() - parsed.direction.get()).length() < 1e-5);
        assert!(ron::de::from_str::<UnitVec3>("(0.0, 0.0, 0.0)").is_err());
    }

    #[test]
    fn unit_quat_accepts_only_unit_or_normalizable() {
        assert!(UnitQuat::try_from_quat(Quat::IDENTITY).is_ok());
        assert!(UnitQuat::try_from_quat(Quat::from_array([0.0, 0.0, 0.0, 0.0])).is_err());
        let doubled = Quat::from_array([0.0, 0.0, 0.0, 2.0]);
        assert!(UnitQuat::try_from_quat(doubled).is_err());
        assert!(UnitQuat::normalize(doubled).is_some());
        let q: Quat = UnitQuat::IDENTITY.into();
        assert_eq!(q, Quat::IDENTITY);
    }

    #[test]
    fn angular_speed_wraps_raw() {
        let w = RadiansPerSecond::new(1.5);
        assert!(w.is_finite());
        assert_eq!(f32::from(w), 1.5);
        assert_eq!(RadiansPerSecond::from(2.0).get(), 2.0);
        assert!(!RadiansPerSecond::new(f32::NAN).is_finite());
    }

    #[test]
    fn gear_ratio_rejects_zero_and_non_finite() {
        assert!(GearRatio::try_new(2.0).is_some());
        assert!(GearRatio::try_new(0.0).is_none());
        assert!(GearRatio::try_new(f32::NAN).is_none());
        assert!(GearRatio::try_new(f32::INFINITY).is_none());
        assert_eq!(f32::from(GearRatio::try_new(-1.5).expect("valid")), -1.5);
    }

    #[test]
    fn kilograms_classifies_static() {
        assert!(Kilograms::new(2.0).as_positive().is_some());
        assert!(Kilograms::ZERO.is_static());
        assert!(Kilograms::new(-1.0).is_static());
        assert!(!Kilograms::new(2.0).is_static());
    }
}
