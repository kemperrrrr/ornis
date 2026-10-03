//! Invariant-preserving newtypes for the physics pipeline.
//!
//! Comment-documented contracts (`mass`/`inv_mass` consistency, `len ==
//! rows * cols` heightfields, 1..=4 manifold points, unit directions)
//! become construction-time types here: fallible constructors return
//! `None`/`Err` instead of silently substituting a default.

use glam::Vec3;
use thiserror::Error;

use crate::constants::DEGENERATE_LEN2;
use crate::shape::Shape;

/// Positive finite `f32`: mass values, spacings, radii inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositiveF32(f32);

impl PositiveF32 {
    /// Checked constructor: finite and `> 0`.
    pub fn try_new(v: f32) -> Option<Self> {
        if v.is_finite() && v > 0.0 {
            Some(Self(v))
        } else {
            None
        }
    }

    /// Raw value.
    pub fn get(self) -> f32 {
        self.0
    }
}

/// Unit-length direction vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitVec3(Vec3);

impl UnitVec3 {
    /// Tolerance for the already-normalized check.
    pub const UNIT_TOL: f32 = 1e-4;

    /// Accept only finite vectors already of unit length (within tolerance).
    pub fn try_from(v: Vec3) -> Option<Self> {
        if !v.is_finite() {
            return None;
        }
        let len2 = v.length_squared();
        if !len2.is_finite() || (len2.sqrt() - 1.0).abs() > Self::UNIT_TOL {
            return None;
        }
        Some(Self(v / len2.sqrt()))
    }

    /// Normalize any finite non-zero vector; `None` on zero/non-finite input.
    pub fn normalize_checked(v: Vec3) -> Option<Self> {
        if !v.is_finite() || v.length_squared() < DEGENERATE_LEN2 {
            return None;
        }
        Some(Self(v.normalize()))
    }

    /// Raw unit vector.
    pub fn get(self) -> Vec3 {
        self.0
    }

    /// Wraps an already-unit vector without checking (snapshot restore:
    /// the stored normal came from a validated [`UnitVec3`], so
    /// re-normalizing would cost a bit of precision for nothing).
    ///
    /// # Safety
    ///
    /// The caller guarantees `v` is finite and of unit length; a
    /// non-unit input silently propagates the invariant break into every
    /// downstream query.
    pub(crate) unsafe fn from_unit_unchecked(v: Vec3) -> Self {
        Self(v)
    }
}

/// Length in meters (joint references, rest distances).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Meters(pub f32);

impl Meters {
    /// Checked constructor: finite value.
    pub fn try_new(v: f32) -> Option<Self> {
        if v.is_finite() { Some(Self(v)) } else { None }
    }

    /// Raw value.
    pub fn get(self) -> f32 {
        self.0
    }
}

/// Angle in radians (hinge twists, reference angles).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Radians(pub f32);

impl Radians {
    /// Checked constructor: finite value.
    pub fn try_new(v: f32) -> Option<Self> {
        if v.is_finite() { Some(Self(v)) } else { None }
    }

    /// Raw value.
    pub fn get(self) -> f32 {
        self.0
    }
}

/// Mass specification for rigid/soft construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MassKind {
    /// Infinite mass (static anchor, pinned particle).
    Fixed,
    /// Finite positive mass.
    Free(PositiveF32),
}

impl MassKind {
    /// Classify a raw mass: `<= 0`/non-finite pins (soft parity) or is
    /// fixed (rigid parity); positive finite mass is free.
    ///
    /// Never fails: non-positive input is a valid pinned/fixed state, not
    /// an error. Use [`MassKind::try_free`] when any non-positive input
    /// is a caller error.
    pub fn from_f32(mass: f32) -> Self {
        match PositiveF32::try_new(mass) {
            Some(m) => Self::Free(m),
            None if mass <= 0.0 => Self::Fixed,
            None => Self::Fixed,
        }
    }

    /// Strict constructor: `None` unless the mass is positive and finite.
    pub fn try_free(mass: f32) -> Option<Self> {
        PositiveF32::try_new(mass).map(Self::Free)
    }

    /// Whether this kind integrates.
    pub fn is_fixed(self) -> bool {
        matches!(self, Self::Fixed)
    }

    /// Raw mass value (0 for fixed).
    pub fn mass_value(self) -> f32 {
        match self {
            Self::Fixed => 0.0,
            Self::Free(m) => m.get(),
        }
    }

    /// Cached inverse mass (0 for fixed).
    pub fn inv_mass_value(self) -> f32 {
        match self {
            Self::Fixed => 0.0,
            Self::Free(m) => 1.0 / m.get(),
        }
    }
}

/// Consistent mass triple: `inv == 1/mass` (or 0 when fixed) and the
/// shape-derived inertia. Only constructible via the checked constructors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mass {
    mass: f32,
    inv_mass: f32,
    inertia: Vec3,
}

impl Mass {
    /// Fixed (infinite-mass) triple for `shape`.
    pub fn fixed(shape: &Shape) -> Self {
        Self {
            mass: 0.0,
            inv_mass: 0.0,
            inertia: shape.inertia(0.0),
        }
    }

    /// Dynamic triple for a positive mass and `shape`.
    pub fn dynamic(mass: PositiveF32, shape: &Shape) -> Self {
        let m = mass.get();
        Self {
            mass: m,
            inv_mass: 1.0 / m,
            inertia: shape.inertia(m),
        }
    }

    /// Triple from a [`MassKind`] and `shape`.
    pub fn from_kind(kind: MassKind, shape: &Shape) -> Self {
        match kind {
            MassKind::Fixed => Self::fixed(shape),
            MassKind::Free(m) => Self::dynamic(m, shape),
        }
    }

    /// Total mass (kg, 0 when fixed).
    pub fn mass(self) -> f32 {
        self.mass
    }

    /// Cached `1 / mass` (0 when fixed).
    pub fn inv_mass(self) -> f32 {
        self.inv_mass
    }

    /// Diagonal body-frame inertia.
    pub fn inertia(self) -> Vec3 {
        self.inertia
    }

    /// Whether this triple is fixed.
    pub fn is_fixed(self) -> bool {
        self.mass <= 0.0
    }
}

/// Friction direction frame: isotropic or a single unit anisotropy axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrictionFrame {
    /// Default circular Coulomb cone.
    Isotropic,
    /// Elliptical cone with the first tangent fixed along the axis.
    Aniso(UnitVec3),
}

/// Rejected `friction_dir` input: zero-length or non-finite vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("friction direction must be a finite non-zero vector")]
pub struct FrictionFrameError;

impl FrictionFrame {
    /// Explicit mapping from the legacy `Option<Vec3>`: `None` is
    /// isotropic, `Some` must normalize to a unit axis.
    pub fn from_option(dir: Option<Vec3>) -> Result<Self, FrictionFrameError> {
        match dir {
            None => Ok(Self::Isotropic),
            Some(v) => UnitVec3::normalize_checked(v)
                .map(Self::Aniso)
                .ok_or(FrictionFrameError),
        }
    }

    /// Fixed axis, if anisotropic.
    pub fn axis(self) -> Option<UnitVec3> {
        match self {
            Self::Isotropic => None,
            Self::Aniso(u) => Some(u),
        }
    }
}

/// Capacity of [`Capped4`] / [`NonEmpty4`] (and SI manifolds).
const CAP4: usize = 4;

/// Capped inline vector of at most [`CAP4`] `Copy` points with a count invariant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Capped4<T: Copy> {
    buf: [Option<T>; CAP4],
    len: u8,
}

impl<T: Copy> Default for Capped4<T> {
    fn default() -> Self {
        Self {
            buf: [None; CAP4],
            len: 0,
        }
    }
}

impl<T: Copy> Capped4<T> {
    /// Empty vector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a point; `Err(point)` when full.
    pub fn try_push(&mut self, v: T) -> Result<(), T> {
        if self.len as usize >= CAP4 {
            return Err(v);
        }
        self.buf[self.len as usize] = Some(v);
        self.len += 1;
        Ok(())
    }

    /// Checked construction from a slice: `None` when longer than 4
    /// (empty input yields an empty vector; [`NonEmpty4`] is the
    /// non-empty counterpart).
    pub fn try_from_slice(v: &[T]) -> Option<Self> {
        if v.len() > CAP4 {
            return None;
        }
        let mut out = Self::new();
        for x in v {
            // Length checked above, so the push cannot fail.
            let _ = out.try_push(*x);
        }
        Some(out)
    }

    /// Iterate live points.
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.buf[..self.len as usize].iter().filter_map(|o| *o)
    }

    /// Number of live points.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether no points are stored.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Live point by index.
    pub fn get(&self, i: usize) -> Option<T> {
        if i < self.len as usize {
            self.buf[i]
        } else {
            None
        }
    }
}

/// Non-empty capped vector of 1..=4 `Copy` points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonEmpty4<T: Copy> {
    inner: Capped4<T>,
}

impl<T: Copy> NonEmpty4<T> {
    /// Single-point vector.
    pub fn single(v: T) -> Self {
        let mut inner = Capped4::new();
        let _ = inner.try_push(v);
        Self { inner }
    }

    /// Checked construction from a slice: `None` when empty or longer than 4.
    pub fn try_from_slice(v: &[T]) -> Option<Self> {
        if v.is_empty() || v.len() > CAP4 {
            return None;
        }
        let mut inner = Capped4::new();
        for x in v {
            let _ = inner.try_push(*x);
        }
        Some(Self { inner })
    }

    /// Push an additional point; `Err(point)` when already full.
    pub fn try_push(&mut self, v: T) -> Result<(), T> {
        self.inner.try_push(v)
    }

    /// Number of live points (1..=4).
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether no points are stored (always `false`: construction
    /// requires at least one point).
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Live point by index.
    pub fn get(&self, i: usize) -> Option<T> {
        self.inner.get(i)
    }

    /// Iterate live points.
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.inner.iter()
    }
}

/// Rejected heightfield description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum HeightfieldError {
    /// `rows == 0` or `cols == 0`.
    #[error("heightfield grid is empty (rows == 0 or cols == 0)")]
    EmptyGrid,
    /// Non-positive or non-finite cell spacing.
    #[error("heightfield cell must be positive finite")]
    BadCell,
    /// `heights.len() != rows * cols` (includes overflow).
    #[error("heightfield heights length does not match rows * cols")]
    LengthMismatch,
    /// Non-finite height sample.
    #[error("heightfield has a non-finite height sample")]
    NonFiniteHeight,
}

/// Validate a heightfield description without allocating.
pub fn validate_heightfield(
    heights: &[f32],
    rows: usize,
    cols: usize,
    cell: f32,
) -> Result<(), HeightfieldError> {
    if rows == 0 || cols == 0 {
        return Err(HeightfieldError::EmptyGrid);
    }
    if !cell.is_finite() || cell <= 0.0 {
        return Err(HeightfieldError::BadCell);
    }
    let want = rows
        .checked_mul(cols)
        .ok_or(HeightfieldError::LengthMismatch)?;
    if heights.len() != want {
        return Err(HeightfieldError::LengthMismatch);
    }
    if heights.iter().any(|h| !h.is_finite()) {
        return Err(HeightfieldError::NonFiniteHeight);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_rejects_non_positive_and_non_finite() {
        assert!(PositiveF32::try_new(1.0).is_some());
        assert!(PositiveF32::try_new(0.0).is_none());
        assert!(PositiveF32::try_new(-2.0).is_none());
        assert!(PositiveF32::try_new(f32::NAN).is_none());
        assert!(PositiveF32::try_new(f32::INFINITY).is_none());
    }

    #[test]
    fn unit_accepts_only_unit_or_normalizable() {
        assert!(UnitVec3::try_from(Vec3::X).is_some());
        assert!(UnitVec3::try_from(Vec3::ZERO).is_none());
        assert!(UnitVec3::try_from(Vec3::splat(2.0)).is_none());
        assert!(UnitVec3::normalize_checked(Vec3::new(2.0, 0.0, 0.0)).is_some());
        assert!(UnitVec3::normalize_checked(Vec3::ZERO).is_none());
    }

    #[test]
    fn capped4_enforces_count_invariant() {
        let mut v = Capped4::new();
        for i in 0..4 {
            assert!(v.try_push(i).is_ok());
        }
        assert!(v.try_push(4).is_err());
        assert_eq!(v.len(), 4);
        assert_eq!(v.iter().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        assert!(Capped4::try_from_slice(&[1, 2, 3, 4, 5]).is_none());
        assert_eq!(
            Capped4::try_from_slice(&[1, 2])
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!(NonEmpty4::<i32>::try_from_slice(&[]).is_none());
        assert!(NonEmpty4::<i32>::try_from_slice(&[1, 2, 3, 4, 5]).is_none());
        assert_eq!(NonEmpty4::single(7).len(), 1);
    }

    #[test]
    fn invariant_errors_report_a_reason() {
        use std::error::Error as _;
        let friction = FrictionFrameError;
        assert!(!friction.to_string().is_empty());
        assert!(friction.source().is_none());
        assert!(!HeightfieldError::EmptyGrid.to_string().is_empty());
        assert!(!HeightfieldError::BadCell.to_string().is_empty());
        assert!(!HeightfieldError::LengthMismatch.to_string().is_empty());
        assert!(!HeightfieldError::NonFiniteHeight.to_string().is_empty());
    }

    #[test]
    fn heightfield_validation_catches_mismatch() {
        assert!(validate_heightfield(&[0.0; 6], 2, 3, 1.0).is_ok());
        assert_eq!(
            validate_heightfield(&[0.0; 5], 2, 3, 1.0),
            Err(HeightfieldError::LengthMismatch)
        );
        assert_eq!(
            validate_heightfield(&[0.0; 6], 0, 3, 1.0),
            Err(HeightfieldError::EmptyGrid)
        );
        assert_eq!(
            validate_heightfield(&[0.0; 6], 2, 3, 0.0),
            Err(HeightfieldError::BadCell)
        );
        assert_eq!(
            validate_heightfield(&[f32::NAN; 6], 2, 3, 1.0),
            Err(HeightfieldError::NonFiniteHeight)
        );
    }
}
