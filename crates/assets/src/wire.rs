//! `serde(with = ...)` adapters that keep the scene wire schema unchanged.
//!
//! The typed descriptor fields ([`Vec3`], [`UnitVec3`], [`UnitQuat`],
//! [`Meters`], [`Degrees`]) serialize through the exact legacy shapes —
//! `[f32; 3]`, `[f32; 4]` in `(x, y, z, w)` order and plain `f32` — so RON
//! files (`(1.0, 2.0, 3.0)`) and the JSON editor/WASM protocol
//! (`[1.0, 2.0, 3.0]`) stay byte-compatible.
//!
//! Invariant policy on read: unit types are **normalized**, not rejected,
//! when finite and non-degenerate (old files carry e.g. `direction:
//! (1.0, 1.0, 1.0)` and hand-typed `0.7071` quaternions); zero-length or
//! non-finite directions/quaternions are a deserialization **error** —
//! there is no meaningful direction or rotation to recover.
//!
//! Normalization is made idempotent ([`stable_unit_vec3`],
//! [`stable_unit_quat`]): a single `v / |v|` can move an already-unit value
//! by an ULP and even alternate (`0.57735026` ↔ `0.5773503`), which would
//! make every load/save cycle rewrite the file (the `ornis_core` unit
//! constructors always re-divide, so "keep if already unit" is not
//! available). Choosing a canonical member of that cycle (smallest bit
//! pattern) guarantees `parse(serialize(x)) == x` for every loaded value;
//! for `(1, 1, 1)` that member is exactly `Vec3::normalize((1, 1, 1))`.

use glam::{Quat, Vec3};
use ornis_core::units::{Degrees, Meters, UnitQuat, UnitVec3};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Bound on re-normalization rounds while searching the cycle.
const MAX_RENORMALIZE: usize = 8;

/// Canonical member of the re-normalization cycle reached from `first`.
///
/// `step` (a plain `v / |v|`) does not always have a fixed point in `f32`:
/// it can alternate between two neighbours (`0.57735026` ↔ `0.5773503`).
/// The cycle is a property of the value, so picking its smallest member
/// (by bit pattern) gives `canon(canon(x)) == canon(x)`: a loaded value
/// re-loads to itself.
fn canonical_cycle_member<T: Copy + PartialEq>(
    first: T,
    step: impl Fn(T) -> Option<T>,
    key: impl Fn(T) -> Vec<u32>,
) -> Option<T> {
    let mut seen = vec![first];
    for _ in 0..MAX_RENORMALIZE {
        let next = step(*seen.last()?)?;
        if let Some(start) = seen.iter().position(|v| *v == next) {
            return seen[start..].iter().copied().min_by_key(|v| key(*v));
        }
        seen.push(next);
    }
    // No cycle within the bound (not observed in practice): last value.
    seen.last().copied()
}

/// Normalizes to a value that re-loads to itself (stable under
/// load/save); `None` for zero/non-finite input.
pub(crate) fn stable_unit_vec3(value: Vec3) -> Option<UnitVec3> {
    canonical_cycle_member(
        UnitVec3::normalize(value)?,
        |unit| UnitVec3::normalize(unit.get()),
        |unit| unit.as_array().map(f32::to_bits).to_vec(),
    )
}

/// Quaternion counterpart of [`stable_unit_vec3`].
pub(crate) fn stable_unit_quat(value: Quat) -> Option<UnitQuat> {
    canonical_cycle_member(
        UnitQuat::normalize(value)?,
        |unit| UnitQuat::normalize(unit.get()),
        |unit| unit.get().to_array().map(f32::to_bits).to_vec(),
    )
}

/// [`Vec3`] as `[f32; 3]`.
pub(crate) mod vec3 {
    use super::*;

    pub(crate) fn serialize<S: Serializer>(value: &Vec3, serializer: S) -> Result<S::Ok, S::Error> {
        value.to_array().serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec3, D::Error> {
        <[f32; 3]>::deserialize(deserializer).map(Vec3::from_array)
    }
}

/// [`UnitVec3`] as `[f32; 3]`; normalizes on read, rejects zero/non-finite.
pub(crate) mod unit_vec3 {
    use super::*;

    pub(crate) fn serialize<S: Serializer>(
        value: &UnitVec3,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.as_array().serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<UnitVec3, D::Error> {
        let raw = <[f32; 3]>::deserialize(deserializer)?;
        stable_unit_vec3(Vec3::from_array(raw)).ok_or_else(|| {
            D::Error::custom(format!(
                "direction {raw:?} must be finite and non-zero (it is normalized on load)"
            ))
        })
    }
}

/// [`UnitQuat`] as `[x, y, z, w]`; normalizes on read, rejects
/// zero/non-finite.
pub(crate) mod unit_quat {
    use super::*;

    pub(crate) fn serialize<S: Serializer>(
        value: &UnitQuat,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.get().to_array().serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<UnitQuat, D::Error> {
        let raw = <[f32; 4]>::deserialize(deserializer)?;
        stable_unit_quat(Quat::from_array(raw)).ok_or_else(|| {
            D::Error::custom(format!(
                "rotation {raw:?} (x, y, z, w) must be finite and non-zero (it is normalized on load)"
            ))
        })
    }
}

/// [`Meters`] as plain `f32`.
pub(crate) mod meters {
    use super::*;

    pub(crate) fn serialize<S: Serializer>(
        value: &Meters,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.get().serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Meters, D::Error> {
        f32::deserialize(deserializer).map(Meters::new)
    }
}

/// [`Degrees`] as plain `f32`.
pub(crate) mod degrees {
    use super::*;

    pub(crate) fn serialize<S: Serializer>(
        value: &Degrees,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.get().serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Degrees, D::Error> {
        f32::deserialize(deserializer).map(Degrees::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::approx_constant)] // hand-typed `0.7071` is the point
    fn stable_normalization_is_a_fixed_point() {
        for raw in [
            [1.0, 1.0, 1.0],
            [-0.5, 0.5, -0.5],
            [0.3, -7.0, 2.2],
            [1e-3, 2e-3, -5e-4],
        ] {
            let unit = stable_unit_vec3(Vec3::from_array(raw)).expect("non-zero");
            assert_eq!(stable_unit_vec3(unit.get()), Some(unit), "{raw:?}");
            assert!((unit.get().length() - 1.0).abs() < 1e-6);
        }
        for raw in [
            [0.0, 0.7071, 0.0, 0.7071],
            [1.0, 2.0, 3.0, 4.0],
            [0.1, 0.1, 0.1, 0.9],
        ] {
            let unit = stable_unit_quat(Quat::from_array(raw)).expect("non-zero");
            assert_eq!(stable_unit_quat(unit.get()), Some(unit), "{raw:?}");
        }
        // `(1, 1, 1)` loads to exactly one plain `v / |v|` (what the
        // renderer computed from the raw value before typing); hand-typed
        // near-unit values are normalized (to within an ULP).
        assert_eq!(
            stable_unit_vec3(Vec3::ONE).map(UnitVec3::get),
            Some(Vec3::ONE.normalize())
        );
        let hand = Quat::from_array([0.0, 0.7071, 0.0, 0.7071]);
        let loaded = stable_unit_quat(hand).expect("non-zero").get();
        assert!((loaded.length() - 1.0).abs() < 1e-6);
        assert!(loaded.abs_diff_eq(hand.normalize(), 1e-6));
        assert!(stable_unit_vec3(Vec3::ZERO).is_none());
        assert!(stable_unit_quat(Quat::from_array([0.0; 4])).is_none());
        assert!(stable_unit_vec3(Vec3::new(f32::NAN, 0.0, 0.0)).is_none());
    }
}
