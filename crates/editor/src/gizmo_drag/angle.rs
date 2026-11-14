//! Rotation-angle recovery for gizmo snap (PLAN §i, slice E4): the absolute
//! twist of an orientation about a handle axis, and the increment swept since
//! the press.
//!
//! Pure over `Quat`/`Vec3` — the drag owns the handle axis and the snap step,
//! this module only answers "which angle does this orientation mean?".

use glam::{Quat, Vec3};

/// Swing-twist norm below which the twist about the handle axis is undefined
/// (a half-turn about a perpendicular axis).
const TWIST_EPS: f32 = 1e-6;

/// Absolute twist angle (radians) of `rotation` about the unit `axis`: the
/// angle of the twist factor in `q = swing · twist`, where `twist` turns
/// around `axis` and `swing` carries `axis` to `q · axis`.
///
/// The twist's vector part is the projection of `q`'s vector part on `axis`,
/// so the angle is `2·atan2(q.xyz · axis, q.w)` — `atan2` is scale-invariant,
/// hence no normalization. The quaternion sign is canonicalized to `w >= 0`
/// so the angle lies in `[-π, π]` and does not depend on which of `q`/`-q`
/// happens to be stored.
///
/// `None` when the decomposition degenerates: both the projection and `w`
/// vanish for a half-turn about an axis perpendicular to `axis`, where every
/// twist angle is equally valid and no absolute angle exists.
pub(super) fn twist_angle(rotation: Quat, axis: Vec3) -> Option<f32> {
    let mut sine = Vec3::new(rotation.x, rotation.y, rotation.z).dot(axis);
    let mut cosine = rotation.w;
    if cosine < 0.0 {
        sine = -sine;
        cosine = -cosine;
    }
    if sine.hypot(cosine) < TWIST_EPS {
        return None;
    }
    Some(2.0 * sine.atan2(cosine))
}

/// Angle (radians) swept from `start` to `rotation` around the unit `axis`,
/// recovered from the delta quaternion. Always defined, unlike the absolute
/// twist — the fallback for an orientation that has no twist at all.
pub(super) fn swept_angle(rotation: Quat, start: Quat, axis: Vec3) -> f32 {
    let delta = rotation * start.inverse();
    let sine = Vec3::new(delta.x, delta.y, delta.z).dot(axis);
    2.0 * sine.atan2(delta.w)
}
