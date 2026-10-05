//! Kinematic platform movers (R2, Box3D `mover.c` spirit): engine-driven
//! kinematic bodies with plane-solver passenger carrying.
//!
//! A [`KinematicMover`] binds one [`BodyType::Kinematic`](crate::body::BodyType)
//! body and declares its per-step target displacement. The engine — not the
//! host — moves the platform each step and carries contacting dynamic bodies
//! (passengers) through [`solve_mover_planes`], the [`b3SolvePlanes`](https://github.com/erincatto/box3d/blob/main/src/mover.c)
//! iterative projection with a linear slop: the platform carries instead of
//! pressing (penetration is resolved positionally to the slop, never injected
//! as velocity, so there is no false bounce), and tangential motion rides
//! without slipping even below the CCD travel gate where the velocity-field
//! friction path is blind.
//!
//! The displacement flows through the same driver machinery as a host
//! teleport (`prev_pose` baseline → implied velocity above the gate → the
//! kinematic CCD sweep), so movers participate in CCD travel by
//! construction: the platform cannot tunnel past a swept victim, and riding
//! passengers move with it. Passengers are woken through
//! [`wake_island`](SequentialImpulseEngine::wake_island) — the existing
//! island gate, no side channels — and an active displacing mover keeps the
//! world off the fully-sleeping fast path exactly like a teleporting driver.
//!
//! Translation only in v1: rotation stays with the host-driver path (set the
//! pose plus a matching `angular_velocity`, as before). Movers are transient
//! driver state (like the driver velocity fields): world snapshots and
//! solver migrations do not carry them — re-attach after a restore.

use glam::Vec3;

use crate::body::{BodyHandle, BodyType};
use crate::constants::DEGENERATE_LEN2;
use crate::errors::MoverError;

use super::SequentialImpulseEngine;
use super::queries::sweep_gap;

/// Linear slop (m) for the mover plane solve: penetration at or below this
/// is left alone (Box3D `B3_LINEAR_SLOP` parity — same 5 mm as
/// [`CONTACT_BEGIN_SLOP`](crate::trigger::CONTACT_BEGIN_SLOP), coupled by
/// value, not shared: the event slop gates gameplay signals, this one gates
/// positional correction).
pub(crate) const MOVER_SLOP: f32 = 0.005;
/// Plane-solver iteration cap (Box3D `b3SolvePlanes` parity: 20).
const MOVER_PLANE_ITERS: usize = 20;
/// Support cosine: a contact normal this upward (passenger resting on top,
/// ~60° cone) rides the full displacement; side/below contacts only ride
/// the non-separating part plus the penetration guard, so a platform
/// sliding away never drags a bystander along.
const MOVER_SUPPORT_DOT: f32 = 0.5;

/// Stable index of a [`KinematicMover`] inside its owning
/// [`SequentialImpulseEngine`]. Dense like [`BodyHandle`]: removal swaps
/// the last mover into the freed slot, so surviving handles past the
/// removal shift (same convention as bodies).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MoverHandle(u32);

impl MoverHandle {
    /// Wraps a raw `u32` mover index.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` mover index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Mover index as `usize` for table lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for MoverHandle {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for MoverHandle {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

impl From<MoverHandle> for u32 {
    fn from(h: MoverHandle) -> Self {
        h.0
    }
}

impl From<MoverHandle> for usize {
    fn from(h: MoverHandle) -> Self {
        h.0 as usize
    }
}

/// A kinematic platform mover: the engine moves [`body`](Self::body) by
/// [`displacement`](Self::displacement) every step while
/// [`active`](Self::active) and carries contacting passengers with it.
///
/// Set the displacement once for a constant-velocity platform (it persists
/// until changed — the engine applies it every step); zero it (or clear
/// [`active`](Self::active)) to stop. Only finite displacements are
/// admitted (see
/// [`set_mover_displacement`](SequentialImpulseEngine::set_mover_displacement)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicMover {
    /// Driven kinematic platform body.
    pub body: BodyHandle,
    /// Per-step target displacement (m) applied by the engine at the top
    /// of every step. Persists until overwritten.
    pub displacement: Vec3,
    /// Participation flag: an inactive mover is fully ignored (the platform
    /// keeps whatever pose the host gave it). A nonzero displacement on an
    /// active mover keeps the world awake and wakes carried passengers;
    /// zero displacement holds position and lets the world sleep.
    pub active: bool,
}

impl KinematicMover {
    /// Parked mover on `body`: no displacement, inactive. The engine
    /// activates it through
    /// [`set_mover_active`](SequentialImpulseEngine::set_mover_active) /
    /// [`set_mover_displacement`](SequentialImpulseEngine::set_mover_displacement).
    pub fn new(body: BodyHandle) -> Self {
        Self {
            body,
            displacement: Vec3::ZERO,
            active: false,
        }
    }
}

/// One mover contact plane (Box3D `b3CollisionPlane` parity): separation of
/// the carry delta along `normal` must reach `-gap - slop`, i.e. the
/// post-carry gap must stay above `-slop`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MoverPlane {
    /// Unit contact normal (platform surface toward the passenger).
    pub(crate) normal: Vec3,
    /// Signed separation at the carried pose before the carry (m).
    pub(crate) gap: f32,
    /// Accumulated push along `normal` (solver state, reset per solve).
    pub(crate) push: f32,
    /// Accumulated-push clamp (rigid carry: no bound).
    pub(crate) push_limit: f32,
}

impl MoverPlane {
    /// Rigid contact plane: unlimited push (the platform carries, never yields).
    pub(crate) fn rigid(normal: Vec3, gap: f32) -> Self {
        Self {
            normal,
            gap,
            push: 0.0,
            push_limit: f32::INFINITY,
        }
    }
}

/// Iterative plane projection for mover carries (Box3D `b3SolvePlanes`
/// parity): the closest delta to `target` that keeps every plane's
/// post-carry gap above `-slop`. Pushes accumulate per plane clamped to
/// `[0, push_limit]`; the loop exits early once the total push per sweep
/// drops below the slop. Deterministic: plane order is the caller's.
pub(crate) fn solve_mover_planes(target: Vec3, planes: &mut [MoverPlane], slop: f32) -> Vec3 {
    debug_assert!(target.is_finite(), "mover carry target must be finite");
    debug_assert!(slop.is_finite() && slop >= 0.0, "mover slop must hold");
    for plane in planes.iter_mut() {
        plane.push = 0.0;
    }
    let mut delta = target;
    for _ in 0..MOVER_PLANE_ITERS {
        let mut total_push = 0.0f32;
        for plane in planes.iter_mut() {
            // Separation with the slop folded in (mover.c adds
            // `B3_LINEAR_SLOP` to the plane separation the same way).
            let separation = plane.normal.dot(delta) + plane.gap + slop;
            let push = -separation;
            let accumulated = plane.push;
            plane.push = (plane.push + push).clamp(0.0, plane.push_limit);
            let applied = plane.push - accumulated;
            delta += plane.normal * applied;
            total_push += applied.abs();
        }
        if total_push < slop {
            break;
        }
    }
    delta
}

impl SequentialImpulseEngine {
    /// Attach a mover to a kinematic body (parked: inactive, zero
    /// displacement — configure through
    /// [`set_mover_active`](Self::set_mover_active) and
    /// [`set_mover_displacement`](Self::set_mover_displacement)).
    ///
    /// # Errors
    ///
    /// [`MoverError::InvalidBody`] for an out-of-range body,
    /// [`MoverError::NotKinematic`] unless the body is kinematic (dynamics
    /// are simulated, statics never move — neither takes a driver).
    pub fn add_mover(&mut self, body: BodyHandle) -> Result<MoverHandle, MoverError> {
        let b = self
            .bodies
            .get(body.index())
            .ok_or(MoverError::InvalidBody {
                handle: body.index(),
            })?;
        if b.body_type != BodyType::Kinematic {
            return Err(MoverError::NotKinematic {
                handle: body.index(),
            });
        }
        self.movers.push(KinematicMover::new(body));
        Ok(MoverHandle::from(self.movers.len() - 1))
    }

    /// Detach a mover; no-op for an invalid handle (same convention as
    /// [`remove_body`](PhysicsEngine::remove_body)). Surviving movers past
    /// the removal shift down one slot (swap-remove, like bodies).
    ///
    /// [`remove_body`](PhysicsEngine::remove_body):
    /// [`PhysicsEngine`](crate::engine::PhysicsEngine) trait method for
    /// bodies; movers are dropped/remapped there automatically.
    pub fn remove_mover(&mut self, handle: MoverHandle) {
        if handle.index() < self.movers.len() {
            self.movers.swap_remove(handle.index());
        }
    }

    /// Set the per-step displacement (m) the engine applies to the mover's
    /// platform every step. Persists until overwritten: set once for a
    /// constant-velocity platform, zero to hold position. Does not imply
    /// [`active`](KinematicMover::active) — activate separately.
    ///
    /// # Errors
    ///
    /// [`MoverError::UnknownMover`] for a stale handle,
    /// [`MoverError::BadDisplacement`] for non-finite input (never stored).
    pub fn set_mover_displacement(
        &mut self,
        handle: MoverHandle,
        displacement: Vec3,
    ) -> Result<(), MoverError> {
        if !displacement.is_finite() {
            return Err(MoverError::BadDisplacement);
        }
        let mover = self
            .movers
            .get_mut(handle.index())
            .ok_or(MoverError::UnknownMover {
                handle: handle.index(),
            })?;
        mover.displacement = displacement;
        Ok(())
    }

    /// Activate or park the mover. Only an active mover with a nonzero
    /// displacement moves its platform, carries passengers and keeps the
    /// world awake; an inactive mover is fully ignored.
    ///
    /// # Errors
    ///
    /// [`MoverError::UnknownMover`] for a stale handle.
    pub fn set_mover_active(
        &mut self,
        handle: MoverHandle,
        active: bool,
    ) -> Result<(), MoverError> {
        let mover = self
            .movers
            .get_mut(handle.index())
            .ok_or(MoverError::UnknownMover {
                handle: handle.index(),
            })?;
        mover.active = active;
        Ok(())
    }

    /// Read-only access to a mover, or `None` for an invalid handle.
    pub fn mover(&self, handle: MoverHandle) -> Option<&KinematicMover> {
        self.movers.get(handle.index())
    }

    /// How many movers are currently attached.
    pub fn mover_count(&self) -> usize {
        self.movers.len()
    }

    /// Engine mover pass: runs at the top of every step, before the
    /// sleeping fast-path check. Each active mover with a nonzero finite
    /// displacement moves its platform, carries contacting passengers via
    /// the plane solver and wakes them through the island gate. Returns
    /// whether any platform displaced (the fast-path gate).
    ///
    /// Deterministic: movers in handle order, passengers in body order.
    /// Movers whose body is gone or stopped being kinematic are skipped
    /// (the host owns body lifetimes and types; the pass never fails).
    pub(crate) fn apply_movers(&mut self) -> bool {
        if self.movers.is_empty() {
            return false;
        }
        let mut moved_any = false;
        for mi in 0..self.movers.len() {
            let Some(slot) = self.movers.get(mi) else {
                continue;
            };
            if !slot.active || slot.displacement == Vec3::ZERO {
                continue;
            }
            let body_idx = slot.body.index();
            let displacement = slot.displacement;
            if !displacement.is_finite() {
                continue;
            }
            let Some(platform) = self.bodies.get(body_idx) else {
                continue;
            };
            if platform.body_type != BodyType::Kinematic || platform.is_trigger {
                continue;
            }
            let travel = displacement.length();
            let mover_layer = platform.collision_layer;
            let mover_mask = platform.collision_mask;
            // Passengers: dynamic, non-trigger, filter-passing bodies whose
            // old-pose gap the platform segment can reach (swept proximity:
            // the step displacement plus one slop). Read at the old pose so
            // a descending lift still owns the passenger it is about to
            // uncover.
            let passengers: Vec<usize> = {
                let bodies = &self.bodies;
                let platform = &bodies[body_idx];
                let mut out = Vec::new();
                for (h, target) in bodies.iter().enumerate() {
                    if h == body_idx
                        || target.body_type != BodyType::Dynamic
                        || target.is_trigger
                        || mover_mask & target.collision_layer == 0
                        || target.collision_mask & mover_layer == 0
                    {
                        continue;
                    }
                    let gap = sweep_gap(
                        &platform.shape,
                        platform.position,
                        platform.orientation,
                        crate::distance::ShapeRef {
                            shape: &target.shape,
                            pos: target.position,
                            rot: target.orientation,
                        },
                    );
                    if gap <= travel + MOVER_SLOP {
                        out.push(h);
                    }
                }
                out
            };
            // The engine moves the platform (driver teleports never coexist
            // with a displacing mover on the same body — the mover owns the
            // step delta; host teleports on top compose additively through
            // the shared `prev_pose` baseline below).
            self.bodies[body_idx].position += displacement;
            moved_any = true;
            // Carry each passenger: full displacement when supported from
            // above or pushed into, tangential-plus-guard otherwise, always
            // resolved against the contact plane with the slop.
            let carries: Vec<(usize, Vec3)> = {
                let bodies = &self.bodies;
                let platform = &bodies[body_idx];
                let platform_ref = crate::distance::ShapeRef {
                    shape: &platform.shape,
                    pos: platform.position,
                    rot: platform.orientation,
                };
                let mut out = Vec::with_capacity(passengers.len());
                for &h in &passengers {
                    let target = &bodies[h];
                    let gap = sweep_gap(
                        &platform.shape,
                        platform.position,
                        platform.orientation,
                        crate::distance::ShapeRef {
                            shape: &target.shape,
                            pos: target.position,
                            rot: target.orientation,
                        },
                    );
                    let witness = crate::distance::shape_distance(
                        platform_ref,
                        crate::distance::ShapeRef {
                            shape: &target.shape,
                            pos: target.position,
                            rot: target.orientation,
                        },
                    );
                    // Platform-to-passenger contact normal. Deep overlap
                    // (beyond the slop) separates along the displacement:
                    // this pass drove the platform in along that axis, and
                    // the distance witnesses tie-break arbitrarily between
                    // opposite faces in overlap (a fast wall reads the
                    // victim's back face). Shallow touch uses the witness
                    // delta (exact), then the center delta, then world up.
                    let normal = if gap < -MOVER_SLOP {
                        displacement.normalize_or(Vec3::Y)
                    } else {
                        let mut axis = witness.point_b - witness.point_a;
                        if axis.length_squared() <= DEGENERATE_LEN2 {
                            axis = displacement;
                            if axis.length_squared() <= DEGENERATE_LEN2 {
                                axis = target.position - platform.position;
                            }
                        }
                        axis.normalize_or(Vec3::Y)
                    };
                    let approach = normal.dot(displacement);
                    let supported = normal.y > MOVER_SUPPORT_DOT || approach > 0.0;
                    let target_delta = if supported {
                        displacement
                    } else {
                        displacement - normal * approach
                    };
                    let mut planes = [MoverPlane::rigid(normal, gap)];
                    out.push((h, solve_mover_planes(target_delta, &mut planes, MOVER_SLOP)));
                }
                out
            };
            for (h, carry) in carries {
                if let Some(passenger) = self.bodies.get_mut(h) {
                    passenger.position += carry;
                }
                // Existing wake gate (same call the contact solver uses):
                // riding passengers stay awake, stopped ones re-sleep.
                self.wake_island(h);
            }
        }
        moved_any
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;

    /// No penetration: the carry target passes through untouched.
    #[test]
    fn plane_solver_keeps_clear_target() {
        let target = Vec3::new(0.1, 0.0, 0.0);
        let mut planes = [MoverPlane::rigid(Vec3::Y, 0.02)];
        let got = solve_mover_planes(target, &mut planes, MOVER_SLOP);
        assert_eq!(got, target, "clearance must not correct the carry");
    }

    /// Overlap beyond the slop is pushed out to exactly the slop.
    #[test]
    fn plane_solver_pushes_overlap_out_to_slop() {
        let target = Vec3::new(0.1, 0.0, 0.0);
        let mut planes = [MoverPlane::rigid(Vec3::Y, -0.03)];
        let got = solve_mover_planes(target, &mut planes, MOVER_SLOP);
        assert!((got.x - 0.1).abs() < 1e-6, "tangential carry kept: {got:?}");
        assert!(
            (got.y - (0.03 - MOVER_SLOP)).abs() < 1e-6,
            "overlap resolved to the slop: {got:?}"
        );
    }

    /// Penetration within the slop is left alone (no jitter correction).
    #[test]
    fn plane_solver_ignores_sub_slop_touch() {
        let target = Vec3::new(0.0, -0.001, 0.0);
        let mut planes = [MoverPlane::rigid(Vec3::Y, 0.0)];
        let got = solve_mover_planes(target, &mut planes, MOVER_SLOP);
        assert_eq!(got, target, "sub-slop touch must not jitter");
    }

    /// Two-plane corner: the target's penetrating components are removed
    /// on both axes, the exit direction is preserved.
    #[test]
    fn plane_solver_resolves_corner_on_both_axes() {
        let target = Vec3::new(-0.05, -0.05, 0.0);
        let mut planes = [
            MoverPlane::rigid(Vec3::X, -0.02),
            MoverPlane::rigid(Vec3::Y, -0.02),
        ];
        let got = solve_mover_planes(target, &mut planes, MOVER_SLOP);
        let want = 0.02 - MOVER_SLOP;
        assert!(
            (got.x - want).abs() < 1e-5 && (got.y - want).abs() < 1e-5,
            "corner push-out to the slop on both axes: {got:?}"
        );
    }

    /// Parks and rejects: non-kinematic bodies never take a mover, unknown
    /// handles fail explicitly, non-finite displacements never store.
    #[test]
    fn mover_admission_is_explicit() {
        use crate::engine::PhysicsEngine;
        let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
        let dynamic = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
        assert_eq!(
            physics.add_mover(dynamic),
            Err(MoverError::NotKinematic {
                handle: dynamic.index()
            })
        );
        assert_eq!(
            physics.add_mover(BodyHandle::from_raw(999)),
            Err(MoverError::InvalidBody { handle: 999 })
        );
        let mut platform = RigidBody::new_box(Vec3::ZERO, Vec3::splat(1.0), 1.0);
        platform.body_type = BodyType::Kinematic;
        let body = physics.add_body(platform);
        let mover = physics.add_mover(body).expect("kinematic takes a mover");
        assert_eq!(physics.mover_count(), 1);
        assert_eq!(
            physics.set_mover_displacement(mover, Vec3::NAN),
            Err(MoverError::BadDisplacement)
        );
        assert_eq!(
            physics.set_mover_displacement(MoverHandle::from_raw(7), Vec3::X),
            Err(MoverError::UnknownMover { handle: 7 })
        );
        assert_eq!(
            physics.set_mover_active(MoverHandle::from_raw(7), true),
            Err(MoverError::UnknownMover { handle: 7 })
        );
        assert!(!physics.mover(mover).expect("attached").active);
        physics.set_mover_active(mover, true).expect("valid handle");
        assert!(physics.mover(mover).expect("attached").active);
        physics.remove_mover(mover);
        assert_eq!(physics.mover_count(), 0);
        physics.remove_mover(mover);
    }
}
