//! Gizmo pointer-drag math (PLAN §i, slice E3): turns press/move rays into
//! a preview transform.
//!
//! Pure over [`PickRay`] and [`TransformDesc`] — the replica owns the press
//! edges, the live preview write and the `set_component` commit; this module
//! only answers "what transform does this ray mean?".
//!
//! Rotation handles are tangent bars; the drag never assumes a ring mesh,
//! it sweeps the ray angle around the axis.

use crate::gizmo::{GizmoAxis, GizmoHandle, GizmoMode, axis_direction};
use crate::picking::PickRay;
use crate::snap::SnapSettings;
use glam::{Quat, Vec3};
use ornis_assets::scene::TransformDesc;
use ornis_core::units::UnitQuat;

/// Radial distance below which a rotation drag has no pivot.
const RADIAL_EPS: f32 = 1e-8;
/// Rotation deltas below this angle (radians) keep the orientation.
const ROTATION_EPS: f32 = 1e-7;
/// Scale factors at or below this are ignored (press too close to centre).
const SCALE_FACTOR_EPS: f32 = 1e-4;
/// Floor applied to a dragged component scale (never zero or negative).
const MIN_SCALE: f32 = 1e-3;
/// Ray/plane and ray/axis parallel tolerance.
const PARALLEL_EPS: f32 = 1e-6;

/// One in-progress gizmo drag.
#[derive(Debug, Clone, PartialEq)]
pub struct GizmoDrag {
    /// Handle being dragged.
    pub handle: GizmoHandle,
    /// Transform before the drag started (preview base and one-step undo).
    pub start: TransformDesc,
    /// World-space axis/plane normal the drag is projected on.
    axis: Vec3,
    /// World-space drag plane (plane handles and rotation only).
    plane_point: Vec3,
    /// Scalar along `axis` at press (axis move/scale).
    start_param: f32,
    /// Plane point at press (plane move/rotation).
    start_point: Vec3,
}

impl GizmoDrag {
    /// Starts a drag from the press ray; `None` when the ray cannot be
    /// projected onto the handle (parallel ray, degenerate rotation radius).
    #[must_use]
    pub fn begin(selection: &TransformDesc, handle: GizmoHandle, ray: &PickRay) -> Option<Self> {
        let axis = axis_direction(selection, handle);
        let center = selection.translation;
        let (start_param, start_point) = match handle.axis {
            GizmoAxis::Xy | GizmoAxis::Yz | GizmoAxis::Xz => (0.0, ray_plane(ray, center, axis)?),
            _ => match handle.mode {
                GizmoMode::Move | GizmoMode::Scale => {
                    let param = closest_axis_param(ray, center, axis)?;
                    (param, center + axis * param)
                }
                GizmoMode::Rotate => {
                    let point = ray_plane(ray, center, axis)?;
                    if (point - center).length_squared() < RADIAL_EPS {
                        return None;
                    }
                    (0.0, point)
                }
                GizmoMode::Off => return None,
            },
        };
        Some(Self {
            handle,
            start: selection.clone(),
            axis,
            plane_point: center,
            start_param,
            start_point,
        })
    }

    /// Preview transform for the current pointer ray.
    // qual:allow(iosp) reason: "one dispatch over the handle axis/mode; the per-mode work lives in the apply_* helpers"
    #[must_use]
    pub fn update(&self, ray: &PickRay) -> Option<TransformDesc> {
        let mut transform = self.start.clone();
        match self.handle.axis {
            GizmoAxis::Xy | GizmoAxis::Yz | GizmoAxis::Xz => {
                let point = ray_plane(ray, self.plane_point, self.axis)?;
                transform.translation += point - self.start_point;
            }
            _ => match self.handle.mode {
                GizmoMode::Move => self.apply_move(ray, &mut transform)?,
                GizmoMode::Rotate => self.apply_rotate(ray, &mut transform)?,
                GizmoMode::Scale => self.apply_scale(ray, &mut transform)?,
                GizmoMode::Off => {}
            },
        }
        Some(transform)
    }

    /// Quantizes only the active drag channel (E4): Move snaps translation,
    /// Scale snaps scale, Rotate snaps the swept angle around the handle
    /// axis. Every other channel is returned bit-exact; snap off returns the
    /// preview unchanged.
    // qual:allow(iosp) reason: "one dispatch over the handle axis/mode mirroring update; the rotate leaf lives in snap_rotate"
    #[must_use]
    pub fn apply_snap(&self, mut transform: TransformDesc, snap: &SnapSettings) -> TransformDesc {
        if !snap.enabled {
            return transform;
        }
        match self.handle.axis {
            GizmoAxis::Xy | GizmoAxis::Yz | GizmoAxis::Xz => {
                transform.translation = snap.snap_translate(transform.translation);
            }
            _ => match self.handle.mode {
                GizmoMode::Move => {
                    transform.translation = snap.snap_translate(transform.translation);
                }
                GizmoMode::Scale => transform.scale = snap.snap_scale(transform.scale),
                GizmoMode::Rotate => {
                    transform.rotation = self.snap_rotate(transform.rotation, snap);
                }
                GizmoMode::Off => {}
            },
        }
        transform
    }

    /// Snaps the swept rotation angle around the handle axis relative to the
    /// drag start, keeping the axis and the other channels untouched. The
    /// angle is recovered from the preview's delta over `start.rotation`
    /// (`2·atan2(sin θ/2, cos θ/2)` projected on the axis), so a pre-rotated
    /// selection snaps its *increment*, not its absolute orientation.
    // qual:allow(iosp) reason: "one rotation-snap step: delta-angle recovery, quantization and quaternion composition belong together (mirrors apply_rotate)"
    fn snap_rotate(&self, rotation: UnitQuat, snap: &SnapSettings) -> UnitQuat {
        if snap.rotate_deg <= 0.0 {
            return rotation;
        }
        let delta = rotation.get() * self.start.rotation.get().inverse();
        let sine = Vec3::new(delta.x, delta.y, delta.z).dot(self.axis);
        let angle = 2.0 * sine.atan2(delta.w);
        let snapped = snap.snap_angle(angle);
        UnitQuat::normalize(Quat::from_axis_angle(self.axis, snapped))
            .and_then(|delta| UnitQuat::normalize(delta.get() * self.start.rotation.get()))
            .unwrap_or(rotation)
    }

    /// Axis translation: offset along `axis` since the press.
    fn apply_move(&self, ray: &PickRay, transform: &mut TransformDesc) -> Option<()> {
        let param = closest_axis_param(ray, self.start.translation, self.axis)?;
        transform.translation += self.axis * (param - self.start_param);
        Some(())
    }

    /// Rotation around `axis` by the angle swept from the press point.
    // qual:allow(iosp) reason: "one rotation step: projection, degenerate guard and quaternion composition belong together"
    fn apply_rotate(&self, ray: &PickRay, transform: &mut TransformDesc) -> Option<()> {
        let point = ray_plane(ray, self.plane_point, self.axis)?;
        let from = self.start_point - self.start.translation;
        let to = point - self.start.translation;
        if from.length_squared() < RADIAL_EPS || to.length_squared() < RADIAL_EPS {
            return None;
        }
        let angle = self.axis.dot(from.cross(to)).atan2(from.dot(to));
        if !angle.is_finite() || angle.abs() < ROTATION_EPS {
            return Some(());
        }
        let delta = UnitQuat::normalize(Quat::from_axis_angle(self.axis, angle))?;
        // `self.axis` is world-space in both spaces (for Local it is already
        // `R·e`), so the swept angle is a world-space rotation and the
        // result is `delta * R` in both branches. In Local that equals
        // `R * q(e_local, θ)` by conjugacy: the object turns around its own
        // axis, and the local axis itself stays where it was. Composing
        // `R * delta` instead would pre- and post-rotate by `R` and spin
        // the object off its own axis.
        transform.rotation = UnitQuat::normalize(delta.get() * self.start.rotation.get())?;
        Some(())
    }

    /// Axis scale: multiply the component by the radial ratio.
    ///
    /// The factor multiplies the named *local* component in both spaces: a
    /// world-space handle on a rotated entity scales its own axis, not the
    /// world axis. A true world-axis scale is not representable as a
    /// diagonal scale (it would need a shear), so the gizmo keeps the local
    /// model and documents the difference.
    fn apply_scale(&self, ray: &PickRay, transform: &mut TransformDesc) -> Option<()> {
        let param = closest_axis_param(ray, self.start.translation, self.axis)?;
        if self.start_param.abs() < SCALE_FACTOR_EPS {
            return Some(());
        }
        let factor = param / self.start_param;
        if !factor.is_finite() || factor <= SCALE_FACTOR_EPS {
            return Some(());
        }
        let index = match self.handle.axis {
            GizmoAxis::X => 0,
            GizmoAxis::Y => 1,
            _ => 2,
        };
        transform.scale[index] = (self.start.scale[index] * factor).max(MIN_SCALE);
        Some(())
    }
}

/// Ray/plane intersection through `point` with `normal`; positive distance.
fn ray_plane(ray: &PickRay, point: Vec3, normal: Vec3) -> Option<Vec3> {
    let direction = Vec3::from(ray.direction());
    let denom = direction.dot(normal);
    if !denom.is_finite() || denom.abs() < PARALLEL_EPS {
        return None;
    }
    let t = (point - ray.origin()).dot(normal) / denom;
    if t > PARALLEL_EPS && t.is_finite() {
        Some(ray.at(t))
    } else {
        None
    }
}

/// Parameter of the closest point on the line `center + axis * t` to the ray.
fn closest_axis_param(ray: &PickRay, center: Vec3, axis: Vec3) -> Option<f32> {
    let direction = Vec3::from(ray.direction());
    let w0 = ray.origin() - center;
    let b = direction.dot(axis);
    let denom = 1.0 - b * b;
    if !denom.is_finite() || denom.abs() < PARALLEL_EPS {
        return None;
    }
    let d = direction.dot(w0);
    let e = axis.dot(w0);
    let t = (e - b * d) / denom;
    (t.is_finite()).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gizmo::GizmoSpace;

    /// Float tolerance for the pure-math assertions.
    const TEST_EPS: f32 = 1e-5;

    fn identity() -> TransformDesc {
        TransformDesc::IDENTITY
    }

    fn ray_from(origin: Vec3, target: Vec3) -> PickRay {
        PickRay::try_new(origin, target - origin).expect("valid test ray")
    }

    #[test]
    fn move_drag_shifts_along_the_axis() {
        let start = identity();
        let drag = GizmoDrag::begin(
            &start,
            GizmoHandle {
                mode: GizmoMode::Move,
                space: GizmoSpace::World,
                axis: GizmoAxis::X,
            },
            &ray_from(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO),
        )
        .expect("drag starts");
        let moved = drag
            .update(&ray_from(
                Vec3::new(2.0, 0.0, 5.0),
                Vec3::new(2.0, 0.0, 0.0),
            ))
            .expect("drag updates");
        assert!((moved.translation - Vec3::new(2.0, 0.0, 0.0)).length() < TEST_EPS);
        assert_eq!(moved.rotation, start.rotation);
    }

    #[test]
    fn plane_drag_moves_in_both_plane_axes() {
        let drag = GizmoDrag::begin(
            &identity(),
            GizmoHandle {
                mode: GizmoMode::Move,
                space: GizmoSpace::World,
                axis: GizmoAxis::Xy,
            },
            &ray_from(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO),
        )
        .expect("drag starts");
        let moved = drag
            .update(&ray_from(
                Vec3::new(1.0, 1.0, 5.0),
                Vec3::new(1.0, 1.0, 0.0),
            ))
            .expect("drag updates");
        assert!((moved.translation - Vec3::new(1.0, 1.0, 0.0)).length() < TEST_EPS);
    }

    #[test]
    fn rotate_drag_applies_a_quarter_turn_around_y() {
        let drag = GizmoDrag::begin(
            &identity(),
            GizmoHandle {
                mode: GizmoMode::Rotate,
                space: GizmoSpace::World,
                axis: GizmoAxis::Y,
            },
            &ray_from(Vec3::new(1.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
        )
        .expect("drag starts");
        let rotated = drag
            .update(&ray_from(
                Vec3::new(0.0, 5.0, -1.0),
                Vec3::new(0.0, 0.0, -1.0),
            ))
            .expect("drag updates");
        let got = rotated.rotation.get() * Vec3::X;
        assert!(
            (got - Vec3::new(0.0, 0.0, -1.0)).length() < TEST_EPS,
            "got {got:?}"
        );
    }

    #[test]
    fn scale_drag_multiplies_the_axis_component() {
        let drag = GizmoDrag::begin(
            &identity(),
            GizmoHandle {
                mode: GizmoMode::Scale,
                space: GizmoSpace::World,
                axis: GizmoAxis::X,
            },
            &ray_from(Vec3::new(1.0, 0.0, 5.0), Vec3::new(1.0, 0.0, 0.0)),
        )
        .expect("drag starts");
        let scaled = drag
            .update(&ray_from(
                Vec3::new(2.0, 0.0, 5.0),
                Vec3::new(2.0, 0.0, 0.0),
            ))
            .expect("drag updates");
        assert!((scaled.scale - Vec3::new(2.0, 1.0, 1.0)).length() < TEST_EPS);
    }

    /// Whether two orientations describe the same rotation (`q` and `-q` are
    /// the same rotation).
    fn same_rotation(a: Quat, b: Quat) -> bool {
        (a.dot(b).abs() - 1.0).abs() < 1e-4
    }

    /// A selection rotated by `rotY(90°) · rotX(0.3)` — a deliberately
    /// non-commuting orientation, where `R * delta` and `delta * R` differ.
    fn tilted() -> TransformDesc {
        TransformDesc {
            rotation: UnitQuat::normalize(
                Quat::from_rotation_y(std::f32::consts::FRAC_PI_2) * Quat::from_rotation_x(0.3),
            )
            .expect("unit"),
            ..identity()
        }
    }

    /// A selection rotated by +90° around Z (its local X points at world
    /// +Y).
    fn turned_z() -> TransformDesc {
        TransformDesc {
            rotation: UnitQuat::normalize(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2))
                .expect("unit"),
            ..identity()
        }
    }

    /// A selection rotated by +90° around X (its local Z points at world
    /// -Y, so its local XY plane is the world XZ plane).
    fn turned_x() -> TransformDesc {
        TransformDesc {
            rotation: UnitQuat::normalize(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
                .expect("unit"),
            ..identity()
        }
    }

    /// Starts a drag of `selection` on `handle` with the press ray, if the
    /// ray can be projected onto the handle at all.
    fn begin_drag(
        selection: &TransformDesc,
        handle: GizmoHandle,
        press: PickRay,
    ) -> Option<GizmoDrag> {
        GizmoDrag::begin(selection, handle, &press)
    }

    /// The transform a drag produces: pressed with `press`, previewed with
    /// `release`.
    fn drag_preview(
        selection: &TransformDesc,
        handle: GizmoHandle,
        press: PickRay,
        release: PickRay,
    ) -> TransformDesc {
        let drag = begin_drag(selection, handle, press).expect("drag starts");
        drag.update(&release).expect("drag updates")
    }

    /// Local rotate composes in the local frame: the result is `R · q(e, θ)`
    /// (equal to `delta · R`), and the local rotation axis stays where it
    /// was. The world-space composition `R · delta` would spin the object
    /// off its own axis.
    #[test]
    fn local_rotate_composes_in_the_local_frame() {
        let start = tilted();
        // The local X axis is world -Z, so the swept plane is world XY.
        let rotated = drag_preview(
            &start,
            GizmoHandle {
                mode: GizmoMode::Rotate,
                space: GizmoSpace::Local,
                axis: GizmoAxis::X,
            },
            ray_from(Vec3::new(1.0, 0.0, 5.0), Vec3::new(1.0, 0.0, 0.0)),
            ray_from(Vec3::new(0.0, 1.0, 5.0), Vec3::new(0.0, 1.0, 0.0)),
        );

        // The sweep from +X to +Y around the local X axis is -90°.
        let expected = start.rotation.get() * Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        assert!(
            same_rotation(rotated.rotation.get(), expected),
            "local rotate must equal R·q(e,θ), got {:?}",
            rotated.rotation.get()
        );
        // The local axis itself must not move: a rotation around X leaves
        // the world direction of the local X axis untouched.
        let axis_before = start.rotation.get() * Vec3::X;
        let axis_after = rotated.rotation.get() * Vec3::X;
        assert!(
            (axis_before - axis_after).length() < TEST_EPS,
            "local axis moved: {axis_before:?} -> {axis_after:?}"
        );
        // The world-space composition is the wrong answer for Local.
        let wrong =
            start.rotation.get() * Quat::from_axis_angle(axis_before, -std::f32::consts::FRAC_PI_2);
        assert!(
            !same_rotation(rotated.rotation.get(), wrong),
            "the local composition must differ from R·delta"
        );
    }

    /// Local handles follow the *rotated* axes: a +90° Z rotation sends the
    /// local X to world +Y, and the local XY plane of a +90° X rotation is
    /// the world XZ plane.
    #[test]
    fn local_drag_follows_the_rotated_axes() {
        // Local move: the object slides along world +Y (the rotated local
        // X), not along world X.
        let moved = drag_preview(
            &turned_z(),
            GizmoHandle {
                mode: GizmoMode::Move,
                space: GizmoSpace::Local,
                axis: GizmoAxis::X,
            },
            ray_from(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO),
            ray_from(Vec3::new(0.0, 1.0, 5.0), Vec3::new(0.0, 1.0, 0.0)),
        );
        assert!(
            (moved.translation - Vec3::new(0.0, 1.0, 0.0)).length() < TEST_EPS,
            "local move must follow the rotated axis, got {:?}",
            moved.translation
        );

        // Local plane: the drag stays inside the rotated plane (world XZ
        // for a +90° X rotation).
        let moved = drag_preview(
            &turned_x(),
            GizmoHandle {
                mode: GizmoMode::Move,
                space: GizmoSpace::Local,
                axis: GizmoAxis::Xy,
            },
            ray_from(Vec3::new(1.0, 2.0, 5.0), Vec3::new(1.0, 0.0, 0.0)),
            ray_from(Vec3::new(2.0, 2.0, 5.0), Vec3::new(2.0, 0.0, 0.0)),
        );
        assert!(
            (moved.translation - Vec3::new(1.0, 0.0, 0.0)).length() < TEST_EPS,
            "local plane drag must stay in the rotated plane, got {:?}",
            moved.translation
        );
    }

    /// A ray parallel to the handle axis has no projection: the drag never
    /// starts and never previews.
    #[test]
    fn parallel_ray_has_no_preview() {
        let handle = GizmoHandle {
            mode: GizmoMode::Move,
            space: GizmoSpace::World,
            axis: GizmoAxis::X,
        };
        let parallel = ray_from(Vec3::new(0.0, 0.0, 5.0), Vec3::new(1.0, 0.0, 5.0));
        assert!(
            begin_drag(&identity(), handle, parallel).is_none(),
            "a ray parallel to the axis cannot start a drag"
        );
        let drag = begin_drag(
            &identity(),
            handle,
            ray_from(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO),
        )
        .expect("drag starts");
        assert!(
            drag.update(&parallel).is_none(),
            "a ray parallel to the axis has no preview"
        );
    }

    /// A scale drag multiplies the named *local* component — a world handle
    /// on a rotated entity scales its own axis, not the world axis (a
    /// world-axis scale would need a shear) — and never drives a component
    /// to zero: the floor is `MIN_SCALE`.
    #[test]
    fn scale_drag_multiplies_the_local_component_and_is_floored() {
        // The floor: a tiny factor cannot shrink the component to zero.
        let scaled = drag_preview(
            &identity(),
            GizmoHandle {
                mode: GizmoMode::Scale,
                space: GizmoSpace::World,
                axis: GizmoAxis::X,
            },
            ray_from(Vec3::new(1.0, 0.0, 5.0), Vec3::new(1.0, 0.0, 0.0)),
            ray_from(Vec3::new(0.0005, 0.0, 5.0), Vec3::new(0.0005, 0.0, 0.0)),
        );
        assert!(
            (scaled.scale - Vec3::new(MIN_SCALE, 1.0, 1.0)).length() < TEST_EPS,
            "scale must be floored, got {:?}",
            scaled.scale
        );

        // The world handle on a rotated entity scales its own local axis.
        let scaled = drag_preview(
            &turned_z(),
            GizmoHandle {
                mode: GizmoMode::Scale,
                space: GizmoSpace::World,
                axis: GizmoAxis::X,
            },
            ray_from(Vec3::new(1.0, 0.0, 5.0), Vec3::new(1.0, 0.0, 0.0)),
            ray_from(Vec3::new(2.0, 0.0, 5.0), Vec3::new(2.0, 0.0, 0.0)),
        );
        assert!(
            (scaled.scale - Vec3::new(2.0, 1.0, 1.0)).length() < TEST_EPS,
            "world scale multiplies the local component, got {:?}",
            scaled.scale
        );
    }
}
