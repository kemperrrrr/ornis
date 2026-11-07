//! Gizmo pointer-drag math (PLAN §i, slice E3): turns press/move rays into
//! a preview transform.
//!
//! Pure over [`PickRay`] and [`TransformDesc`] — the replica owns the press
//! edges, the live preview write and the `set_component` commit; this module
//! only answers "what transform does this ray mean?".
//!
//! Rotation handles are tangent bars; the drag never assumes a ring mesh,
//! it sweeps the ray angle around the axis.

use crate::gizmo::{GizmoAxis, GizmoHandle, GizmoMode, GizmoSpace, axis_direction};
use crate::picking::PickRay;
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
    pub fn update(self, ray: &PickRay) -> Option<TransformDesc> {
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
        transform.rotation = match self.handle.space {
            GizmoSpace::World => UnitQuat::normalize(delta.get() * self.start.rotation.get())?,
            GizmoSpace::Local => UnitQuat::normalize(self.start.rotation.get() * delta.get())?,
        };
        Some(())
    }

    /// Axis scale: multiply the component by the radial ratio.
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
}
