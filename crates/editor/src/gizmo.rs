//! Transform-gizmo domain (PLAN §i, slice E3): handle identity, layout and
//! pointer drag math.
//!
//! The authoritative world owns the handles as ordinary
//! [`EditorOnly`](crate::EditorOnly) entities carrying [`GizmoHandle`]; a
//! host system lays them out around the selected entity. The viewport
//! replica picks them before scene content and turns a drag into a
//! [`TransformDesc`] preview, then commits the same value through the
//! regular `set_component` mutation path.
//!
//! The math here is pure (rays in, transforms out) so drag parity and axis
//! conventions are unit-tested without a GPU.
//!
//! Rotation handles are tangent bars on the rotation radius rather than
//! torus rings: the shared instanced mesh is a unit sphere, so a full ring
//! would need a custom mesh per handle. The drag math (angle around the
//! axis) is the same either way.

use glam::{Quat, Vec3};
use ornis_assets::scene::TransformDesc;
use ornis_core::units::UnitQuat;
use serde::{Deserialize, Serialize};

/// Gizmo interaction mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GizmoMode {
    /// Translate along axes or in the three planes.
    #[default]
    Move,
    /// Rotate around an axis.
    Rotate,
    /// Scale along an axis.
    Scale,
    /// Gizmo hidden (select tool); authoring never produces handles here.
    Off,
}

/// Whether handles use world axes or the selected entity's local axes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GizmoSpace {
    /// World-aligned axes.
    #[default]
    World,
    /// Axes rotated by the selected entity's orientation.
    Local,
}

/// Which axis (or plane) a handle drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GizmoAxis {
    /// X axis (or the YZ plane normal).
    X,
    /// Y axis (or the XZ plane normal).
    Y,
    /// Z axis (or the XY plane normal).
    Z,
    /// The XY plane.
    Xy,
    /// The YZ plane.
    Yz,
    /// The XZ plane.
    Xz,
}

/// Marker component of one gizmo handle in the authoritative world.
///
/// Handles are [`EditorOnly`](crate::EditorOnly) entities: they replicate
/// through `/api/scene`, never save into the scene and never reach physics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GizmoHandle {
    /// Interaction mode this handle belongs to.
    pub mode: GizmoMode,
    /// Space (world or the selection's local axes) this handle belongs to.
    pub space: GizmoSpace,
    /// Axis or plane the handle drives.
    pub axis: GizmoAxis,
}

impl GizmoHandle {
    /// All handles the authoritative session authors (both spaces, all modes).
    // qual:allow(dry, boilerplate) reason: "the cartesian product of mode×space×axis IS the handle table; a struct-update base would hide which field varies"
    #[must_use]
    pub fn all() -> Vec<Self> {
        let mut handles = Vec::new();
        for space in [GizmoSpace::World, GizmoSpace::Local] {
            for axis in [
                GizmoAxis::X,
                GizmoAxis::Y,
                GizmoAxis::Z,
                GizmoAxis::Xy,
                GizmoAxis::Yz,
                GizmoAxis::Xz,
            ] {
                handles.push(Self {
                    mode: GizmoMode::Move,
                    space,
                    axis,
                });
            }
            for axis in [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z] {
                handles.push(Self {
                    mode: GizmoMode::Rotate,
                    space,
                    axis,
                });
                handles.push(Self {
                    mode: GizmoMode::Scale,
                    space,
                    axis,
                });
            }
        }
        handles
    }

    /// Whether this handle belongs to `mode`/`space` and is therefore
    /// pickable and visible in that mode.
    #[must_use]
    pub fn matches(self, mode: GizmoMode, space: GizmoSpace) -> bool {
        self.mode == mode && self.space == space
    }
}

/// A handle's world pose around the selected entity.
#[derive(Debug, Clone, PartialEq)]
pub struct GizmoHandlePose {
    /// The handle this pose belongs to.
    pub handle: GizmoHandle,
    /// World transform of the handle entity.
    pub transform: TransformDesc,
}

/// Move bars: centre distance, length, thickness.
const MOVE_OFFSET: f32 = 0.55;
pub(crate) const MOVE_LENGTH: f32 = 0.7;
/// Plane handles: centre offset along both axes, cube half-size.
const PLANE_OFFSET: f32 = 0.3;
const PLANE_SIZE: f32 = 0.16;
/// Rotate bars sit on a wider radius.
const ROTATE_OFFSET: f32 = 1.05;
const ROTATE_LENGTH: f32 = 0.35;
/// Scale bars end cubes sit on the widest radius.
const SCALE_OFFSET: f32 = 1.4;
const SCALE_SIZE: f32 = 0.16;
/// Bar thickness (uniform).
pub(crate) const BAR_THICKNESS: f32 = 0.045;
/// Directions shorter than this are treated as degenerate.
const DIRECTION_EPS: f32 = 1e-12;

/// World direction of `axis` for a handle in `space`.
///
/// Plane axes return the plane normal; the in-plane pair comes from
/// [`plane_directions`].
#[must_use]
pub fn axis_direction(selection: &TransformDesc, handle: GizmoHandle) -> Vec3 {
    let local = match handle.axis {
        GizmoAxis::X | GizmoAxis::Yz => Vec3::X,
        GizmoAxis::Y | GizmoAxis::Xz => Vec3::Y,
        GizmoAxis::Z | GizmoAxis::Xy => Vec3::Z,
    };
    orient(selection, handle.space, local)
}

/// The two in-plane directions of a plane handle (for axis handles this is
/// not meaningful; callers only use it for `Xy`/`Yz`/`Xz`).
#[must_use]
pub fn plane_directions(selection: &TransformDesc, handle: GizmoHandle) -> (Vec3, Vec3) {
    let (a, b) = match handle.axis {
        GizmoAxis::Xy => (Vec3::X, Vec3::Y),
        GizmoAxis::Yz => (Vec3::Y, Vec3::Z),
        GizmoAxis::Xz => (Vec3::X, Vec3::Z),
        GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => (Vec3::X, Vec3::Y),
    };
    (
        orient(selection, handle.space, a),
        orient(selection, handle.space, b),
    )
}

/// Rotates a unit world direction into `space`.
fn orient(selection: &TransformDesc, space: GizmoSpace, unit: Vec3) -> Vec3 {
    match space {
        GizmoSpace::World => unit,
        GizmoSpace::Local => {
            let rotated = selection.rotation.get() * unit;
            if rotated.is_finite() && rotated.length_squared() > DIRECTION_EPS {
                rotated.normalize()
            } else {
                unit
            }
        }
    }
}

/// A direction perpendicular to `handle`'s rotation axis, used to place the
/// tangent bar on the rotation circle.
fn ring_radial(selection: &TransformDesc, handle: GizmoHandle) -> Vec3 {
    let unit = match handle.axis {
        GizmoAxis::X => Vec3::Y,
        GizmoAxis::Y => Vec3::Z,
        _ => Vec3::X,
    };
    orient(selection, handle.space, unit)
}

/// Quaternion aligning the local `+X` bar with `direction`.
fn along_x(direction: Vec3) -> UnitQuat {
    UnitQuat::normalize(Quat::from_rotation_arc(Vec3::X, direction)).unwrap_or(UnitQuat::IDENTITY)
}

/// World pose of `handle` around `selection`.
// qual:allow(iosp) reason: "geometry dispatch over the shared offset table; splitting the plane/bar branches would duplicate the mode→size mapping"
#[must_use]
pub fn layout(selection: &TransformDesc, handle: GizmoHandle) -> GizmoHandlePose {
    let center = selection.translation;
    let (offset, length, thickness) = match handle.mode {
        GizmoMode::Move | GizmoMode::Off => (MOVE_OFFSET, MOVE_LENGTH, BAR_THICKNESS),
        GizmoMode::Rotate => (ROTATE_OFFSET, ROTATE_LENGTH, BAR_THICKNESS),
        GizmoMode::Scale => (SCALE_OFFSET, SCALE_SIZE, SCALE_SIZE),
    };
    let (translation, rotation, scale) = match handle.axis {
        GizmoAxis::Xy | GizmoAxis::Yz | GizmoAxis::Xz => {
            let (a, b) = plane_directions(selection, handle);
            (
                center + (a + b) * PLANE_OFFSET,
                UnitQuat::IDENTITY,
                Vec3::splat(PLANE_SIZE),
            )
        }
        _ => {
            let axis = axis_direction(selection, handle);
            let (translation, direction) = match handle.mode {
                // Tangent bar on the rotation circle: the axis itself points
                // out of the ring, so the visible bar follows the sweep.
                GizmoMode::Rotate => {
                    let radial = ring_radial(selection, handle);
                    let tangent = axis.cross(radial);
                    let tangent = if tangent.length_squared() > DIRECTION_EPS {
                        tangent.normalize()
                    } else {
                        Vec3::Z
                    };
                    (center + radial * offset, tangent)
                }
                _ => (center + axis * offset, axis),
            };
            (
                translation,
                along_x(direction),
                Vec3::new(length, thickness, thickness),
            )
        }
    };
    GizmoHandlePose {
        handle,
        transform: TransformDesc {
            translation,
            rotation,
            scale,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// Float tolerance for the pure-math assertions.
    const TEST_EPS: f32 = 1e-5;

    fn identity() -> TransformDesc {
        TransformDesc::IDENTITY
    }

    #[test]
    fn move_axis_layout_places_bars_along_world_axes() {
        let x = layout(
            &identity(),
            GizmoHandle {
                mode: GizmoMode::Move,
                space: GizmoSpace::World,
                axis: GizmoAxis::X,
            },
        );
        assert!((x.transform.translation - Vec3::new(MOVE_OFFSET, 0.0, 0.0)).length() < TEST_EPS);
        assert_eq!(
            x.transform.scale,
            Vec3::new(MOVE_LENGTH, BAR_THICKNESS, BAR_THICKNESS)
        );
        assert!(x.transform.rotation.get().is_finite());
    }

    #[test]
    fn rotate_handles_sit_on_the_ring_and_point_tangent() {
        let pose = layout(
            &identity(),
            GizmoHandle {
                mode: GizmoMode::Rotate,
                space: GizmoSpace::World,
                axis: GizmoAxis::X,
            },
        );
        // The X ring lies in the YZ plane: the bar sits on +Y and runs
        // tangent (along Z), making the swept plane readable.
        assert!(
            (pose.transform.translation - Vec3::new(0.0, ROTATE_OFFSET, 0.0)).length() < TEST_EPS
        );
        let direction = pose.transform.rotation.get() * Vec3::X;
        assert!(
            direction.z.abs() > 1.0 - TEST_EPS,
            "expected a tangent bar, got {direction:?}"
        );
    }

    #[test]
    fn local_space_rotates_handles_with_the_selection() {
        let selection = TransformDesc {
            rotation: UnitQuat::normalize(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2))
                .expect("unit"),
            ..identity()
        };
        let x = layout(
            &selection,
            GizmoHandle {
                mode: GizmoMode::Move,
                space: GizmoSpace::Local,
                axis: GizmoAxis::X,
            },
        );
        // Local +X points along world -Z after a +90° Y rotation.
        assert!((x.transform.translation - Vec3::new(0.0, 0.0, -MOVE_OFFSET)).length() < TEST_EPS);
    }

    #[test]
    fn all_handles_cover_every_mode_and_space() {
        let all = GizmoHandle::all();
        assert_eq!(all.len(), 24);
        for mode in [GizmoMode::Move, GizmoMode::Rotate, GizmoMode::Scale] {
            for space in [GizmoSpace::World, GizmoSpace::Local] {
                assert!(all.iter().any(|h| h.matches(mode, space)));
            }
        }
    }
}
