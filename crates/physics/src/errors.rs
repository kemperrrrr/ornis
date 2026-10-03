//! Typed physics failures: mesh, collider, query and joint errors.
//!
//! Point 5 replaces `String`/`Option`/`bool`/`panic!` mesh and query
//! plumbing with `thiserror` hierarchies (sample: `TextureUploadError` in
//! `ornis-render`). Happy-path behavior is unchanged; only invalid inputs
//! gain a typed reason plus tests on the new branches.

use thiserror::Error;

use crate::constants::DEGENERATE_LEN2;

/// Mesh construction failure: out-of-range indices, bad grids, non-finite input.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MeshError {
    /// No vertices where at least one was required.
    #[error("mesh has no vertices")]
    EmptyVertices,
    /// No points to bound.
    #[error("no points to bound")]
    EmptyPoints,
    /// Non-finite vertex at `index`.
    #[error("non-finite vertex at index {index}")]
    NonFiniteVertex {
        /// Offending vertex index.
        index: usize,
    },
    /// Triangle `triangle` references `index`, outside `vertices` vertices.
    #[error("triangle {triangle}: index {index} out of range ({vertices} vertices)")]
    OutOfRangeIndex {
        /// Triangle ordinal in the input list.
        triangle: usize,
        /// Offending index value.
        index: u32,
        /// Vertex count.
        vertices: usize,
    },
    /// `heights.len() ({actual}) != rows ({rows}) * cols ({cols}, want {expected})`.
    #[error("heightfield grid {rows}x{cols} wants {expected} heights, got {actual}")]
    InvalidGrid {
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
        /// Required sample count.
        expected: usize,
        /// Provided sample count.
        actual: usize,
    },
    /// Non-positive or non-finite cell spacing.
    #[error("heightfield cell must be positive finite, got {cell}")]
    InvalidCell {
        /// Offending spacing.
        cell: f32,
    },
    /// Empty heightfield grid (`rows == 0` or `cols == 0`).
    #[error("heightfield grid is empty")]
    EmptyGrid,
    /// Non-finite height sample at `index`.
    #[error("non-finite height at index {index}")]
    NonFiniteHeight {
        /// Offending sample index.
        index: usize,
    },
    /// Degenerate soup: no usable triangles for an exact inertia.
    #[error("degenerate mesh: no usable triangles")]
    DegenerateMesh,
    /// Flat index list length is not a multiple of 3.
    #[error("triangle index list length {len} is not a multiple of 3")]
    BadIndexCount {
        /// Offending length.
        len: usize,
    },
}

/// Compound/rounded/half-space construction failure (P5 shapes).
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ShapeError {
    /// Compound with no children (an empty union collides with nothing).
    #[error("compound shape has no children")]
    EmptyCompound,
    /// Rounded-shape border radius is non-positive or non-finite.
    #[error("border radius must be finite and > 0, got {radius}")]
    BadBorderRadius {
        /// Offending radius value.
        radius: f32,
    },
    /// Half-space normal is zero-length or non-finite.
    #[error("half-space normal must be a finite non-zero vector")]
    BadNormal,
    /// Half-space body requested with dynamic mass (half-spaces are
    /// static-only: an infinite plane has no finite inertia).
    #[error("half-space shapes are static-only, got mass {mass}")]
    DynamicHalfSpace {
        /// Offending mass value.
        mass: f32,
    },
}

/// Collider projection failure: `Ok(None)` means "no collider", `Err` means "broken".
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ColliderError {
    /// Flat index list length is not a multiple of 3.
    #[error("triangle index list length {len} is not a multiple of 3")]
    BadIndexCount {
        /// Offending length.
        len: usize,
    },
    /// Flat index `index` is outside `vertices` vertices.
    #[error("soup index {index} out of range ({vertices} vertices)")]
    IndexOutOfRange {
        /// Offending index.
        index: u32,
        /// Vertex count.
        vertices: usize,
    },
    /// Underlying mesh construction failed.
    #[error("invalid mesh: {0}")]
    InvalidMesh(#[from] MeshError),
    /// Underlying compound/rounded/half-space construction failed.
    #[error("invalid shape: {0}")]
    InvalidShape(#[from] ShapeError),
    /// A compound/rounded recipe names no buildable geometry.
    #[error("collider recipe has no buildable geometry: {detail}")]
    EmptyRecipe {
        /// What was missing.
        detail: &'static str,
    },
}

/// Raycast/shapecast input failure: `Ok(None)` is a clean miss.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum QueryError {
    /// Non-finite origin/direction, zero direction, or bad `max_dist`.
    #[error("invalid query input: {reason}")]
    InvalidInput {
        /// Human-readable reason (no allocation on the happy path).
        reason: String,
    },
}

/// Joint admission failure with a typed reason.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum JointError {
    /// Limit/ bound pair is non-finite or inverted.
    #[error("bad bounds: min {min} > max {max} or non-finite")]
    BadBounds {
        /// Lower bound.
        min: f32,
        /// Upper bound.
        max: f32,
    },
    /// Non-finite scalar field.
    #[error("non-finite field `{field}`")]
    NonFinite {
        /// Field name.
        field: String,
    },
    /// Zero-length or non-finite axis/anchor.
    #[error("bad axis: {detail}")]
    BadAxis {
        /// What was wrong.
        detail: String,
    },
    /// Dangling joint/body reference.
    #[error("unknown joint/body reference {handle}")]
    UnknownRef {
        /// Offending handle.
        handle: usize,
    },
    /// Invalid body pair (out of range).
    #[error("invalid bodies {a} and {b}")]
    InvalidHandles {
        /// First handle.
        a: usize,
        /// Second handle.
        b: usize,
    },
    /// A joint cannot connect a body to itself.
    #[error("self joint on body {handle}")]
    SelfJoint {
        /// Offending handle.
        handle: usize,
    },
    /// Joint kind is valid but unsupported by this solver.
    #[error("unsupported joint: {detail}")]
    Unsupported {
        /// What is unsupported.
        detail: String,
    },
}

/// World-snapshot (`serde` RON) failure: version gate, decode, invalid data
/// or an engine/path that cannot be captured or restored (P8).
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SnapshotError {
    /// Snapshot `version` does not match
    /// [`WORLD_SNAPSHOT_VERSION`](crate::snapshot::WORLD_SNAPSHOT_VERSION).
    #[error("unsupported snapshot version {found}, expected {expected}")]
    VersionMismatch {
        /// Required version.
        expected: u32,
        /// Version found in the input.
        found: u32,
    },
    /// RON decoding failed (malformed input or a shape the format rejects).
    #[error("snapshot decode failed: {detail}")]
    Decode {
        /// Underlying parser message.
        detail: String,
    },
    /// Decoded data fails validation (dangling handles, non-finite
    /// scalars, empty compounds, unknown enum tags).
    #[error("invalid snapshot data: {detail}")]
    InvalidData {
        /// What was wrong.
        detail: String,
    },
    /// Engine or routing path outside the v1 snapshot scope (see
    /// [`crate::snapshot`] for the supported surface).
    #[error("snapshot not supported here: {detail}")]
    Unsupported {
        /// What was requested and why it is refused.
        detail: String,
    },
}
/// Validate a raycast query without touching any body.
pub(crate) fn check_ray_input(
    origin: glam::Vec3,
    direction: glam::Vec3,
    max_dist: f32,
) -> Result<(), QueryError> {
    if !origin.is_finite() || !direction.is_finite() {
        return Err(QueryError::InvalidInput {
            reason: "non-finite ray origin/direction".into(),
        });
    }
    if direction.length_squared() < DEGENERATE_LEN2 {
        return Err(QueryError::InvalidInput {
            reason: "zero-length ray direction".into(),
        });
    }
    if !max_dist.is_finite() || max_dist < 0.0 {
        return Err(QueryError::InvalidInput {
            reason: "max_dist must be finite and >= 0".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn ray_input_rejects_degenerate_queries() {
        assert!(check_ray_input(Vec3::ZERO, Vec3::X, 10.0).is_ok());
        assert!(matches!(
            check_ray_input(Vec3::ZERO, Vec3::ZERO, 10.0),
            Err(QueryError::InvalidInput { .. })
        ));
        assert!(matches!(
            check_ray_input(Vec3::ZERO, Vec3::X, -1.0),
            Err(QueryError::InvalidInput { .. })
        ));
        assert!(matches!(
            check_ray_input(Vec3::NAN, Vec3::X, 10.0),
            Err(QueryError::InvalidInput { .. })
        ));
    }
}
