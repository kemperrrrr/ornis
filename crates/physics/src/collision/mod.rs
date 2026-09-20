//! Collision detection: broadphase candidate pairs, shapes and distance queries.
//!
//! Groups the narrow/broad collision pieces shared by both solver engines:
//! [`broadphase`] candidate-pair backends, [`shape`] primitives with AABB
//! projection, analytic [`distance`] queries and the [`gjk`] fallback for
//! cylinder/cone/hull pairs. Moved verbatim from the crate root (phase 3);
//! external paths stay available through the crate-root re-exports.

/// Candidate-pair backends and benchmark diagnostics.
pub mod broadphase;
/// Persistent dynamic AABB-tree broadphase backend.
pub mod broadphase_tree;
/// Analytic closest-point and distance queries between convex shapes.
pub mod distance;
/// GJK/EPA fallback narrow phase for cylinder, cone and convex hull pairs.
pub(crate) mod gjk;
/// Collision shapes with AABB projection and inertia tensors.
pub mod shape;
