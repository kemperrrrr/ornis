//! Runtime mesh editing: canonical mesh data plus boolean operations.
//!
//! [`MeshData`] is the engine-wide source of truth for editable geometry:
//! render, physics and WASM views are derived from it, never the reverse.
//! Boolean kernels come from the `manifold-rust` crate (pure-Rust port of
//! the Manifold geometry library) — this crate only owns the Ornis side:
//! canonical data, edit operations, the FFI-shape bridge and frame budgets.
//!
//! Provenance: `manifold-rust 0.13`, Apache-2.0, crates.io only
//! (workspace forbids git sources),
//! upstream `https://github.com/larsbrubaker/manifold-rust`
//! (port of `https://github.com/elalish/manifold`).
//! No 3D bevel/fillet exists upstream (only 2D offset `JoinType::Bevel`);
//! subdivision upstream is midpoint — both gaps are closed on the Ornis
//! side, not by patching the kernel.

pub mod bridge;
pub mod dirty;
pub mod editable;
pub mod exact;
pub mod mesh_data;
pub mod normals;
pub mod ops;
pub mod stats;

pub use bridge::{BridgeError, boolean, from_manifold, to_manifold};
pub use dirty::MeshDirty;
pub use editable::EditableMesh;
pub use exact::{ExactOp, ExactResult, ExactWorker};
pub use mesh_data::{MeshData, MeshError};
pub use normals::{recompute_normals, to_physics_arrays};
pub use ops::{BooleanKind, EditOp};
pub use stats::FrameStats;
