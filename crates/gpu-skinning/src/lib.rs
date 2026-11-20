//! Skinned g-buffer vertex stage on the stable Rust → LLVM IR → WGSL path.
//!
//! The stage blends bind-pose vertices through a joint palette (linear blend
//! skinning), transforms the result by the instance matrices and writes the
//! varyings the shared g-buffer fragment stage reads. Two palettes are bound —
//! this frame's and the previous one's — because the motion vector needs the
//! previous skin.
//!
//! - [`layout`] holds the record layouts and the camera uniform projection.
//! - [`vertex`] blends, transforms and writes the stage interface.
//!
//! `xtask build-shaders --package ornis-gpu-skinning --entry vs_main_skinned`
//! writes the checked-in artifact.

#![no_std]
#![warn(missing_docs)]

pub mod layout;
pub mod vertex;

pub use layout::Camera;
pub use vertex::vs_main_skinned;
