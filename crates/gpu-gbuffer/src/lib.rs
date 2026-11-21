//! G-buffer pass on the stable Rust → LLVM IR → WGSL path.
//!
//! Two vertex entries transform one mesh instance with its own matrices and
//! write the world-space varyings; `vs_main_indirect` first reads the scene
//! index out of the compacted visibility list. The fragment stage reads one
//! record of the material table and fills the five lighting targets plus the
//! two temporal ones (motion vector and the quantized previous depth the TAA
//! resolve reads).
//!
//! - [`layout`] holds the record layouts and the camera uniform projection.
//! - [`vertex`] transforms a vertex and writes the stage interface.
//! - [`fragment`] fills the seven attachments.
//!
//! `xtask build-shaders --package ornis-gpu-gbuffer --entry all` writes the
//! checked-in artifact; `--entry vs_main` the vertex-only module the
//! forward-PBR pass shares.

#![no_std]
#![warn(missing_docs)]

pub mod fragment;
pub mod layout;
pub mod vertex;

pub use fragment::fs_main;
pub use layout::Camera;
pub use vertex::{vs_main, vs_main_indirect};
