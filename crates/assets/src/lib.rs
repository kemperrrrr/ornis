//! Asset descriptions, colliders and the asset manager.
//!
//! This crate owns what every domain reads but nobody owns: the
//! serde-canonical scene/component descriptions ([`scene`]), the explicit
//! collider recipes ([`collider`]), the glTF→[`scene`] wiring ([`import`])
//! and the asset registry ([`server`]). Baked projections stay with their
//! consumers (GPU meshes in `ornis-render`, solver bodies in
//! `ornis-physics`); this crate holds sources and identity, never copies.

#![warn(missing_docs)]

/// Explicit collider recipes and the mesh→collider mapping.
pub mod collider;
/// glTF→[`scene`] wiring (geometry + scalar materials; textures deferred).
pub mod import;
/// RON-serializable scene description types (moved from `ornis-render`).
pub mod scene;
/// Typed asset registry: ids, handles, events and the reload dirty-set.
pub mod server;

pub use collider::{ColliderDesc, collider_for};
pub use import::scene_from_gltf;
pub use scene::{CameraDesc, EntityDesc, LightDesc, MaterialDesc, MeshDesc, Scene, TransformDesc};
pub use server::{
    AssetEvent, AssetId, AssetKind, AssetServer, MaterialHandle, MeshHandle, SceneLoadError,
    parse_scene_ron,
};
