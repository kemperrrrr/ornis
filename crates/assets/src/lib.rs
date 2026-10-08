//! Asset descriptions, colliders and the asset manager.
//!
//! This crate owns what every domain reads but nobody owns: the
//! serde-canonical scene/component descriptions ([`scene`]), the explicit
//! collider recipes ([`collider`]), the glTF→[`scene`] wiring (`import`,
//! feature `gltf`), pluggable format importers ([`importer`]), typed
//! handles ([`handle`]), the format-neutral [`AssetError`] and the asset
//! registry ([`server`]). Baked projections stay with their
//! consumers (GPU meshes in `ornis-render`, solver bodies in
//! `ornis-physics`); this crate holds sources and identity, never copies.

#![warn(missing_docs)]

/// Explicit collider recipes and the mesh→collider mapping.
pub mod collider;
/// Format-neutral asset loading error.
pub mod error;
/// Typed handles ([`Handle`]) and the [`Asset`] trait.
pub mod handle;
/// glTF [`Model`](ornis_gltf::Model) → flat [`scene`] wiring.
#[cfg(feature = "gltf")]
pub mod import;
/// Format importers and the extension registry.
pub mod importer;
/// RON-serializable scene description types (moved from `ornis-render`).
pub mod scene;
/// Typed asset registry: ids, events and the reload dirty-set.
pub mod server;
#[cfg(not(feature = "gltf"))]
mod tri;
mod wire;

pub use collider::{ColliderDesc, collider_for};
pub use error::AssetError;
pub use handle::{Asset, Handle};
#[cfg(feature = "gltf")]
pub use import::scene_from_model;
#[cfg(feature = "fbx")]
pub use importer::FbxImporter;
#[cfg(feature = "gltf")]
pub use importer::GltfImporter;
pub use importer::{ImportedAsset, Importer, ImporterRegistry, RonSceneImporter, SceneImport};
#[cfg(feature = "gltf")]
pub use ornis_gltf::{Model, ModelNode, ModelPrimitive, NodeIdx};
pub use scene::{
    CameraDesc, CameraProjection, EntityDesc, LightDesc, MaterialDesc, MeshDesc, Scene,
    TransformDesc, TriIndex, Triangle,
};
pub use server::{AssetEvent, AssetId, AssetKind, AssetServer, SceneLoadError, parse_scene_ron};
