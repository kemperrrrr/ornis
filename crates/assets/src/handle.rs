//! Typed asset handles.
//!
//! [`Handle<T>`] is an [`AssetId`] tagged with the asset type at compile
//! time: `Handle<Scene>` cannot be passed where `Handle<Model>` is
//! expected, yet it is `Copy` and costs exactly one id. [`Asset`] ties a
//! Rust type to its runtime [`AssetKind`] and to its storage in the
//! [`AssetServer`]; adding a new asset type means a new [`AssetKind`]
//! variant, a storage lane in the server and one `Asset` impl. `.ron` is
//! [`Scene`]; `.glb`/`.gltf` is [`Model`](crate::Model).

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use crate::scene::Scene;
use crate::server::{AssetId, AssetKind, AssetServer};

mod sealed {
    /// Storage lanes live inside [`AssetServer`](crate::AssetServer), so
    /// asset types are added in this crate.
    pub trait Sealed {}
}

/// A Rust type the [`AssetServer`] can load and store.
///
/// Sealed: storage is a private lane of the server (today scenes only).
pub trait Asset: sealed::Sealed + Sized + 'static {
    /// Runtime kind matched against [`Importer::kind`](crate::Importer::kind).
    const KIND: AssetKind;

    /// Borrows the stored asset behind `id` (`None` when absent).
    fn get(server: &AssetServer, id: AssetId) -> Option<&Self>;
}

impl sealed::Sealed for Scene {}

impl Asset for Scene {
    const KIND: AssetKind = AssetKind::Scene;

    fn get(server: &AssetServer, id: AssetId) -> Option<&Self> {
        server.get_scene(id)
    }
}

#[cfg(feature = "gltf")]
impl sealed::Sealed for crate::Model {}

#[cfg(feature = "gltf")]
impl Asset for crate::Model {
    const KIND: AssetKind = AssetKind::Model;

    fn get(server: &AssetServer, id: AssetId) -> Option<&Self> {
        server.model(id)
    }
}

/// Typed, copyable reference to a loaded asset of type `T`.
///
/// Obtained from [`AssetServer::load`]; resolve with [`AssetServer::get`].
/// Loading the same path twice yields equal handles. A handle does not
/// keep the asset alive: [`AssetServer::unload`] invalidates it (lookups
/// then return `None`).
pub struct Handle<T: Asset> {
    id: AssetId,
    marker: PhantomData<fn() -> T>,
}

impl<T: Asset> Handle<T> {
    /// Wraps a raw id (crate-internal: ids come from the server).
    pub(crate) fn from_id(id: AssetId) -> Self {
        Self {
            id,
            marker: PhantomData,
        }
    }

    /// Untyped id, for the id-based legacy API
    /// ([`AssetServer::get_scene`], [`AssetServer::model`], ...).
    pub fn id(self) -> AssetId {
        self.id
    }
}

impl<T: Asset> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Asset> Copy for Handle<T> {}

impl<T: Asset> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: Asset> Eq for Handle<T> {}

impl<T: Asset> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T: Asset> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Handle<{}>({})", T::KIND.name(), self.id.index())
    }
}

impl<T: Asset> From<Handle<T>> for AssetId {
    fn from(handle: Handle<T>) -> Self {
        handle.id
    }
}
