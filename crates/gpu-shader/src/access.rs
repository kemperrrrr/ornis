//! Marker access modes for [`Storage`](crate::Storage).
//!
//! The access mode lives in the type, not in a runtime flag: a read-only
//! binding has no `store` method, so a kernel that tries to write one fails to
//! compile instead of failing bind-group validation later.

/// Access mode a storage binding is declared with.
///
/// This trait is sealed: only [`ReadOnly`] and [`ReadWrite`] implement it.
pub trait StorageAccess: sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::ReadOnly {}
    impl Sealed for super::ReadWrite {}
}

/// Read-only storage binding (`var<storage>` without `read_write`).
///
/// Kernels cannot write through a read-only binding. The translator emits
/// `var<storage>` (read) for the same reason: a write through it is a WGSL
/// validation error, so the restriction is enforced on both sides.
pub enum ReadOnly {}

impl StorageAccess for ReadOnly {}

/// Read-write storage binding (`var<storage, read_write>`).
pub enum ReadWrite {}

impl StorageAccess for ReadWrite {}
