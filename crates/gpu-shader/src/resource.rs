//! Typed storage bindings.
//!
//! [`Storage<T, G, B, A>`](Storage) is a zero-sized description of one
//! `var<storage>` binding; the element type `T` owns the ABI. Kernels write the
//! short form (`Storage<f32>`, `Storage<u32, ReadWrite>`) and `#[gpu_entry]`
//! fills in the group, the binding, and the access mode from the
//! `#[gpu(…)]` attribute, so a binding's address is written once, where the
//! resource is declared, and never again in the body.
//!
//! `T` is what one element of the buffer is *in the shader* — [`f32`], [`u32`],
//! [`AtomicU32`], [`Vec4U32`] — not the host record: a 208-byte `PerObject` is
//! read as `Storage<f32>` word by word, exactly as the hand written kernels did.
//! The element types implement [`StorageLen`]/[`StorageLoad`]/[`StorageStore`]
//! and own the import each one lowers to, so adding an element type is two trait
//! impls rather than an edit to this facade.

use crate::access::ReadOnly;
use crate::access::ReadWrite;
use crate::access::StorageAccess;
use crate::elements::AtomicU32;
use crate::elements::StorageLen;
use crate::elements::StorageLoad;
use crate::elements::StorageStore;

/// Typed view of a storage buffer `(G, B)`.
///
/// `A` is the access mode: a [`ReadOnly`] binding has no `store`, a
/// [`ReadWrite`] binding does. A mismatch with the host bind group is a
/// validation error, which this type cannot catch; a mismatch with the declared
/// access mode is a compile error, which it can.
pub struct Storage<T, const G: u32, const B: u32, A: StorageAccess = ReadOnly> {
    element: core::marker::PhantomData<fn() -> T>,
    access: core::marker::PhantomData<fn() -> A>,
}

impl<T, const G: u32, const B: u32, A: StorageAccess> Storage<T, G, B, A> {
    /// Bind the parameter that `#[gpu_entry]` stripped from the signature.
    ///
    /// The attribute expansion is the only caller: a kernel never constructs a
    /// binding value, which keeps the address a property of the attribute.
    #[doc(hidden)]
    #[inline(always)]
    pub const fn __bind() -> Self {
        Self {
            element: core::marker::PhantomData,
            access: core::marker::PhantomData,
        }
    }

    /// The `(group, binding)` pair, for diagnostics and tests.
    #[inline(always)]
    pub const fn address() -> (u32, u32) {
        (G, B)
    }
}

impl<T: StorageLen, const G: u32, const B: u32, A: StorageAccess> Storage<T, G, B, A> {
    /// Element count of this binding (`arrayLength`), in elements of `T`.
    #[inline(always)]
    pub fn len(&self) -> u32 {
        let _ = self;
        T::len(G, B)
    }

    /// Whether the binding holds no elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: StorageLoad, const G: u32, const B: u32, A: StorageAccess> Storage<T, G, B, A> {
    /// Read element `index`.
    #[inline(always)]
    pub fn load(&self, index: u32) -> T::Value {
        let _ = self;
        T::load(G, B, index)
    }
}

impl<T: StorageStore, const G: u32, const B: u32> Storage<T, G, B, ReadWrite> {
    /// Write element `index`. Only a read-write binding has this method.
    #[inline(always)]
    pub fn store(&self, index: u32, value: T::Value) {
        let _ = self;
        T::store(G, B, index, value);
    }
}

impl<const G: u32, const B: u32> Storage<AtomicU32, G, B, ReadWrite> {
    /// Reserve a slot in an atomic binding; returns the previous word.
    #[inline(always)]
    pub fn atomic_add(&self, index: u32, value: u32) -> u32 {
        let _ = self;
        AtomicU32::atomic_add(G, B, index, value)
    }
}

impl<T, const G: u32, const B: u32, A: StorageAccess> Clone for Storage<T, G, B, A> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const G: u32, const B: u32, A: StorageAccess> Copy for Storage<T, G, B, A> {}

const _: () = {
    // A binding is a compile-time fact about the signature, not a value: the
    // entry macro binds it to nothing and the methods take `&self` only so the
    // body reads like a method call.
    assert!(core::mem::size_of::<Storage<f32, 0, 0, ReadOnly>>() == 0);
    assert!(core::mem::size_of::<Storage<u32, 0, 0, ReadWrite>>() == 0);
};
