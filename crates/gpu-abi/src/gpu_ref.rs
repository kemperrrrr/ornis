//! Typed buffer tokens. Group and binding stay in the type; the value is an offset.
//!
//! [`GpuRef`] does not implement [`Deref`](core::ops::Deref). A Rust reference
//! would have to point at host linear memory, and a GPU buffer is not that.
//! Call [`GpuRef::read`] or [`GpuRef::write`].

use core::marker::PhantomData;

use crate::gpu_abi;
use crate::layout::GpuSafeLayout;

/// Marker for a uniform binding. Shader code can read it and cannot write it.
#[derive(Clone, Copy, Debug)]
pub enum UniformSpace {}

/// Marker for a storage binding. Shader code can read and write it.
#[derive(Clone, Copy, Debug)]
pub enum StorageSpace {}

/// Offset token for the value `T` inside binding `(GROUP, BINDING)`.
///
/// The token is four bytes: the byte offset. `GROUP` and `BINDING` are not
/// stored in those bits. `'a` ties the token to the [`Uniform`] or [`Storage`]
/// value it was derived from; it does not borrow host shadow memory.
pub struct GpuRef<'a, const GROUP: u32, const BINDING: u32, T: GpuSafeLayout, S> {
    offset: u32,
    _marker: PhantomData<(&'a T, S)>,
}

impl<const GROUP: u32, const BINDING: u32, T: GpuSafeLayout, S> Clone
    for GpuRef<'_, GROUP, BINDING, T, S>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<const GROUP: u32, const BINDING: u32, T: GpuSafeLayout, S> Copy
    for GpuRef<'_, GROUP, BINDING, T, S>
{
}

impl<'a, const G: u32, const B: u32, T: GpuSafeLayout, S> GpuRef<'a, G, B, T, S> {
    /// Read `T` at this offset. On the host this copies out of the shadow
    /// binding installed by [`gpu_abi::with_host`](crate::gpu_abi::with_host).
    ///
    /// # Panics
    ///
    /// On the host, panics if no shadow is installed, the binding is missing,
    /// the offset is misaligned for `T`, or the read would pass the end of the
    /// binding. Those checks run before the byte copy.
    pub fn read(&self) -> T {
        let _ = self;
        gpu_abi::read_at::<T>(G, B, self.offset)
    }

    /// Token for a field at `field_offset` bytes past this token.
    ///
    /// Does not load GPU memory. Offsets come from [`gpu_struct!`](crate::gpu_struct)
    /// / `offset_of!`, not from a caller-supplied raw pointer. A field offset
    /// that is not a multiple of `U::ALIGN` never becomes a token: it would
    /// hand the pointer path a misaligned address, so it is a contract
    /// breach.
    #[inline(always)]
    pub fn field<U: GpuSafeLayout>(&self, field_offset: u32) -> GpuRef<'a, G, B, U, S> {
        if !field_offset.is_multiple_of(U::ALIGN) {
            gpu_abi::contract_breach("GpuRef field offset is misaligned");
        }
        GpuRef {
            offset: add_offset(self.offset, field_offset),
            _marker: PhantomData,
        }
    }

    /// Token for element `element` given an explicit byte `stride`.
    ///
    /// The stride is not inferred: uniform and storage arrays disagree for
    /// some types. Prefer [`index_uniform`](Self::index_uniform) or
    /// [`index_storage`](Self::index_storage). A stride that is not a
    /// multiple of [`T::ALIGN`](GpuSafeLayout::ALIGN) is a contract breach:
    /// the product would be a misaligned byte offset.
    #[inline(always)]
    pub fn index(&self, element: u32, stride: u32) -> Self {
        if !stride.is_multiple_of(T::ALIGN) {
            gpu_abi::contract_breach("GpuRef index stride is misaligned");
        }
        GpuRef {
            offset: add_offset(self.offset, mul_offset(element, stride)),
            _marker: PhantomData,
        }
    }

    /// [`index`](Self::index) with [`T::UNIFORM_STRIDE`](GpuSafeLayout::UNIFORM_STRIDE).
    // qual:api — shader-facing index; in-tree callers are the unit tests.
    #[inline(always)]
    pub fn index_uniform(&self, element: u32) -> Self {
        self.index(element, T::UNIFORM_STRIDE)
    }

    /// [`index`](Self::index) with [`T::STORAGE_STRIDE`](GpuSafeLayout::STORAGE_STRIDE).
    #[inline(always)]
    pub fn index_storage(&self, element: u32) -> Self {
        self.index(element, T::STORAGE_STRIDE)
    }
}

impl<'a, const G: u32, const B: u32, T: GpuSafeLayout> GpuRef<'a, G, B, T, StorageSpace> {
    /// Write `value` at this offset. Uniform tokens do not have this method.
    ///
    /// # Panics
    ///
    /// On the host, panics if the binding was not registered as storage, or
    /// the same bounds and alignment checks as [`read`](GpuRef::read) fail.
    pub fn write(&self, value: T) {
        let _ = self;
        gpu_abi::write_at::<T>(G, B, self.offset, &value);
    }

    /// Byte address of this token in shader code: the `storage_base` tag
    /// for `(G, B)` plus the offset. Storage tokens only — a uniform
    /// binding has no pointer path — and the method exists only where the
    /// shader ABI does (`wasm32`).
    ///
    /// The wasm lowering keeps that provenance through `i32.add` and turns
    /// `i32.load` / `i32.store` at the result into an access chain on the
    /// binding — the pointer path of
    /// `docs/rendering/wasm-segmented-memory.md` §4.
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    pub fn addr(&self) -> u32 {
        gpu_abi::storage_base::<G, B>().wrapping_add(self.offset)
    }

    /// Pointer-path read at `addr()`: one `i32.load` through
    /// the storage provenance, lowered to that binding's access chain.
    ///
    /// Only 4-byte values; a wider type is a contract breach. The import
    /// path with the same result is [`read`](GpuRef::read), which the host
    /// uses too.
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    pub fn load(&self) -> T {
        gpu_abi::ptr_load::<T>(self.addr())
    }

    /// Pointer-path write at `addr()`: one `i32.store` through
    /// the storage provenance, lowered to that binding's access chain.
    ///
    /// Only 4-byte values; a wider type is a contract breach.
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    pub fn store(&self, value: T) {
        gpu_abi::ptr_store::<T>(self.addr(), value);
    }
}

/// Uniform binding `(GROUP, BINDING)` of `T`.
#[derive(Clone, Copy, Debug)]
pub struct Uniform<const GROUP: u32, const BINDING: u32, T: GpuSafeLayout> {
    _marker: PhantomData<T>,
}

impl<const G: u32, const B: u32, T: GpuSafeLayout> Uniform<G, B, T> {
    /// Compile-time proof that `T` matches WGSL uniform (`std140`) placement.
    ///
    /// Evaluated when [`bind`](Self::bind) is monomorphized, so a mismatched
    /// `T` fails the build with the [`LayoutMismatch`](crate::LayoutMismatch)
    /// message.
    const LAYOUT: () = match crate::layout::uniform_layout_mismatch::<T>() {
        None => {}
        Some(mismatch) => mismatch.panic_bind(),
    };

    /// Bind this uniform. The handle carries no address.
    ///
    /// # Panics
    ///
    /// At compile time, when `T`'s size, alignment, or strides disagree with
    /// the WGSL `std140` rules (see [`uniform_layout_mismatch`](crate::uniform_layout_mismatch)).
    pub const fn bind() -> Self {
        let () = Self::LAYOUT;
        Self {
            _marker: PhantomData,
        }
    }

    /// Token at offset 0.
    ///
    /// The pointer path (`addr` / `load`) lives only on storage tokens, so
    /// a uniform token cannot open one:
    ///
    /// ```compile_fail
    /// let uniform = ornis_gpu_abi::Uniform::<0, 0, f32>::bind();
    /// let _loaded = uniform.as_ref().load();
    /// ```
    #[inline(always)]
    pub fn as_ref(&self) -> GpuRef<'_, G, B, T, UniformSpace> {
        let _ = self;
        GpuRef {
            offset: 0,
            _marker: PhantomData,
        }
    }

    /// Read the whole uniform.
    pub fn read(&self) -> T {
        self.as_ref().read()
    }
}

/// Storage binding `(GROUP, BINDING)` of `T`.
#[derive(Clone, Copy, Debug)]
pub struct Storage<const GROUP: u32, const BINDING: u32, T: GpuSafeLayout> {
    _marker: PhantomData<T>,
}

impl<const G: u32, const B: u32, T: GpuSafeLayout> Storage<G, B, T> {
    /// Compile-time proof that `T` matches WGSL storage (`std430`) placement.
    ///
    /// Evaluated when [`bind`](Self::bind) is monomorphized, so a mismatched
    /// `T` fails the build with the [`LayoutMismatch`](crate::LayoutMismatch)
    /// message.
    const LAYOUT: () = match crate::layout::storage_layout_mismatch::<T>() {
        None => {}
        Some(mismatch) => mismatch.panic_bind(),
    };

    /// Bind this storage buffer. The handle carries no address.
    ///
    /// # Panics
    ///
    /// At compile time, when `T`'s size, alignment, or strides disagree with
    /// the WGSL `std430` rules (see [`storage_layout_mismatch`](crate::storage_layout_mismatch)).
    pub const fn bind() -> Self {
        let () = Self::LAYOUT;
        Self {
            _marker: PhantomData,
        }
    }

    /// Token at offset 0.
    #[inline(always)]
    pub fn as_ref(&self) -> GpuRef<'_, G, B, T, StorageSpace> {
        let _ = self;
        GpuRef {
            offset: 0,
            _marker: PhantomData,
        }
    }

    /// Read the whole storage value.
    pub fn read(&self) -> T {
        self.as_ref().read()
    }

    /// Write the whole storage value.
    pub fn write(&self, value: T) {
        self.as_ref().write(value);
    }
}

fn add_offset(base: u32, delta: u32) -> u32 {
    match base.checked_add(delta) {
        Some(offset) => offset,
        None => gpu_abi::contract_breach("GpuRef offset overflow"),
    }
}

fn mul_offset(element: u32, stride: u32) -> u32 {
    match element.checked_mul(stride) {
        Some(offset) => offset,
        None => gpu_abi::contract_breach("GpuRef index overflow"),
    }
}
