//! `gpu_struct!` — a `repr(C)` struct whose Rust layout matches [`place_fields`](crate::place_fields).

#[doc(hidden)]
#[macro_export]
macro_rules! gpu_repeat_one {
    ($any:tt) => {
        1usize
    };
}

/// Define a `repr(C)` struct that implements [`GpuSafeLayout`](crate::GpuSafeLayout).
///
/// Field types must already be [`GpuSafeLayout`](crate::GpuSafeLayout). The
/// generated struct exposes `FIELD_OFFSETS` (`offset_of!`, declaration order)
/// and fails to compile when that layout disagrees with
/// [`place_fields`](crate::place_fields) (insert padding so a following `vec3`
/// starts at a multiple of 16).
#[macro_export]
macro_rules! gpu_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $( $(#[$fmeta:meta])* $fvis:vis $field:ident : $ty:ty ),* $(,)?
        }
    ) => {
        #[repr(C, align(16))]
        #[derive(Clone, Copy, Debug, PartialEq)]
        $(#[$meta])*
        $vis struct $name {
            $( $(#[$fmeta])* $fvis $field : $ty, )*
        }

        // SAFETY: the struct is `repr(C)` and every field is `GpuSafeLayout`,
        // so the aggregate has no niches and its size/align match the constants.
        unsafe impl $crate::GpuSafeLayout for $name {
            const SIZE: u32 = {
                let size = ::core::mem::size_of::<$name>();
                assert!(size <= u32::MAX as usize, "struct exceeds u32");
                size as u32
            };
            const ALIGN: u32 = {
                let align = ::core::mem::align_of::<$name>();
                assert!(align <= u32::MAX as usize, "align exceeds u32");
                align as u32
            };
            const STORAGE_STRIDE: u32 = $crate::storage_stride(Self::SIZE, Self::ALIGN);
            const UNIFORM_STRIDE: u32 = $crate::uniform_stride(Self::SIZE, Self::ALIGN);
            const TYPE_TAG: u8 = 0;
        }

        impl $name {
            /// Byte offset of each field, in declaration order.
            pub const FIELD_OFFSETS: [u32; 0 $(+ $crate::gpu_repeat_one!($field))*] = {
                let mut offsets = [0u32; 0 $(+ $crate::gpu_repeat_one!($field))*];
                let mut index = 0usize;
                $(
                    offsets[index] = ::core::mem::offset_of!($name, $field) as u32;
                    index += 1;
                )*
                let _ = index;
                offsets
            };

            /// `ornis_gpu.types` tag of each field, in declaration order.
            ///
            /// [`GpuSafeLayout::TYPE_TAG`](crate::GpuSafeLayout::TYPE_TAG) `0`
            /// is host padding (`[u8; N]`), not a wasm field.
            pub const FIELD_TAGS: [u8; 0 $(+ $crate::gpu_repeat_one!($field))*] = [
                $( <$ty as $crate::GpuSafeLayout>::TYPE_TAG, )*
            ];
        }

        const _: () = {
            let _ = $name::FIELD_TAGS;
            let mut gpu = $name::FIELD_OFFSETS;
            let span = $crate::place_fields(
                &[
                    $(
                        $crate::FieldLayout {
                            size: <$ty as $crate::GpuSafeLayout>::SIZE,
                            align: <$ty as $crate::GpuSafeLayout>::ALIGN,
                        },
                    )*
                ],
                true,
                &mut gpu,
            );
            assert!(span == ::core::mem::size_of::<$name>() as u32, "gpu span");
            let mut index = 0usize;
            $(
                let _ = stringify!($field);
                assert!(gpu[index] == $name::FIELD_OFFSETS[index], "gpu field offset");
                index += 1;
            )*
            let _ = index;
        };
    };
}
