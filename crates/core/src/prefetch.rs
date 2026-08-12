//! Software prefetch helpers for the component store iteration hot path.
//!
//! `_mm_prefetch` takes its locality hint as a compile-time constant
//! (`const LOCALITY: i32`), so the public helper is generic over it; the
//! non-x86 fallback is an empty function so the call sites stay portable.
//!
//! The x86 bodies are pure CPU hints with no observable semantics, so
//! mutation testing cannot distinguish a removed body from the real one —
//! they are skipped via `#[mutants::skip]`.

#[cfg(target_arch = "x86_64")]
#[mutants::skip]
pub fn prefetch_read<T, const LOCALITY: i32>(ptr: *const T) {
    unsafe { std::arch::x86_64::_mm_prefetch::<LOCALITY>(ptr as *const i8) }
}

#[cfg(target_arch = "x86")]
#[mutants::skip]
pub fn prefetch_read<T, const LOCALITY: i32>(ptr: *const T) {
    unsafe { std::arch::x86::_mm_prefetch::<LOCALITY>(ptr as *const i8) }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
pub fn prefetch_read<T, const LOCALITY: i32>(_ptr: *const T) {}

pub(crate) const PREFETCH_STRIDE: usize = 8;

#[macro_export]
macro_rules! prefetch_iter {
    ($iter:expr, $stride:expr) => {{
        let mut count = 0;
        let stride = $stride;
        $iter.inspect(move |item| {
            if count % stride == 0 {
                let ptr = item as *const _;
                $crate::prefetch::prefetch_read::<_, 3>(ptr);
            }
            count += 1;
        })
    }};
}

pub(crate) use prefetch_iter;
