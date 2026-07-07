#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub fn prefetch_read<T>(ptr: *const T, locality: i32) {
    unsafe { std::arch::x86_64::_mm_prefetch(ptr as *const i8, locality) }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
pub fn prefetch_read<T>(_ptr: *const T, _locality: i32) {}

pub(crate) const PREFETCH_STRIDE: usize = 8;

#[macro_export]
macro_rules! prefetch_iter {
    ($iter:expr, $stride:expr) => {{
        let mut count = 0;
        let stride = $stride;
        $iter.inspect(move |item| {
            if count % stride == 0 {
                let ptr = item as *const _;
                $crate::prefetch::prefetch_read(ptr, 3);
            }
            count += 1;
        })
    }};
}

pub(crate) use prefetch_iter;
