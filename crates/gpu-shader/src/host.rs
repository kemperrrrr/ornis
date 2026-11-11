//! Host stubs for the SDK methods.
//!
//! The typed resource methods are only meaningful in a shader: a host build has
//! no bind group behind `binding = 1`. A kernel crate is still checked on the
//! host, so the methods have to compile there. They panic instead of returning
//! a made-up value, because a silent zero would hide a host-side call that is
//! always a bug.

#![cfg(not(target_arch = "wasm32"))]

/// Diverging stub behind every resource method outside `wasm32`.
///
/// The message names the shader SDK so the panic is not mistaken for a host
/// ABI bug: storage and uniform reads only exist inside a dispatched shader.
#[cold]
#[inline(never)]
#[track_caller]
#[allow(
    clippy::panic,
    reason = "the stub cannot return a value and has no caller to report to"
)]
pub(crate) fn unavailable() -> ! {
    panic!(
        "ornis-gpu-shader: a shader operation requires a wasm32 shader build; \
         the host build of a kernel crate cannot run it"
    )
}
