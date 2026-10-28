//! Failures of the toy translator. These are defined rejections, not panics.

use core::fmt;

/// Why a wasm module was not lowered.
#[derive(Debug)]
pub enum TranslateError {
    /// `wasmparser` rejected the bytes, or a section was not the toy subset.
    Wasm(String),
    /// A linear-memory load, store, or memory section. Textures and buffers
    /// are imports, not `i32.load`.
    LinearMemory,
    /// A group or binding operand was not an `i32.const`.
    ///
    /// The element index of a storage access may be dynamic. A dynamic
    /// group or binding would be bindless descriptor indexing, which this
    /// translator rejects.
    DynamicBinding,
    /// An operator, import, or result shape outside the straight-line subset.
    Unsupported,
    /// `vs_main` or `fs_main` was not exported.
    MissingEntry(&'static str),
    /// naga rejected the lowered module or a round-trip.
    Naga(String),
}

impl fmt::Display for TranslateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wasm(msg) | Self::Naga(msg) => f.write_str(msg),
            Self::LinearMemory => f.write_str("linear memory is not the gpu abi"),
            Self::DynamicBinding => f.write_str("binding is not a constant immediate"),
            Self::Unsupported => f.write_str("operator is outside the toy subset"),
            Self::MissingEntry(name) => write!(f, "missing export {name}"),
        }
    }
}

impl std::error::Error for TranslateError {}
