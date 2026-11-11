//! Failures of the LLVM IR translator. These are defined rejections, not panics.

use core::fmt;

/// Why an LLVM IR module was not lowered.
#[derive(Debug)]
pub enum TranslateError {
    /// The IR text is malformed or outside the supported subset.
    Parse(String),
    /// An operator, intrinsic, or result shape outside the supported subset.
    Unsupported,
    /// A storage element the lowerer will not emit, or a layout mismatch.
    UnsupportedType(String),
    /// `vs_main` or `fs_main` was not emitted.
    MissingEntry(&'static str),
    /// Naga rejected the lowered module or a round-trip.
    Naga(String),
}

impl fmt::Display for TranslateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(msg) => f.write_str(msg),
            Self::Unsupported => f.write_str("llvm ir is outside the supported subset"),
            Self::UnsupportedType(msg) => f.write_str(msg),
            Self::MissingEntry(name) => write!(f, "missing entry {name}"),
            Self::Naga(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for TranslateError {}
