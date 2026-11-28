//! Failures of the LLVM IR translator. These are defined rejections, not panics.

use core::fmt;

/// Why an LLVM IR module was not lowered.
#[derive(Debug)]
pub enum TranslateError {
    /// The IR text is malformed or outside the supported subset.
    Parse(String),
    /// An operator, intrinsic, or result shape outside the supported subset.
    ///
    /// Carries the site the translator was lowering — function, block and the
    /// instruction text — so a rejection points at the line of IR to look at
    /// instead of at the whole kernel. The translator builds it through a
    /// private constructor, so the site travels automatically.
    Unsupported(String),
    /// A storage element the lowerer will not emit, or a layout mismatch.
    UnsupportedType(String),
    /// Naga rejected the lowered module or a round-trip.
    Naga(String),
}

impl TranslateError {
    /// The subset rejected what the translator is looking at right now.
    ///
    /// The position comes from [`crate::site`], which the lowerer updates as it
    /// walks the module, so call sites name no arguments and still report a
    /// precise place.
    pub(crate) fn unsupported() -> Self {
        Self::Unsupported(crate::site::describe())
    }
}

impl fmt::Display for TranslateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(msg) => f.write_str(msg),
            Self::Unsupported(site) => {
                write!(f, "llvm ir is outside the supported subset {site}")
            }
            Self::UnsupportedType(msg) => f.write_str(msg),
            Self::Naga(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for TranslateError {}
