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
    /// An operator, import, or result shape outside the supported subset.
    Unsupported,
    /// Two predecessors disagree on the binding a pointer names.
    ///
    /// The offset may be a phi. The group and binding may not.
    AmbiguousProvenance,
    /// `br` or `br_if` targets the function body.
    ///
    /// Wasm treats the function as an implicit block, so that branch is a
    /// `return`. This translator does not lower it: a return from inside a
    /// nested `if` would have to compose the shader outputs on the way out.
    FunctionBranch,
    /// A direct call reaches a function already on the lowering stack.
    ///
    /// Mutual recursion is the same error. The callee is not inlined and is
    /// not emitted as a naga function.
    Recursion,
    /// `call_indirect`, or a table that would feed one.
    ///
    /// The callee would not be a concrete function, so provenance could not
    /// be specialized.
    IndirectCall,
    /// Too many specialized functions, or too many distinct call interpretations.
    ///
    /// Each distinct tuple of constant `i32` arguments is one naga function.
    /// The same tuple is cached. Abstract interpretation counts each new
    /// `(function, arguments)` pair, not each instruction in a body.
    Budget,
    /// A storage element the lowerer will not emit, or a struct whose lanes
    /// do not match the host layout rules.
    UnsupportedType(String),
    /// The `ornis_gpu.entries` custom section is malformed, or it names an
    /// entry whose stage or dispatch shape the translator cannot use.
    EntryMetadata(String),
    /// `vs_main` or `fs_main` was not exported.
    MissingEntry(&'static str),
    /// naga rejected the lowered module or a round-trip.
    Naga(String),
    /// A workgroup or storage barrier is not reached by every invocation.
    ///
    /// `barrier` is the import name. [`BarrierSite`] is the construct the
    /// analysis proved divergent. A predicate the analysis cannot prove
    /// (for example a value loaded from memory) stays the ABI precondition
    /// and does not produce this error.
    DivergentBarrier {
        /// `workgroup_barrier` or `storage_barrier`.
        barrier: &'static str,
        /// The divergent construct that reaches the call.
        site: BarrierSite,
    },
}

/// Where a barrier call sits when control flow is not uniform.
///
/// Uniform here means every invocation that entered the function is still
/// together. The translator reports a site only when the compute invocation
/// id proves that is no longer true.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierSite {
    /// The call is inside a branch on an invocation-dependent condition.
    Branch,
    /// The call is inside a loop whose continue or exit depends on the invocation.
    Loop,
    /// An invocation-dependent `return` skips the call.
    EarlyReturn,
}

impl fmt::Display for TranslateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wasm(msg) | Self::Naga(msg) => f.write_str(msg),
            Self::LinearMemory => f.write_str("linear memory is not the gpu abi"),
            Self::DynamicBinding => f.write_str("binding is not a constant immediate"),
            Self::Unsupported => f.write_str("operator is outside the toy subset"),
            Self::AmbiguousProvenance => f.write_str("ambiguous provenance at a join"),
            Self::FunctionBranch => f.write_str("branch to the function is not lowered"),
            Self::Recursion => f.write_str("recursive call is not lowered"),
            Self::IndirectCall => f.write_str("indirect call is not lowered"),
            Self::Budget => f.write_str("specialization budget exceeded"),
            Self::UnsupportedType(msg) => f.write_str(msg),
            Self::EntryMetadata(msg) => write!(f, "entry metadata: {msg}"),
            Self::MissingEntry(name) => write!(f, "missing export {name}"),
            Self::DivergentBarrier { barrier, site } => write!(f, "{}", site.message(barrier)),
        }
    }
}

impl BarrierSite {
    /// Sentence that names `barrier` and why its control flow is not uniform.
    pub fn message(self, barrier: &str) -> String {
        match self {
            Self::Branch => format!(
                "{barrier} is not uniform: the call is inside a branch on an invocation-dependent condition"
            ),
            Self::Loop => format!(
                "{barrier} is not uniform: the call is inside a loop controlled by an invocation-dependent condition"
            ),
            Self::EarlyReturn => {
                format!("{barrier} is not uniform: an invocation-dependent return skips the call")
            }
        }
    }
}

impl std::error::Error for TranslateError {}
