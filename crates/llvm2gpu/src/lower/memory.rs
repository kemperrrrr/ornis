//! Word-granular memory: labels, spill locals, pointer bases and stage outputs.

use super::UNSTABLE_COUNTER_DIGITS;
use super::WORD_BYTES;
use super::emit::scalar;
use super::stage::OutSlot;
use crate::error::TranslateError;
use crate::parse::Ty;
use std::collections::BTreeMap;

/// Name of the WGSL local that holds an LLVM value.
///
/// LLVM disambiguates inlined copies by appending an SSA-index counter to the
/// value's name — `%lo.sroa.9.01521` here, `%k.sroa.0.0.i1522` elsewhere — and
/// that counter numbers every value of the *function*, so it moves when the
/// compiler inlines differently. The same shader built on another target then
/// carries `…01519`, and the checked-in WGSL stops matching the machine that
/// generated it. A counter says nothing about the shader, so every counter
/// segment is dropped, wherever it sits: `%0..sroa.0.0.0.i554.i` puts one in
/// front of the bare inlining marker and `%15.i156.sroa.0.0` in the middle.
/// Real member indices (`lo.sroa.9.0`) are one or two digits and stay. Two
/// values that still collide are renamed by the WGSL writer in declaration
/// order, which follows the IR.
pub(crate) fn local_label(name: &str) -> String {
    name.split('.')
        .filter(|segment| *segment != INLINING_MARKER && !is_counter_segment(segment))
        .collect::<Vec<_>>()
        .join(".")
}

/// Segment LLVM appends to a value that went through an inlining step.
pub(crate) const INLINING_MARKER: &str = "i";

/// Operands of a `memset`/`memcpy` call: destination/source, value, size.
pub(crate) const MEMORY_INTRINSIC_ARGS: usize = 3;

/// Bits in one byte, the unit a `memset` fill value repeats with.
pub(crate) const BYTE_BITS: u32 = 8;
/// Bits in one word.
pub(crate) const WORD_BITS: u32 = 32;
/// Mask of one byte.
pub(crate) const BYTE_MASK: u32 = 0xFF;

/// Whether one SSA name segment is an LLVM counter rather than a field index.
///
/// An `i`-prefixed run is a counter at any length (`i11`), but a *bare* short
/// run is an aggregate member index (`lo.sroa.9.0`) and is not.
pub(crate) fn is_counter_segment(segment: &str) -> bool {
    match segment.strip_prefix(INLINING_MARKER) {
        Some(digits) => !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()),
        None => {
            segment.len() >= UNSTABLE_COUNTER_DIGITS && segment.bytes().all(|b| b.is_ascii_digit())
        }
    }
}

/// Local variable the entry returns, holding the stage output.
pub(crate) fn output_local(
    function: &mut naga::Function,
    name: &str,
    ty: naga::Handle<naga::Type>,
) -> naga::Handle<naga::LocalVariable> {
    function.local_variables.append(
        naga::LocalVariable {
            name: Some(name.to_string()),
            ty,
            init: None,
        },
        naga::Span::default(),
    )
}

#[derive(Clone, Copy)]
pub(crate) struct Helper {
    /// The Naga function to call.
    pub(crate) function: naga::Handle<naga::Function>,
    /// Its result type, `Ty::Void` for a function that returns nothing.
    pub(crate) ret: Ty,
}

/// Naga scalar type of a helper's `f32`/`i32` value.
pub(crate) fn scalar_of(module: &mut naga::Module, ty: Ty) -> Option<naga::Handle<naga::Type>> {
    match ty {
        Ty::Float => Some(scalar(module, naga::ScalarKind::Float)),
        Ty::I32 => Some(scalar(module, naga::ScalarKind::Uint)),
        _ => None,
    }
}

/// One word made of `byte` repeated, the word a `memset` writes.
pub(crate) fn repeated_byte(byte: u32) -> u32 {
    let mut word = 0u32;
    let mut shift = 0u32;
    while shift < WORD_BITS {
        word |= byte << shift;
        shift += BYTE_BITS;
    }
    word
}

/// Word count of `bytes`, which must be a multiple of four.
pub(crate) fn word_count(bytes: u32) -> Result<u32, TranslateError> {
    if bytes == 0 || !bytes.is_multiple_of(WORD_BYTES) {
        return Err(TranslateError::unsupported());
    }
    Ok(bytes / WORD_BYTES)
}

/// Where a memory value lives and how far into it the pointer is.
#[derive(Clone, Copy)]
pub(crate) struct MemoryPointer {
    /// Base of the pointer.
    pub(crate) base: MemoryBase,
    /// Offset from the base, in 4-byte words.
    pub(crate) words: naga::Handle<naga::Expression>,
}

/// What a [`MemoryPointer`] starts at.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryBase {
    /// A function-scope allocation (`alloca`).
    Local(naga::Handle<naga::LocalVariable>),
    /// A module constant, lowered to a private array of words.
    Constant(naga::Handle<naga::GlobalVariable>),
}

/// Where the entry writes its stage output.
pub(crate) struct StageOut {
    /// Local variable the entry returns.
    pub(crate) var: naga::Handle<naga::LocalVariable>,
    /// Struct member per output slot; `None` when the output is the fragment's
    /// bare `@location(0) vec4<f32>`.
    pub(crate) locations: Option<BTreeMap<OutSlot, u32>>,
    /// Struct member of `@builtin(position)`.
    pub(crate) position: Option<u32>,
}

impl StageOut {
    /// The fragment's bare `@location(0) vec4<f32>`, stored into the variable.
    pub(crate) fn bare(var: naga::Handle<naga::LocalVariable>) -> Self {
        Self {
            var,
            locations: None,
            position: None,
        }
    }

    /// A struct result, stored into the member each output slot names.
    pub(crate) fn located(
        var: naga::Handle<naga::LocalVariable>,
        locations: BTreeMap<OutSlot, u32>,
        position: Option<u32>,
    ) -> Self {
        Self {
            var,
            locations: Some(locations),
            position,
        }
    }
}
