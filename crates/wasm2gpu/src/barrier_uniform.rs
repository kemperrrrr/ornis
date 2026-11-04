//! Static check that a barrier is reached in uniform control flow.
//!
//! `workgroup_barrier` and `storage_barrier` are workgroup synchronization
//! points: every invocation must reach the call in the same order. This pass
//! rejects a call when the compute invocation id (`cs_main`'s parameter, the
//! lowered `global_invocation_id.x`) proves that some invocations do not.
//! The proof follows copies and injective `i32` updates of that id through
//! `if`, `br_if`, `loop`, and direct calls.
//!
//! A predicate this pass cannot tie to the invocation id — a memory load, an
//! unknown parameter, a comparison of two copies of the id — stays the
//! documented ABI precondition and is allowed through. The walk runs only
//! when the module imports a barrier, so shaders that never synchronize do
//! not pay for it.

use crate::error::BarrierSite;
use crate::error::TranslateError;
use crate::ir::Func;
use crate::ir::Import;
use crate::ir::Kind;
use ornis_gpu_abi::symbols;

mod call;
mod merge;
mod region;
mod tag;

use tag::Tag;

/// Reject proven non-uniform barriers in `func` and the helpers it calls.
///
/// Returns immediately when `imports` has no barrier. Vertex and fragment
/// entries take that path in modules that do not synchronize, and also when
/// they do: the walk is linear in the body and does not run the naga emitter.
pub(super) fn check(func: &Func, imports: &[Import], funcs: &[Func]) -> Result<(), TranslateError> {
    if !imports
        .iter()
        .any(|item| barrier_name(&item.name).is_some())
    {
        return Ok(());
    }
    let mut uni = Uni {
        imports,
        funcs,
        active: Vec::new(),
        labels: Vec::new(),
        st: entry_state(func),
    };
    uni.walk(&func.body)
}

pub(super) fn barrier_name(name: &str) -> Option<&'static str> {
    if name == symbols::WORKGROUP_BARRIER {
        Some(symbols::WORKGROUP_BARRIER)
    } else if name == symbols::STORAGE_BARRIER {
        Some(symbols::STORAGE_BARRIER)
    } else {
        None
    }
}

fn entry_state(func: &Func) -> St {
    let mut locals = vec![Tag::Uniform; func.locals.len()];
    let mut index = 0usize;
    let params = func.sig.params as usize;
    while index < params && index < locals.len() {
        locals[index] = param_tag(func, index);
        index += 1;
    }
    St::fresh(locals)
}

/// `cs_main`'s first parameter is `global_invocation_id.x`.
///
/// Every other parameter is uniform only if a caller proves it. At an entry
/// that caller does not exist, so the tag stays unknown and cannot reject.
fn param_tag(func: &Func, index: usize) -> Tag {
    if index == 0 && func.export == symbols::CS_MAIN && func.locals[index] == Kind::I32 {
        Tag::Inv
    } else {
        Tag::Unknown
    }
}

struct Uni<'a> {
    imports: &'a [Import],
    funcs: &'a [Func],
    /// Defined-function indices on the call stack. A repeat is recursion.
    active: Vec<u32>,
    labels: Vec<Lab>,
    st: St,
}

#[derive(Clone)]
struct St {
    stack: Vec<Tag>,
    locals: Vec<Tag>,
    split: Option<Split>,
    leave: Leave,
}

impl St {
    fn fresh(locals: Vec<Tag>) -> Self {
        Self {
            stack: Vec::new(),
            locals,
            split: None,
            leave: Leave::Fall,
        }
    }
}

/// A split stays until `at` is popped. `None` lasts for the rest of the function.
#[derive(Clone, Copy)]
struct Split {
    site: BarrierSite,
    at: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Leave {
    Fall,
    Return,
    /// Branch to the label that was at `target` when the branch was taken.
    Br {
        target: usize,
    },
    /// `br_table`, or two arms that leave toward different labels.
    Gone,
}

struct Lab {
    kind: LabKind,
    /// Entered while every invocation was still together.
    ///
    /// The loop rewalk that rejects barriers inside a divergent loop clears
    /// this, so the rewalk does not mark the loop a second time.
    clear: bool,
    varies: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LabKind {
    Block,
    Loop,
    If,
}

impl Lab {
    fn new(kind: LabKind, clear: bool) -> Self {
        Self {
            kind,
            clear,
            varies: false,
        }
    }
}
