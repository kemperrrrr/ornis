//! Static check that a barrier is reached in uniform control flow.
//!
//! `workgroup_barrier` and `storage_barrier` are workgroup synchronization
//! points. This pass rejects a call when it can prove that not every
//! `cs_main` invocation reaches it. The proof follows the invocation id
//! (`global_invocation_id.x`), copies (`local.set`, `local.tee`, and helper
//! results), injective `i32` updates, `f32` arithmetic on that id, and
//! `select` with that id as the condition, through `if`, `br_if`,
//! `br_table`, `loop` (including values carried across the back edge), and
//! direct calls. A value defined inside that split is invocation-dependent
//! too: a `local.set` / `local.tee`, an `if` result, a store after
//! `br_if` / `br_table`, or a counter in a loop whose trip count depends on
//! the id. The same rule applies inside a helper. An `if` whose arms leave
//! through different labels, or through `br` and `return`, keeps the split
//! on the continuation that only some invocations reach. The same exits are
//! still walked when the condition is not the id, so a later proved barrier
//! is not skipped, and a `br` arm's locals meet the other arm at the label.
//!
//! A `br_table` whose index is uniform or unknown is explored as a branch to
//! each target, so later proved divergence is not skipped. An index that is
//! the invocation id opens a split until the outermost target. Ordered
//! comparisons (`id < n`) and values loaded from memory are not proved and
//! stay the ABI precondition.
//!
//! The walk runs only when the module imports a barrier. Direct calls are
//! memoized by `(function, incoming split, taint, argument tags)` and the callee
//! body is borrowed, so a deep call dag is not exponential. A `cs_main`
//! whose workgroup is 1×1×1 is skipped: one invocation cannot diverge.

use crate::error::BarrierSite;
use crate::error::TranslateError;
use crate::ir::Func;
use crate::ir::Import;
use crate::ir::Kind;
use ornis_gpu_abi::symbols;

mod call;
mod exit;
mod loops;
mod merge;
mod region;
mod tag;

use tag::Tag;

/// Reject proven non-uniform barriers in `func` and the helpers it calls.
///
/// Returns immediately when `imports` has no barrier, and when `func` is
/// `cs_main` of a 1×1×1 workgroup. Vertex and fragment entries have no
/// invocation id, so they cannot prove divergence. The walk does not run
/// the naga emitter.
pub(super) fn check(
    func: &Func,
    imports: &[Import],
    funcs: &[Func],
    workgroup: [u32; 3],
) -> Result<(), TranslateError> {
    if !imports
        .iter()
        .any(|item| barrier_name(&item.name).is_some())
    {
        return Ok(());
    }
    if func.export == symbols::CS_MAIN && workgroup == [1, 1, 1] {
        return Ok(());
    }
    let func_index = funcs
        .iter()
        .position(|item| std::ptr::eq(item, func))
        .unwrap_or(0) as u32;
    let mut uni = Uni {
        imports,
        funcs,
        active: Vec::new(),
        labels: Vec::new(),
        pending: Vec::new(),
        memo: call::Memo::new(),
        func: func_index,
        pos: 0,
        at: 0,
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
    /// Exits of a `br_table` that have not reached their target yet.
    pending: Vec<St>,
    memo: call::Memo,
    /// Defined-function index currently being walked.
    func: u32,
    /// Next statement index in `func`. Loop rounds restore the entry value.
    pos: u32,
    /// Preorder index of the statement being walked.
    at: u32,
    st: St,
}

#[derive(Clone)]
struct St {
    stack: Vec<Tag>,
    locals: Vec<Tag>,
    split: Option<Split>,
    leave: Leave,
    /// Definitions in this state are invocation-dependent even before `split`.
    ///
    /// A loop whose trip count depends on the invocation id sets this for the
    /// whole body, so a counter updated before the exit test is
    /// invocation-dependent.
    taint: bool,
}

impl St {
    fn fresh(locals: Vec<Tag>) -> Self {
        Self {
            stack: Vec::new(),
            locals,
            split: None,
            leave: Leave::Fall,
            taint: false,
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
}

struct Lab {
    kind: LabKind,
    /// Entered while every invocation was still together.
    ///
    /// The loop rewalk that rejects barriers inside a divergent loop clears
    /// this, so the rewalk does not mark the loop a second time.
    clear: bool,
    varies: bool,
    /// Locals carried by a branch back to this loop. Empty when no branch
    /// has continued it yet.
    carry: Vec<Tag>,
    /// Outermost label a divergent branch uses to leave this loop.
    exit_at: Option<usize>,
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
            carry: Vec::new(),
            exit_at: None,
        }
    }
}
