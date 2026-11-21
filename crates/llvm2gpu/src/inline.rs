//! Inline helpers whose signature carries a pointer.
//!
//! WGSL has no pointer parameters (`unrestricted_pointer_parameters` is not
//! enabled), so a helper like `fn add(a: &f32, b: &f32) -> f32` cannot become a
//! Naga function. The translator inlines those calls at the IR level instead:
//! the pointer never crosses a function boundary, and the body is lowered in
//! the caller, where the pointer's base — a local array, a module constant, or
//! a storage binding — is known.
//!
//! The inliner works on the parsed AST, before lowering:
//!
//! 1. every local of the callee is namespaced (`callee#n.name`), so the two
//!    functions' SSA names cannot collide;
//! 2. the callee's parameters are substituted with the call's arguments — a
//!    parameter is never reassigned, so substitution is sound;
//! 3. the caller block is split at the call, the callee's blocks are spliced
//!    in, and every `ret` stores its value and branches to a continuation.
//!
//! The value of a call is threaded through a fresh `alloca` in the caller,
//! which the continuation loads. That keeps a helper with several early returns
//! inlineable; recursion is rejected, and so is a call whose result type is not
//! a scalar.

use crate::error::TranslateError;
use crate::parse::{BasicBlock, Instr, LlvmFunc, Operand, Terminator, Ty};
use std::collections::HashMap;

/// Bytes of one scalar word, the size of the result slot an inlined call uses.
const WORD_BYTES: u32 = 4;

/// Upper bound on the calls one function may have inlined.
///
/// A recursive helper would otherwise grow the body on every round; the cap
/// turns that into a clear error instead of a runaway translation.
const MAX_INLINED_CALLS: usize = 256;

/// Inline every call to a helper that takes or returns a pointer.
///
/// Returns the functions with the pointer helpers folded into their callers and
/// the helpers themselves dropped: lowering them as Naga functions is
/// impossible, and nothing refers to them any more.
///
/// # Errors
///
/// [`TranslateError::UnsupportedType`] when the call graph recurses (a helper
/// cannot be inlined into itself), and [`TranslateError::Unsupported`] when a
/// call's arguments or result type do not match the callee's signature.
// qual:allow(iosp) reason: "an inliner is a rewrite: the body table, the splice and the result slot are one decision, and splitting them would only move the same loop"
pub(super) fn inline_pointer_calls(funcs: Vec<LlvmFunc>) -> Result<Vec<LlvmFunc>, TranslateError> {
    let bodies: HashMap<String, LlvmFunc> = funcs
        .iter()
        .filter(|func| has_pointer_signature(func) && func.returns_a_value_or_nothing())
        .map(|func| (func.name.clone(), func.clone()))
        .collect();
    if bodies.is_empty() {
        return Ok(funcs);
    }
    let mut out = Vec::with_capacity(funcs.len());
    for mut func in funcs {
        if !bodies.contains_key(&func.name) {
            inline_into(&mut func, &bodies)?;
        }
        out.push(func);
    }
    out.retain(|func| !bodies.contains_key(&func.name));
    Ok(out)
}

impl LlvmFunc {
    /// Whether any block returns a value.
    fn returns_a_value(&self) -> bool {
        self.blocks
            .iter()
            .any(|block| matches!(block.term, Terminator::Ret(Some(_))))
    }

    /// Whether the function can return at all.
    ///
    /// `false` for a panic shim: every path ends in `unreachable`, so a call to
    /// it never comes back and there is nothing to inline.
    fn returns_a_value_or_nothing(&self) -> bool {
        self.blocks
            .iter()
            .any(|block| matches!(block.term, Terminator::Ret(_)))
    }
}

/// Whether `func` takes or returns a pointer, which WGSL cannot express.
fn has_pointer_signature(func: &LlvmFunc) -> bool {
    func.ret_ty == Ty::Ptr || func.params.iter().any(|(ty, _)| *ty == Ty::Ptr)
}

/// One call site to inline.
struct CallSite {
    /// Block the call sits in.
    block: usize,
    /// Instruction index inside that block.
    instr: usize,
    /// Name of the callee.
    callee: String,
}

/// Inline calls in `func` until no pointer helper is called any more.
// qual:allow(iosp) reason: "inlining is a loop that inspects the body and rewrites it; the two halves cannot be separated without a second AST"
fn inline_into(
    func: &mut LlvmFunc,
    bodies: &HashMap<String, LlvmFunc>,
) -> Result<(), TranslateError> {
    let mut inlined = 0usize;
    while let Some(site) = find_call(func, bodies) {
        inlined += 1;
        if inlined > MAX_INLINED_CALLS {
            return Err(TranslateError::UnsupportedType(format!(
                "helper calls in `{}` never resolve: a shader function cannot recurse",
                func.name
            )));
        }
        let callee = bodies
            .get(&site.callee)
            .ok_or(TranslateError::Unsupported)?
            .clone();
        inline_call(func, &site, &callee)?;
    }
    Ok(())
}

/// The first call in `func` that names an inlinable helper.
fn find_call(func: &LlvmFunc, bodies: &HashMap<String, LlvmFunc>) -> Option<CallSite> {
    for (block, basic) in func.blocks.iter().enumerate() {
        for (instr, instruction) in basic.instrs.iter().enumerate() {
            let callee = match instruction {
                Instr::Call { callee, .. } | Instr::CallVoid { callee, .. } => callee,
                _ => continue,
            };
            if bodies.contains_key(callee) {
                return Some(CallSite {
                    block,
                    instr,
                    callee: callee.clone(),
                });
            }
        }
    }
    None
}

/// Inline `callee` at `site` of `caller`.
// qual:allow(iosp) reason: "the whole inlining is one rewrite: names, arguments, blocks and the result slot are decided together"
fn inline_call(
    caller: &mut LlvmFunc,
    site: &CallSite,
    callee: &LlvmFunc,
) -> Result<(), TranslateError> {
    let (dst, args) = match &caller.blocks[site.block].instrs[site.instr] {
        Instr::Call { dst, args, .. } => (Some(dst.clone()), args.clone()),
        Instr::CallVoid { args, .. } => (None, args.clone()),
        _ => return Err(TranslateError::Unsupported),
    };
    if args.len() != callee.params.len() {
        return Err(TranslateError::UnsupportedType(format!(
            "call to `{}` passes {} arguments, the helper takes {}",
            callee.name,
            args.len(),
            callee.params.len()
        )));
    }
    let subst: HashMap<String, Operand> = callee
        .params
        .iter()
        .map(|(_, name)| name.clone())
        .zip(args)
        .collect();
    let prefix = format!("{}#{}.", callee.name, site.instr);
    let entry = prefixed_label(&prefix, &callee.blocks[0].label);
    let continuation = format!("{prefix}cont");
    let slot = match &dst {
        Some(_) if matches!(callee.ret_ty, Ty::Float | Ty::I32) => Some(format!("{prefix}ret")),
        Some(_) => {
            return Err(TranslateError::UnsupportedType(format!(
                "inlined call to `{}` returns a value this translator cannot carry",
                callee.name
            )));
        }
        None => None,
    };
    if slot.is_some() && !callee.returns_a_value() {
        return Err(TranslateError::UnsupportedType(format!(
            "call to `{}` uses a result the helper never returns",
            callee.name
        )));
    }
    let blocks = renamed_blocks(callee, &prefix, &subst, &continuation, &slot)?;
    splice_call(
        caller,
        site,
        Splice {
            entry,
            blocks,
            start: result_load(&slot, dst, callee.ret_ty),
            continuation,
            slot,
        },
    )
}

/// The callee's blocks, renamed and ending in the continuation.
// qual:allow(iosp) reason: "renaming walks the callee's blocks and rewrites each instruction; the walk is the rename"
fn renamed_blocks(
    callee: &LlvmFunc,
    prefix: &str,
    subst: &HashMap<String, Operand>,
    continuation: &str,
    slot: &Option<String>,
) -> Result<Vec<BasicBlock>, TranslateError> {
    let mut spliced = Vec::with_capacity(callee.blocks.len());
    for block in &callee.blocks {
        let mut block = block.clone();
        for instruction in &mut block.instrs {
            instruction.inline_rename(prefix, subst)?;
        }
        block.term = rewrite_term(
            &mut block.instrs,
            block.term,
            &TermRewrites {
                prefix,
                subst,
                continuation,
                slot,
                ret_ty: callee.ret_ty,
            },
        )?;
        block.label = prefixed_label(prefix, &block.label);
        spliced.push(block);
    }
    Ok(spliced)
}

/// The continuation's load of the inline's result, when it has one.
fn result_load(slot: &Option<String>, dst: Option<String>, ret_ty: Ty) -> Vec<Instr> {
    match (slot, dst) {
        (Some(name), Some(dst)) => vec![Instr::Load {
            dst,
            ptr: name.clone(),
            ty: ret_ty,
        }],
        _ => Vec::new(),
    }
}

/// What `splice_call` writes into the caller.
struct Splice {
    /// Block the caller branches to (the callee's entry).
    entry: String,
    /// The callee's renamed blocks.
    blocks: Vec<BasicBlock>,
    /// Instructions the continuation starts with.
    start: Vec<Instr>,
    /// Label of the continuation block.
    continuation: String,
    /// Result slot to declare, when the call has a value.
    slot: Option<String>,
}

/// Replace the call with the callee's blocks and a continuation.
fn splice_call(
    caller: &mut LlvmFunc,
    site: &CallSite,
    splice: Splice,
) -> Result<(), TranslateError> {
    let Splice {
        entry,
        blocks,
        mut start,
        continuation,
        slot,
    } = splice;
    // The caller block keeps everything before the call and branches to the
    // callee; everything after the call moves to the continuation.
    let old_term = caller.blocks[site.block].term.clone();
    // `split_off` leaves the call itself at the front of the tail: it is the
    // instruction this inline replaces.
    let mut tail = caller.blocks[site.block].instrs.split_off(site.instr);
    if !matches!(
        tail.first(),
        Some(Instr::Call { .. } | Instr::CallVoid { .. })
    ) {
        return Err(TranslateError::Unsupported);
    }
    tail.remove(0);
    start.extend(tail);
    // The result slot is inserted after the split: inserting it first would
    // shift the call's index when the call sits in the entry block.
    if let Some(name) = &slot {
        caller.blocks[0].instrs.insert(
            0,
            Instr::Alloca {
                dst: name.clone(),
                bytes: WORD_BYTES,
            },
        );
    }
    caller.blocks[site.block].term = Terminator::BrUncond(entry);
    caller.blocks.extend(blocks);
    caller.blocks.push(BasicBlock {
        label: continuation,
        instrs: start,
        term: old_term,
    });
    Ok(())
}

/// What a callee terminator's rewrite needs.
struct TermRewrites<'a> {
    /// Namespace for every local and label.
    prefix: &'a str,
    /// Parameter substitution.
    subst: &'a HashMap<String, Operand>,
    /// Block every `ret` branches to.
    continuation: &'a str,
    /// Result slot an inlined call stores into.
    slot: &'a Option<String>,
    /// Type of that slot.
    ret_ty: Ty,
}

/// Rewrite a callee terminator: labels are namespaced, and a `ret` stores its
/// value into the result slot and branches to the continuation.
fn rewrite_term(
    instrs: &mut Vec<Instr>,
    term: Terminator,
    rewrites: &TermRewrites<'_>,
) -> Result<Terminator, TranslateError> {
    let TermRewrites {
        prefix,
        subst,
        continuation,
        slot,
        ret_ty,
    } = *rewrites;
    Ok(match term {
        Terminator::Ret(Some(value)) => {
            let name = slot.clone().ok_or(TranslateError::Unsupported)?;
            let value = rename_operand(value, prefix, subst)?;
            instrs.push(Instr::Store {
                ptr: name,
                value,
                ty: ret_ty,
            });
            Terminator::BrUncond(continuation.to_string())
        }
        Terminator::Ret(None) | Terminator::Unreachable => {
            Terminator::BrUncond(continuation.to_string())
        }
        Terminator::BrUncond(label) => Terminator::BrUncond(prefixed_label(prefix, &label)),
        Terminator::BrCond {
            cond,
            true_label,
            false_label,
        } => Terminator::BrCond {
            cond: rename_operand(cond, prefix, subst)?,
            true_label: prefixed_label(prefix, &true_label),
            false_label: prefixed_label(prefix, &false_label),
        },
        Terminator::Switch {
            selector,
            default,
            cases,
        } => Terminator::Switch {
            selector: rename_operand(selector, prefix, subst)?,
            default: prefixed_label(prefix, &default),
            cases: cases
                .into_iter()
                .map(|(value, label)| (value, prefixed_label(prefix, &label)))
                .collect(),
        },
    })
}

/// `label` inside the inlined namespace.
fn prefixed_label(prefix: &str, label: &str) -> String {
    format!("{prefix}{label}")
}

/// Rename one operand: substitute a parameter, namespacing everything else.
fn rename_operand(
    operand: Operand,
    prefix: &str,
    subst: &HashMap<String, Operand>,
) -> Result<Operand, TranslateError> {
    let Operand::Local(name) = operand else {
        return Ok(operand);
    };
    Ok(match subst.get(&name) {
        Some(Operand::Local(replacement)) => Operand::Local(replacement.clone()),
        Some(_) => return Err(TranslateError::Unsupported),
        None => Operand::Local(format!("{prefix}{name}")),
    })
}
