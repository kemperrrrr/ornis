//! Control-flow graph: block structure, lowering context and local definitions.

use super::EXIT;
use super::dominance::{dominators, natural_loop, postdominators, reachable, successors};
use crate::error::TranslateError;
use crate::parse::{BasicBlock, Instr, LlvmFunc, Operand, Terminator, Ty};
use std::collections::{HashMap, HashSet};

/// One `phi` result and the `(value, predecessor)` pairs feeding it.
pub(crate) struct Phi {
    pub(crate) dst: String,
    pub(crate) ty: Ty,
    pub(crate) incoming: Vec<(Operand, String)>,
}

/// Read-only views of a parsed function used by the structurer.
pub(crate) struct Cfg<'a> {
    pub(crate) blocks: HashMap<&'a str, &'a BasicBlock>,
    pub(crate) order: Vec<&'a str>,
    pub(crate) phis: HashMap<&'a str, Vec<Phi>>,
    pub(crate) succs: HashMap<&'a str, Vec<&'a str>>,
    pub(crate) postdoms: HashMap<&'a str, HashSet<&'a str>>,
    /// Loop header -> blocks of its natural loop.
    pub(crate) loops: HashMap<&'a str, HashSet<&'a str>>,
}

impl<'a> Cfg<'a> {
    pub(crate) fn new(func: &'a LlvmFunc) -> Result<Self, TranslateError> {
        let mut blocks = HashMap::new();
        let mut order = Vec::new();
        for block in &func.blocks {
            if blocks.insert(block.label.as_str(), block).is_some() {
                return Err(TranslateError::Parse(format!(
                    "duplicate block {}",
                    block.label
                )));
            }
            order.push(block.label.as_str());
        }
        let entry = *order.first().ok_or_else(TranslateError::unsupported)?;
        let mut succs: HashMap<&'a str, Vec<&'a str>> = HashMap::new();
        let mut preds: HashMap<&'a str, Vec<&'a str>> = HashMap::new();
        let mut phis: HashMap<&'a str, Vec<Phi>> = HashMap::new();
        for block in &func.blocks {
            let out = successors(&block.term);
            for target in &out {
                if !blocks.contains_key(target) {
                    return Err(TranslateError::Parse(format!(
                        "branch to unknown block {target}"
                    )));
                }
                preds.entry(target).or_default().push(block.label.as_str());
            }
            succs.insert(block.label.as_str(), out);
            let mut block_phis = Vec::new();
            for instr in &block.instrs {
                if let Instr::Phi { dst, ty, incoming } = instr {
                    block_phis.push(Phi {
                        dst: dst.clone(),
                        ty: *ty,
                        incoming: incoming.clone(),
                    });
                }
            }
            if !block_phis.is_empty() {
                phis.insert(block.label.as_str(), block_phis);
            }
        }

        let reachable = reachable(&succs, entry);
        let doms = dominators(&reachable, &preds, entry);
        let postdoms = postdominators(&reachable, &succs);
        let mut loops: HashMap<&'a str, HashSet<&'a str>> = HashMap::new();
        for (from, targets) in &succs {
            if !reachable.contains(from) {
                continue;
            }
            for header in targets {
                if doms.get(from).is_some_and(|dom| dom.contains(*header)) {
                    let body = natural_loop(header, from, &preds);
                    loops.entry(*header).or_default().extend(body);
                }
            }
        }
        Ok(Self {
            blocks,
            order,
            phis,
            succs,
            postdoms,
            loops,
        })
    }

    pub(crate) fn entry(&self) -> &'a str {
        self.order[0]
    }

    /// Merge label of a conditional branch.
    ///
    /// When one arm returns directly and the other continues, the
    /// continuing arm is the merge: the branch becomes
    /// `if cond { return; }` and the rest follows the `if`. A plain
    /// post-dominator cannot see that (the returning arm never reaches
    /// the continuation, and a loop on the other arm need not reach the
    /// exit), so special-case the direct-return shape.
    pub(crate) fn branch_merge(
        &self,
        label: &str,
        true_label: &'a str,
        false_label: &'a str,
    ) -> Option<&'a str> {
        let merge = self.merge(label);
        if let Some(merge_label) = merge
            && self.returns_directly(merge_label)
        {
            if true_label == merge_label && false_label != merge_label {
                return Some(false_label);
            }
            if false_label == merge_label && true_label != merge_label {
                return Some(true_label);
            }
        }
        // A loop on one arm never post-dominates the branch (the back
        // edge need not reach the exit), so `merge` is `None` there too.
        if merge.is_none() || merge == Some(EXIT) {
            let true_returns = self.returns_directly(true_label);
            let false_returns = self.returns_directly(false_label);
            if true_returns && !false_returns {
                return self.blocks.get(false_label).map(|_| false_label);
            }
            if false_returns && !true_returns {
                return self.blocks.get(true_label).map(|_| true_label);
            }
        }
        merge
    }

    pub(crate) fn returns_directly(&self, label: &str) -> bool {
        self.blocks
            .get(label)
            .is_some_and(|block| matches!(block.term, Terminator::Ret(_)))
    }

    /// Closest common post-dominator of a conditional branch, or `None`
    /// when the branch has no common successor inside the function.
    pub(crate) fn merge(&self, label: &str) -> Option<&'a str> {
        let set = self.postdoms.get(label)?;
        let mut best: Option<&'a str> = None;
        for candidate in set {
            if *candidate == label || *candidate == EXIT {
                continue;
            }
            let size = self
                .postdoms
                .get(candidate)
                .map_or(usize::MAX, HashSet::len);
            let best_size = best
                .and_then(|b| self.postdoms.get(b))
                .map_or(usize::MAX, HashSet::len);
            if best.is_none() || size > best_size {
                best = Some(candidate);
            }
        }
        if best.is_none() && set.contains(EXIT) {
            return Some(EXIT);
        }
        best
    }
}

pub(crate) fn loop_exits<'a>(cfg: &Cfg<'a>, body: &HashSet<&'a str>) -> HashSet<&'a str> {
    body.iter()
        .flat_map(|block| cfg.succs.get(block).into_iter().flatten())
        .filter(|target| !body.contains(*target))
        .copied()
        .collect()
}

pub(crate) fn def_ty(cfg: &Cfg<'_>, body: &HashSet<&str>, name: &str) -> Option<Ty> {
    for label in body {
        let block = cfg.blocks.get(label)?;
        for instr in &block.instrs {
            if instr.defines(name) {
                return instr.value_ty();
            }
        }
    }
    None
}

pub(crate) fn push_local(op: &Operand, used: &mut Vec<String>) {
    if let Operand::Local(name) = op {
        used.push(name.clone());
    }
}

pub(crate) fn push_term_local(term: &Terminator, used: &mut Vec<String>) {
    match term {
        Terminator::BrCond { cond, .. } => push_local(cond, used),
        Terminator::Switch { selector, .. } => push_local(selector, used),
        Terminator::Ret(Some(value)) => push_local(value, used),
        Terminator::BrUncond(_) | Terminator::Ret(None) | Terminator::Unreachable => {}
    }
}

/// Where a region is emitted: top level or inside one natural loop.
#[derive(Clone, Copy)]
pub(crate) enum Ctx<'a> {
    Plain,
    Loop { header: &'a str },
}

/// How control leaves an emitted region.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flow {
    /// Reached the requested stop label or fell through normally.
    Next,
    /// Emitted `return`.
    Returned,
    /// Emitted `continue` for the innermost loop.
    Continue,
    /// Emitted `break` for the innermost loop.
    Break,
}

pub(crate) type Env = HashMap<String, naga::Handle<naga::Expression>>;
