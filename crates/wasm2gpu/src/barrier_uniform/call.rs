//! Follow a direct call into a helper, carrying the caller's split.
//!
//! An imported barrier is rejected when that split is already open. Any
//! other import pushes unknown results, so it cannot prove a later branch.
//! A defined call is memoized by the function, the incoming split, whether
//! definitions are already tainted, and the argument tags. The body is
//! borrowed from [`Uni::funcs`].

use std::collections::HashMap;

use super::Split;
use super::St;
use super::Uni;
use super::barrier_name;
use super::tag::Tag;
use crate::error::BarrierSite;
use crate::error::TranslateError;
use crate::ir::Import;

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct MemoKey {
    index: u32,
    split: Option<BarrierSite>,
    /// Caller is tainting definitions. Distinct from `split`: a varying loop
    /// taints the body before the exit test opens a split.
    taint: bool,
    args: Vec<Tag>,
}

pub(super) type Memo = HashMap<MemoKey, Result<Vec<Tag>, TranslateError>>;

impl Uni<'_> {
    pub(super) fn on_call(&mut self, index: u32) -> Result<(), TranslateError> {
        if let Some(import) = self.imports.get(index as usize) {
            return self.on_import(import);
        }
        let base = self.imports.len() as u32;
        if index < base {
            return Err(TranslateError::Unsupported);
        }
        self.on_defined(index - base)
    }

    fn on_import(&mut self, import: &Import) -> Result<(), TranslateError> {
        if let Some(name) = barrier_name(&import.name) {
            return self.reject(name);
        }
        self.pop_n(import.sig.params);
        self.push_n(import.sig.results, Tag::Unknown);
        Ok(())
    }

    fn reject(&self, barrier: &'static str) -> Result<(), TranslateError> {
        match self.st.split {
            Some(split) => Err(TranslateError::DivergentBarrier {
                barrier,
                site: split.site,
                function: self.function_name(),
                position: self.at,
            }),
            None => Ok(()),
        }
    }

    fn function_name(&self) -> String {
        match self.funcs.get(self.func as usize) {
            Some(func) if !func.export.is_empty() => func.export.clone(),
            _ => format!("func {}", self.func),
        }
    }

    fn on_defined(&mut self, index: u32) -> Result<(), TranslateError> {
        if self.active.contains(&index) {
            return Err(TranslateError::Recursion);
        }
        let (nlocals, params, results) = {
            let func = self
                .funcs
                .get(index as usize)
                .ok_or(TranslateError::Unsupported)?;
            (func.locals.len(), func.sig.params, func.sig.results)
        };
        let args = self.take_args(params);
        let key = MemoKey {
            index,
            split: self.st.split.map(|item| item.site),
            taint: self.st.taint,
            args: args.clone(),
        };
        if let Some(hit) = self.memo.get(&key).cloned() {
            return self.replay(results, hit);
        }
        let outcome = self.walk_callee(index, nlocals, results, &args);
        self.memo.insert(key, outcome.clone());
        self.replay(results, outcome)
    }

    fn walk_callee(
        &mut self,
        index: u32,
        nlocals: usize,
        results: u32,
        args: &[Tag],
    ) -> Result<Vec<Tag>, TranslateError> {
        let funcs = self.funcs;
        let body = &funcs
            .get(index as usize)
            .ok_or(TranslateError::Unsupported)?
            .body;
        let caller_split = self.st.split;
        let caller_taint = self.st.taint;
        self.active.push(index);
        let saved_labels = std::mem::take(&mut self.labels);
        let saved_pending = std::mem::take(&mut self.pending);
        let saved_func = self.func;
        let saved_pos = self.pos;
        self.func = index;
        self.pos = 0;
        let saved = std::mem::replace(
            &mut self.st,
            callee_state(nlocals, args, caller_split, caller_taint),
        );
        let walked = self.walk(body);
        let stack = self.st.stack.clone();
        self.st = saved;
        self.labels = saved_labels;
        self.pending = saved_pending;
        self.func = saved_func;
        self.pos = saved_pos;
        self.active.pop();
        walked?;
        Ok(top_tags(results, &stack))
    }

    fn replay(
        &mut self,
        n: u32,
        hit: Result<Vec<Tag>, TranslateError>,
    ) -> Result<(), TranslateError> {
        let tags = hit?;
        let mut index = 0usize;
        while index < tags.len() {
            self.push_result(tags[index]);
            index += 1;
        }
        let mut missing = n as usize - tags.len().min(n as usize);
        while missing > 0 {
            self.push_result(Tag::Unknown);
            missing -= 1;
        }
        Ok(())
    }

    fn take_args(&mut self, n: u32) -> Vec<Tag> {
        let mut args = vec![Tag::Unknown; n as usize];
        let mut index = args.len();
        while index > 0 {
            index -= 1;
            args[index] = self.pop();
        }
        args
    }

    fn pop_n(&mut self, n: u32) {
        let mut left = n;
        while left > 0 {
            self.pop();
            left -= 1;
        }
    }

    fn push_n(&mut self, n: u32, tag: Tag) {
        let mut left = n;
        while left > 0 {
            self.push_result(tag);
            left -= 1;
        }
    }

    fn push_result(&mut self, tag: Tag) {
        let tag = if self.st.taint || self.st.split.is_some() {
            Tag::Inv
        } else {
            tag
        };
        self.st.stack.push(tag);
    }
}

fn top_tags(n: u32, stack: &[Tag]) -> Vec<Tag> {
    let n = n as usize;
    let start = stack.len().saturating_sub(n);
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < n {
        out.push(stack.get(start + index).copied().unwrap_or(Tag::Unknown));
        index += 1;
    }
    out
}

fn callee_state(nlocals: usize, args: &[Tag], split: Option<Split>, taint: bool) -> St {
    let mut locals = vec![Tag::Uniform; nlocals];
    let mut index = 0usize;
    while index < args.len() && index < locals.len() {
        locals[index] = args[index];
        index += 1;
    }
    let mut st = St::fresh(locals);
    st.split = split;
    st.taint = taint;
    st
}
