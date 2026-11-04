//! Follow a direct call into a helper, carrying the caller's split.
//!
//! An imported barrier is rejected when that split is already open. Any
//! other import pushes unknown results, so it cannot prove a later branch.

use super::Split;
use super::St;
use super::Uni;
use super::barrier_name;
use super::tag::Tag;
use crate::error::TranslateError;
use crate::ir::Import;

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
            }),
            None => Ok(()),
        }
    }

    fn on_defined(&mut self, index: u32) -> Result<(), TranslateError> {
        if self.active.contains(&index) {
            return Err(TranslateError::Recursion);
        }
        let (nlocals, params, results, body) = {
            let func = self
                .funcs
                .get(index as usize)
                .ok_or(TranslateError::Unsupported)?;
            (
                func.locals.len(),
                func.sig.params,
                func.sig.results,
                func.body.clone(),
            )
        };
        let args = self.take_args(params);
        let caller_split = self.st.split;
        self.active.push(index);
        let saved_labels = std::mem::take(&mut self.labels);
        let saved = std::mem::replace(&mut self.st, callee_state(nlocals, &args, caller_split));
        let walked = self.walk(&body);
        let stack = self.st.stack.clone();
        self.st = saved;
        self.labels = saved_labels;
        self.active.pop();
        walked?;
        self.push_results(results, &stack);
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

    fn push_results(&mut self, n: u32, stack: &[Tag]) {
        let n = n as usize;
        let start = stack.len().saturating_sub(n);
        let mut index = 0usize;
        while index < n {
            let tag = stack.get(start + index).copied().unwrap_or(Tag::Unknown);
            self.st.stack.push(tag);
            index += 1;
        }
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
            self.st.stack.push(tag);
            left -= 1;
        }
    }
}

fn callee_state(nlocals: usize, args: &[Tag], split: Option<Split>) -> St {
    let mut locals = vec![Tag::Uniform; nlocals];
    let mut index = 0usize;
    while index < args.len() && index < locals.len() {
        locals[index] = args[index];
        index += 1;
    }
    let mut st = St::fresh(locals);
    st.split = split;
    st
}
