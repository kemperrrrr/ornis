//! The lowering walker: region and block emission.

use super::builders::Builders;
use super::cfg::{Cfg, Ctx, Env, Flow};
use super::emit::scalar;
use super::image_cache::ImageCache;
use super::memory::{MemoryBase, MemoryPointer, StageOut, local_label};
use super::stage::StageIo;
use crate::error::TranslateError;
use crate::parse::{Terminator, Ty};
use std::collections::{BTreeMap, HashMap, HashSet};

pub(crate) struct Lowerer<'a, 'b> {
    pub(crate) builders: &'b mut Builders,
    pub(crate) cfg: &'b Cfg<'a>,
    pub(crate) func: naga::Function,
    /// `@location` inputs and outputs the body touches.
    pub(crate) io: StageIo,
    pub(crate) id_x: Option<naga::Handle<naga::Expression>>,
    /// `global_invocation_id` vector, when this entry is compute.
    pub(crate) global_id: Option<naga::Handle<naga::Expression>>,
    /// `@builtin(local_invocation_index)`, when the kernel reads it.
    pub(crate) local_index: Option<naga::Handle<naga::Expression>>,
    /// `@builtin(vertex_index)`, when the vertex entry reads it.
    pub(crate) vertex_index: Option<naga::Handle<naga::Expression>>,
    /// `@builtin(instance_index)`, when the vertex entry reads it.
    pub(crate) instance_index: Option<naga::Handle<naga::Expression>>,
    /// Local the entry returns: the fragment color or the stage output struct.
    pub(crate) stage_out: Option<StageOut>,
    /// `@builtin(position)` of `fs_main`, when this entry is a fragment.
    pub(crate) position: Option<naga::Handle<naga::Expression>>,
    /// Declared `@location` arguments of the entry, by location.
    pub(crate) inputs: BTreeMap<u32, naga::Handle<naga::Expression>>,
    pub(crate) phi_vars: HashMap<String, naga::Handle<naga::LocalVariable>>,
    /// Loop values used after the loop, stored while the loop runs.
    pub(crate) spill: HashMap<String, naga::Handle<naga::LocalVariable>>,
    pub(crate) env: Env,
    pub(crate) saw_return: bool,
    /// Enclosing `if`/`loop`/`switch` regions whose condition or trip count is
    /// not proven uniform. A workgroup barrier inside any of them is rejected:
    /// a divergent barrier is UB on native GPUs (Tint rejects that WGSL).
    pub(crate) divergent_depth: u32,
    /// Values that are the same for every invocation in the workgroup.
    pub(crate) uniform: HashSet<String>,
    /// Exit block -> selector value, for loops whose exits go to more than one
    /// block (or leave more than one loop).
    ///
    /// A `break` in Naga leaves only the innermost `Loop`, so the selector says
    /// where control has to resume once the loop statements have unwound.
    pub(crate) exit_ids: HashMap<&'a str, u32>,
    /// Local the selector is stored in, created on first use.
    pub(crate) exit_id_var: Option<naga::Handle<naga::LocalVariable>>,
    /// Pointer SSA values: their base and their offset in 4-byte words.
    pub(crate) pointers: HashMap<String, MemoryPointer>,
    /// Base a pointer `phi` walks; every incoming pointer must share it.
    pub(crate) ptr_bases: HashMap<String, MemoryBase>,
    /// Module constants turned into private globals: handle and word count.
    pub(crate) constants: HashMap<String, (naga::Handle<naga::GlobalVariable>, u32)>,
    /// An arm returned and the region merged afterwards may therefore run on
    /// a subset of invocations. Sticky for the rest of the entry.
    pub(crate) diverged: bool,
    /// Whether `ret` returns a value (a helper) instead of storing it (the
    /// `cs_main(i32) -> f32` toy shape).
    pub(crate) returns_value: bool,
    /// Image queries already emitted, so a second lane of the same texel
    /// reads a component instead of sampling again.
    pub(crate) images: ImageCache,
}

impl<'a, 'b> Lowerer<'a, 'b> {
    pub(crate) fn emit_top(&mut self) -> Result<(), TranslateError> {
        let mut body = std::mem::take(&mut self.func.body);
        let flow = self.emit_region(self.cfg.entry(), None, &mut body, Ctx::Plain)?;
        self.func.body = body;
        match flow {
            Flow::Next | Flow::Returned => Ok(()),
            Flow::Continue | Flow::Break => Err(TranslateError::unsupported()),
        }
    }

    pub(crate) fn alloc_phis(&mut self) -> Result<(), TranslateError> {
        let mut labels: Vec<&str> = self.cfg.phis.keys().copied().collect();
        labels.sort_unstable();
        for label in labels {
            for phi in &self.cfg.phis[label] {
                let ty = self.ty(phi.ty)?;
                let var = self.func.local_variables.append(
                    naga::LocalVariable {
                        name: Some(local_label(&phi.dst)),
                        ty,
                        init: None,
                    },
                    naga::Span::default(),
                );
                self.phi_vars.insert(phi.dst.clone(), var);
            }
        }
        Ok(())
    }

    pub(crate) fn ty(&mut self, ty: Ty) -> Result<naga::Handle<naga::Type>, TranslateError> {
        let kind = match ty {
            Ty::Float => naga::ScalarKind::Float,
            Ty::I32 => naga::ScalarKind::Uint,
            // A pointer phi carries the offset in words; the base is the same
            // for every incoming edge.
            Ty::Ptr => naga::ScalarKind::Uint,
            Ty::I1 => naga::ScalarKind::Bool,
            _ => return Err(TranslateError::unsupported()),
        };
        Ok(scalar(&mut self.builders.module, kind))
    }

    /// Emit `start` until `stop` is reached or control leaves the region.
    pub(crate) fn emit_region(
        &mut self,
        start: &'a str,
        stop: Option<&'a str>,
        out: &mut naga::Block,
        ctx: Ctx<'a>,
    ) -> Result<Flow, TranslateError> {
        if stop == Some(start) {
            return Ok(Flow::Next);
        }
        match ctx {
            Ctx::Plain => {
                if self.cfg.loops.contains_key(start) {
                    return self.emit_loop(start, stop, out, Ctx::Plain);
                }
            }
            Ctx::Loop { header, .. } => {
                if start != header && self.cfg.loops.contains_key(start) {
                    // `break`/`continue` emitted for this header stay inside
                    // its `Loop`; the caller context receives the exit.
                    return self.emit_loop(start, stop, out, ctx);
                }
            }
        }
        let block = *self
            .cfg
            .blocks
            .get(start)
            .ok_or_else(TranslateError::unsupported)?;
        self.emit_block(block, out)?;
        match &block.term {
            Terminator::Ret(value) => self.emit_ret(value.as_ref(), out),
            // The panic path: LLVM calls it undefined behaviour, so the shader
            // ends the invocation here instead of pretending it cannot happen.
            Terminator::Unreachable => self.emit_ret(None, out),
            Terminator::BrUncond(target) => {
                let target = target.as_str();
                self.store_edge(start, target, out)?;
                if let Ctx::Loop { header } = ctx {
                    if target == header {
                        out.push(naga::Statement::Continue, naga::Span::default());
                        return Ok(Flow::Continue);
                    }
                    let leaves_loop = self
                        .cfg
                        .loops
                        .get(header)
                        .is_none_or(|body| !body.contains(target));
                    if leaves_loop {
                        self.store_exit_id(out, target)?;
                        out.push(naga::Statement::Break, naga::Span::default());
                        return Ok(Flow::Break);
                    }
                }
                if stop == Some(target) {
                    return Ok(Flow::Next);
                }
                self.emit_region(target, stop, out, ctx)
            }
            Terminator::BrCond {
                cond,
                true_label,
                false_label,
            } => {
                if let Ctx::Loop { header } = ctx {
                    let (true_in, false_in) =
                        self.cfg.loops.get(header).map_or((false, false), |body| {
                            (
                                body.contains(true_label.as_str()),
                                body.contains(false_label.as_str()),
                            )
                        });
                    if true_in != false_in {
                        return self.emit_loop_exit_branch(
                            header,
                            start,
                            cond,
                            true_label,
                            false_label,
                            true_in,
                            out,
                        );
                    }
                    if !true_in && !false_in {
                        return self.emit_loop_double_exit(
                            start,
                            cond,
                            true_label,
                            false_label,
                            out,
                        );
                    }
                }
                self.emit_if(start, cond, true_label, false_label, stop, out, ctx)
            }
            Terminator::Switch {
                selector,
                default,
                cases,
            } => self.emit_switch(start, selector, default, cases, stop, out, ctx),
        }
    }
}

impl<'a, 'b> Lowerer<'a, 'b> {
    pub(crate) fn bind(
        &mut self,
        dst: &str,
        value: naga::Handle<naga::Expression>,
        out: &mut naga::Block,
    ) {
        if let Some(&var) = self.spill.get(dst) {
            let pointer = self.expr(out, naga::Expression::LocalVariable(var));
            out.push(
                naga::Statement::Store { pointer, value },
                naga::Span::default(),
            );
        }
        self.env.insert(dst.to_string(), value);
    }
}
