//! Lowering of LLVM IR into a Naga module.

use self::builders::Builders;
use self::cfg::Cfg;
use self::memory::Helper;
use self::stage::{Stage, is_barrier};
use self::stage_io::entry_stage;
use crate::error::TranslateError;
use crate::parse::{Instr, LlvmFunc, Terminator};

pub(crate) const EXIT: &str = "$exit";

/// Bytes in one 32-bit storage word.
pub(crate) const WORD_BYTES: u32 = core::mem::size_of::<f32>() as u32;
/// Naga's scalar width field.
pub(crate) const SCALAR_WIDTH: u8 = core::mem::size_of::<f32>() as u8;
/// Uniform (std140) lane stride for one scalar.
pub(crate) const UNIFORM_LANE_BYTES: u32 = core::mem::size_of::<[f32; 4]>() as u32;
/// Largest uniform binding accepted by the translator.
pub(crate) const MAX_UNIFORM_BYTES: u32 = u16::MAX as u32 + 1;
/// Shortest bare digit run that can only be an LLVM SSA-index counter.
///
/// An `i`-prefixed run is a counter at any length (`i11`), but a *bare* short
/// run is an aggregate member index (`lo.sroa.9.0`) and has to stay.
pub(crate) const UNSTABLE_COUNTER_DIGITS: usize = 4;
/// Operands of `uniform_nested_array_vec4_lane`.
pub(crate) const NESTED_ARRAY_VEC4_ARGS: usize = 8;
/// Operands of `uniform_nested_vec4_lane`.
pub(crate) const NESTED_VEC4_ARGS: usize = 6;
/// Lane counts of the GPU vector types.
pub(crate) const VEC2_LANES: u32 = ornis_gpu_abi::VEC2_SIZE / WORD_BYTES;
pub(crate) const VEC3_LANES: u32 = ornis_gpu_abi::VEC3_SIZE / WORD_BYTES;
pub(crate) const VEC4_LANES: u32 = ornis_gpu_abi::VEC4_SIZE / WORD_BYTES;
/// Byte offsets of the second, third, and fourth scalar words.
pub(crate) const WORD_2: u32 = WORD_BYTES + WORD_BYTES;
pub(crate) const WORD_3: u32 = WORD_2 + WORD_BYTES;

/// One Naga module under construction: entries share its globals.
///
/// Entry shapes, per pushed function:
/// - `cs_main(i32) -> float`: result is stored to `storage_0_0[id]`.
/// - `cs_main(i32) -> void` / `cs_main() -> void`: the body stores through ABI
///   calls.
/// - `vs_main() -> void` / `fs_main() -> void`: builtins and stage interface
///   values come from ABI calls, and the lowerer returns the output it stored.
///
/// Resources are cached per `(group, binding)` in [`Builders`], so a vertex and
/// a fragment entry of the same pass share one set of globals — which is what a
/// pipeline needs: both stages read the same bind group. Each call to
/// [`Self::push`] adds one entry point.
pub(crate) struct Lowering {
    builders: Builders,
}

impl Lowering {
    /// Start a module with the given uniform-struct tables.
    pub(crate) fn new(structs: &[&[u8]], sampling: crate::TextureSampling) -> Self {
        Self {
            builders: Builders::new(structs, sampling),
        }
    }

    /// Remember the functions the parser skipped, so a call that reaches one
    /// reports what is wrong with it instead of a generic rejection.
    pub(crate) fn reserve_skipped(&mut self, skipped: Vec<(String, String)>) {
        for (name, reason) in skipped {
            self.builders.skipped_helpers.insert(name, reason);
        }
    }

    /// Remember every function the module defines.
    ///
    /// A call to a defined name is a helper call; everything else is an import
    /// or an intrinsic. This is what replaced guessing from the spelling of the
    /// name, so a kernel helper is never mistaken for an ABI import.
    pub(crate) fn reserve_definitions(&mut self, funcs: &[LlvmFunc]) {
        for func in funcs {
            if Stage::of(&func.name).is_none() {
                self.builders.defined.insert(func.name.clone());
            }
        }
    }

    /// Remember the helpers that reach a workgroup barrier, transitively.
    ///
    /// The divergence check in the lowering only sees the function it is
    /// lowering, so a helper that calls `workgroup_barrier` and is itself
    /// called under an `if` would emit a barrier half the workgroup reaches.
    /// Naga does not reject that and native drivers make it undefined
    /// behaviour, so the call site has to know.
    pub(crate) fn reserve_barriers(&mut self, funcs: &[LlvmFunc]) {
        let mut changed = true;
        while changed {
            changed = false;
            for func in funcs {
                if self.builders.barriered.contains(&func.name) {
                    continue;
                }
                let reaches = func.blocks.iter().any(|block| {
                    block.instrs.iter().any(|instr| match instr {
                        Instr::Call { callee, .. } | Instr::CallVoid { callee, .. } => {
                            is_barrier(callee) || self.builders.barriered.contains(callee)
                        }
                        _ => false,
                    })
                });
                if reaches {
                    changed |= self.builders.barriered.insert(func.name.clone());
                }
            }
        }
    }

    /// Remember the functions that never return, so their call sites can be
    /// dropped instead of translated.
    pub(crate) fn reserve_diverging(&mut self, funcs: &[LlvmFunc]) {
        for func in funcs {
            if !func
                .blocks
                .iter()
                .any(|block| matches!(block.term, Terminator::Ret(_)))
            {
                self.builders.diverging.insert(func.name.clone());
            }
        }
    }

    /// Remember the declared functions LLVM marks `noreturn`: a call to one
    /// never returns, so its call site is dropped.
    pub(crate) fn reserve_noreturn(&mut self, names: Vec<String>) {
        self.builders.diverging.extend(names);
    }

    /// Lower every helper the entries can call.
    ///
    /// Helpers may call each other, so they are lowered in dependency order:
    /// each round lowers the functions whose callees are already in the module.
    /// A helper whose body is outside the subset (the panic shim takes a `ptr`,
    /// for instance) is skipped rather than failing the module: only a call that
    /// actually reaches it is an error. A cycle is never resolved, so recursion
    /// is rejected.
    ///
    /// # Errors
    ///
    /// [`TranslateError::Unsupported`] when the helpers call each other in a
    /// cycle, which a shader cannot express.
    pub(crate) fn add_helpers(&mut self, funcs: &[LlvmFunc]) -> Result<(), TranslateError> {
        let mut pending: Vec<&LlvmFunc> = funcs
            .iter()
            .filter(|func| Stage::of(&func.name).is_none())
            .collect();
        let mut skipped: Vec<&str> = Vec::new();
        while !pending.is_empty() {
            let before = pending.len();
            let mut deferred = Vec::new();
            for func in pending {
                if self.callees_ready(func, &skipped) {
                    self.lower_or_skip(func, &mut skipped);
                } else {
                    deferred.push(func);
                }
            }
            if deferred.len() == before {
                return Err(TranslateError::unsupported());
            }
            pending = deferred;
        }
        Ok(())
    }

    /// Lower `func`, or remember it as unusable when its body leaves the subset.
    ///
    /// The reason travels with the name: a call to a helper that failed to
    /// lower is an error at *that* call, quoting why the helper could not be
    /// translated. Dropping the error instead let the call fall through to the
    /// import dispatch, which silently replaced the helper with whichever
    /// import its name happened to contain.
    pub(crate) fn lower_or_skip<'f>(&mut self, func: &'f LlvmFunc, skipped: &mut Vec<&'f str>) {
        if let Err(error) = self.lower_helper(func) {
            self.builders
                .skipped_helpers
                .insert(func.name.clone(), error.to_string());
            skipped.push(func.name.as_str());
        }
    }

    /// Whether every helper `func` calls is already lowered or skipped.
    ///
    /// A callee the module does not define is an import or an intrinsic, and
    /// needs no ordering. Whether a name *is* a helper is not a question about
    /// its spelling: `is_helper_name` used to answer it from substrings, which
    /// let a helper of the kernel look like an import and skip its turn.
    pub(crate) fn callees_ready(&self, func: &LlvmFunc, skipped: &[&str]) -> bool {
        func.blocks.iter().all(|block| {
            block.instrs.iter().all(|instr| match instr {
                Instr::Call { callee, .. } | Instr::CallVoid { callee, .. } => {
                    !self.builders.defined.contains(callee)
                        || self.builders.helpers.contains_key(callee)
                        || skipped.contains(&callee.as_str())
                }
                _ => true,
            })
        })
    }

    /// Lower one helper into a Naga function.
    pub(crate) fn lower_helper(&mut self, func: &LlvmFunc) -> Result<(), TranslateError> {
        let cfg = Cfg::new(func)?;
        let function = self.builders.build_helper(func, &cfg)?;
        let handle = self
            .builders
            .module
            .functions
            .append(function, naga::Span::default());
        self.builders.helpers.insert(
            func.name.clone(),
            Helper {
                function: handle,
                ret: func.ret_ty,
            },
        );
        Ok(())
    }

    /// Lower one entry function into the module.
    ///
    /// `workgroup` is the `@workgroup_size` of a compute entry and ignored by a
    /// raster one.
    ///
    /// # Errors
    ///
    /// [`TranslateError::Unsupported`] for an entry shape outside the subset
    /// (an unknown name, a raster entry with parameters, a compute entry that
    /// is neither `(i32)` nor `()`), and whatever the body lowers to.
    pub(crate) fn push(
        &mut self,
        func: &LlvmFunc,
        workgroup: [u32; 3],
    ) -> Result<(), TranslateError> {
        // The entry checks below run before the body walk, so the site still
        // holds whatever the parser last recorded — an instruction of an
        // unrelated function. Naming the function here is the whole point of
        // the message: an entry name without a stage prefix is the mistake
        // that reaching for `ret void` sends the reader away from.
        crate::site::enter_function(&func.name);
        let stage = entry_stage(func)?;
        self.lower_entry(func, stage, workgroup)
    }

    /// Structure the body and register the entry point.
    pub(crate) fn lower_entry(
        &mut self,
        func: &LlvmFunc,
        stage: Stage,
        workgroup: [u32; 3],
    ) -> Result<(), TranslateError> {
        let cfg = Cfg::new(func)?;
        let function = self.builders.build(func, &cfg, stage)?;
        self.builders
            .finish(&func.name, function, func.ret_ty, workgroup, stage)
    }

    /// The finished module, with every entry point pushed so far.
    pub(crate) fn finish(self) -> naga::Module {
        self.builders.module
    }
}

mod blocks;
mod build;
mod builder_types;
mod builders;
mod calls;
mod cfg;
mod control;
mod dominance;
mod emit;
mod expr;
mod flow;
mod helper_calls;
mod helpers;
mod image;
mod instructions;
mod io;
mod loop_spill;
mod loops;
mod lowerer;
mod memory;
mod memory_ops;
mod stage;
mod stage_io;
mod stage_out;
mod storage;
mod texture;
mod texture_ops;
mod uniform_fields;
mod uniforms;
mod word_ops;
mod workgroups;

#[cfg(test)]
mod tests;
