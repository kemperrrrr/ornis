//! Basic-block emission, returns and edge stores.

use super::cfg::{Env, Flow};
use super::emit::operand;
use super::lowerer::Lowerer;
use super::memory::MemoryPointer;
use super::texture::Elem;
use crate::error::TranslateError;
use crate::parse::{BasicBlock, Instr, Operand, Ty};

impl<'a, 'b> Lowerer<'a, 'b> {
    pub(crate) fn emit_block(
        &mut self,
        block: &BasicBlock,
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        crate::site::enter_block(&block.label);
        for instr in &block.instrs {
            match instr {
                Instr::Phi { dst, ty, .. } => {
                    let var = *self
                        .phi_vars
                        .get(dst)
                        .ok_or_else(TranslateError::unsupported)?;
                    let ptr = self.expr(out, naga::Expression::LocalVariable(var));
                    let value = self.expr(out, naga::Expression::Load { pointer: ptr });
                    if *ty == Ty::Ptr {
                        let base = *self
                            .ptr_bases
                            .get(dst)
                            .ok_or_else(TranslateError::unsupported)?;
                        self.pointers
                            .insert(dst.clone(), MemoryPointer { base, words: value });
                    } else {
                        self.bind(dst, value, out);
                    }
                }
                Instr::Ret { a } => {
                    self.emit_ret(Some(a), out)?;
                }
                Instr::RetVoid => {
                    self.emit_ret(None, out)?;
                }
                other => self.emit_instr(other, out)?,
            }
        }
        Ok(())
    }

    pub(crate) fn emit_ret(
        &mut self,
        value: Option<&Operand>,
        out: &mut naga::Block,
    ) -> Result<Flow, TranslateError> {
        if let Some(var) = self.stage_out.as_ref().map(|stage_out| stage_out.var) {
            let pointer = self.expr(out, naga::Expression::LocalVariable(var));
            let value = self.expr(out, naga::Expression::Load { pointer });
            out.push(
                naga::Statement::Return { value: Some(value) },
                naga::Span::default(),
            );
            self.saw_return = true;
            return Ok(Flow::Returned);
        }
        if self.returns_value {
            let value = match value {
                Some(op) => Some(self.operand(out, op)?),
                None => None,
            };
            out.push(naga::Statement::Return { value }, naga::Span::default());
            self.saw_return = true;
            return Ok(Flow::Returned);
        }
        if let Some(op) = value {
            let value = self.operand(out, op)?;
            let global = self.builders.storage_global(0, 0, Elem::F32, true)?;
            let base = self.expr(out, naga::Expression::GlobalVariable(global));
            let index = self.id_x.ok_or_else(TranslateError::unsupported)?;
            let pointer = self.expr(out, naga::Expression::Access { base, index });
            out.push(
                naga::Statement::Store { pointer, value },
                naga::Span::default(),
            );
        }
        out.push(
            naga::Statement::Return { value: None },
            naga::Span::default(),
        );
        self.saw_return = true;
        Ok(Flow::Returned)
    }

    /// Assign every `phi` of `to` the value arriving from `from`.
    pub(crate) fn store_edge(
        &mut self,
        from: &str,
        to: &str,
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        let env = self.env.clone();
        self.store_edge_into(from, to, &env, out)
    }

    pub(crate) fn store_edge_into(
        &mut self,
        from: &str,
        to: &str,
        env: &Env,
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        let Some(phis) = self.cfg.phis.get(to) else {
            return Ok(());
        };
        for phi in phis {
            let incoming = phi
                .incoming
                .iter()
                .find_map(|(value, pred)| (pred == from).then_some(value))
                .ok_or_else(TranslateError::unsupported)?;
            let value = if phi.ty == Ty::Ptr {
                let Operand::Local(name) = incoming else {
                    return Err(TranslateError::unsupported());
                };
                let pointer = *self
                    .pointers
                    .get(name)
                    .ok_or_else(TranslateError::unsupported)?;
                match self.ptr_bases.get(&phi.dst) {
                    Some(base) if *base != pointer.base => {
                        return Err(TranslateError::unsupported());
                    }
                    Some(_) => {}
                    None => {
                        self.ptr_bases.insert(phi.dst.clone(), pointer.base);
                    }
                }
                pointer.words
            } else {
                operand(&mut self.func, out, incoming, env)?
            };
            let var = *self
                .phi_vars
                .get(&phi.dst)
                .ok_or_else(TranslateError::unsupported)?;
            let ptr = self.expr(out, naga::Expression::LocalVariable(var));
            out.push(
                naga::Statement::Store {
                    pointer: ptr,
                    value,
                },
                naga::Span::default(),
            );
        }
        Ok(())
    }
}
