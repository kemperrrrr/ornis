//! Branches, flags, and phi temps for lowered wasm control flow.

use naga::Expression;
use naga::Literal;

use crate::cx::Val;
use crate::cx::sp;
use crate::env::Emit;
use crate::env::Flag;
use crate::env::Region;
use crate::env::const_expr;
use crate::error::TranslateError;
use crate::ir::Kind;
use crate::ir::Stmt;

impl Emit<'_, '_> {
    pub(crate) fn region(&mut self, kind: Region, body: &[Stmt]) -> Result<(), TranslateError> {
        let flag = self.make_flag(kind, true)?;
        let block = self.sub(|em| {
            em.clear_flag(flag)?;
            em.labels.push(flag);
            em.run(body)?;
            em.close_region();
            Ok(())
        })?;
        self.push_loop(block);
        Ok(())
    }

    pub(crate) fn close_region(&mut self) {
        self.labels.pop();
        if self.halted {
            self.halted = false;
            return;
        }
        self.block.push(naga::Statement::Break, sp());
    }

    pub(crate) fn br_if(&mut self, depth: u32) -> Result<(), TranslateError> {
        let cond_val = self.pop()?;
        let cond = self.as_bool(cond_val)?;
        let accept = self.sub(|em| em.write_branch(depth))?;
        self.block.push(
            naga::Statement::If {
                condition: cond,
                accept,
                reject: naga::Block::new(),
            },
            sp(),
        );
        Ok(())
    }

    pub(crate) fn write_branch(&mut self, depth: u32) -> Result<(), TranslateError> {
        let n = self.labels.len();
        if depth as usize == n {
            return Err(TranslateError::FunctionBranch);
        }
        let index = n
            .checked_sub(depth as usize + 1)
            .ok_or(TranslateError::Unsupported)?;
        if depth == 0 {
            return self.exit_one(index);
        }
        self.flag_outers(index)?;
        self.block.push(naga::Statement::Break, sp());
        Ok(())
    }

    pub(crate) fn exit_one(&mut self, index: usize) -> Result<(), TranslateError> {
        let target = self
            .labels
            .get(index)
            .copied()
            .ok_or(TranslateError::UnsupportedType("ctx exit_one label".into()))?;
        if !target.naga_loop {
            return Err(TranslateError::UnsupportedType("ctx exit_one naga_loop".into()));
        }
        let stmt = if target.kind == Region::Loop {
            naga::Statement::Continue
        } else {
            naga::Statement::Break
        };
        self.block.push(stmt, sp());
        Ok(())
    }

    pub(crate) fn flag_outers(&mut self, target: usize) -> Result<(), TranslateError> {
        let mut index = target + 1;
        let n = self.labels.len();
        while index < n {
            let flag = self.labels[index];
            if flag.naga_loop {
                self.store_flag(flag.leave, true)?;
            }
            index += 1;
        }
        let flag = self.labels[target];
        if flag.kind == Region::Loop {
            self.store_flag(flag.again, true)
        } else {
            self.store_flag(flag.leave, true)
        }
    }

    /// Stop the arm when a multi-level `br` already left an outer naga loop.
    ///
    /// A statement `if` is not itself a naga loop, so [`Self::labels`]`.last()`
    /// is that `if` after an inner `block`. The `leave` / `again` flags live on
    /// the nearest real loop; checking only the `if` would run the rest of the
    /// arm (`storage_write` after `br 2`).
    pub(crate) fn flush_top(&mut self) -> Result<(), TranslateError> {
        let mut index = self.labels.len();
        while index > 0 {
            index -= 1;
            let flag = self.labels[index];
            if flag.naga_loop {
                return self.flush(flag);
            }
        }
        Ok(())
    }

    pub(crate) fn flush(&mut self, flag: Flag) -> Result<(), TranslateError> {
        let again = self.load_var(flag.again);
        let leave = self.load_var(flag.leave);
        let mut cont = naga::Block::new();
        cont.push(naga::Statement::Continue, sp());
        let mut stop = naga::Block::new();
        stop.push(naga::Statement::Break, sp());
        let mut rest = naga::Block::new();
        rest.push(branch_if(leave, stop, naga::Block::new()), sp());
        self.block.push(branch_if(again, cont, rest), sp());
        Ok(())
    }

    pub(crate) fn push_loop(&mut self, body: naga::Block) {
        self.block.push(
            naga::Statement::Loop {
                body,
                continuing: naga::Block::new(),
                break_if: None,
            },
            sp(),
        );
    }

    pub(crate) fn make_flag(
        &mut self,
        kind: Region,
        naga_loop: bool,
    ) -> Result<Flag, TranslateError> {
        let leave = self.bool_var()?;
        let again = self.bool_var()?;
        Ok(Flag {
            kind,
            naga_loop,
            leave,
            again,
        })
    }

    pub(crate) fn bool_var(&mut self) -> Result<naga::Handle<naga::LocalVariable>, TranslateError> {
        let init = const_expr(self.func, Expression::Literal(Literal::Bool(false)));
        Ok(self.var(self.cx.bool_ty, init))
    }

    pub(crate) fn clear_flag(&mut self, flag: Flag) -> Result<(), TranslateError> {
        self.store_flag(flag.leave, false)?;
        self.store_flag(flag.again, false)
    }

    pub(crate) fn store_flag(
        &mut self,
        var: naga::Handle<naga::LocalVariable>,
        value: bool,
    ) -> Result<(), TranslateError> {
        let bits = self.bool(value);
        self.store_var(var, bits);
        Ok(())
    }

    pub(crate) fn temps(
        &mut self,
        kinds: &[Kind],
    ) -> Result<Vec<naga::Handle<naga::LocalVariable>>, TranslateError> {
        let mut out = Vec::new();
        for kind in kinds {
            out.push(self.temp(*kind)?);
        }
        Ok(out)
    }

    pub(crate) fn temp(
        &mut self,
        kind: Kind,
    ) -> Result<naga::Handle<naga::LocalVariable>, TranslateError> {
        let (ty, init) = if kind == Kind::I32 {
            let init = const_expr(self.func, Expression::Literal(Literal::U32(0)));
            (self.cx.uint, init)
        } else {
            let init = const_expr(self.func, Expression::Literal(Literal::F32(0.0)));
            (self.cx.scalar, init)
        };
        Ok(self.var(ty, init))
    }

    pub(crate) fn store_temps(
        &mut self,
        temps: &[naga::Handle<naga::LocalVariable>],
        kinds: &[Kind],
    ) -> Result<(), TranslateError> {
        let mut index = temps.len();
        while index > 0 {
            index -= 1;
            let val = self.pop()?;
            let bits = self.bits(kinds[index], val)?;
            self.store_var(temps[index], bits);
        }
        Ok(())
    }

    pub(crate) fn load_temps(
        &mut self,
        temps: &[naga::Handle<naga::LocalVariable>],
        kinds: &[Kind],
    ) -> Result<(), TranslateError> {
        let mut index = 0usize;
        while index < temps.len() {
            let loaded = self.load_var(temps[index]);
            let val = if kinds[index] == Kind::I32 {
                Val::Index(loaded)
            } else {
                Val::Scalar(loaded)
            };
            self.stack.push(val);
            index += 1;
        }
        Ok(())
    }

    pub(crate) fn bits(
        &mut self,
        kind: Kind,
        val: Val,
    ) -> Result<naga::Handle<Expression>, TranslateError> {
        if kind == Kind::I32 {
            self.as_uint(val)
        } else {
            crate::env::float_of(val)
        }
    }

    pub(crate) fn pop_n(&mut self, n: usize) -> Result<Vec<Val>, TranslateError> {
        self.take(n)
    }
}

fn branch_if(
    condition: naga::Handle<Expression>,
    accept: naga::Block,
    reject: naga::Block,
) -> naga::Statement {
    naga::Statement::If {
        condition,
        accept,
        reject,
    }
}
