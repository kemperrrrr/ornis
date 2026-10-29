//! Structured wasm statements to naga `if`, `loop`, and `select`.
//!
//! A pure `if` (no calls, stores, or branches) becomes `select`. Anything with
//! a side effect becomes a statement. `loop` and `block` are naga loops so
//! `br` can `break` or `continue`. A branch that skips a loop sets a `leave`
//! or `again` flag and breaks the inner loop; the outer loop checks the flag.

use naga::Expression;

use crate::cx::Val;
use crate::cx::sp;
use crate::env::Emit;
use crate::env::Region;
use crate::error::TranslateError;
use crate::ir::BlockTy;
use crate::ir::Func;
use crate::ir::Kind;
use crate::ir::Module;
use crate::ir::Stmt;
use crate::ops;

pub(super) fn exec(
    cx: &mut crate::cx::Cx<'_>,
    module: &Module,
    func: &Func,
    body: &mut naga::Function,
) -> Result<Vec<Val>, TranslateError> {
    let shadowed = needs_shadow(&func.body);
    let locals = bind_locals(body, func, cx, shadowed)?;
    let block = std::mem::take(&mut body.body);
    let mut em = Emit {
        cx,
        func: body,
        block,
        stack: Vec::new(),
        imports: &module.imports,
        structs: &module.structs,
        locals,
        labels: Vec::new(),
    };
    if shadowed {
        copy_params(&mut em, func.sig.params)?;
    }
    em.run(&func.body)?;
    let Emit {
        block, stack, func, ..
    } = em;
    func.body = block;
    Ok(stack)
}

impl Emit<'_, '_> {
    pub(crate) fn run(&mut self, stmts: &[Stmt]) -> Result<(), TranslateError> {
        for stmt in stmts {
            let nested = is_nested(stmt);
            self.one(stmt)?;
            if nested {
                self.flush_top()?;
            }
        }
        Ok(())
    }

    pub(crate) fn one(&mut self, stmt: &Stmt) -> Result<(), TranslateError> {
        match stmt {
            Stmt::Op(op) => ops::step(self, *op),
            Stmt::Block { ty, body } => self.plain(ty, Region::Block, body),
            Stmt::Loop { ty, body } => self.plain(ty, Region::Loop, body),
            Stmt::If {
                ty,
                then_body,
                else_body,
            } => self.emit_if(ty, then_body, else_body),
            Stmt::Br(depth) => self.write_branch(*depth),
            Stmt::BrIf(depth) => self.br_if(*depth),
        }
    }

    pub(crate) fn plain(
        &mut self,
        ty: &BlockTy,
        kind: Region,
        body: &[Stmt],
    ) -> Result<(), TranslateError> {
        if !ty.params.is_empty() || !ty.results.is_empty() {
            return Err(TranslateError::Unsupported);
        }
        self.region(kind, body)
    }

    pub(crate) fn emit_if(
        &mut self,
        ty: &BlockTy,
        then_body: &[Stmt],
        else_body: &[Stmt],
    ) -> Result<(), TranslateError> {
        match if_kind(ty, then_body, else_body) {
            IfKind::Bad => Err(TranslateError::Unsupported),
            IfKind::Pure => self.pure_if(ty, then_body, else_body),
            IfKind::Target => self.targeted_if(ty, then_body, else_body),
            IfKind::Stmt => self.stmt_if(ty, then_body, else_body, true),
        }
    }

    pub(crate) fn pure_if(
        &mut self,
        ty: &BlockTy,
        then_body: &[Stmt],
        else_body: &[Stmt],
    ) -> Result<(), TranslateError> {
        let cond_val = self.pop()?;
        let cond = self.as_bool(cond_val)?;
        if ty.results.is_empty() {
            return Ok(());
        }
        let saved = self.stack.clone();
        self.run(then_body)?;
        let accept = self.pop_n(ty.results.len())?;
        self.stack = saved;
        self.run(else_body)?;
        let reject = self.pop_n(ty.results.len())?;
        self.push_selects(cond, &accept, &reject)
    }

    pub(crate) fn push_selects(
        &mut self,
        cond: naga::Handle<Expression>,
        accept: &[Val],
        reject: &[Val],
    ) -> Result<(), TranslateError> {
        let mut index = 0usize;
        while index < accept.len() {
            let value = self.select_val(cond, accept[index], reject[index])?;
            self.stack.push(value);
            index += 1;
        }
        Ok(())
    }

    pub(crate) fn select_val(
        &mut self,
        cond: naga::Handle<Expression>,
        accept: Val,
        reject: Val,
    ) -> Result<Val, TranslateError> {
        match (accept, reject) {
            (Val::Scalar(left), Val::Scalar(right)) => {
                Ok(Val::Scalar(self.pick(cond, left, right)))
            }
            (Val::Vec(left), Val::Vec(right)) => Ok(Val::Vec(self.pick(cond, left, right))),
            (left, right) => {
                let accept = self.as_uint(left)?;
                let reject = self.as_uint(right)?;
                Ok(Val::Index(self.pick(cond, accept, reject)))
            }
        }
    }

    pub(crate) fn targeted_if(
        &mut self,
        ty: &BlockTy,
        then_body: &[Stmt],
        else_body: &[Stmt],
    ) -> Result<(), TranslateError> {
        let flag = self.make_flag(Region::If, true)?;
        let body = self.sub(|em| {
            em.clear_flag(flag)?;
            em.labels.push(flag);
            em.stmt_if(ty, then_body, else_body, false)?;
            em.labels.pop();
            em.block.push(naga::Statement::Break, sp());
            Ok(())
        })?;
        self.push_loop(body);
        Ok(())
    }

    pub(crate) fn stmt_if(
        &mut self,
        ty: &BlockTy,
        then_body: &[Stmt],
        else_body: &[Stmt],
        as_label: bool,
    ) -> Result<(), TranslateError> {
        let cond_val = self.pop()?;
        let cond = self.as_bool(cond_val)?;
        let temps = self.temps(&ty.results)?;
        if as_label {
            let flag = self.make_flag(Region::If, false)?;
            self.labels.push(flag);
        }
        let saved = self.stack.clone();
        let accept = self.sub(|em| em.arm(then_body, &temps, &ty.results))?;
        self.stack = saved.clone();
        let reject = self.sub(|em| em.arm(else_body, &temps, &ty.results))?;
        self.stack = saved;
        self.block.push(
            naga::Statement::If {
                condition: cond,
                accept,
                reject,
            },
            sp(),
        );
        self.load_temps(&temps, &ty.results)?;
        if as_label {
            self.labels.pop();
        }
        Ok(())
    }

    pub(crate) fn arm(
        &mut self,
        body: &[Stmt],
        temps: &[naga::Handle<naga::LocalVariable>],
        kinds: &[Kind],
    ) -> Result<(), TranslateError> {
        self.run(body)?;
        self.store_temps(temps, kinds)
    }
}

mod branch;
mod shape;

use shape::IfKind;
use shape::bind_locals;
use shape::copy_params;
use shape::if_kind;
use shape::is_nested;
use shape::needs_shadow;
