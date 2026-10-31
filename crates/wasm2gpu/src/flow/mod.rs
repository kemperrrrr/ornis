//! Structured wasm statements to naga `if`, `loop`, `switch`, and `select`.
//!
//! A pure `if` (no calls, stores, or branches) becomes `select`. Anything with
//! a side effect becomes a statement. `loop` and `block` are naga loops so
//! `br` can `break` or `continue`. A branch that skips a loop sets a `leave`
//! or `again` flag and breaks the inner loop. `br_table` is a `switch` whose
//! arms set those same flags. The following statement checks the nearest
//! enclosing naga loop, including inside a statement `if`.

use naga::Expression;

use crate::cx::Val;
use crate::cx::sp;
use crate::env::Code;
use crate::env::Emit;
use crate::env::ExecIn;
use crate::env::Region;
use crate::error::TranslateError;
use crate::ir::BlockTy;
use crate::ir::Kind;
use crate::ir::Stmt;
use crate::locals::Locals;
use crate::locals::arg_map;
use crate::ops;

pub(super) fn exec(
    cx: &mut crate::cx::Cx<'_>,
    code: &Code<'_>,
    body: &mut naga::Function,
    input: &ExecIn,
) -> Result<Vec<Val>, TranslateError> {
    let baked = input.baked.clone();
    let id_param = input.id_param;
    let (args_at, locals, stmts, results, wasm_locals, params) = {
        let func = code
            .funcs
            .get(input.func_index as usize)
            .ok_or(TranslateError::Unsupported)?;
        // The analyses read the same tree the emitter lowers.
        let stmts = canon::rewrite(func);
        let slots = Slots::of(&stmts, func.locals.len());
        let args_at = arg_map(func, &input.baked, input.id_param);
        let locals = bind_params(body, func, cx, &slots)?;
        let results = func.sig.results;
        let wasm_locals = Locals::of(func, slots.expr);
        (
            args_at,
            locals,
            stmts,
            results,
            wasm_locals,
            func.sig.params,
        )
    };
    let block = std::mem::take(&mut body.body);
    let slots = locals.len();
    let mut em = Emit {
        cx,
        func: body,
        block,
        stack: Vec::new(),
        ptr_slots: vec![None; slots],
        locals,
        wasm_locals,
        labels: Vec::new(),
        baked,
        args_at,
        id_param,
        results,
        halted: false,
        imports: code.imports,
        structs: code.structs,
        funcs: code.funcs,
    };
    copy_params(&mut em, params)?;
    em.run(&stmts)?;
    let Emit {
        block, stack, func, ..
    } = em;
    func.body = block;
    Ok(stack)
}

impl Emit<'_, '_> {
    pub(crate) fn run(&mut self, stmts: &[Stmt]) -> Result<(), TranslateError> {
        for stmt in stmts {
            if self.halted {
                break;
            }
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
            Stmt::BrTable { targets, default } => self.br_table(targets, *default),
            Stmt::Return => self.fn_return(),
        }
    }

    fn fn_return(&mut self) -> Result<(), TranslateError> {
        if self.results != 0 {
            return Err(TranslateError::FunctionBranch);
        }
        self.block
            .push(naga::Statement::Return { value: None }, sp());
        self.halted = true;
        Ok(())
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
        if kind == Region::Block && !Escape::of(body).control {
            // A block without a branch is a scope: no branch can target it,
            // so it needs neither a label nor a loop.
            return self.run(body);
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
        // No results and no side effects: nothing to select. Both arms are
        // dropped and only the condition is consumed.
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
        if let Some(merged) = self.merge_ptr(cond, accept, reject) {
            return Ok(merged);
        }
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
        // A branch targets this `if`, so it needs the `leave` flag.
        let flag = self.make_flag(Region::If, true, true, false)?;
        let body = self.sub(|em| {
            em.clear_flag(flag)?;
            em.labels.push(flag);
            em.stmt_if(ty, then_body, else_body, false)?;
            em.close_region();
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
            // No branch targets a statement `if`, so it needs no flags.
            let flag = self.make_flag(Region::If, false, false, false)?;
            self.labels.push(flag);
        }
        let saved = self.stack.clone();
        let saved_ptrs = self.ptr_slots.clone();
        // Each arm starts fresh: a `return` in one arm leaves that arm, not
        // the other one.
        self.halted = false;
        let accept = self.sub(|em| em.arm(then_body, &temps, &ty.results))?;
        let then_ptrs = self.ptr_slots.clone();
        let then_stopped = self.halted;
        self.stack = saved.clone();
        self.ptr_slots = saved_ptrs;
        self.halted = false;
        let reject = self.sub(|em| em.arm(else_body, &temps, &ty.results))?;
        let else_ptrs = self.ptr_slots.clone();
        let else_stopped = self.halted;
        self.stack = saved;
        self.ptr_slots = final_ptrs(cond_val, then_stopped, then_ptrs, else_ptrs);
        // The code after the `if` is dead only when both arms returned.
        self.halted = then_stopped && else_stopped;
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

/// Storage provenance each shadow slot holds after an `if`.
///
/// A constant condition folds in the analysis, so only the live arm's
/// slots describe the state after the branch. With a dynamic condition
/// both arms are live and the analysis has already forced them onto one
/// binding per slot; a then-arm that returned leaves only the else path,
/// so the else slots win there.
fn final_ptrs(
    cond: Val,
    then_stopped: bool,
    then_slots: Vec<Option<(u32, u32)>>,
    else_slots: Vec<Option<(u32, u32)>>,
) -> Vec<Option<(u32, u32)>> {
    match cond {
        Val::Imm(0) => else_slots,
        Val::Imm(_) => then_slots,
        _ if then_stopped => else_slots,
        _ => then_slots,
    }
}

mod branch;
mod canon;
mod shape;
mod slots;
mod switch;

use shape::Escape;
use shape::IfKind;
use shape::bind_params;
use shape::copy_params;
use shape::if_kind;
use shape::is_nested;
use slots::Slots;
