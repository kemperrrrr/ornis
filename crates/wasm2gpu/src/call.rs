//! Defined wasm calls as naga functions, specialized by immediate provenance.
//!
//! A constant `i32` argument is not a naga parameter. The callee reads it
//! back as an immediate, so `storage_read`'s group and binding stay
//! `i32.const` inside the specialization. A dynamic index and every `f32`
//! are real arguments. A constant result is pushed as an immediate on the
//! caller, so a returned binding can feed the next import. Recursion and
//! `call_indirect` are rejected before a function is emitted.

use naga::Expression;
use naga::Function;
use naga::FunctionArgument;
use naga::Span;
use naga::Statement;

use crate::cx::Cx;
use crate::cx::Spec;
use crate::cx::Val;
use crate::cx::sp;
use crate::env::Emit;
use crate::env::ExecIn;
use crate::error::TranslateError;
use crate::flow;
use crate::ir::Func;
use crate::ir::Import;
use crate::ir::Kind;
use crate::ir::MAX_PARAMS;
use crate::ir::Sig;
use crate::ty::StructDef;

mod ret;

/// Lower `index` (a wasm function index) and emit the call.
pub(super) fn call_defined(em: &mut Emit<'_, '_>, index: u32) -> Result<(), TranslateError> {
    let base = em.imports.len() as u32;
    if index < base {
        return Err(TranslateError::Unsupported);
    }
    let func_index = index - base;
    let sig = em
        .funcs
        .get(func_index as usize)
        .ok_or(TranslateError::Unsupported)?
        .sig;
    let args = em.take(sig.params as usize)?;
    let baked = bake(&args, &sig)?;
    let spec = ensure(em.cx, em.funcs, em.imports, em.structs, func_index, &baked)?;
    invoke(em, &spec, &args)
}

fn bake(args: &[Val], sig: &Sig) -> Result<Vec<Option<u32>>, TranslateError> {
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        out.push(one_baked(sig.param_kind[index], args[index])?);
        index += 1;
    }
    Ok(out)
}

fn one_baked(kind: Kind, val: Val) -> Result<Option<u32>, TranslateError> {
    match (kind, val) {
        (Kind::I32, Val::Imm(value)) => Ok(Some(value)),
        (Kind::I32, Val::Index(_)) => Ok(None),
        (Kind::F32, Val::Scalar(_)) => Ok(None),
        _ => Err(TranslateError::Unsupported),
    }
}

fn ensure(
    cx: &mut Cx<'_>,
    funcs: &[Func],
    imports: &[Import],
    structs: &[StructDef],
    func_index: u32,
    baked: &[Option<u32>],
) -> Result<Spec, TranslateError> {
    let key = baked_key(baked);
    if let Some(found) = find(cx, func_index, baked.len(), &key) {
        return Ok(found);
    }
    if cx.active.contains(&func_index) {
        return Err(TranslateError::Recursion);
    }
    cx.active.push(func_index);
    let lowered = lower_one(cx, funcs, imports, structs, func_index, baked, &key);
    cx.active.pop();
    let spec = lowered?;
    cx.specs.push(spec);
    Ok(spec)
}

fn find(cx: &Cx<'_>, func: u32, nparams: usize, key: &[Option<u32>; MAX_PARAMS]) -> Option<Spec> {
    let mut index = 0usize;
    while index < cx.specs.len() {
        let spec = cx.specs[index];
        if spec.func == func && spec.nparams as usize == nparams && spec.baked == *key {
            return Some(spec);
        }
        index += 1;
    }
    None
}

fn lower_one(
    cx: &mut Cx<'_>,
    funcs: &[Func],
    imports: &[Import],
    structs: &[StructDef],
    func_index: u32,
    baked: &[Option<u32>],
    key: &[Option<u32>; MAX_PARAMS],
) -> Result<Spec, TranslateError> {
    let sig = funcs
        .get(func_index as usize)
        .ok_or(TranslateError::Unsupported)?
        .sig;
    let name = format!("fn{func_index}_{}", cx.specs.len());
    let mut body = Function {
        name: Some(name),
        arguments: naga_args(cx, &sig, baked)?,
        ..Function::default()
    };
    let input = ExecIn {
        func_index,
        baked: baked.to_vec(),
        id_param: None,
    };
    let stack = flow::exec(cx, imports, structs, funcs, &mut body, &input)?;
    let (result_imm, nresults) = ret::finish(&mut body, cx, &stack, &sig)?;
    let handle = cx.module.functions.append(body, Span::default());
    Ok(Spec {
        func: func_index,
        baked: *key,
        nparams: u8::try_from(sig.params).map_err(|_| TranslateError::Unsupported)?,
        handle,
        result_imm,
        nresults,
        result_kind: sig.result_kind,
    })
}

fn naga_args(
    cx: &Cx<'_>,
    sig: &Sig,
    baked: &[Option<u32>],
) -> Result<Vec<FunctionArgument>, TranslateError> {
    let mut out = Vec::new();
    let mut index = 0u32;
    while index < sig.params {
        if baked.get(index as usize).copied().flatten().is_none() {
            let kind = sig.param_kind[index as usize];
            let ty = if kind == Kind::I32 {
                cx.uint
            } else {
                cx.scalar
            };
            out.push(FunctionArgument {
                name: None,
                ty,
                binding: None,
            });
        }
        index += 1;
    }
    Ok(out)
}

fn invoke(em: &mut Emit<'_, '_>, spec: &Spec, args: &[Val]) -> Result<(), TranslateError> {
    let mut arguments = Vec::new();
    let mut index = 0usize;
    while index < spec.nparams as usize {
        if spec.baked[index].is_none() {
            arguments.push(arg_expr(em, args[index])?);
        }
        index += 1;
    }
    let dyn_count = count_dynamic(spec);
    let result = if dyn_count == 0 {
        None
    } else {
        Some(
            em.func
                .expressions
                .append(Expression::CallResult(spec.handle), Span::default()),
        )
    };
    em.block.push(
        Statement::Call {
            function: spec.handle,
            arguments,
            result,
        },
        sp(),
    );
    push_results(em, spec, result, dyn_count)
}

fn arg_expr(em: &mut Emit<'_, '_>, val: Val) -> Result<naga::Handle<Expression>, TranslateError> {
    match val {
        Val::Index(handle) | Val::Scalar(handle) => Ok(handle),
        Val::Imm(value) => Ok(em.u32(value)),
        Val::Vec(_) => Err(TranslateError::Unsupported),
    }
}

fn count_dynamic(spec: &Spec) -> u32 {
    let mut count = 0u32;
    let mut index = 0usize;
    while index < spec.nresults as usize {
        if spec.result_imm[index].is_none() {
            count += 1;
        }
        index += 1;
    }
    count
}

fn push_results(
    em: &mut Emit<'_, '_>,
    spec: &Spec,
    result: Option<naga::Handle<Expression>>,
    dyn_count: u32,
) -> Result<(), TranslateError> {
    let mut dyn_at = 0u32;
    let mut index = 0usize;
    while index < spec.nresults as usize {
        if let Some(value) = spec.result_imm[index] {
            em.push(Val::Imm(value))?;
        } else {
            let base = result.ok_or(TranslateError::Unsupported)?;
            let handle = if dyn_count == 1 {
                base
            } else {
                em.ex(Expression::AccessIndex {
                    base,
                    index: dyn_at,
                })
            };
            let val = if spec.result_kind[index] == Kind::F32 {
                Val::Scalar(handle)
            } else {
                Val::Index(handle)
            };
            em.push(val)?;
            dyn_at += 1;
        }
        index += 1;
    }
    Ok(())
}

fn baked_key(baked: &[Option<u32>]) -> [Option<u32>; MAX_PARAMS] {
    let mut key = [None; MAX_PARAMS];
    let mut index = 0usize;
    while index < baked.len() && index < MAX_PARAMS {
        key[index] = baked[index];
        index += 1;
    }
    key
}
