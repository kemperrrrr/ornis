//! Argument shorthands and the emission helpers built on them.

use super::SCALAR_WIDTH;
use super::cfg::Env;
use crate::error::TranslateError;
use crate::parse::{Instr, LlvmFunc, Operand};

pub(crate) fn const_u32(op: &Operand) -> Result<u32, TranslateError> {
    match op {
        Operand::U32(value) => Ok(*value),
        // A local is not materialised; group/binding must be immediates
        // (same rule as `wasm2gpu`).
        Operand::Local(_) | Operand::F32(_) | Operand::Bool(_) | Operand::Symbol(_) => {
            Err(TranslateError::unsupported())
        }
    }
}

pub(crate) fn scalar(
    module: &mut naga::Module,
    kind: naga::ScalarKind,
) -> naga::Handle<naga::Type> {
    module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar {
                kind,
                width: SCALAR_WIDTH,
            }),
        },
        naga::Span::default(),
    )
}

pub(crate) fn invocation_x(
    module: &mut naga::Module,
    function: &mut naga::Function,
) -> (
    naga::Handle<naga::Expression>,
    naga::Handle<naga::Expression>,
) {
    let ty = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Vector {
                size: naga::VectorSize::Tri,
                scalar: naga::Scalar {
                    kind: naga::ScalarKind::Uint,
                    width: SCALAR_WIDTH,
                },
            },
        },
        naga::Span::default(),
    );
    function.arguments.push(naga::FunctionArgument {
        name: Some("global_invocation_id".to_string()),
        ty,
        binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::GlobalInvocationId)),
    });
    let arg = append(function, naga::Expression::FunctionArgument(0));
    let id_x = append(
        function,
        naga::Expression::AccessIndex {
            base: arg,
            index: 0,
        },
    );
    let mut body = naga::Block::new();
    push_emit(function, &mut body, id_x);
    function.body = body;
    (arg, id_x)
}

pub(crate) fn uses_call(func: &LlvmFunc, needle: &str) -> bool {
    func.blocks.iter().any(|block| {
        block.instrs.iter().any(|instr| match instr {
            Instr::Call { callee, .. } | Instr::CallVoid { callee, .. } => callee.contains(needle),
            _ => false,
        })
    })
}

pub(crate) fn local_index_arg(
    module: &mut naga::Module,
    function: &mut naga::Function,
) -> naga::Handle<naga::Expression> {
    builtin_u32_arg(
        module,
        function,
        "local_invocation_index",
        naga::BuiltIn::LocalInvocationIndex,
    )
}

/// One `u32` builtin argument of the entry.
pub(crate) fn builtin_u32_arg(
    module: &mut naga::Module,
    function: &mut naga::Function,
    name: &str,
    built_in: naga::BuiltIn,
) -> naga::Handle<naga::Expression> {
    let index = function.arguments.len() as u32;
    function.arguments.push(naga::FunctionArgument {
        name: Some(name.to_string()),
        ty: scalar(module, naga::ScalarKind::Uint),
        binding: Some(naga::Binding::BuiltIn(built_in)),
    });
    append(function, naga::Expression::FunctionArgument(index))
}

/// One `@location(location)` argument of the entry.
pub(crate) fn location_arg(
    function: &mut naga::Function,
    ty: naga::Handle<naga::Type>,
    binding: naga::Binding,
    name: &str,
) -> naga::Handle<naga::Expression> {
    let index = function.arguments.len() as u32;
    function.arguments.push(naga::FunctionArgument {
        name: Some(name.to_string()),
        ty,
        binding: Some(binding),
    });
    append(function, naga::Expression::FunctionArgument(index))
}

pub(crate) fn position_arg(
    ty: naga::Handle<naga::Type>,
    function: &mut naga::Function,
) -> naga::Handle<naga::Expression> {
    let index = function.arguments.len() as u32;
    function.arguments.push(naga::FunctionArgument {
        name: Some("position".to_string()),
        ty,
        binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::Position {
            invariant: false,
        })),
    });
    append(function, naga::Expression::FunctionArgument(index))
}

pub(crate) fn append(
    func: &mut naga::Function,
    expr: naga::Expression,
) -> naga::Handle<naga::Expression> {
    func.expressions.append(expr, naga::Span::default())
}

pub(crate) fn push_emit(
    func: &mut naga::Function,
    out: &mut naga::Block,
    handle: naga::Handle<naga::Expression>,
) {
    if !func.expressions[handle].needs_pre_emit() {
        // The range is closed on purpose: an open-ended one would also cover
        // the `CallResult` of a helper call, which Naga rejects because a call
        // statement defines it.
        let range = naga::Range::new_from_bounds(handle, handle);
        out.push(naga::Statement::Emit(range), naga::Span::default());
    }
}

pub(crate) fn operand(
    func: &mut naga::Function,
    out: &mut naga::Block,
    op: &Operand,
    env: &Env,
) -> Result<naga::Handle<naga::Expression>, TranslateError> {
    match op {
        Operand::Local(name) => env
            .get(name)
            .copied()
            .ok_or_else(TranslateError::unsupported),
        Operand::F32(value) => {
            let handle = append(func, naga::Expression::Literal(naga::Literal::F32(*value)));
            push_emit(func, out, handle);
            Ok(handle)
        }
        Operand::U32(value) => {
            let handle = append(func, naga::Expression::Literal(naga::Literal::U32(*value)));
            push_emit(func, out, handle);
            Ok(handle)
        }
        Operand::Bool(value) => {
            let handle = append(func, naga::Expression::Literal(naga::Literal::Bool(*value)));
            push_emit(func, out, handle);
            Ok(handle)
        }
        // A module symbol is a panic location, and the call that carried it is
        // dropped; a zero keeps the value well-typed if one ever surfaces.
        Operand::Symbol(_) => {
            let handle = append(func, naga::Expression::Literal(naga::Literal::U32(0)));
            push_emit(func, out, handle);
            Ok(handle)
        }
    }
}
