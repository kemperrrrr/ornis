//! Mutable lowering state: naga function, current block, and value stack.

use naga::Expression;
use naga::Literal;
use naga::Span;

use crate::cx::Cx;
use crate::cx::Val;
use crate::cx::expr_in;
use crate::cx::sp;
use crate::error::TranslateError;
use crate::ir::Import;
use crate::ty::StructDef;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Region {
    Block,
    Loop,
    If,
}

#[derive(Clone, Copy)]
pub(super) struct Flag {
    pub(super) kind: Region,
    pub(super) naga_loop: bool,
    pub(super) leave: naga::Handle<naga::LocalVariable>,
    pub(super) again: naga::Handle<naga::LocalVariable>,
}

pub(super) struct Emit<'a, 'b> {
    pub(super) cx: &'b mut Cx<'a>,
    pub(super) func: &'b mut naga::Function,
    pub(super) block: naga::Block,
    pub(super) stack: Vec<Val>,
    pub(super) imports: &'b [Import],
    pub(super) structs: &'b [StructDef],
    pub(super) funcs: &'b [crate::ir::Func],
    pub(super) locals: Vec<Option<naga::Handle<naga::LocalVariable>>>,
    /// Storage provenance of the wasm local held in each shadow slot.
    ///
    /// `None` is a plain scalar. `local_set` stores `Some((group, binding))`
    /// with a [`Val::Ptr`] and clears it on any other value, so a later
    /// `local_get` wraps the offset it loaded back into a pointer.
    pub(super) ptr_slots: Vec<Option<(u32, u32)>>,
    pub(super) labels: Vec<Flag>,
    /// Immediate values of specialized parameters. Empty for an entry.
    pub(super) baked: Vec<Option<u32>>,
    /// Wasm local index to naga argument index. `None` is not a parameter.
    pub(super) args_at: Vec<Option<u32>>,
    /// Wasm parameter that reads `global_invocation_id.x`, if this is `cs_main`.
    pub(super) id_param: Option<u32>,
    /// Wasm result count. A `return` is emitted only when this is zero.
    pub(super) results: u32,
    /// A `return` already left this function. The enclosing loop must not
    /// also `break`, or naga sees that break as unreachable.
    pub(super) halted: bool,
}

/// Imports and defined functions visible to one lowering.
pub(super) struct Code<'a> {
    /// Wasm imports, in index order.
    pub imports: &'a [Import],
    /// Struct layouts named by the imports.
    pub structs: &'a [StructDef],
    /// Defined functions. Index 0 is the first function after the imports.
    pub funcs: &'a [crate::ir::Func],
}

/// How [`crate::flow::exec`] binds one function.
pub(super) struct ExecIn {
    /// Index into the module's defined functions.
    pub func_index: u32,
    /// Immediate `i32` parameters. Empty when every parameter is dynamic.
    pub baked: Vec<Option<u32>>,
    /// See [`Emit::id_param`].
    pub id_param: Option<u32>,
}

impl Emit<'_, '_> {
    pub(super) fn ex(&mut self, value: Expression) -> naga::Handle<Expression> {
        expr_in(self.func, &mut self.block, value)
    }

    pub(super) fn sub(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<(), TranslateError>,
    ) -> Result<naga::Block, TranslateError> {
        let parent = std::mem::replace(&mut self.block, naga::Block::new());
        let result = f(self);
        let child = std::mem::replace(&mut self.block, parent);
        result?;
        Ok(child)
    }

    pub(super) fn pop(&mut self) -> Result<Val, TranslateError> {
        self.stack.pop().ok_or(TranslateError::Unsupported)
    }

    pub(super) fn push(&mut self, val: Val) -> Result<(), TranslateError> {
        self.stack.push(val);
        Ok(())
    }

    pub(super) fn take(&mut self, n: usize) -> Result<Vec<Val>, TranslateError> {
        let mut vals = Vec::new();
        let mut index = 0usize;
        while index < n {
            vals.push(self.pop()?);
            index += 1;
        }
        vals.reverse();
        Ok(vals)
    }

    pub(super) fn pop_uint(&mut self) -> Result<naga::Handle<Expression>, TranslateError> {
        let val = self.pop()?;
        self.as_uint(val)
    }

    pub(super) fn pop_imm(&mut self) -> Result<u32, TranslateError> {
        match self.pop()? {
            Val::Imm(value) => Ok(value),
            _ => Err(TranslateError::DynamicBinding),
        }
    }

    pub(super) fn u32(&mut self, value: u32) -> naga::Handle<Expression> {
        self.ex(Expression::Literal(Literal::U32(value)))
    }

    pub(super) fn f32(&mut self, value: f32) -> naga::Handle<Expression> {
        self.ex(Expression::Literal(Literal::F32(value)))
    }

    pub(super) fn bool(&mut self, value: bool) -> naga::Handle<Expression> {
        self.ex(Expression::Literal(Literal::Bool(value)))
    }

    /// Lower a pointer's byte offset to an expression.
    pub(super) fn off_expr(&mut self, off: crate::cx::PtrOff) -> naga::Handle<Expression> {
        match off {
            crate::cx::PtrOff::Imm(value) => self.u32(value),
            crate::cx::PtrOff::Dyn(handle) => handle,
        }
    }

    /// Merge two pointer values behind `cond`, or `None` when either is
    /// not a pointer or their provenance disagrees.
    pub(super) fn merge_ptr(
        &mut self,
        cond: naga::Handle<Expression>,
        accept: Val,
        reject: Val,
    ) -> Option<Val> {
        let (
            Val::Ptr {
                group: accept_group,
                binding: accept_binding,
                off: accept_off,
            },
            Val::Ptr {
                group: reject_group,
                binding: reject_binding,
                off: reject_off,
            },
        ) = (accept, reject) else {
            return None;
        };
        if accept_group != reject_group || accept_binding != reject_binding {
            return None;
        }
        let left = self.off_expr(accept_off);
        let right = self.off_expr(reject_off);
        Some(Val::Ptr {
            group: accept_group,
            binding: accept_binding,
            off: crate::cx::PtrOff::Dyn(self.pick(cond, left, right)),
        })
    }

    pub(super) fn as_uint(&mut self, val: Val) -> Result<naga::Handle<Expression>, TranslateError> {
        match val {
            Val::Imm(value) => Ok(self.u32(value)),
            Val::Index(handle) => Ok(handle),
            _ => Err(TranslateError::Unsupported),
        }
    }

    pub(super) fn as_bool(&mut self, val: Val) -> Result<naga::Handle<Expression>, TranslateError> {
        match val {
            Val::Imm(0) => Ok(self.bool(false)),
            Val::Imm(_) => Ok(self.bool(true)),
            Val::Index(handle) => {
                let zero = self.u32(0);
                Ok(self.ex(Expression::Binary {
                    op: naga::BinaryOperator::NotEqual,
                    left: handle,
                    right: zero,
                }))
            }
            _ => Err(TranslateError::Unsupported),
        }
    }

    pub(super) fn var(
        &mut self,
        ty: naga::Handle<naga::Type>,
        init: naga::Handle<Expression>,
    ) -> naga::Handle<naga::LocalVariable> {
        self.func.local_variables.append(
            naga::LocalVariable {
                name: None,
                ty,
                init: Some(init),
            },
            sp(),
        )
    }

    pub(super) fn load_var(
        &mut self,
        var: naga::Handle<naga::LocalVariable>,
    ) -> naga::Handle<Expression> {
        let pointer = self.ex(Expression::LocalVariable(var));
        self.ex(Expression::Load { pointer })
    }

    pub(super) fn store_var(
        &mut self,
        var: naga::Handle<naga::LocalVariable>,
        value: naga::Handle<Expression>,
    ) {
        let pointer = self.ex(Expression::LocalVariable(var));
        self.block
            .push(naga::Statement::Store { pointer, value }, sp());
    }

    pub(super) fn pick(
        &mut self,
        cond: naga::Handle<Expression>,
        accept: naga::Handle<Expression>,
        reject: naga::Handle<Expression>,
    ) -> naga::Handle<Expression> {
        self.ex(Expression::Select {
            condition: cond,
            accept,
            reject,
        })
    }

    /// Value of wasm parameter `index`.
    ///
    /// A compute entry's parameter is `global_invocation_id.x`, not the
    /// `vec3` argument itself.
    pub(super) fn param_expr(
        &mut self,
        index: u32,
    ) -> Result<naga::Handle<Expression>, TranslateError> {
        let slot = self
            .args_at
            .get(index as usize)
            .copied()
            .flatten()
            .ok_or(TranslateError::Unsupported)?;
        let base = self.ex(Expression::FunctionArgument(slot));
        if self.id_param == Some(index) {
            return Ok(self.ex(Expression::AccessIndex { base, index: 0 }));
        }
        Ok(base)
    }
}

pub(super) fn float_of(val: Val) -> Result<naga::Handle<Expression>, TranslateError> {
    match val {
        Val::Scalar(handle) => Ok(handle),
        _ => Err(TranslateError::Unsupported),
    }
}

pub(super) fn const_expr(func: &mut naga::Function, value: Expression) -> naga::Handle<Expression> {
    func.expressions.append(value, Span::default())
}
