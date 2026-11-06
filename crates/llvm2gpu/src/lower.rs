//! Lower a parsed LLVM function to a Naga compute entry.
//!
//! The spike subset is straight-line scalar arithmetic plus `select`
//! and a single scalar return. The return value is written to storage
//! `0_0` at the invocation id, mirroring the toy `compute` kernel that
//! `wasm2gpu` already tests. Control flow (`phi`/`br`/`loop`) and
//! memory (`load`/`store`) are rejected here; they are phase-2 work.

use crate::error::TranslateError;
use crate::parse::{Instr, LlvmFunc, Operand};
use std::collections::HashMap;

/// Lower `func` (named `cs_main`) to a Naga module with one compute entry.
pub(super) fn lower(func: &LlvmFunc) -> Result<naga::Module, TranslateError> {
    if func.name != "cs_main" || func.ret_ty != crate::parse::Ty::Float {
        return Err(TranslateError::Unsupported);
    }
    if func.params.len() != 1 || func.params[0].0 != crate::parse::Ty::I32 {
        return Err(TranslateError::Unsupported);
    }
    let mut module = Builders::new();
    let function = module.build(func)?;
    module.finish(function)
}

struct Builders {
    module: naga::Module,
    storage_ty: naga::Handle<naga::Type>,
}

impl Builders {
    fn new() -> Self {
        let mut module = naga::Module::default();
        let f32_ty = scalar(&mut module, naga::ScalarKind::Float);
        let array_ty = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Array {
                    base: f32_ty,
                    size: naga::ArraySize::Dynamic,
                    stride: 4,
                },
            },
            naga::Span::default(),
        );
        Self {
            module,
            storage_ty: array_ty,
        }
    }

    fn finish(mut self, function: naga::Function) -> Result<naga::Module, TranslateError> {
        self.module.entry_points.push(naga::EntryPoint {
            name: ornis_gpu_abi::symbols::CS_MAIN.to_string(),
            stage: naga::ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [4, 1, 1],
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });
        Ok(self.module)
    }

    fn build(&mut self, func: &LlvmFunc) -> Result<naga::Function, TranslateError> {
        let mut function = naga::Function {
            name: Some(ornis_gpu_abi::symbols::CS_MAIN.to_string()),
            ..Default::default()
        };
        function.arguments.push(naga::FunctionArgument {
            name: Some("global_invocation_id".to_string()),
            ty: self.module.types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Vector {
                        size: naga::VectorSize::Tri,
                        scalar: naga::Scalar {
                            kind: naga::ScalarKind::Uint,
                            width: 4,
                        },
                    },
                },
                naga::Span::default(),
            ),
            binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::GlobalInvocationId)),
        });
        function.body = naga::Block::new();

        let storage_global = self.module.global_variables.append(
            naga::GlobalVariable {
                name: Some("storage_0_0_".to_string()),
                space: naga::AddressSpace::Storage {
                    access: naga::StorageAccess::LOAD | naga::StorageAccess::STORE,
                },
                binding: Some(naga::ResourceBinding {
                    group: 0,
                    binding: 0,
                }),
                ty: self.storage_ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        );

        let arg = append(&mut function, naga::Expression::FunctionArgument(0));
        push_emit(&mut function, arg);
        let id_x = append(
            &mut function,
            naga::Expression::AccessIndex {
                base: arg,
                index: 0,
            },
        );
        push_emit(&mut function, id_x);

        let mut env: HashMap<String, naga::Handle<naga::Expression>> = HashMap::new();
        env.insert(func.params[0].1.clone(), id_x);

        for instr in &func.body {
            match instr {
                Instr::Ret { a } => {
                    let value = operand(&mut function, a, &env)?;
                    let gvar = append(
                        &mut function,
                        naga::Expression::GlobalVariable(storage_global),
                    );
                    let pointer = append(
                        &mut function,
                        naga::Expression::Access {
                            base: gvar,
                            index: id_x,
                        },
                    );
                    push_emit(&mut function, pointer);
                    function.body.push(
                        naga::Statement::Store { pointer, value },
                        naga::Span::default(),
                    );
                    function.body.push(
                        naga::Statement::Return { value: None },
                        naga::Span::default(),
                    );
                }
                Instr::FAdd { dst, a, b } => {
                    let v = binary(&mut function, naga::BinaryOperator::Add, a, b, &env)?;
                    env.insert(dst.clone(), v);
                }
                Instr::FMul { dst, a, b } => {
                    let v = binary(&mut function, naga::BinaryOperator::Multiply, a, b, &env)?;
                    env.insert(dst.clone(), v);
                }
                Instr::FSub { dst, a, b } => {
                    let v = binary(&mut function, naga::BinaryOperator::Subtract, a, b, &env)?;
                    env.insert(dst.clone(), v);
                }
                Instr::FDiv { dst, a, b } => {
                    let v = binary(&mut function, naga::BinaryOperator::Divide, a, b, &env)?;
                    env.insert(dst.clone(), v);
                }
                Instr::FNeg { dst, a } => {
                    let src = operand(&mut function, a, &env)?;
                    let v = append(
                        &mut function,
                        naga::Expression::Unary {
                            op: naga::UnaryOperator::Negate,
                            expr: src,
                        },
                    );
                    push_emit(&mut function, v);
                    env.insert(dst.clone(), v);
                }
                Instr::Add { dst, a, b } => {
                    let v = binary(&mut function, naga::BinaryOperator::Add, a, b, &env)?;
                    env.insert(dst.clone(), v);
                }
                Instr::Sub { dst, a, b } => {
                    let v = binary(&mut function, naga::BinaryOperator::Subtract, a, b, &env)?;
                    env.insert(dst.clone(), v);
                }
                Instr::Mul { dst, a, b } => {
                    let v = binary(&mut function, naga::BinaryOperator::Multiply, a, b, &env)?;
                    env.insert(dst.clone(), v);
                }
                Instr::Fcmp { dst, pred, a, b } => {
                    let v = compare(&mut function, pred, a, b, &env)?;
                    env.insert(dst.clone(), v);
                }
                Instr::Icmp { dst, pred, a, b } => {
                    let v = compare(&mut function, pred, a, b, &env)?;
                    env.insert(dst.clone(), v);
                }
                Instr::Select { dst, c, a, b } => {
                    let cond = operand(&mut function, c, &env)?;
                    let accept = operand(&mut function, a, &env)?;
                    let reject = operand(&mut function, b, &env)?;
                    let v = append(
                        &mut function,
                        naga::Expression::Select {
                            condition: cond,
                            accept,
                            reject,
                        },
                    );
                    push_emit(&mut function, v);
                    env.insert(dst.clone(), v);
                }
                Instr::Uitofp { dst, a } => {
                    let src = operand(&mut function, a, &env)?;
                    let v = append(
                        &mut function,
                        naga::Expression::As {
                            expr: src,
                            kind: naga::ScalarKind::Float,
                            convert: Some(4),
                        },
                    );
                    push_emit(&mut function, v);
                    env.insert(dst.clone(), v);
                }
                Instr::Call { dst, callee, args } => {
                    let v = call(&mut function, callee, args, &env)?;
                    env.insert(dst.clone(), v);
                }
            }
        }
        Ok(function)
    }
}

fn scalar(module: &mut naga::Module, kind: naga::ScalarKind) -> naga::Handle<naga::Type> {
    module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar { kind, width: 4 }),
        },
        naga::Span::default(),
    )
}

fn append(func: &mut naga::Function, expr: naga::Expression) -> naga::Handle<naga::Expression> {
    func.expressions.append(expr, naga::Span::default())
}

fn push_emit(func: &mut naga::Function, handle: naga::Handle<naga::Expression>) {
    if !func.expressions[handle].needs_pre_emit() {
        let range = func.expressions.range_from(handle.index());
        func.body
            .push(naga::Statement::Emit(range), naga::Span::default());
    }
}

fn operand(
    func: &mut naga::Function,
    op: &Operand,
    env: &HashMap<String, naga::Handle<naga::Expression>>,
) -> Result<naga::Handle<naga::Expression>, TranslateError> {
    match op {
        Operand::Local(name) => env.get(name).copied().ok_or(TranslateError::Unsupported),
        Operand::F32(v) => Ok(append(
            func,
            naga::Expression::Literal(naga::Literal::F32(*v)),
        )),
        Operand::U32(v) => Ok(append(
            func,
            naga::Expression::Literal(naga::Literal::U32(*v)),
        )),
        Operand::Bool(v) => Ok(append(
            func,
            naga::Expression::Literal(naga::Literal::Bool(*v)),
        )),
    }
}

fn binary(
    func: &mut naga::Function,
    op: naga::BinaryOperator,
    a: &Operand,
    b: &Operand,
    env: &HashMap<String, naga::Handle<naga::Expression>>,
) -> Result<naga::Handle<naga::Expression>, TranslateError> {
    let left = operand(func, a, env)?;
    let right = operand(func, b, env)?;
    let v = append(func, naga::Expression::Binary { op, left, right });
    push_emit(func, v);
    Ok(v)
}

fn compare(
    func: &mut naga::Function,
    pred: &str,
    a: &Operand,
    b: &Operand,
    env: &HashMap<String, naga::Handle<naga::Expression>>,
) -> Result<naga::Handle<naga::Expression>, TranslateError> {
    let op = match pred {
        "oeq" | "ueq" | "eq" => naga::BinaryOperator::Equal,
        "one" | "une" | "ne" => naga::BinaryOperator::NotEqual,
        "olt" | "ult" | "slt" => naga::BinaryOperator::Less,
        "ole" | "ule" | "sle" => naga::BinaryOperator::LessEqual,
        "ogt" | "ugt" | "sgt" => naga::BinaryOperator::Greater,
        "oge" | "uge" | "sge" => naga::BinaryOperator::GreaterEqual,
        _ => return Err(TranslateError::Unsupported),
    };
    let left = operand(func, a, env)?;
    let right = operand(func, b, env)?;
    let v = append(func, naga::Expression::Binary { op, left, right });
    push_emit(func, v);
    Ok(v)
}

fn call(
    func: &mut naga::Function,
    callee: &str,
    args: &[Operand],
    env: &HashMap<String, naga::Handle<naga::Expression>>,
) -> Result<naga::Handle<naga::Expression>, TranslateError> {
    if args.len() != 1 {
        return Err(TranslateError::Unsupported);
    }
    let arg = operand(func, &args[0], env)?;
    let fun = if callee.contains("sqrt") {
        naga::MathFunction::Sqrt
    } else if callee.contains("sin") {
        naga::MathFunction::Sin
    } else if callee.contains("cos") {
        naga::MathFunction::Cos
    } else if callee.contains("exp") {
        naga::MathFunction::Exp
    } else if callee.contains("log") {
        naga::MathFunction::Log
    } else {
        return Err(TranslateError::Unsupported);
    };
    let v = append(
        func,
        naga::Expression::Math {
            fun,
            arg,
            arg1: None,
            arg2: None,
            arg3: None,
        },
    );
    push_emit(func, v);
    Ok(v)
}
