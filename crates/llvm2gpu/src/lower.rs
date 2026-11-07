//! Lower a parsed LLVM function to a Naga compute entry.
//!
//! The spike subset is straight-line scalar arithmetic plus `select`,
//! storage byte accesses through the `ornis_gpu` ABI, and a single scalar
//! return or explicit stores. The return value (when present) is written
//! to storage `0_0` at the invocation id, mirroring the toy `compute`
//! kernel that `wasm2gpu` already tests. Control flow (`phi`/`br`/`loop`)
//! is rejected here; it is phase-3 work.

use crate::error::TranslateError;
use crate::parse::{Instr, LlvmFunc, Operand, Ty};
use std::collections::HashMap;

/// Lower `func` (named `cs_main`) to a Naga module with one compute entry.
///
/// Two entry shapes are supported:
/// - `cs_main(i32) -> float`: result is stored to `storage_0_0[id]`.
/// - `cs_main(i32) -> void`: the body stores through ABI calls.
pub(super) fn lower(func: &LlvmFunc) -> Result<naga::Module, TranslateError> {
    if func.name != "cs_main" {
        return Err(TranslateError::Unsupported);
    }
    if !matches!(func.ret_ty, Ty::Float | Ty::Void) {
        return Err(TranslateError::Unsupported);
    }
    if func.params.len() == 1 && func.params[0].0 == Ty::I32 {
        // `cs_main(i32)` — standard.
    } else if func.params.is_empty() {
        // `cs_main()` — allowed for kernels that read `global_invocation_id`
        // themselves; they simply do not use `%id`. Accept and ignore.
    } else {
        return Err(TranslateError::Unsupported);
    }
    let mut module = Builders::new();
    let function = module.build(func)?;
    module.finish(function, func.ret_ty)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Elem {
    F32,
    U32,
    AtomicU32,
}

struct Builders {
    module: naga::Module,
    storage_cache: HashMap<(u32, u32, Elem), naga::Handle<naga::GlobalVariable>>,
}

impl Builders {
    fn new() -> Self {
        Self {
            module: naga::Module::default(),
            storage_cache: HashMap::new(),
        }
    }

    fn finish(
        mut self,
        function: naga::Function,
        ret_ty: Ty,
    ) -> Result<naga::Module, TranslateError> {
        // Only create the implicit `storage_0_0` global when the kernel
        // uses the `-> float` shape; for `-> void` the body creates its
        // own globals lazily.
        if ret_ty == Ty::Float {
            self.storage_global(0, 0, Elem::F32);
        }
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

    fn storage_global(
        &mut self,
        group: u32,
        binding: u32,
        elem: Elem,
    ) -> naga::Handle<naga::GlobalVariable> {
        if let Some(handle) = self.storage_cache.get(&(group, binding, elem)) {
            return *handle;
        }
        let base_ty = match elem {
            Elem::F32 => scalar(&mut self.module, naga::ScalarKind::Float),
            Elem::U32 => scalar(&mut self.module, naga::ScalarKind::Uint),
            Elem::AtomicU32 => self.module.types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Atomic(naga::Scalar {
                        kind: naga::ScalarKind::Uint,
                        width: 4,
                    }),
                },
                naga::Span::default(),
            ),
        };
        let array_ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Array {
                    base: base_ty,
                    size: naga::ArraySize::Dynamic,
                    stride: 4,
                },
            },
            naga::Span::default(),
        );
        let name = match elem {
            Elem::AtomicU32 => format!("atomic_storage_{group}_{binding}_"),
            _ => format!("storage_{group}_{binding}_"),
        };
        let handle = self.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(name),
                space: naga::AddressSpace::Storage {
                    access: naga::StorageAccess::LOAD | naga::StorageAccess::STORE,
                },
                binding: Some(naga::ResourceBinding { group, binding }),
                ty: array_ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        );
        self.storage_cache.insert((group, binding, elem), handle);
        handle
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
        if let Some((_, name)) = func.params.first() {
            env.insert(name.clone(), id_x);
        }

        let mut ret_seen = false;
        for instr in &func.body {
            match instr {
                Instr::Ret { a } => {
                    let value = operand(&mut function, a, &env)?;
                    let global = self.storage_global(0, 0, Elem::F32);
                    let gvar = append(&mut function, naga::Expression::GlobalVariable(global));
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
                    ret_seen = true;
                }
                Instr::RetVoid => {
                    function.body.push(
                        naga::Statement::Return { value: None },
                        naga::Span::default(),
                    );
                    ret_seen = true;
                }
                Instr::CallVoid { callee, args } => {
                    self.call_void(&mut function, callee, args, &env)?;
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
                    let v = self.call_value(&mut function, callee, args, &env)?;
                    env.insert(dst.clone(), v);
                }
            }
        }
        if !ret_seen {
            return Err(TranslateError::Parse("function without ret".to_string()));
        }
        Ok(function)
    }

    /// Value-returning call: math or storage read / atomic add.
    fn call_value(
        &mut self,
        func: &mut naga::Function,
        callee: &str,
        args: &[Operand],
        env: &HashMap<String, naga::Handle<naga::Expression>>,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        // Storage reads.
        if callee.contains("storage_read_f32") {
            let (group, binding, index) = storage_args(func, args, env, 3)?;
            let global = self.storage_global(group, binding, Elem::F32);
            return load_global(func, global, index);
        }
        if callee.contains("storage_read_u32") {
            let (group, binding, index) = storage_args(func, args, env, 3)?;
            let global = self.storage_global(group, binding, Elem::U32);
            return load_global(func, global, index);
        }
        // Atomic add.
        if callee.contains("storage_atomic_add_u32") {
            if args.len() != 4 {
                return Err(TranslateError::Unsupported);
            }
            let group = const_u32(func, &args[0], env)?;
            let binding = const_u32(func, &args[1], env)?;
            let index = operand(func, &args[2], env)?;
            let value = operand(func, &args[3], env)?;
            let global = self.storage_global(group, binding, Elem::AtomicU32);
            let base = append(func, naga::Expression::GlobalVariable(global));
            push_emit(func, base);
            let pointer = append(func, naga::Expression::Access { base, index });
            push_emit(func, pointer);
            let ty = scalar(&mut self.module, naga::ScalarKind::Uint);
            let result = func.expressions.append(
                naga::Expression::AtomicResult {
                    ty,
                    comparison: false,
                },
                naga::Span::default(),
            );
            func.body.push(
                naga::Statement::Atomic {
                    pointer,
                    fun: naga::AtomicFunction::Add,
                    value,
                    result: Some(result),
                },
                naga::Span::default(),
            );
            return Ok(result);
        }
        // Math intrinsics.
        if args.len() == 1 {
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
            return Ok(v);
        }
        Err(TranslateError::Unsupported)
    }

    /// Void call: storage writes (and barrier stubs).
    fn call_void(
        &mut self,
        func: &mut naga::Function,
        callee: &str,
        args: &[Operand],
        env: &HashMap<String, naga::Handle<naga::Expression>>,
    ) -> Result<(), TranslateError> {
        if callee.contains("storage_write_f32") {
            store_void(self, func, callee, args, env, Elem::F32)?;
            return Ok(());
        }
        if callee.contains("storage_write_u32") {
            store_void(self, func, callee, args, env, Elem::U32)?;
            return Ok(());
        }
        Err(TranslateError::Unsupported)
    }
}

fn store_void(
    builders: &mut Builders,
    func: &mut naga::Function,
    callee: &str,
    args: &[Operand],
    env: &HashMap<String, naga::Handle<naga::Expression>>,
    elem: Elem,
) -> Result<(), TranslateError> {
    let _ = callee;
    if args.len() != 4 {
        return Err(TranslateError::Unsupported);
    }
    let group = const_u32(func, &args[0], env)?;
    let binding = const_u32(func, &args[1], env)?;
    let index = operand(func, &args[2], env)?;
    let value = operand(func, &args[3], env)?;
    let global = builders.storage_global(group, binding, elem);
    let base = append(func, naga::Expression::GlobalVariable(global));
    push_emit(func, base);
    let pointer = append(func, naga::Expression::Access { base, index });
    push_emit(func, pointer);
    func.body.push(
        naga::Statement::Store { pointer, value },
        naga::Span::default(),
    );
    Ok(())
}

fn load_global(
    func: &mut naga::Function,
    global: naga::Handle<naga::GlobalVariable>,
    index: naga::Handle<naga::Expression>,
) -> Result<naga::Handle<naga::Expression>, TranslateError> {
    let base = append(func, naga::Expression::GlobalVariable(global));
    push_emit(func, base);
    let pointer = append(func, naga::Expression::Access { base, index });
    push_emit(func, pointer);
    let v = append(func, naga::Expression::Load { pointer });
    push_emit(func, v);
    Ok(v)
}

fn const_u32(
    func: &mut naga::Function,
    op: &Operand,
    env: &HashMap<String, naga::Handle<naga::Expression>>,
) -> Result<u32, TranslateError> {
    match op {
        Operand::U32(v) => Ok(*v),
        Operand::Local(name) => {
            // A local `i32` constant is not materialised; group/binding
            // must be immediates (same rule as `wasm2gpu`).
            let _ = func;
            let _ = env;
            let _ = name;
            Err(TranslateError::Unsupported)
        }
        Operand::F32(_) | Operand::Bool(_) => Err(TranslateError::Unsupported),
    }
}

fn storage_args(
    func: &mut naga::Function,
    args: &[Operand],
    env: &HashMap<String, naga::Handle<naga::Expression>>,
    n: usize,
) -> Result<(u32, u32, naga::Handle<naga::Expression>), TranslateError> {
    if args.len() != n {
        return Err(TranslateError::Unsupported);
    }
    let group = const_u32(func, &args[0], env)?;
    let binding = const_u32(func, &args[1], env)?;
    let index = operand(func, &args[2], env)?;
    Ok((group, binding, index))
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
