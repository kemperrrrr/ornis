//! Lower a parsed LLVM function to a Naga compute entry.
//!
//! The supported subset is reducible control flow after rustc's `SROA`:
//! straight-line scalar arithmetic, `select`, diamond `if`s with `phi`
//! merges, and a single natural loop per region. Phi values are destroyed
//! into local variables on each incoming edge, which lets Naga's structured
//! `If`/`Loop` carry the same values without a general SSA relooper.
//!
//! Storage and uniform accesses go through the `ornis_gpu` ABI exactly as
//! `wasm2gpu` declares them. A scalar `-> float` result is written to
//! `storage_0_0[id]`, mirroring the toy `compute` kernel; `-> void` kernels
//! store through the ABI themselves. Irreducible control flow, nested loops,
//! and loops without an exit are rejected as `Unsupported` so an unknown
//! shape never silently becomes a wrong shader.

use crate::error::TranslateError;
use crate::parse::{BasicBlock, Instr, LlvmFunc, Operand, Terminator, Ty};
use std::collections::{HashMap, HashSet};

/// Synthetic post-dominator sink used while structuring the CFG.
const EXIT: &str = "$exit";

/// Lower `func` (named `cs_main`) to a Naga module with one compute entry.
///
/// Two entry shapes are supported:
/// - `cs_main(i32) -> float`: result is stored to `storage_0_0[id]`.
/// - `cs_main(i32) -> void`: the body stores through ABI calls.
pub(super) fn lower(func: &LlvmFunc, structs: &[&[u8]]) -> Result<naga::Module, TranslateError> {
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
    let cfg = Cfg::new(func)?;
    let mut builders = Builders::new(structs);
    let function = builders.build(func, &cfg)?;
    builders.finish(function, func.ret_ty)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Elem {
    F32,
    U32,
    AtomicU32,
}

/// One `phi` result and the `(value, predecessor)` pairs feeding it.
struct Phi {
    dst: String,
    ty: Ty,
    incoming: Vec<(Operand, String)>,
}

/// Read-only views of a parsed function used by the structurer.
struct Cfg<'a> {
    blocks: HashMap<&'a str, &'a BasicBlock>,
    order: Vec<&'a str>,
    phis: HashMap<&'a str, Vec<Phi>>,
    succs: HashMap<&'a str, Vec<&'a str>>,
    postdoms: HashMap<&'a str, HashSet<&'a str>>,
    /// Loop header -> blocks of its natural loop.
    loops: HashMap<&'a str, HashSet<&'a str>>,
}

impl<'a> Cfg<'a> {
    fn new(func: &'a LlvmFunc) -> Result<Self, TranslateError> {
        let mut blocks = HashMap::new();
        let mut order = Vec::new();
        for block in &func.blocks {
            if blocks.insert(block.label.as_str(), block).is_some() {
                return Err(TranslateError::Parse(format!(
                    "duplicate block {}",
                    block.label
                )));
            }
            order.push(block.label.as_str());
        }
        let entry = *order.first().ok_or(TranslateError::Unsupported)?;
        let mut succs: HashMap<&'a str, Vec<&'a str>> = HashMap::new();
        let mut preds: HashMap<&'a str, Vec<&'a str>> = HashMap::new();
        let mut phis: HashMap<&'a str, Vec<Phi>> = HashMap::new();
        for block in &func.blocks {
            let mut out = Vec::new();
            match &block.term {
                Terminator::Ret(_) => {}
                Terminator::BrUncond(target) => out.push(target.as_str()),
                Terminator::BrCond {
                    true_label,
                    false_label,
                    ..
                } => {
                    out.push(true_label.as_str());
                    out.push(false_label.as_str());
                }
            }
            for target in &out {
                if !blocks.contains_key(target) {
                    return Err(TranslateError::Parse(format!(
                        "branch to unknown block {target}"
                    )));
                }
                preds.entry(target).or_default().push(block.label.as_str());
            }
            succs.insert(block.label.as_str(), out);
            let mut block_phis = Vec::new();
            for instr in &block.instrs {
                if let Instr::Phi { dst, ty, incoming } = instr {
                    block_phis.push(Phi {
                        dst: dst.clone(),
                        ty: *ty,
                        incoming: incoming.clone(),
                    });
                }
            }
            if !block_phis.is_empty() {
                phis.insert(block.label.as_str(), block_phis);
            }
        }

        let reachable = reachable(&succs, entry);
        let doms = dominators(&reachable, &preds, entry);
        let postdoms = postdominators(&reachable, &succs);
        let mut loops: HashMap<&'a str, HashSet<&'a str>> = HashMap::new();
        for (from, targets) in &succs {
            if !reachable.contains(from) {
                continue;
            }
            for header in targets {
                if doms.get(from).is_some_and(|dom| dom.contains(*header)) {
                    let body = natural_loop(header, from, &preds);
                    loops.entry(*header).or_default().extend(body);
                }
            }
        }
        Ok(Self {
            blocks,
            order,
            phis,
            succs,
            postdoms,
            loops,
        })
    }

    fn entry(&self) -> &'a str {
        self.order[0]
    }

    /// Closest common post-dominator of a conditional branch, or `None`
    /// when the branch has no common successor inside the function.
    fn merge(&self, label: &str) -> Option<&'a str> {
        let set = self.postdoms.get(label)?;
        let mut best: Option<&'a str> = None;
        for candidate in set {
            if *candidate == label || *candidate == EXIT {
                continue;
            }
            let size = self
                .postdoms
                .get(candidate)
                .map_or(usize::MAX, HashSet::len);
            let best_size = best
                .and_then(|b| self.postdoms.get(b))
                .map_or(usize::MAX, HashSet::len);
            if best.is_none() || size > best_size {
                best = Some(candidate);
            }
        }
        if best.is_none() && set.contains(EXIT) {
            return Some(EXIT);
        }
        best
    }
}

fn reachable<'a>(succs: &HashMap<&'a str, Vec<&'a str>>, entry: &'a str) -> HashSet<&'a str> {
    let mut seen = HashSet::new();
    let mut stack = vec![entry];
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        if let Some(next) = succs.get(node) {
            stack.extend(next.iter().copied());
        }
    }
    seen
}

fn dominators<'a>(
    reachable: &HashSet<&'a str>,
    preds: &HashMap<&'a str, Vec<&'a str>>,
    entry: &'a str,
) -> HashMap<&'a str, HashSet<&'a str>> {
    let mut doms: HashMap<&'a str, HashSet<&'a str>> = HashMap::new();
    for node in reachable {
        if *node == entry {
            doms.insert(node, HashSet::from([entry]));
        } else {
            doms.insert(node, reachable.clone());
        }
    }
    loop {
        let mut changed = false;
        for node in reachable {
            if *node == entry {
                continue;
            }
            let mut next: Option<HashSet<&'a str>> = None;
            for pred in preds.get(node).into_iter().flatten() {
                if !reachable.contains(pred) {
                    continue;
                }
                let pred_doms = &doms[pred];
                next = Some(match next {
                    None => pred_doms.clone(),
                    Some(acc) => acc.intersection(pred_doms).copied().collect(),
                });
            }
            let mut next = next.unwrap_or_default();
            next.insert(node);
            if next != doms[node] {
                doms.insert(node, next);
                changed = true;
            }
        }
        if !changed {
            return doms;
        }
    }
}

fn postdominators<'a>(
    reachable: &HashSet<&'a str>,
    succs: &HashMap<&'a str, Vec<&'a str>>,
) -> HashMap<&'a str, HashSet<&'a str>> {
    let mut all = reachable.clone();
    all.insert(EXIT);
    let mut post: HashMap<&'a str, HashSet<&'a str>> = HashMap::new();
    for node in &all {
        if *node == EXIT {
            post.insert(node, HashSet::from([EXIT]));
        } else {
            post.insert(node, all.clone());
        }
    }
    loop {
        let mut changed = false;
        for node in reachable {
            let out: Vec<&'a str> = succs.get(node).into_iter().flatten().copied().collect();
            let mut next: Option<HashSet<&'a str>> = if out.is_empty() {
                Some(HashSet::from([EXIT]))
            } else {
                None
            };
            for target in out {
                let target_post = if target == EXIT {
                    HashSet::from([EXIT])
                } else {
                    post[target].clone()
                };
                next = Some(match next {
                    None => target_post,
                    Some(acc) => acc.intersection(&target_post).copied().collect(),
                });
            }
            let mut next = next.unwrap_or_default();
            next.insert(node);
            if next != post[node] {
                post.insert(node, next);
                changed = true;
            }
        }
        if !changed {
            return post;
        }
    }
}

/// Natural loop body for the back edge `latch -> header`.
fn natural_loop<'a>(
    header: &'a str,
    latch: &'a str,
    preds: &HashMap<&'a str, Vec<&'a str>>,
) -> HashSet<&'a str> {
    let mut body = HashSet::from([header]);
    let mut stack = vec![latch];
    while let Some(node) = stack.pop() {
        if node == header || !body.insert(node) {
            continue;
        }
        if let Some(inputs) = preds.get(node) {
            stack.extend(inputs.iter().copied());
        }
    }
    body
}

struct Builders {
    module: naga::Module,
    storage_cache: HashMap<(u32, u32, Elem), naga::Handle<naga::GlobalVariable>>,
    uniform_cache: HashMap<(u32, u32), naga::Handle<naga::GlobalVariable>>,
    uniform_struct_cache: HashMap<(u32, u32, u32), naga::Handle<naga::GlobalVariable>>,
    /// `ornis_gpu.types` field tags per struct id.
    structs: Vec<Vec<u8>>,
}

impl Builders {
    fn new(structs: &[&[u8]]) -> Self {
        Self {
            module: naga::Module::default(),
            storage_cache: HashMap::new(),
            uniform_cache: HashMap::new(),
            uniform_struct_cache: HashMap::new(),
            structs: structs.iter().map(|tags| tags.to_vec()).collect(),
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

    /// Uniform `array<scalar, len>` wrapped as `{ a: array<lane, len> }`.
    fn uniform_global(
        &mut self,
        group: u32,
        binding: u32,
        len: u32,
    ) -> naga::Handle<naga::GlobalVariable> {
        if let Some(handle) = self.uniform_cache.get(&(group, binding)) {
            return *handle;
        }
        let scalar_ty = scalar(&mut self.module, naga::ScalarKind::Float);
        let lane_ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Struct {
                    members: vec![
                        naga::StructMember {
                            name: Some("v".to_string()),
                            ty: scalar_ty,
                            binding: None,
                            offset: 0,
                        },
                        naga::StructMember {
                            name: Some("p1".to_string()),
                            ty: scalar_ty,
                            binding: None,
                            offset: 4,
                        },
                        naga::StructMember {
                            name: Some("p2".to_string()),
                            ty: scalar_ty,
                            binding: None,
                            offset: 8,
                        },
                        naga::StructMember {
                            name: Some("p3".to_string()),
                            ty: scalar_ty,
                            binding: None,
                            offset: 12,
                        },
                    ],
                    span: 16,
                },
            },
            naga::Span::default(),
        );
        let array_ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Array {
                    base: lane_ty,
                    size: naga::ArraySize::Constant(
                        std::num::NonZeroU32::new(len)
                            .unwrap_or(std::num::NonZeroU32::new(1).unwrap()),
                    ),
                    stride: 16,
                },
            },
            naga::Span::default(),
        );
        let outer = self.module.types.insert(
            naga::Type {
                name: Some(format!("U{len}")),
                inner: naga::TypeInner::Struct {
                    members: vec![naga::StructMember {
                        name: Some("a".to_string()),
                        ty: array_ty,
                        binding: None,
                        offset: 0,
                    }],
                    span: 16 * len,
                },
            },
            naga::Span::default(),
        );
        let handle = self.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(format!("uniform_{group}_{binding}_")),
                space: naga::AddressSpace::Uniform,
                binding: Some(naga::ResourceBinding { group, binding }),
                ty: outer,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        );
        self.uniform_cache.insert((group, binding), handle);
        handle
    }

    /// Uniform struct `SID` at `(group, binding)`, laid out with
    /// `place_fields` in uniform mode.
    fn uniform_struct_global(
        &mut self,
        group: u32,
        binding: u32,
        sid: u32,
    ) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
        if let Some(handle) = self.uniform_struct_cache.get(&(group, binding, sid)) {
            return Ok(*handle);
        }
        let tags = self
            .structs
            .get(sid as usize)
            .ok_or_else(|| {
                TranslateError::UnsupportedType(format!(
                    "uniform struct id {sid} has no ornis_gpu.types entry"
                ))
            })?
            .clone();
        if tags.is_empty() {
            return Err(TranslateError::UnsupportedType(
                "uniform struct has no fields".to_string(),
            ));
        }
        let mut members = Vec::with_capacity(tags.len());
        let mut specs = Vec::with_capacity(tags.len());
        let mut offsets = vec![0u32; tags.len()];
        for (index, tag) in tags.iter().enumerate() {
            let (ty, size, align) = self.tag_type(*tag)?;
            specs.push(ornis_gpu_abi::FieldLayout { size, align });
            members.push(naga::StructMember {
                name: Some(format!("f{index}")),
                ty,
                binding: None,
                offset: 0,
            });
        }
        let span = ornis_gpu_abi::place_fields(&specs, true, &mut offsets);
        for (member, offset) in members.iter_mut().zip(offsets) {
            member.offset = offset;
        }
        let struct_ty = self.module.types.insert(
            naga::Type {
                name: Some(format!("UniformStruct{sid}")),
                inner: naga::TypeInner::Struct { members, span },
            },
            naga::Span::default(),
        );
        let handle = self.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(format!("uniform_struct_{group}_{binding}_")),
                space: naga::AddressSpace::Uniform,
                binding: Some(naga::ResourceBinding { group, binding }),
                ty: struct_ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        );
        self.uniform_struct_cache
            .insert((group, binding, sid), handle);
        Ok(handle)
    }

    fn tag_type(
        &mut self,
        tag: u8,
    ) -> Result<(naga::Handle<naga::Type>, u32, u32), TranslateError> {
        use ornis_gpu_abi::{
            F32_ALIGN, F32_SIZE, TYPE_TAG_F32, TYPE_TAG_I32, TYPE_TAG_U32, TYPE_TAG_VEC2,
            TYPE_TAG_VEC3, TYPE_TAG_VEC4, VEC2_ALIGN, VEC2_SIZE, VEC3_ALIGN, VEC3_SIZE, VEC4_ALIGN,
            VEC4_SIZE,
        };
        let (kind, lanes, size, align) = match tag {
            TYPE_TAG_F32 => (naga::ScalarKind::Float, 1, F32_SIZE, F32_ALIGN),
            TYPE_TAG_U32 => (naga::ScalarKind::Uint, 1, F32_SIZE, F32_ALIGN),
            TYPE_TAG_I32 => (naga::ScalarKind::Sint, 1, F32_SIZE, F32_ALIGN),
            TYPE_TAG_VEC2 => (naga::ScalarKind::Float, 2, VEC2_SIZE, VEC2_ALIGN),
            TYPE_TAG_VEC3 => (naga::ScalarKind::Float, 3, VEC3_SIZE, VEC3_ALIGN),
            TYPE_TAG_VEC4 => (naga::ScalarKind::Float, 4, VEC4_SIZE, VEC4_ALIGN),
            _ => {
                return Err(TranslateError::UnsupportedType(format!(
                    "unsupported uniform field tag {tag}"
                )));
            }
        };
        let inner = if lanes == 1 {
            naga::TypeInner::Scalar(naga::Scalar { kind, width: 4 })
        } else {
            let size = match lanes {
                2 => naga::VectorSize::Bi,
                3 => naga::VectorSize::Tri,
                _ => naga::VectorSize::Quad,
            };
            naga::TypeInner::Vector {
                size,
                scalar: naga::Scalar { kind, width: 4 },
            }
        };
        let ty = self
            .module
            .types
            .insert(naga::Type { name: None, inner }, naga::Span::default());
        Ok((ty, size, align))
    }

    fn build(&mut self, func: &LlvmFunc, cfg: &Cfg<'_>) -> Result<naga::Function, TranslateError> {
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
        let arg = append(&mut function, naga::Expression::FunctionArgument(0));
        let id_x = append(
            &mut function,
            naga::Expression::AccessIndex {
                base: arg,
                index: 0,
            },
        );
        let mut body = naga::Block::new();
        push_emit(&mut function, &mut body, id_x);
        function.body = body;

        let mut lowerer = Lowerer {
            builders: self,
            cfg,
            func: function,
            id_x,
            phi_vars: HashMap::new(),
            env: HashMap::new(),
            saw_return: false,
        };
        if let Some((_, name)) = func.params.first() {
            lowerer.env.insert(name.clone(), id_x);
        }
        lowerer.alloc_phis()?;
        lowerer.emit_top()?;
        let saw_return = lowerer.saw_return;
        let function = lowerer.func;
        if !saw_return {
            return Err(TranslateError::Parse(
                "function without lowered ret".to_string(),
            ));
        }
        Ok(function)
    }
}

/// Where a region is emitted: top level or inside one natural loop.
#[derive(Clone, Copy)]
enum Ctx<'a> {
    Plain,
    Loop { header: &'a str },
}

/// How control leaves an emitted region.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Flow {
    /// Reached the requested stop label or fell through normally.
    Next,
    /// Emitted `return`.
    Returned,
    /// Emitted `continue` for the innermost loop.
    Continue,
    /// Emitted `break` for the innermost loop.
    Break,
}

type Env = HashMap<String, naga::Handle<naga::Expression>>;

struct Lowerer<'a, 'b> {
    builders: &'b mut Builders,
    cfg: &'b Cfg<'a>,
    func: naga::Function,
    id_x: naga::Handle<naga::Expression>,
    phi_vars: HashMap<String, naga::Handle<naga::LocalVariable>>,
    env: Env,
    saw_return: bool,
}

impl<'a, 'b> Lowerer<'a, 'b> {
    fn emit_top(&mut self) -> Result<(), TranslateError> {
        let mut body = std::mem::take(&mut self.func.body);
        let flow = self.emit_region(self.cfg.entry(), None, &mut body, Ctx::Plain)?;
        self.func.body = body;
        match flow {
            Flow::Next | Flow::Returned => Ok(()),
            Flow::Continue | Flow::Break => Err(TranslateError::Unsupported),
        }
    }

    fn alloc_phis(&mut self) -> Result<(), TranslateError> {
        for phis in self.cfg.phis.values() {
            for phi in phis {
                let ty = self.ty(phi.ty)?;
                let var = self.func.local_variables.append(
                    naga::LocalVariable {
                        name: Some(phi.dst.clone()),
                        ty,
                        init: None,
                    },
                    naga::Span::default(),
                );
                self.phi_vars.insert(phi.dst.clone(), var);
            }
        }
        Ok(())
    }

    fn ty(&mut self, ty: Ty) -> Result<naga::Handle<naga::Type>, TranslateError> {
        let kind = match ty {
            Ty::Float => naga::ScalarKind::Float,
            Ty::I32 => naga::ScalarKind::Uint,
            Ty::I1 => naga::ScalarKind::Bool,
            _ => return Err(TranslateError::Unsupported),
        };
        Ok(scalar(&mut self.builders.module, kind))
    }

    /// Emit `start` until `stop` is reached or control leaves the region.
    fn emit_region(
        &mut self,
        start: &'a str,
        stop: Option<&'a str>,
        out: &mut naga::Block,
        ctx: Ctx<'a>,
    ) -> Result<Flow, TranslateError> {
        if stop == Some(start) {
            return Ok(Flow::Next);
        }
        match ctx {
            Ctx::Plain => {
                if self.cfg.loops.contains_key(start) {
                    return self.emit_loop(start, stop, out);
                }
            }
            Ctx::Loop { header, .. } => {
                if start != header && self.cfg.loops.contains_key(start) {
                    // Nested loops need multi-level break/continue recovery;
                    // reject rather than emit a wrong nesting.
                    return Err(TranslateError::Unsupported);
                }
            }
        }
        let block = *self
            .cfg
            .blocks
            .get(start)
            .ok_or(TranslateError::Unsupported)?;
        self.emit_block(block, out)?;
        match &block.term {
            Terminator::Ret(value) => self.emit_ret(value.as_ref(), out),
            Terminator::BrUncond(target) => {
                let target = target.as_str();
                self.store_edge(start, target, out)?;
                if let Ctx::Loop { header } = ctx {
                    if target == header {
                        out.push(naga::Statement::Continue, naga::Span::default());
                        return Ok(Flow::Continue);
                    }
                    let leaves_loop = self
                        .cfg
                        .loops
                        .get(header)
                        .is_none_or(|body| !body.contains(target));
                    if leaves_loop {
                        out.push(naga::Statement::Break, naga::Span::default());
                        return Ok(Flow::Break);
                    }
                }
                if stop == Some(target) {
                    return Ok(Flow::Next);
                }
                self.emit_region(target, stop, out, ctx)
            }
            Terminator::BrCond {
                cond,
                true_label,
                false_label,
            } => {
                if let Ctx::Loop { header } = ctx {
                    let (true_in, false_in) =
                        self.cfg.loops.get(header).map_or((false, false), |body| {
                            (
                                body.contains(true_label.as_str()),
                                body.contains(false_label.as_str()),
                            )
                        });
                    if true_in != false_in {
                        return self.emit_loop_exit_branch(
                            header,
                            start,
                            cond,
                            true_label,
                            false_label,
                            true_in,
                            out,
                        );
                    }
                    if !true_in && !false_in {
                        return self.emit_loop_double_exit(
                            start,
                            cond,
                            true_label,
                            false_label,
                            out,
                        );
                    }
                }
                self.emit_if(start, cond, true_label, false_label, stop, out, ctx)
            }
        }
    }

    /// Diamond `if`: both arms run until the closest common post-dominator.
    #[allow(clippy::too_many_arguments)]
    fn emit_if(
        &mut self,
        label: &'a str,
        cond: &Operand,
        true_label: &'a str,
        false_label: &'a str,
        stop: Option<&'a str>,
        out: &mut naga::Block,
        ctx: Ctx<'a>,
    ) -> Result<Flow, TranslateError> {
        let merge = self.cfg.merge(label).ok_or(TranslateError::Unsupported)?;
        if let Ctx::Loop { header } = ctx {
            let merges_in_loop = self
                .cfg
                .loops
                .get(header)
                .is_some_and(|body| body.contains(merge));
            if merge != EXIT && !merges_in_loop {
                // An arm that leaves the loop through a shared merge would
                // need `break` with per-arm edge phis; not supported yet.
                return Err(TranslateError::Unsupported);
            }
        }
        let condition = self.operand(out, cond)?;
        let outer_env = self.env.clone();
        let mut accept = naga::Block::new();
        self.env = outer_env.clone();
        self.store_edge_into(label, true_label, &outer_env, &mut accept)?;
        let accept_flow = self.emit_region(true_label, Some(merge), &mut accept, ctx)?;
        let mut reject = naga::Block::new();
        self.env = outer_env.clone();
        self.store_edge_into(label, false_label, &outer_env, &mut reject)?;
        let reject_flow = self.emit_region(false_label, Some(merge), &mut reject, ctx)?;
        self.env = outer_env;
        if accept_flow != Flow::Next && accept_flow != Flow::Returned {
            return Err(TranslateError::Unsupported);
        }
        if reject_flow != Flow::Next && reject_flow != Flow::Returned {
            return Err(TranslateError::Unsupported);
        }
        out.push(
            naga::Statement::If {
                condition,
                accept,
                reject,
            },
            naga::Span::default(),
        );
        if merge == EXIT {
            if accept_flow == Flow::Returned && reject_flow == Flow::Returned {
                return Ok(Flow::Returned);
            }
            return Err(TranslateError::Unsupported);
        }
        if stop == Some(merge) {
            return Ok(Flow::Next);
        }
        self.emit_region(merge, stop, out, ctx)
    }

    /// Loop-header or latch branch where exactly one successor stays inside
    /// the loop and the other leaves it.
    #[allow(clippy::too_many_arguments)]
    fn emit_loop_exit_branch(
        &mut self,
        header: &'a str,
        start: &'a str,
        cond: &Operand,
        true_label: &'a str,
        false_label: &'a str,
        true_in: bool,
        out: &mut naga::Block,
    ) -> Result<Flow, TranslateError> {
        let (inside, outside) = if true_in {
            (true_label, false_label)
        } else {
            (false_label, true_label)
        };
        let condition = self.operand(out, cond)?;
        let outer_env = self.env.clone();
        let mut inside_block = naga::Block::new();
        self.env = outer_env.clone();
        self.store_edge_into(start, inside, &outer_env, &mut inside_block)?;
        let inside_flow = if inside == header {
            // A branch back to the loop header means the loop repeats.
            inside_block.push(naga::Statement::Continue, naga::Span::default());
            Flow::Continue
        } else {
            let ctx = Ctx::Loop { header };
            self.emit_region(inside, None, &mut inside_block, ctx)?
        };
        if !matches!(inside_flow, Flow::Continue | Flow::Break | Flow::Returned) {
            return Err(TranslateError::Unsupported);
        }
        let mut outside_block = naga::Block::new();
        self.env = outer_env.clone();
        self.store_edge_into(start, outside, &outer_env, &mut outside_block)?;
        outside_block.push(naga::Statement::Break, naga::Span::default());
        self.env = outer_env;
        let (accept, reject) = if true_in {
            (inside_block, outside_block)
        } else {
            (outside_block, inside_block)
        };
        out.push(
            naga::Statement::If {
                condition,
                accept,
                reject,
            },
            naga::Span::default(),
        );
        Ok(Flow::Next)
    }

    /// Both branch targets leave the loop; each arm breaks.
    fn emit_loop_double_exit(
        &mut self,
        start: &'a str,
        cond: &Operand,
        true_label: &str,
        false_label: &str,
        out: &mut naga::Block,
    ) -> Result<Flow, TranslateError> {
        let condition = self.operand(out, cond)?;
        let outer_env = self.env.clone();
        let mut accept = naga::Block::new();
        self.env = outer_env.clone();
        self.store_edge_into(start, true_label, &outer_env, &mut accept)?;
        accept.push(naga::Statement::Break, naga::Span::default());
        let mut reject = naga::Block::new();
        self.env = outer_env.clone();
        self.store_edge_into(start, false_label, &outer_env, &mut reject)?;
        reject.push(naga::Statement::Break, naga::Span::default());
        self.env = outer_env;
        out.push(
            naga::Statement::If {
                condition,
                accept,
                reject,
            },
            naga::Span::default(),
        );
        Ok(Flow::Next)
    }

    /// `naga::Loop` for a natural loop with one header and one exit target.
    fn emit_loop(
        &mut self,
        header: &'a str,
        stop: Option<&'a str>,
        out: &mut naga::Block,
    ) -> Result<Flow, TranslateError> {
        let body = self
            .cfg
            .loops
            .get(header)
            .ok_or(TranslateError::Unsupported)?
            .clone();
        let ctx = Ctx::Loop { header };
        let outer_env = self.env.clone();
        let mut loop_body = naga::Block::new();
        let flow = self.emit_region(header, None, &mut loop_body, ctx)?;
        // Expressions created inside the loop body are out of scope after
        // `Loop`; the exit block reloads whatever its phis need.
        self.env = outer_env;
        if flow == Flow::Continue {
            return Err(TranslateError::Unsupported);
        }
        let exits: HashSet<&'a str> = body
            .iter()
            .flat_map(|block| self.cfg.succs.get(block).into_iter().flatten())
            .filter(|target| !body.contains(*target))
            .copied()
            .collect();
        if exits.is_empty() {
            if flow == Flow::Returned {
                out.push(
                    naga::Statement::Loop {
                        body: loop_body,
                        continuing: naga::Block::new(),
                        break_if: None,
                    },
                    naga::Span::default(),
                );
                return Ok(Flow::Returned);
            }
            return Err(TranslateError::Unsupported);
        }
        if exits.len() != 1 {
            return Err(TranslateError::Unsupported);
        }
        let exit = *exits.iter().next().ok_or(TranslateError::Unsupported)?;
        out.push(
            naga::Statement::Loop {
                body: loop_body,
                continuing: naga::Block::new(),
                break_if: None,
            },
            naga::Span::default(),
        );
        if flow == Flow::Returned {
            return Ok(Flow::Returned);
        }
        if stop == Some(exit) {
            return Ok(Flow::Next);
        }
        self.emit_region(exit, stop, out, Ctx::Plain)
    }

    fn emit_block(
        &mut self,
        block: &BasicBlock,
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        for instr in &block.instrs {
            match instr {
                Instr::Phi { dst, .. } => {
                    let var = *self.phi_vars.get(dst).ok_or(TranslateError::Unsupported)?;
                    let ptr = self.expr(out, naga::Expression::LocalVariable(var));
                    let value = self.expr(out, naga::Expression::Load { pointer: ptr });
                    self.env.insert(dst.clone(), value);
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

    fn emit_ret(
        &mut self,
        value: Option<&Operand>,
        out: &mut naga::Block,
    ) -> Result<Flow, TranslateError> {
        if let Some(op) = value {
            let value = self.operand(out, op)?;
            let global = self.builders.storage_global(0, 0, Elem::F32);
            let base = self.expr(out, naga::Expression::GlobalVariable(global));
            let pointer = self.expr(
                out,
                naga::Expression::Access {
                    base,
                    index: self.id_x,
                },
            );
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
    fn store_edge(
        &mut self,
        from: &str,
        to: &str,
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        let env = self.env.clone();
        self.store_edge_into(from, to, &env, out)
    }

    fn store_edge_into(
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
                .ok_or(TranslateError::Unsupported)?;
            let value = operand(&mut self.func, out, incoming, env)?;
            let var = *self
                .phi_vars
                .get(&phi.dst)
                .ok_or(TranslateError::Unsupported)?;
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

    fn emit_instr(&mut self, instr: &Instr, out: &mut naga::Block) -> Result<(), TranslateError> {
        match instr {
            Instr::FAdd { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Add, a, b, out)?
            }
            Instr::FMul { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Multiply, a, b, out)?
            }
            Instr::FSub { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Subtract, a, b, out)?
            }
            Instr::FDiv { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Divide, a, b, out)?
            }
            Instr::Add { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Add, a, b, out)?
            }
            Instr::Sub { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Subtract, a, b, out)?
            }
            Instr::Mul { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Multiply, a, b, out)?
            }
            Instr::UDiv { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Divide, a, b, out)?
            }
            Instr::SDiv { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Divide, a, b, out)?
            }
            Instr::URem { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Modulo, a, b, out)?
            }
            Instr::SRem { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Modulo, a, b, out)?
            }
            Instr::And { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::And, a, b, out)?
            }
            Instr::Or { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::InclusiveOr, a, b, out)?
            }
            Instr::Xor { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::ExclusiveOr, a, b, out)?
            }
            Instr::Shl { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::ShiftLeft, a, b, out)?
            }
            Instr::LShr { dst, a, b } | Instr::AShr { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::ShiftRight, a, b, out)?
            }
            Instr::FNeg { dst, a } => {
                let src = self.operand(out, a)?;
                let v = self.expr(
                    out,
                    naga::Expression::Unary {
                        op: naga::UnaryOperator::Negate,
                        expr: src,
                    },
                );
                self.env.insert(dst.clone(), v);
            }
            Instr::Fcmp { dst, pred, a, b } | Instr::Icmp { dst, pred, a, b } => {
                let v = self.compare(out, pred, a, b)?;
                self.env.insert(dst.clone(), v);
            }
            Instr::Select { dst, c, a, b } => {
                let condition = self.operand(out, c)?;
                let accept = self.operand(out, a)?;
                let reject = self.operand(out, b)?;
                let v = self.expr(
                    out,
                    naga::Expression::Select {
                        condition,
                        accept,
                        reject,
                    },
                );
                self.env.insert(dst.clone(), v);
            }
            Instr::Uitofp { dst, a } => {
                let src = self.operand(out, a)?;
                let v = self.expr(
                    out,
                    naga::Expression::As {
                        expr: src,
                        kind: naga::ScalarKind::Float,
                        convert: Some(4),
                    },
                );
                self.env.insert(dst.clone(), v);
            }
            Instr::Call { dst, callee, args } => {
                let v = self.call_value(out, callee, args)?;
                self.env.insert(dst.clone(), v);
            }
            Instr::CallVoid { callee, args } => self.call_void(out, callee, args)?,
            Instr::Phi { .. } => return Err(TranslateError::Unsupported),
            Instr::Ret { .. } | Instr::RetVoid => return Err(TranslateError::Unsupported),
        }
        Ok(())
    }

    fn assign_binary(
        &mut self,
        dst: &str,
        op: naga::BinaryOperator,
        a: &Operand,
        b: &Operand,
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        let left = self.operand(out, a)?;
        let right = self.operand(out, b)?;
        let v = self.expr(out, naga::Expression::Binary { op, left, right });
        self.env.insert(dst.to_string(), v);
        Ok(())
    }

    fn compare(
        &mut self,
        out: &mut naga::Block,
        pred: &str,
        a: &Operand,
        b: &Operand,
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
        let left = self.operand(out, a)?;
        let right = self.operand(out, b)?;
        Ok(self.expr(out, naga::Expression::Binary { op, left, right }))
    }

    /// Value-returning call: math or storage read / atomic add.
    fn call_value(
        &mut self,
        out: &mut naga::Block,
        callee: &str,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if callee.contains("uniform_field_f32") {
            return self.uniform_field(out, args, ornis_gpu_abi::TYPE_TAG_F32);
        }
        if callee.contains("uniform_field_u32") {
            return self.uniform_field(out, args, ornis_gpu_abi::TYPE_TAG_U32);
        }
        if callee.contains("uniform_read_f32") {
            if args.len() != 4 {
                return Err(TranslateError::Unsupported);
            }
            let group = const_u32(&args[0])?;
            let binding = const_u32(&args[1])?;
            let len = const_u32(&args[2])?;
            if len == 0 || len.checked_mul(16).is_none_or(|b| b > 65536) {
                return Err(TranslateError::UnsupportedType(
                    "uniform array exceeds 64KiB".to_string(),
                ));
            }
            let index = self.operand(out, &args[3])?;
            let global = self.builders.uniform_global(group, binding, len);
            return load_uniform(&mut self.func, out, global, index);
        }
        if callee.contains("storage_read_f32") {
            let (group, binding, index) = self.storage_args(out, args, 3)?;
            let global = self.builders.storage_global(group, binding, Elem::F32);
            return load_global(&mut self.func, out, global, index);
        }
        if callee.contains("storage_read_u32") {
            let (group, binding, index) = self.storage_args(out, args, 3)?;
            let global = self.builders.storage_global(group, binding, Elem::U32);
            return load_global(&mut self.func, out, global, index);
        }
        if callee.contains("storage_atomic_add_u32") {
            if args.len() != 4 {
                return Err(TranslateError::Unsupported);
            }
            let group = const_u32(&args[0])?;
            let binding = const_u32(&args[1])?;
            let index = self.operand(out, &args[2])?;
            let value = self.operand(out, &args[3])?;
            let global = self
                .builders
                .storage_global(group, binding, Elem::AtomicU32);
            let base = self.expr(out, naga::Expression::GlobalVariable(global));
            let pointer = self.expr(out, naga::Expression::Access { base, index });
            let ty = scalar(&mut self.builders.module, naga::ScalarKind::Uint);
            let result = self.func.expressions.append(
                naga::Expression::AtomicResult {
                    ty,
                    comparison: false,
                },
                naga::Span::default(),
            );
            out.push(
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
        if callee.contains("dot3") && args.len() == 6 {
            let vec_ty = self.vec3_type();
            let a0 = self.operand(out, &args[0])?;
            let a1 = self.operand(out, &args[1])?;
            let a2 = self.operand(out, &args[2])?;
            let b0 = self.operand(out, &args[3])?;
            let b1 = self.operand(out, &args[4])?;
            let b2 = self.operand(out, &args[5])?;
            let va = self.expr(
                out,
                naga::Expression::Compose {
                    ty: vec_ty,
                    components: vec![a0, a1, a2],
                },
            );
            let vb = self.expr(
                out,
                naga::Expression::Compose {
                    ty: vec_ty,
                    components: vec![b0, b1, b2],
                },
            );
            return Ok(self.expr(
                out,
                naga::Expression::Math {
                    fun: naga::MathFunction::Dot,
                    arg: va,
                    arg1: Some(vb),
                    arg2: None,
                    arg3: None,
                },
            ));
        }
        if callee.contains("length3") && args.len() == 3 {
            let vec_ty = self.vec3_type();
            let a0 = self.operand(out, &args[0])?;
            let a1 = self.operand(out, &args[1])?;
            let a2 = self.operand(out, &args[2])?;
            let va = self.expr(
                out,
                naga::Expression::Compose {
                    ty: vec_ty,
                    components: vec![a0, a1, a2],
                },
            );
            return Ok(self.expr(
                out,
                naga::Expression::Math {
                    fun: naga::MathFunction::Length,
                    arg: va,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                },
            ));
        }
        if args.len() == 1 {
            let arg = self.operand(out, &args[0])?;
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
            return Ok(self.expr(
                out,
                naga::Expression::Math {
                    fun,
                    arg,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                },
            ));
        }
        Err(TranslateError::Unsupported)
    }

    /// `uniform_field_f32/u32(group, binding, sid, field)`.
    fn uniform_field(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        expect: u8,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != 4 {
            return Err(TranslateError::Unsupported);
        }
        let group = const_u32(&args[0])?;
        let binding = const_u32(&args[1])?;
        let sid = const_u32(&args[2])?;
        let field = const_u32(&args[3])?;
        let tag = self
            .builders
            .structs
            .get(sid as usize)
            .and_then(|tags| tags.get(field as usize))
            .copied()
            .ok_or_else(|| {
                TranslateError::UnsupportedType(format!(
                    "uniform field {field} of struct {sid} is not declared"
                ))
            })?;
        if tag != expect {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform field {field} of struct {sid} has tag {tag}, expected {expect}"
            )));
        }
        let global = self.builders.uniform_struct_global(group, binding, sid)?;
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let pointer = self.expr(out, naga::Expression::AccessIndex { base, index: field });
        Ok(self.expr(out, naga::Expression::Load { pointer }))
    }

    fn call_void(
        &mut self,
        out: &mut naga::Block,
        callee: &str,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        if callee.contains("storage_write_f32") {
            return self.store_void(out, args, Elem::F32);
        }
        if callee.contains("storage_write_u32") {
            return self.store_void(out, args, Elem::U32);
        }
        if callee.contains("llvm.assume") || callee.contains("llvm.lifetime") {
            return Ok(());
        }
        Err(TranslateError::Unsupported)
    }

    fn store_void(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        elem: Elem,
    ) -> Result<(), TranslateError> {
        if args.len() != 4 {
            return Err(TranslateError::Unsupported);
        }
        let group = const_u32(&args[0])?;
        let binding = const_u32(&args[1])?;
        let index = self.operand(out, &args[2])?;
        let value = self.operand(out, &args[3])?;
        let global = self.builders.storage_global(group, binding, elem);
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let pointer = self.expr(out, naga::Expression::Access { base, index });
        out.push(
            naga::Statement::Store { pointer, value },
            naga::Span::default(),
        );
        Ok(())
    }

    fn storage_args(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        n: usize,
    ) -> Result<(u32, u32, naga::Handle<naga::Expression>), TranslateError> {
        if args.len() != n {
            return Err(TranslateError::Unsupported);
        }
        let group = const_u32(&args[0])?;
        let binding = const_u32(&args[1])?;
        let index = self.operand(out, &args[2])?;
        Ok((group, binding, index))
    }

    fn vec3_type(&mut self) -> naga::Handle<naga::Type> {
        self.builders.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Tri,
                    scalar: naga::Scalar {
                        kind: naga::ScalarKind::Float,
                        width: 4,
                    },
                },
            },
            naga::Span::default(),
        )
    }

    fn operand(
        &mut self,
        out: &mut naga::Block,
        op: &Operand,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let v = match op {
            Operand::Local(name) => self
                .env
                .get(name)
                .copied()
                .ok_or(TranslateError::Unsupported)?,
            Operand::F32(value) => {
                self.expr(out, naga::Expression::Literal(naga::Literal::F32(*value)))
            }
            Operand::U32(value) => {
                self.expr(out, naga::Expression::Literal(naga::Literal::U32(*value)))
            }
            Operand::Bool(value) => {
                self.expr(out, naga::Expression::Literal(naga::Literal::Bool(*value)))
            }
        };
        Ok(v)
    }

    /// Append an expression and emit it in `out` when Naga requires it.
    fn expr(
        &mut self,
        out: &mut naga::Block,
        expr: naga::Expression,
    ) -> naga::Handle<naga::Expression> {
        let handle = self.func.expressions.append(expr, naga::Span::default());
        if !self.func.expressions[handle].needs_pre_emit() {
            let range = self.func.expressions.range_from(handle.index());
            out.push(naga::Statement::Emit(range), naga::Span::default());
        }
        handle
    }
}

fn load_global(
    func: &mut naga::Function,
    out: &mut naga::Block,
    global: naga::Handle<naga::GlobalVariable>,
    index: naga::Handle<naga::Expression>,
) -> Result<naga::Handle<naga::Expression>, TranslateError> {
    let base = append(func, naga::Expression::GlobalVariable(global));
    push_emit(func, out, base);
    let pointer = append(func, naga::Expression::Access { base, index });
    push_emit(func, out, pointer);
    let v = append(func, naga::Expression::Load { pointer });
    push_emit(func, out, v);
    Ok(v)
}

/// Load `uniform_0_{b}.a[index].v` through the lane wrapper.
fn load_uniform(
    func: &mut naga::Function,
    out: &mut naga::Block,
    global: naga::Handle<naga::GlobalVariable>,
    index: naga::Handle<naga::Expression>,
) -> Result<naga::Handle<naga::Expression>, TranslateError> {
    let base = append(func, naga::Expression::GlobalVariable(global));
    push_emit(func, out, base);
    let field = append(func, naga::Expression::AccessIndex { base, index: 0 });
    push_emit(func, out, field);
    let elem = append(func, naga::Expression::Access { base: field, index });
    push_emit(func, out, elem);
    let lane = append(
        func,
        naga::Expression::AccessIndex {
            base: elem,
            index: 0,
        },
    );
    push_emit(func, out, lane);
    let v = append(func, naga::Expression::Load { pointer: lane });
    push_emit(func, out, v);
    Ok(v)
}

fn const_u32(op: &Operand) -> Result<u32, TranslateError> {
    match op {
        Operand::U32(value) => Ok(*value),
        // A local is not materialised; group/binding must be immediates
        // (same rule as `wasm2gpu`).
        Operand::Local(_) | Operand::F32(_) | Operand::Bool(_) => Err(TranslateError::Unsupported),
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

fn push_emit(
    func: &mut naga::Function,
    out: &mut naga::Block,
    handle: naga::Handle<naga::Expression>,
) {
    if !func.expressions[handle].needs_pre_emit() {
        let range = func.expressions.range_from(handle.index());
        out.push(naga::Statement::Emit(range), naga::Span::default());
    }
}

fn operand(
    func: &mut naga::Function,
    out: &mut naga::Block,
    op: &Operand,
    env: &Env,
) -> Result<naga::Handle<naga::Expression>, TranslateError> {
    match op {
        Operand::Local(name) => env.get(name).copied().ok_or(TranslateError::Unsupported),
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_uniform_struct_uses_wgsl_offsets() {
        use ornis_gpu_abi::{TYPE_TAG_F32, TYPE_TAG_U32, TYPE_TAG_VEC3};
        let ir = r#"
declare float @uniform_field_f32(i32, i32, i32, i32)
declare void @storage_write_f32(i32, i32, i32, float)
define void @cs_main(i32 %id) {
start:
  %v = call float @uniform_field_f32(i32 0, i32 0, i32 0, i32 1)
  call void @storage_write_f32(i32 0, i32 2, i32 %id, float %v)
  ret void
}
"#;
        let func = crate::parse::parse(ir).expect("parse");
        // `vec3` is 12 bytes but aligns to 16; the scalar after it sits at
        // byte 12 and the struct rounds up to 32 (uniform alignment).
        let structs: &[&[u8]] = &[&[TYPE_TAG_VEC3, TYPE_TAG_F32, TYPE_TAG_U32]];
        let module = lower(&func, structs).expect("lower");
        let ty = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("UniformStruct0"))
            .map(|(handle, _)| handle)
            .expect("uniform struct type");
        let naga::TypeInner::Struct { members, span } = &module.types[ty].inner else {
            panic!("expected a struct");
        };
        assert_eq!(*span, 32);
        let offsets: Vec<u32> = members.iter().map(|member| member.offset).collect();
        assert_eq!(offsets, [0, 12, 16]);
    }
}
