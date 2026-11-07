//! Lower a parsed LLVM function to a Naga compute entry.
//!
//! The supported subset is reducible control flow after rustc's `SROA`:
//! straight-line scalar arithmetic, `select`, diamond `if`s with `phi`
//! merges, and natural loops, including a loop nested inside another.
//! Phi values are destroyed
//! into local variables on each incoming edge, which lets Naga's structured
//! `If`/`Loop` carry the same values without a general SSA relooper.
//! A value defined in a loop and used after it — LLVM skips the phi when the
//! exit has one predecessor — is spilled to a local and reloaded at the exit.
//!
//! Storage and uniform accesses go through the `ornis_gpu` ABI exactly as
//! `wasm2gpu` declares them. A scalar `-> float` result is written to
//! `storage_0_0[id]`, mirroring the toy `compute` kernel; `-> void` kernels
//! store through the ABI themselves. Irreducible control flow, a loop with
//! several exits, and a `break` that leaves more than the innermost loop
//! are rejected as `Unsupported` so an unknown shape never silently becomes
//! a wrong shader. A natural loop nested in another is emitted as a nested
//! `loop` when each has one exit and the inner exit stays in the outer body.

use crate::error::TranslateError;
use crate::parse::{BasicBlock, Instr, LlvmFunc, Operand, Terminator, Ty};
use std::collections::{HashMap, HashSet};

/// Synthetic post-dominator sink used while structuring the CFG.
const EXIT: &str = "$exit";

/// Bytes in one 32-bit storage word.
const WORD_BYTES: u32 = core::mem::size_of::<f32>() as u32;
/// Naga's scalar width field.
const SCALAR_WIDTH: u8 = core::mem::size_of::<f32>() as u8;
/// Uniform (std140) lane stride for one scalar.
const UNIFORM_LANE_BYTES: u32 = core::mem::size_of::<[f32; 4]>() as u32;
/// Largest uniform binding accepted by the translator.
const MAX_UNIFORM_BYTES: u32 = u16::MAX as u32 + 1;
/// Lane counts of the GPU vector types.
const VEC2_LANES: u32 = ornis_gpu_abi::VEC2_SIZE / WORD_BYTES;
const VEC3_LANES: u32 = ornis_gpu_abi::VEC3_SIZE / WORD_BYTES;
const VEC4_LANES: u32 = ornis_gpu_abi::VEC4_SIZE / WORD_BYTES;
/// Byte offsets of the second, third, and fourth scalar words.
const WORD_2: u32 = WORD_BYTES + WORD_BYTES;
const WORD_3: u32 = WORD_2 + WORD_BYTES;

/// Lower `func` (named `cs_main`) to a Naga module with one compute entry.
///
/// Two entry shapes are supported:
/// - `cs_main(i32) -> float`: result is stored to `storage_0_0[id]`.
/// - `cs_main(i32) -> void`: the body stores through ABI calls.
pub(super) fn lower(
    func: &LlvmFunc,
    structs: &[&[u8]],
    workgroup: [u32; 3],
) -> Result<naga::Module, TranslateError> {
    let fragment = func.name == ornis_gpu_abi::symbols::FS_MAIN;
    if func.name != ornis_gpu_abi::symbols::CS_MAIN && !fragment {
        return Err(TranslateError::Unsupported);
    }
    if fragment {
        // Color is written by `fragment_store_f32x4` and returned at location 0.
        if func.ret_ty != Ty::Void || !func.params.is_empty() {
            return Err(TranslateError::Unsupported);
        }
    } else if !matches!(func.ret_ty, Ty::Float | Ty::Void) {
        return Err(TranslateError::Unsupported);
    } else if func.params.len() == 1 && func.params[0].0 == Ty::I32 {
        // `cs_main(i32)` — standard.
    } else if func.params.is_empty() {
        // `cs_main()` — allowed for kernels that read `global_invocation_id`
        // themselves; they simply do not use `%id`. Accept and ignore.
    } else {
        return Err(TranslateError::Unsupported);
    }
    let cfg = Cfg::new(func)?;
    let mut builders = Builders::new(structs);
    let function = builders.build(func, &cfg, fragment)?;
    builders.finish(function, func.ret_ty, workgroup, fragment)
}

/// Which `texture_*` binding a load addresses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TextureKind {
    /// `texture_depth_2d`. `textureLoad` returns `f32`.
    Depth,
    /// `texture_2d<f32>`. `textureLoad` returns `vec4<f32>`.
    Sampled,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Elem {
    F32,
    U32,
    AtomicU32,
    /// Bit-preserving four-word record chunk (`array<vec4<u32>>`).
    Vec4U32,
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

    /// Merge label of a conditional branch.
    ///
    /// When one arm returns directly and the other continues, the
    /// continuing arm is the merge: the branch becomes
    /// `if cond { return; }` and the rest follows the `if`. A plain
    /// post-dominator cannot see that (the returning arm never reaches
    /// the continuation, and a loop on the other arm need not reach the
    /// exit), so special-case the direct-return shape.
    fn branch_merge(
        &self,
        label: &str,
        true_label: &'a str,
        false_label: &'a str,
    ) -> Option<&'a str> {
        let merge = self.merge(label);
        if let Some(merge_label) = merge
            && self.returns_directly(merge_label)
        {
            if true_label == merge_label && false_label != merge_label {
                return Some(false_label);
            }
            if false_label == merge_label && true_label != merge_label {
                return Some(true_label);
            }
        }
        // A loop on one arm never post-dominates the branch (the back
        // edge need not reach the exit), so `merge` is `None` there too.
        if merge.is_none() || merge == Some(EXIT) {
            let true_returns = self.returns_directly(true_label);
            let false_returns = self.returns_directly(false_label);
            if true_returns && !false_returns {
                return self.blocks.get(false_label).map(|_| false_label);
            }
            if false_returns && !true_returns {
                return self.blocks.get(true_label).map(|_| true_label);
            }
        }
        merge
    }

    fn returns_directly(&self, label: &str) -> bool {
        self.blocks
            .get(label)
            .is_some_and(|block| matches!(block.term, Terminator::Ret(_)))
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

fn loop_exits<'a>(cfg: &Cfg<'a>, body: &HashSet<&'a str>) -> HashSet<&'a str> {
    body.iter()
        .flat_map(|block| cfg.succs.get(block).into_iter().flatten())
        .filter(|target| !body.contains(*target))
        .copied()
        .collect()
}

fn def_ty(cfg: &Cfg<'_>, body: &HashSet<&str>, name: &str) -> Option<Ty> {
    for label in body {
        let block = cfg.blocks.get(label)?;
        for instr in &block.instrs {
            if instr.defines(name) {
                return instr.value_ty();
            }
        }
    }
    None
}

fn push_local(op: &Operand, used: &mut Vec<String>) {
    if let Operand::Local(name) = op {
        used.push(name.clone());
    }
}

fn push_term_local(term: &Terminator, used: &mut Vec<String>) {
    match term {
        Terminator::BrCond { cond, .. } => push_local(cond, used),
        Terminator::Ret(Some(value)) => push_local(value, used),
        Terminator::BrUncond(_) | Terminator::Ret(None) => {}
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
    /// One storage variable per `(group, binding)`. A second element type is
    /// rejected: two variables at one binding would alias, and a writable
    /// binding must not alias anything.
    storage_binding: HashMap<(u32, u32), (Elem, naga::Handle<naga::GlobalVariable>)>,
    uniform_cache: HashMap<(u32, u32), naga::Handle<naga::GlobalVariable>>,
    uniform_struct_cache: HashMap<(u32, u32, u32), naga::Handle<naga::GlobalVariable>>,
    /// Built uniform struct types, keyed by struct id: type, byte size, align.
    struct_types: HashMap<u32, (naga::Handle<naga::Type>, u32, u32)>,
    /// Struct ids currently being laid out, to reject a recursive field.
    struct_stack: Vec<u32>,
    /// `texture_2d` / `texture_depth_2d` globals, one per binding.
    textures: HashMap<(u32, u32), (TextureKind, naga::Handle<naga::GlobalVariable>)>,
    /// `vec2<i32>` used as a `textureLoad` coordinate.
    ivec2: Option<naga::Handle<naga::Type>>,
    /// `vec4<f32>` shared by the fragment output and `fragment_store`.
    vec4_f32: Option<naga::Handle<naga::Type>>,
    /// Plain `array<u32, WORKGROUP_WORDS>` in workgroup memory.
    workgroup: Option<naga::Handle<naga::GlobalVariable>>,
    /// `array<atomic<u32>, WORKGROUP_WORDS>` in workgroup memory.
    workgroup_atomic: Option<naga::Handle<naga::GlobalVariable>>,
    /// `ornis_gpu.types` field tags per struct id.
    structs: Vec<Vec<u8>>,
}

impl Builders {
    fn new(structs: &[&[u8]]) -> Self {
        Self {
            module: naga::Module::default(),
            storage_binding: HashMap::new(),
            uniform_cache: HashMap::new(),
            uniform_struct_cache: HashMap::new(),
            struct_types: HashMap::new(),
            struct_stack: Vec::new(),
            textures: HashMap::new(),
            ivec2: None,
            vec4_f32: None,
            workgroup: None,
            workgroup_atomic: None,
            structs: structs.iter().map(|tags| tags.to_vec()).collect(),
        }
    }

    fn finish(
        mut self,
        mut function: naga::Function,
        ret_ty: Ty,
        workgroup: [u32; 3],
        fragment: bool,
    ) -> Result<naga::Module, TranslateError> {
        // Only create the implicit `storage_0_0` global when the kernel
        // uses the `-> float` shape; for `-> void` the body creates its
        // own globals lazily.
        if ret_ty == Ty::Float {
            self.storage_global(0, 0, Elem::F32, true)?;
        }
        let (name, stage, workgroup_size) = if fragment {
            let ty = self.vec4_f32();
            function.result = Some(naga::FunctionResult {
                ty,
                binding: Some(naga::Binding::Location {
                    location: 0,
                    interpolation: Some(naga::Interpolation::Perspective),
                    sampling: Some(naga::Sampling::Center),
                    blend_src: None,
                    per_primitive: false,
                }),
            });
            (
                ornis_gpu_abi::symbols::FS_MAIN.to_string(),
                naga::ShaderStage::Fragment,
                [0, 0, 0],
            )
        } else {
            (
                ornis_gpu_abi::symbols::CS_MAIN.to_string(),
                naga::ShaderStage::Compute,
                workgroup,
            )
        };
        self.module.entry_points.push(naga::EntryPoint {
            name,
            stage,
            early_depth_test: None,
            workgroup_size,
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
        write: bool,
    ) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
        if let Some(handle) = self.existing_storage(group, binding, elem, write)? {
            return Ok(handle);
        }
        let base_ty = match elem {
            Elem::F32 => scalar(&mut self.module, naga::ScalarKind::Float),
            Elem::U32 => scalar(&mut self.module, naga::ScalarKind::Uint),
            Elem::AtomicU32 => self.module.types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Atomic(naga::Scalar {
                        kind: naga::ScalarKind::Uint,
                        width: SCALAR_WIDTH,
                    }),
                },
                naga::Span::default(),
            ),
            Elem::Vec4U32 => self.module.types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Vector {
                        size: naga::VectorSize::Quad,
                        scalar: naga::Scalar {
                            kind: naga::ScalarKind::Uint,
                            width: SCALAR_WIDTH,
                        },
                    },
                },
                naga::Span::default(),
            ),
        };
        let stride = if elem == Elem::Vec4U32 {
            UNIFORM_LANE_BYTES
        } else {
            WORD_BYTES
        };
        let array_ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Array {
                    base: base_ty,
                    size: naga::ArraySize::Dynamic,
                    stride,
                },
            },
            naga::Span::default(),
        );
        let name = match elem {
            Elem::AtomicU32 => format!("atomic_storage_{group}_{binding}_"),
            _ => format!("storage_{group}_{binding}_"),
        };
        let mut access = naga::StorageAccess::LOAD;
        if write {
            access |= naga::StorageAccess::STORE;
        }
        let handle = self.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(name),
                space: naga::AddressSpace::Storage { access },
                binding: Some(naga::ResourceBinding { group, binding }),
                ty: array_ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        );
        self.storage_binding
            .insert((group, binding), (elem, handle));
        Ok(handle)
    }

    /// Texture at `(group, binding)`. Depth and sampled images are different
    /// resources, and a storage variable already there would share the slot.
    fn texture(
        &mut self,
        group: u32,
        binding: u32,
        want: TextureKind,
    ) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
        if self.storage_binding.contains_key(&(group, binding)) {
            return Err(TranslateError::UnsupportedType(format!(
                "binding {group}:{binding} is both storage and a texture"
            )));
        }
        if let Some((kind, handle)) = self.textures.get(&(group, binding)) {
            if *kind != want {
                return Err(TranslateError::UnsupportedType(format!(
                    "texture binding {group}:{binding} is used as two kinds"
                )));
            }
            return Ok(*handle);
        }
        let ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Image {
                    dim: naga::ImageDimension::D2,
                    arrayed: false,
                    class: image_class(want),
                },
            },
            naga::Span::default(),
        );
        let handle = self.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(format!("tex_{group}_{binding}_")),
                space: naga::AddressSpace::Handle,
                binding: Some(naga::ResourceBinding { group, binding }),
                ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        );
        self.textures.insert((group, binding), (want, handle));
        Ok(handle)
    }

    fn workgroup_words(&mut self) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
        if let Some(var) = self.workgroup {
            return Ok(var);
        }
        let base = scalar(&mut self.module, naga::ScalarKind::Uint);
        let var = self.workgroup_array("workgroup_scratch", base)?;
        self.workgroup = Some(var);
        Ok(var)
    }

    fn workgroup_atomics(&mut self) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
        if let Some(var) = self.workgroup_atomic {
            return Ok(var);
        }
        let base = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Atomic(naga::Scalar {
                    kind: naga::ScalarKind::Uint,
                    width: SCALAR_WIDTH,
                }),
            },
            naga::Span::default(),
        );
        let var = self.workgroup_array("workgroup_atomics", base)?;
        self.workgroup_atomic = Some(var);
        Ok(var)
    }

    fn workgroup_array(
        &mut self,
        name: &str,
        base: naga::Handle<naga::Type>,
    ) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
        let Some(len) = std::num::NonZeroU32::new(ornis_gpu_abi::symbols::WORKGROUP_WORDS) else {
            return Err(TranslateError::UnsupportedType(
                "workgroup array is empty".to_string(),
            ));
        };
        let ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Array {
                    base,
                    size: naga::ArraySize::Constant(len),
                    stride: ornis_gpu_abi::F32_SIZE,
                },
            },
            naga::Span::default(),
        );
        Ok(self.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(name.to_string()),
                space: naga::AddressSpace::WorkGroup,
                binding: None,
                ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        ))
    }

    fn vec4_f32(&mut self) -> naga::Handle<naga::Type> {
        if let Some(ty) = self.vec4_f32 {
            return ty;
        }
        let ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Quad,
                    scalar: naga::Scalar {
                        kind: naga::ScalarKind::Float,
                        width: SCALAR_WIDTH,
                    },
                },
            },
            naga::Span::default(),
        );
        self.vec4_f32 = Some(ty);
        ty
    }

    fn ivec2(&mut self) -> naga::Handle<naga::Type> {
        if let Some(ty) = self.ivec2 {
            return ty;
        }
        let ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Bi,
                    scalar: naga::Scalar {
                        kind: naga::ScalarKind::Sint,
                        width: SCALAR_WIDTH,
                    },
                },
            },
            naga::Span::default(),
        );
        self.ivec2 = Some(ty);
        ty
    }

    /// The variable already created for this binding, upgrading it to
    /// writable when this use stores. A second element type is rejected.
    fn existing_storage(
        &mut self,
        group: u32,
        binding: u32,
        elem: Elem,
        write: bool,
    ) -> Result<Option<naga::Handle<naga::GlobalVariable>>, TranslateError> {
        let Some((existing, handle)) = self.storage_binding.get(&(group, binding)) else {
            return Ok(None);
        };
        if *existing != elem {
            return Err(TranslateError::UnsupportedType(format!(
                "storage binding {group}:{binding} is used as two types"
            )));
        }
        if write
            && let naga::AddressSpace::Storage { access } =
                &mut self.module.global_variables[*handle].space
        {
            *access |= naga::StorageAccess::STORE;
        }
        Ok(Some(*handle))
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
                            offset: WORD_BYTES,
                        },
                        naga::StructMember {
                            name: Some("p2".to_string()),
                            ty: scalar_ty,
                            binding: None,
                            offset: WORD_2,
                        },
                        naga::StructMember {
                            name: Some("p3".to_string()),
                            ty: scalar_ty,
                            binding: None,
                            offset: WORD_3,
                        },
                    ],
                    span: UNIFORM_LANE_BYTES,
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
                        std::num::NonZeroU32::new(len).unwrap_or(std::num::NonZeroU32::MIN),
                    ),
                    stride: UNIFORM_LANE_BYTES,
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
                    span: UNIFORM_LANE_BYTES * len,
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
    // qual:allow(iosp) reason: "the binding is created only after the struct type has been laid out, and a repeat call must reuse it"
    fn uniform_struct_global(
        &mut self,
        group: u32,
        binding: u32,
        sid: u32,
    ) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
        if let Some(handle) = self.uniform_struct_cache.get(&(group, binding, sid)) {
            return Ok(*handle);
        }
        let (struct_ty, _, _) = self.struct_type(sid)?;
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

    /// Type, byte size and alignment of uniform struct `sid`.
    // qual:allow(iosp) reason: "laying a uniform struct out is the check of each field tag together with the naga type it becomes"
    fn struct_type(
        &mut self,
        sid: u32,
    ) -> Result<(naga::Handle<naga::Type>, u32, u32), TranslateError> {
        if let Some(found) = self.struct_types.get(&sid) {
            return Ok(*found);
        }
        if self.struct_stack.contains(&sid) {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform struct {sid} contains itself"
            )));
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
        self.struct_stack.push(sid);
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
        for (member, offset) in members.iter_mut().zip(&offsets) {
            member.offset = *offset;
        }
        let ty = self.module.types.insert(
            naga::Type {
                name: Some(format!("UniformStruct{sid}")),
                inner: naga::TypeInner::Struct { members, span },
            },
            naga::Span::default(),
        );
        self.struct_stack.pop();
        let align = ornis_gpu_abi::UNIFORM_STRUCT_ALIGN;
        let found = (ty, span, align);
        self.struct_types.insert(sid, found);
        Ok(found)
    }

    // qual:allow(iosp) reason: "a field tag either selects a scalar naga type or recurses into the nested struct it names"
    fn tag_type(
        &mut self,
        tag: u8,
    ) -> Result<(naga::Handle<naga::Type>, u32, u32), TranslateError> {
        if tag >= ornis_gpu_abi::TYPE_TAG_NESTED {
            return self.struct_type(u32::from(tag - ornis_gpu_abi::TYPE_TAG_NESTED));
        }
        use ornis_gpu_abi::{
            F32_ALIGN, F32_SIZE, TYPE_TAG_F32, TYPE_TAG_I32, TYPE_TAG_U32, TYPE_TAG_VEC2,
            TYPE_TAG_VEC3, TYPE_TAG_VEC4, VEC2_ALIGN, VEC2_SIZE, VEC3_ALIGN, VEC3_SIZE, VEC4_ALIGN,
            VEC4_SIZE,
        };
        let (kind, lanes, size, align) = match tag {
            TYPE_TAG_F32 => (naga::ScalarKind::Float, 1, F32_SIZE, F32_ALIGN),
            TYPE_TAG_U32 => (naga::ScalarKind::Uint, 1, F32_SIZE, F32_ALIGN),
            TYPE_TAG_I32 => (naga::ScalarKind::Sint, 1, F32_SIZE, F32_ALIGN),
            TYPE_TAG_VEC2 => (naga::ScalarKind::Float, VEC2_LANES, VEC2_SIZE, VEC2_ALIGN),
            TYPE_TAG_VEC3 => (naga::ScalarKind::Float, VEC3_LANES, VEC3_SIZE, VEC3_ALIGN),
            TYPE_TAG_VEC4 => (naga::ScalarKind::Float, VEC4_LANES, VEC4_SIZE, VEC4_ALIGN),
            _ => {
                return Err(TranslateError::UnsupportedType(format!(
                    "unsupported uniform field tag {tag}"
                )));
            }
        };
        let inner = if lanes == 1 {
            naga::TypeInner::Scalar(naga::Scalar {
                kind,
                width: SCALAR_WIDTH,
            })
        } else {
            let size = if lanes == VEC2_LANES {
                naga::VectorSize::Bi
            } else if lanes == VEC3_LANES {
                naga::VectorSize::Tri
            } else {
                naga::VectorSize::Quad
            };
            naga::TypeInner::Vector {
                size,
                scalar: naga::Scalar {
                    kind,
                    width: SCALAR_WIDTH,
                },
            }
        };
        let ty = self
            .module
            .types
            .insert(naga::Type { name: None, inner }, naga::Span::default());
        Ok((ty, size, align))
    }

    fn build(
        &mut self,
        func: &LlvmFunc,
        cfg: &Cfg<'_>,
        fragment: bool,
    ) -> Result<naga::Function, TranslateError> {
        let entry = if fragment {
            ornis_gpu_abi::symbols::FS_MAIN
        } else {
            ornis_gpu_abi::symbols::CS_MAIN
        };
        let mut function = naga::Function {
            name: Some(entry.to_string()),
            ..Default::default()
        };
        let id_x = if fragment {
            None
        } else {
            Some(invocation_x(&mut self.module, &mut function))
        };
        let position = fragment.then(|| position_arg(self.vec4_f32(), &mut function));
        let fragment_out = fragment.then(|| {
            let ty = self.vec4_f32();
            function.local_variables.append(
                naga::LocalVariable {
                    name: Some("fragment_out".to_string()),
                    ty,
                    init: None,
                },
                naga::Span::default(),
            )
        });

        let mut lowerer = Lowerer {
            builders: self,
            cfg,
            func: function,
            id_x,
            fragment_out,
            position,
            phi_vars: HashMap::new(),
            spill: HashMap::new(),
            env: HashMap::new(),
            saw_return: false,
        };
        if let Some((id_x, (_, name))) = id_x.zip(func.params.first()) {
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
    id_x: Option<naga::Handle<naga::Expression>>,
    /// `vec4` local returned from `fs_main` at `@location(0)`.
    fragment_out: Option<naga::Handle<naga::LocalVariable>>,
    /// `@builtin(position)` of `fs_main`, when this entry is a fragment.
    position: Option<naga::Handle<naga::Expression>>,
    phi_vars: HashMap<String, naga::Handle<naga::LocalVariable>>,
    /// Loop values used after the loop, stored while the loop runs.
    spill: HashMap<String, naga::Handle<naga::LocalVariable>>,
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
                    return self.emit_loop(start, stop, out, Ctx::Plain);
                }
            }
            Ctx::Loop { header, .. } => {
                if start != header && self.cfg.loops.contains_key(start) {
                    // `break`/`continue` emitted for this header stay inside
                    // its `Loop`; the caller context receives the exit.
                    return self.emit_loop(start, stop, out, ctx);
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
        let merge = self
            .cfg
            .branch_merge(label, true_label, false_label)
            .ok_or(TranslateError::Unsupported)?;
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
    ///
    /// `outer` is the region that contains this loop. The exit block is
    /// emitted there, so a nested loop's `break` does not leave the outer
    /// `loop` and a branch back to the outer header stays a `continue`.
    fn emit_loop(
        &mut self,
        header: &'a str,
        stop: Option<&'a str>,
        out: &mut naga::Block,
        outer: Ctx<'a>,
    ) -> Result<Flow, TranslateError> {
        let body = self
            .cfg
            .loops
            .get(header)
            .ok_or(TranslateError::Unsupported)?
            .clone();
        let exits = loop_exits(self.cfg, &body);
        let live = self.loop_live_outs(&body, &exits)?;
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
        self.reload_live(&live, out)?;
        if stop == Some(exit) {
            return Ok(Flow::Next);
        }
        if let Ctx::Loop {
            header: outer_header,
        } = outer
        {
            if exit == outer_header {
                out.push(naga::Statement::Continue, naga::Span::default());
                return Ok(Flow::Continue);
            }
            let stays = self
                .cfg
                .loops
                .get(outer_header)
                .is_some_and(|body| body.contains(exit));
            if !stays {
                // Leaving two loops at once needs a break past the inner
                // `Loop`; one `break` would stop at the inner header.
                return Err(TranslateError::Unsupported);
            }
        }
        self.emit_region(exit, stop, out, outer)
    }

    /// Values the single exit reads that are defined inside the loop.
    fn loop_live_outs(
        &mut self,
        body: &HashSet<&'a str>,
        exits: &HashSet<&'a str>,
    ) -> Result<Vec<String>, TranslateError> {
        if exits.len() != 1 {
            return Ok(Vec::new());
        }
        let exit = *exits.iter().next().ok_or(TranslateError::Unsupported)?;
        let block = *self
            .cfg
            .blocks
            .get(exit)
            .ok_or(TranslateError::Unsupported)?;
        let mut used = Vec::new();
        for instr in &block.instrs {
            if matches!(instr, Instr::Phi { .. }) {
                continue;
            }
            instr.each_operand(|op| push_local(op, &mut used));
        }
        push_term_local(&block.term, &mut used);
        // A phi outside the loop can name a value defined inside it. The
        // edge store runs after the loop, so that value has to be spilled.
        for (label, phis) in &self.cfg.phis {
            if body.contains(label) {
                continue;
            }
            for phi in phis {
                for (value, _) in &phi.incoming {
                    push_local(value, &mut used);
                }
            }
        }
        let mut live = Vec::new();
        for name in used {
            self.spill_live(body, &name, &mut live)?;
        }
        Ok(live)
    }

    fn spill_live(
        &mut self,
        body: &HashSet<&str>,
        name: &str,
        live: &mut Vec<String>,
    ) -> Result<(), TranslateError> {
        if live.iter().any(|found| found == name) {
            return Ok(());
        }
        if self.env.contains_key(name) {
            return Ok(());
        }
        if self.spill.contains_key(name) {
            live.push(name.to_string());
            return Ok(());
        }
        let Some(ty) = def_ty(self.cfg, body, name) else {
            return Ok(());
        };
        let ty = self.ty(ty)?;
        let var = self.func.local_variables.append(
            naga::LocalVariable {
                name: Some(format!("out.{name}")),
                ty,
                init: None,
            },
            naga::Span::default(),
        );
        self.spill.insert(name.to_string(), var);
        live.push(name.to_string());
        Ok(())
    }

    fn reload_live(
        &mut self,
        names: &[String],
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        for name in names {
            let var = *self.spill.get(name).ok_or(TranslateError::Unsupported)?;
            let pointer = self.expr(out, naga::Expression::LocalVariable(var));
            let value = self.expr(out, naga::Expression::Load { pointer });
            self.env.insert(name.clone(), value);
        }
        Ok(())
    }

    /// Record `dst` and, when the exit reads it, keep the value past the loop.
    fn bind(&mut self, dst: &str, value: naga::Handle<naga::Expression>, out: &mut naga::Block) {
        if let Some(&var) = self.spill.get(dst) {
            let pointer = self.expr(out, naga::Expression::LocalVariable(var));
            out.push(
                naga::Statement::Store { pointer, value },
                naga::Span::default(),
            );
        }
        self.env.insert(dst.to_string(), value);
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
                    self.bind(dst, value, out);
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
        if self.fragment_out.is_some() {
            let var = self.fragment_out.ok_or(TranslateError::Unsupported)?;
            let pointer = self.expr(out, naga::Expression::LocalVariable(var));
            let value = self.expr(out, naga::Expression::Load { pointer });
            out.push(
                naga::Statement::Return { value: Some(value) },
                naga::Span::default(),
            );
            self.saw_return = true;
            return Ok(Flow::Returned);
        }
        if let Some(op) = value {
            let value = self.operand(out, op)?;
            let global = self.builders.storage_global(0, 0, Elem::F32, true)?;
            let base = self.expr(out, naga::Expression::GlobalVariable(global));
            let index = self.id_x.ok_or(TranslateError::Unsupported)?;
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
                self.bind(dst, v, out);
            }
            Instr::Fcmp { dst, pred, a, b } | Instr::Icmp { dst, pred, a, b } => {
                let v = self.compare(out, pred, a, b)?;
                self.bind(dst, v, out);
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
                self.bind(dst, v, out);
            }
            Instr::Uitofp { dst, a } => {
                let src = self.operand(out, a)?;
                let v = self.expr(
                    out,
                    naga::Expression::As {
                        expr: src,
                        kind: naga::ScalarKind::Float,
                        convert: Some(SCALAR_WIDTH),
                    },
                );
                self.bind(dst, v, out);
            }
            Instr::Bitcast { dst, from, to, a } => {
                if from == to {
                    let value = self.operand(out, a)?;
                    self.bind(dst, value, out);
                } else {
                    let kind = match (from, to) {
                        (Ty::I32, Ty::Float) => naga::ScalarKind::Float,
                        (Ty::Float, Ty::I32) => naga::ScalarKind::Uint,
                        _ => return Err(TranslateError::Unsupported),
                    };
                    let expr = self.operand(out, a)?;
                    let value = self.expr(
                        out,
                        naga::Expression::As {
                            expr,
                            kind,
                            convert: None,
                        },
                    );
                    self.bind(dst, value, out);
                }
            }
            Instr::Call { dst, callee, args } => {
                let v = self.call_value(out, callee, args)?;
                self.bind(dst, v, out);
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
        self.bind(dst, v, out);
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
        if let Some(fun) = min_max(callee) {
            if args.len() != 2 {
                return Err(TranslateError::Unsupported);
            }
            let arg = self.operand(out, &args[0])?;
            let arg1 = self.operand(out, &args[1])?;
            return Ok(self.expr(
                out,
                naga::Expression::Math {
                    fun,
                    arg,
                    arg1: Some(arg1),
                    arg2: None,
                    arg3: None,
                },
            ));
        }
        if callee.contains("fragment_position_lane") {
            return self.fragment_position(out, args);
        }
        if callee.contains("workgroup_atomic_add_u32") {
            return self.workgroup_atomic_add(out, args);
        }
        if callee.contains("workgroup_atomic_load_u32") {
            return self.workgroup_load(out, args, true);
        }
        if callee.contains("workgroup_load_u32") {
            return self.workgroup_load(out, args, false);
        }
        if callee.contains("texture_load_2d_f32x4") {
            return self.texture_load_lane(out, args);
        }
        if callee.contains("texture_load_2d_f32") {
            return self.texture_load_f32(out, args);
        }
        if callee.contains("texture_width_2d_f32x4") {
            return self.texture_size(out, args, 0, TextureKind::Sampled);
        }
        if callee.contains("texture_height_2d_f32x4") {
            return self.texture_size(out, args, 1, TextureKind::Sampled);
        }
        if callee.contains("texture_width_2d") {
            return self.texture_size(out, args, 0, TextureKind::Depth);
        }
        if callee.contains("texture_height_2d") {
            return self.texture_size(out, args, 1, TextureKind::Depth);
        }
        if callee.contains("uniform_nested_field_f32") {
            return self.uniform_nested_field(out, args, ornis_gpu_abi::TYPE_TAG_F32);
        }
        if callee.contains("uniform_nested_field_u32") {
            return self.uniform_nested_field(out, args, ornis_gpu_abi::TYPE_TAG_U32);
        }
        if callee.contains("uniform_field_vec4_plane_dot") {
            return self.uniform_plane_dot(out, args);
        }
        if callee.contains("uniform_field_vec4_component_f32") {
            return self.uniform_field_vec4_component(out, args);
        }
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
            if len == 0
                || len
                    .checked_mul(UNIFORM_LANE_BYTES)
                    .is_none_or(|b| b > MAX_UNIFORM_BYTES)
            {
                return Err(TranslateError::UnsupportedType(
                    "uniform array exceeds 64KiB".to_string(),
                ));
            }
            let index = self.operand(out, &args[3])?;
            let global = self.builders.uniform_global(group, binding, len);
            return load_uniform(&mut self.func, out, global, index);
        }
        if callee.contains("storage_len_f32") {
            if args.len() != 2 {
                return Err(TranslateError::Unsupported);
            }
            let group = const_u32(&args[0])?;
            let binding = const_u32(&args[1])?;
            let global = self
                .builders
                .storage_global(group, binding, Elem::F32, false)?;
            let base = self.expr(out, naga::Expression::GlobalVariable(global));
            return Ok(self.expr(out, naga::Expression::ArrayLength(base)));
        }
        if callee.contains("storage_len_u32") && !callee.contains("storage_len_u32x4") {
            if args.len() != 2 {
                return Err(TranslateError::Unsupported);
            }
            let group = const_u32(&args[0])?;
            let binding = const_u32(&args[1])?;
            let global = self
                .builders
                .storage_global(group, binding, Elem::U32, false)?;
            let base = self.expr(out, naga::Expression::GlobalVariable(global));
            return Ok(self.expr(out, naga::Expression::ArrayLength(base)));
        }
        if callee.contains("storage_len_u32x4") {
            if args.len() != 2 {
                return Err(TranslateError::Unsupported);
            }
            let group = const_u32(&args[0])?;
            let binding = const_u32(&args[1])?;
            let global = self
                .builders
                .storage_global(group, binding, Elem::Vec4U32, false)?;
            let base = self.expr(out, naga::Expression::GlobalVariable(global));
            return Ok(self.expr(out, naga::Expression::ArrayLength(base)));
        }
        if callee.contains("storage_read_u32_lane") {
            if args.len() != 4 {
                return Err(TranslateError::Unsupported);
            }
            let group = const_u32(&args[0])?;
            let binding = const_u32(&args[1])?;
            let index = self.operand(out, &args[2])?;
            let lane = const_u32(&args[3])?;
            let global = self
                .builders
                .storage_global(group, binding, Elem::Vec4U32, false)?;
            let base = self.expr(out, naga::Expression::GlobalVariable(global));
            let chunk = self.expr(out, naga::Expression::Access { base, index });
            let pointer = self.expr(
                out,
                naga::Expression::AccessIndex {
                    base: chunk,
                    index: lane,
                },
            );
            return Ok(self.expr(out, naga::Expression::Load { pointer }));
        }
        if callee.contains("storage_read_f32") {
            let (group, binding, index) = self.storage_args(out, args, 3)?;
            let global = self
                .builders
                .storage_global(group, binding, Elem::F32, false)?;
            return load_global(&mut self.func, out, global, index);
        }
        if callee.contains("storage_read_u32") {
            let (group, binding, index) = self.storage_args(out, args, 3)?;
            let global = self
                .builders
                .storage_global(group, binding, Elem::U32, false)?;
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
                .storage_global(group, binding, Elem::AtomicU32, true)?;
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
        if callee.contains("cross3") {
            return self.cross3_lane(out, args);
        }
        if callee.contains("normalize3") {
            return self.normalize3_lane(out, args);
        }
        if callee.contains("dot3") && args.len() == VEC3_LANES as usize + VEC3_LANES as usize {
            let va = self.vec3_of(out, args, 0)?;
            let vb = self.vec3_of(out, args, VEC3_LANES)?;
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
        if callee.contains("length3") && args.len() == VEC3_LANES as usize {
            let va = self.vec3_of(out, args, 0)?;
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
        if let Some(fun) = ternary_math(callee, args.len()) {
            let arg = self.operand(out, &args[0])?;
            let arg1 = self.operand(out, &args[1])?;
            let arg2 = self.operand(out, &args[2])?;
            return Ok(self.expr(
                out,
                naga::Expression::Math {
                    fun,
                    arg,
                    arg1: Some(arg1),
                    arg2: Some(arg2),
                    arg3: None,
                },
            ));
        }
        if args.len() == 1 {
            let Some(fun) = unary_math(callee) else {
                return Err(TranslateError::Unsupported);
            };
            let arg = self.operand(out, &args[0])?;
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

    /// `cross3(ax, ay, az, bx, by, bz, lane)` — one lane of the cross product.
    fn cross3_lane(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let lanes = VEC3_LANES as usize;
        if args.len() != lanes + lanes + 1 {
            return Err(TranslateError::Unsupported);
        }
        let lane = const_u32(&args[lanes + lanes])?;
        if lane >= VEC3_LANES {
            return Err(TranslateError::Unsupported);
        }
        let va = self.vec3_of(out, args, 0)?;
        let vb = self.vec3_of(out, args, VEC3_LANES)?;
        let crossed = self.expr(
            out,
            naga::Expression::Math {
                fun: naga::MathFunction::Cross,
                arg: va,
                arg1: Some(vb),
                arg2: None,
                arg3: None,
            },
        );
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: crossed,
                index: lane,
            },
        ))
    }

    /// `normalize3(x, y, z, lane)` — one lane of the unit vector.
    fn normalize3_lane(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let lanes = VEC3_LANES as usize;
        if args.len() != lanes + 1 {
            return Err(TranslateError::Unsupported);
        }
        let lane = const_u32(&args[lanes])?;
        if lane >= VEC3_LANES {
            return Err(TranslateError::Unsupported);
        }
        let va = self.vec3_of(out, args, 0)?;
        let unit = self.expr(
            out,
            naga::Expression::Math {
                fun: naga::MathFunction::Normalize,
                arg: va,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        );
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: unit,
                index: lane,
            },
        ))
    }

    fn vec3_of(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        at: u32,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let start = at as usize;
        let ty = self.vec3_type();
        let x = self.operand(out, args.get(start).ok_or(TranslateError::Unsupported)?)?;
        let y = self.operand(out, args.get(start + 1).ok_or(TranslateError::Unsupported)?)?;
        let z = self.operand(out, args.get(start + 2).ok_or(TranslateError::Unsupported)?)?;
        Ok(self.expr(
            out,
            naga::Expression::Compose {
                ty,
                components: vec![x, y, z],
            },
        ))
    }

    /// Lane of `@builtin(position)`. The lane index is a constant `0..4`.
    fn fragment_position(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let lane = const_u32(args.first().ok_or(TranslateError::Unsupported)?)?;
        if lane >= POSITION_LANES {
            return Err(TranslateError::Unsupported);
        }
        let position = self.position.ok_or(TranslateError::Unsupported)?;
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: position,
                index: lane,
            },
        ))
    }

    /// `workgroup_load_u32` or `workgroup_atomic_load_u32`.
    fn workgroup_load(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        atomic: bool,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let index = workgroup_index(args)?;
        let index = self.operand(out, index)?;
        let pointer = self.workgroup_slot(out, index, atomic)?;
        Ok(self.expr(out, naga::Expression::Load { pointer }))
    }

    /// `workgroup_store_u32` or `workgroup_atomic_store_u32`.
    fn workgroup_store(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        atomic: bool,
    ) -> Result<(), TranslateError> {
        if args.len() != 2 {
            return Err(TranslateError::Unsupported);
        }
        let index = self.operand(out, &args[0])?;
        let value = self.operand(out, &args[1])?;
        let pointer = self.workgroup_slot(out, index, atomic)?;
        out.push(
            naga::Statement::Store { pointer, value },
            naga::Span::default(),
        );
        Ok(())
    }

    fn workgroup_slot(
        &mut self,
        out: &mut naga::Block,
        index: naga::Handle<naga::Expression>,
        atomic: bool,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let var = if atomic {
            self.builders.workgroup_atomics()?
        } else {
            self.builders.workgroup_words()?
        };
        let base = self.expr(out, naga::Expression::GlobalVariable(var));
        Ok(self.expr(out, naga::Expression::Access { base, index }))
    }

    /// `workgroup_atomic_add_u32(index, value)` returns the previous word.
    fn workgroup_atomic_add(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != 2 {
            return Err(TranslateError::Unsupported);
        }
        let index = self.operand(out, &args[0])?;
        let value = self.operand(out, &args[1])?;
        let pointer = self.workgroup_slot(out, index, true)?;
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
        Ok(result)
    }

    /// `texture_load_2d_f32(group, binding, x, y)` at mip 0. Depth textures
    /// return the scalar WGSL `textureLoad` gives them.
    fn texture_load_f32(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let (group, binding, x, y) = texture_xy(args)?;
        let image = self.builders.texture(group, binding, TextureKind::Depth)?;
        self.image_load(out, image, x, y)
    }

    /// `texture_load_2d_f32x4(group, binding, x, y, lane)` at mip 0.
    /// One lane of the `vec4<f32>` a sampled `texture_2d<f32>` returns.
    // qual:allow(iosp) reason: "the lane is checked, then one component of the sampled textureLoad is taken"
    fn texture_load_lane(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != UNIFORM_PREFIX + 1 {
            return Err(TranslateError::Unsupported);
        }
        let lane = const_u32(&args[UNIFORM_PREFIX])?;
        if lane >= VEC4_LANES {
            return Err(TranslateError::Unsupported);
        }
        let (group, binding, x, y) = texture_xy(&args[..UNIFORM_PREFIX])?;
        let image = self
            .builders
            .texture(group, binding, TextureKind::Sampled)?;
        let texel = self.image_load(out, image, x, y)?;
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: texel,
                index: lane,
            },
        ))
    }

    /// `textureLoad` at mip 0. Depth images yield `f32`; sampled images yield `vec4<f32>`.
    fn image_load(
        &mut self,
        out: &mut naga::Block,
        image: naga::Handle<naga::GlobalVariable>,
        x: &Operand,
        y: &Operand,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let coordinate = self.texel_coord(out, x, y)?;
        let level = self.expr(out, naga::Expression::Literal(naga::Literal::I32(0)));
        let handle = self.expr(out, naga::Expression::GlobalVariable(image));
        Ok(self.expr(
            out,
            naga::Expression::ImageLoad {
                image: handle,
                coordinate,
                array_index: None,
                sample: None,
                level: Some(level),
            },
        ))
    }

    /// `texture_width_2d` / `texture_height_2d(group, binding)`.
    // qual:allow(iosp) reason: "the binding is resolved, then one lane of textureDimensions is loaded"
    fn texture_size(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        axis: u32,
        kind: TextureKind,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != 2 {
            return Err(TranslateError::Unsupported);
        }
        let group = const_u32(&args[0])?;
        let binding = const_u32(&args[1])?;
        let image = self.builders.texture(group, binding, kind)?;
        let handle = self.expr(out, naga::Expression::GlobalVariable(image));
        let size = self.expr(
            out,
            naga::Expression::ImageQuery {
                image: handle,
                query: naga::ImageQuery::Size { level: None },
            },
        );
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: size,
                index: axis,
            },
        ))
    }

    fn texel_coord(
        &mut self,
        out: &mut naga::Block,
        x: &Operand,
        y: &Operand,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let x = self.operand(out, x)?;
        let y = self.operand(out, y)?;
        let x = self.expr(
            out,
            naga::Expression::As {
                expr: x,
                kind: naga::ScalarKind::Sint,
                convert: None,
            },
        );
        let y = self.expr(
            out,
            naga::Expression::As {
                expr: y,
                kind: naga::ScalarKind::Sint,
                convert: None,
            },
        );
        let ty = self.builders.ivec2();
        Ok(self.expr(
            out,
            naga::Expression::Compose {
                ty,
                components: vec![x, y],
            },
        ))
    }

    /// `uniform_nested_field_f32/u32(group, binding, sid, field, inner)`.
    // qual:allow(iosp) reason: "the nested tag is checked, then the load is two AccessIndex steps into the uniform global"
    fn uniform_nested_field(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        expect: u8,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let (group, binding, sid, field) = uniform_ids(args)?;
        let inner = const_u32(
            args.get(UNIFORM_PREFIX)
                .ok_or(TranslateError::Unsupported)?,
        )?;
        let tag = self.field_tag(sid, field)?;
        if tag < ornis_gpu_abi::TYPE_TAG_NESTED {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform field {field} of struct {sid} is not a nested struct"
            )));
        }
        let nested = u32::from(tag - ornis_gpu_abi::TYPE_TAG_NESTED);
        let inner_tag = self.field_tag(nested, inner)?;
        if inner_tag != expect {
            return Err(TranslateError::UnsupportedType(format!(
                "nested field {inner} of struct {nested} has tag {inner_tag}, expected {expect}"
            )));
        }
        let global = self.builders.uniform_struct_global(group, binding, sid)?;
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let outer = self.expr(out, naga::Expression::AccessIndex { base, index: field });
        let pointer = self.expr(
            out,
            naga::Expression::AccessIndex {
                base: outer,
                index: inner,
            },
        );
        Ok(self.expr(out, naga::Expression::Load { pointer }))
    }

    fn field_tag(&self, sid: u32, field: u32) -> Result<u8, TranslateError> {
        self.builders
            .structs
            .get(sid as usize)
            .and_then(|tags| tags.get(field as usize))
            .copied()
            .ok_or_else(|| {
                TranslateError::UnsupportedType(format!(
                    "uniform field {field} of struct {sid} is not declared"
                ))
            })
    }

    /// `uniform_field_f32/u32(group, binding, sid, field)`.
    // qual:allow(iosp) reason: "the declared tag is checked against the import, then the field is loaded from the uniform global"
    fn uniform_field(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        expect: u8,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != 4 {
            return Err(TranslateError::Unsupported);
        }
        let (group, binding, sid, field) = uniform_ids(args)?;
        let tag = self.field_tag(sid, field)?;
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

    /// `uniform_field_vec4_plane_dot(group, binding, sid, field, x, y, z)`.
    ///
    /// Loads one `vec4` plane and returns `dot((x, y, z, 1), plane)`, the
    /// signed plane distance. One vector uniform load and one `dot`
    /// replace four scalar uniform loads and the scalar FMA chain.
    // qual:allow(iosp) reason: "the vec4 tag is checked, then one uniform load and a dot replace four scalar loads"
    fn uniform_plane_dot(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != 7 {
            return Err(TranslateError::Unsupported);
        }
        let (group, binding, sid, field) = uniform_ids(args)?;
        let tag = self.field_tag(sid, field)?;
        if tag != ornis_gpu_abi::TYPE_TAG_VEC4 {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform field {field} of struct {sid} is not a vec4"
            )));
        }
        let x = self.operand(out, &args[4])?;
        let y = self.operand(out, &args[5])?;
        let z = self.operand(out, &args[6])?;
        let global = self.builders.uniform_struct_global(group, binding, sid)?;
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let field_ptr = self.expr(out, naga::Expression::AccessIndex { base, index: field });
        let plane = self.expr(out, naga::Expression::Load { pointer: field_ptr });
        let one = self.expr(out, naga::Expression::Literal(naga::Literal::F32(1.0)));
        let center_ty = self.vec4_f32_type();
        let center = self.expr(
            out,
            naga::Expression::Compose {
                ty: center_ty,
                components: vec![x, y, z, one],
            },
        );
        Ok(self.expr(
            out,
            naga::Expression::Math {
                fun: naga::MathFunction::Dot,
                arg: center,
                arg1: Some(plane),
                arg2: None,
                arg3: None,
            },
        ))
    }

    /// `uniform_field_vec4_component_f32(group, binding, sid, field, component)`.
    // qual:allow(iosp) reason: "the vec4 tag is checked, then the lane is one AccessIndex on the loaded field"
    fn uniform_field_vec4_component(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let (group, binding, sid, field) = uniform_ids(args)?;
        let component = const_u32(
            args.get(UNIFORM_PREFIX)
                .ok_or(TranslateError::Unsupported)?,
        )?;
        let tag = self.field_tag(sid, field)?;
        if tag != ornis_gpu_abi::TYPE_TAG_VEC4 {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform field {field} of struct {sid} is not a vec4"
            )));
        }
        let global = self.builders.uniform_struct_global(group, binding, sid)?;
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let vector = self.expr(out, naga::Expression::AccessIndex { base, index: field });
        let lane = self.expr(
            out,
            naga::Expression::AccessIndex {
                base: vector,
                index: component,
            },
        );
        Ok(self.expr(out, naga::Expression::Load { pointer: lane }))
    }

    fn call_void(
        &mut self,
        out: &mut naga::Block,
        callee: &str,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        if callee.contains("workgroup_atomic_store_u32") {
            return self.workgroup_store(out, args, true);
        }
        if callee.contains("workgroup_store_u32") {
            return self.workgroup_store(out, args, false);
        }
        if callee.contains("workgroup_barrier") {
            out.push(
                naga::Statement::ControlBarrier(naga::Barrier::WORK_GROUP),
                naga::Span::default(),
            );
            return Ok(());
        }
        if callee.contains("fragment_store_f32x4") {
            return self.fragment_store(out, args);
        }
        if callee.contains("storage_write_u32x4") {
            return self.storage_write_vec4(out, args);
        }
        if callee.contains("storage_copy_u32x4") {
            return self.storage_copy(out, args);
        }
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

    /// `fragment_store_f32x4(r, g, b, a)` into the `fs_main` output.
    // qual:allow(iosp) reason: "the four lanes are composed into the location-0 vec4 and stored for the fragment return"
    fn fragment_store(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        let var = self.fragment_out.ok_or(TranslateError::Unsupported)?;
        let (r, g, b, a) = rgba(args)?;
        let r = self.operand(out, r)?;
        let g = self.operand(out, g)?;
        let b = self.operand(out, b)?;
        let a = self.operand(out, a)?;
        let ty = self.builders.vec4_f32();
        let value = self.expr(
            out,
            naga::Expression::Compose {
                ty,
                components: vec![r, g, b, a],
            },
        );
        let pointer = self.expr(out, naga::Expression::LocalVariable(var));
        out.push(
            naga::Statement::Store { pointer, value },
            naga::Span::default(),
        );
        Ok(())
    }

    /// `storage_write_u32x4(group, binding, index, x, y, z, w)`.
    fn storage_write_vec4(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        if args.len() != 7 {
            return Err(TranslateError::Unsupported);
        }
        let group = const_u32(&args[0])?;
        let binding = const_u32(&args[1])?;
        let index = self.operand(out, &args[2])?;
        let x = self.operand(out, &args[3])?;
        let y = self.operand(out, &args[4])?;
        let z = self.operand(out, &args[5])?;
        let w = self.operand(out, &args[6])?;
        let vec_ty = self.vec4_u32_type();
        let value = self.expr(
            out,
            naga::Expression::Compose {
                ty: vec_ty,
                components: vec![x, y, z, w],
            },
        );
        let global = self
            .builders
            .storage_global(group, binding, Elem::Vec4U32, true)?;
        let base = self.expr(out, naga::Expression::GlobalVariable(global));
        let pointer = self.expr(out, naga::Expression::Access { base, index });
        out.push(
            naga::Statement::Store { pointer, value },
            naga::Span::default(),
        );
        Ok(())
    }

    /// Bit-preserving copy of one 16-byte record chunk.
    fn storage_copy(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        if args.len() != 6 {
            return Err(TranslateError::Unsupported);
        }
        let src_group = const_u32(&args[0])?;
        let src_binding = const_u32(&args[1])?;
        let src_index = self.operand(out, &args[2])?;
        let dst_group = const_u32(&args[3])?;
        let dst_binding = const_u32(&args[4])?;
        let dst_index = self.operand(out, &args[5])?;
        let src = self
            .builders
            .storage_global(src_group, src_binding, Elem::Vec4U32, false)?;
        let dst = self
            .builders
            .storage_global(dst_group, dst_binding, Elem::Vec4U32, true)?;
        let src_base = self.expr(out, naga::Expression::GlobalVariable(src));
        let src_ptr = self.expr(
            out,
            naga::Expression::Access {
                base: src_base,
                index: src_index,
            },
        );
        let value = self.expr(out, naga::Expression::Load { pointer: src_ptr });
        let dst_base = self.expr(out, naga::Expression::GlobalVariable(dst));
        let dst_ptr = self.expr(
            out,
            naga::Expression::Access {
                base: dst_base,
                index: dst_index,
            },
        );
        out.push(
            naga::Statement::Store {
                pointer: dst_ptr,
                value,
            },
            naga::Span::default(),
        );
        Ok(())
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
        let global = self.builders.storage_global(group, binding, elem, true)?;
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

    fn vec4_f32_type(&mut self) -> naga::Handle<naga::Type> {
        self.builders.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Quad,
                    scalar: naga::Scalar {
                        kind: naga::ScalarKind::Float,
                        width: SCALAR_WIDTH,
                    },
                },
            },
            naga::Span::default(),
        )
    }

    fn vec4_u32_type(&mut self) -> naga::Handle<naga::Type> {
        self.builders.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Quad,
                    scalar: naga::Scalar {
                        kind: naga::ScalarKind::Uint,
                        width: SCALAR_WIDTH,
                    },
                },
            },
            naga::Span::default(),
        )
    }

    fn vec3_type(&mut self) -> naga::Handle<naga::Type> {
        self.builders.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Tri,
                    scalar: naga::Scalar {
                        kind: naga::ScalarKind::Float,
                        width: SCALAR_WIDTH,
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

/// `llvm.umin` / `minnum` / `umax` / `maxnum` — the forms rustc emits for
/// `Ord::min`/`max`. WGSL `min`/`max` have the same NaN rule as `minnum`.
fn image_class(kind: TextureKind) -> naga::ImageClass {
    match kind {
        TextureKind::Depth => naga::ImageClass::Depth { multi: false },
        TextureKind::Sampled => naga::ImageClass::Sampled {
            kind: naga::ScalarKind::Float,
            multi: false,
        },
    }
}

fn ternary_math(callee: &str, argc: usize) -> Option<naga::MathFunction> {
    if argc != VEC3_LANES as usize {
        return None;
    }
    if callee.contains("clamp") {
        Some(naga::MathFunction::Clamp)
    } else if callee.contains("mix") {
        Some(naga::MathFunction::Mix)
    } else {
        None
    }
}

fn unary_math(callee: &str) -> Option<naga::MathFunction> {
    // `acos` before `cos`, `log2` before `log`: the shorter name is a substring.
    const FNS: &[(&str, naga::MathFunction)] = &[
        ("sqrt", naga::MathFunction::Sqrt),
        ("acos", naga::MathFunction::Acos),
        ("sin", naga::MathFunction::Sin),
        ("cos", naga::MathFunction::Cos),
        ("exp", naga::MathFunction::Exp),
        ("log2", naga::MathFunction::Log2),
        ("log", naga::MathFunction::Log),
        ("sign", naga::MathFunction::Sign),
        ("abs", naga::MathFunction::Abs),
    ];
    FNS.iter()
        .find(|(name, _)| callee.contains(name))
        .map(|(_, fun)| *fun)
}

fn min_max(callee: &str) -> Option<naga::MathFunction> {
    if callee.contains("llvm.umin")
        || callee.contains("llvm.smin")
        || callee.contains("llvm.minnum")
    {
        Some(naga::MathFunction::Min)
    } else if callee.contains("llvm.umax")
        || callee.contains("llvm.smax")
        || callee.contains("llvm.maxnum")
    {
        Some(naga::MathFunction::Max)
    } else {
        None
    }
}

/// Lanes of `@builtin(position)`.
const POSITION_LANES: u32 = 4;

/// How many leading operands every uniform import uses: group, binding, struct, field.
const UNIFORM_PREFIX: usize = 4;

fn workgroup_index(args: &[Operand]) -> Result<&Operand, TranslateError> {
    args.first().ok_or(TranslateError::Unsupported)
}

/// `fragment_store_f32x4(r, g, b, a)`.
fn rgba(args: &[Operand]) -> Result<(&Operand, &Operand, &Operand, &Operand), TranslateError> {
    if args.len() != UNIFORM_PREFIX {
        return Err(TranslateError::Unsupported);
    }
    Ok((&args[0], &args[1], &args[2], &args[3]))
}

/// `texture_load_2d_f32(group, binding, x, y)`.
fn texture_xy(args: &[Operand]) -> Result<(u32, u32, &Operand, &Operand), TranslateError> {
    if args.len() != UNIFORM_PREFIX {
        return Err(TranslateError::Unsupported);
    }
    Ok((
        const_u32(&args[0])?,
        const_u32(&args[1])?,
        &args[2],
        &args[3],
    ))
}

/// `(group, binding, struct id, field)` shared by every uniform import.
fn uniform_ids(args: &[Operand]) -> Result<(u32, u32, u32, u32), TranslateError> {
    if args.len() < UNIFORM_PREFIX {
        return Err(TranslateError::Unsupported);
    }
    Ok((
        const_u32(&args[0])?,
        const_u32(&args[1])?,
        const_u32(&args[2])?,
        const_u32(&args[3])?,
    ))
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
            inner: naga::TypeInner::Scalar(naga::Scalar {
                kind,
                width: SCALAR_WIDTH,
            }),
        },
        naga::Span::default(),
    )
}

fn invocation_x(
    module: &mut naga::Module,
    function: &mut naga::Function,
) -> naga::Handle<naga::Expression> {
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
    id_x
}

fn position_arg(
    ty: naga::Handle<naga::Type>,
    function: &mut naga::Function,
) -> naga::Handle<naga::Expression> {
    function.arguments.push(naga::FunctionArgument {
        name: Some("position".to_string()),
        ty,
        binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::Position {
            invariant: false,
        })),
    });
    append(function, naga::Expression::FunctionArgument(0))
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
        let module = lower(&func, structs, ornis_gpu_abi::symbols::CS_WORKGROUP).expect("lower");
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
