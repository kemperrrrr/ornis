//! Lower a parsed LLVM function to a Naga entry point.
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
//!
//! Raster entries (`vs_main`, `fs_main`) have no parameters and return `void`:
//! their builtins are `ornis_gpu` imports (`vertex_index`, `fragment_input_lane`)
//! and their outputs are stores into a stage-output local the lowerer returns.
//! The interface is read back from those imports, so a `@location` exists in
//! the emitted WGSL only if the body uses it, and its `vecN` width is the
//! highest lane the body reads or writes. A vertex entry always returns a
//! struct with `@builtin(position)`; a fragment entry keeps the bare
//! `@location(0) vec4<f32>` result when that is its only target and returns a
//! struct otherwise.

use crate::error::TranslateError;
use crate::parse::{BasicBlock, Instr, LlvmFunc, Operand, PtrBase, Terminator, Ty};
use std::collections::{BTreeMap, HashMap, HashSet};

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

/// One Naga module under construction: entries share its globals.
///
/// Entry shapes, per pushed function:
/// - `cs_main(i32) -> float`: result is stored to `storage_0_0[id]`.
/// - `cs_main(i32) -> void` / `cs_main() -> void`: the body stores through ABI
///   calls.
/// - `vs_main() -> void` / `fs_main() -> void`: builtins and stage interface
///   values come from ABI calls, and the lowerer returns the output it stored.
///
/// Resources are cached per `(group, binding)` in [`Builders`], so a vertex and
/// a fragment entry of the same pass share one set of globals — which is what a
/// pipeline needs: both stages read the same bind group. Each call to
/// [`Self::push`] adds one entry point.
pub(super) struct Lowering {
    builders: Builders,
}

impl Lowering {
    /// Start a module with the given uniform-struct tables.
    pub(super) fn new(structs: &[&[u8]], sampling: crate::TextureSampling) -> Self {
        Self {
            builders: Builders::new(structs, sampling),
        }
    }

    /// Remember the functions the parser skipped, so a call that reaches one
    /// reports what is wrong with it instead of a generic rejection.
    /// Remember the functions that never return, so their call sites can be
    /// dropped instead of translated.
    pub(super) fn reserve_diverging(&mut self, funcs: &[LlvmFunc]) {
        for func in funcs {
            if !func
                .blocks
                .iter()
                .any(|block| matches!(block.term, Terminator::Ret(_)))
            {
                self.builders.diverging.insert(func.name.clone());
            }
        }
    }

    /// Remember the declared functions LLVM marks `noreturn`: a call to one
    /// never returns, so its call site is dropped.
    pub(super) fn reserve_noreturn(&mut self, names: Vec<String>) {
        self.builders.diverging.extend(names);
    }

    pub(super) fn reserve_skipped(&mut self, skipped: Vec<(String, String)>) {
        for (name, reason) in skipped {
            self.builders.skipped_helpers.insert(name, reason);
        }
    }

    /// Lower every helper the entries can call.
    ///
    /// Helpers may call each other, so they are lowered in dependency order:
    /// each round lowers the functions whose callees are already in the module.
    /// A helper whose body is outside the subset (the panic shim takes a `ptr`,
    /// for instance) is skipped rather than failing the module: only a call that
    /// actually reaches it is an error. A cycle is never resolved, so recursion
    /// is rejected.
    ///
    /// # Errors
    ///
    /// [`TranslateError::Unsupported`] when the helpers call each other in a
    /// cycle, which a shader cannot express.
    pub(super) fn add_helpers(&mut self, funcs: &[LlvmFunc]) -> Result<(), TranslateError> {
        let mut pending: Vec<&LlvmFunc> = funcs
            .iter()
            .filter(|func| Stage::of(&func.name).is_none())
            .collect();
        let mut skipped: Vec<&str> = Vec::new();
        while !pending.is_empty() {
            let before = pending.len();
            let mut deferred = Vec::new();
            for func in pending {
                if self.callees_ready(func, &skipped) {
                    self.lower_or_skip(func, &mut skipped);
                } else {
                    deferred.push(func);
                }
            }
            if deferred.len() == before {
                return Err(TranslateError::Unsupported);
            }
            pending = deferred;
        }
        Ok(())
    }

    /// Lower `func`, or remember it as unusable when its body leaves the subset.
    fn lower_or_skip<'f>(&mut self, func: &'f LlvmFunc, skipped: &mut Vec<&'f str>) {
        let lowered = self.lower_helper(func);
        if lowered.is_err() {
            skipped.push(func.name.as_str());
        }
    }

    /// Whether every helper `func` calls is already lowered or skipped.
    fn callees_ready(&self, func: &LlvmFunc, skipped: &[&str]) -> bool {
        func.blocks.iter().all(|block| {
            block.instrs.iter().all(|instr| match instr {
                Instr::Call { callee, .. } | Instr::CallVoid { callee, .. } => {
                    !is_helper_name(callee)
                        || self.builders.helpers.contains_key(callee)
                        || skipped.contains(&callee.as_str())
                }
                _ => true,
            })
        })
    }

    /// Lower one helper into a Naga function.
    fn lower_helper(&mut self, func: &LlvmFunc) -> Result<(), TranslateError> {
        let cfg = Cfg::new(func)?;
        let function = self.builders.build_helper(func, &cfg)?;
        let handle = self
            .builders
            .module
            .functions
            .append(function, naga::Span::default());
        self.builders.helpers.insert(
            func.name.clone(),
            Helper {
                function: handle,
                ret: func.ret_ty,
            },
        );
        Ok(())
    }

    /// Lower one entry function into the module.
    ///
    /// `workgroup` is the `@workgroup_size` of a compute entry and ignored by a
    /// raster one.
    ///
    /// # Errors
    ///
    /// [`TranslateError::Unsupported`] for an entry shape outside the subset
    /// (an unknown name, a raster entry with parameters, a compute entry that
    /// is neither `(i32)` nor `()`), and whatever the body lowers to.
    pub(super) fn push(
        &mut self,
        func: &LlvmFunc,
        workgroup: [u32; 3],
    ) -> Result<(), TranslateError> {
        let stage = entry_stage(func)?;
        self.lower_entry(func, stage, workgroup)
    }

    /// Structure the body and register the entry point.
    fn lower_entry(
        &mut self,
        func: &LlvmFunc,
        stage: Stage,
        workgroup: [u32; 3],
    ) -> Result<(), TranslateError> {
        let cfg = Cfg::new(func)?;
        let function = self.builders.build(func, &cfg, stage)?;
        self.builders
            .finish(function, func.ret_ty, workgroup, stage)
    }

    /// The finished module, with every entry point pushed so far.
    pub(super) fn finish(self) -> naga::Module {
        self.builders.module
    }
}

/// Shader stage the entry function belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// `cs_main`: one invocation per `@workgroup_size` thread.
    Compute,
    /// `vs_main`: one invocation per vertex.
    Vertex,
    /// `fs_main`: one invocation per covered sample.
    Fragment,
}

impl Stage {
    /// Stage an entry name denotes, or `None` for anything else.
    fn of(name: &str) -> Option<Self> {
        match name {
            ornis_gpu_abi::symbols::CS_MAIN => Some(Self::Compute),
            ornis_gpu_abi::symbols::VS_MAIN => Some(Self::Vertex),
            ornis_gpu_abi::symbols::FS_MAIN => Some(Self::Fragment),
            _ => None,
        }
    }

    /// Whether the stage draws fragments, i.e. has a stage interface.
    fn raster(self) -> bool {
        !matches!(self, Self::Compute)
    }
}

/// Slot of a stage output: the clip position or one `@location`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum OutSlot {
    /// `@builtin(position)` of a vertex entry.
    Position,
    /// `@location(n)` of a vertex or fragment entry.
    Location(u32),
}

/// `@location` inputs and outputs the entry body touches.
///
/// The subset has no Rust type for an interface value, so the stage interface
/// is read back from the `ornis_gpu` imports the body calls: a `@location` no
/// import mentions is not declared, and the highest lane an import names sets
/// the declared `vecN<f32>` width.
#[derive(Default)]
struct StageIo {
    /// `@builtin(vertex_index)` is read.
    vertex_index: bool,
    /// `@builtin(instance_index)` is read.
    instance_index: bool,
    /// `@builtin(position)` of a vertex entry is written.
    position_out: bool,
    /// Declared inputs: location -> lanes.
    inputs: BTreeMap<u32, u32>,
    /// Written outputs: location -> lanes.
    outputs: BTreeMap<u32, u32>,
}

/// The stage-interface import a call names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IoImport {
    /// `@builtin(vertex_index)`.
    VertexIndex,
    /// `@builtin(instance_index)`.
    InstanceIndex,
    /// Lane of a vertex `@location` input.
    VertexInput,
    /// Clip-space `@builtin(position)` of a vertex entry.
    VertexPosition,
    /// Two-lane vertex `@location` output.
    VertexOutput2,
    /// Four-lane vertex `@location` output.
    VertexOutput4,
    /// `@builtin(position)` of a fragment entry.
    FragmentPosition,
    /// Lane of a fragment `@location` input.
    FragmentInput,
    /// Lane of the fragment's `@location(0)` vec2, the AO-pass shape.
    FragmentUv,
    /// Fragment output at a named location.
    FragmentOutput,
    /// Fragment output at `@location(0)`, the bare-vec4 shape.
    FragmentOutput0,
}

/// Order matters only where one import name is a prefix of another, which is
/// why the located fragment output precedes the `@location(0)` one.
const IO_IMPORTS: [(&str, IoImport); 11] = [
    (
        ornis_gpu_abi::symbols::VERTEX_INPUT_LANE,
        IoImport::VertexInput,
    ),
    (
        ornis_gpu_abi::symbols::VERTEX_STORE_F32X2,
        IoImport::VertexOutput2,
    ),
    (
        ornis_gpu_abi::symbols::VERTEX_STORE_F32X4,
        IoImport::VertexOutput4,
    ),
    (
        ornis_gpu_abi::symbols::VERTEX_POSITION,
        IoImport::VertexPosition,
    ),
    (ornis_gpu_abi::symbols::VERTEX_INDEX, IoImport::VertexIndex),
    (
        ornis_gpu_abi::symbols::INSTANCE_INDEX,
        IoImport::InstanceIndex,
    ),
    (
        ornis_gpu_abi::symbols::FRAGMENT_STORE_LOCATION,
        IoImport::FragmentOutput,
    ),
    (
        ornis_gpu_abi::symbols::FRAGMENT_STORE_F32X4,
        IoImport::FragmentOutput0,
    ),
    (
        ornis_gpu_abi::symbols::FRAGMENT_INPUT_LANE,
        IoImport::FragmentInput,
    ),
    (
        ornis_gpu_abi::symbols::FRAGMENT_UV_LANE,
        IoImport::FragmentUv,
    ),
    (
        ornis_gpu_abi::symbols::FRAGMENT_POSITION_LANE,
        IoImport::FragmentPosition,
    ),
];

/// Import one callee names, if it is a stage-interface import at all.
fn io_import(callee: &str) -> Option<IoImport> {
    IO_IMPORTS
        .iter()
        .find(|(name, _)| callee.contains(name))
        .map(|(_, import)| *import)
}

/// Read the stage interface back out of the body's imports.
fn scan_io(func: &LlvmFunc, stage: Stage) -> Result<StageIo, TranslateError> {
    let mut io = StageIo::default();
    for block in &func.blocks {
        for instr in &block.instrs {
            let (callee, args) = match instr {
                Instr::Call { callee, args, .. } | Instr::CallVoid { callee, args } => {
                    (callee.as_str(), args.as_slice())
                }
                _ => continue,
            };
            let Some(import) = io_import(callee) else {
                continue;
            };
            match (stage, import) {
                (Stage::Vertex, IoImport::VertexIndex) => io.vertex_index = true,
                (Stage::Vertex, IoImport::InstanceIndex) => io.instance_index = true,
                (Stage::Vertex, IoImport::VertexInput) => {
                    let (location, lanes) = input_lanes(args)?;
                    record_lanes(&mut io.inputs, location, lanes)?;
                }
                (Stage::Vertex, IoImport::VertexPosition) => io.position_out = true,
                (Stage::Vertex, IoImport::VertexOutput2 | IoImport::VertexOutput4) => {
                    let lanes = if import == IoImport::VertexOutput2 {
                        VEC2_LANES
                    } else {
                        VEC4_LANES
                    };
                    record_lanes(&mut io.outputs, location_of(args)?, lanes)?;
                }
                // The fragment's `@builtin(position)` argument is always
                // declared, so reading it needs no interface bookkeeping.
                (Stage::Fragment, IoImport::FragmentPosition) => {}
                (Stage::Fragment, IoImport::FragmentInput) => {
                    let (location, lanes) = input_lanes(args)?;
                    record_lanes(&mut io.inputs, location, lanes)?;
                }
                (Stage::Fragment, IoImport::FragmentUv) => {
                    record_lanes(&mut io.inputs, 0, VEC2_LANES)?;
                }
                (Stage::Fragment, IoImport::FragmentOutput0) => {
                    record_lanes(&mut io.outputs, 0, VEC4_LANES)?;
                }
                (Stage::Fragment, IoImport::FragmentOutput) => {
                    record_lanes(&mut io.outputs, location_of(args)?, VEC4_LANES)?;
                }
                // A stage reading the other stage's interface would declare an
                // argument its entry point cannot have.
                _ => return Err(TranslateError::Unsupported),
            }
        }
    }
    Ok(io)
}

/// Operands of a `*_input_lane` import: location, declared lanes, and the lane.
const INPUT_LANE_ARGS: usize = 3;

/// `(location, lanes)` immediates of a `*_input_lane` call.
///
/// The lane being read must also be inside the declared width; that check is
/// in [`Lowerer::input_lane`], which sees the same immediates.
fn input_lanes(args: &[Operand]) -> Result<(u32, u32), TranslateError> {
    if args.len() != INPUT_LANE_ARGS {
        return Err(TranslateError::Unsupported);
    }
    Ok((const_u32(&args[0])?, const_u32(&args[1])?))
}

/// Location immediate of a `*_store_*(location, …)` call.
fn location_of(args: &[Operand]) -> Result<u32, TranslateError> {
    const_u32(args.first().ok_or(TranslateError::Unsupported)?)
}

/// Record that `location` carries `lanes` lanes.
fn record_lanes(
    map: &mut BTreeMap<u32, u32>,
    location: u32,
    lanes: u32,
) -> Result<(), TranslateError> {
    if !(1..=VEC4_LANES).contains(&lanes) {
        return Err(TranslateError::UnsupportedType(format!(
            "@location({location}) needs one to four lanes, got {lanes}"
        )));
    }
    match map.get(&location) {
        Some(&existing) if existing != lanes => Err(TranslateError::UnsupportedType(format!(
            "@location({location}) is used as both a {existing}-lane and a {lanes}-lane value"
        ))),
        Some(_) => Ok(()),
        None => {
            map.insert(location, lanes);
            Ok(())
        }
    }
}

/// `@location(n)` with the perspective interpolation every pass uses today.
fn location_binding(location: u32) -> naga::Binding {
    naga::Binding::Location {
        location,
        interpolation: Some(naga::Interpolation::Perspective),
        sampling: Some(naga::Sampling::Center),
        blend_src: None,
        per_primitive: false,
    }
}

/// Stage-output struct under construction: members, their layout, and the
/// member index of every output slot.
#[derive(Default)]
struct OutputLayout {
    /// Struct members in declaration order.
    members: Vec<naga::StructMember>,
    /// Size and alignment of each member, in the same order.
    specs: Vec<ornis_gpu_abi::FieldLayout>,
    /// Member index per output slot.
    slots: BTreeMap<OutSlot, u32>,
}

impl OutputLayout {
    /// Append the member for `slot`: `lanes` of `ty` at `binding`.
    fn push(
        &mut self,
        slot: OutSlot,
        lanes: u32,
        ty: naga::Handle<naga::Type>,
        binding: naga::Binding,
    ) {
        let name = match slot {
            OutSlot::Position => "position".to_string(),
            OutSlot::Location(location) => format!("o{location}"),
        };
        self.specs.push(ornis_gpu_abi::FieldLayout {
            size: lanes * WORD_BYTES,
            align: lanes * WORD_BYTES,
        });
        self.slots.insert(slot, self.members.len() as u32);
        self.members.push(naga::StructMember {
            name: Some(name),
            ty,
            binding: Some(binding),
            offset: 0,
        });
    }

    /// Size of the struct with every member offset filled in.
    fn span(&mut self) -> u32 {
        let mut offsets = vec![0u32; self.specs.len()];
        let span = ornis_gpu_abi::place_fields(&self.specs, true, &mut offsets);
        for (member, offset) in self.members.iter_mut().zip(&offsets) {
            member.offset = *offset;
        }
        span
    }
}

/// Name the emitted stage-output struct carries in WGSL.
fn output_struct_name(stage: Stage) -> &'static str {
    if stage == Stage::Vertex {
        "VertexOut"
    } else {
        "FragmentOut"
    }
}

/// A raster entry's result type, its located members, and `@builtin(position)`.
struct OutputType {
    /// The emitted struct type.
    ty: naga::Handle<naga::Type>,
    /// Member index per output slot.
    slots: BTreeMap<OutSlot, u32>,
    /// Member index of `@builtin(position)`, absent for a fragment.
    position: Option<u32>,
}

/// Entry arguments a raster or compute body reads.
struct StageArguments {
    /// `global_invocation_id` vector, for a compute entry.
    global_id: Option<naga::Handle<naga::Expression>>,
    /// Lane 0 of `global_invocation_id`, the compute entry's `i32` parameter.
    id_x: Option<naga::Handle<naga::Expression>>,
    /// `@builtin(local_invocation_index)`.
    local_index: Option<naga::Handle<naga::Expression>>,
    /// `@builtin(vertex_index)`.
    vertex_index: Option<naga::Handle<naga::Expression>>,
    /// `@builtin(instance_index)`.
    instance_index: Option<naga::Handle<naga::Expression>>,
    /// `@builtin(position)` of a fragment entry.
    position: Option<naga::Handle<naga::Expression>>,
    /// `@location` arguments by location.
    inputs: BTreeMap<u32, naga::Handle<naga::Expression>>,
}

/// Local variable the entry returns, holding the stage output.
fn output_local(
    function: &mut naga::Function,
    name: &str,
    ty: naga::Handle<naga::Type>,
) -> naga::Handle<naga::LocalVariable> {
    function.local_variables.append(
        naga::LocalVariable {
            name: Some(name.to_string()),
            ty,
            init: None,
        },
        naga::Span::default(),
    )
}

/// Operands of a `memset`/`memcpy` call: destination/source, value, size.
const MEMORY_INTRINSIC_ARGS: usize = 3;

/// Bits in one byte, the unit a `memset` fill value repeats with.
const BYTE_BITS: u32 = 8;
/// Bits in one word.
const WORD_BITS: u32 = 32;
/// Mask of one byte.
const BYTE_MASK: u32 = 0xFF;

/// Whether `callee` names a function the module defines rather than an import.
///
/// Imports are the `ornis_gpu` ABI and the `llvm.*` intrinsics; everything else
/// is a helper the kernel calls.
fn is_helper_name(callee: &str) -> bool {
    const IMPORT_PREFIXES: [&str; 8] = [
        "llvm.",
        "ornis_gpu",
        "storage_",
        "uniform_",
        "texture_",
        "workgroup_",
        "fragment_",
        "vertex_",
    ];
    const IMPORT_NAMES: [&str; 3] = ["instance_index", "gpu_", "global_invocation"];
    !IMPORT_PREFIXES
        .iter()
        .chain(IMPORT_NAMES.iter())
        .any(|prefix| callee.contains(prefix))
        && !callee.contains("local_invocation")
}

/// A helper function lowered into the module.
#[derive(Clone, Copy)]
struct Helper {
    /// The Naga function to call.
    function: naga::Handle<naga::Function>,
    /// Its result type, `Ty::Void` for a function that returns nothing.
    ret: Ty,
}

/// Naga scalar type of a helper's `f32`/`i32` value.
fn scalar_of(module: &mut naga::Module, ty: Ty) -> Option<naga::Handle<naga::Type>> {
    match ty {
        Ty::Float => Some(scalar(module, naga::ScalarKind::Float)),
        Ty::I32 => Some(scalar(module, naga::ScalarKind::Uint)),
        _ => None,
    }
}

/// One word made of `byte` repeated, the word a `memset` writes.
fn repeated_byte(byte: u32) -> u32 {
    let mut word = 0u32;
    let mut shift = 0u32;
    while shift < WORD_BITS {
        word |= byte << shift;
        shift += BYTE_BITS;
    }
    word
}

/// Word count of `bytes`, which must be a multiple of four.
fn word_count(bytes: u32) -> Result<u32, TranslateError> {
    if bytes == 0 || !bytes.is_multiple_of(WORD_BYTES) {
        return Err(TranslateError::Unsupported);
    }
    Ok(bytes / WORD_BYTES)
}

/// Where a memory value lives and how far into it the pointer is.
#[derive(Clone, Copy)]
struct MemoryPointer {
    /// Base of the pointer.
    base: MemoryBase,
    /// Offset from the base, in 4-byte words.
    words: naga::Handle<naga::Expression>,
}

/// What a [`MemoryPointer`] starts at.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MemoryBase {
    /// A function-scope allocation (`alloca`).
    Local(naga::Handle<naga::LocalVariable>),
    /// A module constant, lowered to a private array of words.
    Constant(naga::Handle<naga::GlobalVariable>),
}

/// Where the entry writes its stage output.
struct StageOut {
    /// Local variable the entry returns.
    var: naga::Handle<naga::LocalVariable>,
    /// Struct member per output slot; `None` when the output is the fragment's
    /// bare `@location(0) vec4<f32>`.
    locations: Option<BTreeMap<OutSlot, u32>>,
    /// Struct member of `@builtin(position)`.
    position: Option<u32>,
}

impl StageOut {
    /// The fragment's bare `@location(0) vec4<f32>`, stored into the variable.
    fn bare(var: naga::Handle<naga::LocalVariable>) -> Self {
        Self {
            var,
            locations: None,
            position: None,
        }
    }

    /// A struct result, stored into the member each output slot names.
    fn located(
        var: naga::Handle<naga::LocalVariable>,
        locations: BTreeMap<OutSlot, u32>,
        position: Option<u32>,
    ) -> Self {
        Self {
            var,
            locations: Some(locations),
            position,
        }
    }
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
            let out = successors(&block.term);
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
        Terminator::Switch { selector, .. } => push_local(selector, used),
        Terminator::Ret(Some(value)) => push_local(value, used),
        Terminator::BrUncond(_) | Terminator::Ret(None) | Terminator::Unreachable => {}
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
    /// Filtering samplers, one per binding.
    samplers: HashMap<(u32, u32), naga::Handle<naga::GlobalVariable>>,
    /// `vec2<i32>` used as a `textureLoad` coordinate.
    ivec2: Option<naga::Handle<naga::Type>>,
    /// Plain `array<u32, WORKGROUP_WORDS>` in workgroup memory.
    workgroup: Option<naga::Handle<naga::GlobalVariable>>,
    /// `array<atomic<u32>, WORKGROUP_WORDS>` in workgroup memory.
    workgroup_atomic: Option<naga::Handle<naga::GlobalVariable>>,
    /// Module constants by symbol name, raw little-endian contents.
    module_constants: HashMap<String, Vec<u8>>,
    /// Helper functions lowered into the module, by IR name.
    helpers: HashMap<String, Helper>,
    /// Helpers outside the subset, with the reason a call cannot lower them.
    skipped_helpers: HashMap<String, String>,
    /// Functions whose every path ends in `unreachable` (the panic shims).
    ///
    /// A call to one never returns, so the translator drops it and lets the
    /// caller's own `unreachable` end the invocation: a shader has no place to
    /// report a panic to.
    diverging: HashSet<String>,
    /// `ornis_gpu.types` field tags per struct id.
    structs: Vec<Vec<u8>>,
    /// Which textures declare the multisampled type (and load sample 0).
    sampling: crate::TextureSampling,
}

impl Builders {
    fn new(structs: &[&[u8]], sampling: crate::TextureSampling) -> Self {
        Self {
            module: naga::Module::default(),
            storage_binding: HashMap::new(),
            uniform_cache: HashMap::new(),
            uniform_struct_cache: HashMap::new(),
            struct_types: HashMap::new(),
            struct_stack: Vec::new(),
            textures: HashMap::new(),
            samplers: HashMap::new(),
            ivec2: None,
            workgroup: None,
            workgroup_atomic: None,
            module_constants: HashMap::new(),
            helpers: HashMap::new(),
            skipped_helpers: HashMap::new(),
            diverging: HashSet::new(),
            structs: structs.iter().map(|tags| tags.to_vec()).collect(),
            sampling,
        }
    }

    /// Add the lowered function as this module's next entry point.
    fn finish(
        &mut self,
        function: naga::Function,
        ret_ty: Ty,
        workgroup: [u32; 3],
        stage: Stage,
    ) -> Result<(), TranslateError> {
        // Only create the implicit `storage_0_0` global when the kernel
        // uses the `-> float` shape; for `-> void` the body creates its
        // own globals lazily.
        if ret_ty == Ty::Float {
            self.storage_global(0, 0, Elem::F32, true)?;
        }
        let (name, stage, workgroup_size) = match stage {
            Stage::Compute => (
                ornis_gpu_abi::symbols::CS_MAIN.to_string(),
                naga::ShaderStage::Compute,
                workgroup,
            ),
            Stage::Vertex => (
                ornis_gpu_abi::symbols::VS_MAIN.to_string(),
                naga::ShaderStage::Vertex,
                [0, 0, 0],
            ),
            Stage::Fragment => (
                ornis_gpu_abi::symbols::FS_MAIN.to_string(),
                naga::ShaderStage::Fragment,
                [0, 0, 0],
            ),
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
        Ok(())
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
                    class: image_class(want, self.sampling.multisampled_at(group, binding)),
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

    /// A filtering `sampler` global at `(group, binding)`.
    ///
    /// Samplers carry no data the kernel can read, so nothing checks the
    /// address or filter modes here; the renderer's bind group owns them.
    fn sampler(&mut self, group: u32, binding: u32) -> naga::Handle<naga::GlobalVariable> {
        if let Some(handle) = self.samplers.get(&(group, binding)) {
            return *handle;
        }
        let ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Sampler { comparison: false },
            },
            naga::Span::default(),
        );
        let handle = self.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(format!("sampler_{group}_{binding}_")),
                space: naga::AddressSpace::Handle,
                binding: Some(naga::ResourceBinding { group, binding }),
                ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        );
        self.samplers.insert((group, binding), handle);
        handle
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
        stage: Stage,
    ) -> Result<naga::Function, TranslateError> {
        let mut function = naga::Function {
            name: Some(func.name.clone()),
            ..Default::default()
        };
        let io = scan_io(func, stage)?;
        let uniform = uniform_values(func);
        self.module_constants = func
            .globals
            .iter()
            .map(|global| (global.name.clone(), global.bytes.clone()))
            .collect();
        let args = stage_arguments(&mut self.module, func, stage, &io, &mut function)?;
        let stage_out = stage_output(&mut self.module, stage, &io, &mut function)?;
        let mut lowerer = Lowerer {
            builders: self,
            cfg,
            func: function,
            io,
            id_x: args.id_x,
            global_id: args.global_id,
            local_index: args.local_index,
            vertex_index: args.vertex_index,
            instance_index: args.instance_index,
            stage_out,
            position: args.position,
            inputs: args.inputs,
            phi_vars: HashMap::new(),
            spill: HashMap::new(),
            env: HashMap::new(),
            saw_return: false,
            divergent_depth: 0,
            uniform,
            exit_ids: HashMap::new(),
            exit_id_var: None,
            pointers: HashMap::new(),
            ptr_bases: HashMap::new(),
            constants: HashMap::new(),
            diverged: false,
            returns_value: false,
        };
        if let Some((id_x, (_, name))) = args.id_x.zip(func.params.first()) {
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

    /// Lower a helper function: scalar parameters, scalar or void result.
    ///
    /// A helper has no stage interface and no invocation id; its body is the
    /// same subset as an entry's, and `ret` returns the value instead of
    /// storing it.
    fn build_helper(
        &mut self,
        func: &LlvmFunc,
        cfg: &Cfg<'_>,
    ) -> Result<naga::Function, TranslateError> {
        let mut function = naga::Function {
            name: Some(func.name.clone()),
            ..Default::default()
        };
        let mut args = Vec::new();
        for (index, (ty, name)) in func.params.iter().enumerate() {
            let ty = match ty {
                Ty::Float => scalar(&mut self.module, naga::ScalarKind::Float),
                Ty::I32 => scalar(&mut self.module, naga::ScalarKind::Uint),
                _ => return Err(TranslateError::Unsupported),
            };
            function.arguments.push(naga::FunctionArgument {
                name: Some(name.clone()),
                ty,
                binding: None,
            });
            let arg = append(
                &mut function,
                naga::Expression::FunctionArgument(index as u32),
            );
            push_emit(&mut function, &mut naga::Block::new(), arg);
            args.push((name.clone(), arg));
        }
        function.result = match func.ret_ty {
            Ty::Void => None,
            ret => Some(naga::FunctionResult {
                ty: scalar_of(&mut self.module, ret).ok_or(TranslateError::Unsupported)?,
                binding: None,
            }),
        };
        let mut lowerer = Lowerer {
            builders: self,
            cfg,
            func: function,
            io: StageIo::default(),
            id_x: None,
            global_id: None,
            local_index: None,
            vertex_index: None,
            instance_index: None,
            stage_out: None,
            position: None,
            inputs: BTreeMap::new(),
            phi_vars: HashMap::new(),
            spill: HashMap::new(),
            env: HashMap::new(),
            saw_return: false,
            divergent_depth: 0,
            uniform: HashSet::new(),
            exit_ids: HashMap::new(),
            exit_id_var: None,
            pointers: HashMap::new(),
            ptr_bases: HashMap::new(),
            constants: HashMap::new(),
            diverged: false,
            returns_value: true,
        };
        for (name, arg) in args {
            lowerer.env.insert(name, arg);
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

/// A `loop` with no continuing block and no `break if`.
fn loop_statement(body: naga::Block) -> naga::Statement {
    naga::Statement::Loop {
        body,
        continuing: naga::Block::new(),
        break_if: None,
    }
}

/// One `switch` case; the value comes from the IR, the body was just emitted.
fn switch_case(
    value: naga::SwitchValue,
    body: naga::Block,
    fall_through: bool,
) -> naga::SwitchCase {
    naga::SwitchCase {
        value,
        body,
        fall_through,
    }
}

/// Whether a `switch` case returns, after rejecting flows that cannot surface.
///
/// A case either falls through to the merge or returns from the entry; a
/// `continue`/`break` belongs to a region inside the case.
fn case_flow_returns(flow: Flow) -> Result<bool, TranslateError> {
    match flow {
        Flow::Returned => Ok(true),
        Flow::Next => Ok(false),
        Flow::Continue | Flow::Break => Err(TranslateError::Unsupported),
    }
}

/// Values that are the same for every invocation of the workgroup.
///
/// Greatest fixpoint: every value starts uniform and is dropped as soon as it
/// depends on a non-uniform source. The sources are the invocation id, the
/// local index, storage and workgroup reads, atomics, and texture loads;
/// uniform struct fields, literals, and arithmetic over uniform values stay
/// uniform. A `phi` is uniform only when every incoming value is, which is what
/// makes a loop counter (`i = phi(0, i + 1)`) uniform.
fn uniform_values(func: &LlvmFunc) -> HashSet<String> {
    let mut uniform: HashSet<String> = func
        .blocks
        .iter()
        .flat_map(|block| block.instrs.iter())
        .filter_map(|instr| instr.defined().map(str::to_string))
        .collect();
    loop {
        let mut changed = false;
        for block in &func.blocks {
            for instr in &block.instrs {
                let Some(name) = instr.defined() else {
                    continue;
                };
                if !uniform.contains(name) || instruction_is_uniform(instr, &uniform) {
                    continue;
                }
                uniform.remove(name);
                changed = true;
            }
        }
        if !changed {
            return uniform;
        }
    }
}

/// Whether `instr` produces the same value in every invocation.
fn instruction_is_uniform(instr: &Instr, uniform: &HashSet<String>) -> bool {
    let operands_uniform = {
        let mut all = true;
        instr.each_operand(|op| {
            if let Operand::Local(name) = op
                && !uniform.contains(name)
            {
                all = false;
            }
        });
        all
    };
    if !operands_uniform {
        return false;
    }
    match instr {
        Instr::Call { callee, .. } => uniform_callee(callee),
        _ => true,
    }
}

/// Whether an import returns the same value in every invocation.
///
/// Uniform struct fields, array lengths, texture dimensions, and the scalar
/// math builtins do; the invocation id, workgroup memory, storage reads, and
/// texture fetches do not.
fn uniform_callee(callee: &str) -> bool {
    const UNIFORM: [&str; 5] = [
        "uniform_",
        "storage_len",
        "texture_width",
        "texture_height",
        "gpu_",
    ];
    UNIFORM.iter().any(|prefix| callee.contains(prefix)) || callee.contains("llvm.")
}

/// Blocks control can leave `term` for, in the order the IR names them.
fn successors(term: &Terminator) -> Vec<&str> {
    match term {
        Terminator::Ret(_) | Terminator::Unreachable => Vec::new(),
        Terminator::BrUncond(target) => vec![target.as_str()],
        Terminator::BrCond {
            true_label,
            false_label,
            ..
        } => vec![true_label.as_str(), false_label.as_str()],
        Terminator::Switch { default, cases, .. } => {
            let mut out = vec![default.as_str()];
            out.extend(cases.iter().map(|(_, label)| label.as_str()));
            out
        }
    }
}

/// Stage of `func`, after checking the shape rustc compiles for it.
///
/// A compute entry takes nothing or one `i32` (its invocation id) and may
/// return `f32`; a raster entry is `void` with no parameters, because its
/// builtins and interface come from the stage-IO imports.
fn entry_stage(func: &LlvmFunc) -> Result<Stage, TranslateError> {
    let stage = Stage::of(&func.name).ok_or(TranslateError::Unsupported)?;
    let shape_ok = match stage {
        Stage::Compute => matches!(
            (func.ret_ty, func.params.as_slice()),
            (Ty::Float | Ty::Void, [] | [(Ty::I32, _)])
        ),
        Stage::Vertex | Stage::Fragment => func.ret_ty == Ty::Void && func.params.is_empty(),
    };
    if shape_ok {
        Ok(stage)
    } else {
        Err(TranslateError::Unsupported)
    }
}

/// `vecN<f32>` of a stage-interface value, or `None` for a width the
/// translator does not declare.
fn vec_f32_type(module: &mut naga::Module, lanes: u32) -> Option<naga::Handle<naga::Type>> {
    let size = match lanes {
        VEC2_LANES => naga::VectorSize::Bi,
        VEC3_LANES => naga::VectorSize::Tri,
        VEC4_LANES => naga::VectorSize::Quad,
        _ => return None,
    };
    Some(module.types.insert(
        naga::Type {
            name: None,
            inner: vector_f32(size),
        },
        naga::Span::default(),
    ))
}

/// `vec4<f32>`, the clip position and every four-lane interface value.
fn vec4_f32_type(module: &mut naga::Module) -> naga::Handle<naga::Type> {
    module.types.insert(
        naga::Type {
            name: None,
            inner: vector_f32(naga::VectorSize::Quad),
        },
        naga::Span::default(),
    )
}

/// A `vecN<f32>` type body.
fn vector_f32(size: naga::VectorSize) -> naga::TypeInner {
    naga::TypeInner::Vector {
        size,
        scalar: naga::Scalar {
            kind: naga::ScalarKind::Float,
            width: SCALAR_WIDTH,
        },
    }
}

/// Declare the entry arguments the body reads: builtins and `@location`s.
fn stage_arguments(
    module: &mut naga::Module,
    func: &LlvmFunc,
    stage: Stage,
    io: &StageIo,
    function: &mut naga::Function,
) -> Result<StageArguments, TranslateError> {
    let (global_id, id_x) = if stage == Stage::Compute {
        let (global_id, id_x) = invocation_x(module, function);
        (Some(global_id), Some(id_x))
    } else {
        (None, None)
    };
    let local_index = (stage == Stage::Compute && uses_call(func, "local_invocation_index"))
        .then(|| local_index_arg(module, function));
    let vertex_index = io
        .vertex_index
        .then(|| builtin_u32_arg(module, function, "vertex_index", naga::BuiltIn::VertexIndex));
    let instance_index = io.instance_index.then(|| {
        builtin_u32_arg(
            module,
            function,
            "instance_index",
            naga::BuiltIn::InstanceIndex,
        )
    });
    // A fragment always declares `@builtin(position)`: the checked-in AO
    // artifacts carry it, and an entry may read it from any branch.
    let position = (stage == Stage::Fragment)
        .then(|| vec4_f32_type(module))
        .map(|ty| position_arg(ty, function));
    let mut inputs = BTreeMap::new();
    for (&location, &lanes) in &io.inputs {
        let ty = vec_f32_type(module, lanes).ok_or(TranslateError::Unsupported)?;
        // A fragment's location-0 two-lane interpolant keeps the `uv`
        // parameter name the checked-in AO artifacts already carry.
        let legacy_uv = stage == Stage::Fragment && location == 0 && lanes == VEC2_LANES;
        let name = if legacy_uv {
            "uv".to_string()
        } else {
            format!("in{location}")
        };
        inputs.insert(location, location_arg(function, ty, location, &name));
    }
    Ok(StageArguments {
        global_id,
        id_x,
        local_index,
        vertex_index,
        instance_index,
        position,
        inputs,
    })
}

/// Declare the entry's stage output and its return type.
///
/// A vertex entry always returns a struct: `@builtin(position)` is mandatory in
/// a vertex result. A fragment entry returns a bare `@location(0) vec4<f32>`
/// when that is its only target — the shape the checked-in fragment artifacts
/// already have — and a struct otherwise.
fn stage_output(
    module: &mut naga::Module,
    stage: Stage,
    io: &StageIo,
    function: &mut naga::Function,
) -> Result<Option<StageOut>, TranslateError> {
    if !stage.raster() {
        return Ok(None);
    }
    let bare_fragment = stage == Stage::Fragment
        && io.outputs.len() == 1
        && io.outputs.get(&0) == Some(&VEC4_LANES)
        && !io.position_out;
    if bare_fragment {
        let ty = vec4_f32_type(module);
        function.result = Some(naga::FunctionResult {
            ty,
            binding: Some(location_binding(0)),
        });
        let var = output_local(function, "fragment_out", ty);
        return Ok(Some(StageOut::bare(var)));
    }
    if stage == Stage::Vertex && !io.position_out {
        return Err(TranslateError::Unsupported);
    }
    if io.outputs.is_empty() {
        return Err(TranslateError::Unsupported);
    }
    let OutputType {
        ty,
        slots: locations,
        position,
    } = output_struct(module, stage, io)?;
    function.result = Some(naga::FunctionResult { ty, binding: None });
    let name = if stage == Stage::Vertex {
        "vertex_out"
    } else {
        "fragment_out"
    };
    let var = output_local(function, name, ty);
    Ok(Some(StageOut::located(var, locations, position)))
}

/// Struct type of a raster entry result, plus the member index of every
/// located output and of `@builtin(position)`.
fn output_struct(
    module: &mut naga::Module,
    stage: Stage,
    io: &StageIo,
) -> Result<OutputType, TranslateError> {
    let mut layout = OutputLayout::default();
    if stage == Stage::Vertex {
        layout.push(
            OutSlot::Position,
            VEC4_LANES,
            vec4_f32_type(module),
            naga::Binding::BuiltIn(naga::BuiltIn::Position { invariant: false }),
        );
    }
    for (&location, &lanes) in &io.outputs {
        let ty = vec_f32_type(module, lanes).ok_or(TranslateError::Unsupported)?;
        layout.push(
            OutSlot::Location(location),
            lanes,
            ty,
            location_binding(location),
        );
    }
    let position = layout.slots.get(&OutSlot::Position).copied();
    let slots = layout.slots.clone();
    let span = layout.span();
    let ty = module.types.insert(
        naga::Type {
            name: Some(output_struct_name(stage).to_string()),
            inner: naga::TypeInner::Struct {
                members: layout.members,
                span,
            },
        },
        naga::Span::default(),
    );
    Ok(OutputType {
        ty,
        slots,
        position,
    })
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
    /// `@location` inputs and outputs the body touches.
    io: StageIo,
    id_x: Option<naga::Handle<naga::Expression>>,
    /// `global_invocation_id` vector, when this entry is compute.
    global_id: Option<naga::Handle<naga::Expression>>,
    /// `@builtin(local_invocation_index)`, when the kernel reads it.
    local_index: Option<naga::Handle<naga::Expression>>,
    /// `@builtin(vertex_index)`, when the vertex entry reads it.
    vertex_index: Option<naga::Handle<naga::Expression>>,
    /// `@builtin(instance_index)`, when the vertex entry reads it.
    instance_index: Option<naga::Handle<naga::Expression>>,
    /// Local the entry returns: the fragment color or the stage output struct.
    stage_out: Option<StageOut>,
    /// `@builtin(position)` of `fs_main`, when this entry is a fragment.
    position: Option<naga::Handle<naga::Expression>>,
    /// Declared `@location` arguments of the entry, by location.
    inputs: BTreeMap<u32, naga::Handle<naga::Expression>>,
    phi_vars: HashMap<String, naga::Handle<naga::LocalVariable>>,
    /// Loop values used after the loop, stored while the loop runs.
    spill: HashMap<String, naga::Handle<naga::LocalVariable>>,
    env: Env,
    saw_return: bool,
    /// Enclosing `if`/`loop`/`switch` regions whose condition or trip count is
    /// not proven uniform. A workgroup barrier inside any of them is rejected:
    /// a divergent barrier is UB on native GPUs (Tint rejects that WGSL).
    divergent_depth: u32,
    /// Values that are the same for every invocation in the workgroup.
    uniform: HashSet<String>,
    /// Exit block -> selector value, for loops whose exits go to more than one
    /// block (or leave more than one loop).
    ///
    /// A `break` in Naga leaves only the innermost `Loop`, so the selector says
    /// where control has to resume once the loop statements have unwound.
    exit_ids: HashMap<&'a str, u32>,
    /// Local the selector is stored in, created on first use.
    exit_id_var: Option<naga::Handle<naga::LocalVariable>>,
    /// Pointer SSA values: their base and their offset in 4-byte words.
    pointers: HashMap<String, MemoryPointer>,
    /// Base a pointer `phi` walks; every incoming pointer must share it.
    ptr_bases: HashMap<String, MemoryBase>,
    /// Module constants turned into private globals: handle and word count.
    constants: HashMap<String, (naga::Handle<naga::GlobalVariable>, u32)>,
    /// An arm returned and the region merged afterwards may therefore run on
    /// a subset of invocations. Sticky for the rest of the entry.
    diverged: bool,
    /// Whether `ret` returns a value (a helper) instead of storing it (the
    /// `cs_main(i32) -> f32` toy shape).
    returns_value: bool,
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
        let mut labels: Vec<&str> = self.cfg.phis.keys().copied().collect();
        labels.sort_unstable();
        for label in labels {
            for phi in &self.cfg.phis[label] {
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
            // A pointer phi carries the offset in words; the base is the same
            // for every incoming edge.
            Ty::Ptr => naga::ScalarKind::Uint,
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
            // The panic path: LLVM calls it undefined behaviour, so the shader
            // ends the invocation here instead of pretending it cannot happen.
            Terminator::Unreachable => self.emit_ret(None, out),
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
                        self.store_exit_id(out, target)?;
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
            Terminator::Switch {
                selector,
                default,
                cases,
            } => self.emit_switch(start, selector, default, cases, stop, out, ctx),
        }
    }

    /// `alloca`: a function-scope array of 4-byte words.
    fn alloca(&mut self, dst: &str, bytes: u32) -> Result<(), TranslateError> {
        let words = word_count(bytes)?;
        let ty = self.word_array_type(words);
        let var = self.func.local_variables.append(
            naga::LocalVariable {
                name: Some(dst.to_string()),
                ty,
                init: None,
            },
            naga::Span::default(),
        );
        let zero = self.u32_literal(&mut naga::Block::new(), 0);
        self.pointers.insert(
            dst.to_string(),
            MemoryPointer {
                base: MemoryBase::Local(var),
                words: zero,
            },
        );
        Ok(())
    }

    /// `getelementptr`: move a pointer by `index * stride` bytes.
    fn getelementptr(
        &mut self,
        dst: &str,
        base: &PtrBase,
        stride: u32,
        index: &Operand,
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        let (base, words) = match base {
            PtrBase::Local(name) => {
                let pointer = self.pointers.get(name).ok_or(TranslateError::Unsupported)?;
                (pointer.base, pointer.words)
            }
            PtrBase::Global(symbol) => {
                let (var, _) = self.constant_array(symbol)?;
                let zero = self.u32_literal(out, 0);
                (MemoryBase::Constant(var), zero)
            }
        };
        let step = self.pointer_step(stride, index, out)?;
        let words = self.binary(out, naga::BinaryOperator::Add, words, step);
        self.pointers
            .insert(dst.to_string(), MemoryPointer { base, words });
        Ok(())
    }

    /// Words one `getelementptr` index step covers.
    ///
    /// A stride that is not a multiple of four is only accepted for a constant
    /// index that is: LLVM emits `i8` GEPs for field offsets, and those are
    /// always word-aligned in this subset.
    fn pointer_step(
        &mut self,
        stride: u32,
        index: &Operand,
        out: &mut naga::Block,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if stride.is_multiple_of(WORD_BYTES) {
            let index = self.operand(out, index)?;
            let step = stride / WORD_BYTES;
            if step == 1 {
                return Ok(index);
            }
            let step = self.u32_literal(out, step);
            return Ok(self.binary(out, naga::BinaryOperator::Multiply, index, step));
        }
        let Operand::U32(offset) = index else {
            return Err(TranslateError::Unsupported);
        };
        if !offset.is_multiple_of(WORD_BYTES) {
            return Err(TranslateError::Unsupported);
        }
        Ok(self.u32_literal(out, offset / WORD_BYTES))
    }

    /// `load`: read one 4-byte word and reinterpret it as `ty`.
    fn load(
        &mut self,
        ptr: &str,
        ty: Ty,
        out: &mut naga::Block,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let pointer = self.pointer(ptr)?;
        let word = self.load_word(out, pointer.base, pointer.words);
        match ty {
            Ty::I32 => Ok(word),
            Ty::Float => Ok(self.bitcast_float(out, word)),
            _ => Err(TranslateError::Unsupported),
        }
    }

    /// `store`: write one 4-byte word.
    fn store(
        &mut self,
        ptr: &str,
        value: &Operand,
        ty: Ty,
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        let pointer = self.pointer(ptr)?;
        let value = self.operand(out, value)?;
        let value = match ty {
            Ty::I32 => value,
            Ty::Float => self.bitcast_uint(out, value),
            _ => return Err(TranslateError::Unsupported),
        };
        self.store_word(out, pointer.base, pointer.words, value);
        Ok(())
    }

    /// `select i1 %c, ptr %a, ptr %b`: pick the offset of one of two pointers.
    ///
    /// Both have to start at the same base, the same rule `phi ptr` follows:
    /// Naga has no pointer value, so the selection happens on the word offset.
    /// Returns `None` when the operands are not pointers.
    fn pointer_select(
        &mut self,
        dst: &str,
        cond: &Operand,
        accept: &Operand,
        reject: &Operand,
        out: &mut naga::Block,
    ) -> Result<Option<MemoryPointer>, TranslateError> {
        let (Operand::Local(accept), Operand::Local(reject)) = (accept, reject) else {
            return Ok(None);
        };
        let (Some(accept), Some(reject)) = (
            self.pointers.get(accept).copied(),
            self.pointers.get(reject).copied(),
        ) else {
            return Ok(None);
        };
        if accept.base != reject.base {
            return Err(TranslateError::UnsupportedType(format!(
                "`select` of two pointers from different allocations: `{}` picks between                  memory this translator cannot name at runtime",
                dst
            )));
        }
        let condition = self.operand(out, cond)?;
        let words = self.expr(
            out,
            naga::Expression::Select {
                condition,
                accept: accept.words,
                reject: reject.words,
            },
        );
        let pointer = MemoryPointer {
            base: accept.base,
            words,
        };
        self.pointers.insert(dst.to_string(), pointer);
        Ok(Some(pointer))
    }

    /// Pointer SSA value `name`.
    fn pointer(&self, name: &str) -> Result<MemoryPointer, TranslateError> {
        self.pointers
            .get(name)
            .map(|pointer| MemoryPointer {
                base: pointer.base,
                words: pointer.words,
            })
            .ok_or(TranslateError::Unsupported)
    }

    /// Write `word` at `base + words`.
    fn store_word(
        &mut self,
        out: &mut naga::Block,
        base: MemoryBase,
        words: naga::Handle<naga::Expression>,
        word: naga::Handle<naga::Expression>,
    ) {
        let base = self.memory_base(base, out);
        let slot = self.expr(out, naga::Expression::Access { base, index: words });
        out.push(
            naga::Statement::Store {
                pointer: slot,
                value: word,
            },
            naga::Span::default(),
        );
    }

    /// Read the word at `base + words`.
    fn load_word(
        &mut self,
        out: &mut naga::Block,
        base: MemoryBase,
        words: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        let base = self.memory_base(base, out);
        let slot = self.expr(out, naga::Expression::Access { base, index: words });
        self.expr(out, naga::Expression::Load { pointer: slot })
    }

    /// `(destination, fill, byte count)` of a `memset`-shaped call.
    fn intrinsic_target<'o>(
        &self,
        args: &'o [Operand],
    ) -> Result<(MemoryPointer, &'o Operand, u32), TranslateError> {
        if args.len() < MEMORY_INTRINSIC_ARGS {
            return Err(TranslateError::Unsupported);
        }
        let Operand::Local(name) = &args[0] else {
            return Err(TranslateError::Unsupported);
        };
        let Operand::U32(bytes) = args[2] else {
            return Err(TranslateError::Unsupported);
        };
        Ok((self.pointer(name)?, &args[1], bytes))
    }

    /// `llvm.memset(ptr, fill, size, …)`: fill whole words with one byte.
    ///
    /// The size and the fill byte must be immediates and the size a multiple of
    /// four, which is what `[value; N]` initialization and rustc's array
    /// zeroing produce.
    fn memset(&mut self, out: &mut naga::Block, args: &[Operand]) -> Result<(), TranslateError> {
        let (pointer, fill, bytes) = self.intrinsic_target(args)?;
        let byte = match fill {
            Operand::U32(value) => *value & BYTE_MASK,
            Operand::Bool(value) => u32::from(*value),
            Operand::F32(_) | Operand::Local(_) | Operand::Symbol(_) => {
                return Err(TranslateError::Unsupported);
            }
        };
        let word = self.u32_literal(out, repeated_byte(byte));
        for offset in 0..word_count(bytes)? {
            let index = self.offset_words(out, pointer.words, offset);
            self.store_word(out, pointer.base, index, word);
        }
        Ok(())
    }

    /// `llvm.memcpy(dst, src, size, …)`: copy whole words.
    fn memcpy(&mut self, out: &mut naga::Block, args: &[Operand]) -> Result<(), TranslateError> {
        if args.len() < MEMORY_INTRINSIC_ARGS {
            return Err(TranslateError::Unsupported);
        }
        let Operand::Local(dst) = &args[0] else {
            return Err(TranslateError::Unsupported);
        };
        let Operand::Local(src) = &args[1] else {
            return Err(TranslateError::Unsupported);
        };
        let Operand::U32(bytes) = args[2] else {
            return Err(TranslateError::Unsupported);
        };
        let dst = self.pointer(dst)?;
        let src = self.pointer(src)?;
        for offset in 0..word_count(bytes)? {
            let from = self.offset_words(out, src.words, offset);
            let word = self.load_word(out, src.base, from);
            let to = self.offset_words(out, dst.words, offset);
            self.store_word(out, dst.base, to, word);
        }
        Ok(())
    }

    /// `words + offset`.
    fn offset_words(
        &mut self,
        out: &mut naga::Block,
        words: naga::Handle<naga::Expression>,
        offset: u32,
    ) -> naga::Handle<naga::Expression> {
        if offset == 0 {
            return words;
        }
        let step = self.u32_literal(out, offset);
        self.binary(out, naga::BinaryOperator::Add, words, step)
    }

    /// Pointer expression of a memory base.
    fn memory_base(
        &mut self,
        base: MemoryBase,
        out: &mut naga::Block,
    ) -> naga::Handle<naga::Expression> {
        match base {
            MemoryBase::Local(var) => self.expr(out, naga::Expression::LocalVariable(var)),
            MemoryBase::Constant(var) => self.expr(out, naga::Expression::GlobalVariable(var)),
        }
    }

    /// `array<u32, words>`.
    fn word_array_type(&mut self, words: u32) -> naga::Handle<naga::Type> {
        let base = scalar(&mut self.builders.module, naga::ScalarKind::Uint);
        let size = std::num::NonZeroU32::new(words).unwrap_or(std::num::NonZeroU32::MIN);
        self.builders.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Array {
                    base,
                    size: naga::ArraySize::Constant(size),
                    stride: WORD_BYTES,
                },
            },
            naga::Span::default(),
        )
    }

    /// A module constant as a private `array<u32, N>` global.
    ///
    /// The words are the constant's bytes, so an integer table and a float
    /// table both keep their exact bit patterns; a float load bitcasts.
    fn constant_array(
        &mut self,
        symbol: &str,
    ) -> Result<(naga::Handle<naga::GlobalVariable>, u32), TranslateError> {
        if let Some(found) = self.constants.get(symbol) {
            return Ok(*found);
        }
        let bytes = self
            .builders
            .module_constants
            .get(symbol)
            .ok_or(TranslateError::Unsupported)?
            .clone();
        let words = word_count(bytes.len() as u32)?;
        let ty = self.word_array_type(words);
        let mut components = Vec::with_capacity(words as usize);
        for chunk in bytes.chunks_exact(WORD_BYTES as usize) {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            components.push(self.builders.module.global_expressions.append(
                naga::Expression::Literal(naga::Literal::U32(word)),
                naga::Span::default(),
            ));
        }
        let init = self.builders.module.global_expressions.append(
            naga::Expression::Compose { ty, components },
            naga::Span::default(),
        );
        let var = self.builders.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(format!("const_{}", symbol.replace(['.', '-'], "_"))),
                space: naga::AddressSpace::Private,
                binding: None,
                ty,
                init: Some(init),
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        );
        self.constants.insert(symbol.to_string(), (var, words));
        Ok((var, words))
    }

    /// Selector value for `target`, assigned the first time it is needed.
    fn exit_id(&mut self, target: &'a str) -> u32 {
        if let Some(id) = self.exit_ids.get(target) {
            return *id;
        }
        let id = self.exit_ids.len() as u32 + 1;
        self.exit_ids.insert(target, id);
        id
    }

    /// Record that the current `break` leaves the loop towards `target`.
    ///
    /// Loops with a single exit never assign ids, so their WGSL keeps the plain
    /// `break` it always had.
    fn store_exit_id(
        &mut self,
        out: &mut naga::Block,
        target: &'a str,
    ) -> Result<(), TranslateError> {
        let Some(id) = self.exit_ids.get(target).copied() else {
            return Ok(());
        };
        let var = match self.exit_id_var {
            Some(var) => var,
            None => {
                let var = self.func.local_variables.append(
                    naga::LocalVariable {
                        name: Some("exit_id".to_string()),
                        ty: scalar(&mut self.builders.module, naga::ScalarKind::Uint),
                        init: None,
                    },
                    naga::Span::default(),
                );
                self.exit_id_var = Some(var);
                var
            }
        };
        let pointer = self.expr(out, naga::Expression::LocalVariable(var));
        let value = self.u32_literal(out, id);
        out.push(
            naga::Statement::Store { pointer, value },
            naga::Span::default(),
        );
        Ok(())
    }

    /// Mark the control region a condition opens, returning the depth to restore.
    ///
    /// A proven-uniform condition keeps the region uniform, so a barrier inside
    /// it stays valid; anything else makes the enclosing region divergent.
    fn enter_control(&mut self, cond: &Operand) -> u32 {
        let outer = self.divergent_depth;
        if !self.is_uniform(cond) {
            self.divergent_depth += 1;
        }
        outer
    }

    /// Whether `op` has the same value in every invocation of the workgroup.
    fn is_uniform(&self, op: &Operand) -> bool {
        match op {
            Operand::Local(name) => self.uniform.contains(name),
            Operand::Bool(_) | Operand::F32(_) | Operand::U32(_) | Operand::Symbol(_) => true,
        }
    }

    /// Whether a loop runs the same number of iterations for every invocation.
    ///
    /// Conservative: every edge that leaves the loop must come from a
    /// conditional whose condition is uniform, and the loop must not leave
    /// unconditionally. A `break` behind a non-uniform branch, or a plain
    /// `break`, therefore makes the loop divergent — a barrier inside it is
    /// rejected rather than assumed safe.
    fn loop_is_uniform(&self, header: &str, body: &HashSet<&'a str>) -> bool {
        body.iter().all(|label| {
            let block = self.cfg.blocks[label];
            let leaves: Vec<&str> = self
                .cfg
                .succs
                .get(label)
                .into_iter()
                .flatten()
                .filter(|target| !body.contains(*target))
                .copied()
                .collect();
            let exits_all = !leaves.is_empty();
            if !exits_all {
                // Still check a conditional that can reach the header again.
                return match &block.term {
                    Terminator::BrCond { cond, .. } if *label == header => self.is_uniform(cond),
                    _ => true,
                };
            }
            match &block.term {
                Terminator::BrCond { cond, .. } => self.is_uniform(cond),
                _ => false,
            }
        })
    }

    /// Reject a merge that lies outside the enclosing loop.
    ///
    /// A natural loop cannot have a merge outside it that both arms reach: any
    /// block on the way to the back edge belongs to the loop. Rejecting keeps
    /// the structurer from guessing.
    fn check_merge_in_loop(&self, merge: &str, ctx: Ctx<'a>) -> Result<(), TranslateError> {
        let Ctx::Loop { header } = ctx else {
            return Ok(());
        };
        let merges_in_loop = self
            .cfg
            .loops
            .get(header)
            .is_some_and(|body| body.contains(merge));
        if merge != EXIT && !merges_in_loop {
            return Err(TranslateError::Unsupported);
        }
        Ok(())
    }

    /// Emit one arm of a branch: enter it from `from`, run it until `merge`.
    ///
    /// The caller sets `self.env` before calling; the arm's phi stores are
    /// written from `env`, which is the environment at the branch.
    fn emit_arm(
        &mut self,
        from: &'a str,
        enter: &'a str,
        merge: &'a str,
        env: &Env,
        ctx: Ctx<'a>,
    ) -> Result<(naga::Block, Flow), TranslateError> {
        let mut body = naga::Block::new();
        self.env = env.clone();
        self.store_edge_into(from, enter, env, &mut body)?;
        let flow = self.emit_region(enter, Some(merge), &mut body, ctx)?;
        Ok((body, flow))
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
        self.check_merge_in_loop(merge, ctx)?;
        let condition = self.operand(out, cond)?;
        let outer_env = self.env.clone();
        let outer_depth = self.enter_control(cond);
        let (accept, accept_flow) = self.emit_arm(label, true_label, merge, &outer_env, ctx)?;
        let (reject, reject_flow) = self.emit_arm(label, false_label, merge, &outer_env, ctx)?;
        self.divergent_depth = outer_depth;
        self.env = outer_env;
        if accept_flow == Flow::Returned || reject_flow == Flow::Returned {
            // The merge block only sees the invocations whose arm fell
            // through; a barrier there would not be reached by all lanes.
            self.diverged = true;
        }
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

    /// `switch`: every case runs until the closest common post-dominator.
    ///
    /// Cases that share a target label share one body (the first values fall
    /// through into it), which is what Rust emits for `x | y => …` arms.
    #[allow(clippy::too_many_arguments)]
    fn emit_switch(
        &mut self,
        label: &'a str,
        selector: &Operand,
        default: &'a str,
        cases: &'a [(u32, String)],
        stop: Option<&'a str>,
        out: &mut naga::Block,
        ctx: Ctx<'a>,
    ) -> Result<Flow, TranslateError> {
        let merge = self.cfg.merge(label).ok_or(TranslateError::Unsupported)?;
        self.check_merge_in_loop(merge, ctx)?;
        let selector_expr = self.operand(out, selector)?;
        let outer_env = self.env.clone();
        let outer_depth = self.enter_control(selector);
        let mut targets: Vec<(&'a str, Vec<u32>)> = Vec::new();
        for (value, target) in cases {
            let target = target.as_str();
            match targets.iter_mut().find(|(label, _)| *label == target) {
                Some((_, values)) => values.push(*value),
                None => targets.push((target, vec![*value])),
            }
        }
        let mut naga_cases = Vec::new();
        let mut returned = false;
        for (target, values) in targets {
            let (mut body, flow) = self.emit_arm(label, target, merge, &outer_env, ctx)?;
            returned |= case_flow_returns(flow)?;
            for (index, value) in values.iter().enumerate() {
                let last = index + 1 == values.len();
                let filled = if last {
                    std::mem::take(&mut body)
                } else {
                    naga::Block::new()
                };
                naga_cases.push(switch_case(naga::SwitchValue::U32(*value), filled, !last));
            }
        }
        let (default_body, default_flow) = self.emit_arm(label, default, merge, &outer_env, ctx)?;
        returned |= case_flow_returns(default_flow)?;
        naga_cases.push(switch_case(naga::SwitchValue::Default, default_body, false));
        self.divergent_depth = outer_depth;
        self.env = outer_env;
        if returned {
            // The merge only runs on the lanes whose case fell through.
            self.diverged = true;
        }
        out.push(
            naga::Statement::Switch {
                selector: selector_expr,
                cases: naga_cases,
            },
            naga::Span::default(),
        );
        if merge == EXIT {
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
        // `Next` means the arm ended at a nested exit branch inside the loop,
        // which already emitted its own `break`/`continue`.
        if !matches!(
            inside_flow,
            Flow::Next | Flow::Continue | Flow::Break | Flow::Returned
        ) {
            return Err(TranslateError::Unsupported);
        }
        let mut outside_block = naga::Block::new();
        self.env = outer_env.clone();
        self.store_edge_into(start, outside, &outer_env, &mut outside_block)?;
        self.store_exit_id(&mut outside_block, outside)?;
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
        true_label: &'a str,
        false_label: &'a str,
        out: &mut naga::Block,
    ) -> Result<Flow, TranslateError> {
        let condition = self.operand(out, cond)?;
        let outer_env = self.env.clone();
        let mut accept = naga::Block::new();
        self.env = outer_env.clone();
        self.store_edge_into(start, true_label, &outer_env, &mut accept)?;
        self.store_exit_id(&mut accept, true_label)?;
        accept.push(naga::Statement::Break, naga::Span::default());
        let mut reject = naga::Block::new();
        self.env = outer_env.clone();
        self.store_edge_into(start, false_label, &outer_env, &mut reject)?;
        self.store_exit_id(&mut reject, false_label)?;
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

    /// `naga::Loop` for a natural loop with one header.
    ///
    /// `outer` is the region that contains this loop. The exit block is
    /// emitted there, so a nested loop's `break` does not leave the outer
    /// `loop` and a branch back to the outer header stays a `continue`.
    ///
    /// A loop with one exit keeps the plain `break` shape. Several exits, or an
    /// exit that leaves more than one loop (`break 'outer`), record which block
    /// control has to resume at in a selector local: each `break` unwinds one
    /// `Loop`, and the dispatch after it either resumes the exit block or
    /// breaks the next loop out.
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
        let mut exits: Vec<&'a str> = loop_exits(self.cfg, &body).into_iter().collect();
        exits.sort_unstable();
        let live = self.loop_live_outs(&body)?;
        // A single exit that stays in the enclosing region needs no selector.
        let single = exits.len() == 1 && !self.leaves_outer(exits[0], outer);
        if !single {
            for exit in &exits {
                self.exit_id(exit);
            }
        }
        let ctx = Ctx::Loop { header };
        let outer_env = self.env.clone();
        let outer_depth = self.divergent_depth;
        if !self.loop_is_uniform(header, &body) {
            self.divergent_depth += 1;
        }
        let mut loop_body = naga::Block::new();
        let flow = self.emit_region(header, None, &mut loop_body, ctx)?;
        self.divergent_depth = outer_depth;
        // Expressions created inside the loop body are out of scope after
        // `Loop`; the exit block reloads whatever its phis need.
        self.env = outer_env;
        if flow == Flow::Continue {
            return Err(TranslateError::Unsupported);
        }
        if exits.is_empty() {
            if flow == Flow::Returned {
                out.push(loop_statement(loop_body), naga::Span::default());
                return Ok(Flow::Returned);
            }
            return Err(TranslateError::Unsupported);
        }
        out.push(loop_statement(loop_body), naga::Span::default());
        if flow == Flow::Returned {
            return Ok(Flow::Returned);
        }
        self.reload_live(&live, out)?;
        if single {
            return self.resume_after_loop(exits[0], stop, out, outer);
        }
        self.dispatch_exits(header, &exits, stop, out, outer)
    }

    /// Whether `target` is outside the loop that contains this region.
    fn leaves_outer(&self, target: &'a str, outer: Ctx<'a>) -> bool {
        let Ctx::Loop { header } = outer else {
            return false;
        };
        self.cfg
            .loops
            .get(header)
            .is_none_or(|body| !body.contains(target))
    }

    /// Continue at `exit` after its `Loop`, in the enclosing region.
    fn resume_after_loop(
        &mut self,
        exit: &'a str,
        stop: Option<&'a str>,
        out: &mut naga::Block,
        outer: Ctx<'a>,
    ) -> Result<Flow, TranslateError> {
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
            if self.leaves_outer(exit, outer) {
                // `break 'outer`: this `break` leaves the enclosing loop too.
                out.push(naga::Statement::Break, naga::Span::default());
                return Ok(Flow::Break);
            }
        }
        self.emit_region(exit, stop, out, outer)
    }

    /// Resume at whichever exit the selector names, after the loop unwound.
    ///
    /// Every exit runs until the loop's merge, so the code after the loop is
    /// emitted once, after the dispatch. An exit that leaves the enclosing loop
    /// breaks it instead, and the enclosing loop's own dispatch resumes there.
    fn dispatch_exits(
        &mut self,
        header: &'a str,
        exits: &[&'a str],
        stop: Option<&'a str>,
        out: &mut naga::Block,
        outer: Ctx<'a>,
    ) -> Result<Flow, TranslateError> {
        let merge = self.cfg.merge(header).unwrap_or(EXIT);
        let selector = self.exit_id_var.ok_or(TranslateError::Unsupported)?;
        let mut returned = 0usize;
        for exit in exits {
            let id = self.exit_id(exit);
            let pointer = self.expr(out, naga::Expression::LocalVariable(selector));
            let value = self.expr(out, naga::Expression::Load { pointer });
            let expected = self.u32_literal(out, id);
            let condition = self.binary(out, naga::BinaryOperator::Equal, value, expected);
            let mut accept = naga::Block::new();
            if self.leaves_outer(exit, outer) {
                accept.push(naga::Statement::Break, naga::Span::default());
            } else {
                let flow = self.emit_region(exit, Some(merge), &mut accept, outer)?;
                // `Continue`/`Break` leave the enclosing loop from inside the
                // case, which is a terminal flow for this dispatch too.
                if flow == Flow::Returned {
                    returned += 1;
                }
            }
            out.push(
                naga::Statement::If {
                    condition,
                    accept,
                    reject: naga::Block::new(),
                },
                naga::Span::default(),
            );
        }
        if merge == EXIT {
            if returned == exits.len() {
                return Ok(Flow::Returned);
            }
            return Err(TranslateError::Unsupported);
        }
        if self.leaves_outer(merge, outer) {
            // The merge is past the enclosing loop: break it and let its own
            // dispatch resume there.
            out.push(naga::Statement::Break, naga::Span::default());
            return Ok(Flow::Break);
        }
        if stop == Some(merge) {
            return Ok(Flow::Next);
        }
        self.emit_region(merge, stop, out, outer)
    }

    /// Values the exits read that are defined inside the loop.
    fn loop_live_outs(&mut self, body: &HashSet<&'a str>) -> Result<Vec<String>, TranslateError> {
        let mut used = Vec::new();
        // Anything defined in the loop and read after it has to be spilled:
        // the exit block is not the only reader. Later blocks and their phis
        // run once the loop's expressions are out of scope.
        for label in &self.cfg.order {
            if body.contains(label) {
                continue;
            }
            let block = self.cfg.blocks[label];
            for instr in &block.instrs {
                instr.each_operand(|op| push_local(op, &mut used));
            }
            push_term_local(&block.term, &mut used);
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
                Instr::Phi { dst, ty, .. } => {
                    let var = *self.phi_vars.get(dst).ok_or(TranslateError::Unsupported)?;
                    let ptr = self.expr(out, naga::Expression::LocalVariable(var));
                    let value = self.expr(out, naga::Expression::Load { pointer: ptr });
                    if *ty == Ty::Ptr {
                        let base = *self.ptr_bases.get(dst).ok_or(TranslateError::Unsupported)?;
                        self.pointers
                            .insert(dst.clone(), MemoryPointer { base, words: value });
                    } else {
                        self.bind(dst, value, out);
                    }
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
        if let Some(var) = self.stage_out.as_ref().map(|stage_out| stage_out.var) {
            let pointer = self.expr(out, naga::Expression::LocalVariable(var));
            let value = self.expr(out, naga::Expression::Load { pointer });
            out.push(
                naga::Statement::Return { value: Some(value) },
                naga::Span::default(),
            );
            self.saw_return = true;
            return Ok(Flow::Returned);
        }
        if self.returns_value {
            let value = match value {
                Some(op) => Some(self.operand(out, op)?),
                None => None,
            };
            out.push(naga::Statement::Return { value }, naga::Span::default());
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
            let value = if phi.ty == Ty::Ptr {
                let Operand::Local(name) = incoming else {
                    return Err(TranslateError::Unsupported);
                };
                let pointer = *self.pointers.get(name).ok_or(TranslateError::Unsupported)?;
                match self.ptr_bases.get(&phi.dst) {
                    Some(base) if *base != pointer.base => {
                        return Err(TranslateError::Unsupported);
                    }
                    Some(_) => {}
                    None => {
                        self.ptr_bases.insert(phi.dst.clone(), pointer.base);
                    }
                }
                pointer.words
            } else {
                operand(&mut self.func, out, incoming, env)?
            };
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
                self.assign_signed(dst, naga::BinaryOperator::Divide, a, b, out)?
            }
            Instr::URem { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::Modulo, a, b, out)?
            }
            Instr::SRem { dst, a, b } => {
                self.assign_signed(dst, naga::BinaryOperator::Modulo, a, b, out)?
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
            Instr::LShr { dst, a, b } => {
                self.assign_binary(dst, naga::BinaryOperator::ShiftRight, a, b, out)?
            }
            Instr::AShr { dst, a, b } => {
                // Arithmetic shift: only the shifted value is signed; WGSL
                // keeps the shift amount unsigned.
                let left = self.operand(out, a)?;
                let right = self.operand(out, b)?;
                let left = self.bitcast_sint(out, left);
                let value = self.expr(
                    out,
                    naga::Expression::Binary {
                        op: naga::BinaryOperator::ShiftRight,
                        left,
                        right,
                    },
                );
                let value = self.bitcast_uint(out, value);
                self.bind(dst, value, out);
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
            Instr::Fcmp { dst, pred, a, b } => {
                let v = self.compare_float(out, pred, a, b)?;
                self.bind(dst, v, out);
            }
            Instr::Icmp { dst, pred, a, b } => {
                let v = self.compare_int(out, pred, a, b)?;
                self.bind(dst, v, out);
            }
            Instr::Select { dst, c, a, b } => {
                if let Some(pointer) = self.pointer_select(dst, c, a, b, out)? {
                    let _ = pointer;
                    return Ok(());
                }
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
                let value = self.cast(out, a, naga::ScalarKind::Float)?;
                self.bind(dst, value, out);
            }
            Instr::Sitofp { dst, a } => {
                let value = self.cast_signed(out, a, naga::ScalarKind::Float)?;
                self.bind(dst, value, out);
            }
            Instr::Fptoui { dst, a } => {
                let value = self.cast(out, a, naga::ScalarKind::Uint)?;
                self.bind(dst, value, out);
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
            Instr::Alloca { dst, bytes } => self.alloca(dst, *bytes)?,
            Instr::GetElementPtr {
                dst,
                base,
                stride,
                index,
            } => self.getelementptr(dst, base, *stride, index, out)?,
            Instr::Load { dst, ptr, ty } => {
                let value = self.load(ptr, *ty, out)?;
                self.bind(dst, value, out);
            }
            Instr::Store { ptr, value, ty } => self.store(ptr, value, *ty, out)?,
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

    /// Signed `i32` binary op: bitcast the operands to `i32`, apply the
    /// operator (WGSL picks `sdiv`/`srem` from the operand type), and cast the
    /// result back to the stored `u32` representation.
    fn assign_signed(
        &mut self,
        dst: &str,
        op: naga::BinaryOperator,
        a: &Operand,
        b: &Operand,
        out: &mut naga::Block,
    ) -> Result<(), TranslateError> {
        let left = self.operand(out, a)?;
        let right = self.operand(out, b)?;
        let left = self.bitcast_sint(out, left);
        let right = self.bitcast_sint(out, right);
        let value = self.binary(out, op, left, right);
        let value = self.bitcast_uint(out, value);
        self.bind(dst, value, out);
        Ok(())
    }

    /// Bitcast the stored `u32` to `i32` (`bitcast<i32>(x)`).
    fn bitcast_sint(
        &mut self,
        out: &mut naga::Block,
        value: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.expr(
            out,
            naga::Expression::As {
                expr: value,
                kind: naga::ScalarKind::Sint,
                convert: None,
            },
        )
    }

    /// Bitcast an `i32`-typed result back to the stored `u32` type.
    fn bitcast_uint(
        &mut self,
        out: &mut naga::Block,
        value: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.expr(
            out,
            naga::Expression::As {
                expr: value,
                kind: naga::ScalarKind::Uint,
                convert: None,
            },
        )
    }

    /// A `u32` literal expression.
    fn u32_literal(&mut self, out: &mut naga::Block, value: u32) -> naga::Handle<naga::Expression> {
        self.expr(out, naga::Expression::Literal(naga::Literal::U32(value)))
    }

    /// Bitcast a stored `u32` word to `f32`.
    fn bitcast_float(
        &mut self,
        out: &mut naga::Block,
        value: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.expr(
            out,
            naga::Expression::As {
                expr: value,
                kind: naga::ScalarKind::Float,
                convert: None,
            },
        )
    }

    fn binary(
        &mut self,
        out: &mut naga::Block,
        op: naga::BinaryOperator,
        left: naga::Handle<naga::Expression>,
        right: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.expr(out, naga::Expression::Binary { op, left, right })
    }

    fn negate(
        &mut self,
        out: &mut naga::Block,
        expr: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        self.expr(
            out,
            naga::Expression::Unary {
                op: naga::UnaryOperator::LogicalNot,
                expr,
            },
        )
    }

    /// `fcmp`: ordered predicates are WGSL comparisons. Unordered (`u*`)
    /// predicates must also be true when either operand is NaN, so they lower
    /// through [`Self::float_ordered`]; `one` is the ordered not-equal and is
    /// false for NaN.
    ///
    /// Float `!=` is not used: naga's SPIR-V backend emits `OpFOrdNotEqual`
    /// for it, which is false for NaN, so both `une` and the usual `x != x`
    /// NaN test would silently break on Vulkan (Metal treats `!=` as
    /// unordered, which is why the difference shows only on lavapipe).
    fn compare_float(
        &mut self,
        out: &mut naga::Block,
        pred: &str,
        a: &Operand,
        b: &Operand,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let left = self.operand(out, a)?;
        let right = self.operand(out, b)?;
        let value = match pred {
            "oeq" => self.binary(out, naga::BinaryOperator::Equal, left, right),
            "olt" => self.binary(out, naga::BinaryOperator::Less, left, right),
            "ole" => self.binary(out, naga::BinaryOperator::LessEqual, left, right),
            "ogt" => self.binary(out, naga::BinaryOperator::Greater, left, right),
            "oge" => self.binary(out, naga::BinaryOperator::GreaterEqual, left, right),
            "one" => {
                let lt = self.binary(out, naga::BinaryOperator::Less, left, right);
                let gt = self.binary(out, naga::BinaryOperator::Greater, left, right);
                self.binary(out, naga::BinaryOperator::InclusiveOr, lt, gt)
            }
            "une" => {
                let ordered = self.float_ordered(out, left, right);
                let equal = self.binary(out, naga::BinaryOperator::Equal, left, right);
                let ordered_equal =
                    self.binary(out, naga::BinaryOperator::LogicalAnd, ordered, equal);
                self.negate(out, ordered_equal)
            }
            "ueq" => {
                let ordered = self.float_ordered(out, left, right);
                let unordered = self.negate(out, ordered);
                let equal = self.binary(out, naga::BinaryOperator::Equal, left, right);
                self.binary(out, naga::BinaryOperator::LogicalOr, unordered, equal)
            }
            "ult" => {
                let ge = self.binary(out, naga::BinaryOperator::GreaterEqual, left, right);
                self.negate(out, ge)
            }
            "ule" => {
                let gt = self.binary(out, naga::BinaryOperator::Greater, left, right);
                self.negate(out, gt)
            }
            "ugt" => {
                let le = self.binary(out, naga::BinaryOperator::LessEqual, left, right);
                self.negate(out, le)
            }
            "uge" => {
                let lt = self.binary(out, naga::BinaryOperator::Less, left, right);
                self.negate(out, lt)
            }
            "uno" | "ord" => {
                let ordered = self.float_ordered(out, left, right);
                if pred == "ord" {
                    ordered
                } else {
                    self.negate(out, ordered)
                }
            }
            _ => return Err(TranslateError::Unsupported),
        };
        Ok(value)
    }

    /// Whether both operands are comparable (`a <= a && b <= b`), i.e. not
    /// NaN. `<=` is ordered on every backend; `a != a` is not (see
    /// [`Self::compare_float`]).
    fn float_ordered(
        &mut self,
        out: &mut naga::Block,
        a: naga::Handle<naga::Expression>,
        b: naga::Handle<naga::Expression>,
    ) -> naga::Handle<naga::Expression> {
        let left = self.binary(out, naga::BinaryOperator::LessEqual, a, a);
        let right = self.binary(out, naga::BinaryOperator::LessEqual, b, b);
        self.binary(out, naga::BinaryOperator::LogicalAnd, left, right)
    }

    /// `icmp`: `s*` predicates reinterpret the stored `u32` as `i32`;
    /// unsigned and equality predicates compare the bit pattern directly.
    fn compare_int(
        &mut self,
        out: &mut naga::Block,
        pred: &str,
        a: &Operand,
        b: &Operand,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let left = self.operand(out, a)?;
        let right = self.operand(out, b)?;
        let (left, right, op) = match pred {
            "slt" => (
                self.bitcast_sint(out, left),
                self.bitcast_sint(out, right),
                naga::BinaryOperator::Less,
            ),
            "sle" => (
                self.bitcast_sint(out, left),
                self.bitcast_sint(out, right),
                naga::BinaryOperator::LessEqual,
            ),
            "sgt" => (
                self.bitcast_sint(out, left),
                self.bitcast_sint(out, right),
                naga::BinaryOperator::Greater,
            ),
            "sge" => (
                self.bitcast_sint(out, left),
                self.bitcast_sint(out, right),
                naga::BinaryOperator::GreaterEqual,
            ),
            "eq" => (left, right, naga::BinaryOperator::Equal),
            "ne" => (left, right, naga::BinaryOperator::NotEqual),
            "ult" => (left, right, naga::BinaryOperator::Less),
            "ule" => (left, right, naga::BinaryOperator::LessEqual),
            "ugt" => (left, right, naga::BinaryOperator::Greater),
            "uge" => (left, right, naga::BinaryOperator::GreaterEqual),
            _ => return Err(TranslateError::Unsupported),
        };
        Ok(self.binary(out, op, left, right))
    }

    /// Value-returning call: math or storage read / atomic add.
    /// Handle the void calls that are not ABI imports: a helper of this crate
    /// and the memory intrinsics. Returns whether the call was consumed.
    fn staged_void_call(
        &mut self,
        out: &mut naga::Block,
        callee: &str,
        args: &[Operand],
    ) -> Result<bool, TranslateError> {
        if self.builders.diverging.contains(callee) {
            // The panic path: the caller's own `unreachable` ends the
            // invocation, so the call itself has nothing to translate.
            return Ok(true);
        }
        if let Some(reason) = self.builders.skipped_helpers.get(callee) {
            return Err(TranslateError::UnsupportedType(format!(
                "`{callee}` cannot be lowered ({reason}); mark it `#[inline(always)]` so its body \
                 is compiled into the caller, or keep the pointer out of the helper's signature"
            )));
        }
        if let Some(helper) = self.builders.helpers.get(callee).copied() {
            self.call_helper_void(out, helper, args)?;
            return Ok(true);
        }
        if callee.contains("llvm.memset") {
            self.memset(out, args)?;
            return Ok(true);
        }
        if callee.contains("llvm.memcpy") {
            self.memcpy(out, args)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Call a helper function and return its result expression.
    fn call_helper(
        &mut self,
        out: &mut naga::Block,
        helper: Helper,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        // The result expression is defined by the `Call` statement and must not
        // appear in an `Emit` range, so it is appended outside `Self::expr`.
        let result = self.func.expressions.append(
            naga::Expression::CallResult(helper.function),
            naga::Span::default(),
        );
        let arguments = self.helper_arguments(out, args)?;
        out.push(
            naga::Statement::Call {
                function: helper.function,
                arguments,
                result: Some(result),
            },
            naga::Span::default(),
        );
        if helper.ret == Ty::Void {
            return Err(TranslateError::Unsupported);
        }
        Ok(result)
    }

    /// Call a helper that returns nothing.
    fn call_helper_void(
        &mut self,
        out: &mut naga::Block,
        helper: Helper,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        let arguments = self.helper_arguments(out, args)?;
        out.push(
            naga::Statement::Call {
                function: helper.function,
                arguments,
                result: None,
            },
            naga::Span::default(),
        );
        Ok(())
    }

    /// Operands of a helper call, in signature order.
    fn helper_arguments(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<Vec<naga::Handle<naga::Expression>>, TranslateError> {
        let mut arguments = Vec::with_capacity(args.len());
        for arg in args {
            arguments.push(self.operand(out, arg)?);
        }
        Ok(arguments)
    }

    fn call_value(
        &mut self,
        out: &mut naga::Block,
        callee: &str,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if let Some(reason) = self.builders.skipped_helpers.get(callee) {
            return Err(TranslateError::UnsupportedType(format!(
                "`{callee}` cannot be lowered ({reason}); mark it `#[inline(always)]` so its body \
                 is compiled into the caller, or keep the pointer out of the helper's signature"
            )));
        }
        if let Some(helper) = self.builders.helpers.get(callee).copied() {
            return self.call_helper(out, helper, args);
        }
        if callee.contains("llvm.fptoui.sat") {
            if args.len() != 1 {
                return Err(TranslateError::Unsupported);
            }
            return self.cast(out, &args[0], naga::ScalarKind::Uint);
        }
        if let Some(fun) = min_max(callee) {
            if args.len() != 2 {
                return Err(TranslateError::Unsupported);
            }
            // `llvm.smin`/`smax` compare signed values; WGSL picks the
            // signed overload from the operand type.
            let signed = callee.contains("llvm.smin") || callee.contains("llvm.smax");
            let mut arg = self.operand(out, &args[0])?;
            let mut arg1 = self.operand(out, &args[1])?;
            if signed {
                arg = self.bitcast_sint(out, arg);
                arg1 = self.bitcast_sint(out, arg1);
            }
            let value = self.expr(
                out,
                naga::Expression::Math {
                    fun,
                    arg,
                    arg1: Some(arg1),
                    arg2: None,
                    arg3: None,
                },
            );
            return Ok(if signed {
                self.bitcast_uint(out, value)
            } else {
                value
            });
        }
        if callee.contains("global_invocation_lane") {
            return self.global_invocation_lane(out, args);
        }
        if callee.contains("vertex_index") {
            return self.vertex_index.ok_or(TranslateError::Unsupported);
        }
        if callee.contains("instance_index") {
            return self.instance_index.ok_or(TranslateError::Unsupported);
        }
        if callee.contains("local_invocation_index") {
            return self.local_index.ok_or(TranslateError::Unsupported);
        }
        if callee.contains("fragment_uv_lane") {
            return self.input_lane(out, 0, args.first().ok_or(TranslateError::Unsupported)?);
        }
        if callee.contains("vertex_input_lane") || callee.contains("fragment_input_lane") {
            if args.len() != INPUT_LANE_ARGS {
                return Err(TranslateError::Unsupported);
            }
            return self.input_lane(out, const_u32(&args[0])?, &args[2]);
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
        if callee.contains("texture_sample_level_2d_f32x4") {
            return self.texture_sample_lane(out, args);
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
        if callee.contains("dot2") && args.len() == VEC2_LANES as usize + VEC2_LANES as usize {
            let va = self.vec2_of(out, args, 0)?;
            let vb = self.vec2_of(out, args, VEC2_LANES)?;
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
        if let Some(fun) = binary_math(callee, args.len()) {
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

    fn vec2_of(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
        at: u32,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let start = at as usize;
        let ty = vec_f32_type(&mut self.builders.module, VEC2_LANES)
            .ok_or(TranslateError::Unsupported)?;
        let x = self.operand(out, args.get(start).ok_or(TranslateError::Unsupported)?)?;
        let y = self.operand(out, args.get(start + 1).ok_or(TranslateError::Unsupported)?)?;
        Ok(self.expr(
            out,
            naga::Expression::Compose {
                ty,
                components: vec![x, y],
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

    /// Lane of `@builtin(global_invocation_id)`. The axis is a constant `0..3`.
    fn global_invocation_lane(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let axis = const_u32(args.first().ok_or(TranslateError::Unsupported)?)?;
        if axis >= VEC3_LANES {
            return Err(TranslateError::Unsupported);
        }
        let global_id = self.global_id.ok_or(TranslateError::Unsupported)?;
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: global_id,
                index: axis,
            },
        ))
    }

    /// Lane of a declared `@location(location)` input. Both the location and
    /// the lane are constants, and a lane the declared width does not cover is
    /// rejected instead of reading an undeclared component.
    fn input_lane(
        &mut self,
        out: &mut naga::Block,
        location: u32,
        lane: &Operand,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let lane = const_u32(lane)?;
        let lanes = *self
            .io
            .inputs
            .get(&location)
            .ok_or(TranslateError::Unsupported)?;
        if lane >= lanes {
            return Err(TranslateError::Unsupported);
        }
        let input = *self
            .inputs
            .get(&location)
            .ok_or(TranslateError::Unsupported)?;
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: input,
                index: lane,
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
        let multisampled = self.builders.sampling.multisampled_at(group, binding);
        self.image_load(out, image, multisampled, x, y)
    }

    /// `texture_sample_level_2d_f32x4(group, binding, sampler_group,
    /// sampler_binding, x, y, lane)`: one lane of `textureSampleLevel(…, 0.0)`.
    fn texture_sample_lane(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        if args.len() != 7 {
            return Err(TranslateError::Unsupported);
        }
        let lane = const_u32(&args[6])?;
        if lane >= VEC4_LANES {
            return Err(TranslateError::Unsupported);
        }
        let group = const_u32(&args[0])?;
        let binding = const_u32(&args[1])?;
        let sampler_group = const_u32(&args[2])?;
        let sampler_binding = const_u32(&args[3])?;
        let x = self.operand(out, &args[4])?;
        let y = self.operand(out, &args[5])?;
        let coord_ty = vec_f32_type(&mut self.builders.module, VEC2_LANES)
            .ok_or(TranslateError::Unsupported)?;
        let coordinate = self.expr(
            out,
            naga::Expression::Compose {
                ty: coord_ty,
                components: vec![x, y],
            },
        );
        let image = self
            .builders
            .texture(group, binding, TextureKind::Sampled)?;
        let image = self.expr(out, naga::Expression::GlobalVariable(image));
        let sampler = self.builders.sampler(sampler_group, sampler_binding);
        let sampler = self.expr(out, naga::Expression::GlobalVariable(sampler));
        let level = self.expr(out, naga::Expression::Literal(naga::Literal::F32(0.0)));
        let texel = self.expr(
            out,
            naga::Expression::ImageSample {
                image,
                sampler,
                gather: None,
                coordinate,
                array_index: None,
                offset: None,
                level: naga::SampleLevel::Exact(level),
                depth_ref: None,
                clamp_to_edge: false,
            },
        );
        Ok(self.expr(
            out,
            naga::Expression::AccessIndex {
                base: texel,
                index: lane,
            },
        ))
    }

    /// `texture_load_2d_f32x4(group, binding, x, y, lane)` at mip 0.
    /// One lane of the `vec4<f32>` a sampled `texture_2d<f32>` returns.
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
        let multisampled = self.builders.sampling.multisampled_at(group, binding);
        let texel = self.image_load(out, image, multisampled, x, y)?;
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
        multisampled: bool,
        x: &Operand,
        y: &Operand,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let coordinate = self.texel_coord(out, x, y)?;
        let (sample, level) = if multisampled {
            // A multisampled image has samples, not mip levels. Sample 0 is
            // what the renderer binds for the 4x AO pass.
            let sample = self.expr(out, naga::Expression::Literal(naga::Literal::U32(0)));
            (Some(sample), None)
        } else {
            let level = self.expr(out, naga::Expression::Literal(naga::Literal::I32(0)));
            (None, Some(level))
        };
        let handle = self.expr(out, naga::Expression::GlobalVariable(image));
        Ok(self.expr(
            out,
            naga::Expression::ImageLoad {
                image: handle,
                coordinate,
                array_index: None,
                sample,
                level,
            },
        ))
    }

    /// `texture_width_2d` / `texture_height_2d(group, binding)`.
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

    fn cast(
        &mut self,
        out: &mut naga::Block,
        value: &Operand,
        kind: naga::ScalarKind,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let expr = self.operand(out, value)?;
        Ok(self.expr(
            out,
            naga::Expression::As {
                expr,
                kind,
                convert: Some(SCALAR_WIDTH),
            },
        ))
    }

    /// `sitofp`: reinterpret the stored `u32` as `i32`, then convert to
    /// `kind`; a plain `As` from `u32` would convert the unsigned value.
    fn cast_signed(
        &mut self,
        out: &mut naga::Block,
        value: &Operand,
        kind: naga::ScalarKind,
    ) -> Result<naga::Handle<naga::Expression>, TranslateError> {
        let expr = self.operand(out, value)?;
        let expr = self.bitcast_sint(out, expr);
        Ok(self.expr(
            out,
            naga::Expression::As {
                expr,
                kind,
                convert: Some(SCALAR_WIDTH),
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
        if self.staged_void_call(out, callee, args)? {
            return Ok(());
        }
        if callee.contains("workgroup_atomic_store_u32") {
            return self.workgroup_store(out, args, true);
        }
        if callee.contains("workgroup_store_u32") {
            return self.workgroup_store(out, args, false);
        }
        if callee.contains("workgroup_barrier") {
            // Uniformity is not proved: every invocation of a workgroup must
            // reach a barrier together, so any enclosing `if`/`loop`, or a
            // merge that a returning arm can skip, is rejected instead of
            // emitting WGSL Tint refuses (or native UB).
            if self.divergent_depth > 0 || self.diverged {
                return Err(TranslateError::Unsupported);
            }
            out.push(
                naga::Statement::ControlBarrier(naga::Barrier::WORK_GROUP),
                naga::Span::default(),
            );
            return Ok(());
        }
        if callee.contains("vertex_position") {
            if args.len() != POSITION_LANES as usize {
                return Err(TranslateError::Unsupported);
            }
            return self.position_store(out, args);
        }
        if callee.contains("vertex_store_f32x2")
            || callee.contains("vertex_store_f32x4")
            || callee.contains("fragment_store_location")
        {
            if args.len() < 2 {
                return Err(TranslateError::Unsupported);
            }
            return self.store_lanes(out, const_u32(&args[0])?, &args[1..]);
        }
        if callee.contains("fragment_store_f32x4") {
            return self.store_lanes(out, 0, args);
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

    /// `vertex_position(x, y, z, w)` into the clip-space output slot.
    fn position_store(
        &mut self,
        out: &mut naga::Block,
        args: &[Operand],
    ) -> Result<(), TranslateError> {
        let member = self
            .stage_out
            .as_ref()
            .and_then(|stage_out| stage_out.position)
            .ok_or(TranslateError::Unsupported)?;
        self.store_slot(out, OutSlot::Position, member, args)
    }

    /// Store `lanes` operand lanes into the `@location(location)` slot.
    fn store_lanes(
        &mut self,
        out: &mut naga::Block,
        location: u32,
        lanes: &[Operand],
    ) -> Result<(), TranslateError> {
        let member = match self
            .stage_out
            .as_ref()
            .ok_or(TranslateError::Unsupported)?
            .locations
            .as_ref()
        {
            Some(members) => members.get(&OutSlot::Location(location)).copied(),
            // The fragment's single target stores into the returned `vec4`.
            None if location == 0 => Some(0),
            None => None,
        }
        .ok_or(TranslateError::Unsupported)?;
        self.store_slot(out, OutSlot::Location(location), member, lanes)
    }

    /// Compose `lanes` into one `vecN<f32>` and store it into output `slot`.
    fn store_slot(
        &mut self,
        out: &mut naga::Block,
        slot: OutSlot,
        member: u32,
        lanes: &[Operand],
    ) -> Result<(), TranslateError> {
        let Some(stage_out) = &self.stage_out else {
            return Err(TranslateError::Unsupported);
        };
        let var = stage_out.var;
        if let OutSlot::Location(location) = slot
            && self.io.outputs.get(&location) != Some(&(lanes.len() as u32))
        {
            return Err(TranslateError::Unsupported);
        }
        let mut components = Vec::with_capacity(lanes.len());
        for lane in lanes {
            components.push(self.operand(out, lane)?);
        }
        let width = components.len() as u32;
        let ty =
            vec_f32_type(&mut self.builders.module, width).ok_or(TranslateError::Unsupported)?;
        let value = self.expr(out, naga::Expression::Compose { ty, components });
        let base = self.expr(out, naga::Expression::LocalVariable(var));
        let pointer = if self
            .stage_out
            .as_ref()
            .is_some_and(|stage_out| stage_out.locations.is_none())
        {
            base
        } else {
            self.expr(
                out,
                naga::Expression::AccessIndex {
                    base,
                    index: member,
                },
            )
        };
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
            // A module symbol is a panic location; the call carrying it is
            // dropped, and a zero keeps the value well-typed if one surfaces.
            Operand::Symbol(_) => self.u32_literal(out, 0),
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
        push_emit(&mut self.func, out, handle);
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
fn image_class(kind: TextureKind, multisampled: bool) -> naga::ImageClass {
    match kind {
        TextureKind::Depth => naga::ImageClass::Depth {
            multi: multisampled,
        },
        TextureKind::Sampled => naga::ImageClass::Sampled {
            kind: naga::ScalarKind::Float,
            multi: multisampled,
        },
    }
}

/// Two-operand math builtins the shader SDK imports.
fn binary_math(callee: &str, argc: usize) -> Option<naga::MathFunction> {
    if argc != VEC2_LANES as usize {
        return None;
    }
    if callee.contains(ornis_gpu_abi::symbols::GPU_POW) {
        Some(naga::MathFunction::Pow)
    } else if callee.contains(ornis_gpu_abi::symbols::GPU_ATAN2) {
        Some(naga::MathFunction::Atan2)
    } else {
        None
    }
}

fn ternary_math(callee: &str, argc: usize) -> Option<naga::MathFunction> {
    if argc != VEC3_LANES as usize {
        return None;
    }
    if callee.contains("clamp") {
        Some(naga::MathFunction::Clamp)
    } else if callee.contains(ornis_gpu_abi::symbols::GPU_SMOOTHSTEP) {
        Some(naga::MathFunction::SmoothStep)
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
        ("fract", naga::MathFunction::Fract),
        ("acos", naga::MathFunction::Acos),
        ("sin", naga::MathFunction::Sin),
        ("cos", naga::MathFunction::Cos),
        ("exp2", naga::MathFunction::Exp2),
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
        || callee.contains("llvm.minimumnum")
        || callee.contains("llvm.minnum")
    {
        Some(naga::MathFunction::Min)
    } else if callee.contains("llvm.umax")
        || callee.contains("llvm.smax")
        || callee.contains("llvm.maximumnum")
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
        Operand::Local(_) | Operand::F32(_) | Operand::Bool(_) | Operand::Symbol(_) => {
            Err(TranslateError::Unsupported)
        }
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

fn uses_call(func: &LlvmFunc, needle: &str) -> bool {
    func.blocks.iter().any(|block| {
        block.instrs.iter().any(|instr| match instr {
            Instr::Call { callee, .. } | Instr::CallVoid { callee, .. } => callee.contains(needle),
            _ => false,
        })
    })
}

fn local_index_arg(
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
fn builtin_u32_arg(
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
fn location_arg(
    function: &mut naga::Function,
    ty: naga::Handle<naga::Type>,
    location: u32,
    name: &str,
) -> naga::Handle<naga::Expression> {
    let index = function.arguments.len() as u32;
    function.arguments.push(naga::FunctionArgument {
        name: Some(name.to_string()),
        ty,
        binding: Some(location_binding(location)),
    });
    append(function, naga::Expression::FunctionArgument(index))
}

fn position_arg(
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

fn append(func: &mut naga::Function, expr: naga::Expression) -> naga::Handle<naga::Expression> {
    func.expressions.append(expr, naga::Span::default())
}

fn push_emit(
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
        // A module symbol is a panic location, and the call that carried it is
        // dropped; a zero keeps the value well-typed if one ever surfaces.
        Operand::Symbol(_) => {
            let handle = append(func, naga::Expression::Literal(naga::Literal::U32(0)));
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
        let func = crate::parse::parse_module(ir)
            .expect("parse")
            .funcs
            .into_iter()
            .find(|func| func.name == ornis_gpu_abi::symbols::CS_MAIN)
            .expect("entry");
        // `vec3` is 12 bytes but aligns to 16; the scalar after it sits at
        // byte 12 and the struct rounds up to 32 (uniform alignment).
        let structs: &[&[u8]] = &[&[TYPE_TAG_VEC3, TYPE_TAG_F32, TYPE_TAG_U32]];
        let mut lowering = Lowering::new(structs, crate::TextureSampling::SingleSample);
        lowering
            .push(&func, ornis_gpu_abi::symbols::CS_WORKGROUP)
            .expect("lower");
        let module = lowering.finish();
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
