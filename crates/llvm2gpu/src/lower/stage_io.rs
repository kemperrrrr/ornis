//! Stage inputs and outputs: lane reads, bindings, layout and entry point.

use super::VEC2_LANES;
use super::VEC3_LANES;
use super::VEC4_LANES;
use super::WORD_BYTES;
use super::emit::{
    builtin_u32_arg, const_u32, invocation_x, local_index_arg, location_arg, position_arg,
    uses_call,
};
use super::flow::vec4_f32_type;
use super::memory::{StageOut, output_local};
use super::stage::{IoValue, OutSlot, Stage, StageIo};
use crate::error::TranslateError;
use crate::parse::{LlvmFunc, Operand, Ty};
use std::collections::BTreeMap;

/// Operands of a `*_input_lane` import: location, declared lanes, and the lane.
pub(crate) const INPUT_LANE_ARGS: usize = 3;

/// `(location, lanes)` immediates of a `*_input_lane` call.
///
/// The lane being read must also be inside the declared width; that check is
/// in [`Lowerer::input_lane`], which sees the same immediates.
pub(crate) fn input_lanes(args: &[Operand]) -> Result<(u32, u32), TranslateError> {
    if args.len() != INPUT_LANE_ARGS {
        return Err(TranslateError::unsupported());
    }
    Ok((const_u32(&args[0])?, const_u32(&args[1])?))
}

/// Location immediate of a `*_store_*(location, …)` call.
pub(crate) fn location_of(args: &[Operand]) -> Result<u32, TranslateError> {
    const_u32(args.first().ok_or_else(TranslateError::unsupported)?)
}

/// Record that `location` carries `value`.
pub(crate) fn record_io(
    map: &mut BTreeMap<u32, IoValue>,
    location: u32,
    value: IoValue,
) -> Result<(), TranslateError> {
    if let IoValue::Vec(lanes) = value
        && !(VEC2_LANES..=VEC4_LANES).contains(&lanes)
    {
        return Err(TranslateError::UnsupportedType(format!(
            "@location({location}) needs two to four lanes, got {lanes}"
        )));
    }
    match map.get(&location) {
        Some(&existing) if existing != value => Err(TranslateError::UnsupportedType(format!(
            "@location({location}) is used as both {existing:?} and {value:?}"
        ))),
        Some(_) => Ok(()),
        None => {
            map.insert(location, value);
            Ok(())
        }
    }
}

/// `@location(n)` with the perspective interpolation every pass uses today.
pub(crate) fn location_binding(location: u32) -> naga::Binding {
    naga::Binding::Location {
        location,
        interpolation: Some(naga::Interpolation::Perspective),
        sampling: Some(naga::Sampling::Center),
        blend_src: None,
        per_primitive: false,
    }
}

/// `@location` binding of one interface value.
///
/// An integer varying is `@interpolate(flat)` — WGSL forbids interpolating a
/// `u32` — and a fragment output is never interpolated.
pub(crate) fn value_binding(location: u32, value: IoValue, output: bool) -> naga::Binding {
    let mut binding = location_binding(location);
    if let naga::Binding::Location {
        interpolation,
        sampling,
        ..
    } = &mut binding
    {
        match (value, output) {
            (IoValue::U32 | IoValue::U32x4, false) => {
                *interpolation = Some(naga::Interpolation::Flat);
                *sampling = None;
            }
            (_, true) => {
                *interpolation = None;
                *sampling = None;
            }
            _ => {}
        }
    }
    binding
}

/// Stage-output struct under construction: members, their layout, and the
/// member index of every output slot.
#[derive(Default)]
pub(crate) struct OutputLayout {
    /// Struct members in declaration order.
    pub(crate) members: Vec<naga::StructMember>,
    /// Size and alignment of each member, in the same order.
    pub(crate) specs: Vec<ornis_gpu_abi::FieldLayout>,
    /// Member index per output slot.
    pub(crate) slots: BTreeMap<OutSlot, u32>,
}

impl OutputLayout {
    /// Append the member for `slot`: `value` of `ty` at `binding`.
    ///
    /// The struct is laid out like the WGSL one: a `vec3` occupies 12 bytes at
    /// alignment 16, so its member cannot be padded to four lanes.
    pub(crate) fn push(
        &mut self,
        slot: OutSlot,
        value: IoValue,
        ty: naga::Handle<naga::Type>,
        binding: naga::Binding,
    ) {
        let name = match slot {
            OutSlot::Position => "position".to_string(),
            OutSlot::Location(location) => format!("o{location}"),
        };
        let (size, align) = match value {
            IoValue::Vec(VEC2_LANES) => (VEC2_LANES * WORD_BYTES, VEC2_LANES * WORD_BYTES),
            // A `vec3` is three words wide at a four-word alignment.
            IoValue::Vec(VEC3_LANES) => (VEC3_LANES * WORD_BYTES, VEC4_LANES * WORD_BYTES),
            IoValue::Vec(_) => (VEC4_LANES * WORD_BYTES, VEC4_LANES * WORD_BYTES),
            IoValue::U32 => (WORD_BYTES, WORD_BYTES),
            IoValue::U32x4 => (VEC4_LANES * WORD_BYTES, VEC4_LANES * WORD_BYTES),
        };
        self.specs.push(ornis_gpu_abi::FieldLayout { size, align });
        self.slots.insert(slot, self.members.len() as u32);
        self.members.push(naga::StructMember {
            name: Some(name),
            ty,
            binding: Some(binding),
            offset: 0,
        });
    }

    /// Size of the struct with every member offset filled in.
    pub(crate) fn span(&mut self) -> u32 {
        let mut offsets = vec![0u32; self.specs.len()];
        let span = ornis_gpu_abi::place_fields(&self.specs, true, &mut offsets);
        for (member, offset) in self.members.iter_mut().zip(&offsets) {
            member.offset = *offset;
        }
        span
    }
}

/// Name the emitted stage-output struct carries in WGSL.
pub(crate) fn output_struct_name(stage: Stage) -> &'static str {
    if stage == Stage::Vertex {
        "VertexOut"
    } else {
        "FragmentOut"
    }
}

/// A raster entry's result type, its located members, and `@builtin(position)`.
pub(crate) struct OutputType {
    /// The emitted struct type.
    pub(crate) ty: naga::Handle<naga::Type>,
    /// Member index per output slot.
    pub(crate) slots: BTreeMap<OutSlot, u32>,
    /// Member index of `@builtin(position)`, absent for a fragment.
    pub(crate) position: Option<u32>,
}

/// Entry arguments a raster or compute body reads.
pub(crate) struct StageArguments {
    /// `global_invocation_id` vector, for a compute entry.
    pub(crate) global_id: Option<naga::Handle<naga::Expression>>,
    /// Lane 0 of `global_invocation_id`, the compute entry's `i32` parameter.
    pub(crate) id_x: Option<naga::Handle<naga::Expression>>,
    /// `@builtin(local_invocation_index)`.
    pub(crate) local_index: Option<naga::Handle<naga::Expression>>,
    /// `@builtin(vertex_index)`.
    pub(crate) vertex_index: Option<naga::Handle<naga::Expression>>,
    /// `@builtin(instance_index)`.
    pub(crate) instance_index: Option<naga::Handle<naga::Expression>>,
    /// `@builtin(position)` of a fragment entry.
    pub(crate) position: Option<naga::Handle<naga::Expression>>,
    /// `@location` arguments by location.
    pub(crate) inputs: BTreeMap<u32, naga::Handle<naga::Expression>>,
}

/// Stage of `func`, after checking the shape rustc compiles for it.
///
/// A compute entry takes nothing or one `i32` (its invocation id) and may
/// return `f32`; a raster entry is `void` with no parameters, because its
/// builtins and interface come from the stage-IO imports.
pub(crate) fn entry_stage(func: &LlvmFunc) -> Result<Stage, TranslateError> {
    // The stage comes from the *Rust* name, not the exported symbol: the export
    // is a link-time attribute, while `#[gpu_entry]` names the symbol after the
    // function. A name without one of the canonical prefixes is rejected here,
    // so the message names the function and the prefixes rather than an
    // instruction it was not reading.
    let Some(stage) = Stage::of(&func.name) else {
        return Err(TranslateError::Unsupported(format!(
            "for {}: an entry name must start with {}",
            crate::site::describe(),
            Stage::PREFIX_LIST
        )));
    };
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
        Err(TranslateError::unsupported())
    }
}

/// Declare the entry arguments the body reads: builtins and `@location`s.
pub(crate) fn stage_arguments(
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
    for (&location, &value) in &io.inputs {
        let ty = value.ty(module)?;
        // A fragment's location-0 two-lane interpolant keeps the `uv`
        // parameter name the checked-in AO artifacts already carry.
        let legacy_uv =
            stage == Stage::Fragment && location == 0 && value == IoValue::floats(VEC2_LANES);
        let name = if legacy_uv {
            "uv".to_string()
        } else {
            format!("in{location}")
        };
        let binding = value_binding(location, value, false);
        inputs.insert(location, location_arg(function, ty, binding, &name));
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
pub(crate) fn stage_output(
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
        && io.outputs.get(&0) == Some(&IoValue::floats(VEC4_LANES))
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
        return Err(TranslateError::unsupported());
    }
    // A vertex entry may write `@builtin(position)` alone — a blit has no
    // varying to interpolate — and the struct then has that one member, which
    // is valid WGSL. A fragment entry with no `@location` writes nothing at
    // all, so it is still rejected.
    if io.outputs.is_empty() && !io.position_out {
        return Err(TranslateError::unsupported());
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
pub(crate) fn output_struct(
    module: &mut naga::Module,
    stage: Stage,
    io: &StageIo,
) -> Result<OutputType, TranslateError> {
    let mut layout = OutputLayout::default();
    if stage == Stage::Vertex {
        layout.push(
            OutSlot::Position,
            IoValue::floats(VEC4_LANES),
            vec4_f32_type(module),
            naga::Binding::BuiltIn(naga::BuiltIn::Position { invariant: false }),
        );
    }
    for (&location, &value) in &io.outputs {
        let ty = value.ty(module)?;
        let binding = value_binding(location, value, stage == Stage::Fragment);
        layout.push(OutSlot::Location(location), value, ty, binding);
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
