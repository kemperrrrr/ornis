//! Entry stages, their IO imports and the stage-IO scan.

use super::SCALAR_WIDTH;
use super::VEC2_LANES;
use super::VEC3_LANES;
use super::VEC4_LANES;
use super::emit::scalar;
use super::flow::vec_f32_type;
use super::stage_io::{input_lanes, location_of, record_io};
use crate::error::TranslateError;
use crate::parse::{Instr, LlvmFunc};
use crate::symbol;
use std::collections::BTreeMap;

/// Shader stage the entry function belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    /// `cs_main`: one invocation per `@workgroup_size` thread.
    Compute,
    /// `vs_main`: one invocation per vertex.
    Vertex,
    /// `fs_main`: one invocation per covered sample.
    Fragment,
}

impl Stage {
    /// Canonical entry prefixes, one per stage.
    ///
    /// A pass may carry more than one entry of a stage (`vs_main_indirect`
    /// draws the compacted visibility list, `fs_main_msaa` shades samples), and
    /// the pipeline names the entry it wants. The prefix decides the stage; the
    /// rest of the name only has to be unique within the module.
    /// The canonical prefixes, spelled out for a rejection message.
    pub(crate) const PREFIX_LIST: &'static str = "`cs_main`, `vs_main` or `fs_main`";

    const PREFIXES: [(&'static str, Self); 3] = [
        (ornis_gpu_abi::symbols::CS_MAIN, Self::Compute),
        (ornis_gpu_abi::symbols::VS_MAIN, Self::Vertex),
        (ornis_gpu_abi::symbols::FS_MAIN, Self::Fragment),
    ];

    /// Stage an entry name denotes, or `None` for anything else.
    pub(crate) fn of(name: &str) -> Option<Self> {
        Self::PREFIXES
            .iter()
            .find(|(prefix, _)| name.starts_with(prefix))
            .map(|(_, stage)| *stage)
    }

    /// Whether the stage draws fragments, i.e. has a stage interface.
    pub(crate) fn raster(self) -> bool {
        !matches!(self, Self::Compute)
    }
}

/// Slot of a stage output: the clip position or one `@location`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum OutSlot {
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
pub(crate) struct StageIo {
    /// `@builtin(vertex_index)` is read.
    pub(crate) vertex_index: bool,
    /// `@builtin(instance_index)` is read.
    pub(crate) instance_index: bool,
    /// `@builtin(position)` of a vertex entry is written.
    pub(crate) position_out: bool,
    /// Declared inputs: location -> value kind.
    pub(crate) inputs: BTreeMap<u32, IoValue>,
    /// Written outputs: location -> value kind.
    pub(crate) outputs: BTreeMap<u32, IoValue>,
}

/// Type of one `@location` interface value.
///
/// The subset had only `vecN<f32>`; an integer attachment or a flat varying
/// carries one `u32`, so the declared type is per location instead of a lane
/// count every value shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IoValue {
    /// `vecN<f32>` with `N` in `2..=4`.
    Vec(u32),
    /// Scalar `u32`: `@interpolate(flat)` as a varying, a plain `u32` target.
    U32,
    /// `vec4<u32>`: one flat four-word varying (a `vec4<u32>` joint index).
    U32x4,
}

impl IoValue {
    /// The value a `LANES`-lane float import declares.
    pub(crate) fn floats(lanes: u32) -> Self {
        Self::Vec(lanes)
    }

    /// Naga type of this value in `module`.
    pub(crate) fn ty(
        self,
        module: &mut naga::Module,
    ) -> Result<naga::Handle<naga::Type>, TranslateError> {
        match self {
            Self::Vec(lanes) => vec_f32_type(module, lanes).ok_or_else(TranslateError::unsupported),
            Self::U32 => Ok(scalar(module, naga::ScalarKind::Uint)),
            Self::U32x4 => Ok(module.types.insert(
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
            )),
        }
    }
}

/// The stage-interface import a call names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IoImport {
    /// `@builtin(vertex_index)`.
    VertexIndex,
    /// `@builtin(instance_index)`.
    InstanceIndex,
    /// Lane of a vertex `@location` input.
    VertexInput,
    /// Scalar `u32` vertex `@location` input.
    VertexInputU32,
    /// Lane of a `vec4<u32>` vertex `@location` input.
    VertexInputU32x4,
    /// Clip-space `@builtin(position)` of a vertex entry.
    VertexPosition,
    /// Two-lane vertex `@location` output.
    VertexOutput2,
    /// Three-lane vertex `@location` output.
    VertexOutput3,
    /// Four-lane vertex `@location` output.
    VertexOutput4,
    /// Scalar `u32` vertex `@location` output.
    VertexOutputU32,
    /// `@builtin(position)` of a fragment entry.
    FragmentPosition,
    /// Lane of a fragment `@location` input.
    FragmentInput,
    /// Scalar `u32` fragment `@location` input.
    FragmentInputU32,
    /// Lane of the fragment's `@location(0)` vec2, the AO-pass shape.
    FragmentUv,
    /// Fragment output at a named location.
    FragmentOutput,
    /// Two-lane fragment output at a named location.
    FragmentOutput2,
    /// Scalar `u32` fragment output at a named location.
    FragmentOutputU32,
    /// Fragment output at `@location(0)`, the bare-vec4 shape.
    FragmentOutput0,
}

/// Order matters only where one import name is a prefix of another, which is
/// why the located fragment output precedes the `@location(0)` one.
pub(crate) const IO_IMPORTS: [(&str, IoImport); 18] = [
    (
        ornis_gpu_abi::symbols::VERTEX_INPUT_LANE,
        IoImport::VertexInput,
    ),
    // `vertex_input_u32` is a prefix of `vertex_input_u32x4`: the longer
    // name goes first, because the lookup is a substring match.
    (
        ornis_gpu_abi::symbols::VERTEX_INPUT_U32X4,
        IoImport::VertexInputU32x4,
    ),
    (
        ornis_gpu_abi::symbols::VERTEX_INPUT_U32,
        IoImport::VertexInputU32,
    ),
    (
        ornis_gpu_abi::symbols::VERTEX_STORE_F32X2,
        IoImport::VertexOutput2,
    ),
    (
        ornis_gpu_abi::symbols::VERTEX_STORE_F32X3,
        IoImport::VertexOutput3,
    ),
    (
        ornis_gpu_abi::symbols::VERTEX_STORE_F32X4,
        IoImport::VertexOutput4,
    ),
    (
        ornis_gpu_abi::symbols::VERTEX_STORE_U32,
        IoImport::VertexOutputU32,
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
        ornis_gpu_abi::symbols::FRAGMENT_STORE_F32X2_LOCATION,
        IoImport::FragmentOutput2,
    ),
    (
        ornis_gpu_abi::symbols::FRAGMENT_STORE_U32_LOCATION,
        IoImport::FragmentOutputU32,
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
        ornis_gpu_abi::symbols::FRAGMENT_INPUT_U32,
        IoImport::FragmentInputU32,
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
/// Whether `callee` is a workgroup barrier import.
pub(crate) fn is_barrier(callee: &str) -> bool {
    matches!(
        symbol::imported(callee),
        "workgroup_barrier" | "storage_barrier"
    )
}

pub(crate) fn io_import(callee: &str) -> Option<IoImport> {
    IO_IMPORTS
        .iter()
        .find(|(name, _)| symbol::imported(callee) == *name)
        .map(|(_, import)| *import)
}

/// Read the stage interface back out of the body's imports.
pub(crate) fn scan_io(func: &LlvmFunc, stage: Stage) -> Result<StageIo, TranslateError> {
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
                    record_io(&mut io.inputs, location, IoValue::floats(lanes))?;
                }
                (Stage::Vertex, IoImport::VertexInputU32) => {
                    record_io(&mut io.inputs, location_of(args)?, IoValue::U32)?;
                }
                (Stage::Vertex, IoImport::VertexInputU32x4) => {
                    record_io(&mut io.inputs, location_of(args)?, IoValue::U32x4)?;
                }
                (Stage::Vertex, IoImport::VertexPosition) => io.position_out = true,
                (Stage::Vertex, IoImport::VertexOutputU32) => {
                    record_io(&mut io.outputs, location_of(args)?, IoValue::U32)?;
                }
                (
                    Stage::Vertex,
                    IoImport::VertexOutput2 | IoImport::VertexOutput3 | IoImport::VertexOutput4,
                ) => {
                    let lanes = match import {
                        IoImport::VertexOutput2 => VEC2_LANES,
                        IoImport::VertexOutput3 => VEC3_LANES,
                        _ => VEC4_LANES,
                    };
                    record_io(&mut io.outputs, location_of(args)?, IoValue::floats(lanes))?;
                }
                // The fragment's `@builtin(position)` argument is always
                // declared, so reading it needs no interface bookkeeping.
                (Stage::Fragment, IoImport::FragmentPosition) => {}
                (Stage::Fragment, IoImport::FragmentInput) => {
                    let (location, lanes) = input_lanes(args)?;
                    record_io(&mut io.inputs, location, IoValue::floats(lanes))?;
                }
                (Stage::Fragment, IoImport::FragmentInputU32) => {
                    record_io(&mut io.inputs, location_of(args)?, IoValue::U32)?;
                }
                (Stage::Fragment, IoImport::FragmentUv) => {
                    record_io(&mut io.inputs, 0, IoValue::floats(VEC2_LANES))?;
                }
                (Stage::Fragment, IoImport::FragmentOutput0) => {
                    record_io(&mut io.outputs, 0, IoValue::floats(VEC4_LANES))?;
                }
                (Stage::Fragment, IoImport::FragmentOutput) => {
                    record_io(
                        &mut io.outputs,
                        location_of(args)?,
                        IoValue::floats(VEC4_LANES),
                    )?;
                }
                (Stage::Fragment, IoImport::FragmentOutput2) => {
                    record_io(
                        &mut io.outputs,
                        location_of(args)?,
                        IoValue::floats(VEC2_LANES),
                    )?;
                }
                (Stage::Fragment, IoImport::FragmentOutputU32) => {
                    record_io(&mut io.outputs, location_of(args)?, IoValue::U32)?;
                }
                // A stage reading the other stage's interface would declare an
                // argument its entry point cannot have.
                _ => return Err(TranslateError::unsupported()),
            }
        }
    }
    Ok(io)
}
