//! Uniformity and value-flow classification of operands and regions.

use super::SCALAR_WIDTH;
use super::VEC2_LANES;
use super::VEC3_LANES;
use super::VEC4_LANES;
use super::cfg::Flow;
use crate::error::TranslateError;
use crate::parse::{Instr, LlvmFunc, Operand};
use crate::symbol;
use std::collections::HashSet;

/// A `loop` with no continuing block and no `break if`.
pub(crate) fn loop_statement(body: naga::Block) -> naga::Statement {
    naga::Statement::Loop {
        body,
        continuing: naga::Block::new(),
        break_if: None,
    }
}

/// One `switch` case; the value comes from the IR, the body was just emitted.
pub(crate) fn switch_case(
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
pub(crate) fn case_flow_returns(flow: Flow) -> Result<bool, TranslateError> {
    match flow {
        Flow::Returned => Ok(true),
        Flow::Next => Ok(false),
        Flow::Continue | Flow::Break => Err(TranslateError::unsupported()),
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
pub(crate) fn uniform_values(func: &LlvmFunc) -> HashSet<String> {
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
pub(crate) fn instruction_is_uniform(instr: &Instr, uniform: &HashSet<String>) -> bool {
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

/// The imports that return the same value in every invocation, spelled out.
///
/// Uniform struct reads (the whole `uniform_*` family), the dimension queries,
/// the length queries and the pure scalar math. Everything else the SDK
/// exports is per-invocation by construction — invocation ids, workgroup
/// memory, storage reads, texture fetches — so it is absent here, and a name
/// that only *sounds* uniform (`uniform_helper` in a future helper crate) is
/// not granted uniformity by its prefix. The `llvm.*` intrinsics are pure
/// functions of their operands, so they stay uniform whenever their operands
/// are; the type suffix is why they match by prefix rather than by name.
pub(crate) const UNIFORM_IMPORTS: &[&str] = &[
    // Uniform block reads. Field access with a runtime index is uniform
    // because WGSL allows it on a fixed-size uniform array and the block
    // cannot change mid-pass.
    ornis_gpu_abi::symbols::UNIFORM_READ_STRUCT,
    ornis_gpu_abi::symbols::UNIFORM_INDEX_STRUCT,
    ornis_gpu_abi::symbols::UNIFORM_FIELD_F32,
    ornis_gpu_abi::symbols::UNIFORM_FIELD_U32,
    ornis_gpu_abi::symbols::UNIFORM_FIELD_I32,
    ornis_gpu_abi::symbols::UNIFORM_FIELD_VEC2,
    ornis_gpu_abi::symbols::UNIFORM_FIELD_VEC3,
    ornis_gpu_abi::symbols::UNIFORM_FIELD_VEC4,
    ornis_gpu_abi::symbols::UNIFORM_ARRAY_VEC4_LANE,
    ornis_gpu_abi::symbols::UNIFORM_NESTED_ARRAY_VEC4_LANE,
    ornis_gpu_abi::symbols::UNIFORM_NESTED_VEC4_LANE,
    ornis_gpu_abi::symbols::UNIFORM_READ_F32,
    ornis_gpu_abi::symbols::UNIFORM_READ_U32,
    ornis_gpu_abi::symbols::UNIFORM_READ_I32,
    ornis_gpu_abi::symbols::UNIFORM_READ_VEC2,
    ornis_gpu_abi::symbols::UNIFORM_READ_VEC3,
    ornis_gpu_abi::symbols::UNIFORM_READ_VEC4,
    // The derived uniform accesses the SDK spells on top of the reads above.
    "uniform_field_vec4_component_f32",
    "uniform_field_vec4_plane_dot",
    "uniform_nested_field_f32",
    "uniform_nested_field_u32",
    // Dimension and length queries: the binding is fixed for the pass.
    ornis_gpu_abi::symbols::TEXTURE_WIDTH_2D,
    ornis_gpu_abi::symbols::TEXTURE_HEIGHT_2D,
    "texture_width_2d_f32x4",
    "texture_height_2d_f32x4",
    "storage_len_f32",
    "storage_len_u32",
    "storage_len_u32x4",
    // The pure scalar math: every `gpu_*` import the SDK declares.
    ornis_gpu_abi::symbols::GPU_SQRT,
    ornis_gpu_abi::symbols::GPU_POW,
    ornis_gpu_abi::symbols::GPU_ATAN2,
    ornis_gpu_abi::symbols::GPU_SMOOTHSTEP,
    ornis_gpu_abi::symbols::GPU_DPDX,
    ornis_gpu_abi::symbols::GPU_DPDY,
    ornis_gpu_abi::symbols::GPU_TAN,
    ornis_gpu_abi::symbols::GPU_FLOOR,
    ornis_gpu_abi::symbols::GPU_CEIL,
    ornis_gpu_abi::symbols::GPU_LN,
    ornis_gpu_abi::symbols::GPU_EXP,
    ornis_gpu_abi::symbols::GPU_EXP2,
    ornis_gpu_abi::symbols::GPU_LOG2,
    ornis_gpu_abi::symbols::GPU_ABS,
    ornis_gpu_abi::symbols::GPU_SIGN,
    ornis_gpu_abi::symbols::GPU_FRACT,
    ornis_gpu_abi::symbols::GPU_COS,
    ornis_gpu_abi::symbols::GPU_SIN,
    ornis_gpu_abi::symbols::GPU_ACOS,
    ornis_gpu_abi::symbols::GPU_CLAMP,
    ornis_gpu_abi::symbols::GPU_MIX,
    ornis_gpu_abi::symbols::GPU_DOT2_XY,
    ornis_gpu_abi::symbols::GPU_DOT3_XYZ,
    ornis_gpu_abi::symbols::GPU_LENGTH3_XYZ,
    ornis_gpu_abi::symbols::GPU_CROSS3_LANE,
    ornis_gpu_abi::symbols::GPU_NORMALIZE3_LANE,
];

/// Whether an import returns the same value in every invocation.
///
/// [`UNIFORM_IMPORTS`] is the whole answer; anything not in it is
/// per-invocation, and a name is never granted uniformity by a prefix.
pub(crate) fn uniform_callee(callee: &str) -> bool {
    let symbol = symbol::imported(callee);
    UNIFORM_IMPORTS.contains(&symbol) || symbol.starts_with("llvm.")
}

/// `vecN<f32>` of a stage-interface value, or `None` for a width the
/// translator does not declare.
pub(crate) fn vec_f32_type(
    module: &mut naga::Module,
    lanes: u32,
) -> Option<naga::Handle<naga::Type>> {
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
pub(crate) fn vec4_f32_type(module: &mut naga::Module) -> naga::Handle<naga::Type> {
    module.types.insert(
        naga::Type {
            name: None,
            inner: vector_f32(naga::VectorSize::Quad),
        },
        naga::Span::default(),
    )
}

/// A `vecN<f32>` type body.
pub(crate) fn vector_f32(size: naga::VectorSize) -> naga::TypeInner {
    naga::TypeInner::Vector {
        size,
        scalar: naga::Scalar {
            kind: naga::ScalarKind::Float,
            width: SCALAR_WIDTH,
        },
    }
}
