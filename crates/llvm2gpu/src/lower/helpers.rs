//! Small Naga construction helpers shared across the lowering passes.

use super::VEC2_LANES;
use super::VEC3_LANES;
use super::emit::{append, const_u32, push_emit};
use crate::error::TranslateError;
use crate::parse::Operand;
use crate::symbol;

pub(crate) fn load_global(
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
pub(crate) fn load_uniform(
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

/// Two-operand math builtins the shader SDK imports.
/// Derivative axis of a screen-space derivative import.
///
/// WGSL allows a derivative only in uniform control flow, so the translator
/// passes the axis through and lets naga reject a divergent call rather than
/// papering over it.
pub(crate) fn derivative_axis(callee: &str) -> Option<naga::DerivativeAxis> {
    if symbol::imported(callee) == ornis_gpu_abi::symbols::GPU_DPDX {
        Some(naga::DerivativeAxis::X)
    } else if symbol::imported(callee) == ornis_gpu_abi::symbols::GPU_DPDY {
        Some(naga::DerivativeAxis::Y)
    } else {
        None
    }
}

pub(crate) fn binary_math(callee: &str, argc: usize) -> Option<naga::MathFunction> {
    if argc != VEC2_LANES as usize {
        return None;
    }
    if symbol::imported(callee) == ornis_gpu_abi::symbols::GPU_POW {
        Some(naga::MathFunction::Pow)
    } else if symbol::imported(callee) == ornis_gpu_abi::symbols::GPU_ATAN2 {
        Some(naga::MathFunction::Atan2)
    } else {
        None
    }
}

pub(crate) fn ternary_math(callee: &str, argc: usize) -> Option<naga::MathFunction> {
    if argc != VEC3_LANES as usize {
        return None;
    }
    // Both spellings: the SDK imports are `gpu_*`, and the hand-written IR of
    // the tests names the operation bare.
    let symbol = symbol::imported(callee);
    if symbol == "clamp" || symbol == ornis_gpu_abi::symbols::GPU_CLAMP {
        Some(naga::MathFunction::Clamp)
    } else if symbol == ornis_gpu_abi::symbols::GPU_SMOOTHSTEP {
        Some(naga::MathFunction::SmoothStep)
    } else if symbol == "mix" || symbol == ornis_gpu_abi::symbols::GPU_MIX {
        Some(naga::MathFunction::Mix)
    } else {
        None
    }
}

pub(crate) fn unary_math(callee: &str) -> Option<naga::MathFunction> {
    use naga::MathFunction as F;
    use ornis_gpu_abi::symbols as s;
    // One entry per import the SDK declares: the names are whole, so no entry
    // can swallow another (`gpu_exp` and `gpu_exp2` are two rows). The
    // `llvm.*` rows match by prefix because an intrinsic carries its type in
    // the name (`llvm.sqrt.f32`).
    const FNS: &[(&str, F)] = &[
        (s::GPU_SQRT, F::Sqrt),
        (s::GPU_EXP, F::Exp),
        (s::GPU_EXP2, F::Exp2),
        (s::GPU_LN, F::Log),
        (s::GPU_LOG2, F::Log2),
        (s::GPU_TAN, F::Tan),
        (s::GPU_SIN, F::Sin),
        (s::GPU_COS, F::Cos),
        (s::GPU_ACOS, F::Acos),
        (s::GPU_ABS, F::Abs),
        (s::GPU_SIGN, F::Sign),
        (s::GPU_FRACT, F::Fract),
        (s::GPU_FLOOR, F::Floor),
        (s::GPU_CEIL, F::Ceil),
    ];
    let symbol = symbol::imported(callee);
    if let Some((_, fun)) = FNS.iter().find(|(name, _)| symbol == *name) {
        return Some(*fun);
    }
    // The intrinsics: `abs_f32` as a branch is what a shader writes, and rustc
    // folds it into the intrinsic, so the intrinsic is what the translator sees.
    let intrinsic = [
        ("llvm.sqrt", F::Sqrt),
        ("llvm.fabs", F::Abs),
        ("llvm.sin", F::Sin),
        ("llvm.cos", F::Cos),
        ("llvm.exp2", F::Exp2),
        ("llvm.log2", F::Log2),
        ("llvm.floor", F::Floor),
        ("llvm.ceil", F::Ceil),
    ]
    .into_iter()
    .find(|(name, _)| symbol.starts_with(name));
    intrinsic.map(|(_, fun)| fun)
}

pub(crate) fn min_max(callee: &str) -> Option<naga::MathFunction> {
    if symbol::imported(callee).starts_with("llvm.umin")
        || symbol::imported(callee).starts_with("llvm.smin")
        || symbol::imported(callee).starts_with("llvm.minimumnum")
        || symbol::imported(callee).starts_with("llvm.minnum")
    {
        Some(naga::MathFunction::Min)
    } else if symbol::imported(callee).starts_with("llvm.umax")
        || symbol::imported(callee).starts_with("llvm.smax")
        || symbol::imported(callee).starts_with("llvm.maximumnum")
        || symbol::imported(callee).starts_with("llvm.maxnum")
    {
        Some(naga::MathFunction::Max)
    } else {
        None
    }
}

/// Lanes of `@builtin(position)`.
pub(crate) const POSITION_LANES: u32 = 4;

/// How many leading operands every uniform import uses: group, binding, struct, field.
pub(crate) const UNIFORM_PREFIX: usize = 4;

pub(crate) fn workgroup_index(args: &[Operand]) -> Result<&Operand, TranslateError> {
    args.first().ok_or_else(TranslateError::unsupported)
}

/// `texture_load_2d_f32(group, binding, x, y)`.
pub(crate) fn texture_xy(
    args: &[Operand],
) -> Result<(u32, u32, &Operand, &Operand), TranslateError> {
    if args.len() != UNIFORM_PREFIX {
        return Err(TranslateError::unsupported());
    }
    Ok((
        const_u32(&args[0])?,
        const_u32(&args[1])?,
        &args[2],
        &args[3],
    ))
}

/// `(group, binding, struct id, field)` shared by every uniform import.
pub(crate) fn uniform_ids(args: &[Operand]) -> Result<(u32, u32, u32, u32), TranslateError> {
    if args.len() < UNIFORM_PREFIX {
        return Err(TranslateError::unsupported());
    }
    Ok((
        const_u32(&args[0])?,
        const_u32(&args[1])?,
        const_u32(&args[2])?,
        const_u32(&args[3])?,
    ))
}

/// Error for a struct id the metadata never declared.
pub(crate) fn missing_struct(sid: u32) -> TranslateError {
    TranslateError::UnsupportedType(format!(
        "uniform struct id {sid} has no ornis_gpu.types entry"
    ))
}

/// Error for an ordinal past the declared field count.
pub(crate) fn missing_field(sid: u32, field: u32) -> TranslateError {
    TranslateError::UnsupportedType(format!(
        "uniform field {field} of struct {sid} is not declared"
    ))
}

/// Byte offset of field `field`'s leading tag, walking the multi-byte ones.
pub(crate) fn tag_cursor(tags: &[u8], field: u32) -> usize {
    let mut cursor = 0usize;
    let mut seen = 0u32;
    while seen < field {
        match field_tag_width(tags, cursor) {
            Some(width) => cursor += width,
            None => return cursor,
        }
        seen += 1;
    }
    cursor
}

/// Tag bytes the field at `cursor` occupies, or `None` when the list ends
/// inside that field.
///
/// Scalars and vectors are one byte, a nested struct is the tag plus its id,
/// and an array is the tag plus the element's tag (two bytes when the element
/// is a nested struct) plus the length.
pub(crate) fn field_tag_width(tags: &[u8], cursor: usize) -> Option<usize> {
    match *tags.get(cursor)? {
        ornis_gpu_abi::TYPE_TAG_ARRAY => {
            let element = *tags.get(cursor + 1)?;
            let element_width = if element >= ornis_gpu_abi::TYPE_TAG_NESTED {
                1
            } else if element == ornis_gpu_abi::TYPE_TAG_STRUCT {
                2
            } else {
                1
            };
            tags.get(cursor + 1 + element_width)?;
            Some(2 + element_width)
        }
        ornis_gpu_abi::TYPE_TAG_STRUCT => Some(2),
        _ => Some(1),
    }
}
