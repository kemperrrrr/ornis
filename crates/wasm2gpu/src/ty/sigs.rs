//! Import signatures for storage, uniform, and texture calls.

use super::Elem;
use super::FieldKind;
use super::Imp;
use super::PROV_PARAMS;
use super::STRUCT_HEADER;
use super::Spec;
use super::StructDef;
use super::Uni;
use super::field_is_float;
use super::field_lanes;
use super::struct_lanes;
use super::type_err;

use crate::error::TranslateError;
use crate::ir::Kind;
use crate::ir::Sig;

const SAMPLE_PARAMS: u32 = 6;
const SAMPLE_RESULTS: u32 = 4;
const FLOAT4: [Kind; 4] = [Kind::F32, Kind::F32, Kind::F32, Kind::F32];

pub(crate) fn sig_ok(
    sig: &Sig,
    imp: Imp,
    structs: &[StructDef],
    struct_id: Option<u16>,
) -> Result<(), TranslateError> {
    match imp {
        Imp::ReadUniform => fixed_sig(sig, PROV_PARAMS, &FLOAT4, "raw_read"),
        Imp::Sample => count_only(sig, SAMPLE_PARAMS, SAMPLE_RESULTS, "texture_sample_2d"),
        Imp::Read(Spec::Fixed(elem)) => {
            fixed_sig(sig, PROV_PARAMS, &lanes_of(elem)?, "storage read")
        }
        Imp::Write(Spec::Fixed(elem)) => {
            let mut params = lanes_of(elem)?;
            // Write params are provenance then lanes. `fixed_sig` checks the
            // front of `param_kind`, so lanes are appended after three i32s.
            let mut kinds = vec![Kind::I32, Kind::I32, Kind::I32];
            kinds.append(&mut params);
            fixed_sig_params(sig, &kinds, "storage write")
        }
        Imp::Read(Spec::Struct) => struct_read_sig(sig, structs, struct_id),
        Imp::Write(Spec::Struct) => struct_write_sig(sig, structs, struct_id),
        Imp::Uniform(uni) => uniform_sig(sig, uni, structs, struct_id),
    }
}

const FIELD_PARAMS: u32 = 4;
const ARRAY_PARAMS: u32 = 4;
const INDEX_STRUCT_PARAMS: u32 = 5;

fn uniform_sig(
    sig: &Sig,
    uni: Uni,
    structs: &[StructDef],
    struct_id: Option<u16>,
) -> Result<(), TranslateError> {
    match uni {
        Uni::Field(kind) => field_sig(sig, kind),
        Uni::Array(elem) => array_sig(sig, elem),
        Uni::Struct => struct_at(sig, structs, struct_id, PROV_PARAMS, "uniform_read_struct"),
        Uni::IndexStruct => struct_at(
            sig,
            structs,
            struct_id,
            INDEX_STRUCT_PARAMS,
            "uniform_index_struct",
        ),
    }
}

fn field_sig(sig: &Sig, kind: FieldKind) -> Result<(), TranslateError> {
    let mut lanes = Vec::new();
    push_lanes(&mut lanes, kind)?;
    fixed_sig(sig, FIELD_PARAMS, &lanes, "uniform field")
}

fn array_sig(sig: &Sig, elem: Elem) -> Result<(), TranslateError> {
    fixed_sig(sig, ARRAY_PARAMS, &lanes_of(elem)?, "uniform array")
}

pub(super) fn lanes_of(elem: Elem) -> Result<Vec<Kind>, TranslateError> {
    let kind = match elem {
        Elem::F32 => return Ok(vec![Kind::F32]),
        Elem::U32 | Elem::I32 => return Ok(vec![Kind::I32]),
        Elem::Vec2 => FieldKind::Vec2,
        Elem::Vec3 => FieldKind::Vec3,
        Elem::Vec4 => FieldKind::Vec4,
        Elem::Struct(_) => return Err(type_err("struct lanes need a type id")),
    };
    let mut lanes = Vec::new();
    push_lanes(&mut lanes, kind)?;
    Ok(lanes)
}

pub(super) fn push_lanes(lanes: &mut Vec<Kind>, kind: FieldKind) -> Result<(), TranslateError> {
    let float = field_is_float(kind);
    let n = field_lanes(kind);
    let mut index = 0u32;
    while index < n {
        lanes.push(if float { Kind::F32 } else { Kind::I32 });
        index += 1;
    }
    if lanes.len() + STRUCT_HEADER as usize > crate::ir::MAX_PARAMS {
        return Err(type_err("unsupported struct width"));
    }
    Ok(())
}

pub(super) fn count_only(
    sig: &Sig,
    params: u32,
    results: u32,
    what: &str,
) -> Result<(), TranslateError> {
    if sig.params == params && sig.results == results {
        Ok(())
    } else {
        Err(mismatch(what))
    }
}

fn mismatch(what: &str) -> TranslateError {
    type_err(&format!("{what} signature does not match the element"))
}

pub(super) fn fixed_sig(
    sig: &Sig,
    params: u32,
    results: &[Kind],
    what: &str,
) -> Result<(), TranslateError> {
    if sig.params != params || sig.results as usize != results.len() {
        return Err(mismatch(what));
    }
    let mut index = 0usize;
    while index < params as usize {
        if sig.param_kind[index] != Kind::I32 {
            return Err(mismatch(what));
        }
        index += 1;
    }
    kinds_eq(&sig.result_kind, sig.results, results, what)
}

pub(super) fn fixed_sig_params(
    sig: &Sig,
    params: &[Kind],
    what: &str,
) -> Result<(), TranslateError> {
    if sig.results != 0 || sig.params as usize != params.len() {
        return Err(mismatch(what));
    }
    kinds_eq(&sig.param_kind, sig.params, params, what)
}

pub(super) fn kinds_eq(
    got: &[Kind],
    n: u32,
    expect: &[Kind],
    what: &str,
) -> Result<(), TranslateError> {
    let mut index = 0usize;
    while index < expect.len() {
        if index >= n as usize || got[index] != expect[index] {
            return Err(mismatch(what));
        }
        index += 1;
    }
    Ok(())
}

pub(super) fn struct_read_sig(
    sig: &Sig,
    structs: &[StructDef],
    id: Option<u16>,
) -> Result<(), TranslateError> {
    struct_at(sig, structs, id, STRUCT_HEADER, "storage_read_struct")
}

fn struct_at(
    sig: &Sig,
    structs: &[StructDef],
    id: Option<u16>,
    params: u32,
    what: &str,
) -> Result<(), TranslateError> {
    let id = id.ok_or_else(|| type_err("struct type id is not a constant"))?;
    let def = structs
        .get(id as usize)
        .ok_or_else(|| type_err("struct type id is out of range"))?;
    let lanes = struct_lanes(&def.fields)?;
    fixed_sig(sig, params, &lanes, what)
}

pub(super) fn struct_write_sig(
    sig: &Sig,
    structs: &[StructDef],
    id: Option<u16>,
) -> Result<(), TranslateError> {
    let id = id.ok_or_else(|| type_err("struct type id is not a constant"))?;
    let def = structs
        .get(id as usize)
        .ok_or_else(|| type_err("struct type id is out of range"))?;
    let mut kinds = vec![Kind::I32, Kind::I32, Kind::I32, Kind::I32];
    let mut lanes = struct_lanes(&def.fields)?;
    kinds.append(&mut lanes);
    fixed_sig_params(sig, &kinds, "storage_write_struct")
}
