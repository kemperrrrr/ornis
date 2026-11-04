//! Type-stack handlers for vector math, memory synchronization, and texture reads.

use super::mem::pop_prov;
use super::value::pop_f32;
use super::{Abs, Num, St, pop_num, push};
use crate::error::TranslateError;
use crate::ir::Import;
use crate::ty;
use crate::ty::Imp;

pub(super) fn math_dot3(st: &mut St, import: &Import) -> Result<(), TranslateError> {
    ty::sig_ok(&import.sig, Imp::MathDot3, &[], None)?;
    for _ in 0..(crate::cx::VEC3_LANES * 2) {
        pop_f32(st)?;
    }
    push(st, Abs::F32)
}

pub(super) fn math_vec3(st: &mut St, import: &Import, imp: Imp) -> Result<(), TranslateError> {
    ty::sig_ok(&import.sig, imp, &[], None)?;
    let inputs = if imp == Imp::MathCross3 {
        crate::cx::VEC3_LANES * 2
    } else {
        crate::cx::VEC3_LANES
    };
    for _ in 0..inputs {
        pop_f32(st)?;
    }
    for _ in 0..crate::cx::VEC3_LANES {
        push(st, Abs::F32)?;
    }
    Ok(())
}

pub(super) fn math_length3(st: &mut St, import: &Import) -> Result<(), TranslateError> {
    ty::sig_ok(&import.sig, Imp::MathLength3, &[], None)?;
    for _ in 0..crate::cx::VEC3_LANES {
        pop_f32(st)?;
    }
    push(st, Abs::F32)
}

pub(super) fn storage_atomic_add(st: &mut St, import: &Import) -> Result<(), TranslateError> {
    ty::sig_ok(&import.sig, Imp::StorageAtomicAddU32, &[], None)?;
    pop_num(st)?; // value
    pop_num(st)?; // word index
    pop_prov(st)?; // binding
    pop_prov(st)?; // group
    push(st, Abs::I32(Num::Dyn { joined: false }))
}

pub(super) fn workgroup_load(st: &mut St, import: &Import) -> Result<(), TranslateError> {
    ty::sig_ok(&import.sig, Imp::WorkgroupLoadU32, &[], None)?;
    pop_num(st)?;
    push(st, Abs::I32(Num::Dyn { joined: false }))
}

pub(super) fn workgroup_store(st: &mut St, import: &Import) -> Result<(), TranslateError> {
    ty::sig_ok(&import.sig, Imp::WorkgroupStoreU32, &[], None)?;
    pop_num(st)?;
    pop_num(st)?;
    Ok(())
}

pub(super) fn workgroup_atomic_add(st: &mut St, import: &Import) -> Result<(), TranslateError> {
    ty::sig_ok(&import.sig, Imp::WorkgroupAtomicAddU32, &[], None)?;
    pop_num(st)?;
    pop_num(st)?;
    push(st, Abs::I32(Num::Dyn { joined: false }))
}

pub(super) fn barrier(_st: &mut St, import: &Import, imp: Imp) -> Result<(), TranslateError> {
    ty::sig_ok(&import.sig, imp, &[], None)
}

pub(super) fn texture_load(st: &mut St, import: &Import) -> Result<(), TranslateError> {
    ty::sig_ok(&import.sig, Imp::TextureLoad, &[], None)?;
    pop_num(st)?;
    pop_num(st)?;
    pop_prov(st)?;
    pop_prov(st)?;
    push(st, Abs::Vec)
}
