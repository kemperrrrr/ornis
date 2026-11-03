//! Import name classification.

use super::{Elem, Imp, Spec, symbols};

/// Classify a toy import. `None` means the name is not a gpu opcode we lower.
pub(super) fn classify(name: &str) -> Option<Imp> {
    if name == symbols::STORAGE_BASE {
        return Some(Imp::Base);
    }
    if name == symbols::RAW_READ {
        return Some(Imp::ReadUniform);
    }
    if name == symbols::TEXTURE_SAMPLE_2D {
        return Some(Imp::Sample);
    }
    if name == symbols::TEXTURE_LOAD_2D {
        return Some(Imp::TextureLoad);
    }
    if name == symbols::MATH_DOT3 {
        return Some(Imp::MathDot3);
    }
    if name == symbols::MATH_CROSS3 {
        return Some(Imp::MathCross3);
    }
    if name == symbols::MATH_LENGTH3 {
        return Some(Imp::MathLength3);
    }
    if name == symbols::MATH_NORMALIZE3 {
        return Some(Imp::MathNormalize3);
    }
    if name == symbols::STORAGE_ATOMIC_ADD_U32 {
        return Some(Imp::StorageAtomicAddU32);
    }
    if name == symbols::WORKGROUP_LOAD_U32 {
        return Some(Imp::WorkgroupLoadU32);
    }
    if name == symbols::WORKGROUP_STORE_U32 {
        return Some(Imp::WorkgroupStoreU32);
    }
    if name == symbols::WORKGROUP_ATOMIC_ADD_U32 {
        return Some(Imp::WorkgroupAtomicAddU32);
    }
    if name == symbols::WORKGROUP_BARRIER {
        return Some(Imp::WorkgroupBarrier);
    }
    if name == symbols::STORAGE_BARRIER {
        return Some(Imp::StorageBarrier);
    }
    if name == symbols::TEXTURE_WIDTH_2D {
        return Some(Imp::TextureWidth);
    }
    if name == symbols::TEXTURE_HEIGHT_2D {
        return Some(Imp::TextureHeight);
    }
    if let Some(uni) = uni::imp(name) {
        return Some(Imp::Uniform(uni));
    }
    storage_imp(name)
}
