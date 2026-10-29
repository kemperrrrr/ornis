//! Abstract interpretation of `br_table`.
//!
//! A constant selector follows one arm. A dynamic selector joins every arm
//! and does not fall through the switch.

use super::Ai;
use super::Num;
use super::St;
use super::branch;
use super::pop_num;
use crate::error::TranslateError;

pub(super) fn br_table(
    ai: &mut Ai<'_>,
    st: &mut St,
    targets: &[u32],
    default: u32,
) -> Result<(), TranslateError> {
    let index = pop_num(st)?;
    match index {
        Num::Const(value) => {
            let depth = targets.get(value as usize).copied().unwrap_or(default);
            branch(ai, st, depth, true)
        }
        Num::Dyn { .. } => join_arms(ai, st, targets, default),
    }
}

fn join_arms(
    ai: &mut Ai<'_>,
    st: &mut St,
    targets: &[u32],
    default: u32,
) -> Result<(), TranslateError> {
    let mut seen = Vec::new();
    for depth in targets.iter().copied().chain(std::iter::once(default)) {
        if seen.contains(&depth) {
            continue;
        }
        seen.push(depth);
        let mut jumped = st.clone();
        branch(ai, &mut jumped, depth, true)?;
    }
    st.stop = true;
    Ok(())
}
