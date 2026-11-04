//! Value tags for the invocation id.
//!
//! [`Tag::Inv`] is the compute invocation id and images of it: injective
//! `i32` updates (`add`/`sub`/`xor` with a uniform operand, equality with a
//! uniform operand, `eqz`), `f32` arithmetic and unsigned `i32`-to-`f32`
//! conversion of such a value, and `select` whose condition is [`Tag::Inv`]
//! even when both arms are the same. Anything else is [`Tag::Unknown`]:
//! ordered comparisons (`id < n`), multiplies, and memory loads do not prove
//! divergence. Two [`Tag::Inv`] values joined together stay [`Tag::Inv`]:
//! both differ across invocations. A uniform value joined with [`Tag::Inv`]
//! is [`Tag::Unknown`]. A loop back-edge uses [`widen_tag`], so a local that
//! becomes the id on a later iteration stays [`Tag::Inv`].

use super::St;
use crate::ir::Op;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Tag {
    /// Proved to be the same value for every invocation.
    Uniform,
    /// Proved to differ across invocations.
    Inv,
    /// May depend on the invocation, but this pass did not prove it.
    Unknown,
}

pub(super) fn join_tag(left: Tag, right: Tag) -> Tag {
    match (left, right) {
        (Tag::Uniform, Tag::Uniform) => Tag::Uniform,
        (Tag::Inv, Tag::Inv) => Tag::Inv,
        _ => Tag::Unknown,
    }
}

/// Loop-carried join. [`Tag::Inv`] survives a uniform entry value.
///
/// [`join_tag`] would turn that pair into [`Tag::Unknown`] and hide a branch
/// on the second iteration. Unknown still wins: a lost proof stays lost.
pub(super) fn widen_tag(entry: Tag, back: Tag) -> Tag {
    match (entry, back) {
        (Tag::Uniform, Tag::Uniform) => Tag::Uniform,
        (Tag::Unknown, _) | (_, Tag::Unknown) => Tag::Unknown,
        _ => Tag::Inv,
    }
}

pub(super) fn widen_vec(entry: &[Tag], back: &[Tag]) -> Vec<Tag> {
    let mut out = Vec::new();
    let mut index = 0usize;
    let n = if entry.len() > back.len() {
        entry.len()
    } else {
        back.len()
    };
    while index < n {
        let left = entry.get(index).copied().unwrap_or(Tag::Uniform);
        let right = back.get(index).copied().unwrap_or(Tag::Uniform);
        out.push(widen_tag(left, right));
        index += 1;
    }
    out
}

pub(super) fn join_vec(left: &[Tag], right: &[Tag]) -> Vec<Tag> {
    let mut out = Vec::new();
    let mut index = 0usize;
    let n = if left.len() > right.len() {
        left.len()
    } else {
        right.len()
    };
    while index < n {
        let l = left.get(index).copied().unwrap_or(Tag::Unknown);
        let r = right.get(index).copied().unwrap_or(Tag::Unknown);
        out.push(join_tag(l, r));
        index += 1;
    }
    out
}

pub(super) fn apply(st: &mut St, op: Op) {
    match op {
        Op::I32(_) | Op::F32(_) => st.stack.push(Tag::Uniform),
        Op::Local(index) | Op::LocalI32(index) => push_local(st, index),
        Op::LocalSet(index) => set_local(st, index, false),
        Op::LocalTee(index) => set_local(st, index, true),
        Op::Drop => {
            st.stack.pop();
        }
        Op::Call(_) => {}
        Op::Mul | Op::Add | Op::Sub | Op::Div | Op::F32Min | Op::F32Max => f32_bin(st),
        Op::F32Abs | Op::F32Neg | Op::F32Sqrt | Op::F32ConvertI32U => f32_un(st),
        Op::I32Add | Op::I32Sub | Op::I32Xor => i32_bin(st, true),
        Op::I32Mul | Op::I32Shl | Op::I32ShrU | Op::I32ShrS | Op::I32And | Op::I32Or => {
            i32_bin(st, false)
        }
        Op::I32Eqz => eqz(st),
        Op::I32Eq | Op::I32Ne => i32_bin(st, true),
        Op::I32LtU | Op::I32GtU | Op::I32GeU | Op::I32LeU => i32_bin(st, false),
        Op::F32Lt | Op::F32Le | Op::F32Gt | Op::F32Ge | Op::F32Eq | Op::F32Ne => f32_cmp(st),
        Op::LoadI32(_) | Op::LoadF32(_) => {
            st.stack.pop();
            st.stack.push(Tag::Unknown);
        }
        Op::StoreI32(_) | Op::StoreF32(_) => {
            st.stack.pop();
            st.stack.pop();
        }
        Op::Select => select(st),
    }
}

fn push_local(st: &mut St, index: u32) {
    let tag = st
        .locals
        .get(index as usize)
        .copied()
        .unwrap_or(Tag::Unknown);
    st.stack.push(tag);
}

fn set_local(st: &mut St, index: u32, tee: bool) {
    let tag = st.stack.pop().unwrap_or(Tag::Unknown);
    if let Some(slot) = st.locals.get_mut(index as usize) {
        *slot = tag;
    }
    if tee {
        st.stack.push(tag);
    }
}

fn f32_bin(st: &mut St) {
    let right = st.stack.pop().unwrap_or(Tag::Unknown);
    let left = st.stack.pop().unwrap_or(Tag::Unknown);
    st.stack.push(float_tag(left, right));
}

fn f32_un(st: &mut St) {
    let arg = st.stack.pop().unwrap_or(Tag::Unknown);
    st.stack.push(float_tag(arg, Tag::Uniform));
}

/// An `f32` image of the invocation id still differs across invocations.
fn float_tag(left: Tag, right: Tag) -> Tag {
    match (left, right) {
        (Tag::Inv, _) | (_, Tag::Inv) => Tag::Inv,
        (Tag::Unknown, _) | (_, Tag::Unknown) => Tag::Unknown,
        _ => Tag::Uniform,
    }
}

fn f32_cmp(st: &mut St) {
    st.stack.pop();
    st.stack.pop();
    st.stack.push(Tag::Unknown);
}

/// `preserve` keeps [`Tag::Inv`] through a uniform operand.
///
/// Addition, subtraction, xor, and equality do. Multiplication, shifts,
/// bitwise and/or, and ordered comparisons do not: `id * 0` and `id < 0`
/// are uniform, and this pass will not claim otherwise.
fn i32_bin(st: &mut St, preserve: bool) {
    let right = st.stack.pop().unwrap_or(Tag::Unknown);
    let left = st.stack.pop().unwrap_or(Tag::Unknown);
    st.stack.push(bin_tag(left, right, preserve));
}

fn bin_tag(left: Tag, right: Tag, preserve: bool) -> Tag {
    match (left, right) {
        (Tag::Uniform, Tag::Uniform) => Tag::Uniform,
        (Tag::Inv, Tag::Uniform) | (Tag::Uniform, Tag::Inv) if preserve => Tag::Inv,
        _ => Tag::Unknown,
    }
}

fn eqz(st: &mut St) {
    // `id == 0` still differs across invocations. A uniform zero stays uniform.
    let tag = st.stack.pop().unwrap_or(Tag::Unknown);
    st.stack.push(tag);
}

fn select(st: &mut St) {
    let cond = st.stack.pop().unwrap_or(Tag::Unknown);
    let reject = st.stack.pop().unwrap_or(Tag::Unknown);
    let accept = st.stack.pop().unwrap_or(Tag::Unknown);
    let tag = if cond == Tag::Inv {
        Tag::Inv
    } else if cond == Tag::Unknown {
        Tag::Unknown
    } else {
        join_tag(accept, reject)
    };
    st.stack.push(tag);
}

#[cfg(test)]
mod tests {
    use super::super::St;
    use super::Tag;
    use super::apply;
    use crate::ir::Op;

    fn pushed(op: Op, before: &[Tag]) -> Tag {
        let mut st = St::fresh(Vec::new());
        let mut index = 0usize;
        while index < before.len() {
            st.stack.push(before[index]);
            index += 1;
        }
        apply(&mut st, op);
        st.stack.pop().unwrap_or(Tag::Unknown)
    }

    #[test]
    fn select_on_the_invocation_id_stays_invocation_dependent() {
        let tag = pushed(Op::Select, &[Tag::Uniform, Tag::Uniform, Tag::Inv]);
        assert_eq!(tag, Tag::Inv);
    }

    #[test]
    fn float_ops_keep_an_invocation_dependent_operand() {
        assert_eq!(pushed(Op::F32ConvertI32U, &[Tag::Inv]), Tag::Inv);
        assert_eq!(pushed(Op::F32Neg, &[Tag::Inv]), Tag::Inv);
        assert_eq!(pushed(Op::Add, &[Tag::Inv, Tag::Uniform]), Tag::Inv);
        assert_eq!(pushed(Op::Mul, &[Tag::Uniform, Tag::Uniform]), Tag::Uniform);
        assert_eq!(pushed(Op::F32Abs, &[Tag::Unknown]), Tag::Unknown);
    }
}
