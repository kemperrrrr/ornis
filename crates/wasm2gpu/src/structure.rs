//! Flat wasm operators to a structured [`Stmt`](crate::ir::Stmt) tree.

use crate::error::TranslateError;
use crate::ir::BlockTy;
use crate::ir::Op;
use crate::ir::Stmt;

/// One operator before labels are matched.
#[derive(Clone)]
pub(super) enum Flat {
    /// Stack operator.
    Op(Op),
    /// `block`.
    Block(BlockTy),
    /// `loop`.
    Loop(BlockTy),
    /// `if`.
    If(BlockTy),
    /// `else`.
    Else,
    /// `end`.
    End,
    /// `br`.
    Br(u32),
    /// `br_if`.
    BrIf(u32),
    /// `return`.
    Return,
    /// `br_table`.
    BrTable {
        /// Arm depths.
        targets: Vec<u32>,
        /// Default depth.
        default: u32,
    },
}

enum FrameKind {
    Func,
    Block,
    Loop,
    If,
}

struct Frame {
    kind: FrameKind,
    ty: BlockTy,
    body: Vec<Stmt>,
    /// `then` body, present only while an `else` is open.
    then_body: Option<Vec<Stmt>>,
}

/// Match `block` / `loop` / `if` / `end` into a tree.
///
/// The function frame is closed by the body's final `end`.
pub(super) fn build(ops: Vec<Flat>) -> Result<Vec<Stmt>, TranslateError> {
    let mut stack = vec![func_frame()];
    for op in ops {
        step(&mut stack, op)?;
    }
    finish(stack)
}

fn step(stack: &mut Vec<Frame>, op: Flat) -> Result<(), TranslateError> {
    match op {
        Flat::Op(op) => push_stmt(stack, Stmt::Op(op)),
        Flat::Br(depth) => push_stmt(stack, Stmt::Br(depth)),
        Flat::BrIf(depth) => push_stmt(stack, Stmt::BrIf(depth)),
        Flat::Return => push_stmt(stack, Stmt::Return),
        Flat::BrTable { targets, default } => push_stmt(stack, Stmt::BrTable { targets, default }),
        Flat::Block(ty) => push_frame(stack, open(FrameKind::Block, ty)),
        Flat::Loop(ty) => push_frame(stack, open(FrameKind::Loop, ty)),
        Flat::If(ty) => push_frame(stack, open(FrameKind::If, ty)),
        Flat::Else => open_else(stack),
        Flat::End => close(stack),
    }
}

fn finish(stack: Vec<Frame>) -> Result<Vec<Stmt>, TranslateError> {
    let mut stack = stack;
    let Some(frame) = stack.pop() else {
        return Err(TranslateError::Unsupported);
    };
    if !stack.is_empty() || !is_func(&frame) {
        return Err(TranslateError::Unsupported);
    }
    Ok(frame.body)
}

fn close(stack: &mut Vec<Frame>) -> Result<(), TranslateError> {
    let Some(frame) = stack.pop() else {
        return Err(TranslateError::Unsupported);
    };
    if is_func(&frame) {
        stack.push(frame);
        return Ok(());
    }
    let stmt = into_stmt(frame)?;
    push_stmt(stack, stmt)
}

fn into_stmt(frame: Frame) -> Result<Stmt, TranslateError> {
    match frame.kind {
        FrameKind::Block => Ok(Stmt::Block {
            ty: frame.ty,
            body: frame.body,
        }),
        FrameKind::Loop => Ok(Stmt::Loop {
            ty: frame.ty,
            body: frame.body,
        }),
        FrameKind::If => if_stmt(frame),
        _ => Err(TranslateError::Unsupported),
    }
}

fn if_stmt(frame: Frame) -> Result<Stmt, TranslateError> {
    let had_else = frame.then_body.is_some();
    let (then_body, else_body) = match frame.then_body {
        Some(then_body) => (then_body, frame.body),
        None => (frame.body, Vec::new()),
    };
    if !had_else && !frame.ty.results.is_empty() {
        return Err(TranslateError::Unsupported);
    }
    Ok(Stmt::If {
        ty: frame.ty,
        then_body,
        else_body,
    })
}

fn open_else(stack: &mut [Frame]) -> Result<(), TranslateError> {
    let Some(frame) = stack.last_mut() else {
        return Err(TranslateError::Unsupported);
    };
    if !is_if(frame) || frame.then_body.is_some() {
        return Err(TranslateError::Unsupported);
    }
    frame.then_body = Some(std::mem::take(&mut frame.body));
    Ok(())
}

fn push_frame(stack: &mut Vec<Frame>, frame: Frame) -> Result<(), TranslateError> {
    stack.push(frame);
    Ok(())
}

fn push_stmt(stack: &mut [Frame], stmt: Stmt) -> Result<(), TranslateError> {
    let Some(frame) = stack.last_mut() else {
        return Err(TranslateError::Unsupported);
    };
    frame.body.push(stmt);
    Ok(())
}

fn func_frame() -> Frame {
    open(
        FrameKind::Func,
        BlockTy {
            params: Vec::new(),
            results: Vec::new(),
        },
    )
}

fn open(kind: FrameKind, ty: BlockTy) -> Frame {
    Frame {
        kind,
        ty,
        body: Vec::new(),
        then_body: None,
    }
}

fn is_func(frame: &Frame) -> bool {
    matches!(frame.kind, FrameKind::Func)
}

fn is_if(frame: &Frame) -> bool {
    matches!(frame.kind, FrameKind::If)
}
