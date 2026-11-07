//! Post-lower Naga pass: coalesce 4 scalar storage accesses into vec4.
//!
//! The pass is deliberately narrow and conservative. It only promotes an
//! `array<u32>` storage global to `array<vec4<u32>>` and rewrites its
//! accesses when *every* access in the whole module belongs to one dense
//! run of `Load src[base+k] -> Store dst[dst+k]` pairs of length `N`,
//! with `N % 4 == 0`, `N >= 16`, and all pairs in the same block in
//! order with only `Emit` statements between them. Anything else leaves
//! the module untouched, so other kernels (skinning, auto-exposure, the
//! flag kernel) keep their scalar lowering. The pass exists for the
//! 52-word `PerObjectGpu` record copy in `cull_compact` (13 vec4s).
//!
//! Grouping is by run order, not by reconstructed offsets: each group of
//! four consecutive pairs is assumed to be `k = 4g .. 4g+3` with a base
//! that is a multiple of four, so the vector index is `word_index >> 2`.
//! For the `cull_compact` copy the bases are `id * 52` and `slot * 52`,
//! and the Rust source writes `k = 0 .. 51` in order, so the assumption
//! holds for the only kernel that has both candidate bindings.

use naga::GlobalVariable;

/// Minimum record words considered for promotion (must be a multiple of 4).
const MIN_WORDS: usize = 16;

/// Vector width in words.
const LANES: usize = 4;

/// Stride of one `vec4<u32>` element in bytes.
const VEC_STRIDE: u32 = 16;

/// Try to promote scalar-copy globals to vec4. No-op when not applicable.
pub(crate) fn coalesce(module: &mut naga::Module) {
    let candidates = find_copy_globals(module);
    if candidates.len() != 2 {
        return;
    }
    let Some((dst, src)) = copy_pair(module, &candidates) else {
        return;
    };
    // Verify first, rewrite second: verification touches only reads and
    // bails to no-op, so a kernel that does not fully match keeps scalar.
    let Some(run_len) = verify_module(module, src, dst) else {
        return;
    };
    if count_candidate_accesses(module, &candidates) != 2 * run_len {
        return;
    }
    promote_globals(module, &candidates);
    rewrite_module(module, src, dst);
}

/// Find the (dst, src) pair from candidate handles by binding.
fn copy_pair(
    module: &naga::Module,
    candidates: &[naga::Handle<GlobalVariable>],
) -> Option<(naga::Handle<GlobalVariable>, naga::Handle<GlobalVariable>)> {
    let mut dst = None;
    let mut src = None;
    for handle in candidates {
        let binding = module.global_variables[*handle].binding?;
        if binding.binding == 2 {
            dst = Some(*handle);
        } else if binding.binding == 4 {
            src = Some(*handle);
        }
    }
    Some((dst?, src?))
}

fn find_copy_globals(module: &naga::Module) -> Vec<naga::Handle<GlobalVariable>> {
    let mut out = Vec::new();
    for (handle, global) in module.global_variables.iter() {
        if !matches!(global.space, naga::AddressSpace::Storage { .. }) {
            continue;
        }
        let Some(binding) = global.binding else {
            continue;
        };
        // The copy candidate in this workspace is cull_compact's bindings
        // 2 (dst) and 4 (src words). Other u32 storages (flags, counts) are
        // scalar by design and must not be promoted.
        if binding.group != 0 || (binding.binding != 2 && binding.binding != 4) {
            continue;
        }
        let Some(base) = array_base(module, global.ty) else {
            continue;
        };
        if !is_u32_scalar(module, base) {
            continue;
        }
        out.push(handle);
    }
    out
}

fn array_base(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
) -> Option<naga::Handle<naga::Type>> {
    match &module.types[ty].inner {
        naga::TypeInner::Array { base, .. } => Some(*base),
        _ => None,
    }
}

fn is_u32_scalar(module: &naga::Module, ty: naga::Handle<naga::Type>) -> bool {
    matches!(
        module.types[ty].inner,
        naga::TypeInner::Scalar(scalar) if scalar.kind == naga::ScalarKind::Uint
    )
}

/// One scalar copy pair in a block: `Load src[src_idx] -> Store dst[dst_idx]`.
#[derive(Clone, Copy)]
struct StoreInfo {
    stmt_idx: usize,
    load_h: naga::Handle<naga::Expression>,
    src_ptr_h: naga::Handle<naga::Expression>,
    dst_ptr_h: naga::Handle<naga::Expression>,
    src_idx: naga::Handle<naga::Expression>,
    dst_idx: naga::Handle<naga::Expression>,
}

#[allow(clippy::too_many_arguments)]
fn count_candidate_accesses(
    module: &naga::Module,
    candidates: &[naga::Handle<GlobalVariable>],
) -> usize {
    let mut total = 0usize;
    let mut funcs: Vec<&naga::Function> = Vec::new();
    for (_, f) in module.functions.iter() {
        funcs.push(f);
    }
    for e in &module.entry_points {
        funcs.push(&e.function);
    }
    for func in funcs {
        for (_, expr) in func.expressions.iter() {
            let naga::Expression::Access { base, .. } = expr else {
                continue;
            };
            let naga::Expression::GlobalVariable(var) = &func.expressions[*base] else {
                continue;
            };
            if candidates.contains(var) {
                total += 1;
            }
        }
    }
    total
}

/// Verify the whole module: exactly one block contains a dense run of
/// copy pairs of length that is a multiple of 4 and at least MIN_WORDS.
/// Returns the run length so the caller can compare against the total
/// access count and reject any extra access elsewhere.
fn verify_module(
    module: &naga::Module,
    src: naga::Handle<GlobalVariable>,
    dst: naga::Handle<GlobalVariable>,
) -> Option<usize> {
    let mut funcs: Vec<&naga::Function> = Vec::new();
    for (_, f) in module.functions.iter() {
        funcs.push(f);
    }
    for e in &module.entry_points {
        funcs.push(&e.function);
    }
    let mut found: Option<usize> = None;
    for func in funcs {
        let Some(run) = find_run_in_block(&func.body, func, src, dst) else {
            continue;
        };
        if found.is_some() {
            return None;
        }
        found = Some(run.len());
    }
    let n = found?;
    if n < MIN_WORDS || n % LANES != 0 {
        return None;
    }
    Some(n)
}

fn store_info(
    stmt: &naga::Statement,
    func: &naga::Function,
    src: naga::Handle<GlobalVariable>,
    dst: naga::Handle<GlobalVariable>,
) -> Option<StoreInfo> {
    let naga::Statement::Store { pointer, value } = stmt else {
        return None;
    };
    let dst_ptr_h = *pointer;
    let load_h = *value;
    let naga::Expression::Access {
        base: dst_base,
        index: dst_idx,
    } = &func.expressions[dst_ptr_h]
    else {
        return None;
    };
    let naga::Expression::GlobalVariable(var) = &func.expressions[*dst_base] else {
        return None;
    };
    if *var != dst {
        return None;
    }
    let naga::Expression::Load { pointer: src_ptr } = &func.expressions[load_h] else {
        return None;
    };
    let src_ptr_h = *src_ptr;
    let naga::Expression::Access {
        base: src_base,
        index: src_idx,
    } = &func.expressions[src_ptr_h]
    else {
        return None;
    };
    let naga::Expression::GlobalVariable(svar) = &func.expressions[*src_base] else {
        return None;
    };
    if *svar != src {
        return None;
    }
    Some(StoreInfo {
        stmt_idx: 0,
        load_h,
        src_ptr_h,
        dst_ptr_h,
        src_idx: *src_idx,
        dst_idx: *dst_idx,
    })
}

fn direct_run(
    block: &naga::Block,
    func: &naga::Function,
    src: naga::Handle<GlobalVariable>,
    dst: naga::Handle<GlobalVariable>,
) -> Option<Vec<StoreInfo>> {
    let mut best: Option<Vec<StoreInfo>> = None;
    let mut current: Vec<StoreInfo> = Vec::new();
    for (idx, stmt) in block.iter().enumerate() {
        if matches!(stmt, naga::Statement::Emit(_)) {
            continue;
        }
        if let Some(mut info) = store_info(stmt, func, src, dst) {
            info.stmt_idx = idx;
            current.push(info);
        } else {
            if current.len() >= MIN_WORDS
                && current.len().is_multiple_of(LANES)
                && best.as_ref().is_none_or(|b| current.len() > b.len())
            {
                best = Some(current.clone());
            }
            current.clear();
        }
    }
    if current.len() >= MIN_WORDS
        && current.len().is_multiple_of(LANES)
        && best.as_ref().is_none_or(|b| current.len() > b.len())
    {
        best = Some(current);
    }
    best
}

fn find_run_in_block(
    block: &naga::Block,
    func: &naga::Function,
    src: naga::Handle<GlobalVariable>,
    dst: naga::Handle<GlobalVariable>,
) -> Option<Vec<StoreInfo>> {
    if let Some(run) = direct_run(block, func, src, dst) {
        return Some(run);
    }
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Block(child) => {
                if let Some(run) = find_run_in_block(child, func, src, dst) {
                    return Some(run);
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                if let Some(run) = find_run_in_block(accept, func, src, dst) {
                    return Some(run);
                }
                if let Some(run) = find_run_in_block(reject, func, src, dst) {
                    return Some(run);
                }
            }
            naga::Statement::Loop { body, .. } => {
                if let Some(run) = find_run_in_block(body, func, src, dst) {
                    return Some(run);
                }
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    if let Some(run) = find_run_in_block(&case.body, func, src, dst) {
                        return Some(run);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Promote both candidate globals to `array<vec4<u32>>`.
fn promote_globals(module: &mut naga::Module, candidates: &[naga::Handle<GlobalVariable>]) {
    let vec_ty = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Vector {
                size: naga::VectorSize::Quad,
                scalar: naga::Scalar {
                    kind: naga::ScalarKind::Uint,
                    width: 4,
                },
            },
        },
        naga::Span::default(),
    );
    let arr_ty = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Array {
                base: vec_ty,
                size: naga::ArraySize::Dynamic,
                stride: VEC_STRIDE,
            },
        },
        naga::Span::default(),
    );
    for handle in candidates {
        module.global_variables[*handle].ty = arr_ty;
    }
}

/// One vector group's new statements: Emits then the vec Store.
struct GroupStmts {
    emits: Vec<naga::Statement>,
    store: naga::Statement,
}

fn rewrite_module(
    module: &mut naga::Module,
    src: naga::Handle<GlobalVariable>,
    dst: naga::Handle<GlobalVariable>,
) {
    for entry in &mut module.entry_points {
        let _ = rewrite_function(&mut entry.function, src, dst);
    }
    for (_, func) in module.functions.iter_mut() {
        let _ = rewrite_function(func, src, dst);
    }
}

fn rewrite_function(
    func: &mut naga::Function,
    src: naga::Handle<GlobalVariable>,
    dst: naga::Handle<GlobalVariable>,
) -> bool {
    // Phase 1: discover the run with an immutable borrow.
    let run = {
        let Some(run) = find_run_in_block(&func.body, func, src, dst) else {
            return false;
        };
        run
    };
    // Phase 2: append vector expressions group by group.
    let groups = build_groups(func, &run, src, dst);
    // Phase 3: rebuild the block that held the run.
    rebuild_block(&mut func.body, &run, &groups)
}

/// Append new vector expressions for every group, capturing one `Emit`
/// statement per non-pre-emitted expression as it is created.
fn build_groups(
    func: &mut naga::Function,
    run: &[StoreInfo],
    src: naga::Handle<GlobalVariable>,
    dst: naga::Handle<GlobalVariable>,
) -> Vec<Option<GroupStmts>> {
    let lit2 = func.expressions.append(
        naga::Expression::Literal(naga::Literal::U32(2)),
        naga::Span::default(),
    );
    let g_src = func
        .expressions
        .append(naga::Expression::GlobalVariable(src), naga::Span::default());
    let g_dst = func
        .expressions
        .append(naga::Expression::GlobalVariable(dst), naga::Span::default());
    let mut out: Vec<Option<GroupStmts>> = Vec::new();
    let mut group = 0usize;
    while group * LANES < run.len() {
        let base_info = run[group * LANES];
        let emits = append_group(func, base_info, lit2, g_src, g_dst);
        out.push(Some(emits));
        group += 1;
    }
    out
}

fn append_group(
    func: &mut naga::Function,
    info: StoreInfo,
    lit2: naga::Handle<naga::Expression>,
    g_src: naga::Handle<naga::Expression>,
    g_dst: naga::Handle<naga::Expression>,
) -> GroupStmts {
    let mut emits = Vec::new();
    let shift_src = append_emit(
        func,
        naga::Expression::Binary {
            op: naga::BinaryOperator::ShiftRight,
            left: info.src_idx,
            right: lit2,
        },
        &mut emits,
    );
    let shift_dst = append_emit(
        func,
        naga::Expression::Binary {
            op: naga::BinaryOperator::ShiftRight,
            left: info.dst_idx,
            right: lit2,
        },
        &mut emits,
    );
    let src_ptr = append_emit(
        func,
        naga::Expression::Access {
            base: g_src,
            index: shift_src,
        },
        &mut emits,
    );
    let load_vec = append_emit(
        func,
        naga::Expression::Load { pointer: src_ptr },
        &mut emits,
    );
    let dst_ptr = append_emit(
        func,
        naga::Expression::Access {
            base: g_dst,
            index: shift_dst,
        },
        &mut emits,
    );
    GroupStmts {
        emits,
        store: naga::Statement::Store {
            pointer: dst_ptr,
            value: load_vec,
        },
    }
}

fn append_emit(
    func: &mut naga::Function,
    expr: naga::Expression,
    emits: &mut Vec<naga::Statement>,
) -> naga::Handle<naga::Expression> {
    let handle = func.expressions.append(expr, naga::Span::default());
    let range = func.expressions.range_from(handle.index());
    emits.push(naga::Statement::Emit(range));
    handle
}

/// Rebuild the block subtree that contains `run`, fusing each group of
/// four scalar stores into one vector store. Returns `true` when a
/// block was rebuilt.
fn rebuild_block(
    block: &mut naga::Block,
    run: &[StoreInfo],
    groups: &[Option<GroupStmts>],
) -> bool {
    // Direct run in this block?
    let first_idx = run.first().map(|i| i.stmt_idx);
    let last_idx = run.last().map(|i| i.stmt_idx);
    let here = first_idx.is_some()
        && last_idx.is_some()
        && first_idx.unwrap() < block.len()
        && last_idx.unwrap() < block.len()
        && run
            .iter()
            .all(|i| matches!(&block[i.stmt_idx], naga::Statement::Store { .. }));
    // Confirm the block really holds this run by checking the store
    // statement at each recorded index is a Store (a stronger check is
    // unnecessary: `run` was discovered from this exact block layout,
    // and only one block in the function holds it).
    if here {
        rebuild_direct(block, run, groups);
        return true;
    }
    // Otherwise recurse into child blocks.
    for stmt in block.iter_mut() {
        match stmt {
            naga::Statement::Block(child) => {
                if rebuild_block(child, run, groups) {
                    return true;
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                if rebuild_block(accept, run, groups) {
                    return true;
                }
                if rebuild_block(reject, run, groups) {
                    return true;
                }
            }
            naga::Statement::Loop { body, .. } => {
                if rebuild_block(body, run, groups) {
                    return true;
                }
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    if rebuild_block(&mut case.body, run, groups) {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}

fn rebuild_direct(block: &mut naga::Block, run: &[StoreInfo], groups: &[Option<GroupStmts>]) {
    let original: Vec<naga::Statement> = block.iter().cloned().collect();
    // Handles whose old loads would dereference the promoted globals
    // with scalar indices must not be emitted once the type changes.
    let mut dead: Vec<naga::Handle<naga::Expression>> = Vec::new();
    for info in run {
        dead.push(info.load_h);
        dead.push(info.src_ptr_h);
        dead.push(info.dst_ptr_h);
    }
    // Which original stores start a vector group?
    let mut first_of_group: Vec<usize> = Vec::new();
    let mut drop_stmt: Vec<usize> = Vec::new();
    let mut group_of_stmt: Vec<(usize, usize)> = Vec::new();
    for (k, info) in run.iter().enumerate() {
        let g = k / LANES;
        if k % LANES == 0 {
            first_of_group.push(info.stmt_idx);
            group_of_stmt.push((info.stmt_idx, g));
        } else {
            drop_stmt.push(info.stmt_idx);
        }
    }
    let mut rebuilt: Vec<naga::Statement> = Vec::new();
    for (idx, stmt) in original.into_iter().enumerate() {
        // Drop Emits that only introduced a now-dead promoted load.
        if let naga::Statement::Emit(range) = &stmt {
            let mut emit_dead = true;
            let mut has_any = false;
            for h in range.clone() {
                has_any = true;
                if !dead.contains(&h) {
                    emit_dead = false;
                    break;
                }
            }
            if has_any && emit_dead {
                continue;
            }
        }
        // Drop non-first scalar stores of each group.
        if drop_stmt.contains(&idx) {
            continue;
        }
        // First store of a group becomes the vector store, preceded
        // by the freshly created Emits.
        if let Some((_, g)) = group_of_stmt.iter().find(|(s, _)| *s == idx)
            && let Some(Some(group)) = groups.get(*g)
        {
            rebuilt.extend(group.emits.iter().cloned());
            rebuilt.push(group.store.clone());
            continue;
        }
        rebuilt.push(stmt);
    }
    let _ = first_of_group;
    *block = naga::Block::from_vec(rebuilt);
}
