//! Control-flow graph of the structured wasm body.
//!
//! `block` and `if` join at a merge node. `loop` headers take a back edge from
//! `br` / `br_if` to that loop. Provenance is checked separately by abstract
//! interpretation; the graph is what makes the join explicit.

use crate::ai;
use crate::error::TranslateError;
use crate::ir::Func;
use crate::ir::Import;
use crate::ir::Stmt;
use crate::ty::StructDef;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum NodeKind {
    Code,
    Merge,
    Header,
}

pub(super) struct Node {
    pub(super) kind: NodeKind,
    pub(super) preds: Vec<u32>,
}

pub(super) struct Cfg {
    pub(super) nodes: Vec<Node>,
}

#[derive(Clone, Copy)]
enum LabelKind {
    Block,
    Loop,
    If,
}

struct Label {
    kind: LabelKind,
    header: u32,
    merge: u32,
}

struct Bld {
    nodes: Vec<Node>,
}

/// Reject ambiguous provenance and dynamic bindings before emission.
pub(super) fn check(
    func: &Func,
    imports: &[Import],
    structs: &[StructDef],
) -> Result<(), TranslateError> {
    let cfg = build(&func.body);
    let _ = edge_count(&cfg);
    ai::check(func, imports, structs)
}

fn edge_count(cfg: &Cfg) -> usize {
    let mut total = 0usize;
    let mut index = 0usize;
    while index < cfg.nodes.len() {
        total += cfg.nodes[index].preds.len();
        if cfg.nodes[index].kind == NodeKind::Header {
            total += 1;
        }
        index += 1;
    }
    total
}

fn build(body: &[Stmt]) -> Cfg {
    let mut bld = Bld { nodes: Vec::new() };
    let entry = bld.add(NodeKind::Code);
    let _ = walk(&mut bld, &mut Vec::new(), body, vec![entry]);
    Cfg { nodes: bld.nodes }
}

// qual:recursive
fn walk(
    bld: &mut Bld,
    labels: &mut Vec<Label>,
    stmts: &[Stmt],
    mut incoming: Vec<u32>,
) -> Vec<u32> {
    for stmt in stmts {
        if incoming.is_empty() {
            break;
        }
        incoming = one(bld, labels, stmt, incoming);
    }
    incoming
}

fn one(bld: &mut Bld, labels: &mut Vec<Label>, stmt: &Stmt, incoming: Vec<u32>) -> Vec<u32> {
    match stmt {
        Stmt::Op(_) => code(bld, incoming),
        Stmt::Br(depth) => {
            let node = link_in(bld, incoming);
            jump(bld, labels, node, *depth);
            Vec::new()
        }
        Stmt::BrIf(depth) => {
            let node = link_in(bld, incoming);
            jump(bld, labels, node, *depth);
            vec![node]
        }
        Stmt::Block { body, .. } => nest(bld, labels, body, incoming, LabelKind::Block),
        Stmt::Loop { body, .. } => nest(bld, labels, body, incoming, LabelKind::Loop),
        Stmt::If {
            then_body,
            else_body,
            ..
        } => if_join(bld, labels, then_body, else_body, incoming),
    }
}

fn nest(
    bld: &mut Bld,
    labels: &mut Vec<Label>,
    body: &[Stmt],
    incoming: Vec<u32>,
    kind: LabelKind,
) -> Vec<u32> {
    let header = bld.add(NodeKind::Header);
    let merge = bld.add(NodeKind::Merge);
    for pred in &incoming {
        bld.edge(*pred, header);
    }
    let (target_header, fall_to) = match kind {
        LabelKind::Loop => (header, merge),
        LabelKind::Block | LabelKind::If => (merge, merge),
    };
    labels.push(Label {
        kind,
        header: target_header,
        merge,
    });
    let exits = walk(bld, labels, body, vec![header]);
    labels.pop();
    for exit in exits {
        bld.edge(exit, fall_to);
    }
    if matches!(kind, LabelKind::Loop) {
        // A loop's fallthrough leaves the loop. The back edge is `br`.
        vec![merge]
    } else {
        vec![merge]
    }
}

fn if_join(
    bld: &mut Bld,
    labels: &mut Vec<Label>,
    then_body: &[Stmt],
    else_body: &[Stmt],
    incoming: Vec<u32>,
) -> Vec<u32> {
    let entry = link_in(bld, incoming);
    let merge = bld.add(NodeKind::Merge);
    labels.push(Label {
        kind: LabelKind::If,
        header: merge,
        merge,
    });
    let then_exits = walk(bld, labels, then_body, vec![entry]);
    let else_exits = walk(bld, labels, else_body, vec![entry]);
    labels.pop();
    for exit in then_exits {
        bld.edge(exit, merge);
    }
    for exit in else_exits {
        bld.edge(exit, merge);
    }
    vec![merge]
}

fn code(bld: &mut Bld, incoming: Vec<u32>) -> Vec<u32> {
    vec![link_in(bld, incoming)]
}

fn link_in(bld: &mut Bld, incoming: Vec<u32>) -> u32 {
    let node = bld.add(NodeKind::Code);
    for pred in incoming {
        bld.edge(pred, node);
    }
    node
}

fn jump(bld: &mut Bld, labels: &[Label], from: u32, depth: u32) {
    let Some(label) = labels.iter().rev().nth(depth as usize) else {
        return;
    };
    let target = if matches!(label.kind, LabelKind::Loop) {
        label.header
    } else {
        label.merge
    };
    bld.edge(from, target);
}

impl Bld {
    fn add(&mut self, kind: NodeKind) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(Node {
            kind,
            preds: Vec::new(),
        });
        id
    }

    fn edge(&mut self, from: u32, to: u32) {
        if let Some(node) = self.nodes.get_mut(to as usize) {
            node.preds.push(from);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::BlockTy;
    use crate::ir::Kind;
    use crate::ir::MAX_PARAMS;
    use crate::ir::Op;
    use crate::ir::Sig;

    fn empty() -> BlockTy {
        BlockTy {
            params: Vec::new(),
            results: Vec::new(),
        }
    }

    fn merges(cfg: &Cfg) -> Vec<&Node> {
        cfg.nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Merge)
            .collect()
    }

    #[test]
    fn if_merge_has_two_predecessors() {
        let body = vec![Stmt::If {
            ty: empty(),
            then_body: vec![Stmt::Op(Op::I32(0))],
            else_body: vec![Stmt::Op(Op::I32(1))],
        }];
        let cfg = build(&body);
        let found = merges(&cfg);
        assert!(found.len() == 1);
        assert!(found[0].preds.len() == 2);
    }

    #[test]
    fn loop_header_has_a_back_edge() {
        let body = vec![Stmt::Loop {
            ty: empty(),
            body: vec![Stmt::Br(0)],
        }];
        let cfg = build(&body);
        let mut headers = 0usize;
        let mut index = 0usize;
        while index < cfg.nodes.len() {
            if cfg.nodes[index].kind == NodeKind::Header && cfg.nodes[index].preds.len() > 1 {
                headers += 1;
            }
            index += 1;
        }
        assert!(headers == 1);
    }

    fn i32_sig(params: u32, results: u32) -> Sig {
        let mut param_kind = [Kind::F32; MAX_PARAMS];
        let mut index = 0usize;
        while index < params as usize {
            param_kind[index] = Kind::I32;
            index += 1;
        }
        Sig {
            params,
            results,
            param_kind,
            result_kind: [Kind::F32; MAX_PARAMS],
        }
    }

    fn func(body: Vec<Stmt>) -> Func {
        Func {
            export: "fs".to_string(),
            sig: i32_sig(1, crate::cx::LANES),
            locals: vec![Kind::I32],
            body,
        }
    }

    const PROV_PARAMS: u32 = 3;

    fn storage() -> Import {
        Import {
            name: ornis_gpu_abi::symbols::STORAGE_READ.to_string(),
            sig: i32_sig(PROV_PARAMS, crate::cx::LANES),
        }
    }

    #[test]
    fn joined_bindings_are_ambiguous() {
        let body = vec![
            Stmt::Op(Op::I32(0)),
            Stmt::Op(Op::LocalI32(0)),
            Stmt::If {
                ty: BlockTy {
                    params: Vec::new(),
                    results: vec![Kind::I32],
                },
                then_body: vec![Stmt::Op(Op::I32(0))],
                else_body: vec![Stmt::Op(Op::I32(1))],
            },
            Stmt::Op(Op::I32(0)),
            Stmt::Op(Op::Call(0)),
        ];
        let err = check(&func(body), &[storage()], &[]).expect_err("phi");
        assert!(matches!(err, TranslateError::AmbiguousProvenance));
    }

    #[test]
    fn joined_indices_are_a_phi() {
        let body = vec![
            Stmt::Op(Op::I32(0)),
            Stmt::Op(Op::I32(1)),
            Stmt::Op(Op::LocalI32(0)),
            Stmt::If {
                ty: BlockTy {
                    params: Vec::new(),
                    results: vec![Kind::I32],
                },
                then_body: vec![Stmt::Op(Op::I32(0))],
                else_body: vec![Stmt::Op(Op::I32(1))],
            },
            Stmt::Op(Op::Call(0)),
        ];
        check(&func(body), &[storage()], &[]).expect("index phi");
    }

    fn with_local(body: Vec<Stmt>) -> Func {
        let mut func = func(body);
        func.locals.push(Kind::I32);
        func
    }

    #[test]
    fn unset_local_joined_with_a_store_is_ambiguous() {
        let body = vec![
            Stmt::Op(Op::I32(0)),
            Stmt::Op(Op::LocalI32(0)),
            Stmt::If {
                ty: empty(),
                then_body: vec![Stmt::Op(Op::I32(1)), Stmt::Op(Op::LocalSet(1))],
                else_body: Vec::new(),
            },
            Stmt::Op(Op::LocalI32(1)),
            Stmt::Op(Op::I32(0)),
            Stmt::Op(Op::Call(0)),
        ];
        let err = check(&with_local(body), &[storage()], &[]).expect_err("zero");
        assert!(matches!(err, TranslateError::AmbiguousProvenance));
    }

    #[test]
    fn i32_add_keeps_joined_provenance() {
        let body = vec![
            Stmt::Op(Op::I32(0)),
            Stmt::Op(Op::LocalI32(0)),
            Stmt::If {
                ty: BlockTy {
                    params: Vec::new(),
                    results: vec![Kind::I32],
                },
                then_body: vec![Stmt::Op(Op::I32(0))],
                else_body: vec![Stmt::Op(Op::I32(1))],
            },
            Stmt::Op(Op::I32(0)),
            Stmt::Op(Op::I32Add),
            Stmt::Op(Op::I32(0)),
            Stmt::Op(Op::Call(0)),
        ];
        let err = check(&func(body), &[storage()], &[]).expect_err("add");
        assert!(matches!(err, TranslateError::AmbiguousProvenance));
    }

    #[test]
    fn i32_eqz_keeps_joined_provenance() {
        let body = vec![
            Stmt::Op(Op::I32(0)),
            Stmt::Op(Op::LocalI32(0)),
            Stmt::If {
                ty: BlockTy {
                    params: Vec::new(),
                    results: vec![Kind::I32],
                },
                then_body: vec![Stmt::Op(Op::I32(0))],
                else_body: vec![Stmt::Op(Op::I32(1))],
            },
            Stmt::Op(Op::I32Eqz),
            Stmt::Op(Op::I32(0)),
            Stmt::Op(Op::Call(0)),
        ];
        let err = check(&func(body), &[storage()], &[]).expect_err("eqz");
        assert!(matches!(err, TranslateError::AmbiguousProvenance));
    }

    #[test]
    fn branch_to_the_function_is_rejected() {
        let body = vec![Stmt::Block {
            ty: empty(),
            body: vec![Stmt::Br(1)],
        }];
        let err = check(&func(body), &[], &[]).expect_err("return");
        assert!(matches!(err, TranslateError::FunctionBranch));
    }
}
