//! Graph queries: reachability, dominance and natural loops.

use super::EXIT;
use crate::parse::Terminator;
use std::collections::{HashMap, HashSet};

pub(crate) fn reachable<'a>(
    succs: &HashMap<&'a str, Vec<&'a str>>,
    entry: &'a str,
) -> HashSet<&'a str> {
    let mut seen = HashSet::new();
    let mut stack = vec![entry];
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        if let Some(next) = succs.get(node) {
            stack.extend(next.iter().copied());
        }
    }
    seen
}

pub(crate) fn dominators<'a>(
    reachable: &HashSet<&'a str>,
    preds: &HashMap<&'a str, Vec<&'a str>>,
    entry: &'a str,
) -> HashMap<&'a str, HashSet<&'a str>> {
    let mut doms: HashMap<&'a str, HashSet<&'a str>> = HashMap::new();
    for node in reachable {
        if *node == entry {
            doms.insert(node, HashSet::from([entry]));
        } else {
            doms.insert(node, reachable.clone());
        }
    }
    loop {
        let mut changed = false;
        for node in reachable {
            if *node == entry {
                continue;
            }
            let mut next: Option<HashSet<&'a str>> = None;
            for pred in preds.get(node).into_iter().flatten() {
                if !reachable.contains(pred) {
                    continue;
                }
                let pred_doms = &doms[pred];
                next = Some(match next {
                    None => pred_doms.clone(),
                    Some(acc) => acc.intersection(pred_doms).copied().collect(),
                });
            }
            let mut next = next.unwrap_or_default();
            next.insert(node);
            if next != doms[node] {
                doms.insert(node, next);
                changed = true;
            }
        }
        if !changed {
            return doms;
        }
    }
}

pub(crate) fn postdominators<'a>(
    reachable: &HashSet<&'a str>,
    succs: &HashMap<&'a str, Vec<&'a str>>,
) -> HashMap<&'a str, HashSet<&'a str>> {
    let mut all = reachable.clone();
    all.insert(EXIT);
    let mut post: HashMap<&'a str, HashSet<&'a str>> = HashMap::new();
    for node in &all {
        if *node == EXIT {
            post.insert(node, HashSet::from([EXIT]));
        } else {
            post.insert(node, all.clone());
        }
    }
    loop {
        let mut changed = false;
        for node in reachable {
            let out: Vec<&'a str> = succs.get(node).into_iter().flatten().copied().collect();
            let mut next: Option<HashSet<&'a str>> = if out.is_empty() {
                Some(HashSet::from([EXIT]))
            } else {
                None
            };
            for target in out {
                let target_post = if target == EXIT {
                    HashSet::from([EXIT])
                } else {
                    post[target].clone()
                };
                next = Some(match next {
                    None => target_post,
                    Some(acc) => acc.intersection(&target_post).copied().collect(),
                });
            }
            let mut next = next.unwrap_or_default();
            next.insert(node);
            if next != post[node] {
                post.insert(node, next);
                changed = true;
            }
        }
        if !changed {
            return post;
        }
    }
}

/// Natural loop body for the back edge `latch -> header`.
pub(crate) fn natural_loop<'a>(
    header: &'a str,
    latch: &'a str,
    preds: &HashMap<&'a str, Vec<&'a str>>,
) -> HashSet<&'a str> {
    let mut body = HashSet::from([header]);
    let mut stack = vec![latch];
    while let Some(node) = stack.pop() {
        if node == header || !body.insert(node) {
            continue;
        }
        if let Some(inputs) = preds.get(node) {
            stack.extend(inputs.iter().copied());
        }
    }
    body
}

/// Blocks control can leave `term` for, in the order the IR names them.
pub(crate) fn successors(term: &Terminator) -> Vec<&str> {
    match term {
        Terminator::Ret(_) | Terminator::Unreachable => Vec::new(),
        Terminator::BrUncond(target) => vec![target.as_str()],
        Terminator::BrCond {
            true_label,
            false_label,
            ..
        } => vec![true_label.as_str(), false_label.as_str()],
        Terminator::Switch { default, cases, .. } => {
            let mut out = vec![default.as_str()];
            out.extend(cases.iter().map(|(_, label)| label.as_str()));
            out
        }
    }
}
