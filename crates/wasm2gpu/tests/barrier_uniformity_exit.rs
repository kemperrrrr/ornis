//! Exits of an `if` whose condition is not the invocation id.
//!
//! The walk used to stop when those arms left toward different labels, and
//! a `br` arm's locals were dropped at the merge. Both hide a later barrier.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::BarrierSite;
use ornis_wasm2gpu::TranslateError;
use ornis_wasm2gpu::translate;

#[allow(dead_code)]
mod common;

use common::fragment;
use common::import;
use common::vertex;

fn module(body: &str) -> Vec<u8> {
    let funcs = format!(
        r#"(func (export "{cs}") (param i32) {body})
           {fs}"#,
        cs = symbols::CS_MAIN,
        fs = fragment("f32.const 0 f32.const 0 f32.const 0 f32.const 1"),
    );
    let imports = import(symbols::WORKGROUP_BARRIER, "");
    wat::parse_str(format!("(module {} {} {})", imports, funcs, vertex())).expect("wat")
}

fn expect_uniform(body: &str) {
    translate(&module(body)).expect("uniform barrier");
}

fn expect_divergent(body: &str, site: BarrierSite) {
    let error = translate(&module(body)).expect_err("divergent barrier");
    match &error {
        TranslateError::DivergentBarrier { site: got, .. } => {
            assert_eq!(*got, site, "{error}");
        }
        other => panic!("expected a divergent barrier, got {other}"),
    }
}

#[test]
fn an_unproven_if_does_not_hide_a_later_invocation_barrier() {
    let wb = symbols::WORKGROUP_BARRIER;
    expect_divergent(
        &format!("(block (block i32.const 1 if br 1 else br 2 end)) local.get 0 if call ${wb} end"),
        BarrierSite::Branch,
    );
}

#[test]
fn an_unknown_if_does_not_hide_a_later_invocation_barrier() {
    let wb = symbols::WORKGROUP_BARRIER;
    expect_divergent(
        &format!(
            "(local i32) (block (block local.get 1 if br 1 else br 2 end)) local.get 0 if call ${wb} end"
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn an_unproven_if_does_not_hide_a_later_partial_return() {
    let wb = symbols::WORKGROUP_BARRIER;
    expect_divergent(
        &format!(
            "(block (block i32.const 1 if br 1 else br 2 end)) (block local.get 0 br_if 0 return) call ${wb}"
        ),
        BarrierSite::EarlyReturn,
    );
}

#[test]
fn an_unproven_if_does_not_hide_a_later_br_if() {
    let wb = symbols::WORKGROUP_BARRIER;
    expect_divergent(
        &format!(
            "(block (block (block i32.const 1 if br 1 else br 2 end) local.get 0 br_if 1) call ${wb})"
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn an_unproven_else_br_if_is_not_skipped() {
    let wb = symbols::WORKGROUP_BARRIER;
    expect_divergent(
        &format!(
            "(block (block (block i32.const 1 if br 1 else local.get 0 br_if 3 br 2 end)) call ${wb})"
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn a_copy_on_the_br_arm_of_an_invocation_if_is_rejected() {
    let wb = symbols::WORKGROUP_BARRIER;
    expect_divergent(
        &format!(
            "(local i32) (block local.get 0 if else local.get 0 local.set 1 br 1 end) local.get 1 if call ${wb} end"
        ),
        BarrierSite::Branch,
    );
    expect_divergent(
        &format!(
            "(local i32) (block local.get 0 if local.get 0 local.set 1 br 1 end) local.get 1 if call ${wb} end"
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn a_constant_on_the_br_arm_of_an_invocation_if_is_rejected() {
    let wb = symbols::WORKGROUP_BARRIER;
    expect_divergent(
        &format!(
            "(local i32) (block local.get 0 if else i32.const 1 local.set 1 br 1 end) local.get 1 if call ${wb} end"
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn a_copy_on_the_br_arm_of_a_constant_if_is_rejected() {
    let wb = symbols::WORKGROUP_BARRIER;
    expect_divergent(
        &format!(
            "(local i32) (block i32.const 0 if else local.get 0 local.set 1 br 1 end) local.get 1 if call ${wb} end"
        ),
        BarrierSite::Branch,
    );
    expect_divergent(
        &format!(
            "(local i32) (block i32.const 1 if local.get 0 local.set 1 br 1 end) local.get 1 if call ${wb} end"
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn a_store_on_the_fallthrough_arm_is_still_rejected() {
    let wb = symbols::WORKGROUP_BARRIER;
    expect_divergent(
        &format!(
            "(local i32) (block local.get 0 if local.get 0 local.set 1 else br 1 end) local.get 1 if call ${wb} end"
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn unproven_branches_that_meet_before_the_barrier_stay_uniform() {
    let wb = symbols::WORKGROUP_BARRIER;
    expect_uniform(&format!(
        "(block (block i32.const 1 if br 1 else br 2 end)) call ${wb}"
    ));
}
