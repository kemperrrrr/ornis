//! Barrier uniformity: uniform control flow is lowered, proven divergence is not.
//!
//! The invocation id is `cs_main`'s parameter. A predicate the analysis cannot
//! tie to that id (here, a workgroup load) stays the ABI precondition.

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::BarrierSite;
use ornis_wasm2gpu::TranslateError;
use ornis_wasm2gpu::translate;

#[allow(dead_code)]
mod common;

use common::fragment;
use common::import;
use common::wasm;

fn barriers() -> String {
    format!(
        "{} {}",
        import(symbols::WORKGROUP_BARRIER, ""),
        import(symbols::STORAGE_BARRIER, ""),
    )
}

fn module(helpers: &str, body: &str) -> Vec<u8> {
    let funcs = format!(
        r#"{helpers}
           (func (export "{cs}") (param i32) {body})
           {fs}"#,
        cs = symbols::CS_MAIN,
        fs = fragment("f32.const 0 f32.const 0 f32.const 0 f32.const 1"),
    );
    wasm(&barriers(), &funcs)
}

fn expect_uniform(helpers: &str, body: &str) {
    translate(&module(helpers, body)).expect("uniform barrier");
}

fn expect_divergent(helpers: &str, body: &str, barrier: &str, site: BarrierSite) {
    let error = translate(&module(helpers, body)).expect_err("divergent barrier");
    match &error {
        TranslateError::DivergentBarrier {
            barrier: got,
            site: got_site,
            function,
            position: _,
        } => {
            assert!(*got == barrier, "{error}");
            assert!(*got_site == site, "{error}");
            assert!(!function.is_empty(), "{error}");
        }
        other => panic!("expected a divergent barrier, got {other}"),
    }
    let text = error.to_string();
    assert!(text.contains(barrier), "{text}");
    assert!(text.contains("is not uniform"), "{text}");
    assert!(text.contains("position"), "{text}");
}

#[test]
fn straight_line_barriers_are_uniform() {
    expect_uniform(
        "",
        &format!(
            "call ${workgroup} call ${storage}",
            workgroup = symbols::WORKGROUP_BARRIER,
            storage = symbols::STORAGE_BARRIER,
        ),
    );
}

#[test]
fn barrier_after_a_reconverging_invocation_branch_is_uniform() {
    // The shape of the compute ABI fixture: lane 0 takes the arm, every lane
    // reaches the barrier after `end`.
    expect_uniform(
        "",
        &format!(
            "local.get 0 i32.eqz if end call ${workgroup} call ${storage}",
            workgroup = symbols::WORKGROUP_BARRIER,
            storage = symbols::STORAGE_BARRIER,
        ),
    );
}

#[test]
fn barrier_inside_a_constant_branch_is_uniform() {
    expect_uniform(
        "",
        &format!(
            "i32.const 1 if call ${workgroup} end",
            workgroup = symbols::WORKGROUP_BARRIER,
        ),
    );
}

#[test]
fn barrier_in_a_loop_that_exits_on_a_constant_is_uniform() {
    expect_uniform(
        "",
        &format!(
            "(block (loop call ${storage} i32.const 1 br_if 1))",
            storage = symbols::STORAGE_BARRIER,
        ),
    );
}

#[test]
fn barrier_after_an_unprovable_predicate_stays_allowed() {
    let load = import(symbols::WORKGROUP_LOAD_U32, "(param i32) (result i32)");
    let body = format!(
        "i32.const 0 call ${load} if call ${workgroup} end",
        load = symbols::WORKGROUP_LOAD_U32,
        workgroup = symbols::WORKGROUP_BARRIER,
    );
    let funcs = format!(
        r#"(func (export "{cs}") (param i32) {body})
           {fs}"#,
        cs = symbols::CS_MAIN,
        fs = fragment("f32.const 0 f32.const 0 f32.const 0 f32.const 1"),
    );
    translate(&wasm(&format!("{} {}", load, barriers()), &funcs)).expect("unprovable predicate");
}

#[test]
fn barrier_in_a_helper_called_from_uniform_flow_is_uniform() {
    let helper = format!(
        "(func $sync call ${workgroup})",
        workgroup = symbols::WORKGROUP_BARRIER,
    );
    expect_uniform(&helper, "call $sync");
}

#[test]
fn barrier_inside_an_invocation_branch_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "local.get 0 if call ${workgroup} end",
            workgroup = symbols::WORKGROUP_BARRIER,
        ),
        symbols::WORKGROUP_BARRIER,
        BarrierSite::Branch,
    );
    expect_divergent(
        "",
        &format!(
            "local.get 0 if else call ${storage} end",
            storage = symbols::STORAGE_BARRIER,
        ),
        symbols::STORAGE_BARRIER,
        BarrierSite::Branch,
    );
}

#[test]
fn barrier_under_an_invocation_predicate_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "local.get 0 i32.eqz if call ${workgroup} end",
            workgroup = symbols::WORKGROUP_BARRIER,
        ),
        symbols::WORKGROUP_BARRIER,
        BarrierSite::Branch,
    );
    expect_divergent(
        "",
        &format!(
            "local.get 0 i32.const 1 i32.add if call ${storage} end",
            storage = symbols::STORAGE_BARRIER,
        ),
        symbols::STORAGE_BARRIER,
        BarrierSite::Branch,
    );
}

#[test]
fn barrier_in_a_loop_continued_by_the_invocation_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "(loop call ${workgroup} local.get 0 br_if 0)",
            workgroup = symbols::WORKGROUP_BARRIER,
        ),
        symbols::WORKGROUP_BARRIER,
        BarrierSite::Loop,
    );
}

#[test]
fn barrier_in_a_loop_exited_by_only_some_invocations_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "(block (loop call ${storage} local.get 0 (if (then (br 2))) (br 0)))",
            storage = symbols::STORAGE_BARRIER,
        ),
        symbols::STORAGE_BARRIER,
        BarrierSite::Loop,
    );
}

#[test]
fn barrier_after_an_invocation_return_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "local.get 0 if return end call ${workgroup}",
            workgroup = symbols::WORKGROUP_BARRIER,
        ),
        symbols::WORKGROUP_BARRIER,
        BarrierSite::EarlyReturn,
    );
}

#[test]
fn barrier_in_a_helper_called_from_a_divergent_branch_is_rejected() {
    let helper = format!(
        "(func $sync call ${storage})",
        storage = symbols::STORAGE_BARRIER,
    );
    expect_divergent(
        &helper,
        "local.get 0 if call $sync end",
        symbols::STORAGE_BARRIER,
        BarrierSite::Branch,
    );
}

#[test]
fn helper_that_returns_on_the_invocation_id_before_a_barrier_is_rejected() {
    let helper = format!(
        "(func $maybe (param i32) local.get 0 if return end call ${workgroup})",
        workgroup = symbols::WORKGROUP_BARRIER,
    );
    expect_divergent(
        &helper,
        "local.get 0 call $maybe",
        symbols::WORKGROUP_BARRIER,
        BarrierSite::EarlyReturn,
    );
}
