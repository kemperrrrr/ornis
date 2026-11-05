//! Control-flow shapes the first uniformity tests did not cover.
//!
//! `br_table`, `br_if` leaving a loop, loop-carried locals, `select`, and a
//! deep call dag. A 1×1×1 workgroup is not checked.

use std::time::Instant;

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::BarrierSite;
use ornis_wasm2gpu::TranslateError;
use ornis_wasm2gpu::translate;

#[allow(dead_code)]
mod common;

use common::fragment;
use common::import;
use common::vertex;

fn barriers() -> String {
    import(symbols::WORKGROUP_BARRIER, "")
}

fn module(helpers: &str, body: &str) -> Vec<u8> {
    let funcs = format!(
        r#"{helpers}
           (func (export "{cs}") (param i32) {body})
           {fs}"#,
        cs = symbols::CS_MAIN,
        fs = fragment("f32.const 0 f32.const 0 f32.const 0 f32.const 1"),
    );
    wat::parse_str(format!("(module {} {} {})", barriers(), funcs, vertex())).expect("wat")
}

fn expect_uniform(helpers: &str, body: &str) {
    translate(&module(helpers, body)).expect("uniform barrier");
}

fn expect_divergent(helpers: &str, body: &str, site: BarrierSite) {
    let error = translate(&module(helpers, body)).expect_err("divergent barrier");
    match &error {
        TranslateError::DivergentBarrier {
            barrier,
            site: got,
            function,
            position: _,
        } => {
            assert_eq!(*barrier, symbols::WORKGROUP_BARRIER, "{error}");
            assert_eq!(*got, site, "{error}");
            assert!(!function.is_empty(), "{error}");
        }
        other => panic!("expected a divergent barrier, got {other}"),
    }
    let text = error.to_string();
    assert!(text.contains("global_invocation_id.x"), "{text}");
    assert!(text.contains("position"), "{text}");
}

#[test]
fn br_if_that_leaves_a_loop_on_the_invocation_id_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "(block (loop call ${wb} local.get 0 br_if 1 br 0))",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Loop,
    );
}

#[test]
fn uniform_br_table_does_not_hide_a_later_divergent_branch() {
    expect_divergent(
        "",
        &format!(
            "(block (block (br_table 0 1 (i32.const 0)))) local.get 0 if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn invocation_br_table_with_a_barrier_on_one_arm_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "(block (block (br_table 0 1 (local.get 0))) call ${wb})",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn barrier_after_a_reconverging_br_table_is_uniform() {
    expect_uniform(
        "",
        &format!(
            "(block (block (br_table 0 1 (i32.const 0)))) call ${wb}",
            wb = symbols::WORKGROUP_BARRIER,
        ),
    );
}

#[test]
fn select_on_the_invocation_id_is_rejected_even_when_arms_match() {
    expect_divergent(
        "",
        &format!(
            "i32.const 1 i32.const 1 local.get 0 select if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn copying_the_invocation_id_through_local_set_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "(local i32) local.get 0 local.set 1 local.get 1 if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn copying_the_invocation_id_through_local_tee_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "(local i32) local.get 0 local.tee 1 if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn a_helper_result_that_is_the_invocation_id_is_rejected() {
    let helper = "(func $id (param i32) (result i32) local.get 0)";
    expect_divergent(
        helper,
        &format!(
            "local.get 0 call $id if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn loop_carried_invocation_id_is_rejected_on_the_next_iteration() {
    expect_divergent(
        "",
        &format!(
            "(local i32) (loop local.get 1 if call ${wb} end local.get 0 local.set 1 local.get 0 br_if 0)",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn barrier_after_an_invocation_dependent_loop_is_uniform() {
    expect_uniform(
        "",
        &format!(
            "(loop local.get 0 br_if 0) call ${wb}",
            wb = symbols::WORKGROUP_BARRIER,
        ),
    );
}

#[test]
fn an_inner_loop_exited_by_the_invocation_id_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "(block (loop (loop call ${wb} local.get 0 br_if 2 br 0) br 0))",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Loop,
    );
}

#[test]
fn a_deep_call_dag_stays_in_milliseconds() {
    let depth = 20usize;
    let mut helpers = format!("(func $f0 call ${wb})", wb = symbols::WORKGROUP_BARRIER,);
    let mut index = 1usize;
    while index < depth {
        let prev = index - 1;
        helpers.push_str(&format!("(func $f{index} call $f{prev} call $f{prev})"));
        index += 1;
    }
    let top = depth - 1;
    let started = Instant::now();
    expect_uniform(&helpers, &format!("call $f{top} call $f{top}"));
    let elapsed = started.elapsed();
    assert!(
        elapsed.as_millis() < 500,
        "depth {depth} took {elapsed:?}, memoization missed"
    );
}

#[test]
fn the_error_names_the_function_and_a_position() {
    let error = translate(&module(
        "",
        &format!(
            "local.get 0 if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
    ))
    .expect_err("divergent");
    let TranslateError::DivergentBarrier {
        function, position, ..
    } = &error
    else {
        panic!("expected a divergent barrier, got {error}");
    };
    assert_eq!(function, symbols::CS_MAIN);
    let text = error.to_string();
    assert!(text.contains(symbols::CS_MAIN), "{text}");
    assert!(text.contains(&position.to_string()), "{text}");
}

#[test]
fn a_workgroup_of_one_skips_the_pass() {
    let body = format!(
        r#"(func (export "{cs}") (param i32) local.get 0 if call ${wb} end)
           {fs}"#,
        cs = symbols::CS_MAIN,
        wb = symbols::WORKGROUP_BARRIER,
        fs = fragment("f32.const 0 f32.const 0 f32.const 0 f32.const 1"),
    );
    let wasm = wat::parse_str(format!(
        r#"(module
            (@custom "{entries}" "compute,{cs},1,1,1\n")
            {imports}
            {vertex}
            {body})"#,
        entries = symbols::ENTRY_SECTION,
        cs = symbols::CS_MAIN,
        imports = barriers(),
        vertex = vertex(),
    ))
    .expect("wat");
    translate(&wasm).expect("single invocation cannot diverge");
}

#[test]
fn a_store_under_a_uniform_condition_stays_uniform() {
    expect_uniform(
        "",
        &format!(
            "(local i32) i32.const 1 if i32.const 1 local.set 1 end local.get 1 if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
    );
}

#[test]
fn local_set_inside_an_invocation_if_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "(local i32) local.get 0 if i32.const 1 local.set 1 end local.get 1 if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn local_tee_inside_an_invocation_if_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "(local i32) local.get 0 if i32.const 1 local.tee 1 drop end local.get 1 if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn constants_in_both_arms_of_an_invocation_if_are_rejected() {
    expect_divergent(
        "",
        &format!(
            "(local i32) local.get 0 if i32.const 1 local.set 1 else i32.const 0 local.set 1 end local.get 1 i32.const 1 i32.eq if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn an_invocation_if_result_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "local.get 0 if (result i32) i32.const 0 else i32.const 1 end if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn a_flag_cleared_after_an_invocation_br_if_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "(local i32) (block i32.const 1 local.set 1 local.get 0 br_if 0 i32.const 0 local.set 1) local.get 1 if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn a_store_after_an_invocation_br_table_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "(local i32) (block (block (br_table 0 1 (local.get 0))) i32.const 1 local.set 1) local.get 1 if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn a_counter_of_an_invocation_dependent_loop_is_rejected() {
    expect_divergent(
        "",
        &format!(
            "(local i32) (loop local.get 1 i32.const 1 i32.add local.set 1 local.get 0 br_if 0) local.get 1 if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn a_helper_store_under_the_invocation_id_is_rejected() {
    let helper = "(func $h (param i32) (result i32) (local i32) local.get 0 if i32.const 1 local.set 1 else i32.const 0 local.set 1 end local.get 1)";
    expect_divergent(
        helper,
        &format!(
            "local.get 0 call $h if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn a_helper_return_under_the_invocation_id_is_rejected() {
    let helper = "(func $h (param i32) (result i32) (local i32) i32.const 0 local.set 1 local.get 0 if i32.const 1 local.set 1 local.get 1 return end local.get 1)";
    expect_divergent(
        helper,
        &format!(
            "local.get 0 call $h if call ${wb} end",
            wb = symbols::WORKGROUP_BARRIER,
        ),
        BarrierSite::Branch,
    );
}

#[test]
fn recursion_through_a_barrier_import_is_rejected() {
    let helpers = "(func $a call $b) (func $b call $a)";
    let error = translate(&module(helpers, "call $a")).expect_err("recursion");
    assert!(matches!(error, TranslateError::Recursion), "{error}");
}
