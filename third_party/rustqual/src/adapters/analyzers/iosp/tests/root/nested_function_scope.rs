//! Nested functions are visible only inside the function that encloses them.
//!
//! A nested `fn` used to land in the project-wide function set, so a call to
//! a same-named closure in another module counted as an own call — and the
//! function holding that closure turned into an IOSP violation in a file the
//! patch never touched.
use super::*;

/// The minimal example of the issue: a nested `fn local` in one module and a
/// closure named `local` in another.
fn leak_files() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "src/newcode.rs",
            "pub fn new_feature() -> i32 { fn local(x: i32) -> i32 { x } local(1) }",
        ),
        (
            "src/physics.rs",
            "pub fn remap_spec(x: i32) -> i32 { let shifted = x + 1; let local = |v: i32| -> i32 { v * 2 }; local(shifted) }",
        ),
    ]
}

/// A closure named `local` in another file must not become an own call
/// because some nested `fn local` exists elsewhere in the project.
#[test]
fn nested_function_name_does_not_leak_into_another_file() {
    let results = parse_and_analyze_files(&leak_files());
    let remap = results
        .iter()
        .find(|r| r.name == "remap_spec")
        .expect("remap_spec analysed");
    assert!(
        remap.own_calls.is_empty(),
        "a closure call is not a call to a nested fn of another file, got {:?}",
        remap.own_calls
    );
}

/// The call inside the enclosing function stays an own call: the nested `fn`
/// is in scope there.
#[test]
fn nested_function_call_inside_enclosing_function_is_an_own_call() {
    let code = "pub fn new_feature() -> i32 { fn local(x: i32) -> i32 { x } local(1) }";
    let calls = own_calls_of(code, "new_feature");
    assert!(
        calls.iter().any(|c| c == "local"),
        "the nested fn call is an own call inside its scope, got {calls:?}"
    );
}

/// A nested `fn` in one file does not turn a same-named closure call in
/// another file into a violation — the regression that broke the gate.
#[test]
fn nested_function_in_other_file_does_not_create_violation() {
    let results = parse_and_analyze_files(&leak_files());
    let violations: Vec<&str> = results
        .iter()
        .filter(|r| matches!(r.classification, Classification::Violation { .. }))
        .map(|r| r.name.as_str())
        .collect();
    assert!(
        violations.is_empty(),
        "no function may become a violation because of a nested fn elsewhere, got {violations:?}"
    );
}

/// A nested `fn` inside a method is visible in that method only.
#[test]
fn nested_function_inside_method_is_scoped_to_the_method() {
    let files = [
        (
            "src/impl.rs",
            "struct S; impl S { fn run(&self) -> i32 { fn local(x: i32) -> i32 { x } local(1) } }",
        ),
        (
            "src/other.rs",
            "pub fn other() -> i32 { let local = |v: i32| -> i32 { v * 2 }; local(1) }",
        ),
    ];
    let results = parse_and_analyze_files(&files);
    let other = results
        .iter()
        .find(|r| r.name == "other")
        .expect("other analysed");
    assert!(
        other.own_calls.is_empty(),
        "the method's nested fn is invisible in another file, got {:?}",
        other.own_calls
    );
    let run = results
        .iter()
        .find(|r| r.name == "run")
        .expect("run analysed");
    assert!(
        run.own_calls.iter().any(|c| c == "local"),
        "the nested fn call is an own call inside its method, got {:?}",
        run.own_calls
    );
}

/// A nested `fn` two levels deep is visible in the whole enclosing function.
#[test]
fn deeply_nested_function_is_visible_in_the_outer_scope() {
    let code = "pub fn outer() -> i32 { fn middle() -> i32 { fn inner(x: i32) -> i32 { x } inner(1) } middle() }";
    let calls = own_calls_of(code, "outer");
    assert!(
        calls.iter().any(|c| c == "middle") && calls.iter().any(|c| c == "inner"),
        "nested calls of the outer scope are own calls, got {calls:?}"
    );
}
