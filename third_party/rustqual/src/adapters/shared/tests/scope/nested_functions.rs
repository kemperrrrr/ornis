//! Nested functions live in the scope of their enclosing function, not in
//! the project-wide function set: a same-named closure in another module
//! must not be mistaken for a call to them.
use super::*;

/// A function nested inside another function is not a project-wide function.
#[test]
fn test_nested_function_is_not_a_project_function() {
    let scope = build_scope("fn outer() { fn inner() {} }");
    assert!(
        !scope.functions.contains("inner"),
        "a nested fn is not visible project-wide"
    );
    assert!(
        scope.all_functions.contains("inner"),
        "a nested fn is still a declared function"
    );
}

/// The nested function is filed under the scope of its enclosing function.
#[test]
fn test_nested_function_is_filed_under_enclosing_scope() {
    let scope = build_scope("fn outer() { fn inner() {} }");
    let nested = scope
        .nested_functions
        .get("test.rs::outer")
        .expect("the enclosing scope knows the nested fn");
    assert!(nested.contains("inner"), "got {nested:?}");
}

/// The nested function of another scope is not an own call.
#[test]
fn test_nested_function_of_other_scope_is_not_an_own_call() {
    let scope = build_scope("fn outer() { fn inner() {} } fn other() {}");
    assert!(
        !scope.is_own_function_in_scope("inner", Some("test.rs::other")),
        "a nested fn is invisible outside its scope"
    );
    assert!(
        scope.is_own_function_in_scope("inner", Some("test.rs::outer")),
        "a nested fn is an own call inside its scope"
    );
}

/// A module-level function stays an own call in every scope.
#[test]
fn test_module_level_function_is_own_call_in_any_scope() {
    let scope = build_scope("fn helper() {} fn outer() { fn inner() {} }");
    assert!(scope.is_own_function_in_scope("helper", Some("test.rs::outer")));
    assert!(scope.is_own_function_in_scope("helper", None));
}

/// A nested function inside a method belongs to the method's scope.
#[test]
fn test_nested_function_in_method_uses_method_scope() {
    let scope = build_scope("struct S; impl S { fn run(&self) { fn inner() {} } }");
    let nested = scope
        .nested_functions
        .get("test.rs::S::run")
        .expect("the method scope knows the nested fn");
    assert!(nested.contains("inner"), "got {nested:?}");
}
