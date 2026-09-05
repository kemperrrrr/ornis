//! Rhai language adapter for the [`ScriptEngine`] plugin seam (Phase 6).
//!
//! The core engine knows only [`ScriptEngine`]; this crate is the first
//! concrete language behind it. Modules compile to [`rhai::AST`] on
//! [`ScriptEngine::load`] (and recompile on
//! [`ScriptEngine::hot_reload`], keeping the old AST when the new source
//! fails). Calls run the named function in a fresh [`rhai::Scope`].
//!
//! Argument codec is JSON: `args` must be a JSON array (empty input means
//! no arguments); each element maps to a [`rhai::Dynamic`] (`null` → `()`,
//! bool/number/string/array/object by shape) and the return value is
//! serialized back to JSON bytes. Scripts stay side-effect free with
//! respect to the host: the only shared state is the argument/return
//! payload.

use std::collections::HashMap;

use ornis_core::script::{BatchHandle, BatchResult, ScriptEngine, ScriptError, ScriptHandle};
use rhai::{Dynamic, Engine, Map, Scope};
use serde_json::Value;

/// Rhai implementation of [`ScriptEngine`].
///
/// Each loaded module keeps its own compiled [`rhai::AST`]; function calls
/// execute against the module's AST with a fresh scope, so concurrent calls
/// through `&mut self` never leak script variables between invocations.
#[derive(Debug)]
pub struct RhaiScriptEngine {
    engine: Engine,
    modules: HashMap<u64, rhai::AST>,
    next_id: u64,
}

impl Default for RhaiScriptEngine {
    fn default() -> Self {
        Self {
            engine: Engine::new(),
            modules: HashMap::new(),
            next_id: 0,
        }
    }
}

impl RhaiScriptEngine {
    /// Call an already-resolved module AST.
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError`] if `args` is not a JSON array or the script
    /// traps at runtime.
    fn call_ast(&self, ast: &rhai::AST, func: &str, args: &[u8]) -> Result<Vec<u8>, ScriptError> {
        let dynamic_args = decode_call_args(args)?;
        let mut scope = Scope::new();
        let out: Dynamic = self
            .engine
            .call_fn(&mut scope, ast, func, dynamic_args)
            .map_err(|e| ScriptError(format!("rhai call `{func}` failed: {e}")))?;
        let value = dynamic_to_json(&out)?;
        serde_json::to_vec(&value)
            .map_err(|e| ScriptError(format!("cannot encode return value: {e}")))
    }
}

/// Raw `args` bytes to a [`Dynamic`] argument list: empty input means no
/// arguments, otherwise the payload must be a JSON array.
///
/// # Errors
///
/// Returns [`ScriptError`] on malformed JSON or a non-array payload.
fn decode_call_args(args: &[u8]) -> Result<Vec<Dynamic>, ScriptError> {
    if args.is_empty() {
        return Ok(Vec::new());
    }
    let parsed: Value = serde_json::from_slice(args)
        .map_err(|e| ScriptError(format!("args must be a JSON array: {e}")))?;
    match parsed {
        Value::Array(items) => items.iter().map(json_to_dynamic).collect(),
        _ => Err(ScriptError("args must be a JSON array".into())),
    }
}

/// JSON number to [`Dynamic`]: integers that fit `i64` stay integers so
/// scripts see `INT`, everything else becomes `FLOAT`.
fn json_number_to_dynamic(n: &serde_json::Number) -> Result<Dynamic, ScriptError> {
    if let Some(i) = n.as_i64() {
        Ok(Dynamic::from_int(i))
    } else if let Some(f) = n.as_f64() {
        Ok(Dynamic::from_float(f))
    } else {
        Err(ScriptError(format!("number out of range: {n}")))
    }
}

/// JSON value to [`Dynamic`].
///
/// # Errors
///
/// Returns [`ScriptError`] on numbers that fit neither `i64` nor `f64`.
fn json_to_dynamic(v: &Value) -> Result<Dynamic, ScriptError> {
    match v {
        Value::Null => Ok(Dynamic::UNIT),
        Value::Bool(b) => Ok(Dynamic::from_bool(*b)),
        Value::Number(n) => json_number_to_dynamic(n),
        Value::String(s) => Ok(Dynamic::from(s.clone())),
        Value::Array(items) => json_array_to_dynamic(items),
        Value::Object(obj) => json_object_to_dynamic(obj),
    }
}

/// JSON array to a Rhai array [`Dynamic`].
///
/// # Errors
///
/// Returns [`ScriptError`] on numbers that fit neither `i64` nor `f64`.
fn json_array_to_dynamic(items: &[Value]) -> Result<Dynamic, ScriptError> {
    items
        .iter()
        .map(json_to_dynamic)
        .collect::<Result<Vec<_>, _>>()
        .map(Dynamic::from_array)
}

/// JSON object to a Rhai map [`Dynamic`].
///
/// # Errors
///
/// Returns [`ScriptError`] on numbers that fit neither `i64` nor `f64`.
fn json_object_to_dynamic(obj: &serde_json::Map<String, Value>) -> Result<Dynamic, ScriptError> {
    let mut map = Map::new();
    for (k, val) in obj {
        map.insert(k.as_str().into(), json_to_dynamic(val)?);
    }
    Ok(Dynamic::from_map(map))
}

/// [`Dynamic`] back to JSON. Anything without a JSON shape (including the
/// unit `()`) becomes `null`, so scripts can return `()` for procedures.
// qual:allow(iosp) reason: match-dispatcher over Dynamic shapes; splitting further would scatter one codec across fragments.
fn dynamic_to_json(d: &Dynamic) -> Result<Value, ScriptError> {
    if let Some(scalar) = dynamic_scalar_to_json(d)? {
        return Ok(scalar);
    }
    if let Some(composite) = dynamic_composite_to_json(d)? {
        return Ok(composite);
    }
    Ok(Value::Null)
}

/// Plain Rhai scalars to JSON: bool, `INT`, `FLOAT` (non-finite becomes
/// `null` — JSON has no NaN/Infinity) and strings.
fn dynamic_scalar_to_json(d: &Dynamic) -> Result<Option<Value>, ScriptError> {
    if let Ok(b) = d.as_bool() {
        return Ok(Some(Value::Bool(b)));
    }
    if let Ok(i) = d.as_int() {
        return Ok(Some(Value::Number(i.into())));
    }
    if let Ok(f) = d.as_float() {
        return Ok(Some(
            serde_json::Number::from_f64(f)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        ));
    }
    if d.is_string() {
        let s = d
            .clone()
            .into_string()
            .map_err(|_| ScriptError("shared string cannot move out".into()))?;
        return Ok(Some(Value::String(s)));
    }
    Ok(None)
}

/// Rhai arrays and maps to JSON, recursing into [`dynamic_to_json`].
// qual:allow(iosp) reason: match-dispatcher over Dynamic shapes; splitting further would scatter one codec across fragments.
fn dynamic_composite_to_json(d: &Dynamic) -> Result<Option<Value>, ScriptError> {
    if d.is_array() {
        let items = d
            .clone()
            .cast::<rhai::Array>()
            .into_iter()
            .map(|item| dynamic_to_json(&item))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Some(Value::Array(items)));
    }
    if d.is_map() {
        let mut obj = serde_json::Map::new();
        for (k, v) in d.clone().cast::<Map>().iter() {
            obj.insert(k.to_string(), dynamic_to_json(v)?);
        }
        return Ok(Some(Value::Object(obj)));
    }
    Ok(None)
}

impl ScriptEngine for RhaiScriptEngine {
    fn load(&mut self, _name: &str, source: &str) -> Result<ScriptHandle, ScriptError> {
        let ast = self
            .engine
            .compile(source)
            .map_err(|e| ScriptError(format!("rhai compile failed: {e}")))?;
        let id = self.next_id;
        self.next_id += 1;
        self.modules.insert(id, ast);
        Ok(ScriptHandle(id))
    }

    fn call(
        &mut self,
        handle: &ScriptHandle,
        func: &str,
        args: &[u8],
    ) -> Result<Vec<u8>, ScriptError> {
        let ast = self
            .modules
            .get(&handle.0)
            .ok_or_else(|| ScriptError(format!("unknown handle {}", handle.0)))?;
        self.call_ast(ast, func, args)
    }

    fn batch_call(&mut self, calls: &[(ScriptHandle, String, Vec<u8>)]) -> BatchResult {
        let outcomes = calls
            .iter()
            .enumerate()
            .map(|(i, (h, func, args))| {
                let r = self.call(h, func, args).map(|_| ()).map_err(|e| e.0);
                (BatchHandle(i as u64), r)
            })
            .collect();
        BatchResult { outcomes }
    }

    fn hot_reload(&mut self, handle: &ScriptHandle, new_source: &str) -> Result<(), ScriptError> {
        // Compile first: on failure the previous AST stays live, per the
        // trait contract.
        let ast = self
            .engine
            .compile(new_source)
            .map_err(|e| ScriptError(format!("rhai recompile failed: {e}")))?;
        match self.modules.get_mut(&handle.0) {
            Some(entry) => {
                *entry = ast;
                Ok(())
            }
            None => Err(ScriptError(format!("unknown handle {}", handle.0))),
        }
    }

    fn unload(&mut self, handle: &ScriptHandle) {
        self.modules.remove(&handle.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_with(math: &str) -> (RhaiScriptEngine, ScriptHandle) {
        let mut eng = RhaiScriptEngine::default();
        let h = eng.load("math", math).expect("load");
        (eng, h)
    }

    #[test]
    fn call_adds_integers_and_returns_json() {
        let (mut eng, h) = engine_with("fn add(a, b) { a + b }");
        let out = eng.call(&h, "add", b"[2, 3]").expect("call");
        assert_eq!(out, b"5");
    }

    #[test]
    fn call_round_trips_mixed_shapes() {
        let (mut eng, h) =
            engine_with("fn echo(name, n, flag, list, obj) { [name, n, flag, list, obj] }");
        let out = eng
            .call(&h, "echo", br#"["hi", 2.5, true, [1, 2], {"k": 1}]"#)
            .expect("call");
        let v: Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(v, serde_json::json!(["hi", 2.5, true, [1, 2], {"k": 1}]),);
    }

    #[test]
    fn unknown_function_and_handle_are_errors() {
        let (mut eng, h) = engine_with("fn f() { 1 }");
        assert!(eng.call(&h, "missing", b"[]").is_err());
        assert!(eng.call(&ScriptHandle(999), "f", b"[]").is_err());
        assert!(eng.load("bad", "fn broken( {").is_err());
        assert!(eng.call(&h, "f", b"not json").is_err());
        assert!(eng.call(&h, "f", b"42").is_err());
    }

    #[test]
    fn hot_reload_replaces_and_keeps_old_on_failure() {
        let (mut eng, h) = engine_with("fn v() { 1 }");
        eng.hot_reload(&h, "fn v() { 2 }").expect("reload");
        let out = eng.call(&h, "v", b"").expect("call");
        assert_eq!(out, b"2");
        assert!(eng.hot_reload(&h, "fn broken( {").is_err());
        let out = eng.call(&h, "v", b"").expect("old version live");
        assert_eq!(out, b"2");
        assert!(eng.hot_reload(&ScriptHandle(999), "fn v() { 3 }").is_err());
    }

    #[test]
    fn batch_collects_per_call_outcomes() {
        let (mut eng, h) = engine_with("fn ok() { 1 }");
        let r = eng.batch_call(&[
            (h.clone(), "ok".into(), vec![]),
            (h.clone(), "missing".into(), vec![]),
            (ScriptHandle(999), "ok".into(), vec![]),
        ]);
        assert_eq!(r.outcomes.len(), 3);
        assert!(r.outcomes[&BatchHandle(0)].is_ok());
        assert!(r.outcomes[&BatchHandle(1)].is_err());
        assert!(r.outcomes[&BatchHandle(2)].is_err());
    }

    #[test]
    fn unload_forgets_the_module() {
        let (mut eng, h) = engine_with("fn f() { 1 }");
        eng.unload(&h);
        assert!(eng.call(&h, "f", b"").is_err());
    }
}
