//! Rune language adapter for the [`ScriptEngine`] plugin seam (Phase 6).
//!
//! **Experimental — not production-ready** (see `ornis_core::script` docs).
//!
//! The core engine knows only [`ScriptEngine`]; this crate is the second
//! concrete language behind it (after `ornis-rhai`), exercising the seam
//! for the rule-of-three check. Modules compile to [`rune::Unit`] on
//! [`ScriptEngine::load`] (and recompile on [`ScriptEngine::hot_reload`],
//! keeping the old unit when the new source fails). Calls run through a
//! fresh [`rune::Vm`] per invocation built from the stored unit plus a
//! shared runtime, so concurrent calls through `&mut self` never leak
//! script state.
//!
//! Argument codec is JSON: `args` must be a JSON array (empty input means
//! no arguments); each element maps to a [`rune::Value`] (`null` → `()`,
//! bool/number/string/array/object by shape) and the return value is
//! converted back to JSON bytes. Scripts stay side-effect free with
//! respect to the host: the only shared state is the argument/return
//! payload.

use std::collections::HashMap;
use std::sync::Arc;

use ornis_core::script::{BatchHandle, BatchResult, ScriptEngine, ScriptError, ScriptHandle};
use rune::runtime::{Object, RuntimeContext};
use rune::{Context, Diagnostics, Source, Sources, Unit, Value, Vm};
use serde_json::Value as JsonValue;

/// Rune implementation of [`ScriptEngine`].
///
/// Each loaded module keeps its own compiled [`Unit`]; function calls
/// execute on a fresh [`Vm`] so script state never leaks between
/// invocations. The struct only holds [`Unit`]s plus a shared
/// [`RuntimeContext`].
pub struct RuneScriptEngine {
    runtime: Arc<RuntimeContext>,
    modules: HashMap<u64, Arc<Unit>>,
    next_id: u64,
}

impl std::fmt::Debug for RuneScriptEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuneScriptEngine")
            .field("modules", &self.modules.len())
            .field("next_id", &self.next_id)
            .finish()
    }
}

impl RuneScriptEngine {
    /// Creates an engine with the default Rune modules installed.
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError`] if the default context fails to build
    /// (practically unreachable — no user source is involved yet).
    pub fn new() -> Result<Self, ScriptError> {
        let context = Context::with_default_modules()
            .map_err(|e| ScriptError(format!("rune context: {e}")))?;
        let runtime = Arc::new(
            context
                .runtime()
                .map_err(|e| ScriptError(format!("rune runtime: {e}")))?,
        );
        Ok(Self {
            runtime,
            modules: HashMap::new(),
            next_id: 0,
        })
    }

    /// Compiles `source` into a [`Unit`], appending diagnostics to the
    /// error string on failure.
    fn compile(source: &str) -> Result<Unit, ScriptError> {
        let context = Context::with_default_modules()
            .map_err(|e| ScriptError(format!("rune context: {e}")))?;
        let mut sources = Sources::new();
        sources
            .insert(Source::memory(source).map_err(|e| ScriptError(format!("rune source: {e}")))?)
            .map_err(|e| ScriptError(format!("rune source: {e}")))?;
        let mut diagnostics = Diagnostics::new();
        let result = rune::prepare(&mut sources)
            .with_context(&context)
            .with_diagnostics(&mut diagnostics)
            .build();
        match result {
            Ok(unit) => Ok(unit),
            Err(e) => {
                let mut msg = format!("rune compile failed: {e}");
                if !diagnostics.is_empty() {
                    let mut out = rune::termcolor::NoColor::new(Vec::new());
                    if diagnostics.emit(&mut out, &sources).is_ok()
                        && let Ok(text) = String::from_utf8(out.into_inner())
                    {
                        msg.push('\n');
                        msg.push_str(&text);
                    }
                }
                Err(ScriptError(msg))
            }
        }
    }

    /// Calls an already-compiled module unit.
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError`] if `args` is not a JSON array or the script
    /// traps at runtime.
    fn call_unit(&self, unit: &Arc<Unit>, func: &str, args: &[u8]) -> Result<Vec<u8>, ScriptError> {
        let rune_args = decode_call_args(args)?;
        let mut vm = Vm::new(self.runtime.clone(), unit.clone());
        let out = vm
            .call([func], rune_args)
            .map_err(|e| ScriptError(format!("rune call `{func}` failed: {e}")))?;
        let json = value_to_json(&out)?;
        serde_json::to_vec(&json)
            .map_err(|e| ScriptError(format!("cannot encode return value: {e}")))
    }
}

impl Default for RuneScriptEngine {
    fn default() -> Self {
        Self::new().expect("rune default modules install")
    }
}

/// Raw `args` bytes to a [`Value`] argument list: empty input means no
/// arguments, otherwise the payload must be a JSON array.
///
/// # Errors
///
/// Returns [`ScriptError`] on malformed JSON, a non-array payload, or a
/// value that has no Rune shape.
fn decode_call_args(args: &[u8]) -> Result<Vec<Value>, ScriptError> {
    if args.is_empty() {
        return Ok(Vec::new());
    }
    let parsed: JsonValue = serde_json::from_slice(args)
        .map_err(|e| ScriptError(format!("args must be a JSON array: {e}")))?;
    match parsed {
        JsonValue::Array(items) => items.iter().map(json_to_value).collect(),
        _ => Err(ScriptError("args must be a JSON array".into())),
    }
}

/// JSON value to [`Value`]: integers that fit `i64` stay integers,
/// everything else numeric becomes `FLOAT`.
///
/// # Errors
///
/// Returns [`ScriptError`] on integers that fit neither `i64` nor `f64`.
fn json_to_value(v: &JsonValue) -> Result<Value, ScriptError> {
    match v {
        JsonValue::Null => Ok(Value::from(())),
        JsonValue::Bool(b) => {
            rune::to_value(*b).map_err(|e| ScriptError(format!("arg not supported: {e}")))
        }
        JsonValue::Number(n) => json_number_to_value(n),
        JsonValue::String(s) => {
            rune::to_value(s.clone()).map_err(|e| ScriptError(format!("arg not supported: {e}")))
        }
        JsonValue::Array(items) => {
            let values = items
                .iter()
                .map(json_to_value)
                .collect::<Result<Vec<_>, _>>()?;
            rune::to_value(values).map_err(|e| ScriptError(format!("arg not supported: {e}")))
        }
        JsonValue::Object(obj) => {
            let mut map = HashMap::with_capacity(obj.len());
            for (k, val) in obj {
                map.insert(k.clone(), json_to_value(val)?);
            }
            rune::to_value(map).map_err(|e| ScriptError(format!("arg not supported: {e}")))
        }
    }
}

/// JSON number to [`Value`]: integers that fit `i64` stay integers so
/// scripts see integer arithmetic, everything else becomes `FLOAT`.
///
/// # Errors
///
/// Returns [`ScriptError`] on integers that fit neither `i64` nor `f64`.
fn json_number_to_value(n: &serde_json::Number) -> Result<Value, ScriptError> {
    if let Some(i) = n.as_i64() {
        rune::to_value(i).map_err(|e| ScriptError(format!("arg not supported: {e}")))
    } else if n.as_u64().is_some() {
        Err(ScriptError(format!("number out of range: {n}")))
    } else if let Some(f) = n.as_f64() {
        rune::to_value(f).map_err(|e| ScriptError(format!("arg not supported: {e}")))
    } else {
        Err(ScriptError(format!("number out of range: {n}")))
    }
}

/// [`Value`] back to JSON. Anything without a JSON shape (including the
/// unit `()`) becomes `null`, so scripts can return `()` for procedures.
fn value_to_json(v: &Value) -> Result<JsonValue, ScriptError> {
    if v.into_unit().is_ok() {
        return Ok(JsonValue::Null);
    }
    if let Ok(b) = rune::from_value::<bool>(v.clone()) {
        return Ok(JsonValue::Bool(b));
    }
    if let Ok(i) = rune::from_value::<i64>(v.clone()) {
        return Ok(JsonValue::Number(i.into()));
    }
    if let Ok(u) = rune::from_value::<u64>(v.clone()) {
        return Ok(JsonValue::Number(u.into()));
    }
    if let Ok(f) = rune::from_value::<f64>(v.clone()) {
        return Ok(JsonValue::Number(
            serde_json::Number::from_f64(f).unwrap_or_else(|| 0.into()),
        ));
    }
    if let Ok(s) = rune::from_value::<String>(v.clone()) {
        return Ok(JsonValue::String(s));
    }
    if let Ok(items) = rune::from_value::<Vec<Value>>(v.clone()) {
        let json = items
            .iter()
            .map(value_to_json)
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(JsonValue::Array(json));
    }
    if let Ok(obj) = v.clone().downcast::<Object>() {
        let mut map = serde_json::Map::with_capacity(obj.len());
        for (k, item) in obj.iter() {
            map.insert(k.to_string(), value_to_json(item)?);
        }
        return Ok(JsonValue::Object(map));
    }
    Ok(JsonValue::Null)
}

impl ScriptEngine for RuneScriptEngine {
    fn load(&mut self, _name: &str, source: &str) -> Result<ScriptHandle, ScriptError> {
        let unit = Self::compile(source)?;
        let id = self.next_id;
        self.next_id += 1;
        self.modules.insert(id, Arc::new(unit));
        Ok(ScriptHandle(id))
    }

    fn call(
        &mut self,
        handle: &ScriptHandle,
        func: &str,
        args: &[u8],
    ) -> Result<Vec<u8>, ScriptError> {
        let unit = self
            .modules
            .get(&handle.0)
            .ok_or_else(|| ScriptError(format!("unknown handle {}", handle.0)))?;
        self.call_unit(unit, func, args)
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
        // Compile first: on failure the previous unit stays live, per the
        // trait contract.
        let unit = Self::compile(new_source)?;
        match self.modules.get_mut(&handle.0) {
            Some(entry) => {
                *entry = Arc::new(unit);
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
    use serde_json::json;

    fn engine_with(math: &str) -> (RuneScriptEngine, ScriptHandle) {
        let mut eng = RuneScriptEngine::default();
        let h = eng.load("math", math).expect("load");
        (eng, h)
    }

    #[test]
    fn call_adds_integers_and_returns_json() {
        let (mut eng, h) = engine_with("pub fn add(a, b) { a + b }");
        let out = eng.call(&h, "add", b"[2, 3]").expect("call");
        assert_eq!(out, b"5");
    }

    #[test]
    fn call_round_trips_mixed_shapes() {
        let (mut eng, h) =
            engine_with("pub fn echo(name, n, flag, list) { [name, n, flag, list] }");
        let out = eng
            .call(&h, "echo", br#"["hi", 2.5, true, [1, 2]]"#)
            .expect("call");
        let v: JsonValue = serde_json::from_slice(&out).expect("json");
        assert_eq!(v, json!(["hi", 2.5, true, [1, 2]]),);
    }

    #[test]
    fn call_round_trips_objects_through_maps() {
        let (mut eng, h) = engine_with("pub fn id(x) { x }");
        let out = eng
            .call(&h, "id", br#"[{"k": 1, "s": "v"}]"#)
            .expect("call");
        let v: JsonValue = serde_json::from_slice(&out).expect("json");
        assert_eq!(v, json!({"k": 1, "s": "v"}));
    }

    #[test]
    fn unknown_function_and_handle_are_errors() {
        let (mut eng, h) = engine_with("pub fn f() { 1 }");
        assert!(eng.call(&h, "missing", b"[]").is_err());
        assert!(eng.call(&ScriptHandle(999), "f", b"[]").is_err());
        assert!(eng.load("bad", "fn broken( {").is_err());
        assert!(eng.call(&h, "f", b"not json").is_err());
        assert!(eng.call(&h, "f", b"42").is_err());
    }

    #[test]
    fn hot_reload_replaces_and_keeps_old_on_failure() {
        let (mut eng, h) = engine_with("pub fn v() { 1 }");
        eng.hot_reload(&h, "pub fn v() { 2 }").expect("reload");
        let out = eng.call(&h, "v", b"").expect("call");
        assert_eq!(out, b"2");
        assert!(eng.hot_reload(&h, "fn broken( {").is_err());
        let out = eng.call(&h, "v", b"").expect("old version live");
        assert_eq!(out, b"2");
        assert!(
            eng.hot_reload(&ScriptHandle(999), "pub fn v() { 3 }")
                .is_err()
        );
    }

    #[test]
    fn batch_collects_per_call_outcomes() {
        let (mut eng, h) = engine_with("pub fn ok() { 1 }");
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
        let (mut eng, h) = engine_with("pub fn f() { 1 }");
        eng.unload(&h);
        assert!(eng.call(&h, "f", b"").is_err());
    }
}
