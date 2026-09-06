//! Python language adapter for the [`ScriptEngine`] plugin seam (Phase 6).
//!
//! The core engine knows only [`ScriptEngine`]; this crate is the third
//! concrete language behind it (after Rhai and Rune). Each module is
//! compiled with `Mode::Exec` on [`ScriptEngine::load`] (and recompiled
//! on [`ScriptEngine::hot_reload`], keeping the old globals when the new
//! source fails) and executed once into its own globals dict; calls look
//! the named function up in that dict and invoke it positionally.
//!
//! `rustpython-vm` is `!Send`/`!Sync` by design (interpreter state is
//! bound to the thread that enters it), so the interpreter lives on a
//! dedicated worker thread for the engine's lifetime. Only owned data
//! (source text, JSON bytes, ids) crosses the channel; `PyObjectRef`
//! never leaves the worker. The interpreter is built
//! [`Interpreter::without_stdlib`] — guest code gets builtins only, no
//! `import` machinery — which is enough for the JSON argument/return
//! codec and keeps the dependency surface minimal.
//!
//! Argument codec is JSON, mirroring the Rhai/Rune adapters: `args` must
//! be a JSON array (empty input means no arguments); each element maps
//! to a Python object (`null` → `None`, bool/int/float/string/list/dict
//! by shape) and the return value is serialized back to JSON bytes.
//! Scripts stay side-effect free with respect to the host: the only
//! shared state is the argument/return payload.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

use ornis_core::script::{BatchHandle, BatchResult, ScriptEngine, ScriptError, ScriptHandle};
use rustpython_vm::{
    Interpreter, PyObjectRef, Settings, VirtualMachine,
    builtins::{PyBool, PyDict, PyDictRef, PyFloat, PyInt, PyList, PyNone, PyStr},
    compiler::Mode,
    function::{FuncArgs, KwArgs, PosArgs},
};
use serde_json::Value;

/// Python implementation of [`ScriptEngine`].
///
/// The public handle owns the request channel to the worker thread; the
/// `Mutex` exists only to satisfy the trait's `Sync` bound (all trait
/// methods take `&mut self`, so locking is uncontended in practice).
pub struct PythonScriptEngine {
    tx: Mutex<Sender<Request>>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl PythonScriptEngine {
    /// Spawns the interpreter worker thread.
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || worker_main(rx));
        Self {
            tx: Mutex::new(tx),
            worker: Mutex::new(Some(worker)),
        }
    }

    /// Sends a request to the worker and blocks for its reply.
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError`] if the worker thread is gone or refuses an
    /// answer (both mean the interpreter is unusable).
    fn roundtrip<T>(
        &self,
        send: impl FnOnce(Sender<Result<T, String>>) -> Request,
    ) -> Result<T, ScriptError> {
        let (reply_tx, reply_rx) = mpsc::channel();
        let tx = self
            .tx
            .lock()
            .map_err(|_| ScriptError("python worker lock poisoned".into()))?;
        tx.send(send(reply_tx))
            .map_err(|_| ScriptError("python worker thread is gone".into()))?;
        reply_rx
            .recv()
            .map_err(|_| ScriptError("python worker refused to answer".into()))?
            .map_err(ScriptError)
    }
}

impl Default for PythonScriptEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PythonScriptEngine {
    fn drop(&mut self) {
        // Best effort: tell the worker to exit, then reap it. Failures
        // here only mean the worker already died on its own.
        if let Ok(tx) = self.tx.lock()
            && tx.send(Request::Shutdown).is_ok()
            && let Ok(mut slot) = self.worker.lock()
            && let Some(handle) = slot.take()
        {
            let _ = handle.join();
        }
    }
}

/// Work orders for the interpreter thread. Every variant carries its
/// reply channel; only owned, `Send` data crosses the thread boundary.
enum Request {
    Load {
        name: String,
        source: String,
        reply: Sender<Result<u64, String>>,
    },
    Call {
        id: u64,
        func: String,
        args: Vec<u8>,
        reply: Sender<Result<Vec<u8>, String>>,
    },
    HotReload {
        id: u64,
        source: String,
        reply: Sender<Result<(), String>>,
    },
    Unload {
        id: u64,
    },
    Shutdown,
}

/// One loaded module: its globals dict. Hot reload swaps the dict
/// atomically after the new source compiles and executes cleanly.
struct Module {
    globals: PyDictRef,
}

/// Interpreter thread entry point: a single `enter` for the whole
/// lifetime, dispatching requests until `Shutdown` or channel close.
fn worker_main(rx: Receiver<Request>) {
    let interp = Interpreter::without_stdlib(Settings::default());
    interp.enter(|vm| {
        let mut modules: HashMap<u64, Module> = HashMap::new();
        let mut next_id: u64 = 0;
        for req in rx {
            match req {
                Request::Load {
                    name,
                    source,
                    reply,
                } => {
                    let out = load_module(vm, &mut modules, &mut next_id, &name, &source);
                    let _ = reply.send(out);
                }
                Request::Call {
                    id,
                    func,
                    args,
                    reply,
                } => {
                    let out = call_module(vm, &modules, id, &func, &args);
                    let _ = reply.send(out);
                }
                Request::HotReload { id, source, reply } => {
                    let out = hot_reload_module(vm, &mut modules, id, &source);
                    let _ = reply.send(out);
                }
                Request::Unload { id } => {
                    modules.remove(&id);
                }
                Request::Shutdown => break,
            }
        }
    });
}

/// Compile `source`, execute it into a fresh globals dict and register
/// the module under a new id.
fn load_module(
    vm: &VirtualMachine,
    modules: &mut HashMap<u64, Module>,
    next_id: &mut u64,
    name: &str,
    source: &str,
) -> Result<u64, String> {
    let scope = exec_module(vm, name, source)?;
    let id = *next_id;
    *next_id += 1;
    modules.insert(id, Module { globals: scope });
    Ok(id)
}

/// Recompile `source` and, only on success, swap the module's globals —
/// a failing reload keeps the previous version live, per the trait
/// contract.
fn hot_reload_module(
    vm: &VirtualMachine,
    modules: &mut HashMap<u64, Module>,
    id: u64,
    source: &str,
) -> Result<(), String> {
    if !modules.contains_key(&id) {
        return Err(format!("unknown handle {id}"));
    }
    let name = format!("<hot-reload:{id}>");
    let globals = exec_module(vm, &name, source)?;
    // Recompile + re-exec succeeded: publish atomically.
    if let Some(slot) = modules.get_mut(&id) {
        slot.globals = globals;
    }
    Ok(())
}

/// Compile `source` in `Exec` mode and run it into a fresh scope,
/// returning the populated globals dict.
fn exec_module(vm: &VirtualMachine, name: &str, source: &str) -> Result<PyDictRef, String> {
    let code = vm
        .compile(source, Mode::Exec, name.to_owned())
        .map_err(|e| format!("python compile failed: {e:?}"))?;
    let scope = vm.new_scope_with_builtins();
    vm.run_code_obj(code, scope.clone())
        .map_err(|e| format!("python exec failed: {e:?}"))?;
    Ok(scope.globals)
}

/// Look the function up in the module's globals and call it with the
/// JSON-decoded positional arguments.
fn call_module(
    vm: &VirtualMachine,
    modules: &HashMap<u64, Module>,
    id: u64,
    func: &str,
    args: &[u8],
) -> Result<Vec<u8>, String> {
    let module = modules
        .get(&id)
        .ok_or_else(|| format!("unknown handle {id}"))?;
    let values = decode_call_args(args)?;
    let py_args = values.iter().map(|v| json_to_py(vm, v)).collect::<Vec<_>>();
    let target = module
        .globals
        .get_item(func, vm)
        .map_err(|e| format!("python lookup `{func}` failed: {e:?}"))?;
    let ret = target
        .call(FuncArgs::new(PosArgs::new(py_args), KwArgs::default()), vm)
        .map_err(|e| format!("python call `{func}` failed: {e:?}"))?;
    let json = py_to_json(vm, &ret)?;
    serde_json::to_vec(&json).map_err(|e| format!("cannot encode return value: {e}"))
}

/// Raw `args` bytes to JSON values: empty input means no arguments,
/// otherwise the payload must be a JSON array (mirrors Rhai/Rune).
fn decode_call_args(args: &[u8]) -> Result<Vec<Value>, String> {
    if args.is_empty() {
        return Ok(Vec::new());
    }
    let parsed: Value =
        serde_json::from_slice(args).map_err(|e| format!("args must be a JSON array: {e}"))?;
    match parsed {
        Value::Array(items) => Ok(items),
        _ => Err("args must be a JSON array".into()),
    }
}

/// JSON value to a Python object. Total: every JSON shape maps, so this
/// is infallible (`u64` widens into Python's big ints).
fn json_to_py(vm: &VirtualMachine, v: &Value) -> PyObjectRef {
    match v {
        Value::Null => vm.ctx.none(),
        Value::Bool(b) => vm.ctx.new_bool(*b).into(),
        Value::Number(n) => json_number_to_py(vm, n),
        Value::String(s) => vm.ctx.new_str(s.clone()).into(),
        Value::Array(items) => {
            let elems = items.iter().map(|item| json_to_py(vm, item)).collect();
            vm.ctx.new_list(elems).into()
        }
        Value::Object(obj) => {
            let dict = vm.ctx.new_dict();
            for (k, val) in obj {
                // Keys come from parsed JSON; construction cannot fail.
                let _ = dict.set_item(k.as_str(), json_to_py(vm, val), vm);
            }
            dict.into()
        }
    }
}

/// JSON number to a Python number: integers that fit `i64` stay exact
/// (wider `u64` becomes a big int), everything else becomes a float.
fn json_number_to_py(vm: &VirtualMachine, n: &serde_json::Number) -> PyObjectRef {
    if let Some(i) = n.as_i64() {
        vm.ctx.new_int(i).into()
    } else if let Some(u) = n.as_u64() {
        vm.ctx.new_int(u).into()
    } else if let Some(f) = n.as_f64() {
        vm.ctx.new_float(f).into()
    } else {
        vm.ctx.none()
    }
}

/// Python return value back to JSON. Anything without a JSON shape
/// (tuples, sets, class instances, …) is an error — the codec is
/// deliberately closed, mirroring the Rhai/Rune adapters.
fn py_to_json(vm: &VirtualMachine, obj: &PyObjectRef) -> Result<Value, String> {
    if obj.downcast_ref::<PyNone>().is_some() {
        return Ok(Value::Null);
    }
    // `bool` before `int`: `bool` subclasses `int` in Python.
    if obj.downcast_ref::<PyBool>().is_some() {
        let b = obj
            .clone()
            .is_true(vm)
            .map_err(|e| format!("cannot read bool return: {e:?}"))?;
        return Ok(Value::Bool(b));
    }
    if let Some(i) = obj.downcast_ref::<PyInt>() {
        return int_to_json(i, vm);
    }
    if let Some(f) = obj.downcast_ref::<PyFloat>() {
        return serde_json::Number::from_f64(f.to_f64())
            .map(Value::Number)
            .ok_or_else(|| "non-finite float cannot be JSON".to_owned());
    }
    if let Some(s) = obj.downcast_ref::<PyStr>() {
        let text = s
            .to_str()
            .ok_or_else(|| "non-UTF8 string cannot be JSON".to_owned())?;
        return Ok(Value::String(text.to_owned()));
    }
    if let Some(list) = obj.downcast_ref::<PyList>() {
        let items = list
            .borrow_vec()
            .iter()
            .map(|item| py_to_json(vm, item))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Value::Array(items));
    }
    if let Some(dict) = obj.downcast_ref::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.items_vec() {
            let key = k
                .downcast_ref::<PyStr>()
                .and_then(|s| s.to_str())
                .ok_or_else(|| "non-string dict key cannot be JSON".to_owned())?
                .to_owned();
            map.insert(key, py_to_json(vm, &v)?);
        }
        return Ok(Value::Object(map));
    }
    Err(format!("unsupported return type: {}", obj.class().name()))
}

/// Python int to JSON: fits `i64` exactly, wider big ints fall back to
/// the nearest float (same policy as the Rhai adapter).
fn int_to_json(i: &PyInt, vm: &VirtualMachine) -> Result<Value, String> {
    if let Ok(small) = i.try_to_primitive::<i64>(vm) {
        return Ok(Value::Number(small.into()));
    }
    let decimal = i.to_str_radix_10();
    decimal
        .parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
        .map(Value::Number)
        .ok_or_else(|| format!("integer out of range: {decimal}"))
}

impl ScriptEngine for PythonScriptEngine {
    fn load(&mut self, name: &str, source: &str) -> Result<ScriptHandle, ScriptError> {
        let name = name.to_owned();
        let source = source.to_owned();
        self.roundtrip(|reply| Request::Load {
            name,
            source,
            reply,
        })
        .map(ScriptHandle)
    }

    fn call(
        &mut self,
        handle: &ScriptHandle,
        func: &str,
        args: &[u8],
    ) -> Result<Vec<u8>, ScriptError> {
        let id = handle.0;
        let func = func.to_owned();
        let args = args.to_owned();
        self.roundtrip(|reply| Request::Call {
            id,
            func,
            args,
            reply,
        })
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
        let id = handle.0;
        let source = new_source.to_owned();
        self.roundtrip(|reply| Request::HotReload { id, source, reply })
    }

    fn unload(&mut self, handle: &ScriptHandle) {
        let tx = self.tx.lock().expect("python worker lock poisoned");
        let _ = tx.send(Request::Unload { id: handle.0 });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_with(source: &str) -> (PythonScriptEngine, ScriptHandle) {
        let mut eng = PythonScriptEngine::default();
        let h = eng.load("test", source).expect("load");
        (eng, h)
    }

    #[test]
    fn call_adds_integers_and_returns_json() {
        let (mut eng, h) = engine_with("def add(a, b):\n    return a + b\n");
        let out = eng.call(&h, "add", b"[2, 3]").expect("call");
        assert_eq!(out, b"5");
    }

    #[test]
    fn call_round_trips_mixed_shapes() {
        let (mut eng, h) = engine_with(
            "def echo(name, n, flag, list, obj):\n    return [name, n, flag, list, obj]\n",
        );
        let out = eng
            .call(&h, "echo", br#"["hi", 2.5, true, [1, 2], {"k": 1}]"#)
            .expect("call");
        let v: Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(v, serde_json::json!(["hi", 2.5, true, [1, 2], {"k": 1}]));
    }

    #[test]
    fn call_round_trips_objects_through_dicts() {
        let (mut eng, h) = engine_with("def ident(x):\n    return x\n");
        let out = eng
            .call(&h, "ident", br#"[{"k": 1, "s": "v"}]"#)
            .expect("call");
        let v: Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(v, serde_json::json!({"k": 1, "s": "v"}));
    }

    #[test]
    fn unknown_function_and_handle_are_errors() {
        let (mut eng, h) = engine_with("def f():\n    return 1\n");
        assert!(eng.call(&h, "missing", b"[]").is_err());
        assert!(eng.call(&ScriptHandle(999), "f", b"[]").is_err());
        assert!(eng.load("bad", "def broken(:").is_err());
        assert!(eng.call(&h, "f", b"not json").is_err());
        assert!(eng.call(&h, "f", b"42").is_err());
    }

    #[test]
    fn none_round_trips_as_null() {
        let (mut eng, h) = engine_with("def ident(x):\n    return x\n");
        let out = eng.call(&h, "ident", b"[null]").expect("call");
        assert_eq!(out, b"null");
    }

    #[test]
    fn hot_reload_replaces_and_keeps_old_on_failure() {
        let (mut eng, h) = engine_with("def v():\n    return 1\n");
        eng.hot_reload(&h, "def v():\n    return 2\n")
            .expect("reload");
        let out = eng.call(&h, "v", b"").expect("call");
        assert_eq!(out, b"2");
        assert!(eng.hot_reload(&h, "def broken(:").is_err());
        let out = eng.call(&h, "v", b"").expect("old version live");
        assert_eq!(out, b"2");
        assert!(
            eng.hot_reload(&ScriptHandle(999), "def v():\n    return 3\n")
                .is_err()
        );
    }

    #[test]
    fn batch_collects_per_call_outcomes() {
        let (mut eng, h) = engine_with("def ok():\n    return 1\n");
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
    fn unload_drops_the_module() {
        let (mut eng, h) = engine_with("def f():\n    return 1\n");
        eng.unload(&h);
        assert!(eng.call(&h, "f", b"").is_err());
    }
}
