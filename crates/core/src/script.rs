//! Scripting plugin seam for Ornis (Phase 6, plan D1).
//!
//! The engine core knows only the [`ScriptEngine`] trait — concrete languages
//! (Rhai, Rune, Python, WASM components) are adapters that implement it.
//! Hot-path ECS loops stay typed; this trait is for tooling, FFI and
//! editor integration, mirroring `PhysicsEngine` and `RenderBackend`.

use std::collections::HashMap;

/// Opaque handle to a script instance (one script file / module).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScriptHandle(pub u64);

/// Opaque handle to a component/entity batch entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchHandle(pub u64);

/// Error returned by script operations.
#[derive(Debug, Clone)]
pub struct ScriptError(pub String);

impl std::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ScriptError {}

/// Result of a batch operation.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Per-handle outcome: `Ok(())` or `Err` string.
    pub outcomes: HashMap<BatchHandle, Result<(), String>>,
}

/// Plugin trait for scripting backends.
///
/// The core engine depends only on this trait; language specifics live
/// in adapter crates (e.g. `ornis-rhai`). This keeps the hot ECS
/// path typed while allowing editor/FFI to drive scripts through
/// type-erased handles.
///
/// # Errors
///
/// All methods return [`ScriptError`] on load/compile or runtime failure.
pub trait ScriptEngine: Send + Sync {
    /// Load (or reload) a script module from source text.
    ///
    /// Returns a handle that can be used with [`ScriptEngine::call`].
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError`] if the source fails to parse/compile.
    fn load(&mut self, name: &str, source: &str) -> Result<ScriptHandle, ScriptError>;

    /// Call an exported function on a loaded script.
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError`] if the handle is unknown or the call traps.
    fn call(
        &mut self,
        handle: &ScriptHandle,
        func: &str,
        args: &[u8],
    ) -> Result<Vec<u8>, ScriptError>;

    /// Batch variant: one call instead of `N` individual calls.
    ///
    /// Handles are preferred over raw pointers into `SparseSet` cells —
    /// pointers are only valid in-process and break the WASM sandbox.
    fn batch_call(
        &mut self,
        calls: &[(ScriptHandle, String, Vec<u8>)],
    ) -> BatchResult;

    /// Hot-reload a previously loaded module in place.
    ///
    /// # Errors
    ///
    /// Returns [`ScriptError`] if the new source fails to compile; the
    /// previous version must remain usable.
    fn hot_reload(&mut self, handle: &ScriptHandle, new_source: &str) -> Result<(), ScriptError>;

    /// Unload a script module.
    fn unload(&mut self, handle: &ScriptHandle);
}

/// No-op engine useful for tests and as a baseline adapter.
#[derive(Debug, Default)]
pub struct NoopScriptEngine {
    next_id: u64,
    modules: HashMap<u64, String>,
}

impl ScriptEngine for NoopScriptEngine {
    fn load(&mut self, _name: &str, source: &str) -> Result<ScriptHandle, ScriptError> {
        let id = self.next_id;
        self.next_id += 1;
        self.modules.insert(id, source.to_owned());
        Ok(ScriptHandle(id))
    }

    fn call(
        &mut self,
        handle: &ScriptHandle,
        _func: &str,
        _args: &[u8],
    ) -> Result<Vec<u8>, ScriptError> {
        if self.modules.contains_key(&handle.0) {
            Ok(Vec::new())
        } else {
            Err(ScriptError(format!("unknown handle {}", handle.0)))
        }
    }

    fn batch_call(
        &mut self,
        calls: &[(ScriptHandle, String, Vec<u8>)],
    ) -> BatchResult {
        let outcomes = calls
            .iter()
            .enumerate()
            .map(|(i, (h, _, _))| {
                let r = if self.modules.contains_key(&h.0) {
                    Ok(())
                } else {
                    Err(format!("unknown handle {}", h.0))
                };
                (BatchHandle(i as u64), r)
            })
            .collect();
        BatchResult { outcomes }
    }

    fn hot_reload(&mut self, handle: &ScriptHandle, new_source: &str) -> Result<(), ScriptError> {
        if let Some(entry) = self.modules.get_mut(&handle.0) {
            *entry = new_source.to_owned();
            Ok(())
        } else {
            Err(ScriptError(format!("unknown handle {}", handle.0)))
        }
    }

    fn unload(&mut self, handle: &ScriptHandle) {
        self.modules.remove(&handle.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_load_call_reload() {
        let mut eng = NoopScriptEngine::default();
        let h = eng.load("test", "fn foo() {}").expect("load");
        eng.call(&h, "foo", &[]).expect("call");
        eng.hot_reload(&h, "fn foo() { 42 }").expect("reload");
        let r = eng.batch_call(&[(h.clone(), "foo".into(), vec![])]);
        assert_eq!(r.outcomes.len(), 1);
        eng.unload(&h);
        assert!(eng.call(&h, "foo", &[]).is_err());
    }
}
