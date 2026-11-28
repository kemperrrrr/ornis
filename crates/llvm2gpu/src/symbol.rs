//! The ABI symbol a call names, exactly.
//!
//! A call reaches the translator as the Rust symbol rustc emitted —
//! `_RNvNtCs7qzl3DdH8fB_16ornis_gpu_shader3abi19import_vertex_index` — while
//! the hand-written IR of the tests names the same import bare
//! (`vertex_index`). Matching used to be `callee.contains(symbol)`, which
//! answers for names it was never asked about: a kernel helper called
//! `import_vertex_index_extra` was lowered as `@builtin(vertex_index)`, and a
//! helper whose body left the subset disappeared entirely because its call fell
//! through to the import dispatch.
//!
//! [`imported`] reduces both forms to the symbol and the call sites compare it
//! whole. Whether a callee is a *helper* is not a name question at all: it is
//! decided by whether the module defines it.

/// The ABI symbol `callee` names — `vertex_index` for both mangled and bare
/// forms of the import.
///
/// A mangled Rust symbol ends in `<len>import_<symbol>`, so the tail after the
/// last `import_` is the symbol; a name that does not look like that is
/// returned unchanged, which leaves `llvm.fabs.f32`-style intrinsics and bare
/// test IR intact. A kernel helper named `import_<symbol>` would be
/// indistinguishable from the import — the SDK owns the `import_` prefix, and
/// [`crate::lower`] never dispatches a name the module defines.
pub(crate) fn imported(callee: &str) -> &str {
    let name = if is_mangled(callee) {
        segment(callee).unwrap_or(callee)
    } else {
        callee
    };
    // Several SDK shims are named `import_<symbol>` because they exist only to
    // declare the import; the ABI symbol is the part after the prefix.
    name.strip_prefix("import_").unwrap_or(name)
}

/// Whether `symbol` is a mangled Rust name rather than a bare IR name.
///
/// `_R` is the v0 scheme and `_ZN` the legacy one. The check is not cosmetic:
/// a bare ABI name such as `texture_load_2d_f32x4` ends in `3` + `2x4`, which
/// a length-prefixed scan would happily read as a segment.
fn is_mangled(symbol: &str) -> bool {
    symbol.starts_with("_R") || symbol.starts_with("_ZN")
}

/// Last `<length><identifier>` production of a mangled Rust symbol.
///
/// A v0 symbol is a sequence of productions and the identifier is one of them,
/// each spelled as its decimal length followed by that many identifier chars
/// (`29texture_sample_level_2d_f32x4`). The length makes the boundary exact
/// without parsing the rest of the symbol, which is why this is not a
/// demangler: only the tail is needed, and a name that is not mangled has no
/// such production and is returned whole by the caller.
fn segment(symbol: &str) -> Option<&str> {
    for length in (1..=symbol.len()).rev() {
        let start = symbol.len() - length;
        let candidate = &symbol[start..];
        if !candidate
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            break;
        }
        let prefix = &symbol[..start];
        let digits: String = prefix
            .chars()
            .rev()
            .take_while(char::is_ascii_digit)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if !digits.is_empty() && digits.parse::<usize>() == Ok(length) {
            return Some(candidate);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::imported;

    /// Both spellings of an import reduce to the bare symbol, and names that
    /// merely contain it do not.
    #[test]
    fn a_symbol_is_the_tail_of_the_mangled_name() {
        assert_eq!(
            imported("_RNvNtCs7qzl3DdH8fB_16ornis_gpu_shader3abi19import_vertex_index"),
            "vertex_index"
        );
        assert_eq!(
            imported("_RNvNtCs7qzl3DdH8fB_16ornis_gpu_shader3abi22import_vertex_position"),
            "vertex_position"
        );
        assert_eq!(imported("vertex_index"), "vertex_index");
        assert_eq!(imported("llvm.fabs.f32"), "llvm.fabs.f32");
        assert_eq!(
            imported("_RNvNtCs7qzl3DdH8fB_16ornis_gpu_shader3abi21texture_load_2d_f32x4"),
            "texture_load_2d_f32x4",
            "a segment without the `import_` prefix is not an import name"
        );
    }

    /// A longer name that contains a symbol is a different symbol: this is the
    /// substitution the previous `contains` matching allowed.
    #[test]
    fn a_longer_name_is_not_the_symbol() {
        assert_ne!(
            imported("_RNxxxx25import_vertex_input_u32x4"),
            "vertex_input_u32"
        );
        assert_eq!(
            imported("_RNxxxx25import_vertex_input_u32x4"),
            "vertex_input_u32x4"
        );
        assert_ne!(
            imported("_RNxxxx29import_fragment_uv_lane_extra"),
            "fragment_uv_lane"
        );
    }

    /// The SDK's shim name is what the call names, not the link name: a shim
    /// called `texture_sample_level_f32x4` reaches the translator as
    /// `...29texture_sample_level_2d_f32x4` because that is its Rust name.
    #[test]
    fn a_shim_keeps_its_own_spelling() {
        assert_eq!(
            imported("_RNvNtCs7qzl3DdH8fB_16ornis_gpu_shader3abi29texture_sample_level_2d_f32x4"),
            "texture_sample_level_2d_f32x4"
        );
        assert_eq!(
            imported("_RNvNtCs7qzl3DdH8fB_16ornis_gpu_shader3abi23import_fragment_uv_lane"),
            "fragment_uv_lane"
        );
    }
}
