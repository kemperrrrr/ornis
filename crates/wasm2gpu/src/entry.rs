//! Entry-point metadata lookup for the lowerer.

use ornis_gpu_abi::entry_meta::Stage;
use ornis_gpu_abi::symbols;

use crate::error::TranslateError;
use crate::ir::Module;

/// Workgroup size the guest declared for `cs_main`.
///
/// The `#[shader::compute]` macro records it in the `ornis_gpu.entries`
/// custom section. Modules without that metadata keep the ABI default, which
/// is what hand-written WAT fixtures use.
pub(crate) fn compute_workgroup(module: &Module) -> Result<[u32; 3], TranslateError> {
    let mut found = None;
    for entry in &module.entries {
        if entry.export != symbols::CS_MAIN {
            continue;
        }
        if entry.stage != Stage::Compute {
            return Err(TranslateError::EntryMetadata(format!(
                "{} is marked {:?}, not compute",
                symbols::CS_MAIN,
                entry.stage
            )));
        }
        if found.replace(entry.workgroup).is_some() {
            return Err(TranslateError::EntryMetadata(format!(
                "duplicate {} entry",
                symbols::CS_MAIN
            )));
        }
    }
    Ok(found.unwrap_or(symbols::CS_WORKGROUP))
}
