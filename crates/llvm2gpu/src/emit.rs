//! Validate the naga module and print WGSL plus per-stage SPIR-V.

use crate::error::TranslateError;

/// WGSL text and SPIR-V words for the compute entry.
#[derive(Debug)]
pub struct Translated {
    /// Whole module in WGSL.
    pub wgsl: String,
    /// SPIR-V words for `cs_main`. Empty when there is no compute entry.
    pub compute_spirv: Vec<u32>,
}

/// Validate `module`, print WGSL, and emit compute SPIR-V.
///
/// Vertex and fragment SPIR-V are not emitted by this translator; it
/// produces compute-only modules because the LLVM IR subset it supports
/// describes a single scalar compute function.
pub(super) fn emit(module: &naga::Module) -> Result<Translated, TranslateError> {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(module)
    .map_err(|err| TranslateError::Naga(naga_chain(&err)))?;
    let wgsl =
        naga::back::wgsl::write_string(module, &info, naga::back::wgsl::WriterFlags::empty())
            .map_err(|err| TranslateError::Naga(err.to_string()))?;
    let parsed = naga::front::wgsl::parse_str(&wgsl)
        .map_err(|err| TranslateError::Naga(err.emit_to_string(&wgsl)))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&parsed)
    .map_err(|err| TranslateError::Naga(naga_chain(&err)))?;
    let compute_spirv = match module
        .entry_points
        .iter()
        .find(|entry| entry.stage == naga::ShaderStage::Compute)
    {
        Some(entry) => {
            let words = spirv(module, &info, &entry.name)?;
            // Naga's SPIR-V parser rejects the atomic workgroup initializer
            // that Native mode emits, so this round-trip stops at WGSL.
            if !workgroup_has_atomic(module) {
                reparse_spirv(&words)?;
            }
            words
        }
        None => Vec::new(),
    };
    Ok(Translated {
        wgsl,
        compute_spirv,
    })
}

fn spirv(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    entry_point: &str,
) -> Result<Vec<u32>, TranslateError> {
    let mut options = naga::back::spv::Options::default();
    if workgroup_has_atomic(module) {
        // Polyfill mode stores the whole workgroup array. That store is
        // illegal when the element is `atomic<u32>`. Native mode keeps the
        // zero-initializer in the SPIR-V extension instead.
        options.zero_initialize_workgroup_memory =
            naga::back::spv::ZeroInitializeWorkgroupMemoryMode::Native;
    }
    options
        .flags
        .remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);
    options.fake_missing_bindings = false;
    options.binding_map = binding_map(module);
    options.debug_info = None;
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: entry_point.to_string(),
    };
    let words = naga::back::spv::write_vec(module, info, &options, Some(&pipeline))
        .map_err(|err| TranslateError::Naga(err.to_string()))?;
    Ok(words)
}

fn workgroup_has_atomic(module: &naga::Module) -> bool {
    module.global_variables.iter().any(|(_, var)| {
        let naga::TypeInner::Array { base, .. } = module.types[var.ty].inner else {
            return false;
        };
        var.space == naga::AddressSpace::WorkGroup
            && matches!(module.types[base].inner, naga::TypeInner::Atomic(_))
    })
}

fn binding_map(module: &naga::Module) -> naga::back::spv::BindingMap {
    let mut map = naga::back::spv::BindingMap::default();
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = global.binding else {
            continue;
        };
        map.insert(
            binding,
            naga::back::spv::BindingInfo {
                descriptor_set: binding.group,
                binding: binding.binding,
                binding_array_size: None,
            },
        );
    }
    map
}

fn naga_chain(err: &impl std::error::Error) -> String {
    let mut msg = err.to_string();
    let mut src = std::error::Error::source(err);
    while let Some(inner) = src {
        msg.push_str(": ");
        msg.push_str(&inner.to_string());
        src = std::error::Error::source(inner);
    }
    msg
}

fn reparse_spirv(words: &[u32]) -> Result<(), TranslateError> {
    let mut bytes = Vec::new();
    for word in words {
        bytes.extend(word.to_le_bytes());
    }
    let parsed = naga::front::spv::parse_u8_slice(
        &bytes,
        &naga::front::spv::Options {
            adjust_coordinate_space: false,
            strict_capabilities: false,
            block_ctx_dump_prefix: None,
        },
    )
    .map_err(|err| TranslateError::Naga(err.to_string()))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&parsed)
    .map_err(|err| TranslateError::Naga(naga_chain(&err)))?;
    Ok(())
}
