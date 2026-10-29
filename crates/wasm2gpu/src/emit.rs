//! Validate the naga module and print WGSL plus per-stage SPIR-V.

use crate::error::TranslateError;

/// WGSL text and SPIR-V words for the vertex and fragment entries.
#[derive(Debug)]
pub struct Translated {
    /// Both entry points in one module.
    pub wgsl: String,
    /// SPIR-V words for `vs_main`.
    pub vertex_spirv: Vec<u32>,
    /// SPIR-V words for `fs_main`.
    pub fragment_spirv: Vec<u32>,
    /// SPIR-V words for `cs_main`. Empty when the module has no compute entry.
    pub compute_spirv: Vec<u32>,
}

pub(super) fn emit(module: &naga::Module) -> Result<Translated, TranslateError> {
    if binding_array(module) {
        return Err(TranslateError::Unsupported);
    }
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
    let vertex_spirv = spirv(
        module,
        &info,
        naga::ShaderStage::Vertex,
        ornis_gpu_abi::symbols::VS_MAIN,
    )?;
    let fragment_spirv = spirv(
        module,
        &info,
        naga::ShaderStage::Fragment,
        ornis_gpu_abi::symbols::FS_MAIN,
    )?;
    reparse_spirv(&vertex_spirv)?;
    reparse_spirv(&fragment_spirv)?;
    let compute_spirv = compute_spirv(module, &info)?;
    Ok(Translated {
        wgsl,
        vertex_spirv,
        fragment_spirv,
        compute_spirv,
    })
}

fn compute_spirv(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<Vec<u32>, TranslateError> {
    let Some(entry) = module
        .entry_points
        .iter()
        .find(|entry| entry.stage == naga::ShaderStage::Compute)
    else {
        return Ok(Vec::new());
    };
    let words = spirv(module, info, naga::ShaderStage::Compute, &entry.name)?;
    reparse_spirv(&words)?;
    Ok(words)
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

fn binding_array(module: &naga::Module) -> bool {
    for (_, ty) in module.types.iter() {
        if matches!(ty.inner, naga::TypeInner::BindingArray { .. }) {
            return true;
        }
    }
    false
}

fn spirv(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    stage: naga::ShaderStage,
    entry_point: &str,
) -> Result<Vec<u32>, TranslateError> {
    let mut options = naga::back::spv::Options::default();
    options
        .flags
        .remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);
    options.fake_missing_bindings = false;
    options.binding_map = binding_map(module);
    options.debug_info = None;
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: stage,
        entry_point: entry_point.to_string(),
    };
    let words = naga::back::spv::write_vec(module, info, &options, Some(&pipeline))
        .map_err(|err| TranslateError::Naga(err.to_string()))?;
    Ok(words)
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
    .map_err(|err| TranslateError::Naga(err.to_string()))?;
    Ok(())
}
