//! Workgroup buffers: binding shape, atomics and existing storage lookups.

use super::SCALAR_WIDTH;
use super::builders::Builders;
use super::emit::scalar;
use super::texture::Elem;
use crate::error::TranslateError;

impl Builders {
    pub(crate) fn workgroup_words(
        &mut self,
    ) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
        if let Some(var) = self.workgroup {
            return Ok(var);
        }
        let base = scalar(&mut self.module, naga::ScalarKind::Uint);
        let var = self.workgroup_array("workgroup_scratch", base)?;
        self.workgroup = Some(var);
        Ok(var)
    }

    pub(crate) fn workgroup_atomics(
        &mut self,
    ) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
        if let Some(var) = self.workgroup_atomic {
            return Ok(var);
        }
        let base = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Atomic(naga::Scalar {
                    kind: naga::ScalarKind::Uint,
                    width: SCALAR_WIDTH,
                }),
            },
            naga::Span::default(),
        );
        let var = self.workgroup_array("workgroup_atomics", base)?;
        self.workgroup_atomic = Some(var);
        Ok(var)
    }

    pub(crate) fn workgroup_array(
        &mut self,
        name: &str,
        base: naga::Handle<naga::Type>,
    ) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
        let Some(len) = std::num::NonZeroU32::new(ornis_gpu_abi::symbols::WORKGROUP_WORDS) else {
            return Err(TranslateError::UnsupportedType(
                "workgroup array is empty".to_string(),
            ));
        };
        let ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Array {
                    base,
                    size: naga::ArraySize::Constant(len),
                    stride: ornis_gpu_abi::F32_SIZE,
                },
            },
            naga::Span::default(),
        );
        Ok(self.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(name.to_string()),
                space: naga::AddressSpace::WorkGroup,
                binding: None,
                ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        ))
    }

    pub(crate) fn ivec2(&mut self) -> naga::Handle<naga::Type> {
        if let Some(ty) = self.ivec2 {
            return ty;
        }
        let ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Bi,
                    scalar: naga::Scalar {
                        kind: naga::ScalarKind::Sint,
                        width: SCALAR_WIDTH,
                    },
                },
            },
            naga::Span::default(),
        );
        self.ivec2 = Some(ty);
        ty
    }

    /// The variable already created for this binding, upgrading it to
    /// writable when this use stores. A second element type is rejected.
    pub(crate) fn existing_storage(
        &mut self,
        group: u32,
        binding: u32,
        elem: Elem,
        write: bool,
    ) -> Result<Option<naga::Handle<naga::GlobalVariable>>, TranslateError> {
        let Some((existing, handle)) = self.storage_binding.get(&(group, binding)) else {
            return Ok(None);
        };
        if *existing != elem {
            return Err(TranslateError::UnsupportedType(format!(
                "storage binding {group}:{binding} is used as two types"
            )));
        }
        if write
            && let naga::AddressSpace::Storage { access } =
                &mut self.module.global_variables[*handle].space
        {
            *access |= naga::StorageAccess::STORE;
        }
        Ok(Some(*handle))
    }
}
