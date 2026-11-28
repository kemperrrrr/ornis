//! Module resources: storage, textures, samplers and workgroup buffers.

use super::SCALAR_WIDTH;
use super::UNIFORM_LANE_BYTES;
use super::WORD_BYTES;
use super::emit::scalar;
use super::memory::Helper;
use super::stage::Stage;
use super::texture::{Elem, TextureBinding, TextureKind, image_class};
use crate::error::TranslateError;
use crate::parse::Ty;
use std::collections::{HashMap, HashSet};

pub(crate) struct Builders {
    pub(crate) module: naga::Module,
    /// One storage variable per `(group, binding)`. A second element type is
    /// rejected: two variables at one binding would alias, and a writable
    /// binding must not alias anything.
    pub(crate) storage_binding: HashMap<(u32, u32), (Elem, naga::Handle<naga::GlobalVariable>)>,
    pub(crate) uniform_cache: HashMap<(u32, u32), naga::Handle<naga::GlobalVariable>>,
    pub(crate) uniform_struct_cache: HashMap<(u32, u32, u32), naga::Handle<naga::GlobalVariable>>,
    /// Built uniform struct types, keyed by struct id: type, byte size, align.
    pub(crate) struct_types: HashMap<u32, (naga::Handle<naga::Type>, u32, u32)>,
    /// Struct ids currently being laid out, to reject a recursive field.
    pub(crate) struct_stack: Vec<u32>,
    /// `texture_2d` / `texture_depth_2d` globals, one per binding.
    pub(crate) textures: HashMap<(u32, u32), TextureBinding>,
    /// Samplers, one per binding and comparison mode.
    ///
    /// A comparison sampler is a different type, not a different use: naga
    /// rejects a depth compare through a filtering sampler.
    pub(crate) samplers: HashMap<(u32, u32, bool), naga::Handle<naga::GlobalVariable>>,
    /// `vec2<i32>` used as a `textureLoad` coordinate.
    pub(crate) ivec2: Option<naga::Handle<naga::Type>>,
    /// Plain `array<u32, WORKGROUP_WORDS>` in workgroup memory.
    pub(crate) workgroup: Option<naga::Handle<naga::GlobalVariable>>,
    /// `array<atomic<u32>, WORKGROUP_WORDS>` in workgroup memory.
    pub(crate) workgroup_atomic: Option<naga::Handle<naga::GlobalVariable>>,
    /// Module constants by symbol name, raw little-endian contents.
    pub(crate) module_constants: HashMap<String, Vec<u8>>,
    /// Helper functions lowered into the module, by IR name.
    pub(crate) helpers: HashMap<String, Helper>,
    /// Helpers outside the subset, with the reason a call cannot lower them.
    pub(crate) skipped_helpers: HashMap<String, String>,
    /// Names the module defines: a call to one is a helper call, and every other
    /// callee is an import or an intrinsic.
    pub(crate) defined: HashSet<String>,
    /// Helpers that reach a workgroup barrier, directly or through another
    /// helper. A call to one is only uniform when its call site is.
    pub(crate) barriered: HashSet<String>,
    /// Functions whose every path ends in `unreachable` (the panic shims).
    ///
    /// A call to one never returns, so the translator drops it and lets the
    /// caller's own `unreachable` end the invocation: a shader has no place to
    /// report a panic to.
    pub(crate) diverging: HashSet<String>,
    /// `ornis_gpu.types` field tags per struct id.
    pub(crate) structs: Vec<Vec<u8>>,
    /// Which textures declare the multisampled type (and load sample 0).
    pub(crate) sampling: crate::TextureSampling,
}

impl Builders {
    pub(crate) fn new(structs: &[&[u8]], sampling: crate::TextureSampling) -> Self {
        Self {
            module: naga::Module::default(),
            storage_binding: HashMap::new(),
            uniform_cache: HashMap::new(),
            uniform_struct_cache: HashMap::new(),
            struct_types: HashMap::new(),
            struct_stack: Vec::new(),
            textures: HashMap::new(),
            samplers: HashMap::new(),
            ivec2: None,
            workgroup: None,
            workgroup_atomic: None,
            module_constants: HashMap::new(),
            helpers: HashMap::new(),
            skipped_helpers: HashMap::new(),
            defined: HashSet::new(),
            barriered: HashSet::new(),
            diverging: HashSet::new(),
            structs: structs.iter().map(|tags| tags.to_vec()).collect(),
            sampling,
        }
    }

    /// Add the lowered function as this module's next entry point.
    pub(crate) fn finish(
        &mut self,
        entry: &str,
        function: naga::Function,
        ret_ty: Ty,
        workgroup: [u32; 3],
        stage: Stage,
    ) -> Result<(), TranslateError> {
        // Only create the implicit `storage_0_0` global when the kernel
        // uses the `-> float` shape; for `-> void` the body creates its
        // own globals lazily.
        if ret_ty == Ty::Float {
            self.storage_global(0, 0, Elem::F32, true)?;
        }
        // The entry keeps the kernel's own name: a pass may carry two entries
        // of one stage (`vs_main_indirect`), and naga refuses two entry points
        // that share a name. The pipeline names the entry it draws with, so the
        // name is part of the pass contract either way.
        let (name, stage, workgroup_size) = match stage {
            Stage::Compute => (entry.to_string(), naga::ShaderStage::Compute, workgroup),
            Stage::Vertex => (entry.to_string(), naga::ShaderStage::Vertex, [0, 0, 0]),
            Stage::Fragment => (entry.to_string(), naga::ShaderStage::Fragment, [0, 0, 0]),
        };
        self.module.entry_points.push(naga::EntryPoint {
            name,
            stage,
            early_depth_test: None,
            workgroup_size,
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });
        Ok(())
    }

    pub(crate) fn storage_global(
        &mut self,
        group: u32,
        binding: u32,
        elem: Elem,
        write: bool,
    ) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
        if let Some(handle) = self.existing_storage(group, binding, elem, write)? {
            return Ok(handle);
        }
        let base_ty = match elem {
            Elem::F32 => scalar(&mut self.module, naga::ScalarKind::Float),
            Elem::U32 => scalar(&mut self.module, naga::ScalarKind::Uint),
            Elem::AtomicU32 => self.module.types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Atomic(naga::Scalar {
                        kind: naga::ScalarKind::Uint,
                        width: SCALAR_WIDTH,
                    }),
                },
                naga::Span::default(),
            ),
            Elem::Vec4U32 => self.module.types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Vector {
                        size: naga::VectorSize::Quad,
                        scalar: naga::Scalar {
                            kind: naga::ScalarKind::Uint,
                            width: SCALAR_WIDTH,
                        },
                    },
                },
                naga::Span::default(),
            ),
        };
        let stride = if elem == Elem::Vec4U32 {
            UNIFORM_LANE_BYTES
        } else {
            WORD_BYTES
        };
        let array_ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Array {
                    base: base_ty,
                    size: naga::ArraySize::Dynamic,
                    stride,
                },
            },
            naga::Span::default(),
        );
        let name = match elem {
            Elem::AtomicU32 => format!("atomic_storage_{group}_{binding}_"),
            _ => format!("storage_{group}_{binding}_"),
        };
        let mut access = naga::StorageAccess::LOAD;
        if write {
            access |= naga::StorageAccess::STORE;
        }
        let handle = self.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(name),
                space: naga::AddressSpace::Storage { access },
                binding: Some(naga::ResourceBinding { group, binding }),
                ty: array_ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        );
        self.storage_binding
            .insert((group, binding), (elem, handle));
        Ok(handle)
    }

    /// Texture at `(group, binding)`. Depth and sampled images are different
    /// resources, and a storage variable already there would share the slot.
    pub(crate) fn texture(
        &mut self,
        group: u32,
        binding: u32,
        want: TextureKind,
    ) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
        self.texture_of(group, binding, want, naga::ImageDimension::D2, false)
    }

    /// Texture at `(group, binding)` with an explicit `dim`.
    ///
    /// A cube image is a different type, not a different kind: the class
    /// (`Sampled`/`Depth`) still decides what a sample returns, while `dim`
    /// decides how the coordinate is read.
    pub(crate) fn texture_of(
        &mut self,
        group: u32,
        binding: u32,
        want: TextureKind,
        dim: naga::ImageDimension,
        arrayed: bool,
    ) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
        if self.storage_binding.contains_key(&(group, binding)) {
            return Err(TranslateError::UnsupportedType(format!(
                "binding {group}:{binding} is both storage and a texture"
            )));
        }
        if let Some(existing) = self.textures.get(&(group, binding)) {
            if existing.kind != want || existing.dim != dim || existing.arrayed != arrayed {
                return Err(TranslateError::UnsupportedType(format!(
                    "texture binding {group}:{binding} is used as two kinds"
                )));
            }
            return Ok(existing.handle);
        }
        let ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Image {
                    dim,
                    arrayed,
                    class: image_class(want, self.sampling.multisampled_at(group, binding)),
                },
            },
            naga::Span::default(),
        );
        let handle = self.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(format!("tex_{group}_{binding}_")),
                space: naga::AddressSpace::Handle,
                binding: Some(naga::ResourceBinding { group, binding }),
                ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        );
        self.textures.insert(
            (group, binding),
            TextureBinding {
                kind: want,
                dim,
                arrayed,
                handle,
            },
        );
        Ok(handle)
    }

    /// A filtering `sampler` global at `(group, binding)`.
    ///
    /// Samplers carry no data the kernel can read, so nothing checks the
    /// address or filter modes here; the renderer's bind group owns them.
    /// Sampler at `(group, binding)`, filtering or comparison.
    pub(crate) fn sampler_of(
        &mut self,
        group: u32,
        binding: u32,
        comparison: bool,
    ) -> naga::Handle<naga::GlobalVariable> {
        if let Some(handle) = self.samplers.get(&(group, binding, comparison)) {
            return *handle;
        }
        let ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Sampler { comparison },
            },
            naga::Span::default(),
        );
        let handle = self.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(format!("sampler_{group}_{binding}_")),
                space: naga::AddressSpace::Handle,
                binding: Some(naga::ResourceBinding { group, binding }),
                ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        );
        self.samplers.insert((group, binding, comparison), handle);
        handle
    }
}
