//! Uniform globals, aggregate type construction, and the module build entry points.

use super::SCALAR_WIDTH;
use super::UNIFORM_LANE_BYTES;
use super::VEC2_LANES;
use super::VEC3_LANES;
use super::VEC4_LANES;
use super::WORD_2;
use super::WORD_3;
use super::WORD_BYTES;
use super::builders::Builders;
use super::emit::scalar;
use super::helpers::missing_struct;
use crate::error::TranslateError;

impl Builders {
    /// Uniform `array<scalar, len>` wrapped as `{ a: array<lane, len> }`.
    pub(crate) fn uniform_global(
        &mut self,
        group: u32,
        binding: u32,
        len: u32,
    ) -> naga::Handle<naga::GlobalVariable> {
        if let Some(handle) = self.uniform_cache.get(&(group, binding)) {
            return *handle;
        }
        let scalar_ty = scalar(&mut self.module, naga::ScalarKind::Float);
        let lane_ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Struct {
                    members: vec![
                        naga::StructMember {
                            name: Some("v".to_string()),
                            ty: scalar_ty,
                            binding: None,
                            offset: 0,
                        },
                        naga::StructMember {
                            name: Some("p1".to_string()),
                            ty: scalar_ty,
                            binding: None,
                            offset: WORD_BYTES,
                        },
                        naga::StructMember {
                            name: Some("p2".to_string()),
                            ty: scalar_ty,
                            binding: None,
                            offset: WORD_2,
                        },
                        naga::StructMember {
                            name: Some("p3".to_string()),
                            ty: scalar_ty,
                            binding: None,
                            offset: WORD_3,
                        },
                    ],
                    span: UNIFORM_LANE_BYTES,
                },
            },
            naga::Span::default(),
        );
        let array_ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Array {
                    base: lane_ty,
                    size: naga::ArraySize::Constant(
                        std::num::NonZeroU32::new(len).unwrap_or(std::num::NonZeroU32::MIN),
                    ),
                    stride: UNIFORM_LANE_BYTES,
                },
            },
            naga::Span::default(),
        );
        let outer = self.module.types.insert(
            naga::Type {
                name: Some(format!("U{len}")),
                inner: naga::TypeInner::Struct {
                    members: vec![naga::StructMember {
                        name: Some("a".to_string()),
                        ty: array_ty,
                        binding: None,
                        offset: 0,
                    }],
                    span: UNIFORM_LANE_BYTES * len,
                },
            },
            naga::Span::default(),
        );
        let handle = self.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(format!("uniform_{group}_{binding}_")),
                space: naga::AddressSpace::Uniform,
                binding: Some(naga::ResourceBinding { group, binding }),
                ty: outer,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        );
        self.uniform_cache.insert((group, binding), handle);
        handle
    }

    /// Uniform struct `SID` at `(group, binding)`, laid out with
    /// `place_fields` in uniform mode.
    // qual:allow(iosp) reason: "the binding is created only after the struct type has been laid out, and a repeat call must reuse it"
    pub(crate) fn uniform_struct_global(
        &mut self,
        group: u32,
        binding: u32,
        sid: u32,
    ) -> Result<naga::Handle<naga::GlobalVariable>, TranslateError> {
        if let Some(handle) = self.uniform_struct_cache.get(&(group, binding, sid)) {
            return Ok(*handle);
        }
        let (struct_ty, _, _) = self.struct_type(sid)?;
        let handle = self.module.global_variables.append(
            naga::GlobalVariable {
                name: Some(format!("uniform_struct_{group}_{binding}_")),
                space: naga::AddressSpace::Uniform,
                binding: Some(naga::ResourceBinding { group, binding }),
                ty: struct_ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::default(),
        );
        self.uniform_struct_cache
            .insert((group, binding, sid), handle);
        Ok(handle)
    }

    /// Type, byte size and alignment of uniform struct `sid`.
    // qual:allow(iosp) reason: "laying a uniform struct out is the check of each field tag together with the naga type it becomes"
    pub(crate) fn struct_type(
        &mut self,
        sid: u32,
    ) -> Result<(naga::Handle<naga::Type>, u32, u32), TranslateError> {
        if let Some(found) = self.struct_types.get(&sid) {
            return Ok(*found);
        }
        if self.struct_stack.contains(&sid) {
            return Err(TranslateError::UnsupportedType(format!(
                "uniform struct {sid} contains itself"
            )));
        }
        let tags = self
            .structs
            .get(sid as usize)
            .ok_or_else(|| missing_struct(sid))?
            .clone();
        if tags.is_empty() {
            return Err(TranslateError::UnsupportedType(
                "uniform struct has no fields".to_string(),
            ));
        }
        self.struct_stack.push(sid);
        let mut members = Vec::new();
        let mut specs = Vec::new();
        let mut cursor = 0usize;
        while cursor < tags.len() {
            let index = members.len();
            let (ty, layout, width) = self.field_type(&tags, cursor)?;
            specs.push(layout);
            members.push(naga::StructMember {
                name: Some(format!("f{index}")),
                ty,
                binding: None,
                offset: 0,
            });
            cursor += width;
        }
        let mut offsets = vec![0u32; members.len()];
        let span = ornis_gpu_abi::place_fields(&specs, true, &mut offsets);
        for (member, offset) in members.iter_mut().zip(&offsets) {
            member.offset = *offset;
        }
        let ty = self.module.types.insert(
            naga::Type {
                name: Some(format!("UniformStruct{sid}")),
                inner: naga::TypeInner::Struct { members, span },
            },
            naga::Span::default(),
        );
        self.struct_stack.pop();
        let align = ornis_gpu_abi::UNIFORM_STRUCT_ALIGN;
        let found = (ty, span, align);
        self.struct_types.insert(sid, found);
        Ok(found)
    }

    /// Type, layout and tag width of the field whose leading tag sits at
    /// `cursor`.
    ///
    /// Scalars and vectors are one tag byte; a nested struct is the tag plus
    /// its id; an array is [`TYPE_TAG_ARRAY`](ornis_gpu_abi::TYPE_TAG_ARRAY),
    /// the element tag and the length.
    // qual:allow(iosp) reason: "one field of a uniform struct: the tag decides the naga type, the layout it occupies and how many tag bytes it consumed"
    pub(crate) fn field_type(
        &mut self,
        tags: &[u8],
        cursor: usize,
    ) -> Result<(naga::Handle<naga::Type>, ornis_gpu_abi::FieldLayout, usize), TranslateError> {
        let tag = tags[cursor];
        if tag != ornis_gpu_abi::TYPE_TAG_ARRAY {
            let (ty, size, align) = self.tag_type(tag)?;
            return Ok((ty, ornis_gpu_abi::FieldLayout { size, align }, 1));
        }
        let element_tag = *tags.get(cursor + 1).ok_or_else(|| {
            TranslateError::UnsupportedType("array field has no element tag".to_string())
        })?;
        let length = u32::from(*tags.get(cursor + 2).ok_or_else(|| {
            TranslateError::UnsupportedType("array field has no length".to_string())
        })?);
        let (element, size, align) = self.tag_type(element_tag)?;
        // WGSL cannot spell a uniform array's padded stride: the writer emits
        // `array<E, N>` and re-parsing it recomputes the stride from `E`
        // alone, so an element narrower than the 16-byte uniform alignment
        // (`array<f32, N>`, `array<vec2<f32>, N>`) would only surface as
        // naga's own "array stride 4 is not a multiple of the required
        // alignment 16" — raw, and without naming the struct. The element
        // stride, not the rounded uniform stride, decides: `vec3` is 12 bytes
        // but rounds to 16 and is legal.
        let element_stride = ornis_gpu_abi::storage_stride(size, align);
        if !element_stride.is_multiple_of(ornis_gpu_abi::UNIFORM_STRUCT_ALIGN) {
            return Err(TranslateError::UnsupportedType(format!(
                "a uniform array of {size}-byte elements has a {element_stride}-byte stride, but \
                 WGSL requires a multiple of {} — wrap the element in a padded struct instead of a \
                 plain array",
                ornis_gpu_abi::UNIFORM_STRUCT_ALIGN
            )));
        }
        let stride = ornis_gpu_abi::uniform_stride(size, align);
        let ty = self.module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Array {
                    base: element,
                    size: naga::ArraySize::Constant(
                        std::num::NonZeroU32::new(length).unwrap_or(std::num::NonZeroU32::MIN),
                    ),
                    stride,
                },
            },
            naga::Span::default(),
        );
        let layout =
            ornis_gpu_abi::array_layout(ornis_gpu_abi::FieldLayout { size, align }, length);
        Ok((ty, layout, 3))
    }

    // qual:allow(iosp) reason: "a field tag either selects a scalar naga type or recurses into the nested struct it names"
    pub(crate) fn tag_type(
        &mut self,
        tag: u8,
    ) -> Result<(naga::Handle<naga::Type>, u32, u32), TranslateError> {
        if tag >= ornis_gpu_abi::TYPE_TAG_NESTED {
            return self.struct_type(u32::from(tag - ornis_gpu_abi::TYPE_TAG_NESTED));
        }
        use ornis_gpu_abi::{
            F32_ALIGN, F32_SIZE, TYPE_TAG_F32, TYPE_TAG_I32, TYPE_TAG_U32, TYPE_TAG_VEC2,
            TYPE_TAG_VEC3, TYPE_TAG_VEC4, VEC2_ALIGN, VEC2_SIZE, VEC3_ALIGN, VEC3_SIZE, VEC4_ALIGN,
            VEC4_SIZE,
        };
        let (kind, lanes, size, align) = match tag {
            TYPE_TAG_F32 => (naga::ScalarKind::Float, 1, F32_SIZE, F32_ALIGN),
            TYPE_TAG_U32 => (naga::ScalarKind::Uint, 1, F32_SIZE, F32_ALIGN),
            TYPE_TAG_I32 => (naga::ScalarKind::Sint, 1, F32_SIZE, F32_ALIGN),
            TYPE_TAG_VEC2 => (naga::ScalarKind::Float, VEC2_LANES, VEC2_SIZE, VEC2_ALIGN),
            TYPE_TAG_VEC3 => (naga::ScalarKind::Float, VEC3_LANES, VEC3_SIZE, VEC3_ALIGN),
            TYPE_TAG_VEC4 => (naga::ScalarKind::Float, VEC4_LANES, VEC4_SIZE, VEC4_ALIGN),
            _ => {
                return Err(TranslateError::UnsupportedType(format!(
                    "unsupported uniform field tag {tag}"
                )));
            }
        };
        let inner = if lanes == 1 {
            naga::TypeInner::Scalar(naga::Scalar {
                kind,
                width: SCALAR_WIDTH,
            })
        } else {
            let size = if lanes == VEC2_LANES {
                naga::VectorSize::Bi
            } else if lanes == VEC3_LANES {
                naga::VectorSize::Tri
            } else {
                naga::VectorSize::Quad
            };
            naga::TypeInner::Vector {
                size,
                scalar: naga::Scalar {
                    kind,
                    width: SCALAR_WIDTH,
                },
            }
        };
        let ty = self
            .module
            .types
            .insert(naga::Type { name: None, inner }, naga::Span::default());
        Ok((ty, size, align))
    }
}
