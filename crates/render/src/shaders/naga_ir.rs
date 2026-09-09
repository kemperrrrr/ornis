//! naga IR assembly: declarations built as IR, printed by naga itself.
//!
//! This is the rust-gpu idea at declaration scale: instead of formatting
//! WGSL text from Rust values, the Rust side builds a [`naga::Module`]
//! (types from [`WgslStruct`](ornis_macros::WgslStruct) derives via
//! `naga_add_type`, globals from [`Resource`](super::Resource) tables) and
//! naga's own WGSL writer prints it. No WGSL spelling lives here — member
//! names, offsets, address spaces and binding points are Rust data, and the
//! `vec4<f32>`/`<uniform>` spellings belong to the naga dependency.

use super::{OPENPBR_SLOTS, OPENPBR_WGSL_NAME, Resource, ResourceKind};

/// Insert the shared `OpenPBRMaterial` layout into the module, built from
/// the [`OPENPBR_SLOTS`] name list (20 `vec4` slots, 16 bytes each).
pub fn openpbr_type(module: &mut naga::Module) -> naga::Handle<naga::Type> {
    let vec4f = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Vector {
                size: naga::VectorSize::Quad,
                scalar: naga::Scalar {
                    kind: naga::ScalarKind::Float,
                    width: 4,
                },
            },
        },
        naga::Span::default(),
    );
    let members = OPENPBR_SLOTS
        .iter()
        .enumerate()
        .map(|(i, slot)| naga::StructMember {
            name: Some(slot.to_string()),
            ty: vec4f,
            binding: None,
            offset: (i * 16) as u32,
        })
        .collect();
    module.types.insert(
        naga::Type {
            name: Some(OPENPBR_WGSL_NAME.to_string()),
            inner: naga::TypeInner::Struct {
                members,
                span: (OPENPBR_SLOTS.len() * 16) as u32,
            },
        },
        naga::Span::default(),
    )
}

/// Image class for texture resources; `multisampled` threads the runtime
/// MSAA flag through, mirroring [`ResourceKind::bgl_ty`](super::ResourceKind::bgl_ty).
fn image_class(kind: &ResourceKind, multisampled: bool) -> naga::ImageClass {
    match kind {
        ResourceKind::TextureFloat => naga::ImageClass::Sampled {
            kind: naga::ScalarKind::Float,
            multi: multisampled,
        },
        ResourceKind::TextureUint => naga::ImageClass::Sampled {
            kind: naga::ScalarKind::Uint,
            multi: multisampled,
        },
        ResourceKind::TextureDepth => naga::ImageClass::Depth {
            multi: multisampled,
        },
        _ => panic!("image_class: buffer/sampler resource has no image class"),
    }
}

/// Insert one [`Resource`] as a module-global variable.
/// `ty` is the resource's naga type handle (from a derive or
/// [`openpbr_type`]); textures and samplers get fresh anonymous types.
/// Array resources wrap the passed struct handle using its own span as the
/// stride, so element names still come from the Rust side.
pub fn add_global(
    module: &mut naga::Module,
    ty: naga::Handle<naga::Type>,
    r: &Resource,
    multisampled: bool,
) -> naga::Handle<naga::GlobalVariable> {
    let (space, ty) = match &r.kind {
        ResourceKind::Uniform(_) | ResourceKind::StorageRead(_) | ResourceKind::StorageRw(_) => {
            (resource_space(&r.kind), ty)
        }
        ResourceKind::StorageReadArray(_) => {
            let stride = match module.types[ty].inner {
                naga::TypeInner::Struct { span, .. } => span,
                _ => panic!("StorageReadArray element must be a struct"),
            };
            let arr = module.types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Array {
                        base: ty,
                        size: naga::ArraySize::Dynamic,
                        stride,
                    },
                },
                naga::Span::default(),
            );
            (resource_space(&r.kind), arr)
        }
        ResourceKind::TextureFloat | ResourceKind::TextureUint | ResourceKind::TextureDepth => {
            let image = module.types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Image {
                        dim: naga::ImageDimension::D2,
                        arrayed: false,
                        class: image_class(&r.kind, multisampled),
                    },
                },
                naga::Span::default(),
            );
            (naga::AddressSpace::Handle, image)
        }
        ResourceKind::Sampler => {
            let sampler = module.types.insert(
                naga::Type {
                    name: None,
                    inner: naga::TypeInner::Sampler { comparison: false },
                },
                naga::Span::default(),
            );
            (naga::AddressSpace::Handle, sampler)
        }
    };
    module.global_variables.append(
        naga::GlobalVariable {
            name: Some(r.name.to_string()),
            space,
            binding: Some(naga::ResourceBinding {
                group: r.group,
                binding: r.binding,
            }),
            ty,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        naga::Span::default(),
    )
}

/// Address space for buffer resources, mirroring the WGSL `var<…>` prefix.
fn resource_space(kind: &ResourceKind) -> naga::AddressSpace {
    match kind {
        ResourceKind::Uniform(_) => naga::AddressSpace::Uniform,
        ResourceKind::StorageRead(_) | ResourceKind::StorageReadArray(_) => {
            naga::AddressSpace::Storage {
                access: naga::StorageAccess::LOAD,
            }
        }
        ResourceKind::StorageRw(_) => naga::AddressSpace::Storage {
            access: naga::StorageAccess::LOAD | naga::StorageAccess::STORE,
        },
        _ => panic!("resource_space: texture/sampler is a handle resource"),
    }
}

/// Validate the module and print it with naga's own WGSL writer.
///
/// Works around a naga 30 writer quirk: load-only storage globals print as
/// bare `var<storage>`, which WGSL reads as read-write. The resource table
/// knows the truth, so each `var<storage> <name>` line for a table resource
/// is rewritten to the explicit `var<storage, read>` form before return.
/// Pinned by the round-trip test below.
pub fn write_module(module: &naga::Module, table: &[Resource]) -> String {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .expect("assembled IR module must validate");
    let mut wgsl =
        naga::back::wgsl::write_string(module, &info, naga::back::wgsl::WriterFlags::empty())
            .expect("naga WGSL writer must print the module");
    for r in table {
        if matches!(
            r.kind,
            ResourceKind::StorageRead(_) | ResourceKind::StorageReadArray(_)
        ) {
            wgsl = wgsl.replace(
                &format!("var<storage> {}", r.name),
                &format!("var<storage, read> {}", r.name),
            );
        }
    }
    wgsl
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::{CameraUniform, GpuLight, LightingUniform};
    use crate::shaders::lighting_generated::LIGHTING_RESOURCES;

    /// The full lighting declaration block from IR: 4 types + 10 globals,
    /// printed by naga — no WGSL text authored here.
    #[test]
    fn lighting_decls_round_trip_through_ir() {
        let mut module = naga::Module::default();
        let cam = CameraUniform::naga_add_type(&mut module);
        let light = GpuLight::naga_add_type(&mut module);
        let lighting = LightingUniform::naga_add_type(&mut module);
        let mat = openpbr_type(&mut module);
        // `Light` is only referenced from inside `Lighting`, never global.
        assert_eq!(module.types[light].name.as_deref(), Some("Light"));
        // Buffer globals reuse the derived handles; textures/samplers build
        // their own anonymous types inside `add_global` (the passed handle
        // is ignored for them).
        for r in LIGHTING_RESOURCES {
            let ty = match r.name {
                "camera" => cam,
                "lighting" => lighting,
                "materials" => mat,
                _ => cam,
            };
            add_global(&mut module, ty, &r, false);
        }
        let wgsl = write_module(&module, &LIGHTING_RESOURCES);
        for probe in [
            "struct Camera",
            "struct Light",
            "struct Lighting",
            "struct OpenPBRMaterial",
            "var<uniform> camera: Camera",
            "var<storage, read> materials: array<OpenPBRMaterial>",
            "var lighting_sampler: sampler",
        ] {
            assert!(wgsl.contains(probe), "missing {probe}:\n{wgsl}");
        }
    }
}
