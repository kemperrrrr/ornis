//! `#[pack(gpu)]` half of `#[derive(Pack)]`: SoA columns as storage lanes.
//!
//! The derive sees the struct definition, so it can describe each field
//! column for the GPU without the caller re-declaring it in
//! `#[gpu_pipeline]`: [`ornis_core::PackGpu`] impl emitting one
//! [`ornis_core::PackGpuLane`] per field (declaration order) plus the
//! aligned serializer [`ornis_core::PackGpu::pack_gpu_upload`].
//!
//! Element spelling is deliberately pass-through: `f32`/`u32`/`i32` map
//! to themselves, every other field type keeps its own name as a struct
//! mirror (`glam::Vec3` → `Vec3`, `Option<Entity>` → `Entity`). WGSL
//! storage arrays of `vecN<f32>` have padded strides that glam's packed
//! columns do not match, so a guessed registry spelling would lie about
//! layout; the mirror form states it exactly and naga/wgpu validate it.
//! `bool`/`BVecN` columns are rejected — a storage array has no bool
//! representation (use `u32`, or `ornis_core::GpuBool` as a mirror).

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{DeriveInput, Generics, Ident, Type, TypePath};

use crate::pack::{FieldInfo, PackInfo};

/// Whether the struct carries `#[pack(gpu)]` (the GPU half of
/// `#[derive(Pack)]`). Bare `#[pack]` and any other payload are rejected
/// with a spanned error — on this derive `pack(...)` has exactly one
/// spelling, so a typo fails here instead of silently staying CPU-only.
pub(crate) fn pack_gpu_enabled(input: &DeriveInput) -> Result<bool, syn::Error> {
    let mut enabled = false;
    for attr in &input.attrs {
        if !attr.path().is_ident("pack") {
            continue;
        }
        match &attr.meta {
            syn::Meta::Path(_) => {
                return Err(syn::Error::new_spanned(
                    attr,
                    "#[derive(Pack)] reads `#[pack(...)]` as GPU opt-in; write \
                     `#[pack(gpu)]` (bare `#[pack]` has no meaning on this derive)",
                ));
            }
            syn::Meta::List(list) => {
                if list.tokens.to_string().replace(' ', "") == "gpu" {
                    enabled = true;
                } else {
                    return Err(syn::Error::new_spanned(
                        attr,
                        "only `#[pack(gpu)]` is supported on `#[derive(Pack)]`; \
                         it opts the bundle's columns into GPU storage lanes",
                    ));
                }
            }
            syn::Meta::NameValue(_) => {
                return Err(syn::Error::new_spanned(attr, "expected `#[pack(gpu)]`"));
            }
        }
    }
    Ok(enabled)
}

/// The `#[pack(gpu)]` half of one derive expansion: empty tokens when
/// the struct does not opt in, the `impl PackGpu` when it does, a
/// spanned error for a bad attribute or an unsupported column type.
pub(crate) fn pack_gpu_tokens(
    input: &DeriveInput,
    info: &PackInfo,
) -> Result<TokenStream2, syn::Error> {
    if !pack_gpu_enabled(input)? {
        return Ok(TokenStream2::new());
    }
    generate_pack_gpu(&info.struct_name, &info.generics, &info.fields)
}

/// GPU element spelling of one column. `Err` carries the user-facing
/// message for a type that has no storage-column form.
pub(crate) fn gpu_elem_wgsl(ty: &Type) -> Result<String, String> {
    let Type::Path(TypePath {
        path, qself: None, ..
    }) = ty
    else {
        return Err(
            "pack GPU columns must be named types (`f32`, a struct mirror, …); \
             arrays, tuples and references have no storage-column form"
                .to_string(),
        );
    };
    let Some(seg) = path.segments.last() else {
        return Err("pack GPU column has an empty type path".to_string());
    };
    let name = seg.ident.to_string();
    match name.as_str() {
        "bool" | "BVec2" | "BVec3" | "BVec4" => Err(format!(
            "`{name}` has no storage-buffer representation; use a u32 flag column \
             (see ornis_core::GpuBool for the mirror newtype)"
        )),
        "Option" => Err(
            "`Option` inside a pack GPU column has no storage layout; the outer \
             `Option<T>` field is already the sparse-column form"
                .to_string(),
        ),
        _ => Ok(name),
    }
}

/// `impl PackGpu` for a `#[pack(gpu)]` bundle: lane descriptors in
/// field order plus the aligned column upload. Errors surface as
/// spanned compile errors on the offending field types.
pub(crate) fn generate_pack_gpu(
    struct_name: &Ident,
    generics: &Generics,
    fields: &[FieldInfo],
) -> Result<TokenStream2, syn::Error> {
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    if fields.is_empty() {
        return Err(syn::Error::new(
            struct_name.span(),
            "`#[pack(gpu)]` needs at least one field: a pack with no columns \
             has no GPU lanes to bind",
        ));
    }

    let mut lanes = Vec::with_capacity(fields.len());
    let mut uploads = Vec::with_capacity(fields.len());
    for field in fields {
        let elem = gpu_elem_wgsl(&field.lane_ty)
            .map_err(|msg| syn::Error::new_spanned(&field.lane_ty, msg))?;
        let field_name = field.name.to_string();
        let sparse = field.optional;
        lanes.push(quote! {
            ornis_core::PackGpuLane {
                field: #field_name,
                wgsl_elem: #elem,
                sparse: #sparse,
            }
        });
        uploads.push(upload_field(field));
    }

    let first_wrapper = &fields[0].wrapper_name;
    let field_count = fields.len();
    Ok(quote! {
        impl #impl_generics ornis_core::PackGpu for #struct_name #ty_generics #where_clause {
            fn pack_gpu_lanes() -> &'static [ornis_core::PackGpuLane] {
                &[#(#lanes),*]
            }

            fn pack_gpu_upload(
                store: &ornis_core::SmartStore,
            ) -> Option<ornis_core::PackGpuUpload> {
                let __entities: Vec<ornis_core::Entity> = match store.read_lane::<#first_wrapper>() {
                    Some(__lane) => __lane.iter_entities().collect(),
                    None => return None,
                };
                let mut __columns: Vec<Vec<u8>> = Vec::with_capacity(#field_count);
                #(#uploads)*
                Some(ornis_core::PackGpuUpload {
                    len: __entities.len(),
                    columns: __columns,
                })
            }
        }
    })
}

/// One column's aligned serialization: look every shared-index entity up
/// in its lane, missing entries upload the all-zero element.
fn upload_field(field: &FieldInfo) -> TokenStream2 {
    let wrapper = &field.wrapper_name;
    let ty = &field.lane_ty;
    quote! {
        {
            let __lane = store.read_lane::<#wrapper>()?;
            let mut __col: Vec<#ty> = Vec::with_capacity(__entities.len());
            for __e in &__entities {
                match __lane.get(*__e) {
                    Some(__w) => __col.push(__w.0.clone()),
                    None => __col.push(ornis_core::bytemuck::Zeroable::zeroed()),
                }
            }
            __columns.push(ornis_core::bytemuck::cast_slice::<#ty, u8>(&__col).to_vec());
        }
    }
}
