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
//! layout; the mirror form states it exactly. naga/wgpu validate the
//! module (names, types, bindings); the byte-size agreement between a
//! Rust column type and its WGSL struct is asserted by
//! `#[derive(WgslStruct)]` mirrors and stays the author's contract for a
//! handwritten WGSL struct. `bool`/`BVecN` columns are rejected — a
//! storage array has no bool representation (use `u32`, or
//! `ornis_core::GpuBool` as a mirror).
//!
//! A field name becomes the lane's WGSL variable name, so spellings the
//! WGSL grammar reserves (`ornis_shader_lang::WGSL_RESERVED_WORDS` —
//! `ref`, `type`, `loop`, …) or the `__` identifier prefix are rejected
//! here, on the field's span.

use ornis_shader_lang::WGSL_RESERVED_WORDS;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{DeriveInput, Generics, Ident, Type, TypePath};

use crate::pack::{FieldInfo, PackInfo};

/// Whether the struct carries `#[pack(gpu)]` (the GPU half of
/// `#[derive(Pack)]`). Bare `#[pack]` and any other payload are rejected
/// with a spanned error — on this derive `pack(...)` has exactly one
/// spelling, so a typo fails here instead of silently staying CPU-only.
/// Bare `#[pack]` belongs to `#[derive(AutoPipeline)]`, which rejects
/// the `gpu` payload in turn, so `#[derive(AutoPipeline, Pack)]` cannot
/// serve both derives from one attribute (either spelling errors).
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
                     `#[pack(gpu)]` (bare `#[pack]` is `#[derive(AutoPipeline)]`'s \
                     field-type pack mode and has no meaning on this derive)",
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
/// spanned compile errors on the offending field names/types.
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
        let field_name = wgsl_field_name(field)?;
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

    let index_space = index_space_tokens(fields);
    let field_count = fields.len();
    Ok(quote! {
        impl #impl_generics ornis_core::PackGpu for #struct_name #ty_generics #where_clause {
            fn pack_gpu_lanes() -> &'static [ornis_core::PackGpuLane] {
                &[#(#lanes),*]
            }

            fn pack_gpu_upload(
                store: &ornis_core::SmartStore,
            ) -> Option<ornis_core::PackGpuUpload> {
                #index_space
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

/// The lane's WGSL variable name: the field ident minus a raw-ident
/// prefix, rejected when the WGSL grammar reserves that spelling or
/// the `__` identifier prefix.
fn wgsl_field_name(field: &FieldInfo) -> Result<String, syn::Error> {
    let raw = field.name.to_string();
    let name = raw.strip_prefix("r#").unwrap_or(&raw);
    if WGSL_RESERVED_WORDS.contains(&name) {
        return Err(syn::Error::new(
            field.name.span(),
            format!(
                "field name `{name}` is reserved in WGSL; a pack lane reuses its field name \
                 as the storage-variable name, so rename the field"
            ),
        ));
    }
    if name.starts_with("__") {
        return Err(syn::Error::new(
            field.name.span(),
            format!(
                "field name `{name}` starts with `__`, which WGSL reserves for its own \
                 identifiers; a pack lane reuses its field name as the storage-variable name, \
                 so rename the field"
            ),
        ));
    }
    Ok(name.to_string())
}

/// The shared index space of `pack_gpu_upload`: every column is one row
/// per entity of the first non-sparse lane — a leading `Option<T>`
/// field must not drop entities from the *other* columns — or, when
/// every field is sparse, the sorted union of all lanes (nothing else
/// anchors membership). An unregistered lane (`pack_register` never
/// ran) uploads nothing: `None`.
fn index_space_tokens(fields: &[FieldInfo]) -> TokenStream2 {
    if let Some(anchor) = fields.iter().find(|field| !field.optional) {
        let wrapper = &anchor.wrapper_name;
        quote! {
            let __entities: Vec<ornis_core::Entity> = match store.read_lane::<#wrapper>() {
                Some(__lane) => __lane.iter_entities().collect(),
                None => return None,
            };
        }
    } else {
        let wrappers = fields.iter().map(|field| &field.wrapper_name);
        quote! {
            let mut __entities: Vec<ornis_core::Entity> = Vec::new();
            #(
                match store.read_lane::<#wrappers>() {
                    Some(__lane) => __entities.extend(__lane.iter_entities()),
                    None => return None,
                }
            )*
            __entities.sort_by_key(|__e| __e.id());
            __entities.dedup_by_key(|__e| __e.id());
        }
    }
}

/// One column's aligned serialization: look every shared-index entity up
/// in its lane, missing entries upload the all-zero element, and an
/// empty store uploads one zeroed element so the lane buffer is never
/// zero-sized (wgpu rejects size-0 bindings; an empty scene must still
/// bind). Bytes are written straight into one `Vec<u8>` column — no
/// intermediate `Vec<T>`.
///
/// The `Pod` bound is asserted on the field's own span, so a column
/// type that is not plain-old-data fails with the field highlighted
/// instead of only with a `bytes_of` note inside generated code.
fn upload_field(field: &FieldInfo) -> TokenStream2 {
    let wrapper = &field.wrapper_name;
    let ty = &field.lane_ty;
    let pod_assert = quote_spanned! { ty.span()=>
        {
            fn __ornis_pack_column_pod<T: ornis_core::bytemuck::Pod>() {}
            let _ = __ornis_pack_column_pod::<#ty>;
        }
    };
    quote! {
        {
            #pod_assert
            let __lane = store.read_lane::<#wrapper>()?;
            let mut __col: Vec<u8> =
                Vec::with_capacity(__entities.len().max(1) * core::mem::size_of::<#ty>());
            for __e in &__entities {
                let __value: #ty = match __lane.get(*__e) {
                    Some(__w) => __w.0.clone(),
                    None => ornis_core::bytemuck::Zeroable::zeroed(),
                };
                __col.extend_from_slice(ornis_core::bytemuck::bytes_of(&__value));
            }
            if __col.is_empty() {
                __col.extend_from_slice(ornis_core::bytemuck::bytes_of(
                    &<#ty as ornis_core::bytemuck::Zeroable>::zeroed(),
                ));
            }
            __columns.push(__col);
        }
    }
}
