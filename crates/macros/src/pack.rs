//! Implementation of `#[derive(Pack)]`: struct → per-field component lanes.
//!
//! Each named field gets a unique wrapper type acting as its lane marker.
//! `Option<T>` fields are sparse: the lane stores `T` and only entities
//! with `Some` own a lane entry (`pack_get` maps a missing entry back to
//! `None`, so entity links like `target: Option<Entity>` naturally skip
//! unlinked entities). The derive emits the registration/insertion code
//! that splits the struct across the store's lanes. Only structs with
//! named fields are supported.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};

use crate::pack_mut::{generate_pack_get_mut, generate_pack_mut_group};
use syn::{
    Data, DeriveInput, Fields, FieldsNamed, Generics, Ident, Type, TypePath, Visibility,
    parse_macro_input,
};

#[derive(Clone)]
pub(crate) struct FieldInfo {
    pub(crate) name: Ident,
    pub(crate) lane_ty: Type,
    /// Declared field type (`Option<T>` for sparse fields).
    pub(crate) field_ty: Type,
    // Unique wrapper type for this field's lane
    pub(crate) wrapper_name: Ident,
    // `Option<T>` fields are sparse lanes of `T` (missing entry ⇔ `None`).
    pub(crate) optional: bool,
}

/// Name of the lane wrapper for `field` of `struct_name`. Derived from the
/// struct and field names only (no field index), so `#[smart_system]` can
/// name a lane from the bundle type path and the field identifier.
pub(crate) fn lane_wrapper_name(struct_name: &Ident, field: &Ident) -> Ident {
    format_ident!("{}__{}__PackLane", struct_name, field)
}

#[derive(Clone)]
pub(crate) struct LaneInfo {
    pub(crate) wrapper_ty: Type,
    pub(crate) inner_ty: Type,
    pub(crate) field_ty: Type,
    pub(crate) optional: bool,
}

struct PackInfo {
    struct_name: Ident,
    vis: Visibility,
    generics: Generics,
    fields: Vec<FieldInfo>,
    lanes: Vec<LaneInfo>,
}

fn extract_field_info(struct_name: &Ident, fields: &FieldsNamed) -> Vec<FieldInfo> {
    fields
        .named
        .iter()
        .filter_map(|f| {
            let name = f.ident.as_ref()?.clone();
            let ty = f.ty.clone();
            let (lane_ty, optional) = unpack_option(ty.clone());
            let wrapper_name = lane_wrapper_name(struct_name, &name);
            Some(FieldInfo {
                name,
                lane_ty,
                field_ty: ty,
                wrapper_name,
                optional,
            })
        })
        .collect()
}

/// Splits `Option<T>` into `(T, true)`; any other type maps to itself
/// with `false` (dense lane, every bundled entity owns an entry).
fn unpack_option(ty: Type) -> (Type, bool) {
    match &ty {
        Type::Path(TypePath { path, .. }) => {
            let mut path = path.clone();
            if let Some(last) = path.segments.last_mut()
                && last.ident == "Option"
                && let syn::PathArguments::AngleBracketed(args) = &last.arguments
                && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
            {
                return (inner.clone(), true);
            }
            (ty, false)
        }
        _ => (ty, false),
    }
}

fn generate_lanes(fields: &[FieldInfo]) -> Vec<LaneInfo> {
    fields
        .iter()
        .map(|field| {
            let wrapper_name = &field.wrapper_name;
            let wrapper_ty: Type = syn::parse_quote!(#wrapper_name);
            LaneInfo {
                wrapper_ty,
                inner_ty: field.lane_ty.clone(),
                field_ty: field.field_ty.clone(),
                optional: field.optional,
            }
        })
        .collect()
}

fn extract_pack_info(input: &DeriveInput) -> Result<PackInfo, syn::Error> {
    let struct_name = input.ident.clone();
    let vis = input.vis.clone();
    let generics = input.generics.clone();

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => named,
            _ => {
                return Err(syn::Error::new_spanned(
                    &s.fields,
                    "#[derive(Pack)] requires named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                &struct_name,
                "#[derive(Pack)] only works on structs",
            ));
        }
    };

    let field_infos = extract_field_info(&struct_name, fields);
    if field_infos.is_empty() {
        return Err(syn::Error::new_spanned(
            &struct_name,
            "#[derive(Pack)] requires at least one field",
        ));
    }

    let lanes = generate_lanes(&field_infos);

    Ok(PackInfo {
        struct_name,
        vis,
        generics,
        fields: field_infos,
        lanes,
    })
}

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let info = match extract_pack_info(&input) {
        Ok(info) => info,
        Err(e) => return e.to_compile_error().into(),
    };

    let struct_name = &info.struct_name;
    let (impl_generics, ty_generics, where_clause) = info.generics.split_for_impl();

    let pack_mut_name = format_ident!("PackMut{}", struct_name);

    // Generate wrapper type definitions
    let wrapper_defs = generate_wrapper_defs(&info.vis, &info.lanes);
    let pack_register = generate_pack_register(&info.lanes);
    let pack_insert = generate_pack_insert(&info.fields);
    let pack_get = generate_pack_get(struct_name, &info.fields, &ty_generics);
    let pack_get_mut = generate_pack_get_mut(&info.fields, &ty_generics, &pack_mut_name);
    let (pack_mut_struct, pack_mut_methods, for_each_packed) = generate_pack_mut_group(
        struct_name,
        &info.fields,
        &ty_generics,
        where_clause,
        &pack_mut_name,
        &info.lanes,
    );
    let pack_put = generate_pack_put(&info.fields);
    let pack_entities = generate_pack_entities(&info.lanes);
    let pack_lane_ids = generate_pack_lane_ids(&info.lanes);

    let expanded = quote! {
        #wrapper_defs

        #pack_mut_struct

        #pack_mut_methods

        impl #impl_generics ornis_core::Pack for #struct_name #ty_generics #where_clause {
            type PackMut<'a> = #pack_mut_name<'a, #ty_generics> #where_clause
            where
                Self: 'a;

            fn pack_register(store: &mut ornis_core::SmartStore) {
                #pack_register
            }

            fn pack_insert(&self, store: &mut ornis_core::SmartStore, entity: ornis_core::Entity) {
                #pack_insert
            }

            fn pack_get(store: &ornis_core::SmartStore, entity: ornis_core::Entity) -> Option<Self> {
                #pack_get
            }

            fn pack_get_mut<'a>(store: &'a mut ornis_core::SmartStore, entity: ornis_core::Entity) -> Option<Self::PackMut<'a>> {
                #pack_get_mut
            }

            fn pack_put(&self, store: &ornis_core::SmartStore, entity: ornis_core::Entity) {
                #pack_put
            }

            fn pack_entities(store: &ornis_core::SmartStore) -> Vec<ornis_core::Entity> {
                #pack_entities
            }

            fn pack_lane_ids() -> Vec<std::any::TypeId> {
                #pack_lane_ids
            }
        }

        #for_each_packed
    };

    expanded.into()
}

/// Lane wrappers share the bundle's visibility: a private bundle keeps its
/// lanes private, so their `PackLane::Field` never leaks a private field type.
fn generate_wrapper_defs(vis: &Visibility, lanes: &[LaneInfo]) -> TokenStream2 {
    let defs = lanes.iter().map(|lane| {
        let wrapper_name = &lane.wrapper_ty;
        let inner_ty = &lane.inner_ty;
        let lane_impl = generate_lane_impl(lane);
        quote! {
            #[derive(Clone, Debug, PartialEq)]
            #[repr(transparent)]
            // The name is generated as Struct__field__PackLane — deliberately not CamelCase.
            #[allow(non_camel_case_types)]
            #vis struct #wrapper_name(pub #inner_ty);

            #lane_impl
        }
    });
    quote! { #(#defs)* }
}

/// [`ornis_core::PackLane`] for one wrapper: dense fields index the lane
/// directly, sparse (`Option`) fields insert on `Some` and remove on `None`.
fn generate_lane_impl(lane: &LaneInfo) -> TokenStream2 {
    let wrapper = &lane.wrapper_ty;
    let field_ty = &lane.field_ty;
    if lane.optional {
        quote! {
            impl ornis_core::PackLane for #wrapper {
                type Field = #field_ty;
                const SPARSE: bool = true;

                fn gather(
                    lane: &ornis_core::ComponentStore<Self>,
                    index: Option<usize>,
                ) -> Option<#field_ty> {
                    Some(index.map(|i| lane.data[i].0.clone()))
                }

                fn scatter(
                    lane: &mut ornis_core::ComponentStore<Self>,
                    entity: ornis_core::Entity,
                    index: Option<usize>,
                    value: #field_ty,
                ) {
                    match (index, value) {
                        (Some(i), Some(v)) => lane.data[i] = Self(v),
                        (None, Some(v)) => lane.insert(entity, Self(v)),
                        (Some(_), None) => {
                            lane.remove(entity);
                        }
                        (None, None) => {}
                    }
                }
            }
        }
    } else {
        quote! {
            impl ornis_core::PackLane for #wrapper {
                type Field = #field_ty;
                const SPARSE: bool = false;

                fn gather(
                    lane: &ornis_core::ComponentStore<Self>,
                    index: Option<usize>,
                ) -> Option<#field_ty> {
                    index.map(|i| lane.data[i].0.clone())
                }

                fn scatter(
                    lane: &mut ornis_core::ComponentStore<Self>,
                    entity: ornis_core::Entity,
                    index: Option<usize>,
                    value: #field_ty,
                ) {
                    match index {
                        Some(i) => lane.data[i] = Self(value),
                        None => lane.insert(entity, Self(value)),
                    }
                }
            }
        }
    }
}

fn generate_pack_register(lanes: &[LaneInfo]) -> TokenStream2 {
    let lane_registers = lanes.iter().map(|lane| {
        let wrapper_ty = &lane.wrapper_ty;
        quote! { store.register::<#wrapper_ty>(); }
    });

    quote! { #(#lane_registers)* }
}

fn generate_pack_insert(fields: &[FieldInfo]) -> TokenStream2 {
    let inserts = fields
        .iter()
        .map(|f| insert_field_stmt(f, quote! { store }));

    quote! { #(#inserts)* }
}

/// Per-field insert statement into `lane` (`store` for eager insert,
/// `__lane` for guarded scatter). Sparse (`Option`) fields only write
/// present values. Shared by insert and scatter so the optional-guard
/// shape lives in exactly one place.
fn insert_field_stmt(field: &FieldInfo, lane: TokenStream2) -> TokenStream2 {
    let name = &field.name;
    let wrapper_name = &field.wrapper_name;
    if field.optional {
        quote! {
            if let Some(__inner) = self.#name.clone() {
                #lane.insert(entity, #wrapper_name(__inner));
            }
        }
    } else {
        quote! {
            #lane.insert(entity, #wrapper_name(self.#name.clone()));
        }
    }
}

fn generate_pack_get(
    struct_name: &Ident,
    fields: &[FieldInfo],
    ty_generics: &syn::TypeGenerics,
) -> TokenStream2 {
    let field_reads = fields.iter().map(|f| {
        let name = &f.name;
        let wrapper_name = &f.wrapper_name;
        if f.optional {
            quote! {
                let #name = store.read_lane::<#wrapper_name>()?
                    .get(entity)
                    .map(|__w| __w.0.clone());
            }
        } else {
            quote! {
                let #name = store.read_lane::<#wrapper_name>()?.get(entity)?.0.clone();
            }
        }
    });

    let field_names = fields.iter().map(|f| &f.name);

    quote! {
        #(#field_reads)*
        Some(#struct_name #ty_generics { #(#field_names),* })
    }
}

fn generate_pack_put(fields: &[FieldInfo]) -> TokenStream2 {
    // Scatter one field at a time: each `write_lane` guard is dropped
    // before the next is taken, so lanes are never locked nested. (Nested
    // multi-lane capture does exist in `pack_get_mut`, which holds one
    // guard per field in declaration order; a canonical cross-pack lane
    // order is a v2b item, tracked in PLAN.md.)
    let puts = fields.iter().map(|f| {
        let wrapper_name = &f.wrapper_name;
        let insert = insert_field_stmt(f, quote! { __lane });
        quote! {
            if let Some(mut __lane) = store.write_lane::<#wrapper_name>() {
                #insert
            }
        }
    });

    quote! { #(#puts)* }
}

fn generate_pack_entities(lanes: &[LaneInfo]) -> TokenStream2 {
    let first_wrapper = &lanes[0].wrapper_ty;
    quote! {
        match store.read_lane::<#first_wrapper>() {
            Some(__lane) => __lane.iter_entities().collect(),
            None => Vec::new(),
        }
    }
}

fn generate_pack_lane_ids(lanes: &[LaneInfo]) -> TokenStream2 {
    let ids = lanes.iter().map(|lane| {
        let wrapper = &lane.wrapper_ty;
        quote! { std::any::TypeId::of::<#wrapper>() }
    });
    quote! { vec![#(#ids),*] }
}
