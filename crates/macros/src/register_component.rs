//! `#[derive(RegisterComponent)]` — F0 derive sugar for the component registry.
//!
//! Generates `impl ornis_core::registry::RegisterComponent for T` with a
//! `COMPONENT_NAME` associated constant plus a `FIELDS` schema (one entry
//! per named field, for structs with named fields; empty otherwise).
//! The registry's typed helper `register_component::<T>()` then registers
//! `T` under that name without repeating the string at every call site.
//!
//! # Attributes
//!
//! - `#[component(name = "Custom")]` overrides the default (type ident).
//! - Without the attribute the name defaults to `stringify!(Type)`.
//! - Field schema always follows the Rust field idents (serde renames are
//!   not tracked — keep `serde(rename)` off registered components or
//!   register the schema manually via `register_with_fields`).

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

fn component_name_from_attrs(input: &DeriveInput) -> syn::Result<Option<String>> {
    for attr in &input.attrs {
        if !attr.path().is_ident("component") {
            continue;
        }
        let mut name: Option<String> = None;
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                let value = meta.value()?;
                let lit: syn::LitStr = value.parse()?;
                name = Some(lit.value());
                Ok(())
            } else {
                Err(meta.error("unsupported component attribute, expected `name = \"...\"`"))
            }
        })?;
        if let Some(n) = name {
            return Ok(Some(n));
        }
        return Err(syn::Error::new_spanned(
            attr,
            "component attribute requires `name = \"...\"`",
        ));
    }
    Ok(None)
}

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let name_literal = match component_name_from_attrs(&input) {
        Ok(Some(n)) => n,
        Ok(None) => ident.to_string(),
        Err(e) => return e.to_compile_error().into(),
    };
    let name_lit = syn::LitStr::new(&name_literal, ident.span());

    // One schema entry per named field (raw idents unprefixed to match
    // serde's JSON keys); anything else gets the trait's empty default.
    let field_entries = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => named
                .named
                .iter()
                .filter_map(|field| {
                    let ident = field.ident.as_ref()?;
                    let raw = ident.to_string();
                    let name = raw
                        .strip_prefix("r#")
                        .unwrap_or(&raw)
                        .trim_end_matches('#')
                        .to_string();
                    let name_lit = syn::LitStr::new(&name, ident.span());
                    let ty = &field.ty;
                    Some(quote! {
                        ::ornis_core::FieldMeta {
                            name: #name_lit,
                            path: #name_lit,
                            type_name: stringify!(#ty),
                            writable: true,
                        }
                    })
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    };

    let expanded = quote! {
        impl #impl_generics ::ornis_core::RegisterComponent for #ident #ty_generics #where_clause {
            const COMPONENT_NAME: &'static str = #name_lit;
            const FIELDS: &'static [::ornis_core::FieldMeta] = &[ #(#field_entries),* ];
        }
    };
    expanded.into()
}
