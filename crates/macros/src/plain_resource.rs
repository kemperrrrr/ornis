//! Implementation of `#[derive(PlainResource)]`: structural
//! "no interior mutability" check for `#[smart_system]` resources.
//!
//! The derive emits `impl ornis_core::PlainResource` with one
//! `FieldTy: PlainResource` bound per field (every variant for enums).
//! Interior-mutable types (`Mutex`, `RwLock`, atomics, `Cell`/`RefCell`)
//! never implement `PlainResource`, so a struct holding one fails with a
//! missing-bound error pointing at the offending field type —
//! transitively, since a nested plain struct must itself derive (or
//! implement) the trait. Generic fields work through the same bounds
//! (`Vec<T>` is plain when `T` is plain, via the std impls).

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Type, parse_macro_input};

fn field_types_of(fields: &Fields) -> Vec<&Type> {
    match fields {
        Fields::Named(named) => named.named.iter().map(|f| &f.ty).collect(),
        Fields::Unnamed(unnamed) => unnamed.unnamed.iter().map(|f| &f.ty).collect(),
        Fields::Unit => Vec::new(),
    }
}

fn collect_field_types(input: &DeriveInput) -> Result<Vec<Type>, syn::Error> {
    match &input.data {
        Data::Struct(data) => Ok(field_types_of(&data.fields).into_iter().cloned().collect()),
        Data::Enum(data) => {
            let mut types = Vec::new();
            for variant in &data.variants {
                types.extend(field_types_of(&variant.fields).into_iter().cloned());
            }
            Ok(types)
        }
        Data::Union(_) => Err(syn::Error::new_spanned(
            &input.ident,
            "#[derive(PlainResource)] does not support unions",
        )),
    }
}

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let field_types = match collect_field_types(&input) {
        Ok(types) => types,
        Err(e) => return e.to_compile_error().into(),
    };

    let name = &input.ident;
    let mut generics = input.generics.clone();
    if !field_types.is_empty() {
        let where_clause = generics.make_where_clause();
        for ty in &field_types {
            where_clause
                .predicates
                .push(syn::parse_quote!(#ty: ornis_core::PlainResource));
        }
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics ornis_core::PlainResource for #name #ty_generics #where_clause {}
    };

    expanded.into()
}
