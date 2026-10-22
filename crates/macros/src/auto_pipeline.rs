//! Implementation of `#[derive(AutoPipeline)]`.
//!
//! Without `#[pack]`, the derive registers the component type in the
//! `SmartStore` and routes it to the CPU lane. With `#[pack]`, each named
//! field becomes its own component lane via generated `pack_register` /
//! `pack_insert` helpers. Only structs with named fields are supported.
//!
//! `#[pack]` is payload-free on this derive: a `pack(...)` list is
//! `#[derive(Pack)]`'s GPU opt-in spelling, and one attribute cannot
//! serve both derives — this derive rejects any payload with a spanned
//! error (Pack rejects the bare form in turn), so
//! `#[derive(AutoPipeline, Pack)]` cannot silently swap which
//! `pack_register` a call resolves to.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let has_pack = match pack_opt_in(&input) {
        Ok(has_pack) => has_pack,
        Err(err) => return err.to_compile_error().into(),
    };

    if has_pack {
        let fields = match &input.data {
            Data::Struct(s) => &s.fields,
            _ => {
                return syn::Error::new_spanned(name, "#[pack] only supported on structs")
                    .to_compile_error()
                    .into();
            }
        };
        let field_names: Vec<_> = match fields {
            Fields::Named(named) => named
                .named
                .iter()
                .filter_map(|f| f.ident.as_ref())
                .collect(),
            _ => {
                return syn::Error::new_spanned(fields, "#[pack] requires named fields")
                    .to_compile_error()
                    .into();
            }
        };
        let Fields::Named(named) = fields else {
            return syn::Error::new_spanned(fields, "#[pack] requires named fields")
                .to_compile_error()
                .into();
        };
        let field_tys: Vec<_> = named.named.iter().map(|f| &f.ty).collect();

        let expanded = quote! {
            impl #name {
                pub fn pack_register(store: &mut ornis_core::SmartStore) {
                    #( store.register::<#field_tys>(); )*
                }

                pub fn pack_insert(store: &mut ornis_core::SmartStore, entity: ornis_core::Entity, val: Self) {
                    #(
                        let __field = val.#field_names;
                        store.insert::<#field_tys>(entity, __field);
                    )*
                }
            }
        };
        expanded.into()
    } else {
        let expanded = quote! {
            impl ornis_core::AutoPipeline for #name {
                fn register(store: &mut ornis_core::SmartStore) {
                    store.register::<#name>();
                }
            }
            impl ornis_core::LaneTarget for #name {
                type Target = ornis_core::CpuLane;
            }
        };
        expanded.into()
    }
}

/// `#[pack]` presence, payload checked: this derive's pack mode is the
/// bare attribute only — a `pack(...)` list is `#[derive(Pack)]`'s
/// spelling (`#[pack(gpu)]`), and accepting it here would let one
/// attribute expand two different `pack_register`/`pack_insert`
/// pairs, where the inherent one silently shadows `Pack`'s.
fn pack_opt_in(input: &DeriveInput) -> Result<bool, syn::Error> {
    let mut has_pack = false;
    for attr in &input.attrs {
        if !attr.path().is_ident("pack") {
            continue;
        }
        has_pack = true;
        if !matches!(attr.meta, syn::Meta::Path(_)) {
            return Err(syn::Error::new_spanned(
                attr,
                "`#[derive(AutoPipeline)]` reads only bare `#[pack]` (its field-type pack \
                 mode); a `pack(...)` payload is `#[derive(Pack)]`'s GPU opt-in. On a struct \
                 deriving both, the attribute cannot do both jobs: keep bare `#[pack]` and \
                 drop Pack, or `#[pack(gpu)]` and drop AutoPipeline",
            ));
        }
    }
    Ok(has_pack)
}
