use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let has_pack = input.attrs.iter().any(|a| a.path().is_ident("pack"));

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
        let field_tys: Vec<_> = match fields {
            Fields::Named(named) => named.named.iter().map(|f| &f.ty).collect(),
            _ => unreachable!(),
        };

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
