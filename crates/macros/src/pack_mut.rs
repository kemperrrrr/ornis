//! `PackMut` views and packed traversal for `#[derive(Pack)]`.
//!
//! [`generate_pack_mut_struct`] builds the guard-holding view struct with
//! per-field accessors (optional fields yield `Option<&mut _>`);
//! [`generate_for_each_packed`] stamps the `for_each_packed` traversal
//! over entities owning the first lane.

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Ident, WhereClause};

use crate::pack::{FieldInfo, LaneInfo};

pub(crate) fn generate_pack_get_mut(
    fields: &[FieldInfo],
    ty_generics: &syn::TypeGenerics,
    pack_mut_name: &Ident,
) -> TokenStream2 {
    let mut lane_guards = Vec::new();
    let mut guard_fields_init = Vec::new();

    for field in fields {
        let name = &field.name;
        let wrapper_name = &field.wrapper_name;
        let guard_name = format_ident!("_guard_{}", name);
        if field.optional {
            lane_guards.push(quote! {
                let #guard_name = store.write_lane::<#wrapper_name>()?;
            });
        } else {
            lane_guards.push(quote! {
                let #guard_name = store.write_lane::<#wrapper_name>()?;
                // Membership probe: a dense lane must hold `entity` or the
                // whole view is absent. Spelled with `?` (not
                // `if …is_none() { return None; }`) so the generated code
                // stays clean under `-D clippy::question_mark` in every
                // crate that derives `Pack`.
                #guard_name.get(entity)?;
            });
        }
        guard_fields_init.push(quote! {
            #guard_name,
        });
    }

    quote! {
        #(#lane_guards)*
        Some(#pack_mut_name #ty_generics {
            #(#guard_fields_init)*
            _entity: entity,
        })
    }
}

fn generate_for_each_packed(
    struct_name: &Ident,
    ty_generics: &syn::TypeGenerics,
    where_clause: Option<&WhereClause>,
    pack_mut_name: &Ident,
    lanes: &[LaneInfo],
) -> TokenStream2 {
    // Iterate over entities present in the first generated lane; for each,
    // acquire a `PackMut` and hand it to the caller. This is the packed
    // analogue of `for_each_entity!` — it traverses a `Pack`-derived struct
    // field-by-field (SoA) without materializing the whole struct.
    let first_wrapper = &lanes[0].wrapper_ty;
    quote! {
        impl #ty_generics #struct_name #ty_generics #where_clause {
            /// Iterate every entity that owns this packed component, giving
            /// the closure `&mut PackMut` (field accessors per lane).
            pub fn for_each_packed<F>(store: &mut ornis_core::SmartStore, mut f: F)
            where
                F: FnMut(&mut #pack_mut_name #ty_generics),
            {
                // Collect the entity set first (releasing the read guard),
                // then acquire `PackMut` per entity — a live read-lane guard
                // would otherwise conflict with `pack_get_mut`'s &mut store.
                let entities: Vec<ornis_core::Entity> = match store.read_lane::<#first_wrapper>() {
                    Some(__lane) => __lane.iter_entities().collect(),
                    None => return,
                };
                for __entity in entities {
                    if let Some(mut __pm) =
                        <Self as ornis_core::Pack>::pack_get_mut(store, __entity)
                    {
                        f(&mut __pm);
                    }
                }
            }
        }
    }
}

fn generate_pack_mut_struct(
    _struct_name: &Ident,
    fields: &[FieldInfo],
    ty_generics: &syn::TypeGenerics,
    where_clause: Option<&WhereClause>,
    pack_mut_name: &Ident,
) -> (TokenStream2, TokenStream2) {
    let guard_fields = fields.iter().map(|f| {
        let name = &f.name;
        let wrapper_name = &f.wrapper_name;
        let guard_name = format_ident!("_guard_{}", name);
        quote! { #guard_name: std::sync::RwLockWriteGuard<'a, ornis_core::ComponentStore<#wrapper_name>>, }
    });

    let accessor_methods = fields.iter().map(|f| {
        let name = &f.name;
        let inner_ty = &f.lane_ty;
        let guard_name = format_ident!("_guard_{}", name);
        if f.optional {
            quote! {
                pub fn #name(&mut self) -> Option<&mut #inner_ty> {
                    self.#guard_name.get_mut(self._entity).map(|__w| &mut __w.0)
                }
            }
        } else {
            quote! {
                pub fn #name(&mut self) -> &mut #inner_ty {
                    &mut self.#guard_name.get_mut(self._entity).unwrap().0
                }
            }
        }
    });

    let struct_def = quote! {
        pub struct #pack_mut_name<'a, #ty_generics> #where_clause {
            #(#guard_fields)*
            _entity: ornis_core::Entity,
        }
    };

    let methods = quote! {
        impl<'a, #ty_generics> #pack_mut_name<'a, #ty_generics> #where_clause {
            #(#accessor_methods)*
        }
    };

    (struct_def, methods)
}

/// Generates the `PackMut` view struct, its accessors, and the
/// `for_each_packed` traversal as one subgroup (keeps `derive` short).
pub(crate) fn generate_pack_mut_group(
    struct_name: &Ident,
    fields: &[FieldInfo],
    ty_generics: &syn::TypeGenerics,
    where_clause: Option<&WhereClause>,
    pack_mut_name: &Ident,
    lanes: &[LaneInfo],
) -> (TokenStream2, TokenStream2, TokenStream2) {
    let (pack_mut_struct, pack_mut_methods) = generate_pack_mut_struct(
        struct_name,
        fields,
        ty_generics,
        where_clause,
        pack_mut_name,
    );
    let for_each_packed =
        generate_for_each_packed(struct_name, ty_generics, where_clause, pack_mut_name, lanes);
    (pack_mut_struct, pack_mut_methods, for_each_packed)
}
