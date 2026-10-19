//! Generated per-field lane types for `#[derive(Pack)]`: each field gets a
//! `repr(transparent)` wrapper stored in its own `ComponentStore`, plus the
//! `ornis_core::PackLane` impl that gathers and scatters that field.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::Visibility;

use crate::pack::LaneInfo;

/// Lane wrappers share the bundle's visibility: a private bundle keeps its
/// lanes private, so their `PackLane::Field` never leaks a private field type.
pub(crate) fn generate_wrapper_defs(vis: &Visibility, lanes: &[LaneInfo]) -> TokenStream2 {
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
    if lane.optional {
        generate_sparse_impl(lane)
    } else {
        generate_dense_impl(lane)
    }
}

fn generate_sparse_impl(lane: &LaneInfo) -> TokenStream2 {
    let wrapper = &lane.wrapper_ty;
    let field_ty = &lane.field_ty;
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

            fn gather_at(lane: &ornis_core::ComponentStore<Self>, index: usize) -> #field_ty {
                Some(lane.data[index].0.clone())
            }

            fn scatter_at(
                lane: &mut ornis_core::ComponentStore<Self>,
                index: usize,
                value: #field_ty,
            ) {
                match value {
                    Some(v) => lane.data[index] = Self(v),
                    None => {
                        let entity = lane.entities[index];
                        lane.remove(entity);
                    }
                }
            }
        }
    }
}

fn generate_dense_impl(lane: &LaneInfo) -> TokenStream2 {
    let wrapper = &lane.wrapper_ty;
    let field_ty = &lane.field_ty;
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

            fn gather_at(lane: &ornis_core::ComponentStore<Self>, index: usize) -> #field_ty {
                lane.data[index].0.clone()
            }

            fn scatter_at(
                lane: &mut ornis_core::ComponentStore<Self>,
                index: usize,
                value: #field_ty,
            ) {
                lane.data[index] = Self(value);
            }
        }

        impl ornis_core::DensePackLane for #wrapper {
            #[inline(always)]
            fn inner(&self) -> &Self::Field {
                &self.0
            }

            #[inline(always)]
            fn inner_mut(&mut self) -> &mut Self::Field {
                &mut self.0
            }
        }
    }
}
