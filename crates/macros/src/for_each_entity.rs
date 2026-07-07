use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, FnArg, Ident, Pat, PatType, Token, Type, TypeReference,
};

struct ForEachInput {
    store: Ident,
    closure_args: Vec<ClosureArg>,
    body: proc_macro2::TokenStream,
}

struct ClosureArg {
    name: Ident,
    mutable: bool,
    ty: Type,
}

impl Parse for ForEachInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let store: Ident = input.parse()?;
        input.parse::<Token![,]>()?;

        let mut closure_args = Vec::new();
        if input.peek(Token![|]) {
            input.parse::<Token![|]>()?;
            while !input.peek(Token![|]) {
                let arg: FnArg = input.parse()?;
                match arg {
                    FnArg::Typed(PatType { pat, ty, .. }) => {
                        let name = match &*pat {
                            Pat::Ident(pi) => pi.ident.clone(),
                            _ => return Err(syn::Error::new_spanned(pat, "expected identifier")),
                        };
                        let (mutable, inner_ty) = match &*ty {
                            Type::Reference(TypeReference { elem, mutability, .. }) => {
                                (mutability.is_some(), *elem.clone())
                            }
                            _ => return Err(syn::Error::new_spanned(ty, "expected &T or &mut T")),
                        };
                        closure_args.push(ClosureArg { name, mutable, ty: inner_ty });
                    }
                    _ => return Err(syn::Error::new_spanned(arg, "expected typed argument")),
                }
                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                }
            }
            input.parse::<Token![|]>()?;
        }

        let body: proc_macro2::TokenStream = input.parse()?;
        Ok(ForEachInput { store, closure_args, body })
    }
}

pub fn for_each_entity(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ForEachInput);
    let store = &input.store;

    let read_lanes: Vec<_> = input.closure_args.iter()
        .filter(|a| !a.mutable)
        .map(|a| {
            let name = &a.name;
            let ty = &a.ty;
            let lane = format_ident!("__lane_{}", name);
            quote! {
                let #lane = #store.read_lane::<#ty>()
                    .expect(concat!("lane not registered for ", stringify!(#ty)));
            }
        })
        .collect();

    let write_lanes: Vec<_> = input.closure_args.iter()
        .filter(|a| a.mutable)
        .map(|a| {
            let name = &a.name;
            let ty = &a.ty;
            let lane = format_ident!("__lane_{}", name);
            quote! {
                let mut #lane = #store.write_lane::<#ty>()
                    .expect(concat!("lane not registered for ", stringify!(#ty)));
            }
        })
        .collect();

    let expanded = if input.closure_args.len() == 1 {
        // Single lane: iterate directly, no intersection needed
        let arg = &input.closure_args[0];
        let lane = format_ident!("__lane_{}", arg.name);
        let body = &input.body;
        let pat_name = &arg.name;
        if arg.mutable {
            quote! {
                {
                    #(#read_lanes)*
                    #(#write_lanes)*
                    for #pat_name in #lane.iter_mut() {
                        #body
                    }
                }
            }
        } else {
            quote! {
                {
                    #(#read_lanes)*
                    #(#write_lanes)*
                    for #pat_name in #lane.iter() {
                        #body
                    }
                }
            }
        }
    } else {
        // Multiple lanes: collect entities from bitset intersection, then iterate
        let first = &input.closure_args[0];
        let second = &input.closure_args[1];
        let lane0 = format_ident!("__lane_{}", first.name);
        let lane1 = format_ident!("__lane_{}", second.name);

        let collect_entities = quote! {
            let __entities: Vec<ornis_core::Entity> = {
                let __lane_ref = &*#lane0;
                let __lane_ref2 = &*#lane1;
                __lane_ref.iter_zip(__lane_ref2).map(|(e, _, _)| e).collect()
            };
        };

        let lookups: Vec<_> = input.closure_args.iter().map(|a| {
            let name = &a.name;
            let lane = format_ident!("__lane_{}", name);
            if a.mutable {
                quote! { let #name = #lane.get_mut(entity).unwrap(); }
            } else {
                quote! { let #name = #lane.get(entity).unwrap(); }
            }
        }).collect();

        let body = &input.body;
        quote! {
            {
                #(#read_lanes)*
                #(#write_lanes)*
                #collect_entities
                for entity in __entities {
                    #(#lookups)*
                    #body
                }
            }
        }
    };

    expanded.into()
}
