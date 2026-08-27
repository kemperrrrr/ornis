use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    FnArg, Ident, Pat, PatType, Token, Type, TypeReference,
    parse::{Parse, ParseStream},
    parse_macro_input,
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
        let closure_args = parse_closure_args(input)?;
        let body: proc_macro2::TokenStream = input.parse()?;
        Ok(ForEachInput {
            store,
            closure_args,
            body,
        })
    }
}

/// Parse the optional `|a: &T, mut b: &mut U|` lane-closure arguments.
fn parse_closure_args(input: ParseStream) -> syn::Result<Vec<ClosureArg>> {
    let mut closure_args = Vec::new();
    if !input.peek(Token![|]) {
        return Ok(closure_args);
    }
    input.parse::<Token![|]>()?;
    while !input.peek(Token![|]) {
        let arg: FnArg = input.parse()?;
        closure_args.push(closure_arg(&arg)?);
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }
    }
    input.parse::<Token![|]>()?;
    Ok(closure_args)
}

/// One typed closure argument: `name: &T` (read) or `name: &mut T` (write).
fn closure_arg(arg: &FnArg) -> syn::Result<ClosureArg> {
    let FnArg::Typed(PatType { pat, ty, .. }) = arg else {
        return Err(syn::Error::new_spanned(arg, "expected typed argument"));
    };
    let name = match &**pat {
        Pat::Ident(pi) => pi.ident.clone(),
        _ => return Err(syn::Error::new_spanned(pat, "expected identifier")),
    };
    let (mutable, inner_ty) = match &**ty {
        Type::Reference(TypeReference {
            elem, mutability, ..
        }) => (mutability.is_some(), *elem.clone()),
        _ => return Err(syn::Error::new_spanned(ty, "expected &T or &mut T")),
    };
    Ok(ClosureArg {
        name,
        mutable,
        ty: inner_ty,
    })
}

pub fn for_each_entity(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ForEachInput);
    let expanded = if input.closure_args.len() == 1 {
        expand_single_lane(&input)
    } else {
        expand_multi_lane(&input)
    };
    expanded.into()
}

/// Guard acquisitions for every declared lane: read guards for `&T`,
/// write guards for `&mut T`.
fn lane_guards(
    input: &ForEachInput,
) -> (Vec<proc_macro2::TokenStream>, Vec<proc_macro2::TokenStream>) {
    let store = &input.store;

    let read_lanes: Vec<_> = input
        .closure_args
        .iter()
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

    let write_lanes: Vec<_> = input
        .closure_args
        .iter()
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

    (read_lanes, write_lanes)
}

/// Single lane: iterate directly, no intersection needed.
fn expand_single_lane(input: &ForEachInput) -> proc_macro2::TokenStream {
    let (read_lanes, write_lanes) = lane_guards(input);

    let arg = &input.closure_args[0];
    let lane = format_ident!("__lane_{}", arg.name);
    let pat_name = &arg.name;
    let body = &input.body;
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
}

/// Multiple lanes: collect entities present in ALL lanes — zip the first
/// two, filter by the rest — then iterate. Without the filter, the
/// per-lane lookups below would panic on an entity missing from
/// lane 3..N (partial ownership; audit §2.3, backlog #18).
fn expand_multi_lane(input: &ForEachInput) -> proc_macro2::TokenStream {
    let (read_lanes, write_lanes) = lane_guards(input);

    let first = &input.closure_args[0];
    let second = &input.closure_args[1];
    let lane0 = format_ident!("__lane_{}", first.name);
    let lane1 = format_ident!("__lane_{}", second.name);

    let extra_lanes: Vec<_> = input
        .closure_args
        .iter()
        .skip(2)
        .map(|a| format_ident!("__lane_{}", a.name))
        .collect();

    let collect_entities = quote! {
        let __entities: Vec<ornis_core::Entity> = {
            let __lane_ref = &*#lane0;
            let __lane_ref2 = &*#lane1;
            // `contains` re-checks the generation, so lanes 3..N apply
            // exactly the same staleness semantics as the zipped two.
            __lane_ref
                .iter_zip(__lane_ref2)
                .map(|(e, _, _)| e)
                #(.filter(|e| #extra_lanes.contains(*e)))*
                .collect()
        };
    };

    let lookups: Vec<_> = input
        .closure_args
        .iter()
        .map(|a| {
            let name = &a.name;
            let lane = format_ident!("__lane_{}", name);
            if a.mutable {
                quote! { let #name = #lane.get_mut(entity).unwrap(); }
            } else {
                quote! { let #name = #lane.get(entity).unwrap(); }
            }
        })
        .collect();

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
}
