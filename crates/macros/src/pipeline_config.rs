use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input, Type, TypePath};

fn is_primitive_type(ty: &Type) -> bool {
    if let Type::Path(TypePath { path, .. }) = ty {
        let ident = path.get_ident().map(|i| i.to_string());
        matches!(ident.as_deref(), Some(s) if matches!(s, "f32" | "f64" | "i32" | "u32" | "i64" | "u64" | "bool" | "u8" | "i8" | "u16" | "i16"))
    } else {
        false
    }
}

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let has_gpu = input.attrs.iter().any(|a| a.path().is_ident("gpu"));
    let has_cpu = input.attrs.iter().any(|a| a.path().is_ident("cpu"));

    let target = if has_gpu {
        quote! { ornis_core::TargetDiscriminant::Gpu }
    } else if has_cpu {
        quote! { ornis_core::TargetDiscriminant::Cpu }
    } else {
        let all_primitive = match &input.data {
            Data::Struct(s) => match &s.fields {
                Fields::Named(named) => named.named.iter().all(|f| is_primitive_type(&f.ty)),
                _ => false,
            },
            _ => false,
        };
        if all_primitive {
            quote! { ornis_core::TargetDiscriminant::Gpu }
        } else {
            quote! { ornis_core::TargetDiscriminant::Cpu }
        }
    };

    let expanded = quote! {
        impl ornis_core::PipelineConfig for #name {
            fn lane_target() -> ornis_core::TargetDiscriminant {
                #target
            }
        }
    };
    expanded.into()
}
