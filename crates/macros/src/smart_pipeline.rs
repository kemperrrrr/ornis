use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

pub fn attribute(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let vis = &input.vis;
    let sig = &input.sig;
    let block = &input.block;

    let expanded = quote! {
        #vis #sig {
            ornis_core::pipeline_enter();
            let _result = (|| #block)();
            ornis_core::pipeline_exit();
            _result
        }
    };

    expanded.into()
}
