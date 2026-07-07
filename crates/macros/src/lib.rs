use proc_macro::TokenStream;

#[proc_macro_derive(AutoPipeline)]
pub fn derive_auto_pipeline(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}
