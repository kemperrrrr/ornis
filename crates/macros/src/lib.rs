mod auto_pipeline;
mod for_each_entity;
mod gpu_pipeline;
mod kernel;
mod pack;
mod pipeline_config;
mod smart_pipeline;
mod static_profile;
mod wgsl;

use proc_macro::TokenStream;

#[proc_macro_derive(AutoPipeline, attributes(pack))]
pub fn derive_auto_pipeline(input: TokenStream) -> TokenStream {
    auto_pipeline::derive(input)
}

#[proc_macro_derive(Pack, attributes(pack))]
pub fn derive_pack(input: TokenStream) -> TokenStream {
    pack::derive(input)
}

#[proc_macro]
pub fn for_each_entity(input: TokenStream) -> TokenStream {
    for_each_entity::for_each_entity(input)
}

#[proc_macro_attribute]
pub fn smart_pipeline(attr: TokenStream, item: TokenStream) -> TokenStream {
    smart_pipeline::attribute(attr, item)
}

#[proc_macro_derive(PipelineConfig, attributes(gpu, cpu, auto))]
pub fn derive_pipeline_config(input: TokenStream) -> TokenStream {
    pipeline_config::derive(input)
}

#[proc_macro_attribute]
pub fn gpu_pipeline(attr: TokenStream, item: TokenStream) -> TokenStream {
    gpu_pipeline::gpu_pipeline(attr, item)
}

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    kernel::kernel(attr, item)
}
