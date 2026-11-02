//! `#[shader::compute]`, `#[shader::vertex]`, and `#[shader::fragment]`.
//!
//! The macro keeps a shader entry point an ordinary Rust function and adds
//! the wasm-facing wrapper: `#[unsafe(no_mangle)] pub extern "C"`. It also
//! records the stage and dispatch shape in the `ornis_gpu.entries` custom
//! section, so the offline translator reads `@workgroup_size` from the module
//! instead of from a second literal.
//!
//! Signature checks run at macro expansion time: unsupported parameter or
//! return types, a missing or zero `workgroup_size`, and generic or `async`
//! functions fail with a diagnostic on the offending item.

use proc_macro::TokenStream;
use proc_macro2::Literal;
use quote::format_ident;
use quote::quote;
use syn::{ItemFn, parse_macro_input};

use crate::shader_entry_sig::check_signature;
use crate::shader_entry_sig::parse_args;

/// `#[shader::compute(workgroup_size = N)]` — compute entry point.
pub fn compute(args: TokenStream, item: TokenStream) -> TokenStream {
    entry(Stage::Compute, args, item)
}

/// `#[shader::vertex]` — vertex entry point.
pub fn vertex(args: TokenStream, item: TokenStream) -> TokenStream {
    entry(Stage::Vertex, args, item)
}

/// `#[shader::fragment]` — fragment entry point.
pub fn fragment(args: TokenStream, item: TokenStream) -> TokenStream {
    entry(Stage::Fragment, args, item)
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub(crate) enum Stage {
    Vertex = 0,
    Fragment = 1,
    Compute = 2,
}

const STAGE_TAGS: [&str; Stage::Compute as usize + 1] = ["vertex", "fragment", "compute"];

impl Stage {
    const fn tag(self) -> &'static str {
        STAGE_TAGS[self as usize]
    }

    pub(crate) const fn requires_workgroup(self) -> bool {
        matches!(self, Self::Compute)
    }
}

fn entry(stage: Stage, args: TokenStream, item: TokenStream) -> TokenStream {
    let mut func = parse_macro_input!(item as ItemFn);
    let workgroup = match parse_args(stage, args) {
        Ok(workgroup) => workgroup,
        Err(err) => return err.to_compile_error().into(),
    };
    if let Err(err) = check_signature(stage, &func) {
        return err.to_compile_error().into();
    }
    let export = export_name(&func);
    let record = format!(
        "{},{},{},{},{}\n",
        stage.tag(),
        export,
        workgroup[0],
        workgroup[1],
        workgroup[2],
    );
    let static_ident = format_ident!("__ORNIS_ENTRY_{}", export.to_uppercase());
    let len = record.len();
    let bytes = Literal::byte_string(record.as_bytes());
    func.vis = syn::parse_quote!(pub);
    func.sig.abi = syn::parse_quote!(extern "C");
    quote! {
        #[unsafe(no_mangle)]
        #func

        #[cfg(target_arch = "wasm32")]
        #[used]
        #[unsafe(link_section = "ornis_gpu.entries")]
        static #static_ident: [u8; #len] = *#bytes;
    }
    .into()
}

fn export_name(func: &ItemFn) -> String {
    let name = func.sig.ident.to_string();
    name.strip_prefix("r#").unwrap_or(&name).to_string()
}
