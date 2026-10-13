//! Implementation of the `#[kernel]` attribute: validated Rust fn → WGSL.
//!
//! The function AST is validated first — no heap types, no dynamic
//! allocation, no recursion, no `while` loops, `match` nesting at most 2 —
//! then translated to WGSL via [`crate::wgsl`]. Violations surface as
//! compile errors pointing at the offending syntax.

use proc_macro::TokenStream;
use quote::quote;
use syn::{FnArg, ItemFn, parse_macro_input, visit, visit::Visit};

struct KernelValidator {
    fn_name: String,
    errors: Vec<syn::Error>,
    match_depth: usize,
}

impl KernelValidator {
    fn new(fn_name: &str) -> Self {
        Self {
            fn_name: fn_name.to_string(),
            errors: Vec::new(),
            match_depth: 0,
        }
    }

    fn validate(&mut self, func: &ItemFn) {
        self.check_params(&func.sig.inputs);
        self.check_return_type(&func.sig.output);
        self.visit_block(&func.block);
    }

    fn check_params(&mut self, inputs: &syn::punctuated::Punctuated<FnArg, syn::Token![,]>) {
        for arg in inputs {
            if let FnArg::Typed(pat_ty) = arg
                && let Some(msg) =
                    super::kernel_types::kernel_type_error(&pat_ty.ty, "type", &self.fn_name)
            {
                self.errors.push(syn::Error::new_spanned(&pat_ty.ty, msg));
            }
        }
    }

    fn check_return_type(&mut self, output: &syn::ReturnType) {
        if let syn::ReturnType::Type(_, ty) = output
            && let Some(msg) =
                super::kernel_types::kernel_type_error(ty, "return type", &self.fn_name)
        {
            self.errors.push(syn::Error::new_spanned(ty, msg));
        }
    }
}

impl<'ast> Visit<'ast> for KernelValidator {
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = node.func.as_ref() {
            let callee = p
                .path
                .get_ident()
                .map(|i| i.to_string())
                .unwrap_or_default();
            match callee.as_str() {
                "Box" | "Vec" | "String" | "Rc" | "Arc" => {
                    self.errors.push(syn::Error::new_spanned(
                        node,
                        format!(
                            "dynamic allocation via `{}` is not allowed in GPU kernel `{}`",
                            callee, self.fn_name
                        ),
                    ));
                }
                _ => {}
            }

            if callee == self.fn_name {
                self.errors.push(syn::Error::new_spanned(
                    node,
                    format!(
                        "recursion is not allowed in GPU kernel `{}`; all kernels must be flat",
                        self.fn_name
                    ),
                ));
            }
        }

        visit::visit_expr_call(self, node);
    }

    fn visit_expr_cast(&mut self, node: &'ast syn::ExprCast) {
        if crate::wgsl_lower::cast_target_ty(&node.ty).is_none() {
            let ty = node.ty.as_ref();
            self.errors.push(syn::Error::new_spanned(
                &node.ty,
                format!(
                    "cast to `{}` is not supported in GPU kernel `{}`; supported targets: f32, u32, i32",
                    quote!(#ty),
                    self.fn_name
                ),
            ));
        }
        visit::visit_expr_cast(self, node);
    }

    /// Closures have no WGSL spelling: the lowerer used to smuggle the
    /// `compile_error!` text into the WGSL string, failing only at naga.
    fn visit_expr_closure(&mut self, node: &'ast syn::ExprClosure) {
        self.errors.push(syn::Error::new_spanned(
            node,
            format!(
                "closures are not supported in GPU kernel `{}`; WGSL has no closures, inline the computation instead",
                self.fn_name
            ),
        ));
        visit::visit_expr_closure(self, node);
    }

    /// Scalar-namespaced paths (`u8::MAX`, `f32::consts::PI`) have no
    /// WGSL spelling: `u8::MAX as u32` used to lower to `u32(u8::MAX)`
    /// and fail only at naga. Use a plain `f32`, `u32` or `i32` literal.
    fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
        if super::kernel_types::is_scalar_namespaced(&node.path, node.qself.as_ref()) {
            let path = &node.path;
            self.errors.push(syn::Error::new_spanned(
                &node.path,
                format!(
                    "path `{}` is not supported in GPU kernel `{}`; scalar associated constants have no WGSL spelling, use a plain `f32`, `u32` or `i32` literal instead",
                    quote!(#path),
                    self.fn_name
                ),
            ));
        }
        visit::visit_expr_path(self, node);
    }

    fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
        self.errors.push(syn::Error::new_spanned(
            node,
            format!(
                "`while` loops are not allowed in GPU kernel `{}`; use a fixed `for` loop or avoid unbounded iteration",
                self.fn_name
            ),
        ));
        visit::visit_expr_while(self, node);
    }

    /// `let` annotations must name kernel-supported types: an
    /// annotation like `let v: u8` would otherwise mistranslate while
    /// CPU `eval` wraps.
    fn visit_local(&mut self, node: &'ast syn::Local) {
        if let syn::Pat::Type(pt) = &node.pat
            && let Some(msg) =
                super::kernel_types::kernel_type_error(&pt.ty, "let annotation", &self.fn_name)
        {
            self.errors.push(syn::Error::new_spanned(&pt.ty, msg));
        }
        visit::visit_local(self, node);
    }

    /// Integer literals keep their values only with a 32-bit spelling:
    /// narrow/wide suffixes (`100u8`, `1u64`) would mistranslate while
    /// CPU `eval` wraps or holds 64 bits.
    fn visit_lit_int(&mut self, node: &'ast syn::LitInt) {
        if !matches!(node.suffix(), "" | "u32" | "i32") {
            self.errors.push(syn::Error::new_spanned(
                node,
                format!(
                    "integer suffix `{}` is not supported in GPU kernel `{}`; use plain, `u32` or `i32` literals",
                    node.suffix(),
                    self.fn_name
                ),
            ));
        }
    }

    /// Same as above for floats: `1.0f64` would silently narrow to `f32`.
    fn visit_lit_float(&mut self, node: &'ast syn::LitFloat) {
        if !matches!(node.suffix(), "" | "f32") {
            self.errors.push(syn::Error::new_spanned(
                node,
                format!(
                    "float suffix `{}` is not supported in GPU kernel `{}`; use plain or `f32` literals",
                    node.suffix(),
                    self.fn_name
                ),
            ));
        }
    }

    fn visit_expr_match(&mut self, node: &'ast syn::ExprMatch) {
        self.match_depth += 1;
        if self.match_depth > 2 {
            self.errors.push(syn::Error::new_spanned(
                node,
                format!(
                    "deeply nested `match` (depth > 2) is not allowed in GPU kernel `{}`; simplify the branching",
                    self.fn_name
                ),
            ));
        }
        visit::visit_expr_match(self, node);
        self.match_depth -= 1;
    }
}

/// Validate the optional dispatch id: `#[kernel]` takes no attribute or a
/// single integer literal; anything else (e.g. a string) is rejected
/// instead of being silently ignored.
fn validate_kernel_args(args: TokenStream) -> syn::Result<()> {
    if args.is_empty() {
        return Ok(());
    }
    let literal: syn::LitInt = syn::parse2(args.into())?;
    let _dispatch_id: u64 = literal.base10_parse()?;
    Ok(())
}

pub fn kernel(args: TokenStream, input: TokenStream) -> TokenStream {
    if let Err(error) = validate_kernel_args(args) {
        return TokenStream::from(error.to_compile_error());
    }
    let func = parse_macro_input!(input as ItemFn);
    let fn_name = &func.sig.ident;

    let mut validator = KernelValidator::new(&fn_name.to_string());
    validator.validate(&func);

    if !validator.errors.is_empty() {
        let compile_errors = validator.errors.iter().map(|e| e.to_compile_error());
        return TokenStream::from(quote! {
            #(#compile_errors)*
        });
    }

    // Generate WGSL function source string
    if let Err(e) = ornis_shader_lang::check_builtin_arity(&func) {
        return TokenStream::from(e.to_compile_error());
    }
    let wgsl_fn_src = crate::wgsl::wgsl_fn_source(&func);

    // Keep the compute shader path for backward compatibility
    let wgsl_compute = crate::wgsl::wgsl_source_from_fn(&func);

    // Generate the Rust function body as WGSL for the compute version
    let inputs = &func.sig.inputs;
    let output = &func.sig.output;
    let body = &func.block;

    // One read-only storage entry per input plus one read-write output
    // entry — see `wgsl::storage_bgl_entries`. The pipeline layout
    // references this; callers build their bind groups against it.
    let bgl_entries = crate::wgsl::storage_bgl_entries(func.sig.inputs.len());

    let expanded = quote! {
        // The user-facing `#[kernel] fn` carries the docs; the generated
        // helper module (eval/wgsl_source/...) is intentionally bare and
        // hidden from rustdoc.
        #[doc(hidden)]
        pub mod #fn_name {
            // The user-facing `#[kernel] fn` carries the docs; the generated
            // helper surface (eval/wgsl_source/...) is intentionally bare and
            // hidden from rustdoc.
            #[allow(missing_docs)]
            #[allow(non_snake_case)]
            pub fn eval ( #inputs ) #output {
                use super::*;
                #body
            }

            #[allow(dead_code)]
            pub fn label() -> &'static str {
                stringify!(#fn_name)
            }

            #[allow(dead_code)]
            pub fn wgsl_source() -> &'static str {
                #wgsl_fn_src
            }

            #[allow(dead_code)]
            pub fn wgsl_compute_source() -> &'static str {
                #wgsl_compute
            }

            #[allow(dead_code)]
            pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(stringify!(#fn_name)),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl_compute_source())),
                })
            }

            #[allow(dead_code)]
            pub fn create_bind_group_layout(
                device: &wgpu::Device,
            ) -> wgpu::BindGroupLayout {
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(stringify!(#fn_name)),
                    entries: &[ #(#bgl_entries),* ],
                })
            }

            #[allow(dead_code)]
            pub fn create_pipeline(
                device: &wgpu::Device,
            ) -> wgpu::ComputePipeline {
                let shader = create_shader_module(device);
                let bind_group_layout = create_bind_group_layout(device);
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(stringify!(#fn_name)),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(stringify!(#fn_name)),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            }
        }
    };

    TokenStream::from(expanded)
}

#[cfg(test)]
mod tests {
    use super::KernelValidator;
    use crate::kernel_types::is_scalar_namespaced;

    fn scalar_namespaced(source: &str) -> bool {
        let expr: syn::Expr = syn::parse_str(source).expect("test path must parse");
        let syn::Expr::Path(path) = expr else {
            panic!("{source} must parse as a path");
        };
        is_scalar_namespaced(&path.path, path.qself.as_ref())
    }

    #[test]
    fn scalar_const_paths_are_namespaced() {
        for path in ["u8::MAX", "u32::MAX", "f32::consts::PI", "<u8>::MAX"] {
            assert!(scalar_namespaced(path), "{path}");
        }
    }

    #[test]
    fn locals_and_shader_paths_are_not_namespaced() {
        for path in [
            "x",
            "PI",
            "Vec3::ZERO",
            "glam::Vec3::ZERO",
            "Vec3::new",
            "glam::Vec3::splat",
            "normalize",
        ] {
            assert!(!scalar_namespaced(path), "{path}");
        }
    }

    #[test]
    fn validator_rejects_closures_and_const_paths() {
        let func: syn::ItemFn = syn::parse_quote! {
            fn kernel_fn(x: f32) -> f32 {
                let f = |y: u8| y;
                x + u8::MAX as u32 as f32 + f(0) as f32
            }
        };
        let mut validator = KernelValidator::new("kernel_fn");
        validator.validate(&func);
        assert_eq!(validator.errors.len(), 2);
        let messages: Vec<String> = validator.errors.iter().map(|e| e.to_string()).collect();
        assert!(
            messages
                .iter()
                .any(|m| m.contains("closures are not supported")),
            "{messages:?}"
        );
        assert!(
            messages
                .iter()
                .any(|m| m.contains("scalar associated constants")),
            "{messages:?}"
        );
    }
}
