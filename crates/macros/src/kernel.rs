use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, visit, visit::Visit, FnArg, ItemFn, Type};

fn is_forbidden_type(ty: &Type) -> bool {
    match ty {
        Type::Path(tp) => {
            let last = tp.path.segments.last().map(|s| s.ident.to_string());
            matches!(
                last.as_deref(),
                Some(
                    "Vec" | "String" | "Box" | "Rc" | "Arc" | "HashMap" | "HashSet"
                        | "VecDeque" | "LinkedList" | "BTreeMap" | "BTreeSet"
                )
            )
        }
        _ => false,
    }
}

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
            if let FnArg::Typed(pat_ty) = arg {
                if is_forbidden_type(&pat_ty.ty) {
                    self.errors.push(syn::Error::new_spanned(
                        &pat_ty.ty,
                        format!(
                            "type `{}` is not allowed in GPU kernel `{}`; use f32, i32, u32, or glam vectors instead",
                            quote!(#pat_ty.ty),
                            self.fn_name
                        ),
                    ));
                }
            }
        }
    }

    fn check_return_type(&mut self, output: &syn::ReturnType) {
        if let syn::ReturnType::Type(_, ty) = output {
            if is_forbidden_type(ty) {
                self.errors.push(syn::Error::new_spanned(
                    ty,
                    format!(
                        "return type `{}` is not allowed in GPU kernel `{}`",
                        quote!(#ty),
                        self.fn_name
                    ),
                ));
            }
        }
    }
}

impl<'ast> Visit<'ast> for KernelValidator {
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = node.func.as_ref() {
            let callee = p.path.get_ident().map(|i| i.to_string()).unwrap_or_default();
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

pub fn kernel(args: TokenStream, input: TokenStream) -> TokenStream {
    let _attr_args = args;
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
    let wgsl_fn_src = crate::wgsl::wgsl_fn_source(&func);

    // Keep the compute shader path for backward compatibility
    let wgsl_compute = crate::wgsl::wgsl_source_from_fn(&func);

    // Generate the Rust function body as WGSL for the compute version
    let inputs = &func.sig.inputs;
    let output = &func.sig.output;
    let body = &func.block;

    let expanded = quote! {
        pub mod #fn_name {
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
            pub fn create_pipeline(
                device: &wgpu::Device,
            ) -> wgpu::ComputePipeline {
                let shader = create_shader_module(device);
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(stringify!(#fn_name)),
                    bind_group_layouts: &[],
                    push_constant_ranges: &[],
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
