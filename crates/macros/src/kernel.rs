//! Implementation of the `#[kernel]` attribute: validated Rust fn → WGSL.
//!
//! The function AST is validated first — no heap types, no dynamic
//! allocation, no recursion, no `while` loops, `match` nesting at most 2 —
//! then translated to WGSL via [`crate::wgsl`]. Violations surface as
//! compile errors pointing at the offending syntax.

use proc_macro::TokenStream;
use quote::quote;
use syn::{FnArg, ItemFn, parse_macro_input, visit::Visit};

mod visit;

struct KernelValidator {
    fn_name: String,
    errors: Vec<syn::Error>,
    match_depth: usize,
    /// Whether the expression currently being visited sits in a position
    /// where a value-position `match` lowers (a `let` initializer or a
    /// whole statement). Set just before descending into those nodes.
    match_ok: bool,
    /// Whether that host needs the value form (whose arms must be
    /// single expressions) rather than the statement form.
    match_value: bool,
    /// Whether the body mentions `Some`/`None` at all — checked against
    /// the signature/annotation Option types after the walk.
    uses_option_ctors: bool,
    /// Whether the signature or any `let` annotation mentions
    /// `Option<T>` — the only contexts the lowerer can resolve.
    has_option_type: bool,
}

impl KernelValidator {
    fn new(fn_name: &str) -> Self {
        Self {
            fn_name: fn_name.to_string(),
            errors: Vec::new(),
            match_depth: 0,
            match_ok: false,
            match_value: false,
            uses_option_ctors: false,
            has_option_type: false,
        }
    }

    fn validate(&mut self, func: &ItemFn) {
        self.check_params(&func.sig.inputs);
        self.check_return_type(&func.sig.output);
        self.visit_block(&func.block);
        self.check_option_context();
    }

    fn check_params(&mut self, inputs: &syn::punctuated::Punctuated<FnArg, syn::Token![,]>) {
        for arg in inputs {
            if let FnArg::Typed(pat_ty) = arg {
                if super::kernel_types::is_option_type(&pat_ty.ty) {
                    self.has_option_type = true;
                }
                if let Some(msg) =
                    super::kernel_types::kernel_type_error(&pat_ty.ty, "type", &self.fn_name)
                {
                    self.errors.push(syn::Error::new_spanned(&pat_ty.ty, msg));
                }
            }
        }
    }

    fn check_return_type(&mut self, output: &syn::ReturnType) {
        if let syn::ReturnType::Type(_, ty) = output {
            if super::kernel_types::is_option_type(ty) {
                self.has_option_type = true;
            }
            if let Some(msg) =
                super::kernel_types::kernel_type_error(ty, "return type", &self.fn_name)
            {
                self.errors.push(syn::Error::new_spanned(ty, msg));
            }
        }
    }

    /// `Some(...)`/`None` have no type of their own: the lowerer needs
    /// an `Option<T>` annotation or return type to know which generated
    /// struct to build. Catch the unannotated case at macro time instead
    /// of letting a `compile_error!` string reach naga.
    fn check_option_context(&mut self) {
        if self.uses_option_ctors && !self.has_option_type {
            self.errors.push(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!(
                    "`Some(...)`/`None` need an `Option<T>` type context in GPU kernel `{}`; \
                     annotate the `let` binding (`let x: Option<f32> = Some(1.0);`) or the \
                     function return type",
                    self.fn_name
                ),
            ));
        }
    }

    /// Heap allocation and self-recursion: both have no WGSL spelling.
    fn check_kernel_call(&mut self, node: &syn::ExprCall) {
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
    }

    /// Classify a `match`'s arms: patterns must sit in the shared
    /// subset, and a `_`/binding arm must come last.
    fn check_match_arms(&mut self, node: &syn::ExprMatch) {
        let mut seen_catch_all = false;
        for arm in &node.arms {
            let unreachable_arm = |this: &mut Self| {
                this.errors.push(syn::Error::new_spanned(
                    &arm.pat,
                    format!(
                        "this arm is unreachable in GPU kernel `{}`; a `_` or binding arm \
                         must come last",
                        this.fn_name
                    ),
                ));
            };
            if self.match_value
                && let syn::Expr::Block(b) = arm.body.as_ref()
                && !matches!(b.block.stmts.as_slice(), [syn::Stmt::Expr(_, None)])
            {
                self.errors.push(syn::Error::new_spanned(
                    arm.body.as_ref(),
                    format!(
                        "arms of a value `match` in GPU kernel `{}` must be a single expression; \
                         bind intermediate results with `let` and keep the arm to one trailing \
                         expression",
                        self.fn_name
                    ),
                ));
            }
            match crate::wgsl_lower::classify_pattern(&arm.pat) {
                Ok(crate::wgsl_lower::ArmPattern::Wild)
                | Ok(crate::wgsl_lower::ArmPattern::Bind(_)) => {
                    if seen_catch_all {
                        unreachable_arm(self);
                    }
                    seen_catch_all = true;
                }
                Ok(_) => {
                    if seen_catch_all {
                        unreachable_arm(self);
                    }
                }
                Err(msg) => {
                    self.errors.push(syn::Error::new_spanned(&arm.pat, msg));
                }
            }
        }
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

    // `Entity` gather plan: validates read positions, rejects writes
    // and entity equality, and names the lanes the generated WGSL and
    // `eval` share. Its errors are macro-time like the validator's.
    let plan = crate::wgsl_gather::scan(&func);
    if !plan.errors.is_empty() {
        let compile_errors = plan.errors.iter().map(|e| e.to_compile_error());
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

    // `eval` mirrors the WGSL gather exactly: one trailing `&[T]` lane
    // parameter per recorded (binding, field), the rewritten body
    // calling the slice gather helper.
    let lane_params: Vec<_> = plan
        .lanes()
        .map(|lane| {
            let ident = syn::Ident::new(&lane.lane_ident, proc_macro2::Span::call_site());
            let ty = &lane.elem_ty;
            quote! { #ident: &[#ty] }
        })
        .collect();
    let eval_inputs = if lane_params.is_empty() {
        quote! { ( #inputs ) }
    } else {
        quote! { ( #inputs, #(#lane_params),* ) }
    };
    let entity_items = if plan.has_entity() {
        let entity = crate::wgsl_gather::entity_struct_tokens();
        quote! { #entity }
    } else {
        quote! {}
    };
    let gather_fns = if plan.has_lanes() {
        let gather = crate::wgsl_gather::cpu_gather_tokens();
        quote! { #gather }
    } else {
        quote! {}
    };
    let eval_body = if plan.has_lanes() {
        let rewritten = crate::wgsl_gather::rewrite_eval_body(&func.block, &plan);
        quote! { #rewritten }
    } else {
        quote! { #body }
    };

    let expanded = quote! {
        // The user-facing `#[kernel] fn` carries the docs; the generated
        // helper module (eval/wgsl_source/...) is intentionally bare and
        // hidden from rustdoc.
        #[doc(hidden)]
        pub mod #fn_name {
            #entity_items
            #gather_fns

            // The user-facing `#[kernel] fn` carries the docs; the generated
            // helper surface (eval/wgsl_source/...) is intentionally bare and
            // hidden from rustdoc.
            #[allow(missing_docs)]
            #[allow(non_snake_case)]
            pub fn eval #eval_inputs #output {
                use super::*;
                #eval_body
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
        for path in [
            "u8::MAX",
            "u32::MAX",
            "f32::consts::PI",
            "<u8>::MAX",
            "core::u8::MAX",
            "std::primitive::u8::MAX",
            "std::f32::consts::PI",
        ] {
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

    #[test]
    fn validator_rejects_alloc_and_recursive_calls() {
        let alloc: syn::Expr = syn::parse_str("Vec(x)").expect("call must parse");
        let syn::Expr::Call(call) = alloc else {
            panic!("expected a call expression");
        };
        let mut validator = KernelValidator::new("kernel_fn");
        validator.check_kernel_call(&call);
        assert_eq!(validator.errors.len(), 1);
        assert!(
            validator.errors[0]
                .to_string()
                .contains("dynamic allocation"),
            "{:?}",
            validator.errors[0]
        );

        let recur: syn::Expr = syn::parse_str("kernel_fn(x)").expect("call must parse");
        let syn::Expr::Call(call) = recur else {
            panic!("expected a call expression");
        };
        let mut validator = KernelValidator::new("kernel_fn");
        validator.check_kernel_call(&call);
        assert_eq!(validator.errors.len(), 1);
        assert!(
            validator.errors[0]
                .to_string()
                .contains("recursion is not allowed"),
            "{:?}",
            validator.errors[0]
        );
    }

    #[test]
    fn validator_rejects_arms_after_catch_all() {
        let expr: syn::Expr =
            syn::parse_str("match x { _ => 1, y => 2 }").expect("match must parse");
        let syn::Expr::Match(node) = expr else {
            panic!("expected a match expression");
        };
        let mut validator = KernelValidator::new("kernel_fn");
        validator.check_match_arms(&node);
        assert_eq!(validator.errors.len(), 1);
        assert!(
            validator.errors[0].to_string().contains("unreachable"),
            "{:?}",
            validator.errors[0]
        );
    }
}
