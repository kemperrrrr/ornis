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
            if self.match_value {
                if let syn::Expr::Block(b) = arm.body.as_ref() {
                    let single = matches!(b.block.stmts.as_slice(), [syn::Stmt::Expr(_, None)]);
                    if !single {
                        self.errors.push(syn::Error::new_spanned(
                            arm.body.as_ref(),
                            format!(
                                "arms of a value `match` in GPU kernel `{}` must be a single \
                                 expression; bind intermediate results with `let` and keep the \
                                 arm to one trailing expression",
                                self.fn_name
                            ),
                        ));
                    }
                }
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

impl<'ast> Visit<'ast> for KernelValidator {
    fn visit_expr_cast(&mut self, node: &'ast syn::ExprCast) {
        if crate::wgsl_lower::cast_target_ty(&node.ty).is_none() {
            self.errors.push(syn::Error::new_spanned(
                &node.ty,
                format!(
                    "cast to `{}` is not supported in GPU kernel `{}`; supported targets: f32, u32, i32",
                    super::kernel_types::type_string(&node.ty),
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

    /// `while` has no bounded spelling: a GPU loop must be a fixed
    /// `for`, never an unbounded `while`.
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
        if let syn::Pat::Type(pt) = &node.pat {
            if super::kernel_types::is_option_type(&pt.ty) {
                self.has_option_type = true;
            }
            if let Some(msg) =
                super::kernel_types::kernel_type_error(&pt.ty, "let annotation", &self.fn_name)
            {
                self.errors.push(syn::Error::new_spanned(&pt.ty, msg));
            }
        }
        // `let x = match …`: the value form only lowers as the direct
        // initializer of a binding — annotated or not.
        if let Some(init) = &node.init
            && matches!(init.expr.as_ref(), syn::Expr::Match(_))
        {
            self.match_ok = true;
            self.match_value = true;
        }
        visit::visit_local(self, node);
        self.match_ok = false;
        self.match_value = false;
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
        if !self.match_ok {
            self.errors.push(syn::Error::new_spanned(
                node,
                format!(
                    "`match` in GPU kernel `{}` only lowers as a statement or as the direct \
                     initializer of a `let` binding; bind the result first, then `return` it",
                    self.fn_name
                ),
            ));
        }
        self.check_match_arms(node);
        // Arm bodies do not inherit the value-position context.
        let outer = std::mem::replace(&mut self.match_ok, false);
        visit::visit_expr_match(self, node);
        self.match_ok = outer;
        self.match_depth -= 1;
    }

    /// A statement-position `match` is a supported host for the value
    /// form's sibling: arm bodies may be full blocks there. `return
    /// match …` hosts the value form, whose arms must be single
    /// expressions.
    fn visit_stmt(&mut self, node: &'ast syn::Stmt) {
        let (host, value) = match node {
            syn::Stmt::Expr(syn::Expr::Match(_), _) => (true, false),
            syn::Stmt::Expr(syn::Expr::Return(r), _) => {
                (matches!(r.expr.as_deref(), Some(syn::Expr::Match(_))), true)
            }
            _ => (false, false),
        };
        if host {
            self.match_ok = true;
            self.match_value = value;
            visit::visit_stmt(self, node);
            self.match_ok = false;
            self.match_value = false;
            return;
        }
        visit::visit_stmt(self, node);
    }

    /// `if let` / `while let` patterns share the `match` subset.
    fn visit_expr_let(&mut self, node: &'ast syn::ExprLet) {
        if let Err(msg) = crate::wgsl_lower::classify_pattern(&node.pat) {
            self.errors.push(syn::Error::new_spanned(
                &node.pat,
                format!("{msg} (in `if let` of GPU kernel `{}`)", self.fn_name),
            ));
        }
        visit::visit_expr_let(self, node);
    }

    /// `Some(...)`/`None` mark the body as needing an `Option<T>` type
    /// context somewhere in the signature or a `let` annotation.
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = node.func.as_ref()
            && p.path.segments.len() == 1
            && matches!(
                p.path.segments[0].ident.to_string().as_str(),
                "Some" | "None"
            )
        {
            self.uses_option_ctors = true;
        }
        self.check_kernel_call(node);
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
        if node.path.get_ident().is_some_and(|i| i == "None") {
            self.uses_option_ctors = true;
        }
        if super::kernel_types::is_scalar_namespaced(&node.path, node.qself.as_ref()) {
            self.errors.push(syn::Error::new_spanned(
                &node.path,
                format!(
                    "path `{}` is not supported in GPU kernel `{}`; scalar associated constants have no WGSL spelling, use a plain `f32`, `u32` or `i32` literal instead",
                    super::kernel_types::path_string(&node.path, node.qself.as_ref()),
                    self.fn_name
                ),
            ));
        }
        visit::visit_expr_path(self, node);
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
}
