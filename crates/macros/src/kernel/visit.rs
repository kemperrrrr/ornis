//! The `syn::visit::Visit` traversal for [`KernelValidator`].
//!
//! The walk rejects constructs WGSL has no spelling for — casts to
//! unsupported targets, closures, `while`, stray or deeply nested
//! `match`es, bad literal suffixes — and records the signals
//! [`super::KernelValidator::validate`] reads afterwards (`match`
//! position, `Option` constructor usage, `let` annotations).

use syn::visit::{self, Visit};

use super::KernelValidator;

impl<'ast> Visit<'ast> for KernelValidator {
    fn visit_expr_cast(&mut self, node: &'ast syn::ExprCast) {
        if crate::wgsl_lower::cast_target_ty(&node.ty).is_none() {
            self.errors.push(syn::Error::new_spanned(
                &node.ty,
                format!(
                    "cast to `{}` is not supported in GPU kernel `{}`; supported targets: f32, u32, i32",
                    crate::kernel_types::type_string(&node.ty),
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
            if crate::kernel_types::is_option_type(&pt.ty) {
                self.has_option_type = true;
            }
            if let Some(msg) =
                crate::kernel_types::kernel_type_error(&pt.ty, "let annotation", &self.fn_name)
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
        if crate::kernel_types::is_scalar_namespaced(&node.path, node.qself.as_ref()) {
            self.errors.push(syn::Error::new_spanned(
                &node.path,
                format!(
                    "path `{}` is not supported in GPU kernel `{}`; scalar associated constants have no WGSL spelling, use a plain `f32`, `u32` or `i32` literal instead",
                    crate::kernel_types::path_string(&node.path, node.qself.as_ref()),
                    self.fn_name
                ),
            ));
        }
        visit::visit_expr_path(self, node);
    }
}
