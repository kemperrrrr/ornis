//! Derivability gate for `#[smart_system]` (PLAN m, path 3): everything the
//! macro cannot derive an access set from is a compile error unless the
//! `opaque` hatch is set.
//!
//! - [`SmartSystemArgs`]: `name = "..."`, `reads(...)` / `writes(...)`,
//!   `reads_lane(...)` / `writes_lane(...)`, and `opaque`.
//! - [`EscapeCheck`]: any use of the `resources` ident besides the receiver
//!   of `get::<T>()` / `contains::<T>()` (or side-effect-free `len()` /
//!   `is_empty()`), and any use of a store ident besides the receiver of
//!   `read_lane` / `write_lane` / `create_entity` / `is_alive`, hides an
//!   access. A store escape is additionally covered by explicit
//!   `reads_lane(...)` / `writes_lane(...)`; everything else needs `opaque`.
//! - [`TurbofishCheck`]: `resources.get()` without a turbofish type hides
//!   its type (checked for `smart_system` only).
//! - Known std macros (`assert!`, `println!`, …) hold plain expressions and
//!   are checked inside; any other macro mentioning `resources` or a store
//!   ident is rejected.

use proc_macro2::TokenStream as TokenStream2;
use syn::{Expr, Ident, ItemFn, visit::Visit};

use crate::smart_pipeline::{
    DERIVABLE_MACROS, combine_errors_tokens, macro_exprs, macro_mentions_ident, macro_name,
};

/// `#[smart_system]` arguments: `name = "..."`, `reads(A, ...)` /
/// `writes(B, ...)`, `reads_lane(P, ...)` / `writes_lane(Q, ...)`, and the
/// `opaque` escape hatch (see [`check_derivable`]).
pub(crate) struct SmartSystemArgs {
    pub(crate) name: Option<String>,
    pub(crate) reads: Vec<syn::Type>,
    pub(crate) writes: Vec<syn::Type>,
    pub(crate) reads_lane: Vec<syn::Type>,
    pub(crate) writes_lane: Vec<syn::Type>,
    pub(crate) opaque: bool,
}

fn type_list(tokens: proc_macro2::TokenStream) -> syn::Result<Vec<syn::Type>> {
    struct Types(Vec<syn::Type>);
    impl syn::parse::Parse for Types {
        fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
            Ok(Self(
                <syn::punctuated::Punctuated<syn::Type, syn::Token![,]>>::parse_terminated(input)?
                    .into_iter()
                    .collect(),
            ))
        }
    }
    syn::parse2::<Types>(tokens)
        .map(|types| types.0)
        .map_err(|_| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                "#[smart_system]: expected a comma-separated type list, e.g. `reads(A, B)`",
            )
        })
}

impl syn::parse::Parse for SmartSystemArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut args = Self {
            name: None,
            reads: Vec::new(),
            writes: Vec::new(),
            reads_lane: Vec::new(),
            writes_lane: Vec::new(),
            opaque: false,
        };
        let items =
            <syn::punctuated::Punctuated<syn::Meta, syn::Token![,]>>::parse_terminated(input)?;
        for item in items {
            match item {
                syn::Meta::Path(path) if path.is_ident("opaque") => {
                    args.opaque = true;
                }
                syn::Meta::NameValue(named) if named.path.is_ident("name") => {
                    if let Expr::Lit(expr) = &named.value
                        && let syn::Lit::Str(text) = &expr.lit
                    {
                        if args.name.is_some() {
                            return Err(syn::Error::new_spanned(
                                named.path,
                                "#[smart_system]: duplicate `name`",
                            ));
                        }
                        args.name = Some(text.value());
                    } else {
                        return Err(syn::Error::new_spanned(
                            named.value,
                            "#[smart_system]: expected `name = \"...\"`",
                        ));
                    }
                }
                syn::Meta::List(list) if list.path.is_ident("reads") => {
                    args.reads.extend(type_list(list.tokens)?);
                }
                syn::Meta::List(list) if list.path.is_ident("writes") => {
                    args.writes.extend(type_list(list.tokens)?);
                }
                syn::Meta::List(list) if list.path.is_ident("reads_lane") => {
                    args.reads_lane.extend(type_list(list.tokens)?);
                }
                syn::Meta::List(list) if list.path.is_ident("writes_lane") => {
                    args.writes_lane.extend(type_list(list.tokens)?);
                }
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "#[smart_system]: expected `name = \"...\"`, `reads(A, ...)`, `writes(B, ...)`, `reads_lane(P, ...)`, `writes_lane(Q, ...)` or `opaque`",
                    ));
                }
            }
        }
        Ok(args)
    }
}

/// Flags `resources` and store-ident uses the macro cannot derive an access
/// set from: anything but the receiver of `get::<T>()` / `contains::<T>()`
/// (or the side-effect-free `len()` / `is_empty()`) for `resources`, and
/// anything but the receiver of `read_lane` / `write_lane` / `create_entity`
/// / `is_alive` for a store ident. Runs only with the `opaque` escape hatch
/// absent. Exemptions: explicit `reads(...)` / `writes(...)` declare
/// individual resource types but never blanket-approve an escape; a store
/// escape is additionally covered by explicit `reads_lane(...)` /
/// `writes_lane(...)` (the author takes responsibility for the hidden lanes,
/// mirroring N2), while a `resources` escape or an opaque-macro use needs
/// `opaque` with every touched resource and lane listed.
pub(crate) struct EscapeCheck<'a> {
    pub(crate) resources: &'a Ident,
    pub(crate) store_idents: Vec<Ident>,
    pub(crate) lane_declared: bool,
    pub(crate) errors: Vec<syn::Error>,
}

impl EscapeCheck<'_> {
    fn is_resources_path(&self, expr: &Expr) -> bool {
        matches!(expr, Expr::Path(path) if path.path.is_ident(self.resources))
    }

    fn is_store_path(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Path(path) => path
                .path
                .get_ident()
                .is_some_and(|ident| self.store_idents.iter().any(|stored| stored == ident)),
            _ => false,
        }
    }

    /// A plain store re-alias (`let s2 = store;`, `&store`, `store.clone()`)
    /// is tracked by the collector, not an escape.
    fn is_store_realias(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Path(path) => path
                .path
                .get_ident()
                .is_some_and(|ident| self.store_idents.iter().any(|stored| stored == ident)),
            Expr::Reference(reference) => self.is_store_path(&reference.expr),
            Expr::MethodCall(mc) if mc.method == "clone" && mc.args.is_empty() => {
                self.is_store_path(&mc.receiver)
            }
            _ => false,
        }
    }

    fn deny_resources(&mut self, expr: &Expr) {
        self.errors.push(syn::Error::new_spanned(
            expr,
            "#[smart_system]: `resources` escapes here; access cannot be derived — declare it with `reads(...)`/`writes(...)`",
        ));
    }

    fn deny_store(&mut self, expr: &Expr) {
        // Explicit lane declarations take responsibility for lanes hidden
        // behind a store escape (N2); anything else needs `opaque`.
        if self.lane_declared {
            return;
        }
        self.errors.push(syn::Error::new_spanned(
            expr,
            "#[smart_system]: `store` escapes here; lane accesses cannot be derived — declare them with `reads_lane(...)`/`writes_lane(...)`",
        ));
    }

    /// Shared macro handling: known std macros hold plain expressions, so
    /// the check descends into them via a sub-checker (parsed expressions are
    /// temporaries and cannot be visited with `self` directly); any other
    /// macro mentioning `resources` or a store ident hides an access and is
    /// rejected.
    fn visit_macro_common(&mut self, node: &syn::Macro) -> bool {
        let known = macro_name(node).is_some_and(|name| DERIVABLE_MACROS.contains(&name.as_str()));
        if known && let Some(exprs) = macro_exprs(node) {
            let mut sub = EscapeCheck {
                resources: self.resources,
                store_idents: self.store_idents.clone(),
                lane_declared: self.lane_declared,
                errors: Vec::new(),
            };
            for expr in &exprs {
                sub.visit_expr(expr);
            }
            self.errors.extend(sub.errors);
            return true;
        }
        let idents: Vec<&Ident> = std::iter::once(self.resources)
            .chain(self.store_idents.iter())
            .collect();
        if !known && macro_mentions_ident(node, &idents) {
            self.errors.push(syn::Error::new_spanned(
                node.tokens.clone(),
                "#[smart_system]: `resources`/`store` used inside a macro invocation; accesses cannot be derived — hoist the call into a `let` or declare it explicitly",
            ));
        }
        !known
    }
}

impl<'a> Visit<'a> for EscapeCheck<'a> {
    fn visit_local(&mut self, node: &'a syn::Local) {
        if let Some(init) = node.init.as_ref()
            && self.is_store_realias(&init.expr)
        {
            // Tracked alias, not an escape: skip the init, still check the rest.
            return;
        }
        syn::visit::visit_local(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'a syn::ExprMethodCall) {
        if self.is_resources_path(&node.receiver)
            && (node.method == "get"
                || node.method == "contains"
                || node.method == "len"
                || node.method == "is_empty")
        {
            // Derivable (or side-effect-free): check the arguments only.
            for arg in &node.args {
                syn::visit::visit_expr(self, arg);
            }
            return;
        }
        if self.is_store_path(&node.receiver)
            && (node.method == "read_lane"
                || node.method == "write_lane"
                || node.method == "create_entity"
                || node.method == "is_alive")
        {
            // Derivable lane access (or lane-free store op): check args only.
            for arg in &node.args {
                syn::visit::visit_expr(self, arg);
            }
            return;
        }
        if self.is_resources_path(&node.receiver) {
            self.deny_resources(&node.receiver);
        }
        if self.is_store_path(&node.receiver) {
            self.deny_store(&node.receiver);
        }
        // The receiver is already reported: check the arguments only, so one
        // site yields one diagnostic.
        if self.is_resources_path(&node.receiver) || self.is_store_path(&node.receiver) {
            for arg in &node.args {
                syn::visit::visit_expr(self, arg);
            }
            return;
        }
        syn::visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_path(&mut self, node: &'a syn::ExprPath) {
        if node.path.is_ident(self.resources) {
            self.deny_resources(&Expr::Path(node.clone()));
        }
        if node
            .path
            .get_ident()
            .is_some_and(|ident| self.store_idents.iter().any(|stored| stored == ident))
        {
            self.deny_store(&Expr::Path(node.clone()));
        }
        syn::visit::visit_expr_path(self, node);
    }

    fn visit_macro(&mut self, node: &'a syn::Macro) {
        if self.visit_macro_common(node) {
            return;
        }
        syn::visit::visit_macro(self, node);
    }
}

/// A `resources.get()` / `contains()` call without a valid turbofish hides
/// its type, so the access set cannot be derived. Standalone visitor (kept
/// out of the shared collector so `#[smart_pipeline]` stays silent).
pub(crate) struct TurbofishCheck<'a> {
    pub(crate) resources: &'a Ident,
    pub(crate) macro_name: &'static str,
    pub(crate) errors: Vec<syn::Error>,
}

impl<'a> Visit<'a> for TurbofishCheck<'a> {
    fn visit_expr_method_call(&mut self, node: &'a syn::ExprMethodCall) {
        if (node.method == "get" || node.method == "contains")
            && let Expr::Path(path) = &*node.receiver
            && path.path.is_ident(self.resources)
            && crate::smart_pipeline::turbofish_type(node, self.macro_name).is_err()
        {
            self.errors.push(syn::Error::new_spanned(
                node,
                format!(
                    "{}: `resources.get()` needs a turbofish type so the access set can be derived — write `resources.get::<T>()`",
                    self.macro_name
                ),
            ));
        }
        syn::visit::visit_expr_method_call(self, node);
    }

    fn visit_macro(&mut self, node: &'a syn::Macro) {
        // Same split as the collector: known std macros hold plain
        // expressions and are checked inside via a sub-checker; anything
        // else stays opaque (the escape check reports hidden idents).
        let known = macro_name(node).is_some_and(|name| DERIVABLE_MACROS.contains(&name.as_str()));
        if known && let Some(exprs) = macro_exprs(node) {
            let mut sub = TurbofishCheck {
                resources: self.resources,
                macro_name: self.macro_name,
                errors: Vec::new(),
            };
            for expr in &exprs {
                sub.visit_expr(expr);
            }
            self.errors.extend(sub.errors);
        }
    }
}

/// Runs the derivability checks over the body: typeless `get()` calls
/// (the type would be hidden), escaping `resources`/`store` uses (an access
/// would be hidden), and accesses hidden inside opaque macro invocations.
/// The escape check runs unless the `opaque` hatch is set: explicit
/// `reads(...)` / `writes(...)` / `reads_lane(...)` / `writes_lane(...)`
/// declare individual types, they never blanket-approve an escape. With
/// `opaque` the author takes responsibility and lists every touched
/// resource and lane explicitly. Errors combine into one diagnostic.
pub(crate) fn check_derivable(
    input: &ItemFn,
    resources: &Ident,
    store_idents: Vec<Ident>,
    args: &SmartSystemArgs,
) -> Result<(), TokenStream2> {
    // Turbofish strictness: a typeless `get()` hides its type (smart_system
    // only; `smart_pipeline` stays silent).
    let mut turbofish = TurbofishCheck {
        resources,
        macro_name: "#[smart_system]",
        errors: Vec::new(),
    };
    turbofish.visit_block(&input.block);
    if !turbofish.errors.is_empty() {
        return Err(combine_errors_tokens(turbofish.errors));
    }

    // Escape check: any other use of the `resources` / store idents hides an
    // access. Skipped only with the explicit `opaque` opt-in.
    if !args.opaque {
        let mut escape = EscapeCheck {
            resources,
            store_idents,
            lane_declared: !args.reads_lane.is_empty() || !args.writes_lane.is_empty(),
            errors: Vec::new(),
        };
        escape.visit_block(&input.block);
        if !escape.errors.is_empty() {
            return Err(combine_errors_tokens(escape.errors));
        }
    }
    Ok(())
}

#[cfg(test)]
mod smart_system_checks_tests {
    use super::*;
    use syn::visit::Visit;

    fn parse_item(src: &str) -> ItemFn {
        syn::parse_str(src).expect("fn parses")
    }

    fn resources_ident() -> Ident {
        syn::parse_str("resources").expect("ident")
    }

    fn check_args() -> SmartSystemArgs {
        SmartSystemArgs {
            name: None,
            reads: Vec::new(),
            writes: Vec::new(),
            reads_lane: Vec::new(),
            writes_lane: Vec::new(),
            opaque: false,
        }
    }

    #[test]
    fn check_derivable_passes_clean_bodies() {
        let item = parse_item(
            "fn f(resources: &Resources) { let _ = resources.get::<Cfg>(); let _ = resources.len(); }",
        );
        let args = check_args();
        let resources = resources_ident();
        assert!(check_derivable(&item, &resources, Vec::new(), &args).is_ok());
    }

    #[test]
    fn opaque_skips_escape_check() {
        let item = parse_item("fn f(resources: &Resources) { helper(resources); }");
        let open = SmartSystemArgs {
            opaque: true,
            ..check_args()
        };
        let resources = resources_ident();
        assert!(check_derivable(&item, &resources, Vec::new(), &open).is_ok());
        let closed = SmartSystemArgs {
            reads: vec![syn::parse_str("Cfg").expect("type")],
            ..check_args()
        };
        assert!(check_derivable(&item, &resources, Vec::new(), &closed).is_err());
    }

    #[test]
    fn escape_check_allows_get_len_and_denies_calls() {
        let resources = resources_ident();
        let clean: ItemFn = parse_item(
            "fn f(resources: &Resources) { let _ = resources.get::<Cfg>(); let _ = resources.len(); }",
        );
        let mut ok = EscapeCheck {
            resources: &resources,
            store_idents: Vec::new(),
            lane_declared: false,
            errors: Vec::new(),
        };
        ok.visit_block(&clean.block);
        assert!(ok.errors.is_empty());

        let dirty: ItemFn = parse_item("fn f(resources: &Resources) { helper(resources); }");
        let mut bad = EscapeCheck {
            resources: &resources,
            store_idents: Vec::new(),
            lane_declared: false,
            errors: Vec::new(),
        };
        bad.visit_block(&dirty.block);
        assert_eq!(bad.errors.len(), 1);
    }

    #[test]
    fn escape_check_denies_store_escape_but_allows_lanes() {
        let resources = resources_ident();
        let store: Ident = syn::parse_str("store").expect("ident");
        let clean: ItemFn = parse_item(
            "fn f(resources: &Resources) { let s = store.read_lane::<V>().expect(\"v\"); let _ = s.len(); }",
        );
        let mut ok = EscapeCheck {
            resources: &resources,
            store_idents: vec![store.clone()],
            lane_declared: false,
            errors: Vec::new(),
        };
        ok.visit_block(&clean.block);
        assert!(
            ok.errors.is_empty(),
            "{:?}",
            ok.errors
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        );

        let dirty: ItemFn = parse_item("fn f(resources: &Resources) { helper(store); }");
        let mut bad = EscapeCheck {
            resources: &resources,
            store_idents: vec![store.clone()],
            lane_declared: false,
            errors: Vec::new(),
        };
        bad.visit_block(&dirty.block);
        assert_eq!(bad.errors.len(), 1);
        assert!(
            bad.errors[0].to_string().contains("reads_lane"),
            "{:?}",
            bad.errors[0].to_string()
        );

        // With explicit lane declarations the author takes responsibility
        // for the hidden lanes (N2): the same escape compiles.
        let mut covered = EscapeCheck {
            resources: &resources,
            store_idents: vec![store],
            lane_declared: true,
            errors: Vec::new(),
        };
        covered.visit_block(&dirty.block);
        assert!(covered.errors.is_empty());
    }
    #[test]
    fn escape_check_handles_macro_invocations() {
        fn invocation(src: &str) -> syn::Macro {
            match syn::parse_str::<syn::Stmt>(src).expect("stmt parses") {
                syn::Stmt::Macro(stmt) => stmt.mac,
                _ => panic!("expected a macro statement"),
            }
        }
        let resources = resources_ident();
        let mut check = EscapeCheck {
            resources: &resources,
            store_idents: Vec::new(),
            lane_declared: false,
            errors: Vec::new(),
        };
        // Known macro: derived, no error.
        assert!(
            check.visit_macro_common(&invocation("assert!(resources.get::<Cfg>().is_some());"))
        );
        assert!(check.errors.is_empty());
        // Unknown macro mentioning `resources`: rejected (returns true =
        // handled, with the error recorded).
        assert!(check.visit_macro_common(&invocation("my_macro!(resources);")));
        assert_eq!(check.errors.len(), 1);
        assert!(
            check.errors[0].to_string().contains("macro invocation"),
            "{:?}",
            check.errors[0].to_string()
        );
        // Unknown macro without our idents: not ours to report.
        let mut clean = EscapeCheck {
            resources: &resources,
            store_idents: Vec::new(),
            lane_declared: false,
            errors: Vec::new(),
        };
        // Unknown macro without our idents: not ours to report (still
        // handled: nothing descends into opaque tokens).
        assert!(clean.visit_macro_common(&invocation("my_macro!(other);")));
        assert!(clean.errors.is_empty());

        // The same split through full bodies.
        let known: ItemFn = parse_item(
            "fn f(resources: &Resources) { assert!(resources.get::<Cfg>().is_some()); }",
        );
        let mut ok = EscapeCheck {
            resources: &resources,
            store_idents: Vec::new(),
            lane_declared: false,
            errors: Vec::new(),
        };
        ok.visit_block(&known.block);
        assert!(ok.errors.is_empty());

        let unknown: ItemFn = parse_item("fn f(resources: &Resources) { my_macro!(resources); }");
        let mut bad = EscapeCheck {
            resources: &resources,
            store_idents: Vec::new(),
            lane_declared: false,
            errors: Vec::new(),
        };
        bad.visit_block(&unknown.block);
        assert_eq!(bad.errors.len(), 1);
    }
}
