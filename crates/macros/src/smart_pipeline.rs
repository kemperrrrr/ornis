//! Implementation of the `#[smart_pipeline]` attribute macro.
//!
//! The macro rewrites a function working over `SmartStore` lanes:
//!
//! ```ignore
//! #[smart_pipeline]
//! fn integrate(store: &SmartStore, dt: f32) {
//!     let mut positions = store.write_lane::<Position>().unwrap();
//!     let velocities = store.read_lane::<Velocity>().unwrap();
//!     for (pos, vel) in positions.iter_mut().zip(velocities.iter()) {
//!         pos.x += vel.x * dt;
//!     }
//! }
//! ```
//!
//! - Lane bindings (`let x = store.read_lane::<T>()` / `store.write_lane::<T>()`)
//!   are detected via the turbofish type argument of the call.
//! - `for` loops over lane iterators (`lane.iter()` / `lane.iter_mut()`,
//!   optionally two of them combined with `.zip(..)`) are rewritten to
//!   parallel Rayon iteration **in place**; the rest of the function body is
//!   preserved verbatim.
//! - Loops that cannot be proven parallel-safe (captured mutable state,
//!   cross-iteration indexing, `break`/`continue`/`return`, unrecognized
//!   iterator shapes) are left as ordinary sequential `for` loops and get a
//!   compile-time warning (via the `deprecated`-note trick, which surfaces in
//!   the IDE and the terminal).
//! - R6: every bound lane type is statically asserted `Send + Sync` at the
//!   top of the function, so the diagnostic points at the annotated system
//!   rather than deep generic internals. The rewritten loop is type-checked
//!   inside that same bound, so a rejected lane does not also fail in
//!   `par_iter` / `for_each`.
//! - R3: the access set derived from the lane bindings is exported as a
//!   module-scope `const __SMART_PIPELINE_ACCESS_<fn>: &[(&str, bool)]`
//!   (sorted `(type, is_write)` pairs; a write covers a read of the same
//!   lane; the const carries `allow(dead_code, non_upper_case_globals)`).
//!   `smart_system` consumes the same extraction to generate `System::access`
//!   directly, so declaration and body cannot desync.
//!
//! Known limitations (the analysis is syntactic, not type-directed):
//! - At most two lanes per `zip` are parallelized; longer `zip` chains stay
//!   sequential.
//! - A loop body that captures another lane guard (e.g. calling
//!   `other_lane.get(entity)` inside a parallel loop) will fail to compile
//!   because `RwLock` guards are not `Sync`; hoist such accesses out of the
//!   loop.
//! - The lane variable must be bound by a plain `let` directly from
//!   `store.read_lane::<T>()` / `store.write_lane::<T>()` (`.unwrap()` /
//!   `.expect(..)` wrappers around the call are allowed).
//! - Conscious extension (PLAN m, R3): the `store` receiver may also be an
//!   alias bound from `resources.get::<SmartStore>()` (same wrappers), so
//!   `#[smart_pipeline]` works over `&Resources`-fed functions too. Lane
//!   patterns include `let Type ident` and let-else `Some(ident)` /
//!   `Ok(ident)`; lane calls in any expression position count toward the
//!   access set, but only plain bindings drive loop rewriting. A lane call
//!   chained directly off `resources.get::<SmartStore>()` counts as well,
//!   as do calls inside known std macros (`assert!`, `println!`, …); unknown
//!   macro invocations stay opaque.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, format_ident, quote};
use syn::{
    BinOp, Expr, ExprAssign, ExprBinary, ExprClosure, ExprForLoop, ExprIndex, ExprMethodCall,
    FnArg, Ident, ItemFn, Local, Pat, PatIdent, PatType, Type, TypePath, TypeReference,
    parse_macro_input,
    visit::{self, Visit},
    visit_mut::{self, VisitMut},
};

/// A `let` binding of a `SmartStore` lane guard, usable for loop rewriting.
/// Lane types for assertions and the access set come from `lane_accesses`
/// (every call site, any context), not from bindings.
pub(crate) struct LaneBinding {
    pub(crate) var_name: Ident,
    pub(crate) is_mutable: bool,
}

/// Extracts the single turbofish type argument of `read_lane::<T>()` /
/// `write_lane::<T>()` (and `get::<T>()` / `contains::<T>()`). The turbofish
/// lives in [`ExprMethodCall::turbofish`], not in `args`.
pub(crate) fn turbofish_type(
    node: &ExprMethodCall,
    macro_name: &'static str,
) -> Result<Type, syn::Error> {
    let error = || {
        syn::Error::new_spanned(
            node,
            format!(
                "{macro_name}: `{}` requires exactly one turbofish type argument, \
                 e.g. `store.{}::<Position>()`",
                node.method, node.method
            ),
        )
    };
    let turbofish = node.turbofish.as_ref().ok_or_else(error)?;
    if turbofish.args.len() != 1 {
        return Err(error());
    }
    match turbofish.args.first() {
        Some(syn::GenericArgument::Type(ty)) => Ok(ty.clone()),
        _ => Err(error()),
    }
}

#[cfg(test)]
mod turbofish_tests {
    use super::*;
    use quote::ToTokens;

    fn method_call(src: &str) -> ExprMethodCall {
        let expr: syn::Expr = syn::parse_str(src).expect("parse expr");
        match expr {
            syn::Expr::MethodCall(mc) => mc,
            _ => panic!("expected method call"),
        }
    }

    #[test]
    fn turbofish_extracts_type() {
        let mc = method_call("store.read_lane::<Position>()");
        let ty = turbofish_type(&mc, "#[smart_pipeline]").expect("turbofish present");
        assert_eq!(ty.to_token_stream().to_string(), "Position");
    }

    #[test]
    fn turbofish_write_lane_works() {
        let mc = method_call("store.write_lane::<Vec3>()");
        let ty = turbofish_type(&mc, "#[smart_pipeline]").expect("ok");
        assert_eq!(ty.to_token_stream().to_string(), "Vec3");
    }

    #[test]
    fn turbofish_missing_is_error() {
        let mc = method_call("store.read_lane()");
        match turbofish_type(&mc, "#[smart_pipeline]") {
            Err(err) => assert!(err.to_string().contains("turbofish"), "got: {err}"),
            Ok(_) => panic!("expected error for missing turbofish"),
        }
    }

    #[test]
    #[should_panic(expected = "expected method call")]
    fn non_method_call_panics_helper() {
        // Guard for the test helper itself.
        let _ = method_call("1 + 2");
    }

    #[test]
    fn lifetime_argument_is_rejected() {
        // A turbofish with a non-type argument must error.
        let src = "store.read_lane::<'a>()";
        let expr: syn::Expr = syn::parse_str(src).expect("parses");
        if let syn::Expr::MethodCall(mc) = expr {
            assert!(turbofish_type(&mc, "#[smart_pipeline]").is_err());
        } else {
            panic!("expected method call");
        }
    }

    #[test]
    fn two_arguments_are_rejected() {
        let src = "store.read_lane::<A, B>()";
        let expr: syn::Expr = syn::parse_str(src).expect("parses");
        if let syn::Expr::MethodCall(mc) = expr {
            assert!(turbofish_type(&mc, "#[smart_pipeline]").is_err());
        } else {
            panic!("expected method call");
        }
    }
}

/// Collects the `SmartStore` parameter name and all lane guard bindings of
/// the function. Also validates the turbofish of every `read_lane` /
/// `write_lane` call on the store, accumulating errors instead of panicking.
///
/// Shared with `smart_system`: besides lane bindings it also tracks the
/// `&Resources` parameter (by type) and `SmartStore` aliases bound from
/// `resources.get::<SmartStore>()`, plus `resources.get::<T>()` reads for
/// the access-set export.
#[derive(Default)]
pub(crate) struct LaneCollector {
    pub(crate) macro_name: &'static str,
    pub(crate) store_param: Option<Ident>,
    /// `&Resources` parameter ident (by type, any name).
    pub(crate) resources_param: Option<Ident>,
    /// Locals bound from `resources.get::<SmartStore>()` (through
    /// `.expect()`/`.unwrap()`): additional store receivers.
    pub(crate) store_aliases: Vec<Ident>,
    /// Bindings usable for loop rewriting: `let ident` / `let Type ident` /
    /// `let Some(ident)` / `let Ok(ident)` from a lane call. See `lane_accesses`
    /// for the full access set (any context).
    pub(crate) lanes: Vec<LaneBinding>,
    /// Every `read_lane::<T>()` / `write_lane::<T>()` call on a store
    /// receiver, in any context: `(type, is_write)`. Drives `Send + Sync`
    /// assertions and the access-set export.
    pub(crate) lane_accesses: Vec<(Type, bool)>,
    /// `T` in `resources.get::<T>()` / `resources.contains::<T>()`
    /// (resource reads for the access set).
    pub(crate) resource_reads: Vec<Type>,
    pub(crate) errors: Vec<syn::Error>,
}

impl LaneCollector {
    pub(crate) fn new(macro_name: &'static str) -> Self {
        Self {
            macro_name,
            ..Self::default()
        }
    }

    fn is_store_receiver(&self, receiver: &Expr) -> bool {
        match receiver {
            Expr::Path(p) => p
                .path
                .get_ident()
                .is_some_and(|ident| self.is_store_ident(ident)),
            // `resources.get::<SmartStore>().expect(..).read_lane::<T>()`:
            // the chain itself is a store receiver (N3). Anything else
            // (`.len()`, `.map()`, …) ends the chain on purpose.
            other => {
                let stripped = Self::strip_wrappers(other);
                as_method_call(stripped).is_some_and(|mc| {
                    self.resources_call_type(mc)
                        .is_some_and(|ty| Self::last_segment(&ty).as_deref() == Some("SmartStore"))
                })
            }
        }
    }

    /// The `T` of `node` itself if it is a `resources.get::<T>()` /
    /// `resources.contains::<T>()` call. Only the node itself is checked (no
    /// chain walk, no clone): the visitor descends into receivers on its own,
    /// so wrappers are visited as inner nodes.
    fn resources_read_here(&self, node: &ExprMethodCall) -> Option<Type> {
        self.resources_call_type(node)
    }

    /// Last path segment of a type, for `SmartStore` / `Resources` matching.
    pub(crate) fn last_segment(ty: &Type) -> Option<String> {
        if let Type::Path(TypePath { path, .. }) = ty {
            return path.segments.last().map(|s| s.ident.to_string());
        }
        None
    }

    /// Strips `.unwrap()` / `.expect(..)` wrappers, returning the wrapped
    /// expression. Traversal is deliberately limited to these two methods:
    /// anything else (`.len()`, `.map()`, …) ends the chain, so derived
    /// values are never mistaken for lane guards or store aliases.
    fn strip_wrappers(mut expr: &Expr) -> &Expr {
        loop {
            match expr {
                Expr::MethodCall(mc) if mc.method == "unwrap" || mc.method == "expect" => {
                    expr = &mc.receiver;
                }
                _ => return expr,
            }
        }
    }

    /// Branchless by design (IOSP contract): matching lives in `Option`
    /// combinators, doing in `Vec::push` — the `if` version reads nicer but
    /// mixes deciding with doing in one function body.
    fn lane_binding(&self, expr: &Expr) -> Option<(bool, Type)> {
        as_method_call(Self::strip_wrappers(expr)).and_then(|mc| {
            lane_call_write(mc).and_then(|write| {
                self.is_store_receiver(&mc.receiver)
                    .then(|| {
                        turbofish_type(mc, self.macro_name)
                            .ok()
                            .map(|ty| (write, ty))
                    })
                    .flatten()
            })
        })
    }

    /// If the wrapper-stripped `expr` is `resources.get::<SmartStore>()`,
    /// the bound local is a store alias for lane bindings below.
    /// Branchless like `lane_binding` (see above).
    fn store_alias_source(&self, expr: &Expr) -> bool {
        as_method_call(Self::strip_wrappers(expr))
            .and_then(|mc| self.resources_call_type(mc))
            .is_some_and(|ty| Self::last_segment(&ty).as_deref() == Some("SmartStore"))
    }

    /// The `T` of a `resources.get::<T>()` / `resources.contains::<T>()`
    /// call, or `None` (malformed turbofish is reported by the method-call
    /// visitor, so it stays silent here).
    fn resources_call_type(&self, mc: &ExprMethodCall) -> Option<Type> {
        if (mc.method != "get" && mc.method != "contains")
            || !matches!(&*mc.receiver, Expr::Path(path) if self
                .resources_param
                .as_ref()
                .is_some_and(|r| path.path.is_ident(r)))
        {
            return None;
        }
        turbofish_type(mc, self.macro_name).ok()
    }
}

/// Pure classification of a function parameter for the collector: a
/// `&SmartStore` lane source, the `&Resources` context, or neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParamKind {
    SmartStore,
    Resources,
    Other,
}

/// Classifies `ty` without touching visitor state.
pub(crate) fn classify_param(ty: &Type) -> ParamKind {
    if let Type::Reference(TypeReference { elem, .. }) = ty {
        match LaneCollector::last_segment(elem).as_deref() {
            Some("SmartStore") => ParamKind::SmartStore,
            Some("Resources") => ParamKind::Resources,
            _ => ParamKind::Other,
        }
    } else {
        ParamKind::Other
    }
}

/// Extracts the bound ident from `let` patterns the collector understands:
/// `ident`, `Type ident`, and single-element `Some(ident)` / `Ok(ident)`
/// (let-else). Anything else yields `None`.
pub(crate) fn binding_ident(pat: &Pat) -> Option<Ident> {
    match pat {
        Pat::Ident(pat_ident) => Some(pat_ident.ident.clone()),
        Pat::Type(pat_type) => binding_ident(&pat_type.pat),
        Pat::TupleStruct(tuple) => tuple_struct_binding(tuple),
        _ => None,
    }
}

/// Single-element `Some(ident)` / `Ok(ident)` (let-else) unwrap.
fn tuple_struct_binding(tuple: &syn::PatTupleStruct) -> Option<Ident> {
    if tuple.elems.len() != 1 {
        return None;
    }
    let name = tuple.path.segments.last()?.ident.to_string();
    if name != "Some" && name != "Ok" {
        return None;
    }
    if let Pat::Ident(pat_ident) = &tuple.elems[0] {
        Some(pat_ident.ident.clone())
    } else {
        None
    }
}

/// The node if it is a plain method call. Pure shape test, no calls.
fn as_method_call(expr: &Expr) -> Option<&ExprMethodCall> {
    if let Expr::MethodCall(mc) = expr {
        Some(mc)
    } else {
        None
    }
}

/// Lane-access mutability of a bare method call name, if it is one.
fn lane_call_write(mc: &ExprMethodCall) -> Option<bool> {
    if mc.method == "read_lane" {
        Some(false)
    } else if mc.method == "write_lane" {
        Some(true)
    } else {
        None
    }
}

impl LaneCollector {
    /// Registers one `let`: a store alias and/or a rewritable lane binding.
    /// Resource reads themselves are recorded by `visit_expr_method_call`.
    /// A plain re-alias (`let s2 = store;`, `&store`, `store.clone()`) also
    /// registers `s2` as a store alias (N6); anything else using a store
    /// ident is an escape, reported by `smart_system`, not here.
    ///
    /// Stays combinator-shaped on purpose: the IOSP gate flags branching
    /// statements mixed with calls in one function, so the classifiers below
    /// stay pure and this one only orchestrates.
    fn handle_local(&mut self, ident: &Ident, init: &Expr) {
        // `let store = resources.get::<SmartStore>().expect(..)`.
        let _ = self
            .store_alias_source(init)
            .then(|| self.store_aliases.push(ident.clone()));
        let _ = self.lane_binding(init).map(|(is_mutable, _)| {
            self.lanes.push(LaneBinding {
                var_name: ident.clone(),
                is_mutable,
            })
        });
        let _ = self
            .store_realias_source(init)
            .map(|_| self.store_aliases.push(ident.clone()));
    }

    /// Whether `ident` is the store parameter or a known store alias.
    fn is_store_ident(&self, ident: &Ident) -> bool {
        self.store_param.as_ref() == Some(ident)
            || self.store_aliases.iter().any(|alias| alias == ident)
    }

    /// A plain re-alias of a known store ident: `store`, `&store`,
    /// `&mut store`, `store.clone()`. Returns the aliased store ident.
    /// Integration: no branching, only the pure shape test plus the
    /// membership check below.
    fn store_realias_source(&self, expr: &Expr) -> Option<Ident> {
        realias_candidate(Self::strip_wrappers(expr))
            .filter(|ident| self.is_store_ident(ident))
            .cloned()
    }
}

/// Pure shape of a store re-alias target: a bare ident, `&ident`, or
/// `ident.clone()`. Operation: match and comparisons only, no project calls
/// (the wrapper stripping and membership check stay with the caller above).
fn realias_candidate(expr: &Expr) -> Option<&Ident> {
    match expr {
        Expr::Path(path) => path.path.get_ident(),
        Expr::Reference(reference) => match &*reference.expr {
            Expr::Path(path) => path.path.get_ident(),
            _ => None,
        },
        Expr::MethodCall(mc) if mc.method == "clone" && mc.args.is_empty() => match &*mc.receiver {
            Expr::Path(path) => path.path.get_ident(),
            _ => None,
        },
        _ => None,
    }
}

/// Std macros whose arguments are plain comma-separated expressions, so the
/// collector can derive accesses from inside them (N1): `assert!`,
/// `debug_assert*!`, `println!`/`eprintln!`/`format!`/`write!`/`writeln!`,
/// `panic!`, `dbg!`. Any other macro invocation hides its contents: passing
/// `resources` or a store ident into one is a compile error in
/// `smart_system` (newtypes/aliases around `Mutex` are likewise invisible —
/// declare them explicitly).
pub(crate) const DERIVABLE_MACROS: &[&str] = &[
    "assert",
    "assert_eq",
    "assert_ne",
    "debug_assert",
    "debug_assert_eq",
    "debug_assert_ne",
    "println",
    "eprintln",
    "format",
    "write",
    "writeln",
    "panic",
    "dbg",
    "unreachable",
    "unimplemented",
    "todo",
];

/// Last segment of a macro path (`assert`, `std::assert` → `assert`).
pub(crate) fn macro_name(mac: &syn::Macro) -> Option<String> {
    mac.path.segments.last().map(|s| s.ident.to_string())
}

/// Parses a macro invocation as comma-separated expressions, if possible.
pub(crate) fn macro_exprs(
    mac: &syn::Macro,
) -> Option<syn::punctuated::Punctuated<Expr, syn::Token![,]>> {
    mac.parse_body_with(punctuated_exprs).ok()
}

/// `parse_body_with` needs a higher-ranked function; a closure infers a
/// concrete lifetime and is rejected, so this stays a free function.
fn punctuated_exprs(
    input: syn::parse::ParseStream,
) -> syn::Result<syn::punctuated::Punctuated<Expr, syn::Token![,]>> {
    syn::punctuated::Punctuated::parse_terminated(input)
}

/// Whether the macro tokens mention any of the idents (used to reject
/// accesses hidden inside opaque macros).
pub(crate) fn macro_mentions_ident(mac: &syn::Macro, idents: &[&Ident]) -> bool {
    let tokens = mac.tokens.to_string();
    idents.iter().any(|ident| {
        let name = ident.to_string();
        tokens
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .any(|tok| tok == name)
    })
}

impl Visit<'_> for LaneCollector {
    fn visit_item_fn(&mut self, node: &ItemFn) {
        for arg in &node.sig.inputs {
            if let FnArg::Typed(PatType { pat, ty, .. }) = arg
                && let Pat::Ident(PatIdent { ident, .. }) = &**pat
            {
                match classify_param(ty) {
                    ParamKind::SmartStore => {
                        self.store_param = Some(ident.clone());
                    }
                    ParamKind::Resources => {
                        self.resources_param = Some(ident.clone());
                    }
                    ParamKind::Other => {}
                }
            }
        }
        self.visit_block(&node.block);
    }

    fn visit_local(&mut self, node: &Local) {
        // Combinator-shaped on purpose (IOSP gate): `binding_ident` covers
        // `ident`, `Type ident`, `Some(ident)` / `Ok(ident)`; anything else
        // cannot drive loop rewriting (its lane accesses still count via
        // `visit_expr_method_call`).
        let _ = node
            .init
            .as_ref()
            .zip(binding_ident(&node.pat))
            .map(|(init, ident)| self.handle_local(&ident, &init.expr));
        visit::visit_local(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &ExprMethodCall) {
        if (node.method == "read_lane" || node.method == "write_lane")
            && self.is_store_receiver(&node.receiver)
        {
            match turbofish_type(node, self.macro_name) {
                Err(error) => self.errors.push(error),
                Ok(ty) => self.lane_accesses.push((ty, node.method == "write_lane")),
            }
        }
        // Every `resources.get::<T>()` reads a resource, in any context
        // (dedup happens at emission by type string).
        if let Some(ty) = self.resources_read_here(node) {
            self.resource_reads.push(ty);
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_macro(&mut self, node: &syn::Macro) {
        // Known std macros hold plain expressions: derive accesses from
        // inside them (N1) via a sub-collector seeded with the current
        // store context (parsed expressions are temporaries, so they cannot
        // be visited with `self` directly). Unknown macros stay opaque;
        // `smart_system` rejects `resources`/store idents hidden in them,
        // `smart_pipeline` keeps ignoring them.
        let known = macro_name(node).is_some_and(|name| DERIVABLE_MACROS.contains(&name.as_str()));
        if known && let Some(exprs) = macro_exprs(node) {
            let mut sub = LaneCollector {
                macro_name: self.macro_name,
                store_param: self.store_param.clone(),
                resources_param: self.resources_param.clone(),
                store_aliases: self.store_aliases.clone(),
                ..LaneCollector::default()
            };
            for expr in &exprs {
                sub.visit_expr(expr);
            }
            // Expression position holds no `let` bindings, so no new lanes
            // or aliases can appear; only accesses and errors merge back.
            self.lane_accesses.extend(sub.lane_accesses);
            self.resource_reads.extend(sub.resource_reads);
            self.errors.extend(sub.errors);
            return;
        }
        visit::visit_macro(self, node);
    }
}

/// Collects idents bound by a loop/closure pattern (`x`, `(a, b)`, `&mut x`).
fn collect_pat_idents(pat: &Pat, out: &mut Vec<Ident>) {
    match pat {
        Pat::Ident(p) => out.push(p.ident.clone()),
        Pat::Tuple(t) => {
            for elem in &t.elems {
                collect_pat_idents(elem, out);
            }
        }
        Pat::Reference(r) => collect_pat_idents(&r.pat, out),
        _ => {}
    }
}

/// Safety analysis of a single loop body: collects reasons why the loop
/// cannot be executed in parallel. Conservative by design — a false positive
/// only means the loop stays sequential.
struct LoopBodyAnalyzer {
    macro_name: &'static str,
    /// Idents that may legally be assigned inside the body: loop pattern
    /// variables, body-local `let` bindings, closure params, nested loop vars.
    assignable: Vec<Ident>,
    /// Loop pattern variables (for cross-iteration index detection).
    loop_vars: Vec<Ident>,
    closure_depth: usize,
    loop_depth: usize,
    issues: Vec<String>,
}

impl LoopBodyAnalyzer {
    fn new(macro_name: &'static str, loop_vars: Vec<Ident>) -> Self {
        Self {
            macro_name,
            assignable: loop_vars.clone(),
            loop_vars,
            closure_depth: 0,
            loop_depth: 0,
            issues: Vec::new(),
        }
    }

    fn check_assign_target(&mut self, target: &Expr) {
        if let Expr::Path(p) = target
            && let Some(ident) = p.path.get_ident()
            && !self.assignable.contains(ident)
        {
            self.issues.push(format!(
                "{}: variable `{ident}` is assigned inside the loop but declared \
                 outside it - shared mutable state prevents parallelization; loop left sequential",
                self.macro_name
            ));
        }
    }
}

impl Visit<'_> for LoopBodyAnalyzer {
    fn visit_local(&mut self, node: &Local) {
        if let Pat::Ident(PatIdent { ident, .. }) = &node.pat {
            self.assignable.push(ident.clone());
        }
        visit::visit_local(self, node);
    }

    fn visit_expr_closure(&mut self, node: &ExprClosure) {
        let before = self.assignable.len();
        for input in &node.inputs {
            collect_pat_idents(input, &mut self.assignable);
        }
        self.closure_depth += 1;
        visit::visit_expr_closure(self, node);
        self.closure_depth -= 1;
        self.assignable.truncate(before);
    }

    fn visit_expr_for_loop(&mut self, node: &ExprForLoop) {
        let before = self.assignable.len();
        collect_pat_idents(&node.pat, &mut self.assignable);
        self.loop_depth += 1;
        visit::visit_expr_for_loop(self, node);
        self.loop_depth -= 1;
        self.assignable.truncate(before);
    }

    fn visit_expr_assign(&mut self, node: &ExprAssign) {
        self.check_assign_target(&node.left);
        visit::visit_expr_assign(self, node);
    }

    fn visit_expr_binary(&mut self, node: &ExprBinary) {
        // Compound assignments (`+=`, `-=`, ...) are `Expr::Binary` in syn 2.
        if matches!(
            node.op,
            BinOp::AddAssign(_)
                | BinOp::SubAssign(_)
                | BinOp::MulAssign(_)
                | BinOp::DivAssign(_)
                | BinOp::RemAssign(_)
                | BinOp::BitXorAssign(_)
                | BinOp::BitAndAssign(_)
                | BinOp::BitOrAssign(_)
                | BinOp::ShlAssign(_)
                | BinOp::ShrAssign(_)
        ) {
            self.check_assign_target(&node.left);
        }
        visit::visit_expr_binary(self, node);
    }

    fn visit_expr_index(&mut self, node: &ExprIndex) {
        if let Expr::Path(expr_path) = &*node.expr
            && let Some(ident) = expr_path.path.get_ident()
            && self.loop_vars.contains(ident)
            && let Expr::Binary(binary) = &*node.index
            && matches!(binary.op, BinOp::Add(_) | BinOp::Sub(_))
        {
            self.issues.push(format!(
                "{}: cross-iteration dependency `{ident}[{}]` prevents \
                 parallelization; loop left sequential",
                self.macro_name,
                binary.to_token_stream()
            ));
        }
        visit::visit_expr_index(self, node);
    }

    fn visit_expr_break(&mut self, node: &syn::ExprBreak) {
        // `break`/`continue` of a nested loop or a closure is fine; only
        // control flow targeting the analyzed loop itself (or a label)
        // blocks the rewrite.
        if node.label.is_some() || (self.closure_depth == 0 && self.loop_depth == 0) {
            self.issues.push(format!(
                "{}: `break` in the loop body prevents parallelization; loop \
                 left sequential",
                self.macro_name
            ));
        }
        visit::visit_expr_break(self, node);
    }

    fn visit_expr_continue(&mut self, node: &syn::ExprContinue) {
        if node.label.is_some() || (self.closure_depth == 0 && self.loop_depth == 0) {
            self.issues.push(format!(
                "{}: `continue` in the loop body prevents parallelization; loop \
                 left sequential",
                self.macro_name
            ));
        }
        visit::visit_expr_continue(self, node);
    }

    fn visit_expr_return(&mut self, node: &syn::ExprReturn) {
        if self.closure_depth == 0 {
            self.issues.push(format!(
                "{}: `return` in the loop body prevents parallelization; loop \
                 left sequential",
                self.macro_name
            ));
        }
        visit::visit_expr_return(self, node);
    }
}

/// A lane iterator used by a `for` loop header.
struct LaneIter {
    var_name: Ident,
    mutable: bool,
}

/// Rewrites parallel-safe `for` loops over lane iterators in place, leaving
/// everything else untouched.
pub(crate) struct LoopRewriter<'a> {
    pub(crate) macro_name: &'static str,
    pub(crate) lanes: &'a [LaneBinding],
    pub(crate) warnings: Vec<String>,
}

impl LoopRewriter<'_> {
    fn find_lane(&self, ident: &Ident) -> Option<&LaneBinding> {
        self.lanes.iter().find(|lane| lane.var_name == *ident)
    }

    /// `lane.iter()` / `lane.iter_mut()` where `lane` is a lane binding.
    fn single_lane_iter(&self, expr: &Expr) -> Option<LaneIter> {
        let Expr::MethodCall(mc) = expr else {
            return None;
        };
        if mc.method != "iter" && mc.method != "iter_mut" {
            return None;
        }
        if !mc.args.is_empty() {
            return None;
        }
        let Expr::Path(p) = &*mc.receiver else {
            return None;
        };
        let ident = p.path.get_ident()?;
        let lane = self.find_lane(ident)?;
        Some(LaneIter {
            var_name: ident.clone(),
            mutable: mc.method == "iter_mut" && lane.is_mutable,
        })
    }

    /// The iterator expression of a loop: either a single lane iterator or
    /// `lane_a.iter*().zip(lane_b.iter*())`.
    fn extract_lane_iters(&self, expr: &Expr) -> Option<Vec<LaneIter>> {
        if let Expr::MethodCall(mc) = expr
            && mc.method == "zip"
            && mc.args.len() == 1
        {
            let mut iters = vec![self.single_lane_iter(&mc.receiver)?];
            iters.push(self.single_lane_iter(&mc.args[0])?);
            return Some(iters);
        }
        self.single_lane_iter(expr).map(|iter| vec![iter])
    }

    /// Returns the replacement expression for a parallel-safe loop, or the
    /// list of reasons why it must stay sequential.
    fn plan(&self, node: &ExprForLoop) -> Result<Expr, Vec<String>> {
        let mut issues = Vec::new();

        if node.label.is_some() {
            issues.push(format!(
                "{}: labeled loop left sequential (labels cannot be rewritten \
                 into a Rayon closure)",
                self.macro_name
            ));
        }

        let mut loop_vars = Vec::new();
        collect_pat_idents(&node.pat, &mut loop_vars);
        if loop_vars.is_empty() {
            issues.push(format!(
                "{}: loop pattern binds no variables; loop left sequential",
                self.macro_name
            ));
        }

        let iters = self.extract_lane_iters(&node.expr);
        if iters.is_none() {
            issues.push(format!(
                "{}: cannot parallelize loop over `{}` - expected \
                 `lane.iter()`/`lane.iter_mut()`, optionally two lanes combined with \
                 `.zip(..)`; loop left sequential",
                self.macro_name,
                node.expr.to_token_stream()
            ));
        }

        let mut analyzer = LoopBodyAnalyzer::new(self.macro_name, loop_vars);
        analyzer.visit_block(&node.body);
        issues.extend(analyzer.issues);

        let iters = match iters {
            Some(iters) if issues.is_empty() => iters,
            _ => return Err(issues),
        };

        // The loop runs inside `__ornis_lane_must_be_send_sync`, whose
        // `T: Send + Sync` bound is assumed while `par_iter` is checked.
        // A lane that fails the bound errors at this call only.
        parallel_loop_expr(&iters, &node.pat, &node.body)
    }
}

/// Borrow of the lane guard passed into the parallel helper.
fn lane_ref(iter: &LaneIter) -> TokenStream2 {
    let var = &iter.var_name;
    if iter.mutable {
        quote!(&mut #var)
    } else {
        quote!(&#var)
    }
}

/// Store reference the helper takes. `param` is the lane's type parameter.
fn store_ref(mutable: bool, param: &Ident) -> TokenStream2 {
    if mutable {
        quote!(&mut ornis_core::ComponentStore<#param>)
    } else {
        quote!(&ornis_core::ComponentStore<#param>)
    }
}

/// Item type `par_iter` / `par_iter_mut` yields for `param`.
fn item_ty(mutable: bool, param: &Ident) -> TokenStream2 {
    if mutable {
        quote!(&mut #param)
    } else {
        quote!(&#param)
    }
}

/// Parallel replacement for one or two lane iterators.
///
/// `par_iter` is type-checked only inside `__ornis_lane_must_be_send_sync`,
/// where `Send + Sync` already holds. The call site is what fails when a
/// lane type does not.
fn parallel_loop_expr(
    iters: &[LaneIter],
    pat: &Pat,
    body: &syn::Block,
) -> Result<Expr, Vec<String>> {
    match iters {
        [single] => Ok(single_lane_loop(single, pat, body)),
        [first, second] => Ok(zip_lane_loop(first, second, pat, body)),
        _ => Err(vec![
            "#[smart_pipeline]: more than two zip lanes is unsupported".to_string(),
        ]),
    }
}

/// Audit §3.3, backlog #7: capture the TLS access frame before the parallel
/// section and install it in each task. An empty snapshot (outside
/// `Schedule::run`) is a no-op.
fn single_lane_loop(iter: &LaneIter, pat: &Pat, body: &syn::Block) -> Expr {
    let lane = lane_ref(iter);
    let method = par_iter_method(iter.mutable);
    let param = format_ident!("T");
    let store = store_ref(iter.mutable, &param);
    let item = item_ty(iter.mutable, &param);
    syn::parse_quote! {{
        fn __ornis_lane_must_be_send_sync<T, F>(__ornis_lane: #store, __ornis_body: F)
        where
            T: Send + Sync,
            F: Fn(#item) + Send + Sync,
        {
            use ornis_core::rayon::prelude::*;
            let __ornis_access_frame = ornis_core::schedule::capture_access_frame();
            __ornis_lane.#method().for_each(|__ornis_item| {
                let _ornis_frame_guard = __ornis_access_frame.install();
                __ornis_body(__ornis_item);
            });
        }
        __ornis_lane_must_be_send_sync(#lane, |#pat| #body);
    }}
}

/// Two-lane `zip`, same frame capture as [`single_lane_loop`].
fn zip_lane_loop(first: &LaneIter, second: &LaneIter, pat: &Pat, body: &syn::Block) -> Expr {
    let lane0 = lane_ref(first);
    let lane1 = lane_ref(second);
    let method0 = par_iter_method(first.mutable);
    let method1 = par_iter_method(second.mutable);
    let a = format_ident!("A");
    let b = format_ident!("B");
    let store0 = store_ref(first.mutable, &a);
    let store1 = store_ref(second.mutable, &b);
    let item0 = item_ty(first.mutable, &a);
    let item1 = item_ty(second.mutable, &b);
    syn::parse_quote! {{
        fn __ornis_lane_must_be_send_sync<A, B, F>(
            __ornis_lane0: #store0,
            __ornis_lane1: #store1,
            __ornis_body: F,
        ) where
            A: Send + Sync,
            B: Send + Sync,
            F: Fn((#item0, #item1)) + Send + Sync,
        {
            use ornis_core::rayon::prelude::*;
            let __ornis_access_frame = ornis_core::schedule::capture_access_frame();
            __ornis_lane0.#method0().zip(__ornis_lane1.#method1()).for_each(
                |__ornis_item| {
                    let _ornis_frame_guard = __ornis_access_frame.install();
                    __ornis_body(__ornis_item);
                },
            );
        }
        __ornis_lane_must_be_send_sync(#lane0, #lane1, |#pat| #body);
    }}
}

fn par_iter_method(mutable: bool) -> Ident {
    if mutable {
        format_ident!("par_iter_mut")
    } else {
        format_ident!("par_iter")
    }
}

impl VisitMut for LoopRewriter<'_> {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::ForLoop(for_loop) = expr {
            match self.plan(for_loop) {
                Ok(replacement) => {
                    *expr = replacement;
                    // The loop body moved into the closure; nested loops in it
                    // may still be rewritten.
                    visit_mut::visit_expr_mut(self, expr);
                    return;
                }
                Err(issues) => {
                    // The loop stays an ordinary sequential `for` — header and
                    // body untouched. Nested loops may still be rewritten.
                    self.warnings.extend(issues);
                }
            }
        }
        visit_mut::visit_expr_mut(self, expr);
    }
}

/// Shared emission helpers: error accumulation, sequential-loop warnings,
/// and the `pipeline_enter` / `pipeline_exit` body wrapper.
/// Merge errors into one diagnostic (bridge-free core, unit-testable).
pub(crate) fn combine_errors_tokens(errors: Vec<syn::Error>) -> TokenStream2 {
    let mut errors = errors.into_iter();
    let Some(mut combined) = errors.next() else {
        return TokenStream2::new();
    };
    for error in errors {
        combined.combine(error);
    }
    combined.to_compile_error()
}

/// Emits the deprecated-note warning blocks for loops left sequential. The
/// marker struct is named after the macro so the warning points back at the
/// right attribute (`SmartSystemSequentialLoop` for `smart_system`).
pub(crate) fn warning_tokens(warnings: &[String], macro_name: &'static str) -> Vec<TokenStream2> {
    let marker = if macro_name == "#[smart_system]" {
        format_ident!("SmartSystemSequentialLoop")
    } else {
        format_ident!("SmartPipelineSequentialLoop")
    };
    warnings
        .iter()
        .map(|w| {
            quote! {{
                #[deprecated(note = #w)]
                struct #marker;
                let _ = #marker;
            }}
        })
        .collect()
}

/// Wraps rewritten statements with the profiling enter/exit hooks.
pub(crate) fn wrap_body(stmts: &[syn::Stmt]) -> TokenStream2 {
    quote! {
        // The body runs inside a block so the hook below also fires for
        // functions with a tail expression; `return` still exits early
        // (the exit hook is a no-op profiling marker).
        #[allow(clippy::let_unit_value)]
        let smart_pipeline_result = { #(#stmts)* };
    }
}

/// R6: static `Send + Sync` assertion over lane access types (shared with
/// `smart_system`). Empty when no lanes are touched.
pub(crate) fn send_sync_assert_tokens(lane_accesses: &[(Type, bool)]) -> TokenStream2 {
    let mut seen: Vec<String> = Vec::new();
    let mut lane_tys: Vec<&Type> = Vec::new();
    for (ty, _) in lane_accesses {
        let name = ty.to_token_stream().to_string();
        if !seen.contains(&name) {
            seen.push(name);
            lane_tys.push(ty);
        }
    }
    if lane_tys.is_empty() {
        TokenStream2::new()
    } else {
        quote! {
            {
                fn __ornis_lane_must_be_send_sync<T: Send + Sync>() {}
                #(__ornis_lane_must_be_send_sync::<#lane_tys>();)*
            }
        }
    }
}

/// R3: sorted, deduplicated `(type, is_write)` access entries over lane
/// accesses (shared with `smart_system`). A write covers a read of the same
/// lane, matching the scheduler's "own write covers read" rule. The
/// dedup/sort key is the rendered type string, so the expansion is
/// deterministic.
pub(crate) fn access_entries(accesses: &[(Type, bool)]) -> Vec<(Type, bool)> {
    let mut entries: Vec<(Type, bool)> = Vec::new();
    for (ty, write) in accesses {
        let ty_name = ty.to_token_stream().to_string();
        match entries
            .iter_mut()
            .find(|(kept, _)| kept.to_token_stream().to_string() == ty_name)
        {
            Some(entry) => entry.1 |= write,
            None => entries.push((ty.clone(), *write)),
        }
    }
    entries.sort_by(|a, b| {
        a.0.to_token_stream()
            .to_string()
            .cmp(&b.0.to_token_stream().to_string())
    });
    entries
}

pub fn attribute(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as ItemFn);

    let mut collector = LaneCollector::new("#[smart_pipeline]");
    collector.visit_item_fn(&input);

    if !collector.errors.is_empty() {
        return combine_errors_tokens(collector.errors).into();
    }

    let mut rewriter = LoopRewriter {
        macro_name: "#[smart_pipeline]",
        lanes: &collector.lanes,
        warnings: Vec::new(),
    };
    rewriter.visit_block_mut(&mut input.block);

    // Compile-time warnings via the deprecated-note trick: using a deprecated
    // item emits a warning with the note, visible in the IDE and terminal.
    let warning_tokens = warning_tokens(&rewriter.warnings, "#[smart_pipeline]");

    let attrs = &input.attrs;
    let vis = &input.vis;
    let sig = &input.sig;
    let body = wrap_body(&input.block.stmts);

    // R6: every lane type must be `Send + Sync` (parallel iteration moves
    // lane data across Rayon threads). The store API already requires the
    // same bounds; asserting here points the diagnostic at the annotated
    // system instead of deep generic internals.
    let send_sync_assert = send_sync_assert_tokens(&collector.lane_accesses);

    // R3: export the access set derived from the lane bindings as a
    // module-scope constant next to the function: sorted `(type, is_write)`
    // pairs, so the expansion is deterministic and machine-checkable.
    // Wiring the comparison against `System::access()` at registration is
    // done by `smart_system`; the extraction (the hard, desync-prone half)
    // lives here.
    let access_const_name = format_ident!("__SMART_PIPELINE_ACCESS_{}", input.sig.ident);
    let entries = access_entries(&collector.lane_accesses);
    let access_tys: Vec<syn::LitStr> = entries
        .iter()
        .map(|(ty, _)| {
            syn::LitStr::new(
                &ty.to_token_stream().to_string(),
                proc_macro2::Span::call_site(),
            )
        })
        .collect();
    let access_writes: Vec<bool> = entries.iter().map(|(_, w)| *w).collect();
    let access_const = quote! {
        #[allow(dead_code, non_upper_case_globals)]
        const #access_const_name: &[(&str, bool)] = &[
            #((#access_tys, #access_writes)),*
        ];
    };

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            ornis_core::pipeline_enter();
            #send_sync_assert
            #(#warning_tokens)*
            #body
            ornis_core::pipeline_exit();
            smart_pipeline_result
        }
        #access_const
    };

    expanded.into()
}

#[cfg(test)]
mod collector_tests {
    use super::*;
    use quote::ToTokens;

    fn parse_ty(src: &str) -> Type {
        syn::parse_str(src).expect("type parses")
    }

    fn parse_pat(src: &str) -> Pat {
        // Wrap in a `let` so any pattern form parses (`Local` itself is
        // not `Parse`; go through `Stmt`).
        let stmt: syn::Stmt = syn::parse_str(&format!("let {src} = x;")).expect("let parses");
        match stmt {
            syn::Stmt::Local(local) => local.pat.clone(),
            _ => panic!("expected a `let` statement"),
        }
    }

    #[test]
    fn classify_param_sorts_references() {
        assert_eq!(
            classify_param(&parse_ty("&SmartStore")),
            ParamKind::SmartStore
        );
        assert_eq!(
            classify_param(&parse_ty("&Resources")),
            ParamKind::Resources
        );
        assert_eq!(classify_param(&parse_ty("&Velocity")), ParamKind::Other);
        assert_eq!(classify_param(&parse_ty("Velocity")), ParamKind::Other);
        assert_eq!(
            classify_param(&parse_ty("&mut SmartStore")),
            ParamKind::SmartStore
        );
    }

    #[test]
    fn binding_ident_accepts_documented_forms() {
        assert_eq!(binding_ident(&parse_pat("pos")).unwrap().to_string(), "pos");
        // A type ascription is Pat::Type and unwraps to the inner ident.
        assert_eq!(
            binding_ident(&parse_pat("vel: u32")).unwrap().to_string(),
            "vel"
        );
        assert_eq!(
            binding_ident(&parse_pat("Some(pos)")).unwrap().to_string(),
            "pos"
        );
        assert_eq!(binding_ident(&parse_pat("Ok(v)")).unwrap().to_string(), "v");
        assert!(binding_ident(&parse_pat("(a, b)")).is_none());
    }

    #[test]
    fn access_entries_dedups_sorts_and_folds_writes() {
        let read_v: Type = parse_ty("V");
        let write_p: Type = parse_ty("P");
        let entries = access_entries(&[
            (write_p.clone(), true),
            (read_v.clone(), false),
            (parse_ty("P"), false),
            (parse_ty("A"), false),
        ]);
        let rendered: Vec<(String, bool)> = entries
            .iter()
            .map(|(ty, w)| (ty.to_token_stream().to_string(), *w))
            .collect();
        assert_eq!(
            rendered,
            vec![
                ("A".to_string(), false),
                ("P".to_string(), true),
                ("V".to_string(), false),
            ]
        );
    }

    #[test]
    fn let_else_binding_parallelizes_without_warnings() {
        let mut item: ItemFn = syn::parse_str(
            "fn f(store: &SmartStore) { let Some(mut xs) = store.write_lane::<P>() else { return; }; for x in xs.iter_mut() { x.x += 1.0; } }",
        )
        .expect("fn parses");
        let mut collector = LaneCollector::new("#[smart_pipeline]");
        collector.visit_item_fn(&item);
        assert!(collector.errors.is_empty());
        assert_eq!(collector.lanes.len(), 1);
        let mut rewriter = LoopRewriter {
            macro_name: "#[smart_pipeline]",
            lanes: &collector.lanes,
            warnings: Vec::new(),
        };
        rewriter.visit_block_mut(&mut item.block);
        assert!(rewriter.warnings.is_empty(), "{:?}", rewriter.warnings);
        let rendered = item.block.to_token_stream().to_string();
        assert!(rendered.contains("for_each"), "{rendered}");
    }

    #[test]
    fn rewriter_warning_uses_macro_name() {
        let mut item: ItemFn = syn::parse_str(
            "fn f(store: &SmartStore) { let mut xs = store.write_lane::<P>().unwrap(); for x in xs.iter_mut() { if x.x > 0.0 { break; } } }",
        )
        .expect("fn parses");
        let mut collector = LaneCollector::new("#[smart_system]");
        collector.visit_item_fn(&item);
        assert!(collector.errors.is_empty());
        let mut rewriter = LoopRewriter {
            macro_name: "#[smart_system]",
            lanes: &collector.lanes,
            warnings: Vec::new(),
        };
        rewriter.visit_block_mut(&mut item.block);
        assert_eq!(rewriter.warnings.len(), 1);
        assert!(
            rewriter.warnings[0].starts_with("#[smart_system]:"),
            "got: {}",
            rewriter.warnings[0]
        );
    }
}

#[cfg(test)]
mod collector_unit_tests {
    use super::*;

    fn parse_expr(src: &str) -> Expr {
        syn::parse_str(src).expect("expr parses")
    }

    fn collector_with_store() -> LaneCollector {
        let mut collector = LaneCollector::new("#[smart_pipeline]");
        collector.store_param = Some(syn::parse_str("store").expect("ident"));
        collector.resources_param = Some(syn::parse_str("resources").expect("ident"));
        collector
    }

    #[test]
    fn strip_wrappers_peels_only_unwrap_expect() {
        let chained = parse_expr("store.read_lane::<P>().expect(\"p\").len()");
        assert!(matches!(
            LaneCollector::strip_wrappers(&chained),
            Expr::MethodCall(mc) if mc.method == "len"
        ));
        let wrapped = parse_expr("store.read_lane::<P>().expect(\"p\")");
        assert!(matches!(
            LaneCollector::strip_wrappers(&wrapped),
            Expr::MethodCall(mc) if mc.method == "read_lane"
        ));
        assert!(matches!(
            LaneCollector::strip_wrappers(&parse_expr("x")),
            Expr::Path(_)
        ));
    }

    #[test]
    fn as_method_call_matches_only_calls() {
        assert!(as_method_call(&parse_expr("a.b()")).is_some());
        assert!(as_method_call(&parse_expr("x")).is_none());
    }

    #[test]
    fn lane_binding_reads_and_writes() {
        let collector = collector_with_store();
        let (write, ty) = collector
            .lane_binding(&parse_expr("store.write_lane::<P>().expect(\"p\")"))
            .expect("write lane");
        assert!(write);
        assert_eq!(ty.to_token_stream().to_string(), "P");
        let (write, _) = collector
            .lane_binding(&parse_expr("store.read_lane::<V>()"))
            .expect("read lane");
        assert!(!write);
        assert!(
            collector
                .lane_binding(&parse_expr("other.read_lane::<V>()"))
                .is_none()
        );
        assert!(collector.lane_binding(&parse_expr("store.len()")).is_none());
    }

    #[test]
    fn store_alias_source_needs_smart_store_get() {
        let collector = collector_with_store();
        assert!(
            collector
                .store_alias_source(&parse_expr("resources.get::<SmartStore>().expect(\"s\")"))
        );
        assert!(!collector.store_alias_source(&parse_expr("resources.get::<Cfg>()")));
        assert!(!collector.store_alias_source(&parse_expr("store.read_lane::<P>()")));
    }

    #[test]
    fn is_store_receiver_covers_param_and_aliases() {
        let mut collector = collector_with_store();
        collector
            .store_aliases
            .push(syn::parse_str("cache").expect("ident"));
        let check = |collector: &LaneCollector, src: &str| {
            let Expr::MethodCall(mc) = parse_expr(src) else {
                panic!("expected call");
            };
            collector.is_store_receiver(&mc.receiver)
        };
        assert!(check(&collector, "store.read_lane::<P>()"));
        assert!(!check(&collector, "other.read_lane::<P>()"));
        assert!(check(&collector, "cache.read_lane::<P>()"));
    }

    #[test]
    fn resources_read_here_needs_turbofish_get() {
        let collector = collector_with_store();
        let read = |src: &str| {
            let Expr::MethodCall(mc) = parse_expr(src) else {
                panic!("expected call");
            };
            collector.resources_read_here(&mc)
        };
        assert!(read("resources.get::<Cfg>()").is_some());
        assert!(read("resources.contains::<Cfg>()").is_some());
        assert!(read("resources.get()").is_none());
        assert!(read("resources.len()").is_none());
    }

    #[test]
    fn macro_helpers_classify_invocations() {
        fn invocation(src: &str) -> syn::Macro {
            match syn::parse_str::<syn::Stmt>(src).expect("stmt parses") {
                syn::Stmt::Macro(stmt) => stmt.mac,
                _ => panic!("expected a macro statement"),
            }
        }
        let known = invocation("assert!(resources.get::<Cfg>().is_some());");
        assert_eq!(macro_name(&known).as_deref(), Some("assert"));
        let exprs = macro_exprs(&known).expect("known macros parse as exprs");
        assert_eq!(exprs.len(), 1);
        let resources: Ident = syn::parse_str("resources").expect("ident");
        let other: Ident = syn::parse_str("other").expect("ident");
        assert!(macro_mentions_ident(&known, &[&resources]));
        assert!(!macro_mentions_ident(&known, &[&other]));

        let unknown = invocation("my_macro!(resources);");
        assert_eq!(macro_name(&unknown).as_deref(), Some("my_macro"));
        assert!(macro_mentions_ident(&unknown, &[&resources]));
        assert!(!macro_mentions_ident(&unknown, &[&other]));
    }

    #[test]
    fn realias_candidate_shapes() {
        assert_eq!(
            realias_candidate(&parse_expr("store")).unwrap().to_string(),
            "store"
        );
        assert_eq!(
            realias_candidate(&parse_expr("&store"))
                .unwrap()
                .to_string(),
            "store"
        );
        assert_eq!(
            realias_candidate(&parse_expr("store.clone()"))
                .unwrap()
                .to_string(),
            "store"
        );
        assert!(realias_candidate(&parse_expr("store.len()")).is_none());
        assert!(realias_candidate(&parse_expr("other")).is_some());
    }

    #[test]
    fn len_chain_gives_access_but_no_binding() {
        // `let n = store.read_lane::<V>().unwrap().len()` is a derived value,
        // not a lane guard: no `LaneBinding` (nothing to rewrite), but the
        // lane access still counts.
        let item: ItemFn =
            syn::parse_str("fn f(store: &SmartStore) { let n = store.read_lane::<V>().unwrap().len(); let _ = n; }")
                .expect("fn parses");
        let mut collector = LaneCollector::new("#[smart_pipeline]");
        collector.visit_item_fn(&item);
        assert!(collector.errors.is_empty());
        assert!(collector.lanes.is_empty());
        assert_eq!(collector.lane_accesses.len(), 1);
    }

    #[test]
    fn combine_errors_merges_and_empties() {
        let empty = combine_errors_tokens(Vec::new());
        assert!(empty.to_string().is_empty());
        let err = combine_errors_tokens(vec![
            syn::Error::new_spanned(syn::parse_str::<Ident>("x").expect("ident"), "first"),
            syn::Error::new_spanned(syn::parse_str::<Ident>("y").expect("ident"), "second"),
        ]);
        let rendered = err.to_string();
        assert!(
            rendered.contains("first") && rendered.contains("second"),
            "{rendered}"
        );
    }

    #[test]
    fn warning_tokens_wrap_each_message() {
        assert!(warning_tokens(&[], "#[smart_pipeline]").is_empty());
        let tokens = warning_tokens(&["left sequential".to_string()], "#[smart_system]");
        let rendered: TokenStream2 = tokens.into_iter().collect::<TokenStream2>();
        let rendered = rendered.to_string();
        assert!(rendered.contains("left sequential"));
        assert!(rendered.contains("SmartSystemSequentialLoop"));
    }

    #[test]
    fn wrap_body_preserves_tail_value() {
        let stmts: Vec<syn::Stmt> = vec![syn::parse_str("let x = 1;").expect("stmt")];
        let rendered = wrap_body(&stmts).to_string();
        assert!(rendered.contains("smart_pipeline_result"));
    }

    #[test]
    fn send_sync_assert_covers_each_lane_once() {
        assert!(send_sync_assert_tokens(&[]).to_string().is_empty());
        let pos: Type = syn::parse_str("Pos").expect("type");
        let tokens = send_sync_assert_tokens(&[(pos, true)]).to_string();
        assert!(tokens.contains("__ornis_lane_must_be_send_sync"));
        assert!(tokens.contains("Pos"));
    }
}
