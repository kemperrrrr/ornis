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

/// A `let` binding of a `SmartStore` lane guard.
struct LaneBinding {
    var_name: Ident,
    is_mutable: bool,
}

/// Extracts the single turbofish type argument of `read_lane::<T>()` /
/// `write_lane::<T>()`. The turbofish lives in [`ExprMethodCall::turbofish`],
/// not in `args`.
fn turbofish_type(node: &ExprMethodCall) -> Result<Type, syn::Error> {
    let error = || {
        syn::Error::new_spanned(
            node,
            format!(
                "#[smart_pipeline]: `{}` requires exactly one turbofish type argument, \
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
        let ty = turbofish_type(&mc).expect("turbofish present");
        assert_eq!(ty.to_token_stream().to_string(), "Position");
    }

    #[test]
    fn turbofish_write_lane_works() {
        let mc = method_call("store.write_lane::<Vec3>()");
        let ty = turbofish_type(&mc).expect("ok");
        assert_eq!(ty.to_token_stream().to_string(), "Vec3");
    }

    #[test]
    fn turbofish_missing_is_error() {
        let mc = method_call("store.read_lane()");
        match turbofish_type(&mc) {
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
            assert!(turbofish_type(&mc).is_err());
        } else {
            panic!("expected method call");
        }
    }

    #[test]
    fn two_arguments_are_rejected() {
        let src = "store.read_lane::<A, B>()";
        let expr: syn::Expr = syn::parse_str(src).expect("parses");
        if let syn::Expr::MethodCall(mc) = expr {
            assert!(turbofish_type(&mc).is_err());
        } else {
            panic!("expected method call");
        }
    }
}

/// Collects the `SmartStore` parameter name and all lane guard bindings of
/// the function. Also validates the turbofish of every `read_lane` /
/// `write_lane` call on the store, accumulating errors instead of panicking.
#[derive(Default)]
struct LaneCollector {
    store_param: Option<Ident>,
    lanes: Vec<LaneBinding>,
    errors: Vec<syn::Error>,
}

impl LaneCollector {
    fn is_store_receiver(&self, receiver: &Expr) -> bool {
        match (&self.store_param, receiver) {
            (Some(store), Expr::Path(p)) => p.path.is_ident(store),
            _ => false,
        }
    }

    /// If `expr` is `store.read_lane::<T>()` / `store.write_lane::<T>()`
    /// (possibly wrapped in `.unwrap()` / `.expect(..)`), returns whether the
    /// lane is mutable. Malformed turbofish is reported separately by
    /// `visit_expr_method_call`, so here it simply yields `None`.
    fn lane_binding_mutability(&self, expr: &Expr) -> Option<bool> {
        let mut current = expr;
        loop {
            match current {
                Expr::MethodCall(mc) => {
                    if (mc.method == "read_lane" || mc.method == "write_lane")
                        && self.is_store_receiver(&mc.receiver)
                    {
                        return turbofish_type(mc).ok().map(|_| mc.method == "write_lane");
                    }
                    current = &mc.receiver;
                }
                _ => return None,
            }
        }
    }
}

impl Visit<'_> for LaneCollector {
    fn visit_item_fn(&mut self, node: &ItemFn) {
        for arg in &node.sig.inputs {
            if let FnArg::Typed(PatType { pat, ty, .. }) = arg
                && let Pat::Ident(PatIdent { ident, .. }) = &**pat
                && let Type::Reference(TypeReference { elem, .. }) = &**ty
                && let Type::Path(TypePath { path, .. }) = &**elem
                && path
                    .segments
                    .last()
                    .map(|s| s.ident == "SmartStore")
                    .unwrap_or(false)
            {
                self.store_param = Some(ident.clone());
            }
        }
        self.visit_block(&node.block);
    }

    fn visit_local(&mut self, node: &Local) {
        if let Some(init) = &node.init
            && let Pat::Ident(PatIdent { ident, .. }) = &node.pat
            && let Some(is_mutable) = self.lane_binding_mutability(&init.expr)
        {
            self.lanes.push(LaneBinding {
                var_name: ident.clone(),
                is_mutable,
            });
        }
        visit::visit_local(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &ExprMethodCall) {
        if (node.method == "read_lane" || node.method == "write_lane")
            && self.is_store_receiver(&node.receiver)
            && let Err(error) = turbofish_type(node)
        {
            self.errors.push(error);
        }
        visit::visit_expr_method_call(self, node);
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
    fn new(loop_vars: Vec<Ident>) -> Self {
        Self {
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
                "#[smart_pipeline]: variable `{ident}` is assigned inside the loop but declared \
                 outside it - shared mutable state prevents parallelization; loop left sequential"
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
                "#[smart_pipeline]: cross-iteration dependency `{ident}[{}]` prevents \
                 parallelization; loop left sequential",
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
            self.issues.push(
                "#[smart_pipeline]: `break` in the loop body prevents parallelization; loop \
                 left sequential"
                    .to_string(),
            );
        }
        visit::visit_expr_break(self, node);
    }

    fn visit_expr_continue(&mut self, node: &syn::ExprContinue) {
        if node.label.is_some() || (self.closure_depth == 0 && self.loop_depth == 0) {
            self.issues.push(
                "#[smart_pipeline]: `continue` in the loop body prevents parallelization; loop \
                 left sequential"
                    .to_string(),
            );
        }
        visit::visit_expr_continue(self, node);
    }

    fn visit_expr_return(&mut self, node: &syn::ExprReturn) {
        if self.closure_depth == 0 {
            self.issues.push(
                "#[smart_pipeline]: `return` in the loop body prevents parallelization; loop \
                 left sequential"
                    .to_string(),
            );
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
struct LoopRewriter<'a> {
    lanes: &'a [LaneBinding],
    warnings: Vec<String>,
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
            issues.push(
                "#[smart_pipeline]: labeled loop left sequential (labels cannot be rewritten \
                 into a Rayon closure)"
                    .to_string(),
            );
        }

        let mut loop_vars = Vec::new();
        collect_pat_idents(&node.pat, &mut loop_vars);
        if loop_vars.is_empty() {
            issues.push(
                "#[smart_pipeline]: loop pattern binds no variables; loop left sequential"
                    .to_string(),
            );
        }

        let iters = self.extract_lane_iters(&node.expr);
        if iters.is_none() {
            issues.push(format!(
                "#[smart_pipeline]: cannot parallelize loop over `{}` - expected \
                 `lane.iter()`/`lane.iter_mut()`, optionally two lanes combined with \
                 `.zip(..)`; loop left sequential",
                node.expr.to_token_stream()
            ));
        }

        let mut analyzer = LoopBodyAnalyzer::new(loop_vars);
        analyzer.visit_block(&node.body);
        issues.extend(analyzer.issues);

        let iters = match iters {
            Some(iters) if issues.is_empty() => iters,
            _ => return Err(issues),
        };

        let pat = &node.pat;
        let body = &node.body;
        let expr: Expr = match iters.as_slice() {
            [single] => {
                let var = &single.var_name;
                let method = par_iter_method(single.mutable);
                // Аудит §3.3, бэклог #7: захват TLS-фрейма доступов до
                // входа в параллельную секцию и установка в каждой задаче
                // — принуждение действует и на rayon-потоках. Пустой
                // снимок (вне Schedule::run) — no-op, стоимость ноль.
                syn::parse_quote! {{
                    use ornis_core::rayon::prelude::*;
                    let __ornis_access_frame = ornis_core::schedule::capture_access_frame();
                    #var.#method().for_each(|#pat| {
                        let _ornis_frame_guard = __ornis_access_frame.install();
                        #body
                    });
                }}
            }
            [first, second] => {
                let var0 = &first.var_name;
                let var1 = &second.var_name;
                let method0 = par_iter_method(first.mutable);
                let method1 = par_iter_method(second.mutable);
                // См. одноленточную ветвь: захват/установка фрейма (#7).
                syn::parse_quote! {{
                    use ornis_core::rayon::prelude::*;
                    let __ornis_access_frame = ornis_core::schedule::capture_access_frame();
                    #var0.#method0().zip(#var1.#method1()).for_each(|#pat| {
                        let _ornis_frame_guard = __ornis_access_frame.install();
                        #body
                    });
                }}
            }
            _ => unreachable!("extract_lane_iters returns at most two lanes"),
        };
        Ok(expr)
    }
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

pub fn attribute(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as ItemFn);

    let mut collector = LaneCollector::default();
    collector.visit_item_fn(&input);

    if !collector.errors.is_empty() {
        let mut errors = collector.errors.into_iter();
        let mut combined = errors.next().expect("non-empty error list");
        for error in errors {
            combined.combine(error);
        }
        return combined.to_compile_error().into();
    }

    let mut rewriter = LoopRewriter {
        lanes: &collector.lanes,
        warnings: Vec::new(),
    };
    rewriter.visit_block_mut(&mut input.block);

    // Compile-time warnings via the deprecated-note trick: using a deprecated
    // item emits a warning with the note, visible in the IDE and terminal.
    let warning_tokens: Vec<TokenStream2> = rewriter
        .warnings
        .iter()
        .map(|w| {
            quote! {{
                #[deprecated(note = #w)]
                struct SmartPipelineSequentialLoop;
                let _ = SmartPipelineSequentialLoop;
            }}
        })
        .collect();

    let attrs = &input.attrs;
    let vis = &input.vis;
    let sig = &input.sig;
    let stmts = &input.block.stmts;

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            ornis_core::pipeline_enter();
            #(#warning_tokens)*
            // The body runs inside a block so the hook below also fires for
            // functions with a tail expression; `return` still exits early
            // (the exit hook is a no-op profiling marker).
            #[allow(clippy::let_unit_value)]
            let smart_pipeline_result = { #(#stmts)* };
            ornis_core::pipeline_exit();
            smart_pipeline_result
        }
    };

    expanded.into()
}
