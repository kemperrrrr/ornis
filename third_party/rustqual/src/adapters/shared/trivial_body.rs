//! Trivial-accessor detection: which methods are field access in disguise.
//!
//! A getter (`fn len(&self) -> usize { self.items.len() }`) is an
//! equivalent of a field read, so the project scope treats it as trivial
//! rather than as an own call.

/// Check if a method signature has only `self`/`&self`/`&mut self` with no other parameters.
/// Operation: iteration + pattern matching logic, no own calls.
pub fn has_trivial_self_signature(sig: &syn::Signature) -> bool {
    let has_receiver = sig
        .inputs
        .iter()
        .any(|arg| matches!(arg, syn::FnArg::Receiver(_)));
    let typed_count = sig
        .inputs
        .iter()
        .filter(|arg| matches!(arg, syn::FnArg::Typed(_)))
        .count();
    has_receiver && typed_count == 0
}

/// Check if a method call is a trivial accessor call (no-arg stdlib accessor or `.get()` with
/// a trivial argument like a literal or self field access).
/// Operation: if/match logic with inlined arg check, no own calls.
fn is_trivial_method_call(mc: &syn::ExprMethodCall) -> bool {
    let method_name = mc.method.to_string();
    if mc.args.is_empty() {
        return matches!(
            method_name.as_str(),
            "len"
                | "is_empty"
                | "clone"
                | "as_ref"
                | "as_mut"
                | "as_str"
                | "to_owned"
                | "to_string"
                | "borrow"
                | "borrow_mut"
        );
    }
    if mc.args.len() != 1 || !matches!(method_name.as_str(), "get") {
        return false;
    }
    // Inline trivial-arg check: self field access, literal, or reference thereof
    let mut current = &mc.args[0];
    loop {
        match current {
            syn::Expr::Lit(_) => return true,
            syn::Expr::Field(f) => current = &f.base,
            syn::Expr::Path(p) if p.path.is_ident("self") => return true,
            syn::Expr::Reference(r) => current = &r.expr,
            _ => return false,
        }
    }
}

/// Check if a method body is a trivial accessor (single expression accessing self fields).
/// Handles: `self.x`, `&self.x`, `self.x.clone()`, `self.x.len()`, `self.x as f64`,
/// `self.items.get(self.index)`, etc.
/// Operation: iterative loop with pattern matching, no own calls (closure hides helper).
pub fn is_trivial_accessor_body(block: &syn::Block) -> bool {
    if block.stmts.len() != 1 {
        return false;
    }
    let expr = match &block.stmts[0] {
        syn::Stmt::Expr(e, _) => e,
        _ => return false,
    };
    let check_call = |mc: &syn::ExprMethodCall| is_trivial_method_call(mc);
    let mut current = expr;
    loop {
        match current {
            syn::Expr::Field(_) => return true,
            syn::Expr::Reference(r) => current = &r.expr,
            syn::Expr::Cast(c) => current = &c.expr,
            syn::Expr::Unary(u) => current = &u.expr,
            syn::Expr::Paren(p) => current = &p.expr,
            syn::Expr::MethodCall(mc) if check_call(mc) => {
                current = &mc.receiver;
            }
            _ => return false,
        }
    }
}
