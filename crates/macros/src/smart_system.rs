//! Implementation of the `#[smart_system]` attribute macro (PLAN m, path 3).
//!
//! A `#[smart_system]` function is an ordinary free function taking
//! `&Resources` plus plain parameters:
//!
//! ```ignore
//! #[smart_system]
//! fn player_input(resources: &Resources, speed: f32) {
//!     let store = resources.get::<SmartStore>().expect("store");
//!     let mut vel = store.write_lane::<Velocity>().expect("vel");
//!     // ...
//! }
//! ```
//!
//! The macro emits, next to the (loop-rewritten, [`smart_pipeline`]-style)
//! function:
//!
//! - `struct PlayerInputSystem { speed: f32 }` — one field per non-`Resources`
//!   parameter, in order (a `&T` parameter becomes an owned `T` field passed
//!   as `&self.field`, so no per-frame clone; other parameters are stored
//!   as-is and passed via `.clone()`, hence must be `Clone`);
//! - `impl ornis_core::System` — `name()` (from `name = "..."`, defaulting to
//!   the function name), `access()` derived from the body
//!   (`resources.get::<T>()` → `reads::<T>()`, lane bindings →
//!   `reads_lane`/`writes_lane`, a write covering a read), and `run()`
//!   forwarding to the function with parameters in declaration order.
//!
//! Because `access()` is generated from the same bindings the body uses, the
//! declaration cannot desync from the implementation. What cannot be derived
//! is rejected or declared explicitly:
//!
//! - `#[smart_system(reads(A, B), writes(C))]` adds explicit entries
//!   (a `writes` entry suppresses the same type in `reads`);
//! - a resource type with interior mutability (`Mutex`, `RwLock`, `Atomic*`,
//!   `Cell`, `RefCell`, `OnceLock`) must be listed explicitly, otherwise the
//!   macro errors — a silent `reads` would mislead the level scheduler;
//! - `resources.get()` without turbofish, or any other use of the `resources`
//!   ident (passing it to a helper, aliasing it, capturing it), is a compile
//!   error unless explicit `reads(...)`/`writes(...)` are present.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, quote};
use syn::{
    Expr, FnArg, Ident, ItemFn, Pat, PatType, ReturnType, Type, parse_macro_input, visit::Visit,
    visit_mut::VisitMut,
};

use crate::smart_pipeline::{
    LaneCollector, LoopRewriter, ParamKind, access_entries, classify_param, combine_errors2,
    send_sync_assert_tokens, warning_tokens, wrap_body,
};
use syn::ext::IdentExt;

/// `#[smart_system]` arguments: `name = "..."`, `reads(A, ...)`, `writes(B, ...)`.
struct SmartSystemArgs {
    name: Option<String>,
    reads: Vec<Type>,
    writes: Vec<Type>,
}

fn type_list(tokens: proc_macro2::TokenStream) -> syn::Result<Vec<Type>> {
    struct Types(Vec<Type>);
    impl syn::parse::Parse for Types {
        fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
            Ok(Self(
                <syn::punctuated::Punctuated<Type, syn::Token![,]>>::parse_terminated(input)?
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
        };
        let items =
            <syn::punctuated::Punctuated<syn::Meta, syn::Token![,]>>::parse_terminated(input)?;
        for item in items {
            match item {
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
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "#[smart_system]: expected `name = \"...\"`, `reads(A, ...)` or `writes(B, ...)`",
                    ));
                }
            }
        }
        Ok(args)
    }
}

/// `player_input` → `PlayerInputSystem`. Raw identifiers are unrawed first;
/// the struct span comes from the function name.
fn system_struct_ident(fn_ident: &Ident) -> Ident {
    let plain = fn_ident.unraw().to_string();
    let mut camel = String::new();
    for part in plain.split('_').filter(|part| !part.is_empty()) {
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            camel.extend(first.to_uppercase());
            camel.push_str(chars.as_str());
        }
    }
    camel.push_str("System");
    Ident::new(&camel, fn_ident.span())
}

/// Last path segment of a type (`Mutex<Counter>` → `Mutex`).
fn last_segment(ty: &Type) -> Option<String> {
    LaneCollector::last_segment(ty)
}

/// Whether the resource type has interior mutability: a silent `reads` would
/// mislead the level scheduler, so it must be declared explicitly.
fn has_interior_mutability(ty: &Type) -> bool {
    match last_segment(ty).as_deref() {
        Some("Mutex" | "RwLock" | "Cell" | "RefCell" | "OnceLock" | "OnceCell") => true,
        Some(name) if name.starts_with("Atomic") => true,
        _ => false,
    }
}

/// A non-`Resources` parameter: owned field type plus how `run()` passes it.
struct FieldParam {
    name: Ident,
    /// Type stored in the struct.
    field_ty: Type,
    /// Tokens passed at the call site (`&self.x` or `self.x.clone()`).
    pass: TokenStream2,
}

/// Validates the signature: free function, exactly one `&Resources`, unit
/// return, no generics/async/unsafe. Returns the resources ident.
fn validate_signature(sig: &syn::Signature) -> syn::Result<Ident> {
    if sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            &sig.ident,
            "#[smart_system]: async system functions are not supported",
        ));
    }
    if matches!(sig.safety, syn::Safety::Unsafe(_)) {
        return Err(syn::Error::new_spanned(
            &sig.ident,
            "#[smart_system]: unsafe system functions are not supported",
        ));
    }
    if !sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &sig.ident,
            "#[smart_system]: generic system functions are not supported",
        ));
    }
    if !matches!(sig.output, ReturnType::Default) {
        return Err(syn::Error::new_spanned(
            &sig.ident,
            "#[smart_system]: system functions must return `()`",
        ));
    }
    let mut resources: Option<Ident> = None;
    for arg in &sig.inputs {
        let FnArg::Typed(PatType { pat, ty, .. }) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "#[smart_system]: system functions must be free functions",
            ));
        };
        let Pat::Ident(pat_ident) = &**pat else {
            return Err(syn::Error::new_spanned(
                pat,
                "#[smart_system]: system parameters must be plain identifiers",
            ));
        };
        if classify_param(ty) == ParamKind::Resources {
            if resources.is_some() {
                return Err(syn::Error::new_spanned(
                    pat,
                    "#[smart_system]: expected exactly one `&Resources` parameter",
                ));
            }
            if let Type::Reference(type_ref) = &**ty
                && type_ref.mutability.is_some()
            {
                return Err(syn::Error::new_spanned(
                    ty,
                    "#[smart_system]: `&mut Resources` is not supported — `run` receives `&Resources`",
                ));
            }
            resources = Some(pat_ident.ident.clone());
        }
    }
    resources.ok_or_else(|| {
        syn::Error::new_spanned(
            &sig.ident,
            "#[smart_system]: expected exactly one `&Resources` parameter",
        )
    })
}

/// Builds one struct field: a `&T` parameter becomes an owned `T` field
/// passed as `&self.field` (no per-frame clone); any other parameter is
/// stored as-is and passed via `.clone()`. `&mut` parameters, explicit
/// lifetimes and unsized targets are rejected.
fn field_param(name: Ident, ty: &Type) -> syn::Result<FieldParam> {
    let (pass, field_ty) = match ty {
        Type::Reference(type_ref) => ref_words(type_ref, &name)?,
        _ => (quote! { self.#name.clone() }, ty.clone()),
    };
    Ok(FieldParam {
        pass,
        field_ty,
        name,
    })
}

/// Pass tokens plus stored type for a `&T` parameter: `&self.x` over an
/// owned `T` field, so no per-frame clone happens.
fn ref_words(type_ref: &syn::TypeReference, name: &Ident) -> syn::Result<(TokenStream2, Type)> {
    if type_ref.mutability.is_some() {
        return Err(syn::Error::new_spanned(
            &type_ref.elem,
            "#[smart_system]: `&mut` parameters cannot be stored in the system struct; pass owned values instead",
        ));
    }
    if type_ref.lifetime.is_some() {
        return Err(syn::Error::new_spanned(
            &type_ref.elem,
            "#[smart_system]: explicit lifetimes on parameters are not supported",
        ));
    }
    match &*type_ref.elem {
        Type::Slice(_) => Err(syn::Error::new_spanned(
            &type_ref.elem,
            "#[smart_system]: unsized targets cannot be stored in the system struct",
        )),
        Type::Path(path) if path.path.segments.last().is_some_and(|s| s.ident == "str") => {
            Err(syn::Error::new_spanned(
                &type_ref.elem,
                "#[smart_system]: `&str` parameters cannot be stored in the system struct; use an owned `String`",
            ))
        }
        _ => Ok((quote! { &self.#name }, (*type_ref.elem).clone())),
    }
}

/// Collects struct fields from non-`Resources` parameters, in order.
fn collect_fields(sig: &syn::Signature, resources: &Ident) -> syn::Result<Vec<FieldParam>> {
    let mut fields = Vec::new();
    for arg in &sig.inputs {
        let FnArg::Typed(PatType { pat, ty, .. }) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "#[smart_system]: system functions must be free functions",
            ));
        };
        let Pat::Ident(pat_ident) = &**pat else {
            return Err(syn::Error::new_spanned(
                pat,
                "#[smart_system]: system parameters must be plain identifiers",
            ));
        };
        if pat_ident.ident == *resources {
            continue;
        }
        fields.push(field_param(pat_ident.ident.clone(), ty)?);
    }
    Ok(fields)
}

/// Flags `resources` uses the macro cannot derive an access set from:
/// anything but the receiver of `get::<T>()` / `contains::<T>()` (or the
/// side-effect-free `len()` / `is_empty()`). Runs only when no explicit
/// `reads(...)` / `writes(...)` take responsibility.
struct EscapeCheck<'a> {
    resources: &'a Ident,
    errors: Vec<syn::Error>,
}

impl EscapeCheck<'_> {
    fn is_resources_path(&self, expr: &Expr) -> bool {
        matches!(expr, Expr::Path(path) if path.path.is_ident(self.resources))
    }

    fn deny(&mut self, expr: &Expr) {
        self.errors.push(syn::Error::new_spanned(
            expr,
            "#[smart_system]: `resources` escapes here; access cannot be derived — declare it with `reads(...)`/`writes(...)`",
        ));
    }
}

impl<'a> Visit<'a> for EscapeCheck<'a> {
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
        if self.is_resources_path(&node.receiver) {
            self.deny(&node.receiver);
        }
        syn::visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_path(&mut self, node: &'a syn::ExprPath) {
        if node.path.is_ident(self.resources) {
            self.deny(&Expr::Path(node.clone()));
        }
        syn::visit::visit_expr_path(self, node);
    }
}

/// A `resources.get()` / `contains()` call without a valid turbofish hides
/// its type, so the access set cannot be derived. Standalone visitor (kept
/// out of the shared collector so `#[smart_pipeline]` stays silent).
struct TurbofishCheck<'a> {
    resources: &'a Ident,
    macro_name: &'static str,
    errors: Vec<syn::Error>,
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
}

/// Runs the derivability checks over the body: typeless `get()` calls
/// (the type would be hidden) and escaping `resources` uses (an access would
/// be hidden). The escape check is skipped when explicit `reads(...)` /
/// `writes(...)` take responsibility. Errors combine into one diagnostic.
fn check_derivable(
    input: &ItemFn,
    resources: &Ident,
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
        return Err(combine_errors2(turbofish.errors));
    }

    // Escape check: any other use of the `resources` ident hides an access.
    if args.reads.is_empty() && args.writes.is_empty() {
        let mut escape = EscapeCheck {
            resources,
            errors: Vec::new(),
        };
        escape.visit_block(&input.block);
        if !escape.errors.is_empty() {
            return Err(combine_errors2(escape.errors));
        }
    }
    Ok(())
}

/// Builds the `access()` body tokens: explicit entries first, then derived
/// ones. Canonical order: sorted resource reads (explicit `writes` suppress
/// the same type in `reads`), then sorted read lanes, then sorted write
/// lanes. Interior-mutability types must be listed explicitly.
fn access_tokens(
    collector: &LaneCollector,
    explicit_reads: &[Type],
    explicit_writes: &[Type],
) -> syn::Result<TokenStream2> {
    let type_name = |ty: &Type| ty.to_token_stream().to_string();
    let mut read_tys: Vec<Type> = explicit_reads.to_vec();
    for ty in &collector.resource_reads {
        if !read_tys.iter().any(|kept| type_name(kept) == type_name(ty)) {
            read_tys.push(ty.clone());
        }
    }
    let mut write_tys: Vec<Type> = explicit_writes.to_vec();
    read_tys.retain(|ty| {
        !write_tys
            .iter()
            .any(|written| type_name(written) == type_name(ty))
    });
    let mut error: Option<syn::Error> = None;
    for ty in read_tys.iter().chain(write_tys.iter()) {
        if has_interior_mutability(ty)
            && !explicit_reads
                .iter()
                .chain(explicit_writes.iter())
                .any(|listed| type_name(listed) == type_name(ty))
        {
            let next = syn::Error::new_spanned(
                ty,
                format!(
                    "#[smart_system]: resource `{}` has interior mutability — declare it with `writes(...)` or `reads(...)`",
                    type_name(ty)
                ),
            );
            match &mut error {
                Some(first) => first.combine(next),
                None => error = Some(next),
            }
        }
    }
    if let Some(error) = error {
        return Err(error);
    }

    read_tys.sort_by_key(|ty| type_name(ty));
    write_tys.sort_by_key(|ty| type_name(ty));
    let lane_entries = access_entries(&collector.lane_accesses);
    let mut lane_reads: Vec<&Type> = Vec::new();
    let mut lane_writes: Vec<&Type> = Vec::new();
    for (ty, write) in &lane_entries {
        if *write {
            lane_writes.push(ty);
        } else {
            lane_reads.push(ty);
        }
    }
    Ok(quote! {
        ornis_core::SystemAccess::new()
            #(.reads::<#read_tys>())*
            #(.writes::<#write_tys>())*
            #(.reads_lane::<#lane_reads>())*
            #(.writes_lane::<#lane_writes>())*
    })
}

/// Emission context for one `#[smart_system]` function: everything
/// `struct_tokens` needs, so the generator itself stays small.
struct Emission<'a> {
    struct_ident: Ident,
    fn_ident: &'a Ident,
    vis: &'a syn::Visibility,
    system_name: String,
    sig: &'a syn::Signature,
    resources: &'a Ident,
    fields: &'a [FieldParam],
    access: TokenStream2,
}

/// Builds the struct definition, its `::new()`, and the `System` impl.
/// Call arguments follow the declaration order: `resources` in place,
/// fields via their pass tokens.
fn struct_tokens(emission: &Emission<'_>) -> TokenStream2 {
    let Emission {
        struct_ident,
        fn_ident,
        vis,
        system_name,
        sig,
        resources,
        fields,
        access,
    } = emission;
    let field_names: Vec<&Ident> = fields.iter().map(|f| &f.name).collect();
    let field_tys: Vec<&Type> = fields.iter().map(|f| &f.field_ty).collect();
    let mut call_args: Vec<TokenStream2> = Vec::new();
    for arg in &sig.inputs {
        if let FnArg::Typed(PatType { pat, .. }) = arg
            && let Pat::Ident(pat_ident) = &**pat
        {
            if pat_ident.ident == **resources {
                call_args.push(quote! { resources });
            } else if let Some(field) = fields.iter().find(|f| f.name == pat_ident.ident) {
                let pass = &field.pass;
                call_args.push(quote! { #pass });
            }
        }
    }
    let fn_name = fn_ident.to_string();
    let doc_struct = format!("System generated by `#[smart_system]` from [`{fn_name}`].");
    let doc_new = format!("Creates the system generated by `#[smart_system]` from [`{fn_name}`].");
    let (struct_def, struct_ctor) = if field_names.is_empty() {
        (
            quote! { #vis struct #struct_ident; },
            quote! {
                impl #struct_ident {
                    #[doc = #doc_new]
                    pub fn new() -> Self {
                        Self
                    }
                }
            },
        )
    } else {
        (
            quote! {
                #vis struct #struct_ident {
                    #( #field_names : #field_tys ),*
                }
            },
            quote! {
                impl #struct_ident {
                    #[doc = #doc_new]
                    pub fn new( #( #field_names : #field_tys ),* ) -> Self {
                        Self { #( #field_names ),* }
                    }
                }
            },
        )
    };
    quote! {
        #[doc = #doc_struct]
        #struct_def
        #struct_ctor
        impl ornis_core::System for #struct_ident {
            fn name(&self) -> &'static str {
                #system_name
            }
            fn access(&self) -> ornis_core::SystemAccess {
                #access
            }
            fn run(&self, resources: &ornis_core::Resources) {
                #fn_ident(#(#call_args),*);
            }
        }
    }
}

pub fn attribute(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as SmartSystemArgs);
    let input = parse_macro_input!(item as ItemFn);
    expand(args, input).into()
}

/// Bridge-free expansion core (unit-testable): everything below takes and
/// returns plain syntax trees, so tests exercise the real pipeline without
/// the procedural-macro bridge.
fn expand(args: SmartSystemArgs, mut input: ItemFn) -> TokenStream2 {
    let resources = match validate_signature(&input.sig) {
        Ok(resources) => resources,
        Err(error) => return error.to_compile_error(),
    };
    let fields = match collect_fields(&input.sig, &resources) {
        Ok(fields) => fields,
        Err(error) => return error.to_compile_error(),
    };

    let mut collector = LaneCollector::new("#[smart_system]");
    collector.visit_item_fn(&input);

    if !collector.errors.is_empty() {
        return combine_errors2(collector.errors);
    }
    if let Err(tokens) = check_derivable(&input, &resources, &args) {
        return tokens;
    }
    let access = match access_tokens(&collector, &args.reads, &args.writes) {
        Ok(access) => access,
        Err(error) => return error.to_compile_error(),
    };

    let mut rewriter = LoopRewriter {
        macro_name: "#[smart_system]",
        lanes: &collector.lanes,
        warnings: Vec::new(),
    };
    rewriter.visit_block_mut(&mut input.block);
    let warnings = warning_tokens(&rewriter.warnings);

    let send_sync_assert = send_sync_assert_tokens(&collector.lane_accesses);
    let struct_ident = system_struct_ident(&input.sig.ident);
    let system_name = args
        .name
        .clone()
        .unwrap_or_else(|| input.sig.ident.unraw().to_string());
    let attrs = &input.attrs;
    let vis = &input.vis;
    let sig = &input.sig;
    let body = wrap_body(&input.block.stmts);
    let generated = struct_tokens(&Emission {
        struct_ident,
        fn_ident: &input.sig.ident,
        vis,
        system_name,
        sig: &input.sig,
        resources: &resources,
        fields: &fields,
        access,
    });

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            ornis_core::pipeline_enter();
            #send_sync_assert
            #(#warnings)*
            #body
            ornis_core::pipeline_exit();
            smart_pipeline_result
        }
        #generated
    };
    expanded
}

#[cfg(test)]
mod smart_system_tests {
    use super::*;

    fn ident(name: &str) -> Ident {
        syn::parse_str(name).expect("ident parses")
    }

    fn ty(src: &str) -> Type {
        syn::parse_str(src).expect("type parses")
    }

    #[test]
    fn struct_ident_camel_cases() {
        assert_eq!(
            system_struct_ident(&ident("drive")).to_string(),
            "DriveSystem"
        );
        assert_eq!(
            system_struct_ident(&ident("player_input")).to_string(),
            "PlayerInputSystem"
        );
        assert_eq!(system_struct_ident(&ident("_x")).to_string(), "XSystem");
        assert_eq!(system_struct_ident(&ident("a__b")).to_string(), "ABSystem");
    }

    #[test]
    fn struct_ident_unraws_keywords() {
        let raw: Ident = syn::parse_str("r#loop").expect("raw ident parses");
        assert_eq!(system_struct_ident(&raw).to_string(), "LoopSystem");
    }

    #[test]
    fn interior_mutability_set() {
        for name in [
            "Mutex<C>",
            "RwLock<C>",
            "AtomicBool",
            "AtomicU64",
            "Cell<C>",
            "RefCell<C>",
            "OnceLock<C>",
        ] {
            assert!(has_interior_mutability(&ty(name)), "{name}");
        }
        for name in ["Cfg", "SmartStore", "Vec<u8>", "MutexLike"] {
            assert!(!has_interior_mutability(&ty(name)), "{name}");
        }
    }
}

#[cfg(test)]
mod smart_system_unit_tests {
    use super::*;

    fn parse_item(src: &str) -> ItemFn {
        syn::parse_str(src).expect("fn parses")
    }

    fn resources_ident() -> Ident {
        syn::parse_str("resources").expect("ident")
    }

    #[test]
    fn field_param_owned_and_borrowed() {
        let owned = field_param(
            syn::parse_str("speed").expect("ident"),
            &syn::parse_str("f32").expect("type"),
        )
        .expect("owned ok");
        assert_eq!(owned.pass.to_string(), "self . speed . clone ()");
        let borrowed = field_param(
            syn::parse_str("table").expect("ident"),
            &syn::parse_str("&Vec<u8>").expect("type"),
        )
        .expect("borrowed ok");
        assert_eq!(borrowed.pass.to_string(), "& self . table");
        assert_eq!(
            borrowed.field_ty.to_token_stream().to_string(),
            "Vec < u8 >"
        );
        assert!(
            field_param(
                syn::parse_str::<Ident>("s").expect("ident"),
                &syn::parse_str("&mut Vec<u8>").expect("type"),
            )
            .is_err()
        );
        assert!(
            field_param(
                syn::parse_str::<Ident>("s").expect("ident"),
                &syn::parse_str("&str").expect("type"),
            )
            .is_err()
        );
    }

    #[test]
    fn validate_signature_accepts_and_rejects() {
        let item = parse_item("fn ok(resources: &Resources, speed: f32) {}");
        assert_eq!(
            validate_signature(&item.sig).expect("valid").to_string(),
            "resources"
        );
        for src in [
            "fn no_res(speed: f32) {}",
            "fn two(a: &Resources, b: &Resources) {}",
            "fn ret(resources: &Resources) -> usize { 0 }",
            "async fn asy(resources: &Resources) {}",
            "unsafe fn uns(resources: &Resources) {}",
            "fn gen<T>(resources: &Resources, v: T) {}",
        ] {
            assert!(validate_signature(&parse_item(src).sig).is_err(), "{src}");
        }
    }

    #[test]
    fn collect_fields_skips_resources_in_order() {
        let item = parse_item("fn f(speed: f32, resources: &Resources, n: u32) {}");
        let fields = collect_fields(&item.sig, &resources_ident()).expect("fields");
        let names: Vec<String> = fields.iter().map(|f| f.name.to_string()).collect();
        assert_eq!(names, vec!["speed".to_string(), "n".to_string()]);
    }

    #[test]
    fn check_derivable_passes_clean_bodies() {
        let item = parse_item(
            "fn f(resources: &Resources) { let _ = resources.get::<Cfg>(); let _ = resources.len(); }",
        );
        let args = SmartSystemArgs {
            name: None,
            reads: Vec::new(),
            writes: Vec::new(),
        };
        let resources = resources_ident();
        assert!(check_derivable(&item, &resources, &args).is_ok());
    }

    #[test]
    fn access_tokens_merges_explicit_and_derived() {
        let mut collector = LaneCollector::new("#[smart_system]");
        collector.resources_param = Some(resources_ident());
        let pos: Type = syn::parse_str("Pos").expect("type");
        let vel: Type = syn::parse_str("Vel").expect("type");
        collector.lane_accesses.push((pos, true));
        collector.lane_accesses.push((vel, false));
        let tokens = access_tokens(&collector, &[], &[]).expect("access");
        let rendered = tokens.to_string();
        assert!(rendered.contains("reads_lane :: < Vel >"), "{rendered}");
        assert!(rendered.contains("writes_lane :: < Pos >"), "{rendered}");
    }

    #[test]
    fn escape_check_allows_get_len_and_denies_calls() {
        let resources = resources_ident();
        let clean: ItemFn = parse_item(
            "fn f(resources: &Resources) { let _ = resources.get::<Cfg>(); let _ = resources.len(); }",
        );
        let mut ok = EscapeCheck {
            resources: &resources,
            errors: Vec::new(),
        };
        ok.visit_block(&clean.block);
        assert!(ok.errors.is_empty());

        let dirty: ItemFn = parse_item("fn f(resources: &Resources) { helper(resources); }");
        let mut bad = EscapeCheck {
            resources: &resources,
            errors: Vec::new(),
        };
        bad.visit_block(&dirty.block);
        assert_eq!(bad.errors.len(), 1);
    }

    #[test]
    fn attribute_entry_emits_system() {
        let args = SmartSystemArgs {
            name: None,
            reads: Vec::new(),
            writes: Vec::new(),
        };
        let item: ItemFn = syn::parse_str(
            "fn direct(resources: &Resources, speed: f32) { let _ = (resources.len(), speed); }",
        )
        .expect("fn parses");
        let rendered = expand(args, item).to_string();
        assert!(rendered.contains("DirectSystem"), "{rendered}");
        assert!(rendered.contains("\"direct\""), "{rendered}");
        assert!(rendered.contains("impl ornis_core :: System"), "{rendered}");
    }
}

#[cfg(test)]
mod ref_words_tests {
    use super::*;

    fn ident(name: &str) -> Ident {
        syn::parse_str(name).expect("ident parses")
    }

    #[test]
    fn ref_words_rejects_and_builds() {
        let (pass, ty) = ref_words(
            &match syn::parse_str::<Type>("&Vec<u8>").expect("type") {
                Type::Reference(type_ref) => type_ref,
                _ => panic!("expected reference"),
            },
            &ident("table"),
        )
        .expect("borrowed ok");
        assert_eq!(pass.to_string(), "& self . table");
        assert_eq!(ty.to_token_stream().to_string(), "Vec < u8 >");
        for src in ["&mut Vec<u8>", "&'a Vec<u8>", "&[u8]", "&str"] {
            let ty: Type = syn::parse_str(src).expect("type parses");
            let Type::Reference(type_ref) = ty else {
                panic!("expected reference");
            };
            assert!(ref_words(&type_ref, &ident("x")).is_err(), "{src}");
        }
    }
}
