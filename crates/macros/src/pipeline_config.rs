use proc_macro::TokenStream;
use proc_macro2::{Ident, Literal, TokenStream as TokenStream2};
use quote::quote;
use std::collections::{HashMap, HashSet};
use syn::{
    Attribute, Data, DataStruct, DeriveInput, Expr, ExprCall, ExprField, ExprForLoop, ExprIf,
    ExprLoop, ExprMatch, ExprMethodCall, ExprWhile, Fields, FieldsNamed, GenericArgument,
    GenericParam, Generics, ImplItem, ItemImpl, PathArguments, Type, TypeParam, TypeParamBound,
    WhereClause, WherePredicate, parse_macro_input, visit, visit::Visit,
};

#[derive(Default)]
struct TypeProfile {
    size_estimate: usize,
    has_heap_types: bool,
    has_gpu_types: bool,
    recursive_type: bool,
}

#[derive(Default, Debug)]
struct MethodProfile {
    branch_count: usize,
    loop_count: usize,
    recursive_call_count: usize,
    field_access_count: HashMap<String, usize>,
}

struct ProfileResult {
    type_profile: TypeProfile,
    method_profiles: Vec<MethodProfile>,
    generics: Generics,
    type_name: Ident,
    attrs: Vec<Attribute>,
}

struct TypeAnalyzer {
    type_name: Ident,
    generics: Generics,
    profile: TypeProfile,
    visited_types: HashSet<String>,
    current_type_path: Vec<String>,
    method_profiles: Vec<MethodProfile>,
    current_method: Option<MethodProfile>,
    current_impl_type: Option<Ident>,
    attrs: Vec<Attribute>,
}

impl TypeAnalyzer {
    fn new(input: &DeriveInput) -> Self {
        Self {
            type_name: input.ident.clone(),
            generics: input.generics.clone(),
            profile: TypeProfile::default(),
            visited_types: HashSet::new(),
            current_type_path: Vec::new(),
            method_profiles: Vec::new(),
            current_method: None,
            current_impl_type: None,
            attrs: input.attrs.clone(),
        }
    }

    fn visit_type(&mut self, ty: &Type) {
        let type_key = self.type_to_string(ty);
        if self.visited_types.contains(&type_key) {
            if self.current_type_path.iter().any(|p| p == &type_key) {
                self.profile.recursive_type = true;
            }
            return;
        }

        self.visited_types.insert(type_key.clone());
        self.current_type_path.push(type_key.clone());

        if let Type::Array(ta) = ty {
            let elem_size = self.analyze_type_get_size(&ta.elem);
            if let Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(lit_int),
                ..
            }) = &ta.len
            {
                if let Ok(len) = lit_int.base10_parse::<usize>() {
                    self.profile.size_estimate = self.profile.size_estimate.max(elem_size * len);
                }
            } else {
                self.profile.size_estimate = self.profile.size_estimate.max(elem_size);
            }
        } else {
            self.analyze_type(ty);
        }

        self.current_type_path.pop();
    }

    fn analyze_type_get_size(&mut self, ty: &Type) -> usize {
        if let Type::Path(tp) = ty {
            let path = &tp.path;
            self.estimate_type_size(path)
        } else {
            64
        }
    }

    fn type_to_string(&self, ty: &Type) -> String {
        match ty {
            Type::Path(tp) => tp
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::"),
            Type::Reference(tr) => format!("&{}", self.type_to_string(&tr.elem)),
            Type::Tuple(tt) => format!(
                "({})",
                tt.elems
                    .iter()
                    .map(|e| self.type_to_string(e))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Type::Array(ta) => format!("[{}]", self.type_to_string(&ta.elem)),
            Type::Slice(ts) => format!("[{}]", self.type_to_string(&ts.elem)),
            Type::Paren(tp) => format!("({})", self.type_to_string(&tp.elem)),
            Type::Group(tg) => self.type_to_string(&tg.elem),
            _ => "unknown".to_string(),
        }
    }

    fn is_heap_type(&self, path: &syn::Path) -> bool {
        if let Some(seg) = path.segments.last() {
            matches!(
                seg.ident.to_string().as_str(),
                "Vec"
                    | "String"
                    | "Box"
                    | "Rc"
                    | "Arc"
                    | "HashMap"
                    | "HashSet"
                    | "BTreeMap"
                    | "BTreeSet"
                    | "VecDeque"
                    | "LinkedList"
            )
        } else {
            false
        }
    }

    fn is_gpu_type(&self, path: &syn::Path) -> bool {
        if let Some(seg) = path.segments.last() {
            let name = seg.ident.to_string();
            matches!(
                name.as_str(),
                "f32"
                    | "f64"
                    | "i32"
                    | "u32"
                    | "i64"
                    | "u64"
                    | "i16"
                    | "u16"
                    | "i8"
                    | "u8"
                    | "bool"
                    | "usize"
                    | "isize"
                    | "Vec2"
                    | "Vec2A"
                    | "Vec3"
                    | "Vec3A"
                    | "Vec4"
                    | "Mat4"
                    | "Quat"
                    | "vec2"
                    | "vec3"
                    | "vec4"
                    | "mat4"
                    | "quat"
            )
        } else {
            false
        }
    }

    fn is_primitive(&self, path: &syn::Path) -> bool {
        if let Some(seg) = path.segments.last() {
            matches!(
                seg.ident.to_string().as_str(),
                "f32"
                    | "f64"
                    | "i32"
                    | "u32"
                    | "i64"
                    | "u64"
                    | "i16"
                    | "u16"
                    | "i8"
                    | "u8"
                    | "bool"
                    | "usize"
                    | "isize"
            )
        } else {
            false
        }
    }

    fn estimate_type_size(&self, path: &syn::Path) -> usize {
        let Some(seg) = path.segments.last() else {
            return 64;
        };
        let name = seg.ident.to_string();
        if let Some(size) = fixed_type_size(&name) {
            return size;
        }
        if self.is_gpu_type(path) {
            16
        } else {
            64
        }
    }

    fn analyze_type(&mut self, ty: &Type) {
        if let Type::Path(tp) = ty {
            let path = &tp.path;
            let is_heap = self.is_heap_type(path);
            let is_gpu = self.is_gpu_type(path);
            let is_primitive = self.is_primitive(path);
            let size = self.estimate_type_size(path);

            if is_heap {
                self.profile.has_heap_types = true;
            }
            if is_gpu || is_primitive {
                self.profile.has_gpu_types = true;
            }
            self.profile.size_estimate = self.profile.size_estimate.max(size);

            if let Some(last_seg) = path.segments.last()
                && let PathArguments::AngleBracketed(args) = &last_seg.arguments
            {
                for arg in &args.args {
                    if let GenericArgument::Type(ty) = arg {
                        self.visit_type(ty);
                    }
                }
            }

            if let Some(ident) = path.get_ident()
                && ident == &self.type_name
            {
                self.profile.recursive_type = true;
            }
        }
    }
}

impl<'ast> Visit<'ast> for TypeAnalyzer {
    fn visit_derive_input(&mut self, node: &'ast DeriveInput) {
        self.attrs = node.attrs.clone();
        visit::visit_derive_input(self, node);
    }

    fn visit_data_struct(&mut self, node: &'ast DataStruct) {
        if let Fields::Named(FieldsNamed { named, .. }) = &node.fields {
            for field in named {
                self.visit_type(&field.ty);
            }
        }
        visit::visit_data_struct(self, node);
    }

    fn visit_item_impl(&mut self, node: &'ast ItemImpl) {
        if node.trait_.is_some() {
            return;
        }

        if let Type::Path(tp) = &*node.self_ty {
            if tp.path.is_ident(&self.type_name) {
                self.current_impl_type = Some(self.type_name.clone());
                visit::visit_item_impl(self, node);
                self.current_impl_type = None;
            } else {
                visit::visit_item_impl(self, node);
            }
        } else {
            visit::visit_item_impl(self, node);
        }
    }

    fn visit_impl_item(&mut self, node: &'ast ImplItem) {
        if let ImplItem::Fn(_) = node {
            self.current_method = Some(MethodProfile {
                branch_count: 0,
                loop_count: 0,
                recursive_call_count: 0,
                field_access_count: HashMap::new(),
            });

            visit::visit_impl_item(self, node);

            if let Some(profile) = self.current_method.take() {
                self.method_profiles.push(profile);
            }
        } else {
            visit::visit_impl_item(self, node);
        }
    }

    fn visit_expr_if(&mut self, node: &'ast ExprIf) {
        if let Some(ref mut profile) = self.current_method {
            profile.branch_count += 1;
        }
        visit::visit_expr_if(self, node);
    }

    fn visit_expr_match(&mut self, node: &'ast ExprMatch) {
        if let Some(ref mut profile) = self.current_method {
            profile.branch_count += node.arms.len().saturating_sub(1).max(1);
        }
        visit::visit_expr_match(self, node);
    }

    fn visit_expr_loop(&mut self, node: &'ast ExprLoop) {
        if let Some(ref mut profile) = self.current_method {
            profile.loop_count += 1;
        }
        visit::visit_expr_loop(self, node);
    }

    fn visit_expr_while(&mut self, node: &'ast ExprWhile) {
        if let Some(ref mut profile) = self.current_method {
            profile.loop_count += 1;
        }
        visit::visit_expr_while(self, node);
    }

    fn visit_expr_for_loop(&mut self, node: &'ast ExprForLoop) {
        if let Some(ref mut profile) = self.current_method {
            profile.loop_count += 1;
        }
        visit::visit_expr_for_loop(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Some(ref mut profile) = self.current_method
            && let Expr::Path(path) = &*node.func
            && let Some(ident) = path.path.get_ident()
            && ident == &self.type_name
        {
            profile.recursive_call_count += 1;
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        if let Some(ref mut profile) = self.current_method
            && let Expr::Path(path) = &*node.receiver
            && let Some(ident) = path.path.get_ident()
            && (ident == "self" || ident == &self.type_name)
            && node.method == self.type_name
        {
            profile.recursive_call_count += 1;
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_field(&mut self, node: &'ast ExprField) {
        if let Some(ref mut profile) = self.current_method {
            let field_name = match &node.member {
                syn::Member::Named(ident) => ident.to_string(),
                syn::Member::Unnamed(index) => format!("_{}", index.index),
            };
            *profile.field_access_count.entry(field_name).or_insert(0) += 1;
        }
        visit::visit_expr_field(self, node);
    }
}

/// Fixed byte-size estimate for known named types (scalars and std/glam
/// composites); `None` when the type has no known size.
fn fixed_type_size(name: &str) -> Option<usize> {
    scalar_type_size(name).or_else(|| composite_type_size(name))
}

/// Byte sizes of fixed-width scalars.
fn scalar_type_size(name: &str) -> Option<usize> {
    Some(match name {
        "f32" | "i32" | "u32" => 4,
        "f64" | "i64" | "u64" | "usize" | "isize" => 8,
        "i16" | "u16" => 2,
        "i8" | "u8" | "bool" => 1,
        _ => return None,
    })
}

/// Byte sizes of composite handle / container types.
fn composite_type_size(name: &str) -> Option<usize> {
    Some(match name {
        "Vec2" | "Vec2A" => 8,
        "Vec3" | "Vec3A" => 12,
        "Vec4" | "Quat" => 16,
        "Mat4" => 64,
        "String" | "Vec" => 24,
        "Box" | "Rc" | "Arc" => 8,
        "HashMap" | "HashSet" | "BTreeMap" | "BTreeSet" | "VecDeque" | "LinkedList" => 48,
        "Option" => 8,
        "Result" => 16,
        _ => return None,
    })
}

fn trait_is_name(bound: &TypeParamBound, name: &str) -> bool {
    matches!(bound, TypeParamBound::Trait(tb) if tb.path.is_ident(name))
}

/// Returns (has_send, has_sync) declared on a single generic type parameter's
/// own bounds (`struct Foo<T: Send + Sync>`).
fn param_send_sync(t: &TypeParam) -> (bool, bool) {
    let mut found_send = false;
    let mut found_sync = false;
    for bound in &t.bounds {
        if trait_is_name(bound, "Send") {
            found_send = true;
        }
        if trait_is_name(bound, "Sync") {
            found_sync = true;
        }
    }
    (found_send, found_sync)
}

/// Returns (has_send, has_sync) declared by a where-clause that names the type
/// directly (`where Foo: Send`, `where Foo<T>: Send`). Compares the leading
/// path segment so generic structs are recognized too.
fn where_clause_send_sync(wc: &WhereClause, type_name: &Ident) -> (bool, bool) {
    let mut has_send = false;
    let mut has_sync = false;
    for predicate in &wc.predicates {
        if let WherePredicate::Type(wt) = predicate
            && let Type::Path(tp) = &wt.bounded_ty
            && tp.path.segments.first().is_some_and(|s| s.ident == *type_name)
        {
            for bound in &wt.bounds {
                if trait_is_name(bound, "Send") {
                    has_send = true;
                }
                if trait_is_name(bound, "Sync") {
                    has_sync = true;
                }
            }
        }
    }
    (has_send, has_sync)
}

fn check_send_sync_bounds(generics: &Generics, type_name: &Ident) -> (bool, bool) {
    let mut has_send = true;
    let mut has_sync = true;

    for param in &generics.params {
        if let GenericParam::Type(t) = param {
            let (found_send, found_sync) = param_send_sync(t);
            if !found_send {
                has_send = false;
            }
            if !found_sync {
                has_sync = false;
            }
        }
    }

    if let Some(wc) = &generics.where_clause {
        let (w_send, w_sync) = where_clause_send_sync(wc, type_name);
        has_send |= w_send;
        has_sync |= w_sync;
    }

    (has_send, has_sync)
}

fn compute_threshold(profile: &ProfileResult) -> usize {
    let type_profile = &profile.type_profile;
    let method_profiles = &profile.method_profiles;

    let mut base_threshold = 10_000usize;

    if type_profile.size_estimate > 256 {
        base_threshold = base_threshold.saturating_mul(10);
    } else if type_profile.size_estimate > 128 {
        base_threshold = base_threshold.saturating_mul(5);
    } else if type_profile.size_estimate > 64 {
        base_threshold = base_threshold.saturating_mul(2);
    }

    if type_profile.has_heap_types {
        base_threshold = usize::MAX / 2;
    }

    let total_branches: usize = method_profiles.iter().map(|m| m.branch_count).sum();
    let total_loops: usize = method_profiles.iter().map(|m| m.loop_count).sum();
    let total_recursive: usize = method_profiles.iter().map(|m| m.recursive_call_count).sum();

    if total_branches > 10 {
        base_threshold = usize::MAX / 2;
    } else if total_branches > 5 {
        base_threshold = base_threshold.saturating_mul(10);
    } else if total_branches > 2 {
        base_threshold = base_threshold.saturating_mul(3);
    }

    if total_loops > 5 {
        base_threshold = usize::MAX / 2;
    } else if total_loops > 2 {
        base_threshold = base_threshold.saturating_mul(5);
    }

    if total_recursive > 0 {
        base_threshold = usize::MAX / 2;
    }

    let unique_fields: usize = method_profiles
        .iter()
        .flat_map(|m| m.field_access_count.keys())
        .collect::<HashSet<_>>()
        .len();

    if unique_fields >= 5 {
        base_threshold = base_threshold.saturating_mul(5);
    } else if unique_fields >= 3 {
        base_threshold = base_threshold.saturating_mul(2);
    }

    let (has_send, has_sync) = check_send_sync_bounds(&profile.generics, &profile.type_name);
    if !has_send || !has_sync {
        base_threshold = usize::MAX / 2;
    }

    if type_profile.recursive_type {
        base_threshold = usize::MAX / 2;
    }

    base_threshold.min(1_000_000)
}

fn compute_lane_target(profile: &ProfileResult) -> TokenStream2 {
    let threshold = compute_threshold(profile);

    // An explicit `#[gpu]` / `#[cpu]` / `#[hybrid]` / `#[auto]` attribute
    // overrides automatic classification.
    if let Some(explicit) = explicit_lane_target(profile, threshold) {
        return explicit;
    }

    let (has_send, has_sync) = check_send_sync_bounds(&profile.generics, &profile.type_name);
    classify_lane_target(
        &profile.type_profile,
        &profile.method_profiles,
        has_send,
        has_sync,
        threshold,
    )
}

/// The lane target requested by an explicit placement attribute, if any.
fn explicit_lane_target(profile: &ProfileResult, threshold: usize) -> Option<TokenStream2> {
    if profile.attrs.iter().any(|a| a.path().is_ident("gpu")) {
        return Some(quote! { ornis_core::TargetDiscriminant::Gpu });
    }
    if profile.attrs.iter().any(|a| a.path().is_ident("cpu")) {
        return Some(quote! { ornis_core::TargetDiscriminant::Cpu });
    }
    if profile
        .attrs
        .iter()
        .any(|a| a.path().is_ident("hybrid") || a.path().is_ident("auto"))
    {
        return Some(auto_target(threshold));
    }
    None
}

/// `TargetDiscriminant::Auto(<threshold>)` tokens.
fn auto_target(threshold: usize) -> TokenStream2 {
    let threshold_lit = Literal::usize_unsuffixed(threshold);
    quote! { ornis_core::TargetDiscriminant::Auto(#threshold_lit) }
}

/// Automatic GPU/CPU/Auto classification from the type and method profiles.
fn classify_lane_target(
    type_profile: &TypeProfile,
    method_profiles: &[MethodProfile],
    has_send: bool,
    has_sync: bool,
    threshold: usize,
) -> TokenStream2 {
    let total_branches: usize = method_profiles.iter().map(|m| m.branch_count).sum();
    let total_loops: usize = method_profiles.iter().map(|m| m.loop_count).sum();
    let total_recursive: usize = method_profiles.iter().map(|m| m.recursive_call_count).sum();
    let unique_fields: usize = method_profiles
        .iter()
        .flat_map(|m| m.field_access_count.keys())
        .collect::<HashSet<_>>()
        .len();

    let is_gpu_friendly = type_profile.has_gpu_types
        && !type_profile.has_heap_types
        && type_profile.size_estimate <= 256
        && total_branches <= 2
        && total_loops == 0
        && total_recursive == 0
        && unique_fields <= 2
        && has_send
        && has_sync;

    let is_cpu_forced = type_profile.has_heap_types
        || type_profile.size_estimate > 256
        || total_branches > 5
        || total_loops > 2
        || total_recursive > 0
        || unique_fields >= 5
        || !has_send
        || !has_sync
        || type_profile.recursive_type;

    if is_cpu_forced {
        quote! { ornis_core::TargetDiscriminant::Cpu }
    } else if is_gpu_friendly {
        quote! { ornis_core::TargetDiscriminant::Gpu }
    } else {
        auto_target(threshold)
    }
}

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let generics = &input.generics;

    let mut analyzer = TypeAnalyzer::new(&input);

    if let Data::Struct(DataStruct { fields, .. }) = &input.data
        && let Fields::Named(FieldsNamed { named, .. }) = fields
    {
        for field in named {
            analyzer.visit_type(&field.ty);
        }
    }

    visit::visit_derive_input(&mut analyzer, &input);

    let profile = ProfileResult {
        type_profile: analyzer.profile,
        method_profiles: analyzer.method_profiles,
        generics: analyzer.generics,
        type_name: analyzer.type_name,
        attrs: analyzer.attrs,
    };

    let threshold = compute_threshold(&profile);
    let threshold_lit = Literal::usize_unsuffixed(threshold);
    let lane_target = compute_lane_target(&profile);

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics ornis_core::PipelineConfig for #name #ty_generics #where_clause {
            fn lane_target() -> ornis_core::TargetDiscriminant {
                #lane_target
            }

            const THRESHOLD: usize = #threshold_lit;
        }
    };

    expanded.into()
}

#[cfg(test)]
mod config_tests {
    use super::*;
    use quote::ToTokens;

    fn generics_of(src: &str) -> (Generics, Ident) {
        let item: syn::DeriveInput = syn::parse_str(src).expect("parse struct");
        let name = item.ident.clone();
        (item.generics, name)
    }

    #[test]
    fn plain_struct_is_send_and_sync() {
        let (g, name) = generics_of("struct Foo { x: f32 }");
        let (send, sync) = check_send_sync_bounds(&g, &name);
        assert!(send && sync);
    }

    #[test]
    fn explicit_send_sync_bounds_detected() {
        let (g, name) = generics_of("struct Foo<T: Send + Sync> { t: T }");
        let (send, sync) = check_send_sync_bounds(&g, &name);
        // Bounds are on T, not on Foo itself — the function checks whether
        // *the struct's own generic params* carry the bounds.
        assert!(send && sync);
    }

    #[test]
    fn unbounded_generic_lacks_send_sync() {
        let (g, name) = generics_of("struct Foo<T> { t: T }");
        let (send, sync) = check_send_sync_bounds(&g, &name);
        assert!(!send && !sync);
    }

    #[test]
    fn where_clause_can_restore_send_sync() {
        // The where-clause path matches only *plain* paths: the code uses
        // `path.is_ident(type_name)`, which is false for `Foo<T>` (generic
        // args make is_ident fail). Document that with a non-generic type.
        let item: syn::ItemStruct =
            syn::parse_str("struct Foo { t: std::marker::PhantomData<u8> }").expect("parse");
        let mut generics = item.generics;
        generics
            .where_clause
            .get_or_insert_with(|| syn::parse_quote!(where Foo: Send + Sync));
        let (send, sync) = check_send_sync_bounds(&generics, &item.ident);
        assert!(send && sync);
    }

    #[test]
    fn generic_where_clause_is_detected() {
        // Regression: `where Foo<T>: Send` was previously NOT recognized
        // because path.is_ident() rejects paths with generic arguments.
        // The check now compares the leading segment, so generic components
        // with where-clauses route correctly.
        let item: syn::ItemStruct = syn::parse_str("struct Foo<T> { t: T }").expect("parse");
        let mut generics = item.generics;
        generics
            .where_clause
            .get_or_insert_with(|| syn::parse_quote!(where Foo<T>: Send + Sync));
        let (send, sync) = check_send_sync_bounds(&generics, &item.ident);
        assert!(send && sync);
    }

    #[test]
    fn threshold_small_pod_type() {
        let profile = ProfileResult {
            type_profile: TypeProfile {
                size_estimate: 12,
                has_heap_types: false,
                has_gpu_types: false,
                recursive_type: false,
            },
            method_profiles: vec![],
            generics: Default::default(),
            type_name: syn::parse_str("Foo").unwrap(),
            attrs: vec![],
        };
        // Base 10_000 for a small POD struct with no methods.
        assert_eq!(compute_threshold(&profile), 10_000);
    }

    #[test]
    fn threshold_heap_types_force_cpu() {
        let profile = ProfileResult {
            type_profile: TypeProfile {
                size_estimate: 12,
                has_heap_types: true,
                has_gpu_types: false,
                recursive_type: false,
            },
            method_profiles: vec![],
            generics: Default::default(),
            type_name: syn::parse_str("Foo").unwrap(),
            attrs: vec![],
        };
        assert_eq!(compute_threshold(&profile), 1_000_000);
    }

    #[test]
    fn threshold_many_branches_force_cpu() {
        let method = MethodProfile {
            branch_count: 11,
            ..Default::default()
        };
        let profile = ProfileResult {
            type_profile: TypeProfile {
                size_estimate: 12,
                has_heap_types: false,
                has_gpu_types: false,
                recursive_type: false,
            },
            method_profiles: vec![method],
            generics: Default::default(),
            type_name: syn::parse_str("Foo").unwrap(),
            attrs: vec![],
        };
        assert_eq!(compute_threshold(&profile), 1_000_000);
    }

    #[test]
    fn threshold_size_multipliers() {
        let mk = |size: usize| ProfileResult {
            type_profile: TypeProfile {
                size_estimate: size,
                has_heap_types: false,
                has_gpu_types: false,
                recursive_type: false,
            },
            method_profiles: vec![],
            generics: Default::default(),
            type_name: syn::parse_str("Foo").unwrap(),
            attrs: vec![],
        };
        assert_eq!(compute_threshold(&mk(65)), 20_000); // >64 → ×2
        assert_eq!(compute_threshold(&mk(129)), 50_000); // >128 → ×5
        assert_eq!(compute_threshold(&mk(257)), 100_000); // >256 → ×10
    }

    #[test]
    fn lane_target_tokens_for_gpu_profile() {
        let profile = ProfileResult {
            type_profile: TypeProfile {
                size_estimate: 12,
                has_heap_types: false,
                has_gpu_types: true,
                recursive_type: false,
            },
            method_profiles: vec![],
            generics: Default::default(),
            type_name: syn::parse_str("Foo").unwrap(),
            attrs: vec![],
        };
        let tokens = compute_lane_target(&profile).to_token_stream().to_string();
        assert!(tokens.contains("Gpu"), "got: {tokens}");
    }
}
