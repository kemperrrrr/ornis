use std::collections::{HashMap, HashSet};

use syn::visit::Visit;
use syn::{File, ImplItem, TraitItem};

use super::trivial_body::{has_trivial_self_signature, is_trivial_accessor_body};

/// Scope key of a function: `<file>::<name>` for a free function,
/// `<file>::<Type>::<name>` for a method.
///
/// The key names the lexical scope a function lives in. Nested functions are
/// only visible inside their enclosing function, so their declarations are
/// filed under the enclosing scope instead of the project-wide function set.
pub fn function_scope_key(file_path: &str, parent_type: Option<&str>, name: &str) -> String {
    match parent_type {
        Some(parent) => format!("{file_path}::{parent}::{name}"),
        None => format!("{file_path}::{name}"),
    }
}

/// Collects all declared function/method/type names across a project.
///
/// Used in a two-pass analysis: first pass builds the scope, second pass
/// uses it to distinguish own calls from external ones.
#[derive(Debug, Clone, Default)]
pub struct ProjectScope {
    /// Free functions declared at module level: `classify_function`, `collect_rust_files`, …
    ///
    /// Functions nested inside another function are *not* here — they are
    /// only visible inside their enclosing function and live in
    /// [`ProjectScope::nested_functions`].
    pub functions: HashSet<String>,
    /// Every free function name, nested ones included: used by reachability
    /// lookups where the declaration site does not matter.
    pub all_functions: HashSet<String>,
    /// Functions nested inside another function, grouped by the scope they
    /// belong to: `"src/lib.rs::outer"` → `{"inner"}`.
    ///
    /// A nested function is an own call only inside the scopes that enclose
    /// it, so a same-named closure in another module is not an own call.
    pub nested_functions: HashMap<String, HashSet<String>>,
    /// Methods from impl/trait blocks: `analyze_file`, `load`, …
    pub methods: HashSet<String>,
    /// Struct/enum/trait names: `Analyzer`, `Config`, `Summary`, …
    pub types: HashSet<String>,
    /// Trivial getters: single-receiver, single-statement methods (equivalent to field access).
    pub trivial_methods: HashSet<String>,
    /// Methods that only appear in trait contexts (trait defs or trait impls), never in inherent impls.
    /// Dot-syntax calls to these are polymorphic dispatch, not own calls.
    pub trait_only_methods: HashSet<String>,
    /// Methods grouped by parent type name: `"Config" → {"load", "save"}`.
    pub methods_by_type: HashMap<String, HashSet<String>>,
}

impl ProjectScope {
    /// Build a ProjectScope from a set of already-parsed files.
    pub fn from_files(files: &[(&str, &File)]) -> Self {
        // Pass 0: type definitions (struct/enum/union/trait idents). Impl
        // self-types are only claimed as project types when defined here:
        // a blanket `impl<T> Foo for Vec<T>` must not make `Vec::new`
        // resolve as an own call tree-wide (false IOSP violations).
        let defined_types = collect_defined_types(files);
        let mut collector = ScopeCollector {
            functions: HashSet::new(),
            all_functions: HashSet::new(),
            nested_functions: HashMap::new(),
            methods: HashSet::new(),
            types: HashSet::new(),
            trivial_candidates: HashSet::new(),
            non_trivial_methods: HashSet::new(),
            trait_method_names: HashSet::new(),
            concrete_method_names: HashSet::new(),
            methods_by_type: HashMap::new(),
            defined_types,
            file: String::new(),
            fn_scopes: Vec::new(),
            parent_type: None,
        };
        files.iter().for_each(|(path, file)| {
            collector.file = (*path).to_string();
            collector.visit_file(file);
        });
        ProjectScope {
            functions: collector.functions,
            all_functions: collector.all_functions,
            nested_functions: collector.nested_functions,
            methods: collector.methods,
            types: collector.types,
            trivial_methods: collector
                .trivial_candidates
                .difference(&collector.non_trivial_methods)
                .cloned()
                .collect(),
            trait_only_methods: collector
                .trait_method_names
                .difference(&collector.concrete_method_names)
                .cloned()
                .collect(),
            methods_by_type: collector.methods_by_type,
        }
    }

    /// Is `name` an own *function* call (path-style: `func()` or `Type::func()`)?
    ///
    /// - Single segment (`classify_function`): checks `functions`
    /// - Multi-segment (`Config::load`): checks if first segment is in `types`
    /// - `Self::method`: checks `methods` excluding trait-only
    /// - PascalCase final segment: enum variant constructor, not a call
    ///
    /// Operation: if-let + comparison logic, no own calls.
    pub fn is_own_function(&self, name: &str) -> bool {
        self.is_own_function_in_scope(name, None)
    }

    /// Is `name` an own function call, resolved inside `scope_key`?
    ///
    /// A single-segment name is an own call when it is a module-level
    /// function of the project, or a function nested in the scope of the
    /// calling function. Nested functions of *other* scopes are not visible
    /// from here — a closure that happens to share their name is not an own
    /// call.
    ///
    /// Operation: if-let + comparison logic, no own calls.
    pub fn is_own_function_in_scope(&self, name: &str, scope_key: Option<&str>) -> bool {
        if let Some((prefix, method)) = name.split_once("::") {
            if prefix == "Self" {
                return self.methods.contains(method) && !self.trait_only_methods.contains(method);
            }
            self.types.contains(prefix)
                && !method.starts_with(char::is_uppercase)
                && !self.trait_only_methods.contains(method)
        } else {
            self.functions.contains(name)
                || scope_key.is_some_and(|key| {
                    self.nested_functions
                        .get(key)
                        .is_some_and(|nested| nested.contains(name))
                })
        }
    }

    /// Is `name` an own *method* call (dot-style: `.method()`)?
    /// Fallback for receivers with unknown type.
    /// Operation: boolean logic, no own calls.
    pub fn is_own_method(&self, name: &str) -> bool {
        self.methods.contains(name)
            && !self.trivial_methods.contains(name)
            && !self.trait_only_methods.contains(name)
    }

    /// Is `.method()` an own call when called on `self` inside an impl of `parent_type`?
    /// Layer 1: checks if this specific type defines the method.
    /// Operation: lookup logic, no own calls.
    pub fn is_own_self_method(&self, method: &str, parent_type: &str) -> bool {
        self.methods_by_type
            .get(parent_type)
            .map(|m| m.contains(method))
            .unwrap_or(false)
            && !self.trivial_methods.contains(method)
            && !self.trait_only_methods.contains(method)
    }
}

/// AST visitor that collects declarations (functions, methods, types).
struct ScopeCollector {
    functions: HashSet<String>,
    all_functions: HashSet<String>,
    nested_functions: HashMap<String, HashSet<String>>,
    methods: HashSet<String>,
    types: HashSet<String>,
    trivial_candidates: HashSet<String>,
    non_trivial_methods: HashSet<String>,
    /// Methods from trait definitions and `impl Trait for Struct` blocks.
    trait_method_names: HashSet<String>,
    /// Methods from inherent (non-trait) impl blocks only.
    concrete_method_names: HashSet<String>,
    /// Methods grouped by parent type name: `"Config" → {"load", "save"}`.
    methods_by_type: HashMap<String, HashSet<String>>,
    /// Type names with an item definition in the project (pre-pass).
    /// Guards `types` against blanket impls over external generics.
    defined_types: HashSet<String>,
    /// Path of the file being visited — the first part of every scope key.
    file: String,
    /// Scope keys of the functions currently being visited (outermost first).
    fn_scopes: Vec<String>,
    /// Type of the enclosing impl/trait block — part of a method's scope key.
    parent_type: Option<String>,
}

/// Pre-pass visitor: only type definitions, no impls or functions.
struct DefinitionsCollector {
    defined: HashSet<String>,
}

impl<'ast> Visit<'ast> for DefinitionsCollector {
    fn visit_item_struct(&mut self, node: &'ast syn::ItemStruct) {
        self.defined.insert(node.ident.to_string());
    }

    fn visit_item_enum(&mut self, node: &'ast syn::ItemEnum) {
        self.defined.insert(node.ident.to_string());
    }

    fn visit_item_union(&mut self, node: &'ast syn::ItemUnion) {
        self.defined.insert(node.ident.to_string());
    }

    fn visit_item_trait(&mut self, node: &'ast syn::ItemTrait) {
        self.defined.insert(node.ident.to_string());
    }
}

/// Collect item-defined type names across all files (pre-pass for [`ProjectScope::from_files`]).
fn collect_defined_types(files: &[(&str, &File)]) -> HashSet<String> {
    let mut collector = DefinitionsCollector {
        defined: HashSet::new(),
    };
    files
        .iter()
        .for_each(|(_, file)| collector.visit_file(file));
    collector.defined
}

impl<'ast> Visit<'ast> for ScopeCollector {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        let name = node.sig.ident.to_string();
        self.all_functions.insert(name.clone());
        if self.fn_scopes.is_empty() {
            // Module-level function: visible to the whole project.
            self.functions.insert(name.clone());
        } else {
            // Nested function: visible only inside the enclosing functions,
            // so it is filed under their scopes instead of `functions`.
            for scope in &self.fn_scopes {
                self.nested_functions
                    .entry(scope.clone())
                    .or_default()
                    .insert(name.clone());
            }
        }
        let scope = self.scope_key_for(&name);
        self.fn_scopes.push(scope);
        syn::visit::visit_item_fn(self, node);
        self.fn_scopes.pop();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        let scope = self.method_scope_key(&node.sig.ident.to_string());
        self.fn_scopes.push(scope);
        syn::visit::visit_impl_item_fn(self, node);
        self.fn_scopes.pop();
    }

    fn visit_trait_item_fn(&mut self, node: &'ast syn::TraitItemFn) {
        let scope = self.method_scope_key(&node.sig.ident.to_string());
        self.fn_scopes.push(scope);
        syn::visit::visit_trait_item_fn(self, node);
        self.fn_scopes.pop();
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        let type_name = if let syn::Type::Path(tp) = &*node.self_ty {
            tp.path.segments.last().and_then(|seg| {
                let name = seg.ident.to_string();
                // Claim the self-type only when the project defines it.
                // A blanket `impl<T> Marker for Vec<T>` leaves `Vec` out,
                // so `Vec::new` keeps resolving as external (stdlib).
                if self.defined_types.contains(&name) {
                    self.types.insert(name.clone());
                    Some(name)
                } else {
                    None
                }
            })
        } else {
            None
        };
        let is_trait_impl = node.trait_.is_some();
        for item in &node.items {
            if let ImplItem::Fn(method) = item {
                let name = method.sig.ident.to_string();
                self.methods.insert(name.clone());
                if let Some(ref tn) = type_name {
                    self.methods_by_type
                        .entry(tn.clone())
                        .or_default()
                        .insert(name.clone());
                }
                if is_trait_impl {
                    self.trait_method_names.insert(name.clone());
                } else {
                    self.concrete_method_names.insert(name.clone());
                }
                if has_trivial_self_signature(&method.sig)
                    && is_trivial_accessor_body(&method.block)
                {
                    self.trivial_candidates.insert(name);
                } else {
                    self.non_trivial_methods.insert(name);
                }
            }
        }
        // The scope key of a method spells its parent type the same way the
        // IOSP analyzer does, so both sides address the same scope.
        let previous_parent = self.parent_type.replace(
            crate::adapters::shared::impl_type_name::impl_type_name(node).unwrap_or_default(),
        );
        syn::visit::visit_item_impl(self, node);
        self.parent_type = previous_parent;
    }

    fn visit_item_trait(&mut self, node: &'ast syn::ItemTrait) {
        self.types.insert(node.ident.to_string());
        for item in &node.items {
            if let TraitItem::Fn(method) = item {
                let name = method.sig.ident.to_string();
                self.methods.insert(name.clone());
                self.trait_method_names.insert(name);
            }
        }
        let previous_parent = self.parent_type.replace(node.ident.to_string());
        syn::visit::visit_item_trait(self, node);
        self.parent_type = previous_parent;
    }

    fn visit_item_struct(&mut self, node: &'ast syn::ItemStruct) {
        self.types.insert(node.ident.to_string());
        syn::visit::visit_item_struct(self, node);
    }

    fn visit_item_enum(&mut self, node: &'ast syn::ItemEnum) {
        self.types.insert(node.ident.to_string());
        syn::visit::visit_item_enum(self, node);
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        // Recurse into inline modules to collect their declarations too
        syn::visit::visit_item_mod(self, node);
    }
}

impl ScopeCollector {
    /// Scope key of a function declared in the current scope.
    fn scope_key_for(&self, name: &str) -> String {
        match self.fn_scopes.last() {
            Some(parent) => format!("{parent}::{name}"),
            None => function_scope_key(&self.file, None, name),
        }
    }

    /// Scope key of a method: `<file>::<Type>::<method>`.
    fn method_scope_key(&self, name: &str) -> String {
        match self.parent_type.as_deref() {
            Some(parent) if !parent.is_empty() => {
                function_scope_key(&self.file, Some(parent), name)
            }
            _ => function_scope_key(&self.file, None, name),
        }
    }
}
