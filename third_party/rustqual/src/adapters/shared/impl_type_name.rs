//! The type name of an `impl` block, spelled once for the whole analyzer.
//!
//! The IOSP analyzer uses it as the parent type of a method, and the
//! project-scope collector uses it to build the scope key of that method.
//! Both sides must spell the type identically, or a method and its scope
//! stop matching.

use syn::ItemImpl;

/// Extract the type name from an impl block, including generic parameters.
/// Operation: if-let logic.
pub fn impl_type_name(item_impl: &ItemImpl) -> Option<String> {
    if let syn::Type::Path(tp) = &*item_impl.self_ty {
        tp.path.segments.last().map(|s| {
            let name = s.ident.to_string();
            match &s.arguments {
                syn::PathArguments::AngleBracketed(args) => {
                    let params: Vec<String> = args
                        .args
                        .iter()
                        .map(|a| quote::quote!(#a).to_string())
                        .collect();
                    format!("{name}<{}>", params.join(","))
                }
                _ => name,
            }
        })
    } else {
        None
    }
}
