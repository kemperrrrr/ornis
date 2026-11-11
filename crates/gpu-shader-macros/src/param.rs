//! Parsing of `#[gpu(…)]` and resolution of the parameter type.
//!
//! The attribute names the binding and its access mode; the parameter's Rust
//! type names the element. The macro joins them into the full type the SDK
//! expects — `Storage<f32>` with `group = 0, binding = 1, read_only` becomes
//! `Storage<f32, 0, 1, ReadOnly>` — so there is exactly one place where a
//! binding's group and index are written, and the kernel body sees a type whose
//! methods already know them.
//!
//! Both spellings resolve to the same type: a kernel may write the short form
//! and let the attribute fill the generics in, or spell the generics out and let
//! the macro check that they agree with the attribute.

use syn::Attribute;
use syn::GenericArgument;
use syn::LitInt;
use syn::PathArguments;
use syn::Type;

/// Which binding kind the parameter declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// `var<uniform>` — one struct, read only.
    Uniform,
    /// `var<storage>` — an array, read or read-write.
    StorageBuffer,
    /// Sampled `texture_2d<f32>`.
    Texture2d,
    /// `texture_depth_2d`.
    DepthTexture2d,
    /// Filtering sampler (`sampler`).
    Sampler,
}

impl Kind {
    /// SDK type name the parameter must be written with.
    fn type_name(self) -> &'static str {
        match self {
            Self::Uniform => "Uniform",
            Self::StorageBuffer => "Storage",
            Self::Texture2d => "Texture2d",
            Self::DepthTexture2d => "DepthTexture2d",
            Self::Sampler => "Sampler",
        }
    }

    /// Whether the binding carries only group and binding, no element type.
    fn is_binding_only(self) -> bool {
        matches!(self, Self::Texture2d | Self::DepthTexture2d | Self::Sampler)
    }
}

/// One resolved resource parameter.
#[derive(Clone)]
pub(super) struct Param {
    /// Binding kind of the parameter.
    pub(super) kind: Kind,
    /// Full SDK type, generics filled in.
    pub(super) ty: Type,
    /// Element type: `f32`/`u32`/`AtomicU32` for storage, the struct for a
    /// uniform.
    pub(super) element: Type,
}

impl Param {
    /// Whether the parameter declares a uniform struct.
    pub(super) fn is_uniform(&self) -> bool {
        self.kind == Kind::Uniform
    }
}

/// Parse `#[gpu(…)]` on a parameter and resolve its type.
///
/// `Ok(None)` means the attribute is not `#[gpu]`, so the caller keeps the
/// parameter (the invocation id) untouched.
pub(super) fn parse(attr: &Attribute, ty: &Type) -> syn::Result<Option<Param>> {
    if !attr.path().is_ident("gpu") {
        return Ok(None);
    }
    let mut kind = None;
    let mut group = None;
    let mut binding = None;
    let mut read_only = false;
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("uniform") {
            return set_kind(&mut kind, Kind::Uniform, &meta);
        }
        if meta.path.is_ident("storage_buffer") {
            return set_kind(&mut kind, Kind::StorageBuffer, &meta);
        }
        if meta.path.is_ident("texture_2d") {
            return set_kind(&mut kind, Kind::Texture2d, &meta);
        }
        if meta.path.is_ident("depth_texture_2d") {
            return set_kind(&mut kind, Kind::DepthTexture2d, &meta);
        }
        if meta.path.is_ident("sampler") {
            return set_kind(&mut kind, Kind::Sampler, &meta);
        }
        if meta.path.is_ident("read_only") {
            read_only = true;
            return Ok(());
        }
        if meta.path.is_ident("group") {
            group = Some(index(&meta)?);
            return Ok(());
        }
        if meta.path.is_ident("binding") {
            binding = Some(index(&meta)?);
            return Ok(());
        }
        Err(meta.error(
            "expected `uniform`, `storage_buffer`, `texture_2d`, `depth_texture_2d`, `sampler`, \
             `read_only`, `group = N`, or `binding = N`",
        ))
    })?;
    let kind = kind.ok_or_else(|| {
        syn::Error::new_spanned(
            attr,
            "expected `uniform`, `storage_buffer`, `texture_2d`, `depth_texture_2d`, or \
             `sampler` in the gpu attribute",
        )
    })?;
    if read_only && kind != Kind::StorageBuffer {
        return Err(syn::Error::new_spanned(
            attr,
            "`read_only` applies to `storage_buffer`; a uniform, a texture, or a sampler is already read-only",
        ));
    }
    let group = group
        .ok_or_else(|| syn::Error::new_spanned(attr, "missing `group = N` in the gpu attribute"))?;
    let binding = binding.ok_or_else(|| {
        syn::Error::new_spanned(attr, "missing `binding = N` in the gpu attribute")
    })?;
    let (ty, element) = if kind.is_binding_only() {
        resolve_texture(ty, kind, group, binding)?
    } else {
        resolve(ty, kind, group, binding, read_only)?
    };
    Ok(Some(Param { kind, ty, element }))
}

fn set_kind(
    slot: &mut Option<Kind>,
    kind: Kind,
    meta: &syn::meta::ParseNestedMeta<'_>,
) -> syn::Result<()> {
    if slot.is_some_and(|existing| existing != kind) {
        return Err(meta.error("a parameter declares one binding kind"));
    }
    *slot = Some(kind);
    Ok(())
}

fn index(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<u32> {
    let value: LitInt = meta.value()?.parse()?;
    value.base10_parse()
}

/// Fill in the group, binding, and access generics of the parameter type.
///
/// # Errors
///
/// When the type is not the SDK's `Uniform`/`Storage`, when it carries a
/// different number of generic arguments than the two accepted forms, or when
/// the access mode it names contradicts `read_only`.
fn resolve(
    ty: &Type,
    kind: Kind,
    group: u32,
    binding: u32,
    read_only: bool,
) -> syn::Result<(Type, Type)> {
    let Type::Path(path) = ty else {
        return Err(syn::Error::new_spanned(
            ty,
            format!("a gpu parameter must be `{}(…)`", kind.type_name()),
        ));
    };
    let Some(segment) = path.path.segments.last() else {
        return Err(syn::Error::new_spanned(ty, "empty parameter type"));
    };
    if segment.ident != kind.type_name() {
        return Err(syn::Error::new_spanned(
            ty,
            format!(
                "a `{}` parameter must be `{}`, not `{}`",
                match kind {
                    Kind::Uniform => "uniform",
                    Kind::StorageBuffer => "storage_buffer",
                    Kind::Texture2d => "texture_2d",
                    Kind::DepthTexture2d => "depth_texture_2d",
                    Kind::Sampler => "sampler",
                },
                kind.type_name(),
                segment.ident
            ),
        ));
    }
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return Err(syn::Error::new_spanned(
            ty,
            format!("`{}` needs an element type", kind.type_name()),
        ));
    };
    let generic: Vec<&GenericArgument> = arguments.args.iter().collect();
    let element = match generic.first() {
        Some(GenericArgument::Type(element)) => (*element).clone(),
        _ => {
            return Err(syn::Error::new_spanned(
                ty,
                format!("`{}` needs an element type", kind.type_name()),
            ));
        }
    };
    let access = access_mode(&generic, ty, read_only)?;
    let group = syn::Index::from(group as usize);
    let binding = syn::Index::from(binding as usize);
    let mut resolved = path.clone();
    let Some(segment) = resolved.path.segments.last_mut() else {
        return Err(syn::Error::new_spanned(ty, "empty parameter type"));
    };
    let arguments = match kind {
        Kind::Uniform => syn::parse_quote!(<#element, #group, #binding>),
        Kind::StorageBuffer => syn::parse_quote!(<#element, #group, #binding, #access>),
        Kind::Texture2d | Kind::DepthTexture2d | Kind::Sampler => {
            return Err(syn::Error::new_spanned(
                ty,
                "a texture or sampler binding has no element type",
            ));
        }
    };
    segment.arguments = syn::PathArguments::AngleBracketed(arguments);
    Ok((Type::Path(resolved), element))
}

/// Fill the group and binding of a texture parameter that names no generics.
fn resolve_texture(ty: &Type, kind: Kind, group: u32, binding: u32) -> syn::Result<(Type, Type)> {
    let Type::Path(path) = ty else {
        return Err(syn::Error::new_spanned(
            ty,
            format!("a gpu parameter must be `{}`", kind.type_name()),
        ));
    };
    let Some(segment) = path.path.segments.last() else {
        return Err(syn::Error::new_spanned(ty, "empty parameter type"));
    };
    if segment.ident != kind.type_name() {
        return Err(syn::Error::new_spanned(
            ty,
            format!(
                "a `{}` parameter must be `{}`, not `{}`",
                match kind {
                    Kind::Texture2d => "texture_2d",
                    Kind::DepthTexture2d => "depth_texture_2d",
                    Kind::Sampler => "sampler",
                    Kind::Uniform | Kind::StorageBuffer => kind.type_name(),
                },
                kind.type_name(),
                segment.ident
            ),
        ));
    }
    if !matches!(segment.arguments, PathArguments::None) {
        return Err(syn::Error::new_spanned(
            ty,
            format!(
                "`{}` takes no generic arguments; `group` and `binding` belong to the gpu attribute",
                kind.type_name()
            ),
        ));
    }
    let group = syn::Index::from(group as usize);
    let binding = syn::Index::from(binding as usize);
    let mut resolved = path.clone();
    let Some(segment) = resolved.path.segments.last_mut() else {
        return Err(syn::Error::new_spanned(ty, "empty parameter type"));
    };
    segment.arguments = syn::PathArguments::AngleBracketed(syn::parse_quote!(<#group, #binding>));
    Ok((Type::Path(resolved), ty.clone()))
}

/// Access-mode token of a storage parameter.
///
/// `read_only` in the attribute wins when the type does not name one; when it
/// does, the two must agree.
fn access_mode(
    generic: &[&GenericArgument],
    ty: &Type,
    read_only: bool,
) -> syn::Result<syn::Ident> {
    let named = generic.get(1).and_then(|argument| match argument {
        GenericArgument::Type(Type::Path(path)) => path.path.get_ident().cloned(),
        _ => None,
    });
    if generic.len() > 2 {
        return Err(syn::Error::new_spanned(
            ty,
            "`Storage` takes the element type and optionally an access mode; \
             `group` and `binding` belong to the gpu attribute",
        ));
    }
    match named {
        Some(name) if name == "ReadOnly" || name == "ReadWrite" => {
            let declared_read_only = name == "ReadOnly";
            if declared_read_only != read_only {
                return Err(syn::Error::new_spanned(
                    ty,
                    "the access mode of the type disagrees with the gpu attribute's `read_only`",
                ));
            }
            Ok(name)
        }
        Some(other) => Err(syn::Error::new_spanned(
            ty,
            format!("unknown storage access mode `{other}`; expected `ReadOnly` or `ReadWrite`"),
        )),
        None => Ok(syn::Ident::new(
            if read_only { "ReadOnly" } else { "ReadWrite" },
            syn::spanned::Spanned::span(ty),
        )),
    }
}
