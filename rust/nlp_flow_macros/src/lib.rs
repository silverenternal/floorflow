//! Procedural macros for natural-language flow declarations.

use std::collections::HashSet;

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Attribute, Data, DeriveInput, Error, Fields, LitBool, LitInt, LitStr, Variant,
    parse_macro_input,
};

#[derive(Default)]
struct FlowAttrs {
    id: Option<String>,
    name: Option<String>,
    role: Option<String>,
    max_state_tokens: Option<usize>,
    kind: Option<String>,
    structural: Option<bool>,
    schema: Option<bool>,
    executable: Option<bool>,
}

#[proc_macro_derive(TypedChannel, attributes(nlp_flow))]
pub fn derive_typed_channel(input: TokenStream) -> TokenStream {
    expand_typed_channel(parse_macro_input!(input as DeriveInput))
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

#[proc_macro_derive(HierarchyLevels, attributes(nlp_flow))]
pub fn derive_hierarchy_levels(input: TokenStream) -> TokenStream {
    expand_hierarchy_levels(parse_macro_input!(input as DeriveInput))
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

#[proc_macro_derive(BenchmarkTargets, attributes(nlp_flow))]
pub fn derive_benchmark_targets(input: TokenStream) -> TokenStream {
    expand_benchmark_targets(parse_macro_input!(input as DeriveInput))
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

#[proc_macro_derive(OutputProfiles, attributes(nlp_flow))]
pub fn derive_output_profiles(input: TokenStream) -> TokenStream {
    expand_output_profiles(parse_macro_input!(input as DeriveInput))
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn variants<'a>(
    data: &'a Data,
    macro_name: &str,
) -> syn::Result<&'a syn::punctuated::Punctuated<Variant, syn::token::Comma>> {
    match data {
        Data::Enum(data) => Ok(&data.variants),
        _ => Err(Error::new(
            proc_macro2::Span::call_site(),
            format!("{macro_name} can only be derived for enums"),
        )),
    }
}

fn parse_flow_attrs(attrs: &[Attribute]) -> syn::Result<FlowAttrs> {
    let mut parsed = FlowAttrs::default();
    for attr in attrs {
        if !attr.path().is_ident("nlp_flow") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("id") {
                parsed.id = Some(meta.value()?.parse::<LitStr>()?.value());
            } else if meta.path.is_ident("name") {
                parsed.name = Some(meta.value()?.parse::<LitStr>()?.value());
            } else if meta.path.is_ident("role") {
                parsed.role = Some(meta.value()?.parse::<LitStr>()?.value());
            } else if meta.path.is_ident("max_state_tokens") {
                parsed.max_state_tokens = Some(meta.value()?.parse::<LitInt>()?.base10_parse()?);
            } else if meta.path.is_ident("kind") {
                parsed.kind = Some(meta.value()?.parse::<LitStr>()?.value());
            } else if meta.path.is_ident("structural") {
                parsed.structural = Some(meta.value()?.parse::<LitBool>()?.value());
            } else if meta.path.is_ident("schema") {
                parsed.schema = Some(meta.value()?.parse::<LitBool>()?.value());
            } else if meta.path.is_ident("executable") {
                parsed.executable = Some(meta.value()?.parse::<LitBool>()?.value());
            } else {
                return Err(meta.error("unsupported nlp_flow option"));
            }
            Ok(())
        })?;
    }
    Ok(parsed)
}

fn snake_case(value: &str) -> String {
    let mut rendered = String::with_capacity(value.len() + 4);
    for (index, ch) in value.chars().enumerate() {
        if ch.is_uppercase() {
            if index > 0 {
                rendered.push('_');
            }
            rendered.extend(ch.to_lowercase());
        } else {
            rendered.push(ch);
        }
    }
    rendered
}

fn expand_output_profiles(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = input.ident;
    let variants = variants(&input.data, "OutputProfiles")?;
    let mut names = HashSet::new();
    let mut specs = Vec::new();
    let mut arms = Vec::new();

    for variant in variants {
        if !matches!(variant.fields, Fields::Unit) {
            return Err(Error::new_spanned(
                variant,
                "OutputProfiles variants must be unit variants",
            ));
        }
        let attrs = parse_flow_attrs(&variant.attrs)?;
        let rendered = attrs
            .name
            .unwrap_or_else(|| snake_case(&variant.ident.to_string()));
        if !names.insert(rendered.clone()) {
            return Err(Error::new_spanned(
                &variant.ident,
                format!("duplicate output profile name: {rendered}"),
            ));
        }
        let structural = attrs.structural.ok_or_else(|| {
            Error::new_spanned(&variant.ident, "missing nlp_flow structural requirement")
        })?;
        let schema = attrs.schema.ok_or_else(|| {
            Error::new_spanned(&variant.ident, "missing nlp_flow schema requirement")
        })?;
        let executable = attrs.executable.ok_or_else(|| {
            Error::new_spanned(&variant.ident, "missing nlp_flow executable requirement")
        })?;
        let ident = variant.ident.clone();
        specs.push(quote! {
            ::nlp_flow_core::OutputContractSpec::new(
                #rendered,
                #structural,
                #schema,
                #executable,
            )
        });
        arms.push(quote! {
            Self::#ident => ::nlp_flow_core::OutputContractSpec::new(
                #rendered,
                #structural,
                #schema,
                #executable,
            )
        });
    }

    Ok(quote! {
        impl ::nlp_flow_core::OutputContracts for #name {
            const CONTRACTS: &'static [::nlp_flow_core::OutputContractSpec] =
                &[#(#specs),*];

            fn contract(self) -> ::nlp_flow_core::OutputContractSpec {
                match self {
                    #(#arms),*
                }
            }
        }
    })
}

fn expand_benchmark_targets(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = input.ident;
    let variants = variants(&input.data, "BenchmarkTargets")?;
    let mut names = HashSet::new();
    let mut arms = Vec::new();

    for variant in variants {
        if !matches!(variant.fields, Fields::Unit) {
            return Err(Error::new_spanned(
                variant,
                "BenchmarkTargets variants must be unit variants",
            ));
        }
        let attrs = parse_flow_attrs(&variant.attrs)?;
        let rendered = attrs
            .name
            .unwrap_or_else(|| snake_case(&variant.ident.to_string()));
        if !names.insert(rendered.clone()) {
            return Err(Error::new_spanned(
                &variant.ident,
                format!("duplicate benchmark target name: {rendered}"),
            ));
        }
        let kind = match attrs.kind.as_deref() {
            Some("scalar") => quote!(::nlp_flow_core::BenchmarkTargetKind::Scalar),
            Some("token") => quote!(::nlp_flow_core::BenchmarkTargetKind::Token),
            Some("regression") => quote!(::nlp_flow_core::BenchmarkTargetKind::Regression),
            Some(value) => {
                return Err(Error::new_spanned(
                    &variant.ident,
                    format!("unsupported benchmark target kind: {value}"),
                ));
            }
            None => {
                return Err(Error::new_spanned(
                    &variant.ident,
                    "missing nlp_flow kind = \"scalar|token|regression\"",
                ));
            }
        };
        let ident = variant.ident.clone();
        arms.push(quote!(Self::#ident => #kind));
    }

    Ok(quote! {
        impl ::nlp_flow_core::BenchmarkTarget for #name {
            fn kind(self) -> ::nlp_flow_core::BenchmarkTargetKind {
                match self {
                    #(#arms),*
                }
            }
        }
    })
}

fn expand_typed_channel(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = input.ident;
    let variants = variants(&input.data, "TypedChannel")?;
    let mut names = HashSet::new();
    let mut idents = Vec::new();
    let mut rendered_names = Vec::new();
    let mut indices = Vec::new();

    for (index, variant) in variants.iter().enumerate() {
        if !matches!(variant.fields, Fields::Unit) {
            return Err(Error::new_spanned(
                variant,
                "TypedChannel variants must be unit variants",
            ));
        }
        let attrs = parse_flow_attrs(&variant.attrs)?;
        let rendered = attrs
            .name
            .unwrap_or_else(|| snake_case(&variant.ident.to_string()));
        if !names.insert(rendered.clone()) {
            return Err(Error::new_spanned(
                &variant.ident,
                format!("duplicate TypedChannel name: {rendered}"),
            ));
        }
        idents.push(variant.ident.clone());
        rendered_names.push(rendered);
        indices.push(index);
    }

    Ok(quote! {
        impl ::nlp_flow_core::TypedChannel for #name {
            const ALL: &'static [Self] = &[#(Self::#idents),*];

            fn as_str(self) -> &'static str {
                match self {
                    #(Self::#idents => #rendered_names),*
                }
            }

            fn index(self) -> usize {
                match self {
                    #(Self::#idents => #indices),*
                }
            }

            fn from_name(name: &str) -> Option<Self> {
                match name {
                    #(#rendered_names => Some(Self::#idents),)*
                    _ => None,
                }
            }
        }

        impl ::core::fmt::Display for #name {
            fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                formatter.write_str(::nlp_flow_core::TypedChannel::as_str(*self))
            }
        }
    })
}

fn expand_hierarchy_levels(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = input.ident;
    let variants = variants(&input.data, "HierarchyLevels")?;
    let mut ids = HashSet::new();
    let mut arms = Vec::new();
    let mut from_arms = Vec::new();
    let mut specs = Vec::new();

    for variant in variants {
        if !matches!(variant.fields, Fields::Unit) {
            return Err(Error::new_spanned(
                variant,
                "HierarchyLevels variants must be unit variants",
            ));
        }
        let attrs = parse_flow_attrs(&variant.attrs)?;
        let id = attrs
            .id
            .ok_or_else(|| Error::new_spanned(&variant.ident, "missing nlp_flow(id = \"...\")"))?;
        if !ids.insert(id.clone()) {
            return Err(Error::new_spanned(
                &variant.ident,
                format!("duplicate hierarchy id: {id}"),
            ));
        }
        let rendered_name = attrs
            .name
            .unwrap_or_else(|| snake_case(&variant.ident.to_string()));
        let role = attrs
            .role
            .ok_or_else(|| Error::new_spanned(&variant.ident, "missing nlp_flow role"))?;
        let max_tokens = attrs.max_state_tokens.ok_or_else(|| {
            Error::new_spanned(&variant.ident, "missing nlp_flow max_state_tokens")
        })?;
        if max_tokens == 0 {
            return Err(Error::new_spanned(
                &variant.ident,
                "max_state_tokens must be positive",
            ));
        }
        let ident = variant.ident.clone();
        specs.push(quote! {
            ::nlp_flow_core::HierarchySpec::new(#id, #rendered_name, #role, #max_tokens)
        });
        arms.push(quote! {
            Self::#ident => ::nlp_flow_core::HierarchySpec::new(
                #id,
                #rendered_name,
                #role,
                #max_tokens,
            )
        });
        from_arms.push(quote!(#id => Some(Self::#ident)));
    }

    Ok(quote! {
        impl ::nlp_flow_core::HierarchyLevels for #name {
            const SPECS: &'static [::nlp_flow_core::HierarchySpec] = &[#(#specs),*];

            fn spec(self) -> ::nlp_flow_core::HierarchySpec {
                match self {
                    #(#arms),*
                }
            }

            fn from_id(id: &str) -> Option<Self> {
                match id {
                    #(#from_arms,)*
                    _ => None,
                }
            }
        }

        impl ::core::fmt::Display for #name {
            fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                formatter.write_str(::nlp_flow_core::HierarchyLevels::level_id(*self))
            }
        }
    })
}
