//! Architecture registry (`docs/EN/architecture.md` Sections 1, 7 and 9).
//!
//! A model family is selected by the `model_type` of its `config.json`. Each family turns the
//! config into a [`Layout`] (which sequence mixer / channel mixer every layer uses) assembled from
//! the shared modules in [`blocks`]; the [`Architecture`] built from it provides the component
//! tree on which placement configs (`lumen-config`) are resolved.
//!
//! This crate (together with `lumen-chat`) is the only place that knows model family names.
//!
//! Not implemented yet: weight schemas / name aliases, MoE normalization and the logical
//! operator graph (`Architecture::{weights, normalize, build}` in the architecture doc). They land
//! with the end-to-end CPU pipeline (roadmap Phase 3).

pub mod blocks;
mod families;
pub mod hf;

use lumen_core::{ComponentTree, LumenError, Result};
use lumen_format::LtqHeader;
use serde_json::Value;

pub use blocks::{AttentionSpec, Ffn, LayerSpec, Layout, LinearProjections, Mixer};
pub use hf::HfConfig;

/// A model family instantiated for a concrete `config.json`.
pub trait Architecture: Send + Sync {
    /// Family name (shown by `lumen inspect`).
    fn name(&self) -> &str;

    /// The structural description this architecture was built from.
    fn layout(&self) -> &Layout;

    fn num_hidden_layers(&self) -> usize {
        self.layout().layers.len()
    }

    /// Node tree for configuration and placement.
    fn components(&self) -> ComponentTree {
        self.layout().components()
    }
}

struct ConfiguredArchitecture {
    layout: Layout,
}

impl Architecture for ConfiguredArchitecture {
    fn name(&self) -> &str {
        &self.layout.name
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }
}

/// `model_type` values the registry can instantiate.
pub fn supported_model_types() -> Vec<&'static str> {
    families::FAMILIES
        .iter()
        .flat_map(|f| f.model_types.iter().copied())
        .collect()
}

/// Instantiates the architecture for a HuggingFace-style `config.json`.
///
/// The family is chosen by the root `model_type`; for multimodal wrappers whose root type is not
/// registered, the `model_type` inside `text_config` is tried next. An unknown `model_type` is an
/// error: guessing a family would silently produce a wrong placement namespace.
pub fn architecture_from_config(config: &Value) -> Result<Box<dyn Architecture>> {
    let cfg = HfConfig::new(config)?;
    let find = |mt: Option<&str>| {
        mt.and_then(|mt| {
            families::FAMILIES
                .iter()
                .find(|f| f.model_types.contains(&mt))
        })
    };
    let family = find(cfg.root_model_type())
        .or_else(|| find(cfg.text_model_type()))
        .ok_or_else(|| {
            let seen = cfg
                .root_model_type()
                .or_else(|| cfg.text_model_type())
                .unwrap_or("<missing>");
            LumenError::Format(format!(
                "unsupported model_type '{seen}' (supported: {})",
                supported_model_types().join(", ")
            ))
        })?;
    let mut layout = (family.parse)(&cfg)?;
    layout.name = family.name.to_string();
    Ok(Box::new(ConfiguredArchitecture { layout }))
}

/// Instantiates the architecture described by an `.ltq` header's embedded `config`.
pub fn architecture_from_header(header: &LtqHeader) -> Result<Box<dyn Architecture>> {
    architecture_from_config(&header.config)
}

/// The namespace of every component path any supported architecture can have. Pass it to
/// `RuntimeConfig::resolve_plan` so rules for components that a particular model lacks are skipped
/// instead of rejected as typos.
pub fn known_components() -> ComponentTree {
    blocks::superset_layout().components()
}
