//! Typed access to a HuggingFace-style `config.json` (the `config` object of an `.ltq` header).
//!
//! Configs are untrusted input: every accessor returns an error with the offending key instead of
//! panicking, and sizes are bounded before anything is allocated from them.

use lumen_core::{LumenError, Result};
use serde_json::Value;

/// Upper bound on `num_hidden_layers`, to keep a hostile config from requesting absurd allocations.
pub const MAX_HIDDEN_LAYERS: usize = 4096;

/// A model `config.json`: the root object and, for multimodal wrappers, its `text_config`.
#[derive(Clone, Copy, Debug)]
pub struct HfConfig<'a> {
    root: &'a Value,
    text: &'a Value,
}

impl<'a> HfConfig<'a> {
    pub fn new(root: &'a Value) -> Result<Self> {
        if !root.is_object() {
            return Err(err("config must be a JSON object"));
        }
        let text = root
            .get("text_config")
            .filter(|v| v.is_object())
            .unwrap_or(root);
        Ok(Self { root, text })
    }

    /// `model_type` of the root object (e.g. a multimodal wrapper).
    pub fn root_model_type(&self) -> Option<&'a str> {
        self.root.get("model_type").and_then(Value::as_str)
    }

    /// `model_type` of the text model (inside `text_config` when present).
    pub fn text_model_type(&self) -> Option<&'a str> {
        self.text.get("model_type").and_then(Value::as_str)
    }

    /// Whether the root object has `key` (used for optional towers such as `vision_config`).
    pub fn root_has(&self, key: &str) -> bool {
        self.root.get(key).is_some_and(|v| !v.is_null())
    }

    /// Looks `key` up in the text config.
    pub fn get(&self, key: &str) -> Option<&'a Value> {
        self.text.get(key).filter(|v| !v.is_null())
    }

    pub fn opt_usize(&self, key: &str) -> Result<Option<usize>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .map(Some)
                .ok_or_else(|| err(format!("'{key}' must be a non-negative integer"))),
        }
    }

    pub fn usize_or(&self, key: &str, default: usize) -> Result<usize> {
        Ok(self.opt_usize(key)?.unwrap_or(default))
    }

    pub fn bool_or(&self, key: &str, default: bool) -> Result<bool> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v
                .as_bool()
                .ok_or_else(|| err(format!("'{key}' must be a boolean"))),
        }
    }

    /// `num_hidden_layers`, required and bounded by [`MAX_HIDDEN_LAYERS`].
    pub fn num_hidden_layers(&self) -> Result<usize> {
        let n = self
            .opt_usize("num_hidden_layers")?
            .ok_or_else(|| err("missing 'num_hidden_layers'"))?;
        if n == 0 || n > MAX_HIDDEN_LAYERS {
            return Err(err(format!(
                "'num_hidden_layers' = {n} is outside 1..={MAX_HIDDEN_LAYERS}"
            )));
        }
        Ok(n)
    }

    /// Optional `layer_types` array; when present it must have exactly one string per layer.
    pub fn layer_types(&self, num_layers: usize) -> Result<Option<Vec<&'a str>>> {
        let Some(v) = self.get("layer_types") else {
            return Ok(None);
        };
        let arr = v
            .as_array()
            .ok_or_else(|| err("'layer_types' must be an array"))?;
        if arr.len() != num_layers {
            return Err(err(format!(
                "'layer_types' has {} entries but 'num_hidden_layers' is {num_layers}",
                arr.len()
            )));
        }
        arr.iter()
            .map(|x| {
                x.as_str()
                    .ok_or_else(|| err("'layer_types' entries must be strings"))
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    /// Optional array of layer indices (e.g. `mlp_only_layers`), each bounded by `num_layers`.
    pub fn layer_index_list(&self, key: &str, num_layers: usize) -> Result<Vec<usize>> {
        let Some(v) = self.get(key) else {
            return Ok(Vec::new());
        };
        let arr = v
            .as_array()
            .ok_or_else(|| err(format!("'{key}' must be an array")))?;
        arr.iter()
            .map(|x| {
                x.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|&n| n < num_layers)
                    .ok_or_else(|| {
                        err(format!(
                            "'{key}' entries must be layer indices < {num_layers}"
                        ))
                    })
            })
            .collect()
    }
}

pub(crate) fn err(msg: impl Into<String>) -> LumenError {
    LumenError::Format(format!("model config: {}", msg.into()))
}
