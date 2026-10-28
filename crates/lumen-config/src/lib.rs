//! Configuration parser, cascade resolver, layer-range selector grammar, and `PlacementPlan`
//! builder (`docs/RU/config.md` and `docs/RU/architecture.md` Section 3).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use lumen_core::{
    AccessMode, ComponentKind, ComponentPath, ComponentTree, Device, LumenError, MemLoc, Result,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// ============================================================================
// Resolved Placement Types (Architecture Section 3)
// ============================================================================

/// Resolved placement for a single component at decode and prefill time.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Place {
    /// Compute device during token generation (decode).
    pub compute: Device,
    /// Persistent memory location for weights/cache/state.
    pub store: MemLoc,
    /// Optional compute device override during prompt processing (prefill).
    pub prefill_compute: Option<Device>,
}

impl Place {
    /// Access mode during decode for a component of the given `ComponentKind`.
    #[inline]
    pub const fn decode_access_mode(&self, kind: ComponentKind) -> AccessMode {
        let sparse = matches!(kind, ComponentKind::SparseWeight | ComponentKind::KvCache);
        AccessMode::resolve(self.store, self.compute, sparse)
    }

    /// Effective compute device during prefill.
    #[inline]
    pub const fn effective_prefill_compute(&self) -> Device {
        match self.prefill_compute {
            Some(dev) => dev,
            None => self.compute,
        }
    }

    /// Access mode during prefill for a component of the given `ComponentKind`.
    /// When `prefill-compute` is explicitly overridden to a GPU for RAM-stored weights
    /// (e.g. `moe/routed-experts` or `dense-mlp`), weights are streamed H2D per chunk (`Streamed`).
    #[inline]
    pub const fn prefill_access_mode(&self, kind: ComponentKind) -> AccessMode {
        let sparse = if self.prefill_compute.is_some() {
            false
        } else {
            matches!(kind, ComponentKind::SparseWeight | ComponentKind::KvCache)
        };
        AccessMode::resolve(self.store, self.effective_prefill_compute(), sparse)
    }
}

/// Per-layer KV cache placement parameters.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvPlacement {
    pub layer: usize,
    pub place: Place,
    pub k_bits: Option<u8>,
    pub v_bits: Option<u8>,
}

/// Per-layer recurrent state placement parameters (e.g., `GatedDeltaNet` state).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct StatePlacement {
    pub layer: usize,
    pub place: Place,
}

/// Records the JSON configuration rule path from which each property of a `Place` originated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlaceOrigin {
    pub compute_source: String,
    pub store_source: String,
    pub prefill_source: Option<String>,
}

impl PlaceOrigin {
    pub fn summary(&self) -> String {
        if self.compute_source == self.store_source {
            match &self.prefill_source {
                Some(p) if p != &self.compute_source => {
                    format!("{} (prefill: {})", self.compute_source, p)
                }
                _ => self.compute_source.clone(),
            }
        } else {
            format!(
                "compute: {}, store: {}",
                self.compute_source, self.store_source
            )
        }
    }
}

/// Provenance tracking for `lumen inspect`: rule origins, skipped rules, and non-fatal warnings.
#[derive(Clone, Debug, Default)]
pub struct Provenance {
    pub global_origins: BTreeMap<ComponentPath, PlaceOrigin>,
    pub layer_origins: Vec<BTreeMap<ComponentPath, PlaceOrigin>>,
    pub skipped_rules: Vec<String>,
    pub warnings: Vec<String>,
}

/// Flat, validated placement plan produced once at startup from the JSON config and `ComponentTree`.
#[derive(Clone, Debug)]
pub struct PlacementPlan {
    pub tree: ComponentTree,
    /// Placement of global (non-layer) components (`input-layer`, `ple/table`, `output-layer/lm-head`, etc.).
    pub global: BTreeMap<ComponentPath, Place>,
    /// Per-layer placement indexed by `0..num_hidden_layers`.
    pub layers: Vec<BTreeMap<ComponentPath, Place>>,
    /// Per-layer KV cache placement for layers that contain an `attention/kv-cache` node.
    pub kv: Vec<KvPlacement>,
    /// Per-layer recurrent state placement for layers that contain a `linear-attention/state` node.
    pub state: Vec<StatePlacement>,
    /// Rule provenance and diagnostics for `lumen inspect`.
    pub provenance: Provenance,
}

// ============================================================================
// Runtime & Backend Configuration Sections (Config Section 4)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChatTemplateConfig {
    #[serde(default)]
    pub kwargs: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextConfig {
    pub size: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self { size: 32768 }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KvQuantizationConfig {
    pub k_bits: u8,
    pub v_bits: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CudaRuntimeConfig {
    pub prefill_chunk_size: usize,
    pub cuda_graphs: bool,
}

impl Default for CudaRuntimeConfig {
    fn default() -> Self {
        Self {
            prefill_chunk_size: 4096,
            cuda_graphs: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuRuntimeConfig {
    pub threads: usize,
    pub reserve_cores: usize,
    pub numa_spread: bool,
}

impl Default for CpuRuntimeConfig {
    fn default() -> Self {
        Self {
            threads: 0,
            reserve_cores: 1,
            numa_spread: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiskRuntimeConfig {
    pub io_threads: usize,
    pub mlock: bool,
}

impl Default for DiskRuntimeConfig {
    fn default() -> Self {
        Self {
            io_threads: 4,
            mlock: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerRuntimeConfig {
    pub host: String,
    pub port: u16,
    pub max_concurrent_requests: usize,
}

impl Default for ServerRuntimeConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_concurrent_requests: 1,
        }
    }
}

/// Complete parsed configuration document.
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub model_path: String,
    pub sampling: SamplingConfig,
    pub chat_template: ChatTemplateConfig,
    pub context: ContextConfig,
    pub kv_quantization: Option<KvQuantizationConfig>,
    pub cuda: CudaRuntimeConfig,
    pub cpu: CpuRuntimeConfig,
    pub disk: DiskRuntimeConfig,
    pub server: ServerRuntimeConfig,
    pub attention_mechanism: String,
    raw_model: Map<String, Value>,
}

// ============================================================================
// Partial Placement Rule Node & Layer Range Grammar
// ============================================================================

#[derive(Clone, Debug, Default)]
struct PartialPlace {
    compute: Option<(Device, String)>,
    store: Option<(MemLoc, String)>,
    prefill_compute: Option<(Device, String)>,
}

impl PartialPlace {
    fn merge_from(&mut self, child: &Self) {
        if let Some(c) = &child.compute {
            self.compute = Some(c.clone());
        }
        if let Some(s) = &child.store {
            self.store = Some(s.clone());
        }
        if let Some(p) = &child.prefill_compute {
            self.prefill_compute = Some(p.clone());
        }
    }
}

/// Parsed layer selector (e.g., `"12"`, `"0-15"`, or `"0-7,24-31"`).
#[derive(Clone, Debug)]
pub struct LayerSelector {
    pub raw: String,
    pub ranges: Vec<(usize, usize)>,
}

impl LayerSelector {
    /// Returns `true` if `key` looks like a layer selector (starts with an ASCII digit).
    pub fn looks_like_selector(key: &str) -> bool {
        key.chars().next().is_some_and(|c| c.is_ascii_digit())
    }

    pub fn parse(key: &str) -> Result<Self> {
        let mut ranges = Vec::new();
        for part in key.split(',') {
            let part = part.trim();
            if part.is_empty() {
                return Err(LumenError::Config(format!(
                    "empty segment in layer selector '{key}'"
                )));
            }
            if let Some((start_s, end_s)) = part.split_once('-') {
                let start = start_s.trim().parse::<usize>().map_err(|_| {
                    LumenError::Config(format!("invalid layer start in selector '{key}'"))
                })?;
                let end = end_s.trim().parse::<usize>().map_err(|_| {
                    LumenError::Config(format!("invalid layer end in selector '{key}'"))
                })?;
                if start > end {
                    return Err(LumenError::Config(format!(
                        "layer range start ({start}) > end ({end}) in selector '{key}'"
                    )));
                }
                ranges.push((start, end));
            } else {
                let idx = part.parse::<usize>().map_err(|_| {
                    LumenError::Config(format!("invalid layer index in selector '{key}'"))
                })?;
                ranges.push((idx, idx));
            }
        }
        Ok(Self {
            raw: key.to_string(),
            ranges,
        })
    }

    pub fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.ranges.iter().flat_map(|&(s, e)| s..=e)
    }
}

// ============================================================================
// Known Keys & Typo Suggestion (Damerau-Levenshtein)
// ============================================================================

const TOP_LEVEL_KEYS: &[&str] = &[
    "model",
    "chat-template",
    "context",
    "kv-quantization",
    "cuda",
    "cpu",
    "disk",
    "server",
    "attention-mechanism",
];

const PLACEMENT_KEYS: &[&str] = &["compute", "store", "prefill-compute"];

const MODEL_META_KEYS: &[&str] = &[
    "path",
    "compute",
    "store",
    "prefill-compute",
    "sampling",
    "hidden-layers",
];

/// Canonical global component prefixes allowed under `"model"` even if a specific model
/// does not use them (in which case they are recorded as skipped in `Provenance`).
const KNOWN_GLOBAL_PREFIXES: &[&str] = &[
    "input-layer",
    "ple",
    "ple/table",
    "ple/projections",
    "vision-tower",
    "audio-tower",
    "output-layer",
    "output-layer/norm",
    "output-layer/lm-head",
    "mtp",
];

/// Canonical per-layer component prefixes allowed under `"hidden-layers"` even if a specific
/// model/layer does not use them.
const KNOWN_LAYER_PREFIXES: &[&str] = &[
    "norms",
    "norms/input-norm",
    "norms/post-attention-norm",
    "attention",
    "attention/q",
    "attention/k",
    "attention/v",
    "attention/o",
    "attention/qk-norm",
    "attention/rope",
    "attention/indexer",
    "attention/indexer/keys",
    "attention/kv-cache",
    "linear-attention",
    "linear-attention/projections",
    "linear-attention/projections/qkv",
    "linear-attention/projections/z",
    "linear-attention/projections/b",
    "linear-attention/projections/a",
    "linear-attention/projections/qkvz",
    "linear-attention/projections/ba",
    "linear-attention/projections/out",
    "linear-attention/conv",
    "linear-attention/norm",
    "linear-attention/state",
    "hyper-connections",
    "dense-mlp",
    "dense-mlp/gate",
    "dense-mlp/up",
    "dense-mlp/down",
    "moe",
    "moe/router",
    "moe/shared-experts",
    "moe/shared-experts/gate",
    "moe/shared-experts/projections",
    "moe/routed-experts",
    "moe/routed-experts/projections",
];

fn edit_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
        row[0] = i;
    }
    for (j, val) in dp[0].iter_mut().enumerate().take(n + 1) {
        *val = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                dp[i][j] = dp[i][j].min(dp[i - 2][j - 2] + 1);
            }
        }
    }
    dp[m][n]
}

fn suggest_key<'a>(unknown: &str, candidates: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    candidates
        .into_iter()
        .map(|cand| (cand, edit_distance(unknown, cand)))
        .filter(|(_, dist)| *dist <= 3)
        .min_by_key(|(_, dist)| *dist)
        .map(|(cand, _)| cand)
}

fn unknown_key_error(scope: &str, key: &str, valid: &[&str]) -> LumenError {
    if let Some(suggestion) = suggest_key(key, valid.iter().copied()) {
        LumenError::Config(format!(
            "unknown key '{key}' in {scope}; did you mean '{suggestion}'?"
        ))
    } else {
        LumenError::Config(format!("unknown key '{key}' in {scope}"))
    }
}

// ============================================================================
// Parsing & Resolution Implementation
// ============================================================================

impl RuntimeConfig {
    /// Reads and parses a JSON configuration file from disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::from_json_str(&content)
    }

    /// Parses a JSON configuration string and validates all top-level and section keys.
    pub fn from_json_str(json_str: &str) -> Result<Self> {
        let root_val: Value = serde_json::from_str(json_str)
            .map_err(|e| LumenError::Config(format!("invalid JSON syntax: {e}")))?;
        let root = root_val
            .as_object()
            .ok_or_else(|| LumenError::Config("root JSON value must be an object".to_string()))?;

        for key in root.keys() {
            if !TOP_LEVEL_KEYS.contains(&key.as_str()) {
                return Err(unknown_key_error("top-level config", key, TOP_LEVEL_KEYS));
            }
        }

        let model_obj = root
            .get("model")
            .and_then(Value::as_object)
            .ok_or_else(|| LumenError::Config("missing required 'model' object".to_string()))?;

        let model_path = model_obj
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| LumenError::Config("missing required 'model.path' string".to_string()))?
            .to_string();

        let sampling = if let Some(s_val) = model_obj.get("sampling") {
            parse_sampling(s_val)?
        } else {
            SamplingConfig::default()
        };

        let chat_template = if let Some(ct_val) = root.get("chat-template") {
            parse_chat_template(ct_val)?
        } else {
            ChatTemplateConfig::default()
        };

        let context = if let Some(ctx_val) = root.get("context") {
            parse_context(ctx_val)?
        } else {
            ContextConfig::default()
        };

        let kv_quantization = if let Some(kv_val) = root.get("kv-quantization") {
            Some(parse_kv_quantization(kv_val)?)
        } else {
            None
        };

        let cuda = if let Some(cuda_val) = root.get("cuda") {
            parse_cuda_config(cuda_val)?
        } else {
            CudaRuntimeConfig::default()
        };

        let cpu = if let Some(cpu_val) = root.get("cpu") {
            parse_cpu_config(cpu_val)?
        } else {
            CpuRuntimeConfig::default()
        };

        let disk = if let Some(disk_val) = root.get("disk") {
            parse_disk_config(disk_val)?
        } else {
            DiskRuntimeConfig::default()
        };

        let server = if let Some(srv_val) = root.get("server") {
            parse_server_config(srv_val)?
        } else {
            ServerRuntimeConfig::default()
        };

        let attention_mechanism = match root.get("attention-mechanism") {
            Some(v) => v
                .as_str()
                .ok_or_else(|| {
                    LumenError::Config("'attention-mechanism' must be a string".to_string())
                })?
                .to_string(),
            None => "flash-attention".to_string(),
        };

        Ok(Self {
            model_path,
            sampling,
            chat_template,
            context,
            kv_quantization,
            cuda,
            cpu,
            disk,
            server,
            attention_mechanism,
            raw_model: model_obj.clone(),
        })
    }

    /// Infers a default architecture preset and layer count when inspecting a config file
    /// without a concrete `.ltq` file on disk.
    pub fn infer_tree_for_inspect(
        &self,
        override_preset: Option<&str>,
        override_layers: Option<usize>,
    ) -> Result<ComponentTree> {
        // Scan layer selectors under hidden-layers to find the highest referenced layer index
        let mut max_selector_layer: Option<usize> = None;
        let mut mentions_moe = false;
        let mut mentions_dense_mlp = false;
        let mut mentions_ple = self.raw_model.contains_key("ple");

        if let Some(hl) = self.raw_model.get("hidden-layers").and_then(Value::as_object) {
            if hl.contains_key("moe") {
                mentions_moe = true;
            }
            if hl.contains_key("dense-mlp") {
                mentions_dense_mlp = true;
            }
            for (k, v) in hl {
                if LayerSelector::looks_like_selector(k) {
                    let sel = LayerSelector::parse(k)?;
                    if let Some(max_l) = sel.layers().max() {
                        max_selector_layer =
                            Some(max_selector_layer.map_or(max_l, |prev| prev.max(max_l)));
                    }
                    if let Some(sub) = v.as_object() {
                        if sub.contains_key("moe") {
                            mentions_moe = true;
                        }
                        if sub.contains_key("dense-mlp") {
                            mentions_dense_mlp = true;
                        }
                    }
                }
            }
        }

        let path_lower = self.model_path.to_ascii_lowercase();
        if path_lower.contains("gemma") {
            mentions_ple = true;
        }

        let preset = if let Some(p) = override_preset {
            p
        } else if path_lower.contains("flash-next") || path_lower.contains("qwen4") || (mentions_moe && mentions_ple) {
            "qwen4_exp"
        } else if path_lower.contains("qwen3-next") || path_lower.contains("qwen3_next") || mentions_moe {
            "qwen3_next"
        } else if path_lower.contains("gemma") {
            "gemma4"
        } else if mentions_dense_mlp && !mentions_moe {
            "dense"
        } else {
            "canonical-superset"
        };

        let num_layers = if let Some(n) = override_layers {
            n
        } else if let Some(max_l) = max_selector_layer {
            max_l + 1
        } else if path_lower.contains("27b") {
            64
        } else if path_lower.contains("flash-next") {
            48
        } else {
            32
        };

        Ok(ComponentTree::preset(preset, num_layers))
    }

    /// Resolves the cascading configuration rules against the given `ComponentTree` into a
    /// validated, flat `PlacementPlan`.
    pub fn resolve_plan(&self, tree: ComponentTree, strict: bool) -> Result<PlacementPlan> {
        let mut provenance = Provenance {
            global_origins: BTreeMap::new(),
            layer_origins: vec![BTreeMap::new(); tree.num_hidden_layers],
            skipped_rules: Vec::new(),
            warnings: Vec::new(),
        };

        // 1. Validate keys in "model" and collect global rules first (so typos like "stor"
        //    produce a "did you mean 'store'?" error before checking required base keys).
        let mut global_rules: BTreeMap<ComponentPath, PartialPlace> = BTreeMap::new();
        let valid_global_roots: Vec<&str> = KNOWN_GLOBAL_PREFIXES
            .iter()
            .copied()
            .filter(|p| !p.contains('/'))
            .chain(MODEL_META_KEYS.iter().copied())
            .collect();

        for (key, val) in &self.raw_model {
            if MODEL_META_KEYS.contains(&key.as_str()) {
                continue;
            }
            let path = ComponentPath::new(key);
            if !KNOWN_GLOBAL_PREFIXES.contains(&path.as_str())
                && !tree.is_valid_global_path(&path)
            {
                return Err(unknown_key_error("model", key, &valid_global_roots));
            }
            collect_subtree_rules(
                val,
                &path,
                &format!("model.{key}"),
                false,
                &tree,
                &mut global_rules,
                &mut provenance.skipped_rules,
            )?;
        }

        // 2. Extract root placement from "model"
        let root_place = extract_partial_place(&self.raw_model, "model")?;
        let base_compute = root_place.compute.clone().ok_or_else(|| {
            LumenError::Config("missing required default 'model.compute' device".to_string())
        })?;
        let base_store = root_place.store.clone().ok_or_else(|| {
            LumenError::Config("missing required default 'model.store' location".to_string())
        })?;

        // 3. Collect general "hidden-layers" rules and layer-selector rules
        let mut general_layer_rules: BTreeMap<ComponentPath, PartialPlace> = BTreeMap::new();
        let mut selector_layer_rules: Vec<BTreeMap<ComponentPath, PartialPlace>> =
            vec![BTreeMap::new(); tree.num_hidden_layers];
        let mut layer_claimed_by: BTreeMap<usize, String> = BTreeMap::new();

        if let Some(hl_val) = self.raw_model.get("hidden-layers") {
            let hl_obj = hl_val.as_object().ok_or_else(|| {
                LumenError::Config("'model.hidden-layers' must be an object".to_string())
            })?;

            // Base placement keys directly on "hidden-layers"
            let hl_self = extract_partial_place(hl_obj, "hidden-layers")?;
            if hl_self.compute.is_some()
                || hl_self.store.is_some()
                || hl_self.prefill_compute.is_some()
            {
                general_layer_rules.insert(ComponentPath::new(""), hl_self);
            }

            let valid_layer_roots: Vec<&str> = KNOWN_LAYER_PREFIXES
                .iter()
                .copied()
                .filter(|p| !p.contains('/'))
                .chain(PLACEMENT_KEYS.iter().copied())
                .collect();

            for (key, val) in hl_obj {
                if PLACEMENT_KEYS.contains(&key.as_str()) {
                    continue;
                }
                if LayerSelector::looks_like_selector(key) {
                    let selector = LayerSelector::parse(key)?;
                    let mut out_of_range = Vec::new();
                    let mut in_range = Vec::new();

                    for layer_idx in selector.layers() {
                        if let Some(prev_sel) =
                            layer_claimed_by.insert(layer_idx, selector.raw.clone())
                        {
                            return Err(LumenError::Config(format!(
                                "overlapping layer selectors '{}' and '{}' both match layer {}",
                                prev_sel, selector.raw, layer_idx
                            )));
                        }
                        if layer_idx >= tree.num_hidden_layers {
                            out_of_range.push(layer_idx);
                        } else {
                            in_range.push(layer_idx);
                        }
                    }

                    if !out_of_range.is_empty() {
                        let msg = format!(
                            "layer selector '{}' references layers {:?} outside model range [0, {})",
                            selector.raw, out_of_range, tree.num_hidden_layers
                        );
                        if strict {
                            return Err(LumenError::Config(msg));
                        }
                        provenance.warnings.push(msg);
                    }

                    // Parse subtree inside the selector
                    let sel_obj = val.as_object().ok_or_else(|| {
                        LumenError::Config(format!(
                            "layer selector '{}' in 'hidden-layers' must be an object",
                            selector.raw
                        ))
                    })?;
                    let sel_origin = format!("hidden-layers[\"{}\"]", selector.raw);
                    let mut parsed_sel_rules: BTreeMap<ComponentPath, PartialPlace> =
                        BTreeMap::new();

                    let sel_self = extract_partial_place(sel_obj, &sel_origin)?;
                    if sel_self.compute.is_some()
                        || sel_self.store.is_some()
                        || sel_self.prefill_compute.is_some()
                    {
                        parsed_sel_rules.insert(ComponentPath::new(""), sel_self);
                    }

                    for (sub_k, sub_v) in sel_obj {
                        if PLACEMENT_KEYS.contains(&sub_k.as_str()) {
                            continue;
                        }
                        let sub_path = ComponentPath::new(sub_k);
                        if !KNOWN_LAYER_PREFIXES.contains(&sub_path.as_str())
                            && !tree.is_valid_layer_path(&sub_path)
                        {
                            return Err(unknown_key_error(
                                &sel_origin,
                                sub_k,
                                &valid_layer_roots,
                            ));
                        }
                        collect_subtree_rules(
                            sub_v,
                            &sub_path,
                            &format!("{sel_origin}.{sub_k}"),
                            true,
                            &tree,
                            &mut parsed_sel_rules,
                            &mut provenance.skipped_rules,
                        )?;
                    }

                    for layer_idx in in_range {
                        for (p, r) in &parsed_sel_rules {
                            // Check if this rule applies to any active component on this specific layer
                            let applies_on_layer = p.is_empty()
                                || tree.layer_leaves.values().any(|leaf| {
                                    leaf.path.starts_with_path(p)
                                        && leaf.active_layers.contains(&layer_idx)
                                });
                            if !applies_on_layer {
                                let skip_msg = format!(
                                    "{sel_origin}.{p} skipped on layer {layer_idx} (component absent on this layer)"
                                );
                                if !provenance.skipped_rules.contains(&skip_msg) {
                                    provenance.skipped_rules.push(skip_msg);
                                }
                            }
                            selector_layer_rules[layer_idx].insert(p.clone(), r.clone());
                        }
                    }
                } else {
                    let path = ComponentPath::new(key);
                    if !KNOWN_LAYER_PREFIXES.contains(&path.as_str())
                        && !tree.is_valid_layer_path(&path)
                    {
                        return Err(unknown_key_error(
                            "hidden-layers",
                            key,
                            &valid_layer_roots,
                        ));
                    }
                    collect_subtree_rules(
                        val,
                        &path,
                        &format!("hidden-layers.{key}"),
                        true,
                        &tree,
                        &mut general_layer_rules,
                        &mut provenance.skipped_rules,
                    )?;
                }
            }
        }

        // 4. Resolve every global leaf component
        let mut global = BTreeMap::new();
        for path in tree.global_leaves.keys() {
            let mut resolved = PartialPlace {
                compute: Some(base_compute.clone()),
                store: Some(base_store.clone()),
                prefill_compute: root_place.prefill_compute.clone(),
            };
            // Walk prefixes from shortest to longest
            let mut prefix = ComponentPath::new("");
            for seg in path.segments() {
                prefix = prefix.join(seg);
                if let Some(rule) = global_rules.get(&prefix) {
                    resolved.merge_from(rule);
                }
            }
            let (place, origin) = finalize_place(resolved);
            global.insert(path.clone(), place);
            provenance.global_origins.insert(path.clone(), origin);
        }

        // 5. Resolve every per-layer leaf component according to priority order:
        //    Priority 3: Root "model"
        //    Priority 2: General "hidden-layers" (from shallowest to deepest prefix)
        //    Priority 1: Layer selector matching layer `l` (from shallowest to deepest prefix)
        let mut layers = vec![BTreeMap::new(); tree.num_hidden_layers];
        let mut kv = Vec::new();
        let mut state = Vec::new();

        let kv_path = ComponentPath::new("attention/kv-cache");
        let state_path = ComponentPath::new("linear-attention/state");

        for layer_idx in 0..tree.num_hidden_layers {
            for (path, leaf) in &tree.layer_leaves {
                if !leaf.active_layers.contains(&layer_idx) {
                    continue;
                }
                let mut resolved = PartialPlace {
                    compute: Some(base_compute.clone()),
                    store: Some(base_store.clone()),
                    prefill_compute: root_place.prefill_compute.clone(),
                };

                // Priority 2: general hidden-layers rules
                if let Some(root_hl) = general_layer_rules.get(&ComponentPath::new("")) {
                    resolved.merge_from(root_hl);
                }
                let mut prefix = ComponentPath::new("");
                for seg in path.segments() {
                    prefix = prefix.join(seg);
                    if let Some(rule) = general_layer_rules.get(&prefix) {
                        resolved.merge_from(rule);
                    }
                }

                // Priority 1: layer selector rules (override general hidden-layers at any depth)
                if let Some(root_sel) = selector_layer_rules[layer_idx].get(&ComponentPath::new(""))
                {
                    resolved.merge_from(root_sel);
                }
                let mut prefix = ComponentPath::new("");
                for seg in path.segments() {
                    prefix = prefix.join(seg);
                    if let Some(rule) = selector_layer_rules[layer_idx].get(&prefix) {
                        resolved.merge_from(rule);
                    }
                }

                let (place, origin) = finalize_place(resolved);
                if *path == kv_path {
                    kv.push(KvPlacement {
                        layer: layer_idx,
                        place,
                        k_bits: self.kv_quantization.map(|q| q.k_bits),
                        v_bits: self.kv_quantization.map(|q| q.v_bits),
                    });
                } else if *path == state_path {
                    state.push(StatePlacement {
                        layer: layer_idx,
                        place,
                    });
                }
                layers[layer_idx].insert(path.clone(), place);
                provenance.layer_origins[layer_idx].insert(path.clone(), origin);
            }
        }

        Ok(PlacementPlan {
            tree,
            global,
            layers,
            kv,
            state,
            provenance,
        })
    }
}

fn finalize_place(resolved: PartialPlace) -> (Place, PlaceOrigin) {
    let (compute, compute_source) = resolved
        .compute
        .expect("root compute is guaranteed to be present");
    let (store, store_source) = resolved
        .store
        .expect("root store is guaranteed to be present");
    let (prefill_compute, prefill_source) = match resolved.prefill_compute {
        Some((dev, src)) => (Some(dev), Some(src)),
        None => (None, None),
    };
    (
        Place {
            compute,
            store,
            prefill_compute,
        },
        PlaceOrigin {
            compute_source,
            store_source,
            prefill_source,
        },
    )
}

fn extract_partial_place(obj: &Map<String, Value>, origin: &str) -> Result<PartialPlace> {
    let mut partial = PartialPlace::default();
    if let Some(c_val) = obj.get("compute") {
        let s = c_val.as_str().ok_or_else(|| {
            LumenError::Config(format!("'{origin}.compute' must be a string"))
        })?;
        partial.compute = Some((s.parse::<Device>()?, origin.to_string()));
    }
    if let Some(s_val) = obj.get("store") {
        let s = s_val
            .as_str()
            .ok_or_else(|| LumenError::Config(format!("'{origin}.store' must be a string")))?;
        partial.store = Some((s.parse::<MemLoc>()?, origin.to_string()));
    }
    if let Some(p_val) = obj.get("prefill-compute") {
        let s = p_val.as_str().ok_or_else(|| {
            LumenError::Config(format!("'{origin}.prefill-compute' must be a string"))
        })?;
        partial.prefill_compute = Some((s.parse::<Device>()?, origin.to_string()));
    }
    Ok(partial)
}

fn collect_subtree_rules(
    val: &Value,
    current_path: &ComponentPath,
    origin: &str,
    is_layer: bool,
    tree: &ComponentTree,
    out_rules: &mut BTreeMap<ComponentPath, PartialPlace>,
    skipped_rules: &mut Vec<String>,
) -> Result<()> {
    let obj = val
        .as_object()
        .ok_or_else(|| LumenError::Config(format!("'{origin}' must be an object")))?;

    let partial = extract_partial_place(obj, origin)?;
    let exists_in_tree = if is_layer {
        tree.layer_leaves.values().any(|leaf| {
            leaf.path.starts_with_path(current_path) && !leaf.active_layers.is_empty()
        })
    } else {
        tree.is_valid_global_path(current_path)
    };

    if partial.compute.is_some() || partial.store.is_some() || partial.prefill_compute.is_some() {
        if exists_in_tree {
            out_rules.insert(current_path.clone(), partial);
        } else {
            let msg = format!(
                "{origin} skipped (component '{}' absent in model '{}')",
                current_path, tree.model_name
            );
            if !skipped_rules.contains(&msg) {
                skipped_rules.push(msg);
            }
        }
    }

    // Valid child keys at `current_path`
    let known_prefixes = if is_layer {
        KNOWN_LAYER_PREFIXES
    } else {
        KNOWN_GLOBAL_PREFIXES
    };
    let prefix_slash = format!("{}/", current_path.as_str());
    let mut valid_children: BTreeSet<&str> = PLACEMENT_KEYS.iter().copied().collect();
    for &kp in known_prefixes {
        if let Some(rest) = kp.strip_prefix(&prefix_slash) {
            let child_seg = rest.split('/').next().unwrap_or(rest);
            valid_children.insert(child_seg);
        }
    }

    for (child_k, child_v) in obj {
        if PLACEMENT_KEYS.contains(&child_k.as_str()) {
            continue;
        }
        let child_path = current_path.join(child_k);
        let child_known = known_prefixes.contains(&child_path.as_str())
            || if is_layer {
                tree.is_valid_layer_path(&child_path)
            } else {
                tree.is_valid_global_path(&child_path)
            };
        if !child_known {
            let cand_vec: Vec<&str> = valid_children.iter().copied().collect();
            return Err(unknown_key_error(origin, child_k, &cand_vec));
        }
        collect_subtree_rules(
            child_v,
            &child_path,
            &format!("{origin}.{child_k}"),
            is_layer,
            tree,
            out_rules,
            skipped_rules,
        )?;
    }

    Ok(())
}

// ============================================================================
// Section Parsers
// ============================================================================

fn parse_sampling(val: &Value) -> Result<SamplingConfig> {
    let obj = val
        .as_object()
        .ok_or_else(|| LumenError::Config("'model.sampling' must be an object".to_string()))?;
    const VALID: &[&str] = &["temperature", "top-p", "top-k", "min-p"];
    for k in obj.keys() {
        if !VALID.contains(&k.as_str()) {
            return Err(unknown_key_error("model.sampling", k, VALID));
        }
    }
    let mut cfg = SamplingConfig::default();
    if let Some(v) = obj.get("temperature") {
        cfg.temperature = v.as_f64().ok_or_else(|| {
            LumenError::Config("'model.sampling.temperature' must be a number".to_string())
        })? as f32;
    }
    if let Some(v) = obj.get("top-p") {
        cfg.top_p = v.as_f64().ok_or_else(|| {
            LumenError::Config("'model.sampling.top-p' must be a number".to_string())
        })? as f32;
    }
    if let Some(v) = obj.get("top-k") {
        cfg.top_k = v.as_u64().ok_or_else(|| {
            LumenError::Config("'model.sampling.top-k' must be an integer".to_string())
        })? as u32;
    }
    if let Some(v) = obj.get("min-p") {
        cfg.min_p = v.as_f64().ok_or_else(|| {
            LumenError::Config("'model.sampling.min-p' must be a number".to_string())
        })? as f32;
    }
    Ok(cfg)
}

fn parse_chat_template(val: &Value) -> Result<ChatTemplateConfig> {
    let obj = val
        .as_object()
        .ok_or_else(|| LumenError::Config("'chat-template' must be an object".to_string()))?;
    const VALID: &[&str] = &["kwargs"];
    for k in obj.keys() {
        if !VALID.contains(&k.as_str()) {
            return Err(unknown_key_error("chat-template", k, VALID));
        }
    }
    let mut kwargs = BTreeMap::new();
    if let Some(kw_val) = obj.get("kwargs") {
        let kw_obj = kw_val.as_object().ok_or_else(|| {
            LumenError::Config("'chat-template.kwargs' must be an object".to_string())
        })?;
        for (k, v) in kw_obj {
            kwargs.insert(k.clone(), v.clone());
        }
    }
    Ok(ChatTemplateConfig { kwargs })
}

fn parse_context(val: &Value) -> Result<ContextConfig> {
    let obj = val
        .as_object()
        .ok_or_else(|| LumenError::Config("'context' must be an object".to_string()))?;
    const VALID: &[&str] = &["size"];
    for k in obj.keys() {
        if !VALID.contains(&k.as_str()) {
            return Err(unknown_key_error("context", k, VALID));
        }
    }
    let mut cfg = ContextConfig::default();
    if let Some(v) = obj.get("size") {
        cfg.size = v
            .as_u64()
            .ok_or_else(|| LumenError::Config("'context.size' must be a positive integer".to_string()))?
            as usize;
        if cfg.size == 0 {
            return Err(LumenError::Config(
                "'context.size' must be greater than 0".to_string(),
            ));
        }
    }
    Ok(cfg)
}

fn parse_kv_quantization(val: &Value) -> Result<KvQuantizationConfig> {
    let obj = val
        .as_object()
        .ok_or_else(|| LumenError::Config("'kv-quantization' must be an object".to_string()))?;
    const VALID: &[&str] = &["k-bits", "v-bits"];
    for k in obj.keys() {
        if !VALID.contains(&k.as_str()) {
            return Err(unknown_key_error("kv-quantization", k, VALID));
        }
    }
    let k_bits = obj
        .get("k-bits")
        .and_then(Value::as_u64)
        .ok_or_else(|| LumenError::Config("'kv-quantization.k-bits' is required".to_string()))?
        as u8;
    let v_bits = obj
        .get("v-bits")
        .and_then(Value::as_u64)
        .ok_or_else(|| LumenError::Config("'kv-quantization.v-bits' is required".to_string()))?
        as u8;
    if !(2..=8).contains(&k_bits) || !(2..=8).contains(&v_bits) {
        return Err(LumenError::Config(format!(
            "'kv-quantization' bits must be in 2..=8 (got k-bits={k_bits}, v-bits={v_bits})"
        )));
    }
    Ok(KvQuantizationConfig { k_bits, v_bits })
}

fn parse_cuda_config(val: &Value) -> Result<CudaRuntimeConfig> {
    let obj = val
        .as_object()
        .ok_or_else(|| LumenError::Config("'cuda' must be an object".to_string()))?;
    const VALID: &[&str] = &["prefill-chunk-size", "cuda-graphs"];
    for k in obj.keys() {
        if !VALID.contains(&k.as_str()) {
            return Err(unknown_key_error("cuda", k, VALID));
        }
    }
    let mut cfg = CudaRuntimeConfig::default();
    if let Some(v) = obj.get("prefill-chunk-size") {
        cfg.prefill_chunk_size = v.as_u64().ok_or_else(|| {
            LumenError::Config("'cuda.prefill-chunk-size' must be an integer".to_string())
        })? as usize;
    }
    if let Some(v) = obj.get("cuda-graphs") {
        cfg.cuda_graphs = v.as_bool().ok_or_else(|| {
            LumenError::Config("'cuda.cuda-graphs' must be a boolean".to_string())
        })?;
    }
    Ok(cfg)
}

fn parse_cpu_config(val: &Value) -> Result<CpuRuntimeConfig> {
    let obj = val
        .as_object()
        .ok_or_else(|| LumenError::Config("'cpu' must be an object".to_string()))?;
    const VALID: &[&str] = &["threads", "reserve-cores", "numa-spread"];
    for k in obj.keys() {
        if !VALID.contains(&k.as_str()) {
            return Err(unknown_key_error("cpu", k, VALID));
        }
    }
    let mut cfg = CpuRuntimeConfig::default();
    if let Some(v) = obj.get("threads") {
        cfg.threads = v
            .as_u64()
            .ok_or_else(|| LumenError::Config("'cpu.threads' must be an integer".to_string()))?
            as usize;
    }
    if let Some(v) = obj.get("reserve-cores") {
        cfg.reserve_cores = v.as_u64().ok_or_else(|| {
            LumenError::Config("'cpu.reserve-cores' must be an integer".to_string())
        })? as usize;
    }
    if let Some(v) = obj.get("numa-spread") {
        cfg.numa_spread = v
            .as_bool()
            .ok_or_else(|| LumenError::Config("'cpu.numa-spread' must be a boolean".to_string()))?;
    }
    Ok(cfg)
}

fn parse_disk_config(val: &Value) -> Result<DiskRuntimeConfig> {
    let obj = val
        .as_object()
        .ok_or_else(|| LumenError::Config("'disk' must be an object".to_string()))?;
    const VALID: &[&str] = &["io-threads", "mlock"];
    for k in obj.keys() {
        if !VALID.contains(&k.as_str()) {
            return Err(unknown_key_error("disk", k, VALID));
        }
    }
    let mut cfg = DiskRuntimeConfig::default();
    if let Some(v) = obj.get("io-threads") {
        cfg.io_threads = v.as_u64().ok_or_else(|| {
            LumenError::Config("'disk.io-threads' must be an integer".to_string())
        })? as usize;
    }
    if let Some(v) = obj.get("mlock") {
        cfg.mlock = v
            .as_bool()
            .ok_or_else(|| LumenError::Config("'disk.mlock' must be a boolean".to_string()))?;
    }
    Ok(cfg)
}

fn parse_server_config(val: &Value) -> Result<ServerRuntimeConfig> {
    let obj = val
        .as_object()
        .ok_or_else(|| LumenError::Config("'server' must be an object".to_string()))?;
    const VALID: &[&str] = &["host", "port", "max-concurrent-requests"];
    for k in obj.keys() {
        if !VALID.contains(&k.as_str()) {
            return Err(unknown_key_error("server", k, VALID));
        }
    }
    let mut cfg = ServerRuntimeConfig::default();
    if let Some(v) = obj.get("host") {
        cfg.host = v
            .as_str()
            .ok_or_else(|| LumenError::Config("'server.host' must be a string".to_string()))?
            .to_string();
    }
    if let Some(v) = obj.get("port") {
        cfg.port = v
            .as_u64()
            .ok_or_else(|| LumenError::Config("'server.port' must be an integer".to_string()))?
            as u16;
    }
    if let Some(v) = obj.get("max-concurrent-requests") {
        cfg.max_concurrent_requests = v.as_u64().ok_or_else(|| {
            LumenError::Config("'server.max-concurrent-requests' must be an integer".to_string())
        })? as usize;
    }
    Ok(cfg)
}

// ============================================================================
// Human-Readable Formatter for `lumen inspect`
// ============================================================================

impl PlacementPlan {
    /// Formats a complete, human-readable inspection report of the resolved `PlacementPlan`.
    pub fn format_inspect(&self, cfg: &RuntimeConfig) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        let _ = writeln!(out, "=== Lumen Placement Plan Inspection ===");
        let _ = writeln!(out, "Model path:          {}", cfg.model_path);
        let _ = writeln!(
            out,
            "Architecture tree:   {} ({} hidden layers)",
            self.tree.model_name, self.tree.num_hidden_layers
        );
        let _ = writeln!(out, "Context size:        {} tokens", cfg.context.size);
        match cfg.kv_quantization {
            Some(kvq) => {
                let _ = writeln!(
                    out,
                    "KV quantization:     K = {} bits, V = {} bits",
                    kvq.k_bits, kvq.v_bits
                );
            }
            None => {
                let _ = writeln!(out, "KV quantization:     BF16 (unquantized)");
            }
        }
        let _ = writeln!(
            out,
            "CUDA runtime:        prefill-chunk-size = {}, cuda-graphs = {}",
            cfg.cuda.prefill_chunk_size, cfg.cuda.cuda_graphs
        );
        let _ = writeln!(
            out,
            "CPU runtime:         threads = {} (reserve = {}, numa-spread = {})",
            cfg.cpu.threads, cfg.cpu.reserve_cores, cfg.cpu.numa_spread
        );
        let _ = writeln!(
            out,
            "Disk runtime:        io-threads = {}, mlock = {}",
            cfg.disk.io_threads, cfg.disk.mlock
        );

        // Global components table
        let _ = writeln!(out, "\n--- Global Components ---");
        let _ = writeln!(
            out,
            "{:<26} {:<8} {:<8} {:<10} {:<12} {:<12} Rule Source",
            "Component", "Store", "Compute", "Prefill", "Decode Mode", "Prefill Mode"
        );
        let _ = writeln!(out, "{}", "-".repeat(106));

        for (path, place) in &self.global {
            let leaf = &self.tree.global_leaves[path];
            let prefill_str = place
                .prefill_compute
                .map_or_else(|| "-".to_string(), |d| d.to_string());
            let origin_str = self.provenance.global_origins[path].summary();
            let _ = writeln!(
                out,
                "{:<26} {:<8} {:<8} {:<10} {:<12} {:<12} {}",
                path.as_str(),
                place.store.to_string(),
                place.compute.to_string(),
                prefill_str,
                place.decode_access_mode(leaf.kind).as_str(),
                place.prefill_access_mode(leaf.kind).as_str(),
                origin_str
            );
        }

        // Group hidden layers with identical placement maps and provenance
        let _ = writeln!(
            out,
            "\n--- Hidden Layers (0..{}) ---",
            self.tree.num_hidden_layers.saturating_sub(1)
        );
        let groups = self.group_identical_layers();

        for (layers_list, map_idx) in groups {
            let range_label = format_layer_index_list(&layers_list);
            let _ = writeln!(out, "\n[Layers {range_label}] ({} layers):", layers_list.len());
            let _ = writeln!(
                out,
                "  {:<36} {:<8} {:<8} {:<10} {:<12} {:<12} Rule Source",
                "Component", "Store", "Compute", "Prefill", "Decode Mode", "Prefill Mode"
            );
            let _ = writeln!(out, "  {}", "-".repeat(114));

            for (path, place) in &self.layers[map_idx] {
                let leaf = &self.tree.layer_leaves[path];
                let prefill_str = place
                    .prefill_compute
                    .map_or_else(|| "-".to_string(), |d| d.to_string());
                let origin_str = self.provenance.layer_origins[map_idx][path].summary();
                let _ = writeln!(
                    out,
                    "  {:<36} {:<8} {:<8} {:<10} {:<12} {:<12} {}",
                    path.as_str(),
                    place.store.to_string(),
                    place.compute.to_string(),
                    prefill_str,
                    place.decode_access_mode(leaf.kind).as_str(),
                    place.prefill_access_mode(leaf.kind).as_str(),
                    origin_str
                );
            }
        }

        if !self.provenance.skipped_rules.is_empty() {
            let _ = writeln!(out, "\n--- Skipped Rules ---");
            for s in &self.provenance.skipped_rules {
                let _ = writeln!(out, "  * {s}");
            }
        }

        if !self.provenance.warnings.is_empty() {
            let _ = writeln!(out, "\n--- Warnings ---");
            for w in &self.provenance.warnings {
                let _ = writeln!(out, "  ! {w}");
            }
        }

        out
    }

    fn group_identical_layers(&self) -> Vec<(Vec<usize>, usize)> {
        let mut groups: Vec<(Vec<usize>, usize)> = Vec::new();
        for l in 0..self.layers.len() {
            let found = groups.iter_mut().find(|(_, rep)| {
                self.layers[*rep] == self.layers[l]
                    && self.provenance.layer_origins[*rep] == self.provenance.layer_origins[l]
            });
            if let Some((list, _)) = found {
                list.push(l);
            } else {
                groups.push((vec![l], l));
            }
        }
        groups
    }
}

fn format_layer_index_list(indices: &[usize]) -> String {
    if indices.is_empty() {
        return String::new();
    }
    let mut parts = Vec::new();
    let mut start = indices[0];
    let mut prev = indices[0];

    for &idx in &indices[1..] {
        if idx == prev + 1 {
            prev = idx;
        } else {
            if start == prev {
                parts.push(format!("{start}"));
            } else {
                parts.push(format!("{start}-{prev}"));
            }
            start = idx;
            prev = idx;
        }
    }
    if start == prev {
        parts.push(format!("{start}"));
    } else {
        parts.push(format!("{start}-{prev}"));
    }
    parts.join(", ")
}

impl fmt::Display for Place {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.prefill_compute {
            Some(p) => write!(
                f,
                "store={}, compute={} (prefill={})",
                self.store, self.compute, p
            ),
            None => write!(f, "store={}, compute={}", self.store, self.compute),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_dense_config_resolution() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/example-dense.json");
        let cfg = RuntimeConfig::from_file(&path).expect("example-dense.json should parse");
        let tree = cfg
            .infer_tree_for_inspect(None, None)
            .expect("tree inference should succeed");
        assert_eq!(tree.num_hidden_layers, 64);

        let plan = cfg
            .resolve_plan(tree, true)
            .expect("plan resolution should succeed");

        // Input layer has store overridden to ram:0, compute inherited as cuda:0 -> ZeroCopy
        let input_place = plan.global[&ComponentPath::new("input-layer")];
        assert_eq!(input_place.store, MemLoc::Ram(0));
        assert_eq!(input_place.compute, Device::Cuda(0));
        assert_eq!(
            input_place.decode_access_mode(ComponentKind::SparseWeight),
            AccessMode::ZeroCopy
        );

        // Layers 0..35: dense-mlp/gate is on cpu:0 / ram:0 with prefill-compute = cuda:0
        let mlp_l0 = plan.layers[0][&ComponentPath::new("dense-mlp/gate")];
        assert_eq!(mlp_l0.compute, Device::Cpu(0));
        assert_eq!(mlp_l0.store, MemLoc::Ram(0));
        assert_eq!(mlp_l0.prefill_compute, Some(Device::Cuda(0)));
        assert_eq!(
            mlp_l0.decode_access_mode(ComponentKind::DenseWeight),
            AccessMode::Direct
        );
        assert_eq!(
            mlp_l0.prefill_access_mode(ComponentKind::DenseWeight),
            AccessMode::Streamed
        );

        // Layers 36..63: dense-mlp/gate overridden to cuda:0 / vram:0
        let mlp_l36 = plan.layers[36][&ComponentPath::new("dense-mlp/gate")];
        assert_eq!(mlp_l36.compute, Device::Cuda(0));
        assert_eq!(mlp_l36.store, MemLoc::Vram(0));
        assert_eq!(
            mlp_l36.decode_access_mode(ComponentKind::DenseWeight),
            AccessMode::Direct
        );
    }

    #[test]
    fn test_example_moe_config_resolution() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/example-moe.json");
        let cfg = RuntimeConfig::from_file(&path).expect("example-moe.json should parse");
        let tree = cfg
            .infer_tree_for_inspect(None, None)
            .expect("tree inference should succeed");
        assert_eq!(tree.num_hidden_layers, 48);

        let plan = cfg
            .resolve_plan(tree, true)
            .expect("plan resolution should succeed");

        // PLE table is stored on disk, computed on cuda:0 -> FetchH2d
        let ple_table = plan.global[&ComponentPath::new("ple/table")];
        assert_eq!(ple_table.store, MemLoc::Disk);
        assert_eq!(ple_table.compute, Device::Cuda(0));
        assert_eq!(
            ple_table.decode_access_mode(ComponentKind::SparseWeight),
            AccessMode::FetchH2d
        );

        // Routed experts on layer 0: cpu:0 / ram:0 (prefill: cuda:0)
        let exp_l0 = plan.layers[0][&ComponentPath::new("moe/routed-experts/projections")];
        assert_eq!(exp_l0.compute, Device::Cpu(0));
        assert_eq!(exp_l0.store, MemLoc::Ram(0));
        assert_eq!(exp_l0.prefill_compute, Some(Device::Cuda(0)));
        assert_eq!(
            exp_l0.decode_access_mode(ComponentKind::SparseWeight),
            AccessMode::Direct
        );
        assert_eq!(
            exp_l0.prefill_access_mode(ComponentKind::SparseWeight),
            AccessMode::Streamed
        );

        // Routed experts on layer 32: cuda:0 / vram:0
        let exp_l32 = plan.layers[32][&ComponentPath::new("moe/routed-experts/projections")];
        assert_eq!(exp_l32.compute, Device::Cuda(0));
        assert_eq!(exp_l32.store, MemLoc::Vram(0));
    }

    #[test]
    fn test_overlapping_layer_selectors_rejected() {
        let json = r#"{
            "model": {
                "path": "test.ltq",
                "compute": "cuda:0",
                "store": "vram:0",
                "hidden-layers": {
                    "0-15": { "moe": { "compute": "cpu:0" } },
                    "12-23": { "moe": { "compute": "cuda:0" } }
                }
            }
        }"#;
        let cfg = RuntimeConfig::from_json_str(json).unwrap();
        let tree = ComponentTree::preset("moe", 24);
        let err = cfg.resolve_plan(tree, true).unwrap_err();
        assert!(err.to_string().contains("overlapping layer selectors"));
    }

    #[test]
    fn test_unknown_key_suggestion() {
        let json = r#"{
            "model": {
                "path": "test.ltq",
                "compute": "cuda:0",
                "stor": "vram:0"
            }
        }"#;
        let cfg = RuntimeConfig::from_json_str(json).unwrap();
        let tree = ComponentTree::preset("dense", 16);
        let err = cfg.resolve_plan(tree, true).unwrap_err();
        assert!(
            err.to_string().contains("did you mean 'store'?"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_out_of_range_selector_warning_vs_strict() {
        let json = r#"{
            "model": {
                "path": "test.ltq",
                "compute": "cuda:0",
                "store": "vram:0",
                "hidden-layers": {
                    "10-20": { "dense-mlp": { "compute": "cpu:0" } }
                }
            }
        }"#;
        let cfg = RuntimeConfig::from_json_str(json).unwrap();
        let tree = ComponentTree::preset("dense", 16);

        // Non-strict produces a warning
        let plan = cfg.resolve_plan(tree.clone(), false).unwrap();
        assert_eq!(plan.provenance.warnings.len(), 1);

        // Strict returns an error
        assert!(cfg.resolve_plan(tree, true).is_err());
    }
}
