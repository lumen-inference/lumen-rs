//! Configuration parser, cascade resolver, layer-range selector grammar, and `PlacementPlan`
//! builder (`docs/RU/config.md` and `docs/RU/architecture.md` Section 3).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use lumen_core::{
    AccessMode, ComponentKind, ComponentPath, ComponentTree, Device, LumenError, MemLoc, Result,
};
use serde::de::DeserializeOwned;
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
            format!("compute: {}, store: {}", self.compute_source, self.store_source)
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
    pub global: BTreeMap<ComponentPath, Place>,
    pub layers: Vec<BTreeMap<ComponentPath, Place>>,
    pub kv: Vec<KvPlacement>,
    pub state: Vec<StatePlacement>,
    pub provenance: Provenance,
}

// ============================================================================
// Runtime & Backend Configuration Sections (Config Section 4)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
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
#[serde(default, rename_all = "kebab-case")]
pub struct ChatTemplateConfig {
    pub kwargs: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct ContextConfig {
    pub size: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self { size: 32768 }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct KvQuantizationConfig {
    pub k_bits: u8,
    pub v_bits: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
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
#[serde(default, rename_all = "kebab-case")]
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
#[serde(default, rename_all = "kebab-case")]
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
#[serde(default, rename_all = "kebab-case")]
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
    #[inline]
    pub fn looks_like_selector(key: &str) -> bool {
        key.as_bytes().first().is_some_and(u8::is_ascii_digit)
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
// Known Keys & Stack-Allocated Damerau-Levenshtein Typo Suggestion
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

/// Zero-allocation stack-based Damerau-Levenshtein distance for ASCII config keys (`len <= 63`).
fn edit_distance(a: &str, b: &str) -> usize {
    const MAX_LEN: usize = 63;
    let a = a.as_bytes();
    let b = b.as_bytes();
    let m = a.len().min(MAX_LEN);
    let n = b.len().min(MAX_LEN);

    let mut prev2 = [0u8; MAX_LEN + 1];
    let mut prev1 = [0u8; MAX_LEN + 1];
    let mut curr = [0u8; MAX_LEN + 1];

    for (j, slot) in prev1.iter_mut().enumerate().take(n + 1) {
        *slot = j as u8;
    }

    for i in 1..=m {
        curr[0] = i as u8;
        for j in 1..=n {
            let cost = u8::from(a[i - 1] != b[j - 1]);
            let mut best = (prev1[j] + 1).min(curr[j - 1] + 1).min(prev1[j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                best = best.min(prev2[j - 2] + 1);
            }
            curr[j] = best;
        }
        prev2 = prev1;
        prev1 = curr;
    }
    usize::from(prev1[n])
}

fn suggest_key<'a>(unknown: &str, candidates: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    candidates
        .into_iter()
        .map(|cand| (cand, edit_distance(unknown, cand)))
        .filter(|(_, dist)| *dist <= 3)
        .min_by_key(|(_, dist)| *dist)
        .map(|(cand, _)| cand)
}

fn unknown_key_error<'a>(
    scope: &str,
    key: &str,
    valid: impl IntoIterator<Item = &'a str>,
) -> LumenError {
    if let Some(suggestion) = suggest_key(key, valid) {
        LumenError::Config(format!(
            "unknown key '{key}' in {scope}; did you mean '{suggestion}'?"
        ))
    } else {
        LumenError::Config(format!("unknown key '{key}' in {scope}"))
    }
}

fn parse_section<T: DeserializeOwned>(
    val: Option<&Value>,
    scope: &str,
    valid_keys: &[&str],
) -> Result<Option<T>> {
    let Some(v) = val else { return Ok(None) };
    let obj = v
        .as_object()
        .ok_or_else(|| LumenError::Config(format!("'{scope}' must be an object")))?;
    for k in obj.keys() {
        if !valid_keys.contains(&k.as_str()) {
            return Err(unknown_key_error(scope, k, valid_keys.iter().copied()));
        }
    }
    serde_json::from_value(v.clone())
        .map(Some)
        .map_err(|e| LumenError::Config(format!("invalid '{scope}': {e}")))
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
                return Err(unknown_key_error(
                    "top-level config",
                    key,
                    TOP_LEVEL_KEYS.iter().copied(),
                ));
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

        let sampling = parse_section::<SamplingConfig>(
            model_obj.get("sampling"),
            "model.sampling",
            &["temperature", "top-p", "top-k", "min-p"],
        )?
        .unwrap_or_default();

        let chat_template = parse_section::<ChatTemplateConfig>(
            root.get("chat-template"),
            "chat-template",
            &["kwargs"],
        )?
        .unwrap_or_default();

        let context = parse_section::<ContextConfig>(root.get("context"), "context", &["size"])?
            .unwrap_or_default();
        if context.size == 0 {
            return Err(LumenError::Config(
                "'context.size' must be greater than 0".to_string(),
            ));
        }

        let kv_quantization = parse_section::<KvQuantizationConfig>(
            root.get("kv-quantization"),
            "kv-quantization",
            &["k-bits", "v-bits"],
        )?;
        if let Some(kvq) = kv_quantization
            && (!(2..=8).contains(&kvq.k_bits) || !(2..=8).contains(&kvq.v_bits))
        {
            return Err(LumenError::Config(format!(
                "'kv-quantization' bits must be in 2..=8 (got k-bits={}, v-bits={})",
                kvq.k_bits, kvq.v_bits
            )));
        }

        let cuda = parse_section::<CudaRuntimeConfig>(
            root.get("cuda"),
            "cuda",
            &["prefill-chunk-size", "cuda-graphs"],
        )?
        .unwrap_or_default();

        let cpu = parse_section::<CpuRuntimeConfig>(
            root.get("cpu"),
            "cpu",
            &["threads", "reserve-cores", "numa-spread"],
        )?
        .unwrap_or_default();

        let disk =
            parse_section::<DiskRuntimeConfig>(root.get("disk"), "disk", &["io-threads", "mlock"])?
                .unwrap_or_default();

        let server = parse_section::<ServerRuntimeConfig>(
            root.get("server"),
            "server",
            &["host", "port", "max-concurrent-requests"],
        )?
        .unwrap_or_default();

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

    /// Resolves the cascading configuration rules against the given `ComponentTree` into a
    /// validated, flat `PlacementPlan`.
    pub fn resolve_plan(&self, tree: ComponentTree, strict: bool) -> Result<PlacementPlan> {
        let superset = ComponentTree::canonical_superset(1);
        let mut provenance = Provenance {
            global_origins: BTreeMap::new(),
            layer_origins: vec![BTreeMap::new(); tree.num_hidden_layers],
            skipped_rules: Vec::new(),
            warnings: Vec::new(),
        };

        // 1. Validate keys in "model" and collect global rules
        let mut global_rules: BTreeMap<ComponentPath, PartialPlace> = BTreeMap::new();
        let valid_global_roots: Vec<&str> = superset
            .global_prefixes
            .iter()
            .map(ComponentPath::as_str)
            .filter(|p| !p.contains('/'))
            .chain(MODEL_META_KEYS.iter().copied())
            .collect();

        for (key, val) in &self.raw_model {
            if MODEL_META_KEYS.contains(&key.as_str()) {
                continue;
            }
            let path = ComponentPath::new(key);
            if !superset.is_valid_global_path(&path) && !tree.is_valid_global_path(&path) {
                return Err(unknown_key_error(
                    "model",
                    key,
                    valid_global_roots.iter().copied(),
                ));
            }
            collect_subtree_rules(
                val,
                &path,
                &format!("model.{key}"),
                false,
                &tree,
                &superset,
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

            let hl_self = extract_partial_place(hl_obj, "hidden-layers")?;
            if hl_self.compute.is_some()
                || hl_self.store.is_some()
                || hl_self.prefill_compute.is_some()
            {
                general_layer_rules.insert(ComponentPath::new(""), hl_self);
            }

            let valid_layer_roots: Vec<&str> = superset
                .layer_prefixes
                .iter()
                .map(ComponentPath::as_str)
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

                    let sel_origin = format!("hidden-layers[\"{}\"]", selector.raw);
                    let mut parsed_sel_rules: BTreeMap<ComponentPath, PartialPlace> =
                        BTreeMap::new();
                    collect_subtree_rules(
                        val,
                        &ComponentPath::new(""),
                        &sel_origin,
                        true,
                        &tree,
                        &superset,
                        &mut parsed_sel_rules,
                        &mut provenance.skipped_rules,
                    )?;

                    for layer_idx in in_range {
                        for (p, r) in &parsed_sel_rules {
                            let applies_on_layer = p.is_empty()
                                || tree.layer_leaves.values().any(|leaf| {
                                    leaf.path.starts_with_path(p)
                                        && leaf.active_layers.contains(layer_idx)
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
                    if !superset.is_valid_layer_path(&path) && !tree.is_valid_layer_path(&path) {
                        return Err(unknown_key_error(
                            "hidden-layers",
                            key,
                            valid_layer_roots.iter().copied(),
                        ));
                    }
                    collect_subtree_rules(
                        val,
                        &path,
                        &format!("hidden-layers.{key}"),
                        true,
                        &tree,
                        &superset,
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

        // 5. Resolve every per-layer leaf component
        let mut layers = vec![BTreeMap::new(); tree.num_hidden_layers];
        let mut kv = Vec::new();
        let mut state = Vec::new();

        let kv_path = ComponentPath::new("attention/kv-cache");
        let state_path = ComponentPath::new("linear-attention/state");
        let empty_path = ComponentPath::new("");

        for layer_idx in 0..tree.num_hidden_layers {
            for (path, leaf) in &tree.layer_leaves {
                if !leaf.active_layers.contains(layer_idx) {
                    continue;
                }
                let mut resolved = PartialPlace {
                    compute: Some(base_compute.clone()),
                    store: Some(base_store.clone()),
                    prefill_compute: root_place.prefill_compute.clone(),
                };

                for rules_map in [&general_layer_rules, &selector_layer_rules[layer_idx]] {
                    if let Some(root_rule) = rules_map.get(&empty_path) {
                        resolved.merge_from(root_rule);
                    }
                    let mut prefix = ComponentPath::new("");
                    for seg in path.segments() {
                        prefix = prefix.join(seg);
                        if let Some(rule) = rules_map.get(&prefix) {
                            resolved.merge_from(rule);
                        }
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
        let s = c_val
            .as_str()
            .ok_or_else(|| LumenError::Config(format!("'{origin}.compute' must be a string")))?;
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

#[allow(clippy::too_many_arguments)]
fn collect_subtree_rules(
    val: &Value,
    current_path: &ComponentPath,
    origin: &str,
    is_layer: bool,
    tree: &ComponentTree,
    superset: &ComponentTree,
    out_rules: &mut BTreeMap<ComponentPath, PartialPlace>,
    skipped_rules: &mut Vec<String>,
) -> Result<()> {
    let obj = val
        .as_object()
        .ok_or_else(|| LumenError::Config(format!("'{origin}' must be an object")))?;

    let partial = extract_partial_place(obj, origin)?;
    let exists_in_tree = current_path.is_empty()
        || if is_layer {
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

    let known_prefixes = if is_layer {
        &superset.layer_prefixes
    } else {
        &superset.global_prefixes
    };
    let mut valid_children: BTreeSet<&str> = PLACEMENT_KEYS.iter().copied().collect();
    for kp in known_prefixes {
        let rest_opt = if current_path.is_empty() {
            Some(kp.as_str())
        } else {
            kp.as_str()
                .strip_prefix(current_path.as_str())
                .and_then(|s| s.strip_prefix('/'))
        };
        if let Some(rest) = rest_opt {
            valid_children.insert(rest.split('/').next().unwrap_or(rest));
        }
    }

    for (child_k, child_v) in obj {
        if PLACEMENT_KEYS.contains(&child_k.as_str()) {
            continue;
        }
        let child_path = current_path.join(child_k);
        let child_known = known_prefixes.contains(&child_path)
            || if is_layer {
                tree.is_valid_layer_path(&child_path)
            } else {
                tree.is_valid_global_path(&child_path)
            };
        if !child_known {
            return Err(unknown_key_error(
                origin,
                child_k,
                valid_children.iter().copied(),
            ));
        }
        collect_subtree_rules(
            child_v,
            &child_path,
            &format!("{origin}.{child_k}"),
            is_layer,
            tree,
            superset,
            out_rules,
            skipped_rules,
        )?;
    }

    Ok(())
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
            let _ = writeln!(
                out,
                "{:<26} {:<8} {:<8} {:<10} {:<12} {:<12} {}",
                path.as_str(),
                place.store.to_string(),
                place.compute.to_string(),
                prefill_str,
                place.decode_access_mode(leaf.kind).as_str(),
                place.prefill_access_mode(leaf.kind).as_str(),
                self.provenance.global_origins[path].summary()
            );
        }

        let _ = writeln!(
            out,
            "\n--- Hidden Layers (0..{}) ---",
            self.tree.num_hidden_layers.saturating_sub(1)
        );
        for (layers_list, map_idx) in self.group_identical_layers() {
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
                let _ = writeln!(
                    out,
                    "  {:<36} {:<8} {:<8} {:<10} {:<12} {:<12} {}",
                    path.as_str(),
                    place.store.to_string(),
                    place.compute.to_string(),
                    prefill_str,
                    place.decode_access_mode(leaf.kind).as_str(),
                    place.prefill_access_mode(leaf.kind).as_str(),
                    self.provenance.layer_origins[map_idx][path].summary()
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
            parts.push(if start == prev {
                format!("{start}")
            } else {
                format!("{start}-{prev}")
            });
            start = idx;
            prev = idx;
        }
    }
    parts.push(if start == prev {
        format!("{start}")
    } else {
        format!("{start}-{prev}")
    });
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
        let tree = ComponentTree::preset("dense", 64);

        let plan = cfg
            .resolve_plan(tree, true)
            .expect("plan resolution should succeed");

        let input_place = plan.global[&ComponentPath::new("input-layer")];
        assert_eq!(input_place.store, MemLoc::Ram(0));
        assert_eq!(input_place.compute, Device::Cuda(0));
        assert_eq!(
            input_place.decode_access_mode(ComponentKind::SparseWeight),
            AccessMode::ZeroCopy
        );

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
        let tree = ComponentTree::preset("qwen4_exp", 48);

        let plan = cfg
            .resolve_plan(tree, true)
            .expect("plan resolution should succeed");

        let ple_table = plan.global[&ComponentPath::new("ple/table")];
        assert_eq!(ple_table.store, MemLoc::Disk);
        assert_eq!(ple_table.compute, Device::Cuda(0));
        assert_eq!(
            ple_table.decode_access_mode(ComponentKind::SparseWeight),
            AccessMode::FetchH2d
        );

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

        let exp_l32 = plan.layers[32][&ComponentPath::new("moe/routed-experts/projections")];
        assert_eq!(exp_l32.compute, Device::Cuda(0));
        assert_eq!(exp_l32.store, MemLoc::Vram(0));
    }

    #[test]
    fn test_absent_components_recorded_in_skipped_rules() {
        let json = r#"{
            "model": {
                "path": "Qwen3.8-27B.ltq",
                "compute": "cuda:0",
                "store": "vram:0",
                "ple": {
                    "table": { "store": "disk" }
                },
                "hidden-layers": {
                    "attention": {
                        "indexer": { "store": "ram:0" }
                    },
                    "hyper-connections": { "compute": "cpu:0" }
                }
            }
        }"#;
        let cfg = RuntimeConfig::from_json_str(json).unwrap();
        let tree = ComponentTree::preset("dense", 64);
        let plan = cfg.resolve_plan(tree, true).unwrap();
        assert_eq!(plan.provenance.skipped_rules.len(), 3);
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

        let plan = cfg.resolve_plan(tree.clone(), false).unwrap();
        assert_eq!(plan.provenance.warnings.len(), 1);
        assert!(cfg.resolve_plan(tree, true).is_err());
    }
}
