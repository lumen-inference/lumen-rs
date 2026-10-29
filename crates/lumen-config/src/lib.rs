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
        AccessMode::resolve(self.store, self.compute, kind.sparse_in_decode())
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
        AccessMode::resolve(
            self.store,
            self.effective_prefill_compute(),
            kind.sparse_in_prefill(),
        )
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

impl SamplingConfig {
    /// Keys accepted in `model.sampling` (kept next to the fields; a test checks they agree).
    pub const KEYS: &'static [&'static str] = &["temperature", "top-p", "top-k", "min-p"];
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

impl ChatTemplateConfig {
    pub const KEYS: &'static [&'static str] = &["kwargs"];
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct ContextConfig {
    pub size: usize,
}

impl ContextConfig {
    pub const KEYS: &'static [&'static str] = &["size"];
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

impl KvQuantizationConfig {
    pub const KEYS: &'static [&'static str] = &["k-bits", "v-bits"];
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct CudaRuntimeConfig {
    pub prefill_chunk_size: usize,
    pub cuda_graphs: bool,
}

impl CudaRuntimeConfig {
    pub const KEYS: &'static [&'static str] = &["prefill-chunk-size", "cuda-graphs"];
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

impl CpuRuntimeConfig {
    pub const KEYS: &'static [&'static str] = &["threads", "reserve-cores", "numa-spread"];
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

impl DiskRuntimeConfig {
    pub const KEYS: &'static [&'static str] = &["io-threads", "mlock"];
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

impl ServerRuntimeConfig {
    pub const KEYS: &'static [&'static str] = &["host", "port", "max-concurrent-requests"];
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

/// Attention implementation selected by `attention-mechanism`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AttentionMechanism {
    #[default]
    FlashAttention,
}

impl AttentionMechanism {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FlashAttention => "flash-attention",
        }
    }
}

impl std::str::FromStr for AttentionMechanism {
    type Err = LumenError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "flash-attention" => Ok(Self::FlashAttention),
            other => Err(LumenError::Config(format!(
                "unknown attention-mechanism '{other}' (supported: 'flash-attention')"
            ))),
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
    pub attention_mechanism: AttentionMechanism,
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

    /// Every layer index of the selector. Unbounded for huge ranges: prefer
    /// [`Self::layers_within`] or work on [`Self::ranges`] directly.
    pub fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.ranges.iter().flat_map(|&(s, e)| s..=e)
    }

    /// The layer indices of the selector that exist in a model with `num_layers` layers.
    pub fn layers_within(&self, num_layers: usize) -> impl Iterator<Item = usize> + '_ {
        self.ranges
            .iter()
            .filter(move |&&(s, _)| s < num_layers)
            .flat_map(move |&(s, e)| s..=e.min(num_layers - 1))
    }

    /// The parts of the selector at or beyond `num_layers`, as inclusive ranges.
    pub fn out_of_range(&self, num_layers: usize) -> Vec<(usize, usize)> {
        self.ranges
            .iter()
            .filter(|&&(_, e)| e >= num_layers)
            .map(|&(s, e)| (s.max(num_layers), e))
            .collect()
    }
}

/// Fails if any two layer ranges (of the same or of different selectors) share a layer.
fn check_selector_overlaps(selectors: &[(LayerSelector, &Value)]) -> Result<()> {
    let mut spans: Vec<(usize, usize, &str)> = selectors
        .iter()
        .flat_map(|(sel, _)| sel.ranges.iter().map(|&(s, e)| (s, e, sel.raw.as_str())))
        .collect();
    spans.sort_unstable();
    // Sorted by start: if some span overlaps a later one, it also overlaps its successor.
    for pair in spans.windows(2) {
        let (_, prev_end, prev_raw) = pair[0];
        let (start, _, raw) = pair[1];
        if start <= prev_end {
            return Err(LumenError::Config(format!(
                "overlapping layer selectors '{prev_raw}' and '{raw}' both match layer {start}"
            )));
        }
    }
    Ok(())
}

/// `[(0, 3), (7, 7)]` -> `"0-3, 7"`.
fn format_ranges(ranges: &[(usize, usize)]) -> String {
    ranges
        .iter()
        .map(|&(s, e)| {
            if s == e {
                s.to_string()
            } else {
                format!("{s}-{e}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
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

fn suggest_key<'a>(
    unknown: &str,
    candidates: impl IntoIterator<Item = &'a str>,
) -> Option<&'a str> {
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

/// Range checks for the runtime sections (the JSON shape was already validated by serde).
fn validate_runtime_sections(
    sampling: &SamplingConfig,
    cuda: &CudaRuntimeConfig,
    disk: &DiskRuntimeConfig,
    server: &ServerRuntimeConfig,
) -> Result<()> {
    let bad = |msg: String| Err(LumenError::Config(msg));
    if !sampling.temperature.is_finite() || sampling.temperature < 0.0 {
        return bad(format!(
            "'model.sampling.temperature' must be finite and >= 0 (got {})",
            sampling.temperature
        ));
    }
    if !(sampling.top_p > 0.0 && sampling.top_p <= 1.0) {
        return bad(format!(
            "'model.sampling.top-p' must be in (0, 1] (got {})",
            sampling.top_p
        ));
    }
    if !(0.0..=1.0).contains(&sampling.min_p) {
        return bad(format!(
            "'model.sampling.min-p' must be in [0, 1] (got {})",
            sampling.min_p
        ));
    }
    if cuda.prefill_chunk_size == 0 {
        return bad("'cuda.prefill-chunk-size' must be greater than 0".to_string());
    }
    if disk.io_threads == 0 {
        return bad("'disk.io-threads' must be greater than 0".to_string());
    }
    if server.max_concurrent_requests == 0 {
        return bad("'server.max-concurrent-requests' must be greater than 0".to_string());
    }
    if server.host.trim().is_empty() {
        return bad("'server.host' must not be empty".to_string());
    }
    Ok(())
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
            SamplingConfig::KEYS,
        )?
        .unwrap_or_default();

        let chat_template = parse_section::<ChatTemplateConfig>(
            root.get("chat-template"),
            "chat-template",
            ChatTemplateConfig::KEYS,
        )?
        .unwrap_or_default();

        let context =
            parse_section::<ContextConfig>(root.get("context"), "context", ContextConfig::KEYS)?
                .unwrap_or_default();
        if context.size == 0 {
            return Err(LumenError::Config(
                "'context.size' must be greater than 0".to_string(),
            ));
        }

        let kv_quantization = parse_section::<KvQuantizationConfig>(
            root.get("kv-quantization"),
            "kv-quantization",
            KvQuantizationConfig::KEYS,
        )?;
        if let Some(kvq) = kv_quantization
            && (!(2..=8).contains(&kvq.k_bits) || !(2..=8).contains(&kvq.v_bits))
        {
            return Err(LumenError::Config(format!(
                "'kv-quantization' bits must be in 2..=8 (got k-bits={}, v-bits={})",
                kvq.k_bits, kvq.v_bits
            )));
        }

        let cuda =
            parse_section::<CudaRuntimeConfig>(root.get("cuda"), "cuda", CudaRuntimeConfig::KEYS)?
                .unwrap_or_default();

        let cpu =
            parse_section::<CpuRuntimeConfig>(root.get("cpu"), "cpu", CpuRuntimeConfig::KEYS)?
                .unwrap_or_default();

        let disk =
            parse_section::<DiskRuntimeConfig>(root.get("disk"), "disk", DiskRuntimeConfig::KEYS)?
                .unwrap_or_default();

        let server = parse_section::<ServerRuntimeConfig>(
            root.get("server"),
            "server",
            ServerRuntimeConfig::KEYS,
        )?
        .unwrap_or_default();

        let attention_mechanism = match root.get("attention-mechanism") {
            Some(v) => v
                .as_str()
                .ok_or_else(|| {
                    LumenError::Config("'attention-mechanism' must be a string".to_string())
                })?
                .parse()?,
            None => AttentionMechanism::default(),
        };

        validate_runtime_sections(&sampling, &cuda, &disk, &server)?;

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
    ///
    /// `known` is a tree containing *every* component path any supported architecture can have
    /// (`lumen_models::known_components`). A rule for a path that exists in `known` but not in
    /// `tree` (e.g. `moe` on a dense model) is skipped and recorded in
    /// [`Provenance::skipped_rules`]; a path that is in neither is a typo and is rejected.
    /// This keeps `lumen-config` free of any knowledge about concrete models.
    pub fn resolve_plan(
        &self,
        tree: ComponentTree,
        known: &ComponentTree,
        strict: bool,
    ) -> Result<PlacementPlan> {
        let superset = known;
        let num_layers = tree.num_hidden_layers;
        let mut provenance = Provenance {
            global_origins: BTreeMap::new(),
            layer_origins: vec![BTreeMap::new(); num_layers],
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
                superset,
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
            vec![BTreeMap::new(); num_layers];

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

            // Parse every selector first: overlaps are decided on the ranges themselves, so the
            // cost never depends on how large the numbers in the config are.
            let mut selectors: Vec<(LayerSelector, &Value)> = Vec::new();
            for (key, val) in hl_obj {
                if PLACEMENT_KEYS.contains(&key.as_str()) {
                    continue;
                }
                if LayerSelector::looks_like_selector(key) {
                    selectors.push((LayerSelector::parse(key)?, val));
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
                        superset,
                        &mut general_layer_rules,
                        &mut provenance.skipped_rules,
                    )?;
                }
            }
            check_selector_overlaps(&selectors)?;

            for (selector, val) in &selectors {
                let out_of_range = selector.out_of_range(num_layers);
                if !out_of_range.is_empty() {
                    let msg = format!(
                        "layer selector '{}' references layers {} outside model range [0, {})",
                        selector.raw,
                        format_ranges(&out_of_range),
                        num_layers
                    );
                    if strict {
                        return Err(LumenError::Config(msg));
                    }
                    provenance.warnings.push(msg);
                }

                let sel_origin = format!("hidden-layers[\"{}\"]", selector.raw);
                let mut parsed_sel_rules: BTreeMap<ComponentPath, PartialPlace> = BTreeMap::new();
                collect_subtree_rules(
                    val,
                    &ComponentPath::new(""),
                    &sel_origin,
                    true,
                    &tree,
                    superset,
                    &mut parsed_sel_rules,
                    &mut provenance.skipped_rules,
                )?;

                for layer_idx in selector.layers_within(num_layers) {
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
                if leaf.kind == ComponentKind::KvCache {
                    kv.push(KvPlacement {
                        layer: layer_idx,
                        place,
                        k_bits: self.kv_quantization.map(|q| q.k_bits),
                        v_bits: self.kv_quantization.map(|q| q.v_bits),
                    });
                } else if leaf.kind == ComponentKind::RecurrentState {
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
            let _ = writeln!(
                out,
                "\n[Layers {range_label}] ({} layers):",
                layers_list.len()
            );
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
    use lumen_core::LayerMask;

    // ------------------------------------------------------------------------------------
    // Hand-built component trees. `lumen-config` must not know any model family, so the tests
    // describe the shapes they need directly (real architectures are tested in `lumen-models`).
    // ------------------------------------------------------------------------------------

    #[derive(Clone)]
    struct Shape {
        layers: usize,
        attention: LayerMask,
        linear_attention: LayerMask,
        dense_mlp: LayerMask,
        moe: LayerMask,
        ple: bool,
    }

    impl Shape {
        fn dense(layers: usize) -> Self {
            Self {
                layers,
                attention: LayerMask::all(layers),
                linear_attention: LayerMask::empty(),
                dense_mlp: LayerMask::all(layers),
                moe: LayerMask::empty(),
                ple: false,
            }
        }

        fn moe(layers: usize) -> Self {
            Self {
                layers,
                attention: LayerMask::from_fn(layers, |l| (l + 1) % 4 == 0),
                linear_attention: LayerMask::from_fn(layers, |l| (l + 1) % 4 != 0),
                dense_mlp: LayerMask::empty(),
                moe: LayerMask::all(layers),
                ple: true,
            }
        }

        fn tree(&self) -> ComponentTree {
            use ComponentKind::*;
            let mut t = ComponentTree::new("test-model", self.layers);
            t.add_global("input-layer", RowTable);
            t.add_global("output-layer/norm", DenseWeight);
            t.add_global("output-layer/lm-head", DenseWeight);
            if self.ple {
                t.add_global("ple/table", RowTable);
                t.add_global("ple/projections", DenseWeight);
            }
            t.add_layer("norms/input-norm", DenseWeight, LayerMask::all(self.layers));
            for p in ["q", "k", "v", "o"] {
                t.add_layer(
                    format!("attention/{p}"),
                    DenseWeight,
                    self.attention.clone(),
                );
            }
            t.add_layer("attention/kv-cache", KvCache, self.attention.clone());
            t.add_layer(
                "linear-attention/projections/qkv",
                DenseWeight,
                self.linear_attention.clone(),
            );
            t.add_layer(
                "linear-attention/state",
                RecurrentState,
                self.linear_attention.clone(),
            );
            for p in ["gate", "up", "down"] {
                t.add_layer(
                    format!("dense-mlp/{p}"),
                    DenseWeight,
                    self.dense_mlp.clone(),
                );
            }
            t.add_layer("moe/router", DenseWeight, self.moe.clone());
            t.add_layer(
                "moe/routed-experts/projections",
                ExpertBank,
                self.moe.clone(),
            );
            t
        }
    }

    /// Every path the "supported architectures" could have: what `lumen-models` provides.
    fn known() -> ComponentTree {
        let mut s = Shape::moe(1);
        s.dense_mlp = LayerMask::all(1);
        let mut t = s.tree();
        t.add_layer(
            "hyper-connections",
            ComponentKind::DenseWeight,
            LayerMask::empty(),
        );
        t.add_layer(
            "attention/indexer/keys",
            ComponentKind::IndexerCache,
            LayerMask::empty(),
        );
        t.add_global("vision-tower", ComponentKind::DenseWeight);
        t
    }

    fn cfg(model_body: &str) -> RuntimeConfig {
        RuntimeConfig::from_json_str(&format!(
            r#"{{"model": {{"path": "model.ltq", "compute": "cuda:0", "store": "vram:0", {model_body} }}}}"#
        ))
        .unwrap()
    }

    const MIN: &str = r#"{"model":{"path":"m","compute":"cuda:0","store":"vram:0"}}"#;

    // ------------------------------------------------------------------------------------
    // Cascade and access modes
    // ------------------------------------------------------------------------------------

    #[test]
    fn test_dense_cascade_and_access_modes() {
        let cfg = cfg(r#"
            "input-layer": { "store": "ram:0" },
            "hidden-layers": {
                "dense-mlp": { "compute": "cpu:0", "store": "ram:0", "prefill-compute": "cuda:0" },
                "36-63": { "dense-mlp": { "compute": "cuda:0", "store": "vram:0" } }
            }"#);
        let plan = cfg
            .resolve_plan(Shape::dense(64).tree(), &known(), true)
            .unwrap();

        let input = plan.global[&ComponentPath::new("input-layer")];
        assert_eq!(
            (input.store, input.compute),
            (MemLoc::Ram(0), Device::Cuda(0))
        );
        assert_eq!(
            input.decode_access_mode(ComponentKind::RowTable),
            AccessMode::ZeroCopy
        );

        let gate0 = plan.layers[0][&ComponentPath::new("dense-mlp/gate")];
        assert_eq!(gate0.compute, Device::Cpu(0));
        assert_eq!(gate0.store, MemLoc::Ram(0));
        assert_eq!(gate0.prefill_compute, Some(Device::Cuda(0)));
        assert_eq!(
            gate0.decode_access_mode(ComponentKind::DenseWeight),
            AccessMode::Direct
        );
        assert_eq!(
            gate0.prefill_access_mode(ComponentKind::DenseWeight),
            AccessMode::Streamed
        );

        // Layer selector beats the general hidden-layers rule.
        let gate36 = plan.layers[36][&ComponentPath::new("dense-mlp/gate")];
        assert_eq!(
            (gate36.compute, gate36.store),
            (Device::Cuda(0), MemLoc::Vram(0))
        );
        // Each key cascades independently: the selector did not mention `prefill-compute`, so the
        // value from the general `dense-mlp` rule is still in effect.
        assert_eq!(gate36.prefill_compute, Some(Device::Cuda(0)));
        assert_eq!(
            gate36.prefill_access_mode(ComponentKind::DenseWeight),
            AccessMode::Direct
        );

        // Components not mentioned inherit the model root.
        let q = plan.layers[10][&ComponentPath::new("attention/q")];
        assert_eq!((q.compute, q.store), (Device::Cuda(0), MemLoc::Vram(0)));
    }

    #[test]
    fn test_moe_cascade_and_access_modes() {
        let cfg = cfg(r#"
            "input-layer": { "store": "ram:0" },
            "ple": { "table": { "store": "disk" } },
            "hidden-layers": {
                "moe": { "routed-experts": {
                    "compute": "cpu:0", "store": "ram:0", "prefill-compute": "cuda:0" } },
                "32-47": { "moe": { "routed-experts": { "compute": "cuda:0", "store": "vram:0" } } }
            }"#);
        let plan = cfg
            .resolve_plan(Shape::moe(48).tree(), &known(), true)
            .unwrap();

        let table = plan.global[&ComponentPath::new("ple/table")];
        assert_eq!(table.store, MemLoc::Disk);
        assert_eq!(
            table.decode_access_mode(ComponentKind::RowTable),
            AccessMode::FetchH2d
        );

        let experts = ComponentPath::new("moe/routed-experts/projections");
        let e0 = plan.layers[0][&experts];
        assert_eq!((e0.compute, e0.store), (Device::Cpu(0), MemLoc::Ram(0)));
        assert_eq!(
            e0.decode_access_mode(ComponentKind::ExpertBank),
            AccessMode::Direct
        );
        // A prefill chunk touches (almost) every expert: the bank is streamed, not gathered.
        assert_eq!(
            e0.prefill_access_mode(ComponentKind::ExpertBank),
            AccessMode::Streamed
        );

        let e32 = plan.layers[32][&experts];
        assert_eq!((e32.compute, e32.store), (Device::Cuda(0), MemLoc::Vram(0)));
    }

    #[test]
    fn test_prefill_compute_does_not_turn_embedding_gather_into_streaming() {
        let cfg = cfg(r#""prefill-compute": "cuda:0", "input-layer": { "store": "ram:0" }"#);
        let plan = cfg
            .resolve_plan(Shape::dense(4).tree(), &known(), true)
            .unwrap();
        let p = plan.global[&ComponentPath::new("input-layer")];
        assert_eq!(
            p.decode_access_mode(ComponentKind::RowTable),
            AccessMode::ZeroCopy
        );
        assert_eq!(
            p.prefill_access_mode(ComponentKind::RowTable),
            AccessMode::ZeroCopy
        );
    }

    #[test]
    fn test_kv_and_state_placements_follow_component_kinds() {
        let shape = Shape::moe(8); // attention on layers 3 and 7, linear attention elsewhere
        let cfg = RuntimeConfig::from_json_str(
            r#"{"model": {"path": "m", "compute": "cuda:0", "store": "vram:0",
                "hidden-layers": { "attention": { "kv-cache": { "store": "ram:0" } } }},
                "kv-quantization": { "k-bits": 8, "v-bits": 5 }}"#,
        )
        .unwrap();
        let plan = cfg.resolve_plan(shape.tree(), &known(), true).unwrap();

        assert_eq!(
            plan.kv.iter().map(|k| k.layer).collect::<Vec<_>>(),
            vec![3, 7]
        );
        assert!(plan.kv.iter().all(|k| k.place.store == MemLoc::Ram(0)));
        assert!(
            plan.kv
                .iter()
                .all(|k| (k.k_bits, k.v_bits) == (Some(8), Some(5)))
        );
        assert_eq!(
            plan.state.iter().map(|s| s.layer).collect::<Vec<_>>(),
            vec![0, 1, 2, 4, 5, 6]
        );

        // A dense model has neither recurrent state nor (when attention is absent) a KV cache.
        let plan = cfg_min()
            .resolve_plan(Shape::dense(4).tree(), &known(), true)
            .unwrap();
        assert_eq!(plan.kv.len(), 4);
        assert!(plan.state.is_empty());
    }

    fn cfg_min() -> RuntimeConfig {
        RuntimeConfig::from_json_str(MIN).unwrap()
    }

    #[test]
    fn test_more_than_128_layers_resolve_completely() {
        let plan = cfg_min()
            .resolve_plan(Shape::dense(200).tree(), &known(), true)
            .unwrap();
        assert_eq!(plan.layers.len(), 200);
        let per_layer = plan.layers[0].len();
        assert!(per_layer > 0);
        assert!(plan.layers.iter().all(|l| l.len() == per_layer));
        assert_eq!(plan.kv.len(), 200);
    }

    // ------------------------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------------------------

    #[test]
    fn test_absent_components_recorded_in_skipped_rules() {
        let cfg = cfg(r#"
            "ple": { "table": { "store": "disk" } },
            "hidden-layers": {
                "attention": { "indexer": { "keys": { "store": "ram:0" } } },
                "hyper-connections": { "compute": "cpu:0" }
            }"#);
        let plan = cfg
            .resolve_plan(Shape::dense(64).tree(), &known(), true)
            .unwrap();
        assert_eq!(
            plan.provenance.skipped_rules.len(),
            3,
            "{:?}",
            plan.provenance.skipped_rules
        );
    }

    #[test]
    fn test_overlapping_layer_selectors_rejected() {
        let cfg = cfg(r#""hidden-layers": {
            "0-15": { "moe": { "compute": "cpu:0" } },
            "12-23": { "moe": { "compute": "cuda:0" } } }"#);
        let err = cfg
            .resolve_plan(Shape::moe(24).tree(), &known(), true)
            .unwrap_err();
        assert!(
            err.to_string().contains("overlapping layer selectors"),
            "{err}"
        );

        // Within a single selector, and adjacent-but-disjoint selectors are fine.
        let cfg = super::tests::cfg(r#""hidden-layers": { "0-5,3-8": {} }"#);
        assert!(
            cfg.resolve_plan(Shape::moe(24).tree(), &known(), true)
                .is_err()
        );
        let cfg = super::tests::cfg(r#""hidden-layers": { "0-7": {}, "8-15": {}, "20": {} }"#);
        assert!(
            cfg.resolve_plan(Shape::moe(24).tree(), &known(), true)
                .is_ok()
        );
    }

    #[test]
    fn test_unknown_key_suggestion() {
        let cfg = RuntimeConfig::from_json_str(
            r#"{"model": {"path": "t", "compute": "cuda:0", "stor": "vram:0"}}"#,
        )
        .unwrap();
        let err = cfg
            .resolve_plan(Shape::dense(16).tree(), &known(), true)
            .unwrap_err();
        assert!(err.to_string().contains("did you mean 'store'?"), "{err}");
    }

    #[test]
    fn test_out_of_range_selector_warning_vs_strict() {
        let cfg = cfg(r#""hidden-layers": { "10-20": { "dense-mlp": { "compute": "cpu:0" } } }"#);
        let tree = Shape::dense(16).tree();

        let plan = cfg.resolve_plan(tree.clone(), &known(), false).unwrap();
        assert_eq!(plan.provenance.warnings.len(), 1);
        assert!(
            plan.provenance.warnings[0].contains("16-20"),
            "{:?}",
            plan.provenance.warnings
        );
        // The in-range part is still applied.
        let gate = plan.layers[15][&ComponentPath::new("dense-mlp/gate")];
        assert_eq!(gate.compute, Device::Cpu(0));
        assert!(cfg.resolve_plan(tree, &known(), true).is_err());
    }

    #[test]
    fn test_huge_selector_ranges_are_handled_without_enumerating_layers() {
        let start = std::time::Instant::now();
        let cfg =
            cfg(r#""hidden-layers": { "0-4000000000": { "dense-mlp": { "compute": "cpu:0" } } }"#);
        let plan = cfg
            .resolve_plan(Shape::dense(16).tree(), &known(), false)
            .unwrap();
        assert_eq!(plan.provenance.warnings.len(), 1);
        assert!(plan.provenance.warnings[0].contains("16-4000000000"));
        assert!(
            cfg.resolve_plan(Shape::dense(16).tree(), &known(), true)
                .is_err()
        );

        let cfg = super::tests::cfg(r#""hidden-layers": { "0-4000000000": {}, "5-6": {} }"#);
        assert!(
            cfg.resolve_plan(Shape::dense(16).tree(), &known(), false)
                .is_err()
        );
        assert!(
            start.elapsed().as_secs() < 5,
            "selector handling must not enumerate layers"
        );
    }

    #[test]
    fn test_layer_selector_parsing() {
        let s = LayerSelector::parse("0-3, 7").unwrap();
        assert_eq!(s.ranges, vec![(0, 3), (7, 7)]);
        assert_eq!(s.layers_within(5).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        assert_eq!(s.out_of_range(5), vec![(7, 7)]);
        assert_eq!(s.out_of_range(100), vec![]);
        assert!(LayerSelector::parse("5-3").is_err());
        assert!(LayerSelector::parse("1,,2").is_err());
        assert!(LayerSelector::parse("a-b").is_err());
        assert_eq!(
            LayerSelector::parse("2").unwrap().layers_within(0).count(),
            0
        );
    }

    #[test]
    fn test_runtime_section_validation() {
        let bad = [
            r#""model": {"path":"m","compute":"cuda:0","store":"vram:0","sampling":{"temperature":-1}}"#,
            r#""model": {"path":"m","compute":"cuda:0","store":"vram:0","sampling":{"top-p":0}}"#,
            r#""model": {"path":"m","compute":"cuda:0","store":"vram:0","sampling":{"top-p":1.5}}"#,
            r#""model": {"path":"m","compute":"cuda:0","store":"vram:0","sampling":{"min-p":2}}"#,
            r#""model": {"path":"m","compute":"cuda:0","store":"vram:0"}, "cuda": {"prefill-chunk-size": 0}"#,
            r#""model": {"path":"m","compute":"cuda:0","store":"vram:0"}, "disk": {"io-threads": 0}"#,
            r#""model": {"path":"m","compute":"cuda:0","store":"vram:0"}, "server": {"max-concurrent-requests": 0}"#,
            r#""model": {"path":"m","compute":"cuda:0","store":"vram:0"}, "server": {"host": " "}"#,
            r#""model": {"path":"m","compute":"cuda:0","store":"vram:0"}, "context": {"size": 0}"#,
            r#""model": {"path":"m","compute":"cuda:0","store":"vram:0"}, "kv-quantization": {"k-bits": 1, "v-bits": 8}"#,
            r#""model": {"path":"m","compute":"cuda:0","store":"vram:0"}, "attention-mechanism": "paged-magic""#,
        ];
        for body in bad {
            let json = format!("{{{body}}}");
            assert!(
                RuntimeConfig::from_json_str(&json).is_err(),
                "must be rejected: {json}"
            );
        }
        assert!(RuntimeConfig::from_json_str(MIN).is_ok());
    }

    #[test]
    fn test_known_keys_match_struct_fields() {
        fn keys<T: Serialize>(v: &T) -> BTreeSet<String> {
            serde_json::to_value(v)
                .unwrap()
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect()
        }
        fn expected(k: &[&str]) -> BTreeSet<String> {
            k.iter().map(|s| (*s).to_string()).collect()
        }
        assert_eq!(
            keys(&SamplingConfig::default()),
            expected(SamplingConfig::KEYS)
        );
        assert_eq!(
            keys(&ChatTemplateConfig::default()),
            expected(ChatTemplateConfig::KEYS)
        );
        assert_eq!(
            keys(&ContextConfig::default()),
            expected(ContextConfig::KEYS)
        );
        assert_eq!(
            keys(&KvQuantizationConfig {
                k_bits: 8,
                v_bits: 8
            }),
            expected(KvQuantizationConfig::KEYS)
        );
        assert_eq!(
            keys(&CudaRuntimeConfig::default()),
            expected(CudaRuntimeConfig::KEYS)
        );
        assert_eq!(
            keys(&CpuRuntimeConfig::default()),
            expected(CpuRuntimeConfig::KEYS)
        );
        assert_eq!(
            keys(&DiskRuntimeConfig::default()),
            expected(DiskRuntimeConfig::KEYS)
        );
        assert_eq!(
            keys(&ServerRuntimeConfig::default()),
            expected(ServerRuntimeConfig::KEYS)
        );
    }

    #[test]
    fn test_defaults_and_attention_mechanism() {
        let cfg = cfg_min();
        assert_eq!(cfg.attention_mechanism, AttentionMechanism::FlashAttention);
        assert_eq!(cfg.attention_mechanism.as_str(), "flash-attention");
        assert_eq!(cfg.context.size, 32768);
        assert!(cfg.cuda.cuda_graphs);
    }

    #[test]
    fn test_format_inspect_groups_identical_layers() {
        let cfg = cfg(r#""hidden-layers": { "2-3": { "dense-mlp": { "compute": "cpu:0" } } }"#);
        let plan = cfg
            .resolve_plan(Shape::dense(6).tree(), &known(), true)
            .unwrap();
        let text = plan.format_inspect(&cfg);
        assert!(text.contains("[Layers 0-1, 4-5] (4 layers)"), "{text}");
        assert!(text.contains("[Layers 2-3] (2 layers)"), "{text}");
    }
}
