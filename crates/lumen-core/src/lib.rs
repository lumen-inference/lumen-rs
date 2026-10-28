//! Foundational types for `lumen-rs`: data types (`DType`), tensor shapes (`Shape`),
//! compute devices (`Device`), storage locations (`MemLoc`), data access modes (`AccessMode`),
//! model component paths (`ComponentPath`, `ComponentTree`), and error types (`LumenError`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

pub use half::{bf16, f16};

/// Standard result alias used across `lumen-rs`.
pub type Result<T> = std::result::Result<T, LumenError>;

/// Unified error type for `lumen-rs` crates.
#[derive(Debug, Error)]
pub enum LumenError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("format error: {0}")]
    Format(String),

    #[error("shape mismatch: {0}")]
    Shape(String),

    #[error("memory error: {0}")]
    Memory(String),

    #[error("unsupported operation on {device}: {op}")]
    UnsupportedOp { device: Device, op: String },

    #[error("device error: {0}")]
    Device(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// ============================================================================
// DType
// ============================================================================

/// Scalar element data type for tensors and container descriptors.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
}

impl DType {
    /// Size of a single scalar element in bytes.
    #[inline]
    pub const fn element_size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::I64 | Self::U64 => 8,
        }
    }

    /// Canonical lowercase name used in `.ltq` and `safetensors` headers.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::U64 => "u64",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for DType {
    type Err = LumenError;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" | "fp32" => Ok(Self::F32),
            "f16" | "float16" | "fp16" => Ok(Self::F16),
            "bf16" | "bfloat16" => Ok(Self::BF16),
            "i8" | "int8" => Ok(Self::I8),
            "u8" | "uint8" => Ok(Self::U8),
            "i16" | "int16" => Ok(Self::I16),
            "u16" | "uint16" => Ok(Self::U16),
            "i32" | "int32" => Ok(Self::I32),
            "u32" | "uint32" => Ok(Self::U32),
            "i64" | "int64" => Ok(Self::I64),
            "u64" | "uint64" => Ok(Self::U64),
            other => Err(LumenError::Format(format!("unknown dtype '{other}'"))),
        }
    }
}

// ============================================================================
// Shape
// ============================================================================

/// Multi-dimensional tensor shape.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Shape(Vec<usize>);

impl Shape {
    #[inline]
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements across all dimensions.
    #[inline]
    pub fn numel(&self) -> usize {
        self.0.iter().copied().product()
    }

    /// Total size in bytes for the given `DType`.
    #[inline]
    pub fn byte_len(&self, dtype: DType) -> usize {
        self.numel().saturating_mul(dtype.element_size())
    }

    /// Contiguous row-major strides in elements.
    pub fn contiguous_strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.0.len()];
        for i in (0..self.0.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.0[i + 1];
        }
        strides
    }
}

impl From<&[usize]> for Shape {
    fn from(s: &[usize]) -> Self {
        Self(s.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(arr: [usize; N]) -> Self {
        Self(arr.to_vec())
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

// ============================================================================
// Device & MemLoc & AccessMode
// ============================================================================

/// Compute device executing operators (`"cpu:0"`, `"cuda:0"`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Device {
    /// CPU compute node (NUMA index, e.g., `cpu:0`).
    Cpu(u16),
    /// CUDA GPU device index (e.g., `cuda:0`).
    Cuda(u16),
}

impl Device {
    #[inline]
    pub const fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu(_))
    }

    #[inline]
    pub const fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    #[inline]
    pub const fn index(self) -> u16 {
        match self {
            Self::Cpu(i) | Self::Cuda(i) => i,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu(idx) => write!(f, "cpu:{idx}"),
            Self::Cuda(idx) => write!(f, "cuda:{idx}"),
        }
    }
}

impl FromStr for Device {
    type Err = LumenError;

    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim();
        if s == "cpu" {
            return Ok(Self::Cpu(0));
        }
        if s == "cuda" {
            return Ok(Self::Cuda(0));
        }
        if let Some((kind, idx_str)) = s.split_once(':') {
            let idx = idx_str.parse::<u16>().map_err(|_| {
                LumenError::Config(format!("invalid device index in '{s}' (expected u16)"))
            })?;
            match kind {
                "cpu" => Ok(Self::Cpu(idx)),
                "cuda" => Ok(Self::Cuda(idx)),
                _ => Err(LumenError::Config(format!(
                    "unknown compute device '{s}' (expected 'cpu:N' or 'cuda:G')"
                ))),
            }
        } else {
            Err(LumenError::Config(format!(
                "invalid compute device '{s}' (expected 'cpu:N' or 'cuda:G')"
            )))
        }
    }
}

impl Serialize for Device {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for Device {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Persistent memory location where weights, KV pages, or recurrent states are stored.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum MemLoc {
    /// System RAM on a NUMA node (e.g., `ram:0`).
    Ram(u16),
    /// Device VRAM on a GPU (e.g., `vram:0`).
    Vram(u16),
    /// Direct file mapping on NVMe storage (`disk`).
    Disk,
}

impl MemLoc {
    #[inline]
    pub const fn is_ram(self) -> bool {
        matches!(self, Self::Ram(_))
    }

    #[inline]
    pub const fn is_vram(self) -> bool {
        matches!(self, Self::Vram(_))
    }

    #[inline]
    pub const fn is_disk(self) -> bool {
        matches!(self, Self::Disk)
    }
}

impl fmt::Display for MemLoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ram(idx) => write!(f, "ram:{idx}"),
            Self::Vram(idx) => write!(f, "vram:{idx}"),
            Self::Disk => f.write_str("disk"),
        }
    }
}

impl FromStr for MemLoc {
    type Err = LumenError;

    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim();
        match s {
            "disk" => return Ok(Self::Disk),
            "ram" => return Ok(Self::Ram(0)),
            "vram" => return Ok(Self::Vram(0)),
            _ => {}
        }
        if let Some((kind, idx_str)) = s.split_once(':') {
            let idx = idx_str.parse::<u16>().map_err(|_| {
                LumenError::Config(format!("invalid memory location index in '{s}' (expected u16)"))
            })?;
            match kind {
                "ram" => Ok(Self::Ram(idx)),
                "vram" => Ok(Self::Vram(idx)),
                _ => Err(LumenError::Config(format!(
                    "unknown memory location '{s}' (expected 'ram:N', 'vram:G', or 'disk')"
                ))),
            }
        } else {
            Err(LumenError::Config(format!(
                "invalid memory location '{s}' (expected 'ram:N', 'vram:G', or 'disk')"
            )))
        }
    }
}

impl Serialize for MemLoc {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for MemLoc {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Data access mode determined at plan compilation from `(store, compute, sparse_access)`.
///
/// Corresponds to the 3x4 access matrix in `docs/RU/architecture.md` (Section 3).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum AccessMode {
    /// Local memory access (`vram:G` -> `cuda:G` or `ram:N` -> `cpu:N`).
    Direct,
    /// Pinned RAM accessed directly over PCIe by a GPU kernel for sparse row/page lookups.
    ZeroCopy,
    /// Pinned RAM streamed asynchronously (H2D double-buffered ring) into VRAM ahead of compute.
    Streamed,
    /// NVMe disk rows pre-fetched by I/O threads into a double-buffered pinned RAM staging buffer.
    Fetch,
    /// NVMe disk rows pre-fetched into pinned RAM and then copied H2D to GPU VRAM.
    FetchH2d,
    /// VRAM data copied back (D2H) to pinned RAM for CPU execution.
    Readback, PeerCopy,
}

impl AccessMode {
    /// Resolves the access mode from storage location, compute device, and whether the operator
    /// reads only a sparse subset of rows/blocks (`true`) or the full tensor (`false`).
    pub const fn resolve(store: MemLoc, compute: Device, sparse_access: bool) -> Self {
        match (store, compute) {
            (MemLoc::Vram(g1), Device::Cuda(g2)) if g1 == g2 => Self::Direct,
            (MemLoc::Vram(_), Device::Cuda(_)) => Self::PeerCopy,
            (MemLoc::Vram(_), Device::Cpu(_)) => Self::Readback,
            (MemLoc::Ram(_), Device::Cpu(_)) => Self::Direct,
            (MemLoc::Ram(_), Device::Cuda(_)) => {
                if sparse_access {
                    Self::ZeroCopy
                } else {
                    Self::Streamed
                }
            }
            (MemLoc::Disk, Device::Cpu(_)) => Self::Fetch,
            (MemLoc::Disk, Device::Cuda(_)) => Self::FetchH2d,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "Direct",
            Self::ZeroCopy => "ZeroCopy",
            Self::Streamed => "Streamed",
            Self::Fetch => "Fetch",
            Self::FetchH2d => "Fetch+H2D",
            Self::Readback => "Readback",
            Self::PeerCopy => "PeerCopy",
        }
    }
}

impl fmt::Display for AccessMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// ComponentPath & ComponentTree
// ============================================================================

/// Slash-separated path identifying a component in the model hierarchy
/// (e.g., `"input-layer"`, `"ple/table"`, `"moe/routed-experts"`).
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ComponentPath(String);

impl ComponentPath {
    pub fn new(path: impl Into<String>) -> Self {
        let s = path.into();
        let trimmed = s.trim_matches('/').to_string();
        Self(trimmed)
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn segments(&self) -> impl Iterator<Item = &str> {
        self.0.split('/').filter(|s| !s.is_empty())
    }

    #[must_use]
    pub fn join(&self, child: &str) -> Self {
        let child = child.trim_matches('/');
        if self.0.is_empty() {
            Self(child.to_string())
        } else if child.is_empty() {
            self.clone()
        } else {
            Self(format!("{}/{child}", self.0))
        }
    }

    /// Returns `true` if `self` is equal to `prefix` or is a descendant path under `prefix/`.
    pub fn starts_with_path(&self, prefix: &ComponentPath) -> bool {
        if prefix.0.is_empty() {
            return true;
        }
        if self.0 == prefix.0 {
            return true;
        }
        self.0
            .strip_prefix(&prefix.0)
            .is_some_and(|rest| rest.starts_with('/'))
    }
}

impl fmt::Display for ComponentPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for ComponentPath {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

/// Classification of a leaf component in a `ComponentTree`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ComponentKind {
    /// Dense matrix or weight tensor accessed in full during forward pass.
    DenseWeight,
    /// Row-indexed lookup table (`RowTable`) or routed expert bank (`ExpertBank`) accessed sparsely.
    SparseWeight,
    /// Paged attention KV cache (`KvPool`).
    KvCache,
    /// Recurrent linear-attention or PLE state (`StatePool`).
    RecurrentState,
    /// Pure compute operator node without persistent weights (e.g., `rope`).
    ComputeOnly,
}

impl ComponentKind {
    /// Whether accesses to this component are sparse (selecting rows/experts/pages).
    #[inline]
    pub const fn is_sparse(self) -> bool {
        matches!(self, Self::SparseWeight)
    }
}

/// Metadata for a single leaf component in `ComponentTree`.
#[derive(Clone, Debug)]
pub struct ComponentLeaf {
    pub path: ComponentPath,
    pub kind: ComponentKind,
    /// For per-layer components: the set of layer indices `[0, num_layers)` where this
    /// component actually exists in the loaded model. Empty for global components.
    pub active_layers: BTreeSet<usize>,
}

/// Model component hierarchy produced by `Architecture::components()`.
///
/// Used by `lumen-config` to validate JSON paths, resolve cascading rules into a flat
/// `PlacementPlan`, and power `lumen inspect` without hardcoding model-specific logic in
/// the config parser.
#[derive(Clone, Debug)]
pub struct ComponentTree {
    pub model_name: String,
    pub num_hidden_layers: usize,
    /// Global (non-layer) leaf components (e.g. `input-layer`, `ple/table`, `output-layer/lm-head`).
    pub global_leaves: BTreeMap<ComponentPath, ComponentLeaf>,
    /// Per-layer leaf components relative to `hidden-layers/` (e.g. `attention/q`, `moe/routed-experts`).
    pub layer_leaves: BTreeMap<ComponentPath, ComponentLeaf>,
    /// All valid intermediate directory prefixes for global nodes.
    pub global_prefixes: BTreeSet<ComponentPath>,
    /// All valid intermediate directory prefixes under `hidden-layers/`.
    pub layer_prefixes: BTreeSet<ComponentPath>,
}

impl ComponentTree {
    pub fn new(model_name: impl Into<String>, num_hidden_layers: usize) -> Self {
        Self {
            model_name: model_name.into(),
            num_hidden_layers,
            global_leaves: BTreeMap::new(),
            layer_leaves: BTreeMap::new(),
            global_prefixes: BTreeSet::new(),
            layer_prefixes: BTreeSet::new(),
        }
    }

    /// Registers a global (non-layer) leaf component and all its parent directory prefixes.
    pub fn add_global(&mut self, path: impl Into<ComponentPath>, kind: ComponentKind) {
        let path = path.into();
        self.register_prefixes(&path, false);
        self.global_leaves.insert(
            path.clone(),
            ComponentLeaf {
                path,
                kind,
                active_layers: BTreeSet::new(),
            },
        );
    }

    /// Registers a per-layer leaf component active on the given `layers` and all its parent prefixes.
    pub fn add_layer(
        &mut self,
        path: impl Into<ComponentPath>,
        kind: ComponentKind,
        layers: impl IntoIterator<Item = usize>,
    ) {
        let path = path.into();
        self.register_prefixes(&path, true);
        let active_layers: BTreeSet<usize> = layers.into_iter().collect();
        self.layer_leaves
            .entry(path.clone())
            .and_modify(|leaf| leaf.active_layers.extend(&active_layers))
            .or_insert(ComponentLeaf {
                path,
                kind,
                active_layers,
            });
    }

    fn register_prefixes(&mut self, path: &ComponentPath, is_layer: bool) {
        let mut accum = ComponentPath::new("");
        for seg in path.segments() {
            accum = accum.join(seg);
            if is_layer {
                self.layer_prefixes.insert(accum.clone());
            } else {
                self.global_prefixes.insert(accum.clone());
            }
        }
    }

    /// Checks whether `path` is a valid global node (either a leaf or a group prefix).
    pub fn is_valid_global_path(&self, path: &ComponentPath) -> bool {
        self.global_prefixes.contains(path)
    }

    /// Checks whether `path` is a valid node under `hidden-layers` (either a leaf or a group prefix).
    pub fn is_valid_layer_path(&self, path: &ComponentPath) -> bool {
        self.layer_prefixes.contains(path)
    }

    /// Canonical superset tree containing all standard architectural modules from `docs/RU/config.md`.
    ///
    /// Used when inspecting a standalone config file before loading a concrete `.ltq` model,
    /// or as a base template for preset architectures (`dense`, `moe`, `qwen3_next`, `qwen4_exp`, `gemma4`).
    pub fn canonical_superset(num_hidden_layers: usize) -> Self {
        Self::preset("canonical-superset", num_hidden_layers)
    }

    /// Builds a component tree for a named architecture preset (`"dense"`, `"moe"`, `"qwen3_next"`,
    /// `"qwen4_exp"`, `"gemma4"`, or `"canonical-superset"`).
    pub fn preset(preset: &str, num_hidden_layers: usize) -> Self {
        let all_layers: Vec<usize> = (0..num_hidden_layers).collect();
        let mut tree = Self::new(preset, num_hidden_layers);

        // Global components
        tree.add_global("input-layer", ComponentKind::SparseWeight);
        tree.add_global("output-layer/norm", ComponentKind::DenseWeight);
        tree.add_global("output-layer/lm-head", ComponentKind::DenseWeight);

        let is_moe = matches!(preset, "moe" | "qwen3_next" | "qwen4_exp" | "canonical-superset");
        let has_ple = matches!(preset, "moe" | "qwen4_exp" | "gemma4" | "canonical-superset");
        let has_vision = matches!(preset, "gemma4" | "canonical-superset");
        let has_mtp = matches!(preset, "qwen3_next" | "qwen4_exp" | "canonical-superset");

        if has_ple {
            tree.add_global("ple/table", ComponentKind::SparseWeight);
            tree.add_global("ple/projections", ComponentKind::DenseWeight);
        }
        if has_vision {
            tree.add_global("vision-tower", ComponentKind::DenseWeight);
            tree.add_global("audio-tower", ComponentKind::DenseWeight);
        }
        if has_mtp {
            tree.add_global("mtp", ComponentKind::DenseWeight);
        }

        // Determine which layers have full/sparse attention vs linear attention (GDN)
        let (attn_layers, gdn_layers): (Vec<usize>, Vec<usize>) = match preset {
            "qwen3_next" | "qwen4_exp" => {
                // 3:1 GDN : Attention hybrid ratio (every 4th layer is Attention: 3, 7, 11, ...)
                let attn = (0..num_hidden_layers).filter(|l| (l + 1) % 4 == 0).collect();
                let gdn = (0..num_hidden_layers).filter(|l| (l + 1) % 4 != 0).collect();
                (attn, gdn)
            }
            "gemma4" => (all_layers.clone(), Vec::new()),
            _ => (all_layers.clone(), all_layers.clone()),
        };

        // Layer norms
        tree.add_layer("norms/input-norm", ComponentKind::DenseWeight, all_layers.clone());
        tree.add_layer(
            "norms/post-attention-norm",
            ComponentKind::DenseWeight,
            all_layers.clone(),
        );

        // Attention module
        for sub in ["attention/q", "attention/k", "attention/v", "attention/o", "attention/qk-norm"] {
            tree.add_layer(sub, ComponentKind::DenseWeight, attn_layers.clone());
        }
        tree.add_layer("attention/rope", ComponentKind::ComputeOnly, attn_layers.clone());
        tree.add_layer("attention/kv-cache", ComponentKind::KvCache, attn_layers.clone());
        tree.add_layer(
            "attention/indexer/keys",
            ComponentKind::SparseWeight,
            if matches!(preset, "moe" | "qwen4_exp" | "canonical-superset") {
                attn_layers.clone()
            } else {
                Vec::new()
            },
        );

        // Linear Attention (Gated DeltaNet) module
        for sub in [
            "linear-attention/projections/qkv",
            "linear-attention/projections/z",
            "linear-attention/projections/b",
            "linear-attention/projections/a",
            "linear-attention/projections/qkvz",
            "linear-attention/projections/ba",
            "linear-attention/projections/out",
            "linear-attention/conv",
            "linear-attention/norm",
        ] {
            tree.add_layer(sub, ComponentKind::DenseWeight, gdn_layers.clone());
        }
        tree.add_layer(
            "linear-attention/state",
            ComponentKind::RecurrentState,
            gdn_layers,
        );

        // Hyper-Connections module
        if matches!(preset, "qwen4_exp" | "canonical-superset") {
            tree.add_layer(
                "hyper-connections",
                ComponentKind::DenseWeight,
                all_layers.clone(),
            );
        } else {
            tree.add_layer("hyper-connections", ComponentKind::DenseWeight, Vec::new());
        }

        // Dense MLP vs MoE
        let dense_layers = if matches!(preset, "dense" | "gemma4" | "canonical-superset") {
            all_layers.clone()
        } else {
            Vec::new()
        };
        for sub in ["dense-mlp/gate", "dense-mlp/up", "dense-mlp/down"] {
            tree.add_layer(sub, ComponentKind::DenseWeight, dense_layers.clone());
        }

        let moe_layers = if is_moe { all_layers } else { Vec::new() };
        tree.add_layer("moe/router", ComponentKind::DenseWeight, moe_layers.clone());
        tree.add_layer(
            "moe/shared-experts/gate",
            ComponentKind::DenseWeight,
            moe_layers.clone(),
        );
        tree.add_layer(
            "moe/shared-experts/projections",
            ComponentKind::DenseWeight,
            moe_layers.clone(),
        );
        tree.add_layer(
            "moe/routed-experts/projections",
            ComponentKind::SparseWeight,
            moe_layers,
        );

        tree
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_and_memloc_parsing() {
        assert_eq!("cpu:0".parse::<Device>().unwrap(), Device::Cpu(0));
        assert_eq!("cuda:1".parse::<Device>().unwrap(), Device::Cuda(1));
        assert_eq!("cpu".parse::<Device>().unwrap(), Device::Cpu(0));
        assert!("tpu:0".parse::<Device>().is_err());

        assert_eq!("ram:0".parse::<MemLoc>().unwrap(), MemLoc::Ram(0));
        assert_eq!("vram:2".parse::<MemLoc>().unwrap(), MemLoc::Vram(2));
        assert_eq!("disk".parse::<MemLoc>().unwrap(), MemLoc::Disk);
        assert!("nvme:0".parse::<MemLoc>().is_err());
    }

    #[test]
    fn test_access_mode_matrix() {
        // VRAM -> GPU = Direct
        assert_eq!(
            AccessMode::resolve(MemLoc::Vram(0), Device::Cuda(0), false),
            AccessMode::Direct
        );
        // VRAM -> CPU = Readback
        assert_eq!(
            AccessMode::resolve(MemLoc::Vram(0), Device::Cpu(0), false),
            AccessMode::Readback
        );
        // RAM -> CPU = Direct
        assert_eq!(
            AccessMode::resolve(MemLoc::Ram(0), Device::Cpu(0), true),
            AccessMode::Direct
        );
        // RAM -> GPU (full tensor) = Streamed
        assert_eq!(
            AccessMode::resolve(MemLoc::Ram(0), Device::Cuda(0), false),
            AccessMode::Streamed
        );
        // RAM -> GPU (sparse row/page lookup) = ZeroCopy
        assert_eq!(
            AccessMode::resolve(MemLoc::Ram(0), Device::Cuda(0), true),
            AccessMode::ZeroCopy
        );
        // Disk -> CPU = Fetch
        assert_eq!(
            AccessMode::resolve(MemLoc::Disk, Device::Cpu(0), true),
            AccessMode::Fetch
        );
        // Disk -> GPU = FetchH2d
        assert_eq!(
            AccessMode::resolve(MemLoc::Disk, Device::Cuda(0), true),
            AccessMode::FetchH2d
        );
    }

    #[test]
    fn test_component_path_prefix_matching() {
        let p = ComponentPath::new("moe/routed-experts/projections");
        assert!(p.starts_with_path(&ComponentPath::new("moe")));
        assert!(p.starts_with_path(&ComponentPath::new("moe/routed-experts")));
        assert!(p.starts_with_path(&ComponentPath::new("moe/routed-experts/projections")));
        assert!(!p.starts_with_path(&ComponentPath::new("moe/routed")));
    }

    #[test]
    fn test_shape_and_dtype() {
        let s = Shape::from([16, 256]);
        assert_eq!(s.rank(), 2);
        assert_eq!(s.numel(), 4096);
        assert_eq!(s.byte_len(DType::BF16), 8192);
        assert_eq!(s.contiguous_strides(), vec![256, 1]);
        assert_eq!(s.to_string(), "[16, 256]");
    }
}
