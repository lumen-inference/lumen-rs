//! Foundational types for `lumen-rs`: data types (`DType`), tensor shapes (`Shape`),
//! compute devices (`Device`), storage locations (`MemLoc`), data access modes (`AccessMode`),
//! model component paths (`ComponentPath`, `ComponentTree`, `LayerMask`), and error types (`LumenError`).

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
// DType & Shape
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
// Device, MemLoc & AccessMode
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
        match s.trim() {
            "cpu" => Ok(Self::Cpu(0)),
            "cuda" => Ok(Self::Cuda(0)),
            other => {
                let (kind, idx_str) = other.split_once(':').ok_or_else(|| {
                    LumenError::Config(format!(
                        "invalid compute device '{other}' (expected 'cpu:N' or 'cuda:G')"
                    ))
                })?;
                let idx = idx_str.parse::<u16>().map_err(|_| {
                    LumenError::Config(format!("invalid device index in '{other}' (expected u16)"))
                })?;
                match kind {
                    "cpu" => Ok(Self::Cpu(idx)),
                    "cuda" => Ok(Self::Cuda(idx)),
                    _ => Err(LumenError::Config(format!(
                        "unknown compute device '{other}' (expected 'cpu:N' or 'cuda:G')"
                    ))),
                }
            }
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
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
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
        match s.trim() {
            "disk" => Ok(Self::Disk),
            "ram" => Ok(Self::Ram(0)),
            "vram" => Ok(Self::Vram(0)),
            other => {
                let (kind, idx_str) = other.split_once(':').ok_or_else(|| {
                    LumenError::Config(format!(
                        "invalid memory location '{other}' (expected 'ram:N', 'vram:G', or 'disk')"
                    ))
                })?;
                let idx = idx_str.parse::<u16>().map_err(|_| {
                    LumenError::Config(format!(
                        "invalid memory location index in '{other}' (expected u16)"
                    ))
                })?;
                match kind {
                    "ram" => Ok(Self::Ram(idx)),
                    "vram" => Ok(Self::Vram(idx)),
                    _ => Err(LumenError::Config(format!(
                        "unknown memory location '{other}' (expected 'ram:N', 'vram:G', or 'disk')"
                    ))),
                }
            }
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
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// Data access mode determined at plan compilation from `(store, compute, sparse_access)`.
///
/// Corresponds to the $3 \times 4$ access matrix in `docs/RU/architecture.md` (Section 3).
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
    Readback,
    /// Cross-GPU peer-to-peer copy (`vram:G1` -> `cuda:G2`).
    PeerCopy,
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
// ComponentPath, LayerMask & ComponentTree
// ============================================================================

/// Slash-separated path identifying a component in the model hierarchy
/// (e.g., `"input-layer"`, `"ple/table"`, `"moe/routed-experts"`).
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ComponentPath(String);

impl ComponentPath {
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into().trim_matches('/').to_string())
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
        prefix.0.is_empty()
            || self.0 == prefix.0
            || self
                .0
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

/// Zero-allocation $O(1)$ bitmask of active layer indices (supports up to 128 layers).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct LayerMask(pub u128);

impl LayerMask {
    pub const EMPTY: Self = Self(0);

    /// Bitmask with all layers `0..num_layers` set.
    #[inline]
    pub const fn all(num_layers: usize) -> Self {
        if num_layers == 0 {
            Self(0)
        } else if num_layers >= 128 {
            Self(u128::MAX)
        } else {
            Self((1u128 << num_layers) - 1)
        }
    }

    /// Bitmask built from a predicate `f(layer_idx)`.
    pub fn from_fn(num_layers: usize, mut f: impl FnMut(usize) -> bool) -> Self {
        let mut bits = 0u128;
        for l in 0..num_layers.min(128) {
            if f(l) {
                bits |= 1u128 << l;
            }
        }
        Self(bits)
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn contains(self, layer: usize) -> bool {
        layer < 128 && ((self.0 >> layer) & 1) != 0
    }

    #[inline]
    pub fn insert(&mut self, layer: usize) {
        if layer < 128 {
            self.0 |= 1u128 << layer;
        }
    }

    #[inline]
    pub fn union_with(&mut self, other: Self) {
        self.0 |= other.0;
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
    /// Bitmask of layer indices `[0, num_layers)` where this component exists (`EMPTY` for globals).
    pub active_layers: LayerMask,
}

/// Model component hierarchy produced by `Architecture::components()`.
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
                active_layers: LayerMask::EMPTY,
            },
        );
    }

    /// Registers a per-layer leaf component active on `mask` and all its parent prefixes.
    pub fn add_layer(
        &mut self,
        path: impl Into<ComponentPath>,
        kind: ComponentKind,
        mask: LayerMask,
    ) {
        let path = path.into();
        self.register_prefixes(&path, true);
        self.layer_leaves
            .entry(path.clone())
            .and_modify(|leaf| leaf.active_layers.union_with(mask))
            .or_insert(ComponentLeaf {
                path,
                kind,
                active_layers: mask,
            });
    }

    fn register_prefixes(&mut self, path: &ComponentPath, is_layer: bool) {
        let target = if is_layer {
            &mut self.layer_prefixes
        } else {
            &mut self.global_prefixes
        };
        let mut accum = ComponentPath::new("");
        for seg in path.segments() {
            accum = accum.join(seg);
            target.insert(accum.clone());
        }
    }

    #[inline]
    pub fn is_valid_global_path(&self, path: &ComponentPath) -> bool {
        self.global_prefixes.contains(path)
    }

    #[inline]
    pub fn is_valid_layer_path(&self, path: &ComponentPath) -> bool {
        self.layer_prefixes.contains(path)
    }

    /// Canonical superset tree containing all standard architectural modules from `docs/RU/config.md`.
    pub fn canonical_superset(num_hidden_layers: usize) -> Self {
        Self::preset("canonical-superset", num_hidden_layers)
    }

    /// Builds a component tree for a named architecture preset (`"dense"`, `"moe"`, `"qwen3_next"`,
    /// `"qwen4_exp"`, `"gemma4"`, or `"canonical-superset"`).
    pub fn preset(preset: &str, num_hidden_layers: usize) -> Self {
        let all = LayerMask::all(num_hidden_layers);
        let none = LayerMask::EMPTY;
        let mut tree = Self::new(preset, num_hidden_layers);

        tree.add_global("input-layer", ComponentKind::SparseWeight);
        tree.add_global("output-layer/norm", ComponentKind::DenseWeight);
        tree.add_global("output-layer/lm-head", ComponentKind::DenseWeight);

        if matches!(preset, "moe" | "qwen4_exp" | "gemma4" | "canonical-superset") {
            tree.add_global("ple/table", ComponentKind::SparseWeight);
            tree.add_global("ple/projections", ComponentKind::DenseWeight);
        }
        if matches!(preset, "gemma4" | "canonical-superset") {
            tree.add_global("vision-tower", ComponentKind::DenseWeight);
            tree.add_global("audio-tower", ComponentKind::DenseWeight);
        }
        if matches!(preset, "qwen3_next" | "qwen4_exp" | "canonical-superset") {
            tree.add_global("mtp", ComponentKind::DenseWeight);
        }

        let (attn_mask, gdn_mask) = match preset {
            "qwen3_next" | "qwen4_exp" => (
                LayerMask::from_fn(num_hidden_layers, |l| (l + 1) % 4 == 0),
                LayerMask::from_fn(num_hidden_layers, |l| (l + 1) % 4 != 0),
            ),
            "gemma4" => (all, none),
            _ => (all, all),
        };

        for norm in ["norms/input-norm", "norms/post-attention-norm"] {
            tree.add_layer(norm, ComponentKind::DenseWeight, all);
        }

        for sub in ["attention/q", "attention/k", "attention/v", "attention/o", "attention/qk-norm"] {
            tree.add_layer(sub, ComponentKind::DenseWeight, attn_mask);
        }
        tree.add_layer("attention/rope", ComponentKind::ComputeOnly, attn_mask);
        tree.add_layer("attention/kv-cache", ComponentKind::KvCache, attn_mask);
        tree.add_layer(
            "attention/indexer/keys",
            ComponentKind::SparseWeight,
            if matches!(preset, "moe" | "qwen4_exp" | "canonical-superset") {
                attn_mask
            } else {
                none
            },
        );

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
            tree.add_layer(sub, ComponentKind::DenseWeight, gdn_mask);
        }
        tree.add_layer("linear-attention/state", ComponentKind::RecurrentState, gdn_mask);

        let hc_mask = if matches!(preset, "qwen4_exp" | "canonical-superset") {
            all
        } else {
            none
        };
        tree.add_layer("hyper-connections", ComponentKind::DenseWeight, hc_mask);

        let dense_mask = if matches!(preset, "dense" | "gemma4" | "canonical-superset") {
            all
        } else {
            none
        };
        for sub in ["dense-mlp/gate", "dense-mlp/up", "dense-mlp/down"] {
            tree.add_layer(sub, ComponentKind::DenseWeight, dense_mask);
        }

        let moe_mask = if matches!(preset, "moe" | "qwen3_next" | "qwen4_exp" | "canonical-superset")
        {
            all
        } else {
            none
        };
        tree.add_layer("moe/router", ComponentKind::DenseWeight, moe_mask);
        tree.add_layer("moe/shared-experts/gate", ComponentKind::DenseWeight, moe_mask);
        tree.add_layer(
            "moe/shared-experts/projections",
            ComponentKind::DenseWeight,
            moe_mask,
        );
        tree.add_layer(
            "moe/routed-experts/projections",
            ComponentKind::SparseWeight,
            moe_mask,
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
        assert_eq!(
            AccessMode::resolve(MemLoc::Vram(0), Device::Cuda(0), false),
            AccessMode::Direct
        );
        assert_eq!(
            AccessMode::resolve(MemLoc::Vram(0), Device::Cpu(0), false),
            AccessMode::Readback
        );
        assert_eq!(
            AccessMode::resolve(MemLoc::Ram(0), Device::Cpu(0), true),
            AccessMode::Direct
        );
        assert_eq!(
            AccessMode::resolve(MemLoc::Ram(0), Device::Cuda(0), false),
            AccessMode::Streamed
        );
        assert_eq!(
            AccessMode::resolve(MemLoc::Ram(0), Device::Cuda(0), true),
            AccessMode::ZeroCopy
        );
        assert_eq!(
            AccessMode::resolve(MemLoc::Disk, Device::Cpu(0), true),
            AccessMode::Fetch
        );
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
}
