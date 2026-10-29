//! Foundational types for `lumen-rs`: data types (`DType`), tensor shapes (`Shape`),
//! compute devices (`Device`), storage locations (`MemLoc`), data access modes (`AccessMode`),
//! model component paths (`ComponentPath`, `ComponentTree`, `LayerMask`), and error types (`LumenError`).
//!
//! This crate knows nothing about concrete model families: component trees are built by
//! `lumen-models` (`Architecture::components`) and only *described* here.

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
#[non_exhaustive]
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
///
/// Serialized as the canonical lowercase name (`"bf16"`); deserialization goes through
/// [`FromStr`], so it also accepts the spellings used by `safetensors` (`"BF16"`, `"F32"`, ...).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
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

impl Serialize for DType {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for DType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
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

    /// Total number of elements across all dimensions, or `None` if it overflows `usize`.
    ///
    /// Shapes come from untrusted file headers, so there is deliberately no unchecked variant.
    #[inline]
    pub fn checked_numel(&self) -> Option<usize> {
        self.0.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
    }

    /// Total size in bytes for the given `DType`, or `None` if it overflows `usize`.
    #[inline]
    pub fn checked_byte_len(&self, dtype: DType) -> Option<usize> {
        self.checked_numel()?.checked_mul(dtype.element_size())
    }

    /// Contiguous row-major strides in elements, or `None` if a stride overflows `usize`.
    pub fn contiguous_strides(&self) -> Option<Vec<usize>> {
        let mut strides = vec![1usize; self.0.len()];
        for i in (0..self.0.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1].checked_mul(self.0[i + 1])?;
        }
        Some(strides)
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

impl From<String> for ComponentPath {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// Set of layer indices on which a component exists. A growable bitset: the number of layers is
/// not limited (models with more than 128 layers exist).
///
/// Invariant: no trailing zero words, so derived `Eq`/`Hash` are set equality.
#[derive(Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct LayerMask {
    words: Vec<u64>,
}

impl LayerMask {
    /// The empty set.
    #[inline]
    pub const fn empty() -> Self {
        Self { words: Vec::new() }
    }

    /// All layers `0..num_layers`.
    pub fn all(num_layers: usize) -> Self {
        let mut words = vec![u64::MAX; num_layers / 64];
        if num_layers % 64 != 0 {
            words.push((1u64 << (num_layers % 64)) - 1);
        }
        Self { words }
    }

    /// Layers `l in 0..num_layers` for which `f(l)` is true.
    pub fn from_fn(num_layers: usize, mut f: impl FnMut(usize) -> bool) -> Self {
        let mut mask = Self::empty();
        for l in 0..num_layers {
            if f(l) {
                mask.insert(l);
            }
        }
        mask
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    #[inline]
    pub fn contains(&self, layer: usize) -> bool {
        self.words
            .get(layer / 64)
            .is_some_and(|w| (w >> (layer % 64)) & 1 != 0)
    }

    pub fn insert(&mut self, layer: usize) {
        let word = layer / 64;
        if self.words.len() <= word {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1u64 << (layer % 64);
    }

    pub fn union_with(&mut self, other: &Self) {
        if self.words.len() < other.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= b;
        }
    }

    /// Number of layers in the set.
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterates the layer indices in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(i, &w)| {
            (0..64)
                .filter(move |b| (w >> b) & 1 != 0)
                .map(move |b| i * 64 + b)
        })
    }
}

/// Classification of a leaf component in a `ComponentTree`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ComponentKind {
    /// Dense matrix or weight tensor accessed in full during forward pass.
    DenseWeight,
    /// Row-indexed lookup table (`RowTable`: embeddings, PLE, n-gram tables). Only the rows of the
    /// current tokens are read, in decode and in prefill alike.
    RowTable,
    /// Bank of routed experts (`ExpertBank`). Decode reads only the selected `top-k` experts;
    /// during prefill a chunk touches (almost) all of them, so it is read in full.
    ExpertBank,
    /// Paged attention KV cache (`KvPool`).
    KvCache,
    /// Compact per-token key cache scanned by the sparse-attention indexer; read like `KvCache`.
    IndexerCache,
    /// Recurrent linear-attention or PLE state (`StatePool`).
    RecurrentState,
    /// Pure compute operator node without persistent weights (e.g., `rope`).
    ComputeOnly,
}

impl ComponentKind {
    /// Whether a decode step reads only a sparse subset of rows/blocks/experts.
    #[inline]
    pub const fn sparse_in_decode(self) -> bool {
        matches!(
            self,
            Self::RowTable | Self::ExpertBank | Self::KvCache | Self::IndexerCache
        )
    }

    /// Whether a prefill step reads only a sparse subset of rows/blocks.
    #[inline]
    pub const fn sparse_in_prefill(self) -> bool {
        matches!(self, Self::RowTable | Self::KvCache | Self::IndexerCache)
    }
}

/// Metadata for a single leaf component in `ComponentTree`.
#[derive(Clone, Debug)]
pub struct ComponentLeaf {
    pub path: ComponentPath,
    pub kind: ComponentKind,
    /// Layer indices `[0, num_layers)` where this component exists (empty for globals).
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
                active_layers: LayerMask::empty(),
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
            .and_modify(|leaf| leaf.active_layers.union_with(&mask))
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

    #[test]
    fn test_layer_mask_beyond_128_layers() {
        for n in [0usize, 1, 63, 64, 65, 127, 128, 129, 130, 200] {
            let all = LayerMask::all(n);
            assert_eq!(all.count(), n, "n={n}");
            assert!(n == 0 || all.contains(n - 1));
            assert!(!all.contains(n));
            assert_eq!(all, LayerMask::from_fn(n, |_| true), "n={n}");
            assert_eq!(all.iter().collect::<Vec<_>>(), (0..n).collect::<Vec<_>>());
        }
        let mut m = LayerMask::empty();
        assert!(m.is_empty());
        m.insert(300);
        assert!(m.contains(300) && !m.contains(299) && !m.is_empty());
        let mut n = LayerMask::from_fn(10, |l| l % 3 == 0);
        n.union_with(&m);
        assert_eq!(n.iter().collect::<Vec<_>>(), vec![0, 3, 6, 9, 300]);
        // Equality is set equality regardless of construction order.
        assert_eq!(LayerMask::from_fn(70, |l| l == 69), {
            let mut x = LayerMask::empty();
            x.insert(69);
            x
        });
    }

    #[test]
    fn test_dtype_serde_accepts_safetensors_names() {
        for (json, want) in [
            ("\"bf16\"", DType::BF16),
            ("\"BF16\"", DType::BF16),
            ("\"F32\"", DType::F32),
            ("\"u8\"", DType::U8),
        ] {
            assert_eq!(serde_json::from_str::<DType>(json).unwrap(), want);
        }
        assert_eq!(serde_json::to_string(&DType::F16).unwrap(), "\"f16\"");
        assert!(serde_json::from_str::<DType>("\"BOOL\"").is_err());
    }

    #[test]
    fn test_shape_checked_arithmetic() {
        let s = Shape::from([4usize, 8, 2]);
        assert_eq!(s.checked_numel(), Some(64));
        assert_eq!(s.checked_byte_len(DType::BF16), Some(128));
        assert_eq!(s.contiguous_strides(), Some(vec![16, 2, 1]));
        assert_eq!(Shape::from([0usize, usize::MAX]).checked_numel(), Some(0));
        let huge = Shape::from([usize::MAX, 2]);
        assert_eq!(huge.checked_numel(), None);
        assert_eq!(
            Shape::from([usize::MAX / 2 + 1]).checked_byte_len(DType::U16),
            None
        );
        assert_eq!(huge.contiguous_strides(), Some(vec![2, 1]));
        assert_eq!(
            Shape::from([2usize, usize::MAX, 2]).contiguous_strides(),
            None
        );
    }

    #[test]
    fn test_component_kind_access_semantics() {
        // Embedding tables are gathered in prefill too; routed experts are read in full there.
        assert!(ComponentKind::RowTable.sparse_in_prefill());
        assert!(ComponentKind::ExpertBank.sparse_in_decode());
        assert!(!ComponentKind::ExpertBank.sparse_in_prefill());
        assert!(!ComponentKind::DenseWeight.sparse_in_decode());
    }
}
