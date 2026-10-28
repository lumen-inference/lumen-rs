//! Tensor descriptors (`TensorRef`, `WeightRef`) and canonical weight descriptors (`LtqLinear`,
//! `RowTable`, `DenseLinear`, `ExpertBank`) defined in `docs/RU/architecture.md` Section 6.

use lumen_core::{DType, Shape};
use lumen_mem::BufferId;
use serde::{Deserialize, Serialize};

/// Reference to a contiguous byte region inside a weight or activation `Buffer`.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct WeightRef {
    pub buffer: BufferId,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub dtype: DType,
    pub shape: Shape,
}

/// Symbolic reference to an activation or intermediate tensor inside `Program`.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TensorRef {
    pub id: BufferId,
    pub dtype: DType,
    pub shape: Shape,
}

/// Tile memory layout inside an `LtqLinear` matrix (`docs/RU/architecture.md` Section 6).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum TileLayout {
    /// On-disk SoA layout per slice (`codes`, `planes`, `lsb`, `gains` arrays).
    Canonical,
    /// CPU-optimized layout: 16-row horizontal bands laid out sequentially along $K$.
    CpuBand16,
    /// GPU-optimized layout: 32-bit word-aligned bit planes for coalesced warp reads.
    GpuWarp,
}

/// Horizontal row slice of an `LtqLinear` matrix assigned a specific LTQ format from the menu.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct LtqSlice {
    /// First row index (inclusive, multiple of 16).
    pub row_start: u32,
    /// End row index (exclusive, multiple of 16).
    pub row_end: u32,
    /// Index into the container's `quantization.formats` table.
    pub format_idx: u32,
    /// Byte offset of this slice within `{p}.ltq.data`.
    pub byte_offset: u64,
}

impl LtqSlice {
    #[inline]
    pub const fn num_rows(&self) -> u32 {
        self.row_end - self.row_start
    }
}

/// Canonical LTQ-quantized linear weight descriptor ($y = W x + b$).
#[derive(Clone, Debug)]
pub struct LtqLinear {
    /// Logical output dimension $M$ (rows).
    pub m: u32,
    /// Logical input dimension $K$ (columns).
    pub k: u32,
    /// Padded output dimension $M_{\text{pad}}$ (multiple of 16).
    pub m_pad: u32,
    /// Padded input dimension $K_{\text{pad}}$ (multiple of 256).
    pub k_pad: u32,
    /// Horizontal row slices and their LTQ format assignments.
    pub slices: Vec<LtqSlice>,
    /// Packed tile data (`{p}.ltq.data`).
    pub data: WeightRef,
    /// Per-row base scales in `fp16` (`{p}.ltq.d_row`, $[M]$).
    pub d_row: WeightRef,
    /// Per-block exponents in `int8` (`{p}.ltq.e`, $[M, K_{\text{pad}} / 256]$).
    pub e: WeightRef,
    /// Input sign-and-scale vector in `fp16` (`{p}.ltq.suh`, $[K_{\text{pad}}]$).
    pub suh: WeightRef,
    /// Optional additive bias vector (`{p}.bias`, $[M]$).
    pub bias: Option<WeightRef>,
    /// Physical tile packing layout in memory.
    pub layout: TileLayout,
}

impl LtqLinear {
    /// Computes padded $(M_{\text{pad}}, K_{\text{pad}})$ from logical $(M, K)$.
    #[inline]
    pub const fn pad_dims(m: u32, k: u32) -> (u32, u32) {
        let m_pad = m.div_ceil(16) * 16;
        let k_pad = k.div_ceil(256) * 256;
        (m_pad, k_pad)
    }
}

/// Encoding format of a row-indexed lookup table (`RowTable`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum RowCodec {
    /// Unquantized dense rows (`bf16`, `f16`, or `f32`).
    Dense(DType),
    /// Per-row `int8` quantization with a per-row `fp16` scale.
    Int8Rows,
    /// Independent 1D LTQ tail-biting rings of length `ring` with block Hadamard `hadamard`
    /// (`docs/RU/ltq-format.md` Section 9).
    LtqRing {
        format: u32,
        ring: u32,
        hadamard: u32,
    },
}

/// Canonical descriptor for row-indexed lookup tables (vocabulary embeddings, PLE tables, n-gram hash tables).
#[derive(Clone, Debug)]
pub struct RowTable {
    pub rows: u64,
    pub dim: u32,
    pub codec: RowCodec,
    pub shards: Vec<WeightRef>,
    pub signs: Option<WeightRef>,
    pub head_offsets: Option<Vec<u64>>,
    pub head_bias: Option<WeightRef>,
}

/// Canonical descriptor for unquantized linear projections (`lm_head`, routers, small adapters).
#[derive(Clone, Debug)]
pub struct DenseLinear {
    pub m: u32,
    pub k: u32,
    pub dtype: DType,
    pub weight: WeightRef,
    pub bias: Option<WeightRef>,
}

/// Unified weight enum for linear operators.
#[derive(Clone, Debug)]
pub enum MatWeight {
    Ltq(LtqLinear),
    Dense(DenseLinear),
}

/// Scoring function used by an MoE router (`docs/RU/models/README.md`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum RouterScoring {
    Softmax,
    Sigmoid,
}

/// Normalized MoE router specification across model families (`Qwen3-MoE`, `Qwen3-Next`, `gpt-oss`, `Gemma 4`).
#[derive(Clone, Debug)]
pub struct RouterSpec {
    pub num_experts: u32,
    pub top_k: u32,
    pub scoring: RouterScoring,
    pub renormalize_topk: bool,
    pub weight_scale: f32,
    pub router_weight: DenseLinear,
    pub e_score_correction_bias: Option<WeightRef>,
}

/// Single expert's 3-matrix MLP weights (`gate`, `up`, `down`).
#[derive(Clone, Debug)]
pub struct ExpertWeights {
    pub gate: MatWeight,
    pub up: MatWeight,
    pub down: MatWeight,
}

/// Bank of routed experts for a Mixture-of-Experts (`MoE`) layer.
#[derive(Clone, Debug)]
pub struct ExpertBank {
    pub router: RouterSpec,
    pub experts: Vec<ExpertWeights>,
}
