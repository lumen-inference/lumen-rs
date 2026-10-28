//! Closed operator set `enum Op`, backend interface `trait Executor`, and hardware bandwidth
//! profile `HwProfile` (`docs/RU/architecture.md` Sections 1, 7, 9 and `docs/RU/performance.md`).

use lumen_core::{Device, MemLoc, Result};
use lumen_tensor::{ExpertBank, MatWeight, RowTable, TensorRef};

/// RoPE positional encoding variant (`docs/RU/kernels.md` Section 2).
#[derive(Clone, Debug, PartialEq)]
pub enum RopeSpec {
    /// Standard full-head RoPE with base $\theta$.
    Standard { theta: f32, head_dim: u32 },
    /// Partial RoPE (rotates only the first `rotary_dim` dimensions of each head).
    Partial {
        theta: f32,
        head_dim: u32,
        rotary_dim: u32,
    },
    /// Proportional RoPE (`p-RoPE`): frequencies $\theta^{-2i/d}$ for the first $p \cdot d / 2$ angles
    /// with full-head `rotate-half`.
    Proportional {
        theta: f32,
        head_dim: u32,
        routed_fraction: f32,
    },
    /// YaRN context-extended RoPE.
    Yarn {
        theta: f32,
        head_dim: u32,
        factor: f32,
        original_max_pos: u32,
    },
    /// 3D Multimodal RoPE (`M-RoPE`) splitting angles into $(t, h, w)$ sections.
    MRope {
        theta: f32,
        head_dim: u32,
        sections: [u32; 3],
    },
}

/// Elementwise activation function for gated MLP blocks.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum GatedActivation {
    SwiGlu,
    GeGlu,
}

/// Closed set of primitive operators executed by `trait Executor`.
///
/// Architectural invariant (`docs/RU/architecture.md` Section 9):
/// Backends must match on `Op` exhaustively without a wildcard `_` branch so adding a new
/// operator forces compile-time handling across all backends.
#[derive(Clone, Debug)]
pub enum Op {
    /// Row lookup from a `RowTable` (embeddings, PLE tables, n-gram hash tables).
    RowLookup {
        table: RowTable,
        indices: TensorRef,
        out: TensorRef,
    },
    /// Linear projection $y = W x + b$ (`LtqLinear` or `DenseLinear`).
    Linear {
        weight: MatWeight,
        input: TensorRef,
        out: TensorRef,
    },
    /// Root-Mean-Square layer normalization (`unit_offset = true` uses scale $1 + w$).
    RmsNorm {
        input: TensorRef,
        weight: Option<TensorRef>,
        out: TensorRef,
        eps: f32,
        unit_offset: bool,
    },
    /// Gated RMSNorm used at the output of `GatedDeltaNet`.
    RmsNormGated {
        input: TensorRef,
        gate: TensorRef,
        weight: TensorRef,
        out: TensorRef,
        eps: f32,
    },
    /// Rotary positional embedding (`RoPE`) applied to $Q$ and $K$.
    Rope {
        q: TensorRef,
        k: TensorRef,
        spec: RopeSpec,
    },
    /// Paged full, sliding-window, or sparse (`QSA`) attention.
    Attention {
        layer: usize,
        q: TensorRef,
        k: TensorRef,
        v: TensorRef,
        out: TensorRef,
        sliding_window: Option<u32>,
        k_eq_v: bool,
        softcap: Option<f32>,
        has_output_gate: bool,
    },
    /// Sparse attention block indexer (`QSA`): computes $\text{ReLU}(q \cdot k)$ and selects top-$k$ blocks.
    IndexerScore {
        layer: usize,
        q: TensorRef,
        k: TensorRef,
        selected_blocks: TensorRef,
        top_k_blocks: u32,
    },
    /// Recurrent/chunked `GatedDeltaNet` (`GDN`) linear attention step + short causal `conv1d`.
    DeltaNet {
        layer: usize,
        qkv: TensorRef,
        z: TensorRef,
        b: TensorRef,
        a: TensorRef,
        out: TensorRef,
    },
    /// Gated MLP elementwise activation (`SwiGLU` or `GeGLU`).
    GatedMlpAct {
        gate: TensorRef,
        up: TensorRef,
        out: TensorRef,
        act: GatedActivation,
    },
    /// MoE router scoring and top-$k$ expert selection.
    Router {
        bank: ExpertBank,
        input: TensorRef,
        expert_ids: TensorRef,
        expert_weights: TensorRef,
    },
    /// Grouped MoE matrix multiplication (`mgemm`) across active experts in `ExpertBank`.
    MoEGemm {
        bank: ExpertBank,
        input: TensorRef,
        expert_ids: TensorRef,
        expert_weights: TensorRef,
        out: TensorRef,
    },
    /// Hyper-Connections (`HcMix`): mixes $n$ parallel residual streams before a sub-block.
    HcMix {
        streams: TensorRef,
        alpha: TensorRef,
        out: TensorRef,
        num_streams: u32,
    },
    /// Hyper-Connections (`HcInject`): injects a sub-block's output back into the residual streams.
    HcInject {
        streams: TensorRef,
        subblock_out: TensorRef,
        beta: TensorRef,
        num_streams: u32,
    },
    /// Elementwise residual addition $y = a + \text{scale} \cdot b$.
    ResidualAdd {
        a: TensorRef,
        b: TensorRef,
        out: TensorRef,
        scale: f32,
    },
    /// Cross-device transfer synchronized via pinned RAM flags (`A[l]`, `B[l]`).
    Transfer {
        src_loc: MemLoc,
        dst_loc: MemLoc,
        src: TensorRef,
        dst: TensorRef,
        bytes: usize,
        flag_slot: u32,
    },
    /// Asynchronous pre-fetch of rows from an NVMe-backed `RowTable` into a pinned staging buffer.
    Fetch {
        table: RowTable,
        indices: TensorRef,
        staging: TensorRef,
        flag_slot: u32,
    },
    /// On-device token sampling chain (`softcap -> repeat-penalty -> temp -> top-k -> top-p -> min-p -> Philox`).
    Sample {
        logits: TensorRef,
        out_token_id: TensorRef,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        min_p: f32,
    },
}

/// Measured hardware bandwidth profile produced by `lumen probe` (`docs/RU/performance.md`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HwProfile {
    /// Sustained GPU VRAM read bandwidth in bytes/sec (`B_vram`).
    pub vram_bw_bytes_per_sec: f64,
    /// Sustained CPU DRAM read bandwidth in bytes/sec (`B_ram`).
    pub ram_bw_bytes_per_sec: f64,
    /// Sustained PCIe Host-to-Device bandwidth in bytes/sec (`B_h2d`).
    pub pcie_h2d_bw_bytes_per_sec: f64,
    /// Sustained PCIe Device-to-Host bandwidth in bytes/sec (`B_d2h`).
    pub pcie_d2h_bw_bytes_per_sec: f64,
    /// Sustained NVMe random/sequential read bandwidth in bytes/sec (`B_disk`).
    pub nvme_bw_bytes_per_sec: f64,
}

impl HwProfile {
    /// Reference workstation profile from `docs/RU/performance.md`:
    /// RTX 5080 (16 GB GDDR7, ~780 GB/s) + 128 GB DDR4-3200 (~76 GB/s) + PCIe 4.0 x16 (~25 GB/s).
    pub const fn reference_workstation() -> Self {
        Self {
            vram_bw_bytes_per_sec: 780e9,
            ram_bw_bytes_per_sec: 76e9,
            pcie_h2d_bw_bytes_per_sec: 25e9,
            pcie_d2h_bw_bytes_per_sec: 25e9,
            nvme_bw_bytes_per_sec: 7e9,
        }
    }
}

/// Unified compute backend interface implemented by `lumen-ref`, `lumen-cpu`, and `lumen-cuda`.
pub trait Executor: Send + Sync {
    /// Physical or logical device managed by this executor.
    fn device(&self) -> Device;

    /// Returns `true` if this backend supports executing `op`.
    fn supports(&self, op: &Op) -> bool;

    /// Repacks or uploads an `LtqLinear` or `DenseLinear` weight for this backend (`CpuBand16` / `GpuWarp`).
    fn prepare_weight(&mut self, weight: &mut MatWeight) -> Result<()>;

    /// Executes a single primitive operator `op`.
    fn execute_op(&mut self, op: &Op) -> Result<()>;
}
