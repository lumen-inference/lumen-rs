//! Closed operator set `enum Op`, backend interface `trait Executor`, and hardware bandwidth
//! profile `HwProfile` (`docs/RU/architecture.md` Sections 1, 7, 9 and `docs/RU/performance.md`).

use lumen_core::{Device, MemLoc, Result};
use lumen_tensor::{ExpertBank, MatWeight, RowTable, TensorRef};

// ============================================================================
// Weight table: operators refer to weights by id, they never own them
// ============================================================================

/// Index of a linear weight in [`WeightTable::mats`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct MatId(pub u32);

/// Index of a lookup table in [`WeightTable::row_tables`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct RowTableId(pub u32);

/// Index of an expert bank in [`WeightTable::banks`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct BankId(pub u32);

/// All weight descriptors of a `Program`, stored once.
///
/// `Op` carries only small ids into this table, so a `Program` stays cheap, plain data
/// (cloning or printing an operator never copies per-expert descriptor vectors).
#[derive(Clone, Debug, Default)]
pub struct WeightTable {
    pub mats: Vec<MatWeight>,
    pub row_tables: Vec<RowTable>,
    pub banks: Vec<ExpertBank>,
}

impl WeightTable {
    pub fn add_mat(&mut self, weight: MatWeight) -> MatId {
        self.mats.push(weight);
        MatId(u32::try_from(self.mats.len() - 1).expect("more than u32::MAX weights"))
    }

    pub fn add_row_table(&mut self, table: RowTable) -> RowTableId {
        self.row_tables.push(table);
        RowTableId(u32::try_from(self.row_tables.len() - 1).expect("more than u32::MAX tables"))
    }

    pub fn add_bank(&mut self, bank: ExpertBank) -> BankId {
        self.banks.push(bank);
        BankId(u32::try_from(self.banks.len() - 1).expect("more than u32::MAX banks"))
    }

    pub fn mat(&self, id: MatId) -> Option<&MatWeight> {
        self.mats.get(id.0 as usize)
    }

    pub fn row_table(&self, id: RowTableId) -> Option<&RowTable> {
        self.row_tables.get(id.0 as usize)
    }

    pub fn bank(&self, id: BankId) -> Option<&ExpertBank> {
        self.banks.get(id.0 as usize)
    }
}

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
    /// $\text{SiLU}(g) \cdot u$.
    SwiGlu,
    /// $\text{GELU}_{\tanh}(g) \cdot u$ (the tanh approximation used by `gelu_pytorch_tanh`).
    GeGlu,
}

/// Closed set of primitive operators executed by `trait Executor`.
///
/// Architectural invariant (`docs/RU/architecture.md` Section 9):
/// Backends must match on `Op` exhaustively without a wildcard `_` branch so adding a new
/// operator forces compile-time handling across all backends.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// Row lookup from a `RowTable` (embeddings, PLE tables, n-gram hash tables).
    RowLookup {
        table: RowTableId,
        indices: TensorRef,
        out: TensorRef,
    },
    /// Linear projection $y = W x + b$ (`LtqLinear` or `DenseLinear`).
    Linear {
        weight: MatId,
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
        /// Token positions: `[T]` for 1D variants, `[3, T]` for `MRope`.
        positions: TensorRef,
        spec: RopeSpec,
    },
    /// Paged full, sliding-window, or sparse (`QSA`) attention.
    Attention {
        layer: usize,
        q: TensorRef,
        k: TensorRef,
        v: TensorRef,
        out: TensorRef,
        /// Softmax scale applied to $q \cdot k$ (e.g. $1/\sqrt{d}$, or `1.0` with QK-norm).
        scale: f32,
        sliding_window: Option<u32>,
        k_eq_v: bool,
        softcap: Option<f32>,
        has_output_gate: bool,
        /// Learned per-head attention sink logits.
        sinks: Option<TensorRef>,
        /// Bidirectional intra-span mask for image tokens, unioned with the causal mask.
        bidirectional_spans: Option<TensorRef>,
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
        /// Clamp applied to the gate/up inputs (e.g. `7.0` for SwiGLU in gpt-oss-style MLPs).
        clamp: Option<f32>,
    },
    /// MoE router scoring and top-$k$ expert selection.
    Router {
        bank: BankId,
        input: TensorRef,
        expert_ids: TensorRef,
        expert_weights: TensorRef,
    },
    /// Grouped MoE matrix multiplication (`mgemm`) across active experts in `ExpertBank`.
    MoEGemm {
        bank: BankId,
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
        table: RowTableId,
        indices: TensorRef,
        staging: TensorRef,
        flag_slot: u32,
    },
    /// On-device token sampling chain (`softcap -> repeat-penalty -> temp -> top-k -> top-p -> min-p -> Philox`).
    Sample {
        logits: TensorRef,
        out_token_id: TensorRef,
        softcap: Option<f32>,
        repeat_penalty: f32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        min_p: f32,
        /// Philox key; the counter is derived from the step index.
        seed: u64,
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
    /// Reference rig from `docs/EN/performance.md` Section 4: RTX 5070 Ti 16 GB (GDDR7, 896 GB/s;
    /// PCIe 5.0 x16, ~63 GB/s per direction) + Ryzen 9 9950X (dual-channel DDR5-6000, 96 GB/s
    /// theoretical peak) + NVMe SSD.
    ///
    /// The document gives no NVMe throughput (disk reads are hidden behind early layers and are
    /// not on the ceiling); `7e9` is a placeholder for a PCIe 4.0 x4 SSD.
    pub const fn reference_workstation() -> Self {
        Self {
            vram_bw_bytes_per_sec: 896e9,
            ram_bw_bytes_per_sec: 96e9,
            pcie_h2d_bw_bytes_per_sec: 63e9,
            pcie_d2h_bw_bytes_per_sec: 63e9,
            nvme_bw_bytes_per_sec: 7e9,
        }
    }
}

/// Unified compute backend interface implemented by `lumen-ref`, `lumen-cpu`, and `lumen-cuda`.
///
/// Backends match on [`Op`] exhaustively (no `_` arm), so a new operator does not compile until
/// every backend decides how to handle it. A backend that has no kernel for an operator returns
/// `false` from [`Executor::supports`]; loading then fails instead of silently skipping work.
pub trait Executor: Send {
    /// Physical or logical device managed by this executor.
    fn device(&self) -> Device;

    /// Returns `true` if this backend can execute `op`.
    fn supports(&self, op: &Op) -> bool;

    /// Repacks or uploads an `LtqLinear` or `DenseLinear` weight for this backend (`CpuBand16` / `GpuWarp`).
    fn prepare_weight(&mut self, weight: &mut MatWeight) -> Result<()>;

    /// Prepares every linear weight of `table` for this backend.
    fn prepare_weights(&mut self, table: &mut WeightTable) -> Result<()> {
        table
            .mats
            .iter_mut()
            .try_for_each(|w| self.prepare_weight(w))
    }

    /// Executes a single primitive operator `op`; `weights` resolves the ids it carries.
    ///
    /// Must return an error (never `Ok(())`) for an operator it does not execute.
    fn execute_op(&mut self, op: &Op, weights: &WeightTable) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_core::{DType, Shape};
    use lumen_mem::BufferId;
    use lumen_tensor::{DenseLinear, WeightRef};

    fn dense(m: u32) -> MatWeight {
        MatWeight::Dense(DenseLinear {
            m,
            k: 4,
            dtype: DType::BF16,
            weight: WeightRef {
                buffer: BufferId(0),
                byte_offset: 0,
                byte_len: u64::from(m) * 8,
                dtype: DType::BF16,
                shape: Shape::from([m as usize, 4]),
            },
            bias: None,
        })
    }

    #[test]
    fn test_weight_table_ids_resolve() {
        let mut t = WeightTable::default();
        let a = t.add_mat(dense(2));
        let b = t.add_mat(dense(3));
        assert_eq!((a, b), (MatId(0), MatId(1)));
        assert!(matches!(t.mat(b), Some(MatWeight::Dense(d)) if d.m == 3));
        assert!(t.mat(MatId(2)).is_none());
        assert!(t.row_table(RowTableId(0)).is_none());
        assert!(t.bank(BankId(0)).is_none());
    }

    #[test]
    fn test_op_is_cheap_to_clone() {
        // An operator holds ids, not descriptor vectors.
        let op = Op::Linear {
            weight: MatId(7),
            input: TensorRef {
                id: BufferId(1),
                dtype: DType::F32,
                shape: Shape::from([1usize, 4]),
            },
            out: TensorRef {
                id: BufferId(2),
                dtype: DType::F32,
                shape: Shape::from([1usize, 2]),
            },
        };
        assert_eq!(op.clone(), op);
        assert!(std::mem::size_of::<Op>() <= 512);
    }

    #[test]
    fn test_reference_rig_matches_performance_doc() {
        let hw = HwProfile::reference_workstation();
        assert_eq!(hw.vram_bw_bytes_per_sec, 896e9);
        assert_eq!(hw.ram_bw_bytes_per_sec, 96e9);
        assert_eq!(hw.pcie_h2d_bw_bytes_per_sec, 63e9);
        assert_eq!(hw.pcie_d2h_bw_bytes_per_sec, 63e9);
    }
}
