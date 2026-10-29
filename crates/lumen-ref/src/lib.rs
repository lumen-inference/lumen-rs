//! Scalar `f32` reference executor (`RefExecutor`) used as the numerical oracle for differential
//! tests across all operators and `PlacementPlan` configurations (`docs/RU/architecture.md` Sections 2 & 9).

use lumen_backend::{Executor, GatedActivation, Op, WeightTable};
use lumen_core::{Device, LumenError, Result, f16};
use lumen_ltq::{LtqFormat, decode_tile, rotate_input_256};
use lumen_tensor::{MatWeight, TileLayout};

/// Scalar `f32` reference backend implementing `trait Executor`.
///
/// **Status:** the pure kernels below ([`ref_rmsnorm_f32`], [`ref_gated_mlp_act_f32`],
/// [`ref_ltq_band16_gemv_f32`]) exist, but binding them to `Op`/`TensorRef` needs the buffer
/// model of `lumen-exec`, which is not written yet. Until an operator is wired up the executor
/// says so honestly: [`Executor::supports`] is `false` and [`Executor::execute_op`] fails.
/// An oracle that answers "ok" without computing anything would make every differential test
/// pass vacuously.
pub struct RefExecutor {
    device: Device,
}

impl RefExecutor {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Short operator name used in error messages.
    fn op_name(op: &Op) -> &'static str {
        match op {
            Op::RowLookup { .. } => "RowLookup",
            Op::Linear { .. } => "Linear",
            Op::RmsNorm { .. } => "RmsNorm",
            Op::RmsNormGated { .. } => "RmsNormGated",
            Op::Rope { .. } => "Rope",
            Op::Attention { .. } => "Attention",
            Op::IndexerScore { .. } => "IndexerScore",
            Op::DeltaNet { .. } => "DeltaNet",
            Op::GatedMlpAct { .. } => "GatedMlpAct",
            Op::Router { .. } => "Router",
            Op::MoEGemm { .. } => "MoEGemm",
            Op::HcMix { .. } => "HcMix",
            Op::HcInject { .. } => "HcInject",
            Op::ResidualAdd { .. } => "ResidualAdd",
            Op::Transfer { .. } => "Transfer",
            Op::Fetch { .. } => "Fetch",
            Op::Sample { .. } => "Sample",
        }
    }
}

impl Default for RefExecutor {
    fn default() -> Self {
        Self::new(Device::Cpu(0))
    }
}

impl Executor for RefExecutor {
    fn device(&self) -> Device {
        self.device
    }

    /// Exhaustive match over `Op` without a wildcard `_` arm (Architecture Section 9 invariant):
    /// flip an arm to `true` in the same change that implements the operator.
    fn supports(&self, op: &Op) -> bool {
        match op {
            Op::RowLookup { .. }
            | Op::Linear { .. }
            | Op::RmsNorm { .. }
            | Op::RmsNormGated { .. }
            | Op::Rope { .. }
            | Op::Attention { .. }
            | Op::IndexerScore { .. }
            | Op::DeltaNet { .. }
            | Op::GatedMlpAct { .. }
            | Op::Router { .. }
            | Op::MoEGemm { .. }
            | Op::HcMix { .. }
            | Op::HcInject { .. }
            | Op::ResidualAdd { .. }
            | Op::Transfer { .. }
            | Op::Fetch { .. }
            | Op::Sample { .. } => false,
        }
    }

    /// The reference executor reads the canonical on-disk layout only, so there is nothing to
    /// repack; a weight that was already repacked for another backend is rejected rather than
    /// relabelled.
    fn prepare_weight(&mut self, weight: &mut MatWeight) -> Result<()> {
        match weight {
            MatWeight::Ltq(ltq) if ltq.layout != TileLayout::Canonical => {
                Err(LumenError::Format(format!(
                    "reference executor reads the canonical tile layout only, got {:?}",
                    ltq.layout
                )))
            }
            MatWeight::Ltq(_) | MatWeight::Dense(_) => Ok(()),
        }
    }

    /// Exhaustive match over `Op` without a wildcard `_` arm (Architecture Section 9 invariant).
    fn execute_op(&mut self, op: &Op, _weights: &WeightTable) -> Result<()> {
        match op {
            Op::RowLookup { .. }
            | Op::Linear { .. }
            | Op::RmsNorm { .. }
            | Op::RmsNormGated { .. }
            | Op::Rope { .. }
            | Op::Attention { .. }
            | Op::IndexerScore { .. }
            | Op::DeltaNet { .. }
            | Op::GatedMlpAct { .. }
            | Op::Router { .. }
            | Op::MoEGemm { .. }
            | Op::HcMix { .. }
            | Op::HcInject { .. }
            | Op::ResidualAdd { .. }
            | Op::Transfer { .. }
            | Op::Fetch { .. }
            | Op::Sample { .. } => Err(LumenError::UnsupportedOp {
                device: self.device,
                op: Self::op_name(op).to_string(),
            }),
        }
    }
}

// ============================================================================
// Pure Reference Scalar f32 Kernels (Oracle Functions)
// ============================================================================

/// Reference scalar RMSNorm:
/// $y_i = \frac{x_i}{\sqrt{\frac{1}{D}\sum_j x_j^2 + \varepsilon}} \cdot \text{scale}_i$
/// where $\text{scale}_i = 1 + w_i$ if `unit_offset` is true, else $w_i$.
///
/// The reduction runs in `f64` so that the oracle is more precise than the `f32` kernels it
/// checks.
///
/// # Panics
/// Panics if `x`, `out` (and `weight`, when given) differ in length.
pub fn ref_rmsnorm_f32(
    x: &[f32],
    weight: Option<&[f32]>,
    eps: f32,
    unit_offset: bool,
    out: &mut [f32],
) {
    assert_eq!(x.len(), out.len());
    if let Some(w) = weight {
        assert_eq!(w.len(), x.len(), "norm weight length must match the input");
    }
    let d = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let inv_rms = 1.0 / (sum_sq / d + f64::from(eps)).sqrt();

    for (i, (&xi, yi)) in x.iter().zip(out.iter_mut()).enumerate() {
        let scale = match weight {
            Some(w) if unit_offset => 1.0 + f64::from(w[i]),
            Some(w) => f64::from(w[i]),
            None => 1.0,
        };
        *yi = (f64::from(xi) * inv_rms * scale) as f32;
    }
}

/// Reference scalar gated MLP activation (`SwiGLU` or `GeGLU` with the tanh GELU approximation),
/// optionally clamping `gate` and `up` to `[-clamp, clamp]` first.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn ref_gated_mlp_act_f32(
    gate: &[f32],
    up: &[f32],
    act: GatedActivation,
    clamp: Option<f32>,
    out: &mut [f32],
) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), out.len());
    let clamp = |v: f32| clamp.map_or(v, |c| v.clamp(-c, c));
    for ((&g, &u), y) in gate.iter().zip(up.iter()).zip(out.iter_mut()) {
        let (g, u) = (f64::from(clamp(g)), f64::from(clamp(u)));
        let activated = match act {
            GatedActivation::SwiGlu => g / (1.0 + (-g).exp()),
            GatedActivation::GeGlu => {
                // GELU, tanh approximation: 0.5 g (1 + tanh(sqrt(2/pi) (g + 0.044715 g^3)))
                let c = (2.0f64 / std::f64::consts::PI).sqrt();
                0.5 * g * (1.0 + (c * (g + 0.044715 * g * g * g)).tanh())
            }
        };
        *y = (activated * u) as f32;
    }
}

/// Reference scalar LTQ linear reconstruction and matrix-vector product for a single
/// 16-row band (`m_band = 16`, `k_pad` multiple of 256) per Section 2 of `docs/RU/ltq-format.md`:
///
/// $$\tilde{x} = H_{256}(\text{suh} \odot x)$$
/// $$y[r] = \sum_{b} d_{\text{row}}[r] \cdot 2^{e[r, b] / 4} \sum_{k \in b} q[r, k] \cdot \tilde{x}[k] + \text{bias}[r]$$
///
/// Sums are accumulated in `f64`. `tiles_*` hold one entry per 16-column tile along $K$.
///
/// # Panics
/// Panics if `x.len()` is not a multiple of 256 or any of the per-tile / per-block slices does
/// not have exactly the matching length.
#[allow(clippy::too_many_arguments)]
pub fn ref_ltq_band16_gemv_f32(
    fmt: &LtqFormat,
    x: &[f32],
    suh: &[f16],
    d_row: &[f16; 16],
    e_blocks: &[[i8; 16]], // indexed as [block_b][row_r]
    tiles_codes: &[[u16; 16]],
    tiles_planes: &[Vec<[u16; 16]>],
    tiles_lsb: &[Vec<u16>],
    tiles_gains: &[Vec<u16>],
    bias: Option<&[f32; 16]>,
) -> [f32; 16] {
    let k_pad = x.len();
    assert_eq!(k_pad % 256, 0, "K must be padded to a multiple of 256");
    let num_tiles_k = k_pad / 16;
    let num_blocks_256 = k_pad / 256;
    assert_eq!(tiles_codes.len(), num_tiles_k);
    assert_eq!(tiles_planes.len(), num_tiles_k);
    assert_eq!(tiles_lsb.len(), num_tiles_k);
    assert_eq!(tiles_gains.len(), num_tiles_k);
    assert_eq!(
        e_blocks.len(),
        num_blocks_256,
        "one exponent row per 256-block"
    );

    let mut x_tilde = vec![0.0f32; k_pad];
    rotate_input_256(x, suh, &mut x_tilde);

    let mut y = [0.0f64; 16];
    for (b, e_block) in e_blocks.iter().enumerate() {
        let mut block_acc = [0.0f64; 16];
        for tb in 0..16 {
            let kb = b * 16 + tb;
            let q_tile = decode_tile(
                fmt,
                &tiles_codes[kb],
                &tiles_planes[kb],
                &tiles_lsb[kb],
                &tiles_gains[kb],
            );
            let x_sub = &x_tilde[kb * 16..(kb + 1) * 16];
            for (r, acc) in block_acc.iter_mut().enumerate() {
                let row_q = &q_tile[r * 16..(r + 1) * 16];
                *acc += row_q
                    .iter()
                    .zip(x_sub)
                    .map(|(&qv, &xv)| f64::from(qv) * f64::from(xv))
                    .sum::<f64>();
            }
        }
        for r in 0..16 {
            let d = f64::from(d_row[r].to_f32()) * 2f64.powf(f64::from(e_block[r]) / 4.0);
            y[r] += d * block_acc[r];
        }
    }

    let mut out = [0.0f32; 16];
    for r in 0..16 {
        out[r] = (y[r] + bias.map_or(0.0, |b| f64::from(b[r]))) as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_backend::{MatId, RowTableId};
    use lumen_core::{DType, Shape};
    use lumen_ltq::Frac;
    use lumen_mem::BufferId;
    use lumen_tensor::{LtqLinear, TensorRef, WeightRef};

    fn tensor(id: u32) -> TensorRef {
        TensorRef {
            id: BufferId(id),
            dtype: DType::F32,
            shape: Shape::from([4usize]),
        }
    }

    #[test]
    fn test_executor_does_not_pretend_to_execute() {
        let mut exec = RefExecutor::default();
        let weights = WeightTable::default();
        let ops = [
            Op::Linear {
                weight: MatId(0),
                input: tensor(0),
                out: tensor(1),
            },
            Op::RowLookup {
                table: RowTableId(0),
                indices: tensor(0),
                out: tensor(1),
            },
            Op::ResidualAdd {
                a: tensor(0),
                b: tensor(1),
                out: tensor(2),
                scale: 1.0,
            },
        ];
        for op in &ops {
            assert!(!exec.supports(op), "{op:?}");
            let err = exec.execute_op(op, &weights).unwrap_err();
            assert!(matches!(err, LumenError::UnsupportedOp { .. }), "{err}");
        }
    }

    fn ltq_linear(layout: TileLayout) -> LtqLinear {
        let wr = |dtype| WeightRef {
            buffer: BufferId(0),
            byte_offset: 0,
            byte_len: 0,
            dtype,
            shape: Shape::from([0usize]),
        };
        LtqLinear {
            m: 16,
            k: 256,
            m_pad: 16,
            k_pad: 256,
            slices: vec![],
            data: wr(DType::U8),
            d_row: wr(DType::F16),
            e: wr(DType::I8),
            suh: wr(DType::F16),
            bias: None,
            layout,
        }
    }

    #[test]
    fn test_prepare_weight_does_not_relabel_foreign_layouts() {
        let mut exec = RefExecutor::default();
        let mut canonical = MatWeight::Ltq(ltq_linear(TileLayout::Canonical));
        exec.prepare_weight(&mut canonical).unwrap();
        let mut packed = MatWeight::Ltq(ltq_linear(TileLayout::CpuBand16));
        assert!(exec.prepare_weight(&mut packed).is_err());
        assert!(matches!(&packed, MatWeight::Ltq(l) if l.layout == TileLayout::CpuBand16));
    }

    #[test]
    fn test_ref_rmsnorm_standard_and_unit_offset() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [0.5f32, 0.5, 0.5, 0.5];
        let mut out_std = [0.0f32; 4];
        let mut out_unit = [0.0f32; 4];

        ref_rmsnorm_f32(&x, Some(&w), 1e-6, false, &mut out_std);
        ref_rmsnorm_f32(&x, Some(&w), 1e-6, true, &mut out_unit);

        // With w = 0.5, unit_offset scale is 1.5 = 3 * 0.5
        for i in 0..4 {
            assert!((out_unit[i] - 3.0 * out_std[i]).abs() < 1e-6);
        }
        // mean(x^2) = 7.5 -> inv_rms = 1/sqrt(7.5)
        let want = 1.0 / 7.5f32.sqrt() * 0.5;
        assert!((out_std[0] - want).abs() < 1e-6);

        let mut no_w = [0.0f32; 4];
        ref_rmsnorm_f32(&x, None, 1e-6, false, &mut no_w);
        assert!((no_w[3] - 4.0 / 7.5f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_ref_gated_mlp_known_values() {
        let gate = [0.0f32, 1.0, -1.0, 100.0];
        let up = [2.0f32; 4];
        let mut out = [0.0f32; 4];

        ref_gated_mlp_act_f32(&gate, &up, GatedActivation::SwiGlu, None, &mut out);
        // silu(1) = 1 / (1 + e^-1) = 0.731058...
        assert_eq!(out[0], 0.0);
        assert!((out[1] - 2.0 * 0.731_058_6).abs() < 1e-6);
        assert!((out[2] - 2.0 * -0.268_941_4).abs() < 1e-6);
        assert!((out[3] - 200.0).abs() < 1e-3);

        ref_gated_mlp_act_f32(&gate, &up, GatedActivation::GeGlu, None, &mut out);
        // gelu_tanh(1) = 0.841192...
        assert!((out[1] - 2.0 * 0.841_192).abs() < 1e-5);

        // Clamp to 7: the 100.0 gate saturates and silu(7) * min(up, 7)
        ref_gated_mlp_act_f32(
            &gate,
            &[50.0; 4],
            GatedActivation::SwiGlu,
            Some(7.0),
            &mut out,
        );
        let silu7 = 7.0 / (1.0 + (-7.0f64).exp());
        assert!((f64::from(out[3]) - silu7 * 7.0).abs() < 1e-3);
    }

    /// Independent check of the band/tile/block indexing of the GEMV: dequantize the whole
    /// `16 x K` band explicitly, rotate `x`, and compute a plain dot product per row.
    #[test]
    fn test_ltq_band16_gemv_matches_explicit_dequantized_matmul() {
        let mut state = 0x1234_5678_9ABC_DEF1u64;
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let fmt = LtqFormat {
            name: "t".into(),
            k: 3,
            frac: Frac::D2,
            masks: vec![359, 241],
            h: 1,
            sub: 16,
            lut: (0..8 * 4).map(|i| (i as f32) * 0.0625 - 1.0).collect(),
            gain: vec![0.8, 1.2],
        };
        fmt.validate().unwrap();

        let k_pad = 512usize; // two 256-blocks, 32 tiles
        let tiles = k_pad / 16;
        let words = |rnd: &mut dyn FnMut() -> u64, n| -> Vec<u16> {
            (0..n).map(|_| (rnd() >> 20) as u16).collect()
        };

        let mut codes = Vec::new();
        let mut planes = Vec::new();
        let mut lsb = Vec::new();
        let mut gains = Vec::new();
        for _ in 0..tiles {
            codes.push(<[u16; 16]>::try_from(words(&mut rnd, 16)).unwrap());
            let p = words(&mut rnd, 16 * (fmt.k - 2));
            planes.push(
                p.chunks_exact(16)
                    .map(|c| <[u16; 16]>::try_from(c).unwrap())
                    .collect::<Vec<_>>(),
            );
            lsb.push(words(&mut rnd, fmt.lsb_words(256)));
            gains.push(words(&mut rnd, fmt.gain_words(256)));
        }

        let x: Vec<f32> = (0..k_pad)
            .map(|i| ((i * 37 % 101) as f32) / 50.0 - 1.0)
            .collect();
        let suh: Vec<f16> = (0..k_pad)
            .map(|i| f16::from_f32(if i % 3 == 0 { -1.0 } else { 0.75 }))
            .collect();
        let d_row: [f16; 16] = std::array::from_fn(|r| f16::from_f32(0.5 + r as f32 * 0.125));
        let e_blocks: Vec<[i8; 16]> = (0..k_pad / 256)
            .map(|b| std::array::from_fn(|r| (r as i8) - 8 + (b as i8) * 3))
            .collect();
        let bias: [f32; 16] = std::array::from_fn(|r| r as f32 * 0.01);

        let got = ref_ltq_band16_gemv_f32(
            &fmt,
            &x,
            &suh,
            &d_row,
            &e_blocks,
            &codes,
            &planes,
            &lsb,
            &gains,
            Some(&bias),
        );

        let mut x_tilde = vec![0.0f32; k_pad];
        rotate_input_256(&x, &suh, &mut x_tilde);
        for r in 0..16 {
            let mut want = f64::from(bias[r]);
            for kb in 0..tiles {
                let q = decode_tile(&fmt, &codes[kb], &planes[kb], &lsb[kb], &gains[kb]);
                let scale =
                    f64::from(d_row[r].to_f32()) * 2f64.powf(f64::from(e_blocks[kb / 16][r]) / 4.0);
                for c in 0..16 {
                    want += scale * f64::from(q[r * 16 + c]) * f64::from(x_tilde[kb * 16 + c]);
                }
            }
            assert!(
                (f64::from(got[r]) - want).abs() < 1e-4 * want.abs().max(1.0),
                "row {r}: {} vs {want}",
                got[r]
            );
        }
    }
}
