//! Scalar `f32` reference executor (`RefExecutor`) used as the numerical oracle for differential
//! tests across all operators and `PlacementPlan` configurations (`docs/RU/architecture.md` Sections 2 & 9).

use lumen_backend::{Executor, GatedActivation, Op};
use lumen_core::{f16, Device, Result};
use lumen_ltq::{decode_tile, rotate_input_256, LtqFormat};
use lumen_tensor::{MatWeight, TileLayout};

/// Scalar `f32` reference backend implementing `trait Executor`.
pub struct RefExecutor {
    device: Device,
}

impl RefExecutor {
    pub fn new(device: Device) -> Self {
        Self { device }
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

    /// Exhaustive match over `Op` without a wildcard `_` arm (Architecture Section 9 invariant).
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
            | Op::Sample { .. } => true,
        }
    }

    fn prepare_weight(&mut self, weight: &mut MatWeight) -> Result<()> {
        if let MatWeight::Ltq(ltq) = weight {
            ltq.layout = TileLayout::Canonical;
        }
        Ok(())
    }

    /// Exhaustive match over `Op` without a wildcard `_` arm (Architecture Section 9 invariant).
    fn execute_op(&mut self, op: &Op) -> Result<()> {
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
            | Op::Sample { .. } => Ok(()),
        }
    }
}

// ============================================================================
// Pure Reference Scalar f32 Kernels (Oracle Functions)
// ============================================================================

/// Reference scalar `f32` RMSNorm:
/// $y_i = \frac{x_i}{\sqrt{\frac{1}{D}\sum_j x_j^2 + \varepsilon}} \cdot \text{scale}_i$
/// where $\text{scale}_i = 1 + w_i$ if `unit_offset` is true, else $w_i$.
pub fn ref_rmsnorm_f32(
    x: &[f32],
    weight: Option<&[f32]>,
    eps: f32,
    unit_offset: bool,
    out: &mut [f32],
) {
    assert_eq!(x.len(), out.len());
    let d = x.len() as f32;
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / d + eps).sqrt();

    for (i, (&xi, yi)) in x.iter().zip(out.iter_mut()).enumerate() {
        let scale = match weight {
            Some(w) if unit_offset => 1.0 + w[i],
            Some(w) => w[i],
            None => 1.0,
        };
        *yi = xi * inv_rms * scale;
    }
}

/// Reference scalar `f32` Gated MLP activation (`SwiGLU` or `GeGLU`).
pub fn ref_gated_mlp_act_f32(gate: &[f32], up: &[f32], act: GatedActivation, out: &mut [f32]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), out.len());
    for ((&g, &u), y) in gate.iter().zip(up.iter()).zip(out.iter_mut()) {
        let activated = match act {
            GatedActivation::SwiGlu => g / (1.0 + (-g).exp()),
            GatedActivation::GeGlu => {
                // Exact GELU(g) = 0.5 * g * (1 + erf(g / sqrt(2))) approximated via tanh
                let c = (2.0f32 / std::f32::consts::PI).sqrt();
                0.5 * g * (1.0 + (c * (g + 0.044715 * g * g * g)).tanh())
            }
        };
        *y = activated * u;
    }
}

/// Reference scalar `f32` LTQ linear reconstruction and matrix-vector product for a single
/// 16-row band (`m_band = 16`, `k_pad` multiple of 256) per Section 2 of `docs/RU/ltq-format.md`:
///
/// $$\tilde{x} = H_{256}(\text{suh} \odot x)$$
/// $$y[r] = \sum_{b} d_{\text{row}}[r] \cdot 2^{e[r, b] / 4} \sum_{k \in b} q[r, k] \cdot \tilde{x}[k] + \text{bias}[r]$$
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
    assert_eq!(k_pad % 256, 0);
    let num_tiles_k = k_pad / 16;
    assert_eq!(tiles_codes.len(), num_tiles_k);

    let mut x_tilde = vec![0.0f32; k_pad];
    rotate_input_256(x, suh, &mut x_tilde);

    let mut y = [0.0f32; 16];
    let num_blocks_256 = k_pad / 256;

    for (b, e_block) in e_blocks.iter().enumerate().take(num_blocks_256) {
        let mut block_acc = [0.0f32; 16];
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
            for r in 0..16 {
                let row_q = &q_tile[r * 16..(r + 1) * 16];
                let dot: f32 = row_q.iter().zip(x_sub.iter()).map(|(&qv, &xv)| qv * xv).sum();
                block_acc[r] += dot;
            }
        }
        for r in 0..16 {
            let d = d_row[r].to_f32() * 2.0f32.powf((e_block[r] as f32) / 4.0);
            y[r] += d * block_acc[r];
        }
    }

    if let Some(b_vec) = bias {
        for r in 0..16 {
            y[r] += b_vec[r];
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

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
    }

    /// CI Architectural Boundary Check (`docs/RU/architecture.md` Section 9 & `roadmap.md` Phase 1):
    /// Verifies that model family identifiers (`qwen`, `gemma`, `llama`, `gpt_oss`, `mistral`)
    /// never leak into scheduler or backend crates (`lumen-backend`, `lumen-exec`, `lumen-cpu`,
    /// `lumen-cuda`, `lumen-ref`).
    #[test]
    fn test_ci_architectural_model_isolation_invariant() {
        let crates_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let restricted_crates = [
            "lumen-backend",
            "lumen-exec",
            "lumen-cpu",
            "lumen-cuda",
            "lumen-cuda-kernels",
            "lumen-ref",
        ];
        let forbidden_model_tokens = ["qwen", "gemma", "llama", "gpt_oss", "gpt-oss", "mistral"];

        for crate_name in restricted_crates {
            let src_dir = crates_root.join(crate_name).join("src");
            assert!(src_dir.exists(), "missing src dir for {crate_name}");
            for entry in std::fs::read_dir(&src_dir).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                    continue;
                }
                let content = std::fs::read_to_string(&path).unwrap();
                // Strip test modules when scanning `lumen-ref/src/lib.rs` itself
                let prod_code = content.split("#[cfg(test)]").next().unwrap_or(&content);
                // Strip doc/line comments
                let code_no_comments: String = prod_code
                    .lines()
                    .filter(|l| !l.trim_start().starts_with("//"))
                    .collect::<Vec<_>>()
                    .join("\n")
                    .to_ascii_lowercase();

                for token in forbidden_model_tokens {
                    assert!(
                        !code_no_comments.contains(token),
                        "Architectural violation: forbidden model identifier '{token}' found in {}",
                        path.display()
                    );
                }
            }
        }
    }
}
