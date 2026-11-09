# LTQ Design Rationale, Quality Comparisons, and Quantizer

[Русский](../RU/ltq-design.md)

This document explains why LTQ is used and what quantization experiments showed. The tables apply to the stated models and research conditions. See the [glossary](glossary.md) for definitions and the separate [specification](ltq-format.md) for reading file bytes.

> The migration to C17 with CPU and Vulkan backends preserves the LTQ v1 byte specification. The research GPU results below are not measurements of the new Vulkan backend.

**Implementation status:** this document records the LTQ design and research workflow. The new engine's native quantizer is not implemented; the earlier Rust `lumen-quantize` crate remains a placeholder and does not define its future API. Quantizer development is a separate track from first inference; see [roadmap.md](roadmap.md).

This document presents experimental comparisons of the **LTQ** format against existing low-bit quantization formats (`EXL3` and `GGUF IQ*_K`), analyzes the sources of its accuracy gains, and describes the planned `velri-quantize` quantization pipeline.

For the bit-level format specification, see [ltq-format.md](ltq-format.md).

---

## 1. Quality Comparison Against EXL3 and GGUF

The primary accuracy metric for quantized models is mean **KL Divergence (KLD)** of output token distributions relative to the unquantized `BF16` baseline under teacher forcing. Evaluations are run on two held-out traces disjoint from the calibration data: English Nemotron dialogues and a multilingual trace covering Russian, Chinese, German, French, and Spanish.

In experimental benchmarks ([`vtq-research/`](https://github.com/lumen-inference/vtq-research)), LTQ was evaluated against **EXL3** (`exllamav3`) and lookup-table **`IQ*_K`** formats from `GGUF`:

| Model | Architecture | Baseline (at matched bits/weight) | Relative KLD Change of LTQ vs. EXL3 (English / Multilingual, lower is better) |
| :--- | :--- | :--- | :--- |
| **Llama-3.2-1B** | Dense | Stock EXL3 converter (`convert.py`, 2.5…4.5 bpw) | Better or parity across all five bitrates |
| **Llama-3.2-3B** | Dense | Stock EXL3 converter | Parity at 2.5 & 3.0; **−10% / −4%** at 3.44; parity / **−6%** at 4.0; **−17% / −18%** at 4.5 |
| **Qwen3-8B** | Dense | Stock EXL3 converter (3.0 / 3.44 / 4.5 bpw) | **−10% / −5%** at 3.0; **−11% / −18%** at 3.44; **−28% / −31%** at 4.5 |
| **Qwen3-30B-A3B** | MoE (128 experts) | Stock EXL3 converter (3.0 / 4.5 bpw) | **−52% / −51%** at 3.0; **−14% / −12%** at 4.5 |
| **Qwen3.5-4B** | Hybrid Gated DeltaNet + Attention | Per-matrix EXL3 quantizer on identical Hessians (3.0 / 3.5 / 4.0) | **−14…−18% / −18…−21%** |
| **Gemma 4 E4B** | PLE, Shared KV, Sliding Attention | Per-matrix EXL3 quantizer on identical Hessians | **−15…−26% / −17…−22%** |

In the research implementation, LTQ quantization on a single GPU runs **2.4–7× faster** than EXL3 conversion: quantizing `Qwen3-8B` takes 4–5 minutes, and `Qwen3-30B-A3B` takes 11–13 minutes on an RTX PRO 6000.

### Comparison Against Published Quants on `Qwen3.8-27B`

LTQ was compared against published quantizations of `Qwen3.8-27B` at matched sizes: **GSQ-RCO** (GGUF by IST-DASLab with gradient-searched per-tensor type allocation) and **EXL3** (including self-calibrated `SC`). To ensure an exact comparison, decoder linear layers of each quant were reconstructed to `BF16` with identical unquantized embeddings and `lm_head` ([`results/q27_report.txt`](https://github.com/lumen-inference/vtq-research/blob/main/results/q27_report.txt)):

| Bits/Weight | LTQ KLD (Eng / Multi) | Competitor KLD | KLD Difference |
| :---: | :---: | :--- | :---: |
| **2.94** | `0.0246 / 0.0278` | GSQ-RCO `IQ3_XXS`: `0.0480 / 0.0549` | **−49% / −49%** |
| **3.00** | `0.0239 / 0.0259` | EXL3 SC 3.0: `0.0260 / 0.0263` | **−8% / −1%** (within noise) |
| **3.48** | `0.0119 / 0.0133` | GSQ-RCO `IQ3_S`: `0.0221 / 0.0244` | **−46% / −46%** |
| **3.50** | `0.0124 / 0.0130` | EXL3 3.5: `0.0186 / 0.0191` | **−33% / −32%** |

*Note:* The LTQ calibration set in this benchmark (self-calibrated chat and agent traces) is closer in domain to the evaluation traces than the standard `imatrix` used by GSQ-RCO.

---

## 2. Sources of Accuracy Gains (Ablation Breakdown)

1. **Core Trellis Quantizer on Par with EXL3.**
   Ungerboeck TCQ with 1024 states, tail-biting tiles, extended subset labels (up to 16 subsets), and Gaussian-trained codebooks matches EXL3 on individual matrices of `Qwen3-8B` (median layer difference between $-0.03$ and $+0.04$ bits).
2. **Fractional Bitrates Without Layer Mixing (`D2` and `D4`).**
   EXL3 natively supports only integer bitrates plus 2.5 and 3.5 bpw, forcing it to mix 4-bit and 5-bit layers to achieve a 4.5 bpw average. LTQ's pair ($K - 0.5$) and quad ($K - 0.25$) parity modes provide a ~0.25-bit menu step, yielding a 28% KLD reduction on `Qwen3-8B` at 4.5 bpw.
3. **Fisher-Weighted Row-Slice Bit Allocation (DP Knapsack).**
   In MoE models such as `Qwen3-30B-A3B`, attention layers account for only 3% of parameters but dominate output sensitivity. Weighting quantization error by Fisher information allows the DP solver to allocate ~4.4 bits to attention slices at the cost of only 0.05 bits on routed experts — cutting KLD at a 3.0 bpw budget by 52%.
4. **Sensitivity Profile Smoothing.**
   Empirical expert sensitivity measured on a narrow calibration set can overfit to specific topics or languages (for instance, aggressively dropping cold experts to 2 bits degrades multilingual KLD on 30B by 55%). Consequently, the native quantizer requires sensitivity profile smoothing and diverse multilingual calibration traces.
5. **Hadamard-Rotated LTQ Rings for Row Lookup Tables (`LtqRing`).**
   On the `Gemma 4 E4B` PLE table (5.25 GiB in `bf16`), `TCQ4X` rings with block Hadamard rotation compress the table to 1.39 GiB at `0.0010` KLD (compared to 2.65 GiB and `0.0006` KLD for `int8`). Omitting the Hadamard rotation increases KLD by 2.4× due to coordinate outliers.

---

## 3. Planned Native Quantizer Pipeline

The pipeline brings the research algorithm into the C17 engine. Forward passes use [common model blocks](blocks.md), while quantization operations are checked through the [backend registry](backends.md). CPU or Vulkan inference support alone does not imply support for Hessians, backward passes, or Viterbi search. An unsupported stage needs a supported path selected in advance or an explicit preparation error.

Processing proceeds layer by layer, with one current layer's weights in working memory. Activations, Hessians, profiles, and temporary representations require separate memory budgeting. The algorithm proceeds as follows:

1. **Calibration Pass and Hessian Collection:**
   The engine runs a forward pass over the calibration dataset, accumulating input Hessians $H = X X^T$ for each group of linear layers sharing an input. Calibration activations flow through already-quantized layers so subsequent layers compensate for accumulated quantization error.
2. **Row-Wise Fisher Information Estimation:**
   Measures loss sensitivity to perturbations in each matrix row over sampled model continuations. Until backward passes for all supported modules are implemented natively in the engine, Fisher profiles can be imported from the reference Python scripts (`fisher_full.py` / `fisher_offload.py`).
3. **Slice Profiling and DP Format Assignment:**
   Each row slice is evaluated with a fast proxy quantizer, and its Fisher-weighted error is scaled by the calibrated distortion curve of each of the 19 menu formats. Dynamic programming solves the knapsack problem to assign optimal formats under the target bit budget.
4. **Matrix Quantization (GPU in the Research Implementation):**
   - Computes input signs and scales `suh` (via Cauchy–Schwarz), applies Hadamard-256 to the weights and Hessian, and damps the Hessian diagonal ($0.2\times$ mean diagonal).
   - Runs **Block16-LDLQ** across 16-column blocks, propagating quantization error into remaining unquantized columns.
   - Executes tail-biting **Viterbi** search across $16 \times 16$ tiles, followed by linear scale polishing of `d_row` and `suh`. The search contract is specified in [LTQ v1, Section 8](ltq-format.md#8-quantizer-contract). The research implementation uses a GPU kernel for the search; the new engine must separately establish support on the selected backend.
5. **`.vtq` Container Serialization:**
   Writes the compressed tensors to the `.vtq` file and reconstructs the quantized weights to advance calibration activations to the next layer.

The file format is separate from a device's internal weight representation. Preparation explicitly includes the loading, packing, unpacking, and transfers required by the selected implementation; copying bytes does not convert layouts. Adding a backend does not change LTQ v1: the backend declares support for the required operations and representations through the common contract. Research CUDA code can remain a reference without becoming a mandatory engine dependency or a claim of an available Vulkan path.

---

## References

- LTQ v1 Specification: [ltq-format.md](ltq-format.md); experimental repository: [`vtq-research/`](https://github.com/lumen-inference/vtq-research)
- M. W. Marcellin, T. R. Fischer, *«Trellis Coded Quantization of Memoryless and Gauss-Markov Sources»*, IEEE Trans. Commun., 1990
- G. Ungerboeck, *«Channel Coding with Multilevel/Phase Signals»*, IEEE Trans. Inf. Theory, 1982
- *QTIP: Quantization with Trellises and Incoherence Processing* ([arXiv:2406.11235](https://arxiv.org/abs/2406.11235))
- [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3) — EXL3 reference implementation
- Published comparison quants: [`Vemloir/Qwen3.8-Flash-Next-sc-exl3`](https://huggingface.co/Vemloir/Qwen3.8-Flash-Next-sc-exl3), [`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF), [`turboderp/Qwen3.8-27B-exl3`](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)
