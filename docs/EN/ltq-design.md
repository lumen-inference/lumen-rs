# LTQ — Format Rationale and Quantizer

Oct 27, 2026 · @Vemloir

Why the engine has its own quantization format, where it wins, and how the quantizer works. The specification is [ltq-format.md](ltq-format.md); how the format fits into the engine is in [section 4 of architecture.md](architecture.md#4-weights-descriptors-and-loading).

**Why LTQ.** The format was chosen empirically ([`ltq-research/`](https://github.com/lumen-inference/ltq-research)) against EXL3 — the strongest open format of the same class — and the table-based `IQ*_K` types from GGUF. The metric is the mean teacher-forced KL divergence from BF16 on two traces that take no part in calibration: English Nemotron dialogues and a multilingual trace (ru, zh, de, fr, es):

| Model | Architecture | Compared against | LTQ KLD relative to EXL3 (English / other languages) |
| --- | --- | --- | --- |
| Llama-3.2-1B | dense | stock `convert.py`, 2.5…4.5 bits/weight | better or on par at all five points |
| Llama-3.2-3B | dense | same | on par at 2.5 and 3.0, −10 / −4 % at 3.44, on par / −6 % at 4.0, −17 / −18 % at 4.5 |
| Qwen3-8B | dense | same, 3.0 / 3.44 / 4.5 | −10 / −5 %, −11 / −18 %, −28 / −31 % |
| Qwen3-30B-A3B | MoE, 128 experts | same, 3.0 / 4.5 | **−52 / −51 %**, −14 / −12 % |
| Qwen3.5-4B | Gated DeltaNet + full attention | the EXL3 quantizer per matrix on the same Hessians, 3.0 / 3.5 / 4.0 | −14…−18 / −18…−21 % |
| Gemma 4 E4B | PLE, shared KV, sliding attention | same (`convert.py` does not support Gemma 4 PLE) | −15…−26 / −17…−22 % |

LTQ quantization on one GPU is 2.4–7 times faster than EXL3 conversion (Qwen3-8B: 4–5 min, Qwen3-30B-A3B: 11–13 min on an RTX PRO 6000).

Where the gain comes from, per the ablations:

- **A quantization core on par with EXL3.** Ungerboeck TCQ with 1024 states, a tail-biting trellis, extended labels (up to 16 level subsets) and tables trained on a Gaussian source. On individual Qwen3-8B matrices, with the error measured on a held-out half of the calibration set, the difference from EXL3, as a median over layers, is −0.03 to +0.04 bits for every matrix type.
- **Fractional rates without a mixing penalty.** Parity of the low uncoded bits of pairs (K − 0.5) and of groups of four (K − 0.25) gives a menu with quarter-bit steps. EXL3 natively has only integer rates and 2.5 / 3.5, so at 4.5 it mixes 4 and 5 bits per layer — hence −28 % on 8B.
- **Bit width per row slice from Fisher information** (a DP knapsack). On MoE at a 3.0 budget, attention (3 % of the parameters) gets 4.4 bits at the cost of 0.05 bits for the experts — that is the −52 % on Qwen3-30B. Without Fisher weights the same menu is twice as bad on 3B.
- **A limit.** Sensitivity measured on calibration data transfers poorly to other data: a "bold" allocation (cold experts at 2 bits) is 55 % worse on the 30B multilingual trace. That is why the DP profile is smoothed and why calibration data must cover the languages and tasks the model will be used for.

**Against published quants: Qwen3.8-27B** (`qwen3_5_dense`, 27.10). Against published quants of the same size: GSQ-RCO (GGUF by IST-DASLab, the type of each tensor chosen by gradient search under a budget) and turboderp's EXL3, including the self-calibrated one. The decoder linear layers of every quant are reconstructed to BF16 and measured on one rig; embeddings, lm\_head and other tensors are BF16 for all, and bits are counted over the same set of matrices (24.3B parameters) with all format fields. KLD nem / mul, the difference is a paired bootstrap over dialogs ([`results/q27_report.txt`](https://github.com/lumen-inference/ltq-research/blob/main/results/q27_report.txt)):

| Bits/weight | LTQ | Competitor | Difference |
| --- | --- | --- | --- |
| 2.94 | 0.0246 / 0.0278 | GSQ-RCO IQ3\_XXS: 0.0480 / 0.0549 | **−49 / −49 %** |
| 3.00 | 0.0239 / 0.0259 | EXL3 SC 3.0: 0.0260 / 0.0263 | −8 / −1 % (within noise) |
| 3.48 | 0.0119 / 0.0133 | GSQ-RCO IQ3\_S: 0.0221 / 0.0244 | **−46 / −46 %** |
| 3.50 | 0.0124 / 0.0130 | EXL3 3.5: 0.0186 / 0.0191 | **−33 / −32 %** |

Caveat: the LTQ calibration (Qwen3.8-Flash-Next self-calibration, chat and agents) is closer in domain to the evaluation traces than the GSQ-RCO imatrix; a check with neutral calibration and an external trace is not done yet.

**Structure.** A weight is reconstructed as:

```
x̃ = H₂₅₆(suh ⊙ x)
y[m] = Σ_b d_row[m] · 2^(e[m,b]/4) · Σ_{k ∈ b} q[m,k] · x̃[k] + bias[m]
q_t = gain[g] · lut[label(s_t) · 2^(K−1) + u_t]
```

- **Tile:** 16 rows × 16 columns, 256 positions in row-major order. A position has one coded bit and K − 1 uncoded bits. The trellis state is the window of the last 10 coded bits, and the trellis is tail-biting within the tile. The subset label is the position's own bit plus 1–3 parities of the window under the format masks.
- **Scales:** fp16 per row, an exponent in \[−16, 15\] per (row, 256-block) stored as int8, an h-bit gain index per 16 or 32 weights. The input is rotated by `suh` and a block Hadamard-256; the output is not rotated.
- **Menu:** 19 formats from 2.04 to 5.29 bits per weight, K = 2…5, integer, D2 and D4. The quantizer assigns a format to every row slice (256 attention rows, 512 MLP rows, a whole expert matrix).

**Row tables** (Qwen3.8-Flash-Next n-grams, embeddings, PLE) are quantized with LTQ rings ([specification, section 9](ltq-format.md#9-row-tables)). A row is cut into rings of length 160–256; each ring is a tail-biting trellis with an fp16 scale, rotated by a block Hadamard, and the rings of a row are consecutive. Rows are independent, so a lookup reads exactly the bytes it needs in one chunk. On Gemma 4 E4B PLE, rings at 4.25 bits with rotation give KLD 0.0010 at 1.39 GiB against 0.0006 for int8 at 2.65 GiB; without rotation KLD is 2.4 times higher. Dense storage (bf16, int8 with a scale) remains for tables where every hundredth of KLD matters.

**Quantizer** (`lumen-quantize`, prototype in [`ltq-research/src/`](https://github.com/lumen-inference/ltq-research/tree/main/src)). The model is processed layer by layer, one layer in memory at a time:

1. Calibration pass through the engine (`lumen-exec` on BF16 weights): the input Hessian of every group of linear layers sharing an input, in execution order. The calibration stream goes through the already quantized layers.
2. Per-row Fisher information for every matrix: a backward pass over samples of the model's own answers (2 samples per position, fixed random numbers). This needs a backward pass of the supported blocks — attention, Gated DeltaNet, MLP, MoE, PLE. Until the engine has it, Fisher comes from the Python prototype (`fisher_full.py`, `fisher_offload.py`).
3. Profile: every row slice is quantized once with a cheap table quantizer, and the Fisher-weighted error, multiplied by a calibrated per-format coefficient, gives the slice's cost in every menu format. A DP knapsack distributes formats over slices under the bits-per-weight budget.
4. Matrix quantization: `suh` (signs and Cauchy–Schwarz scales), Hadamard-256, the Hessian damped by 0.2 of the mean diagonal, Block16-LDLQ over blocks of 16 columns, 16×16 tiles through Viterbi on the tail-biting trellis on the GPU (two passes: a 32-position window up to the middle, then a full shifted pass), then polishing of the row scales and `suh`.
5. Writing the container; the reconstruction for later layers is computed from the written fields.

## Sources

- [LTQ v1 specification](ltq-format.md) and experiments: [`ltq-research/`](https://github.com/lumen-inference/ltq-research)
- M. W. Marcellin, T. R. Fischer, "Trellis Coded Quantization of Memoryless and Gauss-Markov Sources", IEEE Trans. Commun., 1990
- G. Ungerboeck, "Channel Coding with Multilevel/Phase Signals", IEEE Trans. Inf. Theory, 1982
- [QTIP: Quantization with Trellises and Incoherence Processing](https://arxiv.org/abs/2406.11235) — tail-biting trellis, incoherence via Hadamard
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) — 1.6.3: the quality baseline; `modules/row_table.py`, `modules/ngram_embedding.py`, `modules/ple.py`, `modules/qsa_indexer.py`, `modules/hyperconnections.py` — reference for the architecture modules
- [Vemloir/Qwen3.8-Flash-Next-sc-exl3: an EXL3 quant of the same bit width — tensor sizes for the ceiling estimate, self-calibration data](https://huggingface.co/Vemloir/Qwen3.8-Flash-Next-sc-exl3)
- [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)
- [turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)
