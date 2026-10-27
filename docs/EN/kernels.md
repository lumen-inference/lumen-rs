# lumen-rs — Kernels on GPU and CPU

Oct 27, 2026 · @Vemloir

On the GPU, our CUDA C++ kernels decode LTQ tiles; dense GEMM goes to CUTLASS and attention to FlashInfer. LTQ decode costs more than table formats: a weight's label depends on the 10 previous coded bits. But the label is a linear function of the window over GF(2), and neighbouring weights share the window, so kernels build labels for a whole group of weights from small tables instead of computing them one by one. Below is the kernel design and what the prototypes have already measured ([`ltq-research/src/cpu/tcq_avx2_bench.c`](https://github.com/lumen-inference/ltq-research/blob/main/src/cpu/tcq_avx2_bench.c), [`ltq-research/src/cuda/ltq_matvec_bench.cu`](https://github.com/lumen-inference/ltq-research/blob/main/src/cuda/ltq_matvec_bench.cu)).

**GPU**

| Operator | Implementation | Kernel design |
| --- | --- | --- |
| LTQ GEMV, M ≤ 8 (decode) | own kernel | a warp per tile row group, a tile's x̃ is loaded once for R rows. A weight's value is one read from a shared "10-bit window + uncoded bits → fp16" table in shared memory (4–32 KB per format), or a label via `popc` over the masks plus a read of `lut` replicated across banks; × gain × d; dot product with x̃ |
| LTQ GEMM, 8 < M ≤ 32 | own kernel, MMA via CuTe | Marlin-style pipeline: tiles decoded into shared memory, then MMA |
| LTQ, M > 32 (prefill) | own `reconstruct` + CUTLASS | `reconstruct` tiles into an fp16 scratch in chunks along M, then a CUTLASS GEMM |
| MoE (mgemm) | own kernel | one grid over all active experts via the descriptor table; tokens grouped by expert; each expert has its own format |
| Hadamard-256 | own kernel | 8 elements per lane × 32 lanes, butterflies via warp shuffle; fused with ⊙suh |
| Row decoding | own kernel | gather of table rows, decode of LTQ rings (or int8) with the ring scale, inverse block Hadamard of size B, ⊙ signs |
| Attention | FlashInfer + own variants | flash-attention over the paged KV; parameters: window, sinks, mask, KV layout (K and V, V only, another layer's cache) and bit width, output σ-gate from `q_proj` (`qwen3_next`, `qwen3_5_*`), block list from the indexer, scaling, softcap; decode — split-K |
| QSA indexer | own kernel | block scoring relu(q·k) over heads, top-k blocks, pooling of completed blocks |
| Gated DeltaNet | own kernel (reference: flash-linear-attention) | chunked algorithm for prefill, recurrent step for decode (with Q and K L2-norm), causal short conv1d, output `RMSNormGated` (`norm(y) · silu(z)` per V head) |
| Hyper-connections | own kernel | stream mixing before a sub-block, output injection, stream collapse before the final norm |
| RMSNorm (standard, `1 + w`, and `RMSNormGated`), RoPE, SwiGLU/GeGLU (incl. gpt-oss clamp), residual, softcap | own kernels | elementwise operations and reductions fused with neighbours |
| Router + top-k, sampling | own kernel | warp reductions, Philox |

- **RoPE** is described by a `RopeSpec`: default; partial (Qwen) — the first r dimensions rotate with frequencies θ^(−2i/r); proportional (p-RoPE, Gemma 4) — frequencies θ^(−2i/d) for the first p·d/2 angles at full d, the remaining angles are zero, and rotate-half spans the whole head, so dimensions \[0, p·d/2) and \[d/2, d/2 + p·d/2) rotate; YaRN; M-RoPE — angles are split into (t, h, w) sections.
- **Attention inside a CUDA Graph.** The grid size is baked into the graph while lengths change every step, so the split-K partitioning is computed on the device from the lengths in `StepInputs`. FlashInfer's paged decode takes a page list per request, so the indexer's block list is passed in that form; what it lacks (4-token QSA blocks inside 64-token pages, K = V, quantized KV) is checked by a prototype at the start of phase 4, before GEMV.
- **LTQ decode cost.** The minimum is about 5–6 integer operations and one shared-memory read per weight. A GEMV prototype on fp32 FMA (RTX PRO 6000) is decode-bound: 0.45–0.72 of the ceiling of the same kernel for a 4-bit table format. Bank conflicts on the shared table turned out not to be critical. To be VRAM-bound rather than decode-bound at batch 1 on a 5070 Ti we need int8 activations with `dp4a`, value lookup via `prmt`, and labels for 32 weights at once from bit planes. This is the first task of phase 4; `__dp4a` and `__byte_perm` (`prmt`) are plain CUDA intrinsics.
- PDL (programmatic dependent launch via `cudaLaunchKernelEx`, sm\_90+) shortens the gaps between consecutive decode kernels: the next kernel starts its prologue while the previous one finishes.
- The floor is sm\_80 (Ampere): bf16 MMA and `cp.async`. The kernel library carries SASS for every supported architecture, so the driver JIT-compiles PTX only on GPUs newer than the build.

**CPU** — the speed reference is the measured DRAM ceiling ([performance.md](performance.md)). The main target is Zen 4/5 (AVX-512 VNNI and VBMI); Intel without AVX-512 and Zen 2 take the AVX2 path.

1. Input per token row: zero-pad to K\_pad, ⊙suh, block Hadamard-256, then quantize to int8 in blocks of 256 with an fp32 scale.
2. The kernel walks K in a band of 16 rows (a tile row group) with accumulators in registers. Per 32 weights (two tile rows):
   - labels — XOR of three 16-byte vectors from tables indexed by bytes of coded bits (9 bits of history + 2 bytes of the row, 16 KB of tables per format, resident in L1); the label bits land directly in their places in the level index;
   - level index = label | uncoded bits (shift and mask of the planes; for D2 and D4 the low bit is restored with a byte shift and a mask);
   - the int8 value is a `vpshufb` over the level table (16 entries; 32 or 64 entries — 2 or 4 `vpshufb` + `vpblendvb`, on VBMI a single `vpermb`);
   - `vpmaddubsw` / `vpdpbusd` with the int8 input (the sign is moved onto the input with `vpsignb`), the sum of a row's 16 weights is scaled by gain × d in fp32.
3. Output: + bias.

- **Measured** (AVX2, one Zen 5 thread, data in L2): 25.6 billion weights/s at K = 2, 25.7 at K = 3, 23.3 at K = 4, 18.0 at K = 5; fractional modes are 12–14 % slower. For comparison: `IQ3_K` 35, `IQ5_K` / `IQ6_K` 20–21 on the same rig. By instruction count the kernel spends ~290 operations per 256 weights against a theoretical minimum of ~200 for this format.
- **What this means for target CPUs** (6-core Skylake / Zen 2, DDR4 ~40 GB/s): at 3+ bits per weight the kernel is memory-bound, like GGUF. At 2–2.6 bits per weight it is estimated to be compute-bound: up to ~25 % tokens/s loss on a 2.5-bits/weight model with the current kernel, up to ~7 % at the minimum. Zen 4/5 with AVX-512 has more headroom. To be checked on real CPUs in phase 3.
- int8 activation accuracy is about 0.9% RMS of the output per call. Table levels are int8 with a per-format scale; the effect of rounding on KLD is checked by differential tests against `lumen-ref`.
- Dispatch: `Isa::{Avx2, Avx512Bw, Vnni, Vbmi}` is detected once at startup; kernels are `#[target_feature]` functions specialized by the constants `K`, the number of subsets and the fractional mode, in a pointer table \[ISA\]\[format\].
- Work is split statically: in MoE the unit is an (expert, 16-row band) pair over active experts. Small experts (Qwen3.8-Flash-Next: 640 × 2560) produce many units, so the cost of handing out work is critical — which is what the spin pool removes.
- M > 4 runs in chunks of 4 rows. A large prefill is better handed to the GPU — `prefill-compute` in [section 5 of architecture.md](architecture.md#5-scheduler-and-execution-pipeline).

**Correctness.** The `lumen-ltq` reference is checked bit for bit against [`ltq-research/src/ltq_format.py`](https://github.com/lumen-inference/ltq-research/blob/main/src/ltq_format.py) on golden fixtures: tiles of every menu format produced by the real quantizer, and whole matrices with scales and rotation. CPU and GPU kernels are checked against the reference on random shapes. Whole models are checked through `lumen-ref` ([roadmap.md](roadmap.md)).

## Sources

- [NVIDIA CUTLASS and CuTe](https://github.com/NVIDIA/cutlass)
- [FlashInfer — attention kernels with paged KV](https://github.com/flashinfer-ai/flashinfer)
- cuda-oxide against CUDA C++ on the LTQ GEMV: [`results/kbench_gemv_g4.json`](https://github.com/lumen-inference/ltq-research/blob/main/results/kbench_gemv_g4.json), [`src/kbench/`](https://github.com/lumen-inference/ltq-research/tree/main/src/kbench)
