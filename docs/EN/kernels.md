# GPU and CPU Compute Kernels (lumen-cuda-kernels and lumen-cpu)

This document describes the architecture of the `lumen-rs` compute kernels on GPU (CUDA C++) and CPU (AVX2 / AVX-512), including the algorithms used for fast parallel decoding of the trellis-coded LTQ format.

---

## 1. Fast LTQ Trellis Decoding Without Bitwise Loops

Unlike independent lookup-table quantization formats, the subset index (`label`) of each weight in **LTQ** ([see specification](ltq-format.md)) depends on the bit parity of a **10-bit sliding window** of preceding code bits within the $16 \times 16$ tile. Evaluating this window weight-by-weight in a scalar loop requires 10–15 integer instructions per weight and bottlenecks execution on ALU throughput.

**Key Property Used by Kernels:**
Parity is linear over $GF(2)$ (reducing to `XOR`), and adjacent weights in a tile row share overlapping 10-bit history windows. Consequently, GPU and CPU kernels **never evaluate the history window one weight at a time**; instead, they assemble subset labels for 16–32 weights in parallel using small precomputed lookup tables in fast on-chip memory (`shared memory` on GPU and L1 cache on CPU) combined with vector `XOR` operations.

---

## 2. GPU Kernels (`lumen-cuda-kernels`)

On the GPU, custom CUDA C++ kernels decode LTQ tiles and execute model operators, **CUTLASS** provides dense GEMM during prefill and Tensor Core MMA primitives (CuTe), and **FlashInfer** provides the foundation for paged attention:

| Operator | Implementation | Kernel Design |
| :--- | :--- | :--- |
| **LTQ GEMV ($M \le 8$, Decode)** | Custom CUDA C++ | One warp per tile row band; input vector $\tilde{x}$ is loaded once per row group. Weight values are either read from a unified "10-bit window + uncoded bits $\to$ `fp16`" lookup table in `shared memory` (4–32 KB per format), or labels are computed via `__popc` over masks with byte selection via `__byte_perm` (`prmt`) and **`__dp4a`** integer dot products against `int8`-quantized activations. |
| **LTQ GEMM ($8 < M \le 32$)** | Custom CUDA C++ + CuTe MMA | Marlin-style pipeline decoding LTQ tiles into `shared memory` followed by Tensor Core MMA (`mma.sync`). |
| **LTQ Prefill ($M > 32$)** | Custom `reconstruct` + **CUTLASS** | Unpacks LTQ tiles into a temporary `fp16` scratch buffer in chunks of $M$ via `reconstruct`, then executes dense GEMM via CUTLASS. |
| **MoE (`mgemm`)** | Custom CUDA C++ | Single thread-block grid across all active experts via the `ExpertBank` descriptor table; tokens grouped by expert; supports distinct LTQ formats per expert. |
| **Hadamard-256 + `suh`** | Custom CUDA C++ | 8 elements per lane $\times$ 32 warp lanes; all butterfly stages run in registers via `__shfl_xor_sync` fused with `suh` scaling. |
| **Row Decode (`RowTable`)** | Custom CUDA C++ | Gathers table rows (from VRAM or over PCIe via `ZeroCopy`), decodes `LtqRing` rings (or `int8`) with ring scales, applies block Hadamard $H_B$ and signs. |
| **Attention (`Attention`)** | **FlashInfer** + custom variants | Paged attention over `KvPool`. Parameters: sliding window, attention sinks, mask, KV layout ($K$ and $V$, $V$-only with on-the-fly $K$ reconstruction, shared KV) and bit width, sigmoid output gate from `q_proj`, sparse page lists from the QSA indexer, `softcap`; uses `split-K` on decode. |
| **QSA Indexer (`IndexerScore`)** | Custom CUDA C++ | Computes per-head block scores $\text{ReLU}(q \cdot k)$, selects `top-k` blocks, and averages keys of completed 4-token blocks. |
| **Gated DeltaNet (GDN)** | Custom CUDA C++ | Chunked algorithm during prefill (modeled on `flash-linear-attention`), recurrent state update on decode (with $L_2$-normalized $Q$ and $K$), causal `short conv1d`, and gated output `RMSNormGated`. |
| **Hyper-Connections (`HcMix`, `HcInject`)** | Custom CUDA C++ | Mixes 4 residual streams before a sub-block, injects outputs back into the streams, and reduces streams before the final norm. |
| **Norms (`RMSNorm`, `1 + w`, `RMSNormGated`), `RoPE`, `SwiGLU`/`GeGLU`, `Router`, `Sample`** | Custom CUDA C++ | Elementwise ops and warp reductions fused with neighboring operators, plus Philox RNG sampling. |

### GPU Implementation Details
- **`RoPE` Variants (`RopeSpec`):** default; `partial` (Qwen) — rotates the first $r$ dimensions; proportional (`p-RoPE`, Gemma 4) — frequencies $\theta^{-2i/d}$ for the first $p \cdot d / 2$ angles at full $d$, remaining angles zero, with `rotate-half` across the full head (rotating slices $[0, p \cdot d / 2)$ and $[d/2, d/2 + p \cdot d / 2)$); `YaRN`; 3D `M-RoPE` partitioning angles into $(t, h, w)$ sections.
- **Attention inside CUDA Graphs:** because grid dimensions are baked into a captured graph while sequence lengths change every step, `split-K` partitioning parameters are computed on-device from `StepInputs`.
- **Programmatic Dependent Launch (PDL):** on `sm_90+` GPUs, launching via `cudaLaunchKernelEx` with PDL allows the next kernel to begin its prologue while the previous kernel finishes writing its final blocks.

---

## 3. CPU Kernels (`lumen-cpu`)

The performance target for the CPU backend is the measured DRAM memory bandwidth ([performance.md](performance.md)). The primary target platform is **AVX-512 (VNNI and VBMI)** (AMD Zen 4/5 and server Intel CPUs), with a full **AVX2** fallback path.

### CPU LTQ Kernel Algorithm

1. **Per-Token Input Preparation:**
   The input vector is zero-padded to $K_{\text{pad}}$, multiplied by `suh`, transformed via block Hadamard-256, and quantized to `int8` in 256-element blocks with `fp32` scales (activation `int8` quantization introduces ~0.9% RMS error per call).
2. **Horizontal Band Traversal (`CpuBand16`):**
   The kernel traverses a 16-row horizontal tile band along the $K$ axis while keeping accumulators in vector registers. The `CpuBand16` layout ensures all tiles in the band are stored contiguously in DRAM.
3. **Decoding and Multiplying 32 Weights (Two Tile Rows) per Step:**
   - **Label assembly (`label`):** labels for 16 weights in a row are computed by bitwise `XOR` (`vpxor`) of three 16-byte vectors fetched from precomputed L1 tables (16 KB per format) indexed by the `codes` bytes.
   - **Level index assembly:** the label is combined with uncoded bits (`planes` and `lsb`; in `D2` and `D4` modes the lowest bit is reconstructed via byte shifts and masks).
   - **`int8` codebook lookup (`lut`):** executed via `vpshufb` on AVX2 (using 2 or 4 `vpshufb` + `vpblendvb` for 32- or 64-entry tables) or a single `vpermb` instruction on AVX-512 VBMI.
   - **Dot product accumulation:** multiplies `int8` weights with the `int8` input via `vpmaddubsw` / `vpdpbusd` (transferring signs via `vpsignb`) and scales the 16-element row sum by $\text{gain} \times d$ in `fp32`.

### Measured CPU Kernel Throughput

In single-thread Zen 5 benchmarks with data resident in L2 cache (AVX2 path, [`ltq-research/src/cpu/tcq_avx2_bench.c`](https://github.com/lumen-inference/ltq-research/blob/main/src/cpu/tcq_avx2_bench.c)), decoding and multiplication throughput reaches:
- **25.6 billion weights/s** at $K = 2$;
- **25.7 billion weights/s** at $K = 3$;
- **23.3 billion weights/s** at $K = 4$;
- **18.0 billion weights/s** at $K = 5$ (fractional `D2`/`D4` modes are 12–14% slower).

For comparison, on the same testbed GGUF's table-based `IQ3_K` achieves 35 billion weights/s and `IQ5_K` / `IQ6_K` achieves 20–21 billion weights/s. Across all cores on target CPUs, the kernel is strictly DRAM-bandwidth bound for all formats of 3 bits/weight and above.

---

## 4. Kernel Verification

1. **Bit-exact `lumen-ltq` verification:** the Rust reference decoder in `lumen-ltq` is tested bit-for-bit against the Python reference [`ltq_format.py`](https://github.com/lumen-inference/ltq-research/blob/main/src/ltq_format.py) across golden fixtures covering all 19 menu formats and ring tables.
2. **Per-kernel tests:** every SIMD and CUDA kernel is verified against `lumen-ltq` across randomized matrix shapes.
3. **End-to-end model differential tests:** full model graphs are verified against the scalar `lumen-ref` oracle ([roadmap.md](roadmap.md)).
