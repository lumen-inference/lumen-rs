# Model architecture reference

[Русский](../../RU/models/README.md)

Start with the [common block overview](../blocks.md) to see how these mechanisms compose a model. This reference covers differences between families and the parameters needed to implement them. Common terms are defined in the [glossary](../glossary.md).

These are family references and implementation requirements, not a list of working models. The new C17 engine composes models from common blocks; CPU and Vulkan are its first planned compute backends. [blocks.md](../blocks.md) and [architecture.md](../architecture.md) define composition and ownership; [backends.md](../backends.md) defines the device boundary. Module names below describe their meaning, not public C types.

A family describes layer order and shared-block parameters. QSA is not a universal sparse-attention algorithm; another algorithm needs its own composition of operations. State sizes and memory access depend on model parameters and device capabilities.

Detailed specifications for families with specialized modules:

- **[qwen3_next.md](qwen3_next.md)** — hybrid `qwen3_next`, `qwen3_5_dense` (`qwen3_5`), and `qwen3_5_moe` family (3:1 interleaving of `Gated DeltaNet` linear attention and gated full attention).
- **[qwen4_exp.md](qwen4_exp.md)** — `qwen4_exp` architecture (`Qwen3.8-Flash-Next`: `QSA` sparse attention, 4-stream `Hyper-Connections`, and n-gram `PLE` table).
- **[gemma4.md](gemma4.md)** — `Gemma 4` family ($K = V$ reconstruction, proportional `p-RoPE`, shared KV cache on final layers, `PLE`, and parallel Dense MLP + MoE).

---

## 1. Core Architectural Modules

Layers of transformer and hybrid models are composed from the following standard modules:

| Category | Module | Role | Used By |
| :--- | :--- | :--- | :--- |
| **Sequence Mixers** | **`FullAttention` / `SlidingAttention`** | Causal attention over the full context or a sliding window of the last $W$ tokens. State resources and representations, including paging, are explicit. | All model families |
| | **`SparseAttention` (QSA)** | Sparse attention with block indexer: a lightweight indexer scores 4-token blocks and selects $\le 512$ relevant blocks. Keeping the main cache in RAM requires a supported gather and transfer path. | `Qwen3.8-Flash-Next` (`qwen4_exp`) |
| | **`GatedDeltaNet` (GDN)** | Linear recurrent attention maintaining a state matrix of fixed size for a given configuration (for example, ~3 MiB per layer), updated in $O(1)$ with respect to context length per token instead of a growing KV cache. | `qwen3_next`, `qwen3_5_dense`, `qwen3_5_moe`, `Qwen3.8-Flash-Next` |
| **Channel Mixers (FFN)** | **`DenseMlp`** | Dense feed-forward block (`SwiGLU` or `GeGLU`). | `Llama`, `Qwen3`, `qwen3_5_dense`, `Gemma 4` |
| | **`MoE` (Mixture of Experts)** | Mixture of Experts: a router selects `top-k` experts from a weight bank, plus an optional always-active `Shared Expert`. CPU execution is one placement option considered during preparation. | `Qwen3-MoE`, `qwen3_next`, `qwen3_5_moe`, `Qwen3.8-Flash-Next`, `Gemma 4 26B-A4B`, `gpt-oss` |
| **Adapters & Residuals** | **`PLE` (Per-Layer Embeddings)** | Per-layer embedding or n-gram lookup table indexed by token IDs. NVMe storage requires explicit row fetching, decoding, and delivery to the consumer. | `Gemma 4 E2B/E4B`, `Qwen3.8-Flash-Next` |
| | **`HyperConnections`** | Multi-stream residual connection (4 parallel `residual` streams mixed before each sub-block). | `Qwen3.8-Flash-Next` |

Preparation checks operations, shapes, types, layouts, and state changes through the backend registry. K and V are input roles: shared storage or K derived from V must appear in the graph, without automatically allocating a pair of caches. Transfers and representation conversions belong in the plan. An unsupported path is rejected before execution or placed on another allowed device during preparation; there is no silent CPU fallback. Adding a backend for these operations does not require changing family descriptions.

---

## 2. Model Family Summary

| Family | Required Modules and Parameters |
| :--- | :--- |
| **`Llama`, `Qwen3`** | Standard dense path: `FullAttention` + `DenseMlp` (`SwiGLU`) + `RMSNorm` + `RoPE`. |
| **`Qwen3-MoE`** | `FullAttention` + `MoE` expert bank (`softmax → top-k` routing). |
| **`qwen3_next`** | 3:1 hybrid of `GatedDeltaNet` (fused `in_proj_qkvz`/`ba` projections packed by $K$-groups) + `Gated Attention` (double-width `q_proj` with output $\sigma$-gate), `zero-centered RMSNorm` (`1 + w`), `RMSNormGated`, `MoE` with `shared_expert_gate`, partial 1D RoPE 0.25, `MTP`. |
| **`qwen3_5_dense`** (`qwen3_5`) | 3:1 hybrid of `GatedDeltaNet` (split `in_proj_{qkv,z,b,a}`) + `Gated Attention`, dense `DenseMlp` on every layer, `zero-centered RMSNorm`, interleaved 3D `M-RoPE` `[11, 11, 10]` (partial 0.25), `ViT` with axial 2D RoPE, `MTP` (e.g. `Qwen3.8-27B`: 64 layers 48:16, hidden 5120). |
| **`qwen3_5_moe`** | 3:1 hybrid of `GatedDeltaNet` (split projections) + `Gated Attention`, `MoE` (routed experts + shared expert with `shared_expert_gate`), `M-RoPE`, `ViT`, `MTP`. |
| **`gpt-oss`** | Learned attention sinks, `sliding`/`full` alternation per `layer_types`, `YaRN`, `SwiGLU` with 7.0 `clamp`, projection `bias`, `Harmony` chat format. |
| **`Qwen3.8-Flash-Next`** (`qwen4_exp`) | Extension of `qwen3_next` / `qwen3_5_moe`: `QSA` sparse attention in full layers, `Hyper-Connections`, n-gram `PLE` table, `MTP`, `M-RoPE`, `ViT`. |
| **`Gemma 4`** | $K = V$ reconstruction, proportional `p-RoPE`, shared KV cache on final layers, `PLE`, parallel Dense MLP + MoE (on `26B-A4B`), encoder-free multimodality on `12B`. |

---

## 3. Multimodal Encoders

| Architecture | Models | Encoder Design |
| :--- | :--- | :--- |
| **Dedicated Encoder Tower (`vision-tower`, `audio-tower`)** | `qwen3_5_dense`, `qwen3_5_moe`, `Qwen3.8-Flash-Next`: 27-block ViT (axial 2D RoPE), hidden 1152, patch 16, $2 \times 2$ merger $\to$ decoder hidden, video temporal patch 2.<br/>`Gemma 4 26B / 31B`: 27-layer ViT, hidden 1152.<br/>`Gemma 4 E2B / E4B`: 16-layer ViT; audio — 12-layer Conformer with chunked attention. | Bidirectional attention without a KV cache; executed once per image or audio clip with outputs cached by content hash. |
| **Direct Patch Projection (Encoder-Free)** | `Gemma 4 12B` (`Gemma4Unified…`) | Images are sliced into $48 \times 48$ patches: `LayerNorm → Linear 6912 → 3840 → LayerNorm → + factorized (x, y) pos embeddings → LayerNorm → RMSNorm → Linear` (up to 280 tokens per image). Audio uses 640-sample frames (40 ms at 16 kHz) `→ RMSNorm → Linear 640 → 3840` (up to 750 tokens). All subsequent processing occurs inside the main decoder. |
