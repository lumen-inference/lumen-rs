# `qwen4_exp` Architecture (`Qwen3.8-Flash-Next`)

[Русский](../../RU/models/qwen4_exp.md)

These are family implementation requirements, not a description of a working execution path. The C17 engine composes it from [common blocks](../blocks.md); the [backend registry](../backends.md) determines support for concrete operations and representations, starting with CPU and Vulkan. Terms are defined in the [glossary](../glossary.md).

The **`qwen4_exp`** architecture (`Qwen3.8-Flash-Next`) extends the hybrid [`qwen3_next` / `qwen3_5_moe`](qwen3_next.md) foundation with three additional modules: Query-Sparse Attention (`QSA`), four-stream residual connections (`Hyper-Connections`), and n-gram Per-Layer Embeddings (`PLE`).

---

## 1. Summary of Parameters and Modules

| Component | `config.json` Parameters | Engine Requirements |
| :--- | :--- | :--- |
| **3 : 1 Hybrid Backbone** | 48 layers: 36 `GatedDeltaNet` layers and 12 `full_attention` layers (`full_attention_interval: 4`) | `DeltaNet` kernel and session-owned recurrent state ([storage ownership](../architecture.md#how-memory-is-organized)) |
| **Query-Sparse Attention (`QSA`) in full layers** | `indexer_*`: 4 heads of dim 128, 2048-token budget, 4-token blocks | The indexer selects $\le 512$ blocks plus the incomplete tail per query, intersected with the causal mask. Keeping the main cache in RAM requires supported block gathering and transfer into a representation accepted by attention; direct GPU access to RAM is optional |
| **Multi-Stream `Hyper-Connections`** | `hc_count: 4`, `hc_lowrank: 320` | The `residual` state consists of 4 parallel streams; streams are mixed into a single input vector before each sub-block (`HcMix`), the sub-block output is injected back via `block_inject_weight` (`HcInject`), and all streams are reduced before the final norm (`hyper_connection_mixer`) |
| **N-Gram `PLE` Module** | `ple_*`, `ngram_*` | A row table that permits NVMe storage (`store: "disk"`) with explicit fetching, decoding, and transfer of rows; within each layer — per-stream key projection, shared value projection, gate, and dilated causal convolution (kernel 4, dilation 3) |
| **Mixture of Experts (`MoE`)** | 512 routed experts (`top-10`, `intermediate_size: 640`) + 1 shared expert (640) with sigmoid gate | One placement option keeps routed expert weights in RAM and computes on CPU; selection depends on allowed devices, format support, and the memory budget |
| **`MTP` Draft Layer** | 1 additional layer (attention, MoE, and Hyper-Connections) | A separately described speculative drafter with explicit state and placement; execution belongs to the speculative stage ([roadmap.md](../roadmap.md)) |
| **`M-RoPE` Positional Encoding** | `mrope_section: [11, 11, 10]`, interleaved, `partial 0.25` of `head_dim = 256` | 3D position coordinates $[3, T]$ |
| **Vision Encoder** | 27-block ViT | Separate encoder program; integration belongs to the multimodal stage ([roadmap.md](../roadmap.md)) |

QSA is a composition of indexer and attention operations with explicit selected blocks, not a model-name switch inside a backend. Ordinary attention support does not imply QSA support. If the required operations, representations, or transfers are unavailable, preparation chooses another allowed path or reports an error before execution; it does not silently substitute an algorithm or device.

---

## 2. N-Gram Lookup Table Structure and Disk Offloading

At each token position, the model computes $(\text{ngram\_size} - 1) \times \text{heads\_per\_ngram} = 2 \times 8 = 16$ hash indices (8 bigram heads and 8 trigram heads). Each head looks up a 160-element row, and the 16 rows concatenate into a vector of dimension $16 \times 160 = 2560$.

The full table comprises 16 partitions of ~20 million rows each (exact partition sizes are derived from `ngram_vocab_size_base`):
- in `bf16`, the table occupies **102 GB**;
- in `int8` (162 bytes per row with scale) — **~52 GB**;
- in `TCQ4X` LTQ rings (86 bytes per row) — **~28 GB**.

The row payload per generated token is $16 \times 86 = 1376\text{ bytes}$ ($\le 1.4\text{ KB}$), and all 16 row indices depend solely on token IDs known immediately after sampling. This makes NVMe storage and background row prefetching options to consider. Actual I/O volume depends on read granularity and caching; keeping I/O off the critical path is not guaranteed. The plan must account for fetch buffers, row decoding, delivery to the consumer, and waiting for readiness. Disk is a data source, not a compute backend.

---

## References

- Model configuration: [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- Reference implementations of Qwen4-Exp modules (`GatedResidual`, `QSA`, `PLE`) in `transformers 5.16.0` and `exllamav3` (`modules/qsa_indexer.py`, `modules/hyperconnections.py`, `modules/ngram_embedding.py`, `modules/ple.py`).
