# `qwen4_exp` (Qwen3.8-Flash-Next)

Oct 27, 2026 · @Vemloir

| What | Config | What the engine needs |
| --- | --- | --- |
| 36 Gated DeltaNet layers and 12 full ones, every 4th is full | `layer_types`, `linear_*`, `full_attention_interval: 4` | a DeltaNet kernel, recurrent state ([section 3 of architecture.md](../architecture.md#3-memory-storage-locations-and-access-modes)) |
| QSA in full layers | `indexer_*`: 4 heads of 128, a 2048-token budget, 4-token blocks | the indexer leaves each query ≤ 512 blocks plus the incomplete tail; the selection is ANDed with the causal mask. Attention reads a bounded number of positions, so the KV of full layers at long context could stay in RAM with the selected blocks fetched — an offload candidate |
| Hyper-connections | `hc_count: 4`, `hc_lowrank: 320` | the residual is 4 streams; before a sub-block they are mixed into its input, the output is injected back (`block_inject_weight`), and the streams are collapsed before the final norm (`hyper_connection_mixer`) |
| N-gram PLE | `ple_*`, `ngram_*` | an on-disk `RowTable` + `Fetch`; the PLE layer: a key per stream and a shared value, a gate from the stream, a dilated causal conv (kernel 4, dilation 3) |
| MoE | 512 experts, top-10, intermediate 640; a 640-wide shared expert with a sigmoid gate | experts almost entirely on the CPU |
| MTP | 1 layer: attention, MoE, hyper-connections | the `Mtp` drafter, its own place in the plan |
| M-RoPE | `mrope_section: [11, 11, 10]`, interleaved, partial 0.25 of head\_dim 256 | \[3, T\] positions |
| Vision | a 27-block ViT | [section 7 of architecture.md](../architecture.md#7-multimodality) |

**The n-gram table.** In Qwen3.8-Flash-Next a position yields (ngram\_size − 1) × heads\_per\_ngram = 2 × 8 = 16 hashed rows of 160 values; together they form a 2560-wide vector. The table is 16 heads of ~20M rows each (head sizes are derived from `ngram_vocab_size_base`): 102 GB in bf16, ~52 GB in int8 (162 bytes per row with the scale), ~28 GB as TCQ4X rings (86 bytes per row).

## Sources

- [Qwen/Qwen3.8-Flash-Next: config.json, model.safetensors.index.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [transformers 5.16.0: Qwen4-Exp (GatedResidual, QSA, PLE)](https://newreleases.io/project/pypi/transformers/release/5.16.0)
