# Model Architectures

Oct 27, 2026 · @Vemloir

What each family requires from the engine. Model quirks are operator parameters, not separate executor code ([section 8 of architecture.md](../architecture.md#8-extensibility)). A family with notable quirks has its own file:

- [qwen3_next.md](qwen3_next.md) — `qwen3_next`, `qwen3_5_dense` (`qwen3_5`), `qwen3_5_moe`
- [qwen4_exp.md](qwen4_exp.md) — `qwen4_exp` (Qwen3.8-Flash-Next)
- [gemma4.md](gemma4.md) — Gemma 4

## Families

| Family | What it requires from the engine |
| --- | --- |
| Llama, Qwen3 | the basic dense path |
| Qwen3-MoE | expert banks, softmax → top-k |
| `qwen3_next` | 3:1 hybrid Gated DeltaNet (fused `in_proj_qkvz`/`ba` grouped by K heads) + Gated Attention (`q_proj` ×2 with output σ-gate), zero-centered RMSNorm (`1 + w`), `RMSNormGated`, MoE with `shared_expert_gate`, partial 1D RoPE 0.25, MTP |
| `qwen3_5_dense` (`qwen3_5`) | 3:1 hybrid Gated DeltaNet (split `in_proj_{qkv,z,b,a}`) + Gated Attention, dense SwiGLU MLP in every layer, zero-centered RMSNorm, interleaved M-RoPE `[11, 11, 10]` (partial 0.25), ViT with axial 2D RoPE, MTP; e.g. Qwen3.8-27B: 64 layers (48 : 16), hidden 5120 |
| `qwen3_5_moe` | 3:1 hybrid Gated DeltaNet (split `in_proj_{qkv,z,b,a}`) + Gated Attention, MoE (router + routed experts + shared expert with `shared_expert_gate`), M-RoPE, ViT, MTP |
| gpt-oss | sinks, sliding/full alternation per `layer_types`, YaRN, SwiGLU with clamp 7.0, bias in projections, Harmony |
| Qwen3.8-Flash-Next (`qwen4_exp`) | built on `qwen3_next`/`qwen3_5_moe`: QSA in full layers, hyper-connections, n-gram PLE, MTP, M-RoPE, vision |
| Gemma 4 | K = V, p-RoPE, shared KV, PLE, dense MLP in parallel with MoE, encoder-free multimodality in 12B |

## Encoders

| Scheme | Models | What the encoder computes |
| --- | --- | --- |
| Encoder tower | `qwen3_5_dense`, `qwen3_5_moe`, Qwen3.8-Flash-Next: a 27-block ViT (axial 2D RoPE), hidden 1152, patch 16, 2×2 merger → decoder hidden, video with temporal patch 2. Gemma 4 26B and 31B: a 27-layer ViT, hidden 1152. Gemma 4 E2B/E4B: a 16-layer ViT; audio — a 12-layer conformer with chunked attention | bidirectional attention, no KV, once per image or audio clip; components `vision-tower`, `audio-tower` |
| Encoder-free | Gemma 4 12B (`Gemma4Unified…`) | the image is cut into 48×48 patches: LayerNorm → Linear 6912 → 3840 → LayerNorm → + factorized (x, y) position embeddings → LayerNorm → RMSNorm → Linear; up to 280 tokens per image. Audio — raw frames of 640 samples (40 ms at 16 kHz) → RMSNorm → Linear 640 → 3840, up to 750 tokens. The decoder itself does the rest |
