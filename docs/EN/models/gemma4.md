# Gemma 4

Oct 27, 2026 · @Vemloir

| Model | Layers (sliding : full) | Window | Sliding KV | Global: K = V | Shared KV | PLE | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2B | 35 (28 : 7) | 512 | 1 × 256 | no | last 20 | 256 per layer | double-wide MLP in shared-KV layers; audio conformer |
| E4B | 42 (35 : 7) | 512 | 2 × 256 | no | last 18 | 256 per layer | audio conformer |
| 12B | 48 (40 : 8) | 1024 | 8 × 256 | yes, 1 head × 512 | no | no | encoder-free (`Gemma4Unified…`) |
| 26B-A4B | 30 (25 : 5) | 1024 | 8 × 256 | yes, 2 × 512 | no | no | dense MLP in parallel with MoE (128 experts, top-8) |
| 31B | 60 (50 : 10) | 1024 | 16 × 256 | yes, 4 × 512 | no | no | — |

Common to the family: sliding layers use head\_dim 256 and RoPE with θ = 10⁴, global layers use head\_dim 512 and p-RoPE (p = 0.25, θ = 10⁶); QK-norm and attention scaling 1.0; norms before and after attention and MLP; `layer_scalar` multiplies the entire hidden state at the end of a layer; embeddings × √hidden (the factor is rounded to bf16); logit softcap 30; tied embeddings, a 262,144-token vocabulary.

**How V depends on K.** Global layers of 12B, 26B, and 31B (`attention_k_eq_v: true`) have no `v_proj`: K and V come from one projection.

```
raw = W_k · x                         per KV head, 512 dimensions
K   = RoPE_p(RMSNorm(raw) ⊙ w_k)      norm with the k_norm weight, then p-RoPE
V   = RMSNorm(raw)                    the same norm, without weight and without RoPE
⇒ K = RoPE_p(V ⊙ w_k)
```

The cache therefore stores only V, and the attention kernel reconstructs K on the fly: a multiplication by the vector w\_k and a rotation of 128 of the 512 dimensions. The global KV is half the size, and decode reads half the bytes. The only difference from transformers, which caches K and V separately, is bf16 rounding. At a 262,144-token context:

| Model | Global KV: K and V | V only | Sliding KV (without the T\_max margin) |
| --- | --- | --- | --- |
| 12B | 4 GiB | 2 GiB | 320 MiB |
| 26B-A4B | 5 GiB | 2.5 GiB | 200 MiB |
| 31B | 20 GiB | 10 GiB | 800 MiB |

**p-RoPE.** rope\_angles = 0.25 × 512 / 2 = 64; frequencies θ^(−2i/512) for i < 64, the remaining 192 angles are zero; rotate-half spans the whole head, so dimensions \[0, 64) and \[256, 320) rotate, not the first 128.

**Shared KV** (E2B/E4B). The last `num_kv_shared_layers` layers have no k/v projections and read the cache of the last layer of the same type (sliding or full) before the sharing point. They have no cache of their own, and during prefill they are needed only for rows with logits ([section 5 of architecture.md](../architecture.md#5-scheduler-and-execution-pipeline)).

**PLE** (E2B/E4B). The table part is `embed_tokens_per_layer` \[262,144, layers × 256\] × √256, looked up by id: in E4B that is 262,144 × 42 × 256 × bf16 ≈ 5.6 GB, about a third of all weights — a candidate for `disk` or `ram`. The context part is a projection of the main embedding × hidden^(−½) → RMSNorm. The result is (context + lookup) / √2. In every layer after the MLP: gelu(W\_gate h) ⊙ ple\_ℓ → W\_proj → RMSNorm → + residual.

**MoE in 26B-A4B.** In every layer a dense MLP (intermediate 2112) and 128 experts (top-8, intermediate 704) run in parallel; the router looks at the residual before the MLP, so the experts do not wait for the dense MLP. The dense MLP is natural GPU work while the CPU computes the experts.

**Vision mask**: in sliding layers — AND(window, OR(causal, within the image)); global layers — causal only.

**Assistant drafter** (`gemma-4-*-assistant`, the `Gemma4Assistant` drafter — [section 6 of architecture.md](../architecture.md#6-runtime-generation-core-and-speculative-decoding)). 4 layers without k/v projections: attention reads the cache of the target's last sliding and last full layer, and the drafter has no KV of its own. Input — `pre_projection` of the concatenation of the token embedding (from the target's table) and the target's last hidden state; the output goes through `post_projection` back to the target's size and serves as the hidden state for the next draft step. Optionally a centroid head (2048 clusters, top-32) instead of the full lm\_head (`use_ordered_embeddings`; off in 12B)

## Sources

- Gemma 4 configs: [E2B](https://huggingface.co/google/gemma-4-E2B-it), [E4B](https://huggingface.co/google/gemma-4-E4B-it), [12B](https://huggingface.co/google/gemma-4-12B-it), [26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it), [31B](https://huggingface.co/google/gemma-4-31B-it), [12B assistant](https://huggingface.co/google/gemma-4-12B-it-assistant)
- [transformers: modeling\_gemma4.py, modeling\_gemma4\_unified.py, modeling\_gemma4\_unified\_assistant.py, modeling\_rope\_utils.py](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4)
- [Gemma 4 Technical Report](https://arxiv.org/pdf/2607.02770)
- [Sebastian Raschka: Gemma 4 Architecture and Benchmark Notes](https://www.sebastianraschka.com/blog/2026/gemma-4-release-notes.html)
- [Maarten Grootendorst: A Visual Guide to Gemma 4](https://newsletter.maartengrootendorst.com/i/193064129/kv)
