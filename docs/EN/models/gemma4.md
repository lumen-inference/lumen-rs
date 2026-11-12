# `Gemma 4` Model Family

[Русский](../../RU/models/gemma4.md)

This is a reference for the architecture and its implementation requirements, not a report of working Velri execution. The C23 engine composes the model from [common blocks](../blocks.md); support for concrete operations and representations is checked through the [backend registry](../backends.md), starting with CPU and Vulkan. Terms are defined in the [glossary](../glossary.md).

The **Gemma 4** family comprises five models (`E2B`, `E4B`, `12B`, `26B-A4B`, `31B`) that alternate **sliding-window attention layers (`sliding`)** with **global full-attention layers (`full`)**.

| Model | Layers (sliding : full) | Sliding Window | Sliding KV | Global Layers: $K = V$ | Shared KV | PLE Module | Notable Features |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Gemma 4 E2B** | 35 (28 : 7) | 512 | $1 \times 256$ | No | Last 20 layers | 256 per layer | Double MLP on shared-KV layers; audio Conformer |
| **Gemma 4 E4B** | 42 (35 : 7) | 512 | $2 \times 256$ | No | Last 18 layers | 256 per layer | Audio Conformer + ViT |
| **Gemma 4 12B** | 48 (40 : 8) | 1024 | $8 \times 256$ | Yes ($1 \times 512$) | No | No | Encoder-free multimodal patch projection (`Gemma4Unified…`) |
| **Gemma 4 26B-A4B** | 30 (25 : 5) | 1024 | $8 \times 256$ | Yes ($2 \times 512$) | No | No | Dense MLP in parallel with MoE (128 experts, top-8) |
| **Gemma 4 31B** | 60 (50 : 10) | 1024 | $16 \times 256$ | Yes ($4 \times 512$) | No | No | 27-layer ViT |

Shared family parameters:
- On `sliding` layers: `head_dim = 256` and standard `RoPE` with $\theta = 10^4$.
- On global `full` layers: `head_dim = 512` and proportional `p-RoPE` ($p = 0.25$, $\theta = 10^6$).
- All models use `QK-norm` and attention scaling of $1.0$, pre- and post-norms on both attention and MLP, a per-layer `layer_scalar` multiplier at the end of each layer, input embeddings scaled by $\sqrt{\text{hidden}}$ (rounded to `bf16`), logit `softcap` of $30.0$, and tied embeddings with a 262,144-token vocabulary.

---

## 1. Key-From-Value Reconstruction on Global Layers ($K = V$)

On the global layers of `12B`, `26B-A4B`, and `31B` (`attention_k_eq_v: true`), there is no `v_proj` matrix: both keys ($K$) and values ($V$) are derived from the single `k_proj` projection:

$$\text{raw} = W_k \cdot x \quad \text{(per KV head, 512 dimensions)}$$
$$K = \text{RoPE}_p(\text{RMSNorm}(\text{raw}) \odot w_k) \quad \text{(norm with weight } k_{\text{norm}}\text{, followed by p-RoPE)}$$
$$V = \text{RMSNorm}(\text{raw}) \quad \text{(same norm without learned weight and without RoPE)}$$
$$\implies K = \text{RoPE}_p(V \odot w_k)$$

Because $K$ is uniquely determined by $V$, global layers can **store only $V$ and reconstruct $K$ when computing attention** (multiplying by $w_k$ and rotating 128 of the 512 dimensions). This halves global KV cache storage. Halving reads of stored K/V requires a kernel that derives K at consumption without writing and rereading a full temporary copy. This optimized path is not implemented. The graph explicitly describes the V resource and K derived from it using the norm weight and positions; a backend does not recognize the family by name or automatically allocate a separate K cache. If fused consumption is unsupported, preparation may choose an equivalent decomposition with temporary storage included in the budget; if no supported path exists, it returns an error. At a 262,144-token context length:

| Model | Global KV (Storing $K$ and $V$) | Global KV (Storing $V$ Only) | Sliding KV (Excluding $T_{\max}$ Margin) |
| :--- | :---: | :---: | :---: |
| **12B** | 4 GiB | **2 GiB** | 320 MiB |
| **26B-A4B** | 5 GiB | **2.5 GiB** | 200 MiB |
| **31B** | 20 GiB | **10 GiB** | 800 MiB |

---

## 2. Proportional RoPE (`p-RoPE`)

On global layers with head dimension $d = 512$ and rotary fraction $p = 0.25$, the number of active rotation angles is $0.25 \times 512 / 2 = 64$:
- Frequencies $\theta^{-2i / 512}$ are assigned to the first 64 angles ($i < 64$), while the remaining 192 angles are zero.
- Because `rotate-half` splits the full 512-dimensional head in half, the rotated coordinate slices are **`[0, 64)`** and **`[256, 320)`**.

---

## 3. Shared KV Cache on Final Layers (`Shared KV` in E2B and E4B)

The final `num_kv_shared_layers` layers (20 of 35 in `E2B` and 18 of 42 in `E4B`) have no `k_proj` or `v_proj` projections and allocate no KV cache of their own. Instead, they read the KV cache of the last reference layer of the same type (`sliding` or `full`) preceding the sharing boundary.

The plan represents these reads as references to one state resource with explicit dependencies on the reference layer's writes. The shared resource's lifetime and size are accounted for once; additional representations or transfers between devices are accounted for separately.

During prompt processing (prefill), these final layers write nothing to the KV cache. This permits considering evaluation only for rows that require output logits, after checking all dependencies and side effects. The optimization is not implemented; it belongs in [execution preparation](../architecture.md#what-is-done-ahead-of-time-and-at-each-step).

---

## 4. Per-Layer Embeddings (`PLE` in E2B and E4B)

- **Lookup table component:** the `embed_tokens_per_layer` tensor of shape $[262\,144, \text{layers} \times 256] \times \sqrt{256}$, indexed by token ID. In `E4B`, this table occupies ~5.6 GB in `bf16` (roughly one-third of all model weights). Once compressed with `LtqRing` rings, RAM or NVMe storage can be considered: preparation must provide row fetching, supported decoding, and delivery to the consumer. This does not promise direct GPU reads of RAM or hidden I/O latency.
- **Contextual component:** a linear projection of the main token embedding $\times \text{hidden}^{-1/2} \to \text{RMSNorm}$, combined with the table lookup scaled by $1/\sqrt{2}$.
- **Per-layer injection:** after the MLP on each layer, the model computes $\text{GELU}(W_{\text{gate}} h) \odot \text{ple}_\ell \to W_{\text{proj}} \to \text{RMSNorm} \to + \text{residual}$.

---

## 5. Parallel Dense MLP + MoE (`26B-A4B`) and `Gemma4Assistant` Drafter

- **Parallel MLP + MoE in `26B-A4B`:** on every layer, a dense MLP (`intermediate_size: 2112`) and a 128-expert MoE block (`top-8`, `intermediate_size: 704`) form independent branches from the same pre-MLP residual input. One possible placement runs the dense MLP on Vulkan and routed experts on CPU. Preparation checks support and adds transfers and the dependency for joining results; actual overlap and speedup require measurement.
- **Vision attention mask:** `sliding` layers apply the intersection of the sliding window with the union of the causal mask and bidirectional intra-image span mask; global layers remain strictly causal.
- **`Gemma4Assistant` drafter (`gemma-4-*-assistant`):** a 4-layer speculative drafter without its own `k_proj`/`v_proj` that reads the target model's embedding table and the KV caches of the target's last `sliding` and last `full` layers. Speculative execution is a separate stage in [roadmap.md](../roadmap.md).

---

## References

- Model configurations: [`E2B`](https://huggingface.co/google/gemma-4-E2B-it), [`E4B`](https://huggingface.co/google/gemma-4-E4B-it), [`12B`](https://huggingface.co/google/gemma-4-12B-it), [`26B-A4B`](https://huggingface.co/google/gemma-4-26B-A4B-it), [`31B`](https://huggingface.co/google/gemma-4-31B-it), [`12B assistant`](https://huggingface.co/google/gemma-4-12B-it-assistant)
- `transformers` implementation: [`modeling_gemma4.py`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4)
- *Gemma 4 Technical Report* ([arXiv:2607.02770](https://arxiv.org/pdf/2607.02770))
