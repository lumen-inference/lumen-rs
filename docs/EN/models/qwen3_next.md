# Hybrid Qwen Family: `qwen3_next`, `qwen3_5_dense`, and `qwen3_5_moe`

[Русский](../../RU/models/qwen3_next.md)

These are family implementation requirements, not a list of runnable models. In the C23 engine, the family defines the order and parameters of [common blocks](../blocks.md). Preparation checks operations through the [backend registry](../backends.md), starting with CPU and Vulkan. Terms are defined in the [glossary](../glossary.md).

Models in this family follow a **3 : 1 hybrid schedule** defined by the `layer_types` array in the model configuration:
- Three consecutive layers use linear recurrent attention **`GatedDeltaNet` (GDN)** (`linear_attention`), which requires no growing KV cache.
- Every fourth layer uses gated full attention **`Gated Attention`** (`full_attention`).

This structure reduces paged KV cache memory by 4× while retaining global full attention on every fourth layer.

---

## 1. `GatedDeltaNet` Linear Attention Module

Instead of accumulating keys and values for all past tokens, `GatedDeltaNet` maintains a **fixed-size recurrent state matrix** $S$ of shape $[H_v, D_k, D_v]$ in `fp32` on each layer (for example, $48 \times 128 \times 128 \times 4\text{ bytes} = 3\text{ MiB}$ per layer) along with a short convolution state of `conv_kernel - 1` positions.

Both resources belong to the session and are explicit in the plan, including reads and updates. A backend receives operations and parameters, not the family name. Linear support alone does not imply support for convolution or the delta rule: unsupported paths are rejected during preparation or placed on another allowed device in advance, with explicit transfers and state representation conversions.

Computation steps inside `GatedDeltaNet`:
1. **Input projections:** computes $q, k, v$, the output gate $z$, and per-head update scalars $b, a$.
2. **Causal short convolution (`causal conv1d`):** applies a width-4 causal sliding convolution followed by `SiLU` activation across $[q, k, v]$.
3. **Delta-rule state update:** normalizes $Q$ and $K$ in $L_2$; decays the state $S$ by $\exp(g)$ where $g = -\exp(A_{\text{log}}) \cdot \text{softplus}(a + \text{dt\_bias})$, updates $S$ via the delta learning rule with step size $\beta = \sigma(b)$, and contracts the updated state with the query $Q$.
4. **Gated output norm and projection:** applies `RMSNormGated(head_v_dim)` multiplied by $\text{SiLU}(z)$ per V-head, followed by `out_proj`.

---

## 2. `Gated Attention` Full Attention Module

Every 4th layer executes paged full attention with a sigmoid output gate:
- The `q_proj` projection has doubled output width $2 \cdot H \cdot D$ and splits within each head into the query vector $q$ and the gate vector $\text{gate}$.
- Per-head normalization (`q_norm`, `k_norm`) and partial `RoPE` (`partial_rotary_factor: 0.25`) are applied to $q$ and $k$.
- The attention output is multiplied elementwise by $\sigma(\text{gate})$ before `o_proj`.

---

## 3. Subfamily Comparison

| Layer Component | `qwen3_next` | `qwen3_5_dense` (`qwen3_5`) | `qwen3_5_moe` |
| :--- | :--- | :--- | :--- |
| **`GatedDeltaNet` Input** | 2 fused projections: `in_proj_qkvz` and `in_proj_ba`, packed by `num_k_heads` groups (`fix_query_key_value_ordering`) | 4 split projections: `in_proj_qkv`, `in_proj_z` (LTQ quantized) and narrow dense `in_proj_b`, `in_proj_a` of width `num_v_heads` | Same as `qwen3_5_dense` (4 split projections) |
| **`GatedDeltaNet` Core** | `causal conv1d` (window 4, SiLU) $\to$ delta rule with $L_2$-normed $Q, K$, $\beta = \sigma(b)$, $g = -\exp(A_{\text{log}}) \cdot \text{softplus}(a + \text{dt\_bias})$ $\to$ `RMSNormGated(head_v_dim)` gated by $\text{SiLU}(z)$ $\to$ `out_proj` | Same | Same |
| **`full_attention`** | **Gated Attention**: `q_proj` of width $2 \cdot H \cdot D$ split into $(q, \text{gate})$; `q_norm` & `k_norm` over `head_dim`; output multiplied by $\sigma(\text{gate})$ before `o_proj` | Same | Same |
| **Positional Encoding (`RoPE`)** | Partial 1D RoPE (`partial_rotary_factor: 0.25`) | Interleaved 3D **M-RoPE** `mrope_section: [11, 11, 10]`, `partial 0.25` | Same as `qwen3_5_dense` |
| **FFN / MoE Block** | **MoE**: `routed-experts` + $\text{shared\_expert} \cdot \sigma(\text{shared\_expert\_gate}(x))$ (supporting `mlp_only_layers` / `decoder_sparse_step`) | Dense `dense-mlp` (`SwiGLU`) in all layers | **MoE**: `routed-experts` + $\text{shared\_expert} \cdot \sigma(\text{shared\_expert\_gate}(x))$ |
| **Norms & Multimodality** | `zero-centered RMSNorm`: $x_{\text{norm}} \cdot (1 + w)$ in `fp32`; text-only + `MTP` | `zero-centered RMSNorm`; optional `vision-tower` (ViT with axial 2D RoPE and $2 \times 2$ merger), text weights under `model.language_model.*` | Same as `qwen3_5_dense` |
