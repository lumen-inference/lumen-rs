# `qwen3_next`, `qwen3_5_dense` (`qwen3_5`), and `qwen3_5_moe`

Oct 27, 2026 · @Vemloir

The baseline hybrid Qwen family: per `layer_types`, three `linear_attention` layers (Gated DeltaNet) alternate with one `full_attention` layer (Gated Attention).

| Sub-block | `qwen3_next` | `qwen3_5_dense` (`qwen3_5`) | `qwen3_5_moe` |
| --- | --- | --- | --- |
| `GatedDeltaNet` input | 2 projections: `in_proj_qkvz` and `in_proj_ba`, packed by `num_k_heads` groups (`fix_query_key_value_ordering`) | 4 projections: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` (narrow `b` and `a` of width `num_v_heads` stay `DenseLinear`) | same as `qwen3_5_dense` |
| `GatedDeltaNet` core | causal `conv1d` (kernel 4, SiLU) over `[q, k, v]` → delta rule with Q/K L2-norm, `beta = σ(b)`, `g = −exp(A_log)·softplus(a + dt_bias)` → `RMSNormGated(head_v_dim)` with `silu(z)` gate → `out_proj` | same | same |
| `full_attention` | **Gated Attention**: `q_proj` of width `2 · H · D` splits in half per head into `(q, gate)`; `q_norm` and `k_norm` over `head_dim`; attention output is multiplied by `σ(gate)` before `o_proj` | same | same |
| RoPE | partial 1D RoPE (`partial_rotary_factor: 0.25`) | interleaved M-RoPE `mrope_section: [11, 11, 10]`, partial 0.25 | same as `qwen3_5_dense` |
| FFN / MoE | MoE (`routed-experts` + `shared_expert · σ(shared_expert_gate(x))`), with `mlp_only_layers` / `decoder_sparse_step` support | dense `dense-mlp` (SwiGLU) in all layers | MoE (`routed-experts` + `shared_expert · σ(shared_expert_gate(x))`) |
| Norms and multimodality | zero-centered RMSNorm: `x_norm · (1 + w)` in fp32; text-only | zero-centered RMSNorm; optional `vision-tower` (ViT with axial 2D RoPE and 2×2 merger), text weights under `model.language_model.*` | same as `qwen3_5_dense` |
