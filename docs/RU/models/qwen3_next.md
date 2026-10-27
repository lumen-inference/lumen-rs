# `qwen3_next`, `qwen3_5_dense` (`qwen3_5`) и `qwen3_5_moe`

Oct 27, 2026 · @Vemloir

Базовое гибридное семейство Qwen: по массиву `layer_types` три слоя `linear_attention` (Gated DeltaNet) чередуются с одним слоем `full_attention` (Gated Attention).

| Узел | `qwen3_next` | `qwen3_5_dense` (`qwen3_5`) | `qwen3_5_moe` |
| --- | --- | --- | --- |
| Вход `GatedDeltaNet` | 2 проекции: `in_proj_qkvz` и `in_proj_ba`, упакованные по `num_k_heads` группам (`fix_query_key_value_ordering`) | 4 проекции: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` (узкие `b` и `a` на `num_v_heads` — `DenseLinear`) | как у `qwen3_5_dense` |
| Ядро `GatedDeltaNet` | causal `conv1d` (окно 4, SiLU) по `[q, k, v]` → дельта-правило с L2-нормой Q и K, `beta = σ(b)`, `g = −exp(A_log)·softplus(a + dt_bias)` → `RMSNormGated(head_v_dim)` с гейтом `silu(z)` → `out_proj` | то же | то же |
| `full_attention` | **Gated Attention**: `q_proj` ширины `2 · H · D` делится в каждой голове пополам на `(q, gate)`; `q_norm` и `k_norm` по `head_dim`; после attention выход умножается на `σ(gate)` перед `o_proj` | то же | то же |
| RoPE | partial 1D RoPE (`partial_rotary_factor: 0.25`) | interleaved M-RoPE `mrope_section: [11, 11, 10]`, partial 0.25 | как у `qwen3_5_dense` |
| FFN / MoE | MoE (`routed-experts` + `shared_expert · σ(shared_expert_gate(x))`), с поддержкой `mlp_only_layers` / `decoder_sparse_step` | плотный `dense-mlp` (SwiGLU) во всех слоях | MoE (`routed-experts` + `shared_expert · σ(shared_expert_gate(x))`) |
| Нормы и мультимодальность | zero-centered RMSNorm: `x_norm · (1 + w)` в fp32; текстовая модель | zero-centered RMSNorm; опционально `vision-tower` (ViT с axial 2D RoPE и merger 2×2), веса текста под `model.language_model.*` | как у `qwen3_5_dense` |
