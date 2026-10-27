# Архитектуры моделей

Oct 27, 2026 · @Vemloir

Что каждое семейство требует от движка. Особенности моделей — параметры операторов, а не отдельный код исполнителя ([раздел 8 architecture.md](../architecture.md#8-расширяемость)). У семейства с заметными особенностями свой файл:

- [qwen3_next.md](qwen3_next.md) — `qwen3_next`, `qwen3_5_dense` (`qwen3_5`), `qwen3_5_moe`
- [qwen4_exp.md](qwen4_exp.md) — `qwen4_exp` (Qwen3.8-Flash-Next)
- [gemma4.md](gemma4.md) — Gemma 4

## Семейства

| Семейство | Что требует от движка |
| --- | --- |
| Llama, Qwen3 | базовый плотный путь |
| Qwen3-MoE | банки экспертов, softmax → top-k |
| `qwen3_next` | гибрид 3:1 Gated DeltaNet (слитые `in_proj_qkvz`/`ba` по K-группам) + Gated Attention (`q_proj` ×2 с выходным σ-гейтом), zero-centered RMSNorm (`1 + w`), `RMSNormGated`, MoE с `shared_expert_gate`, partial 1D RoPE 0.25, MTP |
| `qwen3_5_dense` (`qwen3_5`) | гибрид 3:1 Gated DeltaNet (раздельные `in_proj_{qkv,z,b,a}`) + Gated Attention, плотный SwiGLU MLP на каждом слое, zero-centered RMSNorm, interleaved M-RoPE `[11, 11, 10]` (partial 0.25), ViT с axial 2D RoPE, MTP; например, Qwen3.8-27B: 64 слоя (48 : 16), hidden 5120 |
| `qwen3_5_moe` | гибрид 3:1 Gated DeltaNet (раздельные `in_proj_{qkv,z,b,a}`) + Gated Attention, MoE (роутер + routed-эксперты + общий эксперт с `shared_expert_gate`), M-RoPE, ViT, MTP |
| gpt-oss | синки, чередование sliding/full по `layer_types`, YaRN, SwiGLU с clamp 7.0, bias в проекциях, Harmony |
| Qwen3.8-Flash-Next (`qwen4_exp`) | надстройка над `qwen3_next`/`qwen3_5_moe`: QSA в full-слоях, hyper-connections, n-gram PLE, MTP, M-RoPE, зрение |
| Gemma 4 | K = V, p-RoPE, общий KV, PLE, плотный MLP параллельно MoE, мультимодальность без энкодера у 12B |

## Энкодеры

| Схема | Модели | Что считает энкодер |
| --- | --- | --- |
| Энкодер-башня | `qwen3_5_dense`, `qwen3_5_moe`, Qwen3.8-Flash-Next: ViT на 27 блоков (axial 2D RoPE), hidden 1152, патч 16, merger 2×2 → hidden декодера, видео с temporal patch 2. Gemma 4 26B и 31B: ViT на 27 слоёв, hidden 1152. Gemma 4 E2B/E4B: ViT на 16 слоёв; аудио — conformer на 12 слоёв с чанковым attention | двунаправленное attention, без KV, раз на картинку или аудиофрагмент; компоненты `vision-tower`, `audio-tower` |
| Без энкодера | Gemma 4 12B (`Gemma4Unified…`) | картинка режется на патчи 48×48: LayerNorm → Linear 6912 → 3840 → LayerNorm → + факторизованные позиционные эмбеддинги (x, y) → LayerNorm → RMSNorm → Linear; до 280 токенов на картинку. Аудио — сырые кадры по 640 отсчётов (40 мс при 16 кГц) → RMSNorm → Linear 640 → 3840, до 750 токенов. Остальную работу делает сам декодер |
