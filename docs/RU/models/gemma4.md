# Gemma 4

Oct 27, 2026 · @Vemloir

| Модель | Слои (sliding : full) | Окно | KV sliding | Global: K = V | Общий KV | PLE | Особое |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2B | 35 (28 : 7) | 512 | 1 × 256 | нет | последние 20 | 256 на слой | двойной MLP у слоёв с общим KV; аудио-conformer |
| E4B | 42 (35 : 7) | 512 | 2 × 256 | нет | последние 18 | 256 на слой | аудио-conformer |
| 12B | 48 (40 : 8) | 1024 | 8 × 256 | да, 1 голова × 512 | нет | нет | без энкодеров (`Gemma4Unified…`) |
| 26B-A4B | 30 (25 : 5) | 1024 | 8 × 256 | да, 2 × 512 | нет | нет | плотный MLP параллельно MoE (128 экспертов, top-8) |
| 31B | 60 (50 : 10) | 1024 | 16 × 256 | да, 4 × 512 | нет | нет | — |

Общее для семейства: на sliding-слоях head\_dim 256 и RoPE с θ = 10⁴, на глобальных — head\_dim 512 и p-RoPE (p = 0,25, θ = 10⁶); QK-norm и scaling attention 1,0; нормы до и после attention и MLP; `layer_scalar` умножает всё скрытое состояние в конце слоя; эмбеддинги × √hidden (множитель округлён до bf16); softcap логитов 30; связанные эмбеддинги, словарь 262 144.

**Как V зависит от K.** На глобальных слоях 12B, 26B и 31B (`attention_k_eq_v: true`) нет `v_proj`: K и V получаются из одной проекции.

```
raw = W_k · x                         на KV-голову, 512 измерений
K   = RoPE_p(RMSNorm(raw) ⊙ w_k)      норма с весом k_norm, затем p-RoPE
V   = RMSNorm(raw)                    та же норма, без веса и без RoPE
⇒ K = RoPE_p(V ⊙ w_k)
```

Поэтому кэш хранит только V, а ядро attention восстанавливает K на лету: умножение на вектор w\_k и поворот 128 из 512 измерений. Глобальный KV вдвое меньше, и на decode читается вдвое меньше байт. Расхождение с transformers, где K и V кэшируются отдельно, — только в округлении bf16. На контексте 262 144:

| Модель | Глобальный KV: K и V | Только V | Sliding KV (без запаса T\_max) |
| --- | --- | --- | --- |
| 12B | 4 ГиБ | 2 ГиБ | 320 МиБ |
| 26B-A4B | 5 ГиБ | 2,5 ГиБ | 200 МиБ |
| 31B | 20 ГиБ | 10 ГиБ | 800 МиБ |

**p-RoPE.** rope\_angles = 0,25 × 512 / 2 = 64; частоты θ^(−2i/512) для i < 64, остальные 192 угла нулевые; rotate-half идёт по всей голове, поэтому вращаются измерения \[0, 64) и \[256, 320), а не первые 128.

**Общий KV** (E2B/E4B). Последние `num_kv_shared_layers` слоёв не имеют k/v-проекций и читают кэш последнего слоя того же типа (sliding или full) перед точкой разделения. Своего кэша у них нет, а на prefill они нужны только строкам с логитами ([раздел 5 architecture.md](../architecture.md#5-планировщик-и-пайплайн)).

**PLE** (E2B/E4B). Табличная часть — `embed_tokens_per_layer` \[262 144, слои × 256\] × √256, выборка по id: у E4B это 262 144 × 42 × 256 × bf16 ≈ 5,6 ГБ, около трети всех весов, — кандидат на `disk` или `ram`. Контекстная часть — проекция основного эмбеддинга × hidden^(−½) → RMSNorm. Итог — (контекст + выборка) / √2. В каждом слое после MLP: gelu(W\_gate h) ⊙ ple\_ℓ → W\_proj → RMSNorm → + residual.

**MoE 26B-A4B.** В каждом слое плотный MLP (intermediate 2112) и 128 экспертов (top-8, intermediate 704) работают параллельно; роутер смотрит на residual до MLP, поэтому эксперты не ждут плотный MLP. Плотный MLP — естественная работа GPU, пока CPU считает экспертов.

**Маска для зрения**: на sliding-слоях — И(окно, ИЛИ(причинная, внутри картинки)); глобальные слои — только причинные.

**Черновик-ассистент** (`gemma-4-*-assistant`, черновик `Gemma4Assistant` — [раздел 6 architecture.md](../architecture.md#6-рантайм-ядро-генерации-и-спекулятивка)). 4 слоя без своих k/v-проекций: attention читает кэш последнего sliding- и последнего full-слоя цели, своего KV у черновика нет. Вход — `pre_projection` от конкатенации эмбеддинга токена (из таблицы цели) и последнего скрытого состояния цели; выход через `post_projection` возвращается в размер цели и служит скрытым состоянием на следующем шаге черновика. Опционально — голова по центроидам (2048 кластеров, top-32) вместо полного lm\_head (`use_ordered_embeddings`; у 12B выключена)

## Источники

- Конфиги Gemma 4: [E2B](https://huggingface.co/google/gemma-4-E2B-it), [E4B](https://huggingface.co/google/gemma-4-E4B-it), [12B](https://huggingface.co/google/gemma-4-12B-it), [26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it), [31B](https://huggingface.co/google/gemma-4-31B-it), [12B assistant](https://huggingface.co/google/gemma-4-12B-it-assistant)
- [transformers: modeling\_gemma4.py, modeling\_gemma4\_unified.py, modeling\_gemma4\_unified\_assistant.py, modeling\_rope\_utils.py](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4)
- [Gemma 4 Technical Report](https://arxiv.org/pdf/2607.02770)
- [Sebastian Raschka: Gemma 4 Architecture and Benchmark Notes](https://www.sebastianraschka.com/blog/2026/gemma-4-release-notes.html)
- [Maarten Grootendorst: A Visual Guide to Gemma 4](https://newsletter.maartengrootendorst.com/i/193064129/kv)
