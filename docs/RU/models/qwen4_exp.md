# `qwen4_exp` (Qwen3.8-Flash-Next)

Oct 27, 2026 · @Vemloir

| Что | Конфиг | Что нужно движку |
| --- | --- | --- |
| 36 слоёв Gated DeltaNet и 12 full, full каждый 4-й | `layer_types`, `linear_*`, `full_attention_interval: 4` | ядро DeltaNet, рекуррентное состояние ([раздел 3 architecture.md](../architecture.md#3-память-места-хранения-и-режимы-доступа)) |
| QSA в full-слоях | `indexer_*`: 4 головы по 128, бюджет 2048 токенов, блоки по 4 | индексатор оставляет каждому запросу ≤ 512 блоков плюс незавершённый хвост; выбор пересекается с причинной маской. Attention читает ограниченное число позиций, поэтому KV full-слоёв на длинном контексте можно держать в RAM и подтягивать выбранные блоки — кандидат на оффлоад |
| Hyper-connections | `hc_count: 4`, `hc_lowrank: 320` | residual — 4 потока; перед подблоком они смешиваются в его вход, выход впрыскивается обратно (`block_inject_weight`), перед финальной нормой потоки сворачиваются (`hyper_connection_mixer`) |
| PLE на n-граммах | `ple_*`, `ngram_*` | `RowTable` на диске + `Fetch`; PLE-слой: ключ на каждый поток и общее значение, гейт от потока, dilated causal conv (ядро 4, dilation 3) |
| MoE | 512 экспертов, top-10, intermediate 640; общий эксперт 640 с сигмоидным гейтом | эксперты почти целиком на CPU |
| MTP | 1 слой: attention, MoE, hyper-connections | черновик `Mtp`, своё место в плане |
| M-RoPE | `mrope_section: [11, 11, 10]`, interleaved, partial 0.25 от head\_dim 256 | позиции \[3, T\] |
| Зрение | ViT на 27 блоков | [раздел 7 architecture.md](../architecture.md#7-мультимодальность) |

**Таблица n-грамм.** У Qwen3.8-Flash-Next позиция даёт (ngram\_size − 1) × heads\_per\_ngram = 2 × 8 = 16 хэш-строк по 160 значений; вместе они дают вектор 2560. Таблица — 16 голов по ~20 млн строк (размеры голов выводятся из `ngram_vocab_size_base`): 102 ГБ в bf16, ~52 ГБ в int8 (162 байта на строку с масштабом), ~28 ГБ кольцами TCQ4X (86 байт на строку).

## Источники

- [Qwen/Qwen3.8-Flash-Next: config.json, model.safetensors.index.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [transformers 5.16.0: Qwen4-Exp (GatedResidual, QSA, PLE)](https://newreleases.io/project/pypi/transformers/release/5.16.0)
