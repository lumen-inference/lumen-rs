# lumen-rs — конфиг и каскад размещения

Oct 27, 2026 · @Vemloir

Конфиг — JSON с моделью и её размещением. При загрузке он один раз разрешается в плоский `PlacementPlan` ([раздел 2 architecture.md](architecture.md#2-план-размещения)), после чего JSON никто не читает. Примеры — [`examples/`](../../examples/).

**Имена компонентов.** Дерево задаёт архитектура (`Architecture::components()`); типичные имена:

```
input-layer                        embed
ple/                               table, projections  (Gemma 4 E2B/E4B; Qwen3.8-Flash-Next — таблица n-грамм)
vision-tower, audio-tower          Qwen3.5 / 3.8, Gemma 4
hidden-layers/
  norms/                           input-norm, post-attention-norm, pre/post-feedforward-norm
  attention/                       q (вкл. выходной гейт), k, v, o, qk-norm, rope, indexer/keys, kv-cache
  linear-attention/                projections/{qkv,z,b,a,qkvz,ba,out}, conv, norm, state  (Gated DeltaNet)
  hyper-connections/
  dense-mlp/                       gate, up, down
  moe/                             router, shared-experts/{gate,projections/…}, routed-experts/projections/{gate,up,down}
output-layer/                      norm, lm-head
mtp/                               поддерево того же вида
```

**Правила разрешения каскада**

1. Наследуются только `compute`, `store` и `prefill-compute`; остальные ключи (`kv-cache.*`) действуют в своём узле.
2. Ключ узла — свойство или имя компонента из дерева модели. Прямо под `hidden-layers`, и только там, допустим ещё селектор слоёв: `N` и `N-M` через запятую. Значение селектора — поддерево того же вида, что `hidden-layers`: свойства и компоненты слоя. Так селектор выбирает слои, а не компоненты, и одинаково работает для моделей, где плотные и MoE-слои идут вперемешку.
3. Значение для пары (слой ℓ, путь компонента) сначала ищется в селекторе, совпавшем с ℓ: берётся самый глубокий узел пути внутри селектора, где оно задано. Если селектор его не задаёт — самый глубокий узел обычного пути под `hidden-layers`, затем предки. Селектор бьёт обычный узел на любой глубине.
4. Два селектора, совпавшие на одном слое, — ошибка. Индексы вне `[0, num_hidden_layers)` — предупреждение, `--strict` делает их ошибкой.
5. Неизвестный ключ — ошибка с подсказкой ближайшего (`stor` → `store`).
6. Компоненты дерева, которых нет у конкретной модели или конкретного слоя (`qk-norm` и `shared-experts` у gpt-oss, `moe` в плотных слоях), пропускаются с пометкой в `lumen inspect`; так `dense-mlp` и `moe` под `hidden-layers` действуют каждый на свои слои, а один конфиг подходит нескольким моделям.

Как это работает на `routed-experts` из примера MoE (Qwen3.8-Flash-Next, 48 слоёв):

| Слой | compute / store проекций | Откуда значение |
| --- | --- | --- |
| 0 | cpu:0 / ram:0 | `hidden-layers → moe → routed-experts` |
| 32 | cuda:0 / vram:0 | `hidden-layers → "32-47" → moe → routed-experts`, селектор перекрыл обычный узел |

Селектор может задать и весь слой: `"45-47": { "compute": "cuda:0", "store": "vram:0" }` переносит слои 45–47 на GPU целиком, включая узлы, которые обычное дерево задаёт глубже. Селекторы пишутся под число слоёв конкретной модели: на модели с меньшим числом слоёв лишние индексы дадут предупреждение.

**Секция `model` — только модель и её размещение:** путь, каскад компонентов и `sampling`. Всё, что не относится к размещению, лежит на верхнем уровне конфига рядом с `model`:

```json
"chat-template": { "kwargs": { "enable_thinking": true } },
"context": { "size": 8192 },
"kv-quantization": { "k-bits": 8, "v-bits": 5 },
"attention-mechanism": "flash-attention"
```

- `chat-template` — шаблон чата и его kwargs ([api.md](api.md)).
- `context.size` — максимальная длина последовательности в токенах, по нему выделяются пул KV и состояния.
- `kv-quantization` — битность K и V (`k-bits`, `v-bits`: по 2–8); если секции нет, KV в bf16 ([раздел 3 architecture.md](architecture.md#3-память-места-хранения-и-режимы-доступа)).
- `attention-mechanism` — реализация attention; если поле не задано, берётся `flash-attention`. Окно sliding-слоёв и остальные параметры attention берутся из конфига модели.

Размер контекста, квантование KV и реализация attention — свойства последовательности, они одинаковы для всех слоёв.

Секции `cpu`, `cuda` и `disk` — свойства бэкендов, а не узлы каскада: `threads`, `reserve-cores`, `numa-spread`, `prefill-chunk-size`, `io-threads`, `mlock` описаны в [разделах 3 и 5 architecture.md](architecture.md#3-память-места-хранения-и-режимы-доступа); `cuda.cuda-graphs: false` отключает CUDA Graphs ([раздел 5 architecture.md](architecture.md#5-планировщик-и-пайплайн)) для отладки и профилирования. Секция `server` описана в [api.md](api.md).

**Порядок реализации.** В фазе 1 хватает плоского плана — одно место на компонент, без селекторов. Грамматика селекторов и каскад появляются, когда первая модель заработает на CPU.
