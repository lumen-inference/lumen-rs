# lumen-rs — Configuration and Placement Cascade

Oct 27, 2026 · @Vemloir

The config is JSON describing the model and its placement. At load time it is resolved once into a flat `PlacementPlan` ([section 2 of architecture.md](architecture.md#2-placement-plan)), after which nobody reads the JSON. Examples are in [`examples/`](../../examples/).

**Component names.** The tree comes from the architecture (`Architecture::components()`); typical names:

```
input-layer                        embed
ple/                               table, projections  (Gemma 4 E2B/E4B; Qwen3.8-Flash-Next — the n-gram table)
vision-tower, audio-tower          Qwen3.5 / 3.8, Gemma 4
hidden-layers/
  norms/                           input-norm, post-attention-norm, pre/post-feedforward-norm
  attention/                       q (incl. output gate), k, v, o, qk-norm, rope, indexer/keys, kv-cache
  linear-attention/                projections/{qkv,z,b,a,qkvz,ba,out}, conv, norm, state  (Gated DeltaNet)
  hyper-connections/
  dense-mlp/                       gate, up, down
  moe/                             router, shared-experts/{gate,projections/…}, routed-experts/projections/{gate,up,down}
output-layer/                      norm, lm-head
mtp/                               a subtree of the same shape
```

**Cascade Resolution Rules**

1. Only `compute`, `store`, and `prefill-compute` are inherited; other keys (`kv-cache.*`) apply only within their own node.
2. A node key is a property or a component name from the model's tree. Directly under `hidden-layers`, and only there, a layer selector is also allowed: comma-separated `N` and `N-M` ranges. A selector's value is a subtree of the same shape as `hidden-layers`: layer properties and components. A selector thus picks layers rather than components and works the same for models that interleave dense and MoE layers.
3. The value for a (layer ℓ, component path) pair is first looked up in the selector matching ℓ: the deepest node along the path inside the selector where it is set. If the selector does not set it, the deepest node along the regular path under `hidden-layers` is used, then the ancestors. A selector beats a regular node at any depth.
4. Two selectors matching the same layer is an error. Layer indices outside `[0, num_hidden_layers)` produce a warning; `--strict` turns it into an error.
5. An unknown key is an error with a suggestion of the closest match (`stor` → `store`).
6. Tree components that a particular model or layer lacks (`qk-norm` and `shared-experts` in gpt-oss, `moe` in dense layers) are skipped and annotated in `lumen inspect`; this way `dense-mlp` and `moe` under `hidden-layers` each apply to their own layers, and one config serves several models.

How this plays out for `routed-experts` in the MoE example (Qwen3.8-Flash-Next, 48 layers):

| Layer | compute / store of projections | Source of the value |
| --- | --- | --- |
| 0 | cpu:0 / ram:0 | `hidden-layers → moe → routed-experts` |
| 32 | cuda:0 / vram:0 | `hidden-layers → "32-47" → moe → routed-experts`; the selector overrode the regular node |

A selector can also cover a whole layer: `"45-47": { "compute": "cuda:0", "store": "vram:0" }` moves layers 45–47 to the GPU entirely, including nodes the regular tree sets deeper. Selectors are written for a specific model's layer count: on a model with fewer layers, the extra indices produce warnings.

**The `model` section holds only the model and its placement:** the path, the component cascade, and `sampling`. Everything unrelated to placement lives at the top level of the config, next to `model`:

```json
"chat-template": { "kwargs": { "enable_thinking": true } },
"context": { "size": 8192 },
"kv-quantization": { "k-bits": 8, "v-bits": 5 },
"attention-mechanism": "flash-attention"
```

- `chat-template` — the chat template and its kwargs ([api.md](api.md)).
- `context.size` — the maximum sequence length in tokens; the KV pool and states are allocated from it.
- `kv-quantization` — the bit width of K and V (`k-bits`, `v-bits`: 2–8 each); if omitted, KV is in bf16 ([section 3 of architecture.md](architecture.md#3-memory-storage-locations-and-access-modes)).
- `attention-mechanism` — the attention implementation; if omitted, `flash-attention` is used. The sliding-layer window and other attention parameters come from the model config.

The context size, KV quantization, and the attention implementation are properties of the sequence and are the same for every layer.

The `cpu`, `cuda`, and `disk` sections are backend properties, not cascade nodes: `threads`, `reserve-cores`, `numa-spread`, `prefill-chunk-size`, `io-threads`, and `mlock` are described in [sections 3 and 5 of architecture.md](architecture.md#3-memory-storage-locations-and-access-modes); `cuda.cuda-graphs: false` turns off CUDA Graphs ([section 5 of architecture.md](architecture.md#5-scheduler-and-execution-pipeline)) for debugging and profiling. The `server` section is described in [api.md](api.md).

**Implementation order.** Phase 1 only needs a flat plan — one place per component, no selectors. The selector grammar and the cascade arrive once the first model runs on the CPU.
