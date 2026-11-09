> **Historical Rust implementation reference.** Preserved for Velri’s earlier Rust prototype; package and type names below belong to that implementation. Velri's target C/Vulkan design is in [config.md](config.md). Commands and types below do not describe the new C API.

# Configuration and Cascading Placement (lumen-config)

[Русский](../RU/rust-config.md)

This is the JSON reference for the existing Rust CLI. For the new implementation, read the [configuration design](config.md); terms are defined in the [glossary](glossary.md).

**Status:** parsing, placement resolution, and `inspect` exist. Acceptance of a runtime/backend setting does not mean that its execution path is implemented or supported; see [architecture.md](architecture.md).

A configuration file in `velri` is a JSON document that defines:
1. **The model and the placement of its components across hardware devices** (`model` section).
2. **Runtime and backend parameters**: context length, KV cache quantization, CPU and disk thread pools, CUDA prefill chunking, and HTTP server settings.

At load time, the configuration is validated and resolved once into a flat `PlacementPlan` (historical source: `crates/lumen-config/src/lib.rs`). The compute path receives resolved settings, not the original JSON.

Example configurations are located in [`examples/`](../../examples/):
- [`examples/example-dense.json`](../../examples/example-dense.json) — 27B hybrid dense model (`Qwen3.8-27B`).
- [`examples/example-moe.json`](../../examples/example-moe.json) — hybrid MoE model with n-gram table and sparse attention (`Qwen3.8-Flash-Next`).

---

## 1. Cascading Placement Principle

A model contains dozens of layers, each with roughly ten internal sub-components (attention projections, cache, router, shared and routed experts, normalizations). To avoid listing device assignments for every tensor manually, the configuration uses **cascading inheritance**:

1. At the top level of `"model"`, default placement parameters are defined for the entire model (for example, `"store": "vram:0"` and `"compute": "cuda:0"`).
2. All child modules and layers inherit these defaults automatically.
3. For large components that do not fit in VRAM (such as routed experts `moe/routed-experts` or the `ple/table` lookup table), `store` and `compute` are overridden on the corresponding subtree.
4. To place specific layers on a different device, **layer range selectors** (such as `"32-47"`) are placed inside `"hidden-layers"`.

### Inherited Placement Properties

Any node in the model tree may specify three inherited keys:
- **`"store"`** — permanent storage location (`"vram:0"`, `"ram:0"`, `"disk"`).
- **`"compute"`** — compute device during token generation (`"cuda:0"`, `"cpu:0"`).
- **`"prefill-compute"`** *(optional)* — compute device during prompt processing (prefill). For example, if a component specifies `"store": "ram:0"`, `"compute": "cpu:0"`, and `"prefill-compute": "cuda:0"`, it runs on the CPU during token decode, while during prefill its weights are streamed over PCIe into a VRAM ring buffer and computed on the GPU.

---

## 2. Model Component Tree

Valid configuration paths are defined by the loaded model's `Architecture::components()` method. A typical component tree has the following structure:

```text
model
├── input-layer/                  # Input token embedding table (embed)
├── ple/                          # Per-Layer Embeddings (Gemma 4) or n-gram table (Qwen3.8)
│   ├── table                     # Row lookup table ("store": "disk" or "ram:0")
│   └── projections               # Per-layer PLE linear projections
├── vision-tower/                 # Vision encoder (ViT)
├── audio-tower/                  # Audio encoder (Conformer)
├── hidden-layers/                # Repeating decoder layers (0 .. N-1)
│   ├── norms/                    # Layer normalizations (input-norm, post-attention-norm, etc.)
│   ├── attention/                # Full, Sliding, or Sparse Attention module
│   │   ├── q, k, v, o            # Attention projection matrices (q includes output gate)
│   │   ├── qk-norm, rope         # Q/K normalization and RoPE positional encoding
│   │   ├── indexer/keys          # Indexer key cache (for QSA sparse attention)
│   │   └── kv-cache              # Paged KV cache of the layer
│   ├── linear-attention/         # Linear attention module (Gated DeltaNet / GDN)
│   │   ├── projections/          # Projections (qkv, z, b, a, qkvz, ba, out)
│   │   ├── conv, norm            # Short causal conv1d and gated output norm
│   │   └── state                 # Layer recurrent state (StatePool)
│   ├── hyper-connections/        # Multi-stream residual module (Hyper-Connections)
│   ├── dense-mlp/                # Dense FFN block (gate, up, down)
│   ├── moe/                      # Mixture-of-Experts module
│   │   ├── router                # Expert router
│   │   ├── shared-experts/       # Shared experts (gate, projections/...)
│   │   └── routed-experts/       # Routed expert bank
│   │       └── projections/      # Expert gate, up, down matrices
│   └── "<layer-selector>"/       # Layer range override: e.g. "0-15" or "32-47"
├── output-layer/                 # Final norm (norm) and output vocabulary head (lm-head)
└── mtp/                          # Multi-Token Prediction module (subtree of the same shape)
```

---

## 3. Cascade and Layer-Selector Resolution Rules

1. **Inheritance:** only the `compute`, `store`, and `prefill-compute` keys are inherited down the tree.
2. **Layer selectors:** directly inside `"hidden-layers"`, layer index selectors of the form `"N"` and `"N-M"` (including comma-separated ranges) are permitted. Each selector contains a subtree with the same structure as `"hidden-layers"`.
3. **Precedence order:** the value for `(layer ℓ, component_path)` is resolved in the following order (from highest to lowest priority):
   - **Priority 1 (layer selector):** the deepest matching node inside a layer selector matching layer $\ell$. A layer selector overrides the general `hidden-layers` subtree at any depth.
   - **Priority 2 (general layer tree):** the deepest matching node directly under `hidden-layers`.
   - **Priority 3 (model root):** default values at the root of `"model"`.
4. **Validation:**
   - Two selectors overlapping on the same layer index produce a load error.
   - Layer indices outside `[0, num_hidden_layers)` produce a warning (or an error with `--strict`).
   - An unknown key produces an error suggesting the closest valid name (e.g., `"stor"` → `"store"`).
   - Valid tree components absent from a specific model or layer (e.g., `moe` in dense layers or `shared-experts` in `gpt-oss`) are ignored and noted in `lumen inspect`.

### Resolution Example (`examples/example-moe.json`, 48 layers)

```json
"model": {
  "path": "models/qwen3.8-flash-next/model.vtq",
  "compute": "cuda:0",
  "store": "vram:0",
  "hidden-layers": {
    "moe": {
      "routed-experts": {
        "compute": "cpu:0", "store": "ram:0", "prefill-compute": "cuda:0"
      }
    },
    "32-47": {
      "moe": {
        "routed-experts": {
          "compute": "cuda:0", "store": "vram:0"
        }
      }
    }
  }
}
```

| Layer | Component | Resolved `compute` / `store` | Rule Source |
| :---: | :--- | :---: | :--- |
| `0..47` | `attention`, `linear-attention`, `moe/router`, `moe/shared-experts` | `cuda:0` / `vram:0` | `"model"` root |
| `0..31` | `moe/routed-experts` | `cpu:0` / `ram:0` (`prefill`: `cuda:0`) | `hidden-layers → moe → routed-experts` |
| `32..47` | `moe/routed-experts` | `cuda:0` / `vram:0` | `hidden-layers → "32-47" → moe → routed-experts` |

---

## 4. Top-Level Sections and Backend Settings

Outside the `"model"` object, top-level sections configure sequence, backend, and server parameters:

```json
{
  "chat-template": {
    "kwargs": { "enable_thinking": true }
  },
  "context": {
    "size": 262144
  },
  "kv-quantization": {
    "k-bits": 8,
    "v-bits": 5
  },
  "cuda": {
    "prefill-chunk-size": 4096,
    "cuda-graphs": true
  },
  "cpu": {
    "threads": 16,
    "reserve-cores": 1,
    "numa-spread": false,
    "affinity": "auto"
  },
  "disk": {
    "io-threads": 4,
    "mlock": false
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8080,
    "max-concurrent-requests": 1
  },
  "attention-mechanism": "flash-attention"
}
```

- **`chat-template.kwargs`** — stored settings for Jinja chat-template variables. Template execution is not connected yet.
- **`context.size`** — maximum sequence context length in tokens, determining KV and state pool allocations.
- **`kv-quantization`** — bit widths for keys (`k-bits`) and values (`v-bits`) from 2 to 8 bits; if omitted, the KV cache is stored in `bf16`.
- **`attention-mechanism`** — attention implementation to use (defaults to `"flash-attention"`).
- **`cuda`** — prompt chunk size `prefill-chunk-size` and `cuda-graphs` toggle (defaults to `true`; disabling is useful during profiling).
- **`cpu`** — persistent CPU worker pool settings, described below.
- **`disk`** — disk I/O worker count `io-threads` and `mlock` toggle to pin disk tables in RAM when memory permits.
- **`server`** — bind host, port, and `max-concurrent-requests` limit for `lumen serve`.

### CPU workers and affinity

| Key | Default | Behavior |
| :--- | :--- | :--- |
| `threads` | `0` | Automatic physical-core count when zero; otherwise the exact requested worker count. |
| `reserve-cores` | `1` | Cores subtracted from the automatic count, with at least one worker retained. Ignored for an explicit nonzero `threads` count. |
| `numa-spread` | `false` | Prefer nearby cores when false; spread selection across available NUMA nodes when true. Automatic sizing uses the first available node when false, and all available nodes when true. |
| `affinity` | `"auto"` | Worker binding policy: `"auto"`, `"none"`, or `"required"`. |

On Linux, topology detection respects the calling thread's allowed CPU mask, including restrictions imposed by `taskset` or a cpuset. Automatic selection uses one logical CPU per available physical core in the selected NUMA nodes, then subtracts `reserve-cores`. An explicit count can use additional nodes and SMT siblings as needed. `numa-spread` controls CPU selection; it does not change memory allocation policy, and NUMA nodes do not necessarily correspond to sockets or CCDs.

`affinity: "auto"` pins workers to selected CPUs when possible. If topology or binding is unavailable, or the requested count exceeds the available placement, execution falls back to unbound workers and exposes a diagnostic through `CpuExecutor::placement_note()`. An explicit worker count is preserved; if automatic topology detection fails, sizing uses available logical CPUs minus the reserve, with at least one worker. `"required"` makes a placement or binding failure an error. `"none"` disables worker binding while retaining automatic-count and reserve behavior. Automatic one-worker execution and small operations run on the calling thread; the executor does not change that thread's affinity.

The parser and `lumen inspect` expose these settings, and `lumen_engine::cpu_executor(isa, &config.cpu)` constructs a backend from them. The full engine and CLI `lumen run` command are still unimplemented, so a JSON config does not yet launch inference. The synthetic decoder example uses the same CPU threading API directly:

```bash
cargo run --release -p lumen-exec --example dense_decode -- avx512 8
cargo run --release -p lumen-exec --example dense_decode -- auto 0
```

The example's optional second argument is the CPU worker count; omitting it preserves the existing one-worker behavior. `0` selects the automatic count with one reserved core, one NUMA node, and automatic affinity. The `ref` mode does not accept a thread-count argument.
