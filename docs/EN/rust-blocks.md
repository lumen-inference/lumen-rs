> **Historical Rust implementation reference.** Preserved for Velri’s earlier Rust prototype; package and type names below belong to that implementation. Velri's target C/Vulkan design is in [blocks.md](blocks.md). Commands and types below do not describe the new C API.

# Computation blocks and the first working path

[Русский](../RU/rust-blocks.md)

The exact types and limits below belong to the earlier implementation. The [block overview](blocks.md) explains model composition, and the [glossary](glossary.md) defines common terms.

A block describes how to connect operations. It expands into `GraphBuilder` during construction; token execution traverses a prepared schedule, not block objects. Model descriptions can reuse the same MLP implementation while choosing their own layer connections.

The synthetic decoder (`../../crates/lumen-exec/examples/support/mod.rs`) assembles two layers from public blocks:

```sh
cargo run -p lumen-exec --example dense_decode
```

It computes logits from token IDs using small, fixed weights. Checkpoint loading, tokenization and text generation through `lumen run` are not connected yet.

## Graph identities

`lumen-models::graph` separates three kinds of handle:

| Handle | Meaning |
| --- | --- |
| `ValueId` | An operation result with a dtype, row width and row domain: current tokens or the history of a particular store. |
| `WeightId` | An immutable weight declaration. Several operations can share it, including embedding and output projection. |
| `StoreId` | The owner of a mutable sequence tensor, preserved between steps. It has no inherent K/V role. |

Handles belong to one graph; a matching numeric index from another graph is rejected. Operations validate shapes and parameters before insertion. A finished `Graph` is immutable.

`Component` identifies a placement path. For example, `Component::layer(2, "mlp/gate")` belongs to the third layer and has the weight binding key `hidden-layers/2/mlp/gate`. A model adapter must translate checkpoint names to these keys; the executor does not know checkpoint names. The global `hidden-layers` prefix is reserved to prevent collisions with layer weight keys.

## Available blocks

| Block | Computation and responsibilities |
| --- | --- |
| `Linear` | `x Wᵀ + bias`, referring to separately declared weights. |
| `RmsNorm` | RMS normalization with epsilon and scale `w` or `1+w`. |
| `GatedMlp` | `down(activation(gate(x)) * up(x))`; SwiGLU or GeGLU with tanh-approximate GELU and optional symmetric gate/up input clamping. |
| `AttentionBlock` | Causal full or sliding-window attention over explicit Q/K/V inputs. Supports grouped heads and different key/value dimensions; owns no projections or stores. |
| `ProjectedAttention` | A recipe with separate Q/K/V projections, optional RoPE, two explicitly supplied stores, attention and output projection. |
| `PreNormResidual<B>` | `x + scale * B(norm(x))` for any unary block. |

`Block::build` is a convenience for unary compositions. Multi-input blocks such as `AttentionBlock` use their own build method. Values can connect different layers; there is no mandatory layer superclass.

RoPE is an explicit rotate-half operation on each head's first `rotary_dim` features. Current-token positions start at the committed history length. An attention window includes the query's own position.

Discard a builder after a composite block returns an error: insertion of the partial composition is not rolled back. Individual primitives validate arguments before mutating the graph.

## Shared and derived history

Storage is declared separately from attention. `append(store, current)` writes the current rows and returns the available history. `read_store(store)` returns that same value without copying or allocating. Each store has exactly one writer per step; several blocks or layers can read it after that writer.

With `q`, `current`, `store` and a transform weight already declared:

```rust,ignore
let history = graph.append(store, current)?;
let derived = graph.linear(history, transform, None)?;
let out = attention.build(&mut graph, q, derived, history)?;
```

Here K is derived and V directly uses the history. Swapping the last two arguments derives V instead. Passing `history` twice shares a single buffer. Storage contains no family switch or “V only” flag.

The current executor materializes a **full-history transform** in an activation buffer. Its entire capacity appears in `activation_bytes`; saving persistent storage does not hide this temporary cost. Chunked derivation and fusion with attention are future optimizations. Sliding attention restricts reads but still retains all history up to `max_context`; it does not yet use a ring buffer.

## Preparation and execution

`Program::prepare` receives a graph, `PlacementPlan`, `DenseWeights`, capacity limits and an executor. It validates weight keys, dimensions, finite data, placement, operation support and the memory budget. Missing and extra weight keys are errors.

For weights used exclusively by Linear, the executor chooses physical storage through `dense_layout`. `RowMajor` keeps the ordinary matrix; `ColumnTiles` stores output-column bands with a zero-padded final band. Logical M/K are separate from physical dimensions. Preparation happens once, padding is included in the budget, and multiple consumers share one owner. A weight also used by embedding or another operation stays row-major without an extra copy. This does not change LTQ storage.

`Program::session()` creates one sequence. Immutable weights are shared across sessions through `Arc`; mutable buffers are private to each session. `Session::run` accepts token IDs and executes the immutable schedule. `output` borrows the result slice, `position` returns the committed length, and `reset` starts a new sequence.

Appends only write after the committed frontier. The frontier advances after every operation succeeds. Following a failure, output is unavailable and the next step overwrites uncommitted rows; earlier committed history remains intact. This is a synchronous executor contract. Asynchronous backends will need an explicit completion contract.

`MemoryReport` counts reserved weight, state, activation and token-input payload for one session, including one share of its immutable weights. It excludes Rust metadata, allocator overhead, input `DenseWeights` copies during preparation and other sessions. Fixed SIMD stack tiles and worker-pool metadata/stacks are also outside this tensor payload report. Graph activations are not reused yet. All computational buffers are reserved before stepping.

## Scope and extension

The implemented path handles one sequence, `ram:0` / `cpu:0`, dense-f32 weights and the operations above. `CpuExecutor` offers Scalar, AVX2/FMA and AVX-512 implementations. The independent scalar `lumen-ref` remains the automated numerical oracle. See [cpu-performance.md](cpu-performance.md) for measurements and limits. CUDA, offload and quantized state are rejected. QSA, DSA, GDN, MoE, PLE and other algorithms are not implemented in this graph; they are not silently approximated by dense attention or MLP.

Build new blocks from existing operations where possible. Missing mechanisms need an operation with explicit operands, parameters and resources, validation, and reference execution. Recurrent state, for example, needs its own update contract rather than pretending to append sequence rows.

The legacy `Architecture → Layout` path still serves config parsing and `inspect`; family adapters do not populate the new graph yet. The graph's placement tree currently classifies sequence stores using the existing dense-read `KvCache` category. That category does not determine physical buffer count or roles. Sparse access needs a separate description when those algorithms are added.

The decoder test compares every partition of its input against an independent Python oracle (`../../crates/lumen-exec/tests/fixtures/generate.py`). Additional tests cover shared and derived state, temporary memory accounting, invalid shapes and recovery after a failure following append.
