# How to assemble a model from blocks

[Русский](../RU/blocks.md)

Common mathematical blocks belong to Velri-lib; model-family adapters in Velri choose, parameterize and connect them. See the [project boundary](velri-lib.md).

A new architecture should not require writing the entire engine again. We want to describe a model using reusable parts: choose suitable blocks, set their parameters, and connect them in the required order.

**Status:** this document describes the future C and Vulkan implementation. A mechanism appearing in an example does not mean its kernel is ready. The existing Rust API and its limitations are documented in [rust-blocks.md](rust-blocks.md). Terms are defined in the [glossary](glossary.md).

## Operation, block, layer, and architecture

These levels differ in the scope of the task they describe.

| Level | Example | What the author specifies |
| :--- | :--- | :--- |
| Operation | Linear projection, normalization, row lookup, attention computation | Its inputs, computation, and the state it reads or changes |
| Block | MLP, MoE, attention with projections | How operations connect to one another |
| Layer | Attention and MLP with normalization and residual connections | Block order and data flow between blocks |
| Architecture | A sequence of different layers, input embeddings, and an output head | The complete model structure, including connections between layers and additional branches |

A block can contain other blocks. For example, MoE selects experts and combines their outputs, while each expert can use an ordinary MLP. The internals of that MLP do not need to be written again.

An operation is large enough to have an efficient implementation. We do not need a separate graph node for every arithmetic instruction. When needed, a coarse operation can be decomposed into simpler ones for reference computation.

## Example: one MLP in different places

One MLP variant has two input projections and one output projection:

```text
input ──→ Linear(gate) ──→ Activation ──┐
                                      × ──→ Linear(down) ──→ output
input ──→ Linear(up) ──────────────────┘
```

This block can be used as a standalone dense MLP or as an expert inside MoE. The latter adds expert selection, dispatching inputs to experts, and combining results outside the block. The MLP itself need not know where it was called.

Reusing a block means reusing a computation description. **Weights and state do not automatically become shared.** If two blocks really share weights or history, this is specified separately.

Other MLPs can have different projections and activations. The diagram therefore describes one reusable block, rather than a mandatory structure for every MLP in the engine.

## Example: a hybrid architecture

Consider a hybrid structure alternating GDN and QSA. In this example, a GDN layer includes GDN itself, MoE, and the connections around them. A QSA layer has a similar structure but uses a different mechanism to mix information between tokens.

We can sketch the sequence as follows:

```text
gdn_layer = Layer(
    mixer = GDN(...),
    feed_forward = MoE(router, experts = GatedMLP(...)),
    connections = ...
)

qsa_layer = Layer(
    mixer = Attention(
        projections = ...,
        position_transform = ...,
        selection = QSASelection(...),
        attention_core = ...
    ),
    feed_forward = MoE(router, experts = GatedMLP(...)),
    connections = ...
)

layers = Repeat([gdn_layer, gdn_layer, gdn_layer, qsa_layer], count)
```

This illustrates an idea; it is **neither a finished C API nor a complete description of a particular model**. Repetition creates layer instances; their weights and state are mapped separately.

The `connections` field must explicitly specify normalization order, residual connections, and other transitions. The common structure does not impose pre-norm, gated MLP, or the same projection set on every family. An expanded residual scheme also includes reading and writing through GR Read and GR Write. These cannot be replaced with ordinary addition just because another model's layer works that way.

Besides the layer list, the architecture describes additional branches. For example, an n-gram embedding can connect to a particular layer, while MTP can use separate outputs. PLE likewise connects through its own computations and dependencies. These mechanisms must not hide inside attention or depend on a model identifier in common code.

## The parts of attention

Saying that a model “uses attention” is not enough to execute it. Several parts need descriptions:

| Part | What needs to be known |
| :--- | :--- |
| Input construction | How Q, K, and V are obtained: through projections, transformation of another value, or reading stored data |
| Position handling | How token positions are used, for example through RoPE, and with which parameters |
| History selection | Which positions are available: the entire permitted context, a window, or a selected subset |
| Result computation | Which formulas, scales, masks, and normalization apply |
| State persistence | What survives between steps, where it is stored, and in which order it is updated |

Full attention and sliding-window attention differ in the available positions. QSA and DSA can have their own selection algorithms and indexers. A single `sparse=true` switch cannot describe them correctly.

Suppose an indexer first scans the entire history and then selects a few blocks for attention. Attention itself reads only the selected blocks, but the indexer's work still includes a full scan. Descriptions and measurements must account for both parts.

A Flash-style kernel is a way to execute particular attention mathematics efficiently by processing data in tiles. This is a separate question from which positions may be read. Position selection, formulas, and their optimized execution are described separately.

## Why state does not mean two K and V buffers

Attention receives data in the K and V roles. How that data is stored is a separate architectural decision.

Different arrangements are possible:

- K and V are stored in separate memory regions.
- Both roles use one region when the model's mathematics allows it.
- One input is stored, and the other is computed from it.
- Both inputs are obtained from common latent state.

The mechanism is the same in every case: declare the actual storage, references to it, and the required transformations. The allocator allocates memory from this description. It does not create a pair of buffers by default or check a model family's name.

If a transformation produces temporary data, its size and lifetime are known in advance. Sometimes the transformation can be fused with its consumer without writing the whole intermediate result to memory. The selected kernel must explicitly support that input.

The same rule applies outside attention. GDN stores its recurrent state, convolution stores the history it needs, and an expanded residual scheme stores its own data. Their sizes are determined by the corresponding mechanism, rather than a common formula based on attention head count.

## What an operation must describe

For preparation to select a kernel and plan memory, an operation needs a precise description:

- the computation and its parameters;
- inputs and outputs: shapes, data types, and resource references;
- state effects: reading, writing, or appending to history;
- index semantics when accessing selected elements;
- temporary data requirements.

Weights, session state, and temporary scratch have different owners. The description must identify who is responsible for each resource and when it can be reused.

Preparation compares the requirements of the concrete operation against registered backend capabilities, accounting for strides, permitted aliasing, memory properties, state access, bounded dynamic dimensions, and numerical requirements. A backend returns a supported implementation with layout, alignment, and memory requirements, or a rejection reason. Preparation selects an implementation and explicitly declares required packing, transfers, and scratch. The common block does not choose a backend name, AVX2, a Vulkan queue, subgroup size, or thread count. Adding a device for existing mathematics does not require changing that block; the boundary is described in the [backend interface](backends.md).

A closed operation enum and a common registry with handler tables are enough for the first implementation. Each operation must have descriptor validation, a diagnostic name, an independent reference, and an explicit backend support response. The internal C interface queries operation support and prepares implementations; it does not gain a separate function pointer for every new operation. A separate binary interface for external plugins is needed only when an actual use case for such plugins appears.

A backend can support only some operations. Preparation may choose a declared equivalent decomposition or another device allowed by placement policy, with explicit conversions and transfers. If explicitly specified placement is unsupported, preparation returns an error. Optional fusion must preserve external values, state changes, and intermediate results read outside the fused region.

## Steps for adding a new model

1. **Understand the formulas and data.** Record dimensions, parameters, weight contents, state, and its update order.
2. **Map file weights to blocks.** Recognizing the family name in a config is not enough.
3. **Assemble the structure from existing blocks.** Describe layers, additional branches, and every shared resource.
4. **Add a missing mechanism.** First try to assemble it from existing operations. If a new operation is needed, describe it in general terms and write a simple reference. Then add an optimized implementation.
5. **Check the parts and the whole model.** Compare results with an independent implementation, check several consecutive steps, and process the same text both as a whole and in chunks, including prefill/decode equivalence.

Also check irregular dimensions, different K/V representations, shared memory, and the absence of unnecessary allocations. A correct result for one step does not guarantee that state will remain correct at the next step.

A family is supported after the full model has been verified. Current family registration in Rust provides configuration loading and information for `inspect`; it does not yet provide working generation. The overall implementation order is in the [roadmap](roadmap.md).
