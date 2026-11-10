# Velri documentation

[Русский](../RU/README.md)

Computations are being separated into **Velri-lib**. Start with the [project boundary and build instructions](velri-lib.md) to distinguish library work from model/request work.

Velri should run models on CPU and GPU, including when weights do not fit entirely in GPU memory. The new engine's main language is C23. CPU and Vulkan are its first two backends; other backends will connect through the same internal C interface.

**These documents preserve the experience from the earlier Rust prototype.** It could read settings, validate LTQ and run a small synthetic decoder on CPU. The C/Vulkan engine and real-model execution remain to be implemented. Describing a design below does not mean its implementation is ready.

## Where to start

1. [How the engine works](architecture.md) — the path from loading a model to producing a result, responsibilities and memory management.
2. [How to assemble a model from blocks](blocks.md) — operations, layers, attention, MLP/MoE and adding an architecture.
3. [How computations execute](kernels.md) — common kernel requirements, the C/LLVM generator, and CPU and Vulkan implementations.
4. [How to connect another backend](backends.md) — what it implements and how it joins the common execution plan.

These explain the main idea. Definitions live in a separate [glossary](glossary.md). Refer to it as needed; there is no need to memorize the entire list before reading.

## When you need details

| Task | Document |
| :--- | :--- |
| Understand the language, build and third-party libraries | [Stack and dependencies](dependencies.md) |
| Add execution on another device | [Backend interface and checks](backends.md) |
| Configure where data resides and computation runs | [Configuration](config.md) |
| Understand worker dispatch and CPU/GPU exchange | [Execution](execution-design.md) |
| Connect an application and understand data lifetimes and errors | [Proposed C API](api.md) |
| Measure speed correctly and compare with a ceiling | [Performance](performance.md) |
| Find the implementation order and readiness criteria | [Roadmap](roadmap.md) |

## Exact specifications and accumulated experience

These documents support implementation and checking of individual parts. Read them after the general description of the corresponding mechanism.

| Document | What to look for |
| :--- | :--- |
| [GEMM and GEMV implementation](kernel-reference.md) | Formulas, weight packing, SIMD, work partitioning and memory rules preserved for the new implementation |
| [Existing CPU kernel measurements](cpu-performance.md) | Results and limitations of the earlier Rust/C/LLVM path, not measurements of the future C runtime |
| [LTQ v1](ltq-format.md) | Exact weight-reconstruction formulas and on-disk bytes |
| [LTQ research](ltq-design.md) | Quantization results, quality comparisons and the quantizer design |
| [Model families](models/README.md) | Details of individual architectures; a description does not mean working support |
| [Current Rust blocks](rust-blocks.md) | Existing types, a working example and limits of the earlier API |
| [Current Rust settings](rust-config.md) | JSON accepted by the existing CLI; future Vulkan settings are documented separately |

Experiment reports retain their measurement conditions. Their numbers cannot be carried over to other hardware or a new implementation without verification.

## Keeping translations in sync

Russian and English pages describe the same decisions, examples and limits. When a contract, formula, example or implementation status changes, update both versions in the same change. Every page starts with a link to the other language. Historical command, file and measurement names are retained so references remain verifiable.

The first implemented C kernel is documented in [CPU linear](cpu-linear.md), including build and exactness checks.
