# Velri

Velri is an inference engine being designed in **C23**, with **CPU and Vulkan** as its first compute backends. It targets Windows, Linux and Android, including models whose weights do not fit entirely in GPU memory. Backends are implemented and managed entirely by Velri-lib through its common internal C interface.

The separate **[Velri-lib](https://github.com/velri-inference/velri-lib)** project provides
mathematical operations and tools for building, preparing and executing kernel
launch graphs. Velri uses those operations to describe reusable model blocks and
architectures, and owns model loading and inference requests. See the
[project boundary and build instructions](docs/EN/velri-lib.md).

Documentation: **[English](docs/EN/README.md)** · **[Русский](docs/RU/README.md)**

Start with the [architecture](docs/EN/architecture.md), learn how to [assemble models from blocks](docs/EN/blocks.md), or see how to [add a backend](docs/EN/backends.md). Definitions live in a separate [glossary](docs/EN/glossary.md).

## Current status

The earlier Rust prototype provided: configuration and `lumen inspect`, LTQ container validation and reference decoding, reusable computation blocks, CPU execution preparation, and a synthetic dense-f32 decoder. Its CPU kernels use Rust, C and common LLVM IR generation. AVX2/AVX-512 checks and ARM tests under QEMU exist; emulation tests do not establish performance on physical ARM hardware.

**The first C CPU linear module is implemented and extracted into Velri-lib**:
prepared FP32 weights, sequential FP64 arithmetic, reference checks and CPU
variants. This retained prototype does not yet implement the target **F64,
FP16, BF16 and INT8** paths. Velri currently builds a library integration test;
the C graph, executor, Vulkan and real-model inference remain planned work.
[Build the two projects](docs/EN/velri-lib.md).

In a checkout of the historical Rust prototype, the decoder example runs with:

```sh
cargo run -p lumen-exec --example dense_decode
```

Its existing API is described in [Rust blocks](docs/EN/rust-blocks.md), and the current configuration schema in [Rust configuration](docs/EN/rust-config.md). The new C API and Vulkan JSON examples are proposals. Existing CPU results and limitations are retained in [CPU performance](docs/EN/cpu-performance.md).

Velri was previously named Lumen. Historical Rust package names, commands and benchmark labels are retained. Engine headers are planned under `include/velri/`; computation-library headers live under `include/velri_lib/` in Velri-lib. Model files use the `.vtq` extension. The LTQ v1 byte layout, magic and internal tensor keys remain unchanged.

## Design principles

- Build architectures from layers, reusable blocks and operations. Shared mechanisms do not branch on model-family names.
- Separate values from storage. K and V are input roles that may share memory or derive from other state; two allocations are not mandatory.
- Prepare kernels, memory and repeated job geometry before execution. Tokens, lengths and routes remain dynamic.
- Keep the C library independent of CLI, HTTP, tokenization and quantization applications.
- Add backends as modules with optional build dependencies. A common RAM transfer route avoids a separate protocol for every backend pair.
- Check device capabilities. Users do not need NVIDIA hardware, AVX-512 or the developer's CPU.
- Validate optimizations through correctness checks and complete execution costs. Changing implementation language does not itself prove a speedup.

## Documentation

English and Russian pages cover the same topics, examples and requirements. Each page links to its counterpart. When a contract, example or implementation status changes, update both versions together.

| English | Русский | Contents |
| :--- | :--- | :--- |
| [Reading guide](docs/EN/README.md) | [Порядок чтения](docs/RU/README.md) | Where to start and find details |
| [Glossary](docs/EN/glossary.md) | [Словарь](docs/RU/glossary.md) | Definitions used throughout the documentation |
| [Architecture](docs/EN/architecture.md) | [Архитектура](docs/RU/architecture.md) | Module boundaries, memory, preparation and execution |
| [Velri-lib boundary](docs/EN/velri-lib.md) | [Граница Velri-lib](docs/RU/velri-lib.md) | Separate computation project, ownership and connecting the builds |
| [Backends](docs/EN/backends.md) | [Бэкенды](docs/RU/backends.md) | Internal C interface and adding another device backend |
| [Dependencies](docs/EN/dependencies.md) | [Стек и библиотеки](docs/RU/dependencies.md) | C23, build tools, proposed libraries and open choices |
| [Blocks](docs/EN/blocks.md) | [Блоки](docs/RU/blocks.md) | Layer composition and state ownership |
| [Execution](docs/EN/execution-design.md) | [Исполнение](docs/RU/execution-design.md) | CPU/GPU dependencies, staging and prepared work |
| [Configuration](docs/EN/config.md) | [Конфигурация](docs/RU/config.md) | Proposed placement and backend settings |
| [Kernels](docs/EN/kernels.md) | [Кернелы](docs/RU/kernels.md) | Common contracts, CPU/LLVM and Vulkan compute |
| [C linear module](docs/EN/cpu-linear.md) | [C-модуль Linear](docs/RU/cpu-linear.md) | Build, exact arithmetic, preparation and memory contracts |
| [GEMM/GEMV reference](docs/EN/kernel-reference.md) | [Устройство GEMM/GEMV](docs/RU/kernel-reference.md) | Preserved formulas, weight packing and execution rules |
| [Performance](docs/EN/performance.md) | [Производительность](docs/RU/performance.md) | Profiling, TFLOPS, conditional ceilings and reproducibility |
| [API](docs/EN/api.md) | [API](docs/RU/api.md) | Ownership, errors, sessions and applications |
| [Roadmap](docs/EN/roadmap.md) | [Порядок работ](docs/RU/roadmap.md) | Implementation stages and acceptance criteria |
| [LTQ v1](docs/EN/ltq-format.md) | [LTQ v1](docs/RU/ltq-format.md) | The unchanged byte-level format specification |
| [LTQ research](docs/EN/ltq-design.md) | [Исследование LTQ](docs/RU/ltq-design.md) | Quantization research and the planned quantizer |
| [Model families](docs/EN/models/README.md) | [Семейства моделей](docs/RU/models/README.md) | Architecture references, not a list of implemented models |
