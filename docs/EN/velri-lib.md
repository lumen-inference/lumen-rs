# Velri and Velri-lib

[Русский](../RU/velri-lib.md)

**[Velri-lib](https://github.com/velri-inference/velri-lib) is a separate C23 computation project.** Velri depends on it; the library does not depend on the engine. `external/velri-lib` is a Git submodule pinned to a library commit. A separate sibling checkout can override it during joint development. Neither Git nor CMake updates the dependency to a moving branch automatically.

Its main purpose is to provide mathematical operations and tools for building,
preparing and executing kernel launch graphs. Velri uses this API to describe
model blocks; a new block made from supported operations requires no library
changes. This is the target design, not a list of implemented features.

| Velri-lib | Velri |
| :--- | :--- |
| Tensor views, storage, layouts and number types | Model configuration and weight-name mapping |
| Mathematical operations and their kernels | Reusable MLP/attention/MoE blocks, layers and architectures |
| Graph builder, validation, kernel selection and fusion | Describing block computations through the library API |
| Memory plan, weight packing, transfers and execution dependencies | Placement policy, memory budgets and workload limits |
| Backend registration and settings, devices, CPU workers, queues and completion | Requests, batching policy, generation, tokenization and serving |
| Declared numerical formats and their mathematical decoding | Model files, containers and metadata |
| Independent operation references and kernel benchmarks | Model quality, logits and end-to-end performance |

For example, a gated MLP block connects two Linear operations, an activation,
elementwise multiplication and an output Linear. Graph construction only records
these operations, values and state accesses. During preparation, the library
selects implementations and builds a kernel launch graph: jobs, copies,
conversions and their dependencies. Several operations can fuse into one kernel;
one operation can require several launches. The prepared graph can be replayed
with new inputs within its prepared bounds. A model does not specify AVX
instructions or Vulkan barriers. Common compositions in the library may provide
convenient graph-building functions without restricting how clients build blocks.

Velri owns request state and chooses which sequences share a batch. Velri-lib
schedules the resulting execution jobs. The two projects do not create competing
CPU pools. Separate execution slots hold mutable bindings and workspace; immutable
plans and prepared weights may be shared. K/V roles never imply two allocations.
Accepted work retains resources until completion, including cancellation paths.

All backends are developed and connected in Velri-lib. The library enumerates
devices, validates module settings, creates resources and manages their
lifetimes. Velri uses the common API, passes user settings and limits, and
displays returned diagnostics. The engine has no second device registry, GPU API
calls or copies of backend configuration rules. A new backend for known
operations requires changes to the library and its build, not to engine code.

Target formats are **F64, FP16, BF16 and INT8**. Storage, accumulator and output
types are separate fields; a device must support the complete combination.
Future floating kernels have documented error bounds rather than a blanket
requirement to match one sequential FP64 loop. Repeatability is checked separately.

## What has actually moved

The CPU linear implementation, independent reference, frozen fixture, correctness
tests, benchmark tools, ARM toolchain and kernel CI belong to Velri-lib. Current
inputs remain FP32 with ordered FP64 accumulation: extraction does not implement
the target formats or change arithmetic. Graphs, the executor and Vulkan remain
design work.

Velri retains an integration test that includes `<velri_lib/linear.h>` and links
`VelriLib::cpu_linear`. This experimental header is not the final graph API or a
stable ABI. Private library headers are not part of the engine interface.

## Build with the pinned dependency

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang
cmake --build build
ctest --test-dir build --output-on-failure
```

For a fresh clone use `git clone --recurse-submodules`. Updating the library is
an explicit change to the submodule commit, reviewed with integration tests.

## Build the two checkouts

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DVELRI_LIB_SOURCE_DIR=../Velri-lib
cmake --build build
ctest --test-dir build --output-on-failure
```

This runs the integration check and, by default, the library's correctness tests.
For an installed dependency, omit `VELRI_LIB_SOURCE_DIR` and supply
`-DCMAKE_PREFIX_PATH=/path/to/velri-lib-install`. The required package is currently
`VelriLib 0.1.0 EXACT`. No source-copy fallback is used if the dependency is absent.
The library README documents standalone builds and installation.

The library's EN/RU architecture pages define its internal boundaries. Engine
documents retain model composition, file formats, request behavior and historical
measurements. Backend and execution pages here describe Velri-lib responsibilities
until their detailed specifications move with the corresponding implementations.
