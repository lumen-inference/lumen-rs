# Adding a compute backend

[Русский](../RU/backends.md)

**Project ownership:** the computation contracts and implementations on this page belong to **Velri-lib**. Velri uses them through the library API; it owns model and request policy. See the [project boundary](velri-lib.md).

A backend executes operations on a device. Velri-lib's first two backends are **CPU** and **Vulkan**. The library manages their registration, settings, devices and execution; Velri uses its common C API. The engine and internal backend interface use C23. Vulkan GPU kernels use GLSL and ship as SPIR-V.

**Status: interface design.** The C types and `src/` layout below are not implemented. They define proposed boundaries for the new engine, not the current Rust API. Terms are defined in the [glossary](glossary.md).

The central requirement is that adding a backend for existing operations must not require changes to model descriptions, operation formulas or the common planner. New mathematics may require a new common operation; a new device alone does not.

## What a backend author implements

In the **Velri-lib** repository, add a directory such as `src/backend/example/` with five parts:

1. **Registration and device discovery:** the backend's name and its available devices.
2. **Memory:** allocation and release of its buffers, with transfers to and from ordinary CPU memory.
3. **Support queries:** the concrete operations, dimensions, types and layouts the device can execute.
4. **Preparation and execution:** selection of its kernels and execution of prepared jobs.
5. **Contract tests:** numerical correctness, memory bounds, errors and resource lifetimes.

Connect the registration to the library build. Common preparation then discovers the module through the Velri-lib registry. Nothing needs registration in the engine. No `if backend == example` branch belongs in the engine's `model/` or the library's common `plan/` and `runtime/`.

This reduces the surrounding integration work. Fast kernels still need implementation and verification: the interface does not translate GLSL into code for any other accelerator.

## Shared and backend responsibilities

| Shared Velri-lib responsibility | Backend responsibility |
| :--- | :--- |
| Operation formulas, tensor shapes and permitted state changes | Concrete implementations of operations |
| Operation graph validation and launch graph preparation | Selection of its instructions or GPU kernels |
| Placement across permitted devices | Checking whether its device supports the proposed work |
| Resource lifetimes, total memory budget and job dependencies | Physical allocation and execution of dependencies through device facilities |
| Execution slots, accepted-work tracking and completion on cancellation | Running accepted work and reporting its actual completion |
| Diagnostics and common tests | Rejection reasons, device properties and measurement data |

Model assembly from blocks, inference sessions, batching policy and token selection remain in Velri. The engine specifies permitted devices and resource limits through the library API; Velri-lib enforces those limits during graph execution.

A backend does not read model-family configuration or infer QSA, GDN or MoE semantics from a model name. It receives operations and parameters. Each module in Velri-lib validates its device settings, such as CPU thread count, when creating the context.

## One C interface with several function groups

A module has a registration record. An open device has a function table and a pointer to private data. These are ordinary C structures: they do not require class inheritance, global constructors or a plugin for every operation.

| Group | Example responsibilities |
| :--- | :--- |
| Registration | Name, internal interface version, validation of module settings, device enumeration |
| Device | Opening, properties and limits, closing after dependent objects are released |
| Memory | Size and alignment checks, allocation, release, preparation of transfers |
| Preparation | `query_operation`, preparation of supported operations, release of prepared implementations |
| Execution | `submit` of prepared work with buffers and parameters for the current step |
| Completion | `poll` or `wait`, work status, release of completion records |

These names describe roles; the first implementation will fix exact signatures. Common helpers can provide an already-completed result for a synchronous CPU call, the registration list and rejection diagnostics. Each backend need not implement another session system or duplicate these helpers.

The table has no separate `gemm`, `rope`, `qsa` or similar entry for every kernel. A common descriptor identifies the operation, and preparation selects its handler. Adding an operation does not expand the backend function table. Its validation and reference are added to the common operation registry; each backend either supports the new kind or explicitly rejects it.

Internal backend headers belong in the Velri-lib repository's `src/backend/`. The engine uses the library's public headers under `include/velri_lib/`; its own model and request API lives in `include/velri/` in the Velri repository. The common interface exposes neither `VkBuffer`, API-specific queues nor SIMD types. Initially all backends build from the same library source version. A stable ABI for third-party binary plugins is not promised.

## Querying support for an operation

“Does this device support GEMM?” is insufficient: a device may support only particular types, dimensions or storage representations. `query_operation` therefore receives a concrete contract:

- The operation kind and its parameters.
- Input and output shapes and types.
- Strides, layouts and permitted aliasing.
- Memory kinds and resource properties.
- State reads and changes.
- Bounded dimensions and indices that may vary between steps.
- Required numerical behavior.

The query performs no computation and does not mutate the model. It returns a rejection reason or a supported implementation with conditions: required layout, alignment, temporary and persistent memory, and size limits. Preparation can obtain these requirements before final buffer allocation. When several implementations are available, it considers them within the configured search budget.

For example, a Vulkan kernel may accept FP32 matrices but require a different packing for constant weights. Preparation adds a one-time conversion and budgets its memory. A backend must not report support and then silently create a full weight copy on every call.

Common code validates the operation itself; the backend checks its own constraints. Support and preparation must agree for the same device and descriptor. Actual allocation can still fail, and a device can be lost; these remain reported errors.

An unknown performance estimate remains unknown. A backend need not invent a time estimate, and support does not imply that its implementation is faster than CPU execution.

## Model preparation

Preparation runs once within the chosen limits:

1. Common code validates the graph, shapes and state accesses.
2. It queries implementations from devices permitted by the user.
3. It selects placement, layouts and necessary transfers or conversions.
4. It computes lifetimes and total memory, including weight packing and temporary data.
5. Each backend prepares its operations; common code connects them into execution jobs.
6. It creates session buffers and reserved slots for work that may run concurrently.

An operation is the basic unit of support. Initial backends can prepare operations separately and assemble a job sequence. A backend may also offer a fused implementation of a suitable graph region. This is optional: the region must preserve external inputs and outputs, state changes, and intermediate results required elsewhere in the graph.

Compiling the entire graph is not a mandatory backend method; that would make every module repeat the planner. The interface nevertheless permits submitting prepared operations in batches and reusing commands.

For an unsupported operation, common code may choose a declared mathematically equivalent decomposition or another permitted device. Explicit placement on an unsupported device fails preparation. Rejection does not authorize a backend to run CPU code secretly or alter the formula.

## Memory and state

A memory resource has an owner, a device and an object opaque to common code. A tensor refers to that resource with an offset, shape and strides. A GPU resource is not presented as an ordinary pointer that CPU code can dereference.

The common plan determines required regions, their sizes and reuse points. The backend performs physical allocation under device constraints. It reports actual reservation, alignment and overhead so the budget counts more than useful elements. Driver memory without a reliable accounting interface is identified separately as unknown.

K and V remain input roles. A backend does not allocate a pair of caches automatically or inspect the model name. Shared resources, recurrent state and deriving one value from another must be visible in the graph and memory plan.

Prepared weights are immutable and tied to the source owner and data version, device, implementation and packing format. Compatible plans may share a representation; matching matrix dimensions alone is insufficient. Budget the source and every representation needed simultaneously. Changed weights require new preparation or explicit invalidation of the previous preparation.

Several sessions may share an immutable prepared plan. Workspace, mutable parameters and commands that cannot safely be shared belong to a session or execution slot. Finishing one step does not permit freeing weights while live sessions still depend on them.

Direct CPU access is optional. Available mappings need explicit rules for beginning and ending access. The backend performs required memory-visibility operations; model code does not own this task. Importing external memory is also optional and does not transfer its ownership without an explicit agreement.

## Transfers without an adapter for every backend pair

A new backend must not need a separate transfer implementation for every existing backend. The baseline route uses ordinary CPU memory:

```text
device A resource → RAM staging buffer → device B resource
```

Each backend supports transfers between its resources and an agreed host buffer. Common code connects the two paths. A backend decides whether it needs additional pinned memory; the plan must know that memory's size and lifetime. CPU execution can use a compatible buffer directly.

Copying bytes does not convert a layout. Transfers use a representation agreed during preparation. A private representation needs an explicit supported packing or unpacking path before another backend can consume it. Without such a path, preparation rejects that placement. The same rule applies to state whose representation cannot simply be copied.

The producer must finish its writes before transfer and consumer access. A RAM buffer cannot be reused while either transfer still uses it. Execution limits bound staging capacity and the number of outstanding buffers.

Direct device copies, shared memory and external-object import may become optional faster routes. The corresponding transfer module checks compatibility. Their absence neither requires planner changes nor prevents the baseline RAM route.

## Submission and completion

`submit` receives a prepared job, the current session's resource references, bounded dynamic parameters and dependencies. The hot path does not look up backend names, repack immutable weights or allocate an object for every scalar operation.

Submission returns an **acceptance status** and, when work is accepted, a **completion object**. Even if an error is discovered after some work has been accepted, the engine needs a way to wait for all resource accesses already launched. Returning only an error and losing track of that work is forbidden. Rejection before acceptance means nothing was launched and no supplied temporary data remains borrowed.

Completion means the work has finished using its borrowed inputs, parameters and workspace. It also carries the final status. It does not automatically make GPU memory accessible to CPU code: required transfers and visibility operations belong in the prepared access path.

A synchronous implementation returns an already-completed object. For asynchronous execution, `poll` reports progress without waiting; `wait` waits for the relevant work without stopping every device. Completion records and mutable data come from preallocated slots and remain live through their last user. A record need not map to a separate device fence for every operation; the backend can track completion of submitted batches.

Dependencies within one backend may use its own facilities. A dependency across backends follows the common order: finish the producer, perform the required transfer, then provide input to the consumer. Optional inter-device synchronization may remove a CPU wait after support is checked. Neither an ordinary C atomic nor matching memory addresses proves such support.

Cancellation before any work of a step is accepted leaves the previous session state intact. After the first job is accepted, cancellation stops future submissions, drains accepted work and leaves the session requiring explicit reset. The first implementation promises neither rollback nor continuation from a partially executed prefix, even when it cannot prove that state was written. The runtime commits one terminal outcome; cancellation arriving after successful completion does not retroactively invalidate the result.

A backend need not forcibly interrupt a running kernel. A failure that affects state makes the session unusable until a defined reset. Device loss invokes the device's shutdown and release procedure. Until safe release is established, resources are not reused and affected sessions remain failed.

## Applying the contract to CPU and Vulkan

| Shared responsibility | CPU | Vulkan |
| :--- | :--- | :--- |
| Device | CPU context and OS-available cores | Open Vulkan device with checked capabilities |
| Constant weights | Source or prepared layout for the selected kernel | Buffers and layouts supported by the GPU kernel |
| Preparation | SIMD/LLVM variant, worker ranges and workspace | Shader selection, pipeline creation, resource and command preparation |
| Execution | Direct call or jobs for a persistent pool | Submission of prepared commands to queues |
| Completion | Function return or worker completion | Vulkan completion of submitted work |
| Transfers | Direct access to compatible memory or copying | Prepared copies and visibility operations through host memory |

AVX2 and AVX-512 are kernel variants within the CPU backend, not separate backends. CPU execution uses a common C/LLVM generator across architectures. An independent simple reference remains available without LLVM at runtime.

Vulkan translates common dependencies into the required barriers, waits and memory operations. The Vulkan 1.1 baseline uses facilities available at that version; timeline semaphores and synchronization2 require checked support. Calling `vkQueueWaitIdle` after every operation is not the plan's general execution model. The [Khronos synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html) illustrate dependencies, but use synchronization2 and do not define Velri's minimum Vulkan version.

## Registration, build and configuration

Proposed layout within the Velri-lib repository:

```text
src/backend/
    interface.h        common internal contract
    registry.c         lookup of included registrations
    common/            helpers and the shared RAM transfer route
    cpu/               registration, memory, preparation, CPU kernels
    vulkan/            registration, memory, preparation, shaders
    example/           a future backend under the same contract

tests/backend/         common contract tests for every module
```

The registration list is generated from modules explicitly enabled at build time. An author adds a directory and its build description; an explicit reference includes its registration function. Registration does not depend on global constructors or a linker accidentally retaining an otherwise unused object file. A small generated build manifest also names known modules that are disabled; it is separate from enabled registrations, so diagnostics can distinguish a disabled module from an unknown name without linking its implementation.

CPU builds require no Vulkan SDK. A disabled backend does not add its headers, libraries or tools to mandatory dependencies. Applications use included modules through the common C API. The C++ dependency policy and conditional VMA choice are discussed separately in [dependencies.md](dependencies.md).

Configuration identifies a device as `<backend-name>:<index>`, such as `cpu:0` or `vulkan:0`. Names are checked against the registry, not a closed list of GPU APIs in the parser. Module settings live under `backends.<name>`; the module validates them once and converts them into internal parameters. Kernels do not receive model JSON.

Device storage uses the same unambiguous name, for example `store: "vulkan:0"`. `ram:0` and `disk` are stores, not compute backends. The earlier proposed `device:0` spelling is replaced because its meaning would be ambiguous with several backends. See the full example in [config.md](config.md).

An index is meaningful within a run; device ordering is not promised to remain stable between runs. `inspect` reports actual device identity, backend, applied settings and selection reason. Diagnostics distinguish an unknown module, a module disabled at build time and the absence of a suitable device.

Loading `.dll` or `.so` modules dynamically is a possible extension with separate ABI and versioning rules. Adding a source module now must not depend on resolving that future design first.

## Starting small and testing extensibility

A new backend can initially support transfers and one operation, such as FP32 Linear. That is enough to run its operation tests. A whole model can run only after every required operation has a prepared implementation on that backend or other devices allowed by the placement mode.

If a mixed plan is allowed, its transfers and costs appear in the report. Moving an unsupported operation to CPU midway through a running step is forbidden: state, formats, memory and dependencies must be agreed in advance.

Common tests cover:

- Support and rejection for concrete shapes, types, strides and layouts.
- Numerical agreement with an independent reference, tails and memory bounds.
- Shared owners and absence of implicit K/V allocations.
- RAM round trips, including irregular sizes and conversions.
- Plan reuse with changing data and independence of multiple sessions.
- Rejection before acceptance, failure after acceptance, waiting and cancellation without early release.
- Absence of repeated preparation and undeclared hot-path allocations.

A small test backend with deliberately limited support and controllable delayed completion verifies the interface itself. It does not promise support for another GPU API. If connecting it requires editing models or adding backend-name branches to the planner, the boundary is wrong.

Implement the common contract and test module first, then CPU, Vulkan and the shared CPU/Vulkan path. A third production backend is not required just to validate extensibility.

## Why this boundary

A function pointer per kernel is simple for the first few operations but expands with new mathematics and poorly expresses different memory formats. Requiring each backend to compile the whole model duplicates planning and state management. This design keeps the common plan and operation descriptors, while backends own prepared implementations and optional fusion.

The [GGML backend interface](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend-impl.h) also separates registration, devices, buffers and execution. It informs responsibility boundaries; this is neither a compatible API nor a copy. Velri's contract retains explicit state ownership and preparation of repeated work.
