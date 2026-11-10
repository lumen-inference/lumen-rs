# Migration to C and Vulkan

[Русский](../RU/roadmap.md)

The goal is a working, verifiable engine that people can contribute to through understandable C code. Technical terms in this plan are explained in the [glossary](glossary.md).

A language change does not itself establish acceleration. Verify correct results, memory use and complete execution time for every migrated part.

**What exists now:** the CPU linear prototype, references and tests have been extracted into the separate Velri-lib project. Velri consumes its pinned revision and checks the CMake/header boundary. The generic graph, executor, target F64/FP16/BF16/INT8 paths, Vulkan and real-model inference are not implemented. Earlier Rust results remain historical evidence. See [project boundaries](velri-lib.md).

## Order of work

These stages express dependencies. Some work can proceed in parallel: Vulkan experiments may start after buffer and basic-operation contracts, while format support can migrate alongside CPU computation. A real model can be connected before reaching maximum TFLOPS.

### 1. Minimal C library

Prepare CMake, error statuses, opaque objects, size and overflow checks, CPU references and Unity/CTest tests. Introduce a small internal [backend interface](backends.md) and build-time module registration alongside them.

The interface includes operation support queries, prepared implementations, acceptance and completion. A backend can initially support one operation, but a whole model needs a complete prepared plan. A third production backend is unnecessary for this check.

This stage is complete when the library builds for Windows, Linux and Android arm64 and the first test runs without a Rust runtime. Argument errors, ownership and correct resource release are verified.

A test backend with one operation and controllable delayed completion must connect without changes to models or the planner. Shared checks cover rejection before launch, failure after acceptance, cancellation, RAM transfers and session independence. CPU builds do not depend on disabled-backend SDKs. This tests module boundaries, not another production backend.

### 2. End-to-end CPU execution

Connect small weights, model blocks, `Program` and `Session` into a working synthetic decoder.

This stage is complete when prefill and decode work and logits and state agree with independent fixtures. Shared storage owners and derived K/V must work without extra allocations. The CPU module uses the common interface; session slots are independent and cancellation is safe.

### 3. Optimized CPU computation

Port common C/LLVM kernels, weight preparation and the worker pool.

This stage is complete after reference comparisons and guard-page checks. Test tails and concurrent calls. The hot path has no allocations. Execution profiles and complete GEMM/GEMV call measurements are available.

### 4. Loading a real model

Add configuration, LTQ and weight loading, a supported tokenizer and chat template, and `inspect` and `run` commands.

This stage is complete when token IDs and logits agree with an independent implementation and memory reports truthfully describe usage.

### 5. First Vulkan execution

Add device discovery and capabilities, GPU memory, SPIR-V, operations for the selected model formats and a small decoder. Vulkan uses the same interface as CPU and the test module. The common planner needs no new device-name branches.

This stage is complete after numerical comparison with CPU and validation checks. Measurements are needed across GPU families and on Android. Missing device capabilities must be reported correctly.

### 6. CPU and GPU working together

Add CPU/GPU data and compute placement, the common RAM route, explicit conversions and operation dependencies. Then connect disk-table loading.

This stage is complete when changing placement preserves results within numerical tolerances. Check rejection for insufficient support, permitted alternative selection during preparation, cancellation, peak memory and crossing latency at p50 and p99.

### 7. Broader computation support

Add further attention and recurrent mechanisms, MoE, LTQ fast paths and fusion with measured benefits.

This stage is complete when each mechanism and the whole model are tested. Common blocks have no model-family branches.

### 8. Applications and multiple requests

Develop CLI and chat, HTTP, batching, prefix reuse, speculation and media support.

This stage is complete when sessions are isolated, queues are bounded and capacity is controlled. Partial acceptance and rollback are tested.

## What survives code replacement

The current Rust code may be removed during the transition. What matters long term is the knowledge and verifiable results it helped establish:

- LTQ v1 and external fixtures;
- common mathematical block contracts;
- storage-owner accounting rules;
- verified numerical tests and independent reference data;
- measurements identifying the implementation they describe.

Old sources may be archived for reference if useful. The plan does not require continuously maintaining a runnable Rust version until its replacement exists. Before deletion, preserve knowledge, test scenarios and data for checking the new implementation. Archived Rust measurements must not be renamed as C/Vulkan results.

Do not automatically port twenty build units, unused settings, CUDA protocols and additional barriers. Check a concrete use case before fixing each C contract.

Pin exact library versions with a working build. Proposals and open decisions are listed in [dependencies.md](dependencies.md).

The C++ dependency policy and conditional VMA choice remain open. Dynamic backend loading and a stable binary ABI are separate future decisions.

## Separate quantizer work

Calibration and native quantization develop separately from applications. They need model loading and execution, but must not depend on HTTP or the serving scheduler.

Until a native backward path exists, verified research profiles can be imported. Modular forward execution does not mean backward is implemented. Format and quantization details are in the [LTQ design](ltq-design.md).
