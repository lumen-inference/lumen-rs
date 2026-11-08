# C/Vulkan compute kernels

[Русский](../RU/kernels.md)

This document explains how to move model computations into the new engine: the rules kernels must follow, how CPU and GPU variants are selected, and how results are checked. Terms are in the [glossary](glossary.md); the exact existing CPU implementation is described in the [GEMM/GEMV reference](kernel-reference.md).

**Status:** requirements and migration order. Measurements in [cpu-performance.md](cpu-performance.md) describe the existing Rust/C/LLVM path. They do not mean the new C engine is ready. Vulkan kernels are not implemented; research CUDA results are not results of the new backend either.

## Rules shared by all kernels

Each operation specifies its inputs, outputs and permitted memory access. The description includes array dimensions, number types, element order, strides along each axis and temporary-memory size. It separately defines which input and output regions may overlap and which must be disjoint. These rules tell the caller what to allocate and when a call is valid.

An operation has three separate parts: a simple reference for checking, a fast implementation and variant selection. CPU instructions and GPU capabilities are checked during preparation. Matrix dimensions can be arbitrary within the operation contract. When a width is not a multiple of the block size, the kernel handles the tail; SIMD convenience does not authorize reading or writing beyond the allocation.

A backend reports support through a query describing the complete operation, including shape, type, layout and state. A new kernel does not add another pointer to the common interface. The selected handler is retained in prepared execution, with no per-token lookup by a string name. The interface and shared checks are described in [backends.md](backends.md).

A support response specifies layout, alignment, temporary and persistent memory before final buffer allocation. At submission, the prepared implementation receives only the declared resources and bounded dynamic parameters.

The reference checks the formula and state changes. It may accumulate in FP64 even if the fast kernel uses FP32. Every operation defines numerical tolerance and behavior for NaN and infinities. Acceleration preserves these rules.

## CPU: a common computation description

Retain a common C/LLVM GEMM/GEMV generator as the basis of CPU execution. It describes computation and data traversal; target CPU parameters determine working-block sizes. LLVM IR is generated and compiled **at build time** for x86-64 or ARM64. The distributed engine executes machine code without JIT or a runtime LLVM dependency.

This approach carries one algorithm across architectures. Each architecture still needs its own machine-code files, and the compiler may choose different instructions. Runtime selection chooses only a variant supported by both the CPU and operating system.

Handwritten AVX2/AVX-512 implementations are allowed where they provide a measured benefit and follow the same operation rules. ARM uses the common computation description; do not duplicate an entire second algorithm family. The independent simple C reference must build without LLVM. LLVM is required on the machine building optimized generated variants.

AVX2 and AVX-512 are variants within the CPU backend, not separately registered backends.

Weight storage is a separate choice. An ordinary call retains the original element order. If prepared weights are selected, immutable weights are rearranged once during preparation and reused on subsequent calls. The common generator supports both modes: choosing prepared weights is not choosing between handwritten SIMD and LLVM.

Measure single-token GEMV, small batches and prefill GEMM separately. In each case, measure computation, memory access and the complete call including dispatch to the worker pool. Even a register-efficient inner loop can be surrounded by data preparation and worker-start overhead. [performance.md](performance.md) describes the method.

## Vulkan: start with a verifiable baseline

The first GPU kernel set must execute a small model end to end. It needs FP32 GEMM/GEMV, normalization, activations, residual addition, positional transforms and attention. Kernels use GLSL compute and compile to SPIR-V ahead of time. The host engine controlling their execution is written in C.

The baseline uses ordinary storage buffers and checked Vulkan 1.1 limits. Workgroup size, group-shared memory and buffer sizes must fit device capabilities. A subgroup cannot be assumed to have 32 or 64 participants: check its limits and required operations before choosing a kernel. See the [Khronos subgroup guide](https://docs.vulkan.org/guide/latest/subgroups.html).

Check FP16 storage and FP16 arithmetic separately. Cooperative matrices and integer dot product may provide additional accelerated variants. If unavailable, retain a correct baseline kernel; if the operation itself is unsupported, report that explicitly.

A backend may start with one operation. Only preparation may choose a declared equivalent decomposition or another permitted device, including required conversions and transfers. Explicit placement on an unsupported device fails; a kernel does not silently switch to CPU.

Choose pipeline variants and specializations during preparation and bound their count. Pipeline creation is excluded from ordinary-step timing: cold startup is measured separately. A disk pipeline cache requires compatible device and driver identities; incompatible caches are safely rejected.

## LTQ weights and attention

Migrating LTQ v1 must preserve its [byte specification](ltq-format.md). Start with a reference C decoder and independently prepared test data. CPU and Vulkan kernels can then decode weights during multiplication. Checks cover integer and fractional formats and incomplete final blocks.

A backend-specific element order needs a layout identifier, version and parameters. The old `GpuWarp` name does not prescribe Vulkan subgroup size. Changing the meaning of an existing disk layout requires a new version.

For attention, verify full-context and windowed execution first. Add QSA, DSA and recurrent mechanisms with explicit formulas and state, reusing projections and common operations.

K/V storage must match the model graph. If the graph declares another representation, a kernel cannot create both full K and V arrays merely for convenience. All temporary conversions count in the memory plan.

## Checking results and speed

The shared backend suite checks support and rejection, transfer round trips, immutable plan reuse, independent session slots, acceptance and completion errors, and resource lifetimes during cancellation. A test backend with delayed completion verifies these without a third production GPU backend.

Compare a new kernel with an independent reference on small and odd dimensions, different row strides, padded buffers, shared read-only inputs and consecutive model steps. Numerical agreement is insufficient: guard pages, sanitizers and boundary checks must detect extra reads and writes.

Enable API and synchronization validation in diagnostic Vulkan builds. Measure performance separately without those layers.

For a complete decoder, compare logits and state changes with an independent implementation. Compare speed with BLIS/OpenBLAS and available GPU implementations at equal task sizes and precision. Report preparation, copying and worker counts. Real model-step time determines the final result: a fast inner loop does not establish a fast whole engine.
