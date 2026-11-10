# Stack and libraries

[Русский](../RU/dependencies.md)

The compute build, CPU/Vulkan dependencies and kernel CI now belong to the separate Velri-lib project. Velri links its exported CMake target or a local checkout, without source copies; see [build instructions](velri-lib.md). The current prototype and future dependencies remain distinct.

This document describes libraries and tools for Velri's future move to C and Vulkan. Technical terms are explained in the [glossary](glossary.md); the [architecture](architecture.md) describes the engine itself.

**Status:** these libraries are not integrated yet. The list was checked against upstream documentation on 8 November 2026. **Proposed** means a choice for future implementation. **Candidate** means suitability still needs evaluation. The conditional VMA choice depends on the open C++ policy.

## Language and build

Engine code and public headers are planned in **C23**, using structs and functions. Every memory region needs clear ownership and responsibility for release. Internal structures do not have to become public ABI. Platform calls and compiler SIMD extensions stay in small dedicated modules.

The internal [backend interface](backends.md) also uses C23, opaque resources and function tables. This does not turn the application API into a stable ABI for third-party plugins.

**CMake and Ninja** are used by the first [C CPU linear module](cpu-linear.md); the full engine build remains to be implemented. CMake describes how to build the library, applications and tests, and how to generate compute kernels. Ninja executes those actions.

Set three properties for our C targets:

```text
C_STANDARD=23
C_STANDARD_REQUIRED=ON
C_EXTENSIONS=OFF
```

This requires C23 without enabling language extensions by default. Without `C_STANDARD_REQUIRED`, CMake may lower the requested standard. The C23 property became available in CMake 3.21, which is Velri's minimum CMake version. Configuration also checks that the compiler accepts `-std=c23`. CMake may emit the equivalent `-std=c2x` spelling for these compilers. [CMake documentation](https://cmake.org/cmake/help/latest/prop_tgt/C_STANDARD.html).

The C23 build uses Clang 18+ or GCC 14+. These releases accept C23 mode; this does not claim that every C23 feature and library facility is implemented by every supported toolchain. [Clang 18 release notes](https://releases.llvm.org/18.1.8/tools/clang/docs/ReleaseNotes.html#c23-feature-support), [GCC 14 release notes](https://gcc.gnu.org/gcc-14/changes.html).

CI specifies Clang 18 and GCC 14 on Linux, including sanitizers and a baseline-only build. The ARM64 job uses GCC 14 and QEMU. Windows Clang and Android NDK Clang builds remain planned and need their own toolchain checks.

Claim MSVC and additional ABI support after verification in CI. A successful build establishes that the program can be built; performance still requires a device measurement.

### Build-time and runtime requirements

Retain common build-time LLVM IR generation for optimized CPU kernels. Port the generator to C. The existing small helper that queries LLVM TTI may remain a C++ build tool.

**The distributed engine does not need LLVM, Rust or a shader compiler to run.** During cross-compilation, generators run on the build host and their output is compiled for the destination. For example, a generator may run on Linux x86-64 while its output is built for Android arm64.

## Small baseline dependency set

This is the proposed set for the first implementation. Build tools in this table do not automatically become dependencies of the running engine.

| Purpose | Proposed library or tool | Where it is used |
| :--- | :--- | :--- |
| Configuration and metadata JSON | [yyjson](https://github.com/ibireme/yyjson), MIT, C | During loading. Prepared execution uses checked structures without JSON objects on the hot path |
| Vulkan declarations | [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers), Apache-2.0/MIT by file | Building the Vulkan backend. Headers do not replace a driver |
| Vulkan function loading | [volk](https://github.com/zeux/volk), MIT, C | Windows, Linux and Android. Each device has its own function table; multiple GPUs must not overwrite global pointers |
| GPU kernel compilation | [glslc from shaderc](https://github.com/google/shaderc), Apache-2.0 | Compiles GLSL compute into SPIR-V at build time. Shaders ship with the library; the driver still creates Vulkan pipelines |
| SPIR-V validation | [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools), Apache-2.0 | Runs `spirv-val` during builds and CI, using the Vulkan target environment for which the shader was built |
| C tests | [Unity](https://github.com/ThrowTheSwitch/Unity), MIT | A small C assertion library. CTest runs tests; Python provides independent numerical fixtures |

A CPU build does not need the Vulkan SDK. Vulkan build headers and tools belong on the developer's machine. Users need a working Vulkan loader and driver: `volk` does not provide the driver.

Each backend is a separate build module with private dependencies. Disabled modules introduce no headers, libraries, or code-generation tools into the build, require no SDK, and add no libraries to the distributed engine. The enabled registration list is generated at build time; new names are not inserted manually into models or the planner. The common C interface and checks for this integration are described in [backends.md](backends.md).

When integrating shaderc, check licenses of all included components, not only shaderc itself.

## Vulkan memory and the open C++ decision

**The C++ dependency policy remains OPEN.** The GPU allocator choice is therefore conditional.

The preferred option is **[Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)**, or **VMA**, under MIT. It provides a C interface, but its implementation requires C++14. It helps select GPU memory types and distribute large allocations among buffers. Technical requirements are in the [VMA documentation](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/faq.html).

The next choice depends on project policy:

- **If C++ is allowed inside libraries**, integrate VMA as a private Vulkan-backend module. This is an internal implementation boundary in Velri; VMA's source remains open. Separately check whether the distributed binary requires a C++ runtime.
- **If C++ is limited to build tools**, a bounded Vulkan allocator written in C is needed. It must handle alignment, memory types, dedicated allocations and memory accesses that have not completed.

There is no plan to maintain two GPU allocators without a need. A VMA-based implementation would not be entirely C.

Velri determines model and request lifetimes; VMA and `malloc` do not. The backend receives a prepared memory plan and uses allocated regions during each step.

The common planner owns resource lifetimes and the total budget. The backend allocates physical memory and reports actual reservation, alignment and overhead. Driver memory without reliable reporting is explicitly marked unknown.

## Optional dependencies and candidates

These tasks may become necessary later. Before adding a library, test it against a real Velri use case.

| Purpose | Candidate or current decision | Considerations |
| :--- | :--- | :--- |
| Unicode | [utf8proc](https://github.com/JuliaStrings/utf8proc), MIT and Unicode data license | Normalization and character categories for tokenization. Match the Unicode version to the tokenizer behavior being verified |
| Regular expressions | [PCRE2](https://github.com/PCRE2Project/pcre2), BSD-3-Clause WITH PCRE2-exception | Only needed for required pre-tokenization rules. JIT is optional. Fixtures verify compatibility with the source tokenizer's regex dialect |
| HTTP/SSE | [CivetWeb](https://github.com/civetweb/civetweb), MIT | Candidate for a separate server application. Check streaming, backpressure and cancellation. Account for TLS dependencies separately |
| GEMM/GEMV comparison | [BLIS](https://github.com/flame/blis) or [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS), BSD-3-Clause | External control implementation for tests and measurements. Inference use requires a separate decision based on measurements |
| Threads and files | Small `platform/` layer over Win32 and POSIX/Android | Retain our persistent worker pool. Obtain cores and topology from OS data where available. No dependency on Ryzen or a particular CCD count |
| General allocation | System allocator by default, with a caller-supplied option | Replacing the allocator does not remove per-token allocations. Model, state and scratch have different lifetimes |

### Tokenizer and chat template

`utf8proc` and PCRE2 solve individual tasks; they do not provide a complete tokenizer. BPE/Unigram, added tokens, byte fallback and streaming decoding need our own implementation or a verified adapter.

A small template engine also cannot be assumed to execute every Hugging Face Jinja template. **No library is selected for tokenization or templates yet.** First inspect real metadata from the initial models and check exact token IDs.

Unsupported tokenizers or templates must produce a useful error. While that part is under development, the library can accept token IDs directly, allowing independent compute development.

### What is not part of the stack yet

CUDA, CUTLASS and FlashInfer are not part of the new baseline. Their ideas and results may inform comparisons. Vulkan execution requires corresponding GPU kernels; changing the host language does not turn a CUDA kernel into a Vulkan kernel.

The static backend registry does not need a dynamic plugin loader. A stable binary plugin ABI, a general asynchronous runtime and a container library are not introduced yet: each needs a concrete use case.

## Library integration rules

At the first integration, pin an exact release or commit, a checksum and license notices. Acquiring dependencies must be a separate explicit step. Afterwards, the project must build offline. The build does not download a moving `main` branch.

Start with upstream code without maintaining a fork. Dependency types must not appear in Velri's public interface. Translate errors into engine statuses while preserving useful diagnostic text.

Before integration, find out whether a dependency creates threads or stores global state. Refine the list and exact versions alongside a working build: the dependency table alone does not prove that they work together.
