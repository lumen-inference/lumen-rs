# Glossary

[Русский](../RU/glossary.md)

This file is a reference for the Velri documentation. You do not need to read it from beginning to end before learning about the engine: return here when you encounter an unfamiliar term. English names are retained so they are easy to find in code and other documents.

Sections: [model and text](#model-and-text), [model blocks](#model-blocks), [engine structure](#engine-structure), [memory](#memory), [kernels and numbers](#kernels-and-numbers), [CPU threads](#cpu-threads), [Vulkan](#vulkan), [measurements](#measurements), [builds and applications](#builds-and-applications).

## Model and text

**Inference** — running an already trained model on new input. Ordinary text generation is inference; training weights is a different task.

**Model architecture** — the structure of its computations: which parts exist, their execution order, and how they pass data to one another. **Weights** are numerical parameters obtained during training. One architecture can use different weight sets.

**Tensor** — an array of numbers with specified dimensions. A vector is a one-dimensional tensor, and a matrix is two-dimensional. For example, shape `[3, 4]` denotes a matrix with three rows and four columns.

**Shape** — a tensor's dimensions along its axes. **Dtype, data type** — the representation of one element, such as FP32 or an 8-bit integer. Together, shape and type determine the data size before any additional memory padding.

**Token, token ID** — a unit processed by a language model and its vocabulary index. A token can represent part of a word, a whole word, punctuation, or a special marker. One token does not necessarily equal one character.

**Tokenizer** — converts text to token IDs and back according to a particular model's rules. **Chat template** — formats roles and messages before tokenization, adding separators, special markers, and other required elements.

**BPE and Unigram** — different methods of building a vocabulary and splitting text into tokens. **Added tokens** define extra tokens with special rules. **Byte fallback** represents content unsupported by token segmentation through bytes. Compatibility depends on the model's exact set of rules.

**Context** — the tokens a model considers in the current computation. In “execution context,” the word means something else: engine resources, described below as `Context`.

**Prefill** — processing the input token sequence, either all at once or in chunks. **Decode** — subsequent continuation steps; in the simplest case, each step processes one new token.

**Logits** — scores for possible next tokens before conversion to probabilities. **Sampling** in generation is the rule for choosing a token from those scores. In profiling, sampling has a different meaning.

**Activations, hidden states** — intermediate values produced as data passes through a model. They depend on the current input; weights usually remain unchanged.

**Embedding** — a numerical vector representing a token or another input object. An embedding table provides such a vector by index. **Output head, prediction head, lm_head** transforms the model's result into the required output; for a language model, this often means logits over the vocabulary.

**Quantization** — representing numbers with less storage, usually at some loss of precision. A **quantizer** chooses such a representation; **weight decoding** reconstructs values from it for computation. The latter should not be confused with the decode stage of text generation.

**LTQ** — the weight quantization and storage format in Velri. The [LTQ v1 specification](ltq-format.md) defines the exact rules for reconstructing numbers and arranging bytes.

**Calibration** — using selected data examples to estimate a model's sensitivity to quantization and choose its parameters. **KLD, KL divergence** — a measure of difference between probability distributions; LTQ research uses it to compare outputs of quantized and original models.

**Forward and backward** — computing a model's result and computing derivatives in reverse to change parameters. Forward support alone does not provide backward support.

## Model blocks

**Operation** — a computation with defined inputs, outputs, and state effects. A **block** connects operations and other blocks. A **layer** is a larger model component assembled from blocks. [blocks.md](blocks.md) explains their boundaries and gives examples.

**Linear projection, Linear** — multiplication of an input by a weight matrix, sometimes with an added `bias`. It transforms an input vector into a new set of numbers. Code usually keeps the name Linear even when bias is present.

**Activation function** — a nonlinear function applied to values, such as SiLU or GELU. This should not be confused with “activations” as the name for intermediate data.

**MLP, FFN** — a feature transformation block, usually containing several linear projections and activation functions. A **dense MLP** applies its weight set to every input. In a **gated MLP**, one computation branch controls another branch's contribution element by element.

**MoE, mixture of experts** — a block with several experts. A **router** chooses which experts process a particular input and the weights used to combine their results. An expert is often an MLP. **Top-k** means selecting the `k` elements with the highest scores.

**Attention** — a mechanism that computes the contribution of available positions and combines their data. **Q, query** participates in the query; **K, key** in scoring matches; **V, value** in forming the result. These are roles of values, not a requirement to allocate three independent storage regions.

**Attention head** — one of the parallel parts of attention computation, with its own vectors. **GQA, grouped-query attention** allows several query heads to share key/value heads.

**Attention mask** — a rule excluding certain positions. A **causal mask** prevents a token from reading future positions.

**Full attention, FA** — attention over all positions permitted by the model's other rules. **Sliding-window attention, SWA** limits access to a window of positions. In these documents, FA means full attention; FlashAttention is named separately.

**Sparse attention** — a general name for attention methods that access selected positions or blocks. **QSA** means Qwen Sparse Attention; **DSA** means DeepSeek Sparse Attention. They are different mechanisms, rather than two names for one operation. An **indexer** computes scores or indices used to select data; its own work may still require a full history scan.

**FlashAttention, Flash-style kernel** — a way of organizing attention computation in tiles to reduce memory access and intermediate storage. This method alone does not specify which positions may be considered.

**RoPE** — a positional transformation of vectors using rotations of component pairs. The model defines its parameters and which inputs it applies to.

**Normalization, RMSNorm, QK-norm** — transformations of the scale of values according to a specified formula. RMSNorm uses the mean square of components; QK-norm denotes normalization of Q and K. Exact formulas, weights, and application points must come from the model description.

**Residual connection** — a path along which data bypasses some computation and is then combined with its result. A common form is `output = input + block(input)`, but more complex schemes are possible.

**Expanded residual, GR Read / GR Write** — elements of the discussed scheme with several residual-state streams. Read obtains a block's input from that state; Write puts the result back. These are separate computations with their own rules, rather than another name for ordinary residual addition.

**Recurrent state** — data updated while processing a sequence and used at the next step. **GDN, Gated DeltaNet** is a mechanism with such an update. Its own state must be described rather than automatically creating a KV cache.

**KV cache** — stored data from which attention obtains K and V for earlier positions. The name does not require two separate arrays: the model defines the storage arrangement.

**PLE, per-layer embeddings** — additional input representations for model layers. **N-gram embedding** uses groups of adjacent tokens. Their structure depends on the architecture; matching general names do not guarantee matching formulas.

**MTP, multi-token prediction** — a mechanism for predicting several future tokens. **Speculative decoding** first proposes a continuation and then verifies it with the main model. Accepted and rejected tokens require explicit state management.

## Engine structure

**Backend** — the engine component that executes operations on particular hardware. The target Velri design includes a CPU backend and a Vulkan GPU backend.

**Backend registration and registry** — a description of a connected module and the common list of those descriptions. The engine uses the registry to find a backend's devices and functions. A backend can provide several devices; for example, one Vulkan module can open several GPUs.

**Support query** — checking whether a device is suitable for a particular operation, its sizes, types, and storage arrangements. The response contains limits and memory requirements or a rejection reason. This does not execute the computation.

**Prepared execution** — selected kernels and saved settings that let a backend execute the work. Mutable data of individual sessions is kept separately from the common immutable plan.

**Work acceptance and completion object** — confirmation that a backend has accepted a job, and a way to learn when it has finished and with which status. Successful submission does not yet mean successful computation. Even after a later error, resource access must cease before those resources can be released.

**Static backend integration** — including its registration in the engine build. A **dynamic plugin** is loaded as a separate binary at startup and needs additional compatibility rules. Adding a source module does not require a dynamic plugin.

**Offload** — moving data storage or some computation to another resource, for example from the GPU to the CPU or disk. Data access methods and necessary transfers are specified separately.

**Kernel, compute kernel** — an implementation of a particular computation, such as matrix multiplication. Here, “kernel” does not mean an operating system kernel, and a compute kernel is not a physical processor core.

**Computation graph** — operations and connections between their inputs, outputs, and state. The connections specify what must be ready before the next computation begins.

**Operation contract** — precise requirements and guarantees: permitted dimensions and types, result semantics, memory access, and error behavior. An optimized variant must obey the same contract.

**Preparation, prepare** — validating a model and creating an execution plan: selecting kernels and calculating memory and repeated parts of jobs. The **runtime** uses this plan to execute steps.

**`Context`** — a proposed C API object owning devices, the thread pool, and platform resources. **`Model`** stores the model description and weights. **`Program`** contains the prepared plan. **`Session`** holds the mutable state of an individual session. These are target roles; the exact public C API has not yet been fixed.

**Execution step** — one call processing the next part of a sequence. It can process several tokens; a step does not necessarily equal one token.

**Hot path** — code executed frequently while the model runs. Repeatedly preparing unchanged data is especially costly here.

**Fusion** — executing several operations together in one kernel or job. **Overlap** — independent work overlapping in time, such as data transfer and computation.

**Batch, batching** — a group of inputs processed together, and the formation of that group. Depending on context, these may be tokens of one request or several requests; the specific meaning is stated alongside the size.

**Prefix reuse** — reusing computed state for a common beginning of sequences. **Commit** confirms new state, **rollback** restores previous state, and **reset** resets the session. These actions have different meanings and are not interchangeable.

## Memory

**Buffer** — a memory region for data. A **resource** is an object that must be created, used, and released at the appropriate time; a buffer is one kind of resource.

**Owner** — an object responsible for a resource's lifetime. **Borrowing** grants access to someone else's data for an agreed duration without transferring ownership.

**View** — a way to read an existing region with a specified shape, offset, and strides. A view does not itself create a copy. **Aliasing** means that several accesses touch the same memory.

**Layout** — the arrangement of elements in memory. In **row-major** layout, elements of a row are contiguous. **Stride** specifies the distance between neighboring elements along a chosen axis; the contract must state the unit. In our linear kernels, `ld_y` is the output row stride in elements.

**Packing** — rearranging data into an order convenient for a kernel. It does not necessarily compress data. **Prepared weights** — immutable weights in a representation prepared in advance, compatible with the selected kernel and reused across calls.

**Padding** — extra elements or bytes to meet size and alignment requirements. For example, a row of 17 numbers can be padded to 32; these additions occupy real memory even though they are not model outputs.

**Alignment** — a requirement that an address be a multiple of a certain value. For example, an address aligned to 64 bytes is divisible by 64 without a remainder.

**Allocator** — a mechanism for allocating and freeing memory. The **stack** stores function call data, including some temporary arrays. The **heap** is used for dynamically allocated regions. **Scratch** is a kernel's temporary working memory; its placement and lifetime are specified separately.

**Staging** — intermediate buffers for data transfer, for example between the CPU and GPU. **In-flight** describes work that has started but not yet finished. Its buffers cannot be released or reused before their last access.

**RAM, DRAM** — a computer's main memory and a common technology used to implement it. **VRAM** usually refers to graphics card memory. **UMA** — a system arrangement in which the CPU and GPU share physical memory; synchronization rules still apply.

**CPU cache, L1/L2/L3** — small, fast memory near the cores that retains recently used data. A **cache line** is the block in which a processor typically transfers and maintains coherence of cached data. The **page cache** is an OS-managed file cache in main memory; this is a different storage level.

**NUMA** — a system arrangement where the cost of memory access depends on which core accesses it. Binding threads to cores does not by itself specify where memory pages are placed.

**Zero-copy** — a path that avoids a particular intermediate copy. It does not promise the absence of transfers inside the hardware or greater speed for every access pattern.

## Kernels and numbers

**GEMM** — matrix multiplication; **GEMV** — matrix-vector multiplication. Our linear layer uses the more specific form `Y = X × Wᵀ + bias`. Its dimensions and constraints are described in the [kernel reference](kernel-reference.md).

**Dot product** — the sum of pairwise products of two vectors' components. An **outer product** forms a matrix of all pairwise products. Both can be used as parts of matrix multiplication.

**Tile; panel** — parts of a matrix that a kernel processes together. A **tail** is the remainder that does not fill a whole tile or SIMD vector. Handling it must not read beyond the array bounds.

**SIMD** — executing one instruction on several numbers at once. **ISA** — the processor's instruction set. AVX2 and AVX-512 are examples of x86 extensions; availability is checked before use.

**Intrinsics** — special compiler functions that express hardware instructions in C or another language. They provide more control than an ordinary loop, but tie that code to particular CPU capabilities.

**Shuffle** — a rearrangement of SIMD vector components. A **register tile** is a small result tile that a kernel keeps in registers during accumulation.

**Register** — a small storage location directly inside the processor. An **accumulator** in a kernel holds a running sum. A **horizontal reduction** combines several SIMD vector components into one result, for example by adding them.

**FP32, FP64, FP16, BF16** — floating-point formats. The first three use 32, 64, and 16 bits respectively; BF16 also uses 16 bits, but distributes them differently between precision and range. Support for storing a format does not guarantee arithmetic support for it; these capabilities are checked separately.

**FMA** — the operation `a × b + c` with one rounding of the result. Performance calculations usually count it as two arithmetic operations per element.

**NaN, Inf, nonfinite** — special floating-point values: NaN denotes a not-a-number result, and Inf denotes infinity; both are nonfinite values. Their permitted behavior must be defined separately.

**Bitwise equality** — the stored bits must match, including the sign of zero. A small numerical error is still a failure.

**NaN payload; canonical NaN** — NaN has several possible bit patterns. The payload is part of those bits; a canonical NaN is one explicitly chosen representation.

**Subnormal; gradual underflow** — very small nonzero floating-point values with reduced precision, and keeping them instead of abruptly replacing them with zero.

**Round-to-nearest, ties-to-even** — choose the nearest representable number; if two are equally close, choose the one whose last significand bit is zero.

**LLVM IR** — an intermediate program representation from which LLVM generates machine code for a selected platform. **TTI** provides the compiler with information about the target architecture's capabilities and operation costs.

**Target triple** — a string describing the compiler's target platform, including its architecture and OS or environment information. **Baseline CPU** in a build profile means the platform's baseline capabilities. Such code can still use SIMD; baseline does not necessarily mean scalar computation.

**`restrict`** — a C promise about which pointers are used to access memory. It helps optimization but requires obeying aliasing rules. It cannot be added to every kernel input when the contract permits those inputs to read shared memory.

**AOT** — generating machine code in advance, at build time. **JIT** — generating it while the program runs. Velri's target CPU kernel generator uses AOT, so LLVM should not be a runtime dependency of the engine.

**Reference** — a simple implementation used to check results. **Fallback** — an alternative execution path used only when selection policy permits it. **Dispatch** in CPU code often means selecting or invoking an implementation; in Vulkan, it also means a command that launches computation.

## CPU threads

**Thread** — a sequence of instructions scheduled by the OS. A **worker** is a pool thread executing engine jobs. The **caller** is the thread that called a function. The caller does not necessarily participate in arithmetic.

**Thread pool** — threads created in advance and given successive jobs. A **callback** is a function passed to other code for later invocation. If it uses borrowed data, that data's lifetime must cover every call.

**SMT** — running several hardware threads on one physical core. They share core resources, so doubling the number of logical CPUs does not mean doubling speed. **CCD** is the name of a compute chiplet in some AMD processors; the engine must discover topology from the system rather than infer it from a CPU model name.

**Affinity** — restricting the set of CPUs on which a thread can run. **CPU topology** describes relationships between logical CPUs, physical cores, and shared caches.

**False sharing** — unnecessary cache coherence work when threads modify different data in the same cache line. There may be no logically shared variables, yet a slowdown still occurs.

**Mutex** — a mechanism allowing only one thread to execute a protected region at a time. **Thread-local, TLS** — data separate for each thread.

**Atomic; Release, Acquire, AcqRel** — operations and rules for ordering access from different CPU threads. Release can publish data, and a corresponding Acquire read can make it available to the reader. AcqRel combines both sides. This is not universal CPU/GPU synchronization.

**Spin, park, unpark** — actively waiting in a loop, suspending a waiting thread, and waking it. Active waiting can reduce latency for a short task, but consumes CPU resources.

**Generation, epoch; ABA** — the number of a new job publication and the risk of confusing old state with new when a visible value repeats. A pool protocol must distinguish different jobs even when the worker set is the same.

## Vulkan

**Vulkan** — an API for working with GPUs. Velri focuses on its compute functionality. A **shader** is a GPU program; **SPIR-V** is the intermediate format in which such a program is supplied to Vulkan.

**Loader and driver** — the loader finds available Vulkan implementations, while the driver operates the device. A function-loading library such as volk does not replace the driver.

**Feature, capability, limit** — a supported device capability or restriction. These must be queried and checked; Vulkan availability does not imply support for every shader.

**Storage buffer** — a buffer accessible to a shader for reading and, when permitted, writing. A **descriptor** describes a resource supplied to a shader. A **pipeline** combines a shader with its execution settings.

**Workgroup and subgroup** — a group of shader invocations and a subgroup within it. They have different cooperation rules. Subgroup size depends on device capabilities and operating mode; it cannot always be assumed to be 32 or 64.

**Cooperative matrices, integer dot** — additional capabilities for cooperative matrix computation and integer dot products. An accelerated kernel may require them; the baseline path must state its own requirements separately.

**Command buffer, queue, submission** — a recorded command sequence, a device queue, and sending commands for execution. A **queue family** groups queues with particular capabilities; moving resources between families may require ownership transfer.

**Fence and semaphore** — execution synchronization mechanisms. A fence lets the CPU wait for device work; a semaphore connects queue work. A **timeline semaphore** uses increasing values instead of one binary signal and requires corresponding Vulkan support.

**Memory barrier** — establishes a dependency between memory accesses and the necessary visibility rules. Waiting in time is insufficient if it does not give the consumer access to current data.

**HOST_VISIBLE, device-local** — Vulkan memory properties: the ability to map memory for the CPU and placement preferred by the device. They can coexist; the names alone do not prove that they describe two physical kinds of RAM.

**Coherent memory; flush and invalidate** — properties and actions for keeping memory contents consistent between the CPU and device. With noncoherent memory, CPU writes are flushed when required, and memory is invalidated before reading device results. Ranges account for `nonCoherentAtomSize`. Completion and work ordering must be ensured separately, even for coherent memory.

**Dedicated allocation** — allocating memory for an individual resource according to device requirements or preferences. **Target environment** — the environment version and capabilities for which SPIR-V is built and validated.

## Measurements

**FLOP and TFLOPS** — a floating-point arithmetic operation and a trillion such operations per second. For our linear layer, the main work is counted as `2 × N × M × K`; units and any additional operations counted must be explicit.

**Latency; throughput** — the time for one job and the amount of work per unit of time. For generation, time to first token and tokens per second are useful, alongside kernel TFLOPS.

**p50 and p99** — values below which 50% and 99% of measurements respectively fall. p50 shows typical latency; p99 helps reveal rare slow cases.

**Profiling** — measuring where a program spends time and resources. **Sampling** here means periodically taking observations. A **stall** delays instruction progress, for example while waiting for data. The **critical path** is a dependent chain of work that limits the completion time of the entire task.

**Benchmark, baseline** — a reproducible measurement and an initial version used for comparison. **Warmup** — preliminary runs that separate warming up from the main measurement. A **regression** is deterioration relative to the baseline.

**Cached and streaming GEMV in measurements** — modes where weights are mostly read from cache or pass through slower memory again. Separately, **Streaming GEMV in the existing implementation** means a sequential traversal of weight rows. This algorithm does not bypass the cache and does not by itself guarantee reads from RAM.

**Physical ceiling; conditional ceiling** — a hardware limit and a calculated reference point under stated assumptions. For example, an estimate based on frequency and FMA count does not account for every delay. **Roofline** relates arithmetic and data-transfer limits to the amount of computation per byte read.

**Timestamp query** — recording time using GPU facilities. **Validation** in Vulkan measurements usually means diagnostic API checks that can change execution time. These runs must be separated from measurements of execution speed alone.

**Guard pages** — inaccessible memory pages around test data that help detect out-of-bounds access. A **fixture** is a prepared input and expected test result. **QEMU** can execute code for another architecture; the result of such a test is not a speed measurement of a real ARM processor.

**Canary** — control values near data whose modification reveals an unwanted write. **Sanitizers** — checks inserted by build tools to detect errors such as invalid memory access.

## Builds and applications

**API and ABI** — rules for calling a library from source code and compatibility rules for already compiled components. An ABI includes, for example, calling conventions and the layout of public data in memory.

**Opaque type** — a type whose internal structure is hidden from API users. The application uses the provided functions instead of modifying internal structure fields.

**C17, CMake, Ninja** — the C language standard and tools for organizing and executing a build. **CI** — automated builds and checks after changes. Project dependencies and their status are listed in [dependencies.md](dependencies.md).

**Cross-compilation** — building a program for a different platform. In this context, the **host** is the build machine and the **target** is the execution platform. A code generator must run on the host even when its output is intended for the target.

**CLI, HTTP, SSE** — a command-line interface, a protocol for communication with a server, and a way to send events from an HTTP server to a client. **Backpressure** limits data production or accumulation when the receiver cannot read it fast enough.

**Dependency** — third-party code or a tool required to build or run part of the project. Open-source C++ code used inside a library does not become a “closed-source dependency” because its types are hidden from the public C API.
