# Execution proposals

**Status: proposed, not validated in this engine.** These are implementation candidates under the boundaries in [architecture.md](architecture.md). They do not block a simpler correct executor. A decision becomes **validated** only with a reproducible experiment and its hardware/workload recorded; it becomes **implemented** when the runtime path and correctness checks exist. No proposal below has either status yet.

## Hybrid CPU/GPU execution

For a layer with GPU attention/router, CPU routed experts, and an independent GPU branch, the proposed dependency chain is:

1. GPU computes routing and copies the input and selected expert IDs to host staging.
2. CPU waits for that copy, then computes selected experts. The independent GPU branch may run concurrently.
3. The CPU result is copied back; the GPU joins both branches before consuming the result.

Start with explicit completion events. An optimization candidate uses two pinned-memory flags per execution slot: GPU writes `A = 1` after D2H; CPU observes and resets A, computes, and publishes `B = 1`; GPU waits for B, resets it, and performs H2D. CPU workers could spin instead of sleeping.

This sketch is not a complete synchronization specification. Before adopting it, verify host/device memory visibility, ordering, graph-capture support, slot reuse, multiple in-flight steps, error propagation, and shutdown. A layer index alone does not identify an in-flight execution. Compare segmented execution with captured regions on the target hardware, including delayed CPU work and failed I/O; measure p50/p99 crossing latency and overlap. Keep the simpler path if the additional protocol does not help.

## Storage and prefetch

Possible strategies are local access, peer/device copies, direct reads of registered host memory, and staged disk reads. Choose among supported strategies during preparation; a sparse access pattern alone does not establish which is fastest.

For disk tables, prefetch starts when row addresses become known. Record the first consumer and the independent work available before it. Test cold physical reads, page-cache hits, I/O failures, and limited staging space. Logical row bytes are not physical I/O volume. `io_uring` versus `pread`, double buffering, page size, and prefetch distance remain implementation choices.

For chunked sliding attention, a ring covering `W + T_max - 1` positions is the current sizing approach. Validate overwrite safety against the actual execution and rollback order before using it as a runtime guarantee. Recurrent state history and prefix snapshots must also fit the full sequence budget.

## Optimization candidates

| Candidate | Evidence required before adopting it |
| :--- | :--- |
| CUDA Graphs for regions or a full step; capacity buckets | Correct changing lengths/bindings, bounded memory, and measured launch savings; no requirement for one graph covering the whole model |
| Projection, norm, residual, and selection fusion | Reference equivalence, preserved state effects, and measured benefit |
| Computing attention inputs from shared storage inside a kernel | Correct transformations with declared scratch; no hidden full-sized materialization of either K or V |
| Prefill output-row elimination | Proof that omitted rows have no required outputs or state effects, followed by differential tests |
| CPU/GPU weight repacking and prefill streaming | Numerical equivalence, physical byte accounting, and measured bandwidth/contention |
| Automatic placement and `Program::floor` | Hardware measurements and dependency/resource accounting that distinguish lower bounds, predictions, and observed time |

`CpuBand16` and `GpuWarp` are descriptor layout names already present in the foundation. Backend repacking, `lumen pack`, fused kernels, and automatic placement are still future work. Kernel algorithms and research measurements are recorded in [kernels.md](kernels.md); projected performance is in [performance.md](performance.md).

## Calibration and later runtime features

Calibration needs observable block execution, teacher forcing, and controlled replacement of weights. It should share model and execution mechanisms without depending on the serving scheduler. Use existing research exports to test inference while developing the native quantizer independently. Native Fisher/backward support needs its own mathematical implementations and checks; modular forward blocks do not provide it automatically.

Prefix sharing, speculative decoding, and asynchronous submission of the next step require explicit committed/speculative state boundaries and resource lifetimes. Snapshot frequency, rollback representation, eviction, draft interfaces, and request queues remain open. First test partial acceptance, stop/cancel with work in flight, and capacity exhaustion on a small model; then optimize throughput.

## References for implementation work

- [NVIDIA CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [CUTLASS](https://github.com/NVIDIA/cutlass) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [LTQ research prototype](https://github.com/lumen-inference/ltq-research)

These references and the proposed flow are starting points for experiments, not evidence that a runtime mechanism has been validated in this repository.
