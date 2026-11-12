# How we evaluate performance

[Русский](../RU/performance.md)

This document defines measurement rules for the move to C and Vulkan. Terms are explained in the [glossary](glossary.md).

**Status:** the first C CPU linear module and its measurements are described in [cpu-linear.md](cpu-linear.md). The C runtime and Vulkan backend are not implemented. Historical Rust measurements remain in [cpu-performance.md](cpu-performance.md) and the experiment archive (`../benchmarks/`); they do not establish performance of the C module or the future engine. CLI commands `probe` and `bench` are still being designed.

## What we measure

Every result identifies one of three levels:

1. **An individual compute kernel** on prepared data. For example, one GEMV without preparing a job for the worker pool.
2. **A complete operation call**, including implementation selection, dispatch to the pool and waiting for completion.
3. **A model step**, which executes several operations and changes model state.

These measurements answer different questions. A faster GEMV may barely affect a model step if most time goes into other operations or waits. Model-level measures also include decode latency, prefill throughput, memory use and time to first token.

### First launch and repeated work

Measure cold loading, weight packing and pipeline creation separately. For steady execution, state how the program was warmed up before measurement.

Also distinguish data that fits in cache from weights read from main memory. Repeating GEMV on one matrix differs from traversing successive weight matrices of a model. The report must identify which case was measured.

## Units

For dense GEMM with input `[N,K]` and weights `[M,K]`, use:

```text
TFLOPS = 2*N*M*K / seconds / 1e12
```

Here `seconds` is the measured time in seconds. GEMV uses the same formula with `N=1`.

The numerator counts only useful arithmetic under this convention. Extra padding, packing and control operations do not increase its FLOP count. Their time counts if they are part of the measured call.

For GB/s, always specify which bytes are counted: logical data volume or measured traffic at a particular memory level. Cache reads and DRAM reads must not silently be treated as the same traffic.

## What we compare against

The estimated maximum compute rate depends on:

- active core count;
- instruction width and throughput;
- data type;
- processor frequency under the measured workload.

Brief boost frequency and a rated maximum do not establish sustained GEMM speed. For GEMV, also estimate read volume and available DRAM or relevant-cache bandwidth. Measure memory bandwidth, inter-chiplet bandwidth and working thread counts on the current machine.

Roofline and `max(compute time, transfer time)` help build an estimate if their assumptions are explicit. A different microbenchmark's speed does not set a strict physical bound for our model step.

Label the result's ratio to that estimate as a **ratio to a conditional reference**. For example, reaching 80% of a reference does not itself prove 80% hardware utilization or that a further 20% speedup is available. Report all arithmetic-rate references in TFLOPS as well.

## What we check before optimizing

First identify where time is spent. On CPU, separate:

1. call preparation;
2. job publication to workers;
3. each worker's start time;
4. computation;
5. call completion.

Sampling locates CPU-time consumption. Available hardware counters help check causes of stalls and traffic volume. The timeline identifies the critical path.

Worker waits may overlap. When several workers wait simultaneously, their waiting times cannot be added and presented as one call's latency.

For Vulkan, separately measure GPU commands, CPU waits, copies and pipeline creation. Use timestamp queries only when supported by the relevant queue. GPU and CPU clock values cannot be subtracted directly without a valid correspondence.

Profilers and validation affect execution. Confirm an apparent improvement in a separate run without them.

Before changing code, state a testable cause: repeated packing, an unnecessary intermediate buffer, too little work per job, uneven matrix partitions or a slow kernel region. Changing the language, compiler or alignment does not by itself explain lost time.

## Making measurements reproducible

### Keep a kernel comparison table

Each optimization report includes a table of the measured implementations, including
the baseline, retained candidates, rejected candidates and external implementations
that were tested. Identify a kernel by its source revision or hash and describe the
change; the compiler name alone does not identify a different algorithm.

For every row, show the workload and dimensions, hardware/ISA, compiler, worker count,
median TFLOPS and variation, numerical contract and bitwise-check result. A shared
heading may hold conditions that are identical for all rows. Include the relevant
compute or memory reference in TFLOPS and state whether it is theoretical, conditional
or measured. An unknown physical ceiling stays unknown.

Keep a run identifier and a link to the raw measurements. Results from separate runs
must remain distinguishable; do not choose the best sample from each and call it a
paired comparison. Include regressions. An implementation awaiting code or measurement
gets an explicit pending entry, with no invented speed or correctness result.

Update the table as candidates are evaluated. Raw benchmark data remains local under
`docs/benchmarks/`, excluded from commits; concise published summaries stay synchronized
between English and Russian documentation.

### Record the conditions

Keep the following in the report:

- source or binary version and compiler;
- CPU, GPU, driver and ISA;
- shapes, dtype and layout;
- threading policy and affinity;
- cache regime, warmup order and repetition count.

Alternate the baseline and candidate when comparing them. Show variation, regressions and complete call cost. Measurements under competing load, such as a game, are not evidence of an optimization's benefit.

Every optimization passes numerical comparisons and memory-bound checks. Choose implementations for different CPUs and GPUs using capabilities and measurements. Users must not need the developer's processor.

Builds and checks under emulation can validate selected correctness scenarios. Measure real ARM performance on the actual device.
