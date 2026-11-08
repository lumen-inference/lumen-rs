# CPU kernels: correctness and measurement

[Русский](../RU/cpu-performance.md)

> Historical measurement reports, raw experimental data and old source snapshots are kept locally under `docs/benchmarks/` and excluded from Git. Plain paths to these artifacts are references, not download links.

This document records the existing implementation and its measurement history. For the general approach, start with the [kernel overview](kernels.md) and [measurement rules](performance.md). Terms are defined in the [glossary](glossary.md).

> This document describes the current Rust/C/LLVM CPU path. Its historical Lumen labels, commands, and measurements remain comparison references for Velri's [C/Vulkan migration](architecture.md), not results of a new C runtime.

`lumen-cpu` executes the current dense-f32 graph through `CpuExecutor`, with Scalar, AVX2/FMA and AVX-512F modes. Runtime feature detection rejects unsupported explicit selections. The portable binary does not require `target-cpu=native`. `Isa::detect()` chooses the widest supported ISA, not an empirically tuned winner for every workload.

`CpuExecutor::new(isa, threads)` creates a persistent private worker pool when threads > 1. On Linux it binds workers within the calling thread's allowed CPU mask, selecting physical cores before SMT siblings. Worker order follows NUMA node, shared cache, physical core and sibling. Discovery uses OS topology, without processor-specific CPU numbering. Neither caller affinity nor system policy is changed. Internal dispatcher selection is described below.

`CpuExecutor::with_threading(isa, CpuThreading { .. })` also supports automatic sizing (`threads = 0`), reserved cores, NUMA spreading and affinity policy. `Auto` reports why binding was unavailable, `None` leaves workers unbound, and `Required` returns an error if binding fails. Startup waits for every binding result. An explicit thread count is preserved, including oversubscription in unbound mode; counts exceeding Rayon's supported limit are rejected. `worker_cpus()` and `placement_note()` expose the result. `Default` remains one inline worker; adaptive matrix scheduling learns whether to keep work on the caller. A required one-worker pool routes matrix and attention calls through its worker.

The parsed [CPU configuration](rust-config.md) reaches this API through `lumen_engine::cpu_executor`. This adapter is available; real-model CLI execution remains unfinished. The synthetic decoder accepts a thread count, for example `cargo run --release -p lumen-exec --example dense_decode -- avx512 32`.


## Comparison with AOCL, BLIS, OpenBLAS and LIBXSMM (2026-11-08)

A common FP32 harness compared current Lumen with AOCL-BLAS 5.3.2, upstream BLIS 2.1, OpenBLAS 0.3.34 and LIBXSMM 2.1.0. The final series contains 405 separate processes, five repeats per configuration and 21 samples per process; 108 correctness processes passed. With a 16-thread limit on Ryzen 9 9950X, prepared GEMM 256×4096×4096 measured **4.381 TFLOPS for Lumen**, **4.309 for AOCL prepared**, and **4.291 for LIBXSMM with a benchmark adapter**. The conditional arithmetic reference from these Lumen runs' frequency counters is **4.823 TFLOPS**, placing our result at about **90.8%** of that reference.

Cached GEMV 1024×1024 measured **1.075 TFLOPS for Lumen prepared**, 0.733 for OpenBLAS and 0.641 for LIBXSMM; the conditional L2 weight-delivery reference is **2.689 TFLOPS**, or about **40.0%** for Lumen. BLIS ran this GEMV on one observed thread despite the larger configured limit. Skinny N=4 GEMM favors LIBXSMM by median: **2.045 versus 1.963 TFLOPS**, with substantial process variation. Streaming GEMV with 512 MiB weights is around **0.033–0.036 TFLOPS** for the main multithreaded variants; the physical RAM ceiling was not established. These are repeated linear calls without graph overhead; preparation is measured separately. All shapes, thread counts, setup costs, sources and limitations (`../benchmarks/linear-libraries-2026-11-08/README.md`).

## Prepared graph weights and output checks (2026-11-08)

`Program::prepare` now requests a physical `DenseLayout` from the executor and packs weights used exclusively by Linear once. Padding is budgeted before allocation; multiple Linear consumers and sessions share one owner. Tied embedding/projection weights retain a single row-major representation. Graph steps borrow prepared weights without copying or allocating metadata. NaN/Inf checks remain enabled, using a reduction the compiler can vectorize.

On 16 bound workers, the complete synthetic embedding → Linear → output check → reset path improved cached GEMV from **0.507 to 0.643 TFLOPS for AVX2**, and **0.537 to 0.645 for AVX-512**. GEMM 256×4096×4096 improved from **1.919 to 2.038** and **3.175 to 3.625 TFLOPS**, respectively. These effective rates include graph costs. Standalone prepared GEMV remained around **1.05–1.07 TFLOPS**, AVX-512 GEMM around **4.39 TFLOPS**; the broader series includes small regressions up to 2.8%. Arithmetic C kernels are unchanged. Profiles, measurements, ARM validation and limitations (`../benchmarks/linear-bottlenecks-2026-11-08/README.md`).

The result tables also include ceiling references and fractions of them: 16-core GEMM **2.473 TFLOPS AVX2 / 4.803 TFLOPS AVX-512**, L2-fed cached GEMV **2.674 TFLOPS**. These are conditional estimates using earlier frequency observations; the exact ceiling of the latest runs was not measured. Standalone AVX-512 GEMM reaches about **91.4%** of that reference, cached GEMV **40.0%**; the complete graph reaches **75.5%** and **24.1%**, respectively.

## Loop and measurement update (night of 2026-11-08)

Prepared GEMV remainders now reuse grouped arithmetic and independent single-tile accumulation without padding more weights. The retained comparison improves AVX2 N1/M1025/K1023 on 16 workers **0.758 → 0.931 TFLOPS** (paired +22.48%) and M256/K1024 **0.343 → 0.407 TFLOPS** (+19.41%). Large prepared AVX512 GEMM remains around **4.4 TFLOPS**. This is a documented tradeoff: short bound single-worker M49/K2–4 calls retain regressions, and Adaptive choices still vary. Measurements, verification and limitations (`../benchmarks/linear-night-2026-11-08/prepared-tail/README.md`).

The shared LLVM generator uses two independent GEMV accumulation chains when target vector geometry groups two narrower prepared-weight bands. A single wide band retains its previous loop. No CPU-model check, weight-layout change, or extra storage is involved. A repeated comparison on 16 AVX2 workers with a 1024×1024 matrix measured **0.892 → 1.021 TFLOPS**. AVX-512 remains around **1.07 TFLOPS**. The tiny K=3 case lost about 2%; other shapes and limitations are recorded in the night report (`../benchmarks/linear-night-2026-11-08/README.md`).

Blocked GEMM now reuses weights across six rows when the target register budget permits it. Direct full-K calls and padded tails retain four rows: switching those to six regressed several small/narrow shapes. A separate matched comparison of 256×4096×4096 prepared GEMM on 16 workers measured **2.111 → 2.259 TFLOPS for AVX2** and **4.310 → 4.413 for AVX-512**, with cached GEMV essentially unchanged. Arithmetic generation, packed weight format and scratch limits remain shared and unchanged. Native ARM performance is still unmeasured.

Narrow prepared matrices now retain only their actual weights: AVX-512 M1/K1024 uses **4 KiB instead of 256 KiB**, and the complete call measures **0.00195 → 0.112 TFLOPS**. This removes padded work; it is not a comparable gain on wide layers. The common preparation loop, cold preparation boundary and 32-byte prepared-entry alignment were validated together. Wide cached AVX2 still has a plausible **roughly 1% cost**; an initial 5% loss with a configured 32-thread pool did not reproduce at that magnitude in the focused comparison. The storage fix is accepted with this limitation. That storage stage passed 151 native release tests and 17 tests on each AArch64/ARMv7 target through QEMU. Full measurements and rejected candidates are retained in the night report.

Adaptive prepared GEMM now also learns a row/column partition. In the final matched comparison, tall AVX2 N4096/M256/K4096 improved **1.960 → 2.236 TFLOPS**, and odd N1025/M257/K4096 **1.333 → 2.054 TFLOPS** (paired +13.81%/+54.18%). Main AVX-512 GEMM remains around **4.38 TFLOPS**, cached prepared GEMV around **1.07 TFLOPS**. The change retains losses too: inline AVX-512 M1 is down 9.02% and ordinary Adaptive cached GEMV 5.05% in paired blocks. The latter coincides with different learned streaming choices; it does not isolate a raw-kernel slowdown. The final seed recheck does not guarantee optimal selection. Final measurements and tradeoffs (`../benchmarks/linear-night-2026-11-08/README.md`) include all controls and the rejected single-worker shortcut. The retained source passed 160 native release CPU tests, 135 without default features, 270 workspace tests and 39 targeted tests on each ARM/QEMU target, plus formatting and Clippy.

Configured thread count is pool capacity. In the Fixed/Required cached measurements, the 32-thread pool uses **16 active physical workers**; `cached32` does not mean 32 simultaneous SMT arithmetic workers. Task count can reduce activity further: N96/M3 has twelve tasks with sixteen configured workers. Small work may execute on the caller. Adaptive can select a different group, recorded in its learned schedule.

Profiles and diagnostic frequency counters qualify the earlier roof comparison: real GEMM runs at different clocks from the synthetic FMA test, whose values settle to constant bit patterns. Counter snapshots around the timed gate measured AVX-512 GEMM at **4.37 TFLOPS** against a frequency-scaled arithmetic reference of **4.80 TFLOPS**. That is about 91% of this reference, not measured hardware utilization or proof that only 9% improvement remains.

Cached GEMV has a different constraint. AMD's published L2-to-L1D path width and the separately observed frequency give a conditional reference of **2.67 TFLOPS** across 16 cores if weights arrive from L2 and other costs are negligible. The real complete call remains around **1.07 TFLOPS**. A separate diagnostic with 256 local repetitions per callback measured **1.72 TFLOPS**, under a different execution regime. Checked reads had a similar median with much greater variability; they cannot establish a physical ceiling. Sources, ranges and assumptions are in the night report (`../benchmarks/linear-night-2026-11-08/README.md`).

`LUMEN_BENCH_WARMUP_MS` sets a minimum warmup duration outside timing; default 0 preserves the previous call floor. Longer warmup alone does not guarantee stable frequency or worker placement.

## Prepared immutable weights (2026-11-08)

With `llvm-kernels`, `cpu.prepare_linear(&weights, m, k)` creates an owned, aligned matrix. Outputs narrower than one target SIMD vector retain row-major weights without padding; wider matrices use blocked storage with a padded final tile. Layout and its matching kernel are selected together at preparation time. Reuse the object with `cpu.linear_prepared(&input, &prepared, bias, &mut output)`. Preparation retains one copy; `prepared.payload_bytes()` reports its payload. The source can then be freed. Keeping both objects keeps both copies. A prepared object requires the same executor ISA policy and can be shared by reference or `Arc`.

Repeated calls perform no weight packing. Large prefill retains cache blocking and up to 64 KiB of partial sums per active kernel, plus a small output-tail tile. Removing this blocking made GEMM slower in two measured experiments. GEMV processes independent outputs across packed bands using compiler-derived geometry. Scheduling observations are separate from ordinary matrices and available through `prepared_linear_schedule(rows, m, k)`.

Prepared GEMM also explores up to three distinct row/column grids on the full physical worker group. Their requested row splits are 2, 4 and 8; actual partitions are recomputed from the current shape and tile width. Existing worker-count choices remain available. `requested_row_groups` in the learned schedule reports the request before rounding. Fixed execution keeps the previous partition.

On 16 bound workers, Fixed cached GEMV measured **0.891 → 1.003 TFLOPS**, the main GEMM **4.291 → 4.362**, and the longer-K GEMM **4.014 → 4.363**. Prepared unbound GEMM regressed against the ordinary path; preparation must also be amortized. These historical measurements cover the explicit API; subsequent graph integration with memory budgeting is described above. Full results, limitations and reproduction (`../benchmarks/linear-prepared-2026-11-08/README.md`).

All production pools now use the same fixed submission protocol, including unbound and Required single-worker execution. Affinity remains optional. This removes the periodic Rayon queue-block allocation found by ARM testing. The ordinary unbound cached GEMV measured **0.218 → 0.649 TFLOPS**. Rayon remains only in differential tests. Earlier sections below record historical dispatcher versions.

## Adapting worker counts

`CpuExecutor` defaults to `LinearPolicy::Adaptive`. For each matrix shape it compares caller execution, half and all verified physical workers, and the configured logical group when SMT makes it distinct. Without verified topology it compares configured pool groups. GEMV can choose ordinary or sequential weight-row arithmetic even with one, two or four workers. LLC capacity supplies the initial hint; actual calls refine it. No CPU-model name participates in selection.

Learning uses at most 36 successful, uncontended calls per shape before eviction. Each call computes its result once, with no extra matrices, weight copies or heap allocations. Calls that bypass a busy learning lock do not advance observations. Construction performs four empty pool submissions to estimate handoff cost. Serial exploration is admitted only when an estimate from observed elapsed time and handoff cost makes it inexpensive; this is not a latency guarantee. Each candidate gets one genuine warm-up call and three timed calls. After the candidate observations, the original schedule gets another warmup and three measured calls. A candidate’s median must beat the best initial and final observations of the original schedule before switching. This final check cannot reopen skipped inline exploration. A lone fast candidate observation cannot decide the winner; the final control can expose capacity hidden by a slow initial seed. Selection can still be suboptimal if both observation windows differ from later execution. After convergence, the normal path reads neither a timer nor the learning lock. A fixed 32-entry cache bounds metadata; colliding shapes can restart learning. `LinearPolicy::MAX_LEARNING_CALLS` exposes the bound for warmups. A single Required worker retains owned-worker execution.

Initial latency can differ from settled latency. Outside timed calls, `cpu.linear_schedule(rows, m, k)` reports a converged task cap and GEMV mode; `None` means no matching converged observation is available. System load, memory placement, or cold versus hot weights can invalidate an old choice: there is no periodic re-learning. A new executor resets observations. `cpu.with_linear_policy(LinearPolicy::Fixed)` selects the previous deterministic dispatcher for comparisons or workloads that cannot accept learning. Fixed retains its earlier thresholds; Adaptive does not use them.

`bench_linear` accepts `LUMEN_LINEAR_POLICY=adaptive|fixed` and reports policy and warm-up explicitly. Results and validation limits (`../benchmarks/linear-adaptive-2026-11-02/README.md`). Restricting resources on the 9950X exercises smaller pools, not another microarchitecture. ARM/QEMU checks correctness, not ARM performance.

## Submitting short worker jobs (2026-11-07)

The bound pool records sleeping workers in machine-word bitmaps. Submission consumes only the selected group's bits and calls `unpark` for those workers. Active pollers need no permit. An acquire/release RMW chain connects publication with registration and the worker's final recheck before sleeping; excluded workers retain their bits. Membership comes from the existing topology, without CPU-model conditions.

With unchanged handwritten arithmetic, cached GEMV `[1024, 1024]` measured **0.865 → 0.945 TFLOPS** on 16 workers and **0.880 → 0.950** on 32. The common LLVM path also benefits, but the arithmetic experiment still has shape-dependent regressions. Deterministic park-entry tests, cross-word groups and both ARM word sizes are covered. Measurements, remaining losses and validation (`../benchmarks/linear-gemv-followup-2026-11-07/README.md`).

## Common LLVM kernel

### Experimental vector cleanup (2026-11-07)

`lumen-cpu/experimental-dot-kernels` enables the common LLVM dot kernel's vector cleanup and implies `llvm-kernels`. After the unrolled loop it consumes remaining complete SIMD vectors, leaving fewer than one vector of scalar work. Each output keeps its own LLVM horizontal reduction. The normal `llvm-kernels` feature retains its previous arithmetic schedule. Geometry comes from the target; no CPU-model table or separate ARM implementation was added.

The candidate passed workspace/release tests, Clippy, Rust 1.89 checks and AArch64/ARMv7 execution through QEMU. New tests cover exact numerical results, bias, unaligned inputs, output gaps, NaN/Inf isolation and protected-page bounds, including a main unroll followed by both vector and scalar remainders. Native ARM speed remains unmeasured.

This remains **opt-in**. The follow-up uses separate original and cleanup GEMV functions when the target-derived vector remainder permits it. Fixed K=65 recovered from **0.078288 to 0.091945 TFLOPS**; with the updated sampler and pool, Adaptive cached8 measured **0.636343 → 0.740650**. Fixed cached8 still regressed from the preceding experimental binary by 7.2% (and by 2.0% against the updated feature-off build). The follow-up report (`../benchmarks/linear-gemv-followup-2026-11-07/README.md`) retains every shape, rejected variant, source snapshot and raw comparison. The initial report (`../benchmarks/linear-dot-2026-11-07/README.md`) remains a historical record. The hardware-ceiling objective remains open.

Validation also exposed a preexisting unbound Rayon pool startup race: worker stack metadata could allocate after the constructor returned. That revision waited for an empty broadcast to finish on every worker; the current fixed unbound pool also removes periodic queue allocations. The allocation test subsequently passed 80 consecutive runs, including adaptive learning and settled execution; construction allocations remain outside inference calls.

The vector-cleanup scheduling idea was compared with [MIT gemm](https://github.com/sarah-quinones/gemm/blob/main/gemm-common/src/gemv.rs) and implemented in our generator without copying upstream source or adding dependencies. Compare separate builds with `--features llvm-kernels` and `--features experimental-dot-kernels`; `--all-features` also enables the experiment. The benchmark prints `llvm_dot_schedule=baseline|experimental`.

### Existing path

The `llvm-kernels` Cargo feature enables the common high-performance dense-f32 GEMM/GEMV path. With it, standard `CpuExecutor::default`, `new` and `with_threading`, including the engine factory, use generated code for `Linear`. Engine and CLI forward the same feature. `CpuExecutor::with_native_linear(isa, options)` explicitly retains the handwritten kernels for comparisons and fallback. Builds without the feature keep the previous path and require no LLVM tools.

One generator (`../../crates/lumen-cpu/codegen/linear.rs`) describes vector reductions, outer products and weight transpose. The common schedule (`../../crates/lumen-cpu/codegen/linear.c`) and packing (`../../crates/lumen-cpu/codegen/packing.c`) define reuse across rows, tile boundaries and bias application. Narrower outputs select smaller column tiles; row reuse follows the same register budget. LLVM TargetTransformInfo provides destination vector width and register capacity; geometry is derived from those properties. There is no separate ARM algorithm or CPU-model schedule table. AVX2/FMA and AVX-512F differ in compiler flags and are guarded before execution.

Rows/M/K, output row stride and optional bias remain runtime parameters. Row, column and K tails need no tensor padding or special alignment. Large GEMM uses up to 32 KiB packed weights and 64 KiB intermediate output sums on each active call's stack. Successful computation performs no heap allocations or full weight copies. Cached GEMV shares inputs across outputs; matrices exceeding discovered aggregate cache capacity use sequential weight rows through the same reduction generator.

Building requires Clang/LLVM 18+ and LLVM development headers, library and `llvm-config`. Clang and `llvm-config` must have the same major version; local validation used 21.1.8. `LLVM_CLANG`, `LLVM_AR`, `LLVM_CONFIG` and `LLVM_CXX` override tool paths. A small build-host C++ utility queries destination properties; it and libLLVM are **not linked into the application**. Kernels compile ahead of time for Cargo TARGET. Inference needs no JIT or compiler, and no new Rust crates were added.

```sh
cargo build --release -p lumen-cli --features llvm-kernels
cargo build --release -p lumen-cpu --features llvm-kernels --example bench_linear
LUMEN_LINEAR_KERNEL=llvm taskset -c 0-15 target/release/examples/bench_linear avx512 16 256 4096 4096 21 required 0 2 0
LUMEN_LINEAR_KERNEL=native taskset -c 0-15 target/release/examples/bench_linear avx512 16 256 4096 4096 21 required 0 2 0
```

Direct API: `lumen_cpu::llvm_linear(x, w, bias, out, m, k)`. Explicit executor selection: `with_llvm_linear(options)` or `with_llvm_linear_for_isa(isa, options)`. Program preparation and execution are unchanged. Readonly inputs may alias each other; output is disjoint. Invalid shapes fail before writes. Reduction reassociation, multiply-add contraction and signed-zero changes are permitted; absence of NaN/Inf is never assumed. Long reductions and packed execution are tested separately in every available profile.

The pool also observes the caller's real OS CPU. When a selected worker is bound to that same CPU, both caller and that worker stop active waiting sooner. This addresses measured CPU competition without changing user affinity. Publication and borrowed-job lifetime correctness remain independent of the scheduling hint.

Validation includes x86-64 AVX2/AVX-512 and execution of the same generator on AArch64 and ARMv7 under QEMU, including 32-bit ABI, packing and strided outputs. Real ARM speed remains unmeasured. Final measurements, reviews, source and limitations (`../benchmarks/linear-packed-2026-11-01/README.md`). The earlier scalar prototype (`../benchmarks/linear-universal-2026-11-01/README.md`) is preserved as historical evidence and has been replaced.

## Implemented operations

- **Linear:** decode GEMV uses independent accumulation chains. Small token groups share weight loads. Prefill uses bounded stack weight tiles and outer-product microkernels. Workers partition output features, and narrow outputs also partition token groups. There is no full additional weight copy. Stack packing uses up to 8 KiB for AVX2 and 32 KiB for AVX-512 per worker. Selected AVX-512 tiles also use up to 64 KiB for temporary sums, totaling up to 96 KiB of stack scratch. Current selection policies and measurements are described below.
- **RMSNorm:** SIMD reduction and scaling, including optional `1+w`; exceptional magnitudes use wider scalar arithmetic.
- **SwiGLU / GeGLU:** SIMD range-reduced degree-six exponential approximation with exceptional-lane fallback. GeGLU follows tanh-approximate GELU; optional clamping is symmetric on both inputs.
- **Residual and RoPE:** vector multiplication/addition. RoPE angles still use scalar f64 calculations reused across heads; they remain a cost.
- **Attention:** causal full/windowed GQA, distinct key/value dimensions and shared K/V storage. Each QK score is calculated once, with tiled softmax using a 64-score, 512-byte stack buffer. No full score matrix or history copy is allocated. Decode distributes independent heads across workers; prefill distributes query rows. A dot-error estimate accounts for cancellation and scale; risky dots are recomputed in f64.
- **Lookup and append:** slice copies without per-element binding lookup.

The executor obtains validated readonly inputs and a disjoint mutable output once per operation. Inputs may alias each other; they cannot alias the output. Finite-output validation remains at the dispatch boundary, preserving failed-step history semantics.

## Automated validation

`cargo test --workspace` compares each available ISA against independent reference kernels: tails, unaligned data, odd matrices, biases, head grouping, windows, rotary positions, extreme values and malformed shapes. The composed decoder also matches an independent Python fixture for all 32 input partitions. Shared/derived storage and failure recovery are exercised with the CPU executor too.

CI additionally runs `cargo test --release -p lumen-cpu -p lumen-exec` to check optimized machine code. Unavailable ISAs are skipped; complete SIMD coverage needs hardware with both AVX-512F and AVX2/FMA. Rust 1.89 remains the minimum, including [standard-library AVX-512 intrinsics](https://doc.rust-lang.org/core/arch/x86/fn._mm512_fmadd_ps.html).

Tolerances live in the tests. Typical vector checks use a few `1e-6` absolute/relative error; attention fixtures use `2e-5 * max(1, abs(reference))`; the decoder uses `2e-5` absolute error. These are tested fixture envelopes, not universal guarantees for arbitrarily ill-conditioned inputs. SIMD/FMA and reduction ordering change rounding.

## Reproducing benchmarks

```sh
cargo build --release -p lumen-cpu --examples
# ISA, threads, tokens, output features, input features, samples, optional affinity:
taskset -c 0 target/release/examples/bench_linear avx512 1 128 2048 2048 21
taskset -c 0-15 target/release/examples/bench_linear avx512 16 1 8192 8192 21 required
# Other operations: ISA, samples, small-kernel repetitions:
taskset -c 0 target/release/examples/bench_ops all 21 512
# Attention: ISA, threads, tokens, history, Q heads, KV heads, K/V dimensions, samples:
taskset -c 0-7 target/release/examples/bench_attention avx512 8 1 2048 32 8 128 128 25 required
# Read baseline: ISA, threads, MiB, samples:
taskset -c 0-15 target/release/examples/bench_ceiling avx512 16 256 21
# Register FMA: ISA, threads, iterations per worker, samples:
taskset -c 0-15 target/release/examples/bench_fma avx512 16 1000000 9
```

`bench_linear` includes shape checks and pool dispatch, but excludes graph preparation, buffer binding and the executor's final finite-output scan. Allocation and warm-up are untimed. Effective GB/s is minimum weight + input + output bytes divided by time, not measured memory-controller traffic. `bench_ops` QK+PV throughput excludes softmax FLOPs. Compare numerical tolerances and checksums rather than expecting bitwise equality across ISAs.

The calibration programs provide observed streaming-read and register-FMA rates on the same machine. Neither is a universal physical ceiling for model execution. Cache state, clock/power limits, active cores, SMT, OS scheduling and shapes matter. Actual generation throughput requires a separate real-model benchmark.

`bench_linear` and `bench_attention` print verified worker placement. Use `none` to compare unbound execution. After affinity, `bench_linear` accepts output offset, calls per sample, and input/weight offset. Offsets are f32 elements from a 64-byte boundary (`0..15`); actual alignment is printed. For example, `avx512 16 256 4096 4096 41 required 0 8 0` aligns all three matrices to 64 bytes. Kernels continue to accept any natural f32 alignment.

## Current change: short GEMM with uneven SMT placement, 2026-11-01

Build `89c01f33` selects the existing physical-core group for short SIMD batches when the configured worker count is not divisible by the physical-worker count. Profiles and actual work partitioning expose unequal work per core when some cores own one worker and others own two. Tested shapes improve 37–46% at 17 threads and 73–107% at 24. Configured pool size, CPU ordering and arithmetic kernels are preserved.

Large GEMM is essentially unchanged: 4.222 → 4.171 TFLOPS in the paired series. Small controls still show unexplained differences down to −5.67%; maximum engine performance has not been achieved. The broad policy for all short batches was rejected. 252 runs, eight profiles and limitations (`../benchmarks/linear-short-group-2026-11-01/README.md`) are preserved. Validation passes: 216 workspace tests, 115 release tests, Clippy/fmt, Rust 1.89 and four dispatcher tests under CPU0 affinity.

## Fresh profiles and rejected experiments, 2026-11-01

Eleven profiles cover production build `e674478b` and a short-GEMV experiment. Large GEMM spends 87–88% of aggregate CPU time in its K loop and about 7.5% packing weights. The pool and waiting account for 59–63% with 4 MiB GEMV weights and 80–82% for GEMV 2049 → 129. Streaming GEMV with 256 MiB weights spends about 99% inside its kernel. These shares neither predict elapsed-latency improvements nor measure hardware stall causes.

The experiment reduced short-GEMV participation and command-polling share, but repeated unprofiled runs showed regressions. It was rejected and production source bytes restored. Separate current-build runs reach 4.236 / 4.265 TFLOPS for large GEMM at 16 / 32 configured threads and 0.04115 TFLOPS for streaming GEMV. Raw clock packets, reports, sources, 197 unprofiled process runs and restored-binary checks are preserved in the report (`../benchmarks/linear-targeted-profile-2026-11-01/README.md`). All 114 release tests and formatting passed. Reducing GEMM panel depth to 96/64 also failed to provide a general improvement: 112 runs and rejected prototype (`../benchmarks/linear-wide-depth-2026-11-01/README.md`).

A GEMM instruction-count hypothesis was also tested: FMA memory broadcasts reduced the loop from 28 to 24 instructions, but large GEMM lost about 19–20% (4.245 → 3.402 TFLOPS on 16 workers). The prototype passed numerical checks and review but was rejected on performance. The installed LLVM-MCA `znver5` model also failed the AVX-512 calibration check: its register-FMA prediction contradicts measured throughput. 26 runs, machine code, model and restoration (`../benchmarks/linear-memory-broadcast-2026-11-01/README.md`).

Reusing eight accumulators for long AVX2 single-output tails reaches 0.04914 → 0.08345 TFLOPS at K=4097 and 0.04714 → 0.06823 at K=8193 on one core. Integration remains deferred: longer controls confirm about 7–8% losses for four-row GEMM on 32 SMT workers. Production source was restored; prototypes, 1287 runs, two fresh profiles of that regression and independent reviews are preserved in the report (`../benchmarks/linear-single-output-2026-11-01/README.md`). Increased arithmetic work is unproven: both R4 loops contain 34 instructions and aggregate arithmetic CPU time is close. Next checks concern isolating that loop and selecting physical cores for short batches.

## Current candidate: wide GEMM on the physical group, 2026-11-01

Build `e674478b` enables the wide AVX-512 kernel from the selected physical group's properties. Prefill selects that group when row count and width satisfy the kernel's shared geometry limits. Configured pool size and CPU order are preserved. For X `[256, 4096]` and W `[4096, 4096]`, medians of three process medians are **17 threads: 2.342 → 4.248 TFLOPS; 24: 3.029 → 4.246; 32: 3.772 → 4.269**. The 16-thread case reaches 4.235 → 4.260 TFLOPS.

After 326 main and additional control runs, longer series did not reproduce the initial large GEMV/attention losses. Four-row GEMM at 16 threads retains an approximately 2.2% loss; universal improvement is not claimed. A 32-configured-thread profile confirms 16 active workers and 97.50% of aggregate CPU time in `wide::run`. Validation passed 215 workspace tests, 114 release tests, Clippy/fmt, Rust 1.89 and dispatcher tests restricted to CPU 0. Full report, limitations, sources and commands (`../benchmarks/linear-wide-group-2026-11-01/README.md`). The overall maximum-performance goal remains open.

The FMA calibration now verifies placement and runs equal work on every actual worker. On 16 physical cores it consistently measures **5.538 TFLOPS AVX-512** and **2.663 TFLOPS AVX2**; these are register-arithmetic references, not matrix throughput. Calibration changes and read-reference limitations (`../benchmarks/cpu-calibration-workers-2026-11-01/README.md`). A tested 5×64 GEMM block did not provide a repeatable improvement, so 4×64 is retained after repeated production calls and machine-code inspection. Rejected experiment (`../benchmarks/linear-wide-r5-2026-11-01/README.md`).

## Previous candidate: one pool command and fresh profiling, 2026-11-01

Build `c6e41ff7` preserves registered groups and publishes work through one atomic command. Every (group, active-prefix) pair has an independent generation, including wrap and shutdown. At 32 configured threads, 4 MiB GEMV improves **0.622 → 0.955 TFLOPS** over the preceding candidate; 1 MiB improves **0.226 → 0.389 TFLOPS**. The 16-thread 4 MiB case reaches 0.919 TFLOPS and still trails the earlier C4 (0.961). Performance acceptance remains open.

Seven exact-build profiles passed raw-clock audits. Large GEMM spends 87.31% of aggregate CPU time in its K loop and 7.13% packing weights. The pool accounts for 62.99% / 59.84% of 4 MiB GEMV CPU time at 16 / 32 configured threads; these are neither elapsed-latency shares nor speedup estimates. Unprofiled large GEMM reaches 4.228 / 3.782 TFLOPS. Validation passed 214 workspace tests, 113 release tests, Clippy/fmt, Rust 1.89 and 41 threading tests restricted to CPU 0. Profiles, TFLOPS, limitations and reproduction (`../benchmarks/linear-command-publication-2026-11-01/README.md`).

## Previous experiment: independent worker groups, 2026-11-01

Candidate `4179b204` preserves the configured pool size and original CPU ordering. GEMV uses a separately registered group containing one worker per physical core; GEMM and attention still use the native group. At 32 configured threads, AVX-512 GEMV with 4 MiB of weights measured **0.165 → 0.606 TFLOPS**, and 1 MiB **0.0412 → 0.2245 TFLOPS**. Some 16- and 24-thread cases regress, so performance acceptance remains open. Profiling the 32-thread physical group still attributes about 75% of CPU time to the pool. Group design, regressions, validation and raw data (`../benchmarks/linear-worker-groups-2026-11-01/README.md`).

## Previous candidate profiles with the SMT pool, 2026-11-01

Fresh profiling of `aaf5c8a4`, which uses FixedPool for verified bound SMT workers as well, confirms separate priorities. Large 16-worker GEMM spends **85.64%** of aggregate CPU time in the K loop and **8.21%** packing weights. GEMV with 4 MiB of weights spends **58.17% / 93.05%** in the pool on **16 / 32** workers; 1 MiB shares are **78.26% / 97.13%**. CPU-time shares are not potential elapsed-latency savings. All nine profiles passed raw-packet audits; 27 separate unprofiled runs measure speed. Performance acceptance remains pending. Report, TFLOPS, limitations and raw data (`../benchmarks/linear-layout-profile-2026-11-01/README.md`).

## Previous GEMV candidate profiles, 2026-11-01

Ten exact-build profiles were collected after separating C4 GEMV from the prefill scratch frame. Large 16-worker GEMM spends 88.25% of aggregate CPU time in its K loop and 7.06% packing weights. Streaming GEMV spends 98.71% in its K loop. Cached GEMV needs separate scheduling work: pool functions account for 56.63% with 4 MiB weights and 73.53% with 1 MiB. On 32 workers, which use Rayon, those shares reach 89.99% and 97.34%. These are summed CPU-time shares, not an elapsed-latency speedup estimate.

An unprofiled 4 MiB control measured **0.740 / 0.948 / 0.153 TFLOPS** on **8 / 16 / 32** workers: three processes per variant, 61 samples of 2048 calls, explicit 64-byte X/W/Y alignment. Simply reducing worker count does not resolve the issue. GEMM also selects different kernels at 16 and 32 workers (`wide::run` / `x86::avx512::run`), so SMT needs a separate controlled comparison. Broad performance acceptance of the candidate remains pending. Reports, sources, instruction partitions and reproduction (`../benchmarks/linear-current-profile-2026-11-01/README.md`).

## Compact output sums, 2026-11-01

Selected AVX-512 GEMM calls now keep partial sums in a contiguous tile and copy each result to the public output once. This limits repeated writes to misaligned worker-stripe boundaries. Selection preserves the existing wide-kernel conditions and additionally requires at most 256 local rows, K at least 1024, and an output address or stride not divisible by 64 bytes. Packed weights use 32 KiB and temporary sums another 64 KiB. Both stay on the selected function's stack; K order and bias handling are unchanged.

On 16 physical Ryzen 9 9950X cores, large GEMM with output +12 bytes improved from **4.064 to 4.292 TFLOPS**. For 129 tokens and 1025 → 1057 features with output +4 bytes: **2.628 → 3.064 TFLOPS**. These are medians of three process medians, each using 41 samples of eight calls. Applying compact sums to every shape and writing the last K panel directly to output were rejected after regressions. An intermediate dispatcher's extra division and function call were also removed.

Gains are not universal: a separate AVX2 GEMM control measured **2.224 → 2.208 TFLOPS**, and AVX-512 cached GEMV with 4 MiB weights measured **0.864 → 0.850 TFLOPS**. GEMV arithmetic and pool source did not change in this stage; the causes of these small losses are not established. Short cases retain substantial process variation. The build passed 188 workspace tests, 87 release CPU/executor tests, Clippy, fmt, Rust 1.89 and independent review. All shapes, limitations, sources and reproduction (`../benchmarks/linear-compact-output-2026-11-01/README.md`). The maximum-performance objective remains open.

## Fixed worker assignment, 2026-11-01

Production now assigns jobs to fixed workers when they are bound to distinct physical cores. Repeated calls keep the same matrix stripes on the same cores. SMT, unknown and unbound placement retain Rayon. Each executor owns one pool; a common interface serves both linear and attention. Configured thread counts and affinity policies are preserved.

Idle workers park. The calling operation selects its active-wait budget: brief GEMV avoids an unnecessary park/unpark round trip, while attention releases the caller's CPU sooner. This is a general dispatch parameter independent of arithmetic kernels and model descriptions. Required single-worker execution retains direct submission to its bound worker. Weights retain their canonical representation without a persistent duplicate.

The final Ryzen 9 9950X comparison uses three separate processes per case, shuffled order and medians of process medians. Inputs/output are 64-byte aligned; the separate odd-output case explicitly requests an offset. Timing includes validation, packing, submission and completion. The broad series has 41 samples of 8 calls; cached GEMV uses 61 samples of 512. Rows below use AVX-512 and 16 workers unless stated otherwise.

| Workload, input → output | Before, TFLOPS | Current, TFLOPS |
| --- | ---: | ---: |
| GEMM, 256 tokens, 4096 → 4096 | 4.059 | 4.249 |
| GEMM, 1024 tokens, 4096 → 4096 | 4.003 | 4.131 |
| GEMM, 64 tokens, 8192 → 1024 | 3.354 | 3.708 |
| GEMM, 4 tokens, 4096 → 4096 | 0.297 | 1.388 |
| GEMV, 1024 → 1024, 4 MiB weights | 0.234 | 0.847 |
| GEMV, 512 → 512, 1 MiB weights | 0.0712 | 0.3630 |
| GEMV, 8192 → 8192, 256 MiB weights | 0.0380 | 0.0414 |
| AVX2, GEMV, 1024 → 1024 | 0.222 | 0.767 |
| AVX2, GEMM, 256 tokens, 4096 → 4096 | 2.189 | 2.234 |
| 32 workers, GEMM, 256 tokens, 4096 → 4096 | 3.754 | 3.724 |
| 32 workers, GEMV, 1024 → 1024 | 0.1527 | 0.1500 |

The short batch is about 4.7× faster; 4 MiB GEMV is about 3.6× faster. The main GEMM on 15 workers measured 3.733 → 4.009 TFLOPS. SMT-32 stays close to its earlier performance and remains below 16 physical workers on these cases. Process variation is material: one new odd-output cohort was substantially slower than the others. These results do not guarantee gains for every shape or CPU state. Full GEMM/GEMV data (`../benchmarks/linear-fixed-workers-2026-11-01/broad-comparison.json`), cached GEMV (`../benchmarks/linear-fixed-workers-2026-11-01/cached-comparison.json`).

Attention checks use the same updated harness for both builds: 31 samples of 256 calls and identical Q/K/V/Y alignment. Decode on 16 workers measured 0.328 → 0.355 TFLOPS; the tested prefill measured 0.146 → 0.158. Required single-worker medians differed, but process ranges overlapped: approximately 0.0097–0.0114 versus 0.0098–0.0113 TFLOPS. QK+PV counts exclude softmax FLOPs. Attention data (`../benchmarks/linear-fixed-workers-2026-11-01/attention-aligned-comparison.json`).

Fresh profiles of the current binary show cached GEMV spending 37.18% of aggregate CPU time inside its arithmetic kernel and 56.47% in dispatch/waiting, versus 14.38% and 84.63% before this change. Kernel shares are 98.34% for large GEMM, 90.36% for the 64-token case and 98.83% for streaming GEMV. These are not fractions of call latency. Hardware counters remain unavailable; individual hardware stall causes are unmeasured. Reports and raw-sampling audit (`../benchmarks/linear-fixed-workers-2026-11-01/profiles/audit.json`).

A process-wide Rust allocator counter found zero allocation/deallocation calls in ten tested sequences of warmed linear/attention calls on the new backend with 3 and 16 workers. Rayon retains occasional bookkeeping allocations, recorded separately in the audit (`../benchmarks/linear-fixed-workers-2026-11-01/allocation-audit.json`). Tests cover numerical results, borrowed-job joins, wakeups, errors, simultaneous panics and shutdown. Final verification passed 186 workspace tests, 85 release CPU/executor tests, Clippy, fmt and Rust 1.89. Checks (`../benchmarks/linear-fixed-workers-2026-11-01/verification/results.json`), independent review (`../benchmarks/linear-fixed-workers-2026-11-01/review.txt`), reproduction (`../benchmarks/linear-fixed-workers-2026-11-01/reproduce.txt`).

Instruction-address analysis of the current executable refines the picture: large GEMM spends 86.83% of process CPU samples in the K loop and 8.35% packing weights; the 64-token shares are 73.84% and 12.58%. Three longer cached-GEMV repeats confirm 55–57% of CPU time in the pool, including 46–48% of all samples on `pause` while waiting for the next job. Callback dispatch in the worker wrapper itself accounts for only 0.31–0.57%. This lowers the priority of small callback-capture changes and directs investigation toward task granularity and synchronization. Removing active polling still needs a latency comparison: it can increase wakeup time. Analysis, original reports and reproduction (`../benchmarks/linear-fixed-workers-2026-11-01/instruction-analysis/README.md`).

The maximum-performance objective remains open: cached GEMV still spends substantial CPU time coordinating workers, SMT retains its earlier dispatcher, and other GEMM panels/temporary output layouts remain candidates. The conditional FP32 roof for 16 cores simultaneously at 5.7 GHz is **5.8368 TFLOPS**. All-core frequency was not fixed in this series; these results do not establish physical saturation.

## Kernel integration and baseline profiling, 2026-11-01

Large AVX-512 GEMM now uses a 4×64 tile, packing depth 128 and a 64-byte-aligned 32 KiB stack panel. Selection requires at least 128 input rows **per worker**, a worker stripe of at least 64 features, and verified placement on distinct physical cores. The original kernel handles remaining columns with the full output-row stride. Scheduling distributes whole tiles evenly; narrow outputs also partition tokens.

Single-row GEMV gains a sequential weight-row pass with eight accumulators when weights exceed twice the aggregate selected LLC capacity. Selection also requires at least eight workers and eight stripes. Unknown cache capacity retains the original path. Metadata is read at executor construction; there is no persistent packed model copy. Policies depend on shapes and topology, without model or processor names.

The following unprofiled Ryzen 9 9950X comparison uses three separate processes per case, shuffled order, 41 samples of eight calls, and the same explicit 64-byte input/weight alignment. Timing includes checks, packing, worker submission and completion. “Before” uses the previous `linear.rs` and `executor.rs` with the same updated harness and current pool-construction code; this is a separate normalized series.

| ISA / threads / tokens / input → output | Before, TFLOPS | Current, TFLOPS |
| --- | ---: | ---: |
| AVX-512 / 16 / 256 / 4096 → 4096 | 3.792 | 4.054 |
| AVX-512 / 16 / 1024 / 4096 → 4096 | 3.629 | 4.006 |
| AVX-512 / 32 / 256 / 4096 → 4096 | 3.745 | 3.751 |
| AVX-512 / 15 / 256 / 4096 → 4096 | 3.529 | 3.722 |
| AVX-512 / 16 / 129 / 513 → 1057, output +4 bytes | 2.021 | 2.284 |
| AVX-512 / 16 / 64 / 8192 → 1024 | 3.346 | 3.348 |
| AVX2 / 16 / 256 / 4096 → 4096 | 2.213 | 2.178 |
| AVX-512 / 16 / 1 / 8192 → 8192 | 0.0343 | 0.0377 |
| AVX2 / 16 / 1 / 8192 → 8192 | 0.0368 | 0.0388 |

Gains are not universal: AVX2 GEMM is 1.6% lower in this series. Additional 61-sample series cover short batches and cached GEMV. The four-token workload varies with offset and process; 1024² GEMV on eight workers measured 0.303 → 0.264 TFLOPS with substantial process variation. These remain open limitations; overall performance acceptance is incomplete. Full data: main series (`../benchmarks/linear-integration-2026-11-01/aligned-verified-comparison.json`), short batches (`../benchmarks/linear-integration-2026-11-01/aligned-verified-smallbatch.json`), cached GEMV (`../benchmarks/linear-integration-2026-11-01/aligned-verified-cached.json`).

Repeated profiling of the verified current executable gives the following shares of aggregate **exclusive CPU time** on 16 workers:

| Workload | Kernel functions | Rayon/Crossbeam and `sched_yield` |
| --- | ---: | ---: |
| GEMM: 256 tokens, 4096 → 4096 | 98.68% | 1.20% |
| GEMM: 64 tokens, 8192 → 1024 | 88.33% | 10.74% |
| GEMV: 8192 → 8192, 256 MiB weights | 99.25% | 0.64% |
| GEMV: 1024 → 1024, 4 MiB weights | 14.38% | 84.63% |

For large GEMM, the K loop accounts for 85.28% of process CPU samples, packing 9.68%, and partial-sum reloads 1.37%. The corresponding 64-token shares are 66.85%, 17.75% and 2.62%. These guide further work on short-task scheduling and GEMM packing/inner loops. CPU-time shares are not fractions of call latency and do not prove FMA or memory saturation. Hardware `perf` events remain unavailable; clock sampling does not measure cache misses or stall causes.

Profiles use 2 ms sampling only around timed calls. Raw-packet auditing found no samples outside the gate and reconciled CPU ticks exactly against reports. A separate series was rejected: restoring a source file with an old modification time caused Cargo to reuse an experimental library. The library was rebuilt and current dispatcher symbols verified. Reports and audit (`../benchmarks/linear-integration-2026-11-01/profiles/audit.json`), instruction partition (`../benchmarks/linear-integration-2026-11-01/ranges.json`), reproduction commands (`../benchmarks/linear-integration-2026-11-01/reproduce.txt`).

After rebuilding, 172 workspace tests and 71 release CPU/executor tests passed, along with Clippy denying warnings, formatting and Rust 1.89 all-target checks. Logs (`../benchmarks/linear-integration-2026-11-01/verification/results.json`). The conditional FP32 roof for 16 cores simultaneously at 5.7 GHz is **5.8368 TFLOPS**; that frequency assumption is not a measured sustained roof for this series. SMT does not double physical execution units.

## Worker placement comparison

On the same Ryzen 9 9950X, measured 2026-10-31 with release/LTO, the saved previous binary was compared with the current binary in both `none` and `required` modes. Each case ran in three independently started processes, with shuffled case order. The table reports median process medians. All variants used process masks `0–7`, `0–15` or `0–31`; only `required` pinned individual workers. Prefill below uses 256 × 4096 × 4096 (51 samples per process).

| ISA / workers | Previous unbound | Current unbound | Current bound | Bound change vs previous |
| --- | ---: | ---: | ---: | ---: |
| AVX2 / 8 | 1.19 TFLOPS | 1.21 TFLOPS | 1.24 TFLOPS | +4.3% |
| AVX2 / 16 | 1.24 TFLOPS | 1.24 TFLOPS | 2.14 TFLOPS | +72.2% |
| AVX2 / 32 | 2.06 TFLOPS | 1.83 TFLOPS | 1.87 TFLOPS | -9.2% |
| AVX-512 / 8 | 2.22 TFLOPS | 2.16 TFLOPS | 2.24 TFLOPS | +1.3% |
| AVX-512 / 16 | 2.19 TFLOPS | 2.24 TFLOPS | 3.71 TFLOPS | +69.2% |
| AVX-512 / 32 | 2.69 TFLOPS | 2.61 TFLOPS | 3.49 TFLOPS | +29.9% |

For AVX-512 on 32 workers, bound prefill process medians ranged from 2.428 to 2.463 ms, versus 2.849–3.241 ms before. On 16 workers the bound range was 2.293–2.428 ms. Thus 32 workers improved over the previous implementation but remained about 6% slower than 16 on this shape. AVX2 on 32 workers did not improve; its bound range was wider (4.078–5.993 ms). Binding is configurable rather than a universal speed guarantee.

Decode (8192 × 8192, AVX-512 / 32) stayed at approximately 4.09 ms. Smaller matrices and attention showed more run-to-run variation; the complete results include these cases and outliers. Clocks, background load and NUMA memory placement were not controlled. Checksums agree between revisions and placement policies at each fixed ISA, shape and worker count; different partitions may change f32 rounding.

A separate five-process, 501-sample experiment compared static per-worker prefill stripes with the dynamic Rayon scheduler, using controlled aligned and offset outputs. Some AVX-512 cases improved, AVX2 slightly regressed and static outliers worsened. The static path was removed: the final implementation changes placement, preserving dynamic scheduling and kernel arithmetic.

Raw log (`../benchmarks/2026-10-31-threading.txt`), all measurements (`../benchmarks/2026-10-31-threading.json`), comparison runner (`../benchmarks/compare-threading.py`), static experiment (`../benchmarks/2026-10-31-threading-static-experiment.json`). The runner needs saved before/after binaries; its CPU masks describe this host and must be adapted on other hardware.

## GEMM packing hypothesis checks

On 2026-10-31, a separate diagnostic binary compared seven AVX-512 GEMM variants on the same Ryzen 9 9950X. The process ran only on CPU 15 with `nice -n 19`, compiled with `rustc -C opt-level=3`. Measurements cover the raw kernel without `CpuExecutor`. The table reports medians of nine samples after two warm-ups per variant, with variant order shuffled each round. Shape: 256 × 4096 × 4096; output aligned to 64 bytes.

| Variant | Kernel time, ms |
| --- | ---: |
| Baseline packing in K=128 blocks | 29.117 |
| Same blocks, scratch using `MaybeUninit` | 29.135 |
| Retained packed weights, K=128 blocks | 28.204 |
| Retained packed weights, full K per pass | 26.325 |
| Packing in K=1024 blocks, without a full weight copy | 28.259 |

Removing scratch zero-initialization gave no measurable benefit here. Retained packing with a full-K pass increased throughput by approximately 10.6% and reduced time by 9.6%. With output offset by 12 bytes, time changed from 30.047 to 26.307 ms. Both retained-weight variants, however, require an additional 64 MiB for this shape. Packing into an already allocated buffer separately took 3.69–3.74 ms; adding its median to the kernel median excludes allocation and is not an end-to-end measurement. The full-K pass changes both data reuse and K-block overhead, so the result does not isolate one cause of the improvement.

Larger stack blocks without a full weight copy gave inconsistent results: almost no gain on 128 × 2048 × 2048, and slower execution than baseline for K=256/512/1024 on 8 × 512 × 4096. The latter case used 11 samples of four calls each. These single-core results do not establish performance on 16/32 threads or end-to-end inference; production code is unchanged.

All seven variants matched baseline bits for every output of the checked matrices. They also passed an independent f64 oracle on nine small shapes with and without bias, plus checks of 16 selected outputs from each large matrix. Complete log, including individual samples (`../benchmarks/2026-10-31-gemm-hypotheses.txt`), experiment preparation script (`../benchmarks/gemm-hypotheses.py`), measurement harness (`../benchmarks/gemm-hypotheses-main.rs`).

## Follow-up: GEMM tile size and scratch space

The next series compared 16 variants on one core of the same Ryzen 9 9950X: CPU 15, `nice -n 19`, and a standalone binary compiled with `rustc -C opt-level=3`. Two warm-ups preceded seven samples; the large-matrix repeat used nine, and the small case used nine samples of four calls. Packing is included, allocation is outside timing, and no full weight copy is retained. The `heap128_r8` control keeps the original 8×32 tile and K=128, using caller-provided scratch instead of stack storage.

The `wide4x64_k128` candidate showed a useful balance of speed and memory: four rows, 64 output columns, K=128, and just 32 KiB of scratch. The shared harness allocated a larger buffer for other variants outside kernel calls. Below are medians and throughput gains over the original kernel; output is 64-byte aligned except where an offset is stated.

| Shape N × M × K | Original kernel, ms | 4×64/K128, ms | Throughput gain |
| --- | ---: | ---: | ---: |
| 256 × 4096 × 4096 | 29.197 | 27.967 | +4.4% |
| Same shape, repeat | 29.297 | 28.011 | +4.6% |
| 128 × 2048 × 2048 | 3.754 | 3.585 | +4.7% |
| 64 × 1024 × 8192 | 4.349 | 4.093 | +6.3% |
| 256 × 4096 × 4096, output +12 bytes | 30.208 | 28.515 | +5.9% |

The 4×64 assembly hot loop has 16 FMAs, four broadcasts and four weight loads: 28 instructions per K step, without spills. Increasing the row tile to 12/14 did not help; unrolling K by two reduced instruction count but gave no consistent speedup. Compact partial-output scratch improved throughput by about 3% on aligned large output and 6% with an offset. Packing the entire K depth needed 512 KiB at K=4096 and gained roughly 3–5% on the large matrix, but reduced small-case throughput by about 7%.

On 8 × 512 × 4096, the 4×64 candidate was about 1% slower than `heap128_r8`. This diagnostic forced packing; production chooses another path for eight rows. These results do not establish gains for multiple cores or end-to-end inference; production code is unchanged.

All 16 variants passed bitwise baseline comparison and an independent f64 oracle on 21 small shapes with and without bias. Every measured shape also had all outputs compared bitwise and 16 selected values checked against f64. Full report and measurements (`../benchmarks/2026-10-31-gemm-panels.txt`), experiment generator (`../benchmarks/gemm-panels.py`).

## Multicore GEMM and CPU profiling

The 2026-10-31 continuation tested 16 physical cores and 32 SMT workers on the Ryzen 9 9950X. The main repeated 16-worker comparison improved from **3.746 to 4.016 TFLOPS**, a **7.2%** throughput gain. This remains an experimental 4×64/K128 kernel: another shape regressed, so it has not replaced the production kernel.

The main task is 256 tokens, projecting 4096 input features to 4096 output features: `X[256,4096] × W[4096,4096]ᵀ → Y[256,4096]`. Results are medians of three process medians, each with 41 shuffled paired samples of eight calls. Weight packing, shape/ISA checks and persistent-pool dispatch are timed; tensor/scratch allocation and correctness checks are excluded. Workers are individually pinned to CPUs 0–15 or SMT pairs 0,16,1,17,…,15,31. The standalone binary uses `rustc -C opt-level=3 -C codegen-units=1 -C debuginfo=1`, without Cargo LTO.

| Workers and kernel | TFLOPS | Gain over same-worker baseline | Fraction of nominal 5.8368 TFLOPS |
| --- | ---: | ---: | ---: |
| 16, baseline 8×32/K128 | 3.746 | — | 64.2% |
| 16, 4×64/K128 | 4.016 | +7.2% | 68.8% |
| 32 SMT, baseline 8×32/K128 | 3.478 | — | 59.6% |
| 32 SMT, 4×64/K128 | 3.498 | +0.6% | 59.9% |
| 32 SMT, 4×64/K64 | 3.612 | +3.9% | 61.9% |

The nominal FP32 FMA reference is `16 cores × 2 FMA × 16 lanes × 2 operations × 5.7 GHz = 5.8368 TFLOPS`. It assumes **every core sustains 5.7 GHz**, which was not measured under this workload. SMT does not double the physical execution units. Sources: [AMD's Zen 5 diagram](https://hc2024.hotchips.org/assets/program/conference/day2/24_HC2024.AMD.Cohen.Subramony.final.pdf#page=9) and [9950X specifications](https://ir.amd.com/news-events/press-releases/detail/1202/amd-unveils-next-gen-zen-5-ryzen-processors-to-power-advanced-ai-experiences).

Separate independent register-FMA calibration reached **5.538 TFLOPS** with 16 pinned workers and **5.521 TFLOPS** with 32. Main-shape GEMM reaches 72.5% of the first empirical reference. Unsynchronized `cpuinfo_avg_freq` samples cover other process phases as well as timed calls; they are not used as an exact frequency-normalized efficiency denominator. See the [Linux interface documentation](https://www.kernel.org/doc/html/latest/admin-guide/pm/cpufreq.html).

Shape limits matter. A single process with 1024 tokens and the same feature dimensions measured **3.845 → 4.280 TFLOPS**; that is a different task. For 64 tokens, 8192 → 1024 features, two comparisons measured **3.213 → 2.979** and **3.203 → 2.921 TFLOPS**, a 7–9% regression. Main-shape output offset by 12 bytes measured **3.616 → 3.921 TFLOPS**. Smaller tasks, separate CCD queues and per-call input packing gave no consistent general benefit. Input packing included its own cost and required an additional 4 MiB input buffer.

A later comparison of other 32-worker candidates showed substantial absolute drift: baseline 3.051–3.355 TFLOPS, compact partial-output scratch 3.230–3.528 TFLOPS, and 4×64/K64 3.316–3.436 TFLOPS. The table therefore records one repeated series, not a final SMT dispatch policy. Temperature, clocks and background load were not held constant.

### What the profiles show

Hardware `perf` events are unavailable with the current `perf_event_paranoid=4`. The installed `gprofng collect app -p 2` provided instruction-pointer clock sampling without changing system settings. Five profiles cover whole processes, including warmups and checks: 101×32 main-shape calls or 101×128 calls for the 64-token shape. Percentages below concern accumulated **worker CPU time**, not sequential wall-time phases of one GEMM.

| Code region | Baseline, 256 tokens | 4×64/K128, 256 tokens | Baseline, 64 tokens | 4×64/K128, 64 tokens |
| --- | ---: | ---: | ---: | ---: |
| Weight packing | 8.84% | 10.10% | 18.73% | 19.65% |
| Inner K loop: loads, addressing, FMA | 83.61% | 84.47% | 67.72% | 63.27% |
| Partial-accumulator reloads | 4.67% | 1.93% | 2.95% | 8.09% |
| Accumulator stores | 0.72% | 0.99% | 0.45% | 0.57% |
| Other kernel code | 0.66% | 0.70% | 0.63% | 0.66% |
| Outside selected kernel | 1.50% | 1.81% | 9.52% | 7.76% |

The large shape spends most CPU time inside GEMM. For the 64-token shape, sampled kernel CPU time increases by 4.172 seconds; the reload region accounts for 3.888 seconds of that increase. This is a useful investigation lead, not proof of cache misses or another hardware stall. Whole profiled process times were nearly tied at 5.159 versus 5.148 seconds; separate unprofiled runs establish the throughput regression.

Follow-up tests were mixed. Reducing K to 64/32 worsened the 64-token shape. Keeping 8×32 and increasing K to 1024 measured **2.905 → 3.103** and **2.968 → 3.130 TFLOPS** in two paired processes, requiring 128 KiB scratch instead of 16 KiB. Full-K packing regressed. No automatic policy was adopted from this small shape set.

Profile partitions sum disjoint instruction ranges and are checked against the entire function's sampled time. Inclusive Rayon `wait_until_cold` includes nested GEMM execution; it is not waiting time. A sample at an FMA does not establish that instruction's latency. Off-CPU blocked time is absent from this CPU sample distribution.

Tested modes passed full-output bitwise comparisons, an independent f64 oracle on 24 small shapes with/without bias, row/column tails, and untouched gaps between strided output stripes. These checks belong to the diagnostic harness; production `linear.rs` is unchanged by this series.

Artifacts: measurements and commands (`../benchmarks/2026-10-31-gemm-multicore.txt`), generator (`../benchmarks/gemm-multicore.py`), harness (`../benchmarks/gemm-multicore-main.rs`), profiles (`../benchmarks/2026-10-31-gemm-profile.txt`), profile partition (`../benchmarks/2026-10-31-gemm-profile-breakdown.json`), range analysis script (`../benchmarks/gemm-profile-ranges.py`), frequency telemetry (`../benchmarks/2026-10-31-gemm-frequency.json`). Reproduction instructions are appended to the measurement log.

## Production executor profiles: GEMM and GEMV

The next 2026-10-31 series profiles the actual `CpuExecutor::linear` build. Collection is enabled only around timed calls; allocation, pool construction, warmup and f64 checks remain outside. Binary hashes, source snapshots, reports and an independent sampling-cadence audit (`../benchmarks/linear-profile-2026-10-31/sampling-audit.json`) are archived.

| Workload | Threads | Inside the kernel function | Rayon/Crossbeam and `sched_yield` |
| --- | ---: | ---: | ---: |
| GEMM: 256 tokens, 4096 → 4096 features | 16 | 98.63% | 1.14% |
| Same workload | 32 | 96.72% | 2.99% |
| GEMM: 64 tokens, 8192 → 1024 features | 16 | 87.87% | 11.15% |
| GEMV: one token, 8192 → 8192, 256 MiB weights | 16 | 99.49% | 0.45% |
| GEMV: one token, 1024 → 1024, 4 MiB weights | 16 | 14.81% | 84.20% |
| Same workload, inline on one pinned core | 1 | 99.48% | 0% |

These are shares of aggregate **exclusive CPU time**, not fractions of one call's latency. The kernel function includes computation, packing and memory accesses. Other functions account for the remainder. Hardware counters remain unavailable; these samples do not measure cache misses, execution-port stalls or physical memory limits.

For the large GEMM, the inner K loop accounts for 84.73% of process CPU samples, packing 9.32%, and partial-sum reloads 3.28%. The corresponding shares for the 64-token workload are 65.14%, 18.77% and 2.69%. Disjoint instruction ranges reconcile exactly against function totals: data (`../benchmarks/linear-profile-2026-10-31/ranges.json`), analysis script (`../benchmarks/linear-profile-ranges.py`).

Unprofiled latency was measured separately for the 4 MiB GEMV: three processes per thread count, 31 samples of 1024 calls, shuffled process order. Medians of process medians were **0.062 / 0.102 / 0.188 / 0.253 / 0.219 / 0.142 TFLOPS** for **1 / 2 / 4 / 8 / 16 / 32** threads. Thus a high pool CPU share does not make one thread fastest; eight was best among the tested counts for this shape. These measurements have not yet become an automatic worker-count policy. Raw measurements (`../benchmarks/linear-profile-2026-10-31/cached-gemv-threads/manifest.json`).

The redundant-addressing hypothesis was also tested. Hoisting eight input-row bases reduced the K loop from 37 to 30 instructions in one diagnostic executable and gained about 2% in paired measurements. That gain did not survive integration into the production build: the main shape changed from 3.842 to 3.821 TFLOPS, and a shape with tails from 1.666 to 1.587 TFLOPS. The change was reverted; it passed correctness checks but failed performance acceptance. Production-build comparison (`../benchmarks/linear-profile-2026-10-31/addressing-production-comparison.json`).

The installed GNU gprofng 2.46 has a timer-initialization defect: its caller thread was sampled every 100 ms despite the requested 2 ms. The first series was rejected. The reported results use an isolated diagnostic shim guarded by the collector's exact build ID; neither system libraries nor the engine depend on it. Repeated auditing confirmed a 2 ms cadence, no warnings and no samples outside the timed interval. Defect evidence (`../benchmarks/linear-profile-2026-10-31/collector-timer-evidence.txt`), reproduction commands (`../benchmarks/linear-profile-2026-10-31/reproduce.txt`).

## Initial measurements before worker placement

Historical baseline on Ryzen 9 9950X, measured 2026-10-31 with Rust 1.97.1, release + LTO. Single-thread runs use CPU 0; 16-worker runs are restricted to CPUs 0–15, one logical CPU per physical core. Individual workers were not pinned within that mask. Matrix values below are medians of 25 warmed samples. These are not measurements of the current default placement; use `none` for an equivalent unbound policy.

| Matrix / mode | AVX2, 1 thread | AVX-512, 1 thread | AVX2, 16 threads | AVX-512, 16 threads |
| --- | ---: | ---: | ---: | ---: |
| Decode: 1 × 8192 × 8192, 256 MiB weights | 4.897 ms | 4.835 ms | 3.805 ms | 3.966 ms |
| Decode effective bandwidth | 54.8 GB/s | 55.5 GB/s | 70.6 GB/s | 67.7 GB/s |
| Prefill: 128 × 2048 × 2048 | 6.724 ms | 3.675 ms | 0.567 ms | 0.564 ms |
| Prefill throughput | 159.7 GFLOP/s | 292.1 GFLOP/s | 1893.3 GFLOP/s | 1902.3 GFLOP/s |

The one-thread scalar f32 path took 25.531 ms for decode and 187.983 ms for prefill. For these shapes, one-thread AVX-512 is approximately 5.3× and 51× faster. These are kernel comparisons, not model-generation speedups.

The 256 MiB read calibration reached 76.1 GB/s with AVX2 and 16 workers; decode reached about 93% of that observed baseline. Initial register-FMA calibration reached 365.5 GFLOP/s with one AVX-512 thread and 2876.0 GFLOP/s with 16 workers. Subsequent per-worker pinning showed that the latter value substantially understated available arithmetic throughput; it must not be used as a hardware ceiling or an efficiency denominator. These workload-specific observations are not interchangeable theoretical ceilings.

AVX-512 improves compute-heavy single-core prefill here but does not beat AVX2 in multi-thread decode. The widest ISA is not always fastest. Do not transfer these ratios to other hardware, shapes, cold memory, or end-to-end inference.

The complete command/result log (`../benchmarks/2026-10-31-cpu.txt`) records shapes, checksums and sample counts.

Other kernels, one thread, 4096 elements (median microseconds):

| Operation | Scalar | AVX2 | AVX-512 |
| --- | ---: | ---: | ---: |
| RMSNorm | 2.330 | 0.423 | 0.246 |
| Residual | 0.720 | 0.141 | 0.127 |
| SwiGLU | 9.248 | 1.537 | 1.018 |
| GeGLU | 27.921 | 1.998 | 1.168 |
| RoPE, 32 heads × 128, one token | 2.974 | 1.201 | 1.131 |

The separate `bench_attention` dataset with history 2048, 32 Q heads, 8 KV heads and dimension 128 measured **14.026 ms** for Scalar / 1 thread, **1.189 ms** for AVX-512 / 1 thread, and **0.190 ms** for AVX-512 / 8 threads. Maximum absolute error against the f64 oracle was approximately `6.1e-6`. With 16 workers, the median was worse at **0.214 ms**: more workers are not always beneficial.

## Remaining limits

This implementation covers dense f32. Optimized LTQ/BF16/INT8 execution, GPU and offload remain future work. Prefill attention does not yet use a multi-query tiled QK matrix kernel. Graph activations are not reused yet; sliding attention retains full history and derived-history operations can still materialize full intermediates.

`MemoryReport` covers tensor payload, excluding bounded kernel stack tiles and worker stacks/metadata. Account for them separately in process budgets. Affinity currently uses Linux sysfs and scheduler APIs; other platforms retain unbound execution with a diagnostic under `Auto`. NUMA memory binding is not implemented; graph preparation now handles weights used exclusively by Linear; scheduling adaptation and the explicit prepared-weight API are described above. No universal hardware-saturation claim follows from these microbenchmarks.
