# GEMM and GEMV performance implementation plan

> **For agentic workers:** Use the executing-plans workflow for coordinated implementation. Root owns all builds and measurements; agents must not run competing CPU workloads.

**Goal:** Bring the production FP32 GEMM and GEMV paths to the strongest performance supported by measurements, while preserving their correctness, memory bounds and configurable threading.

**Architecture:** Keep canonical row-major weights and the checked public linear API. Separate arithmetic kernels from workload partitioning. Use bounded packing and ISA-specific microkernels; choose policies by matrix shape, available workers and measured machine capabilities, never by model-family names.

**Tech stack:** Rust 2024, MSRV 1.89, AVX2/FMA, AVX-512F, Rayon, Linux affinity, independent f64 reference, gprofng clock sampling.

**Spec:** The active user goal and existing contracts in `docs/EN/architecture.md`, `docs/EN/blocks.md`, `crates/lumen-cpu/src/linear.rs`; prior measurements in `docs/EN/cpu-performance.md` are evidence, not completion criteria.

## Constraints and review focus

- Preserve positive M/K, zero-row support, exact slice validation and unchanged output on validation/ISA failure.
- Preserve arbitrary f32 alignment, odd dimensions, bias/no-bias, cancellation accuracy and disjoint strided output ownership.
- No persistent duplicate weight matrix or model-specific kernel switches. Document all scratch costs and avoid per-call heap allocations in the hot path.
- Thread count remains user-configurable; measure 1/8/16/32 workers and cache-resident versus streaming matrices. SMT is not an extra set of physical execution units.
- Report compute in TFLOPS. Include packing, dispatch and synchronization in speed comparisons. Report frequency assumptions separately from measured throughput.
- A single fast shape, a best sample, or passing correctness tests cannot establish maximum performance. Compare repeated processes, latency tails, observed memory throughput and an optimized external implementation when available.
- Work in the supplied checkout: required kernel/foundation files are already uncommitted here. Preserve unrelated work and save source/binary snapshots for comparison. No branch merge or push is required by this goal.

## Tasks

### 1. Establish comparable measurements

- [x] Update `examples/bench_linear.rs` with TFLOPS, repetitions per sample, latency distribution and an untimed sampled f64 check.
- [x] Update `examples/bench_ceiling.rs` with the same placement policy and an explicitly labelled observed read-bound GEMV estimate.
- [ ] Build and save the current production baseline, hashes and commands. Measure narrow/small batches, GEMV cache regimes, medium/deep/large GEMM, odd output strides, AVX2/AVX-512 and worker counts.
- [x] Compare against an optimized BLAS if one is available; do not use unoptimized Netlib BLAS as a performance target.

### 2. Use available workers for small batches

- [x] Demonstrate the existing 2–15-row underutilization with a baseline benchmark: current token partitioning produces at most four jobs.
- [x] Extend the existing disjoint feature-stripe path to those workloads. Keep tiny operations inline and preserve four-row reuse inside each stripe.
- [x] Add meaningful parallel tests for rows 2/3/4/7/15, odd feature widths, unaligned inputs/output, bias, guards and invalid extents.
- [x] Compare the actual production binary before/after across small batches and both ISAs; retain only a verified improvement without damaging the established paths.

### 3. Optimize GEMV arithmetic and memory access

- [x] Profile and compare output-row tiling, independent accumulators and reduction overhead in cache and streaming regimes.
- [x] Validate candidates against independent f64 results, including long odd K and cancellation, before timing them.
- [x] Integrate the supported streaming ISA policies in production, retaining safe tails and bounded memory. Verify actual engine dispatch; keep cache-resident scheduling and a physical bandwidth-efficiency claim open.

### 4. Optimize GEMM structure and microkernels

- [x] Compare the current column-stripe/K-panel scheme with a bounded cache-blocked design and an optimized external baseline. Use profiles to select changes rather than enumerate constants indefinitely.
- [x] Validate AVX2/AVX-512 microkernels and all edge tiles before parallel integration.
- [ ] Integrate an evidence-based production policy covering small, deep and large batches. Include scratch initialization/packing costs; do not hide a regression behind an isolated kernel number.
- [x] Recheck scheduling and SMT after the arithmetic changes, including 32-column/odd tails and misaligned outputs.

### 5. Verification and completion audit

- [x] Run targeted release correctness tests, workspace tests, clippy, fmt and MSRV checks after production changes.
- [x] Obtain an independent safety/correctness review of the current production diff; resolve important findings. Repeat for subsequent unsafe changes.
- [x] Save reproducible paired performance results and update RU/EN documentation with tradeoffs and residual limits.
- [ ] Audit the complete GEMM/GEMV goal against production behavior and all matrix/thread/cache regimes. Keep the goal active while a material known gap, unintegrated winning implementation, or missing validation remains.

## Evidence log

The initial 2026-10-31 phase produced reproducible CPU profiles and correctness-checked candidates without changing the production arithmetic. Integration and remaining performance work were still open at that point.

The small-batch implementation now partitions both output features and token groups. Feature-only partitioning regressed narrow outputs; the two-dimensional policy removed that regression. Three-process comparisons and the 10 release linear tests passed. Raw series remain `/tmp/lumen-goal-{smallbatch,smallbatch-narrow,2d}-comparison.json` (the narrow-series filename is `/tmp/lumen-goal-smallbatch-narrow.json`).

The profiling follow-up is archived under `docs/benchmarks/linear-profile-2026-10-31/` and explained in both CPU performance documents. Six production profiles exclude setup/oracles. An independently audited, isolated fix for the installed GNU collector's timer initialization was necessary: the first profiles undersampled the caller and were rejected. Corrected samples have the requested 2 ms cadence and reconcile exactly with report CPU totals.

Cached GEMV spends most aggregate CPU time on the worker pool. A separate unprofiled sweep found eight workers faster than sixteen on the 4 MiB shape; one worker was slower. Large GEMM spends most CPU time in its K loop, with measurable packing cost. Hoisting row addresses passed numerical checks and helped a diagnostic binary, but failed the seven-case production comparison and was reverted exactly. No new arithmetic policy is accepted from that experiment.

After this follow-up: 165 workspace tests pass; Clippy with warnings denied, formatting, MSRV 1.89 all-target checks and Python syntax/range reconciliation pass. These checks cover the current state, not future arithmetic changes. The full goal remains open: verified GEMM/GEMV policies still require production integration and broader performance acceptance.

### Production integration and current profiling, 2026-11-01

The production code now includes an aligned 32 KiB AVX-512 4×64/K128 panel,
single-row streaming AVX2/AVX-512 kernels, optional LLC capacity discovery at
startup, balanced tile partitioning and a common globally checked dispatcher.
Wide selection requires at least 128 rows per worker and verified distinct
physical cores. Streaming requires weights larger than twice the aggregate
selected LLC, at least eight workers and at least eight stripes. Unknown
metadata retains the ordinary path. No model names or persistent duplicate
weights were introduced.

Tests cover f64 accuracy, cancellation, bias, odd shapes, unaligned slices,
strided row gaps and malformed full extents before writes. Private tests force
the streaming/wide parallel branches with three workers even on unbound or
unknown-topology hosts. Independent read-only review found no important safety
or correctness defect; policy-boundary unit tests were a minor future suggestion.

The updated benchmark can explicitly align X/W as well as Y while retaining
identical logical data. Three normalized paired series are archived in
`docs/benchmarks/linear-integration-2026-11-01/`. Main AVX-512 GEMM changed from
3.792 to 4.054 TFLOPS, deep-batch GEMM from 3.629 to 4.006, odd-width GEMM from
2.021 to 2.284, and streaming GEMV from 0.0343 to 0.0377. AVX2 streaming GEMV
changed from 0.0368 to 0.0388. SMT-32 and the 64-token case were essentially
unchanged. These series are not proof of universal acceptance: AVX2 GEMM was
1.6% lower; short-batch results vary by offset/process, and cached GEMV on eight
workers measured 0.303 to 0.264 TFLOPS with substantial process variation.

Fresh profiles of the current verified executable show kernel/pool CPU shares
of 98.68/1.20% for large GEMM, 88.33/10.74% for 64-token GEMM, 99.25/0.64% for
streaming GEMV and 14.38/84.63% for 4 MiB cached GEMV. Large GEMM's K loop and
packing account for 85.28% and 9.68% of process CPU samples. This prioritizes
short-task scheduling and GEMM inner-loop/packing work. Hardware stall causes
remain unmeasured; CPU sampling does not establish physical saturation.

One intermediate build reused the rejected const-specialized library because
restoring the source preserved an old modification time. Its profile series
was rejected. The accepted library was forcibly rebuilt; current dispatcher
symbols and source hashes were checked. Fresh verification then passed 172
workspace tests, 71 release CPU/executor tests, Clippy denying warnings, fmt
and Rust 1.89 all-target checks. The new raw-packet audit also reproduces all
six previous independently audited profile totals and checks gated sampling.

The goal stays active. Complete cross-regime performance acceptance, cached
GEMV scheduling, the small AVX2 regression, short-batch variability and the
larger-panel deep-GEMM candidate remain unresolved. Do not mark maximum
performance or a physical roof reached from the main-shape gain alone.

### Fixed worker assignment, 2026-11-01 follow-up

Previous goal turn: progress. It produced audited current profiles, normalized
paired measurements and reproducible source/binary records, identifying cached
GEMV scheduling as the next material bottleneck.

A separate temporary crate at `/tmp/lumen-fixed-pool-2026-11-01` now compares the
same arithmetic under Rayon and fixed worker assignment. The v4 hybrid sleeps
idle workers and a waiting caller; it publishes stack-borrowed jobs through an
epoch and joins per-worker packed completion/waiter states. No hot allocations
or duplicate model weights. It retains every panic payload until all workers
join; multiple payloads transfer as one caller-owned aggregate. An independent
review accepted the actual v4 synchronization and panic correction. Forced
fixed-dispatch tests cover the f64 oracle, streaming/wide tails, narrow outputs,
bias, unaligned buffers and guards.

Three-process comparisons establish a reason to integrate: main GEMM is
4.058 → 4.291 TFLOPS, four-token GEMM 0.292 → 1.362, and v3 cached GEMV
0.230 → 0.730. Same-binary Rayon controls are included. A pure-spin 32-worker
experiment created 33 runnable threads on 32 logical CPUs and was intentionally
stopped; continuous spinning and the temporary attention fallback are rejected.

Implementation decisions within the authorized optimization goal:

- Own exactly one worker backend, behind a private common dispatch interface.
- Select fixed assignment for verified distinct physical cores with >1 worker.
  Keep Rayon for SMT, unbound/unknown placement, and required single-worker
  execution. The configured pool size and affinity/fallback API remain intact.
- Route both linear and attention through that interface. A scoped mutable-chunk
  helper owns disjoint slices, propagates errors after joining, and handles tails.
- Keep arithmetic kernels unchanged; reuse the balanced rectangular linear plan.
- Keep fixed dispatch private to leaf kernels. Same-pool nesting is inline;
  arbitrary cross-pool callback cycles are not a supported execution mechanism.
- Preserve startup binding checks, partial-construction cleanup, shutdown joins,
  zero-work handling, shape failures before writes, and Rust 1.89 support.

Remaining steps: RED→GREEN common dispatch tests; integrate the reviewed fixed
pool and both consumers; test forced arithmetic on both backends and single-worker
affinity; run release/workspace/Clippy/fmt/MSRV; independently review the actual
production integration; repeat paired end-to-end timings including attention and
SMT fallbacks; archive evidence. Acceptance of the whole performance goal remains
open after this scheduling improvement.

### Fixed backend integrated and verified

The scheduling integration above is now complete in the actual worktree:
`src/threading/fixed.rs` owns the reviewed borrowed-job protocol;
`src/threading/pool.rs` supplies joined ranges, disjoint mutable chunks and a
single scoped FnOnce/result transfer. `Workers` owns exactly one backend.
Linear and attention both use the interface. Fixed selection requires confirmed
distinct physical cores and >1 worker; other placement and required-one-worker
execution retain Rayon. No persistent weights or second pool were added.

Common chunk dispatch and single-job transfer were watched RED and then GREEN.
The multiple-panic regression was reproduced with Drop-panicking payloads before
the per-worker ownership fix. Fixed/Rayon linear contracts share f64/guard tests,
including more jobs than workers. A forced fixed attention test covers decode
head tails, five-query partitions, windows, unaligned slices and malformed full
extents before writes. Final verification: 186 workspace tests, 85 release
CPU/executor tests, Clippy denying warnings, fmt and Rust 1.89 all targets.
Independent actual-source review found no remaining Critical/Important defect.

The initial attention smoke exposed a smaller-prefill regression. A single
global caller-spin budget was insufficient. Repeated controls with identical
aligned buffers established generic completion hints: linear uses Long (4096),
attention chunks Short (256). Required-one-worker routing delegates directly to
the original install path. The hints leave all arithmetic and the wakeup
protocol unchanged. The updated attention harness preserves CLI defaults while
adding calls/sample, RAW averages, explicit 64-byte alignment and TFLOPS.

Final three-process actual-engine medians (before fixed scheduling → after):

- Main AVX-512 GEMM: 4.059 → 4.249 TFLOPS; 1024-token case 4.003 → 4.131.
- 64-token 8192 → 1024 GEMM: 3.354 → 3.708; four-token 4096²: 0.297 → 1.388.
- Cached 4 MiB GEMV: 0.234 → 0.847; cached 1 MiB: 0.0712 → 0.3630.
- AVX2 cached 4 MiB: 0.222 → 0.767; AVX2 main GEMM: 2.189 → 2.234.
- Streaming GEMV: 0.0380 → 0.0414; AVX2: 0.0385 → 0.0428.
- SMT-32 main: 3.754 → 3.724; cached 4 MiB: 0.1527 → 0.1500.
- Aligned, repeated attention: decode 0.328 → 0.355, prefill 0.146 → 0.158.
  Required-one-worker medians differ but process ranges overlap substantially.

Process variability remains material, especially the odd/misaligned output case.
No universal speed or physical saturation claim is made. All final commands,
raw outputs, source snapshots, reviews and checks are archived under
`docs/benchmarks/linear-fixed-workers-2026-11-01/`; RU/EN performance docs lead
with the actual current dispatcher and these results.

The final raw-clock audit reconciles every CPU tick and finds no samples outside
the gate. Cached GEMV is now 37.18% kernel and 56.47% pool CPU, versus 14.38/84.63%
before; CPU-time fractions are not latency fractions. Process-wide allocator
audits on the current build find zero Rust allocations/deallocations in all ten
warmed fixed-backend sequences (3/16 workers, linear/attention). Existing Rayon
fallbacks retain periodic bookkeeping allocations, recorded rather than hidden.

The full goal remains active. Next useful investigations, based on evidence:

1. Inspect the fixed callback assembly/CPU profile for avoidable context
   indirection. Hoisting worker count and capturing immutable job fields by
   value are bounded candidates; compare actual binaries before retaining them.
2. Trial bounded compact output scratch for AVX-512 4×64. Prior diagnostic work
   showed 3–6% gains, and odd output strides still have severe process outliers.
   An output tile can avoid repeated cross-worker shared-line writes while
   preserving K order. Keep storage bounded, document its full stack cost,
   test every initialized prefix/tail and preserve canonical weights. Start
   with rows≤256; avoid repacking large-N weights blindly.
3. Re-evaluate the correctness-checked larger-K R8×32 deep-GEMM candidate under
   the new scheduler. It remains unintegrated, so maximum performance is unproven.
4. Address configured SMT pools and coordination cost without a second pool.
   Active-worker subsets need a proven ownership protocol; the current global
   epoch cannot simply stop joining inactive readers of its stack context.
5. Refresh matched compute/memory calibration before any physical-efficiency
   claim. The 5.8368 TFLOPS nominal roof assumes all 16 cores simultaneously at 5.7 GHz;
   read-only calibration is observed throughput rather than a memory roof.

### Instruction profiling follow-up

The user requested profiling. Current sources, the saved 8a856e30 executable and
the final four profile manifests were reconciled, then their actual disassembly
was partitioned. No production code changed. Main GEMM has 86.83% of aggregate
CPU samples in its K loop, 8.35% in packing, 1.17% reloading accumulators and
1.62% storing them. The deep 64-token GEMM shares are 73.84%, 12.58%, 2.67% and
0.56%. Streaming GEMV spends 98.68% in its K loop and only 0.12% in reduction.
The main/deep loops have 28/37 static instructions per K step; this does not
itself establish instruction costs or a throughput ceiling.

Three longer independent cached-GEMV profiles (5.70–5.82 seconds per gate,
46,622–48,444 clock packets) reproduce 55–57% pool CPU and 36–38% arithmetic.
Worker epoch polling alone has 53–55% of process CPU samples, including 46–48%
on PAUSE. Callback dispatch in the worker wrapper is only 0.31–0.57%, completion
publication/waiter testing 1.31–1.40%. This lowers the priority of candidate 1
above; coordination/task granularity deserves investigation before capture
hoisting. A large polling share is not a latency speedup estimate: sleeping can
increase dispatch latency. All raw ticks reconcile; no samples are outside the
gates. Hardware cycles/instructions/cache-misses were probed and remain denied
with perf_event_paranoid=4. No host policy was changed.

Reports, audited repeats, exact-binary address maps, input hashes and a replayable
partition script are archived under
`docs/benchmarks/linear-fixed-workers-2026-11-01/instruction-analysis/`.
The optimization objective remains active; this phase supplies evidence for
the next controlled kernel/scheduling experiments.

### Compact-output GEMM implemented and measured

The prior instruction profile motivated a bounded output-tile experiment.
Four actual-root builds were compared against the saved fixed-pool executable
(8a856e30). V1 copied compact sums to output after all K panels; applying it
universally lost 2.45% on aligned short K. V2 stored the final K panel directly
to output, but lost 16.04% on an odd output shape. Its zero-row test also formed
an invalid pointer. Review caught this; the retained implementation returns
before any zero-row pointer arithmetic. V3 restricted selection to misaligned
output but still lost on short K. Disassembly exposed an extra DIV and call in
its target-feature routing wrapper.

Retained V4 removes that wrapper and selects in the existing run_avx512
dispatcher: preserve the original wide gate, require 128–256 local rows, K at
least 1024 (eight 128-depth panels), and an output base or stride not aligned to
64 bytes. Original wide::run and outer remain intact. Compact sums use 64 KiB
beside 32 KiB packed weights, then copy only complete owned 64-column intervals
once. No heap scratch, persistent weights, model/processor-name switches,
second pool, K reordering or bias changes. Public Rust/RU/EN documentation
states the full 96 KiB scratch bound.

Actual V4 executable SHA256:
3a3bc6efeb8b1ab9014efa74bfe76afbeb33ed7c4c43950c18fe7b3512d46b38.
Three-process measurements, 41 samples × 8 calls, shuffled serial order:

- 16 workers, 256 tokens, 4096² GEMM, output +12 B:
  4.064145 → 4.292232 TFLOPS (+5.61%).
- Same 15-worker case: 3.897277 → 4.031769; 8 workers: 2.357642 → 2.403001.
- 128 tokens, 2048², output +4 B: 4.042520 → 4.141831.
- 129 tokens, 1025 → 1057, output +4 B: 2.628146 → 3.064482 (+16.60%).
- 129 tokens, 1024 → 2048, output +4 B: 4.070137 → 4.068683 (near-neutral).

There is no universal no-regression claim. The unchanged AVX2 GEMM control is
1.05% lower in the broad series and 0.75% lower in a separate five-process
repeat (2.224359 → 2.207738 TFLOPS). Unchanged cached-GEMV paths measure
0.864061 → 0.850432 for AVX-512/4 MiB, 0.379226 → 0.362512 for AVX-512/1 MiB,
and 0.773280 → 0.783482 for AVX2/4 MiB. Short cases retain bimodal process
outliers; do not count their apparent multi-fold gains. Binary layout, clocks
and scheduling are hypotheses for unchanged-path effects, not demonstrated
causes. These controls remain performance work.

The compact oracle was watched RED against a stub and GREEN against the kernel.
It covers full f64 output, bias/no bias, K/row/column tails, row gaps, guards and
unaligned data. A second oracle reaches actual routing at 128/256/257 rows,
K=1025, odd stride and the original column-tail kernel. Final checks: 188
workspace tests, 87 release CPU/executor tests, Clippy with warnings denied,
fmt and Rust 1.89 all targets. Independent read-only actual-source review has
no remaining Critical/Important finding.

Current-source profiles were collected and raw-clock audited again: the
selected compact GEMM has 98.19% kernel CPU; 4 MiB cached GEMV has 39.44% kernel
and 53.53% pool CPU; 1 MiB cached GEMV has 18.04% kernel and 72.83% pool CPU.
No samples lie outside the timed gates. Hardware event access remains
unavailable.

All comparisons, final/rejected sources, verification and profiles are in
docs/benchmarks/linear-compact-output-2026-11-01/. The full objective stays
active. Next prioritize per-worker command publication, coordination and SMT
utilization. Inactive worker subsets must never read a stack context that can
be overwritten before those readers join. Recheck the unchanged-path losses
and matched physical calibration rather than treating this GEMM gain as
overall completion.

### Per-worker command publication experiment

The previous turn made production progress: bounded compact sums improved
selected GEMM workloads, with unchanged-path control losses recorded honestly.
The current exact-build profile still locates substantial cached-GEMV CPU time
in worker polling/coordination. The next experiment changes dispatch only;
the whole GEMM/GEMV objective remains active.

Replace the shared epoch and unconditional worker unparks with a command word
per worker. Each cache-separated slot owns its command, completion and panic
payload. Command and completion words each encode an even generation plus a
low waiter bit. The caller writes its borrowed job under the existing serial
mutex, advances each selected worker's own command by two, and swaps it with
AcqRel ordering. It unparks that worker only if the previous command had a
waiter bit. A worker exhausts a bounded spin, fetch_or(1, AcqRel), and parks
only if the returned generation is still its last completed command.

The one-word RMW modification order closes the sleep/publication race: either
the caller observes the waiter bit and posts a permit, or the worker acquires
the new command and skips parking. The caller joins each selected completion
before changing or ending its stack context. Inactive workers get no command
and cannot read a newer context; prior readers were joined in the previous
submission. Per-slot generations avoid an inactive worker's old generation
matching a global counter after wraparound. Consecutive selected generations
differ even across wraparound. The existing packed completion-waiter handshake,
panic ownership, same-pool inline nesting, serialized submitters, startup
binding checks and partial-startup/drop joins remain required.

Add a private run_active(prefix_count, job, spins) operation. Full dispatch uses
all owned workers. Common for_each uses min(job_count, pool_size), and install
uses one worker; there are no empty-job callbacks on unused workers. This phase
keeps topology selection, the Rayon/SMT fallback, pool size, configured affinities
and all arithmetic unchanged. It does not claim an SMT optimization yet.

Acceptance: first observe a bounded alternating-prefix test fail against the
old full dispatch; cover parked wakeups, zero active workers, borrowed contexts,
completion parking, nesting, concurrent submitters and panic/reuse. Independently
review the memory protocol, then compare actual binaries with cached GEMV,
short/large GEMM, attention and existing SMT fallbacks. Keep the change only on
measured merit; run release/workspace/Clippy/fmt/Rust1.89 checks and archive the
actual source/binary evidence. Successful jobs must retain no hot heap allocation.

### Worker-publication candidates rejected

All four dispatch candidates were slower on the key 16-worker cached GEMV.
Per-worker RMW commands measured 0.857566 → 0.730658 TFLOPS; batched stores with
fenced sleeper flags 0.852189 → 0.738029; split command/completion lines
0.859970 → 0.816291. Restoring global epoch publication with fenced conditional
wakes still measured 0.863504 → 0.760281. Independent reviews accepted their
memory-ordering/lifetime proofs, but correctness did not establish performance.

A same-executable empty-dispatch probe measured roughly 0.70 µs originally
versus 1.16 µs for per-worker batched publication on 16 workers. Normalized
AVX-512/AVX2 arithmetic instruction sequences matched between original/global
fenced builds, while addresses changed. CPU-15-pinned local-stripe controls,
without a pool, were about 5% faster for AVX-512 and near-neutral for AVX2 in the
fenced build. That single-core control did not reproduce parallel losses.

Every production CPU source was restored byte-for-byte to the prior compact
GEMM integration (3a3bc6ef). No active-prefix API or sleeper protocol from this
series is retained. Full sources, comparisons, bounded correctness tests and
controls are saved in docs/benchmarks/linear-worker-publication-2026-11-01/.
The standard Rust/C++20 SC fence rules support the reviewed fenced protocols;
the announced-sleeper stress test does not prove the initial stale-false race.

The full objective remains active. Next experiment targets GEMV arithmetic:
eight output features sharing input vectors, with an accumulator shape that
fits the ISA's registers (AVX-512 C8/U2 and AVX2 C8/U1), preserving C4 tails and
testing a longer independent reduction for scalar output tails. Extend the
existing public-API f64/guard sweep around output/vector/unroll boundaries,
then compare the actual engine and pinned local-stripe controls against saved
3a3bc6ef.

### Current candidate profiling, 2026-11-01

The C8/U2 and C8/U1 variants did not beat the retained C4 shape and were
restored. The current aa4bf94d candidate separates single-row GEMV from the
prefill frame and reuses it for a final odd batch row. This also eliminates an
outlined C4 helper. Earlier release-oracle checks passed, but broad acceptance
and the complete final verification gate remain pending. The planned scalar
tail unroll experiment has not been applied.

At the user's request, further arithmetic changes were deferred for fresh
exact-build profiling. Ten gated 2 ms profiles and nine instruction reports
were collected. All raw ticks reconcile with exclusive CPU totals, no packets
are outside the timed gates, and active-worker cadence matches the requested
period. Hardware counters remain denied at perf_event_paranoid=4.

Large 16-worker GEMM spends 88.25% of aggregate CPU samples in its K loop,
7.06% packing, 1.10% reloading sums and 1.38% storing them. Streaming GEMV
spends 98.71% in its K loop. Cached 4 MiB GEMV has 56.63% pool CPU at 16
workers, including 53.58% in epoch polling (46.44% on PAUSE). Callback dispatch
is 0.32%; completion publication/waiter testing is 2.31%. For 1 MiB weights,
pool CPU is 73.53%. These shares are not latency speedup estimates.

The 32-worker cached profiles locate 89.99% / 97.34% in Rayon/pool code for
4 / 1 MiB weights. A separate three-process unprofiled sweep gives
0.739850 / 0.948292 / 0.152688 TFLOPS at 8 / 16 / 32 workers with 4 MiB
weights. Reducing to eight workers therefore does not improve this candidate.
Prior conditional-publication losses still apply; more polling samples do not
justify repeating the same failed sleeper designs.

GEMM selects wide::run at 16 workers and x86::avx512::run at 32, through the
existing separate_cores/allow_wide policy. A future controlled SMT experiment
must hold the kernel constant. Priorities are now an isolated 32-worker
scheduling experiment for cached GEMV and K-loop/packing work for GEMM,
followed by the pending broad correctness/performance acceptance. No
production source was changed during this profiling phase. Data and replay
scripts are in docs/benchmarks/linear-current-profile-2026-11-01/.

### Bound SMT dispatcher experiment

Previous turn: progress. Ten exact-build profiles and a 36-process thread
sweep identify Rayon scheduling as substantial work for cached 32-worker GEMV.
The next controlled candidate changes only backend selection for a checked
bound CPU list with more than one worker. It uses the existing reviewed
FixedPool even when physical cores are shared, keeping separate_cores false
so arithmetic/kernel policy remains identical. Unknown/unbound placement and
required one-worker behavior stay on their existing paths. No second pool,
new model policy, persistent weights, or reduced configured thread count.

This is an authorized performance experiment in the existing feature checkout;
the dirty foundation sources must not be moved or discarded. Existing f64,
guard, affinity, completion, panic and waiter tests cover the dispatch contract.
The failing performance case is the measured 32-worker cached GEMV, not a test
of a private enum variant. Compare exact saved executables with serialized,
shuffled repeated processes; use bounded process timeouts because the caller
adds a runnable task when all logical CPUs are occupied. Retain only after
GEMM, small batches, streaming GEMV, attention, unbound and oracle controls.
Snapshots and raw evidence: /tmp/lumen-smt-dispatch-2026-11-01/.

The first bound-SMT FixedPool candidate with unchanged 4096 caller spins is
rejected for full occupancy: 32-worker cached 4 MiB GEMV measured
0.151831 → 0.020679 TFLOPS. At 24 / 17 workers it improved
0.186103 → 0.663203 / 0.213644 → 0.732654. Thus the dispatcher change and
the fully occupied caller wait must be evaluated separately.

The second candidate computes construction-time caller headroom and uses zero
caller spins when workers occupy available logical slots. A three-process
cached series measured 0.150888 → 0.183805 TFLOPS on 32 workers, with a wide
0.150–0.205 after-process range; 24 / 17 improved to 0.690641 / 0.735677.
No physical limit or stable 32-worker speedup follows from a pilot alone.

Independent review found no Critical/Important synchronization issue. Root
confirmed a portability limitation in installed Rust 1.97.1: its fixed-size
affinity query can fall back to online CPUs. The refined candidate combines
that quota-aware estimate with the existing growable affinity mask. The
crate-private current_affinity wrapper is now usable outside cfg(test), with
no new syscall implementation. Existing exact-once and panic/join/reuse tests
now run with both zero and 4096 caller spins. Release CPU/executor tests pass.

The mask-refined zero-spin candidate is 5f071483f467540e2a1844c1482625f53320ec37b9f18abb22a5475ee630a0c3.
Its 162-process broad series compares retained 3a3bc6ef, C4 aa4bf94d and this
candidate over 18 cases. Several 32-worker small batches improve 2–3.7x;
streaming improves about 9%, and 32-worker large GEMM about 3%. Two concerns
remain: aligned 16-worker large GEMM is 4.291 → 4.155 TFLOPS and narrow
32-worker GEMV (M17/K16385) is 0.0530 → 0.0427. All 716 normalized wide::run
instructions are identical before/after, at different addresses; this does not
identify the cause of the 16-worker regression. Attention controls improve on
17/24/32 workers; the unchanged 16-worker path is about 1.7% lower.

Current experiment (not yet accepted): retain a short maximum of 256 caller
spins at full occupancy rather than forcing zero. This tests whether narrow
jobs can avoid an unnecessary park/unpark round trip without returning to the
4096-spin saturation loss. Current source snapshot short-src/, executable
short-bench_linear SHA e47fc8d00a69ff17911b5512d21802cebb63b425406ce9668803ebf750638ff8;
87 release CPU/executor tests passed. short-cached.json compares before,
zero-spin and short-spin executables with longer narrow/cached controls.
Broad acceptance, physical-only restricted-mask controls, final workspace /
Clippy / fmt / MSRV and documentation are still pending. No candidate from
this SMT phase is declared retained yet.

### Partial worker groups: proposed publication design

The short-spin experiment failed: on cached 32-worker 4 MiB GEMV it measured
0.109150 TFLOPS versus 0.151745 before and 0.167150 for zero caller spins.
On M17/K16385 it was 0.018443 versus 0.046516 before and 0.028379 at zero.
It is rejected; zero-spin code has been restored.

A narrow-column grain experiment uses C4-sized feature tiles when 16-column
tiles cannot occupy all workers. It improves physical-16 M17/K16385 from
0.08312 to 0.30826 TFLOPS and M33/K8193 from 0.19402 to 0.39457. However,
M129/K2049 drops about 7%, and full-32 M17 remains slower than Rayon. A future
grain rule should change only clearly underoccupied plans, not replace a
near-full cache-line plan. The source is currently this diagnostic grain
candidate (b4526b5e), not a retained final policy.

Fundamental remaining cost: FixedPool joins every global-epoch reader even
when only a few jobs exist. Simply skipping inactive joins would let a late
reader touch an expired stack context. A globally packed active-count/epoch
also has wrap/late-reader problems unless extra reset machinery is added.
Do not implement either shortcut.

Proposed bounded design, within the existing single owned FixedPool:

- Keep the current global epoch for full worker groups. Only these submissions
  increment it; every worker participates and is joined, preserving its fast
  single-publication path.
- For a partial group, publish a separate monotonically wrapping command word
  to each selected worker, then unpark only those workers. Increment each word
  relative to that worker's own last command, not a global epoch. Inactive
  workers receive no command, never read the borrowed context and are not joined.
- A worker polls the global epoch first, then its own command. Both acquire
  loads precede copying Job. These channels cannot have concurrent uncompleted
  jobs: serial publication waits for every participant of the prior job.
- Use disjoint completion tags: bit 0 remains the caller-waiter flag; bit 1
  distinguishes partial commands from global commands. Shift the respective
  generation by two. Consecutive uses of each channel on a worker always join
  that worker, so generation truncation/wrap cannot match a stale completion.
  Global epochs remain consecutive for every worker; private commands remain
  consecutive for their selected worker, even across arbitrary inactive spans.
- Store the private command in a separately cache-line-aligned field within
  the currently 128-byte-padded Completion. Keep its state/panic in the first
  line and command in the second; avoid another allocation or growing per-worker
  storage if layout permits. Full groups do not write private command lines.
- Add run_active(active, job, spin_budget); existing run/run_with_spins use
  all workers. Pool::for_each selects min(jobs, pool size), and install selects
  one. The callback mapping for jobs >= pool size is unchanged; for fewer jobs,
  exactly the workers owning a job execute it.
- Compute caller headroom against the active participant count and the checked
  startup capacity. Full occupancy parks promptly; a partial group may use the
  original bounded active wait. Idle-worker parking and unconditional selected
  permits retain the existing no-lost-wakeup rules.
- Preserve same-pool inline nesting, panic payload ownership, concurrent caller
  serialization, error routing, partial construction cleanup and shutdown.
  Shutdown must wake every worker through the global channel before joining.

Before implementation: independent adversarial review of this publication and
wrap proof. Verification must exercise full→partial→full transitions, changing
partial prefixes, borrowed context expiration, parked inactive workers, zero
caller spins, panic/reuse and concurrent callers, alongside all existing
numeric guards. Compare full-16 cached GEMV to ensure the extra idle command
load does not erase the original global-epoch advantage. Then repeat the broad
and attention controls and final toolchain gates. This design is proposed,
not yet implemented or performance-accepted.

The independent design review accepted the two-channel proof subject to
explicit initial seen counters, joining every participant before next
publication, no inactive Job reads, mode-tagged completions, rechecking both
channels after park, and owned late notifications. The implementation now
exists in threading/fixed.rs and pool.rs. An empty run_active stub produced
the expected exact-count RED; the implementation passed the prefix test and
extended fixed-pool tests. Coherently seeded test pools exercise completion-tag
and command-word wrap; a barrier delays startup until a command is already
published. Panic/reuse, concurrent prefixes, nested partial calls, zero active
participants and inactive-worker shutdown are covered.

Independent actual-source review found no Critical/Important defect. Its
minor constrained-mask coverage issue is addressed by cycling allowed CPU IDs
to keep three test participants even on a one-CPU mask. This is test setup,
not a change to production placement. No implementation re-review is inferred
for future changes. Current verification/build logs and upcoming paired
measurements are in /tmp/lumen-dual-publication-2026-11-01/; the before source
and executable are the b4526b5e grain candidate. Full performance acceptance
and workspace/toolchain gates remain pending.

The first dual-channel candidate passes 93 release CPU/executor tests and all
15 fixed-pool tests under a one-CPU mask. Actual-source review found no
Critical/Important issue. v3 executable SHA is
f3591674c543a5ecc1c81629bb38df68343ec834976f0084610ad212d05fb1bc.
Initial paired timings improve partial narrow groups substantially, but full
groups regress: 16-worker cached 4 MiB about 0.958 → 0.732 TFLOPS. The extra
private-command polling load must not enter the steady full-group loop.
cached-v3.json is still the authoritative repeated comparison for this build.

Proposed refinement before another implementation: use the global control
word's low bit as a private-mode hint, and the remaining bits as the full-group
generation. Full publication clears the hint and advances its counter by two;
partial publication sets the hint once before publishing selected private
commands. A full-mode worker polls only the global word, as in the original
fast loop. On observing the hint it switches to private-mode polling (global
control plus its own command), without reading Job. Only changed selected
private commands authorize a borrowed Job read. A changed full control word
authorizes a global job and returns to the full-only loop. Completion remains
mode-tagged: global target is control<<1, private target is private_generation<<2|2.
The test initializer must seed even full control words coherently with worker
seen values and matching prior completion targets. Both counter boundaries
must be exercised again. Shutdown checks stop before any hint transition or Job
copy and wakes all workers. Owner serial/join invariants do not change.
This is a mode hint, not an unjoined globally mutable active-count descriptor;
partial context access still requires the independently acquired per-worker
command. Independent review of the refinement is required before coding it.

The hint refinement passed independent design and actual-source reviews and
95 release tests. It recovered some full-group speed, but three processes still
measured 0.953771 → 0.894985 TFLOPS for cached 16-worker AVX-512. Forced generic
inlining was worse and is rejected. Const channel specialization and direct
one-job-per-worker calls remove an observed StepBy integer division and closure
layer; preserving the old Completion hot-field order improves a short control.
The current layout-clean aaf5c8a4 candidate passes 96 release tests and Clippy.
cached-layout.json compares it with aa4bf94d and direct60549430. Nothing from
this publication phase is yet declared accepted.

### Simpler candidate: per-group generations in one command word

A possible simplification avoids the second publication channel entirely.
The earlier packed active-count proposal was unsafe with one global generation
advancing through excluded groups. Instead give each immutable prefix group
(active count N, members 0..N) its OWN generation, updated only for submissions
to that group. Store these N counters in the already held submission mutex's
payload; they are caller-only data, not worker-polled atomics.

For active > 0, increment generations[active-1] under serial. Publish one
AtomicUsize descriptor `(generation << group_bits) | (active-1)`, where
group_bits = ceil(log2(pool size)). Unpark only selected workers and join only
their completions, using descriptor<<1 as target and bit0 as waiter flag.
Every worker has the original one-word polling loop. On a changed descriptor,
it decodes immutable prefix membership. Nonmembers update seen without copying
Job or writing completion; members acquire Job and complete as before.

Wrap argument: different groups have disjoint low ID bits. Within a given group,
every member participates in every use and is joined before its generation can
advance again. If a newly selected worker's last seen/completed group is the
same group, it is the immediately preceding generation of that group; if it is
another group, the ID bits differ. Inactive spans advance only other groups'
counters. This avoids both descriptor equality and stale-completion ABA across
truncation/wrap, provided at least one generation bit remains after reserving
group and waiter bits (valid CPU slice/allocation limits ensure more than one).
Workers must still initialize seen from coherent constructor seeds, not late
atomic snapshots, and never read Job unless selected by the acquired descriptor.

Shutdown CANNOT use arbitrary descriptor.fetch_add(1): a lagging inactive
worker could already have that other group's word as its seen value. Set stop,
advance/publish a fresh full-group descriptor using its own generation, unpark
all created workers and join them. The full group's same consecutive-member
proof makes the stop publication unseen by every worker, including at wrap.

A deterministic ABA test can seed a completed full-group generation at 1 and
the one-member group's independent generation at MAX-1, with inactive workers
held before their loops. Two one-member calls wrap its generation; the next
full call must advance the FULL counter to 2 and await the delayed members.
Using the latest global descriptor's generation incorrectly would recreate
full generation 1 and could mistake old completions for new ones.

This is a proposed design, not current code. Obtain adversarial review first.
It should remove private command lines, mode hints, separate polling loops and
tag-mode branching, while preserving a single owned pool and scoped borrowing.
The numerical/grain policy is separate and still requires broad acceptance.

### Fresh profile requested by the user: current layout-clean source

Before replacing the publication protocol, froze/rebuilt current aaf5c8a4
and collected nine gated 2 ms clock profiles. No production source changes.
All nine raw-packet audits pass; no sample lies outside the measurement gate.
An independent 27-process unprofiled control verifies numerics and TFLOPS.
Archive: docs/benchmarks/linear-layout-profile-2026-11-01/README.md.

Current large GEMM16: 96.87% arithmetic, 85.64% K loop, 8.21% packing.
Streaming GEMV16: 98.75% arithmetic, 98.62% K loop.
Cached 4 MiB GEMV16/32: 58.17%/93.05% pool; full epoch polling alone
50.52%/92.42%, including 43.95%/88.98% PAUSE samples. Callback authorization
and dispatch are only 0.32%/0.02%. Cached 1 MiB GEMV16/32: 78.26%/97.13% pool.
This strengthens the case for studying the selected physical-core execution
group and wait policy, with arithmetic held fixed. A publication simplification
alone is not evidence that the full 32-worker path will improve.

Current unprofiled GEMM16 is 4.171 TFLOPS (process medians 4.159–4.183),
GEMM32 3.369 (3.079–3.715); they still use different kernels, so this comparison
does not isolate SMT or CCD effects. Cached 4 MiB GEMV16/32 is 0.843/0.185 TFLOPS,
1 MiB 0.401/0.04372. X/W use allocator alignment, unlike older fully aligned
controls; do not treat these as before/after gains or losses. Hardware perf
events remain denied. No new implementation or performance acceptance is
implied by this diagnostic step. The single-command draft remains outside
production source and must still pass its own correctness and performance gates.

### Physical-core execution group inside the configured pool

The user already authorized autonomous CPU experiments and preserving the
configured owned pool size. Implement inline in the existing dirty feature
checkout; snapshot all CPU sources first. The current publication protocol is
already safe for a selected prefix, so do not replace it for this experiment.

Hypothesis: cached GEMV32 wastes substantial CPU time with all SMT workers
polling; selecting one logical CPU per physical core can improve throughput
without changing the arithmetic kernel or creating another pool.

Files and contracts:
- topology.rs: retain the existing CPU selection set and NUMA/LLC ordering
  within each SMT rank, but put the first logical CPU of every selected core
  before second siblings. Restricted masks may expose only secondary siblings.
- threading.rs: record the selected physical-core count only after binding
  succeeds; unknown/unbound placement has no physical execution-group hint.
- threading/pool.rs: add a joined for_each_on(workers, jobs, wait, callback)
  operation. FixedPool uses exactly the selected prefix and covers every index
  once, including jobs exceeding that prefix size; Rayon limits job lanes
  without promising physical affinity. Preserve the existing full-pool route.
- executor.rs / linear.rs: single-row linear may use the verified physical
  count; prefill uses the configured count. Partition stripes using the actual
  execution-group width and submit through for_each_on. Arithmetic functions
  and the existing wide-kernel eligibility flag remain unchanged.

Checks:
- [ ] RED: a synthetic uneven SMT topology proves the first four workers occupy
  four different physical cores; selecting more workers retains the same CPU set.
- [ ] GREEN: reorder workers and update existing exact placement fixtures.
- [ ] RED/GREEN: restricted execution groups visit every job exactly once, with
  real FixedPool worker indices bounded by the group; test 0, 1 and many jobs,
  nesting and panic/reuse. Numerical guards cover both pool backends and group
  widths around uneven stripes; configuration still reports the full pool size.
- [ ] Compare baseline, ordering-only and physical-GEMV candidates in serialized
  shuffled runs on 16/17/24/32 workers for AVX2/AVX-512, cached/streaming/narrow
  GEMV, then GEMM and attention. Save exact sources and executable hashes.
- [ ] Independent review, workspace/release tests, Clippy/fmt and MSRV before
  accepting production changes; archive failures and restore any rejected policy.

Success for this experiment means a verified speed improvement for the target
cases without material unexplained regressions. It does not complete the
overall maximum-performance goal or prove a physical roof has been reached.

Ruling after 348 process runs (144 cached, 132 large, 72 attention): reject
global SMT-rank worker reordering. Although physical GEMV32 improves 4 MiB
0.163114→0.664347 TFLOPS and 1 MiB 0.042911→0.246956, attention32 falls
0.378398→0.338946 and odd prefill 0.144439→0.120183 (ordering-only 0.103349).
GEMV17 and some 16-worker cases also regress. Exact data and three source
snapshots remain in /tmp/lumen-physical-group-2026-11-01. The initial candidate
passes 99 release tests; review found no Critical/Important but three coverage
minors. Performance rejection supersedes that correctness-only approval.

Revised design, independently reviewed: preserve the original native worker
order and register immutable ordered groups in FixedPool::Shared before spawn.
Group 0 is exactly 0..pool.len(); optional groups contain validated unique native
indices and an inverse rank map with a sentinel for excluded workers. Startup
derives the physical group from OS topology, not IDs or CPU brand.
run_group(group_id, active, callback(rank)) publishes/unparks/joins exactly
members[..active]. Job.group_id and inverse are read only after the existing
command authorizes that native worker. Full global publication is used only
for native group0 with all workers; other groups retain the current private
native-worker generations. Nested calls run logical ranks inline. Zero/no-op,
panic draining, late wake and borrowed-context joins retain their contracts.

Pool::for_each_on takes a Group { id, workers } descriptor; unknown/unbound
Rayon placement supports native group0 only. GEMV uses the verified physical
descriptor, GEMM and attention keep their existing native group. Tests must
cover nonconsecutive/permuted groups [2,0], overlapping groups [0,2]/[1,2],
same active count across groups, constructor errors before spawn, nested calls
from an excluded native worker, and panic/reuse/wrap. The existing private
generation proof still holds because it belongs to each native worker.

If subsequently testing the single-command protocol, every (group_id, active)
pair needs its own immutable identity and independent generation. Shutdown must
advance the full native pair. One counter per group is insufficient.

Registered-group implementation is now in the working tree. Candidate SHA256
4179b204ac7587bc890658b579d032494bfc0a2b7efb7f5fc79ebf04df2fbcf6.
207 workspace tests, 106 release CPU/executor tests, Clippy/fmt and Rust1.89
pass. Actual-source review has no Critical/Important; its remaining minors were
closed by scoped install_on, narrow regular/streaming GEMV guards, nested panic
and a dispatcher test that observes both Job.group and published participants.
An always-native dispatcher mutation fails that test. Test-only observations
hold serial and read no expired context pointer.

261 serialized process comparisons (108 cached,99 large,54 attention) confirm
4 MiB AVX512 GEMV32 0.165430→0.605985 TFLOPS and1 MiB0.041196→0.224510.
AVX2 cached32 0.183340→0.566938. Native ordering restores attention decode32
0.367278→0.369885; no global worker-order change remains. Cached16 still
regresses0.937205→0.875916,24 regresses0.654104→0.606944; narrow33 also loses.
Thus performance acceptance is OPEN, despite substantial32-worker gains.
Main GEMM16 is4.267253→4.234393 and32 is3.796291→3.792633; kernel policy
for prefill still uses native workers. Distinct physical execution groups could
enable the existing wide kernel on SMT-configured pools in a future experiment.

Two fresh gated profiles of4179b204 pass raw packet audits and sample16 active
workers in both configured16 and32 cases. Pool shares remain54.31% and74.79%
respectively; selected-group handoff remains a priority. Input alignment is the
allocator default for these profiles, whereas throughput comparisons explicitly
align X/W/Y; do not compare their kernel CPU per call as matched controls.

A native-identity specialization (b2cde317) passed106 release tests and review
but is rejected after72 cached+18 large+40 attention process runs. Full16 gains
little, while cached32 regresses0.623921→0.574543 and1 MiB0.240263→0.196482.
Restore4179 source bytes; do not stack more incidental native-path edits.
Longer five-process attention controls still show overlapping, broad ranges;
odd prefill median before0.135780 vs groups0.131182 remains an open limitation.

Next architectural experiment: simplify publication with one command word and
independent counters for every immutable (registered_group, active_prefix) pair.
Encode group and active in disjoint low tag bits; reserve at least one effective
generation bit after the waiter shift. Worker decodes membership/rank from
immutable maps before authorizing Job access; nonmembers only update seen and
park. Full native pair supplies a fresh shutdown generation. Reuse the prior
counterexample RED (never run a deliberately broken borrowed-context protocol
on real workers), and port delayed-startup, cross-group wrap, nested/panic/drop
tests. Compare directly with4179 and aa4b; arithmetic and native ordering stay
fixed. Separately, GEMM still needs physical-group/wide selection and K-loop
experiments before any claim of maximum performance.

The single-command experiment is implemented as c6e41ff7 (linear) /
8f8bb721 (attention), with unchanged arithmetic and native placement. Each
(group, active-prefix) owns its own generation; the codec keeps at least one
generation bit after the completion waiter shift. The pure per-group-authority
RED reproduces the wrap counterexample without exposing a broken borrowed Job
to real workers. Immutable membership precedes all Job access. The final source
passed 214 workspace and 113 release tests, Clippy/fmt, Rust 1.89 and 41 threading
tests restricted to CPU 0. Actual-source review has no Critical/Important; a
configured-five/created-two shutdown fixture closes partial-creation coverage.

348 shuffled serialized process controls (144 cached, 132 large, 72 attention)
show cached AVX-512 GEMV32 0.621762 → 0.954699 TFLOPS over 4179, 1 MiB
0.225780 → 0.389135; configured 17/24 reach 0.954963/0.953386. Cached16
0.918962 remains below earlier aa4b 0.960741. Big GEMM16 reaches 4.228374,
32 reaches 3.782119; short four-row and some narrow cases remain open. A subsequent
15-process attention32 control (five per candidate, 51×512 calls) gives native
0.372953, groups 0.380483, single 0.374464 with overlapping ranges. It does not
reproduce the initial large attention32 loss or establish universal equivalence.

Seven fresh c6 gated profiles use explicitly 64-byte-aligned X/W/Y and pass raw
packet/gate audits. Large GEMM16: 97.22% CPU in arithmetic functions, 87.31%
in its main K loop, 7.13% packing; GEMV DRAM16: 99.02% arithmetic, 98.89% K loop.
Cached 4 MiB 16/32: 36.18/38.99% arithmetic, 62.99/59.84% pool. Instruction-level
poll shares 55.19/51.45%, membership+callback 0.61/0.82%, completion 1.47/1.80%.
These are aggregate exclusive CPU shares, not elapsed-latency saving estimates.
Four separately audited whole-process attention profiles include startup/oracle;
never mix their collection boundaries or use profiled speed as a comparison.

Artifacts, sources, correctness logs, hash-matched clean rebuild, 11 profiles,
363 unprofiled process runs and known limitations are archived in
linear-command-publication-2026-11-01. A temporary constructor-layout diagnostic
was removed; clean rebuilt bench_linear and bench_attention match the frozen
c6/8f8 binaries exactly. Performance acceptance and the maximum goal stay OPEN.
Next work should target GEMM K-loop/packing or shape-based physical-group/wide
selection, and actual GEMV task geometry. Do not repeat previously rejected
sleeper flags or native-identity micro-specializations solely from PAUSE shares.
Hardware events remain inaccessible at perf_event_paranoid=4; clock profiles
cannot measure IPC, cache misses, DRAM saturation or a physical TFLOPS roof.


### Physical-group wide prefill experiment, 2026-11-01

Previous goal turn: progress. Seven new exact-build profiles and raw audits
establish that large GEMM spends 87.31% CPU in its K loop; the remaining
16-versus-SMT policy selects distinct arithmetic kernels. CPU workloads are
terminal before this experiment; c6e41ff7/8f8bb721 are frozen as the baseline.

Design: reuse the registered physical Group for AVX-512 prefill when global
rows meet the wide kernel's shared minimum (128) and complete 64-column tiles
number at least the physical workers. This guarantees feature parallelism
without splitting token rows below the wide minimum. A final short stripe may
still use the existing tail kernel. Expose these two geometry constants from
wide.rs and reuse them at its current dispatch guard. No arithmetic, packing,
worker ordering, pool size, affinity policy or additional weight copy changes.

Executor derives allow_wide from the actual selected verified physical group,
rather than the configured native pool's separate_cores flag alone. Short
batches, insufficient width, AVX2/scalar and absent/unbound topology retain the
native prefill route. Existing GEMV selection is retained exactly.

Checks: actual FixedPool dispatcher RED/GREEN at 127/128/129 rows, narrow and
odd output widths, physical widths1/2, absent hint and other ISAs; assert group,
joined participants, configured pool size, f64 result with bias and output guards.
Then release/workspace, clippy/fmt/MSRV and exact-source review. Compare frozen
before/after in serialized shuffled processes: 16/17/24/32 GEMM, depth and row
thresholds, narrow/odd/misaligned output, AVX2, cached/streaming GEMV and attention.
Retain only supported gains without unexplained material regressions. This
experiment alone cannot complete the whole maximum-performance objective.


Physical-group wide prefill is implemented as e674478b / attention29732ed7.
The actual dispatcher test was observed RED against c6e41ff7 (group0 versus1),
then GREEN. Independent source review has no Critical/Important production
finding. Its coverage Minor was regraded as an essential activation gate:
Linux-test-only TLS observes actual wide run/compact entry, and removing only
physical wide permission fails[0,0,0] versus[1,0,1]. Final source passes215
workspace,114 release tests, Clippy/fmt, Rust1.89 and all3 fixed-dispatch tests
restricted toCPU0. Non-Linux builds omit these observations; non-x86 omits the
record function. No release instrumentation is present.

246 main process runs plus80 longer controls confirm main GEMM16
4.235254->4.259886,17 2.342424->4.248127,24 3.029248->4.245786,32
3.771713->4.268822 TFLOPS. Odd129/1057/1025 output+4B improves
1.766729->3.033882;1024-row prefill32 3.544646->4.098891. Small-depth
K1/17/127/128/257/1024 all improve on32. Initial10-12% attention losses
and5-8% cached-control losses do not reproduce in five longer processes:
attention decode32 .380517->.377003,odd prefill .128040->.126489;
1MiB GEMV16 .405617->.406000,129/2049 GEMV32 .401666->.401667.
Four-row GEMM16 remains1.378912->1.349147 (~-2.16%), and AVX2 cached32
.850821->.842944. Keep these limits explicit; do not claim universal gains.

A fresh gated profile confirms16 active workers plus caller with32 configured,
97.50% CPU inside wide::run. Raw tick/gate audit passes. Exact new-PC partition:
Kloop86.55%,packing7.61%,sum reloads0.94%,stores2.09%. Arithmetic function
structure retains all716 relative instruction offsets from the prior build.
Eight selected attention/worker machine functions are byte-identical at the
same addresses; this does not prove all data placement or scheduling equivalent.
Artifacts (362 comparison processes including36 pilot, plus9 calibrations),
source snapshots, review and logs: linear-wide-group-2026-11-01. Release tests
rebuild examples without debug info; restore CARGO_PROFILE_RELEASE_DEBUG=1
build afterward. Restored targets match frozen e674/297 exactly.

Ruling: retain the verified physical/wide selection as the current candidate
with the small four-row control loss explicitly open. Large selected cases
improve13-81%; major apparent collateral losses fail longer replication.
Cost if wrong: some unchanged-path shapes may retain small losses and need
subsequent correction. Overall performance acceptance/completion is not proved.

New calibration evidence: old synthetic FMA uses unpinned Rayon inside a
0-15 taskset mask; process medians vary4.025-4.784 TFLOPS (best sample5.535),
AVX2 medians1.831-1.960, below measured GEMM2.20. Verified-bound read helper
reports about0.0339 GEMV-equivalent TFLOPS, below actual streaming GEMV0.0414.
These are observed helper rates, not physical upper bounds. Before efficiency
claims, verify equal work on each individually bound calibration worker;
Rayon broadcast is one candidate to inspect in the locked primary source.
Do not reuse these medians as roofs or repeat broad kernel sweeps to explain
a faulty roof assumption. Report new calibration output in TFLOPS.

Next arithmetic hypothesis, not yet implemented or benchmarked: five-row
AVX512 outer product may amortize weight loads/control while fitting32 ZMM
registers (20 accumulators plus weights/broadcasts/packing constants). No prior
five-/six-row wide experiment was found in the evidence log. Confirm actual
register pressure/spills, cover the four-row remainder explicitly (129 leaves4
under R5), keep K order, bias and compact/direct tails correct, and compare
actual CpuExecutor binaries. Prior rejected C8/unroll GEMV work is a different
experiment and must not be repeated blindly. Main Kloop/packing and cached
GEMV task geometry still require work; the original maximum goal remains active.


### Equal-work calibration and R5 experiment, 2026-11-01

Previous turn: progress (physical/wide selection,362 process runs and validated
current e674/297 source). Authoritative verifier passes and no benchmark/compiler
process is live before this next experiment. Keep the GEMM/GEMV production source
frozen while changing only calibration helpers first.

Calibration design: share the existing verified affinity setup between FMA and
read examples. Use Rayon broadcast, whose locked1.13 primary source guarantees
one call per worker and result order by worker index, instead of work-stealing
parallel iterators. Each worker runs the exact same FMA iteration count; reads
use a fixed disjoint chunk selected by actual worker index. Preflight validates
TLS worker index and actual CPU against verified placement before timing. Keep
unbound/inline fallback and required failure behavior. Count allocation and
broadcast costs in measured time. Add FMA's optional affinity flag and report
TFLOPS. No public inference API or arithmetic kernel change is needed.

RED: existing FMA CLI rejects explicit required affinity. Check new real CLI
receipts, per-worker work ownership, 1/3/16/32 workers, restricted/unbound masks,
checksums and independently recomputed FLOPs. Obtain a stable observed reference
using three serialized processes per configuration; it is still not a universal
physical upper bound. Then prototype R5 only after its odd-row/remainder guards
are in place; inspect assembly for spills before comparing production calls.

Calibration result: equal-work bound FMA16 is stable5.536-5.539 TFLOPS AVX512
(median5.537662) versus old3.704-4.714; AVX2 median2.663424 versus1.789522.
New32-worker reference:5.525313 /2.762238 TFLOPS. The read reference remains
~.034134 TFLOPS-equivalent, below production GEMV~.0414. Controlled same-binary
alignment experiment(default,64B,+16B) gives.034079/.034457/.034037; alignment
helps only about1%, not an explanation of the entire gap. Keep read helper as
an observed lower reference, not an upper bound; do not invent saturation or
cache-stall explanations without evidence. CLI full independent1MiB checksum
and offsets0/1/15 passed forScalar/AVX2/AVX512 on3 actual workers. Ownership
mutation(all callbacks serial on one pool thread) is correctly rejected.
Independent frozen-source review has no Critical/Important; two doc clarities
were included when documenting the added read-offset argument. Locked Rayon
implementation orders broadcast results; runtime preflight enforces that
assumption instead of relying solely on prose documentation.

Proceed to R5 arithmetic with the now-stable observed compute reference.
Keep the remaining read-baseline investigation explicit; it is not a blocker
for paired production-kernel comparisons or reason to postpone them indefinitely.


Calibration is retained and archived: cpu-calibration-workers-2026-11-01.
Three-process FMA references:16 AVX5125.537662,AVX22.663424;32 AVX5125.525313,
AVX22.762238 TFLOPS.27 initial plus9 read-alignment measurements;14 small CLI
validations. Ownership-serial mutation fails; real required/unbound/restricted
receipts and full checksum tests pass. Review no Critical/Important; minor
comment clarities corrected, complete FMA.text unchanged. Runtime preflight
checks worker-index order rather than silently assuming library compatibility.
Read reference~.0345 remains below real streaming GEMV~.0414; no bandwidth
upper-bound/saturation claim is supported by it.

R5 arithmetic candidate0b1eed9a passed115 release tests including real R5-route
counts, f64/guards/bias/direct+compact and all remainders. Removing the R4 tail
is caught at rows9; source was restored before timing. R5 has20 register FMA,
4 weight loads,5 broadcasts,4 address-generation instructions and4 loop-control
instructions:37 total, no accumulator spills in its innermost K loop. The
precomputed-pointer variant54f9eb07 preserves the same37-instruction loop;
its numerical/integration checks pass, but no redundant performance run follows.
This is a new R5-specific pointer chain; it does not justify repeating the
previously rejected R4 address-hoist experiment.

After16 pilot processes,42 repeated production comparisons reject R5. Main256
rows16:4.219829->4.194620,32:4.230496->4.210854; odd129/1057/1025 on32:
3.083242->2.761982;1024 rows16:4.127838->4.024970 TFLOPS. Compact+12B is
essentially unchanged. Restore exact before-src wide.rs bytes, removing all
R5 code and test instrumentation. Current matrix/attention binaries match
prior e674/297; final FMA/read match997aee63/af1ee857. Current source passes
215 workspace/114 release tests, Clippy/fmt and Rust1.89 after restoration.
Rejected sources, tests, assembler and measurements: linear-wide-r5-2026-11-01.

New hardware evidence: sysfs L1D48KiB,64 sets,12 ways,64B lines. Packed32KiB
weights use substantial L1 capacity; input rows with power-of-two K may alias
sets. Cache conflict is a HYPOTHESIS, not established cause of R5 loss. Next
inspect prior depth64/96/128 experiments before changing panel depth; then
use a controlled packing/depth or input-stride experiment with the same useful
math. Keep selection thresholds stable while isolating panel cost. Do not
retry source-level R5 pointer arrays (same machine loop) or force an unproven
row-count-specific policy. GEMM K-loop/packing, cached GEMV geometry, AVX2 and
memory-reference efficiency still prevent completion of the original goal.


### Controlled four-row panel depth, 2026-11-01

Previous goal turn: progress: stable equal-work calibration and tested rejection
of R5, exact e674/297 arithmetic restored. All four target hashes are verified;
no compiler/benchmark is live before this experiment. Prior diagnostic archives
already test4x64 depth64/128 on older Rayon/full-SMT layouts;96 was not found.
Retest is specifically for the newer physical group, FixedPool and compact sums,
with identical useful work. Avoid drawing claims from cross-era timings.

Design: parameterize only private AVX512 wide packing depth using a nested
Panel<const D> array (stable MSRV; exact D*64 floats), leaving the four-row
microkernel and canonical weights intact. Default D128 provides a normalized
refactor control before D96/64. Keep COMPACT_MIN_K1024 independent of D so the
experiment isolates panel depth, not selection. Shared default selection is
compile-time; no CPU/model name or public inference API change.

Tests observe actual pack requests, requiring depth<=D and exact K coverage for
each column tile, plus complete f64 results, bias and guards at D-1/D/D+1/2D+1
for64/96/128, direct and compact output. RED first delegates to the old128
packing regardless of requested D; its request violates the smaller frame cap
before any memory write. After GREEN, build frozen normalized128/96/64 binaries
sequentially and compare a bounded representative matrix set. Preserve default
128 if smaller panels fail. Only a supported winning policy and necessary tests
are retained; the whole maximum-performance goal remains open.


Depth experiment rejected after28 pilot+84 repeated processes. Default128
normalization is not a performance win by itself. Relative to original e674:
main16 before4.247855,D964.169171,D644.076433; compact before4.293241,
D964.216986,D644.137522. ShortK768 misaligned before3.387540,D963.141795,
D642.511003. D96 long1024 rows4.139583 versus4.117200 is too small and
not general enough to offset losses. Review found no Critical/Important in
nested layout, capacity, tails or policy; optional capture gating is deferred
because the entire generic-depth experiment is removed. Restore exact original
linear.rs and wide.rs. All core source files match the pre-experiment snapshot.

### GEMV geometry experiment, 2026-11-01

Now test the known coarse/fine-stripe gap. Existing single-row SIMD selects
four-column stripes whenever cacheline-sized stripes cannot use every worker,
including cases where most workers could already own a separate cache line.
New conservative rule: keep four-column stripes if coarse stripes use at most
half the group; otherwise use16-float/cacheline-sized stripes. This changes
only the intermediate-occupancy band; very narrow and fully occupied shapes
retain their routes. No ISA arithmetic or pool protocol change.

Share the cacheline-stripe count helper with streaming classification, retaining
its existing conservative8-cacheline-chunk requirement (do not silently enable
streaming for skinny long-K matrices). Test actual participants of a permuted
physical group[4,1,5,0] inside a6-worker pool. RED33/8193 currently wakes4
instead of expected3, while17/16385 must retain4 and64/4097 must retain4.
Assert group/prefix/count preservation plus bias, unaligned slices, guards and
full f64 output. Compare16/32 for both ISAs across intermediate M113/129/193/241
and K2049/4097/8193/16385 plus severely narrow/full-width controls. Gate by
repeatable actual API gains; no per-model or CPU-brand switch. Overall goal
remains active even if this individual policy succeeds.

Unrestricted coarse pilot(72 processes) falsifies the simple half-occupancy
rule:129/8193 loses~21-24%,129/16385~32-45%; AVX2 mostly loses too. AVX512
129/2049 gains~13-18%. Therefore do not add cache metadata to explain a loss
that already occurs with a sub-L2 footprint. The tradeoff is useful work versus
handoff/parallelism; a cache-capacity explanation is not established.

Revised bounded hypothesis: only AVX512, with at least half cacheline-sized
jobs and no more than MIN_PARALLEL_WORK/8 coefficients per configured group
worker; otherwise retain original fine/coarse choice. Share MIN_PARALLEL_WORK
with the existing executor cutoff. This scales the operation budget by actual
group width instead of hardcoding K or CPU/model names, retains long-K and
AVX2 routes, and covers common M128 as well as129. The budget is a measured
heuristic, not a hardware law. Test a16-member permuted group inside17 workers:
128/2048 uses8,129/2049 uses9,129/4097 retains16,33/8193 retains9 and
256/1024 retains16. AVX2 retains original participation. Observe RED against
the overbroad rule before GREEN; validate smaller3/8-worker and unbound cases
alongside16/32, then repeated paired controls before retaining anything.

### User-directed fresh profiling and GEMV disposition, 2026-11-01

User explicitly requested profiling. Reconciled the completed bounded pilot
(56 processes); no benchmark was restarted. Froze baseline e674478b and bounded
70ea7122 with their own source snapshots. Collected 11 gated 2 ms gprofng
profiles sequentially; all raw-packet audits pass, with zero samples outside
the timed gate and exact CPU-tick/report reconciliation. Eighteen disassembly
reports also reconcile exactly to function totals. Compressed raw packets and
logs are archived, so replay does not depend on temporary binary paths.

Large GEMM16/32: K loop87.48/87.89%, packing7.51/7.46% aggregate CPU time.
Cached4MiB pool62.68/59.34%; short129/2049 pool81.86/79.61%, worker command
poll70.00/67.14%. Streaming256MiB kernel98.97%. These are aggregate CPU shares,
not elapsed-latency savings or hardware stall counters. perf remains denied
at paranoid4; no system settings changed. Do not infer IPC or cache saturation.

Bounded short GEMV reduces participants16->9 and polling to51.49/51.66%,
but a separate shuffled60-process comparison(3 processes/case/variant,
61 samples x8192 calls) rejects the policy. Short129/2049 medians16:
.406549->.357666,32:.403715->.362284 TFLOPS;192/2049 also loses;8-worker
64/4096:.370832->.337869. Initial pilot gains are not replicated. No new
worker-count/ISA threshold is justified. Restore exact before linear.rs and
executor.rs; verify every CPU library source against the before snapshot.
All experimental geometry/test changes removed. Calibration changes retained.

Separate9-process baseline control: GEMM16/32 4.236325/4.264922 TFLOPS;
streaming16 .041149. Conditional5.8368 roof assumes all16 cores5.7GHz;
actual frequencies unmeasured. Prior verified FMA5.538 is a different workload.
Final114 CPU/executor release tests and fmt pass. Rebuild all four debug-release
examples after tests: e674/297/997aee63/af1ee857 match frozen expectations.

Archives: linear-targeted-profile-2026-11-01(11 profiles,197 unprofiled runs
including both geometry pilots, raw-packet verifier) and
linear-wide-depth-2026-11-01(112 runs, rejected depth sources and checks).
Both rejected hypotheses are closed; overall goal stays active. Next work must
use a concrete instruction/critical-path hypothesis, not another unconstrained
grain/depth sweep or extrapolation from lower aggregate polling CPU time.

### FMA memory broadcast hypothesis, 2026-11-01

Previous goal turn is progress:11 audited profiles, rejected geometry with
60 repeated comparisons, exact source/binary restoration. Current four hashes
match e674/297/997aee63/af1ee857 and no heavy process is live.

Installed LLVM-MCA21 accepts znver5 but uses Zn4 resources. Its register-only
16-FMA control models16 cycles/block, corresponding to at most2.9184 TFLOPS
at all16 cores5.7GHz, contradicted by measured5.538. Reject this model as a
quantitative AVX512 roof/port model; retain outputs as diagnostic evidence.
Do not optimize against its claimed FP1 bottleneck.

Concrete next hypothesis: fold four explicit input broadcasts in the4x64
microkernel into the16 FMA memory-broadcast operands. This removes4 source
instructions/register temporaries but increases scalar memory reads from4 to16
per K; actual performance may lose. Keep panel128,packing,output paths,group
selection and all public contracts unchanged. Implement a private AVX512-only
4-row accumulation helper, preserving ordered FMA and caller's initial sums.
R1/R2 tails retain the established code. No CPU/model switch.

TDD: helper starts as an inert stub; direct exact-buffer tests check all256
outputs, initial sums, arbitrary input stride/unalignment, D0/1/2/17/127/128
and input immutability against independent f64. Observe RED before assembly;
then GREEN and existing direct/compact/bias/tail integration tests. Inline asm
may read memory but must not write it or touch stack: readonly/nostack, explicit
read-write accumulators and pointer/count operands, early-clobber weight scratch
registers (lateout would wrongly permit overwriting live accumulators).

Inspect actual integrated assembly for folded loads, loop size and spills;
compare frozen actual CpuExecutor binaries on main16/32,compact output,
shortK,long batch and odd tails. Retain only repeatable useful gains with
necessary boundary tests and source review; otherwise restore exact baseline.

FMA memory-broadcast experiment is rejected. New helper RED at depth1 on inert
stub; GREEN and115 prototype release tests pass. Actual run/compact loops both
have24 instructions,16 folded memory broadcasts, no explicit broadcast or RSP
operand in the inner loop. Pilot14 processes loses18-21% on main/compact/long
batch. Repeated12 processes confirm16:4.244517->3.401829,32:4.197587->3.400421
TFLOPS. Short64-row unchanged control is neutral. No reason to retain a niche
selector for the one noisy odd-case gain. Increased scalar load count is visible
in source; hardware cause remains unmeasured, so do not claim port saturation.

Snapshot review /root/review_memory_broadcast found no Critical/Important, one
Minor: safety text requires four full k-rows from the already panel-shifted x,
while actual valid reads require depth floats at x+r*k for each r<4. Preserve
the measured snapshot and finding, remove entire helper rather than polish
rejected code. Restore exact original wide.rs; all CPU library files match
before.114 restored release tests/fmt pass and four debug-release binaries
again match e674/297/997aee63/af1ee857. Archive: linear-memory-broadcast-2026-11-01,
including the rejected quantitative LLVM-MCA model and register-FMA control.

Next remaining concrete GEMV gap: the old plan's scalar-output-tail reduction
experiment is still unimplemented. Current C1 uses only4 accumulators on AVX512
and2 on AVX2, while the existing contiguous streaming primitive has8 for either
ISA. Inspect reuse of that primitive for sufficiently long single-output tails
without changing C4 geometry or pool/group selection. Avoid repeating rejected
C8 output tiling or coarse participant rules. First isolate pinned one-thread
C1/C2/C3 and full-C4 controls, then actual odd-stripe parallel cases; preserve
small-K behavior and numerical/unaligned/bias/extent contracts. Overall goal
remains active; no broad physical-efficiency or completion claim is established.

### Long single-output GEMV reuse, 2026-11-01

Previous goal turn is progress: FMA memory-broadcast26-run experiment rejected,
MCA roof falsified by a register-only control, original library and four frozen
binaries restored. Revalidate e674/297/997aee63/af1ee857 and no live heavy job.

Hypothesis: C1 scalar-output tails have insufficient independent reductions
(AVX2 two, AVX512 four). Reuse the existing eight-chain contiguous-row kernel
for C==1 when K spans at least four groups of eight vectors. This amortizes
its longer final reduction, preserves small-K behavior, leaves C4 tiling,
prefill packing, pool size/order/groups and all public contracts intact.
Use a shared accumulator-count constant. No duplicate kernel body or model/CPU
name. The change is a private primitive dispatch, not enabling the large-weight
streaming executor policy for skinny matrices.

TDD: per-thread test-only observations count actual eight-chain row entries.
New full-result/f64/bias/guard/unaligned test crosses the two ISA-derived K
boundaries and checks m1/2/3/4/5/7, rows0/1/2/3/17. Odd final rows use C1
only for feature remainders; even batches/full-C4 and shortK retain old routes.
Observe old-route RED, then reuse borrowed exact x/weight/one-output slices
with rebased one-element bias; assert GREEN and full release correctness.

Compare frozen before/after in pinned inline one-worker runs with affinity none,
C1/C2/C3 plus C4 controls across short/long K and unaligned buffers; then actual
parallel odd stripes and regular GEMM/GEMV controls. Measurement must include
the real CpuExecutor API, not just direct kernel. Retain only repeated gains
with controlled collateral, independent source review and normal final checks.

Initial tail-reuse prototype498a7bf2 passes115 release tests and clean snapshot
review. Pinned local16-process pilot and156 repeated processes establish AVX2
long-tail gain: C1 K4097 .049108->.082423,8193 .047088->.068772 TFLOPS;
C3K8193 .047814->.066995. C4 control is neutral. AVX2K256/257 loses4-7%;
AVX512K512/513 loses~9%,1025 loses5.5%,long8193 is near neutral.

Refine from measurements: private kernel-macro enable flag true only forAVX2,
and require8 groups of8 vectors(K>=512). AVX512 retains its original C1
reduction. Revised actual-route/f64 test RED catches K511 incorrectly entering
dot8 against the first prototype; GREEN/full115release pass for e8778cb6.
Read-only followup review clean. Pinned90-process repeat confirms C1K4097
.049012->.082389,8193 .047067->.069025;C3 .047869->.067100;C7 .069399->.084876.
Unaligned C1 .046907->.056218. AVX512 controls within~1.4%.

Parallel40-process pilot then102 repeated runs show no established odd-width
win; some full-C4 controls also lose~3-4%. Therefore performance acceptance
is NOT complete. Do not hide collateral behind isolated one-thread gains.
Disassembly shows a new out-of-line call from AVX2 gemv to streaming::avx2,
and changed general-register allocation in C4. Both main C4 loops still have
14 instructions/8 FMAs and equal stack-frame size; scheduling cause unproven.

Next bounded experiment changes only an inline hint on the shared primitive,
so the exact-one-output call can specialize and avoid the extra call/division
machinery. Inline variant40181347 passes115release; actual GEMV now has no
streaming call (only cold panic calls). Frozen before/refined/inline pilots
separate effect of the hint from tail selection. This is a codegen hypothesis,
not proof that call removal fixes parallel losses. No pool/group/geometry edit.

Leaf variant28b38592 isolates dispatch from arithmetic: scalar gemv wrapper
splits a full-C4 prefix and1..3 tails, gemv_base/decode_base preserve arithmetic;
packed-prefill decode wrapper retains its C1 fast path. No new arithmetic body.
Route test extends rows65,m33 to observe packed-prefill C1 as well.115release
pass; snapshot review clean. Both gemv_base functions only call cold panics.
Inline-hint variant is superseded; shared streaming primitive has no new hint.

Leaf324-run broader verification preserves large GEMM and streaming but finds
short scalar K257/511 overhead7/5%. Static inspection proves scalar gemv wrapper
was still outlined (six saved GPRs, extra call even on short K). Make only this
scalar wrapper inline(always), which is legal because it has no target_feature
or SIMD operations. Direct b82f08ed passes115release and wrapper symbol vanishes;
review of the sole attribute change is clean. Another324-run paired series:
AVX2 C1K4097 .049355->.083464(+69.1%),K8193 .047050->.067857(+44.2%),
C3K8193 .047838->.067713(+41.6%),C7 .069374->.084920(+22.4%).
K257 nearly neutral; K511 retains~3% loss. Large GEMM and streaming are near
baseline. Some parallel controls still need resolution: AVX51216 m511/K8193
-7.6% while32 gains5.5%; four-row32 AVX512 -8.7%,AVX2 -4.5%;cached~1-3% loss.
These are not accepted collateral or universal improvements. Final acceptance
is still pending, even though isolated target gain is strong.

Static source audit finds an avoidable new ABI/check cost: private unsafe gemv
already receives validated extents, but checked w/bias prefix slicing retains
extra lengths and panic branches throughout scalar dispatch. Next bounded
variant forms exact dimension-derived views from pointers under that existing
contract, including k-element x, columns*k weights and suffix extents, and
rebases bias by pointer. This removes duplicated checks without changing legal
inputs, selection threshold or math. Normalize optional bias extent to m.
Requires fresh source review,115release and targeted controls before adoption.
Do not switch thresholds again while unresolved collateral remains. Do not
infer measured stall causes from source-only register/allocation changes.

### C1 investigation checkpoint: integration deferred, original restored

Views d45ba808 passes115release and full-workspace/all-target Clippy with
warnings denied; fresh raw-view safety review is clean. Five-process targeted
controls retain C1K4097 .049139->.083449(+69.82%) andK8193 .047138->.068227
(+44.74%), full C4 control .105010->.103341(-1.59%),K511 .045221->.043883(-2.96%).
Cached4MiB now within1% and odd AVX512 prefill -0.5%; earlier cached/m511
large losses do not repeat. But four-row32 still loses AVX2 .971802->.898092
(-7.58%) and AVX512 1.130865->1.037908(-8.22%). These persistent common-path
losses prevent retaining C1 integration. No additional threshold search.

Two gated profiles of exactly that AVX512 four-row32 case pass raw-clock audits:
31x512 calls, no samples outside gate, exact tick/report reconciliation. Before
kernel70.24%/pool29.02%;views kernel67.70%/pool31.59%. Kernel CPU seconds39.490
versus39.276 are close; profiled gate2.1166vs2.0824s even reverses the unprofiled
latency ordering. Do not use profiled throughput as acceptance or infer that
poll reduction is wall-speedup. Source/profile evidence does not establish
the hardware cause of the unprofiled loss.

Actual R4/C4 dot loop is34 instructions in both builds:16FMA,8vector loads,
4vector-register moves,2RSP reads,4control. Only some GPR assignments and
alignment differ (before mod64=0,views=16). The4moves and2stack reads were
ALREADY PRESENT in before; do not attribute newly added arithmetic work to
the C1 change. The larger ISA run function includes both packing and dot paths
(~3400instructions forAVX512), so self-contained dot-leaf codegen is a concrete
next hypothesis. First inspect whether isolating batch<R4,C4/C2> removes moves
or spills; compare compiled code before timing, keeping memory loads/FMA order
and all public contracts unchanged. Avoid redoing failed FMA-memory-broadcast.

Separately, present executor selects physical group only for GEMV or eligible
wide AVX512 prefill; four-row GEMM still uses native32SMT. Same-series16-worker
four-row controls are~1.40TFLOPS AVX512 and~1.07AVX2 versus32~1.04/.90. Prior
SMT archives changed pools/publication/GEMV groups, not a shape-based R4 GEMM
physical-only policy in the current command/group implementation. Inspect
historical data before any repeat. Trial must preserve configured pool/native
CPU order, derive eligibility from shared kernel/job geometry and OS topology,
retain unknown/unbound fallbacks and not use CPU/model names. Evaluate all
affected batched/narrow/odd cases, not just rows4. Isolate arithmetic and policy
changes with separate before/isolation-only/physical-only/combined snapshots.
Then reapply archived C1 source and compare against the matching new baseline,
so another speedup cannot hide the C1 collateral loss.

Archive linear-single-output-2026-11-01 contains7 full source snapshots,
24 timing series/1287 processes,2 raw-clock profiles, source reviews and gates.
All experimental library/test changes removed by exact byte restoration of
linear.rs and streaming.rs.114 restored release tests/fmt pass; debug release
rebuild of all four examples matches e674/297/997aee63/af1ee857. Root CPU source
is the accepted baseline; calibration is retained. No heavy process remains.
Goal is active: strong C1 arithmetic win is an unintegrated candidate, not
completion. Next goal turn should start from these exact restored sources and
the specific R4/group hypotheses above, not rerun the completed six C1 layouts.

### Short dot-batch physical group, 2026-11-01

Previous turn is progress:1287 measurements/two new profiles, strong C1 gain
isolated but unaccepted, exact baseline source/four binaries restored. Current
preflight confirms e674/297/997aee63/af1ee857 and no live CPU workload.
Ruling: test the existing physical group on short GEMM before attempting a new
arithmetic leaf. Profiles did not show extra arithmetic work; same-series16
workers outperform32 on the failed case. This isolates a scheduling hypothesis
with unchanged vector arithmetic. Cost if wrong: another bounded measured
policy must be rejected; do not hide the pending C1 issue with a new speedup.

Eligibility: SIMD, at least one complete4-row register group, below the ISA's
packing-row threshold(AVX51216,AVX264), and at least one4-column partition tile
per physical worker. Derive limits from shared existing geometry. This changes
the short dot-batch regime only; packed/wide prefill,GEMV,Scalar and unverified
placement retain existing policy. Selected short rows can never activate wide
arithmetic (their upper bound is below128). Configured pool/order immutable.

TDD actual FixedPool test uses3 native workers and permuted group[2,0], widths
1/2; rows3/4, width7/8, packing boundaries15/16 and63/64, zero/tiny work,
no hint andScalar. Observe selected group and exact native member IDs, preserved
configured count/CPU list, no wide-kernel entries, full f64/bias/unaligned/
guard results, and invalid extents leaving output/submission unchanged. RED
against existing native short-batch selection before implementation.

Share DOT_ROWS/DOT_COLUMNS and packing thresholds with actual kernel/partition
guards; build a constants-only normalization and compare .text before enabling
policy so bookkeeping does not masquerade as an arithmetic change. Then GREEN,
release tests, frozen policy build, bounded paired pilots across short rows,
thread counts17/24/32 and unchanged16, widths/K/odd strides and packed controls.
Only repeat affected promising cases/failed controls; do not rerun earlier C1
layouts. Arithmetic extraction and C1 reapplication remain separate work.

### Profiling-directed restriction of short-group selection

Broad candidate749 is not accepted: repeated large32 high-row loses6-14%,
while cached4MiB gains58-119%, partial24 gains73-105%; small AVX512 stripes
lose8-13%. Do not encode cache/row/work thresholds from this conflicting sweep.
Eight new exact-binary gated2ms profiles compare32large,24large,32cached and
32narrow. Raw CPUIDs show partial24 CPUs0..7/16..23 dominate sampled CPU,
while CPUs8..15 are lightly occupied. Equal feature stripes per logical worker
place twice the work per physical core on the SMT-populated half. The physical
group equalizes that distribution without changing kernel arithmetic.

Ruling: restrict new short-batch policy to native_count % physical_count !=0,
with positive physical_count and native_count > physical_count. A nonzero
remainder proves equal integer logical-worker counts across cores impossible;
a zero remainder does NOT prove balanced placement. Conservatively retain old
behavior there. Keep existing geometry and packing/wide/GEMV/fallback contracts.
This preserves demonstrated partial17/24 gains and avoids speculative cached
or balanced32 rules. Future cached32 improvements remain independent work.

TDD extends the real permuted FixedPool test to native3/4 with physical2;
balanced4 expects native participants, uneven3 expects physical participants.
Width1 short cases now remain inline via existing native partition fallback.
Profile audit must permit sleeping workers explicitly: default strict policy
stays unchanged; opt-in busy-workers checks at least one worker within10% of
peak sampled CPU time, reports CPUID/ticks/cadence for every worker and still
rejects wrong-period, out-of-gate and tick/report mutations. No hardware stall
or physical-utilization claims follow from these statistical CPU samples.

### Short-group outcome

Retain restricted89c01f33 as a measured scheduling change. Actual balanced4/2
dispatch RED/GREEN,216workspace/115release/Clippy/fmt/MSRV/fourCPU0checks pass.
Final debug build matches frozen89/297/997aee63/af1ee857. Independent refinement
and raw-clock audit review has no required issues. Archive
linear-short-group-2026-11-01 verifies266files,252unprofiled runs,eightprofiles.
Partial17/24 targets improve37-107%; large GEMM4.222->4.171TFLOPS is unchanged
within~1.2%; small controls still show unexplained losses up to5.67%. A -14.28%
32-worker short-series loss becomes -.04% in the longer5-process control.
Do not treat normalized identical arithmetic instructions or CPU-clock samples
as proof of identical stalls; code addresses/alignment changed. Overall goal
remains open. User explicitly objects to10hours and repeated experiments;
no more broad retries or six-layout C1 searches without a new concrete basis.

### Task15: foreign implementations and bounded language comparison

User asks for prior art and C/assembly after calling out excessive sweeps.
Inspect pinned AMD BLIS Zen5 SGEMM32x12/6x64, OpenBLAS AVX512 and Rust gemm
sources. Compare an isolated downloaded OpenBLAS0.3.32 pthread library with
the exact frozen89 Lumen through the same benchmark data/oracle/timing loop.
No project/system dependency change. External taskset masks bound both;
OpenBLAS individual worker binding is not controlled and must be disclosed.
Use only representative shapes and a bounded repeat, not threshold sweeps.

Separately create throwaway same-ABI 4x64 AVX512 packed outer-product leaves:
current Rust intrinsics, equivalent GCC/Clang C intrinsics, and hand assembly.
All use the same coefficients/FMA order, dynamic K/depth/stride and arbitrary
f32 alignment. Full f64/guard checks precede timing. Compare actual codegen
and pinned one-core throughput; this isolates language/codegen under hot L1,
not whole-engine speed. Integrate only a demonstrated winner and then check
the complete CpuExecutor; do not infer that any C/asm rewrite automatically
helps or copy model/CPU-name dispatch into engine code. No new goals.

Task15 outcome: handwritten4x64assembly is only+.17% versus Rust in corrected
12-run hotL1 comparison; retain no C/asm in engine. Foreign comparisons and
caller-placement controls reveal actual scheduling headroom; archive
linear-foreign-2026-11-01 preserves46files/59checked runs and12provisional.
Universal kernel constraint now explicit. LLVMIR is being discussed as a
possible representation, not an approved compiler implementation. Preserve
model-independent dimensions/numerics/layout contracts and target-dependent
scheduling. Existing Op::Linear already supplies semantic operation.

### Clarified universal-kernel constraint

User rejects per-architecture manual implementations and hand-maintained
ISA schedules. One shared semantic definition plus common automatic
optimization is required; LLVM's target models/backends handle hardware
adaptation. The previous per-target retargetable probe is evidence about
codegen only, not the proposed final architecture. Investigate existing
LLVM/Polly capabilities before building any compiler subsystem. Shared
source, shape correctness, numerics and temporary-memory accounting must
be verified alongside performance. No implementation of this direction
has been approved or started; the current step is a concrete design proposal.
