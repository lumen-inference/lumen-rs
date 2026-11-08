# Prepared linear kernels and submission overhead

> **For agentic workers:** Use superpowers:executing-plans. Continue in the existing dirty feature checkout; preserve the unrelated foundation. Do not commit or push this combined worktree.

**Goal:** Remove repeated weight packing for explicitly prepared immutable matrices and reduce redundant work on repeated pool submissions, with measured before/after results.

**Architecture:** A CPU-owned prepared matrix stores one blocked copy, with dimensions and the selected generated kernel. The same target-independent LLVM outer product consumes that layout on x86 and ARM. Preparation is explicit: the ordinary slice API permits changing weights and cannot safely cache them by address. Prepared execution avoids weight packing, retains cache blocking for large prefill, handles tails without writing beyond the output, and allocates nothing on successful calls. Whole-K accumulation is limited to decode and short batches after experiments rejected it for large prefill. Pool changes retain publication/join and panic safety.

**Tech Stack:** Rust 1.89, existing C driver and LLVM IR generator, existing thread pool.

**Spec:** User's existing requirements: arbitrary matrix sizes and CPUs including ARM; no CPU-model thresholds; fix GEMM/GEMV overhead; ten minutes of measurements authorized after preparation.

## Constraints and review focus

- Preparation owns its data; invalid shapes fail before output writes. No pointer-address cache or implicit extra model copy.
- No new dependencies, no architecture-specific math implementation. Native/raw paths remain available.
- Validate K/column/row tails, unaligned output, exceptional values and checked size arithmetic.
- Validate repeated and concurrent prepared calls, no execution allocations, caller changes and blocked workers.
- A prepared API is not automatic graph integration: generic model bindings currently expose row-major weights. Document the explicit API and memory cost rather than silently exceeding model budgets.
- Do not infer a speedup from instruction removal. Keep only measured pool improvements; report limitations of the prepared layout by shape.

## Tasks

- [x] Save baseline sources and executable; add failing prepared API correctness tests against the independent reference.
- [x] Implement owned prepared weights and generated packed-input entry, with checked CpuExecutor preparation/execution and parallel output partitioning. Reuse generated primitives. Add benchmark prepared mode outside the timed preparation phase.
- [x] Test same-caller reuse and caller handoff; remove redundant caller-handle locking while preserving completion/parking invariants.
- [x] Run focused correctness/allocation checks, workspace tests, formatting/Clippy and available ARM cross checks with bounded build concurrency.
- [x] Freeze candidate binaries. Run matched measurements within one ten-minute wall-clock budget; retain/reject changes using the results. Record TFLOPS, source/binary hashes and limits.
- [x] Obtain independent code review, resolve material findings, update EN/RU performance docs and record verification.

## Progress

Initial diagnosis: old profiles identify packing, scratch traffic and pool waiting; current LLVM stall breakdown remains unknown. Baseline sources saved under `/tmp/lumen-linear-prepared-2026-11-07/before`. No baseline timing was run before the new authorization.

Follow-up: LLVM CPU tests passed. Independent review found no runtime safety defect; corrected prepared schedule reporting and expanded concurrent cases above the parallel threshold. Default workspace, all-features workspace, Clippy and Rust 1.89 passed before the pool follow-up.

Ruling: ARM allocation failure was reproducible 10/10 times. Temporary allocator backtrace showed a 1520-byte `crossbeam_deque::Injector::push` block allocation through Rayon submission. The previous 40-call test could miss this periodic behavior. Changed unbound and required-single-worker execution to the existing FixedPool protocol; affinity is an explicit constructor option, unknown CPU sentinels never imply CPU sharing. Rayon remains only a differential-test backend. Allocation tests now cover 160 calls per phase and required single-worker placement. This is within the original inference-overhead fix; it needs fresh full validation.

Follow-up independent review approved constructor/unknown-affinity/lifetime behavior and found one stale unit-test constructor reference, fixed before verification. The budget had not started at that checkpoint.

Final measurements: 403 successful processes; first publication 2026-11-07 19:06:35 UTC, last benchmark finished 19:16:26 UTC (591.22 seconds). All were within the single 597-second guarded budget. Accepted measured executable is `bands`, hash 34297018bc48277b0c55375b97f92162c81485318d939886b57b7aaadf22ae66. No further performance runs are authorized by this expired window.

Ruling: full-K large-prefill schedules (wide and narrow) regressed and were rejected. Keep existing K blocking/partial sums and directly read prepared panels. Use the existing aligned host buffer. The initial prepared AVX2 GEMV regressed; target-derived grouped packed bands restored its performance. Independent static review found no safety defect in these changes.

Limitations: explicit prepared API is not wired into graph bindings; whole-model storage ownership/budget integration remains separate. Prepared Fixed unbound GEMM regresses, so do not present this format as universally faster. The caller-handle cache has small mixed measured effects; main unbound gain comes from removing queue dispatch.

Final validation: default workspace 238 tests; all-feature workspace 260 tests; release CPU/executor 159 tests. Clippy, fmt, Rust1.89 and AArch64/ARMv7 arithmetic/allocation (6 each) and pool (31 each) passed. Rebuilt executable is byte-identical to measured bands. Logs and source archives saved in `docs/benchmarks/linear-prepared-2026-11-08/`. No commit or push performed. The bounded implementation/measurement task is complete; physical saturation and graph integration are not claimed.
