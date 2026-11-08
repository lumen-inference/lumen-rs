# Universal linear kernels implementation plan

> Execute inline with superpowers:executing-plans and TDD; finish with an independent source review.

**Goal:** One CPU-independent LLVM description computes dense-f32 GEMM/GEMV for dynamic shapes, with automatic vectorization and an actual CpuExecutor route.

**Architecture:** An opt-in `llvm-kernels` Cargo feature compiles a shared LLVM module ahead of time. LLVM chooses vector width/unrolling; AVX feature multiversions differ only in compiler flags and symbol names. The baseline target also compiles on AArch64, without a handwritten ARM kernel. No LLVM runtime, JIT, new crate, per-model dispatch, manual ISA tile schedules, packing copy or hot-path allocation.

**Scope:** This is the first executable migration step under the existing maximum-performance goal. Native kernels remain the default comparison until generated performance is accepted. Do not call the whole optimization goal complete from this deliverable.

## Constraints

- Rust MSRV1.89; preserve the supplied dirty feature/block-graph workspace and unrelated foundation changes.
- Existing checked shapes, zero-row behavior, optional bias, unaligned data, output gaps and invalid-input-before-write contracts remain.
- f32 reduction order may vary; allow reassociation/contraction/signed-zero changes explicitly, never assume NaNs/infinities absent. No full fast-math.
- Ordinary builds do not require Clang. Enabling `llvm-kernels` requires a compatible Clang (LLVM18+ opaque-pointer IR); missing tools fail with an actionable build error.
- Output pointers never alias inputs; readonly inputs may alias each other. Every generated call joins synchronously.
- Generic loop has no scratch allocation and no static vector width/tile in its source.

## Task1 — compiled common kernel

Files: `crates/lumen-cpu/Cargo.toml`, `build.rs`, `src/linear/llvm.ll`, `src/linear/compiled.rs`, `src/lib.rs`, `tests/compiled_linear.rs`.

- [x] RED: expose feature/API stub; actual f64/guard/unaligned/bias/zero-row/invalid-shape tests fail because the generated kernel is absent.
- [x] Implement one LLVM function with runtime rows/M/K/output stride and nullable bias. Compile identical IR for the portable target and optional AVX2/FMA/AVX512F flags; automatic LLVM vectorization, no dimension constants.
- [x] Safe public `llvm_linear(x,w,bias,out,m,k)` uses shared validation and runtime feature checks. Private unsafe strided call serves checked partitions.
- [x] GREEN: force every available generated profile against f64; NaN/Inf propagation and readonly-input aliasing covered; mutate generated sum to zero once to prove real generated arithmetic is exercised.

## Task2 — actual executor and reproducibility

Files: `src/executor.rs`, `src/linear.rs`, `examples/bench_linear.rs`, CI and CPU-performance documentation.

- [x] Add `CpuExecutor::with_llvm_linear(CpuThreading)` using the same joined worker pool and partitioner. Compilation occurs at build time, never inside `linear`.
- [x] Test parallel strided outputs/tails/bias/invalid-input guards and real `Executor` dispatch. Native-only builds retain their existing path.
- [x] Add explicit benchmark selection without changing existing positional arguments. Compare only representative GEMM/cached+streaming GEMV; preserve raw timings and source/binary hashes.
- [x] Compile the identical LLVM source to AArch64 object code; disclose that no ARM runtime/performance test was performed without a runner.
- [x] Run workspace/default tests, feature tests in debug/release, Clippy/fmt and MSRV. One fresh independent review; fix required findings and rerun affected checks.

## Review focus

- Nullable/aliased inputs, empty rows and padded output rectangles must preserve Rust borrowing/bounds.
- Generated vector feature requirements must be checked before executing foreign code.
- No missing compiler may silently change requested execution mode.
- Feature selection must reach generated arithmetic through parallel and graph dispatch, rather than only a standalone example.
- AOT cross compilation must use the destination triple and linker; no host-native flags in portable builds.
