# Cached GEMV: common vector cleanup

**Status:** implementation and correctness validation complete; experimental opt-in retained because performance varies by shape. Promotion rejected. The broader cached-GEMV hardware-ceiling objective remains open.

**Goal:** reduce avoidable scalar remainder work in the shared LLVM GEMV/dot kernel, without CPU-model dispatch, another ARM algorithm, packing user tensors, or runtime allocation.

**Final design:** `experimental-dot-kernels` implies `llvm-kernels`. The common generator consumes complete SIMD vectors left after its unrolled loop, then performs the existing independent LLVM horizontal reductions and a scalar remainder shorter than one vector. Ordinary `llvm-kernels` retains its previous schedule. Target-provided width and register budget determine geometry.

The user initially reserved the CPU, then authorized computation until 20:00 Asia/Yekaterinburg on 2026-11-07. Builds, tests and measurements ran under a process-group guard ending at 19:59:30, with a two-second kill grace period.

## Work completed

- [x] Add exact-value comparisons for distinct columns, bias, unaligned inputs and output gaps; cover K=1..257, rows 1/2/4/5 and output widths 1/3/4/7/8/9.
- [x] Check per-column NaN/Inf behavior and protected-page boundaries, including an AVX-512 streaming main iteration followed by vector cleanup and a scalar remainder.
- [x] Add the opt-in feature, common LLVM cleanup and benchmark feature metadata.
- [x] Compare joint reductions, narrower reductions, wider output tiles, a remaining-count cleanup loop and a separate baseline/cleanup dispatch. Reject these additions after measured regressions; keep only vector cleanup.
- [x] Fix a preexisting unbound-pool startup race uncovered by allocation tests: wait for an empty Rayon broadcast on every worker before returning the constructor. Repeat the allocation test 80 times without failure.
- [x] Validate default, LLVM and experimental workspace tests; release CPU/executor tests; Clippy; Rust 1.89; AArch64/ARMv7 execution through QEMU with baseline and experimental features.
- [x] Obtain an independent final review. Extend test coverage in response to its single minor finding; final review has no remaining findings.
- [x] Add baseline/experimental ARM and x86 CI commands. Local equivalents passed; hosted CI was not triggered here.
- [x] Measure matched executables on scalar, AVX2 and AVX-512, including cached GEMV, vector tails, streaming GEMV, small batches and GEMM; include 1/2/4/8/16/32 configured workers and Fixed/Adaptive policies.
- [x] Keep the candidate opt-in and document regressions instead of promoting it.

## Boundaries

No tensor ABI, CPU-specific dispatch table, full weight copy or new runtime dependency was added. Reassociation and signed-zero permissions remain as before; no finite-input assumption was introduced. Shared preexisting foundation files remain uncommitted.

Correctness through QEMU does not establish native ARM speed. Other x86 microarchitectures, sustained frequency, hardware-counter cache traffic and end-to-end model performance remain unmeasured. The experiments do not establish physical saturation or a universal throughput gain.

## Evidence and prior art

Measurements, rejected experiments, raw outputs, source snapshot, review and reproduction (`../../benchmarks/linear-dot-2026-11-07/README.md`).

The [MIT gemm implementation](https://github.com/sarah-quinones/gemm/blob/main/gemm-common/src/gemv.rs) uses vector cleanup before the scalar tail. We implemented that scheduling idea in our generator; no upstream source was copied. [MIT ggml](https://github.com/ggml-org/ggml/blob/master/src/ggml-cpu/ggml-cpu.c) also motivates a separate future investigation into graph-level dispatch; no worker-pool code was imported.
