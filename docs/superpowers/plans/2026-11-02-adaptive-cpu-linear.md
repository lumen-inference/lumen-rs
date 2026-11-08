# Adaptive CPU Linear Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace development-machine scheduling thresholds with bounded learning from real linear calls.

**Architecture:** A policy module owns bounded metadata and chooses safe execution plans; the executor validates and runs exactly one plan. Arithmetic and partitioning retain their existing contracts.

**Tech Stack:** Rust 1.89, std atomics/Mutex/Instant, existing Rayon/fixed pools, optional build-time LLVM.

**Spec:** `docs/superpowers/specs/2026-11-02-adaptive-cpu-linear-design.md`

## Global Constraints

Maintain Rust 1.89, optional build-time LLVM, no new runtime dependencies, no caller affinity changes, and the checked dense-f32 contract.
No hot-path heap allocations, duplicate computation, cached user pointers, CPU-model tables or expanded explicit ISA.
The shared dirty workspace is the authorized implementation; preserve unrelated changes and do not make a broad commit.

## Review Focus

- Concurrent cache replacement must never select invalid group sizes or reuse matrix addresses.
- Explicit Required single-worker placement must survive serial candidate selection.
- Unknown topology and Rayon cannot promise physical-core placement.
- Identical shapes with changed cache residency may retain an old winner; document reset and Fixed policy.
- Empty or invalid operations must not trigger learning or modify output.

### Task 1: Bounded policy

**Files:** Create `crates/lumen-cpu/src/linear/policy.rs`; modify `src/linear.rs` and `src/lib.rs`.
**Interfaces:** Produce public `LinearPolicy::{Adaptive,Fixed}`, private `Plan { group: Option<Group>, streaming: bool }`, and `Tuner::new(&Workers)`, `Tuner::run(shape, streaming_supported, closure)`; closure consumes a Plan and returns Result. Cache entries contain exact keys for learning and atomic fingerprint/plan hints for settled lookup.

- [x] Write policy tests for bounded adjacent sampling, outlier consistency guard, synthetic serial crossover, fewer-worker/streaming winners, no-pool candidates, SMT and unknown topology, collisions and concurrent fallback.
- [x] Run `cargo test -p lumen-cpu --lib linear::policy` and observe RED.
- [x] Implement policy without unsafe code or call-time allocation; settle deterministic ties by candidate order. Time only successful normal calls; recover poisoned metadata.
- [x] Repeat targeted tests and record GREEN.

### Task 2: Real execution integration

**Files:** Modify `src/executor.rs`, `src/linear.rs`, `src/linear/compiled.rs`; add tests in `src/executor.rs` / existing integration tests.
**Interfaces:** CpuExecutor owns Tuner and LinearPolicy; `with_linear_policy(self, LinearPolicy) -> Self`; fixed dispatcher remains private. Promote checked native streaming, add checked compiled streaming entry. Adaptive execution recomputes current rectangles for its Plan.

- [x] Add failing tests proving real serial/group/streaming plans match the oracle, Required single-worker ownership, invalid-before-write and shared executor concurrent correctness.
- [x] Run targeted tests to observe RED.
- [x] Integrate adaptive default after validation; preserve fixed path and ISA/constructor behavior.
- [x] Verify default and feature-enabled CPU tests and numeric exceptional/tail coverage.

### Task 3: Measurements and delivery

**Files:** Modify `examples/bench_linear.rs`, `docs/EN/cpu-performance.md`, `docs/RU/cpu-performance.md`; create `docs/benchmarks/linear-adaptive-2026-11-02/`.
**Interfaces:** Benchmark env `LUMEN_LINEAR_POLICY=adaptive|fixed`; output identifies policy, warm-up and bounded tuning. No library environment reads.

- [x] Add benchmark policy switch and sufficient explicit warm-up; record before binary and source snapshot already preserved in `/tmp/lumen-adaptive-2026-11-02`.
- [x] Run a bounded representative comparison, report actual results and regressions; fix material regressions with new evidence.
- [x] Run fmt, host Clippy/workspace tests, MSRV and available ARM cross/QEMU checks serially; save command logs.
- [x] Request fresh whole-change source review, address important findings with RED/GREEN; publish limits and final validation.
