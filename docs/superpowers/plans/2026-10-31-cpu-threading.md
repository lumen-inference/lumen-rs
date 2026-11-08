# CPU topology and SMT implementation plan

> **For agentic workers:** Use superpowers:executing-plans or superpowers:subagent-driven-development to implement and verify each task.

**Goal:** Improve configurable CPU execution on shared-cache, multicore and SMT machines, with reproducible measurements against the existing implementation.

**Architecture:** Discover allowed CPUs and physical-core/cache/NUMA relationships once at pool creation. Select physical cores before SMT siblings and bind only owned workers. Keep configuration in `lumen-config`, topology and execution in `lumen-cpu`, and config-to-backend wiring in `lumen-engine`.

**Tech stack:** Rust 1.89+, Rayon, Linux affinity/sysfs through libc; portable unbound fallback.

**Spec:** User-approved scope in this conversation: topology-aware placement and measured scheduling improvements, preserving the configured worker count and avoiding processor/model-specific branches.

## Constraints and decisions

- Continue the existing uncommitted SIMD work in this checkout; preserve unrelated changes and baseline binaries under `/tmp/lumen-threading-before`.
- Explicit worker count is honored; auto sizing uses physical cores minus reserved cores, at least one. Do not silently cap 32 requested workers to 16.
- Respect the calling thread's allowed CPU mask; never alter its affinity or system policy.
- `affinity`: auto with a reported fallback, none for unbound execution, required for a checked binding guarantee on pool workers. Small inline operations retain caller placement.
- Distinguish NUMA nodes from shared-cache domains; no hardcoded CCD count, CPU IDs, vendor or model names.
- Performance measurements run serially, without concurrent builds/tests. Preserve checksums, shapes, placement and variation; do not claim a physical ceiling.

## Review focus

Restricted/sparse CPU IDs and SMT-only masks; absent/malformed topology; oversubscription and pinning errors; output tails and unaligned stripes; configured values reaching executor creation.

## Tasks

- [x] Add topology selection and affinity with synthetic topology tests and an isolated real affinity test. Check physical-first selection, contiguous work ordering, reserve, NUMA spread and mask limits.
- [x] Add `CpuThreading`, `Affinity`, checked worker startup and placement diagnostics. Keep existing single-thread default and `new(isa, 0)` validation. Test unbound/auto/required paths and numerical equivalence.
- [x] Add config affinity parsing, engine adapter and runnable decoder thread argument; test defaults, explicit settings and invalid values.
- [x] Measure static topology-ordered prefill scheduling against dynamic stripes; keep only justified changes. Preserve full-token weight reuse and safe disjoint output ownership.
- [x] Run workspace tests, release CPU/exec tests, fmt, clippy and Rust 1.89 build. Obtain an independent review.
- [x] Compare saved old and final binaries for 8/16/32 workers, AVX2/AVX-512, several matrix sizes and attention; record methodology and results in RU/EN CPU docs.

## Execution notes

2026-10-31: existing `CpuExecutor` uses an unbound persistent Rayon pool; config CPU values are parsed but engine wiring is absent. Existing `taskset` process masks do not bind individual workers. Saved pre-change source and release benchmark binaries before edits. Independent review identified potential unaligned output cache-line sharing; static placement is an experiment, not an assumed win.

2026-10-31 completion: topology discovery, worker affinity and configuration adapter implemented. Explicit counts remain exact; startup errors and Rayon's maximum worker count are checked. Independent review identified the oversized-count startup deadlock; a pre-build check and regressions now cover it.

Static per-worker prefill was implemented and measured, then removed: five process runs of 501 samples showed mixed improvements and worse outliers. Final kernel arithmetic and dynamic work distribution are unchanged from the saved baseline. The new output-guard regression remains.

Final A/B: 270 cases across 3 variants, 2 ISAs, 3 worker counts and 5 workloads, in three shuffled rounds. Fixed-ISA/count checksums match before/after; attention's independent oracle passes. A comparison-script assumption of identical checksums across worker counts was corrected after confirming the same small rounding difference in both versions. Full raw data and the corrected runner are in docs/benchmarks/2026-10-31-threading.* and compare-threading.py.

Verification: cargo test --workspace (164 passed); cargo test --release -p lumen-cpu -p lumen-exec (63 passed); cargo clippy --workspace --all-targets -- -D warnings; cargo fmt --all --check; cargo +1.89.0 build --workspace --all-targets; git diff --check. All succeeded. RU/EN CPU and configuration docs describe the actual implementation, measured regressions as well as gains, and current CLI limitations.
