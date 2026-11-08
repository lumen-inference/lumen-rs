# GEMV regression follow-up — completed

User authorized one CPU hour from 2026-11-07 16:54:47 UTC. Guarded heavy commands had a 17:54:15 UTC deadline (22:54:15 Asia/Yekaterinburg), with termination grace. Heavy work finished around 17:45 UTC. No commit/push requested.

1. Reproduced K65 and adaptive cached8 losses using frozen binaries and actual learning traces.
2. Rejected pre-merge chain cleanup. Retained separate original/cleanup GEMV functions behind the existing experimental feature. K65 recovered; Fixed cached8 still loses speed, so no default promotion. A later attempt to limit the original loop to one unrolled chunk was also rejected after matched measurements. Its historical binary name `retained` does not mean acceptance; the selected binary is `final`.
3. Added deterministic failing tests for a delayed candidate and noisy seed. Retained candidate median-of-three against the seed minimum, at most 32 successful uncontended calls without eviction, one output computation per call, no inference allocations. Expanded oracle coverage through convergence.
4. Tested conditional worker waking. Rejected per-worker SeqCst flags. Retained machine-word bitmaps with an AcqRel RMW register/recheck chain, preserved excluded bits and linear metadata. Added deterministic park-entry probes, sparse/reordered masks spanning more than two words, and real cross-word workers. Independent review found no remaining blockers.
5. Completed matched Fixed/Adaptive/native/attention comparisons. After the final rejection, restored the selected source, repeated all 21 validation commands and confirmed byte-identical benchmark executables. Default workspace 238 tests, LLVM/all-features 255 each, release CPU/executor 154; fmt, Clippy, Rust 1.89, AArch64/ARMv7 kernel/API/pool/allocation tests passed. Allocation and publication suites passed 80 repeats.
6. Updated EN/RU performance documentation and archived accepted/rejected measurements, sources, hashes and logs in `docs/benchmarks/linear-gemv-followup-2026-11-07/`.

The bounded follow-up is complete. Experimental arithmetic performance remains shape-dependent; the broader hardware-ceiling objective and real ARM performance are not complete. GPU/Vulkan work remains a separate direction and was not changed during these CPU experiments.
