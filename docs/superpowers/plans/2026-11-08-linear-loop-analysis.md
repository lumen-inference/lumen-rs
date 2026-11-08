# Shared linear-kernel optimization: night continuation

> Use superpowers:executing-plans. Preserve the dirty foundation checkout; no commit or push. This work is part of the active goal «ускорь кернелы до максимального уровня», not a replacement for it.

The user renewed CPU permission with «у тебя вся ночь». The conservative cutoff is **06:00 November 8, Asia/Yekaterinburg (01:00 UTC)**. Individual experiment timeouts do not replace this authorization. Run comparisons sequentially, without concurrent builds; retain commands, sources, hashes, reference checks and raw samples.

The implementation must support arbitrary matrix sizes and CPU architectures including ARM through one arithmetic generator. AVX feature multiversions are allowed; CPU-name rules and a separate ARM implementation are not. Preparation remains explicit, with one owned weight copy in the selected layout and no allocation during execution.

## Completed work

- [x] Inspect baseline loops and extend K=1/2/3, depth-boundary and tail correctness coverage.
- [x] Add optional benchmark warmup duration, outside timing/profile gates. Default zero preserves existing behavior. Tests and independent review passed.
- [x] Compare LLVM loop-unroll-count=2 against frozen baseline in 156 randomized processes. Reject: no consistent whole-call improvement.
- [x] Collect and audit six fresh gprofng profiles. GEMM mainly samples in arithmetic; GEMV includes substantial aggregate worker waiting. CPU-time shares are not call-latency shares.
- [x] Collect standalone APERF/MPERF evidence and independently review interpretation. Frequency and throughput windows differ, so the approximately 92% estimate is illustrative, not measured utilization or a hard ceiling. The register-FMA recurrence converges to constant bits and cannot establish representative GEMM clocks/power.
- [x] Prototype two independent K chains for prepared GEMV: 78 pilot and 130 confirmation processes. AVX2 improves; unconditional AVX-512 use regresses streaming calls, so reject unconditional generation.
- [x] Implement two chains only when target register/tile geometry groups two narrower packed bands. One wide band retains one chain. Factor common address/broadcast/FMA emission; no compiler wrapper is needed at runtime or in ordinary builds.
- [x] Expand nonfinite coverage to every available profile and both chains; establish baseline, then verify implementation. Native release: 150 tests. AArch64 and ARMv7: 16 tests each via QEMU. Clippy passed. Independent review found no Critical/Important issues; its minor causal-wording objection was corrected.
- [x] Compare the actual implementation in 180 processes. Cached AVX2: 0.892166 → 1.021252 TFLOPS. Keep tiny/streaming tradeoffs and unchanged-arithmetic GEMM controls visible. The default executable reproduces the measured binary after cleanup.
- [x] Inspect emitted AVX2 paired loop: eight vector FMAs and no hot-loop stack access per two K steps. AVX-512 object remains byte-identical to baseline.
- [x] Test explicit LLVM prefetch four/sixteen K steps ahead in 81 processes. Reject both; restore default executable byte-for-byte.
- [x] Isolate current-pool overhead and repeated local GEMV. Empty dispatch is about 0.7 µs for 16 workers; amortized AVX-512 reaches 1.67 TFLOPS versus 1.09 with one GEMV per callback. This is diagnostic, not an inference speedup or exact decomposition.
- [x] Measure GEMM worker imbalance and test dynamic task claiming. Equal work takes about 1.86/1.99 ms across the two core groups. Four column tasks per worker are too coarse to help; finer row/column splitting regresses. Neither is retained.

## Current hypothesis: six-row register tiles

The prepared-only doubled K block passed reference/allocation checks but regressed GEMM about 5–7% across both AVX profiles in 96 processes. Reject it.

An isolated copy of the same generator tests six rows rather than four in the widest GEMM tile. AVX2 uses twelve FMA accumulators and AVX-512 twenty-four, fitting the emitted register allocation without accumulator spills. A 108-run pilot measured about 7% faster AVX2 GEMM and 14% faster AVX-512 batch16. Cached AVX2 GEMV regressed about 13%, despite keeping its arithmetic primitive unchanged.

A 192-run factorial control split GEMV into a separate noinline function for both four/six-row variants. It improved tiny inputs but did not resolve the cached AVX2 regression, so neither split is accepted. GDB observations show matching X/packed-weight offsets within pages across binaries; these observations are not timed measurements and do not identify a stall cause.

Two focused controls isolate function alignment and row scheduling: align the prepared function to 32 bytes, and use six rows only in blocked/raw GEMM while preserving the four-row direct prepared path. A 120-run comparison recovered cached GEMV performance. This demonstrates sensitivity to generated code layout, without identifying a particular hardware stall.

The full six-row implementation then ran 240 broader comparisons. It improved rows6/7 but regressed rows8 and narrow padded outputs by about 6–10%. Keep four rows for direct full-K execution and output tails, while using the register-derived six-row tile for blocked GEMM. This intentionally forgoes the small-batch gains to retain its established schedule; no row8 special case or CPU-name rule is added. A further 180 processes measured AVX2 GEMM 2.125319 → 2.257803 TFLOPS and AVX-512 batch16 3.681002 → 4.209135, with the major small/narrow regressions removed.

- [x] Reference-check six-row and split-function prototypes; inspect emitted hot loops.
- [x] Finish the function-layout and blocked-only controls. Aligning the prepared entry to 32 bytes restored the cached AVX2 regression in matched controls while retaining GEMM gains. The specific hardware cause is not established.
- [x] Validate the actual register-budget-based implementation. Expanded reference/allocation checks, 150 native release tests, 16 AArch64 and 16 ARMv7 tests through QEMU, Clippy, formatting and a fresh read-only review of the refined four/six-row policy passed.
- [x] Resolve the noisy base-profile cached GEMV comparison with a longer matched aligned/unaligned control. In 189 processes, explicit alignment reduced base cached GEMV by 3.79%; the default-alignment variant matched baseline. Reject the hint after restoring four direct rows. The retained default-alignment candidate measured AVX2 GEMM 2.110745 → 2.258931 and AVX-512 GEMM 4.309742 → 4.412747 TFLOPS; cached GEMV remains effectively unchanged from the first-stage geometry candidate.
- [x] Restore the ordinary build; archive results and explicitly reject unsuccessful variants. The final executable matches `rows6-blocked-bin` byte-for-byte (SHA256 `df045b9960d3c7ff488688f0093751119136d80a1be349ff4985a672e757ffb2`). The final native release, Clippy and both ARM suites pass again after removing the alignment hint. Sources, logs, hashes and matched comparisons are in the night archive.
- [x] Improve frequency-window correspondence using a standalone diagnostic attached to the benchmark's existing collector gate. All 24 runs independently reconstructed by review. Final AVX-512 GEMM is about 4.37 TFLOPS against an inferred frequency-scaled FMA reference of 4.80 TFLOPS. Preserve CPU-wide/staggered counters, raw versus adjusted clock domains, and independently aggregated medians; this does not establish a hard ceiling.
- [x] Test caller placement using unchanged binaries and verified OS topology in 160 processes. Same-L3 caller placement helps the smaller pool; SMT adjacency alone is not the principal measured issue. No production affinity change is justified. The caller-only archived profile identifies completion polling as its main sampled activity, without equating waiting time to removable latency.
- [x] Diagnose fewer pause hints in isolated copies of the current pool. A 200-process trial was confounded by a 16-byte shift of the unchanged AVX2 GEMV loop: even amortized kernel calls changed dramatically. A second 200-process comparison fixes prepared entry alignment across all variants; no stable coordination gain remains. Reject all polling-cadence changes.
- [x] Test the 64-byte entry alignment in the real executable across 32 cases and 320 processes. Reject: cached AVX2 loses about 3.5–6% while main GEMM is unchanged. ELF inspection shows the hint raises the whole text section's alignment and shifts Rust functions too. Restore the ordinary executable byte-for-byte to `final-tile-bin`.
- [x] Measure completion-signaling cost with an isolated busy-join diagnostic. Concurrency review and 34 private correctness tests pass; the inherited low-CPU-use contract deliberately fails (40.053 ms CPU during40 ms blocked wait). Across140 controlled processes, short-call gains are small/inconsistent and amortized calls regress. Reject the protocol candidate; no production pool edits.
- [x] Calibrate cached GEMV against checked sequential reads of matching per-worker sizes in the same diagnostic binary/pool, with one versus256 inner calls. All252 processes pass output/visit-count checks. At256 KiB/worker, AVX-512 has similar median GEMV/read rates but substantially noisier read results; this is not saturation evidence. At32 KiB, a clearer gap motivates the next arithmetic experiment. Archive all ranges and preserve the distinct conditional L2 bandwidth reference.
- [x] Test and reject two K chains only when the complete padded local weight matrix fits the generator's32 KiB weight budget. Existing grouped-band profiles already use two chains and are unchanged. This is a portable working-set heuristic, not detected L1 capacity. Keep long matrices on their current loop; compare tiny K, boundary/tail shapes, cached and streaming calls before any production change.

The bounded-chain pilot passes nine reference/prepared tests and 132 complete-call measurements. AVX-51264×128 inline improves0.191657→0.251824 TFLOPS, while unchanged AVX2 arithmetic regresses~15% in cached calls. Its identical hot loop moved16 bytes. The candidate is not accepted. Test one further unroll of the current paired-K loop, which differs from the earlier rejected single-chain experiment, to reduce loop overhead and evaluate layout sensitivity without adding architecture-specific arithmetic.

The paired-loop unroll control passes reference tests but fails performance acceptance in 198 processes: AVX2 cached16 loses~17% and tiny K loses~8–9%. Both variants are rejected. Next isolate loop alignment itself with the compiler's portable `-falign-loops=64` option in both ordinary and bounded-chain binaries. Unlike the previous prepared-entry hint, this controls loop boundaries; no production flag is added without broad evidence.

The portable loop-alignment control also fails acceptance in 198 processes: AVX2 cached16 loses6.50% for ordinary arithmetic and 4.15% for the bounded-chain version. Both are rejected. A subsequent instruction-scheduling diagnostic reorders only independent paired-K SSA definitions: expose both input broadcasts early, then interleave their independent FMAs per output vector. Verify emitted machine code before timing; identical output would end this experiment without another benchmark.

The input-scheduling control passes nine reference/prepared tests in each variant. The baseline/preload AVX2 loops have matching addresses and instruction counts. In264 processes, preload-only cached AVX2 improves1.045510→1.066694 TFLOPS (+2.03%), while principal controls remain close. Bounded-chain combinations still regress AVX2 and are rejected.

- [x] Express preload-only ordering through shared generator helpers. Review caught an unintended ordinary multi-row reorder in the first refactor; corrected before acceptance. Differential generation over 12 profile/index combinations matches the paired-only prototype exactly, with identical C drivers and unchanged actual ARM IR. Corrected source passes150 native release tests, Clippy and 16 tests on each ARM target through QEMU; read-only re-review finds no remaining substantive issue. Final binary correspondence and broader timing remain below.
- [x] Run420 broader matched processes on the actual generator, byte-identical to its prototype. Reject: AVX2 cached16 gains only 0.94%, while base cached32 loses6.88% and AVX2 deep GEMV loses2.94%. Restore the previous retained generator; keep the candidate source and successful correctness checks as rejected-experiment evidence.

## Evidence and remaining limits

### Narrow prepared matrices and remaining acceptance work

The constructor always pads to the full packed output width. For AVX-512, one output currently owns64 output lanes of weights and computes the padded lanes as well. The existing ordinary kernel already uses direct dot products, without packing, whenever M is smaller than the target's SIMD lane count, for every batch size.

First compare prepared and ordinary execution on the same retained binary for narrow outputs and record payload sizes. If this confirms the waste, select an owned row-major representation only below that target-derived minimum width, reusing the existing ordinary kernel. Keep the whole output width as its partition unit so batched calls retain row partitioning rather than creating false sharing across adjacent outputs. Packing choice and kernel pointer must be selected together at preparation time; execution remains allocation-free, with one owned weight copy and no per-call packing. The private format is not serialized. Do not generate new ISA-specific arithmetic or change wider matrices.

- [x] Measure180 narrow prepared/ordinary baseline processes. At M1/K1024, AVX-512 owns256 KiB instead of 4 KiB and yields0.001955 TFLOPS versus ordinary0.092752. The storage regression test fails baseline (base profile K1:32 bytes instead of 4) before implementation.
- [x] Add typed representation selection below the LLVM-derived lane count, reusing existing direct dots. Keep columns=m as the narrow partition unit. Eleven targeted tests pass after adding storage, reference/ownership, parallel batches, nonfinite, concurrency and allocation coverage. Single-plan adaptive execution correctly has no learned schedule. Read-only review finds no Critical/Important issues; public API comments corrected.
- [x] Finish initial performance checks (this binary is rejected). Native151 tests, Clippy/fmt and 17 tests per ARM target pass. In408 matched processes, narrow AVX-512 M1/K1024 improves0.001955→0.112599 TFLOPS with 256→4 KiB payload, and AVX2 M7 improves0.019392→0.115552. However, unchanged wide AVX2 arithmetic moved with executable layout and cached16 regresses15.51%; base cached16 loses4.23%. Keep the layout change pending while addressing these controls.

Loop experiment (`../../benchmarks/linear-loop-analysis-2026-11-08/README.md`) and night report (`../../benchmarks/linear-night-2026-11-08/README.md`) contain sources, raw measurements, verification and qualifications. Frozen binaries remain in `/tmp/lumen-kernel-analysis-2026-11-08/` and `/tmp/lumen-kernel-night-2026-11-08/`.

LLVM 21's local `znver5` MCA model reports Zn4 resources; it is not a valid measured Zen 5 roof. `perf` hardware counters remain unavailable, and no system policy was changed. Unbound GEMM variability is unresolved. QEMU verifies tested arithmetic, not native ARM speed. The maximum-performance goal remains active.

### Next GEMV hypothesis: more outputs per input load

Current grouped AVX2 GEMV uses eight accumulators as four output vectors across two K chains. An isolated shared-generator prototype will reuse the same accumulator capacity for eight independent output vectors across four packed bands, with one K chain. This halves input loads and outer passes for a64-output slice and removes chain-reduction adds. Keep the existing smaller-group and padded-tail paths as fallbacks. Derive width from tile/register geometry, retain the packed format, and stream each weight vector into its accumulator to limit live temporaries. Other targets use the same construction; no CPU-name rule or new ARM arithmetic. Verify correctness and emitted code before timing.

The wider-output prototype passes reference checks but fails acceptance in 342 processes. AVX2 cached16 remains4.1% below the accepted baseline. Passing four precomputed band pointers removes the two hot-loop LEAs, but a separate342-process comparison still loses4.87% cached16,6.88% cached32 and 13% tiny K, while streaming loses3–4.5%. Reject both variants. No band-pointer ABI is added to production.

The isolated three-chain prototype passes 11 targeted tests and uses twelve accumulators without hot-loop spills. Reject it after 228 processes: main AVX2 cached16 recovers baseline, but smaller cached cases and four-worker calls lose about 7–8%. Keep the accepted arithmetic.

### Preparation boundary and final narrow-format decision

- [x] Mark `PreparedLinear::new` cold: explicit weight preparation happens before inference. Confirm that it becomes a separate function. This is a code-generation hint, not a change to execution semantics or affinity.
- [x] Compare cold and explicit no-inlining boundaries on original and narrow storage in 360 processes. Reconstructing original unannotated sources reproduces the accepted binary exactly. Reject the explicit no-inlining candidates: AVX2 cached16 loses 14–16%. The cold narrow candidate recovers that case, but smaller/base results vary and require a closer paired comparison.
- [x] Isolate caller SMT overlap with 60 frozen-binary processes, using 15/16 workers and callers30/31 in the same L3 group. The narrow penalty persists when the caller has no active SMT sibling; this does not support an SMT-specific explanation. Cold narrow storage recovers baseline across all four configurations. No pool protocol or affinity change.
- [x] Verify the actual cold candidate again: 151 native release tests, Clippy/formatting, and 17 tests on each ARM target through QEMU pass. Ordinary build matches frozen candidate SHA256 `5349516d9ede145b6293cd46ba049f5084bfe43c639548086756eee59c35c296`. Read-only review finds no substantive correctness issue.
- [x] Finish and independently audit all 896 confirmation processes. Main GEMM/GEMV and narrow gains survive, but reject the candidate: AVX2 batch16 loses a median paired 7.88%, base cached16 3.80%, and base cached32 4.93%, with losses in every block. The batch16 independent medians hide three slower candidate processes. Preserve the paired distributions rather than claiming blanket equivalence.
- [x] Test one shared preparation loop. Row-major is packing width one; its output partition remains M. Wider layout/indexing and all arithmetic remain unchanged. Seven targeted tests and read-only review pass. The 448-process screen rejects the unaligned candidate: cached AVX2 loses 13–14%, while batch16 is close to baseline and narrow gains survive.
- [x] Revisit an explicit 32-byte prepared-entry boundary as a placement control. Frozen AVX2 loops are 76 bytes and instruction-identical; slow variants repeatedly start at 0 modulo32, while faster ones start at 16. This is an association, not proof of a hardware stall cause. Compare original/narrow storage with/without the boundary in mirrored blocks. The rebuilt original aligned control exactly matches the earlier rejected `blocked-tile-bin`; this is explicitly a test of constructor-dependent layout sensitivity, not a new arithmetic claim. All small/base controls still matter.

- [x] Verify shared packing plus cold preparation and the common32-byte entry boundary in the actual ordinary build. Native151 tests, Clippy/formatting and 17 tests on each ARM target pass. SHA25684fe56590c7df6072969d04a9fe11798a34595f824beb4b1e22bdfde66af163c matches the prototype.
- [x] Audit896 broad and 240 focused paired processes. Accept the structural narrow-weight correction with a documented plausible1% AVX2 cached cost. The broad configured 32 loss of 4.97% is1.01% in the focused cohort; this is not proof of equivalence. Configured32 uses 16 active physical workers in these Fixed prepared calls. Narrow M1/K1024 AVX-512 payload256→4 KiB and throughput0.001954→0.112290 TFLOPS are retained. Independent reviewer reconstructed all 1,136 records and 284 block ratios.
- [x] Run a private real-executor row/column-grid diagnostic in 120 processes with one binary and 35 relevant test invocations. Tall AVX2 can gain about 14%, but wider shapes regress substantially. No universal grid change was justified by this diagnostic alone; the later Adaptive follow-up is recorded below.
- [x] Check private rank-indexed completion cells: move caller join, worker panic and completion access together, keeping native wake/placement state and full capacity. Baseline/candidate full release tests pass. Actual diff review and matched performance checks are recorded below; the root pool remains unchanged.

The216-process rank comparison fails performance acceptance: no robust configured 32 AVX gain, with multiple2–4% regressions. Reject without changing the root pool. Actual read-only diff review found no safety blocker for the diagnostic, but additional non-native panic/concurrent-owner tests would be required before shipping it.

### Adaptive prepared GEMM grid investigation

The real-executor grid diagnostic shows a shape-dependent opportunity, so test a bounded alternative inside the existing adaptive policy. Preserve Fixed behavior and the8-candidate/32-learning-call limit. Retain the existing worker-group candidates, then use spare candidate slots to explore distinct legal row/column grids on the full physical group, in powers of two from the current column-first partition. Reuse one actual inference call per observation, selected with the existing conservative comparison. No processor name, extra weight copy, pool-protocol change or runtime allocation. Expose the chosen row partition in diagnostics. Validate numerical partition coverage and bounded learning before any timing. Keep it private until broad complete-call controls pass.


The first adaptive prototype passes156 native tests and Clippy but is rejected after 384 matched processes: tall AVX2 gains14.53%, while two wide AVX-512 processes select a slower grid and cause a median paired4.09% loss. Add a final seed recheck; a deterministic regression test fails before this change and passes afterwards. Four final calls fit the then32-call bound because prepared grids have at most seven candidates. A second384-process pilot preserves tall gains and recovers wide GEMM, but unchanged cached cases still vary.

The selection audit also finds an existing failure outside grid search: in the first cohort, four of eight baseline configured 32 AVX2 processes select eight workers (~0.539 TFLOPS) instead of 16 (~1.07 TFLOPS). Therefore generalize the control to all multi-plan learning, with a separate ninth observation slot. The public maximum becomes36 successful calls, including one final warmup and three seed observations. Fixed, single-plan and settled inference retain their execution behavior. This supersedes the earlier self-imposed32-call design constraint; no user requirement mandated32. Independently review and measure universal seed control alone and with grids using the same private binary, then verify any retained root implementation.


### Final retained adaptive implementation and cutoff

- [x] Compare universal seed control alone and with grids within one private binary: 432 processes confirm targeted tall gains; cross-binary cached losses remain. The 72-process learning-start probe does not reproduce the wrong-worker selection even without prewarm, so it establishes no prewarm cure.
- [x] Implement one `PreparedGrid` for candidate deduplication and execution, up to three distinct row/column alternatives, and a separate ninth seed-observation slot. The public bound is 36 genuine calls. Fixed partitions and the pool protocol stay unchanged. Geometry participates in learning identity; stale hints cannot supply memory bounds.
- [x] Verify arbitrary/tail dimensions, concurrent prepared reuse, output guards, all candidate hints, the eighth ordinary candidate, final-seed ordering and allocation-free learning. The first production candidate passes correctness but shows broader small/cached losses in 544 Adaptive and 216 Fixed processes.
- [x] Remove three runtime divisions from grid construction for one to four input rows. Compare the baseline, intermediate candidate and shorter calculation in 612 Adaptive and 144 Fixed mirrored processes. Retain `11df1c94c0dc2736f59eec671cadc2bca8467ba68f675fbddf8efb417117f258` with the documented tradeoffs: tall AVX2 +13.81%, odd tall +54.18%; inline AVX-512 M1 −9.02% and raw Adaptive cached GEMV −5.05%. Raw streaming choices differ. Fixed base cached aggregate −3.22% uses only two opposite-sign blocks, not a stable recurring penalty estimate.
- [x] Reject the final single-worker tuner bypass after 384 Adaptive and 120 Fixed controls. Although correctness passes and some inline calls improve, AVX2 batch16 loses 7.57% in all four Adaptive blocks. Restore the preceding checked source and frozen executable exactly. The `final-*` experiment directories contain rejected `8ca8…`, not the retained result.
- [x] Record final retained verification: 160 native release CPU tests, 135 without default features, 270 workspace tests, 39 targeted tests per AArch64/ARMv7 QEMU target, formatting and Clippy. An earlier wrong ARM test filter ran zero tests; corrected commands explicitly execute the forced-grid test and are the only ones counted as coverage.
- [x] Independently audit all 756 retained-candidate records and the rejected final Adaptive comparison. Confirm all 35 recorded source hashes and both frozen/target executable hashes. Archive exact accepted source, measurement manifests, raw logs and verification evidence. No new build was needed after restoration.

Heavy CPU work ended by the conservative 06:00 local cutoff. Remaining documentation and source-hash checks do not reopen that window. No commit or push was made. The maximum-performance objective remains open: cached GEMV coordination, learning variability, small-call regressions and native ARM performance still require work. Measured throughput is not proof that the physical ceiling has been reached.

### Post-cutoff selection diagnosis

- [x] Revalidate the retained source/executable against all recorded hashes. Classify the preceding goal turn as progress: it completed the exact restoration, evidence archive and independent audit.
- [x] Reanalyze existing measurements without CPU load. Five cohorts contain 2,356 records; 336 selected GEMV processes form 72 conditional plan groups. Retained odd M1025/K1023 has cap8 in four runs at 0.481183 TFLOPS and cap16 in two at 0.756941. Choice is completely confounded with block, so this is not a counterfactual gain estimate. Raw cached rates likewise separate by streaming choice. Archive reproducible analysis and independent review.
- [x] Trace the warmup contract: the 250 ms duration does not restart bounded learning. Do not infer that warming the selected plan validates the alternatives. The old prewarm probe was a negative reproduction.
- [x] Prepare an isolated patch to snapshot existing learning observations after timing, for both ordinary and prepared calls. No hot-path instrumentation or production edit. Rustfmt checks pass; compilation, tests and execution remain pending. Prepare a concrete 80-process mirrored comparison against frozen accepted `11df…`; its deadline remains expired.
- [ ] After renewed CPU authorization, build/check the private diagnostic and capture learning samples in the five selected cases. Verify reported winners against the actual selection rule. If the bad choices do not reproduce, retain that negative result instead of changing production from speculation. A request for up to five minutes is pending.

This continuation made progress through new evidence and a reviewable diagnostic. It does not establish an optimization or complete the throughput objective. No heavy CPU work resumed after cutoff.

- [x] Prepare and exercise offline decision replay before collecting new traces. All 72 old decisions match the accepted winner rule; all select the original schedule, so this is still a negative reproduction. Fifteen synthetic checks cover non-seed selection and rejection of corrupted traces. New shape/tuner headers are tested synthetically only. Bind the checker to the archived accepted policy hash and reject zero-match validation.

The next continuation has a concrete dependency: the private diagnostic has no compiled executable or measurements, and the five-minute CPU request is still unanswered. The original night deadline remains expired. Further tuning changes need the missing observation evidence; another speculative policy change is not justified by the offline replay.


### User correction of the invented cutoff

The user clarified that they would say when CPU work must stop. The 06:00 cutoff was an assistant assumption, not a user deadline; treating it as expired permission and blocking the goal was an error. The goal is active again. Continue authorized implementation, correctness checks and measurements until the user stops CPU load. Timeouts bound individual experiments only. The prepared private diagnostic is now being compiled and checked before its controlled run; no production policy change has been made from the offline evidence alone.


### Continued diagnostic status

- [x] Build/test the isolated trace diagnostic and collect 80 trace-only controls plus 80 original-binary/GDB controls. Replay all captured choices against the original policy.
- [x] Test optional challenger confirmation privately: 180-process V1 and 612-process V2 cohorts, native correctness and Clippy, exact decision replay. Reject both for retention; no robust performance benefit established. Preserve production source and 36-call bound.
- [x] Capture eight-worker selections with actual observations showing slower 16-worker execution during learning. Do not label these wrong choices based on other-process throughput.
- [ ] Compare forced eight/sixteen-worker prefixes and raw read modes within the same warmed process, including the original learned execution as a control. Keep the checked production implementation untouched until there is actionable evidence.


### Reproduced transient selection failure

- [x] Complete/audit 48 within-process counterfactual probes and32 targets with matched8/16-worker predecessors. Three of four half-preconditioned AVX2 targets select8, while forced16 later improves approximately88%; odd AVX2 gains approximately49%. Full-preconditioned controls all select16. These are actual same-process opportunities after warming, without a claimed hardware cause.
- [x] Reject a private1ms-per-candidate useful-call warmup after48 controlled processes. All warmed AVX2 cached targets still select8; prepared AVX-512 selection also worsens. All32 instrumented decisions replay exactly. At least2331 successful pre-timing warmup calls in every process prove timing excluded learning. The experiment takes531–1325 learning calls versus12–28 in its control and can open additional inline candidates. No production change.
- [ ] Test an initial-seed-only useful-call warmup of20ms, bounded by8192 calls, using the duration that established the faster regime in the forced diagnostic. Keep other observation windows unchanged. Explicitly check convergence before timing. This is an isolated candidate; any production integration needs a coherent public learning bound, simplified state and broad validation.


### Checkpoint: learning experiments rejected, coordination investigation next

- [x] Complete48 controlled and408 broad dynamic-window processes;166 native tests andClippy, exact replay of272 broad decisions. Reject retention because rawcached andpreparedAVX-512cached16 regress against the same-binary control.
- [x] Check longer500µs/256-call windows in48 focused processes using the same executable. They do not consistently improve decisions; archive all32 decision replays. Stop this line of sampling-budget trials rather than escalating another speculative warmup threshold.
- [x] Preserve root35 source hashes and accepted11df executable; no production changes or commits in this continuation. CPU remains authorized until the user stops it; objective active.
- [ ] Prepare an independent private hierarchical-completion prototype from accepted root sources, not from experimental tuning code. Use verified cache-domain membership; unknown/unbound topology retains flat joining. Precompute immutable per-group trees with first-ranked domain leaders and rank-ordered followers.
- [ ] Register pool-owned thread handles before first publication. Preserve original-command wake routing: followers wake their leader, leaders wake the caller. A leader must join every selected follower and clear waiter bits before releasing its own completion, even after its callback panics. Never inspect completed-job data or follower completion cells after that release. Retain native excluded-worker checks, per-pair generation encoding, full-active caller overlap/spin policy, nesting andshutdown semantics.
- [ ] First establish blocking correctness with zero leader spins; include explicit waiter registration/park races, delayedoldwakes, same-CPU andoversubscribed progress, reduced/reordered groups, leaderchanges, excludedmembers, generationwrap, allpanic combinations, concurrentcallers, nesting andconstructorfailure. No spin-only shortcut may substitute for a correct parking protocol.
- [ ] Benchmark flat/tree settings within one frozen binary and against accepted root only after thosechecks. Inspect complete-call GEMV/GEMM andwarmup/learning choices. A smaller number of root completion reads is a hypothesis, not a speedup or proven hardwarecause.

### Hierarchical completion: completed private evaluation

- [x] Build strict optional cache-domain discovery, immutable prefix-safe trees and parked leader/follower completion with original-command wake routing.
- [x] Verify wrap, exclusions, panic ownership, concurrent/nested calls, partial construction, same-CPU progress and actual parking. Address independent review; final178 native release tests and Clippy pass.
- [x] Audit648 processes in102 mirrored blocks, including same-binary flat controls, accepted executable controls and whole-process context-switch counts.
- [x] Reject immediate parking and64-spin versions.4096 iterations remove the large odd-shape parking regression, but do not establish an acceptable overall production improvement. Preserve the prototype and evidence in `docs/benchmarks/linear-night-2026-11-08/hierarchical-join/`; keep accepted root sources/binary unchanged.

The goal is active, with CPU authorization until explicit user stop. No permission request is pending. Next: a separate sampled per-worker timeline diagnostic based on accepted sources, with an explicit instrumentation-overhead control. Do not repeat rejected learning-budget or tree-budget sweeps.


### Worker timeline and prepared tail investigation

- [x] Build/review a private sampled per-worker recorder without changing numerical code, publication or completion.169 release tests with active recording, Clippy and format pass.
- [x] Audit148 processes/3591 timestamped calls across absent/armed/sparse/denser controls and caller23/31 relocation. Independent audit confirms start skew reverses, while the odd AVX2 final worker retains a~2.3us callback. Retain instrumentation caveats and simultaneous LLC/SMT placement change.
- [x] Archive diagnostic source, checks and all measurements under `worker-timeline/`. No trace hooks enter production.
- [x] Review exact code/assembly: final49 useful outputs in64 stored AVX2 columns execute grouped32 + ordinary16 + padded16. Prepare a universal grouped-tail candidate from accepted sources in /tmp/lumen-prepared-group-tail-2026-11-08.
- [x] Add exact guard-page coverage for inputs, padded weights, bias and output before the edit; baseline passes. Candidate161 all-feature release tests and Clippy pass. No allocation/weight format change. Reuse grouped arithmetic only when the existing final storage tiles complete a whole group.
- [ ] Finish mirrored baseline/candidate/accepted performance controls, independent review and broad validation. No production integration until tradeoffs are known.

CPU remains authorized until explicit user stop. Goal active; no pending permission request.


### Prepared remainders: retained V4 and next work

- [x] Finish1576 controlled processes across grouped-tail, independent single-tile, geometry-scoped, short-reduction/early-return and outlined variants. Preserve every cohort and its limitations in `prepared-tail/`.
- [x] Verify171 native CPU/execution tests, guard-page bounds, all-output f64 uneven stripes, expanded nonfinite K4/K5 checks, and12 targeted tests per ARM/QEMU target.
- [x] Retain V4 as an explicit tradeoff: repeated~22–23% oddAVX2 gains,~19–21% M256/K1024 gains, mainGEMM near4.4TFLOPS. Short M49/K2–4 regressions and base/Adaptive variability remain; this is not an overall speedup claim.
- [x] Reject V5:~9% regular AVX2 loss in both Fixed and Adaptive, including equal selectedplans. Restore exact V4 source hashes and b9b825 benchmark binary. No commit/push.
- [ ] Next isolated investigation must start from retained rootV4 and target residual short-call/compiler-layout effects without discarding its long-operation gains. Do not repeat arbitrary alignment, learning-budget or completion-spin sweeps.

Goal active; no permission request pending. CPU authorization has no invented cutoff. Current identity: `accepted-tail-identity.json`; original11df is historical baseline.
