# Implementation roadmap

Build a small correct execution path first, then justify additional mechanisms with measurements. The phase numbers below retain existing code references; they are milestones, not a requirement to finish every optimization before proceeding. In particular, a hybrid synchronization experiment can run alongside the first GPU work.

**Current position:** the configuration/inspect and format/decoder foundation exists. End-to-end inference does not. The Python golden fixtures required below are still missing; Rust differential decoder tests do not replace them. See [architecture.md](architecture.md) for implemented scope and [execution-design.md](execution-design.md) for proposed mechanisms.

## Milestones and acceptance

| Phase | Deliverable | Evidence needed |
| :--- | :--- | :--- |
| 1. Foundation | Types, model component descriptions, placement and `inspect` | Config/selector tests, truthful reporting, and existing dependency-boundary checks |
| 2. Format interoperability | Container reader, LTQ decoder, external fixtures | Bit-for-bit agreement with Python exports and rejection of malformed input; a native quantizer is not required |
| 3. First CPU execution | Minimal prepared schedule, reference/CPU execution, one tiny dense model and then a recurrent hybrid, `lumen run` | Logits against an independent implementation; prefill/decode agreement; declared and observed memory bounds |
| 4. First GPU execution | ABI/build wiring and the kernels needed by those fixtures | CPU/GPU numerical comparison, launch/memory checks, and a small CPU/GPU transfer experiment |
| 5. Hybrid execution | One model with CPU experts and GPU computation, explicit transfers, basic `probe`/`bench` | Placement-invariant results within tolerances; measured crossing latency, overlap and peak memory; correct completion on error/cancel |
| 6. Disk tables | Bounded staging and row fetch | Correct rows, cold/hot I/O latency, bounded buffering and error handling; measure available overlap |
| 7. Measured optimization | Chunked prefill, streaming, repacking/`pack`, selected fusion or graph capture | Differential checks and a reproducible benefit for each adopted optimization |
| 8. Broader model composition | Additional attention, selection, residual and state mechanisms | Reference parity and resource accounting, including shared and derived representations; no family branches in shared mechanisms |
| 9. Serving and speculation | Batching, prefix reuse, drafts, chat and HTTP adapters | Sequence isolation, partial acceptance/rollback, cancellation and capacity limits |
| 10. Media | Preprocessing and encoder/decoder integration | Reference outputs and bounded state for mixed media/text inputs |

Before phase 3 fixes the execution interfaces, exercise a resource description with shared storage, explicitly derived attention inputs, and recurrent state. Count owners once and include scratch. Define step capacities and state ownership early; implementing full serving can wait until phase 9.

## Independent quantizer track

Use ready-made research LTQ exports and tiny dense fixtures to develop inference. Develop native calibration, LDLQ/Viterbi, and bit allocation separately, checking decoder compatibility and quantization quality against the research pipeline. Share model loading and lower-level execution, without making generation scheduling a calibration prerequisite. Import Fisher profiles until native backward implementations have their own checks. See [ltq-design.md](ltq-design.md).

## Decisions that need evidence

- Validate CPU/GPU synchronization and graph capture on the target driver before choosing the flag protocol.
- Measure decoding cost and shared memory/bus contention before assuming bandwidth-bound execution.
- Test cold NVMe latency against the actual first consumer; small logical reads do not imply hidden I/O.
- Budget recurrent history, prefix snapshots and temporary buffers before enabling speculation or more concurrent sequences.
- Treat [performance.md](performance.md) numbers as projections and targets, not completed milestones or required properties of the architecture.

Keep the existing dependency checks while the crate layout is unchanged. Split or combine crates when actual dependencies and ownership justify it, updating checks together with the code. [Architectural boundaries](architecture.md#9-extensibility-rules-and-architectural-boundaries) remain the guide.
