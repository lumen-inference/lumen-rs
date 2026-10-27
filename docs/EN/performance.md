# lumen-rs — Performance: the Physical Ceiling

Oct 27, 2026 · @Vemloir

The typical machine has one consumer GPU with 16–32 GB, a desktop CPU with dual-channel DDR5, and NVMe. The typical model is an MoE with tens to hundreds of billions of parameters. Experts run almost entirely on the CPU; attention, shared experts, and dense layers run on the GPU; large lookup tables stay on disk.

The engine's yardstick is the hardware, not other engines. For each `Program` and machine, the scheduler computes a lower bound on step time from the physical limits of the resources, and `lumen bench` puts the measured time next to it. The ratio η = t\_floor / t\_measured shows how much the engine loses beyond what physics forbids. The work splits in two: raising η toward one, and lowering the ceiling itself — through placement, bit width, and batching.

**Resources and their ceilings.** Each resource has two numbers: the spec value and the measured one.

| Resource | Spec — source | Measurement — `lumen probe` |
| --- | --- | --- |
| DRAM reads per node | channels × 8 bytes × transfer rate (SMBIOS) | streaming reads with the same pattern as the expert kernel, by all of the node's workers |
| VRAM reads | bus width × memory clock (CUDA attributes) | reading a large buffer with a GEMV-shaped kernel |
| PCIe, H2D and D2H | link generation and width (sysfs) | copies from pinned RAM of various sizes |
| GPU ↔ CPU crossing | — | ping-pong on flags: `WriteValue32` → CPU spin → CPU write → `WaitValue32` |
| NVMe | random-read IOPS and latency | random 4 KiB reads at queue depths 1, 16, and 64 |
| Tensor cores, CPU int8 | peak throughput per dtype | `mma` and `vpdpbusd` microbenchmarks |

The spec value is the physical ceiling; the measurement is what the hardware actually gives any program (DRAM gives nobody 100% of spec). The headline metric is η against spec; η against the measurement separates the engine's own losses from the hardware's. Latencies have no spec value, so their ceiling is the measurement. The `lumen probe` profile is saved to `hw-profile.json`, which `inspect`, `bench`, and auto-placement read. Spec numbers that cannot be read from a device are entered into the profile by hand.

**Step model.** A segment cannot finish faster than its busiest resource: t\_seg ≥ max(bytes\_r / BW\_r, FLOP / peak). Segments of one token are linked by dependencies: layer ℓ+1 waits for layer ℓ, and a layer's CPU part waits for the GPU part before it. So there are two ceilings:

- **Critical path** — the sum of segments along the dependency chain, with overlap where branches are independent (shared experts on the GPU in parallel with routed experts on the CPU), plus the measured latency of every GPU ↔ CPU crossing. This is the ceiling for a single sequence.
- **Resources** — max\_r(Σ bytes\_r / BW\_r) over the whole step. Only overlapping the work of different tokens (micro-batches in a batch) reaches it; a single sequence cannot.

`Program::floor(&HwProfile)` returns both ceilings broken down by resource and layer; `lumen inspect` prints the breakdown, `lumen bench` puts measurements next to it, and `--trace` writes a timeline of segments (GPU events and spin-pool timestamps) that shows where η is lost.

**Reference rig.** RTX 5070 Ti 16 GB (GDDR7, 896 GB/s; PCIe 5.0 x16, ~63 GB/s each way) + Ryzen 9 9950X (dual-channel DDR5: 96 GB/s at DDR5-6000; `probe` fills in the actual transfer rate) + NVMe. The model is Qwen3.8-Flash-Next in LTQ at 4.05 bits per weight, routed experts of 45 of 48 layers on the CPU, the n-gram table on disk, KV in bf16. Bytes per decode token come from the `quantization_config.json` of an EXL3 quant of the same bit width (SC 4.05) — at equal bit width the volumes are the same:

| Resource | What is read per token | Volume | Spec ceiling |
| --- | --- | --- | --- |
| DRAM | 10 routed experts × 45 layers at 2.48 MB | 1.11 GB | 11.6 ms |
| VRAM, weights | DeltaNet 1.65 GB; hyper-connections 1.28 GB (fp16); lm\_head 0.64 GB; full attention 0.48 GB; shared experts 0.24 GB; routers 0.13 GB; routed experts of 3 layers 0.07 GB; PLE 0.07 GB | 4.54 GB | 5.1 ms |
| VRAM, state | DeltaNet: read and write of 36 × 3 MB in fp32 | 0.23 GB | 0.25 ms |
| VRAM, KV at 262k | indexer: pooled keys of 65,536 blocks × 12 layers; attention: ≤ 2048 positions × 12 layers | ~0.25 GB | ~0.3 ms |
| PCIe | input and output of 45 CPU parts at 5 KB | 0.45 MB | 90 crossings × measured latency |
| NVMe | 16 n-gram rows of 86 bytes (TCQ4X rings), page-cache misses only | ≤ 1.4 KB | hidden behind the first layers |

- Critical-path ceiling: 5.6 ms GPU + 11.6 ms CPU − 0.25 ms of overlap with shared experts ≈ 17 ms plus 90 crossings — about 58 tok/s at a 262k context.
- Resource ceiling: 11.6 ms (DRAM) — about 86 tok/s; out of reach for a single sequence.
- Prefill with expert streaming (`prefill-compute`): every chunk pushes all experts of the CPU layers over PCIe — 57 GB, ≈ 0.9 s. That caps throughput at ~1100 tok/s with 1024-token chunks and ~4500 tok/s with 4096-token chunks, so chunks in this mode must be large. The tensor-core ceiling adds to it; `probe` provides that one.

**What moves the ceiling itself**

- Moving one layer of routed experts to VRAM shortens the critical path by about 0.23 ms per token and costs 1.27 GB of VRAM; auto-placement balances this against context.
- The bit width of CPU experts shrinks the DRAM part linearly.
- Hyper-connections in fp16 are 1.28 GB, more than a quarter of VRAM traffic: quantizing these matrices is the cheapest way to lower the GPU part.
- Speculative decoding amortizes the GPU part over accepted tokens, but with top-10 of 512 routed experts barely overlap between tokens, so the DRAM part grows almost linearly with k+1.
- Batching and micro-batches approach the resource ceiling.

**η targets** (against the measured ceiling; η against spec is the headline number in the report):

| What | Target |
| --- | --- |
| Routed-expert kernel on the CPU | ≥ 90% of measured DRAM bandwidth |
| GEMV and attention on the GPU | ≥ 85% of measured VRAM bandwidth |
| GPU ↔ CPU crossing | no longer than the measured ping-pong + 1 µs |
| Step overhead (graph launch, sampling, events) | ≤ 3% of the step |
| Whole decode step | η ≥ 0.85 of the critical-path ceiling |

Hence the order of work: the CPU expert kernel at the DRAM limit ([kernels.md](kernels.md)); the per-layer synchronization path — flags, spin pool, overlap ([sections 3 and 5 of architecture.md](architecture.md#3-memory-storage-locations-and-access-modes)); `Fetch` with prefetching so the disk stays off the critical path ([section 3 of architecture.md](architecture.md#3-memory-storage-locations-and-access-modes)); prefill with expert streaming ([section 5 of architecture.md](architecture.md#5-scheduler-and-execution-pipeline)); a hot-expert cache in VRAM — after measuring the routing distribution on real traces.

## Sources

- [Qwen/Qwen3.8-Flash-Next: config.json, model.safetensors.index.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Vemloir/Qwen3.8-Flash-Next-sc-exl3: an EXL3 quant of the same bit width — tensor sizes for the ceiling estimate, self-calibration data](https://huggingface.co/Vemloir/Qwen3.8-Flash-Next-sc-exl3)
