# Performance: Hardware Physical Ceiling Model

**Status: performance model proposal.** `Program::floor`, `probe`, and `bench` are not implemented. The reference-rig numbers below are analytical projections, not measured engine throughput. Overlap, cold physical I/O, compute cost, and resource contention must be validated before treating a projection as a usable bound; see [execution-design.md](execution-design.md).

The target platform for `lumen-rs` is a workstation or desktop PC with a single 16–32 GB consumer GPU, a desktop CPU with dual-channel DDR5, and an NVMe SSD. The target workload is a large MoE or hybrid model (tens to hundreds of billions of parameters) in which:
- attention (`Attention`), linear recurrent layers (`Gated DeltaNet`), shared experts, and a subset of layers reside in **VRAM** and execute on the **GPU**;
- the bulk of routed MoE experts resides in system **RAM** and executes on the **CPU**;
- large lookup tables (n-grams, PLE) reside on **NVMe disk**.

---

## 1. Evaluating Against Physical Hardware Limits ($\eta$)

For every compiled `Program` and hardware profile, the scheduler computes the analytical lower bound on step time $t_{\text{floor}}$ dictated by memory bandwidths and synchronization latencies, while `lumen bench` compares it against the measured execution time $t_{\text{measured}}$.

The ratio
$$\eta = \frac{t_{\text{floor}}}{t_{\text{measured}}} \in (0, 1]$$
measures how close the engine comes to the physical limits of the machine. Optimization is thus divided into two distinct tasks:
- **raising efficiency $\eta$ toward $1.0$** (optimizing CPU/GPU kernels, reducing synchronization overhead, and hiding disk reads);
- **lowering the ceiling $t_{\text{floor}}$ itself** (moving hot layers into VRAM via `--auto`, reducing expert bit widths, or increasing batch size).

---

## 2. Hardware Resources and Measurement (`lumen probe`)

Each hardware resource is characterized by two numbers: its **theoretical spec limit** and its **measured limit** (the sustainable throughput achieved by the `lumen probe` microbenchmark and stored in `hw-profile.json`):

| Resource | Theoretical Spec Source | What `lumen probe` Measures |
| :--- | :--- | :--- |
| **DRAM Read per Node** | Channels $\times$ 8 bytes $\times$ memory clock (SMBIOS) | Streaming reads across all worker threads using the exact access pattern of the MoE expert kernel |
| **VRAM Read** | GPU bus width $\times$ memory clock (CUDA attributes) | Reading a large VRAM buffer with a minimal GEMV-shaped kernel |
| **PCIe Bus (H2D and D2H)** | PCIe generation and lane width (`sysfs`) | Asynchronous copies of varying buffer sizes between `pinned RAM` and VRAM |
| **GPU ↔ CPU Crossing** | *(measured only)* | Full flag ping-pong cycle: `cuStreamWriteValue32` $\to$ CPU spin pool $\to$ CPU write $\to$ `cuStreamWaitValue32` |
| **NVMe Storage** | Rated IOPS and random read latency | Random 4 KiB reads at queue depths (QD) 1, 16, and 64 |
| **Compute Peak (GPU Tensor Cores / CPU int8)** | Rated peak performance per dtype | Dense matrix microbenchmarks for `mma` (GPU) and `vpdpbusd` (CPU) |

Benchmark reports display $\eta$ relative to both the theoretical spec ceiling and the measured hardware ceiling (isolating engine software overhead from hardware memory-controller overhead).

---

## 3. Step Model: Critical-Path Ceiling vs. Resource Ceiling

A segment cannot finish faster than its most heavily loaded resource: $t_{\text{seg}} \ge \max(\text{bytes}_r / \text{BW}_r, \text{FLOP} / \text{peak})$. `Program::floor(&HwProfile)` computes two ceilings:

1. **Critical-Path Ceiling (single sequence, batch = 1):**
   The sum of segment times along the layer dependency chain, accounting for concurrent overlap of independent branches within a layer (such as shared experts on GPU running in parallel with routed experts on CPU), plus the measured latency of every GPU ↔ CPU crossing.
2. **Resource Ceiling (batching limit):**
   The quantity $\max_r (\sum \text{bytes}_r / \text{BW}_r)$ summed across the entire step. Achievable only when overlapping work across multiple sequences in a batch.

`lumen inspect` prints both ceilings broken down by resource and layer, `lumen bench` compares them against runtime measurements, and `--trace` exports a segment timeline (GPU events and CPU spin-pool timestamps).

---

## 4. Reference Rig Ceiling Calculation

- **Reference rig:** RTX 5070 Ti 16 GB (GDDR7, 896 GB/s; PCIe 5.0 x16, ~63 GB/s per direction) + Ryzen 9 9950X (dual-channel DDR5-6000, 96 GB/s theoretical peak) + NVMe SSD.
- **Model and placement ([`examples/example-moe.json`](../../examples/example-moe.json)):** `Qwen3.8-Flash-Next` in LTQ 4.05 bpw; routed experts of 45 out of 48 layers placed in RAM on CPU; 3 layers of experts and all other weights in VRAM on GPU; n-gram table on NVMe disk; context length 262,144 tokens.

| Resource | Data Read per Decode Token | Volume | Spec Ceiling |
| :--- | :--- | :---: | :---: |
| **DRAM (CPU)** | 10 routed experts $\times$ 45 layers at 2.48 MB each | **1.11 GB** | **11.6 ms** |
| **VRAM (GPU weights)** | 36 Gated DeltaNet layers (1.65 GB) + Hyper-Connections (1.28 GB in fp16) + `lm_head` (0.64 GB) + 12 Full Attention layers (0.48 GB) + shared experts (0.24 GB) + routers (0.13 GB) + 3 expert layers (0.07 GB) + PLE projections (0.07 GB) | **4.54 GB** | **5.1 ms** |
| **VRAM (GDN state)** | Read and write $36 \times 3\text{ MB}$ `fp32` state matrices | **0.23 GB** | **0.25 ms** |
| **VRAM (KV at 262k)** | QSA indexer (averaged keys of 65,536 blocks $\times$ 12 layers) + attention ($\le 2048$ positions $\times$ 12 layers) | **~0.25 GB** | **~0.3 ms** |
| **PCIe (GPU ↔ CPU)** | Input and output of 45 CPU segments at ~5 KB each | **0.45 MB** | 90 crossings $\times$ measured latency |
| **NVMe Disk** | 16 n-gram rows at 86 bytes each (`TCQ4X` rings), page-cache misses | **$\le 1.4$ KB logical payload** | Overlap assumed; physical read volume and cold latency need measurement |

- **Critical-Path Ceiling:** $5.6\text{ ms (GPU)} + 11.6\text{ ms (CPU)} - 0.25\text{ ms (shared-expert overlap)} \approx \mathbf{17\text{ ms}}$ plus 90 crossings — **approximately 58 tokens/s** at 262k context.
- **Resource Ceiling:** $11.6\text{ ms}$ (DRAM-bound) — approximately 86 tokens/s (unreachable by a single sequence).
- **Prefill with Expert Streaming (`prefill-compute: "cuda:0"`):** each chunk streams 57 GB of CPU-layer experts over PCIe ($\approx 0.9\text{ s}$ on PCIe 5.0), setting a ceiling of **~1,100 tokens/s** at chunk size 1024 and **~4,500 tokens/s** at chunk size 4096.

### Factors That Lower the Ceiling $t_{\text{floor}}$
- Moving one layer of routed experts from RAM to VRAM shortens the critical path by roughly 0.23 ms/token at the cost of 1.27 GB of VRAM.
- Reducing the bit width of CPU experts reduces DRAM read time linearly.
- Quantizing `Hyper-Connections` matrices (which occupy 1.28 GB in `fp16`, over a quarter of VRAM traffic) directly reduces the GPU portion of the step.

---

## 5. Target Efficiency Goals ($\eta$)

Target $\eta$ values relative to the measured hardware ceiling (`lumen probe`):

| Component | Target $\eta$ |
| :--- | :--- |
| **CPU routed-expert kernel (`lumen-cpu`)** | $\ge 90\%$ of measured DRAM bandwidth |
| **GPU GEMV and Attention kernels** | $\ge 85\%$ of measured VRAM bandwidth |
| **GPU ↔ CPU flag crossing** | Within measured ping-pong $+ 1\text{ }\mu\text{s}$ |
| **Step overhead (graph launch, sampling, events)** | $\le 3\%$ of step time |
| **End-to-end Decode step** | $\eta \ge 0.85$ of critical-path ceiling |
