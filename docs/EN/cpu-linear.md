# The first C GEMV/GEMM module

[Русский](../RU/cpu-linear.md)

Velri now has a standalone **internal CPU linear kernel** in `src/backend/cpu/`.
It computes `Y = X × Wᵀ + bias`, with one input row for GEMV or several for GEMM.
This is a tested building block for the C engine. The graph, worker pool, public
application API and Vulkan backend are not implemented by this module.

## Build and check

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang
cmake --build build
ctest --test-dir build --output-on-failure
```

The build requires CMake 3.21 or later and a C17 compiler. Clang and GCC are
supported by this initial build. There are no downloaded dependencies. The
library needs neither Rust nor LLVM at runtime. Clang uses LLVM during compilation;
GCC also builds the same C source.

`VELRI_CPU_MULTIVERSION=OFF` builds only the baseline variant. With the option
enabled on x86-64, AVX2/FMA and AVX-512F/FMA variants are built too. CPU and OS
support are checked before selecting them. The dispatcher itself uses baseline
instructions. `AUTO` chooses the widest available variant; it is not an autotuner
and does not promise the best speed on every processor.

The same algorithm builds for ARM64. Local verification includes ARM64 under
QEMU; that checks correctness, not native ARM speed. Windows and Android remain
target platforms, not verified builds of this first module. CI covers native
Linux, sanitizers, the baseline-only build and Linux ARM64 under QEMU.

## What “the same result” means

Every output follows this fixed sequence:

1. Start an FP64 accumulator at `-0.0`.
2. In increasing K order, convert the two FP32 inputs to FP64, multiply them,
   and add the product to that accumulator.
3. Add bias in FP64 after the entire dot product, or `+0.0` when bias is absent.
4. Round once to FP32.

The optimized code must match the independent reference **bit for bit**. There
is no numerical tolerance, split K reduction, or FP32 partial accumulator.
Signed zeros, subnormals and overflow are included in the tests. The caller must
use round-to-nearest/ties-to-even, gradual underflow and disabled floating-point
traps. The kernel does not change the caller's floating-point environment;
exception flags are not part of the output contract.

NaN outputs use exactly `0x7fc00000`. This makes their representation consistent
across variants. It explicitly refines the old Rust oracle's unspecified NaN
payload behavior; Rust does not promise portable or deterministic NaN result
bits. We do **not** claim to preserve those old payloads.
[Rust floating-point documentation](https://doc.rust-lang.org/std/primitive.f32.html#nan-bit-patterns).

The ordinary, row-major C reference is in `tests/cpu/reference.c`. It shares no
packing or arithmetic helpers with the fast implementation. A frozen binary
fixture from the unmodified historical Rust reference checks both implementations;
its provenance and hash are in `tests/cpu/fixtures/README.md`.

The previous FP32 fast kernels used tolerance checks and do not satisfy this
new exactness requirement. Their historical TFLOPS values cannot be presented
as performance of this C implementation.

## How the kernel gets faster

The common source is `linear_kernel.c`. CMake compiles it with different tile
sizes and target instructions, then generates the variant table. There is no
separate ARM algorithm and no processor-model name in the arithmetic code.

SIMD lanes calculate different output features. Each output retains its original
K order. Fully unrolling the short column loop lets the compiler hold sums in
registers instead of repeatedly writing them to the stack.

FMA can combine the FP64 multiply and addition here without changing finite
results: two FP32 values have at most 24 significant bits each, so their product
needs at most 48 bits and fits exactly in FP64's 53 bits. Its exponent also fits.
The reference keeps separate multiply and add; the optimized path may fuse them.
This argument does not allow reassociating the sum or using FP32 FMA instead.

GEMM processes two input rows together, sharing weight loads and conversion to
FP64. It finishes the rows of one weight tile before moving to the next tile,
improving cache reuse. A narrower row-pair tile keeps AVX2 accumulator pressure
within its register budget. GEMV uses one row and more independent output sums.

## Preparation and memory

`linear.h` defines three operations:

| Operation | Responsibility |
| :--- | :--- |
| `velri_linear_query` | Validate M/K, select a supported variant and tile, report packed size |
| `velri_linear_pack` | Rearrange caller-provided W once into caller-provided storage |
| `velri_linear_run` | Validate views and execute using the existing packing |

W has M rows of K FP32 values. The packed order is `[column tile][K][column
within tile]`. For tile width C, element `(c, i)` is stored at
`(c / C) * C * K + i * C + c % C`. Unused columns in the last tile contain zero.
The required capacity is `ceil(M / C) * C * K` floats.

The selected tile fits at least once in M. Thus packing uses less than twice
the actual weight count; M=1 uses exactly K floats. The plan reports all padding.
Weights stay FP32: there is no FP64 weight copy. The original W may be released
after packing, but both allocations count while they are simultaneously alive.

The caller owns the plan, packing, inputs and outputs. A plan must come from
`query` and must not be modified. A packing belongs to the plan that produced
it; equal byte counts alone do not make layouts interchangeable. Plans and
packed weights may be shared by concurrent calls with separate outputs.
Execution allocates no heap memory and does not repack weights.

M and K must be positive; zero input rows are allowed. Counts and strides are
in float elements. Strides must cover a logical row, including when rows=0.
Only natural float alignment is required. Read-only views may overlap each
other. Output must not overlap an input view, its row gaps, or the plan.
Packing requires separate source and destination regions. Size overflow,
insufficient capacities and forbidden overlap are rejected before any write.

This internal API does not create threads. A future scheduler can partition
validated complete column tiles into disjoint output ranges. The direct kernel
function requires those checks to have already succeeded; ordinary callers use
`velri_linear_run`. The common backend interface remains a separate design.

## Measure without losing the correctness check

```sh
cmake -S . -B build -DVELRI_BUILD_BENCHMARK=ON
cmake --build build
./build/bench_linear auto 1 1024 1024 1000 21
```

Arguments are variant, input rows, M, K, repetitions per sample and sample count.
The optional final argument is an independently justified **arithmetic ceiling
in TFLOPS for this single-thread run**. If it is absent, the tool reports the
ceiling as unknown rather than inventing a CPU frequency or memory limit.

The tool checks every output before timing and after every sample. It reports
preparation separately and includes view validation in call time. TFLOPS counts
the useful `2 × rows × M × K` operations, excluding padded work and conversions.
Repeated fixed weights may be cached. This tool does not measure a worker pool
or a model graph; the [measurement rules](performance.md) still apply.

Address/undefined-behavior checks can be enabled with `VELRI_SANITIZE=ON` and a
compiler with the corresponding runtimes installed. Tests also use guard pages
and concurrent shared-weight calls on POSIX. A passing arithmetic test alone
does not establish correct memory access or thread-safe ownership.

## Exact-kernel measurements, 10 November 2026

A matched comparison used the new C arithmetic source inside the archived Rust
worker-pool harness on Ryzen 9 9950X. It measures complete prepared linear calls,
not the future C runtime. Each configuration has three alternating process pairs,
21 samples per process and every output checked bitwise. The baseline is the first
**exact FP64-accumulating C version**, not the faster historical FP32 implementation.

| Workload, 16 workers | First exact version, TFLOPS | Final version, TFLOPS | FP64 arithmetic reference, TFLOPS |
| :--- | ---: | ---: | ---: |
| Cached GEMV, M=1024 K=1024, AVX-512 | 0.460 | 0.773 | 2.425 |
| Odd GEMV, M=1025 K=1023, AVX2 | 0.173 | 0.538 | 1.287 |
| GEMM, rows=64 M=512 K=512, AVX-512 | 0.497 | 1.133 | 2.448 |
| GEMM, rows=256 M=4096 K=4096, AVX-512 | 0.424 | 1.419 | 2.258 |
| Streaming GEMV, M=8192 K=8192, AVX-512 | 0.0433 | 0.0427 | RAM limit unknown |

The arithmetic references use CPU-wide APERF/MPERF observations bracketing the
final timed loops: 32 FP64 FLOP/cycle/core for AVX-512, 16 for AVX2. Counter
snapshots are staggered and include OS activity. These are conditional FMA
references, not attainable ceilings for a loop containing conversions, loads and
synchronization. The streaming case is slightly slower and has no established
physical RAM ceiling; its unrelated arithmetic reference is about 2.86 TFLOPS.

The retained AVX2 width change improves cached GEMV but loses roughly 2–3% on
the measured GEMM controls. In the final comparison, eight-worker cached GEMV is
0.539 TFLOPS against a 1.359 TFLOPS arithmetic reference. There is no universal
claim that 16 workers or the widest ISA is optimal.

A separate direct call to the shipped C API, pinned to one core with M=64 K=1024,
measured 0.0899 TFLOPS. The supplied 0.1824 TFLOPS nominal arithmetic reference
assumes 5.7 GHz on that core; it is not a measured sustained clock. This check
includes API validation and no worker pool.

Software sampling confirms substantial worker coordination and waiting CPU time
in the archived short-call harness. Sampling percentages are not latency
fractions. Hardware event profiling was unavailable (`perf_event_paranoid=4`).
A read-only XOR control is retained separately; it is not treated as a physical
memory ceiling. Raw data, source snapshots, disassembly, profiles, executable
hashes and rejected/confounded comparisons remain local under
`docs/benchmarks/linear-bottlenecks-2026-11-10/`, excluded from Git.
