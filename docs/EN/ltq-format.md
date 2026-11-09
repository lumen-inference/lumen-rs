# LTQ v1 Quantization Format Specification (Lumen Trellis Quant)

[Русский](../RU/ltq-format.md)

This is the precise reference for implementing LTQ readers and writers. For an introduction to the engine, start with the [architecture](architecture.md); terms are defined in the [glossary](glossary.md). The formulas, field sizes, and bit order here define file compatibility and must not be replaced by approximate descriptions.

**LTQ** (*Lumen Trellis Quant*) is the weight quantization format used by Velri. It is designed to compress neural network weight matrices down to **2–5 bits per weight** and package the entire model (metadata, tokenizer, and weights) into a single self-contained `.vtq` container file. The `.vtq` extension changes the file name only: the LTQ v1 magic (`LTQ\0`), version, tensor keys (`.ltq.data` and `.ltq_rows.data`) and byte layout stay unchanged. The existing reader and writer do not depend on the file extension.

This document specifies the mathematical reconstruction model, the $16 \times 16$ tile structure, the format menu, the ring-encoded row lookup tables, and the on-disk binary container layout. The reference encoder and decoder implementation is in [`vtq-research/src/ltq_format.py`](https://github.com/lumen-inference/vtq-research/blob/main/src/ltq_format.py), and the automated bit-exact self-test results (512 Gaussian tiles $\times$ 19 menu formats and 256 rings at $L = 160$ $\times$ 17 ring-eligible formats — 0 mismatches) are recorded in [`results/ltq_format_selftest.json`](https://github.com/lumen-inference/vtq-research/blob/main/results/ltq_format_selftest.json). Quality comparisons against EXL3 and GGUF and the design of the quantizer are covered in [ltq-design.md](ltq-design.md).

---

## 1. Core Principles

When compressing weight matrices from 16-bit floating-point (`bf16`) down to 3–4 bits per element, naive independent rounding of each weight degrades model accuracy for two main reasons:
1. **Outliers:** If a group of weights contains a single large-magnitude value, a uniform quantization grid stretches to accommodate that maximum, collapsing smaller weights to zero or merging them into a single quantization level.
2. **Limited codebook resolution:** Independent $K$-bit quantization provides only $2^K$ levels per weight (e.g., 8 levels at 3 bits), which is too coarse to capture smooth weight distributions accurately.

LTQ addresses these limitations through three mechanisms:

1. **Input Hadamard Transform (`suh` and $H_{256}$) for Outlier Suppression.**
   Before quantization, the weights undergo coordinated channel scaling, sign changes, and a blockwise 256-point Walsh–Hadamard transform ($H_{256}$). The orthogonal rotation spreads the energy of isolated outliers across all 256 coordinates in each block, making the distribution closer to Gaussian. The stored `suh` vector defines the signs and scales applied to inference inputs: Section 2 specifies $\tilde{x} = H_{256}(\text{suh} \odot x)$, performed once per input vector. The weight-side transform must preserve that section's reconstruction relation.
2. **Trellis-Coded Quantization (TCQ) on $16 \times 16$ Tiles.**
   Each matrix is partitioned into $16 \times 16$ tiles (256 weights). Instead of independent per-weight quantization, LTQ uses Ungerboeck Trellis-Coded Quantization: the decoder maintains a 10-bit sliding window of the most recent code bits. At each position, 1 code bit (`codes`) shifts into this 10-bit state, bit parities under format bitmasks select a subset (bucket) in the lookup table `lut`, and the remaining $K - 1$ uncoded bits select the specific level within that subset. During quantization, the Viterbi algorithm searches the trellis for the optimal bit sequence; during inference, decoding reduces to bitwise operations and table lookups.
3. **Fractional Bitrates (`D2`, `D4`) and Per-Slice Format Assignment (`slices`).**
   Coupling the lowest uncoded bits across pairs (`D2`) or quads (`D4`) of weights yields fractional bitrates of $K - 0.5$ and $K - 0.25$ bits per weight. Furthermore, each matrix is partitioned horizontally into row slices (`slices`), allowing the quantizer to assign different formats from the menu to different row bands according to their error sensitivity.

---

## 2. Linear Layer and Reconstruction Formula

A linear layer computes $y = W \cdot x + b$, where the weight matrix $W$ has logical shape $[M, K]$. During quantization, $K$ is zero-padded to a multiple of 256 ($K_{\text{pad}}$) and $M$ is padded to a multiple of 16 ($M_{\text{pad}}$).

$$\tilde{x} = H_{256}(\text{suh} \odot x) \quad \text{(input rotation, performed once per input vector)}$$

$$y[m] = \sum_{b} d[m, b] \cdot \sum_{k \in \text{block } b} q[m, k] \cdot \tilde{x}[k] + \text{bias}[m]$$

$$d[m, b] = d_{\text{row}}[m] \cdot 2^{e[m, b] / 4}$$

Where:
- **`suh`** — `fp16` vector of length $K_{\text{pad}}$ containing random signs ($\pm 1$) and per-channel input scales.
- **$H_{256}$** — block-normalized Walsh–Hadamard Transform: each 256-element subvector is multiplied by the $256 \times 256$ Hadamard matrix scaled by $1/16$.
- **$b = \lfloor k / 256 \rfloor$** — index of the 256-column block along the $K$ axis.
- **`d_row[m]`** — base `fp16` row scale for row $m$.
- **`e[m, b]`** — signed integer block exponent (`int8`, values in $[-16, 15]$) for row $m$ and 256-block $b$.
- **$q[m, k]$** — decoded weight value from the corresponding $16 \times 16$ tile (Section 3).

The output vector is not rotated, as output rotation showed no accuracy benefit in ablation studies.

---

## 3. $16 \times 16$ Tile Structure

A tile covers 16 rows ($16i \dots 16i+15$) and 16 columns ($16 \cdot kb \dots 16 \cdot kb + 15$), where $kb \in [0, K_{\text{pad}}/16 - 1]$ (with 256-block index $b = \lfloor kb / 16 \rfloor$). The 256 positions inside a tile are indexed in row-major order: $t = 16r + c \in [0, 255]$, where $r \in [0, 15]$ is the tile row and $c \in [0, 15]$ is the tile column.

### Trellis and State
Each position $t$ has one code bit $c_t \in \{0, 1\}$. The trellis state $s_t$ is a window of the 10 most recent code bits. The trellis is tail-biting (closed modulo 256): the beginning of the tile wraps around to its end, so no separate initial state is stored.

$$s_t = \sum_{j=0}^{9} c_{(t - j) \bmod 256} \cdot 2^j \quad \text{(bit 0 is the position's own code bit } c_t\text{)}$$

### Subset Index (`label`)
The subset label $\text{label}_t$ selects the codebook subset in `lut`. It is formed from the lowest bit of $s_t$ and the parities (`parity` = `popcnt(x) & 1`) of the remaining 9 history bits masked by the format's polynomial masks $m_0, m_1, m_2$:

$$\text{label}_t = (s_t \ \& \ 1) \ \mid \ (\text{parity}((s_t \gg 1) \ \& \ m_0) \ll 1) \ \mid \ (\text{parity}((s_t \gg 1) \ \& \ m_1) \ll 2) \ \mid \ (\text{parity}((s_t \gg 1) \ \& \ m_2) \ll 3)$$

The base mask $m_0 = 359$ is present in all formats and defines 4 subsets. Optional masks $m_1, m_2$ are used in lower-bitrate formats (Section 5) to expand the codebook to 8 or 16 subsets.

### Value Reconstruction ($q_t$)
The $K - 1$ uncoded bits $u_t$ select a level within subset $\text{label}_t$, while the local gain index $g$ scales a subgroup of $\text{sub}$ consecutive positions ($\text{sub} = 16$ for a single tile row, $\text{sub} = 32$ for a pair of tile rows):

$$q_t = \text{gain}[g_{\lfloor t / \text{sub} \rfloor}] \cdot \text{lut}[\text{label}_t \cdot 2^{K - 1} + u_t]$$

Both `lut` (`float32`, $N_{\text{labels}} \cdot 2^{K-1}$ entries) and `gain` (`float32`, $2^h$ entries) are stored explicitly in the `.vtq` file header. This avoids differences in rounding $2^{(g - (2^h - 1)/2) \cdot \text{step}}$ across CPU and GPU architectures.

### Fractional Modes (`Frac`)
Fractional modes constrain the lowest uncoded bits ($u_t \ \& \ 1$) of neighboring positions:

| Mode | Constraint on Lowest Uncoded Bits | Effective Rate |
| :--- | :--- | :--- |
| **`none`** | Unconstrained (1 bit per position) | $K$ bits/weight |
| **`d2`** | In each pair $(2q, 2q+1)$, the lowest bits are equal | $K - 0.5$ bits/weight |
| **`d4`** | In each quad $(4q \dots 4q+3)$, the modulo-2 sum of lowest bits is even | $K - 0.25$ bits/weight |

The quantizer enforces this constraint during Viterbi search, and the decoder reconstructs the elided bits as defined in Section 4.

---

## 4. Tile Bit Fields

All fields of a tile are packed as arrays of 16-bit little-endian words `u16` (bit $j$ of word $w$ corresponds to stream element $16w + j$):

| Field | Number of `u16` Words | Contents |
| :--- | :--- | :--- |
| **`codes`** | $16$ | Code bits $c_t$: word $r$ holds the 16 bits of tile row $r$, bit $c$ is position $16r + c$. |
| **`planes`** | $16 \cdot (K - 2)$ | Uncoded bit planes $1 \dots K-2$, 16 `u16` words per bit plane. Omitted when $K = 2$. |
| **`lsb`** | $16$ / $8$ / $12$ | Lowest uncoded bit (bit 0 of $u_t$): 256 bits (16 words) in `none` mode; 128 bits (8 words) in `d2` mode (bit $q$ shared by pair $2q, 2q+1$); 192 bits (12 words) in `d4` mode (bits $3q, 3q+1, 3q+2$ store positions $4q, 4q+1, 4q+2$, while position $4q+3$ is their `XOR`). |
| **`gains`** | $\lceil (256 / \text{sub}) \cdot h / 16 \rceil$ | Packed $h$-bit gain indices $g$. Omitted when $h = 0$. |

Each tile occupies $2 \cdot (16 + 16(K - 2) + |\text{lsb}| + |\text{gains}|)$ bytes — ranging from 64 bytes (`TCQ2X0`) to 168 bytes (`TCQ5X`).

---

## 5. LTQ v1 Format Menu

The v1 menu defines 19 formats trained via Lloyd–Viterbi optimization on a Gaussian source (`vtq-research/src/train_tcq_x.py`):

| Format | $K$ | Mode | Subsets | Extra Masks | $h$ | Gain Step | `sub` | `lut` Entries | Bytes / Tile | Bits / Weight ($K=4096$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TCQ2X0** | 2 | — | 16 | 58, 299 | 0 | — | 16 | 32 | 64 | **2.035** |
| **TCQ2X** | 2 | — | 16 | 58, 299 | 1 | 0.40 | 16 | 32 | 66 | **2.098** |
| **TCQ25X** | 3 | D2 | 8 | 241 | 1 | 0.40 | 16 | 32 | 82 | **2.598** |
| **TCQ275X** | 3 | D4 | 8 | 241 | 1 | 0.40 | 16 | 32 | 90 | **2.848** |
| **TCQ275X2** | 3 | D4 | 8 | 241 | 2 | 0.22 | 16 | 32 | 92 | **2.910** |
| **TCQ3X1** | 3 | — | 8 | 241 | 1 | 0.40 | 16 | 32 | 98 | **3.098** |
| **TCQ3X** | 3 | — | 8 | 241 | 2 | 0.22 | 16 | 32 | 100 | **3.160** |
| **TCQ35X1** | 4 | D2 | 4 | — | 1 | 0.40 | 16 | 32 | 114 | **3.598** |
| **TCQ35X** | 4 | D2 | 4 | — | 2 | 0.22 | 16 | 32 | 116 | **3.660** |
| **TCQ375X** | 4 | D4 | 4 | — | 2 | 0.22 | 16 | 32 | 124 | **3.910** |
| **TCQ4X1** | 4 | — | 4 | — | 1 | 0.40 | 16 | 32 | 130 | **4.098** |
| **TCQ4W2** | 4 | — | 4 | — | 2 | 0.22 | 32 | 32 | 130 | **4.098** |
| **TCQ4W3** | 4 | — | 4 | — | 3 | 0.10 | 32 | 32 | 132 | **4.160** |
| **TCQ4X2** | 4 | — | 4 | — | 2 | 0.22 | 16 | 32 | 132 | **4.160** |
| **TCQ4X** | 4 | — | 4 | — | 3 | 0.15 | 16 | 32 | 134 | **4.223** |
| **TCQ45X** | 5 | D2 | 4 | — | 2 | 0.22 | 16 | 64 | 148 | **4.660** |
| **TCQ475X** | 5 | D4 | 4 | — | 2 | 0.22 | 16 | 64 | 156 | **4.910** |
| **TCQ5X3** | 5 | — | 4 | — | 3 | 0.15 | 16 | 64 | 166 | **5.223** |
| **TCQ5X** | 5 | — | 4 | — | 4 | 0.06 | 16 | 64 | 168 | **5.285** |

Effective bits per weight equals $8 \cdot (\text{tile bytes}) / 256 + 8/256\text{ (exponent } e\text{)} + 16/K\text{ (scale } d_{\text{row}}\text{)}$. The `suh` vector adds $16/M$ bits/weight. Format IDs in `{p}.ltq.slices` index into the container's `formats` header list, allowing new formats to be introduced as data.

---

## 6. Row Slices and the `.vtq` Container

### Row Slices (`slices`)
Each matrix is partitioned horizontally into contiguous row ranges that are multiples of 16 rows (`slices`). Dynamic programming assigns each slice its own format: attention matrices typically use 256-row slices, MLP matrices use 512-row slices, and each MoE expert matrix forms a single slice. The `suh` vector and `d_row`, `e` scale arrays are shared across the full matrix.

### Matrix Tensors in the Container
Each quantized matrix `{p}` (the original tensor name without `.weight`) is stored as five tensors (plus optional `{p}.bias`):

| Tensor Name | Dtype & Shape | Contents |
| :--- | :--- | :--- |
| **`{p}.ltq.slices`** | `int64`, $[S, 4]$ | Descriptor for each of the $S$ slices: `[start_row, end_row, format_id, byte_offset]` |
| **`{p}.ltq.data`** | `uint8`, $[\text{bytes}]$ | Concatenated slice payload bytes |
| **`{p}.ltq.d_row`** | `fp16`, $[M]$ | Per-row scales |
| **`{p}.ltq.e`** | `int8`, $[M, K/256]$ | Per-block scale exponents in $[-16, 15]$ |
| **`{p}.ltq.suh`** | `fp16`, $[K]$ | Input sign and scale vector |

Inside `{p}.ltq.data`, each slice consists of four consecutive 64-byte aligned arrays (Structure of Arrays, SoA):
```text
codes   [T][16]        u16
planes  [T][K - 2][16] u16
lsb     [T][|lsb|]     u16
gains   [T][|gains|]   u16
```
where $T = (\text{slice\_rows} / 16) \cdot (K_{\text{pad}} / 16)$ is the number of tiles in the slice, ordered by 16-row bands and then by column block $kb$.

### Binary File Layout (`.vtq`, little-endian)

```text
Offset    Size    Description
-----------------------------------------------------------------------------------
0         4       Magic bytes "LTQ\0"
4         4       u32: container version (1)
8         8       u64: JSON header length H in bytes
16        H       UTF-8 JSON header padded with spaces so that 16 + H is a multiple of 4096
16 + H    ...     Tensor payload area, starts on a 4096-byte boundary. Tensor offsets are relative
                  to its start: 64-byte aligned (4096-byte aligned for RowTable data), so they are
                  aligned in the file and in an mmap of it as well
```

JSON header schema:
```json
{
  "config": { "...": "original model config.json" },
  "generation_config": { "...": "..." },
  "tokenizer": { "...": "tokenizer.json contents" },
  "tokenizer_config": { "...": "..." },
  "chat_template": "...",
  "processor_config": { "...": "present for multimodal models" },
  "quantization": {
    "quant_method": "ltq",
    "version": 1,
    "formats": [
      {
        "name": "TCQ3X", "K": 3, "frac": "none", "masks": [359, 241], "h": 2, "sub": 16,
        "lut": [ "... float32 ..." ],
        "gain": [ "... float32 ..." ]
      }
    ],
    "row_tables": { "{p}": { "format": 0, "ring": 160, "hadamard": 32 } },
    "modules_not_quantized": ["lm_head", "model.embed_tokens"],
    "bits_per_weight": 3.0,
    "calibration": { "...": "..." }
  },
  "tensors": {
    "{name}": { "dtype": "u8", "shape": [ "..." ], "offset": 0 }
  }
}
```

---

## 7. Reference Tile Decoder (`decode_tile`)

The following Rust example, retained from the `lumen-ltq` reference, defines the exact semantics of decoding a $16 \times 16$ tile. It is not an implementation of the new C API. Optimized CPU and GPU kernels ([kernels.md](kernels.md)) must match this reference output bit-for-bit when evaluating multiple weights in parallel:

```rust
/// Decodes 256 tile values without row scale:
/// q[t] = gain[g] * lut[label * 2^(K - 1) + u], where t = 16 * r + c.
fn decode_tile(
    f: &LtqFormat,
    codes: &[u16; 16],
    planes: &[[u16; 16]],
    lsb: &[u16],
    gains: &[u16],
) -> [f32; 256] {
    let bit = |w: &[u16], i: usize| ((w[i >> 4] >> (i & 15)) & 1) as u32;
    let mut q = [0f32; 256];

    for t in 0..256 {
        let mut s = 0u32; // 10-bit code window, tail-biting modulo 256
        for j in 0..10 {
            s |= bit(codes, (t + 256 - j) % 256) << j;
        }

        let mut label = (s & 1) | (((s >> 1) & f.masks[0]).count_ones() & 1) << 1;
        for (i, m) in f.masks[1..].iter().enumerate() {
            label |= (((s >> 1) & m).count_ones() & 1) << (2 + i);
        }

        let mut u = match f.frac {
            Frac::None => bit(lsb, t),
            Frac::D2 => bit(lsb, t / 2),
            Frac::D4 if t % 4 < 3 => bit(lsb, 3 * (t / 4) + t % 4),
            Frac::D4 => {
                bit(lsb, 3 * (t / 4))
                    ^ bit(lsb, 3 * (t / 4) + 1)
                    ^ bit(lsb, 3 * (t / 4) + 2)
            }
        };

        for p in 1..f.k - 1 {
            u |= bit(&planes[p - 1], t) << p;
        }

        let mut g = 0usize;
        for i in 0..f.h {
            g |= (bit(gains, (t / f.sub) * f.h + i) as usize) << i;
        }
        q[t] = f.gain[g] * f.lut[((label as usize) << (f.k - 1)) | u as usize];
    }
    q
}
```

---

## 8. Quantizer Contract

1. **Reconstruction fidelity:** The reconstructed weights used by the quantizer for error feedback and subsequent layer calibration must match the decoded output of the serialized fields (`d_row` and `suh` in `fp16`, `e` in `int8`, and the `lut`, `gain` tables).
2. **Tail-biting Viterbi search:** Every tile is quantized via two-pass tail-biting Viterbi search (a 32-position warmup to the middle of the tile followed by a full shifted pass).
3. **Fractional mode constraints:** The quantizer enforces `D2` / `D4` parity constraints during path search; the decoder reconstructs the implied bits without runtime validation.

---

## 9. Row Lookup Tables (`RowTable` and `LtqRing` Rings)

Index-addressed lookup tables of shape $[\text{rows}, \text{dim}]$ (token embeddings, PLE tables, n-gram hash tables) must support reading an individual row without decoding neighboring rows. In `RowCodec::LtqRing` mode, each row is compressed independently into a sequence of **LTQ rings**:

- A row of dimension `dim` is partitioned into rings of length $L$, where $L$ is a multiple of 16 that divides `dim` (e.g. $L = 256$ for Gemma 4 PLE; $L = 160$ for Qwen3.8-Flash-Next n-grams).
- Each ring is a tail-biting trellis of length $L$: $s_t = \sum_{j=0}^{9} c_{(t - j) \bmod L} \cdot 2^j$. Field word counts scale by $L$ instead of 256 (only formats with $\text{sub} = 16$ are permitted).
- Ring values are reconstructed using a per-ring `fp16` scale $d$, a block Walsh–Hadamard transform $H_B$ (where $B \le 256$ is the largest power of two dividing $L$: $B = 256$ for $L = 256$, $B = 32$ for $L = 160$), and a table-wide sign vector $\sigma \in \{-1, +1\}^L$:
  $$v = \sigma \odot H_B(d \cdot q)$$
- On disk, each ring is stored as a contiguous record (`fp16` scale $d$ followed by `codes`, `planes`, `lsb`, `gains`) of $2 + \text{tile\_bytes}(L)$ bytes (for `TCQ4X` at $L = 160$, $86\text{ B}$ vs. $162\text{ B}$ for scaled `int8`), and all rings of a row are stored back-to-back in `{p}.ltq_rows.data` (`uint8`, $[\text{rows}, \text{row\_bytes}]$) alongside `{p}.ltq_rows.signs` (`int8`, $[L]$). Gathering any row from disk or over PCIe therefore requires only a single contiguous read.
- Benchmark on the `embed_tokens_per_layer` tables of **Gemma 4 E4B** (5.25 GiB in `bf16`, $262\,144 \times 10\,752$, [`results/g4_ple_gemma4.json`](https://github.com/lumen-inference/vtq-research/blob/main/results/g4_ple_gemma4.json)): rotated `TCQ3X` rings compress the table to **1.05 GiB** at KLD `0.0023 / 0.0030` (Wiki / Code), and rotated `TCQ4X` rings compress it to **1.39 GiB** at KLD `0.0010 / 0.0011` (vs. 2.65 GiB and `0.0006` for `int8`), accounting for $3\%$ and $11\%$ of total full-model KLD at 3.0 and 4.5 bpw respectively.

---

## 10. Outside v1

1. **Packing the block exponent $e$ into dense 5-bit fields** (saving $-0.012$ bits/weight compared to the byte-aligned `int8` representation).
2. **Backend-specific tile layouts (`CpuBand16` and `GpuWarp`):** on-disk `.vtq` files store the canonical layout of Section 6 (`TileLayout::Canonical`) by default. Repacking into CPU vector bands or GPU warp layouts is planned at model load time or ahead of time through `velri pack` (see [architecture.md](architecture.md)); the command is not implemented.
