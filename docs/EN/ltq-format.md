# LTQ v1 — the lumen-rs weight quantization format

Oct 27, 2026 · @Vemloir

LTQ (*Lumen Trellis Quant*) is the engine's only format for quantized matrices. Every 256 weights are encoded with trellis-coded quantization (Ungerboeck TCQ with 1024 states) on a tail-biting trellis, with trained level tables and a gain header per 16 or 32 weights. Bit width is assigned per row slice: one matrix can mix formats from 2 to 5.3 bits per weight, including the fractional rates K − 0.5 and K − 0.25.

This document fixes what the quantizer writes and the engine reads: the weight reconstruction formula, the tile layout, the format menu, and the on-disk container. The reference is [`ltq-research/src/ltq_format.py`](https://github.com/lumen-inference/ltq-research/blob/main/src/ltq_format.py), a packer and decoder written from this document. Its self-test quantizes Gaussian tiles and row rings with the CUDA quantizer and checks the decode bit for bit: 512 tiles for each of the 19 menu formats and 256 rings of length 160 for each of the 17 formats allowed for rings — 0 mismatches ([`results/ltq_format_selftest.json`](https://github.com/lumen-inference/ltq-research/blob/main/results/ltq_format_selftest.json)). How the format fits into the engine is in [section 4 of architecture.md](architecture.md#4-weights-descriptors-and-loading), and why it looks like this is in [ltq-design.md](ltq-design.md).

## 1. Linear layer

The layer is y = W·x + b, with W of size \[M, K\]. During quantization K is zero-padded to a multiple of 256 and M to a multiple of 16.

```
x̃ = H₂₅₆(suh ⊙ x)                  input rotation, once per vector
y[m] = Σ_b d[m, b] · Σ_{k ∈ block b} q[m, k] · x̃[k]  + bias[m]
d[m, b] = d_row[m] · 2^(e[m, b] / 4)
```

- `suh` is an fp16 vector of length K: signs (random ±1 from the quantizer seed) and input channel scales.
- H₂₅₆ is the blockwise normalized Walsh–Hadamard transform: each block of 256 elements is multiplied by the 256 × 256 Hadamard matrix scaled by 1/16.
- b = k / 256 is the index of the 256-block along K. `d_row` is the fp16 row scale; `e` is an integer exponent in \[−16, 15\] per (row, block) pair.
- q\[m, k\] is the value from the tile (section 2) without the row scale.

The output is not rotated: rotating outputs gave no gain on real matrices (ablations of 26.10).

## 2. Tile

A tile is 16 rows × 16 columns: rows 16i..16i+15, columns 16·kb..16·kb+15, where kb = 0..K/16−1 (block b = kb / 16). Inside the tile a weight's position is t = 16·r + c (r is the row, c the column): row-major order. Everything below refers to one tile.

**Trellis.** Each position has one coded bit c\_t. The state of a position is the window of the last 10 coded bits, and the trellis is tail-biting: the start of the tile continues its end, so no initial state is stored.

```
s_t = Σ_{j=0..9} c_{(t − j) mod 256} · 2^j        (bit 0 is the position's own bit)
```

**Label** is the index of the level subset. It is the position's bit plus parities of the window of previous bits under the format masks:

```
label_t = (s_t & 1) | parity((s_t >> 1) & m₀) << 1 | parity((s_t >> 1) & m₁) << 2 | parity((s_t >> 1) & m₂) << 3
```

m₀ = 359 in every format. Only some formats have the extra masks m₁, m₂ (section 4); they give 8 or 16 subsets instead of 4.

**Value.** K − 1 uncoded bits u\_t select a level inside the subset, and a gain index g selects a multiplier for `sub` consecutive positions (sub = 16: per tile row, 32: per pair of rows):

```
q_t = gain[g_{⌊t / sub⌋}] · lut[label_t · 2^(K−1) + u_t]
```

`lut` (float32, n\_labels · 2^(K−1) entries, ordered by label, then u) and `gain` (float32, 2^h entries) come from the format table in the file. Their values are stored explicitly: different devices round the gain formula 2^((g − (2^h − 1)/2)·step) differently, and the decoder must use exactly the numbers the quantizer used.

**Fractional modes** tie the low uncoded bits (u & 1) of neighbouring positions:

| Mode | Constraint | Rate |
| --- | --- | --- |
| none | — | K bits per weight |
| D2 | in a pair (2q, 2q+1) the low bits are equal | K − 0.5 |
| D4 | in a group of four (4q..4q+3) the sum of the low bits is even | K − 0.25 |

The quantizer enforces the constraint inside Viterbi, and the decoder reconstructs the missing bits (section 3).

## 3. Tile fields

All fields are u16 arrays; bit j of word w is stream element 16·w + j (least significant bit first).

| Field | u16 words | Content |
| --- | --- | --- |
| `codes` | 16 | coded bits c\_t: word r is tile row r, bit c is position 16r + c |
| `planes` | 16 · (K − 2) | uncoded bits 1..K−2, one plane per bit, same layout as `codes` |
| `lsb` | 16 / 8 / 12 | low uncoded bit: no mode — 256 bits; D2 — bit q belongs to pair q; D4 — bits 3q, 3q+1, 3q+2 belong to positions 4q, 4q+1, 4q+2 of group q, and the bit of position 4q+3 is their XOR |
| `gains` | ⌈(256 / sub) · h / 16⌉ | gain indices back to back, h bits each |

Bytes per tile are 2 · (16 + 16(K − 2) + |lsb| + |gains|), from 64 (TCQ2X0) to 168 (TCQ5X). K = 2 has no `planes` field, h = 0 has no `gains` field.

## 4. Format menu

The v1 menu has 19 formats (`TCQPQ` in the experiments). The level tables are trained with Lloyd iterations on a Gaussian source through the same Viterbi (`ltq-research/src/train_tcq_x.py`) and are stored in the model file.

| Format | K | Mode | Subsets | Extra masks | h | Step | sub | lut entries | Bytes / tile | Bits/weight at K = 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TCQ2X0 | 2 | — | 16 | 58, 299 | 0 | — | 16 | 32 | 64 | 2.035 |
| TCQ2X | 2 | — | 16 | 58, 299 | 1 | 0.4 | 16 | 32 | 66 | 2.098 |
| TCQ25X | 3 | D2 | 8 | 241 | 1 | 0.4 | 16 | 32 | 82 | 2.598 |
| TCQ275X | 3 | D4 | 8 | 241 | 1 | 0.4 | 16 | 32 | 90 | 2.848 |
| TCQ275X2 | 3 | D4 | 8 | 241 | 2 | 0.22 | 16 | 32 | 92 | 2.910 |
| TCQ3X1 | 3 | — | 8 | 241 | 1 | 0.4 | 16 | 32 | 98 | 3.098 |
| TCQ3X | 3 | — | 8 | 241 | 2 | 0.22 | 16 | 32 | 100 | 3.160 |
| TCQ35X1 | 4 | D2 | 4 | — | 1 | 0.4 | 16 | 32 | 114 | 3.598 |
| TCQ35X | 4 | D2 | 4 | — | 2 | 0.22 | 16 | 32 | 116 | 3.660 |
| TCQ375X | 4 | D4 | 4 | — | 2 | 0.22 | 16 | 32 | 124 | 3.910 |
| TCQ4X1 | 4 | — | 4 | — | 1 | 0.4 | 16 | 32 | 130 | 4.098 |
| TCQ4W2 | 4 | — | 4 | — | 2 | 0.22 | 32 | 32 | 130 | 4.098 |
| TCQ4W3 | 4 | — | 4 | — | 3 | 0.1 | 32 | 32 | 132 | 4.160 |
| TCQ4X2 | 4 | — | 4 | — | 2 | 0.22 | 16 | 32 | 132 | 4.160 |
| TCQ4X | 4 | — | 4 | — | 3 | 0.15 | 16 | 32 | 134 | 4.223 |
| TCQ45X | 5 | D2 | 4 | — | 2 | 0.22 | 16 | 64 | 148 | 4.660 |
| TCQ475X | 5 | D4 | 4 | — | 2 | 0.22 | 16 | 64 | 156 | 4.910 |
| TCQ5X3 | 5 | — | 4 | — | 3 | 0.15 | 16 | 64 | 166 | 5.223 |
| TCQ5X | 5 | — | 4 | — | 4 | 0.06 | 16 | 64 | 168 | 5.285 |

Bits per weight = 8 · tile bytes / 256 + 8/256 (exponent `e`) + 16/K (`d_row`). `suh` adds another 16/M and is not included. A format's number in the file is its index in the container's `formats` list (section 6), not its position in this table. The menu can be extended: the decoder only sees the format parameters.

## 5. Row slices

A matrix is split by rows into slices, and DP assigns each slice its own format. A slice is a contiguous row range that is a multiple of 16: attention is usually cut into 256 rows, MLP into 512, a MoE expert into one slice per matrix. The `suh` rotation and the `d_row`, `e` scales are shared by the whole matrix. MoE experts are separate matrices with their own fields.

## 6. Container

The model is one `.ltq` file holding both metadata and weights, like GGUF: the engine needs nothing next to it. The quantizer reads the source BF16 checkpoint (safetensors) and writes this file. A quantized matrix `{p}` (the source `.weight` name without the suffix) is stored as five tensors:

| Tensor | dtype, shape | Content |
| --- | --- | --- |
| `{p}.ltq.slices` | int32, \[S, 4\] | per slice: first row, end (exclusive), format number, byte offset of the slice data within `{p}.ltq.data` |
| `{p}.ltq.data` | uint8, \[bytes\] | slice data back to back |
| `{p}.ltq.d_row` | fp16, \[M\] | row scale |
| `{p}.ltq.e` | int8, \[M, K/256\] | block scale exponent, values in \[−16, 15\] |
| `{p}.ltq.suh` | fp16, \[K\] | input signs and scales |
| `{p}.bias` | as in the source | optional |

**Slice data** is four regions in a row, each starting at a 64-byte-aligned address:

```
codes  [T][16]  u16
planes [T][K−2][16]  u16
lsb    [T][|lsb|]  u16
gains  [T][|gains|]  u16
```

T = (rows in the slice / 16) · (K / 16). Tiles go by row group, and by kb within a group. Fields of one kind lie together (SoA): a kernel loads codes, planes and gains with wide reads, and a backend can repack them for itself at load time ([section 4 of architecture.md](architecture.md#4-weights-descriptors-and-loading)).

**File layout** (little-endian):

```
offset   size   content
0        4      magic "LTQ\0"
4        4      u32: container version, 1
8        8      u64: header length H
16       H      header: UTF-8 JSON, padded with spaces so that the data starts at a multiple of 4096
16 + H   …      data: tensors; each starts at a multiple of 64 bytes, row tables at a multiple of 4096
```

**Header:**

```json
{
  "config": { "...": "config.json of the source model" },
  "generation_config": { "...": "..." },
  "tokenizer": { "...": "tokenizer.json" },
  "tokenizer_config": { "...": "..." }, "chat_template": "...",
  "processor_config": { "...": "multimodal models only" },
  "quantization": {
    "quant_method": "ltq", "version": 1,
    "formats": [
      { "name": "TCQ3X", "K": 3, "frac": "none", "masks": [359, 241], "h": 2, "sub": 16,
        "lut": [ ... float32 ... ], "gain": [ ... float32, 2^h ... ] }
    ],
    "row_tables": { "{p}": { "format": 0, "ring": 160, "hadamard": 32 } },
    "modules_not_quantized": ["lm_head", "model.embed_tokens", "..."],
    "bits_per_weight": 3.0, "calibration": { "...": "..." }
  },
  "tensors": { "{name}": { "dtype": "u8", "shape": [ ... ], "offset": 0 } }
}
```

In `tensors`, dtype is one of `f32`, `f16`, `bf16`, `i32`, `i8`, `u8`; `offset` is counted from the start of the data section, and the size follows from dtype and shape. `frac` is `none`, `d2` or `d4`. The `lut` and `gain` values are written as float32 without loss (9 significant digits in JSON). The format number in `{p}.ltq.slices` is an index into `formats`. Tensors with `store = disk` are read straight from the file ([section 3 of architecture.md](architecture.md#3-memory-storage-locations-and-access-modes)), hence the page alignment of the data.

## 7. Reference decode

A port of `decode_tile` from [`ltq_format.py`](https://github.com/lumen-inference/ltq-research/blob/main/src/ltq_format.py) — the reference for the `lumen-ltq` crate and golden tests:

```rust
/// Tile values without the row scale: q[t] = gain[g] · lut[label · 2^(K−1) + u], t = 16·r + c.
fn decode_tile(f: &LtqFormat, codes: &[u16; 16], planes: &[[u16; 16]], lsb: &[u16], gains: &[u16]) -> [f32; 256] {
    let bit = |w: &[u16], i: usize| ((w[i >> 4] >> (i & 15)) & 1) as u32;
    let mut q = [0f32; 256];
    for t in 0..256 {
        let mut s = 0u32;                                     // window of 10 coded bits, tail-biting trellis
        for j in 0..10 { s |= bit(codes, (t + 256 - j) % 256) << j; }
        let mut label = (s & 1) | (((s >> 1) & f.masks[0]).count_ones() & 1) << 1;
        for (i, m) in f.masks[1..].iter().enumerate() {
            label |= (((s >> 1) & m).count_ones() & 1) << (2 + i);
        }
        let mut u = match f.frac {
            Frac::None => bit(lsb, t),
            Frac::D2 => bit(lsb, t / 2),
            Frac::D4 if t % 4 < 3 => bit(lsb, 3 * (t / 4) + t % 4),
            Frac::D4 => bit(lsb, 3 * (t / 4)) ^ bit(lsb, 3 * (t / 4) + 1) ^ bit(lsb, 3 * (t / 4) + 2),
        };
        for p in 1..f.k - 1 { u |= bit(&planes[p - 1], t) << p; }
        let mut g = 0usize;
        for i in 0..f.h { g |= (bit(gains, (t / f.sub) * f.h + i) as usize) << i; }
        q[t] = f.gain[g] * f.lut[((label as usize) << (f.k - 1)) | u as usize];
    }
    q
}
```

This is the reference, not a kernel path. Kernels do not recompute the window per position: on the CPU the labels of a tile row are assembled by XORing three table vectors (the label is linear over GF(2) in the window of coded bits), on the GPU they come from a shared "window + u → value" table in shared memory ([kernels.md](kernels.md)).

## 8. Quantizer contract

- The reconstruction the quantizer treats as its own and measures error against must equal the decode of the written fields: d\_row and e after scale polishing, `suh` in fp16, tables and gains — exactly the numbers that went into the file. The reference prototype stores the block scale as the fp16 product d\_row·c\_row·2^(e/4); in the Rust port, row scale polishing goes into `d_row`.
- Every tile is quantized by Viterbi on the tail-biting trellis (two passes: a 32-position window up to the middle of the tile, then a full shifted pass). The quantizer chooses gains and levels itself; the format allows any coded bit sequence.
- The fractional mode constraint holds for every group of the tile. The decoder does not check it; it reconstructs the missing bit.

## 9. Row tables

Lookup tables \[rows, dim\] — embeddings, PLE, n-grams — are quantized with LTQ rings (`RowCodec::LtqRing`) or stored dense (bf16 / fp16 / int8 with an fp16 scale per ring). A row is cut into rings of length L: L divides dim and is a multiple of 16 (Gemma 4 PLE: 256, the per-layer piece of a row; Qwen3.8-Flash-Next n-grams: 160, the whole row).

- **A ring** is the same trellis as in a tile, tail-biting over the length L: s\_t = Σ\_{j=0..9} c\_{(t − j) mod L} · 2^j; labels, values and fractional modes as in section 2. Fields as in section 3 with L instead of 256: `codes` and every plane are L/16 words, `lsb` is ⌈L / 16⌉, ⌈L / 32⌉ or ⌈3L / 64⌉ words, `gains` is ⌈(L / 16) · h / 16⌉ words. Only formats with sub = 16 are allowed (not TCQ4W2, TCQ4W3).
- **Scale and rotation.** A ring's value is v = σ ⊙ H\_B(d · q), where d is the fp16 ring scale, H\_B is the blockwise normalized Hadamard transform of size B (the largest power of two dividing L, at most 256: 256 for L = 256, 32 for L = 160), and σ is ±1 signs of length L shared by the table. The rotation is mandatory: on Gemma 4 E4B PLE it lowers KLD 2.2–2.4 times at the same MSE — without it the error is dominated by outlier coordinates.
- **A ring record** is the fp16 scale followed by `codes`, `planes`, `lsb`, `gains`; the rings of a row are consecutive, so a lookup reads one row as one contiguous chunk. Bytes per ring = 2 + field bytes; for TCQ4X at L = 160 that is 86 bytes against 162 for int8 with a scale.
- **Container:** `{p}.ltq_rows.data` uint8 \[rows, row bytes\], `{p}.ltq_rows.signs` int8 \[L\]; in the header — `quantization.row_tables`: `{"{p}": {"format": i, "ring": L, "hadamard": B}}`.

Measured on Gemma 4 E4B PLE (a 5.25 GiB table in bf16, everything else BF16, [`results/g4_ple_gemma4.json`](https://github.com/lumen-inference/ltq-research/blob/main/results/g4_ple_gemma4.json)): TCQ3X with rotation — 1.05 GiB, KLD 0.0023 / 0.0030; TCQ4X with rotation — 1.39 GiB, KLD 0.0010 / 0.0011; int8 — 2.65 GiB, KLD 0.0006. That is 3 % and 11 % of the KLD of the whole quantized model at 3.0 and 4.5 bits per weight respectively. The table's bit width is chosen from the model budget. Rings of length 160 with B = 32 are verified for the format but not for quality — that is done on Qwen3.8-Flash-Next.

## 10. Outside v1

- **Packing `e` into 5 bits** instead of int8 (−0.012 bits per weight).
- **Backend layouts.** The canonical layout is on disk; the CPU and GPU repack tiles for their kernels at load time or once with `lumen pack`.
