//! Reference implementation of the LTQ v1 (Lumen Trellis Quant) specification
//! (`docs/RU/ltq-format.md`):
//! - $16 \times 16$ tile and 1D ring (`LtqRing`) packing and decoding (`decode_tile`, `decode_ring`)
//! - Tail-biting 10-bit trellis state ($O(1)$ shift register) and Ungerboeck parity masks
//! - Fractional bitrate constraints (`None`, `D2`, `D4`)
//! - Normalized block Walsh-Hadamard transform ($H_{256}$ and $H_B$)

use lumen_core::{LumenError, Result, f16};
use lumen_kabi::{LTQ_MASK_M0, LTQ_MAX_MASKS, LTQ_NU, LTQ_TILE, LtqFormatAbi, LtqFracMode};
use serde::{Deserialize, Serialize};

// The constants below are defined once, in `lumen-kabi`, because the CUDA/C++ side consumes the
// same values through the generated `lumen_kabi.h`.

/// Trellis sliding window length in bits ($s_t$ covers the last 10 code bits).
pub const NU: usize = LTQ_NU as usize;

/// Number of weights in a standard $16 \times 16$ matrix tile.
pub const TILE: usize = LTQ_TILE as usize;

/// Primary Ungerboeck polynomial parity mask over $(s_t \gg 1)$ used in all LTQ v1 formats.
pub const MASK_M0: u32 = LTQ_MASK_M0;

/// Largest supported nominal bit width $K$ (the v1 menu tops out at `TCQ5X`).
pub const MAX_K: usize = 5;

/// Largest supported number of gain-index bits $h$ (the v1 menu tops out at `TCQ5X`, $h = 4$).
pub const MAX_H: usize = 4;

/// Fractional bitrate constraint on the lowest uncoded bit ($u_t \ \& \ 1$).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Frac {
    /// Independent LSB per position ($K$ bits/weight).
    None,
    /// Shared LSB in pairs $(2q, 2q+1)$ ($K - 0.5$ bits/weight).
    D2,
    /// Even-sum parity across quads $(4q \dots 4q+3)$ ($K - 0.25$ bits/weight).
    D4,
}

/// Complete specification of a single LTQ format entry in `.ltq` header (`quantization.formats`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LtqFormat {
    pub name: String,
    #[serde(rename = "K")]
    pub k: usize,
    pub frac: Frac,
    pub masks: Vec<u32>,
    pub h: usize,
    pub sub: usize,
    pub lut: Vec<f32>,
    pub gain: Vec<f32>,
}

impl From<Frac> for LtqFracMode {
    fn from(f: Frac) -> Self {
        match f {
            Frac::None => Self::None,
            Frac::D2 => Self::D2,
            Frac::D4 => Self::D4,
        }
    }
}

impl LtqFormat {
    /// Number of trellis subsets (`labels`): $2^{|\text{masks}| + 1}$ (4, 8, or 16).
    #[inline]
    pub fn num_labels(&self) -> usize {
        1usize << (self.masks.len() + 1)
    }

    /// Number of uncoded levels per subset: $U = 2^{K - 1}$.
    #[inline]
    pub fn uncoded_levels(&self) -> usize {
        1usize << (self.k - 1)
    }

    /// Number of `u16` words in the `lsb` field for a tile (`len = 256`) or ring of length `len`.
    #[inline]
    pub fn lsb_words(&self, len: usize) -> usize {
        if self.k <= 1 {
            return 0;
        }
        let bits = match self.frac {
            Frac::None => len,
            Frac::D2 => len / 2,
            Frac::D4 => (3 * len) / 4,
        };
        bits.div_ceil(16)
    }

    /// Number of `u16` words in the `gains` field for a tile (`len = 256`) or ring of length `len`.
    #[inline]
    pub fn gain_words(&self, len: usize) -> usize {
        ((len / self.sub) * self.h).div_ceil(16)
    }

    /// Total packed size in bytes for a tile (`len = 256`) or a ring of length `len`
    /// (excluding the ring's `fp16` scale).
    #[inline]
    pub fn tile_bytes(&self, len: usize) -> usize {
        let code_words = len / 16;
        let plane_words = code_words * self.k.saturating_sub(2);
        2 * (code_words + plane_words + self.lsb_words(len) + self.gain_words(len))
    }

    /// Effective bits per weight for a matrix of input width `k_in`:
    /// tile bits + block exponent `e` ($8/256$) + row scale `d_row` ($16/K_{\text{in}}$).
    #[inline]
    pub fn bpw(&self, k_in: usize) -> f64 {
        (self.tile_bytes(TILE) as f64) * 8.0 / (TILE as f64) + 8.0 / 256.0 + 16.0 / (k_in as f64)
    }

    /// Validates that the format parameters and `lut`/`gain` table lengths satisfy the LTQ v1 spec.
    ///
    /// This is the single gate for formats coming from an (untrusted) `.ltq` header: after it
    /// succeeds, every shift, index and table size derived from the format is in range.
    pub fn validate(&self) -> Result<()> {
        let bad = |msg: String| LumenError::Format(format!("LTQ format '{}': {msg}", self.name));

        if !(2..=MAX_K).contains(&self.k) {
            return Err(bad(format!(
                "K={} outside supported range 2..={MAX_K}",
                self.k
            )));
        }
        if self.masks.is_empty() || self.masks.len() > LTQ_MAX_MASKS {
            return Err(bad(format!(
                "expected 1..={LTQ_MAX_MASKS} parity masks, got {}",
                self.masks.len()
            )));
        }
        if self.masks[0] != MASK_M0 {
            return Err(bad(format!(
                "first parity mask must be the base mask {MASK_M0}, got {}",
                self.masks[0]
            )));
        }
        // Masks are applied to the 9 history bits `s >> 1`.
        if let Some(m) = self.masks.iter().find(|&&m| m >= 1 << (NU - 1)) {
            return Err(bad(format!(
                "parity mask {m} does not fit the {}-bit history",
                NU - 1
            )));
        }
        if self.h > MAX_H {
            return Err(bad(format!(
                "h={} outside supported range 0..={MAX_H}",
                self.h
            )));
        }
        if self.sub != 16 && self.sub != 32 {
            return Err(bad(format!("sub must be 16 or 32, got {}", self.sub)));
        }
        let expected_lut = self.num_labels() * self.uncoded_levels();
        if self.lut.len() != expected_lut {
            return Err(bad(format!(
                "expected lut length {expected_lut}, got {}",
                self.lut.len()
            )));
        }
        let expected_gain = 1usize << self.h;
        if self.gain.len() != expected_gain {
            return Err(bad(format!(
                "expected gain length {expected_gain}, got {}",
                self.gain.len()
            )));
        }
        if self.lut.iter().chain(&self.gain).any(|v| !v.is_finite()) {
            return Err(bad("lut/gain contain NaN or infinity".to_string()));
        }
        Ok(())
    }

    /// Converts a **validated** format into the `#[repr(C)]` descriptor passed to kernels.
    ///
    /// # Panics
    /// Panics if the format was not validated (more than `LTQ_MAX_MASKS` masks).
    pub fn to_abi(&self) -> LtqFormatAbi {
        let mut masks = [0u32; LTQ_MAX_MASKS];
        masks[..self.masks.len()].copy_from_slice(&self.masks);
        LtqFormatAbi {
            k: self.k as u32,
            frac: LtqFracMode::from(self.frac) as u32,
            num_masks: self.masks.len() as u32,
            masks,
            h: self.h as u32,
            sub: self.sub as u32,
        }
    }

    /// Checks that all buffers passed to [`decode_ring`] have the lengths this format requires
    /// for a ring (or tile, `len == 256`) of `len` weights.
    ///
    /// Loaders call this once per table; [`decode_ring`] repeats it as a (cheap) assertion.
    pub fn check_ring_buffers(
        &self,
        len: usize,
        codes: &[u16],
        planes: &[u16],
        lsb: &[u16],
        gains: &[u16],
    ) -> Result<()> {
        let bad = |msg: String| LumenError::Format(format!("LTQ ring (len={len}): {msg}"));
        if len < 16 || len % 16 != 0 {
            return Err(bad("length must be a non-zero multiple of 16".to_string()));
        }
        if len % self.sub != 0 {
            return Err(bad(format!(
                "length must be a multiple of sub={}",
                self.sub
            )));
        }
        let words = len / 16;
        let checks = [
            ("codes", codes.len(), words),
            ("planes", planes.len(), words * self.k.saturating_sub(2)),
            ("lsb", lsb.len(), self.lsb_words(len)),
            ("gains", gains.len(), self.gain_words(len)),
        ];
        for (name, got, want) in checks {
            if got < want {
                return Err(bad(format!("`{name}` has {got} words, need {want}")));
            }
        }
        Ok(())
    }

    /// Computes subset label from a 10-bit trellis state `s`.
    #[inline]
    fn state_to_label(&self, s: u32) -> u32 {
        let hist = s >> 1;
        let mut label = (s & 1) | ((hist & self.masks[0]).count_ones() & 1) << 1;
        for (i, &m) in self.masks[1..].iter().enumerate() {
            label |= ((hist & m).count_ones() & 1) << (2 + i);
        }
        label
    }
}

/// Decodes 256 unscaled values of a $16 \times 16$ tile per Section 7 of `docs/RU/ltq-format.md`:
/// `q[t] = gain[g] * lut[label * 2^(K - 1) + u]`, where `t = 16 * r + c`.
pub fn decode_tile(
    f: &LtqFormat,
    codes: &[u16; 16],
    planes: &[[u16; 16]],
    lsb: &[u16],
    gains: &[u16],
) -> [f32; 256] {
    let mut q = [0f32; TILE];
    decode_ring(f, TILE, codes, planes.as_flattened(), lsb, gains, &mut q);
    q
}

/// Decodes a tail-biting LTQ ring of length `len` (`len` is a multiple of 16, e.g. `256` or `160`)
/// using an $O(1)$ 10-bit shift register for the trellis state $s_t$.
///
/// # Panics
/// Panics if `len` is not a non-zero multiple of 16, if `out.len() != len`, or if any input slice
/// is shorter than [`LtqFormat::check_ring_buffers`] requires. `f` must be validated.
pub fn decode_ring(
    f: &LtqFormat,
    len: usize,
    codes: &[u16],
    planes: &[u16],
    lsb: &[u16],
    gains: &[u16],
    out: &mut [f32],
) {
    if let Err(e) = f.check_ring_buffers(len, codes, planes, lsb, gains) {
        panic!("{e}");
    }
    assert_eq!(
        out.len(),
        len,
        "output buffer length must equal ring length"
    );
    let words_per_plane = len / 16;
    let bit = |w: &[u16], i: usize| ((w[i >> 4] >> (i & 15)) & 1) as u32;

    // Seed the 9 history bits from the tail of the ring: (len - 9)..len
    let mut s = 0u32;
    for idx in (len - (NU - 1))..len {
        s = (s << 1) | bit(codes, idx);
    }

    for (t, out_val) in out.iter_mut().enumerate() {
        // Advance 10-bit shift register in O(1): bit 0 is c_t, bits 1..=9 are c_{t-1}..c_{t-9}
        s = ((s << 1) & 0x3FF) | bit(codes, t);
        let label = f.state_to_label(s);

        let mut u = match f.frac {
            Frac::None => bit(lsb, t),
            Frac::D2 => bit(lsb, t / 2),
            Frac::D4 if t % 4 < 3 => bit(lsb, 3 * (t / 4) + t % 4),
            Frac::D4 => {
                let base = 3 * (t / 4);
                bit(lsb, base) ^ bit(lsb, base + 1) ^ bit(lsb, base + 2)
            }
        };

        for p in 1..f.k - 1 {
            let plane = &planes[(p - 1) * words_per_plane..p * words_per_plane];
            u |= bit(plane, t) << p;
        }

        let mut g = 0usize;
        let g_base = (t / f.sub) * f.h;
        for i in 0..f.h {
            g |= (bit(gains, g_base + i) as usize) << i;
        }
        *out_val = f.gain[g] * f.lut[((label as usize) << (f.k - 1)) | (u as usize)];
    }
}

/// Computes the trellis subset labels $\text{label}_t$ from `code_bits` using an $O(1)$ shift register.
///
/// # Panics
/// Panics if `code_bits.len()` is smaller than the 9 history bits the window needs.
pub fn labels_from_code_bits(f: &LtqFormat, code_bits: &[u8]) -> Vec<u32> {
    let len = code_bits.len();
    assert!(
        len >= NU - 1,
        "ring must hold at least {} code bits",
        NU - 1
    );
    let mut s = 0u32;
    for &b in &code_bits[len - (NU - 1)..] {
        s = (s << 1) | u32::from(b & 1);
    }
    code_bits
        .iter()
        .map(|&b| {
            s = ((s << 1) & 0x3FF) | u32::from(b & 1);
            f.state_to_label(s)
        })
        .collect()
}

/// Packs `labels`, uncoded indices `u`, and subgroup gain indices `gidx` directly into
/// `(codes, planes, lsb, gains)` `u16` word arrays without intermediate bit-vector allocations.
///
/// The inputs must already satisfy the format's constraints (checked with `debug_assert!`):
/// for `D2` the two lowest uncoded bits of each pair are equal, for `D4` the lowest bits of each
/// quad have even parity, and the gain index is constant inside every `sub`-sized group.
///
/// # Panics
/// Panics if the slice lengths differ or are not a multiple of 16.
pub fn pack_ring(
    f: &LtqFormat,
    labels: &[u32],
    u: &[u32],
    gidx: &[u32],
) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    let len = labels.len();
    assert_eq!(len % 16, 0);
    assert_eq!(u.len(), len);
    assert_eq!(gidx.len(), len);

    #[cfg(debug_assertions)]
    {
        match f.frac {
            Frac::None => {}
            Frac::D2 => assert!(
                u.chunks_exact(2).all(|p| (p[0] ^ p[1]) & 1 == 0),
                "D2: lowest uncoded bits of a pair must be equal"
            ),
            Frac::D4 => assert!(
                u.chunks_exact(4)
                    .all(|q| (q[0] ^ q[1] ^ q[2] ^ q[3]) & 1 == 0),
                "D4: lowest uncoded bits of a quad must have even parity"
            ),
        }
        assert!(
            gidx.chunks(f.sub).all(|g| g.iter().all(|&x| x == g[0])),
            "gain index must be constant inside each sub-group"
        );
    }

    let words_per_plane = len / 16;
    let set_bit = |words: &mut [u16], bit_idx: usize, val: u32| {
        if (val & 1) != 0 {
            words[bit_idx >> 4] |= 1u16 << (bit_idx & 15);
        }
    };

    let mut codes = vec![0u16; words_per_plane];
    for (i, &l) in labels.iter().enumerate() {
        set_bit(&mut codes, i, l);
    }

    let num_planes = f.k.saturating_sub(2);
    let mut planes = vec![0u16; words_per_plane * num_planes];
    for p in 1..f.k.saturating_sub(1) {
        let slice = &mut planes[(p - 1) * words_per_plane..p * words_per_plane];
        for (i, &val) in u.iter().enumerate() {
            set_bit(slice, i, val >> p);
        }
    }

    let mut lsb = vec![0u16; f.lsb_words(len)];
    match f.frac {
        Frac::None => {
            for (i, &v) in u.iter().enumerate() {
                set_bit(&mut lsb, i, v);
            }
        }
        Frac::D2 => {
            for (i, &v) in u.iter().step_by(2).enumerate() {
                set_bit(&mut lsb, i, v);
            }
        }
        Frac::D4 => {
            for (q, quad) in u.chunks_exact(4).enumerate() {
                set_bit(&mut lsb, 3 * q, quad[0]);
                set_bit(&mut lsb, 3 * q + 1, quad[1]);
                set_bit(&mut lsb, 3 * q + 2, quad[2]);
            }
        }
    }

    let mut gains = vec![0u16; f.gain_words(len)];
    if f.h > 0 {
        for (grp, &g) in gidx.iter().step_by(f.sub).enumerate() {
            for bit_idx in 0..f.h {
                set_bit(&mut gains, grp * f.h + bit_idx, g >> bit_idx);
            }
        }
    }

    (codes, planes, lsb, gains)
}

// ============================================================================
// Normalized Fast Walsh-Hadamard Transform (H_B and H_256)
// ============================================================================

/// Applies an in-place normalized Walsh-Hadamard transform of block size `block_size`
/// (where `block_size` is a power of two, e.g. `256` or `32`) to every contiguous block of `data`.
pub fn hadamard_blocks_inplace(data: &mut [f32], block_size: usize) {
    assert!(
        block_size.is_power_of_two() && block_size > 0,
        "Hadamard block_size must be a power of two"
    );
    assert_eq!(
        data.len() % block_size,
        0,
        "data length must be a multiple of block_size"
    );
    let norm = 1.0 / (block_size as f32).sqrt();

    for block in data.chunks_exact_mut(block_size) {
        let mut h = 1;
        while h < block_size {
            for i in (0..block_size).step_by(h * 2) {
                for j in i..(i + h) {
                    let x = block[j];
                    let y = block[j + h];
                    block[j] = x + y;
                    block[j + h] = x - y;
                }
            }
            h *= 2;
        }
        for v in block.iter_mut() {
            *v *= norm;
        }
    }
}

/// Applies the LTQ input rotation $\tilde{x} = H_{256}(\text{suh} \odot x)$ per Section 2 of
/// `docs/RU/ltq-format.md`.
pub fn rotate_input_256(x: &[f32], suh: &[f16], out: &mut [f32]) {
    assert_eq!(x.len(), suh.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 256, 0);

    for ((o, &xi), &si) in out.iter_mut().zip(x.iter()).zip(suh.iter()) {
        *o = xi * si.to_f32();
    }
    hadamard_blocks_inplace(out, 256);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that `LtqFormat::tile_bytes(256)` and `LtqFormat::bpw(4096)` match all 19 formats
    /// in the LTQ v1 menu table (`docs/RU/ltq-format.md` Section 5).
    #[test]
    fn test_ltq_v1_menu_sizes_and_bpw() {
        let menu = [
            ("TCQ2X0", 2, Frac::None, 3, 0, 16, 32, 64, 2.035),
            ("TCQ2X", 2, Frac::None, 3, 1, 16, 32, 66, 2.098),
            ("TCQ25X", 3, Frac::D2, 2, 1, 16, 32, 82, 2.598),
            ("TCQ275X", 3, Frac::D4, 2, 1, 16, 32, 90, 2.848),
            ("TCQ275X2", 3, Frac::D4, 2, 2, 16, 32, 92, 2.910),
            ("TCQ3X1", 3, Frac::None, 2, 1, 16, 32, 98, 3.098),
            ("TCQ3X", 3, Frac::None, 2, 2, 16, 32, 100, 3.160),
            ("TCQ35X1", 4, Frac::D2, 1, 1, 16, 32, 114, 3.598),
            ("TCQ35X", 4, Frac::D2, 1, 2, 16, 32, 116, 3.660),
            ("TCQ375X", 4, Frac::D4, 1, 2, 16, 32, 124, 3.910),
            ("TCQ4X1", 4, Frac::None, 1, 1, 16, 32, 130, 4.098),
            ("TCQ4W2", 4, Frac::None, 1, 2, 32, 32, 130, 4.098),
            ("TCQ4W3", 4, Frac::None, 1, 3, 32, 32, 132, 4.160),
            ("TCQ4X2", 4, Frac::None, 1, 2, 16, 32, 132, 4.160),
            ("TCQ4X", 4, Frac::None, 1, 3, 16, 32, 134, 4.223),
            ("TCQ45X", 5, Frac::D2, 1, 2, 16, 64, 148, 4.660),
            ("TCQ475X", 5, Frac::D4, 1, 2, 16, 64, 156, 4.910),
            ("TCQ5X3", 5, Frac::None, 1, 3, 16, 64, 166, 5.223),
            ("TCQ5X", 5, Frac::None, 1, 4, 16, 64, 168, 5.285),
        ];

        for (name, k, frac, n_masks, h, sub, lut_len, exp_bytes, exp_bpw) in menu {
            let fmt = LtqFormat {
                name: name.to_string(),
                k,
                frac,
                masks: vec![359; n_masks],
                h,
                sub,
                lut: vec![0.0; lut_len],
                gain: vec![1.0; 1 << h],
            };
            fmt.validate().expect("format should be valid");
            assert_eq!(
                fmt.tile_bytes(256),
                exp_bytes,
                "tile_bytes mismatch for {name}"
            );
            let bpw_rounded = (fmt.bpw(4096) * 1000.0).round() / 1000.0;
            assert!(
                (bpw_rounded - exp_bpw).abs() < 1e-6,
                "bpw mismatch for {name}: got {bpw_rounded}, expected {exp_bpw}"
            );
        }
    }

    #[test]
    fn test_pack_and_decode_tile_roundtrip_all_modes() {
        for (k, frac, masks, h, sub) in [
            (2, Frac::None, vec![359, 58, 299], 1, 16),
            (3, Frac::D2, vec![359, 241], 1, 16),
            (3, Frac::D4, vec![359, 241], 2, 16),
            (4, Frac::None, vec![359], 2, 32),
            (5, Frac::D4, vec![359], 3, 16),
        ] {
            let n_labels = 1usize << (masks.len() + 1);
            let u_levels = 1usize << (k - 1);
            let lut: Vec<f32> = (0..(n_labels * u_levels))
                .map(|i| (i as f32) * 0.125 - 2.0)
                .collect();
            let gain: Vec<f32> = (0..(1usize << h))
                .map(|g| 0.5 + (g as f32) * 0.25)
                .collect();

            let fmt = LtqFormat {
                name: "test".to_string(),
                k,
                frac,
                masks,
                h,
                sub,
                lut,
                gain,
            };
            fmt.validate().unwrap();

            let code_bits: Vec<u8> = (0..256).map(|t| ((t * 7 + 3) % 2) as u8).collect();
            let labels = labels_from_code_bits(&fmt, &code_bits);

            let mut u: Vec<u32> = (0..256).map(|t| ((t * 13 + 5) % u_levels) as u32).collect();
            match frac {
                Frac::None => {}
                Frac::D2 => {
                    for pair in u.chunks_exact_mut(2) {
                        pair[1] = (pair[1] & !1) | (pair[0] & 1);
                    }
                }
                Frac::D4 => {
                    for quad in u.chunks_exact_mut(4) {
                        let parity = (quad[0] ^ quad[1] ^ quad[2]) & 1;
                        quad[3] = (quad[3] & !1) | parity;
                    }
                }
            }

            let gidx: Vec<u32> = (0..256)
                .map(|t| (((t / sub) * 3 + 1) % (1 << h)) as u32)
                .collect();

            let (codes, planes_flat, lsb, gains) = pack_ring(&fmt, &labels, &u, &gidx);
            let codes_arr: [u16; 16] = codes.try_into().unwrap();
            let planes_arr: Vec<[u16; 16]> = planes_flat
                .chunks_exact(16)
                .map(|c| c.try_into().unwrap())
                .collect();

            let decoded = decode_tile(&fmt, &codes_arr, &planes_arr, &lsb, &gains);

            for t in 0..256 {
                let expected = fmt.gain[gidx[t] as usize]
                    * fmt.lut[(labels[t] as usize) * u_levels + u[t] as usize];
                assert_eq!(
                    decoded[t].to_bits(),
                    expected.to_bits(),
                    "bit-exact mismatch at position {t} for k={k}, frac={frac:?}"
                );
            }
        }
    }

    #[test]
    fn test_hadamard_256_involution() {
        let orig: Vec<f32> = (0..512).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let mut buf = orig.clone();
        hadamard_blocks_inplace(&mut buf, 256);
        let norm_orig: f32 = orig.iter().map(|x| x * x).sum();
        let norm_rot: f32 = buf.iter().map(|x| x * x).sum();
        assert!((norm_orig - norm_rot).abs() < 1e-3);

        hadamard_blocks_inplace(&mut buf, 256);
        for (a, b) in orig.iter().zip(buf.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // ------------------------------------------------------------------------------------
    // Differential test against the literal decoder from docs/EN/ltq-format.md Section 7.
    // ------------------------------------------------------------------------------------

    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn words(&mut self, n: usize) -> Vec<u16> {
            (0..n).map(|_| (self.next() >> 24) as u16).collect()
        }
    }

    /// Straight transcription of the spec decoder (window rebuilt from scratch for every `t`,
    /// tail-biting modulo `len`). Deliberately slow and independent of the optimized path.
    fn spec_decode(
        f: &LtqFormat,
        len: usize,
        codes: &[u16],
        planes: &[u16],
        lsb: &[u16],
        gains: &[u16],
    ) -> Vec<f32> {
        let bit = |w: &[u16], i: usize| ((w[i >> 4] >> (i & 15)) & 1) as u32;
        let wpp = len / 16;
        let mut q = vec![0f32; len];
        for (t, slot) in q.iter_mut().enumerate() {
            let mut s = 0u32;
            for j in 0..10 {
                s |= bit(codes, (t + len - j % len) % len) << j;
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
                    bit(lsb, 3 * (t / 4)) ^ bit(lsb, 3 * (t / 4) + 1) ^ bit(lsb, 3 * (t / 4) + 2)
                }
            };
            for p in 1..f.k - 1 {
                u |= bit(&planes[(p - 1) * wpp..p * wpp], t) << p;
            }
            let mut g = 0usize;
            for i in 0..f.h {
                g |= (bit(gains, (t / f.sub) * f.h + i) as usize) << i;
            }
            *slot = f.gain[g] * f.lut[((label as usize) << (f.k - 1)) | u as usize];
        }
        q
    }

    fn random_format(
        rng: &mut XorShift,
        k: usize,
        frac: Frac,
        n_masks: usize,
        h: usize,
        sub: usize,
    ) -> LtqFormat {
        let masks: Vec<u32> = [359, 58, 299].into_iter().take(n_masks).collect();
        let n_lut = (1usize << (n_masks + 1)) << (k - 1);
        LtqFormat {
            name: "rand".into(),
            k,
            frac,
            masks,
            h,
            sub,
            lut: (0..n_lut)
                .map(|_| (rng.next() % 2001) as f32 / 500.0 - 2.0)
                .collect(),
            gain: (0..1usize << h)
                .map(|_| 0.25 + (rng.next() % 100) as f32 / 50.0)
                .collect(),
        }
    }

    #[test]
    fn test_fast_decoder_matches_literal_spec_decoder() {
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let shapes = [
            (2, Frac::None, 3, 0, 16),
            (2, Frac::None, 3, 1, 16),
            (3, Frac::D2, 2, 1, 16),
            (3, Frac::D4, 2, 2, 16),
            (3, Frac::None, 2, 2, 16),
            (4, Frac::D2, 1, 2, 16),
            (4, Frac::None, 1, 2, 32),
            (4, Frac::None, 1, 3, 32),
            (5, Frac::D4, 1, 2, 16),
            (5, Frac::None, 1, 4, 16),
        ];
        for (k, frac, n_masks, h, sub) in shapes {
            let f = random_format(&mut rng, k, frac, n_masks, h, sub);
            f.validate().unwrap();
            for len in [256usize, 160, 64, 32] {
                if len % sub != 0 {
                    continue;
                }
                for _ in 0..8 {
                    let codes = rng.words(len / 16);
                    let planes = rng.words(len / 16 * (k - 2));
                    let lsb = rng.words(f.lsb_words(len));
                    let gains = rng.words(f.gain_words(len));
                    f.check_ring_buffers(len, &codes, &planes, &lsb, &gains)
                        .unwrap();

                    let want = spec_decode(&f, len, &codes, &planes, &lsb, &gains);
                    let mut got = vec![0f32; len];
                    decode_ring(&f, len, &codes, &planes, &lsb, &gains, &mut got);
                    for t in 0..len {
                        assert_eq!(
                            got[t].to_bits(),
                            want[t].to_bits(),
                            "k={k} frac={frac:?} h={h} sub={sub} len={len} t={t}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_validate_rejects_malformed_formats() {
        let mut rng = XorShift(1);
        let good = random_format(&mut rng, 3, Frac::None, 2, 2, 16);
        good.validate().unwrap();

        type Mutation = Box<dyn Fn(&mut LtqFormat)>;
        let mutations: Vec<(&str, Mutation)> = vec![
            ("huge h", Box::new(|f| f.h = 70)),
            ("h above menu", Box::new(|f| f.h = 5)),
            ("k too small", Box::new(|f| f.k = 1)),
            ("k too large", Box::new(|f| f.k = 6)),
            ("no masks", Box::new(|f| f.masks.clear())),
            ("too many masks", Box::new(|f| f.masks = vec![359; 4])),
            ("wrong base mask", Box::new(|f| f.masks[0] = 58)),
            ("mask exceeds history", Box::new(|f| f.masks[1] = 512)),
            ("bad sub", Box::new(|f| f.sub = 8)),
            (
                "short lut",
                Box::new(|f| {
                    f.lut.pop();
                }),
            ),
            (
                "short gain",
                Box::new(|f| {
                    f.gain.pop();
                }),
            ),
            ("nan in lut", Box::new(|f| f.lut[0] = f32::NAN)),
            ("inf in gain", Box::new(|f| f.gain[0] = f32::INFINITY)),
        ];
        for (name, mutate) in mutations {
            let mut f = good.clone();
            mutate(&mut f);
            assert!(f.validate().is_err(), "'{name}' must be rejected");
        }
    }

    #[test]
    fn test_check_ring_buffers_rejects_short_inputs() {
        let mut rng = XorShift(7);
        let f = random_format(&mut rng, 4, Frac::D2, 1, 2, 16);
        let (codes, planes) = (rng.words(10), rng.words(20));
        let (lsb, gains) = (rng.words(f.lsb_words(160)), rng.words(f.gain_words(160)));
        f.check_ring_buffers(160, &codes, &planes, &lsb, &gains)
            .unwrap();
        assert!(
            f.check_ring_buffers(160, &codes[..9], &planes, &lsb, &gains)
                .is_err()
        );
        assert!(
            f.check_ring_buffers(160, &codes, &planes[..19], &lsb, &gains)
                .is_err()
        );
        assert!(
            f.check_ring_buffers(160, &codes, &planes, &lsb[..lsb.len() - 1], &gains)
                .is_err()
        );
        assert!(
            f.check_ring_buffers(160, &codes, &planes, &lsb, &[])
                .is_err()
        );
        assert!(f.check_ring_buffers(0, &[], &[], &[], &[]).is_err());
        assert!(
            f.check_ring_buffers(24, &codes, &planes, &lsb, &gains)
                .is_err()
        );
    }

    #[test]
    fn test_hadamard_matches_explicit_sylvester_matrix() {
        // H_8 via Sylvester construction, entry (i, j) = (-1)^popcount(i & j) / sqrt(8)
        let x: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.5 - 1.3).collect();
        let mut got = x.clone();
        hadamard_blocks_inplace(&mut got, 8);
        for (i, g) in got.iter().enumerate() {
            let want: f32 = (0..8)
                .map(|j| {
                    if (i & j).count_ones() % 2 == 0 {
                        x[j]
                    } else {
                        -x[j]
                    }
                })
                .sum::<f32>()
                / 8f32.sqrt();
            assert!((g - want).abs() < 1e-5, "row {i}: {g} vs {want}");
        }
    }

    #[test]
    fn test_abi_conversion() {
        let mut rng = XorShift(3);
        let f = random_format(&mut rng, 3, Frac::D4, 2, 2, 16);
        let abi = f.to_abi();
        assert_eq!(
            (abi.k, abi.frac, abi.num_masks, abi.h, abi.sub),
            (3, 2, 2, 2, 16)
        );
        assert_eq!(abi.masks, [359, 58, 0]);
    }
}
