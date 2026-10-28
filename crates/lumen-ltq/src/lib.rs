//! Reference implementation of the LTQ v1 (Lumen Trellis Quant) specification
//! (`docs/RU/ltq-format.md`):
//! - $16 \times 16$ tile and 1D ring (`LtqRing`) packing and decoding (`decode_tile`, `decode_ring`)
//! - Tail-biting 10-bit trellis state and Ungerboeck polynomial parity masks
//! - Fractional bitrate constraints (`None`, `D2`, `D4`)
//! - Normalized block Walsh-Hadamard transform ($H_{256}$ and $H_B$)

use lumen_core::{f16, LumenError, Result};
use serde::{Deserialize, Serialize};

/// Trellis sliding window length in bits ($s_t$ covers the last 10 code bits).
pub const NU: usize = 10;

/// Number of weights in a standard $16 \times 16$ matrix tile.
pub const TILE: usize = 256;

/// Primary Ungerboeck polynomial parity mask over $(s_t \gg 1)$ used in all LTQ v1 formats.
pub const MASK_M0: u32 = 359;

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
        let plane_words = (len / 16) * self.k.saturating_sub(2);
        2 * (code_words + plane_words + self.lsb_words(len) + self.gain_words(len))
    }

    /// Effective bits per weight for a matrix of input width `k_in`:
    /// tile bits + block exponent `e` ($8/256$) + row scale `d_row` ($16/K_{\text{in}}$).
    #[inline]
    pub fn bpw(&self, k_in: usize) -> f64 {
        (self.tile_bytes(TILE) as f64) * 8.0 / (TILE as f64)
            + 8.0 / 256.0
            + 16.0 / (k_in as f64)
    }

    /// Validates that the format parameters and `lut`/`gain` table lengths satisfy the LTQ v1 spec.
    pub fn validate(&self) -> Result<()> {
        if !(2..=8).contains(&self.k) {
            return Err(LumenError::Format(format!(
                "LTQ format '{}': K={} outside supported range 2..=8",
                self.name, self.k
            )));
        }
        if self.masks.is_empty() || self.masks.len() > 3 {
            return Err(LumenError::Format(format!(
                "LTQ format '{}': expected 1..=3 parity masks, got {}",
                self.name,
                self.masks.len()
            )));
        }
        if self.sub != 16 && self.sub != 32 {
            return Err(LumenError::Format(format!(
                "LTQ format '{}': sub must be 16 or 32, got {}",
                self.name, self.sub
            )));
        }
        let expected_lut = self.num_labels() * self.uncoded_levels();
        if self.lut.len() != expected_lut {
            return Err(LumenError::Format(format!(
                "LTQ format '{}': expected lut length {}, got {}",
                self.name,
                expected_lut,
                self.lut.len()
            )));
        }
        let expected_gain = 1usize << self.h;
        if self.gain.len() != expected_gain {
            return Err(LumenError::Format(format!(
                "LTQ format '{}': expected gain length {}, got {}",
                self.name,
                expected_gain,
                self.gain.len()
            )));
        }
        Ok(())
    }
}

/// Decodes 256 unscaled values of a $16 \times 16$ tile per Section 7 of `docs/RU/ltq-format.md`:
/// `q[t] = gain[g] * lut[label * 2^(K - 1) + u]`, where `t = 16 * r + c`.
#[allow(clippy::needless_range_loop)]
pub fn decode_tile(
    f: &LtqFormat,
    codes: &[u16; 16],
    planes: &[[u16; 16]],
    lsb: &[u16],
    gains: &[u16],
) -> [f32; 256] {
    let bit = |w: &[u16], i: usize| ((w[i >> 4] >> (i & 15)) & 1) as u32;
    let mut q = [0f32; 256];

    for t in 0..256 {
        let mut s = 0u32; // 10-bit sliding window, tail-biting modulo 256
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

/// Decodes a general tail-biting LTQ ring of length `len` (where `len` is a multiple of 16,
/// e.g. `len = 256` for tiles/PLE or `len = 160` for n-gram row tables).
#[allow(clippy::needless_range_loop)]
pub fn decode_ring(
    f: &LtqFormat,
    len: usize,
    codes: &[u16],
    planes: &[u16],
    lsb: &[u16],
    gains: &[u16],
    out: &mut [f32],
) {
    debug_assert_eq!(len % 16, 0);
    debug_assert_eq!(out.len(), len);
    let words_per_plane = len / 16;
    let bit = |w: &[u16], i: usize| ((w[i >> 4] >> (i & 15)) & 1) as u32;

    for t in 0..len {
        let mut s = 0u32;
        for j in 0..NU {
            s |= bit(codes, (t + len - j) % len) << j;
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
            let plane_slice = &planes[(p - 1) * words_per_plane..p * words_per_plane];
            u |= bit(plane_slice, t) << p;
        }

        let mut g = 0usize;
        for i in 0..f.h {
            g |= (bit(gains, (t / f.sub) * f.h + i) as usize) << i;
        }
        out[t] = f.gain[g] * f.lut[((label as usize) << (f.k - 1)) | u as usize];
    }
}

/// Computes the 256-element trellis states $s_t$ and subset labels $\text{label}_t$ from `codes`.
pub fn labels_from_code_bits(f: &LtqFormat, code_bits: &[u8]) -> Vec<u32> {
    let len = code_bits.len();
    let mut labels = vec![0u32; len];
    for t in 0..len {
        let mut s = 0u32;
        for j in 0..NU {
            s |= u32::from(code_bits[(t + len - j) % len] & 1) << j;
        }
        let mut label = (s & 1) | (((s >> 1) & f.masks[0]).count_ones() & 1) << 1;
        for (i, m) in f.masks[1..].iter().enumerate() {
            label |= (((s >> 1) & m).count_ones() & 1) << (2 + i);
        }
        labels[t] = label;
    }
    labels
}

/// Packs `labels`, uncoded indices `u`, and subgroup gain indices `gidx` (one per element, sampled
/// every `f.sub` positions) into `(codes, planes, lsb, gains)` `u16` word arrays.
///
/// Matches `pack_tile` in `ltq_format.py`.
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

    let words_per_plane = len / 16;
    let pack_bits = |bits: &[u32], n_words: usize| -> Vec<u16> {
        let mut out = vec![0u16; n_words];
        for (i, &b) in bits.iter().enumerate() {
            if (b & 1) != 0 {
                out[i >> 4] |= 1u16 << (i & 15);
            }
        }
        out
    };

    let code_bits: Vec<u32> = labels.iter().map(|&l| l & 1).collect();
    let codes = pack_bits(&code_bits, words_per_plane);

    let mut planes = Vec::with_capacity(words_per_plane * f.k.saturating_sub(2));
    for p in 1..f.k.saturating_sub(1) {
        let p_bits: Vec<u32> = u.iter().map(|&val| (val >> p) & 1).collect();
        planes.extend_from_slice(&pack_bits(&p_bits, words_per_plane));
    }

    let lsb_bits: Vec<u32> = match f.frac {
        Frac::None => u.iter().map(|&v| v & 1).collect(),
        Frac::D2 => u.iter().step_by(2).map(|&v| v & 1).collect(),
        Frac::D4 => u
            .chunks_exact(4)
            .flat_map(|q| [q[0] & 1, q[1] & 1, q[2] & 1])
            .collect(),
    };
    let lsb = pack_bits(&lsb_bits, f.lsb_words(len));

    let gains = if f.h > 0 {
        let mut g_bits = Vec::with_capacity((len / f.sub) * f.h);
        for &g in gidx.iter().step_by(f.sub) {
            for bit_idx in 0..f.h {
                g_bits.push((g >> bit_idx) & 1);
            }
        }
        pack_bits(&g_bits, f.gain_words(len))
    } else {
        Vec::new()
    };

    (codes, planes, lsb, gains)
}

// ============================================================================
// Normalized Fast Walsh-Hadamard Transform (H_B and H_256)
// ============================================================================

/// Applies an in-place normalized Walsh-Hadamard transform of block size `block_size`
/// (where `block_size` is a power of two, e.g. `256` or `32`) to every contiguous block of `data`.
///
/// Since $H_B$ is symmetric and orthogonal with scale $1 / \sqrt{B}$, applying it twice is the identity.
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
        // (name, K, frac, masks_count, h, sub, expected_lut_len, expected_bytes, expected_bpw_k4096)
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
            let gain: Vec<f32> = (0..(1usize << h)).map(|g| 0.5 + (g as f32) * 0.25).collect();

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

            // Deterministic pseudo-random code bits, uncoded bits (satisfying D2/D4), and gains
            let code_bits: Vec<u8> = (0..256).map(|t| ((t * 7 + 3) % 2) as u8).collect();
            let labels = labels_from_code_bits(&fmt, &code_bits);

            let mut u: Vec<u32> = (0..256)
                .map(|t| ((t * 13 + 5) % u_levels) as u32)
                .collect();
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
            let codes_arr: [u16; 16] = codes.clone().try_into().unwrap();
            let planes_arr: Vec<[u16; 16]> = planes_flat
                .chunks_exact(16)
                .map(|c| c.try_into().unwrap())
                .collect();

            let decoded = decode_tile(&fmt, &codes_arr, &planes_arr, &lsb, &gains);

            for t in 0..256 {
                let expected =
                    fmt.gain[gidx[t] as usize] * fmt.lut[(labels[t] as usize) * u_levels + u[t] as usize];
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
        // After one transform, energy (L2 norm) is preserved
        let norm_orig: f32 = orig.iter().map(|x| x * x).sum();
        let norm_rot: f32 = buf.iter().map(|x| x * x).sum();
        assert!((norm_orig - norm_rot).abs() < 1e-3);

        // Applying H_256 a second time recovers the original vector
        hadamard_blocks_inplace(&mut buf, 256);
        for (a, b) in orig.iter().zip(buf.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
