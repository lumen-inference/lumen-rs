//! `#[repr(C)]` kernel ABI structures and LTQ format constants shared between the Rust host
//! (`lumen-cuda`, `lumen-cpu`, `lumen-ltq`) and CUDA C++ kernels (`lumen-cuda-kernels`).
//!
//! `cbindgen` generates `lumen_kabi.h` from this crate.

#![no_std]

/// Trellis sliding window length in bits ($s_t$ covers the last 10 coded bits).
pub const LTQ_NU: u32 = 10;

/// Number of weights in a standard $16 \times 16$ LTQ matrix tile.
pub const LTQ_TILE: u32 = 256;

/// Tile row height in elements.
pub const LTQ_TILE_ROWS: u32 = 16;

/// Tile column width in elements.
pub const LTQ_TILE_COLS: u32 = 16;

/// Width of an input Hadamard block and scale exponent block along the $K$ axis.
pub const LTQ_BLOCK_K: u32 = 256;

/// Primary Ungerboeck polynomial parity mask over $(s_t \gg 1)$ used in all LTQ v1 formats.
pub const LTQ_MASK_M0: u32 = 359;

/// Maximum number of parity masks per format ($m_0, m_1, m_2 \to$ up to 16 subsets).
pub const LTQ_MAX_MASKS: usize = 3;

/// Fractional bitrate constraint on the lowest uncoded bit ($u_t \ \& \ 1$).
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum LtqFracMode {
    /// Independent LSB for every position ($K$ bits/weight).
    None = 0,
    /// Shared LSB across pairs $(2q, 2q+1)$ ($K - 0.5$ bits/weight).
    D2 = 1,
    /// Even-sum parity across quads $(4q \dots 4q+3)$ ($K - 0.25$ bits/weight).
    D4 = 2,
}

/// `#[repr(C)]` descriptor of a single LTQ format entry passed to CPU/GPU kernels.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct LtqFormatAbi {
    /// Nominal bits per weight before fractional constraint ($2 \dots 5$).
    pub k: u32,
    /// Fractional mode (`0 = None`, `1 = D2`, `2 = D4`).
    pub frac: u32,
    /// Number of active parity masks in `masks` ($1 \dots 3$).
    pub num_masks: u32,
    /// Parity masks $[m_0, m_1, m_2]$ applied to $(s_t \gg 1)$.
    pub masks: [u32; LTQ_MAX_MASKS],
    /// Number of bits per local gain index ($0 \dots 4$).
    pub h: u32,
    /// Subgroup length sharing a single gain index (`16` or `32`).
    pub sub: u32,
}

/// `#[repr(C)]` descriptor of a horizontal row slice inside an `LtqLinear` matrix.
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LtqSliceAbi {
    /// Inclusive start row index (multiple of 16).
    pub row_start: u32,
    /// Exclusive end row index (multiple of 16).
    pub row_end: u32,
    /// Format index in the container's `formats` table.
    pub format_idx: u32,
    /// Byte offset of this slice's packed data inside `{p}.ltq.data`.
    pub byte_offset: u64,
}
