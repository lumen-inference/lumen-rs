//! Memory subsystem (`docs/RU/architecture.md` Section 4):
//! - 256-byte aligned host buffers (`AlignedHostBuffer`) and file-backed `mmap` regions
//! - Static interval-coloring activation arena planner (`ArenaPlanner`, `ArenaLayout`)
//! - Paged KV-cache pool sizing (`KvPoolSpec`, page = 64 tokens, sliding ring $W + T_{\max} - 1$)
//! - Recurrent state pool sizing (`StatePoolSpec`) with speculative rollback history slots

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::Arc;

use lumen_core::{LumenError, MemLoc, Result};
use memmap2::Mmap;

/// Standard byte alignment for weight sub-allocations and activation arena buffers (256 bytes).
pub const MEM_ALIGN: usize = 256;

/// Standard KV-cache page size in tokens.
pub const KV_PAGE_TOKENS: usize = 64;

/// Opaque identifier of a buffer inside a compiled `Program` or memory pool.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct BufferId(pub u32);

/// Memory lifetime class (`docs/RU/architecture.md` Section 4).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MemoryClass {
    /// Model weights (session lifetime, 256-byte aligned sub-allocation).
    Weights,
    /// Intermediate step activations (single-step lifetime, static arena offsets).
    Activations,
    /// Attention KV cache (sequence lifetime, paged pool or sliding-window ring).
    KvCache,
    /// Recurrent linear-attention / PLE state (sequence lifetime + speculative rollback history).
    RecurrentState,
}

/// Rounds `size` up to the next multiple of `align` (`align` must be a power of two).
/// Returns `None` if the result does not fit in `usize`.
#[inline]
pub const fn checked_align_up(size: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    match size.checked_add(align - 1) {
        Some(v) => Some(v & !(align - 1)),
        None => None,
    }
}

/// Rounds `size` up to the next multiple of `align` (`align` must be a power of two).
///
/// # Panics
/// Panics on `usize` overflow; use [`checked_align_up`] for sizes derived from untrusted input.
#[inline]
pub const fn align_up(size: usize, align: usize) -> usize {
    match checked_align_up(size, align) {
        Some(v) => v,
        None => panic!("align_up overflow"),
    }
}

// ============================================================================
// 256-Byte Aligned Host Buffer
// ============================================================================

/// Owned host RAM buffer guaranteed to be aligned to `MEM_ALIGN` (256 bytes) for AVX-512
/// and CUDA `cuMemHostRegister` pinned DMA transfers.
pub struct AlignedHostBuffer {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
    numa_node: u16,
}

// SAFETY: `AlignedHostBuffer` uniquely owns its heap allocation of `u8` bytes.
unsafe impl Send for AlignedHostBuffer {}
unsafe impl Sync for AlignedHostBuffer {}

impl AlignedHostBuffer {
    /// Allocates a zero-initialized, 256-byte aligned buffer of `len` bytes in system RAM.
    pub fn new_zeroed(len: usize, numa_node: u16) -> Result<Self> {
        let alloc_len = len.max(1);
        let layout = Layout::from_size_align(alloc_len, MEM_ALIGN)
            .map_err(|e| LumenError::Memory(format!("invalid buffer layout (len={len}): {e}")))?;
        // SAFETY: `alloc_len >= 1` and `MEM_ALIGN` is a valid power-of-two alignment.
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw)
            .ok_or_else(|| LumenError::Memory(format!("host allocation failed for {len} bytes")))?;
        Ok(Self {
            ptr,
            len,
            layout,
            numa_node,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn numa_node(&self) -> u16 {
        self.numa_node
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: `ptr` is valid for `self.len` initialized bytes.
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: `ptr` is uniquely owned and valid for `self.len` initialized bytes.
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl Drop for AlignedHostBuffer {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated with `self.layout` via `alloc::alloc_zeroed`.
        unsafe {
            alloc::dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

impl std::fmt::Debug for AlignedHostBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedHostBuffer")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("numa_node", &self.numa_node)
            .finish()
    }
}

/// Unified storage backing for weights, arenas, or staging regions.
#[derive(Debug)]
pub enum Buffer {
    /// Host RAM allocation (`MemLoc::Ram`).
    Host(AlignedHostBuffer),
    /// Read-only memory-mapped file on NVMe disk (`MemLoc::Disk`).
    Mmap {
        mmap: Arc<Mmap>,
        offset: usize,
        len: usize,
    },
    /// Device VRAM allocation (`MemLoc::Vram`), managed by the device backend.
    Device {
        gpu_index: u16,
        device_ptr: u64,
        len: usize,
    },
}

impl Buffer {
    pub fn loc(&self) -> MemLoc {
        match self {
            Self::Host(b) => MemLoc::Ram(b.numa_node()),
            Self::Mmap { .. } => MemLoc::Disk,
            Self::Device { gpu_index, .. } => MemLoc::Vram(*gpu_index),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Host(b) => b.len(),
            Self::Mmap { len, .. } | Self::Device { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a host byte slice if the buffer resides in host RAM or an `mmap` region.
    pub fn host_slice(&self) -> Option<&[u8]> {
        match self {
            Self::Host(b) => Some(b.as_slice()),
            Self::Mmap { mmap, offset, len } => mmap.get(*offset..offset.checked_add(*len)?),
            Self::Device { .. } => None,
        }
    }
}

// ============================================================================
// Static Activation Arena Planner
// ============================================================================

/// Liveness interval `[first_step, last_step]` and byte size of a temporary activation buffer.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ActivationInterval {
    pub id: BufferId,
    pub byte_size: usize,
    pub first_op: usize,
    pub last_op: usize,
}

/// Statically computed arena layout assigning every activation buffer a fixed byte offset.
#[derive(Clone, Debug)]
pub struct ArenaLayout {
    pub total_bytes: usize,
    /// `(id, byte_offset, byte_size)`, sorted by `id`.
    pub offsets: Vec<(BufferId, usize, usize)>,
}

impl ArenaLayout {
    /// Looks up `(byte_offset, byte_size)` for `id`.
    pub fn get(&self, id: BufferId) -> Option<(usize, usize)> {
        self.offsets
            .binary_search_by_key(&id, |&(bid, _, _)| bid)
            .ok()
            .map(|i| (self.offsets[i].1, self.offsets[i].2))
    }
}

/// Static interval-packing planner for step activation arenas (`docs/RU/architecture.md` Section 4).
///
/// Reuses memory regions across non-overlapping operator lifetimes so the runtime performs
/// zero heap/VRAM allocations during token generation.
#[derive(Default)]
pub struct ArenaPlanner {
    intervals: Vec<ActivationInterval>,
}

impl ArenaPlanner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a buffer that is live during ops `first_op..=last_op`.
    ///
    /// # Errors
    /// Fails on a duplicate `id`, on `last_op < first_op` (a planner bug upstream, not something
    /// to paper over) and on sizes whose aligned value overflows `usize`.
    pub fn add_buffer(
        &mut self,
        id: BufferId,
        byte_size: usize,
        first_op: usize,
        last_op: usize,
    ) -> Result<()> {
        if last_op < first_op {
            return Err(LumenError::Memory(format!(
                "buffer {} has last_op {last_op} < first_op {first_op}",
                id.0
            )));
        }
        if self.intervals.iter().any(|i| i.id == id) {
            return Err(LumenError::Memory(format!("duplicate buffer id {}", id.0)));
        }
        let byte_size = checked_align_up(byte_size, MEM_ALIGN)
            .ok_or_else(|| LumenError::Memory(format!("buffer {} size overflows", id.0)))?;
        self.intervals.push(ActivationInterval {
            id,
            byte_size,
            first_op,
            last_op,
        });
        Ok(())
    }

    /// Computes a static `ArenaLayout`: buffers are visited from largest to smallest and each one
    /// takes the lowest aligned offset that does not collide with any buffer that is live at the
    /// same time (greedy first-fit by size, as in interval-coloring allocators).
    ///
    /// # Errors
    /// Fails if the total arena size overflows `usize`.
    pub fn compile(&self) -> Result<ArenaLayout> {
        let overflow = || LumenError::Memory("arena size overflows usize".to_string());

        let mut sorted = self.intervals.clone();
        sorted.sort_by(|a, b| {
            b.byte_size
                .cmp(&a.byte_size)
                .then_with(|| a.first_op.cmp(&b.first_op))
                .then_with(|| a.id.cmp(&b.id))
        });

        let mut placed: Vec<(ActivationInterval, usize)> = Vec::with_capacity(sorted.len());
        let mut occupied: Vec<(usize, usize)> = Vec::with_capacity(sorted.len());
        let mut total_bytes = 0usize;

        for item in sorted {
            if item.byte_size == 0 {
                placed.push((item, 0));
                continue;
            }

            // `[start, end)` ranges of already placed buffers whose lifetime overlaps `item`.
            occupied.clear();
            occupied.extend(
                placed
                    .iter()
                    .filter(|(p, _)| {
                        p.byte_size > 0 && p.first_op <= item.last_op && item.first_op <= p.last_op
                    })
                    .map(|(p, off)| (*off, *off + p.byte_size)),
            );
            occupied.sort_unstable_by_key(|&(start, _)| start);

            let mut candidate = 0usize;
            for &(occ_start, occ_end) in &occupied {
                let end = candidate.checked_add(item.byte_size).ok_or_else(overflow)?;
                if end <= occ_start {
                    break;
                }
                candidate =
                    candidate.max(checked_align_up(occ_end, MEM_ALIGN).ok_or_else(overflow)?);
            }

            let end = candidate.checked_add(item.byte_size).ok_or_else(overflow)?;
            total_bytes = total_bytes.max(end);
            placed.push((item, candidate));
        }

        placed.sort_by_key(|(p, _)| p.id);
        let offsets = placed
            .into_iter()
            .map(|(p, off)| (p.id, off, p.byte_size))
            .collect();

        Ok(ArenaLayout {
            total_bytes,
            offsets,
        })
    }
}

// ============================================================================
// KV Cache & Recurrent State Pool Sizing
// ============================================================================

/// KV cache storage layout mode for an attention layer.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KvLayoutMode {
    /// Separate $K$ and $V$ tensors (`k_bits`, `v_bits` in `2..=16`).
    Standard { k_bits: u8, v_bits: u8 },
    /// $K = V$ shared key/value mode (Gemma 4 global layers): stores only $V$.
    SharedKV { v_bits: u8 },
    /// Layer reuses another reference layer's KV cache (`num_kv_shared_layers` in Gemma 4 E2B/E4B).
    ExternalRef { source_layer: usize },
}

/// Sizing specification for a single attention layer's KV cache.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvLayerSpec {
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub mode: KvLayoutMode,
    /// Optional sliding window size $W$. When `Some(w)`, the ring holds $W + T_{\max} - 1$ tokens.
    /// `u32`, like `sliding_window` in `lumen_backend::Op::Attention`.
    pub sliding_window: Option<u32>,
}

impl KvLayerSpec {
    /// Effective number of token slots required for a sequence of `max_context` tokens
    /// and a maximum step/draft batch of `t_max` tokens.
    pub fn effective_token_capacity(&self, max_context: usize, t_max: usize) -> usize {
        if matches!(self.mode, KvLayoutMode::ExternalRef { .. }) {
            return 0;
        }
        match self.sliding_window {
            Some(w) => (w as usize)
                .saturating_add(t_max.saturating_sub(1))
                .min(max_context),
            None => max_context,
        }
    }

    /// Total bytes required for this layer's KV cache for one sequence.
    ///
    /// Layers with a sliding window use a ring buffer and are sized exactly; full-attention layers
    /// live in the paged pool and are rounded up to whole [`KV_PAGE_TOKENS`]-token pages.
    ///
    /// The result does not include the scale/zero-point metadata of quantized KV (its layout is
    /// not fixed yet).
    ///
    /// # Errors
    /// Fails if the size does not fit in `usize`.
    pub fn sequence_bytes(&self, max_context: usize, t_max: usize) -> Result<usize> {
        let overflow = || LumenError::Memory("KV cache size overflows usize".to_string());

        let mut tokens = self.effective_token_capacity(max_context, t_max);
        if self.sliding_window.is_none() && tokens > 0 {
            tokens = checked_align_up(tokens, KV_PAGE_TOKENS).ok_or_else(overflow)?;
        }
        let elems_per_tok = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .ok_or_else(overflow)?;
        let bits_per_elem = match self.mode {
            KvLayoutMode::Standard { k_bits, v_bits } => usize::from(k_bits) + usize::from(v_bits),
            KvLayoutMode::SharedKV { v_bits } => usize::from(v_bits),
            KvLayoutMode::ExternalRef { .. } => 0,
        };
        let bits = tokens
            .checked_mul(elems_per_tok)
            .and_then(|v| v.checked_mul(bits_per_elem))
            .ok_or_else(overflow)?;
        checked_align_up(bits.div_ceil(8), MEM_ALIGN).ok_or_else(overflow)
    }
}

/// Sizing specification for a recurrent state layer (`GatedDeltaNet` or `PLE`) in `StatePool`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct StateLayerSpec {
    /// Bytes for the recurrent state matrix (e.g., $H_v \times D_k \times D_v \times 4$ for GDN).
    pub matrix_bytes: usize,
    /// Bytes for the causal `conv1d` ring buffer (`(conv_kernel - 1) * channels * 2`).
    pub conv_bytes: usize,
    /// Number of speculative draft rollback slots ($k + 1$).
    pub history_slots: usize,
}

impl StateLayerSpec {
    /// Bytes for one sequence slot including all rollback history slots (at least one).
    /// Returns `None` on `usize` overflow.
    pub const fn total_slot_bytes(&self) -> Option<usize> {
        let Some(sum) = self.matrix_bytes.checked_add(self.conv_bytes) else {
            return None;
        };
        let Some(single) = checked_align_up(sum, MEM_ALIGN) else {
            return None;
        };
        let slots = if self.history_slots == 0 {
            1
        } else {
            self.history_slots
        };
        single.checked_mul(slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_host_buffer_alignment() {
        let mut buf = AlignedHostBuffer::new_zeroed(1000, 0).unwrap();
        assert_eq!(buf.as_ptr() as usize % MEM_ALIGN, 0);
        assert_eq!(buf.len(), 1000);
        buf.as_mut_slice()[999] = 42;
        assert_eq!(buf.as_slice()[999], 42);
    }

    #[test]
    fn test_arena_planner_reuses_disjoint_intervals() {
        let mut planner = ArenaPlanner::new();
        // Buffer 0 lives in ops 0..=2 (1024 bytes)
        planner.add_buffer(BufferId(0), 1024, 0, 2).unwrap();
        // Buffer 1 lives in ops 1..=3 (1024 bytes) -> overlaps with Buffer 0
        planner.add_buffer(BufferId(1), 1024, 1, 3).unwrap();
        // Buffer 2 lives in ops 3..=5 (1024 bytes) -> disjoint from Buffer 0, can reuse offset 0!
        planner.add_buffer(BufferId(2), 1024, 3, 5).unwrap();

        let layout = planner.compile().unwrap();
        assert_eq!(layout.total_bytes, 2048);
        let (off0, _) = layout.get(BufferId(0)).unwrap();
        let (off1, _) = layout.get(BufferId(1)).unwrap();
        let (off2, _) = layout.get(BufferId(2)).unwrap();
        assert_ne!(off0, off1);
        assert_eq!(off0, off2);
    }

    #[test]
    fn test_kv_layer_spec_sliding_and_shared_kv() {
        // Sliding window W = 1024, T_max = 64 -> ring capacity = 1087 tokens
        let spec = KvLayerSpec {
            num_kv_heads: 8,
            head_dim: 128,
            mode: KvLayoutMode::Standard {
                k_bits: 8,
                v_bits: 8,
            },
            sliding_window: Some(1024),
        };
        assert_eq!(spec.effective_token_capacity(262144, 64), 1087);

        // SharedKV (K=V) uses half the memory of Standard
        let std_full = KvLayerSpec {
            num_kv_heads: 8,
            head_dim: 128,
            mode: KvLayoutMode::Standard {
                k_bits: 16,
                v_bits: 16,
            },
            sliding_window: None,
        };
        let shared_kv = KvLayerSpec {
            num_kv_heads: 8,
            head_dim: 128,
            mode: KvLayoutMode::SharedKV { v_bits: 16 },
            sliding_window: None,
        };
        assert_eq!(
            std_full.sequence_bytes(4096, 1).unwrap(),
            shared_kv.sequence_bytes(4096, 1).unwrap() * 2
        );
    }

    #[test]
    fn test_kv_sequence_bytes_page_rounding_and_overflow() {
        let full = KvLayerSpec {
            num_kv_heads: 1,
            head_dim: 8,
            mode: KvLayoutMode::Standard {
                k_bits: 8,
                v_bits: 8,
            },
            sliding_window: None,
        };
        // 65 tokens occupy two 64-token pages: 128 tokens * 8 elems * 16 bits = 2048 B
        assert_eq!(full.sequence_bytes(65, 1).unwrap(), 2048);

        let huge = KvLayerSpec {
            num_kv_heads: usize::MAX,
            head_dim: 2,
            ..full
        };
        assert!(huge.sequence_bytes(4096, 1).is_err());
        assert!(full.sequence_bytes(usize::MAX, 1).is_err());
    }

    #[test]
    fn test_state_slot_bytes() {
        let s = StateLayerSpec {
            matrix_bytes: 3_145_728,
            conv_bytes: 100,
            history_slots: 0,
        };
        assert_eq!(s.total_slot_bytes(), Some(3_145_728 + 256));
        let s = StateLayerSpec {
            history_slots: 4,
            ..s
        };
        assert_eq!(s.total_slot_bytes(), Some((3_145_728 + 256) * 4));
        let s = StateLayerSpec {
            matrix_bytes: usize::MAX,
            conv_bytes: 1,
            history_slots: 1,
        };
        assert_eq!(s.total_slot_bytes(), None);
    }

    #[test]
    fn test_arena_planner_rejects_bad_buffers() {
        let mut p = ArenaPlanner::new();
        p.add_buffer(BufferId(0), 10, 0, 1).unwrap();
        assert!(p.add_buffer(BufferId(0), 10, 0, 1).is_err(), "duplicate id");
        assert!(
            p.add_buffer(BufferId(1), 10, 5, 2).is_err(),
            "inverted lifetime"
        );
        assert!(
            p.add_buffer(BufferId(2), usize::MAX, 0, 1).is_err(),
            "size overflow"
        );
    }

    #[test]
    fn test_arena_never_overlaps_live_buffers() {
        // Pseudo-random intervals; verify the invariant instead of specific offsets.
        let mut x = 12345u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut planner = ArenaPlanner::new();
        let mut specs = Vec::new();
        for i in 0..200u32 {
            let first = (next() % 50) as usize;
            let last = first + (next() % 10) as usize;
            let size = (next() % 5000) as usize;
            planner.add_buffer(BufferId(i), size, first, last).unwrap();
            specs.push((BufferId(i), first, last));
        }
        let layout = planner.compile().unwrap();
        for (i, &(a, a0, a1)) in specs.iter().enumerate() {
            let (ao, asz) = layout.get(a).unwrap();
            assert_eq!(ao % MEM_ALIGN, 0);
            assert!(ao + asz <= layout.total_bytes);
            for &(b, b0, b1) in &specs[i + 1..] {
                let (bo, bsz) = layout.get(b).unwrap();
                let live_together = a0 <= b1 && b0 <= a1;
                let disjoint = ao + asz <= bo || bo + bsz <= ao;
                assert!(
                    !live_together || asz == 0 || bsz == 0 || disjoint,
                    "{a:?} overlaps {b:?}"
                );
            }
        }
    }

    #[test]
    fn test_checked_align_up() {
        assert_eq!(checked_align_up(1, 256), Some(256));
        assert_eq!(checked_align_up(256, 256), Some(256));
        assert_eq!(checked_align_up(usize::MAX, 256), None);
    }
}
