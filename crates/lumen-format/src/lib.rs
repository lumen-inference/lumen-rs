//! `.ltq` v1 container reader and header parser (`docs/RU/ltq-format.md` Section 6).
//!
//! Binary layout (little-endian):
//! - `0..4`:   Magic signature `"LTQ\0"`
//! - `4..8`:   `u32` container version (`1`)
//! - `8..16`:  `u64` JSON header byte length `H`
//! - `16..16+H`: UTF-8 JSON header padded with spaces (`0x20`) so that `16 + H` is a multiple of 4096
//! - `16+H..`: Tensor payload area, starting on a 4096-byte boundary; tensor `offset`s are relative
//!   to its start and 64-byte aligned (4096 for row-table data), which makes them aligned in the
//!   file (and in the `mmap`) as well
//!
//! This matches the reference writer/reader in `experements/src/ltq_file.py`.
//!
//! The header is untrusted input: every size, offset and index it contains is validated before
//! any other crate sees it. This crate has no knowledge of model architectures.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use lumen_core::{DType, LumenError, Result, Shape};
use lumen_ltq::LtqFormat;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Magic 4-byte signature at offset 0 of every `.ltq` file.
pub const LTQ_MAGIC: [u8; 4] = *b"LTQ\0";

/// Supported `.ltq` container (binary layout) version.
pub const LTQ_CONTAINER_VERSION: u32 = 1;

/// Supported version of the `quantization` section (`quantization.version`). Independent of the
/// container version: the layout of the file and the meaning of the quantization tables can evolve
/// separately.
pub const LTQ_QUANT_VERSION: u32 = 1;

/// Length of the binary preamble (magic + version + header length).
pub const LTQ_PREAMBLE_LEN: usize = 16;

/// The tensor payload starts at file offset `16 + H`, which is a multiple of this many bytes.
pub const LTQ_HEADER_ALIGN: usize = 4096;

/// Alignment of every tensor offset (relative to the payload start, hence also absolute).
pub const LTQ_TENSOR_ALIGN: u64 = 64;

/// Alignment of the data of a row table (`{p}.ltq_rows.data`), for page-granular disk reads.
pub const LTQ_ROW_TABLE_ALIGN: u64 = 4096;

/// Metadata for a single `RowTable` compressed with `LtqRing` in `quantization.row_tables`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LtqRowTableMeta {
    pub format: u32,
    pub ring: u32,
    pub hadamard: u32,
}

impl LtqRowTableMeta {
    /// Checks the ring parameters against `formats` (`docs/EN/ltq-format.md` Section 9).
    pub fn validate(&self, table: &str, formats: &[LtqFormat]) -> Result<()> {
        let bad = |msg: String| LumenError::Format(format!("row table '{table}': {msg}"));
        let fmt = formats.get(self.format as usize).ok_or_else(|| {
            bad(format!(
                "format index {} out of range ({} formats)",
                self.format,
                formats.len()
            ))
        })?;
        if fmt.sub != 16 {
            return Err(bad(format!(
                "format '{}' has sub={}, rings require sub=16",
                fmt.name, fmt.sub
            )));
        }
        if self.ring == 0 || self.ring % 16 != 0 {
            return Err(bad(format!(
                "ring length {} is not a multiple of 16",
                self.ring
            )));
        }
        // B is the largest power of two dividing L, capped at 256.
        let expected_hadamard = (1u32 << self.ring.trailing_zeros()).min(256);
        if self.hadamard != expected_hadamard {
            return Err(bad(format!(
                "hadamard block {} must be the largest power of two (<= 256) dividing ring {}, i.e. {expected_hadamard}",
                self.hadamard, self.ring
            )));
        }
        Ok(())
    }
}

/// `"quantization"` section inside the `.ltq` JSON header.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LtqQuantizationMeta {
    pub quant_method: String,
    pub version: u32,
    pub formats: Vec<LtqFormat>,
    #[serde(default)]
    pub row_tables: BTreeMap<String, LtqRowTableMeta>,
    #[serde(default)]
    pub modules_not_quantized: Vec<String>,
    pub bits_per_weight: f64,
    #[serde(default)]
    pub calibration: Value,
}

/// Entry in the `"tensors"` map inside the `.ltq` JSON header.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LtqTensorEntry {
    pub dtype: DType,
    pub shape: Shape,
    /// Byte offset relative to the start of the tensor data area (`16 + H`).
    pub offset: u64,
}

impl LtqTensorEntry {
    /// Size of the tensor in bytes; an error if `shape` overflows.
    pub fn byte_len(&self) -> Result<u64> {
        self.shape
            .checked_byte_len(self.dtype)
            .map(|n| n as u64)
            .ok_or_else(|| LumenError::Format(format!("tensor shape {} overflows", self.shape)))
    }
}

/// Complete JSON header of an `.ltq` v1 container.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LtqHeader {
    pub config: Value,
    #[serde(default)]
    pub generation_config: Value,
    #[serde(default)]
    pub tokenizer: Value,
    #[serde(default)]
    pub tokenizer_config: Value,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub processor_config: Value,
    #[serde(default)]
    pub video_processor_config: Value,
    pub quantization: LtqQuantizationMeta,
    pub tensors: BTreeMap<String, LtqTensorEntry>,
}

impl LtqHeader {
    /// Validates the quantization metadata, format tables and row-table parameters.
    ///
    /// Tensor *placement* needs the payload size and is checked by [`Self::validate_tensor_layout`].
    pub fn validate(&self) -> Result<()> {
        let q = &self.quantization;
        if q.quant_method != "ltq" {
            return Err(LumenError::Format(format!(
                "unsupported quant_method '{}' (expected 'ltq')",
                q.quant_method
            )));
        }
        if q.version != LTQ_QUANT_VERSION {
            return Err(LumenError::Format(format!(
                "unsupported LTQ quantization version {} (expected {LTQ_QUANT_VERSION})",
                q.version
            )));
        }
        if !q.bits_per_weight.is_finite() || q.bits_per_weight < 0.0 {
            return Err(LumenError::Format(format!(
                "invalid bits_per_weight {}",
                q.bits_per_weight
            )));
        }
        for fmt in &q.formats {
            fmt.validate()?;
        }
        for (name, meta) in &q.row_tables {
            meta.validate(name, &q.formats)?;
        }
        Ok(())
    }

    /// Checks that every tensor lies inside a payload of `payload_len` bytes, is aligned
    /// (64 bytes; 4096 for row-table data) and that no two non-empty tensors overlap.
    ///
    /// Offsets are relative to the payload start, which is itself 4096-byte aligned, so relative
    /// alignment is absolute alignment.
    pub fn validate_tensor_layout(&self, payload_len: u64) -> Result<()> {
        let row_table_data: Vec<String> = self
            .quantization
            .row_tables
            .keys()
            .map(|p| format!("{p}.ltq_rows.data"))
            .collect();
        let mut ranges = Vec::with_capacity(self.tensors.len());
        for (name, entry) in &self.tensors {
            let align = if row_table_data.contains(name) {
                LTQ_ROW_TABLE_ALIGN
            } else {
                LTQ_TENSOR_ALIGN
            };
            if entry.offset % align != 0 {
                return Err(LumenError::Format(format!(
                    "tensor '{name}' offset {} is not {align}-byte aligned",
                    entry.offset
                )));
            }
            let len = entry
                .byte_len()
                .map_err(|e| LumenError::Format(format!("tensor '{name}': {e}")))?;
            let end = entry
                .offset
                .checked_add(len)
                .ok_or_else(|| LumenError::Format(format!("tensor '{name}' offset overflow")))?;
            if end > payload_len {
                return Err(LumenError::Format(format!(
                    "tensor '{name}' range [{}..{end}) exceeds payload size {payload_len}",
                    entry.offset
                )));
            }
            if len > 0 {
                ranges.push((entry.offset, end, name.as_str()));
            }
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            let (_, prev_end, prev_name) = pair[0];
            let (start, _, name) = pair[1];
            if start < prev_end {
                return Err(LumenError::Format(format!(
                    "tensors '{prev_name}' and '{name}' overlap"
                )));
            }
        }
        Ok(())
    }

    /// Serializes this header into a `.ltq` prefix: the 16-byte binary preamble followed by the
    /// JSON padded with spaces so that `16 + H` is a multiple of [`LTQ_HEADER_ALIGN`]; the tensor
    /// payload may start right after.
    pub fn to_padded_header_bytes(&self) -> Result<Vec<u8>> {
        let mut json_bytes = serde_json::to_vec(self)
            .map_err(|e| LumenError::Format(format!("failed to serialize LTQ header: {e}")))?;
        let total =
            (LTQ_PREAMBLE_LEN + json_bytes.len()).div_ceil(LTQ_HEADER_ALIGN) * LTQ_HEADER_ALIGN;
        json_bytes.resize(total - LTQ_PREAMBLE_LEN, b' ');

        let mut out = Vec::with_capacity(LTQ_PREAMBLE_LEN + json_bytes.len());
        out.extend_from_slice(&LTQ_MAGIC);
        out.extend_from_slice(&LTQ_CONTAINER_VERSION.to_le_bytes());
        out.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&json_bytes);
        debug_assert_eq!(out.len() % LTQ_HEADER_ALIGN, 0);
        Ok(out)
    }
}

/// Parses the `.ltq` binary preamble and JSON header from the beginning of a byte slice,
/// returning `(LtqHeader, data_section_byte_offset)`. Fully validates the header, including the
/// placement of all tensors inside `bytes`.
pub fn parse_ltq_header_from_slice(bytes: &[u8]) -> Result<(LtqHeader, usize)> {
    if bytes.len() < LTQ_PREAMBLE_LEN {
        return Err(LumenError::Format(
            "file too small to contain a 16-byte .ltq preamble".to_string(),
        ));
    }
    if bytes[0..4] != LTQ_MAGIC {
        return Err(LumenError::Format(
            "invalid .ltq magic signature (expected 'LTQ\\0')".to_string(),
        ));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("4-byte slice"));
    if version != LTQ_CONTAINER_VERSION {
        return Err(LumenError::Format(format!(
            "unsupported .ltq container version {version} (expected {LTQ_CONTAINER_VERSION})"
        )));
    }
    let header_len = u64::from_le_bytes(bytes[8..16].try_into().expect("8-byte slice"));
    let header_len = usize::try_from(header_len)
        .map_err(|_| LumenError::Format(format!("header length {header_len} too large")))?;
    let data_offset = LTQ_PREAMBLE_LEN.checked_add(header_len).ok_or_else(|| {
        LumenError::Format("overflow computing .ltq header end offset".to_string())
    })?;
    if header_len == 0 || data_offset % LTQ_HEADER_ALIGN != 0 {
        return Err(LumenError::Format(format!(
            "16 + JSON header length ({data_offset}) must be a non-zero multiple of {LTQ_HEADER_ALIGN}"
        )));
    }
    if bytes.len() < data_offset {
        return Err(LumenError::Format(format!(
            ".ltq file truncated: header specifies {header_len} bytes, file has {}",
            bytes.len() - LTQ_PREAMBLE_LEN
        )));
    }

    let header: LtqHeader = serde_json::from_slice(&bytes[LTQ_PREAMBLE_LEN..data_offset])
        .map_err(|e| LumenError::Format(format!("invalid JSON in .ltq header: {e}")))?;
    header.validate()?;
    header.validate_tensor_layout((bytes.len() - data_offset) as u64)?;

    Ok((header, data_offset))
}

/// Memory-mapped `.ltq` model container on disk.
pub struct LtqContainer {
    pub header: LtqHeader,
    pub data_offset: usize,
    mmap: Arc<Mmap>,
}

impl LtqContainer {
    /// Opens and memory-maps an `.ltq` file, validating its header and tensor table bounds.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let file = File::open(p).map_err(|e| {
            LumenError::Format(format!("cannot open model file '{}': {e}", p.display()))
        })?;
        // SAFETY: Read-only mapping of an immutable model checkpoint file.
        let mmap = unsafe { Mmap::map(&file)? };
        let (header, data_offset) = parse_ltq_header_from_slice(&mmap)?;
        Ok(Self {
            header,
            data_offset,
            mmap: Arc::new(mmap),
        })
    }

    /// Returns the raw bytes for a named tensor entry.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let entry =
            self.header.tensors.get(name).ok_or_else(|| {
                LumenError::Format(format!("missing tensor '{name}' in .ltq file"))
            })?;
        let out_of_range = || LumenError::Format(format!("tensor '{name}' is out of file bounds"));
        let start = usize::try_from(entry.offset)
            .ok()
            .and_then(|o| self.data_offset.checked_add(o))
            .ok_or_else(out_of_range)?;
        let len = usize::try_from(entry.byte_len()?).map_err(|_| out_of_range())?;
        let end = start.checked_add(len).ok_or_else(out_of_range)?;
        self.mmap.get(start..end).ok_or_else(out_of_range)
    }

    #[inline]
    pub fn mmap_arc(&self) -> Arc<Mmap> {
        Arc::clone(&self.mmap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_ltq::Frac;

    fn test_format() -> LtqFormat {
        LtqFormat {
            name: "TCQ3X".to_string(),
            k: 3,
            frac: Frac::None,
            masks: vec![359, 241],
            h: 2,
            sub: 16,
            lut: vec![0.0; 32],
            gain: vec![1.0; 4],
        }
    }

    fn test_header() -> LtqHeader {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.norm.weight".to_string(),
            LtqTensorEntry {
                dtype: DType::BF16,
                shape: Shape::from([128]),
                offset: 0,
            },
        );
        LtqHeader {
            config: serde_json::json!({ "model_type": "test_model", "num_hidden_layers": 28 }),
            generation_config: Value::Null,
            tokenizer: Value::Null,
            tokenizer_config: Value::Null,
            chat_template: Some("{{ messages }}".to_string()),
            processor_config: Value::Null,
            video_processor_config: Value::Null,
            quantization: LtqQuantizationMeta {
                quant_method: "ltq".to_string(),
                version: 1,
                formats: vec![test_format()],
                row_tables: BTreeMap::new(),
                modules_not_quantized: vec!["lm_head".to_string()],
                bits_per_weight: 3.16,
                calibration: Value::Null,
            },
            tensors,
        }
    }

    /// Header bytes followed by `payload_len` zero bytes.
    fn file_with_payload(header: &LtqHeader, payload_len: usize) -> Vec<u8> {
        let mut bytes = header.to_padded_header_bytes().unwrap();
        bytes.resize(bytes.len() + payload_len, 0);
        bytes
    }

    #[test]
    fn test_ltq_header_roundtrip_and_padding() {
        let header = test_header();
        let bytes = file_with_payload(&header, 256);

        // 16 + H is a multiple of 4096: the payload starts on a page boundary.
        let h_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        assert_eq!((16 + h_len) % 4096, 0);
        let (parsed, data_offset) = parse_ltq_header_from_slice(&bytes).unwrap();
        assert_eq!(data_offset, 16 + h_len);
        assert_eq!(data_offset, 4096);
        assert_eq!(parsed.quantization.formats[0].name, "TCQ3X");
        assert_eq!(parsed.tensors["model.norm.weight"].byte_len().unwrap(), 256);
    }

    #[test]
    fn test_large_header_pads_to_next_page() {
        let mut header = test_header();
        header.chat_template = Some("x".repeat(10_000));
        let bytes = file_with_payload(&header, 256);
        let h_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        assert_eq!(16 + h_len, 12_288);
        parse_ltq_header_from_slice(&bytes).unwrap();
    }

    /// The layout produced by the reference writer (`experements/src/ltq_file.py`): `H` is a
    /// "reserve" with `(16 + H) % 4096 == 0` and the JSON is space-padded to exactly `H` bytes.
    #[test]
    fn test_reads_the_layout_of_the_reference_writer() {
        let json = serde_json::to_vec(&test_header()).unwrap();
        let reserve = 4096 * 3 - 16;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"LTQ\0");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(reserve as u64).to_le_bytes());
        bytes.extend_from_slice(&json);
        bytes.resize(16 + reserve, b' ');
        bytes.resize(16 + reserve + 256, 0);
        let (_, data_offset) = parse_ltq_header_from_slice(&bytes).unwrap();
        assert_eq!(data_offset, 4096 * 3);
    }

    #[test]
    fn test_rejects_payload_not_starting_on_a_page_boundary() {
        let mut bytes = file_with_payload(&test_header(), 256);
        // H = 4096 would put the payload at 4112.
        bytes[8..16].copy_from_slice(&4096u64.to_le_bytes());
        assert!(parse_ltq_header_from_slice(&bytes).is_err());
        bytes[8..16].copy_from_slice(&0u64.to_le_bytes());
        assert!(parse_ltq_header_from_slice(&bytes).is_err());
    }

    #[test]
    fn test_unknown_header_sections_are_tolerated_and_known_ones_kept() {
        // The reference writer also stores `video_processor_config`.
        let mut v = serde_json::to_value(test_header()).unwrap();
        v["video_processor_config"] = serde_json::json!({ "fps": 2 });
        v["some_future_section"] = serde_json::json!(1);
        let h: LtqHeader = serde_json::from_value(v).unwrap();
        assert_eq!(h.video_processor_config["fps"], 2);
    }

    #[test]
    fn test_tensor_alignment_is_enforced() {
        let mut header = test_header();
        header.tensors.get_mut("model.norm.weight").unwrap().offset = 32;
        let err = header
            .validate_tensor_layout(1 << 20)
            .unwrap_err()
            .to_string();
        assert!(err.contains("64-byte aligned"), "{err}");
        header.tensors.get_mut("model.norm.weight").unwrap().offset = 64;
        header.validate_tensor_layout(1 << 20).unwrap();

        // Row-table data must be page aligned.
        header.quantization.row_tables.insert(
            "ple".into(),
            LtqRowTableMeta {
                format: 0,
                ring: 160,
                hadamard: 32,
            },
        );
        let rows = |offset| LtqTensorEntry {
            dtype: DType::U8,
            shape: Shape::from([100]),
            offset,
        };
        header
            .tensors
            .insert("ple.ltq_rows.data".into(), rows(4096 + 64));
        let err = header
            .validate_tensor_layout(1 << 20)
            .unwrap_err()
            .to_string();
        assert!(err.contains("4096-byte aligned"), "{err}");
        header
            .tensors
            .insert("ple.ltq_rows.data".into(), rows(8192));
        header.validate_tensor_layout(1 << 20).unwrap();
    }

    #[test]
    fn test_rejects_bad_preamble_and_truncation() {
        let good = file_with_payload(&test_header(), 256);
        assert!(parse_ltq_header_from_slice(&good[..10]).is_err());

        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert!(parse_ltq_header_from_slice(&bad_magic).is_err());

        let mut bad_version = good.clone();
        bad_version[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert!(parse_ltq_header_from_slice(&bad_version).is_err());

        assert!(parse_ltq_header_from_slice(&good[..4096]).is_err());

        let mut huge = good.clone();
        huge[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(parse_ltq_header_from_slice(&huge).is_err());
    }

    #[test]
    fn test_rejects_tensor_outside_payload() {
        // Needs 256 payload bytes; give 255.
        let bytes = file_with_payload(&test_header(), 255);
        let err = parse_ltq_header_from_slice(&bytes).unwrap_err().to_string();
        assert!(err.contains("exceeds payload"), "{err}");
    }

    #[test]
    fn test_rejects_overflowing_shape_and_offset() {
        let mut header = test_header();
        header.tensors.get_mut("model.norm.weight").unwrap().shape = Shape::from([usize::MAX, 4]);
        assert!(header.validate_tensor_layout(1 << 40).is_err());

        let mut header = test_header();
        header.tensors.get_mut("model.norm.weight").unwrap().offset = u64::MAX - 10;
        assert!(header.validate_tensor_layout(1 << 40).is_err());
    }

    #[test]
    fn test_rejects_overlapping_tensors_but_allows_adjacent() {
        let entry = |offset| LtqTensorEntry {
            dtype: DType::U8,
            shape: Shape::from([100]),
            offset,
        };
        let mut header = test_header();
        header.tensors.insert("a".into(), entry(0));
        header.tensors.insert("b".into(), entry(128));
        header.tensors.remove("model.norm.weight");
        header.validate_tensor_layout(256).unwrap();

        // 100 bytes at offset 0 and at 64 overlap.
        header.tensors.insert("b".into(), entry(64));
        let err = header.validate_tensor_layout(256).unwrap_err().to_string();
        assert!(err.contains("overlap"), "{err}");

        // Empty tensors never overlap anything.
        header.tensors.insert("b".into(), entry(128));
        header.tensors.insert(
            "empty".into(),
            LtqTensorEntry {
                dtype: DType::U8,
                shape: Shape::from([0]),
                offset: 64,
            },
        );
        header.validate_tensor_layout(256).unwrap();
    }

    #[test]
    fn test_header_validation_of_quantization_section() {
        let mut h = test_header();
        h.quantization.quant_method = "gptq".into();
        assert!(h.validate().is_err());

        let mut h = test_header();
        h.quantization.version = 2;
        assert!(h.validate().is_err());

        let mut h = test_header();
        h.quantization.formats[0].h = 70;
        assert!(
            h.validate().is_err(),
            "invalid format must be rejected, not panic"
        );

        let mut h = test_header();
        h.quantization.bits_per_weight = f64::NAN;
        assert!(h.validate().is_err());
    }

    #[test]
    fn test_row_table_validation() {
        let formats = vec![test_format()]; // sub = 16
        let ok = |ring, hadamard| LtqRowTableMeta {
            format: 0,
            ring,
            hadamard,
        };
        ok(160, 32).validate("t", &formats).unwrap();
        ok(256, 256).validate("t", &formats).unwrap();
        ok(512, 256).validate("t", &formats).unwrap();
        ok(48, 16).validate("t", &formats).unwrap();

        assert!(
            ok(160, 64).validate("t", &formats).is_err(),
            "wrong hadamard block"
        );
        assert!(
            ok(160, 16).validate("t", &formats).is_err(),
            "not the largest block"
        );
        assert!(
            ok(100, 4).validate("t", &formats).is_err(),
            "ring not multiple of 16"
        );
        assert!(ok(0, 1).validate("t", &formats).is_err());
        assert!(
            LtqRowTableMeta {
                format: 1,
                ring: 160,
                hadamard: 32
            }
            .validate("t", &formats)
            .is_err()
        );

        let mut wide = test_format();
        wide.sub = 32;
        assert!(
            ok(160, 32).validate("t", &[wide]).is_err(),
            "rings need sub=16"
        );
    }

    #[test]
    fn test_container_open_and_tensor_bytes() {
        let header = test_header();
        let mut bytes = file_with_payload(&header, 256);
        let data_offset = bytes.len() - 256;
        bytes[data_offset..].fill(0xAB);

        let path =
            std::env::temp_dir().join(format!("lumen-format-test-{}.ltq", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let container = LtqContainer::open(&path).unwrap();
        let t = container.tensor_bytes("model.norm.weight").unwrap();
        assert_eq!(t.len(), 256);
        assert!(t.iter().all(|&b| b == 0xAB));
        assert!(container.tensor_bytes("missing").is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
