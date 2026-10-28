//! `.ltq` v1 container reader and header parser (`docs/RU/ltq-format.md` Section 6).
//!
//! Binary layout (little-endian):
//! - `0..4`:   Magic signature `"LTQ\0"`
//! - `4..8`:   `u32` container version (`1`)
//! - `8..16`:  `u64` JSON header byte length `H`
//! - `16..16+H`: UTF-8 JSON header padded with spaces (`0x20`) so `16 + H` is a multiple of 4096
//! - `16+H..`: Tensor payload area (64-byte aligned; 4096-byte aligned for row tables)

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use lumen_core::{ComponentTree, DType, LumenError, Result, Shape};
use lumen_ltq::LtqFormat;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Magic 4-byte signature at offset 0 of every `.ltq` file.
pub const LTQ_MAGIC: [u8; 4] = *b"LTQ\0";

/// Supported `.ltq` container format version.
pub const LTQ_CONTAINER_VERSION: u32 = 1;

/// Alignment of the start of the tensor data section (`16 + H` is a multiple of 4096).
pub const LTQ_DATA_SECTION_ALIGN: usize = 4096;

/// Metadata for a single `RowTable` compressed with `LtqRing` in `quantization.row_tables`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LtqRowTableMeta {
    pub format: u32,
    pub ring: u32,
    pub hadamard: u32,
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
    #[inline]
    pub fn byte_len(&self) -> u64 {
        self.shape.byte_len(self.dtype) as u64
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
    pub quantization: LtqQuantizationMeta,
    pub tensors: BTreeMap<String, LtqTensorEntry>,
}

impl LtqHeader {
    /// Validates the quantization metadata and format tables.
    pub fn validate(&self) -> Result<()> {
        if self.quantization.quant_method != "ltq" {
            return Err(LumenError::Format(format!(
                "unsupported quant_method '{}' (expected 'ltq')",
                self.quantization.quant_method
            )));
        }
        if self.quantization.version != LTQ_CONTAINER_VERSION {
            return Err(LumenError::Format(format!(
                "unsupported LTQ quantization version {} (expected {})",
                self.quantization.version, LTQ_CONTAINER_VERSION
            )));
        }
        for fmt in &self.quantization.formats {
            fmt.validate()?;
        }
        Ok(())
    }

    /// Builds the `ComponentTree` for this `.ltq` model from its embedded `config` metadata.
    pub fn component_tree(&self) -> Result<ComponentTree> {
        let cfg = self
            .config
            .get("text_config")
            .unwrap_or(&self.config);
        let model_type = cfg
            .get("model_type")
            .or_else(|| self.config.get("model_type"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                LumenError::Format("missing 'model_type' in .ltq header config".to_string())
            })?;
        let num_layers = cfg
            .get("num_hidden_layers")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                LumenError::Format("missing 'num_hidden_layers' in .ltq header config".to_string())
            })? as usize;

        let preset = match model_type {
            "qwen4_exp" => "qwen4_exp",
            "qwen3_next" => "qwen3_next",
            "qwen3_moe" | "mixtral" | "deepseek_v3" | "gpt_oss" => "moe",
            "gemma4" | "gemma4_text" => "gemma4",
            _ => "dense",
        };
        Ok(ComponentTree::preset(preset, num_layers))
    }

    /// Serializes this header into a `.ltq` prefix (16-byte binary preamble + space-padded JSON
    /// aligned to `LTQ_DATA_SECTION_ALIGN` = 4096 bytes).
    pub fn to_padded_header_bytes(&self) -> Result<Vec<u8>> {
        let mut json_bytes = serde_json::to_vec(self)
            .map_err(|e| LumenError::Format(format!("failed to serialize LTQ header: {e}")))?;
        let total_unpadded = 16 + json_bytes.len();
        let total_aligned = total_unpadded.div_ceil(LTQ_DATA_SECTION_ALIGN) * LTQ_DATA_SECTION_ALIGN;
        let pad_len = total_aligned - total_unpadded;
        json_bytes.resize(json_bytes.len() + pad_len, b' ');

        let h_len = json_bytes.len() as u64;
        let mut out = Vec::with_capacity(total_aligned);
        out.extend_from_slice(&LTQ_MAGIC);
        out.extend_from_slice(&LTQ_CONTAINER_VERSION.to_le_bytes());
        out.extend_from_slice(&h_len.to_le_bytes());
        out.extend_from_slice(&json_bytes);
        debug_assert_eq!(out.len() % LTQ_DATA_SECTION_ALIGN, 0);
        Ok(out)
    }
}

/// Parses the `.ltq` binary preamble and JSON header from the beginning of a byte slice,
/// returning `(LtqHeader, data_section_byte_offset)`.
pub fn parse_ltq_header_from_slice(bytes: &[u8]) -> Result<(LtqHeader, usize)> {
    if bytes.len() < 16 {
        return Err(LumenError::Format(
            "file too small to contain a 16-byte .ltq preamble".to_string(),
        ));
    }
    if bytes[0..4] != LTQ_MAGIC {
        return Err(LumenError::Format(
            "invalid .ltq magic signature (expected 'LTQ\\0')".to_string(),
        ));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != LTQ_CONTAINER_VERSION {
        return Err(LumenError::Format(format!(
            "unsupported .ltq container version {version} (expected {LTQ_CONTAINER_VERSION})"
        )));
    }
    let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let data_offset = 16usize.checked_add(header_len).ok_or_else(|| {
        LumenError::Format("overflow computing .ltq header end offset".to_string())
    })?;
    if bytes.len() < data_offset {
        return Err(LumenError::Format(format!(
            ".ltq file truncated: header specifies {header_len} bytes, file has {}",
            bytes.len()
        )));
    }

    let header: LtqHeader = serde_json::from_slice(&bytes[16..data_offset])
        .map_err(|e| LumenError::Format(format!("invalid JSON in .ltq header: {e}")))?;
    header.validate()?;

    // Verify all tensor ranges fall within `bytes[data_offset..]`
    let payload_len = (bytes.len() - data_offset) as u64;
    for (name, entry) in &header.tensors {
        let end = entry
            .offset
            .checked_add(entry.byte_len())
            .ok_or_else(|| LumenError::Format(format!("tensor '{name}' offset overflow")))?;
        if end > payload_len {
            return Err(LumenError::Format(format!(
                "tensor '{name}' range [{}..{end}) exceeds payload size {payload_len}",
                entry.offset
            )));
        }
    }

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
        let entry = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| LumenError::Format(format!("missing tensor '{name}' in .ltq file")))?;
        let start = self.data_offset + (entry.offset as usize);
        let end = start + (entry.byte_len() as usize);
        Ok(&self.mmap[start..end])
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

    #[test]
    fn test_ltq_header_roundtrip_and_alignment() {
        let fmt = LtqFormat {
            name: "TCQ3X".to_string(),
            k: 3,
            frac: Frac::None,
            masks: vec![359, 241],
            h: 2,
            sub: 16,
            lut: vec![0.0; 32],
            gain: vec![1.0; 4],
        };
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.norm.weight".to_string(),
            LtqTensorEntry {
                dtype: DType::BF16,
                shape: Shape::from([128]),
                offset: 0,
            },
        );

        let header = LtqHeader {
            config: serde_json::json!({ "model_type": "qwen3", "num_hidden_layers": 28 }),
            generation_config: Value::Null,
            tokenizer: Value::Null,
            tokenizer_config: Value::Null,
            chat_template: Some("{{ messages }}".to_string()),
            processor_config: Value::Null,
            quantization: LtqQuantizationMeta {
                quant_method: "ltq".to_string(),
                version: 1,
                formats: vec![fmt],
                row_tables: BTreeMap::new(),
                modules_not_quantized: vec!["lm_head".to_string()],
                bits_per_weight: 3.16,
                calibration: Value::Null,
            },
            tensors,
        };

        let mut file_bytes = header.to_padded_header_bytes().unwrap();
        assert_eq!(file_bytes.len() % 4096, 0);
        // Append 256 bytes for `model.norm.weight` (128 * 2 bytes)
        file_bytes.extend_from_slice(&vec![0u8; 256]);

        let (parsed, data_offset) = parse_ltq_header_from_slice(&file_bytes).unwrap();
        assert_eq!(data_offset, 4096);
        assert_eq!(parsed.quantization.formats[0].name, "TCQ3X");
        assert_eq!(parsed.tensors["model.norm.weight"].byte_len(), 256);
    }
}
